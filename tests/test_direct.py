"""
The direct solver (loadpath/direct.py) and the optimizer's choice between it,
the multigrid and AMG-CG (simp._Analysis._route): a factorization must give
the iterative solvers' answer, refuse a mechanism the way they do, and take
the systems it is quickest on.

Run: `python3 tests/test_direct.py` or pytest.
"""
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage

from loadpath import direct
from loadpath.direct import DirectSolver
from loadpath.fem import HexMesh, Solver, Stiffness

CASES = Path(__file__).resolve().parents[1] / "loadpath" / "evals" / "cases"


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _bar(nx=24, ny=8, nz=8, pitch=0.7, seed=0, loose=False):
    """A cantilever with soft cells scattered through it (and, `loose`, a
    block no support holds), clamped at x = 0 with three tip loads."""
    ijk = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    if loose:
        far = ijk + np.array([nx + 3, 0, 0])
        ijk = np.vstack([ijk, far[far[:, 0] < nx + 6]])
    mesh = HexMesh(ijk, pitch, np.zeros(3))
    fixed = np.zeros(mesh.n_dof, bool)
    root = np.nonzero(mesh.node_ijk[:, 0] == 0)[0]
    for i in range(3):
        fixed[3 * root + i] = True
    tip = np.nonzero(mesh.node_ijk[:, 0] == nx)[0]
    fs = []
    for axis in range(3):
        f = np.zeros(mesh.n_dof)
        f[3 * tip + axis] = -1.0 / len(tip)
        fs.append(f)
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter(rng.random((ijk[:, 0].max() + 1, ny, nz)), 1.5)[tuple(ijk.T)]
    x = np.where(g > np.quantile(g, 0.3), 1.0, 1e-3)
    return mesh, fixed, fs, 1e-6 + x ** 3


def test_matches_iterative_solver():
    mesh, fixed, fs, e = _bar()
    k = Stiffness(mesh, 0.3, fixed).assemble(e)
    ref = [Solver(mesh, rtol=1e-11, maxiter=5000).solve(k, f) for f in fs]
    for no_pardiso in (None, "1"):
        with _env(LOADPATH_NO_PARDISO=no_pardiso):
            if no_pardiso is None and direct.backend() != "pardiso":
                continue
            s = DirectSolver(mesh.node_ijk)
            us = s.solve_many(k, fs)
            for u, r in zip(us, ref):
                assert np.abs(u - r).max() < 1e-6 * np.abs(r).max(), (direct.backend(), np.abs(u - r).max())
                assert not u[fixed].any()
            assert s.last_iters == 0


def test_one_factor_for_a_matrix():
    """The adjoints of a design solve against the matrix of its load cases:
    the same matrix object reuses the factor, a new one replaces it."""
    mesh, fixed, fs, e = _bar()
    stiff = Stiffness(mesh, 0.3, fixed)
    k = stiff.assemble(e)
    s = DirectSolver(mesh.node_ijk)
    s.solve_many(k, fs[:1])
    fac = s._fac
    u = s.solve(k, fs[1])
    assert s._fac is fac
    assert np.linalg.norm(k @ u - fs[1]) < 1e-8 * np.linalg.norm(fs[1])
    k2 = stiff.assemble(2.0 * e)
    u2 = s.solve(k2, fs[1])
    assert s._fac is not fac
    assert np.abs(2.0 * u2 - u).max() < 1e-8 * np.abs(u).max()


def test_mechanism_raises():
    """A block nothing holds makes K singular. The factorization may or may
    not notice; the residual check must, with the RuntimeError the iterative
    solvers raise (simp._Analysis falls back to the full grid on it)."""
    mesh, fixed, fs, e = _bar(loose=True)
    k = Stiffness(mesh, 0.3, fixed).assemble(e)
    f = fs[2].copy()
    far = np.nonzero(mesh.node_ijk[:, 0] > 24 + 2)[0]
    f[3 * far + 2] = -1.0            # pushes the loose block
    for no_pardiso in (None, "1"):
        with _env(LOADPATH_NO_PARDISO=no_pardiso):
            try:
                DirectSolver(mesh.node_ijk).solve(k, f)
            except RuntimeError as exc:
                assert "mechanism" in str(exc)
            else:
                raise AssertionError(f"{direct.backend()}: a floating block solved")


def test_sizes_and_switches():
    with _env(LOADPATH_DIRECT="0"):
        assert direct.backend() is None and direct.small_dofs() == 0 and direct.large_dofs() == 0
    with _env(LOADPATH_DIRECT=None, LOADPATH_NO_PARDISO="1", LOADPATH_DIRECT_DOFS=None):
        assert direct.backend() == "superlu"
        assert direct.small_dofs() == direct.large_dofs() < 20_000
    with _env(LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS="5000,70000"):
        assert (direct.small_dofs(), direct.large_dofs()) == (5000, 70000)
    with _env(LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS="8000"):
        assert (direct.small_dofs(), direct.large_dofs()) == (8000, 8000)


def _mbb(cells=6000):
    from loadpath.domain import build_domain
    from loadpath.simp import filter_radius
    from loadpath.spec import study_from_dict

    d = json.loads((CASES / "mbb-beam.json").read_text())
    d["resolution"] = {"maxElements": cells}
    st = study_from_dict(d)
    dom = build_domain(st)
    return st, dom, filter_radius(dom, st.min_member)


def test_route():
    """Small or thin: a factorization. A grid of MG_AUTO_CELLS or more with
    a system too big to factor quickly: the multigrid, until it has taken
    MG_SLOW_ITERS iterations on a design; then a factorization again while
    it fits, AMG-CG beyond. LOADPATH_SOLVER overrides."""
    from loadpath import simp
    from loadpath.simp import _Analysis

    st, dom, _ = _mbb()
    fe = _Analysis(dom, st.elastic(design=True), 1e-3)
    full = np.ones(dom.n_elem, bool)
    thin = np.zeros(dom.n_elem, bool)
    thin[: dom.n_elem // 10] = True
    with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS="1000000,2000000"):
        assert fe._route(full) == "direct" and fe._route(thin) == "direct"
    with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT="0"):
        assert fe._route(full) == "amg"
    for forced, want in (("mg", "mg"), ("assembled", "amg"), ("amg", "amg"), ("direct", "direct")):
        with _env(LOADPATH_SOLVER=forced, LOADPATH_DIRECT=None):
            assert fe._route(full) == want, (forced, fe._route(full))
    old = simp.MG_AUTO_CELLS
    simp.MG_AUTO_CELLS = 1000          # this grid counts as one big enough for the multigrid
    try:
        with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS=f"{dom.mesh.n_dof // 3},{dom.mesh.n_dof}"):
            assert fe._route(full) == "mg"
            assert fe._route(thin) == "direct"         # what the live cells leave fits
            fe._mg_ok = False
            assert fe._route(full) == "direct"         # ... and the whole grid still under `large`
        with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS="10,20"):
            assert fe._route(full) == "amg"
            fe._mg_ok = True
            assert fe._route(full) == "mg"
    finally:
        simp.MG_AUTO_CELLS = old


def test_simp_direct_matches_assembled():
    """The optimizer on a factorization lands on AMG-CG's design."""
    from loadpath.simp import SimpOptions, optimize

    st, dom, radius = _mbb()
    opts = SimpOptions(volume_fraction=0.3, filter_radius=radius, max_iterations=25)
    out = {}
    for mode in ("assembled", "direct"):
        with _env(LOADPATH_SOLVER=mode):
            t = time.time()
            out[mode] = optimize(dom, st.elastic(design=True), opts)
            out[mode + "_s"] = time.time() - t
    a, b = out["assembled"], out["direct"]
    assert abs(a.compliance - b.compliance) < 2e-3 * a.compliance, (a.compliance, b.compliance)
    assert ((a.xphys > 0.5) != (b.xphys > 0.5)).mean() < 0.01
    assert all(r["cgIterations"] == 0 for r in b.history)


def test_slow_multigrid_hands_over():
    """A run whose multigrid takes more than MG_SLOW_ITERS iterations on a
    design finishes on the other solvers, with the same design as a run
    that never used it."""
    from loadpath import simp
    from loadpath.simp import SimpOptions, optimize

    st, dom, radius = _mbb()
    opts = SimpOptions(volume_fraction=0.3, filter_radius=radius, max_iterations=12)
    old = simp.MG_AUTO_CELLS, simp.MG_SLOW_ITERS
    simp.MG_AUTO_CELLS, simp.MG_SLOW_ITERS = 1000, 1      # every multigrid solve counts as slow
    seen = []
    solve = simp._Analysis.solve

    def spy(self, *a, **k):
        out = solve(self, *a, **k)
        seen.append(self.last_solver)
        return out

    simp._Analysis.solve = spy
    try:
        with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS=f"10,{10 * dom.mesh.n_dof}"):
            got = optimize(dom, st.elastic(design=True), opts)
    finally:
        simp._Analysis.solve = spy.__wrapped__ if hasattr(spy, "__wrapped__") else solve
        simp.MG_AUTO_CELLS, simp.MG_SLOW_ITERS = old
    assert seen and "mg" not in seen and set(seen) == {"direct"}, seen
    with _env(LOADPATH_SOLVER="direct"):
        ref = optimize(dom, st.elastic(design=True), opts)
    assert abs(got.compliance - ref.compliance) < 1e-6 * ref.compliance, (got.compliance, ref.compliance)


TESTS = [test_matches_iterative_solver, test_one_factor_for_a_matrix, test_mechanism_raises,
         test_sizes_and_switches, test_route, test_simp_direct_matches_assembled, test_slow_multigrid_hands_over]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
