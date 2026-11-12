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


def _threads() -> int:
    """This process's threads, once the ones that have ended are gone: a
    joined thread is still listed while the kernel takes it down, so on a
    busy machine one read can be one high (seen: 3 then 4). The least of a
    few reads."""
    import time

    n = len(os.listdir("/proc/self/task"))
    for _ in range(10):
        time.sleep(0.05)
        n = min(n, len(os.listdir("/proc/self/task")))
    return n


def test_threads_do_not_pile_up():
    """Factorizations from short-lived threads (every pool here is made per
    call) must not each leave a team of MKL's workers behind: the wrist
    yoke's search ended on 229 threads holding 5.9 GB, and the constrained
    method was skipped for memory (nonlinear.on_mkl_thread)."""
    import threading

    if direct.backend() != "pardiso" or not os.path.isdir("/proc/self/task"):
        return
    mesh, fixed, fs, e = _bar()
    k = Stiffness(mesh, 0.3, fixed).assemble(e)
    out = []

    def work():
        out.append(DirectSolver(mesh.node_ijk).solve(k, fs[0]))

    def run():
        t = threading.Thread(target=work)
        t.start()
        t.join()

    with _env(LOADPATH_THREADS="3"):
        run()
        n0 = _threads()
        for _ in range(6):
            run()
        n1 = _threads()
    assert n1 <= n0, (n0, n1)
    assert all(np.abs(u - out[0]).max() == 0.0 for u in out)


_KERNELS_AFTER_MKL = """
import os, sys, threading, time
sys.path.insert(0, sys.argv[1])
import numpy as np, scipy.sparse as sp
from numba import njit, prange
from loadpath.direct import DirectSolver

@njit(parallel=True)
def kern(a):
    s = 0.0
    for i in prange(a.shape[0]):
        s += a[i] * a[i]
    return s

n = 4000
k = (sp.identity(n) * 4.0 - sp.eye(n, k=1) - sp.eye(n, k=-1)).tocsr()
DirectSolver().solve(k, np.ones(n))          # MKL first, as the optimizer's first solve is
a = np.ones(200000)
def run():
    t = threading.Thread(target=kern, args=(a,)); t.start(); t.join()
def threads():
    # the least of a few reads: a joined thread is listed until the kernel has taken it down
    n = len(os.listdir("/proc/self/task"))
    for _ in range(10):
        time.sleep(0.05)
        n = min(n, len(os.listdir("/proc/self/task")))
    return n
run()
n0 = threads()
for _ in range(6):
    run()
print(n0, threads())
"""


def test_kernels_after_a_factorization_do_not_pile_up():
    """numba's kernels bind to the OpenMP runtime that is loaded when they
    start. After MKL's, each kernel called from a pool thread left a team
    behind (the wrist yoke: 27 threads a constrained part), so MKL's first
    call starts the kernels' threads first (nonlinear.mkl_threads). In a
    fresh interpreter: a factorization, then kernels from short-lived
    threads."""
    import subprocess

    if direct.backend() != "pardiso" or not os.path.isdir("/proc/self/task"):
        return
    env = dict(os.environ, LOADPATH_THREADS="3")
    run = subprocess.run([sys.executable, "-c", _KERNELS_AFTER_MKL, str(Path(__file__).resolve().parents[1])],
                         capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    n0, n1 = (int(v) for v in run.stdout.split()[-2:])
    assert n1 <= n0, (n0, n1)


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
    it fits. Beyond that the multigrid still, while most of the grid is
    live, and AMG-CG on the live cells once little of it is. A design
    with a load on cells that are not live is the whole grid's system.
    LOADPATH_SOLVER overrides."""
    from loadpath import simp
    from loadpath.simp import _Analysis

    st, dom, _ = _mbb()
    fe = _Analysis(dom, st.elastic(design=True), 1e-3)
    full = np.ones(dom.n_elem, bool)
    thin = fe.anchor.copy()            # (the cells at loads and supports are always live)
    thin[: dom.n_elem // 10] = True
    assert thin.mean() < 0.5
    # the same without the cells under the load: a system that cannot be cut
    # down to the live cells, so the whole grid's
    bare = thin & ~dom.load_nodes[dom.mesh.enode].any(axis=1)
    assert fe._live_nodes(thin) is not None and fe._live_nodes(bare) is None and fe._live_nodes(full) is None
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
            # too big to factor: the multigrid, slow as it has been, while
            # most of the grid is live; AMG-CG on what is live once little is
            assert fe._route(full) == "mg"
            assert fe._route(thin) == "amg"
            assert fe._route(bare) == "mg"             # (a free-flying part: loads on every node)
            fe._mg_ok = True
            assert fe._route(full) == "mg" and fe._route(thin) == "mg"
            fe._direct_ok = False          # (a factor that would not fit in memory)
        with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS="1000000,2000000"):
            assert fe._route(full) == "mg"
            fe._mg_ok = False
            assert fe._route(full) == "mg" and fe._route(thin) == "amg"
    finally:
        simp.MG_AUTO_CELLS = old


def test_simp_direct_matches_assembled():
    """The optimizer on a factorization lands on AMG-CG's design."""
    from loadpath.simp import SimpOptions, optimize

    st, dom, radius = _mbb()
    opts = SimpOptions(volume_fraction=0.3, filter_radius=radius, max_iterations=25)
    out = {}
    for mode in ("assembled", "direct"):
        # (the sizes: SuperLU, where there is no PARDISO, factors this grid too)
        with _env(LOADPATH_SOLVER=mode, LOADPATH_DIRECT_DOFS=f"10,{10 * dom.mesh.n_dof}"):
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
    that never used it. The slow answer itself stands (it converged inside
    the multigrid's own stall limit): that design is not solved twice."""
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
    assert seen[0] == "mg" and len(seen) > 1 and set(seen[1:]) == {"direct"}, seen
    with _env(LOADPATH_SOLVER="direct", LOADPATH_DIRECT_DOFS=f"10,{10 * dom.mesh.n_dof}"):
        ref = optimize(dom, st.elastic(design=True), opts)
    # (one design read to the multigrid's tolerance, the rest exactly)
    assert abs(got.compliance - ref.compliance) < 2e-3 * ref.compliance, (got.compliance, ref.compliance)


def test_stalled_multigrid_answer_is_not_kept():
    """Past the multigrid's own stall limit its answer is in doubt: that
    design is solved again by the solver the run goes on with."""
    from loadpath import simp
    from loadpath.mg import GridElasticity
    from loadpath.simp import SimpOptions, optimize

    st, dom, radius = _mbb()
    opts = SimpOptions(volume_fraction=0.3, filter_radius=radius, max_iterations=3)
    old = simp.MG_AUTO_CELLS, simp.MG_SLOW_ITERS, GridElasticity.STALL
    seen = []
    solve = simp._Analysis.solve

    def spy(self, *a, **k):
        # (as if every multigrid solve had run past its stall limit)
        GridElasticity.STALL = 0
        try:
            out = solve(self, *a, **k)
        finally:
            GridElasticity.STALL = old[2]
        seen.append(self.last_solver)
        return out

    simp.MG_AUTO_CELLS, simp.MG_SLOW_ITERS = 1000, 1
    simp._Analysis.solve = spy
    try:
        with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS=f"10,{10 * dom.mesh.n_dof}"):
            optimize(dom, st.elastic(design=True), opts)
    finally:
        simp._Analysis.solve = solve
        simp.MG_AUTO_CELLS, simp.MG_SLOW_ITERS, GridElasticity.STALL = old
    assert seen and set(seen) == {"direct"}, seen


def test_multigrid_that_fails_on_the_whole_grid_costs_one_design():
    """A multigrid that does not converge on the whole grid (its coarse
    levels average a sharp design's moduli and can lose a thin load path: a
    drone frame's run died at beta 8 on 2000 CG iterations) ends that
    design's solve, not the run. Where the system is too big to factor,
    AMG-CG takes the design, its hierarchy built from the matrix itself,
    and the multigrid is tried again on the next while most of the grid is
    live (simp.MG_SLOW_LIVE)."""
    from loadpath import simp
    from loadpath.mg import GridElasticity
    from loadpath.simp import SimpOptions, optimize

    st, dom, radius = _mbb()
    opts = SimpOptions(volume_fraction=0.3, filter_radius=radius, max_iterations=4)
    old = simp.MG_AUTO_CELLS
    seen = []
    solve, solve_many = simp._Analysis.solve, GridElasticity.solve_many

    def spy(self, *a, **k):
        out = solve(self, *a, **k)
        seen.append(self.last_solver)
        return out

    def flaky(self, fs, x0s=None):
        if len(seen) == 1:             # the second design
            raise RuntimeError("linear solve did not converge (2000 iterations)")
        return solve_many(self, fs, x0s)

    simp.MG_AUTO_CELLS = 1000
    simp._Analysis.solve = spy
    out = {}
    try:
        with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS="10,20"):
            out["ref"] = optimize(dom, st.elastic(design=True), opts)
            assert seen and set(seen) == {"mg"}, seen
            del seen[:]
            GridElasticity.solve_many = flaky
            out["got"] = optimize(dom, st.elastic(design=True), opts)
    finally:
        simp._Analysis.solve, GridElasticity.solve_many = solve, solve_many
        simp.MG_AUTO_CELLS = old
    assert seen[:3] == ["mg", "amg", "mg"], seen
    a, b = out["ref"], out["got"]
    assert abs(a.compliance - b.compliance) < 2e-3 * a.compliance, (a.compliance, b.compliance)


def test_stalled_cg_is_factored():
    """CG that is not going to finish hands its system to a factorization
    (fem.Solver.STALL): the answer is the factorization's, the factor serves
    that matrix's further right-hand sides, and a solve that converges is
    left alone."""
    mesh, fixed, fs, e = _bar()
    k = Stiffness(mesh, 0.3, fixed).assemble(e)
    ref = DirectSolver(mesh.node_ijk).solve_many(k, fs)
    # a healthy solve: CG's own answer, nothing factored
    s = Solver(mesh, rtol=1e-8)
    us = s.solve_many(k, fs)
    assert s._factored is None and 0 < s.last_iters < s.STALL
    for u, r in zip(us, ref):
        assert np.linalg.norm(u - r) < 1e-5 * np.linalg.norm(r)
    # not there in the iterations it is given
    for many in (True, False):
        s = Solver(mesh, rtol=1e-13, maxiter=4000)
        s.STALL = 3
        us = s.solve_many(k, fs) if many else [s.solve(k, f) for f in fs]
        assert s._factored is not None, many
        for u, r in zip(us, ref):
            assert np.linalg.norm(u - r) < 1e-9 * np.linalg.norm(r), many
        f2 = fs[0] + 2.0 * fs[1]
        it = s.last_iters
        u2 = s.solve(k, f2)                                   # straight from the factor
        assert np.linalg.norm(u2 - (ref[0] + 2.0 * ref[1])) < 1e-9 * np.linalg.norm(u2) and s.last_iters == it
    # nothing to factor with: CG's full allowance, and its own error where it fails
    with _env(LOADPATH_DIRECT="0"):
        s = Solver(mesh, rtol=1e-8, maxiter=4000)
        s.STALL = 3
        u = s.solve(k, fs[0])
        assert s._factored is None and np.linalg.norm(u - ref[0]) < 1e-5 * np.linalg.norm(ref[0])
    # a mechanism is still reported as one
    mesh, fixed, fs, e = _bar(loose=True)
    k = Stiffness(mesh, 0.3, fixed).assemble(e)
    f = fs[2].copy()
    far = np.nonzero(mesh.node_ijk[:, 0] > 24 + 2)[0]
    f[3 * far + 2] = -1.0            # pushes the loose block
    s = Solver(mesh, rtol=1e-8, maxiter=60)
    s.STALL = 20
    try:
        s.solve_many(k, [f, fs[0]])
    except RuntimeError as exc:
        assert "mechanism" in str(exc)
    else:
        raise AssertionError("a part no support holds solved")


def test_factor_that_would_not_fit_is_not_made():
    """PARDISO's analysis says what a factor will hold before any of it is
    allocated. One over its share of the free memory (direct.FACTOR_SHARE)
    is not made: a check's one-off solve hands back to the iterative solver
    (None), and the optimizer finishes on its iterative solvers with the
    design a factorization gives."""
    from loadpath import simp
    from loadpath.nonlinear import _Pardiso
    from loadpath.simp import SimpOptions, optimize

    if direct.backend() != "pardiso":
        return                          # (SuperLU has no analysis to ask)
    mesh, fixed, fs, e = _bar()
    k = Stiffness(mesh, 0.3, fixed).assemble(e)
    fac = _Pardiso(k)
    held = fac.bytes
    fac.close()
    assert held and 1e5 < held < 1e9, held
    try:
        _Pardiso(k, budget=0.5 * held)
    except MemoryError as exc:
        assert "GB" in str(exc)
    else:
        raise AssertionError("a factor twice its budget was made")
    assert _Pardiso(k, budget=2.0 * held).bytes == held
    old = direct.FACTOR_SHARE
    st, dom, radius = _mbb()
    opts = SimpOptions(volume_fraction=0.3, filter_radius=radius, max_iterations=6)
    seen = []
    solve = simp._Analysis.solve

    def spy(self, *a, **kw):
        out = solve(self, *a, **kw)
        seen.append(self.last_solver)
        return out

    try:
        with _env(LOADPATH_SOLVER=None, LOADPATH_DIRECT=None, LOADPATH_DIRECT_DOFS=f"{10 * dom.mesh.n_dof},{10 * dom.mesh.n_dof}"):
            assert direct.solve_many(k, fs, mesh.node_ijk) is not None
            ref = optimize(dom, st.elastic(design=True), opts)
            direct.FACTOR_SHARE = 1e-9
            assert direct.solve_many(k, fs, mesh.node_ijk) is None
            simp._Analysis.solve = spy
            got = optimize(dom, st.elastic(design=True), opts)
    finally:
        simp._Analysis.solve = solve
        direct.FACTOR_SHARE = old
    assert seen and "direct" not in seen, seen
    assert abs(got.compliance - ref.compliance) < 2e-3 * ref.compliance, (got.compliance, ref.compliance)


def _box(a, b):
    return {"type": "box", "min": a, "max": b}


# a free craft (as tests/test_vibration.py's): thrust at four corners, a heavy canopy carried on top
_PADS = [_box([x - 1, y - 1, -1], [x + 7, y + 7, 4]) for x in (0, 54) for y in (0, 54)]
_CANOPY = _box([20, 20, 26], [40, 40, 31])
_CRAFT = {"designSpace": _box([0, 0, 0], [60, 60, 30]),
          "preserve": [{"shape": m} for m in _PADS] + [{"shape": _CANOPY}],
          "loadCases": [{"name": "thrust", "loads": [{"shape": m, "force": [0, 0, 3]} for m in _PADS]}],
          "material": "pa12", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 3.0},
          "inertiaRelief": {"masses": [{"shape": _CANOPY, "mass": 0.14}] + [{"shape": m, "mass": 0.02} for m in _PADS]}}


def test_a_design_repeats_to_the_bit():
    """The same study designed again is the same design, bit for bit, as it
    is on the iterative solvers. Left to its defaults PARDISO's solve of
    this free part on its 3-2-1 hold returned a different vector almost
    every call (ten from one matrix), and ARPACK draws a new start vector
    every call, so a design held to a frequency floor rang at 410, 153 or
    78 Hz from one run to the next (nonlinear.CNR_THREADS, modal.py)."""
    from loadpath.domain import build_domain
    from loadpath.simp import SimpOptions, optimize
    from loadpath.spec import study_from_dict

    from loadpath import simp

    seen = []
    solve = simp._Analysis.solve

    def spy(self, *a, **k):
        out = solve(self, *a, **k)
        seen.append(self.last_solver)
        return out

    simp._Analysis.solve = spy
    try:
        for vib in (None, {"minFrequency": 600.0}):
            runs = []
            for _ in range(3):
                st = study_from_dict({**_CRAFT, **({"vibration": vib} if vib else {})})
                res = optimize(build_domain(st), 0.3, SimpOptions(volume_fraction=0.1, filter_radius=4.5, max_iterations=5))
                runs.append((res.xphys.tobytes(), [repr(h["compliance"]) for h in res.history],
                             [h.get("frequencies") for h in res.history]))
            assert runs[1] == runs[0] and runs[2] == runs[0], (vib, [r[1] for r in runs])
            assert vib is None or any(runs[0][2]), "the frequency floor was not read"
    finally:
        simp._Analysis.solve = solve
    assert direct.backend() is None or set(seen) == {"direct"}, set(seen)


TESTS = [test_matches_iterative_solver, test_one_factor_for_a_matrix, test_threads_do_not_pile_up,
         test_kernels_after_a_factorization_do_not_pile_up, test_mechanism_raises,
         test_sizes_and_switches, test_route, test_simp_direct_matches_assembled, test_slow_multigrid_hands_over,
         test_stalled_multigrid_answer_is_not_kept,
         test_multigrid_that_fails_on_the_whole_grid_costs_one_design, test_stalled_cg_is_factored,
         test_factor_that_would_not_fit_is_not_made,
         test_a_design_repeats_to_the_bit]


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
