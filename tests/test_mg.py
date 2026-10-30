"""
Tests for the matrix-free multigrid solver (loadpath/mg.py) and the large-grid
SIMP path it backs: it must give the assembled solver's answer, on grids
small enough to assemble.

Run: `python3 tests/test_mg.py`.
"""
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from loadpath.fem import HexMesh, Layered, Solver, Stiffness
from loadpath.mg import GridElasticity, prolong, restrict

try:
    import torch  # noqa: F401
    DEVICES = ("cpu", "torch")
except ImportError:
    DEVICES = ("cpu",)


def _bar(nx, ny, nz, pitch=0.7, hole=True, seed=0, void=True):
    ijk = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    if hole:
        c = np.array([nx / 2, ny / 2])
        ijk = ijk[np.linalg.norm(ijk[:, :2] + 0.5 - c, axis=1) > ny / 4]
    mesh = HexMesh(ijk, pitch, np.zeros(3))
    fixed = np.zeros(mesh.n_dof, bool)
    root = np.nonzero(mesh.node_ijk[:, 0] == 0)[0]
    for i in range(3):
        fixed[3 * root + i] = True
    f = np.zeros(mesh.n_dof)
    tip = np.nonzero(mesh.node_ijk[:, 0] == nx)[0]
    f[3 * tip + 2] = -1.0 / len(tip)
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter(rng.random((nx, ny, nz)), 1.5)[tuple(ijk.T)]
    x = np.where(g > np.quantile(g, 0.4), 1.0, 1e-3) if void else np.ones(len(ijk))
    return mesh, fixed, f, 1e-6 + x ** 3


def _reference(mesh, nu, fixed, e, f, springs=None):
    k = Stiffness(mesh, nu, fixed).assemble(e)
    if springs is not None:
        k = (k + springs).tocsr()
    return Solver(mesh, rtol=1e-11, maxiter=5000).solve(k, f)


def test_transfers_are_adjoint():
    rng = np.random.default_rng(1)
    c = rng.random((3, 4, 5, 6))
    f = rng.random((3, 7, 9, 11))
    assert abs(np.vdot(prolong(c), f) - np.vdot(c, restrict(f))) < 1e-10
    # trilinear: a linear field prolongs exactly
    x = np.arange(4.0)[None, :, None, None] * np.ones((3, 4, 5, 6))
    assert np.allclose(prolong(x)[0, :, 0, 0], np.arange(7) / 2.0)


def test_matches_assembled_solver():
    for dims, cc in [((16, 9, 9), 200), ((24, 12, 12), 50), ((21, 11, 13), 300)]:
        mesh, fixed, f, e = _bar(*dims)
        ref = _reference(mesh, 0.3, fixed, e, f)
        g = GridElasticity(mesh, 0.3, fixed, rtol=1e-9, coarse_cells=cc, maxiter=3000, device="cpu")
        g.set_moduli(e)
        assert len(g.levels) >= 2, len(g.levels)
        u = g.solve_many([f])[0]
        err = np.linalg.norm(u - ref) / np.linalg.norm(ref)
        assert err < 1e-6, (dims, err)
        se = g.strain_energy(ref)
        se_ref = Stiffness(mesh, 0.3, fixed).strain_energy(ref)
        assert np.abs(se - se_ref).max() < 1e-9 * se_ref.max()


def test_layered_springs_and_removed_cells():
    mesh, fixed, f, e = _bar(18, 10, 10, seed=3)
    nu = Layered(0.35, ez=0.6, gz=0.5, normal=(0, 1, 1))
    # springs on a few tip nodes, and a third of the cells removed outright
    tip = np.nonzero(mesh.node_ijk[:, 0] == 18)[0][:5]
    dofs = (3 * tip[:, None] + np.arange(3)).ravel()
    springs = sp.csr_matrix((np.full(len(dofs), 0.05), (dofs, dofs)), shape=(mesh.n_dof, mesh.n_dof))
    e = e.copy()
    e[mesh.elem_ijk[:, 2] >= 7] = 0.0
    keep = e > 0
    sub = HexMesh(mesh.elem_ijk[keep], mesh.pitch, mesh.origin)
    fn = mesh.node_id(sub.node_ijk)
    fd = (3 * fn[:, None] + np.arange(3)).ravel()
    ref_sub = _reference(sub, nu, fixed[fd], e[keep], f[fd], springs[fd][:, fd])
    ref = np.zeros(mesh.n_dof)
    ref[fd] = ref_sub
    for dev in DEVICES:
        g = GridElasticity(mesh, nu, fixed, springs=springs, rtol=1e-9, coarse_cells=100, maxiter=3000, device=dev)
        g.set_moduli(e)
        u = g.solve_many([f * g.live_dofs()])[0]
        err = np.linalg.norm(u - ref) / np.linalg.norm(ref)
        assert err < 1e-6, (dev, err)


def test_torch_backend_matches_numpy():
    """gpu.py runs the same algorithm in the same precisions: same answer,
    same CG count; warm starts give the same answer. (Warm-start CG counts
    are left out: from a start already within ~1e-8, the count depends on
    the last digits of the previous solve, 2 against 22 on the 21x11x13
    bar with either backend.)"""
    if "torch" not in DEVICES:
        print("  (torch not installed: skipped)")
        return
    for dims, cc in [((16, 9, 9), 200), ((21, 11, 13), 300)]:
        mesh, fixed, f, e = _bar(*dims)
        res = {}
        for dev in ("cpu", "torch"):
            g = GridElasticity(mesh, 0.3, fixed, rtol=1e-8, coarse_cells=cc, maxiter=3000, device=dev)
            g.set_moduli(e)
            u = g.solve_many([f])[0]
            it = g.last_iters
            g.set_moduli(e * 1.1)
            w = g.solve_many([f], [u])[0]
            res[dev] = (u, it, w, g.last_iters)
        (u0, i0, w0, j0), (u1, i1, w1, j1) = res["cpu"], res["torch"]
        assert abs(i0 - i1) <= 2, (dims, i0, i1)
        assert np.linalg.norm(u1 - u0) < 1e-6 * np.linalg.norm(u0)
        assert np.linalg.norm(w1 - w0) < 1e-6 * np.linalg.norm(w0)


def test_simp_mg_matches_assembled():
    import json

    from loadpath.domain import build_domain
    from loadpath.simp import SimpOptions, filter_radius, optimize
    from loadpath.spec import study_from_dict

    here = Path(__file__).resolve().parents[1] / "loadpath" / "evals" / "cases"
    d = json.loads((here / "mbb-beam.json").read_text())
    d["resolution"] = {"maxElements": 6000}
    st = study_from_dict(d)
    dom = build_domain(st)
    opts = SimpOptions(volume_fraction=0.3, filter_radius=filter_radius(dom, st.min_member), max_iterations=25)
    out = {}
    old = os.environ.get("LOADPATH_SOLVER")
    try:
        for mode in ("assembled", "mg"):
            os.environ["LOADPATH_SOLVER"] = mode
            out[mode] = optimize(dom, st.elastic(design=True), opts)
    finally:
        if old is None:
            os.environ.pop("LOADPATH_SOLVER", None)
        else:
            os.environ["LOADPATH_SOLVER"] = old
    a, b = out["assembled"], out["mg"]
    assert abs(a.compliance - b.compliance) < 1e-3 * a.compliance, (a.compliance, b.compliance)
    assert ((a.xphys > 0.5) != (b.xphys > 0.5)).mean() < 0.005


TESTS = [test_transfers_are_adjoint, test_matches_assembled_solver, test_layered_springs_and_removed_cells,
         test_torch_backend_matches_numpy, test_simp_mg_matches_assembled]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
