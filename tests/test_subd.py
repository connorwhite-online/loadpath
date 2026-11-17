"""
Organic surfacing (loadpath/subd.py): a quad cage's Catmull-Clark surface
fitted to a mesh and written as smooth patches in one closed STEP solid.
Needs the `surface` extra. About a minute.

Run: `python3 tests/test_subd.py`.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import trimesh  # noqa: E402

from loadpath import subd  # noqa: E402


def _torus_cage(R=30.0, r=10.0, nu=24, nv=12):
    U, V = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    u, v = 2 * np.pi * U / nu, 2 * np.pi * V / nv
    P = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], -1).reshape(-1, 3)
    idx = lambda i, j: (i % nu) * nv + (j % nv)  # noqa: E731
    Q = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(nu) for j in range(nv)])
    return subd.Cage(P, Q)


def test_regular_cage_gives_exact_c2_patches():
    """All-regular cage: every patch is the limit surface itself (bicubic),
    and neighbours meet with the same normal; the exact limit normals
    agree with the patches'."""
    cage = _torus_cage()
    H = subd.Hierarchy(cage, 3)
    bs, lay = subd.patches(H, cage.V)
    q = subd.patch_quality(H, cage.V, bs, lay)
    assert q["patchToLimit"]["max"] < 1e-9, q
    assert q["edgeAngle"]["max"] < 1e-4, q
    N = subd.grid_normals(H, cage.V)
    t = np.arange(9) / 8
    n0 = subd._patch_normals(bs[0], np.repeat(t, 9), np.tile(t, 9)).reshape(9, 9, 3)
    assert np.abs(np.einsum("ijk,ijk->ij", n0, N[0]) - 1).max() < 1e-9


def _blob():
    m = trimesh.creation.icosphere(5, radius=20.0)
    V = m.vertices.copy()
    V[:, 2] *= 0.6
    V[:, 0] += 3 * np.sin(V[:, 1] / 5)
    return trimesh.Trimesh(V, m.faces)


def test_blob_to_step():
    """A curved body with extraordinary vertices: fitted within the
    tolerance, the patches G1 to a fraction of a degree, one valid closed
    solid holding the mesh's volume, and the cage written beside it."""
    m = _blob()
    with tempfile.TemporaryDirectory() as td:
        rep = subd.export_surface(m, os.path.join(td, "blob.step"), cage_path=os.path.join(td, "blob.cage.obj"))
        assert os.path.getsize(rep["cage"]) > 0
    rep.pop("shape")
    assert rep["valid"] and rep["freeEdges"] == 0, rep
    assert rep["within"], rep["deviation"]
    assert rep["folds"] == 0
    assert rep["irregularVertices"] > 0
    assert rep["patches"]["edgeAngle"]["max"] < 1.0, rep["patches"]
    assert rep["patches"]["patchToLimit"]["regularMax"] < 1e-9
    assert abs(rep["volume"] / rep["meshVolume"] - 1) < 2e-3, rep


def test_tolerance_from_process_and_noise():
    tol, why = subd.pick_tolerance()
    assert abs(tol - 0.1) < 1e-12 and "accuracy" in why
    tol, why = subd.pick_tolerance(accuracy=0.2, noise=0.2)
    assert abs(tol - 0.4) < 1e-12 and "noise" in why
    # a clean mesh reads as nearly noiseless; jitter reads as about itself
    m = _blob()
    assert subd.noise(m) < 0.01
    j = m.copy()
    j.vertices += np.random.default_rng(0).normal(0, 0.05, j.vertices.shape)
    assert 0.03 < subd.noise(j) < 0.08


def test_cage_repairs():
    """The remesher's leftovers: a 4-sided hole closes with one quad, a
    6-sided one round a centre; a doublet's two quads become one."""
    cage = _torus_cage(nu=8, nv=6)
    V, Q = cage.V, cage.Q
    v, f = subd._close_holes(V, Q[1:])
    assert len(f) == len(Q) and len(v) == len(V)
    subd.Cage(v, f)
    # a doublet: split one quad (a, b, c, d) with a vertex on its diagonal
    a, b, c, d = Q[0]
    w = len(V)
    v2 = np.vstack([V, V[[a, c]].mean(0)])
    f2 = np.vstack([Q[1:], [[w, a, b, c], [w, c, d, a]]])
    v3, f3 = subd._dissolve_doublets(v2, f2)
    assert len(f3) == len(Q) and len(v3) == len(V)
    subd.Cage(v3, f3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
