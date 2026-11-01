"""The compiled mesh kernels (meshops.py, selfx.py, struts' strut kernel)
against the numpy paths they replace."""
import numpy as np
import pytest

from loadpath import crease, meshops, selfx, struts
from loadpath.fields import FieldGraph, field_mesh
from loadpath.narrowband import _face_cross_np, _triple_np

pytestmark = pytest.mark.skipif(not meshops.available(), reason="numba not installed")


def _mesh(pitch=0.6):
    doc = {"nodes": {
        "box": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [12, 12, 8]}},
        "px": {"op": "coordinate", "axis": "x"},
        "wall": {"op": "remap", "input": "px", "from": [0, 12], "to": [0.4, 1.0]},
        "g": {"op": "lattice", "type": "gyroid", "cell": 4, "thickness": "wall"},
        "part": {"op": "intersect", "inputs": ["g", "box"]}}, "output": "part"}
    g = FieldGraph(doc)
    return g, field_mesh(g.field(), *g.bounds(), pitch=pitch, project=0)


def _numpy(monkeypatch):
    monkeypatch.setenv("LOADPATH_MESHOPS", "numpy")
    monkeypatch.setenv("LOADPATH_SELFX", "numpy")


def test_bookkeeping_matches_numpy(monkeypatch):
    _, m = _mesh()
    V, F = np.asarray(m.vertices), np.asarray(m.faces, np.int64)
    n = len(V)
    seam = np.zeros(n, bool)
    seam[::9] = True
    X = np.random.default_rng(0).normal(size=(len(F), 3))
    verts = np.arange(0, n, 13)

    def run():
        vf = meshops.vertex_faces(F, n)
        N = meshops.vertex_sum(*vf, X)
        return (meshops.face_cross(V, F), meshops.triple(V, F), vf, meshops.neighbors(F, n, vf=vf),
                meshops.neighbors(F, n, seam=seam, vf=vf), N, meshops.min_cos(vf, X, N),
                meshops.faces_of(vf, verts, len(F)), meshops.edge_pairs(F, vf),
                meshops.row_dots(V, V, F[:, 0], F[:, 1]))

    fast = run()
    _numpy(monkeypatch)
    slow = run()
    np.testing.assert_allclose(fast[0], _face_cross_np(V, F), rtol=0, atol=1e-12)
    np.testing.assert_allclose(fast[1], _triple_np(V, F), rtol=1e-12, atol=1e-9)
    for a, b in zip(fast, slow):
        a, b = (a, b) if isinstance(a, tuple) else ((a,), (b,))
        for x, y in zip(a, b):
            if x.dtype.kind == "f":
                np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_array_equal(x, y)


def test_crossing_faces_matches_numpy(monkeypatch):
    """A relaxed mesh with folds unchecked: the compiled search finds the
    same crossing faces as the k-d tree one."""
    import loadpath.fields as LF

    g, _ = _mesh()
    monkeypatch.setattr(LF, "_UNFOLD", False)
    m = field_mesh(g.field(), *g.bounds(), pitch=0.6)
    V, F = np.asarray(m.vertices), np.asarray(m.faces, np.int64)
    every = np.arange(len(F))
    some = every[::5]
    fast = [selfx.crossing_faces(V, F, every), selfx.crossing_faces(V, F, some)]
    assert len(fast[0]) > 0
    _numpy(monkeypatch)
    slow = [crease.crossing_faces(V, F, every), crease.crossing_faces(V, F, some)]
    for a, b in zip(fast, slow):
        np.testing.assert_array_equal(a, b)


def test_two_crossing_triangles():
    V = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 1.5, 0.5],
                  [5, 5, 5], [6, 5, 5], [5, 6, 5]], float)
    F = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    np.testing.assert_array_equal(selfx.crossing_faces(V, F, np.array([0])), [0, 1])
    np.testing.assert_array_equal(selfx.crossing_faces(V, F, np.array([2])), [])


def test_strut_kernel_matches_numpy(monkeypatch):
    U = np.random.default_rng(1).uniform(-3, 3, (20_000, 3))
    blend = np.where(np.arange(len(U)) % 2, 0.06, 0.0)
    for kind in ("octet", "bcc", "kelvin", "diamond-strut"):
        fast = struts.unit_cell_distance(kind, U, blend)
        monkeypatch.setattr(struts, "_nb", None)
        slow = struts.unit_cell_distance(kind, U, blend)
        monkeypatch.undo()
        np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)


def test_field_mesh_same_with_numpy_paths(monkeypatch):
    """The whole field mesh (relax, unfold) is the same mesh either way."""
    g, _ = _mesh()
    a = field_mesh(g.field(), *g.bounds(), pitch=0.6)
    _numpy(monkeypatch)
    b = field_mesh(g.field(), *g.bounds(), pitch=0.6)
    assert len(a.faces) == len(b.faces)
    assert abs(a.volume / b.volume - 1) < 1e-6
    np.testing.assert_allclose(np.sort(np.asarray(a.vertices), axis=0), np.sort(np.asarray(b.vertices), axis=0),
                               atol=1e-6)
