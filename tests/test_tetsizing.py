"""crosscheck's fTetWild sizing field: coarse outside the part, the asked
size inside it and near its surface, so the kept mesh is the same."""
import numpy as np
import pytest

pytetwild = pytest.importorskip("pytetwild")
m3d = pytest.importorskip("manifold3d")

from loadpath import crosscheck as cc
from loadpath.shapes import trimesh_from_manifold


def _part():
    # a slender bracket in a mostly empty box, like the GRIP links
    body = m3d.Manifold.cube((60, 8, 8)) + m3d.Manifold.cube((8, 8, 40))
    hole = m3d.Manifold.cylinder(20, 2.5, 2.5, 32).translate((30, 4, -5))
    return trimesh_from_manifold((body - hole).refine(4))


def test_field_is_fine_in_and_near_the_part():
    mesh = _part()
    v = np.asarray(mesh.vertices)
    size = 1.5
    G, tets, val = cc._outside_coarse(mesh, v, size)
    assert tets.min() >= 0 and tets.max() < len(G)
    x = G[tets]
    assert np.all(np.linalg.det(x[:, 1:] - x[:, :1]) > 0)
    inside = mesh.contains(G)
    assert np.all(val[inside] == size)
    # every background tet that holds a surface point is fine at all corners
    from scipy.spatial import Delaunay
    surf = mesh.sample(2000, seed=0) if hasattr(mesh, "sample") else v
    owner = Delaunay(G).find_simplex(surf)
    assert np.all(owner >= 0)
    lo = G.min(0)
    h = 4 * size
    cell = np.floor((surf - lo) / h).astype(int)
    n = np.array([len(np.unique(G[:, i])) for i in range(3)])
    for c in cell[::50]:
        corners = [((c[0] + a) * n[1] + c[1] + b) * n[2] + c[2] + d for a in (0, 1) for b in (0, 1) for d in (0, 1)]
        assert np.all(val[corners] == size)
    assert val.max() == pytest.approx(cc._COARSE_OUTSIDE * size)


def test_same_mesh_in_the_part(monkeypatch):
    monkeypatch.setenv("LOADPATH_TET_CACHE", "0")
    mesh = _part()
    size = 1.5

    def stats(pts, tets):
        x = pts[tets]
        e = np.linalg.norm(x[:, [0, 0, 0, 1, 1, 2]] - x[:, [1, 2, 3, 2, 3, 3]], axis=2)
        return len(tets), np.abs(np.linalg.det(x[:, 1:] - x[:, :1])).sum() / 6, e.mean()

    new = stats(*cc._tetwild(mesh, size))
    monkeypatch.setenv("LOADPATH_TET_SIZING", "0")
    old = stats(*cc._tetwild(mesh, size))
    assert new[1] == pytest.approx(mesh.volume, rel=0.01)
    assert new[1] == pytest.approx(old[1], rel=0.01)
    assert new[0] == pytest.approx(old[0], rel=0.25)
    assert new[2] == pytest.approx(old[2], rel=0.1)
