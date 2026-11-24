"""Isosurface stuffing (stuffing.py): the tet mesh the cross-check solves on."""
import numpy as np
import pytest

m3d = pytest.importorskip("manifold3d")

from loadpath.shapes import trimesh_from_manifold
from loadpath.stuffing import stuff


def _part():
    # a slender bracket with a bore, like the robot-arm links
    body = m3d.Manifold.cube((60, 8, 8)) + m3d.Manifold.cube((8, 8, 40))
    hole = m3d.Manifold.cylinder(20, 2.5, 2.5, 32).translate((30, 4, -5))
    return trimesh_from_manifold((body - hole).refine(4))


def _faces(tets):
    f = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), 1)
    return np.unique(f, axis=0, return_counts=True)


def _dihedrals(x):
    out = []
    for i, j, k, l in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]:
        e = x[:, j] - x[:, i]
        e /= np.linalg.norm(e, axis=1)[:, None]
        a = x[:, k] - x[:, i]
        a -= (a * e).sum(1)[:, None] * e
        b = x[:, l] - x[:, i]
        b -= (b * e).sum(1)[:, None] * e
        c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.array(out)


def test_fills_the_part_and_conforms():
    import trimesh

    mesh = _part()
    info = {}
    pts, tets = stuff(mesh, 1.5, info)
    x = pts[tets]
    vol = np.linalg.det(x[:, 1:] - x[:, :1]) / 6
    assert vol.min() > 0
    # the volume, less the chords' cut off a curved surface
    assert vol.sum() == pytest.approx(mesh.volume, rel=0.01)
    faces, count = _faces(tets)
    assert count.max() == 2                      # no face shared three ways
    skin = trimesh.Trimesh(pts, faces[count == 1], process=False)
    assert skin.area == pytest.approx(mesh.area, rel=0.02)
    # every surface vertex is on the given surface
    on = np.unique(faces[count == 1])
    d = np.abs(trimesh.proximity.signed_distance(mesh, pts[on]))
    assert d.max() < 1e-6
    # warped lattice tets keep their shape (the paper's bound: 10.7-164.8)
    a = _dihedrals(x)
    assert a.min() > 10.7 and a.max() < 164.8, (a.min(), a.max())
    assert info["missedCuts"] == 0 and info["snapped"] > 0


def test_box_edges_stay_sharp():
    # a square beam off the lattice: its four edges are creases, and the
    # surface vertices beside them move onto them, so the beam keeps its
    # section (cut off as chamfers it was 3.5% light and 13% too soft)
    mesh = trimesh_from_manifold(m3d.Manifold.cube((100, 10, 10)).refine(20))
    mesh.apply_translation((0.37, 0.61, 0.83))
    pts, tets = stuff(mesh, 2.5)
    x = pts[tets]
    vol = np.linalg.det(x[:, 1:] - x[:, :1]) / 6
    assert vol.min() > 0
    assert vol.sum() == pytest.approx(10000.0, rel=0.002)
    pts, tets = stuff(mesh, 2.5, creases=False)
    x = pts[tets]
    assert np.linalg.det(x[:, 1:] - x[:, :1]).sum() / 6 < 9700.0


def test_same_mesh_every_time():
    mesh = _part()
    p1, t1 = stuff(mesh, 1.5)
    p2, t2 = stuff(mesh, 1.5)
    assert np.array_equal(p1, p2) and np.array_equal(t1, t2)


def test_moved_part_moves_the_mesh_with_it():
    # the lattice is anchored to the world in steps of L: a part moved by
    # whole steps gets the same mesh, moved
    mesh = _part()
    p1, t1 = stuff(mesh, 1.5)
    moved = mesh.copy()
    moved.apply_translation((3.0, -4.5, 6.0))
    p2, t2 = stuff(moved, 1.5)
    assert np.array_equal(t1, t2)
    d = np.abs(p2 - (3.0, -4.5, 6.0) - p1).max(1)
    # (a vertex halfway between two cuts may take the other one: the
    # rounding of the moved coordinates decides a tie)
    assert (d < 1e-9).mean() > 0.995 and d.max() < 1.5
