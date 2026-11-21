"""
A fitted quad cage edited in place (loadpath/subdedit.py): a hole the cage
bridged is cut into it, two holes the cage ran together get the bridge of
material between them back, faces too wide for the shape are inset, and
only the control points round an edit are fitted again. Needs the
`surface` extra. Under a minute.

Run: `python3 tests/test_subdedit.py`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import trimesh  # noqa: E402

from loadpath import subd, subdedit  # noqa: E402

TOL, BOUND, MIN_HOLE = 0.1, 0.3, 2.0


def _voxel_cage(solid, size, origin):
    """The outside of a block of voxels as a closed quad cage."""
    nx, ny, nz = solid.shape
    pad = np.pad(solid, 1)
    ids, V, Q = {}, [], []

    def vid(i, j, k):
        if (i, j, k) not in ids:
            ids[(i, j, k)] = len(V)
            V.append(origin + np.array([i, j, k]) * size)
        return ids[(i, j, k)]
    # each side's corners, counter-clockwise seen from outside
    sides = {(1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
             (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
             (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]}
    for i, j, k in np.argwhere(solid):
        for d, cs in sides.items():
            if not pad[i + 1 + d[0], j + 1 + d[1], k + 1 + d[2]]:
                Q.append([vid(i + a, j + b, k + c) for a, b, c in cs])
    return subd.Cage(np.array(V, float), np.array(Q))


def _plate(holes, r):
    """A 40 x 20 x 4 plate with round through-holes at `holes` (x, y)."""
    m = trimesh.creation.box(extents=(40.0, 20.0, 4.0))
    for x, y in holes:
        c = trimesh.creation.cylinder(radius=r, height=10.0, sections=48)
        c.apply_translation([x, y, 0.0])
        m = trimesh.boolean.difference([m, c], engine="manifold")
    return m


def _fit(cage, mesh):
    data = subd._samples(mesh, 60_000)
    check = subd._samples(mesh, 40_000, seed=1)
    cage, H = subd.fit_cage(cage, mesh, data=data, settle=TOL / 50)
    dev, (d1, d2, _) = subd.deviation(H, cage.V, mesh, data=check)
    return cage, H, data, check, dev, d1, d2


def _euler(c):
    return c.n - len(c.E) + len(c.Q)


def test_bridged_hole_is_cut_in_place():
    """A cage without the plate's 5 mm hole: the hole is found (two
    membranes over its mouths, air between them) and cut in with four
    faces; a hole too small to print is left closed."""
    mesh = _plate([(1.0, 1.0)], 2.5)
    cage = _voxel_cage(np.ones((10, 5, 1), bool), np.array([4.0, 4.0, 4.0]), np.array([-20.0, -10.0, -2.0]))
    cage, H, data, check, dev, d1, d2 = _fit(cage, mesh)
    n0 = len(cage.Q)
    found = subdedit.holes(H, cage.V, mesh, d1 > BOUND, d1)
    assert len(found) == 1 and found[0]["faces"] is not None, found
    assert 4.0 < found[0]["diameter"] < 6.0 and 3.0 < found[0]["depth"] < 5.0, found
    assert dev["max"] > 2.0
    c2, H2, dv, _, left, log = subdedit.repair(cage, H, mesh, data, check, TOL, BOUND, MIN_HOLE, 3, dev, d1, d2)
    assert log[0]["holesCut"] == 1 and left["holes"] == 0, (log, left)
    assert _euler(cage) == 2 and _euler(c2) == 0
    assert log[0]["faces"] == n0 + 2, log                      # two faces out, four in
    assert dv["max"] < 0.5 * dev["max"], (dev, dv)
    assert len(subd._folds(H2, c2.V)) == 0
    # an unprintable hole is left closed
    small = _plate([(1.0, 1.0)], 0.8)
    c3 = _voxel_cage(np.ones((10, 5, 1), bool), np.array([4.0, 4.0, 4.0]), np.array([-20.0, -10.0, -2.0]))
    c3, H3, data, check, dev, d1, d2 = _fit(c3, small)
    c4, _, _, _, left, log = subdedit.repair(c3, H3, small, data, check, TOL, BOUND, MIN_HOLE, 3, dev, d1, d2)
    assert _euler(c4) == 2 and left["holes"] == 0 and not any(e["holesCut"] for e in log), log


def test_slit_between_two_holes_is_bridged():
    """A cage with one long slot where the plate has two holes and solid
    between them: the slot's walls sit inside the solid; a bridge of four
    faces across it gives the plate its two holes back."""
    mesh = _plate([(-7.0, 0.0), (7.0, 0.0)], 3.0)
    solid = np.ones((20, 10, 1), bool)
    solid[4:16, 4:6] = False                                   # x -12..12, y -2..2
    cage = _voxel_cage(solid, np.array([2.0, 2.0, 4.0]), np.array([-20.0, -10.0, -2.0]))
    cage, H, data, check, dev, d1, d2 = _fit(cage, mesh)
    assert _euler(cage) == 0 and mesh.euler_number == -2
    found = subdedit.slits(H, cage.V, mesh, d1, BOUND)
    assert found, "no slit found"
    c2, H2, dv, _, left, log = subdedit.repair(cage, H, mesh, data, check, TOL, BOUND, MIN_HOLE, 3, dev, d1, d2)
    assert log[0]["slitsBridged"] >= 1, log
    assert _euler(c2) == -2, (_euler(c2), log)
    assert dv["max"] < dev["max"] and left["holes"] == 0, (dev, dv, left)
    # the middle of the bridge is inside the surface now
    P = H2.M @ c2.V
    Qf = H2.fine.Q
    lm = trimesh.Trimesh(P, np.concatenate([Qf[:, [0, 1, 2]], Qf[:, [0, 2, 3]]]), process=False)
    assert subd._signed_distance(lm, np.zeros((1, 3)))[0] < -1.0


def test_inset_adds_a_ring_and_keeps_the_rest():
    """Two flagged faces: a ring of six quads inside their outline, eight
    star points (the outline's four corners and their copies), the cage
    closed, every face outside unchanged."""
    cage = _voxel_cage(np.ones((6, 6, 2), bool), np.array([5.0, 5.0, 5.0]), np.zeros(3))
    top = [q for q in range(len(cage.Q)) if (cage.V[cage.Q[q], 2] == 10.0).all()]
    cen = cage.V[cage.Q[top]].mean(axis=1)
    a = top[int(np.argmin(np.linalg.norm(cen - [12.5, 12.5, 10.0], axis=1)))]
    b = top[int(np.argmin(np.linalg.norm(cen - [17.5, 12.5, 10.0], axis=1)))]
    V, Q, touched = subdedit.inset(cage.V, cage.Q, [a, b])
    c2 = subd.Cage(V, Q)
    assert len(Q) == len(cage.Q) + 6 and _euler(c2) == _euler(cage) == 2
    assert int((c2.valence != 4).sum()) == int((cage.valence != 4).sum()) + 8
    keep = np.ones(len(cage.Q), bool)
    keep[[a, b]] = False
    assert (Q[:len(cage.Q)][keep] == cage.Q[keep]).all()
    # an L of three faces is filled to a square first (a notch costs two more star points)
    c = top[int(np.argmin(np.linalg.norm(cen - [12.5, 17.5, 10.0], axis=1)))]
    V, Q, _ = subdedit.inset(cage.V, cage.Q, [a, b, c])
    assert len(Q) == len(cage.Q) + 8
    assert int((subd.Cage(V, Q).valence != 4).sum()) == int((cage.valence != 4).sum()) + 8


def test_refit_moves_only_the_control_points_round_the_edit():
    mesh = trimesh.creation.icosphere(4, radius=20.0)
    cage = _voxel_cage(np.ones((6, 6, 6), bool), np.array([5.0, 5.0, 5.0]), np.array([-15.0, -15.0, -15.0]))
    cage, H, data, check, dev, d1, d2 = _fit(cage, mesh)
    V0 = cage.V.copy()
    i = int(np.argmax(V0[:, 2]))
    cage.V[i] += [0.0, 0.0, 3.0]
    free = subdedit.refit(cage, H, mesh, [i], data, settle=TOL / 50)
    held = np.setdiff1d(np.arange(cage.n), free)
    assert len(free) < 0.5 * cage.n and (cage.V[held] == V0[held]).all()
    assert np.abs(cage.V - V0).max() < 1.0, np.abs(cage.V - V0).max()       # the 3 mm push is undone
    dv, _ = subd.deviation(H, cage.V, mesh, data=check)
    assert dv["max"] < dev["max"] + 0.05, (dev, dv)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
