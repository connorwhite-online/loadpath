import numpy as np
import pytest
import trimesh

from loadpath.retention import body_from_mesh, nudge_check


def _box(c, e):
    m = trimesh.creation.box(extents=e)
    m.apply_translation(c)
    return m


def _cup():
    """A closed box with a 6 mm hole in its top: walls 2 mm."""
    outer = _box((0, 0, 0), (30, 30, 20))
    return outer


def _shell_sdf(hole_r):
    def f(P):
        P = np.asarray(P, float)
        q = np.abs(P) - np.array([15.0, 15.0, 10.0])
        out = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
        wall = np.maximum(out, -(out + 2.0))
        hole = np.maximum(np.hypot(P[:, 0], P[:, 1]) - hole_r, -(P[:, 2] - 5.0))
        return np.maximum(wall, -hole)
    return f


def _plug(lip):
    """A cylinder r 2.9 in a hole r 3, through the 2 mm top wall, with a
    lip under the wall if `lip`."""
    m = trimesh.creation.cylinder(radius=2.9, height=4.0, sections=48)
    m.apply_translation((0, 0, 9.0))
    if lip:
        l = trimesh.creation.cylinder(radius=5.0, height=1.0, sections=48)
        l.apply_translation((0, 0, 7.45))
        m = trimesh.util.concatenate([m, l])
    return m


def test_cap_without_lip_falls_out_with_lip_is_held():
    from loadpath.retention import body_from_sdf
    shell = body_from_sdf("shell", _shell_sdf(3.0), (-16, -16, -11), (16, 16, 11), pitch=0.5)
    # something below so the plug cannot drop in
    floor = body_from_mesh("floor", _box((0, 0, 6.0), (12, 12, 1.0)))
    for lip, ok in ((False, False), (True, True)):
        plug = body_from_mesh("plug", _plug(lip))
        rep = nudge_check([shell, floor, plug], held={"shell": "screw", "floor": "glue"}, directions=40, axes=8)
        assert rep["bodies"]["plug"]["ok"] is ok, rep["bodies"]["plug"]
        if not ok:
            # it goes out through the top
            assert any(d[2] > 0.9 for d in rep["bodies"]["plug"]["freeDirections"])


def test_declared_fastening_passes_and_unknown_kind_is_refused():
    a = body_from_mesh("a", _box((0, 0, 0), (5, 5, 5)))
    b = body_from_mesh("b", _box((20, 0, 0), (5, 5, 5)))
    rep = nudge_check([a, b], held={"a": "screw", "b": {"by": "glue"}}, directions=20, axes=4)
    assert rep["ok"]
    rep = nudge_check([a, b], held={"a": "screw"}, directions=20, axes=4)
    assert not rep["ok"] and rep["failed"] == ["b"]
    with pytest.raises(ValueError):
        nudge_check([a, b], held={"a": "hope"}, directions=20, axes=4)


def test_group_moves_as_one():
    a = body_from_mesh("a", _box((0, 0, 0), (5, 5, 5)))
    b = body_from_mesh("b", _box((5, 0, 0), (5, 5, 5)))
    rep = nudge_check([a, b], groups={"ab": ["a", "b"]}, held={"ab": "bolt"}, directions=20, axes=4)
    assert rep["ok"] and list(rep["bodies"]) == ["ab"]
