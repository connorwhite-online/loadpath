"""enclosure on a designer's fixed part: supports and mounts must land on
material that part has, or verify stops with "touches no material"."""
import numpy as np

from loadpath.enclosure import _mounts_on_material, _support_points
from loadpath.shapes import shape_from_spec

EXT = shape_from_spec({"type": "cylinder", "center": [0, 0, 4], "axis": "z", "length": 8, "r": 29}, None)
FLOOR = {"type": "cylinder", "center": [0, 0, 0.425], "axis": "z", "length": 0.85, "r": 29}
HOLE = {"type": "box", "min": [-15, -15, -1], "max": [15, 15, 2]}


def _housing(window=True):
    spec = {"type": "subtract", "shapes": [FLOOR, HOLE]} if window else FLOOR
    return shape_from_spec(spec, None)


def test_support_stays_in_the_middle_on_a_solid_floor():
    pts, moved = _support_points(EXT, np.array([0, 0, 4.0]), np.array([0, 0, -1.0]), 3.0, [_housing(False)],
                                 np.array([-29, -29, 0.0]), np.array([29, 29, 8.0]))
    assert not moved and len(pts) == 1
    assert np.allclose(pts[0][:2], 0, atol=0.1)


def test_supports_move_off_a_floor_window_onto_material():
    h = _housing(True)
    pts, moved = _support_points(EXT, np.array([0, 0, 4.0]), np.array([0, 0, -1.0]), 3.0, [h],
                                 np.array([-29, -29, 0.0]), np.array([29, 29, 8.0]))
    assert moved and len(pts) == 3
    for q in pts:
        assert h.sdf(np.atleast_2d(q) + [0, 0, 0.3])[0] <= 0.05, q      # just inside the floor
        assert np.max(np.abs(q[:2])) > 15                                 # outside the window
    a = sorted(np.degrees(np.arctan2(q[1], q[0])) % 360 for q in pts)
    assert min(np.diff(a + [a[0] + 360])) >= 100                           # spread round the window


def test_supports_without_fixed_parts_are_unchanged():
    pts, moved = _support_points(EXT, np.array([0, 0, 4.0]), np.array([0, 0, -1.0]), 3.0, [],
                                 np.array([-29, -29, 0.0]), np.array([29, 29, 8.0]))
    assert not moved and len(pts) == 1


def test_a_mount_over_the_window_hands_its_mass_to_one_on_material():
    over = {"type": "box", "min": [-8, -8, 0.05], "max": [8, 8, 0.85]}
    ring = {"type": "box", "min": [18, -4, 0.0], "max": [26, 4, 0.85]}
    mounts = {"engine": {"shape": over, "mass": 0.004}, "magnets": {"shape": ring, "mass": 0.0065}}
    lay = {"components": {"engine": {"centre": [0, 0, 4]}, "magnets": {"centre": [22, 0, 1.4]}}}
    warn = []
    out = _mounts_on_material(mounts, [_housing(True)], lay, None, warn)
    assert set(out) == {"magnets"}
    assert abs(out["magnets"]["mass"] - 0.0105) < 1e-9
    assert any("mount-engine" in w for w in warn)
    # on a solid floor nothing moves
    warn = []
    out = _mounts_on_material(mounts, [_housing(False)], lay, None, warn)
    assert set(out) == {"engine", "magnets"} and not warn


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    import sys
    import traceback
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:
            bad += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    sys.exit(1 if bad else 0)
