"""
Patterns of a body (the field graph's `pattern` op, patterns.py): linear
grids, circular (turned or not), mirror and along a curve, the copies'
union exact where they meet and the bounds covering every copy.

Run: `python tests/test_patterns.py` (or pytest). Under a minute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loadpath.fields import FieldGraph
from loadpath.patterns import instances

BOSS = {"op": "shape", "shape": {"type": "cylinder", "center": [20, 0, 5], "r": 3, "length": 10, "axis": "z"}}
EACH = np.pi * 9 * 10


def _g(**p):
    return FieldGraph({"nodes": {"boss": BOSS, "pat": {"op": "pattern", "input": "boss", **p}}, "output": "pat"})


@pytest.mark.parametrize("spec, copies", [
    ({"type": "circular", "count": 6}, 6),
    ({"type": "circular", "count": 5, "angle": 90}, 5),
    ({"type": "linear", "directions": [{"step": [10, 0, 0], "count": 5}, {"step": [0, 10, 0], "count": 3}]}, 15),
    ({"type": "mirror", "plane": {"normal": "x"}}, 2),
    ({"type": "curve", "path": [[0, 0, 0], [50, 0, 0], [50, 50, 0]], "count": 7}, 7),
])
def test_each_copy_meshes_whole(spec, copies):
    m = _g(**spec).mesh(pitch=0.5)
    assert len(m.split()) == copies
    assert abs(m.volume / (copies * EACH) - 1) < 0.03


def test_circular_copies_sit_on_the_circle():
    g = _g(type="circular", count=4)
    ring = np.array([[20, 0, 5], [0, 20, 5], [-20, 0, 5], [0, -20, 5], [14.1, 14.1, 5]], float)
    v = g.evaluate(ring, "pat")
    assert np.allclose(v[:4], -3, atol=1e-6) and v[4] > 0
    lo, hi = g.bounds("pat")
    assert np.allclose(lo, [-23, -23, 0]) and np.allclose(hi, [23, 23, 10])


def test_nearest_copies_match_every_copy():
    P = np.random.default_rng(1).uniform([-30, -30, -2], [30, 30, 12], (4000, 3))
    a = _g(type="circular", count=24).evaluate(P, "pat")
    b = _g(type="circular", count=24, exact=True).evaluate(P, "pat")
    inside = b < 0
    assert np.array_equal(a < 0, inside)
    assert np.allclose(a[inside | (b < 3)], b[inside | (b < 3)], atol=1e-9)


def test_curve_copies_turn_with_the_curve():
    inst = instances({"type": "curve", "path": [[0, 0, 0], [10, 0, 0], [10, 10, 0]], "count": 3})
    R_last = inst[-1][0]
    assert np.allclose(R_last[:, 0], [0, 1, 0])           # x along the second leg


def test_bad_patterns_are_refused():
    with pytest.raises(ValueError, match="pattern type"):
        _g(type="spiral").evaluate(np.zeros((1, 3)), "pat")
    with pytest.raises(ValueError, match="mirror pattern needs plane"):
        _g(type="mirror").evaluate(np.zeros((1, 3)), "pat")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
