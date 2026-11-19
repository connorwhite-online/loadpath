"""
envelope.py: a first exterior round the parts. A box part wrapped by a
style that is too small for it, the wrap's clearance, where the part pushed
the style out, a view funnel that keeps a field of view clear (and one
that does not), the keep-out field far from its part, and the hand data.

Run: `python tests/test_envelope.py` (or pytest). A few seconds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath import envelope as E
from loadpath.sdf_kit import smax


def _part():
    return trimesh.creation.box(extents=(40, 20, 10))       # centred on the origin


def _style(P):                                              # a 30 x 30 x 30 ball: too small for the part
    return np.linalg.norm(P, axis=1) - 15.0


def test_keepout_far_field_is_a_lower_bound():
    k = E.Keepout("box", _part(), 2.0)
    P = np.array([[0, 0, 0], [0, 0, 8.0], [0, 0, 40.0], [100.0, 0, 0]])
    d = k.sdf(P)
    assert d[0] < -4.0
    assert abs(d[1] - 3.0) < 0.6                 # 8 - 5
    assert d[2] > 30.0                           # 35 exactly; never short by more than the grid
    assert d[3] > 60.0


def test_wrap_contains_the_part_and_reports_the_push():
    ks = [E.Keepout("box", _part(), 2.0)]
    f = E.wrap(_style, ks, k=3.0)
    cl = E.clearances(f, ks, n=2000)["box"]
    assert cl["ok"], cl
    push = E.pushes(_style, f, ((-30, -20, -15), (30, 20, 15)), pitch=1.0)
    assert push and push[0]["out_mm"] > 4.0      # the box's ends stick out of the ball
    xs = [r["box"][0][0] for r in push] + [r["box"][1][0] for r in push]
    assert min(xs) < -17 and max(xs) > 17


def test_view_funnel_keeps_a_field_of_view_clear():
    body = lambda P: np.max(np.abs(P - [0, -5, 0]), axis=1) - 10.0      # a 20 mm cube, its front (y = 5) a cowl ahead of the glass
    apex = (0, -2.0, 0)
    for ang, ok in (((30, 20), True), ((2, 2), False)):
        fun = lambda P, a=ang: E.funnel(P, (0, -0.5, 0), (0, 1, 0), (1.5, 1.5), a, 9.0)
        f = lambda P, fn=fun: smax(body(P), -fn(P), 0.3)
        v = E.view_check(f, apex, (0, 1, 0), (25, 15), length=30)
        assert (v["blocked"] == 0) == ok, (ang, v)


def test_sweep_section_sizes():
    spine = [(0, 0, 0), (0, 0, -50), (0, 0, -100)]
    P = np.array([[17.0, 0, -50], [0, 26.0, -50], [0, 0, -50]])
    d = E.sweep(P, spine, 17.0, 26.0, n=2.6)
    assert abs(d[0]) < 0.2 and abs(d[1]) < 0.2 and d[2] < -10


def test_sweep_sizes_stay_with_unevenly_spaced_points():
    # a grip closing to a round end: most points along the shaft, a few
    # bunched in the last 15 mm. The sizes belong to the points, not to
    # their index (2026-11-19: they followed the index, and the shaft
    # tapered from halfway down)
    z = [0.0, -40.0, -80.0, -100.0, -108.0, -113.0, -115.0]
    w = [25.0, 25.0, 25.0, 25.0, 18.0, 10.0, 2.0]
    spine = [(0, 0, zi) for zi in z]
    P = np.array([[25.0, 0, -60.0], [25.0, 0, -95.0]])
    d = E.sweep(P, spine, w, 15.0, n=2.6)
    assert np.all(np.abs(d) < 0.3), d


def test_hand_targets():
    h = E.hands()
    assert h["hand_length"]["male"]["p50"] > h["hand_length"]["female"]["p50"]
    t = E.grip_targets(h["hand_length"]["female"]["p50"])
    assert 30 < t["diameter"] < 45 and 55 < t["trigger_reach"] < 75


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
