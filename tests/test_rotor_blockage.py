"""Structure in a rotor's stream (aero.rotor_blockage): drag of what the
disk's air meets, from momentum theory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath.aero import RotorStream, rotor_blockage


def _bar(z, length=40.0, width=4.0, x0=5.0):
    b = trimesh.creation.box(extents=[length, width, 4.0])
    b.apply_translation([x0 + length / 2, 0, z])
    return b


def test_bar_under_a_prop_drags_as_momentum_theory_says():
    # a 4 x 40 mm bar 10 mm under a 45 mm disk at 0.78 N, hub 10 mm. The
    # wake has narrowed by sqrt(v_i / v) there, so less of the bar is in it
    r = rotor_blockage(_bar(-12.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0)
    v_i = np.sqrt(0.78 / (2 * 1.2 * np.pi * 0.045 ** 2))
    k = 1 + 10 / np.hypot(45, 10)                 # v / v_i at the bar's face
    wet = (45 - 10) / np.sqrt(k) * 4              # mm^2 of the bar inside the narrowed stream
    expect = 1.2 * 0.5 * 1.2 * (v_i * k) ** 2 * wet * 1e-6
    assert abs(r["drag"] - expect) < 0.06 * expect, (r, expect)
    # the area is the disk's whose tubes meet it: the bar's, widened back out
    assert abs(r["downstream"]["blockedArea"] - wet * k) < 0.06 * wet * k, r
    assert r["upstream"]["blockedArea"] == 0
    # just under the disk the stream is still the disk's
    near = rotor_blockage(_bar(-2.1), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0)
    assert abs(near["downstream"]["blockedArea"] - 140) < 6, near
    # far behind it the air moves twice as fast through half the area: four
    # times the pressure on a bar whose reach into the stream fell by sqrt(2)
    far = rotor_blockage(_bar(-200.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0, reach=300)
    want = 4 * (45 / np.sqrt(2) - 10 / np.sqrt(2)) / 35
    assert abs(far["drag"] / near["drag"] - want) < 0.25, (far["drag"], near["drag"], want)
    # ahead of the disk the air is slow: much less than at it
    ahead = rotor_blockage(_bar(30.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0)
    assert 0 < ahead["upstream"]["drag"] < 0.3 * near["drag"]
    # in a duct the stream leaves at sqrt(2) v_i: twice the open disk's pressure
    duct = rotor_blockage(_bar(-2.1), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0, ducted=True)
    assert 1.7 < duct["drag"] / near["drag"] < 2.2


def test_the_wake_narrows_and_the_first_material_shadows_the_rest():
    one = rotor_blockage(_bar(-12.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78)
    behind = rotor_blockage(_bar(-30.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78)
    both = rotor_blockage(trimesh.util.concatenate([_bar(-12.0), _bar(-30.0)]), [0, 0, 0], [0, 0, -1], 45.0, 0.78)
    # the second bar sits in the first one's wake: only the tubes that
    # closed in past the first one's sides still find it
    assert one["drag"] <= both["drag"] < one["drag"] + 0.25 * behind["drag"], (one, behind, both)
    off = rotor_blockage(_bar(-12.0, x0=60.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78)
    assert off["drag"] == 0.0
    # a bar out past the disk's edge is clear of the wake behind it, but
    # ahead of the disk the inflow comes from wider than the disk
    edge = _bar(-40.0, length=10.0, x0=40.0)
    assert rotor_blockage(edge, [0, 0, 0], [0, 0, -1], 45.0, 0.78)["drag"] == 0.0
    edge.apply_translation([8, 0, 80])
    assert rotor_blockage(edge, [0, 0, 0], [0, 0, -1], 45.0, 0.78)["upstream"]["drag"] > 0


def test_stream_drag_gradient():
    # a tilted rotor over a grey field: the gradient against finite differences
    rng = np.random.default_rng(0)
    st = RotorStream([-20, -20, -30], 2.0, (20, 20, 10), [0, 0, 0], [0.2, 0.1, -1], 18.0, 0.5, hub=3)
    x = rng.random(4000) * 0.9
    _, g = st.drag(x, grad=True)
    live = np.flatnonzero(np.abs(g) > 1e-7)
    assert len(live) > 200
    for e in rng.choice(live, 8, replace=False):
        xa, xb = x.copy(), x.copy()
        xa[e] += 1e-5
        xb[e] -= 1e-5
        fd = (st.drag(xa)["drag"] - st.drag(xb)["drag"]) / 2e-5
        assert abs(fd - g[e]) < 1e-5 * max(abs(fd), 1e-6), (e, fd, g[e])


PLATE = {
    "designSpace": {"type": "box", "min": [0, -30, 0], "max": [100, 30, 6]},
    "supports": [{"shape": {"type": "box", "min": [-1, -31, -1], "max": [0.01, 31, 7]}}],
    "loadCases": [{"name": "tip", "loads": [{"shape": {"type": "box", "min": [99, -6, -1], "max": [101, 6, 7]},
                                             "force": [0, 0, -20]}]}],
    "material": "pa12", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 2.5},
}
PROP = {"name": "prop", "center": [50, 0, 20], "axis": "-z", "radius": 25, "thrust": 0.24}


def test_optimizer_routes_round_the_stream():
    # a plate from a wall to a tip load, with a prop over its middle at a
    # drone's disk loading: left alone the optimizer runs a beam straight
    # under the prop; charged for the thrust it blocks, it goes round
    from loadpath.domain import build_domain
    from loadpath.simp import SimpOptions, filter_radius, optimize
    from loadpath.spec import study_from_dict

    plain = build_domain(study_from_dict(PLATE))
    aero = build_domain(study_from_dict({**PLATE, "aero": {"rotors": [PROP]}}))
    assert plain.blockage is None and aero.blockage is not None
    got = {}
    for name, d in (("plain", plain), ("aero", aero)):
        r = optimize(d, 0.39, SimpOptions(volume_fraction=0.3, filter_radius=filter_radius(d, 5.0), max_iterations=60))
        cells, _, drag = aero.blockage.cost(r.xphys)
        got[name] = (drag, r.compliance, r.history[-1])
    cell_g = 1.01e-3 * 2.5 ** 3                    # PA12, g per cell
    # the straight beam blocks grams' worth of thrust, on a part of ~10 g
    assert got["plain"][0] * 1000 / 9.81 > 2.0, got["plain"][0]
    assert got["aero"][0] < 0.1 * got["plain"][0], got
    assert got["aero"][1] > got["plain"][1]        # the detour is paid for in stiffness
    assert "blockageDrag" in got["aero"][2] and "blockageDrag" not in got["plain"][2]
    assert abs(aero.blockage.cells_per_newton - 1000 / 9.81 / cell_g) < 1e-6 * aero.blockage.cells_per_newton


def test_aero_block_is_checked():
    import pytest

    from loadpath.spec import study_from_dict

    s = study_from_dict({**PLATE, "aero": {"rotors": [{**PROP, "axis": [0, 0, -2], "ducted": True}],
                                           "inOptimizer": False}})
    assert np.allclose(s.aero.rotors[0].axis, [0, 0, -1]) and s.aero.rotors[0].ducted and not s.aero.in_optimizer
    for bad in ({}, {"rotors": []}, {"rotors": [{**PROP, "thrust": 0}]}, {"rotors": [{**PROP, "axis": "up"}]},
                {"rotors": [{**PROP, "hub": 30}]}, {"rotors": [{k: v for k, v in PROP.items() if k != "center"}]}):
        with pytest.raises(ValueError):
            study_from_dict({**PLATE, "aero": bad})


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
