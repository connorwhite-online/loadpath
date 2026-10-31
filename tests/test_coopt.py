"""
Tests for co-optimizing a mechanism's links (coopt.py): mass properties of
a swapped part against trimesh and a hand calculation, joint torque and
payload capacity against statics on a two-link arm, and the loop's order,
tolerance and resume on stand-in derive and solve steps. Seconds.
Run: `python3 tests/test_coopt.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.assembly import Payload  # noqa: E402
from loadpath.coopt import ArmModel, MassProps, co_optimize, joint_demand, load_change, mass_props  # noqa: E402
from test_assembly import _robot  # noqa: E402


def _box(ext, at):
    b = trimesh.creation.box(extents=ext)
    b.apply_translation(at)
    return b


def test_mass_props_match_a_box():
    b = _box((100.0, 20.0, 10.0), (50.0, 0.0, 0.0))
    p = mass_props(b, 1.0)                       # 20 cm³ at 1 g/cm³
    assert abs(p.mass - 0.02) < 1e-9
    assert np.allclose(p.com, [0.05, 0.0, 0.0])
    # a box about its centre: m (b² + c²) / 12
    assert abs(p.inertia[2, 2] - 0.02 * (0.1 ** 2 + 0.02 ** 2) / 12) < 1e-12


def test_add_then_subtract_is_identity():
    a = MassProps(1.0, np.array([0.1, 0.0, 0.0]), np.diag([1e-4, 3e-3, 3e-3]))
    b = mass_props(_box((40.0, 20.0, 20.0), (150.0, 0.0, 0.0)), 1.22)
    c = (a + b) - b
    assert abs(c.mass - a.mass) < 1e-9 and np.allclose(c.com, a.com) and np.allclose(c.inertia, a.inertia, atol=1e-9)
    # two point-ish halves: the combined centre of mass is the weighted mean
    s = a + b
    assert np.allclose(s.com, (a.mass * a.com + b.mass * b.com) / s.mass)


def test_swapping_a_part_moves_the_body():
    r = _robot()
    old = _box((150.0, 20.0, 20.0), (75.0, 0.0, 0.0))     # link2's part, link frame = world at home + j2's origin
    T = r.fk({})["link2"]
    frame = np.eye(4)
    frame[:3, 3] = T[:3, 3] * 1000.0                       # part drawn in link2's frame
    arm = ArmModel(r, {"link2": [(old, 1.0, frame)]})
    hw = arm.hardware["link2"].mass
    assert abs(hw - (0.5 - 0.06)) < 1e-6                  # 60 cm³ at 1 g/cm³ out of 0.5 kg
    arm.set_part("link2", _box((75.0, 20.0, 20.0), (37.5, 0.0, 0.0)), 1.0, frame)   # half as long
    b = r.bodies["link2"]
    assert abs(b.mass - (hw + 0.03)) < 1e-6
    assert b.com[0] < 0.1                                  # the centre of mass moved in toward the joint
    assert abs(arm.printed_kg("link2") - 0.03) < 1e-6


def test_joint_demand_matches_statics():
    r = _robot()
    pay = Payload(0.2, "link2", (0.2, 0.0, 0.0))
    d = joint_demand(r, pay, ramp=None, samples=200)
    # worst still pose for j2 is link2 level: (0.5 kg * 0.1 m + p * 0.2 m) g
    g = 9.81
    assert abs(d["joints"]["j2"]["still"]["peakNm"] - (0.5 * 0.1 + 0.2 * 0.2) * g) < 1e-3
    cap = (0.6 / g - 0.05) / 0.2                          # effort 0.6 N·m
    assert abs(d["joints"]["j2"]["still"]["capacityKg"] - cap) < 1e-3
    assert d["capacity"]["still"]["limitedBy"] == "j2"
    # a counterbalance on j2 takes the arm's own weight, not the payload's
    d2 = joint_demand(r, pay, ramp=None, samples=200, balanced=("j2",))
    assert abs(d2["joints"]["j2"]["still"]["capacityKg"] - 0.6 / g / 0.2) < 1e-3
    # moving needs more than holding still
    d3 = joint_demand(r, pay, ramp=0.3, samples=200)
    assert d3["joints"]["j2"]["moving"]["peakNm"] > d3["joints"]["j2"]["still"]["peakNm"]


def test_load_change():
    a = [{"loads": [{"region": "r", "force": [10.0, 0, 0], "moment": [0, 100.0, 0]}]}]
    b = [{"loads": [{"region": "r", "force": [9.0, 0, 0], "moment": [0, 105.0, 0]}]}]
    assert abs(load_change(a, b, "r") - 0.1) < 1e-9


def test_loop_goes_tip_first_and_stops_when_loads_settle():
    r = _robot()
    arm = ArmModel(r, {})
    calls, solved = [], []
    # a toy chain: link1's load follows link2's mass; link2's is fixed
    def derive(link):
        calls.append(link)
        f = 10.0 if link == "link2" else 10.0 + 100.0 * r.bodies["link2"].mass
        return [{"loads": [{"region": "r", "force": [f, 0, 0], "moment": [0, 0, 0]}]}], {"region": "r"}

    def solve(link, cases):
        solved.append(link)
        f = cases[0]["loads"][0]["force"][0]
        part = _box((10.0, 10.0, 10.0 * f), (0.0, 0.0, 0.0))     # mass proportional to the load
        return [(part, 1.0, None)], {"g": f}

    state = co_optimize(arm, ["link2", "link1"], derive, solve, passes=4, tol=0.02, log=lambda s: None)
    assert solved[:2] == ["link2", "link1"]
    assert state["converged"]
    assert solved.count("link2") == 1                      # its loads never moved
    # resume: a finished state does nothing more
    n = len(solved)
    co_optimize(arm, ["link2", "link1"], derive, solve, passes=4, state=state, log=lambda s: None)
    assert len(solved) == n


def test_a_passing_part_is_not_traded_for_a_heavier_one():
    r = _robot()
    arm = ArmModel(r, {})
    load = {"f": 10.0}

    def derive(link):
        return [{"loads": [{"region": "r", "force": [load["f"], 0, 0], "moment": [0, 0, 0]}]}], {"region": "r"}

    def solve(link, cases):
        return [(_box((10.0, 10.0, 10.0), (0, 0, 0)), 1.0, None)], {"g": 5.0}

    def verify(link, cases):
        return {"g": 1.0, "passes": True}

    state = co_optimize(arm, ["link2"], derive, solve, verify=verify, passes=1, log=lambda s: None)
    load["f"] = 8.0                                         # lighter loads: regrown, but the old part is lighter
    state = co_optimize(arm, ["link2"], derive, solve, verify=verify, passes=2, state={**state, "pass": 0,
                        "done": {}, "converged": False, "changed": {}}, log=lambda s: None)
    assert "heavier than the current part; kept" in state["history"][-2]["result"] or \
        "heavier than the current part; kept" in state["history"][-1]["result"]
    assert state["links"]["link2"]["cases"][0]["loads"][0]["force"][0] == 8.0


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"ok   {name} ({time.time() - t:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
