"""
Tests for loads and design space from a mechanism (assembly.py): URDF
kinematics against finite differences, statics against a hand calculation,
actuator effort limits on the dynamic and contact samples, the governing
case pick, and a swept volume against its analytic size. A two-link arm
written on the fly; well under a minute.
Run: `python3 tests/test_assembly.py`.
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.assembly import (  # noqa: E402
    LoadSweep, Payload, Robot, _greedy_cover, derive_study, governing_cases, intrusion, swept_keepouts,
    sweep_loads,
)

URDF = """<?xml version="1.0"?>
<robot name="two">
  <link name="base"/>
  <link name="link1">
    <inertial><origin xyz="0.1 0 0"/><mass value="1.0"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="3e-3" iyz="0" izz="3e-3"/></inertial>
  </link>
  <link name="link2">
    <inertial><origin xyz="0.1 0 0"/><mass value="0.5"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/></inertial>
    <collision><geometry><mesh filename="package://two/meshes/link2.stl" scale="0.001 0.001 0.001"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-3.1" upper="3.1" effort="5.0" velocity="2.0"/></joint>
  <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/>
    <origin xyz="0.2 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5708" upper="1.5708" effort="0.6" velocity="3.0"/></joint>
</robot>
"""


def _robot():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "meshes"))
    box = trimesh.creation.box(extents=(150.0, 20.0, 20.0))
    box.apply_translation((75.0, 0.0, 0.0))
    box.export(os.path.join(d, "meshes", "link2.stl"))
    path = os.path.join(d, "two.urdf")
    with open(path, "w") as fh:
        fh.write(URDF)
    return Robot.from_urdf(path)


def test_kinematics_match_finite_differences():
    r = _robot()
    q, qd, qdd = {"j1": 0.4, "j2": -0.7}, {"j1": 1.3, "j2": -2.1}, {"j1": 4.0, "j2": 7.0}
    T, w, al, a = r.state(q, qd, qdd)
    h = 1e-4

    def com(t):
        qq = {n: q[n] + qd[n] * t + 0.5 * qdd[n] * t * t for n in q}
        tt = r.fk(qq)["link2"]
        return tt[:3, :3] @ r.bodies["link2"].com + tt[:3, 3]

    fd = (com(h) - 2 * com(0) + com(-h)) / h / h
    rc = T["link2"][:3, :3] @ r.bodies["link2"].com
    an = a["link2"] + np.cross(al["link2"], rc) + np.cross(w["link2"], np.cross(w["link2"], rc))
    assert np.allclose(fd, an, atol=1e-3), (fd, an)


def test_static_wrench_by_hand():
    r = _robot()
    sw = sweep_loads(r, "link1", samples=1, corners=True, speed=0, stall=False, push=0)
    # the home pose is a corner (j1 mid-range 0, j2 mid-range 0)
    i = next(k for k, s in enumerate(sw.states) if abs(s["q"]["j1"]) < 1e-9 and abs(s["q"]["j2"]) < 1e-9)
    f, m = sw.wrenches[i, :3], sw.wrenches[i, 3:]
    # 0.5 kg, 100 mm past j2, about j2's origin
    assert np.allclose(f, [0, 0, -0.5 * 9.81], atol=1e-6), f
    assert np.allclose(m, [0, 0.5 * 9.81 * 100.0, 0], atol=1e-3), m
    assert np.allclose(sw.point, [200.0, 0.0, 100.0])


def test_dynamics_and_contacts_respect_effort():
    r = _robot()
    sw = sweep_loads(r, "link1", payload=Payload(0.05, "link2", (0.15, 0, 0)), samples=300, speed=1.0,
                     ramp=0.01, push=50.0, contact_max=1e6)
    kinds = np.array([k.split(":")[0] for k in sw.kinds])
    # j2's axis is y in link1's frame, and moments are about j2's origin:
    # nothing j2 carries can need more than its 0.6 N·m
    my = np.abs(sw.wrenches[:, 4])
    assert my[kinds == "dynamic"].max() <= 600.0 + 1e-6, my[kinds == "dynamic"].max()
    assert my[kinds == "stall"].max() <= 600.0 + 1e-6
    # bang-bang at ramp 0.01 s asks for 300 rad/s²: the effort cut it back
    scales = [s["accelScale"] for s, k in zip(sw.states, kinds) if k == "dynamic"]
    assert max(scales) < 1.0
    assert my[kinds == "dynamic"].max() > 0.9 * 600.0
    assert {"pose", "dynamic", "push", "stall"} <= set(kinds)


def test_greedy_cover_picks_the_dominant_rows():
    rng = np.random.default_rng(0)
    R = rng.uniform(0, 0.5, size=(200, 50))
    R[17, :25] = 1.0
    R[123, 25:] = 1.0
    chosen, hist, ratio, env = _greedy_cover(R, k=5)
    assert sorted(chosen[:2]) == [17, 123], chosen
    assert hist[-1] >= 0.98 and ratio.min() >= 0.95


def test_stress_pick_on_a_cantilever():
    study = {
        "name": "bar", "designSpace": [{"type": "box", "min": [0, -5, -5], "max": [60, 5, 5]}],
        "preserve": [{"id": "tip", "shape": {"type": "box", "min": [56, -5, -5], "max": [60, 5, 5]}}],
        "supports": [{"shape": {"type": "box", "min": [0, -5, -5], "max": [1, 5, 5]}}],
        "loadCases": [{"name": "x", "loads": [{"region": "tip", "force": [0, 0, -1]}]}],
        "material": "pla", "resolution": {"maxElements": 3000},
    }
    W = np.zeros((4, 6))
    W[0, 0] = 100.0           # axial: low stress
    W[1, 2] = -100.0          # bending: governs
    W[2, 2] = -50.0           # the same, smaller
    W[3, 3] = 100.0           # a little torsion
    sw = LoadSweep("bar", "j", np.zeros(3), W, ["a", "b", "c", "d"], [{}] * 4, region="tip")
    g = governing_cases(sw, study, k=3, max_elements=3000, cycles={"b": 1e6, "d": 50})
    assert g["chosen"][0] == 1, g
    assert g["cases"][0]["cycles"] == 1e6
    assert 2 not in g["chosen"]
    hull = governing_cases(sw, None, k=4, method="hull")
    assert 2 not in hull["chosen"]


def test_swept_volume_of_a_swinging_bar():
    r = _robot()
    # link2's 150 x 20 x 20 mm bar swings +-90 degrees about j2 (y) at (200, 0, 100):
    # a half disc of radius 150 mm (to its far corners), 20 mm thick, plus the
    # bar's 10 mm half-width past the axis at each end of the swing
    box = ((150.0, -20.0, -100.0), (400.0, 20.0, 300.0))
    e = swept_keepouts(r, "link1", box=box, pitch=2.0, clearance=0.0)
    assert [x["id"] for x in e] == ["sweep-link2"], e
    vol = e[0]["volume"]
    exact = (0.5 * np.pi * np.hypot(150.0, 10.0) ** 2 + 2 * 10.0 * 150.0) * 20.0
    assert abs(vol - exact) < 0.03 * exact, (vol, exact)
    assert e[0]["keepOut"]["bore"] is False and e[0]["mesh"].is_watertight
    # a block in the way of the swing is found, one behind j2 is not
    hit = trimesh.creation.box(extents=(20, 20, 20))
    hit.apply_translation((260.0, 0.0, 200.0))
    clear = trimesh.creation.box(extents=(20, 20, 20))
    clear.apply_translation((170.0, 0.0, 100.0))
    assert intrusion(e, hit)["sweep-link2"] > 0.5 * 8000
    assert intrusion(e, clear)["sweep-link2"] == 0


TURNED = URDF.replace('<origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>', '<origin xyz="0 0 0.1" rpy="0 0 1.5708"/><axis xyz="0 0 1"/>')


def test_link_frame_matches_world_frame():
    # j1's frame turned 90 degrees about z, so the link's axes are not the world's
    d = tempfile.mkdtemp()
    path = os.path.join(d, "turned.urdf")
    with open(path, "w") as fh:
        fh.write(TURNED.replace('<collision><geometry><mesh filename="package://two/meshes/link2.stl" '
                                'scale="0.001 0.001 0.001"/></geometry></collision>', ""))
    r = Robot.from_urdf(path)
    T0 = r.fk({})["link1"]
    kw = {"samples": 20, "push": 10.0, "seed": 3}
    pw = np.array([5.0, 180.0, 110.0])                       # mm, world at home
    pl = (np.linalg.inv(T0) @ np.append(pw / 1000.0, 1.0))[:3] * 1000.0
    sw = sweep_loads(r, "link1", point=pw, **kw)
    sl = sweep_loads(r, "link1", point=pl, frame="link", **kw)
    R = T0[:3, :3]
    assert np.allclose(sl.wrenches[:, :3], sw.wrenches[:, :3] @ R, atol=1e-6)
    assert np.allclose(sl.wrenches[:, 3:], sw.wrenches[:, 3:] @ R, atol=1e-4)
    # by hand at home, link frame: j2 200 mm along the link's x, 0.5 kg 100 mm past it
    st = sweep_loads(r, "link1", samples=1, speed=0, stall=False, push=0, frame="link")
    i = next(k for k, s in enumerate(st.states) if abs(s["q"]["j1"]) < 1e-9 and abs(s["q"]["j2"]) < 1e-9)
    assert np.allclose(st.point, [200.0, 0.0, 0.0], atol=1e-6), st.point
    assert np.allclose(st.wrenches[i], [0, 0, -0.5 * 9.81, 0, 0.5 * 9.81 * 100.0, 0], atol=1e-3)
    # the keep-outs follow: link1's frame is world at home moved up 100 mm (the unturned arm)
    r2 = _robot()
    box_w = ((150.0, -20.0, -100.0), (400.0, 20.0, 300.0))
    box_l = ((150.0, -20.0, -200.0), (400.0, 20.0, 200.0))
    ew = swept_keepouts(r2, "link1", box=box_w, pitch=4.0, clearance=0.0)[0]
    el = swept_keepouts(r2, "link1", box=box_l, pitch=4.0, clearance=0.0, frame="link")[0]
    assert abs(ew["volume"] - el["volume"]) < 0.02 * ew["volume"]
    assert np.allclose(el["mesh"].bounds, ew["mesh"].bounds - [0, 0, 100.0], atol=4.0)


BAR = {
    "name": "bar", "designSpace": [{"type": "box", "min": [150, -10, 90], "max": [210, 10, 110]}],
    "preserve": [{"id": "tip", "shape": {"type": "box", "min": [196, -10, 90], "max": [200, 10, 110]}}],
    "supports": [{"shape": {"type": "box", "min": [150, -10, 90], "max": [151, 10, 110]}}],
    "material": "pla",
}


def test_derive_with_fasteners_and_no_load_cases():
    r = _robot()
    # no loadCases at all: the derived cases are the study's only ones
    s, rep = derive_study(r, "link1", BAR, "tip", k=2, method="hull", sweep_space=False,
                          sweep_kw={"samples": 20})
    assert s["loadCases"] and all(c["name"].startswith("sweep") for c in s["loadCases"]), s["loadCases"]
    # fasteners: expanded for the unit solves, without their preload as a 7th
    fs = dict(BAR, fasteners=[{"size": "M3", "seat": [160, 0, 110], "direction": [0, 0, -1], "grip": 20}])
    s, rep = derive_study(r, "link1", fs, "tip", k=2, sweep_space=False, sweep_kw={"samples": 20})
    assert rep["method"] == "stress" and s["fasteners"] == fs["fasteners"]
    from loadpath.spec import study_from_dict
    names = [c.name for c in study_from_dict(s).load_cases]
    assert "preload" in names and len(names) == len(s["loadCases"]) + 1, names


TESTS = [test_kinematics_match_finite_differences, test_static_wrench_by_hand,
         test_dynamics_and_contacts_respect_effort, test_greedy_cover_picks_the_dominant_rows,
         test_stress_pick_on_a_cantilever, test_swept_volume_of_a_swinging_bar,
         test_link_frame_matches_world_frame, test_derive_with_fasteners_and_no_load_cases]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
