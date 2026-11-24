"""
Printed joints sized from physics (joints.py): threads, hinges, snaps.

Each test holds one physical fact the sizing rests on: the turns' load
sharing reproduces the steel-nut result it was calibrated to, a stress
across the layers is judged against the across strength, a standing
screw's core is weaker than a lying one's, the clamp relaxes as its most
creeping member, the process's tolerance sets the finest pitch, a hand's
over-tightening sizes a hand-turned joint unless the knob slips, a
clevis's middle knuckle is twice an outer one, a hinge loaded along the
build is held by the layers' bond, a snap's strain is the cantilever's.

Run: `python3 tests/test_joints.py` (a few seconds) or pytest.
"""
import math
import sys
import time
import traceback

import numpy as np

from loadpath import joints as J


def test_turns_share_like_a_steel_nut():
    """C is calibrated so a standard steel nut (0.8 d, ~5.5 turns) puts
    about a third of the load on its first turn (Sopwith)."""
    prof = J.thread_profile(12.0, 1.75, (30, 30), 0.0)
    E = 210000.0
    A_s = math.pi * prof["screwMinor"] ** 2
    A_n = math.pi * (9.5 ** 2 - 6.0 ** 2)
    peak, _ = J._share_peak(9.6, 1.75, prof, E, A_s, A_n)
    first = peak / (9.6 / 1.75)
    assert 0.28 < first < 0.40, first
    # longer engagement: the first turn's share keeps falling, but slower than 1/n
    peak2, _ = J._share_peak(19.2, 1.75, prof, E, A_s, A_n)
    assert peak2 / (19.2 / 1.75) < first
    assert peak2 / (19.2 / 1.75) > 0.5 * first


def test_utilization_reads_the_layers():
    st = {"inPlane": 50.0, "across": 20.0, "bondShear": 20.0 / math.sqrt(3)}
    s = np.zeros((3, 3))
    s[2, 2] = 10.0
    assert abs(J._utilization(s, st, (0, 0, 1)) - 0.5) < 1e-9        # across the layers: 10 / 20
    assert abs(J._utilization(s, st, (1, 0, 0)) - 0.2) < 1e-9        # along them: 10 / 50
    assert abs(J._utilization(s, st, None) - 0.2) < 1e-9             # isotropic: von Mises
    sh = np.zeros((3, 3))
    sh[0, 2] = sh[2, 0] = 5.0                                        # shear on the layer plane
    assert abs(J._utilization(sh, st, (0, 0, 1)) - 5.0 / st["bondShear"]) < 1e-9
    c = -s                                                           # compression across: no bond opens
    assert J._utilization(c, st, (0, 0, 1)) < 0.25


def test_profile_gap_and_widths():
    p = J.thread_profile(12.0, 2.5, (45, 45), 0.25)
    assert abs(p["nutMajor"] - p["screwMajor"] - 0.25) < 1e-12
    assert abs(p["nutMinor"] - p["screwMinor"] - 0.25) < 1e-12
    assert abs(p["overlap"] - (p["depth"] - 0.25)) < 1e-12
    # the nut's crest keeps a P/8 flat, the screw's crest is P/8, the teeth never overlap
    assert abs(p["nutTooth"](p["nutMinor"]) - p["pitch"] / 8) < 1e-9
    assert p["screwTooth"](p["nutMinor"]) + p["nutTooth"](p["nutMinor"]) < p["pitch"]


def test_process_gap_along_and_across():
    along = J.process_gap("fdm", 12.0, along=1.0)["gap"]
    across = J.process_gap("fdm", 12.0, along=0.0)["gap"]
    resin = J.process_gap("sla", 12.0, along=1.0)["gap"]
    assert 0.15 < along < across < 0.45, (along, across)
    assert resin < along


def test_finest_pitch_comes_from_tolerance():
    """A pitch whose depth the process's worst case eats fails the
    engagement row on FDM, and passes on resin."""
    kw = dict(clamp=50, material="petg", d=8, pitch=1.0, engage=8, torque=200)
    fdm = J.thread(process="fdm", **kw)
    sla = J.thread(process="sla", **{**kw, "material": "resin"})
    row = {r["id"]: r for r in fdm["checks"]}
    assert not row["engagement"]["pass"] or not row["tooth-print"]["pass"]
    assert {r["id"]: r for r in sla["checks"]}["engagement"]["pass"]


def test_standing_screw_core_is_weaker_than_lying():
    kw = dict(clamp=80, material="petg", process="fdm", d=12, pitch=2.5, engage=12, knob=30,
              axis=(0, 0, 1), nut_build=(0, 0, 1))
    stand = {r["id"]: r for r in J.thread(screw_build=(0, 0, 1), **kw)["checks"]}
    lie = {r["id"]: r for r in J.thread(screw_build=(1, 0, 0), **kw)["checks"]}
    assert lie["screw-core"]["sf"] > stand["screw-core"]["sf"]
    # lying on a flat: its teeth carry the turn's load on less of each turn
    assert lie["screw-strip"]["sf"] < stand["screw-strip"]["sf"]


def test_clamp_relaxes_as_its_most_creeping_member():
    a = J.thread(clamp=100, material="petg", process="fdm", d=12, pitch=2.5, engage=12, torque=500)
    b = J.thread(clamp=100, material="petg", process="fdm", d=12, pitch=2.5, engage=12, torque=500,
                 clamp_material="tpu")
    assert b["preload"]["retention"] < a["preload"]["retention"] < 1.0
    assert b["preload"]["fresh"] > a["preload"]["fresh"] > 100


def test_hand_torque_sizes_the_knob_and_slip_caps_the_overtightening():
    kw = dict(clamp=95, material="petg", process="fdm", d=12, pitch=2.0, engage=12, axis=(0, 0, 1),
              screw_build=(1, 0, 0), nut_build=(0, 0, 1))
    small = J.thread(knob=24, **kw)
    assert not {r["id"]: r for r in small["checks"]}["torque"]["pass"]
    auto = J.thread(knob="auto", **kw)
    assert {r["id"]: r for r in auto["checks"]}["torque"]["pass"]
    assert auto["knob"] > 24
    # a strong hand on a smooth thread puts in several times the snug clamp ...
    assert auto["preload"]["most"] > 3.0 * auto["preload"]["snug"]
    slip = J.thread(knob="auto", slip=1.3, **kw)
    # ... a knob that slips at 1.3x snug caps it near 1.3 x (K / K at the smooth end)
    assert slip["preload"]["most"] < 2.2 * slip["preload"]["snug"]
    sf = lambda d: {r["id"]: r["sf"] for r in d["checks"]}["screw-core"]
    assert sf(slip) > 1.8 * sf(auto)


def test_thread_search_passes_and_sizes_the_nut_wall():
    t = J.thread(clamp=60, material="petg", process="fdm", knob="auto", slip=1.3, axis=(0, 0, 1),
                 screw_build=(1, 0, 0), nut_build=(0, 0, 1))
    assert t["pass"], [(r["id"], r["sf"]) for r in t["checks"] if not r["pass"]]
    assert t["boss"] > t["d"]
    # the nut's wall measured thinner than it needs: the burst fails
    thin = J.check_joint({"type": "thread", "clamp": 60, "knob": "auto", "slip": 1.3, "screwBuild": [1, 0, 0],
                          "axis": [0, 0, 1]}, {"boss": t["profile"]["nutMajor"] * 2 + 0.3}, "petg", "fdm", (0, 0, 1))
    assert not {r["id"]: r for r in thin["checks"]}["nut-wall"]["pass"]


def test_hinge_clevis_and_layers():
    # printed with the pin standing: the knuckles are plates in the layers' plane
    h = J.hinge(load=250, material="petg", process="fdm", d=6, width=14, ligament=3, axis=(1, 0, 0),
                load_dir=(0, 0, 1), build=(1, 0, 0))
    assert abs(h["knuckle"] - 2 * h["outerKnuckle"]) < 1e-9          # the middle knuckle carries twice
    rows = {r["id"]: r for r in h["checks"]}
    # the pin lying, loaded along the build: the knuckles' shear-out is the layers' bond
    v = J.hinge(load=250, material="petg", process="fdm", d=6, width=14, ligament=3, axis=(1, 0, 0),
                load_dir=(0, 0, 1), build=(0, 0, 1))
    vrows = {r["id"]: r for r in v["checks"]}
    assert vrows["inner-shearout"]["sf"] < rows["inner-shearout"]["sf"]
    # a steel pin does not bend where a printed one does
    s = J.hinge(load=250, material="petg", process="fdm", d=6, width=14, ligament=3, pin="304")
    assert {r["id"]: r for r in s["checks"]}["pin-bending"]["sf"] > 3 * rows["pin-bending"]["sf"]
    # pressed: the bore prints under the pin (drawn wider by the hole's bias along the build)
    from loadpath.fits import HOLES

    assert h["press"] + HOLES["fdm"]["bias"] < 6.0 < h["press"]


def test_print_in_place_hinge_prints():
    h = J.hinge(load=100, material="petg", process="fdm", kind="print-in-place", axis=(1, 0, 0),
                load_dir=(0, 0, 1), build=(0, 0, 1))
    rows = {r["id"]: r for r in h["checks"]}
    assert h["pass"], [(k, r["sf"]) for k, r in rows.items() if not r["pass"]]
    assert rows["bore-roof"]["pass"] and rows["pin-bridge"]["pass"]
    assert h["gap"] >= J.process_gap("fdm", h["d"], 0.0)["gap"] - 1e-9


def test_snap_matches_the_cantilever():
    from loadpath.shells import snap_allowable, snap_cantilever

    s = J.snap(hold=15, material="petg", process="fdm", build=(0, 0, 1), length_dir=(1, 0, 0),
               deflect_dir=(0, 0, 1))
    assert s["pass"]
    ref = snap_cantilever(s["L"], s["t"], s["b"], s["undercut"], 1500.0)
    assert abs(ref["strain"] - s["strain"]) < 1e-3 * s["strain"]
    assert s["strain"] <= snap_allowable("petg", "fdm")["strain"] + 1e-12
    # a beam along the build bends across the layers: a smaller allowance, a longer beam
    z = J.snap(hold=15, material="petg", process="fdm", build=(0, 0, 1), length_dir=(0, 0, 1),
               deflect_dir=(1, 0, 0), t=s["t"])
    assert z["allowable"] < s["allowable"] and z["L"] > s["L"]


TESTS = [test_turns_share_like_a_steel_nut, test_utilization_reads_the_layers, test_profile_gap_and_widths,
         test_process_gap_along_and_across, test_finest_pitch_comes_from_tolerance,
         test_standing_screw_core_is_weaker_than_lying, test_clamp_relaxes_as_its_most_creeping_member,
         test_hand_torque_sizes_the_knob_and_slip_caps_the_overtightening,
         test_thread_search_passes_and_sizes_the_nut_wall, test_hinge_clevis_and_layers,
         test_print_in_place_hinge_prints, test_snap_matches_the_cantilever]


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
