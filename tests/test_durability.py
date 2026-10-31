"""
Tests for the durability checks: the fatigue curve and Miner's rule, creep
with temperature, creep rupture, bolt-preload relaxation, the verification
report block and the study fields. Small fixtures, well under a minute.
Run: `python3 tests/test_durability.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.durability import (  # noqa: E402
    DURABILITY, assess, creep_ratio, cycles_to_failure, durability_data, fatigue_strength, rupture_strength,
    strength_retention,
)
from loadpath.spec import study_from_dict  # noqa: E402


def _box(lo, hi):
    b = trimesh.creation.box(extents=np.subtract(hi, lo))
    b.apply_translation(np.add(lo, hi) / 2)
    return b


def _bar(**extra):
    """A 10 x 10 x 40 PETG bar held at its foot and pulled at its top."""
    case = {"name": "pull", "loads": [{"shape": {"type": "box", "min": [-1, -1, 39.9], "max": [11, 11, 41]},
                                       "force": [0, 0, 500]}]}
    case.update(extra.pop("case", {}))
    return {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [10, 10, 40]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [11, 11, 0.1]}}],
            "loadCases": [case], "material": "petg", "goal": {"volumeFraction": 0.5}, **extra}


def test_pla_design_curve_matches_ezeh_susmel():
    """0.1 UTS at 2e6 cycles, inverse slope 5.5, and a life that inverts it."""
    d = DURABILITY["pla"]
    assert abs(fatigue_strength(d, 50.0, 2e6) - 5.0) < 1e-9
    s1e5 = fatigue_strength(d, 50.0, 1e5)
    assert abs(s1e5 / 5.0 - 20 ** (1 / 5.5)) < 1e-9
    assert abs(cycles_to_failure(d, 50.0, s1e5) - 1e5) / 1e5 < 1e-9
    # capped at the static strength, and a stress above it fails at once
    assert fatigue_strength(d, 50.0, 1.0) == 50.0
    assert cycles_to_failure(d, 50.0, 60.0) == 1.0


def test_creep_grows_with_time_and_temperature_and_rupture_falls():
    d = durability_data("petg")
    assert creep_ratio(d, 1000.0) == d.creep_ratio
    assert creep_ratio(d, 10000.0) > creep_ratio(d, 1000.0)
    warm = strength_retention("petg", 60.0)
    assert 0.6 < warm < 0.8
    assert creep_ratio(d, 1000.0, warm) > 1.2 * creep_ratio(d, 1000.0)
    assert rupture_strength(d, 45.0, 0.05) == 45.0
    assert abs(rupture_strength(d, 45.0, 1000.0) - 0.6 * 45.0) < 1e-9
    assert rupture_strength(d, 45.0, 1e12) == 0.25 * 45.0
    assert durability_data("alsi10mg").creep_ratio == 0.0
    # study overrides replace the table and say so
    o = durability_data("petg", {"fatigue_ratio": 0.2})
    assert o.fatigue_ratio == 0.2 and o.quality == "study"


def test_study_fields_parse_and_validate():
    s = study_from_dict(_bar(case={"cycles": 1e6, "hours": 500, "temperature": 40},
                             durability={"temperature": 35, "constrain": True, "safetyFactor": 2}))
    lc = s.load_cases[0]
    assert (lc.cycles, lc.hours, lc.temperature) == (1e6, 500.0, 40.0)
    assert s.durability.constrain and s.durability.safety_factor == 2 and s.durability.temperature == 35
    for bad in ({"case": {"cycles": -1}}, {"case": {"hours": "long"}}, {"durability": {"safetyFactor": 0}}):
        try:
            study_from_dict(_bar(**bad))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    from loadpath.schema import lint
    assert not [i for i in lint(_bar(case={"cycles": 1e6}, durability={"temperature": 40}))
                if i["level"] == "error"]
    assert [i for i in lint(_bar(durability={"temp": 40})) if i["path"] == "$.durability.temp"]


def test_miner_sum_and_safety_factor():
    """Two cyclic cases add damage; the fatigue safety factor is the load
    factor that brings the Miner sum to 1."""
    s = study_from_dict(_bar(case={"cycles": 1e6}))
    s.load_cases.append(type(s.load_cases[0])("lift", s.load_cases[0].loads, cycles=1e5))
    cases = [{"peakStress": 10.0, "deflection": 0.1}, {"peakStress": 15.0, "deflection": 0.15}]
    r = assess(s, cases)
    d = durability_data("petg")
    uts = s.material.strength
    expect = 1e6 / cycles_to_failure(d, uts, 10.0) + 1e5 / cycles_to_failure(d, uts, 15.0)
    assert abs(r["fatigueDamage"] - expect) / expect < 1e-9
    assert abs(r["fatigueSafetyFactor"] - expect ** (-1 / 5.5)) < 1e-9
    scaled = assess(s, [{"peakStress": c["peakStress"] * r["fatigueSafetyFactor"], "deflection": 0}
                        for c in cases])
    assert abs(scaled["fatigueDamage"] - 1.0) < 1e-6
    assert r["creepSafetyFactor"] is None and r["safetyFactor"] == r["fatigueSafetyFactor"]


def test_verify_reports_fatigue_creep_and_layer_bond():
    """A bar pulled a million times: in-plane fatigue far below its static
    safety factor; printed standing up, the layer bond limits it; held warm,
    its deflection creeps."""
    from loadpath.verify import verify_part

    mesh = _box((0, 0, 0), (10, 10, 40))
    flat = verify_part(study_from_dict(_bar(case={"cycles": 1e6})), mesh, max_cells=8000, refine=0)
    dur = flat["durability"]
    assert dur["fatigueSafetyFactor"] < 0.5 * flat["safetyFactor"]
    assert dur["cases"][0]["life"] > 1e6 * dur["fatigueSafetyFactor"] ** 5.5 * 0.99
    up = verify_part(study_from_dict(_bar(case={"cycles": 1e6}, process={"type": "fdm", "buildDirection": "+z"})),
                     mesh, max_cells=8000, refine=0)
    assert up["durability"]["fatigueLimitedBy"] == "layer bond"
    assert up["durability"]["fatigueSafetyFactor"] < dur["fatigueSafetyFactor"]
    held = verify_part(study_from_dict(_bar(case={"hours": 5000}, durability={"temperature": 60})), mesh,
                       max_cells=8000, refine=0)
    row = held["durability"]["cases"][0]
    assert row["creepDeflection"] > 1.5 * held["cases"][0]["deflection"]
    assert held["durability"]["creepSafetyFactor"] < held["safetyFactor"]
    # a plain study gets no durability block
    assert "durability" not in verify_part(study_from_dict(_bar()), mesh, max_cells=8000, refine=0)


def test_fastener_preload_relaxes_more_when_warm():
    base = _bar()
    base["fasteners"] = [{"id": "m4", "size": "M4", "seat": [5, 5, 40], "direction": "-z", "grip": 8,
                          "end": "face"}]
    cool = study_from_dict({**base, "durability": {"temperature": 23}})
    assert cool.joints and cool.joints[0]["id"] == "m4" and "_joints" not in cool.raw
    warm = study_from_dict({**base, "durability": {"temperature": 60, "preloadHours": 8760}})
    cases = [{"peakStress": 5.0, "deflection": 0.05}, {"peakStress": 3.0, "deflection": 0.01}]
    rc, rw = assess(cool, cases), assess(warm, cases)
    jc, jw = rc["joints"][0], rw["joints"][0]
    assert 0.5 < jc["preloadRetention"] < 0.8 and jw["preloadRetention"] < jc["preloadRetention"]
    assert jc["bearingSafetyFactor"] > 1.5
    assert any("m4" in w and "clamp load" in w for w in rw["warnings"])
    # the preload case counts as held for preloadHours
    assert rc["cases"][0]["name"] == "preload" and rc["cases"][0]["hours"] == 8760
    # without a durability block, fasteners are not assessed
    assert "joints" not in assess(study_from_dict(base), cases)


def test_margin_includes_durability_only_when_constrained():
    from loadpath.study import _margin

    report = {"safetyFactor": 4.0, "cases": [{"deflection": 0.1}],
              "durability": {"safetyFactor": 0.75, "target": 1.5, "cases": [{"creepDeflection": 0.3}]}}
    off = study_from_dict(_bar(case={"cycles": 1e6}, goal={"safetyFactor": 2, "maxDeflection": 0.2}))
    on = study_from_dict(_bar(case={"cycles": 1e6}, goal={"safetyFactor": 2, "maxDeflection": 0.2},
                              durability={"constrain": True}))
    assert _margin(off, report) == 2.0
    assert _margin(on, report) == 0.5


def test_filled_grade_reads_its_base_polymer_curve():
    # GRIP's links are "pa12-gf": they used to fall through to the generic
    # FDM (PLA) curve, a third of PA12's fatigue strength
    from loadpath.durability import DURABILITY, durability_data

    class M:
        name = "pa12-gf"
    d = durability_data(M())
    assert d.fatigue_ratio == DURABILITY["pa12"].fatigue_ratio and d.quality == "assumed"
    M.name = "no-such-plastic"
    assert d.fatigue_ratio != durability_data(M()).fatigue_ratio


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


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
