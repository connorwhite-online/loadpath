"""
Tests for the design memory (loadpath/recall.py): a rerun of a study replays
what the last run found, and a rescaled study starts where the stored
margins say the answer is.

Run: `python3 tests/test_recall.py`.
"""
import copy
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loadpath.recall as recall
from loadpath.spec import study_from_dict
from loadpath.study import run_study

LUG = {
    "name": "lug",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                         "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                        "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
    "material": "pa12",
    "goal": {"safetyFactor": 3.0, "volumeFraction": 0.5},
    "symmetry": ["z"],
    "resolution": {"pitch": 2.0},
    "maxIterations": 25,
}


def _run(raw):
    r = run_study(study_from_dict(copy.deepcopy(raw)))
    return r, r.outcomes[r.best]


def test_off_by_default():
    os.environ.pop("LOADPATH_RECALL", None)
    assert not recall.enabled()
    r, _ = _run({**LUG, "goal": {"volumeFraction": 0.3}})
    assert "recall" not in r.domain


def test_rerun_replays_and_rescales():
    one = {**LUG, "goal": {"volumeFraction": 0.3}}
    with tempfile.TemporaryDirectory() as d:
        os.environ["LOADPATH_RECALL"] = d
        try:
            first, a = _run(one)
            assert first.domain["recall"]["replayed"] == 0 and a.optimizer["iterations"] > 0

            # the same study again: replayed, no optimizing, the same part
            again, b = _run(one)
            assert again.domain["recall"]["replayed"] == 1 and b.optimizer["iterations"] == 0
            assert b.report["start"] == "recalled 0.300"
            assert abs(b.report["mass"] - a.report["mass"]) < 1e-6 * a.report["mass"]
            assert abs(b.report["safetyFactor"] - a.report["safetyFactor"]) < 1e-6 * a.report["safetyFactor"]

            # a 20% heavier load: the same design, its safety factor / 1.2,
            # and the stored margin predicted it before anything ran
            heavy = copy.deepcopy(one)
            heavy["loadCases"][0]["loads"][0]["force"] = [0, -96, 0]
            heavy["goal"] = {"safetyFactor": 1.0, "volumeFraction": 0.3}
            st = study_from_dict(heavy)
            from loadpath.domain import build_domain
            from loadpath.simp import filter_radius
            from loadpath.study import _making
            dom = build_domain(st)
            mem = recall.Recall.open(st, dom, filter_radius(dom, st.min_member), _making(st))
            [p] = mem.predictions()
            want = a.report["safetyFactor"] / 1.2
            assert p.exact and abs(p.margin - want) < 1e-9 * want
            r, c = _run({**heavy, "goal": {"volumeFraction": 0.3}})
            assert c.optimizer["iterations"] == 0
            assert abs(c.report["safetyFactor"] - want) < 1e-6 * want, (c.report["safetyFactor"], want)

            # a code change retires the replay but keeps the starting design
            recall._CODE = "another version"
            r, e = _run(one)
            assert r.domain["recall"] == {"replayed": 0, "warm": 1, "confirmsSkipped": 0}
            assert e.report["start"] == "warm from recalled 0.300" and e.optimizer["iterations"] > 0
        finally:
            recall._CODE = None
            os.environ.pop("LOADPATH_RECALL", None)


def test_search_starts_at_the_predicted_fraction():
    """Stored readings, rescaled: start at the lightest predicted pass; an
    unconfirmed one is discounted by the worst confirm/analysis ratio seen."""
    class Mat:
        strength, E = 100.0, 1000.0

    class Goal:
        safety_factor, max_deflection = 2.0, None

    class St:
        material, goal, durability = Mat, Goal, None

    m = recall.Recall.__new__(recall.Recall)
    m.study, m.scale, m.keys = St, 1.0, {"loads": "L", "model": "M"}
    m.index = [{"vf": v, "sf": sf, "defl": 1.0, "loads": "L", "model": "M", "loadScale": 1.0,
                "strength": 100.0, "E": 1000.0, "time": 0.0, **({"confirm": c} if c else {})}
               for v, sf, c in [(0.1, 1.6, None), (0.15, 2.3, {"safetyFactor": 2.07, "cases": []}),
                                (0.2, 3.0, None), (0.4, 6.0, None)]]
    vf, why = m.first_fraction(0.3)
    assert vf == 0.15 and "recalled 0.150" in why, (vf, why)   # 2.07 / 2 = 1.035 passes
    # a stronger material: 0.1 now reads 1.6 * 1.5 / 2 = 1.2 analysis, 1.08 discounted (0.9)
    Mat.strength = 150.0
    vf, why = m.first_fraction(0.3)
    assert vf == 0.1, (vf, why)


TESTS = [test_off_by_default, test_rerun_replays_and_rescales, test_search_starts_at_the_predicted_fraction]


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
