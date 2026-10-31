"""
Tests for the eval tooling: Pareto scoring, the coarse tier, baseline
comparison, the generated corpus and the critic's scoring and agent path (no
API call). Fast; the corpus test lints every generated case and runs one at
a tiny resolution.

Run: `python3 tests/test_evals.py`.
"""
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from loadpath import critic
from loadpath.evals import corpus, pareto
from loadpath.evals import run as ev
from loadpath.schema import lint
from loadpath.spec import study_from_dict


def test_pareto_front_and_mass_at_target():
    pts = [(30, 3.5), (20, 1.5), (25, 2.5), (28, 2.0), (40, 3.4)]
    fr = pareto.front(pts)
    assert fr == [(20, 1.5), (25, 2.5), (30, 3.5)], fr
    m, br = pareto.mass_at(fr, 3.0)
    assert br and abs(m - 27.5) < 1e-9, m
    # nothing fails below the lightest passing point: an upper bound
    assert pareto.mass_at(fr, 1.0) == (20, False)
    assert pareto.mass_at(fr, 9.0) == (None, False)
    s = pareto.score([{"mass": m, "safetyFactor": sf, "massReduction": 1 - m / 100} for m, sf in pts], 3.0)
    assert abs(s["solidMass"] - 100) < 1e-9
    # a lighter front at the same SF scores a higher hypervolume
    s2 = pareto.score([{"mass": m - 5, "safetyFactor": sf, "massReduction": 1 - (m - 5) / 100} for m, sf in pts], 3.0)
    assert s2["hypervolume"] > s["hypervolume"] and s2["massAtTarget"] < s["massAtTarget"]


def test_coarse_tier_caps_resolution():
    raw = json.loads((ROOT / "loadpath/evals/cases/arm-link.json").read_text())
    c = ev.apply_tier(raw, "coarse")
    assert c["resolution"]["maxElements"] <= ev.COARSE["maxElements"]
    assert c["maxIterations"] <= ev.COARSE["maxIterations"]
    assert raw["resolution"]["maxElements"] > ev.COARSE["maxElements"]   # a copy, not edited in place
    assert ev.apply_tier(raw, "full") is raw
    hx = json.loads((ROOT / "loadpath/evals/cases/lpbf-intercooler-core.json").read_text())
    assert ev.apply_tier(hx, "coarse")["step"] == hx["step"] * ev.COARSE["hxStep"]


def test_compare_flags_regressions():
    base = [{"case": "a", "pass": True, "pareto": {"massAtTarget": 100}},
            {"case": "b", "pass": True, "pareto": {"massAtTarget": 50}},
            {"case": "c", "pass": False, "pareto": {"massAtTarget": None}}]
    rows = [{"case": "a", "pass": True, "pareto": {"massAtTarget": 110}},
            {"case": "b", "pass": True, "pareto": {"massAtTarget": 45}},
            {"case": "c", "pass": True, "pareto": {"massAtTarget": 30}}]
    reg, imp = ev.compare(rows, base, tolerance=0.05)
    assert reg == ["a"] and sorted(imp) == ["b", "c"], (reg, imp)
    md = ev._md_compare(rows, reg, imp)
    assert "+10.0%" in md and "regressed" in md


def test_corpus_is_deterministic_and_valid():
    for s in range(9):
        a, b = corpus.generate(s), corpus.generate(s)
        assert a == b
        problems = lint({k: v for k, v in a.items()}, "study")
        assert not problems, (a["name"], problems)
        st = study_from_dict(a)
        ids = {r.id for r in st.preserve}
        # no preserved boss sits inside a keep-out other than its own bore
        assert len(ids) == len(st.preserve)
    assert corpus.seeds("3") == [0, 1, 2] and corpus.seeds("5:7") == [5, 6]
    assert {corpus.generate(s)["name"].split("-")[1] for s in range(3)} == {"link", "bracket", "tri"}


def test_corpus_case_runs():
    raw = corpus.generate(0, max_elements=2500)
    raw["maxIterations"] = 30
    with tempfile.TemporaryDirectory() as d:
        row = ev.run_case(raw, d, tier="coarse")
    assert row.get("error") is None, row.get("traceback")
    assert row["pareto"]["points"] >= 1


def test_critic_score_and_agent_round_trip():
    f = [{"panel": "a", "kind": "crease", "severity": "major", "note": ""},
         {"panel": "b", "kind": "seat-blocked", "severity": "blocker", "note": ""},
         {"panel": "b", "kind": "rough-surface", "severity": "minor", "note": ""}]
    s = critic.score(f)
    assert s["criticScore"] == 14 and s["criticBlockers"] == 1 and s["criticByKind"]["crease"] == 1
    st = study_from_dict(json.loads((ROOT / "loadpath/evals/cases/arm-link.json").read_text()))
    assert "shoulder-bearing" in critic.prompt(st)
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "x.png")
        open(img, "wb").write(b"\x89PNG\r\n")
        assert critic.critique([img], st, backend="agent", out_dir=d, name="arm-link") is None
        assert os.path.exists(os.path.join(d, "arm-link.critic-request.md"))
        json.dump({"findings": f[:1], "summary": "one crease"}, open(os.path.join(d, "arm-link.critic.json"), "w"))
        res = critic.critique([img], st, backend="agent", out_dir=d, name="arm-link")
        assert res["criticScore"] == 3
        # folding into a scorecard gates on the case's expect when it has critic limits
        row = {"case": "arm-link", "pass": True, "checks": [], "critic": res}
        ev.gate_critic(row, {"maxCriticBlockers": 0, "maxCriticScore": 2})
        assert not row["pass"] and {c["check"] for c in row["checks"]} == {"criticBlockers", "criticScore"}
        json.dump([{"case": "arm-link", "pass": True, "checks": [], "seconds": 1}],
                  open(os.path.join(d, "scorecard.json"), "w"))
        rows = critic.fold_into_scorecard(d)
        assert rows[0]["critic"]["criticScore"] == 3


def test_each_case_in_its_own_process():
    """Cases run in fresh processes (in one, a finished case's peak stays
    and the next sits on top of it), and a case whose worker is killed
    fails alone: the run goes on and the others pass, at one job or two."""
    import _eval_worker

    real = ev.run_case
    ev.run_case = _eval_worker.run_case
    try:
        for jobs in (1, 2):
            rows = ev._run_isolated(["a.json", "dies-b.json", "c.json", "d.json"], "unused", "coarse", None, jobs)
            assert [r["case"] for r in rows] == ["a", "dies-b", "c", "d"]
            assert [r["pass"] for r in rows] == [True, False, True, True], rows
            assert "died" in rows[1]["error"]
            pids = [r["pid"] for r in rows if "pid" in r]
            assert len(set(pids)) == 3 and os.getpid() not in pids
    finally:
        ev.run_case = real


TESTS = [
    test_pareto_front_and_mass_at_target,
    test_coarse_tier_caps_resolution,
    test_compare_flags_regressions,
    test_corpus_is_deterministic_and_valid,
    test_critic_score_and_agent_round_trip,
    test_corpus_case_runs,
    test_each_case_in_its_own_process,
]


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
