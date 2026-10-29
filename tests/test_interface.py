"""
Tests for the agent interface: the JSON Schemas and linter, the CLI and the
MCP server's tool functions (called directly, as the server does). Small
problems only; runs in well under a minute.

Run: `python3 tests/test_interface.py`. With `jsonschema` installed it also
checks the schemas are valid JSON Schema and agree with the linter.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import mcp_server
from loadpath.schema import SCHEMAS, lint
from loadpath.tasks import check_study, example, examples

CASES = sorted((ROOT / "loadpath" / "evals" / "cases").glob("*.json"))
TINY = {  # test_loadpath's LUG, cut short
    "name": "tiny",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.4},
    "minMember": 3.0,
    "maxIterations": 25,
    "resolution": {"pitch": 2.0},
}


def _docs():
    for p in CASES:
        yield p.name, "study", json.loads(p.read_text())
    for e in examples():
        yield e["name"], e["kind"], example(e["name"])


def test_examples_and_cases_lint_clean():
    for name, kind, doc in _docs():
        issues = lint(doc, kind)
        assert not issues, (name, issues)


def test_schemas_are_valid_json_schema():
    try:
        import jsonschema
    except ImportError:
        print("  (jsonschema not installed: skipped)")
        return
    for kind, s in SCHEMAS.items():
        jsonschema.Draft202012Validator.check_schema(s)
    for name, kind, doc in _docs():
        errs = list(jsonschema.Draft202012Validator(SCHEMAS[kind]).iter_errors(doc))
        assert not errs, (name, [e.message for e in errs])


def test_published_schema_files_match():
    for kind, fname in (("study", "study.schema.json"), ("heatExchanger", "heat-exchanger.schema.json"),
                        ("lattice", "lattice.schema.json")):
        on_disk = json.loads((ROOT / "schema" / fname).read_text())
        assert on_disk == json.loads(json.dumps(SCHEMAS[kind])), \
            f"schema/{fname} is stale: python -m loadpath schema {kind} > schema/{fname}"


def test_lint_reports_every_problem_with_suggestions():
    bad = {"designSpace": [{"type": "cube", "min": [0, 0, 0], "max": [1, 1, 1]}],
           "keepout": [], "supports": [{"region": "a", "fix": "pinned"}],
           "loads": [{"region": "a", "force": [0, 1]}],
           "preserve": [{"shape": {"type": "cylinder", "center": [0, 0, 0], "length": 2, "radius": 1}}],
           "goal": {"safetyfactor": 2}, "whatever": 1}
    issues = lint(bad)
    text = json.dumps(issues)
    for want in ("did you mean 'box'", "did you mean 'keepOut'", "did you mean 'r'",
                 "did you mean 'safetyFactor'", "$.supports[0].fix", "$.loads[0].force"):
        assert want in text, (want, issues)
    assert [i for i in issues if i["path"] == "$.whatever"][0]["level"] == "warning"


def test_check_returns_all_errors_at_once_and_parses_good_studies():
    r = check_study({"designSpace": {"type": "box", "min": [0, 0, 0], "max": [1, 1]}, "supports": []})
    assert not r["ok"] and len([i for i in r["issues"] if i["level"] == "error"]) >= 3, r
    r = check_study(json.dumps(TINY))
    assert r["ok"] and r["cells"] > 0 and r["loadCases"] == ["pull"], r


def test_cli_check_schema_examples():
    def run(*a):
        return subprocess.run([sys.executable, "-m", "loadpath", *a], capture_output=True, text=True, cwd=ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.json")
        with open(bad, "w") as fh:
            json.dump({"designSpace": {"type": "box"}}, fh)
        p = run("check", bad)
        assert p.returncode == 2 and json.loads(p.stdout)["ok"] is False, p.stdout
    p = run("schema", "heatExchanger")
    assert json.loads(p.stdout)["title"] == "loadpath heat exchanger"
    p = run("examples")
    assert {e["name"] for e in json.loads(p.stdout)} >= {"shelf-bracket", "heat-exchanger-crossflow"}


def test_size_heat_exchanger_is_quick():
    t0 = time.time()
    r = mcp_server.size_heat_exchanger(example("heat-exchanger-crossflow"))
    assert r["best"] and r["best"]["effectiveness"] > 0, r
    assert time.time() - t0 < 30
    r = mcp_server.size_heat_exchanger({"core": {"type": "box", "min": [0, 0, 0], "max": [9, 9, 9]}})
    assert r["ok"] is False and "duty" in r["error"]


def test_mcp_study_job_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["LOADPATH_HOME"] = tmp
        try:
            r = mcp_server.start({**TINY, "keepout": []})
            assert r["ok"] is False and "keepOut" in r["error"], r
            r = mcp_server.start(TINY)
            assert r["ok"], r
            job = r["jobId"]
            t0 = time.time()
            while True:
                info, pngs = mcp_server.status(job)
                if info["status"] != "running":
                    break
                assert time.time() - t0 < 300, info
                time.sleep(1)
            assert info["status"] == "done", info
            o = info["outcomes"][0]
            assert os.path.exists(o["stl"]) and o["massGrams"] > 0 and "attention" in o, o
            assert pngs and pngs[0][:4] == b"\x89PNG"
            # a restarted server still knows the job
            mcp_server._procs.clear()
            assert mcp_server.status(job)[0]["status"] == "done"
            assert mcp_server.jobs()[0]["jobId"] == job
            try:
                mcp_server.status("0" * 32)
                raise AssertionError("an unknown job should be refused")
            except ValueError:
                pass
        finally:
            os.environ.pop("LOADPATH_HOME", None)


TESTS = [
    test_examples_and_cases_lint_clean,
    test_schemas_are_valid_json_schema,
    test_published_schema_files_match,
    test_lint_reports_every_problem_with_suggestions,
    test_check_returns_all_errors_at_once_and_parses_good_studies,
    test_cli_check_schema_examples,
    test_size_heat_exchanger_is_quick,
    test_mcp_study_job_end_to_end,
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
