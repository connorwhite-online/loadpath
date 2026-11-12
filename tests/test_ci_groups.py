"""
Every test file runs somewhere in CI. The tests job (.github/workflows/
tests.yml) runs the files its matrix groups list, so a file left out of
every group is never run, and on 2026-11-12 57 were, three of them red for
days. This holds every tests/ file to a place in the workflows, and every
file a group runs as plain Python (`files:`) to having a __main__ (one
without runs no test and passes).

Run: `python3 tests/test_ci_groups.py` (a second).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
# helpers the tests import, not tests of their own
NOT_TESTS = {"_eval_worker.py"}


def _listed():
    """{file name: [(workflow, key)]} for every tests/*.py a workflow names."""
    out = {}
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        key = None
        for line in wf.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = re.match(r"\s*-?\s*([A-Za-z_]+):", line)
            if m:
                key = m.group(1)
            for f in re.findall(r"tests/([A-Za-z0-9_]+\.py)", line):
                out.setdefault(f, []).append((wf.name, key))
    return out


def test_every_test_file_is_in_a_ci_group():
    listed = _listed()
    files = sorted(p.name for p in (ROOT / "tests").glob("*.py") if p.name not in NOT_TESTS)
    missing = [f for f in files if f not in listed]
    assert not missing, ("in no CI group (add each to a group in .github/workflows/tests.yml that stays "
                         f"under ~20 minutes): {missing}")
    gone = sorted(f for f in listed if not (ROOT / "tests" / f).exists())
    assert not gone, f"the workflows name test files that do not exist: {gone}"


def test_plain_python_files_run_their_tests():
    listed = _listed()
    silent = sorted(f for f, where in listed.items() if any(k == "files" for _, k in where)
                    and "__main__" not in (ROOT / "tests" / f).read_text())
    assert not silent, f"run as `python FILE` but have no __main__, so no test runs: {silent} (list under pytest:)"


def test_a_file_runs_in_one_group_only():
    twice = {f: w for f, w in _listed().items() if len([k for _, k in w if k in ("files", "pytest")]) > 1}
    assert not twice, f"listed in more than one group: {twice}"


TESTS = [test_every_test_file_is_in_a_ci_group, test_plain_python_files_run_their_tests,
         test_a_file_runs_in_one_group_only]


def main():
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    if failures:
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
