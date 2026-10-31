"""
The tutorials in docs/tutorials/ as tests, so the docs can't drift from the
code. For each tutorial folder: copy it to a scratch folder, run every
command in its README's ```bash blocks there in quick mode (LOADPATH_QUICK=1:
coarse grid, see loadpath/quick.py; no renders), then hold the outputs to
the folder's expect.json:

    {"quick": [{"file": "out/report.json", "value": "r['outcomes'][r['best']]['mass']",
                "min": 30, "max": 60, "what": "best outcome's mass, g"}, ...],
     "full":  [...],                     # the same, for the numbers the README quotes
     "files": ["out/*.stl"]}             # globs that must match something

`value` is a Python expression over `r` (the file's JSON). Commands in
```sh or ```text blocks are shown, not run (a browser editor, a slow sweep).

    python tests/test_tutorials.py                  # every tutorial, quick
    python tests/test_tutorials.py fdm-bracket      # one
    python tests/test_tutorials.py --full [names]   # full grid, the README's numbers (slow)
    python tests/test_tutorials.py --keep DIR       # leave the runs in DIR
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "tutorials"

_FENCE = re.compile(r"^```bash\s*\n(.*?)^```", re.S | re.M)


def tutorials():
    return sorted(p.name for p in TUTORIALS.iterdir() if (p / "README.md").exists() and (p / "expect.json").exists())


def commands(name):
    """The README's ```bash commands, one per entry (backslash continuations
    joined, comments dropped)."""
    text = (TUTORIALS / name / "README.md").read_text()
    out = []
    for block in _FENCE.findall(text):
        cur = ""
        for line in block.splitlines():
            line = re.sub(r"\s+#.*$", "", line) if not line.lstrip().startswith("#") else ""
            if not line.strip():
                continue
            if line.rstrip().endswith("\\"):
                cur += line.rstrip()[:-1] + " "
                continue
            out.append((cur + line).strip())
            cur = ""
    return out


def _value(expr, r):
    return eval(expr, {"__builtins__": {"min": min, "max": max, "len": len, "sum": sum, "abs": abs,
                                        "any": any, "all": all, "round": round}}, {"r": r})


def check(name, work, tier):
    exp = json.loads((TUTORIALS / name / "expect.json").read_text())
    failures = []
    for pat in exp.get("files", []):
        if not glob.glob(os.path.join(work, pat)):
            failures.append(f"no file matches {pat}")
    for c in exp.get(tier, []):
        f = os.path.join(work, c["file"])
        try:
            with open(f) as fh:
                r = json.load(fh)
            v = _value(c["value"], r)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{c.get('what', c['value'])}: {type(exc).__name__}: {exc}")
            continue
        ok = True
        if "equals" in c:
            ok = v == c["equals"]
        if "min" in c:
            ok = ok and v is not None and v >= c["min"]
        if "max" in c:
            ok = ok and v is not None and v <= c["max"]
        line = f"{c.get('what', c['value'])} = {v!r}"
        print(("  ok   " if ok else "  FAIL ") + line + f"  (want {_want(c)})", flush=True)
        if not ok:
            failures.append(line)
    return failures


def _want(c):
    if "equals" in c:
        return f"== {c['equals']!r}"
    return " and ".join(s for s in ((f">= {c['min']}" if "min" in c else ""),
                                    (f"<= {c['max']}" if "max" in c else "")) if s)


def run(name, tier="quick", keep=None):
    src = TUTORIALS / name
    base = keep or tempfile.mkdtemp(prefix="lp-tutorial-")
    work = os.path.join(base, name)
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("images", "*.png"))
    env = {**os.environ, "LOADPATH_NO_HINT": "1", "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    if tier == "quick":
        env |= {"LOADPATH_QUICK": "1", "LOADPATH_NO_RENDER": "1"}
    failures = []
    t0 = time.time()
    for cmd in commands(name):
        if cmd.startswith("loadpath "):
            cmd = f"{sys.executable} -m loadpath " + cmd[len("loadpath "):]
        elif cmd.startswith("python "):
            cmd = f"{sys.executable} " + cmd[len("python "):]
        print(f"  $ {cmd}", flush=True)
        t = time.time()
        p = subprocess.run(cmd, shell=True, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True)
        print(f"    ({time.time() - t:.0f} s, exit {p.returncode})", flush=True)
        if p.returncode != 0:
            failures.append(f"`{cmd}` exited {p.returncode}:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
            break
    if not failures:
        failures = check(name, work, tier)
    if keep is None:
        shutil.rmtree(base, ignore_errors=True)
    return failures, time.time() - t0


def main():
    args = sys.argv[1:]
    tier = "full" if "--full" in args else "quick"
    keep = None
    if "--keep" in args:
        keep = os.path.abspath(args[args.index("--keep") + 1])
        args = [a for i, a in enumerate(args) if a != "--keep" and (i == 0 or args[i - 1] != "--keep")]
    names = [a for a in args if not a.startswith("--")] or tutorials()
    bad = 0
    for name in names:
        print(f"== {name} ({tier})", flush=True)
        failures, sec = run(name, tier, keep)
        if failures:
            bad += 1
            print(f"FAIL {name} ({sec:.0f} s)")
            for f in failures:
                print("  " + f.replace("\n", "\n  "))
        else:
            print(f"PASS {name} ({sec:.0f} s)")
    if bad:
        print(f"\n{bad}/{len(names)} FAILED")
        sys.exit(1)
    print(f"\nall {len(names)} passed")


if __name__ == "__main__":
    main()
