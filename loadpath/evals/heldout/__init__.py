"""
Held-out parts: a cold benchmark of first-try success.

The eval cases (evals/cases) and the briefs (evals/briefs) are the parts
loadpath was developed against: every one has been run many times, and
defaults, thresholds and fixes were chosen while watching them. A number
that only ever improves on the parts it was tuned on says nothing about
the next part a stranger brings.

The cases here are that next part: ordinary printed parts (a corner
bracket, a hinge leaf, a drawer pull, a caster fork, a C-clamp, ...)
written once, from the agent guide alone, by someone who had not run them.
Each is a brief in plain words and the study a careful first reading of it
gives ("gold", the same layout as evals/briefs). They are run as written.

The rule that makes it a benchmark: **nothing is tuned against this set.**
A case is never edited to make it pass, no default or threshold is chosen
by its score here, and a fix is kept only if it is right for a reason that
does not mention these parts. When a case turns out to be physically wrong
as written (see `note` in the case) it stays, as a probe of whether the
tool says so. New cases may be added; old ones are not changed. README.md
has the history of every run.

    python -m loadpath.evals.heldout list
    python -m loadpath.evals.heldout run --out DIR [--jobs 2] [--time-limit 40] [ids...]
    python -m loadpath.evals.heldout table DIR        # scorecard.json / scorecard.md from a run folder

First-try success is strict: `check` passes with no warning, and the run
delivers a part that passes (report.deliver.passes) without the agent
changing anything. The scorecard records each stage on its own, because a
warning the tool raises against its own geometry and a part that comes out
in two pieces are different failures.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")


def cases(ids=None) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(CASES, "*.json"))):
        c = json.load(open(p))
        if not ids or c["id"] in ids:
            out.append(c)
    return out


def _run_one(c, out, time_limit, threads):
    d = os.path.join(out, c["id"])
    os.makedirs(d, exist_ok=True)
    json.dump(c["gold"], open(os.path.join(d, "study.json"), "w"), indent=1)
    env = dict(os.environ, LOADPATH_THREADS=str(threads))
    chk = subprocess.run([sys.executable, "-m", "loadpath", "check", "study.json"], cwd=d, env=env,
                         capture_output=True, text=True)
    open(os.path.join(d, "check.json"), "w").write(chk.stdout)
    open(os.path.join(d, "check.err"), "w").write(chk.stderr)
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as lo, open(os.path.join(d, "run.err"), "w") as le:
        rc = subprocess.run([sys.executable, "-m", "loadpath", "run", "study.json", "--out", "out",
                             "--time-limit", str(time_limit)], cwd=d, env=env, stdout=lo, stderr=le).returncode
    json.dump({"returncode": rc, "seconds": round(time.time() - t0, 1)}, open(os.path.join(d, "ran.json"), "w"))
    return c["id"]


def score(c, d) -> dict:
    """One case's row from its run folder."""
    row = {"id": c["id"], "tags": c.get("tags", [])}
    try:
        chk = json.load(open(os.path.join(d, "check.json")))
        row["check"] = {"ok": bool(chk.get("ok")), "warnings": len(chk.get("warnings", [])),
                        "first": (chk.get("warnings") or [chk.get("error", "")])[0][:160]}
    except (OSError, ValueError):
        row["check"] = {"ok": False, "warnings": None, "first": "check wrote no report"}
    try:
        ran = json.load(open(os.path.join(d, "ran.json")))
        row["seconds"] = ran["seconds"]
    except (OSError, ValueError):
        ran = None
    try:
        rep = json.load(open(os.path.join(d, "out", "report.json")))
    except (OSError, ValueError):
        row.update(delivered=False, passes=False, firstTry=False, why="no report: the run did not finish"
                   if ran is None or ran["returncode"] else "no report")
        return row
    dl = rep.get("deliver") or {}
    row.setdefault("seconds", rep.get("seconds"))
    best = rep["outcomes"][rep["best"]] if rep.get("best") is not None and rep.get("outcomes") else {}
    row.update(delivered=bool(dl.get("stl")), passes=bool(dl.get("passes")),
               mass=dl.get("mass"), safetyFactor=dl.get("safetyFactor"), method=dl.get("safetyFactorMethod"),
               bodies=best.get("bodies"), governedBy=dl.get("governedBy"), outcomes=len(rep.get("outcomes", [])),
               unconfirmed=(dl.get("confirmation") or {}).get("unconfirmed") or [])
    if not row["passes"]:
        skip = ("no outcome meets", "no duration or repeat", "unconfirmed:")
        ws = [w for w in dl.get("warnings", []) if not w.startswith(skip)]
        row["why"] = (ws or dl.get("warnings") or ["?"])[0][:240]
    want = c.get("pass", {})
    if "diagnosis" in want:
        # a case that cannot be met as written: the pass is the tool saying so
        said = [w for w in (json.load(open(os.path.join(d, "check.json"))).get("warnings", [])
                            if os.path.exists(os.path.join(d, "check.json")) else [])]
        row["probe"] = want["diagnosis"]
        row["diagnosed"] = any("piece" in w or "split" in w or "no path" in w for w in said)
        row["firstTry"] = row["diagnosed"]
    else:
        row["firstTry"] = bool(row["check"]["ok"] and row["check"]["warnings"] == 0 and row["passes"])
    return row


def table(out, ids=None) -> dict:
    rows = [score(c, os.path.join(out, c["id"])) for c in cases(ids) if os.path.isdir(os.path.join(out, c["id"]))]
    n = len(rows)
    card = {"cases": n, "firstTry": sum(r["firstTry"] for r in rows), "passes": sum(r["passes"] for r in rows),
            "checkClean": sum(bool(r["check"]["ok"] and r["check"]["warnings"] == 0) for r in rows), "rows": rows}
    json.dump(card, open(os.path.join(out, "scorecard.json"), "w"), indent=1)
    lines = [f"first try {card['firstTry']}/{n} | delivered a passing part {card['passes']}/{n} | "
             f"check clean {card['checkClean']}/{n}", "",
             "| case | check | part | mass g | SF | min | why not |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ck = "clean" if r["check"]["ok"] and not r["check"]["warnings"] else \
            (f"{r['check']['warnings']} warnings" if r["check"]["ok"] else "error")
        part = "passes" if r["passes"] else ("fails" if r.get("delivered") else "none")
        if "probe" in r:
            part = "said so" if r["diagnosed"] else "not diagnosed"
        mass = f"{r['mass']:.1f}" if r.get("mass") else ""
        sf = f"{r['safetyFactor']:.2f}" if r.get("safetyFactor") is not None else ""
        mins = f"{r['seconds'] / 60:.0f}" if r.get("seconds") else ""
        lines.append(f"| {r['id']} | {ck} | {part} | {mass} | {sf} | {mins} | {r.get('why', '')} |")
    open(os.path.join(out, "scorecard.md"), "w").write("\n".join(lines) + "\n")
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m loadpath.evals.heldout", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=2)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--time-limit", type=float, default=40, help="minutes per case (loadpath run --time-limit)")
    r.add_argument("ids", nargs="*")
    t = sub.add_parser("table")
    t.add_argument("out")
    t.add_argument("ids", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for c in cases():
            print(f"{c['id']:42s} {' '.join(c.get('tags', []))}")
        return 0
    if a.cmd == "run":
        os.makedirs(a.out, exist_ok=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            for cid in ex.map(lambda c: _run_one(c, a.out, a.time_limit, a.threads), cases(a.ids)):
                print("ran", cid, flush=True)
    card = table(a.out, getattr(a, "ids", None))
    print(open(os.path.join(a.out, "scorecard.md")).read())
    return 0 if card["firstTry"] == card["cases"] else 1
