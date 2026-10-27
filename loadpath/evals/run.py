"""
Eval suite: real parts with hard requirements, run end to end and scored on
the delivered mesh.

Each case in cases/*.json is an ordinary study plus an `expect` block:

    "expect": {
      "maxMass": 45,              # g, best outcome
      "minMassReduction": 0.5,    # vs the solid design space
      "minSafetyFactor": 3.0,     # every load case
      "maxDeflection": 1.0,       # mm at the loads, every case
      "maxKeepOutIntrusion": 1.0, # mm^3 of part inside keep-outs
      "maxPreserveMissing": 1.0,  # mm^3 of preserved geometry lost
      "bodies": 1,
      "minMemberAtLeast": 2.4,    # mm, 5th percentile member thickness
      "maxSeconds": 600
    }

Run:  python -m loadpath.evals.run [case ...] --out evals-out/
Writes scorecard.json, scorecard.md and a render per case. Exit code is the
number of failing cases, so it can gate a change to the solver.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CHECKS = {
    # name: (expect key, how to measure from (summary, report), comparison)
    "mass": ("maxMass", lambda o: o["mass"], "<="),
    "massReduction": ("minMassReduction", lambda o: o["massReduction"], ">="),
    "safetyFactor": ("minSafetyFactor", lambda o: o["safetyFactor"], ">="),
    "deflection": ("maxDeflection", lambda o: max(c["deflection"] for c in o["cases"]), "<="),
    "keepOutIntrusion": ("maxKeepOutIntrusion", lambda o: o["keepOutIntrusion"], "<="),
    "preserveMissing": ("maxPreserveMissing", lambda o: o["preserveMissing"], "<="),
    "bodies": ("bodies", lambda o: o["bodies"], "=="),
    "member": ("minMemberAtLeast", lambda o: o["p5Member"], ">="),
}


def _cmp(v, op, ref):
    if v is None:
        return False
    return {"<=": v <= ref + 1e-9, ">=": v >= ref - 1e-9, "==": v == ref}[op]


def score_case(raw: dict, result, seconds: float) -> dict:
    expect = raw.get("expect", {})
    best = result.best
    out = {"case": raw.get("name"), "seconds": seconds, "checks": [], "outcomes": len(result.outcomes)}
    if best is None:
        out["pass"] = False
        out["error"] = "no outcome met the goal"
        best = min(range(len(result.outcomes)),
                   key=lambda i: -(result.outcomes[i].report.get("safetyFactor") or 0)) if result.outcomes else None
        if best is None:
            return out
    o = result.outcomes[best]
    flat = dict(o.report)
    out["best"] = {"volumeFraction": o.volume_fraction, "mass": flat["mass"],
                   "massReduction": flat["massReduction"], "safetyFactor": flat["safetyFactor"],
                   "p5Member": flat["p5Member"], "minMember": flat["minMember"],
                   "deflection": max(c["deflection"] for c in flat["cases"]),
                   "iterations": o.optimizer["iterations"], "optimizerSeconds": o.optimizer["seconds"]}
    ok = out.get("error") is None
    for name, (key, measure, op) in CHECKS.items():
        if key not in expect:
            continue
        v = measure(flat)
        passed = _cmp(v, op, expect[key])
        ok &= passed
        out["checks"].append({"check": name, "value": v, "op": op, "limit": expect[key], "pass": passed})
    if "maxSeconds" in expect:
        passed = seconds <= expect["maxSeconds"]
        ok &= passed
        out["checks"].append({"check": "seconds", "value": seconds, "op": "<=",
                              "limit": expect["maxSeconds"], "pass": passed})
    out["pass"] = bool(ok)
    return out


def _md(rows):
    lines = ["| Case | Pass | Mass g | Lighter | SF | Defl. mm | p5 member mm | Time s | Failed checks |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        b = r.get("best") or {}
        failed = ", ".join(f"{c['check']} {c['value']:.3g} (limit {c['op']} {c['limit']})"
                           for c in r.get("checks", []) if not c["pass"] and c["value"] is not None)
        if r.get("error"):
            failed = (r["error"] + "; " + failed).strip("; ")
        fmt = lambda v, f: (f.format(v) if v is not None else "-")  # noqa: E731
        lines.append(f"| {r['case']} | {'yes' if r.get('pass') else 'NO'} | {fmt(b.get('mass'), '{:.1f}')} | "
                     f"{fmt(b.get('massReduction'), '{:.0%}')} | {fmt(b.get('safetyFactor'), '{:.2f}')} | "
                     f"{fmt(b.get('deflection'), '{:.3f}')} | {fmt(b.get('p5Member'), '{:.2f}')} | "
                     f"{r['seconds']:.0f} | {failed} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    from ..__main__ import _jsonable
    from ..render import render_views
    from ..spec import study_from_dict
    from ..study import run_study

    p = argparse.ArgumentParser(prog="loadpath.evals.run")
    p.add_argument("cases", nargs="*", help="case names (default: all)")
    p.add_argument("--out", default="evals-out")
    args = p.parse_args(argv)
    files = sorted(glob.glob(os.path.join(HERE, "cases", "*.json")))
    if args.cases:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in args.cases]
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in files:
        with open(path) as fh:
            raw = json.load(fh)
        name = raw.get("name") or os.path.basename(path)
        print(f"== {name}", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            study = study_from_dict(raw, base_dir=os.path.dirname(path))
            result = run_study(study, on_progress=lambda e, _x: (
                print(json.dumps(_jsonable({k: v for k, v in e.items()
                                            if k in ("stage", "volumeFraction", "safetyFactor", "mass")})),
                      file=sys.stderr, flush=True) if e.get("stage") in ("outcome",) else None))
            row = score_case(raw, result, time.time() - t0)
            if result.best is not None or result.outcomes:
                o = result.outcomes[result.best if result.best is not None else -1]
                o.mesh.export(os.path.join(args.out, f"{name}.stl"))
                render_views(o.mesh, os.path.join(args.out, f"{name}.png"), study,
                             title=f"{name}: {o.report['mass']:.1f} g, SF {o.report['safetyFactor']:.2f}")
            row["trace"] = [o.summary() for o in result.outcomes]
            for t in row["trace"]:
                t.pop("cases", None)
        except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
            import traceback
            row = {"case": name, "pass": False, "seconds": time.time() - t0,
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        rows.append(row)
        print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "best", "error")})),
              file=sys.stderr, flush=True)
    with open(os.path.join(args.out, "scorecard.json"), "w") as fh:
        json.dump(_jsonable(rows), fh, indent=2)
    md = _md(rows)
    with open(os.path.join(args.out, "scorecard.md"), "w") as fh:
        fh.write(md)
    print(md)
    return sum(0 if r.get("pass") else 1 for r in rows)


if __name__ == "__main__":
    sys.exit(main())
