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
      "minMemberAtLeast": 2.4,    # mm, 5th percentile member thickness (surface rays,
                                  # verify.surface_thickness; the voxel ridge reading
                                  # is quantized to the analysis cell)
      "maxCreaseLength": 10,      # mm of sharp concave edge (verify._creases)
      "maxClosedBores": 0,        # bore ends the part closes over (verify._closed_bores)
      "maxUndercutVolume": 150,   # mm^3 a cast/milled part's mold or cutter cannot reach,
                                  # outside keep-outs (report.manufacturing.access)
      "minProcessWall": 0.8,      # mm, 1st-percentile wall (SLS/MJF/SLA/DLP: processcheck)
      "maxTrappedVolume": 1,      # mm^3 of powder or resin that cannot get out
      "maxWarpSpan": 100,         # mm, longest thin flat panel (SLS/MJF)
      "maxCupArea": 10,           # mm^2, largest suction cup (SLA/DLP)
      "maxSeconds": 600
    }

A case with a "core" is a heat exchanger (loadpath.tasks.heat_exchanger)
instead, scored on its own checks (HX_CHECKS):

    "expect": {
      "maxMass": 140,                  # g, solid volume x the duty's wall material
      "minWallBetweenFluids": 0.9,     # mm, measured on the grid
      "maxInternalFlatArea": 200,      # mm^2 of flat ceiling inside (unsupportable in metal)
      "maxTrappedVolume": 5,           # mm^3 of open space no port reaches (trapped powder)
      "minEffectiveness": 0.5,         # rated (needs duty)
      "maxPressureDrop": {"air": 20000},  # Pa per fluid
      "maxSeconds": 300
    }

and always on isolated, connected and watertight.

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
    # organic material sawn flat on the design box where nothing mates
    "boundaryCut": ("maxBoundaryCutArea", lambda o: o.get("boundaryCutArea") or 0.0, "<="),
    "member": ("minMemberAtLeast", lambda o: o["p5Thickness"] if o.get("p5Thickness") is not None
               else o["p5Member"], ">="),
    # sharp concave edges the optimizer left (verify._creases), mm
    "crease": ("maxCreaseLength", lambda o: o.get("creaseLength"), "<="),
    # bores the part closes over: no screw can be driven through (verify._closed_bores)
    "closedBores": ("maxClosedBores", lambda o: len(o.get("closedBores") or []), "<="),
    "overhang": ("maxOverhangFraction", lambda o: o.get("designOverhangFraction"), "<="),
    # SLS/MJF/SLA/DLP process checks on the delivered part (processcheck.py)
    "processWall": ("minProcessWall", lambda o: _proc(o, "walls", "p1WallMm"), ">="),
    "trappedVolume": ("maxTrappedVolume", lambda o: _proc(o, "trappedVolumeMm3"), "<="),
    "warpSpan": ("maxWarpSpan", lambda o: _proc(o, "warp", "maxSpan"), "<="),
    "cupArea": ("maxCupArea", lambda o: _proc(o, "cups", "maxArea"), "<="),
    # cast or milled: mm^3 of void no mold half or cutter reaches, outside the
    # keep-outs (an undercut keep-out is the study's own, and needs a core)
    "undercut": ("maxUndercutVolume", lambda o: _undercut(o), "<="),
}


def _undercut(o):
    a = (o.get("manufacturing") or {}).get("access")
    return None if a is None else a["undercutVolume"] - a["undercutInKeepOuts"]


def _proc(o, *keys):
    v = o.get("process")
    for k in keys:
        v = v.get(k) if isinstance(v, dict) else None
    return v


HX_CHECKS = {
    "mass": ("maxMass", lambda r: r.get("mass"), "<="),
    "wallBetweenFluids": ("minWallBetweenFluids", lambda r: r["report"].get("minWallBetweenFluids"), ">="),
    "internalFlatArea": ("maxInternalFlatArea",
                         lambda r: (r["report"].get("internalCeilings") or {}).get("flatArea"), "<="),
    "trappedVolume": ("maxTrappedVolume", lambda r: r["report"].get("trappedVolume"), "<="),
    "effectiveness": ("minEffectiveness", lambda r: (r.get("rating") or {}).get("effectiveness"), ">="),
}


def _cmp(v, op, ref):
    if v is None:
        return False
    return {"<=": v <= ref + 1e-9, ">=": v >= ref - 1e-9, "==": v == ref}[op]


def score_case(raw: dict, result, seconds: float, study=None) -> dict:
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
    if getattr(o, "mesh", None) is not None:
        from ..verify import surface_thickness

        flat["p1Thickness"], flat["p5Thickness"] = surface_thickness(o.mesh, study=study)
    out["best"] = {"volumeFraction": o.volume_fraction, "mass": flat["mass"],
                   "massReduction": flat["massReduction"], "safetyFactor": flat["safetyFactor"],
                   "p5Member": flat["p5Member"], "minMember": flat["minMember"],
                   "creaseLength": flat.get("creaseLength"),
                   "p5Thickness": flat.get("p5Thickness"), "p1Thickness": flat.get("p1Thickness"),
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


def score_hx_case(raw: dict, out: dict, seconds: float) -> dict:
    """A heat-exchanger case: isolated, connected and watertight always, then
    the `expect` limits (HX_CHECKS, maxPressureDrop per fluid, maxSeconds)."""
    from ..materials import material_from_spec

    expect = raw.get("expect", {})
    rep = out["report"]
    row = {"case": raw.get("name"), "kind": "heatExchanger", "seconds": seconds, "checks": []}
    wall = ((raw.get("duty") or {}).get("wall"))
    try:
        density = material_from_spec(wall).density if isinstance(wall, str) else None
    except (KeyError, ValueError):
        density = None
    r = {"report": rep, "rating": out.get("rating"),
         "mass": rep["solidVolume"] * density / 1000.0 if density else None}
    ok = True
    for name, v in (("isolated", rep.get("isolated")), ("connected", all((rep.get("connected") or {}).values())),
                    ("watertight", rep.get("watertight"))):
        ok &= bool(v)
        row["checks"].append({"check": name, "value": bool(v), "op": "==", "limit": True, "pass": bool(v)})
    for name, (key, measure, op) in HX_CHECKS.items():
        if key not in expect:
            continue
        v = measure(r)
        passed = _cmp(v, op, expect[key])
        ok &= passed
        row["checks"].append({"check": name, "value": v, "op": op, "limit": expect[key], "pass": passed})
    for fl, limit in (expect.get("maxPressureDrop") or {}).items():
        v = (((r["rating"] or {}).get("fluids") or {}).get(fl) or {}).get("pressureDrop")
        passed = _cmp(v, "<=", limit)
        ok &= passed
        row["checks"].append({"check": f"pressureDrop {fl}", "value": v, "op": "<=", "limit": limit, "pass": passed})
    if "maxSeconds" in expect:
        passed = seconds <= expect["maxSeconds"]
        ok &= passed
        row["checks"].append({"check": "seconds", "value": seconds, "op": "<=", "limit": expect["maxSeconds"],
                              "pass": passed})
    if out.get("ratingError"):
        row["error"] = out["ratingError"]
        ok = False
    row["pass"] = bool(ok)
    ic = rep.get("internalCeilings") or {}
    rt = r["rating"] or {}
    row["exchanger"] = {"mass": r["mass"], "solidVolume": rep["solidVolume"],
                        "minWallBetweenFluids": rep.get("minWallBetweenFluids"),
                        "internalFlatArea": ic.get("flatArea"), "internalOverhangArea": ic.get("overhangArea"),
                        "trappedVolume": rep.get("trappedVolume"), "effectiveness": rt.get("effectiveness"),
                        "duty": rt.get("duty"),
                        "pressureDrop": {k: v.get("pressureDrop") for k, v in (rt.get("fluids") or {}).items()},
                        "cleanouts": len(rep.get("cleanouts") or [])}
    return row


def _md_hx(rows):
    lines = ["| Exchanger | Pass | Mass g | Wall mm | Flat ceiling mm² | Trapped mm³ | Effectiveness | "
             "Pressure drop Pa | Time s | Failed checks |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda v, f: (f.format(v) if v is not None else "-")  # noqa: E731
    for r in rows:
        x = r.get("exchanger") or {}
        failed = ", ".join(f"{c['check']} {c['value']} (limit {c['op']} {c['limit']})"
                           for c in r.get("checks", []) if not c["pass"])
        if r.get("error"):
            failed = (r["error"] + "; " + failed).strip("; ")
        dp = ", ".join(f"{k} {v:.0f}" for k, v in (x.get("pressureDrop") or {}).items() if v is not None) or "-"
        lines.append(f"| {r['case']} | {'yes' if r.get('pass') else 'NO'} | {fmt(x.get('mass'), '{:.1f}')} | "
                     f"{fmt(x.get('minWallBetweenFluids'), '{:.2f}')} | {fmt(x.get('internalFlatArea'), '{:.0f}')} | "
                     f"{fmt(x.get('trappedVolume'), '{:.1f}')} | {fmt(x.get('effectiveness'), '{:.2f}')} | {dp} | "
                     f"{r['seconds']:.0f} | {failed} |")
    return "\n".join(lines) + "\n"


def _md(rows):
    hx = [r for r in rows if r.get("kind") == "heatExchanger"]
    rows = [r for r in rows if r.get("kind") != "heatExchanger"]
    return _md_study(rows) + ("\n" + _md_hx(hx) if hx else "")


def _md_study(rows):
    lines = ["| Case | Pass | Mass g | Lighter | SF | Defl. mm | p5 member mm | p5 ray mm | Crease mm | Time s | Failed checks |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        b = r.get("best") or {}
        failed = ", ".join(f"{c['check']} {c['value']:.3g} (limit {c['op']} {c['limit']})"
                           for c in r.get("checks", []) if not c["pass"] and c["value"] is not None)
        if r.get("error"):
            failed = (r["error"] + "; " + failed).strip("; ")
        fmt = lambda v, f: (f.format(v) if v is not None else "-")  # noqa: E731
        lines.append(f"| {r['case']} | {'yes' if r.get('pass') else 'NO'} | {fmt(b.get('mass'), '{:.1f}')} | "
                     f"{fmt(b.get('massReduction'), '{:.0%}')} | {fmt(b.get('safetyFactor'), '{:.2f}')} | "
                     f"{fmt(b.get('deflection'), '{:.3f}')} | {fmt(b.get('p5Member'), '{:.2f}')} | {fmt(b.get('p5Thickness'), '{:.2f}')} | "
                     f"{fmt(b.get('creaseLength'), '{:.1f}')} | "
                     f"{r['seconds']:.0f} | {failed} |")
    return "\n".join(lines) + "\n"


def run_hx_case(raw: dict, path: str, out: str) -> dict:
    from ..__main__ import _jsonable
    from ..tasks import heat_exchanger

    name = raw.get("name") or os.path.basename(path)
    t0 = time.time()
    try:
        spec = {k: v for k, v in raw.items() if k != "expect"}
        res = heat_exchanger(spec, out, base_dir=os.path.dirname(path), render=True, viewer=False)
        row = score_hx_case(raw, res, time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row = {"case": name, "kind": "heatExchanger", "pass": False, "seconds": time.time() - t0,
               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "exchanger", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


def run_case(path: str, out: str) -> dict:
    from ..__main__ import _jsonable
    from ..render import render_views
    from ..spec import study_from_dict
    from ..study import run_study

    with open(path) as fh:
        raw = json.load(fh)
    name = raw.get("name") or os.path.basename(path)
    print(f"== {name}", file=sys.stderr, flush=True)
    if "core" in raw:
        return run_hx_case(raw, path, out)
    t0 = time.time()
    try:
        study = study_from_dict(raw, base_dir=os.path.dirname(path))
        result = run_study(study, on_progress=lambda e, _x: (
            print(json.dumps(_jsonable({k: v for k, v in e.items()
                                        if k in ("stage", "volumeFraction", "safetyFactor", "mass")}
                                       | {"case": name})),
                  file=sys.stderr, flush=True) if e.get("stage") in ("outcome",) else None))
        row = score_case(raw, result, time.time() - t0, study)
        if result.best is not None or result.outcomes:
            o = result.outcomes[result.best if result.best is not None else -1]
            o.mesh.export(os.path.join(out, f"{name}.stl"))
            render_views(o.mesh, os.path.join(out, f"{name}.png"), study,
                         title=f"{name}: {o.report['mass']:.1f} g, SF {o.report['safetyFactor']:.2f}")
        row["trace"] = [o.summary() for o in result.outcomes]
        for t in row["trace"]:
            t.pop("cases", None)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row = {"case": name, "pass": False, "seconds": time.time() - t0,
               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "best", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


def main(argv=None):
    from ..__main__ import _jsonable

    p = argparse.ArgumentParser(prog="loadpath.evals.run")
    p.add_argument("cases", nargs="*", help="case names (default: all)")
    p.add_argument("--out", default="evals-out")
    p.add_argument("--jobs", type=int, default=1,
                   help="cases to run at once (each case also parallelizes its own outcomes)")
    args = p.parse_args(argv)
    # a scorecard measures the optimizer, not what earlier runs left behind
    os.environ.pop("LOADPATH_RECALL", None)
    files = sorted(glob.glob(os.path.join(HERE, "cases", "*.json")))
    if args.cases:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in args.cases]
    os.makedirs(args.out, exist_ok=True)
    if args.jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # One process per case; outcomes inside a case stay in that process.
        os.environ.setdefault("LOADPATH_WORKERS", "1")
        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(run_case, files, [args.out] * len(files)))
    else:
        rows = [run_case(f, args.out) for f in files]
    with open(os.path.join(args.out, "scorecard.json"), "w") as fh:
        json.dump(_jsonable(rows), fh, indent=2)
    md = _md(rows)
    with open(os.path.join(args.out, "scorecard.md"), "w") as fh:
        fh.write(md)
    print(md)
    return sum(0 if r.get("pass") else 1 for r in rows)


if __name__ == "__main__":
    sys.exit(main())
