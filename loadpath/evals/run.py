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

Tiers (--tier):
    full     the suite as written (~2 h on the 4-core cloud box at --jobs 1).
    coarse   every case at <= COARSE["maxElements"] cells, fewer iterations,
             a shorter search and no fine-grid confirm: a smoke run for every
             push (minutes, not hours). Its numbers are not the full tier's;
             compare coarse with coarse (--baseline).

Every study row also gets a Pareto score (pareto.py): massAtTarget, the
front's mass at the goal SF, and its hypervolume. They move when a case gets
lighter at the same SF, which pass/fail does not.

--baseline <scorecard.json> diffs against an earlier run of the same tier:
a case regresses when it passed and now fails, or its massAtTarget grew more
than --tolerance (default 5%), and the regressions are listed. With
--gate regressions the exit code counts regressions instead of failures, so
a coarse run can gate a push on "no worse than before".

--corpus 8 (or 100:108) adds generated cases (corpus.py): parts nobody has
tuned on. --critic anthropic|agent renders interface close-ups of each pick
and has a vision model file findings (loadpath/critic.py); expect keys
maxCriticBlockers and maxCriticScore gate on them.
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


# coarse tier: cells, optimizer iterations, outcome-search runs, fine-grid
# confirm runs, heat-exchanger grid step multiplier
COARSE = {"maxElements": 6000, "maxIterations": 30, "searchRuns": 3, "confirmRuns": 0, "hxStep": 1.0}
# checks the coarse grid cannot be held to (they resolve features finer than
# its cells, or time a different run)
COARSE_SKIP = {"member", "crease", "seconds", "processWall", "boundaryCut", "overhang", "warpSpan",
               "maxMass", "mass", "massReduction", "deflection", "wallBetweenFluids", "internalFlatArea",
               "trappedVolume"}
CRITIC = {"criticBlockers": ("maxCriticBlockers", "<="), "criticScore": ("maxCriticScore", "<=")}


def apply_tier(raw: dict, tier: str) -> dict:
    """The case as the tier runs it (a copy)."""
    if tier == "full":
        return raw
    if tier != "coarse":
        raise ValueError(f"unknown tier {tier!r} (full, coarse)")
    raw = json.loads(json.dumps(raw))
    # a case can ask for more cells (expect.coarse): thin load pads and
    # pin-held pieces vanish below a few thousand cells
    cfg = COARSE | {k: v for k, v in ((raw.get("expect") or {}).get("coarse") or {}).items() if k != "skip"}
    if "core" in raw:
        raw["step"] = raw.get("step", 0.4) * cfg["hxStep"]
        return raw
    res = raw.get("resolution") or {}
    raw["resolution"] = {"maxElements": min(int(res.get("maxElements", 80000)), cfg["maxElements"])}
    raw["maxIterations"] = min(int(raw.get("maxIterations", 120)), cfg["maxIterations"])
    return raw


def _tier_globals(tier: str):
    """Search limits are study.py constants; the coarse tier lowers them in
    this process only."""
    if tier == "coarse":
        # recall.py's design memory would start a rerun where the last one
        # ended: a baseline diff must compare cold runs
        os.environ.pop("LOADPATH_RECALL", None)
        from .. import study as st
        st._MAX_SEARCH_RUNS = COARSE["searchRuns"]
        st._CONFIRM_RUNS = COARSE["confirmRuns"]


def gate_critic(row: dict, expect: dict | None = None):
    """Add the critic's checks to a row (expect from the row's case file when
    not given) and redo its pass."""
    c = row.get("critic")
    if not c:
        return row
    if expect is None:
        expect = _case_expect(row.get("case")) or {}
    row["checks"] = [k for k in row.get("checks", []) if k["check"] not in CRITIC]
    for name, (key, op) in CRITIC.items():
        if key in expect:
            v = c.get(name)
            row["checks"].append({"check": name, "value": v, "op": op, "limit": expect[key],
                                  "pass": _cmp(v, op, expect[key])})
    row["pass"] = row.get("error") is None and all(k["pass"] for k in row["checks"])
    return row


def _case_expect(name):
    f = os.path.join(HERE, "cases", f"{name}.json")
    if os.path.exists(f):
        with open(f) as fh:
            return json.load(fh).get("expect")
    return None


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


def score_case(raw: dict, result, seconds: float, study=None, tier: str = "full") -> dict:
    from .pareto import score as pareto

    expect = raw.get("expect", {})
    best = result.best
    out = {"case": raw.get("name"), "seconds": seconds, "checks": [], "outcomes": len(result.outcomes),
           "tier": tier,
           "pareto": pareto([o.report for o in result.outcomes],
                            (raw.get("goal") or {}).get("safetyFactor") or expect.get("minSafetyFactor"))}
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
        if key not in expect or (tier == "coarse" and (name in COARSE_SKIP or key in COARSE_SKIP)):
            continue
        v = measure(flat)
        passed = _cmp(v, op, expect[key])
        ok &= passed
        out["checks"].append({"check": name, "value": v, "op": op, "limit": expect[key], "pass": passed})
    if "maxSeconds" in expect and tier != "coarse":
        passed = seconds <= expect["maxSeconds"]
        ok &= passed
        out["checks"].append({"check": "seconds", "value": seconds, "op": "<=",
                              "limit": expect["maxSeconds"], "pass": passed})
    out["pass"] = bool(ok)
    return out


def score_hx_case(raw: dict, out: dict, seconds: float, tier: str = "full") -> dict:
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
        if key not in expect or (tier == "coarse" and name in COARSE_SKIP):
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
    if "maxSeconds" in expect and tier != "coarse":
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


def _mat(pa):
    """Pareto massAtTarget for the table: '~' marks an upper bound (no
    failing outcome below it to interpolate from)."""
    m = (pa or {}).get("massAtTarget")
    if m is None:
        return "-"
    return f"{m:.1f}" if pa.get("bracketed") else f"~{m:.1f}"


def _critic_cell(r):
    c = r.get("critic")
    if not c:
        return "pending" if r.get("criticPending") else "-"
    if c.get("error"):
        return "error"
    return f"{c['criticScore']} ({c['criticBlockers']}B/{c['criticMajors']}M)"


def _md_study(rows):
    lines = ["| Case | Pass | Mass g | Lighter | SF | Mass at target SF g | Defl. mm | p5 member mm | p5 ray mm | "
             "Crease mm | Critic | Time s | Failed checks |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"| {r['case']} | skipped | " + "- | " * 10 + f"0 | {r['skipped']} |")
            continue
        b = r.get("best") or {}
        failed = ", ".join(f"{c['check']} {c['value']:.3g} (limit {c['op']} {c['limit']})"
                           for c in r.get("checks", []) if not c["pass"] and c["value"] is not None)
        if r.get("error"):
            failed = (r["error"] + "; " + failed).strip("; ")
        fmt = lambda v, f: (f.format(v) if v is not None else "-")  # noqa: E731
        lines.append(f"| {r['case']} | {'yes' if r.get('pass') else 'NO'} | {fmt(b.get('mass'), '{:.1f}')} | "
                     f"{fmt(b.get('massReduction'), '{:.0%}')} | {fmt(b.get('safetyFactor'), '{:.2f}')} | "
                     f"{_mat(r.get('pareto'))} | "
                     f"{fmt(b.get('deflection'), '{:.3f}')} | {fmt(b.get('p5Member'), '{:.2f}')} | {fmt(b.get('p5Thickness'), '{:.2f}')} | "
                     f"{fmt(b.get('creaseLength'), '{:.1f}')} | {_critic_cell(r)} | "
                     f"{r['seconds']:.0f} | {failed} |")
    return "\n".join(lines) + "\n"


def run_hx_case(raw: dict, path: str, out: str, tier: str = "full") -> dict:
    from ..__main__ import _jsonable
    from ..tasks import heat_exchanger

    name = raw.get("name") or os.path.basename(path)
    t0 = time.time()
    try:
        spec = {k: v for k, v in raw.items() if k != "expect"}
        res = heat_exchanger(spec, out, base_dir=os.path.dirname(path), render=tier == "full", viewer=False)
        row = score_hx_case(raw, res, time.time() - t0, tier)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row = {"case": name, "kind": "heatExchanger", "pass": False, "seconds": time.time() - t0,
               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "exchanger", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


def run_case(path, out: str, tier: str = "full", critic: str | None = None) -> dict:
    """Run one case: `path` is a case file, or a generated case dict."""
    from ..__main__ import _jsonable
    from ..render import render_interfaces, render_views
    from ..spec import study_from_dict
    from ..study import run_study

    _tier_globals(tier)
    if isinstance(path, dict):
        raw, base = path, HERE
    else:
        with open(path) as fh:
            raw = json.load(fh)
        base = os.path.dirname(path)
    name = raw.get("name") or os.path.basename(str(path))
    skip = ((raw.get("expect") or {}).get("coarse") or {}).get("skip") if tier == "coarse" else None
    if skip:
        print(f"== {name} (skipped in the coarse tier: {skip})", file=sys.stderr, flush=True)
        return {"case": name, "tier": tier, "skipped": skip, "pass": True, "seconds": 0.0, "checks": []}
    raw = apply_tier(raw, tier)
    print(f"== {name} ({tier})", file=sys.stderr, flush=True)
    if "core" in raw:
        return run_hx_case(raw, path if isinstance(path, str) else os.path.join(HERE, name), out, tier)
    t0 = time.time()
    try:
        study = study_from_dict(raw, base_dir=base)
        result = run_study(study, on_progress=lambda e, _x: (
            print(json.dumps(_jsonable({k: v for k, v in e.items()
                                        if k in ("stage", "volumeFraction", "safetyFactor", "mass")}
                                       | {"case": name})),
                  file=sys.stderr, flush=True) if e.get("stage") in ("outcome",) else None))
        row = score_case(raw, result, time.time() - t0, study, tier)
        if result.best is not None or result.outcomes:
            o = result.outcomes[result.best if result.best is not None else -1]
            o.mesh.export(os.path.join(out, f"{name}.stl"))
            render_views(o.mesh, os.path.join(out, f"{name}.png"), study,
                         title=f"{name}: {o.report['mass']:.1f} g, SF {o.report['safetyFactor']:.2f}")
            if critic:
                from ..critic import critique

                shot = render_interfaces(o.mesh, study, os.path.join(out, f"{name}.interfaces.png"))
                try:
                    c = critique([shot], study, backend=critic, out_dir=out, name=name)
                except Exception as exc:  # noqa: BLE001 - a critic outage is not a part failure
                    c = {"error": f"{type(exc).__name__}: {exc}"}
                if c is not None:
                    row["critic"] = c
                    gate_critic(row, raw.get("expect", {}))
                elif critic == "agent":
                    row["criticPending"] = f"{name}.critic-request.md"
        row["trace"] = [o.summary() for o in result.outcomes]
        for t in row["trace"]:
            t.pop("cases", None)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row = {"case": name, "pass": False, "seconds": time.time() - t0, "tier": tier,
               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "best", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


def compare(rows, base_rows, tolerance=0.05):
    """Regressions against an earlier scorecard: a case that passed and now
    fails, or whose Pareto massAtTarget grew by more than `tolerance`. Also
    lists improvements. Adds row["vsBaseline"]."""
    base = {r["case"]: r for r in base_rows}
    regress, improve = [], []
    for r in rows:
        b = base.get(r["case"])
        if b is None or r.get("skipped") or b.get("skipped"):
            continue
        d = {}
        if b.get("pass") and not r.get("pass"):
            d["pass"] = "lost"
        elif r.get("pass") and not b.get("pass"):
            d["pass"] = "gained"
        m, mb = (r.get("pareto") or {}).get("massAtTarget"), (b.get("pareto") or {}).get("massAtTarget")
        if m is not None and mb:
            d["massAtTarget"] = m / mb - 1.0
        elif mb and m is None:
            d["massAtTarget"] = "lost"
        r["vsBaseline"] = d
        worse = d.get("pass") == "lost" or d.get("massAtTarget") == "lost" or \
            (isinstance(d.get("massAtTarget"), float) and d["massAtTarget"] > tolerance)
        better = d.get("pass") == "gained" or \
            (isinstance(d.get("massAtTarget"), float) and d["massAtTarget"] < -tolerance)
        (regress if worse else improve if better else []).append(r["case"])
    return regress, improve


def _md_compare(rows, regress, improve):
    lines = ["", f"vs baseline: {len(regress)} regressed, {len(improve)} improved", "",
             "| Case | Pass | massAtTarget g | vs baseline |", "| --- | --- | --- | --- |"]
    for r in rows:
        d = r.get("vsBaseline")
        if r.get("skipped"):
            continue
        if d is None:
            continue
        m = (r.get("pareto") or {}).get("massAtTarget")
        dm = d.get("massAtTarget")
        txt = (f"{dm:+.1%}" if isinstance(dm, float) else (dm or "-")) + (f", pass {d['pass']}" if "pass" in d else "")
        flag = " **regressed**" if r["case"] in regress else ""
        lines.append(f"| {r['case']} | {'yes' if r.get('pass') else 'NO'} | "
                     f"{'-' if m is None else f'{m:.1f}'} | {txt}{flag} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    from ..__main__ import _jsonable

    p = argparse.ArgumentParser(prog="loadpath.evals.run")
    p.add_argument("cases", nargs="*", help="case names (default: all)")
    p.add_argument("--out", default="evals-out")
    p.add_argument("--jobs", type=int,
                   help="cases to run at once (each case also parallelizes its own outcomes); default 1, "
                        "or every core for the coarse tier (its cases are small)")
    p.add_argument("--tier", choices=["full", "coarse"], default="full")
    p.add_argument("--corpus", help="add generated cases: N (seeds 0..N-1), A:B, or a,b,c")
    p.add_argument("--only-corpus", action="store_true", help="run the generated cases only")
    p.add_argument("--critic", choices=["anthropic", "agent"], help="vision critic on interface close-ups")
    p.add_argument("--baseline", help="an earlier scorecard.json to diff against")
    p.add_argument("--tolerance", type=float, default=0.05, help="massAtTarget growth that counts as a regression")
    p.add_argument("--save-baseline", help="also write this run's scores (no traces) here, for --baseline")
    p.add_argument("--gate", choices=["failures", "regressions"], default="failures",
                   help="what the exit code counts")
    args = p.parse_args(argv)
    # a scorecard measures the optimizer, not what earlier runs left behind
    os.environ.pop("LOADPATH_RECALL", None)
    files = sorted(glob.glob(os.path.join(HERE, "cases", "*.json")))
    if args.cases:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in args.cases]
    if args.only_corpus:
        files = []
    if args.corpus:
        from .corpus import generate, seeds
        files += [generate(s) for s in seeds(args.corpus)]
    os.makedirs(args.out, exist_ok=True)
    n = len(files)
    if args.jobs is None:
        args.jobs = (os.cpu_count() or 1) if args.tier == "coarse" else 1
    if args.jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # One process per case; outcomes inside a case stay in that process.
        os.environ.setdefault("LOADPATH_WORKERS", "1")
        # and one BLAS / OpenMP thread each: four cases each spinning up a
        # core's worth of BLAS threads ran 3-10x slower than one at a time
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")
        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(run_case, files, [args.out] * n, [args.tier] * n, [args.critic] * n))
    else:
        rows = [run_case(f, args.out, args.tier, args.critic) for f in files]
    md = _md(rows)
    regress = []
    if args.baseline:
        with open(args.baseline) as fh:
            regress, improve = compare(rows, json.load(fh), args.tolerance)
        md += _md_compare(rows, regress, improve)
    with open(os.path.join(args.out, "scorecard.json"), "w") as fh:
        json.dump(_jsonable(rows), fh, indent=2)
    with open(os.path.join(args.out, "scorecard.md"), "w") as fh:
        fh.write(md)
    if args.save_baseline:
        keep = ("case", "tier", "pass", "skipped", "best", "pareto", "checks", "seconds", "error")
        with open(args.save_baseline, "w") as fh:
            json.dump(_jsonable([{k: r[k] for k in keep if k in r} for r in rows]), fh, indent=1)
    print(md)
    pending = [r["criticPending"] for r in rows if r.get("criticPending")]
    if pending:
        print(f"critic: {len(pending)} request(s) in {args.out} for the agent to answer "
              f"(<case>.critic.json), then: python -m loadpath.critic score {args.out}", file=sys.stderr)
    if args.gate == "regressions":
        return len(regress)
    return sum(0 if r.get("pass") else 1 for r in rows)


if __name__ == "__main__":
    sys.exit(main())
