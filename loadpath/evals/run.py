"""
Eval suite: real parts with hard requirements, run end to end and scored on
the delivered mesh.

Each case in cases/*.json is an ordinary study plus an `expect` block (a
case with a `graph` is a field graph instead, analyzed with its lattice
homogenized on `cells` cells under `study`, and each of its `mustFail`
variants must miss a limit: run_field_case):

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
      "maxUnderDraftedArea": 50,  # mm^2 of cast wall under casting.draft (report.manufacturing.draft)
      "maxToolUnreachableVolume": 50,  # mm^3 of void a milled part's smallest cutter cannot reach
      "maxMachiningMinutes": 90,  # machining estimate (report.manufacturing.machining)
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
    full     the suite as written (~1 h 50 min on the 4-core cloud box at its default --jobs 2; every
             case in its own fresh process).
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
    # cast: mm^2 of wall with less than casting.draft (manufacture.draft_report)
    "draft": ("maxUnderDraftedArea", lambda o: _making(o, "draft", "underDraftedArea"), "<="),
    # milled: void the smallest cutter cannot reach, and the machining estimate
    "toolReach": ("maxToolUnreachableVolume", lambda o: _making(o, "machining", "toolUnreachableVolume"), "<="),
    "machiningMinutes": ("maxMachiningMinutes", lambda o: _making(o, "machining", "minutes", "total"), "<="),
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
    """The case as the tier runs it (a copy). LOADPATH_EVAL_CELLS_SCALE
    scales every case's maxElements (the pitch moves by its cube root): run
    the suite at 1 and at 0.91 to see how far each result moves with a 3%
    pitch change."""
    raw = _apply_tier(raw, tier)
    scale = float(os.environ.get("LOADPATH_EVAL_CELLS_SCALE") or 1.0)
    if scale != 1.0 and "core" not in raw and "graph" not in raw:
        raw = json.loads(json.dumps(raw))
        res = raw.setdefault("resolution", {})
        if "pitch" not in res:
            res["maxElements"] = max(1, int(round(int(res.get("maxElements", 60000)) * scale)))
    return raw


def _apply_tier(raw: dict, tier: str) -> dict:
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
    raw["resolution"] = {"maxElements": min(int(res.get("maxElements", 80000)), cfg["maxElements"]),
                         **{k: v for k, v in res.items() if k == "pitchStarts"}}
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


def _making(o, *keys):
    v = o.get("manufacturing")
    for k in keys:
        v = v.get(k) if isinstance(v, dict) else None
    return v


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
    buckles = (raw.get("goal") or {}).get("buckling", True)
    # the safety factor a part holds counts buckling as well (verify._buckling)
    held = [{**o.report, "safetyFactor": min(o.report["safetyFactor"], o.report["bucklingFactor"])}
            if buckles and o.report.get("bucklingFactor") is not None and o.report.get("safetyFactor") is not None
            else o.report for o in result.outcomes]
    out = {"case": raw.get("name"), "seconds": seconds, "checks": [], "outcomes": len(result.outcomes),
           "tier": tier,
           "pareto": pareto(held,
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
                   "iterations": o.optimizer["iterations"], "optimizerSeconds": o.optimizer["seconds"],
                   **({"bucklingFactor": flat["bucklingFactor"]} if flat.get("bucklingFactor") is not None else {})}
    ok = out.get("error") is None
    for name, (key, measure, op) in CHECKS.items():
        if key not in expect or (tier == "coarse" and (name in COARSE_SKIP or key in COARSE_SKIP)):
            continue
        v = measure(flat)
        passed = _cmp(v, op, expect[key])
        ok &= passed
        out["checks"].append({"check": name, "value": v, "op": op, "limit": expect[key], "pass": passed})
    if buckles and "minSafetyFactor" in expect and flat.get("bucklingFactor") is not None:
        passed = _cmp(flat["bucklingFactor"], ">=", expect["minSafetyFactor"])
        ok &= passed
        out["checks"].append({"check": "buckling", "value": flat["bucklingFactor"], "op": ">=",
                              "limit": expect["minSafetyFactor"], "pass": passed})
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
    enc = [r for r in rows if r.get("kind") == "enclosure"]
    rows = [r for r in rows if r.get("kind") not in ("heatExchanger", "enclosure")]
    return (_md_study(rows) + ("\n" + _md_hx(hx) if hx else "") + ("\n" + _md_enclosure(enc) if enc else "")
            + _md_spread(rows))


def _md_enclosure(rows):
    """Products from the outside in: one row per product, the checks that
    passed out of those run."""
    lines = ["Products from the outside in (loadpath enclosure)", "",
             "| Product | Pass | Checks passed | Parts | Mass g | Worst SF | Exterior untouched | Time s | Failed checks |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"| {r['case']} | skipped | - | - | - | - | - | 0 | {r['skipped']} |")
            continue
        cs = r.get("checks") or []
        parts = r.get("parts") or {}
        sfs = [p.get("safetyFactor") for p in parts.values() if p.get("safetyFactor") is not None]
        ext = all((p.get("exterior") or {}).get("untouched") for p in parts.values()) if parts else None
        failed = ", ".join(f"{c['check']} {c['value']}" + (f" (limit {c['op']} {c['limit']})"
                                                           if not isinstance(c["value"], bool) else "")
                           for c in cs if not c["pass"])
        if r.get("error"):
            failed = (r["error"] + "; " + failed).strip("; ")
        mass = (r.get("best") or {}).get("mass")
        lines.append(f"| {r['case']} | {'yes' if r.get('pass') else 'NO'} | {sum(c['pass'] for c in cs)}/{len(cs)} | "
                     f"{', '.join(parts) or '-'} | {'-' if mass is None else f'{mass:.1f}'} | "
                     f"{'-' if not sfs else f'{min(sfs):.2f}'} | {'-' if ext is None else ('yes' if ext else 'NO')} | "
                     f"{r.get('seconds', 0):.0f} | {failed} |")
    return "\n".join(lines) + "\n"


def _md_spread(rows):
    rows = [r for r in rows if r.get("spread")]
    if not rows:
        return ""
    lines = ["", "Run-to-run spread: massAtTarget (g) with the search started at each fraction", "",
             "| Case | Starts | massAtTarget g | Spread | Delivered g | Spread |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        sp = r["spread"]
        fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
        rng = "-" if sp.get("range") is None else f"{100 * sp['range']:.1f}%"
        mrng = "-" if sp.get("massRange") is None else f"{100 * sp['massRange']:.1f}%"
        lines.append(f"| {r['case']} | {', '.join(str(x['start']) for x in sp['runs'])} | "
                     f"{', '.join(fmt(x['massAtTarget']) for x in sp['runs'])} | {rng} | "
                     f"{', '.join(fmt(x.get('mass')) for x in sp['runs'])} | {mrng} |")
    return "\n".join(lines) + "\n"


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


def run_field_case(raw: dict, base: str, out: str, tier: str = "full") -> dict:
    """A field-graph case ("graph"): the part the graph makes, its lattice
    homogenized, analyzed under the case's study on `cells` cells (the
    coarse tier: expect.coarse.cells, default 120k) and held to `expect`.
    Each `mustFail` variant (the graph with some numbers set, tune paths)
    must miss a limit: a design the search was once fooled into passing."""
    from ..__main__ import _jsonable
    from ..homogenized import homogenized_check
    from ..tune import _study, set_at

    name = raw.get("name")
    expect = raw.get("expect", {})
    cells = int(raw.get("cells", 1_000_000))
    if tier == "coarse":
        cells = int((expect.get("coarse") or {}).get("cells", 120_000))
    t0 = time.time()
    row = {"case": name, "kind": "field", "tier": tier, "checks": []}
    try:
        study = _study(raw, base)
        graph = raw["graph"]
        rep = homogenized_check(graph, study, cells=cells, base_dir=base)
        row["best"] = {"mass": rep["mass"], "safetyFactor": rep["safetyFactor"],
                       "deflection": max(c["deflection"] for c in rep["cases"]),
                       "analysisCells": rep["analysisCells"], "latticeDensity": rep.get("latticeDensity")}
        ok = True
        for check, (key, measure, op) in CHECKS.items():
            if key not in expect:
                continue
            v = measure(rep)
            passed = _cmp(v, op, expect[key])
            ok &= passed
            row["checks"].append({"check": check, "value": v, "op": op, "limit": expect[key], "pass": passed})
        for var in raw.get("mustFail") or []:
            d = json.loads(json.dumps(graph))
            for path, v in var["set"].items():
                set_at(d, path, v)
            r = homogenized_check(d, study, cells=cells, base_dir=base)
            defl = max(c["deflection"] for c in r["cases"])
            missed = (defl > expect.get("maxDeflection", float("inf"))
                      or r["safetyFactor"] < expect.get("minSafetyFactor", 0.0))
            ok &= missed
            row["checks"].append({"check": f"rejects {var['name']}", "value": defl, "op": ">",
                                  "limit": expect.get("maxDeflection"), "pass": bool(missed),
                                  "mass": r["mass"], "safetyFactor": r["safetyFactor"]})
        row["pass"] = bool(ok)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row.update({"pass": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    row["seconds"] = time.time() - t0
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "best", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


def run_switch_case(raw: dict, base: str, out: str, tier: str = "full") -> dict:
    """A push-pad or feel case (a `switch` and/or `feel` block): `loadpath
    switch` or `loadpath feel` on the study,
    held to `expect`: presses in their windows on the design (minInWindow)
    and read back from the delivered mesh (minDeliveredInWindow), the part's
    volume (maxVolume, mm^3), each hold case on the delivered mesh, and a
    watertight mesh. The coarse tier runs expect.coarse.iterations (default
    40) on the same grid."""
    from ..__main__ import _jsonable
    from ..spec import study_from_dict
    from ..switch import run_switch

    name = raw.get("name")
    expect = raw.get("expect", {})
    t0 = time.time()
    row = {"case": name, "kind": "switch", "tier": tier, "checks": []}
    assets = raw.get("assets")
    if assets:
        # Geometry that is not in this repository (a product's own CAD):
        # read from the directory an environment variable names, and the
        # case skipped where it is not set.
        where = os.environ.get(assets["env"], "")
        missing = [f for f in assets.get("files", []) if not (where and os.path.exists(os.path.join(where, f)))]
        if missing:
            why = f"needs {assets['env']} (a directory with {', '.join(missing)}; {assets.get('make', '')})"
            print(f"== {name} (skipped: {why})", file=sys.stderr, flush=True)
            return {"case": name, "kind": "switch", "tier": tier, "skipped": why, "pass": True, "seconds": 0.0,
                    "checks": []}
        base = where
    try:
        doc = {k: v for k, v in raw.items() if k not in ("expect", "assets")}
        st = study_from_dict(doc, base_dir=base)
        its = pitches = None
        if tier == "coarse":
            # one short level (expect.coarse.pitches, else the study's
            # first), held to expect.coarse's own limits where it sets them
            co = expect.get("coarse") or {}
            its = [int(co.get("iterations", 40))]
            pitches = co.get("pitches")
            expect = {**expect, **{k: v for k, v in co.items() if k.startswith(("min", "max"))}}
        # the large-deflection re-read on the full tier (about half a minute
        # a press on the touch microphone's 1 mm grid with PARDISO)
        if "feel" in doc:
            from ..feel import run_feel
            rep = run_feel(st, os.path.join(out, name), iterations=its, nonlinear=tier != "coarse")
        else:
            rep = run_switch(st, os.path.join(out, name), pitches=pitches, iterations=its,
                             nonlinear=tier != "coarse")
        vol = float(rep["mass"] / st.material.density * 1000.0)
        row["best"] = {"mass": rep["mass"], "volume": vol, "inWindow": rep["inWindow"],
                       "deliveredInWindow": rep["deliveredInWindow"]}
        row["presses"] = [{k: p.get(k) for k in ("at", "forceGf", "thresholdedForceGf", "deliveredForceGf",
                                                 "nonlinearForceGf")} for p in rep["presses"]]
        checks = [("in window (design)", rep["inWindow"], ">=", expect.get("minInWindow")),
                  ("in window (delivered mesh)", rep["deliveredInWindow"], ">=", expect.get("minDeliveredInWindow")),
                  ("volume mm^3", vol, "<=", expect.get("maxVolume")),
                  ("watertight", bool(rep["watertight"]), "==", True)]
        # loadpath never edits the exterior or removes what is drawn: a
        # keep-out crossing the skin (a plug's path) or swallowing a
        # preserve (a board's clear zone round its rails) fails the case
        if "exterior" in rep:
            checks.append(("exterior kept (coverage)", rep["exterior"]["coverage"], ">=", 0.999))
            checks.append(("exterior kept (max proud mm)", rep["exterior"]["maxProud"], "<=", 0.2))
        for g in (rep.get("drawn") or {}).get("regions") or []:
            if "missingShare" in g:
                checks.append((f"drawn {g['id']} kept (missing share)", g["missingShare"], "<=", 0.02))
        if "nonlinear" in rep:
            # every large-deflection press solved to the click (it crawled
            # and stopped short on thin TPU skins before PARDISO)
            nl_ok = sum(bool(n.get("converged")) for n in rep["nonlinear"])
            checks.append(("large-deflection presses converged", nl_ok, ">=", len(rep["nonlinear"])))
        if "feel" in doc:
            # feel targets: each direction of each target in its stiffness
            # window, on the design and read back from the delivered mesh;
            # an isolate target's modal check too
            fr = rep["feel"]
            n_ok = sum(d["ok"] for r in fr for d in r["directions"])
            n_del = sum(d["ok"] for r in fr for d in (r.get("delivered") or []))
            n_modal = sum(d["ok"] for r in fr for d in ((r.get("modal") or {}).get("directions") or []))
            row["best"].update({"feelInWindow": n_ok, "feelDeliveredInWindow": n_del, "feelModalOk": n_modal})
            row["feel"] = [{"name": r["name"], "stiffness": [d["stiffness"] for d in r["directions"]],
                            "delivered": [d["stiffness"] for d in (r.get("delivered") or [])],
                            "window": [r["minStiffness"], r["maxStiffness"]]} for r in fr]
            checks += [("feel in window (design)", n_ok, ">=", expect.get("minFeelInWindow")),
                       ("feel in window (delivered mesh)", n_del, ">=", expect.get("minFeelDeliveredInWindow")),
                       ("isolation modal check", n_modal, ">=", expect.get("minFeelModalOk"))]
        for h, v in zip((st.raw.get("switch") or {}).get("hold") or [], rep.get("deliveredHold") or []):
            checks.append((f"hold {h['case']} mm (delivered mesh)", v, "<=", h["maxDeflection"]))
        ok = True
        for check, v, op, limit in checks:
            if limit is None:
                continue
            passed = bool(v == limit) if op == "==" else _cmp(v, op, limit)
            ok &= passed
            row["checks"].append({"check": check, "value": v, "op": op, "limit": limit, "pass": passed})
        row["pass"] = bool(ok)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row.update({"pass": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    row["seconds"] = time.time() - t0
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "best", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


def run_enclosure_case(raw: dict, base: str, out: str, tier: str = "full") -> dict:
    """A product from its outside in (`loadpath enclosure --run`): the case's
    `enclosure` is {"product": name} (products.py writes its exterior, parts
    and brief) or {"brief": path}. Scored on what the designer and the
    engineer would check: every part placed and every way in found, no way
    in cutting into the skin, and per delivered part (each half of a split
    shell) the exterior untouched, a watertight mesh, the safety factor of
    every load case on the delivered part, nothing in a keep-out, and the
    switch, feel or cooling targets met.

    expect: brief (keys merged into the product's brief: a coarser grid),
    minSafetyFactor (default the brief's goal), maxKeepOutIntrusion
    (mm^3, default 1), maxPathsIntoSkin (mm^3, default 0.05), maxMass (g,
    all parts), maxSeconds; designTargets (default true: switch/feel/cooling
    passes). expect.coarse: brief (keys merged into the brief: a coarser
    resolution), pitches and iterations (switch/feel/cooling), timeLimit
    (minutes, `run`), and its own min/max limits."""
    from ..__main__ import _jsonable
    from ..enclosure import design_and_check, run_enclosure

    name = raw.get("name")
    expect = dict(raw.get("expect") or {})
    t0 = time.time()
    row = {"case": name, "kind": "enclosure", "tier": tier, "checks": []}
    work = os.path.join(out, name)
    try:
        spec = raw["enclosure"]
        if spec.get("product"):
            from .products import write
            brief_path = write(spec["product"], os.path.join(work, "product"))
        else:
            brief_path = os.path.join(base, spec["brief"])
        brief = json.load(open(brief_path))
        # the tier's own changes to the product's brief (a coarser grid
        # than the product's own: an eval box has 15 GB)
        brief.update(expect.get("brief") or {})
        pitches = expect.get("pitches")
        its = expect.get("iterations")
        tl = expect.get("timeLimit")
        quick = False
        if tier == "coarse":
            co = expect.get("coarse") or {}
            brief.update(co.get("brief") or {})
            pitches = co.get("pitches", pitches)
            its = co.get("iterations", [30])
            tl = co.get("timeLimit", 10)
            # not `run --quick`: it coarsens the grid past the brief's own
            # (a 1.4 mm skin on 2.4 mm cells reads as nothing)
            expect = {**expect, **{k: v for k, v in co.items() if k.startswith(("min", "max"))}}
        brief_run = os.path.join(os.path.dirname(brief_path), f"{name}.{tier}.brief.json")
        with open(brief_run, "w") as fh:
            json.dump(brief, fh, indent=1)
        enc = os.path.join(work, "enclosure")

        def note(e, _x=None):
            if e.get("stage") in ("placement", "assembly", "split", "check", "mesh", "verify", "domain"):
                print(json.dumps(_jsonable({k: v for k, v in e.items() if isinstance(v, (str, int, float))}
                                           | {"case": name, "t": round(time.time() - t0)})),
                      file=sys.stderr, flush=True)
        r = run_enclosure(brief_run, enc, on_progress=note)
        rep = r["report"]
        comps = [c for c in brief.get("components") or []]
        placed = len(rep["placement"]["components"])
        blocked = [k for k, v in rep["assembly"]["components"].items() if v.get("insertFrom") is None
                   and not str(v.get("how", "")).startswith("split")]
        if rep.get("split"):
            blocked += [k for k, v in (rep["split"].get("assembly") or {}).get("paths", {}).items()
                        if v.get("insertFrom") is None]
        into_skin = max([0.0] + list((rep.get("pathsIntoSkin") or {}).values()))
        row["placement"] = {k: {kk: v.get(kk) for kk in ("centre", "roomToSkin", "roomToParts", "auto")}
                            for k, v in rep["placement"]["components"].items()}
        row["assembly"] = {k: v.get("insertFrom") for k, v in rep["assembly"]["components"].items()}
        row["warnings"] = rep.get("warnings")
        checks = [("components placed", placed, ">=", len(comps)),
                  ("ways in blocked", len(blocked), "<=", 0),
                  ("ways in into the skin mm^3", into_skin, "<=", expect.get("maxPathsIntoSkin", 0.05)),
                  ("study lint errors", len(rep.get("lint") or []), "<=", 0)]
        res = design_and_check(r, enc, pitches=pitches, iterations=its, time_limit=tl, quick=quick, on_progress=note)
        sf_goal = expect.get("minSafetyFactor", (brief.get("goal") or {}).get("safetyFactor"))
        mass = 0.0
        row["parts"] = {}
        for half, c in res["parts"].items():
            tag = "" if half == "whole" else f" ({half})"
            if c.get("error"):
                row.setdefault("errors", {})[half] = c["error"]
                checks.append((f"designed{tag}", False, "==", True))
                continue
            v = c.get("verify") or {}
            dr = c.get("designReport") or {}
            st = json.load(open(r["split"][half] if half != "whole" else r["studyPath"]))
            from ..spec import study_from_dict
            dens = study_from_dict(st, os.path.dirname(os.path.abspath(r["studyPath"]))).material.density
            m = float(c.get("volumeCm3") or 0.0) * dens
            mass += m
            row["parts"][half] = {"mass": round(m, 2), "exterior": c.get("exterior"), "watertight": c.get("watertight"),
                                  "safetyFactor": v.get("safetyFactor"), "keepOutIntrusion": v.get("keepOutIntrusion"),
                                  "designPasses": dr.get("passes"), "seconds": c.get("designSeconds")}
            checks += [(f"exterior untouched{tag}", bool((c.get("exterior") or {}).get("untouched")), "==", True),
                       (f"watertight{tag}", bool(c.get("watertight")), "==", True)]
            if v:
                checks += [(f"safety factor{tag}", v.get("safetyFactor"), ">=", sf_goal),
                           (f"keep-out intrusion mm^3{tag}", v.get("keepOutIntrusion"), "<=",
                            expect.get("maxKeepOutIntrusion", 1.0))]
            if "passes" in dr and expect.get("designTargets", True):
                checks.append((f"switch/feel/cooling targets{tag}", bool(dr["passes"]), "==", True))
        row["best"] = {"mass": round(mass, 2), "placed": placed, "blocked": blocked}
        row["seconds"] = time.time() - t0
        checks += [("mass g", mass, "<=", expect.get("maxMass")),
                   ("seconds", row["seconds"], "<=", expect.get("maxSeconds") if tier == "full" else None)]
        ok = True
        for check, v, op, limit in checks:
            if limit is None:
                continue
            passed = bool(v == limit) if op == "==" else (v is not None and _cmp(v, op, limit))
            ok &= passed
            row["checks"].append({"check": check, "value": v, "op": op, "limit": limit, "pass": passed})
        row["pass"] = bool(ok)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row.update({"pass": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    row["seconds"] = time.time() - t0
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "best", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


# --spread: the search started this much lower and higher than the case's own
# goal.volumeFraction (one extra full search each)
SPREAD_STARTS = (0.8, 1.25)


def run_case(path, out: str, tier: str = "full", critic: str | None = None, spread: bool = False) -> dict:
    """Run one case: `path` is a case file, or a generated case dict. With
    `spread`, a safety-factor case is searched again from other starting
    fractions (SPREAD_STARTS) and row["spread"] says how far its massAtTarget
    moved: the search is deterministic on one machine, so this is the honest
    run-to-run number (a different start, a different machine's rounding, a
    small change to the code all send it down another path)."""
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
    if "graph" in raw:
        print(f"== {name} ({tier})", file=sys.stderr, flush=True)
        return run_field_case(raw, base, out, tier)
    if "enclosure" in raw:
        print(f"== {name} ({tier})", file=sys.stderr, flush=True)
        return run_enclosure_case(raw, base, out, tier)
    if "switch" in raw or "feel" in raw:
        print(f"== {name} ({tier})", file=sys.stderr, flush=True)
        return run_switch_case(raw, base, out, tier)
    raw = apply_tier(raw, tier)
    print(f"== {name} ({tier})", file=sys.stderr, flush=True)
    if "core" in raw:
        return run_hx_case(raw, path if isinstance(path, str) else os.path.join(HERE, name), out, tier)
    t0 = time.time()
    try:
        study = study_from_dict(raw, base_dir=base)
        result = run_study(study, on_progress=lambda e, _x: (
            print(json.dumps(_jsonable({k: v for k, v in e.items()
                                        if k in ("stage", "volumeFraction", "safetyFactor", "mass", "method",
                                                 "bucklingFactor", "margin")}
                                       | {"case": name, "t": round(time.time() - t0)})),
                  file=sys.stderr, flush=True) if e.get("stage") in ("outcome", "method", "confirmed") else None))
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
        if getattr(result, "choice", None):
            # goal.method auto: which method delivered, and why (stress
            # utilization, constrained vs search mass, a skipped or
            # time-boxed constrained stage)
            row["methodChoice"] = result.choice
        row["trace"] = [o.summary() for o in result.outcomes]
        for t in row["trace"]:
            t.pop("cases", None)
        if spread and study.goal.safety_factor is not None:
            row["spread"] = _spread(raw, base, row, tier)
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
        import traceback
        row = {"case": name, "pass": False, "seconds": time.time() - t0, "tier": tier,
               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    print(json.dumps(_jsonable({k: row.get(k) for k in ("case", "pass", "best", "error")})),
          file=sys.stderr, flush=True)
    return _jsonable(row)


def _spread(raw: dict, base: str, row: dict, tier: str) -> dict:
    """massAtTarget from the case's own start and from each SPREAD_STARTS
    multiple of it; "range" is (max - min) / min."""
    import copy

    from ..spec import study_from_dict
    from ..study import run_study

    vf0 = float((raw.get("goal") or {}).get("volumeFraction") or 0.3)
    runs = [{"start": vf0, "massAtTarget": (row.get("pareto") or {}).get("massAtTarget"),
             "mass": (row.get("best") or {}).get("mass")}]
    for f in SPREAD_STARTS:
        r2 = copy.deepcopy(raw)
        r2.setdefault("goal", {})["volumeFraction"] = round(vf0 * f, 4)
        st = study_from_dict(r2, base_dir=base)
        t0 = time.time()
        res = run_study(st)
        sc = score_case(r2, res, time.time() - t0, st, tier)
        runs.append({"start": r2["goal"]["volumeFraction"], "massAtTarget": (sc.get("pareto") or {}).get("massAtTarget"),
                     "mass": (sc.get("best") or {}).get("mass"), "pass": sc.get("pass")})
    def rng(key):
        v = [x[key] for x in runs if x.get(key) is not None]
        return (max(v) - min(v)) / min(v) if len(v) > 1 and min(v) > 0 else None
    # range: of massAtTarget (the front's mass at the goal SF); massRange: of
    # the delivered part's mass, which can swing wider (the wall bracket:
    # 68.7, 80.6, 71.4 g delivered, 67.1, 71.9, 68.9 g at target)
    return {"runs": runs, "range": rng("massAtTarget"), "massRange": rng("mass")}


def compare(rows, base_rows, tolerance=0.05):
    """Regressions against an earlier scorecard: a case that passed and now
    fails, or whose Pareto massAtTarget grew by more than `tolerance`, or by
    more than the case's own measured run-to-run spread (row "spread", from
    --spread, in either scorecard) where that is wider: the wall bracket's
    massAtTarget moved 7% with the starting fraction alone, so a 6% move
    there is not a regression. Also lists improvements. Adds
    row["vsBaseline"]."""
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
        tol = max([tolerance] + [(x.get("spread") or {}).get("range") or 0.0 for x in (r, b)])
        if tol > tolerance:
            d["tolerance"] = tol
        r["vsBaseline"] = d
        worse = d.get("pass") == "lost" or d.get("massAtTarget") == "lost" or \
            (isinstance(d.get("massAtTarget"), float) and d["massAtTarget"] > tol)
        better = d.get("pass") == "gained" or \
            (isinstance(d.get("massAtTarget"), float) and d["massAtTarget"] < -tol)
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


def _case_bytes(path: str) -> float:
    """Peak memory a case is expected to need (study._run_bytes by the most
    cells it solves at once)."""
    from ..spec import study_from_dict
    from ..study import _run_bytes, _study_cells

    with open(path) as fh:
        raw = json.load(fh)
    if "core" in raw:
        return _run_bytes(60_000)
    if "graph" in raw:
        return 2.5e9           # a 1M-cell homogenized check on multigrid peaks near 1.4 GB
    if "enclosure" in raw:
        return float((raw.get("expect") or {}).get("peakBytes", 6e9))
    try:
        return _run_bytes(_study_cells(study_from_dict(raw, base_dir=os.path.dirname(path))))
    except Exception:  # noqa: BLE001 - it fails properly when run
        return _run_bytes(60_000)


def _died(f, tier, exc) -> dict:
    """The row of a case whose worker process died (killed for memory, a
    crash in native code): a failure of that case, not of the run."""
    if isinstance(f, dict):
        name = f.get("name", "case")
    else:
        name = os.path.splitext(os.path.basename(f))[0]
    return {"case": name, "tier": tier, "pass": False, "skipped": False, "checks": [], "seconds": 0.0,
            "error": f"worker process died ({type(exc).__name__}: {exc}); out of memory?"}


def _run_isolated(files: list, out: str, tier: str = "full", critic=None, jobs: int = 1, spread=False) -> list:
    """Run each case in a fresh spawned process, `jobs` at a time. In one
    long process the cases add up: a finished case leaves its peak behind
    (glibc keeps freed arenas and Python's allocator keeps its pools), the
    next case sits on top, and a full --jobs 1 run on the 15 GB cloud box
    was killed at ~14 GB on its seventh case although every case fits
    alone. A worker that dies (out of memory) costs its own case: the pool
    is rebuilt for the rest."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    # keyed by position: a generated corpus case is a dict, not a path
    rows = {}
    todo = list(range(len(files)))
    while todo:
        ex = ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn"), max_tasks_per_child=1)
        futs = {ex.submit(run_case, files[i], out, tier, critic, spread): i for i in todo}
        broken = False
        for fut, i in futs.items():
            try:
                rows[i] = fut.result()
            except BrokenProcessPool as exc:
                # one at a time, the case that died is the first without a
                # result; with several running it could be any of them, so
                # the ones cut off run again one at a time
                if not broken and jobs == 1:
                    rows[i] = _died(files[i], tier, exc)
                    print(json.dumps({"stage": "caseDied", "case": rows[i]["case"]}), flush=True)
                broken = True
            except Exception as exc:  # noqa: BLE001 - one case's failure is its own row
                rows[i] = _died(files[i], tier, exc)
        ex.shutdown(wait=True, cancel_futures=True)
        todo = [i for i in todo if i not in rows]
        if broken:
            jobs = 1
    return [rows[i] for i in range(len(files))]


def _run_budgeted(files: list, out: str, tier: str = "full", critic=None, spread=False) -> list:
    """Run cases in processes, largest first, starting each only while the
    expected peaks of everything running fit in 80% of the memory available (the
    container's limit, when it has one)
    (and no more at once than there are cores). The 4-core 15 GB cloud box ran
    out of memory at a flat --jobs 2: two robot-arm cases at their 150k-cell
    confirmation solves."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from ..study import _available_bytes

    os.environ.setdefault("LOADPATH_WORKERS", "1")
    need = {f: _case_bytes(f) for f in files}
    budget = 0.8 * (_available_bytes() or 8e9)
    slots = max(1, os.cpu_count() or 1)
    queue = sorted(files, key=lambda f: -need[f])
    rows, running = {}, {}
    # a fresh process per case: a finished case's worker keeps its peak
    # (Python does not hand freed memory back), which the next case then
    # sits on top of
    with ProcessPoolExecutor(slots, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        while queue or running:
            used = sum(need[f] for f in running.values())
            for f in list(queue):
                if len(running) >= slots:
                    break
                if not running or used + need[f] <= budget:
                    running[ex.submit(run_case, f, out, tier, critic, spread)] = f
                    used += need[f]
                    queue.remove(f)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                f = running.pop(fut)
                try:
                    rows[f] = fut.result()
                except Exception as exc:  # noqa: BLE001 - a dead worker fails its case, not the run
                    rows[f] = _died(f, tier, exc)
    return [rows[f] for f in files]


def main(argv=None):
    from ..__main__ import _jsonable

    p = argparse.ArgumentParser(prog="loadpath.evals.run")
    p.add_argument("cases", nargs="*", help="case names (default: all)")
    p.add_argument("--out", default="evals-out")
    p.add_argument("--jobs",
                   help="cases to run at once (each case also parallelizes its own outcomes); default 2 (full tier), "
                        "or every core for the coarse tier (its cases are small); 'auto': as many as fit in "
                        "the memory available, by each case's size")
    p.add_argument("--tier", choices=["full", "coarse"], default="full")
    p.add_argument("--corpus", help="add generated cases: N (seeds 0..N-1), A:B, or a,b,c")
    p.add_argument("--only-corpus", action="store_true", help="run the generated cases only")
    p.add_argument("--spread", action="store_true",
                   help="search each safety-factor case again from other starting fractions and report how far "
                        "its massAtTarget moves (row spread; about 3x the time)")
    p.add_argument("--critic", choices=["anthropic", "agent"], help="vision critic on interface close-ups")
    p.add_argument("--baseline", help="an earlier scorecard.json to diff against")
    p.add_argument("--tolerance", type=float, default=0.05, help="massAtTarget growth that counts as a regression")
    p.add_argument("--save-baseline", help="also write this run's scores (no traces) here, for --baseline")
    p.add_argument("--gate", choices=["failures", "regressions"], default="failures",
                   help="what the exit code counts")
    p.add_argument("--allow-voxel-confirm", action="store_true",
                   help="run although no tetrahedral mesher is installed (the safety-factor cases then "
                        "confirm on voxels and come out heavier than the baselines)")
    args = p.parse_args(argv)
    from ..crosscheck import mesher_missing

    no_tets = mesher_missing()
    if no_tets and not args.allow_voxel_confirm:
        # The baselines were taken with the tet confirm: without it every
        # safety-factor case steps down on the voxel reading alone, and
        # the scorecard reads as a regression of the optimizer (13/14 with
        # the MBB beam at 40.4 g against 14.5 g) when it is the install.
        print(f"loadpath.evals.run: {no_tets}. The safety-factor cases confirm on tetrahedra; install it, "
              "or pass --allow-voxel-confirm to run without (not comparable to the baselines).", file=sys.stderr)
        return 2
    # a scorecard measures the optimizer, not what earlier runs left behind
    os.environ.pop("LOADPATH_RECALL", None)
    # and the same optimizer whatever else the box is running: the confirm
    # grid otherwise shrinks to the memory free when it starts (study.py
    # _confirm_cells), so a case's numbers would depend on its neighbours
    os.environ.setdefault("LOADPATH_CONFIRM_CELLS", "150000")
    files = sorted(glob.glob(os.path.join(HERE, "cases", "*.json")))
    if args.cases:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in args.cases]
    if args.only_corpus:
        files = []
    if args.corpus:
        from .corpus import generate, seeds
        files += [generate(s) for s in seeds(args.corpus)]
    os.makedirs(args.out, exist_ok=True)
    if args.jobs is None:
        # full: two at a time. With the AMG coarse-factor leak fixed (fem.py)
        # a robot-arm case peaks near 2 GB, and two ran the suite in 1 h 50 min
        # against 2 h 30 min one at a time on the 4-core 15 GB box, the
        # box at 3.8 GB, every number the same as --jobs 1
        args.jobs = (os.cpu_count() or 1) if args.tier == "coarse" else min(2, os.cpu_count() or 1)
    if args.jobs == "auto":
        rows = _run_budgeted(files, args.out, args.tier, args.critic, args.spread)
    elif os.environ.get("LOADPATH_EVAL_INPROCESS") == "1":
        # debugging: every case in this process (it accumulates; see _run_isolated)
        rows = [run_case(f, args.out, args.tier, args.critic, args.spread) for f in files]
    else:
        if int(args.jobs) > 1:
            # One process per case; outcomes inside a case stay in that process.
            os.environ.setdefault("LOADPATH_WORKERS", "1")
            # and one BLAS / OpenMP thread each: four cases each spinning up a
            # core's worth of BLAS threads ran 3-10x slower than one at a time
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(v, "1")
            # and the compiled kernels' threads split between the cases
            # (study._share_cores)
            os.environ.setdefault("LOADPATH_THREADS", str(max(1, (os.cpu_count() or 1) // int(args.jobs))))
        rows = _run_isolated(files, args.out, args.tier, args.critic, int(args.jobs), args.spread)
    md = _md(rows)
    if no_tets:
        md = f"**No tet confirm** ({no_tets}): safety-factor cases confirmed on voxels; not comparable to the baselines.\n\n" + md
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
        keep = ("case", "tier", "pass", "skipped", "best", "pareto", "checks", "seconds", "error", "spread",
                "methodChoice")
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
