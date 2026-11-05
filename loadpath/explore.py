"""
Explore: one study, many outcomes, one per way of making the part.

Fusion 360's generative design takes a study and a list of materials and
manufacturing methods, and hands back an outcome for each pair with its
mass, safety factor and a cost estimate, to filter and compare. This is
that, for an agent: the study is solved for every route (a manufacturing
method and a material it is made in), each with that method's constraints,
and the routes are ranked and put on Pareto fronts.

    loadpath explore study.json --out explore-out            # the default routes
    loadpath explore study.json --methods fdm,mjf,lpbf,cast,milled-3axis --quantity 50 --rank cost

or in the study itself:

    "explore": {"methods": ["fdm", "sls", "lpbf", "cast", "milled-3axis"],
                "materials": ["petg", "pa12", "alsi10mg", "a356-t6", "6061-t6"],
                "quantity": 10, "rank": "cost"}

Methods and what each one sets on the study (`METHODS`):

  unrestricted   the study as written (its own process and constraints, if
                 any) in each material: Fusion's "unrestricted", the
                 lightest shape the loads ask for; no cost (no method).
  fdm, sls, mjf, sla, dlp, jetting, volumetric, lpbf
                 process.type (overhang control, layer strength, minimum
                 feature and hole, drains); the study's buildDirection is
                 kept for the layered ones.
  cast           casting {draw, sides 2}: no undercuts along the draw, which
                 defaults to the design space's thinnest axis (a flat part
                 opens along its thickness).
  milled-2.5axis, milled-3axis, milled-5axis
                 milling {axes, directions, toolDiameter}: every void
                 reachable by a cutter. 2.5 and 3 axes cut from both ends
                 of the thinnest axis (two set-ups), 5 axes from all six
                 sides; toolDiameter 6 mm unless given.

A route's material comes from the library and must be made that way (its
card's `processes`: `loadpath materials`); `materials` narrows or widens the
defaults (`DEFAULT_MATERIALS`), `routes` names pairs outright. Each route is
a full `loadpath run` of the varied study in its own folder (its report,
STL, render), so everything a run checks (tet-confirmed safety factor,
print and process checks, drains, durability) holds per route.

Speed. The optimizer's layout at a volume fraction does not depend on the
material's E or strength (recall.py), so routes sharing a method run one
after another with design memory on (a cache in the output folder): the
second material replays the first one's designs and rescales its margins,
and only searches where they do not reach. Methods run in parallel
(`jobs`). A route whose study has not changed since an earlier explore in
the same folder is read back, not run again.

`screen: true` (`--screen`) runs every route on the coarse grid first
(quick.py) and the full runs only for the routes on a Pareto front or in the
top `refineTop` (2) of the ranking, and any whose coarse pick still had more
than SCREEN_MARGIN over its goal (the coarse grid could not go lighter, as
thin metal members do); the rest stay in the report with their
coarse numbers, marked `screenedOnly`. Methods differ by far more than the
coarse grid's few to tens of percent, so the screen keeps the right ones.

Each route then gets costing.estimate (unit cost at `quantity`, machine
hours, lead time) from its delivered part. Ranked by `rank` (mass, cost or
leadTime; passing routes first), with Pareto fronts on mass-cost and
cost-lead time among the passing, and the cheapest route at each quantity
(`crossover`): where a cast part's tooling stops mattering.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

__all__ = ["METHODS", "DEFAULT_METHODS", "DEFAULT_MATERIALS", "plan", "explore", "RANKS"]

# what each method sets on the study (see the module docstring)
METHODS = {
    "unrestricted": {"family": None},
    "fdm": {"process": "fdm", "family": "fdm"},
    "sls": {"process": "sls", "family": "sls"},
    "mjf": {"process": "mjf", "family": "mjf"},
    "sla": {"process": "sla", "family": "sla"},
    "dlp": {"process": "dlp", "family": "dlp"},
    "jetting": {"process": "jetting", "family": "jetting"},
    "volumetric": {"process": "volumetric", "family": "volumetric"},
    "lpbf": {"process": "lpbf", "family": "lpbf"},
    "cast": {"casting": True, "family": "cast"},
    "milled-2.5axis": {"milling": 2.5, "family": "milled"},
    "milled-3axis": {"milling": 3, "family": "milled"},
    "milled-5axis": {"milling": 5, "family": "milled"},
}
_METHOD_ALIASES = {"milled": "milled-3axis", "milling": "milled-3axis", "cnc": "milled-3axis",
                   "3axis": "milled-3axis", "5axis": "milled-5axis", "2.5axis": "milled-2.5axis",
                   "casting": "cast", "die-cast": "cast", "dmls": "lpbf", "slm": "lpbf", "polyjet": "jetting"}
# the default spread: one route per family a desktop-robot part is made by
DEFAULT_METHODS = ("fdm", "mjf", "sla", "lpbf", "cast", "milled-3axis")
# each method's materials when the explore names none
DEFAULT_MATERIALS = {
    "unrestricted": (), "fdm": ("petg",), "sls": ("pa12",), "mjf": ("pa12",), "sla": ("resin-tough",), "dlp": ("resin-tough",),
    "jetting": ("vero",), "volumetric": ("resin",), "lpbf": ("alsi10mg",), "cast": ("a356-t6",),
    "milled-2.5axis": ("6061-t6",), "milled-3axis": ("6061-t6",), "milled-5axis": ("6061-t6",),
}
RANKS = ("mass", "cost", "leadTime")
TOOL_DIAMETER = 6.0
# the screen's coarse search runs per route
SCREEN_SEARCH_RUNS = 5
# a screened route whose pick still has this much margin over its goal is
# refined whatever its rank: the coarse grid could not go lighter (thin metal
# members break on it: the arm link's Ti64 screen stopped at 225 g, margin 42)
SCREEN_MARGIN = 1.3
# keys a route sets or that only make sense for some methods
_ROUTE_KEYS = ("explore", "material", "materials", "process", "casting", "milling", "printDirection")


def _method(name) -> str:
    m = str(name).strip().lower()
    m = _METHOD_ALIASES.get(m, m)
    if m not in METHODS:
        raise ValueError(f"unknown method {name!r}; known: {', '.join(METHODS)}")
    return m


def _makes(material: str, method: str) -> bool:
    """Is `material` (a library name or alias) made by `method`?"""
    from .materials import material_from_spec
    try:
        m = material_from_spec(material)
    except ValueError:
        return False
    fam = METHODS[method]["family"]
    if fam is None:
        return True
    procs = set(m.processes or ())
    if not procs:
        return True
    if fam in ("sla", "dlp", "volumetric"):
        return bool(procs & {"sla", "dlp", "volumetric"})
    if fam == "milled":
        return bool(procs & {"milled", "milling"})
    if fam == "cast":
        return bool(procs & {"cast", "casting"})
    return fam in procs


def _canonical(material) -> str:
    from .materials import material_from_spec
    return material_from_spec(material).name if isinstance(material, str) else str(material.get("name", "custom"))


def _thin_axis(study: dict, base_dir) -> str:
    from .spec import study_from_dict
    d = {k: v for k, v in study.items() if k not in _ROUTE_KEYS}
    d["material"] = "pa12"
    lo, hi = study_from_dict(d, base_dir=base_dir).bounds()
    return "xyz"[int(min(range(3), key=lambda i: hi[i] - lo[i]))]


def _route_study(study: dict, method: str, material, opts: dict, thin: str) -> dict:
    spec = METHODS[method]
    rid = f"{method}-{_canonical(material)}"
    if spec["family"] is None:
        d = {k: copy.deepcopy(v) for k, v in study.items() if k not in ("explore", "material", "materials")}
        d["name"] = f"{study.get('name', 'part')}-{rid}"
        d["material"] = material
        return {"id": rid, "method": method, "material": _canonical(material), "study": d}
    d = {k: copy.deepcopy(v) for k, v in study.items() if k not in _ROUTE_KEYS}
    d["name"] = f"{study.get('name', 'part')}-{rid}"
    d["material"] = material
    base_proc = study.get("process")
    base_proc = {"type": base_proc} if isinstance(base_proc, str) else (base_proc or {})
    if "process" in spec:
        p = {"type": spec["process"]}
        if base_proc.get("type") == spec["process"]:
            p = {**base_proc}                       # the study's own settings for this process
        elif base_proc.get("buildDirection") is not None and spec["process"] in ("fdm", "lpbf", "sla", "dlp",
                                                                                 "jetting", "sls", "mjf"):
            p["buildDirection"] = base_proc["buildDirection"]
        d["process"] = p
        if spec["process"] != "fdm":
            d.pop("slicer", None)
    else:
        d.pop("slicer", None)
        d.pop("drains", None)
        bs = d.get("buildSim")
        if bs:
            d.pop("buildSim", None)
    if spec.get("casting"):
        cast = study.get("casting")
        draw = opts.get("draw") or (cast if isinstance(cast, str) else (cast or {}).get("draw")) or f"+{thin}"
        d["casting"] = {"draw": draw, "sides": int(opts.get("castSides", 2))}
    if spec.get("milling") is not None:
        axes = spec["milling"]
        mill = {"axes": axes, "toolDiameter": float(opts.get("toolDiameter", TOOL_DIAMETER))}
        if axes != 5:
            mill["directions"] = list(opts.get("millingDirections") or [f"+{thin}", f"-{thin}"])
        elif opts.get("millingDirections"):
            mill["directions"] = list(opts["millingDirections"])
        d["milling"] = mill
    return {"id": rid, "method": method, "material": _canonical(material), "study": d}


def plan(study: dict, explore_opts: dict | None = None, base_dir=None) -> dict:
    """The routes an explore runs: {"routes": [{id, method, material,
    study}], "skipped": [...why]}. `explore_opts` overrides the study's own
    `explore` block key by key."""
    opts = {**(study.get("explore") or {}), **(explore_opts or {})}
    if study.get("multiMaterial"):
        raise ValueError("explore runs single-material studies (a multiMaterial study already chooses materials)")
    thin = _thin_axis(study, base_dir)
    routes, skipped = [], []
    if opts.get("routes"):
        pairs = [(_method(r["method"]), r["material"]) for r in opts["routes"]]
    else:
        methods = [_method(m) for m in (opts.get("methods") or DEFAULT_METHODS)]
        mats = opts.get("materials")
        pairs = []
        for m in methods:
            if mats:
                fit = [x for x in mats if not isinstance(x, str) or _makes(x, m)]
                if not fit:
                    skipped.append({"method": m, "why": f"none of {mats} is made by {m}" + (
                        f"; its default is {DEFAULT_MATERIALS[m][0]}" if DEFAULT_MATERIALS[m] else "")})
                pairs += [(m, x) for x in fit]
            elif m == "unrestricted":
                own = study.get("materials") or [study.get("material", "pa12")]
                pairs += [(m, x) for x in (own if isinstance(own, list) else [own])]
            else:
                pairs += [(m, x) for x in DEFAULT_MATERIALS[m]]
    seen = set()
    for m, mat in pairs:
        if isinstance(mat, str) and not _makes(mat, m):
            skipped.append({"method": m, "material": mat, "why": f"{mat} is not made by {m} (its card's processes)"})
            continue
        r = _route_study(study, m, mat, opts, thin)
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        routes.append(r)
    if not routes:
        raise ValueError("no route to run: " + "; ".join(s["why"] for s in skipped))
    return {"routes": routes, "skipped": skipped, "thinAxis": thin}


def _digest(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def _emit(progress, event):
    if progress is not None:
        progress(event)


def _run_route(route: dict, folder: str, base_dir: str, quick: bool, env: dict, progress) -> dict:
    """`loadpath run` the route's study in folder/<id>; its report.json."""
    rdir = os.path.join(folder, route["id"])
    out = os.path.join(rdir, "out")
    if quick and "confirm" not in (route["study"].get("goal") or {}):
        # a first look: the body-fitted cross-check costs minutes per outcome
        # (20 min a pick on the arm link's coarse grid), the voxel reading errs safe
        route = {**route, "study": {**route["study"], "goal": {**(route["study"].get("goal") or {}),
                                                                "confirm": "voxel"}}}
    os.makedirs(rdir, exist_ok=True)
    key = _digest({"study": route["study"], "quick": quick,
                   "env": {k: env.get(k) for k in ("LOADPATH_PITCH_STARTS", "LOADPATH_QUICK_SEARCH_RUNS",
                                                   "LOADPATH_CONFIRM_CELLS")}})
    rep_path = os.path.join(out, "report.json")
    stamp = os.path.join(rdir, "digest")
    if os.path.exists(rep_path) and os.path.exists(stamp) and open(stamp).read().strip() == key:
        _emit(progress, {"stage": "route", "route": route["id"], "state": "reused"})
        with open(rep_path) as fh:
            return {"report": json.load(fh), "reused": True, "seconds": 0.0, "dir": out}
    path = os.path.join(rdir, "study.json")
    with open(path, "w") as fh:
        json.dump(route["study"], fh, indent=1)
    argv = [sys.executable, "-m", "loadpath", "run", path, "--out", out, "--viewer", "none", "--base-dir", base_dir]
    if quick:
        argv.append("--quick")
    _emit(progress, {"stage": "route", "route": route["id"], "state": "running"})
    t0 = time.time()
    with open(os.path.join(rdir, "run.log"), "wb") as log:
        code = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    secs = time.time() - t0
    if code != 0 or not os.path.exists(rep_path):
        tail = open(os.path.join(rdir, "run.log"), errors="replace").read().strip().splitlines()[-6:]
        _emit(progress, {"stage": "route", "route": route["id"], "state": "failed", "seconds": round(secs)})
        return {"report": None, "error": "\n".join(tail) or f"exit {code}", "seconds": secs, "dir": out}
    with open(stamp, "w") as fh:
        fh.write(key)
    with open(rep_path) as fh:
        rep = json.load(fh)
    dv = rep.get("deliver") or {}
    _emit(progress, {"stage": "route", "route": route["id"], "state": "done", "seconds": round(secs),
                     "mass": dv.get("mass"), "passes": dv.get("passes")})
    return {"report": rep, "seconds": secs, "dir": out}


def _row(route: dict, res: dict, quantity: int, rates: dict | None) -> dict:
    from . import costing

    row = {"id": route["id"], "method": route["method"], "material": route["material"],
           "folder": os.path.dirname(res["dir"]), "seconds": round(res.get("seconds") or 0.0, 1)}
    if res.get("reused"):
        row["reused"] = True
    rep = res.get("report")
    if rep is None:
        row.update({"passes": False, "error": res.get("error")})
        return row
    dv = rep.get("deliver") or {}
    row.update({"passes": bool(dv.get("passes")), "mass": dv.get("mass"),
                "safetyFactor": dv.get("governingSafetyFactor", dv.get("safetyFactor")),
                "safetyFactorMethod": dv.get("safetyFactorMethod"), "maxDeflection": dv.get("maxDeflection"),
                "governedBy": dv.get("governedBy"), "volumeFraction": dv.get("volumeFraction")})
    outs = rep.get("outcomes") or []
    if dv.get("outcome") is not None and dv["outcome"] < len(outs) and outs[dv["outcome"]].get("margin"):
        row["margin"] = outs[dv["outcome"]]["margin"]
    for k in ("stl", "render"):
        if dv.get(k):
            row[k] = os.path.join(res["dir"], os.path.basename(dv[k]))
    warns = dv.get("warnings") or []
    if warns:
        row["warnings"] = warns[:4]
        if len(warns) > 4:
            row["moreWarnings"] = len(warns) - 4
    if METHODS[route["method"]]["family"] is None:
        row["note"] = "unrestricted: the study as written, no manufacturing method to cost"
    elif row.get("stl") and os.path.exists(row["stl"]):
        import trimesh
        mesh = trimesh.load(row["stl"], force="mesh")
        st = route["study"]
        best = (rep.get("outcomes") or [{}])[dv.get("outcome") or 0] if rep.get("outcomes") else {}
        secs = ((best.get("slicer") or {}).get("printTime")) if isinstance(best, dict) else None
        proc = st.get("process") or {}
        build = None
        if isinstance(proc, dict) and proc.get("buildDirection") not in (None, "auto"):
            from .am import parse_build_direction
            build = parse_build_direction(proc["buildDirection"])
        mill = st.get("milling") or {}
        # a fixture set-up per side cut on 2.5/3 axes; 5 axes reach the rest from two (3+2 positioning)
        sides = 2 if mill.get("axes") == 5 else len(mill.get("directions") or []) or 1
        c = costing.estimate(route["method"], route["material"], mesh, mass=dv.get("mass"), quantity=quantity,
                             build=build, milling_sides=sides, milling_axes=mill.get("axes", 3),
                             print_seconds=secs, rates=rates,
                             machining=((best.get("manufacturing") or {}).get("machining")
                                        if isinstance(best, dict) else None))
        row["cost"] = c
        row["unitCost"] = c["unitCost"]
        row["leadDays"] = c["leadDays"]
    return row


def _front(rows: list, a: str, b: str) -> list:
    """Ids of the rows no other row beats on both a and b (lower is better)."""
    pts = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
    out = []
    for r in pts:
        if not any(o is not r and o[a] <= r[a] and o[b] <= r[b] and (o[a] < r[a] or o[b] < r[b]) for o in pts):
            out.append(r["id"])
    return sorted(out, key=lambda i: next(r[a] for r in pts if r["id"] == i))


_RANK_KEY = {"mass": "mass", "cost": "unitCost", "leadTime": "leadDays"}


def explore(study: dict, out_dir: str, explore_opts: dict | None = None, base_dir=None, jobs: int | None = None,
            quick: bool = False, progress=None) -> dict:
    """Run every route of `plan(study, explore_opts)` and return the
    comparison (also written to out_dir/explore.json)."""
    from . import costing

    opts = {**(study.get("explore") or {}), **(explore_opts or {})}
    rank = opts.get("rank", "mass")
    if rank not in RANKS:
        raise ValueError(f"explore.rank is one of {', '.join(RANKS)}")
    quantity = int(opts.get("quantity", 1))
    rates = opts.get("costs")
    base_dir = os.path.abspath(base_dir or os.getcwd())
    p = plan(study, explore_opts, base_dir)
    routes = p["routes"]
    os.makedirs(out_dir, exist_ok=True)
    folder = os.path.abspath(out_dir)
    ncpu = os.cpu_count() or 1
    env = dict(os.environ, PYTHONUNBUFFERED="1", LOADPATH_NO_HINT="1")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(v, "1")
    if opts.get("shareDesigns", True):
        env["LOADPATH_RECALL"] = os.path.join(folder, "designs")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()

    def run_all(rs, where, coarse):
        # one queue per method: a method's materials run in turn and share designs
        groups: dict[str, list] = {}
        for r in rs:
            groups.setdefault(r["method"], []).append(r)
        n = max(1, min(len(groups), int(jobs) if jobs else max(1, ncpu // 2)))
        e = dict(env, LOADPATH_WORKERS=str(max(1, ncpu // n)))
        if coarse:
            # a screen: one grid per route (resolution.pitchStarts' extra grids triple the time)
            e.setdefault("LOADPATH_PITCH_STARTS", "1")
            # five coarse runs (quick's 3 end at a loose bracket: the arm link's
            # MJF screen stopped at margin 1.5, 20% heavy)
            e.setdefault("LOADPATH_QUICK_SEARCH_RUNS", str(SCREEN_SEARCH_RUNS))
        _emit(progress, {"stage": "plan", "routes": [r["id"] for r in rs], "jobs": n, "quick": coarse,
                         "skipped": p["skipped"]})
        results: dict[str, dict] = {}

        def run_group(g):
            for r in g:
                results[r["id"]] = _run_route(r, where, base_dir, coarse, e, progress)

        with ThreadPoolExecutor(n) as ex:
            list(ex.map(run_group, sorted(groups.values(), key=len, reverse=True)))
        return [_row(r, results[r["id"]], quantity, rates) for r in rs]

    key = _RANK_KEY[rank]
    big = float("inf")
    screened = None
    if opts.get("screen") and not quick:
        # every route on the coarse grid first; the full runs only for the
        # routes on a Pareto front or near the top of the ranking
        srows = run_all(routes, os.path.join(folder, "screen"), True)
        sp = [r for r in srows if r.get("passes")]
        keep = set(_front(sp, "mass", "unitCost")) | set(_front(sp, "unitCost", "leadDays"))
        keep |= {r["id"] for r in sorted(sp, key=lambda r: r.get(key) if r.get(key) is not None else big)
                 [:int(opts.get("refineTop", 2))]}
        # an upper bound, not an answer: the coarse search ended well over the goal
        loose = {r["id"] for r in sp if (r.get("margin") or 1.0) > SCREEN_MARGIN
                 and r.get("governedBy") != "volumeFraction floor"}
        keep |= loose
        if not sp:      # nothing passed coarse: run the strongest routes properly
            keep = {r["id"] for r in sorted(srows, key=lambda r: -(r.get("safetyFactor") or 0))[:2]}
        screened = {"routes": len(srows), "refined": sorted(keep), "loose": sorted(loose & keep)}
        rows = run_all([r for r in routes if r["id"] in keep], folder, False)
        for r in srows:
            if r["id"] not in keep:
                rows.append({**r, "screenedOnly": True})
        rows.sort(key=lambda r: [x["id"] for x in routes].index(r["id"]))
    else:
        rows = run_all(routes, folder, quick)
    passing = [r for r in rows if r.get("passes") and not r.get("screenedOnly")]
    order = sorted(rows, key=lambda r: (bool(r.get("screenedOnly")), not r.get("passes"),
                                        r.get(key) if r.get(key) is not None else big))
    picks = {}
    for name, k in (("lightest", "mass"), ("cheapest", "unitCost"), ("fastest", "leadDays")):
        c = [r for r in passing if r.get(k) is not None]
        if c:
            picks[name] = min(c, key=lambda r: r[k])["id"]
    q1 = []
    for r in passing:
        if r.get("cost"):
            c = r["cost"]
            var = (c["orderCost"] - c["fixedCost"]) / c["quantity"]
            q1.append({"route": r["id"], "fixedCost": c["fixedCost"], "orderCost": var + c["fixedCost"]})
    summary = {
        "name": study.get("name"),
        "goal": study.get("goal"),
        "quantity": quantity,
        "rank": rank,
        "ranking": [r["id"] for r in order],
        "picks": picks,
        "pareto": {"massCost": _front(passing, "mass", "unitCost"),
                   "costLeadTime": _front(passing, "unitCost", "leadDays"),
                   "massLeadTime": _front(passing, "mass", "leadDays")},
        "crossover": costing.crossover(q1),
        "routes": order,
        "skipped": p["skipped"],
        **({"screen": screened} if screened else {}),
        "thinAxis": p["thinAxis"],
        "seconds": round(time.time() - t0, 1),
        "notes": ["unitCost and leadDays are service-bureau estimates (costing.py), +-50%: rank with them, quote "
                  "before buying (materialize_get_quote, a machine shop)",
                  "mass and safety factor are each route's own verified run (its folder's report.json)"],
    }
    with open(os.path.join(folder, "explore.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    summary["table"] = table(summary)
    with open(os.path.join(folder, "explore.md"), "w") as fh:
        fh.write(summary["table"] + "\n")
    return summary


def table(summary: dict) -> str:
    """The comparison as a Markdown table, ranked."""
    q = summary.get("quantity", 1)
    lines = [f"| route | passes | mass g | SF | unit $ (x{q}) | lead days | machine h |",
             "|---|---|---|---|---|---|---|"]
    for r in summary["routes"]:
        sf = r.get("safetyFactor")
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["id"] + (" (coarse)" if r.get("screenedOnly") else ""), "yes" if r.get("passes") else "no",
            f"{r['mass']:.1f}" if r.get("mass") is not None else "-",
            f"{sf:.2f}" if sf is not None else "-",
            f"{r['unitCost']:.0f}" if r.get("unitCost") is not None else "-",
            f"{r['leadDays']:.0f}" if r.get("leadDays") is not None else "-",
            f"{r['cost']['machineHours']:.1f}" if (r.get("cost") or {}).get("machineHours") else "-"))
    pk = summary.get("picks") or {}
    if pk:
        lines.append("")
        lines.append("; ".join(f"{k}: {v}" for k, v in pk.items()))
    return "\n".join(lines)
