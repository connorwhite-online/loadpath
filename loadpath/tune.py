"""
Tune a field graph's numbers against a simulation of the part it makes.

A field graph is a recipe with knobs (a web's height, a fillet, a shell's
thickness, a lattice's wall). `tune_field` turns chosen knobs until the part
is as light as it can be while it still carries the study's loads: every
candidate is meshed, analyzed by `verify_part` (the same FE check a study's
outcome gets) and scored by mass, worst-case safety factor and deflection.
The search is COBYQA over the knobs scaled to [0, 1] (derivative-free: an
analysis on a voxel grid is not smooth in its inputs), and what comes back
is the lightest candidate that met every limit, checked again on a grid
twice as fine (and, for a part with a thin shell, on two grids with at
least two cells through its thinnest wall, CELLS_THROUGH_WALL), not
wherever the search stopped. A candidate that fails that check is never
returned as ok. A part with a lattice in
it is analyzed with the lattice homogenized (homogenized.py): its walls are
far finer than a grid a search can afford.

The spec lives in the graph under `tune`:

    "tune": {"study": "example:arm-link",
             "params": [{"path": "nodes/web/shape/max/2", "also": ["-nodes/web/shape/min/2"],
                         "min": 3, "max": 9}],
             "safetyFactor": 3, "maxDeflection": 0.5, "evaluations": 20}

`path` is where the number sits in the graph ("/"-separated, the same keys
the editor's sliders use); `also` are numbers that move with it, "-" in
front for ones that move the other way (a box's min when its max grows).
Limits default to the study's goal.
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

__all__ = ["tune_field", "get_at", "set_at", "thinnest_wall", "check_cells"]

# A thin wall read on a grid about one cell through it moves 10-35% as the
# wall slides across the grid (the arm-link sandwich: 0.554 mm on 120k
# cells, 0.748 on 240k, 0.569 on 500k, 0.580 on 1M). From about two cells
# through the thinnest wall the readings settle to a few percent, so the
# confirming check runs there, on two grids, and both must pass.
CELLS_THROUGH_WALL = 2.0
MAX_CHECK_CELLS = 2_000_000


def _keys(path):
    neg = path.startswith("-")
    keys = [int(k) if k.lstrip("-").isdigit() else k for k in path.lstrip("-").split("/")]
    return neg, keys


def get_at(doc, path):
    neg, keys = _keys(path)
    v = doc
    for k in keys:
        try:
            v = v[k]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"tune: nothing at {path!r}") from None
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ValueError(f"tune: {path!r} is not a number ({v!r})")
    return -float(v) if neg else float(v)


def set_at(doc, path, value):
    neg, keys = _keys(path)
    v = doc
    for k in keys[:-1]:
        v = v[k]
    v[keys[-1]] = -float(value) if neg else float(value)


def _study(spec, base_dir, material=None):
    from .fields import load_study

    s = spec.get("study")
    if not isinstance(s, str):
        raise ValueError("tune needs a 'study' (a study JSON path or example:<name>) to load the part with")
    return load_study(s, base_dir, material)


def tune_field(doc, base_dir=None, out_dir=None, progress=None, evaluations=None):
    """Tune `doc["tune"]["params"]` for the lightest part that meets the
    limits. Returns {"best", "start", "history", "params", "doc"} where
    `doc` is the graph with the best values written in (and `tune` kept);
    with `out_dir`, also writes tuned.json and tune.json there."""
    from scipy.optimize import minimize

    from .editor import _Resources
    from .fields import FieldGraph, expand_groups, node_doc
    from .homogenized import homogenized_check, lattice_nodes
    from .verify import verify_part

    spec = doc.get("tune")
    if not isinstance(spec, dict):
        raise ValueError("the graph has no 'tune' block (study, params, limits)")
    params = spec.get("params") or []
    if not params:
        raise ValueError("tune: give at least one param {path, min, max}")
    for p in params:
        if not (isinstance(p.get("min"), (int, float)) and isinstance(p.get("max"), (int, float))
                and p["max"] > p["min"]):
            raise ValueError(f"tune: param {p.get('path')!r} needs min < max")
        for q in [p["path"], *p.get("also", [])]:
            get_at(doc, q)
    study = _study(spec, base_dir, doc.get("material"))
    goal = study.goal
    lim = {"safetyFactor": spec.get("safetyFactor", goal.safety_factor),
           "maxDeflection": spec.get("maxDeflection", goal.max_deflection)}
    if lim["safetyFactor"] is None and lim["maxDeflection"] is None:
        raise ValueError("tune: no limit to hold (safetyFactor or maxDeflection, here or in the study's goal)")
    sweep = spec.get("sweep") or {}
    if sweep and (len(sweep) != 1 or next(iter(sweep)) not in lim
                  or not all(isinstance(v, (int, float)) and v > 0 for v in next(iter(sweep.values())))):
        raise ValueError("tune.sweep is one limit and its values: {\"maxDeflection\": [0.4, 0.6]} "
                         "or {\"safetyFactor\": [2, 4]}")
    node = spec.get("node") or doc.get("output")
    pitch, cells = float(spec.get("pitch", 1.0)), int(spec.get("cells", 30_000))
    max_check = int(spec.get("checkCells", MAX_CHECK_CELLS))
    from . import quick
    budget = quick.tune_evaluations(int(evaluations or spec.get("evaluations", 20)))

    # a lattice's walls are finer than any analysis grid a search can
    # afford: a part with one is analyzed with the lattice homogenized
    analysis = spec.get("analysis", "auto")
    if analysis not in ("auto", "solid", "homogenized"):
        raise ValueError("tune.analysis must be auto, solid or homogenized")
    probe = FieldGraph(node_doc(doc, node) if node else doc, base_dir=base_dir)
    homogenized = analysis == "homogenized" or (analysis == "auto" and bool(lattice_nodes(probe)))
    resources = _Resources()      # a `result` node's study runs once, not once per candidate
    lo = np.array([p["min"] for p in params], float)
    hi = np.array([p["max"] for p in params], float)
    start = np.array([get_at(doc, p["path"]) for p in params])
    # the other numbers keep their offset from the param (a box's min tracks its max)
    offsets = [[get_at(doc, q) - s for q in p.get("also", [])] for p, s in zip(params, start)]
    base = {k: v for k, v in doc.items() if k != "tune"}

    def apply(values):
        d = copy.deepcopy(base)
        for p, v, offs in zip(params, values, offsets):
            set_at(d, p["path"], v)
            for q, o in zip(p.get("also", []), offs):
                set_at(d, q, v + o)
        return d

    history, memo = [], {}
    # The search runs on a coarse grid, whose readings drift from a finer
    # one's by a few percent (more on thin sections). The start is analyzed
    # on both and every coarse reading is scaled by that ratio before it is
    # held to the limits.
    bias = {"safetyFactor": 1.0, "deflection": 1.0}

    def judge(rec):
        if "error" in rec:
            rec["ok"] = False
            return rec
        raw = rec.setdefault("coarse", {"safetyFactor": rec["safetyFactor"], "deflection": rec["deflection"]})
        rec["safetyFactor"] = raw["safetyFactor"] * bias["safetyFactor"]
        rec["deflection"] = raw["deflection"] * bias["deflection"]
        rec["ok"] = margin(rec) >= 0
        return rec

    def evaluate(x):
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        key = tuple(np.round(x, 6))
        if key in memo:
            return memo[key]
        rec = judge(analyze(lo + x * (hi - lo), cells))
        history.append(rec)
        memo[key] = rec
        if progress:
            progress({"event": "tune", "evaluation": len(history), "of": budget,
                      **{k: v for k, v in rec.items() if isinstance(v, (int, float, bool, str))},
                      "values": ", ".join(f"{v:.3g}" for v in rec["values"])})
        return rec

    def analyze(values, n_cells):
        """mass, worst safety factor and deflection of the part at these
        values, uncorrected; `ok` against the limits as read."""
        t0 = time.time()
        d = apply(values)
        rec = {"values": [float(v) for v in values]}
        try:
            dd = node_doc(d, node) if node else expand_groups(d)
            g = resources.graph(dd, base_dir)
            if homogenized:
                rep = homogenized_check(dd, study, base_dir=base_dir, cells=n_cells, graph=g)
            else:
                rep = verify_part(study, g.mesh(pitch=pitch, project=1), max_cells=n_cells, refine=0, fractional=True)
            resources.keep(g, dd)
            rec.update(mass=rep["mass"], safetyFactor=rep["safetyFactor"],
                       deflection=max((c.get("deflection") or 0.0) for c in rep["cases"]),
                       bodies=rep["bodies"])
            rec["ok"] = margin(rec) >= 0
        except Exception as exc:  # noqa: BLE001 - a knob setting that breaks the part is a bad candidate
            rec.update(error=str(exc), ok=False)
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    def confirm(values, rec=None):
        """The candidate on the confirming grid (check_cells): twice the
        search's cells, more for a thin shell, and then on a second grid a
        little coarser too, so a wall that happens to sit well on one grid
        can't pass; the worse reading of the two counts."""
        try:
            d = apply(values)
            dd = node_doc(d, node) if node else expand_groups(d)
            g = resources.graph(dd, base_dir)
            vol = None
            if not homogenized and rec is not None and "mass" in rec:
                vol = rec["mass"] * 1000.0 / float(study.material.density)
            n, wall = check_cells(g, None, cells, homogenized, volume=vol, cap=max_check)
        except Exception as exc:  # noqa: BLE001
            return {"values": [float(v) for v in values], "error": str(exc), "ok": False}
        out = analyze(values, n)
        out["cells"], out["thinnestWall"] = [n], wall
        if wall is not None and n > 2 * cells and "error" not in out:
            other = analyze(values, int(0.7 * n))
            out["cells"].append(int(0.7 * n))
            if "error" in other:
                return {**out, "error": other["error"], "ok": False}
            out["safetyFactor"] = min(out["safetyFactor"], other["safetyFactor"])
            out["deflection"] = max(out["deflection"], other["deflection"])
            out["seconds"] = round(out["seconds"] + other["seconds"], 1)
            out["ok"] = margin(out) >= 0
        return out

    def margin(rec):
        """>= 0 when every limit holds; the worst limit's relative slack."""
        if "error" in rec:
            return -1.0
        m = [] if rec["bodies"] == 1 else [-0.5]
        if lim["safetyFactor"] is not None:
            m.append(rec["safetyFactor"] / lim["safetyFactor"] - 1.0)
        if lim["maxDeflection"] is not None:
            m.append(1.0 - rec["deflection"] / max(lim["maxDeflection"], 1e-12))
        return min(m)

    def report_check(r, check):
        r["check"] = {k: check.get(k) for k in ("safetyFactor", "deflection", "ok", "seconds", "cells",
                                                 "thinnestWall")}
        if progress:
            progress({"event": "tune-check", "mass": r.get("mass"),
                      **{k: v for k, v in r["check"].items() if v is not None},
                      "values": ", ".join(f"{v:.3g}" for v in r["values"])})

    first, first_fine = evaluate((start - lo) / (hi - lo)), None
    if "error" not in first:
        fine = confirm(first["values"], first)
        if "error" not in fine:
            for k in bias:
                c = first["coarse"][k]
                bias[k] = fine[k] / c if c > 0 else 1.0
            judge(first)
            first_fine = fine
    m0 = first.get("mass") or 1.0
    cons = [{"type": "ineq", "fun": lambda x: margin(evaluate(x))}]

    def objective(x):
        rec = evaluate(x)
        return rec["mass"] / m0 if "mass" in rec else 10.0

    def search(x0, fev=None):
        # COBYQA keeps every candidate inside the knobs' ranges and spends
        # one analysis per point (objective and limits come from the same
        # one, cached across searches)
        minimize(objective, x0, method="COBYQA", bounds=[(0.0, 1.0)] * len(params), constraints=cons,
                 options={"initial_tr_radius": 0.25, "final_tr_radius": 5e-3, "maxfev": max(fev or budget, 2)})

    checks = {}

    def learn(r, check):
        """A candidate that passed on the search's grid failed on the
        confirming one: the search's correction was too kind there. Take
        the ratio this check saw, if it is worse than the one in use, and
        judge every candidate again by it."""
        raw = r.get("coarse")
        if not raw or "error" in check:
            return
        for k, worse in (("deflection", max), ("safetyFactor", min)):
            if raw.get(k) and check.get(k):
                bias[k] = worse(bias[k], check[k] / raw[k])
        for h in history:
            judge(h)

    def pick(tries: int = 4):
        """The lightest candidate that meets the current limits on the fine
        grid too. The coarse/fine ratio holds only roughly away from the
        start, and a voxel reading moves a few percent as the part slides
        across its grid: the lightest that passes is analyzed again on the
        confirming grid (confirm); if it fails there, the ratio it showed
        is learned (learn) and the next lightest that still passes is
        tried, up to `tries` checks."""
        for _ in range(tries):
            good = sorted((r for r in history if "error" not in r and margin(r) >= 0), key=lambda r: r["mass"])
            r = next((g for g in good if tuple(g["values"]) not in checks), None)
            if r is None:
                break
            key = tuple(r["values"])
            checks[key] = first_fine if r is first and first_fine is not None else confirm(r["values"], r)
            check = checks[key]
            report_check(r, {**check, "ok": "error" not in check and margin(check) >= 0})
            if r["check"]["ok"]:
                return {**r, "ok": True}
            learn(r, check)
        for key, check in checks.items():
            # an earlier check of a candidate that still passes
            r = next((h for h in history if tuple(h["values"]) == key), None)
            if r is not None and "error" not in check and margin(check) >= 0:
                return {**r, "ok": True}
        # passed on the search's grid, not on the confirming one: not a
        # result to build, so not ok
        good = sorted((r for r in history if "error" not in r and margin(r) >= 0), key=lambda r: r["mass"])
        return {**good[0], "ok": False, "unconfirmed": True} if good else None

    def x_of(r):
        return (np.asarray(r["values"]) - lo) / (hi - lo)

    search(np.clip(x_of(first), 0.0, 1.0))
    best = pick()
    if best is not None and best.get("unconfirmed"):
        # the corrections learned from the failed checks move the limit:
        # search again from the lightest candidate that meets it now
        near = [r for r in history if "error" not in r and margin(r) >= 0]
        x0 = x_of(min(near, key=lambda r: r["mass"])) if near else x_of(first)
        if progress:
            progress({"event": "tune-research", "bias": dict(bias)})
        search(np.clip(x0, 0.0, 1.0), fev=max(budget // 2, 4))
        best = pick()
    out = {"best": best, "start": first, "analysis": "homogenized" if homogenized else "solid",
           "limits": dict(lim), "params": [{"path": p["path"], "min": p["min"], "max": p["max"]} for p in params]}
    if sweep:
        # the trade-off: the same search held to each value of one limit in
        # turn, each warm-started from the lightest candidate that already
        # meets it (or the start)
        key, values = next(iter(sweep.items()))
        base_limit, out["sweep"] = lim[key], []
        for v in values:
            lim[key] = float(v)
            near = [r for r in history if "error" not in r and margin(r) >= 0]
            x0 = x_of(min(near, key=lambda r: r["mass"])) if near else x_of(first)
            if progress:
                progress({"event": "tune-sweep", key: float(v)})
            search(np.clip(x0, 0.0, 1.0))
            out["sweep"].append({key: float(v), "best": pick()})
        lim[key] = base_limit
    out["history"] = history

    def assign(values):
        """Every number these values set, as written into the graph."""
        d = apply(values)
        return [[q.lstrip("-"), get_at(d, q.lstrip("-"))] for p in params for q in [p["path"], *p.get("also", [])]]

    out["front"] = [{**r, "assign": assign(r["values"])} for r in _front(history)]
    for sp in out.get("sweep", []):
        if sp["best"] is not None:
            sp["best"]["assign"] = assign(sp["best"]["values"])
    if best is not None:
        best["assign"] = assign(best["values"])
    tuned = doc if best is None else {**apply(best["values"]), "tune": doc["tune"]}
    out["doc"] = tuned
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tuned.json"), "w") as fh:
            json.dump(tuned, fh, indent=2)
            fh.write("\n")
        with open(os.path.join(out_dir, "tune.json"), "w") as fh:
            json.dump({k: v for k, v in out.items() if k != "doc"}, fh, indent=2)
            fh.write("\n")
    return out


def thinnest_wall(graph, name=None, samples: int = 20_000):
    """The thinnest shell wall (mm) in the part: every `shell` node the
    output depends on, its thickness a number or the least its field takes
    over the part's box. None when the part has no shell."""
    name = name or graph.output
    b = graph.bounds(name)
    pts = None
    seen, todo, walls = set(), [name], []
    while todo:
        k = todo.pop()
        if k in seen or k not in graph.nodes:
            continue
        seen.add(k)
        todo.extend(graph.deps.get(k, []))
        nd = graph.nodes[k]
        if nd.op != "shell":
            continue
        ref = nd.refs.get("thickness", nd.spec.get("thickness"))
        if isinstance(ref, (int, float)) and not isinstance(ref, bool):
            walls.append(float(ref))
        elif isinstance(ref, str) and b is not None:
            if pts is None:
                pts = np.random.default_rng(0).uniform(b[0], b[1], (samples, 3))
            t = np.asarray(graph.evaluate(pts, ref), float)
            t = t[np.isfinite(t) & (t > 0)]
            if t.size:
                walls.append(float(t.min()))
    return min(walls) if walls else None


def check_cells(graph, name, cells: int, homogenized: bool, volume: float | None = None,
                cap: int = MAX_CHECK_CELLS):
    """Analysis cells for the confirming check: twice the search's, and
    enough for CELLS_THROUGH_WALL cells through the thinnest shell wall
    (capped at `cap`). Returns (cells, thinnest wall or None)."""
    wall = thinnest_wall(graph, name)
    n = 2 * cells
    if wall is not None:
        pitch = wall / CELLS_THROUGH_WALL
        if homogenized:
            lo, hi = graph.bounds(name)
            need = float(np.prod(np.asarray(hi) - np.asarray(lo))) * 0.6 / pitch ** 3
        else:
            need = (volume or 0.0) / pitch ** 3
        n = max(n, int(min(need, cap)))
    return n, wall


def _front(history):
    """The candidates no other beats on mass, deflection and safety factor
    all at once (coarse readings, corrected), lightest first."""
    rs = [r for r in history if "error" not in r and r.get("bodies") == 1]
    out = []
    for r in rs:
        if not any(o["mass"] <= r["mass"] and o["deflection"] <= r["deflection"]
                   and o["safetyFactor"] >= r["safetyFactor"]
                   and (o["mass"], o["deflection"], -o["safetyFactor"]) != (r["mass"], r["deflection"], -r["safetyFactor"])
                   for o in rs):
            out.append({k: r[k] for k in ("values", "mass", "deflection", "safetyFactor")})
    return sorted(out, key=lambda r: r["mass"])
