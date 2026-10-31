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
twice as fine, not wherever the search stopped. A part with a lattice in
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

__all__ = ["tune_field", "get_at", "set_at"]


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


def _study(spec, base_dir):
    from .spec import study_from_dict

    s = spec.get("study")
    if not isinstance(s, str):
        raise ValueError("tune needs a 'study' (a study JSON path or example:<name>) to load the part with")
    if s.startswith("example:"):
        from .tasks import example

        return study_from_dict(example(s.split(":", 1)[1]), base_dir=base_dir)
    sp = s if os.path.isabs(s) else os.path.join(base_dir or ".", s)
    with open(sp) as fh:
        return study_from_dict(json.load(fh), base_dir=os.path.dirname(sp))


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
    study = _study(spec, base_dir)
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
    budget = int(evaluations or spec.get("evaluations", 20))

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
                rep = verify_part(study, g.mesh(pitch=pitch, project=1), max_cells=n_cells, refine=0)
            resources.keep(g, dd)
            rec.update(mass=rep["mass"], safetyFactor=rep["safetyFactor"],
                       deflection=max((c.get("deflection") or 0.0) for c in rep["cases"]),
                       bodies=rep["bodies"])
            rec["ok"] = margin(rec) >= 0
        except Exception as exc:  # noqa: BLE001 - a knob setting that breaks the part is a bad candidate
            rec.update(error=str(exc), ok=False)
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

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
        r["check"] = {k: check.get(k) for k in ("safetyFactor", "deflection", "ok", "seconds")}
        if progress:
            progress({"event": "tune-check", "mass": r.get("mass"),
                      **{k: v for k, v in r["check"].items() if v is not None},
                      "values": ", ".join(f"{v:.3g}" for v in r["values"])})

    first, first_fine = evaluate((start - lo) / (hi - lo)), None
    if "error" not in first:
        fine = analyze(first["values"], 2 * cells)
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

    def search(x0):
        # COBYQA keeps every candidate inside the knobs' ranges and spends
        # one analysis per point (objective and limits come from the same
        # one, cached across searches)
        minimize(objective, x0, method="COBYQA", bounds=[(0.0, 1.0)] * len(params), constraints=cons,
                 options={"initial_tr_radius": 0.25, "final_tr_radius": 5e-3, "maxfev": max(budget, 2)})

    checks = {}

    def pick():
        """The lightest candidate that meets the current limits on the fine
        grid too. The coarse/fine ratio holds only roughly away from the
        start, and a voxel reading moves a few percent as the part slides
        across its grid: the lightest few that pass are analyzed again on
        twice the cells, and the first that still passes is the answer."""
        good = sorted((r for r in history if "error" not in r and margin(r) >= 0), key=lambda r: r["mass"])
        for r in good[:3]:
            key = tuple(r["values"])
            if key not in checks:
                checks[key] = first_fine if r is first and first_fine is not None else analyze(r["values"], 2 * cells)
            check = checks[key]
            report_check(r, {**check, "ok": "error" not in check and margin(check) >= 0})
            if r["check"]["ok"]:
                return {**r, "ok": True}
        return {**good[0], "ok": True, "unconfirmed": True} if good else None    # passed coarse, not fine

    def x_of(r):
        return (np.asarray(r["values"]) - lo) / (hi - lo)

    search(np.clip(x_of(first), 0.0, 1.0))
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
