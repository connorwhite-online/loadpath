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
twice as fine, not wherever the search stopped.

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

    from .fields import FieldGraph, node_doc
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
    sf_min = spec.get("safetyFactor", goal.safety_factor)
    defl_max = spec.get("maxDeflection", goal.max_deflection)
    if sf_min is None and defl_max is None:
        raise ValueError("tune: no limit to hold (safetyFactor or maxDeflection, here or in the study's goal)")
    node = spec.get("node") or doc.get("output")
    pitch, cells = float(spec.get("pitch", 1.0)), int(spec.get("cells", 30_000))
    budget = int(evaluations or spec.get("evaluations", 20))

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
            g = FieldGraph(node_doc(d, node) if node else d, base_dir=base_dir)
            mesh = g.mesh(pitch=pitch, project=1)
            rep = verify_part(study, mesh, max_cells=n_cells, refine=0)
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
        if sf_min is not None:
            m.append(rec["safetyFactor"] / sf_min - 1.0)
        if defl_max is not None:
            m.append(1.0 - rec["deflection"] / max(defl_max, 1e-12))
        return min(m)

    def report_check(r, check):
        r["check"] = {k: check.get(k) for k in ("safetyFactor", "deflection", "ok", "seconds")}
        if progress:
            progress({"event": "tune-check", "mass": r.get("mass"),
                      **{k: v for k, v in r["check"].items() if v is not None},
                      "values": ", ".join(f"{v:.3g}" for v in r["values"])})

    first = evaluate((start - lo) / (hi - lo))
    if "error" not in first:
        fine = analyze(first["values"], 2 * cells)
        if "error" not in fine:
            for k in bias:
                c = first["coarse"][k]
                bias[k] = fine[k] / c if c > 0 else 1.0
            judge(first)
            report_check(first, fine)
    m0 = first.get("mass") or 1.0
    cons = [{"type": "ineq", "fun": lambda x: margin(evaluate(x))}]

    def objective(x):
        rec = evaluate(x)
        return rec["mass"] / m0 if "mass" in rec else 10.0

    x0 = np.clip((start - lo) / (hi - lo), 0.0, 1.0)
    # COBYQA keeps every candidate inside the knobs' ranges and spends one
    # analysis per point (objective and limits come from the same one, cached)
    minimize(objective, x0, method="COBYQA", bounds=[(0.0, 1.0)] * len(params), constraints=cons,
             options={"initial_tr_radius": 0.25, "final_tr_radius": 5e-3, "maxfev": max(budget, 2)})
    # The ratio holds only roughly away from the start, and a voxel reading
    # moves a few percent as the part slides across its grid: the lightest
    # few that passed are analyzed again on the fine grid, and the first that
    # still passes is the answer.
    good = sorted((r for r in history if r["ok"]), key=lambda r: r["mass"])
    best = None
    for r in good[:3]:
        if r is first and "check" in r:
            check = r["check"]
        else:
            check = analyze(r["values"], 2 * cells)
            report_check(r, check)
        if check["ok"]:
            best = r
            break
    if best is None and good:
        best = {**good[0], "unconfirmed": True}     # passed on the coarse grid, not on the fine one
    out = {"best": best, "start": first, "history": history,
           "limits": {"safetyFactor": sf_min, "maxDeflection": defl_max},
           "params": [{"path": p["path"], "min": p["min"], "max": p["max"]} for p in params]}
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
