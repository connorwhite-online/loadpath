"""
Pick a part's core for the process it will be printed by.

A stiffness-bound part like the arm link can be built several ways inside the
same envelope: a plain hollow skin, a box section with ribs, a skin over a
lattice core. Which is lightest depends on the analysis; which can be built
depends on the process. A hollow with a 26 mm flat roof over a sealed cavity
is the lightest arm link on paper and prints on SLS or MJF (nothing needs
support there, the powder comes out through escape holes), but on FDM the
roof has nothing under it and support printed inside can never come out.

`choose_core` tunes every option (tune.tune_field, each confirmed on its
fine grids), meshes each tuned part, and reads it the way each process
would build it (printcheck.check_print along `up`):

  fdm       - support trapped inside a sealed cavity (`trappedSupportArea`)
              must stay under `trappedLimit` mm^2 (default 50: a few
              stray cells of a stencil that flags every 45-degree patch).
  sls, mjf  - nothing needs support; a sealed void needs escape holes
              for the powder (`drains`), reported, not refused.

The lightest option that met its loads (on the confirming grid) and the
process wins, by its meshed mass. A tuned option is reused from a previous
run in the same folder when its graph has not changed.

    spec = {"options": [{"name": "hollow", "graph": "arm-link-hollow-tune.json"},
                        {"name": "box", "graph": "arm-link-box-tune.json"},
                        {"name": "sandwich", "graph": "arm-link-sandwich-tune.json"}],
            "processes": ["fdm", "sls"], "up": "+z"}
    rep = choose_core(spec, base_dir="examples/fields", out_dir="cores-out")
    rep["winners"]["fdm"]   # {"name": ..., "mass": ..., ...}
"""
from __future__ import annotations

import hashlib
import json
import os

__all__ = ["choose_core", "PROCESSES", "TRAPPED_LIMIT"]

PROCESSES = ("fdm", "sls", "mjf")
TRAPPED_LIMIT = 50.0


def _digest(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def _up(up):
    """'+z' -> (2, 1), the build direction check_print takes."""
    if isinstance(up, (list, tuple)):
        return up
    s = str(up).strip()
    sign = -1.0 if s.startswith("-") else 1.0
    return ({"x": 0, "y": 1, "z": 2}[s.lstrip("+-").lower()], sign)


def choose_core(spec: dict, base_dir=None, out_dir=None, progress=None, pitch: float = 0.3) -> dict:
    """Tune each of spec["options"], read each tuned part for each of
    spec["processes"] and return {"options": [...], "winners": {process:
    option or None}}. Each option: name, graph (path or graph doc with a
    `tune` block); spec: processes (default ["fdm"]), up (default "+z"),
    trappedLimit, evaluations (overrides each option's)."""
    from .fields import FieldGraph
    from .materials import material_from_spec
    from .printcheck import check_print
    from .tune import _study, tune_field

    options = spec.get("options") or []
    if len(options) < 2:
        raise ValueError("cores: give at least two options {name, graph}")
    processes = [p.lower() for p in spec.get("processes") or ["fdm"]]
    bad = [p for p in processes if p not in PROCESSES]
    if bad:
        raise ValueError(f"cores: unknown process {bad} (have {', '.join(PROCESSES)})")
    up = _up(spec.get("up", "+z"))
    trapped_limit = float(spec.get("trappedLimit", TRAPPED_LIMIT))
    rows = []
    for opt in options:
        name = opt["name"]
        doc = opt["graph"]
        gdir = base_dir
        if isinstance(doc, str):
            path = doc if os.path.isabs(doc) or base_dir is None else os.path.join(base_dir, doc)
            with open(path) as fh:
                doc = json.load(fh)
        odir = os.path.join(out_dir, name) if out_dir else None
        key = _digest({"doc": doc, "evaluations": spec.get("evaluations")})
        cached = None
        if odir and os.path.exists(os.path.join(odir, "core.json")):
            with open(os.path.join(odir, "core.json")) as fh:
                c = json.load(fh)
            if c.get("key") == key:
                cached = c
        if progress:
            progress({"event": "core-option", "name": name, "reused": cached is not None})
        if cached is None:
            out = tune_field(doc, gdir, odir, progress=progress, evaluations=spec.get("evaluations"))
            best = out["best"]
            row = {"name": name, "key": key, "tuned": out["doc"],
                   "best": None if best is None else {k: best.get(k) for k in
                                                     ("values", "mass", "safetyFactor", "deflection", "ok",
                                                      "unconfirmed", "check")}}
        else:
            row = cached
        best = row["best"]
        row["meetsLoads"] = bool(best and best.get("ok"))
        if row["meetsLoads"] and "print" not in row:
            tuned = {k: v for k, v in row["tuned"].items() if k != "tune"}
            mesh = FieldGraph(tuned, base_dir=gdir).mesh(pitch=pitch)
            dens = (material_from_spec(tuned["material"]).density if "material" in tuned
                    else _study(doc["tune"], gdir).material.density)
            row["meshedMass"] = float(abs(mesh.volume) * dens / 1000.0)
            pr = check_print(mesh, up, "fdm", pitch=0.4)
            row["print"] = {k: pr.get(k) for k in ("supportArea", "trappedSupportArea", "sealedVoidVolume",
                                                   "islands", "maxBridgeSpan", "bridges")}
            if odir:
                os.makedirs(odir, exist_ok=True)
                mesh.export(os.path.join(odir, f"{name}.stl"))
        if odir:
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "core.json"), "w") as fh:
                json.dump(row, fh, indent=1)
        row["processes"] = {}
        for proc in processes:
            if not row["meetsLoads"]:
                row["processes"][proc] = {"ok": False, "why": "no tuned candidate met the loads on the "
                                                               "confirming grid"}
                continue
            pr = row["print"]
            if proc == "fdm":
                ok = pr["trappedSupportArea"] <= trapped_limit
                why = None if ok else (f"{pr['trappedSupportArea']:.0f} mm^2 of support trapped inside a sealed "
                                       f"cavity (limit {trapped_limit:g})")
                row["processes"][proc] = {"ok": ok, "why": why}
            else:
                note = "sealed void: add escape holes (drains)" if (pr.get("sealedVoidVolume") or 0) > 1.0 else None
                row["processes"][proc] = {"ok": True, "why": None, "note": note}
        rows.append(row)
        if progress:
            progress({"event": "core-read", "name": name, "meetsLoads": row["meetsLoads"],
                      "meshedMass": row.get("meshedMass"), **{p: v["ok"] for p, v in row["processes"].items()}})
    winners = {}
    for proc in processes:
        ok = [r for r in rows if r["processes"][proc]["ok"]]
        w = min(ok, key=lambda r: r["meshedMass"]) if ok else None
        winners[proc] = None if w is None else {"name": w["name"], "meshedMass": w["meshedMass"],
                                                 "deflection": w["best"]["check"]["deflection"]
                                                 if w["best"].get("check") else w["best"]["deflection"],
                                                 "safetyFactor": w["best"]["safetyFactor"]}
    rep = {"processes": processes, "up": spec.get("up", "+z"), "trappedLimit": trapped_limit,
           "options": [{k: v for k, v in r.items() if k not in ("tuned", "key")} for r in rows],
           "winners": winners}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cores.json"), "w") as fh:
            json.dump(rep, fh, indent=1)
        for proc, w in winners.items():
            if w is not None:
                tuned = next(r["tuned"] for r in rows if r["name"] == w["name"])
                with open(os.path.join(out_dir, f"{proc}.json"), "w") as fh:
                    json.dump(tuned, fh, indent=1)
    return rep
