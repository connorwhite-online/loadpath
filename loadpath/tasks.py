"""
The jobs behind the CLI and the MCP server, besides the study itself: a
lattice infill of a finished part, a heat exchanger, and sizing one. Each
takes plain JSON and an output folder and returns a JSON-able report, so the
command line, the MCP server and Python callers share one code path.

    python -m loadpath lattice study.json part.stl --out out/
    python -m loadpath heatx exchanger.json --out out/
"""
from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["jsonable", "check_study", "lattice_part", "heat_exchanger", "size_heat_exchanger",
           "examples", "example"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(_HERE, "examples")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def _emit(on_progress, **ev):
    if on_progress is not None:
        on_progress(ev)


def check_study(raw, base_dir: str | None = None) -> dict:
    """Lint against the schema (every problem at once), then parse and build
    the voxel problem. {"ok": False, "error", "issues"} or {"ok": True,
    "pitch", "cells", ..., "warnings"}."""
    from .domain import build_domain
    from .schema import lint
    from .simp import filter_radius
    from .spec import study_from_dict

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"study is not valid JSON: {exc}"}
    issues = lint(raw, "study")
    errors = [i for i in issues if i["level"] == "error"]
    if errors:
        return {"ok": False, "error": "; ".join(f"{i['path']}: {i['message']}" for i in errors),
                "issues": issues, "schema": "loadpath schema study (MCP: loadpath_schema)"}
    try:
        s = study_from_dict(raw, base_dir=base_dir or os.getcwd())
        d = build_domain(s)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {"ok": False, "error": str(exc) or type(exc).__name__, "issues": issues}
    warnings = [f"{i['path']}: {i['message']}" for i in issues] + list(d.warnings)
    return jsonable({
        "ok": True, "name": s.name, "pitch": d.pitch, "cells": d.n_elem, "dims": list(d.dims),
        "preservedCells": int(d.preserve.sum()), "filterRadius": filter_radius(d, s.length_scale, s.min_hole),
        "supportNodes": int(d.support_nodes.sum()), "loadNodes": int(d.load_nodes.sum()),
        "regions": [r.id for r in s.preserve] + [r.id for r in s.keep_out],
        "loadCases": [c.name for c in s.load_cases], "material": s.material.name,
        "minMember": s.min_member, "warnings": warnings,
    })


# --- lattice infill ---------------------------------------------------------

def _lattice_options(spec, study):
    from .lattice import LatticeOptions
    from .schema import format_issues, lint

    spec = dict(spec or {})
    errors = [i for i in lint(spec, "lattice") if i["level"] == "error"]
    if errors:
        raise ValueError("lattice options: " + format_issues(errors))
    return LatticeOptions.from_spec(spec, study.process, study.goal.safety_factor)


def lattice_part(study, mesh, options: dict | None, out_dir: str, stem: str, render: bool = True,
                 on_progress=None) -> dict:
    """Fill `mesh` (a finished part for `study`) with a TPMS lattice graded by
    the part's own stress, verify the result and write <stem>-lattice.stl
    (and .png). Returns the report."""
    from .lattice import lattice_infill
    from .verify import verify_part

    o = _lattice_options(options, study)
    _emit(on_progress, stage="lattice-stress", note="analyzing the solid part")
    solid = verify_part(study, mesh, refine=0, fields=o.target_sf is None)
    _emit(on_progress, stage="lattice-build", type=o.kind, cell=o.cell,
          sizing="homogenized" if o.target_sf else "solid-stress")
    lat, info = lattice_infill(study, mesh, solid.get("fields"), o)
    _emit(on_progress, stage="lattice-verify")
    rep = verify_part(study, lat, refine=0)
    os.makedirs(out_dir, exist_ok=True)
    stl = os.path.join(out_dir, stem + "-lattice.stl")
    lat.export(stl)
    out = {"stl": stl, "lattice": info, "massGrams": info["mass"],
           "solidMassGrams": solid["mass"], "safetyFactor": rep.get("safetyFactor"),
           "solidSafetyFactor": solid.get("safetyFactor"), "bodies": rep.get("bodies"),
           "keepOutIntrusion": rep.get("keepOutIntrusion"), "preserveMissing": rep.get("preserveMissing"),
           "enclosedCavityVolume": rep.get("enclosedCavityVolume"),
           "cases": [{"name": c["name"], "peakStressMPa": c["peakStress"], "safetyFactor": c["safetyFactor"],
                      "deflectionMm": c["deflection"]} for c in rep.get("cases") or []]}
    attention = []
    target = study.goal.safety_factor
    sf = out["safetyFactor"]
    if target and sf is not None and sf < target:
        attention.append(f"safety factor {sf:.2f} is under the goal's {target:g}: raise safetyFactor, minWall "
                         "or maxWall, or use a smaller cell")
    if (out["bodies"] or 1) > 1:
        attention.append(f"{out['bodies']} separate bodies")
    out["attention"] = attention
    out["printability"] = info.get("printability")
    if info.get("enclosedVoid", 0) > 1.0 and study.process is not None and study.process.kind != "fdm":
        out["warning"] = (f"{info['enclosedVoid']:.0f} mm^3 of lattice is still walled in, which traps "
                          f"{study.process.kind} powder or resin; set drains to 'yes', or skin to 0")
    if render:
        from .render import render_views
        png = os.path.join(out_dir, stem + "-lattice.png")
        sf = out["safetyFactor"]
        render_views(lat, png, study, title=f"{stem} lattice {o.kind} {o.cell:g}mm  {info['mass']:.1f} g"
                     + (f"  SF {sf:.2f}" if sf is not None else ""))
        out["render"] = png
    return jsonable(out)


# --- heat exchangers --------------------------------------------------------

_HX_OWN = ("name", "description", "step", "duty", "expect")


def _hx_spec(spec, base_dir):
    from .heatx import heat_exchanger_from_spec
    from .schema import format_issues, lint

    errors = [i for i in lint(spec, "heatExchanger") if i["level"] == "error"]
    if errors:
        raise ValueError(format_issues(errors))
    hx = heat_exchanger_from_spec({k: v for k, v in spec.items() if k not in _HX_OWN}, base_dir)
    if "step" in spec:
        hx.step = float(spec["step"])
    return hx


def _duty(spec, hx):
    duty = spec.get("duty")
    if not duty:
        return None
    missing = [n for n in hx.names if n not in duty]
    if missing:
        raise ValueError(f"duty needs an entry for each fluid {list(hx.names)}; missing {missing}")
    return duty


def heat_exchanger(spec: dict, out_dir: str, base_dir: str | None = None, render: bool = True,
                   on_progress=None, viewer: bool = True) -> dict:
    """Design a heat exchanger, check it, rate it if `duty` is given (and
    simulate it whole with "simulate": true; porous.py), and write <name>.stl, <name>.png, the fluid spaces (<name>-<fluid>.stl) and
    an interactive viewer (<name>.html)."""
    from .heatx import design_heat_exchanger
    from .heatx_thermal import rate_design

    hx = _hx_spec(spec, base_dir)
    duty = _duty(spec, hx)
    name = str(spec.get("name") or "heat-exchanger")
    _emit(on_progress, stage="design", type=hx.kind, cell=hx.cell, wall=hx.wall, layout=hx.layout)
    sim_opts = spec.get("simulate")
    sim_opts = ({} if sim_opts is True else dict(sim_opts)) if sim_opts and duty is not None else None
    mesh, report, spaces, grids = design_heat_exchanger(hx, fluid_meshes=True, grids=True)
    if sim_opts is None:
        del grids
    os.makedirs(out_dir, exist_ok=True)
    stl = os.path.join(out_dir, name + ".stl")
    mesh.export(stl)
    out = {"name": name, "stl": stl, "report": report, "fluidSpaces": {}}
    for fl, m in spaces.items():
        p = os.path.join(out_dir, f"{name}-{fl}.stl")
        m.export(p)
        out["fluidSpaces"][fl] = p
    ok = bool(report.get("isolated")) and all((report.get("connected") or {}).values())
    out["passes"] = ok and bool(report.get("watertight"))
    if duty is not None:
        _emit(on_progress, stage="rate")
        try:
            out["rating"] = rate_design(hx, report, duty)
        except ValueError as exc:
            out["ratingError"] = str(exc)
    if sim_opts is not None:
        _emit(on_progress, stage="simulate")
        from .porous import plot_fields, simulate_design
        try:
            sim = simulate_design(hx, report, grids, duty, sim_opts.get("voxel"), sim_opts.get("closures", "design"),
                                  fields=render)
            f = sim.pop("_fields", None)
            if f is not None:
                png = os.path.join(out_dir, name + "-simulation.png")
                plot_fields(f, png, axis=hx.build if isinstance(hx.build, int) else 2,
                            title=f"{name}: whole-exchanger simulation, middle of the core")
                sim["render"] = png
            out["simulation"] = sim
        except ValueError as exc:
            out["simulationError"] = str(exc)
        del grids
    if render:
        _emit(on_progress, stage="render")
        from .render import render_heat_exchanger
        png = os.path.join(out_dir, name + ".png")
        render_heat_exchanger(mesh, spaces, hx.core, png, title=f"{name}  {hx.kind} {hx.cell:g}/{hx.wall:g}mm  "
                              f"{hx.layout}")
        out["render"] = png
    if viewer:
        from .viewer import heat_exchanger_viewer
        html = os.path.join(out_dir, name + ".html")
        heat_exchanger_viewer(mesh, spaces, report, html, title=name)
        out["viewer"] = html
    return jsonable(out)


def size_heat_exchanger(spec: dict, base_dir: str | None = None, max_pressure_drop: dict | None = None,
                        min_wall: float = 0.8, objective: str = "effectiveness",
                        min_effectiveness: float | None = None, splits=None) -> dict:
    """Sweep cell and wall for the core and duty in `spec` before designing
    it (seconds, not minutes): the candidates that meet the pressure-drop
    limits ({fluid: Pa}) and `min_effectiveness`, best effectiveness first,
    or least metal first with objective "metal". `splits` (the first
    fluid's share of the void) sweeps the sheet's offset too; default the
    spec's split. Flow lengths are the core's extent along each fluid's
    axis."""
    from .heatx_thermal import size_core
    from .shapes import manifold_from_trimesh  # noqa: F401  (manifold3d present)

    hx = _hx_spec(spec, base_dir)
    duty = _duty(spec, hx)
    if duty is None:
        raise ValueError("sizing needs a duty: {\"A\": {\"fluid\": \"water\", \"lPerMin\": 2, \"tIn\": 80}, "
                         "\"B\": {...}, \"wall\": \"petg\"}")
    core_volume = float(hx.core.to_manifold().volume())
    lo, hi = (np.asarray(b, float) for b in hx.core.bounds())
    ext = hi - lo
    a0, a1 = (a if isinstance(a, (int, np.integer)) else "xyz".index(a) for a in hx.axes)
    lengths = [float(ext[a0]), float(ext[a0] if hx.layout == "counter" else ext[a1])]
    duty = {**{n: duty[n] for n in hx.names}, **({"wall": duty["wall"]} if "wall" in duty else {})}
    res = size_core(hx.kind, core_volume, lengths, duty, "counter" if hx.layout == "counter" else "cross",
                    hx.port_diameter, min_wall=min_wall, max_pressure_drop=max_pressure_drop,
                    split=splits if splits is not None else hx.split, objective=objective,
                    min_effectiveness=min_effectiveness)
    res.update(coreVolume=core_volume, flowLengths=dict(zip(hx.names, lengths)),
               next="Put the best cell, wall and split in the spec and run it "
                    "(loadpath heatx / loadpath_heat_exchanger).")
    return jsonable(res)


# --- examples ---------------------------------------------------------------

def examples() -> list[dict]:
    """The packaged example documents: name, kind and description."""
    out = []
    for f in sorted(os.listdir(_EXAMPLES)):
        if f.endswith(".json"):
            with open(os.path.join(_EXAMPLES, f)) as fh:
                d = json.load(fh)
            out.append({"name": f[:-5], "kind": "heatExchanger" if "core" in d else "study",
                        "description": d.get("description", "")})
    return out


def example(name: str) -> dict:
    path = os.path.join(_EXAMPLES, os.path.basename(name).removesuffix(".json") + ".json")
    if not os.path.exists(path):
        raise ValueError(f"no example {name!r}; known: {[e['name'] for e in examples()]}")
    with open(path) as fh:
        return json.load(fh)
