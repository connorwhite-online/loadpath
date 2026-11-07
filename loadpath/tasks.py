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

__all__ = ["jsonable", "check_study", "check_document", "check_heat_exchanger", "lattice_part", "heat_exchanger",
           "size_heat_exchanger", "examples", "example"]

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


_HX_KEYS = ("core", "envelope", "portDiameter", "fluids", "layout", "zones", "cleanouts")
_LATTICE_KEYS = ("minWall", "maxWall", "solidAbove", "collar", "fillPockets", "drainDiameter")


def document_kind(raw) -> str:
    """Which kind of document this is: "study", "heatExchanger", "lattice"
    (lattice options) or "field" (a field graph), from its keys."""
    if not isinstance(raw, dict):
        return "study"
    if "designSpace" in raw or "loadCases" in raw or "loads" in raw:
        return "study"
    if "nodes" in raw:
        return "field"
    if any(k in raw for k in _HX_KEYS) or isinstance(raw.get("duty"), dict):
        return "heatExchanger"
    if any(k in raw for k in _LATTICE_KEYS) or raw.get("sizing") in ("homogenized", "solid-stress"):
        return "lattice"
    return "study"


def read_doc(raw, base_dir: str | None = None):
    """(document, base_dir) for a tool argument that may be the document, its
    JSON text, or a path to a .json file (agents pass all three; a path used
    to fail as "not valid JSON"). A file's folder becomes the base_dir when
    none is given. Raises json.JSONDecodeError for bad JSON text."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text.startswith(("{", "[")):
            path = os.path.expanduser(text)
            if base_dir and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            if os.path.isfile(path):
                with open(path) as fh:
                    return json.load(fh), base_dir or os.path.dirname(os.path.abspath(path))
        return json.loads(raw), base_dir
    return raw, base_dir


def check_document(raw, base_dir: str | None = None) -> dict:
    """`loadpath check` on any document: a study (check_study), a heat
    exchanger (check_heat_exchanger), lattice options or a field graph
    (schema lint)."""
    if isinstance(raw, str):
        try:
            raw, base_dir = read_doc(raw, base_dir)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"not valid JSON (pass the document, its JSON text, or a path to "
                                         f"a .json file): {exc}"}
    kind = document_kind(raw)
    if kind == "heatExchanger":
        return check_heat_exchanger(raw, base_dir)
    if kind in ("lattice", "field"):
        from .schema import format_issues, lint
        issues = lint(raw, kind)
        errors = [i for i in issues if i["level"] == "error"]
        out = {"ok": not errors, "kind": kind, "issues": issues, "schema": f"loadpath schema {kind}"}
        if errors:
            out["error"] = format_issues(errors)
        return out
    return {"kind": "study", **check_study(raw, base_dir)}


def check_heat_exchanger(spec: dict, base_dir: str | None = None) -> dict:
    """Lint and build a heat exchanger spec without designing it
    (milliseconds): the design grid's step, point count and estimated peak
    memory against what is free, the zones, and what the process allows.
    An envelope spec is checked for sizing (loadpath heatx --size)."""
    from .heatx import grid_plan, zones_for
    from .schema import format_issues, lint
    from .sizing import PROCESS_LIMITS
    from .study import _available_bytes

    issues = lint(spec, "heatExchanger")
    errors = [i for i in issues if i["level"] == "error"]
    if errors:
        return {"ok": False, "kind": "heatExchanger", "error": format_issues(errors), "issues": issues,
                "schema": "loadpath schema heatExchanger"}
    warnings = [f"{i['path']}: {i['message']}" for i in issues]
    notes = []
    names = list(spec.get("fluids") or ["A", "B"])
    duty = spec.get("duty")
    if "envelope" in spec and "core" not in spec:
        if not duty or any(n not in duty for n in names):
            return {"ok": False, "kind": "heatExchanger", "error": f"an envelope spec is sized for its duty: "
                    f"\"duty\" needs an entry for each fluid {names}", "issues": issues}
        sz = spec.get("sizing") or {}
        proc = sz.get("process", spec.get("process", "fdm"))
        return jsonable({"ok": True, "kind": "heatExchanger", "mode": "envelope", "envelope": spec["envelope"],
                         "fluids": names, "process": proc, "processLimits": PROCESS_LIMITS.get(proc),
                         "warnings": warnings,
                         "next": "loadpath heatx SPEC --size --out DIR: ranks cores that fit the envelope (each "
                                 "candidate's spec designs with loadpath heatx)"})
    try:
        hx = _hx_spec(spec, base_dir)
        duty = _duty(spec, hx)
        zones = zones_for(hx)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {"ok": False, "kind": "heatExchanger", "error": str(exc) or type(exc).__name__, "issues": issues}
    from . import quick
    quick.heatx(hx, None)
    plan = grid_plan(hx)
    lim = PROCESS_LIMITS.get(hx.process, PROCESS_LIMITS["fdm"])
    if hx.wall < lim["minWall"]:
        warnings.append(f"wall {hx.wall:g} mm is under {hx.process}'s {lim['minWall']:g} mm minimum")
    if hx.shell < lim["minWall"]:
        warnings.append(f"shell {hx.shell:g} mm is under {hx.process}'s {lim['minWall']:g} mm minimum")
    avail = _available_bytes()
    if avail and plan["fits"] and plan["peakMemoryGB"] * 1e9 > 0.9 * avail:
        warnings.append(f"the design needs about {plan['peakMemoryGB']:.1f} GB at its peak and "
                        f"{avail / 1e9:.1f} GB is free: loadpath heatx refuses it (--force runs anyway); set a "
                        f"coarser step (at most wall / 2 = {0.5 * hx.wall:.2f} mm) or --max-points")
    if duty is None:
        notes.append("no duty: the design is not rated (add one entry per fluid and \"wall\")")
    elif not isinstance(duty.get("wall"), str) or _wall_density(duty.get("wall")) is None:
        notes.append("duty.wall is not a known material name: the report carries no mass")
    lo, hi = (np.asarray(b, float) for b in hx.core.bounds())
    out = {"ok": bool(plan["fits"]), "kind": "heatExchanger", "name": spec.get("name"),
           "type": hx.kind, "layout": hx.layout, "fluids": list(hx.names), "cell": hx.cell, "wall": hx.wall,
           "process": hx.process, "core": {"min": lo.round(3).tolist(), "max": hi.round(3).tolist()},
           "zones": [f"{hx.names[z.fluid]}-{z.end}" for z in zones],
           "gridStep": plan["step"], "gridPoints": plan["points"], "maxPoints": plan["maxPoints"],
           "peakMemoryGB": plan["peakMemoryGB"], "availableGB": round(avail / 1e9, 2) if avail else None,
           "processLimits": lim, "warnings": warnings, "notes": notes}
    if not plan["fits"]:
        out["error"] = plan["error"]
        out["maxPointsNeeded"] = plan["maxPointsNeeded"]
    return jsonable(out)


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
            raw, base_dir = read_doc(raw, base_dir)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"study is not valid JSON (pass the study, its JSON text, or a "
                                         f"path to a .json file): {exc}"}
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
    notes = []
    warnings += _preserve_cut(s)
    notes += _resting_supports(s)
    notes += _glued_faces(s, warnings)
    warnings.extend(_split_shafts(s))
    w, n = _blind_bores(s)
    warnings += w
    notes += n
    # only moments the study wrote: one loadpath derived (mass or force 'at'
    # an offset) is in N·mm by construction
    typed = set()
    for rc in (raw.get("loadCases") or []) if isinstance(raw, dict) else []:
        for li, rl in enumerate((rc.get("loads") or []) if isinstance(rc, dict) else []):
            if isinstance(rl, dict) and ("moment" in rl or "torque" in rl):
                typed.add((str(rc.get("name")), li))
    for ci, c in enumerate(s.load_cases):
        for li, ld in enumerate(c.loads):
            mm = float((ld.moment ** 2).sum() ** 0.5)
            if 0 < mm < 50 and (str(c.name).split("@")[0], li) in typed:
                warnings.append(f"loadCases[{ci}].loads[{li}].moment: {mm:g} N·mm is tiny; moments are in N·mm, "
                                f"so {mm:g} N·m is {mm * 1000:g}")
    dur = raw.get("durability") if isinstance(raw, dict) else None
    if isinstance(dur, dict) and not dur.get("constrain"):
        notes.append("durability is checked on the delivered part, not designed for: the search picks the "
                     "lightest static pass, whose fatigue/creep safety factor can be under target. Set "
                     "durability.constrain: true to size the part for it"
                     + ("" if "safetyFactor" in dur else
                        f" (durability.safetyFactor defaults to 1.5, not the goal's {s.goal.safety_factor:g})"
                        if s.goal.safety_factor else ""))
    if not s.pitch and s.min_member and abs(d.pitch - 0.65 * s.min_member) <= 1e-3 * d.pitch:
        notes.append(f"pitch {d.pitch:.2f} mm is set by minMember {s.min_member:g} mm (two cells per filter "
                     f"radius; finer cells only cost time), so the grid has {d.n_elem} cells, fewer than "
                     f"resolution.maxElements {s.max_elements}")
    planar = [c.name for c in s.load_cases if "@" in c.name]
    if planar:
        src = len({c.name.split("@")[0] for c in s.load_cases if "@" in c.name})
        notes.append(f"{len(s.load_cases)} load cases ({len(planar)} of them are {src} in-plane load case"
                     f"{'s' if src > 1 else ''} expanded into one case per direction, 8 or 16 each): every "
                     "optimizer step solves each one, so the run takes about that many times as long as one case")
    from .study import _available_bytes, _confirm_cells, _run_bytes
    g = s.goal
    peak = max(d.n_elem, _confirm_cells(s) if (g.safety_factor or g.max_deflection) else 0)
    mem = _run_bytes(peak) / 1e9
    avail = _available_bytes()
    if mem > 4 and (g.safety_factor or g.max_deflection) and not os.environ.get("LOADPATH_CONFIRM_CELLS"):
        notes.append(f"the fine re-check sets the peak ({mem:.1f} GB at full size); the run shrinks it to fit "
                     "the free memory by itself, so nothing needs setting. To leave room for other jobs on a "
                     "shared machine, LOADPATH_CONFIRM_CELLS caps it (e.g. 50000 for about a third)")
    if avail and mem * 1e9 > 0.8 * avail:
        warnings.append(f"this run needs about {mem:.1f} GB at its peak and {avail / 1e9:.1f} GB is free: "
                        "lower resolution.maxElements, or set LOADPATH_CONFIRM_CELLS for the confirm stage")
    from .pools import threading_layer

    layer = threading_layer()
    if layer == "workqueue" and len(s.load_cases) > 1:
        warnings.append(f"numba's threading layer is workqueue, which aborts when two threads run kernels at "
                        f"once, so the {len(s.load_cases)} load cases solve one after another (slower). "
                        "Install a thread-safe layer: `brew install libomp` on macOS (omp), or "
                        "`pip install tbb` (tbb)")
    return jsonable({
        "ok": True, "name": s.name, "pitch": d.pitch, "threadingLayer": layer, "peakMemoryGB": round(mem, 1), "cells": d.n_elem, "dims": list(d.dims),
        "preservedCells": int(d.preserve.sum()), "filterRadius": filter_radius(d, s.length_scale, s.min_hole),
        "supportNodes": int(d.support_nodes.sum()), "loadNodes": int(d.load_nodes.sum()),
        "regions": [r.id for r in s.preserve] + [r.id for r in s.keep_out],
        "loadCases": [c.name for c in s.load_cases], "material": s.material.name,
        # what each case resolves to (mass, 'at', fasteners and in-plane
        # directions expanded): to check before a half-hour run
        "resolvedLoads": [{"case": c.name, "loads": [
            {"region": ld.region, "force": np.round(ld.force, 3).tolist(),
             **({"moment": np.round(ld.moment, 1).tolist()} if np.any(ld.moment) else {}),
             **({"direction": ld.direction} if ld.direction else {}),
             **({"distribution": ld.distribution} if ld.distribution != "uniform" else {})}
            for ld in c.loads if not ld.pressure]} for c in s.load_cases],
        "minMember": s.min_member, "warnings": warnings, "notes": notes,
    })


def verify_mesh(raw, path: str, base_dir: str | None = None, max_cells: int = 60_000) -> dict:
    """Check any mesh (an earlier outcome, a hand-edited STL, a part from
    elsewhere) against a study: mass, safety factor and deflection per load
    case, keep-out intrusion, preserve loss, bodies, and pass/fail against
    the study's goal. The same analysis `run` gives its own outcomes, on the
    voxel grid (no tet cross-check)."""
    import trimesh

    from .delivery import attention
    from .spec import study_from_dict
    from .study import _margin
    from .verify import verify_part

    if isinstance(raw, str):
        try:
            raw, base_dir = read_doc(raw, base_dir)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"study is not valid JSON: {exc}"}
    checked = check_study(raw, base_dir)
    if not checked["ok"]:
        return checked
    full = path if base_dir is None or os.path.isabs(path) else os.path.join(base_dir, path)
    if not os.path.isfile(full):
        return {"ok": False, "error": f"no mesh at {full}"}
    study = study_from_dict(raw, base_dir=base_dir or os.getcwd())
    rep = verify_part(study, trimesh.load(full, force="mesh"), max_cells=max_cells)
    margin = _margin(study, rep)
    out = {k: rep.get(k) for k in ("mass", "volume", "massReduction", "safetyFactor", "keepOutIntrusion",
                                   "preserveMissing", "outsideDesignSpace", "bodies", "enclosedCavityVolume",
                                   "minMember", "p5Member", "closedBores", "analysisCells")}
    out["cases"] = [{k: c.get(k) for k in ("name", "safetyFactor", "peakStress", "peakAt", "deflection",
                                           "worstDirection", "interlayerStress") if c.get(k) is not None}
                    for c in rep["cases"]]
    if rep.get("durability"):
        out["durability"] = {k: rep["durability"].get(k) for k in ("safetyFactor", "target", "warnings")}
    out["margin"] = margin
    out["passes"] = bool(margin is not None and margin >= 1.0 and rep.get("bodies") == 1
                         and (rep.get("keepOutIntrusion") or 0) <= 1.0 and (rep.get("preserveMissing") or 0) <= 1.0)
    out["attention"] = attention({**rep, "passes": out["passes"]}, raw)
    return jsonable({"ok": True, "stl": full, "safetyFactorMethod": "voxel", **out})


def _leaves(shape):
    return [x for c in shape.shapes for x in _leaves(c)] if hasattr(shape, "shapes") else [shape]


def _bore_through(p, k) -> bool:
    """A keep-out cylinder whose axis runs through the preserved piece: a
    bore or bolt hole through its boss or plate, the idiom, not a mistake.
    A driver path clipping the side of a boss has its axis outside it."""
    import numpy as np

    if getattr(k, "kind", None) != "cylinder":
        return False
    t = np.linspace(0.0, 1.0, 33)[:, None]
    axis = k.p0 + t * (k.p1 - k.p0)
    return bool((p.sdf(axis) < 0).any())


def _resting_supports(s) -> list[str]:
    """A support on a whole round hole (a rod, a peg, a pin the part hangs
    on) holds it all round, bonded. A part that only rests on it bears on a
    narrow line where the load pushes it; holding the rest of the hole
    hides the bending at the hook's mouth (agents under-read such parts by
    up to 3x). Say so; a bolted or clamped bore is the user's call."""
    import numpy as np

    out = []
    try:
        ko = {str(k.id): k for k in s.keep_out}
        for sp in s.supports:
            k = ko.get(str(sp.region)) if sp.region else None
            if k is None or sp.pin or str(k.id).endswith("-bore") or not all(sp.fix):
                continue
            leaves = [c for c in _leaves(k.shape) if getattr(c, "kind", None) == "cylinder"]
            if len(leaves) != 1 or leaves[0].r < 4:
                continue
            c = leaves[0]
            net = [np.sum([ld.force for ld in lc.loads], axis=0) for lc in s.load_cases]
            radial = [f - c.axis * (c.axis @ f) for f in net]
            if not any(np.linalg.norm(f) > 1e-9 for f in radial):
                continue
            out.append(f"supports: {k.id!r} is held all round, bonded. If the part only rests on it (a hook on a "
                       "rod, a cradle, a peg), it bears on a narrow line where the load pushes it: support a thin "
                       "box over that line instead (e.g. ±0.2 r wide, the hole's top), or the bending at the "
                       "opening is hidden")
    except Exception:  # noqa: BLE001 - a lint
        return out
    return out


def _glued_faces(s, warnings_=None) -> list[str]:
    """A support on a thin slab under a whole face, holding the face's normal,
    in a study with bolts through that face: the part is held down
    everywhere, as if glued to the plate, while a bolted foot is held only
    where the bolts clamp it (the plate pushes, it does not pull). A load
    that lifts or tips the part then reads far too stiff and strong: a hot
    stepper bracket with its whole base fixed in z read SF 3.5, and 0.98
    held under its two bolts (brief evals, 2026-11-01)."""
    import numpy as np

    out = []
    if warnings_ is None:
        warnings_ = out
    try:
        lo, hi = s.bounds()
        ext = np.maximum(hi - lo, 1e-9)
        for sp in s.supports:
            sh = sp.shape
            if getattr(sh, "kind", None) != "box" or sp.pin:
                continue
            size = np.minimum(sh.hi, hi) - np.maximum(sh.lo, lo)
            thin = int(np.argmin(size))
            if size[thin] > 1.5 or not sp.fix[thin]:
                continue
            others = [a for a in range(3) if a != thin]
            if np.prod(np.clip(size[others], 0, None)) < 0.4 * np.prod(ext[others]):
                continue
            axis = "xyz"[thin]
            fastened = any(abs(np.asarray(j.get("axis", (0, 0, 0)), float)[thin]) > 0.9 for j in (s.joints or []))
            bolted = fastened
            # plain bolt-hole keep-outs through that face count too (a plate
            # milled from +z held at four M5 holes and its glued bottom)
            for k in s.keep_out:
                for c in _leaves(k.shape):
                    if getattr(c, "kind", None) == "cylinder" and abs(float(c.axis[thin])) > 0.9 \
                            and min(c.p0[thin], c.p1[thin]) <= sh.hi[thin] and max(c.p0[thin], c.p1[thin]) >= sh.lo[thin]:
                        bolted = True
            if not bolted:
                continue
            # does a load pull the part off that face? (the face is at the
            # low end of the axis: a pull is a + force; at the high end a -)
            sign = 1.0 if sh.hi[thin] - lo[thin] < hi[thin] - sh.lo[thin] else -1.0
            lifts = [c.name for c in s.load_cases
                     if sum(sign * float(np.asarray(ld.force, float)[thin]) for ld in c.loads
                            if getattr(ld, "force", None) is not None) > 0]
            if lifts:
                warnings_.append(f"supports: a slab over the whole {axis} face holds it in {axis} everywhere, "
                                 f"so load case {lifts[0]!r}, which pulls the part off that face, is reacted by "
                                 "the face as if it were glued down: far too strong and stiff. A bolted part "
                                 f"is held only under its bolts (the plate pushes, it cannot pull): hold {axis} "
                                 "on a washer-sized disc under each bolt, or on the bolt bores alone")
                continue
            if not fastened:
                continue
            out.append(f"supports: a slab over the whole {axis} face holds it in {axis} everywhere, as if glued "
                       f"to what it sits on. A part bolted down is held only under its bolts (the plate pushes, "
                       f"it cannot pull), so a load that lifts or tips it reads too strong: hold {axis} on a "
                       "washer-sized disc under each bolt instead, unless the face really is bonded")
    except Exception:  # noqa: BLE001 - a lint
        return out
    return out


def _split_shafts(s) -> list[str]:
    """Keep-out cylinders on one axis (a shaft through two bosses written as
    separate pieces) with design space between them that no keep-out
    covers: the part may fill that gap and block the shaft. A turret's
    shoulder shaft written as boss bores plus outer stubs left 3 mm at each
    upright's inner face open, and the part closed over it (339 mm^3)."""
    import numpy as np

    out = []
    try:
        ds = list(s.design_space)
        cyl = []
        for k in s.keep_out:
            for c in _leaves(k.shape):
                if getattr(c, "kind", None) == "cylinder" and getattr(c, "r_inner", 0) <= 0:
                    cyl.append((k, c))
        seen = set()
        for i, (ka, a) in enumerate(cyl):
            for kb, b in cyl[i + 1:]:
                if abs(abs(float(a.axis @ b.axis)) - 1.0) > 1e-3 or abs(a.r - b.r) > 0.2 * max(a.r, b.r):
                    continue
                d = b.p0 - a.p0
                off = d - (d @ a.axis) * a.axis
                if float(np.linalg.norm(off)) > 0.1 * min(a.r, b.r):
                    continue
                ta = sorted((0.0, float((a.p1 - a.p0) @ a.axis)))
                tb = sorted((float((b.p0 - a.p0) @ a.axis), float((b.p1 - a.p0) @ a.axis)))
                lo, hi = (ta[1], tb[0]) if ta[1] <= tb[0] else (tb[1], ta[0]) if tb[1] <= ta[0] else (None, None)
                if lo is None or hi - lo < 1e-6:
                    continue
                between = False
                for _, c in cyl:
                    if c is a or c is b or abs(abs(float(a.axis @ c.axis)) - 1.0) > 1e-3:
                        continue
                    dc = c.p0 - a.p0
                    if float(np.linalg.norm(dc - (dc @ a.axis) * a.axis)) > 0.1 * min(a.r, c.r):
                        continue
                    tc = sorted((float((c.p0 - a.p0) @ a.axis), float((c.p1 - a.p0) @ a.axis)))
                    if tc[0] >= lo - 1e-6 and tc[1] <= hi + 1e-6:
                        between = True      # report the neighbours, not every pair down the line
                        break
                if between:
                    continue
                ts = np.linspace(lo, hi, 25)[1:-1]
                pts = a.p0[None, :] + ts[:, None] * a.axis[None, :]
                inside = np.zeros(len(ts), bool)
                for dsh in ds:
                    inside |= dsh.sdf(pts) < 0
                covered = np.zeros(len(ts), bool)
                for k in s.keep_out:
                    covered |= k.shape.sdf(pts) < 0
                open_ = inside & ~covered
                if not open_.any():
                    continue
                key = tuple(sorted((str(ka.id), str(kb.id))))
                if key in seen:
                    continue
                seen.add(key)
                g = pts[open_]
                span = float(open_.sum()) * (hi - lo) / 24
                out.append(f"keepOut {ka.id!r} and {kb.id!r} lie on one axis with {span:.0f}+ mm of design "
                           f"space between them (near [{', '.join(f'{v:g}' for v in np.round(g[0], 1))}]) that no "
                           "keep-out covers: the part may fill it and block a shaft through both. If one shaft "
                           "or pin runs through them, make it one cylinder end to end")
    except Exception:  # noqa: BLE001 - a lint
        pass
    return out


def _blind_bores(s) -> tuple[list[str], list[str]]:
    """A bore keep-out (a shaft, pin or bearing seat) that ends inside the
    design space: the optimizer may close material over its end, and then the
    shaft cannot go in. Sealed at both ends is almost always a mistake; at
    one end it is a blind pocket, which is right only if nothing passes
    through."""
    import numpy as np

    warn, note = [], []
    try:
        ds = list(s.design_space)
        others = [k for k in s.keep_out]

        def end(pt, own):
            pt = np.asarray(pt, float)[None, :]
            if min(float(d.sdf(pt)[0]) for d in ds) >= 0:
                return "open"       # outside the design space
            if any(float(k.shape.sdf(pt)[0]) < 0 for k in others if k is not own):
                return "pocket"     # into another keep-out
            return "closed"

        for k in s.keep_out:
            # a fastener's head and driver path: `driver` sizes it, and its
            # far end against material is the user's call (a side tool)
            # (and its tap: blind by design, the screw comes from the mating part)
            if not getattr(k, "bore", True) or str(k.id).endswith(("-head", "-tap")):
                continue
            for c in _leaves(k.shape):
                if getattr(c, "kind", None) != "cylinder" or c.r_inner > 0:
                    continue
                e = max(1.0, 0.5 * c.r)
                ends = [end(c.p0 - e * c.axis, k), end(c.p1 + e * c.axis, k)]
                ax = "[" + ", ".join(f"{v:g}" for v in np.round(c.axis, 3)) + "]"
                if "closed" not in ends:
                    continue
                if "open" not in ends:
                    how = "at both ends" if ends == ["closed", "closed"] else "at one end and into another keep-out at the other"
                    warn.append(f"keepOut {k.id!r}: the cylinder along {ax} ends inside the design space {how}, "
                                "so the part can close over it and a shaft or pin may have no way in. Extend "
                                "it past the design space on the side it is inserted from (or set "
                                "\"bore\": false if that is on purpose)")
                elif _centred_out(c, ds) is not None:
                    # most of one half sticks out of the part and the other end stops inside:
                    # nearly always `center` read as the cylinder's end, not its middle
                    a, b, lo, hi = _centred_out(c, ds)
                    warn.append(f"keepOut {k.id!r}: the cylinder along {ax} runs from {a} to {b} and "
                                f"{lo:g} to {hi:g} mm of it is outside the design space, while its other end stops "
                                "inside the part (a blind pocket). `center` is the cylinder's middle, not its end: "
                                "to run it from the outside face through the part, move `center` by half the "
                                "length or give `p0` and `p1`; if a blind pocket is meant, set \"bore\": false")
                elif c.r >= 2.5:    # a blind tapped or insert hole is normal; a blind shaft seat is worth a look
                    note.append(f"keepOut {k.id!r}: the cylinder along {ax} ends inside the design space at "
                                "one end, a blind pocket. If a shaft, pin or bearing passes through, extend "
                                "it past the design space")
                break
    except Exception:  # noqa: BLE001 - a lint
        pass
    return list(dict.fromkeys(warn)), list(dict.fromkeys(note))


def _centred_out(c, ds):
    """For a cylinder with one end open and one closed: (start, end, outside
    from, outside to) when 40% or more of its length lies outside the design
    space, else None. Points are printed as [x, y, z]."""
    import numpy as np

    ts = np.linspace(0.0, 1.0, 41)
    pts = c.p0[None, :] + ts[:, None] * (c.p1 - c.p0)[None, :]
    out = np.ones(len(ts), bool)
    for d in ds:
        out &= d.sdf(pts) >= 0
    if out.mean() < 0.4 or out.all():
        return None
    length = float(np.linalg.norm(c.p1 - c.p0))
    fmt = lambda p: "[" + ", ".join(f"{v:g}" for v in np.round(p, 2)) + "]"
    t_out = ts[out]
    return fmt(c.p0), fmt(c.p1), round(float(t_out.min()) * length, 1), round(float(t_out.max()) * length, 1)


def _preserve_cut(s) -> list[str]:
    """Keep-outs win over preserves, silently: a fastener's driver path
    through a preserved pad, or a clearance box over a boss, deletes what the
    study asked to keep. Say so when one takes a real share of a preserved
    piece (bores through their own bosses excepted)."""
    from .verify import _union

    out = []
    try:
        for r in s.preserve:
            for p in _leaves(r.shape):
                pm = _union([p])
                vol = float(pm.volume())
                for k in s.keep_out:
                    kid, rid = str(k.id), str(r.id)
                    if "-" in kid and "-" in rid and kid.rsplit("-", 1)[0] == rid.rsplit("-", 1)[0]:
                        continue    # one fastener's own tap through its own seat or sleeve
                    # a head and driver path is not a bore: it cuts what it passes through
                    head = kid.endswith("-head")
                    cut = 0.0
                    for kl in _leaves(k.shape):
                        if head or not _bore_through(p, kl):
                            cut += float((pm ^ _union([kl])).volume())
                    if vol > 0 and cut > 1.0 and cut > 0.05 * vol:
                        out.append(f"keepOut {k.id!r} removes {cut:.0f} mm^3 ({cut / vol:.0%}) of preserve {r.id!r}: "
                                   "keep-outs win, so that material will not be there. Move or shrink one of "
                                   "them (a fastener's head and driver path are keep-outs too: check its "
                                   "direction and `driver` length)")
                        break
    except Exception:  # noqa: BLE001 - a lint, never a reason to refuse the study
        return out
    return list(dict.fromkeys(out))


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

_HX_OWN = ("name", "description", "step", "duty", "expect", "envelope", "sizing", "simulate")


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


def _wall_density(wall):
    """g/cm^3 of the duty's wall material (a name), or None."""
    from .heatx_thermal import WALL_DENSITY

    if not isinstance(wall, str):
        return None
    if wall in WALL_DENSITY:
        return WALL_DENSITY[wall]
    try:
        from .materials import material_from_spec
        return float(material_from_spec(wall).density)
    except (ValueError, KeyError, TypeError):
        return None


def _hx_memory(hx, force=False) -> dict:
    """The design grid's plan (heatx.grid_plan), refused up front when the
    point cap leaves too coarse a grid for the wall, or its peak memory is
    over what this machine has free (unless `force`)."""
    from .heatx import grid_plan
    from .study import _available_bytes

    plan = grid_plan(hx)
    if not plan["fits"]:
        raise ValueError(plan["error"])
    avail = _available_bytes()
    if avail:
        plan["availableGB"] = round(avail / 1e9, 2)
        if plan["peakMemoryGB"] * 1e9 > 0.9 * avail and not force:
            raise ValueError(f"the design grid is {plan['points'] / 1e6:.1f}M points at {plan['step']:.3f} mm, about "
                             f"{plan['peakMemoryGB']:.1f} GB at its peak, and {avail / 1e9:.1f} GB is free: set a "
                             f"coarser \"step\" (at most wall / 2 = {0.5 * hx.wall:.2f} mm), a lower --max-points, "
                             "or a smaller core; or --force to run anyway")
    return plan


def heat_exchanger(spec: dict, out_dir: str, base_dir: str | None = None, render: bool = True,
                   on_progress=None, viewer: bool = True, max_points: int | None = None,
                   force: bool = False) -> dict:
    """Design a heat exchanger, check it, rate it if `duty` is given (and
    simulate it whole with "simulate": true; porous.py), and write <name>.stl, <name>.png, the fluid spaces (<name>-<fluid>.stl) and
    an interactive viewer (<name>.html). `max_points` caps the design grid
    (default 16M; about 400 bytes a point at the peak); a design whose
    estimated peak is over the free memory is refused unless `force`."""
    from .heatx import design_heat_exchanger
    from .heatx_thermal import rate_design

    hx = _hx_spec(spec, base_dir)
    if max_points:
        hx.max_points = int(max_points)
    duty = _duty(spec, hx)
    name = str(spec.get("name") or "heat-exchanger")
    sim_opts = spec.get("simulate")
    sim_opts = ({} if sim_opts is True else dict(sim_opts)) if sim_opts and duty is not None else None
    from . import quick
    sim_opts = quick.heatx(hx, sim_opts)
    plan = _hx_memory(hx, force)
    _emit(on_progress, stage="design", type=hx.kind, cell=hx.cell, wall=hx.wall, layout=hx.layout,
          gridStep=plan["step"], gridPoints=plan["points"], peakMemoryGB=plan["peakMemoryGB"])
    mesh, report, spaces, grids = design_heat_exchanger(hx, fluid_meshes=True, grids=True)
    rho = _wall_density((duty or {}).get("wall"))
    if rho is not None:
        # the whole part (core, shell, plenums, ports), as printed dense
        report["mass"] = float(report["solidVolume"]) * rho / 1000.0
        report["massMaterial"] = {"name": duty["wall"], "density": rho}
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
                # the plane both fluids flow in (a counterflow: A end to end, B across)
                plot_fields(f, png, axis=({0, 1, 2} - {int(a) for a in hx.axes}).pop(),
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


def _size_in_envelope(spec, max_pressure_drop, objective, min_effectiveness, min_wall=None, splits=None,
                      axes=None, max_points=None):
    """Size core, cell, wall, split, layout and ports to fit `envelope` (sizing.py)."""
    from .schema import format_issues, lint
    from .sizing import size_exchanger

    errors = [i for i in lint(spec, "heatExchanger") if i["level"] == "error"]
    if errors:
        raise ValueError(format_issues(errors))
    duty = spec.get("duty")
    names = spec.get("fluids", ["A", "B"])
    if not duty or any(n not in duty for n in names):
        raise ValueError(f"sizing needs a duty with an entry for each fluid {names}")
    sz = spec.get("sizing") or {}
    # what the caller passes wins over the spec's sizing block
    obj = objective or sz.get("objective") or "effectiveness"
    kw = {"splits": tuple(float(x) for x in splits)} if splits is not None else {}
    res = size_exchanger({**{n: duty[n] for n in names}, **({"wall": duty["wall"]} if "wall" in duty else {})},
                         spec["envelope"], process=sz.get("process", spec.get("process", "fdm")),
                         max_pressure_drop=max_pressure_drop if max_pressure_drop is not None
                         else sz.get("maxPressureDrop"),
                         min_effectiveness=min_effectiveness if min_effectiveness is not None
                         else sz.get("minEffectiveness"),
                         target_duty=sz.get("targetDuty"), objective=obj, kind=spec.get("type", "gyroid"),
                         layout=spec.get("layout", "auto"), min_wall=min_wall, axes=axes or spec.get("axes"),
                         max_points=max_points, **kw)
    for c in res["candidates"]:
        c["spec"]["name"] = f"{spec.get('name', 'sized')}-{c['layout']}-{c['cell']:g}-{c['wall']:g}"
    res["next"] = ("Design the best one: write its \"spec\" to a file and run loadpath heatx on it (add "
                   "\"simulate\": true to check flow sharing and outlet temperatures whole).")
    return res


def size_heat_exchanger(spec: dict, base_dir: str | None = None, max_pressure_drop: dict | None = None,
                        min_wall: float | None = None, objective: str | None = None,
                        min_effectiveness: float | None = None, splits=None, axes: str | None = None,
                        max_points: int | None = None, out_dir: str | None = None) -> dict:
    """Sweep cell and wall for the core and duty in `spec` before designing
    it (seconds, not minutes): the candidates that meet the pressure-drop
    limits ({fluid: Pa}) and `min_effectiveness`, ranked by `objective`:
    "effectiveness" (default; best first), "mass" (least metal first;
    "metal" is the same) or "pressure" (least of the worst fluid's share of
    its limit). `splits` (the first fluid's share of the void) sweeps the
    sheet's offset too; default the spec's split. `min_wall` floors the
    walls (default the process's minimum). Flow lengths are the core's
    extent along each fluid's axis. With an `envelope` instead of a core,
    sizing.py sizes the core too (`axes` pins which way the first fluid
    runs). Only walls the design step can mesh within `max_points` are
    kept. With `out_dir`, writes sizing.json (the ranked result) and
    best.json (the best candidate's spec, ready for loadpath heatx)."""
    from .shapes import manifold_from_trimesh  # noqa: F401  (manifold3d present)

    if "envelope" in spec:
        res = jsonable(_size_in_envelope(spec, max_pressure_drop, objective, min_effectiveness, min_wall, splits,
                                         axes, max_points))
        return _write_sizing(res, out_dir, (res.get("best") or {}).get("spec"))
    res = _size_core_spec(spec, base_dir, max_pressure_drop, min_wall, objective, min_effectiveness, splits,
                          max_points)
    best = None
    if res.get("best"):
        b = res["best"]
        best = {k: v for k, v in spec.items() if k not in ("expect",)}
        best.update(cell=b["cell"], wall=b["wall"], split=b["split"])
        if "step" in best:
            best["step"] = min(float(best["step"]), 0.5 * b["wall"])
    return _write_sizing(res, out_dir, best)


def _write_sizing(res, out_dir, best_spec):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "sizing.json")
        res["files"] = {"sizing": path}
        if best_spec is not None:
            bp = os.path.join(out_dir, "best.json")
            with open(bp, "w") as fh:
                json.dump(best_spec, fh, indent=2)
            res["files"]["best"] = bp
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
    return res


def _size_core_spec(spec, base_dir, max_pressure_drop, min_wall, objective, min_effectiveness, splits, max_points):
    from .heatx import grid_plan
    from .heatx_thermal import size_core
    from .sizing import PROCESS_LIMITS

    hx = _hx_spec(spec, base_dir)
    if max_points:
        hx.max_points = int(max_points)
    if min_wall is None:
        min_wall = PROCESS_LIMITS.get(hx.process, PROCESS_LIMITS["fdm"])["minWall"]
    objective = objective or (spec.get("sizing") or {}).get("objective") or "effectiveness"
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
    # only walls the design step can mesh within its point cap (a grid of at most wall / 2)
    keep, dropped, nearest = [], 0, None
    step0 = hx.step
    for c in res["candidates"]:
        hx.wall = c["wall"]
        # a spec's own step, but never past what the candidate's wall allows
        hx.step = min(step0, 0.5 * c["wall"]) if step0 else None
        plan = grid_plan(hx)
        if plan["fits"]:
            c["grid"] = {k: plan[k] for k in ("step", "points", "peakMemoryGB")}
            keep.append(c)
        else:
            dropped += 1
            nearest = plan if nearest is None or plan["maxPointsNeeded"] < nearest["maxPointsNeeded"] else nearest
    res["candidates"], res["best"] = keep, keep[0] if keep else None
    res["unmeshable"], res["maxPoints"] = dropped, int(hx.max_points)
    if dropped:
        res["note"] = (f"{dropped} candidate(s) dropped: their walls are too thin for the design step to mesh "
                       f"this core within {hx.max_points / 1e6:g}M grid points (the nearest needs "
                       f"{nearest['maxPointsNeeded'] / 1e6:.0f}M, about "
                       f"{nearest['memoryNeededGB']:.1f} GB): raise --max-points or "
                       "shrink the core")
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
