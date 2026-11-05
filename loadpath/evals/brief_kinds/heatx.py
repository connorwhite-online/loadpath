"""
Heat-exchanger briefs: the agent delivers a two-fluid exchanger (loadpath's
heatExchanger spec and the part it designs), not a structural part.

The scorer never takes the agent's numbers. It designs the delivered spec
again (heatx.design_heat_exchanger: the same geometry the agent's run wrote)
and rates it (heatx_thermal.rate_design) with the GOLD duty put in place of
the agent's: the brief's fluids, flows, inlet temperatures and wall
material, matched to the agent's fluid names. Then the brief's pass block:

    "pass": {
      "envelope": {"min": [x, y, z], "max": [x, y, z]},    # the whole part, ports included
      "ports": {"oil-in": {"at": [x, y, z], "direction": "-x"},   # gold fluid names; "at"
                "coolant-out": {"direction": "y"}, ...},           # optional, "y" = either way
      "portTolerance": {"lateral": 3, "axial": 5},          # mm off the port's axis / short of "at"
      "counterflow": ["oil", "coolant"],                    # optional: the second runs against the first
      "minEffectiveness": 0.6, "minDuty": 400,              # W (either or both)
      "maxPressureDrop": {"oil": 50000, "coolant": 10000},  # Pa per gold fluid
      "maxMass": 120,                                       # g, solid volume x the wall's density
      "maxTrappedVolume": 20, "maxInternalFlatArea": 300,   # powder-bed printability (optional)
      "process": "lpbf"                                     # wall/channel limits (sizing.PROCESS_LIMITS);
    }                                                       # default the gold spec's process

and always: the spec is a designed core (not a sizing request), the
process and build axis are the brief's, the two fluids are isolated and
each reaches its own two ports, the mesh is watertight, the walls between
the fluids and to the outside and the channels meet the process's limits,
and the delivered exchanger.stl is the part that spec designs (volume and
bounds), so a hand-made block or an edited mesh does not pass.

Cost: one design of the delivered spec at its own grid (about 20 s to 2
minutes for these briefs on two threads, up to the 16M-point cap heatx
itself works to). `quick=True` coarsens the grid like `loadpath heatx
--quick` (tests).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

DELIVER = """  exchanger.json   the heat exchanger you designed: loadpath's heatExchanger spec (a core, not
                   only an envelope), with its duty
  exchanger.stl    the part to print, as loadpath heatx wrote it for exchanger.json (keep the
                   fluid-space STLs it writes beside it)
  notes.md         two or three lines: its effectiveness or heat duty, each fluid's pressure drop,
                   its mass, and your assumptions
"""

ENVELOPE_TOL = 0.5      # mm: marching cubes rounds the part's outside by up to a grid step
PORT_COS = np.cos(np.radians(15.0))
PORT_TOL = {"lateral": 3.0, "axial": 5.0}
WALL_SLACK = 0.85       # measured walls (on the design grid) against the process's minimum
CHANNEL_SLACK = 0.9
MESH_VOLUME_TOL = 0.05  # delivered STL against the redesign (a --quick run differs by a few %)
MESH_BOUNDS_TOL = 1.5   # mm


def _axis(v) -> tuple[np.ndarray, bool]:
    """'-x' -> (unit vector, signed); 'x' -> (e_x, False): either way."""
    s = str(v).strip()
    signed = s[0] in "+-"
    e = np.zeros(3)
    e["xyz".index(s.lstrip("+-"))] = -1.0 if s.startswith("-") else 1.0
    return e, signed


def _fluid_map(spec: dict, gold: dict) -> tuple[dict | None, str | None]:
    """The agent's fluid name for each gold fluid. Same names map by name;
    otherwise by each side's duty (fluid type, then inlet temperature).
    Never by where the ports are: a swapped exchanger must not be mapped
    back onto the brief."""
    g_names = list(gold.get("fluids") or ["A", "B"])
    a_names = list(spec.get("fluids") or ["A", "B"])
    if set(a_names) == set(g_names):
        return {g: g for g in g_names}, None
    a_duty, g_duty = spec.get("duty") or {}, gold["duty"]
    if not all(isinstance(a_duty.get(n), dict) for n in a_names):
        return None, (f"exchanger.json's fluids {a_names} are not the brief's and it has no duty to tell "
                      "which is which")

    def fit(g, a):
        gd, ad = g_duty[g], a_duty[a]
        s = 10.0 * (gd.get("fluid", "water") == ad.get("fluid", "water"))
        if gd.get("tIn") is not None and ad.get("tIn") is not None:
            s += 5.0 / (1.0 + abs(float(gd["tIn"]) - float(ad["tIn"])))
        return s

    straight = fit(g_names[0], a_names[0]) + fit(g_names[1], a_names[1])
    crossed = fit(g_names[0], a_names[1]) + fit(g_names[1], a_names[0])
    if abs(straight - crossed) < 1e-9:
        return None, f"cannot tell which of exchanger.json's fluids {a_names} is which of the brief's"
    pair = a_names if straight > crossed else a_names[::-1]
    return dict(zip(g_names, pair)), None


def _density(wall) -> float | None:
    from ...heatx_thermal import WALL_DENSITY
    if isinstance(wall, str):
        if wall in WALL_DENSITY:
            return WALL_DENSITY[wall]
        try:
            from ...materials import material_from_spec
            return float(material_from_spec(wall).density)
        except (KeyError, ValueError):
            return None
    return None


def _at_mouth(hx, rep, at) -> bool:
    """Whether a point lies at a port's open end, inside its teardrop."""
    if at is None:
        return False
    at = np.asarray(at, float)
    step = float(rep["step"])
    for z in rep["zones"].values():
        p1, ax = np.asarray(z["port"][0], float), np.asarray(z["port"][1], float)
        d = at - p1
        a = float(d @ ax)
        lateral = float(np.linalg.norm(d - a * ax))
        if -(hx.shell + 3 * step) <= a <= 3.0 and lateral <= 0.5 * z["portDiameter"] * np.sqrt(2) + hx.shell + 2 * step:
            return True
    return False


def _round(v, n=3):
    return None if v is None else round(float(v), n)


def evaluate(brief: dict, spec: dict, base_dir: str | None = None, delivered=None, quick: bool = False) -> dict:
    """Design `spec` (a heatExchanger document), rate it under the brief's
    gold duty and check the pass block. `delivered`: the agent's mesh
    (a path or trimesh), compared with the redesign. Returns the score row."""
    from ... import quick as quick_mode
    from ...heatx import design_heat_exchanger
    from ...heatx_thermal import rate_design
    from ...sizing import PROCESS_LIMITS
    from ...tasks import _hx_spec

    gold, want = brief["gold"], brief["pass"]
    row = {"pass": False, "failures": []}
    f = row["failures"]
    if "core" not in spec:
        f.append("exchanger.json has no core: an envelope is a sizing request, not a designed exchanger "
                 "(design the sized spec and deliver that)")
        return row
    spec = {k: v for k, v in spec.items() if k not in ("simulate", "expect")}
    hx = _hx_spec(spec, base_dir)
    fmap, err = _fluid_map(spec, gold)
    if err:
        f.append(err)
        return row
    row["fluids"] = fmap

    # the brief's process and build direction
    g_proc = gold.get("process", "fdm")
    if hx.process != g_proc:
        f.append(f"designed for process {hx.process!r}, the brief is {g_proc!r}")
    g_build = gold.get("build", "z")
    g_build = None if g_build in (None, False, "none") else "xyz".index(g_build)
    if hx.build != g_build:
        have = "none" if hx.build is None else "xyz"[hx.build]
        f.append(f"built along {have}, the brief prints along {'none' if g_build is None else 'xyz'[g_build]}")
    lim_key = want.get("process", g_proc)
    lim = PROCESS_LIMITS[lim_key]
    if hx.wall < lim["minWall"] - 1e-9:
        f.append(f"{hx.wall:g} mm wall is under {lim_key}'s {lim['minWall']:g} mm")

    if quick:
        # as `loadpath heatx --quick` coarsens it (at most 0.45 wall: heatx refuses past half)
        hx.step = min((hx.step or hx.wall / 3.0) * quick_mode.HX_STEP, hx.wall * 0.45)
    t0 = time.time()
    mesh, rep = design_heat_exchanger(hx)
    row["designSeconds"] = round(time.time() - t0, 1)
    row["step"] = _round(rep["step"])

    # isolation and ports reached
    row["isolated"] = bool(rep.get("isolated"))
    row["connected"] = {g: bool((rep.get("connected") or {}).get(a)) for g, a in fmap.items()}
    row["watertight"] = bool(rep.get("watertight"))
    if not row["isolated"]:
        f.append("the two fluids are not isolated (a port reaches the other fluid's space, or none)")
    for g, ok in row["connected"].items():
        if not ok:
            f.append(f"{g} does not reach both its ports")
    if not row["watertight"]:
        f.append("the part's mesh is not watertight")

    # envelope
    lo, hi = (np.asarray(b, float) for b in rep["bounds"])
    row["bounds"] = [lo.round(2).tolist(), hi.round(2).tolist()]
    env = want.get("envelope")
    if env:
        elo, ehi = np.asarray(env["min"], float), np.asarray(env["max"], float)
        over = np.maximum(elo - lo, hi - ehi).max()
        row["outsideEnvelope"] = _round(max(over, 0.0), 2)
        if over > ENVELOPE_TOL:
            f.append(f"the part reaches {over:.1f} mm outside the envelope "
                     f"({row['bounds']} vs {[elo.tolist(), ehi.tolist()]})")

    # ports where the brief puts them
    tol = {**PORT_TOL, **(want.get("portTolerance") or {})}
    row["ports"] = {}
    for key, req in (want.get("ports") or {}).items():
        g, end = key.rsplit("-", 1)
        z = rep["zones"].get(f"{fmap[g]}-{end}")
        if z is None:
            f.append(f"no {key} port")
            continue
        mouth, ax = np.asarray(z["port"][0], float), np.asarray(z["port"][1], float)
        e, signed = _axis(req["direction"])
        c = float(ax @ e)
        got = {"mouth": mouth.round(2).tolist(), "axis": ax.round(3).tolist(), "diameter": z["portDiameter"]}
        if (c if signed else abs(c)) < PORT_COS:
            f.append(f"{key} port points {np.round(ax, 2).tolist()}, the brief has it along {req['direction']}")
        if "at" in req:
            at = np.asarray(req["at"], float)
            if not signed:
                e = e * np.sign(c or 1.0)
            d = at - mouth
            axial = float(d @ e)
            lateral = float(np.linalg.norm(d - axial * e))
            got.update(lateral=round(lateral, 2), axial=round(axial, 2))
            if lateral > tol["lateral"]:
                f.append(f"{key} port is {lateral:.1f} mm off where the brief has it ({at.tolist()}; "
                         f"its mouth is at {mouth.round(1).tolist()})")
            if axial > tol["axial"]:
                f.append(f"{key} port ends {axial:.1f} mm short of {at.tolist()}")
        if req.get("minDiameter") and z["portDiameter"] < req["minDiameter"]:
            f.append(f"{key} port is {z['portDiameter']:g} mm, under {req['minDiameter']:g} mm")
        row["ports"][key] = got
    cf = want.get("counterflow")
    if cf:
        a, b = (fmap[n] for n in cf)
        za = rep["zones"]
        u = np.asarray(za[f"{a}-out"]["center"], float) - np.asarray(za[f"{a}-in"]["center"], float)
        run = float((np.asarray(za[f"{b}-out"]["center"], float) - np.asarray(za[f"{b}-in"]["center"], float)) @ u)
        row["counterflow"] = run < 0
        if run >= 0:
            f.append(f"{cf[1]} does not run against {cf[0]} (its inlet is not at {cf[0]}'s outlet end)")

    # process limits, measured on the design grid
    walls = {"betweenFluids": rep.get("minWallBetweenFluids"), "outer": rep.get("minOuterWall")}
    if walls["outer"] is not None and _at_mouth(hx, rep, rep.get("minOuterWallAt")):
        # heatx leaves a port's mouth out of the outer-wall check by a cylinder r + shell
        # around it, but a sideways port's teardrop peaks at r * sqrt(2): on ports over
        # ~11 mm the open lip of the teardrop reads as a one-voxel wall
        row["outerWallAtMouth"] = _round(walls["outer"])
        walls["outer"] = None
    row["minWall"] = {k: _round(v) for k, v in walls.items()}
    for k, v in walls.items():
        if v is not None and v < WALL_SLACK * lim["minWall"]:
            where = rep.get("minWallBetweenFluidsAt" if k == "betweenFluids" else "minOuterWallAt")
            f.append(f"{'wall between the fluids' if k == 'betweenFluids' else 'outer wall'} is {v:.2f} mm "
                     f"at {where}, under {lim_key}'s {lim['minWall']:g} mm")
    row["hydraulicDiameter"] = {}
    for g, a in fmap.items():
        dh = (rep.get(a) or {}).get("hydraulicDiameter")
        row["hydraulicDiameter"][g] = _round(dh)
        if dh is not None and dh < CHANNEL_SLACK * lim["minChannel"]:
            f.append(f"{g}'s channels are {dh:.2f} mm across, under {lim_key}'s {lim['minChannel']:g} mm")
    ic = rep.get("internalCeilings") or {}
    row["trappedVolume"] = _round(rep.get("trappedVolume"), 1)
    row["internalFlatArea"] = _round(ic.get("flatArea"), 1)
    if want.get("maxTrappedVolume") is not None and (rep.get("trappedVolume") or 0) > want["maxTrappedVolume"]:
        f.append(f"{rep['trappedVolume']:.1f} mm^3 of open space no port reaches (trapped powder), "
                 f"over {want['maxTrappedVolume']:g}")
    if want.get("maxInternalFlatArea") is not None and (ic.get("flatArea") or 0) > want["maxInternalFlatArea"]:
        f.append(f"{ic['flatArea']:.0f} mm^2 of flat ceiling inside (unsupportable), over "
                 f"{want['maxInternalFlatArea']:g}")

    # rated under the gold duty
    gd = gold["duty"]
    duty = {fmap[g]: gd[g] for g in fmap}
    if "wall" in gd:
        duty["wall"] = gd["wall"]
    try:
        r = rate_design(hx, rep, duty)
    except ValueError as exc:
        f.append(f"cannot rate it: {exc}")
        r = None
    if r is not None:
        row["effectiveness"] = _round(r["effectiveness"])
        row["duty"] = _round(abs(r.get("duty") or 0.0), 1)
        row["pressureDrop"] = {g: _round(r["fluids"][a]["pressureDrop"], 0) for g, a in fmap.items()}
        row["tOut"] = {g: _round(r["tOut"][a], 1) for g, a in fmap.items()} if r.get("tOut") else None
        if want.get("minEffectiveness") is not None and r["effectiveness"] < want["minEffectiveness"]:
            f.append(f"effectiveness {r['effectiveness']:.3f} < {want['minEffectiveness']}")
        if want.get("minDuty") is not None and row["duty"] < want["minDuty"]:
            f.append(f"moves {row['duty']:.0f} W < {want['minDuty']} W")
        for g, cap in (want.get("maxPressureDrop") or {}).items():
            if row["pressureDrop"][g] > cap:
                f.append(f"{g} pressure drop {row['pressureDrop'][g]:.0f} Pa > {cap} Pa")
    rho = _density(gd.get("wall"))
    row["solidVolume"] = _round(rep["solidVolume"], 0)
    row["mass"] = _round(rep["solidVolume"] * rho / 1000.0, 1) if rho else None
    if want.get("maxMass") is not None and row["mass"] is not None and row["mass"] > want["maxMass"]:
        f.append(f"{row['mass']:.1f} g > {want['maxMass']} g")

    # the delivered part is the one this spec designs
    if delivered is not None:
        import trimesh
        m = trimesh.load(delivered, force="mesh") if isinstance(delivered, str) else delivered
        vol = abs(float(m.volume))
        dv = abs(vol - rep["solidVolume"]) / max(rep["solidVolume"], 1e-9)
        db = float(np.abs(np.asarray(m.bounds) - np.stack([lo, hi])).max())
        row["deliveredMesh"] = {"volume": round(vol, 0), "volumeDiff": round(dv, 3), "boundsDiff": round(db, 2)}
        if dv > MESH_VOLUME_TOL * (2 if quick else 1) or db > MESH_BOUNDS_TOL:
            f.append(f"exchanger.stl is not the part exchanger.json designs (volume {vol:.0f} vs "
                     f"{rep['solidVolume']:.0f} mm^3, bounds off by {db:.1f} mm)")
    row["pass"] = not f
    return row


def _find_mesh(folder: str, spec: dict) -> str | None:
    for n in ("exchanger.stl", f"{spec.get('name') or 'heat-exchanger'}.stl"):
        p = os.path.join(folder, n)
        if os.path.exists(p):
            return p
    return None


def score(brief: dict, folder: str, quick: bool = False) -> dict:
    """Score what the agent left in `folder` (exchanger.json, exchanger.stl)."""
    path = os.path.join(folder, "exchanger.json")
    if not os.path.exists(path):
        return {"pass": False, "failures": ["no exchanger.json"]}
    try:
        with open(path) as fh:
            spec = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"pass": False, "failures": [f"exchanger.json does not read: {exc}"]}
    if not isinstance(spec, dict):
        return {"pass": False, "failures": ["exchanger.json is not a heatExchanger document"]}
    mesh = _find_mesh(folder, spec)
    row = evaluate(brief, spec, base_dir=folder, delivered=mesh, quick=quick)
    if mesh is None:
        row["failures"].append("no exchanger.stl")
        row["pass"] = False
    return row


def gold(brief: dict, quick: bool = False) -> dict:
    """The gold spec's own check: it lints, its duty and the pass block name
    its fluids, and it passes the scorer (warnings where a margin is thin)."""
    t0 = time.time()
    g, want = brief.get("gold") or {}, brief.get("pass") or {}
    names = list(g.get("fluids") or ["A", "B"])
    problems = []
    if "core" not in g:
        problems.append("gold needs a core")
    if sorted(n for n in (g.get("duty") or {}) if n != "wall") != sorted(names):
        problems.append(f"gold duty must name each of {names}")
    for key in want.get("ports") or {}:
        n, _, end = key.rpartition("-")
        if n not in names or end not in ("in", "out"):
            problems.append(f"pass port {key!r} is not <fluid>-in/out of {names}")
    for n in (want.get("maxPressureDrop") or {}):
        if n not in names:
            problems.append(f"pass maxPressureDrop names {n!r}, not one of {names}")
    if not any(k in want for k in ("minEffectiveness", "minDuty")):
        problems.append("pass needs minEffectiveness or minDuty")
    if problems:
        return {"ok": False, "error": "; ".join(problems)}
    try:
        row = evaluate(brief, g, quick=quick)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"gold does not design: {str(exc).strip()[:300]}"}
    warn = []
    if row.get("effectiveness") is not None and want.get("minEffectiveness"):
        if row["effectiveness"] < 1.05 * want["minEffectiveness"]:
            warn.append(f"effectiveness margin thin: {row['effectiveness']} vs {want['minEffectiveness']}")
    if row.get("duty") is not None and want.get("minDuty") and row["duty"] < 1.05 * want["minDuty"]:
        warn.append(f"duty margin thin: {row['duty']} vs {want['minDuty']}")
    for n, cap in (want.get("maxPressureDrop") or {}).items():
        if (row.get("pressureDrop") or {}).get(n, 0) > 0.95 * cap:
            warn.append(f"{n} pressure drop margin thin: {row['pressureDrop'][n]} vs {cap}")
    if want.get("maxMass") and row.get("mass") and row["mass"] > 0.95 * want["maxMass"]:
        warn.append(f"mass margin thin: {row['mass']} vs {want['maxMass']}")
    out = {"ok": bool(row["pass"]), "score": row, "warnings": warn, "seconds": round(time.time() - t0, 1),
           "cells": f"step {row.get('step')} mm"}
    if not row["pass"]:
        out["error"] = "; ".join(row["failures"])
    return out
