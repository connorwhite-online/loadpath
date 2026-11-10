"""
Heat-exchanger cores shaped by their own duty: the exchanger form of a
generated exterior (shroud.py).

Exchanger cores are boxes out of habit: a box is easy to header, and the
rating (heatx_thermal.rate) only sees a core's volume and flow length. In
the whole exchanger (porous.py's simulation) a box's corners far from the
ports get little flow, and the end of each stream, where the two fluids
have nearly met in temperature, moves little heat: material, weight and
pressure drop that do little.

`grow_core` simulates the exchanger in its packaging space (the spec's
core, usually that box), reads where the heat actually crosses the wall
(per cell, h_A |T_A - T_wall| + h_B |T_B - T_wall|, both sides of the
sheet), keeps the `keep` share of the core that moves the most of it,
smooths that into one sweeping closed surface (signed distance smoothed
over `smooth` mm, clipped to the packaging space with a rounded edge) and
writes it as an SDF core. heatx then designs headers, ports and seals onto
the new core exactly as onto a box (its zones come from the way the surface
faces), so the comparison against the box is like for like: same duty,
same cell and wall, same ports.

`compare` runs the box, the grown core and a control (the box scaled down
to the grown core's volume, so a win is the shape's and not just less
core) and tabulates mass, effectiveness and pressure drops.

Units are millimetres.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from scipy import ndimage

__all__ = ["duty_field", "grow_core", "compare"]


def duty_field(hx, duty, voxel=None):
    """Simulate `hx` under `duty` and return (domain, per-cell heat crossing
    the wall (W per cell, core cells), simulation summary, design report)."""
    from .porous import simulate_exchanger

    mesh, rep, sim = simulate_exchanger(hx, duty, voxel=voxel, fields=True)
    rep = {**rep, "_mesh": mesh}
    fl = sim.pop("_fields")
    dom, heat = fl["domain"], fl["heat"]
    if heat is None:
        raise ValueError("grow_core needs inlet temperatures (duty.<fluid>.tIn) to find where heat crosses")
    q = np.zeros(dom.core.shape)
    tw = heat["Twall"]
    for f in range(2):
        h = np.asarray(heat["h"][f], float)
        dt = np.nan_to_num(np.abs(heat["T"][f] - tw))
        q += np.nan_to_num(h) * dt
    q = np.where(dom.core, q, 0.0)
    return dom, q, sim, rep


def _summary(sim, rep, density):
    out = {"effectiveness": sim.get("effectiveness"), "duty": sim.get("duty"),
           "solidVolume": rep.get("solidVolume"), "coreVolume": rep.get("coreVolume"),
           "mass": None if rep.get("solidVolume") is None else rep["solidVolume"] * density / 1000.0,
           "fluids": {n: {"pressureDrop": r["pressureDrop"], "pressureDropCore": r["pressureDropCore"],
                          "uniformity": r["entry"].get("uniformity"), "deadShare": r["deadShare"],
                          "tOut": r.get("tOut")} for n, r in sim["fluids"].items()}}
    ic = rep.get("internalCeilings") or {}
    if ic:
        out["internalFlatArea"] = ic.get("flatArea", ic.get("area"))
    return out


def grow_core(spec: dict, out_path: str, base_dir: str | None = None, keep: float = 0.8,
              smooth: float | None = None, voxel: float | None = None):
    """Shape `spec`'s core by its duty and write it to `out_path` (.npz SDF,
    inside base_dir so a spec can name it). Returns (the new spec, info)."""
    from .heatx import heat_exchanger_from_spec
    from .shroud import _sdf_from_mask, _smax

    if not 0 < keep < 1:
        raise ValueError("keep is the share of the core kept, between 0 and 1")
    hx = heat_exchanger_from_spec(spec, base_dir)
    dom, q, sim, rep = duty_field(hx, spec["duty"], voxel)
    core = dom.core
    qc = q[core]
    thr = float(np.quantile(qc, 1.0 - keep))
    kept = core & (q >= thr)
    lab, n = ndimage.label(kept)
    if n > 1:
        sizes = ndimage.sum(kept, lab, range(1, n + 1))
        kept = lab == (1 + int(np.argmax(sizes)))
    kept = ndimage.binary_fill_holes(kept)
    hc = dom.h * 1e3                                   # mm per simulation cell
    F = _sdf_from_mask(kept, hc)                        # at cell centres
    sm = float(smooth if smooth is not None else max(hx.cell, 2 * hc))
    # onto a finer node grid over the packaging space, then smoothed there
    lo_s, hi_s = hx.core.bounds()
    pe = min(hc / 2.0, 1.0)
    pad = 2 * pe
    lo, hi = np.asarray(lo_s) - pad, np.asarray(hi_s) + pad
    dims = np.ceil((hi - lo) / pe).astype(int) + 1
    axes = [lo[i] + pe * np.arange(dims[i]) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    c = ((P - dom.lo) / hc - 0.5).T
    G = ndimage.map_coordinates(F, c, order=1, mode="nearest").reshape(dims)
    G = ndimage.gaussian_filter(G, sigma=sm / pe / 2.0, mode="nearest")
    box = hx.core.sdf(P).reshape(dims)
    G = _smax(G, box, 0.5 * sm).astype(np.float32)
    # the same core volume as kept (smoothing rounds it down a little)
    target = float(kept.sum()) * hc ** 3
    for _ in range(30):
        v = float((G <= 0).sum()) * pe ** 3
        if abs(v - target) < 0.01 * target:
            break
        G = _smax(G - (target - v) / max(_area(G, pe), 1.0), box, 0.5 * sm).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, distance=G, origin=lo, pitch=float(pe), units="mm")
    new = copy.deepcopy(spec)
    rel = os.path.relpath(os.path.abspath(out_path), os.path.abspath(base_dir or os.getcwd()))
    new["core"] = {"type": "sdf", "path": rel}
    new["name"] = f"{spec.get('name', 'exchanger')}-grown"
    from .heatx_thermal import WALL_DENSITY
    dens = WALL_DENSITY.get(spec["duty"].get("wall", "petg"), 1.0) if isinstance(spec["duty"].get("wall"), str) else 1.0
    info = {"keep": keep, "smooth": sm, "threshold": thr, "kept": float((G <= 0).sum()) * pe ** 3,
            "box": _summary(sim, rep, dens),
            "dutyShareKept": float(q[kept].sum() / max(qc.sum(), 1e-30)), "_boxMesh": rep.get("_mesh")}
    return new, info


def _area(G, pe):
    band = np.abs(G) < pe
    return float(band.sum()) * pe ** 2


def compare(spec: dict, out_dir: str, base_dir: str | None = None, keep: float = 0.8, voxel=None,
            progress=None) -> dict:
    """The box, the duty-grown core and the box scaled to the grown core's
    volume, each designed and simulated under the spec's duty."""
    from .heatx import heat_exchanger_from_spec
    from .heatx_thermal import WALL_DENSITY
    from .porous import simulate_exchanger

    base = base_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    wall = spec["duty"].get("wall", "petg")
    dens = WALL_DENSITY.get(wall, 1.0) if isinstance(wall, str) else 1.0
    grown, info = grow_core(spec, os.path.join(out_dir, "grown-core.npz"), base, keep, voxel=voxel)
    if progress:
        progress({"stage": "grown"})
    rows = {"box": info["box"]}
    meshes = {"box": info.pop("_boxMesh")}
    hx_g = heat_exchanger_from_spec(grown, base)
    m, rep, sim = simulate_exchanger(hx_g, grown["duty"], voxel=voxel)
    rows["grown"] = _summary(sim, rep, dens)
    meshes["grown"] = m
    # control: the box scaled about its centre to the grown core's volume
    lo, hi = (np.asarray(x, float) for x in heat_exchanger_from_spec(spec, base).core.bounds())
    s = (rows["grown"]["coreVolume"] / rows["box"]["coreVolume"]) ** (1 / 3)
    c = 0.5 * (lo + hi)
    small = copy.deepcopy(spec)
    small["core"] = {"type": "box", "min": (c - s * (c - lo)).tolist(), "max": (c + s * (hi - c)).tolist()}
    m, rep, sim = simulate_exchanger(heat_exchanger_from_spec(small, base), small["duty"], voxel=voxel)
    rows["smallBox"] = _summary(sim, rep, dens)
    meshes["smallBox"] = m
    out = {"keep": keep, "dutyShareKept": info["dutyShareKept"], "rows": rows, "grownSpec": grown}
    with open(os.path.join(out_dir, "compare.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k, m in meshes.items():
        m.export(os.path.join(out_dir, f"{k}.stl"))
    return out
