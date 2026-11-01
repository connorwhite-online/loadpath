"""
Do field-graded lattices beat solid topology optimization for heat sinks?

An air-cooled sink in a duct: a 40 x 40 x 3 mm AlSi10Mg base (LPBF), 30 W
into a 20 x 20 mm die under its centre, a 40 x 40 x 15 mm fin space above
it, the duct closed over the fins and at both sides, 10 mm of empty duct
before and after. Every design is rated the same way (conjugate.py): flow by
lattice Boltzmann at a fixed pressure drop, then conjugate heat; the score is
the die's hottest point over the inlet air, per watt, at equal metal in the
fin space.

Designs:
- plate fins (the reference anyone would draw),
- solid topology optimization (thermal.optimize_thermal, constant h on the
  surface, MMA): the best loadpath could do before this,
- uniform gyroid network lattice (the solid side of a gyroid),
- gyroid graded by the heat flux of the uniform lattice's own conjugate
  solution (more metal where more heat runs), at the same mass.

Run: `python bench/lattice_heatsink.py [--pitch 0.5] [--dp 20] [--vf 0.15]`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage

from loadpath.conjugate import at_pressure_drop, solve_heat
from loadpath.heatx_thermal import FLUIDS
from loadpath.thermal import CONDUCTIVITY

L_UP, L_FIN, L_DOWN = 10.0, 40.0, 10.0
W, BASE, FIN_H = 40.0, 3.0, 15.0
DIE = 20.0
DIE_X = L_FIN / 2          # the die's centre along the flow (from the fins' leading edge)
POWER = 30.0


class Sink:
    def __init__(self, pitch):
        self.h = pitch
        self.nx = int(round((L_UP + L_FIN + L_DOWN) / pitch))
        self.ny = int(round(W / pitch))
        self.nz = int(round((BASE + FIN_H) / pitch))
        x = (np.arange(self.nx) + 0.5) * pitch - L_UP
        y = (np.arange(self.ny) + 0.5) * pitch
        z = (np.arange(self.nz) + 0.5) * pitch
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")
        self.under = (self.X >= 0) & (self.X <= L_FIN)
        self.base = self.under & (self.Z < BASE)
        self.space = self.under & (self.Z >= BASE)          # where fins go
        self.duct = self.Z >= BASE                          # fluid can be here
        bz = int(round(BASE / pitch))
        self.die = self.base & (self.Z < pitch) & (np.abs(self.X - DIE_X) <= DIE / 2) & \
            (np.abs(self.Y - W / 2) <= DIE / 2)

    def masks(self, fins):
        solid = self.base | (fins & self.space)
        fluid = self.duct & ~solid
        return solid, fluid

    def vf(self, fins):
        return float((fins & self.space).sum() / self.space.sum())


def plate_fins(s: Sink, vf):
    """Straight fins along the flow, full height, a whole number of voxels
    thick (>= 2) with gaps >= 2 mm: of the counts whose metal comes within 3%
    of vf, the most fins (voxel rounding of a fixed thickness otherwise
    misses vf by up to 7% on a fine grid)."""
    ny = s.ny
    fits = []
    for n in range(3, 20):
        k = int(round(vf * ny / n))
        if k >= 2 and W / n - k * s.h >= 2.0:
            fits.append((abs(n * k / ny - vf) / vf, n, k))
    close = [f for f in fits if f[0] <= 0.03]
    _, n, k = max(close, key=lambda f: f[1]) if close else min(fits)
    m1 = np.zeros(ny, bool)
    for c in (np.arange(n) + 0.5) * ny / n:
        lo = int(round(c - k / 2))
        m1[lo:lo + k] = True
    return m1[None, :, None] & s.space


def gyroid_value(s: Sink, cell):
    k = 2 * np.pi / cell
    return np.sin(k * s.X) * np.cos(k * s.Y) + np.sin(k * s.Y) * np.cos(k * s.Z) + np.sin(k * s.Z) * np.cos(k * s.X)


def match(s: Sink, build, vf, lo, hi, iters=40):
    """Bisect a design's one parameter (more metal as it rises) until its
    connected fins hold `vf` of the fin space: every design is compared at
    the same mass of metal that actually touches the base."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = keep_connected(s, build(mid))
        lo, hi = (mid, hi) if s.vf(m) < vf else (lo, mid)
    return keep_connected(s, build(hi))


def gyroid_uniform(s: Sink, vf, cell):
    g = gyroid_value(s, cell)
    return match(s, lambda t: (g <= t) & s.space, vf, -1.5, 1.5)


# the thinnest a graded lattice gets (solid fraction): below ~0.1 the gyroid's
# network ligaments are under two voxels and fall apart
RHO_MIN = 0.10
# and the thickest: a near-solid layer of lattice over the base walls the
# flow off from it (the first flux-graded runs put rho 0.6 there and rated
# 2-3x worse than uniform)
RHO_MAX = 0.4


def gyroid_graded(s: Sink, vf, cell, density):
    """Network gyroid whose local solid fraction follows `density` (field on
    the grid, any scale): rho = clip(c x density, RHO_MIN, 0.6), c bisected for
    the mean vf. The level that gives a fraction rho is read off the uniform
    gyroid's own value distribution."""
    g = gyroid_value(s, cell)
    vals = np.sort(g[s.space])
    d = np.clip(density, 0, None) / max(float(density[s.space].mean()), 1e-30)

    def build(c):
        rho = np.clip(c * d, RHO_MIN, RHO_MAX)
        return (g <= vals[np.clip((rho * (len(vals) - 1)).astype(int), 0, len(vals) - 1)]) & s.space

    return match(s, build, vf, 0.0, 5.0)


def solid_to(s: Sink, vf, h_conv=50.0, pitch=1.0, radius=2.0):
    """thermal.optimize_thermal on the fin space with the base pinned:
    constant convection h_conv on the fins' surface and on the fin block's
    faces open to the flow (not the duct's walls), air at 0 K rise. The
    design (on its own `pitch` grid, symmetric about the duct's mid-plane)
    is resampled onto the rating grid from its filtered density."""
    from loadpath.shapes import Box
    from loadpath.thermal import Convection, ThermalOptions, optimize_thermal, thermal_bc, thermal_grid

    ds = Box([0, 0, 0], [L_FIN, W, BASE + FIN_H])
    base = Box([0, 0, 0], [L_FIN, W, BASE])
    grid = thermal_grid([ds], pitch, preserve=[base], symmetry=["y"])
    die = Box([DIE_X - DIE / 2, W / 2 - DIE / 2, -0.01], [DIE_X + DIE / 2, W / 2 + DIE / 2, 0.01])
    open_x = lambda c, n: np.abs(n[:, 0]) > 0.5
    bc = thermal_bc(grid.mesh, ambient=0.0, heat=[(die, POWER)],
                    convection=[Convection(h_conv, 0.0, boundary=False),
                                Convection(h_conv, 0.0, where=open_x, internal=False)])
    t0 = time.time()
    res = optimize_thermal(grid, bc, ThermalOptions(volume_fraction=vf, filter_radius=radius,
                                                    k=CONDUCTIVITY["alsi10mg"]))
    full = np.zeros(grid.dims)
    full[tuple(grid.mesh.elem_ijk.T)] = res.xphys
    full = ndimage.gaussian_filter(full, 0.5)
    coords = [(s.X - grid.origin[0]) / pitch - 0.5, (s.Y - grid.origin[1]) / pitch - 0.5,
              (s.Z - grid.origin[2]) / pitch - 0.5]
    dens = ndimage.map_coordinates(full, coords, order=1, mode="constant")
    fins = match(s, lambda t: (dens > 1.0 - t) & s.space, vf, 0.0, 1.0)
    print(json.dumps({"solidTO": {"iterations": res.iterations, "seconds": time.time() - t0,
                                  "constantHRise": res.objective, "vf": res.volume_fraction}}), flush=True)
    return fins & s.space


# what plate fins pass at 20 Pa in the 0.5 mm lattice Boltzmann rating
# (bench/lattice_heatsink.json): the Darcy model's calibration
PLATES_LPM_AT_20PA = 91.1


def flow_to(s: Sink, vf, dp, pitch=1.0, radius=2.0, calib_lpm=PLATES_LPM_AT_20PA, calib_dp=20.0):
    """flowtopo.optimize_flow_thermal: Darcy flow and advection-conduction in
    the loop, at the duct's pressure drop, on a `pitch` grid (open-duct
    permeability calibrated so plate fins pass what LBM says they do), the
    density resampled onto the rating grid and matched to vf."""
    from loadpath.flowtopo import FlowSink, FlowTopoOptions, calibrate_alpha, optimize_flow_thermal

    c = Sink(pitch)
    plates = plate_fins(c, vf)
    ps, pf = c.masks(plates)
    alpha_f = calibrate_alpha(ps, pf, pitch, calib_dp, calib_lpm)
    fs = FlowSink(c.space, c.base, c.duct & ~c.space, c.die, pitch, CONDUCTIVITY["alsi10mg"], FLUIDS["air"], dp,
                  alpha_f)
    t0 = time.time()
    res = optimize_flow_thermal(fs, c.space, pitch, FlowTopoOptions(vf, filter_radius=radius, equal_volume=True))
    last = res["history"][-1]
    print(json.dumps({"flowTO": {"iterations": len(res["history"]), "seconds": time.time() - t0,
                                 "darcyRiseKperW": last["objective"], "darcyLpm": last["flowLpm"],
                                 "alphaF": alpha_f}}), flush=True)
    full = ndimage.gaussian_filter(res["x"], 0.5)
    coords = [(s.X - c.X[0, 0, 0]) / pitch, (s.Y - c.Y[0, 0, 0]) / pitch, (s.Z - c.Z[0, 0, 0]) / pitch]
    dens = ndimage.map_coordinates(full, coords, order=1, mode="nearest")
    return match(s, lambda t: (dens > 1.0 - t) & s.space, vf, 0.0, 1.0)


def keep_connected(s: Sink, fins):
    """Drop fin pieces that do not touch the base (they carry no heat)."""
    solid = s.base | (fins & s.space)
    lbl, _ = ndimage.label(solid)
    keep = np.unique(lbl[s.base])
    return np.isin(lbl, keep[keep > 0]) & s.space


LBM = {"cs": 0.17}


def rate(s: Sink, fins, dp, guess, name):
    """Two flow rates bracketing `dp` (log-log interpolation of the
    resistance to it): the flow search's runs are each rated, then the
    resistance at exactly dp is read off the line through the two nearest."""
    from loadpath.conjugate import duct_flow

    solid, fluid = s.masks(fins)
    air = FLUIDS["air"]
    t0 = time.time()
    pts = []
    q = guess
    for i in range(3):
        fl = duct_flow(fluid, s.h, q, "air", **LBM)
        if not np.isfinite(fl["velocity"]).all():
            # a jet through a narrow gap can still blow the LBM up at cs
            # 0.17: more Smagorinsky damping for this run
            fl = duct_flow(fluid, s.h, q, "air", **{**LBM, "cs": 0.25})
            if not np.isfinite(fl["velocity"]).all():
                raise RuntimeError(f"{name}: lattice Boltzmann diverged at {q:.1f} L/min")
        ht = solve_heat(solid, fluid, fl["velocity"], s.h, CONDUCTIVITY["alsi10mg"], air, s.die, POWER)
        pts.append((q, fl["dp"], ht["resistance"], ht["meanResistance"], ht["balance"]))
        lo = [p for p in pts if p[1] <= dp]
        hi = [p for p in pts if p[1] > dp]
        if lo and hi:
            break
        slope = 1.5
        if len(pts) > 1:
            (q0, d0, *_), (q1, d1, *_) = pts[-2], pts[-1]
            slope = float(np.clip(np.log(d1 / d0) / np.log(q1 / q0), 1.0, 2.0))
        # aim 15% past the target so the next run brackets it
        aim = dp * (1.15 if not hi else 1 / 1.15)
        q = q * (aim / fl["dp"]) ** (1.0 / slope)
    a = max((p for p in pts if p[1] <= dp), key=lambda p: p[1], default=None)
    b = min((p for p in pts if p[1] > dp), key=lambda p: p[1], default=None)
    if a is None or b is None:
        a, b = sorted(pts, key=lambda p: abs(np.log(p[1] / dp)))[:2]
    w = np.log(dp / a[1]) / np.log(b[1] / a[1])
    at = lambda i: float(np.exp((1 - w) * np.log(a[i]) + w * np.log(b[i])))
    out = {"design": name, "vf": s.vf(fins), "dp": dp, "flowLpm": at(0), "resistance": at(2),
           "meanResistance": at(3), "balance": [p[4] for p in pts],
           "runs": [{"flowLpm": p[0], "dp": p[1], "resistance": p[2]} for p in pts], "seconds": time.time() - t0}
    print(json.dumps(out), flush=True)
    return out, ht, fl


def main():
    global DIE, DIE_X, FIN_H
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=0.5)
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--vf", type=float, default=0.2)
    ap.add_argument("--cells", default="8")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="bench/lattice_heatsink.json")
    # a second geometry: die size and position, fin height
    ap.add_argument("--die", type=float, default=DIE)
    ap.add_argument("--die-x", type=float, default=DIE_X)
    ap.add_argument("--fin-h", type=float, default=FIN_H)
    a = ap.parse_args()
    DIE, DIE_X, FIN_H = a.die, a.die_x, a.fin_h
    s = Sink(a.pitch)
    # resume: designs already rated in --out are kept, not rerun
    done = {}
    if Path(a.out).exists():
        done = {r["design"]: r for r in json.loads(Path(a.out).read_text())["results"]}
    results = list(done.values())
    designs = {}
    only = set(a.only.split(",")) - {""}

    def want(n):
        return not only or n in only

    if want("plates"):
        designs["plates"] = plate_fins(s, a.vf)
        print(json.dumps({"platesVf": s.vf(designs["plates"])}), flush=True)
    if want("solid-to"):
        designs["solid-to"] = solid_to(s, a.vf)
    if want("flow-to"):
        designs["flow-to"] = flow_to(s, a.vf, a.dp)
    cells = [float(c) for c in a.cells.split(",")]
    for c in cells:
        if want(f"gyroid-{c:g}"):
            designs[f"gyroid-{c:g}"] = gyroid_uniform(s, a.vf, c)
    guess = 100.0

    def save():
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"pitch": a.pitch, "dp": a.dp, "vf": a.vf, "power": POWER,
                                           "die": DIE, "dieX": DIE_X, "finH": FIN_H,
                                           "results": results}, indent=1))

    for name, fins in designs.items():
        if name in done and not name.startswith("gyroid-"):
            continue
        if name in done:
            # the graded ones need the uniform lattice's temperature field
            out, ht, fl = rate(s, fins, a.dp, done[name]["flowLpm"], name)
        else:
            out, ht, fl = rate(s, fins, a.dp, guess, name)
            results.append(out)
            save()
        np.save(Path(a.out).with_suffix(f".{name}.npy"), np.packbits(fins))
        guess = out["flowLpm"]
        tag = f"graded-{name[7:]}"
        if name.startswith("gyroid-") and (want(tag) or any(o.startswith(tag + "-") for o in only)):
            # grade by the heat flux through the uniform lattice's metal:
            # |grad T| x k in the solid, smoothed over a cell
            T = np.nan_to_num(ht["T"])
            g = np.stack(np.gradient(T, s.h), 0)
            flux = np.where(fins | s.base, np.linalg.norm(g, axis=0), 0.0)
            c = float(name[7:])
            sm = ndimage.uniform_filter(flux, size=int(c / s.h)) / np.maximum(
                ndimage.uniform_filter((fins | s.base).astype(float), size=int(c / s.h)), 1e-3)
            grades = {f"graded-{c:g}-flux{e:g}": np.nan_to_num(np.clip(sm, 0, None)) ** e for e in (0.25, 0.5)}
            # and by height: denser at the base, where all the heat enters
            zf = (s.Z - BASE) / FIN_H
            for r in (2.0, 3.0):
                grades[f"graded-{c:g}-z{r:g}"] = r - (r - 1.0) * zf
            for gname, field in grades.items():
                if gname in done or (only and gname not in only and tag not in only):
                    continue
                gf = gyroid_graded(s, a.vf, c, field)
                o2, _, _ = rate(s, gf, a.dp, guess, gname)
                results.append(o2)
                save()
                np.save(Path(a.out).with_suffix(f".{gname}.npy"), np.packbits(gf))


if __name__ == "__main__":
    main()
