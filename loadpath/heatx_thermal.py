"""
Thermal and hydraulic rating of a sheet TPMS heat exchanger, and sizing its
cell and wall: effectiveness-NTU, with the heat transfer coefficient and
the pressure drop from published correlations for gyroid channels.

Each fluid fills one side of the sheet. Seen from one fluid, its side is
the void of a network (solid) gyroid of the same porosity, so pressure
drop uses a Darcy-Forchheimer fit for network gyroids:

    dp/dx = mu U_s / K + rho C_F U_s^2 / sqrt(K),  K = 0.0157 phi^2.72 a^2,
    C_F = 0.0784 phi^-2.04
    (Savoldi, Cammi, Gajetti, Marocco 2026, Int. J. Heat Fluid Flow 121:110631;
    gyroid, 20 <= Re <= 100, 0.3 <= phi <= 0.7, +-15%. Against the 1KZ
    cooler's LBM cells, gyroid sheets, it holds to 6-9% median (0.79-1.20)
    out to Re 1,500 and down to phi 0.17, so it is flagged only past those)

where U_s is the superficial velocity (flow over the core's frontal area),
phi the fluid's share of the core volume and a the cell. Heat transfer:

    Nu = 0.49 Re^0.62 Pr^0.4
    (Reynolds, Fee, Morison, Holland 2023, IJHMT 212:124264; printed polymer
    sheet gyroid, water and air, 100 < Re < 2500; Re and Nu on
    Dh = 4 V_f / A_wet and the interstitial velocity U_s / phi)

which the rating designs on. Two more fits bound it:

    Nu = 1.66 Re^0.464 Pr^0.217
    (LBM, periodic gyroid-sheet cells, fully developed, isothermal wall with
    a uniform heat source in the fluid; 39 points, 8-16 mm cells at 1.0-1.5 mm
    walls, air Re 190-1470 and water Re 11-72; within 7% of every point.
    From the 1KZ charge-cooler project, connorwhite-online/intercooler.
    Same Dh and velocity as above; it runs 1.27x Reynolds 2023 on air, 1.53x
    on water)

    Nu = 0.148 Re^0.704 Pr^0.395
    (Cordisco et al. 2025, SAE 2025-24-0015, CHT RANS of whole exchangers;
    velocity 2 U_s / phi = U_s / phi_i, Dh = 4 V_f / A_wall)

Cordisco's h is not a local coefficient: it is the exchanger's heat over
its wall area times (mean wall temperature - the arithmetic mean of inlet
and outlet bulk temperatures). When one side's own NTU is large, as it is in
a TPMS core, the arithmetic mean overstates the driving difference (at an
NTU of 4 by about 2x against the log mean), so it reads 2-4x below the
local fits, more so at low Re. It is the low end of the reported band
(nuSpread), not a design value. Diamond uses Cordisco's 0.255 Re^0.640
Pr^0.328 (the only diamond fit, so conservative); Schwarz P falls back to
the gyroid's (a rough stand-in, flagged).

The two networks need not share the void equally: `split` (fluid A's share
of the void, default 0.5) moves the sheet off the mid-surface toward B. An
air side wants the bigger share against a liquid. The geometry is then
measured on one periodic cell (phi, wetted area and Dh per side).

The wall is a plane conductor (it is thin against the channels), so
1/U = 1/h_A + t/k_wall + 1/h_B on the wetted area. Effectiveness is the
counterflow formula for the "counter" layout and the unmixed-unmixed cross
flow formula otherwise. Port losses add 1.5 velocity heads per port pair;
the channels' mouths add a contraction in (less with rounded mouths, after
Idelchik's rounded inlets) and a Borda-Carnot expansion out.

Units: SI inside, mm for geometry at the edges (cells, walls, lengths).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Fluid", "FLUIDS", "WALL_CONDUCTIVITY", "tpms_geometry", "rate", "rate_design", "size_core"]


@dataclass
class Fluid:
    name: str
    rho: float      # kg/m^3
    cp: float       # J/kg K
    k: float        # W/m K
    mu: float       # Pa s

    @property
    def pr(self):
        return self.cp * self.mu / self.k


# near room temperature; pass a Fluid for anything else
FLUIDS = {
    "water": Fluid("water", 997.0, 4180.0, 0.60, 0.89e-3),
    "water60": Fluid("water60", 983.0, 4185.0, 0.65, 0.47e-3),
    "air": Fluid("air", 1.18, 1007.0, 0.026, 1.85e-5),
    "glycol50": Fluid("glycol50", 1065.0, 3300.0, 0.38, 3.5e-3),
    "oil": Fluid("oil", 870.0, 1900.0, 0.14, 0.03),
}

# W/m K, as printed (dense). Polymers: Tychanicz-Kwiecien et al. 2025 for PLA
# and PETG; PA, PP and PA-CF from datasheets. Metals: LPBF as built.
WALL_CONDUCTIVITY = {"pla": 0.19, "petg": 0.20, "pa": 0.27, "pp": 0.18, "pa-cf": 0.4, "resin": 0.18,
                     "alsi10mg": 125.0, "316l": 13.0}

# mid-surface area per period cube, in a^2
_SURFACE = {"gyroid": 3.09, "diamond": 3.84, "schwarz-p": 2.35}


# fit ranges: Nu (Reynolds 2023 100-2500, the LBM cells down to 11), pressure drop
# (Savoldi, extended by the LBM cells to Re 1,500 and phi 0.17)
_NU_RE = (10.0, 2500.0)
_DP_RE = (1.0, 1500.0)
_DP_PHI = (0.15, 0.7)


def tpms_geometry(kind: str, cell: float, wall: float, split: float = 0.5) -> dict:
    """One fluid's side of a sheet TPMS: its share of the volume, wetted
    area per volume (1/mm) and hydraulic diameter (mm). With split 0.5,
    analytic and the same for both sides; otherwise measured on a periodic
    cell, and "sides" holds fluid A's and B's (A has `split` of the void)."""
    s = _SURFACE[kind]
    phi_sheet = 1.0 - s * wall / cell
    if phi_sheet <= 0.2:
        raise ValueError(f"a {wall:g} mm wall closes a {cell:g} mm {kind}")
    if abs(split - 0.5) < 1e-9:
        phi = 0.5 * phi_sheet
        beta = s / cell
        g = {"phi": phi, "phiSheet": phi_sheet, "areaDensity": beta, "dh": 4.0 * phi / beta}
        return {**g, "sides": [dict(g), dict(g)], "offset": 0.0}
    cellg = _cell_geometry(kind, round(wall / cell, 4), round(float(split), 4))
    sides = [{"phi": p, "phiSheet": cellg["phiSheet"], "areaDensity": a / cell, "dh": 4.0 * p / (a / cell)}
             for p, a in zip(cellg["phi"], cellg["area"])]
    return {**sides[0], "sides": sides, "offset": cellg["offset"] * cell}


_CELL_CACHE = {}


def _cell_field(kind):
    from .lattice import tpms

    return lambda pts: tpms(kind, pts, 1.0)


def sheet_offset(field, wall: float, split: float, n: int = 48) -> float:
    """Level (in the field's units, cell = 1) to move a sheet of thickness
    `wall` (a fraction of the cell) to, so that the side where field > level
    gets `split` of the void. `field(pts)` is a signed distance-like TPMS
    field over a unit cell."""
    if abs(split - 0.5) < 1e-9:
        return 0.0
    ax = (np.arange(n) + 0.5) / n
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    d = np.asarray(field(pts), float)
    h = 0.5 * wall

    def share(c):
        a = np.count_nonzero(d > c + h)
        b = np.count_nonzero(d < c - h)
        return a / max(a + b, 1)

    lo, hi = float(d.min()), float(d.max())
    if not share(lo) >= split >= share(hi):
        raise ValueError(f"no sheet of {wall:g} cells gives fluid A {split:g} of the void")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if share(mid) > split:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cell_geometry(kind, wall, split, n=48):
    """phi and wetted area (per unit cell, cell = 1) of each side of a sheet
    `wall` thick (cells) moved so that A has `split` of the void."""
    key = (kind, wall, split)
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    from skimage.measure import marching_cubes

    field = _cell_field(kind)
    c = sheet_offset(field, wall, split, n)
    ax = (np.arange(n + 1)) / n                   # closed grid, so the periodic faces close
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    d = np.asarray(field(pts), float).reshape(n + 1, n + 1, n + 1)
    h = 0.5 * wall
    inner = d[:-1, :-1, :-1]
    phi = [float(np.mean(inner > c + h)), float(np.mean(inner < c - h))]
    area = []
    for lvl in (c + h, c - h):
        v, f, _, _ = marching_cubes(d, lvl, spacing=(1.0 / n,) * 3)
        tri = v[f]
        area.append(float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()))
    out = {"phi": phi, "area": area, "phiSheet": phi[0] + phi[1], "offset": c}
    _CELL_CACHE[key] = out
    return out


def _flow(spec, fluid):
    """kg/s from {"kgPerS"} or {"lPerMin"}."""
    if "kgPerS" in spec:
        return float(spec["kgPerS"])
    if "lPerMin" in spec:
        return float(spec["lPerMin"]) / 60000.0 * fluid.rho
    raise ValueError("each fluid's duty needs a flow: \"kgPerS\" or \"lPerMin\"")


def _fluid(v):
    if isinstance(v, Fluid):
        return v
    if v not in FLUIDS:
        raise ValueError(f"fluid must be one of {sorted(FLUIDS)} or a Fluid, got {v!r}")
    return FLUIDS[v]


# loss coefficient of an inlet from a large space by its edge radius over Dh (Idelchik,
# rounded inlet flush with a wall): sharp 0.5 down to 0.04 once r/Dh reaches 0.15
_ENTRY = (np.array([0.0, 0.02, 0.04, 0.06, 0.10, 0.15]), np.array([0.5, 0.28, 0.24, 0.15, 0.09, 0.04]))


def _mouths(fluid, us, phi, dh, mouth_radius):
    """Pressure lost getting from the plenum into the channels and back out:
    a contraction to the open share `phi` of the face, its loss cut by
    rounded mouths (r/Dh), and a Borda-Carnot expansion out, which rounding
    barely helps. Both on the channel velocity U_s / phi."""
    q = 0.5 * fluid.rho * (us / phi) ** 2
    k_in = float(np.interp(mouth_radius / dh, *_ENTRY)) * (1.0 - phi)
    k_out = (1.0 - phi) ** 2
    return k_in * q, k_out * q, k_in


def _side(kind, fluid, m, phi, phi_sheet, dh, cell, length, frontal, port_area, mouth_radius=0.0):
    """One fluid's channel: velocities, Re, h (both correlations), pressure drop."""
    us = m / (fluid.rho * frontal)                       # superficial, m/s
    dh_m = dh * 1e-3
    re = fluid.rho * (us / phi) * dh_m / fluid.mu
    pr = fluid.pr
    flags = []
    if kind == "diamond":
        nu = 0.255 * re ** 0.640 * pr ** 0.328
        fits = {"cordisco2025": nu}
    else:
        nu = 0.49 * re ** 0.62 * pr ** 0.4
        fits = {"reynolds2023": nu, "lbmCells": 1.66 * re ** 0.464 * pr ** 0.217,
                "cordisco2025": 0.148 * re ** 0.704 * pr ** 0.395}
        if kind != "gyroid":
            flags.append(f"{kind}: gyroid correlations used")
    if not _NU_RE[0] <= re <= _NU_RE[1]:
        flags.append(f"Re {re:.0f} is outside the Nu fits ({_NU_RE[0]:g}-{_NU_RE[1]:g})")
    h = nu * fluid.k / dh_m
    a = cell * 1e-3
    k_perm = 0.0157 * phi ** 2.72 * a ** 2
    cf = 0.0784 * phi ** -2.04
    dpdx = fluid.mu * us / k_perm + fluid.rho * cf * us ** 2 / np.sqrt(k_perm)
    dp_core = dpdx * length * 1e-3
    v_port = m / (fluid.rho * port_area)
    dp_ports = 1.5 * 0.5 * fluid.rho * v_port ** 2
    dp_in, dp_out, k_in = _mouths(fluid, us, phi, dh, mouth_radius)
    if not _DP_RE[0] <= re <= _DP_RE[1] or not _DP_PHI[0] <= phi <= _DP_PHI[1]:
        flags.append(f"pressure drop extrapolated past its fit (Re up to {_DP_RE[1]:g}, "
                     f"phi {_DP_PHI[0]:g}-{_DP_PHI[1]:g})")
    return {"superficialVelocity": us, "re": re, "pr": pr, "nu": nu, "nuFits": fits,
            "nuSpread": [min(fits.values()), max(fits.values())],
            "hSpread": [min(fits.values()) * fluid.k / dh_m, max(fits.values()) * fluid.k / dh_m],
            "h": h, "pressureDrop": dp_core + dp_ports + dp_in + dp_out,
            "pressureDropCore": dp_core, "pressureDropPorts": dp_ports, "pressureDropMouths": dp_in + dp_out,
            "mouthEntryK": k_in, "flags": flags}


def _effectiveness(ntu, cr, arrangement):
    if arrangement == "counter":
        if abs(1 - cr) < 1e-6:
            return ntu / (1 + ntu)
        e = np.exp(-ntu * (1 - cr))
        return (1 - e) / (1 - cr * e)
    # cross flow, both fluids unmixed
    if cr < 1e-9:
        return 1 - np.exp(-ntu)
    return 1 - np.exp(ntu ** 0.22 / cr * (np.exp(-cr * ntu ** 0.78) - 1))


def rate(kind, cell, wall, core_volume, lengths, duty, arrangement="cross", port_diameter=8.0, geometry=None,
         mouth_radius=0.0, split=0.5):
    """Rate one design. `core_volume` mm^3; `lengths` (mm) each fluid's flow
    path through the core; `duty` = {"A": {"fluid": "water", "lPerMin": 2,
    "tIn": 80}, "B": {...}, "wall": "petg" | W/mK}. `geometry` overrides the
    analytic one per fluid (from a design report: phi, areaDensity, dh).
    `lengths` and `geometry` follow the order the fluids are named in `duty`.
    `mouth_radius` (mm) rounds the channels' mouths (a design's blend).
    `split`: the first fluid's share of the void (the sheet moved toward the
    second). The result's "range" rates the design again at the low and
    high ends of the Nu band (effectivenessRange, tOutRange)."""
    kw = duty.get("wall", "petg")
    k_wall = float(WALL_CONDUCTIVITY[kw] if isinstance(kw, str) else kw)
    names = [n for n in duty if n != "wall"]
    if len(names) != 2:
        raise ValueError("duty names two fluids, e.g. {\"A\": {...}, \"B\": {...}}")
    base = tpms_geometry(kind, cell, wall, split)
    sides, caps, areas = {}, [], []
    for i, n in enumerate(names):
        geo = dict(base["sides"][i])
        if geometry is not None:
            geo.update(geometry[i])
        fl = _fluid(duty[n].get("fluid", "water"))
        m = _flow(duty[n], fl)
        frontal = core_volume / lengths[i] * 1e-6        # m^2
        pd = port_diameter[i] if isinstance(port_diameter, (tuple, list)) else port_diameter
        port_area = np.pi * (0.5 * pd * 1e-3) ** 2
        s = _side(kind, fl, m, geo["phi"], geo.get("phiSheet", 2 * geo["phi"]), geo["dh"], cell, lengths[i],
                  frontal, port_area, mouth_radius)
        s["massFlow"] = m
        sides[n] = s
        caps.append(m * fl.cp)
        areas.append(geo["areaDensity"] * core_volume * 1e-6)      # 1/mm * mm^3 = mm^2 -> m^2
    area = float(np.mean(areas))
    r_wall = wall * 1e-3 / k_wall
    cmin, cmax = min(caps), max(caps)
    cr = cmin / cmax
    t = [duty[n].get("tIn") for n in names]

    def solve(ha, hb):
        u = 1.0 / (1.0 / ha + r_wall + 1.0 / hb)
        ntu = u * area / cmin
        return u, ntu, float(_effectiveness(ntu, cr, arrangement))

    u, ntu, eff = solve(sides[names[0]]["h"], sides[names[1]]["h"])
    out = {"U": u, "UA": u * area, "area": area, "NTU": ntu, "Cr": cr, "effectiveness": eff,
           "wallResistanceShare": r_wall / (1.0 / u), "arrangement": arrangement, "fluids": sides}
    lo_hi = [solve(sides[names[0]]["hSpread"][j], sides[names[1]]["hSpread"][j])[2] for j in (0, 1)]
    out["effectivenessRange"] = lo_hi
    if None not in t:
        q = eff * cmin * (t[0] - t[1])
        out["duty"] = q
        out["tOut"] = {names[0]: t[0] - q / caps[0], names[1]: t[1] + q / caps[1]}
        out["tOutRange"] = {n: sorted(tin - (1 if i == 0 else -1) * e * cmin * (t[0] - t[1]) / caps[i]
                                      for e in lo_hi) for i, (n, tin) in enumerate(zip(names, t))}
    return out


def _lengths(hx, report):
    """Each fluid's path through the core: from its inlet zone's centre to
    its outlet's, on the core's surface (from the design report)."""
    out = []
    for n in hx.names:
        a = np.asarray(report["zones"][f"{n}-in"]["center"])
        b = np.asarray(report["zones"][f"{n}-out"]["center"])
        out.append(float(np.linalg.norm(b - a)))
    return out


def rate_design(hx, report, duty):
    """Rate a designed exchanger from its report: the measured channel
    geometry (share of volume, wetted area, hydraulic diameter) and the
    zones' flow paths."""
    v = report["coreVolume"]
    geo = []
    for n in hx.names:
        r = report[n]
        if not r.get("hydraulicDiameter"):
            raise ValueError(f"fluid {n} reaches none of the core; fix the design (its report) before rating it")
        geo.append({"phi": r["coreVolume"] / v, "areaDensity": r["coreWettedArea"] / v, "dh": r["hydraulicDiameter"],
                    "phiSheet": (report[hx.names[0]]["coreVolume"] + report[hx.names[1]]["coreVolume"]) / v})
    duty = {**{k: v_ for k, v_ in duty.items() if k == "wall"},
            **{n: duty[n] for n in hx.names}}
    return rate(hx.kind, hx.cell, hx.wall, v, _lengths(hx, report), duty,
                "counter" if hx.layout == "counter" else "cross", hx.port_diameter, geo,
                1.2 * float(report.get("blend", 0.0)))      # a smooth union of radius k rounds about 1.2 k


# g/cm^3, for the core's mass in sizing (as printed, dense)
WALL_DENSITY = {"pla": 1.24, "petg": 1.27, "pa": 1.01, "pp": 0.90, "pa-cf": 1.15, "resin": 1.18,
                "alsi10mg": 2.67, "316l": 7.98}


def size_core(kind, core_volume, lengths, duty, arrangement="cross", port_diameter=8.0, min_wall=0.8,
              max_pressure_drop=None, cells=None, walls=None, min_channel=1.5, split=0.5,
              objective="effectiveness", min_effectiveness=None):
    """Sweep cell and wall for a core and duty. Keeps the candidates that
    meet the pressure-drop limits ({"A": Pa, "B": Pa}) and print (the wall
    at least `min_wall`, channels at least `min_channel` mm across) and, if
    given, `min_effectiveness`. Each carries its core's solid volume (mm^3)
    and mass (g, from the duty's wall material). Ranked by `objective`:
    "effectiveness" (best first) or "metal" (least solid first: the
    lightest core that meets the limits; thin the wall and keep the
    hydraulic diameter, rather than grow the cell). `split` is the first
    fluid's share of the void, or a list of shares to sweep too."""
    if objective not in ("effectiveness", "metal"):
        raise ValueError('objective is "effectiveness" or "metal"')
    cells = cells if cells is not None else np.arange(4.0, 20.01, 1.0)
    walls = walls if walls is not None else np.round(np.arange(min_wall, 2.01, 0.2), 2)
    splits = [float(x) for x in np.atleast_1d(split)]
    names = [n for n in duty if n != "wall"]
    wall_mat = duty.get("wall", "petg")
    density = WALL_DENSITY.get(wall_mat) if isinstance(wall_mat, str) else None
    ok, rejected = [], []
    for a in cells:
        for t in walls:
            if t < min_wall or a < 3 * t:
                continue
            for sp in splits:
                try:
                    geo = tpms_geometry(kind, a, t, sp)
                except ValueError:
                    continue
                if min(g["dh"] for g in geo["sides"]) < min_channel:
                    continue
                r = rate(kind, a, t, core_volume, lengths, duty, arrangement, port_diameter, split=sp)
                solid = core_volume * (1.0 - geo["phiSheet"])
                row = {"cell": float(a), "wall": float(t), "split": sp, "effectiveness": r["effectiveness"],
                       "effectivenessRange": r["effectivenessRange"], "UA": r["UA"],
                       "dh": {n: g["dh"] for n, g in zip(names, geo["sides"])},
                       "pressureDrop": {n: r["fluids"][n]["pressureDrop"] for n in names},
                       "duty": r.get("duty"), "solidVolume": solid,
                       "mass": solid * density / 1000.0 if density else None}
                lim = max_pressure_drop or {}
                if not all(row["pressureDrop"][n] <= lim.get(n, np.inf) for n in names):
                    rejected.append(row)
                elif min_effectiveness is not None and row["effectiveness"] < min_effectiveness:
                    rejected.append(row)
                else:
                    ok.append(row)
    if objective == "metal":
        ok.sort(key=lambda x: (x["solidVolume"], -x["effectiveness"]))
    else:
        ok.sort(key=lambda x: -x["effectiveness"])
    return {"best": ok[0] if ok else None, "candidates": ok[:10], "objective": objective,
            "rejected": len(rejected)}
