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
    gyroid, 20 <= Re <= 100, 0.3 <= phi <= 0.7, +-15%)

where U_s is the superficial velocity (flow over the core's frontal area),
phi the fluid's share of the core volume and a the cell. Heat transfer:

    Nu = 0.49 Re^0.62 Pr^0.4
    (Reynolds, Fee, Morison, Holland 2023, IJHMT 212:124264; printed polymer
    sheet gyroid, water and air, 100 < Re < 2500; Re and Nu on
    Dh = 4 V_f / A_wet and the interstitial velocity U_s / phi)

and, as a second opinion, Nu = 0.148 Re^0.704 Pr^0.395 (Cordisco et al.
2025, SAE 2025-24-0015, CFD, velocity 2 U_s / phi_sheet). The literature
scatters by up to 50% between gyroid datasets, so the rating reports both
and designs on the lower. Diamond uses Cordisco's 0.255 Re^0.640 Pr^0.328;
Schwarz P falls back to the gyroid's (a rough stand-in, flagged).

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


def tpms_geometry(kind: str, cell: float, wall: float) -> dict:
    """One fluid's side of a sheet TPMS, analytically: its share of the
    volume, wetted area per volume (1/mm) and hydraulic diameter (mm)."""
    s = _SURFACE[kind]
    phi_sheet = 1.0 - s * wall / cell
    if phi_sheet <= 0.2:
        raise ValueError(f"a {wall:g} mm wall closes a {cell:g} mm {kind}")
    phi = 0.5 * phi_sheet
    beta = s / cell
    return {"phi": phi, "phiSheet": phi_sheet, "areaDensity": beta, "dh": 4.0 * phi / beta}


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
        nu_main = 0.255 * re ** 0.640 * pr ** 0.328
        nu_alt = nu_main
    else:
        nu_main = 0.49 * re ** 0.62 * pr ** 0.4
        re2 = fluid.rho * (2 * us / phi_sheet) * dh_m / fluid.mu
        nu_alt = 0.148 * re2 ** 0.704 * pr ** 0.395
        if kind != "gyroid":
            flags.append(f"{kind}: gyroid correlations used")
    if not 100 <= re <= 2500:
        flags.append(f"Re {re:.0f} is outside the Nu fit (100-2500)")
    nu = min(nu_main, nu_alt)
    h = nu * fluid.k / dh_m
    a = cell * 1e-3
    k_perm = 0.0157 * phi ** 2.72 * a ** 2
    cf = 0.0784 * phi ** -2.04
    dpdx = fluid.mu * us / k_perm + fluid.rho * cf * us ** 2 / np.sqrt(k_perm)
    dp_core = dpdx * length * 1e-3
    v_port = m / (fluid.rho * port_area)
    dp_ports = 1.5 * 0.5 * fluid.rho * v_port ** 2
    dp_in, dp_out, k_in = _mouths(fluid, us, phi, dh, mouth_radius)
    if not 20 <= re <= 100 or not 0.3 <= phi <= 0.7:
        flags.append("pressure drop extrapolated past its fit (Re 20-100, phi 0.3-0.7)")
    return {"superficialVelocity": us, "re": re, "pr": pr, "nu": nu, "nuSpread": [min(nu_main, nu_alt),
            max(nu_main, nu_alt)], "h": h, "pressureDrop": dp_core + dp_ports + dp_in + dp_out,
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
         mouth_radius=0.0):
    """Rate one design. `core_volume` mm^3; `lengths` (mm) each fluid's flow
    path through the core; `duty` = {"A": {"fluid": "water", "lPerMin": 2,
    "tIn": 80}, "B": {...}, "wall": "petg" | W/mK}. `geometry` overrides the
    analytic one per fluid (from a design report: phi, areaDensity, dh).
    `lengths` and `geometry` follow the order the fluids are named in `duty`.
    `mouth_radius` (mm) rounds the channels' mouths (a design's blend)."""
    kw = duty.get("wall", "petg")
    k_wall = float(WALL_CONDUCTIVITY[kw] if isinstance(kw, str) else kw)
    names = [n for n in duty if n != "wall"]
    if len(names) != 2:
        raise ValueError("duty names two fluids, e.g. {\"A\": {...}, \"B\": {...}}")
    base = tpms_geometry(kind, cell, wall)
    port_area = np.pi * (0.5 * port_diameter * 1e-3) ** 2
    sides, caps, areas = {}, [], []
    for i, n in enumerate(names):
        geo = dict(base)
        if geometry is not None:
            geo.update(geometry[i])
        fl = _fluid(duty[n].get("fluid", "water"))
        m = _flow(duty[n], fl)
        frontal = core_volume / lengths[i] * 1e-6        # m^2
        s = _side(kind, fl, m, geo["phi"], geo.get("phiSheet", 2 * geo["phi"]), geo["dh"], cell, lengths[i],
                  frontal, port_area, mouth_radius)
        s["massFlow"] = m
        sides[n] = s
        caps.append(m * fl.cp)
        areas.append(geo["areaDensity"] * core_volume * 1e-6)      # 1/mm * mm^3 = mm^2 -> m^2
    area = float(np.mean(areas))
    r_wall = wall * 1e-3 / k_wall
    u = 1.0 / (1.0 / sides[names[0]]["h"] + r_wall + 1.0 / sides[names[1]]["h"])
    ua = u * area
    cmin, cmax = min(caps), max(caps)
    ntu = ua / cmin
    cr = cmin / cmax
    eff = float(_effectiveness(ntu, cr, arrangement))
    out = {"U": u, "UA": ua, "area": area, "NTU": ntu, "Cr": cr, "effectiveness": eff,
           "wallResistanceShare": r_wall / (1.0 / u), "arrangement": arrangement, "fluids": sides}
    t = [duty[n].get("tIn") for n in names]
    if None not in t:
        q = eff * cmin * (t[0] - t[1])
        out["duty"] = q
        out["tOut"] = {names[0]: t[0] - q / caps[0], names[1]: t[1] + q / caps[1]}
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


def size_core(kind, core_volume, lengths, duty, arrangement="cross", port_diameter=8.0, min_wall=0.8,
              max_pressure_drop=None, cells=None, walls=None, min_channel=1.5):
    """Sweep cell and wall for a core and duty; return the candidates that
    meet the pressure-drop limits ({"A": Pa, "B": Pa}) and print (the wall
    at least `min_wall`, channels at least `min_channel` mm across), best
    effectiveness first, with the rest kept for reference."""
    cells = cells if cells is not None else np.arange(4.0, 20.01, 1.0)
    walls = walls if walls is not None else np.round(np.arange(min_wall, 2.01, 0.2), 2)
    names = [n for n in duty if n != "wall"]
    ok, rejected = [], []
    for a in cells:
        for t in walls:
            if t < min_wall or a < 3 * t:
                continue
            try:
                geo = tpms_geometry(kind, a, t)
            except ValueError:
                continue
            if geo["dh"] < min_channel:
                continue
            r = rate(kind, a, t, core_volume, lengths, duty, arrangement, port_diameter)
            row = {"cell": float(a), "wall": float(t), "effectiveness": r["effectiveness"], "UA": r["UA"],
                   "dh": geo["dh"], "pressureDrop": {n: r["fluids"][n]["pressureDrop"] for n in names},
                   "duty": r.get("duty")}
            lim = max_pressure_drop or {}
            if all(row["pressureDrop"][n] <= lim.get(n, np.inf) for n in names):
                ok.append(row)
            else:
                rejected.append(row)
    ok.sort(key=lambda x: -x["effectiveness"])
    return {"best": ok[0] if ok else None, "candidates": ok[:10], "rejectedForPressure": len(rejected)}
