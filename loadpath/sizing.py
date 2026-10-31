"""
Size a two-fluid TPMS heat exchanger for a duty: the core's dimensions
inside an envelope, the cell, the wall, the split of the void between the
fluids and each fluid's ports, under a pressure-drop budget per fluid and
the print process's limits. Seconds: it sweeps the epsilon-NTU rating
(heatx_thermal.rate), corrected for what the whole-exchanger simulation
and the LBM showed the rating misses, and the best few can then be
designed and simulated whole (`verify`).

What the rating misses, and how it is put back:

- Edge bands. Outside its own zones each fluid is sealed within `cap` of
  the core's surface, and partial channels next to that seal carry
  almost nothing (porous.py, fitted to LBM of the 30 mm crossflow cube).
  So each sealed face loses a band about `cap + cell / 4` deep from the
  core that actually works. On a core a few cells across that is a third
  of it; it is why small cells (more cells across) win more than the
  rating alone says.
- Ports. A port that jets into its plenum loses more than the rating's
  1.5 velocity heads, and on the cube a 6 mm port at 1 L/min lost more
  at the core's face than the porous medium can show. Each port is sized
  so its velocity heads stay under a share of the fluid's budget
  (`port_share`), as large as its face allows.
- Plenums. The port turns into its plenum through a ring pi d around and
  `header` deep; a velocity head of that speed in and out is added. On
  Connor's intercooler duty (0.143 kg/s charge air) a 6 mm plenum behind
  a 47 mm port cost more than the core, so the header depth is swept too
  (6-20 mm) and pays for itself out of the envelope.

Checked whole (porous.simulate_box, 3 mm voxels) on intercooler-sized
cores 200 x 70-100 x 62 mm: effectiveness within 5% (the sim reads
higher), core pressure drop within 10%.

The envelope is the part's bounding box. Each face with a zone gives up
header + shell + port length to its plenum and port; the other faces a
shell.

Process limits (walls that hold pressure and stay leak-tight, channels
that drain powder or resin): `PROCESS_LIMITS`.
"""
from __future__ import annotations

import itertools

import numpy as np

from .heatx_thermal import WALL_DENSITY, _flow, _fluid, rate, tpms_geometry

__all__ = ["PROCESS_LIMITS", "size_exchanger", "verify"]

# per process: thinnest leak-tight wall (mm), narrowest channel (hydraulic diameter, mm,
# to clear powder or drain resin), and the heatx process that builds it
PROCESS_LIMITS = {
    "fdm": {"minWall": 0.8, "minChannel": 1.5, "heatx": "fdm"},
    "lpbf": {"minWall": 0.5, "minChannel": 2.0, "heatx": "lpbf"},
    "sls": {"minWall": 0.8, "minChannel": 2.5, "heatx": "lpbf"},
    "mjf": {"minWall": 0.8, "minChannel": 2.5, "heatx": "lpbf"},
    "sla": {"minWall": 0.6, "minChannel": 1.5, "heatx": "lpbf"},
    "dlp": {"minWall": 0.6, "minChannel": 1.5, "heatx": "lpbf"},
}

from .porous import _BAND_CELLS  # noqa: E402  (the dead band under a sealed face, in cells, beyond the seal)


# port speed to aim for (m/s) when the face has room: whole-exchanger sims put 3-5
# velocity heads in the headers, so a port sized only to the budget leaves a
# liquid side over it (intercooler coolant: 11.5 mm port, 18 kPa of headers; 22 mm, 1.8)
_PORT_SPEED = {"liquid": 1.0, "gas": 25.0}


def _port(m, fl, dp_budget, share, d_max, d_min=4.0):
    """The port (mm, 0.5 mm steps): wide enough that its 1.5 velocity heads
    stay under `share` of the budget and its speed under _PORT_SPEED, as the
    face allows. Returns (d, whether the face capped it below the budget's)."""
    v_aim = _PORT_SPEED["liquid" if fl.rho > 100 else "gas"]
    if dp_budget is None or not np.isfinite(dp_budget):
        v = v_aim
    else:
        v = np.sqrt(share * dp_budget / (0.75 * fl.rho))
    d_need = 2e3 * np.sqrt(m / (fl.rho * np.pi * v))
    d = max(d_need, min(2e3 * np.sqrt(m / (fl.rho * np.pi * v_aim)), d_max))
    d = max(d_min, np.ceil(d * 2.0) / 2.0)
    return float(min(d, d_max)), d_need > d_max


def size_exchanger(duty: dict, envelope, *, process: str = "fdm", max_pressure_drop: dict | None = None,
                   layout: str = "auto", kind: str = "gyroid", cells=None, walls=None, splits=(0.4, 0.5, 0.6, 0.66),
                   min_effectiveness: float | None = None, target_duty: float | None = None,
                   objective: str = "mass", header: float | None = None, shell: float | None = None,
                   port_length: float = 8.0, cap: float | None = None, port_share: float = 0.25,
                   fractions=(0.4, 0.55, 0.7, 0.85, 1.0), keep: int = 8) -> dict:
    """Sweep the core's dimensions (fractions of what the envelope leaves),
    cell, wall, split and layout for `duty` ({"A": {"fluid", "kgPerS" |
    "lPerMin", "tIn"}, "B": {...}, "wall": material}). `envelope` (mm) is
    the part's [x, y, z] extent; one fluid runs along x and the other
    along y (cross), or back along the first with side ports on the other
    (counter), both ways round; z carries no ports. Keeps designs within
    `max_pressure_drop` ({fluid: Pa}) and, if given, `min_effectiveness`
    or `target_duty` (W). `objective`: "mass" (lightest that meets them),
    "effectiveness" (best), or "pressure" (least of the worst fluid's
    share of its budget). Returns the best few, each with a heatx spec
    ready to design ("spec")."""
    if process not in PROCESS_LIMITS:
        raise ValueError(f"process is one of {sorted(PROCESS_LIMITS)}")
    if objective not in ("mass", "effectiveness", "pressure"):
        raise ValueError('objective is "mass", "effectiveness" or "pressure"')
    lim = PROCESS_LIMITS[process]
    names = [n for n in duty if n != "wall"]
    if len(names) != 2:
        raise ValueError("duty names two fluids")
    fls = [_fluid(duty[n].get("fluid", "water")) for n in names]
    ms = [_flow(duty[n], f) for n, f in zip(names, fls)]
    t_in = [duty[n].get("tIn") for n in names]
    caps = [m * f.cp for m, f in zip(ms, fls)]
    if target_duty is not None:
        if None in t_in:
            raise ValueError("target_duty needs each fluid's tIn")
        min_effectiveness = max(min_effectiveness or 0.0, target_duty / (min(caps) * abs(t_in[0] - t_in[1])))
    budget = max_pressure_drop or {}
    wall_mat = duty.get("wall", "petg")
    density = WALL_DENSITY.get(wall_mat) if isinstance(wall_mat, str) else None
    shell = shell if shell is not None else max(1.6, 2 * lim["minWall"])
    env = np.asarray(envelope, float)
    headers = [float(header)] if header is not None else [6.0, 9.0, 12.0, 16.0, 20.0]
    walls = walls if walls is not None else [w for w in (0.5, 0.6, 0.8, 1.0, 1.2, 1.5) if w >= lim["minWall"]]
    cells = cells if cells is not None else [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0]
    layouts = ["cross", "counter"] if layout == "auto" else [layout]
    rows, rejected = [], 0
    for hd, lay, axes in itertools.product(headers, layouts, ("xy", "yx")):
        allow = hd + shell + port_length
        # what the envelope leaves the core: A's zones on its axis, B's on the other (both
        # layouts); `room` is in A's frame (A along its first entry)
        e = env if axes == "xy" else env[[1, 0, 2]]
        room = np.array([e[0] - 2 * allow, e[1] - 2 * allow, e[2] - 2 * shell])
        if (room <= 0).any():
            continue
        for fx, fy, fz in itertools.product(fractions, repeat=3):
            dims = room * np.array([fx, fy, fz])
            for a, t, sp in itertools.product(cells, walls, splits):
                if a < 3 * t or min(dims) < a:
                    continue
                cp_ = cap if cap is not None else t + 1.0
                try:
                    geo = tpms_geometry(kind, a, t, sp)
                except ValueError:
                    continue
                if min(g["dh"] for g in geo["sides"]) < lim["minChannel"]:
                    continue
                # the band each sealed face loses, and the core that works
                band = cp_ + _BAND_CELLS * a
                eff_dims = dims - 2 * band
                if (eff_dims <= 0.25 * dims).any():
                    continue
                v_eff = float(np.prod(eff_dims))
                if lay == "cross":
                    lengths = [dims[0], dims[1]]
                    frontal = [eff_dims[1] * eff_dims[2], eff_dims[0] * eff_dims[2]]
                    faces = [(dims[1], dims[2]), (dims[0], dims[2])]
                else:
                    lengths = [dims[0], dims[0]]
                    frontal = [eff_dims[1] * eff_dims[2], eff_dims[1] * eff_dims[2]]
                    faces = [(dims[1], dims[2]), (0.4 * dims[0], dims[2])]
                ports, too_big = [], False
                for i, n in enumerate(names):
                    d_max = min(faces[i]) - 2 * cp_ - 2.0
                    d, over = _port(ms[i], fls[i], budget.get(n, np.inf), port_share, d_max)
                    ports.append(d)
                    too_big |= over
                # the port turns into the plenum through a ring pi d deep `header`: a
                # velocity head of that speed (whole-exchanger sims put most of the
                # header loss there; a shallow plenum on a big air port dominates)
                if min(ports) < 3.0:
                    continue
                # rate the working core: its volume, each fluid's frontal area through
                # its full length (so the dead bands still cost pressure)
                r = rate(kind, a, t, v_eff, [v_eff / f for f in frontal],
                         {**{n: duty[n] for n in names}, "wall": wall_mat},
                         "counter" if lay == "counter" else "cross", tuple(ports), split=sp)
                dp, plen = {}, {}
                for i, n in enumerate(names):
                    s = r["fluids"][n]
                    v_ring = ms[i] / (fls[i].rho * np.pi * ports[i] * 1e-3 * hd * 1e-3)
                    plen[n] = 2 * 0.5 * fls[i].rho * v_ring ** 2  # in and out
                    dp[n] = s["pressureDropCore"] * lengths[i] / (v_eff / frontal[i]) + \
                        s["pressureDropPorts"] + s["pressureDropMouths"] + plen[n]
                solid = float(np.prod(dims)) * (1.0 - geo["phiSheet"])
                world = dims if axes == "xy" else dims[[1, 0, 2]]
                row = {"layout": lay, "axes": axes, "core": np.round(world, 1).tolist(), "cell": float(a), "wall": float(t),
                       "split": float(sp), "cap": cp_, "header": hd,
                       "pressureDropPlenums": plen, "ports": dict(zip(names, ports)),
                       "effectiveness": r["effectiveness"], "effectivenessRange": r["effectivenessRange"],
                       "duty": r.get("duty"), "tOut": r.get("tOut"), "pressureDrop": dp,
                       "workingCore": round(v_eff / float(np.prod(dims)), 3),
                       "solidVolume": solid, "mass": solid * density / 1000.0 if density else None,
                       "portLimited": bool(too_big)}
                ok = all(dp[n] <= budget.get(n, np.inf) for n in names) and \
                    (min_effectiveness is None or r["effectiveness"] >= min_effectiveness)
                if ok:
                    rows.append(row)
                else:
                    rejected += 1
    if objective == "mass":
        rows.sort(key=lambda x: (x["solidVolume"], -x["effectiveness"]))
    elif objective == "effectiveness":
        rows.sort(key=lambda x: -x["effectiveness"])
    else:
        rows.sort(key=lambda x: max(x["pressureDrop"][n] / budget.get(n, 1.0) for n in names))
    best = []
    for x in rows:
        # skip near-duplicates (same cell, wall and layout) so the list shows real options
        if any(b["cell"] == x["cell"] and b["wall"] == x["wall"] and b["layout"] == x["layout"]
               and b["axes"] == x["axes"] for b in best):
            continue
        x["spec"] = _spec(x, names, kind, x["header"], shell, port_length, lim["heatx"], duty)
        best.append(x)
        if len(best) >= keep:
            break
    return {"best": best[0] if best else None, "candidates": best, "objective": objective,
            "considered": len(rows) + rejected, "feasible": len(rows), "process": process,
            "minEffectiveness": min_effectiveness}


def _spec(row, names, kind, header, shell, port_length, heatx_process, duty):
    c = row["core"]
    return {"name": f"sized-{row['layout']}-{row['cell']:g}-{row['wall']:g}",
            "core": {"type": "box", "min": [0, 0, 0], "max": c}, "type": kind, "layout": row["layout"],
            "axes": row["axes"], "fluids": list(names), "cell": row["cell"], "wall": row["wall"],
            "split": row["split"], "cap": row["cap"], "header": header, "shell": shell,
            "portLength": port_length, "portDiameter": row["ports"], "process": heatx_process,
            "duty": duty}


def verify(candidate: dict, voxel=None, closures="design", progress=None, max_points=None, method="auto"):
    """Simulate a sized candidate whole (porous.py): returns the
    simulation's effectiveness, pressure drops and flow sharing next to the
    sizing's own. `method` "design" designs it first (heatx, wall-resolving:
    small cores); "box" builds the porous grid straight from the spec
    (porous.box_domain, cross layout, any size); "auto" takes "box" for a
    cross core over 1e5 mm3. The box sim's header losses read 3-5 port
    velocity heads at 3 mm voxels where cell-resolved LBM of the tuned cube
    measured ~1.4: take its header dp as an upper bound."""
    from .heatx import heat_exchanger_from_spec
    from .porous import simulate_box, simulate_exchanger

    spec = dict(candidate["spec"])
    duty = spec.pop("duty")
    hx = heat_exchanger_from_spec(spec)
    if max_points:
        hx.max_points = int(max_points)
    if method == "auto":
        method = "box" if spec.get("layout") == "cross" and float(np.prod(candidate["core"])) > 1e5 else "design"
    if method == "box":
        sim = simulate_box(hx, duty, voxel=voxel, closures=closures, progress=progress)
        rep = {}
    else:
        _, rep, sim = simulate_exchanger(hx, duty, voxel=voxel, closures=closures, progress=progress)
    return {"sized": {k: candidate[k] for k in ("effectiveness", "pressureDrop", "tOut")},
            "simulated": {"effectiveness": sim.get("effectiveness"),
                          "pressureDrop": {n: f["pressureDrop"] for n, f in sim["fluids"].items()},
                          "tOut": {n: f.get("tOut") for n, f in sim["fluids"].items()},
                          "midplaneCv": {n: f["midplane"]["cv"] for n, f in sim["fluids"].items()}},
            "design": {"isolated": rep.get("isolated"), "connected": rep.get("connected"),
                       "minWallBetweenFluids": rep.get("minWallBetweenFluids"), "solidVolume": rep.get("solidVolume"),
                       "warnings": rep.get("warnings")},
            "method": method, "seconds": sim.get("seconds")}
