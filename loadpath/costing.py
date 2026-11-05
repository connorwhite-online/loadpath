"""
What a part costs to make, and how long until it is in hand, for each way
of making it: the numbers `explore` (explore.py) ranks outcomes by, next to
mass and safety factor.

These are ESTIMATES of a service-bureau quote (2026 USD), good to tell a
$20 part from a $200 one and to see where a method's fixed costs stop
mattering as the quantity grows; not a quote. Every rate below is a typical
published or quoted figure for the method, rounded, and every one can be
overridden (`explore.costs`, or `estimate(..., rates=...)`). A real quote
(Materialize's `materialize_get_quote`, a machine shop) replaces them.

What each model counts, per part at `quantity` parts:

  fdm        filament (mass x 1.1 for purge and support) x $/kg, printer hours
             (the slicer's when the run sliced it, else volume over a solid
             deposition rate plus a tenth of an hour per 10 mm of height for
             travel and layer changes) x $/h, and a handling fee.
  sls, mjf   powder for the part (part volume x $/cm^3, refresh included),
             the share of a build it occupies (its bounding box x $/cm^3)
             and a handling fee; built in a shared batch, so lead time is
             set by the bureau's queue, not the part.
  sla, dlp,  resin (volume x 1.3 for supports) x $/cm^3, printer hours
  jetting    (height / layer x seconds per layer) x $/h, wash and cure fee.
  volumetric as SLA, without supports.
  lpbf       powder (mass x 1.3 for supports) x $/kg, machine hours (volume /
             melt rate + height / layer x recoat seconds) x $/h, a build
             set-up share and post-processing (stress relief, cut from the
             plate, support removal) per part.
  cast       tooling (pattern and core boxes) once per order, metal (mass x 2
             for gates and risers) x $/kg, foundry labour per part, and
             machining of the preserved interfaces (bores, faces) per part.
  milled     programming once per order, a fixture set-up per side the cutter
             comes from, the stock (bounding box + 3 mm a side) x $/kg, and
             machine hours: the removed volume over a roughing rate plus the
             part's surface over a finishing rate.

Lead time: the method's queue (days to start and ship) plus its machine
hours for the whole order at `hoursPerDay`.
"""
from __future__ import annotations

import copy

import numpy as np

__all__ = ["RATES", "MATERIAL_PRICE", "estimate", "method_family"]

# $/kg of feedstock, by material (filament, powder, bar stock, ingot)
MATERIAL_PRICE = {
    "pla": 25, "petg": 25, "abs": 25, "asa": 30, "tpu": 40, "pa-cf": 70, "pp": 40,
    "pa12": 60, "pa11": 70, "pa12-gb": 60, "pa12-gf": 70, "tpu88a": 90,
    "resin": 150, "resin-tough": 175, "resin-rigid": 250, "vero": 300, "agilus30": 350,   # $/kg ~ $/L
    "alsi10mg": 80, "ti64": 350, "316l": 60, "in625": 120, "15-5ph": 70,
    "a356-t6": 4, "6061-t6": 9, "7075-t6": 16, "304": 6,
}

# per method; see the module docstring for what each number is
RATES = {
    "fdm": {"materialFactor": 1.1, "machinePerHour": 3.0, "depositionCm3PerHour": 12.0,
            "hoursPer10mmHeight": 0.1, "perPart": 5.0, "queueDays": 2.0, "hoursPerDay": 20.0},
    "sls": {"partPerCm3": 0.45, "boxPerCm3": 0.04, "perPart": 10.0, "queueDays": 6.0},
    "mjf": {"partPerCm3": 0.40, "boxPerCm3": 0.04, "perPart": 10.0, "queueDays": 5.0},
    "sla": {"materialFactor": 1.3, "machinePerHour": 5.0, "layerMm": 0.05, "secondsPerLayer": 9.0,
            "perPart": 15.0, "queueDays": 3.0, "hoursPerDay": 20.0},
    "dlp": {"materialFactor": 1.3, "machinePerHour": 5.0, "layerMm": 0.05, "secondsPerLayer": 5.0,
            "perPart": 15.0, "queueDays": 3.0, "hoursPerDay": 20.0},
    "jetting": {"materialFactor": 1.6, "machinePerHour": 12.0, "layerMm": 0.03, "secondsPerLayer": 9.0,
                "perPart": 15.0, "queueDays": 5.0, "hoursPerDay": 20.0},
    "volumetric": {"materialFactor": 1.0, "machinePerHour": 20.0, "secondsPerPart": 120.0,
                   "perPart": 15.0, "queueDays": 5.0, "hoursPerDay": 16.0},
    "lpbf": {"materialFactor": 1.3, "machinePerHour": 60.0, "meltCm3PerHour": 18.0, "layerMm": 0.04,
             "recoatSeconds": 8.0, "buildSetup": 150.0, "postPerPart": 80.0, "queueDays": 10.0,
             "hoursPerDay": 22.0},
    "cast": {"tooling": 900.0, "materialFactor": 2.0, "perPart": 25.0, "interfaceMachining": 30.0,
             "queueDays": 25.0, "hoursPerDay": 8.0, "hoursPerPart": 0.5},
    "milled": {"programming": 180.0, "setupPerSide": 45.0, "stockMarginMm": 3.0,
               "roughCm3PerMinute": 8.0, "finishCm2PerMinute": 3.0,
               "machinePerHour": {"2.5": 70.0, "3": 80.0, "5": 120.0},
               "queueDays": 6.0, "hoursPerDay": 16.0},
}
# per-method melt rate where the alloy differs much from AlSi10Mg (cm^3/h)
_LPBF_MELT = {"ti64": 8.0, "316l": 6.0, "in625": 5.0, "15-5ph": 6.0}


def method_family(method: str) -> str:
    """'milled-3axis' -> 'milled'; a process name stays itself."""
    m = str(method).lower()
    return "milled" if m.startswith("milled") or m.startswith("mill") else m


def _rates(family: str, rates: dict | None) -> dict:
    r = copy.deepcopy(RATES[family])
    for k, v in ((rates or {}).get(family) or {}).items():
        if k == "machinePerHour" and isinstance(r.get(k), dict) and isinstance(v, (int, float)):
            r[k] = {a: float(v) for a in r[k]}
        elif isinstance(r.get(k), dict) and isinstance(v, dict):
            r[k].update(v)
        else:
            r[k] = v
    return r


def _price(material: str, rates: dict | None) -> float:
    over = (rates or {}).get("materialPrice") or {}
    if material in over:
        return float(over[material])
    if material in MATERIAL_PRICE:
        return float(MATERIAL_PRICE[material])
    base = material.split("-custom")[0]
    return float(over.get(base, MATERIAL_PRICE.get(base, 50.0)))


def _geometry(mesh, build=None) -> dict:
    """Volume (cm^3), surface (cm^2), bounding box (mm, cm^3) and height
    along the build direction (mm)."""
    lo, hi = mesh.bounds
    ext = np.asarray(hi) - np.asarray(lo)
    vol = float(abs(mesh.volume)) / 1000.0
    if build is None:
        height = float(ext[2])
    else:
        if isinstance(build, (tuple, list)) and len(build) == 2 and not isinstance(build[0], (list, tuple)):
            d = np.zeros(3)
            d[int(build[0])] = float(build[1])
        else:
            d = np.asarray(build, float)
        d = d / (np.linalg.norm(d) or 1.0)
        h = np.asarray(mesh.vertices) @ d
        height = float(h.max() - h.min())
    return {"volume": vol, "surface": float(mesh.area) / 100.0, "extent": ext.tolist(),
            "box": float(np.prod(ext)) / 1000.0, "height": height}


def estimate(method: str, material: str, mesh=None, *, mass: float | None = None, quantity: int = 1,
             build=None, milling_sides: int = 1, milling_axes: float = 3, print_seconds: float | None = None,
             rates: dict | None = None, geometry: dict | None = None) -> dict:
    """{"unitCost", "orderCost", "fixedCost", "machineHours", "leadDays",
    "breakdown", ...} for `quantity` parts of `mesh` made by `method` in
    `material`. `mass` (g) is the verified mass; `print_seconds` the
    slicer's print time when there is one (FDM)."""
    fam = method_family(method)
    if fam == "milled" and str(method).lower().startswith("milled-"):
        milling_axes = float(str(method).lower()[7:].replace("axis", "") or milling_axes)
    if fam not in RATES:
        raise ValueError(f"no cost model for {method!r}; known: {sorted(RATES)}")
    q = max(1, int(quantity))
    r = _rates(fam, rates)
    g = geometry or _geometry(mesh, build)
    kg = (mass if mass is not None else g["volume"]) / 1000.0
    price = _price(material, rates)
    fixed = 0.0
    parts = {}
    hours = 0.0
    note = None
    if fam == "fdm":
        if print_seconds:
            hours = print_seconds / 3600.0
            note = "printer hours from the slicer"
        else:
            hours = g["volume"] / r["depositionCm3PerHour"] + r["hoursPer10mmHeight"] * g["height"] / 10.0
            note = "printer hours estimated (no slicer run): solid deposition rate and height"
        parts = {"material": kg * r["materialFactor"] * price, "machine": hours * r["machinePerHour"],
                 "handling": r["perPart"]}
    elif fam in ("sls", "mjf"):
        parts = {"powder": g["volume"] * r["partPerCm3"], "buildShare": g["box"] * r["boxPerCm3"],
                 "handling": r["perPart"]}
        note = "built in a shared batch: lead time is the bureau's queue"
    elif fam in ("sla", "dlp", "jetting"):
        hours = g["height"] / r["layerMm"] * r["secondsPerLayer"] / 3600.0
        parts = {"material": g["volume"] * r["materialFactor"] * price / 1000.0 * _density_cm3(mass, g),
                 "machine": hours * r["machinePerHour"], "handling": r["perPart"]}
    elif fam == "volumetric":
        hours = r["secondsPerPart"] / 3600.0
        parts = {"material": g["volume"] * price / 1000.0 * _density_cm3(mass, g),
                 "machine": hours * r["machinePerHour"], "handling": r["perPart"]}
    elif fam == "lpbf":
        melt = (rates or {}).get("lpbf", {}).get("meltCm3PerHour") or _LPBF_MELT.get(material, r["meltCm3PerHour"])
        hours = g["volume"] * r["materialFactor"] / melt + g["height"] / r["layerMm"] * r["recoatSeconds"] / 3600.0
        fixed = r["buildSetup"]
        parts = {"powder": kg * r["materialFactor"] * price, "machine": hours * r["machinePerHour"],
                 "postProcessing": r["postPerPart"]}
        note = "recoating is paid per build; parts sharing a plate share it (counted once per order here)"
    elif fam == "cast":
        fixed = r["tooling"]
        hours = r["hoursPerPart"]
        parts = {"metal": kg * r["materialFactor"] * price, "foundry": r["perPart"],
                 "interfaceMachining": r["interfaceMachining"]}
        note = "tooling (pattern, core boxes) is paid once per order"
    elif fam == "milled":
        m = float(r["stockMarginMm"])
        ext = np.asarray(g["extent"], float) + 2.0 * m
        stock_cm3 = float(np.prod(ext)) / 1000.0
        density = (mass / g["volume"]) if mass and g["volume"] > 0 else 2.7
        removed = max(0.0, stock_cm3 - g["volume"])
        hours = removed / r["roughCm3PerMinute"] / 60.0 + g["surface"] / r["finishCm2PerMinute"] / 60.0
        key = "2.5" if float(milling_axes) == 2.5 else str(int(float(milling_axes)))
        rate = r["machinePerHour"][key] if isinstance(r["machinePerHour"], dict) else float(r["machinePerHour"])
        fixed = r["programming"] + r["setupPerSide"] * max(1, int(milling_sides))
        parts = {"stock": stock_cm3 * density / 1000.0 * price, "machine": hours * rate}
        note = (f"stock {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm; programming and {milling_sides} "
                "fixture set-up(s) paid once per order")
        if float(milling_axes) == 5:
            note += "; 5-axis reaches every side from 2 set-ups (3+2 positioning)"
    unit_var = float(sum(parts.values()))
    order = unit_var * q + fixed
    lead = float(r["queueDays"]) + hours * q / float(r.get("hoursPerDay", 24.0))
    out = {
        "method": method, "material": material, "quantity": q,
        "unitCost": round(order / q, 2), "orderCost": round(order, 2), "fixedCost": round(fixed, 2),
        "machineHours": round(hours, 3) if hours else None,
        "leadDays": round(lead, 1),
        "breakdown": {k: round(float(v), 2) for k, v in parts.items()},
        "materialPricePerKg": price,
        "geometry": {k: (round(v, 2) if isinstance(v, float) else [round(x, 1) for x in v]) for k, v in g.items()},
    }
    if note:
        out["note"] = note
    return out


def _density_cm3(mass, g) -> float:
    """g/cm^3 of the part (resin priced by kg ~ litre)."""
    if mass and g["volume"] > 0:
        return mass / g["volume"]
    return 1.15


def crossover(rows: list[dict], quantities=(1, 10, 100, 1000)) -> list[dict]:
    """Which method is cheapest at each quantity, from estimate() rows that
    carry fixedCost and orderCost at quantity 1 (unit cost scales linearly
    above the fixed part)."""
    out = []
    for q in quantities:
        best = None
        for row in rows:
            fixed = row["fixedCost"]
            var = row["orderCost"] - fixed
            unit = (var * q + fixed) / q
            if best is None or unit < best[1]:
                best = (row, unit)
        if best:
            out.append({"quantity": q, "route": best[0].get("route"), "unitCost": round(best[1], 2)})
    return out
