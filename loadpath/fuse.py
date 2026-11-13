"""
Designed weak links (`fuses` in a study): a printed part that is meant to break, at a force chosen so that
everything behind it survives.

A crash brings more energy than a light part can store intact. A drone's duct, a robot's bumper cover, a
bracket that holds a shelf on a wall, a tool's breakaway handle: the honest design is a link that lets go
before the expensive part does, then gets replaced. Aircraft hang their engines on fuse pins for the same
reason, and roadside signposts stand on slip bases. Two numbers make one work:

- it must HOLD every service load, with the service safety factor, at the weakest a print of it comes out;
- it must BREAK below what the protected part can carry, at the strongest a print of it comes out, loaded at
  impact rate (polymers are stronger the faster they are loaded).

The band between the weakest and the strongest print is the fuse's window. The library sizes the link to
the smallest section that still holds service (so the window sits as low as it can), reports the window,
and caps every crash case that runs through the link at the window's top: the hit cannot put more on the
part behind it than the link passes before it breaks (`impact.maxForce`). Whether that cap is low enough is
then the protected part's own design problem, solved by the same study.

Two links (v1):

- `shear-pin`: a printed pin through lugs on both sides, cut on `planes` shear planes (1 single shear,
  2 double). Breaks at planes x tau_u x pi d^2 / 4, tau_u = strength / sqrt(3) (von Mises; ductile
  polymers and metals shear at 0.55-0.6 of their tensile strength).
- `tension-link`: a waisted tab pulled along its length. Breaks at strength x width x thickness.

Strength is the material card's (failure.py's layer-bond factor when the break runs along a layer plane,
`acrossLayers`). Scatter is the coefficient of variation of a printed coupon's strength (STRENGTH_SCATTER
by process; a calibrated card's own `scatter` wins); the window is +-WINDOW_SIGMA of it. What happens after
the link breaks (the rest of the body arriving at the wall) is a second hit, and a study has to give it as
its own case.
"""
from __future__ import annotations

import copy
import math

import numpy as np

__all__ = ["FUSE_TYPES", "STRENGTH_SCATTER", "RATE_FACTOR", "WINDOW_SIGMA", "break_force", "size_fuse",
           "expand_fuses", "fuse_pin_mesh"]

FUSE_TYPES = ("shear-pin", "tension-link")

# coefficient of variation of a printed coupon's tensile strength, by process: (cov, source, quality).
# Powder beds and resins are repeatable; filament is set by its layer bonds.
STRENGTH_SCATTER = {
    "sls": (0.05, "SLS PA12 coupons typically 46-50 MPa within a build, a few percent between builds", "assumed"),
    "mjf": (0.04, "MJF PA12 as SLS, slightly tighter", "assumed"),
    "fdm": (0.10, "filament: in-plane 5-10%, layer bonds 10-20% between prints", "assumed"),
    "sla": (0.06, "cured resin coupons", "assumed"),
    "dlp": (0.06, "cured resin coupons", "assumed"),
    "volumetric": (0.06, "cured resin coupons", "assumed"),
    "jetting": (0.08, "photopolymer jetting", "assumed"),
    "lpbf": (0.04, "as-built metal powder bed", "assumed"),
}
DEFAULT_SCATTER = (0.08, "no process given: a middle value", "assumed")

# strength at impact rate over the quasi-static datasheet value, by material family (failure.py's families):
# the top of the window is read at it, the bottom (service loads are slow) is not
RATE_FACTOR = {
    "semicrystalline": (1.3, "nylons and polypropylene: yield 10-30% higher at impact rates (docs/IMPACT.md); "
                             "the high end, so the protected part is designed for the strongest break",
                        "assumed"),
    "glassy": (1.3, "glassy polymers: 10-30% higher at impact rates; the high end", "assumed"),
    "elastomer": (1.4, "elastomers stiffen and strengthen with rate (viscoelastic.py's crush-time factor)",
                  "assumed"),
    "metal": (1.1, "aluminium and steels: a few to 10% at crash rates", "assumed"),
}
# the window spans +-2 standard deviations: about 95% of prints break inside it
WINDOW_SIGMA = 2.0
# the smallest fuse section, in the process's minimum features: a printed pin under about two walls across
# does not come out round, and a tab under one wall does not come out at all
PIN_MIN_FEATURES = 2.0
TAB_MIN_FEATURES = 1.0
STEP = 0.05                      # mm: sizes are rounded up to this


def _process(process):
    if isinstance(process, dict):
        return process.get("type") or process.get("kind")
    return process


def _scatter(material, process, scatter=None):
    if scatter is not None:
        return float(scatter), "given", "measured"
    card = getattr(material, "scatter", None)
    if card:
        return float(card), "the material card's", "measured"
    return STRENGTH_SCATTER.get(_process(process), DEFAULT_SCATTER)


def _rate(material):
    from .failure import _family

    return RATE_FACTOR.get(_family(material), RATE_FACTOR["semicrystalline"])


def _strength(material, process, across_layers):
    from .spec import PROCESSES

    s = float(material.strength)
    if not across_layers:
        return s, 1.0
    kind = _process(process)
    # a datasheet's across-layer strength holds for the process its specimens were made by
    if getattr(material, "strength_z", None) is not None and (not kind or getattr(material, "measured", "") in
                                                              ("", kind)):
        return float(material.strength_z), float(material.strength_z) / s
    f = float(PROCESSES.get(kind, {}).get("interlayer", 1.0))
    return s * f, f


def break_force(kind: str, size, material="pa12", process="sls", planes: int = 1, count: int = 1,
                across_layers: bool = False) -> float:
    """The nominal (quasi-static, mean) force in N that breaks `count` links in parallel: a shear pin of
    diameter `size` (mm) cut on `planes` planes, or a tension link of `size` = (width, thickness) mm."""
    from .materials import material_from_spec

    mat = material_from_spec(material)
    s, _ = _strength(mat, process, across_layers)
    if kind == "shear-pin":
        return count * planes * s / math.sqrt(3.0) * math.pi * float(size) ** 2 / 4.0
    if kind == "tension-link":
        w, t = size
        return count * s * float(w) * float(t)
    raise ValueError(f"fuse type {kind!r}: one of {FUSE_TYPES}")


def size_fuse(kind: str, service_force: float, material="pa12", process="sls", service_sf: float = 2.0,
              planes: int = 1, count: int = 1, across_layers: bool = False, thickness: float | None = None,
              protect_force: float | None = None, scatter: float | None = None) -> dict:
    """The smallest link that holds `service_force` (N, the largest service load through it) at
    `service_sf` on its weakest print, and the window it then breaks in.

    `thickness` (mm) fixes a tension link's thickness and sizes its width (default: square). With
    `protect_force` (N, the most the part behind can take) it also says whether the window's top is under
    it. Returns {size, breakNominal, breakLow, breakHigh, window, holds, protects, why...}."""
    from .materials import material_from_spec
    from .spec import PROCESSES

    if kind not in FUSE_TYPES:
        raise ValueError(f"fuse type {kind!r}: one of {FUSE_TYPES}")
    mat = material_from_spec(material)
    cov, cov_src, cov_q = _scatter(mat, process, scatter)
    rate, rate_src, rate_q = _rate(mat)
    low_f = max(1.0 - WINDOW_SIGMA * cov, 0.05)
    high_f = (1.0 + WINDOW_SIGMA * cov) * rate
    need = service_sf * max(float(service_force), 0.0) / low_f        # nominal break force that holds service
    s, bond = _strength(mat, process, across_layers)
    feat = float(PROCESSES.get(_process(process), {}).get("minFeature", 1.0))

    def up(x):
        return math.ceil(x / STEP - 1e-9) * STEP

    if kind == "shear-pin":
        d_need = math.sqrt(need * 4.0 / (count * planes * s / math.sqrt(3.0) * math.pi)) if need > 0 else 0.0
        floor = PIN_MIN_FEATURES * feat
        size = up(max(d_need, floor))
        limited = "print" if d_need < floor else "service"
        nominal = break_force(kind, size, mat, process, planes, count, across_layers)
        geom = {"diameter": round(size, 3), "planes": int(planes)}
    else:
        floor = TAB_MIN_FEATURES * feat
        t = float(thickness) if thickness else max(up(math.sqrt(need / (count * s))) if need > 0 else 0.0, floor)
        t = max(t, floor)
        w_need = need / (count * s * t) if need > 0 else 0.0
        w = up(max(w_need, floor))
        limited = "print" if w_need < floor else "service"
        size = (w, t)
        nominal = break_force(kind, size, mat, process, planes, count, across_layers)
        geom = {"width": round(w, 3), "thickness": round(t, 3)}
    lo, hi = nominal * low_f, nominal * high_f
    out = {"type": kind, "count": int(count), **geom, "material": mat.name, "process": _process(process),
           "acrossLayers": bool(across_layers), "strength": round(s, 2),
           "serviceForce": round(float(service_force), 3), "serviceSafetyFactor": service_sf,
           "breakNominal": round(nominal, 2), "breakLow": round(lo, 2), "breakHigh": round(hi, 2),
           "window": round(hi / lo, 3), "sizedBy": limited,
           "serviceMargin": round(lo / max(service_force, 1e-9), 2) if service_force > 0 else None,
           "scatter": {"cov": cov, "sigmas": WINDOW_SIGMA, "source": cov_src, "quality": cov_q},
           "rate": {"factor": rate, "source": rate_src, "quality": rate_q}}
    if across_layers:
        out["layerBond"] = round(bond, 3)
    if protect_force is not None:
        out["protectForce"] = float(protect_force)
        out["protects"] = bool(hi <= protect_force)
        if not out["protects"]:
            out["why"] = (f"the strongest print breaks at {hi:.0f} N, over the {protect_force:.0f} N the part behind "
                          f"takes; the window is {hi / lo:.2f}x wide, so the service load through it can be at most "
                          f"{protect_force / (hi / lo) / service_sf:.1f} N at SF {service_sf}")
    if limited == "print":
        out["note"] = (f"the service load needs less than the process prints ({floor:.2f} mm): the link is at the "
                       f"printable minimum, so it breaks higher than the load needs")
    return out


def _region_force(case, region):
    """The resultant force (N) of a case's loads on `region` (loads that name it), and how many there were."""
    tot = np.zeros(3)
    n = 0
    for ld in case.get("loads") or []:
        if isinstance(ld, dict) and ld.get("region") == region and ld.get("force") is not None:
            tot += np.asarray(ld["force"], float)
            n += 1
    return tot, n


def _through(kind, f, axis):
    """The part of force `f` that works the link: across a pin (the axial part is carried by its heads), along
    a tension link (a push is carried in bearing)."""
    if axis is None:
        return float(np.linalg.norm(f)), 0.0
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    along = float(f @ a)
    if kind == "shear-pin":
        return float(np.linalg.norm(f - along * a)), along
    return max(along, 0.0), along


def _hit_energy(imp: dict) -> float | None:
    """An impact block's energy in N mm (J x 1000), when it is given or follows from mass and velocity."""
    if imp.get("energy") is not None:
        return float(imp["energy"]) * 1000.0
    if imp.get("velocity") is not None and imp.get("mass") is not None:
        return 0.5 * float(imp["mass"]) * float(imp["velocity"]) ** 2 * 1000.0
    return None


def expand_fuses(d: dict) -> dict:
    """The study with each fuse sized from its service cases and every crash case through it capped at the
    window's top (`impact.maxForce`); the sizing is recorded in d["_fuses"] (report.fuses). A fuse:

        {"id", "type": "shear-pin" | "tension-link", "region": <the region its loads act on: the breakaway
         part bears on this part there>, "axis": [x, y, z] (pin axis / link direction), "count", "planes",
         "acrossLayers", "thickness", "material" (default the study's), "serviceSafetyFactor" (default the
         goal's), "holds": [service case names] (default: every case without an impact block),
         "breaks": [crash case names] (default: every impact case with a load on the region)}"""
    if d.get("_fuses") is not None or not d.get("fuses"):
        return d
    d = copy.deepcopy(d)
    cases = d.get("loadCases") or []
    by_name = {c.get("name"): c for c in cases if isinstance(c, dict)}
    proc = d.get("process")
    goal_sf = float((d.get("goal") or {}).get("safetyFactor") or 2.0)
    rows = []
    for i, fz in enumerate(d["fuses"]):
        lab = f"fuses[{i}]"
        if not isinstance(fz, dict):
            raise ValueError(f"{lab} is {{id, type, region, axis, count, ...}}")
        kind = fz.get("type", "shear-pin")
        if kind not in FUSE_TYPES:
            raise ValueError(f"{lab}.type is one of {FUSE_TYPES}")
        reg = fz.get("region")
        if not isinstance(reg, str):
            raise ValueError(f"{lab}.region names the region (a preserve id) the breakaway part's loads act on")
        fid = fz.get("id") or f"fuse{i + 1}"
        axis = fz.get("axis")
        holds = fz.get("holds")
        if holds is None:
            holds = [c.get("name") for c in cases if isinstance(c, dict) and c.get("impact") is None]
        breaks = fz.get("breaks")
        if breaks is None:
            breaks = [c.get("name") for c in cases if isinstance(c, dict) and c.get("impact") is not None
                      and _region_force(c, reg)[1] > 0]
        for nm in list(holds) + list(breaks):
            if nm not in by_name:
                raise ValueError(f"{lab}: no load case named {nm!r}")
        # each held case at its own safety factor (a crash case the link must ride out usually carries a
        # lower one than flight); an explicit serviceSafetyFactor applies to all of them
        sf_all = fz.get("serviceSafetyFactor")
        service, service_sf, axial, need = 0.0, float(sf_all or goal_sf), 0.0, 0.0
        for nm in holds:
            c = by_name[nm]
            f, _ = _region_force(c, reg)
            work, along = _through(kind, f, axis)
            sf_c = float(sf_all or c.get("safetyFactor") or goal_sf)
            if sf_c * work > need:
                need, service, service_sf = sf_c * work, work, sf_c
            axial = max(axial, abs(along))
        sized = size_fuse(kind, service, fz.get("material") or d.get("material") or "pa12", proc,
                          service_sf, int(fz.get("planes", 1)),
                          int(fz.get("count", 1)), bool(fz.get("acrossLayers", False)), fz.get("thickness"),
                          fz.get("protectForce"), fz.get("scatter"))
        cap = sized["breakHigh"]
        capped, warn = [], []
        for nm in breaks:
            c = by_name[nm]
            f, n = _region_force(c, reg)
            work, along = _through(kind, f, axis)
            if c.get("impact") is not None:
                imp = c["impact"]
                # the link passes no more than its strongest print breaks at; across a pin that is its
                # shear share, so the whole hit is capped in proportion
                share = work / max(float(np.linalg.norm(f)), 1e-12) if n else 1.0
                total = cap / max(share, 1e-6)
                prev = imp.get("maxForce")
                imp["maxForce"] = round(min(total, prev) if prev else total, 3)
                row_c = {"case": nm, "maxForce": imp["maxForce"], "acrossShare": round(share, 3)}
                e_hit = _hit_energy(imp)
                if e_hit is not None:
                    # the crusher alone up to the cap (the part's own give adds a little: the analysis
                    # reports releaseEnergy with it); breaking the link stops nothing
                    from .impact import release_energy
                    cr = imp.get("crusher") or {}
                    e_rel = release_energy(imp["maxForce"], 0.0, cr.get("distance"), cr.get("force"))
                    row_c["hitEnergy"] = round(e_hit / 1000.0, 3)
                    row_c["releaseEnergy"] = round(min(e_rel, e_hit) / 1000.0, 3)
                    left = max(e_hit - e_rel, 0.0) / 1000.0
                    row_c["energyPastRelease"] = round(left, 3)
                    if left > 0.02 * e_hit / 1000.0:
                        warn.append(f"{nm}: the link lets go with {left:.2f} J of {e_hit / 1000:.2f} J still "
                                    "moving; the next contact (what the body hits once the part is gone) needs "
                                    "its own case")
                capped.append(row_c)
            elif work > cap:
                # a plain force case over the window: the link is gone before it is reached
                k = cap / work
                for ld in c.get("loads") or []:
                    if isinstance(ld, dict) and ld.get("region") == reg and ld.get("force") is not None:
                        ld["force"] = [round(v * k, 4) for v in ld["force"]]
                capped.append({"case": nm, "scaledTo": round(cap, 2), "from": round(work, 2)})
        row = {"id": fid, "region": reg, **sized, "serviceSafetyFactor": service_sf, "holds": list(holds),
               "breaks": capped}
        if warn:
            row["warnings"] = warn
        if kind == "shear-pin" and axial > 0.3 * max(service, 1e-9) and axial > 0:
            row.setdefault("warnings", []).append(
                f"{axial:.1f} N of the service load runs along the pin: its heads or a shoulder must carry it")
        if not holds:
            row.setdefault("warnings", []).append("no service case runs through it: it is sized to the printable "
                                                  "minimum, so it breaks at whatever that section holds")
        rows.append(row)
    d["_fuses"] = rows
    return d


def fuse_pin_mesh(diameter: float, length: float, head: float | None = None, head_height: float | None = None,
                  segments: int = 48):
    """A printable shear pin (mm, axis z from 0 to `length`): the sized shank with a head 1.6x its
    diameter (head_height 0.6x), the end that seats against the outer lug. Watertight."""
    import trimesh

    head = head or 1.6 * diameter
    hh = head_height or 0.6 * diameter
    shank = trimesh.creation.cylinder(radius=diameter / 2, height=length, sections=segments)
    shank.apply_translation([0, 0, length / 2])
    cap = trimesh.creation.cylinder(radius=head / 2, height=hh, sections=segments)
    cap.apply_translation([0, 0, length + hh / 2 - 1e-3])
    m = trimesh.boolean.union([shank, cap], engine="manifold")
    return m
