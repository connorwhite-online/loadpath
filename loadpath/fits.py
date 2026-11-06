"""
Fits: holes that come out the size the mating part needs.

A printed hole is not the hole that was drawn. FDM holes come out a few
tenths small (the nozzle's corner rounding, the outer perimeter squashed
inward), more so across the layers, where the top of the hole sags; powder
sinters onto SLS and MJF hole walls; LPBF downskins grow dross; a cast hole
is only as good as its pattern and shrinks with the metal. And a bearing
needs a different hole from a dowel pin, a sliding shaft or a bolt. This
module turns a fit intent into the hole to draw for the process, and checks
the delivered part's holes against it.

Give it on a cylindrical keep-out (drawn at the mating part's diameter: the
bearing's OD, the pin's, the shaft's) or on a fastener:

    "keepOut": [{"id": "bearing", "shape": {...cylinder r 11...}, "fit": "press"}]
    "fasteners": [{"id": "top", "size": "M5", ..., "fit": "close"}]

Fit intents (hole minus mating diameter, the clearance; negative is an
interference):

  press       the part goes in with a press and stays: a bearing's outer
              ring, a dowel pin. Metal: H7/p6. Plastic: 0.02-0.06 mm plus
              0.2-0.6% of the diameter of interference (plastics creep, so
              the band is wider than the metal one, and a loaded bearing in
              a plastic seat still wants a shoulder or a retaining lip).
  transition  located with no play, goes in by hand or a light tap. H7/k6.
  slip        slides in by hand with no visible play: a locating pin, a
              shaft that is put in and taken out. H7/g6.
  clearance   turns or slides freely: a shaft running in a printed bore, a
              hinge pin. H8/f7.
  close, normal, loose
              bolt holes, ISO 273 fine, medium and coarse (fasteners: normal
              unless set).
  nominal     hold the drawn diameter as built (the default for every
              cylindrical bore keep-out that names no fit, once the process
              is known: a CAD import's holes, an insert's hole, a channel).

The compensation. Each process has a hole bias (the mean of as-built minus
drawn diameter, a little more negative across the layers than along them)
and a scatter band (+-, growing with the diameter). The hole drawn is the
fit's target minus the bias; the report predicts the as-built band from the
delivered mesh's hole (measured, so a coarse tessellation counts) and says
whether the fit holds:

  holds  the whole band sits inside the fit's window;
  tune   the band is wider than the window but a printer that has been
         tuned (its repeatability, about a third of the band) holds it:
         print the hole once, measure, and set process.holeBias;
  ream   neither: the hole is drawn undersize by the band plus stock and
         reamed, drilled or bored to the window (finish "auto" switches to
         this; "as-built" keeps the as-printed hole and says so).

Milled parts are bored to the window directly; cast holes for a fit are
cast undersize and machined. The numbers in HOLES are conservative design
guide ranges (Hubs/Protolabs process guides, the HP MJF and Formlabs
datasheets, typical LPBF and ISO 8062 casting tolerances), not measurements
of any one machine: the first fit off a new printer is worth a test hole,
and process.holeBias / process.holeTolerance take the measured values.
"""
from __future__ import annotations

import copy

import numpy as np

__all__ = ["HOLES", "FITS", "BOLT_HOLES", "expand_fits", "fit_window", "fits_report", "measure_bore"]

# Per process: bias (mm, as built minus drawn hole diameter, axis along the
# build), biasAcross (axis across the layers), tol (+- mm) + tolPerMm x D,
# repeat (the fraction of the band left once a printer's bias is tuned).
HOLES = {
    "fdm": {"bias": -0.20, "biasAcross": -0.35, "tol": 0.10, "tolPerMm": 0.002, "repeat": 0.35,
            "note": "FDM: holes print small (perimeter squash, corner rounding); across the layers the top sags"},
    "sls": {"bias": -0.15, "biasAcross": -0.20, "tol": 0.15, "tolPerMm": 0.002, "repeat": 0.4,
            "note": "SLS: powder sinters onto hole walls; bureau tolerance about +-0.3 mm or 0.3%"},
    "mjf": {"bias": -0.10, "biasAcross": -0.15, "tol": 0.15, "tolPerMm": 0.0015, "repeat": 0.4,
            "note": "MJF: HP quotes +-0.2 mm up to 100 mm; holes close a little"},
    "sla": {"bias": -0.05, "biasAcross": -0.08, "tol": 0.05, "tolPerMm": 0.0015, "repeat": 0.4,
            "note": "resin: slight over-cure closes holes; post-cure shrinks"},
    "dlp": {"bias": -0.05, "biasAcross": -0.08, "tol": 0.05, "tolPerMm": 0.0015, "repeat": 0.4,
            "note": "resin (DLP/LCD): as SLA"},
    "jetting": {"bias": -0.05, "biasAcross": -0.05, "tol": 0.08, "tolPerMm": 0.001, "repeat": 0.4,
                "note": "material jetting: about +-0.1 mm"},
    "volumetric": {"bias": 0.0, "biasAcross": 0.0, "tol": 0.10, "tolPerMm": 0.002, "repeat": 0.5,
                   "note": "volumetric resin: no layers; tolerances still maturing"},
    "lpbf": {"bias": -0.10, "biasAcross": -0.20, "tol": 0.08, "tolPerMm": 0.002, "repeat": 0.4,
             "note": "LPBF metal: dross on downskins closes holes across the layers"},
    "milled": {"bias": 0.0, "biasAcross": 0.0, "tol": 0.01, "tolPerMm": 0.0, "repeat": 1.0,
               "note": "milled: holes bored or reamed to the window"},
    "cast": {"bias": 0.0, "biasAcross": 0.0, "tol": 0.40, "tolPerMm": 0.004, "repeat": 0.5,
             "note": "cast: about ISO 8062 CT9; holes for a fit are cast undersize and machined"},
}
FITS = ("press", "transition", "slip", "clearance", "close", "normal", "loose", "nominal")
_SHAFT_FITS = ("press", "transition", "slip", "clearance")
_BOLT_FITS = ("close", "normal", "loose")
# ISO 273 clearance holes (fine, medium, coarse), mm
BOLT_HOLES = {"M2": (2.2, 2.4, 2.6), "M2.5": (2.7, 2.9, 3.1), "M3": (3.2, 3.4, 3.6), "M4": (4.3, 4.5, 4.8),
              "M5": (5.3, 5.5, 5.8), "M6": (6.4, 6.6, 7.0), "M8": (8.4, 9.0, 10.0), "M10": (10.5, 11.0, 12.0)}
# diametral stock left for a reamer or a boring bar: plastic, metal (mm)
_STOCK = {"plastic": 0.10, "metal": 0.20, "cast": 1.5}
_ISO_STEPS = (0, 3, 6, 10, 18, 30, 50, 80, 120, 180, 250, 315, 400, 500)


def _it(grade: int, d: float) -> float:
    """ISO 286 standard tolerance, mm, for nominal size d (mm), from the
    standard tolerance factor at the geometric mean of d's size step."""
    hi = next((s for s in _ISO_STEPS[1:] if d <= s), _ISO_STEPS[-1])
    lo = _ISO_STEPS[_ISO_STEPS.index(hi) - 1]
    dm = np.sqrt(max(lo, 1.0) * hi)
    i = 0.45 * dm ** (1 / 3) + 0.001 * dm                  # micrometres
    k = {5: 7, 6: 10, 7: 16, 8: 25, 9: 40, 10: 64, 11: 100}[grade]
    return k * i / 1000.0, dm


def _iso_clearance(fit: str, d: float) -> tuple[float, float, str]:
    """(min, max) clearance, mm, of the hole-basis ISO 286 fit for `fit`."""
    it6, dm = _it(6, d)
    it7, _ = _it(7, d)
    if fit == "press":                                   # H7/p6
        ei = it7 + 0.001
        return -(ei + it6), it7 - ei, "H7/p6"
    if fit == "transition":                              # H7/k6
        ei = 0.6 * dm ** (1 / 3) / 1000.0
        return -(ei + it6), it7 - ei, "H7/k6"
    if fit == "slip":                                    # H7/g6
        es = -2.5 * dm ** 0.34 / 1000.0
        return -es, it7 - (es - it6), "H7/g6"
    it8, _ = _it(8, d)                                   # H8/f7
    es = -5.5 * dm ** 0.41 / 1000.0
    return -es, it8 - (es - it7), "H8/f7"


def _plastic_clearance(fit: str, d: float) -> tuple[float, float, str]:
    """(min, max) clearance, mm, for a hole in a printed or moulded plastic
    part against a metal (or hard) mating part: typical practice, wider than
    ISO because plastics creep and print with scatter."""
    if fit == "press":
        return -(0.06 + 0.006 * d), -(0.02 + 0.002 * d), "press (plastic)"
    if fit == "transition":
        return -(0.02 + 0.001 * d), 0.02 + 0.001 * d, "transition (plastic)"
    if fit == "slip":
        return 0.05, 0.12 + 0.003 * d, "slip (plastic)"
    return 0.15 + 0.002 * d, 0.35 + 0.005 * d, "clearance (plastic)"


def _bolt_size(d: float, size: str | None):
    if size in BOLT_HOLES:
        return BOLT_HOLES[size]
    return 1.06 * d, 1.1 * d, 1.2 * d


def fit_window(fit: str, mate: float, family: str = "plastic", size: str | None = None) -> dict:
    """The as-built hole a fit needs: {"lo", "hi", "target", "standard"} in
    mm, for a mating diameter `mate`. `family` is "plastic" or "metal"
    (the part's material); `size` an ISO metric screw ("M5") for bolt fits."""
    if fit in _SHAFT_FITS:
        lo, hi, std = (_iso_clearance if family != "plastic" else _plastic_clearance)(fit, mate)
        lo, hi = float(lo), float(hi)
        return {"lo": mate + lo, "hi": mate + hi, "target": mate + 0.5 * (lo + hi), "standard": std,
                "clearance": [round(lo, 4), round(hi, 4)]}
    if fit in _BOLT_FITS:
        fine, medium, coarse = _bolt_size(mate, size)
        if fit == "close":
            lo, hi, tg = max(mate + 0.1, fine - 0.1), medium, fine
        elif fit == "normal":
            lo, hi, tg = fine, coarse, medium
        else:
            lo, hi, tg = medium, coarse + (coarse - medium), coarse
        return {"lo": lo, "hi": hi, "target": tg, "standard": f"ISO 273 {'fine' if fit == 'close' else 'medium' if fit == 'normal' else 'coarse'}",
                "clearance": [round(lo - mate, 4), round(hi - mate, 4)]}
    if fit == "nominal":
        return {"lo": None, "hi": None, "target": mate, "standard": "as drawn"}
    raise ValueError(f"fit must be one of {list(FITS)}, got {fit!r}")


# --- the process -----------------------------------------------------------

def _process_of(d: dict):
    """(process kind, build direction or None, holes dict) for the fits, or
    (None, None, None) when the study does not say how it is made."""
    if d.get("milling") is not None:
        return "milled", None, dict(HOLES["milled"])
    if d.get("casting") is not None:
        return "cast", None, dict(HOLES["cast"])
    p = d.get("process")
    if p is None:
        return None, None, None
    if isinstance(p, str):
        p = {"type": p}
    kind = str(p.get("type", "")).lower()
    if kind not in HOLES:
        return None, None, None
    holes = dict(HOLES[kind])
    if p.get("holeBias") is not None:
        holes["bias"] = holes["biasAcross"] = float(p["holeBias"])
        holes["calibrated"] = True
    if p.get("holeTolerance") is not None:
        holes["tol"], holes["tolPerMm"], holes["repeat"] = float(p["holeTolerance"]), 0.0, 1.0
        holes["calibrated"] = True
    build = p.get("buildDirection", "+z" if kind in ("fdm", "lpbf") else None)
    if isinstance(build, str):
        s = build.strip().lower()
        if s == "auto":
            build = None
        else:
            ax = {"x": 0, "y": 1, "z": 2}[s.lstrip("+-")]
            v = np.zeros(3)
            v[ax] = -1.0 if s.startswith("-") else 1.0
            build = v
    elif build is not None:
        build = np.asarray(build, float) / np.linalg.norm(build)
    return kind, build, holes


def _family(d: dict) -> str:
    from .materials import material_from_spec

    m = d.get("material", "pla")
    try:
        mat = material_from_spec(m[0] if isinstance(m, list) else m)
    except Exception:
        return "plastic"
    return "plastic" if float(getattr(mat, "density", 1.2)) < 2.0 else "metal"


def _bias(holes: dict, axis, build, d: float):
    """(bias, tol) for a hole of diameter d along `axis`."""
    tol = holes["tol"] + holes["tolPerMm"] * d
    along, across = holes["bias"], holes["biasAcross"]
    if build is None:
        # orientation unknown: the middle, the difference in the band
        return 0.5 * (along + across), tol + 0.5 * abs(along - across), "unknown"
    c = abs(float(np.dot(axis, build)))
    w = float(np.clip((c - 0.4) / 0.5, 0.0, 1.0))         # 1 along the build, 0 across
    return w * along + (1 - w) * across, tol, "along" if w > 0.5 else "across"


def _verdict(lo, hi, centre, tol, repeat):
    if lo is None:
        return "holds"
    if centre - tol >= lo - 1e-9 and centre + tol <= hi + 1e-9:
        return "holds"
    if lo <= centre <= hi and centre - repeat * tol >= lo - 1e-9 and centre + repeat * tol <= hi + 1e-9:
        return "tune"
    return "ream"


# --- the study -------------------------------------------------------------

def _cyl_ends(s: dict):
    if s.get("p0") is not None and s.get("p1") is not None:
        return np.asarray(s["p0"], float), np.asarray(s["p1"], float)
    if s.get("center") is not None and s.get("length") is not None:
        a = s.get("axis", "z")
        if isinstance(a, str):
            v = np.zeros(3)
            v["xyz".index(a.strip().lower().lstrip("+-"))] = 1.0
        else:
            v = np.asarray(a, float) / np.linalg.norm(a)
        c = np.asarray(s["center"], float)
        h = 0.5 * float(s["length"])
        return c - h * v, c + h * v
    return None


def _coaxial(a0, a1, b0, b1) -> bool:
    ua = (a1 - a0) / np.linalg.norm(a1 - a0)
    ub = (b1 - b0) / np.linalg.norm(b1 - b0)
    if abs(abs(float(np.dot(ua, ub))) - 1.0) > 1e-6:
        return False
    w = b0 - a0
    return float(np.linalg.norm(w - np.dot(w, ua) * ua)) < 1e-6


def _pressurized(d: dict) -> set:
    out = set()
    for c in d.get("loadCases") or []:
        for ld in (c.get("loads") or []) if isinstance(c, dict) else []:
            if isinstance(ld, dict) and ld.get("pressure") is not None and ld.get("region"):
                out.add(ld["region"])
    for ld in d.get("loads") or []:
        if isinstance(ld, dict) and ld.get("pressure") is not None and ld.get("region"):
            out.add(ld["region"])
    return out


def expand_fits(d: dict) -> dict:
    """The study with each fitted bore's keep-out resized to the hole the
    process should be given, and the fits recorded in d["_fits"] (the
    report's rows). Coaxial preserved tubes whose inner radius was the
    bore's follow it. Returns d unchanged when there is nothing to do."""
    if d.get("_fits") is not None:
        return d                                          # already expanded (a re-parse of study.raw)
    kos = d.get("keepOut") or []
    explicit = any(isinstance(k, dict) and k.get("fit") not in (None, "none") for k in kos)
    kind, build, holes = _process_of(d)
    if not explicit and kind is None:
        # no fit asked for and no process to compensate for: nothing to change
        if any(isinstance(k, dict) and ("fit" in k or "mate" in k or "finish" in k) for k in kos):
            d = copy.deepcopy(d)
            for k in d["keepOut"]:
                for key in ("fit", "mate", "finish", "_size"):
                    k.pop(key, None)
        return d
    d = copy.deepcopy(d)
    family = _family(d)
    if kind == "cast":
        fam_stock = "cast"
    else:
        fam_stock = family
    press = _pressurized(d)
    rows = []
    for k in d.get("keepOut") or []:
        if not isinstance(k, dict):
            continue
        fit = k.pop("fit", None)
        mate = k.pop("mate", None)
        finish = k.pop("finish", "auto")
        size = k.pop("_size", None)
        sh = k.get("shape") or {}
        if fit == "none" or k.get("bore", True) is False or k.get("id") in press:
            continue
        if sh.get("type") not in ("cylinder", "tube") or sh.get("rInner"):
            if fit is not None:
                raise ValueError(f"keepOut {k.get('id')!r}: a fit needs a plain cylinder (no rInner)")
            continue
        if fit is None:
            if kind is None:
                continue
            fit = "nominal"
        if fit not in FITS:
            raise ValueError(f"keepOut {k.get('id')!r}.fit must be one of {list(FITS)}, got {fit!r}")
        if finish not in ("auto", "as-built", "ream"):
            raise ValueError(f"keepOut {k.get('id')!r}.finish must be 'auto', 'as-built' or 'ream'")
        ends = _cyl_ends(sh)
        if ends is None:
            continue
        p0, p1 = ends
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        drawn = 2.0 * float(sh["r"])
        mate = float(mate) if mate is not None else drawn
        win = fit_window(fit, mate, family, size)
        row = {"id": k.get("id"), "fit": fit, "standard": win["standard"], "mate": round(mate, 4),
               "drawn": round(drawn, 4), "window": [None if win["lo"] is None else round(win["lo"], 4),
                                                    None if win["hi"] is None else round(win["hi"], 4)],
               "target": round(win["target"], 4), "process": kind,
               "axis": axis.round(4).tolist(), "p0": p0.round(4).tolist(), "p1": p1.round(4).tolist()}
        if size:
            row["size"] = size
        if kind is None:
            # a fit with no process: draw the target, say it is uncompensated
            designed, bias, tol, orient, how = win["target"], 0.0, None, None, "as-built"
            row["note"] = ("no process: the hole is drawn at the fit's size with no allowance for how it "
                           "prints; give process (fdm, sls, ...) to compensate")
        else:
            bias, tol, orient = _bias(holes, axis, build, win["target"])
            verdict = _verdict(win["lo"], win["hi"], win["target"], tol, holes["repeat"])
            how = "ream" if finish == "ream" or (finish == "auto" and verdict == "ream") else "as-built"
            if kind == "milled":
                how = "machine" if fit != "nominal" else "as-built"
            if how == "ream":
                if kind == "cast":
                    stock = _STOCK["cast"]
                    designed = (win["lo"] if win["lo"] is not None else win["target"]) - stock
                else:
                    stock = _STOCK[fam_stock]
                    designed = (win["lo"] if win["lo"] is not None else win["target"]) - bias - tol - stock
                row["finishTo"] = row["window"]
            else:
                designed = win["target"] - bias if kind != "milled" else win["target"]
            row.update({"bias": round(bias, 4), "tolerance": round(tol, 4), "orientation": orient,
                        "verdictAsBuilt": verdict})
        row["finish"] = how
        designed = max(designed, 0.2 * mate)
        row["designed"] = round(designed, 4)
        r_old, r_new = float(sh["r"]), 0.5 * designed
        sh["r"] = round(r_new, 5)
        # bosses and seat rings drawn round the old bore follow it
        for pr in d.get("preserve") or []:
            ps = (pr or {}).get("shape") or {}
            if ps.get("type") in ("cylinder", "tube") and ps.get("rInner") and \
                    abs(float(ps["rInner"]) - r_old) < 1e-6:
                pe = _cyl_ends(ps)
                if pe is not None and _coaxial(p0, p1, *pe) and r_new < float(ps["r"]):
                    ps["rInner"] = round(r_new, 5)
        rows.append(row)
    d["_fits"] = rows
    return d


# --- the delivered part ----------------------------------------------------

def _point_segment(c, a, b):
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", c - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
    return np.linalg.norm(a + t[:, None] * ab - c, axis=1)


def measure_bore(mesh, p0, p1, r_drawn: float, slices: int = 24) -> dict:
    """The hole the delivered mesh has along the axis p0-p1: the largest pin
    that passes (twice the smallest distance from the axis to the part's
    surface, slice by slice, so a coarse tessellation's chord sag counts),
    the wall round it, and whether the part wraps the hole at all."""
    import trimesh

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / L
    reach = r_drawn + max(3.0, r_drawn)
    # the triangles near the axis only
    v = mesh.vertices
    rel = v - p0
    t = rel @ a
    radial = np.linalg.norm(rel - np.outer(t, a), axis=1)
    tri = mesh.faces
    edge = np.max(np.linalg.norm(v[tri] - v[np.roll(tri, 1, axis=1)], axis=2), axis=1)
    near = (radial[tri].min(axis=1) < reach + edge) & (t[tri].max(axis=1) > -edge) & (t[tri].min(axis=1) < L + edge)
    if not near.any():
        return {"measured": None, "walled": False}
    sub = trimesh.Trimesh(v, tri[near], process=False)
    best, walled_slices, n = np.inf, 0, 0
    for s in np.linspace(0.0, L, slices + 2)[1:-1] + 1e-4:
        c = p0 + s * a
        segs = trimesh.intersections.mesh_plane(sub, a, c)
        if len(segs) == 0:
            continue
        n += 1
        dmin = float(_point_segment(np.repeat(c[None], len(segs), 0), segs[:, 0], segs[:, 1]).min())
        if dmin < reach:
            walled_slices += 1
            best = min(best, dmin)
    if not np.isfinite(best):
        return {"measured": None, "walled": False}
    return {"measured": round(2.0 * best, 4), "walled": True, "walledSlices": walled_slices}


def fits_report(study, mesh) -> list:
    """report["fits"]: each fitted hole as delivered: the hole the mesh has,
    the band it should print to, the clearance against its mating part, and
    whether the fit holds."""
    rows = getattr(study, "fits", None) or []
    out = []
    raw = getattr(study, "raw", {}) or {}
    _, _, holes = _process_of(raw)
    build = _chosen_build(study)
    for row in rows:
        r = dict(row)
        if r.get("orientation") == "unknown" and build is not None and holes is not None:
            # the build direction was chosen after the hole was drawn: the
            # bias for the hole's real orientation
            r["bias"], r["tolerance"], r["orientation"] = (
                round(x, 4) if isinstance(x, float) else x
                for x in _bias(holes, np.asarray(r["axis"], float), build, r["target"]))
        try:
            m = measure_bore(mesh, r["p0"], r["p1"], 0.5 * r["designed"])
        except Exception as exc:                          # a measurement must not sink a run
            m = {"measured": None, "error": str(exc)}
        r.update(m)
        tol = r.get("tolerance")
        if r.get("measured") is not None and tol is not None:
            built = r["measured"] + r["bias"]
            r["asBuilt"] = [round(built - tol, 3), round(built + tol, 3)]
            r["clearanceAsBuilt"] = [round(built - tol - r["mate"], 3), round(built + tol - r["mate"], 3)]
            lo, hi = r["window"]
            if r["finish"] in ("ream", "machine"):
                r["verdict"] = "ream" if r["finish"] == "ream" else "machine"
                if built + tol > (hi if hi is not None else np.inf):
                    r["verdict"] = "oversize"
            else:
                r["verdict"] = _verdict(lo, hi, built, tol, (holes or {}).get("repeat", 0.4))
        elif r.get("measured") is not None:
            r["asBuilt"] = None
            r["verdict"] = "uncompensated"
        else:
            r["verdict"] = "not-walled" if m.get("walled") is False else "unmeasured"
        r["instruction"] = _instruction(r)
        for key in ("p0", "p1", "axis"):
            r.pop(key, None)
        out.append(r)
    return out


def _chosen_build(study):
    proc = getattr(study, "process", None)
    b = getattr(proc, "build", None) if proc is not None else None
    if b is None or isinstance(b, str):
        return None
    if isinstance(b, tuple) and len(b) == 2:
        v = np.zeros(3)
        v[int(b[0])] = float(b[1])
        return v
    v = np.asarray(b, float).reshape(-1)
    return v / np.linalg.norm(v) if v.shape == (3,) and np.linalg.norm(v) > 0 else None


def _mm(x):
    return f"{x:.2f}"


def _instruction(r: dict) -> str:
    """One line a person (or an agent writing a build note) can act on."""
    v = r.get("verdict")
    name = f"{r['id']} ({r['fit']}, mate {_mm(r['mate'])} mm)" if r["fit"] != "nominal" else f"{r['id']}"
    lo, hi = r.get("window") or (None, None)
    if v == "holds":
        return f"{name}: prints to fit as drawn ({_mm(r['designed'])} mm in the file)"
    if v == "tune":
        return (f"{name}: holds once the printer's hole size is tuned: print the part (or a test ring) and "
                f"measure; if the hole is not {_mm(lo)}-{_mm(hi)} mm set process.holeBias to the measured "
                "minus the drawn diameter")
    if v in ("ream", "machine"):
        tool = "ream or drill" if v == "ream" else "bore or ream"
        return f"{name}: {tool} to {_mm(lo)}-{_mm(hi)} mm after {'printing' if v == 'ream' else 'roughing'} " \
               f"(drawn {_mm(r['designed'])} mm, {r.get('standard')})"
    if v == "oversize":
        return f"{name}: the delivered hole is already over {_mm(hi)} mm: report a bug"
    if v == "not-walled":
        return f"{name}: the part does not wrap this hole: nothing to fit against"
    if v == "uncompensated":
        return f"{name}: drawn {_mm(r['designed'])} mm with no process allowance (give process to compensate)"
    return f"{name}: not measured"


def fit_warnings(rows: list) -> list[str]:
    """The fits a person has to act on, for the report's warnings."""
    out = []
    for r in rows or []:
        if r.get("verdict") in ("ream", "machine", "tune", "oversize") or \
                (r.get("verdict") == "uncompensated" and r.get("fit") != "nominal"):
            out.append("fit " + r["instruction"])
    return out
