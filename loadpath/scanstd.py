"""
Standard sizes a scanned feature can snap to, and the reference objects a
scan can take its scale from.

A scan measures a bar at 31.79 +- 0.39 mm. Bars are not made at 31.79: a
handlebar's clamp area is 31.8, a seatpost 31.6, a steel tube 31.75 (1.25
inch). When the measured band covers exactly one standard size of the
family the feature can be, the part is designed to that size and its
tolerance, which is far tighter than any scan. When it covers several, the
question to ask is which one (a pick, not a measurement); when it covers
none, the feature is not a standard part and a caliper reading is the way
to tighten it.

Each family lists diameters (mm) and the +- tolerance the standard holds
them to, with the source. Where a standard gives no tolerance the entry
says what it is read as.

Reference objects (for a splat or a photogrammetry cloud's scale): an
ID-1 card and common coins, by their standards' dimensions.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["FAMILIES", "REFERENCE_OBJECTS", "snap", "snap_features", "reference_from_features"]

FAMILIES = {
    "handlebar": {
        "kind": "bar", "sizes": [22.2, 25.4, 26.0, 31.8, 35.0],
        "tol": 0.1,
        "source": "bicycle handlebar clamp and grip diameters (22.2 grip, 25.4 / 26.0 / 31.8 / 35.0 clamp; "
                  "ISO 4210-5 tests bars, the sizes are the industry's); +-0.1 mm read as the bar's tolerance"},
    "seatpost": {
        "kind": "bar", "sizes": [25.4, 27.2, 28.6, 30.0, 30.9, 31.6, 34.9],
        "tol": 0.1, "source": "bicycle seatpost diameters; +-0.1 mm read as the tolerance"},
    "steerer": {
        "kind": "bar", "sizes": [25.4, 28.6, 31.8, 38.1],
        "tol": 0.1, "source": "fork steerer diameters (1, 1-1/8, 1-1/4, 1-1/2 in)"},
    "pipe": {
        "kind": "bar", "sizes": [10.2, 13.5, 17.2, 21.3, 26.9, 33.7, 42.4, 48.3, 60.3, 76.1, 88.9],
        "tolRel": 0.01, "tolMin": 0.4,
        "source": "ISO 65 / EN 10255 steel pipe outside diameters (DN 6-80); OD tolerance about +-1% "
                  "(EN 10255 gives +-0.4-1.0 mm in this range), read as +-1%, at least 0.4 mm"},
    "tube-inch": {
        "kind": "bar", "sizes": [12.7, 15.875, 19.05, 22.225, 25.4, 28.575, 31.75, 38.1, 44.45, 50.8],
        "tol": 0.13, "source": "fractional-inch tube ODs (1/2-2 in); ASTM A513 holds about +-0.005 in"},
    "shaft": {
        "kind": "bar", "sizes": [3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 25, 30, 35, 40, 50],
        "iso": "h6", "source": "metric ground shafts and linear rods, ISO 286 h6"},
    "extrusion": {
        "kind": "bar", "sizes": [16.0, 20.0, 22.0, 25.0, 30.0, 32.0, 40.0, 50.0],
        "tol": 0.2, "source": "round aluminium extrusion / tube ODs, EN 755-9 (about +-0.2 mm at this size)"},
    "bolt-hole": {
        "kind": "bore",
        "sizes": {"M2": 2.4, "M2.5": 2.9, "M3": 3.4, "M4": 4.5, "M5": 5.5, "M6": 6.6, "M8": 9.0, "M10": 11.0},
        "tol": 0.1, "source": "ISO 273 medium clearance holes"},
    "tapped": {
        "kind": "bore",
        "sizes": {"M2": 1.6, "M2.5": 2.05, "M3": 2.5, "M4": 3.3, "M5": 4.2, "M6": 5.0, "M8": 6.8, "M10": 8.5},
        "tol": 0.1, "source": "ISO 2306 tap drill (the minor diameter a scan of a tapped hole reads)"},
    "insert": {
        "kind": "bore",
        "sizes": {"M2": 3.2, "M2.5": 3.6, "M3": 4.0, "M4": 5.6, "M5": 6.4, "M6": 8.0},
        "tol": 0.1, "source": "brass threaded inserts' outside diameters (common heat-set sizes)"},
    "bearing-bore": {
        "kind": "bore", "sizes": [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 16, 19, 22, 24, 26, 28, 30, 32, 35, 37, 42, 47, 52],
        "iso": "H7", "source": "ball bearing outside diameters (ISO 15 dimension series 8/9/0/2), a seat for each"},
}

# a reference object's known size, for a scan with no scale
REFERENCE_OBJECTS = {
    "card": {"shape": "rect", "size": [85.60, 53.98], "thickness": 0.76, "tol": 0.12,
             "source": "ISO/IEC 7810 ID-1 (bank, ID, transit cards): 85.60 x 53.98 x 0.76 mm, +-0.12 on width"},
    "us-quarter": {"shape": "disc", "d": 24.26, "thickness": 1.75, "tol": 0.05, "source": "US Mint specification"},
    "us-penny": {"shape": "disc", "d": 19.05, "thickness": 1.52, "tol": 0.05, "source": "US Mint specification"},
    "euro-1": {"shape": "disc", "d": 23.25, "thickness": 2.33, "tol": 0.05, "source": "ECB euro coin specification"},
    "euro-2": {"shape": "disc", "d": 25.75, "thickness": 2.20, "tol": 0.05, "source": "ECB euro coin specification"},
    "uk-pound": {"shape": "disc", "d": 23.43, "thickness": 2.80, "tol": 0.05, "source": "Royal Mint specification"},
}


def _family_sizes(fam):
    s = FAMILIES[fam]["sizes"]
    if isinstance(s, dict):
        return [(k, float(v)) for k, v in s.items()]
    return [(None, float(v)) for v in s]


def _tol(fam, d):
    f = FAMILIES[fam]
    if "iso" in f:
        from .fits import _it
        # half the ISO 286 grade's band (h6 sits below the size, H7 above:
        # the half-band about the middle is the +- that matters here)
        return 0.5 * float(_it(int(f["iso"][1:]), d)[0])
    if "tolRel" in f:
        return max(f["tolRel"] * d, f.get("tolMin", 0.0))
    return float(f["tol"])


def snap(diameter: float, band: float, kind: str, families=None) -> dict:
    """The standard sizes within a feature's measured band (diameter +-
    band, both mm) among the families its kind can be ('bar' or 'bore').
    Returns {"candidates": [...], "snapped": the one when it is unique by
    value, else None}. Sizes shared by several families (31.8 is a bar and a
    steerer) count once."""
    fams = [k for k, f in FAMILIES.items() if f["kind"] == kind and (families is None or k in families)]
    cands = []
    for fam in fams:
        for name, d in _family_sizes(fam):
            t = _tol(fam, d)
            if abs(d - diameter) <= band + t:
                cands.append({"family": fam, "diameter": d, "tolerance": round(t, 4), **({"name": name} if name else {}),
                              "offBy": round(d - diameter, 4), "source": FAMILIES[fam]["source"]})
    values = sorted({round(c["diameter"], 3) for c in cands})
    out = {"candidates": sorted(cands, key=lambda c: abs(c["offBy"])), "snapped": None}
    if len(values) == 1:
        best = min((c for c in cands), key=lambda c: c["tolerance"])
        out["snapped"] = best
    return out


def snap_features(feats: dict, families=None) -> dict:
    """Snap every cylinder whose band covers exactly one standard size (in
    place). A snapped feature keeps what was fitted under "fitted" and takes
    the standard's diameter and tolerance (+- read as 2 sigma); one whose
    band covers several gets "standardChoices" (ask which); none, nothing.
    `families` limits the families per feature id ({id: [names]}) or for
    all ([names]), e.g. from the brief ("a handlebar")."""
    from .scan import z_value
    z2 = z_value(feats.get("confidence", 0.95), two_sided=True)
    for f in feats.get("cylinders", []):
        if f.get("source", "").startswith("known"):
            continue
        fam = families.get(f["id"]) if isinstance(families, dict) else families
        r = snap(f["diameter"], f["diameterBand"], f["kind"], fam)
        if r["snapped"]:
            s = r["snapped"]
            f["fitted"] = {"diameter": f["diameter"], "diameterBand": f["diameterBand"]}
            f["diameter"] = s["diameter"]
            f["r"] = s["diameter"] / 2
            f["diameterBand"] = round(z2 * s["tolerance"] / 2, 4)
            f["rBand"] = round(f["diameterBand"] / 2, 4)
            f["source"] = f"standard: {s['family']} {s.get('name', s['diameter'])}"
            f["standard"] = s
        elif r["candidates"]:
            vals = sorted({c["diameter"] for c in r["candidates"]})
            f["standardChoices"] = vals[:8]
    return feats


def reference_from_features(feats: dict, objects=("card",), spacing: float | None = None) -> list:
    """Reference objects found among the fitted planes: a card is a flat
    rectangle with sides in 85.60 : 53.98 (1.586) filling its bounding
    rectangle, a coin a flat disc filling pi/4 of it (its denomination must
    be named: coins of a size look alike). Each match gives scale
    references for apply_reference, one per measured side ({"points",
    "distance", "tolerance", "edgeSigma", "object"}).

    A side is read between the outermost inlier points, which sit within
    about a spacing of the real edge (sparser points, and normals bent by
    the edge, keep the last ones back): no correction is added, and half a
    spacing at each end is its sigma (synthetic splats at 1 mm spacing read
    the card's 85.6 mm side 0.2 mm short, inside that)."""
    out = []
    for p in feats.get("planes", []):
        rect = p.get("rect")
        if not rect:
            continue
        L, W = rect["length"], rect["width"]
        h = rect.get("spacing") or spacing or 0.0
        c = np.asarray(rect["center"], float)
        u = np.asarray(rect["axis"], float)
        v = np.cross(np.asarray(p["normal"], float), u)
        v /= max(np.linalg.norm(v), 1e-12)
        for name in objects:
            o = REFERENCE_OBJECTS[name]
            if o["shape"] == "rect":
                ratio = o["size"][0] / o["size"][1]
                # the aspect's own uncertainty from the sides' (half a spacing at each end)
                s_ratio = ratio * math.hypot(0.71 * h / L, 0.71 * h / W)
                if not (abs(L / W - ratio) <= 3 * s_ratio + 0.01 and rect["fill"] > 0.9):
                    continue
                sides = [(L, u, o["size"][0]), (W, v, o["size"][1])]
            elif o["shape"] == "disc":
                if not (abs(L - W) / max(L, W) < 3 * 0.71 * h / L + 0.01 and abs(rect["fill"] - math.pi / 4) < 0.04):
                    continue
                sides = [(L, u, o["d"]), (W, v, o["d"])]
            else:
                continue
            for meas, ax, known in sides:
                a, b = c - 0.5 * meas * ax, c + 0.5 * meas * ax
                out.append({"object": name, "plane": p["id"], "points": [a.tolist(), b.tolist()],
                            "measured": round(meas, 6), "distance": known, "tolerance": o["tol"],
                            "edgeSigma": round(0.5 * h, 6), "aspect": round(L / W, 4)})
            break
    return out
