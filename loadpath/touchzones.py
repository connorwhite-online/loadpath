"""
Touch zones: how each place on a fixed exterior gives under a fingertip.

A designer marks places on the surfacing and says how each should feel:
the touch area "distinctly pliable", the rim and the sides "firm". Each
zone is a `touch` feel target (feel.py): fingertip presses sampled over the
zone, each a stiffness window between where the product is held and a
fingertip disc on the exterior. A preset names the window, or the study
gives it in N/mm, or as a force and a travel window ("a 5 N press goes in
0.5 to 2 mm"; a `travel` max alone is a firmness floor: "it takes at least
5 N to push it 0.3 mm").

Two things make zones more than separate windows:

- Contrast. "Distinctly" is a ratio, not a pair of windows: `contrast`
  asks that the stiffest press in the softer zone be at least `ratio`
  times softer than the softest press in the firmer one. One constraint
  per pair of zones, on log stiffness with smooth max/min (KS) over the
  presses of each side, so it holds at the worst pair.

- The skin. An interior can only stiffen a press: a zone cannot be softer
  than the skin over it with nothing behind. A pliable zone whose bare skin
  is already too stiff gets a thinner skin there (`skinThickness`, or
  "auto": the thickest skin whose bare presses stay under `autoMargin` of
  the ceiling), the exterior untouched (skin.zones, spec._zoned: only the
  inner surface moves). Never under the process's minimum wall: a zone
  that would need less is flagged with the thickness it would need, and
  its ceiling is worked to what the thinnest printable skin gives, the way
  switch.py relaxes a skin-limited press.

Firm zones get their stiffness from the interior (ribs and walls from the
skin to where the product is held), which the optimizer grows only where a
floor needs it.
"""
from __future__ import annotations

import copy

import numpy as np

__all__ = ["TOUCH_FEELS", "touch_window", "touch_points", "plan_touch_skins", "zone_summary"]

# N/mm under a 4 mm-radius fingertip (an 8 mm pad, about a fingertip's
# contact at a light press), held where the product rests.
#   pliable: a light 2 N touch sinks 0.5 to 2 mm, plainly felt, not mushy
#   firm:    a 5 N press moves it under 0.33 mm, felt as solid
#   rigid:   under 0.1 mm at 5 N, a hard shell
TOUCH_FEELS = {"pliable": (1.0, 4.0), "firm": (15.0, None), "rigid": (50.0, None)}

FINGERTIP = 4.0       # mm, radius of the pressed disc
SPACING = 6.0         # mm between sampled presses
AUTO_MARGIN = 0.75    # an auto skin's bare presses at most this share of the ceiling


def touch_window(t: dict, label: str):
    """(kmin, kmax) N/mm of a touch target: `stiffness`, `force` and
    `travel`, or a `feel` preset (explicit numbers override the preset's
    side they give)."""
    from .feel import _rng
    lo = hi = None
    if t.get("feel") is not None:
        if t["feel"] not in TOUCH_FEELS:
            raise ValueError(f"{label}.feel must be one of {sorted(TOUCH_FEELS)}, got {t['feel']!r}")
        lo, hi = TOUCH_FEELS[t["feel"]]
    if "stiffness" in t:
        a, b = _rng(t["stiffness"])
        lo, hi = (a if a is not None else lo), (b if b is not None else hi)
    elif "force" in t or "travel" in t:
        if "force" not in t or "travel" not in t:
            raise ValueError(f"{label}: a touch window by force needs both force (N) and travel (mm)")
        force = float(t["force"])
        a, b = _rng(t["travel"])
        if b is not None:
            lo = force / b
        if a is not None:
            hi = force / a
    if lo is None and hi is None:
        raise ValueError(f"{label}: a touch zone needs a feel (one of {sorted(TOUCH_FEELS)}), a stiffness "
                         "[min, max] N/mm, or a force and a travel window")
    return lo, hi


def _press_axis(t: dict):
    from .switch import _axis
    return _axis(t.get("direction", "-z"))


def touch_points(study, t: dict, label: str) -> np.ndarray:
    """The fingertip presses of a touch target, as points on the exterior
    (n x 3). `points` as given (2 in-plane coordinates or 3), or a grid of
    `spacing` mm over the zone, centred on the zone's bounds, each kept
    where a ray along the press direction meets the design space inside
    the zone at least `inset` mm from its edge (default half the
    fingertip, at most a quarter of the spacing). The same points on every grid (they come from the shapes,
    not the mesh), so a coarse-to-fine run and the read-back on the
    delivered mesh read the same presses."""
    from .shapes import Union, shape_from_spec
    from .switch import _first_hit
    ax, sg = _press_axis(t)
    others = [i for i in range(3) if i != ax]
    base = getattr(study, "base_dir", None)
    ds = study.design_space[0] if len(study.design_space) == 1 else Union(study.design_space)
    lo, hi = study.bounds()
    reach = float(hi[ax] - lo[ax]) + 2.0
    start = (hi[ax] + 1.0) if sg < 0 else (lo[ax] - 1.0)
    direction = np.zeros(3)
    direction[ax] = sg
    zone = shape_from_spec(t["zone"], base) if "zone" in t else None

    def onto(uv):
        pts = np.zeros((len(uv), 3))
        pts[:, others[0]], pts[:, others[1]] = uv[:, 0], uv[:, 1]
        pts[:, ax] = start
        d = _first_hit(ds, pts, direction, reach, 0.25)
        pts[:, ax] = start + sg * d
        return pts, np.isfinite(d)

    if t.get("points"):
        raw = [np.asarray(p, float) for p in t["points"]]
        uv = np.array([p[others] if len(p) == 3 else p[:2] for p in raw])
        pts, ok = onto(uv)
        if not ok.all():
            bad = [list(map(float, uv[i])) for i in np.flatnonzero(~ok)]
            raise ValueError(f"{label}.points: a press along {t.get('direction', '-z')} misses the part at {bad}")
        return pts
    if zone is None:
        raise ValueError(f"{label}: a touch target needs a zone (a shape) or points")
    zlo, zhi = zone.bounds()
    zlo = np.maximum(zlo, lo)
    zhi = np.minimum(zhi, hi)
    sp = float(t.get("spacing", SPACING))
    c = 0.5 * (zlo + zhi)
    axes = []
    for o in others:
        n = int(np.floor((zhi[o] - c[o]) / sp + 1e-9))
        axes.append(c[o] + sp * np.arange(-n, n + 1))
    U, V = np.meshgrid(axes[0], axes[1], indexing="ij")
    uv = np.stack([U.ravel(), V.ravel()], axis=1)
    pts, ok = onto(uv)
    # presses whose centre is inside by a little: a press on the zone's edge
    # is half on its neighbour (and on the skin's thickness ramp)
    inset = float(t.get("inset", min(0.5 * float(t.get("fingertip", FINGERTIP)), 0.25 * sp)))
    ok &= _inside(zone, np.where(ok[:, None], pts, 0.0), others, inset)
    if not ok.any():
        # a zone narrower than the spacing: its centre
        pts, ok = onto(c[others][None])
        ok &= _inside(zone, np.where(ok[:, None], pts, 0.0), others, 0.0)
        if not ok.any():
            raise ValueError(f"{label}: no press along {t.get('direction', '-z')} meets the part inside the zone")
    return pts[ok]


def _inside(zone, pts, others, inset):
    """Points inside the zone with a ring of `inset` mm around each, across
    the press direction, inside too (the zone's depth along the press does
    not count)."""
    ok = zone.sdf(pts) <= 1e-6
    if inset > 0:
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            q = pts.copy()
            q[:, others[0]] += inset * np.cos(a)
            q[:, others[1]] += inset * np.sin(a)
            ok &= zone.sdf(q) <= 1e-6
    return ok


# ---------------------------------------------------------------------------
# the skin under each zone


def _min_wall(study) -> float:
    proc = getattr(study, "process", None)
    return float(getattr(proc, "min_feature", None) or 0.8)


def _with_zones(study, zones: list[dict]):
    """The study re-parsed with these skin zones added."""
    from .spec import study_from_dict
    raw = copy.deepcopy(study.raw)
    skin = raw.get("skin")
    if isinstance(skin, (int, float)):
        skin = {"thickness": float(skin)}
    skin = dict(skin)
    skin["zones"] = list(skin.get("zones") or []) + zones
    raw["skin"] = skin
    st = study_from_dict(raw, getattr(study, "base_dir", None))
    for k in ("_touch_relax",):
        if hasattr(study, k):
            setattr(st, k, getattr(study, k))
    return st


def _bare_stiffness(study, name: str, pitch: float) -> np.ndarray:
    """N/mm of each press of touch target `name` on the bare skin (nothing
    inside), on cut skin cells at `pitch`."""
    from .domain import build_domain
    from .feel import FeelProblem
    from .switch import _thin_skin
    dom, floor = _thin_skin(study, build_domain(study, pitch))
    prob = FeelProblem(study, dom, study.elastic(design=True), float(study.material.E), only_zone=name)
    bare = np.maximum(np.maximum(dom.preserve.astype(float), floor), 1e-9)
    comp = prob.compliances(bare, grads=False)
    return np.array([1.0 / max(comp[(ti, 0)][0], 1e-30) for ti in range(len(prob.targets))])


def plan_touch_skins(study, pitch: float, on_progress=None):
    """Skin thickness under each touch zone that asks for one
    (`skinThickness`: mm or "auto"). Returns (study with the skin zones,
    report rows). An auto zone keeps the study's skin if its bare presses
    already sit under AUTO_MARGIN of the ceiling, else the thickest skin
    (0.05 mm steps) that gets them there, from a power law fitted between
    the study's skin and the process minimum and checked on the grid.
    Never under the minimum wall: a zone that needs less is flagged, and
    its ceiling is worked to the thinnest printable skin's bare stiffness
    (x 1.02), the true ceiling still what the part is read against."""
    fb = study.raw.get("feel") or {}
    skin = study.raw.get("skin")
    if skin is None:
        return study, []
    t0 = float(skin if isinstance(skin, (int, float)) else skin["thickness"])
    tmin = _min_wall(study)
    rows, zones = [], []
    relax = dict(getattr(study, "_touch_relax", {}))
    cur = study
    for i, t in enumerate(fb.get("targets") or []):
        if t.get("kind") != "touch" or t.get("skinThickness") is None:
            continue
        name = t.get("name") or f"touch-{i}"
        label = f"feel.targets[{i}]"
        if "zone" not in t:
            raise ValueError(f"{label}: skinThickness needs a zone (the shape the skin changes under)")
        blend = float(t.get("skinBlend", 2.0))
        kmin, kmax = touch_window(t, label)
        want = t["skinThickness"]
        row = {"zone": name, "skin": t0, "minWall": tmin}

        def zone_at(th):
            return {"shape": t["zone"], "thickness": round(float(th), 3), "blend": blend}

        if want != "auto":
            th = float(want)
            row.update(thickness=th, chosen="given")
            if th < tmin - 1e-9:
                row["belowMinWall"] = True
                row["note"] = (f"{th:g} mm is under the process's {tmin:g} mm minimum wall: it may not print "
                               "whole; ask the vendor before ordering")
            if abs(th - t0) > 1e-9:
                zones.append(zone_at(th))
                cur = _with_zones(study, zones)
            rows.append(row)
            continue
        if kmax is None:
            # a firm zone: the interior stiffens it, the skin stays
            row.update(thickness=t0, chosen="auto", why="no ceiling: the interior makes it firm")
            rows.append(row)
            continue
        goal = AUTO_MARGIN * kmax
        if on_progress is not None:
            on_progress({"stage": "skin", "zone": name, "check": f"bare skin at {t0:g} mm"})
        k0 = float(_bare_stiffness(cur, name, pitch).max())
        row["bareAtSkin"] = k0
        if k0 <= goal:
            row.update(thickness=t0, chosen="auto", why=f"bare skin {k0:.2f} N/mm already under "
                                                       f"{AUTO_MARGIN:g} x the {kmax:g} N/mm ceiling")
            rows.append(row)
            continue
        if t0 <= tmin + 1e-9:
            th, k_at = t0, k0
            n_exp = 3.0
        else:
            st_min = _with_zones(study, zones + [zone_at(tmin)])
            if on_progress is not None:
                on_progress({"stage": "skin", "zone": name, "check": f"bare skin at {tmin:g} mm"})
            k1 = float(_bare_stiffness(st_min, name, pitch).max())
            row["bareAtMinWall"] = k1
            n_exp = float(np.log(k0 / k1) / np.log(t0 / tmin)) if k1 < k0 else 3.0
            n_exp = float(np.clip(n_exp, 1.0, 4.0))
            if k1 > goal:
                th, k_at = tmin, k1
            else:
                th = t0 * (goal / k0) ** (1.0 / n_exp)
                th = max(tmin, np.floor(th / 0.05 + 1e-9) * 0.05)
                k_at = k1 if th <= tmin + 1e-9 else None
                for _ in range(4 if k_at is None else 0):
                    if on_progress is not None:
                        on_progress({"stage": "skin", "zone": name, "check": f"bare skin at {th:g} mm"})
                    k_at = float(_bare_stiffness(_with_zones(study, zones + [zone_at(th)]), name, pitch).max())
                    if k_at <= goal or th <= tmin + 1e-9:
                        break
                    th = max(tmin, round(th - 0.05, 3))
        row.update(thickness=float(th), chosen="auto", bare=k_at, fitExponent=round(n_exp, 2))
        if k_at > kmax:
            # the thinnest printable skin is still too stiff: say what it
            # would take, and work the ceiling to what it gives
            need = th * (kmax / k_at) ** (1.0 / n_exp)
            row["belowMinWall"] = True
            row["needs"] = round(float(need), 2)
            row["note"] = (f"the bare skin at the {tmin:g} mm minimum wall is {k_at:.2f} N/mm, over the "
                           f"{kmax:g} N/mm ceiling: this zone would need about {need:.2f} mm, under what the "
                           "process prints. Options: a softer material, a larger zone (a longer span bends "
                           "more), or a higher ceiling")
            relax[name] = 1.02 * k_at
        if abs(th - t0) > 1e-9:
            zones.append(zone_at(th))
        cur = _with_zones(study, zones) if zones else study
        rows.append(row)
    if zones:
        cur = _with_zones(study, zones)
    cur._touch_relax = relax
    if relax:
        for r in rows:
            if r["zone"] in relax:
                r["workingCeiling"] = relax[r["zone"]]
    return cur, rows


# ---------------------------------------------------------------------------
# reading zones


def zone_summary(rows: list[dict], study, key: str = "directions") -> dict:
    """Per touch zone: the softest and stiffest press and how many presses
    meet the window; per contrast: the ratio it reaches (softest press of
    the firmer zone over the stiffest of the softer one). `key` picks the
    reading ("directions": design, "thresholded", "delivered")."""
    zones: dict[str, dict] = {}
    for r in rows:
        if r.get("kind") != "touch" or not r.get(key):
            continue
        z = r["params"]["zone"]
        k = float(r[key][0]["stiffness"])
        ok = bool(r[key][0]["ok"])
        e = zones.setdefault(z, {"min": k, "max": k, "presses": 0, "ok": 0, "minStiffness": r["minStiffness"],
                                 "maxStiffness": r["maxStiffness"], "feel": r["params"].get("feel")})
        e["min"], e["max"] = min(e["min"], k), max(e["max"], k)
        e["presses"] += 1
        e["ok"] += ok
    contrasts = []
    for c in ((study.raw.get("feel") or {}).get("contrast") or []):
        s, f = zones.get(c["softer"]), zones.get(c["firmer"])
        if s is None or f is None:
            continue
        ratio = f["min"] / max(s["max"], 1e-30)
        contrasts.append({"softer": c["softer"], "firmer": c["firmer"], "ratio": ratio,
                          "wanted": float(c.get("ratio", 2.0)), "ok": bool(ratio >= 0.98 * float(c.get("ratio", 2.0)))})
    return {"zones": zones, "contrast": contrasts,
            "ok": all(z["ok"] == z["presses"] for z in zones.values()) and all(c["ok"] for c in contrasts)}


def contrast_pairs(study, targets) -> list[tuple[list[int], list[int], float, str]]:
    """(soft target indices, firm target indices, ratio, name) per
    feel.contrast entry, on the expanded target list."""
    out = []
    names = {}
    for ti, t in enumerate(targets):
        if t.kind == "touch":
            names.setdefault(t.params["zone"], []).append(ti)
    for j, c in enumerate((study.raw.get("feel") or {}).get("contrast") or []):
        for side in ("softer", "firmer"):
            if c.get(side) not in names:
                raise ValueError(f"feel.contrast[{j}].{side}: {c.get(side)!r} is not a touch zone "
                                 f"(touch zones: {sorted(names)})")
        r = float(c.get("ratio", 2.0))
        if r <= 1:
            raise ValueError(f"feel.contrast[{j}].ratio must be over 1")
        out.append((names[c["softer"]], names[c["firmer"]], r, f"contrast:{c['softer']}<{c['firmer']}"))
    return out
