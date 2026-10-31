"""
Can a powder-bed polymer (SLS, MJF) or resin (SLA, DLP) printer make it?

FDM and LPBF parts fail on overhangs and bridges (printcheck.py, cavities.py).
These processes fail elsewhere:

  powder bed (SLS, MJF: PA12, PA11, TPU) - the part sits in its own powder,
      so nothing needs support, but
        walls      below ~0.8 mm break in depowdering;
        powder     inside a cavity has to come out: a fully enclosed void
                   keeps it (and the weight), and a chamber that only reaches
                   the outside through a slot narrower than an escape hole
                   (~4-5 mm) keeps most of it;
        warp       long thin flat panels cool unevenly and curl: under ~2 mm
                   thick and over ~100 mm long, a plate bows.
  resin vat (SLA, DLP, bottom-up) - layers are peeled off a film, so
        walls      below ~0.6 mm (unsupported) tear or droop;
        supports   every overhang past 45 degrees and every island (a point
                   that starts in mid-air) needs a support, which leaves
                   marks; the count and volume come from printcheck;
        peel       a large cross-section pulls hard on every peel: tilt it;
        cups       a void open only toward the vat, sealed on the plate side
                   (a cup), is a suction cup at every peel until it closes: it
                   tears layers or the part off the plate. Vent it near the
                   plate side, or turn it over;
        resin      trapped in a closed cavity, or behind a drain narrower than
                   ~3.5 mm, stays liquid inside and cracks the part later.

Everything is read off one voxel grid of the part (a trimesh, or an implicit
field with bounds) at a pitch fine enough for the process's minimum wall,
within a cell budget. A probe that could not run says so (None), it never
guesses. The numbers are heuristics with published-range thresholds (RULES),
meant to flag what a print bureau would reject or a print would lose, not to
replace the bureau's own check.

    rep = check_process(mesh, "mjf", build="+z", material="pa12")
    rep["ok"], rep["warnings"]
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["RULES", "POWDER", "RESIN", "check_process", "rules_for"]

POWDER = ("sls", "mjf")
RESIN = ("sla", "dlp")

# Typical published design-guide values (EOS/Formlabs Fuse/HP MJF and
# Formlabs/Asiga resin guides), on the conservative side. mm, mm^2.
RULES = {
    "sls": {"minWall": 0.8, "escapeHole": 4.0, "escapeHoles": 2, "minClearance": 0.5,
            "warpThickness": 2.0, "warpSpan": 100.0,
            "note": "SLS nylon: no supports; leave >= 2 escape holes of >= 4 mm in hollow parts; long flat "
                    "panels under 2 mm warp"},
    "mjf": {"minWall": 0.8, "escapeHole": 5.0, "escapeHoles": 2, "minClearance": 0.4,
            "warpThickness": 2.0, "warpSpan": 100.0,
            "note": "HP Multi Jet Fusion: as SLS; HP asks for >= 2 escape holes of >= 5 mm"},
    "sla": {"minWall": 0.6, "minSupportedWall": 0.4, "escapeHole": 3.5, "escapeHoles": 2, "minClearance": 0.5,
            "peelArea": 2500.0, "cupArea": 10.0,
            "note": "resin vat (bottom-up SLA): supports below 45 degrees and at islands; drains >= 3.5 mm "
                    "plus a vent near the plate in hollow parts; tilt large cross-sections"},
    "dlp": {"minWall": 0.6, "minSupportedWall": 0.4, "escapeHole": 3.5, "escapeHoles": 2, "minClearance": 0.3,
            "peelArea": 2000.0, "cupArea": 10.0,
            "note": "resin vat (DLP/LCD, bottom-up): as SLA; the whole layer cures at once, so peel force "
                    "grows with cross-section faster"},
}
# Material-specific changes to a process's rules: TPU powder tears thin walls
# out in depowdering and does not warp like nylon.
_MATERIAL_RULES = {
    "tpu88a": {"minWall": 1.0, "warpSpan": None},
}
# Cell budget for the grid (one byte per cell, plus float distance fields).
_BUDGET = 12_000_000
# Cells per minimum wall the pitch aims for; below 2 the wall verdict is None.
_CELLS_PER_WALL = 2.5
_EIGHT = np.ones((3, 3), bool)
_CONN = np.ones((3, 3, 3), bool)


def rules_for(process: str, material=None, overrides: dict | None = None) -> dict:
    """The rule card for a process (RULES), adjusted for the material and
    any overrides (same keys)."""
    kind = str(getattr(process, "kind", process)).lower()
    if kind not in RULES:
        raise ValueError(f"check_process covers {sorted(RULES)}; got {kind!r} (FDM and LPBF: printcheck, cavities)")
    out = dict(RULES[kind])
    mat = str(getattr(material, "name", material) or "").lower()
    out.update(_MATERIAL_RULES.get(mat, {}))
    if getattr(process, "escape_hole", None):
        out["escapeHole"] = float(process.escape_hole)
    out.update({k: v for k, v in (overrides or {}).items() if v is not None})
    min_feature = getattr(process, "min_feature", None)
    if min_feature is not None:
        out["minWall"] = max(out["minWall"], float(min_feature))
    out["process"] = kind
    return out


def _grid(part, rot, bounds, pitch, pad=2):
    """Solid cells of the part with the build direction turned to +z (k = 0
    at the plate). Returns (solid, origin in the build frame)."""
    from .voxels import solid_grid

    if callable(part) and not hasattr(part, "faces"):
        lo, hi = (np.asarray(b, float) for b in bounds)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        rc = corners @ rot.T
        # world = rot^T @ build, so as row vectors P_world = P_build @ rot
        field = part if np.allclose(rot, np.eye(3)) else (lambda p: part(np.asarray(p) @ rot))
        solid, origin, _ = solid_grid(field, pitch, pad=pad, bounds=(rc.min(axis=0), rc.max(axis=0)))
        return solid, origin
    m = part.copy()
    m.apply_transform(np.block([[rot, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    solid, origin, _ = solid_grid(m, pitch, pad=pad)
    return solid, origin


def _choose_pitch(extents, min_wall, pitch=None, budget=_BUDGET):
    if pitch:
        return float(pitch)
    want = min_wall / _CELLS_PER_WALL
    floor = float((np.prod(np.asarray(extents, float) + 4 * want) / budget) ** (1.0 / 3.0))
    return max(want, floor)


def _walls(solid, pitch, origin, min_wall):
    from .voxels import wall_thickness

    if pitch > min_wall / 2.0 + 1e-9:
        return {"minWallMm": None, "ok": None,
                "inconclusive": f"pitch {pitch:.3g} mm (cell budget) is coarser than half the {min_wall:g} mm wall"}
    w = wall_thickness(solid, pitch, cap_mm=4.0 * min_wall, limit_mm=min_wall, spots=5, origin=origin)
    p1 = w.get("p1Mm")
    out = {"minWallMm": w.get("minMm"), "p1WallMm": p1, "thinAreaMm2": w.get("thinAreaMm2", 0.0),
           "thinSpots": w.get("thinSpots", []), "toleranceMm": pitch,
           # judged on the 1st percentile with the half-cell estimator error,
           # as check_dfm does: one sliver does not condemn a part
           "ok": p1 is None or p1 + 0.5 * pitch >= min_wall,
           "basis": "ok is judged on p1WallMm (1st percentile, +/- half a cell); minWallMm is the single "
                    "thinnest spot, which can sit under the minimum on a passing part (a sliver at an edge)"}
    return out


def _escapes(solid, pitch, origin, hole, rot):
    """Enclosed voids, and chambers that only reach the outside through
    openings narrower than `hole` mm (a ball that size cannot get out)."""
    void = ~solid
    lbl, n = ndimage.label(void, _CONN)
    border = set()
    for face in (lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1], lbl[:, :, 0], lbl[:, :, -1]):
        border.update(np.unique(face).tolist())
    border.discard(0)
    sizes = np.bincount(lbl.ravel(), minlength=n + 1)
    cell3 = pitch ** 3
    world = lambda ijk: [round(float(c), 2) for c in rot.T @ (origin + (np.asarray(ijk, float) + 0.5) * pitch)]  # noqa: E731
    enclosed = []
    tiny = max(8, int(np.ceil(1.0 / cell3)))   # under ~1 mm^3 is aliasing, not a pocket
    keep = [k for k in range(1, n + 1) if k not in border and sizes[k] >= tiny]
    if keep:
        for k, c in zip(keep, ndimage.center_of_mass(void, lbl, keep)):
            enclosed.append({"volumeMm3": round(float(sizes[k] * cell3), 2), "center": world(c)})
    enclosed.sort(key=lambda v: -v["volumeMm3"])

    # balls of the escape hole's size: their centres (wide), the ones that
    # can roll out to the boundary, and the void they sweep
    r = 0.5 * hole / pitch
    dist = ndimage.distance_transform_edt(void)
    wide = dist >= r
    wl, wn = ndimage.label(wide, _CONN)
    wborder = set()
    for face in (wl[0], wl[-1], wl[:, 0], wl[:, -1], wl[:, :, 0], wl[:, :, -1]):
        wborder.update(np.unique(face).tolist())
    wborder.discard(0)
    chambers = []
    if wn:
        inner = [k for k in range(1, wn + 1) if k not in wborder]
        if inner:
            # only chambers in void that does reach the outside (the rest are
            # enclosed, above): a wide component lies in one void component
            host = ndimage.maximum(lbl, wl, inner)
            inner = [k for k, h in zip(inner, np.atleast_1d(host)) if int(h) in border]
        if inner:
            # what the chambers' balls sweep: their centres grown back by r
            sel = np.isin(wl, inner)
            swept = (ndimage.distance_transform_edt(~sel) <= r) & void
            gl, gn = ndimage.label(swept, _CONN)
            vols = np.bincount(gl.ravel(), minlength=gn + 1)
            ball = 4.0 / 3.0 * np.pi * (0.5 * hole) ** 3
            big = [k for k in range(1, gn + 1) if vols[k] * cell3 >= 2.0 * ball]
            if big:
                for k, c in zip(big, ndimage.center_of_mass(swept, gl, big)):
                    chambers.append({"volumeMm3": round(float(vols[k] * cell3), 1), "center": world(c)})
    chambers.sort(key=lambda v: -v["volumeMm3"])
    return {
        "escapeHoleMm": hole,
        "enclosedVoids": enclosed[:10], "enclosedCount": len(enclosed),
        "enclosedVolumeMm3": round(float(sum(v["volumeMm3"] for v in enclosed)), 2),
        "narrowExitChambers": chambers[:10], "narrowExitCount": len(chambers),
        "narrowExitVolumeMm3": round(float(sum(v["volumeMm3"] for v in chambers)), 1),
    }


def _panels(solid, pitch, origin, rot, t_warp, span_warn):
    """Thin flat panels: the medial ridge of members thinner than t_warp,
    grouped, kept when flat and filled in (a sheet, not a strut or the
    medial lines running into a thick part's corners)."""
    edt = ndimage.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1, 1:-1]
    ridge = solid & (edt >= ndimage.maximum_filter(edt, size=3)) & (edt > 0)
    thick = (2.0 * edt - 0.5) * pitch
    thin = ridge & (thick < t_warp)
    if not thin.any():
        return {"warpThicknessMm": t_warp, "panels": [], "maxSpan": 0.0}
    lbl, n = ndimage.label(ndimage.binary_dilation(thin, _CONN) & solid, _CONN)
    panels = []
    for k, sl in enumerate(ndimage.find_objects(lbl), start=1):
        if sl is None:
            continue
        sub = (lbl[sl] == k) & thin[sl]
        pts = np.argwhere(sub).astype(float)
        if len(pts) < 20:
            continue
        pts = (pts + [s.start for s in sl] + 0.5) * pitch
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        proj = (pts - c) @ vt.T
        ext = np.ptp(proj, axis=0) + pitch
        span, width, depth = float(ext[0]), float(ext[1]), float(ext[2])
        # a ridge sheet is a cell or two thick: area ~ ridge cells x pitch^2
        area = float(len(pts) * pitch * pitch)
        if width < 4 * t_warp or depth > max(2.0 * t_warp, 0.15 * width) or area < 0.3 * span * width:
            continue
        panels.append({"span": round(span, 1), "width": round(width, 1), "area": round(min(area, span * width), 0),
                       "thickness": round(float(np.median(thick[sl][sub])), 2),
                       "center": [round(float(v), 2) for v in rot.T @ (origin + c)],
                       "normal": [round(float(v), 3) for v in rot.T @ vt[2]]})
    panels.sort(key=lambda p: -p["span"])
    out = {"warpThicknessMm": t_warp, "panels": panels[:5], "maxSpan": panels[0]["span"] if panels else 0.0}
    if span_warn is not None:
        out["ok"] = out["maxSpan"] < span_warn
    return out


def _layers(solid, pitch):
    """Per layer from the plate: cross-section area and suction cups (void
    in the layer enclosed by it and sealed below by part, plate or an
    earlier cup)."""
    a = pitch * pitch
    nz = solid.shape[2]
    area = solid.sum(axis=(0, 1)) * a
    sealed_prev = None
    cup_area = np.zeros(nz)
    worst = None
    for k in range(nz):
        layer = solid[:, :, k]
        if not layer.any():
            sealed_prev = np.zeros_like(layer)
            continue
        lbl, n = ndimage.label(~layer, _EIGHT)
        if n == 0:
            sealed_prev = np.zeros_like(layer)
            continue
        edge = set(np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])).tolist())
        if k == 0 or sealed_prev is None:
            leak = np.zeros_like(layer)     # the plate seals the first layer
        else:
            leak = ~solid[:, :, k - 1] & ~sealed_prev
        leaky = set(np.unique(lbl[leak & (lbl > 0)]).tolist()) | edge
        sealed = (lbl > 0) & ~np.isin(lbl, list(leaky))
        cup_area[k] = sealed.sum() * a
        if sealed.any() and (worst is None or cup_area[k] > worst[1]):
            worst = (k, cup_area[k], np.argwhere(sealed).mean(axis=0))
        sealed_prev = sealed
    return area, cup_area, worst


def check_process(part, process="sls", build=None, material=None, pitch: float | None = None,
                  bounds=None, rules: dict | None = None, slicer: bool = True) -> dict:
    """Process checks for powder-bed polymer (sls, mjf) and resin (sla, dlp)
    parts. `part` is a trimesh (mm) or a field f(P) < 0 inside with
    `bounds`. `process` is a name or a spec.Process; `build` the direction
    layers stack ((axis, sign), a vector or '+z'; default the process's,
    else +z), which only resin and the layer report depend on. `rules`
    overrides any RULES entry. slicer=True (meshes, resin) adds printcheck's
    support and island read."""
    from .am import parse_build_direction
    from .printcheck import rotation_to_z

    kind = str(getattr(process, "kind", process)).lower()
    card = rules_for(process, material, rules)
    if build is None:
        build = getattr(process, "build", None)
    if build is None or build == "auto":
        build = (2, 1)
    if isinstance(build, str):
        build = parse_build_direction(build)
    rot = rotation_to_z(build)
    is_field = callable(part) and not hasattr(part, "faces")
    if is_field:
        if bounds is None:
            raise ValueError("check_process on a field needs bounds = (lo, hi) in mm")
        extents = np.asarray(bounds[1], float) - np.asarray(bounds[0], float)
    else:
        extents = np.asarray(part.extents, float)
    ext_b = np.abs(rot) @ extents
    p = _choose_pitch(ext_b, card["minWall"], pitch)
    # pad past an escape ball's radius, so balls outside the part reach the
    # grid's boundary however the part sits in its box
    solid, origin = _grid(part, rot, bounds, p, pad=int(np.ceil(0.5 * card["escapeHole"] / p)) + 3)
    vol = float(solid.sum() * p ** 3)
    rep = {"process": kind, "rules": card, "pitch": round(p, 4), "volumeMm3": round(vol, 1),
           "buildDirection": [round(float(c), 4) for c in rot.T @ np.array([0.0, 0.0, 1.0])],
           "material": getattr(material, "name", material)}
    warnings = []
    mat = getattr(material, "name", material)
    procs = getattr(material, "processes", None)
    if procs and kind not in procs:
        warnings.append(f"{mat} is not a {kind} material (made by {', '.join(procs)}): pick one of the "
                        f"{kind} cards in materials.MATERIALS")

    rep["walls"] = _walls(solid, p, origin, card["minWall"])
    if rep["walls"]["ok"] is False:
        warnings.append(f"walls down to {rep['walls']['p1WallMm']:.2f} mm (1st percentile) are under the "
                        f"{card['minWall']:g} mm {kind} minimum ({rep['walls']['thinAreaMm2']:.0f} mm^2 of surface)")

    esc = _escapes(solid, p, origin, card["escapeHole"], rot)
    rep["escape"] = esc
    what = "powder" if kind in POWDER else "resin"
    if esc["enclosedCount"]:
        warnings.append(f"{esc['enclosedCount']} enclosed void(s) ({esc['enclosedVolumeMm3']:.0f} mm^3) keep their "
                        f"{what}: add >= {card['escapeHoles']} holes of >= {card['escapeHole']:g} mm into each")
    if esc["narrowExitCount"]:
        warnings.append(f"{esc['narrowExitCount']} chamber(s) ({esc['narrowExitVolumeMm3']:.0f} mm^3) open only "
                        f"through gaps under {card['escapeHole']:g} mm: {what} will not clear; widen an exit")
    rep["trappedVolumeMm3"] = round(esc["enclosedVolumeMm3"] + esc["narrowExitVolumeMm3"], 1)

    if kind in POWDER:
        rep["warp"] = _panels(solid, p, origin, rot, card["warpThickness"], card.get("warpSpan"))
        if rep["warp"].get("ok") is False:
            w = rep["warp"]["panels"][0]
            warnings.append(f"a {w['span']:.0f} x {w['width']:.0f} mm panel only {w['thickness']:.1f} mm thick will "
                            f"warp in {kind}: thicken it past {card['warpThickness']:g} mm, rib it, or break it up")
    else:
        area, cups, worst = _layers(solid, p)
        rep["peel"] = {"maxLayerArea": round(float(area.max()), 1),
                       "atHeight": round(float((int(area.argmax()) + 0.5) * p), 2), "limit": card["peelArea"]}
        if area.max() > card["peelArea"]:
            warnings.append(f"a {area.max():.0f} mm^2 layer pulls hard on every peel: tilt the part 15-30 degrees "
                            "off the plate")
        rep["cups"] = {"maxArea": round(float(cups.max()), 1), "layers": int((cups > card["cupArea"]).sum()),
                       "volumeMm3": round(float(cups.sum() * p), 1)}
        if worst is not None:
            ij = np.r_[worst[2], worst[0]]
            rep["cups"]["at"] = [round(float(c), 2) for c in rot.T @ (origin + (ij + 0.5) * p)]
        if cups.max() > card["cupArea"]:
            warnings.append(f"a {cups.max():.0f} mm^2 suction cup (a void sealed on the plate side, open to the "
                            "vat) pulls at every peel: vent it near the plate, or turn the part over")
        if slicer and not is_field:
            from .printcheck import check_print

            pc = check_print(part, build, process=kind, material=material)
            rep["supports"] = {k: pc[k] for k in ("supportArea", "supportVolume", "islands", "islandArea",
                                                  "maxBridgeSpan")}
            if pc["islands"]:
                warnings.append(f"{pc['islands']} island(s) start in mid-air: the slicer must put a support "
                                "under each (check they are not on a mating face)")
    rep["warnings"] = warnings
    probes = [rep["walls"]["ok"]]
    if kind in POWDER and "ok" in rep["warp"]:
        probes.append(rep["warp"]["ok"])
    rep["probesRan"] = all(v is not None for v in probes)
    rep["ok"] = bool(all(v is True for v in probes) and rep["trappedVolumeMm3"] < 1.0
                     and (kind in POWDER or (rep["cups"]["maxArea"] <= card["cupArea"]
                                             and rep["peel"]["maxLayerArea"] <= card["peelArea"])))
    return rep
