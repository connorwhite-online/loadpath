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
# minWall is never under Materialize's published minimum wall for the
# process (dfm_reference.PROCESSES: SLS/MJF 0.8, SLA/DLP 0.6), the DFM
# reference for printed parts; every card carries it as "materialize".
# Escape holes, clearance, warp and peel are loadpath's own: Materialize
# publishes none of them.
RULES = {
    "sls": {"minWall": 0.8, "escapeHole": 4.0, "escapeHoles": 2, "minClearance": 0.5,
            "warpThickness": 2.0, "warpSpan": 100.0,
            "note": "SLS nylon: no supports; leave >= 2 escape holes of >= 4 mm in hollow parts; long flat "
                    "panels under 2 mm warp"},
    "mjf": {"minWall": 0.8, "escapeHole": 5.0, "escapeHoles": 2, "minClearance": 0.4,
            "warpThickness": 2.0, "warpSpan": 100.0,
            "note": "HP Multi Jet Fusion: as SLS; HP asks for >= 2 escape holes of >= 5 mm"},
    "sla": {"minWall": 0.6, "escapeHole": 3.5, "escapeHoles": 2, "minClearance": 0.5,
            "peelArea": 2500.0, "cupArea": 10.0,
            "note": "resin vat (bottom-up SLA): supports below 45 degrees and at islands; drains >= 3.5 mm "
                    "plus a vent near the plate in hollow parts; tilt large cross-sections"},
    "dlp": {"minWall": 0.6, "escapeHole": 3.5, "escapeHoles": 2, "minClearance": 0.3,
            "peelArea": 2000.0, "cupArea": 10.0,
            "note": "resin vat (DLP/LCD, bottom-up): as SLA; the whole layer cures at once, so peel force "
                    "grows with cross-section faster"},
}
# Material-specific changes to a process's rules: TPU powder tears thin walls
# out in depowdering and does not warp like nylon. TPU's 1.0 mm is a margin
# kept on purpose over Materialize's 0.8 mm SLS/MJF TPU minimum wall, which
# stays the hard limit (Connor, 2026-11-08: "seems safer").
_MATERIAL_RULES = {
    "tpu88a": {"minWall": 1.0, "warpSpan": None,
               "minWallWhy": "1.0 mm is a margin over Materialize's 0.8 mm SLS/MJF TPU minimum, the hard limit"},
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
    from .dfm_reference import limits

    ref = limits(kind, material)
    if ref is not None:
        out["materialize"] = {k: ref[k] for k in ("minWall", "minDetail", "accessed")}
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


def _walls(solid, pitch, origin, min_wall, rot=None, drawn=None):
    """Walls under the process's minimum, judged by what the print loses.

    A powder bed forms what a ball of the minimum wall's diameter can sweep
    inside the part; what it cannot reach (the part less its opening by that
    ball) does not form, or breaks off in depowdering. Where that material
    reaches no further than the process resolves from the printed part (the
    minimum wall, plus a cell for the grid), the print only rounds an edge:
    the feather edge where a skin meets a bore at a slant comes out as a
    small round on the bore's lip, as every printed edge does. Where it
    reaches further, a wall, strut or fin is missing (`unformed`), and the
    part fails. A percentile of ridge thickness judged this before, which
    let the same thin edges pass on a part with more material elsewhere and
    fail on a lighter one (the wrist link's bore lips: p1 1.31 mm at 30.7 g,
    0.57 mm at 20.7 g, the same 107 cells)."""
    from .voxels import wall_thickness

    if pitch > min_wall / 2.0 + 1e-9:
        return {"minWallMm": None, "ok": None,
                "inconclusive": f"pitch {pitch:.3g} mm (cell budget) is coarser than half the {min_wall:g} mm wall"}
    w = wall_thickness(solid, pitch, cap_mm=4.0 * min_wall, limit_mm=min_wall, spots=5, origin=origin,
                       thin_cells=True)
    reach = min_wall + pitch
    unformed, rounded, judged = _unformed(solid, w.get("thinCells"), pitch, min_wall, reach, cells=bool(drawn))
    R = rot if rot is not None else np.eye(3)
    world = (lambda ijk: [round(float(c), 2) for c in R.T @ (origin + (np.asarray(ijk, float) + 0.5) * pitch)])
    for u in unformed:
        u["at"] = world(u.pop("ijk"))
        cells = u.pop("cells", None)
        u.pop("cellReach", None)
        if drawn and cells is not None:
            pts = (origin + (cells + 0.5) * pitch) @ R
            inside = [(rid, float((sh.sdf(pts) < 0).mean())) for rid, sh in drawn]
            u["drawn"] = [rid for rid, f in inside if f >= 0.1]
            u["drawnShare"] = round(float((np.min([sh.sdf(pts) for _, sh in drawn], axis=0) < 0).mean()), 2)
    out = {"minWallMm": w.get("minMm"), "p1WallMm": w.get("p1Mm"), "thinAreaMm2": w.get("thinAreaMm2", 0.0),
           "thinSpots": w.get("thinSpots", []), "toleranceMm": pitch,
           # the thinnest wall the print is missing (None: it forms every wall)
           "unformedWallMm": judged,
           "unformed": unformed[:10], "unformedVolumeMm3": round(float(sum(u["volumeMm3"] for u in unformed)), 2),
           "roundedEdgeVolumeMm3": round(float(rounded), 2),
           "ok": not unformed,
           "basis": "ok: no material the process cannot form reaches more than the minimum wall plus a cell from "
                    "what it does form (unformed); thinner material nearer than that only rounds an edge "
                    "(roundedEdgeVolumeMm3). minWallMm and p1WallMm are the ridge readings, for reference"}
    # the wall the check holds to the minimum: the thinnest missing wall, or
    # the thinnest one formed
    out["judgedWallMm"] = judged if judged is not None else max(w.get("minMm") or min_wall, min_wall)
    return out


def _unformed(solid, thin, pitch, min_wall, reach, cells=False):
    """Material a ball of `min_wall` diameter cannot sweep, near the ridge
    cells `thin` (grid indices) thinner than that: each piece's volume and
    how far it reaches from the swept part. Works in boxes round clusters of
    thin cells (what is not swept lies within a ball's radius of them).
    Returns (unformed pieces reaching past `reach` mm, volume of the rest,
    thinnest ridge reading among the unformed). With `cells`, each piece
    also carries its grid indices ("cells", an (n, 3) array) and each cell's
    reach from the swept part ("cellReach", mm)."""
    if thin is None or not len(thin):
        return [], 0.0, None
    r = 0.5 * min_wall / pitch
    halo = int(np.ceil(2.0 * r + reach / pitch)) + 3
    # clusters: thin cells within a ball's diameter of each other
    mark = np.zeros(solid.shape, bool)
    mark[tuple(thin.T)] = True
    lo_all, hi_all = thin.min(axis=0), thin.max(axis=0) + 1
    box = tuple(slice(max(0, a - halo), min(n, b + halo)) for a, b, n in zip(lo_all, hi_all, solid.shape))
    grown = ndimage.binary_dilation(mark[box], iterations=int(np.ceil(2.0 * r)) + 1)
    lbl, n = ndimage.label(grown, _CONN)
    off = np.array([s.start for s in box])
    unformed, rounded, judged = [], 0.0, None
    for k, sl in enumerate(ndimage.find_objects(lbl), start=1):
        if sl is None:
            continue
        lo = np.maximum(np.array([s.start for s in sl]) + off - halo, 0)
        hi = np.minimum(np.array([s.stop for s in sl]) + off + halo, solid.shape)
        crop = tuple(slice(a, b) for a, b in zip(lo, hi))
        sub = solid[crop]
        # void outside the crop where the crop meets the grid's edge
        din = ndimage.distance_transform_edt(np.pad(sub, 1, constant_values=False))[1:-1, 1:-1, 1:-1]
        core = din > r
        swept = (ndimage.distance_transform_edt(~core) <= r) & sub if core.any() else np.zeros_like(sub)
        lost = sub & ~swept
        mine = np.zeros_like(sub)
        cl = (lbl == k) & mark[box]
        # this cluster's own thin cells, in the crop's frame
        idx = np.argwhere(cl) + off - lo
        idx = idx[np.all((idx >= 0) & (idx < sub.shape), axis=1)]
        mine[tuple(idx.T)] = True
        ll, ln = ndimage.label(lost, _CONN)
        keep = np.unique(ll[mine & lost])
        keep = keep[keep > 0]
        if not len(keep):
            continue
        far = ndimage.distance_transform_edt(~swept) * pitch if swept.any() else np.full(sub.shape, np.inf)
        cell3 = pitch ** 3
        for j in keep:
            piece = ll == j
            depth = float(far[piece].max())
            vol = float(piece.sum() * cell3)
            if depth > reach:
                # the ridge reading of the thinnest thin cell in the piece
                t_here = (round(float(2.0 * din[piece & mine].min() - 0.5) * pitch, 3)
                          if (piece & mine).any() else None)
                unformed.append({"volumeMm3": round(vol, 2), "reachMm": round(depth, 2), "wallMm": t_here,
                                 "ijk": (np.argwhere(piece).mean(axis=0) + lo).tolist()})
                if cells:
                    unformed[-1]["cells"] = np.argwhere(piece) + lo
                    unformed[-1]["cellReach"] = far[piece]
                if t_here is not None:
                    judged = t_here if judged is None else min(judged, t_here)
            else:
                rounded += vol
    unformed.sort(key=lambda u: -u["volumeMm3"])
    return unformed, rounded, judged


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
                  bounds=None, rules: dict | None = None, slicer: bool = True, drawn=None) -> dict:
    """Process checks for powder-bed polymer (sls, mjf) and resin (sla, dlp)
    parts. `part` is a trimesh (mm) or a field f(P) < 0 inside with
    `bounds`. `process` is a name or a spec.Process; `build` the direction
    layers stack ((axis, sign), a vector or '+z'; default the process's,
    else +z), which only resin and the layer report depend on. `rules`
    overrides any RULES entry. slicer=True (meshes, resin) adds printcheck's
    support and island read. `drawn`: the study's preserved regions (id,
    shape), so a wall that will not form says whose it is: a drawn shape
    cut thin by the keep-outs is the study's to change."""
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

    from .dfm_reference import review

    ref = review(kind, material, extents=extents,
                 wall_floor=getattr(process, "min_feature", None) if hasattr(process, "kind") else None)
    if ref is not None:
        rep["materialize"] = ref
        warnings.extend(ref["warnings"])

    rep["walls"] = _walls(solid, p, origin, card["minWall"], rot, drawn)
    if rep["walls"]["ok"] is False:
        u = rep["walls"]["unformed"][0]
        who = sorted({d for x in rep["walls"]["unformed"] for d in x.get("drawn", [])})
        warnings.append(f"walls down to {rep['walls']['judgedWallMm']:.2f} mm are under the {card['minWall']:g} mm "
                        f"{kind} minimum and will not form: {rep['walls']['unformedVolumeMm3']:.1f} mm^3 the "
                        f"process cannot make, reaching up to {u['reachMm']:.1f} mm from the rest (largest at "
                        f"{u['at']})" + (f"; it is drawn geometry ({', '.join(who)}) the keep-outs cut thin, so "
                                         "the fix is in the study: move the keep-out or the drawn shape so a "
                                         f"{card['minWall']:g} mm wall is left, or cut the drawn shape through"
                                         if who else "; thicken it to the minimum or let it go"))

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

            pc = check_print(part, build, process=kind, material=material,
                             bridge_span=getattr(process, "bridge_span", None))
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
