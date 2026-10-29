"""
Can it be printed as it stands, in this orientation, without supports?

verify._overhang counts down-facing surface by its normal. That is the right
first number, but it calls every flat ceiling a defect, and FDM prints many
of them fine: a ceiling whose every point is held at both ends by walls
already printed is a bridge, and a slicer strings filament across it. It
also says nothing about the first layer, which is where most failed prints
fail. This module slices the part along the build direction and reads each
layer the way a slicer would:

  support   - cells of a layer with nothing under them within 45 degrees
              (Langelaar's stencil, the same one am.AMFilter designs to),
              grouped into regions per layer. A region is a bridge when a
              straight span through every one of its cells, in one common
              direction, lands on supported material at both ends and is no
              longer than `bridge_span`; otherwise it needs support. Short
              ledges (`ledge` mm, a few extrusion widths) print as overhang.
  islands   - regions with no printed material beside them at all: a new
              start in mid-air, which no slicer can print unsupported.
  bed       - the first layer: contact area, how many separate pieces touch
              the bed and the smallest, whether the centre of mass sits over
              the contact's hull (and by how much), and how tall the part
              stands over its narrowest base width. A part that tips, or a
              2 mm^2 foot, fails before anything else matters.
  support volume - what a slicer would add, column by column from every
              unbridged cell down to the part or the bed: a cost to compare
              orientations by, not a slicer's exact number.

The part is voxelized at `pitch` (the layer here is a cell, not the printer's
layer height: 45-degree geometry is scale-free, bridges and ledges are
measured in mm). Results carry the pitch they are good to.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["check_print", "rotation_to_z", "BRIDGE_SPAN"]

# Unsupported spans a slicer bridges reliably, mm: PLA/PETG with part
# cooling manage 10-20 mm cleanly and sag past ~25-50. Conservative default.
BRIDGE_SPAN = {"fdm": 15.0}
# A ledge this deep prints as an overhang off the wall below it.
_LEDGE = 1.0
# Bed pieces smaller than this lift: a foot under ~3x3 mm rarely holds.
_MIN_FOOT = 9.0
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
_EIGHT = np.ones((3, 3), bool)


def rotation_to_z(direction) -> np.ndarray:
    """Rotation matrix taking the build direction to +z (proper, det +1)."""
    from .am import build_vector

    u = build_vector(direction)
    z = np.array([0.0, 0.0, 1.0])
    c = float(u @ z)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(u, z)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)


def _spans(region, anchor, pitch, max_steps):
    """For each direction (0..165 deg), the chord through every cell of a
    layer region and whether both ends land on anchor cells. Returns the
    best direction's (fraction of cells spanned, longest chord in mm)."""
    cells = np.argwhere(region)
    shape = np.array(region.shape)
    best = (0.0, np.inf)
    for th in np.radians(np.arange(0, 180, 15)):
        d = np.array([np.cos(th), np.sin(th)])
        ok = np.ones(len(cells), bool)
        length = np.zeros(len(cells))
        for sgn in (1.0, -1.0):
            done = np.zeros(len(cells), bool)
            hit = np.zeros(len(cells), bool)
            for step in range(1, max_steps + 1):
                q = np.rint(cells + sgn * step * d).astype(int)
                inside = np.all((q >= 0) & (q < shape), axis=1)
                qc = np.clip(q, 0, shape - 1)
                in_reg = inside & region[qc[:, 0], qc[:, 1]]
                on_anchor = inside & anchor[qc[:, 0], qc[:, 1]]
                newly = ~done & ~in_reg
                hit |= newly & on_anchor
                length += np.where(~done & in_reg, 1.0, 0.0)
                done |= newly
                if done.all():
                    break
            ok &= done & hit
        frac = float(ok.mean())
        span = float((length[ok].max() + 1.0) * pitch) if ok.any() else np.inf
        if frac > best[0] + 1e-9 or (abs(frac - best[0]) < 1e-9 and span < best[1]):
            best = (frac, span)
    return best


def _hull_margin(pts2, com2):
    """Signed distance from the centre of mass to the contact's convex hull
    edge (positive inside)."""
    from scipy.spatial import ConvexHull, QhullError

    if len(pts2) < 3:
        return -float(np.min(np.linalg.norm(pts2 - com2, axis=1))) if len(pts2) else -np.inf
    try:
        hull = ConvexHull(pts2)
    except QhullError:
        return 0.0
    # hull.equations: n.x + d <= 0 inside
    return float(-(hull.equations[:, :2] @ com2 + hull.equations[:, 2]).max())


def check_print(mesh, direction, process: str = "fdm", pitch: float | None = None,
                bridge_span: float | None = None, ledge: float = _LEDGE, material=None,
                max_cells: int = 1_500_000) -> dict:
    """Slice `mesh` (trimesh, mm) along the build `direction` ((axis, sign)
    or a vector, the way layers stack) and report support, bridges, islands
    and bed contact. Nothing here passes or fails a part: the numbers are
    costs and risks an orientation choice weighs (orient.py)."""
    import trimesh

    from .verify import voxelize

    rot = rotation_to_z(direction)
    m = mesh.copy()
    m.apply_transform(np.block([[rot, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    m.apply_translation([0.0, 0.0, -m.bounds[0, 2]])
    ext = m.extents
    vol = float(abs(m.volume))
    if pitch is None:
        pitch = max(0.4, float((np.prod(ext) / max_cells) ** (1.0 / 3.0)))
    span_mm = float(bridge_span if bridge_span is not None else BRIDGE_SPAN.get(process, 15.0))
    # whole cells from the bed up, so no cell centre sits on the bottom face
    lo = m.bounds[0]
    n_cells = np.maximum(np.ceil((m.extents - 1e-6) / pitch), 1)
    solid, origin = voxelize(m, pitch, box=(lo, lo + n_cells * pitch))
    # first layer at the bed: drop empty layers the grid padding left below
    zs = np.nonzero(solid.any(axis=(0, 1)))[0]
    solid = solid[:, :, zs[0]:zs[-1] + 1]
    nx, ny, nz = solid.shape
    a_cell = pitch * pitch
    out = {"pitch": pitch, "height": float(ext[2]), "volume": vol,
           "buildDirection": [float(c) for c in rot.T @ np.array([0.0, 0.0, 1.0])]}

    max_steps = int(np.ceil(span_mm / pitch)) + 2
    ledge_cells = ledge / pitch
    need = np.zeros_like(solid)          # unbridged, unsupported cells
    bridges, n_islands, island_area = [], 0, 0.0
    bridged_area = 0.0
    for k in range(1, nz):
        below = ndimage.binary_dilation(solid[:, :, k - 1], _CROSS)
        free = solid[:, :, k] & ~below
        if not free.any():
            continue
        anchor = solid[:, :, k] & ~free
        lbl, n = ndimage.label(free, _EIGHT)
        for r in range(1, n + 1):
            reg = lbl == r
            ring = ndimage.binary_dilation(reg, _EIGHT) & ~reg
            if not (ring & anchor).any():
                n_islands += 1
                island_area += reg.sum() * a_cell
                need[:, :, k] |= reg
                continue
            # depth of the region from its nearest anchor: a narrow ledge
            # off a wall prints as overhang, no bridge needed
            depth = ndimage.distance_transform_edt(~anchor)[reg]
            if depth.max() <= ledge_cells + 1.0:
                continue
            frac, span = _spans(reg, anchor, pitch, max_steps)
            if frac >= 0.97 and span <= span_mm:
                bridges.append({"z": float((k + 0.5) * pitch), "span": span, "area": float(reg.sum() * a_cell)})
                bridged_area += reg.sum() * a_cell
            else:
                need[:, :, k] |= reg
    # support columns: from each unbridged cell straight down to the part or the bed
    sup_cells = 0
    run = np.zeros((nx, ny), bool)
    for k in range(nz - 1, 0, -1):
        run = (run | need[:, :, k]) & ~solid[:, :, k - 1]
        sup_cells += int(run.sum())
    out["supportArea"] = float(need.sum() * a_cell)
    out["supportVolume"] = float(sup_cells * a_cell * pitch)
    out["islands"] = n_islands
    out["islandArea"] = float(island_area)
    out["bridges"] = len(bridges)
    out["bridgedArea"] = float(bridged_area)
    out["maxBridgeSpan"] = max((b["span"] for b in bridges), default=0.0)
    out["bridgeList"] = sorted(bridges, key=lambda b: -b["span"])[:8]
    out["supportFree"] = bool(out["supportArea"] == 0.0)

    first = solid[:, :, 0]
    lbl, n = ndimage.label(first, _EIGHT)
    feet = np.bincount(lbl.ravel())[1:] * a_cell if n else np.zeros(0)
    xy = (np.argwhere(first) + 0.5) * pitch + origin[:2]
    try:
        com = m.center_mass if m.is_volume else m.centroid
    except Exception:  # noqa: BLE001 - a bad mass property is not worth failing a check
        com = m.centroid
    margin = _hull_margin(xy, np.asarray(com[:2])) if len(xy) else -np.inf
    if len(xy) >= 3:
        # narrowest width of the contact (minimum over directions of its extent)
        th = np.radians(np.arange(0, 180, 5))
        widths = [np.ptp(xy @ np.array([np.cos(t), np.sin(t)])) + pitch for t in th]
        base_w = float(min(widths))
    else:
        base_w = pitch
    out["bed"] = {
        "contactArea": float(first.sum() * a_cell),
        "contactFraction": float(first.sum() * a_cell / max(vol ** (2 / 3), 1e-9)),
        "feet": int(n),
        "smallestFoot": float(feet.min()) if n else 0.0,
        "comMargin": margin,
        "baseWidth": base_w,
        "aspect": float(ext[2] / base_w) if base_w > 0 else np.inf,
        "footprint": [float(ext[0]), float(ext[1])],
    }
    warnings = []
    bed = out["bed"]
    if bed["comMargin"] < 0:
        warnings.append(f"the centre of mass is {-bed['comMargin']:.1f} mm outside the bed contact: it tips over "
                        "unless supports or a brim hold it")
    if n and bed["smallestFoot"] < _MIN_FOOT:
        warnings.append(f"a {bed['smallestFoot']:.1f} mm^2 foot on the bed will likely lift: add a brim or "
                        "pick a face with more contact")
    if bed["aspect"] > 4.0:
        warnings.append(f"stands {bed['aspect']:.1f}x taller than its narrowest base: it will wobble late in the "
                        "print; a brim or a lower orientation helps")
    if n_islands:
        warnings.append(f"{n_islands} region(s) start in mid-air ({island_area:.0f} mm^2): they need support")
    mat = getattr(material, "name", material)
    if mat in ("abs", "asa", "pa-cf", "pa12") and max(ext[0], ext[1]) > 120 and process == "fdm":
        warnings.append(f"{max(ext[0], ext[1]):.0f} mm across the bed in {mat}: warp risk; use an enclosure "
                        "and a brim")
    out["warnings"] = warnings
    return out
