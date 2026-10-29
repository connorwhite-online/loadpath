"""
Printability checks on a final part: wall thickness, overhangs, and trapped
volumes.

Watertight and manifold tell you the mesh is a valid solid. They say nothing
about whether it can be MADE. These three decide that:

  min wall   — a 0.4 mm rib is watertight and unprintable. Measured on a voxel
               grid: twice the distance to void along the medial ridge (the
               same reading verify.py uses), from a distance transform run in
               tiles, so memory stays flat however many triangles the part
               has. Reported in mm with the grid's tolerance (+/- one pitch).
               `wallMethod: "ray"` keeps the older ray cast from each sampled
               face along -normal: not quantized, but it goes through
               trimesh's ray broad phase, which on a 6.6M-triangle TPMS pod
               reached 63 GB. Use it only on small meshes.
  overhangs  — a downward-facing surface shallower than ~45 deg from
               horizontal needs support. Reported as area and as a fraction
               of total area, not a pass/fail, because "needs some support"
               is a cost, not a defect.
  trapped    — an enclosed void cannot drain resin or release powder. Found by
               labeling void space and keeping the components that never
               reach the grid's boundary.

The part is a trimesh, or an implicit field f(P) < 0 inside (pass
spec["bounds"] = (lo, hi)); a field skips meshing and voxelizing altogether.
Overhangs need a mesh.

Resolution honesty, same contract as networks.py: the voxel probes are
verified at the reported `pitch`. A trapped void smaller than a voxel, or an
escape channel narrower than one, is below this check's resolution, and a
wall reading carries +/- pitch. When the voxel budget forces a pitch coarser
than half the minimum wall, the wall verdict is None (not checked), never a
guess.
"""
import os

import numpy as np
from scipy import ndimage

from ..voxels import choose_pitch, solid_grid, wall_thickness

__all__ = ["check_dfm"]


def _env(name, default):
    """LOADPATH_<name>, falling back to the older CAD_<name>."""
    return os.environ.get(f"LOADPATH_{name}", os.environ.get(f"CAD_{name}", default))


_DEFAULT_MIN_WALL_MM = 1.5
_DEFAULT_OVERHANG_DEG = 45.0

# Ray budget for the optional ray-cast thickness probe (wallMethod "ray").
_MAX_THICKNESS_RAYS = int(_env("DFM_MAX_RAYS", "4000"))

# Voxel ceiling for the grid (1 byte each, plus a 4-byte label per voxel
# while trapped voids are labeled; the distance transform runs in tiles).
_VOXEL_BUDGET = int(_env("DFM_VOXEL_BUDGET", str(100_000_000)))

# Cells across the minimum wall the pitch aims for, and the fewest it will
# accept before calling the wall reading inconclusive.
_CELLS_PER_WALL = 3.0
_MIN_CELLS_PER_WALL = 2.0

# 26-connectivity for the void: the LEAKIEST choice, so a diagonal voxel
# chain counts as an escape path. Better to miss a trapped void than invent
# one: a false "trapped" reading has someone drilling a drain hole through a
# part that never needed one.
_CONN = np.ones((3, 3, 3), bool)

# A void component this small is voxelization aliasing between shell voxels,
# not a pocket.
_TINY_VOID_VOX = 32


def _choose_pitch(extents, min_wall, requested=None):
    """Voxel pitch that fits the budget. Honors `requested` (else aims for
    three cells across the minimum wall) unless it would blow the grid, in
    which case it coarsens and the report says so."""
    extents = np.asarray(extents, float)
    span = float(max(extents.max(), 1e-6))
    target = float(requested) if requested else max(min_wall / _CELLS_PER_WALL, span / 2048.0, 0.02)
    return choose_pitch(extents, target, _VOXEL_BUDGET)


def _voxel_wall(solid, origin, pitch, min_wall):
    """Grid wall thickness: (min, p1, thin area, thin fraction, spots)."""
    w = wall_thickness(solid, pitch, cap_mm=_cap_mm(min_wall, pitch), limit_mm=min_wall, origin=origin)
    if w["minMm"] is None:
        return None, None, 0.0, 0.0, []
    return w["minMm"], w["p1Mm"], w["thinAreaMm2"], w["thinSurfaceFraction"], w.get("thinSpots", [])


def _cap_mm(min_wall, pitch):
    """Thickest wall the voxel probe resolves; thicker reads as this cap. It
    sets the tile overlap, so it stays a few walls deep, not the whole part."""
    return max(8.0 * min_wall, 16.0 * pitch)


def _ray_wall(mesh, min_wall):
    """Ray-cast wall thickness (wallMethod "ray"), in mm, sampled over faces.

    From each sampled face centroid, step just inside the surface and cast
    along -normal; the first hit is the far wall. Returns (min, p1, area
    below `min_wall`, sampled area) — p1 as well as min because a single
    degenerate sliver triangle drags the minimum down and says nothing about
    the part, while the 1st percentile tracks real thin regions."""
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return None, None, 0.0, 0.0

    areas = np.asarray(mesh.area_faces, float)
    normals = np.asarray(mesh.face_normals, float)
    centers = np.asarray(mesh.triangles_center, float)

    n = min(len(faces), _MAX_THICKNESS_RAYS)
    if n < len(faces):
        # Area-weighted without replacement: sample where the surface is.
        weights = areas / areas.sum() if areas.sum() > 0 else None
        rng = np.random.default_rng(0)  # deterministic runs
        idx = rng.choice(len(faces), size=n, replace=False, p=weights)
    else:
        idx = np.arange(len(faces))

    # Step inside by a fraction of the local triangle scale, so the ray does
    # not immediately re-hit its own source face.
    eps = np.maximum(np.sqrt(areas[idx]) * 1e-3, 1e-6)[:, None]
    origins = centers[idx] - normals[idx] * eps
    directions = -normals[idx]

    # Deliberately NOT caught here. trimesh's ray engine needs `rtree`, and a
    # missing optional dep must surface as a recorded probe error, never as a
    # quiet "no thin walls found" — the same silent-degradation trap
    # requirements.txt already warns about for shapely/rtree/mapbox-earcut.
    hits, ray_idx, _tri = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    if len(ray_idx) == 0:
        return None, None, 0.0, 0.0

    dist = np.linalg.norm(hits - origins[ray_idx], axis=1)
    sampled_area = float(areas[idx][ray_idx].sum())
    thin = dist < min_wall
    return (
        float(dist.min()),
        float(np.percentile(dist, 1)),
        float(areas[idx][ray_idx][thin].sum()),
        sampled_area,
    )


def _wall_thickness(part, min_wall, method="voxel", grid=None):
    """Wall thickness by `method`: (min, p1, thin area mm2, thin fraction,
    thin spots). `grid` = (solid, origin, pitch) for the voxel probe."""
    if method == "ray":
        if callable(part) and not hasattr(part, "faces"):
            raise ValueError("the ray probe needs a mesh; use the voxel probe for a field")
        lo, p1, thin, sampled = _ray_wall(part, min_wall)
        return lo, p1, thin, (thin / sampled if sampled > 0 else 0.0), []
    solid, origin, pitch = grid
    return _voxel_wall(solid, origin, pitch, min_wall)


def _overhangs(mesh, overhang_deg, build_dir):
    """Downward-facing area shallower than `overhang_deg` from horizontal.

    A face's inclination from horizontal is acos(-n . build_dir): a normal
    pointing straight down is a 0-degree (horizontal ceiling) overhang, the
    worst case. Faces sitting ON the build plate are excluded — the plate
    supports them."""
    normals = np.asarray(mesh.face_normals, float)
    areas = np.asarray(mesh.area_faces, float)
    total = float(areas.sum())
    if total <= 0:
        return 0.0, 0.0

    down = -(normals @ build_dir)
    with np.errstate(invalid="ignore"):
        incline = np.degrees(np.arccos(np.clip(down, -1.0, 1.0)))

    # Exclude the footprint: faces whose centroid sits within one layer of
    # the lowest point along the build direction rest on the plate.
    heights = np.asarray(mesh.triangles_center, float) @ build_dir
    on_plate = heights <= heights.min() + 1e-3

    flagged = (down > 0) & (incline < overhang_deg) & ~on_plate
    return float(areas[flagged].sum()), total


def _trapped_voids(solid, origin, pitch):
    """Enclosed voids that cannot drain resin or release powder: label the
    void with 26-connectivity and keep the components that never touch the
    padded boundary. The scanline grid keeps internal cavities void (parity
    sees the inner shell), which is what makes them findable here."""
    lbl, count = ndimage.label(~solid, structure=_CONN)
    if count == 0:
        return [], 0.0
    outside = set()
    for face in (lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1], lbl[:, :, 0], lbl[:, :, -1]):
        outside.update(np.unique(face).tolist())
    outside.discard(0)
    sizes = np.bincount(lbl.ravel(), minlength=count + 1)
    keep = [k for k in range(1, count + 1) if k not in outside and sizes[k] >= _TINY_VOID_VOX]
    voxel_mm3 = pitch ** 3
    voids = []
    if keep:
        centers = ndimage.center_of_mass(np.ones(lbl.shape, np.uint8), lbl, keep)
        for k, c in zip(keep, centers):
            voids.append({
                "volumeMm3": round(float(sizes[k]) * voxel_mm3, 3),
                "center": [round(float(v), 2) for v in (origin + np.asarray(c, float) * pitch)],
                "voxels": int(sizes[k]),
            })
    voids.sort(key=lambda v: v["volumeMm3"], reverse=True)
    return voids, float(sum(v["volumeMm3"] for v in voids))


def check_dfm(part, spec=None):
    """Printability report for a final part (a trimesh, or an implicit field
    with spec["bounds"]). Never raises for a geometry it cannot measure: an
    unmeasurable field comes back None/empty with the rest of the report
    intact, so one bad probe never fails a run.

    spec: minWall (mm, 1.5), overhangDeg (45), build (vector, +z), pitch (mm,
    default a third of minWall within the voxel budget), wallMethod ("voxel"
    or "ray"), bounds ((lo, hi) mm, required for a field)."""
    spec = spec or {}
    min_wall = float(spec.get("minWall", _DEFAULT_MIN_WALL_MM))
    overhang_deg = float(spec.get("overhangDeg", _DEFAULT_OVERHANG_DEG))
    build_dir = np.asarray(spec.get("build", (0.0, 0.0, 1.0)), float)
    norm = np.linalg.norm(build_dir)
    build_dir = build_dir / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    method = str(spec.get("wallMethod", "voxel"))
    is_field = callable(part) and not hasattr(part, "faces")
    bounds = spec.get("bounds")
    if is_field:
        if bounds is None:
            raise ValueError("check_dfm on a field needs spec['bounds'] = (lo, hi) in mm")
        bounds = tuple(np.asarray(b, float) for b in bounds)
        extents = bounds[1] - bounds[0]
    else:
        extents = np.asarray(part.extents, float) if len(part.faces) else np.ones(3)

    pitch = _choose_pitch(extents, min_wall, spec.get("pitch"))

    report = {
        "pitch": round(pitch, 4),
        "source": "field" if is_field else "mesh",
        "minWallTargetMm": min_wall,
        "overhangLimitDeg": overhang_deg,
        "wallMethod": method,
    }
    if is_field:
        report["watertight"] = True  # an implicit solid is closed by construction
    else:
        report.update(watertight=bool(part.is_watertight), triangles=int(len(part.faces)),
                      volumeMm3=round(float(abs(part.volume)), 3) if len(part.faces) else 0.0)

    grid = None
    grid_error = None
    try:
        solid, origin, info = solid_grid(part, pitch, pad=2, bounds=bounds)
        grid = (solid, origin, pitch)
        report["gridCells"] = [int(n) for n in solid.shape]
        if is_field:
            report["volumeMm3"] = round(float(solid.sum()) * pitch ** 3, 3)
        else:
            if any(info.get("unclosedColumns", {}).values()):
                report["unclosedColumns"] = info["unclosedColumns"]
            if info.get("overlapMm3", 0.0) > 0.0:
                # bodies concatenated without a union (or a cavity shell
                # facing out): parity reads their overlap as void
                report["overlapMm3"] = round(info["overlapMm3"], 3)
                report["overlapWarning"] = (
                    "the mesh has overlapping shells (bodies joined without a boolean union) or an "
                    "outward-facing cavity; their overlap reads as void, so walls and voids there are "
                    "wrong. Union the bodies first.")
    except Exception as err:  # noqa: BLE001 — failure-isolated per probe
        grid_error = f"voxelizing failed: {err}"

    try:
        if method != "ray" and pitch > min_wall / _MIN_CELLS_PER_WALL + 1e-9:
            # Too coarse to resolve the wall: say so rather than guess.
            report["minWallOk"] = None
            report["minWallInconclusive"] = (
                f"pitch {pitch:g} mm (voxel budget) is coarser than half the {min_wall:g} mm minimum "
                "wall; pass a smaller part, a finer pitch with a larger LOADPATH_DFM_VOXEL_BUDGET, or "
                "bounds around the region to check")
        else:
            if method != "ray" and grid is None:
                raise RuntimeError(grid_error)
            thin_min, thin_p1, thin_area, thin_frac, spots = _wall_thickness(part, min_wall, method, grid)
            report["minWallMm"] = None if thin_min is None else round(thin_min, 3)
            report["wallP1Mm"] = None if thin_p1 is None else round(thin_p1, 3)
            # the voxel reading is within half a cell of the true thickness,
            # on a grid that itself staircases by half a cell
            report["wallToleranceMm"] = 0.0 if method == "ray" else round(pitch, 4)
            if method != "ray":
                # members thicker than this all read as about this
                report["wallCapMm"] = round(_cap_mm(min_wall, pitch), 3)
            report["thinAreaMm2"] = round(thin_area, 3)
            report["thinAreaFraction"] = round(thin_frac, 4)
            if spots:
                report["thinSpots"] = spots
            # Judge on the 1st percentile, not the raw minimum: one sliver
            # must not condemn an otherwise printable part. Give the voxel
            # reading its half-cell estimator error, not the whole tolerance.
            slack = 0.0 if method == "ray" else 0.5 * pitch
            report["minWallOk"] = thin_p1 is None or thin_p1 + slack >= min_wall
            if thin_p1 is not None and method != "ray" and abs(thin_p1 - min_wall) <= pitch:
                report["minWallBorderline"] = True
    except Exception as err:  # noqa: BLE001
        # Unknown, NOT pass. Only checks that RAN are ever counted as verified.
        report["minWallError"] = str(err)
        report["minWallOk"] = None

    if is_field:
        report["overhangError"] = "overhangs need a mesh; pass the meshed part"
    else:
        try:
            over_area, total_area = _overhangs(part, overhang_deg, build_dir)
            report["overhangAreaMm2"] = round(over_area, 3)
            report["overhangFraction"] = (
                round(over_area / total_area, 4) if total_area > 0 else 0.0
            )
        except Exception as err:  # noqa: BLE001
            report["overhangError"] = str(err)

    try:
        if grid is None:
            raise RuntimeError(grid_error)
        voids, trapped_mm3 = _trapped_voids(*grid)
        report["trappedVoids"] = voids[:10]
        report["trappedVoidCount"] = len(voids)
        report["trappedVolumeMm3"] = round(trapped_mm3, 3)
        report["drainsOk"] = len(voids) == 0
    except Exception as err:  # noqa: BLE001
        report["trappedError"] = str(err)
        report["drainsOk"] = None

    # `ok` is "every probe RAN and passed": an unrun probe (None) makes the
    # part unverified, not fine. `ok: False` with a *Error or *Inconclusive
    # field beside it means "could not check", not a geometry defect; that is
    # what the explicit `probesRan` flag is for.
    report["probesRan"] = (
        report.get("minWallOk") is not None and report.get("drainsOk") is not None
    )
    report["ok"] = bool(
        report.get("watertight")
        and report.get("minWallOk") is True
        and report.get("drainsOk") is True
    )
    return report
