"""
Printability checks on the produced mesh: wall thickness, overhangs, and
trapped volumes.

Watertight/manifold (app.py) tell you the mesh is a valid solid. They say
nothing about whether it can be MADE. These three are the checks that decide
that, and none of them existed before:

  min wall   — a 0.4mm rib is watertight and unprintable. Measured by casting
               a ray inward from each sampled face along -normal and taking
               the distance to the far wall.
  overhangs  — a downward-facing surface shallower than ~45 deg from
               horizontal needs support. Reported as area and as a fraction
               of total area, not a pass/fail, because "needs some support"
               is a cost, not a defect.
  trapped    — an enclosed void cannot drain resin or release SLS powder.
               Found by labeling void space and keeping the components that
               never reach the build volume's boundary.

Resolution honesty, same contract as networks.py: the voxel checks are
verified at the reported `pitch`. A trapped void smaller than a voxel, or an
escape channel narrower than one, is below this check's resolution — the
report carries the pitch so callers can state the bound they actually have.

Every value is measured off the FINAL exported mesh, so it reflects what
ships, not what the script intended.
"""
import os

import numpy as np
from scipy import ndimage

from sdf_kit import _solid_voxels

__all__ = ["check_dfm"]

# Defaults are the conservative multi-process envelope (lib/cad/knowledge/dfm.ts
# carries the per-process numbers; the caller passes the real ones in `spec`).
_DEFAULT_MIN_WALL_MM = 1.5
_DEFAULT_OVERHANG_DEG = 45.0

# Ray budget for the thickness probe. Thickness is sampled per face, and the
# pure-numpy ray engine is O(rays x triangles) — a 400k-triangle TPMS mesh
# would take minutes at one ray per face. Area-weighted sampling keeps the
# estimate honest (big faces get proportionally more rays) at a fixed cost.
_MAX_THICKNESS_RAYS = int(os.environ.get("CAD_DFM_MAX_RAYS", "4000"))

# Voxel ceiling for the trapped-void grid, mirroring networks.py's budget.
_VOXEL_BUDGET = int(os.environ.get("CAD_DFM_VOXEL_BUDGET", str(64_000_000)))

# 26-connectivity for the void: the LEAKIEST choice, so a diagonal voxel
# chain counts as an escape path. We would rather miss a trapped void than
# invent one — a false "trapped" reading sends the repair loop drilling a
# drain hole through a part that never needed one.
_CONN = np.ones((3, 3, 3), bool)

# A void component this small is voxelization aliasing between shell voxels,
# not a pocket (see the same triage in sdf_kit._solid_voxels).
_TINY_VOID_VOX = 32


def _choose_pitch(mesh, requested=None):
    """Voxel pitch that fits the budget. Honors `requested` unless it would
    blow the grid, in which case it coarsens and the report says so."""
    extents = np.asarray(mesh.extents, float)
    span = float(max(extents.max(), 1e-6))
    pitch = float(requested) if requested else max(span / 128.0, 0.25)
    while np.prod(np.maximum(extents / pitch, 1.0)) > _VOXEL_BUDGET:
        pitch *= 1.5
    return pitch


def _wall_thickness(mesh, min_wall):
    """Ray-cast wall thickness, in mm, sampled over faces.

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


def _trapped_voids(mesh, pitch):
    """Enclosed voids that cannot drain resin or release powder.

    Voxelize, label the void with 26-connectivity, and keep the components
    that never touch the padded boundary. `_solid_voxels` already classifies
    internal cavities as void (a naive fill would pave them over), which is
    exactly what makes them findable here."""
    solid, origin = _solid_voxels(mesh, pitch)
    lbl, count = ndimage.label(~solid, structure=_CONN)
    if count == 0:
        return [], 0.0

    outside = set()
    for face in (lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1], lbl[:, :, 0], lbl[:, :, -1]):
        outside.update(np.unique(face).tolist())
    outside.discard(0)

    sizes = np.bincount(lbl.ravel(), minlength=count + 1)
    voxel_mm3 = pitch ** 3
    voids = []
    for label in range(1, count + 1):
        if label in outside or sizes[label] < _TINY_VOID_VOX:
            continue
        center = np.asarray(ndimage.center_of_mass(lbl == label), float)
        voids.append(
            {
                "volumeMm3": round(float(sizes[label]) * voxel_mm3, 3),
                "center": [round(float(v), 2) for v in (origin + center * pitch)],
                "voxels": int(sizes[label]),
            }
        )
    voids.sort(key=lambda v: v["volumeMm3"], reverse=True)
    return voids, float(sum(v["volumeMm3"] for v in voids))


def check_dfm(mesh, spec=None):
    """Printability report for a final mesh. Never raises for a geometry it
    cannot measure — an unmeasurable field comes back None/empty with the
    rest of the report intact, so one bad probe never fails a run (the
    caller in app.py wraps this in its own failure isolation too)."""
    spec = spec or {}
    min_wall = float(spec.get("minWall", _DEFAULT_MIN_WALL_MM))
    overhang_deg = float(spec.get("overhangDeg", _DEFAULT_OVERHANG_DEG))
    build_dir = np.asarray(spec.get("build", (0.0, 0.0, 1.0)), float)
    norm = np.linalg.norm(build_dir)
    build_dir = build_dir / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    pitch = _choose_pitch(mesh, spec.get("pitch"))

    report = {
        "pitch": round(pitch, 4),
        "watertight": bool(mesh.is_watertight),
        "triangles": int(len(mesh.faces)),
        "volumeMm3": round(float(abs(mesh.volume)), 3),
        "minWallTargetMm": min_wall,
        "overhangLimitDeg": overhang_deg,
    }

    try:
        thin_min, thin_p1, thin_area, sampled = _wall_thickness(mesh, min_wall)
        report["minWallMm"] = None if thin_min is None else round(thin_min, 3)
        report["wallP1Mm"] = None if thin_p1 is None else round(thin_p1, 3)
        report["thinAreaMm2"] = round(thin_area, 3)
        report["thinAreaFraction"] = (
            round(thin_area / sampled, 4) if sampled > 0 else 0.0
        )
        # Judge on the 1st percentile, not the raw minimum: one degenerate
        # sliver triangle must not condemn an otherwise printable part.
        report["minWallOk"] = thin_p1 is None or thin_p1 >= min_wall
    except Exception as err:  # noqa: BLE001 — failure-isolated per probe
        # Unknown, NOT pass. Only checks that RAN are ever counted as
        # verified (the honesty rail lib/cad/dimension-check.ts holds to).
        report["minWallError"] = str(err)
        report["minWallOk"] = None

    try:
        over_area, total_area = _overhangs(mesh, overhang_deg, build_dir)
        report["overhangAreaMm2"] = round(over_area, 3)
        report["overhangFraction"] = (
            round(over_area / total_area, 4) if total_area > 0 else 0.0
        )
    except Exception as err:  # noqa: BLE001
        report["overhangError"] = str(err)

    try:
        voids, trapped_mm3 = _trapped_voids(mesh, pitch)
        report["trappedVoids"] = voids[:10]
        report["trappedVoidCount"] = len(voids)
        report["trappedVolumeMm3"] = round(trapped_mm3, 3)
        report["drainsOk"] = len(voids) == 0
    except Exception as err:  # noqa: BLE001
        report["trappedError"] = str(err)
        report["drainsOk"] = None

    # `ok` is "every probe RAN and passed" — an unrun probe (None) makes the
    # part unverified, not fine. `ok: False` with a *Error field beside it
    # means "could not check", which the repair loop must not read as a
    # geometry defect; that is what the explicit `probesRan` flag is for.
    report["probesRan"] = (
        report.get("minWallOk") is not None and report.get("drainsOk") is not None
    )
    report["ok"] = bool(
        report.get("watertight")
        and report.get("minWallOk") is True
        and report.get("drainsOk") is True
    )
    return report
