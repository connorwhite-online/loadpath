"""
From a density field to a part you can print.

1. Smooth the projected density on a 2x finer grid and take the 0.5
   isosurface (marching cubes). Upsampling first is what turns voxel steps
   into the organic surface; smoothing the field (not the mesh) keeps it
   watertight.
2. Drop specks: components too small to matter are optimizer noise.
3. Exact booleans with manifold3d: union the preserved primitives, then
   subtract the keep-outs. The isosurface can only approximate a bore to
   within a cell; after the subtract, a 22mm bearing seat is 22mm.
4. Keep the bodies connected to the preserved geometry (and the largest
   body), decimate to a triangle budget, and return a closed manifold mesh.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .domain import Domain
from .shapes import manifold_from_trimesh, trimesh_from_manifold
from .spec import Study

__all__ = ["reconstruct", "ReconstructOptions"]


class ReconstructOptions:
    def __init__(self, upsample=2, smooth_sigma=0.6, level=0.5, max_triangles=250_000,
                 min_body_fraction=0.002):
        self.upsample = upsample            # field refinement before meshing
        self.smooth_sigma = smooth_sigma    # in fine cells
        self.level = level
        self.max_triangles = max_triangles
        self.min_body_fraction = min_body_fraction


def _isosurface(domain: Domain, xphys: np.ndarray, o: ReconstructOptions, keep_out=()):
    import trimesh
    from skimage.measure import marching_cubes

    grid = domain.full_grid(xphys)
    pad = 2
    grid = np.pad(grid, pad)
    fine = ndimage.zoom(grid, o.upsample, order=1, mode="nearest", grid_mode=True)
    if o.smooth_sigma > 0:
        fine = ndimage.gaussian_filter(fine, o.smooth_sigma)
    step = domain.pitch / o.upsample
    # zoom(grid_mode=True) maps fine cell centers to coarse cell centers, so
    # fine index i sits at coarse-cell coordinate (i + 0.5) / upsample - 0.5.
    origin = domain.origin - pad * domain.pitch + 0.5 * step
    if keep_out:
        # Fade the field to zero at every keep-out surface. Smoothing and
        # upsampling otherwise carry material across a keep-out (a cable
        # channel through the middle of a link), and the exact subtract
        # afterwards leaves a skin a fraction of a millimetre thick wrapped
        # round it: unprintable, and found by the eval suite's member check.
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in fine.shape], indexing="ij"), -1).reshape(-1, 3)
        pts = origin + idx * step
        clearance = 0.5 * domain.pitch
        d = np.min([s.sdf(pts) for s in keep_out], axis=0)
        fine = fine * np.clip(d / clearance, 0.0, 1.0).reshape(fine.shape)
    if fine.max() < o.level:
        raise ValueError("the optimized density never reaches solid; nothing to mesh")
    verts, faces, _, _ = marching_cubes(fine, o.level, spacing=(step, step, step))
    verts = verts + origin
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _bodies(man):
    return sorted(man.decompose(), key=lambda b: -b.volume())


def reconstruct(study: Study, domain: Domain, xphys: np.ndarray, opts: ReconstructOptions | None = None):
    """Returns (trimesh.Trimesh, info dict)."""
    import manifold3d as m3d

    o = opts or ReconstructOptions()
    iso = _isosurface(domain, xphys, o, [r.shape for r in study.keep_out])
    info = {"isoTriangles": int(len(iso.faces))}
    try:
        organic = manifold_from_trimesh(iso)
    except ValueError:
        # Marching cubes is closed by construction, but a degenerate sliver can
        # still trip manifold3d; trimesh's repair is enough for those.
        import trimesh
        trimesh.repair.fix_winding(iso)
        trimesh.repair.fill_holes(iso)
        organic = manifold_from_trimesh(iso)

    total = organic.volume()
    keep = [b for b in organic.decompose() if b.volume() >= o.min_body_fraction * total]
    organic = m3d.Manifold.batch_boolean(keep, m3d.OpType.Add) if len(keep) > 1 else keep[0]

    part = organic
    if study.preserve:
        pres = m3d.Manifold.batch_boolean([r.shape.to_manifold() for r in study.preserve], m3d.OpType.Add)
        part = part + pres
    if study.keep_out:
        ko = m3d.Manifold.batch_boolean([r.shape.to_manifold() for r in study.keep_out], m3d.OpType.Add)
        part = part - ko

    bodies = _bodies(part)
    if not bodies:
        raise ValueError("reconstruction produced an empty part")
    # A body is part of the result if it is the main one or it carries
    # preserved geometry; anything else is a fragment the subtract cut loose.
    kept = [bodies[0]]
    dropped = 0.0
    if study.preserve:
        pres_man = m3d.Manifold.batch_boolean([r.shape.to_manifold() for r in study.preserve], m3d.OpType.Add)
    for b in bodies[1:]:
        if study.preserve and (b ^ pres_man).volume() > 1e-6:
            kept.append(b)
        else:
            dropped += b.volume()
    part = m3d.Manifold.batch_boolean(kept, m3d.OpType.Add) if len(kept) > 1 else kept[0]
    info["bodies"] = len(kept)
    info["droppedFragmentVolume"] = float(dropped)

    mesh = trimesh_from_manifold(part)
    if len(mesh.faces) > o.max_triangles:
        try:
            simplified = mesh.simplify_quadric_decimation(face_count=o.max_triangles)
            if simplified.is_watertight:
                mesh = simplified
        except Exception:  # noqa: BLE001 - decimation is an optimization, never fatal
            pass
    info["triangles"] = int(len(mesh.faces))
    info["volume"] = float(mesh.volume)
    info["watertight"] = bool(mesh.is_watertight)
    return mesh, info
