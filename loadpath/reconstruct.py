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
    def __init__(self, upsample=3, smooth_sigma=0.9, level=0.5, max_triangles=250_000,
                 min_body_fraction=0.002, taubin_iterations=25, blend=3.0, blend_k=0.8):
        self.upsample = upsample            # field refinement before meshing
        self.smooth_sigma = smooth_sigma    # in fine cells
        self.level = level
        self.max_triangles = max_triangles
        self.min_body_fraction = min_body_fraction
        # Volume-preserving Taubin smoothing of the organic surface, before
        # the exact booleans (which then restore every crisp face and bore).
        self.taubin_iterations = taubin_iterations
        # Fillet between the organic body and preserved geometry (mm of ramp
        # in the preserve field, and the smooth-max strength).
        self.blend = blend
        self.blend_k = blend_k


def _smooth_max(a, b, k):
    """Polynomial smooth maximum: max(a, b) plus a bump of up to k/4 where the
    two are close. On density fields this is a fillet where they meet."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * k * 0.25


def _isosurface(domain: Domain, xphys: np.ndarray, o: ReconstructOptions, keep_out=(), min_member=None,
                flush=(), preserve=(), design=()):
    import trimesh
    from skimage.measure import marching_cubes

    grid = domain.full_grid(xphys)
    pad = 3
    # Pad with void, so material that reaches the design space's boundary
    # rounds off just inside it: the exact intersection with the design
    # space then has nothing to cut, and no flat sawn-off faces appear where
    # the box happened to end (GRIP's wrist yoke had several). Only where a
    # part must sit flush against something (`flush`: its supports, a wall)
    # is the edge replicated, so the material runs on through the boundary
    # and the intersection cuts it off flat there, on purpose (the printed
    # hook's wall face). Two cells of that, then one of void so marching
    # cubes still closes the surface.
    padded = np.pad(grid, pad)
    if flush:
        edge = np.pad(np.pad(grid, pad - 1, mode="edge"), 1)
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in padded.shape], indexing="ij"), -1).reshape(-1, 3)
        centers = domain.origin + (idx - pad + 0.5) * domain.pitch
        near = (np.min([sh.sdf(centers) for sh in flush], axis=0) < 1.5 * domain.pitch).reshape(padded.shape)
        padded = np.where(near, edge, padded)
    grid = padded
    # Cubic upsampling: linear interpolation of a voxel field leaves the
    # staircase in the isosurface as faceted terraces.
    fine = np.clip(ndimage.zoom(grid, o.upsample, order=3, mode="nearest", grid_mode=True), 0.0, 1.0)
    if o.smooth_sigma > 0:
        fine = ndimage.gaussian_filter(fine, o.smooth_sigma)
    if min_member:
        # Morphological opening with a ball of the minimum member's diameter:
        # anything thinner than that cannot fit the ball and goes. Smoothing a
        # member that ends inside the design space tapers it to a knife edge,
        # and that sliver is both unprintable and a stress raiser that grew
        # with every refinement of the verification grid (14 -> 58 MPa on the
        # arm link at a flange tip).
        step_ = domain.pitch / o.upsample
        r = 0.5 * float(min_member) / step_
        if r >= 1.0:
            k = int(np.ceil(r))
            zz, yy, xx = np.mgrid[-k:k + 1, -k:k + 1, -k:k + 1]
            ball = (xx * xx + yy * yy + zz * zz) <= r * r
            fine = ndimage.grey_opening(fine, footprint=ball)
    step = domain.pitch / o.upsample
    # zoom(grid_mode=True) maps fine cell centers to coarse cell centers, so
    # fine index i sits at coarse-cell coordinate (i + 0.5) / upsample - 0.5.
    origin = domain.origin - pad * domain.pitch + 0.5 * step
    if preserve and o.blend > 0:
        # Fillet the organic body into the preserved geometry: the preserved
        # shapes as a density field that crosses 0.5 at their surface and
        # ramps over `blend` mm, smooth-max'd into the optimized field. Where
        # a member meets a boss the bump fills the crease; elsewhere the
        # preserve field stays under 0.5 and changes nothing. (The exact union
        # in reconstruct() still restores the crisp faces and bores.)
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in fine.shape], indexing="ij"), -1).reshape(-1, 3)
        pts = origin + idx * step
        sd = np.min([sh.sdf(pts) for sh in preserve], axis=0).reshape(fine.shape)
        pf = np.clip(0.5 - sd / (2.0 * o.blend), 0.0, 1.0)
        fine = _smooth_max(fine, pf, o.blend_k)
    if design:
        # Round material off before the design space's boundary, the way a
        # keep-out is faded below. Void padding alone left the 0.5 level
        # right on the boundary where material ran up to it, so the exact
        # intersection still sawed a flat face there (2,600 mm2 of them on
        # GRIP's forearm). Not where the part must sit flush (`flush`).
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in fine.shape], indexing="ij"), -1).reshape(-1, 3)
        pts = origin + idx * step
        inside = -np.min([sh.sdf(pts) for sh in design], axis=0)
        fade = np.clip(inside / domain.pitch, 0.0, 1.0)
        if flush:
            near = np.min([sh.sdf(pts) for sh in flush], axis=0) < 1.5 * domain.pitch
            fade[near] = 1.0
        fine = fine * fade.reshape(fine.shape)
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
    flush = []
    for sup in study.supports:
        try:
            flush.append(sup.shape if sup.shape is not None else study.region_shape(sup.region)[0])
        except ValueError:
            pass
    iso = _isosurface(domain, xphys, o, [r.shape for r in study.keep_out], study.min_member,
                      flush=flush, preserve=[r.shape for r in study.preserve], design=list(study.design_space))
    info = {"isoTriangles": int(len(iso.faces))}
    if o.taubin_iterations:
        import trimesh
        trimesh.smoothing.filter_taubin(iso, lamb=0.5, nu=-0.53, iterations=o.taubin_iterations)
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

    # Trim to the design space (and preserved geometry) exactly: flat,
    # flush faces wherever the design reaches its boundary.
    envelope = m3d.Manifold.batch_boolean(
        [sh.to_manifold() for sh in study.design_space] + [r.shape.to_manifold() for r in study.preserve],
        m3d.OpType.Add)
    organic = organic ^ envelope
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
