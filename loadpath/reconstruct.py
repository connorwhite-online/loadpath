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
                 min_body_fraction=0.002, taubin_iterations=25, blend=None, blend_k=0.8, fillet=True, seat=True,
                 close=None):
        self.upsample = upsample            # field refinement before meshing
        self.smooth_sigma = smooth_sigma    # in fine cells
        self.level = level
        self.max_triangles = max_triangles
        self.min_body_fraction = min_body_fraction
        # Volume-preserving Taubin smoothing of the organic surface, before
        # the exact booleans (which then restore every crisp face and bore).
        self.taubin_iterations = taubin_iterations
        # Fillet between the organic body and preserved geometry: its radius
        # in mm (None: 1.5 x minMember, at least 3), and the smooth-max
        # strength of the older density blend (fillet=False). At 3 mm the
        # motor mount's pads, which stand 1 mm proud of the plate round them,
        # kept 108 mm of crease; at 4.5 (its minMember is 3) about 4.
        self.blend = blend
        self.blend_k = blend_k
        # Fillet in distance space (True) or the older density smooth max.
        self.fillet = fillet
        # Move material at flush faces onto the face (_seat_on_flush_faces).
        self.seat = seat
        # Round every concave edge of the organic body itself, not only where
        # it meets preserved geometry: a morphological closing in distance
        # space by this radius in mm (None: half the minMember, at least 1;
        # 0 turns it off). It only ever adds material, into corners and into
        # slots narrower than twice the radius (below the minimum member,
        # so unprintable anyway).
        self.close = close


def _smooth_max(a, b, k):
    """Polynomial smooth maximum: max(a, b) plus a bump of up to k/4 where the
    two are close. On density fields this is a fillet where they meet."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * k * 0.25


def _smooth_min(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def _smooth_union(a, b, k):
    """Polynomial smooth minimum of two signed distances: their union with a
    fillet where the surfaces meet. The fillet's sag from the sharp corner is
    k/4; at a right angle it reaches about k (1 - 1/sqrt(2)) out along each
    surface, roughly a round of radius k / (2 (sqrt(2) - 1)) = 1.2 k."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def _field_distance(fine, step, level=0.5):
    """Signed distance (mm, negative inside) to the `level` surface of a
    smooth density field: exact distance transforms away from the surface,
    and the field's own gradient within a couple of cells of it, where a
    distance transform would only resolve whole cells."""
    inside = fine >= level
    d = (ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * step
    d += np.where(inside, 0.5 * step, -0.5 * step)
    g = np.linalg.norm(np.stack(np.gradient(fine, step)), axis=0)
    near = (np.abs(fine - level) < 0.45) & (g > 1e-6)
    lin = (level - fine) / np.maximum(g, 1e-6)
    ok = near & (np.abs(lin) < 2.0 * step)
    d[ok] = lin[ok]
    return d


def _seat_on_flush_faces(fine, domain: Domain, upsample: int, pad: int, flush, design):
    """Move material at a flush face (a wall the part is held against) onto
    the face itself.

    The grid is centered on the design space, for exact symmetry, so along an
    axis whose length is not a whole number of cells the outer cell layer
    hangs past each end face by up to half a cell. A plate the optimizer
    builds against a wall occupies that layer: it was analyzed as a full cell
    thick, but the part is trimmed at the real face, so it was delivered
    thinner by the overhang (the printed hook's wall plate, 1.46 mm p5 against
    a 1.95 mm cell; the wall bracket's seat). Near a box face that a flush
    region touches, the field is translated inward by that face's overhang,
    so the plate keeps its analyzed thickness from the face in, easing back
    to the field as it was over the next cells."""
    from .shapes import Box

    step = domain.pitch / upsample
    origin = domain.origin - pad * domain.pitch + 0.5 * step
    shape = fine.shape
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    pts = origin + idx * step
    near = None
    out = fine
    grid_lo = domain.origin
    grid_hi = domain.origin + np.asarray(domain.dims) * domain.pitch
    for b in design:
        if not isinstance(b, Box):
            continue
        for a in range(3):
            for sign, face, hang in ((1, b.lo[a], b.lo[a] - grid_lo[a]), (-1, b.hi[a], grid_hi[a] - b.hi[a])):
                if hang < 0.5 * step:
                    continue
                if near is None:
                    near = np.min([sh.sdf(pts) for sh in flush], axis=0) < 1.5 * domain.pitch
                inward = sign * (pts[:, a] - face)       # mm inside the face
                w = np.clip((2.5 * domain.pitch - inward) / (1.5 * domain.pitch), 0.0, 1.0) * near
                if not w.any():
                    continue
                shift = np.zeros(3)
                shift[a] = sign * hang / step
                moved = ndimage.shift(out, shift, order=1, mode="nearest")
                w = w.reshape(shape)
                out = w * moved + (1.0 - w) * out
    return out


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
    if flush and design and o.seat:
        fine = _seat_on_flush_faces(fine, domain, o.upsample, pad, flush, design)
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
    if preserve and o.blend > 0 and not o.fillet:
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
        # Round material off where it meets the design space's boundary: a
        # smooth minimum with the envelope's own density field, which is 0.5
        # on the boundary and 1 a cell inside it. Material that runs into the
        # boundary ends there with rounded edges instead of a sawn-off face
        # with sharp ones; material lying along it keeps its full thickness.
        # (A multiplicative fade, tried first, thinned every member lying on
        # a boundary face by half a cell: the arm link's flange plates went
        # from 3.4 to 1.5 mm and failed the member check.)
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in fine.shape], indexing="ij"), -1).reshape(-1, 3)
        pts = origin + idx * step
        sd = np.min([sh.sdf(pts) for sh in design], axis=0)
        env = np.clip(0.5 - sd / (2.0 * domain.pitch), 0.0, 1.0)
        if flush:
            near = np.min([sh.sdf(pts) for sh in flush], axis=0) < 1.5 * domain.pitch
            env[near] = 1.0
        if preserve and o.fillet and o.blend > 0:
            # Not round a preserved shape either: where a boss runs the full
            # height of the design space, material beside it rounded off
            # just under the boss's crisp end face and left a lip all round
            # its rim (the bell crank's bosses, 40 mm of crease). Within
            # `blend` of preserved geometry the material runs on to the
            # boundary and ends flush with the boss's face, easing back to
            # the rounding further out.
            dp = np.min([sh.sdf(pts) for sh in preserve], axis=0)
            env = np.maximum(env, np.clip(2.0 - dp / o.blend, 0.0, 1.0))
        fine = _smooth_min(fine, env.reshape(fine.shape), 0.5)
    if preserve and o.blend > 0 and o.fillet:
        # (after the boundary rounding: a fillet made before it was rounded
        # away again where a member's top met a full-height boss, bell crank)
        # Fillet the organic body into the preserved geometry in distance
        # space: the organic surface and the preserved shapes as signed
        # distances, joined by a smooth union of radius about `blend` mm, then
        # back to a density with its 0.5 level on the blended surface. A
        # density-space smooth max (fillet=False, above) only bumps the
        # field by k/4 within a cell or two of the junction, so members still
        # met bosses in a hard crease (GRIP's wrist yoke, QA 10/28). The
        # exact union in reconstruct() then adds the preserved shapes, which
        # the fillet meets tangentially, so it leaves no crease of its own.
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in fine.shape], indexing="ij"), -1).reshape(-1, 3)
        pts = origin + idx * step
        sd = np.min([sh.sdf(pts) for sh in preserve], axis=0).reshape(fine.shape)
        k = o.blend / 1.2
        du = _smooth_union(_field_distance(fine, step, o.level), sd, k)
        # back to density, over the same width the smoothing gave the field
        width = 2.0 * step
        filleted = np.clip(o.level - du / (2.0 * width), 0.0, 1.0)
        if design:
            # after the boundary rounding, and never past the boundary itself
            filleted = np.minimum(filleted, env.reshape(fine.shape))
        fine = np.maximum(fine, filleted)
    if o.fillet and o.close and o.close > 0:
        # Closing: grow the surface out by `close`, re-distance, and shrink it
        # back. Where the body is convex or flat that returns it unchanged;
        # into a concave edge between two members it leaves a fillet of that
        # radius. Struts meeting at a sharp angle otherwise keep a crease the
        # preserve fillet never sees (GRIP's upper arm, 51 mm of it once its
        # thrust disk was no longer pinned solid).
        grown = _field_distance(fine, step, o.level) - o.close < 0
        d2 = (ndimage.distance_transform_edt(~grown) - ndimage.distance_transform_edt(grown)) * step
        d2 += np.where(grown, 0.5 * step, -0.5 * step)
        # the re-distance is exact only to the grid's cells: smooth it over a
        # cell, or its staircase lands just outside the body as ridges
        closed = ndimage.gaussian_filter(np.clip(o.level - (d2 + o.close) / (4.0 * step), 0.0, 1.0), 1.0)
        if design:
            closed = np.minimum(closed, env.reshape(fine.shape))
        fine = np.maximum(fine, closed)
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
        fade = np.clip(d / clearance, 0.0, 1.0)
        if preserve and o.fillet and o.blend > 0:
            # ...except beside preserved geometry: a pad that bears on a
            # keep-out (a motor face) ends exactly at it, and the faded
            # material round it ended a clearance short, leaving the pad's
            # end standing proud in a step all round (the motor mount's
            # pads and pilot ring). There the exact subtract does the cut.
            dp = np.min([sh.sdf(pts) for sh in preserve], axis=0)
            fade = np.maximum(fade, np.clip(2.0 - dp / o.blend, 0.0, 1.0))
        fine = fine * fade.reshape(fine.shape)
    if fine.max() < o.level:
        raise ValueError("the optimized density never reaches solid; nothing to mesh")
    verts, faces, _, _ = marching_cubes(fine, o.level, spacing=(step, step, step))
    verts = verts + origin
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _flat_faces(shape):
    """The flat faces of a preserved shape, as (point, outward normal,
    lateral) with lateral(p) the distance from p's projection to the face
    (0 on it). Box faces and cylinder end caps; anything else has none."""
    from .shapes import Box, Cylinder

    out = []
    if isinstance(shape, Cylinder):
        def lateral(p, sh=shape):
            d = p - sh.p0
            radial = np.linalg.norm(d - (d @ sh.axis)[:, None] * sh.axis, axis=1)
            return np.maximum(np.maximum(radial - sh.r, sh.r_inner - radial), 0.0)
        out += [(shape.p0, -shape.axis, lateral), (shape.p1, shape.axis, lateral)]
    elif isinstance(shape, Box):
        for a in range(3):
            others = [i for i in range(3) if i != a]
            def lateral(p, sh=shape, o=others):
                q = p[:, o]
                return np.linalg.norm(np.maximum(np.maximum(sh.lo[o] - q, q - sh.hi[o]), 0.0), axis=1)
            for sign, at in ((-1.0, shape.lo), (1.0, shape.hi)):
                n = np.zeros(3)
                n[a] = sign
                out.append((at, n, lateral))
    return out


def _snap_to_flat_faces(mesh, preserve, reach: float, spread: float):
    """Move the organic surface onto a preserved shape's flat face where it
    runs alongside it within `reach` (mm), facing the same way.

    The density field resolves a millimetre at best, so where the body runs
    on flush with a boss's end face its surface lands a fraction of a
    millimetre off the face, and the exact union leaves the boss's rim
    standing in a hairline step all round (GRIP's turret: 70 mm of crease
    round its J1 hub's mating face, and the body half a millimetre proud of
    that face where the fillet had bulged past it). On the face's own
    footprint the surface goes onto the face's plane; past its edge the move
    eases off over `spread` mm, so the body rises from the face's rim in a
    slope instead of a step."""
    v = np.asarray(mesh.vertices, float).copy()
    vn = np.asarray(mesh.vertex_normals, float)
    moved = False
    for sh in preserve:
        for at, n, lateral in _flat_faces(sh):
            s = (v - at) @ n
            # only ever out onto the plane, adding material: moving a surface
            # in onto it thins whatever stands on the face (the motor mount's
            # web over a bolt boss: SF 3.35 -> 2.73)
            cand = (s < 0) & (s > -reach) & (vn @ n > np.cos(np.radians(35.0)))
            if not cand.any():
                continue
            i = np.nonzero(cand)[0]
            # (fully onto the plane for a cell past the edge too: a vertex
            # left a hundredth of a millimetre off it is still a step)
            w = np.clip(1.0 - (lateral(v[i]) - reach) / spread, 0.0, 1.0)
            w *= np.clip(2.0 * (reach - np.abs(s[i])) / reach, 0.0, 1.0)
            v[i] -= (w * s[i])[:, None] * n
            moved = moved or bool((w > 0).any())
    if moved:
        mesh.vertices = v
    return mesh


def _bodies(man):
    return sorted(man.decompose(), key=lambda b: -b.volume())


def reconstruct(study: Study, domain: Domain, xphys: np.ndarray, opts: ReconstructOptions | None = None):
    """Returns (trimesh.Trimesh, info dict)."""
    import manifold3d as m3d

    o = opts or ReconstructOptions()
    if o.blend is None:
        import copy
        o = copy.copy(o)
        o.blend = max(3.0, 1.5 * float(study.min_member or 0.0))
    if o.close is None:
        import copy
        o = copy.copy(o)
        o.close = max(1.0, 0.5 * float(study.min_member or 0.0))
    # where the part must reach the boundary: its supports and its loads
    flush = []
    for sup in study.supports:
        try:
            flush.append(sup.shape if sup.shape is not None else study.region_shape(sup.region)[0])
        except ValueError:
            pass
    for case in study.load_cases:
        for ld in case.loads:
            try:
                flush.append(ld.shape if ld.shape is not None else study.region_shape(ld.region)[0])
            except ValueError:
                pass
    if o.fillet and o.blend > 0:
        # Material against a preserved shape runs on through the design
        # space's boundary like a flush face, so the exact trim below cuts it
        # level with the shape's own end face instead of leaving the end a
        # fraction of a millimetre proud of it (a lip round every full-height
        # boss: the bell crank's).
        flush = flush + [r.shape for r in study.preserve]
    iso = _isosurface(domain, xphys, o, [r.shape for r in study.keep_out], study.min_member,
                      flush=flush, preserve=[r.shape for r in study.preserve], design=list(study.design_space))
    info = {"isoTriangles": int(len(iso.faces))}
    if o.taubin_iterations:
        import trimesh
        trimesh.smoothing.filter_taubin(iso, lamb=0.5, nu=-0.53, iterations=o.taubin_iterations)
    if study.preserve and o.fillet and o.blend > 0:
        _snap_to_flat_faces(iso, [r.shape for r in study.preserve], domain.pitch / o.upsample, o.blend)
    if len(iso.faces) > o.max_triangles:
        # Decimate the organic surface before the booleans, never after: a
        # decimated part shaves the exact preserved faces (a bore wall moved
        # a few hundredths of a millimetre reads as preserved geometry lost).
        try:
            simplified = iso.simplify_quadric_decimation(face_count=o.max_triangles)
            if simplified.is_watertight:
                iso = simplified
        except Exception:  # noqa: BLE001 - decimation is an optimization, never fatal
            pass
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
    info["triangles"] = int(len(mesh.faces))
    info["volume"] = float(mesh.volume)
    info["watertight"] = bool(mesh.is_watertight)
    return mesh, info
