"""
From a density field to a part you can print.

1. Smooth the projected density on a 2x finer grid and take the 0.5
   isosurface (marching cubes). Upsampling first is what turns voxel steps
   into the organic surface; smoothing the field (not the mesh) keeps it
   watertight. Before the blur, along members (rods) the surface is
   averaged along the member's own axis (_axial_smooth), which takes out
   the grid's staircase on members tilted a little off the grid.
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
                 close=None, axial=True, axial_debias=None):
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
        # Average the field along each member's own axis before smoothing
        # (_axial_smooth): a member tilted a few degrees off the grid
        # otherwise keeps the grid's staircase as a row of bumps and necks.
        self.axial = axial
        # Debias that average (see _axial_smooth) so thin members keep their
        # volume; None follows resolution.keepThinMembers (off by default).
        self.axial_debias = axial_debias


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


class _GridSDF:
    """The primitives' signed distances on one fine grid, each group (the
    flush faces, preserved shapes, design space, keep-outs) evaluated once
    and kept: _isosurface asked for the same ones up to three times, and
    each was a min over a stacked (shapes, cells) array. A running minimum
    gives the same values."""

    def __init__(self, shape, origin, step):
        self.shape, self.origin, self.step = tuple(shape), origin, step
        self._pts = None
        self._min = {}

    @property
    def pts(self):
        if self._pts is None:
            idx = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), -1).reshape(-1, 3)
            self._pts = self.origin + idx * self.step
        return self._pts

    def min(self, shapes):
        key = tuple(id(s) for s in shapes)
        if key not in self._min:
            self._min[key] = _min_sdf(shapes, self.pts)
        return self._min[key]


def _min_sdf(shapes, pts):
    """min over `shapes` of their signed distance at `pts`."""
    out = None
    for sh in shapes:
        d = sh.sdf(pts)
        out = np.array(d, float) if out is None else np.minimum(out, d, out=out)
    return out


def _seat_on_flush_faces(fine, domain: Domain, upsample: int, pad: int, flush, design, grid=None):
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
    grid = grid if grid is not None else _GridSDF(shape, origin, step)
    pts = grid.pts
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
                    near = grid.min(flush) < 1.5 * domain.pitch
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


FINE_CELLS = 8e6

# _axial_smooth: the Hessian's scale and the axis tensor's spread (Gaussian
# sigmas in grid cells), the smoothing length along the axis (sigma, cells),
# the coherence range over which it fades in, and the bilateral range (cells):
# samples whose surface lies further than that from the point's own are
# another feature (a branch, a member's end), not a step of the staircase.
AXIAL_SCALE = 1.5
AXIAL_SPREAD = 1.5
AXIAL_LENGTH = 3.0
AXIAL_COHERENCE = (0.5, 0.8)
AXIAL_RANGE = 1.5


def _rod_axes(g, scale=AXIAL_SCALE, spread=AXIAL_SPREAD):
    """Where the density `g` has rods, and along which axis: the tensor
    sum of s e e^T spread over `spread` cells, with e the Hessian's flat
    direction at `scale` and s its rod strength (the smaller of the two
    strong curvatures, both negative: a ridge line, not a plate or a blob)."""
    H = np.empty(g.shape + (3, 3))
    for i in range(3):
        for j in range(i, 3):
            order = [0, 0, 0]
            order[i] += 1
            order[j] += 1
            H[..., i, j] = H[..., j, i] = ndimage.gaussian_filter(g, scale, order=order)
    lam, vec = np.linalg.eigh(H)
    del H
    o = np.argsort(np.abs(lam), axis=-1)
    lam = np.take_along_axis(lam, o, -1)
    e = np.take_along_axis(vec, o[..., None, :], -1)[..., :, 0]
    del vec
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    s = np.where((l2 < 0) & (l3 < 0), np.clip(-l2 - np.abs(l1), 0.0, None), 0.0)
    del lam
    T = {}
    for i in range(3):
        for j in range(i, 3):
            T[i, j] = ndimage.gaussian_filter(s * e[..., i] * e[..., j], spread)
    return T


def _axial_smooth(fine, coarse, upsample, length=AXIAL_LENGTH, coherence=AXIAL_COHERENCE, rng=AXIAL_RANGE,
                  reach=2.5, samples=21, debias=False):
    """Average the surface's position along the local member axis.

    A member tilted slightly off the grid is voxelized as a staircase: its
    top and bottom step a cell at a time, a cell / tan(tilt) apart along it,
    so its section swings by about a cell, out of phase top and bottom. The
    isotropic upsample and blur below only round each step off, and the part
    came out with a row of bumps and necks along every long stem (GRIP's
    upper arm: +-0.6 mm on an 11 mm stem every 19 mm, about 9% less bending
    stiffness than the same mass at constant section, and the stress
    concentrated at every neck). The steps carry no design intent: the
    optimizer's filter blurs over more than a cell, and its sensitivities saw
    the staircase only as the grid's error.

    Where the material round a point is a rod with one clear axis
    (_rod_axes: coherent, and strong against the part's own rods), the
    field's signed distance there is replaced by its Gaussian average along
    that axis over `length` cells, and the field's 0.5 level moved to the
    averaged surface (averaging the density instead shrank thin members,
    whose upsampled density never reaches 1). The average is bilateral:
    samples whose distance differs from the point's by much more than a
    step (`rng` cells) count less, so a branch leaving the member, or the
    member's own end, is not smeared along it. Plates, blobs and junctions
    of several directions keep the field as it was; only voxels within a
    cell and a half of the surface change.

    Averaging a round member's distance along a line that is not exactly
    its axis is biased outward: the distance to a convex surface is convex,
    so every sample off the axis reads further out than the point (about
    offset^2 / 2r, so the thinner the member the worse), and the member
    shrank. Thin members lost the most: a graded rod two cells across lost
    15% of its volume, 1.5 cells 38%, and the MJF wrist link (sub-cell
    struts on 2.44 mm cells) meshed 2-4% light at every density. The bias
    is a steady offset where the staircase is a ripple, so the average is
    taken twice and extrapolated back (2 A d - A A d, as Taubin's inflate
    step undoes Laplacian shrinkage): the bias cancels to first order and
    the ripple, already nearly gone after one pass, stays small. The rods
    keep their volume to 0.5-2%.

    It is opt-in (debias, on with resolution.keepThinMembers), like the
    volume-keeping surface smoothing, and for the same reason: the shrinking
    one-pass average reads better on the evals. Debiased on everywhere, the
    full tier went mbb-beam 13.8 -> 15.0 g at target, motor-mount 19.8 ->
    22.1, hook 23.9 -> 24.8, wrist yoke 16.7 -> 17.2: those parts verify
    lighter with their thin members slightly shrunk. The wrist link went
    34.2 -> 33.2 (31.4 with the pass off), but its verified SF swings 2-3x
    between near-identical meshes of the same density (its peak sits on
    sub-cell struts), so that case's 31 vs 34 g is mostly search noise."""
    g = np.asarray(coarse, float)
    T = _rod_axes(g)
    size = 2 * int(np.ceil(1.5 * upsample)) + 1
    band = (ndimage.maximum_filter(fine, size=size) > 0.02) & (ndimage.minimum_filter(fine, size=size) < 0.98)
    fi = np.argwhere(band)
    del band
    if not len(fi):
        return fine
    # fine index i sits at coarse index (i + 0.5) / upsample - 0.5 (see _isosurface)
    cc = ((fi + 0.5) / upsample - 0.5).T
    t = np.empty((len(fi), 3, 3))
    for (i, j), a in T.items():
        t[:, i, j] = t[:, j, i] = ndimage.map_coordinates(a, cc, order=1, mode="nearest")
    del cc
    mu, vec = np.linalg.eigh(t)
    del t
    top = np.maximum(mu[:, 2], 1e-30)
    coh = (mu[:, 2] - mu[:, 1]) / top
    # strength against the part's own rods, so a design's scale does not matter
    ref = np.percentile(top, 90)
    lo, hi = coherence
    w = np.clip((coh - lo) / (hi - lo), 0.0, 1.0) * np.clip(2.0 * top / max(ref, 1e-30), 0.0, 1.0)
    keep = w > 0.0
    fi, axis, w = fi[keep], vec[keep, :, 2].T, w[keep]
    if not len(fi):
        return fine
    z = np.linspace(-reach, reach, samples)
    gz = np.exp(-0.5 * z * z)
    base = fi.T.astype(float)
    dist = _field_distance(fine, 1.0, 0.5)        # in fine cells
    i, j, k = fi.T
    d0 = dist[i, j, k]
    r2 = (rng * upsample) ** 2

    def along(field, ref):
        num = np.zeros(len(fi))
        den = np.zeros(len(fi))
        for zk, gk in zip(z, gz):
            dk = ndimage.map_coordinates(field, base + (w * (zk * length * upsample)) * axis, order=1, mode="nearest")
            wk = gk * np.exp(-0.5 * (dk - ref) ** 2 / r2)
            num += wk * dk
            den += wk
        return num / den

    avg = along(dist, d0)
    if debias:
        A = dist.copy()
        A[i, j, k] = avg
        avg2 = along(A, avg)
        del A
        avg = 2.0 * avg - avg2
    # move the field's 0.5 level by the change in distance, keeping its ramp
    grad = np.linalg.norm(np.stack(np.gradient(fine)), axis=0)[i, j, k]
    out = fine.copy()
    out[i, j, k] = np.clip(fine[i, j, k] + (d0 - avg) * np.maximum(grad, 0.25), 0.0, 1.0)
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
        near = (_min_sdf(flush, centers) < 1.5 * domain.pitch).reshape(padded.shape)
        padded = np.where(near, edge, padded)
    grid = padded
    # Cubic upsampling: linear interpolation of a voxel field leaves the
    # staircase in the isosurface as faceted terraces.
    fine = np.clip(ndimage.zoom(grid, o.upsample, order=3, mode="nearest", grid_mode=True), 0.0, 1.0)
    # the fine grid's cell centers (see `origin` below), and the shapes'
    # distances on them, computed once for all the steps that follow
    sdfs = _GridSDF(fine.shape, domain.origin - pad * domain.pitch + 0.5 * (domain.pitch / o.upsample),
                    domain.pitch / o.upsample)
    if flush and design and o.seat:
        fine = _seat_on_flush_faces(fine, domain, o.upsample, pad, flush, design, grid=sdfs)
    if o.axial:
        fine = _axial_smooth(fine, grid, o.upsample, debias=bool(o.axial_debias))
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
        sd = sdfs.min(preserve).reshape(fine.shape)
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
        sd = sdfs.min(design)
        env = np.clip(0.5 - sd / (2.0 * domain.pitch), 0.0, 1.0)
        if flush:
            near = sdfs.min(flush) < 1.5 * domain.pitch
            env[near] = 1.0
        if preserve and o.fillet and o.blend > 0:
            # Not round a preserved shape either: where a boss runs the full
            # height of the design space, material beside it rounded off
            # just under the boss's crisp end face and left a lip all round
            # its rim (the bell crank's bosses, 40 mm of crease). Within
            # `blend` of preserved geometry the material runs on to the
            # boundary and ends flush with the boss's face, easing back to
            # the rounding further out.
            dp = sdfs.min(preserve)
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
        sd = sdfs.min(preserve).reshape(fine.shape)
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
        clearance = 0.5 * domain.pitch
        d = sdfs.min(keep_out)
        fade = np.clip(d / clearance, 0.0, 1.0)
        if preserve and o.fillet and o.blend > 0:
            # ...except beside preserved geometry: a pad that bears on a
            # keep-out (a motor face) ends exactly at it, and the faded
            # material round it ended a clearance short, leaving the pad's
            # end standing proud in a step all round (the motor mount's
            # pads and pilot ring). There the exact subtract does the cut.
            dp = sdfs.min(preserve)
            fade = np.maximum(fade, np.clip(2.0 - dp / o.blend, 0.0, 1.0))
        fine = fine * fade.reshape(fine.shape)
    if fine.max() < o.level:
        raise ValueError("the optimized density never reaches solid; nothing to mesh")
    # One more layer of void all round: marching cubes leaves the surface open
    # wherever the field is solid on the array's edge, and smoothing, the
    # fillet and the closing above can carry it there through the single
    # void layer of padding. Unseen below ~100k cells; at 845k (0.53 mm
    # cells) the arm link's flush faces came out with 1700 open edges and
    # manifold3d refused the mesh. Where the edge was void this changes
    # nothing.
    fine = np.pad(fine, 1)
    origin = origin - step
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


def _attach_plan(mesh, study, organic):
    """Remember how the part was built, exact shapes and all, for STEP
    export (step.py): ((organic & envelope) | preserved) - keep-outs."""
    from .shapes import Union
    from .sharp import Exact, Op, Organic
    from .step import attach_plan

    def one(shapes):
        return shapes[0] if len(shapes) == 1 else Union(shapes)

    org = Organic(None, ["organic"], mesh=trimesh_from_manifold(organic))   # a trimesh: parts cross processes
    env = list(study.design_space) + [r.shape for r in study.preserve]
    p = Op("intersect", [org, Exact(one(env), "envelope")])
    if study.preserve:
        pres = one([r.shape for r in study.preserve])
        p = Op("union", [p, Exact(pres, "preserve")])
        p.keep_with = pres
    if study.keep_out:
        keep = getattr(p, "keep_with", None)
        p = Op("subtract", [p, Exact(one([r.shape for r in study.keep_out]), "keep-out")])
        if keep is not None:
            p.keep_with = keep
    attach_plan(mesh, p)


def _drop_slivers(mesh):
    """Drop zero-volume shells (a pair of coincident triangles facing apart)
    that a boolean can leave on a flat face. manifold3d keeps them on vertices
    of their own, so the mesh in memory is closed, but any STL reader welds
    them into the part, where they read as non-manifold edges (the wrist link
    in docs/tutorials/agent-mcp was not watertight once saved). Returns the
    welded mesh without them, or `mesh` itself when there are none."""
    import trimesh

    m = mesh.copy()
    m.merge_vertices()
    if not len(m.faces):
        return mesh
    lab = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(m.faces))
    tri = m.vertices[m.faces]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    per = np.bincount(lab, weights=vol)
    sliver = np.abs(per) < 1e-6 * max(abs(per.sum()), 1e-9)
    if not sliver.any():
        return mesh
    # the welded copy: in manifold3d's own topology the part's faces either
    # side of a sliver are joined through it, so dropping it unwelded opens holes
    m.update_faces(~sliver[lab])
    m.remove_unreferenced_vertices()
    return m


def _smooth_surface(iso, domain: Domain, xphys: np.ndarray, iterations: int, keep_thin=False, lamb=0.5,
                    nu=0.53):
    """Smooth the isosurface: alternating steps v += f (L v - v), the first
    shrinking (lamb), the second shrinking on thick members and inflating
    (Taubin's) on members about two cells thick or less.

    This code long passed nu = -0.53 to trimesh.smoothing.filter_taubin,
    which subtracts its nu, so both steps shrank: plain Laplacian smoothing.
    On thick members that rounds the staircase off in a way the evals are
    tuned to: true Taubin everywhere read worse (mass at target SF +3% to
    +10% on six full-tier cases, the wrist yoke failing on creases). But it
    is curvature flow, and members one or two cells thick shrink until they
    nearly part: the GRIP wrist yoke's stress-constrained design at 15k
    cells lost 20% of its mesh volume (verified peak 33 -> 176 MPa) and the
    camera boom's two-cell walls came out one cell thick (16x softer).
    Capping each vertex's shift instead (0.1 / 0.25 cell) only halved the
    damage. So it is opt-in (resolution.keepThinMembers): the member's
    half-thickness at a vertex is the largest distance to void within two
    cells; Taubin at 1.5 cells or less, as before at 2.5 or more, blended
    between. That keeps the yoke at SF 1.22 and the motor mount's 15k
    design at 1.42 (0.95 before); a narrower window (1.0-1.5) kept only
    0.43 on the yoke. Off, every vertex smooths as before."""
    import trimesh

    w = 1.0
    if keep_thin:
        grid = domain.full_grid(np.asarray(xphys, float)) > 0.5
        half = ndimage.maximum_filter(ndimage.distance_transform_edt(grid), size=5)
        ijk = np.floor((np.asarray(iso.vertices) - domain.origin) / domain.pitch).astype(int)
        ijk = np.clip(ijk, 0, np.array(grid.shape) - 1)
        w = np.clip((half[ijk[:, 0], ijk[:, 1], ijk[:, 2]] - THIN[0]) / (THIN[1] - THIN[0]), 0.0, 1.0)[:, None]
    second = nu * (2.0 * w - 1.0)          # -nu thin (Taubin), +nu thick (as before)
    lap = trimesh.smoothing.laplacian_calculation(iso)
    v = np.asarray(iso.vertices, float).copy()
    for k in range(iterations):
        v = v + (lamb if k % 2 == 0 else second) * (lap @ v - v)
    iso.vertices = v


THIN = (1.5, 2.5)    # member half-thickness (cells): Taubin below the first, shrinking above the second


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
    if o.axial_debias is None:
        import copy
        o = copy.copy(o)
        o.axial_debias = bool((getattr(study, "raw", None) or {}).get("resolution", {}).get("keepThinMembers", False))
    if o.axial and getattr(study, "access", None) is not None:
        # A cast or milled part's field is already shaped for its draw or
        # cutter directions, and averaging it along a member's tilted axis
        # leans those walls into undercuts: the cast arm link went from
        # 141 to 630 mm^3 of undercut (limit 150) with it on.
        import copy
        o = copy.copy(o)
        o.axial = False
    # Upsampling refines a coarse grid's staircase; a fine grid has none to
    # speak of, and 3x on 845k cells meant a 23M-cell field and 330 s of
    # meshing for 2M triangles. Cap the refined field near FINE_CELLS (eval
    # grids, all under ~150k box cells, keep their 3x).
    box = float(np.prod(domain.dims))
    cap = max(1, int((FINE_CELLS / box) ** (1.0 / 3.0)))
    if o.upsample > cap:
        import copy
        o = copy.copy(o)
        o.upsample = cap
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
        keep_thin = bool((getattr(study, "raw", None) or {}).get("resolution", {}).get("keepThinMembers", False))
        _smooth_surface(iso, domain, xphys, o.taubin_iterations, keep_thin)
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
    # flush faces wherever the design reaches its boundary. Circles fine
    # enough to be 0.002 mm true (sharp.exact_manifold): at a fixed 96
    # segments a 48 mm sleeve's facets were 0.026 mm off its cylinder.
    from .sharp import exact_manifold

    envelope = m3d.Manifold.batch_boolean(
        [exact_manifold(sh) for sh in study.design_space] + [exact_manifold(r.shape) for r in study.preserve],
        m3d.OpType.Add)
    organic_raw = organic
    organic = organic ^ envelope
    part = organic
    if study.preserve:
        pres = m3d.Manifold.batch_boolean([exact_manifold(r.shape) for r in study.preserve], m3d.OpType.Add)
        part = part + pres
    if study.keep_out:
        ko = m3d.Manifold.batch_boolean([exact_manifold(r.shape) for r in study.keep_out], m3d.OpType.Add)
        part = part - ko

    # decompose() lists an enclosed cavity as a body of negative volume: it
    # is the part's (a sealed plenum's), not a fragment
    bodies = [b for b in _bodies(part) if b.volume() > 0]
    if not bodies:
        raise ValueError("reconstruction produced an empty part")
    # A body is part of the result if it is the main one or it carries
    # preserved geometry; anything else is a fragment the subtract cut loose.
    kept = [bodies[0]]
    loose = []
    dropped = 0.0
    if study.preserve:
        pres_man = m3d.Manifold.batch_boolean([exact_manifold(r.shape) for r in study.preserve], m3d.OpType.Add)
    for b in bodies[1:]:
        if study.preserve and (b ^ pres_man).volume() > 1e-6:
            kept.append(b)
        else:
            loose.append(b)
            dropped += b.volume()
    if loose:
        # cut the fragments away rather than rebuilding from the kept
        # bodies, which would fill every enclosed cavity
        part = part - (m3d.Manifold.batch_boolean(loose, m3d.OpType.Add) if len(loose) > 1 else loose[0])
    info["bodies"] = len(kept)
    info["droppedFragmentVolume"] = float(dropped)

    mesh = _drop_slivers(trimesh_from_manifold(part))
    _attach_plan(mesh, study, organic_raw)
    info["triangles"] = int(len(mesh.faces))
    info["volume"] = float(mesh.volume)
    info["watertight"] = bool(mesh.is_watertight)
    return mesh, info
