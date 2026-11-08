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


SWEEPS = 3            # face-neighbour passes of _surface_distance's closest-point sweep
MEMBER_FLOOR = 0.67   # share of minMember: thinner members go, thicker ones are built out to minMember
OPEN_TOL = 0.25       # cells: how far past the part's surface _open and _thicken read it


def _surface_distance(fine, step, level=0.5, reach=None):
    """Signed distance (mm, negative inside) to the `level` surface, to a
    fraction of a cell everywhere: the field's own gradient within a cell or
    so of the surface, and further away the distance to the nearest of those
    cells plus that cell's own distance. (_field_distance's transforms away
    from the surface count whole cells: half a cell of error in a member's
    depth. Its gradient band also reached the ridge of a member two cells
    across, where the gradient vanishes and the linear reading ran deep.)"""
    inside = fine >= level
    grad = np.stack(np.gradient(fine, step))
    g = np.linalg.norm(grad, axis=0)
    lin = (level - fine) / np.maximum(g, 1e-6)
    band = (np.abs(fine - level) < 0.3) & (g > 1e-6) & (np.abs(lin) < 0.75 * step)
    if not band.any():
        return _field_distance(fine, step, level)
    _, idx = ndimage.distance_transform_edt(~band, return_indices=True)
    # the surface point each band cell sees along its gradient, and the
    # distance to the one nearest
    shape = fine.shape
    flat = np.ravel_multi_index(tuple(idx), shape).ravel()
    del idx
    # each band cell's surface point, along its gradient
    bc = np.flatnonzero(band.ravel())
    cs = np.stack(np.unravel_index(bc, shape), axis=1) * step
    gb = np.stack([grad[a].ravel()[bc] for a in range(3)], axis=1)
    cs += (lin.ravel()[bc] / np.maximum(g.ravel()[bc], 1e-6))[:, None] * gb
    surf = np.zeros((fine.size, 3))
    surf[bc] = cs
    del cs, gb, grad
    surf = surf.astype(np.float32)
    sp = surf[flat]                 # the nearest band cell's surface point
    del surf, flat
    pts = (np.stack(np.unravel_index(np.arange(fine.size), shape), axis=1) * step).astype(np.float32)
    best = ((sp - pts) ** 2).sum(axis=1)
    # It is only near the nearest surface point (it always reads too far):
    # take the nearest of the surface points the cell's neighbours see too,
    # a closest-point sweep. (Any surface point is a fair candidate, so a
    # neighbour index that wraps round a row does no harm.) Only within
    # `reach` mm of the surface, if given.
    sel = np.flatnonzero(~band.ravel() & (best < reach * reach)) if reach else np.flatnonzero(~band.ravel())
    spl, ptl, bl = sp[sel], pts[sel], best[sel]
    del pts
    strides = np.array([shape[1] * shape[2], shape[2], 1])
    offs = [s * int(st) for st in strides for s in (-1, 1)]
    for _ in range(SWEEPS):
        for off in offs:
            cand = sp[np.clip(sel + off, 0, fine.size - 1)]
            dc = ((cand - ptl) ** 2).sum(axis=1)
            better = dc < bl
            bl[better] = dc[better]
            spl[better] = cand[better]
        sp[sel] = spl
    best[sel] = bl
    dist = np.sqrt(best).reshape(shape)
    return np.where(band, lin, np.where(inside, -dist, dist))


def _hold_members(fine, step, min_member, level=0.5, floor=MEMBER_FLOOR, tol=OPEN_TOL):
    """The field with every member at least `min_member` across, or gone.

    Each point's member half-thickness h is the deepest point of the part
    within reach of it (read on the grid's cells, so a member whose axis
    falls between cells reads up to about half a cell thin, which only
    builds it out further). Where h is under `floor` x half the minimum the
    member goes, by an opening in distance space: the part's points at
    least that deep, grown back out by as much, which leaves the rest
    exactly as it was. Where h is between that and half the minimum the
    surface moves out by the difference, to the minimum.

    The voxel-ball grey opening this replaces had a ball no wider than its
    whole cells reach (1.15 mm across a rod, 0.81 across a plate at minMember
    3 on 0.81 mm cells), so it passed 2.3 mm rods and 1.6 mm plates, and a
    strut nearer the minimum came out necked to 1.3-2.2 mm instead of whole.
    An exact opening by the minimum alone met struts that wander round it
    along their length (a tilted strut's staircase on the analysis grid) and
    kept them as a string of beads, necked between; building out first
    leaves it only the struts well under the minimum, which go whole."""
    R = 0.5 * float(min_member)
    width = 2.0 * step
    d = _surface_distance(fine, step, level, reach=2.0 * R + step)
    k = int(np.ceil(R / step)) + 1
    h = ndimage.maximum_filter(-d, size=2 * k + 1)
    target = R + tol * step
    need = np.where((h >= floor * R) & (h < target), target - h, 0.0)
    if need.any():
        fine = np.maximum(fine, np.where(need > 0, np.clip(level - (d - need) / (2.0 * width), 0.0, 1.0), 0.0))
        d = _surface_distance(fine, step, level, reach=2.0 * R + step)
    r = floor * R
    e = d + r                       # distance to the eroded surface
    core = e <= 0.0
    if not core.any():
        return np.zeros_like(fine)
    far, idx = ndimage.distance_transform_edt(~core, return_indices=True)
    # from outside the core: to its nearest cell, plus that cell's own
    # (sub-cell) distance to the core's surface
    de = np.where(core, e, far * step + e[tuple(idx)])
    del idx, far
    # read `tol` cells past the opening's surface, so it never shaves a
    # member it keeps
    return np.minimum(fine, np.clip(level - (de - r - tol * step) / (2.0 * width), 0.0, 1.0))


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
            self._min[key] = _shared_min(tuple(shapes), self.shape, self.origin, self.step, lambda: self.pts)
        return self._min[key]


# The same primitives on the same fine grid, across outcomes: every
# outcome of a search meshes on its study's one design grid, so the
# preserved shapes', keep-outs' and design space's distances repeat. They
# were 6 s of each outcome's meshing on the wrist yoke (Cylinder.sdf on
# a few million points). Kept by the shapes themselves (held here, so
# their ids stay theirs) and the grid; least recently used first out past
# _SDF_CACHE_BYTES. The values are the ones _min_sdf gives.
_SDF_CACHE: "OrderedDict" = None
_SDF_CACHE_BYTES = 200e6
_SDF_LOCK = None


def _shared_min(shapes, shape, origin, step, pts):
    global _SDF_CACHE, _SDF_LOCK
    import threading
    from collections import OrderedDict

    if _SDF_CACHE is None:
        _SDF_CACHE, _SDF_LOCK = OrderedDict(), threading.Lock()
    key = (tuple(id(s) for s in shapes), tuple(shape), tuple(np.asarray(origin, float).tolist()), float(step))
    with _SDF_LOCK:
        hit = _SDF_CACHE.get(key)
        if hit is not None and all(a is b for a, b in zip(hit[0], shapes)):
            _SDF_CACHE.move_to_end(key)
            return hit[1]
    val = _min_sdf(shapes, pts())
    val.setflags(write=False)
    with _SDF_LOCK:
        _SDF_CACHE[key] = (shapes, val)
        total = sum(v[1].nbytes for v in _SDF_CACHE.values())
        while total > _SDF_CACHE_BYTES and len(_SDF_CACHE) > 1:
            _, (_, old) = _SDF_CACHE.popitem(last=False)
            total -= old.nbytes
    return val


def _min_sdf(shapes, pts):
    """min over `shapes` of their signed distance at `pts`."""
    out = None
    for sh in shapes:
        d = sh.sdf(pts)
        out = np.array(d, float) if out is None else np.minimum(out, d, out=out)
    return out



def _fill_unmachinable(part, study, domain, envelope):
    """Fill the void a milled part's cutter cannot reach, on the design grid.

    The optimizer's field honours milling access, but meshing (upsampling,
    smoothing, the flush trims) moves the surface by a fraction of a cell,
    and every such step that leans a wall out or tightens a corner under the
    tool radius is a pocket the shop cannot cut: a plate milled from +z came
    out with 760 mm^3 unreachable (limit ~100), most of it along its bottom
    edges. Filling those cells (the morphological closing the cutter itself
    would leave) makes the delivered part machinable as measured by
    `manufacture.report`, which voxelizes the same grid. Returns (part,
    filled volume in mm^3); only ever adds material, inside the design space.
    """
    import manifold3d as m3d

    from .manufacture import _reach_by_direction
    from .verify import voxelize

    acc = study.access
    pitch = float(domain.pitch)
    lo = np.asarray(domain.origin, float)
    hi = lo + np.asarray(domain.dims) * pitch
    mesh = trimesh_from_manifold(part)
    solid, _ = voxelize(mesh, pitch, box=(lo, hi))
    cut = np.zeros_like(solid)
    for c in _reach_by_direction(solid, acc, pitch, acc.tool_radius or 0.0):
        cut |= c
    b = np.asarray(mesh.bounds, float)
    ctr = [lo[i] + (np.arange(solid.shape[i]) + 0.5) * pitch for i in range(3)]
    inb = [(c > b[0][i]) & (c < b[1][i]) for i, c in enumerate(ctr)]
    stock = inb[0][:, None, None] & inb[1][None, :, None] & inb[2][None, None, :]
    fill = ~solid & ~cut & stock
    if not fill.any():
        return part, 0.0
    pts = np.stack(np.meshgrid(*ctr, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(solid.size, bool)
    for sh in study.design_space:
        inside |= sh.contains(pts)
    fill &= inside.reshape(solid.shape)
    if study.keep_out:
        keep = np.zeros(solid.size, bool)
        for r in study.keep_out:
            keep |= r.shape.contains(pts)
        fill &= ~ndimage.binary_dilation(keep.reshape(solid.shape), iterations=1)
    n = int(fill.sum())
    if n == 0:
        return part, 0.0
    # one box per run of filled cells along x, a hair oversize so the
    # boxes fuse with each other and the part
    eps = 1e-3 * pitch
    boxes = []
    for j, k in zip(*np.nonzero(fill.any(axis=0))):
        col = np.concatenate(([False], fill[:, j, k], [False]))
        edges = np.flatnonzero(col[1:] != col[:-1])
        for i0, i1 in zip(edges[::2], edges[1::2]):
            a = lo + np.array([i0, j, k], float) * pitch - eps
            size = np.array([(i1 - i0) * pitch, pitch, pitch]) + 2 * eps
            boxes.append(m3d.Manifold.cube(size.tolist()).translate(a.tolist()))
    filler = m3d.Manifold.batch_boolean(boxes, m3d.OpType.Add) ^ envelope
    out = m3d.Manifold.batch_boolean([part, filler], m3d.OpType.Add)
    # A box meets the organic surface a fraction of a cell off, which can
    # seal a sliver of air inside (decompose lists it as a body of negative
    # volume) and leave zero-volume scraps. A milled part has no enclosed
    # void, so rebuild it from its solid bodies: that fills them.
    bodies = out.decompose()
    total = max(b.volume() for b in bodies)
    solid_bodies = [b for b in bodies if b.volume() > 1e-3 * total]
    if len(solid_bodies) != len(bodies):
        out = (m3d.Manifold.batch_boolean(solid_bodies, m3d.OpType.Add)
               if len(solid_bodies) > 1 else solid_bodies[0])
    return out, float(n * pitch ** 3)

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
    came out with a row of bumps and necks along every long stem (the robot arm's
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
                flush=(), preserve=(), design=(), draft=None):
    import trimesh
    from skimage.measure import marching_cubes

    grid = domain.full_grid(xphys)
    pad = 3
    # Pad with void, so material that reaches the design space's boundary
    # rounds off just inside it: the exact intersection with the design
    # space then has nothing to cut, and no flat sawn-off faces appear where
    # the box happened to end (the robot arm's wrist yoke had several). Only where a
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
        # Every member at the minimum or gone (_hold_members): a member well
        # under it goes, one a little under is built out to it. Smoothing a
        # member that ends inside the design space tapers it to a knife edge,
        # and that sliver is both unprintable and a stress raiser that grew
        # with every refinement of the verification grid (14 -> 58 MPa on the
        # arm link at a flange tip).
        step_ = domain.pitch / o.upsample
        if 0.5 * float(min_member) >= step_:
            opened = _hold_members(fine, step_, float(min_member), o.level)
            fine = _keep_joints(fine, opened, step_, float(min_member), o.level)
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
        rounded = _smooth_min(fine, env.reshape(fine.shape), 0.5)
        # (it thins a member lying along the boundary by about half a cell:
        # where one so thinned was all that joined two bodies, it stays)
        fine = _keep_joints(fine, rounded, step, float(min_member), o.level) if min_member else rounded
    if preserve and o.blend > 0 and o.fillet:
        # (after the boundary rounding: a fillet made before it was rounded
        # away again where a member's top met a full-height boss, bell crank)
        # Fillet the organic body into the preserved geometry in distance
        # space: the organic surface and the preserved shapes as signed
        # distances, joined by a smooth union of radius about `blend` mm, then
        # back to a density with its 0.5 level on the blended surface. A
        # density-space smooth max (fillet=False, above) only bumps the
        # field by k/4 within a cell or two of the junction, so members still
        # met bosses in a hard crease (the robot arm's wrist yoke, QA 10/28). The
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
        # preserve fillet never sees (the robot arm's upper arm, 51 mm of it once its
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
    parting = None
    if draft is not None:
        # Draft for a cast or molded part (manufacture.drafted_distance):
        # every wall leans in by the draft angle going away from the
        # parting plane, so each mold half pulls clear. Done on the smooth
        # field, after every rounding and fillet, so the walls the mesh
        # gets are the drafted ones; it only adds material. Keep-outs and
        # preserved shapes stay exact (the booleans below): bores and
        # bosses are machined after casting or drafted in CAD.
        from .manufacture import DRAFT_MARGIN, DRAFT_MARGIN_STEP, drafted_distance, parting_index

        a, sgn, sides, angle, part_at = draft
        inside = fine >= o.level
        if sides == 2:
            if part_at is None:
                p_idx = parting_index(inside, a)
            else:
                p_idx = int(round((part_at - origin[a]) / step))
            parting = float(origin[a] + p_idx * step)
        else:
            p_idx = None
        dist = _field_distance(fine, step, o.level)
        # built DRAFT_MARGIN steeper than asked: meshing reads a wall's
        # draft up to about half a degree shallow (manufacture.DRAFT_MARGIN),
        # more on coarse cells
        angle = angle + DRAFT_MARGIN * max(1.0, step / DRAFT_MARGIN_STEP)
        dd = drafted_distance(dist, a, sgn, sides, p_idx, step, angle)
        width = 2.0 * step
        fine = np.maximum(fine, np.clip(o.level - dd / (2.0 * width), 0.0, 1.0))
        if design:
            # The design space is a hard limit, so where the part runs
            # along its walls the draft leans in from them instead (the
            # space's largest drafted subset, which only removes), except
            # at the faces the part is held and loaded at, which stay flush.
            env_d = drafted_distance(sdfs.min(design).reshape(fine.shape), a, sgn, sides, p_idx, step, angle,
                                     grow=False)
            # (only at the faces it is held and loaded at: material running
            # on past a preserved boss to the space's wall, as the fillet
            # lets it, came out a sawn-off face with no draft at all, 160
            # mm^2 of the coarse cast arm link's 330 under draft)
            held = [f for f in flush if not any(f is p for p in preserve)]
            if held:
                near = (sdfs.min(held) < 1.5 * domain.pitch).reshape(fine.shape)
                env_d = np.where(near, -np.inf, env_d)
            fine = np.minimum(fine, np.clip(o.level - env_d / (2.0 * width), 0.0, 1.0))
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
    if parting is not None:
        mesh.metadata["parting"] = parting
    if min_member and parting is None:
        # Each vertex's member half-thickness (mm) on this field, before the
        # mesh is smoothed: the deepest point inside within reach of it.
        # _smooth_surface holds members to the minimum with it.
        reach = int(np.ceil(0.5 * float(min_member) / step)) + 1
        depth = np.clip(-_field_distance(fine, step, o.level), 0.0, None)
        half = ndimage.maximum_filter(depth, size=2 * reach + 1)
        ijk = np.rint((np.asarray(mesh.vertices) - origin) / step).astype(int)
        ijk = np.clip(ijk, 0, np.array(fine.shape) - 1)
        mesh.metadata["halfThickness"] = half[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
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
    standing in a hairline step all round (the robot arm's turret: 70 mm of crease
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


PINCH_R = 0.08


def _fill_pinches(part, rounds: int = 6, clip=None, keep_out=None, hold=None):
    """Join bodies that touch along an edge or at a point (balls PINCH_R, then 1.5x larger each round a
    contact is left). manifold3d keeps
    such a contact on separate vertices, so the part is closed in memory,
    but an STL has no topology and every reader welds the two by position:
    a non-manifold edge, and a slicer or printability check reads the file
    as not watertight (the touch microphone's pad: 26 edges where the exact
    skin met the optimized interior along a line). A ball of PINCH_R mm at
    each such edge unites the two into a neck. `clip` (a manifold) bounds
    the balls: the envelope less the keep-outs, so a neck never puts
    material in a keep-out or proud of a preserved skin. A contact on a
    keep-out's face (`keep_out`) cannot be filled there, so it is opened
    instead: the ball is cut away, all but what `hold` (the preserved
    shapes) keeps. Returns (part, count)."""
    import manifold3d as m3d

    total = 0
    for k in range(rounds):
        # a ball can leave a contact of its own: each round a size larger
        r = PINCH_R * 1.5 ** k
        m = trimesh_from_manifold(part)
        # as an STL stores it: float32, then welded by position
        m.vertices = m.vertices.astype(np.float32).astype(float)
        m.merge_vertices()
        e = m.edges_sorted
        if not len(e):
            break
        u, inv, c = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        bad = u[(c > 2) | (c == 1)]   # kissing edges, and slits a weld opens
        if not len(bad):
            break
        mids = m.vertices[bad].mean(axis=1)
        # one ball per cluster of pinched edges a ball apart
        keep = []
        for q in mids:
            if not keep or np.min(np.linalg.norm(np.asarray(keep) - q, axis=1)) > r:
                keep.append(q)
        balls = [m3d.Manifold.sphere(r, 12).translate(tuple(map(float, q))) for q in keep]
        if keep_out is not None:
            at_ko = [b for b in balls if (b ^ keep_out).volume() > 1e-9]
            balls = [b for b in balls if (b ^ keep_out).volume() <= 1e-9]
            if at_ko:
                cut = m3d.Manifold.batch_boolean(at_ko, m3d.OpType.Add)
                part = part - (cut - hold if hold is not None else cut)
        if clip is not None and balls:
            balls = [m3d.Manifold.batch_boolean(balls, m3d.OpType.Add) ^ clip]
        part = m3d.Manifold.batch_boolean([part] + balls, m3d.OpType.Add)
        total += len(keep)
    return part, total


def _stl_watertight(mesh) -> bool:
    """Watertight as an STL reader sees it: float32 vertices welded by
    position, every edge on exactly two faces."""
    m = mesh.copy()
    m.vertices = m.vertices.astype(np.float32).astype(float)
    m.merge_vertices()
    if not len(m.faces):
        return False
    _, c = np.unique(m.edges_sorted, axis=0, return_counts=True)
    return bool((c == 2).all())


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
                    nu=0.53, min_member=None):
    """Smooth the isosurface: alternating steps v += f (L v - v), the first
    shrinking (lamb), the second shrinking on thick members and inflating
    (Taubin's) on members about two cells thick or less.

    This code long passed nu = -0.53 to trimesh.smoothing.filter_taubin,
    which subtracts its nu, so both steps shrank: plain Laplacian smoothing.
    On thick members that rounds the staircase off in a way the evals are
    tuned to: true Taubin everywhere read worse (mass at target SF +3% to
    +10% on six full-tier cases, the wrist yoke failing on creases). But it
    is curvature flow, and members one or two cells thick shrink until they
    nearly part: the robot-arm wrist yoke's stress-constrained design at 15k
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
    v0 = np.asarray(iso.vertices, float).copy()
    v = v0.copy()
    for k in range(iterations):
        v = v + (lamb if k % 2 == 0 else second) * (lap @ v - v)
    half = iso.metadata.get("halfThickness") if min_member else None
    iso.vertices = v
    if half is not None and len(half) == len(v):
        # Never smooth a member below the minimum: shrinking smoothing takes
        # 1.2-1.7 mm off a 3-4 mm strut's diameter (a rod the opening in
        # _isosurface left 3.7 mm across came out 2.0, one left 2.4 came
        # out 0.85), so a re-mesh of the same density grew slivers a third
        # of minMember that set the part's safety factor (the robot arm's wrist link,
        # 0.95 mm against a 3 mm minimum, SF 0.66 where its neighbours read
        # 1.6-2.2). Where the smoothing's net inward shift (averaged over a
        # few rings, so the staircase still comes off) is more than the
        # member's half-thickness over half the minimum allows, the vertex
        # goes back out by the difference: never past where it started, and
        # members thick enough to spare it smooth exactly as before.
        # (Letting it go further out, to make members the opening passed at
        # 2.4 mm a full 3, changed the wrist link's readings by under 2%.)
        allow = np.maximum(np.asarray(half, float) - 0.5 * float(min_member), 0.0)
        n = np.asarray(iso.vertex_normals, float)
        dn = np.einsum("ij,ij->i", v - v0, n)
        for _ in range(HOLD_SPREAD):
            dn = lap @ dn
        back = np.maximum(-dn - allow, 0.0)
        if back.any():
            for _ in range(2):
                back = lap @ back
            iso.vertices = v + n * back[:, None]


HOLD_SPREAD = 10     # averaging passes over the smoothing's shift before _smooth_surface holds a member
THIN = (1.5, 2.5)    # member half-thickness (cells): Taubin below the first, shrinking above the second


JOINT_BODY = 0.01    # a body the opening leaves is worth rejoining at this share of the part's volume


def _keep_joints(before, opened, step: float, min_member: float, level: float = 0.5):
    """The opening, except where it cuts the part in two: a member thinner
    than the minimum that is the only thing joining two sizeable bodies (the
    Ti64 arm link's wrist bearing hung on 2.9 mm members at a 3 mm minimum,
    and with them opened away it was a separate body, its load on nothing)
    is kept, built out to the minimum member round its own ridge. A thin
    piece that joins nothing, a fin or a knife edge, still goes."""
    solid = opened >= level
    lab, n = ndimage.label(solid)
    if n == 0:
        return opened
    size = np.bincount(lab.ravel())
    size[0] = 0
    big = size >= JOINT_BODY * size.sum()
    if big.sum() < 2:
        return opened
    # everything the opening took or left in scraps: what joins the bodies
    # can run through a scrap the opening left on its own
    removed = (before >= level) & ~big[lab]
    rl, rn = ndimage.label(removed)
    if rn == 0:
        return opened
    # which sizeable bodies each removed piece touches
    near = ndimage.binary_dilation(removed)
    touch = near & big[lab]
    # the removed piece beside each touching cell
    owner = ndimage.grey_dilation(rl, size=(3, 3, 3))
    pairs = np.unique(np.stack([owner[touch], lab[touch]], axis=1), axis=0)
    pairs = pairs[pairs[:, 0] > 0]
    if not len(pairs):
        return opened
    counts = np.bincount(pairs[:, 0], minlength=rn + 1)
    joint = counts >= 2
    if not joint.any():
        return opened
    neck = joint[rl]
    # the neck's ridge, and a tube of the minimum member's radius round it
    depth = ndimage.distance_transform_edt(before >= level)
    ridge = neck & (depth >= ndimage.maximum_filter(depth, size=3))
    if not ridge.any():
        ridge = neck
    d = ndimage.distance_transform_edt(~ridge) * step
    tube = np.clip(level - (d - 0.5 * min_member) / (2.0 * step), 0.0, 1.0)
    tube = np.where(ndimage.binary_dilation(neck, iterations=int(np.ceil(0.5 * min_member / step)) + 2), tube, 0.0)
    return np.maximum(opened, np.maximum(np.where(neck, before, 0.0), tube))


BRIDGE_FLOOR = 0.15   # lowest density a bridge (_bridge_pieces) may run through
BRIDGE_SIDE = 0.6     # density given the cells beside a bridge's path


def _bridge_pieces(domain: Domain, xphys: np.ndarray, level: float = 0.5):
    """Densities with every loaded piece joined back to the supports, and a
    note of what was added (None when nothing was cut off).

    At low fractions the optimizer can hold a loaded boss on members it left
    grey (0.2-0.5): the Ti64 arm link at 0.2-0.25 carried its wrist bearing
    on cells that never reach 0.5, so the 0.5 surface left the bearing a
    separate body, verification found the load on nothing, and the search
    climbed to 140-240 g at SF 60-180. Here each loaded piece the 0.5 level
    leaves apart from the supported material is joined to it along the path
    of least missing density through cells at BRIDGE_FLOOR or more (the
    optimizer's own grey route), raised to solid; the opening and the member
    hold downstream size it to minMember. Mirrored for symmetric studies."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    x = np.asarray(xphys, float)
    mesh = domain.mesh
    ijk = mesh.elem_ijk
    solid = x >= level
    lut = np.full(domain.dims, -1, int)
    lut[tuple(ijk.T)] = np.arange(len(x))
    grid = np.zeros(domain.dims, bool)
    grid[tuple(ijk[solid].T)] = True
    lab, n = ndimage.label(grid)
    el_lab = lab[tuple(ijk.T)]
    sup_el = domain.support_nodes[mesh.enode].any(axis=1)
    load_el = domain.load_nodes[mesh.enode].any(axis=1)
    main = set(np.unique(el_lab[sup_el & solid]).tolist()) - {0}
    if not main:
        return x, None
    loaded = set(np.unique(el_lab[load_el & solid]).tolist()) - {0}
    cut = sorted(loaded - main)
    # a load region with no solid at all next to it is cut off too
    if not cut and not (load_el.any() and not (load_el & solid).any()):
        return x, None
    ok = x >= BRIDGE_FLOOR
    # 6-neighbour graph over the cells a bridge may use; entering a cell
    # costs its missing density (solid cells almost nothing)
    rows, cols = [], []
    for a in range(3):
        off = np.zeros(3, int)
        off[a] = 1
        nb = ijk + off
        inside = (nb < np.array(domain.dims)).all(axis=1)
        j = np.full(len(x), -1)
        j[inside] = lut[tuple(nb[inside].T)]
        m = (j >= 0) & ok & np.where(j >= 0, ok[np.maximum(j, 0)], False)
        rows += [np.nonzero(m)[0], j[m]]
        cols += [j[m], np.nonzero(m)[0]]
    r, c = np.concatenate(rows), np.concatenate(cols)
    w = np.maximum(level - x[c], 0.0) + 1e-3
    gph = csr_matrix((w, (r, c)), shape=(len(x), len(x)))
    src = np.nonzero(np.isin(el_lab, list(main)) & solid)[0]
    added = np.zeros(len(x), bool)
    joined = 0
    targets = [np.nonzero((el_lab == k) & solid)[0] for k in cut]
    if not (load_el & solid).any():
        targets.append(np.nonzero(load_el & ok)[0])
    for tgt in targets:
        if not len(tgt):
            continue
        dist, pred, _ = dijkstra(gph, indices=src, min_only=True, return_predecessors=True)
        t = tgt[np.argmin(dist[tgt])]
        if not np.isfinite(dist[t]):
            continue
        while t >= 0 and not added[t]:
            added[t] = True
            t = pred[t]
        joined += 1
        src = np.union1d(src, np.nonzero(added)[0])
        src = np.union1d(src, tgt)
    if not added.any():
        return x, None
    for _axis, mi in (domain.mirror or {}).items():
        mm = mi[added]
        added[mm[mm >= 0]] = True
    out = x.copy()
    out[added] = np.maximum(out[added], 1.0)
    # one cell either side as well: a path one cell across is under most
    # minimum members, and the opening would take it straight out again
    g = np.zeros(domain.dims, bool)
    g[tuple(ijk[added].T)] = True
    side = ndimage.binary_dilation(g)[tuple(ijk.T)] & ~added
    out[side] = np.maximum(out[side], BRIDGE_SIDE)
    return out, {"pieces": joined, "cells": int((added & ~solid).sum())}


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
        # Not the design space's skin: it wraps the whole exterior, and as a
        # flush face it seats the field outward everywhere and fills the
        # interior (a push pad's 7.8 cm^3 interior meshed as 17 cm^3).
        flush = flush + [r.shape for r in study.preserve if r.id != "skin"]
    acc = getattr(study, "access", None)
    if acc is not None and acc.kind == "milling":
        # The stock face no cutter reaches (the bottom of a part milled from
        # +z) stays flat stock: flush, so the mesh cannot round its edges
        # into a sliver of undercut. A plate milled from +z with only its bolt
        # holes held read 590-790 mm^3 unreachable, all at its bottom edges.
        from .shapes import Box

        lo, hi = (np.asarray(b, float) for b in study.bounds())
        t = 0.5 * domain.pitch
        for a in range(3):
            signs = {sg for ax, sg in acc.directions if ax == a}
            if len(signs) != 1:
                continue
            blo, bhi = lo.copy(), hi.copy()
            if signs == {1}:      # cut from +a: the -a face is stock
                blo[a], bhi[a] = lo[a] - t, lo[a] + t
            else:
                blo[a], bhi[a] = hi[a] - t, hi[a] + t
            flush.append(Box(blo, bhi))
    draft = None
    if acc is not None and acc.kind == "casting" and acc.draft > 0:
        from .manufacture import symmetric_parting

        centre = domain.origin + 0.5 * np.asarray(domain.dims) * domain.pitch
        draft = (acc.draw[0], acc.draw[1], len(acc.directions), acc.draft, symmetric_parting(study, acc, centre))
    xphys, bridged = _bridge_pieces(domain, xphys, o.level)
    # The field is filleted and run on into preserved shapes (below), but not
    # into the design space's skin: it wraps the whole exterior, so a 3 mm
    # fillet to it lines every inside corner, and the run-on near it fills
    # the cavity outright (a push pad's 7.8 cm^3 interior meshed as 16.9).
    # The exact union afterwards still adds the skin itself.
    inner = [r.shape for r in study.preserve if r.id != "skin"]
    try:
        iso = _isosurface(domain, xphys, o, [r.shape for r in study.keep_out], study.min_member,
                          flush=flush, preserve=inner, design=list(study.design_space),
                          draft=draft)
    except ValueError as exc:
        # Nothing designed, but preserved geometry to deliver: a fixed
        # exterior whose bare skin already meets every target (a touch
        # zone the skin alone makes pliable) is the skin itself.
        if "never reaches solid" not in str(exc) or not study.preserve:
            raise
        iso = None
    parting = iso.metadata.get("parting") if iso is not None else None
    info = {"isoTriangles": int(len(iso.faces)) if iso is not None else 0}
    if bridged:
        info["bridged"] = bridged
    if iso is None:
        pass
    elif o.taubin_iterations and draft is None:
        # (not on a drafted cast part: smoothing rounds the ledge where a
        # section steps in just past the parting plane down across it,
        # leaning those walls the wrong way for the other mold half; its
        # field was smoothed before the draft and meshes clean without)
        keep_thin = bool((getattr(study, "raw", None) or {}).get("resolution", {}).get("keepThinMembers", False))
        _smooth_surface(iso, domain, xphys, o.taubin_iterations, keep_thin, min_member=study.min_member)
    if iso is not None and study.preserve and o.fillet and o.blend > 0:
        _snap_to_flat_faces(iso, [r.shape for r in study.preserve], domain.pitch / o.upsample, o.blend)
    if iso is None:
        organic = m3d.Manifold()
    elif len(iso.faces) > o.max_triangles:
        # Decimate the organic surface before the booleans, never after: a
        # decimated part shaves the exact preserved faces (a bore wall moved
        # a few hundredths of a millimetre reads as preserved geometry lost).
        try:
            simplified = iso.simplify_quadric_decimation(face_count=o.max_triangles)
            if simplified.is_watertight:
                iso = simplified
        except Exception:  # noqa: BLE001 - decimation is an optimization, never fatal
            pass
    if iso is not None:
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
    # Drop the small bodies by cutting them away, not by rebuilding from the
    # big ones: decompose() lists an enclosed cavity's surface as a body of
    # negative volume, and rebuilding without it filled the cavity. Inside a
    # closed exterior (a handheld's skin, every rib joined to it) the whole
    # interior is one such cavity: the part came out solid, 90 cm^3 for 17%.
    small = [b for b in organic.decompose() if 0 < b.volume() < o.min_body_fraction * total]
    if small:
        organic = organic - (m3d.Manifold.batch_boolean(small, m3d.OpType.Add) if len(small) > 1 else small[0])

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
    if acc is not None and acc.kind == "milling":
        part, filled = _fill_unmachinable(part, study, domain, envelope)
        if filled:
            info["unmachinableFilled"] = filled
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

    clip, ko_all, hold = envelope, None, None
    if study.keep_out:
        ko_all = m3d.Manifold.batch_boolean([exact_manifold(r.shape) for r in study.keep_out], m3d.OpType.Add)
        clip = envelope - ko_all
    if study.preserve:
        hold = m3d.Manifold.batch_boolean([exact_manifold(r.shape) for r in study.preserve], m3d.OpType.Add)
    part, pinches = _fill_pinches(part, clip=clip, keep_out=ko_all, hold=hold)
    if pinches:
        info["pinchesFilled"] = pinches
    mesh = _drop_slivers(trimesh_from_manifold(part))
    if not _stl_watertight(mesh):
        # a boolean can leave sliver pairs only an STL's weld exposes
        # (edges shared by four faces); a pass back through manifold3d
        # drops them. It refuses a mesh that is not a 2-manifold (the full-
        # tier wrist yoke's search part): keep the mesh as it is then.
        try:
            again = _drop_slivers(trimesh_from_manifold(manifold_from_trimesh(mesh)))
        except ValueError:
            again = None
        if again is not None and _stl_watertight(again):
            mesh = again
    if parting is not None:
        mesh.metadata["parting"] = parting
        info["parting"] = parting
    _attach_plan(mesh, study, organic_raw)
    info["triangles"] = int(len(mesh.faces))
    info["volume"] = float(mesh.volume)
    info["watertight"] = bool(mesh.is_watertight) and _stl_watertight(mesh)
    return mesh, info
