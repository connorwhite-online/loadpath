"""
SDF toolkit for the text-to-CAD "functional skeleton + organic skin" path
(Tier 1). The model defines EXACT functional anchors (bosses, holes, faces) and
smooth-unions an ORGANIC connecting body, then `to_mesh` marching-cubes it to a
watertight trimesh. This is how you get organic, load-path-looking forms that
still hold real tolerances — beauty draped over fixed function, not a pile of
joined primitives (and not an unconstrained generative blob).

All primitives are vectorized: they take P, an (N,3) array of sample points, and
return an (N,) signed distance (negative inside the solid). Compose with smin
(smooth union), subtract (cut), and smax (smooth intersect).

v2 adds TPMS lattice fields (gyroid / schwarz_p / diamond), region masking,
shell/offset field operators, point-warp transforms, and `from_mesh` — the
B-rep bridge that turns any trimesh (e.g. a tessellated build123d flange) into
an SDF you can smin into an implicit body.

v3 adds the dual-fluid vocabulary: `tpms_dist` (signed mm distance to a TPMS
midsurface), `seal_ramp` (0..1 face-seal weights), and `dual_sheet` (the
two-fluid separator with per-point channel capping). `exchanger.py` composes
them into a complete manifolded heat-exchanger block.

v4 adds the EXACT-FEATURE path: `mesh_cyl` / `mesh_box` build exactly
tessellated primitives, and `mesh_subtract` / `mesh_union` / `mesh_intersect`
boolean them into a meshed field with manifold3d. Marching cubes cannot
represent a crease, so cutting a bore as a FIELD rounds its mouth into a
chamfer; cutting it as a MESH, after the organic body is meshed, keeps the
rim sharp and any flat mating face planar. Mesh-level primitives take the
same arguments as their field counterparts (`mesh_cyl(x, y, r, z0, z1)`
mirrors `cyl_z(P, x, y, r, z0, z1)`) so swapping one for the other is a
one-word edit.

Importable by sidecar-exec'd model code: `from sdf_kit import *`.
"""
import functools

import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh

__all__ = [
    "smin", "smax", "subtract", "union", "intersect",
    "sphere", "box", "capsule", "tapered_capsule", "spline_tube", "cyl_z",
    "sq_prism", "pocket", "largest_body",
    "superellipsoid",
    "gyroid", "schwarz_p", "diamond",
    "tpms_dist", "seal_ramp", "dual_sheet",
    "mask", "shell_field", "offset_field",
    "translate", "rotate_z",
    "from_mesh", "to_mesh", "split_shell",
    "mesh_union", "mesh_subtract", "mesh_intersect", "mesh_cyl", "mesh_box",
    "mesh_rod",
]


# ---- combinators -----------------------------------------------------------

def _lift(fn):
    """Let a combinator take field FUNCTIONS as well as evaluated arrays.

    The combinators and field operators work on evaluated distances
    (`offset_field(sphere(P, c, r), 2)`), while `to_mesh`, `from_mesh` and
    `split_shell` deal in callables `f(P)`. Generated code mixes the two up
    constantly, since the prompt calls the field `f`: passing `f` itself made
    `offset_field` compute `f - d` and fail with "unsupported operand type(s)
    for -: 'function' and 'float'" (local SDF run, 2026-09-23). Now, if any
    argument is callable, the result is a callable that evaluates every
    callable argument at P first. Arrays go through unchanged, so existing
    code is unaffected.
    """
    @functools.wraps(fn)
    def lifted(*args, **kwargs):
        if not any(callable(a) for a in args):
            return fn(*args, **kwargs)

        def field(P):
            return fn(*(a(P) if callable(a) else a for a in args), **kwargs)

        return field

    return lifted


@_lift
def smin(a, b, k):
    """Smooth union (the organic-blend workhorse). k = blend radius in mm."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


@_lift
def smax(a, b, k):
    """Smooth intersection."""
    return -smin(-a, -b, k)


@_lift
def union(a, b):
    return np.minimum(a, b)


@_lift
def intersect(a, b):
    return np.maximum(a, b)


@_lift
def subtract(d, hole):
    """Cut `hole` out of `d` (hard edge — use for exact bores/pockets)."""
    return np.maximum(d, -hole)


# ---- primitives (negative = inside) ----------------------------------------
def sphere(P, center, r):
    return np.linalg.norm(P - np.asarray(center, float), axis=1) - r


def box(P, center, half):
    """Axis-aligned box. `half` = (hx, hy, hz) half-extents."""
    q = np.abs(P - np.asarray(center, float)) - np.asarray(half, float)
    return (
        np.linalg.norm(np.maximum(q, 0.0), axis=1)
        + np.minimum(np.max(q, axis=1), 0.0)
    )


def capsule(P, a, b, r):
    """Round-capped capsule from a to b (any axis) — the organic strut."""
    a = np.asarray(a, float); b = np.asarray(b, float); ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)[:, None]
    return np.linalg.norm(P - (a + t * ab), axis=1) - r


def _gradient_normalized(F, P, h):
    """Turn an implicit function F (level 0 = surface, negative inside) into
    an approximate distance in real mm: F / |grad F|, gradient by central
    differences. Exact to first order near the surface, which is where
    offsets and shells are measured. Sign is always exact."""
    g = np.zeros((len(P), 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        g[:, k] = (F(P + e) - F(P - e)) / (2.0 * h)
    grad = np.linalg.norm(g, axis=1)
    return F(P) / np.maximum(grad, 1e-9)


def _superellipse_r(x, y, a, b, n):
    return (np.abs(x / a) ** n + np.abs(y / b) ** n) ** (1.0 / n)


def tapered_capsule(P, a, b, ra, rb):
    """A capsule whose radius runs from `ra` at point a to `rb` at point b:
    the organic strut. Chain several (sharing endpoints, radii matching at
    each joint) and smin them with a small k for a prong, handle or limb that
    tapers along a curved spine, a "spline sweep" without a kernel. Exact
    distance (the round-cone SDF), so offsets and shells stay even."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    l2 = float(ba @ ba)
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    x = pa * l2 - y[:, None] * ba
    x2 = np.einsum("ij,ij->i", x, x)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d = np.empty(len(P))
    m_b = np.sign(z) * a2 * z2 > k
    m_a = np.sign(y) * a2 * y2 < k
    d[m_b] = np.sqrt(x2[m_b] + z2[m_b]) * il2 - rb
    d[m_a & ~m_b] = np.sqrt(x2[m_a & ~m_b] + y2[m_a & ~m_b]) * il2 - ra
    rest = ~(m_a | m_b)
    d[rest] = (np.sqrt(x2[rest] * a2 * il2) + y[rest] * rr) * il2 - ra
    return d


def _catmull_rom(points, samples):
    """Centripetal-free uniform Catmull-Rom through `points` (ends clamped),
    returning `samples` points per span plus the last point, and the span
    parameter of each (for interpolating radii the same way)."""
    pts = np.asarray(points, float)
    ext = np.vstack([pts[0], pts, pts[-1]])
    out, ts = [], []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for t in np.linspace(0.0, 1.0, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            ts.append(i + t)
    out.append(pts[-1])
    ts.append(float(len(pts) - 1))
    return np.array(out), np.array(ts)


def spline_tube(P, points, radii, samples=12):
    """A tube swept along a smooth spline through `points`, radius running
    through `radii` (one per point): a prong, handle, limb or cable guide
    that flows without kinks. A hand-built chain of straight segments shows
    an elbow at every control point; this samples a Catmull-Rom curve
    finely and joins the pieces with union (not smin, which would bead
    them). Radii are interpolated along the same curve."""
    pts = np.asarray(points, float)
    rs = np.asarray(radii, float)
    if len(pts) != len(rs) or len(pts) < 2:
        raise ValueError("spline_tube needs >= 2 points and one radius per point")
    curve, t = _catmull_rom(pts, samples)
    rad, _ = _catmull_rom(np.c_[rs, np.zeros((len(rs), 2))], samples)
    r = np.maximum(rad[:, 0], 1e-3)
    d = tapered_capsule(P, curve[0], curve[1], r[0], r[1])
    for i in range(1, len(curve) - 1):
        d = np.minimum(d, tapered_capsule(P, curve[i], curve[i + 1], r[i], r[i + 1]))
    return d


def sq_prism(P, a, b, n, z0, z1):
    """Superellipse prism (|x/a|^n + |y/b|^n = 1 cross-section, z-extruded) —
    the measured plan-form language of premium enclosures (see
    docs/text-to-cad/design-studies/modem-works.md): n~4.5 reads as a
    squircle desk device, n~1.7-2.2 as a handheld lens/pebble. Use this for
    organic product shells INSTEAD of a box with corner fillets — a
    superellipse has no straight-to-arc tangency breaks, which is what makes
    the form read as designed.

    Distance in real mm near the surface (gradient-normalized), so
    offset_field and shell_field give UNIFORM walls. It used to be the
    normalized radius scaled by min(a, b), which is only a distance along the
    short axis: a 2mm offset of a 27x14 prism came out 3.9mm thick at the
    ends. Every enclosure built this way had walls nearly twice as thick at
    the ends as at the sides."""
    h = 1e-3 * min(a, b)
    d2 = _gradient_normalized(
        lambda Q: _superellipse_r(Q[:, 0], Q[:, 1], a, b, n) - 1.0, P, h
    )
    return np.maximum(d2, np.maximum(P[:, 2] - z1, z0 - P[:, 2]))


def pocket(P, center, half, floor_z, n=4.0, r=None):
    """An OPEN cavity: squircle plan (half = (a, b) around center (x, y)),
    a floor at floor_z rounded by r, and NO top. It runs up through
    everything above the floor, so subtract(body, pocket(...)) always opens
    at the rim. Containers built from superellipsoid or box cavities kept
    stopping just under the top surface (a closed blob), or had material
    blended back over them; a pocket cannot close.

    One pocket per compartment; the walls between them are what remains."""
    a, b = half
    r = min(a, b) * 0.35 if r is None else min(r, min(a, b) * 0.95)
    Q = P - np.array([center[0], center[1], 0.0])
    ai, bi = a - r, b - r
    h = 1e-3 * min(ai, bi)
    d2 = _gradient_normalized(
        lambda X: _superellipse_r(X[:, 0], X[:, 1], ai, bi, n) - 1.0, Q, h
    )
    dz = (floor_z + r) - P[:, 2]
    outside = np.sqrt(np.maximum(d2, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    return outside + np.minimum(np.maximum(d2, dz), 0.0) - r


def superellipsoid(P, center, radii, n=3.0, m=3.5):
    """A pebble: superellipse plan (exponent n, like sq_prism) with a rounded
    vertical profile (exponent m). n~2.5-3 plan and m~3-3.5 profile read as
    a soft river stone; m=2 is an ellipsoid, and m>=5 gives slab sides.
    Distance in real mm near the surface, so offset_field / shell_field give
    an even wall: the drape-and-split enclosure recipe with an organic
    cavity instead of a squircle box.

    Flatten the base with smax(..., z0 - P[:, 2], k) for a stable footprint."""
    c = np.asarray(center, float)
    a, b, cz = (float(v) for v in radii)

    def F(Q):
        q = Q - c
        rp = _superellipse_r(q[:, 0], q[:, 1], a, b, n)
        return (rp ** m + np.abs(q[:, 2] / cz) ** m) ** (1.0 / m) - 1.0

    return _gradient_normalized(F, P, 1e-3 * min(a, b, cz))


def cyl_z(P, x, y, r, z0, z1):
    """Flat-capped vertical cylinder (Z axis) — for exact bosses + bolt holes.
    Pass z0<z1 for a boss; extend past the body for a through-hole."""
    dr = np.hypot(P[:, 0] - x, P[:, 1] - y) - r
    dz = np.abs(P[:, 2] - 0.5 * (z0 + z1)) - 0.5 * (z1 - z0)
    outside = np.linalg.norm(np.maximum(np.stack([dr, dz], 1), 0.0), axis=1)
    inside = np.minimum(np.maximum(dr, dz), 0.0)
    return outside + inside


# ---- TPMS lattices ----------------------------------------------------------
# The raw trig fields are NOT distances — their spatial gradient magnitude is
# ~2π/cell, not 1, and it varies across the cell. Each field below divides the
# raw value by its analytic per-point gradient magnitude (first-order Taylor),
# so `thickness` means real millimeters of wall (within a few % near the
# surface; verified by measuring meshed sheets in the test suite). The gradient
# magnitude is floored away from critical points so the far-field stays
# bounded (sign is always exact; only far-from-surface magnitudes are approx).

def _thickness_at(thickness, P):
    """Resolve scalar / per-point array / callable(P) thickness — graded cores
    pass a callable and get a spatially varying wall."""
    if callable(thickness):
        return np.asarray(thickness(P), float)
    return np.asarray(thickness, float)


def _tpms_d(g, gx, gy, gz, w):
    """Raw trig field -> signed real-mm distance to the F=0 midsurface."""
    grad = w * np.sqrt(gx * gx + gy * gy + gz * gz)
    return g / np.maximum(grad, 0.15 * w)


def _tpms(g, gx, gy, gz, w, thickness, P, solid):
    d = _tpms_d(g, gx, gy, gz, w)
    t = _thickness_at(thickness, P)
    if solid:
        # one network: everything on the g<0 side, fattened by t/2 past the
        # level surface. t=0 keeps exactly half of space.
        return d - 0.5 * t
    # sheet: a t-thick wall on the level surface — the two-network separator
    # (heat-exchanger core). The two voids it leaves are fully disjoint.
    return np.abs(d) - 0.5 * t


def _gyroid_field(P, w):
    sx, cx = np.sin(w * P[:, 0]), np.cos(w * P[:, 0])
    sy, cy = np.sin(w * P[:, 1]), np.cos(w * P[:, 1])
    sz, cz = np.sin(w * P[:, 2]), np.cos(w * P[:, 2])
    g = sx * cy + sy * cz + sz * cx
    gx = cx * cy - sz * sx
    gy = cy * cz - sx * sy
    gz = cz * cx - sy * sz
    return g, gx, gy, gz


def _schwarz_p_field(P, w):
    sx, sy, sz = (np.sin(w * P[:, i]) for i in range(3))
    cx, cy, cz = (np.cos(w * P[:, i]) for i in range(3))
    return cx + cy + cz, -sx, -sy, -sz


def _diamond_field(P, w):
    sx, sy, sz = (np.sin(w * P[:, i]) for i in range(3))
    cx, cy, cz = (np.cos(w * P[:, i]) for i in range(3))
    g = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
    gx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
    gy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
    gz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
    return g, gx, gy, gz


_TPMS_FIELDS = {
    "gyroid": _gyroid_field,
    "schwarz_p": _schwarz_p_field,
    "diamond": _diamond_field,
}


def gyroid(P, cell, thickness, solid=False):
    """Gyroid TPMS. `cell` = lattice period in mm; `thickness` = wall in mm
    (scalar, per-point array, or callable(P) for graded density). Default sheet
    form is the two-network separator; solid=True keeps one network as solid
    infill. Clip to a region with `mask` — the field fills all of space."""
    P = np.asarray(P, float)
    w = 2.0 * np.pi / cell
    g, gx, gy, gz = _gyroid_field(P, w)
    return _tpms(g, gx, gy, gz, w, thickness, P, solid)


def schwarz_p(P, cell, thickness, solid=False):
    """Schwarz P TPMS (cos x + cos y + cos z) — straighter, boxier channels
    than the gyroid. Same contract as `gyroid`."""
    P = np.asarray(P, float)
    w = 2.0 * np.pi / cell
    g, gx, gy, gz = _schwarz_p_field(P, w)
    return _tpms(g, gx, gy, gz, w, thickness, P, solid)


def diamond(P, cell, thickness, solid=False):
    """Schwarz D (diamond) TPMS — denser strut network than the gyroid at the
    same cell size. Same contract as `gyroid`."""
    P = np.asarray(P, float)
    w = 2.0 * np.pi / cell
    g, gx, gy, gz = _diamond_field(P, w)
    return _tpms(g, gx, gy, gz, w, thickness, P, solid)


def tpms_dist(P, cell, kind="gyroid"):
    """Signed real-mm distance to a TPMS midsurface (F=0). Positive on fluid
    A's side of the sheet, negative on fluid B's — the substrate `dual_sheet`
    builds on, and handy on its own for probing which labyrinth a point is in
    (e.g. placing `fluid_ports` probes clear of the wall)."""
    P = np.asarray(P, float)
    w = 2.0 * np.pi / cell
    g, gx, gy, gz = _TPMS_FIELDS[kind](P, w)
    return _tpms_d(g, gx, gy, gz, w)


def seal_ramp(region, width):
    """Seal weight in [0, 1] from a region field: 1 where region <= 0 (fully
    sealed), smoothstepping to 0 where region >= width mm. Feed the result to
    `dual_sheet`'s seal_a/seal_b; combine seals from several faces with
    np.maximum (NOT +, which can overshoot 1 and is fine but sloppy)."""
    u = np.clip(1.0 - np.asarray(region, float) / max(float(width), 1e-9),
                0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def dual_sheet(P, cell, wall, kind="gyroid", seal_a=0.0, seal_b=0.0):
    """Two-fluid TPMS separator with per-point channel sealing — the manifold
    capping primitive for heat-exchanger cores.

    The solid is { -h_b(x) <= d(x) <= h_a(x) } where d is the signed mm
    distance to the midsurface and h = wall/2 + seal * (a bound above max|d|).
    With seal_a = seal_b = 0 this is exactly the sheet form of `gyroid` et al.
    Where a seal weight (0..1, from `seal_ramp`, a per-point array, or a
    callable(P)) approaches 1, that fluid's labyrinth is swallowed by solid —
    so sealing fluid B near fluid A's port face leaves only A's channels
    reaching that face, which is how each manifold stays single-fluid.

    Isolation is unconditional regardless of seals: the solid always contains
    the d=0 midsurface (h_a, h_b >= wall/2 > 0), so any path between the two
    voids would have to cross solid. Keep `wall` >= the printable minimum —
    the thinnest point of a gyroid sheet runs ~12% under nominal."""
    P = np.asarray(P, float)
    d = tpms_dist(P, cell, kind)
    # > max|d| for every kind: |g| <= 3 (schwarz_p worst case) over the
    # 0.15*w gradient floor gives |d| <= 3/(0.15*w) ~= 3.18*cell.
    swallow = 3.5 * cell
    t = _thickness_at(wall, P)
    ha = 0.5 * t + _thickness_at(seal_a, P) * swallow
    hb = 0.5 * t + _thickness_at(seal_b, P) * swallow
    return np.maximum(d - ha, -d - hb)


# ---- field operators ---------------------------------------------------------
@_lift
def mask(field_a, region, k=0.0):
    """Solid only where `region` < 0 — the tool for "gyroid only inside this
    jacket". k=0 is a hard clip; k>0 blends the field into the region wall over
    k mm (smooth intersection)."""
    if k <= 0.0:
        return np.maximum(field_a, region)
    return smax(field_a, region, k)


@_lift
def shell_field(field, t):
    """Hollow a field into a t-mm-thick wall on its surface: |field| - t/2.
    Operates on evaluated field values, like the combinators."""
    return np.abs(field) - 0.5 * t


@_lift
def offset_field(field, d):
    """Grow (d>0) or shrink (d<0) a field by d mm: field - d. Rounds convex
    edges by d as a side effect — that is usually the point."""
    return field - d


# ---- point-warp transforms ---------------------------------------------------
# Transforms warp the SAMPLE POINTS, not the field: apply to P *before* passing
# it to a primitive, and that primitive moves by the transform. e.g.
# `sphere(translate(P, (10, 0, 0)), (0, 0, 0), 5)` is a sphere centered at
# x=+10. Compose left-to-right: the outermost warp applies first.

def translate(P, offset):
    """Move a primitive by `offset` mm: evaluates it at P - offset."""
    return np.asarray(P, float) - np.asarray(offset, float)


def rotate_z(P, degrees, center=(0, 0, 0)):
    """Rotate a primitive by `degrees` CCW about a vertical axis through
    `center`: evaluates it at inverse-rotated points."""
    P = np.asarray(P, float)
    c = np.asarray(center, float)
    a = -np.radians(degrees)
    ca, sa = np.cos(a), np.sin(a)
    Q = P - c
    out = np.empty_like(P)
    out[:, 0] = ca * Q[:, 0] - sa * Q[:, 1]
    out[:, 1] = sa * Q[:, 0] + ca * Q[:, 1]
    out[:, 2] = Q[:, 2]
    return out + c


# ---- B-rep bridge ------------------------------------------------------------
def _crossing_parity(mesh, pts, rng):
    """Point-in-solid by ray-crossing parity (odd crossings = inside), pure
    numpy Möller–Trumbore over all triangles. Orientation-INDEPENDENT — it
    survives inconsistent winding and `fix_normals`-flipped cavity shells,
    which vertex/face-normal side tests do not. Random ray directions make
    edge/vertex grazes measure-zero. O(len(pts) * faces): only for the handful
    of classification samples below, never per-voxel."""
    tri = mesh.triangles
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    inside = np.zeros(len(pts), bool)
    for i, o in enumerate(pts):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = o - v0
        u = np.einsum("ij,ij->i", tvec, p) * inv
        q = np.cross(tvec, e1)
        v = np.einsum("j,ij->i", d, q) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hits = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        inside[i] = (int(hits.sum()) % 2) == 1
    return inside


# Enclosed void pockets under this many voxels are voxelization aliasing,
# not geometry — filled as material without a ray test (see _solid_voxels).
_TINY_POCKET_VOX = 32


def _solid_voxels(mesh, pitch, pad=2):
    """Voxel occupancy of a mesh's solid material at `pitch`, padded by `pad`
    void voxels per side. Returns (bool grid, world coords of voxel (0,0,0)
    center). Surface voxels come from trimesh voxelization; each enclosed
    pocket is then classified material-vs-cavity by ray-parity majority vote
    over a few sample voxels, so internal channels stay VOID — a naive
    `fill()` would pave over them and break both `from_mesh` and the network
    verifier."""
    vg = mesh.voxelized(pitch)
    origin = np.asarray(vg.transform, float)[:3, 3] - pad * pitch
    surf = np.pad(np.asarray(vg.matrix, bool), pad)
    # 6-connected labeling of non-surface space: a closed voxel shell can leak
    # diagonally under 26-connectivity, which would misclassify interiors.
    lbl, n = ndimage.label(~surf)
    open_air = set()
    for face in (lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1], lbl[:, :, 0], lbl[:, :, -1]):
        open_air.update(np.unique(face))
    open_air.discard(0)
    solid = surf.copy()
    open_lut = np.zeros(n + 1, bool)
    if open_air:
        open_lut[list(open_air)] = True
    # Sub-resolution pocket triage. A curved surface voxelized at pitch p
    # traps a huge number of MICROSCOPIC enclosed bubbles between shell
    # voxels (measured: a 150mm gyroid at 0.6mm -> 102k pockets, and the old
    # per-pocket full-grid scan + ray test priced that at ~24 minutes). A
    # pocket a few voxels big can neither carry fluid nor matter to an SDF
    # at this pitch — it is aliasing, not geometry — so fill it as material
    # in one vectorized pass and spend ray-parity classification only on
    # real cavities (plenums, labyrinths, deliberate voids). Filling adds
    # solid, which can only ever SEPARATE void networks, and any channel
    # thin enough to thread a sub-32-voxel pocket is already below the
    # check's honest resolution bound (~pitch).
    sizes = np.bincount(lbl.ravel(), minlength=n + 1)
    tiny_lut = (sizes < _TINY_POCKET_VOX) & ~open_lut
    tiny_lut[0] = False
    if tiny_lut.any():
        solid |= tiny_lut[lbl]
    rng = np.random.default_rng(0)  # deterministic runs
    enclosed, samples = [], []
    for lab in range(1, n + 1):
        if open_lut[lab] or tiny_lut[lab]:
            continue
        vox = np.argwhere(lbl == lab)
        take = vox[:: max(1, len(vox) // 5)][:5]
        enclosed.append((lab, len(samples), len(take)))
        samples.extend(origin + take * pitch)
    if enclosed:
        inside = _crossing_parity(mesh, np.asarray(samples), rng)
        for lab, start, count in enclosed:
            if np.mean(inside[start:start + count]) >= 0.5:
                solid[lbl == lab] = True
    return solid, origin


def from_mesh(mesh, pitch=1.0):
    """SDF of an arbitrary trimesh — the B-rep bridge. Build a flange in
    build123d, tessellate it, wrap it here, then smin it into an implicit
    body. Precomputes a signed voxel grid at `pitch` (occupancy + euclidean
    distance transform inside and out) and returns a `field(P)` callable that
    trilinearly interpolates it. Accuracy is ~pitch/2 near the surface; points
    outside the grid get a positive bounding-sphere distance (large and
    positive, so smin composition stays sane). Cost is paid once up front —
    queries are cheap."""
    solid, origin = _solid_voxels(mesh, pitch, pad=3)
    grid = (ndimage.distance_transform_edt(~solid)
            - ndimage.distance_transform_edt(solid)) * pitch
    # EDT measures center-to-center; the true surface sits between a solid and
    # a void voxel center, so pull both sides in by half a voxel.
    grid -= 0.5 * pitch * np.sign(grid)
    n = np.asarray(grid.shape, float)
    center = origin + 0.5 * (n - 1) * pitch
    r_bound = 0.5 * pitch * float(np.linalg.norm(n - 1))

    def field(P):
        P = np.asarray(P, float)
        idx = (P - origin) / pitch
        in_grid = np.all((idx >= 0.0) & (idx <= n - 1), axis=1)
        out = np.maximum(np.linalg.norm(P - center, axis=1) - r_bound, pitch)
        if np.any(in_grid):
            out[in_grid] = ndimage.map_coordinates(
                grid, idx[in_grid].T, order=1, mode="nearest")
        return out

    return field


def split_shell(body_field, cavity_field, z_split, lip_h=3.0, lip_t=1.1,
                fit_gap=0.25):
    """Split a draped shell into two watertight printed-fit halves — the
    keystone of the drape-and-split enclosure recipe (docs/text-to-cad/
    design-studies/modem-works.md): soft skin outside, component cavity
    inside, two pieces joined on an inner lip. Returns (bottom, top) field
    callables. The bottom keeps a lip ring (the innermost lip_t of the wall,
    rising lip_h above the split, fused 1mm into its own wall); the top gets
    the matching recess grown by fit_gap — a real printed clearance you can
    assert numerically. Keep lip_t comfortably under the wall thickness. Cut port/connector
    apertures AFTER splitting (subtract from both returned fields) so they
    notch the lip like a real product seam — the ring is built from the
    cavity and cannot know about ports."""
    def bottom(P):
        c = cavity_field(P)
        below = np.maximum(body_field(P), P[:, 2] - z_split)
        ring = np.maximum(
            np.maximum(-c, c - lip_t),
            np.maximum(P[:, 2] - (z_split + lip_h), (z_split - 1.0) - P[:, 2]),
        )
        return np.minimum(below, ring)

    def top(P):
        c = cavity_field(P)
        above = np.maximum(body_field(P), z_split - P[:, 2])
        recess = np.maximum(
            np.maximum(-(c + fit_gap), c - (lip_t + fit_gap)),
            np.maximum(P[:, 2] - (z_split + lip_h + fit_gap),
                       (z_split - 0.5) - P[:, 2]),
        )
        return np.maximum(above, -recess)

    return bottom, top


# ---- meshing ---------------------------------------------------------------
# Marching-cubes grid ceiling. 60M cells ≈ 480MB for the float64 field (plus
# ~2x transient inside skimage) — comfortable under the child's 8GB RLIMIT_AS
# and the 600s exec ceiling (measured: ~160s to evaluate+march 44M cells of a
# hooded exchanger field). The old 8M budget dated from the 60s-timeout era
# and forced 150mm-class exchangers to mesh at pitch 1.0 against 1.2mm walls
# (over the wall/3 contract): visibly stair-stepped hoods and a real pinhole
# risk in the printed part.
_CELL_BUDGET = 60_000_000
def largest_body(mesh, max_debris_frac=0.02):
    """Drop marching-cubes specks: return the largest connected body when
    everything else together is under `max_debris_frac` of the total volume,
    else the mesh unchanged (a real second body still fails the one-body
    gate). For concept blockouts, where a pinhead island beside the part is
    noise, not a defect worth discarding the concept over. Finished parts
    keep the strict gate."""
    try:
        bodies = list(mesh.split(only_watertight=False))
    except Exception:  # noqa: BLE001
        return mesh
    if len(bodies) < 2:
        return mesh
    vols = [abs(float(b.volume)) for b in bodies]
    total = sum(vols)
    i = int(np.argmax(vols))
    if total <= 0 or (total - vols[i]) > max_debris_frac * total:
        return mesh
    return bodies[i]


def to_mesh(field, lo, hi, pitch=0.7):
    """Evaluate `field(P)->(N,) sdf` on a grid over [lo,hi] (mm) and
    marching-cubes the level-0 surface into a watertight trimesh. The grid is
    padded with 'void' so the surface always closes. Grids over 8M cells are
    refused before allocation with an error naming a pitch that fits — a
    self-explanatory failure the repair loop can act on."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    # Sub-resolution origin shift: a surface that runs EXACTLY along a grid
    # plane with locally-flat field gradient (e.g. a superellipse's x=0
    # tangent zone on a symmetric grid) makes marching cubes emit degenerate
    # slivers that weld non-manifold. An irrational sub-pitch offset makes
    # exact alignment impossible without measurably changing the bounds.
    lo = lo - pitch * 0.013717
    n = [int(np.ceil((hi[i] - lo[i]) / pitch)) + 1 for i in range(3)]
    cells = n[0] * n[1] * n[2]
    if cells > _CELL_BUDGET:
        ext = hi - lo
        p = float(np.prod(ext) / _CELL_BUDGET) ** (1.0 / 3.0)
        p = max(np.ceil(p * 10.0) / 10.0, 0.1)
        while np.prod([int(np.ceil(e / p)) + 1 for e in ext]) > _CELL_BUDGET:
            p = round(p + 0.1, 1)
        raise ValueError(
            f"grid {n[0]}x{n[1]}x{n[2]} = {cells / 1e6:.1f}M cells exceeds "
            f"the {_CELL_BUDGET // 1_000_000}M budget — use pitch >= {p:g}")
    axes = [np.linspace(lo[i], hi[i], n[i]) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    F = np.asarray(field(P), float).reshape(*n)
    F = np.pad(F, 1, mode="constant", constant_values=float(pitch * 3 + 1.0))
    # Exact zeros happen whenever a hard clip (mask/subtract) lands on a grid
    # coordinate; marching cubes then emits coincident vertices there and the
    # merge below welds them into non-manifold edges. Nudge them off the level
    # by more than the vertex-merge tolerance (shifts the surface ~1e-4 voxels
    # at those nodes — far below print resolution).
    F[F == 0.0] = -1e-4 * pitch
    verts, faces, _, _ = measure.marching_cubes(F, level=0.0)
    for i in range(3):
        verts[:, i] = lo[i] + (verts[:, i] - 1.0) * (hi[i] - lo[i]) / (n[i] - 1)
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.merge_vertices()
    m.fix_normals()
    # Drop sub-voxel debris: hard clips (mask/subtract/split_shell planes) can
    # leave marching-cubes slivers — closed bodies a few cells across with
    # ~zero volume. They are unprintable dust, and the sidecar's fragment gate
    # rightly fails any part that ships disconnected bodies. Anything at least
    # a few voxels big is kept — genuinely disjoint REAL geometry still reaches
    # the gate and fails loudly there, as it should.
    try:
        bodies = m.split(only_watertight=False)
        if len(bodies) > 1:
            tol = (2.0 * pitch) ** 3
            kept = [b for b in bodies if abs(float(b.volume)) > tol]
            if kept and len(kept) < len(bodies):
                m = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    except Exception:  # noqa: BLE001 — debris filtering is best-effort
        pass
    return m


# ---- exact-feature booleans (manifold3d) ------------------------------------
# WHY these exist, measured on a flat plate with a through-bore at pitch 0.8:
#
#                       bore r at the rim   flat-face planarity
#   field subtract          2.789 mm            0.0281 mm
#   mesh boolean            2.500 mm            0.0000 mm   (true r = 2.500)
#
# Away from edges both are fine — a field subtract measures 2.488-2.495 across
# pitch 0.4-1.2, well inside print tolerance. SIZES were never the problem.
# EDGES are: marching cubes cannot represent a crease, so it rounds a bore
# mouth into a 0.29mm chamfer, which is a press fit or a fastener seat gone.
#
# So the pattern to reach for is: mesh the ORGANIC field once, then boolean
# the EXACT features into the result. Not the other way around, and not by
# subtracting exact features as fields.
#
# manifold3d also guarantees manifold output by construction, which the
# marching-cubes path cannot: non-manifold blends are a live failure mode in
# production (one build shipped euler number 184 against a closed-surface 2,
# with 425 faces on open boundary edges).


def _to_manifold(mesh):
    import manifold3d

    return manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=np.asarray(mesh.vertices, np.float32),
            tri_verts=np.asarray(mesh.faces, np.uint32),
        )
    )


def _from_manifold(man):
    md = man.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(md.vert_properties[:, :3], np.float64),
        faces=np.asarray(md.tri_verts, np.int64),
    )


def _boolean(a, b, op):
    ma, mb = _to_manifold(a), _to_manifold(b)
    out = {"union": ma + mb, "subtract": ma - mb, "intersect": ma ^ mb}[op]
    if out.is_empty():
        raise ValueError(
            f"mesh_{op} produced an empty solid — check the operands overlap "
            f"and that you are cutting the bore OUT of the body, not the body "
            f"out of the bore")
    return _from_manifold(out)


def mesh_union(a, b):
    """Fuse two meshes into one manifold solid."""
    return _boolean(a, b, "union")


def mesh_subtract(a, b):
    """Cut mesh `b` out of mesh `a`, keeping `b`'s edges crisp. THE exact-hole
    operator: mesh the organic body with to_mesh, then subtract mesh_cyl bores
    so tolerances survive the sampling grid."""
    return _boolean(a, b, "subtract")


def mesh_intersect(a, b):
    """Keep only the volume common to both meshes."""
    return _boolean(a, b, "intersect")


def mesh_cyl(x, y, r, z0, z1, sections=96):
    """Exactly tessellated vertical cylinder — the mesh twin of `cyl_z`, with
    the same arguments. Use for bores, counterbores and pin seats that must
    hold a real tolerance. Extend past the body for a through-hole."""
    if z1 <= z0:
        raise ValueError(f"mesh_cyl needs z1 > z0, got z0={z0}, z1={z1}")
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    c.apply_translation([x, y, 0.5 * (z0 + z1)])
    return c


def mesh_rod(a, b, r, sections=96):
    """Exactly tessellated cylinder from point `a` to point `b`, on any axis:
    the mesh twin of `capsule`, but flat-ended. For holes that aren't
    vertical: radial set screws, cross pins, side ports. Without it generated
    code had no exact non-vertical cut at all, and a knob's radial M3 set-screw
    hole came out as a square slot ("rotate not available -> use box slot",
    local SDF run 2026-09-23). Extend past the body for a through-hole."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.linalg.norm(b - a) <= 0:
        raise ValueError(f"mesh_rod needs two distinct points, got a={a}, b={b}")
    return trimesh.creation.cylinder(
        radius=r, segment=np.stack([a, b]), sections=sections
    )


def mesh_box(center, half):
    """Exactly tessellated axis-aligned box — the mesh twin of `box`, with the
    same arguments. Use for flat mating faces, slots and rectangular pockets
    that must stay planar and square."""
    b = trimesh.creation.box(extents=[2.0 * h for h in half])
    b.apply_translation(np.asarray(center, float))
    return b
