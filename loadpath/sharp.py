"""
Sharp features for field-built parts, and a measure of how sharp a part is.

A field meshed by marching cubes cannot hold a crease: wherever a hard
boolean meets two surfaces at an angle (a bearing bore through a flat face,
a bolt face, a keep-out's wall) the mesh comes out rounded over about one
grid step, and a bore cut from the field is a polygon a little undersized
near its rims. The study path (reconstruct.py) has never had this problem
for its interfaces because it does the last booleans on exact meshes; this
module gives field graphs the same treatment.

`plan(graph)` splits a graph at its hard booleans (union, intersect and
subtract with no blend) wherever one side is a plain shape (box, cylinder,
sphere, a union of those) or an imported mesh: those sides become `Exact`
leaves, cut, added or clipped with manifold3d, and everything else stays
an `Organic` leaf meshed from its field. The organic side is meshed from
an *extended* field: inside the region an exact shape will cut away (or
add, or clip off) anyway, the field is carried on from just outside it
along the shape's normal, so the organic mesh runs straight through where
the exact surface will be instead of trying to draw it (a collar built as
a shell round a bore has its own inner wall exactly on the bore's: meshed
as-is, the exact cut would leave a film of slivers there). The exact
boolean then draws the crease, on the exact shape's surface.

`edge_accuracy(mesh, refs, solid)` measures the result: it samples the
surfaces of the exact shapes where they are part of the delivered part's
boundary and reports how far the mesh is from them, which is where a
rounded crease shows (a 90 degree edge rounded with radius r is
0.41 r off at its corner).

The same plan is what `step.export_step` turns into a B-rep: exact leaves
become exact OpenCascade solids (planes and cylinders in the STEP file),
organic leaves faceted solids.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dfield

import numpy as np

__all__ = ["Exact", "Organic", "Op", "plan", "mesh_plan", "plan_refs", "has_exact", "exact_manifold",
           "surface_samples", "edge_accuracy", "segments_for", "study_refs", "mesh_distance"]

# chordal tolerance (mm) of the exact cylinders and spheres in a sharp mesh
CHORD_TOL = 0.002


@dataclass
class Exact:
    """A plain shape (shapes.Shape) taken exactly into the result."""
    shape: object
    name: str = ""


@dataclass
class Organic:
    """A field leaf: meshed from `field(P)` (negative inside)."""
    field: object
    names: list = dfield(default_factory=list)
    mesh: object = None          # set by mesh_plan
    extended: bool = False


@dataclass
class Op:
    op: str                      # union | intersect | subtract
    kids: list


def _hard(nd):
    b = nd.refs.get("blend")
    return b is None or (isinstance(b, float) and abs(b) <= 1e-12)


def _exact_leaf(g, name):
    """The exact shape behind a graph node, or None."""
    from .shapes import MeshShape, shape_from_spec

    nd = g.nodes[name]
    if nd.op == "shape":
        return g._resource(nd, lambda: shape_from_spec(nd.spec["shape"], g.base_dir))
    if nd.op == "mesh":
        mf = g._resource(nd, lambda: g._mesh_field(nd))
        try:
            return MeshShape(mf.mesh)
        except ValueError:
            return None
    return None


def _combine(op, fields):
    if len(fields) == 1:
        return fields[0]
    if op == "union":
        return lambda P: np.min([f(P) for f in fields], axis=0)
    if op == "intersect":
        return lambda P: np.max([f(P) for f in fields], axis=0)
    first, rest = fields[0], fields[1:]
    return lambda P: np.max([first(P)] + [-f(P) for f in rest], axis=0)


def plan(g, name=None):
    """The CSG plan of graph node `name` (default the output): an Organic
    leaf for a graph with nothing to do exactly, else a tree of Op, Exact
    and Organic. Hard booleans between field-only inputs stay in the field
    (two field surfaces that coincide, a skin and the envelope it was
    shelled from, must not meet in a mesh boolean)."""
    return _plan(g, name or g.output)


def _plan(g, name):
    nd = g.nodes[name]
    shape = _exact_leaf(g, name)
    if shape is not None:
        return Exact(shape, name)
    if nd.op in ("union", "intersect", "subtract") and _hard(nd):
        kids = [_plan(g, x) if isinstance(x, str) else None for x in nd.refs["inputs"]]
        if any(k is None for k in kids) or all(isinstance(k, Organic) for k in kids):
            return Organic(g.field(name), [name])
        if nd.op in ("union", "intersect"):
            org = [k for k in kids if isinstance(k, Organic)]
            rest = [k for k in kids if not isinstance(k, Organic)]
            if len(org) > 1:
                org = [Organic(_combine(nd.op, [k.field for k in org]), sum((k.names for k in org), []))]
            return Op(nd.op, org + rest)
        first, cutters = kids[0], kids[1:]
        org = [k for k in cutters if isinstance(k, Organic)]
        if isinstance(first, Organic) and org:
            first = Organic(_combine("subtract", [first.field] + [k.field for k in org]),
                            first.names + sum((k.names for k in org), []))
            cutters = [k for k in cutters if not isinstance(k, Organic)]
        if not cutters:
            return first
        return Op("subtract", [first] + cutters)
    return Organic(g.field(name), [name])


def has_exact(p):
    return not isinstance(p, Organic)


def _leaves(p):
    if isinstance(p, Op):
        for k in p.kids:
            yield from _leaves(k)
    else:
        yield p


def plan_refs(p, sign=1):
    """[(shape, role)] of the plan's exact leaves: 'cut' (the part is outside
    it), 'add' (unioned on) or 'clip' (the part is inside it)."""
    out = []
    if isinstance(p, Exact):
        return [(p.shape, "add" if sign > 0 else "cut")]
    if isinstance(p, Op):
        for i, k in enumerate(p.kids):
            if p.op == "subtract" and i > 0:
                out += plan_refs(k, -sign)
            elif p.op == "intersect" and isinstance(k, Exact):
                out.append((k.shape, "clip" if sign > 0 else "cut"))
            elif p.op == "subtract" and i == 0 and isinstance(k, Exact):
                out.append((k.shape, "clip" if sign > 0 else "cut"))
            else:
                out += plan_refs(k, sign)
    return out


def study_refs(study):
    """[(shape, role)] of a study's exact interfaces: keep-outs are cut,
    preserved shapes added, the design space clips."""
    return ([(r.shape, "cut") for r in study.keep_out] + [(r.shape, "add") for r in study.preserve]
            + [(sh, "clip") for sh in study.design_space])


# ---- the extended field ----------------------------------------------------------

def _dont_care(p, inherited=()):
    """Yield (Organic, [(shape, +1 inside | -1 outside)]): for each organic
    leaf, the regions where its value cannot change the result."""
    if isinstance(p, Organic):
        yield p, list(inherited)
        return
    if not isinstance(p, Op):
        return
    exact = [k for k in p.kids if isinstance(k, Exact)]
    for i, k in enumerate(p.kids):
        if not isinstance(k, (Op, Organic)):
            continue
        extra = []
        if p.op == "union":
            extra = [(e.shape, +1) for e in exact]
        elif p.op == "intersect":
            extra = [(e.shape, -1) for e in exact]
        elif p.op == "subtract":
            if i == 0:
                extra = [(e.shape, +1) for e in exact[0:] if e is not p.kids[0]]
            elif isinstance(p.kids[0], Exact):
                extra = [(p.kids[0].shape, -1)]
        yield from _dont_care(k, list(inherited) + extra)


class _Extended:
    """field(P) carried on into the don't-care region D along its normals:
    with g the signed distance to D (negative inside), a point with
    g < eps reads the field at its projection onto the g = eps level, and
    from 2 * `core` inside D it is solid regardless (the projection
    jumps across D's medial axis; whatever the field does there is cut
    away, so it is kept from drawing surfaces)."""

    def __init__(self, field, regions, eps, core):
        self.field = field
        self.regions = regions
        self.eps = float(eps)
        self.core = float(core)

    def g(self, P):
        return np.min([s * sh.sdf(P) for sh, s in self.regions], axis=0)

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        gv = self.g(P)
        near = gv < self.eps
        if not near.any():
            return self.field(P)
        Q = P.copy()
        Pn = P[near]
        h = 1e-3
        grad = np.stack([(self.g(Pn + h * e) - self.g(Pn - h * e)) / (2 * h) for e in np.eye(3)], axis=1)
        nrm = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
        Q[near] = Pn + (self.eps - gv[near])[:, None] * nrm
        out = np.asarray(self.field(Q), float)
        # clamped from below, the carried-on field gives way to the solid
        # core by 2 * core deep: continuous there, whatever the projection
        # does at the medial axis (a jump would wreck narrowband's
        # Lipschitz bound)
        return np.minimum(np.maximum(out, -self.core), gv + self.core)


# ---- meshing ---------------------------------------------------------------------

def segments_for(r, tol=CHORD_TOL):
    """Segments per circle so a radius-r polygon is within tol of the circle."""
    if r <= tol:
        return 8
    return int(min(max(math.ceil(math.pi / math.acos(1.0 - tol / r)), 32), 2048))


def exact_manifold(shape, tol=CHORD_TOL):
    """shape.to_manifold() with circles fine enough to be `tol` mm true."""
    import manifold3d as m3d

    from .shapes import Box, Cylinder, Shell, Sphere, Union

    if isinstance(shape, Cylinder):
        n = segments_for(shape.r, tol)
        outer = m3d.Manifold.cylinder(shape.length, shape.r, shape.r, n)
        if shape.r_inner > 0:
            ni = segments_for(shape.r_inner, tol)
            inner = m3d.Manifold.cylinder(shape.length + 2.0, shape.r_inner, shape.r_inner, ni).translate((0, 0, -1.0))
            outer = outer - inner
        return shape._align(outer)
    if isinstance(shape, Sphere):
        return m3d.Manifold.sphere(shape.r, segments_for(shape.r, tol)).translate(tuple(shape.center))
    if isinstance(shape, Union):
        mans = [exact_manifold(s, tol) for s in shape.shapes]
        return m3d.Manifold.batch_boolean(mans, m3d.OpType.Add) if len(mans) > 1 else mans[0]
    if isinstance(shape, Shell):
        man = exact_manifold(shape.outer, tol) - exact_manifold(shape.inner, tol)
        return man if shape.clip is None else man ^ exact_manifold(shape.clip, tol)
    if isinstance(shape, Box):
        return shape.to_manifold()
    return shape.to_manifold()


def mesh_plan(p, lo, hi, pitch=None, project=3, progress=None, tol=CHORD_TOL, decimate=None):
    """Mesh a plan: organic leaves by fields.field_mesh over [lo, hi] (from
    their extended fields), exact leaves exactly, booleans by manifold3d.
    Returns a trimesh; each Organic leaf keeps its mesh in `.mesh`."""
    import manifold3d as m3d

    from .fields import _finish, field_mesh
    from .shapes import manifold_from_trimesh, trimesh_from_manifold

    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if pitch is None:
        pitch = float((np.prod(hi - lo) / 4e6) ** (1 / 3))
    pitch = float(pitch)
    stats = {}
    for leaf, regions in _dont_care(p):
        fld = leaf.field
        if regions:
            fld = _Extended(leaf.field, regions, eps=pitch, core=pitch)
            leaf.extended = True
        leaf.mesh = field_mesh(fld, lo, hi, pitch=pitch, project=project, progress=progress, decimate=decimate)
        if leaf.mesh.metadata.get("sampling"):
            stats = leaf.mesh.metadata["sampling"]

    def build(q):
        if isinstance(q, Exact):
            return exact_manifold(q.shape, tol)
        if isinstance(q, Organic):
            m = q.mesh
            try:
                return manifold_from_trimesh(m)
            except ValueError:
                import trimesh
                m = m.copy()
                trimesh.repair.fix_winding(m)
                trimesh.repair.fill_holes(m)
                try:
                    return manifold_from_trimesh(m)
                except ValueError:
                    # fill_holes leaves loops over four edges: a few slivers a
                    # thin lattice wall left open at a coarse pitch
                    return manifold_from_trimesh(_close_loops(m))
        kids = [build(k) for k in q.kids]
        if q.op == "union":
            return m3d.Manifold.batch_boolean(kids, m3d.OpType.Add)
        if q.op == "intersect":
            return m3d.Manifold.batch_boolean(kids, m3d.OpType.Intersect)
        return m3d.Manifold.batch_boolean(kids, m3d.OpType.Subtract)

    man = build(p)
    if man.is_empty():
        raise ValueError("the part is empty after its exact booleans")
    mesh = trimesh_from_manifold(man)
    mesh = _finish(mesh.vertices, mesh.faces, None, pitch, 0, stats or None)
    mesh.metadata["sharp"] = True
    from .step import attach_plan

    return attach_plan(mesh, p)


def _close_loops(m, max_len=64):
    """Close each open boundary loop of up to `max_len` edges with a fan
    from its centroid, wound against the faces beside it."""
    import trimesh

    F = np.asarray(m.faces)
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(d, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    nxt = {int(a): int(b) for a, b in d[cnt[inv.ravel()] == 1]}
    V = [np.asarray(m.vertices, float)]
    new = []
    n = len(V[0])
    while nxt:
        a0, a = nxt.popitem()
        loop = [a0]
        while a is not None and a != a0:
            loop.append(a)
            a = nxt.pop(a, None)
        if a is None or len(loop) < 3 or len(loop) > max_len:
            continue
        V.append(V[0][loop].mean(axis=0)[None])
        new += [[loop[(i + 1) % len(loop)], loop[i], n] for i in range(len(loop))]
        n += 1
    if not new:
        return m
    return trimesh.Trimesh(np.vstack(V), np.vstack([F, np.asarray(new)]), process=False)


# ---- measuring -------------------------------------------------------------------

def _frame(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, t)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def surface_samples(shape, spacing=0.25):
    """(points, outward normals) spread over a shape's surface about
    `spacing` mm apart."""
    from .shapes import Box, Cylinder, MeshShape, Sphere, Union

    P, N = [], []
    if isinstance(shape, Box):
        for a in range(3):
            b, c = [i for i in range(3) if i != a]
            ub = np.arange(shape.lo[b] + spacing / 2, shape.hi[b], spacing)
            uc = np.arange(shape.lo[c] + spacing / 2, shape.hi[c], spacing)
            B, C = (x.ravel() for x in np.meshgrid(ub, uc, indexing="ij"))
            for sgn, at in ((-1.0, shape.lo[a]), (1.0, shape.hi[a])):
                q = np.empty((len(B), 3))
                q[:, a], q[:, b], q[:, c] = at, B, C
                n = np.zeros((len(B), 3))
                n[:, a] = sgn
                P.append(q)
                N.append(n)
    elif isinstance(shape, Cylinder):
        a, u, v = _frame(shape.axis)
        ts = np.arange(spacing / 2, shape.length, spacing)
        for r, sgn in ((shape.r, 1.0), (shape.r_inner, -1.0)):
            if r <= 0:
                continue
            th = np.arange(0, 2 * math.pi, spacing / r)
            T, TH = (x.ravel() for x in np.meshgrid(ts, th, indexing="ij"))
            radial = np.cos(TH)[:, None] * u + np.sin(TH)[:, None] * v
            P.append(shape.p0 + T[:, None] * a + r * radial)
            N.append(sgn * radial)
        rings = np.arange(max(shape.r_inner, 0) + spacing / 2, shape.r, spacing)
        for at, sgn in ((shape.p0, -1.0), (shape.p1, 1.0)):
            for r in rings:
                th = np.arange(0, 2 * math.pi, spacing / r)
                radial = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
                P.append(at + r * radial)
                N.append(np.repeat(sgn * a[None], len(th), axis=0))
    elif isinstance(shape, Sphere):
        n = max(int(4 * math.pi * shape.r ** 2 / spacing ** 2), 16)
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = math.pi * (1 + 5 ** 0.5) * i
        d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
        P.append(shape.center + shape.r * d)
        N.append(d)
    elif isinstance(shape, Union):
        for s in shape.shapes:
            p, n = surface_samples(s, spacing)
            P.append(p)
            N.append(n)
    elif isinstance(shape, MeshShape):
        import trimesh

        m = shape.mesh
        cnt = int(min(max(m.area / spacing ** 2, 100), 400_000))
        p, fi = trimesh.sample.sample_surface_even(m, cnt, seed=1)
        P.append(np.asarray(p))
        N.append(np.asarray(m.face_normals)[fi])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(N)


def edge_accuracy(mesh, refs, solid=None, boundary=None, spacing=0.25, tol=0.01, delta=0.05, near=1.0):
    """How exactly `mesh` reproduces the exact shapes it was built with.

    refs: [(shape, role)] (see plan_refs; studies: keep-outs 'cut', preserved
    'add', design space 'clip'). A sample of each shape's surface counts
    where it is the part's boundary: `boundary(P)` (True where P is on the
    ideal part's surface, from an exact field) when given, else the
    `solid(P)` oracle (default: the mesh itself) a `delta` step off the
    surface on the material side. For each counted point the distance to
    the mesh is the error; samples more than max(0.5, 4 delta) mm from the
    mesh are taken as not on the part, so that is the largest error read.

    Returns overall and per-role stats: mean, p99 and max error (mm), the
    fraction of interface area within `tol` (`exactFraction`), and the
    same over the points within `near` mm of a crease of the interface (the
    edges: where the rounding is)."""
    if solid is None:
        def solid(P):
            # the probes are a step off the surface: the closest triangles' side
            return mesh_distance(mesh, P, signed=True) < 0
    rows = {}
    allP, allE, allEdge = [], [], []
    for shape, role in refs:
        P, N = surface_samples(shape, spacing)
        if not len(P):
            continue
        # the shape's own surface, not the parts of it buried in another
        # piece of the same shape (a union's members)
        if not _has_mesh(shape):
            own = np.abs(shape.sdf(P)) < 1e-6
            P, N = P[own], N[own]
        # a sample the mesh does not come near is not on the part (most of
        # a design space's faces): drop those before the costlier tests
        far = max(0.5, 4 * delta)
        dist = mesh_distance(mesh, P, cap=far)
        near_mesh = dist < far
        P, N, dist = P[near_mesh], N[near_mesh], dist[near_mesh]
        if boundary is not None:
            keep = boundary(P)
        else:
            side = -1.0 if role == "clip" else 1.0
            probe = P + side * delta * N
            inside = solid(probe)
            keep = inside if role in ("cut", "clip") else ~inside
            if role == "add":
                # out in the air beside a preserved face, and on the part
                keep &= solid(P - delta * N)
        P, N, dist = P[keep], N[keep], dist[keep]
        if not len(P):
            continue
        # near an edge of this interface: a counted point with an uncounted
        # sample within `near` (the interface stops there: a crease)
        allP.append(P)
        allE.append(dist)
        rows.setdefault(role, []).append(dist)
    if not allE:
        return {"points": 0}
    P = np.concatenate(allP)
    E = np.concatenate(allE)
    edge = _near_boundary(P, near, spacing)

    def stats(e):
        if not len(e):
            return {"points": 0}
        return {"points": int(len(e)), "mean": round(float(e.mean()), 4), "p99": round(float(np.percentile(e, 99)), 4),
                "max": round(float(e.max()), 4), "exactFraction": round(float((e <= tol).mean()), 4)}

    if hasattr(mesh, "_loadpath_dist"):
        del mesh._loadpath_dist         # the split triangles: big, and only for this measurement
    out = stats(E)
    out["edges"] = stats(E[edge])
    out["tol"] = tol
    out["byRole"] = {k: stats(np.concatenate(v)) for k, v in rows.items()}
    return out


def mesh_distance(mesh, P, edge=0.5, k=16, cap=np.inf, signed=False):
    """Distance from points P to a triangle mesh's surface (signed,
    negative inside, with `signed`: by the normals of the closest
    triangles, so for points near a closed mesh).

    trimesh.proximity.closest_point gathers candidates by triangle
    bounding boxes, and an exact boolean's long slivers (a bore wall is a
    ring of triangles the length of the bore) make every point a candidate
    of all of them; here the mesh is split to `edge` mm and each point
    tries the k triangles with the nearest centroids. Distances are exact
    up to `cap`; beyond it they are only known to be at least cap."""
    import trimesh
    from scipy.spatial import cKDTree

    from .fields import _closest_on_triangles

    key = (id(mesh), edge)
    cache = getattr(mesh, "_loadpath_dist", None)
    if cache is None or cache[0] != key:
        v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
        v, f = trimesh.remesh.subdivide_to_size(v, f, max_edge=edge, max_iter=16)
        tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
        cache = (key, tri, nrm, cKDTree(tri.mean(axis=1)))
        mesh._loadpath_dist = cache
    _, tri, nrm, tree = cache
    P = np.asarray(P, float).reshape(-1, 3)
    out = np.empty(len(P))
    side = np.ones(len(P))
    k = min(k, len(tri))
    reach = edge / math.sqrt(3.0)          # no point of a split triangle is further from its centroid

    def best(q, cand):
        n = cand.shape[1]
        step = max(1, 2_000_000 // n)
        if len(q) > step:
            parts = [best(q[j:j + step], cand[j:j + step]) for j in range(0, len(q), step)]
            return np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts])
        T = tri[cand]
        Q = np.repeat(q[:, None, :], n, axis=1).reshape(-1, 3)
        C = _closest_on_triangles(Q, T[:, :, 0].reshape(-1, 3), T[:, :, 1].reshape(-1, 3),
                                  T[:, :, 2].reshape(-1, 3)).reshape(len(q), n, 3)
        d2 = np.sum((C - q[:, None, :]) ** 2, axis=2)
        b2 = d2.min(axis=1)
        if not signed:
            return np.sqrt(b2), np.ones(len(q))
        # the side: normals of every candidate sharing the closest point
        # (an edge or a corner), as fields.MeshField does
        tie = d2 <= b2[:, None] * (1 + 1e-6) + 1e-18
        cb = C[np.arange(len(q)), np.argmin(d2, axis=1)]
        sn = np.einsum("ij,ij->i", q - cb, (nrm[cand] * tie[:, :, None]).sum(axis=1))
        return np.sqrt(b2), np.where(sn < 0, -1.0, 1.0)

    for i in range(0, len(P), 50_000):
        q = P[i:i + 50_000]
        d = np.full(len(q), np.inf)
        sg = np.ones(len(q))
        todo = np.arange(len(q))
        kk = k

        def take(idx, cand):
            dd, ss = best(q[idx], cand)
            better = dd < d[idx]
            d[idx[better]] = dd[better]
            sg[idx[better]] = ss[better]

        # the k nearest centroids miss the closest triangle only if a
        # centroid beyond them is within d + reach: widen k for those
        while len(todo):
            kk = min(kk, len(tri))
            dk = np.empty((len(todo), kk))
            for j in range(0, len(todo), max(1, 2_000_000 // kk)):
                part = todo[j:j + max(1, 2_000_000 // kk)]
                dkj, cand = tree.query(q[part], k=kk)
                dk[j:j + len(part)] = np.asarray(dkj).reshape(len(part), kk)
                take(part, np.asarray(cand).reshape(len(part), kk))
            if kk == len(tri):
                break
            todo = todo[dk[:, -1] < np.minimum(d[todo], cap) + reach]
            kk *= 4
            if kk > 4096 and len(todo):
                for j in todo:          # a handful of points among thousands of tiny triangles
                    c = tree.query_ball_point(q[j], d[j] + reach)
                    if c:
                        take(np.array([j]), np.asarray(c)[None])
                break
        out[i:i + 50_000] = d
        side[i:i + 50_000] = sg
    return out * side if signed else out


def _has_mesh(shape):
    from .shapes import MeshShape, Union

    if isinstance(shape, Union):
        return any(_has_mesh(s) for s in shape.shapes)
    return isinstance(shape, MeshShape)


def _near_boundary(P, near, spacing):
    """Points with fewer neighbours within `near` than a flat interior
    point would have: the rims of an interface patch."""
    from scipy.spatial import cKDTree

    if len(P) < 4:
        return np.ones(len(P), bool)
    tree = cKDTree(P)
    cnt = np.asarray([len(x) for x in tree.query_ball_point(P, near, workers=-1)])
    full = math.pi * near ** 2 / spacing ** 2
    return cnt < 0.8 * full
