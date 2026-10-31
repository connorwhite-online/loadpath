"""
Smooth B-rep from a finished part: the organic body as large B-spline
patches, the bores, bosses and flat faces as true cylinders and planes.

step.py's default STEP keeps the exact shapes exact and facets the rest.
Facets are fine to print from and to mate against, but a CAD user cannot
fillet, offset or shell them. This module works on the finished mesh, after
the booleans, so no OpenCascade boolean ever has to cut a B-spline:

1. label: every triangle lying on one of the plan's exact surfaces (a box
   face, a bore, a boss, a cap, a sphere) gets that surface's label; the
   rest are organic (label 0);
2. coarsen: decimate.py's passes, no field, with a rule that keeps the
   seams: a vertex only collapses into a neighbour carrying all its labels,
   seam vertices only along their seam, corners (three labels) never. Every
   coarse vertex is one of the mesh's own vertices, on the surface;
3. edges: a coarse edge on a seam is a B-spline through the mesh's seam
   path between its ends, an edge inside the organic body one through the
   shortest path over the surface, an edge inside an exact face a curve on
   that surface;
4. faces: a coarse triangle on an exact surface is that surface trimmed by
   its three edges (they all merge back into one face per surface later);
   an organic one is a plate-filled B-spline surface through its edges and
   through points of the mesh inside it;
5. sew, make the solid, merge same-surface faces.

The result is checked (valid, volume against the mesh) by the caller,
which falls back to facets if it is not right.
"""
from __future__ import annotations

import numpy as np

__all__ = ["surfaces", "label", "coarsen", "smooth_brep"]

TOL = 0.005          # mm: a vertex this close to an exact surface lies on it
PATCHES = 1500       # organic B-spline patches
DEBUG = False
SEW = 0.01           # mm: sewing tolerance (exact faces' seams sit within TOL of the patches')
SKIP_FAILED = False
ASPECT = 0.1         # least 2*area / longest edge^2 of a patch (equilateral 0.87)
LEAN = 0.5           # least cosine between a patch's plane and the surface normal at its corners
TURN = 0.5           # least cosine between a coarsened face's normal and its old one


class Surf:
    """An exact surface piece: plane (point, normal, in-bounds test),
    cylinder (p0, axis, r, axial range) or sphere (center, r)."""

    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    def key(self):
        if self.kind == "plane":
            n = self.n if self.n[np.argmax(np.abs(self.n))] > 0 else -self.n
            return ("plane", *np.round(n, 6), round(float(self.p @ n), 5))
        if self.kind == "cylinder":
            a = self.axis if self.axis[np.argmax(np.abs(self.axis))] > 0 else -self.axis
            q = self.p0 - (self.p0 @ a) * a
            return ("cylinder", *np.round(a, 6), *np.round(q, 5), round(self.r, 5))
        return ("sphere", *np.round(self.c, 5), round(self.r, 5))

    def dist(self, P):
        if self.kind == "plane":
            return np.abs((P - self.p) @ self.n)
        if self.kind == "cylinder":
            d = P - self.p0
            t = d @ self.axis
            return np.abs(np.linalg.norm(d - t[:, None] * self.axis, axis=1) - self.r)
        return np.abs(np.linalg.norm(P - self.c, axis=1) - self.r)

    def normal(self, P):
        if self.kind == "plane":
            return np.broadcast_to(self.n, P.shape)
        if self.kind == "cylinder":
            d = P - self.p0
            d = d - (d @ self.axis)[:, None] * self.axis
            return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        d = P - self.c
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    def inside(self, P, tol=0.02):
        if self.kind == "plane" and self.box is not None:
            lo, hi = self.box
            return np.all((P >= lo - tol) & (P <= hi + tol), axis=1)
        if self.kind == "plane" and self.disk is not None:
            c, r0, r1 = self.disk
            d = np.linalg.norm(P - c - ((P - c) @ self.n)[:, None] * self.n, axis=1)
            return (d >= r0 - tol) & (d <= r1 + tol)
        if self.kind == "cylinder":
            t = (P - self.p0) @ self.axis
            return (t >= -tol) & (t <= self.length + tol)
        return np.ones(len(P), bool)

    def geom(self):
        from OCP.Geom import Geom_CylindricalSurface, Geom_Plane, Geom_SphericalSurface
        from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt

        if self.kind == "plane":
            return Geom_Plane(gp_Pnt(*map(float, self.p)), gp_Dir(*map(float, self.n)))
        if self.kind == "cylinder":
            return Geom_CylindricalSurface(gp_Ax3(gp_Pnt(*map(float, self.p0)), gp_Dir(*map(float, self.axis))),
                                           float(self.r))
        return Geom_SphericalSurface(gp_Ax3(gp_Pnt(*map(float, self.c)), gp_Dir(0, 0, 1)), float(self.r))


def _shape_surfaces(shape, out):
    from .shapes import Box, Cylinder, Shell, Sphere, Union

    if isinstance(shape, Box):
        for a in range(3):
            n = np.zeros(3)
            n[a] = 1.0
            for at in (shape.lo, shape.hi):
                lo, hi = shape.lo.copy(), shape.hi.copy()
                lo[a] = hi[a] = at[a]
                out.append(Surf("plane", p=at.copy(), n=n.copy(), box=(lo, hi), disk=None))
    elif isinstance(shape, Cylinder):
        out.append(Surf("cylinder", p0=shape.p0.copy(), axis=shape.axis.copy(), r=shape.r, length=shape.length))
        if shape.r_inner > 0:
            out.append(Surf("cylinder", p0=shape.p0.copy(), axis=shape.axis.copy(), r=shape.r_inner,
                            length=shape.length))
        for c in (shape.p0, shape.p1):
            out.append(Surf("plane", p=c.copy(), n=shape.axis.copy(), box=None, disk=(c.copy(), shape.r_inner, shape.r)))
    elif isinstance(shape, Sphere):
        out.append(Surf("sphere", c=np.asarray(shape.center, float), r=float(shape.r)))
    elif isinstance(shape, Union):
        for s in shape.shapes:
            _shape_surfaces(s, out)
    elif isinstance(shape, Shell):
        for s in (shape.outer, shape.inner, shape.clip):
            if s is not None:
                _shape_surfaces(s, out)
    return out


def surfaces(plan):
    """The exact surfaces of a CSG plan's exact leaves, one per distinct
    plane, cylinder or sphere (pieces of one plane share it)."""
    from .sharp import Exact, _leaves

    out = []
    for lf in _leaves(plan):
        if isinstance(lf, Exact):
            _shape_surfaces(lf.shape, out)
    return out


def label(mesh, surfs, tol=TOL):
    """Per face: 0 for organic, k + 1 for a face lying on surfs[k] (all three
    corners within `tol`, inside its bounds, normal along the surface's).
    Surfaces with the same geometry share a label."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    C = V[F].mean(axis=1)
    fn = np.asarray(mesh.face_normals)
    lab = np.zeros(len(F), np.int64)
    best = np.full(len(F), np.inf)
    keys = {}
    for k, s in enumerate(surfs):
        lk = keys.setdefault(s.key(), k + 1)
        dv = s.dist(V)
        ok = (dv < tol) & s.inside(V)
        on = ok[F].all(axis=1)
        if not on.any():
            continue
        i = np.flatnonzero(on)
        para = np.abs(np.einsum("ij,ij->i", fn[i], s.normal(C[i]))) > 0.995
        i = i[para]
        d = dv[F[i]].sum(axis=1)
        better = d < best[i]
        lab[i[better]] = lk
        best[i[better]] = d[better]
    return lab


def _vertex_sets(F, lab, n):
    K = int(lab.max()) + 1 if len(lab) else 1
    pairs = np.unique(F.ravel().astype(np.int64) * K + np.repeat(lab, 3))
    v, l = pairs // K, pairs % K
    nl = np.bincount(v, minlength=n)
    h = np.zeros(n)
    rnd = np.random.default_rng(7).random(K) + 1.0
    np.add.at(h, v, rnd[l])
    one = np.full(n, -1)
    one[v[nl[v] == 1]] = l[nl[v] == 1]
    return K, pairs, nl, h, one


def _edge_faces(F):
    """(edges a<b, face on each side): closed manifold, so each edge has two."""
    a = np.r_[F[:, 0], F[:, 1], F[:, 2]]
    b = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    f = np.tile(np.arange(len(F)), 3)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * (int(F.max()) + 1) + hi
    o = np.argsort(key, kind="stable")
    key, f = key[o], f[o]
    first = np.r_[True, key[1:] != key[:-1]]
    i = np.flatnonzero(first)
    j = i + 1
    j = np.minimum(j, len(key) - 1)
    E = np.stack([lo[o][i], hi[o][i]], axis=1)
    return E, key[i], f[i], f[j]


def coarsen(mesh, lab, patches=PATCHES, progress=None):
    """The labelled mesh coarsened until about `patches` organic triangles
    remain and the exact regions are as coarse as their seams allow. Returns
    (vertex indices into the mesh, faces, labels)."""
    from .decimate import decimate

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    n = len(V)
    st = {}
    rng = np.random.default_rng(0)

    def allow(E, Fc, lb):
        K, pairs, nl, h, one = _vertex_sets(Fc, lb, n)
        a, b = E[:, 0], E[:, 1]
        a_in_b = (nl[a] == 1) & np.isin(b * K + one[a], pairs)
        b_in_a = (nl[b] == 1) & np.isin(a * K + one[b], pairs)
        ok = a_in_b | b_in_a
        # along a seam: both ends on the same two surfaces, the edge between them
        both2 = (nl[a] == 2) & (nl[b] == 2) & (h[a] == h[b])
        if both2.any():
            Ee, key, f0, f1 = _edge_faces(Fc)
            kq = np.minimum(a, b).astype(np.int64) * (int(Fc.max()) + 1) + np.maximum(a, b)
            pos = np.clip(np.searchsorted(key, kq), 0, len(key) - 1)
            seam = (key[pos] == kq) & (lb[f0[pos]] != lb[f1[pos]])
            ok |= both2 & seam
        # organic triangles: stop at the budget
        org = (one[a] == 0) | (one[b] == 0)
        n_org = int((lb == 0).sum())
        room = max(0, (n_org - int(patches)) // 2)
        oi = np.flatnonzero(ok & org)
        if len(oi) > 4 * room:               # near the budget: a spread-out few, not a cluster
            ok[rng.choice(oi, len(oi) - 4 * room, replace=False)] = False
        st.update(nl=nl, one=one, K=K, pairs=pairs)
        return ok

    def place(W, a, b):
        nl, one, K, pairs = st["nl"], st["one"], st["K"], st["pairs"]
        a_in_b = (nl[a] == 1) & np.isin(b * K + one[a], pairs)
        return np.where(a_in_b[:, None], W[b], W[a])

    from scipy.spatial import cKDTree

    tree = cKDTree(V)
    vn = np.asarray(mesh.vertex_normals)

    def shape(T):
        e = np.stack([T[:, 1] - T[:, 0], T[:, 2] - T[:, 1], T[:, 0] - T[:, 2]], axis=1)
        n = np.cross(e[:, 0], -e[:, 2])
        area2 = np.linalg.norm(n, axis=1)
        q = area2 / np.maximum((e ** 2).sum(axis=2).max(axis=1), 1e-300)
        n = n / np.maximum(area2[:, None], 1e-300)
        _, i = tree.query(T.reshape(-1, 3))
        lean = np.einsum("kj,kcj->kc", n, vn[i].reshape(-1, 3, 3)).min(axis=1)
        return q, lean

    def quality(Told, Tnew, lb):
        """No slivers, and no patch leaning far off the surface at its
        corners (a big patch over a tight bend fills badly), unless it was
        already that bad. Exact faces merge back whole: any shape will do."""
        q0, l0 = shape(Told)
        q1, l1 = shape(Tnew)
        return (((q1 > ASPECT) | (q1 >= q0)) & ((l1 > LEAN) | (l1 >= l0))) | (lb > 0)

    V2, F2, stats, _ = decimate(V, F, None, 0.0, 1.0, max_passes=400, min_gain=0.0, progress=progress,
                                place=place, turn=TURN, labels=lab, allow=allow, max_reject=40, quality=quality)
    # every coarse vertex sits where one of the mesh's vertices is: which one
    d, vid = tree.query(V2)
    assert d.max() < 1e-9, d.max()
    return vid, F2, stats["labels"]


def _graph(mesh, mask_v=None, mask_e=None):
    import scipy.sparse as sp

    E = np.asarray(mesh.edges_unique)
    w = np.linalg.norm(mesh.vertices[E[:, 0]] - mesh.vertices[E[:, 1]], axis=1)
    keep = np.ones(len(E), bool)
    if mask_v is not None:
        keep &= mask_v[E[:, 0]] & mask_v[E[:, 1]]
    if mask_e is not None:
        keep &= mask_e
    n = len(mesh.vertices)
    return sp.coo_matrix((w[keep], (E[keep, 0], E[keep, 1])), shape=(n, n)).tocsr()


def _path(G, a, b, limit):
    from scipy.sparse.csgraph import dijkstra

    d, pred = dijkstra(G, directed=False, indices=a, return_predecessors=True, limit=limit)
    if not np.isfinite(d[b]):
        return None
    p = [b]
    while p[-1] != a:
        p.append(pred[p[-1]])
        if p[-1] < 0:
            return None
    return p[::-1]


def _fit(P, tol):
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.OCP.collections import Array1_gp_Pnt

    from .step import _pnt

    if len(P) > 40:                          # a long path: keep the ends, thin the middle
        i = np.unique(np.r_[np.linspace(0, len(P) - 1, 40).round().astype(int)])
        P = P[i]
    if len(P) == 2:
        P = np.vstack([P[0], P.mean(axis=0), P[1]])
    arr = Array1_gp_Pnt(1, len(P))
    for j, p in enumerate(P):
        arr.SetValue(j + 1, _pnt(p))
    return GeomAPI_PointsToBSpline(arr, 2, 5, GeomAbs_C2, tol).Curve()


def smooth_brep(mesh, plan, patches=PATCHES, order=3, progress=None):
    """The part as an OpenCascade solid: exact faces on the plan's planes,
    cylinders and spheres, smooth B-spline patches elsewhere. Returns
    (solid, report). Raises if the construction fails."""
    import time

    import trimesh
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid,
                                    BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing)
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0
    from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Solid
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from .step import _pnt, _volume

    t0 = time.time()
    surfs = surfaces(plan) if plan is not None else []
    lab = label(mesh, surfs)
    keys = {}
    geo = {}
    for k, s in enumerate(surfs):
        lk = keys.setdefault(s.key(), k + 1)
        geo.setdefault(lk, s)
    vid, F, flab = coarsen(mesh, lab, patches)
    V = np.asarray(mesh.vertices, float)
    CV = V[vid]
    t1 = time.time()

    # fine-mesh graphs for the paths
    fl = lab
    Ef, keyf, f0, f1 = _edge_faces(np.asarray(mesh.faces, np.int64))
    seam_e = fl[f0] != fl[f1]
    org_v = np.zeros(len(V), bool)
    org_v[np.asarray(mesh.faces)[fl == 0].ravel()] = True
    import scipy.sparse as sp

    def graph(mask):
        w = np.linalg.norm(V[Ef[mask, 0]] - V[Ef[mask, 1]], axis=1)
        return sp.coo_matrix((w, (Ef[mask, 0], Ef[mask, 1])), shape=(len(V), len(V))).tocsr()

    Gseam = graph(seam_e)
    Gorg = graph(org_v[Ef[:, 0]] & org_v[Ef[:, 1]])

    CE, ckey, cf0, cf1 = _edge_faces(F)
    N = int(order) + 1                              # points per patch side
    org_mesh = trimesh.Trimesh(V, np.asarray(mesh.faces)[fl == 0], process=False)

    def pull(P, nrm):
        """Points onto the organic surface, where its normal there agrees."""
        if not len(P):
            return P
        cp, _, tri = trimesh.proximity.closest_point(org_mesh, P)
        ok = np.einsum("ij,ij->i", org_mesh.face_normals[tri], nrm) > 0.3
        return np.where(ok[:, None], cp, P)

    # every coarse edge: a polyline over the surface, and its midpoint by length
    poly, split = {}, {}
    for (a, b), la, lb in zip(CE, flab[cf0], flab[cf1]):
        a, b = int(a), int(b)
        pa, pb = CV[a], CV[b]
        L = float(np.linalg.norm(pa - pb))
        if la == lb and la > 0:
            t = np.linspace(0, 1, 9)[:, None]
            P = _onto(geo[int(la)], pa * (1 - t) + pb * t)
            P[0], P[-1] = pa, pb
        else:
            G = Gseam if la != lb else Gorg
            pth = _path(G, int(vid[a]), int(vid[b]), limit=4 * L + 1.0)
            if pth is None:
                raise RuntimeError("no surface path for a patch edge")
            P = V[pth]
        poly[(a, b)] = P
        split[(a, b)] = la == 0 or lb == 0        # organic on a side: two half edges

    def resample(P, n):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if d[-1] <= 0:
            return np.repeat(P[:1], n, axis=0)
        t = np.linspace(0, d[-1], n)
        return np.stack([np.interp(t, d, P[:, k]) for k in range(3)], axis=1)

    def half(a, b):
        """Points from corner a to the midpoint of edge (a, b)."""
        P = poly[(min(a, b), max(a, b))]
        if a > b:
            P = P[::-1]
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        m = 0.5 * d[-1]
        i = int(np.searchsorted(d, m))
        w = (m - d[i - 1]) / max(d[i] - d[i - 1], 1e-300)
        M = P[i - 1] * (1 - w) + P[i] * w
        return resample(np.vstack([P[:i], M]), N)

    Bi = np.linalg.inv(_bernstein(N - 1))           # points at even parameters -> Bezier poles

    def bezier_curve(P):
        from OCP.Geom import Geom_BezierCurve
        from OCP.OCP.collections import Array1_gp_Pnt

        C = Bi @ P
        arr = Array1_gp_Pnt(1, len(C))
        for j, c in enumerate(C):
            arr.SetValue(j + 1, _pnt(c))
        return Geom_BezierCurve(arr)

    def bezier_face(Q):
        from OCP.Geom import Geom_BezierSurface
        from OCP.OCP.collections import Array2_gp_Pnt

        C = np.einsum("ui,ijk,vj->uvk", Bi, Q, Bi)
        arr = Array2_gp_Pnt(1, N, 1, N)
        for i in range(N):
            for j in range(N):
                arr.SetValue(i + 1, j + 1, _pnt(C[i, j]))
        return BRepBuilderAPI_MakeFace(Geom_BezierSurface(arr), 1e-7).Face()

    edge_cache = {}

    def edge(key, P):
        if key not in edge_cache:
            edge_cache[key] = BRepBuilderAPI_MakeEdge(bezier_curve(P) if len(P) == N else _fit(P, 1e-5)).Edge()
        return edge_cache[key]

    t2 = time.time()
    sew = BRepBuilderAPI_Sewing(SEW)
    u = np.linspace(0, 1, N)
    nq = 0
    for f, lf in zip(F, flab):
        f = [int(x) for x in f]
        if lf > 0:
            w = BRepBuilderAPI_MakeWire()
            for k in range(3):
                a, b = f[k], f[(k + 1) % 3]
                e = (min(a, b), max(a, b))
                if split[e]:
                    for x, y in ((a, b), (b, a)):
                        w.Add(edge(("h", x, e), half(x, y)))
                else:
                    w.Add(edge(("e", e), poly[e]))
            mf = BRepBuilderAPI_MakeFace(geo[int(lf)].geom(), w.Wire(), True)
            if not mf.IsDone():
                raise RuntimeError("an exact face failed")
            fx = ShapeFix_Face(mf.Face())
            fx.Perform()
            sew.Add(fx.Face())
            continue
        T = CV[f]
        n = np.cross(T[1] - T[0], T[2] - T[0])
        n /= max(np.linalg.norm(n), 1e-300)
        H = {(x, y): half(x, y) for x in f for y in f if x != y}
        Mid = {(min(x, y), max(x, y)): H[(x, y)][-1] for (x, y) in H}
        Gc = pull(np.mean([*T, *Mid.values()], axis=0)[None], n[None])[0]
        spoke = {}
        for e, M in Mid.items():                 # centroid -> edge midpoint
            P = Gc * (1 - u[:, None]) + M * u[:, None]
            P[1:-1] = pull(P[1:-1], np.repeat(n[None], N - 2, axis=0))
            spoke[e] = P
        for k in range(3):
            c, j, i = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            e1, e2 = (min(c, j), max(c, j)), (min(c, i), max(c, i))
            bottom = H[(c, j)]                   # c -> M(c, j)
            left = H[(c, i)]                     # c -> M(c, i)
            right = spoke[e1][::-1]              # M(c, j) -> G
            top = spoke[e2][::-1]                # M(c, i) -> G
            Q = np.zeros((N, N, 3))
            Q[:, 0], Q[:, -1], Q[0, :], Q[-1, :] = bottom, top, left, right
            # Coons blend of the four sides, then onto the surface
            U, W = np.meshgrid(u, u, indexing="ij")
            C = ((1 - W)[..., None] * bottom[:, None] + W[..., None] * top[:, None]
                 + (1 - U)[..., None] * left[None] + U[..., None] * right[None]
                 - ((1 - U) * (1 - W))[..., None] * Q[0, 0] - (U * (1 - W))[..., None] * Q[-1, 0]
                 - ((1 - U) * W)[..., None] * Q[0, -1] - (U * W)[..., None] * Q[-1, -1])
            inner = C[1:-1, 1:-1].reshape(-1, 3)
            Q[1:-1, 1:-1] = pull(inner, np.repeat(n[None], len(inner), axis=0)).reshape(N - 2, N - 2, 3)
            sew.Add(bezier_face(Q))
            nq += 1
    sew.Perform()
    t3 = time.time()
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    shells = []
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if DEBUG:
        print("quads", nq, "shells", len(shells), "free", sew.NbFreeEdges(), "t", round(time.time() - t0, 1), flush=True)
    if len(shells) != 1 or sew.NbFreeEdges():
        raise RuntimeError(f"patches did not close ({len(shells)} shells, {sew.NbFreeEdges()} free edges)")
    fx = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(shells[0]).Solid())
    fx.Perform()
    solid = fx.Solid()
    if _volume(solid) < 0:
        solid.Reverse()
    u = ShapeUpgrade_UnifySameDomain(solid, False, True, False)
    u.Build()
    solid = u.Shape()
    rep = {"patches": nq, "triangles": int((flab == 0).sum()), "exactTriangles": int((flab > 0).sum()),
           "exactSurfaces": int(len(set(int(x) for x in np.unique(lab) if x > 0))),
           "seconds": {"coarsen": round(t1 - t0, 1), "edges": round(t2 - t1, 1), "faces": round(t3 - t2, 1),
                       "solid": round(time.time() - t3, 1)}}
    return solid, rep


def _bernstein(d):
    """B[i, j] = j-th Bernstein polynomial of degree d at i / d."""
    from math import comb

    t = np.linspace(0, 1, d + 1)
    return np.array([[comb(d, j) * x ** j * (1 - x) ** (d - j) for j in range(d + 1)] for x in t])


def _onto(s, P):
    """Points pulled onto exact surface `s`."""
    if s.kind == "plane":
        return P - ((P - s.p) @ s.n)[:, None] * s.n
    if s.kind == "cylinder":
        d = P - s.p0
        t = d @ s.axis
        r = d - t[:, None] * s.axis
        return s.p0 + t[:, None] * s.axis + r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12) * s.r
    d = P - s.c
    return s.c + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12) * s.r
