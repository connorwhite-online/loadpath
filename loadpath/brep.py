"""
Smooth B-rep from a finished part: the organic body as large Bezier
patches, the bores, bosses and flat faces as true cylinders and planes.

step.py's default STEP keeps the exact shapes exact and facets the rest.
Facets are fine to print from and to mate against, but a CAD user cannot
fillet, offset or shell them. This module works on the finished mesh, after
the booleans, so no OpenCascade boolean ever has to cut a smooth surface:

1. label: every triangle lying on one of the plan's exact surfaces (a box
   face, a bore, a boss, a cap) gets that surface's label; the organic rest
   splits into smooth regions at creases; slivers join their surroundings;
2. coarsen: decimate.py's passes, no field, with a rule that keeps the
   seams (a vertex only collapses into a neighbour carrying all its labels,
   seam vertices only along their seam, corners never) and relaxation
   rounds between them. Organic triangles stay well shaped and do not lean
   far off the surface, so a patch never wraps a tight bend;
3. edges: a coarse edge on a seam follows the mesh's seam path, one inside
   the organic body is its chord draped evenly over the surface; both
   become Bezier curves of the patch degree (5 by default);
4. faces: organic triangles pair up into quads, one patch each (the odd
   ones out split into three quads), its sides those curves and its inside
   interpolating points pulled onto the mesh along the patch normal; each
   connected exact region is one face on its surface, bounded by its seams;
5. sew, make the solid.

The caller (step.export_step, surface="smooth") checks the result (valid,
volume against the mesh) and falls back to facets if it is not right.
"""
from __future__ import annotations

import numpy as np

__all__ = ["coarsen", "label", "smooth_brep", "surfaces"]

TOL = 0.005          # mm: a vertex this close to an exact surface lies on it
PATCHES = 2000       # organic triangles to start the layout from (two make one patch; refined where it misses)
RAY_MAX = 3.0        # mm: farthest a patch point is pulled along its normal onto the surface
CREASE = 40.0        # degrees: organic faces meeting sharper than this are separate smooth regions
SEW = 0.01           # mm: sewing tolerance (exact faces' seams sit within TOL of the patches')
ASPECT = 0.1         # least 2*area / longest edge^2 of a patch (equilateral 0.87)
LEAN = 0.5           # least cosine between a patch's plane and the surface normal at its corners
REFINE = 0.07        # mm: a patch or edge missing the mesh by more is split
REFINE_ROUNDS = 8
MIN_SPLIT = 0.3      # mm: shortest edge that splits
ISLAND = 1.0         # mm^2: label regions smaller than this, inside one other region, join it
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



def _closest(mesh, P, edge=0.5, k=16):
    """Closest points on a mesh to P and the normals there: trimesh's
    closest_point gathers candidates by triangle bounding boxes, and an
    exact boolean's slivers (a bore wall the length of the bore) make every
    point a candidate of all of them, minutes on a part; here the mesh is
    split to `edge` and each point tries the k nearest centroids."""
    import trimesh
    from scipy.spatial import cKDTree

    from .fields import _closest_on_triangles

    cache = getattr(mesh, "_loadpath_closest", None)
    if cache is None or cache[0] != edge:
        v, f = trimesh.remesh.subdivide_to_size(np.asarray(mesh.vertices, float),
                                                np.asarray(mesh.faces, np.int64), max_edge=edge, max_iter=16)
        tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
        cache = (edge, tri, nrm, cKDTree(tri.mean(axis=1)))
        mesh._loadpath_closest = cache
    _, tri, nrm, tree = cache
    P = np.asarray(P, float).reshape(-1, 3)
    k = min(k, len(tri))
    out, nout = np.empty_like(P), np.empty_like(P)
    for j in range(0, len(P), 100_000):
        q = P[j:j + 100_000]
        cand = tree.query(q, k=k)[1].reshape(len(q), k)
        T = tri[cand]
        Q = np.repeat(q[:, None], k, axis=1).reshape(-1, 3)
        C = _closest_on_triangles(Q, T[:, :, 0].reshape(-1, 3), T[:, :, 1].reshape(-1, 3),
                                  T[:, :, 2].reshape(-1, 3)).reshape(len(q), k, 3)
        b = np.argmin(np.sum((C - q[:, None]) ** 2, axis=2), axis=1)
        r = np.arange(len(q))
        out[j:j + len(q)] = C[r, b]
        nout[j:j + len(q)] = nrm[cand[r, b]]
    return out, nout

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


def creases(mesh, lab, O, angle=CREASE):
    """Organic faces (label 0) split into smooth regions at creases (faces
    meeting at more than `angle` degrees), labelled O, O + 1, ...: a patch
    must not round over a sharp edge of the field."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    org = lab == 0
    A = np.asarray(mesh.face_adjacency)
    keep = org[A[:, 0]] & org[A[:, 1]] & (np.asarray(mesh.face_adjacency_angles) < np.radians(angle))
    n = len(lab)
    G = sp.coo_matrix((np.ones(keep.sum()), (A[keep, 0], A[keep, 1])), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    out = lab.copy()
    _, ids = np.unique(comp[org], return_inverse=True)
    out[org] = O + ids
    return out


def peninsulas(mesh, lab, O, rounds=4):
    """Exact faces poking into the organic body (two of their three
    neighbours organic, of one region) turn organic: a seam that juts out
    and back makes a face boundary cross itself."""
    A = np.asarray(mesh.face_adjacency)
    lab = lab.copy()
    for _ in range(rounds):
        nb = [[] for _ in range(len(lab))]
        for x, y in A:
            nb[x].append(y)
            nb[y].append(x)
        flip = []
        for f in np.flatnonzero(lab < O):
            other = [lab[g] for g in nb[f] if lab[g] != lab[f]]
            if len(other) >= 2 and other[0] >= O and all(o == other[0] for o in other):
                flip.append((f, other[0]))
        if not flip:
            break
        for f, l in flip:
            lab[f] = l
    return lab


def islands(mesh, lab, area=ISLAND, surfs=None, tol=0.02):
    """Small label regions (under `area` mm^2) wholly surrounded by one other
    label take that label: a sliver of organic faces inside a flat face (its
    normals too noisy for label()) would cut a slit into the plane."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    A = np.asarray(mesh.face_adjacency)
    fa = np.asarray(mesh.area_faces)
    n = len(lab)
    for _ in range(3):
        same = lab[A[:, 0]] == lab[A[:, 1]]
        G = sp.coo_matrix((np.ones(same.sum()), (A[same, 0], A[same, 1])), shape=(n, n))
        nc, comp = connected_components(G, directed=False)
        ca = np.bincount(comp, fa, minlength=nc)
        small = ca < area
        if not small.any():
            break
        d = ~same
        a, b = A[d, 0], A[d, 1]
        # neighbour labels of each small component
        cs = np.r_[comp[a], comp[b]]
        nb = np.r_[lab[b], lab[a]]
        m = small[cs]
        cs, nb = cs[m], nb[m]
        if not len(cs):
            break
        K = int(lab.max()) + 1
        u = np.unique(cs.astype(np.int64) * K + nb)
        uc, ul = u // K, u % K
        cnt = np.bincount(uc, minlength=nc)
        one = np.full(nc, -1)
        one[uc[cnt[uc] == 1]] = ul[cnt[uc] == 1]
        flip = small & (one >= 0)
        if surfs:                    # onto an exact surface only if the island lies on it
            V = np.asarray(mesh.vertices, float)
            F = np.asarray(mesh.faces)
            for c in np.flatnonzero(flip):
                t = int(one[c])
                if t in surfs:
                    P = V[F[comp == c].ravel()]
                    if surfs[t].dist(P).max() > tol:
                        flip[c] = False
        if not flip.any():
            break
        f = flip[comp]
        lab = lab.copy()
        lab[f] = one[comp[f]]
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


def coarsen(mesh, lab, patches=PATCHES, curved=None, surfs=None, O=None, progress=None):
    """The labelled mesh coarsened until about `patches` organic triangles
    remain and the exact regions are as coarse as their seams allow. Returns
    (vertex indices into the mesh, faces, labels)."""
    from .decimate import decimate

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    st = {}
    rng = np.random.default_rng(0)
    if O is None:                   # labels from O up are organic (0 alone, by default)
        O = 0
    if curved is None:
        curved = np.zeros(int(lab.max()) + 1, bool)

    def allow(E, Fc, lb):
        K, pairs, nl, h, one = _vertex_sets(Fc, lb, int(Fc.max()) + 1)
        a, b = E[:, 0], E[:, 1]
        a_in_b = (nl[a] == 1) & np.isin(b * K + one[a], pairs)
        b_in_a = (nl[b] == 1) & np.isin(a * K + one[b], pairs)
        ok = a_in_b | b_in_a
        # along a seam: both ends on the same two surfaces, the edge between them
        both2 = (nl[a] == 2) & (nl[b] == 2) & (h[a] == h[b])
        if both2.any():
            _, key, f0, f1 = _edge_faces(Fc)
            kq = np.minimum(a, b).astype(np.int64) * (int(Fc.max()) + 1) + np.maximum(a, b)
            pos = np.clip(np.searchsorted(key, kq), 0, len(key) - 1)
            seam = (key[pos] == kq) & (lb[f0[pos]] != lb[f1[pos]])
            ok |= both2 & seam
        # organic triangles: stop at the budget
        org = (one[a] >= O) | (one[b] >= O)
        n_org = int((lb >= O).sum())
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
        organic = ((q1 > ASPECT) | (q1 >= q0)) & ((l1 > LEAN) | (l1 >= l0))
        bend = (l1 > LEAN) | (l1 >= l0)             # a bore's triangle: under ~60 degrees of arc
        return np.where(lb >= O, organic, np.where(curved[np.minimum(lb, len(curved) - 1)], bend, True))

    import trimesh

    org = trimesh.Trimesh(V, F[lab >= O], process=False) if (lab >= O).any() else None
    W, Fc, lb = V, F, np.asarray(lab)
    for _ in range(60):
        before = len(Fc)
        W, Fc, stats, _ = decimate(W, Fc, None, 0.0, 1.0, max_passes=6, min_gain=0.0, progress=progress,
                                   place=place, turn=TURN, labels=lb, allow=allow, max_reject=40, quality=quality)
        lb = stats["labels"]
        if (lb >= O).sum() <= patches or len(Fc) > 0.995 * before:
            break
        W = _relax(W, Fc, lb, org, surfs or {}, O)
    # which coarse vertices still sit on one of the mesh's vertices (seams, corners)
    d, vid = tree.query(W)
    vid[d > 1e-9] = -1
    return W, vid, Fc, lb


def _relax(W, F, lb, org, surfs, O, rounds=2, step=0.5):
    """Even out the patch layout: each vertex inside one region moves halfway
    to its neighbours' average, back onto its surface (the organic mesh, or
    its exact surface). Seam and corner vertices stay put."""
    import trimesh

    n = len(W)
    _, _, nl, _, one = _vertex_sets(F, lb, n)
    free = (nl == 1)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    W = W.copy()
    for _ in range(rounds):
        S = np.zeros_like(W)
        c = np.zeros(n)
        np.add.at(S, E[:, 0], W[E[:, 1]])
        np.add.at(c, E[:, 0], 1)
        A = S / np.maximum(c, 1)[:, None]
        T = W + step * (A - W)
        o = free & (one >= O)
        if org is not None and o.any():
            cp, cn = _closest(org, T[o])
            vn = trimesh.Trimesh(W, F, process=False).vertex_normals[o]
            ok = np.einsum("ij,ij->i", cn, vn) > 0.5
            idx = np.flatnonzero(o)
            W[idx[ok]] = cp[ok]
        for k, sf in surfs.items():
            e = free & (one == k)
            if e.any():
                W[e] = _onto(sf, T[e])
    return W


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
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
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
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GeomAbs import GeomAbs_BezierSurface
    from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL
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
    curved = np.zeros(int(lab.max()) + 1, bool)
    for lk, sf in geo.items():
        if lk < len(curved):
            curved[lk] = sf.kind != "plane"
    O = (max(geo) if geo else 0) + 1
    lab = islands(mesh, peninsulas(mesh, creases(mesh, lab, O), O))
    CV, vid, F, flab = coarsen(mesh, lab, patches, curved, geo, O)
    # a coarse patch can end up an island inside another face (a sliver of the
    # body on a flat face): it joins that face
    flab = islands(trimesh.Trimesh(CV, F, process=False), flab, ISLAND, geo)
    V = np.asarray(mesh.vertices, float)
    t1 = time.time()

    # fine-mesh graphs for the paths
    fl = lab
    Ef, _, f0, f1 = _edge_faces(np.asarray(mesh.faces, np.int64))
    seam_e = fl[f0] != fl[f1]
    import scipy.sparse as sp

    CE, ckey, cf0, cf1 = _edge_faces(F)
    N = int(order) + 1                              # points per patch side
    # the whole surface is the target: a patch near a seam lands on the true
    # surface either side, and a demoted exact region (below) is patches too
    org_mesh = trimesh.Trimesh(V, np.asarray(mesh.faces), process=False)

    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector  # embreex: ~100x faster
    except ImportError:
        from trimesh.ray.ray_triangle import RayMeshIntersector

    rays = RayMeshIntersector(org_mesh)

    def pull(P, nrm, reach=None):
        """Points onto the organic surface along the patch normal (the nearer
        hit either way, where the surface faces the same way); else the
        closest point, where its normal agrees; else left where they are."""
        if not len(P):
            return P
        best = np.full(len(P), np.inf)
        out = P.copy()
        lim = np.full(len(P), RAY_MAX) if reach is None else np.minimum(reach, RAY_MAX)
        for sgn in (1.0, -1.0):
            loc, ri, ti = rays.intersects_location(P, sgn * nrm, multiple_hits=False)
            if not len(ri):
                continue
            d = np.linalg.norm(loc - P[ri], axis=1)
            ok = (np.einsum("ij,ij->i", org_mesh.face_normals[ti], nrm[ri]) > 0.3) & (d < lim[ri]) & (d < best[ri])
            out[ri[ok]] = loc[ok]
            best[ri[ok]] = d[ok]
        miss = ~np.isfinite(best)
        if miss.any():
            cp, cn = _closest(org_mesh, P[miss])
            idx = np.flatnonzero(miss)
            # where it faces the same way; else (a layout triangle wrapped
            # round a thin wall's edge, its normal side-on to the surface)
            # wherever it is within reach: a point left off the surface
            # drags its whole patch off
            ok = (np.einsum("ij,ij->i", cn, nrm[miss]) > 0.3) | (np.linalg.norm(cp - P[miss], axis=1) < lim[miss])
            out[idx[ok]] = cp[ok]
        return out

    def resample(P, n):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if d[-1] <= 0:
            return np.repeat(P[:1], n, axis=0)
        t = np.linspace(0, d[-1], n)
        return np.stack([np.interp(t, d, P[:, k]) for k in range(3)], axis=1)

    def drape(P, nrm, rounds=4):
        """Polylines (m, n, 3) between pairs of surface points onto the
        surface, evenly spaced: pulling a chord along one direction bunches
        its points up at a bend, and the gaps between them cut the corner."""
        P = np.array(P, float)
        m, n = P.shape[:2]
        R = np.repeat(np.asarray(nrm, float)[:, None], n - 2, axis=1).reshape(-1, 3)
        # the surface under a chord is never further from it than half its
        # length: a hit beyond is another sheet (a thin wall's far side)
        reach = np.repeat(0.5 * np.linalg.norm(P[:, -1] - P[:, 0], axis=1) + TOL, n - 2)
        for k in range(rounds + 1):
            P[:, 1:-1] = pull(P[:, 1:-1].reshape(-1, 3), R, reach).reshape(m, n - 2, 3)
            if k < rounds:
                P = np.stack([resample(x, n) for x in P])
        return P

    def build(attempt, refine=False):
        """Edges and faces from the coarse layout: (sewing, counts), or
        (None, regions to demote) when an exact region needs patches."""
        # every coarse edge: a polyline over the surface
        poly, split, fl_edge = {}, {}, {}
        cfn = np.cross(CV[F[:, 1]] - CV[F[:, 0]], CV[F[:, 2]] - CV[F[:, 0]])
        cfn /= np.maximum(np.linalg.norm(cfn, axis=1, keepdims=True), 1e-300)
        # seam paths over the seam vertices alone (a small graph: fast Dijkstra)
        sv = np.unique(Ef[seam_e])
        loc = np.full(len(V), -1)
        loc[sv] = np.arange(len(sv))
        w = np.linalg.norm(V[Ef[seam_e, 0]] - V[Ef[seam_e, 1]], axis=1)
        Gs = sp.coo_matrix((w, (loc[Ef[seam_e, 0]], loc[Ef[seam_e, 1]])), shape=(len(sv), len(sv))).tocsr()
        inner, t17 = [], np.linspace(0, 1, 17)[:, None]
        for (a, b), la, lb, f0e, f1e in zip(CE, flab[cf0], flab[cf1], cf0, cf1):
            a, b = int(a), int(b)
            pa, pb = CV[a], CV[b]
            L = float(np.linalg.norm(pa - pb))
            if la == lb and la < O:
                t = np.linspace(0, 1, 9)[:, None]
                P = _onto(geo[int(la)], pa * (1 - t) + pb * t)
                P[0], P[-1] = pa, pb
            elif la == lb:
                # inside the organic body: the chord draped over the surface (the
                # shortest path over the mesh's edges is a staircase, and two
                # from one corner can share their first half); below, all at once
                nn = cfn[f0e] + cfn[f1e]
                inner.append(((a, b), pa * (1 - t17) + pb * t17, nn / max(np.linalg.norm(nn), 1e-300)))
                P = None
            else:
                pth = _path(Gs, int(loc[vid[a]]), int(loc[vid[b]]), limit=4 * L + 1.0) if vid[a] >= 0 and vid[b] >= 0 else None
                if pth is not None:
                    P = V[sv[pth]]
                elif la >= O and lb >= O:            # between two patched regions: drape it
                    nn = cfn[f0e] + cfn[f1e]
                    inner.append(((a, b), pa * (1 - t17) + pb * t17, nn / max(np.linalg.norm(nn), 1e-300)))
                    P = None
                else:
                    raise RuntimeError(f"no surface path for a seam edge ({vid[a]}, {vid[b]}, labels {la}, {lb})")
            poly[(a, b)] = P
            split[(a, b)] = False                     # set below: next to a three-quad triangle, two half edges
            fl_edge[(a, b)] = la >= O or lb >= O      # organic on a side: a Bezier edge
        if inner:
            Pd = drape(np.stack([x[1] for x in inner]), np.stack([x[2] for x in inner]))
            for (e, _, _), P in zip(inner, Pd):
                poly[e] = P

        D = N - 1                                       # patch and edge degree
        Bm = _bernstein(D)                              # Bm[i, j]: j-th Bernstein at i / D
        Bi = np.linalg.inv(Bm)                          # points at even parameters -> poles
        Bint = np.linalg.inv(Bm[1:-1, 1:-1])

        # organic triangles pair up into quads (one patch for two triangles);
        # the rest split into three quads each
        pair = _pairs(CV, F, flab, O)
        single = np.ones(len(F), bool)
        single[[i for i, j in pair]] = False
        single[[j for i, j in pair]] = False
        single &= flab >= O
        for i in np.flatnonzero(single):
            for k in range(3):
                a, b = int(F[i, k]), int(F[i, (k + 1) % 3])
                split[(min(a, b), max(a, b))] = True

        poles = {}

        def curve(a, b):
            """Poles of coarse edge (a, b), from a to b."""
            e = (min(a, b), max(a, b))
            if e not in poles:
                poles[e] = Bi @ resample(poly[e], N)
            return poles[e] if a < b else poles[e][::-1]

        def casteljau(C):
            """Poles of the two halves (t <= 1/2, t >= 1/2) of a Bezier curve."""
            L, R = [C[0]], [C[-1]]
            W = C.copy()
            for _ in range(D):
                W = 0.5 * (W[:-1] + W[1:])
                L.append(W[0])
                R.append(W[-1])
            return np.array(L), np.array(R[::-1])

        def half(a, b):
            """Poles from corner a to the midpoint of edge (a, b)."""
            return casteljau(curve(a, b))[0]

        def value(C, t):
            from math import comb

            t = np.atleast_1d(t)
            B = np.array([[comb(D, j) * x ** j * (1 - x) ** (D - j) for j in range(D + 1)] for x in t])
            return B @ C

        def make_curve(C):
            from OCP.Geom import Geom_BezierCurve
            from OCP.OCP.collections import Array1_gp_Pnt

            arr = Array1_gp_Pnt(1, len(C))
            for j, c in enumerate(C):
                arr.SetValue(j + 1, _pnt(c))
            return Geom_BezierCurve(arr)

        def coons(bottom, top, left, right):
            """Points of the Coons blend of a patch's four sides (poles: bottom,
            top along u; left, right along v; corners shared) at even parameters."""
            sb, st_, sl, sr = (Bm @ x for x in (bottom, top, left, right))
            U, W = np.meshgrid(u, u, indexing="ij")
            X = ((1 - W)[..., None] * sb[:, None] + W[..., None] * st_[:, None]
                 + (1 - U)[..., None] * sl[None] + U[..., None] * sr[None]
                 - ((1 - U) * (1 - W))[..., None] * sb[0] - (U * (1 - W))[..., None] * sb[-1]
                 - ((1 - U) * W)[..., None] * st_[0] - (U * W)[..., None] * st_[-1])
            return X

        def poles_of(bottom, top, left, right, X):
            """Poles of the Bezier patch with those sides through points X
            (N, N, 3): its inside poles solved for, so it passes through them."""
            C = np.zeros((N, N, 3))
            C[:, 0], C[:, -1], C[0, :], C[-1, :] = bottom, top, left, right
            Cb = C.copy()
            Cb[1:-1, 1:-1] = 0.0
            R = X[1:-1, 1:-1] - np.einsum("ia,abk,jb->ijk", Bm[1:-1], Cb, Bm[1:-1])
            C[1:-1, 1:-1] = np.einsum("ia,abk,jb->ijk", Bint, R, Bint)
            return C

        def patch(C):
            from OCP.Geom import Geom_BezierSurface
            from OCP.OCP.collections import Array2_gp_Pnt

            arr = Array2_gp_Pnt(1, N, 1, N)
            for i in range(N):
                for j in range(N):
                    arr.SetValue(i + 1, j + 1, _pnt(C[i, j]))
            return BRepBuilderAPI_MakeFace(Geom_BezierSurface(arr), 1e-7).Face()

        edge_cache = {}
        surf_geom = {}

        def edge(key, C=None, P=None):
            if key not in edge_cache:
                edge_cache[key] = BRepBuilderAPI_MakeEdge(make_curve(C) if C is not None else _fit(P, 1e-5)).Edge()
            return edge_cache[key]

        sew = BRepBuilderAPI_Sewing(SEW)
        u = np.linspace(0, 1, N)
        nq = 0
        def side(a, b):
            """[(edge, reversed?)] along coarse edge a -> b."""
            e = (min(a, b), max(a, b))
            if split[e]:
                return [(edge(("h", a, e), half(a, b)), False), (edge(("h", b, e), half(b, a)), True)]
            if fl_edge[e]:
                return [(edge(("e", e), curve(e[0], e[1])), a > b)]
            return [(edge(("p", e), P=poly[e]), a > b)]

        # each connected exact region is one face, bounded by its seams
        import scipy.sparse as sp2
        from scipy.sparse.csgraph import connected_components as _cc

        n_region = n_fallback = 0
        why, demote = [], []
        exact = flab < O
        same = exact[cf0] & (flab[cf0] == flab[cf1])
        G = sp2.coo_matrix((np.ones(same.sum()), (cf0[same], cf1[same])), shape=(len(F), len(F)))
        _, comp = _cc(G, directed=False)
        comp = np.where(exact, comp, -1)
        for c in np.unique(comp[exact]):
            tris = np.flatnonzero(comp == c)
            lf = int(flab[tris[0]])
            if lf not in surf_geom:
                surf_geom[lf] = geo[lf].geom()
            inside = {int(t) for t in tris}
            out_edges = {}
            for t in tris:
                f = [int(x) for x in F[t]]
                for k in range(3):
                    a, b = f[k], f[(k + 1) % 3]
                    e = (min(a, b), max(a, b))
                    pos = int(np.searchsorted(ckey, e[0] * (int(F.max()) + 1) + e[1]))
                    other = int(cf1[pos]) if int(cf0[pos]) == t else int(cf0[pos])
                    if other not in inside:
                        out_edges.setdefault(a, []).append(b)
            # a seam path running through another corner of this boundary
            # doubles back: those corners' triangles go to patches first
            bv = sorted(set(out_edges) | {b for bs in out_edges.values() for b in bs})
            spikes = set()
            if attempt < 4:
                from scipy.spatial import cKDTree as _KD

                kd = _KD(CV[bv])
                for a, bs in out_edges.items():
                    for b in bs:
                        Pp = poly[(min(a, b), max(a, b))]
                        if Pp is None or len(Pp) < 3:
                            continue
                        for hit in kd.query_ball_point(Pp[1:-1], 1e-6):
                            spikes.update(bv[h] for h in hit if bv[h] not in (a, b))
            loops = []
            while out_edges:
                a0 = next(iter(out_edges))
                loop, a = [], a0
                while True:
                    b = out_edges[a].pop()
                    if not out_edges[a]:
                        del out_edges[a]
                    loop += side(a, b)
                    a = b
                    if a == a0 or a not in out_edges:
                        break
                loops.append(loop)
            T = CV[F[tris]]
            tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            flip = float(np.einsum("ij,ij->", tn, geo[lf].normal(T.mean(axis=1)))) < 0
            try:
                sew.Add(_region_face(geo[lf], surf_geom[lf], loops, flip))
                n_region += 1
                continue
            except Exception as e:  # noqa: BLE001 - patches or one face per triangle still work
                why.append(f"{geo[lf].kind} with {len(loops)} loops: {e}"[:200])
            if attempt < 4:
                near = np.isin(F[tris], list(spikes)).any(axis=1) if spikes else np.zeros(len(tris), bool)
                demote.append(tris[near] if near.any() and attempt == 0 else tris)
                continue
            n_fallback += 1
            for t in tris:                       # one face per triangle, merged later
                f = [int(x) for x in F[t]]
                loop = side(f[0], f[1]) + side(f[1], f[2]) + side(f[2], f[0])
                sew.Add(_exact_face(geo[lf], surf_geom[lf], loop, CV[f].mean(axis=0)))
        if demote:
            # a region whose boundary will not make one valid face (a seam
            # that doubles back on itself): smooth patches there instead,
            # flat on its plane, and build again
            return None, demote
        todo, owner = [], []                         # (sides, normal) of every patch, its triangles
        # pairs: one patch over the quad a, c, b, d (shared edge a-b, c and d the far corners)
        for i, j in pair:
            fi, fj = [int(x) for x in F[i]], [int(x) for x in F[j]]
            a, b = sorted(set(fi) & set(fj))
            c = (set(fi) - {a, b}).pop()
            d = (set(fj) - {a, b}).pop()
            n = cfn[i] + cfn[j]
            n /= max(np.linalg.norm(n), 1e-300)
            # corners 00 = a, 10 = c, 01 = d, 11 = b
            todo.append(((curve(a, c), curve(d, b), curve(a, d), curve(c, b)), n))
            owner.append((i, j))
            nq += 1
        # odd triangles: three patches each, centroid to edge midpoints;
        # the centroids and spokes of all of them pulled in one go
        odd = []
        for i in np.flatnonzero(single):
            f = [int(x) for x in F[i]]
            H = {(x, y): half(x, y) for x in f for y in f if x != y}
            Mid = {(min(x, y), max(x, y)): H[(x, y)][-1] for (x, y) in H}
            odd.append((i, f, H, Mid, np.mean([*CV[f], *Mid.values()], axis=0)))
        if odd:
            Gc = pull(np.stack([o[4] for o in odd]), cfn[[o[0] for o in odd]],
                      np.array([np.ptp(CV[o[1]], axis=0).max() for o in odd]) + TOL)
            t = np.linspace(0, 1, 13)[:, None]
            lines = np.stack([Gc[q] * (1 - t) + M * t for q, o in enumerate(odd) for M in o[3].values()])
            lines = drape(lines, np.repeat(cfn[[o[0] for o in odd]], 3, axis=0))
        for q, (i, f, H, Mid, _) in enumerate(odd):
            n = cfn[i]
            spoke = {e: Bi @ resample(lines[3 * q + r], N) for r, e in enumerate(Mid)}
            for k in range(3):
                c, j, i2 = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
                e1, e2 = (min(c, j), max(c, j)), (min(c, i2), max(c, i2))
                # corners 00 = c, 10 = M(c, j), 01 = M(c, i), 11 = G
                todo.append(((H[(c, j)], spoke[e2][::-1], H[(c, i2)], spoke[e1][::-1]), n))
                owner.append((i,))
                nq += 1
        # every patch's inside onto the surface in one go
        if todo:
            X = np.stack([coons(*sd) for sd, _ in todo])
            inner = X[:, 1:-1, 1:-1].reshape(-1, 3)
            nrm = np.repeat(np.stack([n for _, n in todo]), (N - 2) ** 2, axis=0)
            # a point of the inside is never further off the surface than
            # its patch is across: a hit further than that is another sheet
            size = np.linalg.norm(X[:, 0, 0] - X[:, -1, -1], axis=1) + np.linalg.norm(X[:, 0, -1] - X[:, -1, 0], axis=1)
            reach = np.repeat(0.5 * size, (N - 2) ** 2)
            X[:, 1:-1, 1:-1] = pull(inner, nrm, reach).reshape(len(todo), N - 2, N - 2, 3)
            Cs = [poles_of(*sd, x) for (sd, _), x in zip(todo, X)]
            if refine:
                split = deviant(Cs, owner, edge_cache, poly)
                if split:
                    return "split", (split, poly)
            for C in Cs:
                sew.Add(patch(C))
        sew.Perform()
        return sew, (nq, n_region, n_fallback, why)

    def close(sew):
        """The sewn faces as a solid; exact regions that fell back to one
        face per triangle merge back (patches stay patches)."""
        ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
        shells = []
        while ex.More():
            shells.append(TopoDS.Shell(ex.Current()))
            ex.Next()
        if len(shells) != 1 or sew.NbFreeEdges():
            raise RuntimeError(f"patches did not close ({len(shells)} shells, {sew.NbFreeEdges()} free edges)")
        fx = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(shells[0]).Solid())
        fx.Perform()
        solid = fx.Solid()
        if _volume(solid) < 0:
            solid.Reverse()
        u = ShapeUpgrade_UnifySameDomain(solid, False, True, False)
        fe = TopExp_Explorer(solid, TopAbs_FACE)
        while fe.More():
            if BRepAdaptor_Surface(TopoDS.Face(fe.Current())).GetType() == GeomAbs_BezierSurface:
                ee = TopExp_Explorer(fe.Current(), TopAbs_EDGE)
                while ee.More():
                    u.KeepShape(ee.Current())
                    ee.Next()
            fe.Next()
        u.Build()
        merged = u.Shape()
        if not BRepCheck_Analyzer(merged).IsValid():
            sf = ShapeFix_Shape(merged)
            sf.Perform()
            merged = sf.Shape()
        return merged if BRepCheck_Analyzer(merged).IsValid() or not BRepCheck_Analyzer(solid).IsValid() else solid

    def bad_regions(solid):
        """Coarse triangles of the exact regions behind the solid's invalid
        faces (sewing can still tangle a boundary a lone face had right)."""
        from scipy.spatial import cKDTree

        ex_t = np.flatnonzero(flab < O)
        if not len(ex_t):
            return []
        kd = cKDTree(CV[F[ex_t]].mean(axis=1))
        out = []
        fe = TopExp_Explorer(solid, TopAbs_FACE)
        while fe.More():
            f = TopoDS.Face(fe.Current())
            if BRepAdaptor_Surface(f).GetType() != GeomAbs_BezierSurface and not BRepCheck_Analyzer(f).IsValid():
                tm = shape_to_trimesh(f, deflection=0.05, angle=0.3)
                if len(tm.faces):
                    _, i = kd.query(tm.triangles_center[np.argmax(tm.area_faces)])
                    t = ex_t[i]
                    out.append(np.flatnonzero(comp_of(t)))
            fe.Next()
        return out

    def comp_of(t):
        """Mask of the exact region (connected, one label) holding triangle t."""
        import scipy.sparse as sp3
        from scipy.sparse.csgraph import connected_components

        same = (flab[cf0] == flab[cf1]) & (flab[cf0] == flab[t])
        G = sp3.coo_matrix((np.ones(same.sum()), (cf0[same], cf1[same])), shape=(len(F), len(F)))
        _, c = connected_components(G, directed=False)
        return (c == c[t]) & (flab == flab[t])

    from math import comb as _comb

    from .sharp import mesh_distance
    from .step import shape_to_trimesh

    g9 = np.linspace(0, 1, 9)
    B9 = np.array([[_comb(N - 1, j) * x ** j * (1 - x) ** (N - 1 - j) for j in range(N)] for x in g9])

    vn = np.asarray(mesh.vertex_normals)

    def bend(e):
        """How much the surface turns along coarse edge e, times its length:
        the edge to split across a tight bend (the longest one would cut along it)."""
        _, i = vtree.query(CV[list(e)])
        c = float(np.clip(vn[i[0]] @ vn[i[1]], -1, 1))
        return (np.arccos(c) + 0.05) * float(np.linalg.norm(CV[e[0]] - CV[e[1]]))

    worst = [0.0]

    def deviant(Cs, owner, edges, poly):
        """Coarse edges to split: those whose curve misses the mesh by more
        than REFINE, and the longest edge of each triangle under a patch that
        does. Edges shorter than MIN_SPLIT stay."""
        from OCP.BRep import BRep_Tool

        S = np.einsum("ua,nabk,vb->nuvk", B9, np.stack(Cs), B9).reshape(len(Cs), -1, 3)
        dp = mesh_distance(mesh, S.reshape(-1, 3), cap=2.0).reshape(len(Cs), -1).max(axis=1)
        keys = list(edges)
        P = []
        for k in keys:
            c = BRep_Tool.Curve_s(edges[k], 0.0, 1.0)
            t = np.linspace(c.FirstParameter(), c.LastParameter(), 12)
            P.append([[q.X(), q.Y(), q.Z()] for q in (c.Value(x) for x in t)])
        de = mesh_distance(mesh, np.array(P, float).reshape(-1, 3), cap=2.0).reshape(len(keys), -1).max(axis=1) if keys else []
        out = {k[-1] for k, d in zip(keys, de) if d > REFINE}
        worst[0] = float(dp.max()) if len(dp) else 0.0
        # and every edge whose draped path is off the surface (a coarse
        # triangle bridging a groove: its chord found no surface to fall on)
        pe = [e for e, P in poly.items() if P is not None]
        if pe:
            dl = mesh_distance(mesh, np.concatenate([poly[e] for e in pe]), cap=2.0)
            cut = np.cumsum([len(poly[e]) for e in pe])[:-1]
            out |= {e for e, d in zip(pe, np.split(dl, cut)) if d.max() > REFINE}
        for C_owner, d in zip(owner, dp):
            if d > REFINE:
                for t in C_owner:
                    f = [int(x) for x in F[t]]
                    es = [(min(f[k], f[(k + 1) % 3]), max(f[k], f[(k + 1) % 3])) for k in range(3)]
                    out.add(max(es, key=bend))
        return {e for e in out if np.linalg.norm(CV[e[0]] - CV[e[1]]) > MIN_SPLIT and e in poly and poly[e] is not None}

    from scipy.spatial import cKDTree as _KDT

    vtree = _KDT(V)

    def split_edges(split, poly):
        """The coarse mesh with those edges cut at their middle on the
        surface (a seam's at its mesh vertex), triangles split to match."""
        nonlocal CV, vid, F, flab
        mid = {}
        kq = {int(k): (int(a), int(b)) for k, a, b in zip(ckey, cf0, cf1)}
        for e in split:
            P = poly[e]
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            i = int(np.argmin(np.abs(d - 0.5 * d[-1])))
            g0, g1 = kq[e[0] * (int(F.max()) + 1) + e[1]]
            seam = flab[g0] != flab[g1]
            if seam and not 0 < i < len(P) - 1:
                continue                         # a seam splits only at one of its mesh vertices
            if 0 < i < len(P) - 1:
                q = P[i]
            else:
                w = 0.5 * d[-1]
                j = min(max(int(np.searchsorted(d, w)), 1), len(P) - 1)
                q = P[j - 1] + (P[j] - P[j - 1]) * (w - d[j - 1]) / max(d[j] - d[j - 1], 1e-300)
            dd, vi = vtree.query(q)
            mid[e] = (q, int(vi) if dd < 1e-9 else -1)
        idx = {}
        CVl, vidl = list(CV), list(vid)
        for e, (q, vi) in mid.items():
            idx[e] = len(CVl)
            CVl.append(q)
            vidl.append(vi)
        nF, nL = [], []
        for f, lf in zip(F, flab):
            v = [int(x) for x in f]
            m = [idx.get((min(v[k], v[(k + 1) % 3]), max(v[k], v[(k + 1) % 3]))) for k in range(3)]
            cnt = sum(x is not None for x in m)
            if cnt == 0:
                tri = [v]
            elif cnt == 3:
                tri = [[v[0], m[0], m[2]], [m[0], v[1], m[1]], [m[2], m[1], v[2]], [m[0], m[1], m[2]]]
            else:
                r = 0
                if cnt == 1:
                    r = next(k for k in range(3) if m[k] is not None)
                else:
                    r = (next(k for k in range(3) if m[k] is None) + 1) % 3
                v = v[r:] + v[:r]
                m = m[r:] + m[:r]
                if cnt == 1:
                    tri = [[v[0], m[0], v[2]], [m[0], v[1], v[2]]]
                else:                          # m[0] on v0-v1, m[1] on v1-v2, v2-v0 whole
                    tri = [[m[0], v[1], m[1]], [v[0], m[0], m[1]], [v[0], m[1], v[2]]]
            nF += tri
            nL += [int(lf)] * len(tri)
        CV, vid = np.array(CVl, float), np.array(vidl)
        F, flab = np.array(nF, np.int64), np.array(nL, np.int64)

    def snap():
        """A vertex inside the organic body that is off the mesh (one of an
        exact region's, slid along its surface's extension past the region,
        until the region became patches): onto the mesh, or every patch
        round it misses by as much and no splitting can help."""
        nonlocal CV, vid
        _, _, nl, _, one = _vertex_sets(F, flab, len(CV))
        o = np.flatnonzero((nl == 1) & (one >= O))
        if len(o):
            cp, _ = _closest(mesh, CV[o])
            far = np.flatnonzero(np.linalg.norm(cp - CV[o], axis=1) > TOL)
            if len(far):
                CV, vid = CV.copy(), vid.copy()
                CV[o[far]] = cp[far]
                vid[o[far]] = -1
        unfold()

    def unfold(passes=4):
        """Flip edges of organic coarse triangles that face against the
        surface under them (a collapse or a split folded the layout there):
        the pulls along a backwards normal miss, and the patch folds over
        its neighbours."""
        nonlocal F
        for _ in range(passes):
            org = np.flatnonzero(flab >= O)
            fn = lambda T: np.cross(T[..., 1, :] - T[..., 0, :], T[..., 2, :] - T[..., 0, :])
            T = CV[F[org]]
            n = fn(T)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
            _, sn = _closest(mesh, T.mean(axis=1))
            bad = org[np.einsum("ij,ij->i", n, sn) < 0.3]
            if not len(bad):
                return
            owner_e = {}
            for t, f in enumerate(F):
                for k in range(3):
                    owner_e.setdefault((min(f[k], f[(k + 1) % 3]), max(f[k], f[(k + 1) % 3])), []).append(t)
            done, changed = set(), False
            for t in bad:
                if t in done:
                    continue
                f = [int(x) for x in F[t]]
                best = None
                for k in range(3):
                    a, b, c = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
                    ts = owner_e[(min(a, b), max(a, b))]
                    if len(ts) != 2:
                        continue
                    u = ts[0] if ts[1] == t else ts[1]
                    if u in done or flab[u] != flab[t]:
                        continue
                    d = next(int(x) for x in F[u] if x not in (a, b))
                    if (min(c, d), max(c, d)) in owner_e:
                        continue                        # the flipped edge exists already
                    new = np.array([[c, a, d], [c, d, b]])
                    Tn = CV[new]
                    nn = fn(Tn)
                    area = np.linalg.norm(nn, axis=1)
                    if area.min() < 1e-9:
                        continue
                    _, s2 = _closest(mesh, Tn.mean(axis=1))
                    q = np.einsum("ij,ij->i", nn / area[:, None], s2).min()
                    if q > 0.5 and (best is None or q > best[0]):
                        best = (q, u, new)
                if best is not None:
                    _, u, new = best
                    F = F.copy()
                    F[t], F[u] = new[0], new[1]
                    done |= {t, int(u)}
                    changed = True
            if not changed:
                return

    # refine while it helps: a split can also fold the layout somewhere
    # (a big triangle wrapped round a thin wall's edge), so the best layout
    # so far, by its worst patch, is kept and refining stops once two
    # rounds in a row are no better
    refined, best, worse = 0, None, 0
    for r in range(REFINE_ROUNDS + 1):
        snap()
        CE, ckey, cf0, cf1 = _edge_faces(F)
        sew, got = build(0, refine=True)
        if sew is None:                          # a region to patch first
            for tris in got:
                flab[tris] = int(flab.max()) + 1
            best, worse = None, 0
            continue
        if not isinstance(sew, str):
            best = None                          # nothing left to split: this one
            break
        split, poly_ = got
        if best is None or worst[0] < best[0]:
            best, worse = (worst[0], CV, vid, F, flab, refined), 0
        else:
            worse += 1
        if r == REFINE_ROUNDS or worse >= 2:
            break
        split_edges(split, poly_)
        refined += len(split)
    if best is not None:
        _, CV, vid, F, flab, refined = best
    CE, ckey, cf0, cf1 = _edge_faces(F)

    demoted = 0
    for attempt in range(5):
        snap()
        CE, ckey, cf0, cf1 = _edge_faces(F)
        sew, got = build(attempt)
        if sew is None:
            for tris in got:
                flab[tris] = int(flab.max()) + 1
            demoted += len(got)
            continue
        solid = close(sew)
        bad = bad_regions(solid) if not BRepCheck_Analyzer(solid).IsValid() and attempt < 4 else []
        if not bad:
            break
        for tris in bad:
            flab[tris] = int(flab.max()) + 1
        demoted += len(bad)
    nq, n_region, n_fallback, why = got
    solid = _forward(solid)
    t3 = time.time()
    rep = {"patches": nq, "degree": N - 1, "exactFaces": n_region, "refinedEdges": refined, "exactDemoted": demoted, "exactFallbacks": n_fallback, "reasons": why[:5],
           "triangles": int((flab >= O).sum()), "exactTriangles": int((flab < O).sum()),
           "exactSurfaces": len({int(x) for x in np.unique(lab) if x < O}),
           "smoothRegions": len({int(x) for x in np.unique(lab) if x >= O}),
           "seconds": {"coarsen": round(t1 - t0, 1), "patches": round(t3 - t1, 1),
                       "solid": round(time.time() - t3, 1)}}
    return solid, rep


def _forward(solid, tries=20):
    """The solid rebuilt with every face carrying its own outward
    orientation: STEP keeps faces' orientations but not a reversed shell's
    or solid's, so an inside-out shell would read back inside out. Readers
    then check the orientation with OpenCascade's infinite-point test,
    which shoots from the first face and can misjudge a patch; the faces go
    in an order it gets right, or a reader would turn the part inside out."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_FACE, TopAbs_OUT
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid

    from .step import _volume

    faces = []
    ex = TopExp_Explorer(solid, TopAbs_FACE)
    while ex.More():
        faces.append(ex.Current())
        ex.Next()

    def make(fs):
        bb = BRep_Builder()
        shell = TopoDS_Shell()
        bb.MakeShell(shell)
        for f in fs:
            bb.Add(shell, f)
        shell.Closed(True)
        out = TopoDS_Solid()
        bb.MakeSolid(out)
        bb.Add(out, shell)
        return out

    out = make(faces)
    if _volume(out) <= 0:
        return solid
    for k in range(min(tries, len(faces))):
        cand = make(faces[k:] + faces[:k]) if k else out
        cl = BRepClass3d_SolidClassifier(cand)
        cl.PerformInfinitePoint(1e-6)
        if cl.State() == TopAbs_OUT:
            return cand
    return out


def _pairs(V, F, lab, O, cos_min=0.5, angle_max=160.0):
    """Organic triangles matched in pairs across an edge inside one region,
    best-shaped quads first: [(i, j)]. A pair must be convex, with corner
    angles under `angle_max` degrees and the two normals within ~60 degrees."""
    E, _, f0, f1 = _edge_faces(F)
    ok = (lab[f0] == lab[f1]) & (lab[f0] >= O)
    E, f0, f1 = E[ok], f0[ok], f1[ok]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    a, b = E[:, 0], E[:, 1]
    c = F[f0].sum(axis=1) - a - b
    d = F[f1].sum(axis=1) - a - b
    Q = np.stack([V[a], V[c], V[b], V[d]], axis=1)          # around the quad
    p = np.roll(Q, 1, axis=1) - Q
    q = np.roll(Q, -1, axis=1) - Q
    cosang = np.einsum("ikj,ikj->ik", p, q) / np.maximum(np.linalg.norm(p, axis=2) * np.linalg.norm(q, axis=2), 1e-300)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    nn = n[f0] + n[f1]
    turn = np.einsum("ikj,ij->ik", np.cross(q, p), nn)       # convex: every corner turns the same way
    good = ((np.einsum("ij,ij->i", n[f0], n[f1]) > cos_min) & (ang.max(axis=1) < angle_max)
            & ((turn > 0).all(axis=1) | (turn < 0).all(axis=1)))
    score = np.abs(ang - 90.0).max(axis=1)
    used = np.zeros(len(F), bool)
    out = []
    for k in np.flatnonzero(good)[np.argsort(score[good], kind="stable")]:
        i, j = int(f0[k]), int(f1[k])
        if not used[i] and not used[j]:
            used[i] = used[j] = True
            out.append((i, j))
    return out


def _uv(s, P, ref=None):
    """(u, v) of points on exact surface `s` in OpenCascade's parameters for
    s.geom(): planes and cylinders (angle unwrapped around `ref`)."""
    if s.kind == "plane":
        X, Y = _frame(s.n)
        d = P - s.p
        return np.stack([d @ X, d @ Y], axis=1)
    X, Y = _frame(s.axis)
    d = P - s.p0
    u = np.arctan2(d @ Y, d @ X)
    if ref is not None:
        u = ref + (u - ref + np.pi) % (2 * np.pi) - np.pi
    return np.stack([u, d @ s.axis], axis=1)


def _frame(n):
    """The X and Y directions OpenCascade gives gp_Ax3(origin, n)."""
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt

    ax = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(*map(float, n)))
    x, y = ax.XDirection(), ax.YDirection()
    return np.array([x.X(), x.Y(), x.Z()]), np.array([y.X(), y.Y(), y.Z()])


def _exact_face(s, geom, loop, centre, tol=0.01):
    """A face on exact surface `s` (its OpenCascade surface `geom`) bounded by
    `loop`, [(edge, reversed)], with each edge's curve on the surface given
    explicitly (sampled, mapped to (u, v), fitted), so a bore's face covers
    the arc it should and not the rest of the cylinder."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepLib import BRepLib
    from OCP.Geom2dAPI import Geom2dAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.gp import gp_Pnt2d
    from OCP.OCP.collections import Array1_double, Array1_gp_Pnt2d
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopoDS import TopoDS_Face, TopoDS_Wire

    if s.kind == "sphere":
        raise RuntimeError("spherical exact faces are not built yet")
    ref = None
    if s.kind == "cylinder":
        ref = float(_uv(s, centre[None])[0, 0])
    bb = BRep_Builder()
    face = TopoDS_Face()
    bb.MakeFace(face, geom, 1e-7)
    wire = TopoDS_Wire()
    bb.MakeWire(wire)
    ring = []
    for e, rev in loop:
        c = BRep_Tool.Curve_s(e, 0.0, 1.0)
        t = np.linspace(c.FirstParameter(), c.LastParameter(), 12)
        P = np.array([[p.X(), p.Y(), p.Z()] for p in (c.Value(x) for x in t)])
        UV = _uv(s, P, ref)
        pts, par = Array1_gp_Pnt2d(1, len(t)), Array1_double(1, len(t))
        for j, (q, x) in enumerate(zip(UV, t)):
            pts.SetValue(j + 1, gp_Pnt2d(float(q[0]), float(q[1])))
            par.SetValue(j + 1, float(x))
        pc = Geom2dAPI_PointsToBSpline(pts, par, 1, 3, GeomAbs_C2, 1e-7).Curve()
        bb.UpdateEdge(e, pc, face, tol)
        bb.Add(wire, e.Oriented(TopAbs_REVERSED) if rev else e)
        ring.append(UV[::-1] if rev else UV)
    R = np.vstack(ring)
    area = 0.5 * np.sum(R[:, 0] * np.roll(R[:, 1], -1) - np.roll(R[:, 0], -1) * R[:, 1])
    if area < 0:
        wire.Reverse()
    bb.Add(face, wire)
    BRepLib.SameParameter_s(face, tol)
    from OCP.ShapeFix import ShapeFix_Face

    fx = ShapeFix_Face(face)                     # the wire's own checks (uv gaps at the corners)
    fx.Perform()
    return fx.Face()


def _region_face(s, geom, loops, flip, tol=0.01):
    """One face on exact surface `s` bounded by `loops` ([(edge, reversed)]
    each, counterclockwise about the part's outward normal; `flip` when that
    is against the surface's own normal). Each edge gets its curve on the
    surface, the angle around a cylinder unwrapped along the loop, so a ring
    round a bore closes in 3D but not in (u, v): ShapeFix adds the seam."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepLib import BRepLib
    from OCP.Geom2dAPI import Geom2dAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.gp import gp_Pnt2d
    from OCP.OCP.collections import Array1_double, Array1_gp_Pnt2d
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopoDS import TopoDS_Face, TopoDS_Wire

    if s.kind == "sphere":
        raise RuntimeError("spherical exact faces are not built yet")
    bb = BRep_Builder()
    face = TopoDS_Face()
    bb.MakeFace(face, geom, 1e-7)
    for loop in loops:
        if flip:
            loop = [(e, not r) for e, r in reversed(loop)]
        parts = []
        for e, rev in loop:
            c = BRep_Tool.Curve_s(e, 0.0, 1.0)
            t = np.linspace(c.FirstParameter(), c.LastParameter(), 12)
            P = np.array([[p.X(), p.Y(), p.Z()] for p in (c.Value(x) for x in t)])
            parts.append((t, P[::-1] if rev else P))
        UV = _uv(s, np.vstack([P for _, P in parts]))
        if s.kind == "cylinder":
            UV[:, 0] = np.unwrap(UV[:, 0])
        wire = TopoDS_Wire()
        bb.MakeWire(wire)
        for k, ((e, rev), (t, _)) in enumerate(zip(loop, parts)):
            uv = UV[12 * k:12 * (k + 1)]
            if rev:
                uv = uv[::-1]
            pts, par = Array1_gp_Pnt2d(1, len(t)), Array1_double(1, len(t))
            for j, (q, x) in enumerate(zip(uv, t)):
                pts.SetValue(j + 1, gp_Pnt2d(float(q[0]), float(q[1])))
                par.SetValue(j + 1, float(x))
            pc = Geom2dAPI_PointsToBSpline(pts, par, 1, 3, GeomAbs_C2, 1e-7).Curve()
            bb.UpdateEdge(e, pc, face, tol)
            bb.Add(wire, e.Oriented(TopAbs_REVERSED) if rev else e)
        bb.Add(face, wire)
    BRepLib.SameParameter_s(face, tol)
    fx = ShapeFix_Face(face)
    fx.Perform()
    out = fx.Face()
    if not BRepCheck_Analyzer(out).IsValid():
        raise RuntimeError("exact region face is not valid")
    return out


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
