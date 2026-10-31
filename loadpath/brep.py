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
RAY_MAX = 3.0        # mm: farthest a patch point is pulled along its normal onto the surface
CREASE = 40.0        # degrees: organic faces meeting sharper than this are separate smooth regions
DEBUG_AT = None
DEBUG_MISS = []
DEBUG_Q = None       # a list collects every organic patch's points
SEW = 0.01           # mm: sewing tolerance (exact faces' seams sit within TOL of the patches')
SKIP_FAILED = False
ASPECT = 0.1         # least 2*area / longest edge^2 of a patch (equilateral 0.87)
MAX_EDGE = np.inf     # mm: longest organic patch edge
LEAN = 0.5           # least cosine between a patch's plane and the surface normal at its corners
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
    n = len(V)
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
            Ee, key, f0, f1 = _edge_faces(Fc)
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
        longest = np.sqrt(((Tnew - np.roll(Tnew, 1, axis=1)) ** 2).sum(axis=2).max(axis=1))
        organic = ((q1 > ASPECT) | (q1 >= q0)) & ((l1 > LEAN) | (l1 >= l0)) & (longest <= MAX_EDGE)
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
    K, pairs, nl, h, one = _vertex_sets(F, lb, n)
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
            cp, _, tri = trimesh.proximity.closest_point(org, T[o])
            vn = trimesh.Trimesh(W, F, process=False).vertex_normals[o]
            ok = np.einsum("ij,ij->i", org.face_normals[tri], vn) > 0.5
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
    curved = np.zeros(int(lab.max()) + 1, bool)
    for lk, sf in geo.items():
        if lk < len(curved):
            curved[lk] = sf.kind != "plane"
    O = (max(geo) if geo else 0) + 1
    lab = islands(mesh, creases(mesh, lab, O))
    CV, vid, F, flab = coarsen(mesh, lab, patches, curved, geo, O)
    # a coarse patch can end up an island inside another face (a sliver of the
    # body on a flat face): it joins that face
    flab = islands(trimesh.Trimesh(CV, F, process=False), flab, ISLAND, geo)
    V = np.asarray(mesh.vertices, float)
    t1 = time.time()

    # fine-mesh graphs for the paths
    fl = lab
    Ef, keyf, f0, f1 = _edge_faces(np.asarray(mesh.faces, np.int64))
    seam_e = fl[f0] != fl[f1]
    org_v = np.zeros(len(V), bool)
    org_v[np.asarray(mesh.faces)[fl >= O].ravel()] = True
    import scipy.sparse as sp

    def graph(mask):
        w = np.linalg.norm(V[Ef[mask, 0]] - V[Ef[mask, 1]], axis=1)
        return sp.coo_matrix((w, (Ef[mask, 0], Ef[mask, 1])), shape=(len(V), len(V))).tocsr()

    Gseam = graph(seam_e)
    Gorg = graph(org_v[Ef[:, 0]] & org_v[Ef[:, 1]])

    CE, ckey, cf0, cf1 = _edge_faces(F)
    N = int(order) + 1                              # points per patch side
    org_mesh = trimesh.Trimesh(V, np.asarray(mesh.faces)[fl >= O], process=False)

    from trimesh.ray.ray_triangle import RayMeshIntersector

    rays = RayMeshIntersector(org_mesh)

    def pull(P, nrm):
        """Points onto the organic surface along the patch normal (the nearer
        hit either way, where the surface faces the same way); else the
        closest point, where its normal agrees; else left where they are."""
        if not len(P):
            return P
        best = np.full(len(P), np.inf)
        out = P.copy()
        for sgn in (1.0, -1.0):
            loc, ri, ti = rays.intersects_location(P, sgn * nrm, multiple_hits=False)
            if not len(ri):
                continue
            d = np.linalg.norm(loc - P[ri], axis=1)
            ok = (np.einsum("ij,ij->i", org_mesh.face_normals[ti], nrm[ri]) > 0.3) & (d < RAY_MAX) & (d < best[ri])
            out[ri[ok]] = loc[ok]
            best[ri[ok]] = d[ok]
        miss = ~np.isfinite(best)
        if miss.any():
            cp, _, tri = trimesh.proximity.closest_point(org_mesh, P[miss])
            ok = np.einsum("ij,ij->i", org_mesh.face_normals[tri], nrm[miss]) > 0.3
            idx = np.flatnonzero(miss)
            out[idx[ok]] = cp[ok]
            if DEBUG_Q is not None:
                DEBUG_MISS.extend(P[idx[~ok]].tolist())
                DEBUG_MISS.extend(("ray", ) for _ in range(int(ok.sum())))
        return out

    def resample(P, n):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if d[-1] <= 0:
            return np.repeat(P[:1], n, axis=0)
        t = np.linspace(0, d[-1], n)
        return np.stack([np.interp(t, d, P[:, k]) for k in range(3)], axis=1)

    def drape(P, nrm, rounds=4):
        """A polyline between two surface points onto the surface, evenly
        spaced: pulling a chord along one direction bunches its points up at
        a bend, and the gaps between them cut the corner."""
        P = P.copy()
        R = np.repeat(nrm[None], len(P) - 2, axis=0)
        for _ in range(rounds):
            P[1:-1] = pull(P[1:-1], R)
            P = resample(P, len(P))
        P[1:-1] = pull(P[1:-1], R)
        return P

    # every coarse edge: a polyline over the surface, and its midpoint by length
    poly, split, fl_edge = {}, {}, {}
    cfn = np.cross(CV[F[:, 1]] - CV[F[:, 0]], CV[F[:, 2]] - CV[F[:, 0]])
    cfn /= np.maximum(np.linalg.norm(cfn, axis=1, keepdims=True), 1e-300)
    for (a, b), la, lb, f0e, f1e in zip(CE, flab[cf0], flab[cf1], cf0, cf1):
        a, b = int(a), int(b)
        pa, pb = CV[a], CV[b]
        L = float(np.linalg.norm(pa - pb))
        if la == lb and la < O:
            t = np.linspace(0, 1, 9)[:, None]
            P = _onto(geo[int(la)], pa * (1 - t) + pb * t)
            P[0], P[-1] = pa, pb
        elif la == lb:
            # inside the organic body: the chord pulled onto the surface (the
            # shortest path over the mesh's edges is a staircase, and two
            # from one corner can share their first half)
            t = np.linspace(0, 1, 17)[:, None]
            nn = cfn[f0e] + cfn[f1e]
            nn /= max(np.linalg.norm(nn), 1e-300)
            P = drape(pa * (1 - t) + pb * t, nn)
        else:
            pth = _path(Gseam, int(vid[a]), int(vid[b]), limit=4 * L + 1.0)
            if pth is None:
                raise RuntimeError("no surface path for a patch edge")
            P = V[pth]
        poly[(a, b)] = P
        split[(a, b)] = False                     # set below: next to a three-quad triangle, two half edges
        fl_edge[(a, b)] = la >= O or lb >= O      # organic on a side: a Bezier edge

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

    def patch(bottom, top, left, right, nrm):
        """A Bezier patch from its four sides' poles (bottom, top along u;
        left, right along v; corners shared), its inside through the surface:
        a Coons blend of the sides pulled onto the mesh, interpolated."""
        from OCP.Geom import Geom_BezierSurface
        from OCP.OCP.collections import Array2_gp_Pnt

        C = np.zeros((N, N, 3))
        C[:, 0], C[:, -1], C[0, :], C[-1, :] = bottom, top, left, right
        sb, st_, sl, sr = (Bm @ x for x in (bottom, top, left, right))
        U, W = np.meshgrid(u, u, indexing="ij")
        X = ((1 - W)[..., None] * sb[:, None] + W[..., None] * st_[:, None]
             + (1 - U)[..., None] * sl[None] + U[..., None] * sr[None]
             - ((1 - U) * (1 - W))[..., None] * sb[0] - (U * (1 - W))[..., None] * sb[-1]
             - ((1 - U) * W)[..., None] * st_[0] - (U * W)[..., None] * st_[-1])
        inner = X[1:-1, 1:-1].reshape(-1, 3)
        X[1:-1, 1:-1] = pull(inner, np.repeat(nrm[None], len(inner), axis=0)).reshape(N - 2, N - 2, 3)
        # interior poles so the patch passes through those points
        Cb = C.copy()
        Cb[1:-1, 1:-1] = 0.0
        R = X[1:-1, 1:-1] - np.einsum("ia,abk,jb->ijk", Bm[1:-1], Cb, Bm[1:-1])
        C[1:-1, 1:-1] = np.einsum("ia,abk,jb->ijk", Bint, R, Bint)
        if DEBUG_Q is not None:
            DEBUG_Q.append(np.einsum("ia,abk,jb->ijk", Bm, C, Bm))
        arr = Array2_gp_Pnt(1, N, 1, N)
        for i in range(N):
            for j in range(N):
                arr.SetValue(i + 1, j + 1, _pnt(C[i, j]))
        return BRepBuilderAPI_MakeFace(Geom_BezierSurface(arr), 1e-7).Face()

    edge_cache = {}
    surf_geom = {}
    sides = []

    def edge(key, C=None, P=None):
        if key not in edge_cache:
            edge_cache[key] = BRepBuilderAPI_MakeEdge(make_curve(C) if C is not None else _fit(P, 1e-5)).Edge()
        return edge_cache[key]

    t2 = time.time()
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
        inside = set(int(t) for t in tris)
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
        except Exception:
            n_fallback += 1
            for t in tris:                       # one face per triangle, merged later
                f = [int(x) for x in F[t]]
                loop = side(f[0], f[1]) + side(f[1], f[2]) + side(f[2], f[0])
                sew.Add(_exact_face(geo[lf], surf_geom[lf], loop, CV[f].mean(axis=0)))
    # pairs: one patch over the quad a, c, b, d (shared edge a-b, c and d the far corners)
    for i, j in pair:
        fi, fj = [int(x) for x in F[i]], [int(x) for x in F[j]]
        a, b = sorted(set(fi) & set(fj))
        c = (set(fi) - {a, b}).pop()
        d = (set(fj) - {a, b}).pop()
        n = cfn[i] + cfn[j]
        n /= max(np.linalg.norm(n), 1e-300)
        # corners 00 = a, 10 = c, 01 = d, 11 = b
        sew.Add(patch(curve(a, c), curve(d, b), curve(a, d), curve(c, b), n))
        nq += 1
    for i in np.flatnonzero(single):
        f = [int(x) for x in F[i]]
        T = CV[f]
        n = cfn[i]
        H = {(x, y): half(x, y) for x in f for y in f if x != y}
        Mid = {(min(x, y), max(x, y)): H[(x, y)][-1] for (x, y) in H}
        Gc = pull(np.mean([*T, *Mid.values()], axis=0)[None], n[None])[0]
        spoke = {}
        for e, M in Mid.items():                 # centroid -> edge midpoint
            t = np.linspace(0, 1, 13)[:, None]
            spoke[e] = Bi @ resample(drape(Gc * (1 - t) + M * t, n), N)
        for k in range(3):
            c, j, i2 = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            e1, e2 = (min(c, j), max(c, j)), (min(c, i2), max(c, i2))
            # corners 00 = c, 10 = M(c, j), 01 = M(c, i), 11 = G
            sew.Add(patch(H[(c, j)], spoke[e2][::-1], H[(c, i2)], spoke[e1][::-1], n))
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
    if DEBUG and sides:
        from collections import Counter
        key = lambda P: tuple(np.round(np.sort(np.stack([P[0], P[-1], P[len(P) // 2]]), axis=0).ravel(), 4))
        cnt = Counter(key(P) for P, *_ in sides)
        odd = [(t, k, f, P[0].round(3), P[-1].round(3)) for P, t, k, f in sides if cnt[key(P)] == 1]
        print("unmatched quad sides", len(odd), flush=True)
        q = np.array(DEBUG_AT) if DEBUG_AT is not None else None
        if q is not None:
            for P, t, k, f in sides:
                if min(np.linalg.norm(P[0] - q), np.linalg.norm(P[-1] - q)) < 1e-3:
                    print("side at", t, k, f, P.round(3).tolist(), cnt[key(P)], flush=True)
    if DEBUG and sew.NbFreeEdges():
        from OCP.BRep import BRep_Tool as _T
        from OCP.TopExp import TopExp as _X
        for i in range(1, sew.NbFreeEdges() + 1):
            e = sew.FreeEdge(i)
            p0, p1 = _T.Pnt_s(_X.FirstVertex_s(e)), _T.Pnt_s(_X.LastVertex_s(e))
            print("free edge", round(p0.X(), 3), round(p0.Y(), 3), round(p0.Z(), 3), "len", round(p0.Distance(p1), 4), flush=True)
    if len(shells) != 1 or sew.NbFreeEdges():
        raise RuntimeError(f"patches did not close ({len(shells)} shells, {sew.NbFreeEdges()} free edges)")
    if DEBUG:
        from OCP.BRepCheck import BRepCheck_Analyzer
        raw = BRepBuilderAPI_MakeSolid(shells[0]).Solid()
        print("raw shell: valid", BRepCheck_Analyzer(shells[0]).IsValid(), "solid vol", _volume(raw), flush=True)
    fx = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(shells[0]).Solid())
    fx.Perform()
    solid = fx.Solid()
    if _volume(solid) < 0:
        solid.Reverse()
    if DEBUG:
        from OCP.BRepCheck import BRepCheck_Analyzer
        print("before unify: valid", BRepCheck_Analyzer(solid).IsValid(), "vol", _volume(solid), flush=True)
    from OCP.BRepCheck import BRepCheck_Analyzer

    u = ShapeUpgrade_UnifySameDomain(solid, False, True, False)
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_BezierSurface
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE

    fe = TopExp_Explorer(solid, TopAbs_FACE)          # patches stay patches: only exact faces merge
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
        from OCP.ShapeFix import ShapeFix_Shape

        sf = ShapeFix_Shape(merged)
        sf.Perform()
        merged = sf.Shape()
    if BRepCheck_Analyzer(merged).IsValid() or not BRepCheck_Analyzer(solid).IsValid():
        solid = merged                           # else keep the faces unmerged: valid beats tidy
    if DEBUG:
        print("after unify: valid", BRepCheck_Analyzer(solid).IsValid(), "vol", _volume(solid), flush=True)
    rep = {"patches": nq, "exactFaces": n_region, "exactFallbacks": n_fallback, "triangles": int((flab >= O).sum()), "exactTriangles": int((flab < O).sum()),
           "exactSurfaces": int(len(set(int(x) for x in np.unique(lab) if x < O))),
           "smoothRegions": int(len(set(int(x) for x in np.unique(lab) if x >= O))),
           "seconds": {"coarsen": round(t1 - t0, 1), "edges": round(t2 - t1, 1), "faces": round(t3 - t2, 1),
                       "solid": round(time.time() - t3, 1)}}
    return solid, rep


def _pairs(V, F, lab, O, cos_min=0.5, angle_max=160.0):
    """Organic triangles matched in pairs across an edge inside one region,
    best-shaped quads first: [(i, j)]. A pair must be convex, with corner
    angles under `angle_max` degrees and the two normals within ~60 degrees."""
    E, key, f0, f1 = _edge_faces(F)
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
    from OCP.gp import gp_Pnt2d
    from OCP.Geom2dAPI import Geom2dAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_C2
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
