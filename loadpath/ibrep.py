"""
Smooth B-rep patches straight from an implicit field: one level set f = 0
(f < 0 inside) as a quad layout of bicubic Bezier patches that sit on the
true surface, ready to sew into a closed OpenCascade shell.

1. surface nets: the zero set on a grid of spacing h as a quad mesh (one
   quad per grid edge the surface crosses), made manifold (one vertex per
   surface sheet in a cell, ambiguous faces decided by their centre sample,
   the grid shifted until no edge has more than two quads), vertices on the
   surface (Newton along the gradient of the exact field);
2. patches: each quad edge a cubic Bezier through points draped onto the
   surface, each quad a bicubic Bezier patch through its sides and four
   inside points on the surface. Shared edges are computed once, so
   neighbouring patches meet exactly and sew at 1e-4 mm.

Accuracy is to the field itself, not to a mesh of it: the patches sit within
microns of the level set where it is smooth at the scale of h (the report
samples them). Used by latticestep.py for a skinned TPMS part's cavities.

(A general builder for intersections of several families, trimming patches
along their creases, was tried first, 2026-11-02: 1-3% of trimmed faces
failed at grazing creases, so the cavity route that needs no trimming won.)
"""
from __future__ import annotations

from math import comb

import numpy as np

__all__ = ["Family", "Sheet", "manifold_nets", "surface_nets"]

DEG = 3                  # patch and edge degree
EPS = 1e-4               # mm: finite-difference step for gradients


class Family:
    """A smooth implicit surface: f(P) (n, 3) -> (n,) mm, negative inside,
    meshed at grid spacing h (mm)."""

    def __init__(self, name, f, h):
        self.name, self.f, self.h = name, f, float(h)

    def vg(self, P):
        P = np.asarray(P, float)
        n = len(P)
        Q = np.concatenate([P] + [P + d for d in np.eye(3) * EPS] + [P - d for d in np.eye(3) * EPS])
        v = np.asarray(self.f(Q), float)
        g = np.stack([(v[(1 + a) * n:(2 + a) * n] - v[(4 + a) * n:(5 + a) * n]) / (2 * EPS) for a in range(3)], 1)
        return v[:n], g


def _project(fam, P, it=10, step=None):
    """Points onto fam = 0 (Newton along the gradient, steps capped)."""
    P = np.array(P, float)
    if not len(P):
        return P
    for _ in range(it):
        v, g = fam.vg(P)
        dx = (v / np.maximum((g * g).sum(1), 1e-12))[:, None] * g
        if step is not None:
            L = np.linalg.norm(dx, axis=1, keepdims=True)
            dx *= np.minimum(1.0, step / np.maximum(L, 1e-300))
        P -= dx
        if np.abs(v).max() < 1e-7:
            break
    return P


def surface_nets(fam, lo, hi):
    """Quad mesh of fam's zero set over the box lo..hi: (V (n, 3), Q (m, 4)),
    quads counter-clockwise seen from outside (f > 0). Manifold: a cell the
    surface passes through more than once (a gap thinner than the grid) gets
    one vertex per sheet, the sheets told apart on each cube face as
    marching cubes would (an ambiguous face by its centre's sign), so two
    sheets never share a vertex."""
    h = fam.h
    lo = np.asarray(lo, float)
    n = np.ceil((np.asarray(hi, float) - lo) / h).astype(int) + 1
    ax = [lo[i] + np.arange(n[i]) * h for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    f = np.empty(len(G))
    for s0 in range(0, len(G), 400_000):
        f[s0:s0 + 400_000] = fam.f(G[s0:s0 + 400_000])
    f = f.reshape(tuple(n))
    inside = f < 0
    nc = n - 1
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    S = np.stack([inside[i:i + nc[0], j:j + nc[1], k:k + nc[2]] for i, j, k in corners], -1)
    act = S.any(-1) & ~S.all(-1)
    cells = np.argwhere(act)
    cid = -np.ones(tuple(nc), np.int64)
    cid[tuple(cells.T)] = np.arange(len(cells))
    # the 12 cell edges: (axis, offset in the two other axes)
    edges = []
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        for o1 in (0, 1):
            for o2 in (0, 1):
                off = np.zeros(3, int)
                off[b], off[c] = o1, o2
                edges.append((a, off))
    eix = {(a, tuple(off)): k for k, (a, off) in enumerate(edges)}
    # crossings per cell and edge
    cross = np.zeros((len(cells), 12), bool)
    X = np.zeros((len(cells), 12, 3))
    for k, (a, off) in enumerate(edges):
        e = np.zeros(3, int)
        e[a] = 1
        p0 = cells + off
        p1 = p0 + e
        f0, f1 = f[tuple(p0.T)], f[tuple(p1.T)]
        ch = (f0 < 0) != (f1 < 0)
        t = np.where(ch, f0 / np.where(ch, f0 - f1, 1.0), 0.0)
        cross[:, k] = ch
        X[:, k] = lo + (p0 + t[:, None] * e) * h
    # the 6 faces: normal axis, side; their 4 edges in order round the face
    faces = []
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        for sd in (0, 1):
            def E(ax_, ob, oc, a=a, b=b, c=c, sd=sd):
                off = np.zeros(3, int)
                off[a], off[b], off[c] = sd, ob, oc
                off[ax_] = 0
                return eix[(ax_, tuple(off))]
            # corners c00 (b=0,c=0), c10, c11, c01; edges bottom (c00-c10, along b, c=0), right (c10-c11, along c, b=1),
            # top (c01-c11, along b, c=1), left (c00-c01, along c, b=0)
            faces.append((a, sd, b, c, [E(b, 0, 0), E(c, 1, 0), E(b, 0, 1), E(c, 0, 0)]))
    # union-find of crossed edges per cell, across its faces
    parent = np.tile(np.arange(12), (len(cells), 1))

    def find(r, x):
        while parent[r, x] != x:
            parent[r, x] = parent[r, parent[r, x]]
            x = parent[r, x]
        return x

    def union(rows, x, y):
        for r, a_, b_ in zip(rows, x, y):
            ra, rb = find(r, a_), find(r, b_)
            if ra != rb:
                parent[r, ra] = rb

    for a, sd, b, c, (eb, er, et, el) in faces:
        cc = cross[:, [eb, er, et, el]]
        cnt = cc.sum(1)
        two = np.flatnonzero(cnt == 2)
        if len(two):
            idx = np.argsort(~cc[two], axis=1, kind="stable")[:, :2]
            loc = np.array([eb, er, et, el])[idx]
            union(two, loc[:, 0], loc[:, 1])
        four = np.flatnonzero(cnt == 4)
        if len(four):
            off = np.zeros(3)
            off[a] = sd
            off[b] = off[c] = 0.5
            Pc = lo + (cells[four] + off) * h
            fc = np.asarray(fam.f(Pc), float)
            p00 = cells[four].copy()
            p00[:, a] += sd
            c00 = inside[tuple(p00.T)]
            joined = (fc < 0) == c00              # c00 and c11 joined through the centre
            # joined: the curves cut off c10 (bottom-right) and c01 (top-left)
            union(four[joined], np.full(joined.sum(), eb), np.full(joined.sum(), er))
            union(four[joined], np.full(joined.sum(), et), np.full(joined.sum(), el))
            union(four[~joined], np.full((~joined).sum(), eb), np.full((~joined).sum(), el))
            union(four[~joined], np.full((~joined).sum(), et), np.full((~joined).sum(), er))
    roots = np.array([[find(r, k) for k in range(12)] for r in range(len(cells))]) if len(cells) else np.zeros((0, 12), int)
    # one vertex per (cell, component)
    key = np.where(cross, np.arange(len(cells))[:, None] * 12 + roots, -1)
    uk, inv = np.unique(key[cross], return_inverse=True)
    vid = -np.ones((len(cells), 12), np.int64)
    vid[cross] = inv
    acc = np.zeros((len(uk), 3))
    np.add.at(acc, inv, X[cross])
    cnt = np.bincount(inv, minlength=len(uk))
    V = acc / np.maximum(cnt, 1)[:, None]
    V = _project(fam, V, step=0.5 * h)
    quads = []
    for a in range(3):
        e = np.zeros(3, int)
        e[a] = 1
        s0 = inside[:n[0] - e[0], :n[1] - e[1], :n[2] - e[2]]
        s1 = inside[e[0]:, e[1]:, e[2]:]
        ch = np.argwhere(s0 != s1)
        b, c = [(1, 2), (2, 0), (0, 1)][a]
        ids, ok = [], np.ones(len(ch), bool)
        for db, dc in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
            q = ch.copy()
            q[:, b] += db
            q[:, c] += dc
            good = (q >= 0).all(1) & (q < nc).all(1)
            ok &= good
            qc = np.where(good[:, None], q, 0)
            ci = cid[tuple(qc.T)]
            off = np.zeros((len(ch), 3), int)
            off[:, b], off[:, c] = -db, -dc            # the grid edge's place in this cell
            k = np.array([eix[(a, tuple(o))] for o in off]) if len(ch) else np.zeros(0, int)
            ids.append(np.where(good & (ci >= 0), vid[np.maximum(ci, 0), k], -1))
        ids = np.stack(ids, 1)[ok]
        flip = s0[tuple(ch[ok].T)]
        ids[~flip] = ids[~flip][:, ::-1]
        quads.append(ids)
    Q = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    Q = Q[(Q >= 0).all(1)]
    if len(Q):
        # the grid's own orientation is consistent; check its sense once, on the majority
        nrm = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
        _, g = fam.vg(V[Q].mean(axis=1))
        if (np.einsum("ij,ij->i", nrm, g) < 0).mean() > 0.5:
            Q = Q[:, ::-1]
    return V, Q


def nonmanifold(Q):
    """Edges of a quad mesh not shared by exactly two quads."""
    if not len(Q):
        return 0
    E = np.sort(np.concatenate([Q[:, [k, (k + 1) % 4]] for k in range(4)]), axis=1)
    _, c = np.unique(E, axis=0, return_counts=True)
    return int((c != 2).sum())


def manifold_nets(fam, lo, hi, tries=16, split=True):
    """surface_nets on a grid shifted until the mesh is a closed manifold
    (a cell the surface crosses in the ambiguous way marching cubes cannot
    split moves off the grid), or the best of `tries` shifts; then (split)
    each try's pinches (split_pinches) pulled apart first. Returns (V, Q,
    non-manifold edges left)."""
    best = None
    rng = np.random.default_rng(7)
    for k in range(tries):
        shift = np.zeros(3) if k == 0 else rng.uniform(0, 1, 3) * fam.h
        V, Q = surface_nets(fam, np.asarray(lo, float) - shift, np.asarray(hi, float) - shift + fam.h)
        bad = nonmanifold(Q)
        if bad and split:
            V, Q = split_pinches(fam, V, Q)
            bad = nonmanifold(Q)
        if best is None or bad < best[0]:
            best = (bad, V, Q)
        if bad == 0:
            break
    bad, V, Q = best
    return V, Q, bad


def split_pinches(fam, V, Q):
    """Pull a quad mesh apart where two sheets of the surface pinch together
    at a near-zero saddle (two struts' fillets that barely touch, a skin
    kissed by a fillet): one cell meets the surface twice and its one vertex
    joins both, so a quad edge has four quads, or a vertex two separate
    fans. Each such edge's quads are paired (one each way along the edge,
    neighbours round it) and every vertex gets one copy per fan of its
    quads; a copy is moved a little towards its own fan and back onto the
    surface, so the sheets part by the width of the saddle. Returns (V, Q):
    a manifold mesh (every edge two quads, every vertex one fan)."""
    Q = np.array(Q, np.int64)
    if not len(Q):
        return V, Q
    # directed edges: quad q, side k runs Q[q, k] -> Q[q, k+1]
    a = Q.reshape(-1)
    b = np.roll(Q, -1, axis=1).reshape(-1)
    qid = np.repeat(np.arange(len(Q)), 4)
    lo_, hi_ = np.minimum(a, b), np.maximum(a, b)
    key = lo_ * (len(V) + 1) + hi_
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    many = np.flatnonzero(cnt > 2)
    # pairs: half-edge index -> partner half-edge index (normal edges: the other one)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    partner = -np.ones(len(a), np.int64)
    two = np.flatnonzero(cnt == 2)
    h0, h1 = order[starts[two]], order[starts[two] + 1]
    partner[h0], partner[h1] = h1, h0
    C = V[Q].mean(axis=1)
    for e in many:
        hs = order[starts[e]:starts[e] + cnt[e]]
        va, vb = V[lo_[hs[0]]], V[hi_[hs[0]]]
        ax = vb - va
        ax /= max(np.linalg.norm(ax), 1e-12)
        ref = np.cross(ax, [1.0, 0, 0])
        if np.linalg.norm(ref) < 0.1:
            ref = np.cross(ax, [0, 1.0, 0])
        ref /= np.linalg.norm(ref)
        ref2 = np.cross(ax, ref)
        d = C[qid[hs]] - va
        ang = np.arctan2(d @ ref2, d @ ref)
        hs = hs[np.argsort(ang)]
        # neighbours round the edge, one each way: two choices of pairing
        # (0-1, 2-3, ...) or (1-2, 3-0, ...); take the one whose pairs sit
        # closer in angle (the same sheet)
        m = len(hs)
        best = None
        for sh in (0, 1):
            pairs = [(hs[(i + sh) % m], hs[(i + sh + 1) % m]) for i in range(0, m, 2)]
            if any((a[x] == lo_[x]) == (a[y] == lo_[y]) for x, y in pairs):
                continue
            gap = sum(np.linalg.norm(C[qid[x]] - C[qid[y]]) for x, y in pairs)
            if best is None or gap < best[0]:
                best = (gap, pairs)
        if best is None:                                  # odd or unpaired: leave it
            continue
        for x, y in best[1]:
            partner[x], partner[y] = y, x
    # fans: at each vertex, quads joined across the paired edges at it
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    # a corner (q, k) is vertex Q[q, k]; its two sides are half-edges k
    # (leaving it) and k-1 (arriving); join the corner to the corner of
    # the partner quad at the same vertex
    corner = np.arange(len(a))
    side_out = corner                                    # half-edge leaving corner
    side_in = qid * 4 + (corner % 4 + 3) % 4            # half-edge arriving at corner
    rows, cols = [], []
    for hs_ in (side_out, side_in):
        p = partner[hs_]
        ok = p >= 0
        # partner half-edge runs the other way: its corner at this vertex is
        # its end (k+1) when hs_ leaves, its start when hs_ arrives
        pq, pk = p[ok] // 4, p[ok] % 4
        if hs_ is side_out:
            pc = pq * 4 + (pk + 1) % 4
        else:
            pc = pq * 4 + pk
        rows.append(corner[ok])
        cols.append(pc)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    good = a[rows] == a[cols]
    G = sp.coo_matrix((np.ones(good.sum()), (rows[good], cols[good])), shape=(len(a), len(a)))
    _, lab = connected_components(G, directed=False)
    # one vertex per (vertex, fan)
    vk = a * (len(a) + 1) + lab
    ufan, finv = np.unique(vk, return_inverse=True)
    vert = ufan // (len(a) + 1)
    first = np.zeros(len(V), bool)
    newid = np.empty(len(ufan), np.int64)
    extra = []
    for i, v in enumerate(vert):
        if not first[v]:
            first[v] = True
            newid[i] = v
        else:
            newid[i] = len(V) + len(extra)
            extra.append(i)
    Q2 = newid[finv].reshape(-1, 4)
    if extra:
        # each copy (and the vertex it came from) nudged towards its own fan
        split_v = np.unique(vert[extra])
        fans = np.flatnonzero(np.isin(vert, split_v))
        cen = np.zeros((len(ufan), 3))
        np.add.at(cen, finv, C[qid])
        cen /= np.maximum(np.bincount(finv, minlength=len(ufan)), 1)[:, None]
        P = V[vert[fans]] + 0.2 * (cen[fans] - V[vert[fans]])
        P = _project(fam, P, step=0.1 * fam.h)
        Vn = np.concatenate([V, np.zeros((len(extra), 3))])
        Vn[newid[fans]] = P
        V = Vn
    return V, Q2


_B = np.array([[comb(DEG, j) * x ** j * (1 - x) ** (DEG - j) for j in range(DEG + 1)]
               for x in np.linspace(0, 1, DEG + 1)])
_BI = np.linalg.inv(_B)
_BINT = np.linalg.inv(_B[1:-1, 1:-1])
_U = np.linspace(0, 1, DEG + 1)


def bern(t):
    t = np.atleast_1d(np.asarray(t, float))
    return np.stack([comb(DEG, j) * t ** j * (1 - t) ** (DEG - j) for j in range(DEG + 1)], -1)


def dbern(t):
    t = np.atleast_1d(np.asarray(t, float))
    d = DEG
    out = np.zeros(t.shape + (d + 1,))
    for j in range(d + 1):
        a = j * t ** (j - 1) * (1 - t) ** (d - j) if j > 0 else 0.0
        b = (d - j) * t ** j * (1 - t) ** (d - j - 1) if j < d else 0.0
        out[..., j] = comb(d, j) * (a - b)
    return out


def _resample(P, n):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, P[:, k]) for k in range(3)], 1)


class Sheet:
    """One family's patches: V, Q, edge poles {(a, b) a < b: (4, 3) from a
    to b}, patch poles C (m, 4, 4, 3) (C[i, j]: i along u, j along v; u
    from corner 0 to 1, v from corner 0 to 3)."""

    def __init__(self, fam, V, Q):
        self.fam, self.V, self.Q = fam, V, Q
        E = np.sort(np.concatenate([Q[:, [k, (k + 1) % 4]] for k in range(4)]), axis=1)
        self.E = np.unique(E, axis=0)
        self.eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(self.E)}
        n = DEG + 1
        A = self.E
        # drape: onto the surface, evenly spaced by arc length
        t = np.linspace(0, 1, 9)[1:-1][None, :, None]
        P = V[A[:, 0]][:, None] * (1 - t) + V[A[:, 1]][:, None] * t
        for r in range(4):
            P = _project(fam, P.reshape(-1, 3), step=0.5 * fam.h).reshape(P.shape)
            full = np.concatenate([V[A[:, 0]][:, None], P, V[A[:, 1]][:, None]], 1)
            P = np.stack([_resample(x, 9 if r < 3 else n) for x in full])[:, 1:-1]
        P = _project(fam, P.reshape(-1, 3), step=0.5 * fam.h).reshape(P.shape)
        full = np.concatenate([V[A[:, 0]][:, None], P, V[A[:, 1]][:, None]], 1)
        self.EP = np.einsum("ij,ejk->eik", _BI, full)           # (e, 4, 3) poles a -> b
        # an edge whose drape ran off (a pull onto another sheet, a fold)
        # is the chord: wrong by its sag, never by a spike
        S = np.einsum("ta,eax->etx", bern(np.linspace(0, 1, 9)), self.EP)
        a0, a1 = V[A[:, 0]], V[A[:, 1]]
        L = np.linalg.norm(a1 - a0, axis=1)
        wild = np.linalg.norm(S - 0.5 * (a0 + a1)[:, None], axis=2).max(1) > 0.5 * L + 0.5 * fam.h
        if wild.any():
            t_ = _U[None, :, None]
            self.EP[wild] = np.einsum("ij,ejk->eik", _BI, a0[wild][:, None] * (1 - t_) + a1[wild][:, None] * t_)
        self.wild_edges = int(wild.sum())
        sides = [self.curve(Q[:, 0], Q[:, 1]), self.curve(Q[:, 3], Q[:, 2]),
                 self.curve(Q[:, 0], Q[:, 3]), self.curve(Q[:, 1], Q[:, 2])]
        sb, st, sl, sr = (np.einsum("ij,mjk->mik", _B, s) for s in sides)
        Uu, W = np.meshgrid(_U, _U, indexing="ij")
        X = ((1 - W)[None, ..., None] * sb[:, :, None] + W[None, ..., None] * st[:, :, None]
             + (1 - Uu)[None, ..., None] * sl[:, None] + Uu[None, ..., None] * sr[:, None]
             - ((1 - Uu) * (1 - W))[None, ..., None] * sb[:, :1, None] - (Uu * (1 - W))[None, ..., None] * sb[:, -1:, None]
             - ((1 - Uu) * W)[None, ..., None] * st[:, :1, None] - (Uu * W)[None, ..., None] * st[:, -1:, None])
        X0 = X.copy()
        inner = _project(fam, X[:, 1:-1, 1:-1].reshape(-1, 3), step=0.5 * fam.h)
        X[:, 1:-1, 1:-1] = inner.reshape(len(Q), n - 2, n - 2, 3)
        # an inside point pulled further than the patch is across went to
        # another sheet: that patch keeps its Coons blend
        span = np.linalg.norm(X0[:, 0, 0] - X0[:, -1, -1], axis=1) + np.linalg.norm(X0[:, -1, 0] - X0[:, 0, -1], axis=1)
        moved = np.linalg.norm(X - X0, axis=3).max((1, 2))
        off = moved > 0.35 * span
        X[off] = X0[off]
        self.wild_patches = int(off.sum())
        C = np.zeros((len(Q), n, n, 3))
        C[:, :, 0], C[:, :, -1], C[:, 0, :], C[:, -1, :] = sides[0], sides[1], sides[2], sides[3]
        Cb = C.copy()
        Cb[:, 1:-1, 1:-1] = 0
        R = X[:, 1:-1, 1:-1] - np.einsum("ia,mabk,jb->mijk", _B[1:-1], Cb, _B[1:-1])
        C[:, 1:-1, 1:-1] = np.einsum("ia,mabk,jb->mijk", _BINT, R, _BINT)
        self.C = C

    def curve(self, a, b):
        """Poles of edges a -> b (arrays of vertex ids): (m, 4, 3)."""
        a, b = np.asarray(a), np.asarray(b)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        i = np.array([self.eidx[(int(x), int(y))] for x, y in zip(lo, hi)], np.int64)
        P = self.EP[i]
        rev = a > b
        P[rev] = P[rev][:, ::-1]
        return P

    def at(self, qi, u, v):
        """Patch points: qi (k,), u, v (k,) or grids broadcast."""
        return np.einsum("ka,kabx,kb->kx", bern(u), self.C[qi], bern(v))

    def grid(self, qi, m):
        """(len(qi), m, m, 3) points at an m x m grid of (u, v)."""
        B = bern(np.linspace(0, 1, m))
        return np.einsum("ua,qabx,vb->quvx", B, self.C[qi], B)

    def invert(self, q, x, m=9, it=12):
        """(u, v) of the point of patch q closest to x, not clamped: outside
        [0, 1]^2 when x is past the patch's sides."""
        G = self.grid(np.array([q]), m)[0]
        k = int(np.argmin(np.linalg.norm(G.reshape(-1, 3) - x, axis=1)))
        u, v = (k // m) / (m - 1), (k % m) / (m - 1)
        C = self.C[q]
        for _ in range(it):
            bu, bv = bern(u)[0], bern(v)[0]
            du, dv = dbern(u)[0], dbern(v)[0]
            P = np.einsum("a,abx,b->x", bu, C, bv)
            Pu = np.einsum("a,abx,b->x", du, C, bv)
            Pv = np.einsum("a,abx,b->x", bu, C, dv)
            J = np.stack([Pu, Pv], 1)
            step = np.linalg.lstsq(J, x - P, rcond=None)[0]
            u, v = u + float(step[0]), v + float(step[1])
            if abs(step).max() < 1e-9:
                break
        return u, v

    def folds(self, m=5):
        """Patches whose normal turns against the surface's somewhere (a
        fold: the patch is too coarse for the surface there): (m,) bool."""
        t = np.linspace(0.05, 0.95, m)
        B, D = bern(t), dbern(t)
        C = self.C
        P = np.einsum("ua,qabx,vb->quvx", B, C, B)
        Su = np.einsum("ua,qabx,vb->quvx", D, C, B)
        Sv = np.einsum("ua,qabx,vb->quvx", B, C, D)
        N = np.cross(Su, Sv)
        _, g = self.fam.vg(P.reshape(-1, 3))
        s = np.sign(np.einsum("ix,ix->i", N.reshape(-1, 3), g)).reshape(len(C), -1)
        major = np.sign(s.sum()) or 1.0
        return (s != major).any(1)

    def deviation(self, m=7):
        return self.deviation_of(np.arange(len(self.Q)), m)

    def deviation_of(self, qi, m=7):
        """Distance (mm, first order) of patches qi from the level set at an
        m x m grid each."""
        P = self.grid(np.asarray(qi), m).reshape(-1, 3)
        v, g = self.fam.vg(P)
        return np.abs(v) / np.maximum(np.linalg.norm(g, axis=1), 1e-9)


