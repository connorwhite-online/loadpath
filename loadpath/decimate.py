"""Adaptive decimation of a field mesh against the field itself.

Marching cubes puts triangles a pitch apart everywhere: a flat face gets
as many as a fillet. `decimate` collapses edges, shortest first, wherever
the result stays within `tol` mm of the true surface, so flat and gently
curved regions thin out and creases, small fillets and thin lattice necks
keep their triangles.

Each pass works on many edges at once (numpy, no priority queue):

1. candidates: edges whose two ends share exactly two neighbours (the
   link condition, so the collapse keeps the mesh a closed manifold) and
   touch no locked vertex;
2. an independent set: an edge goes ahead when it is the shortest
   candidate around both its ends and their neighbours, so no two
   collapses in a pass share a triangle;
3. each collapse moves the pair to its midpoint projected onto the field's
   zero level (two Newton steps) and is kept only if every triangle it
   reshapes stays within `tol` of the surface at its centroid and edge
   midpoints (|field| there, which is the distance for the distance-like
   fields in fields.py) and does not turn over (normal within ~60 degrees
   of before).

Passes repeat until one removes under 1% of the triangles. The mesh
stays watertight, and every vertex stays on the surface.
"""
from __future__ import annotations

import numpy as np

__all__ = ["decimate"]


def _normals(T):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def decimate(V, F, field, tol, pitch, locked=None, max_passes=80, min_gain=0.01, progress=None,
             target=None, place=None, turn=0.5, labels=None, allow=None, max_reject=3, quality=None):
    """(V, F, stats, kept): the mesh with edges collapsed wherever it stays within
    `tol` mm of field = 0. `locked` vertices (a mask) never move or go:
    the streamed export's seams. Unused vertices are dropped (locked ones
    keep their order relative to each other); `kept` is the mask of the
    input vertices that remain.

    With `field` None there is no surface to hold to: a collapse of edges
    (a, b) puts the new vertex at `place(V, a, b)` (the midpoints if not
    given), and passes stop once the mesh is down to `target` faces.
    Shortest edges first makes that an even coarsening. `turn` is the
    least cosine between a reshaped face's normal and its old one.
    `labels` (one per face) ride along with the faces and come back in
    stats["labels"]; `allow(E, F, labels)` filters each pass's candidate
    edges (a mask), say to keep region seams in place. A vertex whose
    collapses failed `max_reject` times is left alone. `quality(Told, Tnew,
    labels)` (the reshaped triangles' corners before and after, (k, 3, 3),
    and their labels) can veto collapses too."""
    from .fields import _project
    from .narrowband import peval

    V = np.array(V, float)
    F = np.array(F, np.int64)
    n = len(V)
    lock = np.zeros(n, bool) if locked is None else np.array(locked, bool)
    alive = np.ones(n, bool)
    rej = np.zeros(n, np.int64)
    f0 = len(F)
    lab = None if labels is None else np.asarray(labels).copy()
    rng = np.random.default_rng(0)
    passes = 0
    for passes in range(1, max_passes + 1):
        nf = len(F)
        a = np.r_[F[:, 0], F[:, 1], F[:, 2]]
        b = np.r_[F[:, 1], F[:, 2], F[:, 0]]
        # closed manifold: every edge is a half-edge each way, so the edges
        # are the half-edges with a < b
        h = a < b
        E = np.stack([a[h], b[h]], axis=1)
        # an edge that failed before yields to its neighbours, or it would
        # block them pass after pass; a vertex that failed three times is done
        frozen = lock | (rej >= max_reject)
        E = E[~(frozen[E[:, 0]] | frozen[E[:, 1]])]
        if allow is not None and len(E):
            E = E[allow(E, F, lab)]
        if not len(E):
            break
        d = V[E[:, 0]] - V[E[:, 1]]
        prio = (np.einsum("ij,ij->i", d, d) * np.exp2(2 * (rej[E[:, 0]] + rej[E[:, 1]]))
                * (1.0 + 1e-6 * rng.random(len(E))))
        # independent set: shortest around both ends and their neighbours
        m = np.full(n, np.inf)
        np.minimum.at(m, E[:, 0], prio)
        np.minimum.at(m, E[:, 1], prio)
        M = m.copy()
        np.minimum.at(M, a, m[b])
        win = (prio == M[E[:, 0]]) & (prio == M[E[:, 1]])
        W = E[win]
        if not len(W):
            break
        # the triangles around each collapse: no triangle holds ends of two
        # collapses (their ends are not neighbours), so each belongs to at
        # most one
        wid = np.full(n, -1)
        wid[W[:, 0]] = np.arange(len(W))
        wid[W[:, 1]] = np.arange(len(W))
        fw = wid[F].max(axis=1)
        fi = np.flatnonzero(fw >= 0)
        tri, w = F[fi], fw[fi]
        wa, wb = W[w, 0], W[w, 1]
        in_a = (tri == wa[:, None]).any(axis=1)
        in_b = (tri == wb[:, None]).any(axis=1)
        # link condition: the two ends share exactly the two opposite vertices
        nk = []
        for side in (in_a, in_b):
            t, ww = tri[side], w[side]
            v = t.ravel()
            k = np.repeat(ww, 3) * n + v
            k = k[(v != np.repeat(W[ww, 0], 3)) & (v != np.repeat(W[ww, 1], 3))]
            nk.append(np.unique(k))
        common = np.bincount(np.intersect1d(nk[0], nk[1], assume_unique=True) // n, minlength=len(W))
        link = common == 2
        rej[W[~link].ravel()] += 1
        sel = link[w] & ~(in_a & in_b)            # the two triangles on the edge go
        tri, w, wa, wb = tri[sel], w[sel], wa[sel], wb[sel]
        pr = np.stack([w, fi[sel]], axis=1)
        # the new vertex: midpoint, onto the surface
        P = np.zeros((len(W), 3))
        li = np.flatnonzero(link)
        if len(li) and field is None:
            P[li] = (place(V, W[li, 0], W[li, 1]) if place is not None
                     else 0.5 * (V[W[li, 0]] + V[W[li, 1]]))
        elif len(li):
            q = _project(field, 0.5 * (V[W[li, 0]] + V[W[li, 1]]), pitch)
            P[li] = _project(field, q, pitch)
        Told = V[tri]
        Tnew = Told.copy()
        hit = (tri == wa[:, None]) | (tri == wb[:, None])
        Tnew[hit] = np.repeat(P[pr[:, 0]], hit.sum(axis=1), axis=0)
        good = (_normals(Told) * _normals(Tnew)).sum(axis=1) > turn
        if quality is not None and len(Tnew):
            good &= quality(Told, Tnew, None if lab is None else lab[pr[:, 1]])
        if field is not None:
            S = np.concatenate([Tnew.mean(axis=1), 0.5 * (Tnew[:, 0] + Tnew[:, 1]),
                                0.5 * (Tnew[:, 1] + Tnew[:, 2]), 0.5 * (Tnew[:, 2] + Tnew[:, 0])])
            dev = np.abs(peval(field, S)).reshape(4, -1).max(axis=0)
            good &= dev <= tol
        bad = ~link
        np.logical_or.at(bad, pr[:, 0], ~good)
        acc = ~bad
        rej[W[bad & link].ravel()] += 1
        if target is not None:              # each collapse takes two triangles: do not overshoot
            room = max(0, (len(F) - int(target)) // 2)
            acc[np.flatnonzero(acc)[room:]] = False
        if not acc.any():
            break
        Wa, Wb = W[acc, 0], W[acc, 1]
        V[Wa] = P[acc]
        remap = np.arange(n)
        remap[Wb] = Wa
        alive[Wb] = False
        F = remap[F]
        keepf = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[keepf]
        if lab is not None:
            lab = lab[keepf]
        if progress:
            progress({"stage": "decimate", "pass": passes, "faces": int(len(F))})
        if nf - len(F) < min_gain * nf or (target is not None and len(F) <= target):
            break
    used = np.zeros(n, bool)
    used[F] = True
    used |= lock & alive
    new = np.cumsum(used) - 1
    stats = {"tolerance": float(tol), "facesBefore": int(f0), "faces": int(len(F)), "passes": int(passes)}
    if lab is not None:
        stats["labels"] = lab
    return V[used], new[F], stats, used
