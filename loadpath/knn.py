"""
k nearest neighbours, compiled: the same neighbours, in the same order, as
scipy's cKDTree.query(P, k), several times faster on the queries a mesher
makes.

MeshField reads a mesh's exact distance from the triangles behind the k=12
nearest centroids of each point; on the robot-arm links (500k triangles, a 10 mm
band) that query was ~80% of evaluating the whole field graph. This is a
k-d tree (median splits on the widest axis, 16 points a leaf) searched
depth first, nearer child first, in numba (nogil: peval's threads share
it). What makes it fast is the warm start: the points a mesher asks about
come in order (a block of grid samples, a mesh's vertices), so the previous
point's k neighbours, measured from this one, bound its k-th distance
before the search starts, and the search only opens the leaves inside that
ball.

The answer is the exact k nearest by squared distance, nearest first;
centroids at exactly equal distances come in index order here and in the
tree's own order in scipy, the only way the two can differ (none did over
millions of the arm and arm-link queries, tests/test_fieldjit.py).

`available()` is False without numba or with LOADPATH_KNN=kdtree.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import numba as nb

    from . import meshops as _meshops  # noqa: F401 - caps numba's threads
except ImportError:  # pragma: no cover
    nb = None

__all__ = ["available", "KDTree"]

_LEAF = 16


def available():
    return nb is not None and os.environ.get("LOADPATH_KNN", "").lower() != "kdtree"


if nb is not None:

    @nb.njit(cache=True, nogil=True)
    def _build(pts, leaf):
        n = pts.shape[0]
        idx = np.arange(n)
        cap = 4 * (n // leaf + 1) + 8         # leaves hold leaf/2 .. leaf points
        start = np.empty(cap, np.int64)
        end = np.empty(cap, np.int64)
        left = np.full(cap, -1, np.int64)
        lo = np.empty((cap, 3))
        hi = np.empty((cap, 3))
        stack = np.empty(cap, np.int64)
        start[0], end[0] = 0, n
        nn = 1
        sp = 0
        stack[sp] = 0
        sp += 1
        while sp > 0:
            sp -= 1
            nd = stack[sp]
            s, e = start[nd], end[nd]
            for d in range(3):
                a = np.inf
                b = -np.inf
                for t in range(s, e):
                    v = pts[idx[t], d]
                    if v < a:
                        a = v
                    if v > b:
                        b = v
                lo[nd, d] = a
                hi[nd, d] = b
            if e - s <= leaf:
                continue
            dim = 0
            w = hi[nd, 0] - lo[nd, 0]
            for d in range(1, 3):
                if hi[nd, d] - lo[nd, d] > w:
                    w = hi[nd, d] - lo[nd, d]
                    dim = d
            sub = idx[s:e].copy()
            key = np.empty(e - s)
            for t in range(e - s):
                key[t] = pts[sub[t], dim]
            order = np.argsort(key, kind="mergesort")
            for t in range(e - s):
                idx[s + t] = sub[order[t]]
            mid = (s + e) // 2
            left[nd] = nn
            start[nn], end[nn] = s, mid
            start[nn + 1], end[nn + 1] = mid, e
            stack[sp] = nn
            stack[sp + 1] = nn + 1
            sp += 2
            nn += 2
        return idx, start[:nn].copy(), end[:nn].copy(), left[:nn].copy(), lo[:nn].copy(), hi[:nn].copy()

    @nb.njit(cache=True, nogil=True, inline="always")
    def _box2(px, py, pz, lo, hi, nd):
        # squared distance from the point to the node's box
        d = 0.0
        for a in range(3):
            v = px if a == 0 else (py if a == 1 else pz)
            if v < lo[nd, a]:
                u = lo[nd, a] - v
                d += u * u
            elif v > hi[nd, a]:
                u = v - hi[nd, a]
                d += u * u
        return d

    @nb.njit(cache=True, nogil=True)
    def _query(Q, pts, spts, idx, start, end, left, lo, hi, k, out_i):
        bestd = np.empty(k)
        besti = np.empty(k, np.int64)
        stack = np.empty(256, np.int64)
        for i in range(Q.shape[0]):
            px, py, pz = Q[i, 0], Q[i, 1], Q[i, 2]
            bound = np.inf
            if i > 0:
                # the previous point's k neighbours: k points within the largest
                # of their distances from this one, so its k-th nearest is too
                bound = 0.0
                for j in range(k):
                    p = out_i[i - 1, j]
                    dx = px - pts[p, 0]
                    dy = py - pts[p, 1]
                    dz = pz - pts[p, 2]
                    d = dx * dx + dy * dy + dz * dz
                    if d > bound:
                        bound = d
                bound = bound * (1.0 + 1e-12) + 1e-300
            nb_ = 0
            sp = 0
            stack[sp] = 0
            sp += 1
            while sp > 0:
                sp -= 1
                nd = stack[sp]
                lim = bestd[k - 1] if nb_ == k else bound
                if _box2(px, py, pz, lo, hi, nd) > lim * (1.0 + 1e-12):
                    continue
                c = left[nd]
                if c < 0:
                    for t in range(start[nd], end[nd]):
                        p = idx[t]
                        dx = px - spts[t, 0]          # the points in tree order: a leaf is contiguous
                        dy = py - spts[t, 1]
                        dz = pz - spts[t, 2]
                        d = dx * dx + dy * dy + dz * dz
                        if nb_ < k:
                            if d > bound:
                                continue
                            j = nb_
                            nb_ += 1
                        elif d < bestd[k - 1] or (d == bestd[k - 1] and p < besti[k - 1]):
                            j = k - 1
                        else:
                            continue
                        # insert: by distance, then index (a fixed order for ties)
                        while j > 0 and (bestd[j - 1] > d or (bestd[j - 1] == d and besti[j - 1] > p)):
                            bestd[j] = bestd[j - 1]
                            besti[j] = besti[j - 1]
                            j -= 1
                        bestd[j] = d
                        besti[j] = p
                    continue
                # the nearer child on top of the stack
                d0 = _box2(px, py, pz, lo, hi, c)
                d1 = _box2(px, py, pz, lo, hi, c + 1)
                if d0 <= d1:
                    stack[sp] = c + 1
                    stack[sp + 1] = c
                else:
                    stack[sp] = c
                    stack[sp + 1] = c + 1
                sp += 2
            for j in range(k):
                out_i[i, j] = besti[j]


class KDTree:
    """cKDTree(pts).query(P, k)[1], compiled (see the module docstring)."""

    def __init__(self, pts, leaf=_LEAF):
        self.pts = np.ascontiguousarray(pts, np.float64)
        if len(self.pts) == 0:
            raise ValueError("KDTree needs points")
        self.idx, self.start, self.end, self.left, self.lo, self.hi = _build(self.pts, int(leaf))
        self.spts = np.ascontiguousarray(self.pts[self.idx])

    def query(self, P, k):
        P = np.ascontiguousarray(np.asarray(P, np.float64).reshape(-1, 3))
        k = min(int(k), len(self.pts))
        out = np.empty((len(P), k), np.int64)
        if len(P):
            _query(P, self.pts, self.spts, self.idx, self.start, self.end, self.left, self.lo, self.hi, k, out)
        return out
