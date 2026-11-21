"""
Closest points on a triangle mesh, compiled: each query point against its
k candidate triangles in one pass (Ericson, Real-Time Collision Detection
5.1.5), in parallel over the points. The numpy route (fields.
_closest_on_triangles over k copies of every point) builds a dozen
(N k, 3) temporaries per call; here nothing but the answer is written.
None of it when numba is missing: brep._closest falls back to numpy.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _nb
except ImportError:  # pragma: no cover - numba is a dependency, kept optional
    _nb = None


def available() -> bool:
    return _nb is not None


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, inline="always")
    def _one(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
        abx, aby, abz = bx - ax, by - ay, bz - az
        acx, acy, acz = cx - ax, cy - ay, cz - az
        apx, apy, apz = px - ax, py - ay, pz - az
        d1 = abx * apx + aby * apy + abz * apz
        d2 = acx * apx + acy * apy + acz * apz
        if d1 <= 0.0 and d2 <= 0.0:
            return ax, ay, az
        bpx, bpy, bpz = px - bx, py - by, pz - bz
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0.0 and d4 <= d3:
            return bx, by, bz
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            t = d1 / (d1 - d3)
            return ax + t * abx, ay + t * aby, az + t * abz
        cpx, cpy, cpz = px - cx, py - cy, pz - cz
        d5 = abx * cpx + aby * cpy + abz * cpz
        d6 = acx * cpx + acy * cpy + acz * cpz
        if d6 >= 0.0 and d5 <= d6:
            return cx, cy, cz
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            t = d2 / (d2 - d6)
            return ax + t * acx, ay + t * acy, az + t * acz
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return bx + t * (cx - bx), by + t * (cy - by), bz + t * (cz - bz)
        s = va + vb + vc
        if s == 0.0:                       # degenerate triangle: its first corner
            return ax, ay, az
        v = vb / s
        w = vc / s
        return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w

    @_nb.njit(cache=True, nogil=True, parallel=True)
    def _closest_k(P, tri, cand, out, best):
        n, k = cand.shape
        for i in _nb.prange(n):
            px, py, pz = P[i, 0], P[i, 1], P[i, 2]
            dmin = np.inf
            for j in range(k):
                t = cand[i, j]
                qx, qy, qz = _one(px, py, pz, tri[t, 0, 0], tri[t, 0, 1], tri[t, 0, 2],
                                  tri[t, 1, 0], tri[t, 1, 1], tri[t, 1, 2], tri[t, 2, 0], tri[t, 2, 1], tri[t, 2, 2])
                d = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2
                if d < dmin:
                    dmin = d
                    out[i, 0], out[i, 1], out[i, 2] = qx, qy, qz
                    best[i] = t


def closest_k(P, tri, cand):
    """Closest point to each P[i] over triangles tri[cand[i]]: (points
    (N, 3), the index of the triangle each lies on)."""
    P = np.ascontiguousarray(P, np.float64)
    out = np.empty_like(P)
    best = np.empty(len(P), np.int64)
    _closest_k(P, np.ascontiguousarray(tri, np.float64), np.ascontiguousarray(cand, np.int64), out, best)
    return out, best
