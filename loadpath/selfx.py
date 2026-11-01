"""Self-intersection search over a triangle mesh, as a compiled kernel.

`crossing_faces(V, F, faces)` gives the faces among `faces` that cross
another face of the mesh (sharing no vertex with it), with the same exact
test as crease.crossing_faces' numpy path (no crossing if one triangle
lies wholly on one side of the other's plane; else a crossing if an edge
of either passes strictly through the other).

The numpy path finds candidate pairs with a k-d tree over face centroids
and then tests ~50 candidates per face as arrays: on a lattice of a few
million faces that was most of the meshing time (the relax's unfold and
the crease pass call it some twenty times a mesh). Here the query faces'
bounding boxes go into a spatial hash (cells about the size of the larger
faces), and every face of the mesh near them (a coarse bitmap first)
looks up the query faces whose boxes overlap its own and tests each pair
once, in parallel, with no intermediate arrays: ~35x faster on 4 cores.
Candidates are found exactly by box overlap, so this also finds the rare
crossing the centroid search's radius missed (a face much longer than
the 99th percentile).

Needs numba (a loadpath dependency); without it `available()` is False
and crease.crossing_faces keeps its numpy path. LOADPATH_SELFX=numpy
forces that path.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["available", "crossing_faces"]

_FAILED = False


def available():
    global _FAILED
    if _FAILED or os.environ.get("LOADPATH_SELFX", "").lower() == "numpy":
        return False
    try:
        _kernels()
    except Exception:  # noqa: BLE001 - no numba, or it can't compile here: the numpy path
        _FAILED = True
        return False
    return True


try:
    import numba as nb

    from . import meshops as _meshops  # noqa: F401 - caps numba's threads (LOADPATH_THREADS)
except ImportError:  # pragma: no cover - numba is a dependency; the numpy path without it
    nb = None


def _kernels():
    if nb is None:
        raise ImportError("numba")
    return boxes, build, query


if nb is not None:

    @nb.njit(cache=True)
    def query_box(lo, hi, faces):
        qlo = np.full(3, np.inf)
        qhi = np.full(3, -np.inf)
        for a in faces:
            for k in range(3):
                qlo[k] = min(qlo[k], lo[a, k])
                qhi[k] = max(qhi[k], hi[a, k])
        return qlo, qhi

    @nb.njit(cache=True, inline="always")
    def _hash(ix, iy, iz, mask):
        h = (ix * np.int64(73856093)) ^ (iy * np.int64(19349663)) ^ (iz * np.int64(83492791))
        return h & mask

    @nb.njit(cache=True, parallel=True)
    def boxes(V, F):
        n = F.shape[0]
        lo = np.empty((n, 3))
        hi = np.empty((n, 3))
        for i in nb.prange(n):
            for k in range(3):
                a = V[F[i, 0], k]
                b = V[F[i, 1], k]
                c = V[F[i, 2], k]
                lo[i, k] = min(a, b, c)
                hi[i, k] = max(a, b, c)
        return lo, hi

    @nb.njit(cache=True)
    def build(lo, hi, ids, origin, h, mask):
        """CSR spatial hash of the faces `ids`: items[count[b]:count[b + 1]]
        are the faces in bucket b."""
        n = ids.shape[0]
        nb_ = mask + 1
        count = np.zeros(nb_ + 1, np.int64)
        for r in range(n):
            i = ids[r]
            x0 = np.int64(np.floor((lo[i, 0] - origin[0]) / h))
            y0 = np.int64(np.floor((lo[i, 1] - origin[1]) / h))
            z0 = np.int64(np.floor((lo[i, 2] - origin[2]) / h))
            x1 = np.int64(np.floor((hi[i, 0] - origin[0]) / h))
            y1 = np.int64(np.floor((hi[i, 1] - origin[1]) / h))
            z1 = np.int64(np.floor((hi[i, 2] - origin[2]) / h))
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    for iz in range(z0, z1 + 1):
                        count[_hash(ix, iy, iz, mask) + 1] += 1
        for b in range(nb_):
            count[b + 1] += count[b]
        fill = count[:-1].copy()
        items = np.empty(count[nb_], np.int64)
        for r in range(n):
            i = ids[r]
            x0 = np.int64(np.floor((lo[i, 0] - origin[0]) / h))
            y0 = np.int64(np.floor((lo[i, 1] - origin[1]) / h))
            z0 = np.int64(np.floor((lo[i, 2] - origin[2]) / h))
            x1 = np.int64(np.floor((hi[i, 0] - origin[0]) / h))
            y1 = np.int64(np.floor((hi[i, 1] - origin[1]) / h))
            z1 = np.int64(np.floor((hi[i, 2] - origin[2]) / h))
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    for iz in range(z0, z1 + 1):
                        b = _hash(ix, iy, iz, mask)
                        items[fill[b]] = i
                        fill[b] += 1
        return count, items

    @nb.njit(cache=True, inline="always")
    def seg_tri(p, q, t0, t1, t2):
        # crease._seg_tri for one segment and one triangle (eps 1e-9)
        eps = 1e-9
        d0, d1, d2 = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        e10, e11, e12 = t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2]
        e20, e21, e22 = t2[0] - t0[0], t2[1] - t0[1], t2[2] - t0[2]
        h0 = d1 * e22 - d2 * e21
        h1 = d2 * e20 - d0 * e22
        h2 = d0 * e21 - d1 * e20
        a = e10 * h0 + e11 * h1 + e12 * h2
        if not abs(a) > 1e-14:
            return False
        f = 1.0 / a
        s0, s1, s2 = p[0] - t0[0], p[1] - t0[1], p[2] - t0[2]
        u = f * (s0 * h0 + s1 * h1 + s2 * h2)
        if not u > eps:
            return False
        q0 = s1 * e12 - s2 * e11
        q1 = s2 * e10 - s0 * e12
        q2 = s0 * e11 - s1 * e10
        v = f * (d0 * q0 + d1 * q1 + d2 * q2)
        if not (v > eps and u + v < 1 - eps):
            return False
        t = f * (e20 * q0 + e21 * q1 + e22 * q2)
        return t > eps and t < 1 - eps

    @nb.njit(cache=True, inline="always")
    def apart(V, FX, FY):
        # every vertex of Y strictly on one side of X's plane
        x0, x1, x2 = V[FX[0]], V[FX[1]], V[FX[2]]
        a0, a1, a2 = x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]
        b0, b1, b2 = x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]
        n0 = a1 * b2 - a2 * b1
        n1 = a2 * b0 - a0 * b2
        n2 = a0 * b1 - a1 * b0
        tol = 1e-12 * (n0 * n0 + n1 * n1 + n2 * n2)
        pos = 0
        neg = 0
        for k in range(3):
            y = V[FY[k]]
            s = (y[0] - x0[0]) * n0 + (y[1] - x0[1]) * n1 + (y[2] - x0[2]) * n2
            if s > tol:
                pos += 1
            elif s < -tol:
                neg += 1
        return pos == 3 or neg == 3

    @nb.njit(cache=True, inline="always")
    def pair_hit(V, fa, fb):
        """Do faces fa and fb (vertex triples) cross, sharing no vertex?"""
        for u in range(3):
            for w in range(3):
                if fa[u] == fb[w]:
                    return False
        if apart(V, fa, fb) or apart(V, fb, fa):
            return False
        for k in range(3):
            if seg_tri(V[fa[k]], V[fa[(k + 1) % 3]], V[fb[0]], V[fb[1]], V[fb[2]]):
                return True
            if seg_tri(V[fb[k]], V[fb[(k + 1) % 3]], V[fa[0]], V[fa[1]], V[fa[2]]):
                return True
        return False

    @nb.njit(cache=True)
    def coarse_marks(lo, hi, faces, origin, H, cmask):
        """A bitmap (hashed) of the coarse cells, H across, the query
        faces' boxes reach."""
        m = np.zeros(cmask + 1, np.bool_)
        for r in range(faces.shape[0]):
            i = faces[r]
            x0 = np.int64(np.floor((lo[i, 0] - origin[0]) / H))
            y0 = np.int64(np.floor((lo[i, 1] - origin[1]) / H))
            z0 = np.int64(np.floor((lo[i, 2] - origin[2]) / H))
            x1 = np.int64(np.floor((hi[i, 0] - origin[0]) / H))
            y1 = np.int64(np.floor((hi[i, 1] - origin[1]) / H))
            z1 = np.int64(np.floor((hi[i, 2] - origin[2]) / H))
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    for iz in range(z0, z1 + 1):
                        m[_hash(ix, iy, iz, cmask)] = True
        return m

    @nb.njit(cache=True, parallel=True)
    def query(V, F, lo, hi, count, items, origin, h, mask, cm, H, cmask, out):
        """Every face of the mesh whose box reaches a coarse cell of the
        query faces looks up the query faces whose boxes overlap its own
        (the hash holds only the query faces), and tests each pair once."""
        for a in nb.prange(F.shape[0]):
            # the coarse cells first: most faces are nowhere near a query face
            x0 = np.int64(np.floor((lo[a, 0] - origin[0]) / H))
            y0 = np.int64(np.floor((lo[a, 1] - origin[1]) / H))
            z0 = np.int64(np.floor((lo[a, 2] - origin[2]) / H))
            x1 = np.int64(np.floor((hi[a, 0] - origin[0]) / H))
            y1 = np.int64(np.floor((hi[a, 1] - origin[1]) / H))
            z1 = np.int64(np.floor((hi[a, 2] - origin[2]) / H))
            near = False
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    for iz in range(z0, z1 + 1):
                        if cm[_hash(ix, iy, iz, cmask)]:
                            near = True
            if not near:
                continue
            fa = F[a]
            x0 = np.int64(np.floor((lo[a, 0] - origin[0]) / h))
            y0 = np.int64(np.floor((lo[a, 1] - origin[1]) / h))
            z0 = np.int64(np.floor((lo[a, 2] - origin[2]) / h))
            x1 = np.int64(np.floor((hi[a, 0] - origin[0]) / h))
            y1 = np.int64(np.floor((hi[a, 1] - origin[1]) / h))
            z1 = np.int64(np.floor((hi[a, 2] - origin[2]) / h))
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    for iz in range(z0, z1 + 1):
                        bk = _hash(ix, iy, iz, mask)
                        for j in range(count[bk], count[bk + 1]):
                            b = items[j]
                            if b == a:
                                continue
                            if (lo[b, 0] > hi[a, 0] or lo[b, 1] > hi[a, 1] or lo[b, 2] > hi[a, 2]
                                    or hi[b, 0] < lo[a, 0] or hi[b, 1] < lo[a, 1] or hi[b, 2] < lo[a, 2]):
                                continue
                            # test the pair once: in the cell holding the
                            # low corner of the boxes' overlap
                            cx = np.int64(np.floor((max(lo[a, 0], lo[b, 0]) - origin[0]) / h))
                            cy = np.int64(np.floor((max(lo[a, 1], lo[b, 1]) - origin[1]) / h))
                            cz = np.int64(np.floor((max(lo[a, 2], lo[b, 2]) - origin[2]) / h))
                            if cx != ix or cy != iy or cz != iz:
                                continue
                            if pair_hit(V, fa, F[b]):
                                out[a] = 1
                                out[b] = 1


def crossing_faces(V, F, faces):
    """Faces among `faces` that cross another face of (V, F) (sharing no
    vertex with it), and the faces they cross: crease.crossing_faces."""
    faces = np.ascontiguousarray(faces, np.int64)
    if not len(faces):
        return faces
    _kernels()
    V = np.ascontiguousarray(V, np.float64)
    F = np.ascontiguousarray(F, np.int64)
    lo, hi = boxes(V, F)
    qlo, qhi = query_box(lo, hi, faces)
    # cells about a large face (90th percentile of a sample): most faces
    # sit in 1-8 cells
    s = faces if len(faces) <= 100_000 else faces[:: len(faces) // 100_000]
    ext = (hi[s] - lo[s]).max(axis=1)
    h = float(max(np.percentile(ext, 90), 1e-9 * max(float(np.abs(qhi - qlo).max()), 1e-30)))
    origin = qlo - 0.5 * h
    n = len(faces)
    mask = np.int64((1 << max(10, int(np.ceil(np.log2(max(4 * n, 1)))))) - 1)
    count, items = build(lo, hi, faces, origin, h, mask)
    # coarse cells 8 fine ones across mark where the query faces are
    H = 8.0 * h
    cmask = np.int64((1 << max(10, int(np.ceil(np.log2(max(2 * n, 1)))))) - 1)
    cm = coarse_marks(lo, hi, faces, origin, H, cmask)
    out = np.zeros(len(F), np.uint8)
    query(V, F, lo, hi, count, items, origin, h, mask, cm, H, cmask, out)
    return np.flatnonzero(out)
