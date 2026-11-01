"""Mesh bookkeeping for the field mesher as compiled kernels.

field_mesh's relax, unfold and debris filter spend most of their time not
in the field but in array bookkeeping over millions of faces: face
normals, vertex normals, the vertex-face incidence and the edge
adjacency the relax smooths over. As numpy each is several passes of
random-access gathers and a sort (an edge list through scipy's coo->csr
is ~2 s on a 3M-face lattice); here each is one parallel pass.

Every function has the numpy result it replaces as its fallback (no
numba, or LOADPATH_MESHOPS=numpy), and gives the same numbers up to the
order floating-point sums are taken in.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import numba as nb
except ImportError:  # pragma: no cover - numba is a dependency; the numpy paths without it
    nb = None

__all__ = ["available", "face_cross", "triple", "vertex_faces", "vertex_sum", "neighbors", "faces_of"]


def available():
    return nb is not None and os.environ.get("LOADPATH_MESHOPS", "").lower() != "numpy"


def _cap_threads():
    # LOADPATH_THREADS caps the compiled kernels' threads as it does peval's
    env = os.environ.get("LOADPATH_THREADS")
    if nb is not None and env:
        try:
            nb.set_num_threads(max(1, min(int(env), nb.config.NUMBA_NUM_THREADS)))
        except Exception:  # noqa: BLE001 - a bad value: numba's default
            pass


_cap_threads()


if nb is not None:

    @nb.njit(cache=True, parallel=True)
    def _face_cross(V, F):
        n = F.shape[0]
        out = np.empty((n, 3))
        for i in nb.prange(n):
            a, b, c = F[i, 0], F[i, 1], F[i, 2]
            b0, b1, b2 = V[b, 0] - V[a, 0], V[b, 1] - V[a, 1], V[b, 2] - V[a, 2]
            c0, c1, c2 = V[c, 0] - V[a, 0], V[c, 1] - V[a, 1], V[c, 2] - V[a, 2]
            out[i, 0] = b1 * c2 - b2 * c1
            out[i, 1] = b2 * c0 - b0 * c2
            out[i, 2] = b0 * c1 - b1 * c0
        return out

    @nb.njit(cache=True, parallel=True)
    def _triple(V, F):
        n = F.shape[0]
        out = np.empty(n)
        for i in nb.prange(n):
            a, b, c = F[i, 0], F[i, 1], F[i, 2]
            out[i] = (V[a, 0] * (V[b, 1] * V[c, 2] - V[b, 2] * V[c, 1])
                      + V[a, 1] * (V[b, 2] * V[c, 0] - V[b, 0] * V[c, 2])
                      + V[a, 2] * (V[b, 0] * V[c, 1] - V[b, 1] * V[c, 0]))
        return out

    @nb.njit(cache=True)
    def _vertex_faces(F, n):
        nf = F.shape[0]
        ptr = np.zeros(n + 1, np.int64)
        for i in range(nf):
            for k in range(3):
                ptr[F[i, k] + 1] += 1
        for v in range(n):
            ptr[v + 1] += ptr[v]
        fill = ptr[:-1].copy()
        out = np.empty(3 * nf, np.int64)
        for i in range(nf):          # faces in ascending order within each vertex
            for k in range(3):
                v = F[i, k]
                out[fill[v]] = i
                fill[v] += 1
        return ptr, out

    @nb.njit(cache=True, parallel=True)
    def _vertex_sum(ptr, fidx, X):
        n = ptr.shape[0] - 1
        m = X.shape[1]
        out = np.zeros((n, m))
        for v in nb.prange(n):
            for j in range(ptr[v], ptr[v + 1]):
                f = fidx[j]
                for k in range(m):
                    out[v, k] += X[f, k]
        return out

    @nb.njit(cache=True, parallel=True)
    def _neighbor_count(F, ptr, fidx, seam, use_seam):
        n = ptr.shape[0] - 1
        cnt = np.zeros(n, np.int64)
        for v in nb.prange(n):
            m = ptr[v + 1] - ptr[v]
            buf = np.empty(2 * m, np.int64)
            k = 0
            for j in range(ptr[v], ptr[v + 1]):
                f = fidx[j]
                for c in range(3):
                    u = F[f, c]
                    if u == v:
                        continue
                    if use_seam and seam[v] and not seam[u]:
                        continue
                    dup = False
                    for t in range(k):
                        if buf[t] == u:
                            dup = True
                            break
                    if not dup:
                        buf[k] = u
                        k += 1
            cnt[v] = k
        return cnt

    @nb.njit(cache=True, parallel=True)
    def _neighbor_fill(F, ptr, fidx, seam, use_seam, indptr, indices):
        n = ptr.shape[0] - 1
        for v in nb.prange(n):
            k = indptr[v]
            k0 = k
            for j in range(ptr[v], ptr[v + 1]):
                f = fidx[j]
                for c in range(3):
                    u = F[f, c]
                    if u == v:
                        continue
                    if use_seam and seam[v] and not seam[u]:
                        continue
                    dup = False
                    for t in range(k0, k):
                        if indices[t] == u:
                            dup = True
                            break
                    if not dup:
                        indices[k] = u
                        k += 1
            # sorted columns, as scipy keeps them
            indices[k0:k].sort()

    @nb.njit(cache=True, parallel=True)
    def _min_cos(ptr, fidx, fn, N):
        n = ptr.shape[0] - 1
        out = np.ones(n)
        for v in nb.prange(n):
            m = np.inf if ptr[v + 1] > ptr[v] else 1.0
            for j in range(ptr[v], ptr[v + 1]):
                f = fidx[j]
                c = fn[f, 0] * N[v, 0] + fn[f, 1] * N[v, 1] + fn[f, 2] * N[v, 2]
                if c < m:
                    m = c
            out[v] = m
        return out

    @nb.njit(cache=True, parallel=True)
    def _row_dots(A, B, ia, ib):
        n = ia.shape[0]
        out = np.empty(n)
        for i in nb.prange(n):
            a, b = ia[i], ib[i]
            out[i] = A[a, 0] * B[b, 0] + A[a, 1] * B[b, 1] + A[a, 2] * B[b, 2]
        return out

    @nb.njit(cache=True, parallel=True)
    def _edge_mates(F, ptr, fidx):
        nf = F.shape[0]
        mate = np.full(3 * nf, -1, np.int64)
        for f in nb.prange(nf):
            for k in range(3):
                a = F[f, k]
                b = F[f, (k + 1) % 3]
                best = -1
                for j in range(ptr[a], ptr[a + 1]):
                    g = fidx[j]
                    if g == f:
                        continue
                    for c in range(3):
                        u = F[g, c]
                        w = F[g, (c + 1) % 3]
                        if (u == a and w == b) or (u == b and w == a):
                            h = 3 * g + c
                            if best < 0 or h < best:
                                best = h
                mate[3 * f + k] = best
        return mate

    @nb.njit(cache=True, parallel=True)
    def _mark_faces(ptr, fidx, verts, out):
        for r in nb.prange(verts.shape[0]):
            v = verts[r]
            for j in range(ptr[v], ptr[v + 1]):
                out[fidx[j]] = True


def _ints(F):
    return np.ascontiguousarray(F, np.int64)


def face_cross(V, F):
    """(b - a) x (c - a) for every face (a, b, c)."""
    if available() and len(F):
        return _face_cross(np.ascontiguousarray(V, np.float64), _ints(F))
    from .narrowband import _face_cross_np

    return _face_cross_np(V, F)


def triple(V, F):
    """a . (b x c) per face."""
    if available() and len(F):
        return _triple(np.ascontiguousarray(V, np.float64), _ints(F))
    from .narrowband import _triple_np

    return _triple_np(V, F)


def vertex_faces(F, n):
    """CSR vertex -> faces: faces ptr[v]:ptr[v + 1] of `idx` hold vertex v
    (each face once per corner it has there; ascending within a vertex)."""
    F = _ints(F)
    if available():
        return _vertex_faces(F, int(n))
    flat = F.ravel()
    o = np.argsort(flat, kind="stable")
    ptr = np.searchsorted(flat[o], np.arange(int(n) + 1))
    return ptr, o // 3


def vertex_sum(ptr, idx, X):
    """Per vertex, the sum of X over its faces (rows of X are faces)."""
    X = np.ascontiguousarray(X, np.float64)
    if available():
        return _vertex_sum(ptr, idx, X.reshape(len(X), -1)).reshape((len(ptr) - 1,) + X.shape[1:])
    v = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    out = np.zeros((len(ptr) - 1,) + X.shape[1:])
    np.add.at(out, v, X[idx])
    return out


def neighbors(F, n, seam=None, vf=None):
    """The mesh's edge adjacency as CSR (indptr, indices): each vertex's
    distinct neighbours along an edge, sorted. With `seam` (bool per
    vertex), a seam vertex keeps only its seam neighbours."""
    F = _ints(F)
    n = int(n)
    if available():
        ptr, idx = vf if vf is not None else _vertex_faces(F, n)
        use = seam is not None
        s = np.ascontiguousarray(seam, np.bool_) if use else np.zeros(1, np.bool_)
        cnt = _neighbor_count(F, ptr, idx, s, use)
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum(cnt, out=indptr[1:])
        indices = np.empty(int(indptr[-1]), np.int64)
        _neighbor_fill(F, ptr, idx, s, use, indptr, indices)
        return indptr, indices
    import scipy.sparse as sp

    a = np.r_[F[:, 0], F[:, 1], F[:, 2]]
    b = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    r, c = np.r_[a, b], np.r_[b, a]
    if seam is not None:
        seam = np.asarray(seam, bool)
        k = ~seam[r] | seam[c]
        r, c = r[k], c[k]
    A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)


def min_cos(vf, fn, N):
    """Per vertex, the least cosine between its unit normal N and the unit
    normals fn of the faces round it (1 where it has none)."""
    ptr, idx = vf
    if available():
        return _min_cos(ptr, idx, np.ascontiguousarray(fn, np.float64), np.ascontiguousarray(N, np.float64))
    cnt = np.diff(ptr)
    c = (fn[idx] * N[np.repeat(np.arange(len(cnt)), cnt)]).sum(axis=1)
    lo = np.ones(len(cnt))
    has = np.flatnonzero(cnt)
    lo[has] = np.minimum.reduceat(c, ptr[has])
    return lo


def row_dots(A, B, ia, ib):
    """(A[ia] * B[ib]).sum(axis=1) for (n, 3) arrays."""
    ia = np.ascontiguousarray(ia, np.int64)
    ib = np.ascontiguousarray(ib, np.int64)
    if available():
        return _row_dots(np.ascontiguousarray(A, np.float64), np.ascontiguousarray(B, np.float64), ia, ib)
    return (A[ia] * B[ib]).sum(axis=1)


def edge_pairs(F, vf=None):
    """The mesh's edges as pairs of half-edges (h1 < h2; half-edge h is
    corner h % 3 of face h // 3, from it to the next corner): the two
    faces that share each edge. An edge with one face is left out, and
    one with more than two gives its first two."""
    F = _ints(F)
    if available():
        ptr, idx = vf if vf is not None else _vertex_faces(F, int(F.max()) + 1 if len(F) else 0)
        mate = _edge_mates(F, ptr, idx)
        h1 = np.flatnonzero(mate > np.arange(len(mate)))
        h2 = mate[h1]
        # a non-manifold edge: keep only its first pair
        return h1, h2
    a, b = F.ravel(), np.roll(F, -1, axis=1).ravel()
    key = np.minimum(a, b) * np.int64(F.max() + 1) + np.maximum(a, b)
    o = np.argsort(key, kind="stable")
    ks = key[o]
    p = np.flatnonzero(ks[1:] == ks[:-1])
    first = np.r_[True, ks[1:] != ks[:-1]]
    p = p[first[p]]                       # the first two of each edge
    h1, h2 = o[p], o[p + 1]
    s = np.argsort(h1)
    return h1[s], h2[s]


def faces_of(vf, verts, nf):
    """Sorted distinct faces round the vertices `verts` (vf: vertex_faces)."""
    ptr, idx = vf
    verts = np.ascontiguousarray(verts, np.int64)
    out = np.zeros(int(nf), np.bool_)
    if available():
        _mark_faces(ptr, idx, verts, out)
    else:
        a, ln = ptr[verts], ptr[verts + 1] - ptr[verts]
        j = np.repeat(a - np.cumsum(ln) + ln, ln) + np.arange(int(ln.sum()))
        out[idx[j]] = True
    return np.flatnonzero(out)
