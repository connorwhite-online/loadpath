"""Compiled kernels for reading stress off a solved grid (verify.py's
material-length average, cutcell.CutGrid's surface fit), on numba's
threads.

verify._MaterialAverage pairs every node with the material samples within
the material length of it. scipy's cKDTree.sparse_distance_matrix did that
on one core (1.1 s of a 2 s build on the arm link's 155k nodes and 1.1M
samples, and again in every hotspot submodel). Here the samples are binned
on a grid one radius wide and each node scans its 27 bins, nodes on all
cores. A pair is kept when its squared distance, summed x, y, z in that
order (as cKDTree sums it), is at most radius squared: the same pairs, so
the operator built from them, and every stress read through it, is the
same to the bit.

`available()` is False without numba or with LOADPATH_STRESSOPS=numpy,
which restores the cKDTree path.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import numba as nb

    from . import (
        meshops as _meshops,  # noqa: F401 - caps numba's threads (LOADPATH_THREADS)
    )
except ImportError:  # pragma: no cover - numba is a dependency
    nb = None

__all__ = ["available", "ball_pairs", "corner_sum", "element_fields", "fit_apply", "fit_weights", "point_stress"]


def available():
    return nb is not None and os.environ.get("LOADPATH_STRESSOPS", "").lower() != "numpy"


if nb is not None:

    @nb.njit(cache=True, nogil=True)
    def _bin_of(p, lo, h, dims):
        i = min(max(int(np.floor((p[0] - lo[0]) / h)), 0), dims[0] - 1)
        j = min(max(int(np.floor((p[1] - lo[1]) / h)), 0), dims[1] - 1)
        k = min(max(int(np.floor((p[2] - lo[2]) / h)), 0), dims[2] - 1)
        return i, j, k

    @nb.njit(cache=True, nogil=True)
    def _row(x, xs, start, order, lo, h, dims, r2, out, fill):
        # samples within the radius of x, by the 27 bins round x's own;
        # writes them to out (when fill) and returns how many
        i, j, k = _bin_of(x, lo, h, dims)
        cnt = 0
        for a in range(max(i - 1, 0), min(i + 2, dims[0])):
            for b in range(max(j - 1, 0), min(j + 2, dims[1])):
                for c in range(max(k - 1, 0), min(k + 2, dims[2])):
                    bid = (a * dims[1] + b) * dims[2] + c
                    for t in range(start[bid], start[bid + 1]):
                        s = order[t]
                        d = x[0] - xs[s, 0]
                        acc = 0.0
                        acc += d * d
                        d = x[1] - xs[s, 1]
                        acc += d * d
                        d = x[2] - xs[s, 2]
                        acc += d * d
                        if acc <= r2:
                            if fill:
                                out[cnt] = s
                            cnt += 1
        return cnt

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _count(nodes, xs, start, order, lo, h, dims, r2, cnt):
        dummy = np.empty(0, np.int64)
        for n in nb.prange(nodes.shape[0]):
            cnt[n] = _row(nodes[n], xs, start, order, lo, h, dims, r2, dummy, False)

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _fill(nodes, xs, start, order, lo, h, dims, r2, indptr, indices):
        for n in nb.prange(nodes.shape[0]):
            a, b = indptr[n], indptr[n + 1]
            row = indices[a:b]
            _row(nodes[n], xs, start, order, lo, h, dims, r2, row, True)
            row.sort()


def ball_pairs(nodes, xs, radius):
    """For each of `nodes` (n, 3), the samples `xs` (m, 3) within `radius`
    of it: CSR (indptr (n + 1,), indices sorted within each row), the pairs
    cKDTree(nodes).sparse_distance_matrix(cKDTree(xs), radius) finds.
    None where the kernels are not available."""
    if not available():
        return None
    nodes = np.ascontiguousarray(nodes, np.float64)
    xs = np.ascontiguousarray(xs, np.float64)
    n = len(nodes)
    if n == 0 or len(xs) == 0 or not radius > 0:
        return np.zeros(n + 1, np.int64), np.zeros(0, np.int64)
    r = float(radius)
    # bins a little over a radius wide: a pair in bins two apart is farther
    # than the radius whatever the rounding of the bin arithmetic
    h = r * (1.0 + 1e-9)
    lo = np.minimum(nodes.min(axis=0), xs.min(axis=0))
    hi = np.maximum(nodes.max(axis=0), xs.max(axis=0))
    dims = np.maximum(np.floor((hi - lo) / h).astype(np.int64) + 1, 1)
    if int(np.prod(dims)) > 8 * (len(xs) + n) + 1_000_000:
        return None             # points too sparse in their box for a dense bin grid
    ijk = np.clip(np.floor((xs - lo) / h).astype(np.int64), 0, dims - 1)
    bid = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(bid, kind="stable")
    start = np.zeros(int(np.prod(dims)) + 1, np.int64)
    np.cumsum(np.bincount(bid, minlength=int(np.prod(dims))), out=start[1:])
    r2 = r * r
    cnt = np.empty(n, np.int64)
    _count(nodes, xs, start, order, lo, h, dims, r2, cnt)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(cnt, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), np.int64)
    _fill(nodes, xs, start, order, lo, h, dims, r2, indptr, indices)
    return indptr, indices


# --- cut-cell stress (cutcell.CutGrid) ----------------------------------------
#
# CutGrid read its sample points' and its nodes' stress group by group
# with numpy (one small GEMM per Gauss point or sub-cell centre, a gather
# of every element's 24 values each time) and fitted each surface node's
# 4x4 least squares as batched einsums. Here each is one parallel pass:
# the same arithmetic, the sums taken in their written order (numpy's
# BLAS takes them in its own), so the numbers agree to rounding (1e-13 of
# the stress on the arm link, the surface fit's 4x4 solves widening it),
# far under the run-to-run spread of the multigrid solve that makes the
# displacements (1e-9).

if nb is not None:

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _element_fields(u, edof, els, pos, r_full, r_cut, mbar_cut, ue, alpha, off):
        for t in nb.prange(els.shape[0]):
            e = els[t]
            for a in range(24):
                ue[t, a] = u[edof[e, a]]
            q = pos[e]
            for b in range(9):
                acc = 0.0
                if q >= 0:
                    for a in range(24):
                        acc += r_cut[q, b, a] * ue[t, a]
                else:
                    for a in range(24):
                        acc += r_full[b, a] * ue[t, a]
                alpha[t, b] = acc
            for i in range(6):
                acc = 0.0
                if q >= 0:
                    for b in range(9):
                        acc += mbar_cut[q, i, b] * alpha[t, b]
                off[t, i] = acc

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _point_stress(ue, alpha, off, rows, pts, mod, th, bu, ba, c, pitch, out):
        # out[s] = mod[s] * c @ ((bu[p] @ ue[r] + ba[p] @ alpha[r] - off[r]) / pitch - th[s]),
        # r = rows[s], p = pts[s]; th has length 0 for no thermal strain
        hot = th.shape[0] > 0
        for s in nb.prange(rows.shape[0]):
            r, p = rows[s], pts[s]
            eps = np.empty(6)
            for i in range(6):
                acc = 0.0
                for a in range(24):
                    acc += bu[p, i, a] * ue[r, a]
                acc2 = 0.0
                for b in range(9):
                    acc2 += ba[p, i, b] * alpha[r, b]
                eps[i] = ((acc + acc2) - off[r, i]) / pitch
            if hot:
                for i in range(3):
                    eps[i] -= th[s]
            for i in range(6):
                acc = 0.0
                for j in range(6):
                    acc += eps[j] * c[i, j]
                out[s, i] = acc * mod[s]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _corner_sum(ue, alpha, off, mod, th, bu, ba, c, pitch, nek, acc):
        # each node's sum over the elements it is a corner of (nek[n, k]: the
        # element whose corner k it is, -1 none), corner by corner in k order
        hot = th.shape[0] > 0
        for n in nb.prange(nek.shape[0]):
            eps = np.empty(6)
            for j in range(6):
                acc[n, j] = 0.0
            for k in range(8):
                e = nek[n, k]
                if e < 0:
                    continue
                for i in range(6):
                    s1 = 0.0
                    for a in range(24):
                        s1 += bu[k, i, a] * ue[e, a]
                    s2 = 0.0
                    for b in range(9):
                        s2 += ba[k, i, b] * alpha[e, b]
                    eps[i] = ((s1 + s2) - off[e, i]) / pitch
                if hot:
                    for i in range(3):
                        eps[i] -= th[e]
                for i in range(6):
                    s = 0.0
                    for j in range(6):
                        s += eps[j] * c[i, j]
                    acc[n, i] += s * mod[e]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _fit_apply(wts, idx, sig, cand, out):
        for q in nb.prange(cand.shape[0]):
            for i in range(6):
                acc = 0.0
                for k in range(idx.shape[1]):
                    acc += wts[q, k] * sig[idx[q, k], i]
                out[cand[q], i] = acc

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _fit_weights(p, xyz, w, dist, idx, h, taper, out):
        # the weighted linear least-squares fit's value at p[q] as weights on
        # its neighbours' values: solve (A^T W A + 1e-9 I) g = e0, out = W A g
        nk = idx.shape[1]
        for q in nb.prange(p.shape[0]):
            aw = np.empty((nk, 4))
            a = np.empty((nk, 4))
            for k in range(nk):
                s = idx[q, k]
                t = dist[q, k] / (taper * h)
                ww = w[s] * np.exp(-(t * t))
                a[k, 0] = 1.0
                for d in range(3):
                    a[k, d + 1] = (xyz[s, d] - p[q, d]) / h
                for i in range(4):
                    aw[k, i] = a[k, i] * ww
            m = np.zeros((4, 5))
            for i in range(4):
                for j in range(4):
                    acc = 0.0
                    for k in range(nk):
                        acc += aw[k, i] * a[k, j]
                    m[i, j] = acc
                m[i, i] += 1e-9
            m[0, 4] = 1.0
            # Gaussian elimination, partial pivoting (as LAPACK's getrf)
            for col in range(4):
                piv = col
                big = abs(m[col, col])
                for r in range(col + 1, 4):
                    if abs(m[r, col]) > big:
                        big = abs(m[r, col])
                        piv = r
                if piv != col:
                    for j in range(5):
                        tmp = m[col, j]
                        m[col, j] = m[piv, j]
                        m[piv, j] = tmp
                inv = 1.0 / m[col, col]
                for r in range(col + 1, 4):
                    f = m[r, col] * inv
                    for j in range(col, 5):
                        m[r, j] -= f * m[col, j]
            g = np.empty(4)
            for i in range(3, -1, -1):
                acc = m[i, 4]
                for j in range(i + 1, 4):
                    acc -= m[i, j] * g[j]
                g[i] = acc / m[i, i]
            for k in range(nk):
                acc = 0.0
                for i in range(4):
                    acc += g[i] * aw[k, i]
                out[q, k] = acc


def element_fields(u, edof, els, pos, r_full, r_cut, mbar_cut):
    """Per element els[t]: its 24 nodal values of u (edof), its 9 bubble
    amplitudes (r_cut[pos[e]] @ ue on a cut cell, pos[e] >= 0, else
    r_full @ ue) and the mean bubble strain taken off (mbar_cut[pos[e]] @
    alpha, 0 on a full cell): (ue, alpha, off)."""
    n = len(els)
    ue, alpha, off = np.empty((n, 24)), np.empty((n, 9)), np.empty((n, 6))
    if n:
        r_cut = np.ascontiguousarray(r_cut, np.float64).reshape(-1, 9, 24)
        mbar_cut = np.ascontiguousarray(mbar_cut, np.float64).reshape(-1, 6, 9)
        _element_fields(np.ascontiguousarray(u, np.float64), np.ascontiguousarray(edof, np.int64),
                        np.ascontiguousarray(els, np.int64), np.ascontiguousarray(pos, np.int64),
                        np.ascontiguousarray(r_full, np.float64), r_cut, mbar_cut, ue, alpha, off)
    return ue, alpha, off


def point_stress(ue, alpha, off, rows, pts, mod, th, bu, ba, c, pitch):
    """Stress (s, 6) at sample s: element row rows[s] of ue (24 values),
    alpha (9 bubble amplitudes), off (6, the mean bubble strain taken off),
    at the point whose strain rows are bu[pts[s]] (6, 24), ba[pts[s]]
    (6, 9); modulus mod[s], free expansion th[s] (or None)."""
    out = np.empty((len(rows), 6))
    if len(rows):
        th = np.zeros(0) if th is None else np.ascontiguousarray(th, np.float64)
        _point_stress(np.ascontiguousarray(ue, np.float64), np.ascontiguousarray(alpha, np.float64),
                      np.ascontiguousarray(off, np.float64), np.ascontiguousarray(rows, np.int64),
                      np.ascontiguousarray(pts, np.int64), np.ascontiguousarray(mod, np.float64), th,
                      np.ascontiguousarray(bu, np.float64), np.ascontiguousarray(ba, np.float64),
                      np.ascontiguousarray(c, np.float64), float(pitch), out)
    return out


def corner_sum(ue, alpha, off, mod, th, bu, ba, c, pitch, nek):
    """Per node, the sum over its elements of mod[e] * c @ strain at that
    corner (the strain as point_stress forms it, bu[k] / ba[k] at corner
    k): (nodes, 6)."""
    acc = np.empty((len(nek), 6))
    th = np.zeros(0) if th is None else np.ascontiguousarray(th, np.float64)
    _corner_sum(np.ascontiguousarray(ue, np.float64), np.ascontiguousarray(alpha, np.float64),
                np.ascontiguousarray(off, np.float64), np.ascontiguousarray(mod, np.float64), th,
                np.ascontiguousarray(bu, np.float64), np.ascontiguousarray(ba, np.float64),
                np.ascontiguousarray(c, np.float64), float(pitch), np.ascontiguousarray(nek, np.int64), acc)
    return acc


def fit_apply(wts, idx, sig, cand, out):
    """out[cand[q]] = sum_k wts[q, k] * sig[idx[q, k]]."""
    if len(cand):
        _fit_apply(np.ascontiguousarray(wts, np.float64), np.ascontiguousarray(idx, np.int64),
                   np.ascontiguousarray(sig, np.float64), np.ascontiguousarray(cand, np.int64), out)


def fit_weights(p, xyz, w, dist, idx, h, taper):
    """cutcell._fit_weights in one pass: (points, k)."""
    out = np.empty(idx.shape)
    if len(p):
        _fit_weights(np.ascontiguousarray(p, np.float64), np.ascontiguousarray(xyz, np.float64),
                     np.ascontiguousarray(w, np.float64), np.ascontiguousarray(dist, np.float64),
                     np.ascontiguousarray(idx, np.int64), float(h), float(taper), out)
    return out
