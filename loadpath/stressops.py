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

__all__ = ["available", "ball_pairs"]


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
