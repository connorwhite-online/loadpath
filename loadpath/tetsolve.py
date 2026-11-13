"""Compiled kernels for the tet confirm's quadratic elasticity solve
(crosscheck._solve) and its assembly.

The solve is CG with a two-level preconditioner: symmetric Gauss-Seidel on
the quadratic (P2) system and an exact solve of the linear (vertex, P1)
system. It used to run one CG per load case on threads, each case's
Gauss-Seidel on one core and its coarse level one smoothed-aggregation
V-cycle. On the robot-arm upper arm (2.25M dofs, three cases) that took
65-71 iterations and 250 s; the V-cycle was the weak part: with the P1
level factored (PARDISO, 1.3 GB) the same cases take 19-21 iterations.
Here the cases run in step (one CG per case, each with its own step
lengths and stopping test, so the same answers as one at a time), so one
factor solve serves all of them, and the Gauss-Seidel runs on every core:

- Nodes are numbered by colour (`color_order`): a greedy colouring in
  reverse Cuthill-McKee order of the node graph, no two nodes of one
  element alike (about 20 colours). Nodes of one colour share no matrix
  entry, so a colour is relaxed all at once, every load case together
  (`sgs`). Colours ascending then descending is a symmetric sweep, so the
  preconditioner is symmetric and CG applies.
- `spmm` is the matrix times all cases at once (the matrix is read once
  for all of them).

Every kernel's arithmetic is independent of the number of threads, so a
solve repeats to the bit. Indices go through uint64, as in amgfast.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb

    from . import meshops  # noqa: F401 - caps numba's threads at LOADPATH_THREADS
except ImportError:  # pragma: no cover - numba is a dependency
    nb = None

U = np.uint64


def _u(a):
    """An index array viewed unsigned (they are never negative)."""
    a = np.ascontiguousarray(a)
    return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint64)


if nb is not None:

    @nb.njit(cache=True, nogil=True)
    def _greedy_colors(nptr, nind, order, width):
        n = nptr.size - 1
        color = np.full(n, -1, np.int64)
        stamp = np.full(width, -1, np.int64)
        ncol = 0
        for t in range(n):
            i = order[t]
            for jj in range(nptr[i], nptr[i + 1]):
                c = color[nind[jj]]
                if c >= 0:
                    stamp[c] = t
            c = 0
            while stamp[c] == t:
                c += 1
            color[i] = c
            ncol = max(ncol, c + 1)
        return color, ncol

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _sgs(Ap, Aj, Ax, X, B, bounds):
        ncol = bounds.size - 1
        k = X.shape[1]
        for s in range(2 * ncol):
            back = s >= ncol
            c = 2 * ncol - 1 - s if back else s
            lo = bounds[c]
            hi = bounds[c + 1]
            nch = 64
            step = (hi - lo + nch - 1) // nch
            for t in nb.prange(nch):
                acc = np.empty(k)
                for v in range(lo + t * step, min(hi, lo + (t + 1) * step)):
                    for d in range(3):
                        dd = 2 - d if back else d
                        ui = U(3 * v + dd)
                        for r in range(k):
                            acc[r] = B[ui, r]
                        diag = 0.0
                        jj = U(Ap[ui])
                        end = U(Ap[ui + U(1)])
                        while jj < end:
                            j = U(Aj[jj])
                            a = Ax[jj]
                            if j == ui:
                                diag = a
                            else:
                                for r in range(k):
                                    acc[r] -= a * X[j, r]
                            jj += U(1)
                        if diag != 0.0:
                            for r in range(k):
                                X[ui, r] = acc[r] / diag

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _spmm(Ap, Aj, Ax, X, Y):
        n = Y.shape[0]
        k = Y.shape[1]
        nch = 256
        step = (n + nch - 1) // nch
        for c in nb.prange(nch):
            acc = np.empty(k)
            for i in range(c * step, min(n, (c + 1) * step)):
                ui = U(i)
                for r in range(k):
                    acc[r] = 0.0
                jj = U(Ap[ui])
                end = U(Ap[ui + U(1)])
                while jj < end:
                    a = Ax[jj]
                    j = U(Aj[jj])
                    for r in range(k):
                        acc[r] += a * X[j, r]
                    jj += U(1)
                for r in range(k):
                    Y[ui, r] = acc[r]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _scatter(cc, ke, nptr, nind, data, nparts):
        ne, m = cc.shape
        n = nptr.size - 1
        for t in nb.prange(nparts):
            r0 = t * n // nparts
            r1 = (t + 1) * n // nparts
            for e in range(ne):
                for a in range(m):
                    i = cc[e, a]
                    if i < r0 or i >= r1:
                        continue
                    lo, hi = nptr[i], nptr[i + 1]
                    deg = hi - lo
                    for b in range(m):
                        j = cc[e, b]
                        l, h = lo, hi
                        while l < h:            # nind[lo:hi] is sorted
                            mid = (l + h) >> 1
                            if nind[mid] < j:
                                l = mid + 1
                            else:
                                h = mid
                        off = 3 * (l - lo)
                        for ci in range(3):
                            base = 9 * lo + 3 * deg * ci + off
                            for cj in range(3):
                                data[base + cj] += ke[e, 3 * a + ci, 3 * b + cj]


def available() -> bool:
    return nb is not None


def color_order(nptr, nind):
    """(q, bounds): new node i is old node q[i]; colour c is new nodes
    bounds[c]:bounds[c + 1]. Greedy in reverse Cuthill-McKee order of the
    node graph (nptr, nind), so each colour's nodes keep some locality."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    n = len(nptr) - 1
    nptr = np.asarray(nptr, np.int64)
    nind = np.asarray(nind, np.int64)
    g = sp.csr_matrix((np.ones(len(nind), np.int8), nind, nptr), shape=(n, n))
    rcm = reverse_cuthill_mckee(g, symmetric_mode=True).astype(np.int64)
    del g
    width = int(np.diff(nptr).max(initial=0)) + 2
    color, ncol = _greedy_colors(nptr, nind, rcm, width)
    rank = np.empty(n, np.int64)
    rank[rcm] = np.arange(n)
    q = np.lexsort((rank, color))
    bounds = np.searchsorted(color[q], np.arange(ncol + 1)).astype(np.int64)
    return q, bounds


def scatter(cc, ke, nptr, nind, data):
    """Add element matrices `ke` (ne, 3m, 3m) on nodes `cc` (ne, m) into the
    node-blocked CSR `data` (crosscheck._scatter's layout), rows split among
    threads: each part adds the rows it owns, elements in order, so every
    entry is summed in the serial order (the same bits on any thread
    count)."""
    _scatter(np.ascontiguousarray(cc, np.int64), np.ascontiguousarray(ke), nptr, nind, data, 64)


class Blocked:
    """A node-blocked CSR matrix (3 rows per node, colours `bounds`) as
    the solve's operators: `mv(X)` = K X and `sgs(X, B)` one symmetric
    multicolour Gauss-Seidel sweep on K X = B, for X, B (dofs, cases)."""

    def __init__(self, k, bounds):
        self.k = k
        self.bounds = np.asarray(bounds, np.int64)
        self._p, self._j = _u(k.indptr), _u(k.indices)

    def mv(self, X):
        Y = np.empty_like(X)
        _spmm(self._p, self._j, self.k.data, X, Y)
        return Y

    def sgs(self, X, B):
        _sgs(self._p, self._j, self.k.data, X, B, self.bounds)


def cg(B, precond, mv, rtol=1e-7, maxiter=2000):
    """One preconditioned CG per column of B, in step: each its own step
    lengths, stopping at ||r|| <= rtol ||b|| as scipy's cg does, so each
    column's answer is what a CG of its own would give. Returns (X, the
    iterations each took)."""
    _n, k = B.shape
    X = np.zeros_like(B)
    R = B.copy()
    tol = rtol * np.linalg.norm(B, axis=0)
    active = np.linalg.norm(R, axis=0) > tol
    its = np.zeros(k, np.int64)
    if not active.any():
        return X, its
    Z = precond(R)
    P = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    for _ in range(maxiter):
        AP = mv(P)
        pap = np.einsum("ij,ij->j", P, AP)
        alpha = np.where(active, rz / np.where(pap != 0.0, pap, 1.0), 0.0)
        X += P * alpha
        R -= AP * alpha
        its += active
        active = active & (np.linalg.norm(R, axis=0) > tol)
        if not active.any():
            break
        Z = precond(R)
        rz2 = np.einsum("ij,ij->j", R, Z)
        beta = np.where(active, rz2 / np.where(rz != 0.0, rz, 1.0), 0.0)
        P = np.where(active, Z + P * beta, P)
        rz = np.where(active, rz2, rz)
    return X, its
