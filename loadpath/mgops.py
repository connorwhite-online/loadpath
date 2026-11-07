"""Compiled kernels for the matrix-free multigrid (mg.py), on numba's
threads.

mg._Level.apply_k applied K u on the node grid tile by tile with numpy:
gather the 8 corner displacements of a tile's cells (8 slice copies), one
GEMM with the 24x24 cell matrix (eight on the first coarse level), scale,
scatter back. On a constrained trial part (the wrist yoke, 119k cells) that
was 75% of the run, on one core: 720k tile calls of a few hundred cells,
half their time in the copies, and the smoother's vector updates, the grid
transfers and the CG's dot products single-threaded numpy around it.

Here the operator is two parallel passes over the level's live cells:

- product: blocks of BLOCK cells gather their 24 corner values (one
  precomputed offset per corner dof) into a (24, BLOCK) tile and multiply
  by the cell matrix, the loop over the block's cells innermost so it
  vectorizes; the (24, cells) result is kept on `Cells`.
- scatter, node by node: each node sums its (up to 8) cells' rows, so every
  node is written by one thread, once.

Around it, the Chebyshev smoother's and CG's updates are fused passes, the
dot products run here (OpenBLAS's threads, woken by np.vdot, spun beside
numba's and halved the operator's speed), and prolong / restrict are one
gather pass each. Sums are taken in the field's precision (float32 in the
V-cycle, float64 in the CG) as numpy took them, in another order: the
numbers agree to rounding, not to the bit (the yoke's first six optimizer
iterations: the same stress readings to 7 digits, 390 s -> 148 s).

Not thread-safe per level: a level's product buffer is shared, so one
GridElasticity must not solve on two threads at once (none does).

`available()` is False without numba or with LOADPATH_MGOPS=numpy, which
restores the numpy path.
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

__all__ = ["Cells", "apply_explicit", "apply_scaled", "apply_weights", "available", "dot", "prolong", "restrict"]


def available():
    return nb is not None and os.environ.get("LOADPATH_MGOPS", "").lower() != "numpy"


if nb is not None:
    # Indices go through uint64 (as in amgfast.py): numba wraps negative
    # signed indices, a compare and branch on every load that kept these
    # loops from vectorizing (a plain 24x24-by-cells product ran 1.2 GFLOP/s
    # signed, about BLAS's speed unsigned).
    U = np.uint64
    BLOCK = 256        # cells per block in the product pass

    @nb.njit(cache=True, inline="always")
    def _gather_block(uf, base, off, c0, nb_, ubt):
        # ubt[i, c] = u at corner dof i of cell c0 + c: base[c] is the cell's
        # first node in the flattened field, off[i] the dof's offset from it
        for i in range(24):
            ui = U(i)
            o = off[ui]
            for c in range(nb_):
                ubt[ui, U(c)] = uf[U(base[U(c0 + c)] + o)]

    @nb.njit(cache=True, parallel=True, nogil=True, fastmath=True)
    def _product_scaled(uf, base, off, ke, e, ft):
        """ft[:, c] = e[c] * ke @ (corner values of cell c)."""
        n = base.shape[0]
        for blk in nb.prange((n + BLOCK - 1) // BLOCK):
            c0 = blk * BLOCK
            nb_ = min(BLOCK, n - c0)
            ubt = np.empty((24, BLOCK), uf.dtype)
            _gather_block(uf, base, off, c0, nb_, ubt)
            for i in range(24):
                ui = U(i)
                for c in range(nb_):
                    ft[ui, U(c0 + c)] = 0
                for j in range(24):
                    kij = ke[ui, U(j)]
                    for c in range(nb_):
                        ft[ui, U(c0 + c)] += kij * ubt[U(j), U(c)]
                for c in range(nb_):
                    ft[ui, U(c0 + c)] *= e[U(c0 + c)]

    @nb.njit(cache=True, parallel=True, nogil=True, fastmath=True)
    def _product_weights(uf, base, off, mk, w, ft):
        """ft[:, c] = (sum_q w[q, c] M_q) @ (corner values of cell c)."""
        n = base.shape[0]
        for blk in nb.prange((n + BLOCK - 1) // BLOCK):
            c0 = blk * BLOCK
            nb_ = min(BLOCK, n - c0)
            ubt = np.empty((24, BLOCK), uf.dtype)
            t = np.empty((24, BLOCK), uf.dtype)
            _gather_block(uf, base, off, c0, nb_, ubt)
            for i in range(24):
                for c in range(nb_):
                    ft[U(i), U(c0 + c)] = 0
            for q in range(8):
                uq = U(q)
                for j in range(24):
                    for c in range(nb_):
                        t[U(j), U(c)] = w[uq, U(c0 + c)] * ubt[U(j), U(c)]
                for i in range(24):
                    ui = U(i)
                    for j in range(24):
                        m = mk[uq, ui, U(j)]
                        for c in range(nb_):
                            ft[ui, U(c0 + c)] += m * t[U(j), U(c)]

    @nb.njit(cache=True, parallel=True, nogil=True, fastmath=True)
    def _product_explicit(uf, base, off, kc, ft):
        """ft[:, c] = kc[:, :, c] @ (corner values of cell c)."""
        n = base.shape[0]
        for blk in nb.prange((n + BLOCK - 1) // BLOCK):
            c0 = blk * BLOCK
            nb_ = min(BLOCK, n - c0)
            ubt = np.empty((24, BLOCK), uf.dtype)
            _gather_block(uf, base, off, c0, nb_, ubt)
            for i in range(24):
                ui = U(i)
                for c in range(nb_):
                    ft[ui, U(c0 + c)] = 0
                for j in range(24):
                    uj = U(j)
                    for c in range(nb_):
                        ft[ui, U(c0 + c)] += kc[ui, uj, U(c0 + c)] * ubt[uj, U(c)]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _scatter(ft, rows, out):
        """out += ft[:, r] at the corners of each cell r, gathered node by
        node (rows (nx, ny, nz): a cell's column in ft, or -1): each node is
        written by one thread, once."""
        nx, ny, nz = rows.shape
        for xy in nb.prange((nx + 1) * (ny + 1)):
            x = xy // (ny + 1)
            y = xy % (ny + 1)
            for z in range(nz + 1):
                uz = U(z)
                a0 = out[U(0), U(x), U(y), uz]
                a1 = out[U(1), U(x), U(y), uz]
                a2 = out[U(2), U(x), U(y), uz]
                for k in range(8):
                    cx = x - (k & 1)
                    cy = y - ((k >> 1) & 1)
                    cz = z - ((k >> 2) & 1)
                    if cx < 0 or cx >= nx or cy < 0 or cy >= ny or cz < 0 or cz >= nz:
                        continue
                    r = rows[U(cx), U(cy), U(cz)]
                    if r < 0:
                        continue
                    ur = U(r)
                    a0 += ft[U(3 * k), ur]
                    a1 += ft[U(3 * k + 1), ur]
                    a2 += ft[U(3 * k + 2), ur]
                out[U(0), U(x), U(y), uz] = a0
                out[U(1), U(x), U(y), uz] = a1
                out[U(2), U(x), U(y), uz] = a2


class Cells:
    """A level's cells that carry stiffness, x-major; rows[x, y, z] is a
    cell's index in that list, or -1."""

    def __init__(self, active: np.ndarray):
        ijk = np.argwhere(active)
        self.cells = np.ascontiguousarray(ijk, np.int64)
        self.n = len(ijk)
        self.rows = np.full(active.shape, -1, np.int64)
        self.rows[tuple(ijk.T)] = np.arange(self.n)
        nshape = tuple(int(d) + 1 for d in active.shape)
        self.base = np.ascontiguousarray(np.ravel_multi_index(tuple(ijk.T), nshape), np.int64)
        nn = int(np.prod(nshape))
        sy, sz = nshape[1] * nshape[2], nshape[2]
        self.off = np.array([d * nn + (k & 1) * sy + (k >> 1 & 1) * sz + (k >> 2 & 1)
                             for k in range(8) for d in range(3)], np.int64)
        self.buffers = {}

    def ft(self, dtype):
        """The (24, n) product buffer, kept: a fresh 10-25 MB array every
        apply cost page faults."""
        key = np.dtype(dtype).str
        if key not in self.buffers:
            self.buffers[key] = np.empty((24, self.n), dtype)
        return self.buffers[key]


def apply_scaled(u, ke, e_c, cells: Cells, out):
    """out += sum over cells of e_c[cell] * ke @ u[cell corners] (u, out:
    (3, nodes...)); ke and e_c (the cells' moduli) in u's dtype."""
    ft = cells.ft(u.dtype)
    _product_scaled(u.reshape(-1), cells.base, cells.off, ke, e_c, ft)
    _scatter(ft, cells.rows, out)


def apply_weights(u, mk, w_c, cells: Cells, out):
    """out += sum over cells of (sum_q w_c[q, cell] M_q) @ u[cell corners];
    mk (8, 24, 24), w_c (8, n)."""
    ft = cells.ft(u.dtype)
    _product_weights(u.reshape(-1), cells.base, cells.off, mk, w_c, ft)
    _scatter(ft, cells.rows, out)


def apply_explicit(u, k_c, cells: Cells, out):
    """out += sum over cells of K[cell] @ u[cell corners]; k_c (24, 24, n)."""
    ft = cells.ft(u.dtype)
    _product_explicit(u.reshape(-1), cells.base, cells.off, k_c, ft)
    _scatter(ft, cells.rows, out)


# --- fused vector updates (one parallel pass instead of numpy's 2-4) ------

if nb is not None:

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _mul(a, b, out):
        for i in nb.prange(a.size):
            out[i] = a[i] * b[i]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _finish(out, m, u):
        for i in nb.prange(out.size):
            out[i] = out[i] * m[i] + (1 - m[i]) * u[i]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _residual_scaled(b, ax, dinv, s, r, d):
        # r = b - ax; d = s * dinv * r
        for i in nb.prange(b.size):
            ri = b[i] - ax[i]
            r[i] = ri
            d[i] = s * dinv[i] * ri

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _cheb(x, d, r, ad, dinv, c1, c2, last):
        # x += d; then (unless last) r -= ad; d = c1 d + c2 dinv r
        for i in nb.prange(x.size):
            di = d[i]
            x[i] += di
            if not last:
                ri = r[i] - ad[i]
                r[i] = ri
                d[i] = c1 * di + c2 * dinv[i] * ri

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _cg_step(x, r, p, ap, alpha):
        # x += alpha p; r -= alpha ap; returns |r|^2
        acc = 0.0
        for i in nb.prange(x.size):
            x[i] += alpha * p[i]
            ri = r[i] - alpha * ap[i]
            r[i] = ri
            acc += ri * ri
        return acc

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _xpby(z, beta, p):
        # p = z + beta p
        for i in nb.prange(z.size):
            p[i] = z[i] + beta * p[i]


if nb is not None:

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _dot(a, b):
        acc = 0.0
        for i in nb.prange(a.size):
            acc += a[i] * b[i]
        return acc


def dot(a, b) -> float:
    """vdot of two same-shape contiguous fields on numba's threads. np.vdot
    and np.linalg.norm go to OpenBLAS, whose threads then spin beside
    numba's through the next operator pass: with them in the CG loop the
    compiled operator ran 2x slower on the 4-core box."""
    if available() and a.flags.c_contiguous and b.flags.c_contiguous:
        return float(_dot(a.reshape(-1), b.reshape(-1)))
    return float(np.vdot(a, b))


# --- grid transfers (mg.prolong / mg.restrict as one gather pass) --------

if nb is not None:

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _prolong(c, f):
        nd, fx, fy, fz = f.shape
        for xy in nb.prange(fx * fy):
            x = xy // fy
            y = xy % fy
            x0 = x >> 1
            y0 = y >> 1
            x1 = x0 + (x & 1)
            y1 = y0 + (y & 1)
            wx = 0.5 if x & 1 else 1.0
            wy = 0.5 if y & 1 else 1.0
            for d in range(nd):
                ud = U(d)
                for z in range(fz):
                    z0 = z >> 1
                    z1 = z0 + (z & 1)
                    wz = 0.5 if z & 1 else 1.0
                    a = c[ud, U(x0), U(y0), U(z0)]
                    if z & 1:
                        a = a + c[ud, U(x0), U(y0), U(z1)]
                    if y & 1:
                        a = a + c[ud, U(x0), U(y1), U(z0)]
                        if z & 1:
                            a = a + c[ud, U(x0), U(y1), U(z1)]
                    if x & 1:
                        b = c[ud, U(x1), U(y0), U(z0)]
                        if z & 1:
                            b = b + c[ud, U(x1), U(y0), U(z1)]
                        if y & 1:
                            b = b + c[ud, U(x1), U(y1), U(z0)]
                            if z & 1:
                                b = b + c[ud, U(x1), U(y1), U(z1)]
                        a = a + b
                    f[ud, U(x), U(y), U(z)] = a * (wx * wy * wz)

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _restrict(f, c):
        nd, fx, fy, fz = f.shape
        cx_, cy_, cz_ = c.shape[1], c.shape[2], c.shape[3]
        for xy in nb.prange(cx_ * cy_):
            X = xy // cy_
            Y = xy % cy_
            for d in range(nd):
                ud = U(d)
                for Z in range(cz_):
                    acc = f[ud, U(2 * X), U(2 * Y), U(2 * Z)] * 0
                    for a in range(-1, 2):
                        x = 2 * X + a
                        if x < 0 or x >= fx:
                            continue
                        wa = 0.5 if a != 0 else 1.0
                        for b in range(-1, 2):
                            y = 2 * Y + b
                            if y < 0 or y >= fy:
                                continue
                            wb = wa * (0.5 if b != 0 else 1.0)
                            for e in range(-1, 2):
                                z = 2 * Z + e
                                if z < 0 or z >= fz:
                                    continue
                                acc += f[ud, U(x), U(y), U(z)] * (wb * (0.5 if e != 0 else 1.0))
                    c[ud, U(X), U(Y), U(Z)] = acc


def prolong(c):
    """mg.prolong for a (3, nodes...) field, or None to use numpy's."""
    if not (available() and c.ndim == 4 and c.flags.c_contiguous):
        return None
    f = np.empty((c.shape[0],) + tuple(2 * n - 1 for n in c.shape[1:]), c.dtype)
    _prolong(c, f)
    return f


def restrict(f):
    """mg.restrict for a (3, nodes...) field, or None to use numpy's."""
    if not (available() and f.ndim == 4 and f.flags.c_contiguous):
        return None
    c = np.empty((f.shape[0],) + tuple((n + 1) // 2 for n in f.shape[1:]), f.dtype)
    _restrict(f, c)
    return c
