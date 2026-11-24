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

__all__ = ["Cells", "add_csr", "apply_explicit", "apply_masked", "apply_scaled", "apply_weights", "available", "csr_rows", "dot", "prolong", "restrict"]


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
    def _gather_block(uf, mf, base, off, c0, nb_, ubt):
        # ubt[i, c] = u at corner dof i of cell c0 + c: base[c] is the cell's
        # first node in the flattened field, off[i] the dof's offset from it.
        # With a mask mf (size > 1) it is u * mf there: the masked operator's
        # u * mask, read in the gather instead of a pass of its own.
        if mf.size > 1:
            for i in range(24):
                ui = U(i)
                o = off[ui]
                for c in range(nb_):
                    k = U(base[U(c0 + c)] + o)
                    ubt[ui, U(c)] = uf[k] * mf[k]
        else:
            for i in range(24):
                ui = U(i)
                o = off[ui]
                for c in range(nb_):
                    ubt[ui, U(c)] = uf[U(base[U(c0 + c)] + o)]

    @nb.njit(cache=True, parallel=True, nogil=True, fastmath=True)
    def _product_scaled(uf, mf, base, off, ke, e, ft):
        """ft[:, c] = e[c] * ke @ (corner values of cell c)."""
        n = base.shape[0]
        for blk in nb.prange((n + BLOCK - 1) // BLOCK):
            c0 = blk * BLOCK
            nb_ = min(BLOCK, n - c0)
            ubt = np.empty((24, BLOCK), uf.dtype)
            _gather_block(uf, mf, base, off, c0, nb_, ubt)
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
    def _product_weights(uf, mf, base, off, mk, w, ft):
        """ft[:, c] = (sum_q w[q, c] M_q) @ (corner values of cell c)."""
        n = base.shape[0]
        for blk in nb.prange((n + BLOCK - 1) // BLOCK):
            c0 = blk * BLOCK
            nb_ = min(BLOCK, n - c0)
            ubt = np.empty((24, BLOCK), uf.dtype)
            t = np.empty((24, BLOCK), uf.dtype)
            _gather_block(uf, mf, base, off, c0, nb_, ubt)
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
    def _product_explicit(uf, mf, base, off, kc, ft):
        """ft[:, c] = kc[:, :, c] @ (corner values of cell c)."""
        n = base.shape[0]
        for blk in nb.prange((n + BLOCK - 1) // BLOCK):
            c0 = blk * BLOCK
            nb_ = min(BLOCK, n - c0)
            ubt = np.empty((24, BLOCK), uf.dtype)
            _gather_block(uf, mf, base, off, c0, nb_, ubt)
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


    @nb.njit(cache=True, parallel=True, nogil=True)
    def _scatter_into(ft, rows, m, u, out, finish):
        """_scatter into an unset `out` (each node's sum starts at zero, as
        _scatter's did on a zeroed out); with `finish`, the masked
        operator's last step too: out = sum * m + (1 - m) * u."""
        nx, ny, nz = rows.shape
        for xy in nb.prange((nx + 1) * (ny + 1)):
            x = xy // (ny + 1)
            y = xy % (ny + 1)
            zero = np.zeros(1, out.dtype)[0]
            for z in range(nz + 1):
                uz = U(z)
                a0 = zero
                a1 = zero
                a2 = zero
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
                if finish:
                    m0 = m[U(0), U(x), U(y), uz]
                    m1 = m[U(1), U(x), U(y), uz]
                    m2 = m[U(2), U(x), U(y), uz]
                    a0 = a0 * m0 + (1 - m0) * u[U(0), U(x), U(y), uz]
                    a1 = a1 * m1 + (1 - m1) * u[U(1), U(x), U(y), uz]
                    a2 = a2 * m2 + (1 - m2) * u[U(2), U(x), U(y), uz]
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


def apply_masked(kind, u, a, b, cells: Cells, m, out, finish):
    """K (u * m) into an unset `out` (u, m, out: contiguous (3, nodes...) of
    one dtype), in one product and one scatter pass, the mask read in the
    gather; with `finish` also out = out * m + (1 - m) * u (mg._Level.apply
    when the level has no patch or extra terms). The same operations on
    each value as the separate passes, so the same numbers."""
    ft = cells.ft(u.dtype)
    uf, mf = u.reshape(-1), m.reshape(-1)
    if kind == "scaled":
        _product_scaled(uf, mf, cells.base, cells.off, a, b, ft)
    elif kind == "weights":
        _product_weights(uf, mf, cells.base, cells.off, a, b, ft)
    else:
        _product_explicit(uf, mf, cells.base, cells.off, a, ft)
    _scatter_into(ft, cells.rows, m, u, out, finish)


_NOMASK = {}


def _nomask(dtype):
    key = np.dtype(dtype).str
    if key not in _NOMASK:
        _NOMASK[key] = np.zeros(1, dtype)
    return _NOMASK[key]


def apply_scaled(u, ke, e_c, cells: Cells, out):
    """out += sum over cells of e_c[cell] * ke @ u[cell corners] (u, out:
    (3, nodes...)); ke and e_c (the cells' moduli) in u's dtype."""
    ft = cells.ft(u.dtype)
    _product_scaled(u.reshape(-1), _nomask(u.dtype), cells.base, cells.off, ke, e_c, ft)
    _scatter(ft, cells.rows, out)


def apply_weights(u, mk, w_c, cells: Cells, out):
    """out += sum over cells of (sum_q w_c[q, cell] M_q) @ u[cell corners];
    mk (8, 24, 24), w_c (8, n)."""
    ft = cells.ft(u.dtype)
    _product_weights(u.reshape(-1), _nomask(u.dtype), cells.base, cells.off, mk, w_c, ft)
    _scatter(ft, cells.rows, out)


def apply_explicit(u, k_c, cells: Cells, out):
    """out += sum over cells of K[cell] @ u[cell corners]; k_c (24, 24, n)."""
    ft = cells.ft(u.dtype)
    _product_explicit(u.reshape(-1), _nomask(u.dtype), cells.base, cells.off, k_c, ft)
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
    def _cheb_start(b, dinv, s, x, r, d):
        # the smoother's start from zero: x = 0; r = b; d = (dinv b) s, with
        # s in the field's precision (numpy's d *= s)
        for i in nb.prange(b.size):
            bi = b[i]
            x[i] = 0
            r[i] = bi
            d[i] = (dinv[i] * bi) * s

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _cast_into(a, out):
        for i in nb.prange(a.size):
            out[i] = a[i]

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

    # out += A u for a CSR matrix with entries in few of its rows (a grid
    # level's extra terms), a row a thread, each row summed in its stored
    # order in float64 and added in the field's precision: the numbers of
    # (A @ u.astype(float64)).astype(u.dtype), without scipy's pass over the
    # empty rows, its one thread and the three temporaries.
    @nb.njit(cache=True, parallel=True, nogil=True)
    def _add_csr32(rows, indptr, indices, data, u, out):
        for r in nb.prange(rows.size):
            i = rows[r]
            acc = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                acc += data[j] * np.float64(u[indices[j]])
            out[i] += np.float32(acc)

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _add_csr64(rows, indptr, indices, data, u, out):
        for r in nb.prange(rows.size):
            i = rows[r]
            acc = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                acc += data[j] * u[indices[j]]
            out[i] += acc


if nb is not None:

    # The same products by node blocks: a level's extra terms couple whole
    # nodes (a cut cell's 24 x 24, its Galerkin images below), so the three
    # rows of node a hold the same neighbour nodes in each of the three
    # component segments. One index per neighbour serves its nine entries
    # and each u value is read once for the three rows; every row still sums
    # its entries in its stored order (component segment, then neighbour),
    # in float64, so the numbers are _add_csr's to the bit.
    @nb.njit(cache=True, parallel=True, nogil=True)
    def _add_blocks(nodes, ptr, nbr, data, nn, u, out):
        for r in nb.prange(nodes.size):
            a = nodes[r]
            a0, a1, a2 = 0.0, 0.0, 0.0
            for cj in range(3):
                off = cj * nn
                for t in range(ptr[r], ptr[r + 1]):
                    v = np.float64(u[off + nbr[t]])
                    a0 += data[cj, t, 0] * v
                    a1 += data[cj, t, 1] * v
                    a2 += data[cj, t, 2] * v
            out[a] += a0
            out[nn + a] += a1
            out[2 * nn + a] += a2

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _add_blocks32(nodes, ptr, nbr, data, nn, u, out):
        for r in nb.prange(nodes.size):
            a = nodes[r]
            a0, a1, a2 = 0.0, 0.0, 0.0
            for cj in range(3):
                off = cj * nn
                for t in range(ptr[r], ptr[r + 1]):
                    v = np.float64(u[off + nbr[t]])
                    a0 += data[cj, t, 0] * v
                    a1 += data[cj, t, 1] * v
                    a2 += data[cj, t, 2] * v
            out[a] += np.float32(a0)
            out[nn + a] += np.float32(a1)
            out[2 * nn + a] += np.float32(a2)


def _padded_blocks(a):
    """`a` with explicit zeros filling out every node block any entry
    touches: in the sorted order a zero adds nothing, so every row sums the
    same numbers in the same order."""
    import scipy.sparse as sp

    n = a.shape[0]
    nn = n // 3
    c = a.tocoo()
    pr, pc = c.row % nn, c.col % nn
    g = sp.csr_matrix((np.ones(len(pr), np.int8), (pr, pc)), shape=(nn, nn))
    g.sum_duplicates()
    g = g.tocoo()
    rows = np.concatenate([g.row + ci * nn for ci in range(3) for _ in range(3)])
    cols = np.concatenate([g.col + cj * nn for _ in range(3) for cj in range(3)])
    key = np.unique(rows.astype(np.int64) * n + cols)            # sorted: by row, then column
    data = np.zeros(len(key))
    data[np.searchsorted(key, c.row.astype(np.int64) * n + c.col)] = c.data
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(key // n, minlength=n), out=indptr[1:])
    out = sp.csr_matrix((data, (key % n).astype(np.int64), indptr), shape=a.shape)
    out.has_sorted_indices = True
    return out


def _node_blocks(a):
    """(nodes, ptr, neighbour nodes, data (3, pairs, 3) [cj, pair, ci]) of a
    square CSR on the (3, nodes) layout when every node's three rows hold
    whole node blocks (above), else None."""
    n = a.shape[0]
    if n % 3 or a.shape[1] != n or a.nnz == 0:
        return None
    nn = n // 3
    ip = np.asarray(a.indptr, np.int64)
    ln = np.diff(ip)
    l3 = ln.reshape(3, nn)
    if not (np.all(l3 == l3[0]) and np.all(l3[0] % 3 == 0)):
        return None
    nodes = np.flatnonzero(l3[0])
    m = l3[0][nodes] // 3                              # neighbours per node
    ptr = np.zeros(len(nodes) + 1, np.int64)
    np.cumsum(m, out=ptr[1:])
    pairs = int(ptr[-1])
    # entry positions: row ci * nn + a, segment cj, neighbour t
    rel = np.arange(pairs) - np.repeat(ptr[:-1], m)
    start = ip[:-1].reshape(3, nn)[:, nodes]           # (ci, node)
    mm = np.repeat(m, m)
    nbr = None
    data = np.empty((3, pairs, 3))
    idx = np.asarray(a.indices, np.int64)
    for ci in range(3):
        base = np.repeat(start[ci], m) + rel
        for cj in range(3):
            pos = base + cj * mm
            col = idx[pos] - cj * nn
            if nbr is None:
                nbr = col
                if col.min() < 0 or col.max() >= nn:
                    return None
            elif not np.array_equal(col, nbr):
                return None
            data[cj, :, ci] = a.data[pos]
    return nodes, ptr, nbr.astype(np.int32 if nn < 2 ** 31 else np.int64), data, nn


def add_csr(prepared, u, out) -> bool:
    """out += A u on numba's threads (u, out: same-shape contiguous float32
    or float64 fields; prepared: csr_rows(A)). False when it cannot (the
    caller then does it with scipy)."""
    if not (available() and u.flags.c_contiguous and out.flags.c_contiguous and u.dtype == out.dtype):
        return False
    if len(prepared) == 5:
        if u.dtype == np.float32:
            _add_blocks32(*prepared, u.reshape(-1), out.reshape(-1))
        elif u.dtype == np.float64:
            _add_blocks(*prepared, u.reshape(-1), out.reshape(-1))
        else:
            return False
        return True
    if u.dtype == np.float32:
        _add_csr32(*prepared, u.reshape(-1), out.reshape(-1))
    elif u.dtype == np.float64:
        _add_csr64(*prepared, u.reshape(-1), out.reshape(-1))
    else:
        return False
    return True


def csr_rows(a):
    """(rows with entries, indptr, indices, data) of a scipy sparse matrix, for add_csr."""
    a = a.tocsr()
    if available() and a.shape[0] == a.shape[1] and a.shape[0] % 3 == 0 and a.nnz:
        blocks = _node_blocks(a)
        if blocks is None and a.has_canonical_format:
            blocks = _node_blocks(_padded_blocks(a))
        if blocks is not None:
            return blocks
    # 32-bit column indices where they fit: the pass reads 12 bytes an
    # entry instead of 16, the same sums (10-20% on the arm link's 9.7M)
    it = np.int32 if a.shape[1] < 2 ** 31 else np.int64
    return (np.nonzero(np.diff(a.indptr))[0].astype(np.int64), np.asarray(a.indptr, np.int64),
            np.asarray(a.indices, it), np.ascontiguousarray(a.data, np.float64))


if nb is not None:

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _dot(a, b):
        acc = 0.0
        for i in nb.prange(a.size):
            acc += a[i] * b[i]
        return acc


def cast(a, dtype):
    """a.astype(dtype) on numba's threads (numpy's is one thread, and the
    V-cycle casts the CG's residual and correction every iteration)."""
    dtype = np.dtype(dtype)
    if a.dtype == dtype:
        return a.copy()
    if not (available() and a.flags.c_contiguous and dtype in (np.float32, np.float64)
            and a.dtype in (np.float32, np.float64)):
        return a.astype(dtype)
    out = np.empty(a.shape, dtype)
    _cast_into(a.reshape(-1), out.reshape(-1))
    return out


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


if nb is not None:

    # The V-cycle's steps between a level and the next, each one pass where
    # numpy took three or four (and their temporaries). Every value goes
    # through the same operations as the numpy steps they replace, rounded
    # to the field's precision at the same points (`tmp` holds a value in
    # it), so the numbers are the same.

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _residual_pad(b, ax, m, rf):
        # rf = (b - ax) * m on b's nodes, zero on the rest of the (larger) rf
        nd, sx, sy, sz = b.shape
        fx, fy, fz = rf.shape[1], rf.shape[2], rf.shape[3]
        for xy in nb.prange(fx * fy):
            x = xy // fy
            y = xy % fy
            for d in range(nd):
                ud = U(d)
                for z in range(fz):
                    if x < sx and y < sy and z < sz:
                        rf[ud, U(x), U(y), U(z)] = (b[ud, U(x), U(y), U(z)] - ax[ud, U(x), U(y), U(z)]) \
                            * m[ud, U(x), U(y), U(z)]
                    else:
                        rf[ud, U(x), U(y), U(z)] = 0

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _restrict_masked(f, cm, c):
        # c = restrict(f) * cm
        nd, fx, fy, fz = f.shape
        cx_, cy_, cz_ = c.shape[1], c.shape[2], c.shape[3]
        for xy in nb.prange(cx_ * cy_):
            X = xy // cy_
            Y = xy % cy_
            tmp = np.empty(1, c.dtype)
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
                    tmp[0] = acc
                    c[ud, U(X), U(Y), U(Z)] = tmp[0] * cm[ud, U(X), U(Y), U(Z)]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _prolong_add(c, cm, m, xf):
        # xf += prolong(c * cm)[:, :sx, :sy, :sz] * m
        nd, sx, sy, sz = xf.shape
        for xy in nb.prange(sx * sy):
            x = xy // sy
            y = xy % sy
            x0 = x >> 1
            y0 = y >> 1
            x1 = x0 + (x & 1)
            y1 = y0 + (y & 1)
            wx = 0.5 if x & 1 else 1.0
            wy = 0.5 if y & 1 else 1.0
            tmp = np.empty(1, xf.dtype)
            for d in range(nd):
                ud = U(d)
                for z in range(sz):
                    z0 = z >> 1
                    z1 = z0 + (z & 1)
                    wz = 0.5 if z & 1 else 1.0
                    a = c[ud, U(x0), U(y0), U(z0)] * cm[ud, U(x0), U(y0), U(z0)]
                    if z & 1:
                        a = a + c[ud, U(x0), U(y0), U(z1)] * cm[ud, U(x0), U(y0), U(z1)]
                    if y & 1:
                        a = a + c[ud, U(x0), U(y1), U(z0)] * cm[ud, U(x0), U(y1), U(z0)]
                        if z & 1:
                            a = a + c[ud, U(x0), U(y1), U(z1)] * cm[ud, U(x0), U(y1), U(z1)]
                    if x & 1:
                        b = c[ud, U(x1), U(y0), U(z0)] * cm[ud, U(x1), U(y0), U(z0)]
                        if z & 1:
                            b = b + c[ud, U(x1), U(y0), U(z1)] * cm[ud, U(x1), U(y0), U(z1)]
                        if y & 1:
                            b = b + c[ud, U(x1), U(y1), U(z0)] * cm[ud, U(x1), U(y1), U(z0)]
                            if z & 1:
                                b = b + c[ud, U(x1), U(y1), U(z1)] * cm[ud, U(x1), U(y1), U(z1)]
                        a = a + b
                    tmp[0] = a * (wx * wy * wz)
                    xf[ud, U(x), U(y), U(z)] += tmp[0] * m[ud, U(x), U(y), U(z)]


def coarse_residual(b, ax, m, cm, coarse_shape):
    """restrict(((b - ax) * m) zero-padded to the grid twice coarse_shape's
    cells) * cm, as the V-cycle takes it down a level; None to use numpy."""
    if not (available() and all(a.flags.c_contiguous for a in (b, ax, m, cm))
            and b.dtype == ax.dtype == m.dtype == cm.dtype):
        return None
    rf = np.empty((3,) + tuple(2 * n - 1 for n in coarse_shape[1:]), b.dtype)
    _residual_pad(b, ax, m, rf)
    c = np.empty(coarse_shape, b.dtype)
    _restrict_masked(rf, cm, c)
    return c


def add_correction(xc, cm, m, x) -> bool:
    """x += prolong(xc * cm)[:, :x's nodes] * m in one pass; False when it
    cannot (the caller then does it with numpy)."""
    if not (available() and all(a.flags.c_contiguous for a in (xc, cm, m, x))
            and xc.dtype == cm.dtype == m.dtype == x.dtype):
        return False
    _prolong_add(xc, cm, m, x)
    return True


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
