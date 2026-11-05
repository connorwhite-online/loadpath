"""pyamg's Gauss-Seidel smoother without the GIL.

fem.Solver.solve_many runs a study's load cases on threads, one CG each,
sharing one AMG hierarchy: the sparse matrix products release the GIL,
but pyamg's compiled Gauss-Seidel (amg_core.gauss_seidel,
bsr_gauss_seidel and block_gauss_seidel, the V-cycle's smoothers and most
of a solve's time) holds it, so the cases' smoothing ran one at a time:
on the robot-arm upper arm three cases took about as long as three solves.

These are the same loops (pyamg 5.x amg_core/relaxation.h: the same
operations in the same order, so the same bits) as numba kernels that
release the GIL. `install()` swaps them in for pyamg's three functions;
everything else in amg_core stays pyamg's. LOADPATH_AMGFAST=0 leaves
pyamg alone.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import numba as nb

    from . import meshops as _meshops
except ImportError:  # pragma: no cover - numba is a dependency
    nb = None

_INSTALLED = False

if nb is not None:
    # Indices go through uint64: numba wraps negative signed indices, a
    # compare and conditional move on every load that made the kernels half
    # pyamg's speed. The wrappers hand over the index arrays viewed unsigned.
    U = np.uint64

    @nb.njit(cache=True, nogil=True)
    def _gs(Ap, Aj, Ax, x, b, row_start, row_stop, row_step):
        i = row_start
        while i != row_stop:
            ui = U(i)
            rsum = 0.0
            diag = 0.0
            jj = U(Ap[ui])
            end = U(Ap[ui + U(1)])
            while jj < end:
                j = U(Aj[jj])
                if j == ui:
                    diag = Ax[jj]
                else:
                    rsum += Ax[jj] * x[j]
                jj += U(1)
            if diag != 0.0:
                x[ui] = (b[ui] - rsum) / diag
            i += row_step

    @nb.njit(cache=True, nogil=True)
    def _bsr_gs(Ap, Aj, Ax, x, b, row_start, row_stop, row_step, bs):
        # relaxation.h bsr_gauss_seidel: off-diagonal blocks by gemm (each
        # into zeros, then subtracted), then the diagonal block pointwise,
        # in the sweep's direction
        ubs = U(bs)
        B2 = ubs * ubs
        rsum = np.empty(bs)
        axloc = np.empty(bs)
        if row_step < 0:
            step, k0, k1 = -1, bs - 1, -1
        else:
            step, k0, k1 = 1, 0, bs
        i = row_start
        while i != row_stop:
            ui = U(i)
            ib = ui * ubs
            for k in range(bs):
                rsum[k] = b[ib + U(k)]
            diag_ptr = U(0)
            has_diag = False
            jj = U(Ap[ui])
            end = U(Ap[ui + U(1)])
            while jj < end:
                j = U(Aj[jj])
                if j == ui:
                    diag_ptr = jj * B2
                    has_diag = True
                else:
                    col = j * ubs
                    blk = jj * B2
                    for m in range(bs):
                        acc = 0.0
                        row = blk + U(m) * ubs
                        for k in range(bs):
                            acc += Ax[row + U(k)] * x[col + U(k)]
                        axloc[m] = acc
                    for m in range(bs):
                        rsum[m] -= axloc[m]
                jj += U(1)
            if has_diag:
                k = k0
                while k != k1:
                    uk = U(k)
                    diag = 1.0
                    kk = k0
                    while kk != k1:
                        ukk = U(kk)
                        if k == kk:
                            diag = Ax[uk * ubs + ukk + diag_ptr]
                        else:
                            rsum[k] -= Ax[uk * ubs + ukk + diag_ptr] * x[ib + ukk]
                        kk += step
                    if diag != 0.0:
                        x[ib + uk] = rsum[k] / diag
                    k += step
            i += row_step

    @nb.njit(cache=True, nogil=True)
    def _block_gs(Ap, Aj, Ax, x, b, Dinv, row_start, row_stop, row_step, bs):
        # relaxation.h block_gauss_seidel: rsum = sum of off-diagonal block
        # products (each gemm into zeros), x_i = Dinv_i (b_i - rsum)
        ubs = U(bs)
        B2 = ubs * ubs
        rsum = np.empty(bs)
        v = np.empty(bs)
        i = row_start
        while i != row_stop:
            ui = U(i)
            ib = ui * ubs
            for k in range(bs):
                rsum[k] = 0.0
            jj = U(Ap[ui])
            end = U(Ap[ui + U(1)])
            while jj < end:
                j = U(Aj[jj])
                if j != ui:
                    col = j * ubs
                    blk = jj * B2
                    for m in range(bs):
                        acc = 0.0
                        row = blk + U(m) * ubs
                        for k in range(bs):
                            acc += Ax[row + U(k)] * x[col + U(k)]
                        v[m] = acc
                    for k in range(bs):
                        rsum[k] += v[k]
                jj += U(1)
            for k in range(bs):
                rsum[k] = b[ib + U(k)] - rsum[k]
            dblk = ui * B2
            for m in range(bs):
                acc = 0.0
                row = dblk + U(m) * ubs
                for k in range(bs):
                    acc += Dinv[row + U(k)] * rsum[k]
                x[ib + U(m)] = acc
            i += row_step

    @nb.njit(cache=True, nogil=True, inline="always")
    def _row_cols(i, Ap, Aj, Bp, Bj, mark, cols):
        # the block row's output columns in scipy's order: first visit
        n = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = U(Aj[jj])
            for kk in range(Bp[j], Bp[j + U(1)]):
                k = Bj[kk]
                if mark[k] != i:
                    mark[k] = i
                    cols[n] = k
                    n += 1
        return n

    @nb.njit(cache=True, parallel=True)
    def _bsr_count(n_brow, n_bcol, Ap, Aj, Bp, Bj, counts):
        nch = 64
        for c in nb.prange(nch):
            mark = np.full(n_bcol, -1, np.int64)
            cols = np.empty(n_bcol, np.int64)
            for i in range(c * n_brow // nch, (c + 1) * n_brow // nch):
                counts[i] = _row_cols(i, Ap, Aj, Bp, Bj, mark, cols)

    @nb.njit(cache=True, parallel=True)
    def _bsr_fill(n_brow, n_bcol, R, C, N, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx):
        # scipy sparsetools bsr_matmat, a block row at a time: each output
        # block gathers A(i, j) B(j, k) in the same (jj, kk) order, by its
        # gemm (dot from the block's value, d ascending), so the same bits
        RC, RN, NC = R * C, R * N, N * C
        nch = 64
        for c in nb.prange(nch):
            mark = np.full(n_bcol, -1, np.int64)
            slot = np.empty(n_bcol, np.int64)
            cols = np.empty(n_bcol, np.int64)
            for i in range(c * n_brow // nch, (c + 1) * n_brow // nch):
                n = _row_cols(i, Ap, Aj, Bp, Bj, mark, cols)
                base = Cp[i]
                for q in range(n):
                    Cj[base + q] = cols[q]
                    slot[cols[q]] = base + q
                    for t in range(RC):
                        Cx[(base + q) * RC + t] = 0.0
                for jj in range(Ap[i], Ap[i + 1]):
                    j = U(Aj[jj])
                    a0 = U(jj) * U(RN)
                    for kk in range(Bp[j], Bp[j + U(1)]):
                        k = Bj[kk]
                        b0 = U(kk) * U(NC)
                        c0 = U(slot[k]) * U(RC)
                        for r in range(R):
                            for cc in range(C):
                                o = c0 + U(C * r + cc)
                                dot = Cx[o]
                                for d in range(N):
                                    dot += Ax[a0 + U(N * r + d)] * Bx[b0 + U(C * d + cc)]
                                Cx[o] = dot


def _u(a):
    """An index array viewed unsigned (they are never negative)."""
    return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint64)


def _ok(*arrays):
    return all(a.dtype == np.float64 for a in arrays)


from pyamg import amg_core


class _Core:
    """amg_core with the three Gauss-Seidel kernels replaced."""

    def __init__(self, core):
        self._core = core

    def __getattr__(self, name):
        return getattr(self._core, name)

    @staticmethod
    def gauss_seidel(Ap, Aj, Ax, x, b, row_start, row_stop, row_step):
        if not _ok(Ax, x, b):
            return amg_core.gauss_seidel(Ap, Aj, Ax, x, b, row_start, row_stop, row_step)
        _gs(_u(Ap), _u(Aj), Ax, x, b, int(row_start), int(row_stop), int(row_step))

    @staticmethod
    def bsr_gauss_seidel(Ap, Aj, Ax, x, b, row_start, row_stop, row_step, R):
        if not _ok(Ax, x, b):
            return amg_core.bsr_gauss_seidel(Ap, Aj, Ax, x, b, row_start, row_stop, row_step, R)
        _bsr_gs(_u(Ap), _u(Aj), Ax, x, b, int(row_start), int(row_stop), int(row_step), int(R))

    @staticmethod
    def block_gauss_seidel(Ap, Aj, Ax, x, b, Tx, row_start, row_stop, row_step, R):
        if not _ok(Ax, x, b, Tx):
            return amg_core.block_gauss_seidel(Ap, Aj, Ax, x, b, Tx, row_start, row_stop, row_step, R)
        _block_gs(_u(Ap), _u(Aj), Ax, x, b, Tx, int(row_start), int(row_stop), int(row_step), int(R))


_MATMUL_MIN = 20_000      # block nonzeros of A below which scipy's own is as quick


def _bsr_matmul(self, other, _orig=None):
    """scipy's BSR @ sparse (bsr._matmul_sparse) with the product itself on
    numba's threads. pyamg's setup (R A P, the prolongation smoother) does
    these on one core while the others wait; same entries, same order, so
    the same bits."""
    M = self.shape[0]
    N = other.shape[1]
    R, n = self.blocksize
    C = other.blocksize[1] if other.format == "bsr" else 1
    if (self.data.dtype != np.float64 or other.dtype != np.float64 or len(self.indices) < _MATMUL_MIN
            or (R == 1 and n == 1 and C == 1)):
        return _orig(self, other)
    if other.format == "csr" and n == 1:
        other = other.tobsr(blocksize=(n, C), copy=False)
    else:
        other = other.tobsr(blocksize=(n, C))
    n_brow, n_bcol = M // R, N // C
    Ap = np.ascontiguousarray(self.indptr, np.int64)
    Aj = np.ascontiguousarray(self.indices, np.int64)
    Bp = np.ascontiguousarray(other.indptr, np.int64)
    Bj = np.ascontiguousarray(other.indices, np.int64)
    counts = np.empty(n_brow, np.int64)
    _bsr_count(n_brow, n_bcol, Ap, Aj, Bp, Bj, counts)
    Cp = np.zeros(n_brow + 1, np.int64)
    np.cumsum(counts, out=Cp[1:])
    nnz = int(Cp[-1])
    Cj = np.empty(nnz, np.int64)
    Cx = np.empty(R * C * nnz, np.float64)
    _bsr_fill(n_brow, n_bcol, R, C, n, Ap, Aj, np.ascontiguousarray(np.ravel(self.data)), Bp, Bj,
              np.ascontiguousarray(np.ravel(other.data)), Cp, Cj, Cx)
    idx = self._get_index_dtype((self.indptr, self.indices, other.indptr, other.indices), maxval=nnz)
    return self._bsr_container((Cx.reshape(-1, R, C), Cj.astype(idx), Cp.astype(idx)), shape=(M, N),
                               blocksize=(R, C))


_SCIPY_BSR_MATMUL = None


def _bsr_matmul_method(self, other):
    return _bsr_matmul(self, other, _orig=_SCIPY_BSR_MATMUL)


_bsr_matmul_method._loadpath = True


def installed():
    """Whether pyamg's Gauss-Seidel is the GIL-free one now."""
    return _INSTALLED


def install():
    """Use the GIL-free kernels in pyamg's relaxation module (once)."""
    global _INSTALLED
    if _INSTALLED or nb is None or os.environ.get("LOADPATH_AMGFAST", "1") == "0":
        return _INSTALLED
    try:
        import pyamg.relaxation.relaxation as rel
    except ImportError:  # pragma: no cover
        return False
    if not isinstance(rel.amg_core, _Core):
        rel.amg_core = _Core(rel.amg_core)
    global _SCIPY_BSR_MATMUL
    from scipy.sparse import _bsr

    orig = _bsr._bsr_base._matmul_sparse
    if not getattr(orig, "_loadpath", False):
        _SCIPY_BSR_MATMUL = orig
        _bsr._bsr_base._matmul_sparse = _bsr_matmul_method
    _INSTALLED = True
    return True
