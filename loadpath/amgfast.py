"""pyamg's Gauss-Seidel smoother without the GIL.

fem.Solver.solve_many runs a study's load cases on threads, one CG each,
sharing one AMG hierarchy: the sparse matrix products release the GIL,
but pyamg's compiled Gauss-Seidel (amg_core.gauss_seidel,
bsr_gauss_seidel and block_gauss_seidel, the V-cycle's smoothers and most
of a solve's time) holds it, so the cases' smoothing ran one at a time:
on the GRIP upper arm three cases took about as long as three solves.

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
    _INSTALLED = True
    return True
