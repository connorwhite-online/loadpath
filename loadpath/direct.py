"""
A direct factorization where it is the quickest way to solve a stiffness
system: the optimizer's and the verification's small and thin ones.

The iterative paths each have a regime. Smoothed-aggregation AMG (fem.Solver)
handles anything but pays a setup and tens of Gauss-Seidel sweeps for every
new design. Matrix-free geometric multigrid (mg.py) is several times quicker
on a large, bulky grid, but its coarse grids cannot represent members one or
two cells thick: on the wrist yoke's 4% design (1,650 live cells of 27,634)
it took 470-520 CG iterations where AMG took 28. A sparse Cholesky is exact
whatever the design looks like, shares one factor between every load case
and adjoint, and on what a density design leaves alive it is small: thin
members have tiny separators, so the factor fills in very little.

Measured on the 4-core cloud box, PARDISO, one factor plus the solves:

    printed hook     live  9,575 of 17,476 cells   38k dofs   0.45 s
    wrist yoke       live  3,266 of 27,634 cells   17k dofs   0.21 s
    printed hook     all  17,476 cells             61k dofs   0.80 s
    wrist yoke       all  27,634 cells             98k dofs   1.35 s
    printed hook     all  70,856 cells            233k dofs   5.4 s

against 0.8-1.4 s (AMG) and 0.4-0.5 s (multigrid, where it converges) per
optimizer iteration on the first four, 15-20 s and 1.0 s on the last. So:
direct up to SMALL dofs, multigrid above that on a grid big enough for it
(simp._Analysis), direct again up to LARGE where multigrid is not on offer
or has stalled, AMG beyond.

PARDISO (pypardiso: Intel MKL, x86-64) is the factorization meant here.
Without it SuperLU stands in up to a size where it is still quick; past that
the iterative paths run as before.

LOADPATH_DIRECT=0 turns it off; LOADPATH_DIRECT_DOFS="small,large" sets the
two sizes.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["DirectSolver", "backend", "small_dofs", "large_dofs", "rescue_dofs", "solve_many"]

# dofs up to which a factorization is the first choice / still beats AMG-CG
_SIZES = {"pardiso": (110_000, 300_000), "superlu": (9_000, 9_000)}
# ... and up to which one takes over from a CG that has not converged in
# fem.Solver.STALL iterations: slower than a CG that converges, but it ends,
# and by then it costs less than the iterations already spent. A solid
# block of 212k dofs: 400 AMG-CG iterations 118 s, PARDISO 9 s, SuperLU 72 s
# (and 0.8 GB); PARDISO on the wrist yoke's whole 119k-cell grid, 398k dofs,
# 19 s and 3.4 GB.
_RESCUE = {"pardiso": 450_000, "superlu": 200_000}
# relative residual a solve must reach (after one refinement step at most);
# the callers' own tolerances are 1e-3 to 1e-6
RTOL = 1e-7


def backend() -> str | None:
    """"pardiso", "superlu" or None (LOADPATH_DIRECT=0)."""
    if os.environ.get("LOADPATH_DIRECT", "1") == "0":
        return None
    if not os.environ.get("LOADPATH_NO_PARDISO"):
        from .nonlinear import _have_pardiso

        if _have_pardiso():
            return "pardiso"
    return "superlu"


def _sizes() -> tuple[int, int]:
    b = backend()
    if b is None:
        return 0, 0
    env = os.environ.get("LOADPATH_DIRECT_DOFS")
    if env:
        try:
            small, _, large = env.partition(",")
            small = int(float(small))
            return small, max(small, int(float(large))) if large else small
        except ValueError:
            pass
    return _SIZES[b]


def small_dofs() -> int:
    """Up to this many dofs a factorization is the quickest solve there is."""
    return _sizes()[0]


def large_dofs() -> int:
    """Up to this many it still beats AMG-preconditioned CG."""
    return _sizes()[1]


def rescue_dofs() -> int:
    """Up to this many a factorization takes over from a CG that has
    stalled (fem.Solver.STALL)."""
    b = backend()
    return max(_RESCUE[b], large_dofs()) if b is not None else 0


class DirectSolver:
    """Stands in for fem.Solver: `solve_many(k, fs)` factors k once and
    solves every right-hand side as one block; a second call with the same
    matrix object (the adjoints of the same design) reuses the factor.

    A singular k (live cells joined by an edge or a corner: a hinge; a
    piece no support holds) does not always fail to factor, so every answer
    is checked against k: a residual over RTOL after one refinement step
    raises RuntimeError, as a CG that does not converge does."""

    def __init__(self, node_ijk=None, rtol=None):
        self.node_ijk = node_ijk     # SuperLU's nested-dissection order
        self.rtol = rtol             # (unused: the solve is exact; fem.Solver's attribute)
        self.last_iters = 0
        self._k = None
        self._fac = None

    def invalidate(self):
        self._k = None
        self._fac = None

    def _factor(self, k):
        if self._fac is not None and self._k is k:
            return self._fac
        self.invalidate()
        try:
            if backend() == "pardiso":
                from .nonlinear import _Pardiso

                fac = _Pardiso(k)
            else:
                from .nonlinear import _splu

                fac = _splu(k, self.node_ijk)
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - a factorization that broke down is a solve that failed
            raise RuntimeError(f"direct factorization failed ({type(exc).__name__}: {exc}); "
                               "the structure may be a mechanism (check supports)") from exc
        self._k, self._fac = k, fac
        return fac

    def _solve_block(self, fac, b):
        if hasattr(fac, "_s"):          # PARDISO: a block in one call
            return np.asarray(fac.solve(b)).reshape(b.shape)
        return np.column_stack([np.asarray(fac.solve(np.ascontiguousarray(b[:, j]))) for j in range(b.shape[1])])

    def solve_many(self, k, fs, x0s=None):
        fs = [np.asarray(f, float) for f in fs]
        if not fs:
            return []
        b = np.column_stack(fs)
        bn = np.linalg.norm(b, axis=0)
        fac = self._factor(k)
        with np.errstate(all="ignore"):
            x = self._solve_block(fac, b)
            r = b - k @ x
            rel = np.linalg.norm(r, axis=0) / np.where(bn > 0, bn, 1.0)
            if not (np.all(np.isfinite(rel)) and rel.max() <= RTOL):
                # one step of iterative refinement: enough for a stiff-to-void
                # contrast the factor rounded, not for a mechanism
                x = x + self._solve_block(fac, np.where(np.isfinite(r), r, 0.0))
                r = b - k @ x
                rel = np.linalg.norm(r, axis=0) / np.where(bn > 0, bn, 1.0)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(rel)) and rel.max() <= 100 * RTOL):
            self.invalidate()
            raise RuntimeError(f"direct solve left a residual of {float(np.nanmax(rel)):.1e}; "
                               "the structure may be a mechanism (check supports)")
        self.last_iters = 0
        return [np.ascontiguousarray(x[:, j]) for j in range(b.shape[1])]

    def solve(self, k, f, x0=None):
        return self.solve_many(k, [f])[0]


def solve_many(k, fs, node_ijk=None):
    """k x = f for each f on one factorization, or None where that is not
    the quick way (no backend, more than large_dofs()) or it did not hold
    its residual (a hinge, a piece nothing holds): the caller's iterative
    solver then runs, and reports a mechanism in its own words. For the one-off
    systems of a check (verify's grid and its submodels), where the AMG
    hierarchy and tens of CG iterations per right-hand side were most of
    the time: the wrist yoke's cut-cell grid (71k cells) 27 s to 4 s, a
    submodel 11 s to under 1."""
    n = k.shape[0]
    if not 0 < n <= large_dofs():
        return None
    try:
        return DirectSolver(node_ijk).solve_many(k, fs)
    except RuntimeError:
        return None
