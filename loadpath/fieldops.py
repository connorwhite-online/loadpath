"""Compiled kernels for the hot field-graph ops (numba, nogil, so peval's
threads run them in parallel), each giving the numpy op's values:

- `tpms`: gyroid / diamond / Schwarz-P distance in double precision, one
  pass per point (sines and cosines once, no temporaries): ~4x numpy's
  float64, bit for bit. Single precision stays on numpy, whose float32
  sin/cos are SIMD and faster still.
- `warp`: a graded lattice's coordinates (_CellWarp), the three trilinear
  lookups fused, as map_coordinates(order=1, mode="nearest") of the
  clipped point plus the carry-on past the grid: ~5x.

`available()` is False without numba or with LOADPATH_FIELDOPS=numpy,
and callers keep their numpy code.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import numba as nb

    from . import meshops as _meshops  # noqa: F401 - caps numba's threads (LOADPATH_THREADS)
except ImportError:  # pragma: no cover - numba is a dependency; numpy without it
    nb = None

__all__ = ["available", "tpms", "warp"]

_KINDS = {"gyroid": 0, "diamond": 1, "schwarz-p": 2}


def available():
    return nb is not None and os.environ.get("LOADPATH_FIELDOPS", "").lower() != "numpy"


if nb is not None:

    @nb.njit(cache=True, nogil=True)
    def _tpms(kind, P, k, out):
        for i in range(P.shape[0]):
            x = k * P[i, 0]
            y = k * P[i, 1]
            z = k * P[i, 2]
            if kind == 2:
                # lattice.TPMS["schwarz-p"]: cos x + cos y + cos z, gradient -sin
                f = np.cos(x) + np.cos(y) + np.cos(z)
                g0 = -np.sin(x) * k
                g1 = -np.sin(y) * k
                g2 = -np.sin(z) * k
                gn = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
                out[i] = f / max(gn, 0.3 * k)
                continue
            sx, cx, sy, cy, sz, cz = np.sin(x), np.cos(x), np.sin(y), np.cos(y), np.sin(z), np.cos(z)
            if kind == 0:
                f = sx * cy + sy * cz + sz * cx
                g0 = cx * cy - sz * sx
                g1 = cy * cz - sx * sy
                g2 = cz * cx - sy * sz
            else:
                f = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
                g0 = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
                g1 = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
                g2 = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
            gn = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2) * k
            out[i] = f / max(gn, 0.3 * k)

    @nb.njit(cache=True, nogil=True, inline="always")
    def _lerp_index(x, n):
        # map_coordinates order 1, mode nearest, x already in [0, n - 1]
        i0 = int(np.floor(x))
        if i0 > n - 2:
            i0 = n - 2 if n > 1 else 0
        w = x - i0
        i1 = i0 + 1 if n > 1 else 0
        return i0, i1, w

    @nb.njit(cache=True, nogil=True)
    def _warp(P, lo, h, u0, u1, u2, cell, out):
        n0, n1, n2 = u0.shape
        for i in range(P.shape[0]):
            ix = (P[i, 0] - lo[0]) / h
            iy = (P[i, 1] - lo[1]) / h
            iz = (P[i, 2] - lo[2]) / h
            cx = min(max(ix, 0.0), n0 - 1.0)
            cy = min(max(iy, 0.0), n1 - 1.0)
            cz = min(max(iz, 0.0), n2 - 1.0)
            a0, a1, wa = _lerp_index(cx, n0)
            b0, b1, wb = _lerp_index(cy, n1)
            c0, c1, wc = _lerp_index(cz, n2)
            c = cell[i] if cell.shape[0] > 1 else cell[0]
            for comp in range(3):
                u = u0 if comp == 0 else (u1 if comp == 1 else u2)
                v00 = u[a0, b0, c0] * (1 - wc) + u[a0, b0, c1] * wc
                v01 = u[a0, b1, c0] * (1 - wc) + u[a0, b1, c1] * wc
                v10 = u[a1, b0, c0] * (1 - wc) + u[a1, b0, c1] * wc
                v11 = u[a1, b1, c0] * (1 - wc) + u[a1, b1, c1] * wc
                v0 = v00 * (1 - wb) + v01 * wb
                v1 = v10 * (1 - wb) + v11 * wb
                v = v0 * (1 - wa) + v1 * wa
                d = ix - cx if comp == 0 else (iy - cy if comp == 1 else iz - cz)
                out[i, comp] = v + d * h / c


def tpms(kind, pts, cell):
    """lattice.tpms(kind, pts, cell) in double precision (scalar cell)."""
    pts = np.ascontiguousarray(pts, np.float64)
    out = np.empty(len(pts))
    _tpms(_KINDS[kind], pts, 2.0 * np.pi / float(cell), out)
    return out


def warp(P, lo, h, us, cell):
    """_CellWarp.__call__: the three lattice coordinates at P."""
    P = np.ascontiguousarray(P, np.float64)
    cell = np.ascontiguousarray(np.atleast_1d(np.asarray(cell, np.float64)).ravel())
    out = np.empty((len(P), 3))
    _warp(P, np.ascontiguousarray(lo, np.float64), float(h), us[0], us[1], us[2], cell, out)
    return out
