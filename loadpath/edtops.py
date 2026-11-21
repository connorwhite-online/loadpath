"""The exact Euclidean distance transform on numba's threads.

scipy.ndimage.distance_transform_edt runs on one core: 1.7 s of the
printability check's 6.7 s on the arm link (a 4M-cell tile at a time), and
loadpath calls it in over a hundred places. Here it is Felzenszwalb and
Huttenlocher's separable transform (Theory of Computing 8 (2012) 415-428):
the squared distance to the nearest zero along each line of axis 2, then
the lower envelope of parabolas along axis 1, then axis 0, each pass over
its lines in parallel. Every squared distance is an integer and is carried
as one, the envelope's breakpoints compared exactly by cross-multiplying,
so the result is the square root of the same integer scipy finds: the same
float64 to the bit.

`edt(a)` stands for `ndimage.distance_transform_edt(a)` on a 2D or 3D array
with unit sampling, no indices returned (a 2D array is one layer of a 3D
one: along a length-1 axis the envelope is the line's own value); it falls
back to scipy without numba, for other dimensions, or with
LOADPATH_EDTOPS=scipy.
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

__all__ = ["available", "edt"]

_INF = np.int64(1) << np.int64(62)


def available():
    return nb is not None and os.environ.get("LOADPATH_EDTOPS", "").lower() != "scipy"


if nb is not None:

    @nb.njit(cache=True, nogil=True, inline="always")
    def _envelope(f, n, d, v, z_num, z_den, inf):
        # d[q] = min_p f[p] + (q - p)^2 over p with f[p] < inf (inf where none);
        # v: the envelope's parabolas, z: their breakpoints as fractions
        k = -1
        for q in range(n):
            fq = f[q]
            if fq >= inf:
                continue
            if k < 0:
                k = 0
                v[0] = q
                continue
            while k >= 0:
                p = v[k]
                # intersection of parabolas p and q: s = num / den, den > 0
                num = (fq + q * q) - (f[p] + p * p)
                den = 2 * (q - p)
                # drop p while s <= z[k] (the breakpoint where p took over)
                if k > 0 and num * z_den[k] <= z_num[k] * den:
                    k -= 1
                else:
                    break
            p = v[k]
            k += 1
            v[k] = q
            z_num[k] = (fq + q * q) - (f[p] + p * p)
            z_den[k] = 2 * (q - p)
        if k < 0:
            for q in range(n):
                d[q] = inf
            return
        j = 0
        for q in range(n):
            # move on while the next breakpoint is at or before q
            while j < k and z_num[j + 1] <= q * z_den[j + 1]:
                j += 1
            p = v[j]
            d[q] = f[p] + (q - p) * (q - p)

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _pass_z(a, g, inf):
        nx, ny, nz = a.shape
        for t in nb.prange(nx * ny):
            i, j = t // ny, t % ny
            # squared distance to the nearest zero along z, two sweeps
            last = -1
            for k in range(nz):
                if not a[i, j, k]:
                    last = k
                    g[i, j, k] = 0
                elif last >= 0:
                    g[i, j, k] = (k - last) * (k - last)
                else:
                    g[i, j, k] = inf
            last = -1
            for k in range(nz - 1, -1, -1):
                if not a[i, j, k]:
                    last = k
                elif last >= 0:
                    dd = (last - k) * (last - k)
                    g[i, j, k] = min(g[i, j, k], dd)

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _pass_y(g, inf):
        nx, ny, nz = g.shape
        for t in nb.prange(nx * nz):
            i, k = t // nz, t % nz
            f = np.empty(ny, np.int64)
            d = np.empty(ny, np.int64)
            v = np.empty(ny, np.int64)
            zn = np.empty(ny + 1, np.int64)
            zd = np.empty(ny + 1, np.int64)
            for j in range(ny):
                f[j] = g[i, j, k]
            _envelope(f, ny, d, v, zn, zd, inf)
            for j in range(ny):
                g[i, j, k] = d[j]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _pass_x(g, out, inf):
        nx, ny, nz = g.shape
        for t in nb.prange(ny * nz):
            j, k = t // nz, t % nz
            f = np.empty(nx, np.int64)
            d = np.empty(nx, np.int64)
            v = np.empty(nx, np.int64)
            zn = np.empty(nx + 1, np.int64)
            zd = np.empty(nx + 1, np.int64)
            for i in range(nx):
                f[i] = g[i, j, k]
            _envelope(f, nx, d, v, zn, zd, inf)
            for i in range(nx):
                out[i, j, k] = np.sqrt(np.float64(d[i])) if d[i] < inf else np.inf


def edt(a):
    """ndimage.distance_transform_edt(a) for a 2D or 3D array: each nonzero
    cell's Euclidean distance (in cells) to the nearest zero, float64."""
    a = np.asarray(a)
    if a.ndim == 2 and available():
        return edt(a[None])[0]
    if not available() or a.ndim != 3 or a.size == 0 or (a != 0).all():
        from scipy import ndimage

        return ndimage.distance_transform_edt(a)
    m = np.ascontiguousarray(a != 0)
    g = np.empty(m.shape, np.int64)
    _pass_z(m, g, _INF)
    _pass_y(g, _INF)
    out = np.empty(m.shape, np.float64)
    _pass_x(g, out, _INF)
    return out
