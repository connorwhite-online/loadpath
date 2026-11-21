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

    @nb.njit(cache=True, nogil=True, inline="always")
    def _voronoi(ft, line, n, d, coor, f, g):
        # scipy's _VoronoiFT (ni_morphology.c) on one line of ft, decision
        # for decision: f holds each cell's feature (f[i, 0] < 0: none yet),
        # g the sites still on the envelope. Every quantity is a whole
        # number well inside float64's exact range, so these float64 sums
        # and comparisons are scipy's to the bit.
        for i in range(n):
            for c in range(3):
                f[i, c] = ft[c, line[0, i], line[1, i], line[2, i]]
        lst = -1
        for i in range(n):
            if f[i, 0] < 0:
                continue
            fd = np.float64(f[i, d])
            wr = 0.0
            for c in range(3):
                if c != d:
                    tw = np.float64(f[i, c] - coor[c])
                    wr += tw * tw
            while lst >= 1:
                i1 = g[lst]
                i2 = g[lst - 1]
                f1 = np.float64(f[i1, d])
                a = f1 - np.float64(f[i2, d])
                b = fd - f1
                cc_ = a + b
                ur = 0.0
                vr = 0.0
                for c in range(3):
                    if c != d:
                        tu = np.float64(f[i2, c] - coor[c])
                        tv = np.float64(f[i1, c] - coor[c])
                        ur += tu * tu
                        vr += tv * tv
                if cc_ * vr - b * ur - a * wr - a * b * cc_ <= 0.0:
                    break
                lst -= 1
            lst += 1
            g[lst] = i
        mx = lst
        if mx < 0:
            return
        lst = 0
        for i in range(n):
            d1 = 0.0
            for c in range(3):
                t = np.float64(f[g[lst], c] - (i if c == d else coor[c]))
                d1 += t * t
            while lst < mx:
                d2 = 0.0
                for c in range(3):
                    t = np.float64(f[g[lst + 1], c] - (i if c == d else coor[c]))
                    d2 += t * t
                if d1 <= d2:
                    break
                d1 = d2
                lst += 1
            k = g[lst]
            for c in range(3):
                ft[c, line[0, i], line[1, i], line[2, i]] = f[k, c]

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _ft_pass(ft, d):
        shp = ft.shape[1:]
        n = shp[d]
        a1 = 1 if d == 0 else 0
        a2 = 2 if d != 2 else 1
        n1, n2 = shp[a1], shp[a2]
        for t in nb.prange(n1 * n2):
            p, q = t // n2, t % n2
            line = np.empty((3, n), np.int64)
            coor = np.zeros(3, np.int64)
            coor[a1] = p
            coor[a2] = q
            for i in range(n):
                line[a1, i] = p
                line[a2, i] = q
                line[d, i] = i
            f = np.empty((n, 3), np.int64)
            g = np.empty(n, np.int64)
            _voronoi(ft, line, n, d, coor, f, g)

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _ft_init(m, ft):
        # scipy's first pass marks each zero cell its own feature and every
        # other cell -1 (its other components left as they were: zero)
        nx, ny, nz = m.shape
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if m[i, j, k]:
                        ft[0, i, j, k] = -1
                    else:
                        ft[0, i, j, k] = i
                        ft[1, i, j, k] = j
                        ft[2, i, j, k] = k

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _ft_distance(ft, out):
        # scipy: sqrt(add.reduce((ft - indices)^2 in float64, axis=0))
        _, nx, ny, nz = ft.shape
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    a = np.float64(ft[0, i, j, k] - i)
                    b = np.float64(ft[1, i, j, k] - j)
                    c = np.float64(ft[2, i, j, k] - k)
                    out[i, j, k] = np.sqrt(a * a + b * b + c * c)


def edt(a, return_distances=True, return_indices=False):
    """ndimage.distance_transform_edt(a, return_distances=, return_indices=)
    for a 2D or 3D array: each nonzero cell's Euclidean distance (in cells)
    to the nearest zero, float64, and/or that zero's indices, int32 of shape
    (a.ndim,) + a.shape. Returns what scipy returns, to the bit, ties
    included."""
    if not return_distances and not return_indices:
        raise RuntimeError("at least one of return_distances/return_indices must be True")
    a = np.asarray(a)
    if a.ndim == 2 and available():
        r = edt(a[None], return_distances, return_indices)
        if not return_indices:
            return r[0]
        if not return_distances:
            return np.ascontiguousarray(r[1:, 0])
        return r[0][0], np.ascontiguousarray(r[1][1:, 0])
    if not available() or a.ndim != 3 or a.size == 0 or (a != 0).all():
        from scipy import ndimage

        return ndimage.distance_transform_edt(a, return_distances=return_distances,
                                              return_indices=return_indices)
    m = np.ascontiguousarray(a != 0)
    if return_indices:
        return _feature_transform(m, return_distances)
    g = np.empty(m.shape, np.int64)
    _pass_z(m, g, _INF)
    _pass_y(g, _INF)
    out = np.empty(m.shape, np.float64)
    _pass_x(g, out, _INF)
    return out


def _feature_transform(m, return_distances):
    # scipy's euclidean_feature_transform: Maurer's separable Voronoi passes
    # along axis 0, then 1, then 2, each line's ties broken as scipy's are
    ft = np.zeros((3,) + m.shape, np.int32)
    _ft_init(m, ft)
    for d in range(3):
        _ft_pass(ft, d)
    if not return_distances:
        return ft
    out = np.empty(m.shape, np.float64)
    _ft_distance(ft, out)
    return out, ft
