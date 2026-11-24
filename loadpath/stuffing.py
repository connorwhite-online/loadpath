"""Tetrahedra filling a closed triangle mesh by isosurface stuffing
(Labelle & Shewchuk, "Isosurface Stuffing", SIGGRAPH 2007).

A body-centred cubic lattice of edge L is laid over the part. Every lattice
point is inside or outside (one ray per lattice row, crossings counted);
every lattice edge whose ends disagree is cut where it crosses the surface,
found exactly on the given triangles. A lattice point within a fixed
fraction of an edge from one of its cuts is moved onto that cut (warped),
so no element is cut thin against the surface; each lattice tetrahedron
is then clipped to the inside and the clipped pieces split into tets.

Every surface vertex of the result lies on the given surface. The
interior is the BCC lattice itself: its tetrahedra all have the same
shape (dihedral angles 60 and 90 degrees), and the warp bounds those cut
at the surface (the paper proves 10.7 to 164.8 degrees for its stencils;
quadrilaterals here split from their lowest-numbered corner, the same on
both tets that share one, which keeps the mesh conforming). Creases (box
edges) would come out chamfered, so surface vertices beside them are
moved onto them (_snap_creases); the eval parts then measure 9.9 to 162
degrees.

Unlike fTetWild it is deterministic (the same surface and edge give the
same mesh, bit for bit), and takes 0.1-4 s on the eval parts where
fTetWild took 30-130 s, mostly on one core: no envelope optimization,
only the lattice and the cuts.

Measured on 13 eval parts (docs/BENCHMARKS.md, "Tet mesher"): at the same
element size the safety factor is within 0.1-4% of fTetWild's on 12 and
9% lower (safe side) on the robot-arm wrist yoke; against meshes at 0.6x
the edge (converged to ~1%), stuffing is off by 1-5% and 9% on the yoke,
fTetWild by 0.5-6%.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# warp limits as a fraction of the edge (the paper's: long edges are the
# lattice's axis edges, short ones its body diagonals)
ALPHA_LONG = 0.24999
ALPHA_SHORT = 0.41189


@njit(cache=True)
def _parity_rows(tri, x0, nx, ys, zs, L):
    """inside[i, j, k] for the points (x0 + i L, ys[j], zs[k]): crossings of
    the +x ray through each (y, z) row with the triangles, counted beyond
    each point (odd = inside). Rows are nudged a fraction of a micron off
    the lattice, so a ray never runs exactly along a triangle edge."""
    ny, nz = len(ys), len(zs)
    y0, z0 = ys[0], zs[0]
    dy = ys[1] - ys[0] if ny > 1 else L
    dz = zs[1] - zs[0] if nz > 1 else L
    eps_y, eps_z = 1.37e-4 * L, 2.91e-4 * L
    nt = tri.shape[0]
    # pass 1: count crossings per row
    count = np.zeros(ny * nz, np.int64)
    for t in range(nt):
        ay, az = tri[t, 0, 1], tri[t, 0, 2]
        by, bz = tri[t, 1, 1], tri[t, 1, 2]
        cy, cz = tri[t, 2, 1], tri[t, 2, 2]
        jlo = max(0, int(np.ceil((min(ay, by, cy) - eps_y - y0) / dy)))
        jhi = min(ny - 1, int(np.floor((max(ay, by, cy) - eps_y - y0) / dy)))
        klo = max(0, int(np.ceil((min(az, bz, cz) - eps_z - z0) / dz)))
        khi = min(nz - 1, int(np.floor((max(az, bz, cz) - eps_z - z0) / dz)))
        if jlo > jhi or klo > khi:
            continue
        aby, abz, acy, acz = by - ay, bz - az, cy - ay, cz - az
        det = aby * acz - abz * acy
        if abs(det) < 1e-300:
            continue
        for j in range(jlo, jhi + 1):
            py = ys[j] + eps_y - ay
            for k in range(klo, khi + 1):
                pz = zs[k] + eps_z - az
                u = (py * acz - pz * acy) / det
                w = (aby * pz - abz * py) / det
                if u >= 0.0 and w >= 0.0 and u + w <= 1.0:
                    count[j * nz + k] += 1
    start = np.zeros(ny * nz + 1, np.int64)
    for r in range(ny * nz):
        start[r + 1] = start[r] + count[r]
    xs = np.empty(start[-1], np.float64)
    fill = start[:-1].copy()
    for t in range(nt):
        ay, az = tri[t, 0, 1], tri[t, 0, 2]
        by, bz = tri[t, 1, 1], tri[t, 1, 2]
        cy, cz = tri[t, 2, 1], tri[t, 2, 2]
        jlo = max(0, int(np.ceil((min(ay, by, cy) - eps_y - y0) / dy)))
        jhi = min(ny - 1, int(np.floor((max(ay, by, cy) - eps_y - y0) / dy)))
        klo = max(0, int(np.ceil((min(az, bz, cz) - eps_z - z0) / dz)))
        khi = min(nz - 1, int(np.floor((max(az, bz, cz) - eps_z - z0) / dz)))
        if jlo > jhi or klo > khi:
            continue
        aby, abz, acy, acz = by - ay, bz - az, cy - ay, cz - az
        det = aby * acz - abz * acy
        if abs(det) < 1e-300:
            continue
        abx, acx = tri[t, 1, 0] - tri[t, 0, 0], tri[t, 2, 0] - tri[t, 0, 0]
        for j in range(jlo, jhi + 1):
            py = ys[j] + eps_y - ay
            for k in range(klo, khi + 1):
                pz = zs[k] + eps_z - az
                u = (py * acz - pz * acy) / det
                w = (aby * pz - abz * py) / det
                if u >= 0.0 and w >= 0.0 and u + w <= 1.0:
                    r = j * nz + k
                    xs[fill[r]] = tri[t, 0, 0] + u * abx + w * acx
                    fill[r] += 1
    inside = np.zeros((nx, ny, nz), np.bool_)
    for r in prange(ny * nz):
        a, b = start[r], start[r + 1]
        if a == b:
            continue
        row = np.sort(xs[a:b])
        j, k = r // nz, r % nz
        m = 0   # crossings at or below x
        for i in range(nx):
            x = x0 + i * L
            while m < b - a and row[m] <= x:
                m += 1
            inside[i, j, k] = ((b - a) - m) % 2 == 1
    return inside


@njit(cache=True)
def _bin_triangles(tri, lo, cell, dims):
    """Triangles by the grid cells their bounding boxes touch: (start, items)."""
    nt = tri.shape[0]
    n = dims[0] * dims[1] * dims[2]
    count = np.zeros(n + 1, np.int64)
    for pass_ in range(2):
        if pass_ == 1:
            for c in range(n):
                count[c + 1] += count[c]
            items = np.empty(count[n], np.int64)
            fill = count[:n].copy()
        for t in range(nt):
            i0 = np.empty(3, np.int64)
            i1 = np.empty(3, np.int64)
            for d in range(3):
                a = min(tri[t, 0, d], tri[t, 1, d], tri[t, 2, d])
                b = max(tri[t, 0, d], tri[t, 1, d], tri[t, 2, d])
                i0[d] = max(0, int(np.floor((a - lo[d]) / cell)) )
                i1[d] = min(dims[d] - 1, int(np.floor((b - lo[d]) / cell)))
            for i in range(i0[0], i1[0] + 1):
                for j in range(i0[1], i1[1] + 1):
                    for k in range(i0[2], i1[2] + 1):
                        c = (i * dims[1] + j) * dims[2] + k
                        if pass_ == 0:
                            count[c + 1] += 1
                        else:
                            items[fill[c]] = t
                            fill[c] += 1
    return count, items


@njit(cache=True)
def _segment_cut(a, b, a_inside, tri, lo, cell, dims, start, items):
    """Where segment a-b crosses the surface, as t in (0, 1): the crossing
    nearest its inside end. -1 when no triangle is hit (a ray that grazed
    an edge for the parity count)."""
    i0 = np.empty(3, np.int64)
    i1 = np.empty(3, np.int64)
    for d in range(3):
        i0[d] = max(0, int(np.floor((min(a[d], b[d]) - lo[d]) / cell)))
        i1[d] = min(dims[d] - 1, int(np.floor((max(a[d], b[d]) - lo[d]) / cell)))
    best = -1.0
    hit = -1
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    for i in range(i0[0], i1[0] + 1):
        for j in range(i0[1], i1[1] + 1):
            for k in range(i0[2], i1[2] + 1):
                c = (i * dims[1] + j) * dims[2] + k
                for s in range(start[c], start[c + 1]):
                    t = items[s]
                    # Moller-Trumbore
                    e1x = tri[t, 1, 0] - tri[t, 0, 0]
                    e1y = tri[t, 1, 1] - tri[t, 0, 1]
                    e1z = tri[t, 1, 2] - tri[t, 0, 2]
                    e2x = tri[t, 2, 0] - tri[t, 0, 0]
                    e2y = tri[t, 2, 1] - tri[t, 0, 1]
                    e2z = tri[t, 2, 2] - tri[t, 0, 2]
                    px = dy * e2z - dz * e2y
                    py = dz * e2x - dx * e2z
                    pz = dx * e2y - dy * e2x
                    det = e1x * px + e1y * py + e1z * pz
                    if abs(det) < 1e-300:
                        continue
                    inv = 1.0 / det
                    sx, sy, sz = a[0] - tri[t, 0, 0], a[1] - tri[t, 0, 1], a[2] - tri[t, 0, 2]
                    u = (sx * px + sy * py + sz * pz) * inv
                    if u < -1e-9 or u > 1.0 + 1e-9:
                        continue
                    qx = sy * e1z - sz * e1y
                    qy = sz * e1x - sx * e1z
                    qz = sx * e1y - sy * e1x
                    v = (dx * qx + dy * qy + dz * qz) * inv
                    if v < -1e-9 or u + v > 1.0 + 1e-9:
                        continue
                    tt = (e2x * qx + e2y * qy + e2z * qz) * inv
                    if tt < -1e-12 or tt > 1.0 + 1e-12:
                        continue
                    if best < 0.0 or (tt < best if a_inside else tt > best):
                        best = tt
                        hit = t
    return best, hit


@njit(cache=True)
def _cut_or_near(a, b, a_inside, tri, lo, cell, dims, start, items):
    """_segment_cut, else (an end so near the surface that the nudged parity
    ray put it on the other side) the crossing within a twentieth of the
    edge beyond an end, taken at that end: an end that close is warped onto
    its cut anyway."""
    t, h = _segment_cut(a, b, a_inside, tri, lo, cell, dims, start, items)
    if t >= 0.0:
        return t, h
    d = 0.05
    a2 = a - d * (b - a)
    b2 = b + d * (b - a)
    t, h = _segment_cut(a2, b2, a_inside, tri, lo, cell, dims, start, items)
    if t >= 0.0:
        return min(1.0, max(0.0, (t * (1.0 + 2.0 * d)) - d)), h
    # still none (an end on a crease, the ray through it ambiguous): the
    # cut is at the end nearer the surface
    da, ha = _nearest(a, tri, lo, cell, dims, start, items)
    db, hb = _nearest(b, tri, lo, cell, dims, start, items)
    if ha < 0 and hb < 0:
        return -1.0, -1
    return (0.0, ha) if da <= db else (1.0, hb)


@njit(cache=True)
def _closest_on_triangle(p, a, b, c):
    """Squared distance from p to triangle abc (Ericson, Real-Time
    Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        q = a
    else:
        bp = p - b
        d3, d4 = ab @ bp, ac @ bp
        cp = p - c
        d5, d6 = ab @ cp, ac @ cp
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        if d3 >= 0.0 and d4 <= d3:
            q = b
        elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            q = a + d1 / (d1 - d3) * ab
        elif d6 >= 0.0 and d5 <= d6:
            q = c
        elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            q = a + d2 / (d2 - d6) * ac
        elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            q = b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
        else:
            den = 1.0 / (va + vb + vc)
            q = a + ab * (vb * den) + ac * (vc * den)
    e = p - q
    return e @ e


@njit(cache=True)
def _nearest(p, tri, lo, cell, dims, start, items):
    """(squared distance, triangle) of the triangle nearest p among the
    bins within one cell of it."""
    best, hit = np.inf, -1
    i0 = np.empty(3, np.int64)
    i1 = np.empty(3, np.int64)
    for d in range(3):
        k = int(np.floor((p[d] - lo[d]) / cell))
        i0[d] = max(0, k - 1)
        i1[d] = min(dims[d] - 1, k + 1)
    for i in range(i0[0], i1[0] + 1):
        for j in range(i0[1], i1[1] + 1):
            for k in range(i0[2], i1[2] + 1):
                c = (i * dims[1] + j) * dims[2] + k
                for s in range(start[c], start[c + 1]):
                    t = items[s]
                    d2 = _closest_on_triangle(p, tri[t, 0], tri[t, 1], tri[t, 2])
                    if d2 < best:
                        best, hit = d2, t
    return best, hit


@njit(cache=True)
def _inside_points(p, tri, lo, cell, dims, start, items):
    """Parity of the +x ray from each point, over the binned triangles (a
    hit counts in the cell its x falls in, so a triangle in several cells
    counts once)."""
    out = np.zeros(len(p), np.bool_)
    for r in range(len(p)):
        y, z = p[r, 1] + 1.37e-7, p[r, 2] + 2.91e-7
        j = int(np.floor((y - lo[1]) / cell))
        k = int(np.floor((z - lo[2]) / cell))
        if j < 0 or k < 0 or j >= dims[1] or k >= dims[2]:
            continue
        hits = 0
        for i in range(max(0, int(np.floor((p[r, 0] - lo[0]) / cell))), dims[0]):
            c = (i * dims[1] + j) * dims[2] + k
            x0, x1 = lo[0] + i * cell, lo[0] + (i + 1) * cell
            for s in range(start[c], start[c + 1]):
                t = items[s]
                aby = tri[t, 1, 1] - tri[t, 0, 1]
                abz = tri[t, 1, 2] - tri[t, 0, 2]
                acy = tri[t, 2, 1] - tri[t, 0, 1]
                acz = tri[t, 2, 2] - tri[t, 0, 2]
                det = aby * acz - abz * acy
                if abs(det) < 1e-300:
                    continue
                py, pz = y - tri[t, 0, 1], z - tri[t, 0, 2]
                u = (py * acz - pz * acy) / det
                w = (aby * pz - abz * py) / det
                if u < 0.0 or w < 0.0 or u + w > 1.0:
                    continue
                x = tri[t, 0, 0] + u * (tri[t, 1, 0] - tri[t, 0, 0]) + w * (tri[t, 2, 0] - tri[t, 0, 0])
                if x > p[r, 0] and x >= x0 and (x < x1 or i == dims[0] - 1):
                    hits += 1
        out[r] = hits % 2 == 1
    return out


@njit(cache=True)
def _octant_corner(i, j, k, o, ny, nz):
    """Corner id of octant o (bits x, y, z) of the cube (i, j, k)."""
    return ((i + (o >> 2 & 1)) * (ny + 1) + (j + (o >> 1 & 1))) * (nz + 1) + (k + (o & 1))


@njit(cache=True, parallel=True)
def _cuts(sign, origin, L, nx, ny, nz, tri, lo, cell, dims, start, items):
    """t of every lattice edge whose ends disagree (-1 elsewhere): red (corner
    to corner along an axis), black (centre to centre) and diagonal (centre
    to the 8 corners of its cube)."""
    nc = (nx + 1) * (ny + 1) * (nz + 1)
    nb = nx * ny * nz
    red = -np.ones(3 * nc, np.float64)
    black = -np.ones(3 * nb, np.float64)
    diag = -np.ones(8 * nb, np.float64)
    hit = -np.ones(3 * nc + 11 * nb, np.int64)
    miss = np.zeros(nx + 1, np.int64)
    for i in prange(nx + 1):
        a = np.empty(3)
        b = np.empty(3)
        for j in range(ny + 1):
            for k in range(nz + 1):
                c = (i * (ny + 1) + j) * (nz + 1) + k
                sc = sign[c]
                a[0], a[1], a[2] = origin[0] + i * L, origin[1] + j * L, origin[2] + k * L
                for d in range(3):
                    ii, jj, kk = i + (d == 0), j + (d == 1), k + (d == 2)
                    if ii > nx or jj > ny or kk > nz:
                        continue
                    c2 = (ii * (ny + 1) + jj) * (nz + 1) + kk
                    if sign[c2] == sc:
                        continue
                    b[0], b[1], b[2] = origin[0] + ii * L, origin[1] + jj * L, origin[2] + kk * L
                    t, h = _cut_or_near(a, b, sc < 0, tri, lo, cell, dims, start, items)
                    if t < 0.0:
                        t = 0.5
                        miss[i] += 1
                    red[3 * c + d] = t
                    hit[3 * c + d] = h
                if i < nx and j < ny and k < nz:
                    q = (i * ny + j) * nz + k
                    sq = sign[nc + q]
                    a[0], a[1], a[2] = origin[0] + (i + .5) * L, origin[1] + (j + .5) * L, origin[2] + (k + .5) * L
                    for d in range(3):
                        ii, jj, kk = i + (d == 0), j + (d == 1), k + (d == 2)
                        if ii >= nx or jj >= ny or kk >= nz:
                            continue
                        q2 = (ii * ny + jj) * nz + kk
                        if sign[nc + q2] == sq:
                            continue
                        b[0], b[1], b[2] = (origin[0] + (ii + .5) * L, origin[1] + (jj + .5) * L,
                                            origin[2] + (kk + .5) * L)
                        t, h = _cut_or_near(a, b, sq < 0, tri, lo, cell, dims, start, items)
                        if t < 0.0:
                            t = 0.5
                            miss[i] += 1
                        black[3 * q + d] = t
                        hit[3 * nc + 3 * q + d] = h
                    for o in range(8):
                        c2 = _octant_corner(i, j, k, o, ny, nz)
                        if sign[c2] == sq:
                            continue
                        b[0] = origin[0] + (i + (o >> 2 & 1)) * L
                        b[1] = origin[1] + (j + (o >> 1 & 1)) * L
                        b[2] = origin[2] + (k + (o & 1)) * L
                        t, h = _cut_or_near(a, b, sq < 0, tri, lo, cell, dims, start, items)
                        if t < 0.0:
                            t = 0.5
                            miss[i] += 1
                        diag[8 * q + o] = t
                        hit[3 * nc + 3 * nb + 8 * q + o] = h
    return red, black, diag, hit, miss.sum()


@njit(cache=True)
def _point(p, origin, L, nx, ny, nz, out):
    nc = (nx + 1) * (ny + 1) * (nz + 1)
    if p < nc:
        k = p % (nz + 1)
        j = (p // (nz + 1)) % (ny + 1)
        i = p // ((nz + 1) * (ny + 1))
        out[0], out[1], out[2] = origin[0] + i * L, origin[1] + j * L, origin[2] + k * L
    else:
        q = p - nc
        k = q % nz
        j = (q // nz) % ny
        i = q // (nz * ny)
        out[0], out[1], out[2] = origin[0] + (i + .5) * L, origin[1] + (j + .5) * L, origin[2] + (k + .5) * L


@njit(cache=True)
def _edge_ends(e, nx, ny, nz):
    """Lattice point ids (a, b) of structured edge e (red, black, diag)."""
    nc = (nx + 1) * (ny + 1) * (nz + 1)
    nb = nx * ny * nz
    if e < 3 * nc:
        c, d = e // 3, e % 3
        k = c % (nz + 1)
        j = (c // (nz + 1)) % (ny + 1)
        i = c // ((nz + 1) * (ny + 1))
        i, j, k = i + (d == 0), j + (d == 1), k + (d == 2)
        return c, (i * (ny + 1) + j) * (nz + 1) + k
    e -= 3 * nc
    if e < 3 * nb:
        q, d = e // 3, e % 3
        k = q % nz
        j = (q // nz) % ny
        i = q // (nz * ny)
        i, j, k = i + (d == 0), j + (d == 1), k + (d == 2)
        return nc + q, nc + (i * ny + j) * nz + k
    e -= 3 * nb
    q, o = e // 8, e % 8
    k = q % nz
    j = (q // nz) % ny
    i = q // (nz * ny)
    return nc + q, _octant_corner(i, j, k, o, ny, nz)


@njit(cache=True)
def _warp(sign, cut, origin, L, nx, ny, nz):
    """Lattice points within the warp limit of a cut on one of their edges:
    for each, the edge whose cut is nearest (-1 for none). The point moves
    onto that cut and lies on the surface."""
    nc = (nx + 1) * (ny + 1) * (nz + 1)
    nb = nx * ny * nz
    npts = nc + nb
    best = np.full(npts, np.inf)
    where = -np.ones(npts, np.int64)
    short = 0.5 * np.sqrt(3.0) * L
    for e in range(len(cut)):
        t = cut[e]
        if t < 0.0:
            continue
        a, b = _edge_ends(e, nx, ny, nz)
        is_short = e >= 3 * nc + 3 * nb
        length = short if is_short else L
        alpha = ALPHA_SHORT if is_short else ALPHA_LONG
        if t < alpha and t * length < best[a]:
            best[a] = t * length
            where[a] = e
        if 1.0 - t < alpha and (1.0 - t) * length < best[b]:
            best[b] = (1.0 - t) * length
            where[b] = e
    return where


@njit(cache=True)
def _split_quad(q0, q1, q2, q3):
    """Diagonal of quad q0 q1 q2 q3 (in cyclic order) from its lowest id:
    True for q0-q2, False for q1-q3."""
    m = min(q0, q1, q2, q3)
    return m == q0 or m == q2


@njit(cache=True)
def _emit(out, n, a, b, c, d):
    out[n, 0], out[n, 1], out[n, 2], out[n, 3] = a, b, c, d
    return n + 1


@njit(cache=True)
def _prism(out, n, A, B):
    """Three tets of the prism A[0..2] (one triangle) to B[0..2] (the other,
    A[i]-B[i] its side edges), each side quad split from its lowest id
    (Dompierre et al. 1999: never the cyclic choice, so always 3 tets)."""
    V = np.empty(6, np.int64)
    m, mi = A[0], 0
    for i in range(3):
        if A[i] < m:
            m, mi = A[i], i
        if B[i] < m:
            m, mi = B[i], i + 3
    # rotate (and swap the triangles) so the lowest id is V[0]
    if mi < 3:
        r = mi
        for i in range(3):
            V[i], V[i + 3] = A[(r + i) % 3], B[(r + i) % 3]
    else:
        r = mi - 3
        for i in range(3):
            V[i], V[i + 3] = B[(r + i) % 3], A[(r + i) % 3]
    v1, v2, v3, v4, v5, v6 = V[0], V[1], V[2], V[3], V[4], V[5]
    if min(v2, v6) < min(v3, v5):
        n = _emit(out, n, v1, v2, v3, v6)
        n = _emit(out, n, v1, v2, v6, v5)
        n = _emit(out, n, v1, v5, v6, v4)
    else:
        n = _emit(out, n, v1, v2, v3, v5)
        n = _emit(out, n, v1, v5, v3, v6)
        n = _emit(out, n, v1, v5, v6, v4)
    return n


# local edges of a lattice tet (vertices 0 = centre, 1 = next centre,
# 2, 3 = the corners of a face edge)
_LE = np.array([[-1, 0, 1, 2], [0, -1, 3, 4], [1, 3, -1, 5], [2, 4, 5, -1]], np.int64)


@njit(cache=True)
def _stuff_tet(out, n, ids, s, ecut, le):
    """The inside part of one lattice tet (ids, signs s in -1/0/1 after the
    warp, ecut[6] the global ids of the cut points on its edges) as tets."""
    neg = np.empty(4, np.int64)
    zer = np.empty(4, np.int64)
    pos = np.empty(4, np.int64)
    nn = nz_ = np_ = 0
    for v in range(4):
        if s[v] < 0:
            neg[nn] = v
            nn += 1
        elif s[v] == 0:
            zer[nz_] = v
            nz_ += 1
        else:
            pos[np_] = v
            np_ += 1
    if nn == 0:
        if nz_ == 4:
            # every corner on the surface: inside or not is decided later,
            # at its centroid (marked by negated ids)
            return _emit(out, n, -1 - ids[0], -1 - ids[1], -1 - ids[2], -1 - ids[3])
        return n
    if np_ == 0:
        return _emit(out, n, ids[0], ids[1], ids[2], ids[3])
    if nn == 1:
        a = neg[0]
        pts = np.empty(4, np.int64)
        pts[0] = ids[a]
        m = 1
        for i in range(nz_):
            pts[m] = ids[zer[i]]
            m += 1
        for i in range(np_):
            pts[m] = ecut[le[a, pos[i]]]
            m += 1
        return _emit(out, n, pts[0], pts[1], pts[2], pts[3])
    if nn == 2:
        a, b = neg[0], neg[1]
        if np_ == 2:
            p, q = pos[0], pos[1]
            A = np.array([ids[a], ecut[le[a, p]], ecut[le[a, q]]])
            B = np.array([ids[b], ecut[le[b, p]], ecut[le[b, q]]])
            return _prism(out, n, A, B)
        # one zero, one positive: a pyramid on the quad a b bp ap, apex z
        z, p = zer[0], pos[0]
        q0, q1, q2, q3 = ids[a], ids[b], ecut[le[b, p]], ecut[le[a, p]]
        if _split_quad(q0, q1, q2, q3):
            n = _emit(out, n, q0, q1, q2, ids[z])
            n = _emit(out, n, q0, q2, q3, ids[z])
        else:
            n = _emit(out, n, q0, q1, q3, ids[z])
            n = _emit(out, n, q1, q2, q3, ids[z])
        return n
    # three negative, one positive: the prism a b c to their cuts
    p = pos[0]
    A = np.array([ids[neg[0]], ids[neg[1]], ids[neg[2]]])
    B = np.array([ecut[le[neg[0], p]], ecut[le[neg[1], p]], ecut[le[neg[2], p]]])
    return _prism(out, n, A, B)


@njit(cache=True)
def _lattice_tets(sign, red, black, diag, nx, ny, nz):
    """Every lattice tet clipped to the inside: (k, 4) ids, lattice point ids
    below n_points, cut ids above (n_points + structured edge)."""
    nc = (nx + 1) * (ny + 1) * (nz + 1)
    nb = nx * ny * nz
    npts = nc + nb
    off_black = npts + 3 * nc
    off_diag = off_black + 3 * nb
    cap = 1024
    out = np.empty((cap, 4), np.int64)
    n = 0
    ids = np.empty(4, np.int64)
    s = np.empty(4, np.int64)
    ecut = np.empty(6, np.int64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                q = (i * ny + j) * nz + k
                for d in range(3):
                    ii, jj, kk = i + (d == 0), j + (d == 1), k + (d == 2)
                    if ii >= nx or jj >= ny or kk >= nz:
                        continue
                    q2 = (ii * ny + jj) * nz + kk
                    a1, a2 = (d + 1) % 3, (d + 2) % 3
                    for f in range(4):
                        # the face edge: along a1 at a2 offset v, or along a2 at a1 offset u
                        if f < 2:
                            u0, v0, u1, v1, ax = 0, f, 1, f, a1
                        else:
                            u0, v0, u1, v1, ax = f - 2, 0, f - 2, 1, a2
                        op = np.zeros(3, np.int64)
                        oq = np.zeros(3, np.int64)
                        op[d], op[a1], op[a2] = 1, u0, v0
                        oq[d], oq[a1], oq[a2] = 1, u1, v1
                        o_p = op[0] * 4 + op[1] * 2 + op[2]
                        o_q = oq[0] * 4 + oq[1] * 2 + oq[2]
                        cp = _octant_corner(i, j, k, o_p, ny, nz)
                        cq = _octant_corner(i, j, k, o_q, ny, nz)
                        ids[0], ids[1], ids[2], ids[3] = nc + q, nc + q2, cp, cq
                        any_neg = False
                        all_neg = True
                        for v in range(4):
                            s[v] = sign[ids[v]]
                            if s[v] < 0:
                                any_neg = True
                            if s[v] > 0:
                                all_neg = False
                        if not any_neg:
                            continue
                        if not all_neg:
                            # cut ids of the six local edges
                            op2 = op.copy()
                            oq2 = oq.copy()
                            op2[d], oq2[d] = 0, 0
                            ecut[0] = off_black + 3 * q + d
                            ecut[1] = off_diag + 8 * q + o_p
                            ecut[2] = off_diag + 8 * q + o_q
                            ecut[3] = off_diag + 8 * q2 + op2[0] * 4 + op2[1] * 2 + op2[2]
                            ecut[4] = off_diag + 8 * q2 + oq2[0] * 4 + oq2[1] * 2 + oq2[2]
                            ecut[5] = npts + 3 * cp + ax
                        if n + 3 > cap:
                            cap *= 2
                            grown = np.empty((cap, 4), np.int64)
                            grown[:n] = out[:n]
                            out = grown
                        n = _stuff_tet(out, n, ids, s, ecut, _LE)
    return out[:n]


@njit(cache=True)
def _positions(used, red, black, diag, hit, where, origin, L, nx, ny, nz):
    """Coordinates of the used vertex ids, and the triangle each surface
    vertex lies on (-1 for lattice points left where they were)."""
    nc = (nx + 1) * (ny + 1) * (nz + 1)
    nb = nx * ny * nz
    npts = nc + nb
    cut = np.concatenate((red, black, diag))
    out = np.empty((len(used), 3))
    on = -np.ones(len(used), np.int64)
    pa = np.empty(3)
    pb = np.empty(3)
    for r in range(len(used)):
        g = used[r]
        e = -1
        if g < npts:
            if where[g] < 0:
                _point(g, origin, L, nx, ny, nz, pa)
                out[r] = pa
                continue
            e = where[g]
        else:
            e = g - npts
        a, b = _edge_ends(e, nx, ny, nz)
        _point(a, origin, L, nx, ny, nz, pa)
        _point(b, origin, L, nx, ny, nz, pb)
        t = cut[e]
        out[r] = pa + t * (pb - pa)
        on[r] = hit[e]
    return out, on


def stuff(mesh, L, info=None, creases=True):
    """(points, tets) filling the closed triangle mesh `mesh` (trimesh, or
    (vertices, faces)) with a BCC lattice of edge L, warped to its surface.
    Tets are positively oriented. `info`, a dict, gets counts."""
    if isinstance(mesh, tuple):
        v, f = mesh
    else:
        v, f = mesh.vertices, mesh.faces
    v = np.asarray(v, np.float64)
    tri = np.ascontiguousarray(v[np.asarray(f, np.int64)])
    L = float(L)
    lo, hi = v.min(0), v.max(0)
    # the lattice anchored to the world (multiples of L), one cell of air
    # beyond the part on every side
    origin = (np.floor(lo / L) - 1.0) * L
    n = np.ceil((hi - origin) / L).astype(np.int64) + 1
    nx, ny, nz = (int(x) for x in n)
    ys = origin[1] + L * np.arange(ny + 1)
    zs = origin[2] + L * np.arange(nz + 1)
    corners = _parity_rows(tri, origin[0], nx + 1, ys, zs, L)
    centres = _parity_rows(tri, origin[0] + 0.5 * L, nx, ys[:-1] + 0.5 * L, zs[:-1] + 0.5 * L, L)
    sign = np.concatenate([np.where(corners.ravel(), -1, 1), np.where(centres.ravel(), -1, 1)]).astype(np.int8)
    cell = L
    dims = np.ceil((hi - lo) / cell).astype(np.int64) + 1
    start, items = _bin_triangles(tri, lo, cell, dims)
    red, black, diag, hit, miss = _cuts(sign, origin, L, nx, ny, nz, tri, lo, cell, dims, start, items)
    cut = np.concatenate((red, black, diag))
    where = _warp(sign, cut, origin, L, nx, ny, nz)
    sign = sign.copy()
    sign[where >= 0] = 0
    tets = _lattice_tets(sign, red, black, diag, nx, ny, nz)
    onsurf = tets[:, 0] < 0
    if onsurf.any():
        flat = -1 - tets[onsurf]
        used0 = np.unique(flat)
        p0 = _positions(used0, red, black, diag, hit, where, origin, L, nx, ny, nz)[0]
        cen = p0[np.searchsorted(used0, flat)].mean(1)
        keep = _inside_points(cen, tri, lo, cell, dims, start, items)
        tets = np.concatenate([tets[~onsurf], flat[keep]])
    used, inv = np.unique(tets, return_inverse=True)
    pts, on = _positions(used, red, black, diag, hit, where, origin, L, nx, ny, nz)
    tets = np.ascontiguousarray(inv.reshape(-1, 4))
    x = pts[tets]
    vol = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
    neg = vol < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    snapped = _snap_creases(pts, tets, on, v, np.asarray(f, np.int64), L) if creases else 0
    if info is not None:
        info.update(lattice=(nx, ny, nz), onSurface=int(onsurf.sum()), missedCuts=int(miss),
                    warped=int((where >= 0).sum()), snapped=int(snapped))
    return pts, tets


# A crease: two surface triangles meeting at more than this angle between
# their normals (a box edge 90, a 45-degree chamfer's edges 45). The
# delivered surfaces' smooth parts turn a few degrees from one triangle to
# the next (a 1.5 mm fillet in 0.3 mm triangles: 11), so they stay smooth.
CREASE_ANGLE = np.radians(30.0)
# Least quality (6 sqrt2 V / rms edge^3; 1 for the regular tet) a snap may
# leave on a tet it moves, unless the tet was worse already: a fifth, the
# bar fTetWild's meshes met (crosscheck._tet_mesh).
SNAP_QUALITY = 0.2


def _creases(v, f):
    """Patch label per triangle (triangles joined across smooth edges), and
    the crease segments (k, 2, 3)."""
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m = trimesh.Trimesh(v, f, process=False)
    adj = m.face_adjacency
    sharp = m.face_adjacency_angles > CREASE_ANGLE
    smooth = adj[~sharp]
    g = coo_matrix((np.ones(len(smooth)), (smooth[:, 0], smooth[:, 1])), shape=(len(f), len(f)))
    _, patch = connected_components(g, directed=False)
    segs = v[m.face_adjacency_edges[sharp]]
    # a crease between triangles of one patch (it fades out into a smooth
    # surface) separates nothing
    keep = patch[adj[sharp, 0]] != patch[adj[sharp, 1]]
    return patch, segs[keep]


@njit(cache=True)
def _quality(p, t):
    a, b, c, d = p[t[0]], p[t[1]], p[t[2]], p[t[3]]
    u, w, z = b - a, c - a, d - a
    vol = (u[0] * (w[1] * z[2] - w[2] * z[1]) - u[1] * (w[0] * z[2] - w[2] * z[0])
           + u[2] * (w[0] * z[1] - w[1] * z[0])) / 6.0
    s = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            e = p[t[i]] - p[t[j]]
            s += e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
    rms = np.sqrt(s / 6.0)
    return 6.0 * np.sqrt(2.0) * vol / (rms * rms * rms)


@njit(cache=True)
def _snap_loop(pts, tets, ptr, inc, eu, ev, target, dist, order, limit, qmin):
    """Greedy: for each crossing edge (nearest the crease first) move the
    end nearer the crease onto it, if no tet it touches inverts or falls
    below qmin (or below what it was). Returns how many moved."""
    done = np.zeros(len(pts), np.bool_)
    tried = np.zeros(len(pts), np.bool_)
    moved = 0
    for r in order:
        u, v = eu[r], ev[r]
        if done[u] or done[v]:
            continue
        first, second = (u, v) if dist[u] <= dist[v] else (v, u)
        for w in (first, second):
            if tried[w] or dist[w] > limit:
                continue
            tried[w] = True
            old = pts[w].copy()
            before = np.inf
            for s in range(ptr[w], ptr[w + 1]):
                before = min(before, _quality(pts, tets[inc[s]]))
            pts[w] = target[w]
            ok = True
            for s in range(ptr[w], ptr[w + 1]):
                if _quality(pts, tets[inc[s]]) < min(qmin, before):
                    ok = False
                    break
            if ok:
                done[w] = True
                moved += 1
                break
            pts[w] = old
    return moved


def _snap_creases(pts, tets, on, v, f, L):
    """Isosurface stuffing cuts a crease (a box edge) off as a chamfer up to
    an element wide: every surface vertex lies on the surface, but none on
    the edge itself. On a 10 mm square beam meshed at 2.5 mm off the
    lattice that took 13% of its bending stiffness. Surface vertices on
    either side of a crease are moved onto it, one per surface edge that
    crosses it, nearest first, where no tet they touch is spoilt."""
    patch, segs = _creases(v, f)
    if not len(segs):
        return 0
    from scipy.spatial import cKDTree

    # points along every crease, a fiftieth of an element apart
    ln = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    k = np.maximum(1, np.ceil(ln / (L / 50.0)).astype(np.int64))
    seg_of = np.repeat(np.arange(len(segs)), k + 1)
    t = np.concatenate([np.linspace(0.0, 1.0, n + 1) for n in k])
    samples = segs[seg_of, 0] + t[:, None] * (segs[seg_of, 1] - segs[seg_of, 0])
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), 1)
    u, c = np.unique(faces, axis=0, return_counts=True)
    skin = u[c == 1]
    edges = np.unique(np.sort(np.concatenate([skin[:, [0, 1]], skin[:, [1, 2]], skin[:, [0, 2]]]), 1), axis=0)
    lab = np.where(on >= 0, patch[np.maximum(on, 0)], -1)
    cross = edges[(lab[edges[:, 0]] >= 0) & (lab[edges[:, 1]] >= 0) & (lab[edges[:, 0]] != lab[edges[:, 1]])]
    if not len(cross):
        return 0
    ends = np.unique(cross)
    d, j = cKDTree(samples).query(pts[ends])
    dist = np.full(len(pts), np.inf)
    target = pts.copy()
    dist[ends] = d
    target[ends] = samples[j]
    order = np.argsort(np.minimum(dist[cross[:, 0]], dist[cross[:, 1]]), kind="stable")
    # tets around each vertex
    flat = tets.ravel()
    inc = np.argsort(flat, kind="stable") // 4
    ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=len(pts)))])
    return _snap_loop(pts, tets, ptr, inc, cross[:, 0].copy(), cross[:, 1].copy(), target, dist, order,
                      L, SNAP_QUALITY)
