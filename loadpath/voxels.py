"""
Occupancy grids and wall-thickness probes that scale to multi-million
triangle parts.

The checks used to voxelize through trimesh (`mesh.voxelized`, then a ray
parity vote per enclosed pocket) and measure walls with one ray per sampled
face. Both go through trimesh's ray broad phase, which on a 6.6M-triangle
TPMS heat exchanger hands every ray a candidate set covering most of the
mesh: the printability check reached 63 GB and the isolation check took 30
minutes. Everything here is plain numpy and bounded in memory:

  mesh_solid   — scanline rasterizer. Every voxel column along an axis is a
                 ray; each triangle is binned straight to the columns its
                 projection covers (no tree), the crossing heights are kept,
                 and parity along the column fills the solid. Cost is
                 O(triangles + crossings), memory is a bool grid plus 8 bytes
                 per crossing. Orientation independent, so inverted cavity
                 shells and nested bodies come out right; a mesh that is not
                 watertight gets a majority vote over the three axes.
  field_solid  — the same grid from an implicit field f(P) < 0, evaluated a
                 slab at a time. For parts built from a field this skips the
                 mesh altogether, and is exact at the voxel centres.
  wall_thickness — member thickness on the medial ridge of the grid (twice
                 the distance to void at its local maxima), from a distance
                 transform computed in overlapping tiles so it never holds
                 more than one tile of floats.

Grids follow one convention: (bool grid, origin) where origin is the world
position of the CENTRE of voxel (0, 0, 0), with `pad` void voxels on every
side.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["grid_for", "mesh_solid", "field_solid", "solid_grid", "wall_thickness",
           "tiled_edt_min", "choose_pitch"]

# Column-centre jitter, as a share of the pitch: keeps every scanline off the
# exact edges and vertices of the mesh, where a crossing could be counted by
# both triangles sharing the edge (or by neither) and flip the parity.
_JITTER = (0.000137, 0.000291)

# Candidate (triangle, column) pairs tested per batch.
_BATCH = 2_000_000

# Floats held per tile by the tiled distance transform (~20 bytes each).
_TILE_VOXELS = 16_000_000


def choose_pitch(extents, target, budget, pad=2):
    """`target` pitch, coarsened by 1.25x steps until the grid fits `budget`
    voxels. Returns the pitch actually used."""
    extents = np.asarray(extents, float)
    pitch = float(target)
    while float(np.prod(np.ceil(extents / pitch) + 2 * pad + 1)) > budget:
        pitch *= 1.25
    return pitch


def grid_for(lo, hi, pitch, pad=2):
    """(dims, origin) of a grid covering the box lo..hi with `pad` voxels of
    margin. origin is the centre of voxel (0, 0, 0)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    core = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    dims = core + 2 * pad
    origin = lo - 0.5 * (core * pitch - (hi - lo)) + (0.5 - pad) * pitch
    return dims, origin


def _crossings(tri, dims, origin, pitch, axis):
    """Scanline crossings of triangles `tri` (T, 3, 3) with the voxel columns
    along `axis`. Returns (flat column index, crossing position along the
    axis in cells from voxel 0's centre, the sign of the triangle's projected
    area: which way it faces along the axis, up to the axis's handedness)."""
    u_ax, v_ax = [a for a in range(3) if a != axis]
    nu, nv, nw = dims[u_ax], dims[v_ax], dims[axis]
    ou = origin[u_ax] + _JITTER[0] * pitch
    ov = origin[v_ax] + _JITTER[1] * pitch
    ow = origin[axis]

    U, V, W = tri[:, :, u_ax], tri[:, :, v_ax], tri[:, :, axis]
    i0 = np.clip(np.ceil((U.min(1) - ou) / pitch), 0, nu).astype(np.int64)
    i1 = np.clip(np.floor((U.max(1) - ou) / pitch), -1, nu - 1).astype(np.int64)
    j0 = np.clip(np.ceil((V.min(1) - ov) / pitch), 0, nv).astype(np.int64)
    j1 = np.clip(np.floor((V.max(1) - ov) / pitch), -1, nv - 1).astype(np.int64)
    wi = np.maximum(i1 - i0 + 1, 0)
    wj = np.maximum(j1 - j0 + 1, 0)
    count = wi * wj
    # 2x the projected signed area; a triangle seen edge-on crosses no column
    den = (V[:, 1] - V[:, 2]) * (U[:, 0] - U[:, 2]) + (U[:, 2] - U[:, 1]) * (V[:, 0] - V[:, 2])
    keep = (count > 0) & (np.abs(den) > 1e-14)
    idx = np.nonzero(keep)[0]
    cols_out, ks_out, sg_out = [], [], []
    if idx.size == 0:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int8)
    csum = np.cumsum(count[idx])
    start = 0
    while start < idx.size:
        base = csum[start - 1] if start else 0
        stop = int(np.searchsorted(csum, base + _BATCH, side="right"))
        stop = max(stop, start + 1)
        t = idx[start:stop]
        n = count[t]
        rep = np.repeat(np.arange(t.size), n)
        off = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        ti = t[rep]
        ii = i0[ti] + off // wj[ti]
        jj = j0[ti] + off % wj[ti]
        x = ou + ii * pitch
        y = ov + jj * pitch
        Ut, Vt, Wt, d = U[ti], V[ti], W[ti], den[ti]
        l0 = ((Vt[:, 1] - Vt[:, 2]) * (x - Ut[:, 2]) + (Ut[:, 2] - Ut[:, 1]) * (y - Vt[:, 2])) / d
        l1 = ((Vt[:, 2] - Vt[:, 0]) * (x - Ut[:, 2]) + (Ut[:, 0] - Ut[:, 2]) * (y - Vt[:, 2])) / d
        l2 = 1.0 - l0 - l1
        hit = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if hit.any():
            w = l0[hit] * Wt[hit, 0] + l1[hit] * Wt[hit, 1] + l2[hit] * Wt[hit, 2]
            cols_out.append(ii[hit] * nv + jj[hit])
            ks_out.append((w - ow) / pitch)
            sg_out.append(np.sign(d[hit]).astype(np.int8))
        start = stop
    if not cols_out:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int8)
    return np.concatenate(cols_out), np.concatenate(ks_out), np.concatenate(sg_out)


def _parity_axis(tri, dims, origin, pitch, axis, touch=None):
    """Solid by crossing parity along `axis`; returns (grid, odd-column
    count, overlap cells). With `touch` (a bool grid in x, y, z order), also
    marks there every voxel a crossing falls in.

    Overlap cells are where the winding number (crossings counted by the way
    their triangle faces) says solid and parity says void: two closed bodies
    concatenated without a boolean, which parity cancels to void where they
    overlap, or a cavity shell facing the wrong way. The grid stays parity
    (it does not trust winding), but a caller can say the reading is off."""
    u_ax, v_ax = [a for a in range(3) if a != axis]
    nu, nv, nw = int(dims[u_ax]), int(dims[v_ax]), int(dims[axis])
    cols, w, sg = _crossings(tri, dims, origin, pitch, axis)
    # (u, v, axis) cyclic for x and z, anticyclic for y; entering the solid
    # (an outward normal facing down the axis) counts +1
    sg = -sg.astype(np.float64) * (-1.0 if axis == 1 else 1.0)
    if touch is not None and cols.size:
        k = np.rint(w).astype(np.int64)
        ok = (k >= 0) & (k < nw)
        ijk = np.empty((int(ok.sum()), 3), np.int64)
        ijk[:, u_ax] = cols[ok] // nv
        ijk[:, v_ax] = cols[ok] % nv
        ijk[:, axis] = k[ok]
        touch[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    ks = np.clip(np.ceil(w), 0, nw).astype(np.int64)
    del w
    # Columns that did not close (odd crossings) leak to the top of the grid;
    # count them so a caller can tell a sound mesh from a holed one.
    odd_cols = int((np.bincount(cols, minlength=nu * nv) & 1).sum()) if cols.size else 0
    out = np.zeros((nu, nv, nw), bool)
    # A slab of u-rows at a time keeps the toggle counts small.
    rows = max(1, int(_BATCH // max(nv * (nw + 1), 1)))
    order = np.argsort(cols, kind="stable")
    cols, ks, sg = cols[order], ks[order], sg[order]
    row_of = cols // nv
    overlap = 0
    for r0 in range(0, nu, rows):
        r1 = min(nu, r0 + rows)
        a, b = np.searchsorted(row_of, [r0, r1])
        if a == b:
            continue
        flat = (cols[a:b] - r0 * nv) * (nw + 1) + ks[a:b]
        tog = np.bincount(flat, minlength=(r1 - r0) * nv * (nw + 1)).astype(np.uint8)
        tog = tog.reshape(r1 - r0, nv, nw + 1)[:, :, :nw]
        par = (np.cumsum(tog, axis=2, dtype=np.uint8) & 1).astype(bool)
        out[r0:r1] = par
        wind = np.bincount(flat, weights=sg[a:b], minlength=(r1 - r0) * nv * (nw + 1))
        wind = np.cumsum(wind.reshape(r1 - r0, nv, nw + 1)[:, :, :nw], axis=2)
        overlap += int(((np.abs(wind) > 0.5) & ~par).sum())
    # back to (x, y, z) order
    perm = np.argsort([u_ax, v_ax, axis])
    return np.transpose(out, perm), odd_cols, overlap


def mesh_solid(mesh, pitch, pad=2, bounds=None, axes=None, touch=False):
    """(bool grid, origin) of a triangle mesh's solid material. `bounds`
    (lo, hi) limits the grid (crossings outside it still count, so parity
    stays right). `axes`: scanline axes to vote over; default z alone for a
    watertight mesh, all three (majority) otherwise. Also returns a dict with
    the number of unclosed columns per axis, which is 0 for a sound mesh.

    touch: also count as solid every voxel the surface passes through (found
    from the crossings of all three axes), like a surface voxelization. That
    thickens walls by up to a cell, which is what an isolation check wants:
    sampling only voxel centres, a wall two cells thick lets a diagonal
    26-connected chain of void through and reads as a phantom leak."""
    tri = np.asarray(mesh.triangles, float)
    lo, hi = (np.asarray(b, float) for b in (bounds if bounds is not None else mesh.bounds))
    dims, origin = grid_for(lo, hi, pitch, pad)
    if axes is None:
        axes = (2,) if bool(getattr(mesh, "is_watertight", False)) else (0, 1, 2)
    if bounds is not None:
        # crossings below the grid still toggle parity: extend the grid down
        # each scan axis to the mesh's own minimum, then crop
        mlo = np.asarray(mesh.bounds, float)[0]
        ext = np.maximum(np.ceil((lo - mlo) / pitch).astype(int) + 1, 0)
    else:
        ext = np.zeros(3, int)
    votes = np.zeros(tuple(dims), np.uint8)
    unclosed = {}
    surf = np.zeros(tuple(dims), bool) if touch else None
    overlaps = []
    scan = list(axes) + ([a for a in (0, 1, 2) if a not in axes] if touch else [])
    for ax in scan:
        d2, o2 = dims.copy(), origin.copy()
        d2[ax] += ext[ax]
        o2[ax] -= ext[ax] * pitch
        sl = [slice(None)] * 3
        sl[ax] = slice(int(ext[ax]), None)
        big = np.zeros(tuple(d2), bool) if touch else None
        g, odd, over = _parity_axis(tri, d2, o2, pitch, ax, touch=big)
        if touch:
            surf |= big[tuple(sl)]
        if ax in axes:
            votes += g[tuple(sl)]
            unclosed["xyz"[ax]] = odd
            overlaps.append(over)
    solid = votes * 2 > len(axes)
    if touch:
        solid |= surf
        if pad:
            solid[:pad] = solid[-pad:] = False
            solid[:, :pad] = solid[:, -pad:] = False
            solid[:, :, :pad] = solid[:, :, -pad:] = False
    return solid, origin, {"unclosedColumns": unclosed, "axes": ["xyz"[a] for a in axes],
                           "overlapMm3": float(min(overlaps) * pitch ** 3) if overlaps else 0.0}


def _sample(field, origin, dims, pitch, chunk):
    """field < 0 at the points origin + ijk * pitch, a slab at a time."""
    nx, ny, nz = (int(d) for d in dims)
    out = np.zeros((nx, ny, nz), bool)
    ys = origin[1] + np.arange(ny) * pitch
    zs = origin[2] + np.arange(nz) * pitch
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    plane = np.stack([np.zeros(gy.size), gy.ravel(), gz.ravel()], axis=1)
    rows = max(1, chunk // max(plane.shape[0], 1))
    for i0 in range(0, nx, rows):
        i1 = min(nx, i0 + rows)
        P = np.concatenate([plane + [origin[0] + i * pitch, 0.0, 0.0] for i in range(i0, i1)])
        out[i0:i1] = np.asarray(field(P), float).reshape(i1 - i0, ny, nz) < 0.0
    return out


def field_solid(field, lo, hi, pitch, pad=2, chunk=2_000_000, touch=False):
    """(bool grid, origin) of an implicit solid: field(P) < 0 inside, P an
    (N, 3) array in mm. Evaluated at voxel centres, a slab at a time.

    touch: also count a voxel as solid when any of its eight corners is in
    the solid, the field's version of mesh_solid's touch. Corners, not a
    distance threshold: a field is often not a true distance (smooth unions,
    seal ramps, TPMS sheets read 0.35-2.5 per mm of gradient), and a level
    of pitch/2 shut real channel throats where the gradient is shallow
    (counterflow exchanger_core read each fluid split in two)."""
    dims, origin = grid_for(lo, hi, pitch, pad)
    solid = _sample(field, origin, dims, pitch, chunk)
    if touch:
        corner = _sample(field, origin - 0.5 * pitch, dims + 1, pitch, chunk)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    solid |= corner[dx:dx + dims[0], dy:dy + dims[1], dz:dz + dims[2]]
        del corner
    # the pad is void by construction, whatever the field says out there
    if pad:
        solid[:pad] = solid[-pad:] = False
        solid[:, :pad] = solid[:, -pad:] = False
        solid[:, :, :pad] = solid[:, :, -pad:] = False
    return solid, origin


def solid_grid(part, pitch, pad=2, bounds=None, touch=False):
    """Grid for either a mesh or a field. A field needs `bounds` (lo, hi)."""
    if callable(part) and not hasattr(part, "triangles"):
        if bounds is None:
            raise ValueError("an implicit field needs bounds=(lo, hi) in mm")
        solid, origin = field_solid(part, bounds[0], bounds[1], pitch, pad, touch=touch)
        return solid, origin, {"source": "field"}
    solid, origin, info = mesh_solid(part, pitch, pad, bounds, touch=touch)
    info["source"] = "mesh"
    return solid, origin, info


def _tiles(n, cap):
    """Slabs along axis 0: (core start, core stop, halo start, halo stop)."""
    step = max(int(cap) * 4, 32)
    for a in range(0, n, step):
        b = min(n, a + step)
        yield a, b, max(0, a - cap), min(n, b + cap)


def _tile_len(shape, cap):
    plane = int(shape[1]) * int(shape[2])
    return max(int(_TILE_VOXELS // max(plane, 1)) - 2 * cap, 8)


def wall_thickness(solid, pitch, cap_mm=None, limit_mm=None, spots=5, origin=None):
    """Member thickness of a grid's solid, in mm, on the medial ridge (the
    local maxima of the distance to void). The estimate 2*edt - 0.5 cells is
    within half a cell of the true thickness either way, so a reading carries
    +/- pitch/2 on top of the grid's own +/- pitch/2 staircase.

    cap_mm: thicknesses above this are reported as the cap (the distance
    transform only has to look that far, which is what lets it run in tiles).
    limit_mm: also return the share of the surface lying on a member thinner
    than this, and the thinnest spots (world coordinates when `origin` is
    given). Returns a dict."""
    shape = solid.shape
    cap_mm = float(cap_mm) if cap_mm else float(max(shape) * pitch)
    cap = int(np.ceil(cap_mm / (2.0 * pitch))) + 2  # edt cells to look
    vals_all, where_all = [], []
    thin_surf = 0
    surf_total = 0
    grow = int(np.ceil((limit_mm or 0.0) / (2.0 * pitch))) + 1 if limit_mm else 0
    halo = cap + grow + 2
    step = _tile_len(shape, halo)
    for a in range(0, shape[0], step):
        b = min(shape[0], a + step)
        ha, hb = max(0, a - halo), min(shape[0], b + halo)
        sub = solid[ha:hb]
        # void outside the grid on the cross-section sides; along axis 0 the
        # halo stands in for the neighbours (and the grid's pad is void)
        padded = np.pad(sub, ((0, 0), (1, 1), (1, 1)))
        if ha == 0:
            padded = np.pad(padded, ((1, 0), (0, 0), (0, 0)))
        if hb == shape[0]:
            padded = np.pad(padded, ((0, 1), (0, 0), (0, 0)))
        edt = ndimage.distance_transform_edt(padded).astype(np.float32)
        edt = edt[(1 if ha == 0 else 0):edt.shape[0] - (1 if hb == shape[0] else 0), 1:-1, 1:-1]
        np.minimum(edt, cap, out=edt)
        ridge = sub & (edt >= ndimage.maximum_filter(edt, size=3)) & (edt > 0)
        core = slice(a - ha, b - ha)
        r = ridge[core]
        e = edt[core][r]
        vals_all.append((2.0 * e - 0.5) * pitch)
        if spots and limit_mm:
            ijk = np.argwhere(r)
            ijk[:, 0] += a
            where_all.append(ijk)
        if limit_mm:
            thin = ridge & ((2.0 * edt - 0.5) * pitch < limit_mm)
            near = ndimage.binary_dilation(thin, iterations=grow) if thin.any() else thin
            surf = sub & ~ndimage.binary_erosion(np.pad(sub, 1), border_value=0)[1:-1, 1:-1, 1:-1]
            surf_total += int(surf[core].sum())
            thin_surf += int((surf & near)[core].sum())
    vals = np.concatenate(vals_all) if vals_all else np.zeros(0)
    out = {"pitch": float(pitch), "capMm": round(2 * cap * pitch - 0.5 * pitch, 3), "ridgeCells": int(vals.size)}
    if vals.size == 0:
        out.update(minMm=None, p1Mm=None, p5Mm=None)
        return out
    out.update(minMm=float(vals.min()), p1Mm=float(np.percentile(vals, 1)), p5Mm=float(np.percentile(vals, 5)))
    if limit_mm:
        out["thinSurfaceFraction"] = thin_surf / surf_total if surf_total else 0.0
        out["thinAreaMm2"] = float(thin_surf * pitch * pitch)
        if spots:
            ijk = np.concatenate(where_all)
            order = np.argsort(vals, kind="stable")
            picked, taken = [], []
            sep = max(limit_mm * 4, 8 * pitch)
            for k in order:
                if vals[k] >= limit_mm or len(picked) >= spots:
                    break
                p = ijk[k] * pitch + (origin if origin is not None else 0.0)
                if all(np.linalg.norm(p - q) > sep for q in taken):
                    taken.append(p)
                    picked.append({"at": [round(float(c), 2) for c in p], "mm": round(float(vals[k]), 3)})
            out["thinSpots"] = picked
    return out


def tiled_edt_min(source, targets, cap):
    """Smallest Euclidean distance, in cells, from any `source` cell to any
    `targets` cell, looking at most `cap` cells (None beyond that). Tiled
    along axis 0 so memory stays at one tile of floats."""
    shape = source.shape
    best = None
    halo = int(cap) + 1
    step = _tile_len(shape, halo)
    for a in range(0, shape[0], step):
        b = min(shape[0], a + step)
        if not targets[a:b].any():
            continue
        ha, hb = max(0, a - halo), min(shape[0], b + halo)
        src = source[ha:hb]
        if not src.any():
            continue
        d = ndimage.distance_transform_edt(~src)
        vals = d[a - ha:b - ha][targets[a:b]]
        if vals.size:
            m = float(vals.min())
            if m <= cap and (best is None or m < best):
                best = m
    return best
