"""
Free space from a scan's rays: what the sensor saw empty.

A scanner measures a point by looking along a ray from where it stood to
the surface, so everything along that ray short of the point was empty
when it looked. The union of those rays is the space a scan *shows*; the
rest is solid or was never seen, and a part designed in the scan stays out
of both. That is the honest keep-out for a space seen from inside (a bay, a
cavity, under a desk), which no reconstruction closes: there the surface
does not bound an object, it bounds the room.

`carve(scan, lo, hi)` marks the nodes of a grid (pitch p) over the box:

- **crossed**: a ray from a point's sensor to the point passes through the
  node's cell (sampled every half pitch). A crossed node is within half a
  cell diagonal (0.87 p) of a point the sensor saw empty, so it may itself
  sit up to 0.87 p inside a surface the ray grazed without measuring it
  (the far side of a pipe at the tangent, where points stop because the
  angle got too steep). So:
- **deep in what was crossed**: every node within two steps of it (the
  5 x 5 x 5 block round it) was crossed. One step off a surface in the
  best of the 26 directions goes at least p deeper, so a node two steps
  from one more than 0.87 p inside is at least 1.13 p out: the keep-out's
  surface, drawn between it and the next node (keepout_mesh), stays
  outside a surface nobody measured. Before that, small gaps in the
  crossed nodes (at most 8 nodes, every side crossed: the space between
  two rays that both missed a node) count as crossed; something that small
  that no ray hit and no point measured is the limit below;
- **clear of what was measured**: the node is at least U + p + h/2 from the
  nearest measured point, where U is that point's one-sided uncertainty at
  the scan's confidence (Scan.uncertainty), h its spacing (the real surface
  between points can sit up to about half a spacing nearer than the nearest
  point) and p the pitch (the keep-out's surface sits up to a pitch from a
  free node, between it and the next).

A node is free when all three hold. What it can't know: a thing smaller
than about two pitches that no ray hit and no point measured (a wire
across the bay seen edge on). Scan such things or give them as keep-outs.

Needs each point's sensor position (Scan.sensors: PLY sx sy sz, E57 poses,
the Imprint format's rays).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["carve", "keepout_mesh"]


def carve(scan, lo, hi, *, pitch: float | None = None, chunk: int = 4000) -> dict:
    """The free nodes of a grid over [lo, hi]. Returns {"free" (bool grid),
    "origin" (node 0's position), "pitch", "seen" (crossed by a ray, gaps between rays filled),
    "report"}."""
    if scan.sensors is None:
        raise ValueError("free space needs each point's sensor position (PLY sx sy sz, E57 poses, an "
                         ".imprint's rays): this scan has none")
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    h = scan.spacing
    p = float(pitch) if pitch else max(0.5, float(np.median(h)))
    dims = np.floor((hi - lo) / p).astype(int) + 1
    if int(np.prod(dims)) > 60_000_000:
        raise ValueError(f"a {dims.tolist()} grid at {p:g} mm: give a coarser pitch or a smaller box")
    crossed = np.zeros(dims, bool)
    P, S = scan.points, scan.sensors
    step = 0.5 * p
    for k in range(0, len(P), chunk):
        a, b = S[k:k + chunk], P[k:k + chunk]
        d = b - a
        L = np.linalg.norm(d, axis=1)
        u = d / np.maximum(L, 1e-12)[:, None]
        # the stretch of each ray inside the box (slab clipping)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - a) / u
            t2 = (hi - a) / u
        tmin = np.nanmax(np.where(np.isfinite(np.minimum(t1, t2)), np.minimum(t1, t2), -np.inf), 1)
        tmax = np.nanmin(np.where(np.isfinite(np.maximum(t1, t2)), np.maximum(t1, t2), np.inf), 1)
        t0 = np.maximum(tmin, 0.0)
        t_end = np.minimum(tmax, L)
        ok = t_end > t0
        if not ok.any():
            continue
        a, u, t0, t_end = a[ok], u[ok], t0[ok], t_end[ok]
        n = np.ceil((t_end - t0) / step).astype(int) + 1
        # every sample along every ray, flattened
        ray = np.repeat(np.arange(len(n)), n)
        off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        t = np.minimum(t0[ray] + off * step, t_end[ray])
        X = a[ray] + u[ray] * t[:, None]
        ijk = np.round((X - lo) / p).astype(int)
        m = np.all((ijk >= 0) & (ijk < dims), 1)
        ijk = ijk[m]
        crossed[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    # gaps between rays: small pockets of uncrossed nodes walled in by crossed ones
    from scipy import ndimage
    lab, n_lab = ndimage.label(~crossed)
    size = np.bincount(lab.ravel(), minlength=n_lab + 1)
    edge = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                     lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    small = size <= 8
    small[0] = False
    small[edge] = False
    gaps = small[lab]
    seen = crossed | gaps
    # two steps in from the edge of what was crossed (the box's own walls are not an edge)
    deep = ndimage.binary_erosion(seen, structure=np.ones((3, 3, 3), bool), iterations=2, border_value=1)
    # clear of every measured point by its uncertainty
    idx = np.argwhere(deep)
    X = lo + idx * p
    dist, near = scan.nearest(X)
    U = scan.uncertainty(idx=near)
    need = U + p + 0.5 * h[near]
    clear = dist >= need
    free = np.zeros(dims, bool)
    free[tuple(idx[clear].T)] = True
    rep = {"pitch": round(p, 4), "nodes": int(np.prod(dims)), "crossed": int(crossed.sum()),
           "gapsFilled": int(gaps.sum()), "deep": int(deep.sum()),
           "free": int(free.sum()), "freeShare": round(float(free.mean()), 4),
           "clearance": {"median": round(float(np.median(need)), 4) if len(need) else None,
                         "rule": "two nodes in from what the rays crossed, and U + pitch + spacing/2 from the "
                                   "nearest measured point"}}
    return {"free": free, "seen": seen, "origin": lo, "pitch": p, "report": rep}


def keepout_mesh(fs: dict):
    """Everything in the box that is not free, as a closed mesh.

    The kept-out nodes are padded with one more kept-out layer (so the
    keep-out runs to the box's walls) and then one free layer (so the
    surface closes round the outside of the box), lightly smoothed (half a
    pitch: marching cubes on a raw 0/1 field leaves non-manifold edges where
    two kept-out nodes touch only at an edge) and contoured at 0.3, below
    half, so the surface stands outside every kept-out node: the smoothing
    may round a corner of the keep-out outward, never inward."""
    import trimesh
    from scipy import ndimage
    from skimage.measure import marching_cubes
    free = fs["free"]
    p = fs["pitch"]
    lo = np.asarray(fs["origin"], float)
    if not free.any():
        return trimesh.creation.box(bounds=[lo, lo + (np.array(free.shape) - 1) * p])
    solid = np.pad(np.pad((~free).astype(np.float32), 1, constant_values=1.0), 2, constant_values=0.0)
    for sigma in (0.5, 0.6, 0.75):
        field = ndimage.gaussian_filter(solid, sigma, mode="constant")
        verts, faces, _, _ = marching_cubes(field, level=0.3, spacing=(p, p, p))
        mesh = trimesh.Trimesh(verts + (lo - 3 * p), faces, process=True)
        if mesh.is_watertight:
            break
    else:
        raise RuntimeError("the keep-out from the free space did not close: give a different pitch")
    if mesh.volume < 0:
        mesh.invert()
    return mesh
