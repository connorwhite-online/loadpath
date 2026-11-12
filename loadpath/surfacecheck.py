"""
Has the designer's exterior been kept?

A product's outside is the designer's. Whatever is done inside it, or cut
through it on purpose (an antenna split, a port), the surface that comes
out has to be the surface that went in. `compare_exterior` measures that
the way a person checks it: the original and the new assembly are looked
at from the six orthographic directions, the depth of the first surface is
read at every pixel of each view, and the two are compared point by point.
Every point of new surface a view sees is measured to the original, and
every point of original surface to the new assembly: surface added or
moved, and surface gone missing, by more than the tolerance is a
deviation; deviations inside a declared cut are counted apart. `ortho_sheet` draws the same six views of both, side by
side, with the difference under them: the sheet to look at before any
part is sent anywhere.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange
except Exception:  # noqa: BLE001
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
    prange = range

# name, the axis looked along, its sign (+1: looking toward -axis from +), the image's right and up axes
VIEWS = (("top", 2, +1, 0, 1), ("bottom", 2, -1, 0, 1), ("front", 1, -1, 0, 2), ("back", 1, +1, 0, 2),
         ("right", 0, +1, 1, 2), ("left", 0, -1, 1, 2))


@njit(cache=True)
def _zbuffer(tri, nu, nv, u0, v0, px, depth):
    """Nearest depth per pixel (larger is nearer the eye)."""
    for t in range(tri.shape[0]):
        ax, ay, az = tri[t, 0, 0], tri[t, 0, 1], tri[t, 0, 2]
        bx, by, bz = tri[t, 1, 0], tri[t, 1, 1], tri[t, 1, 2]
        cx, cy, cz = tri[t, 2, 0], tri[t, 2, 1], tri[t, 2, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-14:
            continue
        i0 = max(0, int(math.floor((min(ax, bx, cx) - u0) / px)))
        i1 = min(nu - 1, int(math.ceil((max(ax, bx, cx) - u0) / px)))
        j0 = max(0, int(math.floor((min(ay, by, cy) - v0) / px)))
        j1 = min(nv - 1, int(math.ceil((max(ay, by, cy) - v0) / px)))
        for i in range(i0, i1 + 1):
            x = u0 + (i + 0.5) * px
            for j in range(j0, j1 + 1):
                y = v0 + (j + 0.5) * px
                l0 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
                l1 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
                l2 = 1.0 - l0 - l1
                if l0 < -1e-9 or l1 < -1e-9 or l2 < -1e-9:
                    continue
                z = l0 * az + l1 * bz + l2 * cz
                if z > depth[i, j]:
                    depth[i, j] = z
    return depth


@njit(cache=True)
def _pt_tri(p, a, b, c):
    """Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return ap @ ap
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return bp @ bp
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        q = a + ab * (d1 / (d1 - d3)) - p
        return q @ q
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return cp @ cp
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        q = a + ac * (d2 / (d2 - d6)) - p
        return q @ q
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))) - p
        return q @ q
    den = 1.0 / (va + vb + vc)
    q = a + ab * (vb * den) + ac * (vc * den) - p
    return q @ q


@njit(cache=True, parallel=True)
def _grid_dist(P, T, start, items, org, h, n, cap):
    out = np.empty(P.shape[0])
    rmax = int(math.ceil(cap / h)) + 1
    for i in prange(P.shape[0]):
        p = P[i]
        cx = int(math.floor((p[0] - org[0]) / h))
        cy = int(math.floor((p[1] - org[1]) / h))
        cz = int(math.floor((p[2] - org[2]) / h))
        best = cap * cap
        for r in range(rmax + 1):
            if r > 0 and best <= ((r - 1) * h) ** 2:
                break                                # nothing in a farther ring can be nearer
            for x in range(cx - r, cx + r + 1):
                if x < 0 or x >= n[0]:
                    continue
                for y in range(cy - r, cy + r + 1):
                    if y < 0 or y >= n[1]:
                        continue
                    for z in range(cz - r, cz + r + 1):
                        if z < 0 or z >= n[2]:
                            continue
                        if max(abs(x - cx), abs(y - cy), abs(z - cz)) != r:
                            continue                 # the ring's shell only
                        c = (x * n[1] + y) * n[2] + z
                        for k in range(start[c], start[c + 1]):
                            t = items[k]
                            d = _pt_tri(p, T[t, 0], T[t, 1], T[t, 2])
                            if d < best:
                                best = d
        out[i] = math.sqrt(best)
    return out


class _Exact:
    """Exact distance from points to a mesh's surface, capped at `cap`
    (triangles binned on a grid; rings of cells searched outward)."""

    def __init__(self, mesh, cap):
        T = np.ascontiguousarray(mesh.triangles, float)
        # a triangle with no area (booleans leave them) has no surface, and divides by zero in the distance
        area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        T = np.ascontiguousarray(T[area > 1e-12 * max(float(area.max()), 1e-30)])
        lo, hi = T.min(axis=(0, 1)) - 1e-6, T.max(axis=(0, 1)) + 1e-6
        h = max(float(np.max(hi - lo)) / 160.0, 0.25)
        n = np.maximum(np.ceil((hi - lo) / h).astype(np.int64), 1)
        i0 = np.floor((T.min(axis=1) - lo) / h).astype(np.int64)
        i1 = np.floor((T.max(axis=1) - lo) / h).astype(np.int64)
        cells, tris = [], []
        span = i1 - i0 + 1
        for sx in range(int(span[:, 0].max())):
            for sy in range(int(span[:, 1].max())):
                for sz in range(int(span[:, 2].max())):
                    k = (span[:, 0] > sx) & (span[:, 1] > sy) & (span[:, 2] > sz)
                    if k.any():
                        c = i0[k] + [sx, sy, sz]
                        cells.append((c[:, 0] * n[1] + c[:, 1]) * n[2] + c[:, 2])
                        tris.append(np.nonzero(k)[0])
        cells, tris = np.concatenate(cells), np.concatenate(tris)
        o = np.argsort(cells, kind="stable")
        self.items = tris[o].astype(np.int64)
        self.start = np.searchsorted(cells[o], np.arange(int(n.prod()) + 1)).astype(np.int64)
        self.T, self.org, self.h, self.n, self.cap = T, lo, h, n, float(cap)

    def __call__(self, P):
        return _grid_dist(np.ascontiguousarray(P, float), self.T, self.start, self.items, self.org, self.h,
                          self.n, self.cap)


def _meshes(m):
    import trimesh
    if isinstance(m, (str, bytes)):
        m = trimesh.load(m, force="mesh")
    if isinstance(m, (list, tuple)):
        m = trimesh.util.concatenate([_meshes(q) for q in m])
    return m


def depth_maps(mesh, bounds, pixel):
    """{view: depth map} of a mesh over `bounds` ((lo, hi)), NaN where the
    view sees nothing. Depth is the coordinate along the view axis, signed
    so that larger is nearer the eye."""
    mesh = _meshes(mesh)
    lo, hi = (np.asarray(b, float) for b in bounds)
    T = np.asarray(mesh.triangles, float)
    out = {}
    for name, ax, sign, r, u in VIEWS:
        nu = int(math.ceil((hi[r] - lo[r]) / pixel))
        nv = int(math.ceil((hi[u] - lo[u]) / pixel))
        tri = np.stack([T[:, :, r], T[:, :, u], sign * T[:, :, ax]], axis=2)
        d = np.full((nu, nv), -np.inf)
        _zbuffer(np.ascontiguousarray(tri), nu, nv, float(lo[r]), float(lo[u]), float(pixel), d)
        d[~np.isfinite(d)] = np.nan
        out[name] = d
    return out


def compare_exterior(original, new, tolerance=0.05, pixel=None, cuts=(), scale=1.0):
    """The new assembly's outside against the original exterior.

    original, new: a mesh, a path, or a list of them (the parts as
    assembled). tolerance: mm a surface may stand in or out. cuts: boxes
    ((lo, hi), ...) where a deliberate change is declared (a split, a
    port); deviations there are reported, not failed. scale: the new form
    is the original scaled by this about the origin (a smaller device):
    the original is scaled before the comparison.

    Returns {"pass", "maxDeviation", "views": {name: {"maxDeviation",
    "addedOrMoved", "missing", "inCuts"}}, "maps": {name: + the distance of
    new surface from the original, - the distance of original surface the
    new assembly no longer has}} in mm and mm^2 of each view."""
    a, b = _meshes(original).copy(), _meshes(new)
    if scale != 1.0:
        a.apply_scale(scale)
    lo = np.minimum(a.bounds[0], b.bounds[0])
    hi = np.maximum(a.bounds[1], b.bounds[1])
    span = float(np.max(hi - lo))
    px = float(pixel) if pixel else max(span / 1200.0, 0.02)
    lo, hi = lo - 2 * px, hi + 2 * px
    da, db = depth_maps(a, (lo, hi), px), depth_maps(b, (lo, hi), px)
    cap = max(20 * tolerance, 2.0)                       # farther than this reads as "at least cap"
    dist_a, dist_b = _Exact(a, cap), _Exact(b, cap)
    views, maps, worst, ok = {}, {}, 0.0, True
    for name, ax, sign, r, u in VIEWS:
        A, B = da[name], db[name]
        uu = lo[r] + (np.arange(A.shape[0]) + 0.5) * px
        vv = lo[u] + (np.arange(A.shape[1]) + 0.5) * px

        def points(D):
            i, j = np.nonzero(np.isfinite(D))
            P = np.zeros((len(i), 3))
            P[:, r], P[:, u], P[:, ax] = uu[i], vv[j], sign * D[i, j]
            return i, j, P
        # the new surface this view sees, measured to the original; the original surface it sees,
        # measured to the new one (what was added or moved; what went missing)
        ib, jb, Pb = points(B)
        ia, ja, Pa = points(A)
        added = np.zeros(A.shape)
        missing = np.zeros(A.shape)
        if len(Pb):
            added[ib, jb] = dist_a(Pb)
        if len(Pa):
            missing[ia, ja] = dist_b(Pa)
        # a depth pixel sits up to half a pixel off the surface it was read from on a steep wall
        slack = tolerance
        bad_add, bad_miss = added > slack, missing > slack
        in_ray = np.zeros(A.shape, bool)                 # the view's ray passes through a declared cut
        in_box = np.zeros(A.shape, bool)                 # the original point lies inside one
        for clo, chi in cuts:
            col = ((uu[:, None] >= clo[r]) & (uu[:, None] <= chi[r])
                   & (vv[None, :] >= clo[u]) & (vv[None, :] <= chi[u]))
            in_ray |= col
            z = sign * np.where(np.isfinite(A), A, np.inf)
            in_box |= col & (z >= clo[ax]) & (z <= chi[ax])
        fail = (bad_add & ~in_ray) | (bad_miss & ~in_box)
        cut_px = (bad_add & in_ray) | (bad_miss & in_box)
        dev = np.maximum(np.where(bad_add & ~in_ray, added, 0.0), np.where(bad_miss & ~in_box, missing, 0.0))
        m = float(dev.max()) if fail.any() else 0.0
        views[name] = {"maxDeviation": round(m, 3),
                       "addedOrMoved": round(float((bad_add & ~in_ray).sum()) * px * px, 2),
                       "missing": round(float((bad_miss & ~in_box).sum()) * px * px, 2),
                       "inCuts": round(float(cut_px.sum()) * px * px, 2)}
        maps[name] = np.where(bad_add, added, 0.0) - np.where(bad_miss & ~bad_add, missing, 0.0)
        worst = max(worst, m)
        ok = ok and not fail.any()
    return {"pass": bool(ok), "tolerance": tolerance, "pixel": round(px, 4), "maxDeviation": round(worst, 3),
            "views": views, "maps": maps, "depth": {"original": da, "new": db}}


def _shade(d, px):
    """A depth map lit from the upper left: grey where there is surface."""
    z = np.where(np.isfinite(d), d, np.nanmin(d) if np.isfinite(d).any() else 0.0)
    gx, gy = np.gradient(z, px)
    n = np.stack([-gx, -gy, np.ones_like(z)], axis=2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.35, 0.45, 0.82])
    s = np.clip(n @ (light / np.linalg.norm(light)), 0, 1)
    img = 0.25 + 0.7 * s
    return np.where(np.isfinite(d), img, 1.0)


def ortho_sheet(original, new, path, tolerance=0.05, cuts=(), scale=1.0, title=None, result=None,
                highlight=None, highlight_label=None):
    """Six orthographic views of the original and the new assembly side by
    side, with the difference under each pair (red: new surface proud or
    added outline; blue: sunk or missing). Returns compare_exterior's
    result. highlight: the parts of the new assembly to tint (a metal band
    let into a shell changes no surface, and still has to be seen)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = result or compare_exterior(original, new, tolerance, cuts=cuts, scale=scale)
    px = res["pixel"]
    hl = None
    if highlight is not None:
        d0 = res["depth"]["new"]["top"]
        b = _meshes(new).bounds
        lo = np.minimum(_meshes(original).bounds[0] * scale, b[0]) - 2 * px
        hi = lo + np.array([d0.shape[0], d0.shape[1], res["depth"]["new"]["front"].shape[1]]) * px
        hl = depth_maps(highlight, (lo, hi), px)
    fig, axs = plt.subplots(3, 6, figsize=(24, 12.5), dpi=110)
    for k, (name, *_r) in enumerate(VIEWS):
        flip = name in ("bottom", "front", "left")       # seen from the other side: mirror left-right
        for row, key in ((0, "original"), (1, "new")):
            img = _shade(res["depth"][key][name], px).T[::-1]
            img = img[:, ::-1] if flip else img
            axs[row, k].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            if row == 1 and hl is not None:
                on = (np.abs(hl[name] - res["depth"]["new"][name]) < 1e-3).T[::-1]
                on = on[:, ::-1] if flip else on
                tint = np.zeros(on.shape + (4,))
                tint[on] = (0.95, 0.62, 0.15, 0.55)
                axs[row, k].imshow(tint, interpolation="nearest")
            axs[row, k].set_title(f"{name}: {key}", fontsize=11)
        m = res["maps"][name]
        lim = max(4 * tolerance, 0.2)
        mm = np.clip(m, -lim, lim).T[::-1]
        base = _shade(res["depth"]["original"][name], px).T[::-1]
        if flip:
            mm, base = mm[:, ::-1], base[:, ::-1]
        axs[2, k].imshow(base, cmap="gray", vmin=0, vmax=1, alpha=0.35)
        axs[2, k].imshow(np.ma.masked_where(mm == 0, mm), cmap="bwr", vmin=-lim, vmax=lim, interpolation="nearest")
        v = res["views"][name]
        axs[2, k].set_title(f"difference: max {v['maxDeviation']} mm, {v['addedOrMoved'] + v['missing']:.1f} mm2"
                            + (f" (+{v['inCuts']:.1f} in cuts)" if v["inCuts"] else ""), fontsize=10)
    for a in axs.ravel():
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    verdict = "exterior kept" if res["pass"] else "EXTERIOR CHANGED"
    if hl is not None and highlight_label:
        verdict += f"; tinted: {highlight_label}"
    fig.suptitle(f"{title + ': ' if title else ''}{verdict} (tolerance {tolerance} mm, max deviation "
                 f"{res['maxDeviation']} mm)", fontsize=15, color="black" if res["pass"] else "#b00020")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return res
