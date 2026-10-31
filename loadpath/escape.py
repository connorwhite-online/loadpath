"""
Escape holes for optimized designs: open the voids the optimizer sealed.

Stiffness rewards a closed section. Given a fine enough grid the optimizer
will wrap a skin round an empty core or a keep-out (the 845k-cell arm link
came out a sealed box round its cable channel), and in a powder bed (SLS,
MJF, LPBF) or a resin vat that core is full of powder or liquid resin that
can never come out. drains.py opens the voids of field graphs; this opens
those of a density design, on the optimizer's own grid:

- a void traps powder when no face-connected path of empty cells leads from
  it to the outside of the grid (keep-outs inside the part count as void, so
  a cable channel the skin closed over is one), or when it is a room the
  escape ball fits in but can only leave through gaps narrower than itself
  (trapped_voids);
- a sealed void smaller than the escape hole's own ball is filled solid;
- every other one gets `count` straight holes (default 2, so air or rinse
  can flow through), then more, up to 8, while any of it is farther than
  `maxPath` (60 mm) from a hole, measured through the void;
- a hole runs straight out from the void along one of 26 directions, and is
  sited where the cost is least: the length of wall it cuts, times one plus
  the local strain energy density relative to the part's median (thin, calm
  wall first); never through or next to preserved geometry (bosses, pads);
- the second hole goes as far through the void from the first as the costs
  allow, each later one at the point farthest from every hole;
- holes are mirrored across the study's symmetry planes.

The holes are pinned void and the optimizer runs on from its design (simp.
optimize, opts.escape), so the part re-converges round them; a re-converged
design can seal a new void, so up to three rounds.

The default hole is drains.hole_defaults' (SLS 6 mm, MJF 7.5, SLA/DLP 5.5,
volumetric 5, LPBF 6.8: well over each process's minimum escape hole), 6 mm
without a process.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["EscapeOptions", "drill", "sealed_voids", "trapped_voids"]

_FACE = ndimage.generate_binary_structure(3, 1)
_MAX_HOLES = 8
_SITES = 600          # candidate sites per void (subsampled)
_REACH = 40.0         # mm: the longest drill (a wall thicker than this is not a wall)
_STRESS = 2.0         # weight of the local strain energy in a site's cost


@dataclass
class EscapeOptions:
    diameter: float = 6.0             # mm, drilled
    count: int = 2                    # holes per sealed void
    max_path: float | None = 60.0     # mm through the void to the nearest hole (None: off)
    iterations: int = 20              # optimizer iterations after drilling
    fill_below: float | None = None   # mm^3: fill smaller voids (default the hole's ball)

    @classmethod
    def for_process(cls, process) -> "EscapeOptions":
        if process is None:
            return cls()
        from .drains import hole_defaults

        d, _, per = hole_defaults(process)
        return cls(diameter=float(d), count=max(2, int(per)))

    @classmethod
    def from_spec(cls, spec: dict, process) -> "EscapeOptions":
        o = cls.for_process(process)
        keys = {"diameter": "diameter", "count": "count", "maxPath": "max_path", "iterations": "iterations"}
        for k, v in spec.items():
            if k == "enabled":
                continue
            if k not in keys:
                raise ValueError(f"drains: unknown key {k!r}; known: {sorted(keys) + ['enabled']}")
            setattr(o, keys[k], v)
        if not (float(o.diameter) > 0):
            raise ValueError("drains.diameter is a hole diameter in mm, > 0")
        if int(o.count) < 1:
            raise ValueError("drains.count is at least 1")
        o.diameter, o.count, o.iterations = float(o.diameter), int(o.count), int(o.iterations)
        return o


def _dirs():
    d = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
                 float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def sealed_voids(solid: np.ndarray):
    """(labels, n) of the void components of a padded-by-one bool grid that
    no face path of empty cells joins to its border."""
    void = ~solid
    lbl, n = ndimage.label(void, _FACE)
    border = np.unique(np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(), lbl[:, -1].ravel(),
                                       lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()]))
    out = np.isin(lbl, border[border > 0])
    sl, m = ndimage.label(void & ~out, _FACE)
    return sl, m


def trapped_voids(solid: np.ndarray, radius: float):
    """(labels, n) of the voids powder cannot leave, on a padded-by-one bool
    grid: the sealed ones (sealed_voids), and every room an escape ball of
    `radius` cells fits in but cannot roll out of, because it opens to the
    outside only through gaps narrower than the ball (a box with a few small
    holes in it is still a box full of powder: the 845k-cell arm link's
    were 2-4 mm). The ball's reach is grown from the border through the
    cells it fits in, then back out by its radius."""
    void = ~solid
    fits = ndimage.distance_transform_edt(np.pad(void, 1, constant_values=True))[1:-1, 1:-1, 1:-1] >= radius
    lbl, n = ndimage.label(fits, _FACE)
    border = np.unique(np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(), lbl[:, -1].ravel(),
                                       lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()]))
    free = np.isin(lbl, border[border > 0])
    reach = ndimage.distance_transform_edt(~free) <= radius + 0.5
    sealed, _ = sealed_voids(solid)
    rest, m = ndimage.label(void & ~reach, _FACE)
    rooms = np.unique(rest[fits & ~free])
    keep = (sealed > 0) | np.isin(rest, rooms[rooms > 0])
    return ndimage.label(keep, _FACE)


def _geodesic(mask, starts, step):
    from .heatx import _geodesic as g

    return g(mask, starts, step)


def drill(domain, xphys, energy, o: EscapeOptions, void_prev=None) -> dict:
    """Find the sealed voids of a design and the holes that open them.

    Returns {"sealed": voids found, "holes": [{site, exit, axis, diameter,
    wallMm}], "void": bool per element to pin void, "fill": bool per element
    to fill, "filledVolume": mm^3, "maxPath": mm (the farthest any void is
    from a hole after drilling, or None), "unopened": voids left sealed}."""
    p = float(domain.pitch)
    ijk = domain.mesh.elem_ijk
    n_el = len(ijk)
    dims = np.asarray(domain.dims)
    sh = tuple(dims + 2)
    sel = tuple((ijk + 1).T)
    solid = np.zeros(sh, bool)
    solid[sel] = np.asarray(xphys) > 0.5
    active = np.zeros(sh, bool)
    active[sel] = True
    pres = np.zeros(sh, bool)
    pres[sel] = domain.preserve
    r = 0.5 * o.diameter
    lbl, m = trapped_voids(solid, r / p)
    # Open but deep: powder farther than maxPath, through the void, from any
    # opening (a tube open at one end, 100 mm long). The openings are the
    # empty cells that see the grid's border straight along two axis
    # directions or more.
    base = {}
    n_trapped = m
    if o.max_path is not None:
        void = ~solid & (lbl == 0)
        seen = np.zeros(sh, np.int8)
        for ax in range(3):
            for flip in (False, True):
                sv = np.flip(solid, ax) if flip else solid
                hit = np.logical_or.accumulate(sv, axis=ax)
                seen += np.flip(~hit, ax) if flip else ~hit
        # seen along two directions: down the bore of a tube is only one
        vis = seen >= 2
        inner = void & ~vis
        if inner.any():
            seed = vis & void & ndimage.binary_dilation(inner, _FACE)
            g = _geodesic(inner | seed, [tuple(q) for q in np.argwhere(seed)], p)
            deep, nd = ndimage.label(inner & (g > o.max_path), _FACE)
            for k in range(1, nd + 1):
                lbl[deep == k] = m + k
                base[m + k] = g
            m += nd
    out = {"sealed": int(n_trapped), "deep": int(m - n_trapped), "holes": [], "void": np.zeros(n_el, bool),
           "fill": np.zeros(n_el, bool), "filledVolume": 0.0, "maxPath": None, "unopened": 0}
    if m == 0:
        return out

    ball = math.pi / 6.0 * o.diameter ** 3 if o.fill_below is None else float(o.fill_below)
    sizes = np.bincount(lbl.ravel(), minlength=m + 1) * p ** 3
    fill = np.zeros(sh, bool)
    big = []
    for c in range(1, m + 1):
        cells = lbl == c
        if c not in base and sizes[c] < ball and not (cells & ndimage.binary_dilation(~solid & (lbl != c), _FACE)).any():
            # a small void sealed all round (not a room behind a narrow gap)
            if (cells & ~active).any():
                big.append(c)          # a keep-out: cannot fill, must drill
            else:
                fill |= cells
        else:
            big.append(c)
    out["fill"] = fill[sel]
    out["filledVolume"] = float(fill.sum() * p ** 3)
    if not big:
        return out

    # where a drill may go: never within a hole's radius of preserved geometry
    clear = ndimage.distance_transform_edt(~pres) * p >= r + p
    e = np.zeros(sh)
    if energy is not None:
        e[sel] = np.asarray(energy, float)
        k = max(3, int(round(o.diameter / p)) | 1)
        e = ndimage.uniform_filter(e, size=k, mode="nearest")
        med = float(np.median(e[solid])) if solid.any() else 0.0
        e = e / med if med > 0 else np.zeros(sh)
    outside = ~solid & (lbl == 0)
    dirs = _dirs()
    steps = np.arange(1, int(_REACH / (0.5 * p)) + 1) * 0.5 * p      # mm along the ray
    mirror_axes = ["xyz".index(a) for a in sorted(domain.mirror)] if domain.mirror else []

    def cells_of(pts):
        q = np.floor(pts / p).astype(int)
        inside = np.all((q >= 0) & (q < np.asarray(sh)), axis=-1)
        return np.clip(q, 0, np.asarray(sh) - 1), inside

    def candidates(c):
        """(site cells, directions, costs, wall mm, exit points) for void c."""
        cells = lbl == c
        rim = cells & ndimage.binary_dilation(solid, _FACE)
        idx = np.argwhere(rim)
        if len(idx) > _SITES:
            idx = idx[np.linspace(0, len(idx) - 1, _SITES).astype(int)]
        best = []
        for d in dirs:
            pts = (idx[:, None, :] + 0.5) * p + steps[None, :, None] * d[None, None, :]
            q, inside = cells_of(pts)
            qi = tuple(np.moveaxis(q, -1, 0))
            reached = ~inside | outside[qi]
            first = np.where(reached.any(axis=1), reached.argmax(axis=1), -1)
            ok = first >= 0
            ahead = np.arange(len(steps))[None, :] < np.where(ok, first, 0)[:, None]
            s_solid = solid[qi] & ahead
            wall = s_solid.sum(axis=1) * 0.5 * p
            # straight into the wall and out: not back across the void
            bad = (~clear[qi] & ahead).any(axis=1) | (~solid[qi] & ahead & (steps[None, :] > p)).any(axis=1)
            stress = np.where(s_solid.any(axis=1),
                              (e[qi] * s_solid).sum(axis=1) / np.maximum(s_solid.sum(axis=1), 1), 0.0)
            ok &= ~bad
            cost = np.where(ok, np.maximum(wall, p) * (1.0 + _STRESS * stress), np.inf)
            ex = (idx + 0.5) * p + steps[np.maximum(first, 0)][:, None] * d[None, :]
            best.append((cost, wall, ex))
        cost = np.stack([b[0] for b in best], 1)
        j = cost.argmin(axis=1)
        rows = np.arange(len(idx))
        return (idx, dirs[j], cost[rows, j], np.stack([b[1] for b in best], 1)[rows, j],
                np.stack([b[2] for b in best], 1)[rows, j])

    hole_grid = np.zeros(sh, bool)

    def cylinder(a, b):
        lo = np.maximum(np.floor((np.minimum(a, b) - r - p) / p).astype(int), 0)
        hi = np.minimum(np.ceil((np.maximum(a, b) + r + p) / p).astype(int) + 1, np.asarray(sh))
        g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij"), -1)
        c = (g + 0.5) * p
        ab = b - a
        t = np.clip(((c - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        dist = np.linalg.norm(c - (a + t[..., None] * ab), axis=-1)
        cyl = np.zeros(sh, bool)
        cyl[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = dist <= r + 0.5 * p
        return cyl

    def mirrored(site, exit_):
        """The hole and its images across the symmetry planes (mm, padded
        grid), as (site, exit) pairs. An image that would overlap its hole
        (one sited near the plane) is merged into one hole on the plane."""
        holes = [(np.asarray(site, float), np.asarray(exit_, float))]
        span = (dims + 2) * p
        for ax in mirror_axes:
            nxt = []
            for a_, b_ in holes:
                wa, wb = a_.copy(), b_.copy()
                wa[ax], wb[ax] = span[ax] - wa[ax], span[ax] - wb[ax]
                if abs(wa[ax] - a_[ax]) < 2.0 * r + p:
                    a_, b_ = a_.copy(), b_.copy()
                    a_[ax] = b_[ax] = 0.5 * span[ax]
                    nxt.append((a_, b_))
                else:
                    nxt += [(a_, b_), (wa, wb)]
            holes = nxt
        return holes

    def _reach(comp, have, g0):
        """mm through the void from the nearest hole (or, for an open room,
        opening), on the room's cells."""
        geo = np.full(sh, np.inf) if g0 is None else np.where(comp, g0, np.inf)
        if have:
            starts = [tuple(np.clip(np.floor(s / p).astype(int), 0, np.asarray(sh) - 1)) for s in have]
            geo = np.minimum(geo, _geodesic(comp, starts, p))
        return geo

    mouths = {c: [] for c in big}
    worst = 0.0
    unopened = 0
    for c in big:
        cand = None
        for _ in range(_MAX_HOLES):
            comp = lbl == c
            have = mouths[c]
            far = None
            if have or c in base:
                geo = _reach(comp, have, base.get(c))
                gmax = float(np.max(geo[comp & np.isfinite(geo)], initial=0.0))
                # an open room needs no count of holes, only the reach
                if (c in base or len(have) >= o.count) and (o.max_path is None or gmax <= o.max_path):
                    break
                far = (geo, gmax)
            if cand is None:
                cand = candidates(c)
            idx, dvec, cost, wall, ex = cand
            score = cost.copy()
            if far is not None:
                geo, gmax = far
                g = geo[tuple(idx.T)]
                g = np.where(np.isfinite(g), g, gmax)
                # the farthest reach of the void from every hole: sites there only
                score = np.where(g >= 0.8 * gmax, score, np.inf)
            k = int(np.argmin(score))
            if not np.isfinite(score[k]):
                if not have and c not in base:
                    unopened += 1
                break
            site = (idx[k] + 0.5) * p
            exit_ = ex[k] + dvec[k] * p
            for s_img, x_img in mirrored(site, exit_):
                a = s_img - (x_img - s_img) / max(np.linalg.norm(x_img - s_img), 1e-9) * r
                hole_grid |= cylinder(a, x_img)
                cq = tuple(np.clip(np.floor(s_img / p).astype(int), 0, np.asarray(sh) - 1))
                ci = int(lbl[cq])
                if ci in mouths:
                    mouths[ci].append(s_img)
                out["holes"].append({
                    "site": [float(v) for v in (s_img - p + domain.origin)],
                    "exit": [float(v) for v in (x_img - p + domain.origin)],
                    "axis": [float(v) for v in (x_img - s_img) / max(np.linalg.norm(x_img - s_img), 1e-9)],
                    "diameter": o.diameter, "wallMm": float(wall[k])})
    for c, have in mouths.items():
        if have or c in base:
            comp = lbl == c
            geo = _reach(comp, have, base.get(c))
            worst = max(worst, float(np.max(geo[comp & np.isfinite(geo)], initial=0.0)))
    hole_grid &= active & ~pres
    if void_prev is not None:
        hole_grid[sel] &= ~void_prev
    out["void"] = hole_grid[sel]
    out["maxPath"] = worst if any(mouths.values()) or base else None
    out["unopened"] = unopened
    return out
