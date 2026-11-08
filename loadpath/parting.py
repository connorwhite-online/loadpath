"""
Parting lines that are not flat, proposed (shells.py's search).

A flat cut across an axis is the first thing anyone tries, and often it is
the wrong one: the line runs across a face the designer cares about, or
through a connector that then has to be threaded in, or along a wall where
there is no straight run for a snap. Designers draw parting lines that
tilt with the product, step over a part, and ride a crease or the widest
line of the form, where a seam reads as part of the shape. This module
proposes those, for each direction the cover could lift off along, and
`shells._search` checks and ranks them with the flat cuts:

- **tilted planes**: through the best flat levels, tilted 8 and 15 degrees
  about each axis across the lift;
- **stepped lines**: a flat level that climbs over (or dips under) each
  component it would cut through, and each `avoid` region it would cross,
  so the part sits whole in one half;
- **the silhouette**: the widest line of the form seen along the lift (the
  parting line of a mould: neither half has an undercut). Along a straight
  wall it keeps to the middle of the wall, and it rises and falls where a
  curved form makes it;
- **creases**: a line that follows a kink in the exterior's profile all
  the way round (a character line, a shoulder, a reveal). A seam in a
  crease reads as part of the form.

Each comes out as a shells._Cut, so the checks, the joints and the halves
are the same as for a cut given by hand. Seam visibility is scored here
too (`seam_visibility`): seam on a crease, in an opening or on a face the
designer marked `hidden` is counted at a quarter of its length.
"""
from __future__ import annotations

import math

import numpy as np

TILTS = (8.0, 15.0)            # degrees a tilted plane leans off the flat cut
CREASE_DEG = 20.0              # a profile kink this sharp over +-1 mm is a crease
HIDDEN_WEIGHT = 0.25           # a hidden millimetre of seam counts this much


def _profile(ext, cut, n_ang: int = 120, dw: float = 0.25):
    """How far out the exterior reaches at each angle round the part's
    middle (in the plane across the lift) and each height along the lift:
    (angles, heights, R (n_ang, n_heights) mm, NaN off the part, middle
    (u, v)). Each ray comes in from outside, so the first surface it meets
    is the outermost."""
    P = __import__("loadpath.shells", fromlist=["_inside"])._inside(ext)
    u, v = cut.uv(P)
    c0 = np.array([0.5 * (u.min() + u.max()), 0.5 * (v.min() + v.max())])
    w = P @ cut.n
    st = float(np.max(np.subtract(*ext.bounds()[::-1]))) / 80.0
    W = np.arange(w.min() - st, w.max() + st + dw, dw)
    (u0, u1), (v0, v1), _ = cut.footprint(ext, pad=1.0)
    R = float(np.hypot(max(u1 - c0[0], c0[0] - u0), max(v1 - c0[1], c0[1] - v0))) + 1.0
    th = np.linspace(0.0, 2 * np.pi, n_ang, endpoint=False)
    rs = np.arange(R, 0.0, -0.5)
    mid = c0[0] * cut.e1 + c0[1] * cut.e2
    base = mid[None, :] + W[:, None] * cut.n[None, :]
    out = np.full((n_ang, len(W)), np.nan)
    for k, a in enumerate(th):
        d = math.cos(a) * cut.e1 + math.sin(a) * cut.e2
        pts = base[:, None, :] + rs[None, :, None] * d[None, None, :]
        f = ext.sdf(pts.reshape(-1, 3)).reshape(len(W), len(rs))
        hit = f <= 0
        got = hit.any(axis=1)
        i = np.argmax(hit, axis=1)
        r_in = rs[i]
        r_out = np.where(i > 0, rs[np.maximum(i - 1, 0)], rs[i])
        for _ in range(7):
            m = 0.5 * (r_in + r_out)
            inside = ext.sdf(base + m[:, None] * d[None, :]) <= 0
            r_in = np.where(inside, m, r_in)
            r_out = np.where(inside, r_out, m)
        out[k, got] = 0.5 * (r_in + r_out)[got]
    return th, W, out, c0


def _points(cut, c0, th, W, R, wk):
    """World points on the exterior at each angle's chosen height wk (NaN:
    skipped)."""
    pts = []
    for k, a in enumerate(th):
        if not np.isfinite(wk[k]):
            continue
        r = float(np.interp(wk[k], W, np.nan_to_num(R[k], nan=0.0)))
        if r <= 0:
            continue
        d = math.cos(a) * cut.e1 + math.sin(a) * cut.e2
        pts.append(c0[0] * cut.e1 + c0[1] * cut.e2 + r * d + wk[k] * cut.n)
    return np.array(pts)


def _taut(lo, hi, iters=200):
    """A closed line w(angle) inside [lo, hi] at each angle: drawn to the
    middle of the band and to its neighbours' average, clipped back into
    the band. A straight wall's band is wide and the line keeps level
    across it; a curved one is narrow and the line follows it."""
    mid = 0.5 * (lo + hi)
    w = mid.copy()
    for _ in range(iters):
        w = np.clip(0.5 * (0.5 * (np.roll(w, 1) + np.roll(w, -1))) + 0.5 * mid, lo, hi)
    return w


def _smooth_closed(w, passes=3):
    """A closed line's heights, NaN gaps kept, smoothed angle to angle: a
    crease read on a 0.25 mm grid jitters a step either way, and a snap
    needs the seam level across its width."""
    w = np.asarray(w, float).copy()
    ok = np.isfinite(w)
    for _ in range(passes):
        a, b = np.roll(w, 1), np.roll(w, -1)
        st = np.stack([a, w, b])
        med = np.nanmedian(np.where(np.isfinite(st), st, np.nan), axis=0)
        w = np.where(ok, med, np.nan)
    for _ in range(passes):
        a, b = np.roll(w, 1), np.roll(w, -1)
        num = np.nansum(np.stack([a, 2 * w, b]), axis=0)
        den = np.isfinite(a) + 2 * np.isfinite(w) + np.isfinite(b)
        w = np.where(ok, num / np.maximum(den, 1), np.nan)
    return w


def silhouette_line(ext, cut, prof=None, tol=0.15, min_depth=6.0):
    """The widest line round the part seen along the lift: at each angle,
    the band of heights within `tol` mm of the widest; through it, as level
    as it can run. None where the line is flat anyway (an axis cut covers
    it) or does not go all the way round."""
    th, W, R, c0 = prof if prof is not None else _profile(ext, cut)
    Rn = np.nan_to_num(R, nan=-1.0)
    rmax = Rn.max(axis=1)
    ok = rmax > 0
    if ok.mean() < 0.9:
        return None
    wmin, wmax = W[0] + min_depth, W[-1] - min_depth
    lo = np.full(len(th), np.nan)
    hi = np.full(len(th), np.nan)
    for k in range(len(th)):
        band = np.where((Rn[k] >= rmax[k] - tol) & (W >= wmin) & (W <= wmax))[0]
        if len(band) == 0:
            return None
        lo[k], hi[k] = W[band[0]], W[band[-1]]
    w = _taut(lo, hi)
    if float(np.ptp(w)) < 0.5:
        return None
    return _points(cut, c0, th, W, R, w), w


def crease_lines(ext, cut, prof=None, min_depth=6.0, deg=CREASE_DEG, max_lines=3):
    """Lines that follow a kink in the exterior's profile all the way round
    (a character line, a shoulder, a reveal): at each angle, heights where
    the profile's slope turns by more than `deg` over +-1 mm; linked angle
    to angle where they continue within 1.5 mm. Each comes back as (world
    points, heights per angle)."""
    th, W, R, c0 = prof if prof is not None else _profile(ext, cut)
    dw = float(W[1] - W[0])
    m = max(1, round(1.0 / dw))
    n_ang = len(th)
    peaks = []

    def kinks(r, good, m_):
        out = np.zeros(len(W))
        for j in range(m_, len(W) - m_):
            if good[j - m_] and good[j] and good[j + m_]:
                s_up = (r[j + m_] - r[j]) / (m_ * dw)
                s_dn = (r[j] - r[j - m_]) / (m_ * dw)
                out[j] = abs(math.degrees(math.atan(s_up) - math.atan(s_dn)))
        return out

    for k in range(n_ang):
        r = R[k]
        good = np.isfinite(r)
        idx = np.where(good)[0]
        if len(idx) == 0:
            peaks.append([])
            continue
        # not the ends of the profile (the top and bottom faces turn there)
        w_lo, w_hi = W[idx[0]] + max(min_depth, 1.5), W[idx[-1]] - max(min_depth, 1.5)
        coarse = kinks(r, good, m)          # over +-1 mm: a crease's strength
        fine = kinks(r, good, 1)            # over one step: where exactly it is
        above = coarse >= deg
        pk = []
        j = 0
        while j < len(W):
            if not above[j]:
                j += 1
                continue
            j1 = j
            while j1 + 1 < len(W) and above[j1 + 1]:
                j1 += 1
            # a sharp step reads over +-1 mm as a plateau 2 mm wide: the
            # crease is where it turns hardest at the finest scale (the
            # middle of a tie)
            seg = fine[j:j1 + 1]
            best = np.where(seg >= seg.max() - 1e-6)[0]
            jj = j + round(float(best.mean()))
            if w_lo <= W[jj] <= w_hi:
                w_ = float(W[j] + dw * best.mean())
                # a step (a reveal, a shoulder): the seam goes just onto the
                # face that stands in, not along the ledge, where it would
                # wander across it
                r_dn, r_up = r[max(jj - m, 0)], r[min(jj + m, len(W) - 1)]
                if np.isfinite(r_dn) and np.isfinite(r_up) and abs(r_up - r_dn) > 0.2:
                    w_ += 0.6 * dw if r_up < r_dn else -0.6 * dw
                pk.append((w_, float(coarse[j:j1 + 1].max())))
            j = j1 + 1
        peaks.append(pk)
    lines = []
    starts = sorted({round(w_, 1) for p in peaks for w_, _ in p})
    for w0 in starts:
        wk = np.full(n_ang, np.nan)
        cur, gaps = w0, 0
        for k in range(n_ang):
            near = [w_ for w_, _ in peaks[k] if abs(w_ - cur) <= 1.5]
            if near:
                cur = min(near, key=lambda x: abs(x - cur))
                wk[k] = cur
            else:
                gaps += 1
        if gaps > 0.3 * n_ang or abs(cur - w0) > 1.5:
            continue
        if any(np.nanmax(np.abs(wk - l_[1])) < 0.6 for l_ in lines):
            continue
        wk = _smooth_closed(wk)
        lines.append((_points(cut, c0, th, W, R, wk), wk))
        if len(lines) >= max_lines:
            break
    return lines


def _normals(ext, P):
    """Outward unit normals of the exterior at points near its surface: a
    mesh's own face normals (the distance grid's gradient is a voxel
    staircase within a cell of the surface), else the gradient."""
    from .shapes import MeshShape
    from .shells import _grad
    P = np.asarray(P, float)
    if isinstance(ext, MeshShape):
        from trimesh.proximity import closest_point
        _, _, tri = closest_point(ext.mesh, P)
        return np.asarray(ext.mesh.face_normals)[tri]
    g = _grad(ext, P, h=0.05)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def _offsets(ext, P, g, rs):
    """Where the surface is along each point's normal line: the outermost
    r in rs (ascending) with P + r g inside, interpolated to the crossing;
    NaN where the line misses the surface."""
    F = ext.sdf((P[:, None, :] + rs[None, :, None] * g[:, None, :]).reshape(-1, 3)).reshape(len(P), len(rs))
    inside = F <= 0
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        k = np.where(inside[i])[0]
        if len(k) == 0 or k[-1] == len(rs) - 1:
            continue
        j = k[-1]
        f0, f1 = F[i, j], F[i, j + 1]
        out[i] = rs[j] + (rs[j + 1] - rs[j]) * (-f0) / max(f1 - f0, 1e-12)
    return out


def seam_visibility(ext, curves, hidden=None, deg=CREASE_DEG, ds=1.0, reach=1.0):
    """mm of seam seen and mm hidden: hidden where the seam runs along a
    crease or an edge (the exterior's section across the seam turns by more
    than `deg` within +-`reach` mm of it: a reveal, a shoulder, a
    character line), or inside a `hidden` shape (an underside, an opening). A seam
    crossing an edge (round a box's corner) is not hidden by it: the
    section across the seam runs along the edge, not over it."""
    from .shells import _resample
    seen = hid = 0.0
    rs = np.linspace(-1.5, 1.5, 31)
    for c in curves:
        if len(c) < 4:
            continue
        q, _, L = _resample(c, ds)
        seg = L / len(q)
        g = _normals(ext, q)
        tan = np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        across = np.cross(g, tan)
        across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-9)
        r0 = _offsets(ext, q, g, rs)
        rp = _offsets(ext, q + reach * across, g, rs)
        rm = _offsets(ext, q - reach * across, g, rs)
        kink = np.degrees(np.abs(np.arctan((rp - r0) / reach) - np.arctan((r0 - rm) / reach)))
        # a section that runs off the part (a seam just short of a sharp
        # edge, a sliver) hides nothing
        h = np.isfinite(kink) & (kink > deg)
        for s in hidden or []:
            h |= s.sdf(q) <= 0.5
        hid += seg * float(h.sum())
        seen += seg * float((~h).sum())
    return seen, hid


def rim_flat(curves, cut, tol=0.3):
    """Share of the seam that lies on one plane across the lift (the bed,
    when a half is printed standing on its rim)."""
    if not curves:
        return 0.0
    P = np.vstack(curves)
    w = P @ cut.n
    return float(max(np.mean(w <= w.min() + tol), np.mean(w >= w.max() - tol)))


def _tilted(base_cut, levels, mid_uv):
    """Planes through the middle of the part at each level, tilted about
    each axis across the lift."""
    from .shells import _Cut
    out = []
    mid = mid_uv[0] * base_cut.e1 + mid_uv[1] * base_cut.e2
    for lv in levels:
        p0 = mid + lv * base_cut.n
        for e in (base_cut.e1, base_cut.e2):
            for sg in (1, -1):
                for deg in TILTS:
                    a = math.radians(deg)
                    n = math.cos(a) * base_cut.n + sg * math.sin(a) * e
                    c = _Cut.plane(n, float(n @ p0))
                    c.kind = "tilted"
                    c.tilt = {"deg": deg, "toward": _r(sg * e, 3), "through": round(float(lv), 3)}
                    out.append(c)
    return out


def _r(v, k=4):
    return [round(float(x), k) for x in np.asarray(v, float)]


def _stepped(ext, base_cut, levels, comps, samples, avoid_specs, width=0.5, clear=1.0):
    """Flat levels that climb over (or dip under) whatever they would cut
    through: each component standing across the level, and each `avoid`
    region the seam would cross. Only for an axis lift (the steps' regions
    are boxes along it)."""
    from .shells import _stepped_cut
    if base_cut.axis is None:
        return []
    a, sg = base_cut.axis
    lo, hi = ext.bounds()
    out = []
    seen = set()
    for lv in levels:
        # lv is a height along n; on an axis that is sg * coordinate
        crossing = []
        for c in comps:
            w = samples[c["id"]] @ base_cut.n
            if w.min() < lv < w.max():
                crossing.append((c["id"], samples[c["id"]], w))
        for j, (spec, sh) in enumerate(avoid_specs):
            P = _surface_band(ext, sh)
            if len(P) == 0:
                continue
            w = P @ base_cut.n
            if w.min() - 0.5 < lv < w.max() + 0.5:
                crossing.append((f"avoid[{j}]", P, w))
        if not crossing:
            continue
        options = []
        for cid, P, w in crossing:
            up, down = float(w.max() + clear), float(w.min() - clear)
            near = up if abs(up - lv) <= abs(lv - down) else down
            options.append((cid, P, {"up": up, "down": down, "near": near}))
        for mode in ("near", "up", "down"):
            steps = []
            for cid, P, o in options:
                lvl = o[mode]
                bx_lo, bx_hi = P.min(axis=0) - 1.5, P.max(axis=0) + 1.5
                bx_lo[a], bx_hi[a] = lo[a] - 5.0, hi[a] + 5.0
                steps.append({"at": sg * lvl, "where": {"type": "box", "min": _r(bx_lo, 3), "max": _r(bx_hi, 3)},
                              "over": cid})
            key = (round(lv, 2), tuple(round(s["at"], 2) for s in steps))
            if key in seen:
                continue
            seen.add(key)
            try:
                c = _stepped_cut(ext, base_cut.name(), sg * lv, [{k: s[k] for k in ("at", "where")} for s in steps],
                                 width, None)
            except ValueError:
                continue
            c.steps = [{"at": round(s["at"], 3), "over": s["over"], "where": s["where"]} for s in steps]
            c.at = round(sg * lv, 3)
            out.append(c)
    return out


def _surface_band(ext, sh, step=1.0):
    """Points of the exterior's surface inside a shape (an avoid region)."""
    lo, hi = sh.bounds()
    lo2, hi2 = ext.bounds()
    lo, hi = np.maximum(lo, lo2 - 1), np.minimum(hi, hi2 + 1)
    if np.any(hi <= lo):
        return np.zeros((0, 3))
    ax = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    G = G[sh.sdf(G) <= 0]
    return G[np.abs(ext.sdf(G)) <= step] if len(G) else G


def propose(ext, base_cut, flat_levels, comps, samples, avoid_specs=(), min_depth=6.0):
    """Non-flat cuts for one lift direction: tilted planes and stepped
    lines through the given flat levels (heights along the lift), the
    silhouette and the creases. Each is a shells._Cut with .kind set."""
    from .shells import _seam_cut
    out = []
    prof = _profile(ext, base_cut)
    out += _tilted(base_cut, flat_levels, prof[3])
    out += _stepped(ext, base_cut, flat_levels, comps, samples, list(avoid_specs))
    lines = []
    sil = silhouette_line(ext, base_cut, prof, min_depth=min_depth)
    if sil is not None:
        lines.append(("silhouette", sil[0]))
    for pts, _w in crease_lines(ext, base_cut, prof, min_depth=min_depth):
        lines.append(("crease", pts))
    for kind, pts in lines:
        if len(pts) < 8:
            continue
        try:
            c = _seam_cut(ext, list(base_cut.n), pts)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue          # a line too ragged to fill a surface through
        c.kind = kind
        c.seam_points = pts
        out.append(c)
    return out
