"""
Ridges: ribs drawn for printing, not moulding.

A moulded rib is a thin drafted wall: thin because thick plastic sinks
and voids as it cools, drafted so the part leaves the tool. A print has
neither limit, so the section can go where the bending wants material:
away from the skin. A ridge here is a web standing off the skin with a
teardrop bulb along its top edge, swept along a curve (a stress line) and
along the print direction, so every face of it prints without support:

- the web's sides are parallel to the print direction (vertical walls);
- the bulb's top is a round (it faces up);
- the bulb's underside is a chin at `chin` degrees from the bed (45 by
  default), a teardrop pointing down, never a horizontal ledge;
- `windows`: where a web is deep and lightly loaded, diamond openings
  with their sides at 45 degrees (a printed truss: the top of each
  opening is a 45 degree arch, its corners rounded), the bulb and a
  foot along the skin kept whole as the truss's chords.

The section lofts along each line: web thickness, height and bulb
radius each follow a weight per vertex (a stress, as a fraction of its
case's 99th percentile), so a ridge is tall and thick where the load is
and drops to a low rib where it is not. Where ridges cross they are
blended with a smooth union of `blend` mm, so the node is one filleted
lump, not two slabs through each other. The root fillet to the skin is
the caller's (a smooth union with the skin field), as it is for every
other feature on that skin.

`section(...)` integrates a ridge and its share of skin as a T beam (or
the plain wall it replaces), so a design can be compared per gram before
anything is meshed.

    r = Ridges(lines, "-y", weight=w, t=(1.6, 2.4), H=(4, 12), R=(1.0, 2.2))
    f = r(P, v)    # v: the height of P above the skin, along the print direction
"""
from __future__ import annotations

import numpy as np

from .ribs import _axis, _densify, _plane_basis

__all__ = ["Ridges", "WallRidges", "network", "profile", "section", "split", "thin", "tidy"]


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _box2(x, y, bx, by):
    qx, qy = np.abs(x) - bx, np.abs(y) - by
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def profile(u, v, t, H, R, chin=45.0):
    """The ridge's section, a 2D field: u across the ridge, v up from the
    skin (along the print direction). A web of thickness t from v = -1
    (into the skin, so the caller's root fillet has something to blend)
    to the bulb; the bulb a circle of radius R whose top is at v = H, and
    under it a cone with its sides at `chin` degrees to the bed meeting
    the circle on its tangents (a teardrop, point down). R <= t/2 is a
    plain wall with a round top."""
    u = np.abs(np.asarray(u, float))
    v = np.asarray(v, float)
    t, H, R = (np.broadcast_to(np.asarray(x, float), u.shape) for x in (t, H, R))
    R = np.maximum(R, 0.5 * t)
    vc = H - R
    # web: a box from -1 to the bulb's centre
    lo = -1.0
    web = _box2(u, v - 0.5 * (vc + lo), 0.5 * t, 0.5 * (vc - lo))
    disk = np.hypot(u, v - vc) - R
    # the chin: a downward cone, apex below the centre at R / cos(a) where a is the chin
    # angle from the bed, cut off above the tangent points
    a = np.radians(chin)
    ca, sa = np.cos(a), np.sin(a)
    apex = vc - R / ca
    # distance to a cone of half-angle (90 - a) about the vertical, apex up at `apex`
    # (inside: |u| * tan(a) < v - apex): the line through the apex at angle a
    cone = u * sa - (v - apex) * ca
    tangent_v = vc - R * ca
    tear = np.minimum(disk, np.maximum(cone, v - tangent_v))
    return np.minimum(web, tear)


def _window(s, v, s0, v0, a, r):
    """Diamonds side 45 degrees (a square turned 45), half-diagonal a, corners rounded r:
    negative inside the opening."""
    x, y = (s - s0 + v - v0) / np.sqrt(2.0), (s - s0 - v + v0) / np.sqrt(2.0)
    b = np.maximum(a / np.sqrt(2.0) - r, 1e-3)
    return _box2(x, y, b, b) - r


class Ridges:
    """Ridges along polylines, swept along a print direction, lofted by a weight per vertex.

    lines: polylines (N x 3). direction: the print (build) direction, "+z", "-y" or a
    vector. weight: per line, one value per vertex (any scale: ranked to 0..1 by its 5th
    and 95th percentiles over all lines) or None (all 1). t, H, R: (at weight 0, at
    weight 1), mm; H may instead be
    a callable(P) -> height (the room the part has, or where an optimizer put material),
    so a ridge can be deep and lightly loaded: windowed. chin: bulb underside, degrees from the bed. windows: diamond openings
    where the web's clear height (between a `foot` above the skin and the bulb's chin)
    holds one of at least `window_min` mm, and the weight is under `window_below`;
    struts between them `strut` mm. blend: the smooth union where ridges cross."""

    def __init__(self, lines, direction, weight=None, t=(1.6, 2.4), H=(4.0, 12.0), R=(1.0, 2.2), chin=45.0,
                 windows=True, window_min=4.0, window_below=0.6, foot=2.5, strut=2.5, window_round=0.8,
                 blend=2.0, step=1.0, k=4):
        from scipy.spatial import cKDTree

        d = _axis(direction)
        u, w = _plane_basis(d)
        self.M = np.stack([u, w], 1)
        self.chin, self.blend, self.k = float(chin), float(blend), int(k)
        self.windows, self.window_min, self.window_below = bool(windows), float(window_min), float(window_below)
        self.foot, self.strut, self.window_round = float(foot), float(strut), float(window_round)
        A, Bp, WA, WB, SA, SB, ID = [], [], [], [], [], [], []
        self._lines = []
        allw = np.concatenate([np.ravel(x) for x in weight]) if weight is not None else np.ones(1)
        w0, w1 = np.percentile(allw, 5), np.percentile(allw, 95)
        for i, pl in enumerate(lines):
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                continue
            wl = np.ones(len(pl)) if weight is None else np.clip((np.asarray(weight[i], float) - w0) / max(w1 - w0, 1e-9), 0, 1)
            # the weight smoothed along the line (a section that changes over a few mm, not per vertex)
            s_raw = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))])
            dense = _densify(pl, step)
            sd = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
            # dense keeps the original vertices; map arc length of the dense line onto the original's
            wd = np.interp(sd * (s_raw[-1] / max(sd[-1], 1e-9)), s_raw, wl)
            sig = max(1, round(6.0 / step))
            if len(wd) > 2:
                from scipy.ndimage import gaussian_filter1d
                wd = gaussian_filter1d(wd, sig / 2.0, mode="nearest")
            q = dense @ self.M
            A.append(q[:-1]); Bp.append(q[1:]); WA.append(wd[:-1]); WB.append(wd[1:]); SA.append(sd[:-1]); SB.append(sd[1:])
            ID.append(np.full(len(q) - 1, i))
            self._lines.append((i, dense, sd, wd))
        if not A:
            raise ValueError("no ridge lines")
        self.a, self.b = np.concatenate(A), np.concatenate(Bp)
        self.wa, self.wb = np.concatenate(WA), np.concatenate(WB)
        self.sa, self.sb = np.concatenate(SA), np.concatenate(SB)
        self.id = np.concatenate(ID)
        self.step = float(step)
        self.tree = cKDTree(0.5 * (self.a + self.b))
        self.t, self.R = (tuple(float(x) for x in p) for p in (t, R))
        self.H = H if callable(H) else tuple(float(x) for x in H)
        self._place_windows()

    def params(self, w, P=None):
        lerp = lambda p: p[0] + (p[1] - p[0]) * w
        if callable(self.H):
            H = np.broadcast_to(np.asarray(self.H(P), float).reshape(-1, 1), w.shape)
        else:
            H = lerp(self.H)
        return lerp(self.t), H, lerp(self.R)

    def __call__(self, P, v):
        """The ridges' field at P (negative inside), v the height of each point above the
        skin along the print direction."""
        P = np.asarray(P, float).reshape(-1, 3)
        v = np.asarray(v, float).reshape(-1)
        out = np.empty(len(P))
        for a in range(0, len(P), 100_000):
            out[a:a + 100_000] = self._eval(P[a:a + 100_000], v[a:a + 100_000])
        return out

    def _eval(self, P, v):
        Q = P @ self.M
        n = len(Q)
        # enough nearest segments to reach several lines; each line then counts once, by its
        # nearest segment (a line's neighbouring segments would otherwise blend into a bulge and
        # fill its windows with their own, offset, pattern)
        k = min(self.k * 6, len(self.a))
        _, idx = self.tree.query(Q, k=k)
        idx = np.asarray(idx).reshape(n, k)
        a, b = self.a[idx], self.b[idx]
        ab = b - a
        tt = np.clip(np.einsum("nkd,nkd->nk", Q[:, None] - a, ab) / np.maximum(np.einsum("nkd,nkd->nk", ab, ab), 1e-300), 0, 1)
        uu = np.linalg.norm(Q[:, None] - (a + tt[..., None] * ab), axis=2)
        ids = self.id[idx]
        o = np.lexsort((uu, ids), axis=-1)
        ids_s = np.take_along_axis(ids, o, 1)
        first = np.ones((n, k), bool)
        first[:, 1:] = ids_s[:, 1:] != ids_s[:, :-1]
        us = np.where(first, np.take_along_axis(uu, o, 1), np.inf)
        m = min(self.k, k)
        pick = np.take_along_axis(o, np.argsort(us, axis=1)[:, :m], 1)       # the m nearest lines
        valid = np.isfinite(np.take_along_axis(us, np.argsort(us, axis=1)[:, :m], 1))
        g = lambda x: np.take_along_axis(x, pick, 1)
        uu, tt, idx = g(uu), g(tt), g(idx)
        w = self.wa[idx] + tt * (self.wb[idx] - self.wa[idx])
        s = self.sa[idx] + tt * (self.sb[idx] - self.sa[idx])
        t, H, R = self.params(w, P)
        vv = np.broadcast_to(v[:, None], uu.shape)
        f = profile(uu, vv, t, H, R, self.chin)
        if self.windows:
            f = np.maximum(f, -self._windows(self.id[idx], s, vv))
        f = np.where(valid, f, 1e6)
        out = f[:, 0]
        for j in range(1, m):
            out = _smin(out, f[:, j], self.blend)
        return out

    def _clear(self, H, R):
        """The web's clear height between the foot and a strip under the chin: (lo, hi)."""
        apex = H - R - R / np.cos(np.radians(self.chin))
        return self.foot, apex - 0.6

    def _place_windows(self):
        """Diamond openings along each line, in the gaps between its crossings with other lines
        (a crossing is a node: solid), as many as fit at a strut apart, centred in the gap; each
        sized to the web's clear height at its centre, and only where the web is lightly loaded."""
        from scipy.spatial import cKDTree

        keys, A, V0 = [], [], []
        if self.windows and len(self._lines) > 1:
            Q2 = [(i, d @ self.M) for i, d, _, _ in self._lines]
            for i, dense, sd, wd in self._lines:
                others = np.vstack([q for j, q in Q2 if j != i])
                near = cKDTree(others).query(dense @ self.M)[0]
                t, H, R = self.params(wd[:, None], dense)
                t, H, R = t[:, 0], H[:, 0], R[:, 0]
                lo, hi = self._clear(H, R)
                a = 0.5 * (hi - lo)
                node = near < (t + self.blend + 1.0)               # at a crossing
                ok = (~node) & (2 * a >= self.window_min) & (wd < self.window_below)
                # gaps: runs of ok
                edge = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
                for g0, g1 in zip(edge[::2], edge[1::2]):
                    s0, s1 = sd[g0] + self.strut, sd[g1 - 1] - self.strut
                    cs, cur = [], s0
                    while True:
                        ac = min(float(np.interp(cur + 1.0, sd, a)), 0.5 * (s1 - cur))   # shrunk to fit the gap
                        c = cur + ac
                        if c + ac > s1 or 2 * ac < self.window_min:
                            break
                        cs.append((c, ac))
                        cur = c + ac + self.strut * np.sqrt(2.0)
                    if not cs:
                        continue
                    shift = 0.5 * (s1 - (cs[-1][0] + cs[-1][1]))
                    for c, ac in cs:
                        c += shift
                        ac = min(ac, float(np.interp(c, sd, a)))
                        hc = float(np.interp(c, sd, hi))
                        keys.append(i * 1e6 + c); A.append(ac); V0.append(hc - ac)
        o = np.argsort(keys)
        self.win_key = np.asarray(keys, float)[o]
        self.win_a, self.win_v0 = np.asarray(A, float)[o], np.asarray(V0, float)[o]

    def _windows(self, ids, s, v):
        """The nearest placed opening on each candidate's line (negative inside)."""
        if len(self.win_key) == 0:
            return np.full(s.shape, 1e3)
        key = ids * 1e6 + s
        j = np.clip(np.searchsorted(self.win_key, key), 1, len(self.win_key) - 1)
        j = np.where(np.abs(self.win_key[j - 1] - key) < np.abs(self.win_key[j] - key), j - 1, j)
        kc = self.win_key[j]
        same = np.floor(kc / 1e6 + 1e-9) == ids
        o = _window(s, v, kc - ids * 1e6, self.win_v0[j], self.win_a[j], self.window_round)
        return np.where(same, o, 1e3)

    def overhang(self, H_samples=None):
        """The steepest underside a ridge has, by construction: the chin (degrees from the
        bed) and the windows' 45 degree tops; the web is vertical."""
        return {"chin deg": self.chin, "window tops deg": 45.0, "web": "vertical"}


def _nearest_lines(tree, A, B, ID, Q, k, raw=False):
    """For each query point, the k nearest distinct lines: (segment index, t on it, distance),
    each line by its nearest segment, with a validity mask (and, with raw, t unclamped: below 0
    or above 1 past the segment's ends)."""
    n = len(Q)
    # candidates by segment: a line 1 mm per segment fills ~2r of them within r, so two nearby lines
    # could crowd a third (a tall ridge 8 mm off) out of the list, and its ridge blinked in and out
    kk = min(k * 20, len(A))
    _, idx = tree.query(Q, k=kk)
    idx = np.asarray(idx).reshape(n, kk)
    a, b = A[idx], B[idx]
    ab = b - a
    t_raw = np.einsum("nkd,nkd->nk", Q[:, None] - a, ab) / np.maximum(np.einsum("nkd,nkd->nk", ab, ab), 1e-300)
    tt = np.clip(t_raw, 0, 1)
    uu = np.linalg.norm(Q[:, None] - (a + tt[..., None] * ab), axis=2)
    ids = ID[idx]
    o = np.lexsort((uu, ids), axis=-1)
    ids_s = np.take_along_axis(ids, o, 1)
    first = np.ones((n, kk), bool)
    first[:, 1:] = ids_s[:, 1:] != ids_s[:, :-1]
    us = np.where(first, np.take_along_axis(uu, o, 1), np.inf)
    m = min(k, kk)
    srt = np.argsort(us, axis=1)[:, :m]
    pick = np.take_along_axis(o, srt, 1)
    valid = np.isfinite(np.take_along_axis(us, srt, 1))
    g = lambda x: np.take_along_axis(x, pick, 1)
    if raw:
        return g(idx), g(tt), g(uu), valid, g(t_raw)
    return g(idx), g(tt), g(uu), valid


def tidy(lines, weight=None, smooth=4.0, min_length=35.0, step=1.0):
    """Stress lines as ridge paths: each resampled at `step` and smoothed along its length
    (a Gaussian of `smooth` mm: a traced streamline wanders by a cell or two, a ridge shouldn't),
    and lines shorter than `min_length` dropped (a short ridge stiffens nothing and reads as
    clutter). Returns (lines, weights or None)."""
    from scipy.ndimage import gaussian_filter1d

    outL, outW = [], []
    for i, pl in enumerate(lines):
        pl = np.asarray(pl, float)
        if len(pl) < 2:
            continue
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))])
        if s[-1] < min_length:
            continue
        n = max(2, int(np.ceil(s[-1] / step)) + 1)
        u = np.linspace(0, s[-1], n)
        q = np.stack([np.interp(u, s, pl[:, j]) for j in range(3)], 1)
        if smooth > 0 and n > 4:
            sm = gaussian_filter1d(q, smooth / step, axis=0, mode="nearest")
            sm[0], sm[-1] = q[0], q[-1]                 # ends stay where they were
            q = sm
        outL.append(q)
        if weight is not None:
            outW.append(np.interp(u, s, np.asarray(weight[i], float)))
    return outL, (outW if weight is not None else None)


def thin(lines, weight=None, spacing=22.0, parallel=0.8, limit=None):
    """Fewer, deeper ridges: keep lines most loaded first (weight times length), dropping a line
    when most of it runs within `spacing` of a kept line and parallel to it (|cos| > parallel).
    Crossing lines are kept: they are the other family. `limit`: at most that many lines (the
    most loaded)."""
    from scipy.spatial import cKDTree

    L = [np.asarray(x, float) for x in lines]
    W = [np.ones(len(x)) if weight is None else np.asarray(weight[i], float) for i, x in enumerate(L)]
    tan = []
    for x in L:
        d = np.gradient(x, axis=0)
        tan.append(d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9))
    score = [float(np.sum(w[:-1] * np.linalg.norm(np.diff(x, axis=0), axis=1))) for x, w in zip(L, W, strict=True)]
    keep, KP, KT = [], [], []
    for i in np.argsort(score)[::-1]:
        if limit is not None and len(keep) >= limit:
            break
        if KP:
            tree = cKDTree(np.vstack(KP))
            dist, j = tree.query(L[i])
            par = np.abs(np.einsum("nd,nd->n", tan[i], np.vstack(KT)[j])) > parallel
            if np.mean((dist < spacing * 0.75) & par) > 0.5:
                continue
        keep.append(int(i)); KP.append(L[i]); KT.append(tan[i])
    keep.sort()
    return [L[i] for i in keep], ([W[i] for i in keep] if weight is not None else None), keep


def split(lines, weight=None, ok=None, min_length=35.0):
    """Cut lines where ok(points) is False (too near a part, the part line, or out of room) and
    keep the pieces at least `min_length` long. A cut end becomes a free end (WallRidges tapers
    it into the skin) rather than a ridge sliced off where it meets the obstacle."""
    outL, outW = [], []
    for i, pl in enumerate(lines):
        pl = np.asarray(pl, float)
        w = None if weight is None else np.asarray(weight[i], float)
        m = np.ones(len(pl), bool) if ok is None else np.asarray(ok(pl), bool)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
        for a, b in zip(edges[0::2], edges[1::2], strict=True):
            q = pl[a:b]
            if len(q) < 2 or np.linalg.norm(np.diff(q, axis=0), axis=1).sum() < min_length:
                continue
            outL.append(q)
            outW.append(None if w is None else w[a:b])
    return outL, (outW if weight is not None else None)


def network(lines, weight=None, snap=10.0, stub=12.0, anchors=None, angle=70.0, touch=2.0):
    """Join ridge lines into a network whose every end means something. For each end of each
    line, in this order:

    - within `snap` mm of another line (ahead of it, within `angle` degrees of its own direction),
      it is carried straight on to meet that line: a T junction, blended by the ridges' union;
    - within `snap` of an anchor (a boss or a post: (point, radius) pairs), carried on to the
      anchor's edge;
    - when the line crossed another within its last `stub` mm, the stub past that crossing is
      cut off, so the line ends in the crossing;
    - otherwise the end is free: WallRidges tapers its height to nothing over its `taper`, a
      rounded run-out into the skin.

    Returns (lines, weights or None, ends): ends[i] = (start joined, end joined)."""
    from scipy.spatial import cKDTree

    L = [np.asarray(x, float).copy() for x in lines]
    W = None if weight is None else [np.asarray(w, float).copy() for w in weight]
    joined = [[False, False] for _ in L]
    ca = np.cos(np.radians(angle))
    for i in range(len(L)):
        for side in (0, 1):
            pl = L[i]
            if len(pl) < 3:
                continue
            end = pl[0] if side == 0 else pl[-1]
            k = min(4, len(pl) - 1)
            tan = (pl[0] - pl[k]) if side == 0 else (pl[-1] - pl[-1 - k])
            tan = tan / max(np.linalg.norm(tan), 1e-9)
            best = None
            for j, q in enumerate(L):
                if j == i:
                    continue
                d = np.linalg.norm(q - end, axis=1)
                jj = int(np.argmin(d))
                if d[jj] > snap:
                    continue
                v = q[jj] - end
                nv = np.linalg.norm(v)
                if nv > touch and v @ tan < ca * nv:
                    continue                           # behind or beside the end: not a run-on
                if best is None or d[jj] < best[0]:
                    best = (d[jj], q[jj])
            if best is None and anchors is not None:
                for c, r in anchors:
                    c = np.asarray(c, float)
                    v = c - end
                    nv = np.linalg.norm(v)
                    if nv - r <= snap and (nv - r <= touch or v @ tan >= ca * nv):
                        tgt = c - v / max(nv, 1e-9) * r * 0.5          # into the anchor by half its radius
                        if best is None or nv < best[0]:
                            best = (nv, tgt)
            if best is not None:
                d, tgt = best
                n = max(1, int(np.ceil(d)))
                ext = end + (tgt - end) * (np.arange(1, n + 1) / n)[:, None]
                if side == 0:
                    L[i] = np.vstack([ext[::-1], pl])
                    if W is not None:
                        W[i] = np.concatenate([np.full(n, W[i][0]), W[i]])
                else:
                    L[i] = np.vstack([pl, ext])
                    if W is not None:
                        W[i] = np.concatenate([W[i], np.full(n, W[i][-1])])
                joined[i][side] = True
                continue
            # a stub past a crossing: cut it back to the crossing
            others = [q for j, q in enumerate(L) if j != i]
            if not others:
                continue
            tree = cKDTree(np.vstack(others))
            d, _ = tree.query(pl)
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))])
            near = np.flatnonzero(d < touch)
            if side == 0:
                near = near[s[near] <= stub]
                if len(near) and s[-1] - s[near.max()] > stub:
                    c = int(near.max())
                    L[i] = pl[c:]
                    W = W if W is None else [*W[:i], W[i][c:], *W[i + 1:]]
                    joined[i][0] = True
            else:
                near = near[s[-1] - s[near] <= stub]
                if len(near) and s[near.min()] > stub:
                    c = int(near.min())
                    L[i] = pl[:c + 1]
                    W = W if W is None else [*W[:i], W[i][:c + 1], *W[i + 1:]]
                    joined[i][1] = True
    return L, W, [tuple(e) for e in joined]


def _taper(s, length, at_start, at_end):
    """A smooth ramp 0 -> 1 over `length` from each free end (1 elsewhere)."""
    f = np.ones_like(s)
    if length <= 0:
        return f
    ramp = lambda x: np.where(x >= 1, 1.0, np.where(x <= 0, 0.0, 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))))
    if at_start:
        f = np.minimum(f, ramp(s / length))
    if at_end:
        f = np.minimum(f, ramp((s[-1] - s) / length))
    return f


class WallRidges:
    """Ridges that follow a curved skin, side walls included: each stands off the skin along
    its normal and leans toward the print direction just enough that no face under it is
    flatter than 45 degrees (on a side wall: a 45 degree shelf, its teardrop bulb's chin
    pointing down the print direction). On skin that faces the bed it is Ridges' upright web.

    lines, weight, t, R, chin, blend: as Ridges. H: (at weight 0, at weight 1), the height off
    the skin; room: optional callable(points on the lines) -> the free height there (it caps
    H; where it leaves under `min_height` the ridge stops, tapering into the skin either side
    rather than being sliced). normal: callable(points on the lines) -> unit vectors off the
    skin into the part (taken on the lines, smoothed along them, and carried across each section).
    On skin facing away from the bed (a ceiling, normal more than 135 degrees from the print
    direction) there is no ridge.

    Ends (network()'s): a free end tapers its height (and its web, to half) to nothing over
    `taper` mm, a rounded run-out into the skin; a joined end runs into its neighbour at full
    section and is blended. Where ridges cross or join, each takes the taller one's height
    within `match` mm (smoothed along it), so their tops meet flush instead of a step. Every
    ridge ends at its line's ends (a flat cap, of nothing where it tapered): it never runs on
    past them."""

    def __init__(self, lines, direction, normal, weight=None, t=(1.8, 3.0), H=(6.0, 16.0), R=(1.4, 2.6),
                 chin=45.0, room=None, min_height=3.0, blend=2.0, step=1.0, k=4, ends=None, taper=12.0,
                 match=6.0):
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d
        from scipy.spatial import cKDTree

        self.b = _axis(direction)
        self.normal, self.chin, self.blend, self.k = normal, float(chin), float(blend), int(k)
        allw = np.concatenate([np.ravel(x) for x in weight]) if weight is not None else np.ones(1)
        w0, w1 = np.percentile(allw, 5), np.percentile(allw, 95)
        dense_all, par_all, ids_all, f_all = [], [], [], []
        for i, pl in enumerate(lines):
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                continue
            wl = np.ones(len(pl)) if weight is None else np.clip((np.asarray(weight[i], float) - w0) / max(w1 - w0, 1e-9), 0, 1)
            s_raw = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))])
            dense = _densify(pl, step)
            sd = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
            wd = np.interp(sd * (s_raw[-1] / max(sd[-1], 1e-9)), s_raw, wl)
            if len(wd) > 2:
                wd = gaussian_filter1d(wd, 3.0 / step, mode="nearest")
            lerp = lambda p, w=wd: p[0] + (p[1] - p[0]) * w
            tv, Hv, Rv = lerp(t), lerp(H), lerp(R)
            if room is not None:
                rm = np.asarray(room(dense), float)
                if len(rm) > 2:
                    rm = gaussian_filter1d(rm, 3.0 / step, mode="nearest")
                Hv = np.minimum(Hv, rm)
            # runs with room for a ridge, each tapered at its ends: at a free end of the line, and
            # wherever the room runs out mid-line; a joined end keeps its full section
            ok = Hv >= min_height
            f = np.zeros(len(dense))
            je = (False, False) if ends is None else tuple(ends[i])
            edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
            for a0, b0 in zip(edges[0::2], edges[1::2], strict=True):
                ss = sd[a0:b0] - sd[a0]
                if len(ss) < 2 or ss[-1] < min(taper, 0.5 * sd[-1]):
                    continue
                f[a0:b0] = _taper(ss, taper, not (a0 == 0 and je[0]), not (b0 == len(dense) and je[1]))
            Hv = np.where(f > 0, Hv, 0.0)                   # the taper goes on after the flush tops
            dense_all.append(dense); par_all.append(np.stack([tv, Hv, Rv], 1)); ids_all.append(i); f_all.append(f)
        if not dense_all:
            raise ValueError("no ridge lines")
        # flush tops: within `match` of another ridge, a ridge takes the taller one's height
        if match > 0 and len(dense_all) > 1:
            pts = np.vstack(dense_all)
            lid = np.concatenate([np.full(len(d), j) for j, d in enumerate(dense_all)])
            Hall = np.concatenate([p_[:, 1] for p_ in par_all])
            tree = cKDTree(pts)
            for j, (d, p_) in enumerate(zip(dense_all, par_all, strict=True)):
                nb = tree.query_ball_point(d, match)
                Hn = np.array([max([Hall[q] for q in n_ if lid[q] != j and Hall[q] > 0] or [0.0]) for n_ in nb])
                lift = np.where((Hn > p_[:, 1]) & (p_[:, 1] > 0), Hn, p_[:, 1])
                if len(lift) > 2:                       # a plateau over the node, eased in either side
                    lift = maximum_filter1d(lift, max(1, round(match / step)), mode="nearest")
                    lift = np.maximum(gaussian_filter1d(lift, 3.0 / step, mode="nearest"), p_[:, 1])
                    lift = np.where(p_[:, 1] > 0, lift, 0.0)
                p_[:, 1] = lift
        # then the tapers (free ends, room run-outs), so a lifted node never lifts a taper back up
        for p_, f in zip(par_all, f_all, strict=True):
            p_[:, 1] *= f
            p_[:, 0] *= 0.5 + 0.5 * f
            p_[:, 2] = np.minimum(p_[:, 2], np.maximum(0.3 * p_[:, 1], 0.5 * p_[:, 0] * f))
        A, Bs, PA, PB, ID, END, NA, NB, TA, TB = [], [], [], [], [], [], [], [], [], []
        for i, dense, par in zip(ids_all, dense_all, par_all, strict=True):
            A.append(dense[:-1]); Bs.append(dense[1:]); PA.append(par[:-1]); PB.append(par[1:])
            # the skin's normal on the line itself (smoothed along it), not at each query point: a
            # point in a ridge near another wall would otherwise take that wall's normal, and the
            # ridge tore where the two disagreed
            nv = np.asarray(normal(dense), float)
            if len(nv) > 2:
                nv = gaussian_filter1d(nv, 2.0 / step, axis=0, mode="nearest")
            nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
            NA.append(nv[:-1]); NB.append(nv[1:])
            tg = np.gradient(dense, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
            TA.append(tg[:-1]); TB.append(tg[1:])
            ID.append(np.full(len(dense) - 1, i))
            e = np.zeros(len(dense) - 1, np.int8)
            e[0] |= 1
            e[-1] |= 2
            END.append(e)
        self.A, self.B = np.concatenate(A), np.concatenate(Bs)
        self.PA, self.PB, self.id = np.concatenate(PA), np.concatenate(PB), np.concatenate(ID)
        self.end = np.concatenate(END)
        self.NA, self.NB = np.concatenate(NA), np.concatenate(NB)
        self.TA, self.TB = np.concatenate(TA), np.concatenate(TB)
        self.tree = cKDTree(0.5 * (self.A + self.B))

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        out = np.empty(len(P))
        for a in range(0, len(P), 100_000):
            out[a:a + 100_000] = self._eval(P[a:a + 100_000])
        return out

    def _eval(self, P):
        idx, tt, _, valid, traw = _nearest_lines(self.tree, self.A, self.B, self.id, P, self.k, raw=True)
        a, b = self.A[idx], self.B[idx]
        seg = np.linalg.norm(b - a, axis=2)
        e = self.end[idx]
        over = np.maximum(np.where(e & 1, -traw, 0.0), np.where(e & 2, traw - 1.0, 0.0)) * seg   # past the line's end
        # past an inner end of a segment (when the line's next segment didn't make the candidate list)
        # the section would run on along the tangent, a ghost ridge: held back beyond 1 mm of overrun
        over = np.maximum(over, np.maximum(-traw, traw - 1.0) * seg - 1.0)
        # the tangent eased from vertex to vertex, not each segment's own: with a segment's own, the
        # section turned in a step at every vertex, and a bulb 8 mm off a curving line came out
        # stepped (a jump of up to 0.9 mm where the nearest segment changed)
        T = self.TA[idx] + tt[..., None] * (self.TB[idx] - self.TA[idx])
        T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-12)
        C = a + tt[..., None] * (b - a)
        par = self.PA[idx] + tt[..., None] * (self.PB[idx] - self.PA[idx])
        t, H, R = par[..., 0], par[..., 1], par[..., 2]
        perp = lambda x: x - np.einsum("nkd,nkd->nk", x, T)[..., None] * T
        unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        d = perp(P[:, None] - C)
        n = unit(perp(self.NA[idx] + tt[..., None] * (self.NB[idx] - self.NA[idx])))
        g = perp(np.broadcast_to(self.b, C.shape).copy())
        sg = np.linalg.norm(g, axis=-1)
        gh = np.where(sg[..., None] > 1e-6, g / np.maximum(sg[..., None], 1e-12), n)
        # the web's axis: the normal, leaned toward up until its sides are at most 45 degrees
        # from the print direction (in 3D: the in-plane angle allowed grows as the plane tilts)
        phimax = np.arcsin(np.clip(np.cos(np.radians(45.0)) / np.maximum(sg, 1e-6), 0, 1))
        cn = np.clip(np.einsum("nkd,nkd->nk", n, gh), -1, 1)
        th = np.arccos(cn)                                   # angle normal to up
        side = unit(n - cn[..., None] * gh)                  # in-plane, across up, toward the normal
        ph = np.minimum(th, phimax)
        w = np.cos(ph)[..., None] * gh + np.sin(ph)[..., None] * side
        w = np.where((th > phimax)[..., None], w, n)
        ceiling = th > np.radians(135.0)
        wn = np.maximum(np.einsum("nkd,nkd->nk", w, n), 0.2)
        Lw = H / wn                                          # along the axis, to height H off the skin
        along = np.einsum("nkd,nkd->nk", d, w)
        across = np.linalg.norm(d - along[..., None] * w, axis=-1)
        vc = Lw - R
        web = _box2(across, along - 0.5 * (vc - 1.0), 0.5 * t, 0.5 * (vc + 1.0))
        q = d - (vc[..., None] * w)                          # from the bulb's centre
        disk = np.linalg.norm(q, axis=-1) - R
        ca, sa = np.cos(np.radians(self.chin)), np.sin(np.radians(self.chin))
        ax = np.einsum("nkd,nkd->nk", q, gh) + R / ca        # along up, from the chin's apex
        rad = np.linalg.norm(q - np.einsum("nkd,nkd->nk", q, gh)[..., None] * gh, axis=-1)
        cone = rad * sa - ax * ca
        tear = np.minimum(disk, np.maximum(cone, ax - (R / ca - R * ca)))
        f = _smin(web, tear, 1.0)                           # the bulb blended into the web
        f = np.maximum(f, over)                             # ends at the line's end, never past it
        f = np.maximum(f, 0.3 - H)                          # under 0.3 mm tall it is nothing (no sliver line on the skin)
        f = np.where(valid & ~ceiling, f, 1e6)              # a ridge tapered to nothing is a sliver in the skin
        out = f[:, 0]
        for j in range(1, f.shape[1]):
            out = _smin(out, f[:, j], self.blend)
        return out


def section(kind="ridge", t=2.0, H=10.0, R=2.0, chin=45.0, skin=2.0, spacing=11.0, p=0.02, mass=None):
    """A ridge (or a plain wall, kind="wall") with its `spacing` of `skin` as a T beam:
    area (mm2 per ridge, its mass), the neutral axis, the second moment I (mm4) and the
    section modulus at the ridge's tip and at the skin. With `mass` (an area in mm2), a
    plain wall is made that area by its height at thickness t (what the ridge is
    compared against)."""
    u = np.arange(-spacing / 2, spacing / 2, p) + p / 2
    v = np.arange(-skin, H + 2 * R + 1, p) + p / 2
    U, V = np.meshgrid(u, v, indexing="ij")
    if kind == "wall":
        if mass is not None:
            H = (mass - skin * spacing) / t
        solid = (np.abs(U) <= t / 2) & (V <= H) & (V >= 0)
    else:
        solid = (profile(U, V, t, H, R, chin) <= 0) & (V >= 0)
    solid |= V < 0
    dA = p * p
    A = solid.sum() * dA
    yc = (V[solid]).sum() * dA / A
    I = ((V[solid] - yc) ** 2).sum() * dA
    top = V[solid].max()
    return {"area mm2": round(float(A), 2), "neutral axis mm": round(float(yc), 3), "I mm4": round(float(I), 1),
            "Z tip mm3": round(float(I / (top - yc)), 1), "Z skin mm3": round(float(I / (yc + skin)), 1),
            "height mm": round(float(top), 2)}
