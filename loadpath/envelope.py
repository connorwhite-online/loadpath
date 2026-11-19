"""
Envelope: a first exterior drawn round the parts, for the designer to take on.

Everything else in loadpath starts from an exterior somebody drew
(EXTERIOR.md: "the one thing loadpath never designs"). Before there is one,
a designer still wants to see the smallest, best-handling shape the parts
allow: which surfaces the stack fixes, where the hand goes, and how much
room a style has. This module builds that starting form as a signed
distance field from three kinds of input and checks it the way the
designer's own surfacing will be checked later:

- a style field, drawn with the primitives here (superellipse sweeps,
  smooth blends, view funnels) or any SDF;
- keep-outs: every part, grown by the skin and a clearance. `wrap` blends
  them into the style with a smooth union, so the form never cuts a part,
  and `pushes` reports where the parts pushed the style out (the places a
  designer should know are fixed);
- the hand: `grip_report` measures the grip's sections and the trigger
  reach against hand sizes (data/hands.json), and `view_check` casts rays
  through each camera's and laser's field of view to prove the form does
  not vignette them.

The result goes to `exterior.load_exterior` like any SDF and from there to
`loadpath enclosure` (mounts, assembly, loads, heat). It is a constraint-
aware starting form, not surfacing: subd.py turns it into editable
patches, and a designer's own surfacing replaces it through the same
intake.

All lengths are mm; fields take P (N, 3) and return (N,) distances,
negative inside. Distances away from the surface are estimates (first
order), which is all blending and offsets near the surface need.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .sdf_kit import smin, smax

# ---------------------------------------------------------------------------
# primitives


def _se(u, v, a, b, n):
    """Normalized superellipse distance estimate in a 2-D section: level 0 on
    |u/a|^n + |v/b|^n = 1, scaled to roughly mm near the curve."""
    a = np.maximum(a, 1e-6)
    b = np.maximum(b, 1e-6)
    r = (np.abs(u / a) ** n + np.abs(v / b) ** n) ** (1.0 / n)
    return (r - 1.0) * np.minimum(a, b)


def _catmull(points, samples=24):
    pts = np.asarray(points, float)
    if len(pts) < 3:
        t = np.linspace(0, 1, samples * (len(pts) - 1) + 1)[:, None]
        return pts[0] + t * (pts[-1] - pts[0])
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1])
    return np.asarray(out)


def sweep(P, spine, half_w, half_d, n=2.5, side=(1.0, 0.0, 0.0), cap=True, samples=24):
    """A superellipse section swept along a smooth spine (Catmull-Rom through
    `spine`). `half_w`, `half_d`: section half sizes, each a number or a list
    with one value per spine point (interpolated along the curve between
    the points, however unevenly they are spaced), across `side` and across
    side x tangent. A grip, a handle, a boom.

    The distance is measured in the plane of the nearest spine sample, so it
    is good near the surface of a gently curving sweep."""
    curve = _catmull(spine, samples)
    seg = np.diff(curve, axis=0)
    L = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))]
    s = L / L[-1]
    tang = np.gradient(curve, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    side = np.asarray(side, float)
    u = side - (tang @ side)[:, None] * tang
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(tang, u)
    # each spine point's place along the curve (it passes through point i
    # at sample i * samples), so sizes stay with their points however
    # unevenly the points are spaced
    ks = s[np.arange(len(spine)) * samples]
    hw = np.interp(s, ks, np.broadcast_to(np.asarray(half_w, float), (len(spine),)))
    hd = np.interp(s, ks, np.broadcast_to(np.asarray(half_d, float), (len(spine),)))
    P = np.asarray(P, float)
    # nearest sample (chunked brute force: spines have a few hundred samples)
    idx = np.empty(len(P), int)
    for i in range(0, len(P), 20000):
        d2 = ((P[i:i + 20000, None, :] - curve[None]) ** 2).sum(-1)
        idx[i:i + 20000] = d2.argmin(1)
    q = P - curve[idx]
    a = np.einsum("ij,ij->i", q, tang[idx])
    d = _se(np.einsum("ij,ij->i", q, u[idx]), np.einsum("ij,ij->i", q, v[idx]), hw[idx], hd[idx], n)
    if cap:
        ends = np.maximum(-(np.einsum("ij,j->i", P - curve[0], tang[0])),
                          np.einsum("ij,j->i", P - curve[-1], tang[-1]))
        d = np.maximum(d, ends)
    else:
        d = np.where((idx == 0) & (a < 0) | (idx == len(curve) - 1) & (a > 0), np.maximum(d, np.abs(a)), d)
    return d


def funnel(P, floor_center, axis, floor_half, half_angles, depth, up=(0, 0, 1), n=4.0):
    """A view funnel: a superellipse floor `floor_half` (across, up) at
    `floor_center`, opening along `axis` for `depth` with walls sloped at
    `half_angles` (degrees, across and up) from the axis. Cut it out of a
    body (smax(body, -funnel)) to set a window into a sloped recess whose
    walls stay outside a field of view."""
    P = np.asarray(P, float)
    ax = np.asarray(axis, float); ax /= np.linalg.norm(ax)
    up = np.asarray(up, float); up = up - (up @ ax) * ax; up /= np.linalg.norm(up)
    acr = np.cross(up, ax)
    q = P - np.asarray(floor_center, float)
    t = q @ ax
    tc = np.clip(t, 0, depth)
    ha = floor_half[0] + tc * np.tan(np.radians(half_angles[0]))
    hu = floor_half[1] + tc * np.tan(np.radians(half_angles[1]))
    d = _se(q @ acr, q @ up, ha, hu, n)
    cos = 1.0 / np.sqrt(1 + np.tan(np.radians(min(half_angles))) ** 2)
    d = d * cos
    return np.maximum(d, np.maximum(-t, t - depth))


# ---------------------------------------------------------------------------
# keep-outs and the wrap


class Keepout:
    """A part as the envelope sees it: its own SDF (any loadpath Shape, a
    trimesh, or a callable) and how far the outside must stay from it."""

    def __init__(self, name, shape, gap):
        self.name = name
        self.gap = float(gap)
        self._mesh = None
        if callable(shape) and not hasattr(shape, "sdf") and not hasattr(shape, "vertices"):
            self._f = shape
        elif hasattr(shape, "sdf"):
            self._f = shape.sdf
        else:   # a trimesh
            self._mesh = shape
            self._f = _grid_sdf(shape, reach=max(12.0, 3 * self.gap))

    def sdf(self, P):
        return self._f(np.asarray(P, float))

    def surface_points(self, n=4000):
        if self._mesh is not None:
            import trimesh
            pts, _ = trimesh.sample.sample_surface_even(self._mesh, n, seed=1)
            return np.asarray(pts)
        return None


def _grid_sdf(mesh, reach=12.0, pitch=None):
    """Signed distance to a closed trimesh (negative inside) from a voxel
    grid padded by `reach` mm, interpolated trilinearly. Accurate to about
    half a cell within `reach` of the part, which is where blends and gaps
    look; further out it returns a lower bound (the distance to the grid's
    box plus `reach`). A plain MeshShape pads its grid by a few cells only,
    so a keep-out read 10 mm off the part came back several mm short and
    bulged the wrap."""
    from scipy import ndimage
    from .verify import voxelize
    if pitch is None:
        pitch = float(np.clip(np.min(mesh.extents) / 8.0, 0.25, 0.6))
    solid, origin = voxelize(mesh, pitch)
    pad = int(np.ceil(reach / pitch)) + 2
    solid = np.pad(solid, pad)
    origin = np.asarray(origin, float) - pad * pitch
    dist = (ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)) * pitch
    dist -= 0.5 * pitch * np.sign(dist)
    dist = dist.astype(np.float32)
    shape = np.array(dist.shape)
    lo = origin + 0.5 * pitch
    hi = lo + (shape - 1) * pitch

    def f(P):
        P = np.asarray(P, float)
        idx = (P - lo) / pitch
        v = ndimage.map_coordinates(dist, idx.T, order=1, mode="nearest")
        out = np.maximum(np.maximum(lo - P, P - hi), 0.0)
        far = np.linalg.norm(out, axis=1)
        return np.where(far > 0, np.maximum(v, far + reach), v)
    return f


def keepout_field(keepouts):
    def f(P):
        P = np.asarray(P, float)
        return np.min([k.sdf(P) - k.gap for k in keepouts], axis=0)
    return f


def wrap(style, keepouts, k=6.0):
    """The style, with every keep-out blended in (smooth union, blend radius
    k): the form can only grow where a part needs room, never cut it."""
    ko = keepout_field(keepouts)

    def f(P):
        return smin(style(P), ko(P), k)
    return f


def pushes(style, wrapped, bounds, pitch=2.0, threshold=0.3):
    """Where the parts pushed the style out: grid nodes inside the wrapped
    form but outside the style by more than `threshold` mm, grouped into
    connected regions, each with its box and how far out it goes."""
    from scipy import ndimage
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 2)
    axes = [np.linspace(lo[i], lo[i] + pitch * (n[i] - 1), n[i]) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    s = style(G)
    w = wrapped(G)
    m = ((w < 0) & (s > threshold)).reshape(n)
    lab, k = ndimage.label(m)
    out = []
    for i in range(1, k + 1):
        cells = np.argwhere(lab == i)
        if len(cells) < 2:
            continue
        pts = lo + cells * pitch
        sv = s.reshape(n)[lab == i]
        out.append({"box": [pts.min(0).round(1).tolist(), pts.max(0).round(1).tolist()],
                    "out_mm": round(float(sv.max()), 1), "volume_cm3": round(len(cells) * pitch ** 3 / 1000, 2)})
    return sorted(out, key=lambda r: -r["volume_cm3"])


def clearances(f, keepouts, n=3000):
    """For each keep-out with a mesh: the smallest depth of its surface
    inside the form (-f), against the gap it asked for."""
    out = {}
    for k in keepouts:
        pts = k.surface_points(n)
        if pts is None:
            continue
        depth = -f(pts)
        out[k.name] = {"min_depth": round(float(depth.min()), 2), "asked": k.gap,
                       "ok": bool(depth.min() >= k.gap - 0.25)}
    return out


def view_check(f, apex, axis, half_angles, length=60.0, up=(0, 0, 1), rays=21, start=1.0, step=0.25):
    """Rays from `apex` (a lens's entrance pupil, a laser's aperture) through
    its field of view (`half_angles`, across and up, degrees): the fraction
    blocked, and the first blocked ray's direction. The apex is inside the
    solid form (behind its window); a ray is blocked if it never leaves the
    form within `length` mm or comes back into it after leaving. A window
    glass in the opening is not part of `f`, so a clear view reads 0."""
    ax = np.asarray(axis, float); ax /= np.linalg.norm(ax)
    up = np.asarray(up, float); up = up - (up @ ax) * ax; up /= np.linalg.norm(up)
    acr = np.cross(up, ax)
    blocked, worst = 0, None
    ta = np.tan(np.radians(np.linspace(-half_angles[0], half_angles[0], rays)))
    tu = np.tan(np.radians(np.linspace(-half_angles[1], half_angles[1], rays)))
    dirs = np.array([ax + a * acr + b * up for a in ta for b in tu])
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    ts = np.arange(start, length, step)
    pts = np.asarray(apex, float)[None, None, :] + ts[None, :, None] * dirs[:, None, :]
    v = f(pts.reshape(-1, 3)).reshape(len(dirs), len(ts))
    # the apex sits inside the solid form (behind its window): a ray is clear
    # once it has left the form through the opening and never comes back in
    out = v > 0
    left = np.maximum.accumulate(out, axis=1)
    hit = (left & ~out).any(1) | ~out.any(1)
    blocked = int(hit.sum())
    if blocked:
        i = int(np.argmax(hit))
        worst = np.degrees(np.arctan2(dirs[i] @ acr, dirs[i] @ ax)).round(1).item(), \
            np.degrees(np.arctan2(dirs[i] @ up, dirs[i] @ ax)).round(1).item()
    return {"rays": len(dirs), "blocked": blocked, "blocked_fraction": round(blocked / len(dirs), 3),
            "first_blocked_deg": worst}


# ---------------------------------------------------------------------------
# the hand

_HANDS = None


def hands():
    """Hand dimensions by percentile (data/hands.json, with its source)."""
    global _HANDS
    if _HANDS is None:
        with open(os.path.join(os.path.dirname(__file__), "data", "hands.json")) as fh:
            _HANDS = json.load(fh)
    return _HANDS


def grip_targets(hand_length):
    """Working targets for a power grip (cylindrical handle) and an index-
    finger trigger, from hand length (mm). Rules of thumb from the sources in
    data/hands.json, to be confirmed with foam mock-ups, not specs."""
    g = hands()["grip"]
    d = g["diameter_per_hand_length"] * hand_length
    return {"diameter": round(d, 1), "circumference": round(np.pi * d, 1),
            "trigger_reach": round(g["trigger_reach_per_hand_length"] * hand_length, 1)}


def grip_report(mesh, origin, axis, heights, trigger_face=None, backstrap_dir=None):
    """Sections of a grip, perpendicular to its `axis`, at `heights` (mm from
    `origin` along the axis): perimeter, width and depth, and the equivalent
    diameter (perimeter / pi). With `trigger_face` (a point) and
    `backstrap_dir` (unit vector, back of the grip): the trigger reach, the
    distance from the trigger face to the back of the grip at the trigger's
    height. Read against hand percentiles."""
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    origin = np.asarray(origin, float)
    secs = []
    for h in heights:
        sl = mesh.section(plane_origin=origin + h * axis, plane_normal=axis)
        if sl is None:
            secs.append({"at": h, "perimeter": None})
            continue
        path, _ = sl.to_2D()
        polys = path.polygons_full
        if not polys:
            continue
        poly = max(polys, key=lambda p: p.area)
        minx, miny, maxx, maxy = poly.bounds
        per = poly.exterior.length
        secs.append({"at": h, "perimeter": round(per, 1), "equivalent_diameter": round(per / np.pi, 1),
                     "extent": [round(maxx - minx, 1), round(maxy - miny, 1)]})
    out = {"sections": secs}
    if trigger_face is not None and backstrap_dir is not None:
        b = np.asarray(backstrap_dir, float); b /= np.linalg.norm(b)
        tf = np.asarray(trigger_face, float)
        # walk from the trigger face toward the back until we leave the grip
        loc, _, _ = mesh.ray.intersects_location([tf + 0.5 * b], [b])
        if len(loc):
            far = max(loc, key=lambda p: (p - tf) @ b)
            out["trigger_reach"] = round(float((far - tf) @ b), 1)
    rows = []
    for pct in hands()["percentiles"]:
        for sex in ("female", "male"):
            hl = hands()["hand_length"][sex][pct]
            t = grip_targets(hl)
            rows.append({"who": f"{sex} {pct}", "hand_length": hl, **t})
    out["targets"] = rows
    return out
