"""
Ribs on a face: walls of a set thickness standing a set height off a
solid's surface, laid out by curves projected onto that surface, by a
pattern (square grid, isogrid, radial spokes and rings), or along the
principal stress lines a study puts in the part.

The field graph's `ribs` op (fields.py) builds on these:

    "ribs": {"op": "ribs", "surface": "body", "thickness": 1.2, "height": 6,
             "paths": [[[0, -20, 10], [0, 20, 10]]], "direction": "z"}

A rib is the set of points whose wall distance is under half the
thickness and whose depth under the surface (side "inside", the default:
ribs on the inner face of an enclosure or a plate's back) or height over
it (side "outside") is under `height`. Two kinds of wall:

- `direction` given: the curve swept along that direction, the walls
  parallel to it. Ribs a mould or a print can pull straight, and the
  distance is exact (the 2D distance across the direction).
- no `direction`: walls normal to the surface, through the curve's
  projection onto it: the point's foot on the surface (one step down the
  surface field's gradient) against the projected curve. Ribs that stay
  square to a curved skin.

Stress lines (`stress_lines`): the stress tensor at the analysis nodes
from the displacement's local least-squares gradient, turned into the
surface's tangent plane, and traced both ways from the most stressed spot
not yet covered (evenly spaced streamlines, Jobard and Lefer), for the
largest principal stress, the other one, or both (a Michell-like grid).
Ribs along them carry the face's bending and membrane load the way the
material wants to carry it, the layout nTop's rib tools and topology
optimizers both point to.
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = ["pattern_distance", "polylines_distance", "project_paths", "stress_lines", "surface_tensors"]


def _axis(d):
    if isinstance(d, str):
        sgn = -1.0 if d.startswith("-") else 1.0
        return sgn * np.eye(3)["xyz".index(d.lstrip("+-"))]
    a = np.asarray(d, float).reshape(3)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("a rib direction must not be zero")
    return a / n


def _plane_basis(d):
    """Two unit vectors across direction d."""
    a = np.eye(3)[int(np.argmin(np.abs(d)))]
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _densify(path, step):
    p = np.asarray(path, float).reshape(-1, 3)
    if len(p) < 2:
        raise ValueError("a rib path needs at least two points")
    out = [p[0]]
    for a, b in itertools.pairwise(p):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.extend(a + t * (b - a))
    return np.array(out)


class _Segments:
    """Many short segments, nearest distance by a k-d tree over their
    midpoints and the exact segment distance for the k nearest."""

    def __init__(self, polylines, dim):
        from scipy.spatial import cKDTree

        A, B = [], []
        for pl in polylines:
            pl = np.asarray(pl, float)
            if len(pl) >= 2:
                A.append(pl[:-1])
                B.append(pl[1:])
        if not A:
            raise ValueError("no rib segments")
        self.a = np.concatenate(A)[:, :dim]
        self.b = np.concatenate(B)[:, :dim]
        self.tree = cKDTree(0.5 * (self.a + self.b))
        self.half = float(np.max(np.linalg.norm(self.b - self.a, axis=1))) * 0.5

    def __call__(self, Q, k=8):
        k = min(k, len(self.a))
        _, idx = self.tree.query(Q, k=k)
        idx = np.asarray(idx).reshape(len(Q), k)
        a, b = self.a[idx], self.b[idx]
        ab = b - a
        t = np.clip(np.einsum("nkd,nkd->nk", Q[:, None] - a, ab) / np.maximum(np.einsum("nkd,nkd->nk", ab, ab),
                                                                                 1e-300), 0.0, 1.0)
        d = np.linalg.norm(Q[:, None] - (a + t[..., None] * ab), axis=2)
        return d.min(axis=1)


def polylines_distance(polylines, direction=None, step=1.0):
    """callable(P) -> distance to the polylines: across `direction` (the
    curves swept along it) when given, else in 3D."""
    dense = [_densify(p, step) for p in polylines]
    if direction is None:
        seg = _Segments(dense, 3)
        return lambda P: seg(np.asarray(P, float))
    d = _axis(direction)
    u, v = _plane_basis(d)
    M = np.stack([u, v], 1)
    seg = _Segments([p @ M for p in dense], 2)
    return lambda P: seg(np.asarray(P, float) @ M)


def pattern_distance(spec):
    """callable(P) -> distance to a rib pattern's walls (all parallel to
    the pattern's `direction`, default z): type square (two families of
    lines `spacing` apart), triangle (an isogrid: three at 60 degrees),
    radial (`spokes` lines through `center` and rings `spacing` apart),
    turned by `angle` degrees about the direction, through `origin`."""
    kind = spec.get("type", "square")
    s = float(spec.get("spacing", 10.0))
    if s <= 0:
        raise ValueError("a rib pattern's spacing must be positive")
    d = _axis(spec.get("direction", "z"))
    u, v = _plane_basis(d)
    if abs(d[2]) > 0.9:            # keep x and y as the pattern's axes when looking down z
        u = np.array([1.0, 0.0, 0.0]) - d * d[0]
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
    o = np.asarray(spec.get("origin", spec.get("center", (0.0, 0.0, 0.0))), float)
    a0 = np.radians(float(spec.get("angle", 0.0)))

    def lines(x, y, angles):
        out = None
        for a in angles:
            r = -x * np.sin(a) + y * np.cos(a)
            dd = np.abs(r - s * np.round(r / s))
            out = dd if out is None else np.minimum(out, dd)
        return out

    def f(P):
        Q = np.asarray(P, float) - o
        x, y = Q @ u, Q @ v
        if kind == "square":
            return lines(x, y, [a0, a0 + np.pi / 2])
        if kind in ("triangle", "isogrid"):
            return lines(x, y, [a0, a0 + np.pi / 3, a0 + 2 * np.pi / 3])
        if kind == "radial":
            rho = np.hypot(x, y)
            n = int(spec.get("spokes", 8))
            out = np.full(len(Q), np.inf)
            if n > 0:
                th = np.arctan2(y, x) - a0
                k = np.round(th / (2 * np.pi / n))
                dth = th - k * 2 * np.pi / n
                # distance to the nearest spoke (a ray from the centre)
                out = np.where(np.abs(dth) < np.pi / 2, rho * np.abs(np.sin(dth)), rho)
            if spec.get("rings", True):
                r0 = float(spec.get("innerRadius", 0.0))
                rr = rho - r0
                ring = np.abs(rr - s * np.round(rr / s))
                ring = np.where(rr < -0.5 * s, np.inf, ring)
                out = np.minimum(out, ring)
            return out
        raise ValueError(f"rib pattern type must be square, triangle or radial, not {kind!r}")

    return f


def gradient(fn, P, h):
    """Central-difference gradient of a scalar field."""
    P = np.asarray(P, float)
    n = len(P)
    off = np.concatenate([P + h * e for e in np.eye(3)] + [P - h * e for e in np.eye(3)])
    v = np.asarray(fn(off), float)
    return np.stack([(v[i * n:(i + 1) * n] - v[(i + 3) * n:(i + 4) * n]) / (2 * h) for i in range(3)], 1)


def foot(fn, P, h, f0=None, rounds=1):
    """P moved onto the zero set along the field's gradient, and the unit
    normal there (outward for a solid's signed distance)."""
    Q = np.asarray(P, float).copy()
    f = np.asarray(fn(Q), float) if f0 is None else np.asarray(f0, float)
    for i in range(rounds):
        g = gradient(fn, Q, h)
        gg = np.maximum(np.einsum("ij,ij->i", g, g), 1e-12)
        Q = Q - (f / gg)[:, None] * g
        if i + 1 < rounds:
            f = np.asarray(fn(Q), float)
    g = gradient(fn, Q, h)
    return Q, g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def project_paths(paths, fn, h, direction=None, step=1.0, reach=1e3):
    """Polylines dropped onto the surface fn = 0: along `direction` (the
    first crossing from each point, either way along it) or to the nearest
    surface point. Points that find no surface within `reach` are dropped
    (the curve splits there)."""
    out = []
    for p in paths:
        P = _densify(p, step)
        if direction is not None:
            d = _axis(direction)
            ts = np.linspace(-reach, reach, 801)
            Q = P[:, None] + ts[None, :, None] * d
            v = np.asarray(fn(Q.reshape(-1, 3)), float).reshape(len(P), len(ts))
            hit = np.full(len(P), np.nan)
            for i in range(len(P)):
                c = np.nonzero(np.sign(v[i, :-1]) != np.sign(v[i, 1:]))[0]
                if len(c):
                    j = c[np.argmin(np.abs(ts[c]))]
                    t0, t1, f0, f1 = ts[j], ts[j + 1], v[i, j], v[i, j + 1]
                    hit[i] = t0 - f0 * (t1 - t0) / (f1 - f0)
            ok = np.isfinite(hit)
            P = np.where(ok[:, None], P + np.nan_to_num(hit)[:, None] * d, np.nan)
        else:
            ok = np.ones(len(P), bool)
        Q = P.copy()
        if ok.any():
            Q[ok] = foot(fn, P[ok], h, rounds=3)[0]
        # split where a point missed
        run = []
        for q, k in zip(Q, ok):
            if k:
                run.append(q)
            elif len(run) >= 2:
                out.append(np.array(run))
                run = []
            else:
                run = []
        if len(run) >= 2:
            out.append(np.array(run))
    if not out:
        raise ValueError("no rib path reaches the surface")
    return out


# ---- stress lines -----------------------------------------------------------

def surface_tensors(nodes, disp, nu=0.35, k=19):
    """Per node the (unscaled) stress tensor from the displacement's local
    linear least-squares gradient over its k nearest nodes."""
    from scipy.spatial import cKDTree

    X = np.asarray(nodes, float)
    U = np.asarray(disp, float)
    tree = cKDTree(X)
    k = min(k, len(X))
    _, idx = tree.query(X, k=k)
    dX = X[idx] - X[:, None]
    dU = U[idx] - U[:, None]
    A = np.einsum("nki,nkj->nij", dX, dX) + 1e-9 * np.eye(3)
    B = np.einsum("nki,nkj->nij", dX, dU)
    G = np.linalg.solve(A, B)                     # G[i, j] = d u_j / d x_i
    eps = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 0.5 / (1 + nu)
    tr = np.trace(eps, axis1=1, axis2=2)
    return lam * tr[:, None, None] * np.eye(3) + 2 * mu * eps


class _Tensors:
    def __init__(self, X, S):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(X)
        self.S = S

    def __call__(self, Q, k=8):
        d, idx = self.tree.query(Q, k=k)
        w = 1.0 / np.maximum(d, 1e-6) ** 2
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkij->nij", w, self.S[idx])


def _tangent_dirs(S, n):
    """The two principal directions of S in the plane across n, with their
    stresses, the larger |stress| first."""
    Pt = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
    T = Pt @ S @ Pt
    w, V = np.linalg.eigh(T)
    # drop the eigenvector along the normal
    drop = np.argmax(np.abs(np.einsum("nij,ni->nj", V, n)), axis=1)
    keep = np.array([[j for j in range(3) if j != dd] for dd in drop])
    rows = np.arange(len(n))[:, None]
    w2 = w[rows, keep]
    V2 = np.transpose(V, (0, 2, 1))[rows, keep]
    order = np.argsort(-np.abs(w2), axis=1)
    return np.take_along_axis(w2, order, 1), np.take_along_axis(V2, order[..., None], 1)


def stress_lines(fn, fields, spacing, h, family="both", cutoff=0.15, case=None, nu=0.35, step=None,
                 max_length=None, bounds=None, merge=None, bridge=None):
    """Evenly spaced principal stress lines on the surface fn = 0, from a
    verify_part fields dict. family: "max" (the larger |principal stress|
    in the surface), "min" (the other), "both". Lines stop where the stress
    stays under `cutoff` times the surface's 95th percentile for more than
    `bridge` mm (default 1.5 spacings; shorter stretches are crossed
    straight, as a rib must cross a bending moment's zero), where they run
    into a line of the same family (within `merge`, default 0.15 spacing:
    they join it, as ribs converging on a load do), at `bounds`
    ((lo, hi): only the surface inside is seeded and traced, so one face
    of a part can be picked),
    or at a turn of more than 45 degrees in one step (a singular point)."""
    from scipy.spatial import cKDTree

    X = np.asarray(fields["nodes"], float)
    cases = fields["cases"]
    if case is not None:
        cases = [c for c in cases if c["name"] == case]
        if not cases:
            raise ValueError(f"no load case {case!r} in the analysis")
    else:
        # the case with the highest stress
        cases = [max(cases, key=lambda c: float(np.max(c["vonMises"])))]
    S = surface_tensors(X, cases[0]["displacement"], nu=nu)
    pitch = float(np.median(cKDTree(X).query(X[:2000], k=2)[0][:, 1]))
    fx = np.asarray(fn(X), float)
    near = np.abs(fx) < 1.5 * pitch
    if not near.any():
        raise ValueError("no analysis node lies on the rib surface")
    lo, hi = (np.asarray(b, float) for b in bounds) if bounds is not None else (X.min(0), X.max(0))
    lo, hi = lo - 0.5 * pitch, hi + 0.5 * pitch
    near &= np.all((X >= lo) & (X <= hi), axis=1)
    if not near.any():
        raise ValueError("no analysis node on the rib surface inside the stress lines' bounds")
    Xs, Ss = X[near], S[near]
    ten = _Tensors(Xs, Ss)
    foot_s, n_s = foot(fn, Xs, h, f0=fx[near], rounds=2)
    w, _ = _tangent_dirs(Ss, n_s)
    mag = np.abs(w[:, 0])
    ref = float(np.percentile(mag, 95)) or 1.0
    floor = cutoff * ref
    step = step or 0.25 * spacing
    merge = 0.15 * spacing if merge is None else merge
    bridge = 1.5 * spacing if bridge is None else bridge
    max_steps = int((max_length or 4 * np.linalg.norm(hi - lo)) / step)
    fams = {"max": [0], "min": [1], "both": [0, 1]}[family]
    out = []
    for fam in fams:
        pts = []                        # this family's points so far
        tree = None
        order = np.argsort(-mag)
        seeds = foot_s[order][mag[order] >= floor]
        lines = []

        def dirs(Q, ref_dir=None, fam=fam):
            Qf, nn = foot(fn, Q, h, rounds=1)
            ww, VV = _tangent_dirs(ten(Qf), nn)
            v = VV[:, fam]
            if ref_dir is not None:
                v = v * np.sign(np.einsum("ij,ij->i", v, ref_dir) + 1e-30)[:, None]
            return Qf, v, np.abs(ww[:, 0])

        def free(q, tree, sep):
            return tree is None or not tree.query_ball_point(q, sep)

        for sd in seeds:
            if not free(sd, tree, spacing):
                continue
            q0, v0, _ = dirs(sd[None])
            halves = []
            for sgn in (1.0, -1.0):
                q, v = q0[0], sgn * v0[0]
                run, bridged = [], 0
                for _ in range(max_steps):
                    _mid, vm, m0 = dirs((q + 0.5 * step * v)[None], v[None])
                    q1, v1, m1 = dirs((q + step * vm[0])[None], v[None])
                    q1, v1 = q1[0], v1[0]
                    if min(m0[0], m1[0]) < floor:
                        # a low-stress stretch (a bending moment's zero, say):
                        # carry straight on, the way the line was going, for
                        # up to `bridge` mm, so a rib reaches past it
                        q1 = foot(fn, (q + step * v)[None], h)[0][0]
                        v1 = v
                        bridged += 1
                        if bridged * step > bridge:
                            del run[len(run) - bridged + 1:]
                            break
                    else:
                        bridged = 0
                        if v1 @ v < np.cos(np.radians(45)):
                            break
                    if np.any(q1 < lo) or np.any(q1 > hi):
                        break
                    if not free(q1, tree, merge):
                        run.append(q1)       # runs into another line: joins it
                        break
                    if len(run) > 8 and np.linalg.norm(q1 - q0[0]) < 0.5 * step:
                        break            # closed on itself
                    run.append(q1)
                    q, v = q1, v1
                halves.append(run)
            line = np.array(halves[1][::-1] + [q0[0]] + halves[0])
            if len(line) * step < 2 * spacing:
                continue
            lines.append(line)
            pts.extend(line)
            tree = cKDTree(np.array(pts))
        out.extend(lines)
    if not out:
        raise ValueError("no stress line long enough: lower cutoff or spacing")
    return out
