"""
Patterns of a body: copies of one field node laid out in a line or grid,
round an axis, mirrored across a plane, or along a curve, as one solid
(the field graph's `pattern` op, fields.py; nTop's Linear, Circular,
Mirror and Along Curve pattern blocks).

    "bosses": {"op": "pattern", "input": "boss", "type": "circular",
               "axis": "z", "center": [0, 0, 0], "count": 6}

Each copy is a rigid placement (R, t) of the input: the pattern's value at
P is the least of the input at R^T (P - t) over the copies, the union of
the copies (a smooth union with `blend`). The copies whose centres lie
nearest P are the only ones evaluated (`neighbors`, default 4; `exact`
evaluates all), so a pattern of 200 bosses costs about four bosses; for
copies that do not overlap much (the usual pattern) the nearest few hold
the one P is in.
"""
from __future__ import annotations

import numpy as np

__all__ = ["instances", "pattern_value"]


def _axis(a):
    if isinstance(a, str):
        sgn = -1.0 if a.startswith("-") else 1.0
        return sgn * np.eye(3)["xyz".index(a.lstrip("+-"))]
    v = np.asarray(a, float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("a pattern axis must not be zero")
    return v / n


def _rot(axis, angle):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _count(v, name):
    n = int(v)
    if n < 1 or n != v:
        raise ValueError(f"a pattern's {name} must be a whole number of at least 1")
    return n


def instances(spec, pivot=None):
    """[(R, t), ...] for a pattern spec (see the module docstring and the
    schema's pattern op): type linear (`step` [dx, dy, dz] and `count`, or
    `directions` [{step, count}, ...] for a grid), circular (`axis`,
    `center`, `count`, `angle` swept in degrees, default 360 evenly round;
    `rotate` false keeps each copy's orientation), mirror (`plane`
    {point, normal}; `keep` false drops the original), or curve (`path` a
    polyline, `count` copies evenly along it, `orient` turns each copy's x
    axis along the curve with z toward `up`)."""
    kind = spec.get("type", "linear")
    out = []
    if kind == "linear":
        dirs = spec.get("directions") or [{"step": spec.get("step"), "count": spec.get("count", 2)}]
        if not isinstance(dirs, list) or not 1 <= len(dirs) <= 3:
            raise ValueError("a linear pattern takes one to three directions")
        grids = []
        for d in dirs:
            if d.get("step") is None:
                raise ValueError("a linear pattern needs a 'step' [dx, dy, dz] per direction")
            st = np.asarray(d["step"], float).reshape(3)
            n = _count(d.get("count", 2), "count")
            grids.append([i * st for i in range(n)])
        offs = [np.zeros(3)]
        for g in grids:
            offs = [o + s for o in offs for s in g]
        out = [(np.eye(3), o) for o in offs]
    elif kind == "circular":
        ax = _axis(spec.get("axis", "z"))
        c = np.asarray(spec.get("center", (0.0, 0.0, 0.0)), float)
        n = _count(spec.get("count", 4), "count")
        sweep = float(spec.get("angle", 360.0))
        if abs(sweep) >= 360.0 - 1e-9:
            angles = [2 * np.pi * i / n for i in range(n)]
        else:
            angles = [np.radians(sweep) * i / max(n - 1, 1) for i in range(n)]
        for a in angles:
            R = _rot(ax, a)
            t = c - R @ c
            if spec.get("rotate", True):
                out.append((R, t))
            else:
                # carried round the circle, each copy keeping its orientation:
                # its pivot (the input's centre) goes where the turn takes it
                pv = np.asarray(spec.get("pivot", pivot if pivot is not None else c), float)
                out.append((np.eye(3), R @ (pv - c) + c - pv))
    elif kind == "mirror":
        pl = spec.get("plane") or {}
        if "normal" not in pl:
            raise ValueError("a mirror pattern needs plane: {point, normal}")
        nrm = _axis(pl["normal"])
        p0 = np.asarray(pl.get("point", (0.0, 0.0, 0.0)), float)
        M = np.eye(3) - 2 * np.outer(nrm, nrm)
        out = ([(np.eye(3), np.zeros(3))] if spec.get("keep", True) else []) + [(M, p0 - M @ p0)]
    elif kind == "curve":
        path = np.asarray(spec.get("path"), float).reshape(-1, 3)
        if len(path) < 2:
            raise ValueError("a curve pattern needs a 'path' of at least two points")
        n = _count(spec.get("count", 2), "count")
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        closed = bool(spec.get("closed", False))
        L = s[-1]
        at = np.linspace(0, L, n, endpoint=not closed) if n > 1 else np.array([0.0])
        up = _axis(spec.get("up", "z"))
        base = np.asarray(spec.get("anchor", path[0]), float)
        for a in at:
            j = min(int(np.searchsorted(s, a, side="right")) - 1, len(seg) - 1)
            u = (a - s[j]) / max(seg[j], 1e-12)
            p = path[j] + u * (path[j + 1] - path[j])
            if spec.get("orient", True):
                x = (path[j + 1] - path[j]) / max(seg[j], 1e-12)
                y = np.cross(up, x)
                if np.linalg.norm(y) < 1e-9:
                    y = np.cross(np.eye(3)[int(np.argmin(np.abs(x)))], x)
                y /= np.linalg.norm(y)
                R = np.stack([x, y, np.cross(x, y)], 1)
            else:
                R = np.eye(3)
            # the input's anchor point goes to p
            out.append((R, p - R @ base))
    else:
        raise ValueError(f"pattern type must be linear, circular, mirror or curve, not {kind!r}")
    if not out:
        raise ValueError("the pattern has no copies")
    return out


def pattern_value(fn, P, inst, center, neighbors=4, exact=False, blend=0.0, smin=None):
    """The union of fn over the copies at points P (n, 3): fn(Q) evaluated
    for each point's nearest `neighbors` copies (by where each copy puts
    `center`), or all of them with `exact`."""
    P = np.asarray(P, float)
    R = np.array([r for r, _ in inst])
    T = np.array([t for _, t in inst])
    m = len(inst)
    if exact or m <= neighbors:
        idx = np.broadcast_to(np.arange(m), (len(P), m))
    else:
        from scipy.spatial import cKDTree

        C = np.einsum("kij,j->ki", R, center) + T
        _, idx = cKDTree(C).query(P, k=neighbors)
        idx = np.asarray(idx).reshape(len(P), neighbors)
    k = idx.shape[1]
    # Q = R^T (P - t) for each point's copies
    D = P[:, None, :] - T[idx]
    Q = np.einsum("nkji,nkj->nki", R[idx], D)
    v = np.asarray(fn(Q.reshape(-1, 3)), float)
    v = np.broadcast_to(v, (len(P) * k,)).reshape(len(P), k)
    if blend and smin is not None:
        out = v[:, 0]
        for j in range(1, k):
            out = smin(out, v[:, j], blend)
        return out
    return v.min(axis=1)
