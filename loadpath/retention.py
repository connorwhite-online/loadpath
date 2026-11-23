"""
Retention: is every part held once the product is closed?

A way in (handoff.assembly_paths, enclosure) proves a part can get to its
place. It says nothing about whether the part stays there: a button cap
with no lip under the skin goes in fine and falls out of its opening the
first time the product is turned over. This is the check the other way
round. With everything assembled, each part is nudged a short way (2 mm
by default) along a full sphere of directions (about 100, a Fibonacci
sphere) and turned the same distance about axes through its centre (26
axes, both ways), and every motion is stepped until something stops it:

- **blocked**: some point of the part, or of what is round it, goes
  deeper into the other than it started (`tol`, 0.05 mm) before the part
  has moved the full nudge;
- **free**: it gets the whole way.

A part in a close-fitting hole escapes only along the hole's axis, which a
sphere of directions misses (the nearest of 100 is 11 degrees off, and
0.4 mm sideways at 2 mm hits a 0.1 mm gap). So the directions also
include the world axes, the part's own principal axes, and any way out
given (`ways`: the reverse of the part's way in, which by construction
was open before the product closed).

A part passes when no motion is free, or when what holds it is declared
(`held`): a screw, a bolt, a rivet, a pin, glue, tape, a snap, a press fit,
a weld, solder, a clamp, a magnet, a cable tie, or another part it is
installed on (`installed-on`, which then carries it). A flexible part (a
cable) is declared `cable`. Anything else that moves freely fails, with
the directions it goes, so the report says which way it falls out.
Undeclared and blocked is the goal: geometry that holds a part is a
fastening that cannot be forgotten in assembly.

A declared part is still nudged, and the report says whether it is also
held by geometry alone (`blockedAnyway`), which is what a reviewer wants
to know about a press fit or a glued part.

The motion is rigid. Contact that starts in overlap (a pad squeezed
between bosses, a part modelled touching its seat) does not count as a
hit unless the motion drives it deeper, so a part resting on a face can
still lift off it. Points farther from everything than the nudge cannot
hit anything and are dropped first, so a product of a few dozen parts is
checked in seconds to minutes.

    from loadpath.retention import Body, body_from_mesh, nudge_check
    bodies = [body_from_mesh("cap", cap_mesh), body_from_mesh("shell", shell_mesh)]
    rep = nudge_check(bodies, held={"shell": "screw"})
    rep["ok"], rep["bodies"]["cap"]["free"]

`enclosure.design_and_check` runs it on every delivered product, so a
split shell's components are checked against both halves as built.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["Body", "FASTENINGS", "body_from_mesh", "body_from_sdf", "fibonacci_sphere", "nudge_check"]

FASTENINGS = ("screw", "bolt", "rivet", "pin", "glue", "tape", "snap", "press-fit", "weld", "solder",
              "clamp", "magnet", "cable-tie", "installed-on", "cable")


@dataclass
class Body:
    """A rigid part: its signed distance (negative inside, mm) and points
    that stand for it (its surface, densely enough that nothing thinner than
    the spacing slips between them)."""
    name: str
    sdf: Callable[[np.ndarray], np.ndarray]
    points: np.ndarray

    def bounds(self):
        return self.points.min(axis=0), self.points.max(axis=0)


def _mesh_sdf(mesh):
    from .shapes import MeshShape
    sh = MeshShape(mesh)
    sh._GRID_CELLS = 320
    return sh.sdf


def body_from_mesh(name: str, mesh, samples: int = 6000, sdf=None, seed: int = 0) -> Body:
    """A body from a closed mesh: surface samples (area-weighted, plus the
    vertices, so sharp corners are there) and the mesh's distance, or `sdf`
    if one is given (a field the mesh was made from is exact)."""
    import trimesh
    from .integrity import require_one_body
    require_one_body(mesh, f"retention body {name}")
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=int(rng.integers(1 << 31)))
    v = np.asarray(mesh.vertices)
    if len(v) > samples:
        v = v[rng.choice(len(v), samples, replace=False)]
    return Body(name, sdf if sdf is not None else _mesh_sdf(mesh), np.vstack([pts, v]).astype(float))


def body_from_sdf(name: str, sdf, lo, hi, pitch: float = 0.4, samples: int = 6000, seed: int = 0) -> Body:
    """A body from a distance field alone: its zero level meshed at `pitch`
    inside the box (lo, hi) for the points."""
    from .exterior import mesh_from_sdf
    mesh = mesh_from_sdf(sdf, (tuple(lo), tuple(hi)), pitch)[0]
    return body_from_mesh(name, mesh, samples, sdf=sdf, seed=seed)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _kind(decl) -> str | None:
    if decl is None:
        return None
    k = decl.get("by") if isinstance(decl, dict) else str(decl)
    k = str(k).strip().lower().replace(" ", "-").replace("_", "-")
    k = {"screws": "screw", "bolts": "bolt", "pins": "pin", "glued": "glue", "adhesive": "glue",
         "press": "press-fit", "pressfit": "press-fit", "snaps": "snap", "installedon": "installed-on"}.get(k, k)
    if k not in FASTENINGS:
        raise ValueError(f"held by {k!r}: a fastening is one of {', '.join(FASTENINGS)}")
    return k


def nudge_check(bodies: list[Body], held: dict | None = None, nudge: float = 2.0, directions: int = 100,
                axes: int = 26, tol: float = 0.05, step: float | None = None, groups: dict | None = None,
                skip: tuple = (), ways: dict | None = None) -> dict:
    """Nudge every body against all the others; see the module docstring.

    held: {body: kind or {"by": kind, "note": ...}}, kinds in FASTENINGS.
    groups: {name: [body, ...]}: bodies fastened to each other that move as
    one (an assembly screwed together); the group is nudged as a whole and
    its own declaration, if any, is under the group's name. skip: bodies
    used only as obstacles (never nudged themselves). ways: {unit: [direction, ...]}
    more directions to try for that unit (its way in, reversed).
    Returns {"ok", "bodies": {name: {...}}, "failed": [names], "settings"}."""
    held = dict(held or {})
    groups = {g: list(m) for g, m in (groups or {}).items()}
    by = {b.name: b for b in bodies}
    for g, mem in groups.items():
        bad = [m for m in mem if m not in by]
        if bad:
            raise ValueError(f"group {g} names {bad}, which are not bodies")
    member_of = {m: g for g, mem in groups.items() for m in mem}
    units = {g: mem for g, mem in groups.items()}
    for b in bodies:
        if b.name not in member_of:
            units[b.name] = [b.name]
    step = float(step or min(0.25, nudge / 8.0))
    ts = np.arange(step, nudge + 1e-9, step)
    dirs = fibonacci_sphere(directions)
    rax = fibonacci_sphere(2 * axes)
    rax = rax[rax[:, 2] >= 0][:axes] if axes else np.zeros((0, 3))
    out, failed = {}, []
    for u, mem in units.items():
        if u in skip or all(m in skip for m in mem):
            continue
        kind = _kind(held.get(u))
        own = [by[m] for m in mem]
        others = [b for b in bodies if b.name not in mem]
        P = np.vstack([b.points for b in own])
        lo, hi = P.min(axis=0) - (nudge + 1.0), P.max(axis=0) + (nudge + 1.0)

        def own_sdf(X):
            return np.min([b.sdf(X) for b in own], axis=0)
        # what is near: the others' points in reach, and the others in reach
        near, Q = [], []
        for b in others:
            blo, bhi = b.bounds()
            if np.any(bhi < lo) or np.any(blo > hi):
                continue
            q = b.points[np.all((b.points >= lo) & (b.points <= hi), axis=1)]
            near.append(b)
            if len(q):
                Q.append(q)
        if not near:
            out[u] = {"members": mem, "held": kind, "free": {"translations": len(dirs), "rotations": 2 * len(rax)},
                      "freeDirections": [[1, 0, 0]], "blockedAnyway": False,
                      "ok": kind is not None, "why": "nothing within reach of it"}
            if kind is None:
                failed.append(u)
            continue
        Q = np.vstack(Q) if Q else np.zeros((0, 3))

        def env(X):
            return np.min([b.sdf(X) for b in near], axis=0)
        d0 = env(P)
        keep = d0 <= nudge + tol
        P, d0 = P[keep], d0[keep]
        if len(Q):
            e0 = own_sdf(Q)
            kq = e0 <= nudge + tol
            Q, e0 = Q[kq], e0[kq]
        else:
            e0 = np.zeros(0)
        base_p = np.maximum(-d0, -tol)          # penetration allowed without a hit: where it starts
        base_q = np.maximum(-e0, -tol)
        allp = np.vstack([b.points for b in own])
        c = allp.mean(axis=0)
        rad = max(float(np.linalg.norm(allp - c, axis=1).max()), 1e-6)
        _, _, vt = np.linalg.svd(allp - c, full_matrices=False)
        extra = [np.eye(3), vt] + [np.atleast_2d(np.asarray(w, float)) for w in (ways or {}).get(u, [])]
        extra = np.vstack(extra)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        udirs = np.vstack([dirs, extra, -extra])

        def first_hit(move, unmove):
            for t in ts:
                if len(P) and np.any(-env(move(P, t)) > base_p + tol):
                    return float(t)
                if len(Q) and np.any(-own_sdf(unmove(Q, t)) > base_q + tol):
                    return float(t)
            return None
        free_t, play = [], []
        for d in udirs:
            h = first_hit(lambda X, t, d=d: X + d * t, lambda X, t, d=d: X - d * t)
            if h is None:
                free_t.append(d)
            else:
                play.append(h)
        free_r = []
        for a in rax:
            for sg in (1.0, -1.0):
                def mv(X, t, a=a, sg=sg):
                    return (X - c) @ _rot(a, sg * t / rad).T + c

                def un(X, t, a=a, sg=sg):
                    return (X - c) @ _rot(a, -sg * t / rad).T + c
                h = first_hit(mv, un)
                if h is None:
                    free_r.append((sg * a).tolist())
                else:
                    play.append(h)
        n_free = len(free_t) + len(free_r)
        rec = {"members": mem, "held": kind,
               "free": {"translations": len(free_t), "rotations": len(free_r)},
               "freeDirections": [np.round(d, 3).tolist() for d in free_t[:6]],
               "freeRotationAxes": [np.round(a, 3).tolist() for a in free_r[:6]],
               "play": {"min": round(min(play), 3) if play else None, "max": round(max(play), 3) if play else None},
               "blockedAnyway": n_free == 0}
        rec["ok"] = n_free == 0 or kind is not None
        if not rec["ok"]:
            rec["why"] = (f"moves {nudge:g} mm freely in {len(free_t)} of {len(udirs)} directions and "
                          f"{len(free_r)} of {2 * len(rax)} turns, and nothing is declared to hold it")
            failed.append(u)
        out[u] = rec
    return {"ok": not failed, "failed": failed, "bodies": out,
            "settings": {"nudge": nudge, "directions": len(dirs), "turnAxes": len(rax), "tol": tol, "step": step}}
