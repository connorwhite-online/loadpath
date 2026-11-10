"""
Rigid-body assembly motions: a part's way in as slides and turns.

A way in used to be straight legs only, the part never turning. A board
longer than the opening it goes through does not fit that way, though it
goes in easily by hand: one end first under the rim at a tilt, then laid
flat. Here a path is a list of legs, as the part leaves its place:

- a slide `(d, length)`: straight along the unit vector `d`, `length` mm
  (None on the last leg: on out of the exterior's bounds);
- a `Turn(axis, at, angle, along)`: `angle` degrees about the line through
  `at` along `axis` (right-handed), sliding `along` mm down that line as it
  turns (a pivot about an edge; 0 for a pure turn).

In JSON (a keep-out's `insertFrom`) a slide is `{"toward", "length"}` or a
bare axis, a turn `{"turn": degrees, "about": axis, "at": [x, y, z],
"along": mm}`.

What is here: walking a set of points along such a path against obstacles
(`path_check`, what handoff._path_check always did for slides), the pose at
each step (`poses`), reducing a dense route found on a grid
(placement.insertion_motion) to a few slides and pivots that are each
checked exactly (`simplify`), and the numbered steps a person would follow
to put the product together (`steps`).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["Turn", "rotation", "path_check", "poses", "simplify", "settler", "legs_from_spec", "legs_to_spec", "steps",
           "max_tilt"]

_AX = {"x": 0, "y": 1, "z": 2}


class Turn(NamedTuple):
    """A turn about the line through `at` along `axis`, `angle` degrees
    (right-handed, as the part leaves), sliding `along` mm down the axis."""
    axis: np.ndarray
    at: np.ndarray
    angle: float
    along: float = 0.0


def _vec(side) -> np.ndarray:
    if isinstance(side, str):
        s = side.strip().lower()
        d = np.zeros(3)
        d[_AX[s.lstrip("+-")]] = -1.0 if s.startswith("-") else 1.0
        return d
    d = np.asarray(side, float)
    return d / np.linalg.norm(d)


def _name(d):
    """'+z' for an axis direction; any other direction stays a vector."""
    d = np.asarray(d, float)
    k = int(np.argmax(np.abs(d)))
    if abs(d[k]) < 1 - 1e-6:
        return np.round(d / np.linalg.norm(d), 6).tolist()
    return ("+" if d[k] > 0 else "-") + "xyz"[k]


def rotation(axis, degrees: float) -> np.ndarray:
    """The rotation matrix of `degrees` about `axis` (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(degrees)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _turned(pose, leg: Turn, s: float = 1.0):
    """The pose after the share `s` of a turn: x' = Q (x - at) + at + s along axis."""
    R, t = pose
    Q = rotation(leg.axis, s * leg.angle)
    at = np.asarray(leg.at, float)
    return Q @ R, Q @ (t - at) + at + s * leg.along * np.asarray(leg.axis, float)


def _is_turn(leg) -> bool:
    return isinstance(leg, Turn)


def _exit_length(pts, pose, d, exit_lo, exit_hi) -> float:
    """How far along d until every point is past the exit box."""
    R, t = pose
    corners = np.array([[x, y, z] for x in (exit_lo[0], exit_hi[0]) for y in (exit_lo[1], exit_hi[1])
                        for z in (exit_lo[2], exit_hi[2])])
    return float((corners @ d).max() - ((pts @ R.T + t) @ d).min()) + 0.5


def poses(legs, pts=None, exit_lo=None, exit_hi=None, step: float = 0.25, start=None, reach: float | None = None):
    """Every pose (R, t, mm travelled) a part passes through along `legs`,
    `step` mm apart at its farthest point: x = R p + t for a point p of the
    part where it sits. A last slide with no length runs until the points
    are past the exit box (pts and the box are needed for it, and for how
    far the part's farthest point is from a turn's axis, unless `reach`)."""
    R, t = (np.eye(3), np.zeros(3)) if start is None else (np.asarray(start[0], float), np.asarray(start[1], float))
    dist = 0.0
    for leg in legs:
        if _is_turn(leg):
            a, at = np.asarray(leg.axis, float), np.asarray(leg.at, float)
            if reach is not None:
                r = float(reach)
            else:
                q = pts @ R.T + t - at
                r = float(np.linalg.norm(q - np.outer(q @ a, a), axis=1).max())
            arc = abs(np.radians(leg.angle)) * r
            n = max(int(np.ceil(max(arc, abs(leg.along)) / step)), 1)
            for s in np.linspace(0.0, 1.0, n + 1):
                R2, t2 = _turned((R, t), leg, s)
                yield R2, t2, dist + s * max(arc, abs(leg.along))
            R, t = _turned((R, t), leg)
            dist += max(arc, abs(leg.along))
        else:
            d, length = leg
            d = np.asarray(d, float)
            if length is None:
                length = _exit_length(pts, (R, t), d, exit_lo, exit_hi)
            n = max(int(np.ceil(length / step)), 1)
            for s in np.linspace(0.0, length, n + 1):
                yield R, t + s * d, dist + s
            t = t + length * d
            dist += length


def path_check(pts, penetration, legs, exit_lo, exit_hi, tol: float, step: float = 0.25, start=None):
    """Move the points along the legs; a last slide with no length runs
    until every point is outside the exit box. `penetration(q)` gives each
    point's depth into what is in the way (mm, > 0 inside) and which
    obstacle. Returns (clear, worst penetration, obstacle index or None,
    where (point), mm travelled)."""
    worst, where, who_w, dist = 0.0, None, None, 0.0
    for R, t, s in poses(legs, pts, exit_lo, exit_hi, step, start):
        q = pts @ R.T + t
        pen, who = penetration(q)
        j = int(np.argmax(pen))
        if pen[j] > worst:
            worst, where, who_w, dist = float(pen[j]), q[j], int(who[j]), float(s)
        if worst > tol:
            return False, worst, who_w, where, dist
    return True, worst, who_w, where, dist


def end_pose(legs, pts=None, exit_lo=None, exit_hi=None, start=None):
    """The pose at the end of the legs (a last slide with no length ends
    past the exit box when pts and the box are given, else is left out)."""
    R, t = (np.eye(3), np.zeros(3)) if start is None else (np.asarray(start[0], float), np.asarray(start[1], float))
    for leg in legs:
        if _is_turn(leg):
            R, t = _turned((R, t), leg)
        else:
            d, length = leg
            if length is None:
                if pts is None:
                    continue
                length = _exit_length(pts, (R, t), np.asarray(d, float), exit_lo, exit_hi)
            t = t + length * np.asarray(d, float)
    return R, t


def max_tilt(legs) -> float:
    """The most a part is turned from how it sits, anywhere along the legs (degrees)."""
    R, worst = np.eye(3), 0.0
    for leg in legs:
        if _is_turn(leg):
            R = rotation(leg.axis, leg.angle) @ R
            worst = max(worst, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    return worst


# --- a dense route reduced to a few moves -----------------------------------

def _between(A, B):
    """The one move from pose A to pose B when both are turned about the
    same axis direction: a slide when they are turned alike, else a turn
    about the pole (the line that stays put), sliding down it if need be."""
    RA, tA = A
    RB, tB = B
    Q = RB @ RA.T
    ang = np.degrees(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1)))
    if ang < 1e-6:
        u = tB - tA
        n = float(np.linalg.norm(u))
        return None if n < 1e-9 else (u / n, n)
    a = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    a = a / np.linalg.norm(a)
    u = tB - Q @ tA                       # x' = Q x + u
    along = float(u @ a)
    up = u - along * a
    th = np.radians(ang)
    at = 0.5 * (up + np.cross(a, up) / np.tan(th / 2))
    return Turn(a, at, float(ang), along)


def simplify(way, clear, settle=None, tries=(1.0, 0.7, 0.5, 0.35, 0.25, 0.17, 0.12, 0.08, 0.05, 0.03, 0.015),
             centre=None, reach=None):
    """A dense route (poses (R, t), a grid cell or a tilt step apart, the
    first where the part sits and the last out) as the fewest moves this
    finds: from each pose the farthest later pose one slide or one pivot
    reaches with `clear(leg, start_pose)` true. A grid's poses are only
    good to a cell: `settle(pose)` gives the clear pose nearest one (or
    None), and the moves run between settled poses. None if some step of
    the route cannot be made."""
    i, n = 0, len(way) - 1
    cur = way[0]
    anchors = [cur]
    settled = {}

    def target(j):
        if j not in settled:
            settled[j] = way[j] if settle is None else settle(way[j])
        return settled[j]

    while i < n:
        cands = sorted({min(n, i + max(1, int(np.ceil((n - i) * f)))) for f in tries} | {i + 1}, reverse=True)
        for j in cands:
            tgt = target(j)
            if tgt is None:
                continue
            leg = _between(cur, tgt)
            if leg is None or clear(leg, cur):
                break
        else:
            return None
        if leg is not None:
            anchors.append(tgt)
        cur, i = tgt, j
    # the stops were picked from a few along the route: any run of moves
    # that one move replaces goes
    i = 0
    while i < len(anchors) - 2:
        for k in range(len(anchors) - 1, i + 1, -1):
            leg = _between(anchors[i], anchors[k])
            if leg is not None and clear(leg, anchors[i]):
                del anchors[i + 1:k]
                break
        i += 1
    legs = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        leg = _between(a, b)
        if _is_turn(leg) and centre is not None and reach:
            # a small turn joined to a long slide is a turn about a line far
            # off (the pole): true, and no help to a person. Where it is
            # clear, said as the slide and then the turn about the part's
            # own middle (or the turn first)
            ca = a[0] @ np.asarray(centre, float) + a[1]
            q = ca - leg.at
            a_ = np.asarray(leg.axis, float)
            if np.linalg.norm(q - (q @ a_) * a_) > 1.5 * reach:
                cb = b[0] @ np.asarray(centre, float) + b[1]
                u = cb - ca
                slide = (u / np.linalg.norm(u), float(np.linalg.norm(u)))
                for two in ([slide, Turn(a_, cb, leg.angle)], [Turn(a_, ca, leg.angle), slide]):
                    mid = end_pose(two[:1], start=a)
                    if clear(two[0], a) and clear(two[1], mid):
                        legs += two
                        break
                else:
                    legs.append(leg)
                continue
        legs.append(leg)
    return legs


def settler(depth, axis, centre, reach: float, h: float, limit: float, room: float = 0.3):
    """settle(pose) for simplify: a pose off a grid moved, by at most about
    a cell and a tilt step, to where the part is clear. `depth(pose)` is
    the part's deepest penetration there (mm, negative when clear by that
    much). The pose is walked downhill in depth (a slide along x, y or z,
    or a turn about `axis` through the part's centre, in halving steps)
    until it has `room` mm to spare or nothing helps; a pose still deeper
    than `limit` is None."""
    a = np.asarray(axis, float)
    c = np.asarray(centre, float)
    turn = float(np.degrees(1.0 / max(reach, 1e-9)))      # degrees per mm at the part's far end

    def moves(pose, d, turning):
        R, t = pose
        for k in range(3):
            for sg in (1.0, -1.0):
                e = np.zeros(3)
                e[k] = sg * d
                yield R, t + e
        if turning:
            cc = R @ c + t
            for sg in (1.0, -1.0):
                Q = rotation(a, sg * d * turn)
                yield Q @ R, Q @ (t - cc) + cc

    def walk(pose, turning):
        best, w = pose, depth(pose)
        d = 0.5 * h
        for _ in range(4):
            for _ in range(8):
                if w <= -room:
                    break
                better = None
                for cand in moves(best, d, turning):
                    wc = depth(cand)
                    if wc < w - 1e-4 and (better is None or wc < better[1]):
                        better = (cand, wc)
                if better is None:
                    break
                best, w = better
            d *= 0.5
        return best, w

    def settle(pose):
        # slides first: the pose keeps the tilt the grid gave it, so two
        # poses of one tilt are still a slide apart
        best, w = walk(pose, False)
        if w > limit:
            best, w = walk(pose, True)
        return best if w <= limit else None
    return settle


# --- JSON --------------------------------------------------------------------

def legs_from_spec(ins):
    """A keep-out's insertFrom as legs: [(d, length or None) | Turn]."""
    if ins is None:
        return None
    if isinstance(ins, str) or (len(ins) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                      for v in ins)):
        return [(_vec(ins), None)]
    out = []
    for leg in ins:
        if isinstance(leg, dict) and "turn" in leg:
            out.append(Turn(_vec(leg["about"]), np.asarray(leg["at"], float), float(leg["turn"]),
                            float(leg.get("along") or 0.0)))
        elif isinstance(leg, dict):
            out.append((_vec(leg["toward"]), float(leg["length"]) if leg.get("length") else None))
        else:
            out.append((_vec(leg), None))
    return out


def legs_to_spec(legs):
    """Legs as a keep-out's insertFrom: an axis for one straight way out,
    else a list of legs."""
    if len(legs) == 1 and not _is_turn(legs[0]) and legs[0][1] is None:
        return _name(legs[0][0])
    out = []
    for k, leg in enumerate(legs):
        if _is_turn(leg):
            rec = {"turn": round(float(leg.angle), 3), "about": _name(leg.axis),
                   "at": np.round(np.asarray(leg.at, float), 3).tolist()}
            if abs(leg.along) > 5e-4:
                rec["along"] = round(float(leg.along), 3)
            out.append(rec)
        else:
            d, length = leg
            if length is None and k == len(legs) - 1 and isinstance(_name(d), str):
                out.append(_name(d))
            else:
                out.append({"toward": _name(d), **({} if length is None else {"length": round(float(length), 3)})})
    return out


# --- the steps a person follows ----------------------------------------------

def _side(d) -> str:
    """The axis a direction mostly points along: '-z'."""
    d = np.asarray(d, float)
    k = int(np.argmax(np.abs(d)))
    return ("+" if d[k] > 0 else "-") + "xyz"[k]


def _toward(d) -> str:
    """A direction in words: '-y', or '-y, leaning 12° toward +z'."""
    n = _name(d)
    if isinstance(n, str):
        return n
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    k = int(np.argmax(np.abs(d)))
    rest = d.copy()
    rest[k] = 0.0
    j = int(np.argmax(np.abs(rest)))
    lean = float(np.degrees(np.arccos(min(abs(d[k]), 1.0))))
    return f"{_side(d)}, leaning {_mm(lean)}° toward {'+' if d[j] > 0 else '-'}{'xyz'[j]}"


def _mm(v) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v) >= 0.05 else "0"


def _end_moves(R_from, R_to, bounds) -> str | None:
    """Which end of the part goes where as it turns from one pose to the
    next: 'its +y end 8.3 mm toward +z' (the end named as the part sits)."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    half = 0.5 * (hi - lo)
    best = None
    for k in range(3):
        e = np.zeros(3)
        e[k] = half[k]
        move = R_to @ e - R_from @ e
        if best is None or np.linalg.norm(move) > np.linalg.norm(best[1]):
            best = (k, move)
    k, move = best
    if np.linalg.norm(move) < 0.05:
        return None
    j = int(np.argmax(np.abs(move)))
    return f"its +{'xyz'[k]} end {_mm(abs(move[j]))} mm toward {'+' if move[j] > 0 else '-'}{'xyz'[j]}"


def component_steps(name: str, legs, bounds, pts=None, exit_lo=None, exit_hi=None, carries=()) -> list:
    """The moves that put one part in, in order, in words: the legs it
    leaves by, turned round. bounds: the part's box where it sits."""
    who = name + (f" (with {', '.join(carries)} on it)" if carries else "")
    if not legs:
        return [f"Fit {who} in its place."]
    # the pose before each leg, as the part leaves
    before = [(np.eye(3), np.zeros(3))]
    for leg in legs:
        before.append(end_pose([leg], pts, exit_lo, exit_hi, start=before[-1]))
    out = []
    R_out = before[-1][0]
    tilt = float(np.degrees(np.arccos(np.clip((np.trace(R_out) - 1) / 2, -1, 1))))
    last = legs[-1]
    side = None if _is_turn(last) else _side(last[0])
    lead = f"Hold {who}"
    if tilt > 0.5:
        ax = np.array([R_out[2, 1] - R_out[1, 2], R_out[0, 2] - R_out[2, 0], R_out[1, 0] - R_out[0, 1]])
        ends = _end_moves(np.eye(3), R_out, bounds)
        lead += f" tilted {_mm(tilt)}° about {_side(ax)} from how it sits" + \
            (f" ({ends})" if ends else "")
    else:
        lead += " as it will sit"
    lead += f", outside the {side} side." if side else ", outside the exterior."
    out.append(lead)
    for k in range(len(legs) - 1, -1, -1):
        leg = legs[k]
        R_after = before[k][0]              # going in, this move ends at the pose before the leg
        if _is_turn(leg):
            a = np.asarray(leg.axis, float)
            at = np.asarray(leg.at, float)
            j = int(np.argmax(np.abs(a)))
            if abs(a[j]) > 1 - 1e-6:
                o = [i for i in range(3) if i != j]
                line = f"a line along {'xyz'[j]} through " + ", ".join(f"{'xyz'[i]} = {_mm(at[i])}" for i in o)
            else:
                line = f"the line through {np.round(at, 1).tolist()} along {np.round(a, 2).tolist()}"
            flat = float(np.degrees(np.arccos(np.clip((np.trace(R_after) - 1) / 2, -1, 1)))) < 0.5
            s = f"Turn it {_mm(abs(leg.angle))}° about {line}"
            ends = _end_moves(before[k + 1][0], R_after, bounds)
            if ends:
                s += f", {ends}"
            if abs(leg.along) > 0.05:
                s += f", sliding {_mm(abs(leg.along))} mm toward {_toward(-np.sign(leg.along) * a)} as it turns"
            s += ", until it sits square." if flat else "."
            out.append(s)
        else:
            d, length = leg
            d = np.asarray(d, float)
            if length is None:
                out.append(f"Bring it in toward {_toward(-d)}.")
            else:
                out.append(f"Slide it {_mm(length)} mm toward {_toward(-d)}.")
    out[-1] = out[-1].rstrip(".") + ": it is home."
    return out


def steps(order: list, paths: dict, bounds: dict, external=()) -> list:
    """Numbered assembly steps for a product: [{n, component, text}], each
    component's moves in the assembly order. paths: {id: {insertFrom,
    travelsWith}}; bounds: {id: (lo, hi)} where each sits."""
    out = []
    for cid in order:
        rec = paths.get(cid)
        if cid in external:
            texts = [f"Fit {cid} from outside."]
        elif rec is None:
            continue
        elif rec.get("insertFrom") is None:
            texts = [f"{cid}: no clear way in was found; it cannot be fitted as placed."]
        else:
            texts = component_steps(cid, legs_from_spec(rec["insertFrom"]), bounds[cid],
                                    carries=rec.get("travelsWith") or ())
        for t in texts:
            out.append({"n": len(out) + 1, "component": cid, "text": t})
    return out
