"""
Capture: what holds a socket when a plug pushes on it.

A USB-C plug goes in at up to 20 N, comes out at up to 20 N, and gets
levered sideways. connectors.py has those numbers. This module asks where
each of those forces goes once it reaches the socket. There are only two
places it can go:

- **a face**: material of some part standing right against the socket on
  the side the force pushes toward. The force goes through that face in
  compression, into the shell, a few square millimetres of plastic in
  series with the shell. That is stiff, and it is what the optimizer can
  design for;
- **the socket's own legs**: the solder joints and shell tabs, then the
  board, then whatever holds the board. Before any face takes load, the
  joints carry everything that moves the board into it. Pads that lift off
  a board under repeated plug forces are the classic USB-C failure.

Whether a part *can* have a face on a given side of the socket is not a
matter of geometry or taste: it is set by how the product goes together.
Take the parts apart in reverse order. When part X leaves the socket
moving along u (relative to the socket), a face of X on the side d of the
socket (the side a force along d pushes the socket into):

- moves away from the socket if u.d > 0: the face can be **seated**, drawn
  line-to-line (or with crush) against the socket, since nothing ever
  slides over it. An end stop, or a lid closing down onto the socket;
- slides along the socket if u.d is about 0: the face can be there, but
  only as a **press** fit (no play: crush ribs, a transition fit), since it
  rubs along the socket on the way in. A clamshell's front and rear
  shoulders are this;
- would have to pass through the socket if u.d < 0: there can be **no
  face** of X there at all. It would stand in the socket's way in.

For the part the socket goes into, u is minus the socket's own way out.
So a socket slid in along its own axis (from behind, toward its mouth)
can never have a face of that part behind it: the push-in drives it back
the way it came, where by construction there is nothing. Only a part that
goes on later (a cover, a flap, a clamshell half) can put a face there,
and that part is itself held by something: the same rule applied to its
own way off (seated or press on the ground part), or its joints (screws
pull it down with no play; a snap holds with its clearance).

That is the whole decision a designer makes when they clamshell a USB-C
socket between two halves instead of sliding it in: dropped in across its
axis, the half it lands in can stand right behind it and right in front
of it (press faces), the half it lands in holds it from below (seated),
and the other half closes on top of it (seated). Every plug force then has
a face.

A direction counts as captured when some part has a face there and the
play in the chain (socket to part, part to the ground part) is no more
than `maxShift` (the travel the socket's legs may take, connectors.py's
0.1 mm): the face takes the force before the legs do. Anything else goes
through the solder joints, which fails unless the connector's datasheet
says its legs are rated for it (`retention`, N per direction).

Functions:

- `service_loads`: the plug's forces as directions on the socket;
- `plan`: for a way of putting the product together (the socket's way in,
  the parts around it and how each goes on), which part takes each force,
  as what kind of face, with what play, and whether that passes;
- `choose_way_in`: the socket's way in (of the six axis directions, and
  any given) that lets the most force go into faces;
- `measure`: the same on drawn or delivered geometry: the real gap between
  the socket and each part's face along each force, the bearing area, and
  the gap from that part to the ground part;
- `bearing_pad`: the face itself, as a mesh: a slab conforming to the
  socket on the side a force pushes, at the gap the plan allows, inside the
  part that holds it. Preserved in that part's study with the plug force on
  it, the optimizer connects it to the shell (shells.py does this for a
  split).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["service_loads", "leave_direction", "face_kind", "plan", "choose_way_in", "measure", "bearing_pad",
           "connector_frame", "face_room", "SocketRoom"]

# |u.d| below this (10 degrees off square) is a face the part slides along
EPS = math.sin(math.radians(10.0))
JOINT_GAP = {"screws": 0.0, "snaps": None, "glue": 0.0, "weld": 0.0, "none": None}


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("a direction must not be zero")
    return v / n


def _axis_vec(v):
    if isinstance(v, str):
        s = v.strip().lower()
        d = np.zeros(3)
        d["xyz".index(s.lstrip("+-"))] = -1.0 if s.startswith("-") else 1.0
        return d
    return _unit(v)


def _name(d):
    d = np.asarray(d, float)
    k = int(np.argmax(np.abs(d)))
    if abs(abs(d[k]) - 1.0) < 1e-6:
        return ("+" if d[k] > 0 else "-") + "xyz"[k]
    return np.round(d, 4).tolist()


def connector_frame(out, wide=None):
    """(out, wide, thin): the way the plug comes out, the socket's wide
    direction across it (default the axis across `out` nearest +x, else
    +y), and thin = out x wide."""
    out = _axis_vec(out)
    if wide is None:
        for ax in np.eye(3):
            w = ax - (ax @ out) * out
            if np.linalg.norm(w) > 0.5:
                wide = w
                break
    wide = _axis_vec(wide) if isinstance(wide, str) else np.asarray(wide, float)
    wide = _unit(wide - (wide @ out) * out)
    return out, wide, np.cross(out, wide)


def service_loads(kind: str = "usb-c", out="+y", wide=None, insert=None, withdraw=None, side=None,
                  abuse: bool = False, wrench=None, lever=None) -> list[dict]:
    """The plug's forces on the socket as directions: push-in (along -out),
    pull-out (+out), and each side load both ways across the thin and the
    wide side (the plug is levered either way). With abuse, the wrenching
    moments as the side force at the lever that reaches them. Each:
    {name, d (unit, the way the force pushes the socket), force N}."""
    from .connectors import CONNECTORS

    base = dict(CONNECTORS.get(kind) or {})
    if not base and None in (insert, withdraw):
        raise ValueError(f"connector type {kind!r}: not one of {sorted(CONNECTORS)}; give insert and withdraw")
    out, wd, th = connector_frame(out, wide)
    F_in = float(insert if insert is not None else base["insert"])
    F_out = float(withdraw if withdraw is not None else base["withdraw"])
    sides = dict(base.get("side") or {})
    sides.update(side or {})
    rows = [{"name": "insert", "d": -out, "force": F_in}, {"name": "withdraw", "d": out, "force": F_out}]
    across = {"thin": th, "wide": wd}
    for k in ("thin", "wide"):
        if sides.get(k):
            for sg, tag in ((1.0, "+"), (-1.0, "-")):
                rows.append({"name": f"side-{k}{tag}", "d": sg * across[k], "force": float(sides[k])})
    if abuse:
        wr = dict(base.get("wrench") or {})
        wr.update(wrench or {})
        L = float(lever if lever is not None else base.get("lever", 15.0))
        for k in ("thin", "wide"):
            if wr.get(k):
                for sg, tag in ((1.0, "+"), (-1.0, "-")):
                    rows.append({"name": f"wrench-{k}{tag}", "d": sg * across[k], "force": float(wr[k]) / L,
                                 "abuse": True})
    return rows


def leave_direction(insert_from, at=None) -> np.ndarray:
    """The way a part first moves as it leaves its place (its way in run
    backward): the first slide of a keep-out's insertFrom, or the
    direction the point `at` starts to move on a first turn."""
    from .motion import Turn, legs_from_spec

    legs = legs_from_spec(insert_from)
    if not legs:
        raise ValueError("no way in to read a direction from")
    leg = legs[0]
    if isinstance(leg, Turn):
        if at is None:
            raise ValueError("a way in that starts with a turn needs the point to follow (at)")
        w = _unit(leg.axis) * math.copysign(1.0, leg.angle)
        v = np.cross(w, np.asarray(at, float) - np.asarray(leg.at, float)) + leg.along * _unit(leg.axis)
        return _unit(v)
    return _unit(leg[0])


def face_kind(u, d) -> str | None:
    """What face part X can have against the socket on side d, X leaving
    the socket along u: 'seated', 'press' or None (it would block the way)."""
    c = float(np.dot(_unit(u), _unit(d)))
    if c > EPS:
        return "seated"
    if c >= -EPS:
        return "press"
    return None


def _held(part: dict, d, ground: str, parts=(), depth: int = 0):
    """The play between a part and the ground part along d, how it is
    held, and how many press fits that takes: (gap or None, how, fits).
    A part is held by the ground part (a face, if its own way off leaves
    room for one), by a part that goes on after it (a lid over a board:
    that part's face, then that part's own hold), or by its joints."""
    if part["id"] == ground:
        return 0.0, "is the ground part", 0
    w = part.get("leave")
    if w is None:
        return None, f"{part['id']} has no way off given", 0
    w = _axis_vec(w)
    opts = []
    k = face_kind(-w, d)                 # the ground part as this one moves off it
    if k is not None:
        # a press face between two parts that are fitted with play (a
        # cover's lip in the base) keeps that play
        g = float(part.get("pressGap", 0.0)) if k == "press" else 0.0
        opts.append((g, int(k == "press"), f"{part['id']} bears on {ground} ({k}"
                     + (f", {g:g} mm play" if g else "") + ")"))
    if part.get("carrier") and depth < 2:
        # a board is also held by what closes over it after it is in
        for q in parts:
            if q is part or not q.get("after") or q.get("carrier") or q.get("leave") is None:
                continue
            kq = face_kind(_axis_vec(q["leave"]), d)
            if kq is None:
                continue
            g, how, f = _held(q, d, ground, parts, depth + 1)
            if g is not None:
                opts.append((g, f + int(kq == "press"), f"{part['id']} bears on {q['id']} ({kq}); {how}"))
    joints = part.get("joints", "none")
    gap = part.get("jointGap", JOINT_GAP.get(joints))
    if joints == "snaps" and gap is None:
        gap = part.get("clearance")
    if gap is not None:
        opts.append((float(gap), 0, f"{part['id']} held on {ground} by its {joints}"))
    if not opts:
        return None, f"nothing holds {part['id']} on {ground} along {_name(d)} (it comes off that way; joints: {joints})", 0
    g, f, how = min(opts, key=lambda o: (o[0], o[1]))
    return g, how, f


def plan(loads: list, socket_leave, parts: list, ground: str, max_shift: float = 0.1,
         retention: dict | None = None, press_play: float = 0.0, min_area: float = 1.0) -> dict:
    """Which part takes each plug force, for one way of putting it together.

    loads: service_loads(...). socket_leave: the way the socket (with its
    board) first moves leaving its place. parts: [{id, leave (the part's
    own way off, relative to the ground part; the ground part's is not
    needed), after (bool: goes on after the socket), carrier (bool: the
    board the socket is soldered to: it bears only where the socket
    stands on it, and is held as the socket's way in allows), joints
    ('screws', 'snaps', 'glue', 'none'), clearance (a snap's play, mm),
    gap (the drawn gap to the socket, if known), room ({load name: mm2}
    of the socket's face on that side that has room for this part's face:
    out of the plug's path, other parts, and inside this part. Missing:
    room assumed)}]. ground: the id of the part the socket's board is
    mounted in (the one held in the hand). retention: {load name or 'all':
    N} the socket's legs are rated for (its datasheet). press_play: the
    play each press fit in the chain is expected to leave (mm; 0 = crush
    ribs or a fit tuned on the printer). min_area: mm2 of face below which
    a side has no room.

    Returns {loads: [{name, force, d, by, face, gap, held, fits, ok, why}],
    ok (False if a force that could have a face goes to the legs; None
    when the only open question is a force no face can reach, e.g. pulling
    the plug out of a socket whose whole front is the plug's way), captured
    N, legs N, style, socketIn}."""
    v = _axis_vec(socket_leave)
    retention = dict(retention or {})
    rows = []
    for ld in loads:
        d = _unit(ld["d"])
        best = None
        tried = []
        roomless = 0
        for p in parts:
            room = (p.get("room") or {}).get(ld["name"])
            if room is not None and room < min_area:
                roomless += 1
                tried.append(f"{p['id']}: no room for a face there ({room:.1f} mm2)")
                continue
            if p.get("carrier"):
                k = "seated"                         # the socket stands on it
                hold = {**p, "leave": p.get("leave", v)}
            else:
                if p["id"] == ground or not p.get("after"):
                    u = -v                           # the socket leaves this part
                else:
                    if p.get("leave") is None:
                        continue
                    u = _axis_vec(p["leave"])        # this part comes off the socket
                k = face_kind(u, d)
                if k is None:
                    tried.append(f"{p['id']}: a face there would block "
                                 + ("the socket's way in" if p["id"] == ground or not p.get("after")
                                    else f"{p['id']}'s way off"))
                    continue
                hold = p
            g0 = p.get("gap")
            g0 = 0.0 if g0 is None else float(g0)
            hg, how, fits = _held(hold, d, ground, parts)
            if hg is None:
                tried.append(how)
                continue
            fits += int(k == "press")
            gap = g0 + hg + press_play * fits
            cand = {"by": p["id"], "face": k, "gap": round(gap, 4), "held": how, "fits": fits}
            # prefer no play, then the fewest press fits in series (each one
            # a fit the print has to hold), then the ground part itself
            key = (gap > max_shift + 1e-9, fits, gap, p["id"] != ground)
            if best is None or key < best[0]:
                best = (key, cand)
        F = float(ld["force"])
        rec = {"name": ld["name"], "force": round(F, 3), "d": _name(d)}
        if ld.get("abuse"):
            rec["abuse"] = True
        rated = retention.get(ld["name"], retention.get("all"))
        if best is not None and best[1]["gap"] <= max_shift + 1e-9:
            rec.update(best[1], ok=True)
            rec["why"] = (f"{best[1]['by']} bears on the socket ({best[1]['face']}"
                          + (": draw it with no play, crush ribs or a transition fit" if best[1]['face'] == "press"
                             else "") + f"); {best[1]['held']}"
                          + (f"; {best[1]['fits']} press fits in series" if best[1]['fits'] > 1 else ""))
        elif rated is not None and float(rated) >= F:
            rec.update(by="legs", face=None, gap=None, ok=True,
                       why=f"the socket's legs are rated for {float(rated):g} N (datasheet)")
        elif best is None and parts and roomless == len(parts):
            # no part anywhere could put a face there (the plug's own way
            # out): the legs carry it whatever the assembly; the datasheet
            # says whether they can
            rec.update(by="legs", face=None, gap=None, ok=None,
                       why=f"no room for any face on the {_name(d)} side (the plug's way, or other parts): the "
                           f"{F:g} N is the socket's legs' to carry in any assembly; give retention.{ld['name']} "
                           "from its datasheet")
        else:
            rec.update(by="legs", face=None, gap=None if best is None else best[1]["gap"], ok=False)
            if best is not None:
                rec["why"] = (f"{best[1]['by']} could bear here, but with {best[1]['gap']:.2f} mm of play "
                              f"({best[1]['held']}): more than the {max_shift:g} mm the socket's legs may move, "
                              f"so its solder joints take the {F:g} N first")
            else:
                rec["why"] = (f"no part can have a face on the {_name(d)} side of the socket ("
                              + "; ".join(tried) + f"): the {F:g} N goes through its solder joints")
        rows.append(rec)
    core = [r for r in rows if not r.get("abuse")]
    ok = False if any(r["ok"] is False for r in core) else (None if any(r["ok"] is None for r in core) else True)
    captured = sum(r["force"] for r in rows if r["ok"] and r["by"] != "legs")
    legs = sum(r["force"] for r in rows if r["by"] == "legs" and r["ok"] is False)
    holders = sorted({r["by"] for r in rows if r["ok"] and r["by"] != "legs"})
    seated = [(r["by"], _axis_vec(r["d"])) for r in rows if r.get("face") == "seated" and r["ok"]]
    pair = next(((a, b) for a, da in seated for b, db in seated if a != b and float(da @ db) < -0.9), None)
    if pair:
        style = f"clamped between {pair[0]} and {pair[1]}; in along {_name(-v)}"
    elif holders:
        style = f"held by {', '.join(holders)}; in along {_name(-v)}"
    else:
        style = f"slid in along {_name(-v)}, held by its legs"
    return {"loads": rows, "ok": ok, "captured": round(captured, 3), "legs": round(legs, 3), "style": style,
            "socketIn": _name(-v)}


def choose_way_in(loads: list, parts: list, ground: str, candidates=None, max_shift: float = 0.1,
                  retention: dict | None = None, press_play: float = 0.0, room=None) -> list[dict]:
    """Each candidate way in for the socket (default the six axis
    directions it could arrive moving along), planned and ranked: every
    force on a face first, then the least force on the legs, then the
    fewest press fits along the way (each one a fit the print has to
    hold). room: optional callable(socket_leave) -> parts with their
    `room` filled for that way in. Returns the plans, best first, each with
    `socketIn`."""
    cands = list(candidates) if candidates is not None else [s * e for e in np.eye(3) for s in (1.0, -1.0)]
    out = []
    for arrive in cands:
        ps = parts if room is None else room(-_axis_vec(arrive))
        p = plan(loads, -_axis_vec(arrive), ps, ground, max_shift, retention, press_play)
        fits = sum(r.get("fits", 0) for r in p["loads"] if r["ok"])
        # then the force that has to cross a joint (through a part that is
        # not the ground one) before it reaches the hand
        joint = sum(r["force"] for r in p["loads"] if r["ok"] and r["by"] not in ("legs", ground))
        p["fits"] = fits
        p["throughJoints"] = round(joint, 3)
        p["_key"] = (p["ok"] is False, p["legs"], fits, joint)
        out.append(p)
    out.sort(key=lambda p: p.pop("_key"))
    return out


# ---------------------------------------------------------------------------
# geometry


def _surface(shape, step: float, cap: int = 20000, seed: int = 0):
    """(points, outward normals, area per point) on a shape's surface."""
    import trimesh

    from .shapes import MeshShape

    if isinstance(shape, MeshShape):
        m = shape.mesh
    else:
        from skimage import measure as skm
        lo, hi = shape.bounds()
        lo, hi = lo - 2 * step, hi + 2 * step
        ax = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
        G = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
        F = shape.sdf(G.reshape(-1, 3)).reshape(G.shape[:3])
        vt, fc, _, _ = skm.marching_cubes(F, 0.0, spacing=(step, step, step))
        m = trimesh.Trimesh(vt + lo, fc[:, ::-1], process=True)
        if m.volume < 0:
            m.invert()
    n = int(min(cap, max(200, m.area / (step * step))))
    P, fi = trimesh.sample.sample_surface_even(m, n, seed=seed)
    N = m.face_normals[fi]
    return np.asarray(P), np.asarray(N), float(m.area) / max(len(P), 1)


def _ray_gaps(P, d, target, reach: float, step: float = 0.02):
    """Distance along d from each point to where it first enters `target`
    (inf within reach). Marches in steps, then bisects."""
    t = np.full(len(P), np.inf)
    s = 0.0
    prev = np.zeros(len(P))
    alive = np.ones(len(P), bool)
    if target.sdf(P).size == 0:
        return t
    while s <= reach + 1e-9 and alive.any():
        idx = np.nonzero(alive)[0]
        f = target.sdf(P[idx] + s * d)
        hit = f <= 0
        if hit.any():
            h = idx[hit]
            lo_, hi_ = prev[h], np.full(len(h), s)
            for _ in range(6):
                m_ = 0.5 * (lo_ + hi_)
                inside = target.sdf(P[h] + m_[:, None] * d) <= 0
                hi_ = np.where(inside, m_, hi_)
                lo_ = np.where(inside, lo_, m_)
            t[h] = hi_
            alive[h] = False
        prev[alive] = s
        # step by the distance to the target where it is far (an sdf may
        # read low near cuts, never high), never below `step`
        s += max(step, float(np.min(np.maximum(f[~hit], 0.0))) if (~hit).any() else step)
    return t


def face_room(socket, loads: list, within=None, avoid=(), depth: float = 0.8, step: float = 0.15,
              touching=None, contact: float = 0.05, surface=None) -> dict:
    """mm2 of the socket's face on each force's side with room for a face:
    socket surface looking that way (normal within 60 degrees of it) from
    which `depth` mm of material could stand along the force inside
    `within` (the part's space) and out of every `avoid` shape (the plug's
    path, other components). With `touching` (the board the socket stands
    on) instead: the area within `contact` mm of it along the force.
    Returns {load name: mm2}."""
    P, N, dA = surface if surface is not None else _surface(socket, step)
    out = {}
    for ld in loads:
        d = _unit(ld["d"])
        Q = P[(N @ d) > 0.5]
        if len(Q) == 0:
            out[ld["name"]] = 0.0
            continue
        if touching is not None:
            g = _ray_gaps(Q + 0.01 * d, d, touching, contact + 0.02, step=0.01)
            out[ld["name"]] = round(float(np.sum(np.isfinite(g))) * dA, 2)
            continue
        ok = np.ones(len(Q), bool)
        for s_ in np.linspace(0.05, depth, max(3, int(math.ceil(depth / 0.1)))):
            q = Q + s_ * d
            if within is not None:
                ok &= within.sdf(q) <= 0
            for a_ in avoid:
                ok &= a_.sdf(q) > 0
        out[ld["name"]] = round(float(ok.sum()) * dA, 2)
    return out


class SocketRoom:
    """face_room for many candidate parts that share one space: the
    socket's faces and the points a face would fill along each force are
    sampled once, checked once against what no part may fill (outside the
    exterior, the plug's path, other components), and then each part is
    just a test of which of those points it owns (`areas(member)`: a cut's
    side, say). touching: the socket's carrier board, for its own row."""

    def __init__(self, socket, loads: list, within=None, avoid=(), depth: float = 0.8, step: float = 0.15,
                 carrier=None):
        self.loads = loads
        P, N, self.dA = _surface(socket, step)
        self.rows = []
        ks = np.linspace(0.05, depth, max(3, int(math.ceil(depth / 0.1))))
        for ld in loads:
            d = _unit(ld["d"])
            Q = P[(N @ d) > 0.5]
            pts = (Q[None, :, :] + ks[:, None, None] * d[None, None, :]).reshape(-1, 3)
            ok = np.ones(len(pts), bool)
            if within is not None and len(pts):
                ok &= within.sdf(pts) <= 0
            for a_ in avoid:
                if len(pts):
                    ok &= a_.sdf(pts) > 0
            self.rows.append((ld["name"], pts, ok.reshape(len(ks), -1)))
        self.touch = (face_room(socket, loads, touching=carrier, surface=(P, N, self.dA))
                      if carrier is not None else None)

    def areas(self, member) -> dict:
        out = {}
        for name, pts, ok in self.rows:
            if ok.size == 0:
                out[name] = 0.0
                continue
            m = ok & np.asarray(member(pts), bool).reshape(ok.shape)
            out[name] = round(float(m.all(axis=0).sum()) * self.dA, 2)
        return out


def measure(socket, parts: dict, loads: list, ground: str, max_shift: float = 0.1, reach: float = 2.0,
            step: float = 0.15, contact: float = 0.05, retention: dict | None = None,
            material=None) -> dict:
    """Capture on drawn or delivered geometry.

    socket: its shape (sdf). parts: {id: shape} of each part around it
    (the ground part among them: the one the board is mounted in). loads:
    service_loads(...). For each force, every part's real gap to the socket
    along the force (from the socket's faces that look that way: normal
    within 60 degrees), the bearing area within `contact` mm of the
    nearest, and for a part other than the ground, its own gap to the
    ground part along the same force (the play in the chain). Pass when the
    whole chain is within max_shift; the bearing stress (force over area)
    is reported against the material's compressive strength when a
    material is given."""
    P, N, dA = _surface(socket, step)
    yield_ = None
    if material is not None:
        from .materials import material_from_spec
        mat = material_from_spec(material) if not hasattr(material, "strength") else material
        yield_ = float(mat.strength)
    surf_cache = {}
    retention = dict(retention or {})
    rows = []
    for ld in loads:
        d = _unit(ld["d"])
        sel = (N @ d) > 0.5
        Pd = P[sel] + 0.01 * d
        per = {}
        for pid, sh in parts.items():
            g = _ray_gaps(Pd, d, sh, reach)
            fin = np.isfinite(g)
            if not fin.any():
                continue
            gmin = float(g[fin].min())
            area = float(np.sum(g[fin] <= gmin + contact)) * dA
            chain = 0.0
            if pid != ground and ground in parts:
                if pid not in surf_cache:
                    surf_cache[pid] = _surface(sh, step)
                Q, NQ, _ = surf_cache[pid]
                # the part's faces that look the way the force pushes, in
                # the socket's shadow (where it bears), against the ground
                q = Q[(NQ @ d) > 0.5] + 0.01 * d
                gq = _ray_gaps(q, d, parts[ground], reach)
                chain = float(gq[np.isfinite(gq)].min()) if np.isfinite(gq).any() else float("inf")
            per[pid] = {"gap": round(gmin, 3), "area": round(area, 2), "chain": round(chain, 3) if math.isfinite(chain)
                        else None}
        F = float(ld["force"])
        rec = {"name": ld["name"], "force": round(F, 3), "d": _name(d), "parts": per}
        tot = {k: v["gap"] + (v["chain"] if v["chain"] is not None else np.inf) for k, v in per.items()}
        if tot:
            by = min(tot, key=tot.get)
            rec.update(by=by, gap=round(float(tot[by]), 3) if math.isfinite(tot[by]) else None,
                       area=per[by]["area"])
            if per[by]["area"] > 0:
                rec["bearingMPa"] = round(F / per[by]["area"], 3)
                if yield_:
                    rec["bearingFactor"] = round(yield_ / rec["bearingMPa"], 2)
        else:
            rec.update(by=None, gap=None)
        ok = rec.get("gap") is not None and rec["gap"] <= max_shift + 1e-9
        if not ok:
            rated = retention.get(ld["name"], retention.get("all"))
            if rated is not None and float(rated) >= F:
                ok = True
                rec["why"] = f"the socket's legs are rated for {float(rated):g} N"
            elif rec.get("by") is None:
                rec["why"] = f"nothing within {reach:g} mm on the {_name(d)} side: the {F:g} N goes through the legs"
            else:
                v = per[rec["by"]]
                rec["why"] = (f"{rec['by']} is {v['gap']:.2f} mm off the socket"
                              + (f" and {v['chain']:.2f} mm off {ground}" if v["chain"] else "")
                              + f": more than {max_shift:g} mm, so the legs take the {F:g} N first")
        rec["ok"] = bool(ok)
        rows.append(rec)
    return {"loads": rows, "ok": all(r["ok"] for r in rows if not any(ld.get("abuse") and ld["name"] == r["name"]
                                                                      for ld in loads)),
            "maxShift": max_shift}


def bearing_pad(socket, d, gap: float, thickness: float, within=None, avoid=(), step: float = 0.1):
    """The face that takes a force along d: the slab between the socket
    moved `gap` and moved `gap + thickness` along d (so it follows the
    socket's own shape on that side), kept inside `within` (the holding
    part's design space) and out of each `avoid` shape (the plug's path,
    other components). A trimesh, or None when nothing is left."""
    import trimesh
    from skimage import measure as skm

    d = _unit(d)
    lo, hi = socket.bounds()
    lo = np.minimum(lo, lo + (gap + thickness) * d) - 3 * step
    hi = np.maximum(hi, hi + (gap + thickness) * d) + 3 * step
    ax = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    # distance to the socket swept from s0 to s1 along d, by sampling the
    # sweep (min over copies): fine for the few-mm slabs here
    def swept(s0, s1):
        n = max(2, int(math.ceil((s1 - s0) / (0.5 * step))) + 1)
        f = np.full(len(G), np.inf)
        for s in np.linspace(s0, s1, n):
            f = np.minimum(f, socket.sdf(G - s * d))
        return f
    outer = swept(0.0, gap + thickness)
    inner = swept(0.0, gap) if gap > 0 else socket.sdf(G)
    f = np.maximum(outer, -inner)
    if within is not None:
        f = np.maximum(f, within.sdf(G))
    for a in avoid:
        f = np.maximum(f, -a.sdf(G))
    F = f.reshape(len(ax[0]), len(ax[1]), len(ax[2]))
    F[[0, -1], :, :] = F[:, [0, -1], :] = F[:, :, [0, -1]] = step
    if F.min() >= 0:
        return None
    # a box socket's faces land on grid nodes: values of exactly 0 there put
    # several surface vertices on one node, their triangles have no area,
    # and an STL reader drops them and opens holes (a 160 mm handheld's
    # socket face was 174 open edges once saved, and its half's run refused
    # it). Nudge them off the level by a hundredth of a cell.
    F = np.where(np.abs(F) < 0.01 * step, 0.01 * step, F)
    vt, fc, _, _ = skm.marching_cubes(F, 0.0, spacing=(step, step, step))
    m = trimesh.Trimesh(vt + lo, fc[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)        # trimesh on a degenerate sliver piece
        parts = m.split(only_watertight=False)
    if len(parts) > 1:
        # slivers where the slab grazes `within`'s edge are not a face
        keep = [p for p in parts if abs(p.volume) > 0.05 * max(abs(q.volume) for q in parts)]
        m = trimesh.util.concatenate(keep)
    return m if abs(m.volume) > 1e-3 else None
