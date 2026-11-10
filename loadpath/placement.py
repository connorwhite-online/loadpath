"""
Where the parts inside go: component placement inside a fixed exterior.

A designer draws the outside of a product and lists what has to go in it: a
battery, a board with its connector, a haptic motor, a speaker. Where each
one sits is engineering, not styling, and it is the first thing an agent has
to decide before anything inside can be designed. `place` decides it.

Each component (or a group: a board with the connector fitted to it, which
move as one) is tried in every allowed orientation (90-degree turns: all 24,
the four about one axis, or none) at every position on a grid through the
cavity. The cavity is the exterior less its skin, plus any opening a part
may reach into (a connector's mouth in its port). For one orientation every
position is checked at once: the blocked cells (the skin, fixed geometry,
the parts already placed, each grown by the margin) cross-correlated with
the part's own cells by FFT give, per position, how many cells overlap;
zero is a fit. Among the fits the rules pick:

- `against`: push it to that side of the cavity (a battery on the floor);
- `alignTo`: a face of it (or of a part on it) out through an opening, on
  the opening's axis, as far out as it goes (a USB-C receptacle in its
  port, set back by `recess`);
- `under`: a face of it on a line, as close to the skin as `gap` allows (a
  switch under the spot a finger presses);
- `near`: close to other components (short wires, a motor near the hand);
- the whole layout's centre of mass toward a target, or low along an axis
  (`placement.centerOfMass`).

Groups go in one at a time, the most constrained first, as a small beam
search (the best few layouts kept at each step), so an early greedy choice
that boxes a later part in can be undone. Every position found on the grid
is then pushed the rest of the way along its rule, and checked with exact
distances on dense samples of the part: the clearance reported is measured,
not the grid's.

Nothing here changes the exterior. A part that fits nowhere is reported
with the closest it came, which is the designer's (or the agent's) to
resolve: a smaller part, a different orientation, or a change to the
exterior.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

__all__ = ["place", "rotations", "component_mesh"]

_AX = {"x": 0, "y": 1, "z": 2}


def _axis_vec(side) -> np.ndarray:
    if isinstance(side, str):
        s = side.strip().lower()
        sg = -1.0 if s.startswith("-") else 1.0
        d = np.zeros(3)
        d[_AX[s.lstrip("+-")]] = sg
        return d
    d = np.asarray(side, float)
    return d / np.linalg.norm(d)


def _axis_name(d) -> str:
    k = int(np.argmax(np.abs(d)))
    return ("+" if d[k] > 0 else "-") + "xyz"[k]


def _all24() -> list:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = signs[i]
            if np.linalg.det(R) > 0:
                out.append(R)
    # identity first: an unturned part wins ties
    out.sort(key=lambda R: -np.trace(R))
    return out


_R24 = _all24()


def rotations(spec) -> list:
    """The orientations a component may take: 'any' (all 24 quarter turns),
    an axis ('z': the four turns about it), 'none' (as drawn), or a list of
    3x3 matrices."""
    if spec is None or spec is False or spec == "none":
        return [np.eye(3)]
    if spec is True or spec == "any":
        return list(_R24)
    if isinstance(spec, str) and spec.lstrip("+-") in _AX:
        a = _AX[spec.lstrip("+-")]
        return [R for R in _R24 if abs(R[a, a] - 1.0) < 1e-9]
    if isinstance(spec, list):
        return [np.asarray(R, float) for R in spec]
    raise ValueError(f"rotate is 'any', 'none', an axis ('x', 'y', 'z') or a list of matrices, got {spec!r}")


def component_mesh(c: dict, base_dir=None):
    """A component's geometry as a closed trimesh, in its own (kit)
    coordinates: a mesh path, a shape dict, or `size` (a box centred on the
    origin)."""
    import trimesh

    from .shapes import shape_from_spec, trimesh_from_manifold
    if c.get("size") is not None:
        s = np.asarray(c["size"], float)
        m = trimesh.creation.box(extents=s)
        if c.get("center") is not None:
            m.apply_translation(np.asarray(c["center"], float))
        return m
    spec = c["shape"]
    if isinstance(spec, str):
        spec = {"type": "mesh", "path": spec}
    sh = shape_from_spec(spec, base_dir)
    if hasattr(sh, "mesh"):
        return sh.mesh.copy()
    return trimesh_from_manifold(sh.to_manifold())


class _Grid:
    """Cell centres through the exterior's bounds, at pitch h."""

    def __init__(self, lo, hi, h):
        self.h = float(h)
        self.lo = np.asarray(lo, float) - 2 * h
        n = np.ceil((np.asarray(hi, float) + 2 * h - self.lo) / h).astype(int)
        self.n = tuple(int(v) for v in n)
        self.axes = [self.lo[i] + (np.arange(self.n[i]) + 0.5) * h for i in range(3)]

    def points(self):
        return np.stack(np.meshgrid(*self.axes, indexing="ij"), -1).reshape(-1, 3)


def _face_point(verts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """The centre of a part's face that looks along `normal` (in the same
    frame as verts): its bounding box's face centre."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    c = 0.5 * (lo + hi)
    k = int(np.argmax(np.abs(normal)))
    c[k] = hi[k] if normal[k] > 0 else lo[k]
    return c


def _dense(mesh, step: float, cap: int = 6000) -> np.ndarray:
    """Points standing for a part: surface samples plus the vertices."""
    import trimesh
    n = int(min(cap, max(400, mesh.area / max(step, 0.05) ** 2)))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.vstack([np.asarray(pts), np.asarray(mesh.vertices)])


class _Group:
    def __init__(self, gid, members, meshes, masses, rules, rot, margin):
        self.id = gid
        self.members = members                 # ids, root first
        self.meshes = meshes                   # id -> trimesh, kit frame
        V = np.vstack([np.asarray(meshes[m].vertices) for m in members])
        root = np.asarray(meshes[members[0]].vertices)
        self.pivot = 0.5 * (root.min(axis=0) + root.max(axis=0))
        self.verts = V - self.pivot            # pivot frame
        self.masses = masses
        w = np.array([max(masses.get(m) or 0.0, 1e-9) for m in members])
        cs = np.array([np.asarray(meshes[m].bounds).mean(axis=0) for m in members])
        self.com = (w[:, None] * cs).sum(axis=0) / w.sum() - self.pivot
        self.mass = float(sum(masses.get(m) or 0.0 for m in members))
        self.rules = rules                     # list of (member id, rule dict)
        self.rots = rot
        self.margin = margin
        self.volume = float(sum(abs(meshes[m].volume) for m in members))
        self._shapes = None

    def shapes(self):
        from .shapes import MeshShape
        if self._shapes is None:
            out = []
            for m in self.members:
                mm = self.meshes[m].copy()
                mm.apply_translation(-self.pivot)
                sh = MeshShape(mm)
                sh._GRID_CELLS = 160
                out.append(sh)
            self._shapes = out
        return self._shapes

    def sdf(self, R, t, p):
        """Signed distance to the group placed by (R, t), at world points p."""
        q = (np.asarray(p, float) - t) @ R          # R^T (p - t), row-wise
        out = None
        for sh in self.shapes():
            d = sh.sdf(q)
            out = d if out is None else np.minimum(out, d)
        return out

    def member_verts(self, m):
        return np.asarray(self.meshes[m].vertices) - self.pivot


def place(exterior, components: list, *, skin: float = 1.0, openings=None, fixed=None, rules: dict | None = None,
          base_dir=None, pitch: float | None = None, beam: int = 3, margin: float = 0.5,
          on_progress=None, access=None, judge=None) -> dict:
    """Place components inside an exterior. See the module docstring.

    exterior: a Shape (closed). components: [{id, shape | size, mass?,
    installedOn?, place ('auto' | 'given'), rotate, against, alignTo:
    {opening, face, recess}, under: {at, direction, face, gap}, near: [ids],
    margin}]. openings: [{id?, shape (Shape), inward?, depth?}]. fixed:
    [(id, Shape)] the designer's own internal geometry. rules:
    {centerOfMass: {target | low, weight}}. access: optional callable
    (group id, sdf, bounds, placed {id: (sdf, bounds)}) -> bool, asked of
    each position kept: whether the part can still get there (enclosure
    passes its assembly-path search), so a layout no one can assemble is
    dropped while there are others. judge: optional callable ({transforms:
    {component id: (R, T)}, new: [ids just placed], room, origin, pitch})
    -> extra cost, or None to refuse the position: what joins the parts
    (wires.placement_judge lays each cable between the parts placed so far,
    so a layout that cannot be wired is dropped, and a shorter, easier
    harness is preferred). rules.heat: see `_heat_cost`.

    Returns {placements: {id: {rotation, translation, matrix, ...}},
    groups, layout cost, clearance per component, blocked, seconds}. The
    matrix maps a component's own (kit) coordinates into the product's."""
    from scipy.signal import fftconvolve

    t0 = time.time()
    rules = rules or {}
    openings = list(openings or [])
    fixed = list(fixed or [])
    byid = {}
    for i, c in enumerate(components):
        cid = c.get("id") or f"component-{i}"
        if cid in byid:
            raise ValueError(f"duplicate component id {cid!r}")
        byid[cid] = {**c, "id": cid}
    meshes = {cid: component_mesh(c, base_dir) for cid, c in byid.items()}
    masses = {cid: float(c.get("mass") or 0.0) for cid, c in byid.items()}
    kids = {}
    for cid, c in byid.items():
        p = c.get("installedOn")
        if p:
            if p not in byid:
                raise ValueError(f"component {cid!r} is installedOn {p!r}, which is not a component")
            kids.setdefault(p, []).append(cid)

    def members(cid):
        out = [cid]
        for k in kids.get(cid, []):
            out += members(k)
        return out

    def is_auto(c):
        v = c.get("place")
        if v is not None:
            return v == "auto"
        return any(c.get(k) is not None for k in ("against", "alignTo", "under", "near", "rotate")) \
            or c.get("size") is not None

    lo, hi = exterior.bounds()
    ext_len = np.asarray(hi) - np.asarray(lo)
    if pitch is None:
        pitch = float(np.clip((np.prod(ext_len) / 1.2e6) ** (1 / 3), 0.3, 2.0))
    G = _Grid(lo, hi, pitch)
    P = G.points()
    if on_progress:
        on_progress({"stage": "placement", "message": f"cavity grid {G.n} at {pitch:.2f} mm"})
    e = exterior.sdf(P)
    cav = e + float(skin)                                 # < 0 inside the cavity
    op_shapes = {}
    for j, o in enumerate(openings):
        oid = o.get("id") or f"opening-{j}"
        op_shapes[oid] = o
    # room: distance to the nearest thing a part must keep off (approximate
    # where an opening joins, exact enough for the grid search; the final
    # check is exact)
    room_base = -cav
    cav_c0 = P[cav < 0].mean(axis=0) if (cav < 0).any() else 0.5 * (np.asarray(lo) + np.asarray(hi))
    for o in op_shapes.values():
        # the room runs on past the skin behind the mouth, so a part near it
        # is not held off the mouth's own end as if it were a wall
        o.setdefault("reach", float(skin) + 1.0)
        _mouth(o, exterior, cav_c0)
        # an opening is room: a part may reach into it (a cap into its
        # hole, a battery down onto its door), short of the exterior
        room_base = np.maximum(room_base, _opening_room(o, P, e, 0.3))
    obst = np.full(len(P), np.inf)
    for _, sh in fixed:
        obst = np.minimum(obst, sh.sdf(P))
    room_base = np.minimum(room_base, obst)
    room_base = room_base.reshape(G.n)

    def opening_room(oid, recess):
        """Room inside an opening, up to `recess` short of the exterior."""
        return _opening_room(op_shapes[oid], P, e, recess).reshape(G.n)

    # the groups: roots of installedOn chains
    groups = []
    given = []
    for cid, c in byid.items():
        if c.get("installedOn"):
            continue
        mem = members(cid)
        rl = [(m, byid[m]) for m in mem]
        if not is_auto(c):
            given.append(cid)
            continue
        rot = rotations(c.get("rotate", "any"))
        mg = float(c.get("margin", margin))
        g = _Group(cid, mem, meshes, masses, rl, rot, mg)
        groups.append(g)
    # given parts are obstacles as drawn
    occ0 = np.full(len(P), np.inf)
    given_shapes = {}
    for cid in given:
        from .shapes import MeshShape
        for m in members(cid):
            sh = MeshShape(meshes[m])
            given_shapes[m] = sh
            occ0 = np.minimum(occ0, sh.sdf(P))
    occ0 = occ0.reshape(G.n)
    cav_centre = P[cav < 0].mean(axis=0) if (cav < 0).any() else 0.5 * (np.asarray(lo) + np.asarray(hi))

    # most constrained first: line rules, then against, then by volume
    def rank(g):
        r = [rr for _, c in g.rules for rr in ("alignTo", "under") if c.get(rr) is not None]
        a = any(c.get("against") is not None for _, c in g.rules)
        return (-len(r), -int(a), -g.volume)

    groups.sort(key=rank)
    # near targets placed before the parts that want to be near them
    order = []
    pending = list(groups)
    while pending:
        for g in pending:
            wants = {n for _, c in g.rules for n in (c.get("near") or [])}
            roots = {_root(n, byid) for n in wants}
            if not (roots & {x.id for x in pending if x is not g}):
                order.append(g)
                pending.remove(g)
                break
        else:
            order.append(pending.pop(0))

    com_rule = rules.get("centerOfMass") or {}
    w_com = float(com_rule.get("weight", 1.0)) if com_rule else 0.0
    w_center = float(rules.get("centreWeight", rules.get("centerWeight", 0.02)))
    heat = rules.get("heat")
    if heat is None or heat is True:
        heat = {} if any(c.get("watts") for c in byid.values()) else None
    elif heat is False:
        heat = None
    if heat is not None:
        heat = {"weight": 30.0, "ambient": 25.0, "conductivity": 0.1, **heat}
        heat["up"] = _axis_vec(heat["up"]) if heat.get("up") is not None else None

    quick = getattr(judge, "quick", None)

    def line_rule(g, R):
        """(member, face point in the pivot frame rotated, axis, line point,
        tolerance) or None; and whether R is allowed for it."""
        for m, c in g.rules:
            if c.get("alignTo") is not None:
                a = c["alignTo"]
                oid = a.get("opening")
                if oid not in op_shapes:
                    raise ValueError(f"{m}.alignTo.opening {oid!r} is not an opening ({list(op_shapes)})")
                o = op_shapes[oid]
                olo, ohi = o["shape"].bounds()
                oc = 0.5 * (np.asarray(olo) + np.asarray(ohi))
                out = -_axis_vec(o["inward"]) if o.get("inward") else _snap(oc - cav_centre)
                face = _axis_vec(a.get("face", "+x"))
                if np.linalg.norm(R @ face - out) > 1e-6:
                    return None, False
                fp = R @ _face_point(g.member_verts(m), face)
                return {"member": m, "kind": "alignTo", "fp": fp, "axis": out, "p0": oc, "opening": oid,
                        "recess": float(a.get("recess", 0.3))}, True
            if c.get("under") is not None:
                u = c["under"]
                d = _axis_vec(u.get("direction", "-z"))
                face = _axis_vec(u.get("face", _axis_name(-d)))
                if np.linalg.norm(R @ face + d) > 1e-6:
                    return None, False
                fp = R @ _face_point(g.member_verts(m), face)
                return {"member": m, "kind": "under", "fp": fp, "axis": -d, "p0": np.asarray(u["at"], float),
                        "gap": float(u.get("gap", g.margin))}, True
        return None, True

    def candidates(g, occ_state, placed, K):
        """Up to K (cost, R, t, info) for group g given the room left."""
        out = []
        for ri, R in enumerate(g.rots):
            lr, ok = line_rule(g, R)
            if not ok:
                continue
            mg = g.margin
            if lr is not None and lr["kind"] == "alignTo":
                # the recess is the face's own stand-off from the exterior,
                # not on top of the margin the rest of the part keeps
                room = np.minimum(np.maximum(room_base, opening_room(lr["opening"], lr["recess"] - mg)), occ_state)
            else:
                room = np.minimum(room_base, occ_state)
            if lr is not None and lr["kind"] == "under":
                # the switch's actuator keeps `gap` off the skin; the rest of it the margin
                mg = min(mg, lr["gap"])
            blocked = room < mg
            # the part's own cells: a grid aligned with the cavity's
            V = g.verts @ R.T
            vlo, vhi = V.min(axis=0), V.max(axis=0)
            na = np.ceil((vhi - vlo) / G.h).astype(int) + 2
            if np.any(na > np.asarray(G.n)):
                continue
            o_c = vlo - G.h
            ax = [o_c[i] + (np.arange(na[i]) + 0.5) * G.h for i in range(3)]
            Q = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
            A = (g.sdf(R, np.zeros(3), Q) <= 0.15 * G.h).reshape(tuple(na))
            if not A.any():
                continue
            C = fftconvolve(blocked.astype(np.float64), A[::-1, ::-1, ::-1].astype(np.float64), mode="valid")
            ks = np.argwhere(C < 0.5)
            if len(ks) == 0:
                continue
            # pivot position for offset k: cell j=0 of A on cavity cell k
            T = (G.lo + 0.5 * G.h) + ks * G.h - (o_c + 0.5 * G.h)
            com = T + R @ g.com
            cost = w_center * np.linalg.norm(com - cav_centre, axis=1)
            for m, c in g.rules:
                if c.get("against") is not None:
                    d = _axis_vec(c["against"])
                    ext = float((g.member_verts(m) @ R.T @ d).max())
                    cost = cost - (T @ d + ext)
                if c.get("near"):
                    for n in c["near"]:
                        pc = placed.get(n)
                        if pc is not None:
                            cost = cost + 0.2 * np.linalg.norm(com - pc, axis=1)
            if lr is not None:
                fpw = T + lr["fp"]
                rel = fpw - lr["p0"]
                lat = np.linalg.norm(rel - np.outer(rel @ lr["axis"], lr["axis"]), axis=1)
                keep = lat <= 0.75 * G.h + 1e-9
                if not keep.any():
                    continue
                T, com, cost, fpw = T[keep], com[keep], cost[keep], fpw[keep]
                cost = cost - fpw @ lr["axis"]
            if heat is not None:
                cost = cost + _heat_cost(heat, g, R, T, byid, meshes, placed)
            if quick is not None:
                # the cables' straight runs: a pigtail that cannot reach rules a position out here
                cost = cost + quick(g.members, R, T - R @ g.pivot, placed.get("__tf__") or {})
                keep = np.isfinite(cost)
                if not keep.any():
                    continue
                T, com, cost = T[keep], com[keep], cost[keep]
            if w_com:
                pm = placed.get("__mass__", 0.0)
                pcm = placed.get("__com__", np.zeros(3))
                cm = (pm * pcm + g.mass * com) / max(pm + g.mass, 1e-12)
                if com_rule.get("target") is not None:
                    cost = cost + w_com * np.linalg.norm(cm - np.asarray(com_rule["target"], float), axis=1)
                elif com_rule.get("low") is not None:
                    d = _axis_vec(com_rule["low"])
                    cost = cost - w_com * (cm @ d)
            # best few for this orientation, spread apart
            idx = np.argsort(cost)
            took = []
            for i in idx:
                if all(np.linalg.norm(T[i] - T[j]) > 1.0 for j in took):
                    took.append(i)
                if len(took) >= K:
                    break
            for i in took:
                out.append((float(cost[i]), ri, R, T[i].copy(), lr))
        out.sort(key=lambda x: x[0])
        # spread over orientations too
        res, seen = [], {}
        for x in out:
            if seen.get(x[1], 0) >= 4:
                continue
            seen[x[1]] = seen.get(x[1], 0) + 1
            res.append(x)
            if len(res) >= K:
                break
        return res

    def stamp(occ, g, R, t):
        return np.minimum(occ, g.sdf(R, t, P).reshape(G.n))

    gb_all = {x.id: x for x in groups}
    denied = []
    states = [{"cost": 0.0, "occ": occ0, "poses": {}, "centres": {}, "mass": 0.0, "com": np.zeros(3)}]
    blocked = {}
    for gi, g in enumerate(order):
        if on_progress:
            on_progress({"stage": "placement", "message": f"placing {g.id} ({gi + 1}/{len(order)})"})
        nxt = []
        for s in states:
            placed = dict(s["centres"])
            placed["__mass__"] = s["mass"]
            placed["__com__"] = s["com"]
            if quick is not None:
                placed["__tf__"] = _member_tf(s["poses"], gb_all, given_shapes)
            cands = candidates(g, s["occ"], placed, max(3 * beam, 6))
            if not cands:
                continue
            for cost, ri, R, t, lr in cands:
                nxt.append((s, cost, R, t, lr))
        if not nxt:
            blocked[g.id] = _closest(g, np.minimum(room_base, states[0]["occ"]), G, P)
            continue
        nxt.sort(key=lambda x: x[0]["cost"] + x[1])
        new_states = []
        unwired = 0
        gbid = {x.id: x for x in groups}

        def admit(s, cost, R, t, occ=None, poses=None):
            """The state with g at (R, t), or None when it cannot be assembled into."""
            if access is not None:
                others = {k: (lambda p, gg=gb_all[k], RR=RT[0], tt=RT[1]: gg.sdf(RR, tt, p), _pbounds(gb_all[k], *RT))
                          for k, RT in s["poses"].items()}
                if not access(g.id, lambda p, RR=R, tt=t: g.sdf(RR, tt, p), _pbounds(g, R, t), others):
                    denied.append(g.id)
                    return None
            if poses is None:
                poses = dict(s["poses"])
                poses[g.id] = (R, t)
            com = t + R @ g.com
            m_new = s["mass"] + g.mass
            com_tot = (s["mass"] * s["com"] + g.mass * com) / max(m_new, 1e-12)
            centres = dict(s["centres"])
            for m in g.members:
                centres[m] = t + R @ (np.asarray(meshes[m].bounds).mean(axis=0) - g.pivot)
            return {"cost": s["cost"] + cost, "occ": stamp(s["occ"], g, R, t) if occ is None else occ, "poses": poses,
                    "centres": centres, "mass": m_new, "com": com_tot}

        pend = []
        for s, cost, R, t, lr in nxt:
            if len(new_states) >= beam or len(pend) >= 2 * beam:
                break
            t, fits = _refine(g, R, t, lr, G, exterior, skin, op_shapes, fixed, given_shapes,
                              s["poses"], groups_by_id=gbid, rules_g=g.rules)
            if not fits:
                continue        # the grid's fit does not hold on exact distances
            if judge is None:
                st = admit(s, cost, R, t)
                if st is not None:
                    new_states.append(st)
                continue
            # with a judge, a few more than the beam are judged (cables laid),
            # ranked with its cost in, and only then asked whether they can be assembled
            occ = stamp(s["occ"], g, R, t)
            poses = dict(s["poses"])
            poses[g.id] = (R, t)
            extra = judge({"transforms": _member_tf(poses, gb_all, given_shapes), "new": list(g.members),
                           "room": np.minimum(room_base, occ), "origin": G.lo + 0.5 * G.h, "pitch": G.h})
            if extra is None:
                unwired += 1
                continue
            pend.append((s["cost"] + cost + float(extra), s, cost + float(extra), R, t, occ, poses))
        pend.sort(key=lambda x: x[0])
        for _, s, cost, R, t, occ, poses in pend:
            if len(new_states) >= beam:
                break
            st = admit(s, cost, R, t, occ, poses)
            if st is not None:
                new_states.append(st)
        new_states.sort(key=lambda x: x["cost"])
        new_states = new_states[:beam]
        if not new_states and unwired and g.id not in denied:
            blocked[g.id] = {"reason": "fits, but its cables cannot be laid from any position tried (too short, "
                                       "or bent past their radius)", "volume": round(g.volume, 1)}
            continue
        if not new_states:
            why = "no position it fits that it can also be assembled into" if g.id in denied else \
                "fits on the grid only, not on exact distances"
            blocked[g.id] = {"reason": why, "volume": round(g.volume, 1)}
            continue
        states = new_states

    gb = {g.id: g for g in groups}
    layouts = []
    for s in states:
        layouts.append(_layout(s, gb, byid, meshes, given, exterior, skin, op_shapes, fixed, given_shapes,
                               members))
    return {"layouts": layouts, "best": layouts[0] if layouts else None, "blocked": blocked,
            "order": [g.id for g in order], "given": given, "pitch": G.h, "seconds": round(time.time() - t0, 1)}


def _member_tf(poses, groups, given):
    """{component id: (R, T)} of its own frame, from the groups' poses."""
    tf = {m: (RR, tt - RR @ groups[k].pivot) for k, (RR, tt) in poses.items() for m in groups[k].members}
    tf.update({m: (np.eye(3), np.zeros(3)) for m in given})
    return tf


def _heat_cost(heat, g, R, T, byid, meshes, placed):
    """Heat in the search, per candidate position T (n, 3): each pair of a
    part being placed and a part already in costs the share of the cooler
    one's temperature budget the other's heat would use up, by a point
    source in a uniform medium, dT = W / (4 pi k r); half as much again
    when the heat rises into it (the hot part below, `up` known). A rough
    ranking (the skin, vents and conduction paths are not in it): it keeps
    a hot chip off a cell and two hot parts apart; the cooling block
    (cooling.py) is the check."""
    k = float(heat["conductivity"])
    amb = float(heat["ambient"])
    up = heat.get("up")
    cost = np.zeros(len(T))

    def budget(c):
        if c.get("maxTemp") is not None:
            return max(float(c["maxTemp"]) - amb, 1.0)
        return 60.0 if c.get("watts") else None
    for m in g.members:
        ci = byid[m]
        wi, bi = float(ci.get("watts") or 0.0), budget(ci)
        if not wi and bi is None:
            continue
        pm = T + R @ (np.asarray(meshes[m].bounds).mean(axis=0) - g.pivot)
        for n, pc in placed.items():
            if n.startswith("__") or n in g.members:
                continue
            cj = byid.get(n) or {}
            wj, bj = float(cj.get("watts") or 0.0), budget(cj)
            d = pc[None, :] - pm
            r = np.maximum(np.linalg.norm(d, axis=1), 3.0)
            rise = 1.0 / (4 * np.pi * k * r * 1e-3)
            cu = (d @ np.asarray(up, float)) / r if up is not None else np.zeros(len(T))
            if wi and bj is not None:          # this part heats that one (above it: worse)
                cost += wi / bj * rise * (1.0 + 0.5 * np.maximum(cu, 0.0))
            if wj and bi is not None:
                cost += wj / bi * rise * (1.0 + 0.5 * np.maximum(-cu, 0.0))
    return float(heat["weight"]) * cost


def _root(cid, byid):
    c = byid.get(cid)
    while c is not None and c.get("installedOn"):
        c = byid[c["installedOn"]]
    return c["id"] if c is not None else cid


def _snap(v):
    k = int(np.argmax(np.abs(v)))
    d = np.zeros(3)
    d[k] = 1.0 if v[k] > 0 else -1.0
    return d


def _closest(g, room, G, P):
    return {"reason": "no orientation fits anywhere in the room left", "volume": round(g.volume, 1),
            "largestRoom": round(float(room.max()), 2)}


def _pbounds(g, R, t):
    V = g.verts @ R.T + t
    return V.min(axis=0), V.max(axis=0)


def _mouth(o: dict, exterior, centre) -> None:
    """An opening's outward axis and where the exterior's surface would be
    across it (the opening is cut out of the exterior, so it has none
    there): the surface just round its rim, along the axis."""
    from .shapes import sweep
    sh = o["shape"]
    olo, ohi = sh.bounds()
    oc = 0.5 * (np.asarray(olo) + np.asarray(ohi))
    a = -_axis_vec(o["inward"]) if o.get("inward") else _snap(oc - centre)
    o["axis"] = a
    # the room an opening gives runs on through the skin behind it, as the
    # study's skin.openings do (swept `depth` inward)
    depth = float(o.get("depth") or 0.0) + float(o.get("reach") or 0.0)
    o["region"] = sweep(sh, -a, depth) if depth > 0 else sh
    o["plane"] = None
    m = getattr(exterior, "mesh", None)
    if m is None:
        return
    import trimesh
    pts, _ = trimesh.sample.sample_surface(m, 60_000, seed=3)
    d = sh.sdf(pts)
    rim = pts[(d > 0.2) & (d < 3.0)]
    # the rim on the outside of the exterior, facing the way the opening does
    if len(rim):
        h = rim @ a
        rim = rim[h >= np.percentile(h, 50)]
        o["plane"] = float(np.percentile(rim @ a, 90))


def _opening_room(o: dict, pts, e, recess: float):
    """Room inside an opening, up to `recess` short of the exterior's
    surface across it."""
    r = -o.get("region", o["shape"]).sdf(pts)
    if o.get("plane") is not None:
        return np.minimum(r, o["plane"] - float(recess) - pts @ o["axis"])
    return np.minimum(r, -(e + float(recess)))


def _exact_room(pts, exterior, skin, op_shapes, fixed, given_shapes, others, opening=None, recess=0.3):
    """Exact room (mm) at points: off the cavity wall, fixed geometry, given
    parts and other placed parts; an opening adds room inside it."""
    e = exterior.sdf(pts)
    r = -(e + skin)
    for o in op_shapes.values():
        r = np.maximum(r, _opening_room(o, pts, e, 0.3))
    if opening is not None:
        r = np.maximum(r, _opening_room(op_shapes[opening], pts, e, recess))
    for _, sh in fixed:
        r = np.minimum(r, sh.sdf(pts))
    for sh in given_shapes.values():
        r = np.minimum(r, sh.sdf(pts))
    for fn in others:
        r = np.minimum(r, fn(pts))
    return r


def _refine(g, R, t, lr, G, exterior, skin, op_shapes, fixed, given_shapes, poses, groups_by_id, rules_g):
    """Snap a line rule's face onto its line, then push along the rule's
    direction as far as the exact room allows (up to a grid cell)."""
    pts0 = None
    for m in g.members:
        v = _dense(g.meshes[m], 0.6) - g.pivot
        pts0 = v if pts0 is None else np.vstack([pts0, v])
    others = [(lambda p, gg=groups_by_id[k], RR=RT[0], tt=RT[1]: gg.sdf(RR, tt, p)) for k, RT in poses.items()]
    opening = lr["opening"] if lr is not None and lr["kind"] == "alignTo" else None
    recess = lr["recess"] if lr is not None and lr["kind"] == "alignTo" else 0.3
    mg = g.margin

    def ok(tt):
        r = _exact_room(pts0 @ R.T + tt, exterior, skin, op_shapes, fixed, given_shapes, others, opening,
                        recess - mg)
        return float(r.min()) >= mg - 1e-6

    t = np.asarray(t, float)
    if lr is not None:
        fpw = t + lr["fp"]
        rel = fpw - lr["p0"]
        lat = rel - (rel @ lr["axis"]) * lr["axis"]
        if np.linalg.norm(lat) > 1e-9 and ok(t - lat):
            t = t - lat
    pushes = []
    if lr is not None:
        pushes.append(lr["axis"])
    for _, c in rules_g:
        if c.get("against") is not None:
            pushes.append(_axis_vec(c["against"]))
    if not ok(t):
        # the grid is half a cell out at most: look round it on exact distances
        best = None
        js = np.array([-0.5, -0.25, 0.0, 0.25, 0.5]) * G.h
        for dx in js:
            for dy in js:
                for dz in js:
                    dd = np.array([dx, dy, dz])
                    if (best is None or np.linalg.norm(dd) < np.linalg.norm(best)) and ok(t + dd):
                        best = dd
        if best is None:
            return t, False
        t = t + best
    for d in pushes:
        step, moved = 0.5 * G.h, 0.0
        while step > 0.02 and moved < 3 * G.h:
            if ok(t + step * d):
                t = t + step * d
                moved += step
            else:
                step *= 0.5
    return t, True


def _layout(s, gb, byid, meshes, given, exterior, skin, op_shapes, fixed, given_shapes, members):
    """A finished layout: each component's matrix, and its measured room."""
    out = {}
    for gid, (R, t) in s["poses"].items():
        g = gb[gid]
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t - R @ g.pivot
        for m in g.members:
            out[m] = {"group": gid, "rotation": np.round(R, 6).tolist(), "translation": np.round(t - R @ g.pivot, 4)
                      .tolist(), "matrix": np.round(M, 6).tolist(), "auto": True}
    for gid in given:
        for m in members(gid):
            out[m] = {"group": gid, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
                      "matrix": np.eye(4).tolist(), "auto": False}
    # measured room per component: to the cavity wall and to every other part
    from .shapes import MeshShape
    placed = {}
    for m, rec in out.items():
        mm = meshes[m].copy()
        mm.apply_transform(np.asarray(rec["matrix"]))
        placed[m] = mm
    shapes = {m: MeshShape(mm) for m, mm in placed.items()}
    for m, rec in out.items():
        pts = _dense(placed[m], 0.5)
        wall = -(exterior.sdf(pts) + skin)
        # into an opening is not a clash with the wall
        inside_op = np.zeros(len(pts), bool)
        for o in op_shapes.values():
            inside_op |= o.get("region", o["shape"]).sdf(pts) <= 0
        wall_room = float(wall[~inside_op].min()) if (~inside_op).any() else float("inf")
        outside = float(exterior.sdf(pts).max())
        nb = {}
        g = rec["group"]
        for n, sh in shapes.items():
            if n == m or out[n]["group"] == g:
                continue
            nb[n] = float(sh.sdf(pts).min())
        fx = {fid: float(sh.sdf(pts).min()) for fid, sh in fixed}
        lo, hi = placed[m].bounds
        rec.update({"bounds": [np.round(lo, 3).tolist(), np.round(hi, 3).tolist()],
                    "centre": np.round(0.5 * (lo + hi), 3).tolist(),
                    "roomToSkin": round(wall_room, 3), "beyondExterior": round(max(outside, 0.0), 3),
                    "roomToParts": {k: round(v, 3) for k, v in sorted(nb.items(), key=lambda kv: kv[1])[:4]},
                    **({"roomToFixed": {k: round(v, 3) for k, v in fx.items()}} if fx else {})})
        al = byid[m].get("alignTo") if isinstance(byid[m].get("alignTo"), dict) else None
        o = op_shapes.get(al.get("opening")) if al else None
        if o is not None and o.get("plane") is not None:
            # measured: how far the part's outermost point stands back from
            # the exterior across the opening (the brief's `recess`)
            rec["recess"] = round(float(o["plane"] - (pts @ o["axis"]).max()), 3)
    mass = sum(float(byid[m].get("mass") or 0.0) for m in out)
    com = None
    if mass > 0:
        com = sum(float(byid[m].get("mass") or 0.0) * np.asarray(out[m]["centre"]) for m in out) / mass
    return {"cost": round(float(s["cost"]), 3), "components": out, "mass": round(mass, 5),
            "centerOfMass": None if com is None else np.round(com, 3).tolist(), "meshes": placed}


# --- assembly paths in configuration space ---------------------------------

_DIRS = [np.array(v, float) for v in ([0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0])]


# most cells insertion_path's grid may have (the exterior and the part's
# size round it); past it the pitch grows
_PATH_CELLS = 12_000_000


def insertion_path(part_sdf, part_bounds, obstacles_sdf, exit_bounds, *, pitch: float = 0.4, tol: float = 0.25,
                   max_legs: int = 4, hint=None) -> dict | None:
    """The way a part gets from where it sits to outside the exterior's
    bounds, in at most `max_legs` straight axis-aligned legs, or None.

    Every translation of the part is tested at once on a grid (the part's
    cells cross-correlated with the blocked cells by FFT: zero overlap is a
    free position), then the free positions are flooded outward from where
    the part sits, one straight leg per round, so the first path found has
    the fewest legs. part_sdf / obstacles_sdf: callables on (n, 3) points;
    obstacles_sdf already less the part's own place. The legs come back as
    the part leaves: [(axis name, mm or None for 'on out')]."""
    from scipy.signal import fftconvolve

    plo, phi = np.asarray(part_bounds[0], float), np.asarray(part_bounds[1], float)
    elo, ehi = np.asarray(exit_bounds[0], float), np.asarray(exit_bounds[1], float)
    ext = phi - plo
    h = float(pitch)
    # the world grid spans the exterior and the part's own size round it:
    # past _PATH_CELLS cells the pitch grows (a 72 x 96 mm speaker with a
    # 44 mm driver came to 84M cells at 0.4 mm, past 15 GB); the exact
    # check along the legs (enclosure._way_in) does not depend on it
    span = (ehi - elo) + 2.0 * (ext + 2.0 * h + 1.0)
    cells = float(np.prod(span / h))
    if cells > _PATH_CELLS:
        h *= (cells / _PATH_CELLS) ** (1.0 / 3.0)
    # the part's own cells; the world grid is aligned to them
    na = np.ceil(ext / h).astype(int) + 1
    o_c = plo - 0.5 * h * (na * h - ext) / h
    axa = [o_c[i] + (np.arange(na[i]) + 0.5) * h for i in range(3)]
    Q = np.stack(np.meshgrid(*axa, indexing="ij"), -1).reshape(-1, 3)
    A = (part_sdf(Q) <= 0.15 * h).reshape(tuple(na))
    pad = ext + 2.0 * h + 1.0
    m_lo = np.ceil((o_c - (elo - pad)) / h).astype(int)
    m_hi = np.ceil(((ehi + pad) - (o_c + na * h)) / h).astype(int)
    g0 = o_c - m_lo * h
    n = m_lo + na + m_hi
    ax = [g0[i] + (np.arange(n[i]) + 0.5) * h for i in range(3)]
    # blocked cells: the obstacles only inside the exterior's box, a slab
    # at a time
    B = np.zeros(tuple(n), np.float32)
    ins = [np.nonzero((a >= elo[i] - 0.5) & (a <= ehi[i] + 0.5))[0] for i, a in enumerate(ax)]
    if all(len(v) for v in ins):
        yz = np.stack(np.meshgrid(ax[1][ins[1]], ax[2][ins[2]], indexing="ij"), -1).reshape(-1, 2)
        for i in ins[0]:
            P = np.column_stack([np.full(len(yz), ax[0][i]), yz])
            B[i, ins[1][0]:ins[1][-1] + 1, ins[2][0]:ins[2][-1] + 1] = \
                (-obstacles_sdf(P) > tol).reshape(len(ins[1]), len(ins[2]))
    C = fftconvolve(B, A[::-1, ::-1, ::-1].astype(np.float32), mode="valid")
    del B
    free = C < 0.5
    del C
    k0 = tuple(int(v) for v in m_lo)
    if not free[k0]:
        # where it sits already reads as blocked on the grid: let it start
        free[k0] = True
    # an offset is out when the part's box is clear of the exterior's box
    # (per axis, broadcast: no array of every offset's corners)
    out = np.zeros(free.shape, bool)
    for a in range(3):
        sh = (np.arange(free.shape[a]) - m_lo[a]) * h
        o = (phi[a] + sh < elo[a]) | (plo[a] + sh > ehi[a])
        out |= o.reshape([-1 if j == a else 1 for j in range(3)])
    order = list(range(6))
    if hint is not None:
        hv = _axis_vec(hint) if isinstance(hint, str) else None
        if hv is not None:
            order.sort(key=lambda i: -float(_DIRS[i] @ hv))
    reached = [np.zeros(free.shape, bool)]
    reached[0][k0] = True
    for level in range(1, max_legs + 1):
        prev = reached[-1]
        cur = np.zeros(free.shape, bool)
        per_dir = []
        for i in order:
            d = _DIRS[i]
            a = int(np.argmax(np.abs(d)))
            sgn = int(d[a])
            r = prev.copy()
            L = free.shape[a]
            rng = range(1, L) if sgn > 0 else range(L - 2, -1, -1)
            sl = [slice(None)] * 3
            for j in rng:
                sl[a] = j
                sp = list(sl)
                sp[a] = j - sgn
                r[tuple(sl)] |= r[tuple(sp)] & free[tuple(sl)]
            per_dir.append((i, r))
            cur |= r
        reached.append(cur)
        hit = cur & out
        if hit.any():
            # the exit closest to where the part sits, in the first direction that gets there
            for i, r in per_dir:
                hk = np.argwhere(r & out)
                if len(hk):
                    k = hk[np.argmin(np.abs(hk - np.asarray(k0)).sum(axis=1))]
                    break
            legs = _trace(k, i, reached, per_dir_levels=None, free=free, k0=np.asarray(k0), level=level)
            if legs is None:
                continue
            return {"legs": legs, "pitch": h}
    return None


def _trace(k, i_last, reached, per_dir_levels, free, k0, level):
    """Back from an exit offset to the start: one straight leg per level."""
    legs = []
    cur = np.asarray(k)
    d = _DIRS[i_last]
    for lv in range(level, 0, -1):
        prev = reached[lv - 1]
        found = None
        dirs = [d] if lv == level else _DIRS
        for dd in dirs:
            step = -dd.astype(int)
            p = cur.copy()
            steps = 0
            while True:
                if np.any(p < 0) or np.any(p >= np.asarray(free.shape)):
                    break
                if prev[tuple(p)] and (lv > 1 or np.array_equal(p, k0)):
                    found = (dd, steps, p)
                    break
                if not free[tuple(p)]:
                    break
                p = p + step
                steps += 1
            if found is not None:
                break
        if found is None:
            return None
        dd, steps, p = found
        legs.append((dd, steps))
        cur = p
        if lv > 1:
            # the leg before must end here: pick any direction reaching it next round
            pass
    legs.reverse()
    return legs
