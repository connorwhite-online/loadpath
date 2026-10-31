"""
Loads and design space from the mechanism a part sits in.

A study needs loads and a design space, and for a part in a mechanism both
are already decided by the mechanism: the loads by what the part carries in
every pose it reaches, the space by what the parts round it occupy and sweep
through. Writing them by hand means picking one pose ("arm stretched out"),
a factor for everything else (1.5x for accelerations) and boxes for the
space, and the worst case for one member is rarely the worst case for
another. This module reads the mechanism from a URDF and derives both:

    robot = Robot.from_urdf("grip.urdf")
    sweep = sweep_loads(robot, "forearm", payload=Payload(1.0, "tool", (0, 0, 0.07)))
    picked = governing_cases(sweep, study, k=6)          # a few cases that set every peak
    keep = swept_keepouts(robot, "forearm", box=(lo, hi), protect=study_preserve)

**Loads.** `sweep_loads` samples poses over the joint limits (the corners of
the limit box plus random poses), with joint speeds and accelerations
(bang-bang: each joint at plus or minus its limit), a payload whose mass and
position scatter, a contact push at the tool in a random direction, and each
joint upstream stalling at its rated torque against an obstacle at the tool.
For each sample it does Newton-Euler over the bodies past the part's far
joint, so the wrench the part carries there is exact for that state (gravity,
linear and angular inertia, payload, contact), and expresses it in the
study's frame (the part at the home pose, mm, N and N·mm about the load
region's centroid).

**Governing cases.** A few thousand samples are not a study. Stress is
linear in the wrench, so six unit-wrench solves on the solid design space
give every sample's stress everywhere for the price of six solves: `stress`
selection then picks, greedily, the samples that bring every cell within 5%
of its worst stress over the whole sweep, until 98% of the stress-weighted
cells are covered (usually four to eight cases). The `hull` selection (no
solve) does the same on the wrench cloud's support function in random
directions. The report says how close the chosen cases come to the
envelope, so an agent can see what was dropped.

**Design space.** `swept_keepouts` sweeps each neighbour body (the parent
and the children, from the URDF's collision or visual meshes, or a shape
given in their place) through the joint between it and the part, over the
joint's limits, in the part's frame, and returns the swept volume, plus a
clearance, as mesh keep-outs. The part's own interfaces (`protect`) are cut
out of them, so a horn the child bolts onto is not deleted by the child's own
sweep.

URDF units are metres, kg and N·m; the study's are mm, N and N·mm. `frame`
maps the URDF's world at the home pose to the study's frame (default: the
same axes, metres to mm).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Joint", "Body", "Robot", "Payload", "LoadSweep", "sweep_loads", "governing_cases",
           "swept_keepouts", "intrusion", "region_centroid", "derive_study"]

G = 9.81


# --------------------------------------------------------------------------- URDF

def _rpy(r, p, y):
    cr, sr, cp, sp_, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp_ * sr - sy * cr, cy * sp_ * cr + sy * sr],
                     [sy * cp, sy * sp_ * sr + cy * cr, sy * sp_ * cr - cy * sr],
                     [-sp_, cp * sr, cp * cr]])


def _origin(el):
    t = np.eye(4)
    if el is None:
        return t
    t[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    t[:3, :3] = _rpy(*[float(v) for v in el.get("rpy", "0 0 0").split()])
    return t


def _axis_rot(a, q):
    a = np.asarray(a, float)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


def _primitive(geom):
    """A URDF <box>, <cylinder> or <sphere> as (kind, dimensions in metres),
    or None."""
    if geom is None:
        return None
    if (el := geom.find("box")) is not None:
        return "box", tuple(float(v) for v in el.get("size").split())
    if (el := geom.find("cylinder")) is not None:
        return "cylinder", (float(el.get("radius")), float(el.get("length")))
    if (el := geom.find("sphere")) is not None:
        return "sphere", (float(el.get("radius")),)
    return None


@dataclass
class Joint:
    name: str
    type: str                     # revolute, continuous, prismatic, fixed
    parent: str
    child: str
    origin: np.ndarray            # 4x4, child frame at q = 0 in the parent's frame
    axis: np.ndarray              # unit, in the joint (child) frame
    lower: float = -np.pi
    upper: float = np.pi
    effort: float = 0.0           # N·m (N for prismatic)
    velocity: float = 0.0         # rad/s (m/s)
    mimic: tuple | None = None    # (joint, multiplier, offset)

    @property
    def moves(self):
        return self.type != "fixed" and self.mimic is None


@dataclass
class Body:
    name: str
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))      # m, link frame
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))   # kg m², about the COM, link axes
    meshes: list = field(default_factory=list)    # (path, 4x4 origin in link frame, scale xyz)


class Robot:
    """A kinematic tree from a URDF: bodies, joints, forward kinematics and
    the velocities and accelerations of every body for given joint rates."""

    def __init__(self, bodies: dict, joints: list, base_dir: str = "."):
        self.bodies = bodies
        self.joints = {j.name: j for j in joints}
        self.base_dir = base_dir
        children = {j.child for j in joints}
        roots = [b for b in bodies if b not in children]
        if len(roots) != 1:
            raise ValueError(f"a URDF tree needs exactly one root link, found {roots}")
        self.root = roots[0]
        # joints in an order where every parent comes before its child
        order, seen = [], {self.root}
        pending = list(joints)
        while pending:
            nxt = [j for j in pending if j.parent in seen]
            if not nxt:
                raise ValueError("URDF joints do not form a tree from " + self.root)
            for j in nxt:
                order.append(j)
                seen.add(j.child)
                pending.remove(j)
        self.order = order
        self.parent_joint = {j.child: j for j in joints}

    @classmethod
    def from_urdf(cls, path: str) -> "Robot":
        root = ET.parse(path).getroot()
        bodies = {}
        for link in root.findall("link"):
            b = Body(link.get("name"))
            i = link.find("inertial")
            if i is not None:
                b.mass = float(i.find("mass").get("value"))
                o = _origin(i.find("origin"))
                b.com = o[:3, 3].copy()
                ie = i.find("inertia")
                if ie is not None:
                    v = {k: float(ie.get(k, 0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                    it = np.array([[v["ixx"], v["ixy"], v["ixz"]], [v["ixy"], v["iyy"], v["iyz"]],
                                   [v["ixz"], v["iyz"], v["izz"]]])
                    b.inertia = o[:3, :3] @ it @ o[:3, :3].T
            for tag in ("collision", "visual"):
                for el in link.findall(tag):
                    m = el.find("geometry/mesh")
                    if m is not None:
                        scale = np.array([float(v) for v in m.get("scale", "1 1 1").split()])
                        b.meshes.append((m.get("filename"), _origin(el.find("origin")), scale))
                        continue
                    prim = _primitive(el.find("geometry"))
                    if prim is not None:
                        b.meshes.append((prim, _origin(el.find("origin")), np.ones(3)))
                if b.meshes:
                    break
            bodies[b.name] = b
        joints = []
        for j in root.findall("joint"):
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            lim = j.find("limit")
            jt = j.get("type")
            lo, hi = (-np.pi, np.pi) if jt == "continuous" else (0.0, 0.0)
            eff = vel = 0.0
            if lim is not None:
                if jt != "continuous":
                    lo, hi = float(lim.get("lower", 0)), float(lim.get("upper", 0))
                eff, vel = float(lim.get("effort", 0)), float(lim.get("velocity", 0))
            mim = j.find("mimic")
            mimic = None if mim is None else (mim.get("joint"), float(mim.get("multiplier", 1)),
                                              float(mim.get("offset", 0)))
            joints.append(Joint(j.get("name"), jt, j.find("parent").get("link"), j.find("child").get("link"),
                                _origin(j.find("origin")), axis / (np.linalg.norm(axis) or 1.0),
                                lo, hi, eff, vel, mimic))
        return cls(bodies, joints, os.path.dirname(os.path.abspath(path)))

    # ----------------------------------------------------------------- kinematics

    @property
    def movable(self) -> list:
        return [j.name for j in self.order if j.moves]

    def _value(self, j: Joint, q: dict):
        if j.type == "fixed":
            return 0.0
        if j.mimic:
            src, mult, off = j.mimic
            return mult * q.get(src, 0.0) + off
        return q.get(j.name, 0.0)

    def state(self, q: dict | None = None, qd: dict | None = None, qdd: dict | None = None):
        """World transform, angular velocity, angular acceleration and linear
        acceleration of the origin of every link (base fixed)."""
        q, qd, qdd = q or {}, qd or {}, qdd or {}
        T = {self.root: np.eye(4)}
        w = {self.root: np.zeros(3)}
        al = {self.root: np.zeros(3)}
        a = {self.root: np.zeros(3)}
        for j in self.order:
            p = j.parent
            tj = T[p] @ j.origin
            z = tj[:3, :3] @ j.axis
            qv = self._value(j, q)
            if j.mimic:
                src, mult, _ = j.mimic
                dq, ddq = mult * qd.get(src, 0.0), mult * qdd.get(src, 0.0)
            else:
                dq, ddq = qd.get(j.name, 0.0), qdd.get(j.name, 0.0)
            if j.type == "fixed":
                dq = ddq = 0.0
            m = np.eye(4)
            if j.type in ("revolute", "continuous"):
                m[:3, :3] = _axis_rot(j.axis, qv)
            elif j.type == "prismatic":
                m[:3, 3] = j.axis * qv
            tc = tj @ m
            r = tc[:3, 3] - T[p][:3, 3]
            ac = a[p] + np.cross(al[p], r) + np.cross(w[p], np.cross(w[p], r))
            if j.type in ("revolute", "continuous"):
                wc = w[p] + z * dq
                alc = al[p] + z * ddq + np.cross(w[p], z * dq)
            else:
                wc, alc = w[p].copy(), al[p].copy()
                if j.type == "prismatic":
                    ac = ac + 2.0 * np.cross(w[p], z * dq) + z * ddq
            T[j.child], w[j.child], al[j.child], a[j.child] = tc, wc, alc, ac
        return T, w, al, a

    def fk(self, q: dict | None = None) -> dict:
        return self.state(q)[0]

    def subtree(self, joint: str) -> list:
        """Links carried by `joint` (its child and everything past it)."""
        out = [self.joints[joint].child]
        for j in self.order:
            if j.parent in out and j.child not in out:
                out.append(j.child)
        return out

    def path_joints(self, link: str) -> list:
        """Joints from the root down to `link`."""
        out = []
        while link in self.parent_joint:
            j = self.parent_joint[link]
            out.append(j.name)
            link = j.parent
        return out[::-1]

    def far_joint(self, link: str) -> str:
        kids = [j for j in self.order if j.parent == link and j.type != "fixed"]
        if len(kids) != 1:
            raise ValueError(f"{link} has {len(kids)} moving child joints ({[j.name for j in kids]}); "
                             "name the one whose load to sweep with joint=")
        return kids[0].name

    def mesh(self, link: str):
        """The link's mesh in its own frame, metres (trimesh), or None."""
        import trimesh

        parts = self.meshes(link)
        if not parts:
            return None
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    def meshes(self, link: str) -> list:
        """The link's collision (else visual) shapes in its own frame, metres,
        one closed trimesh each (they may overlap)."""
        import trimesh

        parts = []
        for fn, org, scale in self.bodies[link].meshes:
            if not isinstance(fn, str):
                # a URDF box, cylinder or sphere (metres, centred on its origin)
                kind, dims = fn
                m = (trimesh.creation.box(extents=dims) if kind == "box" else
                     trimesh.creation.cylinder(radius=dims[0], height=dims[1], sections=48) if kind == "cylinder"
                     else trimesh.creation.icosphere(subdivisions=3, radius=dims[0]))
                m.apply_transform(org)
                parts.append(m)
                continue
            p = fn.split("://", 1)[1] if "://" in fn else fn
            cands = [os.path.join(self.base_dir, p), os.path.join(self.base_dir, os.path.basename(p))]
            if "://" in fn:
                cands.append(os.path.join(self.base_dir, *p.split("/")[1:]))
            hit = next((c for c in cands if os.path.exists(c)), None)
            if hit is None:
                continue
            m = trimesh.load(hit, force="mesh")
            m.apply_scale(1.0)
            m.vertices = np.asarray(m.vertices) * scale
            m.apply_transform(org)
            parts.append(m)
        return parts


# --------------------------------------------------------------------------- load sweep

@dataclass
class Payload:
    kg: float
    link: str
    at: tuple = (0.0, 0.0, 0.0)   # m, in the link's frame
    radius: float = 0.0           # m: the payload's centre scatters inside this sphere
    min_kg: float | None = None   # sampled between min_kg and kg (default: always kg)


@dataclass
class LoadSweep:
    link: str
    joint: str
    point: np.ndarray             # mm, study frame: what the moments are about
    wrenches: np.ndarray          # (n, 6): force N, moment N·mm, study frame
    kinds: list                   # per sample: "pose", "dynamic", "push", "stall:<joint>"
    states: list                  # per sample: dict(q, qd, qdd, payload)
    region: str | None = None

    def cases(self, idx, prefix="sweep", cycles: dict | None = None) -> list:
        """Study load cases for samples idx. cycles: {kind: times applied over
        the part's life} by sample kind ("pose", "dynamic", "push", "stall",
        or "stall:j2" for one joint), so durability.py checks fatigue."""
        out = []
        for n, i in enumerate(idx):
            f, m = self.wrenches[i, :3], self.wrenches[i, 3:]
            case = {"name": f"{prefix}-{n}-{self.kinds[i].replace(':', '-')}",
                    "loads": [{"region": self.region, "force": [round(float(v), 2) for v in f],
                               "moment": [round(float(v), 1) for v in m]}]}
            kind = self.kinds[i]
            c = (cycles or {}).get(kind, (cycles or {}).get(kind.split(":")[0]))
            if c:
                case["cycles"] = float(c)
            out.append(case)
        return out


def _frame(frame):
    if frame is None:
        f = np.eye(4)
        f[:3, :3] *= 1000.0
        return f
    return np.asarray(frame, float)


class _State:
    """Per-body loads of one mechanism state, so the wrench at any joint is
    a sum over that joint's subtree: f = m (g - a) on each body (and the
    payload, and a contact), with its moment about the world origin less the
    body's rate of change of angular momentum."""

    def __init__(self, robot, q, qd, qdd, gravity, payload=None):
        self.robot = robot
        self.T, w, al, a = robot.state(q, qd, qdd)
        names, f, m0 = [], [], []
        for name, b in robot.bodies.items():
            if b.mass <= 0 or name not in self.T:
                continue
            R = self.T[name][:3, :3]
            o = self.T[name][:3, 3]
            rc = R @ b.com
            ac = a[name] + np.cross(al[name], rc) + np.cross(w[name], np.cross(w[name], rc))
            fi = b.mass * (gravity - ac)
            Iw = R @ b.inertia @ R.T
            names.append(name)
            f.append(fi)
            m0.append(np.cross(o + rc, fi) - (Iw @ al[name] + np.cross(w[name], Iw @ w[name])))
        if payload is not None:
            kg, at, link = payload
            pos = (self.T[link] @ np.append(at, 1.0))[:3]
            r = pos - self.T[link][:3, 3]
            ap = a[link] + np.cross(al[link], r) + np.cross(w[link], np.cross(w[link], r))
            fi = kg * (gravity - ap)
            names.append(link)
            f.append(fi)
            m0.append(np.cross(pos, fi))
            self.tool = pos
        self.names = np.array(names)
        self.f = np.array(f).reshape(-1, 3)
        self.m0 = np.array(m0).reshape(-1, 3)

    def wrench(self, joint, contact=None):
        """Force and moment (world, SI, about the joint's origin) that what
        `joint` carries puts on its parent; contact = (force, world point) on
        a carried body."""
        sub = np.isin(self.names, self.robot.subtree(joint))
        F = self.f[sub].sum(axis=0)
        M = self.m0[sub].sum(axis=0)
        if contact is not None:
            F = F + contact[0]
            M = M + np.cross(contact[1], contact[0])
        o = self.T[self.robot.joints[joint].child][:3, 3]
        return F, M - np.cross(o, F)

    static = None     # the same pose at rest (for a counterbalance's share)

    def gravity_torque(self, joint):
        return (self.static or self).torque(joint)

    def torque(self, joint, contact=None):
        """What the joint's actuator has to hold (about its axis)."""
        j = self.robot.joints[joint]
        F, M = self.wrench(joint, contact)
        return float(-(self.T[j.child][:3, :3] @ j.axis) @ M)


def _largest_scale(a, b, effort):
    """Largest s in [0, 1] (or inf) with |a_j + s b_j| <= effort_j for every j."""
    s = np.inf
    for aj, bj, ej in zip(a, b, effort):
        if ej <= 0 or abs(bj) < 1e-12:
            continue
        lim = (ej - aj) / bj if bj > 0 else (ej + aj) / -bj
        s = min(s, max(lim, 0.0))
    return s


def sweep_loads(robot: Robot, link: str, *, joint: str | None = None, payload: Payload | None = None,
                samples: int = 1500, corners: bool = True, speed: float = 1.0, accel: float | dict | None = None,
                ramp: float = 0.2, push: float = 20.0, stall: bool = True, contact_max: float | None = None,
                balanced: tuple = (), gravity=(0.0, 0.0, -G),
                masses: dict | None = None, frame=None, point=None, region: str | None = None,
                seed: int = 0) -> LoadSweep:
    """Sample the states the mechanism can be in and return the wrench `link`
    carries at its far `joint` in each, in the study frame.

    Every state is one the actuators can produce and hold: accelerations
    are scaled down until no joint needs more than its rated effort, and a
    contact at the payload pushes only as hard as the weakest joint between
    the base and the payload can resist (a harder push back-drives it).

    speed: fraction of each joint's velocity limit. accel: rad/s² per joint
    (a number, or {joint: value}); default velocity / ramp, i.e. full speed
    in `ramp` seconds, before the effort limit. push: N, a contact at the
    payload in a random direction. stall: each actuated joint from the base
    to `joint` at its rated effort against an obstacle at the payload.
    contact_max: N, the hardest an obstacle pushes back on a stalled joint
    (default: push). Near a joint's axis a stalled joint has almost no
    lever and, held by nothing but the structure, would push without limit
    (a roll joint with the wrist straight), so this is the input to set from
    what the arm can meet.
    balanced: joints whose static gravity torque a counterbalance carries
    (a spring), so only the rest counts against their effort.
    masses: {link: kg} overrides (centres of mass kept). point: mm in the
    study frame to take moments about (the load region's centroid; default
    the joint's origin)."""
    rng = np.random.default_rng(seed)
    contact_max = push if contact_max is None else contact_max
    if masses:
        for n, kg in masses.items():
            robot.bodies[n].mass = float(kg)
    joint = joint or robot.far_joint(link)
    carried = robot.subtree(joint)
    frame = _frame(frame)
    jt = robot.joints
    names = robot.movable
    lim = {n: (jt[n].lower, jt[n].upper) for n in names}
    g = np.asarray(gravity, float)
    if payload is not None and payload.link not in carried:
        raise ValueError(f"payload link {payload.link!r} is not carried by {joint}")

    def acc(n):
        if isinstance(accel, dict):
            return float(accel.get(n, jt[n].velocity / ramp))
        return float(accel) if accel is not None else jt[n].velocity / ramp

    # joints that change the load: those above the part (they turn gravity
    # round it) and those past it (they move the carried mass)
    above = robot.path_joints(link)
    relevant = [n for n in names if n in above or jt[n].child in carried]
    # joints whose rated effort bounds what the arm can do: every actuated
    # revolute joint, and for a contact, those between the base and the payload
    actuated = [n for n in names if jt[n].type in ("revolute", "continuous") and jt[n].effort > 0]
    effort = np.array([jt[n].effort for n in actuated])
    reach = robot.path_joints(payload.link) if payload is not None else []
    chain = [n for n in actuated if n in reach]
    chain_effort = np.array([jt[n].effort for n in chain])

    poses = []
    if corners:
        levels = [(lo, 0.5 * (lo + hi), hi) if lo < hi else (lo,) for lo, hi in (lim[n] for n in relevant)]
        if int(np.prod([len(v) for v in levels])) <= 4096:
            for idx in np.ndindex(*[len(v) for v in levels]):
                poses.append({n: levels[i][idx[i]] for i, n in enumerate(relevant)})
    for _ in range(samples):
        poses.append({n: rng.uniform(*lim[n]) for n in relevant})

    T0 = robot.fk({})
    home_link = T0[link]
    o_study = (frame @ np.append(T0[jt[joint].child][:3, 3], 1.0))[:3]
    pt_study = o_study if point is None else np.asarray(point, float)
    scale = float(np.cbrt(abs(np.linalg.det(frame[:3, :3]))))
    rot_frame = frame[:3, :3] / scale

    W, kinds, states = [], [], []

    def record(st, q, qd, qdd, kind, pay, contact=None, extra=None):
        F, M = st.wrench(joint, contact)
        # world at this state -> the part's frame -> world at home -> study frame
        Rl = home_link[:3, :3] @ st.T[link][:3, :3].T
        Fs = rot_frame @ (Rl @ F)
        Ms = rot_frame @ (Rl @ M) * scale                # N·m -> N·mm
        W.append(np.concatenate([Fs, Ms + np.cross(o_study - pt_study, Fs)]))
        kinds.append(kind)
        rec = {"q": {k: round(float(v), 4) for k, v in q.items()},
               "payloadKg": None if pay is None else round(float(pay[0]), 3)}
        if qd:
            rec["qd"] = {k: round(float(v), 3) for k, v in qd.items()}
            rec["qdd"] = {k: round(float(v), 3) for k, v in qdd.items()}
        if extra:
            rec.update(extra)
        states.append(rec)

    def draw_payload():
        if payload is None:
            return None
        lo = payload.kg if payload.min_kg is None else payload.min_kg
        kg = rng.uniform(lo, payload.kg)
        off = np.zeros(3)
        if payload.radius > 0:
            v = rng.normal(size=3)
            off = v / np.linalg.norm(v) * payload.radius * rng.uniform() ** (1 / 3)
        return kg, np.asarray(payload.at, float) + off

    def pay3(p):
        return None if p is None else (p[0], p[1], payload.link)

    def held(st, joints):
        """Torques the actuators hold in state st, less a counterbalance's share."""
        return np.array([st.torque(n) - (st.gravity_torque(n) if n in balanced else 0.0) for n in joints])

    def pushed(st, u, cap):
        """The contact u (unit, at the tool) as hard as the chain holds, up to cap N."""
        a = held(st, chain)
        b = np.array([st.torque(n, (u, st.tool)) - st.torque(n) for n in chain])
        s = min(_largest_scale(a, b, chain_effort), cap)
        return (s * u, st.tool), s

    moving = [n for n in relevant if jt[n].type in ("revolute", "continuous", "prismatic")]
    stall_every = max(1, len(poses) // 200)
    upstream = [n for n in robot.path_joints(jt[joint].child) if n in chain]
    for i, q in enumerate(poses):
        pay = draw_payload()
        st = _State(robot, q, None, None, g, pay3(pay))
        record(st, q, None, None, "pose", pay)
        # the same pose moving: each joint near its speed limit and at its
        # acceleration limit (bang-bang, signs at random), with the
        # acceleration scaled back to what the actuators can deliver
        if speed > 0 and moving:
            qd = {n: rng.choice([-1.0, 1.0]) * speed * jt[n].velocity * rng.uniform(0.5, 1.0) for n in moving}
            qdd = {n: rng.choice([-1.0, 1.0]) * acc(n) for n in moving}
            s0 = _State(robot, q, qd, None, g, pay3(pay))
            s1 = _State(robot, q, qd, qdd, g, pay3(pay))
            s0.static = st
            a = held(s0, actuated)
            b = np.array([s1.torque(n) for n in actuated]) - np.array([s0.torque(n) for n in actuated])
            s = min(_largest_scale(a, b, effort), 1.0)
            qdd = {n: v * s for n, v in qdd.items()}
            record(_State(robot, q, qd, qdd, g, pay3(pay)), q, qd, qdd, "dynamic", pay,
                   extra={"accelScale": round(float(s), 3)})
        if payload is None:
            continue
        if push > 0 and i % 2 == 0:
            v = rng.normal(size=3)
            c, s = pushed(st, v / np.linalg.norm(v), push)
            record(st, q, None, None, "push", pay, c, {"contactN": round(float(s), 2)})
        if stall and i % stall_every == 0:
            for n in upstream:
                tj = st.T[jt[n].child]
                z = tj[:3, :3] @ jt[n].axis
                r = st.tool - tj[:3, 3]
                r = r - z * (z @ r)
                if r @ r < 1e-10:
                    continue
                for sgn in (-1.0, 1.0):
                    # the obstacle's push that loads joint n hardest per newton,
                    # grown until n (or a weaker joint on the way) is at its effort
                    u = sgn * np.cross(z, r) / np.linalg.norm(np.cross(z, r))
                    c, s = pushed(st, u, contact_max)
                    if np.isfinite(s):
                        record(st, q, None, None, f"stall:{n}", pay, c, {"contactN": round(float(s), 2)})
    return LoadSweep(link, joint, pt_study, np.array(W), kinds, states, region)


# --------------------------------------------------------------------------- governing cases

def _greedy_cover(R, k, threshold=0.95, target=0.98, weights=None, chunk=256):
    """Pick rows of R (samples x responses, >= 0) so that every response
    column reaches `threshold` of its maximum over all rows; greedy on the
    weighted coverage sum_c w_c min(1, best_c / (threshold env_c))."""
    env = R.max(axis=0)
    ok = env > 0
    R, env = R[:, ok], env[ok]
    w = (env ** 2 if weights is None else np.asarray(weights, float)[ok])
    w = w / w.sum()
    cur = np.zeros(R.shape[1])
    chosen, history = [], []
    denom = threshold * env
    for _ in range(k):
        best, best_gain = -1, -1.0
        for s in range(0, R.shape[0], chunk):
            frac = np.minimum(1.0, R[s:s + chunk] / denom)
            gain = (np.maximum(frac - cur, 0.0) * w).sum(axis=1)
            j = int(np.argmax(gain))
            if gain[j] > best_gain:
                best, best_gain = s + j, float(gain[j])
        if best_gain <= 1e-9:
            break
        chosen.append(best)
        cur = np.maximum(cur, np.minimum(1.0, R[best] / denom))
        history.append(float((cur * w).sum()))
        if history[-1] >= target:
            break
    ratio = R[chosen].max(axis=0) / env
    return chosen, history, ratio, env


def _unit_stress(study_dict, region, max_elements=20000, band=2.0, base_dir=None):
    """Element stress (6 unit wrenches x elements x 6) on the solid design
    space, and the stress-free mask near loads and supports."""
    import copy

    from scipy.spatial import cKDTree

    from .domain import build_domain
    from .fem import Solver, Stiffness
    from .spec import study_from_dict

    s = copy.deepcopy(study_dict)
    s["loadCases"] = []
    for k in range(6):
        e = [0.0] * 6
        e[k] = 1.0
        s["loadCases"].append({"name": f"unit-{k}", "loads": [{"region": region, "force": e[:3], "moment": e[3:]}]})
    s["resolution"] = {"maxElements": int(max_elements)}
    s.pop("symmetry", None)
    st = study_from_dict(s, base_dir=base_dir)
    dom = build_domain(st)
    stiff = Stiffness(dom.mesh, st.material.nu if hasattr(st.material, "nu") else 0.35, dom.fixed)
    kmat = stiff.assemble(np.ones(dom.n_elem))
    if dom.springs is not None:
        kmat = (kmat + dom.springs).tocsr()
    us = Solver(dom.mesh, rtol=1e-6, maxiter=3000).solve_many(kmat, list(dom.loads))
    S = np.stack([stiff.stress(u, 1.0) for u in us])
    contact = dom.support_nodes | dom.load_nodes
    centers = dom.mesh.elem_centers()
    d = cKDTree(dom.mesh.node_xyz[contact]).query(centers, k=1)[0]
    body = d > band * dom.pitch
    return S[:, body], centers[body], dom


def governing_cases(sweep: LoadSweep, study: dict | None = None, *, k: int = 8, method: str | None = None,
                    threshold: float = 0.95, target: float = 0.98, max_elements: int = 20000,
                    directions: int = 600, seed: int = 0, base_dir: str | None = None,
                    cycles: dict | None = None) -> dict:
    """Collapse a sweep to at most k cases that come within `threshold` of
    the sweep's worst stress at (a `target` share of) every point.

    method "stress" (default when a study is given): six unit-wrench solves
    on the solid design space, then greedy cover of every cell's von Mises
    envelope. "hull": no solve, greedy cover of the wrench cloud's support
    function over random directions (moments scaled by the part's size).
    cycles: {sample kind: life cycles}, written on the cases (LoadSweep.cases)."""
    from .fem import von_mises

    W = sweep.wrenches
    method = method or ("stress" if study is not None else "hull")
    info = {"method": method, "samples": int(len(W))}
    if method == "stress":
        if study is None or sweep.region is None:
            raise ValueError("stress selection needs the study and the sweep's load region")
        S, centers, dom = _unit_stress(study, sweep.region, max_elements=max_elements, base_dir=base_dir)
        R = np.empty((len(W), S.shape[1]), np.float32)
        for s in range(0, len(W), 64):
            sig = np.einsum("nk,kec->nec", W[s:s + 64], S)
            R[s:s + 64] = von_mises(sig.reshape(-1, 6)).reshape(len(sig), -1)
        info.update(cells=int(S.shape[1]), pitch=float(dom.pitch))
    elif method == "hull":
        L = 1.0
        if study is not None:
            from .spec import study_from_dict
            lo, hi = study_from_dict(study, base_dir=base_dir).bounds()
            L = float(np.linalg.norm(hi - lo)) / 2
        Wn = np.concatenate([W[:, :3], W[:, 3:] / L], axis=1)
        d = np.random.default_rng(seed).normal(size=(directions, 6))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        R = np.maximum(Wn @ d.T, 0.0)
        info.update(lengthMm=L, directions=directions)
    else:
        raise ValueError(f"unknown method {method!r}: stress or hull")
    chosen, hist, ratio, env = _greedy_cover(R, k, threshold, target)
    top = env >= np.quantile(env, 0.8)
    info.update({
        "chosen": [int(i) for i in chosen],
        "coverage": [round(h, 4) for h in hist],
        # how close the chosen cases come to the sweep's worst, cell by cell
        "envelopeRatio": {"min": round(float(ratio.min()), 3), "p5": round(float(np.quantile(ratio, 0.05)), 3),
                          "p5Top20": round(float(np.quantile(ratio[top], 0.05)), 3)},
        "kinds": [sweep.kinds[i] for i in chosen],
        "states": [sweep.states[i] for i in chosen],
    })
    if method == "stress":
        info["peakUnitVonMises"] = round(float(env.max()), 4)
    info["cases"] = sweep.cases(chosen, cycles=cycles)
    return info


# --------------------------------------------------------------------------- swept volumes

def swept_keepouts(robot: Robot, link: str, *, box, pitch: float = 1.0, clearance: float = 1.0,
                   neighbours: list | None = None, shapes: dict | None = None, protect: list | None = None,
                   frame=None, steps: int | None = None, out_dir: str | None = None, prefix: str = "sweep"):
    """The space the neighbours of `link` occupy over their joints' ranges,
    in the study frame, as keep-outs.

    box: (lo, hi) mm, the region to resolve (the design space's bounds).
    neighbours: link names; default the parent link and every link on a
    moving child joint, each carrying what is fixed to it. shapes:
    {link: a loadpath Shape or a closed trimesh, in that link's frame, mm}
    in place of its URDF mesh (the part as designed now, or an envelope for
    one not designed yet). protect: Shapes (study frame)
    cut out of the result (the part's own interfaces). Returns a list of
    dicts {id, link, joint, grid, origin, mesh, volume, keepOut} where
    keepOut is a study entry (mesh written to out_dir/<id>.stl if given)."""
    import trimesh
    from scipy import ndimage
    from skimage import measure

    from .verify import voxelize

    frame = _frame(frame)
    lo, hi = (np.asarray(v, float) for v in box)
    lo, hi = lo - clearance - pitch, hi + clearance + pitch
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    T0 = robot.fk({})
    jt = robot.joints

    def rigid_group(name):
        out = [name]
        for j in robot.order:
            if j.parent in out and j.type == "fixed":
                out.append(j.child)
        return out

    if neighbours is None:
        neighbours = []
        pj = robot.parent_joint.get(link)
        if pj is not None:
            up = pj.parent
            while up in robot.parent_joint and robot.parent_joint[up].type == "fixed":
                up = robot.parent_joint[up].parent
            neighbours.append(up)
        neighbours += [j.child for j in robot.order if j.parent == link and j.moves]

    results = []
    for nb in neighbours:
        # the joints between the part and this neighbour
        pa, pb = robot.path_joints(link), robot.path_joints(nb)
        common = 0
        while common < min(len(pa), len(pb)) and pa[common] == pb[common]:
            common += 1
        between = [n for n in pa[common:] + pb[common:] if jt[n].moves]
        # the neighbour (and what is fixed to it) as occupancy on a grid in
        # its own frame, mm, at half the pitch
        fine = 0.5 * pitch
        pts = []
        for b in rigid_group(nb):
            if shapes and b in shapes and not hasattr(shapes[b], "faces"):
                sh = shapes[b]
                slo, shi = sh.bounds()
                g = np.stack(np.meshgrid(*[np.arange(slo[i] + fine / 2, shi[i], fine) for i in range(3)],
                                         indexing="ij"), -1).reshape(-1, 3)
                pb_mm = g[sh.contains(g)]
            else:
                # each shape on its own: ray parity over overlapping shapes
                # would read their overlap as outside
                ms = [shapes[b]] if shapes and b in shapes else robot.meshes(b)
                if not ms:
                    continue
                pb_mm = []
                for m in ms:
                    if not (shapes and b in shapes):
                        m.apply_scale(1000.0)
                    solid, org = voxelize(m, fine)
                    pb_mm.append(org + (np.argwhere(solid) + 0.5) * fine)
                pb_mm = np.concatenate(pb_mm)
            rel = np.linalg.inv(T0[nb]) @ T0[b]
            pts.append(pb_mm @ rel[:3, :3].T + rel[:3, 3] * 1000.0)
        pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        if not len(pts):
            continue
        blo = pts.min(axis=0) - fine
        bdims = np.ceil((pts.max(axis=0) + fine - blo) / fine).astype(int) + 1
        body = np.zeros(bdims, bool)
        body[tuple(np.floor((pts - blo) / fine).astype(int).T)] = True
        corners_nb = np.array([[blo[0] + i * bdims[0] * fine, blo[1] + j * bdims[1] * fine,
                                blo[2] + k * bdims[2] * fine] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        # poses of the joints between, over their ranges: steps small enough
        # that the farthest point moves under half a pitch per step
        reach = float(np.linalg.norm(corners_nb, axis=1).max())
        grids = []
        for n in between:
            lo_q, hi_q = jt[n].lower, jt[n].upper
            travel = (hi_q - lo_q) * (reach if jt[n].type in ("revolute", "continuous") else 1000.0)
            grids.append(np.linspace(lo_q, hi_q, max(steps or int(np.ceil(travel / (0.5 * pitch))) + 1, 2)))
        if np.prod([len(gq) for gq in grids]) > 20000:
            # several joints between: cap the product (the clearance covers the gaps)
            per = int(20000 ** (1 / len(grids)))
            grids = [np.linspace(gq[0], gq[-1], per) for gq in grids]
        occ = np.zeros(dims, bool)
        for vals in np.ndindex(*[len(gq) for gq in grids]):
            q = {n: grids[i][vals[i]] for i, n in enumerate(between)}
            T = robot.fk(q)
            # neighbour frame (mm) -> part frame -> world at home -> study frame
            rel = frame @ T0[link] @ np.linalg.inv(T[link]) @ T[nb] @ np.diag([1e-3, 1e-3, 1e-3, 1.0])
            c = corners_nb @ rel[:3, :3].T + rel[:3, 3]
            i0 = np.clip(np.floor((c.min(axis=0) - lo) / pitch).astype(int), 0, dims)
            i1 = np.clip(np.ceil((c.max(axis=0) - lo) / pitch).astype(int), 0, dims)
            if np.any(i1 <= i0):
                continue
            ax = [lo[d] + (np.arange(i0[d], i1[d]) + 0.5) * pitch for d in range(3)]
            cells = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
            inv = np.linalg.inv(rel)
            pn = cells @ inv[:3, :3].T + inv[:3, 3]
            bi = np.floor((pn - blo) / fine).astype(np.int64)
            ok = np.all((bi >= 0) & (bi < bdims), axis=1)
            hit = np.zeros(len(cells), bool)
            hit[ok] = body[tuple(bi[ok].T)]
            occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= hit.reshape(tuple(i1 - i0))
        if clearance > 0:
            occ = ndimage.distance_transform_edt(~occ) * pitch <= clearance
        if protect:
            c = lo + (np.argwhere(np.ones(dims, bool)) + 0.5) * pitch
            inside = np.zeros(len(c), bool)
            for sh in protect:
                inside |= sh.contains(c)
            occ &= ~inside.reshape(dims)
        vol = float(occ.sum() * pitch ** 3)
        rid = f"{prefix}-{nb}"
        entry = {"id": rid, "link": nb, "joints": between, "grid": occ, "origin": lo, "pitch": pitch,
                 "volume": vol, "mesh": None, "keepOut": None}
        if occ.any():
            pad = np.pad(occ, 1).astype(np.float32)
            v, f, _, _ = measure.marching_cubes(pad, 0.5)
            mesh = trimesh.Trimesh((v - 1.0 + 0.5) * pitch + lo, f[:, ::-1], process=True)
            entry["mesh"] = mesh
            spec = {"type": "mesh"}
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                mesh.export(os.path.join(out_dir, f"{rid}.stl"))
                spec["path"] = f"{rid}.stl"
            else:
                import base64
                spec["stlBase64"] = base64.b64encode(trimesh.exchange.stl.export_stl(mesh)).decode()
            entry["keepOut"] = {"id": rid, "bore": False, "shape": spec}
        results.append(entry)
    return results


def intrusion(entries, mesh, pitch=None) -> dict:
    """mm³ of `mesh` (study frame, a part as designed) inside each swept
    keep-out: where the part hits a neighbour somewhere in its range."""
    out = {}
    for e in entries:
        p = e["pitch"] if pitch is None else pitch
        from .verify import voxelize
        solid, origin = voxelize(mesh, p)
        c = origin + (np.argwhere(solid) + 0.5) * p
        ijk = np.floor((c - e["origin"]) / e["pitch"]).astype(np.int64)
        inb = np.all((ijk >= 0) & (ijk < e["grid"].shape), axis=1)
        hit = np.zeros(len(c), bool)
        hit[inb] = e["grid"][tuple(ijk[inb].T)]
        out[e["id"]] = round(float(hit.sum() * p ** 3), 1)
    return out


# --------------------------------------------------------------------------- whole study

def region_centroid(study_dict: dict, region: str, samples: int = 20000, seed: int = 0,
                    base_dir: str | None = None) -> np.ndarray:
    from .spec import study_from_dict

    st = study_from_dict(study_dict, base_dir=base_dir)
    shape, _ = st.region_shape(region)
    lo, hi = shape.bounds()
    p = np.random.default_rng(seed).uniform(lo, hi, size=(samples, 3))
    inside = p[shape.contains(p)]
    return inside.mean(axis=0) if len(inside) else 0.5 * (lo + hi)


def derive_study(robot: Robot, link: str, study: dict, region: str, *, payload: Payload | None = None,
                 keep_cases: bool = False, sweep_space: bool = True, k: int = 8, method: str | None = None,
                 sweep_kw: dict | None = None, space_kw: dict | None = None, out_dir: str | None = None,
                 base_dir: str | None = None, cycles: dict | None = None):
    """A study with its loads (and, with sweep_space, its neighbour keep-outs)
    derived from the mechanism. `study` supplies the interfaces: supports,
    preserve, keep-outs for hardware, material, goal and a design space
    (its bounds are where the sweep is resolved). The swept keep-outs are
    written to out_dir as STLs the returned study names by file name, so
    save the study there too (without out_dir they are inlined). Returns
    (study, report)."""
    import copy

    from .spec import study_from_dict

    s = copy.deepcopy(study)
    point = region_centroid(s, region, base_dir=base_dir)
    sweep = sweep_loads(robot, link, payload=payload, point=point, region=region, **(sweep_kw or {}))
    picked = governing_cases(sweep, s, k=k, method=method, base_dir=base_dir, cycles=cycles)
    report = {"link": link, "joint": sweep.joint, "region": region,
              "pointMm": [round(float(v), 2) for v in point], **{kk: v for kk, v in picked.items() if kk != "cases"}}
    s["loadCases"] = (s.get("loadCases", []) if keep_cases else []) + picked["cases"]
    if sweep_space:
        st = study_from_dict(s, base_dir=base_dir)
        lo, hi = st.bounds()
        protect = [r.shape for r in st.preserve] + [sp.shape for sp in st.supports if sp.shape is not None]
        protect += [st.region_shape(sp.region)[0] for sp in st.supports if sp.region]
        entries = swept_keepouts(robot, link, box=(lo, hi), protect=protect, out_dir=out_dir, **(space_kw or {}))
        s["keepOut"] = s.get("keepOut", []) + [e["keepOut"] for e in entries if e["keepOut"]]
        report["sweptKeepOuts"] = {e["id"]: {"joints": e["joints"], "mm3": round(e["volume"], 1)} for e in entries}
    return s, report
