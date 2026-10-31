"""
Co-optimizing the links of a mechanism together.

Each link of an arm carries everything past it, so its loads depend on the
masses of the links beyond it, and its own mass loads every link and joint
before it. Sizing the links one at a time against the masses written in
the URDF (the parts as they were) leaves the chain inconsistent: a lighter
wrist is still carried as the old heavy one by the forearm, the upper arm
and the J2 and J3 drives. This module closes the loop:

    arm = ArmModel(robot, hardware_from={"forearm": shipped_forearm_mesh, ...})
    arm.set_part("forearm", new_mesh)            # printed part swapped, hardware kept
    demand = joint_demand(arm.robot, payload)    # per-joint torque needed, payload capacity
    history = co_optimize(arm, order, derive, solve)

**Mass model.** A URDF body is the printed part plus the hardware fixed to
it (motors, drives, bearings, bolts). `ArmModel` subtracts the part as it
was (mesh and density, full mass properties: mass, centre of mass and
inertia by the parallel-axis theorem) to get the hardware alone, then adds
whichever part is current, so a new link changes the body's mass, its
centre of mass and its inertia together.

**Joint demand.** `joint_demand` samples poses over the joint limits (the
corners and random poses), still and moving (each joint near its speed
limit, bang-bang accelerations reached in `ramp` seconds), and for each
state the torque every actuated joint needs with and without the payload.
Torque is affine in the payload mass, so the payload each joint can carry
in each state follows exactly, and the arm's capacity is the least over
joints and states. A counterbalanced joint (a spring) is credited with the
arm's own static weight, not the payload's.

**The loop.** `co_optimize` walks the links from the tip of the chain
toward the base (Gauss-Seidel): a link's loads come from the masses beyond
it, which are already this pass's, so one pass gets most of the way and a
second pass re-derives every link's loads and regrows only those whose
loads moved by more than `tol`. The derive and solve steps are callbacks,
so the study, the sweep settings and the solver stay the caller's.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .assembly import G, Payload, Robot, _State

__all__ = ["MassProps", "mass_props", "ArmModel", "joint_demand", "load_change", "co_optimize"]


# --------------------------------------------------------------------------- mass properties

@dataclass
class MassProps:
    """kg; centre of mass (m) and inertia about it (kg m²), in one frame."""
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def __add__(self, other):
        return _combine(self, other, 1.0)

    def __sub__(self, other):
        return _combine(self, other, -1.0)


def _shift(I, m, d):
    """Inertia about a point displaced by d from the centre of mass."""
    d = np.asarray(d, float)
    return I + m * ((d @ d) * np.eye(3) - np.outer(d, d))


def _combine(a: MassProps, b: MassProps, sign: float) -> MassProps:
    m = a.mass + sign * b.mass
    if m <= 1e-9:
        return MassProps()
    com = (a.mass * a.com + sign * b.mass * b.com) / m
    I = _shift(a.inertia, a.mass, a.com - com) + sign * _shift(b.inertia, b.mass, b.com - com)
    # what is left after taking a part away can come out slightly non-physical
    # when the URDF's inertia was itself an estimate: keep it positive
    I = 0.5 * (I + I.T)
    w, v = np.linalg.eigh(I)
    I = (v * np.maximum(w, 1e-9)) @ v.T
    return MassProps(m, com, I)


def mass_props(mesh, density: float, to_frame=None) -> MassProps:
    """Mass properties of a closed mesh in mm at `density` g/cm³, in the
    frame `to_frame` maps it into (4x4, mm in -> metres out; default: the
    mesh's own frame in metres)."""
    m = mesh.copy()
    if to_frame is None:
        to_frame = np.diag([1e-3, 1e-3, 1e-3, 1.0])
    m.apply_transform(np.asarray(to_frame, float))
    vol = abs(float(m.volume))
    if vol <= 0:
        return MassProps()
    kg = vol * density * 1000.0
    # trimesh's moment_inertia is for unit density about the centre of mass
    I = np.asarray(m.moment_inertia, float) * (density * 1000.0)
    if m.volume < 0:
        I = -I
    return MassProps(kg, np.asarray(m.center_mass, float), I)


def _body_props(body) -> MassProps:
    return MassProps(float(body.mass), np.asarray(body.com, float).copy(), np.asarray(body.inertia, float).copy())


class ArmModel:
    """A robot whose printed parts can be swapped while the hardware stays.

    robot: from Robot.from_urdf, with any hardware changes already applied
    to its bodies (a motor swapped for a lighter one). parts: {link: [(mesh
    mm, density g/cm³, 4x4 mesh frame -> the arm's world at home, mm)]}, the
    printed parts the URDF's inertials include, so they can be taken out.
    A link's parts may be several meshes (a hand: palm, fingers)."""

    def __init__(self, robot: Robot, parts: dict):
        self.robot = robot
        self.T0 = robot.fk({})
        self.hardware = {}
        self.parts = {}
        for name, b in robot.bodies.items():
            hw = _body_props(b)
            for spec in parts.get(name, []):
                hw = hw - self._props(name, *spec)
            self.hardware[name] = hw
            self.parts[name] = list(parts.get(name, []))

    def _props(self, link, mesh, density, frame=None):
        frame = np.eye(4) if frame is None else np.asarray(frame, float)
        to_link = np.linalg.inv(self.T0[link]) @ np.diag([1e-3, 1e-3, 1e-3, 1.0]) @ frame
        return mass_props(mesh, density, to_link)

    def set_parts(self, link: str, parts: list):
        """Replace `link`'s printed parts: [(mesh, density, frame)]."""
        self.parts[link] = list(parts)
        p = self.hardware[link]
        for spec in parts:
            p = p + self._props(link, *spec)
        b = self.robot.bodies[link]
        b.mass, b.com, b.inertia = p.mass, p.com, p.inertia

    def set_part(self, link: str, mesh, density: float, frame=None):
        self.set_parts(link, [(mesh, density, frame)])

    def printed_kg(self, link: str | None = None) -> float:
        links = [link] if link else list(self.parts)
        return float(sum(self._props(n, *s).mass for n in links for s in self.parts[n]))

    def masses(self) -> dict:
        """{link: {kg, printedKg, hardwareKg, comM}}"""
        out = {}
        for n, b in self.robot.bodies.items():
            if b.mass <= 0 and not self.parts.get(n):
                continue
            out[n] = {"kg": round(float(b.mass), 4), "printedKg": round(self.printed_kg(n), 4),
                      "hardwareKg": round(float(self.hardware[n].mass), 4),
                      "comM": [round(float(v), 4) for v in b.com]}
        return out

    def snapshot(self) -> dict:
        return {n: _body_props(b) for n, b in self.robot.bodies.items()}

    def restore(self, snap: dict):
        for n, p in snap.items():
            b = self.robot.bodies[n]
            b.mass, b.com, b.inertia = p.mass, p.com.copy(), p.inertia.copy()


# --------------------------------------------------------------------------- joint demand

def joint_demand(robot: Robot, payload: Payload, *, joints: list | None = None, samples: int = 2000,
                 corners: bool = True, ramp: float | None = 0.5, speed: float = 1.0, balanced: tuple = (),
                 gravity=(0.0, 0.0, -G), seed: int = 0) -> dict:
    """Torque each actuated joint needs, its margin at `payload.kg`, and the
    payload the arm can carry, holding still and moving.

    For every sampled pose: held still, and moving (each joint at speed x
    its velocity limit, signs at random, accelerating at velocity / ramp,
    bang-bang). Torques are taken with the payload (at payload.at, its
    nominal mass) and without; torque is affine in payload mass, so each
    joint's capacity in each state is exact. balanced joints have their
    static gravity torque without the payload carried by a counterbalance
    (a spring tuned to the arm). Returns per joint {effort, still, moving}
    with peak torque (N·m) at the rated payload, margin (effort / peak) and
    capacityKg, plus the arm's capacity (the least)."""
    rng = np.random.default_rng(seed)
    jt = robot.joints
    names = robot.movable
    lim = {n: (jt[n].lower, jt[n].upper) for n in names}
    acts = joints or [n for n in names if jt[n].type in ("revolute", "continuous") and jt[n].effort > 0]
    g = np.asarray(gravity, float)
    reach = robot.path_joints(payload.link)
    relevant = [n for n in names if n in reach]
    poses = []
    if corners:
        levels = [(lo, 0.5 * (lo + hi), hi) if lo < hi else (lo,) for lo, hi in (lim[n] for n in relevant)]
        if int(np.prod([len(v) for v in levels])) <= 4096:
            for idx in np.ndindex(*[len(v) for v in levels]):
                poses.append({n: levels[i][idx[i]] for i, n in enumerate(relevant)})
    for _ in range(samples):
        poses.append({n: rng.uniform(*lim[n]) for n in relevant})
    pay = (1.0, np.asarray(payload.at, float), payload.link)
    effort = np.array([jt[n].effort for n in acts])
    rows = {"still": ([], []), "moving": ([], [])}
    for q in poses:
        s_arm = _State(robot, q, None, None, g)
        s_pay = _State(robot, q, None, None, g, pay)
        spring = np.array([s_arm.torque(n) if n in balanced else 0.0 for n in acts])
        t0 = np.array([s_arm.torque(n) for n in acts]) - spring
        t1 = np.array([s_pay.torque(n) for n in acts]) - spring
        rows["still"][0].append(t0)
        rows["still"][1].append(t1 - t0)
        if ramp:
            qd = {n: rng.choice([-1.0, 1.0]) * speed * jt[n].velocity for n in relevant}
            qdd = {n: rng.choice([-1.0, 1.0]) * jt[n].velocity / ramp for n in relevant}
            d_arm = _State(robot, q, qd, qdd, g)
            d_pay = _State(robot, q, qd, qdd, g, pay)
            t0 = np.array([d_arm.torque(n) for n in acts]) - spring
            t1 = np.array([d_pay.torque(n) for n in acts]) - spring
            rows["moving"][0].append(t0)
            rows["moving"][1].append(t1 - t0)
    out = {"payloadKg": payload.kg, "ramp": ramp, "samples": len(poses), "balanced": list(balanced), "joints": {}}
    cap_arm = {}
    for mode, (a, b) in rows.items():
        if not a:
            continue
        a, b = np.array(a), np.array(b)               # (states, joints): arm alone, per kg of payload
        need = np.abs(a + payload.kg * b)
        # largest p >= 0 with |a + p b| <= e in every state
        with np.errstate(divide="ignore", invalid="ignore"):
            cap = np.where(np.abs(b) > 1e-12, (effort - np.sign(b) * a) / np.abs(b), np.inf)
        cap = np.where(np.abs(a) > effort, -np.inf, cap)    # the arm alone is already over
        per = cap.min(axis=0)
        for i, n in enumerate(acts):
            k = int(np.argmax(need[:, i]))
            out["joints"].setdefault(n, {"effort": float(effort[i])})[mode] = {
                "peakNm": round(float(need[k, i]), 3), "armOnlyNm": round(float(np.abs(a[:, i]).max()), 3),
                "margin": round(float(effort[i] / need[k, i]), 3) if need[k, i] > 0 else None,
                "capacityKg": round(float(max(per[i], 0.0)), 3) if np.isfinite(per[i]) else None}
        j = int(np.argmin(per))
        cap_arm[mode] = {"capacityKg": round(float(max(per[j], 0.0)), 3), "limitedBy": acts[j]}
    out["capacity"] = cap_arm
    return out


# --------------------------------------------------------------------------- the loop

def load_change(old_cases: list, new_cases: list, region: str | None = None) -> float:
    """Relative change between two sets of governing cases: of the largest
    force and the largest moment at `region` (the larger of the two)."""
    def env(cases):
        f = m = 0.0
        for c in cases:
            for ld in c["loads"]:
                if region and ld.get("region") != region:
                    continue
                f = max(f, float(np.linalg.norm(ld.get("force", [0, 0, 0]))))
                m = max(m, float(np.linalg.norm(ld.get("moment", [0, 0, 0]))))
        return f, m
    (f0, m0), (f1, m1) = env(old_cases), env(new_cases)
    return max(abs(f1 - f0) / max(f0, 1e-9), abs(m1 - m0) / max(m0, 1e-9))


def co_optimize(arm: ArmModel, order: list, derive, solve, *, verify=None, passes: int = 3, tol: float = 0.02,
                state: dict | None = None, save=None, log=print) -> dict:
    """Regrow `order` (links, tip of the chain first) until their loads stop
    moving.

    derive(link) -> (load cases, info): the link's governing cases for the
    arm's current masses (arm.robot). solve(link, cases) -> (parts for
    ArmModel.set_parts, report dict with "g") or None when nothing meets the
    goal. verify(link, cases) -> report dict with "g" and "passes": the
    link's current part checked under the new cases. A link is regrown when
    its derived loads moved by more than `tol` (load_change) since the loads
    its part was grown or checked for, or when its part no longer passes.
    The regrown part replaces the current one only when it is lighter or the
    current one fails: an optimizer's outcome moves by a few percent from
    run to run, more than a few percent of load moves the optimum, so a
    passing part is not traded for a heavier one. state: a previous return
    value, to resume; save(state) is called after every step. Returns
    {"history": [...], "links": {link: {cases, report, pass}}, "converged"}."""
    state = state or {"history": [], "links": {}, "converged": False}
    for p in range(passes):
        if state.get("pass", -1) > p:
            continue
        state["pass"] = p
        changed = state.setdefault("changed", {}).setdefault(str(p), 0)
        for link in order:
            done = state.get("done", {}).get(str(p), [])
            if link in done:
                continue
            t = time.time()
            cases, info = derive(link)
            prev = state["links"].get(link)
            change = None if prev is None else load_change(prev["cases"], cases, info.get("region"))
            row = {"pass": p, "link": link, "loadChange": None if change is None else round(change, 4),
                   "deriveSeconds": round(time.time() - t, 1), "massesBefore": arm.masses()}
            cur = verify(link, cases) if (verify and prev is not None) else None
            if cur is not None:
                row["current"] = cur
            ok = cur is None or cur.get("passes", True)
            if prev is not None and change <= tol and ok:
                row["result"] = "loads within tolerance; kept"
            else:
                t = time.time()
                res = solve(link, cases)
                row["solveSeconds"] = round(time.time() - t, 1)
                if res is not None:
                    row["report"] = res[1]
                if res is None:
                    row["result"] = "no outcome met the goal; part kept"
                elif cur is not None and ok and cur["g"] <= res[1]["g"]:
                    # the part it has carries the new loads and is lighter
                    state["links"][link] = {**prev, "cases": cases, "report": {**prev["report"], **cur}, "info": info}
                    row["result"] = f"regrown at {res[1]['g']} g, heavier than the current part; kept"
                else:
                    arm.set_parts(link, res[0])
                    state["links"][link] = {"cases": cases, "report": res[1], "pass": p, "info": info}
                    row["result"] = "regrown"
                    changed += 1
                    state["changed"][str(p)] = changed
            row["printedKg"] = round(arm.printed_kg(link), 4)
            state["history"].append(row)
            state.setdefault("done", {}).setdefault(str(p), []).append(link)
            log(json.dumps({k: v for k, v in row.items() if k not in ("massesBefore", "report", "current")}
                           | {"g": (row.get("report") or {}).get("g"), "currentG": (cur or {}).get("g"),
                              "currentSf": (cur or {}).get("sf")}))
            if save:
                save(state)
        if changed == 0:
            state["converged"] = True
            state["pass"] = passes
            if save:
                save(state)
            break
    return state
