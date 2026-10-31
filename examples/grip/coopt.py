"""
GRIP's five links co-optimized as one arm (loadpath.coopt).

The per-link arm (GRIP generative-redesign, v3/build) sized each link
against loads swept from the URDF, but the URDF's masses are the shipped
parts': the forearm, the wrist and the hand that every link carries were
the old heavy ones. Here the printed part in each URDF body is swapped for
the current one (hardware kept), the loads are swept again with those
masses, and the links are regrown from the tip of the chain toward the
base until their loads stop moving.

    python examples/grip/coopt.py --grip ../grip demand          # masses, joint torques, payload, no solves
    python examples/grip/coopt.py --grip ../grip run             # the loop (resumable: re-run to continue)
    python examples/grip/coopt.py --grip ../grip report          # per-link arm vs co-optimized arm

Mass model (what is assumed):
- Each link's URDF body = the shipped printed part (GRIP exports/stl, SLS
  PA12-GF 1.22 g/cm³, full mass properties) + hardware. The wrist joints'
  stepper drives become Feetech servos as derive.py does.
- The hand (URDF tool + jaw_a + jaw_b) loses the rack-and-pinion gripper:
  palm, frame, camera mount and pinion (tool), the jaws and their fingers
  and LM6UU bearings (the jaw bodies, dropped whole), and the two 6x150 mm
  steel guide rods (66 g). It gains the v0.3 palm, three TPU fingers and
  their nails. The gripper servo (an STS3215 in both) and the camera stay.
- The J2 counterbalance carries the arm's own static weight (derive.py:
  balanced=("j2",)).

The per-link arm is the starting point (pass 0 re-derives every link's
loads with its masses), so the comparison is against what was built.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from loadpath.assembly import Payload  # noqa: E402
from loadpath.coopt import ArmModel, MassProps, co_optimize, joint_demand  # noqa: E402

import derive as grip_derive  # noqa: E402

ORDER = ["wrist_link", "wrist_yoke", "forearm", "upper_arm", "turret"]     # tip of the chain first
PA12GF, TPU = 1.22, 1.21
CYCLES = {"dynamic": 1e5, "pose": 1e5, "push": 1e4, "stall": 1e3}           # as run.py --loads derived
OLD_TOOL_PARTS = ["gripper_palm", "gripper_frame", "gripper_camera_mount", "gripper_pinion"]
RODS_KG, RODS_AT_MM = 2 * 0.033, (0.0, 0.0, 60.0)     # two 6x150 steel guide rods, hand frame
# hand.py: finger roots (x, y) and the palm top; a finger's root block is 10 mm deep
FINGER_ROOTS, ROOT_Z, ROOT_DEPTH = [(-20.0, 30.0), (20.0, 30.0), (0.0, -30.0)], 66.0, 10.0
SDIR = os.path.join(HERE, "studies")


def _mesh(path):
    import trimesh
    return trimesh.load(path, force="mesh")


def _translate(xyz):
    t = np.eye(4)
    t[:3, 3] = xyz
    return t


def arm_model(grip, build):
    """The arm with the shipped parts' masses taken out of the URDF bodies and
    the per-link arm's parts (build/<link>.stl, palm, fingers) put in."""
    r = grip_derive.robot(grip)
    T0 = r.fk({})
    hand = T0["tool"].copy()
    hand[:3, 3] *= 1000.0                       # hand frame (J6 horn at the origin, mm) -> world mm
    for jaw in ("jaw_a", "jaw_b"):
        b = r.bodies[jaw]
        b.mass, b.inertia = 0.0, np.zeros((3, 3))
    exp = os.path.join(grip, "exports", "stl")
    shipped = {link: [(_mesh(os.path.join(exp, f"{link}.stl")), PA12GF, None)] for link in ORDER}
    shipped["tool"] = [(_mesh(os.path.join(exp, f"{p}.stl")), PA12GF, hand) for p in OLD_TOOL_PARTS]
    arm = ArmModel(r, shipped)
    rods = MassProps(RODS_KG, np.array(RODS_AT_MM) / 1000.0, np.zeros((3, 3)))
    arm.hardware["tool"] = arm.hardware["tool"] - rods
    arm.urdf_masses = arm.masses()              # as the per-link arm's loads saw them (rods and jaws aside)
    finger, nail = _mesh(os.path.join(build, "finger.stl")), _mesh(os.path.join(build, "nail.stl"))
    hand_parts = [(_mesh(os.path.join(build, "palm.stl")), PA12GF, hand)]
    for x, y in FINGER_ROOTS:
        at = hand @ _translate((x, y, ROOT_Z - ROOT_DEPTH))
        hand_parts += [(finger, TPU, at), (nail, PA12GF, at)]
    arm.set_parts("tool", hand_parts)
    for link in ORDER:
        arm.set_part(link, _mesh(os.path.join(build, f"{link}.stl")), PA12GF)
    return arm


def urdf_arm(grip):
    """The arm as derive.py swept it for the per-link design: URDF masses
    with the servo swap, the shipped parts and the old gripper."""
    return grip_derive.robot(grip)


def _study(link, cases):
    st = json.load(open(os.path.join(SDIR, f"{link}.json")))
    st["loadCases"] = copy.deepcopy(cases)
    return st


def make_derive(arm, samples):
    def derive(link):
        study = json.load(open(os.path.join(SDIR, f"{link}.json")))
        out, rep = grip_derive.derive(arm.robot, link, study, SDIR, samples=samples)
        cases = out["loadCases"]
        for c in cases:
            kind = c["name"].split("-")[2] if c["name"].startswith("sweep-") else None
            if kind:
                c.setdefault("cycles", CYCLES.get(kind, 1e5))
        info = {"region": grip_derive.LINKS[link], "envelope": rep["envelope"], "kinds": rep["kinds"],
                "envelopeRatio": rep["envelopeRatio"]}
        return cases, info
    return derive


def check(link, mesh, cases):
    """The study's confirm stage, as run.py: 150k cells, contact band one cell."""
    from loadpath.domain import choose_pitch
    from loadpath.spec import study_from_dict
    from loadpath.verify import verify_part
    st = study_from_dict(_study(link, cases), base_dir=SDIR)
    pitch = choose_pitch(st)
    rep = verify_part(st, mesh, max_cells=150_000, min_pitch=0.3 * pitch, contact_band=pitch)
    return {"g": round(rep["mass"], 1), "sf": round(rep["safetyFactor"], 2),
            "durabilitySf": (rep.get("durability") or {}).get("safetyFactor"),
            "keepOutIntrusionMm3": round(rep["keepOutIntrusion"], 3),
            "preserveMissingMm3": round(rep["preserveMissing"], 3), "bodies": rep["bodies"],
            "cases": {c["name"]: {"peakMPa": round(c["peakStress"], 1), "deflMm": round(c["deflection"], 3)}
                      for c in rep["cases"]}}


def make_solve(out):
    def solve(link, cases):
        from loadpath.spec import study_from_dict
        from loadpath.study import run_study
        key = hashlib.sha1(json.dumps(cases, sort_keys=True).encode()).hexdigest()[:10]
        stl = os.path.join(out, f"{link}-{key}.stl")
        json.dump(cases, open(os.path.join(out, f"{link}-{key}.cases.json"), "w"), indent=1)
        if not os.path.exists(stl):          # a solve cut by a restart is picked up again
            r = run_study(study_from_dict(_study(link, cases), base_dir=SDIR))
            if r.best is None:
                return None
            r.outcomes[r.best].mesh.export(stl)
        mesh = _mesh(stl)
        rep = check(link, mesh, cases)
        rep["stl"] = os.path.basename(stl)
        return [(mesh, PA12GF, None)], rep
    return solve


def make_verify(arm):
    def verify(link, cases):
        goal = json.load(open(os.path.join(SDIR, f"{link}.json")))["goal"]["safetyFactor"]
        rep = check(link, arm.parts[link][0][0], cases)
        rep["passes"] = rep["sf"] >= goal and rep["bodies"] == 1
        return rep
    return verify


def _restore(arm, state, out):
    """Put the parts a saved run grew back into the arm."""
    for link, rec in state.get("links", {}).items():
        arm.set_part(link, _mesh(os.path.join(out, rec["report"]["stl"])), PA12GF)


def seed(build):
    """The per-link arm as the loop's starting state: each link grown for the
    loads derive.py swept with the URDF's masses (derived/<link>.derived.json),
    so a link is regrown only when the arm's real masses move its loads."""
    links = {}
    for link in ORDER:
        cases = json.load(open(os.path.join(HERE, "derived", f"{link}.derived.json")))["loadCases"]
        for c in cases:
            c.setdefault("cycles", CYCLES.get(c["name"].split("-")[2], 1e5))
        links[link] = {"cases": cases, "pass": -1, "report": {"stl": os.path.abspath(os.path.join(build, f"{link}.stl"))}}
    return {"history": [], "links": links, "converged": False}


def demand(robot, kg=1.0):
    pay = Payload(kg, "tool", (0.0, 0.0, 0.07))
    return joint_demand(robot, pay, balanced=("j2",), samples=2000)


def totals(arm):
    m = arm.masses()
    printed = {n: round(v["printedKg"] * 1000, 1) for n, v in m.items() if v["printedKg"] > 0}
    return {"armKg": round(sum(v["kg"] for n, v in m.items() if n != "base_link"), 3),
            "printedG": printed, "printedLinksG": round(sum(printed[l] for l in ORDER), 1),
            "massesKg": {n: v["kg"] for n, v in m.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["demand", "run", "report"])
    ap.add_argument("--grip", required=True, help="a GRIP checkout (branch generative-redesign)")
    ap.add_argument("--build", default=None, help="the per-link arm's printable set (default GRIP v3/build)")
    ap.add_argument("--out", default="grip-coopt")
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args(argv)
    os.environ.setdefault("LOADPATH_WORKERS", "1")
    build = args.build or os.path.join(args.grip, "generative", "v3", "build")
    os.makedirs(args.out, exist_ok=True)
    arm = arm_model(args.grip, build)
    path = os.path.join(args.out, "coopt.json")
    if args.cmd == "demand":
        rep = {"urdf": {"masses": arm.urdf_masses, "demand": demand(urdf_arm(args.grip))},
               "perLink": {**totals(arm), "demand": demand(arm.robot)}}
        print(json.dumps(rep, indent=1))
        json.dump(rep, open(os.path.join(args.out, "demand.json"), "w"), indent=1)
        return 0
    state = json.load(open(path)) if os.path.exists(path) else seed(build)
    _restore(arm, state, args.out)
    if args.cmd == "run":
        def save(s):
            s["totals"] = totals(arm)
            tmp = path + ".tmp"
            json.dump(s, open(tmp, "w"), indent=1)
            os.replace(tmp, path)
        state = co_optimize(arm, ORDER, make_derive(arm, args.samples), make_solve(args.out),
                            verify=make_verify(arm), passes=args.passes, tol=args.tol, state=state, save=save)
        print(json.dumps(state["totals"], indent=1))
        return 0
    # report: both arms, each link verified under the loads its own arm puts on it
    per_link = arm_model(args.grip, build)
    rep = {}
    for name, a in (("perLink", per_link), ("coOptimized", arm)):
        d = make_derive(a, args.samples)
        links = {}
        for link in ORDER:
            cases, info = d(link)
            mesh = (_mesh(os.path.join(build, f"{link}.stl")) if name == "perLink"
                    else _mesh(os.path.join(args.out, state["links"][link]["report"]["stl"])))
            links[link] = {**check(link, mesh, cases), "envelope": info["envelope"]}
            print(name, link, links[link]["g"], links[link]["sf"], flush=True)
        rep[name] = {**totals(a), "links": links, "demand": demand(a.robot)}
    rep["urdf"] = {"masses": per_link.urdf_masses, "demand": demand(urdf_arm(args.grip))}
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("armKg", "printedLinksG")} for k, v in rep.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
