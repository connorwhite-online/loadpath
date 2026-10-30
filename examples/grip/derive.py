"""
GRIP's link loads from the arm itself (loadpath.assembly), against the
hand-written ones in studies/ (one pose at full reach x1.5, the joint above
at its rating, a 20 N push along its axis).

For each link: sweep GRIP's URDF (poses, speeds and accelerations the
drives can deliver, a 0-1 kg payload scattered 30 mm round the tool point,
a push at the tool, each joint upstream stalled against an obstacle),
collapse the sweep to a few governing cases on the solid design space, and
verify GRIP's committed v0.3 part under both sets of cases.

    python examples/grip/derive.py --grip ../grip                # all five links
    python examples/grip/derive.py --grip ../grip --links forearm --no-verify
    python examples/grip/derive.py --grip ../grip --space    # also the neighbours' swept keep-outs

Writes OUT/<link>.derived.json (the study with derived loads) and
OUT/derive.json (the sweep's envelope, the cases picked, and the SF of the
committed part under old and new loads). With --space, the parent and child
links (GRIP's committed v0.3 parts where there is one, else the URDF mesh)
swept through their joints' ranges become keep-outs in the derived study
(OUT/sweep-<link>-<neighbour>.stl), and the report says how much of the
committed part is inside them.

GRIP v0.3 masses and efforts are applied as generative/v3.py does: the wrist
joints are Feetech servos (STS3215 on the rolls, 2.9 N·m; STS3250 on the
pitch, 4.9 N·m), J2 is the 13 N·m NEMA 17 drive with its counterbalance
spring carrying the arm's static weight. The spring's own forces on the
turret and upper arm are not modelled here: those two links keep their
spring loads from the hand-written reach case.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from loadpath.assembly import (  # noqa: E402
    Payload, Robot, governing_cases, intrusion, region_centroid, swept_keepouts, sweep_loads,
)

LINKS = {"turret": "j2-sleeve", "upper_arm": "j3-sleeve", "forearm": "j4-mount",
         "wrist_yoke": "j5-mount", "wrist_link": "j6-mount"}
SERVO_KG = {"forearm": 0.055, "wrist_yoke": 0.075, "wrist_link": 0.055}   # v3.py: STS3215 / STS3250
STEPPER_JOINT_KG = 0.37
EFFORT = {"j2": 13.0, "j4": 2.9, "j5": 4.9, "j6": 2.9}
PAYLOAD = Payload(1.0, "tool", (0.0, 0.0, 0.07), radius=0.03, min_kg=0.0)


def robot(grip):
    r = Robot.from_urdf(os.path.join(grip, "urdf", "grip.urdf"))
    for link, kg in SERVO_KG.items():
        r.bodies[link].mass += kg - STEPPER_JOINT_KG
    for j, e in EFFORT.items():
        r.joints[j].effort = e
    return r


def derive(r, link, study, base_dir, k=8, samples=1500, cycles=None):
    region = LINKS[link]
    t = time.time()
    sw = sweep_loads(r, link, payload=PAYLOAD, point=region_centroid(study, region, base_dir=base_dir), region=region,
                     samples=samples, balanced=("j2",))
    t_sweep = time.time() - t
    picked = governing_cases(sw, study, k=k, base_dir=base_dir, cycles=cycles)
    env = {}
    for kind in sorted({kd.split(":")[0] for kd in sw.kinds}):
        m = np.array([kd.split(":")[0] == kind for kd in sw.kinds])
        env[kind] = {"n": int(m.sum()), "maxForceN": round(float(np.linalg.norm(sw.wrenches[m, :3], axis=1).max()), 1),
                     "maxMomentNmm": round(float(np.linalg.norm(sw.wrenches[m, 3:], axis=1).max()), 0)}
    out = copy.deepcopy(study)
    out.pop("chain", None)
    old = study["loadCases"]
    # hand-written loads other than at the main region (the J2 spring) ride along
    extra = [ld for c in old for ld in c["loads"] if ld["region"] != region]
    cases = picked["cases"]
    if extra:
        for c in cases:
            c["loads"] += copy.deepcopy(extra)
    out["loadCases"] = cases
    out["description"] = study["description"] + " Loads derived from the URDF by loadpath.assembly (examples/grip/derive.py)."
    old_m = max(np.linalg.norm(ld.get("moment", [0, 0, 0])) for c in old for ld in c["loads"] if ld["region"] == region)
    old_f = max(np.linalg.norm(ld.get("force", [0, 0, 0])) for c in old for ld in c["loads"] if ld["region"] == region)
    rep = {"region": region, "samples": len(sw.wrenches), "sweepSeconds": round(t_sweep, 1),
           "envelope": env, "handWritten": {"maxForceN": round(float(old_f), 1), "maxMomentNmm": round(float(old_m), 0)},
           **{kk: v for kk, v in picked.items() if kk not in ("cases", "states")},
           "cases": [{"name": c["name"], "force": c["loads"][0]["force"], "moment": c["loads"][0]["moment"],
                      "state": s} for c, s in zip(cases, picked["states"])]}
    return out, rep


def space(r, grip, link, study, base_dir, out_dir):
    """Swept keep-outs of the neighbours, from the committed parts."""
    import trimesh

    from loadpath.spec import study_from_dict

    st = study_from_dict(study, base_dir=base_dir)
    T0 = r.fk({})
    shapes = {}
    for nb in r.bodies:
        path = os.path.join(grip, "generative", "v3", "out", f"{nb}.stl")
        if nb != link and os.path.exists(path):
            m = trimesh.load(path, force="mesh")
            m.apply_translation(-T0[nb][:3, 3] * 1000.0)     # study frame (home, mm) -> the link's frame
            shapes[nb] = m
    entries = swept_keepouts(r, link, box=st.bounds(), pitch=1.0, clearance=1.0, shapes=shapes,
                             protect=[p.shape for p in st.preserve], out_dir=out_dir, prefix=f"sweep-{link}")
    part = trimesh.load(os.path.join(grip, "generative", "v3", "out", f"{link}.stl"), force="mesh")
    hits = intrusion(entries, part)
    return entries, {e["id"]: {"joints": e["joints"], "mm3": round(e["volume"]),
                               "committedPartInsideMm3": hits[e["id"]],
                               "from": "committed part" if e["link"] in shapes else "URDF mesh"} for e in entries}


def verify(study_old, study_new, mesh, base_dir):
    from loadpath.domain import choose_pitch
    from loadpath.spec import study_from_dict
    from loadpath.verify import verify_part

    both = copy.deepcopy(study_old)
    both["loadCases"] = ([dict(c, name="old:" + c["name"]) for c in study_old["loadCases"]]
                         + [dict(c, name="new:" + c["name"]) for c in study_new["loadCases"]])
    st = study_from_dict(both, base_dir=base_dir)
    pitch = choose_pitch(st)
    rep = verify_part(st, mesh, max_cells=150_000, min_pitch=0.3 * pitch, contact_band=pitch)
    sf = {}
    for c in rep["cases"]:
        tag = c["name"].split(":")[0]
        s = c.get("safetyFactor")
        if s is not None:
            sf.setdefault(tag, []).append((s, c["name"]))
    return {"massG": round(rep["mass"], 1),
            "sfOld": round(min(sf["old"])[0], 2), "governingOld": min(sf["old"])[1],
            "sfNew": round(min(sf["new"])[0], 2), "governingNew": min(sf["new"])[1],
            "cases": {c["name"]: {"sf": round(c["safetyFactor"], 2) if c.get("safetyFactor") else None,
                                  "deflectionMm": round(c["deflection"], 3)} for c in rep["cases"]}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", required=True, help="a GRIP checkout (branch generative-redesign)")
    ap.add_argument("--links", default=",".join(LINKS))
    ap.add_argument("--out", default="grip-derive")
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--space", action="store_true", help="add the neighbours' swept keep-outs")
    ap.add_argument("--cycles", default="", help="life cycles by sample kind for fatigue, e.g. "
                    "dynamic=1e6,pose=1e6,push=1e4,stall=1e3")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    r = robot(args.grip)
    path = os.path.join(args.out, "derive.json")
    report = json.load(open(path)) if os.path.exists(path) else {}
    for link in args.links.split(","):
        sdir = os.path.join(HERE, "studies")
        study = json.load(open(os.path.join(sdir, f"{link}.json")))
        cycles = {k: float(v) for k, v in (kv.split('=') for kv in args.cycles.split(',') if kv)}
        new, rep = derive(r, link, study, sdir, samples=args.samples, cycles=cycles)
        if args.space:
            entries, rep["sweptKeepOuts"] = space(r, args.grip, link, study, sdir, args.out)
            new["keepOut"] = new["keepOut"] + [e["keepOut"] for e in entries if e["keepOut"]]
        with open(os.path.join(args.out, f"{link}.derived.json"), "w") as fh:
            json.dump(new, fh, indent=1)
        if not args.no_verify:
            import trimesh
            mesh = trimesh.load(os.path.join(args.grip, "generative", "v3", "out", f"{link}.stl"), force="mesh")
            rep["committedPart"] = verify(study, new, mesh, sdir)
        report[link] = rep
        print(link, json.dumps({kk: rep[kk] for kk in ("envelope", "handWritten", "kinds", "envelopeRatio")}),
              json.dumps(rep.get("committedPart", {}).get("sfOld")), json.dumps(rep.get("committedPart", {}).get("sfNew")),
              flush=True)
        with open(path, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
