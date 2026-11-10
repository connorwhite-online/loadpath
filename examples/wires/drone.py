"""
A small quadcopter's electronics: the board stack, then its leads.

1. The stack (`loadpath stack`): a flight controller and a four-in-one
   speed controller on four M2 standoffs (20 x 20 mm holes), in the 22 mm
   under the canopy. The speed controller gives off 7.5 W in a hard climb;
   the stack sits in the props' wash. Which board goes on the bottom, and
   how far apart? The search solves the air through each gap, every
   board's temperature, the centre of mass, and lays the signal harness
   between the boards. Its answer goes to the frame as one carried mass
   (`inertiaRelief.masses`) and a keep-out box.

2. The leads (`loadpath wires`), in the stack that came out: a motor's
   three 22 AWG wires, 60 mm long, from the speed controller's pads along
   an arm to the motor, held by one clip; and the battery lead, two
   12 AWG silicone wires from the pads to an XT30. In a crash the battery
   slides 6 mm on its strap: the lead is followed through that, and the
   pull it puts on the solder pads is the strain-relief load.

    python examples/wires/drone.py [--out drone-out] [--speed 8] [--strap]
                                   [--lead 48] [--pack 36]

--speed: m/s of prop wash over the stack. --strap: tie the battery lead
to the frame behind the stack (a clip), so the tie takes the tug instead
of the pads. --lead, --pack: the battery lead's length and the height of
the pack's connector. The default (a pack carried above the stack, a
48 mm lead) passes; `--pack 22 --lead 38`, the pack in line behind the
stack on a short lead, comes taut when the pack slides.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


def _stl(man, path):
    import trimesh
    m = man.to_mesh()
    trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)).export(path)
    return os.path.basename(path)


def stack_brief(speed=8.0):
    return {
        "name": "quad-stack",
        "boards": [
            {"id": "fc", "size": [27, 27, 1.6], "mass": 0.0055, "watts": 0.8, "maxTemp": 85, "above": 3.0,
             "below": 1.2, "ports": [{"id": "esc", "at": [0, -12.5, 0.0], "direction": "-y"}]},
            {"id": "esc", "size": [27, 27, 1.6], "mass": 0.0075, "watts": 7.5, "maxTemp": 105, "above": 2.5,
             "below": 1.5, "ports": [{"id": "sig", "at": [0, -12.5, 1.6], "direction": "-y"}]}],
        "space": {"height": 22.0, "base": 0.0},
        "standoffs": {"pattern": 20, "diameter": 4.0, "material": "nylon"},
        "airflow": {"speed": speed, "direction": "+x"}, "ambient": 30,
        "goal": {"centerOfMass": "low"},
        "mount": "stack-posts",
        # the eight-wire signal harness between the boards: 28 AWG silicone, as one loose bundle
        "wires": [{"id": "signal", "from": "esc.sig", "to": "fc.esc",
                   "cable": {"type": "bundle", "wires": 8, "awg": 28, "insulation": "silicone",
                             "name": "8 x 28 AWG silicone"}}],
    }


def wires_brief(out, st, strap=False, lead_length=48.0, pack_z=36.0):
    import manifold3d as m3d

    def box(lo, hi):
        return m3d.Manifold.cube(tuple(np.subtract(hi, lo))).translate(tuple(lo))

    def cyl(r, z0, z1, x=0.0, y=0.0):
        return m3d.Manifold.cylinder(z1 - z0, r, r, 48).translate((x, y, z0))
    # the frame: a plate under the stack and one arm out to a motor (the others are the same)
    frame = box((-20, -20, -2.5), (20, 20, 0)) + box((16, -4.5, -5), (64, 4.5, 0)) + cyl(11, -5, 0, 72)
    files = {"frame": _stl(frame, os.path.join(out, "frame.stl")),
             "motor": _stl(cyl(9, 0, 13, 72), os.path.join(out, "motor.stl")),
             "battery": _stl(box((-96, -17, pack_z - 10), (-50, 17, pack_z + 10)), os.path.join(out, "battery.stl"))}
    bodies = [{"id": "frame", "shape": files["frame"], "color": "#8fa3bf"},
              {"id": "motor", "shape": files["motor"], "color": "#9aa0a6"},
              {"id": "battery", "shape": files["battery"], "moves": True, "color": "#c2a37f"}]
    for bid, z in st["z"].items():
        files[bid] = _stl(box((-13.5, -13.5, z), (13.5, 13.5, z + 1.6)), os.path.join(out, f"{bid}.stl"))
        bodies.append({"id": bid, "shape": files[bid], "color": "#5d8a66"})
    ze = st["z"]["esc"] + 1.6
    lead = {"id": "battery-lead", "current": 25.0,
            "cable": {"type": "bundle", "wires": 2, "awg": 12, "insulation": "silicone", "name": "2 x 12 AWG silicone"},
            # the pads take a peel, not much: what the lead may ask of them
            "from": {"at": [-15.0, 0.0, ze + 3.8], "direction": "-x", "maxPull": 15.0, "maxMoment": 150.0},
            "to": {"at": [-50.0, 0.0, pack_z], "direction": "+x", "body": "battery"},
            "length": lead_length}
    if strap:
        lead["via"] = [{"at": [-28.0, 0.0, ze + 5.0], "along": 16.0}]
    brief = {
        "name": "quad-leads",
        "description": "a motor's leads along an arm, and the battery lead through a 6 mm slide of the pack",
        "bodies": bodies,
        "wires": [
            {"id": "motor-leads", "current": 12.0, "length": 60.0,
             "cable": {"type": "bundle", "wires": 3, "awg": 22, "insulation": "silicone",
                       "name": "3 x 22 AWG silicone"},
             "from": {"at": [13.0, 0.0, ze + 1.5], "direction": "+x", "maxPull": 10.0},
             "to": {"at": [63.0, 0.0, 2.5], "direction": "-x"},
             "via": [{"at": [40.0, 0.0, 2.2]}]},
            lead],
        # the pack slides back on its strap in a crash
        "motion": {"type": "prismatic", "axis": "-x", "range": [0.0, 6.0], "steps": 6, "flexing": False},
        "pitch": 0.5,
    }
    path = os.path.join(out, "quad.wires.json")
    json.dump(brief, open(path, "w"), indent=1)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="drone-out")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--strap", action="store_true")
    ap.add_argument("--lead", type=float, default=48.0, help="battery lead length, mm")
    ap.add_argument("--pack", type=float, default=36.0, help="height of the pack's connector over the frame plate, mm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sb = os.path.join(a.out, "quad.stack.json")
    json.dump(stack_brief(a.speed), open(sb, "w"), indent=1)
    subprocess.call([sys.executable, "-m", "loadpath", "stack", sb, "--out", os.path.join(a.out, "stack")])
    res = json.load(open(os.path.join(a.out, "stack", "stack.json")))
    if not res.get("best"):
        sys.exit(1)
    wb = wires_brief(a.out, res["best"], a.strap, a.lead, a.pack)
    sys.exit(subprocess.call([sys.executable, "-m", "loadpath", "wires", wb, "--out", os.path.join(a.out, "wires")]))
