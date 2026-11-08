"""
A handheld sensor puck: the designer's exterior in, everything inside
designed (loadpath enclosure).

The exterior is a soft pebble, 84 x 56 x 24 mm, with a USB-C port at one
end and a battery door underneath (both openings the designer drew). What
goes inside is a list, not a layout: a 45 x 30 x 6 mm LiPo, a 40 x 28 mm
board with its USB-C receptacle, a coin haptic motor, and the door. The
agent places them (the receptacle out through its port, the battery on the
floor, the motor up under the top by the board), finds how each goes in, puts a cradle under
each, and designs the inside for a 1.5 m drop on any face and a hard
squeeze, in PA12 (SLS). The exterior is never changed.

    python examples/enclosure/sensor_puck.py [--out puck-out] [--run] [--quick]

Writes the exterior and component STLs and the brief to --out, then runs
`loadpath enclosure` (placement and study; --run designs and checks it).
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


def rounded_box(size, r, seg=48):
    import manifold3d as m3d
    sx, sy, sz = [s / 2 - r for s in size]
    return m3d.Manifold.batch_hull([m3d.Manifold.sphere(r, seg).translate((x, y, z))
                                    for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])


def write(out):
    import manifold3d as m3d
    os.makedirs(out, exist_ok=True)
    L, W, H = 84.0, 56.0, 24.0
    ext = rounded_box((L, W, H), 9.0)
    # the USB-C port: a slot through the -y end
    # (each opening is drawn just through the 1.2 mm skin: a deeper cut would
    # be a pocket in the exterior, with walls of its own)
    port = m3d.Manifold.cube((12.0, 5.4, 5.0)).translate((-6.0, -W / 2 - 4.0, -1.0))
    door = m3d.Manifold.cube((52.0, 32.0, 3.4)).translate((-26.0, -16.0, -H / 2 - 2.0))
    files = {
        "exterior": _stl(ext - port - door, os.path.join(out, "exterior.stl")),
        "port": _stl(port, os.path.join(out, "port.stl")),
        "door": _stl(door, os.path.join(out, "door-opening.stl")),
        # the door itself: the designer's, fills the opening's inner 1.6 mm
        "lid": _stl(m3d.Manifold.cube((51.6, 31.6, 1.4)).translate((-25.8, -15.8, -H / 2 + 0.0)),
                    os.path.join(out, "door.stl")),
    }
    # the board and its receptacle in the board's own frame (as its CAD would be)
    board = m3d.Manifold.cube((40.0, 28.0, 1.6)).translate((-20.0, -14.0, -0.8)) + \
        m3d.Manifold.cube((14.0, 10.0, 2.5)).translate((-2.0, -4.0, 0.8))      # the tallest parts on top
    usbc = m3d.Manifold.cube((9.0, 7.5, 3.2)).translate((-4.5, -14.0 - 2.6, 0.8))   # overhangs the edge
    files["board"] = _stl(board, os.path.join(out, "board.stl"))
    files["usbc"] = _stl(usbc, os.path.join(out, "usbc.stl"))
    brief = {
        "name": "sensor-puck",
        "description": "a handheld sensor puck: battery, board, USB-C, haptic motor inside a designer's pebble",
        "exterior": files["exterior"], "skin": 1.2,
        "openings": [{"id": "usb-port", "shape": files["port"], "inward": "+y"},
                     {"id": "battery-door", "shape": files["door"], "inward": "+z"}],
        "components": [
            {"id": "board", "shape": files["board"], "mass": 0.008, "watts": 0.3, "rotate": "z",
             "margin": 0.3},
            {"id": "usbc", "shape": files["usbc"], "installedOn": "board", "mass": 0.001,
             "alignTo": {"opening": "usb-port", "face": "-y", "recess": 0.3}},
            {"id": "battery", "size": [45.0, 30.0, 6.0], "mass": 0.032, "against": "-z", "rotate": "z"},
            {"id": "motor", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 3.2, "r": 5.0},
             "mass": 0.002, "near": ["board"], "against": "+z", "rotate": "none"},
            {"id": "door", "shape": files["lid"], "place": "given", "external": True, "mount": False}],
        "assembly": ["motor", "board", "battery", "door"],
        "rest": "+z",
        "loads": {"drop": {"g": 300, "faces": ["-z", "+z", "-x", "+x", "-y", "+y"]},
                  "squeeze": {"axis": "y", "force": 150, "radius": 10}},
        "material": "pa12", "process": "sls", "minMember": 1.2,
        "goal": {"safetyFactor": 1.5},
        # cells of 0.6 mm: two across the 1.2 mm skin (`run` warns under that)
        "resolution": {"pitch": 0.6},
    }
    path = os.path.join(out, "sensor-puck.brief.json")
    json.dump(brief, open(path, "w"), indent=1)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="puck-out")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    brief = write(a.out)
    cmd = [sys.executable, "-m", "loadpath", "enclosure", brief, "--out", os.path.join(a.out, "enclosure")] + \
        (["--run"] if a.run else []) + (["--quick"] if a.quick else [])
    sys.exit(subprocess.call(cmd))
