"""
Wires across a robot-arm elbow, followed over its range of motion.

Two channel-section links joined by stub pins either side, so the joint's
axis is open in the middle. A motor lead (three 22 AWG silicone wires in a
sleeve) and an encoder cable (a thin jacketed four-core) run from a clip in
the upper link to a clip in the forearm. The question is the one every arm
has: where do the clips go, and how much cable between them, so that
nothing bends tighter than its flexing radius, comes taut, or is pinched,
at any angle from -90 to +90 degrees?

    python examples/wires/joint.py [--out joint-out] [--clip 50] [--offset 0] [--length L]
    python examples/wires/joint.py --compare        # three ways to do it, side by side

--clip: how far each clip sits from the pin's axis (mm). --offset: how far
the cable runs off the axis (0: through it). --compare runs the three
designs an arm designer would try: along the outside of the links (it
comes taut as the elbow closes), through the axis with the clips close in
(it has to fold too tightly), and through the axis with the clips set
back, which passes.
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


def write(out, length=None, offset=0.0, rng=90.0, clip=50.0):
    import manifold3d as m3d
    os.makedirs(out, exist_ok=True)

    def disc(r, y0, y1):          # a cylinder about y
        return m3d.Manifold.cylinder(y1 - y0, r, r, 64).rotate((-90, 0, 0)).translate((0, y0, 0))

    def box(lo, hi):
        return m3d.Manifold.cube(tuple(np.subtract(hi, lo))).translate(tuple(lo))
    # upper link: a channel open on top, its walls carried on to two cheeks round the pins
    upper = box((-100, -16, -13), (-22, 16, -10))
    fore = box((22, -11.5, -11), (100, 11.5, -8))
    for sg in (-1, 1):
        y0, y1 = sorted((sg * 13, sg * 16))
        upper += box((-100, y0, -13), (0, y1, 13)) + disc(13, y0, y1)
        e0, e1 = sorted((sg * 9, sg * 11.5))
        fore += box((0, e0, -11), (100, e1, 11)) + disc(11, e0, e1)
        upper += disc(3.0, *sorted((sg * 8.5, sg * 16)))            # the stub pin
    files = {"upper": _stl(upper, os.path.join(out, "upper-link.stl")),
             "fore": _stl(fore, os.path.join(out, "forearm.stl"))}
    z = float(offset)
    a = float(clip)

    def wire(wid, cab, y, extra=None):
        w = {"id": wid, "cable": cab,
             "from": {"at": [-a, y, z], "direction": "+x", "body": "upper-link"},
             "to": {"at": [a, y, z], "direction": "-x", "body": "forearm"}}
        w.update({"length": length} if length else {"slack": 0.0})
        w.update(extra or {})
        return w
    brief = {
        "name": "elbow-harness",
        "description": "a motor lead and an encoder cable across an elbow, followed from -90 to +90 degrees",
        "bodies": [{"id": "upper-link", "shape": files["upper"], "color": "#8fa3bf"},
                   {"id": "forearm", "shape": files["fore"], "moves": True, "color": "#c2a37f"}],
        "wires": [
            wire("motor", {"type": "bundle", "wires": 3, "awg": 22, "insulation": "silicone",
                           "name": "3 x 22 AWG silicone"}, 3.5, {"current": 4.0}),
            # a continuous-flex cable: its datasheet's radii (6x its diameter flexing), not the house rule
            wire("encoder", {"type": "cable", "wires": 4, "awg": 30, "insulation": "pvc", "jacket": "pur",
                             "jacketWall": 0.4, "name": "4 x 30 AWG, PUR jacket, continuous flex",
                             "minBendRadius": {"static": 9.0, "dynamic": 18.0}}, -3.5)],
        "motion": {"type": "revolute", "axis": "y", "origin": [0, 0, 0], "range": [-rng, rng], "steps": 18},
        "pitch": 0.5,
    }
    path = os.path.join(out, "elbow.wires.json")
    json.dump(brief, open(path, "w"), indent=1)
    return path


DESIGNS = {"outside": {"offset": 16.0, "clip": 35.0},
           "close": {"offset": 0.0, "clip": 14.0},
           "set-back": {"offset": 0.0, "clip": 50.0}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="joint-out")
    ap.add_argument("--length", type=float)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=50.0)
    ap.add_argument("--range", type=float, default=90.0)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if not a.compare:
        brief = write(a.out, a.length, a.offset, a.range, a.clip)
        sys.exit(subprocess.call([sys.executable, "-m", "loadpath", "wires", brief, "--out", a.out]))
    rows = []
    for name, d in DESIGNS.items():
        out = os.path.join(a.out, name)
        brief = write(out, None, d["offset"], a.range, d["clip"])
        subprocess.call([sys.executable, "-m", "loadpath", "wires", brief, "--out", out, "--quiet"])
        rep = json.load(open(os.path.join(out, "wires.json")))
        for wid, r in rep["wires"].items():
            m = r.get("motion") or {}
            rows.append((name, wid, r.get("length"), m.get("minBendRadius"), m.get("allowedBendRadius"),
                         "ok" if r["ok"] else "; ".join(r.get("problems", []) + m.get("problems", []))))
    print(f"{'design':10} {'cable':8} {'length':>7} {'bend':>7} {'allowed':>8}  verdict")
    for n, w, L, b, al, v in rows:
        print(f"{n:10} {w:8} {L or 0:7.1f} {b or 0:7.1f} {al or 0:8.1f}  {v}")
