"""
Render the sensor puck (after `sensor_puck.py --run`): the designer's
exterior, the parts as loadpath placed them, and the delivered part with its
top cut away around them.

    python examples/enclosure/render.py out/enclosure docs/images/enclosure/sensor-puck.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "docs"))

COLORS = {"board": (0.20, 0.55, 0.35), "usbc": (0.75, 0.75, 0.80), "battery": (0.20, 0.45, 0.85),
          "motor": (0.95, 0.55, 0.15), "door": (0.55, 0.35, 0.75)}


def main(argv=None):
    import trimesh
    from render3d import Scene

    argv = argv or sys.argv[1:]
    d, out = argv[0], argv[1]
    cut_z = float(argv[2]) if len(argv) > 2 else 3.0
    checks = json.load(open(os.path.join(d, "checks.json")))
    part = trimesh.load(os.path.join(d, "design", os.path.basename(checks["part"])), force="mesh")
    ext = trimesh.load(os.path.join(d, "cad", "exterior.stl"), force="mesh")
    parts = {k: trimesh.load(os.path.join(d, "cad", f"{k}.stl"), force="mesh") for k in COLORS
             if os.path.exists(os.path.join(d, "cad", f"{k}.stl"))}
    from loadpath.spec import study_from_dict
    sp = [f for f in os.listdir(d) if f.endswith(".study.json")][0]
    study = study_from_dict(json.load(open(os.path.join(d, sp))), base_dir=os.path.abspath(d))
    mass = abs(part.volume) * study.material.density / 1000.0
    sf = checks["verify"]["safetyFactor"]
    s = Scene(f"sensor puck: {mass:.1f} g {study.material.name.upper()}, exterior untouched, safety factor {sf:.1f} in 7 drop/squeeze cases")
    view = (28, -58)
    p = s.panel("the designer's exterior", view=view)
    p.mesh(ext)
    p = s.panel("parts placed and assembled by loadpath", view=view)
    p.mesh(ext, opacity=0.12)
    for k, m in parts.items():
        p.mesh(m, color=COLORS[k])
    p = s.panel(f"delivered interior (cut at z = {cut_z:g} mm)", view=(48, -58))
    low = part.slice_plane([0, 0, cut_z], [0, 0, -1], cap=True)
    p.mesh(low)
    for k, m in parts.items():
        if k != "door" and m.bounds[0][2] < cut_z:
            p.mesh(m.slice_plane([0, 0, cut_z], [0, 0, -1], cap=True), color=COLORS[k])
    s.render(out)
    print(out)
    return np.round(mass, 1)


if __name__ == "__main__":
    main()
