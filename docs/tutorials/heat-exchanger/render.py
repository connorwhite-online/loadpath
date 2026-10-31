"""
The images in this tutorial, from a full run's outputs in hx/ (run the
README's commands without --quick first). Not run by the tests.

    python render.py

Needs Node and Playwright's Chromium (docs/render3d.py).
"""
import json
import os
import shutil
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))                 # docs/, for render3d

from render3d import FLUIDS, Scene  # noqa: E402

OUT = os.path.join(HERE, "hx")
IMAGES = os.path.join(HERE, "images")
VIEW = (22, -60)


def main():
    rep = json.load(open(os.path.join(OUT, "report.json")))
    name = rep["name"]
    part = trimesh.load(os.path.join(OUT, f"{name}.stl"), force="mesh")
    spaces = {fl: trimesh.load(p if os.path.isabs(p) else os.path.join(HERE, p), force="mesh")
              for fl, p in rep["fluidSpaces"].items()}
    names = list(spaces)
    # cut through the axis, keeping the far half (the normal points away from the camera)
    c0 = np.array([0.0, 0.0, 35.0])
    cn = np.array([-np.cos(np.radians(VIEW[1])), -np.sin(np.radians(VIEW[1])), 0.0])
    r = rep["report"]
    title = (f"{name}: gyroid cell {r['cell']:g} mm, wall {r['wall']:g} mm, AlSi10Mg, "
             + ("fluids isolated" if r["isolated"] else "FLUIDS CONNECT"))
    s = Scene(title, panel_size=(440, 520))
    s.panel("part", view=VIEW).mesh(part, metalness=0.35, roughness=0.45)
    s.panel("cut open", view=VIEW).mesh(part.slice_plane(c0, cn, cap=True), metalness=0.35, roughness=0.45)
    p = s.panel(" + ".join(names) + " (the two fluid spaces), cut open", view=VIEW)
    for i, n in enumerate(names):
        p.mesh(spaces[n].slice_plane(c0, cn, cap=True), color=FLUIDS[i % 2])
    os.makedirs(IMAGES, exist_ok=True)
    s.render(os.path.join(IMAGES, f"{name}.png"), out_scale=1.2)

    s = Scene(f"{name}: each fluid's space on its own", panel_size=(440, 520))
    for i, n in enumerate(names):
        s.panel(n, view=VIEW).mesh(spaces[n], color=FLUIDS[i % 2])
    s.render(os.path.join(IMAGES, f"{name}-fluids.png"), out_scale=1.05)

    final = os.path.join(HERE, "hx-final", f"{name}-simulation.png")
    if os.path.exists(final):
        shutil.copy(final, os.path.join(IMAGES, f"{name}-final-simulation.png"))
    sim = rep.get("simulation", {}).get("render")
    if sim:
        shutil.copy(sim if os.path.isabs(sim) else os.path.join(HERE, sim),
                    os.path.join(IMAGES, f"{name}-simulation.png"))


if __name__ == "__main__":
    main()
