"""
The images in this tutorial, from a full run's outputs (not part of the
tutorial; needs Node with Playwright's Chromium, see docs/render3d.py):

    python render.py            # after the README's commands, in this folder
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from render3d import Scene, ramp  # noqa: E402

from loadpath.buildsim import refine_mesh  # noqa: E402

EXAGGERATE = 2.0
VIEW = (18, -62)
img = os.path.join(HERE, "images")


def load(path):
    return trimesh.load(os.path.join(HERE, path), force="mesh")


def report(folder):
    with open(os.path.join(HERE, folder, "report.json")) as fh:
        return json.load(fh)


nominal = load("bracket.stl")
center, span = 0.5 * np.add(*nominal.bounds), 31.0


sim, chk = report("sim"), report("check")
# the design split as buildsim splits it: the printed and compensated STLs
# have the same triangles in the same order, so each vertex has a home
fine = refine_mesh(nominal, sim["pitch"])


def off_design(m):
    """Each vertex's offset from the design (from its own vertex of the
    refined design when the triangles match, else from the nearest point of
    the design's surface), and the shape with that offset exaggerated."""
    if len(m.faces) == len(fine.faces):
        home = np.zeros_like(m.vertices)
        home[m.faces.ravel()] = fine.vertices[fine.faces.ravel()]
    else:
        home = trimesh.proximity.closest_point(nominal, m.vertices)[0]
    off = m.vertices - home
    return np.linalg.norm(off, axis=1), trimesh.Trimesh(home + EXAGGERATE * off, m.faces, process=False)


top = 1.5

# 1. the design as printed, deviation as colour
d, big = off_design(load("sim/bracket-printed.stl"))
s = Scene(f"Ti64 bracket as printed, standing: deviation from the design (max {sim['deviation']['max']:.2f} mm, "
          f"shape x{EXAGGERATE:g})", panel_size=(500, 440),
          colorbar=("deviation from design", "0", f"{top:.2f} mm"))
s.panel("the design", view=VIEW, center=center, span=span).mesh(nominal)
s.panel("as printed and cut off", view=VIEW, center=center, span=span).mesh(big, vertex_colors=ramp(d / top))
s.render(os.path.join(img, "bracket.png"))

# 2. compensation: the pre-deformed shape, and it as printed, on the same scale
comp = load("comp/bracket-compensated.stl")
dc, bigc = off_design(comp)
dp, bigp = off_design(load("check/bracket-compensated-printed.stl"))
s = Scene(f"compensation: print the drooped shape, get the design (shape x{EXAGGERATE:g})", panel_size=(420, 400),
          colorbar=("off the design", "0", f"{top:.2f} mm"))
s.panel(f"as drawn, printed: {sim['deviation']['max']:.2f} mm", view=VIEW, center=center, span=span) \
    .mesh(big, vertex_colors=ramp(d / top))
s.panel("compensated shape to print", view=VIEW, center=center, span=span).mesh(bigc, vertex_colors=ramp(dc / top))
s.panel(f"compensated, printed: {chk['deviation']['max']:.2f} mm", view=VIEW, center=center, span=span) \
    .mesh(bigp, vertex_colors=ramp(dp / top))
s.render(os.path.join(img, "compensation.png"))
