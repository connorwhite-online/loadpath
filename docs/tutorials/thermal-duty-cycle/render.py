"""
The images in this tutorial, from a full run's outputs in cold/ and hot/
(run the README's commands without --quick first). Not run by the tests;
needs Node with Playwright's Chromium (docs/render3d.py).

    python render.py [run folder]       # default: this folder
"""
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from render3d import Scene, ramp  # noqa: E402

from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402

IMAGES = os.path.join(HERE, "images")
RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
VIEW = (22, -128)        # from behind the motor face, the frame below
T_LO, T_HI = 25.0, 75.0


def best(folder):
    rep = json.load(open(os.path.join(RUN, folder, "report.json")))
    o = rep["outcomes"][rep["best"]]
    path = os.path.join(RUN, folder, "best.stl")
    return trimesh.load(path, force="mesh"), o


def on_mesh(mesh, nodes, values):
    """A nodal field on the mesh's vertices (inverse distance, 8 nearest)."""
    dist, idx = cKDTree(nodes).query(mesh.vertices, k=8)
    w = 1.0 / np.maximum(dist, 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    return (w * np.asarray(values)[idx]).sum(axis=1)


hot_study = study_from_dict(json.load(open(os.path.join(HERE, "mount.json"))), base_dir=HERE)
pacf_study = study_from_dict(json.load(open(os.path.join(HERE, "mount-pa-cf.json"))), base_dir=HERE)
motor = trimesh.load(os.path.join(HERE, "motor-body.stl"), force="mesh")
cold_mesh, cold_o = best("cold")
hot_mesh, hot_o = best("hot")
cold_hot = json.load(open(os.path.join(RUN, "cold-hot", "verify.json")))["report"]

s = Scene("NEMA 17 mount: designed cold against designed for the duty cycle", panel_size=(480, 440))
for label, m in ((f"PETG, designed cold: {cold_o['mass']:.1f} g", cold_mesh),
                 (f"PA-CF, designed for the cycle: {hot_o['mass']:.1f} g", hot_mesh)):
    p = s.panel(label, view=VIEW)
    p.mesh(m)
    p.mesh(motor, color=(0.35, 0.35, 0.4), opacity=0.18)
s.render(os.path.join(IMAGES, "mounts.png"), out_scale=1.1)

# temperature at the cycle's hottest moment, solved on each part as delivered
s = Scene("temperature at the end of the 10 minutes on", panel_size=(480, 440),
          colorbar=("temperature", f"{T_LO:.0f} C", f"{T_HI:.0f} C"))
for label, m, st, fat in (("PETG, designed cold", cold_mesh, hot_study, cold_hot["durability"]),
                          ("PA-CF, designed for the cycle", hot_mesh, pacf_study, hot_o["durability"])):
    f = verify_part(st, m, refine=0, fields=True)["fields"]
    t = on_mesh(m, f["nodes"], f["temperature"])
    s.panel(f"{label}: {t.max():.0f} C peak, fatigue SF {fat['fatigueSafetyFactor']:.2f}",
            view=VIEW).mesh(m, vertex_colors=ramp((t - T_LO) / (T_HI - T_LO)))
s.render(os.path.join(IMAGES, "temperature.png"), out_scale=1.1)
