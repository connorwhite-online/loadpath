"""
The images in this tutorial, from a full run's outputs in start/ and graded/
(run the README's commands without --quick first). Not run by the tests.

    python render.py

Needs Node and Playwright's Chromium (docs/render3d.py) for the parts;
the crush curves are matplotlib.
"""
import copy
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", ".."))                 # docs/, for render3d
sys.path.insert(0, ROOT)

from render3d import Scene, ramp  # noqa: E402

from loadpath.fields import FieldGraph  # noqa: E402

IMAGES = os.path.join(HERE, "images")
VIEW = (20, -60)
PITCH = 0.25


def _cut(mesh):
    """The far half, cut through the axis (the normal points away from the camera)."""
    n = np.array([-np.cos(np.radians(VIEW[1])), -np.sin(np.radians(VIEW[1])), 0.0])
    return mesh.slice_plane([0, 0, 0], n, cap=True)


def _by_height(mesh, lo, hi):
    """Vertex colours from the density table, by height: blue light, red dense."""
    return ramp((np.interp(mesh.vertices[:, 2], *lo) - hi[0]) / (hi[1] - hi[0]))


def parts(start, graded_doc):
    imp = start["impact"]
    table = graded_doc["nodes"]["density"]
    graded = trimesh.load(os.path.join(HERE, "graded", "bumper-graded.stl"), force="mesh")
    udoc = copy.deepcopy(json.load(open(os.path.join(HERE, "bumper.json"))))
    rho_u = imp["uniformSameMass"]["density"]
    udoc["nodes"]["density"] = {"op": "constant", "value": rho_u}
    uniform = FieldGraph(udoc, base_dir=HERE).mesh(pitch=PITCH)
    lo, hi = min(table["to"] + [rho_u]), max(table["to"] + [rho_u])
    tab = (table["from"], table["to"])
    s = Scene("bumper: graded gyroid against a uniform core of the same mass", panel_size=(420, 400),
              colorbar=("density", f"{lo:.2f}", f"{hi:.2f}"))
    g_col = _by_height(graded, tab, (lo, hi))
    s.panel("graded", view=VIEW).mesh(graded, vertex_colors=g_col)
    cg = _cut(graded)
    s.panel("graded, cut open", view=VIEW).mesh(cg, vertex_colors=_by_height(cg, tab, (lo, hi)))
    cu = _cut(uniform)
    s.panel(f"uniform (density {rho_u:.3f}), cut open", view=VIEW).mesh(
        cu, vertex_colors=np.tile(ramp([(rho_u - lo) / (hi - lo)]), (len(cu.vertices), 1)))
    os.makedirs(IMAGES, exist_ok=True)
    s.render(os.path.join(IMAGES, "bumper.png"), out_scale=1.05)


def curves(start, graded):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp, gi = start["impact"], graded["impact"]
    lim = imp["forceLimit"] / 1000
    e = imp["energy"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    u = imp["uniformSameMass"]
    for label, c, stop, col in ((f"graded ({gi['coreMass']:.2f} g)", gi["curve"], gi["response"]["crush"], "#d0473a"),
                                (f"uniform, same mass (density {u['density']:.3f})", u["curve"],
                                 u["response"]["crush"], "#3a6fd0")):
        d, f = np.asarray(c["distance"]), np.asarray(c["force"]) / 1000
        ax.plot(d, f, color=col, lw=1.8, label=label)
        ax.axvline(stop, color=col, lw=0.8, ls=":")          # where the rover stops
    ax.axhline(lim, color="k", lw=0.9, ls="--")
    ax.text(0.3, lim * 1.02, f"{lim:g} kN limit", fontsize=8, va="bottom")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, lim * 1.6)
    ax.set_xlabel("crush (mm)")
    ax.set_ylabel("force (kN)")
    ax.set_title(f"force against crush; dotted: where the {e / 1000:.0f} J drop stops", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "bumper-crush.png"))


def main():
    start = json.load(open(os.path.join(HERE, "start", "report.json")))
    graded = json.load(open(os.path.join(HERE, "graded", "report.json")))
    graded_doc = json.load(open(os.path.join(HERE, "start", "bumper-graded.json")))
    os.makedirs(IMAGES, exist_ok=True)
    curves(start, graded)
    parts(start, graded_doc)


if __name__ == "__main__":
    main()
