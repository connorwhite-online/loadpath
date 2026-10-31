"""
The pictures in this tutorial's README, from its outputs (run the README's
commands first, without --quick for the README's numbers):

    python render.py            # writes images/*.png

Needs Node with Playwright's Chromium (see docs/render3d.py). Not run by
the tutorial tests.
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))                 # docs/: render3d
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))           # the repo: loadpath

from render3d import KEEP_OUT, Scene  # noqa: E402

from loadpath.assembly import Robot  # noqa: E402

UPPER = (0.30, 0.55, 0.85)      # the upper arm's sweep
WRIST = KEEP_OUT                # the wrist's sweep
LINKS = (0.55, 0.57, 0.62)      # the rest of the arm
VIEW = (24, -58)


def best_stl(folder):
    with open(os.path.join(HERE, folder, "report.json")) as fh:
        r = json.load(fh)
    o = r["outcomes"][r["best"]]
    return trimesh.load(os.path.join(HERE, folder, os.path.basename(o["stl"]))), o


def sweeps():
    return {n: trimesh.load(os.path.join(HERE, "derived", f"sweep-{n}.stl")) for n in ("upper_arm", "wrist")}


def arm(q=None):
    """The arm's URDF shapes in the study frame (mm), forearm left out."""
    r = Robot.from_urdf(os.path.join(HERE, "robot.urdf"))
    T = r.fk(q or {})
    out = []
    for link in r.bodies:
        if link == "forearm":
            continue
        for m in r.meshes(link):
            m.apply_transform(T[link])
            m.apply_scale(1000.0)
            out.append(m)
    return trimesh.util.concatenate(out)


def case_arrows(panel, study, scale=1.6):
    """Each derived case's force at the wrist pin."""
    for c in study["loadCases"]:
        f = np.asarray(c["loads"][0]["force"], float)
        panel.arrow((140.0, 0.0, 280.0), f * scale)


def main():
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    part, o = best_stl("out")
    sw = sweeps()
    with open(os.path.join(HERE, "derived", "forearm.derived.json")) as fh:
        derived = json.load(fh)

    # 1. the arm at home, with the designed forearm in place
    s = Scene(f"desk arm at home, forearm designed from its URDF: {o['mass']:.0f} g", panel_size=(760, 560))
    p = s.panel(None, view=(18, -62))
    p.mesh(arm(), color=LINKS)
    p.mesh(part)
    s.render(os.path.join(HERE, "images", "arm.png"), out_scale=1.2)

    # 2. what assembly derived: the neighbours' sweeps and the governing cases
    s = Scene("derived: the upper arm's and the wrist's sweeps (1 mm clearance), governing forces")
    p = s.panel("sweeps in the design space", view=VIEW)
    lo, hi = np.array([-22, -24, 256.0]), np.array([162, 24, 304.0])
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    p.mesh(box, color=(0.75, 0.75, 0.78), opacity=0.08)
    p.mesh(sw["upper_arm"], color=UPPER, opacity=0.55)
    p.mesh(sw["wrist"], color=WRIST, opacity=0.35)
    case_arrows(p, derived)
    p = s.panel("the forearm grown round them", view=VIEW)
    p.mesh(part)
    p.mesh(sw["upper_arm"], color=UPPER, opacity=0.18)
    p.mesh(sw["wrist"], color=WRIST, opacity=0.18)
    s.render(os.path.join(HERE, "images", "derived.png"))

    # 3. hand-written loads vs derived
    hand, ho = best_stl("hand")
    with open(os.path.join(HERE, "hand-check", "verify.json")) as fh:
        hv = json.load(fh)
    s = Scene("one hand-written case vs the cases swept from the arm")
    p = s.panel(f"hand-written load: {ho['mass']:.0f} g, SF {hv['safetyFactor']:.2f} under the derived cases",
                view=(30, -120))
    p.mesh(hand)
    p.mesh(sw["upper_arm"], color=UPPER, opacity=0.18)
    p.mesh(sw["wrist"], color=WRIST, opacity=0.18)
    p = s.panel(f"derived loads: {o['mass']:.0f} g, SF {o['safetyFactor']:.2f}", view=(30, -120))
    p.mesh(part)
    p.mesh(sw["upper_arm"], color=UPPER, opacity=0.18)
    p.mesh(sw["wrist"], color=WRIST, opacity=0.18)
    s.render(os.path.join(HERE, "images", "compare.png"))
    for f in ("arm", "derived", "compare"):
        print(f, os.path.getsize(os.path.join(HERE, "images", f + ".png")) // 1024, "KB")


if __name__ == "__main__":
    main()
