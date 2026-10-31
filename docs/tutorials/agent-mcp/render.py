"""
The image in this tutorial, from a full run of transcript.py (not part of the
tutorial; needs Node with Playwright's Chromium, see docs/render3d.py):

    python transcript.py && python render.py
"""
import json
import os
import sys

import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from gallery import _study_panel  # noqa: E402
from render3d import Scene  # noqa: E402

from loadpath.spec import study_from_dict  # noqa: E402

with open(os.path.join(HERE, "transcript.json")) as fh:
    t = json.load(fh)
with open(os.path.join(HERE, "wrist-link.json")) as fh:
    study = study_from_dict(json.load(fh), base_dir=HERE)
# the screws' preload arrows would hide the horn end: draw the three service loads
study.load_cases = [c for c in study.load_cases if c.name != "preload"]
mesh = trimesh.load(os.path.join(HERE, "wrist-link.stl"), force="mesh")
f = t["final"]

s = Scene(f"wrist link: {f['mass']:.1f} g PA12, SF {f['safetyFactor']:.2f}, "
          f"{f['payloadDeflection']:.2f} mm under the payload", panel_size=(560, 420))
_study_panel(s, study, mesh, (28, -60), "loads (blue), horn screws and bearing kept clear (red)")
_study_panel(s, study, mesh, (-32, -50), "from below: the face that sits on the servo horn")
s.render(os.path.join(HERE, "images", "wrist-link.png"))
