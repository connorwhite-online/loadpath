"""
The touch microphone's push pad as the eval surface: its real exterior,
board, switch and opening (connorwhite-online/touch-mic, hardware/, round
17), with the inside designed for a force map.

    python examples/switch/touch_mic.py --touch-mic ../touch-mic --out tm-out [--pitches 1.0] [--baseline]

The part: a 50 x 59 x 12.3 mm SLS TPU 88A pillow with a sunk touch area
over a 6 x 6 mm tactile switch (65 gf trigger, 0.25 mm travel). The cone
under the dish is the plunger; its tip sits 0.4 mm over the switch's
actuator. The underside opens for a flap that carries the switch; the
board goes up through the opening into rails on the body.

What the study keeps:
- the exterior surfacing, as a 1.0 mm skin (SLS TPU's minimum wall:
  Materialize lists 0.8 mm, the owner's rule is 1.0);
- the base ring with the flap's seat, latch groove and feet (z < 3.7),
  the board rails, and the plunger tip;
- clear: the opening (the flap's plug fills it), everything over the
  board, the switch column and the stem with its gap, the USB-C plug's
  path (insertFrom +y).

Everything else is design space: the skin's thickness under the touch area,
the cone, the rim tube's walls. In round 17 the skin is 1.0 mm, the cone
1.5 mm at the centre and the walls 1.8 mm.

The target, after the touch-mic agent's note: a ceiling of 150 gf at the
centre rising to 300 gf at 14 mm, a 100 gf floor, the skin over the board
moving under 1.4 mm, and the flanks holding their shape under a squeeze.
`--baseline` reads the fire forces of round 17 as drawn and of the bare
skin.

The OpenSCAD step needs `openscad` (2021.01 works) and a touch-mic
checkout; the STLs land in --out and are reused. Results, renders and
the force map: docs/SWITCH.md ("On a real product"); the renders come from
render.py and force_chart.py beside this file.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

SCAD = """include <body.scad>
part = "none";
which = "outer";
bx = board_w / 2 + fit + rail_w + 0.4;
if (which == "outer") difference() { outer_solid(); rebate_cut(); hole_cut(); latch_cut(); }
if (which == "base") intersection() { body(); translate([-40, -40, -1]) cube([80, 80, ledge_z + 1]); }
if (which == "bay") board_bay();
if (which == "body") body();
if (which == "opening") translate([0, 0, -2]) linear_extrude(ledge_z + 2.5) plan2(hole_d - 0.5);
if (which == "boardspace") intersection() { cavity(); translate([-bx, board_y0 - 1.8, -1]) cube([2 * bx, 60, H + 2]); }
if (which == "flap") flap_closed(strap = false);
if (which == "switch") dummy_switch();
if (which == "pcb") translate([-board_w / 2, board_y0, board_z0]) cube([board_w, board_l, board_t]);
if (which == "receptacle") translate([0, board_y1 + conn_over - 7.35, board_z1 + conn_h / 2])
    rotate([-90, 0, 0]) hull() for (s = [-1, 1]) translate([s * (8.94 - 3.26) / 2, 0, 0]) cylinder(h = 7.35, d = 3.26, $fn = 24);
if (which == "chip") translate([-4, board_y0 + 4, board_z1]) cube([8, 8, 1.0]);
"""
PARTS = ("outer", "base", "bay", "body", "opening", "boardspace", "flap", "switch", "pcb", "receptacle", "chip")
# what the study reads from a case folder (LOADPATH_TOUCH_MIC_STL)
CASE = ("outer", "base", "bay", "opening", "boardspace", "flap", "switch", "pcb", "receptacle", "chip")

# round 17's numbers (body.scad echo): actuator top 7.7, cone tip 8.1, skin
# top at the dish centre 9.6, USB-C centre z 6.6 at y 27.8
ACT_TOP, TIP, TOP_C = 7.7, 8.1, 9.6
USB_Y, USB_Z = 27.84, 6.6
# the touch-mic agent's press points, mm from the switch: centre first
PTS = [(0, 0), (0, -7), (0, -13), (7, -5), (12, -9), (11, 0), (14, 6), (8, 11), (0, 12),
       (-7, -5), (-12, -9), (-11, 0), (-14, 6), (-8, 11)]


def geometry(touch_mic: str, out: str) -> dict:
    """The STLs from body.scad (OpenSCAD), cached in out."""
    import trimesh
    hw = os.path.join(os.path.abspath(touch_mic), "hardware")
    scad = os.path.join(hw, "_loadpath_extract.scad")
    paths = {w: os.path.abspath(os.path.join(out, f"tm-{w}.stl")) for w in PARTS}
    todo = [w for w in PARTS if not os.path.exists(paths[w])]
    if todo:
        with open(scad, "w") as fh:
            fh.write(SCAD)
        try:
            for w in todo:
                subprocess.run(["openscad", "-D", f'which="{w}"', "-o", paths[w], scad], check=True,
                               capture_output=True)
                m = trimesh.load(paths[w], force="mesh")
                m.export(paths[w])       # binary
        finally:
            os.remove(scad)
    return paths


def target(x, y):
    r = min(np.hypot(x, y) / 14.0, 1.0)
    return 150 + 150 * r


# the flap stands on the desk on a foot under the switch; the switch sits
# in a pocket on it, its body's underside at z 2.7
SEAT = {"type": "box", "min": [-3.3, -3.0, 2.6], "max": [3.3, 3.0, 2.8]}
DESK = {"type": "box", "min": [-30, -35, -0.5], "max": [30, 35, 0.31]}


def study(paths: dict, pitch: float = 1.0) -> dict:
    """The push pad's study, as written by hand from the CAD. With the
    flap's, the board parts' STLs in `paths` (the default since the
    designer-handoff work): the switch bears on the closed flap, whose give
    counts, and each part on the board has its own clearance. `handoff`
    below writes the same thing from the CAD with `loadpath handoff`."""
    full = all(k in paths for k in ("flap", "pcb", "receptacle", "chip"))
    doc = {
        "name": "touch-mic",
        "description": "Touch microphone push pad (round 17 surfacing): interior designed for a fire-force map.",
        "designSpace": {"type": "mesh", "path": paths["outer"]},
        "skin": {"thickness": 1.0},
        "preserve": [
            {"id": "base", "shape": {"type": "mesh", "path": paths["base"]}},
            {"id": "rails", "shape": {"type": "mesh", "path": paths["bay"]}},
            {"id": "tip", "shape": {"type": "cylinder", "p0": [0, 0, TIP], "p1": [0, 0, TOP_C - 1.0], "r": 1.6}},
        ],
        "keepOut": [
            {"id": "opening", "bore": False, "shape": {"type": "mesh", "path": paths["opening"]}},
            # the board's clear zone is the cavity over it, rails and all: the
            # rails stay (a keep-out round a preserve takes it out of the part)
            {"id": "board", "bore": False, "shape": {"type": "subtract", "shapes": [
                {"type": "mesh", "path": paths["boardspace"]}, {"type": "mesh", "path": paths["bay"]}]}},
            {"id": "switch", "bore": False, "insertFrom": "-z",
             "shape": {"type": "box", "min": [-6.8, -5.6, 0], "max": [6.8, 4.6, 7.0]}},
            {"id": "stem", "bore": False, "shape": {"type": "cylinder", "p0": [0, 0, -1], "p1": [0, 0, TIP], "r": 2.7}},
            {"id": "usb-plug", "bore": False, "insertFrom": "+y",
             "shape": {"type": "box", "min": [-6.0, USB_Y - 3.0, USB_Z - 3.0], "max": [6.0, USB_Y + 1.0, USB_Z + 3.0]}},
        ],
        "supports": [{"shape": {"type": "box", "min": [-30, -35, -0.5], "max": [30, 35, 0.31]}}],
        "loadCases": [{"name": "squeeze", "loads": [
            {"shape": {"type": "box", "min": [23.0, -8, 4.5], "max": [26, 8, 9]}, "force": [-2.5, 0, 0]},
            {"shape": {"type": "box", "min": [-26, -8, 4.5], "max": [-23.0, 8, 9]}, "force": [2.5, 0, 0]}]}],
        "material": "tpu88a",
        "minMember": 1.0,
        "goal": {"volumeFraction": 0.2},
        "resolution": {"pitch": pitch},
        "switch": {
            "unit": "gf",
            "output": {"region": "tip", "direction": "-z", "gap": round(TIP - ACT_TOP, 3), "trigger": 65,
                       "travel": 0.25},
            "presses": [{"at": [x, y], "max": round(target(x, y), 1), "min": 100} for x, y in PTS],
            "pressRadius": 1.0,
            "clearance": {"region": "board", "gap": 1.4},
            "hold": [{"case": "squeeze", "maxDeflection": 0.3}],
        },
    }
    if full:
        # no skin across the opening's inner end: round 17's exterior
        # solid ends there, and a skin lining it was a 0.4 mm membrane
        # across the board bay in the first studies' parts
        doc["skin"] = {"thickness": 1.0, "openings": [{"shape": {"type": "mesh", "path": paths["opening"]},
                                                       "inward": "+z", "depth": 1.2}]}
        doc["switch"]["output"]["bearsOn"] = {"part": {"shape": {"type": "mesh", "path": paths["flap"]},
                                                       "seat": SEAT, "supports": [DESK]}}
        doc["switch"]["clearance"] = [{"component": {"type": "mesh", "path": paths[k]}, "id": k, "margin": 0.2,
                                       "region": "skin"} for k in ("pcb", "receptacle", "chip")]
    return doc


def brief(paths: dict) -> dict:
    """The designer's handoff for the same product: the exterior, the
    base and rails as drawn, the components and the assembly order. The
    board goes in first, then the flap with the switch in its pocket
    (its snap bead reaches 0.6 mm past its edge), then the USB-C plug from
    outside. `loadpath handoff` finds the paths, the plunger and the rest."""
    s = study(paths)
    return {
        "name": "touch-mic-handoff",
        "description": "Touch microphone push pad (round 17 surfacing) from the designer's handoff.",
        "exterior": paths["outer"], "skin": 1.0, "openings": [paths["opening"]],
        "material": "tpu88a", "minMember": 1.0,
        "fixed": [{"id": "base", "shape": paths["base"]}, {"id": "rails", "shape": paths["bay"]}],
        "components": [
            {"id": "pcb", "shape": paths["pcb"]},
            {"id": "receptacle", "shape": paths["receptacle"], "installedOn": "pcb"},
            {"id": "chip", "shape": paths["chip"], "installedOn": "pcb"},
            {"id": "flap", "shape": paths["flap"], "role": "seat", "insertFrom": "-z", "pathTolerance": 0.7},
            {"id": "switch", "shape": paths["switch"], "role": "switch", "installedOn": "flap"},
            {"id": "usb-plug", "external": True, "fixed": True, "insertFrom": "+y", "clearance": False,
             "shape": {"type": "box", "min": [-6.0, USB_Y - 3.0, USB_Z - 3.0], "max": [6.0, USB_Y + 1.0, USB_Z + 3.0]}}],
        "assembly": ["pcb", "flap", "usb-plug"],
        # the designer's clear zone: nothing grows over the board
        "keepOut": [{"id": "board-bay", "bore": False, "shape": {"type": "subtract", "shapes": [
            {"type": "mesh", "path": paths["boardspace"]}, {"type": "mesh", "path": paths["bay"]}]}}],
        "switch": {"unit": "gf", "trigger": 65, "travel": 0.25, "gap": 0.4, "plungerRadius": 1.6,
                   "presses": s["switch"]["presses"], "pressRadius": 1.0,
                   "hold": [{"case": "squeeze", "maxDeflection": 0.3}]},
        "loadCases": s["loadCases"],
        "goal": {"volumeFraction": 0.2},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touch-mic", required=True)
    ap.add_argument("--out", default="touch-mic-out")
    ap.add_argument("--pitches", default="1.0")
    ap.add_argument("--iterations", default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--handoff", action="store_true",
                    help="write the designer's handoff brief and build its study with `loadpath handoff`")
    ap.add_argument("--stl-only", action="store_true",
                    help="write the touch-mic-pad eval case's STLs into --out (point LOADPATH_TOUCH_MIC_STL at it)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from loadpath.domain import build_domain
    from loadpath.domain import _fraction
    from loadpath.shapes import MeshShape
    from loadpath.spec import study_from_dict
    from loadpath.switch import GF, _thin_skin, fire_forces, run_switch, switch_from_study

    import trimesh
    paths = geometry(a.touch_mic, a.out)
    if a.stl_only:
        from loadpath.shapes import manifold_from_trimesh
        for w in CASE:
            m = trimesh.load(paths[w], force="mesh")
            if w in ("outer", "base"):
                # within 0.01 mm of the extract, a third of the triangles
                o = manifold_from_trimesh(m).simplify(0.01).to_mesh()
                m = trimesh.Trimesh(o.vert_properties[:, :3], o.tri_verts)
            m.export(os.path.join(a.out, f"touch-mic-{w}.stl"))
        print(os.path.abspath(a.out))
        return
    if a.handoff:
        from loadpath.handoff import run_handoff
        bp = os.path.join(a.out, "touch-mic.brief.json")
        json.dump(brief(paths), open(bp, "w"), indent=1)
        r = run_handoff(bp, os.path.join(a.out, "handoff"))
        print(json.dumps({"study": r["studyPath"], "assembly": {k: v.get("insertFrom") for k, v in
                                                                  r["report"]["assembly"]["components"].items()},
                          "warnings": r["report"]["warnings"]}, default=float))
        return
    pitches = [float(p) for p in a.pitches.split(",")]
    doc = study(paths, pitches[0])
    json.dump(doc, open(os.path.join(a.out, "touch-mic.study.json"), "w"), indent=1)
    st = study_from_dict(doc)
    if a.baseline:
        t0 = time.time()
        dom, floor = _thin_skin(st, build_domain(st, pitches[0]))
        spec = switch_from_study(st, dom)
        body = MeshShape(trimesh.load(paths["body"], force="mesh"))
        drawn = _fraction([body], dom.mesh.elem_centers(), dom.pitch)
        rows = {}
        for name, x in (("round 17 as drawn", np.maximum(drawn, floor)),
                        ("skin, base and rails only", np.maximum(dom.preserve, floor))):
            r = fire_forces(dom, st.elastic(design=True), spec, np.maximum(x, 1e-9), float(st.material.E))
            rows[name] = {"forceGf": (r["forces"] / GF).tolist(), "hold": r["hold"].tolist(),
                          "clearanceMax": r.get("clearanceMax")}
            print(f"{name}: cells {dom.n_elem}, fire gf", np.round(r["forces"] / GF).astype(int).tolist(),
                  "squeeze mm", np.round(r["hold"], 3).tolist(), "clearance mm", round(r.get("clearanceMax", 0), 2),
                  f"{time.time() - t0:.0f}s", flush=True)
        np.save(os.path.join(a.out, f"drawn-{dom.pitch:g}.npy"), drawn)
        json.dump(rows, open(os.path.join(a.out, f"baseline-{dom.pitch:g}.json"), "w"), indent=1)
        return
    its = [int(v) for v in a.iterations.split(",")] if a.iterations else None
    rep = run_switch(st, a.out, pitches=pitches, iterations=its,
                     on_progress=lambda ev, *_: print(json.dumps({k: v for k, v in ev.items() if k != "forces"}),
                                                      flush=True))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    for p in rep["presses"]:
        print(f"  at {p['at']} window {p['min'] / GF:.0f}-{p['max'] / GF:.0f} gf: {p['forceGf']:.0f} "
              f"(as printed {p.get('thresholdedForceGf', 0):.0f}) {'ok' if p['ok'] else 'MISS'}")


if __name__ == "__main__":
    main()
