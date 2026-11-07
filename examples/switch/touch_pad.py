"""
A soft push pad over a tactile switch: keep the industrial design's
exterior, design everything inside so a press anywhere on the touch area
fires the switch at a chosen force, with the least material.

    python examples/switch/touch_pad.py [--out touch-pad-out/] [--pitches 1.0,0.7] [--baseline]

The part (synthetic, after a touch microphone): a 50 x 59 x 9 mm rounded
puck in SLS TPU 88A with a 1 mm deep dish (the touch area) in its top. The
board slides in from below and fills the inside up to its components'
height; a 6 x 6 mm tactile switch (65 gf trigger, 0.25 mm travel) sits at
the centre, and a plunger hanging from the skin clears its stem by 0.4 mm.

Kept exactly: a 0.7 mm skin of the exterior (open underneath, where the
board goes in) and the plunger. Kept clear: the board and the switch,
each with its assembly path (`insertFrom: "-z"`), so nothing generated
blocks putting the pad together. Everything else is design space.

The target is a force map over the touch area: 160 gf at the centre
rising to 200 gf at 12 mm out, never under 110 gf, skin travel under the
board clearance, and the rim's shape held under a 5 N side squeeze.
`--baseline` reads the fire forces of the solid part (everything inside
filled), the analogue of preserving the whole shell.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

W, L, H = 50.0, 59.0, 9.0      # mm
CORNER = 12.0
DISH_R, DISH_DEPTH = 15.0, 1.0
SKIN = 0.7
BOARD_TOP, COMPONENTS = 1.6, 5.6           # board top, component envelope top
SWITCH_TOP, STEM_TOP, GAP = 4.7, 5.6, 0.4


def body_mesh():
    """The exterior: a rounded puck with a spherical dish in its top."""
    import manifold3d as m3d
    import trimesh

    cs = m3d.CrossSection.square((W - 2 * CORNER, L - 2 * CORNER), center=True).offset(
        CORNER, m3d.JoinType.Round, circular_segments=96)
    body = m3d.Manifold.extrude(cs, H)
    R = (DISH_R ** 2 + DISH_DEPTH ** 2) / (2 * DISH_DEPTH)
    sphere = m3d.Manifold.sphere(R, 384).translate((0.0, 0.0, H - DISH_DEPTH + R))
    body = body - sphere
    m = body.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)


def study(stl: str, pitch: float = 1.0) -> dict:
    top = H - DISH_DEPTH            # skin top at the centre
    return {
        "name": "touch-pad",
        "description": "Soft SLS TPU push pad over a 6x6 tactile switch: interior designed for a fire-force map.",
        "designSpace": {"type": "mesh", "path": stl},
        "skin": {"thickness": SKIN, "open": ["-z"]},
        "preserve": [
            {"id": "plunger", "shape": {"type": "cylinder", "p0": [0, 0, STEM_TOP + GAP], "p1": [0, 0, top],
                                        "r": 1.5}},
        ],
        "keepOut": [
            {"id": "board", "bore": False, "insertFrom": "-z",
             "shape": {"type": "box", "min": [-20, -24.5, -1], "max": [20, 24.5, COMPONENTS]}},
            {"id": "switch", "bore": False, "insertFrom": "-z",
             "shape": {"type": "box", "min": [-3.5, -3.5, BOARD_TOP], "max": [3.5, 3.5, SWITCH_TOP]}},
            {"id": "stem", "bore": False,
             "shape": {"type": "cylinder", "p0": [0, 0, SWITCH_TOP], "p1": [0, 0, STEM_TOP + GAP], "r": 2.2}},
        ],
        "supports": [{"shape": {"type": "box", "min": [-30, -35, -0.5], "max": [30, 35, 0.3]}}],
        "loadCases": [{"name": "squeeze", "loads": [
            {"shape": {"type": "box", "min": [W / 2 - 1.2, -8, 3], "max": [W / 2 + 1, 8, H - 1]}, "force": [-2.5, 0, 0]},
            {"shape": {"type": "box", "min": [-W / 2 - 1, -8, 3], "max": [-W / 2 + 1.2, 8, H - 1]}, "force": [2.5, 0, 0]},
        ]}],
        "material": "tpu88a",
        "minMember": 0.8,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": pitch},
        "switch": {
            "unit": "gf",
            "output": {"region": "plunger", "direction": "-z", "gap": GAP, "trigger": 65, "travel": 0.25},
            "field": {"center": [0, 0], "radius": 12, "rings": 2, "perRing": 6, "max": [160, 200], "min": 110},
            "pressRadius": 1.0,
            "clearance": {"region": "skin", "gap": 1.6},
            "hold": [{"case": "squeeze", "maxDeflection": 0.25}],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="touch-pad-out")
    ap.add_argument("--pitches", default="1.0")
    ap.add_argument("--iterations", default=None, help="per level, comma-separated")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from loadpath.domain import build_domain
    from loadpath.spec import study_from_dict
    from loadpath.switch import GF, fire_forces, optimize_switch, switch_from_study

    stl = os.path.abspath(os.path.join(a.out, "touch-pad-body.stl"))
    body_mesh().export(stl)
    pitches = [float(p) for p in a.pitches.split(",")]
    doc = study(stl, pitches[0])
    json.dump(doc, open(os.path.join(a.out, "touch-pad.study.json"), "w"), indent=2)
    st = study_from_dict(doc)
    t0 = time.time()
    if a.baseline:
        from loadpath.switch import _thin_skin
        dom, floor = _thin_skin(st, build_domain(st, pitches[0]))
        spec = switch_from_study(st, dom)
        for name, x in (("filled", np.ones(dom.n_elem)), ("skin only", np.maximum(dom.preserve, floor))):
            r = fire_forces(dom, st.elastic(design=True), spec, x, float(st.material.E))
            print(f"{name}: cells {dom.n_elem}, fire gf", np.round(r["forces"] / GF).astype(int).tolist(),
                  "squeeze mm", np.round(r["hold"], 3).tolist(), f"{time.time() - t0:.0f}s", flush=True)
        return
    its = [int(v) for v in a.iterations.split(",")] if a.iterations else None

    def note(rec, _):
        if rec["iteration"] % 5 == 0 or rec["iteration"] == 1:
            f = np.asarray(rec["forces"]) / GF
            print(f"L{rec['level']} it {rec['iteration']:3d} vol {rec['volume']:.3f} worst {rec['worstViolation']:+.3f} "
                  f"beta {rec['beta']:g} gf {f.min():.0f}-{f.max():.0f} centre {f[0]:.0f} {rec['seconds']}s", flush=True)

    r = optimize_switch(st, pitches=pitches, iterations=its, on_iter=note)
    np.save(os.path.join(a.out, "xphys.npy"), r.xphys)
    json.dump({"levels": r.levels, "report": r.report, "seconds": r.seconds}, open(os.path.join(a.out, "result.json"), "w"),
              indent=2)
    for p in r.report["presses"]:
        print(f"  at {np.round(p['at'], 1).tolist()} target <= {p['max'] / GF if p['max'] else '-':.0f} gf: "
              f"{p['forceGf']:.0f} gf, travel {p['travel']:.2f} mm {'ok' if p['ok'] else 'MISS'}")
    print("volume", round(r.volume, 3), "seconds", round(r.seconds))


if __name__ == "__main__":
    main()
