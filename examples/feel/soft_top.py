"""
Touch zones inside a fixed exterior: a soft-top controller, one piece of
SLS TPU 88A, with a distinctly pliable touch pad and a firm rim and sides.

    python examples/feel/soft_top.py --out soft-top-out [--pitches 1.0,0.7 --iterations 80,40] [--baseline]

A generic handheld, 56 x 40 x 16 mm with 4 mm rounded corners. The
exterior is industrial design: kept as a 2 mm skin, open at the bottom
where a lid closes it. A board slides in from the bottom (`insertFrom`).
The designer marks three places and how each should feel:

- pad: a 24 mm disc in the middle of the top, `pliable` (a light 2 N touch
  sinks 0.5 to 2 mm). Its skin is `auto`: as thick as it can be while the
  bare skin stays under the ceiling, never under the process's minimum
  wall.
- rim: the top outside a 30 mm circle, `firm` (a 5 N press moves it under
  0.33 mm).
- sides: the four walls, pressed from outside, `firm`.

and that the difference is distinct: every press on the rim and the sides
at least 4x as stiff as any press on the pad (`contrast`). Everything
inside the skin is design space; the product is held on the lid.

`--baseline` reads the bare skin and a filled interior against the same
zones before anything is optimized.
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

W, D, H, R = 56.0, 40.0, 16.0, 4.0
PAD_R, RIM_R = 12.0, 15.0


def _rounded_box():
    """The exterior: a box with rounded vertical edges (a union of two
    boxes and four corner cylinders)."""
    hw, hd = W / 2, D / 2
    parts = [{"type": "box", "min": [-hw + R, -hd, 0], "max": [hw - R, hd, H]},
             {"type": "box", "min": [-hw, -hd + R, 0], "max": [hw, hd - R, H]}]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append({"type": "cylinder", "p0": [sx * (hw - R), sy * (hd - R), 0],
                          "p1": [sx * (hw - R), sy * (hd - R), H], "r": R})
    return {"type": "union", "shapes": parts}


def study(pitch: float = 1.0, skin: float = 2.0, contrast: float = 4.0) -> dict:
    top = {"type": "box", "min": [-W, -D, H - 2.0], "max": [W, D, H + 1.0]}
    pad = {"type": "cylinder", "p0": [0, 0, H - 3.0], "p1": [0, 0, H + 1.0], "r": PAD_R}
    rim = {"type": "subtract", "shapes": [top, {"type": "cylinder", "p0": [0, 0, H - 3.0], "p1": [0, 0, H + 1.0],
                                                 "r": RIM_R}]}
    lid = {"face": "-z"}
    sides = []
    for name, d, zone in (
            ("side+x", "-x", {"type": "box", "min": [W / 2 - 2, -D / 2 + R, 3], "max": [W / 2 + 1, D / 2 - R, H - 3]}),
            ("side-x", "+x", {"type": "box", "min": [-W / 2 - 1, -D / 2 + R, 3], "max": [-W / 2 + 2, D / 2 - R, H - 3]}),
            ("side+y", "-y", {"type": "box", "min": [-W / 2 + R, D / 2 - 2, 3], "max": [W / 2 - R, D / 2 + 1, H - 3]}),
            ("side-y", "+y", {"type": "box", "min": [-W / 2 + R, -D / 2 - 1, 3], "max": [W / 2 - R, -D / 2 + 2, H - 3]})):
        sides.append({"kind": "touch", "name": name, "zone": zone, "direction": d, "hold": lid, "feel": "firm",
                      "spacing": 10.0})
    return {
        "name": "soft-top",
        "description": "Soft-top controller (SLS TPU 88A): exterior kept as a 2 mm skin, a pliable touch pad and a "
                       "firm rim and sides, the board's assembly path clear.",
        "designSpace": _rounded_box(),
        "skin": {"thickness": skin, "open": ["-z"]},
        "keepOut": [
            {"id": "board", "bore": False, "insertFrom": "-z",
             "shape": {"type": "box", "min": [-17, -13, 0.0], "max": [17, 13, 5.0]}},
        ],
        "material": "tpu88a",
        "process": {"type": "sls"},
        "minMember": 1.0,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": pitch},
        "feel": {
            "targets": [
                {"kind": "touch", "name": "pad", "zone": pad, "hold": lid, "feel": "pliable",
                 "skinThickness": "auto"},
                {"kind": "touch", "name": "rim", "zone": rim, "hold": lid, "feel": "firm"},
                *sides,
            ],
            "contrast": [{"softer": "pad", "firmer": "rim", "ratio": contrast}] +
                        [{"softer": "pad", "firmer": s["name"], "ratio": contrast} for s in sides],
            "iterations": 80,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="soft-top-out")
    ap.add_argument("--pitches", default="1.0")
    ap.add_argument("--iterations", default=None)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from loadpath.domain import build_domain
    from loadpath.feel import FeelProblem, feel_readings, run_feel
    from loadpath.spec import study_from_dict
    from loadpath.switch import _thin_skin
    from loadpath.touchzones import zone_summary

    pitches = [float(p) for p in a.pitches.split(",")]
    doc = study(pitches[0])
    json.dump(doc, open(os.path.join(a.out, "soft-top.study.json"), "w"), indent=1)
    st = study_from_dict(doc)
    if a.baseline:
        dom, floor = _thin_skin(st, build_domain(st, pitches[0]))
        prob = FeelProblem(st, dom, st.elastic(design=True), float(st.material.E))
        out = {}
        for name, x in (("bare skin", np.maximum(dom.preserve.astype(float), floor)), ("filled", np.ones(dom.n_elem))):
            rows = feel_readings(prob, np.maximum(x, 1e-9))
            out[name] = zone_summary(rows, st)
            print(name, json.dumps(out[name], default=float))
        json.dump(out, open(os.path.join(a.out, f"baseline-{dom.pitch:g}.json"), "w"), indent=1, default=float)
        return
    its = [int(v) for v in a.iterations.split(",")] if a.iterations else None
    t0 = time.time()
    rep = run_feel(st, a.out, pitches=pitches, iterations=its, modal=False,
                   on_progress=lambda ev: print(json.dumps({k: v for k, v in ev.items() if k != "forces"}), flush=True))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    print(json.dumps(rep.get("touchSkin"), default=float))
    for k, zs in rep.get("touchZones", {}).items():
        print(k, json.dumps(zs, default=float))
    print(f"mass {rep['mass']:.2f} g, watertight {rep['watertight']}, passes {rep['passes']}, "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
