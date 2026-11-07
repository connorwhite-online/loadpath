"""
Feel targets inside a fixed exterior: a handheld microphone body.

    python examples/feel/mic_body.py --out mic-out [--pitches 1.0,0.7 --iterations 120,60] [--baseline]

The passing design in docs/FEEL.md was made with LOADPATH_SKIN_MODEL=floor
(the cut-cell skin model stalls on the drop at 0.7 mm; see there).

A generic stick microphone, 24 mm across and 60 mm long, in SLS TPU 88A:
the exterior is industrial design, kept as a 1.0 mm skin, open at the
bottom where a plug closes it. Inside are a 10 mm capsule at the head on a
ring mount, its cable path down the middle, and a board slid in from the
bottom (`insertFrom: "-z"` on both, so nothing grown inside blocks
assembly). Everything else is design space, and three feel targets set
it:

- squeeze: a 20 N grip across the handle gives 0.3 to 1.0 mm. Soft-touch,
  not mushy.
- drop: a 1 m fall on the head. The 4 g capsule sees under 1500 g and
  moves under 3 mm (the room the head leaves it).
- isolate: handling noise from the hand on the handle reaches the capsule
  at no more than half its amplitude from 500 Hz up, in all three
  directions, and a 20 g knock moves the capsule under 0.5 mm.

`--baseline` reads the bare skin and a filled interior against the same
targets before anything is optimized.
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

R, L = 12.0, 60.0
CAP_Z0, CAP_Z1, CAP_R = 47.0, 55.0, 5.0


def study(pitch: float = 1.0) -> dict:
    grip = {"type": "cylinder", "p0": [0, 0, 8], "p1": [0, 0, 36], "r": R + 1}
    return {
        "name": "mic-body",
        "description": "Handheld microphone body (SLS TPU 88A): exterior kept as a 1.0 mm skin, capsule and board "
                       "assembly paths clear, interior designed for grip squeeze, drop cushioning and handling-noise "
                       "isolation.",
        "designSpace": {"type": "cylinder", "p0": [0, 0, 0], "p1": [0, 0, L], "r": R},
        "skin": {"thickness": 1.0, "open": ["-z"]},
        "preserve": [
            {"id": "mount", "shape": {"type": "cylinder", "p0": [0, 0, CAP_Z0 - 1.0], "p1": [0, 0, CAP_Z1],
                                      "r": CAP_R + 1.5}},
        ],
        "keepOut": [
            {"id": "capsule", "bore": False, "insertFrom": "-z",
             "shape": {"type": "cylinder", "p0": [0, 0, CAP_Z0], "p1": [0, 0, CAP_Z1 + 0.01], "r": CAP_R}},
            {"id": "board", "bore": False, "insertFrom": "-z",
             "shape": {"type": "box", "min": [-7, -2.5, 4], "max": [7, 2.5, 30]}},
        ],
        "material": "tpu88a",
        "minMember": 1.0,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": pitch},
        "feel": {
            "targets": [
                {"kind": "squeeze", "name": "grip",
                 "hold": {"face": "-x", "shape": grip}, "press": {"face": "+x", "shape": grip},
                 "direction": "-x", "force": 20, "travel": [0.3, 1.0]},
                {"kind": "drop", "name": "head-drop", "lands": {"face": "+z"}, "mass": {"region": "mount"},
                 "kg": 0.004, "height": 1.0, "maxG": 1500, "maxTravel": 3.0},
                {"kind": "isolate", "name": "handling-noise", "hold": {"region": "skin", "shape": grip},
                 "mount": {"region": "mount"}, "kg": 0.004, "band": 500, "maxTransmissibility": 0.5,
                 "maxTravel": 0.5, "shockG": 20},
            ],
            "iterations": 120,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mic-out")
    ap.add_argument("--pitches", default="1.0")
    ap.add_argument("--iterations", default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--move-sharp", type=float, default=None, help="feel.moveSharp (default 0.03)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from loadpath.domain import build_domain
    from loadpath.feel import FeelProblem, feel_readings, run_feel
    from loadpath.spec import study_from_dict
    from loadpath.switch import _thin_skin

    pitches = [float(p) for p in a.pitches.split(",")]
    doc = study(pitches[0])
    if a.move_sharp:
        doc["feel"]["moveSharp"] = a.move_sharp
    json.dump(doc, open(os.path.join(a.out, "mic-body.study.json"), "w"), indent=1)
    st = study_from_dict(doc)
    if a.baseline:
        dom, floor = _thin_skin(st, build_domain(st, pitches[0]))
        prob = FeelProblem(st, dom, st.elastic(design=True), float(st.material.E))
        rows = {}
        for name, x in (("bare skin", np.maximum(dom.preserve.astype(float), floor)), ("filled", np.ones(dom.n_elem))):
            rows[name] = feel_readings(prob, np.maximum(x, 1e-9))
            for r in rows[name]:
                print(name, r["name"], [(d["direction"], round(d["stiffness"], 2), d["ok"]) for d in r["directions"]],
                      f"window {r['minStiffness']}-{r['maxStiffness']}")
        json.dump(rows, open(os.path.join(a.out, f"baseline-{dom.pitch:g}.json"), "w"), indent=1, default=float)
        return
    its = [int(v) for v in a.iterations.split(",")] if a.iterations else None
    t0 = time.time()
    rep = run_feel(st, a.out, pitches=pitches, iterations=its,
                   on_progress=lambda ev: print(json.dumps({k: v for k, v in ev.items() if k != "forces"}), flush=True))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    for r in rep["feel"]:
        print(r["name"], "ok" if r["ok"] else "MISS", "delivered", "ok" if r.get("deliveredOk") else "MISS")
        for d, dd in zip(r["directions"], r.get("delivered") or r["directions"]):
            print("   ", d["direction"], f"k {d['stiffness']:.2f} (delivered {dd['stiffness']:.2f}) N/mm",
                  f"window {r['minStiffness']}-{r['maxStiffness']}",
                  {k: round(v, 3) for k, v in d.items() if k in ("travel", "peakG", "frequency", "transmissibility")})
        if "modal" in r:
            print("    modes", r["modal"].get("frequencies"), r["modal"].get("directions"))
    print(f"mass {rep['mass']:.2f} g, watertight {rep['watertight']}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
