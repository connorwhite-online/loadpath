"""
Warp-aware design through the study JSON (buildSim.warpWeight): a Ti64
LPBF shelf bracket, 40 x 6 x 24 mm, built standing on the plate (+z),
bolted along its left face and loaded down at the top right corner, at a
fixed 35% volume, run through run_study with warpWeight 0 and 1. Each delivered part is then
verified by the study's own buildSim pass (the full build model on the
delivered mesh, about 8000 cells, 5 mm plate, as built on the plate).

Run: python bench/buildsim_study_warp.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import time

from loadpath.spec import study_from_dict
from loadpath.study import run_study


def study(w):
    return {"name": f"shelf-w{w:g}", "material": "ti64", "process": {"type": "lpbf", "buildDirection": "+z"},
            "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [40, 6, 24]}],
            "preserve": [{"id": "flange", "shape": {"type": "box", "min": [0, 0, 0], "max": [2, 6, 24]}},
                         {"id": "pad", "shape": {"type": "box", "min": [36, 0, 21], "max": [40, 6, 24]}}],
            "supports": [{"region": "flange"}], "loads": [{"region": "pad", "force": [0, 0, -1500]}],
            "goal": {"volumeFraction": 0.35}, "resolution": {"maxElements": 6000}, "extrusion": "y",
            "buildSim": {"warpWeight": w, "cells": 8000, "plate": 5.0, "release": "none"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0])
    a = ap.parse_args()
    out = {}
    for w in a.weights:
        t0 = time.time()
        res = run_study(study_from_dict(study(w)))
        if res.best is None:
            print(json.dumps({"warpWeight": w, "error": "no outcome passed"}), flush=True)
            continue
        o = res.outcomes[res.best]
        b = o.report["buildSim"]
        row = {"warpWeight": w, "mass": o.report["mass"], "safetyFactor": o.report.get("safetyFactor"),
               "compliance": o.report.get("compliance"),
               "passes": o.passes, "volumeFraction": o.volume_fraction, "studySeconds": round(time.time() - t0)}
        if "error" in b:
            row["buildSimError"] = b["error"]
        else:
            row.update(distortionMax=b["distortion"]["max"], distortionRms=b["distortion"]["rms"],
                       recoaterLift=b["recoater"]["maxLift"], vmP95=b["residualStress"]["vmP95"])
        out[f"w{w:g}"] = row
        print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
