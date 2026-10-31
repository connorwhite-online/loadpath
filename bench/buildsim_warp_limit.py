"""
What a warp limit costs in mass: the Ti64 LPBF shelf bracket of
bench/buildsim_study_warp.py (40 x 6 x 24 mm, built standing on the plate,
bolted along its left face, 1500 N down at the top right corner), run
through the study's safety-factor search (SF 2, lightest passing part)
with no warp limit and then with buildSim.maxDistortion and
buildSim.maxRecoaterLift set below what the unlimited design builds to.
Each row is the lightest part that meets SF 2 and the limits; the study's
own buildSim pass re-simulates it on its delivered mesh (about 8000 cells,
5 mm plate, as built on the plate).

Run: python bench/buildsim_warp_limit.py [--fractions 0.6 0.35] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import time

from loadpath.spec import study_from_dict
from loadpath.study import run_study


def study(name, limits):
    return {"name": name, "material": "ti64", "process": {"type": "lpbf", "buildDirection": "+z"},
            "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [40, 6, 24]}],
            "preserve": [{"id": "flange", "shape": {"type": "box", "min": [0, 0, 0], "max": [2, 6, 24]}},
                         {"id": "pad", "shape": {"type": "box", "min": [36, 0, 21], "max": [40, 6, 24]}}],
            "supports": [{"region": "flange"}], "loads": [{"region": "pad", "force": [0, 0, -1500]}],
            "goal": {"safetyFactor": 2.0}, "resolution": {"maxElements": 6000}, "extrusion": "y",
            "buildSim": {"cells": 8000, "plate": 5.0, "release": "none", **limits}}


def row(name, limits):
    t0 = time.time()
    res = run_study(study_from_dict(study(name, limits)))
    out = {"name": name, "limits": limits, "seconds": round(time.time() - t0),
           "tried": [{"vf": round(q.volume_fraction, 4), "mass": round(q.report.get("mass", 0), 2),
                      "sf": q.report.get("safetyFactor"), "warp": q.report.get("warp"), "passes": q.passes}
                     for q in res.outcomes]}
    if res.best is None:
        out["error"] = "no outcome met the goal"
        return out
    o = res.outcomes[res.best]
    b = o.report.get("buildSim", {})
    out.update(mass=o.report["mass"], safetyFactor=o.report.get("safetyFactor"), warp=o.report.get("warp"),
               volumeFraction=o.volume_fraction)
    if "distortion" in b:
        out["buildSim"] = {"distortionMax": b["distortion"]["max"], "recoaterLift": b["recoater"]["maxLift"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.6, 0.35],
                    help="limits as fractions of the unlimited design's as-built max distortion and lift")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    # a limit nothing reaches: the plain SF search, with the warp measured
    base = row("no limit", {"maxDistortion": 100.0})
    rows.append(base)
    print(json.dumps(base), flush=True)
    d0, l0 = base["warp"]["maxDistortion"], base["warp"]["recoaterLift"]
    for f in a.fractions:
        r = row(f"distortion <= {f:g} x", {"maxDistortion": round(f * d0, 4)})
        rows.append(r)
        print(json.dumps(r), flush=True)
    if l0 > 0:
        r = row("lift <= 0.35 x", {"maxRecoaterLift": round(0.35 * l0, 4)})
        rows.append(r)
        print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
