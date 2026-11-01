"""
FDM warp as a limit in the safety-factor search: a 120 x 24 x 12 mm beam
bracket printed flat on the bed (+z), bolted at both end pads and loaded
200 N down at mid-span, in PETG and in nylon (PA12), SF 2. Each material
runs with no limit, then with buildSim.maxDistortion at half of what the
unlimited design builds to (its delivered part re-simulated, which is
what the limit is checked on). maxDistortion for FDM is the largest
movement along the build direction at the end of the print, cooled on the
bed: the corner lift off the bed (in-plane shrink is uniform and the
slicer scales it out).

Run: python bench/buildsim_fdm_warp_limit.py [--materials petg pa12] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import time

from loadpath.spec import study_from_dict
from loadpath.study import run_study


def study(material, limits):
    return {"name": f"beam-{material}", "material": material, "process": {"type": "fdm", "buildDirection": "+z"},
            "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [120, 24, 12]}],
            "preserve": [{"id": "a", "shape": {"type": "box", "min": [0, 0, 0], "max": [10, 24, 4]}},
                         {"id": "b", "shape": {"type": "box", "min": [110, 0, 0], "max": [120, 24, 4]}},
                         {"id": "pad", "shape": {"type": "box", "min": [55, 0, 9], "max": [65, 24, 12]}}],
            "supports": [{"region": "a"}, {"region": "b"}], "loads": [{"region": "pad", "force": [0, 0, -200]}],
            "goal": {"safetyFactor": 2.0}, "resolution": {"maxElements": 6000},
            "buildSim": {"cells": 8000, "release": "none", **limits}}


def row(material, name, limits, verbose=False):
    t0 = time.time()

    def prog(ev, _x):
        if verbose and ev.get("stage") in ("outcome", "printSetup", "buildSim", "verify"):
            print(round(time.time() - t0), json.dumps({k: v for k, v in ev.items() if k != "rows"}, default=float)[:300],
                  flush=True)

    res = run_study(study_from_dict(study(material, limits)), on_progress=prog)
    out = {"material": material, "name": name, "limits": limits, "seconds": round(time.time() - t0),
           "tried": [{"vf": round(q.volume_fraction, 4), "mass": round(q.report.get("mass", 0), 2),
                      "sf": q.report.get("safetyFactor"), "warp": q.report.get("warp"), "passes": q.passes}
                     for q in res.outcomes]}
    if res.best is None:
        out["error"] = "no outcome met the goal"
        out["warnings"] = res.warnings
        out["printSetup"] = next((q.report["printSetup"] for q in res.outcomes if q.report.get("printSetup")), None)
        return out
    o = res.outcomes[res.best]
    out.update(mass=o.report["mass"], safetyFactor=o.report.get("safetyFactor"), warp=o.report.get("warp"),
               volumeFraction=o.volume_fraction, buildSim=o.report.get("buildSim", {}).get("distortion"),
               material=o.report.get("material", {}).get("name"), warnings=res.warnings,
               printSetup={k: v for k, v in (o.report.get("printSetup") or {}).items() if k != "rows"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", nargs="+", default=["petg", "pa12"])
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-setup", type=float, default=None, metavar="MM",
                    help="one PA12 run at maxDistortion MM with buildSim.printSetup over PA12 and PETG")
    a = ap.parse_args()
    if a.print_setup:
        r = row("pa12", "printSetup", {"maxDistortion": a.print_setup,
                                         "printSetup": {"materials": ["pa12", "petg"]}}, verbose=True)
        print(json.dumps(r, default=float), flush=True)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(r, fh, indent=1, default=float)
        return
    rows = []
    for mat in a.materials:
        base = row(mat, "no limit", {"maxDistortion": 100.0})
        rows.append(base)
        print(json.dumps(base), flush=True)
        if base.get("warp"):
            got = base["warp"].get("delivered") or base["warp"]
            r = row(mat, f"maxDistortion {a.fraction:g}x", {"maxDistortion": round(a.fraction * got["maxDistortion"], 4)})
            rows.append(r)
            print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
