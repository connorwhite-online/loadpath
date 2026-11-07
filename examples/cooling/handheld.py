"""
Cooling inside a fixed exterior: a handheld device around its electronics.

    python examples/cooling/handheld.py --out hh-out [--pitches 1.0 --iterations 80] [--baseline] [--vents]

A generic handheld, 120 x 64 x 16 mm with rounded edges, printed in MJF
PA 12. The exterior is industrial design, kept as a 1.5 mm skin. Inside
are a board at one end with a processor (1.5 W, its case kept under 70 C)
and a power chip (0.3 W, under 70 C) on it, and a battery beside the board,
which conducts but must stay under 45 C (the usual charging limit for
lithium cells). The
device is held in the hand while it works, so no part of the outside may
pass 48 C (plastic held 10 min to 8 h, after IEC 62368-1), in a 25 C room.
`--press` adds a structural target designed with the cooling: the back
panel under a 20 N thumb press closes by no more than 0.1 mm.

Everything else inside is design space. Bare, the parts sit in still air:
the processor runs far over its limit and the skin over it is the hot spot.
Filled solid, the skin over the processor still gets too hot. Loadpath
grows pads that take the heat off the processor and spread it under the
skin, with the least material.

`--vents` adds two slots the exterior already has, at the bottom and top
ends (held upright, gravity along -y), and reads what the stack flow
through them carries. `--baseline` reads the bare skin and a solid-filled
interior against the same limits before anything is optimized.
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

L, W, T, R = 120.0, 64.0, 16.0, 7.0      # mm; corner radius


def exterior(path: str) -> str:
    """A rounded slab: the convex hull of eight corner spheres."""
    import trimesh
    s = trimesh.creation.icosphere(subdivisions=4, radius=R)
    pts = []
    for x in (-L / 2 + R, L / 2 - R):
        for y in (-W / 2 + R, W / 2 - R):
            for z in (-T / 2 + R, T / 2 - R):
                pts.append(s.vertices + [x, y, z])
    hull = trimesh.convex.convex_hull(np.vstack(pts))
    hull.export(path)
    return path


def study(ext: str, pitch: float = 1.0, vents: bool = False, press: bool = False) -> dict:
    doc = {
        "name": "handheld",
        "description": "Handheld device (MJF PA 12): exterior kept as a 1.5 mm skin; interior designed to carry "
                       "the processor's heat out and spread it under the skin.",
        "designSpace": {"type": "mesh", "path": ext},
        "skin": {"thickness": 1.5},
        "keepOut": [
            {"id": "board", "bore": False, "shape": {"type": "box", "min": [2, -26, -2.0], "max": [54, 26, -0.4]}},
            {"id": "soc", "bore": False, "shape": {"type": "box", "min": [22, -6, -0.4], "max": [34, 6, 0.8]}},
            {"id": "pmic", "bore": False, "shape": {"type": "box", "min": [8, 14, -0.4], "max": [14, 20, 0.6]}},
            {"id": "battery", "bore": False, "shape": {"type": "box", "min": [-52, -24, -4.5], "max": [-4, 24, 4.5]}},
        ],
        "material": "pa12",
        "process": "mjf",
        "minMember": 1.5,
        "resolution": {"pitch": pitch},
        "cooling": {
            "ambient": 25,
            "components": [
                {"region": "soc", "watts": 1.5, "maxTemp": 70},
                {"region": "pmic", "watts": 0.3, "maxTemp": 70},
                {"region": "board", "conductivity": 20},
                {"region": "battery", "conductivity": 1.5, "maxTemp": 45},
            ],
            "skin": {"contact": "held"},
            "iterations": 80,
        },
    }
    if press:
        # the back panel under a thumb: 20 N in the middle of the +z face,
        # held on the -z face, closes it by no more than 0.1 mm
        pad = {"type": "box", "min": [-15, -12, 0], "max": [15, 12, 10]}
        doc["feel"] = {"targets": [
            {"kind": "squeeze", "name": "panel", "hold": {"face": "-z"}, "press": {"face": "+z", "shape": pad},
             "direction": "-z", "force": 20, "travel": {"max": 0.1}}]}
    if vents:
        doc["cooling"]["gravity"] = "-y"
        doc["cooling"]["vents"] = [
            {"id": "intake", "shape": {"type": "box", "min": [-30, -33, -3], "max": [30, -30, 3]}, "area": 120},
            {"id": "exhaust", "shape": {"type": "box", "min": [-30, 30, -3], "max": [30, 33, 3]}, "area": 120},
        ]
    return doc


def _row(r):
    comps = ", ".join(f"{c['component']} {c['temperature']:.1f} C" + ("" if c["maxTemp"] is None else
                      f" (<= {c['maxTemp']:g})") for c in r["components"])
    s = r.get("skin", {})
    return f"{comps}; skin hot spot {s.get('temperature', float('nan')):.1f} C (<= {s.get('maxTemp')}), mean " \
           f"{s.get('mean', float('nan')):.1f}; heat out {r['heatOut']}; {'ok' if r['ok'] else 'MISS'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hh-out")
    ap.add_argument("--pitches", default="1.0")
    ap.add_argument("--iterations", default=None)
    ap.add_argument("--verify", default=None, help="comma-separated mm grids for the delivered read-back")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--vents", action="store_true")
    ap.add_argument("--press", action="store_true", help="also a feel target: the back panel under a 20 N "
                                                         "thumb press closes by no more than 0.1 mm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from loadpath.cooling import cooling_readings, run_cooling
    from loadpath.domain import build_domain
    from loadpath.spec import study_from_dict
    from loadpath.switch import _thin_skin

    pitches = [float(p) for p in a.pitches.split(",")]
    ext = exterior(os.path.abspath(os.path.join(a.out, "exterior.stl")))
    doc = study(ext, pitches[0], a.vents, a.press)
    json.dump(doc, open(os.path.join(a.out, "handheld.study.json"), "w"), indent=1)
    st = study_from_dict(doc)
    if a.baseline:
        rows = {}
        for p in pitches:
            dom, floor = _thin_skin(st, build_domain(st, p))
            for name, x in (("bare skin", np.maximum(dom.preserve.astype(float), floor)),
                            ("filled", np.ones(dom.n_elem))):
                r = cooling_readings(st, dom, np.maximum(x, 1e-9), floor=floor)
                r = {k: v for k, v in r.items() if not k.startswith("_")}
                rows[f"{name} @ {p:g}"] = r
                print(f"{name} @ {p:g} mm:", _row(r))
        json.dump(rows, open(os.path.join(a.out, "baseline.json"), "w"), indent=1, default=float)
        return
    its = [int(v) for v in a.iterations.split(",")] if a.iterations else None
    vps = [float(v) for v in a.verify.split(",")] if a.verify else None
    t0 = time.time()
    rep = run_cooling(st, a.out, pitches=pitches, iterations=its, verify_pitches=vps,
                      on_progress=lambda ev: print(json.dumps({k: v for k, v in ev.items() if k != "forces"}),
                                                   flush=True))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    c = rep["cooling"]
    print("design:     ", _row(c["design"]))
    print("thresholded:", _row(c["thresholded"]))
    for d in c.get("delivered", []):
        print(f"delivered @ {d['pitch']:.3g} mm:", _row(d))
    if "gridChange" in c:
        print("grid change between the two delivered reads:", c["gridChange"])
    for r in rep.get("feel", []):
        print(r["name"], [(d["direction"], round(d.get("travel", 0), 3)) for d in r["directions"]],
              "ok" if r["ok"] else "MISS")
    print(f"mass {rep['mass']:.1f} g, volume share {rep['volume']:.3f}, watertight {rep['watertight']}, "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
