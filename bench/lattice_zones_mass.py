"""
Cell direction that follows the stress (orient.zones) against one direction
for the whole lattice (loadpath.lattice_orient).

Lattice-only parts between solid blocks, skinless so the lattice carries
everything:

- l-bracket: BCCz (stiff along its z struts), an arm along x from the
  support and an arm up z to the load; one direction cannot serve both;
- deep-beam: simple cubic cantilever 60 x 12 x 24 mm, axial stress at the
  top and bottom, shear through the middle.

For each: the cells as laid at the starting wall set the target safety
factor; then the best single orient (optimize_orientation) and the best
zoned one (optimize_zones) at that wall; then the wall is bisected for each
until it meets the target, and the masses are compared. Homogenized
readings throughout (anisotropic tensors).

    python bench/lattice_zones_mass.py [--cases l-bracket,deep-beam] [--cells 8000] [--zones 3]

Writes bench/results/lattice-zones-mass.json.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.homogenized import homogenized_check  # noqa: E402
from loadpath.lattice_orient import optimize_orientation, optimize_zones  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402

MAT = {"E": 1700.0, "nu": 0.35, "strength": 48.0, "density": 1.01}


def _box(lo, hi):
    return {"type": "box", "min": list(lo), "max": list(hi)}


def _part(kind, wall, cell, solid, blocks, origin):
    nodes = {"lat": {"op": "lattice", "type": kind, "cell": cell, "thickness": wall, "origin": origin},
             "space": {"op": "shape", "shape": {"type": "union", "shapes": solid}},
             "fill": {"op": "intersect", "inputs": ["space", "lat"]},
             "ends": {"op": "shape", "shape": {"type": "union", "shapes": blocks}},
             "part": {"op": "union", "inputs": ["fill", "ends"]}}
    return {"name": kind, "nodes": nodes, "output": "part"}


def case(name):
    if name == "l-bracket":
        arm_x, arm_z = _box((0, -6, -6), (60, 6, 6)), _box((48, -6, -6), (60, 6, 54))
        root, tip = _box((0, -6, -6), (4, 6, 6)), _box((48, -6, 50), (60, 6, 54))
        doc = _part("bccz", 1.4, 6.0, [arm_x, arm_z], [root, tip], [0, -6, -6])
        study = {"name": "l", "designSpace": [arm_x, arm_z],
                 "preserve": [{"id": "root", "shape": root}, {"id": "tip", "shape": tip}],
                 "supports": [{"region": "root"}],
                 "loadCases": [{"name": "down", "loads": [{"region": "tip", "force": [0, 0, -60]}]}],
                 "material": MAT, "goal": {"safetyFactor": 2}}
        return doc, study, 1.4
    if name == "deep-beam":
        space = _box((0, -6, -12), (60, 6, 12))
        root, tip = _box((0, -6, -12), (4, 6, 12)), _box((56, -6, -12), (60, 6, 12))
        doc = _part("cubic", 1.6, 6.0, [space], [root, tip], [0, -6, -12])
        study = {"name": "beam", "designSpace": [space],
                 "preserve": [{"id": "root", "shape": root}, {"id": "tip", "shape": tip}],
                 "supports": [{"region": "root"}],
                 "loadCases": [{"name": "bend", "loads": [{"region": "tip", "force": [0, 0, -100]}]}],
                 "material": MAT, "goal": {"safetyFactor": 2}}
        return doc, study, 1.6
    raise KeyError(name)


def with_wall(doc, t):
    d = copy.deepcopy(doc)
    d["nodes"]["lat"]["thickness"] = float(t)
    return d


def _meets(rep, objective, target):
    if objective == "safetyFactor":
        return rep["safetyFactor"] >= target
    return sum(c["compliance"] for c in rep["cases"]) <= target


def wall_for(doc, st, target, cells, lo, hi, objective="safetyFactor", steps=8):
    rep_hi = homogenized_check(with_wall(doc, hi), st, cells=cells)
    if not _meets(rep_hi, objective, target):
        return hi, rep_hi
    best = (hi, rep_hi)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        rep = homogenized_check(with_wall(doc, mid), st, cells=cells)
        if _meets(rep, objective, target):
            hi, best = mid, (mid, rep)
        else:
            lo = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="l-bracket,deep-beam")
    ap.add_argument("--cells", type=int, default=8000)
    ap.add_argument("--zones", type=int, default=3)
    ap.add_argument("--objective", choices=["safetyFactor", "stiffness"], default="safetyFactor",
                    help="what the orientation and the wall search hold: the safety factor, or the compliance")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "lattice-zones-mass.json"))
    a = ap.parse_args()
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for name in a.cases.split(","):
        t0 = time.time()
        doc, sd, wall0 = case(name)
        st = study_from_dict(sd)
        ob = a.objective
        ref = homogenized_check(doc, st, cells=a.cells)
        key = "safetyFactor" if ob == "safetyFactor" else "compliance"
        target = ref["safetyFactor"] if ob == "safetyFactor" else sum(c["compliance"] for c in ref["cases"])
        uni = optimize_orientation(doc, st, cells=a.cells, refine=12, full_checks=3, objective=ob)
        zon = optimize_zones(doc, st, zones=a.zones, cells=a.cells, objective=ob, start=uni["orient"])
        w_u, r_u = wall_for(uni["doc"], st, target, a.cells, 0.3 * wall0, wall0, ob)
        w_z, r_z = wall_for(zon["doc"], st, target, a.cells, 0.3 * wall0, wall0, ob)
        row = {"objective": ob, "target": target, "wall": wall0, "massAsLaid": ref["mass"],
               "uniform": {"orient": uni["orient"], "atWall": uni["best"][key], "wall": w_u,
                           "mass": r_u["mass"], "massSaved": 1 - r_u["mass"] / ref["mass"]},
               "zones": {"zones": zon["zones"], "atWall": zon["best"][key], "wall": w_z,
                         "mass": r_z["mass"], "massSaved": 1 - r_z["mass"] / ref["mass"]},
               "seconds": time.time() - t0}
        out[f"{name}:{ob}"] = row
        print(f"{name:10s} {key} {target:.4g} as laid {ref['mass']:.2f} g | uniform {uni['best'][key]:.4g}"
              f" -> {r_u['mass']:.2f} g ({w_u:.2f} mm, -{100 * row['uniform']['massSaved']:.0f}%) | zones "
              f"{zon['best'][key]:.4g} -> {r_z['mass']:.2f} g ({w_z:.2f} mm, "
              f"-{100 * row['zones']['massSaved']:.0f}%) | {row['seconds']:.0f} s", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
