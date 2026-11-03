"""
How much mass turning the lattice cells saves (loadpath.lattice_orient).

Lattice-only bars between solid end blocks (bench/lattice_aniso_validate.py's
`bar`), each under one kind of load. For each: the cells as they lie at a
2 mm wall give the target safety factor; optimize_orientation turns the
cells at that wall; then the wall is bisected until the turned lattice
meets the same safety factor, and the two masses are compared. All
readings are homogenized (the anisotropic tensors); the tensors themselves
are validated in bench/results/lattice-aniso*.json.

    python bench/lattice_orient_mass.py [--cases bccz-bar,...] [--cells 8000]

Writes bench/results/lattice-orient-mass.json.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from lattice_aniso_validate import bar  # noqa: E402

from loadpath.homogenized import homogenized_check  # noqa: E402
from loadpath.lattice_orient import optimize_orientation  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def _cases(study, names):
    s = copy.deepcopy(study)
    s["loadCases"] = [c for c in s["loadCases"] if c["name"] in names]
    return s


def case(name):
    if name == "bccz-bar":            # BCCz, its stiff z struts across the bar; bend + pull
        doc, st = bar("bccz", 2.0)
        return doc, _cases(st, ("bend", "pull"))
    if name == "cubic-twist":         # simple cubic rods along the bar, twisted: shear is its weak mode
        doc, st = bar("cubic", 2.0)
        return doc, _cases(st, ("twist",))
    if name == "octet-pull":          # octet along its cube axis (its soft direction) in a pull
        doc, st = bar("octet", 2.0)
        return doc, _cases(st, ("pull",))
    if name == "cubic-bar":           # already aligned: should gain nothing
        doc, st = bar("cubic", 2.0)
        return doc, _cases(st, ("bend", "pull"))
    raise KeyError(name)


def with_wall(doc, t, orient=None):
    d = copy.deepcopy(doc)
    d["nodes"]["lat"]["thickness"] = float(t)
    if orient is not None and any(abs(x) > 1e-9 for x in orient):
        d["nodes"]["lat"]["orient"] = list(orient)
    return d


def wall_for(doc, st, target, cells, orient, lo=0.4, hi=2.0, steps=9):
    """Thinnest wall (bisection) whose worst-case SF meets `target`."""
    rep_hi = homogenized_check(with_wall(doc, hi, orient), st, cells=cells)
    if rep_hi["safetyFactor"] < target:
        return hi, rep_hi
    best = (hi, rep_hi)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        rep = homogenized_check(with_wall(doc, mid, orient), st, cells=cells)
        if rep["safetyFactor"] >= target:
            hi, best = mid, (mid, rep)
        else:
            lo = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="bccz-bar,cubic-twist,octet-pull,cubic-bar")
    ap.add_argument("--cells", type=int, default=8000)
    ap.add_argument("--confirm", default="none", help="optimize_orientation's bare-face guard: none (read the "
                                                     "homogenized gain, the point of this bench) or auto")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "lattice-orient-mass.json"))
    a = ap.parse_args()
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for name in a.cases.split(","):
        t0 = time.time()
        doc, sd = case(name)
        st = study_from_dict(sd)
        ref = homogenized_check(doc, st, cells=a.cells)
        target = ref["safetyFactor"]
        opt = optimize_orientation(doc, st, cells=a.cells, refine=16, full_checks=3, confirm=a.confirm)
        wall, rep = wall_for(doc, st, target, a.cells, opt["orient"])
        row = {"targetSF": target, "wallAsLaid": 2.0, "massAsLaid": ref["mass"],
               "orient": opt["orient"], "sfTurnedAtSameWall": opt["best"]["safetyFactor"],
               "complianceAsLaid": opt["unrotated"]["compliance"], "complianceTurned": opt["best"]["compliance"],
               "wallTurned": wall, "massTurned": rep["mass"], "sfTurned": rep["safetyFactor"],
               "massSaved": 1.0 - rep["mass"] / ref["mass"], "seconds": time.time() - t0}
        out[name] = row
        print(f"{name:12s} SF {target:.3g}: as laid {ref['mass']:.2f} g (2.0 mm); turned {opt['orient']} "
              f"SF {row['sfTurnedAtSameWall']:.3g} at 2.0 mm, {rep['mass']:.2f} g at {wall:.2f} mm "
              f"(-{100 * row['massSaved']:.0f}%), {row['seconds']:.0f} s", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
