"""
Strut and conformal lattices against the gyroid, on the arm link: the graded
arm-link graph (arm-link-graded.json: stress-graded skin, collars round the
bearings, a stress-graded gyroid core) with its core swapped for each strut
cell, every variant's lattice thickness scaled until it weighs the same as
the gyroid version, then analyzed with the lattice homogenized under the
arm-link study's loads. Equal mass, so safety factor and deflection say
which core earns its grams.

    python examples/fields/strut_lattices.py [--out out/] [--cells 30000] [--mesh octet]

Prints a table (mass, safety factor, deflection, FDM overhangs) and writes
summary.json; --mesh NAME also meshes that variant (STL + report).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from loadpath.editor import _Resources  # noqa: E402
from loadpath.homogenized import homogenized_check  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.struts import is_strut, lattice_printability  # noqa: E402
from loadpath.tasks import example  # noqa: E402

# name -> lattice node (its thickness is the stress-graded "wall" node, its
# range scaled to the gyroid's mass)
VARIANTS = {
    "gyroid": {"op": "lattice", "type": "gyroid", "cell": "cell", "thickness": "wall"},
    "octet": {"op": "lattice", "type": "octet", "cell": "cell", "thickness": "wall", "blend": 0.4},
    "kelvin": {"op": "lattice", "type": "kelvin", "cell": "cell", "thickness": "wall", "blend": 0.4},
    "bcc": {"op": "lattice", "type": "bcc", "cell": "cell", "thickness": "wall", "blend": 0.4},
    "fcc": {"op": "lattice", "type": "fcc", "cell": "cell", "thickness": "wall", "blend": 0.4},
    "voronoi": {"op": "lattice", "type": "voronoi", "cell": 5.5, "thickness": "wall", "blend": 0.4},
    "octet-conformal": {"op": "lattice", "type": "octet", "cell": 6, "thickness": "wall", "blend": 0.4,
                        "frame": {"type": "conformal", "surface": "envelope", "axis": "x"}},
    "gyroid-conformal": {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": "wall",
                         "frame": {"type": "conformal", "surface": "envelope", "axis": "x"}},
}
# strut diameters are thicker than sheet walls for the same mass: start the
# search from a wall range that roughly matches
START = {"gyroid": 1.0, "voronoi": 1.6}


def variant(doc, name, scale):
    d = copy.deepcopy(doc)
    d["nodes"]["core"] = copy.deepcopy(VARIANTS[name])
    lo, hi = doc["nodes"]["wall"]["to"]
    d["nodes"]["wall"]["to"] = [lo * scale, hi * scale]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="strut-lattices-out")
    ap.add_argument("--cells", type=int, default=30_000)
    ap.add_argument("--only", help="comma-separated variants")
    ap.add_argument("--mesh", help="mesh this variant at the matched scale")
    ap.add_argument("--pitch", type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    doc = json.loads((HERE / "arm-link-graded.json").read_text())
    study = study_from_dict(example("arm-link"))
    res = _Resources(cap=64)
    cache = {}

    def check(name, scale):
        key = (name, round(scale, 4))
        if key not in cache:
            d = variant(doc, name, scale)
            g = res.graph(d, str(HERE))
            t0 = time.time()
            r = homogenized_check(d, study, cells=a.cells, graph=g, base_dir=str(HERE))
            res.keep(g, d)
            cache[key] = (r, g, time.time() - t0)
        return cache[key]

    target = check("gyroid", 1.0)[0]["mass"]
    print(f"target mass (graded gyroid, homogenized): {target:.1f} g", flush=True)
    rows = {}
    names = a.only.split(",") if a.only else list(VARIANTS)
    for name in names:
        # mass is monotone in the scale: secant/bisection to within 1%
        s0 = START.get(name, 1.4)
        lo_s, hi_s = None, None
        s = s0
        for _ in range(9):
            r = check(name, s)[0]
            err = r["mass"] / target - 1
            if abs(err) < 0.01:
                break
            if err < 0:
                lo_s = s
            else:
                hi_s = s
            if lo_s is not None and hi_s is not None:
                s = 0.5 * (lo_s + hi_s)
            else:
                s = s * (1 + err) ** -1.5
        r, g, sec = check(name, s)
        row = {"scale": round(s, 3), "wall": [round(x * s, 2) for x in doc["nodes"]["wall"]["to"]],
               "mass_g": round(r["mass"], 1), "safetyFactor": round(r["safetyFactor"], 2),
               "deflection_mm": round(max(c["deflection"] for c in r["cases"]), 3),
               "peakInLattice": any(c["inLattice"] for c in r["cases"]),
               "latticeDensity": [round(x, 3) for x in r["latticeDensity"]] if r["latticeDensity"] else None,
               "seconds": round(sec)}
        if is_strut(VARIANTS[name]["type"]):
            try:
                p = lattice_printability(g, "core", solid="envelope")
                row["fdmUnsupported"] = round(p["unsupportedFraction"], 2)
                row["fdmLongestBridge_mm"] = round(p["longestBridge"], 1)
            except Exception as exc:  # noqa: BLE001
                row["fdmUnsupported"] = str(exc)
        rows[name] = row
        print(name, row, flush=True)
        with open(os.path.join(a.out, "summary.json"), "w") as fh:
            json.dump({"targetMass": target, "variants": rows}, fh, indent=2)
    if a.mesh:
        from loadpath.fields import run_field

        s = rows[a.mesh]["scale"] if a.mesh in rows else 1.0
        d = variant(doc, a.mesh, s)
        d["name"] = f"arm-link-{a.mesh}"
        d["mesh"] = {"pitch": a.pitch}
        _, rep = run_field(d, out_dir=a.out, base_dir=str(HERE), render=False,
                           prebuilt=res.prebuilt(d))
        print(json.dumps({k: rep.get(k) for k in ("faces", "watertight", "mass", "bodies", "lattices")},
                         indent=1, default=str))


if __name__ == "__main__":
    main()
