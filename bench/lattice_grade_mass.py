"""
Lattice walls graded to the stress (loadpath.lattice_grade), alone and with
the cells turned (one direction, or zones), against the uniform wall.

The parts are bench/lattice_zones_mass.py's skinless L-bracket (BCCz) and
deep cantilever (simple cubic). The cells as laid at the starting wall set
the target safety factor; each variant is sized to meet the same safety
factor. Homogenized readings (anisotropic tensors, skin junction factor).

    python bench/lattice_grade_mass.py [--cases l-bracket,deep-beam] [--cells 8000]

Writes bench/results/lattice-grade-mass.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from lattice_zones_mass import case  # noqa: E402

from loadpath.homogenized import homogenized_check  # noqa: E402
from loadpath.lattice_grade import grade_lattice  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="l-bracket,deep-beam")
    ap.add_argument("--cells", type=int, default=8000)
    ap.add_argument("--zones", type=int, default=3)
    ap.add_argument("--skin", type=float, default=0.0, help="mm of solid skin over the lattice (0: skinless)")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "lattice-grade-mass.json"))
    a = ap.parse_args()
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for name in a.cases.split(","):
        doc, sd, wall0 = case(name, a.skin)
        st = study_from_dict(sd)
        ref = homogenized_check(doc, st, cells=a.cells)
        row = {"targetSF": ref["safetyFactor"], "wall": wall0, "skin": a.skin, "massAsLaid": ref["mass"],
               "latticeExposed": ref["latticeExposed"]}
        for label, orient in (("graded", None), ("graded+one direction", "uniform"),
                              (f"graded+{a.zones} zones", a.zones)):
            t0 = time.time()
            r = grade_lattice(doc, st, target_sf=ref["safetyFactor"], cells=a.cells, orient=orient, iterations=10)
            b = r["best"]
            row[label] = {"mass": b["mass"], "safetyFactor": b["safetyFactor"], "orient": b["orient"],
                          "densityRange": b["densityRange"], "massSaved": 1 - b["mass"] / ref["mass"],
                          "seconds": time.time() - t0}
            print(f"{name:10s} skin {a.skin:g} {label:22s} {b['mass']:.2f} g (as laid {ref['mass']:.2f} g, "
                  f"-{100 * row[label]['massSaved']:.0f}%) SF {b['safetyFactor']:.3g} / {ref['safetyFactor']:.3g}, "
                  f"{row[label]['seconds']:.0f} s", flush=True)
        out[f"{name}:skin{a.skin:g}"] = row
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
