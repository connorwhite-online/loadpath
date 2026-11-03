"""
Do the lattice orientation gains hold at full resolution?

bench/lattice_orient_mass.py and bench/lattice_zones_mass.py turn the cells
of skinless lattice parts (one direction for the whole lattice) and size
the turned lattice's wall down to the as-laid safety factor (or stiffness),
all on homogenized readings. Here each pair (as laid at its wall, turned at
its thinner wall) is voxelized and solved at `--pitch`
(loadpath.resolved.resolved_check; the bar rows were solved with
lattice_aniso_validate.resolved, the same reading on the grid multigrid) and the two are compared again:

- confirmed: the turned design is at least as strong (stiff) as the as-laid
  one at full resolution, so the mass it saves is real;
- the homogenized/resolved ratio of each design, the error that made or
  broke the claim.

    python bench/lattice_orient_confirm.py [--cases bccz-bar,l-bracket:safetyFactor,...] [--pitch 0.15]

Writes bench/results/lattice-orient-confirm.json.
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

import lattice_orient_mass as lom
import lattice_zones_mass as lzm

from loadpath.homogenized import homogenized_check
from loadpath.resolved import resolved_check
from loadpath.spec import study_from_dict

RES = ROOT / "bench" / "results"


ORIENT_SRC, ZONES_SRC = "lattice-orient-mass.json", "lattice-zones-mass.json"


def designs(name):
    """(doc as laid, doc turned, study dict, objective) from the mass
    benches' saved results (ORIENT_SRC, ZONES_SRC)."""
    if ":" in name:                                     # l-bracket:safetyFactor
        part, objective = name.split(":")
        row = json.loads((RES / ZONES_SRC).read_text())[name]
        doc, sd, _ = lzm.case(part)
        laid = lom.with_wall(doc, row["wall"])
        turned = lom.with_wall(doc, row["uniform"]["wall"], row["uniform"]["orient"])
        return laid, turned, sd, objective
    row = json.loads((RES / ORIENT_SRC).read_text())[name]
    doc, sd = lom.case(name)
    return (lom.with_wall(doc, row["wallAsLaid"]), lom.with_wall(doc, row["wallTurned"], row["orient"]),
            sd, "safetyFactor")


def reading(rep_cases, key):
    return max(c[key] for c in rep_cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="bccz-bar,cubic-twist,octet-pull,l-bracket:safetyFactor,l-bracket:stiffness")
    ap.add_argument("--pitch", type=float, default=0.15)
    ap.add_argument("--cells", type=int, default=30000)
    ap.add_argument("--out", default=str(RES / "lattice-orient-confirm.json"))
    ap.add_argument("--orient-source", default="lattice-orient-mass.json", help="results of bench/lattice_orient_mass.py")
    ap.add_argument("--zones-source", default="lattice-zones-mass.json", help="results of bench/lattice_zones_mass.py")
    a = ap.parse_args()
    global ORIENT_SRC, ZONES_SRC
    ORIENT_SRC, ZONES_SRC = a.orient_source, a.zones_source
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for name in a.cases.split(","):
        laid, turned, sd, objective = designs(name)
        st = study_from_dict(sd)
        cell = float(laid["nodes"]["lat"]["cell"])
        row = {"objective": objective}
        for tag, doc in (("asLaid", laid), ("turned", turned)):
            t0 = time.time()
            h = homogenized_check(copy.deepcopy(doc), st, cells=a.cells, contact_band=cell)
            r = resolved_check(copy.deepcopy(doc), st, pitch=a.pitch, contact_band=cell, cell_size=cell)
            row[tag] = {"wall": doc["nodes"]["lat"]["thickness"], "orient": doc["nodes"]["lat"].get("orient"),
                        "homMass": h["mass"], "resMass": r["mass"],
                        "homPeak": reading(h["cases"], "peakStress"), "resPeak": reading(r["cases"], "peakStress"),
                        "homCompliance": reading(h["cases"], "compliance"),
                        "resCompliance": reading(r["cases"], "compliance"),
                        "latticeTurnedExposed": h.get("latticeTurnedExposed"),
                        "resCases": r["cases"], "homCases": [{k: c[k] for k in ("name", "compliance", "peakStress")}
                                                             for c in h["cases"]],
                        "seconds": time.time() - t0}
            print(f"{name:24s} {tag:7s} wall {row[tag]['wall']:.3f}: hom/res compliance "
                  f"{row[tag]['homCompliance'] / row[tag]['resCompliance']:.2f} peak "
                  f"{row[tag]['homPeak'] / row[tag]['resPeak']:.2f} ({row[tag]['seconds']:.0f} s)", flush=True)
        L, T = row["asLaid"], row["turned"]
        key = "Peak" if objective == "safetyFactor" else "Compliance"
        row["homTurnedOverLaid"] = T["hom" + key] / L["hom" + key]
        row["resTurnedOverLaid"] = T["res" + key] / L["res" + key]
        row["massSavedClaimed"] = 1.0 - T["homMass"] / L["homMass"]
        row["confirmed"] = bool(row["resTurnedOverLaid"] <= 1.02)
        print(f"{name:24s} turned/laid {key.lower()}: hom {row['homTurnedOverLaid']:.2f} res "
              f"{row['resTurnedOverLaid']:.2f}; claimed -{100 * row['massSavedClaimed']:.0f}% "
              f"{'CONFIRMED' if row['confirmed'] else 'NOT confirmed'}", flush=True)
        out[name] = row
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
