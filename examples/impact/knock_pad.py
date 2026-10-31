"""
Small knocks: a TPU bumper (knock-pad.json) for a GRIP-like forearm, 0.75 J
under a 140 N limit. TPU buckles and springs back, so impact.py uses the
elastic crush law for it (crush_law). Compares the graded core with a
uniform core and a solid TPU pad of the same mass and footprint.

    python examples/impact/knock_pad.py [--out knock-out/]
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.impact import impact_report, solid_impact  # noqa: E402
from loadpath.materials import material_from_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="knock-out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    doc = json.loads((HERE / "knock-pad.json").read_text())
    mat = material_from_spec(doc["material"])
    spec = doc["impact"]
    first = impact_report(FieldGraph(doc), spec, mat)
    gdoc = copy.deepcopy(doc)
    gdoc["nodes"]["density"] = first["graded"]["density"]
    gg = FieldGraph(gdoc)
    rg = impact_report(gg, {**spec, "grade": False}, mat)
    udoc = copy.deepcopy(doc)
    udoc["nodes"]["density"] = {"op": "constant", "value": round(rg["meanDensity"], 4)}
    ru = impact_report(FieldGraph(udoc), {**spec, "grade": False}, mat)
    rows = {"graded TPU gyroid": {"mass_g": rg["coreMass"], **rg["response"], "rebound_mJ": rg["reboundEnergy"]},
            "uniform TPU gyroid": {"mass_g": ru["coreMass"], "density": round(rg["meanDensity"], 4),
                                   **ru["response"]}}
    # a solid TPU pad of the same mass on the same 40 mm footprint: k = E A^2 / V
    area, vol = 40.0 * 40.0, rg["coreMass"] / mat.density * 1000.0
    rows["solid TPU pad, same mass"] = {"mass_g": rg["coreMass"], "thickness_mm": round(vol / area, 2),
                                        **solid_impact(mat.E * area ** 2 / vol, 1.0 / area, mat.strength,
                                                       rg["energy"], spec["forceLimit"], spec["mass"])}
    gg.mesh(pitch=doc["mesh"]["pitch"]).export(os.path.join(a.out, "knock-pad-graded.stl"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"rows": rows, "curves": {"graded TPU gyroid": rg["curve"], "uniform TPU gyroid": ru["curve"]},
                   "gradedDensity": first["graded"]["density"]}, fh, indent=1)
    for k, v in rows.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
