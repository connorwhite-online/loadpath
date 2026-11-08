"""
Rib layouts under a plate held at both ends with a load at the middle
(bench/ribs_plate.json): deflection against rib volume for square grids
and principal stress lines (docs/fields.md, Ribs).

Run: python bench/ribs_compare.py   (about 15 minutes on 4 cores)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.fields import FieldGraph
from loadpath.spec import study_from_dict
from loadpath.verify import verify_part

HERE = Path(__file__).resolve().parent
FACE = {"min": [-51, -31, -0.5], "max": [51, 31, 0.5]}
LAYOUTS = {
    "a single cross": {"pattern": {"type": "square", "spacing": 1000}},
    "square grid, 18 mm": {"pattern": {"type": "square", "spacing": 18}},
    "square grid, 12 mm": {"pattern": {"type": "square", "spacing": 12}},
    "stress lines along the span, 8 mm": {"stressLines": {"study": "ribs_plate.json", "part": "body", "spacing": 8,
                                                          "family": "max", "within": FACE}},
    "stress lines both ways, 12 mm": {"stressLines": {"study": "ribs_plate.json", "part": "body", "spacing": 12,
                                                      "family": "both", "within": FACE}},
}


def main():
    study = study_from_dict(json.loads((HERE / "ribs_plate.json").read_text()))
    out = {}
    for name, spec in LAYOUTS.items():
        t0 = time.time()
        doc = {"nodes": {
            "body": {"op": "shape", "shape": {"type": "box", "min": [-50, -30, 0], "max": [50, 30, 2]}},
            "ribs": {"op": "ribs", "surface": "body", "thickness": 1.5, "height": 6, "side": "outside", **spec},
            "clip": {"op": "shape", "shape": {"type": "box", "min": [-50, -30, -6], "max": [50, 30, 0.2]}},
            "part": {"op": "union", "inputs": ["body", {"op": "intersect", "inputs": ["ribs", "clip"]}]}},
            "output": "part"}
        mesh = FieldGraph(doc, base_dir=str(HERE)).mesh(pitch=0.5)
        rep = verify_part(study, mesh, max_cells=400_000)
        out[name] = {"volume": round(float(mesh.volume)), "deflection": rep["cases"][0]["deflection"],
                     "seconds": round(time.time() - t0)}
        print(name, out[name], flush=True)
    res = HERE / "results" / "ribs-plate.json"
    res.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
