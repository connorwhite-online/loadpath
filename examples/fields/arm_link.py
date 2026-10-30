"""
The arm link as a field-driven part, checked against the solid it was graded
from: build the graph, mesh the graded part and its solid envelope, and verify
both under the arm-link example's loads.

    python examples/fields/arm_link.py [--out out/] [--pitch 0.3]

Also builds the same part with a uniform skin, wall and cell (--uniform
skin,wall, picked to weigh about the same as the graded one), which is the
comparison that says whether the grading earned anything. Prints mass,
safety factor and deflection for each, and writes the STLs.
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

from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.tasks import example  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="arm-link-fields-out")
    ap.add_argument("--pitch", type=float)
    ap.add_argument("--cells", type=int, default=150_000)
    ap.add_argument("--uniform", default="1.4,0.9", help="skin,wall (mm) of the uniform comparison")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    doc = json.loads((HERE / "arm-link-graded.json").read_text())
    study = study_from_dict(example("arm-link"))
    g = FieldGraph(doc, base_dir=str(HERE))
    skin, wall = (float(x) for x in a.uniform.split(","))
    u = copy.deepcopy(doc)
    u["nodes"].update(skinWall={"op": "constant", "value": skin}, wall={"op": "constant", "value": wall},
                      cell={"op": "constant", "value": 5.5})
    del u["nodes"]["stress"]
    gu = FieldGraph(u, base_dir=str(HERE))
    rows = {}
    for name, graph, node, pitch in (("solid", g, "solid", 0.4), ("graded", g, "part", a.pitch),
                                     ("uniform", gu, "part", a.pitch)):
        t0 = time.time()
        mesh = graph.mesh(pitch=pitch, name=node)
        mesh.export(os.path.join(a.out, f"arm-link-{name}.stl"))
        rep = verify_part(study, mesh, max_cells=a.cells)
        rows[name] = {"mass_g": round(float(mesh.volume) * study.material.density / 1000, 1),
                      "safetyFactor": round(rep["safetyFactor"], 2),
                      "deflection_mm": round(max(c["deflection"] for c in rep["cases"]), 3),
                      "faces": len(mesh.faces), "seconds": round(time.time() - t0)}
        print(name, rows[name], flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
