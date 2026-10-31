"""Thermal expansion in the design loop: the motor-mount eval case (PETG
NEMA 17 bracket) with the motor face at 60 C and the two M4 bolts into an
aluminium frame at 25 C. The PETG between the bolts and round the hot
motor face wants to grow ~0.1%, and the bolts stop it.

Optimizes twice at the same volume, with the design's own temperature
field softening the PETG both times:
  - without expansion (thermal.optimize_thermomechanical as before),
  - with expansion (alpha=True),
then rates both under the belt loads plus expansion (thermal.thermoelastic).
Run: python bench/thermal_expansion.py [maxElements]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from loadpath import thermal as T  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.shapes import Box, shape_from_spec  # noqa: E402
from loadpath.simp import filter_radius  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def main(max_el=12000, iters=60):
    doc = json.loads((ROOT / "loadpath/evals/cases/motor-mount.json").read_text())
    doc["resolution"] = {"maxElements": max_el}
    doc.pop("symmetry", None)
    st = study_from_dict(doc, base_dir=str(ROOT / "loadpath/evals/cases"))
    d = build_domain(st)
    mesh = d.mesh
    face = shape_from_spec({"type": "cylinder", "p0": [0, -1, 35], "p1": [0, 1.0, 35], "r": 21})
    frame = Box([-25, -1, -1], [25, 51, 0.6])
    bc = T.thermal_bc(mesh, ambient=25, temperatures=[(face, 60.0), (frame, 25.0)],
                      convection=[T.Convection(10.0)])
    opts = T.ThermalOptions(volume_fraction=0.3, filter_radius=filter_radius(d, 3.0), k=T.conductivity("petg"),
                            max_iterations=iters, erode=0.15, target=mesh.node_xyz[:, 1] > 5)
    out = {}
    for name, alpha in (("softening only", None), ("with expansion", True)):
        t0 = time.time()
        r = T.optimize_thermomechanical(d, 0.38, bc, opts, material="petg", alpha=alpha)
        te = r.temperature.elem_temperature
        rate = {k: T.thermoelastic(d, 0.38, r.xphys, te, "petg", alpha=a, t_ref=25.0)
                for k, a in (("belts", None), ("belts+expansion", True))}
        out[name] = {"seconds": round(time.time() - t0, 1), "volume": round(r.volume_fraction, 3),
                     "maxT": round(float(r.temperature.temperature.max()), 1),
                     **{f"{k} strain energy x2 Nmm": round(float(np.sum(v["strainEnergy2"])), 4) for k, v in rate.items()},
                     **{f"{k} max vM MPa": round(v["maxVonMises"], 2) for k, v in rate.items()},
                     **{f"{k} max disp mm": round(v["maxDisplacement"], 4) for k, v in rate.items()}}
        print(name, json.dumps(out[name]), flush=True)
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12000)
