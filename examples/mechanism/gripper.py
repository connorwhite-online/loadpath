"""
A TPU compliant gripper jaw, designed by topology optimization and checked
the way it will be used: driven by a servo's stroke, with large
deflections, against a soft workpiece, and over time (TPU relaxes).

    python examples/mechanism/gripper.py [--out gripper-out/] [--thickness 8]

The layout is the textbook half gripper (loadpath.benchmarks.mechanisms,
after Sigmund 1997) at 1 mm per cell: 60 x 30 mm, the symmetry line along
the bottom, a servo pushing at the bottom-left corner, clamped at the top
left, the jaw at the top of a 10 x 6 mm gap at the bottom right. Printed
flat in TPU 95A, `--thickness` mm thick (forces scale with it).

Three designs at the same volume (30%):

- the textbook code's (Sigmund's 99-line mechanism code, ported);
- loadpath (mechanism.py, MMA, density filter r = 2 mm, projection);
- loadpath robust (eroded / blueprint / dilated, no one-node hinges).

All three are optimized for the same springs (a stiff servo at the input,
a soft workpiece at the jaw), then driven through the servo's stroke with
the nonlinear solver (nonlinear.stroke_mechanism): free jaw travel, and the
grip force on a workpiece, with the actuator force it takes and the peak
strain. TPU's time dependence (viscoelastic.py) turns the grip force into
a curve: held at a fixed stroke the force relaxes, and a fast close takes
more actuator force.
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

from loadpath.benchmarks.mechanisms import (MechEvaluator, gripper, hinges, mech_domain,  # noqa: E402
                                            optimize_reference_mechanism)
from loadpath.materials import material_from_spec  # noqa: E402
from loadpath.mechanism import MechanismOptions, optimize_mechanism  # noqa: E402
from loadpath.nonlinear import stroke_mechanism  # noqa: E402
from loadpath.viscoelastic import DAY, event_factor, modulus_factor  # noqa: E402

NU = 0.45
K_IN = 50.0       # N/mm per mm of plate: a servo through a stiff horn
K_OUT = 0.25      # N/mm per mm of plate: a soft workpiece (2 N/mm for an 8 mm plate)
STROKE = 3.0      # mm of servo travel at the input
CLOSE_TIME = 0.2  # s, a quick close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gripper-out")
    ap.add_argument("--thickness", type=float, default=8.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    mat = material_from_spec("tpu")
    E = float(mat.E)
    p = gripper()
    p.k_in, p.k_out = K_IN / E, K_OUT / E      # the benchmark evaluator works at unit modulus
    dom = mech_domain(p)
    n, pres = dom.mesh.n_elem, int(dom.preserve.sum())
    vf = (p.volfrac * np.prod(p.dims) - pres) / (n - pres)

    designs = {}
    ref = optimize_reference_mechanism(p)
    g = ref["x"]
    designs["textbook (99-line mechanism code)"] = g[tuple(dom.mesh.elem_ijk.T)]
    for name, kw in (("loadpath", {}), ("loadpath robust", {"robust": 0.15, "max_iterations": 200})):
        r = optimize_mechanism(dom, NU, MechanismOptions(volume_fraction=vf, filter_radius=2.0, modulus=E, k_in=K_IN,
                                                         k_out=K_OUT, extrusion=1, **kw))
        designs[name] = r.xphys
        print(name, r.iterations, "iterations", flush=True)

    ev = MechEvaluator(p)
    rows, grids = {}, {}
    T = a.thickness
    for name, x in designs.items():
        grid = np.zeros(dom.dims)
        grid[tuple(dom.mesh.elem_ijk.T)] = x
        # black and white at the same volume, as printed
        from loadpath.benchmarks.evaluate import binarize
        cx = ev.cells(grid)
        bw = np.zeros(p.dims)
        bw[ev.cell] = binarize(cx, ev.active, p.volfrac * cx.size / ev.active.sum())
        grids[name] = bw
        xb = bw[tuple(dom.mesh.elem_ijk.T)]
        row = {"hinges": hinges(bw), "linearJawPerForce_bw": ev.output(bw) / E}
        free = stroke_mechanism(dom, xb, E, NU, STROKE, k_out=0.0, steps=8)
        grip = stroke_mechanism(dom, xb, E, NU, STROKE, k_out=K_OUT, steps=8)
        row["free"] = {k: free[k] for k in ("converged", "strokeReached", "output", "linearOutput", "geometricAdvantage",
                                            "maxStrain", "maxStress")}
        row["free"]["actuatorForce_N"] = free["actuatorForce"] * T
        row["grip"] = {k: grip[k] for k in ("converged", "strokeReached", "output", "linearOutput", "maxStrain",
                                            "maxStress", "mechanicalAdvantage")}
        row["grip"]["gripForce_N"] = grip["gripForce"] * T
        row["grip"]["actuatorForce_N"] = grip["actuatorForce"] * T
        # TPU over time: held at the stroke, force relaxes with the modulus
        f0 = row["grip"]["gripForce_N"]
        row["gripOverTime_N"] = {k: round(f0 * float(modulus_factor(mat, t)), 2)
                                 for k, t in (("1 s", 1.0), ("1 min", 60.0), ("1 h", 3600.0), ("1 day", DAY))}
        row["fastCloseActuatorForce_N"] = round(row["grip"]["actuatorForce_N"] * float(event_factor(mat, CLOSE_TIME)), 2)
        row["gripHistory"] = grip["history"]
        rows[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "gripHistory"}, default=float), flush=True)

    np.savez_compressed(os.path.join(a.out, "designs.npz"), **{k.split(" (")[0].replace(" ", "_"): v[:, 0, :]
                                                                for k, v in grids.items()})
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"thickness_mm": T, "stroke_mm": STROKE, "kIn_N_per_mm": K_IN * T, "kOut_N_per_mm": K_OUT * T,
                   "rows": rows, "seconds": round(time.time() - t0)}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
