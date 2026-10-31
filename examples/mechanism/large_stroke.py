"""
Large strokes: a compliant mechanism designed on the linear model against
one designed on the large-deflection response (mechanism_nl.py), both
judged the same way, by driving them through the stroke with the nonlinear
solver.

    python examples/mechanism/large_stroke.py [--out large-stroke-out/] [--stroke 12]

The case is the textbook force inverter (loadpath.benchmarks.mechanisms,
40 x 20 half model, volume 0.3, unit modulus) driven by a prescribed
stroke at its input, as Pedersen, Buhl & Sigmund (2001) set it up: at
12 cells of stroke (30% of the inverter's length) the members swing far
enough that the linear model's leverage no longer holds. The workpiece is
a soft spring (0.02) at the output.
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

from loadpath.benchmarks.mechanisms import hinges, inverter, mech_domain  # noqa: E402
from loadpath.mechanism import MechanismOptions, optimize_mechanism  # noqa: E402
from loadpath.mechanism_nl import optimize_mechanism_nonlinear  # noqa: E402
from loadpath.nonlinear import stroke_mechanism  # noqa: E402

NU = 0.3
K_OUT = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="large-stroke-out")
    ap.add_argument("--stroke", type=float, default=12.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    p = inverter()
    dom = mech_domain(p)
    n, pres = dom.mesh.n_elem, int(dom.preserve.sum())
    vf = (p.volfrac * np.prod(p.dims) - pres) / (n - pres)
    base = dict(volume_fraction=vf, filter_radius=1.5, k_out=K_OUT, extrusion=1)
    designs = {
        # a stiff actuator spring makes the force-driven linear optimum the stroke-driven one
        "linear design": optimize_mechanism(dom, NU, MechanismOptions(k_in=10.0, **base)),
        "nonlinear design": optimize_mechanism_nonlinear(dom, NU, MechanismOptions(**base), stroke=a.stroke),
    }
    rows, grids = {}, {}
    for name, r in designs.items():
        st = stroke_mechanism(dom, r.xphys, 1.0, NU, a.stroke, k_out=K_OUT, steps=16)
        g = np.zeros(dom.dims)
        g[tuple(dom.mesh.elem_ijk.T)] = r.xphys
        grids[name] = g[:, 0, :]
        curve = [(0.0, 0.0)] + [(h["stroke"], h["output"]) for h in st["history"]]
        best = max(curve, key=lambda c: c[1])
        rows[name] = {"iterations": r.iterations, "seconds": round(r.seconds, 1), "hinges": hinges((g > 0.5) * 1.0),
                      "converged": st["converged"], "output": st["output"], "linearPrediction": st["linearOutput"],
                      "peakOutput": best[1], "peakAtStroke": best[0], "actuatorForce": st["actuatorForce"],
                      "maxStrain": st["maxStrain"], "curve": curve}
        print(name, json.dumps({k: v for k, v in rows[name].items() if k != "curve"}, default=float), flush=True)
    np.savez_compressed(os.path.join(a.out, "designs.npz"), **{k.replace(" ", "_"): v for k, v in grids.items()})
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"stroke": a.stroke, "kOut": K_OUT, "rows": rows, "seconds": round(time.time() - t0)}, fh, indent=1,
                  default=float)


if __name__ == "__main__":
    main()
