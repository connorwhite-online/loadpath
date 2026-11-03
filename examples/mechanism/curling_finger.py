"""
A TPU finger that curls: a compliant mechanism optimized on its
large-deflection response (mechanism_nl.py) against one designed on the
linear model, both judged by driving them through the full stroke with the
nonlinear solver.

    python examples/mechanism/curling_finger.py [--out curling-finger-out/]

A 40 x 10 mm finger (1 mm cells, printed flat as a plate) in TPU 95A
(E = 26 MPa, nu = 0.45), clamped along the top half of its base. A tendon
or servo horn pulls the bottom of the base 5 mm back (-x); the tip should
curl down (-z) against a soft object (0.01 N/mm). Both designs use the
robust option (eroded / blueprint / dilated), so every hinge is at least
a filter radius wide and nothing hangs on grey or one-node joints.
About 20 min.
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

from loadpath.benchmarks.mechanisms import MechProblem, hinges, mech_domain  # noqa: E402
from loadpath.benchmarks.problems import _nodes  # noqa: E402
from loadpath.mechanism import MechanismOptions, optimize_mechanism  # noqa: E402
from loadpath.mechanism_nl import optimize_mechanism_nonlinear  # noqa: E402
from loadpath.nonlinear import stroke_mechanism  # noqa: E402

E, NU = 26.0, 0.45
STROKE, K_OUT = 5.0, 0.01


def finger():
    return MechProblem(name="finger", title="curling TPU finger", source="loadpath", dims=(40, 1, 10),
                       volfrac=0.3, rmin=2.0, fixed=[(_nodes(0, 0, range(6, 11)), "xz")], sym=[],
                       inp=(_nodes(0, 0, range(0, 3)), "x", -1.0), out=(_nodes(40, 0, range(4, 7)), "z", -1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="curling-finger-out")
    ap.add_argument("--starts", type=int, default=1, help="optimizer starts; the best thresholded design is kept")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    p = finger()
    dom = mech_domain(p)
    n, pres = dom.mesh.n_elem, int(dom.preserve.sum())
    vf = (p.volfrac * np.prod(p.dims) - pres) / (n - pres)
    base = dict(volume_fraction=vf, filter_radius=p.rmin, modulus=E, k_out=K_OUT, extrusion=1, robust=0.15)
    designs = {
        # a stiff actuator spring makes the force-driven linear optimum the stroke-driven one
        "linear design": optimize_mechanism(dom, NU, MechanismOptions(k_in=100 * E, **base)),
        "nonlinear design": optimize_mechanism_nonlinear(dom, NU, MechanismOptions(beta_every=10, max_iterations=160,
                                                                                   **base), stroke=STROKE,
                                                          starts=a.starts),
    }
    rows, grids = {}, {}
    for name, r in designs.items():
        g = np.zeros(dom.dims)
        g[tuple(dom.mesh.elem_ijk.T)] = r.xphys
        grids[name] = g[:, 0, :]
        row = {"iterations": r.iterations, "seconds": round(r.seconds, 1), "hinges": hinges((g > 0.5) * 1.0)}
        try:
            st = stroke_mechanism(dom, r.xphys, E, NU, STROKE, k_out=K_OUT, steps=12)
            row.update({"converged": st["converged"], "strokeReached": st["strokeReached"], "tipTravel": st["output"],
                        "linearPrediction": st["linearOutput"], "actuatorForce": st["actuatorForce"],
                        "maxStrain": st["maxStrain"],
                        "curve": [(0.0, 0.0)] + [(h["stroke"], h["output"]) for h in st["history"]]})
        except ValueError as ex:          # the tip came off when the design was thresholded
            row.update({"converged": False, "error": str(ex)})
        rows[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "curve"}, default=float), flush=True)
    np.savez_compressed(os.path.join(a.out, "designs.npz"), **{k.replace(" ", "_"): v for k, v in grids.items()})
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"stroke": STROKE, "kOut": K_OUT, "E": E, "rows": rows, "seconds": round(time.time() - t0)}, fh,
                  indent=1, default=float)


if __name__ == "__main__":
    main()
