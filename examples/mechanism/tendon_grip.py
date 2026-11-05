"""
A tendon-driven TPU finger designed for grip force on an object.

    python examples/mechanism/tendon_grip.py [--out tendon-arm-out/] [--quick]

A 50 x 10 mm finger (1 mm cells, printed flat as a plate, 1 mm half
thickness modelled) in TPU 95A, its base clamped to the palm. A tendon runs
from the servo's spool behind the palm through guides on the palm side of
the finger to a knot at the tip. The servo winds in 8 mm of cable; cable
and servo give 0.5 N/mm, so the pull tops out at 4 N. Below the finger,
centred under its middle, sits a 30 mm cylinder (a arm-sized handle or
bottle), 1.5 mm away. The
optimizer (mechanism_nl, large deflections, contact as a smooth penalty)
maximizes the force the finger presses on the cylinder.

The baselines are what one would print without topology optimization:
a plain bar of the same volume along the palm side, carrying the guides.
Every design is checked thresholded (solid or empty) with the tendon
pulled to full tension.
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
from loadpath.mechanism import MechanismOptions  # noqa: E402
from loadpath.mechanism_nl import RigidCylinder, ServoTendon, optimize_mechanism_nonlinear, tendon_grip  # noqa: E402
from loadpath.nonlinear import nodes_near  # noqa: E402

E, NU = 26.0, 0.45
L, H = 50, 10
STROKE, K_SERVO = 8.0, 0.5          # mm of cable wound in; N/mm of cable + servo give (at most 4 N on the
                                    # 1 mm half plate, 8 N on a 2 mm finger)
GUIDES = (1, 17, 33)                # x of the guides on the palm side; the knot is at the tip
OBJ_R, GAP = 15.0, 1.5


def setup():
    p = MechProblem(name="tendon-finger", title="tendon-driven TPU finger", source="loadpath", dims=(L, 1, H),
                    volfrac=0.3, rmin=2.0, fixed=[(_nodes(0, 0, range(0, H + 1)), "xz")], sym=[],
                    inp=(_nodes(L, 0, range(0, 3)), "x", -1.0), out=(_nodes(L, 0, range(0, 3)), "z", -1.0))
    dom = mech_domain(p)
    m = dom.mesh
    ijk = m.elem_ijk
    for gx in GUIDES:                # guide pads: two cells long, two tall, on the palm side
        dom.preserve[(ijk[:, 0] >= gx - 1) & (ijk[:, 0] <= gx) & (ijk[:, 2] <= 1)] = True
    path = [np.array([-6.0, 0.5, 1.0])] + [nodes_near(m, (gx, 0.5, 1.0), 0.9) for gx in GUIDES] + \
        [nodes_near(m, (L - 0.5, 0.5, 1.0), 0.9)]
    ten = ServoTendon(path=path, stroke=STROKE, stiffness=K_SERVO)
    obj = RigidCylinder(center=(30.0, 0.0, -OBJ_R - GAP), radius=OBJ_R)
    return p, dom, ten, obj


def bar(dom, z0, z1):
    ijk = dom.mesh.elem_ijk
    x = ((ijk[:, 2] >= z0) & (ijk[:, 2] < z1)).astype(float)
    x[dom.preserve] = 1.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tendon-arm-out")
    ap.add_argument("--quick", action="store_true", help="no robust option, 60 iterations")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    p, dom, ten, obj = setup()
    n, pres = dom.mesh.n_elem, int(dom.preserve.sum())
    vf = (p.volfrac * n - pres) / (n - pres)
    o = dict(volume_fraction=vf, filter_radius=p.rmin, modulus=E, extrusion=1, beta_every=10)
    o.update(dict(max_iterations=60) if a.quick else dict(robust=0.15, max_iterations=100, beta_every=8))
    r = optimize_mechanism_nonlinear(dom, NU, MechanismOptions(**o), tendons=[ten], contact=obj,
                                     on_iter=lambda h, x: print(h["iteration"], round(h["outputDisplacement"], 3),
                                                                h["beta"], h.get("tensionScale"), flush=True) if h["iteration"] % 5 == 0
                                     else None)
    designs = {"optimized": r.xphys, "palm-side bar": bar(dom, 0, 3)}
    rows, grids, shapes = {}, {}, {}
    for name, x in designs.items():
        g = np.zeros(dom.dims)
        g[tuple(dom.mesh.elem_ijk.T)] = x
        grids[name] = g[:, 0, :]
        c = tendon_grip(dom, x, E, NU, [ten], obj)
        rows[name] = {"grip": c["grip"], "converged": c["converged"], "loadReached": c["loadReached"], "tension": c["tension"],
                      "contactNodes": c.get("contactNodes"), "curve": [(h["tension"], h["grip"]) for h in c["history"]],
                      "tendonTravel": c.get("tendonTravel"), "volume": float((x > 0.5).mean()),
                      "hinges": hinges((g > 0.5) * 1.0)}
        shapes[name] = c["deformed"][dom.mesh.enode[c["solid"]]][:, :, [0, 2]]
        print(name, json.dumps({k: v for k, v in rows[name].items() if k != "curve"}, default=float), flush=True)
    rows["optimized"].update({"iterations": r.iterations, "seconds": round(r.seconds, 1)})
    np.savez_compressed(os.path.join(a.out, "designs.npz"), **{k.replace(" ", "_"): v for k, v in grids.items()},
                        **{"deformed_" + k.replace(" ", "_"): v for k, v in shapes.items()})
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"stroke": STROKE, "servoStiffness": K_SERVO, "objectRadius": OBJ_R, "gap": GAP, "rows": rows,
                   "seconds": round(time.time() - t0)}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
