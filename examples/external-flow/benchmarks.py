"""
External flow against published drag: spheres and cylinders across Reynolds
number, a square strut, a cube and a disk. Each case is one call of
`loadpath.aero.external_flow`; the table goes in docs/EXTERNAL-FLOW.md.

    python examples/external-flow/benchmarks.py                 # every case (about 1.5 hours on 4 cores)
    python examples/external-flow/benchmarks.py sphere-100 cylinder-100
    python examples/external-flow/benchmarks.py --out results.json

References
- Sphere: Dennis & Walker 1971 (Re 20, 100); Johnson & Patel 1999 (Re 100, 300: Cd 0.656, St 0.137);
  Clift, Grace & Weber 1978 standard drag curve (Re 1000: 0.47); Rodriguez et al. 2011 DNS (Re 3700: 0.394).
- Cylinder: Dennis & Chang 1970 (Re 20: 2.05, Re 40: 1.52); Henderson 1995 and Williamson 1996 (Re 100:
  Cd 1.33-1.35, St 0.164-0.166); Parnaudeau et al. 2008 and Kravchenko & Moin 2000 (Re 3900: 0.98-1.04, St 0.21).
- Square cylinder: Lyn et al. 1995 (Re 22,000: Cd 2.1, St 0.132); Sohankar et al. 1999 (Re 200-500: 1.4-1.9).
- Cube face-on and disk normal to the stream: Hoerner 1965 (1.05, 1.17).
"""
import json
import sys
import time

import numpy as np
import trimesh

from loadpath.aero import external_flow

D = 20.0
NU = 1.85e-5 / 1.18


def sphere():
    return trimesh.creation.icosphere(subdivisions=5, radius=D / 2)


def cylinder(span):
    return trimesh.creation.cylinder(radius=D / 2, height=span * D, sections=256)


def square(span):
    return trimesh.creation.box((D, D, span * D))


def disk():
    return trimesh.creation.cylinder(radius=D / 2, height=D / 8, sections=128)


# name: (mesh, Re, external_flow options, reference Cd (low, high), reference St or None)
SPAN = {"span": (0, 0, 1)}
CASES = {
    "sphere-20": (sphere, 20, {}, (2.73, 2.73), None),
    "sphere-100": (sphere, 100, {}, (1.087, 1.087), None),
    "sphere-300": (sphere, 300, {}, (0.656, 0.671), 0.137),
    "sphere-1000": (sphere, 1000, {"max_time": 70}, (0.47, 0.48), None),
    "cylinder-20": (lambda: cylinder(0.5), 20, SPAN, (2.05, 2.05), None),
    "cylinder-40": (lambda: cylinder(0.5), 40, SPAN, (1.52, 1.52), None),
    "cylinder-100": (lambda: cylinder(0.5), 100, SPAN, (1.33, 1.35), 0.165),
    # past Re 190 the wake is three-dimensional: 1.5 diameters of span, and two surface layers
    "cylinder-3900": (lambda: cylinder(1.5), 3900, {**SPAN, "cells": 16, "layers": 2, "max_time": 60}, (0.98, 1.04), 0.21),
    "square-22000": (lambda: square(2), 22000, {**SPAN, "cells": 16, "max_time": 60}, (2.05, 2.2), 0.132),
    "cube-20000": (lambda: trimesh.creation.box((D, D, D)), 20000, {"cells": 16, "max_time": 60}, (1.05, 1.05), None),
    "disk-20000": (disk, 20000, {"cells": 16, "max_time": 60}, (1.12, 1.17), None),
}


def run(name, verbose=False):
    make, re, kw, ref, st = CASES[name]
    mesh = make()
    if name.startswith("disk"):
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    t0 = time.time()
    r = external_flow(mesh, re * NU / (D * 1e-3), (1, 0, 0), verbose=verbose, **kw)
    mid = 0.5 * (ref[0] + ref[1])
    return {"case": name, "reynolds": re, "reynoldsRun": r["reynoldsRun"], "cd": r["dragCoefficient"],
            "reference": list(ref), "error": r["dragCoefficient"] / mid - 1.0,
            "strouhal": r["strouhal"], "strouhalReference": st, "cellsAcross": r["cellsAcross"],
            "layers": r["layers"], "boundaryLayerCells": r["boundaryLayerCells"], "time": r["time"],
            "converged": r["converged"], "standardError": (r["dragStandardError"] or 0.0) / max(r["drag"], 1e-30), "seconds": round(time.time() - t0, 1), "blockage": r["blockage"]}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out in args:
        args.remove(out)
    rows = []
    for name in args or list(CASES):
        row = run(name, verbose="--verbose" in sys.argv)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if out:
            with open(out, "w") as fh:
                json.dump(rows, fh, indent=1)
