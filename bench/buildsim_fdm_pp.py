"""
FDM warp of a semi-crystalline polymer: the polypropylene plates of Samy
et al. 2022 (Polymers 14:2746, doi 10.3390/polym14132746). 50 x 50 x 2 mm,
four 0.5 mm layers, 100% infill, bed 100 C, enclosed at 25 C, on a modified
Ultimaker 2. They report the z warpage at a point m near a corner, (7.8,
2.1) mm on the top layer, scanned against the part's own corner axis: 0.80
to 1.25 mm over five raster patterns (0.80 mm for 90/90 lines, 0.83 mm
concentric, 1.10-1.25 mm for 0/90 and the zigzags). Their own process
model (crystallization kinetics, viscoelastic) read 0.75-1.09 mm.

loadpath's FDM model is isotropic (no raster), so it is compared with the
range. It runs with no PP-specific fitting: relax and the bed's stiffness
are the ABS values fitted on Armillotta 2018; PP solidifies at its
crystallization temperature (115 C, an estimate) with an estimated 120
um/(m K). How "warpage of element m" was referenced is not fully stated, so
two readings are given: m's lift over the lowest point of the underside
after removal (the plate resting on a table), and m's rise relative to the
plate's own corner (0, 0) as built.

Run: python bench/buildsim_fdm_pp.py [--pitch 0.5] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from loadpath.buildsim import BuildSetup, simulate_build

MEASURED = {"line 90/90": 0.80, "line 0/90": 1.10, "zigzag 45/45": 1.14, "zigzag 45/-45": 1.25,
            "concentric": 0.83}
SAMY_MODEL = {"line 90/90": 0.794, "line 0/90": 0.98, "zigzag 45/45": 0.92, "zigzag 45/-45": 1.09,
              "concentric": 0.75}
M = (7.8, 2.1)


def run(pitch=0.5, bed_stiffness=None, relax=None, release="full"):
    n, nz = int(round(50 / pitch)), int(round(2.0 / pitch))
    solid = np.ones((n, n, nz), bool)
    st = BuildSetup(process="fdm", material="pp", bed=100.0, chamber=25.0, layer=0.5, release=release,
                    bed_stiffness=bed_stiffness, relax=relax, print_layer=0.5)
    t0 = time.time()
    r = simulate_build(solid, pitch, (0, 0, 0), st)
    m = r.mesh
    top = m.node_ijk[:, 2] == nz
    bot = m.node_ijk[:, 2] == 0
    xy = m.node_xyz[:, :2]
    im = np.flatnonzero(top)[np.argmin(np.linalg.norm(xy[top] - M, axis=1))]
    i0 = np.flatnonzero(top)[np.argmin(np.linalg.norm(xy[top], axis=1))]
    u = r.u if release != "none" else r.u_built
    lift_table = float(u[im, 2] - u[bot, 2].min())
    rise_corner = float(u[im, 2] - u[i0, 2])
    corner = float(u[i0, 2] - u[bot, 2].min())
    return {"pitch": pitch, "release": release, "liftOverLowest": lift_table, "riseOverCorner": rise_corner,
            "cornerLift": corner, "maxDistortion": r.report["distortion"]["max"],
            "strain": r.report["strain"], "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [run(a.pitch, release="full"), run(a.pitch, release="none")]
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k != "strain"}), flush=True)
    out = {"measured": MEASURED, "samyModel": SAMY_MODEL, "loadpath": rows}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
