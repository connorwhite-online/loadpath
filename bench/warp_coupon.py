"""
FDM warp calibration coupons: flat bars printed on the bed, measured for
flatness after they come off. The bars follow Armillotta et al. 2018 (20 mm
wide, 60-140 mm long, 1.5 and 3.5 mm tall; bench/buildsim_fdm.py), so the
same flatness measure and the same model apply, but in the user's own
material on an open printer, where the ABS-in-a-75-C-chamber fit is only a
guess. Writes one STL per bar and predictions.json (the model's flatness
and on-bed corner lift for each).

Run: python bench/warp_coupon.py --out DIR [--material petg] [--bed 70] [--chamber 30] [--pitch 0.25]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from loadpath.buildsim import BuildSetup, simulate_build, _fdm_params, FDM_PRINT_LAYER

BARS = [("W-140-1.5", 140, 1.5), ("W-140-3.5", 140, 3.5), ("W-100-1.5", 100, 1.5),
        ("W-100-3.5", 100, 3.5), ("W-60-3.5", 60, 3.5)]
WIDTH = 20.0


def flatness(r):
    """MinMax zone of the released top face: z deviation after removing
    the best-fit plane (as bench/buildsim_fdm.py)."""
    m = r.mesh
    top = r.node_active & (m.node_ijk[:, 2] == m.node_ijk[r.node_active, 2].max())
    x, y = m.node_xyz[top, 0], m.node_xyz[top, 1]
    w = r.u[top, 2]
    a = np.c_[x, y, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(a, w, rcond=None)
    d = w - a @ c
    return float(d.max() - d.min())


def predict(length, height, material, bed, chamber, pitch):
    # width has no effect on a long bar's bow (Armillotta); run a 1 mm strip
    grid = np.ones((int(round(length / pitch)), max(2, int(round(1.0 / pitch))), int(round(height / pitch))), bool)
    out = {}
    for release in ("full", "none"):
        setup = BuildSetup(process="fdm", material=material, bed=bed, chamber=chamber, release=release,
                           layer=pitch)
        r = simulate_build(grid, pitch, (0, 0, 0), setup)
        if release == "full":
            out["flatness"] = round(flatness(r), 3)
        else:
            m = r.mesh
            bottom = r.node_active & (m.node_ijk[:, 2] == m.node_ijk[r.node_active, 2].min()) \
                if r.node_active is not None else (m.node_ijk[:, 2] == m.node_ijk[:, 2].min())
            out["onBedLift"] = round(float(r.u_built[bottom, 2].max()), 3)
    return out


def box_stl(path, lx, ly, lz):
    v = np.array([[x, y, z] for z in (0, lz) for y in (0, ly) for x in (0, lx)], float)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    with open(path, "w") as f:
        f.write("solid coupon\n")
        for a, b, c, d in quads:
            for t in ((a, b, c), (a, c, d)):
                p = v[list(t)]
                n = np.cross(p[1] - p[0], p[2] - p[0])
                n /= np.linalg.norm(n)
                f.write(f" facet normal {n[0]:g} {n[1]:g} {n[2]:g}\n  outer loop\n")
                for q in p:
                    f.write(f"   vertex {q[0]:g} {q[1]:g} {q[2]:g}\n")
                f.write("  endloop\n endfacet\n")
        f.write("endsolid coupon\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--material", default="petg")
    ap.add_argument("--bed", type=float, default=70.0)
    ap.add_argument("--chamber", type=float, default=30.0)
    ap.add_argument("--pitch", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = _fdm_params(BuildSetup(process="fdm", material=a.material, bed=a.bed, chamber=a.chamber))
    rows = {}
    for name, length, height in BARS:
        t0 = time.time()
        box_stl(os.path.join(a.out, f"coupon-{name}.stl"), length, WIDTH, height)
        rows[name] = {"length": length, "width": WIDTH, "height": height,
                      **predict(length, height, a.material, a.bed, a.chamber, a.pitch)}
        print(name, rows[name], f"{time.time() - t0:.0f} s", flush=True)
    out = {"material": a.material, "process": "fdm", "printedLayer": FDM_PRINT_LAYER,
           "bed": a.bed, "chamber": a.chamber, "solidifiesAt": p["tg"], "tgSource": p["tgSource"],
           "relax": p["relax"], "model": "loadpath.buildsim FDM shrink, fitted on ABS (Armillotta 2018); "
           f"{a.pitch} mm voxels extrapolated to {FDM_PRINT_LAYER} mm layers",
           "measure": {"flatness": "mm, after removal: lay the bar top face down on glass, the gap under its "
                                   "highest point (feeler gauge), or the MinMax zone of a scan",
                       "onBedLift": "mm, before removal, after the bed cools: the gap under each end"},
           "bars": rows}
    with open(os.path.join(a.out, "predictions.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
