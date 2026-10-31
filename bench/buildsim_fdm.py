"""
FDM warp check against Armillotta, Bellotti & Cavallaro 2018, "Warpage of
FDM parts: experimental tests and analytic model", Robotics and CIM
50:140-152 (doi 10.1016/j.rcim.2017.09.007): ABSplus bars on a Stratasys
Dimension Elite, closed chamber at 75 C, 0.254 mm layers, flatness of the
top face (MinMax zone) measured by 3D scan after removal. Width had no
effect in their tests, so the bars run here as 1 mm strips at 0.254 mm
voxels (one voxel per printed layer, so no lumping).

Two numbers are fitted on the nine 20 mm-wide bars of their Table 2: the
retained share `relax` of the sub-Tg shrink (least squares; the model is
linear in it) and the bed's stiffness (the parts sat on a soluble-support
raft, neither glued rigid nor free), picked from --beds by the fit.
Run: python bench/buildsim_fdm.py [--beds 1 3 10 30] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from loadpath.buildsim import BuildSetup, simulate_build

LAYER = 0.254
# Table 2, w = 20 mm, mean of the two replicates (mm)
MEASURED = {(60, 1.5): 0.375, (60, 3.5): 0.445, (60, 5.5): 0.415,
            (100, 1.5): 0.58, (100, 3.5): 0.67, (100, 5.5): 0.675,
            (140, 1.5): 0.925, (140, 3.5): 1.045, (140, 5.5): 0.965}


def flatness(r):
    """MinMax zone of the released top face: z deviation after removing
    the best-fit plane."""
    m = r.mesh
    top = r.node_active & (m.node_ijk[:, 2] == m.node_ijk[r.node_active, 2].max())
    x, y = m.node_xyz[top, 0], m.node_xyz[top, 1]
    w = r.u[top, 2]
    a = np.c_[x, y, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(a, w, rcond=None)
    d = w - a @ c
    return float(d.max() - d.min())


def run(length, height, relax=1.0, bed=None):
    nx, ny, nz = int(round(length / LAYER)), 4, int(round(height / LAYER))
    solid = np.ones((nx, ny, nz), bool)
    setup = BuildSetup(process="fdm", material="abs", relax=relax, bed=75.0, chamber=75.0,
                        bed_stiffness=bed)
    t0 = time.time()
    r = simulate_build(solid, LAYER, (0, 0, 0), setup)
    return flatness(r), r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beds", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keys = sorted(MEASURED)
    mv = np.array([MEASURED[k] for k in keys])
    fits = []
    for bed in a.beds:
        sims = {}
        for k in keys:
            f, r, s = run(*k, bed=bed)
            sims[k] = f
            print(bed, k, f"unit-relax flatness {f:.3f} mm, mean peel {r.report['peel']['meanPeelStress']:.3f} MPa, "
                          f"{s:.0f} s", flush=True)
        sv = np.array([sims[k] for k in keys])
        relax = float(sv @ mv / (sv @ sv))
        pred = relax * sv
        fits.append({"bedStiffness": bed, "relax": relax,
                     "rows": [{"length": k[0], "height": k[1], "measured": MEASURED[k],
                               "predicted": round(float(p), 3)} for k, p in zip(keys, pred)],
                     "rmsError": float(np.sqrt(((pred - mv) ** 2).mean())),
                     "meanAbsRelError": float(np.mean(np.abs(pred - mv) / mv))})
        print(json.dumps(fits[-1]), flush=True)
    out = {"fits": fits, "best": min(fits, key=lambda f: f["rmsError"])}
    print(json.dumps(out["best"], indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
