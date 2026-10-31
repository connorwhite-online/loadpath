"""
Warp-aware topology optimization demo (buildsim.optimize_warp): a Ti64
LPBF shelf bracket, 40 x 4 x 24 mm, built standing on the plate (+z). In
service it is bolted along its left face (x = 0) and carries a load down
at the top right corner. Two designs at the same volume: compliance only
(warp_weight 0) and compliance + as-built distortion (warp_weight 1),
both extruded through the thickness (2.5D), the flange and load pad kept.
Both delivered designs are then re-simulated with the full build model
(simulate_build: plate, then cut off) to read distortion and residual
stress, and their service compliance compared.

Run: python bench/buildsim_optimize.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from loadpath.buildsim import BuildSetup, WarpOptions, optimize_warp, simulate_build
from loadpath.fem import HexMesh, Stiffness
from loadpath.thermal import ThermalGrid

NX, NY, NZ = 40, 4, 24
PITCH = 1.0


def problem():
    ijk = np.argwhere(np.ones((NX, NY, NZ), bool))
    mesh = HexMesh(ijk, PITCH, np.zeros(3))
    # the bolting flange (x = 0) and the load pad are kept solid
    e = mesh.elem_ijk
    pres = (e[:, 0] == 0) | ((e[:, 0] >= NX - 2) & (e[:, 2] >= NZ - 2))
    grid = ThermalGrid(PITCH, np.zeros(3), (NX, NY, NZ), mesh, pres)
    supports = np.repeat(mesh.node_ijk[:, 0] == 0, 3)
    f = np.zeros(mesh.n_dof)
    tip = np.flatnonzero((mesh.node_ijk[:, 0] == NX) & (mesh.node_ijk[:, 2] >= NZ - 1))
    f[3 * tip + 2] = -100.0 / len(tip)
    return grid, supports, f


def compliance(grid, supports, f, xp, e0, nu):
    from loadpath.buildsim import _solve
    st = Stiffness(grid.mesh, nu, supports)
    k = st.assemble(e0 * (1e-6 + (1 - 1e-6) * (xp > 0.5)))
    u = _solve(k, f, grid.mesh.node_xyz)
    return float(f @ u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--iterations", type=int, default=80)
    a = ap.parse_args()
    grid, supports, f = problem()
    setup = BuildSetup(process="lpbf", material="ti64")
    out = {}
    for w in (0.0, 1.0):
        opts = WarpOptions(volume_fraction=0.3, filter_radius=1.6, process=setup, warp_weight=w, layer=2.0,
                           max_iterations=a.iterations, extrusion=1)
        r = optimize_warp(grid, supports, f, opts,
                          on_iter=lambda rec, xp: print(f"  w{w} it {rec['iteration']} c {rec['compliance']:.3f} "
                                                        f"D {rec['distortion']:.3e}", flush=True)
                          if rec["iteration"] % 10 == 0 else None)
        solid = grid.full_grid(r.xphys) > 0.5
        c = compliance(grid, supports, f, r.xphys, 110000.0, 0.34)
        rows = {"warpWeight": w, "volume": float(solid.mean()), "serviceCompliance": c,
                "optimizerDistortionMs": r.distortion, "iterations": r.iterations, "seconds": r.seconds}
        for rel in ("none", "full"):
            b = simulate_build(solid, PITCH, (0, 0, 0), BuildSetup(process="lpbf", material="ti64", plate=5.0,
                                                                  release=rel, layer=2.0))
            rows[f"build_{rel}"] = {"distortionMax": b.report["distortion"]["max"],
                                    "distortionRms": b.report["distortion"]["rms"],
                                    "vmP95": b.report["residualStress"]["vmP95"],
                                    "recoaterLift": b.report.get("recoater", {}).get("maxLift")}
        out[f"w{w:g}"] = rows
        np.save(f"/tmp/warp_design_w{w:g}.npy", solid)
        print(json.dumps(rows, indent=1), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
