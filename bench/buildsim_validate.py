"""
Validate loadpath.buildsim against the NIST AM-Bench 2018 bridge
(AMB2018-01): LPBF IN625 and 15-5 PH on an EOS M270, 20 um layers, built
on a 12.7 mm plate of the same alloy, then cut off the plate by wire EDM
along the legs (the base block at the tapered end stays on) and the rise
of 11 ridges on top measured by CMM.

Geometry and measurements: Phan et al. 2019, IMMI 8:318-334
(doi 10.1007/s40192-019-00149-0); NIST AMB2018-01 description and
CHAL-AMB2018-01-PD results pages. Blind model submissions: Levine et al.
2020, IMMI 9:1-15.

The LPBF strain factor k (eps = -k * yield / E) is fitted here, on IN625,
by least squares over the 11 ridges; 15-5 PH is then a prediction with the
same k. Run:  python bench/buildsim_validate.py [--pitch 0.5] [--fit]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from loadpath.buildsim import BuildSetup, simulate_build

# measured rise (mm) at the ridge centres x = 0.5 + 7 i, x = 0 the free end
RIDGE_X = 0.5 + 7.0 * np.arange(11)
MEASURED = {
    "in625": [1.276, 0.997, 0.754, 0.551, 0.387, 0.250, 0.146, 0.065, 0.012, 0.000, 0.003],
    "15-5ph": [1.168, 0.914, 0.695, 0.516, 0.358, 0.231, 0.133, 0.076, 0.008, 0.002, -0.001],
}
# blind predictions of the free-end rise submitted to the 2018 challenge
# (Levine 2020, Fig. 7, read off the plot)
BLIND_TIP = [1.88, 1.86, 1.80, 0.77, 0.74, 0.70]
CUT_END = 56.0          # the EDM cut runs under the legs, x 0..56


def bridge(pitch):
    """Bool grid (x along the bridge, z up) and origin of the AM-Bench
    bridge: 75 x 5 x 12 mm, legs 0-5 mm high, 45 degree overhangs closing
    each 2 mm window from its left side between z 5 and 7, solid above,
    1 mm x 0.5 mm ridges on 7 mm pitch on top. The 45 degree chisel at the
    attached end (2.5 mm in plan) is left square."""
    nx, ny, nz = int(round(75 / pitch)), int(round(5 / pitch)), int(round(12.5 / pitch))
    xc = (np.arange(nx) + 0.5) * pitch
    zc = (np.arange(nz) + 0.5) * pitch
    xx, zz = np.meshgrid(xc, zc, indexing="ij")
    solid = zz < 12.0
    windows = []
    x = 5.0
    for _ in range(4):
        for leg in (0.5, 2.5, 5.0):
            windows.append((x, x + 2.0))
            x += 2.0 + leg
    for a, b in windows:
        open_ = (xx > a) & (xx < b) & ((zz < 5.0) | ((zz < 7.0) & (xx > a + (zz - 5.0))))
        solid &= ~open_
    for i in range(11):
        solid |= (xx > 7.0 * i) & (xx < 7.0 * i + 1.0) & (zz >= 12.0) & (zz < 12.5)
    return np.repeat(solid[:, None, :], ny, axis=1), np.zeros(3)


def run(material, pitch, k=None, plate=12.7, layer=None):
    solid, origin = bridge(pitch)
    setup = BuildSetup(process="lpbf", material=material, plate=plate, plate_margin=1.0, k=k, layer=layer,
                       release="cut", cut=((-5.0, -5.0, -1e-6), (CUT_END, 10.0, pitch)))
    t0 = time.time()
    r = simulate_build(solid, pitch, origin, setup)
    m = r.mesh
    top = m.node_ijk[:, 2] == m.node_ijk[:, 2].max()
    rise = []
    for x in RIDGE_X:
        sel = top & (np.abs(m.node_xyz[:, 0] - x) <= 0.5 + 1e-9)
        rise.append(float((r.u[sel, 2] - r.u_built[sel, 2]).mean()))
    return np.array(rise), r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=0.5)
    ap.add_argument("--plate", type=float, default=12.7)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--fit", action="store_true", help="fit k on IN625 (the model is linear in k)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {"pitch": a.pitch, "plate": a.plate}
    k = a.k
    if a.fit:
        rise1, r, secs = run("in625", a.pitch, k=1.0, plate=a.plate)
        meas = np.array(MEASURED["in625"])
        k = float(rise1 @ meas / (rise1 @ rise1))
        out["fit"] = {"k": k, "secondsPerRun": secs}
        print(f"fitted k = {k:.4f} ({secs:.0f} s)")
    for mat in ("in625", "15-5ph"):
        rise, r, secs = run(mat, a.pitch, k=k, plate=a.plate)
        meas = np.array(MEASURED[mat])
        err = rise - meas
        row = {"rise": rise.round(4).tolist(), "measured": meas.tolist(),
               "tipError": float(err[0] / meas[0]), "rmsError": float(np.sqrt((err ** 2).mean())),
               "seconds": round(secs, 1), "strain": r.report["strain"]["strain"],
               "residualStress": r.report["residualStress"], "cells": r.report["cells"]}
        # longitudinal stress through leg L7 (x 31) on the plate: the contour
        # measurement reads ~-200 MPa at the plate and ~+600 MPa at the top
        m = r.mesh
        c = m.elem_centers()
        sel = r.part & (np.abs(c[:, 0] - 31.0) < a.pitch) & (np.abs(c[:, 1] - 2.5) < a.pitch)
        zs = np.unique(np.round(c[sel, 2], 4))
        row["sxxL7"] = [[float(z), float(r.stress_built[sel & (np.abs(c[:, 2] - z) < 1e-3), 0].mean())] for z in zs]
        out[mat] = row
        print(mat, "tip", f"{rise[0]:.3f} vs {meas[0]:.3f} ({100 * err[0] / meas[0]:+.0f}%)",
              "rms", f"{row['rmsError']:.3f} mm", f"{secs:.0f} s")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
