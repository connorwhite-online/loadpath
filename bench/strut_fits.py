"""
Homogenized fits for the strut cells (lattice.STRUTS): periodic unit-cell
FEA (lattice.homogenize) at a ladder of strut diameters, then power laws
E*/E = C rho^n and peak strut stress / macro stress = A rho^-m.

Struts are cubic but anisotropic, so the fits are the conservative reading:
the modulus is the lesser of the [100] and [111] moduli and the stress
factor the largest of a [100] pull, a [111] pull and a shear. A cell that
is stiff along its struts (octet, BCCz along z) is better than this in its
good direction; the homogenized check does not count on it.

    python bench/strut_fits.py [--n 48] [--kinds bcc,octet]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.lattice import homogenize  # noqa: E402
from loadpath.struts import _BOX, STRUT_KINDS, strut_diameter_for_density  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--kinds", default=",".join(STRUT_KINDS))
    ap.add_argument("--rho", default="0.06,0.1,0.16,0.24,0.34")
    ap.add_argument("--out", default="strut-fits.json")
    a = ap.parse_args()
    out = {}
    for kind in a.kinds.split(","):
        rows = []
        for rho in (float(x) for x in a.rho.split(",")):
            # the unit cell is one periodic box (a Voronoi box is 2 seed spacings)
            d = float(strut_diameter_for_density(kind, rho, 1.0 / _BOX.get(kind, 1.0)))
            t0 = time.time()
            h = homogenize(kind, d, n=a.n)
            h["d"] = d
            h["Emin"] = min(h["E"], h["E111"])
            h["A"] = max(h["stressFactor"], h["stressFactor111"], h["shearStressFactor"])
            rows.append(h)
            print(kind, f"rho {h['rho']:.3f} d {d:.3f} E {h['E']:.4f} E111 {h['E111']:.4f} "
                        f"sf {h['stressFactor']:.2f} sf111 {h['stressFactor111']:.2f} "
                        f"shear {h['shearStressFactor']:.2f} "
                        f"zener {h['zener']:.2f} {time.time() - t0:.0f}s", flush=True)
        r = np.log([x["rho"] for x in rows])
        n_, lc = np.polyfit(r, np.log([x["Emin"] for x in rows]), 1)
        m_, la = np.polyfit(r, np.log([x["A"] for x in rows]), 1)
        fit = {"C": round(float(np.exp(lc)), 3), "n": round(float(n_), 2), "A": round(float(np.exp(la)), 2),
               "m": round(float(-m_), 2)}
        err_e = max(abs(fit["C"] * x["rho"] ** fit["n"] / x["Emin"] - 1) for x in rows)
        err_a = max(abs(fit["A"] * x["rho"] ** -fit["m"] / x["A"] - 1) for x in rows)
        out[kind] = {"fit": fit, "maxErrE": err_e, "maxErrA": err_a, "rows": rows}
        print(kind, fit, f"errE {err_e:.2f} errA {err_a:.2f}", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
