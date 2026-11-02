"""
Anisotropic lattice tables (loadpath/data/lattice_tensors.json): the full
6x6 stiffness and the wall-stress quadratic form of every sheet and strut
cell at a ladder of relative densities, from periodic unit-cell FEA
(lattice_tensors.unit_cell_tensors).

    python bench/lattice_tensors.py [--n 48] [--kinds octet,gyroid] [--out loadpath/data/lattice_tensors.json]

Merges into an existing table, so kinds can be run a few at a time, and
`--rho` adds densities to a kind (a row within 10% of a new one is replaced).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.lattice import SHEETS, wall_for_density  # noqa: E402
from loadpath.lattice_tensors import TABLE_PATH, unit_cell_tensors  # noqa: E402
from loadpath.struts import _BOX, STRUT_KINDS, strut_diameter_for_density  # noqa: E402

RHO = "0.02,0.035,0.06,0.1,0.16,0.24,0.34,0.5,0.7,0.9"
RHO_SHEET = "0.035,0.06,0.1,0.16,0.24,0.34,0.5,0.7,0.9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="voxels a cell (default: 48 under rho 0.2, 40 under 0.4, "
                                                     "32 above; thick walls resolve on fewer)")
    ap.add_argument("--kinds", default=",".join(list(SHEETS) + list(STRUT_KINDS)))
    ap.add_argument("--rho", default=None)
    ap.add_argument("--out", default=str(TABLE_PATH))
    a = ap.parse_args()
    out_path = Path(a.out)
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    table.update({"nu": 0.35, "n": a.n or "48/40/32 by density", "percentile": 99.5,
                  "about": "bench/lattice_tensors.py: C over E (Voigt, engineering shears), P the wall-stress "
                           "surface (lattice_tensors.stress_surface: quartic Q4 over lattice_tensors.MONOMIALS4, "
                           "floored at floor * sqrt(s' P s); Voigt stress, tensor shears), per relative density"})
    kinds = table.setdefault("kinds", {})
    for kind in a.kinds.split(","):
        # rows already in the table are kept unless this run solves the same
        # density again (--rho adds rows to a kind)
        rows = list(kinds.get(kind, []))
        ladder = a.rho or (RHO_SHEET if kind in SHEETS else RHO)
        for rho in (float(x) for x in ladder.split(",")):
            if kind in SHEETS:
                d = float(wall_for_density(kind, rho, 1.0))
            else:
                d = float(strut_diameter_for_density(kind, rho, 1.0 / _BOX.get(kind, 1.0)))
            t0 = time.time()
            n = a.n or (64 if rho < 0.05 else 48 if rho < 0.2 else 40 if rho < 0.4 else 32)
            r = unit_cell_tensors(kind, d, n=n)
            rows = [x for x in rows if abs(np.log(x["rho"] / r["rho"])) > 0.1]
            rows.append({"rho": round(r["rho"], 5), "d": round(d, 5), "n": n,
                         "C": np.round(r["C"], 7).tolist(), "P": [[float(f"{v:.6g}") for v in row] for row in r["P"]],
                         "Q4": [float(f"{v:.6g}") for v in r["Q4"]], "floor": round(r["floor"], 4),
                         "holdout": {k: round(v, 3) for k, v in r["holdout"].items()},
                         "holdoutQuadratic": {k: round(v, 3) for k, v in r["holdoutQuadratic"].items()},
                         "E100": round(r["E100"], 6), "E111": round(r["E111"], 6), "G": round(r["G"], 6)})
            print(kind, f"rho {r['rho']:.3f} E100 {r['E100']:.4f} E111 {r['E111']:.4f} G {r['G']:.4f} "
                        f"P11^.5 {np.sqrt(r['P'][0, 0]):.2f} holdout {r['holdout']['min']:.2f}/{r['holdout']['mean']:.2f}/"
                        f"{r['holdout']['max']:.2f} (quadratic {r['holdoutQuadratic']['mean']:.2f}) "
                        f"{time.time() - t0:.0f}s", flush=True)
        rows.sort(key=lambda x: x["rho"])
        kinds[kind] = rows
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
