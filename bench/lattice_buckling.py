"""
Buckling tables for the lattice buckling check (loadpath/lattice_buckling.py).

For each cell type, a few relative densities and two uniaxial compression
directions ([100], [111]), a periodic block of unit cells is solved
for its smallest elastic buckling load (linear eigenvalue). Strut cells use
a 2x2x2 block (so modes that alternate between neighbouring cells, the
pinned-end column modes, are in it), sheets one cell (eight times the
voxels would not fit the 15 GB box at five voxels through a wall).

    pip install pypardiso        # fast direct solves; SuperLU works but is slow
    python bench/lattice_buckling.py                    # every cell type
    python bench/lattice_buckling.py --kinds octet,kelvin

Writes loadpath/data/lattice_buckling.json (merges with the kinds already
there). Each row is keyed by the voxel cell's own relative density, which is
what the solve saw.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.lattice import TPMS, wall_for_density  # noqa: E402
from loadpath.lattice_buckling import DIRECTIONS, TABLE_PATH, unit_cell_buckling  # noqa: E402
from loadpath.struts import _CELLS, strut_diameter_for_density  # noqa: E402

STRUT_RHO = (0.03, 0.08, 0.2)
SHEET_RHO = (0.15, 0.3)
VOXELS_THROUGH = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(list(_CELLS) + list(TPMS)))
    ap.add_argument("--out", default=str(TABLE_PATH))
    a = ap.parse_args()
    out = Path(a.out)
    table = json.loads(out.read_text()) if out.exists() else {}
    table["about"] = ("Perfect-lattice elastic buckling under uniaxial macro compression (bench/lattice_buckling.py): "
                      "sigmaOverE = critical macro stress over the solid's modulus, per direction in the cell's "
                      "axes; rho is the voxel cell's relative density. Struts: 2x2x2 periodic block; sheets: "
                      "one cell.")
    table.setdefault("cells", {})
    for kind in [k for k in a.kinds.split(",") if k]:
        strut = kind in _CELLS
        cells = 2 if strut else 1
        rows = {"rho": [], "wallOverCell": [], "n": [], "sigmaOverE": {d: [] for d in DIRECTIONS},
                "modulusOverE": {d: [] for d in DIRECTIONS}, "cells": cells}
        for rho in STRUT_RHO if strut else SHEET_RHO:
            t = float(strut_diameter_for_density(kind, rho, 1.0) if strut else wall_for_density(kind, rho, 1.0))
            n = int(np.clip(np.ceil(VOXELS_THROUGH / t), 24, 36 if strut else (48 if kind == "diamond" else 64)))
            got = {}
            for d, vec in DIRECTIONS.items():
                t0 = time.time()
                r = unit_cell_buckling(kind, t, direction=vec, n=n, cells=cells)
                got[d] = r
                print(f"{kind:14s} rho {rho:.3f} (voxels {r['rho']:.3f}) t/c {t:.3f} n {n} [{d}] "
                      f"sigma/E {r['sigmaOverE']:.3e} strain {r['strainAtBuckling']:.3f} "
                      f"dof {r['dof']} {time.time() - t0:.0f}s", flush=True)
            if abs(got["100"]["rho"] / rho - 1.0) > 0.4:
                # under-resolved: the voxel struts are far off the drawn ones
                print(f"{kind}: rho {rho} voxelized to {got['100']['rho']:.3f}, dropped", flush=True)
                continue
            rows["rho"].append(got["100"]["rho"])
            rows["wallOverCell"].append(t)
            rows["n"].append(n)
            for d in DIRECTIONS:
                rows["sigmaOverE"][d].append(got[d]["sigmaOverE"])
                rows["modulusOverE"][d].append(got[d]["modulusOverE"])
        table["cells"][kind] = rows
        out.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
