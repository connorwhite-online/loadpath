"""
Anisotropic lattice homogenization against the resolved lattice.

Lattice-only bars and blocks (no skin, so the lattice carries everything;
solid end blocks take the support and the load), each cell type laid
along the bar and turned 45 degrees (a transform node), analyzed three ways:

- resolved: the field graph voxelized at `--pitch` (several voxels across
  a strut) and solved on matrix-free multigrid, the incompatible-mode hex8
  the rest of loadpath uses;
- homogenized, anisotropic (lattice_tensors, the default);
- homogenized, isotropic (the scalar fits' conservative reading).

    python bench/lattice_aniso_validate.py [--pitch 0.15] [--cells 30000] [--cases octet-bar,...]
    python bench/lattice_aniso_validate.py --ccx octet-block     # also CalculiX on the resolved voxels

Writes bench/results/lattice-aniso.json and prints a table: deflection
and the lattice's peak stress, each method over the resolved one.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.domain import boundary_conditions  # noqa: E402
from loadpath.fem import HexMesh, hex8_b, von_mises, elastic_c, CORNERS  # noqa: E402
from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.homogenized import homogenized_check  # noqa: E402
from loadpath.mg import GridElasticity  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def bar(kind, thickness, rotate=None, cell=6.0, length=48.0, side=12.0, end=4.0, blend=0.0, skin=0.0, aspect=None):
    """A cantilever of lattice (in a `skin` mm shell if given): solid end
    blocks, the lattice between."""
    h = side / 2
    lat = {"op": "lattice", "type": kind, "cell": cell, "thickness": thickness, "origin": [0, -h, -h]}
    if blend:
        lat["blend"] = blend
    if aspect:
        lat["aspect"] = list(aspect)
    nodes = {"lat": lat}
    core = "lat"
    if rotate:
        nodes["turned"] = {"op": "transform", "input": "lat",
                           "rotate": {"axis": rotate[0], "degrees": rotate[1], "center": [length / 2, 0, 0]}}
        core = "turned"
    nodes["box"] = {"op": "shape", "shape": {"type": "box", "min": [0, -h, -h], "max": [length, h, h]}}
    nodes["fill"] = {"op": "intersect", "inputs": ["box", core]}
    nodes["ends"] = {"op": "shape", "shape": {"type": "union", "shapes": [
        {"type": "box", "min": [0, -h, -h], "max": [end, h, h]},
        {"type": "box", "min": [length - end, -h, -h], "max": [length, h, h]}]}}
    parts = ["fill", "ends"]
    if skin:
        nodes["skin"] = {"op": "shell", "input": "box", "thickness": skin}
        parts.append("skin")
    nodes["part"] = {"op": "union", "inputs": parts}
    doc = {"name": f"{kind}-bar", "nodes": nodes, "output": "part"}
    study = {"name": "bar", "designSpace": [{"type": "box", "min": [0, -h, -h], "max": [length, h, h]}],
             "preserve": [{"id": "root", "shape": {"type": "box", "min": [0, -h, -h], "max": [end, h, h]}},
                          {"id": "tip", "shape": {"type": "box", "min": [length - end, -h, -h],
                                                  "max": [length, h, h]}}],
             "supports": [{"region": "root"}],
             "loadCases": [{"name": "bend", "loads": [{"region": "tip", "force": [0, 0, -20]}]},
                           {"name": "pull", "loads": [{"region": "tip", "force": [100, 0, 0]}]},
                           {"name": "twist", "loads": [{"region": "tip", "torque": [2000, 0, 0]}]}],
             "material": {"E": 1700.0, "nu": 0.35, "strength": 48.0, "density": 1.01}, "goal": {"safetyFactor": 2}}
    return doc, study


CASES = {
    "octet-bar": lambda: bar("octet", 1.1),
    "octet-short": lambda: bar("octet", 1.1, ("z", 30), length=30),
    "octet-bar-45x": lambda: bar("octet", 1.1, ("x", 45)),
    "octet-bar-45z": lambda: bar("octet", 1.1, ("z", 45)),
    "cubic-bar": lambda: bar("cubic", 1.3),
    "cubic-bar-45z": lambda: bar("cubic", 1.3, ("z", 45)),
    "bccz-bar": lambda: bar("bccz", 1.1),
    "bccz-bar-90y": lambda: bar("bccz", 1.1, ("y", 90)),
    "gyroid-bar": lambda: bar("gyroid", 0.7),
    "gyroid-bar-45z": lambda: bar("gyroid", 0.7, ("z", 45)),
    # four cells across: less of the lattice is cut struts at the sides
    "octet-beam": lambda: bar("octet", 1.1, length=60, side=24, cell=6.0),
    "octet-beam-45z": lambda: bar("octet", 1.1, ("z", 45), length=60, side=24, cell=6.0),
    "cubic-beam": lambda: bar("cubic", 1.3, length=60, side=24),
    "cubic-beam-45z": lambda: bar("cubic", 1.3, ("z", 45), length=60, side=24),
    "bccz-beam": lambda: bar("bccz", 1.1, length=60, side=24),
    "bccz-beam-90y": lambda: bar("bccz", 1.1, ("y", 90), length=60, side=24),
    # in a 1 mm skin, the way a printed part carries a lattice
    "octet-skin-45z": lambda: bar("octet", 1.1, ("z", 45), length=60, side=24, skin=1.0),
    "cubic-skin-45z": lambda: bar("cubic", 1.3, ("z", 45), length=60, side=24, skin=1.0),
    "bccz-skin-90y": lambda: bar("bccz", 1.1, ("y", 90), length=60, side=24, skin=1.0),
    # stretched cells (aspect): tall BCC, its long axis along the beam and across it
    "bcc-tall-skin-90y": lambda: bar("bcc", 1.1, ("y", 90), length=60, side=24, skin=1.0, aspect=(1, 1, 1.5)),
    "bcc-tall-skin": lambda: bar("bcc", 1.1, length=60, side=24, skin=1.0, aspect=(1, 1, 1.5)),
    "octet-flat-skin-45z": lambda: bar("octet", 1.1, ("z", 45), length=60, side=24, skin=1.0, aspect=(1.4, 1, 1)),
}


def resolved(doc, study_d, pitch, band, cell_size=6.0, skin=0.0):
    """Voxel FE of the field graph's part at `pitch`."""
    st = study_from_dict(study_d)
    g = FieldGraph(doc)
    lo, hi = g.bounds(doc["output"])
    dims = np.ceil((hi - lo) / pitch - 1e-9).astype(int)
    origin = lo
    keep = []
    for i in range(dims[0]):
        x = origin[0] + (i + 0.5) * pitch
        jj, kk = np.meshgrid(np.arange(dims[1]), np.arange(dims[2]), indexing="ij")
        P = np.stack([np.full(jj.size, x), origin[1] + (jj.ravel() + 0.5) * pitch,
                      origin[2] + (kk.ravel() + 0.5) * pitch], 1)
        d = np.asarray(g.evaluate(P, doc["output"]), float)
        s = d < 0
        keep.append(np.stack([np.full(s.sum(), i), jj.ravel()[s], kk.ravel()[s]], 1))
    ijk = np.concatenate(keep)
    from scipy import ndimage

    grid = np.zeros(dims, bool)
    grid[tuple(ijk.T)] = True
    lbl, _ = ndimage.label(grid)
    hm = HexMesh(ijk, pitch, origin)
    bc = boundary_conditions(st, hm, check_loads=False, tol_floor=0.5 * pitch)
    sup = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    sel = np.isin(lbl[tuple(ijk.T)], sup)
    if not sel.all():
        ijk = ijk[sel]
        hm = HexMesh(ijk, pitch, origin)
        bc = boundary_conditions(st, hm, check_loads=False, tol_floor=0.5 * pitch)
    mat = st.material
    nu = st.elastic(mat)
    t0 = time.time()
    solver = GridElasticity(hm, nu, bc.fixed, springs=mat.E * bc.springs if bc.springs is not None else None,
                            rtol=1e-7, maxiter=4000)
    solver.set_moduli(np.full(hm.n_elem, mat.E))
    us = solver.solve_many([bc.loads[ci] for ci in range(len(st.load_cases))])
    centers = origin + (ijk + 0.5) * pitch
    from scipy.spatial import cKDTree

    out = []
    c0 = elastic_c(mat.E, nu)
    bcs = [hex8_b(a, b, c, nu) / pitch for (a, b, c) in 2.0 * CORNERS - 1.0]
    for ci, case in enumerate(st.load_cases):
        u = us[ci]
        ue = u[hm.edof]
        nodal = np.zeros((hm.n_nodes, 6))
        for kk in range(8):
            np.add.at(nodal, hm.enode[:, kk], (ue @ bcs[kk].T) @ c0.T)
        nodal /= np.bincount(hm.enode.ravel(), minlength=hm.n_nodes)[:, None]
        vm = von_mises(nodal)
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        near = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
        ln = bc.case_load_nodes[ci]
        disp = u.reshape(-1, 3)
        # each lattice cell's 99.5th percentile (the unit-cell tables' reading),
        # the worst cell outside the contact band
        far = ~near
        # bins one lattice cell each: a stretched cell's box (cell_size may
        # be per axis), as the unit-cell tables read their percentile
        key = np.floor((hm.node_xyz[far] - origin) / np.asarray(cell_size, float)).astype(int)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        v = vm[far]
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(inv.max() + 2))
        groups = [order[bounds[j]:bounds[j + 1]] for j in range(len(bounds) - 1) if bounds[j + 1] - bounds[j] > 50]
        cell_peaks = [np.percentile(v[gr], 99.5) for gr in groups]
        jmax = int(np.argmax(cell_peaks))
        xyz = hm.node_xyz[far]
        top = groups[jmax][np.argmax(v[groups[jmax]])]
        # the same reading away from the outer faces (a skin and the strut
        # roots in it): how much of the peak is the junction with the skin
        dface = np.minimum(xyz - lo, hi - xyz).min(axis=1)
        half = 0.5 * float(np.min(cell_size))
        inner = dface > skin + half if skin else dface > half
        inner_peaks = [np.percentile(v[gr[inner[gr]]], 99.5) for gr in groups if inner[gr].sum() > 50]
        out.append({"name": case.name, "deflection": float(np.linalg.norm(disp[ln].mean(axis=0))),
                    "compliance": float(bc.loads[ci] @ u),
                    "peak995": float(np.percentile(v, 99.5)), "cellPeak": float(max(cell_peaks)),
                    "cellPeakAt": xyz[top].tolist(), "cellPeakFaceDistance": float(dface[top]),
                    "innerCellPeak": float(max(inner_peaks)) if inner_peaks else None})
    return {"cells": int(hm.n_elem), "pitch": pitch, "seconds": time.time() - t0,
            "mass": float(hm.n_elem * pitch ** 3 * mat.density / 1000.0), "cases": out, "hm": hm, "bc": bc,
            "study": st}


def ccx_check(res, workdir):
    """The resolved voxels as CalculiX C3D8I elements (incompatible modes),
    the same supports and loads; deflection of each case's loaded nodes."""
    hm, bc, st = res["hm"], res["bc"], res["study"]
    mat = st.material
    nu = float(st.elastic(mat))
    cor = CORNERS
    # map CORNERS to ccx's: bottom face (z=0) counter-clockwise then top
    want = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    order = [int(np.nonzero((cor == np.array(w)).all(1))[0][0]) for w in want]
    out = []
    fixed = bc.fixed.reshape(-1, 3)
    for ci, case in enumerate(st.load_cases):
        d = Path(workdir) / f"c{ci}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "job.inp", "w") as fh:
            fh.write("*NODE\n")
            for i, p in enumerate(hm.node_xyz):
                fh.write(f"{i + 1},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")
            fh.write("*ELEMENT,TYPE=C3D8I,ELSET=EALL\n")
            for e, nodes in enumerate(hm.enode):
                fh.write(f"{e + 1}," + ",".join(str(int(nodes[k]) + 1) for k in order) + "\n")
            fh.write(f"*MATERIAL,NAME=M\n*ELASTIC\n{mat.E},{nu}\n*SOLID SECTION,ELSET=EALL,MATERIAL=M\n")
            fh.write("*STEP\n*STATIC\n*BOUNDARY\n")
            for n_ in np.nonzero(fixed.any(1))[0]:
                for j in range(3):
                    if fixed[n_, j]:
                        fh.write(f"{n_ + 1},{j + 1},{j + 1},0\n")
            f = bc.loads[ci].reshape(-1, 3)
            fh.write("*CLOAD\n")
            for n_ in np.nonzero(np.abs(f).sum(1) > 0)[0]:
                for j in range(3):
                    if f[n_, j] != 0:
                        fh.write(f"{n_ + 1},{j + 1},{f[n_, j]:.9e}\n")
            fh.write("*NODE FILE\nU\n*END STEP\n")
        t0 = time.time()
        subprocess.run(["ccx", "-i", "job"], cwd=d, check=True, stdout=subprocess.DEVNULL,
                       env={**os.environ, "OMP_NUM_THREADS": "4"})
        u = _read_frd(d / "job.frd", hm.n_nodes)
        ln = bc.case_load_nodes[ci]
        out.append({"name": case.name, "deflection": float(np.linalg.norm(u[ln].mean(axis=0))),
                    "compliance": float(bc.loads[ci] @ u.ravel()), "seconds": time.time() - t0})
    return out


def _read_frd(path, n):
    u = np.zeros((n, 3))
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if "DISP" in l)
    for l in lines[i + 1:]:
        if l.startswith(" -3"):
            break
        if l.startswith(" -1"):
            node = int(l[3:13])
            u[node - 1] = [float(l[13 + 12 * j:25 + 12 * j]) for j in range(3)]
    return u


def _cell_box(doc):
    """The lattice cell's box in global axes (mm): the cell times its
    aspect, permuted by a quarter-turn transform; a cube's side otherwise."""
    lat = doc["nodes"]["lat"]
    cell = float(lat["cell"])
    a = np.asarray(lat.get("aspect", (1, 1, 1)), float) * cell
    t = doc["nodes"].get("turned")
    if t is None:
        return a
    deg = float(t["rotate"]["degrees"]) % 180
    if np.isclose(deg, 90):
        ax = "xyz".index(t["rotate"]["axis"])
        i, j = [k for k in range(3) if k != ax]
        a[[i, j]] = a[[j, i]]
        return a
    return a if np.isclose(deg, 0) else np.full(3, a.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=0.15)
    ap.add_argument("--cells", type=int, default=30000)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--ccx", default="")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "lattice-aniso.json"))
    ap.add_argument("--reuse", action="store_true", help="keep the resolved results already in --out; "
                                                          "rerun only the homogenized checks")
    ap.add_argument("--band", type=float, default=None, help="contact band, mm (default one cell): stress "
                                                             "this near a support or load is left out of both peaks")
    ap.add_argument("--knockdown", type=float, default=None, help="free-surface knockdown (homogenized.py)")
    a = ap.parse_args()
    results = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for name in a.cases.split(","):
        doc, sd = CASES[name]()
        st = study_from_dict(sd)
        row = {}
        stretched = bool(doc["nodes"]["lat"].get("aspect"))
        for model in ("anisotropic",) if stretched else ("anisotropic", "isotropic"):
            r = homogenized_check(copy.deepcopy(doc), st, cells=a.cells, anisotropic=model == "anisotropic",
                                  contact_band=a.band or float(doc["nodes"]["lat"]["cell"]), free_surface=a.knockdown)
            row[model] = {"mass": r["mass"], "seconds": r["seconds"], "cells": r["analysisCells"],
                          "latticeExposed": r["latticeExposed"],
                          "cases": [{k: c[k] for k in ("name", "deflection", "compliance", "peakStress",
                                                       "safetyFactor", "peakAt", "inLattice")}
                                    for c in r["cases"]]}
        cell = float(doc["nodes"]["lat"]["cell"])
        if a.reuse and name in results:
            row["resolved"] = results[name]["resolved"]
            if "ccx" in results[name]:
                row["ccx"] = results[name]["ccx"]
        else:
            skin = float(doc["nodes"]["skin"]["thickness"]) if "skin" in doc["nodes"] else 0.0
            res = resolved(doc, sd, a.pitch, band=a.band or cell, cell_size=_cell_box(doc), skin=skin)
            row["resolved"] = {k: res[k] for k in ("cells", "pitch", "seconds", "mass", "cases")}
        if name in a.ccx.split(",") and not (a.reuse and "ccx" in row):
            with tempfile.TemporaryDirectory() as td:
                row["ccx"] = ccx_check(res, td)
        results[name] = row
        for ci, c in enumerate(row["resolved"]["cases"]):
            ani = row["anisotropic"]["cases"][ci]
            iso = row["isotropic"]["cases"][ci] if "isotropic" in row else None
            line = (f"{name:16s} {c['name']:6s} compliance res {c['compliance']:.4g} "
                    f"aniso x{ani['compliance'] / c['compliance']:.2f}"
                    + (f" iso x{iso['compliance'] / c['compliance']:.2f}" if iso else "")
                    + f" | cell peak res {c['cellPeak']:.3g} aniso x{ani['peakStress'] / c['cellPeak']:.2f}"
                    + (f" iso x{iso['peakStress'] / c['cellPeak']:.2f}" if iso else ""))
            if "ccx" in row:
                line += f" | ccx/res compliance {row['ccx'][ci]['compliance'] / c['compliance']:.3f}"
            print(line, flush=True)
        print(f"{name:16s} mass res {row['resolved']['mass']:.2f} g hom {row['anisotropic']['mass']:.2f} g "
              f"({row['resolved']['cells']} voxels, {row['resolved']['seconds']:.0f} s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
