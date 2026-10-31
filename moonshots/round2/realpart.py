"""
A delivered part: voxel today (verify.py), cut-cell + incompatible-mode hex
+ surface recovery, and the P2 tet cross-check, on the same study and the
same contact band. Peak von Mises per load case and where it is.

Run: python moonshots/round2/realpart.py study.json part.stl [--cells 60000] [--out r.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cutcell import CutModel, SubcellRule  # noqa: E402

from loadpath.fem import von_mises  # noqa: E402


def fine_indicator(mesh, pitch, n):
    """Subcell indicator of the coarse grid from a ray-parity voxelization at
    pitch / n on an aligned box. Returns (origin, dims, ind (ncell, n^3))."""
    from loadpath.verify import voxelize

    lo, hi = np.asarray(mesh.bounds, float)
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1) + 2
    origin = lo - 0.5 * (dims * pitch - (hi - lo))
    hi2 = origin + dims * pitch
    # shave a hair so ceil() gives exactly dims * n fine cells
    fine, forg = voxelize(mesh, pitch / n, box=(origin + 1e-9, hi2 - 1e-9))
    assert fine.shape == tuple(dims * n), (fine.shape, dims * n)
    assert np.allclose(forg, origin, atol=1e-6 * pitch), (forg, origin)
    d = dims
    ind = fine.reshape(d[0], n, d[1], n, d[2], n).transpose(0, 2, 4, 1, 3, 5).reshape(-1, n ** 3)
    return origin, dims, ind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("part")
    ap.add_argument("--cells", type=int, default=60_000)
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--band", type=float, help="contact band, mm (default: the tet's 2 elements, at least 1.5 cells)")
    ap.add_argument("--tet-size", type=float)
    ap.add_argument("--no-tet", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    import trimesh
    from scipy import ndimage
    from scipy.spatial import cKDTree

    from loadpath import crosscheck as xc
    from loadpath.directions import expand
    from loadpath.domain import boundary_conditions
    from loadpath.fem import HexMesh
    from loadpath.spec import study_from_dict
    from loadpath.verify import _user_edges, verify_part

    d = json.load(open(a.study))
    study = study_from_dict(d, base_dir=os.path.dirname(os.path.abspath(a.study)))
    mesh = trimesh.load(a.part, force="mesh")
    mat = study.material
    vol = float(mesh.volume)
    pitch = (vol / a.cells) ** (1.0 / 3.0)
    tet_size = a.tet_size or xc.fit_size(vol, xc.element_size(study))
    band = a.band or max(2.0 * tet_size, 1.5 * pitch)
    out = {"part": a.part, "pitch": pitch, "band": band, "tetSize": tet_size}
    print(json.dumps({"pitch": pitch, "band": band, "tetSize": tet_size}), flush=True)

    # --- voxel today --------------------------------------------------------
    t0 = time.time()
    rep = verify_part(study, mesh, max_cells=a.cells, contact_band=band, refine=3)
    out["voxel"] = {"seconds": time.time() - t0, "cells": rep["analysisCells"],
                    "cases": [{k: c[k] for k in ("name", "peakStress", "peakAt", "globalPeakStress")} for c in rep["cases"]]}
    print(json.dumps(out["voxel"]), flush=True)

    # --- cut-cell -----------------------------------------------------------
    rule = SubcellRule(study.elastic(mat), a.sub)
    t0 = time.time()
    origin, dims, ind = fine_indicator(mesh, pitch, a.sub)
    t_fill = time.time() - t0
    fill = ind.mean(1)
    centre = np.zeros(len(fill), bool)
    # the centre test of verify.py (the voxel today), for stair runs on the same grid
    from loadpath.verify import voxelize
    sol, org2 = voxelize(mesh, pitch, box=(origin + 1e-9, origin + dims * pitch - 1e-9))
    centre = sol.ravel()
    surf, _ = trimesh.sample.sample_surface_even(mesh, 40_000, seed=1)
    edges = _user_edges(study, band)
    res = {"fillSeconds": t_fill}
    for geom, elem in (("stair", "std"), ("stair", "im"), ("cutQ", "std"), ("cutQ", "im")):
        t0 = time.time()
        sel = centre if geom == "stair" else fill >= a.tau
        grid = sel.reshape(dims)
        lbl, _ = ndimage.label(grid)
        ijk = np.argwhere(grid)
        hm = HexMesh(ijk, pitch, origin)
        tol = 0.5 * band + 0.5 * pitch
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=tol)
        keep_lbl = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
        keep = np.isin(lbl[tuple(ijk.T)], keep_lbl)
        flat = np.ravel_multi_index(ijk[keep].T, tuple(dims))
        ijk = ijk[keep]
        model = CutModel(ijk, pitch, origin, study.elastic(mat), mat.E, ind=ind[flat], geometry=geom,
                         element=elem, rule=rule)
        hm = model.mesh
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=tol)
        spreads = bc.spreads or [None] * len(study.load_cases)
        rhs, layout = expand(bc.loads, spreads, None)
        st, k = model.stiffness(bc.fixed)
        if bc.springs is not None:
            k = (k + mat.E * bc.springs).tocsr()
        t_setup = time.time() - t0
        us = model.solve_many(k, rhs)
        t_solve = time.time() - t0
        cases = []
        for ci, case in enumerate(study.load_cases):
            idx = layout[ci]
            assert len(idx) == 1, "uncertain-direction loads not handled here"
            f = rhs[idx[0]]
            u = us[idx[0]]
            info = 0
            model.u = u
            contact = bc.support_nodes | bc.case_load_nodes[ci]
            ctree = cKDTree(hm.node_xyz[contact])
            near_n = ctree.query(hm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
            near_s = ctree.query(surf, k=1, distance_upper_bound=band)[0] < np.inf
            if edges is not None:
                near_n |= edges(hm.node_xyz)
                near_s |= edges(surf)
            vm_n = von_mises(model.nodal_stress(weight_by_fill=(geom != "stair")))
            body = ~near_n
            i_n = int(np.argmax(np.where(body, vm_n, -1)))
            row = {"name": case.name, "cg": int(info), "nodal": float(vm_n[i_n]), "nodalAt": hm.node_xyz[i_n].tolist()}
            for key, mf, reach in (("surfD", 0.0, 1.0), ("surfX", 0.5, 2.0)):
                v = von_mises(model.surface_stress(surf, min_fill=mf, reach=reach))
                v = np.where(near_s | ~np.isfinite(v), -1, v)
                j = int(np.argmax(v))
                row[key], row[key + "At"] = float(v[j]), surf[j].tolist()
                row[key + "P999"] = float(np.percentile(v[v >= 0], 99.9))
            row["compliance"] = float(f @ u)
            cases.append(row)
        res[f"{geom}/{elem}"] = {"cells": int(hm.n_elem), "setupSeconds": t_setup, "solveSeconds": t_solve, "seconds": time.time() - t0,
                                 "cases": cases}
        print(geom, elem, json.dumps(res[f"{geom}/{elem}"]), flush=True)
    out["cut"] = res

    if not a.no_tet:
        t0 = time.time()
        r = xc.crosscheck(study, mesh, size=tet_size, contact_band=band)
        out["tet"] = {"seconds": time.time() - t0, "elements": r["elements"], "elementSize": r["elementSize"],
                      "cases": [{k: c[k] for k in ("name", "peakStress", "peakAt")} for c in r["cases"]]}
        print(json.dumps(out["tet"]), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
