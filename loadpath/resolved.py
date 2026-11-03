"""
Full-resolution FE of a field graph's lattice part: every strut and sheet
voxelized at a pitch a few voxels across the thinnest wall and solved as
it is, no homogenization.

homogenized_check reads a lattice from its unit cell's tensors, which is
quick and right inside the lattice. Where cells are cut open at a bare face
(no skin), how much they keep depends on how the face cuts them: cells cut
square keep struts running along the face, cells turned and cut on the bias
leave dangling stubs. One knockdown cannot serve both, and an optimizer
comparing the two (lattice_orient) takes the difference for a gain:
bench/results/lattice-orient-confirm.json. `resolved_check` is the arbiter
for those comparisons. It costs minutes (0.5-1M voxels for a 48 mm bar of
6 mm cells at 0.15 mm), so it confirms a pick rather than drives a search.

The peak is read as bench/lattice_aniso_validate.py reads it and the unit-
cell tables are built: each lattice cell's 99.5th percentile of nodal von
Mises, the worst cell outside the contact band.
"""
from __future__ import annotations

import time

import numpy as np

__all__ = ["default_pitch", "resolved_check"]

# voxels across the thinnest wall or strut the default pitch aims at
VOXELS_PER_WALL = 6
MAX_CELLS = 4_000_000
# up to this many voxels the matrix is assembled and solved by AMG (about
# 13 GB at 1M cells), above it the matrix-free grid multigrid
ASSEMBLED_MAX = 600_000


def _lattice_spec(g, name):
    from .homogenized import lattice_nodes

    lats = lattice_nodes(g, name)
    return (lats[0], g.nodes[lats[0]]) if len(lats) == 1 else (None, None)


def default_pitch(doc, node=None, base_dir=None):
    """Pitch (mm) VOXELS_PER_WALL voxels across the lattice's constant wall
    (or strut); a fortieth of its cell when the wall is graded."""
    from .fields import FieldGraph, expand_groups

    full = expand_groups(doc)
    name = node or full.get("output") or list(full["nodes"])[-1]
    g = FieldGraph({**full, "output": name}, base_dir=base_dir)
    _, nd = _lattice_spec(g, name)
    if nd is None:
        raise ValueError("resolved_check needs a pitch for a part with no single lattice node")
    t, cell = nd.spec.get("thickness"), nd.spec.get("cell", 8.0)
    if isinstance(t, (int, float)) and "density" not in nd.spec:
        return float(t) / VOXELS_PER_WALL
    return float(np.min(np.atleast_1d(cell))) / 40.0 if isinstance(cell, (int, float, list)) else 0.15


def resolved_check(doc, study, pitch: float | None = None, node=None, base_dir=None, sources=None,
                   contact_band: float | None = None, cell_size=None, max_cells: int = MAX_CELLS,
                   rtol: float = 1e-7, keep_mesh: bool = False, solver: str = "auto") -> dict:
    """Voxel FE of the graph's part at `pitch` (default_pitch), its lattice
    resolved. Returns {mass, volume, safetyFactor, cases (name, compliance,
    deflection, peakStress, peakAt, peakFaceDistance, peak995), cells,
    pitch, seconds}; with keep_mesh also hm, bc. `contact_band` (mm, default
    one lattice cell): stress this close to a support or load is left out.
    `cell_size` (mm, scalar or per axis): the bins the peak is read over,
    default the lattice's cell. Refuses grids over `max_cells`. `solver`:
    "mg" (matrix-free grid multigrid), "assembled" (AMG on the assembled
    matrix), "auto" assembled up to ASSEMBLED_MAX voxels."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    from .domain import boundary_conditions
    from .fem import CORNERS, HexMesh, elastic_c, hex8_b, von_mises
    from .fields import FieldGraph, expand_groups
    from .mg import GridElasticity
    from .verify import STRESS_ALLOWANCE

    t0 = time.time()
    full = expand_groups(doc)
    name = node or full.get("output") or list(full["nodes"])[-1]
    g = FieldGraph({**full, "output": name}, base_dir=base_dir, sources=sources)
    lat, nd = _lattice_spec(g, name)
    if pitch is None:
        pitch = default_pitch(doc, node=node, base_dir=base_dir)
    pitch = float(pitch)
    if cell_size is None:
        c = nd.spec.get("cell", 8.0) if nd is not None else 8.0
        c = float(np.min(np.atleast_1d(c))) if isinstance(c, (int, float, list)) else 8.0
        cell_size = np.asarray(nd.spec.get("aspect", (1, 1, 1)), float) * c if nd is not None else c
    if contact_band is None:
        contact_band = float(np.min(np.atleast_1d(cell_size)))
    b = g.bounds(name)
    if b is None:
        raise ValueError(f"{name!r} is unbounded")
    lo, hi = (np.asarray(v, float) for v in b)
    dims = np.ceil((hi - lo) / pitch - 1e-9).astype(int)
    if np.prod(dims.astype(float)) * 0.6 > max_cells:
        raise ValueError(f"a resolved check at {pitch:g} mm is ~{np.prod(dims.astype(float)) * 0.6:.3g} voxels "
                         f"(max_cells {max_cells}): coarsen the pitch or check a smaller part")
    origin = lo
    keep = []
    jj, kk = np.meshgrid(np.arange(dims[1]), np.arange(dims[2]), indexing="ij")
    for i in range(dims[0]):
        P = np.stack([np.full(jj.size, origin[0] + (i + 0.5) * pitch), origin[1] + (jj.ravel() + 0.5) * pitch,
                      origin[2] + (kk.ravel() + 0.5) * pitch], 1)
        s = np.asarray(g.evaluate(P, name), float) < 0
        keep.append(np.stack([np.full(s.sum(), i), jj.ravel()[s], kk.ravel()[s]], 1))
    ijk = np.concatenate(keep)
    grid = np.zeros(dims, bool)
    grid[tuple(ijk.T)] = True
    lbl, _ = ndimage.label(grid)
    hm = HexMesh(ijk, pitch, origin)
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    # only what is joined to a support carries load
    sup = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    sel = np.isin(lbl[tuple(ijk.T)], sup)
    if not sel.all():
        ijk = ijk[sel]
        hm = HexMesh(ijk, pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    mat = study.material
    nu = study.elastic(mat)
    rhs = [bc.loads[ci] for ci in range(len(study.load_cases))]
    if solver == "auto":
        solver = "mg" if hm.n_elem > ASSEMBLED_MAX else "assembled"
    if solver == "assembled":
        # smoothed-aggregation AMG on the assembled matrix: on bias-cut
        # lattices (struts ending in thin stubs at every face) the grid
        # multigrid's CG crawls (a turned BCCz bar took 5x as long as laid)
        from .fem import Solver, Stiffness

        stiff = Stiffness(hm, nu, bc.fixed)
        k = stiff.assemble(np.full(hm.n_elem, mat.E))
        if bc.springs is not None:
            k = (k + mat.E * bc.springs).tocsr()
        us = Solver(hm, rtol=rtol, maxiter=4000).solve_many(k, rhs)
    else:
        grid_solver = GridElasticity(hm, nu, bc.fixed, springs=mat.E * bc.springs if bc.springs is not None
                                     else None, rtol=rtol, maxiter=4000)
        grid_solver.set_moduli(np.full(hm.n_elem, mat.E))
        us = grid_solver.solve_many(rhs)
    c0 = elastic_c(mat.E, nu)
    bcs = [hex8_b(a, b_, c, nu) / pitch for (a, b_, c) in 2.0 * CORNERS - 1.0]
    counts = np.bincount(hm.enode.ravel(), minlength=hm.n_nodes)[:, None]
    xyz = hm.node_xyz
    dface = np.minimum(xyz - lo, hi - xyz).min(axis=1)
    cases, worst = [], np.inf
    for ci, case in enumerate(study.load_cases):
        u = us[ci]
        ue = u[hm.edof]
        nodal = np.zeros((hm.n_nodes, 6))
        for k in range(8):
            np.add.at(nodal, hm.enode[:, k], (ue @ bcs[k].T) @ c0.T)
        vm = von_mises(nodal / counts)
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        far = cKDTree(xyz[contact]).query(xyz, k=1, distance_upper_bound=contact_band)[0] == np.inf
        if not far.any():
            far = np.ones(len(xyz), bool)
        key = np.floor((xyz[far] - origin) / np.asarray(cell_size, float)).astype(int)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        v = vm[far]
        order = np.argsort(inv, kind="stable")
        edges = np.searchsorted(inv[order], np.arange(inv.max() + 2))
        groups = [order[edges[j]:edges[j + 1]] for j in range(len(edges) - 1) if edges[j + 1] - edges[j] > 50]
        if not groups:
            groups = [np.arange(len(v))]
        peaks = np.array([np.percentile(v[gr], 99.5) for gr in groups])
        j = int(np.argmax(peaks))
        top = groups[j][np.argmax(v[groups[j]])]
        peak = float(peaks[j])
        sf = mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else np.inf
        worst = min(worst, sf)
        ln = bc.case_load_nodes[ci]
        disp = u.reshape(-1, 3)
        cases.append({"name": case.name, "compliance": float(bc.loads[ci] @ u),
                      "deflection": float(np.linalg.norm(disp[ln].mean(axis=0))) if ln.any() else 0.0,
                      "peakStress": peak, "safetyFactor": sf, "peakAt": xyz[far][top].tolist(),
                      "peakFaceDistance": float(dface[far][top]), "peak995": float(np.percentile(v, 99.5))})
    volume = float(hm.n_elem * pitch ** 3)
    rep = {"mass": volume * mat.density / 1000.0, "volume": volume, "safetyFactor": worst, "cases": cases,
           "cells": int(hm.n_elem), "pitch": pitch, "lattice": lat, "stressAllowance": STRESS_ALLOWANCE,
           "seconds": time.time() - t0}
    if keep_mesh:
        rep["hm"], rep["bc"] = hm, bc
    return rep
