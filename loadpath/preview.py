"""
A live stress preview for the field editor: the part straight from its
field graph on a coarse grid, solved in well under a second or two, so a
heatmap can follow the sliders.

No mesh is made. The graph's distance field is sampled at cell centres and
each cell's filled share (half a pitch either side of the surface) sets its
stiffness, as `verify_part(fractional=True)` does, so the reading moves
smoothly with a slider rather than in whole-cell steps. Supports and loads
come from the study (domain.boundary_conditions, the same problem the
optimizer and the check solve); the solve is the assembled multigrid
Solver, or the matrix-free one on a big grid. What it returns is a
preview: no hot-spot submodels, no thickness or printability checks, no
across-layer check, and a coarse grid. A Run (verify_part on the meshed
part) is the check.
"""
from __future__ import annotations

import base64
import time

import numpy as np
from scipy import ndimage

PREVIEW_CELLS = 16_000


def study_ref(doc):
    """The study the graph is analyzed under: its tune block's, else the
    first `result` node's."""
    t = doc.get("tune")
    if isinstance(t, dict) and isinstance(t.get("study"), str):
        return t["study"]
    for spec in (doc.get("nodes") or {}).values():
        if isinstance(spec, dict) and spec.get("op") == "result" and isinstance(spec.get("study"), str):
            return spec["study"]
    return None


def _fill(g, name, lo, dims, pitch):
    """Each cell's filled share from the distance at its centre."""
    axes = [lo[i] + (np.arange(dims[i]) + 0.5) * pitch for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d = np.concatenate([g.evaluate(P[i:i + 400_000], name) for i in range(0, len(P), 400_000)])
    d = np.nan_to_num(d, nan=1e3).reshape(tuple(dims))
    return np.clip(0.5 - d / pitch, 0.0, 1.0)


def _grid(lo, hi, pitch):
    """Cells on multiples of the pitch, a cell of margin each side, so a
    slider's neighbouring values land on one lattice and compare fairly."""
    a = np.floor(lo / pitch).astype(int) - 1
    b = np.ceil(hi / pitch).astype(int) + 1
    return a * pitch, np.maximum(b - a, 1)


class PreviewState:
    """What one editor session's previews carry between slider moves: the
    grid pitch, kept while the cell count stays near the target. (Starting
    each solve from the last one's displacements was tried: on the arm link
    it saved no CG iterations, 10 either way, and the mapping cost 0.2 s.)"""

    def __init__(self):
        self.pitch = None


def stress_preview(doc, base_dir=None, study=None, graph=None, cells=PREVIEW_CELLS, pitch=None, state=None,
                   device=None):
    """Coarse stress of the graph's output under its study. Returns
    {"safetyFactor", "deflection" (worst case, mm), "mass" (g), "cells",
    "peak" (the highest stress over allowed outside the contact bands, 1 /
    safetyFactor),
    "pitch", "seconds", "timing" {field, setup, solve, stress},
    "cases" [{name, safetyFactor, deflection}], "lo", "hi", "dims", "data"}:
    `data` is the nodal stress over the allowed stress (strength /
    STRESS_ALLOWANCE, as the check reads it) on the node grid, float16,
    x fastest, base64; 1.0 is where the check's safety factor would be 1.
    `pitch` (or `state`, a PreviewState) reuses a previous preview's grid
    while it lands within a factor of two of `cells`. The solve is the matrix-free
    multigrid (mg.py) on `device` (default LOADPATH_DEVICE: a GPU when torch
    sees one, else torch on the CPU, else numpy)."""
    from .domain import boundary_conditions
    from .fem import HexMesh
    from .fields import FieldGraph, load_study
    from .verify import STRESS_ALLOWANCE, nodal_von_mises

    t0 = time.time()
    timing = {}
    if study is None:
        ref = study_ref(doc)
        if ref is None:
            raise ValueError("the stress preview needs a study: give the graph a tune block or a result node")
        study = load_study(ref, base_dir, doc.get("material"))
    g = graph or FieldGraph(doc, base_dir=base_dir)
    name = doc.get("output")
    b = g.bounds(name)
    if b is None:
        raise ValueError("the part is unbounded: intersect it with a solid")
    lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)

    def sample(h):
        o, dm = _grid(lo, hi, h)
        return o, dm, _fill(g, name, o, dm, h)

    if pitch is None and state is not None:
        pitch = state.pitch
    if pitch:
        origin, dims, fill = sample(float(pitch))
        n = int((fill >= 0.05).sum())
        if not 0.5 * cells <= n <= 2.0 * cells:
            pitch = None
    if not pitch:
        # a first look at a quarter of the cells finds the part's volume
        h0 = (float(np.prod(hi - lo)) / (0.25 * cells)) ** (1.0 / 3.0)
        _, _, f0 = sample(h0)
        vol = max(float(f0.sum()) * h0 ** 3, 1e-9)
        pitch = (vol / cells) ** (1.0 / 3.0)
        origin, dims, fill = sample(pitch)
    pitch = float(pitch)
    timing["field"] = time.time() - t0

    t1 = time.time()
    solid = fill >= 0.05
    lbl, _ = ndimage.label(solid)
    ijk = np.argwhere(solid)
    hm = HexMesh(ijk, pitch, origin)
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    held = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(ijk.T)], held)
    if not keep.any():
        raise ValueError("no material touches a support")
    if not keep.all():
        ijk = ijk[keep]
        hm = HexMesh(ijk, pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    share = fill[tuple(ijk.T)]
    mat = study.material
    from .directions import expand

    spreads = bc.spreads or [None] * len(study.load_cases)
    rhs, layout = expand(bc.loads, spreads, None)
    timing["setup"] = time.time() - t1

    t2 = time.time()
    from .mg import GridElasticity

    springs = mat.E * bc.springs if bc.springs is not None else None
    try:
        stiff = GridElasticity(hm, study.elastic(mat), bc.fixed, springs=springs, rtol=1e-3, maxiter=2000,
                               device=device)
        stiff.set_moduli(mat.E * share)
        disp = stiff.solve_many(rhs)
        iters = stiff.last_iters
    except RuntimeError:
        # a thin part whose coarsest multigrid level loses its supports
        # (a bar two cells deep) factors singular: the assembled solve
        from .fem import Solver, Stiffness

        stiff = Stiffness(hm, study.elastic(mat), bc.fixed)
        k = stiff.assemble(mat.E * share)
        if springs is not None:
            k = (k + springs).tocsr()
        solver = Solver(hm, rtol=1e-4, maxiter=2000)
        disp = solver.solve_many(k, rhs)
        iters = solver.last_iters
    if state is not None:
        state.pitch = pitch
    timing["solve"] = time.time() - t2

    t3 = time.time()
    from scipy.spatial import cKDTree

    allowed = mat.strength / STRESS_ALLOWANCE
    util = np.zeros(hm.n_nodes)
    band = 1.5 * pitch
    cases, worst_sf, worst_defl, peak_util = [], np.inf, 0.0, 0.0
    for ci, case in enumerate(study.load_cases):
        idx = [i for i in layout[ci] if i is not None]
        # a load of unknown direction: each axis's solution, the worst at each node
        vm = np.max([nodal_von_mises(stiff, disp[i], mat.E) for i in idx], axis=0)
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        near = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
        body = ~near if (~near).any() else np.ones_like(near)
        peak = float(vm[body].max())
        sf = allowed / peak if peak > 0 else float("inf")
        ln = bc.case_load_nodes[ci]
        defl = max(float(np.linalg.norm(disp[i].reshape(-1, 3)[ln].mean(axis=0))) for i in idx) if ln.any() else 0.0
        cases.append({"name": case.name, "safetyFactor": sf, "deflection": defl})
        worst_sf, worst_defl = min(worst_sf, sf), max(worst_defl, defl)
        util = np.maximum(util, vm / allowed)
        peak_util = max(peak_util, peak / allowed)
    # onto the node grid; empty nodes take their nearest node's value so the
    # GPU's trilinear filter doesn't darken the surface
    ndims = dims + 1
    nijk = np.rint((hm.node_xyz - origin) / pitch).astype(int)
    grid = np.zeros(tuple(ndims), np.float32)
    have = np.zeros(tuple(ndims), bool)
    grid[tuple(nijk.T)] = util
    have[tuple(nijk.T)] = True
    near_idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
    grid = grid[tuple(near_idx)]
    data = np.ascontiguousarray(np.clip(grid, 0, 6e4).astype(np.float16).transpose(2, 1, 0))
    timing["stress"] = time.time() - t3
    return {"safetyFactor": worst_sf, "deflection": worst_defl,
            "mass": float(share.sum()) * pitch ** 3 * mat.density / 1000.0,
            "material": mat.name, "strength": mat.strength,
            "goal": {"safetyFactor": study.goal.safety_factor, "maxDeflection": study.goal.max_deflection},
            "cells": int(hm.n_elem), "pitch": pitch, "peak": peak_util, "allowed": allowed, "iterations": int(iters), "cases": cases,
            "lo": origin.tolist(), "hi": (origin + dims * pitch).tolist(), "dims": [int(x) for x in ndims],
            "data": base64.b64encode(data.tobytes()).decode("ascii"),
            "timing": {k: round(v, 3) for k, v in timing.items()}, "seconds": round(time.time() - t0, 3)}
