"""
A live stress preview for the field editor: the part straight from its
field graph on a coarse grid, solved in well under a second or two, so a
heatmap can follow the sliders.

No mesh is made. The graph's distance field is sampled at cell centres and
each cell's filled share (half a pitch either side of the surface) sets its
stiffness, as `verify_part(fractional=True)` does, so the reading moves
smoothly with a slider rather than in whole-cell steps. Supports and loads
come from the study (domain.boundary_conditions, the same problem the
optimizer and the check solve); the solve is the matrix-free multigrid.
The two highest spots per load case are re-solved finer (_submodel) and
the peak is multiplied by HOTSPOT_FACTOR, calibrated so the preview's
safety factor reads under the full check's on every part of
bench/preview_calibration.py (0.72-0.95 of it). What it returns is still
a preview: no thickness or printability checks, no across-layer check,
and a coarse grid. A Run (verify_part on the meshed part) is the check.
"""
from __future__ import annotations

import base64
import time

import numpy as np
from scipy import ndimage

PREVIEW_CELLS = 16_000
# hot spots per load case re-solved finer (_submodel), and how
REFINE = 2
SUB_HALF = 3.0          # box half-width, coarse cells
SUB_RATIO = 2.5         # coarse pitch / submodel pitch
HOTSPOT_FACTOR = 1.3    # the refined peak times this: bench/preview_calibration.py, docs/fields.md


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


def _hotspots(xyz, vm, body, sep, count):
    """The `count` highest nodes in the body, each at least `sep` from the
    ones before it."""
    order = np.argsort(np.where(body, vm, -1.0))[::-1]
    out = []
    for i in order[:5000]:
        if not body[i]:
            break
        if all(np.linalg.norm(xyz[i] - c) > sep for c in out):
            out.append(xyz[i])
            if len(out) == count:
                break
    return out


def _grid(lo, hi, pitch):
    """Cells on multiples of the pitch, a cell of margin each side, so a
    slider's neighbouring values land on one lattice and compare fairly."""
    a = np.floor(lo / pitch).astype(int) - 1
    b = np.ceil(hi / pitch).astype(int) + 1
    return a * pitch, np.maximum(b - a, 1)


def _submodel(g, name, study, mat, hm, u, center, half, h, contact_xyz, band, margin):
    """The peak von Mises in a box around a hot spot, re-solved at pitch `h`
    from the field, driven on the box's faces by the
    coarse displacements (verify._submodel_peak's submodelling, without a
    mesh). Returns (MPa, position) or None."""
    from scipy.sparse.linalg import splu
    from scipy.spatial import cKDTree

    from .fem import HexMesh, Stiffness
    from .verify import nodal_von_mises

    dims = np.full(3, max(int(np.ceil(2 * half / h)), 2))
    origin = np.asarray(center, float) - 0.5 * dims * h
    # whole cells here, as the check's own submodels are: on the arm link a
    # filled-share box closed 5% of the gap to the check, a whole-cell one 15%
    fill = (_fill(g, name, origin, dims, h) >= 0.5).astype(float)
    solid = fill > 0
    if solid.sum() < 64:
        return None
    sm = HexMesh(np.argwhere(solid), h, origin)
    blo, bhi = origin, origin + dims * h
    eps = 0.25 * h
    cut = np.any((sm.node_xyz < blo + eps) | (sm.node_xyz > bhi - eps), axis=1)
    near_contact = np.zeros(sm.n_nodes, bool)
    if len(contact_xyz):
        near_contact = cKDTree(contact_xyz).query(sm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
    driven = cut | near_contact
    if driven.all() or not driven.any():
        return None
    # material that reaches no driven node would float: left out
    lbl, _ = ndimage.label(solid)
    ijk = np.argwhere(solid)
    held = np.unique(lbl[tuple(ijk[driven[sm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(ijk.T)], held)
    if not keep.all():
        ijk = ijk[keep]
        sm = HexMesh(ijk, h, origin)
        cut = np.any((sm.node_xyz < blo + eps) | (sm.node_xyz > bhi - eps), axis=1)
        near_contact = (cKDTree(contact_xyz).query(sm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
                        if len(contact_xyz) else np.zeros(sm.n_nodes, bool))
        driven = cut | near_contact
    gd, gi = cKDTree(hm.node_xyz).query(sm.node_xyz[driven], k=8)
    w = 1.0 / np.maximum(gd, 1e-9) ** 2
    w /= w.sum(axis=1, keepdims=True)
    up = (w[..., None] * u.reshape(-1, 3)[gi]).sum(axis=1).ravel()
    pd = np.repeat(driven, 3)
    st = Stiffness(sm, study.elastic(mat), np.zeros(sm.n_dof, bool))
    k = st.assemble(mat.E * fill[tuple(ijk.T)]).tocsr()
    free = ~pd
    kff = k[free][:, free].tocsc()
    rhs = -(k[free][:, pd] @ up)
    try:
        uf = splu(kff, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True}).solve(rhs)
    except RuntimeError:
        return None
    us = np.zeros(sm.n_dof)
    us[pd], us[free] = up, uf
    vm = nodal_von_mises(st, us, mat.E)
    inner = np.all((sm.node_xyz > blo + margin) & (sm.node_xyz < bhi - margin), axis=1) & ~near_contact
    if not inner.any():
        return None
    i = int(np.argmax(np.where(inner, vm, -1.0)))
    return float(vm[i]), sm.node_xyz[i].tolist()


class PreviewState:
    """What one editor session's previews carry between slider moves: the
    grid pitch, kept while the cell count stays near the target. (Starting
    each solve from the last one's displacements was tried: on the arm link
    it saved no CG iterations, 10 either way, and the mapping cost 0.2 s.)"""

    def __init__(self):
        self.pitch = None


def stress_preview(doc, base_dir=None, study=None, graph=None, cells=PREVIEW_CELLS, pitch=None, state=None,
                   device=None, refine=REFINE):
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

    mat = study.material
    from .directions import expand
    from .mg import GridElasticity

    def setup(origin, fill, pitch):
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
        return hm, bc, fill[tuple(ijk.T)]

    # The multigrid's coarse levels keep only the fine level's even nodes,
    # and on some grids a thin support's fixed nodes all sit on odd ones: the
    # coarsest level floats and factors singular, or CG stalls (an
    # L-bracket held by one face: 9 iterations at pitch 1.12 and 1.13, none
    # converging in 2000 at 1.127). It converges in 8-13 iterations when it
    # works, so 50 is the cap; a grid 2% coarser falls differently;
    # after two of those the assembled solve (5-80 s) is the last resort.
    disp, base = None, (pitch, origin, dims, fill)
    for attempt in range(3):
        t1 = time.time()
        if attempt:
            pitch *= 1.02
            origin, dims, fill = sample(pitch)
        hm, bc, share = setup(origin, fill, pitch)
        spreads = bc.spreads or [None] * len(study.load_cases)
        rhs, layout = expand(bc.loads, spreads, None)
        timing["setup"] = timing.get("setup", 0.0) + time.time() - t1
        t2 = time.time()
        springs = mat.E * bc.springs if bc.springs is not None else None
        try:
            stiff = GridElasticity(hm, study.elastic(mat), bc.fixed, springs=springs, rtol=1e-3, maxiter=50,
                                   device=device)
            stiff.set_moduli(mat.E * share)
            disp = stiff.solve_many(rhs)
            iters = stiff.last_iters
        except RuntimeError:
            disp = None
        timing["solve"] = timing.get("solve", 0.0) + time.time() - t2
        if disp is not None:
            break
    if disp is None:
        # on the grid it started from, so a slider's next value lands on it too
        from .fem import Solver, Stiffness

        t2 = time.time()
        pitch, origin, dims, fill = base
        hm, bc, share = setup(origin, fill, pitch)
        rhs, layout = expand(bc.loads, bc.spreads or [None] * len(study.load_cases), None)
        springs = mat.E * bc.springs if bc.springs is not None else None
        stiff = Stiffness(hm, study.elastic(mat), bc.fixed)
        k = stiff.assemble(mat.E * share)
        if springs is not None:
            k = (k + springs).tocsr()
        solver = Solver(hm, rtol=1e-4, maxiter=2000)
        disp = solver.solve_many(k, rhs)
        iters = solver.last_iters
        timing["solve"] += time.time() - t2
    if state is not None:
        state.pitch = pitch

    t3 = time.time()
    from scipy.spatial import cKDTree

    allowed = mat.strength / STRESS_ALLOWANCE
    util = np.zeros(hm.n_nodes)
    band = 1.5 * pitch
    cases, worst_sf, worst_defl, peak_util, worst_raw, worst_ref = [], np.inf, 0.0, 0.0, np.inf, np.inf
    for ci, case in enumerate(study.load_cases):
        idx = [i for i in layout[ci] if i is not None]
        # a load of unknown direction: each axis's solution, the worst at each node
        vm = np.max([nodal_von_mises(stiff, disp[i], mat.E) for i in idx], axis=0)
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        near = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1, distance_upper_bound=band)[0] < np.inf
        body = ~near if (~near).any() else np.ones_like(near)
        peak = float(vm[body].max())
        raw_peak = peak
        if refine:
            u_worst = disp[idx[int(np.argmax([nodal_von_mises(stiff, disp[i], mat.E)[body].max() for i in idx]))]] \
                if len(idx) > 1 else disp[idx[0]]
            for c in _hotspots(hm.node_xyz, vm, body, SUB_HALF * pitch, refine):
                r = _submodel(g, name, study, mat, hm, u_worst, c, SUB_HALF * pitch, pitch / SUB_RATIO,
                              hm.node_xyz[contact], band, 0.75 * pitch)
                if r is not None and r[0] > peak:
                    peak = r[0]
        raw_sf = allowed / raw_peak if raw_peak > 0 else float("inf")
        refined_sf = allowed / peak if peak > 0 else float("inf")
        sf = refined_sf / HOTSPOT_FACTOR
        ln = bc.case_load_nodes[ci]
        defl = max(float(np.linalg.norm(disp[i].reshape(-1, 3)[ln].mean(axis=0))) for i in idx) if ln.any() else 0.0
        cases.append({"name": case.name, "safetyFactor": sf, "rawSafetyFactor": raw_sf,
                      "refinedSafetyFactor": refined_sf, "deflection": defl})
        worst_sf, worst_defl = min(worst_sf, sf), max(worst_defl, defl)
        worst_raw, worst_ref = min(worst_raw, raw_sf), min(worst_ref, refined_sf)
        util = np.maximum(util, vm / allowed)
        peak_util = max(peak_util, raw_peak / allowed)          # the map is the coarse solve
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
    return {"safetyFactor": worst_sf, "rawSafetyFactor": worst_raw, "refinedSafetyFactor": worst_ref,
            "hotspotFactor": HOTSPOT_FACTOR, "deflection": worst_defl,
            "mass": float(share.sum()) * pitch ** 3 * mat.density / 1000.0,
            "material": mat.name, "strength": mat.strength,
            "goal": {"safetyFactor": study.goal.safety_factor, "maxDeflection": study.goal.max_deflection},
            "cells": int(hm.n_elem), "pitch": pitch, "peak": peak_util, "allowed": allowed, "iterations": int(iters), "cases": cases,
            "lo": origin.tolist(), "hi": (origin + dims * pitch).tolist(), "dims": [int(x) for x in ndims],
            "data": base64.b64encode(data.tobytes()).decode("ascii"),
            "timing": {k: round(v, 3) for k, v in timing.items()}, "seconds": round(time.time() - t0, 3)}
