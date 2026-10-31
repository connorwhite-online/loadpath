"""
Which way up to print it.

Orientation decides three things on a layered printer, and they pull
against each other:

  strength  - FDM layers bond at 50-70% of the in-plane strength, so a part
              whose main stress runs across its layers is a weaker part.
  supports  - every overhang past 45 degrees costs support material, time
              and a scarred surface.
  the bed   - the first layer has to hold: enough contact, no tipping.

Two entry points, for the two moments the choice is made:

choose_build_direction(study)
    Before designing. Designs the study for each candidate direction on a
    coarse grid at a fixed volume fraction, each with that direction's
    overhang control, by successive halving (all of them half way and
    checked cheaply, the leaders finished), and keeps the direction whose
    verified safety factor per gram (layered stiffness and interlayer bond
    included) is highest; support volume and height break ties. Because the
    design is grown for its direction it is self-supporting in all of them,
    so this is mostly a strength choice: which way can material carry the
    loads in its layers' plane. A study opts in with
    "process": {"type": "fdm", "buildDirection": "auto"}.

rank_orientations(mesh, study=None)
    After designing, or for any mesh. Tries the six axis directions and the
    part's own large flat faces (convex hull facets, the ways it can rest on
    the bed), slices each (printcheck.check_print) and, given the study, reads
    the interlayer tension each direction would see from one stress solve.
    Ranks by: meets the interlayer safety factor, then support volume,
    height and bed risk.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .am import build_vector, direction_label, parse_build_direction
from .printcheck import check_print

__all__ = ["choose_build_direction", "rank_orientations", "AXIS_DIRECTIONS"]

AXIS_DIRECTIONS = [(2, 1), (2, -1), (0, 1), (0, -1), (1, 1), (1, -1)]
# Coarse comparison budget per direction (cells) and the fraction tried
# when the goal has none of its own.
_COMPARE_CELLS = 12_000
_COMPARE_VF = 0.25
# Successive halving (choose_build_direction): every direction runs this
# many iterations and is checked on a grid of _SCREEN_CELLS without
# hotspot refinement; at most _FINALISTS go on to the full count, and none
# below _FINALIST_FLOOR of the leader's strength per gram.
_SCREEN_ITERATIONS = 40
_SCREEN_CELLS = 15_000
_FINALISTS = 3
_FINALIST_FLOOR = 0.35


def _cost(pc: dict) -> float:
    """Print cost of one orientation, dimensionless: support as a fraction of
    the part's volume, height over the part's size (time), bed risks."""
    vol = max(pc["volume"], 1e-9)
    size = vol ** (1.0 / 3.0)
    bed = pc["bed"]
    c = pc["supportVolume"] / vol + 0.15 * pc["height"] / size
    if bed["comMargin"] < 0:
        c += 1.0
    if bed["feet"] and bed["smallestFoot"] < 9.0:
        c += 0.2
    c += 0.1 * max(0.0, bed["aspect"] - 3.0)
    return float(c)


def _stress_nodes(study, mesh, max_cells=60_000):
    """Nodal stress tensors of the delivered part, per load case, from one
    isotropic voxel solve (as verify_part runs it)."""
    from scipy import ndimage

    from .domain import boundary_conditions, gravity_force
    from .fem import HexMesh, Solver, Stiffness
    from .verify import nodal_stress, voxelize

    mat = study.material
    vol = float(np.prod(mesh.extents))
    pitch = (vol / max_cells) ** (1.0 / 3.0)
    solid, origin = voxelize(mesh, pitch)
    lbl, _ = ndimage.label(solid, structure=np.ones((3, 3, 3), bool))
    ijk = np.argwhere(solid)
    hm = HexMesh(ijk, pitch, origin)
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=pitch)
    keep_lbl = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(ijk.T)], keep_lbl)
    if not keep.all():
        hm = HexMesh(ijk[keep], pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=pitch)
    st = Stiffness(hm, mat.nu, bc.fixed)
    k = st.assemble(np.full(hm.n_elem, mat.E))
    if bc.springs is not None:
        k = (k + mat.E * bc.springs).tocsr()
    fg = gravity_force(hm, np.ones(hm.n_elem), bc.gravity, bc.fixed) if bc.gravity is not None else 0.0
    us = Solver(hm, rtol=1e-6, maxiter=3000).solve_many(k, [f + fg for f in bc.loads])
    from scipy.spatial import cKDTree
    contact = np.zeros(hm.n_nodes, bool)
    contact |= bc.support_nodes
    for ln in bc.case_load_nodes:
        contact |= ln
    band = 2.0 * pitch
    body = cKDTree(hm.node_xyz[contact]).query(hm.node_xyz, k=1, distance_upper_bound=band)[0] == np.inf
    return [nodal_stress(st, u, mat.E)[body] for u in us]


def _normal_stress(sig, n):
    return (sig[:, :3] @ (n * n)
            + 2.0 * (sig[:, 3] * n[0] * n[1] + sig[:, 4] * n[1] * n[2] + sig[:, 5] * n[2] * n[0]))


def _facet_directions(mesh, count=8, min_area_frac=0.03):
    """Build directions that put a large flat face of the convex hull on the
    bed (up = minus that face's outward normal)."""
    hull = mesh.convex_hull
    normals = hull.face_normals
    areas = hull.area_faces
    # merge coplanar hull triangles by rounded normal
    key = np.round(normals, 3)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    tot = np.bincount(inv.ravel(), weights=areas)
    order = np.argsort(-tot)
    out = []
    for i in order[:count]:
        if tot[i] < min_area_frac * hull.area:
            break
        n = uniq[i] / np.linalg.norm(uniq[i])
        out.append(parse_build_direction(list(-n)))
    return out


def _same(a, b):
    return float(build_vector(a) @ build_vector(b)) > 1 - 1e-6


def rank_orientations(mesh, study=None, candidates=None, process: str = "fdm", interlayer: float | None = None,
                      safety_factor: float | None = None, pitch: float | None = None,
                      stress_cells: int = 60_000) -> list[dict]:
    """Every candidate build direction with its print check, interlayer
    reading and cost, best first. `study` (optional) supplies loads,
    material, interlayer ratio and goal; without it only printability is
    ranked."""
    cands = list(candidates) if candidates is not None else AXIS_DIRECTIONS + _facet_directions(mesh)
    uniq = []
    for c in cands:
        c = parse_build_direction(c) if not isinstance(c, tuple) or len(c) not in (2, 3) else c
        if not any(_same(c, u) for u in uniq):
            uniq.append(c)
    sigs = None
    mat = None
    if study is not None:
        mat = study.material
        proc = study.process
        if proc is not None:
            process = proc.kind
            interlayer = proc.interlayer if interlayer is None else interlayer
        safety_factor = safety_factor or study.goal.safety_factor
        if study.load_cases:
            sigs = _stress_nodes(study, mesh, stress_cells)
    from .verify import STRESS_ALLOWANCE

    rows = []
    for c in uniq:
        pc = check_print(mesh, c, process=process, pitch=pitch, material=mat)
        row = {"direction": direction_label(c), "build": c, "cost": _cost(pc), "print": pc}
        if sigs is not None and interlayer is not None and interlayer < 1.0:
            n = build_vector(c)
            tens = max(float(np.maximum(_normal_stress(s, n), 0.0).max(initial=0.0)) for s in sigs)
            row["interlayerStress"] = tens
            row["interlayerSafetyFactor"] = (interlayer * mat.strength / (tens * STRESS_ALLOWANCE)
                                             if tens > 0 else float("inf"))
        rows.append(row)
    sf_ok = [r.get("interlayerSafetyFactor", np.inf) >= (safety_factor or 0.0) for r in rows]
    for r, ok in zip(rows, sf_ok):
        r["meetsInterlayer"] = bool(ok)
    rows.sort(key=lambda r: (not r["meetsInterlayer"], r["cost"],
                             -r.get("interlayerSafetyFactor", 0.0)))
    return rows


def _candidate(args):
    """One build direction's design (a process pool worker; module level so
    it pickles): (study, x0, beta0, iterations, verify_cells, refine)."""
    from .domain import build_domain
    from .simp import filter_radius
    from .study import design_at

    st, x0, beta0, iters, cells, refine = args
    domain = build_domain(st)
    o = design_at(st, domain, filter_radius(domain, st.length_scale, st.min_hole), float(st.goal.volume_fraction),
                  x0=x0, beta0=beta0, max_iterations=iters, verify_cells=cells, refine=refine)
    o.xphys = None   # not needed back; keep the pickle small
    return o


def _row(c, o, proc, study):
    rep = o.report
    pc = check_print(o.mesh, c, process=proc.kind, material=study.material)
    inter = [cs.get("interlayerStress") for cs in rep.get("cases", []) if cs.get("interlayerStress") is not None]
    return {"direction": direction_label(c), "build": c, "safetyFactor": rep["safetyFactor"],
            "mass": rep["mass"], "interlayerStress": max(inter) if inter else None,
            "supportVolume": pc["supportVolume"], "supportArea": pc["supportArea"],
            "height": pc["height"], "bed": pc["bed"], "cost": _cost(pc),
            "efficiency": rep["safetyFactor"] / max(rep["mass"], 1e-9)}


def _run_all(jobs, workers):
    if workers <= 1 or len(jobs) == 1:
        return [_candidate(j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_candidate, jobs))


def choose_build_direction(study, candidates=None, max_elements: int = _COMPARE_CELLS, on_progress=None,
                           exhaustive: bool = False) -> dict:
    """Compare build directions for `study` on a coarse grid (see the module
    note). Returns {"study": the study with the winner set as its build
    direction, "chosen": label, "candidates": [{direction, safetyFactor,
    mass, supportVolume, ...}]}. The ranking is stored on the returned study
    too (study.raw["orientation"]).

    With overhang control, by successive halving: every direction is
    designed for _SCREEN_ITERATIONS and checked cheaply (a coarser analysis
    grid, no hotspot refinement); only the best few by strength per gram
    (_FINALISTS) are carried on from where they stopped to the full count
    and verified in full. On the PETG arm link at 12k cells the half-way
    reading put the full search's winner (-x, SF 10.3 against 8.8 for the
    runner-up) first by a factor of two; the full search, every direction
    to 80 iterations, took 44 minutes on 4 cores. (Two cheaper screens did
    not hold up: stress read off one shared unconstrained design ranked -x
    among the worst, since the overhang filter reshapes each direction's
    part; and a grid of an eighth of the cells lost the 4 mm bearing ring.)

    Without overhang control (and with isotropic design stiffness) every
    direction's design is the same one, so it is designed once and verified
    once per build axis (opposite directions load the layers alike).

    `exhaustive=True` runs every direction to the full count."""
    from .domain import build_domain
    from .simp import filter_radius
    from .spec import Goal
    from .study import _COARSE_MIN_CELLS, _WARM_BETA, _coarse_start, _workers, design_at

    proc = study.process
    if proc is None:
        raise ValueError("choose_build_direction needs a layered process (\"process\": {\"type\": \"fdm\"})")
    cands = [parse_build_direction(c) if not isinstance(c, tuple) else c for c in (candidates or AXIS_DIRECTIONS)]
    vf = float(study.goal.volume_fraction or _COMPARE_VF)

    def variant(c, coarse=True):
        p = dataclasses.replace(proc, build=c)
        kw = dict(process=p, print_direction=c if proc.overhangs else study.print_direction)
        if coarse:
            kw.update(goal=Goal(volume_fraction=vf), materials=[study.material],
                      max_elements=min(study.max_elements, max_elements), pitch=None,
                      max_iterations=min(study.max_iterations, 80))
        return dataclasses.replace(study, **kw)

    def emit(ev):
        if on_progress is not None:
            on_progress({"stage": "orient", **ev}, None)

    emit({"candidates": [direction_label(c) for c in cands]})
    rows = []
    if not proc.overhangs and not proc.anisotropic:
        # one design serves every direction: verify it once per build axis
        # and print-check each
        from .verify import verify_part

        st0 = variant(cands[0])
        domain = build_domain(st0)
        radius = filter_radius(domain, st0.length_scale, st0.min_hole)
        big = domain.n_elem >= _COARSE_MIN_CELLS
        x0 = _coarse_start(st0, domain, vf, radius, None) if big else None
        shared = design_at(st0, domain, radius, vf, x0=x0, beta0=_WARM_BETA if big else 1.0, verify=False)
        by_axis = {}
        for c in cands:
            key = tuple(np.round(np.abs(build_vector(c)), 6))
            if key not in by_axis:
                by_axis[key] = verify_part(variant(c), shared.mesh, min_pitch=0.5 * domain.pitch,
                                           contact_band=domain.pitch)
            rows.append(_row(c, dataclasses.replace(shared, report=by_axis[key]), proc, study) | {"designed": True})
    else:
        full = min(study.max_iterations, 80)
        workers = _workers(len(cands), cells=min(study.max_elements, max_elements))
        if exhaustive or full <= _SCREEN_ITERATIONS:
            outs = _run_all([(variant(c), None, 1.0, full, 60_000, 3) for c in cands], workers)
            rows = [_row(c, o, proc, study) | {"designed": True} for c, o in zip(cands, outs)]
        else:
            first = _run_all([(variant(c), None, 1.0, _SCREEN_ITERATIONS, _SCREEN_CELLS, 0) for c in cands],
                             workers)
            screen = [_row(c, o, proc, study) for c, o in zip(cands, first)]
            emit({"screen": [(r["direction"], round(r["safetyFactor"], 2)) for r in screen]})
            lead = max(r["efficiency"] for r in screen)
            order = sorted(range(len(cands)), key=lambda i: -screen[i]["efficiency"])
            go = [i for i in order[:_FINALISTS] if screen[i]["efficiency"] >= _FINALIST_FLOOR * lead]
            # carry each finalist on from where it stopped: its design and
            # its projection sharpness
            jobs = [(variant(cands[i]), first[i].x, first[i].optimizer.get("beta", _WARM_BETA),
                     full - _SCREEN_ITERATIONS, 60_000, 3) for i in go]
            final = dict(zip(go, _run_all(jobs, workers)))
            for i, c in enumerate(cands):
                half = {"screenSafetyFactor": screen[i]["safetyFactor"], "screenMass": screen[i]["mass"]}
                if i in final:
                    rows.append(_row(c, final[i], proc, study) | {"designed": True} | half)
                else:
                    rows.append({k: v for k, v in screen[i].items() if k not in ("safetyFactor", "efficiency")}
                                | {"designed": False, "safetyFactor": None} | half)
    designed = [r for r in rows if r["designed"]]
    if not designed:
        raise RuntimeError("no build direction produced a design")
    # strength per gram decides; within 5% of the best, print cost does
    top = max(r["efficiency"] for r in designed)
    designed.sort(key=lambda r: (r["efficiency"] < 0.95 * top, r["cost"], -r["efficiency"]))
    dropped = sorted((r for r in rows if not r["designed"]), key=lambda r: -r["screenSafetyFactor"])
    rows = designed + dropped
    best = rows[0]
    chosen = variant(best["build"], coarse=False)
    raw = dict(chosen.raw)
    raw["orientation"] = [{k: v for k, v in r.items() if k != "build"} for r in rows]
    chosen = dataclasses.replace(chosen, raw=raw)
    emit({"chosen": best["direction"],
          "ranking": [(r["direction"], round(r["safetyFactor"], 2)) for r in designed],
          "dropped": [r["direction"] for r in dropped]})
    return {"study": chosen, "chosen": best["direction"], "candidates": raw["orientation"]}
