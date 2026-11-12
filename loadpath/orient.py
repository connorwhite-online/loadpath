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
    Before designing. Two ways to print a part in a direction: grown free
    and printed on supports, or grown under that direction's overhang rule
    and printed without them. The free design is one design for every
    direction, so it is designed once on a coarse grid and verified per
    build axis (layered stiffness and interlayer bond included) and
    print-checked per direction. A self-supporting design carries the
    loads no better than the free one in the same direction (the overhang
    rule only takes shapes away), so a direction is designed self-supporting
    only when the free design's cost there, less all of its supports, is
    still under the best supported print. Every option is then priced by
    what it costs to make at equal strength (costing.estimate with the
    process's rates: material scaled to the strength, height, the support's
    own material, machine time and removal), and the cheapest of those that
    come out in one piece and stand on the bed is kept. A study opts in
    with "process": {"type": "fdm", "buildDirection": "auto"}; a study that
    names a material and no process gets it (spec._infer_process).

    Strength per gram alone chose wrong: the LPBF bell crank at 12.5% read
    6% more per gram standing on its 3 mm edge (+y) than flat (+z), on a
    coarse grid, with 1572 mm^3 of support and 66 mm of layers against 193
    mm^3 and 10 mm; designed in full on its edge it read SF 0.60.

rank_orientations(mesh, study=None)
    After designing, or for any mesh. Tries the six axis directions and the
    part's own large flat faces (convex hull facets, the ways it can rest on
    the bed), slices each (printcheck.check_print) and, given the study, reads
    the interlayer tension each direction would see, read as verify reads
    it (verify.interlayer_readings).
    Ranks by: meets the interlayer safety factor, then support volume,
    height and bed risk.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np

from .am import build_vector, direction_label, parse_build_direction
from .printcheck import check_print

__all__ = ["choose_build_direction", "rank_orientations", "settle_supports", "AXIS_DIRECTIONS"]

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
                      stress_cells: int = 60_000, contact_band: float | None = None,
                      analysis_pitch: float | None = None) -> list[dict]:
    """Every candidate build direction with its print check, interlayer
    reading and cost, best first. `study` (optional) supplies loads,
    material, interlayer ratio and goal; without it only printability is
    ranked.

    The interlayer reading is verify's own (verify.interlayer_readings): the
    same analysis grid (stress_cells, no finer than half the study's design
    cell), cut cells, contact band (`contact_band`, default the design cell,
    as `loadpath verify` and the run use) and each direction's layered
    stiffness, so the direction a part was designed and verified for reads
    here what its report reads. `analysis_pitch` pins the grid to the one an
    outcome was verified on (its report's analysis.pitch): a corner's
    stress moves with how the grid falls on it (the coarse wall bracket's
    boss corner read 4.0 and 5.2 MPa across the layers on grids 1% apart),
    so only the same grid reads the same number."""
    cands = list(candidates) if candidates is not None else AXIS_DIRECTIONS + _facet_directions(mesh)
    uniq = []
    for c in cands:
        c = parse_build_direction(c) if not isinstance(c, tuple) or len(c) not in (2, 3) else c
        if not any(_same(c, u) for u in uniq):
            uniq.append(c)
    readings = None
    mat = None
    if study is not None:
        mat = study.material
        proc = study.process
        if proc is not None:
            process = proc.kind
            interlayer = proc.interlayer if interlayer is None else interlayer
        safety_factor = safety_factor or study.goal.safety_factor
        if study.load_cases and proc is not None and interlayer is not None and interlayer < 1.0:
            import dataclasses as _dc

            from . import quick
            from .domain import choose_pitch
            from .verify import interlayer_readings

            design = choose_pitch(study)
            band = design if contact_band is None else contact_band
            st = study
            if proc is not None and proc.interlayer != interlayer:
                st = _dc.replace(study, process=_dc.replace(proc, interlayer=interlayer))
            if analysis_pitch:
                cells, floor = 10 ** 9, float(analysis_pitch)
            else:
                cells, floor = quick.verify_cells(stress_cells), 0.5 * design
            readings = interlayer_readings(st, mesh, uniq, max_cells=cells, min_pitch=floor, contact_band=band)

    rows = []
    for k, c in enumerate(uniq):
        pc = check_print(mesh, c, process=process, pitch=pitch, material=mat)
        row = {"direction": direction_label(c), "build": c, "cost": _cost(pc), "print": pc}
        if readings is not None:
            r = readings[k]
            row["interlayerStress"] = r["interlayerStress"]
            row["interlayerSafetyFactor"] = r["interlayerSafetyFactor"]
            row["interlayerCases"] = r["cases"]
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


def _row(c, o, proc, study, overhangs=None):
    rep = o.report
    pc = check_print(o.mesh, c, process=proc.kind, material=study.material, bridge_span=proc.bridge_span)
    inter = [cs.get("interlayerStress") for cs in rep.get("cases", []) if cs.get("interlayerStress") is not None]
    return {"direction": direction_label(c), "build": c, "safetyFactor": rep["safetyFactor"],
            "mass": rep["mass"], "interlayerStress": max(inter) if inter else None,
            "supportVolume": pc["supportVolume"], "supportArea": pc["supportArea"],
            "trappedSupportArea": pc.get("trappedSupportArea", 0.0), "partVolume": pc["volume"],
            "height": pc["height"], "bed": pc["bed"], "cost": _cost(pc), "bodies": rep.get("bodies", 1),
            "overhangs": bool(proc.overhangs if overhangs is None else overhangs),
            "strength": _strength(study, rep),
            "efficiency": rep["safetyFactor"] / max(rep["mass"], 1e-9)}


def _strength(study, rep) -> float:
    """How far the design is past its goal (study._margin: every limit as a
    ratio that grows about linearly with the material), or its safety factor
    when the goal sets none."""
    from .study import _margin
    try:
        m = _margin(study, rep)
    except (KeyError, TypeError, ValueError):
        m = None
    if m is None or not np.isfinite(m) or m <= 0:
        return max(float(rep["safetyFactor"]), 1e-9)
    return float(m)


def _make_cost(r, study, proc, ref: float) -> float:
    """What the part costs to print in this direction at equal strength
    (costing.estimate, the process's own rates): its material scaled to the
    strength `ref` (a design's strength grows about linearly with its
    material, the same reading the search makes of a margin), its height
    (layers to recoat or lay), and its own supports: their material and
    machine time, and their removal from every supported face. Support
    that cannot come out (under a sealed cavity) makes the direction
    unbuildable. Strength and print cost are then one number in one unit,
    so neither is a tiebreak of the other."""
    from .costing import RATES, estimate
    if r.get("trappedSupportArea"):
        return float("inf")
    kind = proc.kind if proc.kind in RATES else None
    mass = r["mass"] * ref / max(r["strength"], 1e-9)
    vol_cm3 = max(r.get("partVolume") or 0.0, 1e-6) / 1000.0
    density = r["mass"] / vol_cm3 if r.get("mass") else 1.0
    g = {"volume": mass / density, "surface": 0.0, "extent": [0.0, 0.0, 0.0], "box": 0.0,
         "height": float(r["height"])}
    if kind is None:
        return float(r["cost"]) + mass
    name = getattr(study.material, "name", str(study.material))
    est = estimate(kind, name, mass=mass, geometry=g,
                   support={"volume": r["supportVolume"], "area": r["supportArea"]})
    return float(est["unitCost"])


def _sound(r) -> bool:
    """A direction whose design is one piece and stands on the bed. Grown
    under a direction's overhang rule a design can leave a mount it cannot
    reach without an unsupported span (the wall bracket standing on its
    tip, its lower boss loose, read 10x the strength per gram of any real
    part), and a part whose centre of mass is outside its footprint falls
    over before it is finished."""
    bed = r.get("bed") or {}
    return (r.get("bodies") or 1) <= 1 and (bed.get("comMargin") is None or bed["comMargin"] >= 0.0)


def _run_all(jobs, workers):
    if workers <= 1 or len(jobs) == 1:
        return [_candidate(j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from .study import _share_cores
    n = min(workers, len(jobs))
    with _share_cores(n), ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_candidate, jobs))


def choose_build_direction(study, candidates=None, max_elements: int = _COMPARE_CELLS, on_progress=None,
                           exhaustive: bool = False, volume_fraction: float | None = None,
                           self_supporting: bool = True, free_mesh=None) -> dict:
    """Compare build directions for `study` on a coarse grid (see the module
    note). Returns {"study": the study with the winner set as its build
    direction (and process.overhangs False when it is printed on supports),
    "chosen": label, "candidates": [{direction, overhangs, safetyFactor,
    mass, supportVolume, makeCost, ...}]}. The ranking is stored on the
    returned study too (study.raw["orientation"]).

    Self-supporting candidates beyond _FINALISTS go by successive halving:
    designed for _SCREEN_ITERATIONS and checked cheaply (a coarser analysis
    grid, no hotspot refinement), the cheapest few carried on from where
    they stopped to the full count. (Two cheaper screens did not hold up:
    stress read off one shared unconstrained design ranked the PETG arm
    link's best self-supporting direction among the worst, since the
    overhang filter reshapes each direction's part; and a grid of an
    eighth of the cells lost its 4 mm bearing ring.)

    `volume_fraction` is the fraction compared at (default the goal's, or
    _COMPARE_VF); `self_supporting=False` compares only the free design on
    supports (run_study's first pass, settle_supports the second);
    `free_mesh` is that free design when it is already made (the search's
    delivered part), verified here as a coarse one would be.
    `exhaustive=True` runs every direction to the full count."""
    from .domain import build_domain
    from .simp import filter_radius
    from .spec import Goal
    from .study import _COARSE_MIN_CELLS, _WARM_BETA, _coarse_start, _workers, design_at

    proc = study.process
    if proc is None:
        raise ValueError("choose_build_direction needs a layered process (\"process\": {\"type\": \"fdm\"})")
    cands = [parse_build_direction(c) if not isinstance(c, tuple) else c for c in (candidates or AXIS_DIRECTIONS)]
    vf = float(volume_fraction or study.goal.volume_fraction or _COMPARE_VF)

    def variant(c, coarse=True, overhangs=None):
        oh = proc.overhangs if overhangs is None else overhangs
        p = dataclasses.replace(proc, build=c, overhangs=oh)
        kw = dict(process=p, print_direction=c if oh else study.print_direction)
        if coarse:
            kw.update(goal=Goal(volume_fraction=vf), materials=[study.material],
                      max_elements=min(study.max_elements, max_elements), pitch=None,
                      max_iterations=min(study.max_iterations, 80))
        return dataclasses.replace(study, **kw)

    def emit(ev):
        if on_progress is not None:
            on_progress({"stage": "orient", **ev}, None)

    def ranked(rows):
        ok = [r for r in rows if _sound(r)] or rows
        ref = study.goal.safety_factor and 1.0 or max(r["strength"] for r in ok)
        for r in rows:
            r["makeCost"] = _make_cost(r, study, proc, ref)
        return sorted(rows, key=lambda r: (not _sound(r), r["makeCost"], -r["efficiency"]))

    emit({"candidates": [direction_label(c) for c in cands]})
    # Free: the design grown without overhang control is one design for
    # every direction (isotropic design stiffness), printed on supports.
    # Verified once per build axis (opposite directions load the layers
    # alike) and print-checked per direction: its strength there, its
    # height and the supports it needs.
    rows = []
    if not proc.anisotropic:
        from .verify import verify_part

        st0 = variant(cands[0], overhangs=False)
        domain = build_domain(st0)
        radius = filter_radius(domain, st0.length_scale, st0.min_hole)
        big = domain.n_elem >= _COARSE_MIN_CELLS
        if free_mesh is not None:
            shared = types.SimpleNamespace(mesh=free_mesh, report=None)
        else:
            x0 = _coarse_start(st0, domain, vf, radius, None) if big else None
            shared = design_at(st0, domain, radius, vf, x0=x0, beta0=_WARM_BETA if big else 1.0, verify=False)
        by_axis = {}
        for c in cands:
            key = tuple(np.round(np.abs(build_vector(c)), 6))
            if key not in by_axis:
                by_axis[key] = verify_part(variant(c, overhangs=False), shared.mesh, min_pitch=0.5 * domain.pitch,
                                           contact_band=domain.pitch)
            rows.append(_row(c, types.SimpleNamespace(mesh=shared.mesh, report=by_axis[key]), proc, study,
                             overhangs=False)
                        | {"designed": True})
        emit({"supported": [(r["direction"], round(r["safetyFactor"], 2), round(r["supportVolume"]))
                            for r in rows]})
    if (proc.overhangs and self_supporting) or proc.anisotropic:
        # Self-supporting: grown under each direction's overhang rule. It
        # needs no supports but carries the loads no better than the free
        # design in the same direction (the overhang rule only takes shapes
        # away), so its cost there is at least the free design's less that
        # design's supports. A direction whose bound already costs more
        # than the best supported print is not designed.
        bound = {}
        if rows:
            free = ranked([dict(r) for r in rows])
            best_free = min(r["makeCost"] for r in free if _sound(r)) if any(_sound(r) for r in free) else np.inf
            for r in free:
                lb = _make_cost(r | {"supportVolume": 0.0, "supportArea": 0.0, "trappedSupportArea": 0.0},
                                study, proc, study.goal.safety_factor and 1.0 or max(x["strength"] for x in free))
                bound[r["direction"]] = lb
            tries = [c for c in cands if bound[direction_label(c)] < best_free]
        else:
            tries = list(cands)
        emit({"selfSupporting": [direction_label(c) for c in tries]})
        full = min(study.max_iterations, 80)
        workers = _workers(max(len(tries), 1), cells=min(study.max_elements, max_elements))
        if proc.anisotropic and not (proc.overhangs and self_supporting):
            tries = [] if rows else list(cands)
        if not tries:
            pass
        elif exhaustive or full <= _SCREEN_ITERATIONS or len(tries) <= _FINALISTS:
            outs = _run_all([(variant(c), None, 1.0, full, 60_000, 3) for c in tries], workers)
            rows += [_row(c, o, proc, study) | {"designed": True} for c, o in zip(tries, outs)]
        else:
            first = _run_all([(variant(c), None, 1.0, _SCREEN_ITERATIONS, _SCREEN_CELLS, 0) for c in tries],
                             workers)
            screen = ranked([_row(c, o, proc, study) for c, o in zip(tries, first)])
            emit({"screen": [(r["direction"], round(r["safetyFactor"], 2)) for r in screen]})
            index = {direction_label(c): i for i, c in enumerate(tries)}
            go = [index[r["direction"]] for r in screen[:_FINALISTS]]
            jobs = [(variant(tries[i]), first[i].x, first[i].optimizer.get("beta", _WARM_BETA),
                     full - _SCREEN_ITERATIONS, 60_000, 3) for i in go]
            final = dict(zip(go, _run_all(jobs, workers)))
            for i, c in enumerate(tries):
                half = {"screenSafetyFactor": screen[[r["direction"] for r in screen].index(direction_label(c))][
                    "safetyFactor"]}
                if i in final:
                    rows.append(_row(c, final[i], proc, study) | {"designed": True} | half)
                else:
                    srow = next(r for r in screen if r["direction"] == direction_label(c))
                    rows.append({k: v for k, v in srow.items() if k not in ("safetyFactor", "efficiency", "makeCost")}
                                | {"designed": False, "safetyFactor": None} | half)
    designed = [r for r in rows if r["designed"]]
    if not designed:
        raise RuntimeError("no build direction produced a design")
    # one piece standing on the bed first (_sound), then what it costs to
    # make at equal strength (_make_cost)
    designed = ranked(designed)
    dropped = sorted((r for r in rows if not r["designed"]), key=lambda r: -r["screenSafetyFactor"])
    rows = designed + dropped
    best = rows[0]
    chosen = variant(best["build"], coarse=False, overhangs=best["overhangs"])
    raw = dict(chosen.raw)
    raw["orientation"] = [{k: v for k, v in r.items() if k != "build"} for r in rows]
    chosen = dataclasses.replace(chosen, raw=raw)
    emit({"chosen": best["direction"], "selfSupporting": best["overhangs"],
          "ranking": [(r["direction"], "self-supporting" if r["overhangs"] else "on supports",
                       round(r["safetyFactor"], 2), round(r["makeCost"], 1)) for r in designed],
          "dropped": [r["direction"] for r in dropped]})
    return {"study": chosen, "chosen": best["direction"], "candidates": raw["orientation"]}


def _delivered_cost(study, o, proc) -> float:
    """What a delivered outcome costs to make as it stands (_make_cost at
    its own strength): its mass, its height and its own supports along its
    study's build direction."""
    pc = check_print(o.mesh, proc.build, process=proc.kind, material=study.material, bridge_span=proc.bridge_span)
    r = {"mass": o.report["mass"], "strength": 1.0, "partVolume": pc["volume"], "height": pc["height"],
         "supportVolume": pc["supportVolume"], "supportArea": pc["supportArea"],
         "trappedSupportArea": pc.get("trappedSupportArea", 0.0), "cost": _cost(pc)}
    return _make_cost(r, study, proc, 1.0)


def settle_supports(study, result, rerun, on_progress=None):
    """run_study's second pass for a study whose build direction is "auto":
    `result` is the search for the free design printed on supports, in the
    direction choose_build_direction(self_supporting=False) picked. Whether
    growing it under an overhang rule instead is cheaper depends on how thin
    its members are, so it is decided at the fraction the search delivered,
    on the study's own grid: the bell crank at 25% read the same either way
    (SF 21.2 self-supporting against 20.2 on supports), at 12.5% 0.6 against
    7.7. When a self-supporting direction wins there, `rerun(study)` runs
    the search for it and the cheaper of the two delivered parts is kept.
    The comparison is recorded in the delivered study's raw
    ("orientation", "supports")."""
    import dataclasses

    proc = study.process
    if proc is None or not proc.overhangs or result.best is None:
        return result
    o = result.outcomes[result.best]
    vf = float(o.volume_fraction)
    if on_progress is not None:
        on_progress({"stage": "orient", "settle": "supports or self-supporting", "volumeFraction": vf}, None)
    cmp = choose_build_direction(study, max_elements=study.max_elements, on_progress=on_progress,
                                 volume_fraction=vf, free_mesh=o.mesh)
    best = cmp["candidates"][0]
    free_proc = result.study.process
    note = {"volumeFraction": vf, "ranking": [
        {k: r.get(k) for k in ("direction", "overhangs", "safetyFactor", "mass", "supportVolume", "makeCost")}
        for r in cmp["candidates"] if r.get("designed")]}
    keep = result
    if best["overhangs"]:
        other = rerun(cmp["study"])
        if other.best is not None:
            c_free = _delivered_cost(study, o, free_proc)
            c_self = _delivered_cost(study, other.outcomes[other.best], other.study.process)
            note["delivered"] = {"onSupports": round(c_free, 2), "selfSupporting": round(c_self, 2)}
            if c_self < c_free:
                keep = other
    note["chosen"] = "self-supporting" if keep.study.process.overhangs else "on supports"
    raw = dict(keep.study.raw)
    raw["supports"] = note
    keep.study = dataclasses.replace(keep.study, raw=raw)
    if on_progress is not None:
        on_progress({"stage": "orient", "supports": note["chosen"],
                     "direction": direction_label(keep.study.process.build)}, None)
    return keep
