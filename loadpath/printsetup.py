"""
Pick the print setup for a finished FDM part: build orientation, material,
bed temperature and brim, each candidate checked for strength (verify_part
with that material and layer direction) and for warp (buildsim, on the bed
at the end of the print, along the build). The search is on the delivered
shape: it does not re-optimize the part for each setup, so a material or
orientation that wins here may do better still once the part is
re-optimized for it (search_print_setup reports what to set; run the study
again with it).

Why: on a bed-printed part the warp is set by the material, the height on
the bed and how the bed holds it far more than by the topology (on the
120 mm PA12 beam of bench/buildsim_fdm_warp_limit.py every design lifted
2.5-3.7 mm), so these are the levers a warp limit needs.

Stage 1 runs every (material, orientation) at the material's usual bed
temperature without a brim; stage 2 sweeps bed temperature and brim on the
best few that meet the safety factor. Ranking: meets the goal and the warp
limit first, then lightest, then least warp, then no brim.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

AXES = ("+z", "-z", "+x", "-x", "+y", "-y")
BED_MAX = 110.0       # most hobby beds top out near here
BED_STEP = 10.0
TALL = 4.0            # height over narrowest base past which a print needs a brim to stand
TALL_MAX = 10.0       # and past which not even a brim keeps it steady
ROOM = 0.8            # a lift above this share of the limit counts as tight


def _bed_candidates(material, setup, max_bed):
    from .buildsim import BuildSetup, _fdm_params
    p = _fdm_params(dataclasses.replace(setup, material=material))
    base = p["bed"]
    # above Tg an amorphous first layer sags (elephant's foot), so stop
    # short of it; semi-crystalline (Tc far above) stop at the bed's limit
    top = min(max_bed, p["tg"] - 5.0)
    out = sorted({float(b) for b in (base - 2 * BED_STEP, base - BED_STEP, base, base + BED_STEP) if 40.0 <= b <= top})
    return out or [min(base, top)]


def _warp(mesh, setup, build, cells):
    from .buildsim import simulate_part
    r = simulate_part(mesh, dataclasses.replace(setup, release="none"), build=build, cells=cells)
    m = r.mesh
    on = np.zeros(m.n_nodes, bool)
    on[m.enode[r.part].ravel()] = True
    lift = float(np.abs(r.u_built[on, 2]).max())
    return {"lift": lift, "peel": (r.report.get("peel") or {}).get("meanPeelStress"),
            "bedFailed": bool(r.report.get("peel", {}).get("bedFailed", False))}


def _strength(study, mesh, material, build, max_cells):
    from .am import parse_build_direction
    from .materials import material_from_spec
    from .verify import verify_part
    from .study import _margin
    mat = material_from_spec(material)
    b = parse_build_direction(build)
    proc = dataclasses.replace(study.process, build=b) if study.process is not None else None
    st = dataclasses.replace(study, material=mat, materials=[mat], process=proc,
                             print_direction=b if study.print_direction is not None else None)
    rep = verify_part(st, mesh, max_cells=max_cells)
    # the goal's margin without the warp limits (those are checked here)
    goal = _margin(st, {k: v for k, v in rep.items() if k != "warp"})
    return {"mass": rep.get("mass"), "safetyFactor": rep.get("safetyFactor"), "margin": goal,
            "supportVolume": (rep.get("print") or {}).get("supportVolume"),
            "aspect": ((rep.get("print") or {}).get("bed") or {}).get("aspect"),
            "comMargin": ((rep.get("print") or {}).get("bed") or {}).get("comMargin")}


def search_print_setup(study, mesh, materials=None, builds=None, beds=None, brims=(0.0, 5.0), max_bed=BED_MAX,
                       limit=None, cells=4000, verify_cells=30_000, top=2, progress=None, reoptimize=True) -> dict:
    """Rank FDM print setups for the delivered part `mesh` (trimesh, study
    frame). materials: names (default the study's); builds: build
    directions (default the six grid axes); beds: bed temperatures (default
    a few around each material's usual one, below its Tg); brims: brim
    widths, mm; limit: maxDistortion, mm (default buildSim.maxDistortion).
    Returns {"best": row, "rows": [...], "seconds", "notes"}; each row has
    material, build, bed, brim, lift (mm), mass, safetyFactor, margin, and
    passes (goal and warp limit)."""
    t0 = time.time()
    bs = study.build_sim or {}
    from .buildsim import BuildSetup
    setup = bs.get("setup") or BuildSetup(process="fdm", material=study.material.name)
    del reoptimize  # used by the study runner (study._print_setup_rescue)
    if setup.process.lower() != "fdm":
        raise ValueError("search_print_setup is for FDM parts (buildSim with process fdm)")
    if limit is None:
        limit = bs.get("maxDistortion")
    mats = list(materials or [m.name for m in (study.materials or [study.material])])
    builds = list(builds or AXES)

    def note(**kw):
        if progress is not None:
            progress(kw)

    strength = {}
    for mat in mats:
        for b in builds:
            note(stage="strength", material=mat, build=b)
            try:
                strength[(mat, b)] = _strength(study, mesh, mat, b, verify_cells)
            except (RuntimeError, ValueError) as exc:
                strength[(mat, b)] = {"error": str(exc), "margin": 0.0}

    rows = []

    def run(mat, b, bed, brim):
        st = dataclasses.replace(setup, material=mat, bed=bed, brim=brim)
        note(stage="warp", material=mat, build=b, bed=bed, brim=brim)
        w = _warp(mesh, st, b, cells)
        s = strength[(mat, b)]
        ok_goal = s.get("margin") is None or s["margin"] >= 1.0
        ok_warp = limit is None or w["lift"] <= limit
        # a part standing over 4x taller than its narrowest base wobbles
        # late in the print (printcheck), and one whose centre of mass is
        # off its contact tips: both need a brim to stand
        asp = s.get("aspect") or 0.0
        tall = asp > TALL or (s.get("comMargin") or 0.0) < 0
        ok_stand = asp <= TALL_MAX and (not tall or brim > 0)
        row = {"material": mat, "build": b, "bed": bed, "brim": brim, **w,
               **{k: s.get(k) for k in ("mass", "safetyFactor", "margin", "supportVolume", "aspect")},
               "meetsGoal": bool(ok_goal), "meetsWarp": bool(ok_warp), "stands": bool(ok_stand),
               "passes": bool(ok_goal and ok_warp and ok_stand)}
        rows.append(row)
        return row

    from .buildsim import _fdm_params
    first = []
    for mat in mats:
        base = _fdm_params(dataclasses.replace(setup, material=mat))["bed"]
        for b in builds:
            first.append(run(mat, b, base, 0.0))
    # stage 2: bed and brim on the best few that hold the load
    # (per material, so a lighter material is not crowded out by a flatter one)
    can = [r for r in first if (r.get("aspect") or 0.0) <= TALL_MAX]
    strong = []
    for mat in mats:
        mine = [r for r in can if r["material"] == mat]
        mine = [r for r in mine if r["meetsGoal"]] or mine
        strong += sorted(mine, key=lambda r: (r["lift"], r["mass"] or 0))[:top]
    for r in strong:
        for bed in (beds or _bed_candidates(r["material"], setup, max_bed)):
            for brim in brims:
                if bed == r["bed"] and brim == r["brim"]:
                    continue
                run(r["material"], r["build"], float(bed), float(brim))

    def rank(r):
        # among equals: room under the warp limit first (the FDM warp model
        # is good to tens of percent), then less support, no brim, less lift
        tight = limit is not None and r["lift"] > ROOM * limit
        return (not r["passes"], not r["meetsGoal"], round(r["mass"] or 0.0, 2) if r["passes"] else 0.0,
                tight, round(r["supportVolume"] or 0.0, -1), r["brim"], r["lift"], abs(r["bed"]))

    rows.sort(key=rank)
    best = rows[0] if rows else None
    cur = next((r for r in rows if r["material"] == study.material.name and r["brim"] == 0.0
                and r["build"] == _build_name(study)), None)
    return {"best": best, "current": cur, "rerun": rerun_candidate(rows, limit), "rows": rows, "limit": limit, "seconds": round(time.time() - t0, 1),
            "notes": ["the delivered shape is re-checked, not re-optimized, for each setup: re-run the study with "
                      "the best setup to let the shape follow it",
                      "brim is an estimate (more bed hold under the footprint's edge), not validated; bed "
                      "temperature and material are in the validated FDM model"]}


def _build_name(study):
    b = study.process.build if study.process is not None else None
    if isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], (int, np.integer)):
        return ("+" if b[1] > 0 else "-") + "xyz"[b[0]]
    return "+z"


def rerun_candidate(rows, limit, room=ROOM):
    """The setup to re-optimize the part for when none passes as it is:
    among the setups whose lift leaves room under the limit (room x limit,
    else any under it), the most safety factor per gram (a re-optimized
    part trades the two), then no brim, then the least lift."""
    if limit is None:
        return None
    ok = [r for r in rows if r["lift"] <= room * limit] or [r for r in rows if r["lift"] <= limit]
    ok = [r for r in ok if r.get("margin") is not None and r.get("mass") and r.get("stands", True)]
    if not ok:
        return None
    return min(ok, key=lambda r: (-round(r["margin"] / r["mass"], 4), r["brim"], r["lift"]))


def study_for(study, row):
    """The study set up to print as `row` says: material, build direction
    and the build sim's bed temperature and brim (printSetup off, so a
    re-run does not search again)."""
    from .am import parse_build_direction
    from .materials import material_from_spec
    mat = material_from_spec(row["material"])
    b = parse_build_direction(row["build"])
    proc = dataclasses.replace(study.process, build=b)
    bs = dict(study.build_sim)
    bs["setup"] = dataclasses.replace(bs["setup"], material=mat.name, bed=row["bed"], brim=row["brim"])
    bs["printSetup"] = None
    pd = b if study.print_direction is not None else None
    return dataclasses.replace(study, material=mat, materials=[mat], process=proc, build_sim=bs, print_direction=pd)
