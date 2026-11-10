"""
Run a study end to end: voxelize, optimize, mesh, verify.

Two kinds of goal:

- `volumeFraction`: keep that share of the design space (plus any extra
  `outcomes` fractions, for a trade-off gallery).
- `safetyFactor` and/or `maxDeflection`: as light as possible while every
  load case keeps at least that safety factor against the material's
  strength and deflects at most that much at its loads. Neither is something
  compliance optimization controls directly, so this searches over volume
  fractions, verifying the delivered mesh at each step, and returns the
  lightest outcome that passes. Each step warm-starts from the nearest
  fraction above it already run.
  With `durability.constrain` the search also holds the fatigue and creep
  safety factor to `durability.safetyFactor` (and a held load's crept
  deflection to `maxDeflection`); see durability.py.

The search works on one number, the margin: the worst of (safety factor /
target) and (deflection limit / deflection). Both scale roughly linearly with
the amount of material in an optimized part, so the next volume fraction is
the current one divided by the margin.
"""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .domain import Domain, build_domain, cell_cap
from .manufacture import report as manufacturing_report
from .reconstruct import ReconstructOptions, reconstruct
from .simp import SimpOptions, filter_radius, optimize
from .spec import Study
from .verify import verify_part

__all__ = ["Outcome", "StudyResult", "run_study", "design_at"]

# Bounds for the safety-factor search. (A floor of 0.04 stopped the robot arm's upper
# arm there with SF 5.4 against a goal of 2: large design spaces around
# small parts end at a few percent.)
_VF_MIN, _VF_MAX = 0.02, 0.7
_MAX_SEARCH_RUNS = 7
# Coarse-to-fine: below this many cells the fine solve is cheap enough alone.
_COARSE_MIN_CELLS = 12_000
# Projection sharpness a warm-started run resumes at.
_WARM_BETA = 4.0
# runs of the constrained method's limit search (study.run, method constrained)
_MAX_CONSTRAINED_RUNS = 6
# goal.buckling in the constrained method: hold the design's buckling load
# factor in the optimizer itself (bucklingopt.py), starting at the safety
# factor times _BUCKLING_START (recalibrated against the verified part after
# each run). LOADPATH_BUCKLING_OPT=0 leaves buckling to the check alone.
_BUCKLING_IN_OPTIMIZER = os.environ.get("LOADPATH_BUCKLING_OPT", "1") != "0"
_BUCKLING_START = 1.2
# The search (volume fractions, simp.optimize) can hold buckling too: an
# adaptive penalty on its step, turned on once a part's buckling factor is
# under _BUCKLING_SEARCH_ON times the safety factor. OPT-IN
# (LOADPATH_BUCKLING_SEARCH=1): on the full-tier printed hook, the one eval
# part whose search buckling binds (2.61 for 2.5 at 29.3 g), it gave 29.8 g
# (buckling 2.50, SF 3.22) in 1.8x the time; the lighter fractions failed on
# stress (voxel SF 0.1-0.6 at 24-29 g), not buckling. MMA with buckling as a
# hard constraint built weaker layouts still (SF 0.3-2.0 at 28-31 g).
_BUCKLING_SEARCH = os.environ.get("LOADPATH_BUCKLING_SEARCH", "0") == "1"
_BUCKLING_SEARCH_ON = 1.5
# method auto: run the constrained method after the search when the search's
# part idles (verify's stressUtilization, mean von Mises over the near peak,
# under this) and is not deflection-bound. Search picks' utilization, and
# tet-confirmed search / auto masses (2026-11-01): fastener bracket 0.10
# (41.0 / 26.4 g), wall bracket 0.11 (64.4 / 26.6 g), where constrained wins;
# MBB 0.19, bell crank 0.20 and the robot-arm yoke 0.21, where it found nothing
# lighter in 2-5x the search's time; hook 0.30 and arm link 0.33, where it
# was 10-40% heavier.
_AUTO_UTILIZATION = 0.15
# Cut cells (verify's default since 2026-11-05) read the surface stress
# without the staircase's spurious corner peaks, so the same part reads
# utilization 7-16% higher (wrist yoke 0.168 -> 0.184, motor mount 0.160 ->
# 0.186, hook 0.328 -> 0.350): at 0.15 the yoke skipped the constrained
# method and came out 18.6 g against 14.1; at 0.19, 13.9.
_AUTO_UTILIZATION_CUT = 0.19
# Between that cut and this one the constrained method still runs, as a
# probe: utilization alone does not say whether it wins there. The minMember
# fixes (fa17ad8, d1b2b2c) took the coarse wrist yoke's search part from
# 0.166 to 0.226 and it was delivered at 18.3 g, while the constrained method
# finds 13.5 g on it; MBB (0.20), corpus-link-0 (0.19) and the motor mount
# (0.20) sit as close to the cut. A probe stops as soon as one of its parts
# fails on the target while weighing at least _PROBE_HEAVY times the search's
# part: the bisection only adds material to a failing part, so it cannot come
# out lighter. Above this, the hook (0.30-0.35) and arm link (0.33) came out
# 10-40% heavier constrained, so it is not tried.
_AUTO_PROBE_UTILIZATION = 0.28
# ... and up to this one it runs a single part and goes on only if that
# part is promising: the full-tier wrist yoke's search part read 0.30 on
# one run and 0.24 on others (fresh tet meshes), so a hard edge at 0.28
# skipped a 2-3 g gain at random. The hook (0.30-0.35) and bell crank
# (0.32) pay a constrained part or two (full tier: +1500 s, +640 s).
_AUTO_PROBE_OUTER = 0.34
# "promising" there: a part that passes lighter than the search's, or with
# this margin to spare under this times its mass (the yoke's first
# constrained part: 17.6 g against 17.2-18.1, tet margin 1.6-2.7; its later
# parts reached 14.2-15.1 g)
_PROBE_ROOM = 1.3
_PROBE_NEAR = 1.1
# A winning probe runs at least this many parts (up to the box's hard stop,
# twice the box): the box follows the search's time (3.5x), the parts do
# not (1150-2400 s each on the full-tier yoke, with a search of 540-1400
# s), and each part warm-starts from the last passing one, which is where
# the gain comes from: 17.6 then 14.8-15.6 g. (Two parts at once on
# threads, from the same start, both came out near 15 g in 1.6x a part's
# time: not kept.)
_PROBE_ROUNDS = 2
_PROBE_HEAVY = 0.97
# and gives it at most this many times the search's own time
_AUTO_BUDGET = 3.5
# ... but no less than this many finished parts (up to twice the box): the
# box is wall time, so a faster search shortened it while the constrained
# run did not speed up as much; the coarse wrist yoke's run stopped after 2
# parts and delivered the search's 19.2 g against 13.3-14.0 g from 5
_AUTO_MIN_RUNS = 4


# A probe (the band above) gets the plain _AUTO_BUDGET, not the buckling
# one's doubling, and once it holds a passing part lighter than the search's
# it starts no part it cannot finish inside its box (_probe_settled): the
# full-tier wrist yoke's probe ran 7464 s for 13.9 g under the doubled box;
# its parts land about 1300 s apart (17.6, 15.6, 14.4, 14.2 g), so most of
# the gain over the search's 17.2 g is in by the second or third. Capped:
# 15.1 g in 3357 s.


def _box_over(deadline: float, box: float, done: int) -> bool:
    """Whether the constrained run's time box is over, `done` parts in."""
    now = time.time()
    return now > deadline and (done >= _AUTO_MIN_RUNS or now > deadline + box)
# The pick of a safety-factor search is checked again on a grid this many
# cells finer before it is delivered. The in-loop check resolves a fillet
# only as well as its grid: the robot arm's turret read SF 2.26 there and 1.65 at
# 120k cells, its peak at a corner the coarser grid had averaged away.
_CONFIRM_CELLS = 150_000
_CONFIRM_RUNS = 3
# With goal.confirm "tet", at most this many picks are confirmed on the
# tetrahedral mesh, and the search runs at most this many more fractions
# after its first pick (the step down the calibration allows, and back).
_TET_CONFIRMS = 6
_TET_EXTRA_RUNS = 4
# A tet-confirmed pick within this of its target is close enough.
_TET_CLOSE = 1.08
# Optimizations the layout search below a confirmed pick may spend.
_LAYOUT_RUNS = 3


def _making(study: Study) -> dict:
    """The optimizer options that say how the part is made."""
    return dict(build_direction=study.print_direction, extrusion=study.extrusion, access=study.access,
                pattern=study.pattern, max_member=study.max_member,
                dilate=_DILATE if study.min_hole else 0.0, escape=study.drains)


# simp.SimpOptions.dilate when the study has a minimum hole: the erosion's twin
_DILATE = 0.15


def _coarse_start(study: Study, domain: Domain, vf: float, radius: float, should_stop) -> np.ndarray | None:
    """Find the layout on a grid twice as coarse (an eighth of the cells, so
    each iteration is roughly ten times cheaper), stopping while it is still
    soft (beta 4), then interpolate it onto the fine grid as the starting
    design. The fine run only has to sharpen and refine it."""
    from scipy import ndimage

    try:
        coarse = build_domain(study, pitch=2.0 * domain.pitch)
    except ValueError:
        # The coarse grid can lose what the fine one resolves (a fastener's
        # seat and face rings landing on the same nodes, a thin load pad):
        # then start cold on the fine grid, which `check` validated.
        return None
    res = optimize(coarse, study.elastic(design=True),
                   SimpOptions(volume_fraction=vf, filter_radius=max(radius, 1.5 * coarse.pitch),
                               max_iterations=60, beta_max=_WARM_BETA, beta_every=12,
                               **{**_making(study), "escape": None}),
                   should_stop=should_stop)
    grid = coarse.full_grid(res.x)
    coords = (domain.mesh.elem_centers() - coarse.origin) / coarse.pitch - 0.5
    x0 = ndimage.map_coordinates(grid, coords.T, order=1, mode="nearest")
    return np.clip(x0, 0.0, 1.0)


@dataclass
class Outcome:
    volume_fraction: float
    mesh: object
    report: dict
    optimizer: dict
    reconstruction: dict
    passes: bool | None = None
    x: object = None
    # the optimized density field this mesh was drawn from (per domain cell),
    # so a part can be re-meshed without optimizing again
    xphys: object = None
    # infill.grade_infill's result (modifier meshes for the graded 3MF)
    grading: object = None
    # buildsim.compensate_mesh's pre-distorted part (buildSim.compensate)
    compensated: object = None

    def summary(self) -> dict:
        r = self.report
        return {
            "material": r.get("material", {}).get("name"),
            "volumeFraction": self.volume_fraction,
            "mass": r.get("mass"),
            "massReduction": r.get("massReduction"),
            "safetyFactor": r.get("safetyFactor"),
            **({"voxelSafetyFactor": r["voxelSafetyFactor"]} if "voxelSafetyFactor" in r else {}),
            **({"bucklingFactor": r["bucklingFactor"]} if r.get("bucklingFactor") is not None else {}),
            **({"confirm": r["confirm"]} if "confirm" in r else {}),
            **({"calibration": r["calibration"]} if "calibration" in r else {}),
            **({"tetCheck": r["tetCheck"]} if "tetCheck" in r else {}),
            "cases": r.get("cases"),
            "minMember": r.get("minMember"),
            # the grid the outcome was verified on: `loadpath orient` reads
            # its across-layer stress on the same one
            **({"analysis": {"pitch": r["analysisPitch"], "contactBand": r["contactBand"]}}
               if "analysisPitch" in r and "contactBand" in r else {}),
            "overhangArea": r.get("overhangArea"),
            "keepOutIntrusion": r.get("keepOutIntrusion"),
            "preserveMissing": r.get("preserveMissing"),
            "bodies": r.get("bodies"),
            "boundaryCutArea": r.get("boundaryCutArea"),
            **({"closedBores": r["closedBores"]} if r.get("closedBores") else {}),
            **({"fits": r["fits"]} if r.get("fits") else {}),
            **({"thermal": r["thermal"]} if "thermal" in r else {}),
            **({"durability": r["durability"]} if "durability" in r else {}),
            **({"print": {k: v for k, v in r["print"].items() if k != "bridgeList"}} if "print" in r else {}),
            **({"process": r["process"]} if "process" in r else {}),
            **({"manufacturing": r["manufacturing"]} if "manufacturing" in r else {}),
            **({"slicer": r["slicer"]} if "slicer" in r else {}),
            **({"buildSim": r["buildSim"]} if "buildSim" in r else {}),
            **({"graded": r["graded"]} if "graded" in r else {}),
            "warnings": r.get("warnings", []),
            "margin": r.get("margin"),
            "passes": self.passes,
            "start": r.get("start"),
            "alternative": r.get("alternative"),
            "optimizer": self.optimizer,
            "reconstruction": self.reconstruction,
        }


@dataclass
class StudyResult:
    study: Study
    domain: dict
    outcomes: list[Outcome]
    best: int | None
    seconds: float
    warnings: list = field(default_factory=list)
    # method auto: which method made `best` and why
    choice: dict | None = None
    # resolution.pitchStarts: each grid's pitch, pick and whether it passed
    starts: list | None = None

    def to_dict(self) -> dict:
        return {
            **({"pitchStarts": self.starts} if self.starts else {}),
            "name": self.study.name,
            "domain": self.domain,
            "outcomes": [o.summary() for o in self.outcomes],
            "best": self.best,
            "seconds": self.seconds,
            "warnings": self.warnings,
            **({"methodChoice": self.choice} if self.choice else {}),
            **({"orientation": self.study.raw["orientation"]} if "orientation" in self.study.raw else {}),
        }


def _durability_drives(study: Study) -> bool:
    return study.durability is not None and study.durability.constrain


def _margin(study: Study, report: dict) -> float | None:
    g = study.goal
    ms = []
    if g.safety_factor is not None:
        ms.append(report["safetyFactor"] / g.safety_factor)
        if g.buckling and report.get("bucklingFactor") is not None:
            # a part that buckles under safetyFactor x its loads fails as one
            # that yields there does (verify._buckling)
            ms.append(report["bucklingFactor"] / g.safety_factor)
    dur = report.get("durability") if _durability_drives(study) else None
    if g.max_deflection is not None:
        worst = max((c["deflection"] for c in report["cases"]), default=float("inf"))
        if dur:
            # a held load's deflection after creep, at its temperature
            worst = max([worst] + [c["creepDeflection"] for c in dur["cases"] if "creepDeflection" in c])
        ms.append(g.max_deflection / worst if worst > 0 else float("inf"))
    for c in report.get("cases") or []:
        # goal.clearances: travel away from the loads (a deck's middle)
        for r in c.get("clearances") or []:
            if r.get("travel") is not None:
                ms.append(r["maxTravel"] / r["travel"] if r["travel"] > 0 else float("inf"))
    if dur and dur.get("safetyFactor") is not None:
        ms.append(dur["safetyFactor"] / dur["target"])
    mp = report.get("massProperties") or {}
    if mp.get("maxInertia") and mp.get("axisInertia"):
        # inertia follows the material: linear in the fraction
        ms.append(mp["maxInertia"] / mp["axisInertia"])
    from .verify import reaction_margin

    rm = reaction_margin(report.get("reactions"))
    if rm is not None:
        ms.append(rm)
    vib = report.get("vibration")
    if vib and vib.get("firstFrequency") is not None:
        # f = sqrt(k / m) / 2 pi: with carried masses the mass barely moves
        # with the fraction and k roughly follows it, so the fraction needed
        # goes as the frequency squared
        ms.append((vib["firstFrequency"] / vib["minFrequency"]) ** 2)
    warp = report.get("warp")
    if warp:
        # buildSim warp limits: the build of the delivered part when it was
        # re-simulated (_warp_delivered), else the optimizer's on its field
        got = warp.get("delivered") or warp
        lims = warp.get("userLimits") or warp["limits"]
        for key, lim in (("maxDistortion", "maxDistortion"), ("recoaterLift", "maxRecoaterLift")):
            if lims.get(lim) and got.get(key) is not None:
                ms.append(lims[lim] / max(got[key], 1e-12))
    return min(ms) if ms else None


def _whole(report: dict) -> bool:
    """A part that can be delivered at all, whatever its margin: one body,
    and every bore open at both ends. A capped bore takes no screw (the
    wall bracket with fasteners at 5.6%, tets 3.52 for 2.5, came out with
    one closed over; the search delivered it), so the search counts it
    failing and steps on to a fraction that keeps the bore."""
    return report.get("bodies", 1) <= 1 and not report.get("closedBores")


# A warm-started outcome is rerun cold when it meshes into members under
# this share of minMember (verify's p5 member, a voxel ridge reading that
# under-reads by up to ~30% on coarse grids: clean parts read 0.65-2.0 x
# minMember) or leaves more crease than this, mm (clean parts 0-125).
_MALFORMED_MEMBER = 0.55
_MALFORMED_CREASE = 200.0


def _malformed(study: Study, report: dict) -> str | None:
    """Why an outcome's mesh is not the kind of part the study asks for
    (members far under minMember, a long crease), or None."""
    p5 = report.get("p5Member")
    if study.min_member and p5 is not None and p5 < _MALFORMED_MEMBER * float(study.min_member):
        return f"with members of {p5:.2f} mm for minMember {float(study.min_member):g}"
    crease = report.get("creaseLength")
    if crease is not None and crease > _MALFORMED_CREASE:
        return f"with {crease:.0f} mm of crease"
    return None


def _warp_delivered(study: Study, mesh) -> dict | None:
    """The as-built measures of the delivered part (buildsim.simulate_part
    at buildSim.cells, on the plate or bed at the end of the build, as the
    optimizer's limits are): the largest movement (along the build for
    FDM, |u| otherwise) and the recoater lift, mm. The optimizer's build
    model runs on its coarse density grid, which under-resolves thin
    members (FDM warp on the PA12 beam read 2-3.5x low there), so FDM warp
    limits are checked on this one."""
    import dataclasses as _dc
    from .buildsim import simulate_part
    bs = study.build_sim
    build = study.process.build if study.process is not None and study.process.build not in (None, "auto") else "+z"
    if study.print_direction is not None:
        build = study.print_direction
    setup = _dc.replace(bs["setup"], release="none")
    t0 = time.time()
    try:
        r = simulate_part(mesh, setup, build=build, cells=min(int(bs.get("cells") or 8000), 30_000))
    except Exception as exc:  # noqa: BLE001 - leave the optimizer's numbers standing
        return {"error": f"{type(exc).__name__}: {exc}"}
    m = r.mesh
    on = np.zeros(m.n_nodes, bool)
    on[m.enode[r.part].ravel()] = True
    u = r.u_built[on]
    d = np.abs(u[:, 2]) if setup.process.lower() == "fdm" else np.linalg.norm(u, axis=1)
    return {"maxDistortion": float(d.max()),
            "recoaterLift": (r.report.get("recoater") or {}).get("maxLift"),
            "pitch": r.report.get("pitch"), "seconds": round(time.time() - t0, 1)}


def design_at(study: Study, domain: Domain, radius: float, vf: float, x0=None, beta0: float = 1.0,
              max_iterations: int | None = None, on_iter=None, should_stop=None, emit=None,
              verify: bool = True, verify_cells: int = 60_000, refine: int = 3) -> Outcome:
    """One optimize -> mesh -> verify pass at volume fraction `vf`, from
    design `x0` (None: uniform) with projection starting at `beta0`. The
    outcome carries its margin and pass/fail against the study's goal
    (verify=False skips the check: an empty report, for a design that is
    only a starting point; verify_cells and refine set how closely it is
    checked, see verify_part)."""
    def note(event):
        if emit is not None:
            emit(event)

    res = optimize(domain, study.elastic(design=True),
                   SimpOptions(volume_fraction=vf, filter_radius=radius,
                               max_iterations=max_iterations or study.max_iterations, x0=x0, beta_start=beta0,
                               **_making(study)),
                   on_iter=on_iter, should_stop=should_stop)
    note({"stage": "mesh", "volumeFraction": vf})
    mesh, rinfo = reconstruct(study, domain, res.xphys, ReconstructOptions())
    opt = {"iterations": res.iterations, "converged": res.converged, "seconds": res.seconds,
           "compliance": res.compliance, "caseCompliance": res.case_compliance,
           **({"caseDirection": res.case_direction} if any(d is not None for d in res.case_direction) else {}),
           "achievedVolumeFraction": res.volume_fraction,
           "beta": res.history[-1]["beta"] if res.history else beta0,
           **({"escape": res.escape} if res.escape else {})}
    o = Outcome(vf, mesh, {}, opt, rinfo)
    o.xphys = res.xphys
    o.x = res.x
    if not verify:
        return o
    note({"stage": "verify", "volumeFraction": vf})
    try:
        report = verify_part(study, mesh, max_cells=verify_cells, min_pitch=0.5 * domain.pitch,
                             contact_band=domain.pitch, refine=refine, max_pitch=domain.pitch)
    except (RuntimeError, ValueError) as exc:
        # one outcome verification cannot solve (a piece the supports do
        # not hold) is a failed outcome, not a failed study: the search
        # goes on past it (the robot arm's wrist link died here)
        mat = study.material
        report = {"mass": float(mesh.volume) * mat.density / 1000.0, "safetyFactor": 0.0,
                  "cases": [], "warnings": [f"verification failed: {exc}"]}
    making = manufacturing_report(study, domain, mesh)
    if making:
        report["manufacturing"] = making
        report.setdefault("warnings", []).extend(making.get("warnings", []))
    report.setdefault("warnings", []).extend(fragment_warnings(rinfo, mesh))
    if res.warp:
        report["warp"] = res.warp
    o.report = report
    o.voxel_sf = report.get("safetyFactor")
    m = _margin(study, report)
    if m is not None:
        report["margin"] = m
        o.passes = bool(m >= 1.0)
    if report.get("bodies", 1) > 1:
        # A preserved piece no load path reaches (a mount with no load
        # through it) comes out as a loose part. It is not a design that
        # meets the brief, and the fix is in the study: say so.
        o.passes = False
        report.setdefault("warnings", []).append(
            f"the part came out in {report['bodies']} pieces: either each piece has its own supports and "
            "load path and nothing ties them together (two mirror halves, each on its own bolts), or some "
            "preserved geometry is on no load path. For one part, preserve a link between the pieces (a "
            "ring, a web, a plate) or add a load case that runs between them; for a floating preserve, "
            "load it or drop it.")
    if report.get("closedBores"):
        o.passes = False
        report.setdefault("warnings", []).append(
            f"the part closes over {len(report['closedBores'])} bore end(s) "
            f"({', '.join(b['id'] + ' ' + b['end'] for b in report['closedBores'])}): no fastener goes through, "
            "so it does not pass. Extend the keep-out over the head and the driver's reach if this repeats.")
    return o


def _confirm_on_tets(study: Study, outcomes: list) -> bool:
    """goal.confirm says tet (or auto, and the cross-check covers it)."""
    mode = study.goal.confirm
    if mode == "voxel":
        return False
    from .crosscheck import unsupported

    why = unsupported(study)
    if why is None:
        return True
    if outcomes and (mode == "tet" or why.startswith("no tetrahedral mesher")):
        # A missing mesher is the install, not the study: say so on every
        # part it changes. The voxel grid reads peak stress high (the MBB
        # beam 0.86 on voxels, 1.94 on tets at 30%), so without the tet
        # confirm a search stops heavier: the MBB beam's eval went from
        # 14.5 g to 40.4 g and failed its mass reduction, silently.
        outcomes[0].report.setdefault("warnings", []).append(
            f"the tetrahedral cross-check does not cover this study ({why}): confirmed on voxels instead, "
            "which read peak stress high, so the part comes out heavier than it needs to be")
    return False


def _domain_info(d: Domain, radius: float) -> dict:
    return {
        "pitch": d.pitch,
        "cells": d.n_elem,
        "dims": list(d.dims),
        "preservedCells": int(d.preserve.sum()),
        "filterRadius": radius,
        "loadCases": d.case_names,
    }


def save_field(path: str, xphys, pitch: float, cells: int | None = None) -> str | None:
    """Save an outcome's optimized density field (.npz: xphys as float32 and
    the design grid's pitch), so a bad mesh can be re-meshed without
    optimizing again (`loadpath remesh`). None when there is no field, or it
    is not on the grid named (`cells`)."""
    if xphys is None:
        return None
    x = np.asarray(xphys, np.float32).ravel()
    if cells is not None and len(x) != int(cells):
        return None
    np.savez_compressed(path, xphys=x, pitch=np.float64(pitch))
    return path if path.endswith(".npz") else path + ".npz"


def load_field(path: str):
    """(xphys, pitch) from save_field's file (or a bare .npy of xphys:
    pitch None, the study's own grid)."""
    if path.endswith(".npy"):
        return np.load(path).astype(float), None
    with np.load(path) as z:
        return z["xphys"].astype(float), float(z["pitch"]) if "pitch" in z else None


# Peak memory of one worker, roughly, by the most cells it solves at once
# (study._study_cells): the robot arm's turret, a safety-factor search confirmed at
# 150k cells, peaked at 8.4 GB.
_RUN_BASE_BYTES = 0.6e9
_RUN_CELL_BYTES = 55e3
def _run_bytes(cells: int) -> float:
    return _RUN_BASE_BYTES + _RUN_CELL_BYTES * int(cells)


def _cgroup_limit() -> float | None:
    """The memory limit of the container this runs in, if any (cgroup v2,
    then v1): MemAvailable reports the host's, and the cloud box's 14.3 GB
    limit is what the kernel enforces."""
    paths = []
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                _, ctrl, rel = line.strip().split(":", 2)
                if ctrl == "":
                    paths.append(f"/sys/fs/cgroup{rel}/memory.max")
                elif "memory" in ctrl.split(","):
                    paths.append(f"/sys/fs/cgroup/memory{rel}/memory.limit_in_bytes")
    except (OSError, ValueError):
        pass
    for path in paths + ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            with open(path) as f:
                v = f.read().strip()
            if v.isdigit() and int(v) < 1 << 60:
                return float(v)
        except OSError:
            pass
    return None


def _cgroup_free() -> float | None:
    """The container's limit less what it holds now (reclaimable page cache
    not counted): the limit alone let eight brief-eval agents' runs each
    count the whole 14 GB as theirs, and the kernel killed them."""
    lim = _cgroup_limit()
    if not lim:
        return None
    paths = []
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                _, ctrl, rel = line.strip().split(":", 2)
                if ctrl == "":
                    paths.append((f"/sys/fs/cgroup{rel}/memory.current", f"/sys/fs/cgroup{rel}/memory.stat",
                                  "inactive_file"))
                elif "memory" in ctrl.split(","):
                    paths.append((f"/sys/fs/cgroup/memory{rel}/memory.usage_in_bytes",
                                  f"/sys/fs/cgroup/memory{rel}/memory.stat", "total_inactive_file"))
    except (OSError, ValueError):
        return lim
    for used_p, stat_p, key in paths:
        try:
            with open(used_p) as f:
                used = float(f.read().strip())
        except (OSError, ValueError):
            continue
        cache = 0.0
        try:
            with open(stat_p) as f:
                for line in f:
                    k, _, v = line.partition(" ")
                    if k == key:
                        cache = float(v)
        except (OSError, ValueError):
            pass
        return max(0.0, lim - (used - cache))
    return lim


def _available_bytes() -> float | None:
    avail = _meminfo_available()
    cap = _cgroup_free()
    if avail is None:
        return cap
    return min(avail, cap) if cap is not None else avail


def _meminfo_available() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    try:
        import psutil  # optional
        return float(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        pass
    try:
        # macOS has no /proc: count most of physical memory as available
        return 0.6 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _workers(n: int, cells: int | None = None) -> int:
    """Processes for independent outcomes: LOADPATH_WORKERS, else one per
    core, and no more than fit in the memory available when each solves
    `cells` cells (the 4-core, 15 GB cloud box ran out at two robot-arm cases at
    once). (This was half the cores, leaving room for the thread per load
    case each solve runs; but Gauss-Seidel holds the GIL, so those threads
    mostly take turns, and with one BLAS thread four arm-link directions
    took 121 s on four workers against 194 s on two.)"""
    env = os.environ.get("LOADPATH_WORKERS")
    if env:
        return max(1, min(n, int(env)))
    w = max(1, min(n, os.cpu_count() or 1))
    avail = _available_bytes() if cells else None
    if avail:
        w = min(w, max(1, int(0.8 * avail // _run_bytes(cells))))
    return w


def _save_partial(o, pitch: float | None = None) -> None:
    """With LOADPATH_PARTIAL_DIR set (`loadpath run` sets it to out/partial),
    write each outcome's STL and summary as the search finds it: four
    brief-eval runs were killed in their last stage and kept nothing of
    half an hour of verified outcomes."""
    d = os.environ.get("LOADPATH_PARTIAL_DIR")
    if not d or o.mesh is None:
        return
    try:
        import json

        os.makedirs(d, exist_ok=True)
        stem = f"vf{o.volume_fraction:.4f}".replace(".", "p")
        o.mesh.export(os.path.join(d, stem + ".stl"))
        field = save_field(os.path.join(d, stem + ".field.npz"), o.xphys, pitch) if pitch else None
        with open(os.path.join(d, "outcomes.jsonl"), "a") as fh:
            fh.write(json.dumps({"stl": stem + ".stl", **({"field": stem + ".field.npz"} if field else {}),
                                 **o.summary()}, default=str) + "\n")
    except Exception:  # noqa: BLE001  (a convenience: never fail the run for it)
        pass


def _confirm_cells(st: Study | None = None) -> int:
    """Cells for the confirming re-analysis: LOADPATH_CONFIRM_CELLS, else
    150k, but no more than fit in the memory free right now. A brief-eval
    agent's search on a shared 15 GB box was OOM-killed at this stage after
    35 minutes, losing every verified outcome; a coarser confirm (never
    under the design grid) is better than none."""
    env = os.environ.get("LOADPATH_CONFIRM_CELLS")
    if env:
        return max(1000, int(env))
    floor = max(st.max_elements if st is not None else 0, 20_000)
    avail = _available_bytes()
    if not avail:
        return _CONFIRM_CELLS
    fit = int((0.6 * avail - _RUN_BASE_BYTES) // _RUN_CELL_BYTES)
    return int(max(floor, min(_CONFIRM_CELLS, fit)))


def _solve_alone(study: Study) -> StudyResult:
    return _run_single(study)


def _study_cells(st: Study) -> int:
    """The most cells one run of `st` solves at once."""
    g = st.goal
    search = g.safety_factor is not None or g.max_deflection is not None
    return max(cell_cap(st), _CONFIRM_CELLS if search else 60_000)


@contextlib.contextmanager
def _share_cores(workers: int):
    """Worker processes started inside this split this process's cores
    between them: LOADPATH_THREADS (numba kernels, field evaluation) is set
    to its current value (else the core count) // workers. Each worker
    otherwise ran a thread per core: three pitch starts on 4 cores ran 12
    threads, and a compiled kernel's barriers then waited on threads the OS
    had parked (an arm-link stiffness assembly 0.6 s against 0.05 s on one
    thread). Workers also start no background tet checks (_TetPrefetch):
    with every core taken, a mesher per worker only competes for them."""
    if workers <= 1:
        yield
        return
    saved = {k: os.environ.get(k) for k in ("LOADPATH_THREADS", "LOADPATH_TET_PREFETCH")}
    have = int(saved["LOADPATH_THREADS"] or (os.cpu_count() or 1))
    os.environ["LOADPATH_THREADS"] = str(max(1, have // workers))
    os.environ["LOADPATH_TET_PREFETCH"] = "0"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _parallel(studies: list, on_progress, should_stop) -> list:
    """Run independent single-goal studies, in processes when it helps.
    Progress inside a worker is not streamed; each finished outcome is."""
    workers = _workers(len(studies), cells=max(_study_cells(s) for s in studies)) if studies else 1
    if workers <= 1:
        return [_run_single(s, on_progress, should_stop) for s in studies]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    results = [None] * len(studies)
    with _share_cores(workers), ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_solve_alone, s): i for i, s in enumerate(studies)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            i = futs[f]
            results[i] = f.result()
            if on_progress is not None:
                for o in results[i].outcomes:
                    on_progress({"stage": "outcome", **o.summary()}, None)
    return results


# resolution.pitchStarts: the extra grids are this much coarser each (coarser
# so maxElements still holds), and "auto" runs 3 up to this many cells
PITCH_STEP = 0.01
PITCH_STARTS_MAX_CELLS = 30_000


def pitch_starts(study: Study) -> int:
    """How many grids the study's search runs on (resolution.pitchStarts).

    One grid's answer is a sample, not the answer: the search's mass swings
    with sub-percent changes of the pitch, because the optimizer settles in
    a different layout and the verified stress, read near supports and
    sharp junctions, differs between layouts. Over six pitches 0-5% apart,
    the 16 coarse-tier safety-factor cases ranged 24% from lightest to
    heaviest (median; 138% at worst, corpus-bracket-4 82-196 g) and 3 of 96
    runs failed. Paired runs of the coarse tier 3% apart in pitch, one grid
    against three (the lightest passing of grids 1% apart): massAtTarget
    spread mean 17% -> 10%, worst 80% -> 46%, delivered mass 8% lower, no
    lost passes, 2.7x the time; the median spread did not move
    (bench/results/pitch_starts.json, docs/BENCHMARKS.md)."""
    v = (study.raw.get("resolution") or {}).get("pitchStarts", "auto")
    if v == "auto" and os.environ.get("LOADPATH_PITCH_STARTS"):
        v = int(os.environ["LOADPATH_PITCH_STARTS"])     # the default, for a whole run (evals)
    if v != "auto":
        return int(v)
    g = study.goal
    if g.safety_factor is None and g.max_deflection is None and not _durability_drives(study):
        return 1
    if study.pitch:
        return 1
    return 3 if cell_cap(study) <= PITCH_STARTS_MAX_CELLS else 1


def _run_pitches(study: Study, starts: int, on_progress, should_stop) -> StudyResult:
    """The search on `starts` grids, PITCH_STEP coarser each; the result of
    the grid whose lightest part passes (or of the first grid when none
    does), with every grid's pick in result.starts."""
    import dataclasses

    p0 = float(study.pitch) if study.pitch else build_domain(study).pitch
    studies = [dataclasses.replace(study, pitch=p0 * (1.0 + PITCH_STEP * k)) for k in range(starts)]
    t0 = time.time()
    results = _parallel(studies, on_progress, should_stop)
    rows = []
    for st, r in zip(studies, results):
        o = r.outcomes[r.best] if r.best is not None else None
        rows.append({"pitch": st.pitch, "mass": o.report["mass"] if o is not None else None,
                     "passes": bool(o is not None and o.passes), "outcomes": len(r.outcomes)})
    passing = [i for i, x in enumerate(rows) if x["passes"]]
    k = min(passing, key=lambda i: rows[i]["mass"]) if passing else 0
    r = results[k]
    rows[k]["picked"] = True
    r.starts = rows
    r.study = study
    r.seconds = time.time() - t0
    return r


def run_study(study: Study, on_progress=None, should_stop=None) -> StudyResult:
    """on_progress(event: dict, xphys | None) is called with
    {"stage": "domain"|"optimize"|"mesh"|"verify"|"outcome", ...}.

    With several materials, the goal is solved for each (a stiffer material
    meets a deflection limit with less of it, a denser one weighs more for
    the same shape) and the outcomes are pooled; `best` is the lightest that
    passes, or the lightest overall when the goal has no pass/fail."""
    import dataclasses

    if study.process is not None and study.process.build == "auto":
        # opt-in: compare build directions on a coarse grid and keep the best
        from .orient import choose_build_direction

        study = choose_build_direction(study, on_progress=on_progress)["study"]
    mats = study.materials or [study.material]
    if len(mats) == 1:
        if study.goal.method == "auto":
            # the search (with its restarts and pitch starts), then perhaps
            # the constrained method once
            r = _run_auto(study, on_progress, should_stop,
                          search=lambda st: _run_restarts(st, on_progress, should_stop))
        else:
            r = _run_restarts(study, on_progress, should_stop)
        return _build_sim(_slice(_process_checks(r, on_progress), on_progress), on_progress)
    t0 = time.time()
    outcomes, warnings, domain = [], [], None
    for r in _parallel([dataclasses.replace(study, material=m, materials=[m]) for m in mats],
                       on_progress, should_stop):
        outcomes.extend(r.outcomes)
        warnings.extend(w for w in r.warnings if w not in warnings)
        domain = domain or r.domain
    passing = [i for i, o in enumerate(outcomes) if o.passes]
    pool = passing if passing else ([] if any(o.passes is False for o in outcomes) else range(len(outcomes)))
    best = min(pool, key=lambda i: outcomes[i].report["mass"]) if pool else None
    return _build_sim(_slice(_process_checks(StudyResult(study=study, domain=domain, outcomes=outcomes, best=best,
                                                         seconds=time.time() - t0, warnings=warnings), on_progress),
                             on_progress), on_progress)


# goal.restarts: the k-th search starts at the goal's volumeFraction times
# RESTART_STARTS[k] (the first is the study as written; then below, above,
# further below, further above)
RESTART_STARTS = (1.0, 0.8, 1.25, 0.65, 1.5, 0.55, 1.75, 0.45)


def _run_restarts(study: Study, on_progress=None, should_stop=None) -> StudyResult:
    """goal.restarts searches from different starting fractions, pooled: the
    outcomes of all, `best` the lightest of the searches' own (confirmed)
    picks. Each search's other outcomes keep their in-loop readings only, so
    they are not candidates."""
    import dataclasses

    def search(st):
        starts = pitch_starts(st)
        return (_run_single(st, on_progress, should_stop) if starts <= 1 else
                _run_pitches(st, starts, on_progress, should_stop))

    g = study.goal
    n = min(g.restarts, len(RESTART_STARTS))
    if n <= 1 or (g.safety_factor is None and g.max_deflection is None):
        return search(study)
    t0 = time.time()
    vf0 = float(g.volume_fraction or 0.3)
    outcomes, picks, warnings, domain = [], [], [], None
    for k in range(n):
        if k and should_stop is not None and should_stop():
            break
        vf = float(np.clip(vf0 * RESTART_STARTS[k], _VF_MIN, _VF_MAX))
        if on_progress is not None:
            on_progress({"stage": "restart", "run": k + 1, "of": n, "volumeFraction": vf}, None)
        st = dataclasses.replace(study, goal=dataclasses.replace(g, volume_fraction=vf))
        r = search(st)
        for o in r.outcomes:
            o.report["restart"] = k + 1
        if r.best is not None:
            picks.append(len(outcomes) + r.best)
        outcomes.extend(r.outcomes)
        warnings.extend(w for w in r.warnings if w not in warnings)
        domain = domain or r.domain
    ok = [i for i in picks if outcomes[i].passes]
    pool = ok or picks
    best = min(pool, key=lambda i: outcomes[i].report["mass"]) if pool else None
    if domain is not None:
        domain = dict(domain, restarts=[{"run": outcomes[i].report.get("restart"),
                                         "volumeFraction": outcomes[i].volume_fraction,
                                         "mass": outcomes[i].report.get("mass"), "passes": outcomes[i].passes}
                                        for i in picks])
    return StudyResult(study=study, domain=domain, outcomes=outcomes, best=best,
                       seconds=time.time() - t0, warnings=warnings)


def _build_sim(result: StudyResult, on_progress=None) -> StudyResult:
    """buildSim: simulate printing the delivered outcome (buildsim.py) into
    its report as "buildSim" (distortion, residual stress, recoater lift,
    bed peel), with its warnings; with compensate, also a pre-distorted
    mesh on the outcome (`compensated`, written as <name>-compensated.stl)."""
    bs = result.study.build_sim
    if bs is None:
        return result
    if result.best is None:
        if bs.get("printSetup"):
            return _print_setup_rescue(result, on_progress)
        return result
    o = result.outcomes[result.best]
    if getattr(o, "mesh", None) is None:
        return result
    import dataclasses as _dc
    from .buildsim import compensate_mesh, simulate_part

    st = result.study
    mat = (o.report.get("material") or {}).get("name") or bs["setup"].material
    setup = _dc.replace(bs["setup"], material=mat)
    build = st.process.build if st.process is not None and st.process.build not in (None, "auto") else "+z"
    if st.print_direction is not None:
        build = st.print_direction
    if on_progress is not None:
        on_progress({"stage": "buildSim", "process": setup.process}, None)
    try:
        if bs["compensate"]:
            comp, crep = compensate_mesh(o.mesh, setup, build=build, iterations=bs["iterations"], cells=bs["cells"])
            rep = simulate_part(o.mesh, setup, pitch=crep["pitch"], build=build).report
            rep["compensation"] = crep
            o.compensated = comp
        else:
            rep = simulate_part(o.mesh, setup, build=build, cells=bs["cells"]).report
    except Exception as exc:  # noqa: BLE001 - a failed simulation leaves the part's report standing
        rep = {"process": setup.process, "error": f"{type(exc).__name__}: {exc}", "warnings": []}
    o.report["buildSim"] = rep
    o.report.setdefault("warnings", []).extend(rep.get("warnings", []))
    if bs.get("printSetup"):
        # FDM: rank orientation, material, bed temperature and brim for this part
        from .printsetup import search_print_setup
        if on_progress is not None:
            on_progress({"stage": "printSetup"}, None)
        try:
            ps = search_print_setup(st, o.mesh, cells=min(bs["cells"], 4000), **bs["printSetup"])
            o.report["printSetup"] = ps
        except Exception as exc:  # noqa: BLE001 - the search is advice; the part's report stands
            o.report["printSetup"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _build_label(study):
    from .printsetup import _build_name
    return _build_name(study)


def _print_setup_rescue(result: StudyResult, on_progress=None) -> StudyResult:
    """No outcome passed (an FDM warp limit the shape cannot meet): run the
    print setup search on the lightest outcome that meets the rest of the
    goal, so the report says which orientation, material, bed temperature
    and brim would make it pass."""
    from .printsetup import search_print_setup
    st, bs = result.study, result.study.build_sim
    sf = st.goal.safety_factor

    def ok(o):
        return sf is None or (o.report.get("safetyFactor") or 0.0) >= sf

    pool = [o for o in result.outcomes if getattr(o, "mesh", None) is not None and ok(o)]
    if not pool:
        return result
    o = min(pool, key=lambda q: q.report.get("mass", float("inf")))
    if on_progress is not None:
        on_progress({"stage": "printSetup"}, None)
    try:
        ps = search_print_setup(st, o.mesh, cells=min(bs["cells"], 4000), **bs["printSetup"])
    except Exception as exc:  # noqa: BLE001
        ps = {"error": f"{type(exc).__name__}: {exc}"}
    o.report["printSetup"] = ps
    b = ps.get("best")
    re = ps.get("rerun")
    if not (b and b.get("passes")) and re and bs["printSetup"].get("reoptimize", True):
        # no setup makes this shape pass: re-optimize for the setup with
        # room under the warp limit and the most strength per gram
        from .printsetup import study_for
        if on_progress is not None:
            on_progress({"stage": "printSetup", "rerun": {k: re[k] for k in ("material", "build", "bed", "brim")}},
                        None)
        again = run_study(study_for(st, re), on_progress)
        again.warnings.insert(0, f"no outcome met the warp limit as {st.material.name}, build "
                                 f"{_build_label(st)}; re-optimized for {re['material']}, build {re['build']}, "
                                 f"bed {re['bed']:g} C" + (f", {re['brim']:g} mm brim" if re["brim"] else "")
                                 + " (report printSetup has the search)")
        if again.best is not None:
            again.outcomes[again.best].report["printSetup"] = {**ps, "reoptimizedFor": re}
        return again
    if b and b.get("passes"):
        result.warnings.append(
            f"no outcome met the warp limit as set up; the {o.volume_fraction:.0%} outcome does printed as "
            f"{b['material']}, build {b['build']}, bed {b['bed']:g} C"
            + (f", {b['brim']:g} mm brim" if b["brim"] else "") + f" ({b['mass']:.1f} g, lifts {b['lift']:.2f} mm, "
            f"SF {b['safetyFactor']:.2f}): set that in the study and run it again (report printSetup)")
    return result


def _process_checks(result: StudyResult, on_progress=None) -> StudyResult:
    """Powder-bed polymer and resin parts: the process checks
    (processcheck.check_process) on the delivered outcome, into its report
    as "process", with its warnings added to the outcome's. Every printed
    process also gets Materialize's published limits, the material id to
    quote it as and the build-volume fit, as "materialize"
    (dfm_reference.review)."""
    proc = result.study.process
    if proc is None or result.best is None:
        return result
    from .processcheck import RULES, check_process

    o = result.outcomes[result.best]
    if getattr(o, "mesh", None) is None:
        return result
    if proc.kind not in RULES:
        # FDM, metal, jetting: Materialize's limits and build volume only
        # (dfm_reference); their print checks are printcheck's
        from .dfm_reference import review

        mat = (o.report.get("material") or {}).get("name")
        ref = review(proc.kind, mat, extents=o.mesh.extents, wall_floor=proc.min_feature)
        if ref is not None:
            o.report["materialize"] = ref
            o.report.setdefault("warnings", []).extend(ref["warnings"])
        return result
    if on_progress is not None:
        on_progress({"stage": "process", "process": proc.kind}, None)
    try:
        mat = _material_named((o.report.get("material") or {}).get("name"), result.study)
        rep = check_process(o.mesh, proc, material=mat)
    except Exception as exc:  # noqa: BLE001 - a failed check leaves the part's report standing
        rep = {"process": proc.kind, "error": f"{type(exc).__name__}: {exc}", "ok": None, "warnings": []}
    o.report["process"] = rep
    if rep.get("materialize") is not None:
        o.report["materialize"] = rep["materialize"]
    o.report.setdefault("warnings", []).extend(rep.get("warnings", []))
    return result


# goal.minimize -> the slicer report's number it minimizes
_SLICED_COST = {"printTime": "printTime", "filament": "filamentVolume"}
# graded infill is sized for at most this many passing outcomes when the
# goal picks by print time (each is a few FEA solves)
_GRADE_CANDIDATES = 3


def _slice(result: StudyResult, on_progress=None) -> StudyResult:
    """FDM: slice every outcome (slicer.slice_mesh) for print time, filament
    and support, into its report as "slicer"; size graded infill
    (infill.py) when `slicer.grade` asks; and with goal.minimize printTime or
    filament, deliver the passing outcome that is cheapest by it rather than
    the lightest."""
    study = result.study
    opts = study.slicer
    if opts is None or opts.enabled is False or not result.outcomes:
        return result
    from .slicer import find_slicer, slice_mesh

    goal = study.goal
    if find_slicer(opts.executable) is None:
        if opts.enabled is True or goal.minimize != "mass" or opts.grade is not None:
            result.warnings.append("no slicer found, so print time and filament were not measured (and the "
                                   "outcome is the lightest): install PrusaSlicer, or set LOADPATH_SLICER")
        return result
    build = study.process.build if study.process.build != "auto" else None
    for i, o in enumerate(result.outcomes):
        if getattr(o, "mesh", None) is None:
            continue
        if on_progress is not None:
            on_progress({"stage": "slice", "volumeFraction": o.volume_fraction}, None)
        mat = _material_named((o.report.get("material") or {}).get("name"), study)
        try:
            o.report["slicer"] = slice_mesh(o.mesh, build, mat, opts, name=f"{study.name}-{i}")
        except Exception as exc:  # noqa: BLE001 - a failed slice leaves the part's report standing
            o.report["slicer"] = {"error": f"{type(exc).__name__}: {exc}"}
    cost = _SLICED_COST.get(goal.minimize)
    pool = [i for i, o in enumerate(result.outcomes) if o.passes] if any(o.passes is not None
                                                                       for o in result.outcomes) else []
    if opts.grade is not None:
        if cost is not None and pool:
            light = sorted(pool, key=lambda i: result.outcomes[i].report["mass"])[:_GRADE_CANDIDATES]
            todo = sorted(set(light) | ({result.best} if result.best is not None else set()))
        else:
            todo = [result.best] if result.best is not None else []
        for i in todo:
            _grade(result, i, opts, on_progress)
    if cost is not None and pool:
        def sliced(i):
            # the cheaper of its solid and graded prints: both are checked,
            # and a thin part can print slower graded (region boundaries)
            r = result.outcomes[i].report
            vals = [(r.get(k) or {}).get(cost) for k in ("slicer",)] + \
                [((r.get("graded") or {}).get("slicer") or {}).get(cost)]
            vals = [v for v in vals if v is not None]
            return min(vals) if vals else float("inf")
        pick = min(pool, key=lambda i: (sliced(i), result.outcomes[i].report["mass"]))
        if sliced(pick) < float("inf") and pick != result.best:
            prev = result.best
            result.best = pick
            if opts.grade is not None and result.outcomes[pick].grading is None:
                _grade(result, pick, opts, on_progress)
            result.warnings.append(
                f"goal.minimize {goal.minimize}: delivered outcome {pick} (vf {result.outcomes[pick].volume_fraction:.3f}, "
                f"{result.outcomes[pick].report['mass']:.1f} g) over the lightest passing, outcome {prev}")
    return result


def _grade(result: StudyResult, i: int, opts, on_progress=None) -> None:
    """Graded infill for outcome i: report["graded"] (sizing, and its slice
    next to the solid one) and o.grading (the modifier meshes)."""
    from .infill import GradeOptions, grade_infill
    from .slicer import slice_mesh

    study = result.study
    o = result.outcomes[i]
    mat = _material_named((o.report.get("material") or {}).get("name"), study)
    import dataclasses
    st = dataclasses.replace(study, material=mat, materials=[mat])
    if on_progress is not None:
        on_progress({"stage": "grade", "volumeFraction": o.volume_fraction}, None)
    try:
        g = grade_infill(st, o.mesh, o.report, GradeOptions.from_spec(opts.grade, study.goal.safety_factor))
    except Exception as exc:  # noqa: BLE001
        o.report["graded"] = {"error": f"{type(exc).__name__}: {exc}"}
        return
    o.grading = g
    rep = {k: v for k, v in g.items() if k not in ("modifiers", "history")}
    rep["modifiers"] = [{"name": m["name"], "settings": m["settings"], "faces": int(len(m["mesh"].faces))}
                        for m in g["modifiers"]]
    build = study.process.build if study.process.build != "auto" else None
    try:
        rep["slicer"] = slice_mesh(o.mesh, build, mat, opts, name=f"{study.name}-{i}-graded",
                                   object_settings=g["objectSettings"], modifiers=g["modifiers"])
        solid = o.report.get("slicer") or {}
        if solid.get("printTime") and rep["slicer"].get("printTime"):
            rep["printTimeSaved"] = 1.0 - rep["slicer"]["printTime"] / solid["printTime"]
            rep["filamentSaved"] = 1.0 - rep["slicer"]["filamentVolume"] / solid["filamentVolume"]
    except Exception as exc:  # noqa: BLE001
        rep["slicer"] = {"error": f"{type(exc).__name__}: {exc}"}
    goal = study.goal.safety_factor
    if goal and g["safetyFactor"] < goal:
        o.report.setdefault("warnings", []).append(
            f"graded infill reads SF {g['safetyFactor']:.2f} under the goal's {goal:g}: print the solid 3MF")
    o.report["graded"] = rep



def mesh_fidelity(domain, xphys, mesh) -> dict | None:
    """How much of the optimized field the delivered mesh kept: the share of
    solid cells (xphys > 0.5) whose centre lies outside the mesh, overall, on
    the surface of the solid, and in thin members (gone under a one-cell
    opening). The camera boom (2026-11-01) lost 21% of its solid cells: 44%
    of its thin members and 34% of its surface cells, against 8% inside.
    Meshed, it read 16x softer than the optimizer's own model on the same
    grid and supports, so the run reported SF 0.99 for a field that carries
    the load with room to spare."""
    from scipy import ndimage

    from .voxels import mesh_solid

    solid = np.asarray(xphys) > 0.5
    if not solid.any() or mesh is None or len(getattr(mesh, "faces", ())) == 0:
        return None
    step = domain.pitch / 2.0
    lo = np.asarray(domain.origin, float)
    hi = lo + np.asarray(domain.dims) * domain.pitch
    try:
        g, org, *_ = mesh_solid(mesh, step, pad=2, bounds=(lo, hi))
    except (MemoryError, ValueError):
        return None
    c = domain.mesh.elem_centers()
    ijk = np.clip(np.floor((c - np.asarray(org)) / step).astype(int), 0, np.array(g.shape) - 1)
    inside = g[tuple(ijk.T)]
    at = tuple(domain.mesh.elem_ijk.T)
    grid = np.zeros(domain.dims, bool)
    grid[at] = solid
    cross = ndimage.generate_binary_structure(3, 1)
    thin = (grid & ~ndimage.binary_opening(grid, structure=cross))[at]
    surface = (grid & ~ndimage.binary_erosion(grid, structure=cross))[at] & ~thin
    lost = solid & ~inside

    def share(m):
        return float((lost & m).sum() / max(int(m.sum()), 1))

    return {"lostSolidCells": share(solid), "lostThinCells": share(thin), "lostSurfaceCells": share(surface),
            "thinCells": float(thin.sum() / solid.sum())}


MESH_LOSS_WARN = 0.10

# A fragment meshing cut loose and dropped is reported when the field had
# it joined to the part (a reconstruction artifact: any of it past this
# share of the part's volume), or when the drop is large anyway.
FRAGMENT_SEVERED_WARN = 0.005
FRAGMENT_DROPPED_WARN = 0.02


def fragment_warnings(rinfo: dict | None, mesh) -> list[str]:
    """Warnings for material reconstruction dropped as loose fragments.

    The fan plate's ribs were severed at its duct rim's face by meshing and
    dropped as fragments: 113-137 cm^3 of a 468 cm^3 design, and the only
    word of it was the mesh-fidelity warning's share of lost cells. Say how
    much went, where, and whether the optimized field had it joined to the
    part (then it is meshing's fault, not the design's)."""
    if not rinfo:
        return []
    dropped = float(rinfo.get("droppedFragmentVolume") or 0.0)
    if dropped <= 0.0:
        return []
    vol = max(float(getattr(mesh, "volume", 0.0) or 0.0), 1e-9)
    severed = float(rinfo.get("severedFragmentVolume") or 0.0)
    frags = rinfo.get("droppedFragments") or []

    def where(fs):
        return "; ".join(f"{f['volume']:.0f} mm^3 at x {f['bboxMin'][0]:.1f}..{f['bboxMax'][0]:.1f}, "
                         f"y {f['bboxMin'][1]:.1f}..{f['bboxMax'][1]:.1f}, z {f['bboxMin'][2]:.1f}..{f['bboxMax'][2]:.1f}"
                         for f in fs[:3])

    out = []
    if severed > FRAGMENT_SEVERED_WARN * vol:
        cut = [f for f in frags if f.get("connectedInField")]
        out.append(
            f"meshing cut {severed:.0f} mm^3 ({severed / vol:.0%} of the delivered part) loose from material the "
            f"optimized field had joined to the part, and dropped it ({where(cut)}): a reconstruction artifact, "
            "not the design. The delivered part is lighter and weaker than the one optimized, so a search goes "
            "heavier than it needs. Re-mesh the saved field (`loadpath remesh`) after a fix, or report it with "
            "the study and the field.")
    elif dropped > FRAGMENT_DROPPED_WARN * vol:
        out.append(f"meshing dropped {dropped:.0f} mm^3 ({dropped / vol:.0%} of the delivered part) as loose "
                   f"fragments the optimized field also had apart from the part ({where(frags)}).")
    return out

def _material_named(name, study):
    """The study's Material called `name` (a study can try several)."""
    for m in study.materials or [study.material]:
        if m.name == name:
            return m
    return study.material


def _constrained_applies(study: Study) -> bool:
    goal = study.goal
    return (goal.safety_factor is not None or goal.max_deflection is not None) \
        and study.gravity is None and not _durability_drives(study) and not goal.clearances \
        and study.access is None and study.max_member is None and not study.inertia_relief \
        and not any(ld.direction for c in study.load_cases for ld in c.loads) \
        and not any(getattr(c, "safety_factor", None) for c in study.load_cases)


def _run_auto(study: Study, on_progress=None, should_stop=None, search=None) -> StudyResult:
    """goal.method auto: the search, then the constrained method only when the
    search's part idles (stressUtilization under _AUTO_UTILIZATION), or as
    a probe that stops once it cannot win (under _AUTO_PROBE_UTILIZATION),
    keeping the lighter passing part. The constrained method costs about as much as
    the search again, so it runs only where it tends to win."""
    import dataclasses

    goal = study.goal
    st = dataclasses.replace(study, goal=dataclasses.replace(goal, method="search"))
    cst = dataclasses.replace(study, goal=dataclasses.replace(goal, method="constrained"))
    spec = _Speculative(cst) if _Speculative.wanted(study, st) else None
    try:
        return _run_auto_rest(study, st, cst, spec, on_progress, should_stop, search)
    finally:
        if spec is not None:
            spec.cancel()


def _run_auto_rest(study, st, cst, spec, on_progress, should_stop, search) -> StudyResult:
    import dataclasses

    goal = study.goal
    # `search` runs the search side (run_study: restarts and pitch starts)
    r = search(st) if search is not None else _run_single(st, on_progress, should_stop)
    choice = {"method": "search"}
    pick = r.outcomes[r.best] if r.best is not None else None
    u = pick.report.get("stressUtilization") if pick is not None else None
    if u is not None:
        choice["stressUtilization"] = u
    if pick is None or not pick.passes or u is None or not _constrained_applies(study):
        choice["reason"] = "the constrained method does not apply" if pick is not None and pick.passes \
            else "no passing search part to compare"
        r.choice = choice
        return r
    rep = pick.report
    sf = rep.get("safetyFactor")
    if goal.max_deflection and rep.get("cases") \
            and max(c.get("deflection") or 0.0 for c in rep["cases"]) >= 0.85 * goal.max_deflection \
            and (goal.safety_factor is None or (sf is not None and sf >= 1.3 * goal.safety_factor)):
        # stiffness sets the mass (the arm link, the robot-arm upper arm at SF
        # 11 for 2): the compliance search is already the right design
        choice["reason"] = "deflection-bound: the search's compliance design is the right one"
        r.choice = choice
        return r
    from .verify import use_cut_cells
    limit = _AUTO_UTILIZATION_CUT if use_cut_cells(study) else _AUTO_UTILIZATION
    probe = u >= limit
    outer = u >= _AUTO_PROBE_UTILIZATION
    if u >= _AUTO_PROBE_OUTER:
        choice["reason"] = (f"stress utilization {u:.2f} >= {_AUTO_PROBE_OUTER}: "
                            "the search part is worked evenly")
        r.choice = choice
        return r
    if should_stop is not None and should_stop():
        r.choice = choice
        return r
    if on_progress is not None:
        on_progress({"stage": "method", "method": "constrained", "stressUtilization": u,
                     **({"probe": True} if probe else {})}, None)
    # time box: the constrained method runs on a finer grid and took about
    # 2.9x the search on the yoke (2.5x cut it before its 16 g run); on the
    # wrist link it found no passing part in 90 minutes (the search: 24).
    # Past the budget the runs it has stand.
    # The buckling constraint's eigensolve roughly doubles a constrained
    # iteration on a full-size grid (10-25 s of 25-50 at 35k cells on the
    # wall bracket), so it gets twice the time.
    budget = _AUTO_BUDGET * (2.0 if goal.buckling and goal.safety_factor and _BUCKLING_IN_OPTIMIZER
                             and not probe else 1.0)
    from . import quick
    if quick.on():
        # a first look: as long as the search took, with no ten-minute floor
        # (the urdf tutorial's quick run spent 20 of its 30 CI minutes here)
        box = r.seconds
    else:
        box = max(600.0, budget * r.seconds)
    # the box runs from the constrained run's own start (a speculative one
    # started beside the search)
    start = spec.t0 if spec is not None else time.time()
    deadline = start + box
    done = [0]
    beaten = [None]   # a probe's failing part at or over the search's mass
    won = [False]     # a probe's passing part lighter than the search's
    promising = [False]   # ... or one near its mass with room to lighten
    settled = [False]
    ms = pick.report["mass"]

    pending = [None, False]   # a probe's last part, awaiting its tet confirm

    def judge(event, margin=None):
        if beaten[0] is None and _probe_beaten(goal, event, ms, margin):
            beaten[0] = event.get("mass")
        won[0] = won[0] or _probe_won(event, ms, margin)
        mg = margin if margin is not None else event.get("margin")
        promising[0] = promising[0] or won[0] or (
            mg is not None and mg >= _PROBE_ROOM and (event.get("mass") or np.inf) < _PROBE_NEAR * ms)
        # judged between parts (after the tet confirm): stop here rather
        # than cut the next part short at the deadline
        if not quick.on() and _probe_settled(won[0], start, deadline, done[0]):
            settled[0] = True

    def progress(event, x):
        stage = event.get("stage")
        if probe:
            if stage == "confirm" and pending[0] is not None:
                pending[1] = True
            elif stage == "confirmed" and pending[0] is not None:
                # the tet reading stands over the voxel one (it lifts a
                # voxel safety factor by up to 2.5x)
                judge(pending[0], event.get("margin"))
                pending[0] = None
            elif pending[0] is not None and not pending[1]:
                judge(pending[0])
                pending[0] = None
        if stage == "outcome":
            done[0] += 1
            pending[0], pending[1] = event, False
        if on_progress is not None:
            on_progress(event, x)

    def user_stop():
        if probe and pending[0] is not None and not pending[1]:
            # the last part, with no tet confirm under way: judge it now,
            # not at the next event (the next part would start first)
            judge(pending[0])
            pending[0] = None
        # past the outer band: one part, kept going only if it is promising
        short = outer and done[0] >= 1 and pending[0] is None and not promising[0]
        return beaten[0] is not None or settled[0] or short or (should_stop is not None and should_stop())

    def stop():
        over = time.time() > deadline if quick.on() else _box_over(deadline, box, done[0])
        return over or user_stop()

    cstudy = dataclasses.replace(study, goal=dataclasses.replace(goal, method="constrained"))
    fit = _constrained_fit(cstudy)
    if fit is not None and fit.get("skip"):
        # the bonus must not take the run down: a shelf bracket's 78k-cell
        # constrained grid (4x maxElements) was OOM-killed at 6.3 GB on a
        # shared 15 GB box, and the search's passing part went with it
        choice["reason"] = fit["skip"]
        if on_progress is not None:
            on_progress({"stage": "warning", "message": fit["skip"]}, None)
        r.choice = choice
        return r
    if fit is not None and fit.get("pitch"):
        cstudy = dataclasses.replace(cstudy, pitch=fit["pitch"])
    if quick.on() and not study.pitch:
        # and on the search's own coarse grid, not the 4x finer one a full
        # run gives it
        from .domain import choose_pitch
        cstudy = dataclasses.replace(cstudy, pitch=max(choose_pitch(st), cstudy.pitch or 0.0))
    try:
        if spec is not None:
            # running since the search started (on cstudy: wanted() takes no
            # quick run and no grid made coarser for memory): the same time
            # box, from its start
            rc = spec.finish(box, progress, user_stop)
        else:
            rc = _run_single(cstudy, progress, stop, fallback=False)
    except Exception as exc:  # noqa: BLE001 - the constrained run is a bonus: any failure keeps the search's part
        choice["reason"] = f"the constrained method failed ({exc})"
        r.choice = choice
        return r
    mc = None
    if rc.best is not None and rc.outcomes[rc.best].passes:
        mc = rc.outcomes[rc.best].report["mass"]
    choice.update({"searchMass": ms, "constrainedMass": mc})
    if mc is not None and mc < ms:
        if probe:
            choice["probe"] = True
        if settled[0]:
            choice["probeSettled"] = True
        if outer:
            choice["probeOuter"] = True
        choice["method"] = "constrained"
        choice["reason"] = (f"stress utilization {u:.2f}: constrained {mc:.1f} g against "
                            f"search {ms:.1f} g")
        rc.choice = choice
        rc.seconds += r.seconds
        return rc
    if probe:
        choice["probe"] = True
    choice["reason"] = (f"stress utilization {u:.2f}: constrained "
                        + (f"{mc:.1f} g" if mc is not None else "found no passing part")
                        + f" against search {ms:.1f} g"
                        + (f" (probe stopped: a {beaten[0]:.1f} g part failed)" if beaten[0] is not None else ""))
    if settled[0]:
        choice["probeSettled"] = True
    if outer:
        choice["probeOuter"] = True
    r.choice = choice
    r.seconds += rc.seconds
    return r


def _probe_won(event: dict, search_mass: float, margin: float | None = None) -> bool:
    """Whether a constrained part (its outcome event) passes while weighing
    less than the search's part. `margin`, when given, is its tet-confirmed
    margin and decides over the voxel one."""
    m = event.get("mass")
    mg = margin if margin is not None else event.get("margin")
    return m is not None and mg is not None and m < search_mass and mg >= 1.0


def _probe_settled(won: bool, start: float, deadline: float, done: int, now: float | None = None) -> bool:
    """Whether a probe that has run `done` parts since `start` should stop
    before its next one: it already holds a winning part (`won`), has run
    _PROBE_ROUNDS parts, and another, at the mean time of those so far,
    would end past the deadline."""
    if not won or done < _PROBE_ROUNDS:
        return False
    now = time.time() if now is None else now
    return now + (now - start) / done > deadline


def _probe_beaten(goal, event: dict, search_mass: float, margin: float | None = None) -> bool:
    """Whether a constrained part (its outcome event) fails the target while
    weighing at least _PROBE_HEAVY times the search's passing part: then the
    constrained method cannot beat the search here (the next parts only
    add material to pass). `margin`, when given, is the part's tet-confirmed
    margin and decides alone."""
    m = event.get("mass")
    if m is None or m < _PROBE_HEAVY * search_mass:
        return False
    if margin is not None:
        return margin < 1.0
    sf = event.get("safetyFactor")
    if goal.safety_factor and sf is not None and sf < goal.safety_factor:
        return True
    bk = event.get("bucklingFactor")
    if goal.buckling and goal.safety_factor and bk is not None and bk < goal.safety_factor:
        return True
    if goal.max_deflection and event.get("cases"):
        d = max((c.get("deflection") or 0.0) for c in event["cases"])
        return d > goal.max_deflection
    return False


# the constrained method's peak over _run_bytes (MMA's history and the stress
# aggregate's adjoints): 6.3 GB measured at 78k cells against 4.9 estimated
_CONSTRAINED_MEMORY = 1.3


def _constrained_fit(study: Study) -> dict | None:
    """Whether goal.method auto's constrained run fits the free memory:
    None (it fits, or nothing to go by), {"pitch": p} for a coarser grid
    that fits, or {"skip": why} when even that would be under 1.5x the
    search's cells (no finer than the search, so no point)."""
    import dataclasses

    if study.pitch:
        return None
    avail = _available_bytes()
    if not avail:
        return None
    from .domain import build_domain, cell_cap, choose_pitch

    p = choose_pitch(study)
    n = build_domain(study, p).n_elem
    budget = 0.85 * avail   # only near the edge: evals size these grids on purpose
    if _CONSTRAINED_MEMORY * _run_bytes(n) <= budget:
        return None
    cells = (budget / _CONSTRAINED_MEMORY - _RUN_BASE_BYTES) / _RUN_CELL_BYTES
    search_cells = cell_cap(dataclasses.replace(study, goal=dataclasses.replace(study.goal, method="search")))
    if cells < 1.5 * search_cells:
        return {"skip": f"the constrained method's {n}-cell grid needs about "
                        f"{_CONSTRAINED_MEMORY * _run_bytes(n) / 1e9:.1f} GB and {avail / 1e9:.1f} GB is free: "
                        "skipped, the search's part stands"}
    return {"pitch": p * (n / cells) ** (1.0 / 3.0) * 1.01}


def _constrained_main(work: str) -> None:
    """Child process entry (_Speculative): the constrained method on the
    study in `work`/in.pkl, its progress events to `work`/events.jsonl,
    stopping at the epoch time in `work`/deadline (once written) or when
    `work`/stop appears; writes `work`/out.pkl ("ok", StudyResult) or
    ("error", exc)."""
    import json
    import pickle

    with open(os.path.join(work, "in.pkl"), "rb") as fh:
        study = pickle.load(fh)
    events = open(os.path.join(work, "events.jsonl"), "a")

    done = [0]

    def note(event, _x=None):
        if event.get("stage") == "outcome":
            done[0] += 1
        try:
            events.write(json.dumps(event, default=float) + "\n")
            events.flush()
        except (TypeError, ValueError, OSError):
            pass

    def stop():
        if os.path.exists(os.path.join(work, "stop")):
            return True
        try:
            with open(os.path.join(work, "deadline")) as fh:
                deadline, box = (float(x) for x in fh.read().split())
            return _box_over(deadline, box, done[0])
        except (OSError, ValueError):
            return False

    try:
        out = ("ok", _run_single(study, note, stop, fallback=False))
        for o in out[1].outcomes:
            # (as float32: the pickle stays small, and the field comes back
            # to be saved beside the outcome for re-meshing)
            o.xphys = None if o.xphys is None else np.asarray(o.xphys, np.float32)
    except Exception as exc:  # noqa: BLE001 - raised again in the search's process
        out = ("error", exc)
    with open(os.path.join(work, "out.tmp.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    os.replace(os.path.join(work, "out.tmp.pkl"), os.path.join(work, "out.pkl"))


class _Speculative:
    """goal.method auto's constrained run, started beside the search rather
    than after it. Whether it runs at all depends on the search's pick (its
    stress utilization), but it does not start from anything the search
    finds, and the search of a single grid keeps about one core of four
    busy: on the robot-arm turret the constrained method ran 50+ minutes
    after a 17-minute search. Started together, it has the search's time
    in hand; dropped (killed) when the pick says it does not apply. Its
    time box is the same, counted from its own start. Only where a core is
    free for it: one grid (pitch starts run a process each), at least 3
    cores and LOADPATH_THREADS (when set) at least 3, and memory to spare.
    LOADPATH_SPECULATE=0 turns it off."""

    def __init__(self, study: Study):
        import pickle
        import subprocess
        import sys
        import tempfile

        self.work = tempfile.mkdtemp(prefix="loadpath-constrained-")
        self.t0 = time.time()
        self.read = 0
        with open(os.path.join(self.work, "in.pkl"), "wb") as fh:
            pickle.dump(study, fh)
        cores = os.cpu_count() or 1
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, LOADPATH_THREADS=str(max(1, cores // 2)), LOADPATH_SPECULATE="0",
                   PYTHONPATH=os.pathsep.join([pkg_parent] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
        code = "import sys; from loadpath.study import _constrained_main; _constrained_main(sys.argv[1])"
        with open(os.path.join(self.work, "log.txt"), "wb") as log:
            self.proc = subprocess.Popen([sys.executable, "-c", code, self.work], env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)

    @staticmethod
    def wanted(study: Study, st: Study) -> bool:
        from . import quick
        if os.environ.get("LOADPATH_SPECULATE", "1") == "0" or not _constrained_applies(study) or quick.on():
            return False
        cores = os.cpu_count() or 1
        env = os.environ.get("LOADPATH_THREADS")
        if cores < 3 or (env and int(env) < 3):
            return False
        if pitch_starts(st) > 1 or len(study.materials or [study.material]) > 1:
            return False
        import dataclasses
        cst = dataclasses.replace(study, goal=dataclasses.replace(study.goal, method="constrained"))
        if _constrained_fit(cst) is not None:
            return False   # its grid is made coarser to fit, or it is skipped
        avail = _available_bytes()
        return avail is None or avail > 2 * _run_bytes(_study_cells(study))

    def _replay(self, on_progress) -> None:
        import json

        try:
            with open(os.path.join(self.work, "events.jsonl")) as fh:
                lines = fh.read().splitlines()
        except OSError:
            return
        for line in lines[self.read:]:
            self.read += 1
            if on_progress is not None:
                try:
                    on_progress(json.loads(line), None)
                except ValueError:
                    pass

    def finish(self, seconds: float, on_progress=None, should_stop=None) -> StudyResult:
        """Its result, time-boxed to `seconds` from its start (raising what
        it raised)."""
        import pickle

        with open(os.path.join(self.work, "deadline.tmp"), "w") as fh:
            fh.write(f"{self.t0 + seconds!r} {seconds!r}")
        os.replace(os.path.join(self.work, "deadline.tmp"), os.path.join(self.work, "deadline"))
        try:
            while self.proc.poll() is None:
                self._replay(on_progress)
                if should_stop is not None and should_stop():
                    open(os.path.join(self.work, "stop"), "w").close()
                try:
                    self.proc.wait(1.0)
                except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired
                    pass
            self._replay(on_progress)
            try:
                with open(os.path.join(self.work, "out.pkl"), "rb") as fh:
                    kind, val = pickle.load(fh)
            except (OSError, EOFError, pickle.UnpicklingError):
                raise RuntimeError(f"the constrained run's process exited with code {self.proc.returncode}")
        finally:
            self.cancel()
        if kind == "error":
            raise val
        return val

    def cancel(self) -> None:
        import shutil
        import signal

        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (AttributeError, OSError):
                self.proc.kill()
            self.proc.wait()
        shutil.rmtree(self.work, ignore_errors=True)


def _prefetch_main(work: str) -> None:
    """Child process entry (_TetPrefetch): cross-check the outcome in
    `work`/in.pkl, write `work`/out.pkl ("ok", result) or ("error", exc)."""
    import pickle

    import trimesh

    from .crosscheck import crosscheck

    with open(os.path.join(work, "in.pkl"), "rb") as fh:
        study, v, f, band = pickle.load(fh)
    try:
        out = ("ok", crosscheck(study, trimesh.Trimesh(v, f, process=False), contact_band=band))
    except Exception as exc:  # noqa: BLE001 - raised again in the search's process
        out = ("error", exc)
    with open(os.path.join(work, "out.tmp.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    os.replace(os.path.join(work, "out.tmp.pkl"), os.path.join(work, "out.pkl"))


class _TetPrefetch:
    """Tet cross-checks of outcomes the search may confirm later, started in
    a background process as each outcome comes out, while the search
    optimizes the next fraction. The search confirms its picks only once
    its fraction loop ends (the MBB beam's first part, at 30%, was
    confirmed five optimizations later), and the optimizer keeps about one
    core of four busy, so the mesher and the quadratic solve, a fifth to
    a quarter of a search, ran on cores that sat idle. Same function, same
    inputs (fTetWild's mesh is cached by surface in either process), so
    the same readings; an outcome never prefetched is checked inline as
    before. One at a time, the lightest passing part's: a lighter one
    that passes stops the heavier one's check. Only where a core is free
    (LOADPATH_THREADS, when set, at least 2). A plain subprocess,
    not multiprocessing: spawning re-runs a caller's script that has no
    __main__ guard. LOADPATH_TET_PREFETCH=0 turns it off."""

    def __init__(self):
        self.jobs = []      # [outcome, work dir, Popen or None (queued)]
        # only with a core to spare: not in a pitch start's worker
        # (study._share_cores) or an eval case on one core (evals --jobs)
        cores = int(os.environ.get("LOADPATH_THREADS") or (os.cpu_count() or 1))
        self.on = os.environ.get("LOADPATH_TET_PREFETCH", "1") != "0" and cores >= 2

    def submit(self, study: Study, o, band: float) -> None:
        if not self.on or getattr(o, "mesh", None) is None or any(j[0] is o for j in self.jobs):
            return
        avail = _available_bytes()
        if avail is not None and avail < 3e9:
            return   # not with the memory for a second solve
        import pickle
        import tempfile

        work = tempfile.mkdtemp(prefix="loadpath-tets-")
        with open(os.path.join(work, "in.pkl"), "wb") as fh:
            pickle.dump((study, np.asarray(o.mesh.vertices, np.float64), np.asarray(o.mesh.faces, np.int64),
                         float(band)), fh)
        self.jobs.append([o, work, None])
        self._start()

    def drop(self, o) -> None:
        """Forget `o`'s job (stopping it if it runs): a lighter part passed."""
        for j in list(self.jobs):
            if j[0] is o:
                self._remove(j)

    def _start(self) -> None:
        """Start the newest queued job when none is running."""
        if any(j[2] is not None and j[2].poll() is None for j in self.jobs):
            return
        queued = [j for j in self.jobs if j[2] is None]
        if not queued:
            return
        import subprocess
        import sys

        j = queued[-1]
        # the cores the search leaves: it keeps about one busy
        threads = max(1, int(os.environ.get("LOADPATH_THREADS") or (os.cpu_count() or 1)) - 1)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, LOADPATH_THREADS=str(threads),
                   PYTHONPATH=os.pathsep.join([pkg_parent] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
        code = "import sys; from loadpath.study import _prefetch_main; _prefetch_main(sys.argv[1])"
        log = open(os.path.join(j[1], "log.txt"), "wb")
        try:
            j[2] = subprocess.Popen([sys.executable, "-c", code, j[1]], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError:
            self.on = False
            self._remove(j)
        finally:
            log.close()

    def take(self, o) -> dict | None:
        """The prefetched cross-check of `o` (raising what it raised), or
        None when there is none: queued or not prefetched, it is checked
        inline (and a queued job is dropped)."""
        import pickle

        j = next((j for j in self.jobs if j[0] is o), None)
        if j is None:
            return None
        if j[2] is None:
            self._remove(j)
            return None
        j[2].wait()
        out = os.path.join(j[1], "out.pkl")
        try:
            with open(out, "rb") as fh:
                kind, val = pickle.load(fh)
        except (OSError, EOFError, pickle.UnpicklingError):
            kind, val = None, None   # the child died (memory): check inline
        self._remove(j)
        self._start()
        if kind == "error":
            raise val
        return val if kind == "ok" else None

    def _remove(self, j) -> None:
        import shutil

        p = j[2]
        if p is not None and p.poll() is None:
            self._kill(p)
        shutil.rmtree(j[1], ignore_errors=True)
        self.jobs.remove(j)

    @staticmethod
    def _kill(p) -> None:
        import signal

        try:
            os.killpg(p.pid, signal.SIGKILL)   # with fTetWild's own child
        except (AttributeError, OSError):
            p.kill()
        p.wait()

    def close(self) -> None:
        for j in list(self.jobs):
            self._remove(j)


def _run_single(study: Study, on_progress=None, should_stop=None, fallback: bool = True) -> StudyResult:
    if study.goal.method == "auto":
        return _run_auto(study, on_progress, should_stop)
    prefetch = _TetPrefetch()
    try:
        return _search(study, on_progress, should_stop, fallback, prefetch)
    finally:
        prefetch.close()


def _search(study: Study, on_progress, should_stop, fallback: bool, prefetch: _TetPrefetch) -> StudyResult:
    t0 = time.time()

    def emit(event, field_=None):
        if on_progress is not None:
            on_progress(event, field_)

    domain = build_domain(study)
    radius = filter_radius(domain, study.length_scale, study.min_hole)
    info = _domain_info(domain, radius)
    emit({"stage": "domain", **info, "warnings": domain.warnings})
    outcomes: list[Outcome] = []
    run_notes: list[str] = []
    started = False
    # what earlier runs on this domain found (recall.py), when turned on
    from .recall import Recall
    memory = Recall.open(study, domain, radius, _making(study))
    # FDM warp limits are checked on each outcome's delivered part,
    # re-simulated (the optimizer's coarse grid reads FDM warp 2-3.5x low,
    # and the ratio moves from design to design). The optimizer carries
    # them only with buildSim.warpInOptimizer; then its limits for the next
    # attempt are scaled by how far it read off on the last one. Metal
    # limits stay in the optimizer and are checked on its full grid.
    ws = domain.warp
    bsim = study.build_sim or {}
    limits0 = (bsim.get("maxDistortion"), bsim.get("maxRecoaterLift")) \
        if bsim and (bsim.get("maxDistortion") or bsim.get("maxRecoaterLift")) \
        and bsim["setup"].process.lower() == "fdm" else None

    def _check_warp(warp, mesh):
        warp["userLimits"] = {"maxDistortion": limits0[0], "maxRecoaterLift": limits0[1]}
        d = _warp_delivered(study, mesh)
        warp["delivered"] = d
        if not d or "error" in d or ws is None:
            return
        for key, attr, lim in (("maxDistortion", "max_distortion", limits0[0]),
                               ("recoaterLift", "max_lift", limits0[1])):
            if lim and d.get(key) and warp.get(key):
                ratio = d[key] / warp[key]
                setattr(ws, attr, lim / ratio)
                warp.setdefault("deliveredOverOptimizer", {})[key] = round(ratio, 3)

    goal_ = study.goal
    buck_state = {"on": False, "kb": 1.0,
                  "sf": goal_.safety_factor if (goal_.buckling and goal_.safety_factor and _BUCKLING_SEARCH
                                                and study.gravity is None and study.thermal is None) else None}

    def attempt(vf: float, x0, beta0, replay=None) -> Outcome:
        def on_iter(rec, xp):
            emit({"stage": "optimize", "volumeFraction": vf, **rec}, xp)

        if replay is None:
            breq = buck_state["sf"] * buck_state["kb"] if buck_state["on"] else None
            res = optimize(domain, study.elastic(design=True),
                           SimpOptions(volume_fraction=vf, filter_radius=radius,
                                       max_iterations=study.max_iterations, x0=x0, beta_start=beta0,
                                       buckling=breq, buckling_modulus=study.material.E,
                                       **_making(study)),
                           on_iter=on_iter, should_stop=should_stop)
            x, xphys = res.x, res.xphys
            warp = res.warp
            opt = {"iterations": res.iterations, "converged": res.converged, "seconds": res.seconds,
                   "compliance": res.compliance, "caseCompliance": res.case_compliance,
                   **({"caseDirection": res.case_direction} if any(d is not None for d in res.case_direction) else {}),
                   "achievedVolumeFraction": res.volume_fraction,
                   **({"escape": res.escape} if res.escape else {})}
        else:
            # a stored design for this very problem: mesh and verify it, no optimizing
            x, xphys = replay[0], replay[1]
            warp = None
            opt = {"iterations": 0, "converged": True, "seconds": 0.0, "recalled": True,
                   "achievedVolumeFraction": float(xphys.mean())}
        emit({"stage": "mesh", "volumeFraction": vf})
        mesh, rinfo = reconstruct(study, domain, xphys, ReconstructOptions())
        emit({"stage": "verify", "volumeFraction": vf})
        try:
            report = verify_part(study, mesh, min_pitch=0.5 * domain.pitch, contact_band=domain.pitch,
                                 max_pitch=domain.pitch)
        except (RuntimeError, ValueError) as exc:
            # one outcome verification cannot solve (or a load pad the
            # design let go of: the robot arm's wrist link at 5% after 43 g at 6%) (a piece the supports do
            # not hold) is a failed outcome, not a failed study: the search
            # goes on past it (the robot arm's wrist link died here)
            mat = study.material
            report = {"mass": float(mesh.volume) * mat.density / 1000.0, "safetyFactor": 0.0,
                      "cases": [], "warnings": [f"verification failed: {exc}"]}
        if buck_state["sf"] is not None and report.get("bucklingFactor") is not None:
            # Buckling in the search (simp.optimize's buckling constraint):
            # on from the first part whose buckling factor comes within
            # _BUCKLING_SEARCH_ON of the safety factor, at sf * kb, kb the
            # optimizer's own reading over the verified one (last run)
            lv = report["bucklingFactor"]
            lo = res.history[-1].get("buckling") if replay is None and res.history else None
            if lo is not None and np.isfinite(lo) and lo > 0 and lv > 0:
                kb = buck_state["kb"]
                buck_state["kb"] = float(np.clip(lo / lv, 0.5 * kb, 2.0 * kb))
                opt["bucklingFactor"] = float(lo)
            if lv < _BUCKLING_SEARCH_ON * buck_state["sf"]:
                buck_state["on"] = True
        making = manufacturing_report(study, domain, mesh)
        if making:
            report["manufacturing"] = making
            report.setdefault("warnings", []).extend(making.get("warnings", []))
        report.setdefault("warnings", []).extend(fragment_warnings(rinfo, mesh))
        fid = mesh_fidelity(domain, xphys, mesh)
        if fid is not None:
            report["meshFidelity"] = fid
            if fid["lostSolidCells"] > MESH_LOSS_WARN:
                report.setdefault("warnings", []).append(
                    f"the mesh lost {fid['lostSolidCells']:.0%} of the optimized solid cells "
                    f"({fid['lostSurfaceCells']:.0%} of the surface cells, {fid['lostThinCells']:.0%} of the thin "
                    "members): the meshed part is softer and weaker than the design the optimizer found, so "
                    "the outcome search goes heavier than the design needs. The verified numbers are the "
                    "ones that hold.")
        if limits0 is not None and not warp:
            warp = {"limits": {"maxDistortion": limits0[0], "maxRecoaterLift": limits0[1]}}
        if warp:
            report["warp"] = warp
            if limits0 is not None:
                _check_warp(warp, mesh)
        o = Outcome(vf, mesh, report, opt, rinfo)
        o.xphys = xphys
        o.x = x
        o.voxel_sf = report.get("safetyFactor")
        if memory is not None:
            memory.add(vf, x, xphys, report)
        m = _margin(study, report)
        if m is not None:
            report["margin"] = m
            o.passes = bool(m >= 1.0)
        if report.get("bodies", 1) > 1:
            # A preserved piece no load path reaches (a mount with no load
            # through it) comes out as a loose part. It is not a design that
            # meets the brief, and the fix is in the study: say so.
            o.passes = False
            report.setdefault("warnings", []).append(
                f"the part came out in {report['bodies']} pieces: either each piece has its own supports and "
                "load path and nothing ties them together (two mirror halves, each on its own bolts), or some "
                "preserved geometry is on no load path. For one part, preserve a link between the pieces (a "
                "ring, a web, a plate) or add a load case that runs between them; for a floating preserve, "
                "load it or drop it.")
        if report.get("closedBores"):
            o.passes = False
            report.setdefault("warnings", []).append(
                f"the part closes over {len(report['closedBores'])} bore end(s) "
                f"({', '.join(b['id'] + ' ' + b['end'] for b in report['closedBores'])}): no fastener goes through, "
                "so it does not pass. Extend the keep-out over the head and the driver's reach if this repeats.")
        return o

    def run_one(vf: float) -> Outcome:
        nonlocal started
        emit({"stage": "optimize", "volumeFraction": vf})
        big = domain.n_elem >= _COARSE_MIN_CELLS

        def cold():
            if big:
                x0 = _coarse_start(study, domain, vf, radius, should_stop)
                return attempt(vf, x0, _WARM_BETA if x0 is not None else 1.0)
            return attempt(vf, None, 1.0)

        rp = memory.replay(vf) if memory is not None else None
        w = memory.warm_start(vf) if memory is not None and rp is None and not started else None
        if rp is not None:
            o = attempt(vf, None, None, replay=rp)
            o.report["start"] = f"recalled {vf:.3f}"
        elif w is not None:
            try:
                o = attempt(vf, w[0], _WARM_BETA)
                o.report["start"] = w[1]
            except RuntimeError:
                o = cold()
                o.report["start"] = "cold (the recalled start did not solve)"
        elif not any(q.x is not None for q in outcomes):
            o = cold()
            o.report["start"] = "cold"
        else:
            # Each fraction warm-starts from a design already found, which is
            # what lets a stiffness-bound part follow its layout down (the robot arm's
            # upper arm: 7.0% passed warm at margin 1.06 and failed cold at
            # 0.68). From the nearest fraction above, never the last one run:
            # the wall bracket chained 0.150 -> 0.104 -> 0.127 -> 0.115 (up,
            # then down again) read SF 1.69 at 0.115 and 2.34 at 0.110 carried
            # straight down from 0.150, so its mass swung with the order the
            # fractions came in (57-68 g). Carried down, the verified SF rises
            # smoothly with the fraction (0.88, 2.00, 2.34, 2.83, 3.11, 3.36
            # at 9-14%). Not always from the first design either: the turret
            # carried 0.120 -> 0.029 in one jump read SF 1.56, and 2.54 at
            # 0.025 stepped down through 0.068 and 0.042.
            above = [q for q in outcomes if q.x is not None and q.volume_fraction > vf]
            src = (min(above, key=lambda q: q.volume_fraction) if above else
                   max((q for q in outcomes if q.x is not None), key=lambda q: q.volume_fraction))
            ran_cold = False
            try:
                o = attempt(vf, src.x, _WARM_BETA)
                o.report["start"] = f"warm from {src.volume_fraction:.3f}"
            except RuntimeError:
                ran_cold = True
                # A carried layout can leave a support hanging on void
                # (the robot arm's wrist link: a pin-held idler with its material
                # gone, so the stiffness solve never converged). Start this
                # fraction afresh instead.
                o = cold()
                o.report["start"] = "cold (the warm start did not solve)"
            else:
                why = _malformed(study, o.report)
                if why and not (should_stop is not None and should_stop()):
                    # A carried layout can also come out as a thin sheet
                    # with spikes: the full-tier wrist yoke's pick at 5.7%
                    # warm from 6% read p5 member 1.46 mm for minMember 3
                    # and 303 mm of crease, while the same fraction started
                    # cold meshed clean (17.2 g, no crease). Run it cold too
                    # and keep the clean one.
                    c = cold()
                    ran_cold = True
                    c.report["start"] = f"cold (the warm start came out {why})"
                    if not _malformed(study, c.report) or \
                            (c.report.get("margin") or 0.0) > (o.report.get("margin") or 0.0):
                        o, c = c, o
                    o.report["alternative"] = {k: c.report.get(k) for k in
                                               ("start", "mass", "safetyFactor", "margin")} | {"passes": c.passes}
            if not ran_cold and big and len(outcomes) == 1 and vf < outcomes[0].volume_fraction \
                    and "recalled" not in str(outcomes[0].report.get("start")) \
                    and not (should_stop is not None and should_stop()):
                # ...but carried from a fraction well above, the layout can
                # be one the new fraction no longer suits: the wall bracket
                # read SF 1.0-1.5 at 8-14% chained down from 15%, and 3.2 at
                # 10% started fresh. So the first step down is also run from
                # its own coarse layout, and the better margin kept; one
                # extra run per search.
                c = cold()
                c.report["start"] = "cold"
                if (c.report.get("margin") or 0.0) > (o.report.get("margin") or 0.0):
                    o, c = c, o
                o.report["alternative"] = {k: c.report.get(k) for k in
                                           ("start", "mass", "safetyFactor", "margin")} | {"passes": c.passes}
        started = True
        emit({"stage": "outcome", **o.summary()})
        _save_partial(o, domain.pitch)
        prefetch_tets(o)
        return o

    tets_planned = []

    def prefetch_tets(o: Outcome) -> None:
        """Start `o`'s tet cross-check in the background (_TetPrefetch) when
        the search may confirm it: it passes as read now, in one piece."""
        if not tets_planned:
            tets_planned.append(bool(prefetch.on and study.goal.safety_factor is not None
                                     and study.goal.method != "constrained" and _confirm_on_tets(study, [])))
        if not tets_planned[0] or not o.passes or not o.voxel_sf or o.report.get("bodies", 1) > 1 \
                or getattr(o, "tet_k", None) is not None:
            return
        from .crosscheck import element_size

        # a lighter passing part is the likelier pick: heavier ones' checks
        # stop (they are checked inline if they are picked after all)
        for q in outcomes:
            if q.volume_fraction > o.volume_fraction:
                prefetch.drop(q)
        prefetch.submit(study, o, max(2.0 * element_size(study), 1.5 * domain.pitch))

    def _confirm(o: Outcome) -> float:
        # a replayed design's mesh is the one confirmed before: rescale that reading
        rep = memory.confirm_for(o.volume_fraction) if memory is not None and o.optimizer.get("recalled") else None
        recalled = rep is not None
        if rep is None:
            try:
                cc = _confirm_cells(study)
                # finer than the grid it passed on, whatever the budget says
                # (LOADPATH_CONFIRM_CELLS 50k confirmed the fan plate at 2.03 mm
                # cells after a 1.91 mm analysis): at least 1.5x its cells
                ac = o.report.get("analysisCells") or 0
                raised = cc < 1.5 * ac
                cc = max(cc, int(1.5 * ac))
                ap = o.report.get("analysisPitch")
                rep = verify_part(study, o.mesh, max_cells=cc, min_pitch=0.3 * domain.pitch,
                                  contact_band=domain.pitch,
                                  max_pitch=min(domain.pitch, 0.9 * ap) if ap else domain.pitch)
            except RuntimeError as exc:
                o.report.setdefault("warnings", []).append(f"confirmation failed: {exc}")
                o.report["margin"] = 0.0
                return 0.0
            if memory is not None:
                memory.confirmed(o.volume_fraction, rep)
        mc = _margin(study, rep)
        o.report["confirm"] = {"safetyFactor": rep["safetyFactor"], "margin": mc,
                               "analysisPitch": rep["analysisPitch"], "analysisCells": rep["analysisCells"],
                               **({"cellsCapped": "by free memory (LOADPATH_CONFIRM_CELLS sets it)"}
                                  if not recalled and cc < _CONFIRM_CELLS and not os.environ.get("LOADPATH_CONFIRM_CELLS")
                                  else {}),
                               **({"cellsRaised": "to 1.5x the analysis grid's: the confirm is never coarser"}
                                  if not recalled and raised else {}),
                               "cases": [{k: c[k] for k in ("name", "peakStress", "peakAt", "deflection")}
                                         for c in rep["cases"]]}
        # recorded, not adopted: the finer grid also resolves small features
        # the analysis grid smooths over, so its percentile is not comparable
        o.report["confirm"].update(p5Member=rep["p5Member"], minMember=rep["minMember"])
        if recalled:
            o.report["confirm"]["recalled"] = True
        if mc < o.report["margin"]:
            # report the finer (lower) reading: it is the one to trust
            o.report["margin"], o.report["safetyFactor"], o.report["cases"] = mc, rep["safetyFactor"], rep["cases"]
            for k in ("thermal", "durability"):
                if k in rep:
                    o.report[k] = rep[k]
        return mc

    def tet_confirm(o: Outcome) -> float | None:
        """Confirm `o` on the tetrahedral cross-check (crosscheck.py). Its
        safety factor replaces the voxel one, and o.tet_k (tet / voxel) is
        the calibration for the rest of the search. Returns the confirmed
        margin, or None when the cross-check could not run."""
        from .crosscheck import crosscheck, element_size

        emit({"stage": "confirm", "volumeFraction": o.volume_fraction, "method": "tet"})
        # Stress is left out within two elements of a load or support, and
        # no less than the voxel analysis leaves out: nearer, the peak is the
        # edge of an idealized rigid support, not the part. (The MBB beam's
        # read 23-26 MPa there on three meshes of one part, and 13.7-13.9
        # MPa at its real peak mid-span beyond the band.)
        band = max(2.0 * element_size(study), 1.5 * domain.pitch)
        try:
            cc = prefetch.take(o)
            if cc is None:
                cc = crosscheck(study, o.mesh, contact_band=band)
        except Exception as exc:  # noqa: BLE001 - a mesher or solver failure means: confirm on voxels instead
            o.report.setdefault("warnings", []).append(f"tetrahedral confirmation failed: {exc}")
            return None
        r = o.report
        o.tet_k = cc["safetyFactor"] / o.voxel_sf if o.voxel_sf else 1.0
        r["voxelSafetyFactor"] = o.voxel_sf
        r["safetyFactor"] = cc["safetyFactor"]
        r.pop("calibration", None)
        mt = _margin(study, r)
        r["margin"] = mt
        r["confirm"] = {"method": "tet", "safetyFactor": cc["safetyFactor"], "margin": mt,
                        "voxelSafetyFactor": o.voxel_sf, "tetOverVoxel": o.tet_k,
                        "elementSize": cc["elementSize"], "contactBand": band, "nodes": cc["nodes"], "elements": cc["elements"],
                        "seconds": cc["seconds"],
                        "cases": [{k: c[k] for k in ("name", "peakStress", "peakAt", "deflection")}
                                  for c in cc["cases"]]}
        o.passes = bool(mt >= 1.0) and _whole(r)
        emit({"stage": "confirmed", "volumeFraction": o.volume_fraction, "margin": mt, "method": "tet"})
        return mt

    def untet(o: Outcome):
        """Put `o` back on its voxel reading, keeping the tet one as a
        warning (where, how far below target)."""
        r = o.report
        c = r.pop("confirm")
        o.tet_k = None
        r.pop("voxelSafetyFactor", None)
        r["safetyFactor"] = o.voxel_sf
        r["margin"] = _margin(study, r)
        o.passes = bool(r["margin"] >= 1.0) and _whole(r)
        r["tetCheck"] = c
        worst = min(c["cases"], key=lambda q: -q["peakStress"])
        r.setdefault("warnings", []).append(
            f"the tetrahedral cross-check reads SF {c['safetyFactor']:.2f} (voxel {o.voxel_sf:.2f}), its peak "
            f"{worst['peakStress']:.1f} MPa in {worst['name']!r} at "
            f"({', '.join(f'{v:.1f}' for v in worst['peakAt'])}): a stress the voxel grid does not resolve. "
            "Check it there (a fillet, a thicker wall, or how the part is really held).")

    def layout_search(k: float):
        """Other layouts below a tet-confirmed pick with margin to spare.

        The fraction chain follows one layout down, warm-starting each step
        from the last, and a part whose strength is not monotone in the
        fraction stops on it: the wall bracket passed at 12.6% (tets 4.95
        for 2.5) with 10.2% and 11.4% failing (2.29, 2.12), all three
        descendants of the 15% design. So a few differently started designs
        are tried between the lightest confirmed design and the fraction
        its margin points at: warm from it, then a fresh coarse layout.
        Each that passes on its scaled voxel reading, or misses by under
        15%, is confirmed on tets; one that confirms becomes the design the
        next try starts below. (The bell crank went 24.3 -> 22.4 g this
        way, warm from its pick. A 1.5x wider filter, fresh or warm, came
        apart at these fractions, SF 0 on four parts, so it is not tried;
        fresh layouts confirmed nowhere either, but are the one start not
        descended from the chain.)"""
        confirmed = [i for i, q in enumerate(outcomes) if getattr(q, "tet_k", None) and q.passes]
        if not confirmed:
            return
        o = outcomes[min(confirmed, key=lambda i: outcomes[i].report["mass"])]
        mt = o.report["margin"]
        if mt < _TET_CLOSE:
            return
        floor = max(o.volume_fraction * 1.03 / mt, 0.5 * o.volume_fraction, _VF_MIN)
        big = domain.n_elem >= _COARSE_MIN_CELLS
        starts = [("warm from the pick", lambda vf, o: attempt(vf, o.x, _WARM_BETA)),
                  ("cold", lambda vf, o: attempt(vf, _coarse_start(study, domain, vf, radius, should_stop)
                                                 if big else None, _WARM_BETA if big else 1.0))]
        runs = 0
        for name, start in starts:
            while runs < _LAYOUT_RUNS and not (should_stop is not None and should_stop()):
                top = o.volume_fraction
                if o.report["margin"] < _TET_CLOSE:
                    return
                vf = 0.5 * (floor + top)
                retry = False
                if name == "cold":
                    # a fraction the chain failed warm, between the floor
                    # and the pick, first: its own fresh layout is the one
                    # the chain lost (the robot arm's MJF wrist link: 5.25% warm from
                    # 6% read tets 0.99, fresh 4.00 for 2, 37.2 g against
                    # 41.0; the midpoint, 4.2%, read 0.41)
                    fresh = [q.volume_fraction for q in outcomes
                             if str(q.report.get("start", "")).startswith(("cold", "layout search: cold"))]
                    lost = [q.volume_fraction for q in outcomes
                            if not q.passes and floor < q.volume_fraction < top and q.voxel_sf
                            and q.report.get("bodies", 1) <= 1 and str(q.report.get("start", "")).startswith("warm")
                            and not any(abs(q.volume_fraction - v) < 0.003 for v in fresh)]
                    if lost:
                        vf, retry = max(lost), True
                if top - vf < max(0.005, 0.05 * top) and not retry:
                    break
                if name.startswith("warm") and any(abs(q.volume_fraction - vf) < 0.003 for q in outcomes):
                    break   # the chain already ran warm there
                emit({"stage": "optimize", "volumeFraction": vf, "start": name})
                try:
                    q = start(vf, o)
                except RuntimeError:
                    break
                runs += 1
                q.report["start"] = f"layout search: {name}"
                outcomes.append(q)
                calibrate(q, k)
                emit({"stage": "outcome", **q.summary()})
                if q.report.get("bodies", 1) > 1 or (q.report.get("margin") or 0.0) < 0.85:
                    break
                mq = tet_confirm(q)
                if mq is None or mq < 1.0:
                    if mq is None:
                        q.tet_k, q.passes = 0.0, False
                    break
                o = q   # it held: try once more, below it
            if runs >= _LAYOUT_RUNS:
                break

    def calibrate(o: Outcome, k: float):
        """Read `o`'s voxel safety factor through the tet calibration k (1
        restores the voxel reading). A tet-confirmed outcome keeps its own."""
        if getattr(o, "tet_k", None) is not None or not o.voxel_sf:
            return
        r = o.report
        if k == 1.0:
            r.pop("calibration", None)
            m = _margin(study, r)
        else:
            r["calibration"] = {"tetOverVoxel": k, "safetyFactor": o.voxel_sf * k}
            m = _margin(study, {**r, "safetyFactor": o.voxel_sf * k})
        r["margin"] = m
        o.passes = bool(m >= 1.0) and _whole(r)
        prefetch_tets(o)

    def run_constrained(limits, x0) -> Outcome:
        from .constrained import optimize_constrained

        emit({"stage": "optimize", "limits": {"stress": limits.stress, "deflection": limits.deflection}})

        def on_iter(rec, xp):
            emit({"stage": "optimize", **rec}, xp)

        beta0 = _WARM_BETA if x0 is not None else 1.0
        if x0 is None and domain.n_elem >= _COARSE_MIN_CELLS:
            x0 = _coarse_start(study, domain, float(goal.volume_fraction or 0.3), radius, should_stop)
            beta0 = _WARM_BETA if x0 is not None else 1.0
        res = optimize_constrained(
            domain, study.elastic(),
            SimpOptions(volume_fraction=float(goal.volume_fraction or 0.3), filter_radius=radius,
                        max_iterations=study.max_iterations, x0=x0, beta_start=beta0,
                        extrusion=study.extrusion, pattern=study.pattern,
                        build_direction=study.print_direction),
            limits, on_iter=on_iter, should_stop=should_stop)
        mesh, rinfo = reconstruct(study, domain, res.xphys, ReconstructOptions())
        report = verify_part(study, mesh, min_pitch=0.5 * domain.pitch, contact_band=domain.pitch,
                                 max_pitch=domain.pitch)
        report.setdefault("warnings", []).extend(fragment_warnings(rinfo, mesh))
        opt = {"iterations": res.iterations, "converged": res.converged, "seconds": res.seconds,
               "achievedVolumeFraction": res.volume_fraction, "method": "constrained",
               "limits": {"stress": limits.stress, "deflection": limits.deflection,
                          **({"buckling": limits.buckling} if limits.buckling is not None else {})}}
        last = res.history[-1] if res.history else {}
        bks = [v for kk, v in last.items() if kk.startswith("buckling.")]
        if bks:
            opt["bucklingFactor"] = float(min(bks))   # the optimizer's own reading, eroded grey design
        o = Outcome(res.volume_fraction, mesh, report, opt, rinfo)
        o.xphys = res.xphys
        o.voxel_sf = report.get("safetyFactor")
        m = _margin(study, report)
        report["margin"] = m
        o.passes = bool(m >= 1.0)
        o.x = res.x
        emit({"stage": "outcome", **o.summary()})
        return o

    goal = study.goal
    if goal.method == "constrained" and _constrained_applies(study):
        from .constrained import Limits
        from .verify import STRESS_ALLOWANCE

        mat = study.material
        stress = mat.strength / (goal.safety_factor * STRESS_ALLOWANCE) if goal.safety_factor else None
        # The verified mesh reads differently from the optimizer's model
        # (finer grid, refined hotspots, members a cell thick that mesh
        # thinner), so the limits are scaled by one factor t until the
        # verified part lands on the target. Rescaling by each run's ratio
        # alone swung: the wall bracket's first run reads 17 MPa in the
        # optimizer and 377 MPa verified, the next limit (0.8 MPa) built
        # 100 g at SF 13, and four runs ended anywhere from 43 g to 174 g
        # with the target nudged by 2%. Once one run passes and one fails,
        # bisect t between them (geometrically), and warm-start only from a
        # passing design, never from a collapsed one.
        # With the tet cross-check (goal.confirm), each run is read on tets
        # and t is steered by that reading: the voxel one left the wall
        # brackets at SF 2.7-2.8 on voxels and 3.6 on tets for 2.5.
        tets = goal.safety_factor is not None and _confirm_on_tets(study, outcomes)
        # Buckling (goal.buckling): the optimizer holds the design's smallest
        # buckling load factor at sf * kb / t (bucklingopt.py), so t relaxes
        # or tightens stress and stability together and the bisection below
        # stays one-dimensional. kb is the optimizer's reading over the
        # verified part's (the grey eroded design on the fine grid against
        # the meshed part on the check's coarser grid), taken from the last
        # run: the two differ by a factor that holds roughly steady from one
        # design to the next.
        sf_b = goal.safety_factor if (goal.buckling and goal.safety_factor and _BUCKLING_IN_OPTIMIZER) else None
        kb = _BUCKLING_START
        fails = []
        t, t_pass, t_fail, x_pass = 1.0, None, None, None
        for _ in range(_MAX_CONSTRAINED_RUNS):
            if should_stop is not None and should_stop():
                break
            limits = Limits(stress * t if stress is not None else None,
                            goal.max_deflection * t if goal.max_deflection is not None else None, mat.E,
                            buckling=sf_b * kb / t if sf_b is not None else None)
            try:
                o = run_constrained(limits, x_pass)
            except (RuntimeError, ValueError) as exc:
                # (ValueError: the design let go of a load's region, which
                # then touches no material in the verify)
                # a run whose solve broke down (the design let go of a
                # support: the robot arm's yoke, wrist link and upper arm) is a
                # failed run; one that fails first ends the constrained
                # attempt, and the search below takes over
                fails.append(f"t {t:.3g}: {exc}")
                if not any(q.optimizer.get("method") == "constrained" for q in outcomes):
                    break
                t_fail = t if t_fail is None else min(t_fail, t)
                t = (t_pass * t_fail) ** 0.5 if t_pass is not None else 0.5 * t
                continue
            outcomes.append(o)
            if tets and o.voxel_sf and o.report.get("bodies", 1) <= 1 and tet_confirm(o) is None:
                tets = False   # the mesher failed: voxels from here on
            r = o.report
            rs = r["safetyFactor"] / goal.safety_factor if goal.safety_factor else None
            rd = goal.max_deflection / max(c["deflection"] for c in r["cases"]) if goal.max_deflection else None
            rb = None
            if sf_b is not None and r.get("bucklingFactor") is not None:
                rb = r["bucklingFactor"] / sf_b
                lo = o.optimizer.get("bucklingFactor")
                if lo is not None and np.isfinite(lo) and lo > 0:
                    kb = float(np.clip(lo / r["bucklingFactor"], 0.5 * kb, 2.0 * kb))
            ratio = min(v for v in (rs, rd, rb) if v is not None)
            if ratio >= 1.0:
                if ratio < 1.12:
                    break
                if t_pass is None or t > t_pass:
                    t_pass, x_pass = t, o.x
            elif t_fail is None or t < t_fail:
                t_fail = t
            if t_pass is not None and t_fail is not None:
                if t_fail / t_pass < 1.15:
                    break
                t = (t_pass * t_fail) ** 0.5
            elif t_fail is not None:
                # a stress limit acts roughly in proportion; never more than
                # 4x a step, so one wild reading cannot overshoot far
                t *= min(max(ratio, 0.25), 0.9)
            else:
                t *= min(ratio ** 0.5, 4.0)
        passing = [i for i, o in enumerate(outcomes) if o.passes]
        best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
        if best is None and fallback:
            # Search instead, on the search's own grid: the constrained
            # method runs finer (minimum member 2.5 cells across), and the
            # search on that grid broke down on the robot-arm yoke too.
            import dataclasses

            why = ("the constrained method found no passing design"
                   + (f" ({'; '.join(fails)})" if fails else "") + ": searched over volume fractions instead")
            r = _run_single(dataclasses.replace(study, goal=dataclasses.replace(goal, method="search")),
                            on_progress, should_stop)
            r.warnings.insert(0, why)
            return r
    elif goal.safety_factor is not None or goal.max_deflection is not None or _durability_drives(study):
        vf = float(goal.volume_fraction or 0.3)
        if memory is not None:
            vf, why = memory.first_fraction(vf)
            if why:
                emit({"stage": "recall", "volumeFraction": vf, "reason": why})
        lo_fail, hi_pass = None, None   # (vf, margin) brackets
        for _ in range(_MAX_SEARCH_RUNS):
            if should_stop is not None and should_stop():
                break
            o = run_one(vf)
            outcomes.append(o)
            m = o.report["margin"]
            if m >= 1.0 and o.report.get("bodies", 1) > 1:
                # strong enough but in pieces (a fastener's rings or an
                # unloaded pad left floating) is a failed outcome: step up
                # about a third, more material reconnects it. It used to
                # count as passing and end the search with no passing part.
                m = 0.8
            if m >= 1.0:
                hi_pass = (vf, m) if hi_pass is None or vf < hi_pass[0] else hi_pass
            else:
                lo_fail = (vf, m) if lo_fail is None or vf > lo_fail[0] else lo_fail
            # bracket closed to within 10% of the passing fraction (an
            # absolute 0.02 was 30% at the low fractions large design spaces
            # end at: the robot-arm upper arm stopped at 0.075 with 0.058 failing,
            # 222 g where ~0.066 would have passed lighter)
            if hi_pass and lo_fail and hi_pass[0] - lo_fail[0] < max(0.005, 0.1 * hi_pass[0]):
                break
            if hi_pass and 1.0 <= hi_pass[1] < 1.12:
                break   # passing with under 12% to spare: close enough
            if hi_pass and lo_fail:
                # Bracketed: bisect. The margin is not smooth in the volume
                # fraction (a slightly different fraction can settle in a
                # different layout: the arm link read 0.49 at 30% and 1.30 at
                # 38%), so interpolating between brackets kept landing on the
                # wrong side of a jump; halving the bracket always closes it.
                (v1, _), (v2, _) = lo_fail, hi_pass
                nxt = 0.5 * (v1 + v2)
                if memory is not None:
                    # any fraction inside the bracket closes it; a stored one is free
                    nxt = memory.snap(nxt, v1, v2, skip=[q.volume_fraction for q in outcomes])
            else:
                # Aim a little past the target; never more than halve or
                # double in one step, where the linear model stops holding.
                nxt = vf * 1.05 / max(m, 1e-9)
                nxt = float(np.clip(nxt, vf * 0.5, vf * 2.0))
                if memory is not None:
                    nxt = memory.snap(nxt, skip=[q.volume_fraction for q in outcomes])
            nxt = float(np.clip(nxt, _VF_MIN, _VF_MAX))
            # A step under 0.005 stops an unbracketed search. A bracket is
            # left to the 10% rule above: its midpoint sits half the gap from
            # either end, and at low fractions that is under 0.005 while the
            # gap is still worth closing (the robot arm's MJF wrist link passed at 6%,
            # SF 4.36 for 2, failed at 5.2%, and stopped there at 43.5 g
            # without trying 5.6%).
            if abs(nxt - vf) < 0.005 and not (hi_pass and lo_fail):
                break
            vf = nxt
        passing = [i for i, o in enumerate(outcomes) if o.passes]
        best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
        voxel_confirm = True
        if goal.safety_factor is not None and _confirm_on_tets(study, outcomes):
            # The voxel analysis reads peak stress high against a body-fitted
            # mesh (24-49% on the motor mount and bell crank), so a pick
            # that meets the target on voxels carries that much extra
            # material. Confirm the pick on tetrahedra instead, scale every
            # voxel safety factor by what that showed (tet / voxel), and let
            # the search step down to the fraction that meets the target on
            # the tet mesh; each new pick is confirmed there in turn.
            voxel_confirm = False
            k, tet_fail, confirms, extra = 1.0, None, 0, 0
            while not (should_stop is not None and should_stop()):
                passing = [i for i, q in enumerate(outcomes) if q.passes
                           and (tet_fail is None or q.volume_fraction > tet_fail)]
                best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
                if best is None:
                    # nothing passes: step up from the largest fraction tried
                    top = max(outcomes, key=lambda q: q.volume_fraction)
                    if extra >= _TET_EXTRA_RUNS or top.volume_fraction >= _VF_MAX:
                        break
                    vf = float(np.clip(top.volume_fraction * 1.05 / max(top.report["margin"], 1e-9),
                                       top.volume_fraction + 0.01, _VF_MAX))
                    outcomes.append(run_one(vf))
                    calibrate(outcomes[-1], k)
                    extra += 1
                    continue
                o = outcomes[best]
                if getattr(o, "tet_k", None) is None:
                    if confirms >= _TET_CONFIRMS:
                        break
                    confirms += 1
                    mt = tet_confirm(o)
                    if mt is None:
                        voxel_confirm = True
                        break
                    if mt < 1.0 and confirms == 1:
                        # The tet mesh reads the voxel search's own pick
                        # below target: the voxel grid missed a peak, so the
                        # calibration has nothing to step down on. Such a
                        # peak has sat in geometry the search cannot change
                        # (the robot arm's turret: SF 1.20 on the wall of a bolt's
                        # driver access, 1.3 mm from the bore that holds it,
                        # and three steps up to 0.24 left it at 230 g), so
                        # it is not stepped up on that reading either: the
                        # pick is confirmed on voxels, as without the tet
                        # mesh, and the tet reading goes out as a warning.
                        untet(o)
                        voxel_confirm = True
                        break
                    k = o.tet_k
                    for q in outcomes:
                        calibrate(q, k)
                    if mt < 1.0:
                        tet_fail = max(tet_fail or 0.0, o.volume_fraction)
                    continue   # re-pick: the calibration may pass a lighter one
                # o is confirmed and the lightest that passes: step down
                # toward the fraction the confirmed margin points at, unless
                # it is already close, or a failing fraction sits within 10%
                mt = o.report["margin"]
                # The fraction just below that failed only on its scaled voxel
                # reading is confirmed on tets directly before the search
                # stops on it: the voxel reading can fall off a cliff the tet
                # mesh does not see (the wall bracket: voxel SF 2.65 at 7.8%
                # and 0.81 at 7.1%, tets 4.06 at 7.8%), and the ratio of one
                # design says little about the next.
                near = [i for i, q in enumerate(outcomes)
                        if getattr(q, "tet_k", None) is None and not q.passes and q.voxel_sf
                        and q.volume_fraction < o.volume_fraction and q.report.get("bodies", 1) <= 1
                        and (tet_fail is None or q.volume_fraction > tet_fail)]
                if mt >= _TET_CLOSE and near and confirms < _TET_CONFIRMS:
                    q = outcomes[max(near, key=lambda i: outcomes[i].volume_fraction)]
                    confirms += 1
                    mq = tet_confirm(q)
                    if mq is None or mq < 1.0:
                        if mq is None:
                            q.tet_k = 0.0   # do not try it again
                            q.passes = False
                        tet_fail = max(tet_fail or 0.0, q.volume_fraction)
                    continue
                lo = max((q.volume_fraction for q in outcomes
                          if not q.passes and q.volume_fraction < o.volume_fraction), default=None)
                if mt < _TET_CLOSE or extra >= _TET_EXTRA_RUNS or \
                        (lo is not None and o.volume_fraction - lo < max(0.005, 0.1 * o.volume_fraction)):
                    break
                nxt = max(o.volume_fraction * 1.03 / mt, 0.5 * o.volume_fraction)
                if lo is not None:
                    nxt = max(nxt, 0.5 * (lo + o.volume_fraction))
                nxt = float(np.clip(nxt, _VF_MIN, _VF_MAX))
                # already tried there; inside a bracket the bracket sets the
                # scale (the bell crank confirmed 7.03%, SF 4.85 for 3, with
                # 6.25% failing at 2.93, and its midpoint 6.64% sat under
                # 0.005 from 6.25%, so nothing in between was tried)
                tol = 0.005 if lo is None else min(0.005, 0.25 * (o.volume_fraction - lo))
                if any(abs(nxt - q.volume_fraction) < tol for q in outcomes):
                    break
                outcomes.append(run_one(nxt))
                calibrate(outcomes[-1], k)
                extra += 1
            if not voxel_confirm and not (should_stop is not None and should_stop()):
                layout_search(k)
            confirmed_ok = [i for i, q in enumerate(outcomes) if getattr(q, "tet_k", None) is not None and q.passes]
            if not voxel_confirm and not confirmed_ok and should_stop is not None and should_stop() \
                    and any(q.passes for q in outcomes):
                # out of time (--time-limit) before any tet confirm: hand
                # over the voxel-verified pass, marked as such, not nothing
                for q in outcomes:
                    q.report.setdefault("warnings", []).append(
                        "the run stopped at its time limit before the tet re-check: this outcome is "
                        "verified on voxels only (they read 10-40% high near sharp corners); "
                        "`loadpath verify` it, or give the run more time")
                voxel_confirm = True
            if not voxel_confirm:
                # deliver only what the tet mesh confirmed; the rest keep
                # their calibrated reading, for the record
                for i, q in enumerate(outcomes):
                    if i not in confirmed_ok:
                        q.passes = False
                best = min(confirmed_ok, key=lambda i: outcomes[i].report["mass"]) if confirmed_ok else None
            else:
                for q in outcomes:
                    if getattr(q, "tet_k", None) is None:
                        calibrate(q, 1.0)
                passing = [i for i, q in enumerate(outcomes) if q.passes]
                best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
        # Confirm the pick on a finer grid; if it fails there, it no longer
        # passes, and the search steps up from it by the confirmed margin.
        # Once a pick confirms, bisect back down into the gap to the fraction
        # that failed confirmation while that gap is over 10%: the step up
        # overshoots (the robot arm's wrist yoke failed at 0.080 and confirmed at
        # 0.102, 27.5 g, with SF 2 near 0.09).
        confirmed, conf_fail, conf_failed = set(), None, []
        for _ in range(2 * _CONFIRM_RUNS if voxel_confirm else 0):
            if best is None or (should_stop is not None and should_stop()):
                break
            o = outcomes[best]
            if best not in confirmed:
                emit({"stage": "confirm", "volumeFraction": o.volume_fraction})
                mc = _confirm(o)
                emit({"stage": "confirmed", "volumeFraction": o.volume_fraction, "margin": mc})
                if mc < 1.0:
                    o.passes = False
                    conf_fail = max(conf_fail or 0.0, o.volume_fraction)
                    conf_failed.append(o.volume_fraction)
                    up = [i for i in confirmed if outcomes[i].volume_fraction > o.volume_fraction]
                    if up:
                        best = min(up, key=lambda i: outcomes[i].report["mass"])
                        continue
                    # a heavier part that passed its analysis is confirmed
                    # before anything new runs: stepping up from each lighter
                    # pick in turn re-ran the same 0.7 design four times
                    heavier = [i for i, q in enumerate(outcomes) if q.passes and i not in confirmed
                               and q.volume_fraction > o.volume_fraction]
                    if heavier:
                        best = min(heavier, key=lambda i: outcomes[i].report["mass"])
                        continue
                    vf = float(np.clip(o.volume_fraction * 1.05 / max(mc, 1e-9), o.volume_fraction + 0.01, _VF_MAX))
                    if any(abs(vf - q.volume_fraction) < 0.005 for q in outcomes):
                        vf = min(vf + 0.02, _VF_MAX)
                    if memory is not None:
                        vf = memory.snap(vf, skip=[q.volume_fraction for q in outcomes])
                    if any(abs(vf - f) < 0.005 for f in conf_failed):
                        # nowhere heavier to go (the pick sat at _VF_MAX): the
                        # step up re-ran the same recalled design, failed its
                        # confirm each time and delivered the last, never
                        # confirmed copy as passing. Nothing passes.
                        best = None
                        break
                    outcomes.append(run_one(vf))
                    passing = [i for i, q in enumerate(outcomes) if q.passes]
                    best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
                    continue
                confirmed.add(best)
            if conf_fail is None or o.volume_fraction - conf_fail < max(0.005, 0.1 * o.volume_fraction):
                break
            q = run_one(0.5 * (conf_fail + o.volume_fraction))
            outcomes.append(q)
            if q.passes:
                best = len(outcomes) - 1
            else:
                conf_fail = q.volume_fraction
        if voxel_confirm and best is not None and best not in confirmed:
            # the confirm runs ran out on a pick that only passed its
            # analysis grid: confirm it now, or fall back to the lightest
            # confirmed part; never deliver it as passing unconfirmed
            o = outcomes[best]
            if not (should_stop is not None and should_stop()):
                emit({"stage": "confirm", "volumeFraction": o.volume_fraction})
                mc = _confirm(o)
                emit({"stage": "confirmed", "volumeFraction": o.volume_fraction, "margin": mc})
                if mc >= 1.0:
                    confirmed.add(best)
                else:
                    o.passes = False
            if best not in confirmed:
                if o.passes:
                    o.report.setdefault("warnings", []).append(
                        "the run stopped before this outcome's finer-grid confirm: it is verified on the "
                        "analysis grid only; `loadpath verify` it, or give the run more time")
                else:
                    ok = [i for i in confirmed if outcomes[i].passes]
                    best = min(ok, key=lambda i: outcomes[i].report["mass"]) if ok else None
        if voxel_confirm and best is None and conf_failed:
            top = max(outcomes, key=lambda q: q.volume_fraction)
            mcs = [q.report["confirm"]["margin"] for q in outcomes if (q.report.get("confirm") or {}).get("margin")
                   is not None]
            ceiling = " (the search's ceiling)" if top.volume_fraction >= _VF_MAX - 1e-9 else ""
            best_m = f" (best confirmed margin {max(mcs):.2f})" if mcs else ""
            run_notes.append(
                f"no volume fraction passes: up to {top.volume_fraction:.3f}{ceiling}, every part that passed "
                f"its analysis failed the finer-grid confirm{best_m}. Lower the target, use a stiffer or "
                "stronger material, or widen the design space")
        # The voxel-confirmed counterpart of layout_search: a fraction below
        # the pick that failed warm, carried down from the pick's layout, is
        # run once more from its own coarse layout. The warm chain can lose
        # the layout on the way down: the robot arm's MJF wrist link (no tet confirm,
        # a layered process) read SF 1.42 at 5.25% warm from 6% (tets 1.16)
        # and 2.06 started fresh (tets 4.49), 37.2 g against the 41.0 g pick.
        # Each fresh design that passes is confirmed on the finer grid like
        # any pick; one that holds becomes the pick, and the next failed
        # fraction below it is tried the same way.
        cold_runs = 0
        while voxel_confirm and best is not None and best in confirmed and cold_runs < _LAYOUT_RUNS \
                and not (should_stop is not None and should_stop()):
            o = outcomes[best]
            if o.report["margin"] < _TET_CLOSE:
                break
            fresh = [q.volume_fraction for q in outcomes if str(q.report.get("start", "")).startswith(
                ("cold", "layout search: cold"))]
            lost = [q.volume_fraction for q in outcomes
                    if not q.passes and q.volume_fraction < o.volume_fraction and q.voxel_sf
                    and q.report.get("bodies", 1) <= 1 and str(q.report.get("start", "")).startswith("warm")
                    and q.volume_fraction > o.volume_fraction * 1.03 / max(o.report["margin"], 1e-9)
                    and not any(abs(q.volume_fraction - v) < 0.003 for v in fresh)]
            if not lost:
                break
            vf = max(lost)
            emit({"stage": "optimize", "volumeFraction": vf, "start": "layout search: cold"})
            try:
                if domain.n_elem >= _COARSE_MIN_CELLS:
                    x0 = _coarse_start(study, domain, vf, radius, should_stop)
                    q = attempt(vf, x0, _WARM_BETA if x0 is not None else 1.0)
                else:
                    q = attempt(vf, None, 1.0)
            except RuntimeError:
                break
            cold_runs += 1
            q.report["start"] = "layout search: cold"
            outcomes.append(q)
            emit({"stage": "outcome", **q.summary()})
            if not q.passes:
                break   # fresh fails too: the fractions below it would not do better
            emit({"stage": "confirm", "volumeFraction": q.volume_fraction})
            mc = _confirm(q)
            emit({"stage": "confirmed", "volumeFraction": q.volume_fraction, "margin": mc})
            if mc < 1.0:
                q.passes = False
                break
            best = len(outcomes) - 1
            confirmed.add(best)
    else:
        fractions = [float(goal.volume_fraction)] + [v for v in goal.outcomes if v != goal.volume_fraction]
        if len(fractions) > 1 and _workers(len(fractions), cells=_study_cells(study)) > 1:
            import dataclasses
            singles = [dataclasses.replace(study, goal=dataclasses.replace(goal, volume_fraction=vf, outcomes=[]))
                       for vf in fractions]
            for r in _parallel(singles, None, should_stop):
                outcomes.extend(r.outcomes)
                for o in r.outcomes:
                    emit({"stage": "outcome", **o.summary()})
        else:
            for vf in fractions:
                if should_stop is not None and should_stop():
                    break
                outcomes.append(run_one(vf))
        best = 0 if outcomes else None

    if memory is not None:
        memory.save()
        info["recall"] = dict(memory.used)
    return StudyResult(study=study, domain=info, outcomes=outcomes, best=best,
                       seconds=time.time() - t0, warnings=list(domain.warnings) + run_notes)


def _quick_limits():
    """Quick mode's search limits (quick.study sets them in the process that
    reads the study): a spawned worker (pitch starts, materials) imports this
    module afresh, so it takes them from LOADPATH_QUICK here, or each worker
    ran the full search and tet confirms."""
    global _MAX_SEARCH_RUNS, _CONFIRM_RUNS
    from . import quick
    if quick.on():
        # LOADPATH_QUICK_SEARCH_RUNS: a screen that wants the coarse search to
        # close in on the answer (explore.py) rather than stop at a loose bracket
        runs = int(os.environ.get("LOADPATH_QUICK_SEARCH_RUNS") or quick.STUDY["searchRuns"])
        _MAX_SEARCH_RUNS = min(_MAX_SEARCH_RUNS, runs)
        _CONFIRM_RUNS = min(_CONFIRM_RUNS, quick.STUDY["confirmRuns"])


_quick_limits()
