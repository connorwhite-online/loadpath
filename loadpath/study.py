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
  lightest outcome that passes. Each step warm-starts from the last.
  With `durability.constrain` the search also holds the fatigue and creep
  safety factor to `durability.safetyFactor` (and a held load's crept
  deflection to `maxDeflection`); see durability.py.

The search works on one number, the margin: the worst of (safety factor /
target) and (deflection limit / deflection). Both scale roughly linearly with
the amount of material in an optimized part, so the next volume fraction is
the current one divided by the margin.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .domain import Domain, build_domain
from .manufacture import report as manufacturing_report
from .reconstruct import ReconstructOptions, reconstruct
from .simp import SimpOptions, filter_radius, optimize
from .spec import Study
from .verify import verify_part

__all__ = ["Outcome", "StudyResult", "run_study", "design_at"]

# Bounds for the safety-factor search. (A floor of 0.04 stopped GRIP's upper
# arm there with SF 5.4 against a goal of 2: large design spaces around
# small parts end at a few percent.)
_VF_MIN, _VF_MAX = 0.02, 0.7
_MAX_SEARCH_RUNS = 7
# Coarse-to-fine: below this many cells the fine solve is cheap enough alone.
_COARSE_MIN_CELLS = 12_000
# Projection sharpness a warm-started run resumes at.
_WARM_BETA = 4.0
# The pick of a safety-factor search is checked again on a grid this many
# cells finer before it is delivered. The in-loop check resolves a fillet
# only as well as its grid: GRIP's turret read SF 2.26 there and 1.65 at
# 120k cells, its peak at a corner the coarser grid had averaged away.
_CONFIRM_CELLS = 150_000
_CONFIRM_RUNS = 3


def _making(study: Study) -> dict:
    """The optimizer options that say how the part is made."""
    return dict(build_direction=study.print_direction, extrusion=study.extrusion, access=study.access,
                pattern=study.pattern, max_member=study.max_member)


def _coarse_start(study: Study, domain: Domain, vf: float, radius: float, should_stop) -> np.ndarray:
    """Find the layout on a grid twice as coarse (an eighth of the cells, so
    each iteration is roughly ten times cheaper), stopping while it is still
    soft (beta 4), then interpolate it onto the fine grid as the starting
    design. The fine run only has to sharpen and refine it."""
    from scipy import ndimage

    coarse = build_domain(study, pitch=2.0 * domain.pitch)
    res = optimize(coarse, study.elastic(design=True),
                   SimpOptions(volume_fraction=vf, filter_radius=max(radius, 1.5 * coarse.pitch),
                               max_iterations=60, beta_max=_WARM_BETA, beta_every=12, **_making(study)),
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

    def summary(self) -> dict:
        r = self.report
        return {
            "material": r.get("material", {}).get("name"),
            "volumeFraction": self.volume_fraction,
            "mass": r.get("mass"),
            "massReduction": r.get("massReduction"),
            "safetyFactor": r.get("safetyFactor"),
            "cases": r.get("cases"),
            "minMember": r.get("minMember"),
            "overhangArea": r.get("overhangArea"),
            "keepOutIntrusion": r.get("keepOutIntrusion"),
            "preserveMissing": r.get("preserveMissing"),
            "bodies": r.get("bodies"),
            "boundaryCutArea": r.get("boundaryCutArea"),
            **({"durability": r["durability"]} if "durability" in r else {}),
            **({"print": {k: v for k, v in r["print"].items() if k != "bridgeList"}} if "print" in r else {}),
            **({"process": r["process"]} if "process" in r else {}),
            **({"slicer": r["slicer"]} if "slicer" in r else {}),
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

    def to_dict(self) -> dict:
        return {
            "name": self.study.name,
            "domain": self.domain,
            "outcomes": [o.summary() for o in self.outcomes],
            "best": self.best,
            "seconds": self.seconds,
            "warnings": self.warnings,
            **({"orientation": self.study.raw["orientation"]} if "orientation" in self.study.raw else {}),
        }


def _durability_drives(study: Study) -> bool:
    return study.durability is not None and study.durability.constrain


def _margin(study: Study, report: dict) -> float | None:
    g = study.goal
    ms = []
    if g.safety_factor is not None:
        ms.append(report["safetyFactor"] / g.safety_factor)
    dur = report.get("durability") if _durability_drives(study) else None
    if g.max_deflection is not None:
        worst = max((c["deflection"] for c in report["cases"]), default=float("inf"))
        if dur:
            # a held load's deflection after creep, at its temperature
            worst = max([worst] + [c["creepDeflection"] for c in dur["cases"] if "creepDeflection" in c])
        ms.append(g.max_deflection / worst if worst > 0 else float("inf"))
    if dur and dur.get("safetyFactor") is not None:
        ms.append(dur["safetyFactor"] / dur["target"])
    return min(ms) if ms else None


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
           "beta": res.history[-1]["beta"] if res.history else beta0}
    o = Outcome(vf, mesh, {}, opt, rinfo)
    o.xphys = res.xphys
    o.x = res.x
    if not verify:
        return o
    note({"stage": "verify", "volumeFraction": vf})
    try:
        report = verify_part(study, mesh, max_cells=verify_cells, min_pitch=0.5 * domain.pitch,
                             contact_band=domain.pitch, refine=refine)
    except (RuntimeError, ValueError) as exc:
        # one outcome verification cannot solve (a piece the supports do
        # not hold) is a failed outcome, not a failed study: the search
        # goes on past it (GRIP's wrist link died here)
        mat = study.material
        report = {"mass": float(mesh.volume) * mat.density / 1000.0, "safetyFactor": 0.0,
                  "cases": [], "warnings": [f"verification failed: {exc}"]}
    making = manufacturing_report(study, domain, mesh)
    if making:
        report["manufacturing"] = making
    o.report = report
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
            f"the part came out in {report['bodies']} pieces: some preserved geometry is not on any load "
            "path. Add a load case that runs through it (what does it carry?) or drop it from preserve.")
    return o


def _domain_info(d: Domain, radius: float) -> dict:
    return {
        "pitch": d.pitch,
        "cells": d.n_elem,
        "dims": list(d.dims),
        "preservedCells": int(d.preserve.sum()),
        "filterRadius": radius,
        "loadCases": d.case_names,
    }


# Peak memory of one worker, roughly, by the most cells it solves at once
# (study._study_cells): GRIP's turret, a safety-factor search confirmed at
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


def _available_bytes() -> float | None:
    avail = _meminfo_available()
    cap = _cgroup_limit()
    if avail is None:
        return cap
    return min(avail, cap) if cap else avail


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
    `cells` cells (the 4-core, 15 GB cloud box ran out at two GRIP cases at
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


def _solve_alone(study: Study) -> StudyResult:
    return _run_single(study)


def _study_cells(st: Study) -> int:
    """The most cells one run of `st` solves at once."""
    g = st.goal
    search = g.safety_factor is not None or g.max_deflection is not None
    return max(st.max_elements, _CONFIRM_CELLS if search else 60_000)


def _parallel(studies: list, on_progress, should_stop) -> list:
    """Run independent single-goal studies, in processes when it helps.
    Progress inside a worker is not streamed; each finished outcome is."""
    workers = _workers(len(studies), cells=max(_study_cells(s) for s in studies)) if studies else 1
    if workers <= 1:
        return [_run_single(s, on_progress, should_stop) for s in studies]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    results = [None] * len(studies)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_solve_alone, s): i for i, s in enumerate(studies)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            i = futs[f]
            results[i] = f.result()
            if on_progress is not None:
                for o in results[i].outcomes:
                    on_progress({"stage": "outcome", **o.summary()}, None)
    return results


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
        return _slice(_process_checks(_run_single(study, on_progress, should_stop), on_progress), on_progress)
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
    return _slice(_process_checks(StudyResult(study=study, domain=domain, outcomes=outcomes, best=best,
                                              seconds=time.time() - t0, warnings=warnings), on_progress),
                  on_progress)


def _process_checks(result: StudyResult, on_progress=None) -> StudyResult:
    """Powder-bed polymer and resin parts: the process checks
    (processcheck.check_process) on the delivered outcome, into its report
    as "process", with its warnings added to the outcome's."""
    proc = result.study.process
    if proc is None or result.best is None:
        return result
    from .processcheck import RULES, check_process

    if proc.kind not in RULES:
        return result
    o = result.outcomes[result.best]
    if getattr(o, "mesh", None) is None:
        return result
    if on_progress is not None:
        on_progress({"stage": "process", "process": proc.kind}, None)
    try:
        mat = _material_named((o.report.get("material") or {}).get("name"), result.study)
        rep = check_process(o.mesh, proc, material=mat)
    except Exception as exc:  # noqa: BLE001 - a failed check leaves the part's report standing
        rep = {"process": proc.kind, "error": f"{type(exc).__name__}: {exc}", "ok": None, "warnings": []}
    o.report["process"] = rep
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


def _material_named(name, study):
    """The study's Material called `name` (a study can try several)."""
    for m in study.materials or [study.material]:
        if m.name == name:
            return m
    return study.material


def _run_single(study: Study, on_progress=None, should_stop=None) -> StudyResult:
    t0 = time.time()

    def emit(event, field_=None):
        if on_progress is not None:
            on_progress(event, field_)

    domain = build_domain(study)
    radius = filter_radius(domain, study.min_member)
    info = _domain_info(domain, radius)
    emit({"stage": "domain", **info, "warnings": domain.warnings})
    outcomes: list[Outcome] = []
    last_x = None
    # what earlier runs on this domain found (recall.py), when turned on
    from .recall import Recall
    memory = Recall.open(study, domain, radius, _making(study))

    def attempt(vf: float, x0, beta0, replay=None) -> Outcome:
        def on_iter(rec, xp):
            emit({"stage": "optimize", "volumeFraction": vf, **rec}, xp)

        if replay is None:
            res = optimize(domain, study.elastic(design=True),
                           SimpOptions(volume_fraction=vf, filter_radius=radius,
                                       max_iterations=study.max_iterations, x0=x0, beta_start=beta0,
                                       **_making(study)),
                           on_iter=on_iter, should_stop=should_stop)
            x, xphys = res.x, res.xphys
            opt = {"iterations": res.iterations, "converged": res.converged, "seconds": res.seconds,
                   "compliance": res.compliance, "caseCompliance": res.case_compliance,
                   **({"caseDirection": res.case_direction} if any(d is not None for d in res.case_direction) else {}),
                   "achievedVolumeFraction": res.volume_fraction}
        else:
            # a stored design for this very problem: mesh and verify it, no optimizing
            x, xphys = replay[0], replay[1]
            opt = {"iterations": 0, "converged": True, "seconds": 0.0, "recalled": True,
                   "achievedVolumeFraction": float(xphys.mean())}
        emit({"stage": "mesh", "volumeFraction": vf})
        mesh, rinfo = reconstruct(study, domain, xphys, ReconstructOptions())
        emit({"stage": "verify", "volumeFraction": vf})
        try:
            report = verify_part(study, mesh, min_pitch=0.5 * domain.pitch, contact_band=domain.pitch)
        except (RuntimeError, ValueError) as exc:
            # one outcome verification cannot solve (or a load pad the
            # design let go of: GRIP's wrist link at 5% after 43 g at 6%) (a piece the supports do
            # not hold) is a failed outcome, not a failed study: the search
            # goes on past it (GRIP's wrist link died here)
            mat = study.material
            report = {"mass": float(mesh.volume) * mat.density / 1000.0, "safetyFactor": 0.0,
                      "cases": [], "warnings": [f"verification failed: {exc}"]}
        making = manufacturing_report(study, domain, mesh)
        if making:
            report["manufacturing"] = making
        o = Outcome(vf, mesh, report, opt, rinfo)
        o.xphys = xphys
        o.x = x
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
                f"the part came out in {report['bodies']} pieces: some preserved geometry is not on any load "
                "path. Add a load case that runs through it (what does it carry?) or drop it from preserve.")
        return o

    def run_one(vf: float) -> Outcome:
        nonlocal last_x
        emit({"stage": "optimize", "volumeFraction": vf})
        big = domain.n_elem >= _COARSE_MIN_CELLS

        def cold():
            if big:
                return attempt(vf, _coarse_start(study, domain, vf, radius, should_stop), _WARM_BETA)
            return attempt(vf, None, 1.0)

        rp = memory.replay(vf) if memory is not None else None
        w = memory.warm_start(vf) if memory is not None and rp is None and last_x is None else None
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
        elif last_x is None:
            o = cold()
            o.report["start"] = "cold"
        else:
            # Each fraction warm-starts from the last design, which is what
            # lets a stiffness-bound part follow its layout down (GRIP's upper
            # arm: 7.0% passed warm at margin 1.06 and failed cold at 0.68).
            try:
                o = attempt(vf, last_x, _WARM_BETA)
                o.report["start"] = f"warm from {outcomes[-1].volume_fraction:.3f}" if outcomes else "warm"
            except RuntimeError:
                # A carried layout can leave a support hanging on void
                # (GRIP's wrist link: a pin-held idler with its material
                # gone, so the stiffness solve never converged). Start this
                # fraction afresh instead.
                o = cold()
                o.report["start"] = "cold (the warm start did not solve)"
            if big and len(outcomes) == 1 and vf < outcomes[0].volume_fraction \
                    and "recalled" not in str(outcomes[0].report.get("start")) \
                    and not (should_stop is not None and should_stop()):
                # ...but carried from a fraction well above, the layout can
                # be one the new fraction no longer suits: the wall bracket
                # read SF 1.0-1.5 at 8-14% chained down from 15%, and 3.2 at
                # 10% started fresh. So the first step down is also run from
                # its own coarse layout, and the better margin kept (and
                # chained on); one extra run per search.
                c = cold()
                c.report["start"] = "cold"
                if (c.report.get("margin") or 0.0) > (o.report.get("margin") or 0.0):
                    o, c = c, o
                o.report["alternative"] = {k: c.report.get(k) for k in
                                           ("start", "mass", "safetyFactor", "margin")} | {"passes": c.passes}
        last_x = o.x
        emit({"stage": "outcome", **o.summary()})
        return o

    def _confirm(o: Outcome) -> float:
        # a replayed design's mesh is the one confirmed before: rescale that reading
        rep = memory.confirm_for(o.volume_fraction) if memory is not None and o.optimizer.get("recalled") else None
        recalled = rep is not None
        if rep is None:
            try:
                rep = verify_part(study, o.mesh, max_cells=_CONFIRM_CELLS, min_pitch=0.3 * domain.pitch,
                                  contact_band=domain.pitch)
            except RuntimeError as exc:
                o.report.setdefault("warnings", []).append(f"confirmation failed: {exc}")
                o.report["margin"] = 0.0
                return 0.0
            if memory is not None:
                memory.confirmed(o.volume_fraction, rep)
        mc = _margin(study, rep)
        o.report["confirm"] = {"safetyFactor": rep["safetyFactor"], "margin": mc,
                               "analysisPitch": rep["analysisPitch"], "analysisCells": rep["analysisCells"],
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
            if "durability" in rep:
                o.report["durability"] = rep["durability"]
        return mc

    def run_constrained(limits, x0) -> Outcome:
        from .constrained import optimize_constrained

        emit({"stage": "optimize", "limits": {"stress": limits.stress, "deflection": limits.deflection}})

        def on_iter(rec, xp):
            emit({"stage": "optimize", **rec}, xp)

        beta0 = _WARM_BETA if x0 is not None else 1.0
        if x0 is None and domain.n_elem >= _COARSE_MIN_CELLS:
            x0 = _coarse_start(study, domain, float(goal.volume_fraction or 0.3), radius, should_stop)
            beta0 = _WARM_BETA
        res = optimize_constrained(
            domain, study.elastic(),
            SimpOptions(volume_fraction=float(goal.volume_fraction or 0.3), filter_radius=radius,
                        max_iterations=study.max_iterations, x0=x0, beta_start=beta0,
                        extrusion=study.extrusion, pattern=study.pattern),
            limits, on_iter=on_iter, should_stop=should_stop)
        mesh, rinfo = reconstruct(study, domain, res.xphys, ReconstructOptions())
        report = verify_part(study, mesh, min_pitch=0.5 * domain.pitch, contact_band=domain.pitch)
        opt = {"iterations": res.iterations, "converged": res.converged, "seconds": res.seconds,
               "achievedVolumeFraction": res.volume_fraction, "method": "constrained",
               "limits": {"stress": limits.stress, "deflection": limits.deflection}}
        o = Outcome(res.volume_fraction, mesh, report, opt, rinfo)
        m = _margin(study, report)
        report["margin"] = m
        o.passes = bool(m >= 1.0)
        o.x = res.x
        emit({"stage": "outcome", **o.summary()})
        return o

    goal = study.goal
    if goal.method == "constrained" and (goal.safety_factor is not None or goal.max_deflection is not None) \
            and study.print_direction is None and study.gravity is None and not _durability_drives(study) \
            and study.access is None and study.max_member is None \
            and not any(ld.direction for c in study.load_cases for ld in c.loads):
        from .constrained import Limits
        from .verify import STRESS_ALLOWANCE

        mat = study.material
        stress = mat.strength / (goal.safety_factor * STRESS_ALLOWANCE) if goal.safety_factor else None
        limits = Limits(stress, goal.max_deflection, mat.E)
        x0 = None
        for _ in range(4):
            if should_stop is not None and should_stop():
                break
            o = run_constrained(limits, x0)
            outcomes.append(o)
            x0 = o.x
            r = o.report
            rs = r["safetyFactor"] / goal.safety_factor if goal.safety_factor else None
            rd = goal.max_deflection / max(c["deflection"] for c in r["cases"]) if goal.max_deflection else None
            ratios = [v for v in (rs, rd) if v is not None]
            if min(ratios) >= 1.0 and min(ratios) < 1.12:
                break
            # The verified mesh reads differently from the optimizer's model
            # (finer grid, refined hotspots, the delivered rather than the
            # eroded design). Rescale each limit by how far off it was:
            # fully when it failed, halfway (geometric) when it was slack.
            def scale(lim, ratio):
                if lim is None or ratio is None:
                    return lim
                return lim * (ratio if ratio < 1.0 else ratio ** 0.5)
            limits = Limits(scale(limits.stress, rs), scale(limits.deflection, rd), mat.E)
        passing = [i for i, o in enumerate(outcomes) if o.passes]
        best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
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
            if m >= 1.0:
                hi_pass = (vf, m) if hi_pass is None or vf < hi_pass[0] else hi_pass
            else:
                lo_fail = (vf, m) if lo_fail is None or vf > lo_fail[0] else lo_fail
            # bracket closed to within 10% of the passing fraction (an
            # absolute 0.02 was 30% at the low fractions large design spaces
            # end at: the GRIP upper arm stopped at 0.075 with 0.058 failing,
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
                (v1, m1), (v2, m2) = lo_fail, hi_pass
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
            if abs(nxt - vf) < 0.005:
                break
            vf = nxt
        passing = [i for i, o in enumerate(outcomes) if o.passes]
        best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
        # Confirm the pick on a finer grid; if it fails there, it no longer
        # passes, and the search steps up from it by the confirmed margin.
        # Once a pick confirms, bisect back down into the gap to the fraction
        # that failed confirmation while that gap is over 10%: the step up
        # overshoots (GRIP's wrist yoke failed at 0.080 and confirmed at
        # 0.102, 27.5 g, with SF 2 near 0.09).
        confirmed, conf_fail = set(), None
        for _ in range(2 * _CONFIRM_RUNS):
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
                    up = [i for i in confirmed if outcomes[i].volume_fraction > o.volume_fraction]
                    if up:
                        best = min(up, key=lambda i: outcomes[i].report["mass"])
                        continue
                    vf = float(np.clip(o.volume_fraction * 1.05 / max(mc, 1e-9), o.volume_fraction + 0.01, _VF_MAX))
                    if any(abs(vf - q.volume_fraction) < 0.005 for q in outcomes):
                        vf = min(vf + 0.02, _VF_MAX)
                    if memory is not None:
                        vf = memory.snap(vf, skip=[q.volume_fraction for q in outcomes])
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
                       seconds=time.time() - t0, warnings=list(domain.warnings))
