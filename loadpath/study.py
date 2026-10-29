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
from .reconstruct import ReconstructOptions, reconstruct
from .simp import SimpOptions, filter_radius, optimize
from .spec import Study
from .verify import verify_part

__all__ = ["Outcome", "StudyResult", "run_study"]

# Bounds for the safety-factor search.
_VF_MIN, _VF_MAX = 0.04, 0.7
_MAX_SEARCH_RUNS = 7
# Coarse-to-fine: below this many cells the fine solve is cheap enough alone.
_COARSE_MIN_CELLS = 12_000
# Projection sharpness a warm-started run resumes at.
_WARM_BETA = 4.0
# A cold start that fails with at least this margin is retried warm (_run_single).
_RETRY_MARGIN = 0.7
# The pick of a safety-factor search is checked again on a grid this many
# cells finer before it is delivered. The in-loop check resolves a fillet
# only as well as its grid: GRIP's turret read SF 2.26 there and 1.65 at
# 120k cells, its peak at a corner the coarser grid had averaged away.
_CONFIRM_CELLS = 150_000
_CONFIRM_RUNS = 3


def _coarse_start(study: Study, domain: Domain, vf: float, radius: float, should_stop) -> np.ndarray:
    """Find the layout on a grid twice as coarse (an eighth of the cells, so
    each iteration is roughly ten times cheaper), stopping while it is still
    soft (beta 4), then interpolate it onto the fine grid as the starting
    design. The fine run only has to sharpen and refine it."""
    from scipy import ndimage

    coarse = build_domain(study, pitch=2.0 * domain.pitch)
    res = optimize(coarse, study.elastic(design=True),
                   SimpOptions(volume_fraction=vf, filter_radius=max(radius, 1.5 * coarse.pitch),
                               max_iterations=60, beta_max=_WARM_BETA, beta_every=12,
                               build_direction=study.print_direction, extrusion=study.extrusion),
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
            **({"print": {k: v for k, v in r["print"].items() if k != "bridgeList"}} if "print" in r else {}),
            "warnings": r.get("warnings", []),
            "margin": r.get("margin"),
            "passes": self.passes,
            "start": r.get("start"),
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


def _margin(study: Study, report: dict) -> float | None:
    g = study.goal
    ms = []
    if g.safety_factor is not None:
        ms.append(report["safetyFactor"] / g.safety_factor)
    if g.max_deflection is not None:
        worst = max(c["deflection"] for c in report["cases"])
        ms.append(g.max_deflection / worst if worst > 0 else float("inf"))
    return min(ms) if ms else None


def _domain_info(d: Domain, radius: float) -> dict:
    return {
        "pitch": d.pitch,
        "cells": d.n_elem,
        "dims": list(d.dims),
        "preservedCells": int(d.preserve.sum()),
        "filterRadius": radius,
        "loadCases": d.case_names,
    }


def _workers(n: int) -> int:
    """Processes for independent outcomes: LOADPATH_WORKERS, else half the
    cores (each solve already uses a thread per load case)."""
    env = os.environ.get("LOADPATH_WORKERS")
    if env:
        return max(1, min(n, int(env)))
    return max(1, min(n, (os.cpu_count() or 2) // 2))


def _solve_alone(study: Study) -> StudyResult:
    return _run_single(study)


def _parallel(studies: list, on_progress, should_stop) -> list:
    """Run independent single-goal studies, in processes when it helps.
    Progress inside a worker is not streamed; each finished outcome is."""
    if _workers(len(studies)) <= 1:
        return [_run_single(s, on_progress, should_stop) for s in studies]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    results = [None] * len(studies)
    with ProcessPoolExecutor(_workers(len(studies)), mp_context=mp.get_context("spawn")) as ex:
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
        return _run_single(study, on_progress, should_stop)
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
    return StudyResult(study=study, domain=domain, outcomes=outcomes, best=best,
                       seconds=time.time() - t0, warnings=warnings)


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

    def attempt(vf: float, x0, beta0) -> Outcome:
        def on_iter(rec, xp):
            emit({"stage": "optimize", "volumeFraction": vf, **rec}, xp)

        res = optimize(domain, study.elastic(design=True),
                       SimpOptions(volume_fraction=vf, filter_radius=radius,
                                   max_iterations=study.max_iterations, x0=x0, beta_start=beta0,
                                   build_direction=study.print_direction, extrusion=study.extrusion),
                       on_iter=on_iter, should_stop=should_stop)
        emit({"stage": "mesh", "volumeFraction": vf})
        mesh, rinfo = reconstruct(study, domain, res.xphys, ReconstructOptions())
        emit({"stage": "verify", "volumeFraction": vf})
        report = verify_part(study, mesh, min_pitch=0.5 * domain.pitch, contact_band=domain.pitch)
        opt = {"iterations": res.iterations, "converged": res.converged, "seconds": res.seconds,
               "compliance": res.compliance, "caseCompliance": res.case_compliance,
               "achievedVolumeFraction": res.volume_fraction}
        o = Outcome(vf, mesh, report, opt, rinfo)
        o.xphys = res.xphys
        o.x = res.x
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
        if domain.n_elem < _COARSE_MIN_CELLS:
            o = attempt(vf, last_x, _WARM_BETA if last_x is not None else 1.0)
            o.report["start"] = "warm" if last_x is not None else "cold"
        else:
            # Every fraction starts from its own coarse layout. Warm-starting
            # from the previous fraction's design carried its layout down to
            # fractions it no longer suited: the wall bracket read SF 1.0-1.5
            # at 8-14% that way, and 3.2 at 10% started fresh (80 g delivered
            # where 50 g passes). But a stiffness-bound part can be the other
            # way round (GRIP's upper arm: 7.5% passed warm from 9% and failed
            # cold, on deflection), so a cold start that fails is tried again
            # from the nearest passing design above it, and the better kept.
            # Only when it failed narrowly: a retry costs a whole run (the
            # suite took 42 min instead of 26 retrying every failure), and
            # the upper arm's cold miss was at margin 0.79.
            o = attempt(vf, _coarse_start(study, domain, vf, radius, should_stop), _WARM_BETA)
            o.report["start"] = "cold"
            above = [q for q in outcomes if q.passes and q.volume_fraction > vf and getattr(q, "x", None) is not None]
            near = o.report.get("margin") is not None and o.report["margin"] >= _RETRY_MARGIN
            if o.passes is False and near and above and not (should_stop is not None and should_stop()):
                src = min(above, key=lambda q: q.volume_fraction)
                w = attempt(vf, src.x, _WARM_BETA)
                w.report["start"] = f"warm from {src.volume_fraction:.3f}"
                if w.report.get("margin", 0.0) > o.report.get("margin", 0.0) or (w.passes and not o.passes):
                    o = w
        last_x = o.x
        emit({"stage": "outcome", **o.summary()})
        return o

    def _confirm(o: Outcome) -> float:
        rep = verify_part(study, o.mesh, max_cells=_CONFIRM_CELLS, min_pitch=0.3 * domain.pitch,
                          contact_band=domain.pitch)
        mc = _margin(study, rep)
        o.report["confirm"] = {"safetyFactor": rep["safetyFactor"], "margin": mc,
                               "analysisPitch": rep["analysisPitch"], "analysisCells": rep["analysisCells"],
                               "cases": [{k: c[k] for k in ("name", "peakStress", "peakAt", "deflection")}
                                         for c in rep["cases"]]}
        # recorded, not adopted: the finer grid also resolves small features
        # the analysis grid smooths over, so its percentile is not comparable
        o.report["confirm"].update(p5Member=rep["p5Member"], minMember=rep["minMember"])
        if mc < o.report["margin"]:
            # report the finer (lower) reading: it is the one to trust
            o.report["margin"], o.report["safetyFactor"], o.report["cases"] = mc, rep["safetyFactor"], rep["cases"]
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
                        extrusion=study.extrusion),
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
            and study.print_direction is None and study.gravity is None:
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
    elif goal.safety_factor is not None or goal.max_deflection is not None:
        vf = float(goal.volume_fraction or 0.3)
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
            else:
                # Aim a little past the target; never more than halve or
                # double in one step, where the linear model stops holding.
                nxt = vf * 1.05 / max(m, 1e-9)
                nxt = float(np.clip(nxt, vf * 0.5, vf * 2.0))
            nxt = float(np.clip(nxt, _VF_MIN, _VF_MAX))
            if abs(nxt - vf) < 0.005:
                break
            vf = nxt
        passing = [i for i, o in enumerate(outcomes) if o.passes]
        best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
        # Confirm the pick on a finer grid; if it fails there, it no longer
        # passes, and the search steps up from it by the confirmed margin.
        for _ in range(_CONFIRM_RUNS):
            if best is None or (should_stop is not None and should_stop()):
                break
            o = outcomes[best]
            emit({"stage": "confirm", "volumeFraction": o.volume_fraction})
            mc = _confirm(o)
            emit({"stage": "confirmed", "volumeFraction": o.volume_fraction, "margin": mc})
            if mc >= 1.0:
                break
            o.passes = False
            vf = float(np.clip(o.volume_fraction * 1.05 / max(mc, 1e-9), o.volume_fraction + 0.01, _VF_MAX))
            if any(abs(vf - q.volume_fraction) < 0.005 for q in outcomes):
                vf = min(vf + 0.02, _VF_MAX)
            outcomes.append(run_one(vf))
            passing = [i for i, q in enumerate(outcomes) if q.passes]
            best = min(passing, key=lambda i: outcomes[i].report["mass"]) if passing else None
    else:
        fractions = [float(goal.volume_fraction)] + [v for v in goal.outcomes if v != goal.volume_fraction]
        if len(fractions) > 1 and _workers(len(fractions)) > 1:
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

    return StudyResult(study=study, domain=info, outcomes=outcomes, best=best,
                       seconds=time.time() - t0, warnings=list(domain.warnings))
