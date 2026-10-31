"""
Moonshot (round 2): global layout search at a fixed volume fraction.

The safety-factor search treats the verified margin as a function of the
volume fraction, but SIMP settles in a different local layout (basin) from
run to run, so the margin is not monotone in the fraction. This script asks:
at a FIXED fraction, how far apart are the basins we can reach, and does the
strongest of them beat the single run at a heavier fraction?

Strategies, each delivered at the study's own filter radius and verified on
the delivered mesh exactly as the search verifies (study.design_at):

  main      the search's cold start (coarse layout first on big grids)
  warm      warm-started from main's design at the fraction above (the chain)
  deflate   Papadopoulos/Farrell style deflation: run k is repelled from
            every design found so far at this fraction
            (simp.SimpOptions.deflate), then polished undeflated from beta 4
            so it is a local optimum of the plain problem again
  noise     uniform start plus smoothed random density noise
  radius    layout found at 1.6x the filter radius, refined at 1x

Usage (one process, single-threaded BLAS):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
    python moonshots/round2/basins.py loadpath/examples/arm-link.json \\
      --cells 20000 --fractions 0.35 0.30 0.25 --out moonshots/round2/arm-link.json
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import time

import numpy as np

import loadpath.study as S
from loadpath.domain import build_domain
from loadpath.simp import filter_radius
from loadpath.spec import study_from_dict


@contextlib.contextmanager
def simp_extra(**kw):
    """design_at builds its own SimpOptions; inject extra fields into it."""
    if not kw:
        yield
        return
    orig = S.SimpOptions
    S.SimpOptions = functools.partial(orig, **kw)
    try:
        yield
    finally:
        S.SimpOptions = orig


def rms(a, b, design):
    return float(np.sqrt(np.mean((a[design] - b[design]) ** 2)))


def row(tag, vf, o, secs, iters):
    r = o.report
    cases = r.get("cases") or []
    defl = max((c.get("deflection", 0.0) for c in cases), default=float("nan"))
    peak = max((c.get("peakStress", 0.0) for c in cases), default=float("nan"))
    return {"strategy": tag, "vf": vf, "mass": r.get("mass"), "sf": r.get("safetyFactor"),
            "peak": peak, "deflection": defl, "margin": r.get("margin"), "passes": o.passes,
            "compliance": o.optimizer.get("compliance"), "iterations": iters,
            "seconds": round(secs, 1), "bodies": r.get("bodies")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--cells", type=int, default=20000)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.35, 0.30, 0.25])
    ap.add_argument("--deflations", type=int, default=3)
    ap.add_argument("--weight", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--noise", type=int, default=2)
    ap.add_argument("--strategies", default="main,warm,deflate,noise,radius")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    strategies = a.strategies.split(",")

    with open(a.study) as f:
        raw = json.load(f)
    raw["resolution"] = {"maxElements": a.cells}
    study = study_from_dict(raw, base_dir=os.path.dirname(os.path.abspath(a.study)))
    domain = build_domain(study)
    radius = filter_radius(domain, study.length_scale, study.min_hole)
    design = ~domain.preserve
    big = domain.n_elem >= S._COARSE_MIN_CELLS
    print(json.dumps({"cells": int(domain.n_elem), "pitch": domain.pitch, "radius": radius, "big": big}),
          flush=True)

    rows, designs = [], {}
    prev = None   # main's outcome at the fraction above
    rng = np.random.default_rng(0)
    t_all = time.time()

    def run(vf, extra=None, **kw):
        t = time.time()
        n = [0]

        def on_iter(rec, xp):
            n[0] += 1
        with simp_extra(**(extra or {})):
            o = S.design_at(study, domain, kw.pop("radius", radius), vf, on_iter=on_iter, **kw)
        return o, time.time() - t, n[0]

    def record(tag, vf, o, secs, iters, **more):
        rr = row(tag, vf, o, secs, iters)
        found = designs.setdefault(vf, [])
        rr["minDistance"] = min((rms(o.xphys, q, design) for _, q in found), default=None)
        rr.update(more)
        found.append((tag, o.xphys))
        rows.append(rr)
        print(json.dumps(rr), flush=True)
        return o

    def dump():
        with open(a.out, "w") as f:
            json.dump({"study": a.study, "cells": int(domain.n_elem), "pitch": domain.pitch,
                       "weight": a.weight, "power": a.power, "seconds": time.time() - t_all,
                       "rows": rows}, f, indent=1)

    for vf in sorted(a.fractions, reverse=True):
        # 1. main's cold start
        if big:
            t = time.time()
            x0 = S._coarse_start(study, domain, vf, radius, None)
            tc = time.time() - t
            o, s, n = run(vf, x0=x0, beta0=S._WARM_BETA)
            s += tc
        else:
            o, s, n = run(vf)
        main_o = record("main", vf, o, s, n)
        # 2. warm from main's design at the fraction above (the search's chain)
        if "warm" in strategies and prev is not None:
            o, s, n = run(vf, x0=prev.x, beta0=S._WARM_BETA)
            record("warm", vf, o, s, n)
        prev = main_o
        dump()
        # 3. deflation: repel from every design found so far at this fraction
        if "deflate" in strategies:
            for k in range(a.deflations):
                found = tuple(q for _, q in designs[vf])
                od, s1, n1 = run(vf, verify=False,
                                 extra=dict(deflate=found, deflate_weight=a.weight, deflate_power=a.power))
                raw_d = min(rms(od.xphys, q, design) for q in found)
                # polish undeflated from where deflation left it
                o, s2, n2 = run(vf, x0=od.x, beta0=S._WARM_BETA)
                record(f"deflate{k + 1}", vf, o, s1 + s2, n1 + n2, rawDistance=raw_d,
                       rawCompliance=od.optimizer.get("compliance"))
                dump()
        # 4. random density noise on the uniform start
        if "noise" in strategies:
            from scipy import ndimage
            ijk = tuple(domain.mesh.elem_ijk.T)
            for k in range(a.noise):
                g = ndimage.gaussian_filter(rng.standard_normal(domain.dims), sigma=3.0 * radius / domain.pitch)
                g /= g.std() + 1e-12
                x0 = np.clip(vf + 0.25 * g[ijk], 0.0, 1.0)
                o, s, n = run(vf, x0=x0, beta0=1.0)
                record(f"noise{k + 1}", vf, o, s, n)
                dump()
        # 5. a coarser length scale for the layout, then refined
        if "radius" in strategies:
            od, s1, n1 = run(vf, radius=1.6 * radius, verify=False)
            o, s2, n2 = run(vf, x0=od.x, beta0=S._WARM_BETA)
            record("radius", vf, o, s1 + s2, n1 + n2)
            dump()
    dump()


if __name__ == "__main__":
    main()
