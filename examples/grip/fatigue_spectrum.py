"""
Fatigue of GRIP's links under the duty the sweep describes, against the
reading the studies give (each governing case's peak, every cycle).

The derived studies carry two or three governing cases, each the worst of
thousands of sampled states (the heaviest payload, the worst pose, full-speed
bang-bang moves cut back to the rated torque, a stalled joint), and give
each one the whole life's cycles of its kind (1e5 moves, 1e4 pushes, 1e3
stalls; coopt.CYCLES). That is a bound: every move is the worst move. Here
the same sweep is the spectrum instead: each of the 1e5 moves is a random
state of the sweep (pose, 0-1 kg payload, speeds and accelerations), and so
on for pushes and stalls. Stress is linear in the wrench, so the six
unit-wrench solves on the solid design space give each sample's stress
relative to the envelope's; the part's own peak (verified, 150k cells,
under the governing cases) scales it, and Miner's rule adds the damage.

    python examples/grip/fatigue_spectrum.py --grip ../grip --build DIR [--links forearm,...]

Writes OUT/fatigue.json: per link, the part's static SF, fatigue SF with
every cycle at the governing peak (the study's reading), and fatigue SF
under the sweep's own spectrum.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import coopt  # noqa: E402
import derive as grip_derive  # noqa: E402
from loadpath.assembly import _unit_stress, governing_cases, region_centroid, sweep_loads  # noqa: E402
from loadpath.durability import cycles_to_failure, durability_data  # noqa: E402
from loadpath.fem import von_mises  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def fatigue(arm, link, mesh, samples=1500):
    study = json.load(open(os.path.join(coopt.SDIR, f"{link}.json")))
    region = grip_derive.LINKS[link]
    sw = sweep_loads(arm.robot, link, payload=grip_derive.PAYLOAD, region=region, samples=samples,
                     point=region_centroid(study, region, base_dir=coopt.SDIR), balanced=("j2",))
    S, _, _ = _unit_stress(study, region, base_dir=coopt.SDIR)
    R = np.empty((len(sw.wrenches), S.shape[1]), np.float32)
    for s in range(0, len(R), 64):
        sig = np.einsum("nk,kec->nec", sw.wrenches[s:s + 64], S)
        R[s:s + 64] = von_mises(sig.reshape(-1, 6)).reshape(len(sig), -1)
    ratio = R.max(axis=1) / R.max()                  # each sample's peak against the sweep's worst
    picked = governing_cases(sw, study, base_dir=coopt.SDIR)
    cases = picked["cases"]
    for c in cases:
        c["cycles"] = coopt.CYCLES[c["name"].split("-")[2]]
    rep = coopt.check(link, mesh, cases)
    st = study_from_dict(study, base_dir=coopt.SDIR)
    data, uts = durability_data(st.material), st.material.strength
    peak = max(c["peakMPa"] for c in rep["cases"].values())
    kinds = np.array([k.split(":")[0] for k in sw.kinds])
    worst = spectrum = 0.0
    out_k = {}
    for kind, n in coopt.CYCLES.items():
        m = kinds == kind
        if not m.any():
            continue
        r = ratio[m]
        d_worst = n / cycles_to_failure(data, uts, peak * r.max())
        d_spec = float(sum((n / m.sum()) / cycles_to_failure(data, uts, peak * x) for x in r))
        worst += d_worst
        spectrum += d_spec
        out_k[kind] = {"cycles": n, "samples": int(m.sum()), "peakRatioMax": round(float(r.max()), 3),
                       "peakRatioMedian": round(float(np.median(r)), 3),
                       "damageWorstEveryCycle": d_worst, "damageSpectrum": d_spec}
    k = data.fatigue_slope
    return {"g": rep["g"], "staticSf": rep["sf"], "peakMPa": peak, "durabilitySf": rep["durabilitySf"],
            "fatigueSfWorstEveryCycle": round(worst ** (-1 / k), 2),
            "fatigueSfSpectrum": round(spectrum ** (-1 / k), 2), "kinds": out_k,
            "curve": data.to_dict()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", required=True)
    ap.add_argument("--build", required=True, help="the parts: DIR/<link>.stl")
    ap.add_argument("--links", default=",".join(coopt.ORDER))
    ap.add_argument("--out", default="grip-fatigue")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    arm = coopt.arm_model(args.grip, args.build)
    path = os.path.join(args.out, "fatigue.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for link in args.links.split(","):
        res[link] = fatigue(arm, link, arm.parts[link][0][0])
        print(link, {k: res[link][k] for k in ("g", "staticSf", "durabilitySf", "fatigueSfWorstEveryCycle",
                                              "fatigueSfSpectrum")}, flush=True)
        json.dump(res, open(path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
