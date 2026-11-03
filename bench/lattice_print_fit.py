"""
Fit the as-printed lattice stiffness factor (loadpath/lattice_print.py) to
the published compression tests, and hold it out.

Reads bench/results/published-experiments.json (bench/published_experiments.py),
fits log(designed / measured modulus) on process group, strut flag and log
density, and reports:

- in-sample: each specimen's modulus ratio before and after the factor;
- leave one out: each specimen predicted from a fit to the other 32;
- leave one source out: each paper predicted from a fit to the other two
  (a process group only one paper tested keeps no intercept then, so its
  rows read as the strut and density terms alone: the hardest test);
- strength: the first-yield reading under the as-printed factor (stress
  from the strain the softer lattice takes) over the measured strength,
  held out.

    python bench/lattice_print_fit.py            # prints, writes bench/results/lattice-print-fit.json

The fitted coefficients printed at the end are the ones to paste into
lattice_print.PRINT_FIT (tests/test_lattice_print.py checks they match).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.lattice_print import fit, fit_rows, print_factor  # noqa: E402

GROUP_KIND = {"fdm": "fdm", "sls": "sls", "resin": "dlp"}


def factor(fit_, r):
    return float(print_factor(GROUP_KIND[r["group"]], r["type"], r["rho"], fit_)[0])


def summary(ratios):
    a = np.asarray(ratios, float)
    err = np.exp(np.abs(np.log(a)))
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "medianError": float(np.median(err)), "worstError": float(err.max())}


def main(path=ROOT / "bench" / "results" / "published-experiments.json",
         out=ROOT / "bench" / "results" / "lattice-print-fit.json"):
    rows = fit_rows(json.loads(Path(path).read_text()))
    full = fit(rows)
    per = {}
    for i, r in enumerate(rows):
        loo = fit(rows[:i] + rows[i + 1:])
        k_in, k_loo = factor(full, r), factor(loo, r)
        per[r["id"]] = {"source": r["source"], "group": r["group"], "type": r["type"], "rho": r["rho"],
                        "designedOverMeasured": r["E"] / r["Emeas"],
                        "factor": k_in, "asPrintedOverMeasured": k_in * r["E"] / r["Emeas"],
                        "factorHeldOut": k_loo, "heldOutOverMeasured": k_loo * r["E"] / r["Emeas"],
                        "firstYieldOverMeasured": r["firstYield"] / r["strength"],
                        "asPrintedFirstYieldHeldOut": k_loo * r["firstYield"] / r["strength"]}
    loso = {}
    for src in sorted({r["source"] for r in rows}):
        f = fit([r for r in rows if r["source"] != src])
        got = [factor(f, r) * r["E"] / r["Emeas"] for r in rows if r["source"] == src]
        loso[src] = summary(got)
        for r in rows:
            if r["source"] == src:
                per[r["id"]]["sourceHeldOutOverMeasured"] = factor(f, r) * r["E"] / r["Emeas"]
    res = {
        "fit": full,
        "multipliers": {"fdm": float(np.exp(full["intercept"]["fdm"])), "sls": float(np.exp(full["intercept"]["sls"])),
                        "resin": float(np.exp(full["intercept"]["resin"])), "strut": float(np.exp(full["strut"])),
                        "densityPerEfold": float(np.exp(full["densitySlope"]))},
        "designed": summary([p["designedOverMeasured"] for p in per.values()]),
        "inSample": summary([p["asPrintedOverMeasured"] for p in per.values()]),
        "leaveOneOut": summary([p["heldOutOverMeasured"] for p in per.values()]),
        "leaveSourceOut": loso,
        "strength": {"designed": summary([p["firstYieldOverMeasured"] for p in per.values()]),
                     "asPrintedHeldOut": summary([p["asPrintedFirstYieldHeldOut"] for p in per.values()])},
        "specimens": per,
    }
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"{'specimen':22s} {'designed':>8s} {'in':>6s} {'held':>6s} {'src-out':>7s} | strength held-out")
    for sid, p in per.items():
        print(f"{sid:22s} {p['designedOverMeasured']:8.2f} {p['asPrintedOverMeasured']:6.2f} "
              f"{p['heldOutOverMeasured']:6.2f} {p['sourceHeldOutOverMeasured']:7.2f} | "
              f"{p['firstYieldOverMeasured']:.2f} -> {p['asPrintedFirstYieldHeldOut']:.2f}")
    for k in ("designed", "inSample", "leaveOneOut"):
        s = res[k]
        print(f"{k:12s} modulus/measured median {s['median']:.2f}, {s['min']:.2f}-{s['max']:.2f}, "
              f"median error {s['medianError']:.2f}x, worst {s['worstError']:.2f}x")
    for src, s in loso.items():
        print(f"source out {src:16s} median {s['median']:.2f}, {s['min']:.2f}-{s['max']:.2f}")
    s = res["strength"]["asPrintedHeldOut"]
    print(f"strength (first yield / measured) as printed, held out: {s['min']:.2f}-{s['max']:.2f}")
    print("PRINT_FIT =", json.dumps({**full, "intercept": {k: round(v, 5) for k, v in full["intercept"].items()},
                                     "strut": round(full["strut"], 5),
                                     "densitySlope": round(full["densitySlope"], 5)}))
    return res


if __name__ == "__main__":
    main()
