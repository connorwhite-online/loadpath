"""
Why a study's search stopped where it did: run an eval case as the eval
does, record every outcome (fraction, voxel SF, calibrated SF, tet-confirmed
SF, how it was started), then read every lighter outcome the search did not
confirm on tets on the same cross-check (same contact band as study.py's
tet_confirm). One JSON per case.

Run: python moonshots/im-hex8/search_probe.py OUTDIR case [case ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def main():
    from loadpath.crosscheck import crosscheck, element_size
    from loadpath.domain import build_domain
    from loadpath.evals.run import HERE, apply_tier
    from loadpath.spec import study_from_dict
    from loadpath.study import run_study

    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name in sys.argv[2:]:
        path = os.path.join(HERE, "cases", f"{name}.json")
        raw = apply_tier(json.load(open(path)), "full")
        study = study_from_dict(raw, base_dir=os.path.dirname(path))
        events = []

        def prog(e, _x):
            if e.get("stage") in ("outcome", "confirm", "confirmed"):
                keep = {k: e.get(k) for k in ("stage", "volumeFraction", "mass", "safetyFactor", "voxelSafetyFactor",
                                              "margin", "passes", "start", "method", "calibration")}
                events.append(keep)
                print(json.dumps({"case": name, **{k: v for k, v in keep.items() if v is not None}}, default=float),
                      flush=True)

        t0 = time.time()
        res = run_study(study, on_progress=prog)
        seconds = time.time() - t0
        dom = build_domain(study)
        band = max(2.0 * element_size(study), 1.5 * dom.pitch)
        best = res.outcomes[res.best] if res.best is not None else None
        rows = []
        for i, o in enumerate(res.outcomes):
            r = o.report
            row = {"i": i, "volumeFraction": o.volume_fraction, "mass": r.get("mass"),
                   "voxelSF": getattr(o, "voxel_sf", None), "reportedSF": r.get("safetyFactor"),
                   "margin": r.get("margin"), "passes": o.passes, "start": r.get("start"),
                   "bodies": r.get("bodies"), "best": i == res.best,
                   "calibrated": (r.get("calibration") or {}).get("safetyFactor"),
                   "tetConfirmed": (r.get("confirm") or {}).get("safetyFactor")
                   if (r.get("confirm") or {}).get("method") == "tet" else None,
                   "tetCheck": (r.get("tetCheck") or {}).get("safetyFactor"),
                   "voxelConfirm": (r.get("confirm") or {}).get("safetyFactor")
                   if (r.get("confirm") or {}) and (r.get("confirm") or {}).get("method") != "tet" else None}
            lighter = best is not None and o.volume_fraction < best.volume_fraction
            if (row["tetConfirmed"] is None and row["tetCheck"] is None and (lighter or i == res.best)
                    and (r.get("bodies") or 1) <= 1 and (row["voxelSF"] or 0) > 0):
                try:
                    cc = crosscheck(study, o.mesh, contact_band=band)
                    row["tetProbe"] = cc["safetyFactor"]
                except Exception as exc:  # noqa: BLE001
                    row["tetProbe"] = f"failed: {exc}"
            rows.append(row)
            print(json.dumps({"case": name, "row": row}, default=float), flush=True)
            o.mesh.export(os.path.join(out_dir, f"{name}-{i}-{o.volume_fraction:.4f}.stl"))
        json.dump({"case": name, "target": study.goal.safety_factor, "maxDeflection": study.goal.max_deflection,
                   "band": band, "seconds": seconds, "best": res.best, "events": events, "outcomes": rows},
                  open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
