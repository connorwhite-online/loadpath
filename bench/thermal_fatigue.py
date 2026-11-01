"""Designing against thermal-cycling fatigue: examples/thermal/
motor-mount-cycle.json (1 W through the motor face, 10 min on / 10 off,
30,000 cycles) with thermal.cycle.fatigueWeight 0 (the hot moment alone,
the duty-cycle design before this option) and larger, at two resolutions.
Every part is verified under the duty cycle: hot SF (the lower of hot and
cold) and the fatigue SF on the thermal-cycling swing (durability).
Run: python bench/thermal_fatigue.py [maxElements ...] [--weights 0,1,3] [--constrain SF]
     [--fatigue-sf 3,3.5] (thermal.cycle.fatigueSafetyFactor: the swing as an MMA constraint)
(--constrain sizes each part to the lightest that verifies at that hot and
fatigue safety factor, durability.constrain, instead of a fixed volume.)
"""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.study import run_study  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402


def main(sizes=(12000, 45000), weights=(0.0, 1.0, 3.0), constrain=None, fsf=()):
    base = ROOT / "examples/thermal"
    doc = json.loads((base / "motor-mount-cycle.json").read_text())
    if constrain is not None:
        # size the part: the lightest that verifies at hot SF and fatigue SF
        # >= constrain (durability.constrain)
        doc["goal"] = {"safetyFactor": constrain}
        doc["durability"] = {"safetyFactor": constrain, "constrain": True}
    out = {}
    for n in sizes:
        d0 = copy.deepcopy(doc)
        d0["resolution"] = {"maxElements": n}
        check = study_from_dict(copy.deepcopy(d0), base_dir=str(base))
        runs = [("fatigueWeight", w) for w in weights] + [("fatigueSafetyFactor", v) for v in fsf]
        for key, w in runs:
            d = copy.deepcopy(d0)
            d["thermal"]["cycle"][key] = w
            if key == "fatigueSafetyFactor":
                d["thermal"]["cycle"].pop("fatigueWeight", None)
            t0 = time.time()
            res = run_study(study_from_dict(d, base_dir=str(base)))
            if constrain is not None and res.best is None:
                # no candidate met the targets: say so instead of reporting
                # the first one as if it were the answer
                tried = [(round(c.report.get("mass", 0), 1),
                          round(((c.report.get("durability") or {}).get("fatigueSafetyFactor") or 0), 2))
                         for c in res.outcomes]
                print(n, key, w, json.dumps({"no part meets the targets; tried (mass g, fatigue SF)": tried}),
                      flush=True)
                continue
            o = res.outcomes[res.best if res.best is not None else 0]
            try:
                rep = verify_part(check, o.mesh, min_pitch=0.5 * res.domain["pitch"],
                                  contact_band=res.domain["pitch"])
            except ValueError as e:
                print(n, key, w, json.dumps({"failed": str(e)[:120], "mass g": round(o.report.get("mass", 0), 1)}),
                      flush=True)
                continue
            dur = rep["durability"]
            row = {"mass g": round(o.report["mass"], 1), "hot SF": round(rep["safetyFactor"], 2),
                   "fatigue SF": round(dur["fatigueSafetyFactor"], 2),
                   "cycle stress MPa": round(dur["thermalCycling"]["stress"], 2),
                   "cycling life": f'{dur["thermalCycling"]["life"]:.2e}',
                   "max T": round(rep["thermal"]["maxTemperature"], 1), "seconds": round(time.time() - t0)}
            out[f"{n} cells, {key} {w:g}"] = row
            print(n, key, w, json.dumps(row), flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    ws = (0.0, 1.0, 3.0)
    if "--weights" in args:
        i = args.index("--weights")
        ws = tuple(float(v) for v in args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    con = None
    if "--constrain" in args:
        i = args.index("--constrain")
        con = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    fs = ()
    if "--fatigue-sf" in args:
        i = args.index("--fatigue-sf")
        fs = tuple(float(v) for v in args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
        if "--weights" not in sys.argv:
            ws = ()
    main(tuple(int(a) for a in args) or (12000, 45000), ws, con, fs)
