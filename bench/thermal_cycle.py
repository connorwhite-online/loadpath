"""Designing for a duty cycle: examples/thermal/motor-mount-cycle.json (the
motor-mount eval case, 1 W through the motor face for 10 min in every 20,
30,000 cycles) run through run_study three ways at the same volume:
  - without the thermal block (the part designed as if cold),
  - with steady heat at the cycle's mean power (0.5 W, no cycle),
  - with the duty cycle,
then every part verified under the duty cycle: hot and cold safety
factors and thermal-cycling fatigue (durability).
At the eval's 45k cells (2026-10-31), all verified under the duty cycle:
  designed cold            27.5 g  hot SF 4.63  fatigue SF 2.39  cycling life 3.6e6
  mean steady heat         27.4 g  hot SF 4.77  fatigue SF 2.37  cycling life 3.4e6
  for the duty cycle       27.2 g  hot SF 4.54  fatigue SF 2.83  cycling life 9.2e6
(at 12k cells the cycle design led on hot SF instead: 4.24 vs 3.75 cold).
Run: python bench/thermal_cycle.py [maxElements]
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


def main(max_el=12000):
    base = ROOT / "examples/thermal"
    doc = json.loads((base / "motor-mount-cycle.json").read_text())
    doc["resolution"] = {"maxElements": max_el}
    cycle = study_from_dict(copy.deepcopy(doc), base_dir=str(base))
    cold = copy.deepcopy(doc)
    cold.pop("thermal")
    steady = copy.deepcopy(doc)
    c = steady["thermal"].pop("cycle")
    for h in steady["thermal"]["heat"]:
        h["watts"] *= c["on"] / c["period"]
    out = {}
    for name, d in (("designed cold", cold), ("designed for mean steady heat", steady),
                    ("designed for the duty cycle", doc)):
        t0 = time.time()
        res = run_study(study_from_dict(copy.deepcopy(d), base_dir=str(base)))
        o = res.outcomes[0]
        rep = verify_part(cycle, o.mesh, min_pitch=0.5 * res.domain["pitch"], contact_band=res.domain["pitch"])
        dur = rep.get("durability", {})
        out[name] = {"mass g": round(o.report["mass"], 1),
                     "hot SF": round(min(x["safetyFactor"] for x in rep["cases"]), 2),
                     "cold SF": round(min(x["coldSafetyFactor"] for x in rep["cases"]), 2),
                     "cycle stress MPa": round(max(x["cycleStress"] for x in rep["cases"]), 2),
                     "thermal-cycling life": round(dur.get("thermalCycling", {}).get("life", 0)),
                     "fatigue SF": round(dur.get("fatigueSafetyFactor") or 0, 2),
                     "max T": round(rep["thermal"]["maxTemperature"], 1),
                     "seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(out[name]), flush=True)
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12000)
