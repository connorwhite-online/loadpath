"""The study JSON's thermal block end to end: examples/thermal/
motor-mount-hot.json (the motor-mount eval case with the motor face at
75 C over a 25 C aluminium frame) run through run_study as written, and
again with the thermal block removed (the design you get without telling
the optimizer about heat). Both parts are then verified under the hot
study: belt loads plus the PETG's own expansion, moduli and strength
derated by temperature.

At the eval's 45k cells (2026-10-31): designed without heat 27.5 g, hot
safety factor 4.21; designed with heat 27.0 g, 4.71. With the face at
60 C instead the two came out even (4.77 vs 4.71): the belts dominate
until the PETG near the motor gets soft.
Run: python bench/thermal_study.py [maxElements] [study in examples/thermal]
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
    doc = json.loads((base / (sys.argv[2] if len(sys.argv) > 2 else "motor-mount-hot.json")).read_text())
    doc["resolution"] = {"maxElements": max_el}
    hot = study_from_dict(copy.deepcopy(doc), base_dir=str(base))
    cold_doc = copy.deepcopy(doc)
    cold_doc.pop("thermal")
    cold = study_from_dict(cold_doc, base_dir=str(base))
    out = {}
    for name, st in (("designed without heat", cold), ("designed with heat", hot)):
        t0 = time.time()
        res = run_study(st)
        o = res.outcomes[0]
        rep = verify_part(hot, o.mesh, min_pitch=0.5 * res.domain["pitch"], contact_band=res.domain["pitch"])
        cases = {c["name"]: round(c["safetyFactor"], 2) for c in rep["cases"]}
        out[name] = {"mass g": round(o.report["mass"], 1), "hot safety factor": round(rep["safetyFactor"], 2),
                     "per case": cases, "maxT": round(rep["thermal"]["maxTemperature"], 1),
                     "peak MPa": round(max(c["peakStress"] for c in rep["cases"]), 2),
                     "seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(out[name]), flush=True)
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12000)
