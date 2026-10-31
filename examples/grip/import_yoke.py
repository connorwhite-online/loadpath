"""
GRIP's wrist yoke straight from its CAD: `loadpath import` on the STEP file
GRIP exports (exports/fdm/step/wrist_yoke.step), then the drafted study run
with the yoke's real loads.

The import finds the yoke's interfaces on its own: the four counterbored
M3 bolts of the J4 servo horn (the support the hand-written study in
GRIP's generative/studies/wrist_yoke.json uses) and the 61.7 mm J5 socket
(where its loads go). The hand-written loads are in the arm's frame; the
STEP is in the part's, which maps arm (x, y, z) to part (x, z, -y), so
they are rotated across. The suggested placeholder load is replaced by
them; nothing else in the drafted study is touched.

    python examples/grip/import_yoke.py --grip ../grip [--space part|hull] [--out OUT]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from loadpath.cadimport import import_part  # noqa: E402


def arm_to_part(v):
    v = np.asarray(v, float)
    return [round(float(v[0]), 3), round(float(v[2]), 3), round(float(-v[1]), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", required=True, help="a checkout of connorwhite-online/grip")
    ap.add_argument("--space", default="part", choices=["part", "hull", "box"])
    ap.add_argument("--out", default="loadpath-out/grip-import")
    ap.add_argument("--max-elements", type=int, default=60000)
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()

    step = os.path.join(args.grip, "exports/fdm/step/wrist_yoke.step")
    hand = json.load(open(os.path.join(args.grip, "generative/studies/wrist_yoke.json")))
    out = os.path.join(args.out, args.space)
    r = import_part(step, out, space=args.space, material="petg")
    feats, study = r["features"], r["study"]
    sug = feats["suggested"]
    print(json.dumps({"bores": len(feats["bores"]), "interfaces": [i["id"] for i in feats["interfaces"]],
                      "suggested": sug, "healed": feats.get("healed")}, indent=1))

    # the yoke's real loads, rotated into the part's frame, on the suggested load region
    region = study["loadCases"][0]["loads"][0]["region"]
    cases = []
    for c in hand["loadCases"]:
        loads = []
        for ld in c["loads"]:
            x = {"region": region}
            if "force" in ld:
                x["force"] = arm_to_part(ld["force"])
            if "moment" in ld:
                x["moment"] = arm_to_part(ld["moment"])
            loads.append(x)
        cases.append({"name": c["name"], "loads": loads})
    study["loadCases"] = cases
    # the same material, goal and member size as the hand-written study, so the two compare
    study["material"] = hand["material"]
    study["goal"] = {"safetyFactor": hand["goal"]["safetyFactor"]}
    study["minMember"] = hand.get("minMember", 3.0)
    study["resolution"] = {"maxElements": args.max_elements}
    path = os.path.join(out, "wrist_yoke.real-loads.json")
    json.dump(study, open(path, "w"), indent=1)
    print("study:", path)
    if not args.no_run:
        from loadpath.__main__ import main as cli
        return cli(["run", path, "--out", os.path.join(out, "run")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
