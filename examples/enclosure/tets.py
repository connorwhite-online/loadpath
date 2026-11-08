"""
Re-check a delivered part on 10-node tets (crosscheck.py: fTetWild volume
mesh, an independent solver from the cut-cell verify), every load case.

    python examples/enclosure/tets.py out/enclosure
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))


def main(argv=None):
    import trimesh

    from loadpath.crosscheck import crosscheck
    from loadpath.spec import study_from_dict
    d = (argv or sys.argv[1:])[0]
    sp = [f for f in os.listdir(d) if f.endswith(".study.json")][0]
    st = study_from_dict(json.load(open(os.path.join(d, sp))), base_dir=os.path.abspath(d))
    part = os.path.basename(json.load(open(os.path.join(d, "checks.json")))["part"])
    r = crosscheck(st, trimesh.load(os.path.join(d, "design", part), force="mesh"))
    out = {k: r.get(k) for k in ("safetyFactor", "nodes", "elements", "seconds", "mesher", "backend")}
    out["cases"] = [{k: c.get(k) for k in ("name", "peakStress", "safetyFactor", "deflection")} for c in r["cases"]]
    json.dump(out, open(os.path.join(d, "tets.json"), "w"), indent=1, default=float)
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
