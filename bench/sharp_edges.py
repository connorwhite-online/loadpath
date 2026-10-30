"""
Edge accuracy before and after sharp meshing, and what STEP export keeps.

    python bench/sharp_edges.py [--out out/sharp] [--only arm-link-solid,grip-turret] [--no-step]

Field parts (the arm link as a field graph, examples/fields): meshed with
sharp=False (the whole field by marching cubes, as before) and sharp=True
(hard booleans with plain shapes done exactly), measured with
sharp.edge_accuracy against the graph's own exact field. Study parts (the
arm-link example and the GRIP turret and wrist yoke eval cases, coarse
tier): the delivered STL and its STEP, measured against the study's
keep-outs, preserved shapes and design space. Writes summary.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.sharp import edge_accuracy, plan, plan_refs, study_refs  # noqa: E402

FIELD = {"arm-link-solid": ("solid", 0.4), "arm-link-graded": ("part", 0.3)}
STUDIES = {"arm-link": "example:arm-link", "grip-turret": "grip-turret.json", "grip-wrist-yoke": "grip-wrist-yoke.json"}


def _step(mesh, out, name, refs, solid=None, boundary=None):
    from loadpath.step import export_step, read_step, shape_to_trimesh

    path = os.path.join(out, f"{name}.step")
    try:
        rep = export_step(mesh, path, name=name)
    except Exception as exc:  # noqa: BLE001 - record why, measure the rest
        return {"error": f"{type(exc).__name__}: {exc}"}
    rep.pop("shape")
    back = shape_to_trimesh(read_step(path))
    rep["edgeAccuracy"] = edge_accuracy(back, refs, solid=solid, boundary=boundary)
    return rep


def field_part(key, out, step):
    from loadpath.fields import FieldGraph

    node, pitch = FIELD[key]
    doc = json.loads((ROOT / "examples/fields/arm-link-graded.json").read_text())
    g = FieldGraph(doc, base_dir=str(ROOT / "examples/fields"))
    F = g.field(node)
    refs = plan_refs(plan(g, node))

    def boundary(P):
        return np.abs(F(P)) < 1e-6

    row = {"kind": "field", "node": node, "pitch": pitch}
    for sharp in (False, True):
        t0 = time.time()
        m = g.mesh(pitch=pitch, name=node, sharp=sharp)
        tag = "sharp" if sharp else "before"
        m.export(os.path.join(out, f"{key}-{tag}.stl"))
        row[tag] = {"seconds": round(time.time() - t0, 1), "faces": int(len(m.faces)),
                    "watertight": bool(m.is_watertight), "volume": round(float(m.volume), 1),
                    "edgeAccuracy": edge_accuracy(m, refs, boundary=boundary)}
        print(key, tag, json.dumps(row[tag]), flush=True)
        if sharp and step:
            row["step"] = _step(m, out, key, refs, boundary=boundary)
            print(key, "step", json.dumps(row["step"]), flush=True)
    return row


def study_part(key, out, step):
    from loadpath.evals.run import HERE, _tier_globals, apply_tier
    from loadpath.spec import study_from_dict
    from loadpath.study import run_study
    from loadpath.tasks import example

    src = STUDIES[key]
    if src.startswith("example:"):
        raw, base = example(src.split(":", 1)[1]), None
    else:
        p = os.path.join(HERE, "cases", src)
        raw, base = json.load(open(p)), os.path.dirname(p)
    _tier_globals("coarse")
    raw = apply_tier(raw, "coarse")
    study = study_from_dict(raw, base_dir=base)
    t0 = time.time()
    res = run_study(study)
    o = res.outcomes[res.best if res.best is not None else -1]
    m = o.mesh
    m.export(os.path.join(out, f"{key}.stl"))
    refs = study_refs(study)
    row = {"kind": "study", "seconds": round(time.time() - t0, 1), "mass": round(o.report["mass"], 1),
           "safetyFactor": round(o.report["safetyFactor"], 2), "faces": int(len(m.faces)),
           "edgeAccuracy": edge_accuracy(m, refs)}
    print(key, json.dumps(row), flush=True)
    if step:
        row["step"] = _step(m, out, key, refs)
        print(key, "step", json.dumps(row["step"]), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/sharp")
    ap.add_argument("--only")
    ap.add_argument("--no-step", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keys = a.only.split(",") if a.only else list(FIELD) + list(STUDIES)
    path = os.path.join(a.out, "summary.json")
    rows = json.load(open(path)) if os.path.exists(path) else {}
    for k in keys:
        rows[k] = field_part(k, a.out, not a.no_step) if k in FIELD else study_part(k, a.out, not a.no_step)
        with open(path, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
