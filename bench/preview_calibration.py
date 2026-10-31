"""
The editor's live stress preview (preview.py) against the full check
(verify_part on the meshed part) over a set of parts with different
stress raisers: the arm link envelope at three web heights and a sharp
blend, a cantilever bar, an L-bracket with a filleted inner corner and a
plate with a hole in tension. Prints and writes
bench/results/preview_calibration.json: per part the preview's SF (raw
and corrected), the check's, their ratio, deflections and seconds.

    python bench/preview_calibration.py [--quick] [--reuse]
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loadpath.fields import FieldGraph, load_study  # noqa: E402
from loadpath.preview import PreviewState, stress_preview  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402


def _arm(web_h=9.0, web_w=11.0, blend=12.0):
    doc = json.load(open(os.path.join(ROOT, "examples/fields/arm-link-tune.json")))
    n = doc["nodes"]
    n["web"]["shape"]["max"][2], n["web"]["shape"]["min"][2] = web_h, -web_h
    n["web"]["shape"]["max"][1], n["web"]["shape"]["min"][1] = web_w, -web_w
    n["blob"]["blend"] = blend
    study = load_study(doc["tune"]["study"], os.path.join(ROOT, "examples/fields"), doc.get("material"))
    return doc, study


def _bar():
    from tests.test_tune import GRAPH, STUDY

    doc = json.loads(json.dumps(GRAPH))
    doc["nodes"]["bar"]["shape"].update(min=[0, -5, -4], max=[60, 5, 4])
    return doc, study_from_dict(STUDY)


def _bracket(fillet=3.0):
    """An L: a 10 mm wall bolted at its top, a 50 mm arm out from its foot
    loaded down at the tip; the inner corner blended by `fillet`."""
    doc = {"name": "bracket", "nodes": {
        "wall": {"op": "shape", "shape": {"type": "box", "min": [0, -10, 0], "max": [10, 10, 60]}},
        "arm": {"op": "shape", "shape": {"type": "box", "min": [0, -10, 0], "max": [60, 10, 10]}},
        "part": {"op": "union", "inputs": ["wall", "arm"], "blend": fillet}}, "output": "part", "material": "petg"}
    study = study_from_dict({
        "name": "bracket", "designSpace": [{"type": "box", "min": [0, -10, 0], "max": [60, 10, 60]}],
        "preserve": [{"id": "top", "shape": {"type": "box", "min": [0, -10, 52], "max": [10, 10, 60]}},
                     {"id": "tip", "shape": {"type": "box", "min": [52, -10, 0], "max": [60, 10, 10]}}],
        "supports": [{"region": "top"}],
        "loadCases": [{"name": "down", "loads": [{"region": "tip", "force": [0, 0, -40]}]}],
        "material": "petg", "goal": {"safetyFactor": 2}})
    return doc, study


def _plate():
    """A 5 mm plate with a 12 mm hole pulled along its length (Kt about 2.2)."""
    doc = {"name": "plate", "nodes": {
        "plate": {"op": "shape", "shape": {"type": "box", "min": [0, -15, -2.5], "max": [90, 15, 2.5]}},
        "hole": {"op": "shape", "shape": {"type": "cylinder", "center": [45, 0, 0], "axis": "z", "length": 10, "r": 6}},
        "part": {"op": "subtract", "inputs": ["plate", "hole"]}}, "output": "part", "material": "petg"}
    study = study_from_dict({
        "name": "plate", "designSpace": [{"type": "box", "min": [0, -15, -2.5], "max": [90, 15, 2.5]}],
        "preserve": [{"id": "grip", "shape": {"type": "box", "min": [0, -15, -2.5], "max": [8, 15, 2.5]}},
                     {"id": "pull", "shape": {"type": "box", "min": [82, -15, -2.5], "max": [90, 15, 2.5]}}],
        "supports": [{"region": "grip"}],
        "loadCases": [{"name": "pull", "loads": [{"region": "pull", "force": [600, 0, 0]}]}],
        "material": "petg", "goal": {"safetyFactor": 2}})
    return doc, study


PARTS = {
    "arm link, web 9 mm": lambda: _arm(9.0),
    "arm link, web 6.5 mm": lambda: _arm(6.5),
    "arm link, narrow web": lambda: _arm(9.0, 7.0),
    "arm link, sharp blend 3": lambda: _arm(9.0, 11.0, 3.0),
    "cantilever bar": _bar,
    "L-bracket, fillet 3": lambda: _bracket(3.0),
    "L-bracket, fillet 1": lambda: _bracket(1.0),
    "plate with hole": _plate,
}


OUT = os.path.join(ROOT, "bench/results/preview_calibration.json")


def main(quick=False, reuse=False):
    """`reuse` keeps the checks already in the results file (they take
    30-70 s each and don't depend on the preview)."""
    old = {}
    if reuse and os.path.exists(OUT):
        old = {r["part"]: r for r in json.load(open(OUT))}
    rows = []
    for name, make in PARTS.items():
        doc, study = make()
        g = FieldGraph(doc)
        stress_preview(doc, study=study, graph=g)            # warm
        t0 = time.time()
        p = stress_preview(doc, study=study, graph=g, state=PreviewState())
        tp = time.time() - t0
        if name in old:
            chk = {k: old[name][k] for k in ("checkSF", "checkDeflection", "checkSeconds")}
        else:
            t0 = time.time()
            v = verify_part(study, g.mesh(pitch=0.5 if not quick else 0.8), max_cells=60_000 if not quick else 20_000)
            chk = {"checkSF": v["safetyFactor"], "checkDeflection": max(c["deflection"] for c in v["cases"]),
                   "checkSeconds": round(time.time() - t0, 1)}
        row = {"part": name, "previewSF": p["safetyFactor"], "rawSF": p["rawSafetyFactor"],
               "refinedSF": p["refinedSafetyFactor"], "hotspotFactor": p["hotspotFactor"], **chk,
               "ratio": p["safetyFactor"] / chk["checkSF"], "rawRatio": p["rawSafetyFactor"] / chk["checkSF"],
               "refinedRatio": p["refinedSafetyFactor"] / chk["checkSF"],
               "previewDeflection": p["deflection"], "previewSeconds": round(tp, 2), "cells": p["cells"]}
        rows.append(row)
        print(f"{name:26s} SF preview {p['safetyFactor']:6.2f} (coarse {row['rawSF']:6.2f}, refined {row['refinedSF']:6.2f})"
              f"  check {chk['checkSF']:6.2f}  ratio {row['ratio']:.2f} (coarse {row['rawRatio']:.2f},"
              f" refined {row['refinedRatio']:.2f})  deflection {p['deflection']:.3f}/{chk['checkDeflection']:.3f}"
              f"  {tp:.1f} s / {chk['checkSeconds']:.0f} s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=2)
    for k in ("rawRatio", "refinedRatio", "ratio"):
        r = [x[k] for x in rows]
        print(f"{k}: {min(r):.2f} .. {max(r):.2f}")
    return rows


if __name__ == "__main__":
    main("--quick" in sys.argv, "--reuse" in sys.argv)
