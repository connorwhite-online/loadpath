"""
loadpath's lattice models against published compression tests.

Each specimen in loadpath/data/published_experiments.json is a printed cube
of lattice (cut open at its sides) squeezed between flat platens along the
build axis, with its measured modulus and strength. Every geometry is
predicted three ways:

- table: the periodic unit cell's modulus along z and its first-yield
  stress (lattice_tensors: the stiffness and wall-stress tables the
  homogenized check reads), i.e. an infinite lattice;
- homogenized: `homogenized_check` of the specimen itself as a field graph
  (lattice intersected with the cube, 2 mm solid platens bonded on top
  and bottom, the top one held sideways), the check an agent runs on a
  part, free-surface knockdown and all;
- resolved: the same field graph voxelized at `--pitch` and solved on
  matrix-free multigrid (bench/lattice_aniso_validate.resolved): the
  designed geometry as printed, perfectly.

Strength is the first-yield reading: the applied stress at which the
lattice's peak wall stress (99.5th percentile in a cell, contact band left
out) reaches the solid's strength. Papers report a first peak or a plateau,
which a ductile lattice reaches after yielding, so first yield should read
low; a reading above the measurement means the lattice failed before any
wall yielded (buckling, defects).

    python bench/published_experiments.py                      # tables + homogenized + resolved
    python bench/published_experiments.py --only shev --no-resolved
    python bench/published_experiments.py --reuse              # keep resolved results already saved

Writes bench/results/published-experiments.json and prints prediction over
measurement for each specimen and model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from loadpath.validation import (geometry_key, load_experiments, measured, specimen_model,  # noqa: E402
                                 table_prediction)

PLATE = 2.0


def specimen_check(spec, mat, cells, knockdown=None, as_printed=False):
    from loadpath.homogenized import homogenized_check
    from loadpath.spec import study_from_dict

    doc, sd, area, height = specimen_model(spec, mat, plate=PLATE)
    st = study_from_dict(sd)
    cell = float(spec["lattice"]["cell"])
    r = homogenized_check(doc, st, cells=cells, contact_band=cell, free_surface=knockdown, as_printed=as_printed)
    c = r["cases"][0]
    out = _reading(c["deflection"], c["peakStress"], mat, area, height, sd)
    if "bucklingFactor" in c:
        # the applied stress at which the walls buckle (the check is linear)
        out["buckling"] = c["bucklingFactor"] * abs(sd["loadCases"][0]["loads"][0]["force"][2]) / area
    return out, r


def held_out_factors():
    """Each specimen's as-printed factor fitted without it
    (bench/lattice_print_fit.py), or None before that has run."""
    path = ROOT / "bench" / "results" / "lattice-print-fit.json"
    if not path.exists():
        return {}
    return {k: v["factorHeldOut"] for k, v in json.loads(path.read_text())["specimens"].items()}


def _reading(deflection, peak, mat, area, height, sd):
    force = abs(sd["loadCases"][0]["loads"][0]["force"][2])
    stress = force / area
    # the top platen's own squeeze (the bottom one is held throughout)
    d_lat = deflection - stress * PLATE / mat["E"]
    return {"E": stress * height / d_lat, "firstYield": mat["strength"] * stress / peak,
            "deflection": deflection, "peakStress": peak}


def resolved_check(spec, mat, pitch):
    from lattice_aniso_validate import resolved

    doc, sd, area, height = specimen_model(spec, mat, plate=PLATE)
    cell = float(spec["lattice"]["cell"])
    res = resolved(doc, sd, pitch, band=cell, cell_size=cell)
    c = res["cases"][0]
    out = _reading(c["deflection"], c["cellPeak"], mat, area, height, sd)
    # the same reading half a cell or more inside the cut-open sides
    inner = out["firstYield"] * c["cellPeak"] / c["innerCellPeak"] if c.get("innerCellPeak") else None
    out.update({"voxels": res["cells"], "pitch": pitch, "seconds": res["seconds"], "mass": res["mass"],
                "innerFirstYield": inner})
    return out


def _as_printed(row, spec, mat, meas, held, cells):
    """The homogenized check as an agent gets it for the specimen's process:
    the as-printed factor (held out: fitted without this specimen) and the
    buckling check."""
    k = held.get(spec["id"])
    if k is None:
        return
    ap, _ = specimen_check(spec, mat, cells, as_printed=k)
    row["asPrinted"] = ap | {"factor": k}
    line = (f"{spec['id']:22s} as printed (k {k:.2f}): E x{ap['E'] / meas['E']:.2f}, "
            f"first yield x{ap['firstYield'] / meas['strength']:.2f}")
    if "buckling" in ap:
        line += f", buckling x{ap['buckling'] / meas['strength']:.2f}"
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=0.2, help="resolved voxel pitch (mm)")
    ap.add_argument("--cells", type=int, default=60000, help="homogenized analysis cells")
    ap.add_argument("--only", default="", help="comma-separated id prefixes")
    ap.add_argument("--no-resolved", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="keep resolved results already in --out")
    ap.add_argument("--as-printed-only", action="store_true",
                    help="only (re)run the as-printed reading (held-out factor, buckling) on rows already in --out")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "published-experiments.json"))
    a = ap.parse_args()
    data = load_experiments()
    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    want = [p for p in a.only.split(",") if p]
    resolved_cache = {}
    held = held_out_factors()
    for row in results.values() if a.reuse else ():
        if row.get("resolvedUnit"):
            resolved_cache[row["geometry"]] = row["resolvedUnit"]
    for spec in data["specimens"]:
        if want and not any(spec["id"].startswith(p) for p in want):
            continue
        mat = data["materials"][spec["material"]]
        meas = measured(spec)
        if a.as_printed_only and spec["id"] in results:
            row = results[spec["id"]]
            _as_printed(row, spec, mat, meas, held, a.cells)
            out_path.write_text(json.dumps(results, indent=1, default=float))
            continue
        row = {"source": spec["source"], "process": spec["process"], "lattice": spec["lattice"],
               "geometry": geometry_key(spec), "measured": meas, "material": mat}
        t0 = time.time()
        row["table"] = table_prediction(spec, mat)
        hom, rep = specimen_check(spec, mat, a.cells)
        row["homogenized"] = hom | {"cells": rep["analysisCells"], "pitch": rep["analysisPitch"],
                                    "latticeExposed": rep["latticeExposed"], "mass": rep["mass"]}
        hom0, _ = specimen_check(spec, mat, a.cells, knockdown=0.0)
        row["homogenizedNoKnockdown"] = hom0
        # the resolved solve is linear in E and strength: one solve per
        # geometry at unit material, scaled to each specimen's
        key = row["geometry"]
        if not a.no_resolved:
            if key not in resolved_cache:
                unit = {"E": 1000.0, "strength": 1.0, "nu": mat["nu"], "density": 1.0}
                resolved_cache[key] = resolved_check(spec, unit, a.pitch)
            u = resolved_cache[key]
            row["resolvedUnit"] = u
            row["resolved"] = {"E": u["E"] * mat["E"] / 1000.0, "firstYield": u["firstYield"] * mat["strength"],
                               "innerFirstYield": None if u.get("innerFirstYield") is None
                               else u["innerFirstYield"] * mat["strength"],
                               "voxels": u["voxels"], "pitch": u["pitch"], "mass": u["mass"] * mat["density"]}
        elif spec["id"] in results and "resolved" in results[spec["id"]]:
            row["resolved"] = results[spec["id"]]["resolved"]
            row["resolvedUnit"] = results[spec["id"]].get("resolvedUnit")
        _as_printed(row, spec, mat, meas, held, a.cells)
        row["seconds"] = time.time() - t0
        results[spec["id"]] = row
        line = f"{spec['id']:22s} E meas {meas['E']:7.1f}"
        for m in ("table", "homogenized", "resolved"):
            if m in row:
                line += f" | {m[:5]} x{row[m]['E'] / meas['E']:.2f}"
        line += f" || strength meas {meas['strength']:5.2f} ({meas['strengthKind']})"
        for m in ("table", "homogenized", "resolved"):
            if m in row:
                line += f" | {m[:5]} x{row[m]['firstYield'] / meas['strength']:.2f}"
        print(line, flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
