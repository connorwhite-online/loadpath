"""
Mass against safety factor: a topology-optimized solid part, and the same
envelope kept whole with a solid skin over a stress-sized TPMS core.

    python bench/lattice_vs_solid.py [case] --out bench-out/ [--sf 3 5 8] [--kind gyroid]

`case` is an eval case name (default arm-link, the GRIP-like upper-arm
link). Writes results.json, results.md and renders. Every number is the
delivered mesh's own verification (verify_part), not the optimizer's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from loadpath.evals.run import HERE as CASES_DIR  # noqa: E402


def _row(label, rep, seconds, extra=None):
    return {"design": label, "mass": rep["mass"], "safetyFactor": rep["safetyFactor"],
            "deflection": max(c["deflection"] for c in rep["cases"]), "bodies": rep["bodies"],
            "enclosedCavity": rep["enclosedCavityVolume"], "analysisPitch": rep["analysisPitch"],
            "seconds": seconds, **(extra or {})}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("case", nargs="?", default="arm-link")
    p.add_argument("--out", default="bench-out")
    p.add_argument("--sf", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    p.add_argument("--kind", default="gyroid")
    p.add_argument("--cell", type=float, default=10.0)
    p.add_argument("--skin", type=float, default=1.2)
    p.add_argument("--process", help="override the case's process (e.g. sls: drain holes)")
    p.add_argument("--verify-cells", type=int, default=600_000)
    p.add_argument("--skip-solid", action="store_true")
    a = p.parse_args(argv)

    import trimesh

    from loadpath.lattice import LatticeOptions, envelope_mesh, lattice_infill
    from loadpath.render import render_views
    from loadpath.spec import study_from_dict
    from loadpath.study import run_study
    from loadpath.verify import verify_part

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(CASES_DIR, "cases", a.case + ".json")
    raw = json.load(open(path))
    raw.pop("expect", None)
    if a.process:
        raw["process"] = a.process
    study = study_from_dict(raw, base_dir=os.path.dirname(path))
    out_json = os.path.join(a.out, "results.json")
    rows = json.load(open(out_json))["rows"] if a.skip_solid and os.path.exists(out_json) else []

    def save():
        json.dump({"case": a.case, "rows": rows}, open(out_json, "w"), indent=2, default=float)

    if not a.skip_solid:
        t = time.time()
        res = run_study(study)
        secs = time.time() - t
        for i, o in enumerate(res.outcomes):
            label = f"solid TO vf {o.volume_fraction:.2f}" + (" (best)" if i == res.best else "")
            rows.append(_row(label, o.report, secs if i == res.best else None))
            if i == res.best:
                o.mesh.export(os.path.join(a.out, "solid-best.stl"))
                render_views(o.mesh, os.path.join(a.out, "solid-best.png"), study,
                             title=f"solid TO  {o.report['mass']:.1f} g  SF {o.report['safetyFactor']:.2f}")
        save()

    env = envelope_mesh(study)
    if not any(r["design"] == "solid envelope" for r in rows):
        rep = verify_part(study, env, refine=0)
        rows.append(_row("solid envelope", rep, None))
        save()
    for sf in a.sf:
        t = time.time()
        o = LatticeOptions(kind=a.kind, cell=a.cell, skin=a.skin, target_sf=sf)
        if study.process is not None:
            o.wall_min = max(o.wall_min, study.process.min_feature)
        lat, info = lattice_infill(study, env, None, o)
        build = time.time() - t
        stem = f"lattice-{a.kind}-c{a.cell:g}-s{a.skin:g}-sf{sf:g}"
        lat.export(os.path.join(a.out, stem + ".stl"))
        rep = verify_part(study, lat, refine=0, max_cells=a.verify_cells)
        pr = info["printability"]
        rows.append(_row(f"{a.skin:g} mm skin + {a.kind} {a.cell:g} mm core, target SF {sf:g}", rep, time.time() - t, {
            "predictedSF": info["sizing"]["predictedSF"], "buildSeconds": build,
            "solidCellFraction": info["sizing"]["solidCellFraction"], "drainHoles": len(pr["drainHoles"]),
            "trappedVolume": pr["trappedVolume"], "wallP1": pr["wallP1"]}))
        save()
        render_views(lat, os.path.join(a.out, stem + ".png"), study,
                     title=f"{a.kind} core, target SF {sf:g}: {rep['mass']:.1f} g  SF {rep['safetyFactor']:.2f}")
        cut = lat.slice_plane(env.centroid, [0, 0, -1], cap=True)
        render_views(cut, os.path.join(a.out, stem + "-cut.png"), study, title=f"{a.kind} core, cut at mid-height")

    lines = [f"# {a.case}: mass vs safety factor", "",
             "| design | mass g | SF (verified) | SF (predicted) | deflection mm | drains | trapped mm³ |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        pred = f"{r['predictedSF']:.2f}" if r.get("predictedSF") else ""
        lines.append(f"| {r['design']} | {r['mass']:.1f} | {r['safetyFactor']:.2f} | {pred} | "
                     f"{r['deflection']:.3f} | {r.get('drainHoles', '')} | "
                     f"{'' if r.get('trappedVolume') is None else round(r['trappedVolume'], 1)} |")
    open(os.path.join(a.out, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
