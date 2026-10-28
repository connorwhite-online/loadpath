"""
Command line: run a study file and write the parts, a report and renders.

    python -m loadpath run study.json --out out/
    python -m loadpath check study.json        # validate + show the voxel problem, no solve
    python -m loadpath mcp                     # local MCP server for agents (needs loadpath[mcp])

Progress goes to stderr as one JSON object per line, so an agent can tail it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def main(argv=None):
    from .domain import build_domain
    from .simp import filter_radius
    from .spec import study_from_dict

    p = argparse.ArgumentParser(prog="loadpath")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="optimize, mesh and verify a study")
    r.add_argument("study")
    r.add_argument("--out", default="loadpath-out")
    r.add_argument("--no-render", action="store_true")
    r.add_argument("--base-dir", help="folder relative mesh paths resolve against (default: the study's)")
    c = sub.add_parser("check", help="validate a study and report the voxel problem")
    c.add_argument("study")
    sub.add_parser("mcp", help="run a local MCP server (stdio) for agents")
    args = p.parse_args(argv)
    if args.cmd == "mcp":
        from .mcp_server import serve
        serve()
        return 0

    with open(args.study) as fh:
        raw = json.load(fh)
    base = getattr(args, "base_dir", None) or os.path.dirname(os.path.abspath(args.study))
    try:
        study = study_from_dict(raw, base_dir=base)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    if args.cmd == "check":
        try:
            d = build_domain(study)
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}))
            return 2
        print(json.dumps(_jsonable({
            "ok": True, "pitch": d.pitch, "cells": d.n_elem, "dims": list(d.dims),
            "preservedCells": int(d.preserve.sum()), "filterRadius": filter_radius(d, study.min_member),
            "supportNodes": int(d.support_nodes.sum()), "loadNodes": int(d.load_nodes.sum()),
            "warnings": d.warnings,
        }), indent=2))
        return 0

    from .study import run_study

    def progress(event, _field):
        if event.get("stage") == "optimize" and "iteration" in event and event["iteration"] % 5:
            return
        ev = {k: v for k, v in event.items() if k not in ("cases", "optimizer", "reconstruction")}
        print(json.dumps(_jsonable(ev)), file=sys.stderr, flush=True)

    result = run_study(study, on_progress=progress)
    os.makedirs(args.out, exist_ok=True)
    summary = result.to_dict()
    for i, o in enumerate(result.outcomes):
        stem = f"{study.name}-{int(round(o.volume_fraction * 100)):02d}"
        stl = os.path.join(args.out, stem + ".stl")
        o.mesh.export(stl)
        summary["outcomes"][i]["stl"] = stl
        if not args.no_render:
            from .render import render_views
            png = os.path.join(args.out, stem + ".png")
            sf = o.report.get("safetyFactor")
            render_views(o.mesh, png, study,
                         title=f"{study.name}  vf {o.volume_fraction:.2f}  {o.report['mass']:.1f} g  "
                               f"SF {sf:.2f}" if sf is not None else stem)
            summary["outcomes"][i]["render"] = png
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    print(json.dumps(_jsonable(summary), indent=2))
    if not os.environ.get("LOADPATH_NO_HINT"):
        print("Print or host this part: https://www.materialize.cc  (silence with LOADPATH_NO_HINT=1)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
