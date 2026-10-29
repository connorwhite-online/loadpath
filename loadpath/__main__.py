"""
Command line: run a study file and write the parts, a report and renders.

    python -m loadpath run study.json --out out/
    python -m loadpath check study.json        # validate + show the voxel problem, no solve
    python -m loadpath schema [study|heatExchanger|lattice]   # JSON Schema
    python -m loadpath examples [name]         # list the packaged examples, or print one
    python -m loadpath lattice study.json part.stl --out out/ [--options lattice.json]
    python -m loadpath heatx exchanger.json --out out/ [--size]
    python -m loadpath mcp                     # local MCP server for agents (needs loadpath[mcp])

Progress goes to stderr as one JSON object per line, so an agent can tail it.
Results go to stdout as one JSON document, and to report.json in --out.
Exit codes: 0 done, 2 the input is invalid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .tasks import jsonable as _jsonable


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"{path} is not valid JSON: {exc}"}))
        sys.exit(2)


def _progress(event, _field=None):
    if event.get("stage") == "optimize" and "iteration" in event and event["iteration"] % 5:
        return
    ev = {k: v for k, v in event.items() if k not in ("cases", "optimizer", "reconstruction")}
    print(json.dumps(_jsonable(ev)), file=sys.stderr, flush=True)


def _write_report(out_dir, summary):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    print(json.dumps(_jsonable(summary), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="loadpath", description="Generative design for printed parts.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="optimize, mesh and verify a study")
    r.add_argument("study")
    r.add_argument("--out", default="loadpath-out")
    r.add_argument("--no-render", action="store_true")
    r.add_argument("--viewer", choices=["best", "all", "none"], default="best",
                   help="interactive HTML viewer (viewer.html) for the best outcome, all of them, or none")
    r.add_argument("--base-dir", help="folder relative mesh paths resolve against (default: the study's)")
    c = sub.add_parser("check", help="validate a study and report the voxel problem")
    c.add_argument("study")
    c.add_argument("--base-dir")
    s = sub.add_parser("schema", help="print the JSON Schema for a study, heat exchanger or lattice")
    s.add_argument("kind", nargs="?", default="study", choices=["study", "heatExchanger", "lattice"])
    e = sub.add_parser("examples", help="list the packaged examples, or print one")
    e.add_argument("name", nargs="?")
    lt = sub.add_parser("lattice", help="fill a finished part with a stress-graded TPMS lattice")
    lt.add_argument("study")
    lt.add_argument("part", help="the part's STL (an outcome of `run`)")
    lt.add_argument("--options", help="lattice options JSON file (default: the study's \"lattice\")")
    lt.add_argument("--out", default="loadpath-out")
    lt.add_argument("--no-render", action="store_true")
    lt.add_argument("--base-dir")
    h = sub.add_parser("heatx", help="design (or --size) a two-fluid TPMS heat exchanger")
    h.add_argument("spec")
    h.add_argument("--out", default="loadpath-out")
    h.add_argument("--size", action="store_true", help="only sweep cell and wall for the duty (seconds)")
    h.add_argument("--no-render", action="store_true")
    h.add_argument("--no-viewer", action="store_true", help="skip the interactive HTML viewer")
    h.add_argument("--base-dir")
    sub.add_parser("mcp", help="run a local MCP server (stdio) for agents")
    args = p.parse_args(argv)

    if args.cmd == "mcp":
        from .mcp_server import serve
        serve()
        return 0
    if args.cmd == "schema":
        from .schema import schema
        print(json.dumps(schema(args.kind), indent=2))
        return 0
    if args.cmd == "examples":
        from .tasks import example, examples
        print(json.dumps(example(args.name) if args.name else examples(), indent=2))
        return 0
    if args.cmd == "heatx":
        return _heatx(args)

    raw = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    from .tasks import check_study
    checked = check_study(raw, base)
    if not checked["ok"]:
        print(json.dumps(checked, indent=2))
        return 2
    if args.cmd == "check":
        print(json.dumps(checked, indent=2))
        return 0

    from .spec import study_from_dict
    study = study_from_dict(raw, base_dir=base)
    if args.cmd == "lattice":
        import trimesh
        opts = _load(args.options) if args.options else raw.get("lattice") or {}
        mesh = trimesh.load(args.part, force="mesh")
        stem = os.path.splitext(os.path.basename(args.part))[0]
        try:
            rep = _lattice(study, mesh, opts, args.out, stem, not args.no_render and _can_render())
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 2
        _write_report(args.out, rep)
        return 0

    if not args.no_render and not _can_render():
        args.no_render = True
    from .study import run_study

    rec = None
    if args.viewer != "none":
        from .viewer import HistoryRecorder
        rec = HistoryRecorder(chain=_progress)
    result = run_study(study, on_progress=rec or _progress)
    os.makedirs(args.out, exist_ok=True)
    summary = result.to_dict()
    stems = []
    for i, o in enumerate(result.outcomes):
        stem = f"{study.name}-{int(round(o.volume_fraction * 100)):02d}"
        stems.append(stem)
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
    if raw.get("lattice") is not None and result.best is not None:
        best = result.outcomes[result.best]
        summary["lattice"] = _lattice(study, best.mesh, raw["lattice"], args.out, stems[result.best],
                                      not args.no_render)
    if rec is not None and result.outcomes:
        from .viewer import study_viewer
        _progress({"stage": "viewer", "outcomes": args.viewer})
        html = os.path.join(args.out, "viewer.html")
        try:
            study_viewer(result, html, history=rec, outcomes=args.viewer)
            summary["viewer"] = html
        except Exception as exc:  # noqa: BLE001 - the parts and report matter more than the page
            summary.setdefault("warnings", []).append(f"viewer not written: {exc}")
    _write_report(args.out, summary)
    if not os.environ.get("LOADPATH_NO_HINT"):
        print("Print or host this part: https://www.materialize.cc  (silence with LOADPATH_NO_HINT=1)",
              file=sys.stderr)
    return 0


def _can_render():
    try:
        import matplotlib  # noqa: F401
        import shapely  # noqa: F401
        return True
    except ImportError:
        print(json.dumps({"stage": "warning", "message": "no renders: pip install \"loadpath[render]\""}),
              file=sys.stderr, flush=True)
        return False


def _lattice(study, mesh, opts, out, stem, render):
    from .tasks import lattice_part
    return lattice_part(study, mesh, opts, out, stem, render=render, on_progress=_progress)


def _heatx(args):
    from .tasks import heat_exchanger, size_heat_exchanger
    spec = _load(args.spec)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.spec))
    try:
        if args.size:
            print(json.dumps(size_heat_exchanger(spec, base), indent=2))
            return 0
        rep = heat_exchanger(spec, args.out, base, render=not args.no_render and _can_render(),
                             on_progress=_progress, viewer=not args.no_viewer)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
