"""
Command line: run a study file and write the parts, a report and renders.

    python -m loadpath run study.json --out out/
    python -m loadpath check study.json        # validate + show the voxel problem, no solve
    python -m loadpath verify study.json part.stl   # check a finished part (or a run's folder) against a study
    python -m loadpath schema [study|heatExchanger|lattice]   # JSON Schema
    python -m loadpath examples [name]         # list the packaged examples, or print one
    python -m loadpath lattice study.json part.stl --out out/ [--options lattice.json]
    python -m loadpath heatx exchanger.json --out out/ [--size]
    python -m loadpath field part.json --out out/ [--pitch 0.3]   # a field graph (nTop-style implicit part)
    python -m loadpath tune part.json --out out/ [--evaluations 20]   # its `tune` knobs vs an FE check
    python -m loadpath edit part.json [--html editor.html]   # edit a field graph in the browser, live preview
    python -m loadpath mcp                     # local MCP server for agents (needs loadpath[mcp])
    python -m loadpath orient part.stl [--study study.json] [--3mf part.3mf]
                                               # rank build orientations, write the best as 3MF
    python -m loadpath calibrate kit --material petg --out calibration/   # coupons to print + predictions
    python -m loadpath calibrate fit measurements.json [--predictions predictions.json] [--name petg-a1mini --save]
    python -m loadpath assembly robot.urdf --link forearm --study forearm.json --region j4-mount --payload 1
                                               # loads + neighbour keep-outs derived from the mechanism

Progress goes to stderr as one JSON object per line, so an agent can tail it.
Results go to stdout as one JSON document, and to report.json in --out.
Exit codes: 0 done, 2 the input is invalid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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


def _calibrate(args):
    from . import calibration as cal
    if args.action == "kit":
        kit = cal.coupon_kit(args.material, args.process, crush=args.crush)
        paths = cal.write_kit(kit, args.out, args.material, args.process)
        print(json.dumps({"ok": True, "files": paths,
                          "coupons": {c.id: {"title": c.title, "test": c.test, "predicted": c.predicted}
                                      for c in kit}}, indent=2, default=float))
        return 0
    if not args.measurements:
        print(json.dumps({"ok": False, "error": "calibrate fit needs the measurements.json"}))
        return 2
    meas = _load(args.measurements)
    pred = _load(args.predictions) if args.predictions else None
    try:
        card = cal.fit(meas, pred, args.name)
    except (ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{card['name']}.json")
    with open(path, "w") as fh:
        json.dump(card, fh, indent=2)
    if args.save:
        os.makedirs(cal.CALIBRATION_DIR, exist_ok=True)
        with open(os.path.join(cal.CALIBRATION_DIR, f"{card['name']}.json"), "w") as fh:
            json.dump(card, fh, indent=2)
    print(json.dumps(card, indent=2))
    return 0


def _import(args):
    from .cadimport import import_part

    try:
        if args.features_only:
            r = import_part(args.part, tolerance=args.tolerance)
            print(json.dumps({**r["summary"], "suggested": r["features"]["suggested"]}, indent=1))
            return 0
        loads = []
        for item in args.load:
            k, _, v = item.partition("=")
            f = [float(x) for x in v.split(",")]
            if len(f) != 3:
                raise ValueError(f"--load {item!r}: write ID=FX,FY,FZ in N")
            loads.append((k, f))
        out = import_part(args.part, args.out, tolerance=args.tolerance, space=args.space, grow=args.grow,
                          material=args.material or "petg", force=args.force, pad=args.pad,
                          safety_factor=args.safety_factor, process=args.process, support=args.support,
                          loads=loads or None, ring=args.ring)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    f = out["features"]
    stem = out["study"]["name"]
    given = bool(loads)
    todo = [] if given else [f"replace the placeholder load ({f['suggested'].get('note', 'none suggested')}): "
                             "--load ID=FX,FY,FZ with an interface id below, one load case per --load"]
    if not args.material:
        todo.append("material is the default (petg, FDM): pass --material/--process for anything else")
    if args.space == "part" and not args.grow:
        todo.append("the design space is the part as drawn, so the optimizer can only remove material from it: "
                    "to lighten within the same envelope, or let it find a new shape, use --space box (or hull)")
    setup = {"setup": {"support": args.support or f["suggested"].get("support"),
                       "loads": [{"interface": k, "force": v} for k, v in loads]}} if given else \
        {"suggested": f["suggested"]}
    print(json.dumps({"ok": True, "study": os.path.join(args.out, f"{stem}.study.json"),
                      "features": os.path.join(args.out, f"{stem}.features.json"),
                      **out["summary"], **setup, "material": args.material or "petg",
                      "todo": todo, **({"healed": f["healed"]} if "healed" in f else {})}, indent=1))
    return 0


def _assembly(args):
    from .assembly import Payload, Robot, derive_study

    robot = Robot.from_urdf(args.urdf)
    study = _load(args.study)
    base = os.path.dirname(os.path.abspath(args.study))
    payload = None
    if args.payload > 0:
        link = args.payload_link
        if link is None:
            # down the chain of revolute joints from the part to its end (the tool)
            link = robot.joints[args.joint or robot.far_joint(args.link)].child
            while True:
                kids = [j for j in robot.order if j.parent == link and j.type in ("revolute", "continuous")]
                if len(kids) != 1:
                    break
                link = kids[0].child
        payload = Payload(args.payload, link, tuple(float(v) for v in args.payload_at.split(",")),
                          radius=args.payload_radius, min_kg=0.0)
    from . import quick
    sweep_kw = {"samples": quick.assembly_samples(args.samples), "ramp": args.ramp, "push": args.push, "contact_max": args.contact_max,
                "balanced": tuple(j for j in args.balanced.split(",") if j), "joint": args.joint}
    os.makedirs(args.out, exist_ok=True)
    derived, report = derive_study(robot, args.link, study, args.region, payload=payload, keep_cases=args.keep_cases,
                                   sweep_space=not args.no_space, k=args.cases, method=args.method,
                                   sweep_kw=sweep_kw, space_kw={"clearance": args.clearance}, out_dir=args.out,
                                   base_dir=base, frame=args.frame, cycles={k: float(v) for k, v in
                                                          (kv.split("=") for kv in args.cycles.split(",") if kv)})
    # the study's own mesh paths are relative to its folder: make them absolute
    for key in ("designSpace", "preserve", "keepOut"):
        for item in derived.get(key, []):
            sh = item.get("shape", item)
            if sh.get("type") == "mesh" and sh.get("path") and not os.path.isabs(sh["path"]) \
                    and not os.path.exists(os.path.join(args.out, sh["path"])):
                sh["path"] = os.path.join(base, sh["path"])
    name = os.path.splitext(os.path.basename(args.study))[0]
    with open(os.path.join(args.out, f"{name}.derived.json"), "w") as fh:
        json.dump(derived, fh, indent=1)
    with open(os.path.join(args.out, "assembly.json"), "w") as fh:
        json.dump(_jsonable(report), fh, indent=1)
    print(json.dumps(_jsonable({k: v for k, v in report.items() if k != "states"}), indent=1))
    return 0


def _verify(args):
    """A finished part (an STL, or a run's folder: its best outcome) checked
    against a study: the same check `run` gives its outcomes."""
    import trimesh

    from . import quick
    from .domain import choose_pitch
    from .spec import study_from_dict
    from .study import _margin
    from .tasks import check_study
    from .verify import verify_part

    raw = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    checked = check_study(raw, base)
    if not checked["ok"]:
        print(json.dumps(checked, indent=2))
        return 2
    part = args.part
    if os.path.isdir(part):
        rep = os.path.join(part, "report.json")
        if not os.path.exists(rep):
            print(json.dumps({"ok": False, "error": f"{part} is a folder with no report.json: name the STL"}))
            return 2
        r = _load(rep)
        if r.get("best") is None:
            print(json.dumps({"ok": False, "error": f"{rep} has no best outcome: name the STL"}))
            return 2
        # report.json names it as seen from where `run` ran: look beside the report
        part = os.path.join(part, os.path.basename(r["outcomes"][r["best"]]["stl"]))
    if not os.path.exists(part):
        print(json.dumps({"ok": False, "error": f"no such part: {part}"}))
        return 2
    study = study_from_dict(raw, base_dir=base)
    mesh = trimesh.load(part, force="mesh")
    pitch = choose_pitch(study)
    rep = verify_part(study, mesh, max_cells=quick.verify_cells(args.cells), min_pitch=0.5 * pitch,
                      contact_band=pitch)
    m = _margin(study, rep)
    summary = {"ok": True, "part": part, "study": args.study, "mass": rep["mass"],
               "safetyFactor": rep.get("safetyFactor"), "margin": m,
               "passes": None if m is None else bool(m >= 1.0 and rep.get("bodies", 1) <= 1),
               "keepOutIntrusion": rep.get("keepOutIntrusion"), "bodies": rep.get("bodies"),
               "cases": {c["name"]: {"safetyFactor": c.get("safetyFactor"), "deflection": c.get("deflection")}
                         for c in rep.get("cases", [])},
               "report": rep}
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "verify.json"), "w") as fh:
            json.dump(_jsonable(summary), fh, indent=2)
    print(json.dumps(_jsonable({k: v for k, v in summary.items() if k != "report"}), indent=2))
    return 0


def _materials(args):
    from .calibration import list_cards
    from .material_library import card_names, describe
    from .materials import MATERIALS, _ALIASES

    if args.name:
        key = args.name.strip().lower()
        out = describe(_ALIASES.get(key, key))
        if key in list_cards():
            out = {"calibrated": key, "note": "a calibrated card of this name overrides the library one",
                   "library": out}
        if out is None:
            print(f"unknown material {args.name!r}; known: {', '.join(card_names())}", file=sys.stderr)
            return 1
        print(json.dumps(out, indent=1))
        return 0
    rows = [{"name": k, "title": MATERIALS[k].note, "E": MATERIALS[k].E, "strength": MATERIALS[k].strength,
             "strengthZ": MATERIALS[k].strength_z, "density": MATERIALS[k].density,
             "processes": list(MATERIALS[k].processes)} for k in card_names()]
    print(json.dumps({"library": rows, "calibrated": list_cards()}, indent=1))
    return 0


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
    r.add_argument("--step", action="store_true",
                   help="also write the best outcome as STEP: keep-outs, preserved shapes and the design space "
                        "exact (planes, cylinders), the organic body faceted (needs loadpath[step])")
    r.add_argument("--recall", action="store_true",
                   help="start from what earlier runs on the same design space found (recall.py; "
                        "or set LOADPATH_RECALL=1)")
    v = sub.add_parser("verify", help="check a finished part against a study: mass, SF per load case, keep-outs")
    v.add_argument("study")
    v.add_argument("part", help="the part's STL, or a `run` output folder (its best outcome)")
    v.add_argument("--out", help="also write verify.json (with the full report) here")
    v.add_argument("--cells", type=int, default=60_000, help="analysis grid cells (default 60000)")
    v.add_argument("--base-dir", help="folder relative mesh paths resolve against (default: the study's)")
    c = sub.add_parser("check", help="validate a study and report the voxel problem")
    c.add_argument("study")
    c.add_argument("--base-dir")
    s = sub.add_parser("schema", help="print the JSON Schema for a study, heat exchanger, lattice or field graph")
    s.add_argument("kind", nargs="?", default="study", choices=["study", "heatExchanger", "lattice", "field"])
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
    h.add_argument("--objective", choices=["effectiveness", "metal"], default="effectiveness",
                   help="--size ranking: best effectiveness, or least metal meeting the limits")
    h.add_argument("--min-effectiveness", type=float, help="--size: drop candidates below this")
    h.add_argument("--splits", help="--size: first fluid's shares of the void to sweep, e.g. 0.5,0.6,0.66")
    h.add_argument("--max-pressure-drop", metavar="FLUID=PA,...",
                   help="--size: drop candidates over these pressure drops, e.g. coolant=2000,water=2000")
    h.add_argument("--min-wall", type=float, default=0.8, help="--size: thinnest wall to try, mm (default 0.8)")
    h.add_argument("--no-render", action="store_true")
    h.add_argument("--no-viewer", action="store_true", help="skip the interactive HTML viewer")
    h.add_argument("--base-dir")
    fg = sub.add_parser("field", help="mesh a field graph: implicit part with field-driven walls, lattices, fillets")
    fg.add_argument("graph")
    fg.add_argument("--out", default="loadpath-out")
    fg.add_argument("--pitch", type=float, help="mesh grid, mm (overrides the graph's mesh.pitch)")
    fg.add_argument("--node", help="mesh this node instead of the graph's output (a copy's nodes: <copy>__<node>)")
    fg.add_argument("--decimate", type=float, metavar="MM", help="thin the triangles where the mesh stays within "
                                                                 "MM of the surface (e.g. 0.02)")
    fg.add_argument("--stream", action="store_true", help="write the STL slab by slab in bounded memory "
                                                          "(huge lattices; skips the checks that need the mesh)")
    fg.add_argument("--no-render", action="store_true")
    fg.add_argument("--step", action="store_true", help="also write a STEP file (exact planes and cylinders; "
                                                         "needs loadpath[step])")
    fg.add_argument("--base-dir")
    tu = sub.add_parser("tune", help="tune a field graph's numbers (its `tune` block) for the lightest part "
                                     "that still meets a study's safety factor and deflection")
    tu.add_argument("graph")
    tu.add_argument("--out", default="loadpath-tune")
    tu.add_argument("--evaluations", type=int, help="analyses to spend (overrides tune.evaluations)")
    tu.add_argument("--base-dir")
    ed = sub.add_parser("edit", help="edit a field graph in the browser: node graph, sliders, live GPU preview, run")
    ed.add_argument("graph")
    ed.add_argument("--html", help="write one self-contained editor page (every bake inside) instead of serving")
    ed.add_argument("--port", type=int, default=8765)
    ed.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this machine only)")
    ed.add_argument("--out", help="where runs go (default: loadpath-edit/ beside the graph)")
    ed.add_argument("--no-open", action="store_true", help="don't open a browser")
    ed.add_argument("--base-dir")
    sub.add_parser("mcp", help="run a local MCP server (stdio) for agents")
    mt = sub.add_parser("materials", help="list the material library, or print one card with its sources")
    mt.add_argument("name", nargs="?")
    o = sub.add_parser("orient", help="rank build orientations of a mesh (support, bed, layer bond)")
    o.add_argument("mesh")
    o.add_argument("--study", help="the part's study: adds the interlayer reading under its loads")
    o.add_argument("--3mf", dest="threemf", help="write the best orientation as a slicer-ready 3MF")
    o.add_argument("--process", default="fdm")
    cal = sub.add_parser("calibrate", help="coupons to print, and a material card fitted from their measurements")
    cal.add_argument("action", choices=["kit", "fit"])
    cal.add_argument("measurements", nargs="?", help="fit: the filled-in measurements.json")
    cal.add_argument("--material", default="petg", help="kit: the starting card")
    cal.add_argument("--process", default="fdm")
    cal.add_argument("--crush", action="store_true", help="kit: add gyroid crush blocks (impact.py's crush law)")
    cal.add_argument("--out", default="calibration", help="kit: folder for the STLs; fit: folder for the card")
    cal.add_argument("--predictions", help="fit: the kit's predictions.json (adds part-test model factors)")
    cal.add_argument("--name", help="fit: the card's name, usable as a study's material once saved")
    cal.add_argument("--save", action="store_true", help="fit: also save the card to loadpath/calibrations/")
    sub.add_parser("guide", help="print the agent guide (SKILL.md: how to turn a request into a study)")
    im = sub.add_parser("import", help="a STEP/STL part as a draft study: bores, bolt patterns and mounting "
                                       "faces found, interfaces preserved, a support and load suggested")
    im.add_argument("part", help="STEP (.step/.stp, needs loadpath[step]), STL, 3MF or OBJ, in mm")
    im.add_argument("--out", default="loadpath-import", help="folder for the study, its STLs and features.json")
    im.add_argument("--space", choices=["part", "hull", "box"], default="part",
                    help="design space: the part as drawn, its convex hull, or its bounding box")
    im.add_argument("--grow", type=float, default=0.0, help="mm to grow the design space by")
    im.add_argument("--material", help="material card (`loadpath materials`); without it the draft says petg")
    im.add_argument("--process", help="fdm, sls, mjf, sla, dlp, lpbf, volumetric")
    im.add_argument("--support", help="interface id that holds the part (default: the suggested one)")
    im.add_argument("--load", action="append", default=[], metavar="ID=FX,FY,FZ",
                    help="N on an interface, e.g. bore-0=0,0,-250; repeat it for more load cases, one per --load; "
                    "replaces the placeholder")
    im.add_argument("--ring", type=float, help="mm of wall kept round every bore, as exact tubes in the study "
                    "(default: 2 mm + a tenth of the radius, cut to the part)")
    im.add_argument("--force", type=float, default=100.0, help="N: the suggested (placeholder) load")
    im.add_argument("--pad", type=float, default=2.0, help="mm of material kept under each bolted face")
    im.add_argument("--safety-factor", type=float, default=2.0)
    im.add_argument("--tolerance", type=float, default=0.05, help="mm chord error tessellating a STEP")
    im.add_argument("--features-only", action="store_true", help="print the features, write no study")
    asm = sub.add_parser("assembly", help="derive a part's load cases and neighbour keep-outs from a URDF")
    asm.add_argument("urdf")
    asm.add_argument("--link", required=True, help="the URDF link the study designs")
    asm.add_argument("--study", required=True, help="the part's study: interfaces, material, goal, design space")
    asm.add_argument("--region", required=True, help="the study region the far joint loads (a socket, a mount)")
    asm.add_argument("--joint", help="the far joint, if the link has more than one moving child joint")
    asm.add_argument("--frame", choices=["world", "link"], default="world",
                     help="the study's frame: world (the URDF's world at the home pose, mm) or link (the "
                     "designed link's own frame, mm, origin at its joint)")
    asm.add_argument("--payload", type=float, default=0.0, help="kg at the end of the chain")
    asm.add_argument("--payload-link", help="link that carries the payload (default: the chain's last link)")
    asm.add_argument("--payload-at", default="0,0,0", help="payload position in that link's frame, m: x,y,z")
    asm.add_argument("--payload-radius", type=float, default=0.0, help="m: payload position scatters this far")
    asm.add_argument("--samples", type=int, default=1500, help="random poses (on top of the limit corners)")
    asm.add_argument("--ramp", type=float, default=0.2, help="s to full speed (before the effort limit)")
    asm.add_argument("--push", type=float, default=20.0, help="N: contact at the payload, any direction")
    asm.add_argument("--contact-max", type=float, help="N: hardest an obstacle pushes back on a stalled joint "
                     "(default: --push)")
    asm.add_argument("--balanced", default="", help="joints a counterbalance holds against gravity: j2,...")
    asm.add_argument("--cases", type=int, default=8, help="most governing cases to keep")
    asm.add_argument("--method", choices=["stress", "hull"], default="stress")
    asm.add_argument("--keep-cases", action="store_true", help="keep the study's own load cases too")
    asm.add_argument("--no-space", action="store_true", help="loads only: no swept keep-outs")
    asm.add_argument("--clearance", type=float, default=1.0, help="mm round each swept neighbour")
    asm.add_argument("--cycles", default="", help="life cycles by sample kind, for fatigue: "
                     "dynamic=1e6,pose=1e6,push=1e4,stall=1e3 (stall:j2=... for one joint)")
    asm.add_argument("--out", default="loadpath-out")
    for sp in (r, lt, h, fg, tu, ed, asm, v):
        sp.add_argument("--quick", action="store_true",
                        help="coarse grid, fewer iterations: a first look in a minute or two (quick.py)")
    args = p.parse_args(argv)
    if getattr(args, "quick", False):
        from . import quick
        quick.enable()

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
    if args.cmd == "materials":
        return _materials(args)
    if args.cmd == "heatx":
        return _heatx(args)
    if args.cmd == "field":
        return _field(args)
    if args.cmd == "edit":
        return _edit(args)
    if args.cmd == "tune":
        return _tune(args)
    if args.cmd == "orient":
        return _orient(args)
    if args.cmd == "calibrate":
        return _calibrate(args)
    if args.cmd == "assembly":
        return _assembly(args)
    if args.cmd == "verify":
        return _verify(args)
    if args.cmd == "import":
        return _import(args)
    if args.cmd == "guide":
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")) as fh:
            print(fh.read())
        return 0

    raw = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    if getattr(args, "recall", False):
        os.environ.setdefault("LOADPATH_RECALL", "1")
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

    if study.multi_material is not None:
        # several materials placed by the optimizer (multimat.py): bodies,
        # an STL per material, a multi-material 3MF and report.json
        from .multimat import run_multimaterial

        mres = run_multimaterial(study, on_progress=lambda ev, *_: _progress(ev, None))
        files = mres.write(args.out)
        print(json.dumps(_jsonable({"ok": True, "mass": mres.report.get("mass"),
                          "materialMass": mres.report.get("materialMass"),
                          "safetyFactor": mres.report.get("safetyFactor"),
                          "interfaceSafetyFactor": mres.report.get("interfaceSafetyFactor"),
                          "requirementsMet": mres.report.get("requirementsMet"), "files": files}),
                         indent=2, default=str))
        return 0
    if not args.no_render and not _can_render():
        args.no_render = True
    from .study import run_study

    # a run can take half an hour: say which process it is, so a caller polling
    # from elsewhere can tell it apart from other runs and see it finish
    os.makedirs(args.out, exist_ok=True)
    status_path = os.path.join(args.out, "status.json")

    def _status(state, **kw):
        with open(status_path, "w") as fh:
            json.dump({"state": state, "pid": os.getpid(), "study": os.path.abspath(args.study), **kw}, fh)

    _status("running", started=time.time())
    # each outcome as the search finds it, kept if a later stage dies
    os.environ.setdefault("LOADPATH_PARTIAL_DIR", os.path.join(args.out, "partial"))
    rec = None
    if args.viewer != "none" and not os.environ.get("LOADPATH_NO_RENDER"):
        from .viewer import HistoryRecorder
        rec = HistoryRecorder(chain=_progress)
    try:
        result = run_study(study, on_progress=rec or _progress)
    except BaseException as exc:
        _status("failed", error=f"{type(exc).__name__}: {exc}",
                partial=os.environ.get("LOADPATH_PARTIAL_DIR"))
        raise
    os.makedirs(args.out, exist_ok=True)
    summary = result.to_dict()
    stems = []
    for i, o in enumerate(result.outcomes):
        stem = f"{study.name}-{int(round(o.volume_fraction * 100)):02d}"
        if stem in stems:
            # two outcomes within half a percent (a search bisecting): one
            # file each, or the second overwrites the first's STL
            stem = f"{study.name}-{o.volume_fraction * 100:.1f}".replace(".", "p")
            k = 2
            while stem in stems:
                stem, k = f"{study.name}-{int(round(o.volume_fraction * 100)):02d}-{k}", k + 1
        stems.append(stem)
        stl = os.path.join(args.out, stem + ".stl")
        o.mesh.export(stl)
        summary["outcomes"][i]["stl"] = stl
        if getattr(o, "compensated", None) is not None:
            # pre-distorted against the simulated build: print this one
            cstl = os.path.join(args.out, stem + "-compensated.stl")
            o.compensated.export(cstl)
            summary["outcomes"][i]["compensatedStl"] = cstl
        rs = result.study      # with an "auto" build direction resolved
        build = rs.print_direction if rs.print_direction is not None else \
            (rs.process.build if rs.process is not None and rs.process.build != "auto" else None)
        if build is not None:
            # stood the way it was designed to print, for the slicer
            from .threemf import write_3mf
            summary["outcomes"][i]["3mf"] = write_3mf(os.path.join(args.out, stem + ".3mf"), o.mesh, build,
                                                      name=stem, material=rs.material, process=rs.process)
            if o.grading is not None:
                # the slicer grades it: infill and perimeters per region
                summary["outcomes"][i]["graded3mf"] = write_3mf(
                    os.path.join(args.out, stem + "-graded.3mf"), o.mesh, build, name=stem, material=rs.material,
                    process=rs.process, object_settings=o.grading["objectSettings"],
                    modifiers=o.grading["modifiers"])
        if not args.no_render:
            from .render import render_views
            png = os.path.join(args.out, stem + ".png")
            sf = o.report.get("safetyFactor")
            render_views(o.mesh, png, study,
                         title=f"{study.name}  vf {o.volume_fraction:.2f}  {o.report['mass']:.1f} g  "
                               f"SF {sf:.2f}" if sf is not None else stem)
            summary["outcomes"][i]["render"] = png
    if args.step and result.outcomes:
        from .step import export_step

        bi = result.best if result.best is not None else len(result.outcomes) - 1
        try:
            st = export_step(result.outcomes[bi].mesh, os.path.join(args.out, stems[bi] + ".step"), name=stems[bi])
            st.pop("shape", None)
            summary["outcomes"][bi]["step"] = st
        except Exception as exc:  # noqa: BLE001 - the STL is the deliverable; say why STEP is missing
            summary.setdefault("warnings", []).append(f"STEP not written: {type(exc).__name__}: {exc}")
    if result.best is not None:
        # a fixed name for the best outcome, for the next command in a script
        best_stl = os.path.join(args.out, "best.stl")
        result.outcomes[result.best].mesh.export(best_stl)
        summary["bestStl"] = best_stl
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
    from .delivery import delivery
    summary["deliver"] = delivery(summary, raw)
    _write_report(args.out, summary)
    _status("done", report=os.path.join(os.path.abspath(args.out), "report.json"))
    if not os.environ.get("LOADPATH_NO_HINT"):
        print("Print or host this part: https://www.materialize.cc  (silence with LOADPATH_NO_HINT=1)",
              file=sys.stderr)
    return 0


def _can_render():
    if os.environ.get("LOADPATH_NO_RENDER"):
        return False
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
            splits = [float(x) for x in args.splits.split(",")] if args.splits else None
            max_dp = None
            if args.max_pressure_drop:
                try:
                    pairs = (kv.split("=") for kv in args.max_pressure_drop.split(","))
                    max_dp = {k.strip(): float(v) for k, v in pairs}
                except ValueError:
                    raise ValueError(f"--max-pressure-drop takes FLUID=PA pairs, e.g. water=2000,oil=5000; "
                                     f"got {args.max_pressure_drop!r}") from None
                fluids = list(spec.get("fluids") or ["A", "B"])
                unknown = sorted(set(max_dp) - set(fluids))
                if unknown:
                    raise ValueError(f"--max-pressure-drop names {unknown}; the fluids are {fluids}")
            print(json.dumps(size_heat_exchanger(spec, base, max_pressure_drop=max_dp, min_wall=args.min_wall,
                                                 objective=args.objective, min_effectiveness=args.min_effectiveness,
                                                 splits=splits), indent=2))
            return 0
        rep = heat_exchanger(spec, args.out, base, render=not args.no_render and _can_render(),
                             on_progress=_progress, viewer=not args.no_viewer and not os.environ.get("LOADPATH_NO_RENDER"))
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, rep)
    return 0


def _field(args):
    from .fields import run_field
    from .schema import format_issues, lint

    doc = _load(args.graph)
    issues = lint(doc, "field")
    if any(i["level"] == "error" for i in issues):
        print(json.dumps({"ok": False, "issues": issues, "error": format_issues(issues)}, indent=2))
        return 2
    if args.pitch:
        doc = {**doc, "mesh": {**(doc.get("mesh") or {}), "pitch": args.pitch}}
    if args.step and not doc.get("step"):
        doc = {**doc, "step": True}
    if args.stream:
        doc = {**doc, "mesh": {**(doc.get("mesh") or {}), "stream": True}}
    if args.decimate:
        doc = {**doc, "mesh": {**(doc.get("mesh") or {}), "decimate": args.decimate}}
    base = args.base_dir or os.path.dirname(os.path.abspath(args.graph))
    try:
        _, rep = run_field(doc, args.out, base, progress=_progress, render=not args.no_render and _can_render(),
                           node=args.node)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, {"ok": True, **rep, **({"warnings": issues} if issues else {})})
    return 0


def _tune(args):
    from .schema import format_issues, lint
    from .tune import tune_field

    doc = _load(args.graph)
    issues = lint(doc, "field")
    if any(i["level"] == "error" for i in issues):
        print(json.dumps({"ok": False, "issues": issues, "error": format_issues(issues)}, indent=2))
        return 2
    base = args.base_dir or os.path.dirname(os.path.abspath(args.graph))
    try:
        out = tune_field(doc, base, args.out, progress=_progress, evaluations=args.evaluations)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    summary = {"ok": out["best"] is not None, "best": out["best"], "start": out["start"],
               "limits": out["limits"], "evaluations": len(out["history"]),
               "tuned": os.path.join(args.out, "tuned.json")}
    if out["best"] is not None:
        # tuned.json keeps the graph's relative paths (its study): they resolve from the graph's folder
        summary["next"] = f"loadpath field {summary['tuned']} --base-dir {os.path.relpath(base)}"
    if out["best"] is None:
        summary["error"] = "no candidate met the limits: widen the params' ranges or ease the limits"
    _write_report(args.out, summary)
    return 0 if out["best"] is not None else 1


def _edit(args):
    from .schema import lint

    doc = _load(args.graph)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.graph))
    if args.html:
        from .editor import field_editor

        issues = lint(doc, "field")
        if any(i["level"] == "error" for i in issues):
            print(json.dumps({"ok": False, "issues": issues}, indent=2))
            return 2
        field_editor(doc, args.html, base, progress=_progress)
        print(json.dumps({"ok": True, "html": args.html}))
        return 0
    from .editor import serve_editor

    serve_editor(args.graph, host=args.host, port=args.port, out_dir=args.out, base_dir=base,
                 open_browser=not args.no_open)
    return 0


def _orient(args):
    import trimesh

    from .orient import rank_orientations
    from .spec import study_from_dict

    mesh = trimesh.load(args.mesh, force="mesh")
    study = None
    if args.study:
        with open(args.study) as fh:
            study = study_from_dict(json.load(fh), base_dir=os.path.dirname(os.path.abspath(args.study)))
    rows = rank_orientations(mesh, study, process=args.process)
    out = [{k: v for k, v in r.items() if k != "build"} for r in rows]
    for r in out:
        r["print"] = {k: v for k, v in r["print"].items() if k != "bridgeList"}
    if args.threemf:
        from .threemf import write_3mf
        write_3mf(args.threemf, mesh, rows[0]["build"], name=os.path.splitext(os.path.basename(args.mesh))[0],
                  material=study.material if study else None, process=study.process if study else args.process)
    print(json.dumps(_jsonable({"best": out[0]["direction"], "orientations": out,
                                **({"3mf": args.threemf} if args.threemf else {})}), indent=2))
    return 0


def _main():
    # `loadpath check study.json | head` closes the pipe early: exit quietly
    try:
        return main()
    except BrokenPipeError:
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(_main())
