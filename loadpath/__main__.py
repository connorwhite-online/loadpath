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
    python -m loadpath buildsim part.stl --material ti64 [--compensate 2] --out sim/
                                               # simulate the print: warp, residual stress, recoater lift

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
    cases = event.get("cases") or []
    if event.get("stage") == "outcome" and cases:
        # the governing case and the worst deflection, so a failed step says why
        sf = [c for c in cases if c.get("safetyFactor") is not None]
        if sf:
            g = min(sf, key=lambda c: c["safetyFactor"])
            ev["governingCase"] = {"name": g.get("name"), "safetyFactor": g["safetyFactor"]}
        dfl = [c for c in cases if c.get("deflection") is not None]
        if dfl:
            w = max(dfl, key=lambda c: c["deflection"])
            ev["maxDeflection"] = {"name": w.get("name"), "deflection": w["deflection"]}
    print(json.dumps(_jsonable(ev)), file=sys.stderr, flush=True)


def _write_report(out_dir, summary):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    print(json.dumps(_jsonable(summary), indent=2))


def _calibrate(args):
    from . import calibration as cal
    if args.action == "kit":
        kit = cal.coupon_kit(args.material, args.process, crush=args.crush, hole_gauge=args.hole_gauge)
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


def _step(args):
    import trimesh

    from . import step as stp

    m = trimesh.load(args.part, force="mesh")
    out = args.out or os.path.splitext(args.part)[0] + ".step"
    name = os.path.splitext(os.path.basename(args.part))[0]
    rep = stp.export_step(m, out, faces=args.faces or stp.DEFAULT_FACES, name=name, surface=args.surface,
                          tolerance=args.tolerance)
    rep.pop("shape", None)
    rep["volumeError"] = round(rep["volume"] / rep["meshVolume"] - 1, 5) if rep.get("meshVolume") else None
    print(json.dumps(rep, indent=1))
    return 0 if rep["valid"] and rep["solids"] == 1 else 1


def _slices(args):
    import trimesh

    from .layerfiles import export_cli

    m = trimesh.load(args.part, force="mesh")
    if not m.is_watertight:
        print(json.dumps({"error": f"{args.part} is not closed: a powder-bed layer needs closed contours "
                                   "(run `loadpath exterior` on it first)"}))
        return 1
    axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[args.axis]
    out = args.out or os.path.splitext(args.part)[0] + ".cli"
    name = os.path.splitext(os.path.basename(args.part))[0]
    rep = export_cli(m, out, layer=args.layer, hatch=None if args.hatch == 0 else args.hatch, axis=axis,
                     label=name)
    rep["volumeError"] = round(rep["volume"] / rep["meshVolume"] - 1, 5) if rep["meshVolume"] else None
    print(json.dumps(rep, indent=1))
    return 0 if not rep["openLayers"] else 1


def _nest(args):
    import trimesh

    from .nesting import BED_PROCESSES, nest

    parts = []
    for item in args.parts:
        path, _, n = item.rpartition(":") if ":" in item and item.rsplit(":", 1)[1].isdigit() else (item, "", "1")
        parts.append((trimesh.load(path, force="mesh"), int(n)))
    vol = args.volume
    if vol is None:
        from . import dfm_reference

        lim = dfm_reference.limits(args.process, args.material) if args.material else None
        if not lim or "material" not in lim:
            print(json.dumps({"error": "give --volume X Y Z (mm), or --process with --material to use the "
                                       "largest build volume Materialize lists for it"}))
            return 1
        vol = lim["material"]["buildVolume"]
    mode = args.mode or ("bed" if (args.process or "fdm") in BED_PROCESSES else "volume")
    res = nest(parts, vol, gap=args.gap, mode=mode, rotations=args.rotations, pitch=args.pitch)
    rep = res.report()
    out = args.out or "nest.stl"
    res.combined().export(out)
    rep["out"] = out
    print(json.dumps(rep if args.full else {k: v for k, v in rep.items() if k != "placements"}, indent=1))
    return 0 if not rep["unplaced"] else 2


def _external_flow(args):
    import numpy as np
    import trimesh

    from .aero import external_flow

    mesh = trimesh.load(args.part, force="mesh")
    d = args.direction.strip().lower()
    axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    if d.lstrip("+-") in axes:
        vec = np.array(axes[d.lstrip("+-")], float) * (-1 if d.startswith("-") else 1)
    else:
        vec = np.array([float(v) for v in d.split(",")])
    try:
        rep = external_flow(mesh, args.speed, vec, medium=args.medium, cells=args.cells, re_max=args.re_max)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    rep.pop("_surface", None)
    if args.study:
        doc = _load(args.study)
        doc.setdefault("loadCases", []).append(rep["load_case"])
        out = args.out or os.path.splitext(args.study)[0] + "-flow.json"
        with open(out, "w") as fh:
            json.dump(doc, fh, indent=1)
        rep["study"] = out
    print(json.dumps(rep, indent=1))
    return 0


def _fe_export(args):
    import trimesh

    from .fe_export import export_fe
    from .spec import study_from_dict

    raw = _load(args.study)
    base = os.path.dirname(os.path.abspath(args.study))
    try:
        study = study_from_dict(raw, base_dir=base)
        mesh = trimesh.load(args.part, force="mesh")
        out = args.out or os.path.splitext(args.part)[0] + (".inp" if args.format == "abaqus" else ".bdf")
        info = export_fe(study, mesh, out, fmt=args.format, size=args.size)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(json.dumps(info, indent=1))
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


def _making_summary(m: dict) -> dict:
    """The headline numbers of a manufacturing report (manufacture.report)."""
    out = {}
    a = m.get("access")
    if a:
        out["undercutVolume"] = a["undercutVolume"] - a["undercutInKeepOuts"]
    d = m.get("draft")
    if d:
        out["draft"] = {k: d.get(k) for k in ("required", "underDraftedArea", "underDraftedShare", "p5Draft",
                                              "parting")}
    mc = m.get("machining")
    if mc:
        out["machining"] = {"minutes": round(mc["minutes"]["total"], 1), "cost": round(mc["cost"]["total"], 2),
                            "setups": mc["setups"], "toolUnreachableVolume": mc["toolUnreachableVolume"],
                            "maxStickoutToDiameter": round(mc["maxStickoutToDiameter"], 2)}
    if m.get("maxMember"):
        out["maxMember"] = m["maxMember"]
    return out


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
    # the default grid matches the run's own check (no finer than half the
    # design cell); --cells asks for a grid, so it gets one: it used to be
    # floored to that same pitch, and 120k cells read exactly as 60k did
    asked = args.cells is not None
    rep = verify_part(study, mesh, max_cells=quick.verify_cells(args.cells if asked else 60_000),
                      min_pitch=None if asked else 0.5 * pitch, contact_band=pitch)
    if study.access is not None or study.pattern is not None or study.max_member is not None:
        # draft, undercuts, tool reach and the machining estimate, on the
        # study's own design grid (manufacture.report)
        from .domain import build_domain
        from .manufacture import report as manufacturing_report

        making = manufacturing_report(study, build_domain(study, pitch=pitch), mesh)
        if making:
            rep["manufacturing"] = making
            rep.setdefault("warnings", []).extend(making.get("warnings", []))
    m = _margin(study, rep)
    summary = {"ok": True, "part": part, "study": args.study, "mass": rep["mass"],
               "safetyFactor": rep.get("safetyFactor"), "margin": m,
               "passes": None if m is None else bool(m >= 1.0 and rep.get("bodies", 1) <= 1),
               "safetyFactorMethod": "voxel", "analysisPitch": rep.get("analysisPitch"),
               "keepOutIntrusion": rep.get("keepOutIntrusion"), "preserveMissing": rep.get("preserveMissing"),
               "bodies": rep.get("bodies"),
               **({"manufacturing": _making_summary(rep["manufacturing"])} if rep.get("manufacturing") else {}),
               **({"durabilitySafetyFactor": rep["durability"].get("safetyFactor")}
                  if isinstance(rep.get("durability"), dict) else {}),
               "warnings": list(rep.get("warnings") or []),
               "cases": {c["name"]: {"safetyFactor": c.get("safetyFactor"), "deflection": c.get("deflection"),
                                     "peakStress": c.get("peakStress"), "peakAt": c.get("peakAt")}
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


def run_status(out: str) -> dict:
    """out/status.json, with `running` checked against the process: a run
    the OOM killer or a closed shell took down (SIGKILL) never gets to write
    its own end, and its status kept reading `running` with a dead pid."""
    path = os.path.join(out, "status.json")
    if not os.path.exists(path):
        return {"state": "none", "error": f"no {path}: the run has not started, or --out was another folder"}
    with open(path) as fh:
        st = json.load(fh)
    part = os.path.join(out, "partial", "outcomes.jsonl")
    if os.path.exists(part):
        rows = []
        with open(part) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        st["partialOutcomes"] = len(rows)
        ok = [o for o in rows if o.get("passes") and o.get("mass") is not None]
        if ok and st.get("state") != "done":
            # the lightest verified passing part so far: what to hand over if
            # the run has to stop (it may still get lighter, or fail its fine
            # re-check: `loadpath verify` it before delivering)
            b = min(ok, key=lambda o: o["mass"])
            st["bestSoFar"] = {"stl": os.path.join(os.path.abspath(out), "partial", b.get("stl", "")),
                               "mass": b["mass"], "safetyFactor": b.get("safetyFactor"),
                               "volumeFraction": b.get("volumeFraction")}
    if st.get("state") == "running" and st.get("pid"):
        alive = True
        try:
            os.kill(int(st["pid"]), 0)
            with open(f"/proc/{int(st['pid'])}/cmdline", "rb") as fh:
                alive = b"loadpath" in fh.read()
        except ProcessLookupError:
            alive = False
        except (OSError, ValueError):
            pass
        if not alive:
            st["state"] = "died"
            st["hint"] = ("the process is gone without writing an end: killed (out of memory, or by the shell "
                          "that launched it). Outcomes found so far are in out/partial/. Relaunch detached "
                          "(setsid nohup ... > run.log 2>&1 < /dev/null &); if it was memory, set "
                          "LOADPATH_CONFIRM_CELLS lower or resolution.maxElements")
    if st.get("started") and st.get("state") == "running":
        st["elapsedSeconds"] = round(time.time() - float(st["started"]))
    return st


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
    r.add_argument("--3mf-coordinates", dest="mm_coordinates", choices=["bed", "study"], default="bed",
                   help="multiMaterial studies: the 3MF placed for printing (bed: build direction +z, on the bed "
                        "at 128,128; default) or in the study's coordinates (study), as the STLs are")
    r.add_argument("--time-limit", type=float, metavar="MINUTES",
                   help="stop searching after MINUTES and deliver the best outcome verified by then (or set "
                        "LOADPATH_TIME_LIMIT): a run's length depends on how the search goes, an agent's "
                        "time does not")
    r.add_argument("--recall", action="store_true",
                   help="start from what earlier runs on the same design space found (recall.py; "
                        "or set LOADPATH_RECALL=1)")
    stp = sub.add_parser("status", help="is a background run still going? reads out/status.json and checks its "
                                        "process is alive (a killed run cannot update the file itself)")
    stp.add_argument("out", nargs="?", default="out", help="the run's --out folder (default out)")
    stp.add_argument("--wait", type=float, metavar="SECONDS",
                     help="block until the run ends or SECONDS pass, then print its status: one call that fits "
                          "under an agent's tool timeout (say --wait 540), repeated until the state is not "
                          "running. Exit code 0 done, 1 failed/killed/died, 3 still running")
    v = sub.add_parser("verify", help="check a finished part against a study: mass, SF per load case, keep-outs")
    v.add_argument("study")
    v.add_argument("part", help="the part's STL, or a `run` output folder (its best outcome)")
    v.add_argument("--out", help="also write verify.json (with the full report) here")
    v.add_argument("--cells", type=int, default=None,
                   help="analysis grid cells (default: 60000, no finer than half the study's design cell, as "
                        "`run` checks; given, the grid is about this many cells whatever the design cell)")
    v.add_argument("--base-dir", help="folder relative mesh paths resolve against (default: the study's)")
    c = sub.add_parser("check", help="validate a study and report the voxel problem (or a heat exchanger "
                                     "spec: its design grid and memory; a lattice or field graph: lint)")
    c.add_argument("study")
    c.add_argument("--base-dir")
    s = sub.add_parser("schema", help="print the JSON Schema for a study, heat exchanger, lattice or field graph")
    s.add_argument("kind", nargs="?", default="study", choices=["study", "heatExchanger", "lattice", "field",
                                                                "handoff", "split", "enclosure"])
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
    h.add_argument("--out", help="output folder (default loadpath-out; with --size: also write sizing.json "
                                 "and best.json there)")
    h.add_argument("--size", action="store_true", help="only sweep cell and wall for the duty (seconds)")
    h.add_argument("--objective", choices=["effectiveness", "mass", "pressure", "metal"],
                   help="--size ranking: best effectiveness (default), least mass meeting the limits (metal: the "
                        "same), or least pressure drop against the limits; overrides the spec's sizing.objective")
    h.add_argument("--min-effectiveness", type=float, help="--size: drop candidates below this")
    h.add_argument("--splits", help="--size: first fluid's shares of the void to sweep, e.g. 0.5,0.6,0.66")
    h.add_argument("--max-pressure-drop", metavar="FLUID=PA,...",
                   help="--size: drop candidates over these pressure drops, e.g. coolant=2000,water=2000")
    h.add_argument("--min-wall", type=float,
                   help="--size: thinnest wall to try, mm (default the process's minimum: fdm 0.8, lpbf 0.5)")
    h.add_argument("--axes", choices=["xy", "yx"],
                   help="--size with an envelope: the first fluid runs along axes[0] (default: both tried; the "
                        "spec's axes pins it too)")
    h.add_argument("--max-points", type=int,
                   help="design grid cap (default 16000000, about 6.6 GB at its peak); --size keeps only "
                        "designs that mesh within it")
    h.add_argument("--force", action="store_true",
                   help="design even when the estimated peak memory is over what is free")
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
    fg.add_argument("--lattice-3mf", action="store_true",
                    help="also write <name>-lattice.3mf with the lattice as a lattice (implicit TPMS level set or "
                         "beam lattice; needs loadpath[3mf])")
    fg.add_argument("--base-dir")
    tu = sub.add_parser("tune", help="tune a field graph's numbers (its `tune` block) for the lightest part "
                                     "that still meets a study's safety factor and deflection")
    tu.add_argument("graph")
    tu.add_argument("--out", default="loadpath-tune")
    tu.add_argument("--evaluations", type=int, help="analyses to spend (overrides tune.evaluations)")
    tu.add_argument("--base-dir")
    lo_ = sub.add_parser("lattice-orient", help="turn a field graph's lattice cells (their `orient`) to line "
                                                "their stiff directions up with the load: best safety factor "
                                                "or stiffness at the same mass")
    lo_.add_argument("graph")
    lo_.add_argument("--study", help="study JSON or example:<name> (default: the graph's tune.study)")
    lo_.add_argument("--objective", choices=["safetyFactor", "stiffness"], default="safetyFactor")
    lo_.add_argument("--cells", type=int, default=15000, help="homogenized analysis cells per evaluation")
    lo_.add_argument("--refine", type=int, default=16, help="Nelder-Mead evaluations after the screen")
    lo_.add_argument("--zones", type=int, default=1, help="split the lattice into this many zones by stress "
                                                          "direction and turn each its own way (orient.zones)")
    lo_.add_argument("--seam", type=float, default=1.0, help="zones: mm solid plate between zones (orient.seam), "
                                                             "0 for cross-faded zones (unconfirmed, read too stiff)")
    lo_.add_argument("--confirm", choices=["auto", "resolved", "none"], default="auto",
                     help="a lattice cut open at bare faces: auto keeps its cells as laid (the homogenized check "
                          "misreads turned against square-cut cells there), resolved checks the turn at full "
                          "resolution (minutes), none trusts the homogenized reading")
    lo_.add_argument("--resolved-pitch", type=float, help="mm voxels for --confirm resolved (default wall/6)")
    lo_.add_argument("--out", default="loadpath-lattice-orient")
    lo_.add_argument("--base-dir")
    rc = sub.add_parser("resolved-check", help="full-resolution voxel FE of a field graph's part, lattice "
                                               "struts and sheets resolved (the arbiter for homogenized readings)")
    rc.add_argument("graph")
    rc.add_argument("--study", help="study JSON or example:<name> (default: the graph's tune.study)")
    rc.add_argument("--pitch", type=float, help="mm (default: the lattice wall / 6)")
    rc.add_argument("--band", type=float, help="contact band, mm (default one lattice cell)")
    rc.add_argument("--max-cells", type=int, default=4_000_000)
    rc.add_argument("--base-dir")
    lg = sub.add_parser("lattice-grade", help="grade a field graph's lattice wall to the stress (fully stressed, "
                                              "on the anisotropic homogenized check), optionally turning its cells")
    lg.add_argument("graph")
    lg.add_argument("--study", help="study JSON or example:<name> (default: the graph's tune.study)")
    lg.add_argument("--target-sf", type=float, help="safety factor to size to (default: the study's goal)")
    lg.add_argument("--orient", help="'uniform' (one direction) or N (zones) to turn the cells too")
    lg.add_argument("--seam", type=float, default=1.0, help="zones: mm solid plate between zones (0: none)")
    lg.add_argument("--confirm", choices=["auto", "none"], default="auto",
                    help="auto: a lattice cut open at bare faces is graded but not turned (unconfirmed there)")
    lg.add_argument("--wall-min", type=float, help="mm (default: the process's minimum feature, at least 0.6)")
    lg.add_argument("--wall-max", type=float, help="mm (default: density 0.6)")
    lg.add_argument("--cells", type=int, default=15000)
    lg.add_argument("--out", default="loadpath-lattice-grade", help="folder for graded.json and its density "
                                                                    "grid (inside the graph's folder)")
    lg.add_argument("--base-dir")
    co = sub.add_parser("cores", help="tune each core option (hollow, ribbed, lattice...) of a part and pick the "
                                      "lightest that meets its loads and can be built by each process")
    co.add_argument("spec")
    co.add_argument("--out", default="loadpath-cores")
    co.add_argument("--process", help="comma-separated: fdm, sls, mjf (overrides the spec's processes)")
    co.add_argument("--base-dir")
    sw = sub.add_parser("switch", help="a target-force push pad: minimize the volume inside a fixed exterior so "
                                       "a press anywhere on the touch area fires a switch at a chosen force "
                                       "(the study's switch block)")
    sw.add_argument("study")
    sw.add_argument("--out", default="loadpath-switch")
    sw.add_argument("--pitches", help="comma-separated mm, coarse to fine (default: switch.pitches)")
    sw.add_argument("--iterations", help="comma-separated, per pitch")
    sw.add_argument("--nonlinear", action=argparse.BooleanOptionalAction, default=None,
                    help="the large-deflection fire-force check (default: switch.nonlinear, on unless the study "
                    "turns it off); --no-nonlinear skips it")
    sw.add_argument("--skin-review", action="store_true",
                    help="read the bare skin only, in seconds, before designing anything: which presses the fixed "
                         "exterior alone puts over their ceilings, where, and what the designer could change "
                         "(report skinReview; nothing is optimized or edited)")
    sw.add_argument("--lattice", help="a graded lattice instead of ribs: TYPE[:CELL_MM[:MODE]], e.g. "
                    "octet:5:where-needed (default: switch.lattice, if the study has one)")
    sw.add_argument("--base-dir")
    ho = sub.add_parser("handoff", help="a designer's handoff: the exterior and the component CAD in, a push-pad "
                                        "study out (keep-outs and clearance per component, insertFrom from the "
                                        "assembly order, supports, the plunger and its seat); --run designs it")
    ho.add_argument("brief", help="the handoff brief (`loadpath schema handoff`)")
    ho.add_argument("--out", default="loadpath-handoff")
    ho.add_argument("--run", action="store_true", help="then run `loadpath switch` on the study")
    ho.add_argument("--pitches", help="with --run: comma-separated mm")
    ho.add_argument("--iterations", help="with --run: comma-separated, per pitch")
    ho.add_argument("--base-dir", help="where the brief's relative paths start (default: the brief's folder)")
    ex = sub.add_parser("exterior", help="any exterior in (STEP, mesh, an AI-generated or scanned mesh, an SDF "
                                         "function or build123d object in file.py[:name], an .npz SDF): one clean "
                                         "watertight skin out, with a report of what was wrong, what the wrap "
                                         "changed and where the shape is too thin to hold anything")
    ex.add_argument("source")
    ex.add_argument("--out", default="loadpath-exterior")
    ex.add_argument("--size", type=float, help="the real longest extent, mm (generated meshes come at unit scale)")
    ex.add_argument("--scale", type=float, help="a factor instead of --size (metres to mm: 1000)")
    ex.add_argument("--up", help="the source's up axis (default +y for GLB/glTF, else +z)")
    ex.add_argument("--place", choices=["keep", "center", "origin"])
    ex.add_argument("--pitch", type=float, help="mm, the wrap's grid (default longest extent / 320)")
    ex.add_argument("--wrap", choices=["auto", "always", "never"], default="auto")
    ex.add_argument("--keep", choices=["largest", "all"], default="largest")
    ex.add_argument("--skin", type=float, default=1.0, help="the wall the study will keep, mm (for the room report)")
    ex.add_argument("--sdf", action="store_true", help="also write exterior.sdf.npz (a signed distance grid)")
    sd = sub.add_parser("sdf", help="a part (STL/STEP/GLB) or an SDF as a signed distance grid (.npz: distance, "
                                    "origin, pitch; mm, negative inside) for SDF tools; loads back as a study "
                                    "shape {type: sdf}")
    sd.add_argument("source")
    sd.add_argument("--out", help="the .npz (default <source>.sdf.npz)")
    sd.add_argument("--pitch", type=float, help="mm (default longest extent / 256)")
    spl = sub.add_parser("split", help="cut a study's exterior into a base and a cover that go together round "
                                       "its components: the cut checked or searched, an alignment lip, screw "
                                       "posts and snap fits inside the skin; a study per half")
    spl.add_argument("study", help="a study with a skin and a split block (`loadpath schema split`)")
    spl.add_argument("--split", help="the split block as a JSON file, instead of the study's own")
    spl.add_argument("--out", default="loadpath-split")
    spl.add_argument("--base-dir", help="where the study's relative paths start (default: its folder)")
    en = sub.add_parser("enclosure", help="a product from its outside in: the designer's exterior fixed, the "
                                          "components placed inside it (or as given), their assembly paths, "
                                          "mounts, drops, squeezes, heat, a push pad or feel targets; a study "
                                          "out, and with --run the designed part, the exterior checked untouched")
    en.add_argument("brief", help="the enclosure brief (`loadpath schema enclosure`)")
    en.add_argument("--out", default="loadpath-enclosure")
    en.add_argument("--run", action="store_true", help="then design it (run, switch or feel) and check the part")
    en.add_argument("--pitches", help="with --run on a switch/feel study: comma-separated mm")
    en.add_argument("--iterations", help="with --run on a switch/feel study: comma-separated, per pitch")
    en.add_argument("--time-limit", type=float, metavar="MINUTES", help="with --run: as `run --time-limit`")
    en.add_argument("--base-dir", help="where the brief's relative paths start (default: the brief's folder)")
    fe = sub.add_parser("feel", help="design the inside of a fixed exterior for how it feels: grip squeeze, drop "
                                     "cushioning, vibration isolation of a part inside (the study's feel block; "
                                     "with a switch block too, the click force with them)")
    fe.add_argument("study")
    fe.add_argument("--out", default="loadpath-feel")
    fe.add_argument("--pitches", help="comma-separated mm, coarse to fine (default: switch.pitches or feel.pitches)")
    fe.add_argument("--iterations", help="comma-separated, per pitch")
    fe.add_argument("--no-modal", action="store_true", help="skip the modal check of isolate targets")
    fe.add_argument("--base-dir")
    co = sub.add_parser("cooling", help="design the inside of a fixed exterior around the heat its parts make: "
                                        "components under their maxTemp, the touched skin under its touch "
                                        "limit (the study's cooling block; with feel or switch blocks, "
                                        "together with them)")
    co.add_argument("study")
    co.add_argument("--out", default="loadpath-cooling")
    co.add_argument("--pitches", help="comma-separated mm, coarse to fine (default: switch.pitches or "
                                      "cooling.pitches)")
    co.add_argument("--iterations", help="comma-separated, per pitch")
    co.add_argument("--verify", help="comma-separated mm grids for the delivered mesh's temperatures")
    co.add_argument("--base-dir")
    ex = sub.add_parser("explore", help="one study, an outcome per manufacturing method and material (fdm, sls, "
                                        "mjf, sla, lpbf, cast, milled...): mass, safety factor, cost and lead "
                                        "time, ranked, with Pareto fronts")
    ex.add_argument("study")
    ex.add_argument("--out", default="loadpath-explore")
    ex.add_argument("--methods", help="comma-separated (default: the study's explore.methods, else "
                                      "fdm,mjf,sla,lpbf,cast,milled-3axis)")
    ex.add_argument("--materials", help="comma-separated library names; each runs on every method that makes it")
    ex.add_argument("--quantity", type=int, help="parts ordered: spreads tooling and set-up cost (default 1)")
    ex.add_argument("--rank", choices=["mass", "cost", "leadTime"], help="order of the ranking (default mass)")
    ex.add_argument("--jobs", type=int, help="methods run at once (default half the cores)")
    ex.add_argument("--screen", action="store_true", help="every route on the coarse grid first, then full runs "
                                                          "only for the Pareto fronts and the top of the ranking")
    ex.add_argument("--plan", action="store_true", help="list the routes and their studies, run nothing")
    ex.add_argument("--base-dir")
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
    cal.add_argument("--hole-gauge", type=float, default=None,
                     help="kit: diameter (mm) of the drill bit or pin used on the hole-fit coupons (default 5)")
    cal.add_argument("--out", default="calibration", help="kit: folder for the STLs; fit: folder for the card")
    cal.add_argument("--predictions", help="fit: the kit's predictions.json (adds part-test model factors)")
    cal.add_argument("--name", help="fit: the card's name, usable as a study's material once saved")
    cal.add_argument("--save", action="store_true", help="fit: also save the card to loadpath/calibrations/")
    gd = sub.add_parser("guide", help="print the agent guide (SKILL.md: how to turn a request into a study)")
    gd.add_argument("topic", nargs="?", help="core, study (every field), lattice, heat, multimaterial, example, "
                                             "report, or all (default: all of SKILL.md)")
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
    bs = sub.add_parser("buildsim", help="simulate printing a part (lpbf, fdm, sls): distortion, residual stress, "
                                         "recoater lift; --compensate writes the pre-distorted part to print")
    bs.add_argument("part", help="STL, 3MF or OBJ, in mm, in its build position")
    bs.add_argument("--material", default="ti64", help="material card (`loadpath materials`)")
    bs.add_argument("--process", choices=["lpbf", "fdm", "sls"], default="lpbf")
    bs.add_argument("--build", default="+z", help="build direction in the part's frame (default +z)")
    bs.add_argument("--plate", type=float, default=10.0, help="mm of build plate under the part (lpbf)")
    bs.add_argument("--pitch", type=float, help="mm voxel (default: about --cells cells; the LPBF strain is "
                    "calibrated at 0.5 mm)")
    bs.add_argument("--cells", type=int, default=60_000, help="part cells when --pitch is not given")
    bs.add_argument("--scale", type=float, default=1.0, help="multiplies the inherent strain (a calibration)")
    bs.add_argument("--recoater-gap", type=float, default=0.05, help="mm of build-plane lift that is a risk")
    bs.add_argument("--compensate", type=int, default=0, metavar="N",
                    help="also write <part>-compensated.stl, pre-distorted over N rounds")
    bs.add_argument("--nominal", help="the design the printed shape is compared with (default: the part itself; "
                    "give the design when simulating a compensated part)")
    bs.add_argument("--out", default="loadpath-buildsim")
    sx = sub.add_parser("step", help="a closed STL/3MF/OBJ part as a STEP solid (needs loadpath[step]); a part "
                                     "with a gyroid or other TPMS core keeps every tunnel")
    sx.add_argument("part", help="STL, 3MF or OBJ, in mm, closed")
    sx.add_argument("--out", help="STEP path (default: the part's name with .step)")
    sx.add_argument("--faces", type=int, default=None, help="facet budget (default 20000); a TPMS core goes over "
                    "it rather than lose a tunnel")
    sx.add_argument("--tolerance", type=float, default=None,
                    help="mm the facets may move to keep a TPMS core's tunnels (default 0.05)")
    sx.add_argument("--surface", choices=["facets", "smooth"], default="facets")
    sl = sub.add_parser("slices", help="a closed part as powder-bed layers: contours and scan hatches in a CLI "
                                       "file (Common Layer Interface, ASCII) for EOS, 3D Systems, Renishaw and "
                                       "other build processors")
    sl.add_argument("part", help="STL, 3MF or OBJ, in mm, closed")
    sl.add_argument("--out", help="CLI path (default: the part's name with .cli)")
    sl.add_argument("--layer", type=float, default=0.03, help="layer thickness, mm (default 0.03, LPBF)")
    sl.add_argument("--hatch", type=float, default=0.1,
                    help="hatch spacing, mm (default 0.1; 0 for contours only); turned 67 degrees a layer")
    sl.add_argument("--axis", choices=["x", "y", "z"], default="z", help="build axis (default z)")
    ef = sub.add_parser("external-flow", help="a part in a stream of air or water (lattice Boltzmann wind "
                                              "tunnel): drag, lift, moment, and the surface forces as a load case")
    ef.add_argument("part", help="STL, 3MF or OBJ in mm, closed")
    ef.add_argument("--speed", type=float, required=True, help="m/s")
    ef.add_argument("--direction", default="x", help="the way the fluid moves: x, -z, ... or vx,vy,vz (default x)")
    ef.add_argument("--medium", default="air", help="air, water, glycol50, oil (default air)")
    ef.add_argument("--cells", type=int, default=16, help="voxels across the part's frontal size (default 16)")
    ef.add_argument("--re-max", type=float, default=2e4, help="Reynolds number the run is capped at (default 2e4)")
    ef.add_argument("--study", help="append the flow's load case to this study, written to --out")
    ef.add_argument("--out", help="the study with the flow case (default <study>-flow.json)")
    fx = sub.add_parser("fe-export", help="a part with the study's material, supports and load cases as a "
                                          "Nastran (.bdf) or Abaqus / CalculiX (.inp) deck in 10-node tets")
    fx.add_argument("study")
    fx.add_argument("part", help="the part's STL, 3MF or OBJ (closed)")
    fx.add_argument("--out", help="deck path; the extension picks the format (.bdf/.nas/.dat or .inp)")
    fx.add_argument("--format", choices=["nastran", "abaqus"], default=None)
    fx.add_argument("--size", type=float, default=None, help="tet edge, mm (default: the cross-check's)")
    ns = sub.add_parser("nest", help="pack parts (and copies) into one build volume: on the plate (FDM, resin, "
                                     "metal) or stacked in the powder (SLS, MJF), with a measured gap")
    ns.add_argument("parts", nargs="+", help="STL/3MF/OBJ paths, each optionally with :copies (bracket.stl:6)")
    ns.add_argument("--volume", type=float, nargs=3, metavar=("X", "Y", "Z"), help="build volume, mm")
    ns.add_argument("--process", default=None, help="fdm, sla, dlp, lpbf (on the plate) or sls, mjf (stacked)")
    ns.add_argument("--material", default=None, help="with --process and no --volume: Materialize's build volume")
    ns.add_argument("--mode", choices=["bed", "volume"], default=None, help="override the process's mode")
    ns.add_argument("--rotations", choices=["z", "any", "none"], default=None,
                    help="z: quarter turns about the build axis (default on a plate); any: 24 axis-aligned "
                         "orientations (default stacked)")
    ns.add_argument("--gap", type=float, default=2.0, help="least distance between parts, mm (default 2)")
    ns.add_argument("--pitch", type=float, default=None, help="packing grid, mm (default from the volume)")
    ns.add_argument("--out", default=None, help="the packed build as one mesh (default nest.stl; .3mf needs lxml)")
    ns.add_argument("--full", action="store_true", help="print every placement's transform")
    asm = sub.add_parser("assembly",help="derive a part's load cases and neighbour keep-outs from a URDF")
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
    asm.add_argument("--push", type=float, default=20.0, help="N: contact at the payload, any direction (default 20; often sizes the part, 0 if nothing pushes)")
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
    for sp in (r, lt, h, fg, tu, ed, asm, v, bs, ex, en):
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
    if args.cmd == "cores":
        return _cores(args)
    if args.cmd == "switch":
        return _switch(args)
    if args.cmd == "feel":
        return _feel(args)
    if args.cmd == "cooling":
        return _cooling(args)
    if args.cmd == "handoff":
        return _handoff(args)
    if args.cmd == "exterior":
        return _exterior(args)
    if args.cmd == "sdf":
        return _sdf(args)
    if args.cmd == "split":
        return _split(args)
    if args.cmd == "enclosure":
        return _enclosure(args)
    if args.cmd == "explore":
        return _explore(args)
    if args.cmd == "lattice-orient":
        return _lattice_orient(args)
    if args.cmd == "resolved-check":
        return _resolved_check(args)
    if args.cmd == "lattice-grade":
        return _lattice_grade(args)
    if args.cmd == "orient":
        return _orient(args)
    if args.cmd == "calibrate":
        return _calibrate(args)
    if args.cmd == "assembly":
        return _assembly(args)
    if args.cmd == "status":
        st = run_status(args.out)
        if args.wait:
            t0 = time.time()
            t_end = t0 + args.wait
            # a missing status.json gets 30 s to appear (a run just launched), not the whole wait
            while (st.get("state") == "running" or st.get("state") == "none" and time.time() < t0 + 30) \
                    and time.time() < t_end:
                time.sleep(min(5.0, max(0.1, t_end - time.time())))
                st = run_status(args.out)
        print(json.dumps(st, indent=1))
        if not args.wait:
            return 0
        return 0 if st.get("state") == "done" else 3 if st.get("state") in ("running", "none") else 1
    if args.cmd == "verify":
        return _verify(args)
    if args.cmd == "import":
        return _import(args)
    if args.cmd == "step":
        return _step(args)
    if args.cmd == "external-flow":
        return _external_flow(args)
    if args.cmd == "fe-export":
        return _fe_export(args)
    if args.cmd == "nest":
        return _nest(args)
    if args.cmd == "slices":
        return _slices(args)
    if args.cmd == "buildsim":
        return _buildsim(args)
    if args.cmd == "guide":
        if args.topic:
            from .mcp_server import guide
            try:
                print(guide(args.topic))
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
            return 0
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")) as fh:
            print(fh.read())
        return 0

    raw = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    if args.cmd == "check":
        from .tasks import check_document
        checked = check_document(raw, base)
        print(json.dumps(checked, indent=2))
        return 0 if checked["ok"] else 2
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
        return _run_multimaterial(args, study)
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
    import signal

    def _killed(signum, _frame):
        # a hang-up or kill from the shell that launched us (an agent's tool
        # call ending takes its process group down): say so, then go
        try:
            _status("killed", signal=signal.Signals(signum).name, stage=seen.get("stage"),
                    partial=os.environ.get("LOADPATH_PARTIAL_DIR"),
                    hint="relaunch detached: setsid nohup loadpath run ... > run.log 2>&1 < /dev/null &")
        finally:
            os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _killed)
        except (ValueError, OSError):
            pass
    # each outcome as the search finds it, kept if a later stage dies
    os.environ.setdefault("LOADPATH_PARTIAL_DIR", os.path.join(args.out, "partial"))
    rec = None
    if args.viewer != "none" and not os.environ.get("LOADPATH_NO_RENDER"):
        from .viewer import HistoryRecorder
        rec = HistoryRecorder(chain=_progress)
    seen = {"stage": None, "outcomes": 0}

    def _tracked(event, *rest):
        # status.json carries the stage and outcomes so far: one file to poll
        st = event.get("stage")
        if st == "outcome":
            seen["outcomes"] += 1
        if st and st != seen["stage"] or st == "outcome":
            seen["stage"] = st
            try:
                _status("running", started=started, stage=st, outcomes=seen["outcomes"],
                        **({"volumeFraction": event["volumeFraction"]} if "volumeFraction" in event else {}))
            except OSError:
                pass
        return (rec or _progress)(event, *rest)

    started = time.time()
    limit = getattr(args, "time_limit", None) or float(os.environ.get("LOADPATH_TIME_LIMIT") or 0) or None
    should_stop = None
    if limit:
        deadline = started + 60.0 * limit

        def should_stop():
            if time.time() < deadline:
                return False
            if not seen.get("timeUp"):
                seen["timeUp"] = True
                _progress({"stage": "warning", "message": f"time limit {limit:g} min reached: delivering the "
                                                          "best outcome verified so far"}, None)
            return True
    try:
        result = run_study(study, on_progress=_tracked, should_stop=should_stop)
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
    dv = summary["deliver"]
    if dv and dv.get("safetyFactorMethod") != "tet":
        # say why the body-fitted cross-check did not confirm it
        try:
            from .crosscheck import unsupported
            why = unsupported(study)
        except Exception as exc:  # noqa: BLE001
            why = f"{type(exc).__name__}: {exc}"
        dv["tetCrossCheck"] = f"not run: {why}" if why else "not run: goal.confirm chose the voxel re-check"
    _write_report(args.out, summary)
    _status("done", report=os.path.join(os.path.abspath(args.out), "report.json"))
    return 0


def _run_multimaterial(args, study):
    """`run` on a multiMaterial study (multimat.py): bodies, an STL per
    material, a multi-material 3MF and report.json, with out/status.json
    and out/partial/ kept as the single-material run keeps them."""
    from .multimat import run_multimaterial

    os.makedirs(args.out, exist_ok=True)
    status_path = os.path.join(args.out, "status.json")

    def _status(state, **kw):
        with open(status_path, "w") as fh:
            json.dump(_jsonable({"state": state, "pid": os.getpid(), "study": os.path.abspath(args.study),
                                 "multiMaterial": True, **kw}), fh)

    started = time.time()
    _status("running", started=started)
    os.environ.setdefault("LOADPATH_PARTIAL_DIR", os.path.join(args.out, "partial"))
    seen = {"stage": None, "outcomes": []}

    def _tracked(event, *_):
        st = event.get("stage")
        if st == "outcome":
            # each verified fraction: mass and safety factors, per material and bond
            seen["outcomes"].append({k: event.get(k) for k in (
                "volumeFraction", "mass", "safetyFactor", "interfaceSafetyFactor", "margin", "passes")})
        if st and st != seen["stage"] or st == "outcome":
            seen["stage"] = st
            try:
                _status("running", started=started, stage=st, outcomes=len(seen["outcomes"]),
                        tried=seen["outcomes"],
                        **({"volumeFraction": event["volumeFraction"]} if "volumeFraction" in event else {}))
            except OSError:
                pass
        _progress(event, None)

    try:
        mres = run_multimaterial(study, on_progress=_tracked)
        files = mres.write(args.out, coordinates=getattr(args, "mm_coordinates", None) or "bed")
    except BaseException as exc:
        _status("failed", error=f"{type(exc).__name__}: {exc}", partial=os.environ.get("LOADPATH_PARTIAL_DIR"))
        raise
    rep = mres.report
    print(json.dumps(_jsonable({"ok": True, "mass": rep.get("mass"), "materialMass": rep.get("materialMass"),
                                "safetyFactor": rep.get("safetyFactor"),
                                "interfaceSafetyFactor": rep.get("interfaceSafetyFactor"),
                                "requirementsMet": rep.get("requirementsMet"), "passes": rep.get("passes"),
                                **({"unchecked": rep["unchecked"]} if rep.get("unchecked") else {}),
                                **({"warnings": rep["warnings"]} if rep.get("warnings") else {}),
                                "3mf": {k: v for k, v in (getattr(mres, "delivered_3mf", None) or {}).items()
                                        if k != "transform"} or None,
                                "files": files}), indent=2, default=str))
    _status("done", report=os.path.abspath(files["report"]), passes=rep.get("passes"))
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
    status_path = None

    def _status(state, **kw):
        if status_path is None:
            return
        try:
            with open(status_path, "w") as fh:
                json.dump(_jsonable({"state": state, "pid": os.getpid(), "spec": os.path.abspath(args.spec),
                                     "started": started, "updated": time.time(), **kw}), fh)
        except OSError:
            pass

    started = time.time()
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
                                                 splits=splits, axes=args.axes, max_points=args.max_points,
                                                 out_dir=args.out), indent=2))
            return 0
        out = args.out or "loadpath-out"
        # a design can take minutes and gigabytes: status.json says which stage it is
        # in (a run killed for memory leaves "running" with its stage and pid)
        os.makedirs(out, exist_ok=True)
        status_path = os.path.join(out, "status.json")
        _status("running", stage="check")

        seen = {}

        def _tracked(event):
            if event.get("stage"):
                seen.update({k: v for k, v in event.items() if k in ("stage", "gridStep", "gridPoints",
                                                                    "peakMemoryGB")})
                _status("running", **seen)
            _progress(event)

        rep = heat_exchanger(spec, out, base, render=not args.no_render and _can_render(),
                             on_progress=_tracked, viewer=not args.no_viewer and not os.environ.get("LOADPATH_NO_RENDER"),
                             max_points=args.max_points, force=args.force)
    except ValueError as exc:
        _status("failed", error=str(exc))
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    except BaseException as exc:
        _status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _write_report(out, rep)
    _status("done", report=os.path.join(out, "report.json"), passes=rep.get("passes"),
            warnings=(rep.get("report") or {}).get("warnings"))
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
    if getattr(args, "lattice_3mf", False) and not doc.get("lattice3mf"):
        doc = {**doc, "lattice3mf": True}
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
    summary = {"ok": bool(out["best"] and out["best"].get("ok")), "best": out["best"], "start": out["start"],
               "limits": out["limits"], "evaluations": len(out["history"]),
               "tuned": os.path.join(args.out, "tuned.json")}
    if out["best"] is not None:
        # tuned.json keeps the graph's relative paths (its study): they resolve from the graph's folder
        summary["next"] = f"loadpath field {summary['tuned']} --base-dir {os.path.relpath(base)}"
    if out["best"] is None:
        summary["error"] = "no candidate met the limits: widen the params' ranges or ease the limits"
    elif not summary["ok"]:
        summary["error"] = ("the lightest candidates met the limits on the search's grid but not on the confirming "
                            "one: widen the ranges, ease the limits or give more evaluations")
    _write_report(args.out, summary)
    return 0 if summary["ok"] else 1


def _lattice_orient(args):
    from .fields import load_study
    from .lattice_orient import optimize_orientation, optimize_zones

    doc = _load(args.graph)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.graph))
    ref = args.study or (doc.get("tune") or {}).get("study")
    if not ref:
        print(json.dumps({"ok": False, "error": "give --study (or a tune.study in the graph)"}))
        return 2
    try:
        study = load_study(ref, base, doc.get("material"))
        if args.zones > 1:
            out = optimize_zones(doc, study, zones=args.zones, base_dir=base, cells=args.cells,
                                 objective=args.objective, progress=_progress, seam=args.seam,
                                 confirm=args.confirm, resolved_pitch=args.resolved_pitch)
        else:
            out = optimize_orientation(doc, study, base_dir=base, cells=args.cells, objective=args.objective,
                                       refine=args.refine, progress=_progress, confirm=args.confirm,
                                       resolved_pitch=args.resolved_pitch)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "oriented.json")
    with open(path, "w") as fh:
        json.dump(out["doc"], fh, indent=1)
    summary = {"ok": True, **{k: v for k, v in out.items() if k not in ("doc", "history")},
               "oriented": path, "next": f"loadpath field {path} --base-dir {os.path.relpath(base)}"}
    _write_report(args.out, summary)
    return 0


def _resolved_check(args):
    from .fields import load_study
    from .resolved import resolved_check

    doc = _load(args.graph)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.graph))
    ref = args.study or (doc.get("tune") or {}).get("study")
    if not ref:
        print(json.dumps({"ok": False, "error": "give --study (or a tune.study in the graph)"}))
        return 2
    try:
        study = load_study(ref, base, doc.get("material"))
        rep = resolved_check(doc, study, pitch=args.pitch, base_dir=base, contact_band=args.band,
                             max_cells=args.max_cells)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(json.dumps({"ok": True, **rep}, indent=1))
    return 0


def _lattice_grade(args):
    import numpy as np

    from .fields import load_study
    from .lattice_grade import grade_lattice

    doc = _load(args.graph)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.graph))
    ref = args.study or (doc.get("tune") or {}).get("study")
    if not ref:
        print(json.dumps({"ok": False, "error": "give --study (or a tune.study in the graph)"}))
        return 2
    out_dir = os.path.abspath(args.out if os.path.isabs(args.out) else os.path.join(base, args.out))
    if os.path.commonpath([os.path.realpath(base), os.path.realpath(out_dir)]) != os.path.realpath(base):
        print(json.dumps({"ok": False, "error": "--out must be inside the graph's folder (its density grid is "
                                                "read from there)"}))
        return 2
    orient = args.orient
    if orient and orient.isdigit():
        orient = int(orient)
    try:
        study = load_study(ref, base, doc.get("material"))
        out = grade_lattice(doc, study, target_sf=args.target_sf, wall_min=args.wall_min, wall_max=args.wall_max,
                            cells=args.cells, orient=orient, base_dir=base, progress=_progress, seam=args.seam,
                            confirm=args.confirm)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    if out["best"] is None:
        print(json.dumps({"ok": False, "error": "no graded design met the target: raise --wall-max or lower "
                                                "--target-sf", "history": out["history"]}))
        return 1
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "density.npz")
    d = out["density"]
    np.savez(npz, origin=np.asarray(d["origin"]), pitch=d["pitch"], values=d["values"])
    g = out["doc"]
    g["nodes"][out["densityNode"]] = {"op": "grid", "path": os.path.relpath(npz, base)}
    path = os.path.join(out_dir, "graded.json")
    with open(path, "w") as fh:
        json.dump(g, fh, indent=1)
    summary = {"ok": True, "best": out["best"], "target": out["target"], "densityLimits": out["densityLimits"],
               "graded": path, "density": npz, "seconds": out["seconds"], "guard": out.get("guard"),
               "next": f"loadpath field {path} --base-dir {os.path.relpath(base)}"}
    _write_report(out_dir, summary)
    return 0


def _cores(args):
    from .cores import choose_core

    spec = _load(args.spec)
    if args.process:
        spec["processes"] = args.process.split(",")
    base = args.base_dir or os.path.dirname(os.path.abspath(args.spec))
    try:
        rep = choose_core(spec, base, args.out, progress=_progress)
    except (ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, {"ok": any(rep["winners"].values()), **rep})
    return 0 if all(rep["winners"].values()) else 1


def _explore(args):
    from .explore import explore, plan
    from .schema import format_issues, lint

    study = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    opts = {}
    if args.methods:
        opts["methods"] = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.materials:
        opts["materials"] = [m.strip() for m in args.materials.split(",") if m.strip()]
    for k in ("quantity", "rank"):
        if getattr(args, k) is not None:
            opts[k] = getattr(args, k)
    if args.screen:
        opts["screen"] = True
    errors = [x for x in lint({**study, "explore": {**(study.get("explore") or {}), **opts}}, "study")
              if x["level"] == "error"]
    if errors:
        print(json.dumps({"ok": False, "error": format_issues(errors)}))
        return 2
    try:
        if args.plan:
            p = plan(study, opts, base)
            print(json.dumps({"ok": True, **p}, indent=1))
            return 0
        from . import quick
        rep = explore(study, args.out, opts, base_dir=base, jobs=args.jobs, quick=quick.on(),
                      progress=_progress)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(rep.pop("table"), file=sys.stderr)
    _write_report(args.out, {"ok": bool(rep["picks"]), **rep})
    return 0 if rep["picks"] else 1


def _switch(args):
    from .schema import format_issues, lint
    from .spec import study_from_dict
    from .switch import run_switch

    doc = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    errors = [x for x in lint(doc, "study") if x["level"] == "error"]
    if not doc.get("switch"):
        errors.append({"level": "error", "path": "$.switch", "message": "the study needs a switch block"})
    if errors:
        print(json.dumps({"ok": False, "error": format_issues(errors)}))
        return 2
    try:
        if args.lattice:
            parts = args.lattice.split(":")
            lb = dict(doc["switch"].get("lattice") or {}, type=parts[0])
            if len(parts) > 1:
                lb["cell"] = float(parts[1])
            if len(parts) > 2:
                lb["mode"] = parts[2]
            doc = {**doc, "switch": {**doc["switch"], "lattice": lb}}
        st = study_from_dict(doc, base)
        pitches = [float(v) for v in args.pitches.split(",")] if args.pitches else None
        its = [int(v) for v in args.iterations.split(",")] if args.iterations else None
        if args.skin_review:
            from .skinreview import skin_preflight
            sr = skin_preflight(st, pitches[0] if pitches else None)
            _write_report(args.out, {"ok": sr["skinLimited"] == 0, "name": st.name, "skinReview": sr})
            return 0 if sr["skinLimited"] == 0 else 1
        if doc["switch"].get("lattice"):
            from .switch_lattice import run_switch_lattice
            rep = run_switch_lattice(st, args.out, iterations=its[0] if its else 80, on_progress=_progress)
            rep["passes"] = rep["inWindow"] == len(rep["finalReadback"]["forceGf"])
            _write_report(args.out, {"ok": rep["passes"], **rep})
            return 0 if rep["passes"] else 1
        rep = run_switch(st, args.out, pitches=pitches, iterations=its, nonlinear=args.nonlinear,
                         on_progress=_progress)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, {"ok": rep["passes"], **rep})
    return 0 if rep["passes"] else 1


def _split(args):
    from .schema import format_issues, lint
    from .shells import run_split

    doc = _load(args.study)
    split = _load(args.split) if args.split else doc.get("split")
    if not isinstance(split, dict):
        print(json.dumps({"ok": False, "error": "no split block: give the study one, or --split split.json"}))
        return 2
    errors = [x for x in lint(split, "split") if x["level"] == "error"]
    if errors:
        print(json.dumps({"ok": False, "error": format_issues(errors)}))
        return 2
    try:
        r = run_split(args.study, args.out, split=split, base_dir=args.base_dir)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    rep = r["report"]
    ok = not any(rep["lint"].values())
    out = {"ok": ok, "studies": r["studyPaths"], "report": os.path.join(args.out, "split.json"),
           "cut": {k: rep["cut"].get(k) for k in ("normal", "at", "seam", "how")},
           "componentsIn": rep["assembly"]["componentsIn"],
           "screws": len(rep["screws"]), "snaps": (rep["snaps"] or {}).get("count", 0),
           "snapStrain": None if not rep["snaps"] else [rep["snaps"]["strain"], rep["snaps"]["allowable"]],
           "exteriorChanges": rep["exteriorChanges"], "warnings": rep["warnings"]}
    print(json.dumps(out, default=float))
    return 0 if ok else 2


def _handoff(args):
    from .handoff import run_handoff
    from .schema import format_issues, lint

    doc = _load(args.brief)
    errors = [x for x in lint(doc, "handoff") if x["level"] == "error"]
    if errors:
        print(json.dumps({"ok": False, "error": format_issues(errors)}))
        return 2
    try:
        r = run_handoff(args.brief, args.out, base_dir=args.base_dir)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    rep = r["report"]
    out = {"ok": not rep["lint"], "study": r["studyPath"], "report": os.path.join(args.out, "handoff.json"),
           "assembly": {k: v.get("insertFrom") for k, v in rep["assembly"]["components"].items()},
           "plunger": {"tip": rep["plunger"]["tip"], "radius": rep["plunger"]["radius"]},
           "warnings": rep["warnings"]}
    print(json.dumps(out, default=float))
    if args.run and out["ok"]:
        args.study, args.base_dir, args.out = r["studyPath"], None, os.path.join(args.out, "switch")
        args.nonlinear, args.skin_review, args.lattice = None, False, None
        return _switch(args)
    return 0 if out["ok"] else 2


def _exterior(args):
    from .exterior import load_exterior, save_sdf
    try:
        e = load_exterior(args.source, size=args.size, scale=args.scale, up=args.up, place=args.place,
                          pitch=args.pitch, wrap=args.wrap, keep=args.keep, skin=args.skin)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    stl = e.save(args.out)
    r = e.report
    out = {"ok": bool(r["result"]["watertight"]), "stl": stl, "report": os.path.join(args.out, "exterior.json"),
           "kind": r["kind"], "wrapped": "wrap" in r, "extents": r["result"]["extents"],
           "volume": r["result"]["volume"], "warnings": r["warnings"]}
    if "wrap" in r:
        out["onSourceP95"] = r["wrap"]["onSource"]["p95"]
        out["patchedArea"] = r["wrap"]["patched"]["area"]
    if r.get("capacity"):
        out["cavityVolume"] = r["capacity"]["cavityVolume"]
        out["thinRegions"] = [x["center"] for x in r["capacity"].get("thinRegions", [])]
    if args.sdf:
        out["sdf"] = save_sdf(e.mesh, os.path.join(args.out, "exterior.sdf.npz"))["path"]
    print(json.dumps(out, default=float))
    return 0 if out["ok"] else 2


def _sdf(args):
    from .exterior import save_sdf
    src = args.source
    path = args.out or os.path.splitext(src.partition(".py:")[0])[0] + ".sdf.npz"
    try:
        if ".py" in src:
            from .exterior import _from_source
            m, kind, _, sdf = _from_source(src)
            src = m if sdf is None else sdf[0]
            r = save_sdf(src, path, args.pitch, None if sdf is None else sdf[1])
        else:
            r = save_sdf(src, path, args.pitch)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(json.dumps({"ok": True, **r}, default=float))
    return 0
def _enclosure(args):
    from .enclosure import run_enclosure
    from .schema import format_issues, lint

    doc = _load(args.brief)
    errors = [x for x in lint(doc, "enclosure") if x["level"] == "error"]
    if errors:
        print(json.dumps({"ok": False, "error": format_issues(errors)}))
        return 2
    try:
        r = run_enclosure(args.brief, args.out, base_dir=args.base_dir, on_progress=_progress)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    rep = r["report"]
    out = {"ok": not rep["lint"], "study": r["studyPath"], "report": os.path.join(args.out, "enclosure.json"),
           "placement": {k: {"centre": v["centre"], "roomToSkin": v["roomToSkin"], "auto": v["auto"]}
                         for k, v in rep["placement"]["components"].items()},
           "assembly": {k: v.get("insertFrom") for k, v in rep["assembly"]["components"].items()},
           "loadCases": rep["loadCases"], "warnings": rep["warnings"],
           **({"lint": rep["lint"]} if rep["lint"] else {})}
    print(json.dumps(out, default=float))
    if not (args.run and out["ok"]):
        return 0 if out["ok"] else 2
    from .enclosure import check_delivered
    study = r["study"]
    sp = r["studyPath"]
    design = os.path.join(args.out, "design")
    pitches = args.pitches
    if study.get("feel") or study.get("switch"):
        ns = argparse.Namespace(study=sp, base_dir=None, out=design, pitches=pitches, iterations=args.iterations,
                                nonlinear=None, skin_review=False, lattice=None, no_modal=False)
        code = _feel(ns) if study.get("feel") else _switch(ns)
    else:
        argv = ["run", sp, "--out", design, "--viewer", "none"] + \
            (["--time-limit", str(args.time_limit)] if args.time_limit else []) + \
            (["--quick"] if getattr(args, "quick", False) else [])
        code = main(argv)
    chk = check_delivered(sp, design, quick=getattr(args, "quick", False), on_progress=_progress)
    json.dump(chk, open(os.path.join(args.out, "checks.json"), "w"), indent=1, default=float)
    print(json.dumps({"ok": chk["ok"], "checks": os.path.join(args.out, "checks.json"), "part": chk.get("part"),
                      "exterior": chk.get("exterior"), "summary": chk.get("summary")}, default=float))
    return 0 if chk["ok"] and code in (0, None) else 1


def _feel(args):
    from .feel import run_feel
    from .schema import format_issues, lint
    from .spec import study_from_dict

    doc = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    errors = [x for x in lint(doc, "study") if x["level"] == "error"]
    if not doc.get("feel"):
        errors.append({"level": "error", "path": "$.feel", "message": "the study needs a feel block"})
    if errors:
        print(json.dumps({"ok": False, "error": format_issues(errors)}))
        return 2
    try:
        st = study_from_dict(doc, base)
        pitches = [float(v) for v in args.pitches.split(",")] if args.pitches else None
        its = [int(v) for v in args.iterations.split(",")] if args.iterations else None
        rep = run_feel(st, args.out, pitches=pitches, iterations=its, on_progress=_progress,
                       modal=not args.no_modal)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, {"ok": rep["passes"], **rep})
    return 0 if rep["passes"] else 1


def _cooling(args):
    from .cooling import run_cooling
    from .schema import format_issues, lint
    from .spec import study_from_dict

    doc = _load(args.study)
    base = args.base_dir or os.path.dirname(os.path.abspath(args.study))
    errors = [x for x in lint(doc, "study") if x["level"] == "error"]
    if not doc.get("cooling"):
        errors.append({"level": "error", "path": "$.cooling", "message": "the study needs a cooling block"})
    if errors:
        print(json.dumps({"ok": False, "error": format_issues(errors)}))
        return 2
    try:
        st = study_from_dict(doc, base)
        pitches = [float(v) for v in args.pitches.split(",")] if args.pitches else None
        its = [int(v) for v in args.iterations.split(",")] if args.iterations else None
        vps = [float(v) for v in args.verify.split(",")] if args.verify else None
        rep = run_cooling(st, args.out, pitches=pitches, iterations=its, on_progress=_progress,
                          verify_pitches=vps)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, {"ok": rep["passes"], **rep})
    return 0 if rep["passes"] else 1


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


def _buildsim(args):
    """Simulate printing a mesh (buildsim.py): distortion, residual stress,
    recoater lift; with --compensate, the pre-distorted mesh to print."""
    import trimesh

    from . import quick
    from .buildsim import BuildSetup, compensate_mesh, deviation, printed_shape, refine_mesh, simulate_part

    try:
        setup = BuildSetup(process=args.process, material=args.material, plate=args.plate,
                           recoater_gap=args.recoater_gap, scale=args.scale)
        mesh = trimesh.load(args.part, force="mesh")
        nominal = trimesh.load(args.nominal, force="mesh") if args.nominal else mesh
        cells = quick.buildsim_cells(args.cells)
        pitch = args.pitch
        if pitch is None or quick.on():
            pitch = max(pitch or 0.0, float((abs(mesh.volume) / cells) ** (1 / 3)))
        stem = os.path.splitext(os.path.basename(args.part))[0]
        os.makedirs(args.out, exist_ok=True)
        res = simulate_part(mesh, setup, pitch=pitch, build=args.build)
        fine = refine_mesh(mesh, pitch)
        printed = printed_shape(fine, res, args.build)
        dev, aligned = deviation(printed, nominal)
        aligned.export(os.path.join(args.out, f"{stem}-printed.stl"))
        rep = {k: v for k, v in res.report.items() if k != "rotation"}
        summary = {"ok": True, "part": args.part, "nominal": args.nominal or args.part, "build": args.build,
                   **rep, "deviation": dev, "printed": os.path.join(args.out, f"{stem}-printed.stl")}
        if args.compensate:
            comp, crep = compensate_mesh(mesh, setup, pitch=pitch, build=args.build, iterations=args.compensate,
                                         first=res)
            path = os.path.join(args.out, f"{stem}-compensated.stl")
            comp.export(path)
            summary["compensation"] = {**crep, "stl": path}
    except (ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    _write_report(args.out, summary)
    return 0


def _verified_grid(mesh_path):
    """The analysis grid a run verified this STL on ({pitch, contactBand}),
    from the report.json beside it (best.stl is the report's best outcome),
    or None."""
    folder = os.path.dirname(os.path.abspath(mesh_path))
    rep = os.path.join(folder, "report.json")
    if not os.path.exists(rep):
        return None
    try:
        r = _load(rep)
        outs = r.get("outcomes") or []
        name = os.path.basename(mesh_path)
        if name == "best.stl" and r.get("best") is not None:
            o = outs[r["best"]]
        else:
            o = next((q for q in outs if os.path.basename(q.get("stl") or "") == name), None)
        g = (o or {}).get("analysis")
        return g if g and g.get("pitch") and g.get("contactBand") else None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _orient(args):
    import trimesh

    from .orient import rank_orientations
    from .spec import study_from_dict

    mesh = trimesh.load(args.mesh, force="mesh")
    study = None
    if args.study:
        with open(args.study) as fh:
            study = study_from_dict(json.load(fh), base_dir=os.path.dirname(os.path.abspath(args.study)))
    grid = _verified_grid(args.mesh) if study is not None else None
    rows = rank_orientations(mesh, study, process=args.process,
                             **({"analysis_pitch": grid["pitch"], "contact_band": grid["contactBand"]} if grid else {}))
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
