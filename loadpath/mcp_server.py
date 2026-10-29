"""
A local MCP server, so any agent (Claude Code, Cursor, Claude Desktop…) can
run generative design studies on the machine it is working on, with no
hosted service in between:

    {"mcpServers": {"loadpath": {"command": "loadpath", "args": ["mcp"]}}}

Tools:

- loadpath_guide: how to write a study (SKILL.md and the field reference)
- loadpath_schema: the JSON Schema for a study, heat exchanger or lattice
- loadpath_examples: worked example documents to start from
- loadpath_check: validate a study (every problem at once) and show the
  voxel problem, without solving
- loadpath_start: run a study as a background job (optionally with a
  lattice infill of the best outcome)
- loadpath_lattice: fill a finished outcome with a stress-graded lattice
- loadpath_heat_exchanger: design a two-fluid TPMS heat exchanger (a job)
- loadpath_size_heat_exchanger: pick cell and wall for a duty (seconds)
- loadpath_status / loadpath_jobs / loadpath_cancel: follow jobs

Jobs run in a child process, the same `python -m loadpath ...` the CLI
uses, under LOADPATH_HOME (default ~/.loadpath/jobs). Each job folder holds
its input, progress.jsonl (one JSON event per line), out/report.json and
the STLs and renders, and survives a restart of the server.

Needs the `mcp` extra: pip install "loadpath[mcp]".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
_procs: dict[str, subprocess.Popen] = {}
_POLL_HINT = "Poll loadpath_status every 30-60 seconds; a study takes minutes."
_VIEWER_NOTE = ("An interactive 3D page for the user: orbit the part, colour it by stress, safety factor or "
                "displacement, animate the deformation, cut it open, replay the optimization. Give them the path "
                "(or open it in a browser); it is one self-contained HTML file.")


def _home() -> str:
    root = os.environ.get("LOADPATH_HOME") or os.path.join(os.path.expanduser("~"), ".loadpath", "jobs")
    os.makedirs(root, exist_ok=True)
    return root


def _job_dir(job_id: str) -> str:
    if not job_id or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError(f"unknown job id {job_id!r}; loadpath_jobs lists them")
    d = os.path.join(_home(), job_id)
    if not os.path.isdir(d):
        raise ValueError(f"no job {job_id} under {_home()}; loadpath_jobs lists them")
    return d


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _meta(d) -> dict:
    try:
        return json.loads(_read(os.path.join(d, "job.json")) or "{}")
    except json.JSONDecodeError:
        return {}


def _alive(job_id, meta) -> bool:
    proc = _procs.get(job_id)
    if proc is not None:
        return proc.poll() is None
    pid = meta.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a pid the OS has since reused for something else
    cmd = _read(f"/proc/{pid}/cmdline")
    return not cmd or job_id in cmd


def guide() -> str:
    return _read(os.path.join(_HERE, "SKILL.md")) + "\n\n---\n\n" + _read(os.path.join(_HERE, "README.md"))


def check(study, base_dir: str | None = None) -> dict:
    from .tasks import check_study
    return check_study(study, base_dir)


def _spawn(kind: str, name: str, inputs: dict, argv: list, cells=None, warnings=None, parent=None,
           base_dir=None) -> dict:
    job_id = uuid.uuid4().hex
    d = os.path.join(_home(), job_id)
    os.makedirs(os.path.join(d, "out"))
    for fname, doc in inputs.items():
        with open(os.path.join(d, fname), "w") as fh:
            json.dump(doc, fh, indent=1)
    env = dict(os.environ, PYTHONUNBUFFERED="1", LOADPATH_NO_HINT="1")
    env["PYTHONPATH"] = os.path.dirname(_HERE) + os.pathsep + env.get("PYTHONPATH", "")
    argv = [a.replace("{dir}", d) for a in argv]
    with open(os.path.join(d, "progress.jsonl"), "wb") as err, open(os.path.join(d, "stdout.json"), "wb") as out:
        proc = subprocess.Popen([sys.executable, "-m", "loadpath", *argv], stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL, env=env, cwd=d)
    _procs[job_id] = proc
    meta = {"kind": kind, "name": name, "pid": proc.pid, "started": time.time(), "parent": parent,
            "baseDir": base_dir}
    with open(os.path.join(d, "job.json"), "w") as fh:
        json.dump(meta, fh)
    out = {"ok": True, "jobId": job_id, "kind": kind, "dir": d, "next": _POLL_HINT}
    if cells is not None:
        out["cells"] = cells
    if warnings:
        out["warnings"] = warnings
    return out


def start(study, base_dir: str | None = None, lattice: dict | None = None) -> dict:
    """Validate and run a study in the background. `lattice` (options, or
    {} for the defaults) also fills the best outcome with a lattice."""
    if isinstance(study, str):
        study = json.loads(study)
    if lattice is not None:
        study = {**study, "lattice": lattice}
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    return _spawn("study", v.get("name") or "study", {"study.json": study},
                  ["run", "{dir}/study.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()],
                  cells=v["cells"], warnings=v["warnings"], base_dir=base_dir or os.getcwd())


def lattice(job_id: str, outcome: int | None = None, options: dict | None = None) -> dict:
    """Fill one outcome of a finished study job with a stress-graded TPMS
    lattice, as a new job."""
    d = _job_dir(job_id)
    if _meta(d).get("kind", "study") != "study":
        raise ValueError("loadpath_lattice takes a finished study job")
    report = os.path.join(d, "out", "report.json")
    if not os.path.exists(report):
        raise ValueError(f"job {job_id} has no outcomes yet; wait until loadpath_status says done")
    with open(report) as fh:
        r = json.load(fh)
    i = r.get("best") if outcome is None else int(outcome)
    if i is None or not 0 <= i < len(r["outcomes"]):
        raise ValueError(f"outcome must be 0..{len(r['outcomes']) - 1}")
    stl = os.path.join(d, "out", os.path.basename(r["outcomes"][i]["stl"]))
    study = json.loads(_read(os.path.join(d, "study.json")))
    from .schema import format_issues, lint
    errors = [x for x in lint(options or {}, "lattice") if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors)}
    base = _meta(d).get("baseDir") or os.getcwd()
    return _spawn("lattice", f"{study.get('name', 'study')}-lattice",
                  {"study.json": study, "lattice.json": options or {}},
                  ["lattice", "{dir}/study.json", stl, "--options", "{dir}/lattice.json", "--out", "{dir}/out",
                   "--base-dir", base], parent=job_id)


def heat_exchanger(spec, base_dir: str | None = None) -> dict:
    if isinstance(spec, str):
        spec = json.loads(spec)
    from .schema import lint
    issues = lint(spec, "heatExchanger")
    errors = [x for x in issues if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": "; ".join(f"{x['path']}: {x['message']}" for x in errors), "issues": issues}
    try:
        from .tasks import _hx_spec
        _hx_spec(spec, base_dir or os.getcwd())
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}
    out = _spawn("heatExchanger", str(spec.get("name") or "heat-exchanger"), {"exchanger.json": spec},
                 ["heatx", "{dir}/exchanger.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()])
    out["warnings"] = [f"{x['path']}: {x['message']}" for x in issues]
    return out


def size_heat_exchanger(spec, base_dir: str | None = None, max_pressure_drop: dict | None = None,
                        min_wall: float = 0.8) -> dict:
    from .tasks import size_heat_exchanger as size
    if isinstance(spec, str):
        spec = json.loads(spec)
    try:
        return size(spec, base_dir or os.getcwd(), max_pressure_drop, min_wall)
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}


def _attention(o: dict, study: dict) -> list[str]:
    """What SKILL.md tells an agent to look for, already looked for."""
    out = []
    if o.get("passes") is False:
        out.append("does not meet the goal: raise the volume fraction or the material, or check the loads")
    if (o.get("bodies") or 1) > 1:
        out.append(f"{o['bodies']} separate bodies: something is not connected to a support")
    if (o.get("keepOutIntrusion") or 0) > 1.0:
        out.append(f"keepOutIntrusion {o['keepOutIntrusion']:.1f} mm³: material in a keep-out (report a bug)")
    if (o.get("preserveMissing") or 0) > 1.0:
        out.append(f"preserveMissing {o['preserveMissing']:.1f} mm³: preserved geometry lost (report a bug)")
    mm = study.get("minMember")
    if mm and o.get("p5Member") is not None and o["p5Member"] < 0.8 * mm:
        out.append(f"p5Member {o['p5Member']:.2f} mm is under minMember {mm} mm: raise the resolution")
    if o.get("closedBores"):
        out.append(f"closed bores: {o['closedBores']}")
    return out


def _png(d, path):
    if not path:
        return None
    try:
        with open(os.path.join(d, "out", os.path.basename(path)), "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _study_status(d, r):
    study = json.loads(_read(os.path.join(d, "study.json")) or "{}")
    outcomes = []
    for i, o in enumerate(r["outcomes"]):
        cases = [{"name": c["name"], "peakStressMPa": c["peakStress"], "safetyFactor": c["safetyFactor"],
                  "deflectionMm": c["deflection"],
                  **({"interlayerStressMPa": c["interlayerStress"]} if c.get("interlayerStress") else {})}
                 for c in o.get("cases") or []]
        row = {
            "outcome": i, "material": o.get("material"), "volumeFraction": o["volumeFraction"],
            "massGrams": o["mass"], "lighterThanSolid": o.get("massReduction"),
            "safetyFactor": o.get("safetyFactor"), "passes": o.get("passes"),
            "minMemberMm": o.get("minMember"), "p5MemberMm": o.get("p5Member"), "bodies": o.get("bodies"),
            "keepOutIntrusionMm3": o.get("keepOutIntrusion"), "preserveMissingMm3": o.get("preserveMissing"),
            "designOverhangFraction": o.get("designOverhangFraction"), "cases": cases,
            "stl": os.path.join(d, "out", os.path.basename(o["stl"])) if o.get("stl") else None,
            "render": os.path.join(d, "out", os.path.basename(o["render"])) if o.get("render") else None,
        }
        row["attention"] = _attention(o, study)
        outcomes.append({k: v for k, v in row.items() if v is not None})
    best = r.get("best")
    info = {"status": "done", "kind": "study", "best": best, "outcomes": outcomes, "warnings": r.get("warnings"),
            "seconds": r.get("seconds"), "report": os.path.join(d, "out", "report.json")}
    if r.get("viewer"):
        info["viewer"] = os.path.join(d, "out", os.path.basename(r["viewer"]))
        info["viewerNote"] = _VIEWER_NOTE
    pngs = [_png(d, r["outcomes"][best].get("render"))] if best is not None else []
    if r.get("lattice"):
        info["lattice"] = r["lattice"]
        pngs.append(_png(d, r["lattice"].get("render")))
    if best is not None:
        info["next"] = ("Look at the render and the attention list. To lighten it further, loadpath_lattice "
                        "fills it with a stress-graded lattice. Once it passes, offer the user a print quote or a "
                        "hosted page for the STL via Materialize (materialize_import_model, "
                        "materialize_get_quote); never order unasked.")
    return info, pngs


def status(job_id: str) -> tuple[dict, list]:
    """(info, [png bytes]) for a job."""
    from .tasks import jsonable

    d = _job_dir(job_id)
    meta = _meta(d)
    kind = meta.get("kind", "study")
    report = os.path.join(d, "out", "report.json")
    lines = [ln for ln in _read(os.path.join(d, "progress.jsonl")).splitlines() if ln.startswith("{")]
    try:
        latest = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        latest = None
    if _alive(job_id, meta):
        el = time.time() - meta["started"] if meta.get("started") else None
        return {"status": "running", "kind": kind, "name": meta.get("name"),
                "elapsedSeconds": round(el) if el else None, "latest": _round(latest), "next": _POLL_HINT}, []
    if not os.path.exists(report):
        err = _read(os.path.join(d, "stdout.json")).strip()
        tail = [ln for ln in _read(os.path.join(d, "progress.jsonl")).splitlines() if ln.strip()][-15:]
        proc = _procs.get(job_id)
        state = "cancelled" if meta.get("cancelled") or (proc is not None and (proc.returncode or 0) < 0) \
            else "failed"
        errs = [ln for ln in tail if ln and not ln.startswith((" ", "{", "Traceback"))]
        out = {"status": state, "kind": kind, "error": err[-3000:] if err else (errs[-1] if errs else "exited early")}
        if not err and tail:
            out["log"] = "\n".join(tail)[-3000:]
        return out, []
    with open(report) as fh:
        r = json.load(fh)
    if kind == "study":
        info, pngs = _study_status(d, r)
    elif kind == "lattice":
        info = {"status": "done", "kind": kind, "parent": meta.get("parent"), **r}
        pngs = [_png(d, r.get("render"))]
    else:
        rep = r.get("report", {})
        keep = ("isolated", "connected", "minWallBetweenFluids", "minOuterWall", "overhangArea", "watertight",
                "solidVolume", "bounds", "warnings", "zones")
        info = {"status": "done", "kind": kind, "passes": r.get("passes"), "stl": r.get("stl"),
                "viewer": r.get("viewer"),
                "fluidSpaces": r.get("fluidSpaces"), "rating": r.get("rating"), "ratingError": r.get("ratingError"),
                "check": {k: rep.get(k) for k in keep if k in rep},
                "report": report}
        for n in ("A", "B"):
            if n in rep and isinstance(rep[n], dict):
                info.setdefault("fluidReport", {})[n] = rep[n]
        rt = r.get("rating")
        if rt:
            info["summary"] = {
                "effectiveness": rt.get("effectiveness"), "dutyW": rt.get("duty"), "tOutC": rt.get("tOut"),
                "pressureDropPa": {n: f.get("pressureDrop") for n, f in rt.get("fluids", {}).items()},
                "flags": sorted({x for f in rt.get("fluids", {}).values() for x in f.get("flags", [])})}
        info = {k: v for k, v in info.items() if v is not None}
        pngs = [_png(d, r.get("render"))]
    return jsonable(info), [p for p in pngs if p]


def _round(o, nd=4):
    if isinstance(o, float):
        return float(f"{o:.{nd}g}")
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def jobs(limit: int = 20) -> list[dict]:
    """Recent jobs, newest first."""
    rows = []
    root = _home()
    for job_id in os.listdir(root):
        d = os.path.join(root, job_id)
        meta = _meta(d)
        if not meta:
            continue
        done = os.path.exists(os.path.join(d, "out", "report.json"))
        st = "running" if _alive(job_id, meta) else "done" if done else \
            "cancelled" if meta.get("cancelled") else "failed"
        rows.append({"jobId": job_id, "kind": meta.get("kind"), "name": meta.get("name"), "status": st,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta.get("started", 0))),
                     "_t": meta.get("started", 0)})
    rows.sort(key=lambda x: -x["_t"])
    return [{k: v for k, v in r.items() if k != "_t"} for r in rows[:limit]]


def cancel(job_id: str) -> dict:
    d = _job_dir(job_id)
    meta = _meta(d)
    if _alive(job_id, meta):
        proc = _procs.get(job_id)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            import signal
            os.kill(meta["pid"], signal.SIGTERM)
        meta["cancelled"] = True
        with open(os.path.join(d, "job.json"), "w") as fh:
            json.dump(meta, fh)
    return {"status": "cancelled"}


def serve():
    try:
        from mcp.server.mcpserver import Image, MCPServer
    except ImportError:  # pragma: no cover - optional extra
        sys.exit('loadpath mcp needs the MCP SDK: pip install "loadpath[mcp]"')

    from .schema import schema as _schema
    from .tasks import example, examples

    server = MCPServer(
        "loadpath",
        instructions=(
            "Generative design (topology optimization) for printed parts, plus lattice infills and two-fluid "
            "TPMS heat exchangers. Describe the problem, not the geometry: design space, preserved interfaces, "
            "keep-outs, fasteners, supports, load cases, material, process, goal. Units: mm, N, MPa. "
            "Start with loadpath_guide (once) or loadpath_examples, validate with loadpath_check until it is "
            "ok with no warnings you can fix, then loadpath_start and poll loadpath_status. Read every "
            "outcome's attention list and look at the render before trusting a part. When a part passes, "
            "offer to get it quoted and printed, or hosted to share, through Materialize (its MCP server: "
            "materialize_import_model, then materialize_get_quote) without ordering unasked."),
    )

    def _reply(info, pngs):
        out = [json.dumps(info, indent=2)]
        out += [Image(data=p, format="png") for p in pngs]
        return out

    @server.tool()
    def loadpath_guide() -> str:
        """How to write a study: the workflow, every field, a worked example. Read once per session."""
        return guide()

    @server.tool()
    def loadpath_schema(kind: str = "study") -> dict:
        """The JSON Schema (2020-12) for a 'study', a 'heatExchanger' or 'lattice' options."""
        try:
            return _schema(kind)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool()
    def loadpath_examples(name: str | None = None) -> dict:
        """With no name: the example documents (studies and heat exchangers) with a line on each.
        With a name: that document, ready to adapt and pass to loadpath_check / loadpath_start."""
        try:
            return example(name) if name else {"examples": examples()}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool()
    def loadpath_check(study: dict, base_dir: str | None = None) -> dict:
        """Validate a study without solving: every schema problem at once (with 'did you mean'), then
        the voxel problem (cells, pitch, regions, warnings). base_dir resolves relative mesh paths
        (default: the server's working directory)."""
        return check(study, base_dir)

    @server.tool()
    def loadpath_start(study: dict, base_dir: str | None = None, lattice: dict | None = None) -> dict:
        """Validate and start a study in the background; returns a jobId for loadpath_status.
        lattice: optional lattice options ({} for defaults) to also fill the best outcome with a
        stress-graded TPMS lattice."""
        return start(study, base_dir, lattice)

    @server.tool()
    def loadpath_status(job_id: str) -> list:
        """A job's progress, or when done: its results and renders. For a study: every outcome (mass,
        safety factor and deflection per load case, pass/fail, keep-out and preserve checks, an
        'attention' list of problems, STL path) and a render of the best one."""
        try:
            return _reply(*status(job_id))
        except ValueError as exc:
            return [json.dumps({"ok": False, "error": str(exc)})]

    @server.tool()
    def loadpath_lattice(job_id: str, outcome: int | None = None, options: dict | None = None) -> dict:
        """Fill an outcome of a finished study job (default: the best) with a TPMS sheet lattice whose
        wall follows the part's stress, keeping a solid skin and solid material where it works hard.
        options: {type: gyroid|diamond|schwarz-p, cell, skin (0 = open, for powder and resin), minWall,
        maxWall, solidAbove, safetyFactor}. Starts a job; poll loadpath_status."""
        try:
            return lattice(job_id, outcome, options)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool()
    def loadpath_heat_exchanger(spec: dict, base_dir: str | None = None) -> dict:
        """Design a two-fluid heat exchanger from a sheet TPMS in any core shape, with self-supporting
        headers and ports, check the fluids are isolated and connected, and rate it if spec.duty is given.
        See loadpath_schema('heatExchanger') and loadpath_examples. Starts a job; poll loadpath_status."""
        return heat_exchanger(spec, base_dir)

    @server.tool()
    def loadpath_size_heat_exchanger(spec: dict, max_pressure_drop: dict | None = None,
                                     min_wall: float = 0.8, base_dir: str | None = None) -> dict:
        """Before designing: sweep TPMS cell and wall for the spec's core and duty and return the
        candidates that meet max_pressure_drop ({fluid name: Pa}), best effectiveness first. Seconds."""
        return size_heat_exchanger(spec, base_dir, max_pressure_drop, min_wall)

    @server.tool()
    def loadpath_jobs(limit: int = 20) -> dict:
        """Recent jobs (studies, lattices, heat exchangers), newest first, with their status."""
        return {"jobs": jobs(limit)}

    @server.tool()
    def loadpath_cancel(job_id: str) -> dict:
        """Stop a running job."""
        return cancel(job_id)

    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    serve()
