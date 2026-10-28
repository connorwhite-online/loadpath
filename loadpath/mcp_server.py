"""
A local MCP server, so any agent (Claude Code, Cursor, Claude Desktop…) can
run generative design studies on the machine it is working on, with no
hosted service in between:

    {"mcpServers": {"loadpath": {"command": "loadpath", "args": ["mcp"]}}}

Tools: loadpath_guide (how to write a study), loadpath_check (validate,
show the voxel problem), loadpath_start / loadpath_status / loadpath_cancel
(run a study as a background job; status returns the outcomes, the STL
paths and a render). Studies run in a child process, the same
`python -m loadpath run` the CLI uses, under LOADPATH_HOME (default
~/.loadpath/jobs).

Needs the `mcp` extra: pip install "loadpath[mcp]".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
_procs: dict[str, subprocess.Popen] = {}


def _home() -> str:
    root = os.environ.get("LOADPATH_HOME") or os.path.join(os.path.expanduser("~"), ".loadpath", "jobs")
    os.makedirs(root, exist_ok=True)
    return root


def _job_dir(job_id: str) -> str:
    if not job_id or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("unknown job id")
    d = os.path.join(_home(), job_id)
    if not os.path.isdir(d):
        raise ValueError(f"no job {job_id} under {_home()}")
    return d


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def guide() -> str:
    return _read(os.path.join(_HERE, "SKILL.md")) + "\n\n---\n\n" + _read(os.path.join(_HERE, "README.md"))


def check(study: dict, base_dir: str | None = None) -> dict:
    from .domain import build_domain
    from .simp import filter_radius
    from .spec import study_from_dict

    try:
        s = study_from_dict(study, base_dir=base_dir or os.getcwd())
        d = build_domain(s)
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "pitch": d.pitch, "cells": d.n_elem, "dims": list(d.dims),
            "preservedCells": int(d.preserve.sum()), "filterRadius": filter_radius(d, s.min_member),
            "warnings": d.warnings}


def start(study: dict, base_dir: str | None = None) -> dict:
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    job_id = uuid.uuid4().hex
    d = os.path.join(_home(), job_id)
    os.makedirs(os.path.join(d, "out"))
    with open(os.path.join(d, "study.json"), "w") as fh:
        json.dump(study, fh)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.path.dirname(_HERE) + os.pathsep + env.get("PYTHONPATH", "")
    with open(os.path.join(d, "progress.jsonl"), "wb") as err, open(os.path.join(d, "stdout.json"), "wb") as out:
        _procs[job_id] = subprocess.Popen(
            [sys.executable, "-m", "loadpath", "run", os.path.join(d, "study.json"),
             "--out", os.path.join(d, "out"), "--base-dir", base_dir or os.getcwd()],
            stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env)
    return {"ok": True, "jobId": job_id, "dir": d, "cells": v["cells"], "warnings": v["warnings"],
            "next": "Poll loadpath_status every 30-60 seconds; a study takes minutes."}


def status(job_id: str) -> tuple[dict, bytes | None]:
    d = _job_dir(job_id)
    report = os.path.join(d, "out", "report.json")
    proc = _procs.get(job_id)
    running = proc is not None and proc.poll() is None
    lines = [ln for ln in _read(os.path.join(d, "progress.jsonl")).splitlines() if ln.startswith("{")]
    latest = json.loads(lines[-1]) if lines else None
    if running:
        return {"status": "running", "latest": latest}, None
    if not os.path.exists(report):
        err = _read(os.path.join(d, "stdout.json")).strip() or (lines[-1] if lines else "")
        state = "cancelled" if proc is not None and proc.returncode and proc.returncode < 0 else "failed"
        return {"status": state, "error": err[-2000:]}, None
    with open(report) as fh:
        r = json.load(fh)
    outcomes = []
    for i, o in enumerate(r["outcomes"]):
        outcomes.append({
            "outcome": i, "volumeFraction": o["volumeFraction"], "massGrams": o["mass"],
            "lighterThanSolid": o.get("massReduction"), "safetyFactor": o.get("safetyFactor"),
            "passes": o.get("passes"), "minMemberMm": o.get("minMember"), "bodies": o.get("bodies"),
            "cases": [{"name": c["name"], "peakStressMPa": c["peakStress"],
                       "safetyFactor": c["safetyFactor"], "deflectionMm": c["deflection"]}
                      for c in o.get("cases") or []],
            "stl": os.path.join(d, "out", os.path.basename(o["stl"])) if o.get("stl") else None,
        })
    best = r.get("best")
    png = None
    if best is not None and r["outcomes"][best].get("render"):
        try:
            with open(os.path.join(d, "out", os.path.basename(r["outcomes"][best]["render"])), "rb") as fh:
                png = fh.read()
        except OSError:
            png = None
    return {"status": "done", "best": best, "outcomes": outcomes, "warnings": r.get("warnings"),
            "seconds": r.get("seconds")}, png


def cancel(job_id: str) -> dict:
    _job_dir(job_id)
    proc = _procs.get(job_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {"status": "cancelled"}


def serve():
    try:
        from mcp.server.mcpserver import Image, MCPServer
    except ImportError:  # pragma: no cover - optional extra
        sys.exit('loadpath mcp needs the MCP SDK: pip install "loadpath[mcp]"')

    server = MCPServer(
        "loadpath",
        instructions=(
            "Generative design (topology optimization) for printed parts. Describe the problem, not the "
            "geometry: design space, preserved interfaces, keep-outs, supports, load cases, material, goal. "
            "Call loadpath_guide once for the study format. Units: mm, N, MPa."),
    )

    @server.tool()
    def loadpath_guide() -> str:
        """How to write a study: workflow, every field, worked example."""
        return guide()

    @server.tool()
    def loadpath_check(study: dict, base_dir: str | None = None) -> dict:
        """Validate a study and report the voxel problem (cells, pitch, warnings) without solving.
        base_dir resolves relative mesh paths (defaults to the server's working directory)."""
        return check(study, base_dir)

    @server.tool()
    def loadpath_start(study: dict, base_dir: str | None = None) -> dict:
        """Start a study in the background. Returns a jobId for loadpath_status."""
        return start(study, base_dir)

    @server.tool()
    def loadpath_status(job_id: str) -> list:
        """A job's progress, or when done: every outcome (mass, safety factor and deflection per load
        case, pass/fail, STL path) and a render of the best one."""
        info, png = status(job_id)
        out = [json.dumps(info, indent=2)]
        if png:
            out.append(Image(data=png, format="png"))
        return out

    @server.tool()
    def loadpath_cancel(job_id: str) -> dict:
        """Stop a running job."""
        return cancel(job_id)

    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    serve()
