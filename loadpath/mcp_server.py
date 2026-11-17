"""
A local MCP server, so any agent (Claude Code, Cursor, Claude Desktop…) can
run generative design studies on the machine it is working on, with no
hosted service in between:

    {"mcpServers": {"loadpath": {"command": "loadpath", "args": ["mcp"]}}}

Every tool is listed, with its parameters, in docs/agents.md (generated from
this file: `python -m loadpath.mcp_server --reference`), and
schema/mcp-tools.json holds the surface tests/test_mcp_surface.py keeps
stable: a tool renamed, a parameter dropped or one made required fails it.

Conventions every tool follows:

- names are loadpath_<noun> or loadpath_<verb>_<noun>, snake_case;
- a file to read is `path`, resolved against `base_dir` (default: the
  server's working directory); a JSON document (study, brief, spec, graph)
  is an object, its JSON text, or a path to a .json file;
- a tool that takes more than a few seconds starts a job and returns
  {ok, jobId, kind, dir, next}; loadpath_status reads any job;
- a tool never raises: a problem comes back as {"ok": false, "tool",
  "error", "hint"?}.

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
_QUICK_NOTE = ("LOADPATH_QUICK is set in the server's environment: a coarse grid and fewer iterations (a first "
               "look; masses and safety factors move by a few to tens of percent). Restart the server without it "
               "for the part to print.")
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


def _failure(tool: str, exc: BaseException) -> dict:
    """What a tool returns instead of raising: the error an agent can act on,
    never a bare stack trace or a dropped connection."""
    msg = str(exc).strip() or type(exc).__name__
    out = {"ok": False, "tool": tool, "error": msg}
    if isinstance(exc, (FileNotFoundError, IsADirectoryError)):
        out["hint"] = "paths resolve against base_dir (default: the server's working directory); pass an absolute path"
    elif not isinstance(exc, (ValueError, KeyError, TypeError)):
        out["hint"] = ("an internal error, not a problem with the input as far as loadpath can tell; "
                       "loadpath_jobs and the job's log have more")
        out["exception"] = type(exc).__name__
    return out


def guarded(fn):
    """Wrap a tool body so any exception comes back as {"ok": false, ...}
    (or a one-item list of it, for tools that return content lists)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001 - the agent gets the error, the server stays up
            f = _failure(fn.__name__, exc)
            return [json.dumps(f)] if fn.__annotations__.get("return") in (list, "list") else f
    return wrapper


# guide topics: SKILL.md sections by heading, README.md sections by heading.
# The default ("core") fits one MCP reply: the whole guide (~60k chars) came
# back as "result exceeds maximum allowed tokens", saved to a file the agent
# then had to page through.
GUIDE_TOPICS = {
    "core": ("SKILL.md", None),
    "heat": ("SKILL.md", ["Heat, fatigue and creep"]),
    "lattice": ("SKILL.md", ["Lattices and heat exchangers"]),
    "multimaterial": ("SKILL.md", ["Multi-material parts"]),
    "example": ("SKILL.md", ["A robot arm link, end to end"]),
    "study": ("README.md", ["The study"]),
    "report": ("README.md", ["Reading the report", "Limits"]),
}
_CORE_SKIP = {"Heat, fatigue and creep", "Lattices and heat exchangers", "Multi-material parts",
              "A robot arm link, end to end"}


def _sections(text: str) -> list[tuple[str, str]]:
    """[(heading, text)] split on '## ' headings; the preamble has heading ''."""
    import re

    out, head, buf = [], "", []
    for line in text.splitlines(keepends=True):
        m = re.match(r"## (.+)", line)
        if m:
            out.append((head, "".join(buf)))
            head, buf = m.group(1).strip(), []
        buf.append(line)
    out.append((head, "".join(buf)))
    return out


def guide(topic: str | None = None) -> str:
    """The agent guide. topic None/"core": SKILL.md's workflow and the rules
    every study needs; "heat", "lattice", "multimaterial", "example", "study"
    (every study field), "report", or "all" (SKILL.md and README.md whole)."""
    topic = (topic or "core").strip().lower()
    if topic == "all":
        return _read(os.path.join(_HERE, "SKILL.md")) + "\n\n---\n\n" + _read(os.path.join(_HERE, "README.md"))
    if topic not in GUIDE_TOPICS:
        raise ValueError(f"unknown guide topic {topic!r}: one of {sorted(GUIDE_TOPICS) + ['all']}")
    fname, heads = GUIDE_TOPICS[topic]
    secs = _sections(_read(os.path.join(_HERE, fname)))
    if heads is None:
        body = "".join(t for h, t in secs if h not in _CORE_SKIP)
        more = ", ".join(f"'{k}'" for k in GUIDE_TOPICS if k != "core")
        return (body.rstrip() + f"\n\n---\nMore in loadpath_guide(topic=...): {more}, or 'all'. "
                "'study' is the reference for every study field (milling, casting, extrusion, fasteners...).\n")
    return "".join(t for h, t in secs if h in heads)


def check(study, base_dir: str | None = None) -> dict:
    from . import quick
    from .tasks import check_study
    out = check_study(study, base_dir)
    if quick.on():
        out["quick"] = _QUICK_NOTE
    return out


def _spawn(kind: str, name: str, inputs: dict, argv: list, cells=None, warnings=None, parent=None,
           base_dir=None) -> dict:
    job_id = uuid.uuid4().hex
    d = os.path.join(_home(), job_id)
    os.makedirs(os.path.join(d, "out"))
    for fname, doc in inputs.items():
        with open(os.path.join(d, fname), "w") as fh:
            json.dump(doc, fh, indent=1)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.path.dirname(_HERE) + os.pathsep + env.get("PYTHONPATH", "")
    argv = [a.replace("{dir}", d) for a in argv]
    with open(os.path.join(d, "progress.jsonl"), "wb") as err, open(os.path.join(d, "stdout.json"), "wb") as out:
        proc = subprocess.Popen([sys.executable, "-m", "loadpath", *argv], stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL, env=env, cwd=d)
    _procs[job_id] = proc
    from . import quick
    meta = {"kind": kind, "name": name, "pid": proc.pid, "started": time.time(), "parent": parent,
            "baseDir": base_dir, "quick": quick.on()}
    with open(os.path.join(d, "job.json"), "w") as fh:
        json.dump(meta, fh)
    out = {"ok": True, "jobId": job_id, "kind": kind, "dir": d, "next": _POLL_HINT}
    if meta["quick"]:
        out["quick"] = _QUICK_NOTE
    if cells is not None:
        out["cells"] = cells
    if warnings:
        out["warnings"] = warnings
    return out


def start(study, base_dir: str | None = None, lattice: dict | None = None, time_limit: float | None = None) -> dict:
    """Validate and run a study in the background. `lattice` (options, or
    {} for the defaults) also fills the best outcome with a lattice.
    `time_limit` (minutes): stop searching then and deliver the best
    outcome verified so far."""
    if isinstance(study, str):
        from .tasks import read_doc
        study, base_dir = read_doc(study, base_dir)
    if lattice is not None:
        study = {**study, "lattice": lattice}
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    return _spawn("study", v.get("name") or "study", {"study.json": study},
                  ["run", "{dir}/study.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd(),
                   *(["--time-limit", str(float(time_limit))] if time_limit else [])],
                  cells=v["cells"], warnings=v["warnings"], base_dir=base_dir or os.getcwd())


def explore(study, base_dir: str | None = None, methods: list | None = None, materials: list | None = None,
            quantity: int | None = None, rank: str | None = None, screen: bool | None = None) -> dict:
    """Validate a study and solve it once per manufacturing method and
    material (explore.py), as a background job."""
    if isinstance(study, str):
        from .tasks import read_doc
        study, base_dir = read_doc(study, base_dir)
    opts = {k: v for k, v in (("methods", methods), ("materials", materials), ("quantity", quantity),
                              ("rank", rank), ("screen", screen)) if v is not None}
    study = {**study, "explore": {**(study.get("explore") or {}), **opts}}
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    from .explore import plan
    try:
        p = plan(study, None, base_dir or os.getcwd())
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    out = _spawn("explore", v.get("name") or "study", {"study.json": study},
                 ["explore", "{dir}/study.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()],
                 cells=v["cells"], warnings=v["warnings"], base_dir=base_dir or os.getcwd())
    out["routes"] = [r["id"] for r in p["routes"]]
    if p["skipped"]:
        out["skipped"] = p["skipped"]
    out["next"] = (f"{len(p['routes'])} full studies: expect several times one study's time. " + _POLL_HINT)
    return out


def switch(study, base_dir: str | None = None, pitches: list | None = None, skin_review_only: bool = False) -> dict:
    """Validate a study with a `switch` block and design its push pad
    (switch.py) as a background job. skin_review_only: read the bare skin
    alone (skinreview.py), nothing designed."""
    if isinstance(study, str):
        from .tasks import read_doc
        study, base_dir = read_doc(study, base_dir)
    if not (study or {}).get("switch"):
        return {"ok": False, "error": "the study needs a switch block (see loadpath_schema('study'), $.switch)"}
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    args = ["switch", "{dir}/study.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    if pitches:
        args += ["--pitches", ",".join(str(float(p)) for p in pitches)]
    if skin_review_only:
        args.append("--skin-review")
    out = _spawn("switch", v.get("name") or "study", {"study.json": study}, args,
                 cells=v["cells"], warnings=v["warnings"], base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def handoff(brief, base_dir: str | None = None, run: bool = False, pitches: list | None = None) -> dict:
    """Validate a designer's handoff brief (handoff.py) and build its
    push-pad study as a background job (the assembly path search takes a
    minute or two); run: then design it (loadpath switch)."""
    if isinstance(brief, str):
        from .tasks import read_doc
        brief, base_dir = read_doc(brief, base_dir)
    from .schema import format_issues, lint
    errors = [x for x in lint(brief, "handoff") if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors)}
    args = ["handoff", "{dir}/brief.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    if run:
        args.append("--run")
        if pitches:
            args += ["--pitches", ",".join(str(float(p)) for p in pitches)]
    out = _spawn("handoff", brief.get("name") or "handoff", {"brief.json": brief}, args,
                 base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def split(study, split_block: dict | None = None, base_dir: str | None = None) -> dict:
    """Cut a study's exterior into a base and a cover (shells.py) as a
    background job: the cut's checks and the lip mesh take a minute."""
    if isinstance(study, str):
        from .tasks import read_doc
        study, base_dir = read_doc(study, base_dir)
    from .schema import format_issues, lint
    block = split_block if split_block is not None else study.get("split")
    if not isinstance(block, dict):
        return {"ok": False, "error": "no split block: give the study one, or split_block"}
    errors = [x for x in lint(block, "split") if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors)}
    study = {**study, "split": block}
    args = ["split", "{dir}/study.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    out = _spawn("split", study.get("name") or "split", {"study.json": study}, args,
                 base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def wires(brief, base_dir: str | None = None) -> dict:
    """Validate a wires brief (wires.py) and lay and check its cables as a
    background job."""
    if isinstance(brief, str):
        from .tasks import read_doc
        brief, base_dir = read_doc(brief, base_dir)
    from .schema import format_issues, lint
    errors = [x for x in lint(brief, "wires") if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors)}
    args = ["wires", "{dir}/brief.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    out = _spawn("wires", brief.get("name") or "wires", {"brief.json": brief}, args,
                 base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def antenna(brief, base_dir: str | None = None) -> dict:
    """Validate an antenna brief (antenna.py) and solve it as a background
    job."""
    if isinstance(brief, str):
        from .tasks import read_doc
        brief, base_dir = read_doc(brief, base_dir)
    from .schema import format_issues, lint
    errors = [x for x in lint(brief, "antenna") if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors)}
    args = ["antenna", "{dir}/brief.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    out = _spawn("antenna", brief.get("name") or "antenna", {"brief.json": brief}, args,
                 base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def cast(path: str, process: str = "sand", alloy: str = "a356", up: str = "+z", pour_temperature=None,
         mold_temperature=None, head=None, risers: bool = True, molds: bool = True,
         base_dir: str | None = None) -> dict:
    """Pour a part in metal (casting.py, castmold.py) as a background job."""
    from .casting import PROCESSES
    from .casting import alloy as _alloy

    path = _resolve(path, base_dir)
    if process not in PROCESSES:
        return {"ok": False, "error": f"process '{process}': one of {sorted(PROCESSES)}"}
    try:
        _alloy(alloy)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    args = ["cast", path, "--process", process, "--alloy", alloy, "--up", up, "--out", "{dir}/out"]
    for flag, v in (("--pour-temperature", pour_temperature), ("--mold-temperature", mold_temperature),
                    ("--head", head)):
        if v is not None:
            args += [flag, str(float(v))]
    if not risers:
        args.append("--no-risers")
    if not molds:
        args.append("--no-molds")
    out = _spawn("cast", os.path.splitext(os.path.basename(path))[0], {}, args, base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def slipcast(path: str, body: str = "porcelain", wall: float = 4.0, solid: bool = False, up: str = "+z",
             casts: int = 5, pressure: float = 0.0, molds: bool = True, base_dir: str | None = None) -> dict:
    """Slip cast a ceramic part (slipcast.py, castmold.py) as a background job."""
    from .slipcast import body as _body

    path = _resolve(path, base_dir)
    try:
        _body(body)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    args = ["slipcast", path, "--body", body, "--wall", str(float(wall)), "--up", up, "--casts", str(int(casts)),
            "--pressure", str(float(pressure)), "--out", "{dir}/out"]
    if solid:
        args.append("--solid")
    if not molds:
        args.append("--no-molds")
    out = _spawn("slipcast", os.path.splitext(os.path.basename(path))[0], {}, args,
                 base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def _resolve(path: str, base_dir: str | None) -> str:
    """A file a tool reads: absolute, or relative to base_dir (default: the
    server's working directory). Raises FileNotFoundError, which comes back
    as {"ok": false, "error", "hint"}."""
    p = os.path.expanduser(str(path))
    p = p if os.path.isabs(p) else os.path.join(base_dir or os.getcwd(), p)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no such file: {p}")
    return p


def materials(name: str | None = None, stock: bool = False) -> dict:
    """The material library: every card (name, stiffness, strength across
    and along layers, density, processes) and the calibrated ones; with a
    name, that card in full; stock: bought members' cards (laminates,
    adhesives) for a study's `stock`."""
    from .calibration import list_cards
    from .material_library import card_names, describe
    from .materials import _ALIASES, MATERIALS

    if stock:
        from .stock import describe_cards
        return {"ok": True, "stock": describe_cards()}
    if name:
        key = name.strip().lower()
        card = describe(_ALIASES.get(key, key))
        if card is None:
            import difflib
            near = difflib.get_close_matches(key, card_names(), 3)
            raise ValueError(f"unknown material {name!r}" + (f"; did you mean {', '.join(near)}?" if near else
                                                             "; loadpath_materials() lists them"))
        out = {"ok": True, "name": key, "card": card}
        if key in list_cards():
            out["calibrated"] = True
            out["note"] = "a calibrated card of this name overrides the library one"
        return out
    rows = [{"name": k, "title": MATERIALS[k].note, "E": MATERIALS[k].E, "strength": MATERIALS[k].strength,
             "strengthZ": MATERIALS[k].strength_z, "density": MATERIALS[k].density,
             "processes": list(MATERIALS[k].processes)} for k in card_names()]
    return {"ok": True, "units": {"E": "MPa", "strength": "MPa", "strengthZ": "MPa (across layers)",
                                  "density": "g/cm^3"},
            "library": rows, "calibrated": list_cards()}


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def export_step(path: str, out: str | None = None, base_dir: str | None = None, faces: int | None = None,
                surface: str = "facets", tolerance: float | None = None) -> dict:
    """A closed mesh as a STEP solid (step.py) as a background job."""
    p = _resolve(path, base_dir)
    if surface not in ("facets", "smooth"):
        raise ValueError("surface is 'facets' or 'smooth'")
    dest = os.path.abspath(os.path.join(base_dir or os.getcwd(), out)) if out else os.path.splitext(p)[0] + ".step"
    args = ["step", p, "--out", dest, "--surface", surface]
    if faces:
        args += ["--faces", str(int(faces))]
    if tolerance is not None:
        args += ["--tolerance", str(float(tolerance))]
    return _spawn("step", _stem(p), {}, args, base_dir=base_dir or os.getcwd())


def export_bundle(study, path: str, out: str | None = None, base_dir: str | None = None) -> dict:
    """A part verified with its fields and written as a results bundle
    (bundle.py) as a background job."""
    if isinstance(study, str):
        from .tasks import read_doc
        study, base_dir = read_doc(study, base_dir)
    p = _resolve(path, base_dir)
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    dest = os.path.abspath(os.path.join(base_dir or os.getcwd(), out)) if out else os.path.splitext(p)[0] + "-bundle"
    return _spawn("bundle", _stem(p), {"study.json": study},
                  ["bundle", "{dir}/study.json", p, "--out", dest, "--base-dir", base_dir or os.getcwd()],
                  base_dir=base_dir or os.getcwd())


def orient(path: str, study=None, process: str = "fdm", threemf: str | None = None,
           base_dir: str | None = None) -> dict:
    """Rank a mesh's build orientations (orient.py) as a background job."""
    p = _resolve(path, base_dir)
    args, inputs = ["orient", p, "--process", process], {}
    if study is not None:
        if isinstance(study, str):
            from .tasks import read_doc
            study, base_dir = read_doc(study, base_dir)
        v = check(study, base_dir)
        if not v["ok"]:
            return v
        inputs["study.json"] = study
        args += ["--study", "{dir}/study.json"]
    if threemf:
        args += ["--3mf", os.path.abspath(os.path.join(base_dir or os.getcwd(), threemf))]
    return _spawn("orient", _stem(p), inputs, args, base_dir=base_dir or os.getcwd())


def buildsim(path: str, process: str = "lpbf", material: str = "ti64", build: str = "+z", plate: float = 10.0,
             compensate: int = 0, nominal: str | None = None, base_dir: str | None = None) -> dict:
    """Simulate printing a part (buildsim.py) as a background job."""
    p = _resolve(path, base_dir)
    if process not in ("lpbf", "fdm", "sls"):
        raise ValueError("process is 'lpbf', 'fdm' or 'sls'")
    args = ["buildsim", p, "--process", process, "--material", material, "--build", build,
            "--plate", str(float(plate)), "--compensate", str(int(compensate)), "--out", "{dir}/out"]
    if nominal:
        args += ["--nominal", _resolve(nominal, base_dir)]
    return _spawn("buildsim", _stem(p), {}, args, base_dir=base_dir or os.getcwd())


def stack(brief, base_dir: str | None = None) -> dict:
    """Validate a stack brief (stack.py) and search its board order and
    gaps as a background job."""
    if isinstance(brief, str):
        from .tasks import read_doc
        brief, base_dir = read_doc(brief, base_dir)
    from .schema import format_issues, lint
    errors = [x for x in lint(brief, "stack") if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors)}
    args = ["stack", "{dir}/brief.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    out = _spawn("stack", brief.get("name") or "stack", {"brief.json": brief}, args,
                 base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def enclosure(brief, base_dir: str | None = None, run: bool = False) -> dict:
    """Validate an enclosure brief (enclosure.py) and place, path and
    compile it as a background job; run: then design it and check the
    delivered part (exterior untouched, load cases verified)."""
    if isinstance(brief, str):
        from .tasks import read_doc
        brief, base_dir = read_doc(brief, base_dir)
    from .schema import format_issues, lint
    errors = [x for x in lint(brief, "enclosure") if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors)}
    args = ["enclosure", "{dir}/brief.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    if run:
        args.append("--run")
    out = _spawn("enclosure", brief.get("name") or "enclosure", {"brief.json": brief}, args,
                 base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def feel(study, base_dir: str | None = None, pitches: list | None = None) -> dict:
    """Validate a study with a `feel` block and design its interior
    (feel.py) as a background job."""
    if isinstance(study, str):
        from .tasks import read_doc
        study, base_dir = read_doc(study, base_dir)
    if not (study or {}).get("feel"):
        return {"ok": False, "error": "the study needs a feel block (see loadpath_schema('study'), $.feel)"}
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    args = ["feel", "{dir}/study.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    if pitches:
        args += ["--pitches", ",".join(str(float(p)) for p in pitches)]
    out = _spawn("feel", v.get("name") or "study", {"study.json": study}, args,
                 cells=v["cells"], warnings=v["warnings"], base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def build(spec, base_dir: str | None = None, quick: bool = False) -> dict:
    """Validate a build spec (a study with interfaces, components,
    targetMass, assert, references) and build its part (build.py) as a
    background job."""
    if isinstance(spec, str):
        from .tasks import read_doc
        try:
            spec, base_dir = read_doc(spec, base_dir)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"spec is not valid JSON (pass the spec, its JSON text, or a path "
                                         f"to a .json file): {exc}"}
    from .build import has_build
    if not isinstance(spec, dict) or not has_build(spec):
        return {"ok": False, "error": "a build spec needs interfaces or components (see loadpath_schema('study'): "
                                      "$.interfaces, $.components, $.targetMass, $.assert, $.references)"}
    from .schema import lint
    issues = lint(spec, "study")
    errors = [i for i in issues if i["level"] == "error"]
    if errors:
        return {"ok": False, "error": "; ".join(f"{i['path']}: {i['message']}" for i in errors), "issues": issues}
    args = ["build", "{dir}/spec.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    if quick:
        args.append("--quick")
    out = _spawn("build", spec.get("name") or "part", {"spec.json": spec}, args,
                 warnings=[f"{i['path']}: {i['message']}" for i in issues], base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def cooling(study, base_dir: str | None = None, pitches: list | None = None) -> dict:
    """Validate a study with a `cooling` block and design its interior
    (cooling.py) as a background job."""
    if isinstance(study, str):
        from .tasks import read_doc
        study, base_dir = read_doc(study, base_dir)
    if not (study or {}).get("cooling"):
        return {"ok": False, "error": "the study needs a cooling block (see loadpath_schema('study'), $.cooling)"}
    v = check(study, base_dir)
    if not v["ok"]:
        return v
    args = ["cooling", "{dir}/study.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()]
    if pitches:
        args += ["--pitches", ",".join(str(float(p)) for p in pitches)]
    out = _spawn("cooling", v.get("name") or "study", {"study.json": study}, args,
                 cells=v["cells"], warnings=v["warnings"], base_dir=base_dir or os.getcwd())
    out["next"] = _POLL_HINT
    return out


def _explore_status(d, r):
    keep = ("id", "passes", "mass", "safetyFactor", "safetyFactorMethod", "maxDeflection", "unitCost", "leadDays",
            "governedBy", "stl", "render", "warnings", "error")
    routes = [{k: x[k] for k in keep if x.get(k) is not None} for x in r.get("routes") or []]
    info = {"status": "done", "kind": "explore", "quantity": r.get("quantity"), "rank": r.get("rank"),
            "ranking": r.get("ranking"), "picks": r.get("picks"), "pareto": r.get("pareto"),
            "crossover": r.get("crossover"), "routes": routes, "skipped": r.get("skipped"),
            "notes": r.get("notes"), "report": os.path.join(d, "out", "report.json"),
            "next": "Show the user the ranking and picks with the renders; every route's folder holds its own "
                    "report.json, STL and render. Costs are estimates: offer a real quote (a print service the "
                    "user has connected) for the route they pick; never order unasked."}
    pngs = []
    for rid in dict.fromkeys((r.get("picks") or {}).values()):
        row = next((x for x in r.get("routes") or [] if x["id"] == rid), None)
        if row and row.get("render"):
            try:
                with open(row["render"], "rb") as fh:
                    pngs.append(fh.read())
            except OSError:
                pass
    return info, pngs


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
        from .tasks import read_doc
        spec, base_dir = read_doc(spec, base_dir)
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


def field(graph, base_dir: str | None = None) -> dict:
    """Validate a field graph and mesh it in the background."""
    if isinstance(graph, str):
        from .tasks import read_doc
        graph, base_dir = read_doc(graph, base_dir)
    from .schema import format_issues, lint
    issues = lint(graph, "field")
    errors = [x for x in issues if x["level"] == "error"]
    if errors:
        return {"ok": False, "error": format_issues(errors), "issues": issues}
    out = _spawn("field", str(graph.get("name") or "field"), {"graph.json": graph},
                 ["field", "{dir}/graph.json", "--out", "{dir}/out", "--base-dir", base_dir or os.getcwd()],
                 base_dir=base_dir or os.getcwd())
    out["warnings"] = [f"{x['path']}: {x['message']}" for x in issues]
    return out


def size_heat_exchanger(spec, base_dir: str | None = None, max_pressure_drop: dict | None = None,
                        min_wall: float | None = None, objective: str | None = None,
                        min_effectiveness: float | None = None, splits: list | None = None,
                        axes: str | None = None) -> dict:
    from .tasks import size_heat_exchanger as size
    if isinstance(spec, str):
        from .tasks import read_doc
        spec, base_dir = read_doc(spec, base_dir)
    try:
        return size(spec, base_dir or os.getcwd(), max_pressure_drop, min_wall, objective, min_effectiveness, splits,
                    axes)
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}


def _load_mesh(path: str, base_dir: str | None = None):
    import trimesh

    full = path if os.path.isabs(path) else os.path.join(base_dir or os.getcwd(), path)
    if not os.path.exists(full):
        raise ValueError(f"no mesh at {full}")
    if full.lower().endswith((".step", ".stp")):
        from .cadimport import load_part
        return load_part(full).mesh
    mesh = trimesh.load(full, force="mesh")
    if not len(getattr(mesh, "faces", [])):
        raise ValueError(f"{full} has no triangles")
    return mesh


def import_cad(path: str, out_dir: str | None = None, base_dir: str | None = None, space: str = "part",
               material: str = "petg", force: float = 100.0, grow: float = 0.0, process: str | None = None,
               support: str | None = None, loads=None, ring: float | None = None) -> dict:
    """A STEP/STL part's bores, bolt patterns, mounting faces and
    interfaces, and with out_dir a draft study written there (cadimport.py)."""
    from .cadimport import import_part

    full = path if os.path.isabs(path) else os.path.join(base_dir or os.getcwd(), path)
    if not os.path.exists(full):
        return {"ok": False, "error": f"no part at {full}"}
    try:
        if out_dir is None:
            return {"ok": True, **import_part(full)}
        out = os.path.abspath(out_dir if os.path.isabs(out_dir) else os.path.join(base_dir or os.getcwd(), out_dir))
        r = import_part(full, out, space=space, material=material, force=force, grow=grow, process=process,
                        support=support, loads=loads, ring=ring)
    except (ValueError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "baseDir": out, **r}


def scan(path: str, out_dir: str | None = None, base_dir: str | None = None, device=None, units: str | None = None,
         scale: float | None = None, up: str | None = None, crop=None, keep: str = "all", reference=None,
         known=None, support=None, loads=None, material: str = "pa12", process: str | None = "sls",
         grow: float = 15.0, confidence: float = 0.95, fit: str = "slip", brief=None, reference_objects=None,
         clamp: str | None = None, mu: float | None = None) -> dict:
    """A scan's features with their measured uncertainty, and with out_dir,
    support and loads (or a brief) a study built round the scanned object
    (scanstudy.py, scanplan.py)."""
    from .scanstudy import scan_to_study
    full = path if os.path.isabs(path) else os.path.join(base_dir or os.getcwd(), path)
    if not os.path.exists(full):
        return {"ok": False, "error": f"no scan at {full}"}
    if isinstance(support, str):
        support = [support]
    if isinstance(loads, list):
        loads = [it if isinstance(it, dict) and "loads" in it else
                 (it["feature"], it["force"]) if isinstance(it, dict) else tuple(it) for it in loads]
    if isinstance(known, dict):
        known = {k: (v if isinstance(v, dict) else {"diameter": float(v)}) for k, v in known.items()}
    try:
        out = None if out_dir is None else os.path.abspath(
            out_dir if os.path.isabs(out_dir) else os.path.join(base_dir or os.getcwd(), out_dir))
        r = scan_to_study(full, out, device=device, units=units, scale=scale, up=up, crop=crop, keep=keep,
                          reference=reference, known=known, support=support, loads=loads, material=material,
                          process=process, grow=grow, confidence=confidence, bar_fit=fit, brief=brief,
                          families=[brief["family"]] if isinstance(brief, dict) and isinstance(brief.get("family"), str)
                          else None, reference_objects=tuple(reference_objects or ("card",)), clamp=clamp, mu=mu)
    except (ValueError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}
    res = {"ok": True, "summary": r["summary"], "features": r["features"], "asks": r.get("asks", []),
           "todo": r.get("todo", []), "scale": r["report"].get("scale"), "device": r["report"].get("device")}
    for k in ("scaledBy", "plan", "clamp"):
        if r.get(k) is not None:
            res[k] = r[k]
    if "studyPath" in r:
        res.update({"baseDir": out, "study": r["studyPath"], "clearance": r["clearance"]})
    return json.loads(json.dumps(res, default=float))


def exterior(path: str, out_dir: str | None = None, base_dir: str | None = None, size: float | None = None,
             scale: float | None = None, up: str | None = None, place: str | None = None,
             pitch: float | None = None, skin: float | None = 1.0, keep: str = "largest") -> dict:
    """Any exterior through the intake (exterior.py): a clean watertight
    skin written to out_dir/exterior.stl, and the report."""
    from .exterior import load_exterior
    full = path if os.path.isabs(path.partition(".py:")[0]) else os.path.join(base_dir or os.getcwd(), path)
    try:
        e = load_exterior(full, size=size, scale=scale, up=up, place=place, pitch=pitch, skin=skin, keep=keep)
    except (ValueError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}
    out = os.path.abspath(out_dir if out_dir and os.path.isabs(out_dir) else
                          os.path.join(base_dir or os.getcwd(), out_dir or "exterior"))
    stl = e.save(out)
    r = dict(e.report)
    r.pop("droppedBodies", None) if len(r.get("droppedBodies", [])) > 5 else None
    return {"ok": bool(r["result"]["watertight"]), "stl": stl, "report": os.path.join(out, "exterior.json"), **r}


_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def _direction(build):
    if build is None:
        return (0.0, 0.0, 1.0)
    if isinstance(build, str):
        if build.lower() not in _AXES:
            raise ValueError(f"build must be one of {sorted(_AXES)} or a vector [x, y, z], not {build!r}")
        return _AXES[build.lower()]
    v = [float(c) for c in build]
    if len(v) != 3 or not any(v):
        raise ValueError("build must be a non-zero vector [x, y, z]")
    return tuple(v)


def check_printability(path: str, min_wall: float = 1.5, overhang_deg: float = 45.0, build=None,
                       pitch: float | None = None, slicer: bool = False, process: str = "fdm",
                       base_dir: str | None = None, material: str | None = None, slice: bool = False) -> dict:
    """Printability of an STL/3MF/OBJ: wall thickness (mm, +/- the grid
    pitch, with the thinnest spots), overhang area, trapped voids. slicer=True
    adds the layer-by-layer read (bridges, islands, support volume, bed).
    process sls/mjf/sla/dlp adds report["process"] (processcheck). slice=True
    (FDM) runs PrusaSlicer headless: report["sliced"] has print time,
    filament and support volume (slicer.py)."""
    try:
        mesh = _load_mesh(path, base_dir)
        direction = _direction(build)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    from .checks.dfm import check_dfm

    spec = {"minWall": float(min_wall), "overhangDeg": float(overhang_deg), "build": direction}
    if pitch:
        spec["pitch"] = float(pitch)
    report = check_dfm(mesh, spec)
    if slicer:
        try:
            from .printcheck import check_print
            report["slicer"] = check_print(mesh, direction, process=process)
        except Exception as exc:  # noqa: BLE001 - the rest of the report stands
            report["slicerError"] = str(exc)
    if slice and str(process).lower() == "fdm":
        try:
            from .materials import material_from_spec
            from .slicer import slice_mesh

            report["sliced"] = slice_mesh(mesh, direction, material_from_spec(material) if material else None)
        except Exception as exc:  # noqa: BLE001 - the rest of the report stands
            report["slicedError"] = f"{type(exc).__name__}: {exc}"
    from .processcheck import RULES

    if str(process).lower() in RULES:
        try:
            from .materials import material_from_spec
            from .processcheck import check_process

            mat = material_from_spec(material) if material else None
            report["process"] = check_process(mesh, str(process).lower(), build=direction, material=mat,
                                              slicer=slicer)
        except Exception as exc:  # noqa: BLE001 - the rest of the report stands
            report["processError"] = str(exc)
    if report.get("ok"):
        from .delivery import MAKE_IT
        report["next"] = MAKE_IT
    return report


def check_isolation(path: str, ports: list, plugs: list | None = None, min_feature: float | None = None,
                    pitch: float | None = None, base_dir: str | None = None) -> dict:
    """Two-fluid isolation of an STL: each circuit's ports in one void, no
    void shared by two circuits, and the thinnest wall between them in mm."""
    try:
        mesh = _load_mesh(path, base_dir)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    from .checks.networks import check_networks

    bad = [p for p in ports or [] if not {"name", "point", "r"} <= set(p)]
    if not ports or bad:
        return {"ok": False, "error": "ports must be a list of {name: '<circuit>_<role>', point: [x, y, z], r: mm}"}
    return check_networks(mesh, ports, pitch=pitch, plugs=plugs, min_feature=min_feature)


def check_component_fit(path: str, components: list, base_dir: str | None = None) -> dict:
    """Does each component (board, clearance, mounting holes, ports) fit the
    part's cavity: placement in air, walls around it, bosses, port cutouts."""
    try:
        mesh = _load_mesh(path, base_dir)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    from .checks.fit import check_fit

    try:
        return check_fit(mesh, {"components": components})
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}


def _attention(o: dict, study: dict) -> list[str]:
    from .delivery import attention
    return attention(o, study)


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
            "start": o.get("start"),
            "durability": {k: (o.get("durability") or {}).get(k) for k in
                           ("safetyFactor", "fatigueSafetyFactor", "creepSafetyFactor", "fatigueDamage",
                            "temperature", "joints")} if o.get("durability") else None,
            "stl": os.path.join(d, "out", os.path.basename(o["stl"])) if o.get("stl") else None,
            "render": os.path.join(d, "out", os.path.basename(o["render"])) if o.get("render") else None,
        }
        row["attention"] = _attention(o, study)
        outcomes.append({k: v for k, v in row.items() if v is not None})
    best = r.get("best")
    deliver = r.get("deliver")
    if deliver:
        deliver = {**deliver, **{k: os.path.join(d, "out", os.path.basename(deliver[k]))
                                 for k in ("stl", "3mf", "render") if deliver.get(k)}}
    info = {"status": "done", "kind": "study", "deliver": deliver, "best": best, "outcomes": outcomes,
            "warnings": r.get("warnings"),
            "seconds": r.get("seconds"), "report": os.path.join(d, "out", "report.json")}
    if r.get("bundle"):
        # the simulation's fields and their meaning, for other viewers and listings
        info["bundle"] = {"dir": os.path.join(d, "out", "bundle"),
                          "manifest": os.path.join(d, "out", "bundle", "manifest.json"),
                          "files": r["bundle"].get("files")}
    if r.get("viewer"):
        info["viewer"] = os.path.join(d, "out", os.path.basename(r["viewer"]))
        info["viewerNote"] = _VIEWER_NOTE
    pngs = [_png(d, r["outcomes"][best].get("render"))] if best is not None else []
    if r.get("lattice"):
        info["lattice"] = r["lattice"]
        pngs.append(_png(d, r["lattice"].get("render")))
    if best is not None:
        info["next"] = ("Look at the render and the attention list. To lighten it further, loadpath_lattice "
                        "fills it with a stress-graded lattice; loadpath_export_step gives CAD a solid, "
                        "loadpath_export_bundle the fields for other viewers. Once it passes, offer the user a "
                        "print quote for the STL (a print service they have connected); never order unasked.")
    return info, pngs


# Jobs whose command writes its own report (not out/report.json) and prints a
# JSON summary on stdout: done is read from that summary. Before 2026-11-17
# loadpath_status looked for out/report.json only, so every one of these
# came back "failed" with the result in its error text.
_SUMMARY_KINDS = {
    "cast": ("castability.json", "Tell the user whether the part pours sound (ok), where the hot spots and "
             "shrinkage are, the risers it needs and the yield; files are the pattern and mold halves to print."),
    "slipcast": ("slipcast.json", "Give the user the time in the mold, the wall the real shape gets, the shrinkage "
                 "and kiln sag, and the mold files to print and pour."),
    "wires": ("wires.json", "Show each cable's tightest bend against its allowed radius, twist, pull on each "
              "connector and voltage drop; channels/ holds each cable's swept solid to keep clear."),
    "antenna": ("antenna.json", "Show S11 and the resonance bare and as installed, how far the surroundings moved "
                "it, bandwidth and efficiency; sensitivity.npz is where plastic or metal detunes it most."),
    "stack": ("stack.json", "Give the user the board order and gaps chosen (best), each board's temperature "
              "against its limit, and what was rejected and why."),
    "handoff": ("handoff.json", "The study is ready: loadpath_switch it (or it ran, with run=true). Relay any "
                "blocked part and its assembly path to the designer; never change the exterior."),
    "split": ("split.json", "Run each half's study (loadpath_start). Relay exteriorChanges to the designer."),
    "enclosure": ("enclosure.json", "The study is ready (loadpath_start it), or it ran (run=true: checks). "
                  "assembly.steps is the assembly a person can follow."),
    "step": (None, "Hand the user the STEP path. valid with one solid is a part CAD opens; volumeError is how far "
             "the solid's volume is from the mesh's."),
    "bundle": (None, "Hand the user the folder: manifest.json lists every file (glTF with stress and safety "
               "factor colours, VTU fields, tradeoff.csv); docs/BUNDLE.md says how to open each."),
    "orient": (None, "Tell the user the best build direction and why (support, overhang, the layer bond under "
               "the loads when a study was given); 3mf is the slicer-ready file when asked for."),
    "buildsim": ("report.json", "Show the distortion and deviation from the design, residual stress and recoater "
                 "risk; printed is the shape as it comes off the plate, compensation the pre-distorted mesh."),
}


def _json_docs(text: str) -> list:
    """Every JSON object a command printed on stdout, in order."""
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        j = text.find("{", i)
        if j < 0:
            return out
        try:
            doc, i = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(doc, dict):
            out.append(doc)


def _summary_status(d, kind, meta) -> dict:
    fname, nxt = _SUMMARY_KINDS[kind]
    docs = _json_docs(_read(os.path.join(d, "stdout.json")))
    if not docs or (docs[-1].get("ok") is False and "error" in docs[-1]):
        tail = [ln for ln in _read(os.path.join(d, "progress.jsonl")).splitlines() if ln.strip()][-15:]
        state = "cancelled" if meta.get("cancelled") else "failed"
        out = {"status": state, "kind": kind,
               "error": docs[-1]["error"] if docs else ([ln for ln in tail if not ln.startswith(" ")] or
                                                         ["exited early"])[-1]}
        if tail:
            out["log"] = "\n".join(tail)[-3000:]
        return out
    info = {"status": "done", "kind": kind, "name": meta.get("name"), **docs[0]}
    if len(docs) > 1:
        info["run"] = docs[1:] if len(docs) > 2 else docs[1]
    rp = os.path.join(d, "out", fname) if fname else None
    if rp and os.path.exists(rp):
        info["report"] = rp
    info["next"] = nxt
    return _round(info)


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
        out = {"status": "running", "kind": kind, "name": meta.get("name"),
               "elapsedSeconds": round(el) if el else None, "latest": _round(latest), "next": _POLL_HINT}
        if kind == "study":
            # the lightest passing part so far (`loadpath status` shows the
            # same): a confirmed one is a part to hand over while the
            # search goes on, minutes into a run that may take many more
            from .__main__ import run_status

            best = run_status(os.path.join(d, "out")).get("bestSoFar")
            if best:
                out["bestSoFar"] = _round(best)
                out["next"] += (" bestSoFar has passed the re-check the delivered part gets: hand it over now if "
                                "the user is waiting; the run goes on looking for a lighter part."
                                if best.get("confirmed") else
                                " bestSoFar has passed its analysis grid only, not the re-check yet.")
        return out, []
    if kind in _SUMMARY_KINDS:
        info = _summary_status(d, kind, meta)
        if meta.get("quick"):
            info["quick"] = _QUICK_NOTE
        return jsonable(info), []
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
    elif kind == "explore":
        info, pngs = _explore_status(d, r)
    elif kind == "switch":
        info = {"status": "done", "kind": kind, **{k: r.get(k) for k in (
            "passes", "presses", "volume", "mass", "mesh", "watertight", "levels", "hold", "clearanceMax",
            "nonlinear", "seconds", "skinReview")}, "report": report,
            "next": "Show the user each press point's fire force against its window (forceGf linear, "
                    "thresholdedForceGf as printed, nonlinearForceGf with large deflection). skinReview lists the "
                    "misses the fixed exterior decides (limitedBy 'skin'), where, and the target, local skin and "
                    "whole-skin changes that would fix each: relay its summary and options to the designer and "
                    "never change the exterior yourself. limitedBy 'interior' needs a finer or longer run, not "
                    "the designer."}
        pngs = []
    elif kind == "feel":
        info = {"status": "done", "kind": kind, **{k: r.get(k) for k in (
            "passes", "feel", "feelOk", "feelDeliveredOk", "presses", "volume", "mass", "mesh", "watertight",
            "levels", "seconds")}, "report": report,
            "next": "Show the user each feel target's reading against its window (stiffness N/mm, and what it "
                    "means: squeeze travel, drop peakG and travel, isolate frequency and transmissibility; "
                    "delivered = read back from the STL; modal = the isolate check with modes). A window the "
                    "exterior alone already misses cannot be met from inside."}
        pngs = []
    elif kind == "cooling":
        info = {"status": "done", "kind": kind, **{k: r.get(k) for k in (
            "passes", "cooling", "feel", "feelOk", "presses", "volume", "mass", "mesh", "watertight",
            "levels", "seconds")}, "report": report,
            "next": "Show the user each component's hottest temperature against its maxTemp and the skin's hot "
                    "spot against its touch limit (cooling.design; delivered = read back from the STL on two "
                    "finer grids, gridChange = how much they differ), and where the heat leaves (heatOut). A "
                    "limit the filled interior still misses needs a change the designer owns: more exterior "
                    "area, vents, a metal spreader, or less power."}
        pngs = []
    elif kind == "build":
        out_dir = os.path.join(d, "out")
        failed = [c for c in r.get("checks", []) if c.get("ok") is False]
        info = {"status": "done", "kind": kind, "ok": r.get("ok"), "name": r.get("name"),
                "part": os.path.join(out_dir, r["part"]) if r.get("part") else None,
                "mass": r.get("mass"), "targetMass": r.get("targetMass"), "failed": failed,
                "checks": [{k: c.get(k) for k in ("id", "ok", "value", "limit")} for c in r.get("checks", [])],
                "images": [os.path.join(out_dir, i) for i in r.get("images", [])],
                "seconds": r.get("seconds"), "report": report,
                "next": ("Every check passed: look at the close-up of each interface and component "
                         "(images) and the reference sheets before handing the part over."
                         if r.get("ok") else
                         "The part failed the checks in `failed`: each says what was measured, the limit and "
                         "where. Change the spec or the CAD the check points at and build again; do not "
                         "patch the mesh.")}
        pngs = []
        for i in r.get("images", []):
            if os.path.basename(i) == "views.png" or "reference" in i:
                try:
                    with open(os.path.join(out_dir, i), "rb") as fh:
                        pngs.append(fh.read())
                except OSError:
                    pass
    elif kind == "lattice":
        info = {"status": "done", "kind": kind, "parent": meta.get("parent"), **r}
        pngs = [_png(d, r.get("render"))]
    elif kind == "field":
        info = {"status": "done", "kind": kind, **r, "report": report}
        att = []
        if r.get("enclosedVoids"):
            att.append(f"{r['enclosedVoids']} void(s) sealed inside the part ({r['enclosedVoidVolume']:.0f} mm^3): "
                       "powder or resin stays trapped there; " +
                       ("see drains.undrained" if r.get("drains") else
                        "give the graph a 'process' (sls, mjf, sla, dlp, lpbf) and it drills drain holes"))
        esc = (r.get("drains") or {}).get("meshEscape") or {}
        if esc.get("narrowExitCount"):
            att.append(f"{esc['narrowExitCount']} chamber(s) open only through gaps under the escape hole: "
                       "raise drains.diameter")
        if att:
            info["attention"] = att
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
    if meta.get("quick"):
        info["quick"] = _QUICK_NOTE
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
        kind = meta.get("kind")
        if kind in _SUMMARY_KINDS:
            docs = _json_docs(_read(os.path.join(d, "stdout.json")))
            done = bool(docs) and not (docs[-1].get("ok") is False and "error" in docs[-1])
        else:
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
        return {"ok": True, "jobId": job_id, "status": "cancelled"}
    return {"ok": False, "error": f"job {job_id} is not running: nothing to cancel (loadpath_status reads it)"}


def resume(job_id: str) -> dict:
    """Continue a study job whose process died partway (a machine restart,
    an out-of-memory kill): `loadpath run --resume` in the job's folder, from
    the optimizations it finished and the last checkpoint of the one it was in."""
    d = _job_dir(job_id)
    meta = _meta(d)
    if _alive(job_id, meta):
        return {"ok": False, "error": "the job is still running"}
    if meta.get("kind") != "study":
        return {"ok": False, "error": f"only study jobs resume (this one is {meta.get('kind')!r})"}
    if os.path.exists(os.path.join(d, "out", "report.json")):
        return {"ok": False, "error": "the job finished: nothing to resume"}
    if not os.path.exists(os.path.join(d, "out", "checkpoint", "run.json")):
        return {"ok": False, "error": "the job left no checkpoint to resume from: start it again"}
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.path.dirname(_HERE) + os.pathsep + env.get("PYTHONPATH", "")
    with open(os.path.join(d, "progress.jsonl"), "ab") as err, open(os.path.join(d, "stdout.json"), "wb") as out:
        proc = subprocess.Popen([sys.executable, "-m", "loadpath", "run", "--resume", os.path.join(d, "out")],
                                stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env, cwd=d)
    _procs[job_id] = proc
    meta.update(pid=proc.pid, resumed=meta.get("resumed", 0) + 1, resumedAt=time.time())
    with open(os.path.join(d, "job.json"), "w") as fh:
        json.dump(meta, fh)
    return {"ok": True, "jobId": job_id, "status": "running", "next": _POLL_HINT}


# Each tool's title and what it does to the machine, for the MCP tool
# annotations clients show and gate on: "read" reads files and returns
# (read-only, repeatable), "write" writes files next to its input and
# returns, "job" starts a background job (writes under LOADPATH_HOME and the
# output folder it names), "stop" ends a running job. No tool reaches the
# network. tests/test_mcp_surface.py holds every registered tool to an entry.
TOOLS = {
    "loadpath_guide": ("Agent guide", "read"),
    "loadpath_schema": ("JSON Schema for a document", "read"),
    "loadpath_examples": ("Example documents", "read"),
    "loadpath_materials": ("Material library", "read"),
    "loadpath_check": ("Validate a study", "read"),
    "loadpath_start": ("Run a study", "job"),
    "loadpath_status": ("Job status and results", "read"),
    "loadpath_jobs": ("List jobs", "read"),
    "loadpath_cancel": ("Cancel a job", "stop"),
    "loadpath_resume": ("Resume a died study job", "job"),
    "loadpath_explore": ("Outcomes per manufacturing route", "job"),
    "loadpath_verify": ("Verify a mesh against a study", "read"),
    "loadpath_lattice": ("Fill an outcome with a lattice", "job"),
    "loadpath_field": ("Mesh a field graph", "job"),
    "loadpath_heat_exchanger": ("Design a heat exchanger", "job"),
    "loadpath_size_heat_exchanger": ("Size a heat exchanger core", "read"),
    "loadpath_import_cad": ("Import a CAD part as a study", "write"),
    "loadpath_scan": ("Design round a scanned object", "write"),
    "loadpath_exterior": ("Clean an exterior into a skin", "write"),
    "loadpath_kicad": ("Read a KiCad board", "read"),
    "loadpath_enclosure": ("Design a product inside its exterior", "job"),
    "loadpath_build": ("Build a part from a spec in one run", "job"),
    "loadpath_handoff": ("Designer handoff to a push-pad study", "job"),
    "loadpath_split": ("Split an exterior into base and cover", "job"),
    "loadpath_switch": ("Design a push pad to a click force", "job"),
    "loadpath_feel": ("Design an interior for feel", "job"),
    "loadpath_cooling": ("Design an interior for cooling", "job"),
    "loadpath_wires": ("Lay and check cables", "job"),
    "loadpath_antenna": ("Solve a board antenna in place", "job"),
    "loadpath_stack": ("Order and space a board stack", "job"),
    "loadpath_cast": ("Metal casting simulation", "job"),
    "loadpath_slipcast": ("Ceramic slip casting simulation", "job"),
    "loadpath_buildsim": ("Simulate printing a part", "job"),
    "loadpath_orient": ("Rank build orientations", "job"),
    "loadpath_check_printability": ("Check a mesh is printable", "read"),
    "loadpath_check_isolation": ("Check two fluids stay apart", "read"),
    "loadpath_check_fit": ("Check components fit an enclosure", "read"),
    "loadpath_export_step": ("Export a mesh as STEP", "job"),
    "loadpath_export_bundle": ("Export a results bundle", "job"),
}

INSTRUCTIONS = (
    "Generative design (topology optimization) for printed parts, plus lattice infills and two-fluid "
    "TPMS heat exchangers. The user says what they want built; you work out the problem, not the geometry: "
    "design space, preserved interfaces, keep-outs, fasteners, supports, load cases, material, process, "
    "goal. Read interfaces from their CAD (loadpath_import_cad) and loads from what the part does; fill "
    "gaps with stated assumptions instead of asking for numbers they don't have. For a printed part, set "
    "minMember / process.minFeature / skin to at least the process's minimum wall and detail: the "
    "printer's or print service's published limits when you have them, else the run report's defaults. "
    "Give each hole something goes into a fit (press for a bearing, slip for a pin, clearance for a "
    "turning shaft): the hole is drawn for the process and deliver.fits says what it measures as built. "
    "Units: mm, N, MPa. Start with loadpath_guide (once) or loadpath_examples, validate with "
    "loadpath_check until it is ok with no warnings you can fix, then loadpath_start and poll "
    "loadpath_status (wait=... blocks for you). Read every outcome's attention list and look at the "
    "render before trusting a part. Any tool that takes more than seconds returns a jobId: "
    "loadpath_status reads every kind of job. When a part passes, offer to get it made (a print service "
    "the user has connected can quote the STL) without ordering unasked. For a mesh you already have, "
    "loadpath_check_printability, loadpath_check_isolation, loadpath_check_fit and loadpath_verify "
    "check it directly.")


def _annotations(name):
    from mcp.types import ToolAnnotations

    title, effect = TOOLS[name]
    return ToolAnnotations(title=title, readOnlyHint=effect == "read", destructiveHint=effect == "stop",
                           idempotentHint=effect in ("read", "write", "stop"), openWorldHint=False)


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("loadpath")
    except Exception:  # noqa: BLE001 - a source checkout with no install
        return ""


def build_server():
    """The MCP server with every tool registered (serve() runs it on stdio;
    the tests list its tools)."""
    from mcp.server.mcpserver import Image, MCPServer

    from .schema import schema as _schema
    from .tasks import example, examples

    server = MCPServer("loadpath", title="loadpath", instructions=INSTRUCTIONS, version=_version(),
                       website_url="https://github.com/connorwhite-online/loadpath")

    def tool(fn):
        return server.tool(title=TOOLS[fn.__name__][0], annotations=_annotations(fn.__name__))(guarded(fn))

    def _reply(info, pngs):
        out = [json.dumps(info, indent=2)]
        out += [Image(data=p, format="png") for p in pngs]
        return out

    @tool
    def loadpath_guide(topic: str | None = None) -> str:
        """How to turn a request into a study: the workflow and the rules every study needs. Read once per
        session. topic for more: 'study' (every study field: milling, casting, fasteners...), 'lattice'
        (lattices, TPMS, heat exchangers), 'heat' (heat, fatigue, creep), 'multimaterial', 'example'
        (an arm link end to end), 'report', or 'all'."""
        try:
            return guide(topic)
        except ValueError as exc:
            return str(exc)

    @tool
    def loadpath_schema(kind: str = "study") -> dict:
        """The JSON Schema (2020-12) for a document a tool takes: 'study' (loadpath_check, loadpath_start
        and every study tool), 'heatExchanger', 'lattice' (options), 'field' (a field graph), 'handoff',
        'split', 'enclosure', 'wires', 'antenna', 'stack', or 'bundle' (a results bundle's manifest)."""
        return _schema(kind)

    @tool
    def loadpath_examples(name: str | None = None) -> dict:
        """With no name: the example documents (studies and heat exchangers) with a line on each.
        With a name: that document, ready to adapt and pass to loadpath_check / loadpath_start."""
        return example(name) if name else {"examples": examples()}

    @tool
    def loadpath_check(study: dict | str, base_dir: str | None = None) -> dict:
        """Validate a study without solving: every schema problem at once (with 'did you mean'), then
        the voxel problem (cells, pitch, regions, warnings). study: the object, its JSON text, or a path
        to a .json file. base_dir resolves relative mesh paths (default: a study file's folder, else
        the server's working directory)."""
        return check(study, base_dir)

    @tool
    def loadpath_start(study: dict | str, base_dir: str | None = None, lattice: dict | None = None,
                       time_limit: float | None = None) -> dict:
        """Validate and start a study (object, JSON text, or .json path) in the background; returns a
        jobId for loadpath_status.
        lattice: optional lattice options ({} for defaults) to also fill the best outcome with a
        stress-graded TPMS lattice. time_limit: minutes; past it the search stops and delivers the
        best outcome verified so far (a safety-factor search takes 10 minutes to a few hours)."""
        return start(study, base_dir, lattice, time_limit)

    @tool
    def loadpath_status(job_id: str, wait: float = 0) -> list:
        """A job's progress, or when done: its results and renders. For a study: every outcome (mass,
        safety factor and deflection per load case, pass/fail, keep-out and preserve checks, an
        'attention' list of problems, STL path) and a render of the best one.
        wait: seconds to block while the job is still running (up to 540), so one call waits for it
        instead of many polls; keep calling until status is not 'running'."""
        t_end = time.time() + min(max(float(wait or 0), 0.0), 540.0)
        while time.time() < t_end and _alive(job_id, _meta(_job_dir(job_id))):
            time.sleep(min(5.0, max(0.1, t_end - time.time())))
        return _reply(*status(job_id))

    @tool
    def loadpath_explore(study: dict | str, base_dir: str | None = None, methods: list | None = None,
                         materials: list | None = None, quantity: int | None = None,
                         rank: str | None = None, screen: bool | None = None) -> dict:
        """Like Fusion's generative outcomes: solve one study for each manufacturing method and material
        (methods: fdm, sls, mjf, sla, dlp, jetting, volumetric, lpbf, cast, milled-2.5axis, milled-3axis,
        milled-5axis; default fdm, mjf, sla, lpbf, cast, milled-3axis, each in its usual material) and
        rank them by mass, cost or leadTime with Pareto fronts. Each route is a full verified run plus a
        cost and lead-time estimate at `quantity` parts. screen=true runs every route coarse first and the
        full runs only for the Pareto fronts and top of the ranking. Starts a job; poll loadpath_status."""
        return explore(study, base_dir, methods, materials, quantity, rank, screen)

    @tool
    def loadpath_switch(study: dict | str, base_dir: str | None = None, pitches: list | None = None,
                        skin_review_only: bool = False) -> dict:
        """Design everything inside a fixed exterior so a press anywhere on a touch area fires a switch
        at a chosen force, with the least material: a study with a thin `skin` of the exterior mesh,
        components as keepOuts (`insertFrom` keeps their assembly paths clear) and a `switch` block
        (plunger output with gap, trigger force and travel; press points or a radial field of force
        windows; skin clearance; hold cases). pitches: coarse to fine grids in mm (start [1.0]).
        Reports each press's fire force linear, as printed and with large deflection, the part's
        STL, and skinReview: the presses the fixed exterior itself keeps out of window, where, and the
        skin or target change that would fix each, for the designer to decide (never edit the
        exterior). skin_review_only: read the bare skin alone first (minutes), nothing designed.
        Starts a job; poll loadpath_status."""
        return switch(study, base_dir, pitches, skin_review_only)

    @tool
    def loadpath_handoff(brief: dict | str, base_dir: str | None = None, run: bool = False,
                         pitches: list | None = None) -> dict:
        """A designer's handoff to a push-pad study: the exterior surfacing and the component CAD
        (STL/STEP paths or shapes) in, with the assembly order, the switch's trigger and travel, and the
        press windows (loadpath_schema('handoff')). Out: a linted study with every component a keep-out,
        a clearance per component (each node's own room to what is under it), insertFrom found from the
        assembly order (one axis, or slide-then-drop; a part with no clear way in is reported blocked,
        where it hits), supports where the part stands, the plunger tip read off the switch's actuator
        and the part the switch bears on (its give counts in the click force). The exterior is never
        changed. run: then design it, as loadpath_switch. Starts a job; poll loadpath_status, the
        study is out/<name>.study.json and the report out/handoff.json."""
        return handoff(brief, base_dir, run, pitches)

    @tool
    def loadpath_split(study: dict | str, split_block: dict | None = None, base_dir: str | None = None) -> dict:
        """Cut a study's fixed exterior into a base and a cover that go together round its components
        (loadpath_schema('split')): the cut given or searched across x, y, z (shortest seam where every
        component goes into its half with the other off and the cover comes down over them, missing the
        designer's avoid surfaces), then an alignment lip, screw posts (only through a face the designer
        names in screws.from; each hole reported) and cantilever snaps sized to the material's strain
        allowance, all inside the skin. Out: out/<name>-base.study.json, out/<name>-cover.study.json and
        out/split.json (the cut, paths, each joint's numbers, exteriorChanges). Starts a job; poll
        loadpath_status, then run each half."""
        return split(study, split_block, base_dir)

    @tool
    def loadpath_enclosure(brief: dict | str, base_dir: str | None = None, run: bool = False) -> dict:
        """A product from its outside in: the designer's exterior (kept as a skin, never edited) and a
        list of components (CAD paths, shapes or sizes, with masses and watts) in
        (loadpath_schema('enclosure')). Places each component inside the shell unless given (quarter-turn
        orientations; rules: against a side, alignTo an opening, under a press point, near another part,
        a centre of mass), keeps only positions each part can be assembled into, finds every part's way
        in (up to four straight legs, or, where none exists, a tilt: slides and pivots up to the brief's
        `tilt` degrees, every move checked exactly), puts a cradle under each massed part clear of
        every path, and writes a study with drop cases on each face, squeezes, presses, heat, and any
        switch or feel block. enclosure.json's assembly.steps is the assembly as numbered steps a person
        can follow. run: then design it and check the delivered part (the exterior untouched
        on the meshes, every load case verified on the part). Starts a job; poll loadpath_status; the
        study is out/<name>.study.json, the report out/enclosure.json, checks out/checks.json."""
        return enclosure(brief, base_dir, run)

    @tool
    def loadpath_build(spec: dict | str, base_dir: str | None = None, quick: bool = False) -> dict:
        """Make a printable part in one run from a spec, checked: a study (loadpath_schema('study')) whose
        designSpace is the exterior (kept exactly), with `skin` (the wall, and zones of other walls),
        `interfaces` (the features that must be exact, as CAD solids: {"type": "brep", "path":
        "cad.py:name"} is a build123d solid in a Python file beside the spec, or a STEP file; op add |
        cut), `components` (the bought parts as solids where they sit, with clearance, mass, keepIn and
        insertFrom), `targetMass` (grams, reached by moving the walls named in adjust, heavier or
        lighter), `assert` (features measured every build: fillet radius of an interface, wall,
        clearance, keepIn, mass, distance) and `references` (concept images with their cameras, or a
        reference solid). The body between the interfaces is filled from the exterior and the skin and
        blended into each add with a fillet; cuts and pockets are exact booleans, so drawn edges and
        rounds arrive as drawn. No loads or supports needed. Out: out/part.stl, out/checks.json,
        out/report.json, a close-up sheet of every interface and component and a side-by-side per
        reference. ok only when every check passes. Starts a job; poll loadpath_status."""
        return build(spec, base_dir, quick)

    @tool
    def loadpath_wires(brief: dict | str, base_dir: str | None = None) -> dict:
        """Lay and check cables as elastic rods (loadpath_schema('wires')): each wire, bundle, jacketed
        cable, coax or flat cable (FFC/FPC) from its construction or its datasheet, between two ports
        (where the connector is and which way the cable leaves it), among `bodies` it may touch but
        not enter. Finds a way through the free space, lays the cable at its cut length (or cuts it to
        the route plus slack), and reports what a harness engineer asks: tightest bend against the
        cable's allowed radius, twist (a flat cable cannot turn in its own plane), whether it is too
        short, pinched, how hard it pulls on each connector, voltage drop and self-heating. With
        `motion` (a joint's revolute or sliding range) the cable is followed through the whole travel:
        flexing radius, the strain swing that fatigues it, coming taut, snapping through. Writes the
        report (out/wires.json), each cable's swept channel as a solid to keep clear
        (out/channels/<id>.stl) and a viewer (out/wires.html). Inside a product, give the same wires
        in an enclosure brief instead (components[].ports, wires): then only layouts that can be wired
        are kept. Starts a job; poll loadpath_status."""
        return wires(brief, base_dir)

    @tool
    def loadpath_antenna(brief: dict | str, base_dir: str | None = None) -> dict:
        """Solve a board antenna in what surrounds it (loadpath_schema('antenna')): the board's copper
        as drawn (antenna, feed line, ground), its laminate, and `surroundings` in the board's frame
        (the printed shell as a mesh with its material, a battery or motor as metal, screws, cable
        runs). Maxwell's equations are solved in the time domain on that geometry; nothing is tuned by
        rule. Reports, for the bare board and as installed: reflection (S11) across the sweep, the
        resonance and how far the surroundings moved it, worst S11 and mismatch loss inside the band,
        bandwidth, input impedance, radiation and total efficiency, and the heat each material takes.
        Also writes the sensitivity field (out/sensitivity.npz): the change in port impedance per mm3
        of plastic or metal put at each point, which is the antenna's keep-out as the fields give it.
        Printed plastics carry measured permittivity and loss (PLA, PETG, ABS, TPU; PA12 is an
        estimate until measured); `fill` gives infill or a lattice its effective value. Inside a
        product, put the same block on a component of an enclosure brief (`components[].antenna`):
        placement then ranks every layout by its effect on the port and the chosen one is solved in
        full. Starts a job (a minute or two); poll loadpath_status; report out/antenna.json, sweep
        out/s11.csv."""
        return antenna(brief, base_dir)

    @tool
    def loadpath_cast(path: str, process: str = "sand", alloy: str = "a356", up: str = "+z",
                      pour_temperature: float | None = None, mold_temperature: float | None = None,
                      head: float | None = None, risers: bool = True, molds: bool = True,
                      base_dir: str | None = None) -> dict:
        """Pour a part in metal and see whether it comes out sound. `path` is the part as drawn
        (.stl, mm); process is sand, investment, plaster, permanent or die; alloy is a356, a380, wcb
        (cast carbon steel) or silicon-bronze; `up` is which way is up in the mold. The heat equation
        with latent heat is solved in the part and its mold, then mass conservation and gravity decide
        where the shrinkage goes: hot spots, the cavities they leave, and risers sized by solving until
        the part is sound (position, diameter, pipe depth, yield). Also microporosity (dimensionless
        Niyama criterion), dendrite arm spacing, whether the thinnest wall fills before the stream
        freezes, pattern scale from the alloy's contraction, and sprue and ingate areas. Writes
        out/castability.json and, to print, the casting tree, the split pattern and the mold halves
        (STL). Nothing is a rule of thumb; docs/CASTING.md lists what each number was checked against.
        In a study, `casting: {draw, process, alloy}` pours the optimized part the same way. Starts a
        job (under a minute); poll loadpath_status."""
        return cast(path, process, alloy, up, pour_temperature, mold_temperature, head, risers, molds, base_dir)

    @tool
    def loadpath_slipcast(path: str, body: str = "porcelain", wall: float = 4.0, solid: bool = False,
                          up: str = "+z", casts: int = 5, pressure: float = 0.0, molds: bool = True,
                          base_dir: str | None = None) -> dict:
        """Slip cast a ceramic part in a plaster mold. `path` is the fired part's outside (.stl, mm);
        body is porcelain, stoneware, earthenware or alumina; `wall` is the fired wall in mm (drain
        casting) or solid=true. Filtration through the growing cake gives the time in the mold and the
        wall the real shape gets (thicker in inside corners, slip trapped behind necks); conservation
        of solids gives drying and firing shrinkage and the mold scale; a creep solve under self-weight
        gives the kiln sag; Weibull statistics give the chance the green part breaks coming out of the
        mold. Writes out/slipcast.json and, to print or pour, the plaster mold halves at mold scale,
        the boxes to pour them in, and the fired part with its real inside. Starts a job (under a
        minute); poll loadpath_status."""
        return slipcast(path, body, wall, solid, up, casts, pressure, molds, base_dir)

    @tool
    def loadpath_stack(brief: dict | str, base_dir: str | None = None) -> dict:
        """Order and space a stack of boards on standoffs (loadpath_schema('stack')): a flight controller
        over a speed controller, a carrier and its modules. Every order and gap is solved as a thermal
        network (the air speed each gap really gets from the stream past it, developing-flow convection,
        radiation between facing boards, conduction down the standoffs), scored by how much of each
        board's temperature limit it uses, the height and the centre of mass, and the cables between
        the boards (`wires`, ports in each board's frame) are laid as elastic rods: a gap too small for
        its harness to bend in is refused, with that reason, in `rejected`. The result's
        `inertiaRelief.masses` entry (the stack's mass at its centre of mass, on the `mount` region)
        goes into the carrying part's study, `envelope` is its keep-out box, and `best.z` is where each
        board sits, for a wires brief of the leads that leave the stack. Starts a job; poll
        loadpath_status; the report is out/stack.json."""
        return stack(brief, base_dir)

    @tool
    def loadpath_kicad(path: str, base_dir: str | None = None, power: dict | None = None,
                       heights: dict | None = None, models: list | None = None) -> dict:
        """Read a KiCad board (`path`: a .kicad_pcb) the way an enclosure or a stack needs it: the Edge.Cuts outline
        with its cut-outs, the stackup thickness, every footprint's body (its 3D model when the file is
        found, else its fab outline at the package's standard height, each with heightFrom), its mass
        (massFrom), the mounting holes and their screw sizes, each connector and which way it faces (out
        over the board edge it overhangs; USB-C, USB-A and micro-B typed for their plug loads) and any
        heat (a footprint Power field, or `power` {ref: W}). Returns the summary; in an enclosure brief
        give the component {"id": "board", "kicad": {"file", "connectors": {"J1": {"opening": "usb"}},
        "power"}} and in a stack board {"id", "kicad"} to use it directly. Reads only; quick."""
        from .kicad import load_board, _public
        b = load_board({"kicad": {"file": _resolve(path, base_dir), "power": power or {},
                                  "heights": heights or {}, "models": models or []}})
        pb = _public(b)
        return {"ok": True, "size": pb["size"], "thicknessFrom": pb["thicknessFrom"], "massKg": pb["mass"],
                "conductivity": pb["conductivity"], "holes": pb["holes"],
                "parts": [{k: p[k] for k in ("ref", "footprint", "side", "height", "heightFrom", "mass", "massFrom",
                                             "connector", "watts", "maxTemp") if k in p} for p in pb["parts"]],
                "warnings": pb["warnings"]}

    @tool
    def loadpath_feel(study: dict | str, base_dir: str | None = None, pitches: list | None = None) -> dict:
        """Design everything inside a fixed exterior for how the product feels, with the least material:
        a study with a thin `skin`, components as keepOuts (`insertFrom` keeps assembly paths clear) and
        a `feel` block of targets, each a stiffness window: `squeeze` (a grip gives travel [min, max] mm
        at force N between hold and press), `drop` (a part of kg on its mount, dropped height m onto a
        face, sees under maxG and moves under maxTravel mm), `isolate` (handling vibration from the hand
        reaches a part's mount at under maxTransmissibility from band Hz up), `touch` (a zone of the
        exterior under a fingertip: feel pliable|firm|rigid or a stiffness window; feel.contrast makes
        zones distinctly different; skinThickness "auto" thins the skin under a pliable zone, never under
        the process minimum wall, flagged when it would need to be). A `switch` block in the
        same study is designed with it. No supports or load cases needed. Reports each target on the
        design, as printed and read back from the STL, and a modal check of isolate targets. Starts a
        job; poll loadpath_status."""
        return feel(study, base_dir, pitches)

    @tool
    def loadpath_cooling(study: dict | str, base_dir: str | None = None, pitches: list | None = None) -> dict:
        """Design everything inside a fixed exterior around the heat its electronics make, with the least
        material: a study with a thin `skin`, components as keepOuts and a `cooling` block: components
        [{region, watts, maxTemp, conductivity}], skin {maxTemp | contact: continuous|held|brief, shape
        where touched}, ambient, vents (openings the exterior already has; stack flow). Grows pads and
        ribs that carry heat to the skin and spread it. `feel` and `switch` blocks in the same study are
        designed with it. No supports or load cases needed. Reports temperatures on the design, as
        printed, and read back from the STL on two finer grids. Starts a job; poll loadpath_status."""
        return cooling(study, base_dir, pitches)

    @tool
    def loadpath_lattice(job_id: str, outcome: int | None = None, options: dict | None = None) -> dict:
        """Fill an outcome of a finished study job (default: the best) with a TPMS sheet lattice whose
        wall follows the part's stress, keeping a solid skin and solid material where it works hard.
        options: {type: gyroid|diamond|schwarz-p, cell, skin (0 = open, for powder and resin), minWall,
        maxWall, solidAbove, safetyFactor}. Starts a job; poll loadpath_status."""
        return lattice(job_id, outcome, options)

    @tool
    def loadpath_heat_exchanger(spec: dict | str, base_dir: str | None = None) -> dict:
        """Design a two-fluid heat exchanger from a sheet TPMS in any core shape, with self-supporting
        headers and ports, check the fluids are isolated and connected, and rate it if spec.duty is given.
        See loadpath_schema('heatExchanger') and loadpath_examples. Starts a job; poll loadpath_status."""
        return heat_exchanger(spec, base_dir)

    @tool
    def loadpath_field(graph: dict | str, base_dir: str | None = None) -> dict:
        """Mesh an implicit part described as a graph of fields: shapes and meshes as signed
        distances, simulation results and imported scalar fields, remap/expr math, and field-driven
        operations (variable shell walls, variable fillets, TPMS lattices with graded cell size and wall).
        Any numeric parameter can be another node. With a 'process' (sls, mjf, sla, dlp, volumetric, lpbf)
        sealed voids (a skin over a lattice) get drain holes at the calmest flat sites, on bosses, verified
        clear (report 'drains'). See loadpath_schema('field') and examples/fields/arm-link-graded.json.
        Starts a job; poll loadpath_status."""
        return field(graph, base_dir)

    @tool
    def loadpath_size_heat_exchanger(spec: dict, max_pressure_drop: dict | None = None,
                                     min_wall: float | None = None, base_dir: str | None = None,
                                     objective: str | None = None, min_effectiveness: float | None = None,
                                     splits: list | None = None, axes: str | None = None) -> dict:
        """Before designing: sweep TPMS cell and wall (and, with splits, the first fluid's share of the
        void, e.g. [0.5, 0.6, 0.66] for air against a liquid) for the spec's core and duty. Returns the
        candidates that meet max_pressure_drop ({fluid name: Pa}) and min_effectiveness, best
        effectiveness first, or with objective "mass" the lightest core ("pressure": the least pressure
        drop). min_wall defaults to the process minimum; axes "xy"/"yx" pins which fluid runs along x.
        Only candidates the design step can mesh are returned (each with its grid). Each carries
        solidVolume (mm^3) and coreMass (g, the core only: shell, headers and ports add more). Seconds."""
        return size_heat_exchanger(spec, base_dir, max_pressure_drop, min_wall, objective, min_effectiveness, splits,
                                   axes)

    @tool
    def loadpath_check_printability(path: str, min_wall: float = 1.5, overhang_deg: float = 45.0,
                                    build: str | list | None = None, pitch: float | None = None,
                                    slicer: bool = False, process: str = "fdm",
                                    base_dir: str | None = None, material: str | None = None,
                                    slice: bool = False) -> dict:
        """Printability of any mesh file (STL, 3MF, OBJ), not only loadpath's own parts: wall thickness
        in mm with its +/- (the voxel pitch) and the thinnest spots, overhang area for the build
        direction ('+z'... or [x, y, z]), and trapped voids that cannot drain powder or resin.
        minWallOk None means too coarse to judge, not a pass. slicer=True adds a layer-by-layer read
        (bridges, islands, support volume, bed contact). process 'sls'/'mjf' (powder nylon, TPU) or
        'sla'/'dlp' (resin) adds the process checks: walls against its minimum, powder or resin trapped
        in cavities or behind exits under the escape-hole size, warp-prone thin flat panels (powder),
        peel area, suction cups and supports (resin); material (e.g. 'pa12', 'tpu', 'resin-tough')
        flags a card that is not made by that process. slice=True (FDM) slices it headless with
        PrusaSlicer: 'sliced' gives print time (s), filament (cm^3, g with material) and support volume.
        Seconds to a minute on multi-million-triangle parts."""
        return check_printability(path, min_wall, overhang_deg, build, pitch, slicer, process, base_dir, material,
                                  slice)

    @tool
    def loadpath_check_isolation(path: str, ports: list, plugs: list | None = None,
                                 min_feature: float | None = None, pitch: float | None = None,
                                 base_dir: str | None = None) -> dict:
        """Check that a two-fluid part (heat exchanger, manifold) keeps its fluids apart. ports: probe
        spheres [{name: 'a_in', point: [x, y, z], r}] inside each fluid; ports sharing the prefix before
        '_' are one circuit. plugs: port openings to virtually close, capsules {a, b, r} or boxes {c, h},
        or every circuit meets in the outside air. min_feature: the thinnest wall or channel (mm), so the
        grid resolves it (else the verdict is 'inconclusive'). Returns isolated, the void each port
        found, and minWallMm +/- minWallToleranceMm between circuits."""
        return check_isolation(path, ports, plugs, min_feature, pitch, base_dir)

    @tool
    def loadpath_check_fit(path: str, components: list, base_dir: str | None = None) -> dict:
        """Check that components fit an enclosure mesh. components: [{id, boardMm: [L, W, T],
        clearanceMm, aboveMm, belowMm, holes: {positions: [[x, y]], diaMm} | null, ports: [{id, edge:
        '-x', offsetMm, heightMm, wMm, hMm} | {id, face: '+z', xMm, yMm, wMm, hMm}]}], in the
        component's frame (origin at board centre, z up). Solves the placement in the cavity and
        reports cavity, bosses and each port cutout as ok / not / null (could not check)."""
        return check_component_fit(path, components, base_dir)

    @tool
    def loadpath_scan(path: str, out_dir: str | None = None, base_dir: str | None = None,
                      device: str | dict | None = None, units: str | None = None, scale: float | None = None,
                      up: str | None = None, crop: list | None = None, keep: str = "all",
                      reference: list | None = None, known: dict | None = None, support: list | str | None = None,
                      loads: list | None = None,
                      material: str = "pa12", process: str | None = "sls", grow: float = 15.0,
                      confidence: float = 0.95, fit: str = "slip", brief: dict | None = None,
                      reference_objects: list | None = None, clamp: str | None = None,
                      mu: float | None = None) -> dict:
        """A scan in, a part designed round the scanned object out. Reads point clouds (PLY, PCD, LAS, E57,
        XYZ), Gaussian splats (3DGS .ply, .splat) and scanned meshes; measures each point's noise, spacing
        and curvature; fits planes (mounting faces, a wall) and cylinders (bores; bars and tubes a clamp
        goes round) with RANSAC + least squares, each with a +- band at `confidence` from the fit, the
        device's published accuracy (device: iphone-lidar, ipad-lidar, artec-eva, artec-leo, leica-rtc360,
        leica-blk360, faro-focus, or {sigma, scale}) and the scale. units is required unless the format
        has one (LAS/E57/PCD: m). A splat has no scale: give reference [{feature, diameter, tolerance}] or
        [{feature, to, distance}] (a caliper-measured size). Call first without out_dir: summary.features
        lists each feature id. Then with out_dir, support (feature ids that hold the part: the bar it
        clamps) and loads ([{feature, force: [fx, fy, fz] N}], one case each: the object's weight x its
        shock on the face it sits on) it writes a study: the object kept out by its own measured clearance
        plus the process's scatter (no fixed margin; unseen patches by their gap), bars as fitted bores
        with mateUncertainty (fits.py holds the fit for every bar in the band, or says measure it), faces
        as pads set off by their band. known {feature: {diameter, tolerance}} replaces a fitted diameter
        with a caliper's. Then loadpath_check and loadpath_start with base_dir. Seconds to a minute.

        Near-zero input: turn the user's sentence into brief {mountOn: "bar"|"wall"|id, family:
        "handlebar"|"seatpost"|"pipe"|"tube-inch"|"shaft"|..., mass: kg, shock: g, hold?: id, clamp:
        "split"|"ring"} and pass it with out_dir instead of support and loads: it picks the bar and the face
        facing it, finds the mass centre, writes shock cases (each axis both ways), snaps the bar to a
        standard size when its band covers one, and for a bar a split clamp with bolts sized by friction
        (clamp in the result; after the run, `loadpath split-clamp part.stl <name>.clamp.json` cuts body
        and cap). An unscaled scan (a splat) takes its scale from an ID-1 card in the frame, or a coin
        named in reference_objects (e.g. ["us-quarter"]). `asks` lists what only the user can say (mass,
        shock, which bar; a size pick; one caliper reading when the fit needs it): ask those, once."""
        return scan(path, out_dir, base_dir, device, units, scale, up, crop, keep, reference, known, support,
                    loads, material, process, grow, confidence, fit, brief, reference_objects, clamp, mu)

    @tool
    def loadpath_exterior(path: str, out_dir: str | None = None, base_dir: str | None = None,
                          size: float | None = None, scale: float | None = None, up: str | None = None,
                          place: str | None = None, pitch: float | None = None, skin: float | None = 1.0,
                          keep: str = "largest") -> dict:
        """Any exterior in, one clean watertight skin out (out_dir/exterior.stl, default ./exterior): a
        designer's STEP, a mesh, an AI-generated or scanned mesh (Tripo/Meshy GLB: open, self-intersecting,
        double-walled, debris, unit scale, Y-up), an .npz SDF grid, or file.py[:name] (an SDF function of
        (N, 3) points or a build123d/CadQuery object). size: the real longest extent in mm (generated meshes
        have no units); up: the source's up axis (default +y for GLB). The report: what was wrong
        (defects), how far the wrap moved the surface (wrap.onSource), where holes were patched
        (wrap.patched), debris dropped, and the room inside once a `skin` mm wall is in (capacity:
        cavityVolume, thinRegions too thin for anything but skin). Never redesign the exterior: relay
        patched holes and thin regions to the designer. Use the STL as loadpath_handoff's exterior, or
        give handoff {path, size, up} directly. Seconds to a minute."""
        return exterior(path, out_dir, base_dir, size, scale, up, place, pitch, skin, keep)

    @tool
    def loadpath_import_cad(path: str, out_dir: str | None = None, base_dir: str | None = None,
                            space: str = "part", material: str = "petg", force: float = 100.0,
                            grow: float = 0.0, process: str | None = None, support: str | None = None,
                            loads: list | None = None, ring: float | None = None) -> dict:
        """Bring a CAD part in: a STEP (.step/.stp) or STL/3MF/OBJ file in mm. Finds its bores (axis,
        radius, counterbore steps, through or blind; 'bolt' up to M6 clearance, else 'bore'), the flat
        exposed faces it mounts by with the bolts opening onto each, and its interfaces (a bolted face,
        a bolt pattern, a bore), and heals open shells. With out_dir it also writes a draft study there:
        the part (space 'part'), its convex hull ('hull') or box ('box'), grown by grow mm, as design
        space; each interface preserved (rings round bores, a pad under bolted faces, cut to the part);
        bores as keep-outs; mating faces cleared when the space grows; and a suggested support (the
        main bolted interface) and load (force N, a placeholder, across the lever to the farthest
        large interface). Call it first without out_dir: `summary.interfaces` lists each interface id
        with what it is. Then pass material (and process, e.g. 'sls'), support (an interface id) and
        loads ([{interface, force: [fx, fy, fz] N}], one load case each) so the draft carries the real
        setup instead of the placeholder; bolts get washer seats, not bosses. ring (mm) keeps an exact
        tube of that wall round every bore. space 'part' only removes material from the part as drawn:
        to lighten within its envelope use 'box' or 'hull'. Then loadpath_check and loadpath_start with
        base_dir = the returned baseDir. Seconds."""
        return import_cad(path, out_dir, base_dir, space, material, force, grow, process, support, loads, ring)

    @tool
    def loadpath_verify(study: dict | str, path: str, base_dir: str | None = None, max_cells: int = 60000) -> dict:
        """Check any mesh (an earlier outcome, an edited STL, a part from elsewhere) against a study:
        mass, safety factor, peak stress and deflection per load case, keep-out intrusion, preserve
        loss, bodies, and passes against the study's goal. A minute or less; no job needed."""
        from .tasks import verify_mesh
        return verify_mesh(study, path, base_dir, max_cells)

    @tool
    def loadpath_jobs(limit: int = 20) -> dict:
        """Recent jobs of every kind (studies, explores, lattices, casts, enclosures, exports...), newest
        first, each with its jobId, kind, name, status (running, done, failed, cancelled) and start time.
        A jobId from here goes to loadpath_status, loadpath_cancel or loadpath_resume."""
        return {"jobs": jobs(limit)}

    @tool
    def loadpath_cancel(job_id: str) -> dict:
        """Stop a running job (any kind). Its folder and whatever it wrote so far stay; a study stopped
        partway can be continued with loadpath_resume."""
        return cancel(job_id)

    @tool
    def loadpath_resume(job_id: str) -> dict:
        """Continue a study job that died partway (status 'died' or 'failed' after a machine restart or an
        out-of-memory kill): the optimizations it finished are kept and the one it was in continues from its
        last checkpoint (every 8 iterations), delivering what the uninterrupted run would have. Poll
        loadpath_status as before."""
        return resume(job_id)

    @tool
    def loadpath_materials(name: str | None = None, stock: bool = False) -> dict:
        """The material library a study's `material` names. With no name: every card with its stiffness
        E, strength along and across layers (strengthZ), density and the processes it is made by, and
        the calibrated cards (fit from the user's own printed coupons, which win over the library). With
        a name ('pa12', 'petg', 'ti64'...): that card in full, with where each number came from. stock:
        the cards for bought members in a study's `stock` (tubes, laminates, adhesives)."""
        return materials(name, stock)

    @tool
    def loadpath_orient(path: str, study: dict | str | None = None, process: str = "fdm",
                        threemf: str | None = None, base_dir: str | None = None) -> dict:
        """Rank the build orientations of a mesh (STL/3MF/OBJ, mm) for a process (fdm, sls, mjf, sla,
        dlp, lpbf): support volume, overhang area, bed contact, bridges, print time, and with a study
        the layer bond read under its load cases (a printed part is weakest across its layers).
        threemf: also write the best orientation as a slicer-ready 3MF at that path. Starts a job;
        poll loadpath_status."""
        return orient(path, study, process, threemf, base_dir)

    @tool
    def loadpath_buildsim(path: str, process: str = "lpbf", material: str = "ti64", build: str = "+z",
                          plate: float = 10.0, compensate: int = 0, nominal: str | None = None,
                          base_dir: str | None = None) -> dict:
        """Simulate printing a part in its build position (lpbf metal, fdm, sls): distortion and how
        far the printed shape deviates from the design, residual stress, warp off the bed, and the
        recoater risk (powder bed). compensate=N: N passes of pre-distorting the mesh so it prints to
        shape (out: the compensated STL). nominal: the design to compare with (default: the part).
        Starts a job; poll loadpath_status."""
        return buildsim(path, process, material, build, plate, compensate, nominal, base_dir)

    @tool
    def loadpath_export_step(path: str, out: str | None = None, base_dir: str | None = None,
                             faces: int | None = None, surface: str = "facets",
                             tolerance: float | None = None) -> dict:
        """Turn a closed mesh (an outcome's STL, 3MF or OBJ, mm) into one STEP solid any CAD opens.
        surface 'facets' (planar faces, exact to the mesh) or 'smooth' (B-spline patches); faces: the
        facet budget (default 20000). out: the .step path (default: next to the mesh). Reports valid,
        solids and the volume error against the mesh. Needs the step extra (OpenCascade). Starts a
        job; poll loadpath_status."""
        return export_step(path, out, base_dir, faces, surface, tolerance)

    @tool
    def loadpath_export_bundle(study: dict | str, path: str, out: str | None = None,
                               base_dir: str | None = None) -> dict:
        """Write a part's results bundle for other viewers and tools: the part verified against its
        study with its fields (stress, safety factor, displacement per load case) as glTF with
        colours, VTU fields, the report and tradeoff.csv, listed in manifest.json
        (loadpath_schema('bundle'), docs/BUNDLE.md). path: the part's STL (a run's best.stl also
        reads the run's report.json). out: the folder (default <part>-bundle). Starts a job; poll
        loadpath_status."""
        return export_bundle(study, path, out, base_dir)

    return server


def serve():
    try:
        import mcp  # noqa: F401
    except ImportError:  # pragma: no cover - optional extra
        sys.exit('loadpath mcp needs the MCP SDK: pip install "loadpath[mcp]"')
    build_server().run("stdio")


def tool_reference() -> str:
    """docs/agents.md's tool table, from the registered tools."""
    import asyncio
    import inspect

    server = build_server()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    lines = []
    for name in TOOLS:
        t = tools[name]
        props = t.input_schema.get("properties", {})
        req = set(t.input_schema.get("required", []))
        params = ", ".join(f"`{k}`" + ("" if k in req else "?") for k in props)
        desc = inspect.cleandoc(t.description or "").split("\n\n")[0].replace("\n", " ").replace("|", "\\|")
        effect = {"read": "reads", "write": "writes files", "job": "job", "stop": "stops a job"}[TOOLS[name][1]]
        lines.append(f"| `{name}` | {effect} | {params} | {desc} |")
    return "| tool | effect | parameters (? optional) | what it does |\n|---|---|---|---|\n" + "\n".join(lines) + "\n"


def surface() -> dict:
    """The tool surface the tests keep stable: each tool's parameters,
    which are required, and its effect."""
    import asyncio

    server = build_server()
    out = {}
    for t in sorted(asyncio.run(server.list_tools()), key=lambda t: t.name):
        props = t.input_schema.get("properties", {})
        out[t.name] = {"effect": TOOLS[t.name][1], "required": sorted(t.input_schema.get("required", [])),
                       "optional": sorted(k for k in props if k not in t.input_schema.get("required", []))}
    return out


if __name__ == "__main__":  # pragma: no cover
    if sys.argv[1:] == ["--reference"]:
        print(tool_reference(), end="")
    elif sys.argv[1:] == ["--surface"]:
        print(json.dumps(surface(), indent=1))
    else:
        serve()
