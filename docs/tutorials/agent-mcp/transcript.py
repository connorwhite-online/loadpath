"""
An agent's session with loadpath's MCP server, replayed: the exact tool
calls the tutorial's README walks through, made by a real MCP client over
stdio against `python -m loadpath mcp`, so the README cannot drift from what
the server says.

    python transcript.py            # about 15 minutes; LOADPATH_QUICK=1 for a 5 minute first look

Prints each call and its (trimmed) result, and writes:

    transcript.json    every step: tool, arguments, trimmed result; and a summary
    report.json        the final job's report.json
    wrist-link.stl     the part that was delivered

The decisions an agent would make (fix what check reports, tighten the goal
after reading the report) are written out here, not made by a model: this is
the sequence a model should arrive at, with the server's real answers.
"""
import asyncio
import copy
import json
import os
import shutil
import sys
import time

from mcp import ClientSession, StdioServerParameters, stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))
POLL_SECONDS = 5          # an agent polls every 30-60 s; a script can afford less
MAX_DEFLECTION = 0.8      # mm at the bearing: the tighter goal the agent adds in step 9

with open(os.path.join(HERE, "wrist-link.json")) as fh:
    STUDY = json.load(fh)

steps = []


def show(value, width=100):
    text = json.dumps(value)
    return text if len(text) <= width else json.dumps(value, indent=2)


async def call(session, tool, arguments, trim=lambda r: r, label=None, quiet=False):
    """One tool call: the result's JSON (and how many images came with it),
    recorded and printed trimmed."""
    res = await session.call_tool(tool, arguments)
    texts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
    images = sum(1 for c in res.content if getattr(c, "type", "") == "image")
    try:
        full = json.loads(texts[0]) if texts else None
    except json.JSONDecodeError:
        full = texts[0]
    if res.is_error:
        raise SystemExit(f"{tool} raised instead of answering: {texts}")
    result = trim(full)
    if images:
        result = {**result, "images": images}
    if not quiet:
        steps.append({"step": len(steps) + 1, "tool": tool, "arguments": arguments, "result": result})
        print(f"\n[{len(steps)}] -> {tool} {show(label or arguments)}")
        print(f"    <- {show(result)}".replace("\n", "\n       "), flush=True)
    return full


def trim_status(r):
    """What an agent reads in a finished study's loadpath_status."""
    if r.get("status") != "done":
        return r
    d = r["deliver"]
    keep = ("passes", "mass", "safetyFactor", "safetyFactorMethod", "governedBy", "target", "maxDeflection")
    return {
        "status": "done",
        "deliver": {**{k: _r(d.get(k)) for k in keep},
                    "cases": [{k: _r(c[k]) for k in ("name", "safetyFactor", "deflection")} for c in d["cases"]],
                    "warnings": d.get("warnings", [])},
        "outcomes": [{k: _r(o.get(k)) for k in ("outcome", "volumeFraction", "massGrams", "safetyFactor", "passes",
                                                "bodies", "start", "attention")} for o in r["outcomes"]],
        "best": r["best"],
        "stl": os.path.relpath(d["stl"], HERE),
        "viewer": r.get("viewer") and os.path.relpath(r["viewer"], HERE),
        **({"quick": r["quick"]} if r.get("quick") else {}),
    }


def _r(v):
    if isinstance(v, float):
        return float(f"{v:.3g}")
    if isinstance(v, dict):
        return {k: _r(x) for k, x in v.items()}
    return v


async def run_job(session, study, why):
    started = await call(session, "loadpath_start", {"study": study}, label={"study": why},
                         trim=lambda r: {k: r[k] for k in ("ok", "jobId", "kind", "cells", "warnings", "next",
                                                           "quick") if k in r})
    if not started.get("ok"):
        raise SystemExit(f"loadpath_start refused the study: {started}")
    job = started["jobId"]
    polls = 0
    while True:
        await asyncio.sleep(POLL_SECONDS)
        polls += 1
        # the first running poll and the finished one go in the transcript;
        # the polls between are the same call again
        r = await call(session, "loadpath_status", {"job_id": job}, trim=trim_status, quiet=polls > 1)
        if r["status"] != "running":
            if polls > 1:  # the finished answer, asked once more so it goes in the transcript
                r = await call(session, "loadpath_status", {"job_id": job}, trim=trim_status)
            break
        if polls > 1:
            latest = r.get("latest") or {}
            print(f"    ... poll {polls}: {r.get('elapsedSeconds')} s, {latest.get('stage')} "
                  f"{latest.get('volumeFraction', '')}", flush=True)
    if r["status"] != "done":
        raise SystemExit(f"job {job} {r['status']}: {r.get('error')}\n{r.get('log', '')}")
    return job, r, polls


def payload_deflection(status):
    return next(c["deflection"] for c in status["deliver"]["cases"] if c["name"] == "payload")


async def main():
    jobs_dir = os.path.join(HERE, "jobs")
    shutil.rmtree(jobs_dir, ignore_errors=True)
    # An MCP client starts the server with a bare environment (PATH, HOME, ...),
    # so pass on what loadpath reads: LOADPATH_QUICK reaches the jobs this way.
    env = {k: v for k, v in os.environ.items() if k.startswith(("LOADPATH_", "PYTHON", "OMP_", "OPENBLAS_"))}
    env["LOADPATH_HOME"] = jobs_dir
    # design memory: the rerun in step 9 replays the designs the first run already
    # optimized instead of optimizing them again (docs/agents.md, "The loop")
    env["LOADPATH_RECALL"] = os.path.join(jobs_dir, "recall")
    server = StdioServerParameters(command=sys.executable, args=["-m", "loadpath", "mcp"], env=env, cwd=HERE)
    t0 = time.time()
    async with stdio_client(server) as (read, write), ClientSession(read, write) as s:
        await s.initialize()

        # 1-3: find a starting point and the fields it needs
        await call(s, "loadpath_examples", {},
                   trim=lambda r: {"examples": [f"{e['name']} ({e['kind']})" for e in r["examples"]]})
        await call(s, "loadpath_examples", {"name": "arm-link"},
                   trim=lambda r: {k: r[k] for k in ("name", "material", "process", "goal", "minMember")})
        await call(s, "loadpath_schema", {"kind": "study"},
                   trim=lambda r: {"properties": list(r["properties"]),
                                   "$defs.fastener.description": r["$defs"]["fastener"]["description"]})

        # 4: the first draft, with two mistakes: a misspelt key and a direction in words
        draft = copy.deepcopy(STUDY)
        draft = {("keepout" if k == "keepOut" else k): v for k, v in draft.items()}
        draft["loadCases"][0]["loads"][0]["gravity"] = "down"
        first = await call(s, "loadpath_check", {"study": draft},
                           label={"study": "wrist-link.json with 'keepout' and gravity 'down'"})

        # 5: both fixed (wrist-link.json as committed)
        ok = await call(s, "loadpath_check", {"study": STUDY}, label={"study": "wrist-link.json"},
                        trim=lambda r: {k: _r(r.get(k)) for k in ("ok", "cells", "pitch", "regions", "loadCases",
                                                                  "warnings", "notes", "quick") if k in r})

        # 6-8: run it, poll, read the report
        job1, st1, polls1 = await run_job(s, STUDY, "wrist-link.json")

        # 9-11: the payload case bends the link more than a wrist joint can take: add a deflection goal
        tight = copy.deepcopy(STUDY)
        tight["goal"] = {**STUDY["goal"], "maxDeflection": MAX_DEFLECTION}
        job2, st2, polls2 = await run_job(s, tight, f"wrist-link.json with goal {json.dumps(tight['goal'])}")

        # 12: the delivered part, checked as a print
        stl = st2["deliver"]["stl"]
        printed = await call(s, "loadpath_check_printability",
                             {"path": stl, "process": "sls", "material": "pa12", "min_wall": 1.0},
                             label={"path": os.path.relpath(stl, HERE), "process": "sls", "material": "pa12",
                                    "min_wall": 1.0},
                             trim=lambda r: {"ok": r.get("ok"), "watertight": r.get("watertight"),
                                             "wallP1Mm": r.get("wallP1Mm"),
                                             "trappedVoidCount": r.get("trappedVoidCount"),
                                             "process": {k: (r.get("process") or {}).get(k)
                                                         for k in ("ok", "warnings")}})

    shutil.copy(stl, os.path.join(HERE, "wrist-link.stl"))
    shutil.copy(st2["report"], os.path.join(HERE, "report.json"))

    def summary(st, polls):
        o = st["outcomes"][st["best"]]
        return {"passes": st["deliver"]["passes"], "mass": st["deliver"]["mass"],
                "safetyFactor": st["deliver"]["safetyFactor"], "payloadDeflection": payload_deflection(st),
                "bodies": o.get("bodies"), "attention": o.get("attention"), "outcomes": len(st["outcomes"]),
                "recalled": sum(str(x.get("start", "")).startswith("recalled") for x in st["outcomes"]),
                "polls": polls}

    out = {
        "tools": [x["tool"] for x in steps],
        "checkCaught": [i["path"] for i in first.get("issues", []) if i["level"] == "error"],
        "checkError": first.get("error"),
        "checkOk": ok["ok"],
        "first": summary(st1, polls1),
        "final": summary(st2, polls2),
        "maxDeflectionGoal": MAX_DEFLECTION,
        "printable": printed.get("ok"),
        "seconds": round(time.time() - t0),
        "steps": steps,
    }
    with open(os.path.join(HERE, "transcript.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    f, l = out["first"], out["final"]
    print(f"\nfirst run: {f['mass']:.1f} g, SF {f['safetyFactor']:.2f}, payload deflection {f['payloadDeflection']:.2f} mm")
    print(f"final:     {l['mass']:.1f} g, SF {l['safetyFactor']:.2f}, payload deflection {l['payloadDeflection']:.2f} mm "
          f"(goal {MAX_DEFLECTION} mm), passes {l['passes']}; {out['seconds']} s")


if __name__ == "__main__":
    asyncio.run(main())
