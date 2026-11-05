"""
Brief evals: from a plain-language request to a printable part, with no
human in the loop.

Each brief in briefs/*.json is what a person would type to their agent (plus
attachments in briefs/assets/: an STL, a STEP, a URDF, a dimensioned sketch)
and a gold study: the minimum defensible reading of the brief (its
interfaces, envelope, supports, loads, material and process). The agent never
sees the gold study. It writes its own study, runs it and hands back an STL;
the scorer re-verifies that STL against the gold study (verify.verify_part,
the same analysis loadpath uses on its own outcomes), so a part only passes
if it is right for the brief, not for whatever problem the agent wrote down:

    pass = one body, inside the envelope, every gold keep-out clear, every
           gold interface present, gold safety factor >= SF_SLACK x target in
           every gold load case, deflection where the brief asks, and the
           brief's mass limit (or, without one, at most half the solid
           envelope's mass: exporting the design space is not a design);
           cast and milled briefs also read the gold study's mold or cutter
           (maxUndercutVolume, maxUnderDraftedArea, maxToolUnreachableVolume,
           maxMachiningMinutes off manufacture.report)

Run:
    python -m loadpath.evals.briefs list
    python -m loadpath.evals.briefs gold                # lint + build every gold study
    python -m loadpath.evals.briefs prompt <id>         # what the agent is told
    python -m loadpath.evals.briefs score <id> part.stl
    python -m loadpath.evals.briefs run --out briefs-out/ [--jobs 4] [ids...]
        drives a headless agent (`claude -p`, loadpath MCP server + CLI) per
        brief in its own empty folder, then scores what it left behind; an
        agent that stops while its background run goes is resumed when the
        run ends (a headless session gets no notification). Install the
        agent's loadpath with [mcp,crosscheck,render], and on a 4-core box
        use --jobs 2 --threads 1;
        writes briefs-out/scorecard.json and scorecard.md.
    python -m loadpath.evals.briefs table briefs-out/   # re-score / re-tabulate

Briefs are kept out of the agent's reach by running it in a fresh folder with
only the attachments; the prompt also says the answer key is off limits.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEFS = os.environ.get("LOADPATH_BRIEFS") or os.path.join(HERE, "briefs")
ASSETS = os.path.join(BRIEFS, "assets")

SF_SLACK = 0.9           # gold SF must reach 90% of the target: the agent's grid and ours differ
DEFLECTION_SLACK = 1.1
MASS_SLACK = 1.05
MAX_INTRUSION = 30.0     # mm^3 of part inside a gold keep-out (a 0.05 mm bore undersize is ~10)
MAX_MISSING = 30.0       # mm^3 of gold interface missing
MAX_OUTSIDE = 100.0      # mm^3 of part outside the envelope
MAX_SOLID_FRACTION = 0.5  # without a stated mass, at most half the solid envelope (no "export the box")
VERIFY_CELLS = 40_000


def _get(m, *keys):
    for k in keys:
        m = m.get(k) if isinstance(m, dict) else None
    return m


# cast and milled briefs: pass keys read off manufacture.report under the gold study
MAKING = {
    "maxUndercutVolume": lambda m: None if not m.get("access") else
    m["access"]["undercutVolume"] - m["access"]["undercutInKeepOuts"],
    "maxUnderDraftedArea": lambda m: _get(m, "draft", "underDraftedArea"),
    "maxToolUnreachableVolume": lambda m: _get(m, "machining", "toolUnreachableVolume"),
    "maxMachiningMinutes": lambda m: _get(m, "machining", "minutes", "total"),
}

PROMPT = """{brief}

{attachments}You have loadpath for this: its MCP tools (server `loadpath`) and the `loadpath`
command line. Work it through on your own; nobody will answer questions, so make reasonable
engineering assumptions and note them. When you are done, leave in the current folder:
  part.stl     the part to print (your final, verified outcome)
  study.json   the study that produced it
  notes.md     two or three lines: the outcome's mass and safety factor, and your assumptions
This machine is shared: keep resolution.maxElements at or under {max_elements} and run one
optimization at a time. Do not read loadpath/evals/briefs (that is the answer key).
"""


def load(name: str) -> dict:
    path = name if name.endswith(".json") else os.path.join(BRIEFS, name + ".json")
    with open(path) as fh:
        return json.load(fh)


def all_ids() -> list[str]:
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(BRIEFS, "*.json")))


def gold_study(brief: dict):
    from ..spec import study_from_dict
    return study_from_dict(brief["gold"], base_dir=ASSETS)


def kind(brief: dict):
    """The brief's kind module, for briefs that are not a single structural
    part: loadpath/evals/brief_kinds/<kind>.py with DELIVER (what the agent
    leaves), score(brief, folder) and gold(brief) (the gold's own check).
    None for the default "study" kind (part.stl under a gold study)."""
    k = brief.get("kind", "study")
    if k == "study":
        return None
    import importlib
    return importlib.import_module(f"loadpath.evals.brief_kinds.{k}")


def prompt(brief: dict, max_elements: int = 20000) -> str:
    att = ""
    if brief.get("attachments"):
        att = "Attached (in the current folder): " + ", ".join(os.path.basename(a) for a in brief["attachments"]) + "\n\n"
    text = PROMPT.format(brief=brief["brief"], attachments=att, max_elements=max_elements)
    mod = kind(brief)
    if mod is not None:
        head, _, rest = text.partition("When you are done, leave in the current folder:\n")
        tail = rest[rest.index("This machine is shared"):]
        text = head + "When you are done, leave in the current folder:\n" + mod.DELIVER.rstrip("\n") + "\n" + tail
    return text


def score_dir(brief: dict, folder: str, max_cells: int = VERIFY_CELLS) -> dict:
    """Score what an agent left in `folder`, by the brief's kind."""
    mod = kind(brief)
    if mod is None:
        return score(brief, os.path.join(folder, "part.stl"), max_cells)
    try:
        row = mod.score(brief, folder)
    except Exception as exc:  # noqa: BLE001 - a broken delivery is a failed brief, not a crashed eval
        msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
        row = {"pass": False, "failures": [f"scoring failed: {msg[:200]}"]}
    row.setdefault("failures", [])
    row["id"] = brief["id"]
    row["pass"] = bool(row.get("pass")) and not row["failures"]
    return row


def score(brief: dict, stl: str, max_cells: int = VERIFY_CELLS) -> dict:
    """Re-verify the delivered STL under the gold study. Returns
    {pass, failures, mass, cases: [{name, safetyFactor, deflection}], ...}."""
    import trimesh

    from ..verify import verify_part

    want = brief["pass"]
    row = {"id": brief["id"], "pass": False, "failures": []}
    if not stl or not os.path.exists(stl):
        row["failures"].append("no part.stl")
        return row
    try:
        mesh = trimesh.load(stl, force="mesh")
        study = gold_study(brief)
        rep = verify_part(study, mesh, max_cells=max_cells, refine=3)
    except Exception as exc:  # noqa: BLE001 - a broken part is a failed part, not a crashed eval
        msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
        row["failures"].append(f"verify failed: {msg[:200]}")
        return row
    f = row["failures"]
    row["mass"] = round(rep["mass"], 2)
    row["bodies"] = rep.get("bodies")
    row["keepOutIntrusion"] = round(rep.get("keepOutIntrusion", 0.0), 1)
    row["preserveMissing"] = round(rep.get("preserveMissing", 0.0), 1)
    row["outsideDesignSpace"] = round(rep.get("outsideDesignSpace", 0.0), 1)
    row["cases"] = [{"name": c["name"], "safetyFactor": round(c["safetyFactor"], 3),
                     "deflection": round(c["deflection"], 4) if c.get("deflection") is not None else None}
                    for c in rep["cases"]]
    row["safetyFactor"] = round(rep["safetyFactor"], 3)
    if rep.get("bodies") != 1:
        f.append(f"{rep.get('bodies')} bodies")
    if row["outsideDesignSpace"] > MAX_OUTSIDE:
        f.append(f"{row['outsideDesignSpace']} mm^3 outside the envelope")
    if row["keepOutIntrusion"] > MAX_INTRUSION:
        f.append(f"{row['keepOutIntrusion']} mm^3 in keep-outs")
    if row["preserveMissing"] > MAX_MISSING:
        f.append(f"{row['preserveMissing']} mm^3 of interfaces missing")
    sf = want.get("safetyFactor")
    if sf:
        for c in row["cases"]:
            if c["safetyFactor"] < SF_SLACK * sf:
                f.append(f"SF {c['safetyFactor']:.2f} < {sf} in '{c['name']}'")
    dsf = want.get("durabilitySafetyFactor")
    if dsf:
        # fatigue / creep under the gold study's durability block
        dur = rep.get("durability") if isinstance(rep.get("durability"), dict) else {}
        row["durabilitySafetyFactor"] = dur.get("safetyFactor")
        if dur.get("safetyFactor") is None:
            f.append("no durability check under the gold study")
        elif dur["safetyFactor"] < SF_SLACK * dsf:
            f.append(f"durability SF {dur['safetyFactor']:.2f} < {dsf}")
    dmax = want.get("maxDeflection")
    if dmax:
        for c in row["cases"]:
            if c["deflection"] is not None and c["deflection"] > DEFLECTION_SLACK * dmax:
                f.append(f"deflection {c['deflection']:.3f} > {dmax} mm in '{c['name']}'")
    making = {k: want[k] for k in MAKING if want.get(k) is not None}
    if making:
        # cast or milled: the GOLD study's mold or cutter judges the part
        try:
            from ..domain import build_domain, choose_pitch
            from ..manufacture import report as manufacturing_report

            m = manufacturing_report(study, build_domain(study, pitch=choose_pitch(study)), mesh) or {}
        except Exception as exc:  # noqa: BLE001
            m = {}
            f.append(f"manufacturing check failed: {str(exc)[:160]}")
        for key, lim in making.items():
            v = MAKING[key](m)
            row[key[3].lower() + key[4:]] = None if v is None else round(float(v), 1)
            if v is None:
                f.append(f"no {key[3:]} reading under the gold study")
            elif v > lim:
                f.append(f"{key[3:]} {v:.0f} > {lim}")
    mmax = want.get("maxMass")
    row["solidFraction"] = round(rep["mass"] / rep["designSpaceMass"], 3) if rep.get("designSpaceMass") else None
    if mmax and rep["mass"] > MASS_SLACK * mmax:
        f.append(f"{rep['mass']:.1f} g > {mmax} g")
    elif not mmax and row["solidFraction"] and row["solidFraction"] > want.get("maxSolidFraction", MAX_SOLID_FRACTION):
        f.append(f"{row['solidFraction']:.0%} of the solid envelope's mass")
    row["pass"] = not f
    return row


# --- the agent harness -------------------------------------------------------

def _mcp_config(path: str):
    with open(path, "w") as fh:
        json.dump({"mcpServers": {"loadpath": {"command": shutil.which("loadpath") or "loadpath", "args": ["mcp"]}}}, fh)


def _tool_stats(transcript: str) -> dict:
    """Count tool calls by name and pull the final result line out of a
    stream-json transcript."""
    calls, result = {}, {}
    try:
        with open(transcript) as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "assistant":
                    for c in ev.get("message", {}).get("content", []):
                        if c.get("type") == "tool_use":
                            n = c.get("name", "?")
                            calls[n] = calls.get(n, 0) + 1
                elif ev.get("type") == "result":
                    result = ev
    except OSError:
        pass
    return {"toolCalls": calls, "turns": result.get("num_turns"), "costUSD": result.get("total_cost_usd"),
            "agentSeconds": (result.get("duration_ms") or 0) / 1000.0 or None, "agentError": result.get("is_error")}


def prepare(brief: dict, out: str, max_elements: int = 20000) -> tuple[str, str]:
    """A fresh folder for one brief: its attachments and PROMPT.md. Any agent
    can then be pointed at it; `table` scores what it leaves behind."""
    work = os.path.join(out, brief["id"])
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    for a in brief.get("attachments", []):
        shutil.copy(os.path.join(BRIEFS, a), os.path.join(work, os.path.basename(a)))
    text = prompt(brief, max_elements)
    with open(os.path.join(work, "PROMPT.md"), "w") as fh:
        fh.write(text)
    return work, text


# Headless agents get the tools the task needs. Bash is whole: an agent's shell
# work (redirects, a detached run, a quick python one-liner) is ordinary use,
# and a narrow allowlist mostly measured the harness's permission prompts.
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "mcp__loadpath", "Bash", "Monitor", "ToolSearch"]

RESUME = ("The background run you were waiting on has ended (loadpath status shows how). Carry on with the "
          "brief from where you left off and leave the files it asks for.")


def _live_runs(work: str) -> list[str]:
    """--out folders under `work` whose run is still going (status.json with a live pid)."""
    from ..__main__ import run_status

    live = []
    for st in glob.glob(os.path.join(work, "**", "status.json"), recursive=True):
        try:
            if run_status(os.path.dirname(st)).get("state") == "running":
                live.append(os.path.dirname(st))
        except (OSError, ValueError):
            pass
    return live


def _session_id(transcript: str) -> str | None:
    sid = None
    try:
        with open(transcript) as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = ev.get("session_id") or sid
    except OSError:
        pass
    return sid


def run_agent(brief: dict, out: str, agent: str = "claude", model: str | None = None, timeout: float = 3600,
              max_elements: int = 20000, threads: int = 2, resumes: int = 4) -> dict:
    """One headless agent on one brief. A headless agent cannot be told when
    its background run ends (an interactive one gets a notification), so
    when it stops while a loadpath run it started is still going, the
    harness waits for the run and resumes the same session, up to
    `resumes` times: what the agent would have seen, not a hint."""
    work, text = prepare(brief, out, max_elements)
    cfg = os.path.join(out, f".{brief['id']}.mcp.json")
    _mcp_config(cfg)
    env = dict(os.environ, LOADPATH_HOME=os.path.join(out, f".{brief['id']}.jobs"))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[k] = str(threads)
    base = ["--output-format", "stream-json", "--verbose", "--mcp-config", cfg,
            "--allowedTools", *ALLOWED_TOOLS, "--max-turns", "150"]
    if model:
        base += ["--model", model]
    t0 = time.time()
    status = "ok"
    transcript = os.path.join(work, ".transcript.jsonl")
    cmd = [agent, "-p", text, *base]
    n_resumed = 0
    while True:
        left = timeout - (time.time() - t0)
        with open(transcript, "a") as tf, open(os.path.join(work, ".agent.err"), "a") as ef:
            try:
                subprocess.run(cmd, cwd=work, env=env, stdout=tf, stderr=ef, timeout=max(left, 1))
            except subprocess.TimeoutExpired:
                status = "timeout"
                break
        if n_resumed >= resumes or not _live_runs(work):
            break
        while _live_runs(work) and time.time() - t0 < timeout:
            time.sleep(15)
        sid = _session_id(transcript)
        if sid is None or time.time() - t0 >= timeout:
            break
        n_resumed += 1
        cmd = [agent, "-p", RESUME, "--resume", sid, *base]
    seconds = time.time() - t0
    row = score_dir(brief, work)
    row.update(_tool_stats(os.path.join(work, ".transcript.jsonl")))
    row.update({"seconds": round(seconds, 1), "agentStatus": status, "resumed": n_resumed,
                "studyWritten": os.path.exists(os.path.join(work, "study.json")), "tags": brief.get("tags", [])})
    with open(os.path.join(work, "score.json"), "w") as fh:
        json.dump(row, fh, indent=1)
    return row


def table(rows: list[dict]) -> str:
    n = len(rows)
    ok = sum(1 for r in rows if r["pass"])
    lines = [f"**{ok}/{n} briefs pass** ({100.0 * ok / max(n, 1):.0f}%) with no human fixes.", "",
             "| brief | pass | mass g | gold SF (worst case) | failures | turns | min |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: r["id"]):
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["id"], "yes" if r["pass"] else "no", r.get("mass", ""), r.get("safetyFactor", ""),
            "; ".join(r["failures"]) or "", r.get("turns") or "", round(r["seconds"] / 60, 1) if r.get("seconds") else ""))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m loadpath.evals.briefs", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("gold")
    g.add_argument("ids", nargs="*")
    p = sub.add_parser("prompt")
    p.add_argument("id")
    s = sub.add_parser("score")
    s.add_argument("id")
    s.add_argument("path", help="the agent's folder (any kind), or part.stl (a study brief)")
    r = sub.add_parser("run")
    r.add_argument("ids", nargs="*")
    r.add_argument("--out", default="briefs-out")
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--agent", default="claude")
    r.add_argument("--model")
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--max-elements", type=int, default=20000)
    r.add_argument("--threads", type=int, default=2)
    pr = sub.add_parser("prepare")
    pr.add_argument("ids", nargs="*")
    pr.add_argument("--out", default="briefs-out")
    pr.add_argument("--max-elements", type=int, default=20000)
    t = sub.add_parser("table")
    t.add_argument("out")
    t.add_argument("--rescore", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for i in all_ids():
            b = load(i)
            print(f"{i:28s} {','.join(b.get('tags', []))}")
        return 0
    if a.cmd == "gold":
        from ..tasks import check_study
        bad = 0
        for i in a.ids or all_ids():
            b = load(i)
            mod = kind(b)
            res = mod.gold(b) if mod is not None else check_study(b["gold"], base_dir=ASSETS)
            bad += not res["ok"]
            print(i, "ok" if res["ok"] else res["error"], res.get("cells", ""), "; ".join(res.get("warnings", []))[:300])
        return bad
    if a.cmd == "prompt":
        print(prompt(load(a.id)))
        return 0
    if a.cmd == "score":
        b = load(a.id)
        if os.path.isdir(a.path):
            row = score_dir(b, a.path)
        else:
            row = score(b, a.path)
        print(json.dumps(row, indent=1))
        return 0
    if a.cmd == "run":
        os.makedirs(a.out, exist_ok=True)
        ids = a.ids or all_ids()
        rows = []

        def one(i):
            row = run_agent(load(i), a.out, a.agent, a.model, a.timeout, a.max_elements, a.threads)
            print(json.dumps({k: row.get(k) for k in ("id", "pass", "failures", "mass", "safetyFactor", "turns", "seconds")}),
                  flush=True)
            return row

        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            rows = list(ex.map(one, ids))
        _write(a.out, rows)
        return sum(not r["pass"] for r in rows)
    if a.cmd == "prepare":
        for i in a.ids or all_ids():
            print(prepare(load(i), a.out, a.max_elements)[0])
        return 0
    if a.cmd == "table":
        rows = []
        for d in sorted(glob.glob(os.path.join(a.out, "*", "PROMPT.md"))):
            d = os.path.dirname(d)
            f = os.path.join(d, "score.json")
            old = json.load(open(f)) if os.path.exists(f) else None
            row = old
            stl = os.path.join(d, "part.stl")
            stale = old is not None and os.path.exists(stl) and os.path.getmtime(stl) > os.path.getmtime(f)
            if old is None or a.rescore or stale:
                row = score_dir(load(os.path.basename(d)), d)
                row["studyWritten"] = os.path.exists(os.path.join(d, "study.json"))
                if old:
                    row.update({k: old[k] for k in old if k not in row})
                with open(f, "w") as fh:
                    json.dump(row, fh, indent=1)
            rows.append(row)
        _write(a.out, rows)
        return sum(not r["pass"] for r in rows)


def _write(out, rows):
    with open(os.path.join(out, "scorecard.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    md = table(rows)
    with open(os.path.join(out, "scorecard.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
