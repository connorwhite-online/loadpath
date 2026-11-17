"""
The MCP server's tool surface, as an agent (and an MCP registry) sees it:
the tools the server lists, their names, parameters, annotations and
descriptions, the error shape, and every kind of job read back by
loadpath_status. schema/mcp-tools.json is the surface this keeps stable.

Needs the mcp extra (pip install "loadpath[mcp]"). Run:
`python3 tests/test_mcp_surface.py` (about half a minute).
"""
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import mcp_server  # noqa: E402

SERVER = mcp_server.build_server()
LISTED = {t.name: t for t in asyncio.run(SERVER.list_tools())}
SRC = (ROOT / "loadpath" / "mcp_server.py").read_text()


def _call(name, args):
    r = asyncio.run(SERVER.call_tool(name, args))
    return json.loads(r.content[0].text)


def test_every_tool_is_in_the_table_and_named_alike():
    assert set(LISTED) == set(mcp_server.TOOLS), (set(LISTED) ^ set(mcp_server.TOOLS))
    for name in LISTED:
        assert re.fullmatch(r"loadpath(_[a-z]+)+", name), name


def test_annotations_follow_the_effect():
    for name, t in LISTED.items():
        title, effect = mcp_server.TOOLS[name]
        a = t.annotations
        assert a is not None and a.title == title and t.title == title, name
        assert a.read_only_hint is (effect == "read"), name
        assert a.destructive_hint is (effect == "stop"), name
        assert a.open_world_hint is False, name


def test_descriptions_say_enough_and_name_no_vendor():
    assert "materialize" not in mcp_server.INSTRUCTIONS.lower()
    for name, t in LISTED.items():
        d = t.description or ""
        assert len(d) >= 80, (name, d)
        assert "materialize" not in d.lower(), name
        # a job tool tells the agent how to follow it
        if mcp_server.TOOLS[name][0] and mcp_server.TOOLS[name][1] == "job" and name != "loadpath_resume":
            assert "loadpath_status" in d, name
    # replies the server builds for the agent name no vendor either
    assert "materialize" not in SRC.lower()


def test_every_parameter_is_typed():
    # an unannotated parameter is advertised as a string, so an agent sends
    # '[...]' where a list was meant (loadpath_scan's loads, until 2026-11-17)
    for m in re.finditer(r"    def (loadpath_\w+)\((.*?)\) ->", SRC, re.S):
        for part in re.split(r",\s*(?![^\[]*\])", m.group(2)):
            part = part.strip()
            assert not part or ":" in part.split("=")[0], (m.group(1), part)
    for name, t in LISTED.items():
        for k, v in t.input_schema.get("properties", {}).items():
            assert "type" in v or "anyOf" in v, (name, k, v)


def test_file_inputs_are_called_path():
    for name, t in LISTED.items():
        props = t.input_schema.get("properties", {})
        for bad in ("mesh", "board", "file", "stl", "part"):
            assert bad not in props, (name, bad)
        if "path" in props or any(k in props for k in ("study", "brief", "spec", "graph")):
            assert "base_dir" in props, name


def test_surface_matches_the_published_file():
    live = mcp_server.surface()
    pub = json.loads((ROOT / "schema" / "mcp-tools.json").read_text())
    broke = []
    for name, p in pub.items():
        if name not in live:
            broke.append(f"{name} is gone")
            continue
        have = set(live[name]["required"]) | set(live[name]["optional"])
        broke += [f"{name}: parameter {k} is gone" for k in p["required"] + p["optional"] if k not in have]
        broke += [f"{name}: {k} became required" for k in live[name]["required"] if k not in p["required"]]
    assert not broke, ("breaking changes to the MCP surface (keep the old names, or decide to break it and "
                       f"regenerate schema/mcp-tools.json): {broke}")
    assert live == pub, ("schema/mcp-tools.json is stale: "
                         "python -m loadpath.mcp_server --surface > schema/mcp-tools.json")


def test_docs_table_is_current():
    text = (ROOT / "docs" / "agents.md").read_text()
    m = re.search(r"<!-- tools:.*?-->\n(.*?)<!-- /tools -->", text, re.S)
    assert m, "docs/agents.md lost its generated tool table"
    assert m.group(1) == mcp_server.tool_reference(), \
        "docs/agents.md's tool table is stale: replace it with `python -m loadpath.mcp_server --reference`"


def test_errors_come_back_as_ok_false_with_the_tool():
    missing = "/nonexistent/part.stl"
    for name, args in (("loadpath_cast", {"path": missing}), ("loadpath_slipcast", {"path": missing}),
                       ("loadpath_kicad", {"path": "/nonexistent/b.kicad_pcb"}),
                       ("loadpath_orient", {"path": missing}), ("loadpath_buildsim", {"path": missing}),
                       ("loadpath_export_step", {"path": missing}),
                       ("loadpath_export_bundle", {"study": {}, "path": missing})):
        r = _call(name, args)
        assert r["ok"] is False and r["tool"] == name and "no such file" in r["error"], (name, r)
        assert "base_dir" in r["hint"], (name, r)
    for name, args in (("loadpath_schema", {"kind": "nope"}), ("loadpath_examples", {"name": "nope"}),
                       ("loadpath_materials", {"name": "pa13"}), ("loadpath_cancel", {"job_id": "nope"}),
                       ("loadpath_lattice", {"job_id": "nope"})):
        r = _call(name, args)
        assert r["ok"] is False and r["tool"] == name and r["error"], (name, r)
    assert "did you mean" in _call("loadpath_materials", {"name": "pa13"})["error"]
    st = json.loads(asyncio.run(SERVER.call_tool("loadpath_status", {"job_id": "nope"})).content[0].text)
    assert st["ok"] is False and st["tool"] == "loadpath_status"


def test_materials_lists_and_describes():
    r = _call("loadpath_materials", {})
    names = {row["name"] for row in r["library"]}
    assert {"pa12", "petg"} <= names, names
    one = _call("loadpath_materials", {"name": "PETG"})
    assert one["ok"] and one["name"] == "petg" and one["card"], one


def _fake_job(kind, stdout, report=None):
    d = os.path.join(mcp_server._home(), os.urandom(16).hex())
    os.makedirs(os.path.join(d, "out"))
    with open(os.path.join(d, "job.json"), "w") as fh:
        json.dump({"kind": kind, "name": "x", "pid": None, "started": time.time()}, fh)
    with open(os.path.join(d, "stdout.json"), "w") as fh:
        fh.write(stdout)
    if report:
        with open(os.path.join(d, "out", report), "w") as fh:
            fh.write("{}")
    return os.path.basename(d)


def test_status_reads_jobs_that_write_their_own_report():
    # these kinds write castability.json, wires.json... and print a summary:
    # loadpath_status used to call every one of them "failed"
    for kind, (fname, _) in mcp_server._SUMMARY_KINDS.items():
        jid = _fake_job(kind, json.dumps({"ok": False, "report": "r", "risers": 2}, indent=1), fname)
        info, _ = mcp_server.status(jid)
        assert info["status"] == "done" and info["ok"] is False and info["next"], (kind, info)
        if fname:
            assert info["report"].endswith(fname), (kind, info)
        assert [j for j in mcp_server.jobs(200) if j["jobId"] == jid][0]["status"] == "done", kind
        jid = _fake_job(kind, json.dumps({"ok": False, "error": "bad brief"}))
        info, _ = mcp_server.status(jid)
        assert info["status"] == "failed" and info["error"] == "bad brief", (kind, info)
    # a run after the build (handoff run=true) comes back with it
    jid = _fake_job("handoff", json.dumps({"ok": True, "study": "s"}) + "\n" + json.dumps({"passes": True}))
    info, _ = mcp_server.status(jid)
    assert info["status"] == "done" and info["run"] == {"passes": True}, info


def test_orient_job_end_to_end():
    import trimesh

    with tempfile.TemporaryDirectory() as tmp:
        trimesh.creation.box((40, 20, 8)).export(os.path.join(tmp, "plate.stl"))
        r = _call("loadpath_orient", {"path": "plate.stl", "base_dir": tmp})
        assert r["ok"] and r["kind"] == "orient", r
        out = asyncio.run(SERVER.call_tool("loadpath_status", {"job_id": r["jobId"], "wait": 120}))
        info = json.loads(out.content[0].text)
        assert info["status"] == "done", info
        assert info["best"] in ("+z", "-z") and info["orientations"], info


TESTS = [
    test_every_tool_is_in_the_table_and_named_alike,
    test_annotations_follow_the_effect,
    test_descriptions_say_enough_and_name_no_vendor,
    test_every_parameter_is_typed,
    test_file_inputs_are_called_path,
    test_surface_matches_the_published_file,
    test_docs_table_is_current,
    test_errors_come_back_as_ok_false_with_the_tool,
    test_materials_lists_and_describes,
    test_status_reads_jobs_that_write_their_own_report,
    test_orient_job_end_to_end,
]


def main():
    os.environ.setdefault("LOADPATH_HOME", tempfile.mkdtemp(prefix="lp-mcp-"))
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
