"""Tuning a field graph's numbers against an FE check of its part (tune.py),
from Python and through the editor's server."""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from loadpath.schema import lint  # noqa: E402
from loadpath.tune import get_at, tune_field  # noqa: E402

# a 60 mm cantilever, 10 mm wide, its height the knob; 25 N at the tip
STUDY = {"name": "bar", "designSpace": [{"type": "box", "min": [0, -6, -6], "max": [60, 6, 6]}],
         "preserve": [{"id": "root", "shape": {"type": "box", "min": [0, -5, -5], "max": [4, 5, 5]}},
                      {"id": "tip", "shape": {"type": "box", "min": [56, -5, -5], "max": [60, 5, 5]}}],
         "supports": [{"region": "root"}],
         "loadCases": [{"name": "bend", "loads": [{"region": "tip", "force": [0, 0, -25]}]}],
         "material": "petg", "goal": {"safetyFactor": 4}}
GRAPH = {"name": "bar", "nodes": {"bar": {"op": "shape", "shape": {"type": "box", "min": [0, -5, -5],
                                                                   "max": [60, 5, 5]}}},
         "output": "bar",
         "tune": {"study": "study.json",
                  "params": [{"path": "nodes/bar/shape/max/2", "also": ["-nodes/bar/shape/min/2"],
                              "min": 1.5, "max": 5}],
                  "pitch": 0.8, "cells": 6000, "evaluations": 8}}


def _dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "study.json"), "w") as fh:
        json.dump(STUDY, fh)
    with open(os.path.join(d, "graph.json"), "w") as fh:
        json.dump(GRAPH, fh)
    return d


def test_tune_finds_a_lighter_bar_that_still_holds():
    """The bar starts over-strong (SF ~6.6 against 4): the tuner thins it,
    stays inside the knob's range, keeps the box symmetric (`also`) and
    returns the lightest candidate that met the limit."""
    d = _dir()
    assert not lint(GRAPH, "field")
    out = tune_field(json.loads(json.dumps(GRAPH)), d, os.path.join(d, "out"))
    best, start = out["best"], out["start"]
    assert start["ok"] and best["ok"] and best["safetyFactor"] >= 4
    assert best["mass"] < 0.92 * start["mass"], (best, start)
    assert all(1.5 <= r["values"][0] <= 5 for r in out["history"])
    assert len(out["history"]) <= 8
    assert best["mass"] == min(r["mass"] for r in out["history"] if r["ok"])
    h = get_at(out["doc"], "nodes/bar/shape/max/2")
    assert abs(h - best["values"][0]) < 1e-9 and abs(get_at(out["doc"], "nodes/bar/shape/min/2") + h) < 1e-9
    saved = json.load(open(os.path.join(d, "out", "tuned.json")))
    assert saved["nodes"] == out["doc"]["nodes"] and saved["tune"] == GRAPH["tune"]


def test_tune_names_its_mistakes():
    d = _dir()
    for change, msg in (({"path": "nodes/bar/shape/max/7"}, "nothing at"), ({"path": "nodes/bar/op"}, "not a number"),
                        ({"min": 5, "max": 1}, "min < max")):
        g = json.loads(json.dumps(GRAPH))
        g["tune"]["params"][0].update(change)
        with pytest.raises(ValueError, match=msg):
            tune_field(g, d)
    g = json.loads(json.dumps(GRAPH))
    del g["tune"]
    with pytest.raises(ValueError, match="no 'tune' block"):
        tune_field(g, d)


def _request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_editor_server_tunes_and_hands_back_the_values():
    from loadpath.editor import serve_editor

    d = _dir()
    holder = {}
    threading.Thread(target=serve_editor, args=(os.path.join(d, "graph.json"),),
                     kwargs=dict(port=0, open_browser=False, ready=lambda s: holder.update(srv=s)), daemon=True).start()
    for _ in range(100):
        if "srv" in holder:
            break
        time.sleep(0.05)
    base = f"http://127.0.0.1:{holder['srv'].server_port}/"
    try:
        g = json.loads(json.dumps(GRAPH))
        g["tune"]["evaluations"] = 4
        code, body = _request(base + "api/tune", {"doc": {k: v for k, v in g.items() if k != "tune"}})
        assert code == 400
        code, body = _request(base + "api/tune", {"doc": g})
        jid = json.loads(body)["id"]
        for _ in range(1200):
            job = json.loads(_request(base + f"api/job?id={jid}")[1])
            if job["state"] != "running":
                break
            time.sleep(0.1)
        assert job["state"] == "done", job.get("error")
        assert job["kind"] == "tune" and any(e.get("event") == "tune" for e in job["events"])
        assign = dict(job["assign"])
        assert set(assign) == {"nodes/bar/shape/max/2", "nodes/bar/shape/min/2"}
        assert abs(assign["nodes/bar/shape/max/2"] + assign["nodes/bar/shape/min/2"]) < 1e-9
        assert assign["nodes/bar/shape/max/2"] == job["best"]["values"][0]
        assert _request(base + job["files"]["tuned"].lstrip("/"))[0] == 200
    finally:
        holder["srv"].shutdown()
