"""
The editor's live stress preview (preview.py): a coarse solve straight
from the field graph, no mesh. It follows the part's geometry smoothly,
agrees with the check on deflection and mass, reuses its grid between
slider values, says what's missing, and the editor's server serves it.
Run: `python3 -m pytest tests/test_preview.py` (about 30 s).
"""
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from loadpath.fields import FieldGraph, load_study  # noqa: E402
from loadpath.preview import PreviewState, stress_preview  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.verify import verify_part  # noqa: E402
from tests.test_tune import GRAPH, STUDY, _dir  # noqa: E402


def _bar(h):
    g = json.loads(json.dumps(GRAPH))
    g["nodes"]["bar"]["shape"].update(min=[0, -5, -h], max=[60, 5, h])
    return g


def test_preview_follows_the_part_and_agrees_with_the_check():
    """A taller bar reads stronger and stiffer at every step; deflection and
    mass land near the meshed check's, and the SF within its coarse-grid
    error (the preview leaves out the check's hot-spot submodels)."""
    study = study_from_dict(STUDY)
    state = PreviewState()
    rows = [stress_preview(_bar(h), study=study, state=state, cells=6000) for h in (3.5, 4.0, 4.5, 5.0)]
    sf = [r["safetyFactor"] for r in rows]
    defl = [r["deflection"] for r in rows]
    assert all(b > a for a, b in zip(sf, sf[1:])) and all(b < a for a, b in zip(defl, defl[1:])), (sf, defl)
    assert len({r["pitch"] for r in rows}) == 1             # one grid while the slider moves
    chk = verify_part(study, FieldGraph(_bar(5.0)).mesh(pitch=0.5), max_cells=60_000)
    r = rows[-1]
    assert abs(r["mass"] / chk["mass"] - 1) < 0.02, (r["mass"], chk["mass"])
    assert abs(r["deflection"] / max(c["deflection"] for c in chk["cases"]) - 1) < 0.1
    assert 0.8 < r["safetyFactor"] / chk["safetyFactor"] < 1.5, (r["safetyFactor"], chk["safetyFactor"])
    # the map: stress over allowed on the node grid, its peak 1 / SF away from the contact bands
    grid = np.frombuffer(base64.b64decode(r["data"]), np.float16)
    assert grid.size == np.prod(r["dims"]) and abs(r["peak"] - 1 / r["safetyFactor"]) < 1e-9
    assert grid.max() >= r["peak"] * 0.99 and r["goal"]["safetyFactor"] == 4


def test_preview_needs_a_study():
    with pytest.raises(ValueError, match="needs a study"):
        stress_preview({k: v for k, v in GRAPH.items() if k != "tune"})


def test_server_serves_the_preview():
    from loadpath.editor import serve_editor
    from tests.test_editor import _request

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
        code, body = _request(base + "api/preview", {"doc": _bar(4.0), "cells": 4000})
        r = json.loads(body)
        assert code == 200 and r["cells"] > 2000 and r["safetyFactor"] > 1 and r["dims"], r
        code, body = _request(base + "api/preview", {"doc": {k: v for k, v in GRAPH.items() if k != "tune"}})
        assert code == 400 and "study" in json.loads(body)["error"]
    finally:
        holder["srv"].shutdown()


def test_arm_link_preview_is_quick():
    """The latency the editor sees on the arm link: about a second per slider
    value on the 4-core cloud box (the budget here is loose for slow CI)."""
    doc = json.loads((ROOT / "examples/fields/arm-link-tune.json").read_text())
    base = str(ROOT / "examples/fields")
    study = load_study(doc["tune"]["study"], base, doc.get("material"))
    state = PreviewState()
    stress_preview(doc, base, study=study, state=state)          # torch and the grid warm up
    doc["nodes"]["web"]["shape"]["max"][2], doc["nodes"]["web"]["shape"]["min"][2] = 8, -8
    t0 = time.time()
    r = stress_preview(doc, base, study=study, state=state)
    assert time.time() - t0 < 6 and 12_000 < r["cells"] < 30_000, r["timing"]
    assert 0.3 < r["deflection"] < 0.8 and r["safetyFactor"] > 3


_DRIVER = r"""
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path");
const [url, threeDir] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await (await browser.newContext({ viewport: { width: 900, height: 600 } })).newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  if (threeDir) await page.route("https://cdn.jsdelivr.net/**", (r) => r.fulfill({ contentType: "application/javascript",
    body: fs.readFileSync(path.join(threeDir, r.request().url().endsWith("OrbitControls.js") ? "OrbitControls.js" : "three.module.js")) }));
  await page.route("https://fonts.googleapis.com/**", (r) => r.fulfill({ body: "", contentType: "text/css" }));
  await page.goto(url);
  await page.waitForFunction(() => window.lpReady === true, null, { timeout: 180000 });
  await page.evaluate(() => window.lpEditor.view({ stress: true, color: null }));
  const done = () => page.waitForFunction(() => { const s = window.lpEditor.stress(); return !s.busy && (s.result || s.error); }, null, { timeout: 120000 });
  await page.waitForTimeout(300); await done();
  const a = await page.evaluate(() => ({ sf: window.lpEditor.stress().result.safetyFactor, legend: document.querySelector("#legend").innerText }));
  await page.evaluate(() => { window.lpEditor.set(["nodes", "bar", "shape", "max", 2], 3); window.lpEditor.set(["nodes", "bar", "shape", "min", 2], -3); });
  await page.waitForTimeout(300); await done();
  const b = await page.evaluate(() => ({ sf: window.lpEditor.stress().result.safetyFactor, ms: window.lpEditor.stress().ms }));
  console.log(JSON.stringify({ a, b, errors }));
  await browser.close();
})().catch((e) => { console.log(JSON.stringify({ _fatal: String(e) })); process.exit(1); });
"""


def test_editor_draws_the_preview():
    """In the page: picking "stress (live preview)" solves the part, the
    legend says it is a coarse preview, and thinning the bar with a slider
    brings a new, lower reading. Skipped without node and Playwright."""
    import shutil
    import subprocess

    from loadpath.editor import serve_editor

    node = shutil.which("node")
    env = {**os.environ, "NODE_PATH": os.environ.get("NODE_PATH") or "/opt/node22/lib/node_modules"}
    if not node or subprocess.run([node, "-e", "require('playwright')"], env=env, capture_output=True).returncode:
        pytest.skip("needs node and playwright")
    d = _dir()
    with open(os.path.join(d, "graph.json"), "w") as fh:
        json.dump(_bar(4.5), fh)
    holder = {}
    threading.Thread(target=serve_editor, args=(os.path.join(d, "graph.json"),),
                     kwargs=dict(port=0, open_browser=False, ready=lambda s: holder.update(srv=s)), daemon=True).start()
    for _ in range(100):
        if "srv" in holder:
            break
        time.sleep(0.05)
    with open(os.path.join(d, "drive.cjs"), "w") as fh:
        fh.write(_DRIVER)
    try:
        r = subprocess.run([node, os.path.join(d, "drive.cjs"), f"http://127.0.0.1:{holder['srv'].server_port}/",
                            os.environ.get("LOADPATH_THREE_DIR", "")], env=env, capture_output=True, text=True,
                           timeout=600)
    finally:
        holder["srv"].shutdown()
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "_fatal" not in out and not out["errors"], out
    assert "Stress preview (coarse)" in out["a"]["legend"] and "Run checks the part" in out["a"]["legend"], out["a"]
    assert out["b"]["sf"] < out["a"]["sf"], out
