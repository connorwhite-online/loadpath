"""
Tests for editor.py, the browser field-graph editor: bakes, the standalone
page, the local server (lint, save, bake, run) and the CLI; and, when node
and Playwright are installed, the page itself in headless Chromium: the
GLSL it compiles from a graph gives the same numbers as fields.py on every
op. Run: `python3 tests/test_editor.py` (about half a minute).
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.editor import bake_node, editor_payload, field_editor, serve_editor  # noqa: E402
from loadpath.fields import FieldGraph  # noqa: E402
from loadpath.schema import lint  # noqa: E402

# every op the GPU evaluates itself, plus a mesh and a grid it reads from bakes
ALL_OPS = {
    "name": "all-ops",
    "nodes": {
        "box": {"op": "shape", "shape": {"type": "box", "center": [0, 0, 0], "size": [40, 30, 20]}},
        "ball": {"op": "shape", "shape": {"type": "sphere", "center": [12, 8, 6], "r": 12}},
        "rod": {"op": "shape", "shape": {"type": "union", "shapes": [
            {"type": "cylinder", "p0": [-25, 0, 0], "p1": [25, 0, 0], "r": 4},
            {"type": "tube", "center": [0, 0, 0], "axis": "y", "length": 40, "r": 6, "rInner": 3}]}},
        "blob": {"op": "union", "inputs": ["box", "ball"], "blend": 4},
        "cut": {"op": "subtract", "inputs": ["blob", "rod"], "blend": 1.0},
        "ramp": {"op": "grid", "path": "ramp.npz"},
        "zc": {"op": "coordinate", "axis": "z"},
        "dir": {"op": "coordinate", "axis": [1, 1, 0], "origin": [1, 2, 3]},
        "dp": {"op": "distance", "points": [[0, 0, 10], [10, -5, 0]]},
        "dseg": {"op": "distance", "segment": [[-20, 0, 0], [20, 5, 3]]},
        "dpl": {"op": "distance", "plane": {"point": [0, 0, 2], "normal": [0, 1, 1]}},
        "dsh": {"op": "distance", "shape": {"type": "sphere", "center": [0, 0, 0], "r": 5}},
        "dcut": {"op": "distance", "input": "cut"},
        "wall": {"op": "remap", "input": "ramp", "from": [0, 1], "to": [0.6, 2.5], "curve": "smoothstep"},
        "ex": {"op": "expr", "expr": "0.8 + 1.2 * smoothstep(5, 30, k) + where(z - 1, 0.1, -0.1) + max(x, y, 0) * 0.01"
                                     " + (2 if x > 0 else 1) + y ** 2 * 0.001 + x % 7 * 0.01 - sqrt(abs(y)) * 0.1",
               "vars": {"k": "dcut"}},
        "skin": {"op": "shell", "input": "cut", "thickness": "wall"},
        "wallC": {"op": "shell", "input": "ball", "thickness": 1.5, "side": "center"},
        "cellF": {"op": "remap", "input": "dp", "from": [0, 30], "to": [4, 9], "curve": "sqrt"},
        "curveR": {"op": "remap", "input": "dp", "from": [0, 30], "to": [0.5, 3],
                   "curve": [[0, 0], [0.3, 0.8], [0.7, 0.6], [1, 1]]},
        "tableR": {"op": "remap", "input": "zc", "from": [-20, 0, 10, 40], "to": [1, 2, 1.5, 3]},
        "coreG": {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": "ex"},
        "coreD": {"op": "lattice", "type": "diamond", "cell": 7, "density": 0.25, "origin": [1, 1, 1]},
        "coreP": {"op": "lattice", "type": "schwarz-p", "cell": 6, "thickness": 0.8, "form": "network", "offset": 0.3},
        "coreC": {"op": "lattice", "type": "gyroid", "cell": 5, "thickness": 0.9,
                  "frame": {"type": "cylindrical", "axis": "z", "center": [0, 0, 0], "radius": 15}},
        "coreW": {"op": "lattice", "type": "gyroid", "cell": "cellF", "thickness": 1.0},
        "fill": {"op": "intersect", "inputs": ["coreG", "cut"], "blend": 0.5},
        "part": {"op": "union", "inputs": ["skin", "fill"], "blend": 0.8},
        "torus": {"op": "mesh", "path": "torus.stl"},
        "tor2": {"op": "transform", "input": "torus", "translate": [0, 0, 5],
                 "rotate": {"axis": "y", "degrees": 30, "center": [40, 0, 0]}, "scale": 0.9},
        "grow": {"op": "offset", "input": "tor2", "distance": 1.5},
        "all": {"op": "union", "inputs": ["part", {"op": "intersect", "inputs": [
            "coreD", {"op": "shape", "shape": {"type": "box", "min": [-60, -10, -10], "max": [-30, 10, 10]}}]}, "grow"]},
    },
    "output": "all",
    "mesh": {"pitch": 1.0},
    "material": "petg",
    "groups": {"rib": {"description": "a bar fused onto a base", "params": {"w": 1.2}, "inputs": ["base"],
                       "nodes": {"bar": {"op": "shape", "shape": {"type": "box", "min": [-15, -2, 8], "max": [15, 2, 12]}},
                                 "barS": {"op": "offset", "input": "bar", "distance": "w"},
                                 "out": {"op": "union", "inputs": ["base", "barS"], "blend": 0.5}},
                       "output": "out"}},
}
ALL_OPS["nodes"]["ribbed"] = {"op": "group", "group": "rib", "params": {"w": 0.8}, "inputs": {"base": "ball"}}
ALL_OPS["nodes"]["pic"] = {"op": "image", "path": "pic.png", "plane": "xz", "origin": [-20, -10], "size": [40, 20]}
ALL_OPS["nodes"]["embossed"] = {"op": "offset", "input": "box", "distance": "pic"}


def _workdir():
    d = tempfile.mkdtemp(prefix="lp-editor-")
    m = trimesh.creation.torus(major_radius=14, minor_radius=5, major_sections=48, minor_sections=24)
    m.apply_translation([40, 0, 0])
    m.export(os.path.join(d, "torus.stl"))
    ax = [np.linspace(-30, 30, 13)] * 3
    X, Y, _ = np.meshgrid(*ax, indexing="ij")
    np.savez(os.path.join(d, "ramp.npz"), origin=np.array([-30.0, -30, -30]), pitch=5.0, values=(X + 0.5 * Y + 30) / 60)
    from PIL import Image

    yy, xx = np.mgrid[0:16, 0:32]
    pic = (127 + 127 * np.sin(xx / 3.0) * np.cos(yy / 2.5)).astype(np.uint8)
    Image.fromarray(np.stack([pic, pic // 2, 255 - pic], -1)).save(os.path.join(d, "pic.png"))
    with open(os.path.join(d, "graph.json"), "w") as fh:
        json.dump(ALL_OPS, fh)
    return d


def _grid_values(b):
    data = np.frombuffer(base64.b64decode(b["data"]), np.float16).astype(float)
    return data.reshape(b["dims"][::-1]).transpose(2, 1, 0)


def test_graph_with_layout_lints():
    """The editor keeps node positions under `editor`; the schema allows
    it and nothing else changes."""
    doc = {**ALL_OPS, "editor": {"layout": {"box": [0, 0]}}}
    assert not [i for i in lint(doc, "field") if i["level"] == "error"]


def test_bakes_sample_the_graph():
    """A mesh is baked over its own box with a margin, a grid over the
    output; the grid nodes are the graph's own values, x fastest."""
    d = _workdir()
    g = FieldGraph(ALL_OPS, base_dir=d)
    b = bake_node(ALL_OPS, "torus", d, graph=g)
    lo, hi = np.asarray(b["lo"]), np.asarray(b["hi"])
    assert np.all(lo < [26 - 2.9, -19 - 2.9, -5 - 2.9]) and np.all(hi > [54 + 2.9, 19 + 2.9, 5 + 2.9]), (lo, hi)
    vals = _grid_values(b)
    assert list(vals.shape) == b["dims"] and max(b["dims"]) == 96
    axes = [np.linspace(lo[i], hi[i], b["dims"][i]) for i in range(3)]
    idx = [(10, 20, 5), (48, 40, 12), (90, 3, 20)]
    P = np.array([[axes[0][i], axes[1][j], axes[2][k]] for i, j, k in idx])
    want = g.evaluate(P, "torus")
    got = np.array([vals[i, j, k] for i, j, k in idx])
    assert np.allclose(got, want, atol=0.02 + 1e-3 * np.abs(want)), (got, want)
    assert b["spec"] == {"torus": ALL_OPS["nodes"]["torus"]}
    # a grid ships as it is: the GPU's trilinear filter is fields.py's interpolation
    r = bake_node(ALL_OPS, "ramp", d, graph=g)
    assert r["lo"] == [-30, -30, -30] and r["hi"] == [30, 30, 30] and r["dims"] == [13, 13, 13]
    assert np.allclose(_grid_values(r), np.load(os.path.join(d, "ramp.npz"))["values"], atol=1e-3)
    try:
        bake_node(ALL_OPS, "box", d, graph=g)
        raise AssertionError("a shape is not baked")
    except ValueError:
        pass


def test_bake_spec_follows_an_analyzed_part():
    """A result analyzed on one of the graph's own nodes depends on that
    node's whole subtree: the page compares this to tell a stale bake."""
    from loadpath.editor import _closure

    doc = {"nodes": {"a": {"op": "shape", "shape": {"type": "sphere", "center": [0, 0, 0], "r": 5}},
                     "b": {"op": "offset", "input": "a", "distance": "t"},
                     "t": {"op": "constant", "value": 1},
                     "unused": {"op": "constant", "value": 2},
                     "s": {"op": "result", "study": "example:arm-link", "part": "b"}}}
    assert set(_closure(doc, "s")) == {"s", "b", "a", "t"}


def test_standalone_page_carries_its_data():
    d = _workdir()
    html = field_editor(ALL_OPS, base_dir=d)
    assert html.startswith("<!doctype html>") and "</html>" in html
    start = html.index('id="loadpath-data">') + len('id="loadpath-data">')
    data = json.loads(html[start:html.index("</script>", start)])
    assert data["doc"]["nodes"].keys() == ALL_OPS["nodes"].keys()
    assert set(data["bakes"]) == {"torus", "ramp", "pic", "coreW#warp"} and not data["errors"] and not data["server"]
    assert set(data["tables"]) == {"gyroid", "diamond", "schwarz-p"} and len(data["tables"]["gyroid"]) == 19
    frag = field_editor(ALL_OPS, base_dir=d, standalone=False)
    assert "<!doctype" not in frag and "<body>" not in frag
    p = editor_payload({"nodes": {"m": {"op": "mesh", "path": "missing.stl"}}}, d)
    assert p["errors"] and "m:" in p["errors"][0]


def _request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_lints_bakes_saves_and_runs():
    d = _workdir()
    graph = os.path.join(d, "graph.json")
    holder = {}
    t = threading.Thread(target=serve_editor, args=(graph,), kwargs=dict(port=0, open_browser=False,
                                                                        ready=lambda s: holder.update(srv=s)),
                         daemon=True)
    t.start()
    for _ in range(100):
        if "srv" in holder:
            break
        time.sleep(0.05)
    base = f"http://127.0.0.1:{holder['srv'].server_port}/"
    try:
        code, page = _request(base)
        assert code == 200 and b"loadpath-data" in page and b'"server":true' in page
        bad = json.loads(json.dumps(ALL_OPS))
        bad["nodes"]["skin"]["input"] = "nope"
        code, body = _request(base + "api/lint", {"doc": bad})
        assert code == 200 and any("nope" in i["message"] for i in json.loads(body)["issues"])
        code, body = _request(base + "api/save", {"doc": bad})
        assert code == 400
        code, body = _request(base + "api/bake", {"doc": ALL_OPS, "node": "ramp"})
        b = json.loads(body)
        assert code == 200 and b["node"] == "ramp" and b["dims"]
        small = {"name": "ring", "nodes": {"s": {"op": "shape", "shape": {"type": "cylinder", "center": [0, 0, 0],
                                                                          "axis": "z", "length": 6, "r": 10,
                                                                          "rInner": 6}}},
                 "mesh": {"pitch": 0.5}, "material": "petg", "editor": {"layout": {"s": [0, 0]}}}
        code, body = _request(base + "api/save", {"doc": small})
        assert code == 200
        assert json.load(open(graph))["editor"]["layout"] == {"s": [0, 0]}
        code, body = _request(base + "api/run", {"doc": small})
        jid = json.loads(body)["id"]
        for _ in range(600):
            code, body = _request(base + f"api/job?id={jid}")
            job = json.loads(body)
            if job["state"] != "running":
                break
            time.sleep(0.1)
        assert job["state"] == "done", job.get("error")
        rep = job["report"]
        want = np.pi * (10 ** 2 - 6 ** 2) * 6
        assert rep["watertight"] and abs(rep["volume"] / want - 1) < 0.01, rep
        assert job["mesh"]["p"]["t"] == "f32" and job["events"]
        code, html = _request(base + job["files"]["viewer"].lstrip("/"))
        assert code == 200 and b"loadpath-data" in html
        code, _ = _request(base + job["files"]["stl"].lstrip("/"))
        assert code == 200
        code, _ = _request(base + "out/../graph.json")
        assert code == 404
        # one node of a bigger part, meshed alone; a node that isn't there is refused up front
        two = {**small, "nodes": {**small["nodes"], "b": {"op": "shape", "shape": {"type": "box", "min": [20, 0, 0],
                                                                                   "max": [30, 10, 10]}},
                                  "u": {"op": "union", "inputs": ["s", "b"]}}, "output": "u"}
        code, body = _request(base + "api/run", {"doc": two, "node": "nope"})
        assert code == 400 and "nope" in json.loads(body)["error"]
        code, body = _request(base + "api/run", {"doc": two, "node": "s"})
        jid = json.loads(body)["id"]
        for _ in range(600):
            job = json.loads(_request(base + f"api/job?id={jid}")[1])
            if job["state"] != "running":
                break
            time.sleep(0.1)
        assert job["state"] == "done" and job["node"] == "s", job.get("error")
        assert abs(job["report"]["volume"] / want - 1) < 0.01 and job["files"]["stl"].endswith("ring-s.stl")
    finally:
        holder["srv"].shutdown()


def test_server_reuses_what_it_built():
    """Resources (a mesh's field, a grid, an analysis) carry over to the
    next graph while their node's closure is unchanged, and not after."""
    from loadpath.editor import _Resources

    d = _workdir()
    res = _Resources()
    g = res.graph(ALL_OPS, d)
    g.evaluate(np.zeros((1, 3)), "grow")
    res.keep(g, ALL_OPS)
    torus = g._built["torus"]
    doc = json.loads(json.dumps(ALL_OPS))
    doc["nodes"]["grow"]["distance"] = 2.0                 # an edit downstream: the mesh is reused
    assert res.graph(doc, d)._built.get("torus") is torus
    doc["nodes"]["torus"]["path"] = "other.stl"            # the node itself changed: rebuilt
    assert "torus" not in res.graph(doc, d)._built


def test_warp_solve_is_the_1d_integral():
    """The DCT solve of the cell warp: along a cell that grades in x only,
    u_x is the integral of dx / cell, so each period is one local cell."""
    from loadpath.fields import _CellWarp

    w = _CellWarp(lambda P: 4.0 + 0.1 * P[:, 0], np.array([0.0, 0, 0]), np.array([60.0, 10, 10]), max_cells=60_000)
    x = np.array([[5.0, 5, 5], [50.0, 5, 5]])
    ux = w(x, 4.0 + 0.1 * x[:, 0])[:, 0]
    exact = 10 * np.log(4.0 + 0.1 * x[:, 0])
    assert abs((ux[1] - ux[0]) - (exact[1] - exact[0])) < 2e-3, (ux, exact)


def test_cli_writes_a_standalone_editor():
    d = _workdir()
    out = os.path.join(d, "editor.html")
    r = subprocess.run([sys.executable, "-m", "loadpath", "edit", os.path.join(d, "graph.json"), "--html", out],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["html"] == out and os.path.getsize(out) > 100_000


_DRIVER = r"""
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path");
const [page_, parity, threeDir] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await (await browser.newContext({ viewport: { width: 900, height: 600 } })).newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  if (threeDir) await page.route("https://cdn.jsdelivr.net/**", (r) => r.fulfill({ contentType: "application/javascript",
    body: fs.readFileSync(path.join(threeDir, r.request().url().endsWith("OrbitControls.js") ? "OrbitControls.js" : "three.module.js")) }));
  await page.route("https://fonts.googleapis.com/**", (r) => r.fulfill({ body: "", contentType: "text/css" }));
  await page.goto("file://" + path.resolve(page_));
  await page.waitForFunction(() => window.lpReady === true, null, { timeout: 180000 });
  const P = JSON.parse(fs.readFileSync(parity));
  const out = {};
  for (const [name, exp] of Object.entries(P.expected)) {
    const got = await page.evaluate(([n, pts]) => window.lpEditor.probe(n, pts), [name, P.points]);
    let near = 0, band = 0;
    for (let i = 0; i < exp.length; i++) {
      if (Math.abs(exp[i]) < 2) near = Math.max(near, Math.abs(got[i] - exp[i]));
      if (Math.abs(exp[i]) < 0.7) band = Math.max(band, Math.abs(got[i] - exp[i]));
    }
    out[name] = [near, band];
  }
  out._volume = await page.evaluate(() => window.lpEditor.mass(600000).volume);
  // a slider: a uniform changes, the shader is not rebuilt
  const before = await page.evaluate(() => window.lpEditor.glsl());
  await page.evaluate(() => window.lpEditor.set(["nodes", "ball", "shape", "r"], 9));
  const after = await page.evaluate(() => [window.lpEditor.glsl(), window.lpEditor.probe("ball", [12, 8, 6])[0]]);
  out._uniform = { same: before === after[0], center: after[1] };
  out._notes = await page.evaluate(() => document.querySelector("#notes").textContent);
  out._errors = errors;
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.log(JSON.stringify({ _fatal: String(e) })); process.exit(1); });
"""


def test_gpu_preview_matches_fields_py():
    """The page's GLSL, evaluated on the GPU (SwiftShader here), gives
    fields.py's numbers at 4000 points within 2 mm of each node's zero
    level: to float precision for shapes, booleans, shells, distances,
    remaps, expressions, lattices and transforms, to half precision for a
    grid (shipped as it is), and to the bake's resolution for a mesh near
    its surface; its live volume is the meshed part's to 1.5%. Skipped
    without node and Playwright."""
    node = shutil.which("node")
    npath = os.environ.get("NODE_PATH") or "/opt/node22/lib/node_modules"
    env = {**os.environ, "NODE_PATH": npath}
    if not node or subprocess.run([node, "-e", "require('playwright')"], env=env, capture_output=True).returncode:
        print("  (skipped: needs node and playwright)")
        return
    d = _workdir()
    page = field_editor(ALL_OPS, os.path.join(d, "editor.html"), base_dir=d)
    g = FieldGraph(ALL_OPS, base_dir=d)
    rng = np.random.default_rng(1)
    P = rng.uniform([-60, -40, -30], [80, 40, 45], (4000, 3))
    exp = {n: g.evaluate(P, n).tolist() for n in ALL_OPS["nodes"]}
    with open(os.path.join(d, "parity.json"), "w") as fh:
        json.dump({"points": P.tolist(), "expected": exp}, fh)
    with open(os.path.join(d, "drive.cjs"), "w") as fh:
        fh.write(_DRIVER)
    r = subprocess.run([node, os.path.join(d, "drive.cjs"), page, os.path.join(d, "parity.json"),
                        os.environ.get("LOADPATH_THREE_DIR", "")], env=env, capture_output=True, text=True,
                       timeout=600)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "_fatal" not in out and not out["_errors"], out
    # a mesh is resampled on a 96-cell grid; past its exact band (3 voxels) fields.py's own far field is a
    # coarse stand-in, so it is compared where the distance is exact
    meshed = {"torus", "tor2", "grow", "all"}
    for name, v in out.items():
        if name.startswith("_"):
            continue
        near, band = v
        # the graded lattice is drawn through the same warp, solved on a coarser grid
        err, tol = (band, 0.1) if name in meshed else (near, 0.4) if name == "coreW" else (near, 5e-3)
        assert err < tol, (name, err)
        print(f"  {name:6s} {err:.2g} mm")
    assert "earlier graph" not in out["_notes"] and "cross-fade" not in out["_notes"], out["_notes"]
    assert out["_uniform"]["same"] and abs(out["_uniform"]["center"] + 9) < 1e-4, out["_uniform"]
    # the live mass: the GPU's volume estimate against the meshed part
    vol = g.mesh(pitch=0.4).volume
    print(f"  volume {out['_volume']:.0f} mm3 on the GPU, {vol:.0f} meshed")
    assert abs(out["_volume"] - vol) < 0.015 * vol, (out["_volume"], vol)


_WALLS_DRIVER = r"""
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path");
const [page_, threeDir] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await (await browser.newContext({ viewport: { width: 900, height: 600 } })).newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  if (threeDir) await page.route("https://cdn.jsdelivr.net/**", (r) => r.fulfill({ contentType: "application/javascript",
    body: fs.readFileSync(path.join(threeDir, r.request().url().endsWith("OrbitControls.js") ? "OrbitControls.js" : "three.module.js")) }));
  await page.route("https://fonts.googleapis.com/**", (r) => r.fulfill({ body: "", contentType: "text/css" }));
  await page.goto("file://" + path.resolve(page_));
  await page.waitForFunction(() => window.lpReady === true, null, { timeout: 180000 });
  const out = await page.evaluate(() => {
    const E = window.lpEditor;
    E.view({ walls: true, color: null });
    E.camera([0, -0.01, 120], [0, 0, 0]);
    const o = {};
    for (const [k, p] of [["thin", [-15, 0, 10]], ["thick", [15, 0, 10]], ["thinSide", [-22, 0, 6]], ["thickSide", [22, 0, 6]]]) o[k] = E.pick(p)[3];
    o.legend = document.querySelector("#lg-title").textContent;
    return o;
  });
  // overhangs: a sphere's underside is red, its top and the face on the plate are not
  await page.evaluate(() => { window.lpReady = false; window.lpEditor.view({ walls: false, over: true }); window.lpEditor.camera([0, -120, 0.01], [0, 0, 0]); });
  await page.waitForFunction(() => window.lpReady === true, null, { timeout: 180000 });
  out.over = await page.evaluate(() => {
    const gl = document.querySelector("#gl"), c = document.createElement("canvas");
    c.width = gl.width; c.height = gl.height;
    const x = c.getContext("2d"); x.drawImage(gl, 0, 0);
    const at = (p) => { const v = new window.lpEditor.THREE.Vector3(...p).project(window.lpEditor.cam()); return Array.from(x.getImageData(Math.round((v.x + 1) / 2 * c.width), Math.round((1 - v.y) / 2 * c.height), 1, 1).data); };
    return { under: at([15, 0, -9.5]), top: at([15, 0, 9.5]) };
  });
  out.paste = await page.evaluate(() => {
    const E = window.lpEditor;
    const ok = E.paste(JSON.stringify({ nodes: { a: { op: "shape", shape: { type: "sphere", center: [0, 30, 0], r: 4 } },
                                                sa: { op: "shell", input: "a", thickness: "k" }, k: { op: "constant", value: 0.4 } } }));
    const dup = E.duplicate("both");
    const d = E.doc();
    return { ok, dup, names: Object.keys(d.nodes), sa2: d.nodes.sa2, both2: d.nodes[dup[0]], probe: E.probe("sa2", [[0, 30, 3.8]])[0] };
  });
  out.group = await page.evaluate(() => {
    const E = window.lpEditor, pts = [[15, 0, 9.5], [15, 0, 11], [16, 3, 0], [-15, 0, 10]];
    const before = E.probe("both", pts);
    const g = E.makeGroup("sb", "shellGroup");
    const d1 = E.doc(), mid = E.probe("both", pts);
    E.ungroup("sb");
    const d2 = E.doc(), after = E.probe("both", pts);
    return { g, before, mid, after, def: d1.groups && d1.groups[g], inst: d1.nodes.sb, back: Object.keys(d2.nodes).filter((n) => n.startsWith("sb")) };
  });
  out.sweep = await page.evaluate(async () => {
    const pts = await window.lpEditor.sweep("nodes/sb/thickness", 5);
    return { pts, after: window.lpEditor.doc().nodes.sb.thickness, chart: !document.querySelector("#sweep-chart").hidden };
  });
  out.tune = await page.evaluate(() => {
    const E = window.lpEditor;
    E.addTuneKnob("nodes/sa/thickness");
    E.addTuneKnob("nodes/sa/thickness");                      // once only
    return { spec: E.doc().tune, rows: document.querySelectorAll("#tune-params li").length, can: !document.querySelector("#tune").disabled };
  });
  out._errors = errors;
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.log(JSON.stringify({ _fatal: String(e) })); process.exit(1); });
"""


def test_overlays_paste_and_sweep():
    """Colour by wall thickness: a ray traced in along the normal reads a
    0.5 mm and a 2 mm spherical shell at their tops and flanks; the legend
    names the process minimum. Colour by overhangs: a sphere's underside is
    red, its top is not. Pasting and duplicating nodes renames what clashes
    and keeps the pasted nodes wired to each other; a sweep reads the mass
    across a parameter's range. Skipped without node and Playwright."""
    node = shutil.which("node")
    npath = os.environ.get("NODE_PATH") or "/opt/node22/lib/node_modules"
    env = {**os.environ, "NODE_PATH": npath}
    if not node or subprocess.run([node, "-e", "require('playwright')"], env=env, capture_output=True).returncode:
        print("  (skipped: needs node and playwright)")
        return
    d = tempfile.mkdtemp(prefix="lp-walls-")
    doc = {"nodes": {"a": {"op": "shape", "shape": {"type": "sphere", "center": [-15, 0, 0], "r": 10}},
                     "b": {"op": "shape", "shape": {"type": "sphere", "center": [15, 0, 0], "r": 10}},
                     "sa": {"op": "shell", "input": "a", "thickness": 0.5},
                     "sb": {"op": "shell", "input": "b", "thickness": 2.0},
                     "both": {"op": "union", "inputs": ["sa", "sb"]}}, "output": "both", "process": "fdm"}
    page = field_editor(doc, os.path.join(d, "editor.html"), base_dir=d)
    with open(os.path.join(d, "drive.cjs"), "w") as fh:
        fh.write(_WALLS_DRIVER)
    r = subprocess.run([node, os.path.join(d, "drive.cjs"), page, os.environ.get("LOADPATH_THREE_DIR", "")], env=env,
                       capture_output=True, text=True, timeout=600)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "_fatal" not in out and not out["_errors"], out
    for k, want in (("thin", 0.5), ("thick", 2.0), ("thinSide", 0.5), ("thickSide", 2.0)):
        assert abs(out[k] - want) < 0.02, (k, out[k])
    assert "FDM minimum 0.8 mm" in out["legend"], out["legend"]
    t = out["tune"]
    assert t["rows"] == 1 and t["can"] and t["spec"]["params"][0]["path"] == "nodes/sa/thickness", t
    assert t["spec"]["params"][0]["min"] < 0.5 < t["spec"]["params"][0]["max"], t
    red = lambda px: px[0] > 80 and px[0] > 2.5 * max(px[1], px[2])   # noqa: E731
    assert red(out["over"]["under"]) and not red(out["over"]["top"]), out["over"]
    # pasting nodes that clash: renamed, their references to each other follow; duplicating keeps the inputs
    p = out["paste"]
    assert p["ok"] and {"a2", "sa2", "k"} <= set(p["names"]), p
    assert p["sa2"] == {"op": "shell", "input": "a2", "thickness": "k"}, p["sa2"]
    assert p["dup"] == ["both2"] and p["both2"]["inputs"] == ["sa", "sb"], p
    assert abs(p["probe"] - (-0.2)) < 1e-4, p["probe"]       # the pasted shell is live: 0.4 mm wall at r 3.8
    # make group: sb and the sphere only it uses become a group with no inputs; the field does not change,
    # nor after ungrouping
    gr = out["group"]
    assert gr["g"] == "shellGroup" and set(gr["def"]["nodes"]) == {"sb", "b"} and gr["inst"] == {"op": "group", "group": "shellGroup"}, gr
    assert np.allclose(gr["before"], gr["mid"], atol=1e-5) and np.allclose(gr["before"], gr["after"], atol=1e-5), gr
    assert set(gr["back"]) == {"sb", "sb__b"}, gr["back"]
    # a sweep: the 10 mm sphere's shell from 0 to 6 mm thick gains volume all the way, and the value is put back
    sw = out["sweep"]
    vols = [v for _, v in sw["pts"]]
    assert len(vols) == 5 and all(b > a for a, b in zip(vols, vols[1:])) and sw["after"] == 2.0 and sw["chart"], sw


TESTS = [test_graph_with_layout_lints, test_bakes_sample_the_graph, test_bake_spec_follows_an_analyzed_part,
         test_standalone_page_carries_its_data, test_server_lints_bakes_saves_and_runs,
         test_server_reuses_what_it_built, test_warp_solve_is_the_1d_integral,
         test_cli_writes_a_standalone_editor, test_gpu_preview_matches_fields_py, test_overlays_paste_and_sweep]


def main():
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
