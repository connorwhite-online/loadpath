"""
The images in this tutorial (not part of the tutorial; needs Node with
Playwright's Chromium, see docs/render3d.py):

    python render.py [run/]          # after the tutorial's commands (default: in this folder)

images/editor.png is the standalone editor page (editor.html, from `loadpath
edit link.json --html editor.html`) in headless Chromium, drawn on
SwiftShader; images/tuned.png is the meshed link before and after tuning.
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from render3d import Scene, _three_dir  # noqa: E402

run = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
img = os.path.join(HERE, "images")
os.makedirs(img, exist_ok=True)

# the editor page, with three.js served from the local cache
three = _three_dir()
orbit = os.path.join(three, "OrbitControls.js")
if not os.path.exists(orbit):
    with urllib.request.urlopen("https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js",
                                timeout=60) as r:
        data = r.read()
    with open(orbit, "wb") as fh:
        fh.write(data)

DRIVER = r"""
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path");
const [page_, threeDir, png, setup] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })).newPage();
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.route("https://cdn.jsdelivr.net/**", (r) => r.fulfill({ contentType: "application/javascript",
    body: fs.readFileSync(path.join(threeDir, r.request().url().endsWith("OrbitControls.js") ? "OrbitControls.js" : "three.module.js")) }));
  await page.route("https://fonts.googleapis.com/**", (r) => r.fulfill({ body: "", contentType: "text/css" }));
  await page.goto("file://" + path.resolve(page_));
  await page.waitForFunction(() => window.lpReady === true, null, { timeout: 300000 });
  await page.evaluate((s) => { window.lpReady = false; (new Function("E", s))(window.lpEditor); }, setup);
  await page.waitForFunction(() => window.lpReady === true, null, { timeout: 300000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: png });
  await browser.close();
})().catch((e) => { console.error(String(e)); process.exit(1); });
"""

page = os.path.join(run, "editor.html")
if not os.path.exists(page):
    subprocess.run([sys.executable, "-m", "loadpath", "edit", os.path.join(HERE, "link.json"), "--html", page],
                   check=True, cwd=HERE)
work = tempfile.mkdtemp(prefix="lp-editor-shot-")
with open(os.path.join(work, "drive.cjs"), "w") as fh:
    fh.write(DRIVER)
env = {**os.environ, "NODE_PATH": os.environ.get("NODE_PATH") or "/opt/node22/lib/node_modules"}
shots = {"editor.png": 'E.select("window"); E.view({ follow: false, target: "part", color: "stress" }); E.camera([40, -95, 75], [40, 0, 0]);'}
for name, setup in shots.items():
    raw = os.path.join(work, name)
    subprocess.run(["node", os.path.join(work, "drive.cjs"), page, three, raw, setup], check=True, env=env)
    from PIL import Image

    im = Image.open(raw).convert("RGB")
    im = im.resize((im.width * 3 // 4, im.height * 3 // 4), Image.LANCZOS)
    im.save(os.path.join(img, name), optimize=True)

# the link as drawn and as tuned
with open(os.path.join(run, "tune", "report.json")) as fh:
    t = json.load(fh)
start = trimesh.load(os.path.join(run, "out", "link.stl"), force="mesh")
tuned = trimesh.load(os.path.join(run, "tuned", "link.stl"), force="mesh")
with open(os.path.join(run, "out", "report.json")) as fh:
    m0 = json.load(fh)["mass"]
with open(os.path.join(run, "tuned", "report.json")) as fh:
    m1 = json.load(fh)["mass"]
s = Scene("the link as drawn, and as tuned to SF 3 and 0.8 mm", panel_size=(560, 400))
s.panel(f"as drawn: {m0:.1f} g", view=(32, -62)).mesh(start)
s.panel(f"tuned: {m1:.1f} g, {t['best']['check']['deflection']:.2f} mm, SF {t['best']['check']['safetyFactor']:.1f}",
        view=(32, -62)).mesh(tuned)
s.render(os.path.join(img, "tuned.png"), out_scale=1.2)
