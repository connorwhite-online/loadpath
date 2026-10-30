// Headless three.js renderer for docs/render3d.py: opens the scene page in
// Chromium (software WebGL, so no GPU needed) and screenshots it.
//
//   node docs/render3d.cjs <scene dir> <out.png> <device scale>
//
// The scene dir holds index.html, scene.json and the mesh buffers; three.js
// comes from the dir's three/ folder. Everything is served from a made-up
// origin, since module imports don't load from file://.
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".bin": "application/octet-stream" };

(async () => {
  const [dir, out, scale] = process.argv.slice(2);
  const browser = await chromium.launch({
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
  });
  try {
    const [width, height] = JSON.parse(fs.readFileSync(path.join(dir, "scene.json"))).page;
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: Number(scale || 3) });
    page.on("console", (m) => { if (m.type() === "error") console.error("page:", m.text()); });
    page.on("pageerror", (e) => console.error("page:", e.message));
    await page.route("http://render.local/**", (route) => {
      const rel = decodeURIComponent(new URL(route.request().url()).pathname).replace(/^\/+/, "");
      const file = path.resolve(dir, rel);
      if (!file.startsWith(path.resolve(dir)) || !fs.existsSync(file)) return route.fulfill({ status: 404, body: "" });
      route.fulfill({ status: 200, body: fs.readFileSync(file), contentType: TYPES[path.extname(file)] || "application/octet-stream" });
    });
    await page.goto("http://render.local/index.html");
    await page.waitForFunction("window.renderDone === true || window.renderError", null, { timeout: 600000 });
    const err = await page.evaluate("window.renderError || null");
    if (err) throw new Error(err);
    await page.screenshot({ path: out });
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error(e.message || e); process.exit(1); });
