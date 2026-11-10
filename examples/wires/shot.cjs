// Screenshots of a wires viewer page (wires.html) in headless Chromium:
//
//   node examples/wires/shot.cjs <wires.html> <out.png> [query] [width] [height]
//
// query: the page's own options, e.g. "bare=1&step=9&az=-60&el=25&zoom=1.2&env=1&theme=light".
// three.js is served from ~/.cache/loadpath (fetched once), so no network is
// needed beyond that.
const fs = require("fs");
const os = require("os");
const path = require("path");
const https = require("https");
const { chromium } = require("playwright");

const VER = "0.160.0";
const cache = path.join(os.homedir(), ".cache", "loadpath", `three-${VER}`);
const FILES = { "build/three.module.js": "three.module.js", "examples/jsm/controls/OrbitControls.js": "OrbitControls.js",
  "examples/jsm/utils/BufferGeometryUtils.js": "BufferGeometryUtils.js" };

function fetch(url, file) {
  return new Promise((res, rej) => {
    https.get(url, (r) => {
      if (r.statusCode !== 200) return rej(new Error(url + " " + r.statusCode));
      const out = fs.createWriteStream(file + ".part");
      r.pipe(out); out.on("finish", () => { fs.renameSync(file + ".part", file); res(); });
    }).on("error", rej);
  });
}

(async () => {
  const [html, out, query, w, h] = process.argv.slice(2);
  fs.mkdirSync(cache, { recursive: true });
  for (const [src, name] of Object.entries(FILES)) {
    const f = path.join(cache, name);
    if (!fs.existsSync(f)) await fetch(`https://cdn.jsdelivr.net/npm/three@${VER}/${src}`, f);
  }
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  try {
    const page = await browser.newPage({ viewport: { width: +(w || 1280), height: +(h || 800) }, deviceScaleFactor: 2 });
    page.on("pageerror", (e) => console.error("page:", e.message));
    await page.route("**/*", (route) => {
      const u = route.request().url();
      if (u.startsWith("http://shot.local/")) return route.fulfill({ status: 200, body: fs.readFileSync(html), contentType: "text/html" });
      for (const [src, name] of Object.entries(FILES)) if (u.endsWith(src))
        return route.fulfill({ status: 200, body: fs.readFileSync(path.join(cache, name)), contentType: "text/javascript" });
      return route.fulfill({ status: 200, body: "", contentType: u.includes("css") ? "text/css" : "text/plain" });
    });
    await page.goto("http://shot.local/index.html?" + (query || ""));
    await page.waitForFunction("window.renderDone === true", null, { timeout: 120000 });
    await page.waitForTimeout(400);
    await page.screenshot({ path: out });
  } finally { await browser.close(); }
})().catch((e) => { console.error(e.message || e); process.exit(1); });
