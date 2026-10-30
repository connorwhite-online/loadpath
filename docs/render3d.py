"""
Smooth, antialiased renders for the README and docs/GALLERY.md, drawn by
three.js in headless Chromium: full-resolution meshes, creased smooth normals
(sharp at bores and cut faces, smooth on fillets, as in the viewer), studio
lighting, supersampled at 3x and scaled down.

    s = Scene("bell-crank: 24.7 g")
    p = s.panel("front", view=(24, -58))
    p.mesh(mesh)                                     # clay
    p.mesh(keep_out, color=(0.9, 0.25, 0.2), opacity=0.15)
    p.arrow(at, force_direction * 20)
    s.render("out.png")

Panels sit side by side, each with its own camera; meshes take one colour,
or a colour per vertex or per face (stress maps, ceilings). Needs Node with
Playwright's Chromium (`npm i -g playwright && npx playwright install
chromium`); three.js is fetched once into ~/.cache/loadpath.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THREE = "0.160.0"
_THREE_FILES = {"three.module.js": "build/three.module.js",
                "BufferGeometryUtils.js": "examples/jsm/utils/BufferGeometryUtils.js"}
CLAY = (0.78, 0.78, 0.80)
LOAD = (0.15, 0.45, 0.95)
KEEP_OUT = (0.92, 0.28, 0.22)
FLUIDS = [(0.20, 0.45, 0.85), (0.92, 0.55, 0.18)]


def _three_dir():
    d = os.path.join(os.path.expanduser("~"), ".cache", "loadpath", f"three-{THREE}")
    for name, src in _THREE_FILES.items():
        f = os.path.join(d, name)
        if not os.path.exists(f):
            os.makedirs(d, exist_ok=True)
            with urllib.request.urlopen(f"https://cdn.jsdelivr.net/npm/three@{THREE}/{src}", timeout=60) as r:
                data = r.read()
            with open(f + ".part", "wb") as fh:
                fh.write(data)
            os.replace(f + ".part", f)
    return d


def ramp(t):
    """The viewer's stress ramp: Turbo from its blue to its red (its ends are
    near black). t in [0, 1] -> (n, 3) RGB."""
    import matplotlib

    return matplotlib.colormaps["turbo"](0.1 + 0.85 * np.clip(np.asarray(t, float), 0, 1))[..., :3]


class Panel:
    def __init__(self, label, view, center, span):
        self.label, self.view, self.center, self.span = label, view, center, span
        self.meshes, self.arrows = [], []

    def mesh(self, mesh, color=CLAY, opacity=1.0, vertex_colors=None, face_colors=None,
             crease=36.0, roughness=0.6, metalness=0.05):
        """`crease` (degrees): corners sharper than this stay sharp."""
        self.meshes.append(dict(mesh=mesh, color=color, opacity=opacity, vc=vertex_colors, fc=face_colors,
                                crease=crease, roughness=roughness, metalness=metalness))
        return self

    def arrow(self, tip_at, vector, color=LOAD):
        """An arrow of length |vector| pointing along it and ending at `tip_at`."""
        self.arrows.append(dict(tip=np.asarray(tip_at, float).tolist(), vec=np.asarray(vector, float).tolist(),
                                color=list(color)))
        return self

    def points(self):
        """What the camera must fit: the meshes' hulls and the arrows, or the
        box given as center and span."""
        if self.center is not None:
            c = np.asarray(self.center, float)
            return c + (np.array([[i & 1, i >> 1 & 1, i >> 2 & 1] for i in range(8)]) * 2 - 1) * self.span / 2
        pts = []
        for m in self.meshes:
            try:
                pts.append(np.asarray(m["mesh"].convex_hull.vertices))
            except Exception:  # noqa: BLE001 - flat or tiny: its box will do
                lo, hi = m["mesh"].bounds
                pts.append(np.array([[(lo, hi)[i & 1][0], (lo, hi)[i >> 1 & 1][1], (lo, hi)[i >> 2 & 1][2]]
                                     for i in range(8)]))
        for a in self.arrows:
            pts.append(np.array([a["tip"], np.subtract(a["tip"], a["vec"])]))
        return np.concatenate(pts)


class Scene:
    def __init__(self, title=None, panel_size=(560, 480), colorbar=None):
        """`colorbar`: (label, low text, high text) for a ramp legend at the right."""
        self.title, self.panel_size, self.colorbar = title, panel_size, colorbar
        self.panels = []

    def panel(self, label=None, view=(24, -58), center=None, span=None):
        p = Panel(label, view, center, span)
        self.panels.append(p)
        return p

    def render(self, path, scale=3, out_scale=1.5):
        """Render at `scale` x the CSS size, save at `out_scale` x (Lanczos)."""
        work = tempfile.mkdtemp(prefix="render3d-")
        try:
            os.makedirs(os.path.join(work, "three"))
            for name in _THREE_FILES:
                shutil.copy(os.path.join(_three_dir(), name), os.path.join(work, "three", name))
            shutil.copy(os.path.join(HERE, "render3d.html"), os.path.join(work, "index.html"))
            spec = self._write(work)
            with open(os.path.join(work, "scene.json"), "w") as fh:
                json.dump(spec, fh)
            tmp = os.path.join(work, "out.png")
            subprocess.run(["node", os.path.join(HERE, "render3d.cjs"), work, tmp, str(scale)], check=True)
            from PIL import Image

            im = Image.open(tmp).convert("RGB")
            if out_scale != scale:
                im = im.resize((round(im.width * out_scale / scale), round(im.height * out_scale / scale)),
                               Image.LANCZOS)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            im.save(path, optimize=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return path

    def _write(self, work):
        w, h = self.panel_size
        head = 34 if self.title else 0
        label_h = 26 if any(p.label for p in self.panels) else 0
        bar_w = 110 if self.colorbar else 0
        spec = {"page": [w * len(self.panels) + bar_w, h + head + label_h], "title": self.title,
                "panel": [w, h], "labelHeight": label_h, "panels": []}
        if self.colorbar:
            label, lo, hi = self.colorbar
            spec["colorbar"] = {"label": label, "lo": lo, "hi": hi,
                                "stops": [("#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb))
                                          for rgb in ramp(np.linspace(0, 1, 32))]}
        k = 0
        for p in self.panels:
            pts = p.points()
            if len(pts) > 4000:
                pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
            ps = {"label": p.label, "view": list(p.view), "points": np.round(pts, 3).tolist(),
                  "meshes": [], "arrows": p.arrows}
            for m in p.meshes:
                mesh = m["mesh"]
                tris = np.asarray(mesh.vertices, np.float32)[np.asarray(mesh.faces)]
                name = f"m{k}"
                k += 1
                tris.reshape(-1).astype("<f4").tofile(os.path.join(work, name + ".pos.bin"))
                ms = {"pos": name + ".pos.bin", "color": list(m["color"]), "opacity": m["opacity"],
                      "crease": m["crease"], "roughness": m["roughness"], "metalness": m["metalness"]}
                cols = None
                if m["vc"] is not None:
                    cols = np.asarray(m["vc"], np.float32)[np.asarray(mesh.faces)]
                elif m["fc"] is not None:
                    cols = np.repeat(np.asarray(m["fc"], np.float32)[:, None, :], 3, axis=1)
                if cols is not None:
                    cols.reshape(-1).astype("<f4").tofile(os.path.join(work, name + ".col.bin"))
                    ms["col"] = name + ".col.bin"
                ps["meshes"].append(ms)
            spec["panels"].append(ps)
        return spec


def view_dir(view):
    e, a = np.radians(view[0]), np.radians(view[1])
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
