"""
Regenerate the README gallery: renders of the eval case studies, their
stress maps, and a few heat exchangers, written to docs/images/ and listed
in README.md between the gallery markers (and in full in docs/GALLERY.md).

Run after an eval run, from the repo root:

    python -m loadpath.evals.run --out evals-out/
    python docs/gallery.py --evals evals-out/

It reuses the eval run's meshes (evals-out/<case>.stl) and scorecard, so
nothing is optimized again. Each part gets one fresh FEA (verify_part with
fields) for its stress map; --no-stress skips that, --no-heatx skips the heat
exchangers, and naming cases limits the run to them (the README keeps the
other cases' existing images). Renders with three.js in headless Chromium
(docs/render3d.py): needs Node with Playwright's Chromium, and the render
extra (matplotlib, shapely).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render3d import FLUIDS, KEEP_OUT, Scene, ramp  # noqa: E402

VIEWS = [(24, -58), (24, 32), (78, -90)]

CASES = os.path.join(ROOT, "loadpath", "evals", "cases")
IMAGES = os.path.join(ROOT, "docs", "images")
README = os.path.join(ROOT, "README.md")
GALLERY = os.path.join(ROOT, "docs", "GALLERY.md")
START, END = "<!-- gallery:start -->", "<!-- gallery:end -->"
COLS = 3
# Stress maps shown in the README (the rest are in docs/GALLERY.md).
STRESS_PICKS = ["bell-crank", "printed-hook", "grip-upper-arm"]

# The metal heat exchanger case whose internal ceilings the gallery shows.
METAL_CASE = "lpbf-intercooler-core"

# Heat exchangers shown in the gallery: (name, caption, spec, grid step mm, cut normal or None).
HEATX = [
    ("box-crossflow", "Gyroid crossflow in a box: A left to right, B front to back",
     {"core": {"type": "box", "min": [0, 0, 0], "max": [40, 40, 30]},
      "layout": "cross", "axes": "xy", "cell": 10, "wall": 0.8, "header": 5, "portDiameter": 6, "portLength": 5}, 0.4, None),
    ("cylinder-counterflow", "Gyroid counterflow in a cylinder: A through the ends, B in and out the side",
     {"core": {"type": "cylinder", "center": [0, 0, 25], "axis": "z", "length": 50, "r": 15},
      "layout": "counter", "axes": "zx", "cell": 9, "wall": 1.0, "header": 5, "portDiameter": 6, "portLength": 5}, 0.4, (0, 1, 0)),
]


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - outside a checkout
        return None


def _crop(path, pad=12):
    """Trim the white round a render, so thumbnails fill their cell."""
    from PIL import Image, ImageChops

    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).point(lambda v: 255 if v > 8 else 0).getbbox()
    if box:
        l, t, r, b = box
        im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height))).save(path)
    return path


def _shape_mesh(shape):
    import trimesh

    m = shape.to_manifold().to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)


def _study_panel(scene, study, mesh, view, label=None, vertex_colors=None):
    """The part, its keep-outs (translucent red) and loads (blue arrows)."""
    p = scene.panel(label, view=view)
    p.mesh(mesh, vertex_colors=vertex_colors)
    if study is None:
        return p
    span = float(np.max(np.subtract(*reversed(study.bounds()))))
    for r in study.keep_out:
        try:
            p.mesh(_shape_mesh(r.shape), color=KEEP_OUT, opacity=0.16, crease=30)
        except Exception:  # noqa: BLE001 - a keep-out that won't mesh is left out of the picture
            pass
    if vertex_colors is None:
        for case in study.load_cases:
            for ld in case.loads:
                if not np.any(ld.force):
                    continue
                shape = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
                c = 0.5 * np.add(*shape.bounds())
                p.arrow(c, np.asarray(ld.force, float) / np.linalg.norm(ld.force) * span * 0.2)
    return p


def render_case(row, evals, stress):
    import trimesh
    from scipy.spatial import cKDTree

    from loadpath.spec import study_from_dict
    from loadpath.verify import verify_part

    name = row["case"]
    if row.get("kind") == "heatExchanger":
        return None                                # not a study: no stress map, no study render
    stl = os.path.join(evals, f"{name}.stl")
    if not os.path.exists(stl):
        print(f"skip {name}: no {stl}", file=sys.stderr)
        return None
    path = os.path.join(CASES, f"{name}.json")
    with open(path) as fh:
        study = study_from_dict(json.load(fh), base_dir=CASES)
    mesh = trimesh.load(stl, force="mesh")
    b = row.get("best") or {}
    title = f"{name}: {b.get('mass', mesh.volume * study.material.density / 1000):.1f} g"
    if b.get("safetyFactor") is not None:
        title += f", SF {b['safetyFactor']:.2f}"
    out = {"case": name}
    s = Scene(panel_size=(600, 520))
    _study_panel(s, study, mesh, VIEWS[0])
    out["thumb"] = _crop(s.render(os.path.join(IMAGES, "cases", f"{name}.png")))
    s = Scene(title, panel_size=(520, 460))
    for v in VIEWS:
        _study_panel(s, study, mesh, v)
    out["views"] = s.render(os.path.join(IMAGES, "cases", f"{name}-views.png"))
    if stress:
        print(f"   {name}: stress FEA", file=sys.stderr, flush=True)
        fields = verify_part(study, mesh, refine=0, fields=True)["fields"]
        # Coloured against the part's own stress (its 98th percentile, so a
        # support's singular corner doesn't wash it out): this shows the load
        # paths. The margin against the material is the SF in the caption.
        vm_nodes = np.max([c["vonMises"] for c in fields["cases"]], axis=0)
        scale = float(np.percentile(vm_nodes, 98)) or 1.0
        # each vertex takes its nearest analysis nodes' values, inverse-distance
        # weighted, so the voxel grid's steps don't show as speckle
        dist, idx = cKDTree(fields["nodes"]).query(mesh.vertices, k=8)
        w = 1.0 / np.maximum(dist, 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        worst = max(fields["cases"], key=lambda c: np.linalg.norm(c["displacement"], axis=1).max())
        disp = np.einsum("vk,vkd->vd", w, worst["displacement"][idx])
        dmax = np.linalg.norm(disp, axis=1).max()
        k = 0.04 * float(np.ptp(mesh.bounds, axis=0).max()) / dmax if dmax > 0 else 0.0
        bent = trimesh.Trimesh(mesh.vertices + k * disp, mesh.faces, process=False)
        colors = ramp((w * vm_nodes[idx]).sum(axis=1) / scale)
        bar = ("von Mises stress", "0", "98th percentile")
        s = Scene(panel_size=(600, 520), colorbar=bar)
        _study_panel(s, None, bent, VIEWS[0], vertex_colors=colors)
        out["stressThumb"] = _crop(s.render(os.path.join(IMAGES, "stress", f"{name}.png")))
        s = Scene(f"{name}: stress, shape {k:.0f}x exaggerated ({worst['name']})" if k else name,
                  panel_size=(520, 460), colorbar=bar)
        for v in VIEWS[:2]:
            _study_panel(s, None, bent, v, vertex_colors=colors)
        out["stress"] = s.render(os.path.join(IMAGES, "stress", f"{name}-views.png"))
    return out


def render_heatx(name, spec, step, normal=None, view=(28, -55)):
    """The part, the part cut open, both fluid spaces cut open at the same
    plane (A blue, B orange), and A alone. The cut keeps the side its normal
    points to, so point it away from the view."""
    from loadpath.heatx import design_heat_exchanger, heat_exchanger_from_spec

    hx = heat_exchanger_from_spec(spec)
    hx.step = step
    mesh, rep, spaces = design_heat_exchanger(hx, fluid_meshes=True)
    title = f"{name}: cell {hx.cell:g} mm, wall {hx.wall:g} mm, " + \
        ("fluids isolated" if rep.get("isolated") else "FLUIDS NOT ISOLATED")
    lo, hi = (np.asarray(b, float) for b in hx.core.bounds())
    c0 = 0.5 * (lo + hi)
    cn = np.asarray(normal if normal is not None else (0, 0, 1), float)
    cn /= np.linalg.norm(cn)
    names = list(spaces)
    cut = {n: spaces[n].slice_plane(c0, cn, cap=True) for n in names}
    s = Scene(title, panel_size=(480, 460))
    s.panel("part", view=view).mesh(mesh)
    s.panel("part, cut open", view=view).mesh(mesh.slice_plane(c0, cn, cap=True))
    p = s.panel(" + ".join(names) + ", cut open", view=view)
    for i, n in enumerate(names):
        p.mesh(cut[n], color=FLUIDS[i % 2])
    s.panel(names[0], view=view).mesh(cut[names[0]], color=FLUIDS[0])
    return s.render(os.path.join(IMAGES, "heatx", f"{name}.png"))


def render_metal_ceilings(flat=20.0, view=(30, -55)):
    """The lpbf eval exchanger's fluid spaces with and without channel
    closure, flat internal ceilings (within `flat` degrees of level, facing
    down onto the fluid from the metal above) in red."""
    from loadpath.heatx import design_heat_exchanger, heat_exchanger_from_spec

    with open(os.path.join(CASES, f"{METAL_CASE}.json")) as fh:
        spec = json.load(fh)
    step = spec.get("step", 0.45)
    spec = {k: v for k, v in spec.items() if k not in ("name", "description", "duty", "step", "expect")}
    s = Scene("AlSi10Mg intercooler core built +z: fluid spaces, flat ceilings in red (unsupportable in metal)",
              panel_size=(620, 540))
    up = np.array([0.0, 0.0, 1.0])
    red = np.array([0.85, 0.10, 0.08])
    for closure in (False, True):
        hx = heat_exchanger_from_spec({**spec, "closure": closure, "cleanouts": 0})
        hx.step = step
        _, rep, spaces = design_heat_exchanger(hx, fluid_meshes=True)
        flat_cm2 = rep["internalCeilings"]["flatArea"] / 100.0
        p = s.panel(("channels close into peaked roofs" if closure else "flat skin over the channels")
                    + f": {flat_cm2:.1f} cm² flat inside", view=view)
        for i, m in enumerate(spaces.values()):
            ceil = (m.face_normals @ up) > np.cos(np.radians(flat))
            p.mesh(m, face_colors=np.where(ceil[:, None], red[None], np.asarray(FLUIDS[i % 2])[None]))
    return s.render(os.path.join(IMAGES, "heatx", "lpbf-ceilings.png"))


def _rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def _caption(row):
    b = row.get("best") or {}
    bits = []
    if b.get("mass") is not None:
        bits.append(f"{b['mass']:.1f} g")
    if b.get("massReduction") is not None:
        bits.append(f"{b['massReduction']:.0%} lighter")
    if b.get("safetyFactor") is not None:
        bits.append(f"SF {b['safetyFactor']:.2f}")
    bits.append("pass" if row.get("pass") else "**fail**")
    return " · ".join(bits)


def _grid(items, width=280):
    """items: (image path, caption). An HTML table, COLS across."""
    lines = ["<table>"]
    for i in range(0, len(items), COLS):
        lines.append("<tr>")
        for img, cap in items[i:i + COLS]:
            lines.append(f'<td align="center" valign="top"><img src="{_rel(img)}" width="{width}"><br>{cap}</td>')
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _stamp(evals):
    sha = _git("rev-parse", "--short", "HEAD")
    day = datetime.date.today().isoformat()
    return (f"Rendered {day}" + (f" from `{sha}`" if sha else "") +
            " by `python docs/gallery.py`.")


def write_docs(rows, evals):
    """Lists whatever images exist in docs/images, so a partial run keeps the rest."""
    exist = lambda *p: os.path.exists(os.path.join(IMAGES, *p))  # noqa: E731
    img = lambda *p: os.path.join(IMAGES, *p)  # noqa: E731
    cases = [r for r in rows if exist("cases", f"{r['case']}.png")]
    passed = sum(1 for r in rows if r.get("pass"))
    # The README shows the tool off in a screenful; docs/GALLERY.md has everything.
    readme = [START, "", "## Examples", "",
              f"The eval cases ({passed}/{len(rows)} passing): GRIP robot-arm style parts, each optimized "
              f"and then checked by a fresh FEA of the delivered mesh. Blue arrows are loads, red is keep-out. "
              f"Every view, stress map and number is in [docs/GALLERY.md](docs/GALLERY.md).", "",
              _grid([(img("cases", f"{r['case']}.png"), f"<b>{r['case']}</b><br><sub>{_caption(r)}</sub>")
                     for r in cases], width=250)]
    stress = [r for r in cases if exist("stress", f"{r['case']}.png")]
    stress = ([r for r in stress if r["case"] in STRESS_PICKS] or stress)[:COLS]
    if stress:
        readme += ["", "Where the load goes: von Mises stress on the part bent by its worst load case:", "",
                   _grid([(img("stress", f"{r['case']}.png"), f"<sub>{r['case']}</sub>") for r in stress], width=250)]
    hx = [(n, cap) for n, cap, *_ in HEATX if exist("heatx", f"{n}.png")]
    if hx:
        n, cap = hx[0]
        readme += ["", "Two-fluid heat exchangers: two isolated channel networks through one sheet gyroid, "
                   "with headers that print without support.", "", f"![{n}](docs/images/heatx/{n}.png)"]
    if exist("heatx", "lpbf-ceilings.png"):
        readme += ["", "Printed in metal, nothing inside can be supported: the channels end in peaked roofs "
                   "instead of flat ceilings (red).", "", "![metal ceilings](docs/images/heatx/lpbf-ceilings.png)"]
    readme += ["", f"<sub>{_stamp(evals)}</sub>", "", END]

    with open(README) as fh:
        text = fh.read()
    block = "\n".join(readme)
    if START in text and END in text:
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        # first run: before Install
        at = text.index("## Install") if "## Install" in text else len(text)
        text = text[:at] + block + "\n\n" + text[at:]
    with open(README, "w") as fh:
        fh.write(text)

    g = ["# Gallery", "", _stamp(evals), "", "Regenerate with the commands at the top of "
         "[`gallery.py`](gallery.py). Stress maps colour worst-case von Mises stress against the part's "
         "own 98th percentile, to show the load paths, on the shape bent by the worst load case, "
         "exaggerated; the margin against the material is the SF.", ""]
    for r in cases:
        g += [f"## {r['case']}", "", _caption(r), ""]
        b = r.get("best") or {}
        nums = [(k, b.get(k), f) for k, f in (("deflection", "{:.3f} mm"), ("p5Member", "{:.2f} mm"),
                                               ("creaseLength", "{:.1f} mm"), ("volumeFraction", "{:.0%}"))]
        nums = [f"{k} {f.format(v)}" for k, v, f in nums if v is not None]
        if nums:
            g += ["<sub>" + " · ".join(nums) + f" · {r.get('seconds', 0):.0f} s</sub>", ""]
        if exist("cases", f"{r['case']}-views.png"):
            g += [f"![{r['case']}](images/cases/{r['case']}-views.png)", ""]
        if exist("stress", f"{r['case']}-views.png"):
            g += [f"![{r['case']} stress](images/stress/{r['case']}-views.png)", ""]
    if hx:
        g += ["## Heat exchangers", ""]
        for n, cap in hx:
            g += [f"### {n}", "", cap, "", f"![{n}](images/heatx/{n}.png)", ""]
    if exist("heatx", "lpbf-ceilings.png"):
        g += ["### Printed in metal", "",
              f"The `{METAL_CASE}` eval case built along +z in AlSi10Mg. Nothing inside a metal part can be "
              "supported, so every flat ceiling over a channel is a defect: with `closure` the channels end in "
              "peaked roofs under the top skin and flare open over plenums below.", "",
              "![lpbf ceilings](images/heatx/lpbf-ceilings.png)", ""]
    with open(GALLERY, "w") as fh:
        fh.write("\n".join(g))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cases", nargs="*", help="only these cases (default: every case in the scorecard)")
    p.add_argument("--evals", default="evals-out", help="eval output dir (scorecard.json + <case>.stl)")
    p.add_argument("--no-stress", action="store_true", help="skip the stress maps (one FEA per case)")
    p.add_argument("--no-heatx", action="store_true", help="skip the heat exchangers")
    args = p.parse_args(argv)
    for d in ("cases", "stress", "heatx"):
        os.makedirs(os.path.join(IMAGES, d), exist_ok=True)
    with open(os.path.join(args.evals, "scorecard.json")) as fh:
        rows = json.load(fh)
    for row in rows:
        if args.cases and row["case"] not in args.cases:
            continue
        print(f"== {row['case']}", file=sys.stderr, flush=True)
        render_case(row, args.evals, not args.no_stress)
    if not args.no_heatx:
        for name, _cap, spec, step, normal in HEATX:
            print(f"== heatx {name}", file=sys.stderr, flush=True)
            render_heatx(name, spec, step, normal)
        print("== heatx metal ceilings", file=sys.stderr, flush=True)
        render_metal_ceilings()
    write_docs(rows, args.evals)
    print(f"wrote {_rel(README)}, {_rel(GALLERY)} and {_rel(IMAGES)}/", file=sys.stderr)


if __name__ == "__main__":
    main()
