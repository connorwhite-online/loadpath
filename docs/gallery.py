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
other cases' existing images). Needs the render extra (matplotlib, shapely).
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

from loadpath.render import _VIEWS, render_heat_exchanger, render_stress, render_views  # noqa: E402

CASES = os.path.join(ROOT, "loadpath", "evals", "cases")
IMAGES = os.path.join(ROOT, "docs", "images")
README = os.path.join(ROOT, "README.md")
GALLERY = os.path.join(ROOT, "docs", "GALLERY.md")
START, END = "<!-- gallery:start -->", "<!-- gallery:end -->"
COLS = 3
# Stress maps shown in the README (the rest are in docs/GALLERY.md).
STRESS_PICKS = ["bell-crank", "printed-hook", "grip-upper-arm"]

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


def render_case(row, evals, stress):
    import trimesh

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
    out["thumb"] = _crop(render_views(mesh, os.path.join(IMAGES, "cases", f"{name}.png"), study,
                                views=[_VIEWS[0]], size=(7.0, 6.3)))
    out["views"] = render_views(mesh, os.path.join(IMAGES, "cases", f"{name}-views.png"), study, title=title)
    if stress:
        print(f"   {name}: stress FEA", file=sys.stderr, flush=True)
        fields = verify_part(study, mesh, refine=0, fields=True)["fields"]
        # Coloured against the part's own stress (its 98th percentile, so a
        # support's singular corner doesn't wash it out): this shows the load
        # paths. The margin against the material is the SF in the caption.
        vm = np.max([c["vonMises"] for c in fields["cases"]], axis=0)
        scale = float(np.percentile(vm, 98)) or 1.0
        label = "von Mises / part's 98th percentile"
        out["stressThumb"] = _crop(render_stress(mesh, fields, os.path.join(IMAGES, "stress", f"{name}.png"),
                                                 scale, views=[_VIEWS[0]], size=(7.0, 6.3), label=label))
        out["stress"] = render_stress(mesh, fields, os.path.join(IMAGES, "stress", f"{name}-views.png"),
                                      scale, title=name, label=label)
    return out


def render_heatx(name, spec, step, normal=None):
    from loadpath.heatx import design_heat_exchanger, heat_exchanger_from_spec

    hx = heat_exchanger_from_spec(spec)
    hx.step = step
    mesh, rep, spaces = design_heat_exchanger(hx, fluid_meshes=True)
    title = f"{name}: cell {hx.cell:g} mm, wall {hx.wall:g} mm, " + \
        ("fluids isolated" if rep.get("isolated") else "FLUIDS NOT ISOLATED")
    lo, hi = hx.core.bounds()
    cut = None if normal is None else (0.5 * (np.asarray(lo) + np.asarray(hi)), np.asarray(normal, float))
    return render_heat_exchanger(mesh, spaces, hx.core, os.path.join(IMAGES, "heatx", f"{name}.png"),
                                 title=title, cut=cut)


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
    write_docs(rows, args.evals)
    print(f"wrote {_rel(README)}, {_rel(GALLERY)} and {_rel(IMAGES)}/", file=sys.stderr)


if __name__ == "__main__":
    main()
