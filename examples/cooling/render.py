"""
Renders for cooling results (docs/images/cooling/): each part cut through
the processor along its length, across, and from above just under the top
skin, one row per part.

    NODE_PATH="$(npm root -g)" python examples/cooling/render.py out.png "loadpath=handheld.stl" [--z 5.5]
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "docs"))

from render3d import Scene  # noqa: E402

SOC_X = 28.0     # the processor's centre (handheld.py)


def row_scene(label, m, z):
    s = Scene(label, panel_size=(520, 300))
    s.panel("section y = 0 (through the processor)", view=(4, -90)).mesh(m.slice_plane([0, 0, 0], [0, 1, 0], cap=True))
    s.panel(f"section x = {SOC_X:g}", view=(4, 0)).mesh(m.slice_plane([SOC_X, 0, 0], [-1, 0, 0], cap=True))
    top = m.slice_plane([0, 0, z], [0, 0, -1], cap=True)
    s.panel(f"section z = {z:g}, from above", view=(89, -90)).mesh(top)
    return s


def main():
    import tempfile

    import trimesh
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("rows", nargs="+", help="label=path.stl")
    ap.add_argument("--z", type=float, default=5.5)
    a = ap.parse_args()
    ims = []
    with tempfile.TemporaryDirectory() as d:
        for k, spec in enumerate(a.rows):
            label, path = spec.split("=", 1)
            f = os.path.join(d, f"r{k}.png")
            row_scene(label, trimesh.load(path, force="mesh"), a.z).render(f)
            ims.append(Image.open(f).convert("RGB"))
    w = max(i.width for i in ims)
    page = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
    y = 0
    for i in ims:
        page.paste(i, (0, y))
        y += i.height
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    page.save(a.out, optimize=True)
    print(a.out)


if __name__ == "__main__":
    main()
