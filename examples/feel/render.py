"""
Renders for feel results (docs/images/feel/): each design cut in half
along its axis two ways and across at the capsule mount, one row per
design, so the bare skin and loadpath's interior sit side by side.

    NODE_PATH="$(npm root -g)" python examples/feel/render.py out.png "loadpath=mic-body.stl" [--z 50]
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "docs"))

from render3d import Scene  # noqa: E402


def row_scene(label, m, z):
    s = Scene(label, panel_size=(360, 420))
    s.panel("section x = 0", view=(4, 0)).mesh(m.slice_plane([0, 0, 0], [-1, 0, 0], cap=True))
    s.panel("section y = 0", view=(4, -90)).mesh(m.slice_plane([0, 0, 0], [0, 1, 0], cap=True))
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
    ap.add_argument("--z", type=float, default=50.0)
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
