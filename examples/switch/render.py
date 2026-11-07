"""
Renders for push-pad results (docs/images/switch/): each design seen from
below (the way the board and switch go in) and in two sections through the
press points, one row per design, so a drawn interior and loadpath's sit
side by side. The exterior is the same in every row.

    NODE_PATH="$(npm root -g)" python examples/switch/render.py out.png "round 17 as drawn=body.stl" "loadpath=part.stl"
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "docs"))

from render3d import Scene  # noqa: E402


def row_scene(label, m, title=None):
    """One design: from below; section x = 0; section y = 0 (looking at the cut)."""
    s = Scene(title or label, panel_size=(430, 340))
    s.panel("from below", view=(-55, -62)).mesh(m)
    s.panel("section x = 0", view=(4, 0)).mesh(m.slice_plane([0, 0, 0], [-1, 0, 0], cap=True))
    s.panel("section y = 0", view=(4, -90)).mesh(m.slice_plane([0, 0, 0], [0, 1, 0], cap=True))
    return s


def render_rows(rows, out):
    """rows: [(label, trimesh)], one rendered row each, stacked into out."""
    import tempfile

    from PIL import Image
    ims = []
    with tempfile.TemporaryDirectory() as d:
        for k, (label, m) in enumerate(rows):
            f = os.path.join(d, f"r{k}.png")
            row_scene(label, m).render(f)
            ims.append(Image.open(f).convert("RGB"))
    w = max(i.width for i in ims)
    page = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
    y = 0
    for i in ims:
        page.paste(i, (0, y))
        y += i.height
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    page.save(out, optimize=True)
    return out


def main():
    import trimesh
    out = sys.argv[1]
    rows = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        rows.append((label, trimesh.load(path, force="mesh")))
    render_rows(rows, out)
    print(out)


if __name__ == "__main__":
    main()
