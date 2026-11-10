"""
Render coreshape.compare's three exchangers side by side (docs/SHROUD.md):
the box, the duty-grown core and the box scaled to the grown core's volume.

    python examples/shroud/render_exchanger.py COMPARE_DIR OUT.png
"""
import json
import os
import sys

import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "docs"))
from render3d import Scene  # noqa: E402


def lighter(m, faces=400_000):
    if len(m.faces) <= faces:
        return m
    import fast_simplification
    v, f = fast_simplification.simplify(m.vertices, m.faces, 1 - faces / len(m.faces))
    return trimesh.Trimesh(v, f)


def main(d, out):
    rows = json.load(open(os.path.join(d, "compare.json")))["rows"]
    s = Scene(None, panel_size=(620, 520))
    names = {"box": "box core", "grown": "grown by its own duty", "smallBox": "box, same core volume as grown"}
    for k in ("box", "grown", "smallBox"):
        r = rows[k]
        dp = r["fluids"]
        air, liq = list(dp)
        label = (f"{names[k]}: {r['mass']:.0f} g, eff {r['effectiveness']:.2f}, "
                 f"dp {dp[air]['pressureDrop'] / 1000:.1f} / {dp[liq]['pressureDrop'] / 1000:.1f} kPa")
        s.panel(label, view=(26, -55)).mesh(lighter(trimesh.load(os.path.join(d, f"{k}.stl"), force="mesh")),
                                             metalness=0.35, roughness=0.45)
    s.render(out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
