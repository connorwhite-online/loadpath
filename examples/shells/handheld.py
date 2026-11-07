"""
A generic handheld (a remote, a meter): a rounded 96 x 56 x 26 mm exterior
with a board and a battery inside, split into a base and a cover with
`loadpath split`. The exterior and the parts are drawn here, so nothing is
read from outside the repository.

    python examples/shells/handheld.py --out /tmp/handheld            # the study, split
    python examples/shells/handheld.py --out /tmp/handheld --screws   # with screws through the underside
    python examples/shells/handheld.py --out /tmp/handheld --seam     # a parting line that sweeps up toward the back
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _stl(path, man):
    import trimesh
    m = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))
    tm.export(path)
    return os.path.basename(path)


def build(out: str) -> dict:
    """The exterior and component STLs, and the study."""
    import manifold3d as m3d
    os.makedirs(out, exist_ok=True)
    L, W, H, r = 96.0, 56.0, 26.0, 9.0
    # a soft box: the hull of eight spheres, its top crowned a little
    pts = []
    for x in (-L / 2 + r, L / 2 - r):
        for y in (-W / 2 + r, W / 2 - r):
            for z in (r, H - r):
                pts.append(m3d.Manifold.sphere(r, 48).translate((x, y, z)))
    ext = m3d.Manifold.batch_hull(pts)
    board = m3d.Manifold.cube((72, 42, 1.6)).translate((-40, -21, 5.0)) + \
        m3d.Manifold.cube((10, 8, 6)).translate((20, -4, 6.6))        # a connector block on the board
    battery = m3d.Manifold.cube((50, 30, 7.0)).translate((-34, -15, 8.6))
    files = {"exterior": _stl(os.path.join(out, "exterior.stl"), ext),
             "board": _stl(os.path.join(out, "board.stl"), board),
             "battery": _stl(os.path.join(out, "battery.stl"), battery)}
    study = {
        "name": "handheld",
        "description": "a generic handheld: fixed exterior, board and battery inside, split into base and cover",
        "designSpace": {"type": "mesh", "path": files["exterior"]},
        "skin": 1.6,
        "material": "pa12", "process": {"type": "sls"}, "minMember": 1.0,
        "keepOut": [{"id": "board", "shape": {"type": "mesh", "path": files["board"]}, "insertFrom": "+z",
                     "bore": False},
                    {"id": "battery", "shape": {"type": "mesh", "path": files["battery"]}, "insertFrom": "+z",
                     "bore": False}],
        "supports": [{"shape": {"type": "box", "min": [-60, -40, -1], "max": [60, 40, 0.4]}}],
        "loadCases": [{"name": "thumb", "loads": [{"shape": {"type": "box", "min": [-8, -8, 25.0],
                                                               "max": [8, 8, 27]}, "force": [0, 0, -60]}]}],
        "goal": {"volumeFraction": 0.25},
    }
    return study


def seam_line(study_dir: str, n: int = 96):
    """A designer's parting line on the exterior: low along the front, it
    sweeps up toward the back (z = 8 at x = -48 to z = 15 at x = +48, an
    S-curve), traced round the hull by walking out from the middle at each
    angle to where the exterior ends."""
    from loadpath.shapes import shape_from_spec
    ext = shape_from_spec({"type": "mesh", "path": "exterior.stl"}, study_dir)

    def height(x):
        return 11.5 + 3.5 * np.tanh(x / 22.0)

    pts = []
    for a in np.linspace(0.0, 2 * np.pi, n, endpoint=False):
        d = np.array([np.cos(a), np.sin(a), 0.0])
        z = 11.5
        for _ in range(4):
            ts = np.arange(0.0, 80.0, 0.05)
            P = np.array([0.0, 0.0, z]) + ts[:, None] * d
            k = int(np.argmax(ext.sdf(P) > 0))
            p = P[k]
            z = height(p[0])
        pts.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(z), 3)])
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="handheld-split")
    ap.add_argument("--screws", action="store_true", help="screws up through the underside, as well as snaps")
    ap.add_argument("--normal", default="auto")
    ap.add_argument("--at", type=float)
    ap.add_argument("--seam", action="store_true", help="split along a designer's line sweeping up toward the back")
    args = ap.parse_args(argv)
    from loadpath.shells import run_split
    study = build(args.out)
    split = {"normal": args.normal, "lip": {"height": 2.0, "width": 1.2}}
    if args.at is not None:
        split["at"] = args.at
    if args.seam:
        split["seam"] = {"points": seam_line(args.out)}
    if args.screws:
        split["screws"] = {"from": "-z", "size": "M2", "count": 4}
        split["snaps"] = {"count": 2}
    study["split"] = split
    sp = os.path.join(args.out, "handheld.study.json")
    json.dump(study, open(sp, "w"), indent=1)
    r = run_split(sp, os.path.join(args.out, "split"))
    rep = r["report"]
    print(json.dumps({k: rep[k] for k in ("cut", "snaps", "screws", "lip", "exteriorChanges", "warnings", "lint",
                                         "seconds")}, indent=1, default=float))
    return r


if __name__ == "__main__":
    main()
