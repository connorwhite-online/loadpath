"""
GRIP's TPU finger curled by its tendon: a large-deflection check with
loadpath.nonlinear's tendon loads.

The finger (hand/finger.stl, from GRIP's generative/hand.py at 3689f8d) is
one TPU 95A print: three phalanges on flexure strips 2.2 mm thick at the
back, V-notches on the palm side, a tendon channel 3.2 mm in from the palm
face and a knot pocket near the tip. Local frame: root face on z = 0, length
along +z, palm side +y.

The model: the root block is clamped in its palm cup; a cable at constant
tension runs from the palm exit below the root, through guides one channel
diameter apart along the channel (none inside the notches, where there is no
wall), to the knot on the pocket floor. Each step reports the distal
phalanx's curl, the tip's travel and how much cable the servo has pulled in.

    python examples/grip/finger.py --tension 17 --pitch 0.73     # ~3 h on 4 cores
    python examples/grip/finger.py --pitch 0.55                  # 4 cells across the flexure, ~10 GB

At 0.73 mm the flexure is 3 cells thick, which makes it about 2x too stiff
(shear locking; see nonlinear.py); 0.55 mm brings that to a few percent.
On main at 37f5ac8, 0.73 mm: 4.9 deg at 0.83 N, 48 deg at 7.5 N, 120 deg at
16.7 N with 15 mm of cable pulled in, nearly linear; with the locking taken
out that is close to hand.py's beam estimate (3 x 70 deg at 16.7 N). There
is no self-contact, so past about 180 deg the curve is not physical.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TPU_E, TPU_NU = 26.0, 0.45      # MPa; TPU 95A (datasheets 20-40: VERIFY with a coupon)
CHANNEL = (0.0, 3.3)            # tendon channel centre (x, y)
CHANNEL_R = 1.1
HINGES = (10.0, 48.0, 74.0)     # z of the three V-notches
KNOT_FROM_TIP = 7.0             # knot pocket floor, mm below the tip


def voxels(mesh, pitch):
    """Solid cells of a closed mesh: one ray per column along +z, inside
    between hit pairs (mesh.contains on every cell centre runs out of memory)."""
    lo, hi = mesh.bounds
    n = np.ceil((hi - lo) / pitch).astype(int)
    I, J = np.meshgrid(np.arange(n[0]), np.arange(n[1]), indexing="ij")
    I, J = I.ravel(), J.ravel()
    org = np.stack([lo[0] + (I + 0.5) * pitch, lo[1] + (J + 0.5) * pitch, np.full(I.size, lo[2] - 1.0)], 1)
    loc, ri, _ = mesh.ray.intersects_location(org, np.tile([0.0, 0.0, 1.0], (len(org), 1)), multiple_hits=True)
    grid = np.zeros(tuple(n), bool)
    zc = lo[2] + (np.arange(n[2]) + 0.5) * pitch
    for r in range(len(org)):
        z = np.sort(loc[ri == r, 2])
        for a, b in zip(z[0::2], z[1::2]):
            grid[I[r], J[r], (zc > a) & (zc < b)] = True
    return np.argwhere(grid), lo


def main(argv=None):
    import trimesh
    from loadpath.fem import HexMesh
    from loadpath.nonlinear import Tendon, nodes_near, solve_nonlinear

    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", default=os.path.join(HERE, "hand", "finger.stl"))
    ap.add_argument("--pitch", type=float, default=0.73)
    ap.add_argument("--tension", type=float, default=17.0, help="N, at full load")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="finger-curl.json")
    args = ap.parse_args(argv)

    stl = trimesh.load(args.stl)
    ijk, lo = voxels(stl, args.pitch)
    mesh = HexMesh(ijk, args.pitch, lo)
    xyz = mesh.node_xyz
    L = stl.bounds[1][2]
    fixed = np.zeros(mesh.n_dof, bool)
    fixed[(3 * np.where(xyz[:, 2] <= 6.0)[0][:, None] + np.arange(3)).ravel()] = True   # root in its cup
    r = np.hypot(xyz[:, 0] - CHANNEL[0], xyz[:, 1] - CHANNEL[1])
    knot = np.where((np.abs(xyz[:, 2] - (L - KNOT_FROM_TIP)) < 0.6 * args.pitch) & (r > CHANNEL_R - 0.1)
                    & (r < 2.6 + args.pitch))[0]
    path = [np.array([CHANNEL[0], CHANNEL[1], -15.0])]     # palm exit
    for z in np.arange(1.0, L - KNOT_FROM_TIP - 0.5, 2 * CHANNEL_R):
        if min(abs(z - h) for h in HINGES) < 3.5:
            continue
        path.append(nodes_near(mesh, [CHANNEL[0], CHANNEL[1], z], CHANNEL_R + args.pitch))
    path.append(knot)
    tip = xyz[:, 2] >= L - 1.5 * args.pitch
    distal = xyz[:, 2] >= L - 20.0
    probes = {}
    for k, axis in (("tipY", 1), ("tipZ", 2)):
        w = np.zeros(mesh.n_dof)
        w[3 * np.where(tip)[0] + axis] = 1.0 / tip.sum()
        probes[k] = w

    def curl(u):
        """Rotation of the distal phalanx in the yz plane, deg (+ toward the palm)."""
        p = xyz[distal][:, 1:]
        q = p + u.reshape(-1, 3)[distal][:, 1:]
        U, _, Vt = np.linalg.svd((p - p.mean(0)).T @ (q - q.mean(0)))
        R = (U @ Vt).T
        return -float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    print(f"{mesh.n_elem} cells at {args.pitch} mm, {2.2 / args.pitch:.1f} across the flexure", flush=True)
    hist = []

    def step(h, u=None):
        rec = {"tensionN": round(h["loadFactor"] * args.tension, 2), "curlDeg": round(curl(u), 1),
               "tipYmm": round(h["tipY"], 2), "tipZmm": round(h["tipZ"], 2),
               "cableTravelMm": round(h["tendon0Travel"], 2), "newton": h["newton"]}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump({"pitch": args.pitch, "history": hist}, open(args.out, "w"), indent=1)

    t0 = time.time()
    res = solve_nonlinear(mesh, TPU_NU, TPU_E, fixed, None, tendons=[Tendon(args.tension, path)],
                          steps=args.steps, probes=probes, on_step=step)
    out = {"pitch": args.pitch, "elements": int(mesh.n_elem), "converged": res.converged,
           "peakStrain": float(np.max(res.strain)) if res.strain is not None else None,
           "seconds": round(time.time() - t0), "history": hist}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "history"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
