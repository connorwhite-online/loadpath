"""Where are the self-intersections? (debug helper)"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "moonshots/round2")
from crease_bench import self_intersections  # noqa: E402

name, p = sys.argv[1], sys.argv[2]
d = np.load(f"moonshots/round2/mesh_{name}_{p}.npz")
for tag, V, F, cr in (("before", d["V0"], d["F0"], None), ("sharpen", d["V1"], d["F1"], d["cr"])):
    out = []
    n = self_intersections(V, F, np.arange(len(F)), pairs_out=out)
    P = out[0]
    cen = V[F[P[:, 0]]].mean(1)
    print(tag, n)
    if cr is not None and len(P):
        inv = cr[F[P]].any(axis=2)
        print("  pairs touching crease verts (A,B):", inv[:, 0].sum(), inv[:, 1].sum())
    # cluster centres
    if len(P):
        key = np.round(cen / 1.0).astype(int)
        u, c = np.unique(key, axis=0, return_counts=True)
        print("  clusters (1mm):", len(u), " biggest:", [(tuple(x), y) for x, y in sorted(zip(u.tolist(), c), key=lambda t: -t[1])[:6]])
