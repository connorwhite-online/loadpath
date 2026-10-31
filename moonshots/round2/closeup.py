"""Close-up renders (matplotlib) of a crease, before / after.

python moonshots/round2/closeup.py CASE PITCH [cx cy cz radius elev azim]
reads mesh_CASE_PITCH.npz (written by crease_bench.py) and writes
closeup_CASE_PITCH.png: the faces within `radius` of the centre, flat-shaded,
crease vertices in red, the true crease as a thin black line.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).resolve().parent


def panel(ax, V, F, c, r, title, X=None, cr=None, elev=25, azim=40):
    cen = V[F].mean(1)
    sel = np.linalg.norm(cen - c, axis=1) < r
    T = V[F[sel]]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    col = np.stack([0.80 * shade, 0.84 * shade, 0.90 * shade, np.ones_like(shade)], 1)
    pc = Poly3DCollection(T, facecolors=col, edgecolors=(0.15, 0.15, 0.2, 0.55), linewidths=0.25)
    ax.add_collection3d(pc)
    if cr is not None:
        P = V[cr]
        P = P[np.linalg.norm(P - c, axis=1) < r]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=4, c="red", depthshade=False)
    if X is not None:
        Q = X[np.linalg.norm(X - c, axis=1) < r]
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=0.6, c="k", depthshade=False)
    for i, set_ in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        set_(c[i] - r * 0.75, c[i] + r * 0.75)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main():
    name, p = sys.argv[1], sys.argv[2]
    d = np.load(HERE / f"mesh_{name}_{p}.npz")
    X = d["X"]
    if len(sys.argv) > 3:
        c = np.array([float(v) for v in sys.argv[3:6]])
        r = float(sys.argv[6])
        elev, azim = float(sys.argv[7]), float(sys.argv[8])
    else:
        c = X[len(X) // 3]
        r = 8 * float(p)
        elev, azim = 25, 40
    keys = [("V0", "F0", None, "before: marching cubes + relax"), ("V1", "F1", "cr", "after: sharpen"),
            ("V2", "F2", "cr2", "after: crease-aware finish")]
    keys = [k for k in keys if k[0] in d]
    if "V2" in d and d["V2"].shape == d["V1"].shape and (d["V2"] == d["V1"]).all():
        keys = keys[:2]             # the bench ran without --aware
    fig = plt.figure(figsize=(4.2 * len(keys), 4.4), dpi=150)
    for i, (kv, kf, kc, title) in enumerate(keys):
        ax = fig.add_subplot(1, len(keys), i + 1, projection="3d")
        panel(ax, d[kv], d[kf], c, r, title, X=X, cr=d[kc] if kc else None, elev=elev, azim=azim)
    fig.suptitle(f"{name}, pitch {p} mm (black: true crease)", fontsize=10)
    fig.tight_layout()
    out = HERE / f"closeup_{name}_{p}.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
