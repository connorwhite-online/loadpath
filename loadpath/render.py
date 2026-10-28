"""
Headless clay renders of a result, for reports and for judging outcomes by
eye (or by a vision model). matplotlib Agg only: no GL context needed, so it
works on the sidecar and in CI. Keep-outs draw as translucent red, load
regions as arrows.
"""
from __future__ import annotations

import numpy as np

__all__ = ["render_views", "render_stress"]

_VIEWS = [(24, -58), (24, 32), (78, -90)]


def _decimate(mesh, faces=40_000):
    if len(mesh.faces) <= faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=faces)
    except Exception:  # noqa: BLE001 - render proxy only
        return mesh


def render_views(mesh, path, study=None, title=None, views=_VIEWS, size=(5.0, 4.2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = _decimate(mesh)
    tris = m.triangles
    lo, hi = np.asarray(mesh.bounds)
    if study is not None:
        slo, shi = study.bounds()
        lo, hi = np.minimum(lo, slo), np.maximum(hi, shi)
    span = (hi - lo).max()
    mid = 0.5 * (lo + hi)
    fig = plt.figure(figsize=(size[0] * len(views), size[1]), dpi=110)
    for vi, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), vi + 1, projection="3d")
        e, a = np.radians(elev), np.radians(azim)
        view_dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        key = view_dir + np.array([0.3, -0.2, 0.6])
        key /= np.linalg.norm(key)
        shade = np.clip(m.face_normals @ key, 0, 1)
        rim = np.clip(m.face_normals @ view_dir, 0, 1)
        base = np.array([0.80, 0.80, 0.82])
        col = np.clip(base[None] * (0.28 + 0.62 * shade + 0.18 * rim)[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolor="none", linewidth=0))
        if study is not None:
            for r in study.keep_out:
                try:
                    ko = r.shape.to_manifold().to_mesh()
                    v = np.asarray(ko.vert_properties)[:, :3]
                    f = np.asarray(ko.tri_verts)
                    ax.add_collection3d(Poly3DCollection(v[f], facecolors=(0.9, 0.25, 0.2, 0.12),
                                                         edgecolor="none"))
                except Exception:  # noqa: BLE001
                    pass
            for case in study.load_cases:
                for ld in case.loads:
                    shape = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
                    blo, bhi = shape.bounds()
                    c = 0.5 * (blo + bhi)
                    d = ld.force / np.linalg.norm(ld.force) * span * 0.18
                    ax.quiver(*(c - d), *d, color=(0.15, 0.45, 0.95), linewidth=2, arrow_length_ratio=0.3)
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.view_init(elev, azim)
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def render_stress(mesh, fields, path, strength, title=None, views=_VIEWS[:2], size=(5.0, 4.2),
                  exaggerate=None, safety_factor=1.0):
    """Stress map of a verified part: each vertex coloured by the worst
    nodal von Mises stress over the load cases, as a fraction of `strength`
    (verify_part(..., fields=True)["fields"]), divided by `safety_factor`:
    pass the study's goal and red means "at the allowable". `exaggerate` scales the worst
    case's displacement into the drawn shape; by default the largest
    displacement is drawn as 4% of the part's size, so the bend reads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import cKDTree

    m = _decimate(mesh)
    nodes = fields["nodes"]
    _, idx = cKDTree(nodes).query(m.vertices, k=1)
    vm = np.max([c["vonMises"][idx] for c in fields["cases"]], axis=0)
    worst = max(fields["cases"], key=lambda c: np.linalg.norm(c["displacement"], axis=1).max())
    disp = worst["displacement"][idx]
    lo, hi = np.asarray(mesh.bounds)
    span = (hi - lo).max()
    dmax = np.linalg.norm(disp, axis=1).max()
    k = exaggerate if exaggerate is not None else (0.04 * span / dmax if dmax > 0 else 0.0)
    verts = m.vertices + k * disp
    frac = np.clip(vm * safety_factor / strength, 0, 1)
    face = frac[m.faces].max(axis=1)
    norm = colors.Normalize(0, 1)
    cmap = matplotlib.colormaps["turbo"]
    mid = 0.5 * (lo + hi)
    fig = plt.figure(figsize=(size[0] * len(views) + 0.8, size[1]), dpi=110)
    for vi, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), vi + 1, projection="3d")
        e, a = np.radians(elev), np.radians(azim)
        view_dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        key = view_dir + np.array([0.3, -0.2, 0.6])
        key /= np.linalg.norm(key)
        shade = 0.45 + 0.55 * np.clip(m.face_normals @ key, 0, 1)
        col = cmap(norm(face))[:, :3] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(verts[m.faces], facecolors=np.clip(col, 0, 1),
                                             edgecolor="none", linewidth=0))
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.view_init(elev, azim)
        ax.set_axis_off()
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=0.7, pad=0.02)
    cb.set_label("von Mises / allowable (worst case)" if safety_factor != 1.0 else "von Mises / strength (worst case)")
    if title:
        fig.suptitle(title + (f"  ·  shape {k:.0f}x exaggerated ({worst['name']})" if k else ""), fontsize=10)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
