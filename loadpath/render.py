"""
Headless clay renders of a result, for reports and for judging outcomes by
eye (or by a vision model). matplotlib Agg only: no GL context needed, so it
works headless and in CI. Keep-outs draw as translucent red, load
regions as arrows.
"""
from __future__ import annotations

import numpy as np

__all__ = ["render_views", "render_stress", "render_interfaces"]

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


def render_interfaces(mesh, study, path, max_panels=16, cols=4, size=3.2, margin=6.0):
    """Close-ups of every place the part meets the outside world: each
    preserved region and each keep-out the part touches, cropped to a box
    round it and shot from two sides. Whole-part renders show the layout and
    hide the finish: GRIP's wrist yoke passed every number with sawn-off
    faces, hard creases at its bosses and material over its bolt seats, all
    of them a few pixels in the overview (QA, 10/28). Look at these before
    calling a part done."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tc = mesh.triangles_center
    spots = []   # (label, centre, half-size)
    for r in study.preserve:
        shapes = getattr(r.shape, "shapes", [r.shape])   # a union: each piece is its own interface
        for j, sh in enumerate(shapes):
            lo, hi = sh.bounds()
            spots.append((r.id + (f"[{j}]" if len(shapes) > 1 else ""), 0.5 * (lo + hi),
                          min(0.5 * float(np.max(hi - lo)), 20.0) + margin))
    for r in study.keep_out:
        # where the part comes near a keep-out: centre on that contact, not on
        # the keep-out (a driver channel runs 25 mm out past the part)
        near = np.abs(r.shape.sdf(tc)) < 1.5
        if near.any():
            pts = tc[near]
            ext = float(np.max(np.ptp(pts, axis=0)))
            spots.append((r.id, pts.mean(axis=0), min(0.5 * ext, 20.0) + margin))
    # one panel per place: spots closer than a few mm show the same thing
    merged = []
    for lab, c, h in spots:
        for m in merged:
            if np.linalg.norm(m[1] - c) < 4.0:
                m[0].append(lab)
                m[2] = max(m[2], h)
                break
        else:
            merged.append([[lab], c, h])
    merged = merged[:max_panels]
    if not merged:
        return None
    rows = int(np.ceil(len(merged) / cols))
    fig = plt.figure(figsize=(size * cols, size * rows), dpi=110)
    for k, (labs, c, half) in enumerate(merged):
        # crop exactly (boolean faces are long slivers: selecting whole
        # triangles by centre drew planes across the panel)
        crop = mesh
        for ax_i in range(3):
            for sgn in (1, -1):
                n_ = np.zeros(3)
                n_[ax_i] = sgn
                crop = crop.slice_plane(c - half * n_, n_)   # keeps the side n_ points to
                if len(crop.faces) == 0:
                    break
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        if len(crop.faces):
            tris = crop.triangles
            nrm = crop.face_normals
            # look from the side the nearby surface faces, tilted so relief
            # (a crease, a step, a crater) shows instead of flattening out
            vd = (nrm * crop.area_faces[:, None]).sum(axis=0)
            vd = vd / np.linalg.norm(vd) if np.linalg.norm(vd) > 1e-6 else np.array([0.5, -0.5, 0.7])
            elev = float(np.degrees(np.arcsin(np.clip(vd[2], -1, 1)))) + 25.0
            azim = float(np.degrees(np.arctan2(vd[1], vd[0]))) + 30.0
            e, a = np.radians(elev), np.radians(azim)
            view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            key = view + np.array([0.3, -0.2, 0.6])
            key /= np.linalg.norm(key)
            shade = np.clip(nrm @ key, 0, 1)
            col = np.clip(np.array([0.80, 0.80, 0.82])[None] * (0.25 + 0.7 * shade)[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolor="none"))
            ax.view_init(elev, azim)
        for i, axis in enumerate("xyz"):
            getattr(ax, f"set_{axis}lim")(c[i] - half, c[i] + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        ax.set_axis_off()
        title = ", ".join(labs[:3]) + (f" +{len(labs) - 3}" if len(labs) > 3 else "")
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def render_heat_exchanger(mesh, spaces, core, path, title=None, size=(5.0, 4.6)):
    """Three panels: the part, and each fluid's space cut open at the core's
    mid-height so the two interleaved networks show (A blue, B orange)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lo, hi = np.asarray(mesh.bounds)
    span = (hi - lo).max()
    mid = 0.5 * (lo + hi)
    colors = [np.array([0.20, 0.45, 0.85]), np.array([0.92, 0.50, 0.15])]
    names = list(spaces)
    panels = [("part", [(mesh, np.array([0.80, 0.80, 0.82]))])]
    zcut = 0.5 * (core.lo[2] + core.hi[2])
    cut = []
    for i, n in enumerate(names):
        m = spaces[n].slice_plane([0, 0, zcut], [0, 0, -1], cap=True)
        cut.append((m, colors[i % 2]))
    panels.append((" + ".join(names) + f", cut at z={zcut:g}", cut))
    panels.append((names[0], cut[:1]))
    fig = plt.figure(figsize=(size[0] * len(panels), size[1]), dpi=110)
    elev, azim = 28, -55
    e, a = np.radians(elev), np.radians(azim)
    view_dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    key = view_dir + np.array([0.3, -0.2, 0.6])
    key /= np.linalg.norm(key)
    for pi, (label, items) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), pi + 1, projection="3d")
        for m, base in items:
            m = _decimate(m, 60_000)
            shade = np.clip(m.face_normals @ key, 0, 1)
            col = np.clip(base[None] * (0.35 + 0.65 * shade)[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(m.triangles, facecolors=col, edgecolor="none", linewidth=0))
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_box_aspect((1, 1, 1), zoom=1.4)
        ax.view_init(elev, azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
