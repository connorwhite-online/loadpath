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
    mesh = _outside(mesh)
    if len(mesh.faces) <= faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=faces)
    except Exception:  # noqa: BLE001 - render proxy only
        return mesh


def _outside(mesh):
    """The mesh without its enclosed cavities' surfaces (a sealed plenum's
    inside): matplotlib sorts faces by depth only roughly, and a cavity's
    few large faces were drawn over the outside."""
    try:
        parts = mesh.split(only_watertight=False)
        keep = [p for p in parts if p.volume > 0]
        if 0 < len(keep) < len(parts):
            import trimesh
            return trimesh.util.concatenate(keep)
    except Exception:  # noqa: BLE001 - render proxy only
        pass
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
                    if not np.any(ld.force):
                        continue
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
                  exaggerate=None, safety_factor=1.0, label=None):
    """Stress map of a verified part: each vertex coloured by the worst
    nodal von Mises stress over the load cases, as a fraction of `strength`
    (verify_part(..., fields=True)["fields"]), divided by `safety_factor`:
    pass the study's goal and red means "at the allowable". `exaggerate` scales the worst
    case's displacement into the drawn shape; by default the largest
    displacement is drawn as 4% of the part's size, so the bend reads.
    `label` replaces the colour bar's label (say when `strength` is not one)."""
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
    cb.set_label(label if label else "von Mises / allowable (worst case)" if safety_factor != 1.0 else "von Mises / strength (worst case)")
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

    def on_part(c, sh):
        # A ring's or a sleeve's box centre is in its empty bore: centre the
        # panel on the part's surface where it meets the region nearest that
        # centre instead, so the panel shows the ring and not the hole.
        near = np.abs(sh.sdf(tc)) < 1.5
        if not near.any():
            return c
        pts = tc[near]
        return pts[np.argmin(np.linalg.norm(pts - c, axis=1))]

    spots = []   # (label, centre, half-size)
    for r in study.preserve:
        shapes = getattr(r.shape, "shapes", [r.shape])   # a union: each piece is its own interface
        for j, sh in enumerate(shapes):
            lo, hi = sh.bounds()
            c = 0.5 * (lo + hi)
            half = min(0.5 * float(np.max(hi - lo)), 20.0) + margin
            if 0.5 * float(np.max(hi - lo)) > half - margin:   # bigger than a panel: find the ring
                c = on_part(c, sh)
            spots.append((r.id + (f"[{j}]" if len(shapes) > 1 else ""), c, half))
    for r in study.keep_out:
        # where the part comes near a keep-out: centre on that contact, not on
        # the keep-out (a driver channel runs 25 mm out past the part)
        near = np.abs(r.shape.sdf(tc)) < 1.5
        if near.any():
            pts = tc[near]
            ext = float(np.max(np.ptp(pts, axis=0)))
            c = pts.mean(axis=0)
            # contact all round a bore averages to its axis: use the contact
            # point nearest that mean
            spots.append((r.id, pts[np.argmin(np.linalg.norm(pts - c, axis=1))], min(0.5 * ext, 20.0) + margin))
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


def render_heat_exchanger(mesh, spaces, core, path, title=None, size=(5.0, 4.6), cut=None, view=(28, -55)):
    """Four panels: the part, the part cut open, and each fluid's space cut
    open at the same plane so the two interleaved networks show (A blue,
    B orange), then fluid A alone. `cut` is (point, normal), by default
    through the core's centre across z, keeping the side the normal points
    to (point it away from the view); `core` is any shape (its bounds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lo, hi = np.asarray(mesh.bounds)
    span = (hi - lo).max()
    mid = 0.5 * (lo + hi)
    clo, chi = (np.asarray(b, float) for b in core.bounds())
    if cut is None:
        cut = (0.5 * (clo + chi), np.array([0.0, 0.0, 1.0]))
    c0, cn = np.asarray(cut[0], float), np.asarray(cut[1], float)
    cn = cn / np.linalg.norm(cn)
    grey = np.array([0.80, 0.80, 0.82])
    colors = [np.array([0.20, 0.45, 0.85]), np.array([0.92, 0.50, 0.15])]
    names = list(spaces)
    panels = [("part", [(mesh, grey)]),
              ("part, cut open", [(mesh.slice_plane(c0, cn, cap=True), grey)])]
    # keep the half beyond the cut, so the view looks into it
    cutf = [(spaces[n].slice_plane(c0, cn, cap=True), colors[i % 2]) for i, n in enumerate(names)]
    panels.append((" + ".join(names) + ", cut open", cutf))
    panels.append((names[0], cutf[:1]))
    fig = plt.figure(figsize=(size[0] * len(panels), size[1]), dpi=110)
    elev, azim = view
    e, a = np.radians(elev), np.radians(azim)
    view_dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    key = view_dir + np.array([0.3, -0.2, 0.6])
    key /= np.linalg.norm(key)
    for pi, (label, items) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), pi + 1, projection="3d")
        for m, base in items:
            m = _decimate(m, 90_000)
            shade = np.clip(m.face_normals @ key, 0, 1)
            col = np.clip(base[None] * (0.35 + 0.65 * shade)[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(m.triangles, facecolors=col, edgecolor="none", linewidth=0))
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_box_aspect((1, 1, 1), zoom=1.6)
        ax.view_init(elev, azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path

def render_internal_ceilings(variants, up, path, title=None, size=(5.0, 4.6), view=(30, -55), flat=20.0):
    """One panel per variant (label, {fluid: space mesh}): the fluid spaces
    seen from above, the top of each (where it meets metal above it, a
    ceiling) red where it lies within `flat` degrees of level along `up`.
    Inside a metal part nothing there can be supported."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    up = np.asarray(up, float) / np.linalg.norm(up)
    meshes = [m for _, sp in variants for m in sp.values()]
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    span, mid = (hi - lo).max(), 0.5 * (lo + hi)
    colors = [np.array([0.20, 0.45, 0.85]), np.array([0.92, 0.65, 0.25])]
    red = np.array([0.85, 0.08, 0.08])
    fig = plt.figure(figsize=(size[0] * len(variants), size[1]), dpi=110)
    elev, azim = view
    e, a = np.radians(elev), np.radians(azim)
    view_dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    key = view_dir + np.array([0.3, -0.2, 0.6])
    key /= np.linalg.norm(key)
    for pi, (label, spaces) in enumerate(variants):
        ax = fig.add_subplot(1, len(variants), pi + 1, projection="3d")
        for i, m in enumerate(spaces.values()):
            m = _decimate(m, 120_000)
            n = m.face_normals
            shade = np.clip(n @ key, 0, 1)
            base = np.where((n @ up > np.cos(np.radians(flat)))[:, None], red[None], colors[i % 2][None])
            col = np.clip(base * (0.35 + 0.65 * shade)[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(m.triangles, facecolors=col, edgecolor="none", linewidth=0))
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
        ax.view_init(elev, azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def render_lattice(mesh, path, title=None, holes=(), size=(5.0, 4.6), view=(28, -55)):
    """A lattice part: whole, cut open across its length and its height so
    the core and its grading show, and a close-up of the first drain hole
    (`holes` as lattice_infill's printability["drainHoles"])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lo, hi = np.asarray(mesh.bounds)
    mid = 0.5 * (lo + hi)
    ext = hi - lo
    long_ax = int(np.argmax(ext))
    tall = [a for a in np.argsort(ext) if a != long_ax][0]
    side = 3 - long_ax - tall
    n_side, n_tall = np.zeros(3), np.zeros(3)
    n_side[side] = 1.0
    n_tall[tall] = -1.0
    panels = [("part", mesh, None),
              ("cut along its length", mesh.slice_plane(mid, n_side, cap=True), None),
              ("cut through its height", mesh.slice_plane(mid, n_tall, cap=True), None)]
    if holes:
        h = holes[0]
        c, nrm = np.asarray(h["at"]), np.asarray(h["normal"])
        r = 4.0 * h["diameter"]
        panels.append((f"drain hole {h['diameter']:g} mm, cut in half", mesh.slice_plane(c, _perp(nrm), cap=True),
                       (c, r, nrm)))
    fig = plt.figure(figsize=(size[0] * len(panels), size[1]), dpi=110)
    grey = np.array([0.80, 0.80, 0.82])
    for pi, (label, m, zoom) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), pi + 1, projection="3d")
        if zoom is not None:
            c, r, nrm = zoom
            keep = np.all(np.abs(m.triangles_center - c) < r, axis=1)
            m = m.submesh([np.nonzero(keep)[0]], append=True) if keep.any() else m
            # look at the cut face (it faces away from the kept side), a
            # little from outside the part
            v = -_perp(nrm) + 0.5 * nrm
            v /= np.linalg.norm(v)
            elev, azim = np.degrees(np.arcsin(np.clip(v[2], -1, 1))), np.degrees(np.arctan2(v[1], v[0]))
            cen, span = c - nrm * 0.25 * r, 1.6 * r
        else:
            elev, azim = view
            cen, span = mid, ext.max()
        m = _decimate(m, 400_000)
        # matplotlib paints back to front by face centre: the skin's long
        # flat triangles would sort behind the lattice they cover
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=span / 60.0, max_iter=12)
        m = trimesh.Trimesh(v, f, process=False)
        e, a = np.radians(elev), np.radians(azim)
        view_dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        key = view_dir + np.array([0.3, -0.2, 0.6])
        key /= np.linalg.norm(key)
        shade = np.clip(m.face_normals @ key, 0, 1)
        col = np.clip(grey[None] * (0.30 + 0.70 * shade)[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(m.triangles, facecolors=col, edgecolor="none", linewidth=0))
        for k in range(3):
            getattr(ax, "set_" + "xyz"[k] + "lim")(cen[k] - span / 2, cen[k] + span / 2)
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
        ax.view_init(elev, azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _perp(n):
    """A unit vector across `n`: the plane that cuts a drain hole in half."""
    t = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    return t / np.linalg.norm(t)
