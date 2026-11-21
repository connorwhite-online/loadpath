"""
Is the skin fair? Curvature, continuity and zebra, measured before a person
has to see it.

A surface reads as one smooth form when its curvature changes slowly and
its edges carry curvature across. A designer checks that by eye, with
zebra stripes (reflections of a row of strip lights: they kink at a
tangent break, bend sharply at a curvature break and wobble over a lump)
and curvature maps. Here the same checks are numbers, so a script can
catch a phantom ridge or a lump before anyone looks:

- `surface_curvature(s)`: principal and mean curvature over a B-spline
  surface (surfacing.Surface or any OpenCascade face);
- `edge_continuity(shape)`: across every edge two faces share, the gap
  (G0, mm), the angle between their tangent planes (G1, degrees) and the
  jump in curvature across the edge (G2, 1/mm and as a share);
- `mesh_curvature(mesh)`: the same on a triangle mesh (a field model's
  marching-cubes skin, a scan), from smoothed vertex normals;
- `fairness(mesh)`: the flaws, found and placed: *tight* spots (a radius
  under the smallest the design means to have) and *wobbles* (curvature
  that departs from its own smoothed self by more than a tolerance: the
  lumps, dips and phantom ridges a blend leaves). Each is a cluster with
  its place, size and worst value; features meant to be sharp (a groove,
  a grille) are left out with `ignore`;
- `zebra_image` / `curvature_image` and `quality_sheet`: the pictures a
  designer would ask for, from any camera (raster.Camera).
"""
from __future__ import annotations

import numpy as np


# ---- B-rep ----------------------------------------------------------------

def _as_face(s):
    from .surfacing import Surface
    if isinstance(s, Surface):
        return s.face()
    return s


def surface_curvature(s, n=(60, 60)):
    """{"points", "normal", "k1", "k2" (principal, 1/mm, k1 >= k2), "mean",
    "gauss", "minRadius", "maxAbsCurvature"} on an n grid over a face's
    parameter range."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    f = _as_face(s)
    ad = BRepAdaptor_Surface(f)
    u = np.linspace(ad.FirstUParameter(), ad.LastUParameter(), n[0])
    v = np.linspace(ad.FirstVParameter(), ad.LastVParameter(), n[1])
    P = np.zeros(n + (3,))
    N = np.zeros(n + (3,))
    K1 = np.zeros(n)
    K2 = np.zeros(n)
    pr = BRepLProp_SLProps(ad, 2, 1e-7)
    for i, a in enumerate(u):
        for j, b in enumerate(v):
            pr.SetParameters(float(a), float(b))
            p = pr.Value()
            P[i, j] = (p.X(), p.Y(), p.Z())
            if pr.IsCurvatureDefined():
                K1[i, j], K2[i, j] = pr.MaxCurvature(), pr.MinCurvature()
                nn = pr.Normal()
                N[i, j] = (nn.X(), nn.Y(), nn.Z())
            else:
                K1[i, j] = K2[i, j] = np.nan
    a = np.nanmax(np.maximum(np.abs(K1), np.abs(K2)))
    return {"points": P, "normal": N, "k1": K1, "k2": K2, "mean": 0.5 * (K1 + K2), "gauss": K1 * K2,
            "maxAbsCurvature": float(a), "minRadius": float(1.0 / a) if a > 0 else float("inf")}


def _faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    ex = TopExp_Explorer(_as_face(shape), TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face(ex.Current()))
        ex.Next()
    return out


def _normal_curvature(pr, d):
    """Normal curvature of the surface in tangent direction d (unit)."""
    from OCP.gp import gp_Dir
    k1, k2 = pr.MaxCurvature(), pr.MinCurvature()
    dmax, dmin = gp_Dir(), gp_Dir()
    pr.CurvatureDirections(dmax, dmin)
    c = d @ np.array([dmax.X(), dmax.Y(), dmax.Z()])
    return k1 * c * c + k2 * (1.0 - c * c)


def edge_continuity(shape, samples=24, skip_seams=True):
    """Every edge shared by two faces of `shape`: [{"edge" index, "G0" (mm,
    the largest gap), "G1" (degrees, the largest angle between the faces'
    tangent planes), "G2" (1/mm, the largest jump in normal curvature
    across the edge, square to it), "G2share" (that jump over the larger
    curvature, 0 = curvature continuous), "point" (where the worst G1 is)}]
    plus a summary {"edges", "worstG0", "worstG1", "worstG2", "worstG2share"}.
    (Read G2share beside G2: across an edge where both faces are nearly
    flat square to it, a jump no eye sees is still a large share.)"""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    from .surfacing import _edge_face_map
    m = _edge_face_map()
    TopExp.MapShapesAndAncestors_s(_as_face(shape), TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        fl = list(m.FindFromIndex(i))
        e = TopoDS.Edge(m.FindKey(i))
        uniq = []
        for f in fl:
            f = TopoDS.Face(f)
            if not any(f.IsSame(g) for g in uniq):
                uniq.append(f)
        if len(uniq) != 2:
            continue
        fa, fb = uniq
        if skip_seams and BRep_Tool.IsClosed_s(e, fa):
            continue
        ce = BRepAdaptor_Curve(e)
        t0, t1 = ce.FirstParameter(), ce.LastParameter()
        rows = []
        for t in np.linspace(t0, t1, samples + 2)[1:-1]:
            from OCP.gp import gp_Pnt, gp_Vec
            p, tv = gp_Pnt(), gp_Vec()
            ce.D1(float(t), p, tv)
            T = np.array([tv.X(), tv.Y(), tv.Z()])
            T = T / max(np.linalg.norm(T), 1e-30)
            vals = []
            for f in (fa, fb):
                c2 = BRep_Tool.CurveOnSurface_s(e, f, 0.0, 0.0)
                uv = c2.Value(float(t))
                sa = BRepAdaptor_Surface(f)
                pr = BRepLProp_SLProps(sa, float(uv.X()), float(uv.Y()), 2, 1e-7)
                q = pr.Value()
                nn = pr.Normal()
                Nv = np.array([nn.X(), nn.Y(), nn.Z()])
                if f.Orientation() == TopAbs_REVERSED:
                    Nv = -Nv
                d = np.cross(Nv, T)
                d /= max(np.linalg.norm(d), 1e-30)
                k = _normal_curvature(pr, d) if pr.IsCurvatureDefined() else np.nan
                if f.Orientation() == TopAbs_REVERSED:
                    k = -k
                vals.append((np.array([q.X(), q.Y(), q.Z()]), Nv, k))
            (pa, na, ka), (pb, nb, kb) = vals
            g0 = float(np.linalg.norm(pa - pb))
            g1 = float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1, 1))))
            g2 = float(abs(ka - kb))
            share = g2 / max(abs(ka), abs(kb), 1e-9)
            rows.append((g0, g1, g2, share, pa))
        r = np.array([x[:4] for x in rows])
        w = int(np.argmax(r[:, 1]))
        out.append({"edge": i, "G0": float(r[:, 0].max()), "G1": float(r[:, 1].max()), "G2": float(r[:, 2].max()),
                    "G2share": float(r[:, 3].max()), "point": np.round(rows[w][4], 2).tolist()})
    summ = {"edges": len(out),
            "worstG0": max((x["G0"] for x in out), default=0.0),
            "worstG1": max((x["G1"] for x in out), default=0.0),
            "worstG2": max((x["G2"] for x in out), default=0.0),
            "worstG2share": max((x["G2share"] for x in out), default=0.0)}
    return out, summ


# ---- meshes ---------------------------------------------------------------

def _adjacency(mesh):
    import scipy.sparse as sp
    E = np.asarray(mesh.edges_unique)
    n = len(mesh.vertices)
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    return A, E


def _smooth(A, X, iterations):
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iterations):
        X = 0.5 * X + 0.5 * (A @ X) / deg[:, None] if X.ndim == 2 else 0.5 * X + 0.5 * (A @ X) / deg
    return X


def mesh_curvature(mesh, smooth=2, radius=None, k=20):
    """Per-vertex curvature of a triangle mesh: {"mean" (signed, 1/mm,
    positive convex), "k1", "k2" (principal), "max" (the larger |k|),
    "normal" (smoothed)}. A quadric z = a x^2 + b xy + c y^2 + d x + e y
    is fitted at each vertex to its `k` nearest vertices (within `radius`
    mm, default 3 mean edge lengths) in the frame of its smoothed normal,
    so marching-cubes noise and long thin CAD triangles do not read as
    curvature."""
    from scipy.spatial import cKDTree
    A, E = _adjacency(mesh)
    V = np.asarray(mesh.vertices, float)
    N = _smooth(A, np.asarray(mesh.vertex_normals, float), smooth)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    h = float(np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1).mean())
    radius = radius or 3.0 * h
    tree = cKDTree(V)
    dist, idx = tree.query(V, k=min(k, len(V)), workers=-1)
    n = len(V)
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    # a tangent frame per vertex
    ref = np.where(np.abs(N[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    T1 = np.cross(N, ref)
    T1 /= np.linalg.norm(T1, axis=1, keepdims=True)
    T2 = np.cross(N, T1)
    for s0 in range(0, n, 200_000):
        sl = slice(s0, min(n, s0 + 200_000))
        D = V[idx[sl]] - V[sl][:, None, :]
        x = np.einsum("nkj,nj->nk", D, T1[sl])
        y = np.einsum("nkj,nj->nk", D, T2[sl])
        z = np.einsum("nkj,nj->nk", D, N[sl])
        w = np.exp(-(dist[sl] / radius) ** 2)
        M = np.stack([x * x, x * y, y * y, x, y], -1) * w[..., None]
        rhs = z * w
        MtM = np.einsum("nki,nkj->nij", M, M) + 1e-9 * np.eye(5)
        Mtb = np.einsum("nki,nk->ni", M, rhs)
        c = np.linalg.solve(MtM, Mtb[..., None])[..., 0]
        a, b, cc = c[:, 0], c[:, 1], c[:, 2]
        # shape operator of z = f(x, y) at the origin, slope included
        fx, fy = c[:, 3], c[:, 4]
        g = 1.0 + fx * fx + fy * fy
        E1, F1, G1 = 1 + fx * fx, fx * fy, 1 + fy * fy
        L, Mm, Nn = 2 * a / np.sqrt(g), b / np.sqrt(g), 2 * cc / np.sqrt(g)
        det = E1 * G1 - F1 * F1
        H = (E1 * Nn - 2 * F1 * Mm + G1 * L) / (2 * det)
        K = (L * Nn - Mm * Mm) / det
        r = np.sqrt(np.maximum(H * H - K, 0))
        # the normal points out: a convex surface bends away from it (z < 0)
        k1[sl], k2[sl] = -(H - r), -(H + r)
    mean = 0.5 * (k1 + k2)
    return {"mean": mean, "k1": np.maximum(k1, k2), "k2": np.minimum(k1, k2),
            "max": np.maximum(np.abs(k1), np.abs(k2)), "normal": N}


def _geodesic_smooth(mesh, X, radius):
    """X smoothed over about `radius` mm on the surface (repeated
    neighbour averaging: diffusion to that radius)."""
    A, E = _adjacency(mesh)
    V = np.asarray(mesh.vertices, float)
    h = float(np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1).mean())
    rounds = int(min(max((radius / max(h, 1e-6)) ** 2, 1), 600))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    for _ in range(rounds):
        X = 0.5 * X + 0.5 * (A @ X) / deg
    return X


def _clusters(mesh, flag, values, kind, min_area):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    F = np.asarray(mesh.faces)
    E = np.asarray(mesh.edges_unique)
    keep = flag[E[:, 0]] & flag[E[:, 1]]
    n = len(mesh.vertices)
    G = sp.coo_matrix((np.ones(keep.sum()), (E[keep, 0], E[keep, 1])), shape=(n, n))
    nc, lab = connected_components(G, directed=False)
    va = np.zeros(n)
    fa = np.asarray(mesh.area_faces)
    for k in range(3):
        np.add.at(va, F[:, k], fa / 3.0)
    V = np.asarray(mesh.vertices)
    out = []
    for c in np.unique(lab[flag]):
        idx = np.nonzero((lab == c) & flag)[0]
        area = float(va[idx].sum())
        if area < min_area:
            continue
        w = idx[np.argmax(np.abs(values[idx]))]
        out.append({"kind": kind, "at": np.round(V[w], 1).tolist(), "centroid": np.round(V[idx].mean(0), 1).tolist(),
                    "area": round(area, 1), "worst": float(values[w]), "vertices": idx})
    out.sort(key=lambda r: -r["area"] * abs(r["worst"]))
    return out


def fairness(mesh, min_radius=3.0, wobble=0.02, scale=8.0, ignore=None, min_area=4.0, smooth=2):
    """The flaws of a skin, found and placed.

    - tight: the largest normal curvature at a vertex is over 1/min_radius
      (a radius under the smallest the design means to have);
    - wobble: the mean curvature departs from itself smoothed over `scale`
      mm by more than `wobble` 1/mm (a 50 mm radius bump on a flat face is
      0.02): a lump, a dip, or a phantom ridge where blends overlap. The
      broad form (the head's round, a grip) is the smoothed part and is
      not counted.

    `ignore`: a function (N, 3) points -> bool, True where sharp is meant
    (grooves, grilles, lens rims, the part line). Clusters under `min_area`
    mm^2 are dropped (mesh noise). Returns {"tight", "wobble": [clusters],
    "score": {"tightArea", "wobbleArea", "wobbleRMS" (1/mm, over the
    checked area), "checkedArea"}, "curvature": per-vertex arrays}."""
    import trimesh  # noqa: F401
    c = mesh_curvature(mesh, smooth)
    V = np.asarray(mesh.vertices)
    ok = np.ones(len(V), bool)
    if ignore is not None:
        ok &= ~np.asarray(ignore(V), bool)
    base = _geodesic_smooth(mesh, c["mean"], scale)
    dev = c["mean"] - base
    tight = ok & (c["max"] > 1.0 / min_radius)
    wob = ok & (np.abs(dev) > wobble)
    F = np.asarray(mesh.faces)
    va = np.zeros(len(V))
    fa = np.asarray(mesh.area_faces)
    for k in range(3):
        np.add.at(va, F[:, k], fa / 3.0)
    chk = float(va[ok].sum())
    t = _clusters(mesh, tight, c["max"], "tight", min_area)
    w = _clusters(mesh, wob, dev, "wobble", min_area)
    score = {"checkedArea": round(chk, 0),
             "tightArea": round(float(va[tight].sum()), 1),
             "wobbleArea": round(float(va[wob].sum()), 1),
             "wobbleRMS": float(np.sqrt((va[ok] * dev[ok] ** 2).sum() / max(chk, 1e-9))),
             "tightClusters": len(t), "wobbleClusters": len(w)}
    return {"tight": t, "wobble": w, "score": score,
            "curvature": {"mean": c["mean"], "max": c["max"], "deviation": dev, "checked": ok}}


def summary(report, top=8):
    """A fairness report without per-vertex arrays: for JSON."""
    def strip(cl):
        return [{k: v for k, v in x.items() if k != "vertices"} for x in cl[:top]]
    return {"score": report["score"], "tight": strip(report["tight"]), "wobble": strip(report["wobble"])}


# ---- pictures ---------------------------------------------------------------

def _pixel_interp(mesh, r, X):
    """Per-pixel value of a per-vertex quantity X, interpolated across each
    triangle (barycentric), for a raster.render result r of this mesh."""
    cam = r["camera"]
    owner = r["owner"]
    m = owner >= 0
    j, i = np.nonzero(m)
    t = owner[m]
    F = np.asarray(mesh.faces)[t]
    V = np.asarray(mesh.vertices, float)
    S = cam.project(V)[:, :2]
    a, b, c = S[F[:, 0]], S[F[:, 1]], S[F[:, 2]]
    px = np.column_stack([i + 0.5, j + 0.5])
    v0, v1, v2 = b - a, c - a, px - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    wb = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    wc = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    wa = 1.0 - wb - wc
    W = np.clip(np.column_stack([wa, wb, wc]), 0, 1)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    X = np.asarray(X, float)
    vals = (W[:, :, None] * X[F].reshape(len(F), 3, -1)).sum(1)
    out = np.full(m.shape + (vals.shape[1],), np.nan)
    out[m] = vals
    return out if X.ndim == 2 else out[..., 0]


def zebra_image(mesh, camera, stripes=14, axis=None, smooth=2, background=(255, 255, 255)):
    """Zebra stripes (H, W, 3 uint8): the reflection of a row of parallel
    strip lights, as a car designer's studio shows them. The lights run
    along `axis` (world, default the camera's right), so the stripes cross
    the form; they stay smooth over a G2 surface, bend sharply at a
    curvature break, kink at a tangent break, and wobble over a lump."""
    from .raster import render
    import scipy.sparse  # noqa: F401
    r = render(mesh, camera)
    A, _ = _adjacency(mesh)
    N = _smooth(A, np.asarray(mesh.vertex_normals, float), smooth)
    n = _pixel_interp(mesh, r, N)
    m = r["mask"]
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    d = camera.forward
    refl = d - 2.0 * (n @ d)[..., None] * n
    ax = np.asarray(axis if axis is not None else camera.right, float)
    ax /= np.linalg.norm(ax)
    # strip lights running along `ax`: the stripe index is the reflected
    # ray's angle round `ax`
    up = camera.up - (camera.up @ ax) * ax
    up /= np.linalg.norm(up)
    side = np.cross(ax, up)
    ang = np.arctan2(refl @ up, refl @ side)
    val = 0.5 + 0.5 * np.tanh(6.0 * np.sin(stripes * ang))
    img = np.empty(m.shape + (3,), float)
    img[:] = background
    shade = 40 + 200 * val
    img[m] = np.repeat(shade[m][:, None], 3, axis=1)
    return np.clip(img, 0, 255).astype(np.uint8)


def curvature_image(mesh, camera, values, limit=None, background=(255, 255, 255)):
    """Per-vertex `values` (a curvature, a deviation) as colour over the
    shaded form: blue negative, white zero, red positive, clipped at
    `limit` (default the 98th percentile of |values|)."""
    from .raster import render, shade_image
    r = render(mesh, camera)
    g = shade_image(r, background=background).astype(float) / 255.0
    x = _pixel_interp(mesh, r, np.asarray(values, float))
    lim = float(limit if limit is not None else np.nanpercentile(np.abs(values), 98) or 1.0)
    t = np.clip(np.nan_to_num(x) / lim, -1, 1)
    red = np.array([0.80, 0.15, 0.12])
    blue = np.array([0.12, 0.30, 0.80])
    white = np.array([0.97, 0.97, 0.97])
    col = np.where(t[..., None] >= 0, white + t[..., None] * (red - white), white - t[..., None] * (blue - white))
    lum = g.mean(-1, keepdims=True)
    img = np.where(r["mask"][..., None], col * (0.45 + 0.55 * lum), g)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def quality_sheet(meshes, path, cameras, titles=None, reports=None, stripes=14, limit=None, caption=None):
    """One sheet: for each mesh (rows), zebra and curvature-deviation
    images through each camera (columns), the flaws ringed. `reports`:
    fairness() results, one per mesh (computed when missing)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]
    reports = reports or [fairness(m) for m in meshes]
    titles = titles or [f"#{i + 1}" for i in range(len(meshes))]
    nc = 2 * len(cameras)
    fig, axes = plt.subplots(len(meshes), nc, figsize=(3.4 * nc, 3.4 * len(meshes)), squeeze=False)
    for row, (m, rep, tt) in enumerate(zip(meshes, reports, titles)):
        dev = rep["curvature"]["deviation"]
        dev = np.where(rep["curvature"]["checked"], dev, 0.0)
        for k, cam in enumerate(cameras):
            z = zebra_image(m, cam, stripes)
            cimg = curvature_image(m, cam, dev, limit=limit or 0.06)
            for col, (img, kind) in enumerate(((z, "zebra"), (cimg, "curvature off its smooth self"))):
                ax = axes[row, 2 * k + col]
                ax.imshow(img)
                ax.set_axis_off()
                ax.set_title(f"{tt}: {kind}", fontsize=8)
                if col == 1:
                    for cl in rep["wobble"][:12]:
                        p = cam.project(np.asarray([cl["at"]], float))[0]
                        ax.add_patch(plt.Circle((p[0], p[1]), 14, fill=False, color="k", lw=0.8))
        s = rep["score"]
        axes[row, 0].text(0.0, -0.06, f"wobble {s['wobbleArea']:.0f} mm² in {s['wobbleClusters']} spots, "
                          f"RMS {s['wobbleRMS'] * 1000:.1f}/m; tight {s['tightArea']:.0f} mm²",
                          transform=axes[row, 0].transAxes, fontsize=8, va="top")
    if caption:
        fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
