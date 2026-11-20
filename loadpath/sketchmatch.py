"""
How close is the model to the sketch, in millimetres, and where is it off?

A designer's pencil views (front, side, top, back) are the brief for a
form. `reference.score_image` scores a part's outline against an image
through a fixed camera, as a share of the outline. This module goes the
other way round, the way a surfacing artist uses a sketch: the sketch's
outline is traced as a curve, laid over the model's silhouette in the same
orthographic view (scale and position fitted, since a photographed page
has no scale of its own), and the two are compared point by point, both
ways, in mm. The result says by how much, and where round the outline, the
model is fat or thin against the drawing, and the traced outline comes
back as a smooth curve in model space: a rail to build the surface on.

    m = match_view(mesh, "front", "sketch.jpg", crop=(x0, y0, x1, y1))
    m["mean"], m["p95"], m["max"]          # mm, both ways
    m["segments"]                          # signed deviation round the outline
    rail = outline_curve(m)                # surfacing.Curve in model mm
    match_sheet([m, ...], "vs-sketch.png")

Signs: + where the model stands outside the sketch's outline (fat), -
where it falls inside (thin).
"""
from __future__ import annotations

import numpy as np

from .raster import DIRECTIONS, Camera, render


def _view_axes(view):
    """(forward, right, up) world unit vectors of a named view or
    (forward, up) pair, matching raster.Camera."""
    if isinstance(view, str):
        fwd, up = DIRECTIONS[view]
    else:
        fwd, up = view
    cam = Camera(direction=fwd, up=up, width=16, height=16)
    return cam.forward, cam.right, cam.up


def model_silhouette(mesh, view, pixel=0.25):
    """The model's orthographic silhouette in a view: {"mask" (H, W),
    "origin" (mm of pixel (0, 0)'s centre along right and up), "pixel",
    "contour" (k, 2: the outermost outline in mm, right and up)}."""
    fwd, right, up = _view_axes(view)
    V = np.asarray(mesh.vertices, float)
    u, v = V @ right, V @ up
    W = int(np.ceil(np.ptp(u) / pixel)) + 16
    H = int(np.ceil(np.ptp(v) / pixel)) + 16
    cam = Camera(direction=fwd, up=up, ortho=W * pixel, width=W, height=H)
    cam.fit(V)
    cam.ortho = W * pixel
    r = render(mesh, cam)
    mask = r["mask"]
    c = cam.position
    u0 = c @ right - 0.5 * W * pixel + 0.5 * pixel
    v0 = c @ up + 0.5 * H * pixel - 0.5 * pixel
    cont = _largest_contour(mask)
    mm = np.column_stack([u0 + cont[:, 1] * pixel, v0 - cont[:, 0] * pixel])
    return {"mask": mask, "origin": (u0, v0), "pixel": pixel, "contour": mm, "axes": (fwd, right, up)}


def _largest_contour(mask):
    from scipy import ndimage
    from skimage import measure
    filled = ndimage.binary_fill_holes(mask)
    cs = measure.find_contours(np.pad(filled.astype(float), 1), 0.5)
    c = max(cs, key=len) - 1.0
    return c[:-1] if np.allclose(c[0], c[-1]) else c


def sketch_outline(image, crop=None, seed=None, gap=0.02, smooth=1.5, exclude=()):
    """The outermost closed outline a pencil sketch draws (row, col pixel
    coordinates in the crop, (k, 2)), with the crop's RGB. `crop`: (x0, y0,
    x1, y1) pixels of the view on the page; `seed`: (x, y) shares of the
    crop inside the object, when another view runs into this one;
    `exclude`: (x0, y0, x1, y1) boxes of the crop whose ink is ignored (a
    neighbouring view's strokes that run into this one)."""
    from scipy import ndimage

    from .reference import _load_image, sketch_silhouette, sketch_strokes
    rgb, _ = _load_image(image) if isinstance(image, str) else (np.asarray(image), None)
    if crop is not None:
        x0, y0, x1, y1 = map(int, crop)
        rgb = rgb[y0:y1, x0:x1]
    strokes = sketch_strokes(rgb.astype(float))
    for x0, y0, x1, y1 in exclude:
        strokes[int(y0):int(y1), int(x0):int(x1)] = False
    sil = sketch_silhouette(strokes, gap=gap, seed=seed)
    if sil is None:
        raise ValueError("sketch_outline: no closed outline found (try a tighter crop or a seed)")
    if smooth:
        sil = ndimage.gaussian_filter(sil.astype(float), smooth) > 0.5
    return _largest_contour(sil), rgb


def _similarity(src, dst, rotate=False):
    """Least-squares s, R, t with dst ~ s R src + t (Umeyama)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    if rotate:
        U, S, Vt = np.linalg.svd(b.T @ a)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1.0, d])
        R = U @ D @ Vt
        s = float((S * np.diag(D)).sum() / (a * a).sum())
    else:
        R = np.eye(2)
        s = float((a * b).sum() / (a * a).sum())
    return s, R, md - s * (R @ ms)


def _icp(P, M, s, R, t, rotate, scale, iterations):
    from scipy.spatial import cKDTree
    tm = cKDTree(M)
    for _ in range(iterations):
        Q = s * P @ R.T + t
        _, i = tm.query(Q)
        _, j = cKDTree(Q).query(M)
        src = np.vstack([P, P[j]])
        dst = np.vstack([M[i], M])
        s2, R2, t2 = _similarity(src, dst, rotate)
        if scale:
            s2 = scale
            t2 = dst.mean(0) - s2 * (R2 @ src.mean(0))
        done = abs(s2 - s) < 1e-7 * s and np.abs(t2 - t).max() < 1e-5
        s, R, t = s2, R2, t2
        if done:
            break
    Q = s * P @ R.T + t
    err = float(tm.query(Q)[0].mean() + cKDTree(Q).query(M)[0].mean())
    return err, s, R, t


def register(sketch_rc, model_mm, rotate=False, scale=None, iterations=80):
    """Fit the sketch outline (row, col pixels) onto the model outline (mm):
    image rows run down, so (col, -row) is laid on (right, up). The scale
    (mm per pixel) is fitted unless `scale` is given; rotation only with
    `rotate`. Iterative closest points, both ways, from bounding-box starts
    (the widths matched, the heights matched, the areas matched); the start
    that ends closest wins. Returns (s, R, t) with mm = s R (col, -row) + t."""
    P = np.column_stack([sketch_rc[:, 1], -sketch_rc[:, 0]]).astype(float)
    M = np.asarray(model_mm, float)
    wp, hp = np.ptp(P[:, 0]), np.ptp(P[:, 1])
    wm, hm = np.ptp(M[:, 0]), np.ptp(M[:, 1])
    starts = [scale] if scale else [wm / wp, hm / hp, float(np.sqrt(wm * hm / (wp * hp)))]
    best = None
    for s0 in starts:
        t0 = 0.5 * (M.min(0) + M.max(0)) - s0 * 0.5 * (P.min(0) + P.max(0))
        r = _icp(P, M, s0, np.eye(2), t0, rotate, scale, iterations)
        if best is None or r[0] < best[0]:
            best = r
    return best[1:]


def _inside(poly, pts):
    from matplotlib.path import Path
    return Path(poly).contains_points(pts)


def compare(model_mm, sketch_mm, tolerance=2.0, segments=16):
    """Both outlines in mm (k, 2). {"mean", "p95", "max" (mm, both ways),
    "within" (share of both outlines within `tolerance`), "iou",
    "segments": [{"from", "to" (share round the model outline from its
    top), "mean" (signed mm: + model outside the sketch), "at" (mm)}],
    "signed" (per model point)}."""
    from scipy.spatial import cKDTree
    M, K = np.asarray(model_mm, float), np.asarray(sketch_mm, float)
    dm, _ = cKDTree(K).query(M)
    dk, _ = cKDTree(M).query(K)
    sign = np.where(_inside(K, M), -1.0, 1.0)
    signed = sign * dm
    both = np.r_[dm, dk]
    # start the segments at the top of the model outline
    i0 = int(np.argmax(M[:, 1]))
    order = np.roll(np.arange(len(M)), -i0)
    segs = []
    for a in range(segments):
        sl = order[int(a * len(M) / segments):int((a + 1) * len(M) / segments)]
        segs.append({"from": round(a / segments, 3), "to": round((a + 1) / segments, 3),
                     "mean": round(float(signed[sl].mean()), 2),
                     "at": np.round(M[sl].mean(0), 1).tolist()})
    # areas: intersection over union of the two outlines
    lo = np.minimum(M.min(0), K.min(0))
    hi = np.maximum(M.max(0), K.max(0))
    px = max(float(np.ptp(np.r_[M[:, 0], K[:, 0]])) / 400.0, 0.1)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], px), np.arange(lo[1], hi[1], px))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    im, ik = _inside(M, G), _inside(K, G)
    iou = float((im & ik).sum() / max((im | ik).sum(), 1))
    return {"mean": float(both.mean()), "p95": float(np.percentile(both, 95)), "max": float(both.max()),
            "within": float((both <= tolerance).mean()), "tolerance": tolerance, "iou": iou,
            "segments": segs, "signed": signed}


def match_view(mesh, view, image, crop=None, seed=None, rotate=False, scale=None, tolerance=2.0, pixel=0.25,
               name=None, exclude=(), gap=0.02):
    """The model against one sketch view: the model's silhouette, the
    sketch's outline registered onto it, and `compare` of the two. Pass
    `scale` (mm per sketch pixel) to hold the drawing's size (once one
    view's fitted scale is trusted, the others can share it)."""
    sil = model_silhouette(mesh, view, pixel)
    rc, rgb = sketch_outline(image, crop, seed, gap=gap, exclude=exclude)
    s, R, t = register(rc, sil["contour"], rotate=rotate, scale=scale)
    P = np.column_stack([rc[:, 1], -rc[:, 0]])
    K = s * P @ R.T + t
    out = compare(sil["contour"], K, tolerance)
    out.update({"view": view if isinstance(view, str) else "custom", "name": name or (view if isinstance(view, str) else "view"),
                "model": sil["contour"], "sketch": K, "scale": s, "rotation": R, "translation": t,
                "image": rgb, "axes": sil["axes"]})
    return out


def outline_curve(match, depth=0.0, n_poles=48, fair=0.02, which="sketch"):
    """The traced outline (the sketch's, or `which`="model") as a closed
    surfacing.Curve in model mm, in the view's plane at `depth` along the
    view direction: a silhouette rail for a loft."""
    from .surfacing import fit_curve, resample
    fwd, right, up = match["axes"]
    O = match[which]
    P = O[:, :1] * right + O[:, 1:2] * up + depth * fwd
    P = resample(P, 400, closed=True)
    return fit_curve(P, n_poles=n_poles, closed=True, fair=fair)


def summary(match):
    """A match without its arrays: for JSON."""
    keep = ("name", "view", "mean", "p95", "max", "within", "tolerance", "iou", "scale", "segments")
    out = {k: match[k] for k in keep if k in match}
    for k in ("mean", "p95", "max", "within", "iou", "scale"):
        if k in out:
            out[k] = round(float(out[k]), 4 if k == "scale" else 2)
    return out


def match_sheet(matches, path, title=None, limit=6.0):
    """One panel per view: the sketch (its photo, laid in model mm), its
    traced outline (grey) and the model's silhouette coloured by signed
    deviation (red fat, blue thin, clipped at `limit` mm)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    n = len(matches)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.2 * rows), squeeze=False)
    for ax, m in zip(axes.ravel(), matches):
        img = m["image"]
        s, t = m["scale"], m["translation"]
        R = m["rotation"]
        if np.allclose(R, np.eye(2)):
            h, w = img.shape[:2]
            ext = (t[0] - 0.5 * s, t[0] + (w - 0.5) * s, t[1] - (h - 0.5) * s, t[1] + 0.5 * s)
            ax.imshow(img.astype(np.uint8) if img.dtype != np.uint8 else img, extent=ext, alpha=0.45)
        K, M, sg = m["sketch"], m["model"], m["signed"]
        ax.plot(np.r_[K[:, 0], K[:1, 0]], np.r_[K[:, 1], K[:1, 1]], color="0.35", lw=1.0, ls="--")
        seg = np.stack([M, np.roll(M, -1, axis=0)], 1)
        lc = LineCollection(seg, cmap="coolwarm", norm=plt.Normalize(-limit, limit), linewidths=2.2)
        lc.set_array(sg)
        ax.add_collection(lc)
        ax.set_aspect("equal")
        pad = 8
        allp = np.vstack([K, M])
        ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
        ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
        ax.set_title(f"{m['name']}: mean {m['mean']:.1f} mm, 95% {m['p95']:.1f} mm, IoU {m['iou']:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-limit, limit)), ax=axes.ravel().tolist(),
                 shrink=0.6, label="model vs sketch, mm (+ fat, - thin)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
