"""
Does the part still look like the reference?

A form drifts a little on every edit, and nobody measures it: the touch
microphone's outline was caught by eye, round after round. This module
scores a part against its references the same way on every build, so the
drift is a number and a picture instead of a hunch:

- an **image** (a designer's render, a sketch, a photo of a model): the
  part is drawn through the fixed camera the reference was made with
  (raster.Camera), and its outline is laid over the reference's edges.
  Each outline pixel is measured to the nearest edge in the reference:
  the share of the outline that has an edge within `tolerance` mm is the
  match, and the mean and 95th-percentile distances say how far off the
  rest is. Where the reference has a clean object mask (an alpha channel,
  a mask file, or a plain backdrop the object stands clear of) the
  silhouettes are also compared as areas (intersection over union);
- a **mesh** (the designer's form, the last round's part): both are drawn
  through the same cameras (six orthographic views by default) and their
  silhouettes compared exactly, and at any `sections` both are cut by the
  same planes and the cut areas compared (the share of area that differs).

Every reference writes one side-by-side image: the reference, the part
from the same camera, and the reference with the part's outline on it
(green where it lands on an edge of the reference, red where it does
not). An agent reads the numbers; a person looks at the picture.

`fit_camera` finds the camera for a reference that came without one (a
sketch): it turns, moves and scales a starting camera until the outline
of a known shape (the exterior the reference shows) lies on the image's
edges. The fitted camera is written out to be pasted into the spec: a
camera is fitted once and fixed after, so the comparison cannot move with
the part.
"""
from __future__ import annotations

import os

import numpy as np

from .raster import Camera, outline, render, shade_image

__all__ = ["score_image", "score_mesh", "score_references", "fit_camera", "reference_edges", "reference_mask",
           "sketch_strokes", "sketch_silhouette"]

SIX = ("top", "bottom", "front", "back", "left", "right")
MATCH_SHARE = 0.85      # share of the part's outline that must lie on reference edges
IOU = 0.95              # silhouettes of a mesh reference (or a masked image)


def _load_image(path):
    from PIL import Image

    im = Image.open(path)
    alpha = None
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        im = im.convert("RGBA")
        a = np.asarray(im)[..., 3]
        if a.min() < 250:
            alpha = a > 127
    rgb = np.asarray(im.convert("RGB"), float)
    return rgb, alpha


def _resize(rgb, alpha, width, height):
    from PIL import Image

    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb, alpha
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((width, height), Image.LANCZOS)
    if alpha is not None:
        alpha = np.asarray(Image.fromarray(alpha.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) > 127
    return np.asarray(im, float), alpha


def reference_edges(rgb, sigma: float = 2.0):
    """The reference's edges (bool H, W): Canny on its luminance, blurred
    enough that render grain and paper texture are not edges."""
    from skimage.feature import canny

    g = rgb @ np.array([0.299, 0.587, 0.114]) / 255.0
    lo, hi = np.percentile(g, [1, 99])
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    return canny(g, sigma=sigma, low_threshold=0.03, high_threshold=0.09)


def reference_mask(rgb, alpha=None):
    """(mask or None, how): the object in a reference image. Its alpha
    channel when it has one; else the region a flood from the image's
    border cannot reach without crossing an edge, kept only when that
    region is one clean blob clear of the border (a plain backdrop). A
    render on a studio floor with soft shadows often has no such blob;
    then there is no mask and the outline score stands alone."""
    if alpha is not None:
        return alpha, "alpha"
    from scipy import ndimage

    edges = reference_edges(rgb, sigma=2.5)
    wall = ndimage.binary_dilation(edges, iterations=2)
    lab, n = ndimage.label(~wall)
    if n == 0:
        return None, None
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    inside = ~np.isin(lab, border) | wall
    inside = ndimage.binary_fill_holes(ndimage.binary_erosion(inside, iterations=2))
    lab2, n2 = ndimage.label(inside)
    if n2 == 0:
        return None, None
    sizes = ndimage.sum(inside, lab2, range(1, n2 + 1))
    k = int(np.argmax(sizes)) + 1
    blob = lab2 == k
    share = blob.mean()
    touches = blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any()
    if touches or share < 0.01 or share > 0.9 or sizes[k - 1] < 0.9 * sizes.sum():
        return None, None
    # the backdrop inside the blob must differ from the object: a blob the
    # colour of the backdrop is a closed shadow line, not the object
    g = rgb.mean(axis=2)
    if abs(float(np.median(g[blob])) - float(np.median(g[~blob]))) < 6.0:
        return None, None
    return blob, "backdrop"


def sketch_strokes(rgb, dot: float = 0.008, dark: float = 0.82):
    """The pencil or pen strokes of a sketch (bool H, W), without the paper.

    A photographed sketch has uneven light across the page, a printed dot or
    line grid, hatching and smudges, and Canny finds edges in all of them:
    an outline laid anywhere near a grid dot scores as on an edge. Here the
    page's light is divided out (the image over a heavy blur of itself),
    what is darker than `dark` of the local paper is ink, and ink blobs
    smaller than `dot` of the image's shorter side across (grid dots,
    specks) are dropped. Hatching stays: it is ink, and it is inside the
    outline anyway."""
    from scipy import ndimage

    g = rgb @ np.array([0.299, 0.587, 0.114])
    n = min(g.shape)
    paper = ndimage.grey_closing(g, size=max(3, int(0.02 * n)))       # strokes closed over: the page alone
    paper = ndimage.gaussian_filter(paper, max(1.0, 0.02 * n))
    ink = g / np.maximum(paper, 1.0) < dark
    lab, k = ndimage.label(ink, structure=np.ones((3, 3)))
    if k == 0:
        return ink
    # a blob's extent, not its area: a stroke is long and thin, a dot is not
    sl = ndimage.find_objects(lab)
    span = np.array([max(a.stop - a.start, b.stop - b.start) for a, b in sl], float)
    keep = np.r_[False, span >= dot * n * 2.0]
    return keep[lab]


def _enclosed(strokes, gap, seed=None):
    from scipy import ndimage

    r = max(1, int(round(gap * min(strokes.shape))))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    pad = 2 * r + 2                                  # room round a view cropped tight to its outline
    st = np.pad(strokes, pad)
    wall = ndimage.binary_dilation(st, structure=disk)
    lab, n = ndimage.label(~wall)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    inside = ndimage.binary_fill_holes(~np.isin(lab, border[border > 0]))   # enclosed regions and their strokes
    lab2, n2 = ndimage.label(inside)
    if n2 == 0:
        return None, 0.0
    if seed is not None:
        k = lab2[pad + int(seed[1] * strokes.shape[0]), pad + int(seed[0] * strokes.shape[1])]
        if k == 0:
            return None, 0.0
        blob = lab2 == k
    else:
        sizes = ndimage.sum(inside, lab2, range(1, n2 + 1))
        blob = lab2 == (int(np.argmax(sizes)) + 1)
    paper = float((blob & ~wall).sum())               # enclosed page, not counting the strokes' own width
    # the dilation pushed the outline out by r: erode it back, to the
    # middle of the stroke
    return ndimage.binary_erosion(blob, structure=disk, border_value=0)[pad:-pad, pad:-pad], paper


def sketch_silhouette(strokes, gap: float = 0.02, cover: float = 0.6, seed=None):
    """(mask or None): the region the sketch's outermost closed stroke
    encloses. A quick line rarely closes, so gaps are bridged: from `gap`
    of the shorter side up, widening until the enclosed blob spans at least
    `cover` of the box the ink spans (a too small bridge leaves the outside
    leaking in, and the largest closed blob is some inner detail). The
    outside is flooded from the border, the largest enclosed blob is the
    object (or, with `seed` (x, y as shares of the image), the blob that
    holds that point: for a page where another view runs into this one),
    and the bridge is taken back off."""
    if not strokes.any():
        return None
    ys, xs = np.nonzero(strokes)
    ink = np.ptp(np.percentile(xs, [2, 98])) * np.ptp(np.percentile(ys, [2, 98]))
    best = None
    for g in (gap, 1.75 * gap, 2.5 * gap, 4.0 * gap):
        blob, paper = _enclosed(strokes, g, seed)
        if blob is None or not blob.any():
            continue
        if best is None or paper > best[0]:
            best = (paper, blob)
        if paper >= cover * 0.5 * ink:             # a solid outline fills about half its box or more
            break
    if best is None or best[1].mean() < 0.01:
        return None
    return best[1]


def _reference(rgb, alpha, kind):
    """(edges, mask, how) for a reference of this kind ("render" or "sketch")."""
    if kind == "sketch":
        st = sketch_strokes(rgb)
        m = alpha if alpha is not None else sketch_silhouette(st)
        return st | (outline(m) if m is not None else False), m, ("alpha" if alpha is not None else "sketch outline") if m is not None else None
    edges = reference_edges(rgb)
    m, how = reference_mask(rgb, alpha)
    if m is not None:
        # a mask's own border is an edge the reference certainly has
        edges = edges | outline(m)
    return edges, m, how


def _proportions(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float((np.ptp(xs) + 1) / (np.ptp(ys) + 1))


def _edge_distance(edges):
    from scipy import ndimage

    if not edges.any():
        return np.full(edges.shape, np.inf)
    return ndimage.distance_transform_edt(~edges)


def _part_outline(r):
    """The part's outline in a render: its silhouette, and the depth steps
    inside it (a port mouth, a rim): the lines a drawing of it would have."""
    m = r["mask"]
    out = outline(m)
    return out


def _depth_mm(r):
    d = r["depth"][r["mask"]]
    return float(np.median(d)) if len(d) else 1.0


def _overlay(rgb, r, on_edge):
    img = np.clip(0.55 * rgb + 0.45 * 255, 0, 255).astype(np.uint8)
    from scipy import ndimage

    line = outline(r["mask"])
    good = ndimage.binary_dilation(line & on_edge, iterations=1)
    bad = ndimage.binary_dilation(line & ~on_edge, iterations=1)
    img[good] = (20, 150, 70)
    img[bad] = (215, 40, 40)
    return img


def _sheet(panels, titles, path, caption=None):
    from PIL import Image, ImageDraw

    h = max(p.shape[0] for p in panels)
    pad, head, foot = 8, 22, 22 if caption else 0
    w = sum(p.shape[1] for p in panels) + pad * (len(panels) + 1)
    sheet = Image.new("RGB", (w, h + head + foot + pad), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    x = pad
    for p, t in zip(panels, titles):
        sheet.paste(Image.fromarray(p), (x, head))
        d.text((x + 2, 5), t, fill=(30, 30, 30))
        x += p.shape[1] + pad
    if caption:
        d.text((pad + 2, h + head + 4), caption, fill=(30, 30, 30))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path)
    return path


def score_image(mesh, image, camera, tolerance: float = 1.0, mask=None, out=None, name=None,
                min_match: float = MATCH_SHARE, min_iou=None, kind: str = "render") -> dict:
    """A part against one reference image through its camera.

    Returns {"match" (share of the part's outline with a reference edge
    within `tolerance` mm), "outlineMean", "outline95" (mm), "iou" (when
    the reference has a mask), "maskFrom", "ok", "image"}."""
    cam = camera if isinstance(camera, Camera) else Camera.from_spec(camera)
    rgb, alpha = _load_image(image)
    if mask is not None:
        m_rgb, _ = _load_image(mask)
        alpha = m_rgb.mean(axis=2) > 127
    rgb, alpha = _resize(rgb, alpha, cam.width, cam.height)
    r = render(mesh, cam)
    if not r["mask"].any():
        return {"ok": False, "error": "the camera does not see the part", "reference": name or os.path.basename(image)}
    px = cam.mm_per_pixel(_depth_mm(r))
    edges, ref_mask, how = _reference(rgb, alpha, kind)
    dist = _edge_distance(edges)
    line = _part_outline(r)
    d_mm = dist[line] * px
    tol = max(float(tolerance), 1.5 * px)
    res = {"reference": name or os.path.basename(image), "kind": "image",
           "match": float((d_mm <= tol).mean()), "outlineMean": float(np.mean(np.minimum(d_mm, 50.0))),
           "outline95": float(np.percentile(d_mm, 95)), "tolerance": tol, "mmPerPixel": px, "maskFrom": how}
    ok = res["match"] >= min_match
    if ref_mask is not None:
        inter = (ref_mask & r["mask"]).sum()
        union = (ref_mask | r["mask"]).sum()
        res["iou"] = float(inter / max(union, 1))
        res["proportions"] = {"part": _proportions(r["mask"]), "reference": _proportions(ref_mask)}
        if min_iou is not None:
            ok = ok and res["iou"] >= float(min_iou)
    res["minMatch"] = float(min_match)
    res["ok"] = bool(ok)
    if out is not None:
        cap = (f"outline on reference edges: {100 * res['match']:.0f}% within {tol:.2f} mm "
               f"(mean {res['outlineMean']:.2f}, 95% {res['outline95']:.2f} mm)"
               + (f"; silhouette overlap {100 * res['iou']:.1f}%" if "iou" in res else ""))
        shown = np.clip(rgb, 0, 255).astype(np.uint8)
        if kind == "sketch":                       # what was read off the page: ink, and the silhouette tinted
            shown = np.full_like(shown, 255)
            if ref_mask is not None:
                shown[ref_mask] = (215, 228, 245)
            shown[edges] = (40, 40, 40)
        res["image"] = _sheet([shown, shade_image(r),
                               _overlay(rgb, r, dist * px <= tol)],
                              ["reference" if kind != "sketch" else "sketch as read (ink, silhouette)", "part, same camera",
                               "part outline on the reference"], out, cap)
    return res


def _section_polys(mesh, point, normal):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    sec = mesh.section(plane_origin=point, plane_normal=normal)
    if sec is None:
        return None
    try:
        planar, _ = sec.to_2D(normal=normal)
    except TypeError:
        planar, _ = sec.to_2D()
    polys = [p for p in planar.polygons_full if p is not None and p.area > 1e-6]
    return unary_union(polys) if polys else None


def _section_to_plane(mesh, point, normal):
    """The section as shapely polygons in a frame fixed by the plane alone
    (not by the section's own extent), so two meshes' sections compare."""
    import trimesh
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, t)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    sec = mesh.section(plane_origin=point, plane_normal=n)
    if sec is None:
        return None
    rings = []
    for ent in sec.discrete:
        q = np.asarray(ent) - np.asarray(point, float)
        if len(q) >= 4:
            rings.append(Polygon(np.stack([q @ u, q @ v], axis=1)))
    rings = [r.buffer(0) for r in rings if r.is_valid or r.buffer(0).area > 0]
    # even-odd: a ring inside another is a hole
    out = None
    for r in sorted(rings, key=lambda r: -r.area):
        out = r if out is None else out.symmetric_difference(r)
    return out


def score_mesh(mesh, reference, cameras=None, sections=None, out=None, name=None, min_iou: float = IOU,
               max_section: float = 0.05, size: int = 480) -> dict:
    """A part against a reference mesh: silhouettes through the same
    cameras (the six orthographic views when none are given, framed on the
    reference) and the cut areas at `sections` ([{point, normal}])."""
    cams = cameras or [{"direction": d, "width": size, "height": size} for d in SIX]
    rows, panels, titles = [], [], []
    both = np.vstack([reference.bounds, mesh.bounds])
    corners = np.array([[x, y, z] for x in both[:, 0][[0, -1]] for y in both[:, 1][[0, -1]]
                        for z in both[:, 2][[0, -1]]])
    corners = np.array([[x, y, z] for x in (both[:, 0].min(), both[:, 0].max())
                        for y in (both[:, 1].min(), both[:, 1].max())
                        for z in (both[:, 2].min(), both[:, 2].max())])
    for c in cams:
        cam = c if isinstance(c, Camera) else Camera.from_spec(c)
        cam.fit(corners)
        a, b = render(reference, cam), render(mesh, cam)
        inter = (a["mask"] & b["mask"]).sum()
        union = (a["mask"] | b["mask"]).sum()
        label = c.get("direction") if isinstance(c, dict) and isinstance(c.get("direction"), str) else f"view {len(rows) + 1}"
        rows.append({"view": label, "iou": float(inter / max(union, 1)),
                     "addedPixels": int((b["mask"] & ~a["mask"]).sum()),
                     "missingPixels": int((a["mask"] & ~b["mask"]).sum()), "mmPerPixel": cam.mm_per_pixel(_depth_mm(a))})
        if out is not None:
            img = shade_image(b)
            img[a["mask"] & ~b["mask"]] = (215, 40, 40)       # reference there, part not
            img[b["mask"] & ~a["mask"]] = (40, 90, 215)       # part there, reference not
            panels.append(img)
            titles.append(f"{label}: {100 * rows[-1]['iou']:.1f}%")
    secs = []
    for s in sections or []:
        pa = _section_to_plane(reference, s["point"], s["normal"])
        pb = _section_to_plane(mesh, s["point"], s["normal"])
        if pa is None or pb is None or pa.is_empty:
            secs.append({"point": list(s["point"]), "normal": list(s["normal"]), "error": "the plane misses a mesh"})
            continue
        diff = pa.symmetric_difference(pb).area
        secs.append({"point": list(s["point"]), "normal": list(s["normal"]), "referenceArea": float(pa.area),
                     "partArea": float(pb.area), "differs": float(diff / max(pa.area, 1e-9)),
                     "outlineHausdorff": float(pa.exterior.hausdorff_distance(pb.exterior))
                     if pa.geom_type == "Polygon" and pb.geom_type == "Polygon" else None})
    worst = min(r["iou"] for r in rows) if rows else 1.0
    ok = worst >= min_iou and all(("differs" in s and s["differs"] <= max_section) or not sections for s in secs)
    res = {"reference": name or "mesh", "kind": "mesh", "views": rows, "iou": worst, "minIou": float(min_iou),
           "sections": secs, "ok": bool(ok)}
    if out is not None and panels:
        res["image"] = _sheet(panels, titles, out,
                              "red: in the reference, not in the part; blue: in the part, not in the reference")
    return res


def fit_camera(shape_mesh, image, start, iterations: int = 400, kind: str = "render", turn: bool = True) -> dict:
    """The camera a reference image was drawn from, found by laying the
    outline of `shape_mesh` (what the reference shows) on the image's
    edges. `start`: a camera spec roughly right (a named `direction`, or a
    position and lookAt). Returns {"camera" (a spec to fix in the study),
    "outlineMean" (pixels), "match"}.

    `turn=False` keeps the start's direction and fits only the zoom and the
    pan: right for an orthographic sketch view (front, side, top), where
    letting the camera turn lets it hide a wrong rake or a wrong taper
    behind a tilted view. `kind="sketch"` reads the image as a drawing
    (sketch_strokes)."""
    from scipy.optimize import minimize

    cam0 = start if isinstance(start, Camera) else Camera.from_spec(start)
    rgb, alpha = _load_image(image)
    rgb, alpha = _resize(rgb, alpha, cam0.width, cam0.height)
    edges, mask, _ = _reference(rgb, alpha, kind)
    dist = np.minimum(_edge_distance(edges), 40.0)
    centre = np.asarray(shape_mesh.bounds, float).mean(axis=0)
    size = float(np.linalg.norm(shape_mesh.extents))
    cam0.fit(np.asarray(shape_mesh.vertices))
    eye0 = cam0.position - centre
    rad0 = float(np.linalg.norm(eye0)) or size
    persp = cam0.perspective

    def camera(x):
        if not turn:
            x = np.r_[0.0, 0.0, 0.0, x]
        az, el, roll, zoom, pu, pv = x
        # turn the eye about the shape's centre, then pan and zoom
        f0, r0, u0 = cam0.forward, cam0.right, cam0.up
        f = np.cos(el) * (np.cos(az) * f0 + np.sin(az) * r0) + np.sin(el) * u0
        f /= np.linalg.norm(f)
        up = np.cos(roll) * u0 + np.sin(roll) * r0
        look = centre + size * (pu * r0 + pv * u0)
        kw = dict(position=look - f * rad0 * (np.exp(zoom) if persp else 1.0), look_at=look, up=up,
                  width=cam0.width, height=cam0.height)
        if persp:
            c = Camera(**kw)
            c.focal = cam0.focal
        else:
            c = Camera(ortho=cam0.ortho * np.exp(zoom), **kw)
        return c

    def cost(x):
        r = render(shape_mesh, camera(x))
        line = outline(r["mask"])
        if not line.any():
            return 40.0
        return float(dist[line].mean())

    nd = 6 if turn else 3
    best = minimize(cost, np.zeros(nd), method="Nelder-Mead",
                    options={"maxiter": iterations, "xatol": 1e-4, "fatol": 1e-3,
                             "initial_simplex": np.vstack([np.zeros(nd), 0.06 * np.eye(nd)])})
    cam = camera(best.x)
    r = render(shape_mesh, cam)
    line = outline(r["mask"])
    return {"camera": cam.to_spec(), "outlineMean": float(best.fun),
            "match": float((dist[line] <= 2.0).mean()) if line.any() else 0.0}


def score_references(mesh, references, base_dir=None, out_dir=None, exterior=None) -> list:
    """Every reference of a spec's `references` list scored against `mesh`.
    Images need a `camera`; a reference with `"camera": "fit"` (and a
    `start`) has its camera fitted to `exterior` first and reported, to be
    fixed in the spec."""
    import trimesh

    rows = []
    for i, ref in enumerate(references or []):
        name = ref.get("id") or f"reference-{i + 1}"
        out = os.path.join(out_dir, f"{name}.png") if out_dir else None

        def full(p):
            return p if base_dir is None or os.path.isabs(p) else os.path.join(base_dir, p)

        try:
            if ref.get("mesh") is not None:
                rm = trimesh.load(full(ref["mesh"]), force="mesh")
                rows.append(score_mesh(mesh, rm, cameras=ref.get("cameras"), sections=ref.get("sections"), out=out,
                                       name=name, min_iou=ref.get("minIou", IOU),
                                       max_section=ref.get("maxSectionDifference", 0.05)))
                continue
            cam = ref.get("camera")
            fitted = None
            if cam == "fit" or (isinstance(cam, dict) and cam.get("fit")):
                if exterior is None:
                    raise ValueError("fitting a camera needs the exterior the reference shows")
                start = ref.get("start") or (isinstance(cam, dict) and {k: v for k, v in cam.items() if k != "fit"})
                if not start:
                    raise ValueError("camera 'fit' needs a start: a camera roughly right, e.g. {direction: front}")
                fitted = fit_camera(exterior, full(ref["image"]), start, kind=ref.get("kind", "render"),
                                    turn=ref.get("turn", ref.get("kind") != "sketch"))
                cam = fitted["camera"]
            row = score_image(mesh, full(ref["image"]), cam, tolerance=ref.get("tolerance", 1.0),
                              mask=full(ref["mask"]) if ref.get("mask") else None, out=out, name=name,
                              min_match=ref.get("minMatch", MATCH_SHARE), min_iou=ref.get("minIou"),
                              kind=ref.get("kind", "render"))
            if fitted is not None:
                row["fittedCamera"] = fitted["camera"]
                row["note"] = "camera fitted on this run: copy fittedCamera into the reference's camera to fix it"
            rows.append(row)
        except (OSError, ValueError, KeyError) as exc:
            rows.append({"reference": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return rows
