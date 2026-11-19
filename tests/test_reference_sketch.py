"""
Reading a pencil sketch as a reference (loadpath.reference, kind="sketch").

A sketch on dot-grid paper under uneven light is not a render: the default
edge finder reads the dots and the paper's shading as edges, and the
part's outline "matches" them. These hold the sketch reader to the ink:
the dots drop out, the drawn outline closes into a silhouette even with a
gap left in a stroke, and a fixed-direction fit (turn=False) keeps the
view a sketch was drawn from, so a part with the wrong proportions scores
worse instead of being turned until it fits.

Run: `python3 tests/test_reference_sketch.py` (a few seconds).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from loadpath.reference import fit_camera, score_image, sketch_silhouette, sketch_strokes

W, H = 480, 360


def _paper(dots=True):
    """Warm paper, darker toward one corner, with a dot grid."""
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 238 - 40 * (xx / W) * (yy / H)
    rgb = np.stack([shade, shade * 0.97, shade * 0.9], -1)
    if dots:
        for y in range(8, H, 16):
            for x in range(8, W, 16):
                rgb[y:y + 2, x:x + 2] *= 0.7
    return rgb.astype(np.uint8)


def _sketch(box, gap=False):
    """A rounded box drawn in pencil, the true filled region beside it."""
    im = Image.fromarray(_paper())
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(box, radius=30, outline=(70, 68, 66), width=3)
    if gap:                                    # a break in the stroke, as a quick sketch leaves
        x0, y0, x1, _ = box
        d.rectangle([x0 + (x1 - x0) // 2, y0 - 3, x0 + (x1 - x0) // 2 + 6, y0 + 3], fill=(230, 224, 214))
    truth = Image.new("L", (W, H), 0)
    ImageDraw.Draw(truth).rounded_rectangle(box, radius=30, fill=255)
    return np.asarray(im), np.asarray(truth) > 0


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_the_dots_and_the_shading_are_not_ink():
    rgb, truth = _sketch((100, 90, 380, 270))
    ink = sketch_strokes(rgb)
    assert not ink[:60].any() and not ink[:, :60].any(), "dot grid or paper shading read as ink"
    from scipy import ndimage
    edge = truth ^ ndimage.binary_erosion(truth, iterations=3)
    assert ink[edge].mean() > 0.5, ink[edge].mean()


def test_the_outline_closes_into_a_silhouette_across_a_gap():
    for gap in (False, True):
        rgb, truth = _sketch((100, 90, 380, 270), gap=gap)
        sil = sketch_silhouette(sketch_strokes(rgb))
        assert sil is not None and _iou(sil, truth) > 0.93, (gap, None if sil is None else _iou(sil, truth))


def _orthographic_view(mesh, path):
    """A sketch of the mesh's front (looking along +y), drawn as its outline."""
    from loadpath.raster import Camera, outline, render

    cam = Camera.from_spec({"direction": "front", "width": W, "height": H})
    cam.fit(np.asarray(mesh.vertices))
    r = render([mesh], cam)
    line = outline(r["mask"])
    from scipy import ndimage
    line = ndimage.binary_dilation(line, iterations=1)
    rgb = _paper().astype(float)
    rgb[line] = (70, 68, 66)
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return cam


def test_a_fixed_view_fit_scores_the_drawn_shape_and_not_a_stretched_one():
    part = trimesh.creation.box((120.0, 40.0, 60.0))
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "front.png")
        _orthographic_view(part, ref)
        start = {"direction": "front", "width": W, "height": H}
        fit = fit_camera(part, ref, start, kind="sketch", turn=False)
        good = score_image(part, ref, fit["camera"], tolerance=2.0, kind="sketch", out=os.path.join(tmp, "s.png"))
        assert good["maskFrom"] == "sketch outline" and good["iou"] > 0.9, good
        assert abs(good["proportions"]["part"] - good["proportions"]["reference"]) < 0.05, good["proportions"]
        assert os.path.exists(good["image"])
        # a part twice as tall as drawn: the fit can zoom but not turn, so it scores worse
        tall = trimesh.creation.box((120.0, 40.0, 120.0))
        f2 = fit_camera(tall, ref, start, kind="sketch", turn=False)
        bad = score_image(tall, ref, f2["camera"], tolerance=2.0, kind="sketch")
        assert bad["iou"] < good["iou"] - 0.15, (bad["iou"], good["iou"])
        assert bad["proportions"]["part"] < 0.7 * bad["proportions"]["reference"], bad["proportions"]


def test_a_spec_reference_can_be_a_sketch():
    from loadpath.reference import score_references
    from loadpath.schema import SCHEMAS
    try:
        import jsonschema
    except ImportError:                        # not a dependency: the schema check is skipped without it
        jsonschema = None

    part = trimesh.creation.box((120.0, 40.0, 60.0))
    with tempfile.TemporaryDirectory() as tmp:
        _orthographic_view(part, os.path.join(tmp, "front.png"))
        ref = {"id": "front", "image": "front.png", "kind": "sketch", "camera": "fit",
               "start": {"direction": "front", "width": W, "height": H}, "tolerance": 2.0}
        items = SCHEMAS["study"]["properties"]["references"]["items"]
        errs = [] if jsonschema is None else list(jsonschema.Draft202012Validator({**items, "$defs": SCHEMAS["study"].get("$defs", {})}).iter_errors(ref))
        assert not errs, [e.message for e in errs]
        row, = score_references(part, [ref], base_dir=tmp, exterior=part)
        assert row["maskFrom"] == "sketch outline" and row["iou"] > 0.9 and "fittedCamera" in row, row
        bare, = score_references(part, [{**ref, "start": None}], base_dir=tmp, exterior=part)
        assert not bare["ok"] and "needs a start" in bare["error"], bare


TESTS = [
    test_a_spec_reference_can_be_a_sketch,
    test_the_dots_and_the_shading_are_not_ink,
    test_the_outline_closes_into_a_silhouette_across_a_gap,
    test_a_fixed_view_fit_scores_the_drawn_shape_and_not_a_stretched_one,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print("PASS", t.__name__)
        except Exception:  # noqa: BLE001
            import traceback
            failed += 1
            print("FAIL", t.__name__)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
