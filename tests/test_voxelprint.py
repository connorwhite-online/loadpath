"""
Tests for voxelprint.py (GrabCAD Voxel Print PNG slice export of a
multi-material fraction field). Run: `python3 tests/test_voxelprint.py`.
Uses a low-resolution printer override (0.1 mm voxels) so it runs in seconds.
"""
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from loadpath import voxelprint as vp  # noqa: E402

LOWRES = {"voxel_mm": (0.1, 0.1, 0.1)}
MATS = [("VeroCyan", None), ("Agilus30Blk", None)]


def _read_stack(out_dir, files):
    from PIL import Image
    return np.stack([np.asarray(Image.open(os.path.join(out_dir, f)))
                     for f in files])                  # (nz, H, W, 4)


def _gradient_block():
    """10 x 10 x 3 mm block, 0.5 mm cells, linear 0 -> 1 along x; the
    y > 7 mm strip is void."""
    nx, ny, nz = 20, 20, 6
    x = (np.arange(nx) + 0.5) / nx
    frac = np.broadcast_to(x[:, None, None], (nx, ny, nz)).copy()
    solid = np.ones((nx, ny, nz), dtype=bool)
    solid[:, 14:, :] = False                           # y >= 7 mm
    return frac, solid


def _check_gradient(dither):
    frac, solid = _gradient_block()
    with tempfile.TemporaryDirectory() as d:
        s = vp.export_voxel_slices(d, frac, (0, 0, 0), 0.5, MATS,
                                   printer=LOWRES, dither=dither,
                                   solid=solid)
        assert (s["width_px"], s["height_px"], s["layers_total"]) == \
            (100, 100, 30), s
        pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        assert len(pngs) == s["layers_written"] == 30
        assert pngs == s["files"]
        man = json.load(open(os.path.join(d, "manifest.json")))
        assert man["layers_total"] == 30 and len(man["materials"]) == 2
        assert man["materials"][1]["rgb"] == [40, 77, 108]
        img = _read_stack(d, pngs)

    alpha = img[..., 3]
    cyan = np.all(img[..., :3] == (0, 90, 158), axis=-1) & (alpha == 255)
    agil = np.all(img[..., :3] == (40, 77, 108), axis=-1) & (alpha == 255)
    void = alpha == 0
    # every pixel is exactly one palette entry (no blended colours)
    assert np.all(cyan | agil | void)
    # row 0 = +Y edge: rows 0..29 are y > 7 mm (void), rows >= 30 solid
    y = 10.0 - (np.arange(100) + 0.5) * 0.1            # row centre, mm
    assert void[:, y > 7.15].all(), "void region got material"
    assert not void[:, y < 6.85].any(), "solid region has holes"
    # per-column Agilus share follows the 0 -> 1 gradient
    rows = y < 6.85
    got = agil[:, rows].mean(axis=(0, 1))
    xc = (np.arange(100) + 0.5) * 0.1
    want = np.clip(xc / 10.0, 0.025, 0.975)            # edge cells clamp
    err = np.abs(got - want)
    assert err.max() < 0.06, (dither, err.max(), np.argmax(err))
    assert err.mean() < 0.02, (dither, err.mean())
    # summary reports achieved vs requested shares within 1%
    for m in s["materials"]:
        assert abs(m["share_error"]) < 0.01, m
    assert abs(s["materials"][1]["achieved_share"] - 0.5) < 0.01


def test_gradient_floyd_steinberg():
    _check_gradient("floyd-steinberg")


def test_gradient_ordered():
    _check_gradient("ordered")


def test_gradient_blue_noise():
    _check_gradient("blue-noise")


def test_three_materials_porous():
    """k-channel input, void dithered as a channel (crisp=False), max_layers
    truncation, RGB images with a white background."""
    nx, ny, nz = 10, 10, 4
    f = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    f[..., 0], f[..., 1], f[..., 2] = 0.2, 0.3, 0.4    # 10% void
    mats = [("A", "#ff0000"), ("B", (0, 255, 0)), {"name": "C",
                                                   "color": "#0000ff"}]
    with tempfile.TemporaryDirectory() as d:
        s = vp.export_voxel_slices(d, f, (5, 5, 5), 1.0, mats,
                                   printer=LOWRES, crisp=False,
                                   image_mode="RGB", max_layers=7)
        pngs = sorted(p for p in os.listdir(d) if p.endswith(".png"))
        assert len(pngs) == 7 and s["truncated"] and s["layers_total"] == 40
        img = _read_stack(d, pngs)
    assert img.shape[-1] == 3
    n = img.shape[0] * img.shape[1] * img.shape[2]
    shares = [np.all(img == c, axis=-1).sum() / n
              for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (255, 255, 255))]
    for got, want in zip(shares, (0.2, 0.3, 0.4, 0.1)):
        assert abs(got - want) < 0.01, shares
    assert abs(sum(shares) - 1.0) < 1e-9


def test_validation():
    f = np.zeros((4, 4, 4, 7))
    for bad in (dict(fraction=f, materials=["VeroCyan"] * 7),
                dict(fraction=f[..., :2], materials=["Tango+",
                                                     "Agilus30Clr"]),
                dict(fraction=f[..., :2], materials=["VeroCyan", "x"]),
                dict(fraction=f[..., 0], materials=MATS, printer="nope")):
        args = dict(origin=(0, 0, 0), pitch=1.0, printer=LOWRES)
        args.update(bad)
        try:
            with tempfile.TemporaryDirectory() as d:
                vp.export_voxel_slices(d, **args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad.keys()}")


def test_profiles_and_png_fallback():
    p = vp._printer_profile("j850")
    assert (p["dpi_x"], p["dpi_y"], p["layer_mm"]) == (600, 300, 0.027)
    assert vp._printer_profile("j850-hq")["layer_mm"] == 0.014
    # the zlib PNG writer round-trips through Pillow
    from PIL import Image
    import io
    img = np.random.default_rng(0).integers(0, 256, (7, 5, 4), np.uint8)
    back = np.asarray(Image.open(io.BytesIO(vp._png_bytes(img))))
    assert np.array_equal(back, img)


TESTS = [test_gradient_floyd_steinberg, test_gradient_ordered,
         test_gradient_blue_noise, test_three_materials_porous,
         test_validation, test_profiles_and_png_fallback]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
