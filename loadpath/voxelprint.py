"""
Voxel print export: a multi-material fraction field as a GrabCAD Voxel Print
PNG slice stack for PolyJet material jetting (Stratasys J750 / J850 / J8,
J5) and other voxel-addressable printers.

A multi-material design here is a continuous field: per cell, the volume
fraction of each of k materials (0 = rigid Vero, 1 = rubbery Agilus30, in
between a graded mix), the rest void. A material-jetting printer drops
exactly one material per printer voxel, so a mix is made by dithering: the
field is resampled (trilinear) to the printer's native voxel grid and every
printer voxel is assigned one material (or left empty) so that the local
share of each material matches the requested fraction. The result is one
PNG per layer whose pixel colours are the materials, which GrabCAD Print's
Voxel Print Utility turns into a .GCVF print job (Apps > Voxel Print
Utility: give the folder and the slice prefix, then map each colour to a
loaded material).

Format facts, sourced from Stratasys "Guide to Voxel Printing"
(https://support.stratasys.com/en/Software/GrabCAD-Print/Tips-Guides-and-FAQs/Guide-to-Voxel-Printing,
formerly https://help.grabcad.com/article/230-guide-to-voxel-printing):

- PNG is the supported input for new work (one PNG per layer); the per-
  material 1-bit BMP route is legacy.
- J750 slice resolution is 600 DPI in X and 300 DPI in Y (a 1 inch cube is
  600 x 300 pixel slices, so image width is X and height is Y).
- Layer thickness by build mode: High Quality 0.014 mm, High Mix 0.027 mm,
  High Speed 0.027 mm. Slices should be cut at the print layer thickness;
  otherwise the printer repeats or skips slices.
- At least 30 slices.
- Each image may use at most 6 distinct colours (3 in High Speed) plus the
  background colour; the utility matches colours to loaded materials.
  Anti-aliasing / dithering *between colours* (blended RGB values at
  boundaries) causes a "Too many colors" error. The dithering done here
  never blends colours: every pixel is exactly one palette colour, so it is
  the safe kind.
- The guide's example makes background pixels transparent (alpha 0), and
  the utility lets the background colour be checked/set. The slice file
  prefix (e.g. "slice_") is entered in the utility.
- Base-material reference RGB colours (the table in the guide) are in
  STRATASYS_COLORS below, used as default colours by material name.
- The J5 uses 300 x 300 DPI and 0.01875 mm slices (Stratasys J5 GrabCAD
  Print manual, https://manuals.plus/stratasys/j5-grabcad-print-manual).

Assumed (not stated in a public spec; all configurable):

- The J850 / J8-series profile is taken equal to the J750 (600 x 300 DPI,
  0.027 mm High Mix); the guide documents the J750 only. "j850" defaults to
  High Mix, the mode for graded digital materials; "j850-hq" is 0.014 mm.
- Images are 8-bit RGBA PNGs, void = fully transparent (0, 0, 0, 0)
  (`image_mode="RGB"` writes an opaque background colour instead, white by
  default). Bit depth and palette-vs-truecolour aren't specified.
- Files are named prefix + zero-padded layer index + ".png", layer 0 at the
  bottom (lowest z). Image row 0 is the +Y edge (the slice seen from above,
  +X right, +Y up); pass `flip_y=False` for row 0 = -Y. Other voxel
  pipelines (e.g. OpenVCAD's material-inkjet compiler,
  https://matterassembly.org/OpenVCAD-Docs/v3/guides/compilers/material-inkjet.html)
  note the handedness may need checking against the tool: print a test
  coupon with an asymmetric feature first.
- The image covers exactly the field's bounding box (origin to origin +
  shape * pitch), each axis rounded to whole printer voxels.

Dithering. "floyd-steinberg" is multi-channel error diffusion within each
layer: the palette is the simplex vertices (one-hot material or void), so
the nearest palette entry is the channel with the largest value, and the
quantization error goes 7/16, 3/16, 5/16, 1/16 to the unvisited neighbours.
The scan is serial in principle; it is vectorized here as a wavefront
(pixel (y, x) only depends on pixels with x + 2y smaller, so all pixels on
one line x + 2y = t are done at once) and across a batch of layers. Layers
alternate scan orientation, and a small random perturbation of each choice
(`jitter`; the error is still diffused against the true value) breaks up
FS's regular textures, so patterns don't stack into identical columns.
"ordered" (8x8 Bayer) and "blue-noise" (a high-pass noise tile) are
threshold dithers: a threshold u per pixel picks the material whose
cumulative-fraction interval contains u, fully vectorized; the tile shifts
from layer to layer. All three preserve each material's mean share.

Geometry. By default (`crisp=True`) the part's outline is not dithered: a
printer voxel is solid where the resampled total occupancy is >= 0.5, and
there the material shares are renormalized to sum to 1 and dithered. With
`crisp=False` void is one more dithered channel (porous grading).

Memory: the field is resampled and dithered a few layers at a time, so the
printer-resolution volume is never held whole.
"""
from __future__ import annotations

import json
import os
import struct
import time
import zlib

import numpy as np

__all__ = ["export_voxel_slices", "PRINTERS", "STRATASYS_COLORS"]

_MM_PER_IN = 25.4

# dpi_x, dpi_y, layer_mm, max materials per job (guide: 6, High Speed 3)
PRINTERS = {
    "j750": dict(dpi_x=600, dpi_y=300, layer_mm=0.027, max_materials=6,
                 mode="High Mix"),
    "j750-hq": dict(dpi_x=600, dpi_y=300, layer_mm=0.014, max_materials=6,
                    mode="High Quality"),
    "j750-hs": dict(dpi_x=600, dpi_y=300, layer_mm=0.027, max_materials=3,
                    mode="High Speed"),
    "j850": dict(dpi_x=600, dpi_y=300, layer_mm=0.027, max_materials=6,
                 mode="High Mix"),
    "j850-hq": dict(dpi_x=600, dpi_y=300, layer_mm=0.014, max_materials=6,
                    mode="High Quality"),
    "j850-hs": dict(dpi_x=600, dpi_y=300, layer_mm=0.027, max_materials=3,
                    mode="High Speed"),
    # J5: resolution from the J5 manual; 5 materials is the J5 cabinet
    # (assumed as the per-job colour limit)
    "j5": dict(dpi_x=300, dpi_y=300, layer_mm=0.01875, max_materials=5,
               mode="J5"),
}
PRINTERS["j8"] = PRINTERS["j850"]

# Stratasys base-material RGB values from the Guide to Voxel Printing.
STRATASYS_COLORS = {
    "fullcure720": (255, 255, 170),
    "tango+": (255, 255, 100),
    "tangoblack+": (50, 50, 50),
    "agilus30clr": (255, 255, 100),
    "agilus30blk": (40, 77, 108),
    "agilus30wht": (255, 255, 255),
    "veroflexwt": (255, 255, 255),
    "veropurewht": (240, 240, 240),
    "veroblack": (26, 26, 29),
    "veroflexbk": (26, 26, 29),
    "veroclear": (227, 233, 253),
    "veroflexclr": (227, 233, 253),
    "verocyan": (0, 90, 158),
    "veroflexcy": (0, 90, 158),
    "veromgnt": (166, 33, 98),
    "veroflexmgt": (166, 33, 98),
    "veroyellow": (200, 189, 3),
    "veroflexyl": (200, 189, 3),
    "veroultraclr": (247, 247, 247),
    "rgd821": (247, 247, 247),
    "veromgt-v": (198, 0, 88),
    "veroyl-v": (240, 197, 0),
    "verocy-v": (0, 137, 166),
    "veroblue": (156, 208, 237),
    "verogrey": (185, 185, 185),
}
_ALIASES = {"agilus30": "agilus30blk", "vero": "veroblue",
            "verogray": "verogrey", "veromagenta": "veromgnt",
            "agilus30black": "agilus30blk", "agilus30white": "agilus30wht",
            "agilus30clear": "agilus30clr", "tangoplus": "tango+",
            "tangoblackplus": "tangoblack+", "verowhite": "veropurewht"}


# ---------------------------------------------------------------- inputs

def _printer_profile(printer):
    if isinstance(printer, dict):
        p = dict(printer)
        if "voxel_mm" in p:
            dx, dy, dz = p.pop("voxel_mm")
            p.setdefault("dpi_x", _MM_PER_IN / dx)
            p.setdefault("dpi_y", _MM_PER_IN / dy)
            p.setdefault("layer_mm", dz)
        for key in ("dpi_x", "dpi_y", "layer_mm"):
            if key not in p:
                raise ValueError(f"printer dict needs {key!r} (or voxel_mm)")
        p.setdefault("max_materials", 6)
        p.setdefault("mode", "custom")
        p.setdefault("name", "custom")
        return p
    key = str(printer).lower().replace(" ", "")
    if key not in PRINTERS:
        raise ValueError(f"unknown printer {printer!r}; known: "
                         f"{sorted(PRINTERS)} or a dict with dpi_x, dpi_y, "
                         "layer_mm")
    p = dict(PRINTERS[key])
    p["name"] = key
    return p


def _parse_color(c):
    if isinstance(c, str):
        s = c.lstrip("#")
        if len(s) != 6:
            raise ValueError(f"colour {c!r}: want '#rrggbb'")
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    c = tuple(int(round(v)) for v in c)
    if len(c) != 3 or not all(0 <= v <= 255 for v in c):
        raise ValueError(f"colour {c!r}: want 3 ints 0..255")
    return c


def _parse_materials(materials, k):
    out = []
    for m in materials:
        if isinstance(m, dict):
            name, color = m["name"], m.get("color")
        elif isinstance(m, str):
            name, color = m, None
        else:
            name, color = m
        if color is None:
            key = name.lower().replace(" ", "")
            key = _ALIASES.get(key, key)
            if key not in STRATASYS_COLORS:
                raise ValueError(f"no default colour for material {name!r}; "
                                 "give (name, color)")
            color = STRATASYS_COLORS[key]
        out.append({"name": str(name), "color": _parse_color(color)})
    if len(out) != k:
        raise ValueError(f"{len(out)} materials given for {k} fraction "
                         "channels")
    cols = [m["color"] for m in out]
    if len(set(cols)) != len(cols):
        raise ValueError(f"material colours must be distinct, got {cols}; "
                         "the utility maps colours to materials")
    return out


# ------------------------------------------------------------ resampling

def _axis_weights(n_out, step_out, pitch, n_in):
    """Linear interpolation of printer-voxel centres onto cell centres."""
    u = ((np.arange(n_out) + 0.5) * step_out) / pitch - 0.5
    u = np.clip(u, 0.0, n_in - 1)
    i0 = np.minimum(np.floor(u).astype(np.int64), max(n_in - 2, 0))
    i1 = np.minimum(i0 + 1, n_in - 1)
    w = (u - i0).astype(np.float32)
    return i0, i1, w


# -------------------------------------------------------------- dithering

def _bayer(n=8):
    m = np.zeros((1, 1), dtype=np.int64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return ((m + 0.5) / m.size).astype(np.float32)


def _blue_noise(n=64, seed=12345):
    """A blue-ish noise threshold tile: white noise, high-passed in Fourier
    space (periodic, so tiles seamlessly), then rank-mapped to uniform."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal((n, n)))
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    r = np.sqrt(kx ** 2 + ky ** 2)
    f *= 1.0 - np.exp(-(r / 0.18) ** 2)
    g = np.real(np.fft.ifft2(f)).ravel()
    t = np.empty(n * n, dtype=np.float32)
    t[np.argsort(g, kind="stable")] = (np.arange(n * n) + 0.5) / (n * n)
    return t.reshape(n, n)


def _threshold_dither(vals, tile, layer_ids):
    """vals (L, H, W, C) with channel 0 = void. Picks the channel whose
    cumulative interval holds the per-pixel threshold."""
    L, H, W, C = vals.shape
    ty, tx = tile.shape
    c = np.cumsum(vals, axis=-1)
    out = np.empty((L, H, W), dtype=np.uint8)
    for li, iz in enumerate(layer_ids):
        # step the tile offset through all ty*tx shifts in a scrambled order
        # (odd multiplier, coprime to the power-of-two tile size)
        m = (int(iz) * 2654435761) % (ty * tx)
        oy, ox = divmod(m, tx)
        tt = np.roll(tile, (oy, ox), axis=(0, 1))
        thr = np.tile(tt, (H // ty + 1, W // tx + 1))[:H, :W]
        idx = (thr[..., None] >= c[li]).sum(axis=-1)
        out[li] = np.minimum(idx, C - 1)
    return out


def _fs_dither(vals, void_mask, jitter=0.0, seed=0):
    """Multi-channel Floyd-Steinberg on L layers at once, wavefront order.
    vals (L, H, W, C) float32; void_mask (L, H, W) bool or None (pixels
    forced to channel 0, whose error is dropped). `jitter` perturbs only the
    choice (the error is still taken against the true value, so means are
    kept); it breaks up FS's regular textures so they don't repeat layer
    after layer."""
    L, H, W, C = vals.shape
    noise = None
    if jitter > 0:
        rng = np.random.default_rng(seed)
        noise = (rng.random((L, H, W, C), dtype=np.float32) - 0.5) * jitter
    buf = np.zeros((L, H + 1, W + 2, C), dtype=np.float32)
    buf[:, :H, 1:W + 1] = vals
    out = np.empty((L, H, W), dtype=np.uint8)
    ys_all = np.arange(H)
    li = np.arange(L)[:, None]
    for t in range(W + 2 * (H - 1)):
        y_lo = max(0, -(-(t - (W - 1)) // 2))
        y_hi = min(H - 1, t // 2)
        ys = ys_all[y_lo:y_hi + 1]
        xs = t - 2 * ys
        v = buf[:, ys, xs + 1]                          # (L, n, C)
        if noise is None:
            q = v.argmax(axis=-1)                       # (L, n)
        else:
            q = (v + noise[:, ys, xs]).argmax(axis=-1)
        if void_mask is not None:
            m = void_mask[:, ys, xs]
            q[m] = 0
        out[:, ys, xs] = q
        v[li, np.arange(len(ys))[None, :], q] -= 1.0   # v is now the error
        if void_mask is not None:
            v[m] = 0.0
        buf[:, ys, xs + 2] += v * (7.0 / 16.0)
        buf[:, ys + 1, xs] += v * (3.0 / 16.0)
        buf[:, ys + 1, xs + 1] += v * (5.0 / 16.0)
        buf[:, ys + 1, xs + 2] += v * (1.0 / 16.0)
    return out


# ---------------------------------------------------------------- PNG out

def _png_bytes(img):
    """Minimal 8-bit RGB/RGBA PNG writer (no Pillow)."""
    h, w, ch = img.shape
    ctype = {3: 2, 4: 6}[ch]
    raw = np.empty((h, 1 + w * ch), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = img.reshape(h, w * ch)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 3))
            + chunk(b"IEND", b""))


def _png_writer(dpi):
    try:
        from PIL import Image
    except ImportError:
        Image = None

    def write(path, img):
        if Image is not None:
            mode = "RGBA" if img.shape[2] == 4 else "RGB"
            Image.fromarray(img, mode).save(path, compress_level=3,
                                            dpi=dpi)
        else:
            with open(path, "wb") as f:
                f.write(_png_bytes(img))
    return write, ("pillow" if Image is not None else "zlib")


# ------------------------------------------------------------------ main

def export_voxel_slices(out_dir, fraction, origin, pitch, materials,
                        printer="j850", dither="floyd-steinberg", solid=None,
                        max_layers=None, *, prefix="slice_", crisp=True,
                        image_mode="RGBA", background=(255, 255, 255),
                        flip_y=True, jitter=0.25, batch_mb=96, verbose=False) -> dict:
    """Write a GrabCAD Voxel Print PNG slice stack for a material field.

    out_dir: folder for the PNGs and manifest.json (created).
    fraction: (nx, ny, nz, k) per-cell volume fractions of k materials
        (sum over k <= 1; the rest is void), or (nx, ny, nz) for a two-
        material mix (0 = materials[0], 1 = materials[1]) with `solid` the
        occupancy. Cell (0, 0, 0)'s min corner is at `origin` (mm); cells
        are `pitch` mm cubes (a scalar or (px, py, pz)).
    materials: k entries, each a name (colour from STRATASYS_COLORS, e.g.
        "VeroCyan", "Agilus30Blk"), (name, color) or {"name", "color"};
        colours are RGB tuples or "#rrggbb" and must be distinct.
    printer: a PRINTERS key ("j850" = 600 x 300 DPI, 0.027 mm High Mix;
        "j850-hq" 0.014 mm; "j750*"; "j5") or a dict with dpi_x, dpi_y,
        layer_mm (or voxel_mm=(dx, dy, dz)) and optional max_materials.
    dither: "floyd-steinberg", "ordered" (Bayer 8x8) or "blue-noise".
    solid: (nx, ny, nz) bool/float occupancy, multiplies the fractions;
        required meaning for the 2-material form (default: all solid).
    max_layers: write only the first this-many layers (previews, tests).
    crisp: threshold the outline at occupancy 0.5 instead of dithering void.
    image_mode: "RGBA" (void transparent) or "RGB" (void = `background`).
    flip_y: image row 0 is the +Y edge (seen from above).
    jitter: Floyd-Steinberg decision noise amplitude (0 = plain FS).

    Returns a summary dict (also written as manifest.json) with the printer
    grid, files, and per-material requested vs achieved share/volume.
    """
    t_start = time.time()
    prof = _printer_profile(printer)
    f = np.asarray(fraction, dtype=np.float32)
    if f.ndim == 3:
        f = np.stack([1.0 - f, f], axis=-1)
        if solid is None:
            solid = np.ones(f.shape[:3], dtype=np.float32)
    elif f.ndim != 4:
        raise ValueError("fraction must be (nx, ny, nz, k) or (nx, ny, nz)")
    f = np.clip(f, 0.0, 1.0)
    if solid is not None:
        s = np.asarray(solid, dtype=np.float32)
        if s.shape != f.shape[:3]:
            raise ValueError(f"solid shape {s.shape} != grid {f.shape[:3]}")
        f = f * np.clip(s, 0.0, 1.0)[..., None]
    tot = f.sum(axis=-1, keepdims=True)
    f = np.where(tot > 1.0, f / np.maximum(tot, 1e-12), f)
    nx, ny, nz, k = f.shape
    if k > prof["max_materials"]:
        raise ValueError(f"{k} materials; {prof['name']} takes at most "
                         f"{prof['max_materials']} per job")
    mats = _parse_materials(materials, k)
    if dither not in ("floyd-steinberg", "ordered", "blue-noise"):
        raise ValueError(f"unknown dither {dither!r}")
    if image_mode not in ("RGBA", "RGB"):
        raise ValueError("image_mode must be 'RGBA' or 'RGB'")
    bg = _parse_color(background)
    if image_mode == "RGB" and bg in [m["color"] for m in mats]:
        raise ValueError("background colour equals a material colour")

    pitch3 = np.broadcast_to(np.asarray(pitch, dtype=float), (3,))
    origin = np.asarray(origin, dtype=float).reshape(3)
    size = np.array([nx, ny, nz]) * pitch3
    dx = _MM_PER_IN / prof["dpi_x"]
    dy = _MM_PER_IN / prof["dpi_y"]
    dz = float(prof["layer_mm"])
    W = max(1, int(round(size[0] / dx)))
    H = max(1, int(round(size[1] / dy)))
    NZ = max(1, int(round(size[2] / dz)))
    n_write = NZ if max_layers is None else min(NZ, int(max_layers))

    # channels: 0 = void, 1..k = materials
    full = np.concatenate([1.0 - f.sum(axis=-1, keepdims=True), f], axis=-1)
    full = np.ascontiguousarray(full.astype(np.float32))
    C = k + 1
    xi0, xi1, xw = _axis_weights(W, dx, pitch3[0], nx)
    yi0, yi1, yw = _axis_weights(H, dy, pitch3[1], ny)
    zi0, zi1, zw = _axis_weights(NZ, dz, pitch3[2], nz)
    if flip_y:   # row 0 = +Y
        yi0, yi1, yw = yi0[::-1], yi1[::-1], yw[::-1]

    lut = np.zeros((C, 4), dtype=np.uint8)
    lut[0] = (0, 0, 0, 0) if image_mode == "RGBA" else (*bg, 255)
    for j, m in enumerate(mats):
        lut[j + 1] = (*m["color"], 255)
    nch = 4 if image_mode == "RGBA" else 3
    write, png_backend = _png_writer((prof["dpi_x"], prof["dpi_y"]))

    os.makedirs(out_dir, exist_ok=True)
    digits = max(4, len(str(NZ - 1)))
    names = [f"{prefix}{i:0{digits}d}.png" for i in range(n_write)]

    tile = (_bayer(8) if dither == "ordered"
            else _blue_noise() if dither == "blue-noise" else None)
    per_layer_bytes = H * W * C * 4 * (4 if tile is None else 4)
    batch = int(max(1, min(32, batch_mb * 2 ** 20 // max(per_layer_bytes, 1))))

    counts = np.zeros(C, dtype=np.int64)
    req_raw = np.zeros(C, dtype=np.float64)      # resampled fractions
    req_target = np.zeros(C, dtype=np.float64)   # what the dither aimed at
    t_dither = 0.0
    for z0 in range(0, n_write, batch):
        ids = np.arange(z0, min(n_write, z0 + batch))
        # z, then x, then y interpolation (separable trilinear)
        a = full[:, :, zi0[ids]] * (1 - zw[ids])[None, None, :, None] \
            + full[:, :, zi1[ids]] * zw[ids][None, None, :, None]
        a = np.moveaxis(a, 2, 0)                                # (L,nx,ny,C)
        a = a[:, xi0] * (1 - xw)[None, :, None, None] \
            + a[:, xi1] * xw[None, :, None, None]               # (L,W,ny,C)
        a = a[:, :, yi0] * (1 - yw)[None, None, :, None] \
            + a[:, :, yi1] * yw[None, None, :, None]            # (L,W,H,C)
        vals = np.ascontiguousarray(np.swapaxes(a, 1, 2))       # (L,H,W,C)
        del a
        req_raw += vals.sum(axis=2).sum(axis=(0, 1), dtype=np.float64)
        if crisp:
            occ = 1.0 - vals[..., 0]
            void_mask = occ < 0.5
            vals[..., 1:] /= np.maximum(occ, 1e-6)[..., None]
            vals[..., 0] = 0.0
            vals[void_mask] = 0.0
            vals[void_mask, 0] = 1.0
        else:
            void_mask = None
        req_target += vals.sum(axis=2).sum(axis=(0, 1), dtype=np.float64)

        t0 = time.time()
        if tile is None:
            # alternate scan orientation layer to layer (flip x, flip y)
            fl = [(iz % 2 == 1, (iz // 2) % 2 == 1) for iz in ids]
            for li, (fx, fy) in enumerate(fl):
                if fx:
                    vals[li] = vals[li, :, ::-1].copy()
                if fy:
                    vals[li] = vals[li, ::-1].copy()
            vm = None if void_mask is None else void_mask.copy()
            if vm is not None:
                for li, (fx, fy) in enumerate(fl):
                    if fx:
                        vm[li] = vm[li, :, ::-1].copy()
                    if fy:
                        vm[li] = vm[li, ::-1].copy()
            idx = _fs_dither(vals, vm, jitter, seed=int(ids[0]))
            for li, (fx, fy) in enumerate(fl):
                if fx:
                    idx[li] = idx[li, :, ::-1].copy()
                if fy:
                    idx[li] = idx[li, ::-1].copy()
        else:
            idx = _threshold_dither(vals, tile, ids)
            if void_mask is not None:
                idx[void_mask] = 0
        t_dither += time.time() - t0
        del vals
        counts += np.bincount(idx.ravel(), minlength=C)
        for li, iz in enumerate(ids):
            write(os.path.join(out_dir, names[iz]), lut[idx[li]][..., :nch])
        if verbose:
            print(f"voxelprint: layers {ids[0]}..{ids[-1]} of {n_write}")

    vox_mm3 = dx * dy * dz
    n_solid = counts[1:].sum()
    tgt_solid = req_target[1:].sum()
    per_mat = []
    for j, m in enumerate(mats):
        req_share = float(req_target[j + 1] / tgt_solid) if tgt_solid else 0.0
        ach_share = float(counts[j + 1] / n_solid) if n_solid else 0.0
        per_mat.append({
            "name": m["name"],
            "color": "#%02x%02x%02x" % m["color"],
            "rgb": list(m["color"]),
            "png_index": j + 1,
            "requested_share": req_share,
            "achieved_share": ach_share,
            "share_error": ach_share - req_share,
            "requested_volume_mm3": float(req_raw[j + 1] * vox_mm3),
            "achieved_volume_mm3": float(counts[j + 1] * vox_mm3),
            "input_volume_mm3": float(f[..., j].sum() * np.prod(pitch3)),
        })
    elapsed = time.time() - t_start
    summary = {
        "format": "GrabCAD Voxel Print PNG slice stack",
        "printer": prof["name"],
        "build_mode": prof.get("mode"),
        "dpi": [prof["dpi_x"], prof["dpi_y"]],
        "voxel_mm": [dx, dy, dz],
        "layer_thickness_mm": dz,
        "width_px": W,
        "height_px": H,
        "layers_total": NZ,
        "layers_written": n_write,
        "truncated": n_write < NZ,
        "size_mm": [W * dx, H * dy, NZ * dz],
        "requested_size_mm": size.tolist(),
        "origin_mm": origin.tolist(),
        "file_prefix": prefix,
        "files": names,
        "image_mode": image_mode,
        "void_color": ("transparent (0,0,0,0)" if image_mode == "RGBA"
                       else "#%02x%02x%02x" % bg),
        "orientation": ("columns +X, row 0 = " + ("+Y" if flip_y else "-Y")
                        + " edge, layer 0 = bottom (min z)"),
        "dither": dither,
        "crisp_outline": bool(crisp),
        "materials": per_mat,
        "void_voxels": int(counts[0]),
        "solid_voxels": int(n_solid),
        "max_share_error": max((abs(p["share_error"]) for p in per_mat),
                               default=0.0),
        "warnings": ([] if NZ >= 30 else
                     [f"{NZ} slices; the Voxel Print Utility wants >= 30"]),
        "png_backend": png_backend,
        "seconds": elapsed,
        "seconds_dither": t_dither,
        "seconds_per_layer": elapsed / max(n_write, 1),
        "utility_steps": ("GrabCAD Print > Apps > Voxel Print Utility: slice "
                          f"prefix '{prefix}', select this folder, check "
                          f"slice range 0..{n_write - 1}, layer thickness "
                          f"{dz} mm and background colour, then map each "
                          "colour to its material."),
    }
    manifest = {kk: v for kk, v in summary.items() if kk != "files"}
    manifest["file_pattern"] = f"{prefix}{'#' * digits}.png"
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    summary["out_dir"] = os.path.abspath(out_dir)
    return summary
