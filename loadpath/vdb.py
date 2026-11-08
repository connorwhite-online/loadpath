"""
OpenVDB grids in and out: the sparse voxel format that Houdini, Blender,
nTop, OpenVDB tools and most film and simulation software read, written and
read here in pure Python (numpy and zlib, no OpenVDB library needed).

    from loadpath.vdb import write_vdb, read_vdb
    write_vdb("part.vdb", {"name": "sdf", "values": d, "origin": o, "pitch": 0.5,
                           "class": "level set"})
    grids = read_vdb("part.vdb")      # [{"name", "values", "origin", "pitch", ...}]

`loadpath sdf part.stl --out part.vdb` writes a solid's narrow-band level set,
a study shape {"type": "sdf", "path": "part.vdb"} reads one back, and the
field graph's `grid` op samples any float grid in a .vdb (a density or
temperature volume from another tool) the way it samples an .npz.

Scope: float grids with the standard 5-4-3 tree (Tree_float_5_4_3, also
saved as half float), file format 222-224 (OpenVDB 3 to 11). Written
uncompressed or zip; read uncompressed, zip and active-mask compressed,
and blosc when the `blosc` Python package is installed (OpenVDB's default
when the library has it; otherwise a clear error says to resave with zip).
Values are a dense array over the grid's stored nodes, so a sparse grid
covering a huge index space is refused past `max_voxels`.
"""
from __future__ import annotations

import io
import struct
import uuid
import zlib

import numpy as np

__all__ = ["read_vdb", "write_vdb"]

MAGIC = 0x56444220
VERSION = 224
TREE = "Tree_float_5_4_3"
NONE, ZIP, ACTIVE_MASK, BLOSC = 0, 1, 2, 4
# writeCompressedValues metadata codes
NO_MASK_OR_INACTIVE_VALS, NO_MASK_AND_MINUS_BG, NO_MASK_AND_ONE_INACTIVE_VAL = 0, 1, 2
MASK_AND_NO_INACTIVE_VALS, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS = 3, 4, 5
NO_MASK_AND_ALL_VALS = 6
LOG2 = (3, 4, 5)            # leaf, internal 1, internal 2 (child sizes 8, 128, 4096)


# --------------------------------------------------------------- writing

def _s(b, text):
    t = text.encode()
    b.write(struct.pack("<I", len(t)) + t)


def _mask(bits):
    return np.packbits(np.asarray(bits, bool).ravel(), bitorder="little").tobytes()


def _values(b, vals, compression):
    b.write(struct.pack("<b", NO_MASK_AND_ALL_VALS))
    raw = np.ascontiguousarray(vals, "<f4").tobytes()
    if compression & ZIP:
        z = zlib.compress(raw)
        if len(z) < len(raw):
            b.write(struct.pack("<q", len(z)) + z)
        else:
            b.write(struct.pack("<q", -len(raw)) + raw)
    else:
        b.write(raw)


def _level_set(g):
    pitch = float(g["pitch"])
    bg = float(g.get("background", float(g.get("halfWidth", 3.0)) * pitch))
    v = np.clip(np.asarray(g["values"], float), -bg, bg)
    active = np.abs(v) < bg if g.get("active") is None else np.asarray(g["active"], bool)
    return v, active, bg


def _blocks(x, n, fill):
    """x (a, b, c) padded with `fill` to multiples of n and split into
    (a/n, b/n, c/n, n, n, n) blocks."""
    x = np.pad(x, [(0, (-s) % n) for s in x.shape], constant_values=fill)
    L = np.array(x.shape) // n
    return x.reshape(L[0], n, L[1], n, L[2], n).transpose(0, 2, 4, 1, 3, 5)


def _grid_bytes(g, compression):
    cls = g.get("class", "level set")
    pitch = float(g["pitch"])
    origin = np.asarray(g.get("origin", (0.0, 0.0, 0.0)), float).reshape(3)
    if cls == "level set":
        values, active, bg = _level_set(g)
    else:
        values = np.asarray(g["values"], float)
        bg = float(g.get("background", 0.0))
        active = values != bg if g.get("active") is None else np.asarray(g["active"], bool)
    if values.ndim != 3:
        raise ValueError("a VDB grid's values must be a 3D array")
    # the 5-4-3 tree of the dense array, index (0, 0, 0) at values[0, 0, 0]:
    # leaves where any voxel is active; elsewhere inactive tiles holding the
    # block's median (so a level set's inside reads -background)
    V = np.pad(values, [(0, (-s) % 8) for s in values.shape], mode="edge").astype(np.float32)
    Vb = _blocks(V, 8, 0)
    Ab = _blocks(np.asarray(active, bool), 8, False)
    has0 = Ab.any(axis=(3, 4, 5))
    tile0 = np.median(Vb.reshape(*has0.shape, 512), axis=-1).astype(np.float32)
    has1b = _blocks(has0, 16, False)
    has1 = has1b.any(axis=(3, 4, 5))
    tile0b = _blocks(tile0, 16, bg)
    tile1 = np.median(tile0b.reshape(*has1.shape, 4096), axis=-1).astype(np.float32)
    has2b = _blocks(has1, 32, False)
    tile1b = _blocks(tile1, 32, bg)

    b = io.BytesIO()
    b.write(struct.pack("<I", compression))
    meta = [("class", "string", cls.encode()), ("name", "string", g.get("name", "density").encode()),
            ("file_compression", "string", (b"zip" if compression & ZIP else b"none")),
            ("is_saved_as_half_float", "bool", b"\x00")]
    b.write(struct.pack("<i", len(meta)))
    for k, t, v in meta:
        _s(b, k)
        _s(b, t)
        b.write(struct.pack("<i", len(v)) + v)
    _s(b, "UniformScaleTranslateMap")
    s = np.full(3, pitch)
    b.write(np.concatenate([origin, s, s, 1 / s, 1 / s ** 2, 0.5 / s]).astype("<f8").tobytes())

    b.write(struct.pack("<i", 1))
    b.write(struct.pack("<f", bg))
    roots = [r for r in np.ndindex(*has2b.shape[:3]) if has2b[r].any()]
    b.write(struct.pack("<II", 0, len(roots)))
    zero2, zero1 = _mask(np.zeros(32 ** 3, bool)), _mask(np.zeros(16 ** 3, bool))
    leaves = []
    for r in roots:
        b.write(struct.pack("<3i", *(int(k) * 4096 for k in r)))
        h2 = has2b[r]
        b.write(_mask(h2) + zero2)
        _values(b, np.where(h2, 0, tile1b[r]).ravel(), compression)
        for i2 in zip(*np.nonzero(h2)):
            c1 = tuple(np.array(r) * 32 + i2)
            h1 = has1b[c1]
            b.write(_mask(h1) + zero1)
            _values(b, np.where(h1, 0, tile0b[c1]).ravel(), compression)
            for i1 in zip(*np.nonzero(h1)):
                c0 = tuple(np.array(c1) * 16 + i1)
                m = Ab[c0]
                b.write(_mask(m))
                leaves.append((m, Vb[c0]))
    block = b.tell()
    for m, v in leaves:
        b.write(_mask(m))
        _values(b, v.ravel(), compression)
    return b.getvalue(), block


def write_vdb(path, grids, compression="zip"):
    """Write float grids to an OpenVDB file. `grids` is one dict or a list:
    `values` (3D array, index [i, j, k] at world origin + pitch * (i, j, k)),
    `pitch` (mm), `origin` (mm, default 0), `name`, `class` ("level set",
    the default, or "fog volume"/None). A level set's `background` is
    `halfWidth` (default 3) voxels and only |d| < background is active; a fog
    volume's background is 0 and its non-zero voxels are active.
    `compression` "zip" or "none". Returns {"path", "grids", "bytes"}."""
    if isinstance(grids, dict):
        grids = [grids]
    comp = {"zip": ZIP, "none": NONE, None: NONE}.get(compression)
    if comp is None:
        raise ValueError("vdb compression must be 'zip' or 'none'")
    out = io.BytesIO()
    out.write(struct.pack("<qIII", MAGIC, VERSION, 10, 0))
    out.write(b"\x01" + str(uuid.uuid4()).encode())
    out.write(struct.pack("<i", 0))                  # file metadata
    out.write(struct.pack("<i", len(grids)))
    for g in grids:
        _s(out, g.get("name", "density"))
        _s(out, TREE)
        _s(out, "")
        body, block = _grid_bytes(g, comp)
        start = out.tell() + 24
        out.write(struct.pack("<qqq", start, start + block, start + len(body)))
        out.write(body)
    data = out.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return {"path": path, "grids": [g.get("name", "density") for g in grids], "bytes": len(data)}


# --------------------------------------------------------------- reading

class _R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        if self.p + n > len(self.d):
            raise ValueError("the .vdb file is truncated")
        out = self.d[self.p:self.p + n]
        self.p += n
        return out

    def u(self, fmt):
        v = struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))
        return v[0] if len(v) == 1 else v

    def s(self):
        return self.take(self.u("I")).decode("utf-8", "replace")

    def mask(self, n):
        return np.unpackbits(np.frombuffer(self.take(n // 8), np.uint8), bitorder="little").astype(bool)


def _blosc(raw, n):
    try:
        import blosc
    except ImportError:
        raise ValueError("this .vdb is blosc compressed; install the 'blosc' package or resave it with zip "
                         "(OpenVDB: grid.setSaveFloatAsHalf / io::COMPRESS_ZIP)") from None
    return blosc.decompress(raw)


def _read_values(r, count, comp, vmask, bg, half):
    meta = r.u("b")
    v0 = bg if meta == NO_MASK_OR_INACTIVE_VALS else -bg
    v1 = bg
    width = 2 if half else 4
    if meta in (NO_MASK_AND_ONE_INACTIVE_VAL, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS):
        v0 = r.u("f")                               # full floats even in a half grid
        if meta == MASK_AND_TWO_INACTIVE_VALS:
            v1 = r.u("f")
    sel = None
    if meta in (MASK_AND_NO_INACTIVE_VALS, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS):
        sel = r.mask(count)
    n = int(vmask.sum()) if (comp & ACTIVE_MASK and meta != NO_MASK_AND_ALL_VALS) else count
    nbytes = n * width
    if half and n == 0:
        raw = b""                                   # OpenVDB's half writer skips empty buffers
    elif comp & (ZIP | BLOSC):
        size = r.u("q")
        if size <= 0:
            raw = r.take(-size)
        elif comp & BLOSC:
            raw = _blosc(r.take(size), nbytes)
        else:
            raw = zlib.decompress(r.take(size))
    else:
        raw = r.take(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"vdb node values: expected {nbytes} bytes, got {len(raw)} (meta {meta}, flags {comp})")
    vals = np.frombuffer(raw, "<f2" if half else "<f4").astype(np.float32)
    if n == count:
        return vals
    out = np.full(count, v0, np.float32)
    if sel is not None:
        out[sel] = v1
    out[vmask] = vals
    return out


def _transform(r):
    kind = r.s()
    d = lambda k: np.frombuffer(r.take(8 * k), "<f8")
    if kind in ("UniformScaleTranslateMap", "ScaleTranslateMap"):
        x = d(18)
        t, s = x[0:3], x[3:6]
    elif kind in ("UniformScaleMap", "ScaleMap"):
        x = d(15)
        t, s = np.zeros(3), x[0:3]
    elif kind == "TranslationMap":
        t, s = d(3), np.ones(3)
    else:
        raise ValueError(f"vdb transform {kind!r} is not supported (only axis-aligned scale and translate)")
    if not np.allclose(s, s[0], rtol=1e-6):
        raise ValueError(f"vdb voxels must be cubes, this grid's are {s.tolist()}")
    return t, float(s[0])


def _read_grid(r, version, max_voxels):
    name, typ = r.s(), r.s()
    r.s()                                              # instance parent
    half = typ.endswith("_HalfFloat")
    if typ.removesuffix("_HalfFloat") != TREE:
        raise ValueError(f"vdb grid {name!r} is a {typ}; only float grids ({TREE}) are read")
    gpos, bpos, epos = r.u("qqq")
    r.p = gpos
    comp = r.u("I")
    meta = {}
    for _ in range(r.u("i")):
        k, t = r.s(), r.s()
        v = r.take(r.u("i"))
        if t == "string":
            meta[k] = v.decode("utf-8", "replace")
        elif t == "bool":
            meta[k] = bool(v[0])
    half = half or meta.get("is_saved_as_half_float", False)
    origin, pitch = _transform(r)
    if r.u("i") != 1:
        raise ValueError("vdb trees with several buffers are not supported")
    bg = r.u("f")
    ntiles, nkids = r.u("II")
    tiles = []                                         # (lo, size, value)
    for _ in range(ntiles):
        c = r.u("3i")
        v = r.u("f")
        r.take(1)
        tiles.append((np.array(c), 4096, v))
    leaves = []

    def internal(lo, level):
        log = LOG2[level]
        csize = 1 << sum(LOG2[:level])
        n = 1 << (3 * log)
        cm, vm = r.mask(n), r.mask(n)
        vals = _read_values(r, n, comp, vm, bg, half)
        idx = np.stack(np.unravel_index(np.arange(n), (1 << log,) * 3), 1) * csize + lo
        for i in np.nonzero(~cm)[0]:
            tiles.append((idx[i], csize, float(vals[i])))
        for i in np.nonzero(cm)[0]:
            if level == 1:
                leaves.append((idx[i], r.mask(512)))
            else:
                internal(idx[i], level - 1)

    for _ in range(nkids):
        internal(np.array(r.u("3i")), 2)
    r.p = bpos
    data = []
    for lo, _ in leaves:
        m = r.mask(512)
        data.append((lo, _read_values(r, 512, comp, m, bg, half).reshape(8, 8, 8), m.reshape(8, 8, 8)))
    r.p = epos

    if data:
        lo = np.min([d[0] for d in data], 0)
        hi = np.max([d[0] for d in data], 0) + 8
    else:
        act = [t for t in tiles if t[2] != bg]
        if not act:
            lo, hi = np.zeros(3, int), np.ones(3, int)
        else:
            lo = np.min([t[0] for t in act], 0)
            hi = np.max([t[0] + t[1] for t in act], 0)
    shape = hi - lo
    if int(np.prod(shape)) > max_voxels:
        raise ValueError(f"vdb grid {name!r} spans {shape.tolist()} voxels, past max_voxels {max_voxels}")
    vol = np.full(shape, bg, np.float32)
    act = np.zeros(shape, bool)
    for t0, size, v in sorted(tiles, key=lambda t: -t[1]):
        a = np.maximum(t0 - lo, 0)
        b_ = np.minimum(t0 + size - lo, shape)
        if np.all(b_ > a):
            vol[a[0]:b_[0], a[1]:b_[1], a[2]:b_[2]] = v
    for c, v, m in data:
        a = c - lo
        vol[a[0]:a[0] + 8, a[1]:a[1] + 8, a[2]:a[2] + 8] = v
        act[a[0]:a[0] + 8, a[1]:a[1] + 8, a[2]:a[2] + 8] = m
    return {"name": name, "class": meta.get("class", "unknown"), "values": vol, "active": act,
            "origin": origin + lo * pitch, "pitch": pitch, "background": float(bg),
            "index": lo.tolist(), "leaves": len(data)}


def read_vdb(path, name=None, max_voxels=400_000_000):
    """Read the float grids of an OpenVDB file as dense arrays: a list of
    {"name", "class", "values" (float32, [i, j, k] at origin + pitch * (i,
    j, k)), "active", "origin", "pitch", "background"} over each grid's
    stored nodes. `name` picks one grid and returns it alone."""
    with open(path, "rb") as f:
        r = _R(f.read())
    if r.u("q") != MAGIC:
        raise ValueError(f"{path!r} is not an OpenVDB file")
    version = r.u("I")
    r.u("II")
    if version < 222:
        raise ValueError(f"vdb file format {version} is older than OpenVDB 3 (222); resave it")
    has_offsets = r.take(1)[0]
    r.take(36)
    for _ in range(r.u("i")):
        r.s()
        r.s()
        r.take(r.u("i"))
    if not has_offsets:
        raise ValueError("vdb files without grid offsets (streamed) are not supported")
    out = []
    for _ in range(r.u("i")):
        g = _read_grid(r, version, max_voxels)
        out.append(g)
    if name is not None:
        for g in out:
            if g["name"] == name:
                return g
        raise ValueError(f"no grid {name!r} in {path!r} (have {', '.join(g['name'] for g in out)})")
    return out
