"""
Readers for scans: point clouds, Gaussian splats and scanned meshes, as
plain arrays (scan.py turns them into a measured surface).

Formats:

- PLY: point clouds (any vertex properties: normals, colours, confidence),
  3D Gaussian splatting scenes (x, y, z, scale_0..2, rot_0..3, opacity,
  f_dc_*: the INRIA/graphdeco layout every splat trainer writes), and meshes
  (vertices and faces: Polycam, Scaniverse, an ARKit mesh);
- .splat: the 32-byte-per-splat web format (position, linear scale, RGBA,
  rotation as bytes);
- PCD (PCL): ascii, binary and binary_compressed (LZF), with its VIEWPOINT
  (the sensor's position, which orients the normals);
- LAS 1.0-1.4 (terrestrial and mobile LiDAR), uncompressed; LAZ needs
  `laspy` with a LAZ backend;
- E57 (terrestrial scanners' exchange format) needs `pye57`;
- XYZ / TXT / CSV / PTS text columns (x y z [nx ny nz | r g b | intensity]);
- STL / OBJ / GLB / OFF meshes through trimesh.

Every reader returns a dict: points (N, 3) float64 in the file's units,
optional normals, colors, splat (scales (N, 3) in the file's units,
quaternions (N, 4) w x y z, opacity (N,) 0..1), viewpoint (3,) when the
file records the sensor's position, faces for meshes, and "unitsHint" when
the format has a units convention (LAS, E57, PCL: metres).
"""
from __future__ import annotations

import os
import re

import numpy as np

__all__ = ["read_scan", "SCAN_EXT"]

SCAN_EXT = (".ply", ".splat", ".pcd", ".las", ".laz", ".e57", ".xyz", ".txt", ".csv", ".pts",
            ".stl", ".obj", ".glb", ".gltf", ".off")

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


# ---------------------------------------------------------------- PLY

def _ply_header(fh):
    first = fh.readline()
    if not first.startswith(b"ply"):
        raise ValueError("not a PLY file")
    fmt, elements = None, []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("PLY header has no end_header")
        t = line.decode("ascii", "replace").strip().split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            elements.append({"name": t[1], "count": int(t[2]), "props": []})
        elif t[0] == "property":
            if t[1] == "list":
                elements[-1]["props"].append((t[4], "list", _PLY_TYPES[t[2]], _PLY_TYPES[t[3]]))
            else:
                elements[-1]["props"].append((t[2], _PLY_TYPES[t[1]]))
        elif t[0] == "end_header":
            break
    return fmt, elements


def _read_ply(path):
    with open(path, "rb") as fh:
        fmt, elements = _ply_header(fh)
        names = [e["name"] for e in elements]
        if "vertex" not in names:
            raise ValueError(f"{path}: PLY has no vertex element")
        has_faces = any(e["name"] == "face" and e["count"] > 0 for e in elements)
        if has_faces:
            return _read_mesh(path)
        if fmt == "ascii":
            data = None
            for e in elements:
                if any(len(p) > 2 for p in e["props"]):
                    raise ValueError(f"{path}: list properties before the vertices in an ascii PLY")
                rows = np.loadtxt(fh, max_rows=e["count"], ndmin=2) if e["count"] else np.zeros((0, len(e["props"])))
                if e["name"] == "vertex":
                    data = {p[0]: rows[:, i] for i, p in enumerate(e["props"])}
                    break
        else:
            end = "<" if fmt == "binary_little_endian" else ">"
            data = None
            for e in elements:
                if any(len(p) > 2 for p in e["props"]):
                    raise ValueError(f"{path}: list properties before the vertices")
                dt = np.dtype([(p[0], end + p[1]) for p in e["props"]])
                arr = np.frombuffer(fh.read(dt.itemsize * e["count"]), dtype=dt, count=e["count"])
                if e["name"] == "vertex":
                    data = {k: arr[k].astype(np.float64) for k in arr.dtype.names}
                    break
    return _from_columns(data)


def _from_columns(c):
    out = {"points": np.stack([c["x"], c["y"], c["z"]], 1).astype(np.float64)}
    if all(k in c for k in ("nx", "ny", "nz")):
        n = np.stack([c["nx"], c["ny"], c["nz"]], 1)
        ln = np.linalg.norm(n, axis=1)
        if np.median(ln) > 0.5:          # splat trainers write zero normals
            out["normals"] = n / np.maximum(ln, 1e-12)[:, None]
    for r, g, b in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if all(k in c for k in (r, g, b)):
            col = np.stack([c[r], c[g], c[b]], 1)
            out["colors"] = (col / (255.0 if col.max() > 1.0 else 1.0)).astype(np.float32)
            break
    if all(f"scale_{i}" in c for i in range(3)) and all(f"rot_{i}" in c for i in range(4)):
        # 3D Gaussian splatting: log scales, logit opacity, (w, x, y, z) quaternion
        s = np.exp(np.stack([c[f"scale_{i}"] for i in range(3)], 1))
        q = np.stack([c[f"rot_{i}"] for i in range(4)], 1)
        op = 1.0 / (1.0 + np.exp(-np.asarray(c.get("opacity", np.full(len(s), 10.0)))))
        out["splat"] = {"scales": s, "quats": q, "opacity": op}
        if all(f"f_dc_{i}" in c for i in range(3)):
            sh0 = 0.28209479177387814
            out["colors"] = np.clip(0.5 + sh0 * np.stack([c[f"f_dc_{i}"] for i in range(3)], 1), 0, 1).astype(np.float32)
    # the sensor's position for each point (the Imprint format's sx sy sz):
    # the ray each point was measured along, which says what space it saw empty
    if all(k in c for k in ("sx", "sy", "sz")):
        out["viewpoints"] = np.stack([c["sx"], c["sy"], c["sz"]], 1).astype(np.float64)
    for k in ("confidence", "intensity", "scalar_confidence"):
        if k in c:
            out[k.replace("scalar_", "")] = np.asarray(c[k], np.float64)
    return out


# ---------------------------------------------------------------- .splat

def _read_splat(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 32:
        raise ValueError(f"{path}: a .splat file is 32-byte records; this one is {raw.size} bytes")
    rec = raw.reshape(-1, 32)
    pos = rec[:, 0:12].copy().view("<f4").reshape(-1, 3).astype(np.float64)
    sc = rec[:, 12:24].copy().view("<f4").reshape(-1, 3).astype(np.float64)
    rgba = rec[:, 24:28].astype(np.float32) / 255.0
    q = (rec[:, 28:32].astype(np.float64) - 128.0) / 128.0
    return {"points": pos, "colors": rgba[:, :3],
            "splat": {"scales": sc, "quats": q, "opacity": rgba[:, 3].astype(np.float64)}}


# ---------------------------------------------------------------- PCD

def _lzf_decompress_py(src: bytes, n_out: int) -> bytes:
    out = bytearray(n_out)
    i = o = 0
    n = len(src)
    while i < n:
        ctrl = src[i]
        i += 1
        if ctrl < 32:
            ln = ctrl + 1
            out[o:o + ln] = src[i:i + ln]
            i += ln
            o += ln
        else:
            ln = ctrl >> 5
            ref = o - ((ctrl & 0x1F) << 8) - 1
            if ln == 7:
                ln += src[i]
                i += 1
            ref -= src[i]
            i += 1
            ln += 2
            for _ in range(ln):          # overlapping copies run byte by byte
                out[o] = out[ref]
                o += 1
                ref += 1
    return bytes(out)


def _lzf_decompress(src: bytes, n_out: int) -> bytes:
    try:
        from numba import njit
    except ImportError:
        return _lzf_decompress_py(src, n_out)
    global _LZF_NB
    if "_LZF_NB" not in globals():
        @njit(cache=False)
        def _k(s, out):
            i = 0
            o = 0
            n = s.shape[0]
            while i < n:
                ctrl = s[i]
                i += 1
                if ctrl < 32:
                    ln = ctrl + 1
                    for j in range(ln):
                        out[o + j] = s[i + j]
                    i += ln
                    o += ln
                else:
                    ln = ctrl >> 5
                    ref = o - ((ctrl & 0x1F) << 8) - 1
                    if ln == 7:
                        ln += s[i]
                        i += 1
                    ref -= s[i]
                    i += 1
                    ln += 2
                    for _ in range(ln):
                        out[o] = out[ref]
                        o += 1
                        ref += 1
            return o
        _LZF_NB = _k
    out = np.zeros(n_out, np.uint8)
    _LZF_NB(np.frombuffer(src, np.uint8).astype(np.int64), out)
    return out.tobytes()


def _read_pcd(path):
    with open(path, "rb") as fh:
        hdr = {}
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: PCD header has no DATA line")
            s = line.decode("ascii", "replace").strip()
            if not s or s.startswith("#"):
                continue
            k, _, v = s.partition(" ")
            hdr[k.upper()] = v.split()
            if k.upper() == "DATA":
                break
        body = fh.read()
    fields = hdr["FIELDS"]
    sizes = [int(x) for x in hdr["SIZE"]]
    types = hdr["TYPE"]
    counts = [int(x) for x in hdr.get("COUNT", ["1"] * len(fields))]
    npts = int(hdr["POINTS"][0])
    kind = hdr["DATA"][0]
    tmap = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("I", 1): "i1",
            ("I", 2): "i2", ("I", 4): "i4", ("U", 8): "u8", ("I", 8): "i8"}
    dts = [tmap[(t, s)] for t, s in zip(types, sizes)]
    cols = {}
    if kind == "ascii":
        arr = np.loadtxt(body.decode("ascii").splitlines(), ndmin=2)
        j = 0
        for f, c in zip(fields, counts):
            cols[f] = arr[:, j:j + c].squeeze()
            j += c
    elif kind == "binary":
        dt = np.dtype([(f if f != "_" else f"_{i}", "<" + d, (c,) if c > 1 else ()) for i, (f, d, c) in
                       enumerate(zip(fields, dts, counts))])
        arr = np.frombuffer(body[:dt.itemsize * npts], dtype=dt, count=npts)
        cols = {f: arr[f] for f in arr.dtype.names}
    elif kind == "binary_compressed":
        csize, usize = np.frombuffer(body[:8], "<u4")
        raw = _lzf_decompress(body[8:8 + int(csize)], int(usize))
        off = 0
        for i, (f, d, c, s) in enumerate(zip(fields, dts, counts, sizes)):
            n = npts * c * s
            a = np.frombuffer(raw[off:off + n], "<" + d)
            cols[f if f != "_" else f"_{i}"] = a.reshape(npts, c) if c > 1 else a
            off += n
    else:
        raise ValueError(f"{path}: PCD DATA {kind!r} not supported")
    pts = np.stack([np.asarray(cols[a], np.float64) for a in "xyz"], 1)
    ok = np.isfinite(pts).all(1)
    out = {"points": pts[ok]}
    if all(k in cols for k in ("normal_x", "normal_y", "normal_z")):
        n = np.stack([np.asarray(cols[k], np.float64) for k in ("normal_x", "normal_y", "normal_z")], 1)[ok]
        if np.isfinite(n).all():
            out["normals"] = n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    for k in ("rgb", "rgba"):
        if k in cols:
            u = np.asarray(cols[k]).astype("<f4").view("<u4")[ok] if np.asarray(cols[k]).dtype.kind == "f" \
                else np.asarray(cols[k], np.uint32)[ok]
            out["colors"] = (np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1) / 255.0).astype(np.float32)
    if "intensity" in cols:
        out["intensity"] = np.asarray(cols["intensity"], np.float64)[ok]
    vp = [float(x) for x in hdr.get("VIEWPOINT", [])]
    if len(vp) >= 3:
        out["viewpoint"] = np.array(vp[:3])
    out["unitsHint"] = "m"
    return out


# ---------------------------------------------------------------- LAS / E57

def _read_las(path):
    try:
        import laspy
        las = laspy.read(path)
        pts = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], 1)
        out = {"points": pts.astype(np.float64), "unitsHint": "m"}
        try:
            out["colors"] = (np.stack([las.red, las.green, las.blue], 1) / 65535.0).astype(np.float32)
        except Exception:
            pass
        return out
    except ImportError:
        pass
    with open(path, "rb") as fh:
        h = fh.read(375)
        if h[:4] != b"LASF":
            raise ValueError(f"{path}: not a LAS file")
        major, minor = h[24], h[25]
        off_pts = int(np.frombuffer(h[96:100], "<u4")[0])
        fmt = h[104]
        rec_len = int(np.frombuffer(h[105:107], "<u2")[0])
        n = int(np.frombuffer(h[107:111], "<u4")[0])
        if (major, minor) >= (1, 4) and n == 0:
            n = int(np.frombuffer(h[247:255], "<u8")[0])
        scale = np.frombuffer(h[131:155], "<f8")
        offset = np.frombuffer(h[155:179], "<f8")
        if fmt & 0x80 or fmt & 0x40:
            raise ValueError(f"{path}: compressed LAZ; pip install 'laspy[lazrs]'")
        fh.seek(off_pts)
        raw = np.frombuffer(fh.read(rec_len * n), np.uint8).reshape(n, rec_len)
    xyz = raw[:, :12].copy().view("<i4").reshape(n, 3).astype(np.float64) * scale + offset
    out = {"points": xyz, "unitsHint": "m"}
    rgb_at = {2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30}.get(fmt)
    if rgb_at is not None:
        out["colors"] = (raw[:, rgb_at:rgb_at + 6].copy().view("<u2").reshape(n, 3) / 65535.0).astype(np.float32)
    return out


def _read_e57(path):
    try:
        import pye57
    except ImportError as exc:
        raise ValueError(f"{path}: reading E57 needs `pip install pye57`") from exc
    e = pye57.E57(path)
    P, V = [], []
    for i in range(e.scan_count):
        d = e.read_scan(i, ignore_missing_fields=True, transform=True)
        p = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)
        P.append(p[np.isfinite(p).all(1)])
        hdr = e.get_header(i)
        V.append(np.repeat(np.asarray(hdr.translation, float)[None], len(P[-1]), 0))
    return {"points": np.concatenate(P), "viewpoints": np.concatenate(V), "unitsHint": "m"}


# ---------------------------------------------------------------- text, meshes

def _read_text(path):
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    rows = []
    num = re.compile(r"^[\s,;]*[-+]?(\d|\.\d)")
    for ln in lines:
        if not num.match(ln):
            continue
        v = [float(x) for x in re.split(r"[\s,;]+", ln.strip()) if x]
        rows.append(v)
    if path.lower().endswith(".pts") and rows and len(rows[0]) == 1:
        rows = rows[1:]                 # Leica PTS: the point count first
    w = min(len(r) for r in rows) if rows else 0
    if w < 3:
        raise ValueError(f"{path}: no x y z columns")
    a = np.array([r[:w] for r in rows], np.float64)
    out = {"points": a[:, :3]}
    if w >= 6:
        c = a[:, 3:6]
        ln = np.linalg.norm(c, axis=1)
        if np.all(np.abs(c) <= 1.0 + 1e-6) and np.median(np.abs(ln - 1)) < 0.05:
            out["normals"] = c / np.maximum(ln, 1e-12)[:, None]
        elif w >= 7 and np.all(a[:, 4:7] >= 0) and a[:, 4:7].max() <= 255:   # PTS: x y z intensity r g b
            out["colors"] = (a[:, 4:7] / 255.0).astype(np.float32)
        elif np.all(c >= 0) and c.max() <= 255:
            out["colors"] = (c / (255.0 if c.max() > 1 else 1.0)).astype(np.float32)
    return out


def _read_mesh(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    out = {"points": np.asarray(m.vertices, np.float64), "faces": np.asarray(m.faces, np.int64)}
    if len(m.faces):
        out["normals"] = np.asarray(m.vertex_normals, np.float64)
    try:
        vc = np.asarray(m.visual.vertex_colors)
        if vc.ndim == 2 and len(vc) == len(m.vertices):
            out["colors"] = (vc[:, :3] / 255.0).astype(np.float32)
    except Exception:
        pass
    return out


def read_scan(path) -> dict:
    """Read a scan file into arrays (see the module docstring)."""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        out = _read_ply(path)
    elif ext == ".splat":
        out = _read_splat(path)
    elif ext == ".pcd":
        out = _read_pcd(path)
    elif ext in (".las", ".laz"):
        out = _read_las(path)
    elif ext == ".e57":
        out = _read_e57(path)
    elif ext in (".xyz", ".txt", ".csv", ".pts"):
        out = _read_text(path)
    elif ext in (".stl", ".obj", ".glb", ".gltf", ".off"):
        out = _read_mesh(path)
    else:
        raise ValueError(f"{path}: not a scan format loadpath reads ({', '.join(SCAN_EXT)})")
    out["format"] = ext.lstrip(".")
    return out
