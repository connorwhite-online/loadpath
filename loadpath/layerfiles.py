"""
Powder-bed layer files: a part sliced into closed contours with scan
hatches, written as CLI (the Common Layer Interface, ASCII), the format
EOS, 3D Systems, Renishaw, Farsoon, Aconity and most open powder-bed
controllers read, and nTop's Export Slices to CLI. For a lattice or a
field-built part too fine for an STL a build processor would choke on,
or a machine that takes layers rather than a mesh.

- `slice_layers(mesh, layer)`: closed contours per layer (mm), outer
  boundaries counter-clockwise and holes clockwise (seen from +build), found
  by nesting depth, so a tube's wall reads as an outer ring and a hole.
- `hatch_layer(loops, spacing, angle)`: parallel scan vectors inside the
  contours (even-odd), the angle turned by 67 degrees a layer as machines
  do, so no two layers' tracks line up.
- `write_cli(path, layers)` / `read_cli(path)`: the file and back
  (integer micrometres, $$UNITS/0.001); `binary=True` writes binary CLI
  (long commands, float32 coordinates), which `read_cli` reads too.
- `write_slc(path, layers)` / `read_slc(path)`: 3D Systems' SLC 2.0
  (binary contours, outer loops counter-clockwise and holes clockwise, no
  hatches: SLC carries boundaries only, the machine fills them).

EOS's SLI and Renishaw's MTT are closed formats without public
specifications; both machines' build processors (EOSPRINT, QuantAM) take
CLI in, so CLI is the hand-off for them.

Slicing is exact on the mesh (plane-triangle intersections chained into
loops); a layer is sampled at its mid-height, as build processors do.
"""
from __future__ import annotations

import datetime
import struct

import numpy as np

__all__ = ["Layer", "export_cli", "export_layers", "hatch_layer", "read_cli", "read_slc", "slice_layers", "write_cli",
           "write_slc"]

HATCH_TURN = 67.0          # degrees per layer


class Layer:
    """One layer: its top height (mm along the build axis), contours
    [(points (n, 2) closed, direction 1 outer / 0 hole)], hatches (m, 2, 2)."""

    def __init__(self, z, contours, hatches=None):
        self.z = float(z)
        self.contours = contours
        self.hatches = np.zeros((0, 2, 2)) if hatches is None else np.asarray(hatches, float).reshape(-1, 2, 2)

    def area(self) -> float:
        return float(sum((1 if d == 1 else -1) * abs(_signed_area(p)) for p, d in self.contours))


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _chain(segs, tol):
    """Segments (n, 2, 2), in no particular direction, into closed loops
    by matching endpoints (every endpoint of a closed slice is shared by
    exactly two segments)."""
    if not len(segs):
        return []
    key = np.round(segs.reshape(-1, 2) / tol).astype(np.int64)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 2)
    keep = ids[:, 0] != ids[:, 1]
    ids, segs = ids[keep], segs[keep]
    adj = {}
    for k, (a, b) in enumerate(ids):
        adj.setdefault(int(a), []).append(k)
        adj.setdefault(int(b), []).append(k)
    used = np.zeros(len(ids), bool)
    loops = []
    for k0 in range(len(ids)):
        if used[k0]:
            continue
        used[k0] = True
        start, cur = int(ids[k0, 0]), int(ids[k0, 1])
        pts = [segs[k0, 0], segs[k0, 1]]
        while cur != start:
            k = next((k for k in adj.get(cur, []) if not used[k]), None)
            if k is None:
                break                          # an open chain: the mesh has a hole here
            used[k] = True
            if int(ids[k, 0]) == cur:
                cur = int(ids[k, 1])
                pts.append(segs[k, 1])
            else:
                cur = int(ids[k, 0])
                pts.append(segs[k, 0])
        if cur == start and len(pts) >= 4:
            loops.append(np.asarray(pts[:-1], float))
    return loops


def _inside(pt, poly):
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = (ys > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = xs + (y - ys) * (x2 - xs) / (y2 - ys)
    return bool(np.count_nonzero(cross & (x < xi)) % 2)


def _orient(loops):
    """(points, direction): depth-even loops are outer (counter-clockwise,
    direction 1), odd ones holes (clockwise, 0)."""
    out = []
    for i, p in enumerate(loops):
        probe = p[0] + 1e-7 * (p[1] - p[0])
        depth = sum(_inside(probe, q) for j, q in enumerate(loops) if j != i)
        outer = depth % 2 == 0
        ccw = _signed_area(p) > 0
        if ccw != outer:
            p = p[::-1]
        out.append((p, 1 if outer else 0))
    return out


def _frame(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    if np.allclose(a, [0, 0, 1]):
        u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    return a, u, v


def slice_layers(mesh, layer: float, axis=(0.0, 0.0, 1.0)) -> list:
    """Contours of a closed mesh every `layer` mm along `axis` (default +z,
    with x and y kept as the layer's axes). Heights start at the part's
    bottom; each layer is cut at its mid-height and reported at its top."""
    from trimesh.intersections import mesh_multiplane

    if layer <= 0:
        raise ValueError("layer thickness must be positive (mm)")
    a, u, v = _frame(axis)
    h = mesh.vertices @ a
    lo, hi = float(h.min()), float(h.max())
    n = int(np.ceil((hi - lo) / layer - 1e-9))
    if n > 200_000:
        raise ValueError(f"{n} layers: check the layer thickness (mm)")
    mids = lo + (np.arange(n) + 0.5) * layer
    tol = 1e-6 * max(float(np.ptp(mesh.vertices, axis=0).max()), 1.0)
    out = []
    for k0 in range(0, n, 256):
        heights = mids[k0:k0 + 256] - lo
        lines, tfs, _ = mesh_multiplane(mesh, plane_origin=a * lo, plane_normal=a, heights=heights)
        for k, (seg2, tf) in enumerate(zip(lines, tfs)):
            seg2 = np.asarray(seg2, float).reshape(-1, 2)
            seg3 = seg2 @ np.asarray(tf)[:3, :2].T + np.asarray(tf)[:3, 3]      # the plane's frame to world
            seg = np.stack([seg3 @ u, seg3 @ v], axis=-1).reshape(-1, 2, 2)
            out.append(Layer(lo + (k0 + k + 1) * layer, _orient(_chain(seg, tol))))
    return out


def hatch_layer(contours, spacing: float, angle: float = 0.0) -> np.ndarray:
    """Scan vectors (m, 2, 2) filling the contours (even-odd) every
    `spacing` mm at `angle` degrees, each line's segments alternating in
    direction (a serpentine, as scanners run them)."""
    if spacing <= 0 or not contours:
        return np.zeros((0, 2, 2))
    t = np.radians(angle)
    c, s = np.cos(t), np.sin(t)
    rot = np.array([[c, s], [-s, c]])                # into the hatch frame: lines along x'
    edges = []
    for p, _ in contours:
        q = p @ rot.T
        edges.append(np.stack([q, np.roll(q, -1, axis=0)], axis=1))
    E = np.concatenate(edges)
    y0, y1 = float(E[..., 1].min()), float(E[..., 1].max())
    ys = np.arange(np.floor(y0 / spacing) * spacing + 0.5 * spacing, y1, spacing)
    A, B = E[:, 0], E[:, 1]
    lo_y, hi_y = np.minimum(A[:, 1], B[:, 1]), np.maximum(A[:, 1], B[:, 1])
    out = []
    for i, y in enumerate(ys):
        hit = (lo_y <= y) & (hi_y > y)
        if not hit.any():
            continue
        a, b = A[hit], B[hit]
        x = np.sort(a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
        segs = x[: len(x) // 2 * 2].reshape(-1, 2)
        segs = segs[segs[:, 1] - segs[:, 0] > 1e-9]
        if i % 2:
            segs = segs[::-1, ::-1]
        for xa, xb in segs:
            out.append([[xa, y], [xb, y]])
    if not out:
        return np.zeros((0, 2, 2))
    H = np.asarray(out, float)
    return H @ rot                                     # back to the layer's axes


def write_cli(path: str, layers: list, label: str = "part", units: float = 0.001, binary: bool = False) -> dict:
    """An ASCII CLI file: one $$LAYER per layer (its top height), a
    $$POLYLINE per contour (direction 1 outer, 0 hole), one $$HATCHES per
    layer. Coordinates in integer multiples of `units` mm. `binary`: the
    binary CLI form (the same header with $$BINARY, then long commands:
    127 layer, 130 polyline, 132 hatches, float32 in `units`)."""
    if binary:
        return _write_cli_binary(path, layers, label, units)
    def q(v):
        return np.rint(np.asarray(v, float) / units).astype(np.int64)

    pts = [p for L in layers for p, _ in L.contours]
    if pts:
        allp = np.concatenate(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
    else:
        lo = hi = np.zeros(2)
    z0 = layers[0].z if layers else 0.0
    z1 = layers[-1].z if layers else 0.0
    n_poly = n_hatch = 0
    with open(path, "w", newline="\n") as fh:
        fh.write("$$HEADERSTART\n$$ASCII\n")
        fh.write(f"$$UNITS/{units:.6g}\n$$VERSION/200\n$$LABEL/1,{label}\n")
        fh.write(f"$$DATE/{datetime.date.today():%d%m%y}\n")
        d = np.concatenate([q(lo), q([z0]), q(hi), q([z1])])
        fh.write("$$DIMENSION/" + ",".join(str(int(v)) for v in d) + "\n")
        fh.write(f"$$LAYERS/{len(layers)}\n$$HEADEREND\n$$GEOMETRYSTART\n")
        for L in layers:
            fh.write(f"$$LAYER/{int(q([L.z])[0])}\n")
            for p, direc in L.contours:
                c = q(np.vstack([p, p[:1]]))
                fh.write(f"$$POLYLINE/1,{direc},{len(c)}," + ",".join(str(int(v)) for v in c.ravel()) + "\n")
                n_poly += 1
            if len(L.hatches):
                h = q(L.hatches.reshape(-1, 4))
                fh.write(f"$$HATCHES/1,{len(h)}," + ",".join(str(int(v)) for v in h.ravel()) + "\n")
                n_hatch += len(h)
        fh.write("$$GEOMETRYEND\n")
    return {"path": path, "layers": len(layers), "contours": n_poly, "hatches": n_hatch, "units": units}


def _header_extent(layers):
    pts = [p for L in layers for p, _ in L.contours]
    if pts:
        allp = np.concatenate(pts)
        return allp.min(axis=0), allp.max(axis=0)
    return np.zeros(2), np.zeros(2)


def _write_cli_binary(path, layers, label, units):
    lo, hi = _header_extent(layers)
    z0 = layers[0].z if layers else 0.0
    z1 = layers[-1].z if layers else 0.0
    n_poly = n_hatch = 0
    head = ("$$HEADERSTART\n$$BINARY\n"
            f"$$UNITS/{units:.6g}\n$$VERSION/200\n$$LABEL/1,{label}\n$$DATE/{datetime.date.today():%d%m%y}\n"
            f"$$DIMENSION/{lo[0]/units:.6f},{lo[1]/units:.6f},{z0/units:.6f},{hi[0]/units:.6f},{hi[1]/units:.6f},"
            f"{z1/units:.6f}\n$$LAYERS/{len(layers)}\n$$HEADEREND")
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        for L in layers:
            fh.write(struct.pack("<Hf", 127, L.z / units))
            for p, direc in L.contours:
                c = np.vstack([p, p[:1]]) / units
                fh.write(struct.pack("<Hiii", 130, 1, int(direc), len(c)))
                fh.write(c.astype("<f4").tobytes())
                n_poly += 1
            if len(L.hatches):
                h = L.hatches.reshape(-1, 4) / units
                fh.write(struct.pack("<Hii", 132, 1, len(h)))
                fh.write(h.astype("<f4").tobytes())
                n_hatch += len(h)
    return {"path": path, "layers": len(layers), "contours": n_poly, "hatches": n_hatch, "units": units,
            "format": "cli-binary"}


def _read_cli_binary(data, start, units):
    layers, i = [], start
    while i + 2 <= len(data):
        (cmd,) = struct.unpack_from("<H", data, i)
        i += 2
        if cmd == 127:
            (z,) = struct.unpack_from("<f", data, i)
            i += 4
            layers.append(Layer(z * units, []))
        elif cmd == 128:
            (z,) = struct.unpack_from("<H", data, i)
            i += 2
            layers.append(Layer(z * units, []))
        elif cmd == 130:
            _id, direc, n = struct.unpack_from("<iii", data, i)
            i += 12
            p = np.frombuffer(data, "<f4", 2 * n, i).reshape(n, 2).astype(float) * units
            i += 8 * n
            layers[-1].contours.append((p[:-1] if n > 1 and np.allclose(p[0], p[-1]) else p, direc))
        elif cmd == 129:
            _id, direc, n = struct.unpack_from("<HHH", data, i)
            i += 6
            p = np.frombuffer(data, "<u2", 2 * n, i).reshape(n, 2).astype(float) * units
            i += 4 * n
            layers[-1].contours.append((p[:-1] if n > 1 and np.allclose(p[0], p[-1]) else p, direc))
        elif cmd == 132:
            _id, n = struct.unpack_from("<ii", data, i)
            i += 8
            h = np.frombuffer(data, "<f4", 4 * n, i).reshape(n, 2, 2).astype(float) * units
            i += 16 * n
            layers[-1].hatches = np.concatenate([layers[-1].hatches, h])
        elif cmd == 131:
            _id, n = struct.unpack_from("<HH", data, i)
            i += 4
            h = np.frombuffer(data, "<u2", 4 * n, i).reshape(n, 2, 2).astype(float) * units
            i += 8 * n
            layers[-1].hatches = np.concatenate([layers[-1].hatches, h])
        else:
            raise ValueError(f"binary CLI: unknown command {cmd} at byte {i - 2}")
    return layers


def read_cli(path: str) -> list:
    """Layers back from a CLI file, ASCII or binary (contours and hatches, mm)."""
    with open(path, "rb") as fh:
        data = fh.read()
    end = data.find(b"$$HEADEREND")
    if end >= 0 and b"$$BINARY" in data[:end]:
        units = 1.0
        for ln in data[:end].decode("ascii", "replace").splitlines():
            if ln.strip().startswith("$$UNITS/"):
                units = float(ln.strip().split("/", 1)[1])
        return _read_cli_binary(data, end + len(b"$$HEADEREND"), units)
    units, layers = 1.0, []
    geo = False
    with open(path) as fh:
        for raw in fh:
            ln = raw.strip()
            if ln.startswith("$$UNITS/"):
                units = float(ln.split("/", 1)[1])
            elif ln.startswith("$$GEOMETRYSTART"):
                geo = True
            elif not geo:
                continue
            elif ln.startswith("$$LAYER/"):
                layers.append(Layer(float(ln.split("/", 1)[1]) * units, []))
            elif ln.startswith("$$POLYLINE/"):
                v = [float(x) for x in ln.split("/", 1)[1].split(",")]
                direc, n = int(v[1]), int(v[2])
                p = np.asarray(v[3:3 + 2 * n]).reshape(n, 2) * units
                layers[-1].contours.append((p[:-1] if np.allclose(p[0], p[-1]) else p, direc))
            elif ln.startswith("$$HATCHES/"):
                v = [float(x) for x in ln.split("/", 1)[1].split(",")]
                n = int(v[1])
                h = np.asarray(v[2:2 + 4 * n]).reshape(n, 2, 2) * units
                layers[-1].hatches = np.concatenate([layers[-1].hatches, h])
    return layers


SLC_END = 0xFFFFFFFF


def write_slc(path: str, layers: list, layer: float | None = None, label: str = "loadpath") -> dict:
    """3D Systems SLC 2.0: an ASCII header (to CR LF ^Z), 256 reserved
    bytes, a one-entry sampling table (first z, layer thickness, line-width
    compensation 0), then per layer its z (float32), boundary count and each
    boundary as a closed float32 polyline (vertices, gaps 0); the file ends
    with the last z and 0xFFFFFFFF. Contours only, in mm; holes run
    clockwise."""
    lo, hi = _header_extent(layers)
    z0 = layers[0].z if layers else 0.0
    z1 = layers[-1].z if layers else 0.0
    if layer is None:
        layer = float(np.median(np.diff([L.z for L in layers]))) if len(layers) > 1 else 0.1
    head = (f"-SLCVER 2.0 -UNIT MM -TYPE PART -PACKAGE {label} "
            f"-EXTENTS {lo[0]:.4f},{hi[0]:.4f} {lo[1]:.4f},{hi[1]:.4f} {z0 - layer:.4f},{z1:.4f}")
    n_poly = 0
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii") + b"\r\n\x1a")
        fh.write(bytes(256))
        fh.write(struct.pack("<B4f", 1, z0 - layer, layer, 0.0, 0.0))
        for L in layers:
            fh.write(struct.pack("<fI", L.z, len(L.contours)))
            for p, _ in L.contours:
                c = np.vstack([p, p[:1]])
                fh.write(struct.pack("<II", len(c), 0))
                fh.write(c.astype("<f4").tobytes())
                n_poly += 1
        fh.write(struct.pack("<fI", z1, SLC_END))
    return {"path": path, "layers": len(layers), "contours": n_poly, "hatches": 0, "layerThickness": layer,
            "format": "slc"}


def read_slc(path: str) -> list:
    """Layers back from an SLC file (contours, mm; direction from each
    loop's winding: counter-clockwise outer, clockwise hole)."""
    with open(path, "rb") as fh:
        data = fh.read()
    end = data.find(b"\r\n\x1a")
    if end < 0:
        raise ValueError(f"{path}: no SLC header (it ends with CR LF ^Z)")
    head = data[:end].decode("ascii", "replace").upper()
    scale = 25.4 if "-UNIT INCH" in head else 1.0
    i = end + 3 + 256
    (n_tab,) = struct.unpack_from("<B", data, i)
    i += 1 + 16 * n_tab
    layers = []
    while i + 8 <= len(data):
        z, nb = struct.unpack_from("<fI", data, i)
        i += 8
        if nb == SLC_END:
            break
        L = Layer(z * scale, [])
        for _ in range(nb):
            nv, _ng = struct.unpack_from("<II", data, i)
            i += 8
            p = np.frombuffer(data, "<f4", 2 * nv, i).reshape(nv, 2).astype(float) * scale
            i += 8 * nv
            if nv > 1 and np.allclose(p[0], p[-1]):
                p = p[:-1]
            L.contours.append((p, 1 if _signed_area(p) > 0 else 0))
        layers.append(L)
    return layers


def export_layers(mesh, path: str, fmt: str = "cli", layer: float = 0.03, hatch: float | None = 0.1,
                  axis=(0.0, 0.0, 1.0), label: str = "part", turn: float = HATCH_TURN) -> dict:
    """Slice and write one of the layer formats: cli (ASCII), cli-binary or
    slc (contours only)."""
    if fmt == "cli":
        return export_cli(mesh, path, layer, hatch, axis, label, turn)
    if fmt not in ("cli-binary", "slc"):
        raise ValueError(f"layer format must be cli, cli-binary or slc, not {fmt!r}")
    layers = slice_layers(mesh, layer, axis)
    if fmt == "slc":
        info = write_slc(path, layers, layer=layer, label=label)
    else:
        for k, L in enumerate(layers):
            if hatch:
                L.hatches = hatch_layer(L.contours, hatch, (k * turn) % 180.0)
        info = write_cli(path, layers, label=label, binary=True)
    vol = float(sum(L.area() for L in layers) * layer)
    info.update(layerThickness=layer, hatchSpacing=hatch if fmt != "slc" else None, height=len(layers) * layer,
                volume=vol, meshVolume=float(abs(mesh.volume)),
                openLayers=int(sum(1 for L in layers if not L.contours and L is not layers[-1])))
    return info


def export_cli(mesh, path: str, layer: float = 0.03, hatch: float | None = 0.1, axis=(0.0, 0.0, 1.0),
               label: str = "part", turn: float = HATCH_TURN) -> dict:
    """Slice a closed mesh, hatch every layer (`hatch` mm apart, turned
    `turn` degrees a layer; None for contours only) and write the CLI file.
    Returns counts and the build's height and area-weighted volume (a check
    against the mesh's)."""
    layers = slice_layers(mesh, layer, axis)
    for k, L in enumerate(layers):
        if hatch:
            L.hatches = hatch_layer(L.contours, hatch, (k * turn) % 180.0)
    info = write_cli(path, layers, label=label)
    vol = float(sum(L.area() for L in layers) * layer)
    info.update(layerThickness=layer, hatchSpacing=hatch, height=len(layers) * layer,
                volume=vol, meshVolume=float(abs(mesh.volume)),
                openLayers=int(sum(1 for L in layers if not L.contours and L is not layers[-1])))
    return info
