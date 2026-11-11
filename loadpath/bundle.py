"""
The results bundle: a part with what its simulation found, in files other
software opens, and one manifest that says what every number means.

A CAD tool, a slicer or a print service sees the delivered mesh and
nothing else, so the reason the part looks the way it does is lost at the
handoff: where it is weakest and by what criterion, which load each arm
carries, what each keep-out keeps clear, how hot it runs. This writes it
all out:

- `part.glb`: glTF 2.0 (any web viewer, Blender, three.js, USD tools). The
  part coloured by its governing safety factor (COLOR_0), with the real
  fields as extra vertex attributes (_SAFETY_FACTOR, _STRESS_<i>,
  _DISPLACEMENT_<i>, _TEMPERATURE); keep-outs, preserved regions, supports
  and the load arrows as named nodes whose `extras` say what they are.
- `analysis.vtu`: the analysis model itself (ParaView, VisIt, Ansys and
  Abaqus viewers, pyvista): the hexahedral cells the part was verified on,
  with the stress tensor, equivalent stress, displacement, safety factor,
  what limits it and the contact band at every node, per load case.
- `surface.vtp`: the delivered surface with the same fields read onto it.
- flow and temperature grids (`<name>.vti`, and `<name>.vdb` for Houdini,
  Blender and OpenVDB tools) when a run simulated a fluid.
- `tradeoff.csv`: every candidate the search verified (mass, safety factor,
  deflection, buckling), for any charting tool.
- `steps.json`: the numbered assembly moves, when the run planned them.
- `manifest.json`: each field's quantity, units, criterion, target, which
  way is better, its range and where its worst value is, which analysis
  made it and why it is there; each object's role and physical reason;
  each file and what opens it. `loadpath schema bundle` is its JSON Schema.

Every value is the verification's own (verify_part(..., fields=True)):
nothing is painted on. Display choices (the colour scale, an arrow's
drawn length) are named as such in the manifest.

Units in the files are loadpath's (mm, N, MPa, C), z up; glTF's convention
is metres and y up, so its root node carries the scale and rotation that
take one to the other, and a viewer that honours node transforms shows the
part the right size and way up.

    from loadpath.bundle import export_bundle
    export_bundle("out/bundle", study, mesh, report, fields, summary=run_summary)
"""
from __future__ import annotations

import base64
import csv
import json
import os
import struct
import time

import numpy as np

__all__ = ["export_bundle", "write_glb", "write_vtu", "write_vtp", "write_vti", "flow_grids", "exchanger_grids",
           "SF_CAP", "BUNDLE_VERSION"]

BUNDLE_VERSION = 1
# a node that carries (almost) no stress has a safety factor of infinity;
# files hold this instead (glTF accessors and colour ranges need finite numbers)
SF_CAP = 1000.0
_LIMITED = {0: "material (in-plane)", 1: "layer bond", 2: "cold end of the thermal cycle"}
# fem.CORNERS order (bit 0 x, bit 1 y, bit 2 z) to VTK_HEXAHEDRON's
_VTK_HEX = [0, 1, 3, 2, 4, 5, 7, 6]

_OPENS = {
    "glb": ["three.js / any glTF web viewer", "Blender", "Windows 3D Viewer", "macOS Preview / Reality Composer",
            "USD tools (via glTF import)"],
    "vtu": ["ParaView", "VisIt", "pyvista / VTK", "Ansys EnSight (VTK import)"],
    "vtp": ["ParaView", "VisIt", "pyvista / VTK"],
    "vti": ["ParaView", "VisIt", "pyvista / VTK"],
    "vdb": ["Houdini", "Blender (volume object)", "OpenVDB tools", "Unreal / Omniverse"],
    "csv": ["any spreadsheet or charting tool"],
    "json": ["anything"],
    "stl": ["any CAD tool, slicer or print service"],
}


# ------------------------------------------------------------------ helpers

def _finite(a, cap=SF_CAP):
    a = np.asarray(a, np.float64)
    return np.nan_to_num(np.minimum(a, cap), nan=cap, posinf=cap, neginf=-cap)


def _rng(a, where=None, xyz=None, worst="min"):
    a = np.asarray(a, np.float64)
    m = np.isfinite(a) if where is None else (np.isfinite(a) & where)
    if not m.any():
        return {"min": None, "max": None}
    v = a[m]
    out = {"min": float(v.min()), "max": float(v.max()), "p01": float(np.percentile(v, 1)),
           "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99))}
    if xyz is not None:
        i = np.flatnonzero(m)[int(np.argmin(v) if worst == "min" else np.argmax(v))]
        out["worstAt"] = [round(float(c), 4) for c in xyz[i]]
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f or f in (float("inf"), float("-inf")) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


# the safety-factor colour scale: a display choice, written to the manifest
_SF_STOPS = [(0.0, "#7f1d1d"), (1.0, "#dc2626"), ("target", "#f59e0b"), ("2target", "#16a34a"),
             ("4target", "#94a3b8")]


def sf_scale(target):
    t = float(target or 1.0)
    return [{"safetyFactor": (t if s == "target" else 2 * t if s == "2target" else 4 * t if s == "4target"
                              else float(s)), "color": c} for s, c in _SF_STOPS]


def sf_colors(sf, target):
    """sRGB colours (n, 3) for safety factors by sf_scale(target)."""
    stops = sf_scale(target)
    xs = np.array([s["safetyFactor"] for s in stops])
    cs = np.array([_hex(s["color"]) for s in stops])
    v = np.clip(_finite(sf), xs[0], xs[-1])
    return np.stack([np.interp(v, xs, cs[:, k]) for k in range(3)], axis=1)


# ------------------------------------------------------------------ glTF

class _GLB:
    """A minimal glTF 2.0 binary writer: meshes, nodes with extras, PBR
    materials, one buffer."""

    def __init__(self):
        self.bin = bytearray()
        self.doc = {"asset": {"version": "2.0", "generator": "loadpath bundle"}, "scenes": [{"nodes": []}],
                    "scene": 0, "nodes": [], "meshes": [], "materials": [], "accessors": [], "bufferViews": [],
                    "buffers": []}

    def _view(self, data: bytes, target=None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin.extend(data)
        self.doc["bufferViews"].append(bv)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, a, kind, minmax=False, index=False) -> int:
        a = np.ascontiguousarray(a, np.uint32 if index else np.float32)
        acc = {"bufferView": self._view(a.tobytes(), 34963 if index else 34962),
               "componentType": 5125 if index else 5126, "count": int(a.shape[0] if not index else a.size),
               "type": kind}
        if minmax:
            acc["min"] = [float(v) for v in a.reshape(len(a), -1).min(axis=0)]
            acc["max"] = [float(v) for v in a.reshape(len(a), -1).max(axis=0)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def material(self, name, color, alpha=1.0, roughness=0.6, metallic=0.0, vertex_colors=False) -> int:
        lin = _srgb_to_linear(_hex(color) if isinstance(color, str) else color).tolist()
        m = {"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, float(alpha)] if vertex_colors else lin + [float(alpha)],
            "metallicFactor": metallic, "roughnessFactor": roughness}, "doubleSided": alpha < 1.0}
        if alpha < 1.0:
            m["alphaMode"] = "BLEND"
        self.doc["materials"].append(m)
        return len(self.doc["materials"]) - 1

    def mesh(self, name, vertices, faces, material, normals=True, attributes=None, extras=None) -> int:
        import trimesh

        v = np.asarray(vertices, np.float32)
        f = np.asarray(faces, np.uint32)
        prim = {"attributes": {"POSITION": self.accessor(v, "VEC3", minmax=True)},
                "indices": self.accessor(f.ravel(), "SCALAR", index=True), "material": material, "mode": 4}
        if normals:
            n = trimesh.Trimesh(v, f, process=False).vertex_normals
            prim["attributes"]["NORMAL"] = self.accessor(np.asarray(n, np.float32), "VEC3")
        for k, a in (attributes or {}).items():
            a = np.asarray(a, np.float32)
            prim["attributes"][k] = self.accessor(a, {1: "SCALAR", 3: "VEC3", 4: "VEC4"}[1 if a.ndim == 1 else a.shape[1]])
        m = {"name": name, "primitives": [prim]}
        if extras:
            m["extras"] = _jsonable(extras)
        self.doc["meshes"].append(m)
        return len(self.doc["meshes"]) - 1

    def node(self, name, mesh=None, parent=None, extras=None, **trs) -> int:
        n = {"name": name}
        if mesh is not None:
            n["mesh"] = mesh
        for k in ("translation", "rotation", "scale"):
            if trs.get(k) is not None:
                n[k] = [float(x) for x in trs[k]]
        if extras:
            n["extras"] = _jsonable(extras)
        self.doc["nodes"].append(n)
        i = len(self.doc["nodes"]) - 1
        if parent is None:
            self.doc["scenes"][0]["nodes"].append(i)
        else:
            self.doc["nodes"][parent].setdefault("children", []).append(i)
        return i

    def save(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        for k in ("meshes", "materials", "accessors", "bufferViews"):
            if not self.doc[k]:
                self.doc.pop(k)
        js = json.dumps(self.doc, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", 0x46546C67, 2, total))
            fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            fh.write(struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))
        return path


def _arrow(length, radius):
    """An arrow along +z from the origin: shaft and head (trimesh)."""
    import trimesh

    head = min(0.3 * length, 6 * radius)
    shaft = trimesh.creation.cylinder(radius=radius, height=length - head, sections=16)
    shaft.apply_translation([0, 0, (length - head) / 2])
    cone = trimesh.creation.cone(radius=2.4 * radius, height=head, sections=16)
    cone.apply_translation([0, 0, length - head])
    return trimesh.util.concatenate([shaft, cone])


def _quat_from_z(d):
    """The unit quaternion (x, y, z, w) turning +z onto unit d."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ d)
    if c < -0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    ax = np.cross(z, d)
    q = np.array([ax[0], ax[1], ax[2], 1.0 + c])
    return (q / np.linalg.norm(q)).tolist()


def write_glb(path, mesh, attributes=None, colors=None, objects=(), part_extras=None, name="part") -> str:
    """The part (trimesh, mm, z up) with per-vertex `attributes` (name ->
    array, names starting with an underscore) and sRGB `colors`, and
    `objects`: dicts {name, mesh (trimesh) or arrow {at, dir, length},
    color, alpha, extras}. A root node takes loadpath's mm, z-up frame to
    glTF's metres, y up."""
    g = _GLB()
    root = g.node("loadpath (mm, z up)", extras={"loadpath": {
        "units": "mm", "up": "z", "note": "this node scales mm to metres and turns z-up to glTF's y-up"}},
        rotation=[-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], scale=[1e-3] * 3)
    attrs = dict(attributes or {})
    if colors is not None:
        attrs["COLOR_0"] = _srgb_to_linear(colors)
    mat = g.material(name, "#d9dde3", vertex_colors=colors is not None)
    mi = g.mesh(name, mesh.vertices, mesh.faces, mat, attributes=attrs, extras=part_extras)
    g.node(name, mi, parent=root, extras=part_extras)
    groups = {}
    for ob in objects:
        grp = ob.get("group") or "context"
        if grp not in groups:
            groups[grp] = g.node(grp, parent=root)
        matj = g.material(ob["name"], ob.get("color", "#8a94a6"), ob.get("alpha", 1.0))
        if ob.get("arrow"):
            a = ob["arrow"]
            m = _arrow(a["length"], a.get("radius", a["length"] / 40))
            mj = g.mesh(ob["name"], m.vertices, m.faces, matj)
            g.node(ob["name"], mj, parent=groups[grp], extras=ob.get("extras"),
                   translation=a["at"], rotation=_quat_from_z(a["dir"]))
        elif ob.get("point") is not None:
            import trimesh
            s = trimesh.creation.icosphere(subdivisions=2, radius=ob.get("radius", 1.0))
            mj = g.mesh(ob["name"], s.vertices, s.faces, matj)
            g.node(ob["name"], mj, parent=groups[grp], extras=ob.get("extras"), translation=ob["point"])
        else:
            m = ob["mesh"]
            mj = g.mesh(ob["name"], m.vertices, m.faces, matj)
            g.node(ob["name"], mj, parent=groups[grp], extras=ob.get("extras"))
    return g.save(path)


# ------------------------------------------------------------------ VTK XML

def _b64(a, dtype, block=1 << 20):
    """A data array's bytes, zlib-compressed in blocks, as VTK's inline
    binary form: the base64 block header (count, block size, last block's
    size, each compressed size) then the base64 data."""
    import zlib

    raw = np.ascontiguousarray(a, dtype).tobytes()
    chunks = [raw[i:i + block] for i in range(0, len(raw), block)] or [b""]
    comp = [zlib.compress(c, 6) for c in chunks]
    head = struct.pack(f"<{3 + len(comp)}Q", len(chunks), block if len(chunks) > 1 else len(chunks[0]),
                       len(chunks[-1]), *[len(c) for c in comp])
    return base64.b64encode(head).decode("ascii") + base64.b64encode(b"".join(comp)).decode("ascii")


_VTKT = {np.float32: "Float32", np.float64: "Float64", np.int64: "Int64", np.uint8: "UInt8", np.int8: "Int8",
         np.int32: "Int32"}


def _array(name, a, dtype=np.float32, extra=""):
    a = np.asarray(a)
    nc = 1 if a.ndim == 1 else int(a.shape[1])
    comp = f' NumberOfComponents="{nc}"' if nc > 1 else ""
    return (f'<DataArray type="{_VTKT[dtype]}" Name="{name}"{comp} format="binary"{extra}>'
            f'{_b64(a, dtype)}</DataArray>\n')


def _data_block(tag, data):
    if not data:
        return ""
    body = "".join(_array(k, v, np.int8 if np.asarray(v).dtype in (np.int8, np.bool_) else np.float32)
                   for k, v in data.items())
    return f"<{tag}>\n{body}</{tag}>\n"


def _vtk_file(kind, inner, path):
    xml = (f'<?xml version="1.0"?>\n<VTKFile type="{kind}" version="1.0" byte_order="LittleEndian" '
           f'header_type="UInt64" compressor="vtkZLibDataCompressor">\n{inner}</VTKFile>\n')
    with open(path, "w") as fh:
        fh.write(xml)
    return path


def write_vtu(path, points, hexes, point_data=None, cell_data=None) -> str:
    """An unstructured grid of 8-node hexahedra (`hexes` in VTK order)."""
    pts = np.asarray(points, np.float64)
    hx = np.asarray(hexes, np.int64)
    inner = (f'<UnstructuredGrid>\n<Piece NumberOfPoints="{len(pts)}" NumberOfCells="{len(hx)}">\n'
             + _data_block("PointData", point_data) + _data_block("CellData", cell_data)
             + "<Points>\n" + _array("Points", pts, np.float32) + "</Points>\n<Cells>\n"
             + _array("connectivity", hx.ravel(), np.int64)
             + _array("offsets", 8 * np.arange(1, len(hx) + 1), np.int64)
             + _array("types", np.full(len(hx), 12), np.uint8) + "</Cells>\n</Piece>\n</UnstructuredGrid>\n")
    return _vtk_file("UnstructuredGrid", inner, path)


def write_vtp(path, points, triangles, point_data=None) -> str:
    pts = np.asarray(points, np.float64)
    tri = np.asarray(triangles, np.int64)
    inner = (f'<PolyData>\n<Piece NumberOfPoints="{len(pts)}" NumberOfVerts="0" NumberOfLines="0" '
             f'NumberOfStrips="0" NumberOfPolys="{len(tri)}">\n' + _data_block("PointData", point_data)
             + "<Points>\n" + _array("Points", pts, np.float32) + "</Points>\n<Polys>\n"
             + _array("connectivity", tri.ravel(), np.int64)
             + _array("offsets", 3 * np.arange(1, len(tri) + 1), np.int64) + "</Polys>\n</Piece>\n</PolyData>\n")
    return _vtk_file("PolyData", inner, path)


def write_vti(path, origin, spacing, point_data) -> str:
    """A regular grid: `point_data` arrays indexed [i, j, k] (scalar) or
    [i, j, k, c] (vector), at origin + spacing * (i, j, k)."""
    shape = None
    flat = {}
    for k, a in point_data.items():
        a = np.asarray(a)
        shape = a.shape[:3]
        flat[k] = a.transpose(2, 1, 0, 3).reshape(-1, a.shape[3]) if a.ndim == 4 else a.transpose(2, 1, 0).ravel()
        if flat[k].dtype == np.bool_:
            flat[k] = flat[k].astype(np.int8)
    nx, ny, nz = shape
    ext = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    o = " ".join(f"{v:g}" for v in origin)
    sp = " ".join(f"{v:g}" for v in ([spacing] * 3 if np.isscalar(spacing) else spacing))
    inner = (f'<ImageData WholeExtent="{ext}" Origin="{o}" Spacing="{sp}">\n<Piece Extent="{ext}">\n'
             + _data_block("PointData", flat) + "</Piece>\n</ImageData>\n")
    return _vtk_file("ImageData", inner, path)


# ------------------------------------------------------------------ the bundle

def _interp(nodes, verts, k=4):
    """Inverse-distance-squared weights from the k nearest analysis nodes
    to each surface vertex (the viewer's reading)."""
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(nodes).query(np.asarray(verts), k=k)
    w = 1.0 / np.maximum(dist, 1e-6) ** 2
    w /= w.sum(axis=1, keepdims=True)
    return idx, w


def _on(idx, w, a):
    a = np.asarray(a, np.float64)
    return (a[idx] * w).sum(axis=1) if a.ndim == 1 else (a[idx] * w[..., None]).sum(axis=1)


def _from_body(a, body, nodes):
    """`a` with each node outside `body` given its nearest body node's value."""
    from scipy.spatial import cKDTree

    a = np.asarray(a, np.float64).copy()
    if body.all() or not body.any():
        return a
    _, j = cKDTree(nodes[body]).query(nodes[~body], k=1)
    a[~body] = a[np.flatnonzero(body)[j]]
    return a


def _manifold_mesh(shape):
    import trimesh
    mf = shape.to_manifold().to_mesh()
    return trimesh.Trimesh(np.asarray(mf.vert_properties)[:, :3], np.asarray(mf.tri_verts), process=False)


def _keep_out_reason(r) -> str:
    if getattr(r, "wall", None):
        return (f"a pressurized passage: kept open, with {r.wall:g} mm of wall pinned round it to hold the "
                "pressure")
    if not getattr(r, "bore", True):
        return ("a clearance volume (a moving part's or a component's space): no material may enter it, and it "
                "may end against material")
    if not getattr(r, "cuts", True):
        return "kept clear of new material, but not cut through what the designer drew"
    return "a bore or a way in: something passes through it (a fastener, pin, shaft or a part being inserted)"


def _criterion_text(report, study) -> str:
    f = report.get("failure") or {}
    crit = f.get("criterion", "von Mises")
    parts = [f"{crit} equivalent stress against the material's strength"]
    if study.process is not None and getattr(study.process, "build", None) not in (None, "auto") \
            and getattr(study.process, "interlayer", 1.0) < 1.0:
        parts.append(f"the layer bond along the build direction (bond at {study.process.interlayer:.0%} of the "
                     "in-plane strength, tension and shear combined)")
    if study.thermal is not None:
        parts.append("strength retained at each node's temperature")
    return "; ".join(parts)


def export_bundle(out_dir, study, mesh, report, fields, summary=None, name=None, flows=None, steps=None,
                  stl=None, extra_files=None) -> dict:
    """Write the bundle for one verified part to `out_dir`.

    study: the Study; mesh: the delivered trimesh; report: its verification
    (verify_part's, or the run's outcome report); fields: verify_part(...,
    fields=True)["fields"]; summary: the run's report (tradeoff.csv and the
    deliver verdict); flows: flow_grids(...) entries; steps: assembly
    steps [{n, text, ...}]; stl: the part's STL to copy beside the rest.
    Returns the manifest (also written as manifest.json)."""
    from .verify import STRESS_ALLOWANCE

    os.makedirs(out_dir, exist_ok=True)
    name = name or study.name
    mat = study.material
    goal_sf = study.goal.safety_factor
    nodes = np.asarray(fields["nodes"], np.float64)
    cases = fields.get("cases") or []
    files, out_fields, objects = [], [], []
    pitch = fields.get("pitch")
    method = "cut-cell hex8" if fields.get("solidShare") is not None else "voxel hex8"
    source = {"analysis": f"loadpath verify_part, {method} linear static finite elements",
              "pitch": pitch, "nodes": int(len(nodes)),
              "elements": int(len(fields["elements"])) if fields.get("elements") is not None else None,
              "stressAllowance": STRESS_ALLOWANCE,
              "note": "nodal values of the same solve the part was verified on; the reported safety factor "
                      "also reads finer submodels at the hotspots, so a map's least value can sit above it"}

    def fid(q, case=None):
        return f"{q}/{case}" if case is not None else q

    # ---- per-node fields (analysis.vtu) -------------------------------------------------
    pd, envelope, env_case = {}, None, None
    sf_read = {}
    case_meta = []
    for ci, c in enumerate(cases):
        cname = c["name"]
        crep = next((r for r in report.get("cases") or [] if r.get("name") == cname), {})
        lc = next((x for x in study.load_cases if x.name == cname), None)
        need = (lc.safety_factor if lc is not None and lc.safety_factor else goal_sf)
        body = np.asarray(c.get("body"), bool) if c.get("body") is not None else np.ones(len(nodes), bool)
        vm = np.asarray(c["vonMises"], np.float64)
        disp = np.asarray(c["displacement"], np.float64)
        pd[f"equivalentStress {cname}"] = vm
        pd[f"displacement {cname}"] = disp
        if c.get("stress") is not None:
            pd[f"stress {cname}"] = np.asarray(c["stress"])
        out_fields.append({
            "id": fid("equivalentStress", cname), "quantity": "equivalent stress", "case": cname, "units": "MPa",
            "criterion": (report.get("failure") or {}).get("criterion", "von Mises"),
            "limit": mat.strength, "limitMeaning": f"{mat.name} tensile strength (MPa)", "betterIs": "lower",
            "range": _rng(vm, body, nodes, "max"),
            "reason": "what loads the material in this case; read against the strength to give the safety factor",
            "excludes": "the reported peak leaves out nodes within the contact band of a load or support "
                        "(contactBand = 1), where a point load's stress is a modelling artefact"})
        out_fields.append({
            "id": fid("displacement", cname), "quantity": "displacement", "case": cname, "units": "mm",
            "components": "x, y, z", "betterIs": "lower",
            "range": _rng(np.linalg.norm(disp, axis=1), None, nodes, "max"),
            "target": study.goal.max_deflection,
            "reason": "how far each point moves under the case's loads (linear, small strain)"})
        if c.get("safetyFactor") is not None:
            sf = np.asarray(c["safetyFactor"], np.float64)
            lim = np.asarray(c.get("limitedBy", np.zeros(len(sf))), np.int8)
            pd[f"safetyFactor {cname}"] = _finite(sf)
            pd[f"limitedBy {cname}"] = lim
            cf = float(c.get("caseFactor", 1.0))
            # inside the contact band (where a point load's or a clamp's
            # stress is the model's, not the part's) the reading is the
            # nearest node's outside it, as the reported number leaves it out
            sf_read[ci] = _from_body(sf, body, nodes)
            masked = sf_read[ci] * cf
            if envelope is None:
                envelope, env_case = masked.copy(), np.zeros(len(sf), np.int8)
            else:
                env_case[masked < envelope] = ci
                envelope = np.minimum(envelope, masked)
            r = _rng(sf, body, nodes, "min")
            kinds = sorted({int(v) for v in np.unique(lim[body])}) if body.any() else []
            out_fields.append({
                "id": fid("safetyFactor", cname), "quantity": "safety factor", "case": cname, "units": "1",
                "criterion": _criterion_text(report, study), "target": need, "betterIs": "higher",
                "range": r, "reported": crep.get("safetyFactor"),
                "limitedBy": {str(k): _LIMITED[k] for k in kinds},
                "cap": f"values over {SF_CAP:g} (nodes that carry almost nothing) are written as {SF_CAP:g}",
                "reason": "how many times this case's loads the part carries at each point before the "
                          "material (or the bond between layers) reaches its strength",
                "excludes": "nodes in the contact band (contactBand = 1) are not part of the reading"})
            case_meta.append({"case": cname, "index": ci, "requiredSafetyFactor": need, "caseFactor": cf})
            pd[f"contactBand {cname}"] = (~body).astype(np.int8)
    if envelope is not None:
        pd["safetyFactor governing"] = _finite(envelope)
        pd["governingCase"] = env_case
        out_fields.append({
            "id": "safetyFactor/governing", "quantity": "safety factor", "case": "all (governing)", "units": "1",
            "criterion": _criterion_text(report, study) + "; the least over the load cases, each scaled to the "
                                                          "goal's target by its own required safety factor",
            "target": goal_sf, "betterIs": "higher", "range": _rng(envelope, np.isfinite(envelope), nodes, "min"),
            "reported": report.get("safetyFactor"),
            "reason": "the part's margin at every point: where it would fail first, and by how much it clears the "
                      "goal there", "cases": [c["name"] for c in cases],
            "contactBand": "inside a case's contact band the value is the nearest node's outside it, as the reported "
                           "safety factor leaves the band out; 'safetyFactor <case>' in analysis.vtu keeps the raw "
                           "values, with 'contactBand <case>' marking them"})
    if fields.get("temperature") is not None:
        t = np.asarray(fields["temperature"], np.float64)
        pd["temperature"] = t
        out_fields.append({
            "id": "temperature", "quantity": "temperature", "units": "C", "betterIs": "lower",
            "range": _rng(t, None, nodes, "max"),
            "limit": getattr(mat, "max_service_temperature", None),
            "reason": "the solid's temperature at the hot moment of the study's thermal block (the cycle's "
                      "hottest, or the steady state); strength is read at it"})
    if fields.get("elements") is not None:
        hexes = np.asarray(fields["elements"])[:, _VTK_HEX]
        cd = {}
        if fields.get("solidShare") is not None:
            cd["solidShare"] = np.asarray(fields["solidShare"], np.float32)
        write_vtu(os.path.join(out_dir, "analysis.vtu"), nodes, hexes, pd, cd)
        files.append({"path": "analysis.vtu", "format": "VTK XML unstructured grid (hexahedra)",
                      "opensIn": _OPENS["vtu"],
                      "contains": "the analysis cells with every per-node field (arrays named '<field id> <case>'), "
                                  "and solidShare per cell (the share of a cut cell the part fills)"})

    # ---- the surface: glb + vtp ----------------------------------------------------------
    import trimesh
    m = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(*mesh)
    idx, w = _interp(nodes, m.vertices)
    attrs, sdata = {}, {}
    attr_meaning = {}
    if envelope is not None:
        s_env = np.minimum(_on(idx, w, _finite(envelope)), SF_CAP)
        attrs["_SAFETY_FACTOR"] = s_env
        sdata["safetyFactor governing"] = s_env
        attr_meaning["_SAFETY_FACTOR"] = "safetyFactor/governing"
        attr_meaning["COLOR_0"] = "safetyFactor/governing, by colorScale"
    for ci, c in enumerate(cases):
        vm_s = _on(idx, w, c["vonMises"])
        d_s = _on(idx, w, c["displacement"])
        attrs[f"_STRESS_{ci}"] = vm_s
        attrs[f"_DISPLACEMENT_{ci}"] = d_s
        attr_meaning[f"_STRESS_{ci}"] = fid("equivalentStress", c["name"])
        attr_meaning[f"_DISPLACEMENT_{ci}"] = fid("displacement", c["name"])
        sdata[f"equivalentStress {c['name']}"] = vm_s
        sdata[f"displacement {c['name']}"] = d_s
        if ci in sf_read:
            s_c = np.minimum(_on(idx, w, _finite(sf_read[ci])), SF_CAP)
            attrs[f"_SAFETY_FACTOR_{ci}"] = s_c
            attr_meaning[f"_SAFETY_FACTOR_{ci}"] = fid("safetyFactor", c["name"])
            sdata[f"safetyFactor {c['name']}"] = s_c
    if fields.get("temperature") is not None:
        t_s = _on(idx, w, fields["temperature"])
        attrs["_TEMPERATURE"] = t_s
        attr_meaning["_TEMPERATURE"] = "temperature"
        sdata["temperature"] = t_s
    colors = sf_colors(attrs["_SAFETY_FACTOR"], goal_sf) if "_SAFETY_FACTOR" in attrs else None

    size = float(np.ptp(np.asarray(m.vertices), axis=0).max()) or 1.0
    glb_objects = []
    for r in study.keep_out:
        try:
            km = _manifold_mesh(r.shape)
        except Exception:  # noqa: BLE001 - context only
            continue
        why = _keep_out_reason(r)
        ex = {"loadpath": {"role": "keepOut", "id": r.id, "reason": why}}
        glb_objects.append({"name": f"keep-out {r.id}", "mesh": km, "color": "#e0533d", "alpha": 0.25,
                            "group": "keep-outs", "extras": ex})
        objects.append({"id": f"keepOut/{r.id}", "role": "keep-out", "node": f"keep-out {r.id}",
                        "reason": why, "bounds": _jsonable(np.array(km.bounds))})
    for r in study.preserve:
        try:
            pm = _manifold_mesh(r.shape)
        except Exception:  # noqa: BLE001
            continue
        why = "drawn by the designer and kept as drawn: an interface the part must have (a boss, a face, a seat)"
        glb_objects.append({"name": f"preserve {r.id}", "mesh": pm, "color": "#8a94a6", "alpha": 0.3,
                            "group": "preserved", "extras": {"loadpath": {"role": "preserve", "id": r.id,
                                                                          "reason": why}}})
        objects.append({"id": f"preserve/{r.id}", "role": "preserved", "node": f"preserve {r.id}", "reason": why})
    reactions = {r.get("support"): r for r in report.get("reactions") or []}
    for i, s in enumerate(study.supports):
        try:
            shape = s.shape if s.shape is not None else study.region_shape(s.region)[0]
            sm = _manifold_mesh(shape)
        except Exception:  # noqa: BLE001
            continue
        label = s.region or f"support {i + 1}"
        held = "a pin: held radially and along its axis, free to turn" if s.pin else \
            "held in " + ", ".join(a for a, f in zip("xyz", s.fix) if f)
        rr = next((v for k, v in reactions.items() if k and (k == label or k.endswith(label))), None)
        ex = {"loadpath": {"role": "support", "id": label, "held": held, "maxForce": s.max_force,
                           "reaction": rr}}
        glb_objects.append({"name": f"support {label}", "mesh": sm, "color": "#2f9e6f", "alpha": 0.45,
                            "group": "supports", "extras": ex})
        objects.append({"id": f"support/{label}", "role": "support", "node": f"support {label}", "held": held,
                        "ratedForce": s.max_force, "reaction": rr,
                        "reason": "where the part is held by what it mounts to; reactions are what it passes on"})
    for case in study.load_cases:
        for li, ld in enumerate(case.loads):
            try:
                shape = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
            except ValueError:
                continue
            lo, hi = shape.bounds()
            at = 0.5 * (np.asarray(lo) + np.asarray(hi))
            f = np.asarray(ld.force, float)
            for kind, vec, unit in (("force", f, "N"), ("moment", np.asarray(ld.moment, float), "N·mm")):
                mag = float(np.linalg.norm(vec))
                if mag <= 0:
                    continue
                nm = f"{kind} {case.name} {ld.region or li + 1}"
                ex = {"loadpath": {"role": kind, "case": case.name, "region": ld.region, "vector": vec.tolist(),
                                   "magnitude": mag, "units": unit, "distribution": ld.distribution,
                                   "drawnLength": "display only: a quarter of the part's size, not to scale",
                                   **({"pressure": ld.pressure} if ld.pressure else {}),
                                   **({"impactScale": study.impact_scale.get(case.name)}
                                      if getattr(study, "impact_scale", None) else {})}}
                # a force arrow ends at its point of action; a moment's starts there
                d = vec / mag
                L = 0.25 * size
                start = at - d * L if kind == "force" else at
                glb_objects.append({"name": nm, "arrow": {"at": start.tolist(), "dir": d.tolist(), "length": L,
                                                          "radius": L / 40},
                                    "color": "#2f5bd3" if kind == "force" else "#8b5cf6", "group": "loads",
                                    "extras": ex})
                objects.append({"id": f"{kind}/{case.name}/{ld.region or li + 1}", "role": kind, "node": nm,
                                "case": case.name, "at": at.tolist(), "vector": vec.tolist(), "magnitude": mag,
                                "units": unit, "reason": f"applied in load case {case.name!r}"
                                + (f" ({case.cycles:g} cycles)" if case.cycles else "")})
    if envelope is not None and np.isfinite(envelope).any():
        wi = int(np.argmin(np.where(np.isfinite(envelope), envelope, np.inf)))
        wcase = cases[int(env_case[wi])]["name"]
        glb_objects.append({"name": "weakest point", "point": nodes[wi].tolist(), "radius": max(0.01 * size, 0.5),
                            "color": "#111827", "group": "findings",
                            "extras": {"loadpath": {"role": "weakestPoint", "safetyFactor": float(envelope[wi]),
                                                    "case": wcase}}})
        objects.append({"id": "weakestPoint", "role": "weakest point", "node": "weakest point",
                        "at": nodes[wi].tolist(), "safetyFactor": float(envelope[wi]), "case": wcase,
                        "reason": "the least governing safety factor on the field map (the refined reading in "
                                  "verdict.safetyFactor can be lower)"})
    write_glb(os.path.join(out_dir, "part.glb"), m, attrs, colors, glb_objects,
              part_extras={"loadpath": {"attributes": attr_meaning, "manifest": "manifest.json"}}, name=name)
    files.append({"path": "part.glb", "format": "glTF 2.0 binary", "opensIn": _OPENS["glb"],
                  "contains": "the part coloured by its governing safety factor, the fields as vertex attributes, "
                              "keep-outs, preserved regions, supports, loads and the weakest point as named nodes",
                  "attributes": attr_meaning})
    write_vtp(os.path.join(out_dir, "surface.vtp"), m.vertices, m.faces, sdata)
    files.append({"path": "surface.vtp", "format": "VTK XML polydata", "opensIn": _OPENS["vtp"],
                  "contains": "the delivered surface with the fields read onto it"})
    if stl and os.path.exists(stl):
        import shutil
        shutil.copyfile(stl, os.path.join(out_dir, "part.stl"))
        files.append({"path": "part.stl", "format": "STL", "opensIn": _OPENS["stl"],
                      "contains": "the part to make"})

    # ---- flows ---------------------------------------------------------------------------
    for fl in flows or []:
        files.extend(fl["files"])
        out_fields.extend(fl["fields"])

    # ---- tradeoff --------------------------------------------------------------------------
    tables = []
    outs = (summary or {}).get("outcomes") or []
    if outs:
        cols = [("index", "1", "candidate number"), ("volumeFraction", "1", "share of the design space filled"),
                ("mass", "g", "part mass"), ("effectiveMass", "g", "mass plus what its drag in a rotor's stream "
                                                                    "costs, when the study has rotors"),
                ("safetyFactor", "1", "governing safety factor (verify_part)"),
                ("maxDeflection", "mm", "largest deflection at the loads over the cases"),
                ("bucklingFactor", "1", "load multiple at which it buckles (linear)"),
                ("firstFrequency", "Hz", "first natural frequency, when the study asks for one"),
                ("passes", "bool", "meets every goal"), ("best", "bool", "the delivered candidate"),
                ("stl", "path", "the candidate's mesh")]
        with open(os.path.join(out_dir, "tradeoff.csv"), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow([c[0] for c in cols])
            for i, o in enumerate(outs):
                r = o.get("report") or o
                defl = max((c.get("deflection") or 0.0 for c in r.get("cases") or []), default=None)
                freq = (r.get("modal") or {}).get("frequency") if isinstance(r.get("modal"), dict) else None
                row = [i, o.get("volumeFraction"), r.get("mass"), r.get("effectiveMass"), r.get("safetyFactor"),
                       defl, r.get("bucklingFactor"), freq, o.get("passes", r.get("passes")),
                       i == (summary or {}).get("best"), os.path.basename(o.get("stl") or "")]
                wr.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v) for v in row])
        tables.append({"path": "tradeoff.csv", "rows": len(outs), "reason": "every candidate the search verified: "
                       "the mass against margin curve the delivered part was picked from",
                       "columns": [{"name": n, "units": u, "meaning": t} for n, u, t in cols]})
        files.append({"path": "tradeoff.csv", "format": "CSV", "opensIn": _OPENS["csv"],
                      "contains": "mass, margin and stiffness of every candidate"})
    if steps:
        with open(os.path.join(out_dir, "steps.json"), "w") as fh:
            json.dump(_jsonable(steps), fh, indent=1)
        files.append({"path": "steps.json", "format": "JSON", "opensIn": _OPENS["json"],
                      "contains": "the numbered assembly moves (what goes in, from where, along which path)"})
    for ef in extra_files or []:
        files.append(ef)

    dv = (summary or {}).get("deliver") or {}
    verdict = {"passes": dv.get("passes", report.get("passes")), "safetyFactor": report.get("safetyFactor"),
               "safetyFactorMethod": dv.get("safetyFactorMethod"), "governedBy": dv.get("governedBy"),
               "target": goal_sf, "maxDeflection": study.goal.max_deflection,
               "mass": report.get("mass"), "massReduction": report.get("massReduction"),
               "bucklingFactor": report.get("bucklingFactor"),
               "p5Member": dv.get("p5Member", report.get("p5Member")),
               "warnings": list(dv.get("warnings") or report.get("warnings") or [])[:12]}
    from importlib.metadata import PackageNotFoundError, version
    try:
        ver = version("loadpath")
    except PackageNotFoundError:
        ver = None
    proc = study.process
    manifest = {
        "loadpath": {"bundle": BUNDLE_VERSION, "version": ver, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "part": {"name": name, "material": {"name": mat.name, "strength": mat.strength, "E": mat.E,
                                            "density": mat.density, "units": {"strength": "MPa", "E": "MPa",
                                                                              "density": "g/cm^3"}},
                 "process": None if proc is None else {"kind": getattr(proc, "kind", None),
                                                       "build": _jsonable(getattr(proc, "build", None)),
                                                       "interlayer": getattr(proc, "interlayer", None)},
                 "mass": report.get("mass"), "volume": report.get("volume"),
                 "loadCases": [{"name": c.name, "cycles": c.cycles, "hours": c.hours, "temperature": c.temperature,
                                "requiredSafetyFactor": c.safety_factor or goal_sf} for c in study.load_cases]},
        "units": {"length": "mm", "force": "N", "moment": "N·mm", "stress": "MPa", "temperature": "C",
                  "mass": "g", "velocity": "m/s", "pressure": "Pa"},
        "frame": "loadpath's coordinates: mm, z up. part.glb's root node scales to metres and turns to y up.",
        "verdict": verdict,
        "source": source,
        "fields": out_fields,
        "objects": objects,
        "tables": tables,
        "steps": "steps.json" if steps else None,
        "colorScale": {"field": "safetyFactor/governing", "stops": sf_scale(goal_sf),
                       "note": "a display choice: dark red under 1 (fails), red to amber up to the target, green at "
                               "twice it, grey where material carries little"},
        "files": files,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(_jsonable(manifest), fh, indent=1)
    return _jsonable(manifest)


def flow_grids(out_dir, name, fl: dict, what: str, medium: str | None = None) -> dict:
    """A simulated flow (flow.network_flow(..., fields=True)["fields"]:
    velocity m/s, pressure Pa on a voxel grid, voxel (0,0,0) at `origin`,
    size `h` mm) as `<name>.vti` (velocity, speed, pressure, fluid mask)
    and `<name>.vdb` (speed and pressure as fog volumes). Returns
    {files, fields} entries for export_bundle(flows=...)."""
    from .vdb import write_vdb

    os.makedirs(out_dir, exist_ok=True)
    U = np.asarray(fl["velocity"], np.float32)
    p = np.asarray(fl["pressure"], np.float32)
    fluid = np.asarray(fl["fluid"], bool)
    h = float(fl["h"])
    origin = np.asarray(fl.get("origin", (0, 0, 0)), float)
    speed = np.where(fluid, np.linalg.norm(U, axis=-1), 0.0).astype(np.float32)
    pdata = {"velocity": np.where(fluid[..., None], U, 0.0), "speed": speed,
             "pressure": np.where(fluid, p, 0.0), "fluid": fluid}
    if fl.get("temperature") is not None:
        pdata["temperature"] = np.asarray(fl["temperature"], np.float32)
    write_vti(os.path.join(out_dir, f"{name}.vti"), origin, h, pdata)
    grids = [{"name": "speed", "values": speed, "origin": origin, "pitch": h, "class": "fog volume"},
             {"name": "pressure", "values": np.where(fluid, p, 0.0).astype(np.float32), "origin": origin, "pitch": h,
              "class": "fog volume"}]
    if "temperature" in pdata:
        grids.append({"name": "temperature", "values": np.where(fluid, pdata["temperature"], 0.0), "origin": origin,
                      "pitch": h, "class": "fog volume"})
    write_vdb(os.path.join(out_dir, f"{name}.vdb"), grids)
    src = f"loadpath lattice-Boltzmann flow (D3Q19), {h:g} mm voxels, time-averaged"
    flds = [{"id": f"velocity/{name}", "quantity": "velocity", "units": "m/s", "components": "x, y, z",
             "flow": name, "range": _rng(speed[fluid]), "source": src,
             "reason": f"how {medium or 'the fluid'} moves through {what}: where it runs fast, slow or not at all"},
            {"id": f"pressure/{name}", "quantity": "pressure", "units": "Pa", "flow": name,
             "range": _rng(p[fluid]), "source": src, "betterIs": "lower drop",
             "reason": "gauge pressure against the outlet: its drop along the path is what a pump or fan pays"}]
    if "temperature" in pdata:
        flds.append({"id": f"temperature/{name}", "quantity": "temperature", "units": "C", "flow": name,
                     "range": _rng(pdata["temperature"][fluid]), "source": src,
                     "reason": "the fluid's temperature: what heat it picked up or gave"})
    files = [{"path": f"{name}.vti", "format": "VTK XML image data", "opensIn": _OPENS["vti"],
              "contains": f"{what}: velocity (vector), speed, pressure, fluid mask on the flow grid"},
             {"path": f"{name}.vdb", "format": "OpenVDB", "opensIn": _OPENS["vdb"],
              "contains": f"{what}: speed and pressure as fog volumes (mm voxels)"}]
    return {"files": files, "fields": flds}


def exchanger_grids(out_dir, sim_fields: dict, name: str = "exchanger") -> dict:
    """A whole-exchanger porous simulation (porous.simulate_exchanger(...,
    fields=True)["_fields"]) as `<name>.vti`: per fluid its superficial
    velocity at cell centres (m/s), pressure (Pa), temperature (C) and the
    cells it flows in, and the wall's temperature; and `<name>.vdb` with
    each fluid's speed and temperature and the wall's as fog volumes.
    Returns {files, fields} for export_bundle(flows=...)."""
    from .porous import _shift
    from .vdb import write_vdb

    os.makedirs(out_dir, exist_ok=True)
    dom, flows, heat = sim_fields["domain"], sim_fields["flows"], sim_fields.get("heat")
    hmm = float(dom.h) * 1e3
    origin = np.asarray(dom.lo, float) + 0.5 * hmm
    pdata, grids, flds = {}, [], []
    src = (f"loadpath porous-medium exchanger model (Darcy-Forchheimer flow, three-temperature heat), "
           f"{hmm:g} mm cells: the whole part, ports to ports")
    for f, fname in enumerate(dom.names):
        cells = np.asarray(dom.cells[f], bool)
        U = flows[f]["U"]
        v = np.stack([0.5 * (U[ax] + _shift(U[ax], -1, ax)) for ax in range(3)], axis=-1) * cells[..., None]
        sp = np.linalg.norm(v, axis=-1).astype(np.float32)
        pdata[f"velocity {fname}"] = v.astype(np.float32)
        pdata[f"pressure {fname}"] = np.where(cells, flows[f]["p"], 0.0).astype(np.float32)
        pdata[f"cells {fname}"] = cells
        grids.append({"name": f"speed {fname}", "values": sp, "origin": origin, "pitch": hmm, "class": "fog volume"})
        flds.append({"id": f"velocity/{fname}", "quantity": "superficial velocity", "units": "m/s", "flow": fname,
                     "components": "x, y, z", "range": _rng(sp[cells]), "source": src,
                     "reason": f"how the {fname} side's fluid moves through the part: where it runs, where it "
                               "stalls and how evenly the core shares it"})
        dp = flows[f]["p"][cells]
        flds.append({"id": f"pressure/{fname}", "quantity": "pressure", "units": "Pa", "flow": fname,
                     "range": _rng(dp), "source": src, "betterIs": "lower drop",
                     "reason": "its drop from port to port is what the pump or fan pays"})
        if heat is not None:
            T = np.where(cells, heat["T"][f], 0.0).astype(np.float32)
            pdata[f"temperature {fname}"] = T
            grids.append({"name": f"temperature {fname}", "values": T, "origin": origin, "pitch": hmm,
                          "class": "fog volume"})
            flds.append({"id": f"temperature/{fname}", "quantity": "temperature", "units": "C", "flow": fname,
                         "range": _rng(np.asarray(heat["T"][f])[cells]), "source": src,
                         "reason": f"the {fname} side's temperature: the heat it takes up or gives along its path"})
    if heat is not None and heat.get("Twall") is not None:
        tw = np.asarray(heat["Twall"], np.float32)
        pdata["wall temperature"] = tw
        grids.append({"name": "wall temperature", "values": tw, "origin": origin, "pitch": hmm, "class": "fog volume"})
        flds.append({"id": "temperature/wall", "quantity": "temperature", "units": "C", "flow": "wall",
                     "range": _rng(tw[np.asarray(dom.core, bool)]) if getattr(dom, "core", None) is not None
                     else _rng(tw), "source": src,
                     "reason": "the wall between the fluids: what the material has to stand, and where heat crosses"})
    write_vti(os.path.join(out_dir, f"{name}.vti"), origin, hmm, pdata)
    write_vdb(os.path.join(out_dir, f"{name}.vdb"), grids)
    files = [{"path": f"{name}.vti", "format": "VTK XML image data", "opensIn": _OPENS["vti"],
              "contains": "each fluid's velocity, pressure, temperature and cells, and the wall's temperature"},
             {"path": f"{name}.vdb", "format": "OpenVDB", "opensIn": _OPENS["vdb"],
              "contains": "each fluid's speed and temperature, and the wall's, as fog volumes (mm voxels)"}]
    return {"files": files, "fields": flds}
