"""
Slicer-ready output: a 3MF with the part already standing the way it was
designed to print.

An STL has no orientation of its own, so a person drops it in a slicer and
lays it flat on whatever face the slicer guesses, which for a part designed
self-supporting along one direction can be the one way it isn't. The 3MF
keeps the part in its design coordinates (so it still lines up with the
assembly) and puts the print orientation in the build item's transform: the
build direction to +z, the lowest point on the bed, centred on the plate.
PrusaSlicer, OrcaSlicer, Bambu Studio and Cura all honour build transforms.

Core 3MF (3D Manufacturing Format, core spec 1.3): a model part, mm, plus
metadata naming the build direction, material and process. With
`object_settings` or `modifiers` it also carries PrusaSlicer's per-object
config and modifier volumes (Metadata/Slic3r_PE_model.config): extra meshes
in the same object that change print settings (infill density, perimeters)
wherever they overlap the part. That is how graded infill (infill.py)
reaches the slicer; PrusaSlicer reads it, and Orca Slicer and Bambu Studio
import PrusaSlicer 3MFs with their modifiers.
"""
from __future__ import annotations

import io
import os
import re
import uuid
import zipfile
from types import SimpleNamespace
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from .am import direction_label
from .printcheck import rotation_to_z

__all__ = ["write_3mf", "print_transform", "write_multimaterial_3mf", "write_material_stls"]

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""
_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def print_transform(mesh, direction, plate_center=(128.0, 128.0)) -> np.ndarray:
    """4x4 (column-vector) transform from design coordinates to the print
    bed: `direction` to +z, lowest point at z = 0, footprint centred on
    `plate_center` (mm)."""
    rot = rotation_to_z(direction)
    pts = np.asarray(mesh.vertices) @ rot.T
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    t = np.array([plate_center[0] - 0.5 * (lo[0] + hi[0]), plate_center[1] - 0.5 * (lo[1] + hi[1]), -lo[2]])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = t
    return m


def _fmt(v):
    return f"{v:.6g}"


def write_3mf(path, mesh, direction=None, name: str = "part", material=None, process=None,
              plate_center=(128.0, 128.0), metadata: dict | None = None,
              object_settings: dict | None = None, modifiers: list | None = None) -> str:
    """Write `mesh` (trimesh, mm) to `path` as 3MF, placed to print along
    `direction` ((axis, sign), a vector, or None for as-is with the lowest
    point on the bed). `object_settings` ({PrusaSlicer key: value}) apply to
    the whole part; each of `modifiers` ({"mesh", "settings", "name"}, in
    design coordinates like the part) overrides them where it overlaps the
    part, later ones winning. Returns the path."""
    direction = direction if direction is not None else (2, 1)
    tf = print_transform(mesh, direction, plate_center)
    vols = [(name, mesh, None)] + [(m.get("name") or f"modifier {i + 1}", m["mesh"], m.get("settings") or {})
                                   for i, m in enumerate(modifiers or [])]
    vs, fs, ranges, off = [], [], [], 0
    for _, m, _ in vols:
        fi = np.asarray(m.faces, np.int64)
        start = sum(len(a) for a in fs)
        vs.append(np.asarray(m.vertices, float))
        fs.append(fi + off)
        ranges.append((start, start + len(fi) - 1))
        off += len(m.vertices)
    v, f = np.vstack(vs), np.vstack(fs)
    prusa = bool(object_settings) or bool(modifiers)
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    buf.write('<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
              + (' xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06"' if prusa else '') + '>\n')
    if prusa:
        buf.write('  <metadata name="slic3rpe:Version3mf">1</metadata>\n')
    meta = {"Title": name, "Application": "loadpath",
            "loadpath:buildDirection": direction_label(direction)}
    if material is not None:
        meta["loadpath:material"] = getattr(material, "name", str(material))
    if process is not None:
        meta["loadpath:process"] = getattr(process, "kind", str(process))
    meta.update(metadata or {})
    for k, val in meta.items():
        buf.write(f'  <metadata name="{escape(str(k))}">{escape(str(val))}</metadata>\n')
    buf.write('  <resources>\n')
    buf.write(f'    <object id="1" type="model" name="{escape(name)}">\n      <mesh>\n        <vertices>\n')
    buf.write("".join(f'          <vertex x="{_fmt(a)}" y="{_fmt(b)}" z="{_fmt(c)}"/>\n' for a, b, c in v))
    buf.write('        </vertices>\n        <triangles>\n')
    buf.write("".join(f'          <triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in f))
    buf.write('        </triangles>\n      </mesh>\n    </object>\n  </resources>\n')
    # 3MF transforms row vectors: [x y z 1] . M, M 4x3 = (R^T ; t)
    rows = np.vstack([tf[:3, :3].T, tf[:3, 3]])
    buf.write(f'  <build>\n    <item objectid="1" transform="{" ".join(_fmt(x) for x in rows.ravel())}"/>\n'
              '  </build>\n</model>\n')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("3D/3dmodel.model", buf.getvalue())
        if prusa:
            z.writestr("Metadata/Slic3r_PE_model.config", _prusa_config(name, vols, ranges, object_settings))
    return str(path)


def _prusa_config(name, vols, ranges, object_settings) -> str:
    """PrusaSlicer's per-object config: object settings, then one <volume>
    per triangle range, the first the part and the rest modifiers."""
    def meta(kind, k, val):
        return f'    <metadata type="{kind}" key="{escape(str(k))}" value="{escape(str(val))}"/>\n'

    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n <object id="1" instances_count="1">\n',
           meta("object", "name", name)]
    out += [meta("object", k, val) for k, val in (object_settings or {}).items()]
    for (vname, _, settings), (a, b) in zip(vols, ranges):
        out.append(f'  <volume firstid="{a}" lastid="{b}">\n')
        out.append(meta("volume", "name", vname))
        if settings is None:
            out.append(meta("volume", "volume_type", "ModelPart"))
        else:
            out.append(meta("volume", "modifier", 1))
            out.append(meta("volume", "volume_type", "ParameterModifier"))
            out += [meta("volume", k, val) for k, val in settings.items()]
        out.append('  </volume>\n')
    out.append(' </object>\n</config>\n')
    return "".join(out)


# ---------------------------------------------------------------------------
# Multi-material: several touching bodies, each its own material / extruder
# ---------------------------------------------------------------------------

# display colours by material name (lower case); soft/flexible materials dark,
# rigid ones light, so a PETG + TPU part reads at a glance in any viewer
_MATERIAL_COLORS = {
    "tpu": "#3C3C3C", "tpe": "#4A4A4A", "flex": "#3C3C3C",
    "petg": "#E6E9EC", "pla": "#F3F1EA", "abs": "#D9D4C7", "asa": "#CFCAB8",
    "pa12": "#BDB8AE", "pa11": "#C4BCA8", "pa-cf": "#2B2B2B", "pc": "#DCE3EA",
    "support": "#F5E6B8", "pva": "#F5E6B8",
}
# fallback colours, in order, for materials without one (or a clash)
_PALETTE = ["#E6E9EC", "#3C3C3C", "#D9534F", "#2E86DE", "#F0AD4E", "#5CB85C", "#8E44AD", "#16A085"]

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_PROD_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
_BAMBU_NS = "http://schemas.bambulab.com/package/2021"
_MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_BAMBU_OBJECT_FILE = "3D/Objects/object_1.model"
_IDENTITY_3MF = "1 0 0 0 1 0 0 0 1 0 0 0"


def _material_name(m) -> str:
    return str(getattr(m, "name", m))


def _resolve_bodies(bodies):
    """Normalise the `bodies` list: name, mesh, material name, 1-based
    extruder, #RRGGBB colour, settings. Extruders default to the order in
    which each distinct material first appears (skipping numbers given
    explicitly); colours default to `_MATERIAL_COLORS`, then `_PALETTE`."""
    if not bodies:
        raise ValueError("write_multimaterial_3mf needs at least one body")
    out, mat_ext, used = [], {}, set()
    for b in bodies:
        if b.get("extruder") is not None:
            e = int(b["extruder"])
            if e < 1:
                raise ValueError(f"extruder must be 1-based, got {e}")
            used.add(e)
    for i, b in enumerate(bodies):
        mesh = b["mesh"]
        if len(getattr(mesh, "faces", ())) == 0:
            raise ValueError(f"body {i} has an empty mesh")
        mat = _material_name(b.get("material", "material"))
        if b.get("extruder") is not None:
            ext = int(b["extruder"])
            mat_ext.setdefault(mat, ext)
        elif mat in mat_ext:
            ext = mat_ext[mat]
        else:
            ext = 1
            while ext in used:
                ext += 1
            used.add(ext)
            mat_ext[mat] = ext
        out.append({"name": str(b.get("name") or f"body {i + 1}"), "mesh": mesh, "material": mat,
                    "extruder": ext, "color": b.get("color"), "settings": dict(b.get("settings") or {})})
    # one colour per material: explicit first, then by name, then palette
    colors, taken = {}, set()
    for b in out:
        if b["color"] and b["material"] not in colors:
            colors[b["material"]] = b["color"].upper()
            taken.add(colors[b["material"]])
    for b in out:
        m = b["material"]
        if m not in colors:
            c = _MATERIAL_COLORS.get(m.lower())
            if c is None or c in taken:
                c = next((p for p in _PALETTE if p not in taken), _PALETTE[len(colors) % len(_PALETTE)])
            colors[m] = c
            taken.add(c)
    for b in out:
        b["color"] = (b["color"] or colors[b["material"]]).upper()
        if not re.fullmatch(r"#[0-9A-F]{6}([0-9A-F]{2})?", b["color"]):
            raise ValueError(f"color must be #RRGGBB, got {b['color']!r}")
    materials = []                       # distinct, first-appearance order
    for b in out:
        if b["material"] not in [m for m, _ in materials]:
            materials.append((b["material"], colors[b["material"]]))
    index = {m: k for k, (m, _) in enumerate(materials)}
    for b in out:
        b["pindex"] = index[b["material"]]
    return out, materials


def _assembly_transform(bodies, direction, plate_center):
    """One print transform for the whole assembly (union of every body's
    vertices), so the bodies stay registered and the lowest point of the
    assembly sits on the bed."""
    allv = np.vstack([np.asarray(b["mesh"].vertices, float) for b in bodies])
    return print_transform(SimpleNamespace(vertices=allv), direction, plate_center)


def _rows(tf):
    # 3MF transforms row vectors: [x y z 1] . M, M 4x3 = (R^T ; t)
    return " ".join(_fmt(x) for x in np.vstack([tf[:3, :3].T, tf[:3, 3]]).ravel())


def _mesh_xml(v, f, pid=None, pindex=None, indent="      "):
    tri = (f'{indent}    <triangle v1="{{}}" v2="{{}}" v3="{{}}" pid="{pid}" p1="{{}}"/>\n' if pid is not None
           else f'{indent}    <triangle v1="{{}}" v2="{{}}" v3="{{}}"/>\n')
    parts = [f"{indent}<mesh>\n{indent}  <vertices>\n"]
    parts.append("".join(f'{indent}    <vertex x="{_fmt(a)}" y="{_fmt(b)}" z="{_fmt(c)}"/>\n' for a, b, c in v))
    parts.append(f"{indent}  </vertices>\n{indent}  <triangles>\n")
    if pid is None:
        parts.append("".join(tri.format(a, b, c) for a, b, c in f))
    else:
        parts.append("".join(tri.format(a, b, c, k) for (a, b, c), k in zip(f, pindex)))
    parts.append(f"{indent}  </triangles>\n{indent}</mesh>\n")
    return "".join(parts)


def _basematerials_xml(mid, materials, indent="    "):
    out = [f'{indent}<basematerials id="{mid}">\n']
    out += [f'{indent}  <base name={quoteattr(m)} displaycolor="{c}"/>\n' for m, c in materials]
    out.append(f"{indent}</basematerials>\n")
    return "".join(out)


def _model_meta(name, direction, bodies, materials, process, metadata, application):
    meta = {"Title": name, "Application": application,
            "loadpath:buildDirection": direction_label(direction),
            "loadpath:materials": ", ".join(m for m, _ in materials)}
    for b in bodies:
        meta[f"loadpath:material.{b['name']}"] = b["material"]
    if process is not None:
        meta["loadpath:process"] = getattr(process, "kind", str(process))
    meta.update(metadata or {})
    return "".join(f'  <metadata name={quoteattr(str(k))}>{escape(str(val))}</metadata>\n' for k, val in meta.items())


def _uuid(*parts) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "loadpath:" + "/".join(map(str, parts))))


def write_multimaterial_3mf(path, bodies, direction=None, name: str = "part", process=None,
                            plate_center=(128.0, 128.0), metadata: dict | None = None,
                            slicer: str = "all") -> str:
    """Write a part made of several material bodies to `path` as one 3MF
    object with one part (volume) per body, each assigned its own extruder,
    for multi-material FDM (Bambu AMS, Prusa MMU/XL, IDEX: e.g. a PETG core
    with a TPU pad) or multi-material jetting.

    `bodies`: list of {"name", "mesh" (trimesh, mm, design coordinates; the
    bodies touch along shared interfaces and must not overlap), "material"
    (a materials.Material or a name), "extruder" (1-based; default: the
    order in which each distinct material first appears), "color"
    ("#RRGGBB"; default by material name, e.g. TPU dark grey, PETG light),
    "settings" (optional {slicer key: value} for that body only)}.

    All bodies share ONE print transform (`print_transform` of the union of
    their vertices, `direction` as in `write_3mf`), carried by the single
    build item, so they stay registered and the assembly's lowest point
    sits on the bed. Every file also carries a core-3MF `<basematerials>`
    group (one `<base>` per material, with its display colour) referenced
    by pid/pindex, so any 3MF viewer shows the materials, and model
    metadata `loadpath:material.<body name>` per body.

    `slicer` picks the slicer-specific layout, since no single file is read
    as a multi-part object by both families:

    - "prusa", "all" or "generic" (default): one mesh object holding every
      body, with Metadata/Slic3r_PE_model.config giving one
      `<volume firstid lastid>` per body (volume_type ModelPart, name,
      `extruder`, `settings`). PrusaSlicer reads it natively. Bambu Studio
      and OrcaSlicer only hand a 3MF to their PrusaSlicer importer (which
      reads that config) when the model's Application metadata contains
      "PrusaSlicer", so this layout says "loadpath (PrusaSlicer 3MF)"; their
      GUI opens it as one object with per-part filaments. (Their command
      line uses the native importer, which ignores Slic3r_PE_model.config:
      use "bambu" there.)
    - "bambu" (also "orca", "bambustudio", "orcaslicer"): the Bambu Studio
      layout (3MF production extension): 3D/3dmodel.model has one object
      whose `<components>` point (p:path) at one mesh object per body in
      3D/Objects/object_1.model, and Metadata/model_settings.config gives
      the object and each `<part id subtype="normal_part">` its name and
      `extruder` (plus `settings`, which must then be Bambu/Orca keys).
      PrusaSlicer does not read external p:path objects, so it can't open
      this one.

    Returns the path."""
    kind = str(slicer).lower()
    if kind in ("all", "generic", "prusa", "prusaslicer"):
        bambu = False
    elif kind in ("bambu", "bambustudio", "bambu studio", "orca", "orcaslicer"):
        bambu = True
    else:
        raise ValueError(f"slicer must be 'all', 'generic', 'prusa' or 'bambu', got {slicer!r}")
    direction = direction if direction is not None else (2, 1)
    bodies, materials = _resolve_bodies(bodies)
    tf = _assembly_transform(bodies, direction, plate_center)
    if bambu:
        files = _bambu_layout(name, direction, bodies, materials, process, metadata, tf)
    else:
        files = _prusa_layout(name, direction, bodies, materials, process, metadata, tf)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, text in files:
            z.writestr(arc, text)
    return str(path)


def _prusa_layout(name, direction, bodies, materials, process, metadata, tf):
    vs, fs, ranges, pidx, off, start = [], [], [], [], 0, 0
    for b in bodies:
        fi = np.asarray(b["mesh"].faces, np.int64)
        vs.append(np.asarray(b["mesh"].vertices, float))
        fs.append(fi + off)
        pidx.append(np.full(len(fi), b["pindex"]))
        ranges.append((start, start + len(fi) - 1))
        start += len(fi)
        off += len(b["mesh"].vertices)
    v, f, pi = np.vstack(vs), np.vstack(fs), np.concatenate(pidx)
    mat_id, obj_id = 1, 2
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    buf.write(f'<model unit="millimeter" xml:lang="en-US" xmlns="{_CORE_NS}" '
              'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n')
    buf.write('  <metadata name="slic3rpe:Version3mf">1</metadata>\n')
    buf.write(_model_meta(name, direction, bodies, materials, process, metadata, "loadpath (PrusaSlicer 3MF)"))
    buf.write("  <resources>\n")
    buf.write(_basematerials_xml(mat_id, materials))
    buf.write(f'    <object id="{obj_id}" type="model" name={quoteattr(name)} pid="{mat_id}" '
              f'pindex="{bodies[0]["pindex"]}">\n')
    buf.write(_mesh_xml(v, f, mat_id, pi))
    buf.write("    </object>\n  </resources>\n")
    buf.write(f'  <build>\n    <item objectid="{obj_id}" transform="{_rows(tf)}"/>\n  </build>\n</model>\n')

    def meta(kind, k, val):
        return f'    <metadata type="{kind}" key={quoteattr(str(k))} value={quoteattr(str(val))}/>\n'

    cfg = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n',
           f' <object id="{obj_id}" instances_count="1">\n',
           meta("object", "name", name), meta("object", "extruder", bodies[0]["extruder"])]
    for b, (a, z) in zip(bodies, ranges):
        cfg.append(f'  <volume firstid="{a}" lastid="{z}">\n')
        cfg.append(meta("volume", "name", b["name"]))
        cfg.append(meta("volume", "volume_type", "ModelPart"))
        cfg.append(meta("volume", "extruder", b["extruder"]))
        cfg += [meta("volume", k, val) for k, val in b["settings"].items() if k != "extruder"]
        cfg.append("  </volume>\n")
    cfg.append(" </object>\n</config>\n")
    return [("[Content_Types].xml", _CONTENT_TYPES), ("_rels/.rels", _RELS),
            ("3D/3dmodel.model", buf.getvalue()), ("Metadata/Slic3r_PE_model.config", "".join(cfg))]


def _bambu_layout(name, direction, bodies, materials, process, metadata, tf):
    n = len(bodies)
    mat_id, asm_id = n + 1, n + 2        # mesh objects are 1..n (Bambu's part ids)
    sub = io.StringIO()
    sub.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    sub.write(f'<model unit="millimeter" xml:lang="en-US" xmlns="{_CORE_NS}" '
              f'xmlns:BambuStudio="{_BAMBU_NS}" xmlns:p="{_PROD_NS}" requiredextensions="p">\n')
    sub.write('  <metadata name="BambuStudio:3mfVersion">1</metadata>\n')
    sub.write("  <resources>\n")
    sub.write(_basematerials_xml(mat_id, materials))
    for k, b in enumerate(bodies, start=1):
        sub.write(f'    <object id="{k}" p:UUID="{_uuid(name, "object", k)}" type="model" '
                  f'name={quoteattr(b["name"])} pid="{mat_id}" pindex="{b["pindex"]}">\n')
        sub.write('      <metadatagroup>\n'
                  f'        <metadata name="loadpath:material">{escape(b["material"])}</metadata>\n'
                  '      </metadatagroup>\n')
        sub.write(_mesh_xml(np.asarray(b["mesh"].vertices, float), np.asarray(b["mesh"].faces, np.int64)))
        sub.write("    </object>\n")
    sub.write("  </resources>\n  <build/>\n</model>\n")

    top = io.StringIO()
    top.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    top.write(f'<model unit="millimeter" xml:lang="en-US" xmlns="{_CORE_NS}" '
              f'xmlns:BambuStudio="{_BAMBU_NS}" xmlns:p="{_PROD_NS}" requiredextensions="p">\n')
    top.write('  <metadata name="BambuStudio:3mfVersion">1</metadata>\n')
    top.write(_model_meta(name, direction, bodies, materials, process, metadata, "loadpath"))
    top.write("  <resources>\n")
    top.write(f'    <object id="{asm_id}" p:UUID="{_uuid(name, "assembly")}" type="model" name={quoteattr(name)}>\n'
              "      <components>\n")
    for k in range(1, n + 1):
        top.write(f'        <component p:path="/{_BAMBU_OBJECT_FILE}" objectid="{k}" '
                  f'p:UUID="{_uuid(name, "component", k)}" transform="{_IDENTITY_3MF}"/>\n')
    top.write("      </components>\n    </object>\n  </resources>\n")
    top.write(f'  <build p:UUID="{_uuid(name, "build")}">\n'
              f'    <item objectid="{asm_id}" p:UUID="{_uuid(name, "item")}" transform="{_rows(tf)}" printable="1"/>\n'
              "  </build>\n</model>\n")

    def meta(k, val, ind="    "):
        return f'{ind}<metadata key={quoteattr(str(k))} value={quoteattr(str(val))}/>\n'

    cfg = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n', f'  <object id="{asm_id}">\n',
           meta("name", name), meta("extruder", bodies[0]["extruder"])]
    for k, b in enumerate(bodies, start=1):
        cfg.append(f'    <part id="{k}" subtype="normal_part">\n')
        cfg.append(meta("name", b["name"], "      "))
        cfg.append(meta("matrix", "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1", "      "))
        cfg.append(meta("extruder", b["extruder"], "      "))
        cfg += [meta(key, val, "      ") for key, val in b["settings"].items() if key != "extruder"]
        cfg.append(f'      <mesh_stat face_count="{len(b["mesh"].faces)}" edges_fixed="0" degenerate_facets="0" '
                   'facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n')
        cfg.append("    </part>\n")
    cfg.append("  </object>\n</config>\n")
    sub_rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                f' <Relationship Target="/{_BAMBU_OBJECT_FILE}" Id="rel-1" Type="{_MODEL_REL}"/>\n'
                '</Relationships>\n')
    return [("[Content_Types].xml", _CONTENT_TYPES), ("_rels/.rels", _RELS),
            ("3D/3dmodel.model", top.getvalue()), ("3D/_rels/3dmodel.model.rels", sub_rels),
            (_BAMBU_OBJECT_FILE, sub.getvalue()), ("Metadata/model_settings.config", "".join(cfg))]


def write_material_stls(directory, bodies, direction=None, name: str = "part",
                        plate_center=(128.0, 128.0)) -> list:
    """Write one STL per material (all bodies of that material merged) into
    `directory`, for slicers that take separate files per extruder. The
    meshes are already placed on the bed with the same shared print
    transform as `write_multimaterial_3mf` (print_transform of the union of
    all bodies), so loaded together, without auto-arranging or re-centring
    (e.g. answer "multi-part object" in PrusaSlicer/Bambu when asked), they
    re-register. Files are `<name>_<material>_T<extruder>.stl` (T is
    1-based). Returns the paths, in extruder order."""
    import trimesh

    direction = direction if direction is not None else (2, 1)
    bodies, _ = _resolve_bodies(bodies)
    tf = _assembly_transform(bodies, direction, plate_center)
    groups = {}
    for b in bodies:
        groups.setdefault((b["extruder"], b["material"]), []).append(b["mesh"])
    os.makedirs(directory, exist_ok=True)
    safe = (lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "x")
    paths = []
    for (ext, mat), meshes in sorted(groups.items()):
        m = trimesh.util.concatenate([trimesh.Trimesh(np.asarray(x.vertices, float), np.asarray(x.faces), process=False)
                                      for x in meshes])
        m.apply_transform(tf)
        p = os.path.join(str(directory), f"{safe(name)}_{safe(mat)}_T{ext}.stl")
        m.export(p)
        paths.append(p)
    return paths
