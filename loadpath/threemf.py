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

Core 3MF (3D Manufacturing Format, core spec 1.3) only: a model part, mm,
plus metadata naming the build direction, material and process.
"""
from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

import numpy as np

from .am import direction_label
from .printcheck import rotation_to_z

__all__ = ["write_3mf", "print_transform"]

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
              plate_center=(128.0, 128.0), metadata: dict | None = None) -> str:
    """Write `mesh` (trimesh, mm) to `path` as 3MF, placed to print along
    `direction` ((axis, sign), a vector, or None for as-is with the lowest
    point on the bed). Returns the path."""
    direction = direction if direction is not None else (2, 1)
    tf = print_transform(mesh, direction, plate_center)
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, np.int64)
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    buf.write('<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n')
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
    return str(path)
