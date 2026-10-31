"""
Multi-material slicer export (threemf.write_multimaterial_3mf and
write_material_stls): a PETG core with a TPU pad on top. Structure checks
for both layouts, plus a real PrusaSlicer slice in a two-extruder config
when PrusaSlicer is installed. Run: `pytest tests/test_threemf_multi.py`.
"""
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.materials import MATERIALS  # noqa: E402
from loadpath.threemf import write_3mf, write_material_stls, write_multimaterial_3mf  # noqa: E402

CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
PROD = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}"


def _box(lo, hi):
    b = trimesh.creation.box(extents=np.subtract(hi, lo))
    b.apply_translation(np.add(lo, hi) / 2)
    return b


def _bodies(**extra):
    core = {"name": "core", "mesh": _box([0, 0, 0], [20, 20, 10]), "material": MATERIALS["petg"]}
    pad = {"name": "pad", "mesh": _box([0, 0, 10], [20, 20, 13]), "material": "tpu"}
    pad.update(extra)
    return [core, pad]


def _xf(s):
    """3MF transform string -> 4x4 column-vector matrix."""
    a = np.array([float(x) for x in s.split()]).reshape(4, 3)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = a[:3].T, a[3]
    return m


def _apply(m, v):
    return v @ m[:3, :3].T + m[:3, 3]


def test_prusa_layout_volumes_materials_and_placement(tmp_path):
    p = write_multimaterial_3mf(tmp_path / "mm.3mf", _bodies(settings={"perimeters": 4}), direction=(2, 1))
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        assert {"3D/3dmodel.model", "Metadata/Slic3r_PE_model.config", "[Content_Types].xml", "_rels/.rels"} <= names
        model = ET.fromstring(z.read("3D/3dmodel.model"))
        cfg = ET.fromstring(z.read("Metadata/Slic3r_PE_model.config"))
    meta = {m.get("name"): m.text for m in model.iter(CORE + "metadata")}
    assert "PrusaSlicer" in meta["Application"]   # what Orca / Bambu Studio route on
    assert meta["loadpath:material.core"] == "petg" and meta["loadpath:material.pad"] == "tpu"
    bases = [(b.get("name"), b.get("displaycolor")) for b in model.iter(CORE + "base")]
    assert [n for n, _ in bases] == ["petg", "tpu"] and bases[0][1] != bases[1][1]
    (obj,) = list(model.iter(CORE + "object"))
    mid = next(model.iter(CORE + "basematerials")).get("id")
    tris = list(obj.iter(CORE + "triangle"))
    assert len(tris) == 24 and all(t.get("pid") == mid for t in tris)
    assert [t.get("p1") for t in tris] == ["0"] * 12 + ["1"] * 12
    vols = cfg.findall("./object/volume")
    assert cfg.find("./object").get("id") == obj.get("id")
    assert [(v.get("firstid"), v.get("lastid")) for v in vols] == [("0", "11"), ("12", "23")]
    vm = [{m.get("key"): m.get("value") for m in v.findall("metadata")} for v in vols]
    assert [m["name"] for m in vm] == ["core", "pad"]
    assert [m["extruder"] for m in vm] == ["1", "2"]
    assert all(m["volume_type"] == "ModelPart" for m in vm)
    assert vm[1]["perimeters"] == "4" and "perimeters" not in vm[0]
    # the triangle ranges really are the bodies: pad triangles all at z >= 10
    verts = np.array([[float(v.get(a)) for a in "xyz"] for v in obj.iter(CORE + "vertex")])
    idx = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in tris])
    assert verts[idx[12:]][..., 2].min() >= 10 - 1e-9 and verts[idx[:12]][..., 2].max() <= 10 + 1e-9
    # one shared build transform: lowest point of the assembly on the bed, centred
    (item,) = list(model.iter(CORE + "item"))
    placed = _apply(_xf(item.get("transform")), verts)
    assert placed[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert placed[:, :2].min(0) + placed[:, :2].max(0) == pytest.approx([256.0, 256.0])
    # trimesh reads it back: both bodies, 10 mm core + 3 mm pad
    scene = trimesh.load(p, force="scene")
    assert sum(len(g.faces) for g in scene.geometry.values()) == 24
    assert scene.bounds[1][2] - scene.bounds[0][2] == pytest.approx(13.0)


def test_flipped_direction_keeps_bodies_registered(tmp_path):
    p = write_multimaterial_3mf(tmp_path / "f.3mf", _bodies(), direction=(2, -1))
    with zipfile.ZipFile(p) as z:
        model = ET.fromstring(z.read("3D/3dmodel.model"))
    verts = np.array([[float(v.get(a)) for a in "xyz"] for v in model.iter(CORE + "vertex")])
    placed = _apply(_xf(next(model.iter(CORE + "item")).get("transform")), verts)
    # printed pad-down: the pad (vertices 8..15) is now at the bottom, on the bed
    assert placed[8:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert placed[8:, 2].max() == pytest.approx(3.0) and placed[:8, 2].min() == pytest.approx(3.0)


def test_extruder_and_color_defaults():
    from loadpath.threemf import _resolve_bodies
    m = _box([0, 0, 0], [1, 1, 1])
    bodies, mats = _resolve_bodies([
        {"mesh": m, "material": "tpu"}, {"mesh": m, "material": "petg", "extruder": 1},
        {"mesh": m, "material": "tpu"}, {"mesh": m, "material": "pla", "color": "#ff0000"}])
    assert [b["extruder"] for b in bodies] == [2, 1, 2, 3]
    assert [m for m, _ in mats] == ["tpu", "petg", "pla"]
    assert dict(mats)["tpu"] == "#3C3C3C" and dict(mats)["pla"] == "#FF0000"
    assert [b["pindex"] for b in bodies] == [0, 1, 0, 2]
    with pytest.raises(ValueError):
        _resolve_bodies([{"mesh": m, "material": "x", "extruder": 0}])
    with pytest.raises(ValueError):
        write_multimaterial_3mf("/nonexistent/x.3mf", [{"mesh": m}], slicer="cura")


def test_bambu_layout_structure(tmp_path):
    p = write_multimaterial_3mf(tmp_path / "b.3mf", _bodies(), slicer="bambu")
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        assert {"3D/3dmodel.model", "3D/_rels/3dmodel.model.rels", "3D/Objects/object_1.model",
                "Metadata/model_settings.config"} <= names
        top = ET.fromstring(z.read("3D/3dmodel.model"))
        sub = ET.fromstring(z.read("3D/Objects/object_1.model"))
        rels = ET.fromstring(z.read("3D/_rels/3dmodel.model.rels"))
        cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
        content = z.read("[Content_Types].xml").decode()
    assert "model" in content
    assert top.get("requiredextensions") == "p"
    targets = [r.get("Target") for r in rels]
    assert targets == ["/3D/Objects/object_1.model"]
    (asm,) = list(top.iter(CORE + "object"))
    comps = list(asm.iter(CORE + "component"))
    assert [c.get(PROD + "path") for c in comps] == targets * 2
    sub_objs = {o.get("id"): o for o in sub.iter(CORE + "object")}
    assert [c.get("objectid") for c in comps] == ["1", "2"] and set(sub_objs) == {"1", "2"}
    assert all(c.get(PROD + "UUID") for c in comps) and asm.get(PROD + "UUID")
    (item,) = list(top.iter(CORE + "item"))
    assert item.get("objectid") == asm.get("id") and item.get(PROD + "UUID")
    # each sub-object is one body, coloured by its material
    mid = next(sub.iter(CORE + "basematerials")).get("id")
    assert [(sub_objs[k].get("pid"), sub_objs[k].get("pindex")) for k in "12"] == [(mid, "0"), (mid, "1")]
    assert [len(list(sub_objs[k].iter(CORE + "triangle"))) for k in "12"] == [12, 12]
    zs = [float(v.get("z")) for v in sub_objs["2"].iter(CORE + "vertex")]
    assert min(zs) == pytest.approx(10.0)
    # model_settings.config: object = assembly id, parts = sub-object ids
    (o,) = cfg.findall("object")
    assert o.get("id") == asm.get("id")
    parts = o.findall("part")
    assert [(q.get("id"), q.get("subtype")) for q in parts] == [("1", "normal_part"), ("2", "normal_part")]
    pm = [{m.get("key"): m.get("value") for m in q.findall("metadata")} for q in parts]
    assert [m["name"] for m in pm] == ["core", "pad"] and [m["extruder"] for m in pm] == ["1", "2"]
    # same placement as the Prusa layout
    p2 = write_multimaterial_3mf(tmp_path / "p.3mf", _bodies(), slicer="prusa")
    with zipfile.ZipFile(p2) as z:
        t2 = next(ET.fromstring(z.read("3D/3dmodel.model")).iter(CORE + "item")).get("transform")
    assert item.get("transform") == t2


def test_material_stls_register(tmp_path):
    bodies = _bodies()
    paths = write_material_stls(tmp_path / "stl", bodies, direction=(2, 1), name="pad part")
    assert [os.path.basename(x) for x in paths] == ["pad_part_petg_T1.stl", "pad_part_tpu_T2.stl"]
    core, pad = (trimesh.load(x) for x in paths)
    assert core.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert pad.bounds[0][2] == pytest.approx(core.bounds[1][2])   # still touching
    assert np.allclose(core.bounds[:, :2], pad.bounds[:, :2])
    # same placement as the 3MF build transform
    p = write_multimaterial_3mf(tmp_path / "x.3mf", bodies, direction=(2, 1))
    with zipfile.ZipFile(p) as z:
        tf = _xf(next(ET.fromstring(z.read("3D/3dmodel.model")).iter(CORE + "item")).get("transform"))
    assert np.allclose(_apply(tf, bodies[1]["mesh"].bounds), pad.bounds, atol=1e-5)


def test_write_3mf_unchanged(tmp_path):
    with zipfile.ZipFile(write_3mf(tmp_path / "s.3mf", _box([0, 0, 0], [5, 5, 5]), (2, 1))) as z:
        assert sorted(z.namelist()) == ["3D/3dmodel.model", "[Content_Types].xml", "_rels/.rels"]
        assert b'name="Application">loadpath<' in z.read("3D/3dmodel.model")


_PRUSA = shutil.which("prusa-slicer") or shutil.which("PrusaSlicer")
_TWO_EXTRUDERS = """printer_technology = FFF
nozzle_diameter = 0.4,0.4
filament_diameter = 1.75,1.75
extruder_offset = 0x0,0x0
retract_length = 0.8,0.8
temperature = 240,225
first_layer_temperature = 240,225
bed_shape = 0x0,256x0,256x256,0x256
layer_height = 0.2
first_layer_height = 0.2
use_relative_e_distances = 1
wipe_tower = 1
"""


@pytest.mark.skipif(_PRUSA is None, reason="PrusaSlicer not installed")
def test_prusaslicer_slices_two_tools(tmp_path):
    p = write_multimaterial_3mf(tmp_path / "mm.3mf", _bodies())
    ini = tmp_path / "two.ini"
    ini.write_text(_TWO_EXTRUDERS)
    gcode = tmp_path / "mm.gcode"
    proc = subprocess.run([_PRUSA, "--export-gcode", "--load", str(ini), "--output", str(gcode), str(p)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and gcode.exists(), proc.stdout + proc.stderr
    z = tool = None
    tools_at = {}
    tools = set()
    for line in gcode.read_text().splitlines():
        m = re.match(r";Z:([\d.]+)", line)
        if m:
            z = float(m.group(1))
        m = re.match(r"T(\d+)\b", line)
        if m:
            tool = int(m.group(1))
            tools.add(tool)
        if line.startswith("G1") and " E" in line and "X" in line and z is not None and tool is not None:
            tools_at.setdefault(z, set()).add(tool)
    assert tools >= {0, 1}
    # outside the wipe tower each layer has one tool: core (z < 10) T0, pad T1
    assert all(0 in t for zz, t in tools_at.items() if zz < 9.9)
    pad_layers = [t for zz, t in tools_at.items() if zz > 10.1]
    assert pad_layers and all(t == {1} for t in pad_layers)
