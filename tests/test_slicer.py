"""
Tests for the slicer in the FDM loop: G-code reading, the 3MF modifier
config, stress-graded infill sizing, and (when PrusaSlicer is installed) a
real slice. Run: `python3 tests/test_slicer.py`.
"""
import os
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.infill import GradeOptions, grade_infill  # noqa: E402
from loadpath.slicer import SlicerOptions, find_slicer, read_gcode, slice_mesh  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.threemf import write_3mf  # noqa: E402


def _box(lo, hi):
    b = trimesh.creation.box(extents=np.subtract(hi, lo))
    b.apply_translation(np.add(lo, hi) / 2)
    return b


_CANTILEVER = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [80, 16, 16]},
    "supports": [{"shape": {"type": "box", "min": [0, 0, 0], "max": [1, 16, 16]}, "fix": "xyz"}],
    "loads": [{"shape": {"type": "box", "min": [77, 0, 0], "max": [80, 16, 16]}, "force": [0, 0, -60]}],
    "material": "petg",
    "process": {"type": "fdm", "buildDirection": "+y"},
    "goal": {"safetyFactor": 3.0},
}


def test_read_gcode_sums_features():
    """Relative E per ;TYPE: block, support split out, header numbers read."""
    g = "\n".join([
        "M83", ";TYPE:External perimeter", "G1 X1 Y1 E2.0", "G1 X2 Y1 E1.0", "G1 X3 E-0.8",
        ";TYPE:Support material", "G1 X4 E3.0", ";TYPE:Internal infill", "G1 X5 E4.0",
        "; filament used [mm] = 10.00", "; filament used [cm3] = 0.02",
        "; estimated printing time (normal mode) = 1h 2m 3s", "; filament_diameter = 1.75",
        "; layer_height = 0.2", ""])
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.gcode")
        open(p, "w").write(g)
        r = read_gcode(p, density=1.27)
    area = np.pi * 0.875 ** 2 / 1000.0
    assert r["printTime"] == 3723.0
    assert abs(r["supportVolume"] - 3.0 * area) < 1e-9
    assert abs(r["features"]["External perimeter"] - round(3.0 * area, 4)) < 1e-9
    assert abs(r["filamentMass"] - 0.02 * 1.27) < 1e-12 and r["layerHeight"] == 0.2


def test_3mf_carries_modifiers():
    """Object settings and one <volume> per triangle range, part first."""
    part = _box((0, 0, 0), (20, 20, 10))
    mod = _box((0, 0, 0), (10, 20, 10))
    with tempfile.TemporaryDirectory() as tmp:
        p = write_3mf(os.path.join(tmp, "g.3mf"), part, (2, 1), name="g", object_settings={"fill_density": "10%"},
                      modifiers=[{"mesh": mod, "settings": {"fill_density": "60%", "perimeters": "5"}}])
        with zipfile.ZipFile(p) as z:
            cfg = ET.fromstring(z.read("Metadata/Slic3r_PE_model.config"))
            model = ET.fromstring(z.read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    tris = model.findall(".//m:triangle", ns)
    assert len(tris) == len(part.faces) + len(mod.faces)
    obj = cfg.find("object")
    assert {m.get("key"): m.get("value") for m in obj.findall("metadata")}["fill_density"] == "10%"
    vols = obj.findall("volume")
    assert [(int(v.get("firstid")), int(v.get("lastid"))) for v in vols] == \
        [(0, len(part.faces) - 1), (len(part.faces), len(tris) - 1)]
    kv = {m.get("key"): m.get("value") for m in vols[1].findall("metadata")}
    assert kv["volume_type"] == "ParameterModifier" and kv["fill_density"] == "60%" and kv["perimeters"] == "5"
    # no settings, no Prusa metadata: plain core 3MF as before
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(write_3mf(os.path.join(tmp, "p.3mf"), part, (2, 1))) as z:
            assert "Metadata/Slic3r_PE_model.config" not in z.namelist()


def test_spec_and_options():
    """slicer is FDM-only, goal.minimize is checked, grade options parse."""
    s = study_from_dict(dict(_CANTILEVER, slicer={"grade": {"levels": [[20, 3], [10, 2]]}},
                             goal={"safetyFactor": 2, "minimize": "printTime"}))
    assert s.goal.minimize == "printTime" and s.slicer.grade is not None
    assert GradeOptions.from_spec(s.slicer.grade).levels == ((10.0, 2), (20.0, 3))
    for bad in ({"slicer": True, "process": {"type": "sls"}},
                {"goal": {"safetyFactor": 2, "minimize": "speed"}},
                {"slicer": {"enabled": False}, "goal": {"safetyFactor": 2, "minimize": "printTime"}},
                {"slicer": {"grade": {"levels": [[120, 3]]}}}):
        try:
            study_from_dict(dict(_CANTILEVER, **bad))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert study_from_dict(dict(_CANTILEVER, process={"type": "sls"})).slicer is None
    o = SlicerOptions.from_spec({"infill": 100, "supports": False, "settings": {"layer_height": 0.3}})
    cfg = o.config()
    assert cfg["fill_pattern"] == "rectilinear" and cfg["support_material"] == "0" and cfg["layer_height"] == "0.3"


def test_grade_puts_density_where_the_stress_is():
    """A cantilever: the root (high bending stress) grades denser than the
    tip, the graded part keeps the target safety factor on the model, and
    the modifiers are closed meshes."""
    study = study_from_dict(_CANTILEVER)
    mesh = _box((0, 0, 0), (80, 16, 16))
    o = GradeOptions(safety_factor=3.0, analysis_cells=6000, collar=0.0)
    g = grade_infill(study, mesh, {"safetyFactor": 6.0, "cases": [{"deflection": 0.1}]}, o)
    assert g["predicted"]["graded"] >= 0.97 * 3.0, g["predicted"]
    assert g["predicted"]["graded"] < g["predicted"]["solid"]
    assert g["modifiers"] and all(m["mesh"].is_watertight for m in g["modifiers"])
    # the densest modifier sits toward the root
    dense = g["modifiers"][-1]["mesh"]
    assert dense.centroid[0] < 40.0, dense.centroid
    assert g["estimatedVolume"] < 0.9 * mesh.volume
    assert g["deflection"][0] > 0.1 and g["safetyFactor"] < 6.0


def test_slice_real_part():
    """With PrusaSlicer installed: an overhanging T needs support standing
    up and none on its side; a dense modifier adds filament."""
    if find_slicer() is None:
        print("  (skipped: no PrusaSlicer)")
        return
    from loadpath.materials import MATERIALS

    t = trimesh.util.concatenate([_box((-20, -2.5, 0), (-15, 2.5, 25)), _box((-20, -2.5, 20), (20, 2.5, 25))])
    up = slice_mesh(t, (2, 1), MATERIALS["petg"])
    side = slice_mesh(t, (0, 1), MATERIALS["petg"])
    assert up["supportVolume"] > 0.3 and side["supportVolume"] < 0.05, (up["supportVolume"], side["supportVolume"])
    assert up["printTime"] > 0 and up["filamentVolume"] > t.volume / 1000.0
    cube = _box((0, 0, 0), (30, 30, 15))
    sparse = slice_mesh(cube, (2, 1), None, SlicerOptions(infill=10))
    mod = slice_mesh(cube, (2, 1), None, SlicerOptions(infill=10), object_settings={"fill_density": "10%"},
                     modifiers=[{"mesh": _box((0, 0, 0), (15, 30, 15)), "settings": {"fill_density": "60%"}}])
    assert mod["filamentVolume"] > 1.3 * sparse["filamentVolume"], (mod["filamentVolume"], sparse["filamentVolume"])


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


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
