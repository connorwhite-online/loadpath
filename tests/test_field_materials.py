"""
Tests for material bodies in field graphs (fields.material_bodies): a part
split into bodies by solids and field thresholds, earlier bodies first,
sharing their interfaces exactly; the per-body STLs, the multi-material
3MF, the mix body's voxel job, and the report. Small grids; about half a
minute. Run: `python -m pytest tests/test_field_materials.py`.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import trimesh  # noqa: E402

from loadpath.fields import FieldGraph, material_bodies, run_field  # noqa: E402
from loadpath.schema import lint  # noqa: E402

BOX = {"op": "shape", "shape": {"type": "box", "min": [-10, -5, -5], "max": [10, 5, 5]}}


def _graph(materials, **extra):
    return {"name": "t", "nodes": {"part": BOX, "zc": {"op": "coordinate", "axis": "z"},
                                   "xc": {"op": "coordinate", "axis": "x"}, **extra},
            "output": "part", "mesh": {"pitch": 0.5}, "materials": materials}


def _manifold(m):
    from loadpath.shapes import manifold_from_trimesh

    return manifold_from_trimesh(m)


def _bodies(doc):
    g = FieldGraph(doc)
    part = g.mesh()
    pieces, rest, _ = material_bodies(g, part)
    return part, {b["name"]: m for b, m in pieces}, rest


def _check_tiling(part, meshes):
    """Bodies are watertight, sum to the part and do not overlap; returns
    {(a, b): interface area}."""
    ms = {k: m for k, m in meshes.items() if m is not None}
    for k, m in ms.items():
        assert m.is_watertight, k
    assert abs(sum(m.volume for m in ms.values()) / part.volume - 1) < 0.005
    names = list(ms)
    areas = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = _manifold(ms[a]), _manifold(ms[b])
            assert (A ^ B).volume() < 1e-3 * part.volume, (a, b)
            areas[a, b] = 0.5 * (A.surface_area() + B.surface_area() - (A + B).surface_area())
    return areas


def test_threshold_body_and_rest_share_an_exact_interface():
    """TPU where z > 3 in a 20 x 10 x 10 box: a 2 mm slab of 400 mm^3, its
    interface the 200 mm^2 plane, the core the other 1600 mm^3, both
    watertight; the two bodies' interface vertices are the same points."""
    part, ms, rest = _bodies(_graph([{"name": "pad", "material": "tpu", "where": "zc", "above": 3},
                                     {"name": "core", "material": "petg"}]))
    assert rest.is_empty()
    assert abs(ms["pad"].volume - 400) < 0.5 and abs(ms["core"].volume - 1600) < 0.5
    areas = _check_tiling(part, ms)
    assert abs(areas["pad", "core"] - 200) < 0.5, areas
    pad = ms["pad"].vertices
    on = pad[np.abs(pad[:, 2] - 3) < 1e-6]
    assert len(on) >= 4
    core = {tuple(v) for v in np.asarray(ms["core"].vertices)}
    assert all(tuple(v) in core for v in on)


def test_precedence_earlier_bodies_win():
    """Two overlapping solids: the first takes x < 0, the second (x < 5)
    only what is left, 0 < x < 5, and a third with no region the rest."""
    a = {"op": "shape", "shape": {"type": "box", "min": [-20, -9, -9], "max": [0, 9, 9]}}
    b = {"op": "shape", "shape": {"type": "box", "min": [-20, -9, -9], "max": [5, 9, 9]}}
    part, ms, rest = _bodies(_graph([{"name": "a", "material": "petg", "solid": a},
                                     {"name": "b", "material": "tpu", "solid": b},
                                     {"name": "c", "material": "pla"}]))
    vol = {k: m.volume for k, m in ms.items()}
    assert abs(vol["a"] - 1000) < 1 and abs(vol["b"] - 500) < 1 and abs(vol["c"] - 500) < 1, vol
    areas = _check_tiling(part, ms)
    assert abs(areas["a", "b"] - 100) < 0.5 and abs(areas["b", "c"] - 100) < 0.5
    assert areas.get(("a", "c"), 0) < 0.5


def test_field_threshold_and_non_distance_field():
    """`below` as a field (a tilted plane z < 0.2 x), and a threshold on a
    field that is no distance (x^2 < 4 from an expr): the volumes follow."""
    part, ms, _ = _bodies(_graph([{"name": "tilt", "material": "tpu", "where": "zc",
                                   "below": {"op": "expr", "expr": "0.2 * x"}},
                                  {"name": "rest", "material": "petg"}]))
    assert abs(ms["tilt"].volume - 1000) < 2, ms["tilt"].volume   # antisymmetric: half the box
    _check_tiling(part, ms)
    part, ms, _ = _bodies(_graph([{"name": "mid", "material": "tpu", "where": "sq", "below": 4},
                                  {"name": "rest", "material": "petg"}],
                                 sq={"op": "expr", "expr": "x * x"}))
    assert abs(ms["mid"].volume / 400 - 1) < 0.01, ms["mid"].volume
    _check_tiling(part, ms)


def test_shell_body_coincident_with_the_part_surface():
    """A body whose solid is a shell of the part (its outer wall lies on
    the part's surface): the region is carried on outside the part, so the
    split leaves no slivers: a 1 mm skin of a 10 mm sphere, then the core."""
    doc = {"nodes": {"part": {"op": "shape", "shape": {"type": "sphere", "center": [0, 0, 0], "r": 10}},
                     "skin": {"op": "shell", "input": "part", "thickness": 1}},
           "output": "part", "mesh": {"pitch": 0.5},
           "materials": [{"name": "skin", "material": "tpu", "solid": "skin"}, {"name": "core", "material": "petg"}]}
    part, ms, _ = _bodies(doc)
    want = 4 / 3 * np.pi * (1000 - 729)
    assert abs(ms["skin"].volume / want - 1) < 0.005, ms["skin"].volume
    areas = _check_tiling(part, ms)
    assert abs(areas["skin", "core"] / (4 * np.pi * 81) - 1) < 0.005, areas
    assert len(ms["core"].split(only_watertight=False)) == 1
    assert len(ms["skin"].split(only_watertight=False)) == 2      # its outer and inner wall, no slivers


def test_run_field_writes_bodies_report_and_3mf(tmp_path):
    doc = _graph([{"name": "pad", "material": "tpu", "where": "zc", "above": 3},
                  {"name": "half", "material": "pla", "solid": {"op": "shape", "shape": {
                      "type": "box", "min": [0, -6, -6], "max": [11, 6, 6]}}}])
    _, rep = run_field(doc, str(tmp_path), render=False)
    m = rep["materials"]
    vols = {b["name"]: b["volume"] for b in m["bodies"]}
    assert abs(vols["pad"] - 400) < 0.5 and abs(vols["half"] - 800) < 0.5
    assert abs(m["check"]["unassignedVolume"] - 800) < 0.5 and m["warnings"]
    pad = m["bodies"][0]
    assert abs(pad["mass"] - 400 * 1.21 / 1000) < 1e-3 and pad["watertight"]
    assert abs(rep["mass"] - m["mass"]) < 1e-12
    inter = {tuple(i["bodies"]): i["area"] for i in m["interfaces"]}
    assert abs(inter["pad", "half"] - 100) < 0.5
    for b in m["bodies"]:
        assert trimesh.load(b["stl"]).is_watertight
    if not isinstance(m["3mf"], str):
        pytest.skip(f"multi-material 3MF not written: {m['3mf']}")
    import zipfile

    z = zipfile.ZipFile(m["3mf"])
    model = z.read("3D/3dmodel.model").decode()
    assert "pad" in model or any("pad" in z.read(n).decode(errors="ignore") for n in z.namelist())


def test_mix_body_mass_and_voxel_job(tmp_path):
    """A 0..1 ramp of agilus in vero across x weighs the mean of the two
    densities, and goes out as a voxel jetting job when voxelprint is there."""
    doc = _graph([{"name": "grad", "material": ["vero", "agilus30"], "mix": {"op": "remap", "input": "xc",
                                                                            "from": [-10, 10], "to": [0, 1]},
                   "voxel": {"pitch": 0.5, "maxLayers": 2}}])
    _, rep = run_field(doc, str(tmp_path), render=False)
    b = rep["materials"]["bodies"][0]
    from loadpath.materials import material_from_spec

    rho = 0.5 * (material_from_spec("vero").density + material_from_spec("agilus30").density)
    assert abs(b["mix"]["meanFraction"] - 0.5) < 0.01
    assert abs(b["mass"] - 2000 * rho / 1000) < 0.01
    job = b["mix"].get("voxelJob", {})
    if "skipped" in job:
        pytest.skip(job["skipped"])
    assert "error" not in job, job
    assert job["fileCount"] >= 1


def test_material_bodies_are_validated():
    for bad, msg in (([{"name": "a", "material": "tpu", "where": "zc"}], "above"),
                     ([{"name": "a", "material": "tpu", "below": 2}], "where"),
                     ([{"name": "a", "material": "tpu", "mix": 0.5}], "pair"),
                     ([{"name": "a", "material": "tpu"}, {"name": "a", "material": "pla"}], "two bodies"),
                     ([{"name": "a", "material": "tpu", "solid": "nope"}], "not a node")):
        with pytest.raises(ValueError, match=msg):
            FieldGraph(_graph(bad))
    issues = lint(_graph([{"name": "a", "material": "tpu", "within": "part", "where": "zc", "below": 1}]), "field")
    assert any("solid" in i["message"] for i in issues), issues


def test_gripper_example(tmp_path):
    """examples/fields/gripper_pad.json lints clean and splits into a TPU pad
    about 2.5 mm thick on the gripping face and a PETG core (coarse here)."""
    doc = json.loads((ROOT / "examples" / "fields" / "gripper_pad.json").read_text())
    assert lint(doc, "field") == []
    doc["mesh"] = {"pitch": 0.6}
    _, rep = run_field(doc, str(tmp_path), base_dir=str(ROOT / "examples" / "fields"), render=False)
    m = rep["materials"]
    pad, core = m["bodies"]
    assert pad["material"] == "tpu" and core["material"] == "petg"
    assert abs(m["check"]["volumeError"]) < 0.005 and m["check"]["maxOverlap"] < 1.0
    assert 900 < pad["volume"] < 1400 and m["interfaces"][0]["area"] > 450
    lo, hi = trimesh.load(pad["stl"]).bounds
    assert abs((hi[1] - lo[1]) - 2.5) < 0.05
