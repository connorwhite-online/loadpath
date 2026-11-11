"""The results bundle (bundle.py): every file parses back to the values
the verification made, the glTF is well formed, the manifest meets its
schema, and the Materialize adapter builds a listing without uploading."""
import base64
import json
import os
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUDY = {
    "name": "lug",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z", "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]},
                  {"name": "push", "loads": [{"region": "pin", "force": [60, 0, 0]}]}],
    "material": "pa12",
    "process": {"type": "sls"},
    "goal": {"safetyFactor": 2.0},
    "resolution": {"pitch": 2.0},
}


def _part():
    import trimesh
    box = trimesh.creation.box(extents=[50, 30, 8])
    box.apply_translation([25, 0, 0])
    ring = trimesh.creation.cylinder(radius=9, height=8)
    ring.apply_translation([48, 0, 0])
    hole = trimesh.creation.cylinder(radius=4, height=12)
    hole.apply_translation([48, 0, 0])
    return trimesh.boolean.difference([trimesh.boolean.union([box, ring], engine="manifold"), hole],
                                      engine="manifold")


def _vtk_arrays(path):
    """{name: array} from a bundle VTK XML file (zlib blocks, UInt64 header)."""
    root = ET.parse(path).getroot()
    assert root.get("compressor") == "vtkZLibDataCompressor"
    out = {}
    dt = {"Float32": np.float32, "Int64": np.int64, "UInt8": np.uint8, "Int8": np.int8}
    for da in root.iter("DataArray"):
        text = da.text.strip()
        nb = struct.unpack("<Q", base64.b64decode(text[:12])[:8])[0]
        hlen = 8 * (3 + nb)
        hchars = 4 * ((hlen + 2) // 3)
        head = struct.unpack(f"<{3 + nb}Q", base64.b64decode(text[:hchars])[:hlen])
        data = base64.b64decode(text[hchars:])
        raw, off = b"", 0
        for size in head[3:]:
            raw += zlib.decompress(data[off:off + size])
            off += size
        a = np.frombuffer(raw, dt[da.get("type")])
        nc = int(da.get("NumberOfComponents", 1))
        out[da.get("Name")] = a.reshape(-1, nc) if nc > 1 else a
    return out


def _glb(path):
    b = Path(path).read_bytes()
    magic, ver, total = struct.unpack("<III", b[:12])
    assert magic == 0x46546C67 and ver == 2 and total == len(b)
    jl, jt = struct.unpack("<II", b[12:20])
    assert jt == 0x4E4F534A
    doc = json.loads(b[20:20 + jl])
    bl, bt = struct.unpack("<II", b[20 + jl:28 + jl])
    assert bt == 0x004E4942 and bl == doc["buffers"][0]["byteLength"]
    binary = b[28 + jl:28 + jl + bl]
    for acc in doc["accessors"]:
        bv = doc["bufferViews"][acc["bufferView"]]
        n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
        assert bv["byteOffset"] + bv["byteLength"] <= len(binary) and bv["byteLength"] == 4 * n * acc["count"]
        assert bv["byteOffset"] % 4 == 0
    return doc, binary


def test_bundle_round_trips_the_verification():
    from loadpath.adapters.materialize import listing
    from loadpath.bundle import SF_CAP, export_bundle
    from loadpath.schema import SCHEMAS
    from loadpath.spec import study_from_dict
    from loadpath.verify import verify_part

    st = study_from_dict(json.loads(json.dumps(STUDY)), base_dir=str(ROOT))
    mesh = _part()
    rep = verify_part(st, mesh, max_cells=12_000, refine=0, fields=True)
    f = rep["fields"]
    assert f["elements"].shape[1] == 8 and len(f["cases"]) == 2
    for c, rc in zip(f["cases"], rep["cases"]):
        # the map's least over the body is the case's reported safety factor
        # (refine=0: no submodels to read lower)
        sf = np.where(c["body"], c["safetyFactor"], np.inf)
        assert abs(sf.min() / rc["safetyFactor"] - 1) < 1e-6, (sf.min(), rc["safetyFactor"])
    summary = {"best": 0, "outcomes": [{"volumeFraction": 0.5, "report": rep, "stl": "lug.stl"}]}
    with tempfile.TemporaryDirectory() as d:
        mesh.export(os.path.join(d, "lug.stl"))
        man = export_bundle(os.path.join(d, "b"), st, mesh, rep, f, summary=summary, stl=os.path.join(d, "lug.stl"),
                            steps=[{"n": 1, "text": "slide the pin in along -z"}])
        b = os.path.join(d, "b")
        assert sorted(os.listdir(b)) == sorted(["analysis.vtu", "manifest.json", "part.glb", "part.stl",
                                                "steps.json", "surface.vtp", "tradeoff.csv"])
        try:
            import jsonschema
            errs = list(jsonschema.Draft202012Validator(SCHEMAS["bundle"]).iter_errors(
                json.load(open(os.path.join(b, "manifest.json")))))
            assert not errs, [e.message for e in errs]
        except ImportError:
            pass
        # the analysis grid: the verification's nodes and values, exactly (float32)
        a = _vtk_arrays(os.path.join(b, "analysis.vtu"))
        assert np.allclose(a["Points"], f["nodes"], atol=1e-4)
        assert len(a["types"]) == len(f["elements"]) and set(a["types"]) == {12}
        c0 = f["cases"][0]
        assert np.allclose(a["equivalentStress pull"], c0["vonMises"], rtol=1e-5, atol=1e-6)
        assert np.allclose(a["safetyFactor pull"], np.minimum(c0["safetyFactor"], SF_CAP), rtol=1e-5)
        assert a["stress pull"].shape == (len(f["nodes"]), 6)
        # hexahedra in VTK order have positive volume
        P = a["Points"].astype(float)
        H = a["connectivity"].reshape(-1, 8)
        e1, e2, e3 = P[H[:, 1]] - P[H[:, 0]], P[H[:, 3]] - P[H[:, 0]], P[H[:, 4]] - P[H[:, 0]]
        assert (np.einsum("ij,ij->i", np.cross(e1, e2), e3) > 0).all()
        gov = next(x for x in man["fields"] if x["id"] == "safetyFactor/governing")
        assert abs(gov["range"]["min"] / rep["safetyFactor"] - 1) < 1e-6
        assert gov["target"] == 2.0 and gov["reason"] and gov["units"] == "1"
        s = _vtk_arrays(os.path.join(b, "surface.vtp"))
        assert len(s["Points"]) == len(mesh.vertices)
        doc, _ = _glb(os.path.join(b, "part.glb"))
        names = {n["name"] for n in doc["nodes"]}
        assert {"keep-out pin", "preserve eye", "force pull pin", "force push pin", "weakest point"} <= names
        prim = doc["meshes"][0]["primitives"][0]["attributes"]
        assert {"POSITION", "NORMAL", "COLOR_0", "_SAFETY_FACTOR", "_STRESS_0", "_DISPLACEMENT_1"} <= set(prim)
        ko = next(n for n in doc["nodes"] if n["name"] == "keep-out pin")
        assert ko["extras"]["loadpath"]["role"] == "keepOut"
        rows = open(os.path.join(b, "tradeoff.csv")).read().splitlines()
        assert rows[0].startswith("index,volumeFraction,mass") and len(rows) == 2
        plan = listing(b)
        assert plan["metadata"]["recommendedMaterialId"] and "Safety factor" in plan["metadata"]["description"]
        assert len(plan["metadata"]["description"]) <= 5000 and "part.glb" in plan["notCarried"]


def test_exchanger_flow_grids():
    from loadpath.bundle import exchanger_grids
    from loadpath.heatx import HeatExchangerSpec
    from loadpath.porous import simulate_exchanger
    from loadpath.shapes import Box
    from loadpath.vdb import read_vdb

    hx = HeatExchangerSpec(core=Box((0, 0, 0), (24, 24, 24)), cell=8, wall=1.0, layout="cross", step=0.45,
                           header=5, port_diameter=6)
    duty = {"A": {"fluid": "water", "lPerMin": 0.5, "tIn": 80}, "B": {"fluid": "water", "lPerMin": 0.5, "tIn": 20},
            "wall": "alsi10mg"}
    _, _, s = simulate_exchanger(hx, duty, voxel=2.0, fields=True)
    with tempfile.TemporaryDirectory() as d:
        out = exchanger_grids(d, s["_fields"])
        a = _vtk_arrays(os.path.join(d, "exchanger.vti"))
        for n in ("A", "B"):
            assert a[f"velocity {n}"].shape[1] == 3
            t = a[f"temperature {n}"][a[f"cells {n}"] > 0]
            assert 19.9 < t.min() <= t.max() < 80.1
        grids = {g["name"] for g in read_vdb(os.path.join(d, "exchanger.vdb"))}
        assert {"speed A", "temperature B", "wall temperature"} <= grids
        assert {f["id"] for f in out["fields"]} >= {"velocity/A", "pressure/B", "temperature/wall"}


if __name__ == "__main__":
    for fn in (test_bundle_round_trips_the_verification, test_exchanger_flow_grids):
        fn()
        print("PASS", fn.__name__)
