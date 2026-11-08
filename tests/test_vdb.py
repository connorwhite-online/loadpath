"""OpenVDB grids in and out (loadpath/vdb.py): round trips through the
writer and reader, the sdf shape and the field graph's grid op. The writer
was checked against the OpenVDB library itself (pyopenvdb 10 reads the
files back to float precision, vdb_print lists them) and the reader against
files OpenVDB wrote (blosc + active-mask compressed, half float); those
checks need the library, so CI runs the round trips.

Run: `python -m pytest tests/test_vdb.py`. A few seconds.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loadpath.vdb import read_vdb, write_vdb


def _sphere(n=60, pitch=0.5, r=9.0):
    o = np.full(3, -(n - 1) * pitch / 2)
    X = np.indices((n, n, n)).transpose(1, 2, 3, 0) * pitch + o
    return np.linalg.norm(X, axis=-1) - r, o, pitch


@pytest.mark.parametrize("compression", ["zip", "none"])
def test_level_set_round_trip(tmp_path, compression):
    d, o, p = _sphere()
    path = str(tmp_path / "s.vdb")
    write_vdb(path, {"name": "sdf", "values": d, "origin": o, "pitch": p}, compression=compression)
    g = read_vdb(path, "sdf")
    assert g["class"] == "level set" and g["pitch"] == p and g["background"] == pytest.approx(1.5)
    # the dense block starts at a leaf; compare where both are defined
    i0 = np.round((g["origin"] - o) / p).astype(int)
    ref = np.clip(d, -1.5, 1.5)
    X = np.indices(g["values"].shape).reshape(3, -1).T + i0
    ok = np.all((X >= 0) & (X < d.shape), 1)
    assert np.abs(g["values"].reshape(-1)[ok] - ref[tuple(X[ok].T)]).max() < 1e-6
    # the inside away from the band is an inactive tile reading -background
    c = tuple(np.round(-g["origin"] / p).astype(int))
    assert g["values"][c] == pytest.approx(-1.5) and not g["active"][c]
    assert g["active"].sum() == int((np.abs(d) < 1.5).sum())


def test_fog_volume_and_several_grids(tmp_path):
    d, o, p = _sphere(40)
    fog = np.clip(-d / 4, 0, 1)
    path = str(tmp_path / "v.vdb")
    write_vdb(path, [{"name": "density", "values": fog, "origin": o, "pitch": p, "class": "fog volume"},
                     {"name": "sdf", "values": d, "origin": o, "pitch": p}])
    gs = read_vdb(path)
    assert [g["name"] for g in gs] == ["density", "sdf"]
    g = gs[0]
    assert g["background"] == 0 and g["class"] == "fog volume"
    i0 = np.round((g["origin"] - o) / p).astype(int)
    X = np.indices(fog.shape).reshape(3, -1).T - i0
    inside = np.all((X >= 0) & (X < g["values"].shape), 1)
    got = np.zeros(len(X))
    got[inside] = g["values"][tuple(X[inside].T)]
    assert np.abs(got - fog.ravel()).max() < 1e-6


def test_sdf_shape_and_grid_op_read_vdb(tmp_path):
    from loadpath.exterior import save_sdf
    from loadpath.fields import FieldGraph
    from loadpath.shapes import shape_from_spec

    box = shape_from_spec({"type": "box", "center": [0, 0, 0], "size": [20, 10, 6]})
    r = save_sdf(box, str(tmp_path / "box.vdb"), 0.5)
    assert r["format"] == "vdb"
    s = shape_from_spec({"type": "sdf", "path": "box.vdb"}, base_dir=str(tmp_path))
    v = s.sdf(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5], [0.0, 0.0, 3.5], [30.0, 0, 0]]))
    assert v[0] < 0 and v[1] == pytest.approx(-0.5, abs=0.05) and v[2] == pytest.approx(0.5, abs=0.05)
    assert v[3] > 0
    # a fog volume from another tool drives a field
    d, o, p = _sphere(31, 1.0, 8.0)
    write_vdb(str(tmp_path / "t.vdb"), {"name": "temperature", "values": np.clip(-d, 0, None) * 10,
                                       "origin": o, "pitch": p, "class": "fog volume"})
    doc = {"nodes": {"t": {"op": "grid", "path": "t.vdb", "quantity": "temperature"}}, "output": "t"}
    (tmp_path / "f.json").write_text(json.dumps(doc))
    g = FieldGraph(doc, base_dir=str(tmp_path))
    out = g.evaluate(np.array([[0.0, 0.0, 0.0], [5.0, 0, 0]]), "t")
    assert out[0] == pytest.approx(80, rel=0.02) and out[1] == pytest.approx(30, rel=0.05)


def test_rejects_non_vdb(tmp_path):
    p = tmp_path / "x.vdb"
    p.write_bytes(b"not a vdb file at all")
    with pytest.raises(ValueError, match="not an OpenVDB"):
        read_vdb(str(p))
