"""
Powder-bed layer files: slicing into oriented contours, hatching inside
them, and the CLI file written and read back.

Run: `python tests/test_layerfiles.py` (or pytest).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath.layerfiles import export_cli, export_layers, hatch_layer, read_cli, read_slc, slice_layers


def _tube():
    return trimesh.creation.annulus(r_min=4, r_max=10, height=6, sections=96)


def test_tube_layers_have_an_outer_ring_and_a_hole():
    layers = slice_layers(_tube(), 0.5)
    assert len(layers) == 12 and abs(layers[0].z - (-3 + 0.5)) < 1e-9
    for L in layers:
        dirs = sorted(d for _, d in L.contours)
        assert dirs == [0, 1]
        outer = next(p for p, d in L.contours if d == 1)
        hole = next(p for p, d in L.contours if d == 0)
        assert np.linalg.norm(outer, axis=1).max() > 9.9 and np.linalg.norm(hole, axis=1).max() < 4.01
        assert abs(L.area() - np.pi * (10 ** 2 - 4 ** 2)) / (np.pi * 84) < 0.01


def test_hatches_stay_inside_and_turn():
    L = slice_layers(_tube(), 0.5)[5]
    h = hatch_layer(L.contours, 0.2, 30.0)
    r = np.linalg.norm(h, axis=2)
    assert r.min() > 3.95 and r.max() < 10.01
    # covered: the hatch length over the spacing is the section's area
    length = np.linalg.norm(h[:, 1] - h[:, 0], axis=1).sum()
    assert abs(length * 0.2 - L.area()) / L.area() < 0.03
    d = h[:, 1] - h[:, 0]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    assert np.allclose(ang, 30.0, atol=1e-6)


def test_cli_round_trip_and_command(tmp_path):
    out = tmp_path / "tube.cli"
    info = export_cli(_tube(), str(out), layer=0.25, hatch=0.15)
    assert info["openLayers"] == 0 and abs(info["volume"] / info["meshVolume"] - 1) < 1e-3
    text = out.read_text()
    assert text.startswith("$$HEADERSTART") and "$$LAYERS/24" in text and text.rstrip().endswith("$$GEOMETRYEND")
    back = read_cli(str(out))
    assert len(back) == 24 and abs(back[3].z - (-3 + 4 * 0.25)) < 1e-3
    assert abs(back[3].area() - np.pi * 84) / (np.pi * 84) < 0.01
    # consecutive layers' hatches are turned 67 degrees
    a = [np.degrees(np.arctan2(*(L.hatches[0, 1] - L.hatches[0, 0])[::-1])) % 180 for L in back[:2]]
    assert abs(((a[1] - a[0]) % 180) - 67) < 0.5
    stl = tmp_path / "tube.stl"
    _tube().export(stl)
    r = subprocess.run([sys.executable, "-m", "loadpath", "slices", str(stl), "--layer", "0.5", "--axis", "x"],
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "tube.cli").exists()


def test_binary_cli_and_slc_round_trip(tmp_path):
    ref = read_cli(export_cli(_tube(), str(tmp_path / "a.cli"), layer=0.5, hatch=0.3)["path"])
    b = export_layers(_tube(), str(tmp_path / "b.cli"), "cli-binary", layer=0.5, hatch=0.3)
    assert b["format"] == "cli-binary" and (tmp_path / "b.cli").read_bytes().count(b"$$BINARY") == 1
    back = read_cli(b["path"])
    assert len(back) == len(ref) == 12
    for L, R in zip(back, ref, strict=True):
        assert abs(L.z - R.z) < 1e-4 and abs(L.area() - R.area()) < 1e-2
        assert sorted(d for _, d in L.contours) == [0, 1] and len(L.hatches) == len(R.hatches)
    s = export_layers(_tube(), str(tmp_path / "t.slc"), "slc", layer=0.5)
    data = (tmp_path / "t.slc").read_bytes()
    assert data.startswith(b"-SLCVER 2.0 -UNIT MM") and b"\r\n\x1a" in data[:2048]
    assert s["hatches"] == 0 and abs(s["volume"] / s["meshVolume"] - 1) < 2e-3
    back = read_slc(s["path"])
    assert len(back) == 12 and abs(back[0].z - (-2.5)) < 1e-5
    for L in back:
        assert sorted(d for _, d in L.contours) == [0, 1]            # winding gives outer and hole
        assert abs(L.area() - np.pi * 84) / (np.pi * 84) < 0.01


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
