"""A shape dict's mesh path in a brief resolves against the brief's folder,
as a bare path does, so the study written into --out still finds it.

Run: `python3 tests/test_brief_paths.py`
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.handoff import _shape_spec  # noqa: E402


def test_mesh_shape_dict_path_is_made_absolute():
    s = _shape_spec({"type": "mesh", "path": "cad/housing.stl"}, "/briefs/puck")
    assert s["path"] == os.path.abspath("/briefs/puck/cad/housing.stl"), s
    s = _shape_spec({"type": "mesh", "path": "/abs/housing.stl"}, "/briefs/puck")
    assert s["path"] == "/abs/housing.stl"
    box = {"type": "box", "min": [0, 0, 0], "max": [1, 1, 1]}
    assert _shape_spec(box, "/briefs/puck") is box
    assert _shape_spec({"type": "mesh", "path": "a.stl"}, None)["path"] == "a.stl"


if __name__ == "__main__":
    test_mesh_shape_dict_path_is_made_absolute()
    print("ok")
