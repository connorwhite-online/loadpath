"""
Ribs on a face (the field graph's `ribs` op, ribs.py): patterns, curves
projected onto the surface (walls along a direction or square to it), and
ribs along the principal stress lines of a study.

Run: `python tests/test_ribs.py` (or pytest). About two minutes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loadpath.fields import FieldGraph
from loadpath.ribs import pattern_distance, polylines_distance

BOX = {"op": "shape", "shape": {"type": "box", "min": [-40, -30, 0], "max": [40, 30, 20]}}


def _graph(ribs, clip_z=10.0):
    return FieldGraph({"nodes": {
        "body": BOX,
        "skin": {"op": "shell", "input": "body", "thickness": 1.5},
        "ribs": {"op": "ribs", "surface": "body", **ribs},
        "part": {"op": "union", "inputs": ["skin", {"op": "intersect", "inputs": [
            "ribs", {"op": "shape", "shape": {"type": "box", "min": [-41, -31, clip_z], "max": [41, 31, 21]}}]}]}},
        "output": "part"})


def test_patterns_measure_their_walls():
    sq = pattern_distance({"type": "square", "spacing": 10})
    P = np.array([[0, 0, 5], [5, 5, 0], [2, 7, 0], [12, 3, 0]], float)
    assert np.allclose(sq(P), [0, 5, 2, 2])
    iso = pattern_distance({"type": "triangle", "spacing": 10})
    assert iso(np.array([[0.0, 0.0, 0.0]]))[0] == 0
    rad = pattern_distance({"type": "radial", "spacing": 10, "spokes": 4, "rings": True})
    # on a spoke, on a ring, and between
    assert np.allclose(rad(np.array([[7.0, 0, 0], [0, 0, 0], [10 * np.cos(0.3), 10 * np.sin(0.3), 0]])), [0, 0, 0],
                       atol=1e-9)
    d = polylines_distance([[[0, 0, 0], [10, 0, 0]]], direction="z")
    assert np.allclose(d(np.array([[5, 3, 40.0], [12, 0, -7]])), [3, 2])


def test_isogrid_under_a_lid_adds_the_ribs_volume():
    bare = _graph({"thickness": 1.2, "height": 8, "pattern": {"type": "square", "spacing": 1e4,
                                                              "origin": [-1e3, -1e3, 0]}}).mesh(pitch=0.8)
    g = _graph({"thickness": 1.2, "height": 8, "pattern": {"type": "triangle", "spacing": 15}})
    m = g.mesh(pitch=0.8)
    assert m.is_watertight
    # three families of ribs 1.2 thick and 6.5 deep below the 1.5 mm lid, about 80 x 60 mm
    added = m.volume - bare.volume
    lines = sum(80 * 60 / 15 for _ in range(3))          # mm of rib per family ~ area / spacing
    assert 0.6 * lines * 1.2 * 6.5 < added < 1.3 * lines * 1.2 * 6.5, added


def test_paths_along_a_direction_and_square_to_the_surface():
    path = [[[-40, -30, 25], [40, 30, 25]]]
    a = _graph({"thickness": 1.2, "height": 8, "paths": path, "direction": "z"})
    b = _graph({"thickness": 1.2, "height": 8, "paths": path})
    P = np.array([[0, 0, 15], [0, 3, 15], [20, 15, 19], [0, 0, 10]], float)      # the last 10 deep: past the 8
    for g in (a, b):
        r = g.evaluate(P, "ribs")
        assert r[0] < 0 and r[1] > 0 and r[2] < 0 and r[3] > 0, r
    # the projected curve lies on the lid
    assert np.allclose(b.nodes["ribs"].lines[0][:, 2], 20, atol=1e-3)


def test_stress_lines_follow_the_span(tmp_path):
    study = {"designSpace": {"type": "box", "min": [-50, -30, -7], "max": [50, 30, 2]},
             "supports": [{"shape": {"type": "box", "min": [-51, -31, -8], "max": [-47, 31, 3]}},
                          {"shape": {"type": "box", "min": [47, -31, -8], "max": [51, 31, 3]}}],
             "loadCases": [{"name": "press", "loads": [{"shape": {"type": "box", "min": [-5, -5, 1.5],
                                                                  "max": [5, 5, 2.5]}, "force": [0, 0, -50]}]}],
             "material": "pa12", "goal": {"safetyFactor": 2}}
    (tmp_path / "study.json").write_text(json.dumps(study))
    g = FieldGraph({"nodes": {
        "body": {"op": "shape", "shape": {"type": "box", "min": [-50, -30, 0], "max": [50, 30, 2]}},
        "ribs": {"op": "ribs", "surface": "body", "thickness": 1.5, "height": 6, "side": "outside",
                 "stressLines": {"study": "study.json", "spacing": 12, "family": "max", "cells": 30000,
                                 "within": {"min": [-51, -31, -0.5], "max": [51, 31, 0.5]}}}},
        "output": "ribs"}, base_dir=str(tmp_path))
    g.evaluate(np.zeros((1, 3)), "ribs")
    lines = g.nodes["ribs"].lines
    assert len(lines) >= 3
    for ln in lines:
        assert np.allclose(ln[:, 2], 0, atol=0.05)           # on the face asked for
    # a plate spanning x bends along x: the larger principal stress runs along the span
    d = np.concatenate([np.diff(ln, axis=0) for ln in lines])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert np.abs(d[:, 0]).mean() > 0.8
    # the longest lines reach most of the span
    assert max(np.ptp(ln[:, 0]) for ln in lines) > 60
    assert "ribs" in g.analyses


def test_ribs_need_one_layout():
    with pytest.raises(ValueError, match="one of paths, pattern or stressLines"):
        _graph({"thickness": 1, "height": 2})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
