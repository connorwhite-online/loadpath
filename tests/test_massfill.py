"""massfill on a hollow box with a soft lid: the fill stays off a keep-out
and under the lid's travel, reaches a target, and reports when it cannot."""
import numpy as np
import pytest
import trimesh

from loadpath.massfill import run_massfill


def _box(tmp_path):
    outer = trimesh.creation.box([30, 30, 12]); outer.apply_translation([0, 0, 6])
    inner = trimesh.creation.box([26, 26, 9]); inner.apply_translation([0, 0, 6.5])
    shell = outer.difference(inner)
    part = trimesh.creation.box([6, 6, 3]); part.apply_translation([0, 0, 3.5])
    shell.export(tmp_path / "shell.stl"); part.export(tmp_path / "part.stl")
    # the lid moves up to 3 mm at its centre, nothing at its edge
    xs = np.linspace(-13, 13, 27); X, Y = np.meshgrid(xs, xs)
    N = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 11.0)], 1)
    T = 3.0 * np.clip(1 - np.maximum(abs(N[:, 0]), abs(N[:, 1])) / 13, 0, 1)
    np.savez(tmp_path / "env.npz", nodes=N, travel=T)


def test_fill_room_and_target(tmp_path):
    _box(tmp_path)
    brief = {"exterior": "shell.stl", "pitch": 0.5, "material": "tpu88a", "minSection": 1.0,
             "keepouts": [{"id": "part", "mesh": "part.stl", "clearance": 0.5, "noOverhang": True}],
             "travel": {"envelope": "env.npz", "margin": 0.5}, "out": "fill.stl"}
    r = run_massfill(brief, base_dir=str(tmp_path))
    assert r["maxFill_g"] > 1.0 and r["fill_g"] > 0.5
    fill = trimesh.load(tmp_path / "fill.stl")
    part = trimesh.load(tmp_path / "part.stl")
    assert fill.bounds[1][2] < 11.0                                  # under the lid
    # nothing over the part's footprint
    c = fill.vertices
    over = (abs(c[:, 0]) < 3) & (abs(c[:, 1]) < 3) & (c[:, 2] > 3.5)
    assert not over.any()
    assert trimesh.proximity.signed_distance(part, c[::5]).max() < -0.2
    r2 = run_massfill({**brief, "target_g": 1000.0, "out": None}, base_dir=str(tmp_path))
    assert r2["reaches"] is False and "tops out" in r2["note"]
    small = run_massfill({**brief, "target_g": 1.0, "out": None}, base_dir=str(tmp_path))
    assert small["fill_g"] < r["fill_g"]
