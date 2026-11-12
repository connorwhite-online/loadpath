"""The exterior check: the surface that comes out is the surface that went in."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import surfacecheck as S                 # noqa: E402


def _pebble():
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=4)
    m.apply_scale([23, 53, 6])
    return m


def test_exact_distance():
    """Points a known distance off a unit cube's faces, edges and corners."""
    import trimesh
    d = S._Exact(trimesh.creation.box(extents=[2, 2, 2]), cap=5.0)
    P = np.array([[0, 0, 1.3], [0, 0, 0.6], [1.5, 1.5, 0], [2, 2, 2], [0.2, -0.4, 1.0]], float)
    want = [0.3, 0.4, np.sqrt(0.5), np.sqrt(3.0), 0.0]
    assert np.allclose(d(P), want, atol=1e-9), d(P)
    # a triangle with no area (booleans leave them) is ignored, not divided by
    box = trimesh.creation.box(extents=[2, 2, 2])
    v = np.vstack([box.vertices, [[5.0, 5.0, 5.0]]])
    f = np.vstack([box.faces, [[8, 8, 8]]])
    d = S._Exact(trimesh.Trimesh(v, f, process=False), cap=5.0)
    assert np.allclose(d(P), want, atol=1e-9), d(P)


def test_exterior_kept_or_not():
    """The same surface passes; a slot through it fails until it is declared;
    a 0.15 mm dent fails and reads 0.15 mm; a different form fails
    everywhere; a form scaled down passes against the scaled original."""
    import trimesh
    a = _pebble()
    px = 0.3
    same = S.compare_exterior(a, a.copy(), pixel=px)
    assert same["pass"] and same["maxDeviation"] == 0.0
    slot = trimesh.creation.box(bounds=[(-30, 9.25, -2), (30, 10.75, 2)])
    cut = trimesh.boolean.difference([a, slot], engine="manifold")
    r = S.compare_exterior(a, cut, pixel=px)
    assert not r["pass"] and r["views"]["right"]["missing"] > 3, r["views"]
    r = S.compare_exterior(a, cut, pixel=px, cuts=[((-30, 9.0, -2.2), (30, 11.0, 2.2))])
    assert r["pass"] and r["views"]["right"]["inCuts"] > 3, r["views"]
    dent = a.copy()
    v = dent.vertices.copy()
    k = (np.linalg.norm(v[:, :2] - [5, 20], axis=1) < 6) & (v[:, 2] > 0)
    v[k, 2] -= 0.15
    dent.vertices = v
    r = S.compare_exterior(a, dent, pixel=px)
    assert not r["pass"] and 0.13 < r["views"]["top"]["maxDeviation"] <= 0.151, r["views"]["top"]
    assert 80 < r["views"]["top"]["addedOrMoved"] < 130 and r["views"]["bottom"]["maxDeviation"] == 0.0
    box = trimesh.creation.box(extents=[46, 106, 12])
    with tempfile.TemporaryDirectory() as tmp:
        r = S.ortho_sheet(a, box, os.path.join(tmp, "sheet.png"), title="a box is not the pebble")
        assert os.path.getsize(os.path.join(tmp, "sheet.png")) > 20_000
    assert not r["pass"] and all(v["maxDeviation"] >= 1.0 for v in r["views"].values())
    small = a.copy()
    small.apply_scale(0.9)
    assert S.compare_exterior(a, small, pixel=px, scale=0.9)["pass"]
    assert not S.compare_exterior(a, small, pixel=px)["pass"]
    print("exterior check ok")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all surfacecheck tests passed")
