"""
Tests for pressure loads and skin-and-rib (pressure.py, the `wall` of a
keep-out, the study's `skin`). Small grids: the file runs in about a minute.

Run: `python3 tests/test_pressure.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.domain import build_domain
from loadpath.shapes import trimesh_from_manifold
from loadpath.spec import study_from_dict
from loadpath.verify import verify_part


def _sphere_vessel(a=20.0, b=30.0, p=10.0, **extra):
    return {"name": "sphere", "designSpace": {"type": "sphere", "center": [0, 0, 0], "r": b},
            "keepOut": [{"id": "cav", "shape": {"type": "sphere", "center": [0, 0, 0], "r": a}, "wall": b - a}],
            "loadCases": [{"name": "p", "loads": [{"region": "cav", "pressure": p}]}],
            "material": "pla", "goal": {"volumeFraction": 0.3}, **extra}


def test_sealed_cavity_balances_and_needs_no_supports():
    st = study_from_dict(_sphere_vessel(resolution={"pitch": 2.0}))
    assert st.self_equilibrated and [r.id for r in st.preserve] == ["cav-wall"]
    d = build_domain(st)
    f = d.loads[0].reshape(-1, 3)
    total = np.abs(f).sum()
    # a closed staircase: zero net force and moment, but for the few
    # components the 3-2-1 support holds (and so drops)
    assert np.linalg.norm(f.sum(axis=0)) < 1e-3 * total
    r = d.mesh.node_xyz
    assert np.linalg.norm(np.cross(r, f).sum(axis=0)) < 1e-3 * total * 30
    # p times the projected area of the cavity, on each half
    lo_half = f[r[:, 2] < 0, 2].sum()
    assert abs(-lo_half / (10.0 * np.pi * 20.0 ** 2) - 1.0) < 0.1, lo_half
    # held by a 3-2-1 support: three nodes, six dofs
    assert d.support_nodes.sum() == 3 and d.fixed.sum() == 6
    # the wall is pinned solid: the optimizer cannot open the pressure boundary
    assert d.preserve.all()


def test_thick_sphere_matches_lame():
    a, b, p = 20.0, 30.0, 10.0
    st = study_from_dict(_sphere_vessel(a, b, p))
    mesh = trimesh_from_manifold(st.design_space[0].to_manifold() - st.keep_out[0].shape.to_manifold())
    e, nu = st.material.E, st.material.nu
    vm = p * (b ** 3 + 2 * a ** 3) / (2 * (b ** 3 - a ** 3)) + p     # hoop - radial, at the bore
    ua = p * a ** 3 / (e * (b ** 3 - a ** 3)) * ((1 - 2 * nu) * a + (1 + nu) * b ** 3 / (2 * a ** 2))
    rep = verify_part(st, mesh, max_cells=30_000, refine=1)
    c = rep["cases"][0]
    assert c["pressureImbalance"] < 1e-9
    # measured (2026-10-29): 22.4 MPa vs 21.3, bulge 0.068 vs 0.062 mm
    assert 0.95 * vm < c["peakStress"] < 1.15 * vm, (c["peakStress"], vm)
    assert 0.95 * ua < c["deflection"] < 1.15 * ua, (c["deflection"], ua)


def test_open_duct_without_supports_is_refused():
    d = {"name": "duct", "designSpace": {"type": "cylinder", "center": [0, 0, 0], "axis": "x", "length": 40, "r": 12},
         "keepOut": [{"id": "bore", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "x",
                                              "length": 60, "r": 9}}],
         "loadCases": [{"name": "p", "loads": [{"region": "bore", "pressure": 0.2}]}],
         "material": "pla", "minMember": 3.0, "resolution": {"pitch": 1.5}}
    st = study_from_dict(d)
    # open at both ends: no end-cap thrust, so the load still balances
    dom = build_domain(st)
    assert np.linalg.norm(dom.loads[0].reshape(-1, 3).sum(axis=0)) < 1e-6
    # the wall defaulted to minMember and the bore is no fastener path
    assert st.keep_out[0].wall == 3.0 and not st.keep_out[0].bore
    # a capped one is not balanced: its cap is pushed and nothing holds it
    d["keepOut"][0]["shape"] = {"type": "cylinder", "p0": [-30, 0, 0], "p1": [10, 0, 0], "r": 9}
    try:
        build_domain(study_from_dict(d))
    except ValueError as exc:
        assert "supports" in str(exc), exc
    else:
        raise AssertionError("an unbalanced pressure-only study must ask for supports")


def test_pressure_validation_messages():
    base = _sphere_vessel()
    bad = dict(base, loadCases=[{"name": "p", "loads": [{"shape": {"type": "sphere", "center": [0, 0, 0], "r": 5},
                                                         "pressure": 1.0}]}])
    for doc, words in ((bad, "keepOut"),
                       (dict(base, keepOut=[{"id": "cav", "shape": base["keepOut"][0]["shape"]}]), "wall"),
                       (dict(base, gravity=[0, 0, -9.81]), "supports")):
        try:
            study_from_dict(doc)
        except ValueError as exc:
            assert words in str(exc), exc
        else:
            raise AssertionError(f"accepted a study that should name {words!r}")
    # forces still need supports
    doc = dict(base, loadCases=[{"name": "f", "loads": [{"region": "cav", "force": [0, 0, 1]}]}])
    try:
        study_from_dict(doc)
    except ValueError as exc:
        assert "support" in str(exc)
    else:
        raise AssertionError("a force without supports was accepted")


def test_skin_is_a_shell_of_the_design_space_open_where_asked():
    d = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 30, 10]},
         "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.5, 31, 11]}}],
         "loadCases": [{"name": "f", "loads": [{"shape": {"type": "box", "min": [39.5, -1, -1], "max": [41, 31, 11]},
                                                "force": [0, 0, -10]}]}],
         "skin": {"thickness": 1.0, "open": ["-z"]}, "material": "pla"}
    st = study_from_dict(d)
    skin = next(r for r in st.preserve if r.id == "skin")
    assert st.shells == [("skin", 1.0)]
    v = skin.shape.to_manifold().volume()
    # five faces of a 40 x 30 x 10 box, 1 mm thick, the bottom left open
    want = 40 * 30 * 10 - 38 * 28 * 9
    assert abs(v - want) < 1e-6 * want, (v, want)
    p = np.array([[20, 15, 0.5], [20, 15, 9.5], [0.5, 15, 5], [20, 15, 5]])
    assert list(skin.shape.contains(p)) == [False, True, True, False]
    # a skin that fills the box is refused
    try:
        study_from_dict(dict(d, skin=6.0))
    except ValueError as exc:
        assert "thinner" in str(exc)
    else:
        raise AssertionError("a skin thicker than the box was accepted")


def test_wall_shell_is_clipped_to_the_design_space():
    st = study_from_dict(_sphere_vessel(20.0, 30.0, 1.0))
    wall = st.preserve[0].shape
    v = wall.to_manifold().volume()
    want = 4.0 / 3.0 * np.pi * (30.0 ** 3 - 20.0 ** 3)
    assert abs(v - want) < 0.01 * want, (v, want)
    # a duct running out of a box: its wall stops at the box
    d = {"designSpace": {"type": "box", "min": [-10, -10, -10], "max": [10, 10, 10]},
         "keepOut": [{"id": "d", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "x", "length": 40,
                                           "r": 4}, "wall": 2.0}],
         "supports": [{"shape": {"type": "box", "min": [-11, -11, -11], "max": [-9.5, 11, 11]}}],
         "loadCases": [{"name": "p", "loads": [{"region": "d", "pressure": 0.1}]}], "material": "pla"}
    st = study_from_dict(d)
    bb = st.preserve[0].shape.to_manifold().bounding_box()
    assert bb[0] >= -10 - 1e-6 and bb[3] <= 10 + 1e-6, bb


TESTS = [
    test_sealed_cavity_balances_and_needs_no_supports,
    test_thick_sphere_matches_lame,
    test_open_duct_without_supports_is_refused,
    test_pressure_validation_messages,
    test_skin_is_a_shell_of_the_design_space_open_where_asked,
    test_wall_shell_is_clipped_to_the_design_space,
]


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
