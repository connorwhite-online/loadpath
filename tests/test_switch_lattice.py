"""
Push pads with a graded lattice under the skin (loadpath/switch_lattice.py),
on the small box pad of test_switch.py. Well under a few minutes.

Run: `python3 tests/test_switch_lattice.py`
"""
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from loadpath import switch as sw  # noqa: E402
from loadpath import switch_lattice as sl  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from test_switch import pad  # noqa: E402


def lattice_pad(**lattice):
    d = pad()
    d["switch"]["lattice"] = {"type": "octet", "cell": 4.0, **lattice}
    return d


def test_lattice_block_lints_and_validates():
    assert not lint(lattice_pad(mode="where-needed"), "study"), lint(lattice_pad(), "study")
    assert lint(lattice_pad(mode="everywhere"), "study")
    try:
        sl.LatticeOptions.from_block({"type": "gyroid", "cel": 4})
        raise AssertionError("an unknown key passed")
    except ValueError:
        pass
    # 1 mm walls already fill a 2 mm gyroid cell past the densest allowed
    st = study_from_dict(pad())
    try:
        sl.optimize_switch_lattice(st, sl.LatticeOptions(type="gyroid", cell=2.0), pitch=1.5, iterations=1)
        raise AssertionError("an impossible lattice passed")
    except ValueError:
        pass


def test_sensitivities_match_finite_differences():
    """The chain from the lattice's density (and, where-needed, its
    presence) through the homogenized modulus to the fire forces."""
    st = study_from_dict(lattice_pad())
    for mode in ("fill", "where-needed"):
        o = sl.LatticeOptions(type="octet", cell=4.0, mode=mode)
        m = sl.optimize_switch_lattice(st, o, pitch=1.5, probe=True)
        rng = np.random.default_rng(3)
        y = rng.uniform(0.2, 0.8, m.n)
        z = rng.uniform(0.3, 0.7, m.n)
        beta = 2.0
        c0, dc, v0, dv, _ = m.evaluate(y, z, beta)
        des = np.flatnonzero(m.design)
        nd = len(des)
        for j in rng.choice(nd, 2, replace=False):
            for part in ((0, 1) if m.where else (0,)):
                h = 1e-6
                y2, z2 = y.copy(), z.copy()
                (y2 if part == 0 else z2)[des[j]] += h
                c1, _, v1, _, _ = m.evaluate(y2, z2, beta)
                k = j + part * nd
                fd = (c1 - c0) / h
                assert np.allclose(fd, dc[:, k], rtol=2e-3, atol=1e-7 * np.abs(dc).max()), (mode, part, fd, dc[:, k])
                assert abs((v1 - v0) / h - dv[k]) <= 1e-4 * abs(dv[k]) + 1e-9, (mode, part)


def test_optimizer_moves_the_force_map_and_the_part_keeps_paths_clear():
    st = study_from_dict(lattice_pad(mode="where-needed"))
    r = sl.optimize_switch_lattice(st, pitch=1.5, iterations=30)
    assert r.history[0]["worstViolation"] > r.history[-1]["worstViolation"]
    lo, hi = r.report["rhoRange"]
    held = r.presence > 0.5
    assert held.any()
    assert (r.rho[held] >= lo * 0.999).all() and (r.rho[held] <= hi * 1.001).all()
    mesh, info = sl.lattice_part(st, r, step=0.3)
    assert info["watertight"], info
    assert info["latticeOnlyVolume"] > 0
    # nothing grows into the board, the stem, or the way they come in
    clear = sl.path_clearance(st, mesh, samples=50_000)
    assert all(v < 0.5 for v in clear.values()), clear


def test_readback_matches_the_grid_model_on_a_whole_cell_part():
    """A part made of whole grid cells reads the same through the cut-cell
    read-back as through switch.fire_forces on that grid."""
    import trimesh
    st = study_from_dict(pad())
    from loadpath.domain import build_domain
    d = build_domain(st, 1.0)
    spec = sw.switch_from_study(st, d)
    # every design-space cell solid: a box of whole cells
    boxes = [trimesh.creation.box(extents=[d.pitch] * 3).apply_translation(c) for c in d.mesh.elem_centers()]
    solid = trimesh.boolean.union(boxes, engine="manifold")
    ref = sw.fire_forces(d, st.elastic(design=True), spec, np.ones(d.n_elem), float(st.material.E))
    rb = sl.readback(st, solid, 1.0)
    got = np.asarray(rb["forceGf"]) * sw.GF
    assert np.allclose(got, ref["forces"], rtol=0.02), (got / sw.GF, ref["forces"] / sw.GF)


def test_corrected_windows_divide_by_the_model_error():
    w = [(1.0, 2.0), (None, 3.0)]
    out = sl.corrected_windows(w, np.array([1.5, 2.0]), np.array([3.0, 2.0]))
    assert out == [(0.5, 1.0), (None, 3.0)]


def test_powder_escapes_an_open_box_not_a_closed_one():
    import trimesh
    st = study_from_dict(pad())
    outer = trimesh.creation.box(extents=[10, 10, 10])
    inner = trimesh.creation.box(extents=[8, 8, 8])
    closed = trimesh.boolean.difference([outer, inner], engine="manifold")
    hole = trimesh.creation.cylinder(radius=1.5, height=4).apply_translation([0, 0, 5])
    open_ = trimesh.boolean.difference([closed, hole], engine="manifold")
    a = sl.powder_escape(st, closed, step=0.25)
    b = sl.powder_escape(st, open_, step=0.25)
    assert a["trappedRooms"] == 1 and a["trappedVolumeMm3"] > 400, a
    assert b["trappedRooms"] == 0 and b["largestFreeBallMm"] >= 2.5, b


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
