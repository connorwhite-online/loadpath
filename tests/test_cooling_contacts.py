"""
Cooling contacts (an outside face resting on something at its own
temperature, such as an accessory on a phone's back) and components that
conduct differently along each axis (a board). Small problems; seconds.

Run: `python3 tests/test_cooling_contacts.py`
"""
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import cooling  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.thermal import hex8_kt, hex8_kt_axes  # noqa: E402

from test_cooling import _dom, box  # noqa: E402

BOTTOM = {"type": "box", "min": [-25, -15, -1], "max": [25, 15, 0.01]}


def test_axes_sum_to_the_isotropic_matrix():
    kt = hex8_kt_axes()
    assert np.allclose(kt.sum(axis=0), hex8_kt(), atol=1e-12)
    for a in range(3):
        assert np.allclose(kt[a].sum(axis=1), 0.0, atol=1e-12)   # a uniform temperature conducts nothing


def test_contact_lints_and_is_validated():
    doc = box(contacts=[{"shape": BOTTOM, "temperature": 35, "h": 400}])
    assert not lint(doc, "study"), lint(doc, "study")
    assert lint(box(contacts=[{"shape": BOTTOM}]), "study")
    doc = box(components=[{"region": "chip", "watts": 0.5, "maxTemp": 60},
                          {"region": "board", "conductivity": [30, 30, 0.4]}])
    assert not lint(doc, "study"), lint(doc, "study")


def test_contact_heat_balance_and_isothermal_answer():
    # a box filled with a near-perfect conductor: one temperature T with
    # W = h_skin A_skin (T - Ta) + h_c A_c (T - Tc)
    doc = box(contacts=[{"id": "phone", "shape": BOTTOM, "temperature": 35.0, "h": 400.0}])
    doc["cooling"]["conductivity"] = 5000.0
    doc["cooling"]["components"] = [{"region": "chip", "watts": 0.5, "maxTemp": 60, "conductivity": 5000.0},
                                    {"region": "board", "conductivity": 5000.0}]
    st, dom, fl = _dom(doc, 1.0)
    r = cooling.cooling_readings(st, dom, np.ones(dom.n_elem), floor=fl)
    out = r["heatOut"]
    assert abs(out["skin"] + out["contacts"]["phone"] - 0.5) < 1e-3, out
    a_c = 40 * 24
    a_s = 2 * (40 * 24 + 40 * 12 + 24 * 12) - a_c
    t = (0.5 + 10e-6 * a_s * 25.0 + 400e-6 * a_c * 35.0) / (10e-6 * a_s + 400e-6 * a_c)
    got = r["components"][0]["temperature"]
    assert abs(got - t) < 0.05 * (t - 25.0), (got, t)
    # the box sits above ambient from the contact alone: the bottom feeds heat in
    assert out["contacts"]["phone"] < 0.5


def test_contact_faces_are_not_touched():
    doc = box(contacts=[{"shape": BOTTOM, "temperature": 35.0}])
    st, dom, fl = _dom(doc, 1.0)
    mdl = cooling.CoolingModel(st, dom, floor=fl)
    z = mdl.mesh.node_xyz[mdl.skin_nodes][:, 2]
    assert z.min() > 0.5, z.min()            # no touched node on the bottom face
    bottom_only = mdl.mesh.node_xyz[:, 2] < 1e-6
    assert bottom_only.any()


def test_orthotropic_board_spreads_in_plane():
    def chip_t(k):
        doc = box(components=[{"region": "chip", "watts": 0.5, "maxTemp": 90},
                              {"region": "board", "conductivity": k}])
        st, dom, fl = _dom(doc, 1.0)
        x = np.maximum(dom.preserve.astype(float), fl)
        return cooling.cooling_readings(st, dom, np.maximum(x, 1e-9), floor=fl)["components"][0]["temperature"]
    iso_lo, iso_hi = chip_t(0.4), chip_t(30.0)
    same = chip_t([0.4, 0.4, 0.4])
    ortho = chip_t([30.0, 30.0, 0.4])
    assert abs(same - iso_lo) < 1e-6, (same, iso_lo)
    # copper in plane does most of what a fully copper board does
    assert iso_hi < ortho < iso_lo, (iso_hi, ortho, iso_lo)


def test_thin_mesh_board_conducts_like_its_box(tmp=None):
    # a 0.8 mm board given as a mesh, a chip sitting on it: the chip reads
    # what it reads on the same board given as a box (its contact kept, its
    # faces where they are)
    import tempfile

    import trimesh
    d = tempfile.mkdtemp()
    trimesh.creation.box(extents=[36, 22, 0.8], transform=trimesh.transformations.translation_matrix(
        [0, 0, 3.0])).export(f"{d}/board.stl")

    def chip_t(board_shape):
        doc = box(components=[{"region": "chip", "watts": 0.5, "maxTemp": 90},
                              {"id": "board", "shape": board_shape, "conductivity": [30, 30, 0.4]}])
        doc["keepOut"][0]["shape"] = board_shape
        doc["keepOut"][1]["shape"] = {"type": "box", "min": [-10, -4, 3.4], "max": [-2, 4, 4.4]}
        from loadpath.spec import study_from_dict
        from loadpath.domain import build_domain
        st = study_from_dict(doc, base_dir=d)
        dom = build_domain(st, 0.5)
        x = np.maximum(dom.preserve.astype(float), 1e-9)
        return cooling.cooling_readings(st, dom, x)["components"][0]["temperature"]
    t_box = chip_t({"type": "box", "min": [-18, -11, 2.6], "max": [18, 11, 3.4]})
    t_mesh = chip_t({"type": "mesh", "path": "board.stl"})
    assert abs(t_mesh - t_box) < 0.05 * (t_box - 25.0), (t_mesh, t_box)


def test_an_unmeasured_gap_keeps_the_contact():
    # _shape_gap is inf when it finds no surface of one shape near the
    # other (a coarse mesh distance grid); that is no evidence of air
    from loadpath.shapes import Box
    a = cooling.Source("board", 0.0, None, 30.0, Box([-10, -10, 0], [10, 10, 0.8]))
    b = cooling.Source("chip", 0.3, None, 20.0, Box([-2, -2, 0.8], [2, 2, 1.6]))
    c = np.array([[0.0, 0.0, 0.6], [0.0, 0.0, 1.0]])
    a.cells, b.cells = np.array([0]), np.array([1])
    k = np.array([30.0, 20.0])
    real = cooling._shape_gap
    try:
        cooling._shape_gap = lambda *x: float("inf")
        cooling._gap_between_sources([a, b], c, k, 0.5, 0.03)
    finally:
        cooling._shape_gap = real
    assert k.tolist() == [30.0, 20.0], k


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:
            bad += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    sys.exit(1 if bad else 0)
