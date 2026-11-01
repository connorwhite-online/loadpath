"""Lattice briefs (loadpath/evals/brief_kinds/lattice.py): each brief's gold
builds and a reference delivery passes its own scorer (coarse settings; the
riser's and the arm's whole-scorer runs need LOADPATH_SLOW_TESTS=1, their
lattice rules always run), and
deliveries that miss the point fail: a solid pad that cannot crush, a solid
riser that is no lattice, a strut lattice with flat struts that need support,
a boom too floppy for its camera."""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath.evals import briefs
from loadpath.evals.brief_kinds import lattice as L

IDS = ["shoulder-riser-mjf", "elbow-stop-pad-tpu", "tray-arm-lattice-fdm", "camera-boom-mjf"]
# coarse verification; the references keep their margins here too (struts
# 2 mm thick need ~1 mm cells, so the arm gets more)
CELLS = {"tray-arm-lattice-fdm": 20_000}
# verify_part on a lattice mesh spends minutes outside the solve (sliver
# checks on the riser's 800k faces, crease clustering on the arm's struts),
# so their whole-scorer runs are opt-in; their lattice rules run below
SLOW = pytest.mark.skipif(not os.environ.get("LOADPATH_SLOW_TESTS"),
                          reason="minutes of verify_part on a lattice mesh; set LOADPATH_SLOW_TESTS=1")


def _check_row(b, row):
    assert row["pass"], row["failures"]
    want = b["pass"]
    if want.get("maxMass"):
        assert row["mass"] <= want["maxMass"]
    if want.get("safetyFactor"):
        assert row["safetyFactor"] >= want["safetyFactor"]
    if want.get("impact"):
        assert row["impact"]["peakForce"] <= 0.95 * want["impact"]["forceLimit"]
    if want.get("vibration"):
        assert row["vibration"]["frequencies"][0] >= 1.1 * want["vibration"]["minFrequency"]


@pytest.mark.parametrize("bid", IDS)
def test_gold_builds(bid):
    b = briefs.load(bid)
    assert b["kind"] == "lattice"
    g = L.gold(b)
    assert g["ok"], g.get("error")


@pytest.mark.parametrize("bid", ["elbow-stop-pad-tpu", "camera-boom-mjf",
                                 pytest.param("shoulder-riser-mjf", marks=SLOW),
                                 pytest.param("tray-arm-lattice-fdm", marks=SLOW)])
def test_reference_passes_its_scorer(bid, tmp_path):
    b = briefs.load(bid)
    L.reference(b, str(tmp_path), quick=True)
    _check_row(b, L.score(b, str(tmp_path), max_cells=CELLS.get(bid, 8_000)))


def test_riser_reference_is_a_drainable_lattice(tmp_path):
    b = briefs.load("shoulder-riser-mjf")
    mesh = trimesh.load(L.reference(b, str(tmp_path), quick=True), force="mesh")
    spec = b["pass"]["cellular"]
    c = L.cellular(mesh, spec, L._design_box(b))
    assert not L._cellular_failures(c, spec), c
    assert L.enclosed_volume(mesh) <= b["pass"]["drains"]["maxEnclosed"]
    assert abs(mesh.volume) * 1.01e-3 <= b["pass"]["maxMass"]


def test_arm_reference_prints_unsupported(tmp_path):
    b = briefs.load("tray-arm-lattice-fdm")
    mesh = trimesh.load(L.reference(b, str(tmp_path), quick=True), force="mesh")
    row, f = L._print(mesh, b["pass"]["print"], None)
    assert not f, row
    spec = b["pass"]["cellular"]
    c = L.cellular(mesh, spec)
    assert not L._cellular_failures(c, spec), c


def test_prompt_asks_for_the_field_graph():
    text = briefs.prompt(briefs.load("elbow-stop-pad-tpu"))
    assert "field.json" in text and "part.stl" in text and "1.2 kg" in text


def test_solid_pad_fails(tmp_path):
    b = briefs.load("elbow-stop-pad-tpu")
    trimesh.creation.box(extents=[30, 30, 15]).apply_translation([0, 0, 7.5]).export(str(tmp_path / "part.stl"))
    row = briefs.score_dir(b, str(tmp_path))
    assert not row["pass"]
    txt = " ".join(row["failures"])
    assert "no lattice core" in txt and "peak force" in txt
    assert not briefs.score_dir(b, str(tmp_path / "nothing-here"))["pass"]


def test_dense_pad_fails(tmp_path):
    """A gyroid core twice as dense as the reference: over the force limit."""
    from loadpath.fields import FieldGraph

    b = briefs.load("elbow-stop-pad-tpu")
    sh = lambda lo, hi: {"op": "shape", "shape": {"type": "box", "min": lo, "max": hi}}
    doc = {"nodes": {"core": {"op": "lattice", "type": "gyroid", "cell": 8, "density": 0.4},
                     "clip": {"op": "intersect", "inputs": ["core", sh([-15, -15, 1], [15, 15, 14])]},
                     "part": {"op": "union", "inputs": ["clip", sh([-15, -15, 0], [15, 15, 1.2]),
                                                        sh([-15, -15, 13.8], [15, 15, 15])]}}}
    FieldGraph(doc).mesh(pitch=0.3).export(str(tmp_path / "part.stl"))
    row = L.score(b, str(tmp_path))
    assert not row["pass"] and any("peak force" in f for f in row["failures"]), row


def test_solid_riser_is_not_a_lattice():
    b = briefs.load("shoulder-riser-mjf")
    spec = b["pass"]["cellular"]
    block = trimesh.creation.box(extents=[60, 50, 40]).apply_translation([0, 0, 20])
    c = L.cellular(block, spec, (np.array([-30, -25, 0.0]), np.array([30, 25, 40.0])))
    f = L._cellular_failures(c, spec)
    assert any("not a lattice" in x for x in f), c
    assert c["hull"] > 0.99


def test_sealed_skin_traps_powder():
    import manifold3d as m3d

    from loadpath.shapes import trimesh_from_manifold

    shell = L._mbox([-30, -25, 0], [30, 25, 40]) - L._mbox([-28, -23, 2], [28, 23, 38])
    assert L.enclosed_volume(trimesh_from_manifold(shell)) > 1e4
    assert isinstance(shell, m3d.Manifold)


def test_flat_struts_need_support():
    """Horizontal struts every 8 mm (a cubic cell) at an 8 mm bridge limit:
    their undersides need support, where the tall BCC's do not."""
    b = briefs.load("tray-arm-lattice-fdm")
    spec = b["pass"]["print"]
    lo, hi = np.array([4, -15, 0.0]), np.array([74, 15, 40.0])
    rods = []
    for z in np.arange(4, 40, 8.0):
        for y in np.arange(-11, 15, 8.0):
            rods.append(L._mcyl([lo[0], y, z], [hi[0], y, z], 1.0, 8))
        for x in np.arange(8, 74, 8.0):
            rods.append(L._mcyl([x, lo[1], z], [x, hi[1], z], 1.0, 8))
    for x in np.arange(8, 74, 8.0):
        for y in np.arange(-11, 15, 8.0):
            rods.append(L._mcyl([x, y, 0], [x, y, 40], 1.0, 8))
    from loadpath.shapes import trimesh_from_manifold

    cubic = trimesh_from_manifold(L._union(rods + [L._mbox([0, -15, 0], [80, 15, 2.5])]) ^ L._mbox([0, -15, 0], [80, 15, 40]))
    row, f = L._print(cubic, spec, None)
    assert f and row["supportArea"] > 10 * spec["maxSupportArea"], row


def test_floppy_boom_fails(tmp_path):
    """A 6 mm solid rod in place of the tube: light enough, far too soft."""
    from loadpath.shapes import trimesh_from_manifold

    b = briefs.load("camera-boom-mjf")
    body = L._union([L._mbox([0, -18, -18], [3, 18, 18]), L._mbox([3, -3, -3], [95, 3, 3]),
                     L._mbox([70, -11, 7], [95, 11, 11]), L._mbox([70, -3, 3], [95, 3, 7])])
    mesh = trimesh_from_manifold(body)
    row, f = L._vibration(b, mesh, b["pass"]["vibration"])
    assert f and row["frequencies"][0] < 0.5 * b["pass"]["vibration"]["minFrequency"], row


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
