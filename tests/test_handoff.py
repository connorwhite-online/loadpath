"""
The designer's handoff (loadpath/handoff.py): exterior and component CAD
in, a push-pad study out. A small synthetic product; about a minute.

Run: `python3 tests/test_handoff.py`
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.handoff import handoff, run_handoff  # noqa: E402
from loadpath.schema import lint  # noqa: E402

_DIR = None


def _stl(name, man):
    import trimesh
    m = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))
    path = os.path.join(_DIR, name)
    tm.export(path)
    return path


def product():
    """A 30 x 30 x 10 mm pad. Its underside opens 16 x 16 mm for a flap
    that carries the switch; a board's far end rests on the floor past the
    opening's +x edge, so it cannot drop straight out."""
    global _DIR
    import manifold3d as m3d
    if _DIR is None:
        _DIR = tempfile.mkdtemp()
    box = m3d.Manifold.cube((30, 30, 10)).translate((-15, -15, 0))
    hole = m3d.Manifold.cube((16, 16, 3)).translate((-8, -8, -1))
    ext = _stl("exterior.stl", box - hole)
    mouth = _stl("opening.stl", hole)
    flap = _stl("flap.stl", m3d.Manifold.cube((15.6, 15.6, 2.0)).translate((-7.8, -7.8, 0)))
    sw = m3d.Manifold.cube((4, 4, 2)).translate((-2, -2, 2)) + \
        m3d.Manifold.cylinder(1.0, 1.0, 1.0, 32).translate((0, 0, 4))
    switch = _stl("switch.stl", sw)
    board = _stl("board.stl", m3d.Manifold.cube((14, 4, 1)).translate((-4, 3, 2.5)))
    return {
        "name": "pad-handoff",
        "exterior": ext, "skin": 1.0, "openings": [mouth], "material": "tpu88a", "minMember": 1.0,
        "components": [
            {"id": "board", "shape": board},
            {"id": "flap", "shape": flap, "role": "seat"},
            {"id": "switch", "shape": switch, "role": "switch", "installedOn": "flap"}],
        "assembly": ["board", "flap"],
        "switch": {"unit": "gf", "trigger": 65, "travel": 0.25, "gap": 0.4,
                   "presses": [{"at": [0, 0], "max": 300, "min": 100}, {"at": [6, -6], "max": 400, "min": 100}]},
    }


def test_handoff_reads_the_cad():
    b = product()
    assert not lint(b, "handoff"), lint(b, "handoff")
    r = handoff(b)
    st, rep = r["study"], r["report"]
    assert not rep["lint"], rep["lint"]
    paths = rep["assembly"]["components"]
    # the board's far end sits on the floor past the opening: it slides
    # back off it (-x, 2 mm and a little) and then drops out
    ins = paths["board"]["insertFrom"]
    assert isinstance(ins, list) and ins[0]["toward"] == "-x" and 2.0 <= ins[0]["length"] <= 3.5 and ins[1] == "-z", ins
    # the flap, with the switch on it, closes from below
    assert paths["flap"]["insertFrom"] == "-z", paths["flap"]
    assert paths["flap"]["travelsWith"] == ["switch"]
    keep = {k["id"]: k for k in st["keepOut"]}
    assert keep["switch"]["insertFrom"] == "-z"
    # the plunger: 0.4 mm over the actuator's top (z 5), up into the skin
    pl = rep["plunger"]
    assert abs(pl["tip"][2] - 5.4) < 0.02 and abs(pl["actuatorRadius"] - 1.0) < 0.05, pl
    assert abs(pl["tip"][0]) < 0.05 and abs(pl["tip"][1]) < 0.05, pl
    plunger = [p for p in st["preserve"] if p["id"] == "plunger"][0]["shape"]
    assert plunger["p1"][2] > 9.0, plunger
    # it stands on its underside; the switch bears on the flap
    sup = st["supports"][0]["shape"]
    assert sup["max"][2] < 0.5 and sup["min"][2] < 0, sup
    assert st["switch"]["output"]["bearsOn"]["part"]["seat"]["min"][2] < 2.0 < \
        st["switch"]["output"]["bearsOn"]["part"]["seat"]["max"][2]
    assert [c["component"] for c in st["switch"]["clearance"]] == ["board"]
    assert st["skin"]["openings"][0]["inward"] == "+z"


def test_handoff_study_runs():
    """The study it writes is self-contained (meshes copied beside it) and
    builds: plunger, seat compliance and the clearance per component."""
    from loadpath.domain import build_domain
    from loadpath.spec import study_from_dict
    from loadpath.switch import _thin_skin, fire_forces, switch_from_study
    b = product()
    out = tempfile.mkdtemp()
    bp = os.path.join(_DIR, "brief.json")
    json.dump(b, open(bp, "w"))
    r = run_handoff(bp, out)
    doc = json.load(open(r["studyPath"]))
    assert all(k["shape"]["path"].startswith("cad/") for k in doc["keepOut"] if k["shape"]["type"] == "mesh")
    st = study_from_dict(doc, out)
    dom, floor = _thin_skin(st, build_domain(st, 1.0))
    spec = switch_from_study(st, dom)
    assert spec.base_compliance > 0 and spec.clearance_entries[0]["id"] == "board"
    x = np.maximum(np.maximum(dom.preserve.astype(float), floor), 1e-9)
    f = fire_forces(dom, st.elastic(design=True), spec, x, float(st.material.E))
    assert np.isfinite(f["forces"]).all() and (f["forces"] > 0).all()
    # no skin across the opening's inner end: cells over the flap at
    # z 2-3 are design space, not skin
    c = dom.mesh.elem_centers()
    over = (np.abs(c[:, 0]) < 6) & (np.abs(c[:, 1]) < 6) & (c[:, 2] > 2) & (c[:, 2] < 3)
    assert not (floor[over] > 0).any() and not dom.preserve[over].any()


def test_blocked_component_is_reported():
    """A part with no way in, even with two legs, is reported with what it
    hits, not given a path."""
    import manifold3d as m3d
    b = product()
    b["components"].append({"id": "trapped", "shape": _stl(
        "trapped.stl", m3d.Manifold.cube((6, 6, 1.5)).translate((6, -13, 6)))})
    b["assembly"] = ["board", "flap", "trapped"]
    b["maxSlide"] = 3.0
    r = handoff(b)
    t = r["report"]["assembly"]["components"]["trapped"]
    assert t["insertFrom"] is None and t["blocked"]["hits"], t
    assert any("trapped" in w for w in r["report"]["warnings"])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name, flush=True)
            except Exception:
                fails += 1
                print("FAIL", name, flush=True)
                traceback.print_exc()
    sys.exit(1 if fails else 0)
