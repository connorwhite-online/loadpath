"""Choosing a core by process (cores.py): the lightest option that meets its
loads and can be built. tune_field is stubbed (its own tests are in
test_tune.py); the meshing and the print reading are real."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import loadpath.tune as tune  # noqa: E402
from loadpath.cores import choose_core  # noqa: E402

BOX = {"op": "shape", "shape": {"type": "box", "min": [0, -15, -8], "max": [80, 15, 8]}}


def _graph(cavity, rib=False):
    """A 80 x 30 x 16 mm block with a sealed hollow of half-width `cavity`
    (none at 0) and, if `rib`, a 1 mm rib down its middle."""
    nodes = {"block": BOX}
    if cavity:
        nodes["hole"] = {"op": "shape", "shape": {"type": "box", "min": [6, -cavity, -6], "max": [74, cavity, 6]}}
        cut = {"op": "subtract", "inputs": ["block", "hole"]}
        nodes["part"] = ({"op": "union", "inputs": [cut, {"op": "shape", "shape": {
            "type": "box", "min": [6, -0.5, -6], "max": [74, 0.5, 6]}}]} if rib else cut)
    else:
        nodes["part"] = {"op": "intersect", "inputs": ["block", "block"]}
    return {"nodes": nodes, "output": "part", "material": "petg",
            "tune": {"study": "s.json", "params": [{"path": "nodes/block/shape/max/0", "min": 70, "max": 90}]}}


@pytest.fixture
def stub(monkeypatch):
    masses = {}

    def fake(doc, base_dir=None, out_dir=None, progress=None, evaluations=None):
        name = doc["name"]
        return {"doc": doc, "best": {"values": [80.0], "mass": masses[name], "safetyFactor": 3.0,
                                     "deflection": 0.4, "ok": name != "weak",
                                     "check": {"deflection": 0.45, "ok": name != "weak"}}}

    monkeypatch.setattr(tune, "tune_field", fake)
    return masses


def test_fdm_refuses_a_sealed_flat_roof_and_sls_takes_it(stub):
    stub.update(hollow=1, ribbed=2, solid=3, weak=0)
    opts = [{"name": n, "graph": {**g, "name": n}} for n, g in
            (("hollow", _graph(13)), ("ribbed", _graph(13, rib=True)), ("solid", _graph(0)), ("weak", _graph(13)))]
    out = tempfile.mkdtemp()
    rep = choose_core({"options": opts, "processes": ["fdm", "sls"]}, out_dir=out)
    by = {o["name"]: o for o in rep["options"]}
    # a 26 mm flat roof over a sealed void: support trapped inside
    assert by["hollow"]["print"]["trappedSupportArea"] > 500 and not by["hollow"]["processes"]["fdm"]["ok"]
    # the rib halves it to two 12.5 mm bridges
    assert by["ribbed"]["print"]["trappedSupportArea"] <= 50 and by["ribbed"]["processes"]["fdm"]["ok"]
    assert by["hollow"]["processes"]["sls"]["ok"] and "drains" in by["hollow"]["processes"]["sls"]["note"]
    assert not by["weak"]["meetsLoads"] and not by["weak"]["processes"]["sls"]["ok"]
    assert rep["winners"]["fdm"]["name"] == "ribbed" and rep["winners"]["sls"]["name"] == "hollow"
    assert json.load(open(Path(out) / "fdm.json"))["name"] == "ribbed"
    # a second run in the same folder reuses every option
    tune.tune_field = None
    again = choose_core({"options": opts, "processes": ["fdm", "sls"]}, out_dir=out)
    assert again["winners"] == rep["winners"]
