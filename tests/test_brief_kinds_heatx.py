"""Heat-exchanger briefs (loadpath/evals/brief_kinds/heatx.py): each gold
spec passes its own scorer, and deliveries that miss the point (no spec, a
sizing request instead of a design, swapped fluids, a mesh that is not the
spec's part) fail.

The oil cooler runs on the quick grid (about 20 s a design); the other two
golds design on 16M-point grids (1-2 minutes each) and run only with
LOADPATH_SLOW_TESTS=1."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath.evals import briefs  # noqa: E402
from loadpath.evals.brief_kinds import heatx as H  # noqa: E402

IDS = ["hpu-oil-cooler-lpbf", "coolant-isolator-petg", "kart-charge-cooler-lpbf"]
FAST = "hpu-oil-cooler-lpbf"
SLOW = pytest.mark.skipif(not os.environ.get("LOADPATH_SLOW_TESTS"),
                          reason="16M-point designs; set LOADPATH_SLOW_TESTS=1")


def _write(folder, spec, mesh=None):
    with open(os.path.join(folder, "exchanger.json"), "w") as fh:
        json.dump(spec, fh)
    if mesh is not None:
        mesh.export(os.path.join(folder, "exchanger.stl"))
    return str(folder)


def _check_gold(bid, quick):
    b = briefs.load(bid)
    assert b["kind"] == "heatx"
    res = H.gold(b, quick=quick)
    assert res["ok"], res.get("error")
    s, want = res["score"], b["pass"]
    assert s["isolated"] and all(s["connected"].values())
    for n, cap in want["maxPressureDrop"].items():
        assert s["pressureDrop"][n] <= cap
    if "minDuty" in want:
        assert s["duty"] >= want["minDuty"]
    if "minEffectiveness" in want:
        assert s["effectiveness"] >= want["minEffectiveness"]


def test_briefs_load_and_prompt_asks_for_an_exchanger():
    for bid in IDS:
        b = briefs.load(bid)
        assert briefs.kind(b) is H
        text = briefs.prompt(b)
        assert "exchanger.json" in text and "exchanger.stl" in text and "part.stl     the part" not in text
        # the request is a user's words, not loadpath's fields
        for field in ("portDiameter", "layout", "heatExchanger", "\"duty\""):
            assert field not in b["brief"]


def test_fast_gold_passes_its_scorer():
    _check_gold(FAST, quick=True)


@SLOW
@pytest.mark.parametrize("bid", IDS[1:])
def test_slow_gold_passes_its_scorer(bid):
    _check_gold(bid, quick=False)


def test_missing_or_sizing_only_delivery_fails(tmp_path):
    b = briefs.load(FAST)
    row = briefs.score_dir(b, str(tmp_path))
    assert not row["pass"] and row["failures"] == ["no exchanger.json"]
    spec = {"name": "x", "envelope": [90, 50, 40], "fluids": ["oil", "coolant"], "duty": b["gold"]["duty"],
            "sizing": {"process": "lpbf"}}
    row = briefs.score_dir(b, _write(tmp_path, spec))
    assert not row["pass"] and "no core" in row["failures"][0]


def test_swapped_fluids_fail(tmp_path):
    """The gold geometry with the fluid names swapped: each port is the other
    fluid's, so the ports land in the wrong places (and the oil gets the
    coolant's channels)."""
    b = briefs.load(FAST)
    spec = json.loads(json.dumps(b["gold"]))
    spec["fluids"] = ["coolant", "oil"]
    spec["portDiameter"] = {"coolant": 8, "oil": 8}
    row = H.score(b, _write(tmp_path, spec), quick=True)
    assert not row["pass"]
    txt = " ".join(row["failures"])
    assert "oil-in port points" in txt and "coolant-in port points" in txt
    assert "no exchanger.stl" in txt


def test_block_instead_of_the_part_fails(tmp_path):
    """The gold spec, but a solid block delivered as the part (and the
    process changed to FDM): the scorer designs the spec itself, so the
    numbers it checks are the spec's, and the mesh must be that part."""
    import trimesh

    b = briefs.load(FAST)
    spec = dict(b["gold"], process="fdm")
    block = trimesh.creation.box(extents=[84, 49, 31])
    block.apply_translation([45, 25, 20])
    row = H.score(b, _write(tmp_path, spec, block), quick=True)
    assert not row["pass"]
    txt = " ".join(row["failures"])
    assert "exchanger.stl is not the part" in txt and "process 'fdm'" in txt
