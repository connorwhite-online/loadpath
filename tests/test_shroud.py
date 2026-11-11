"""
Tests for generated exteriors (shroud.py): the envelope holds the
interfaces and what `conceal` lists, stays out of the keep-outs it does
not, stays inside the design space, the shape steps never make the skin
worse per volume, and the expanded study is an ordinary skinned study
(the skin preserved, bores cutting it). Small grids only.

Run: `python3 tests/test_shroud.py`.
"""
import copy
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

if not os.environ.get("LOADPATH_CACHE"):
    os.environ["LOADPATH_CACHE"] = tempfile.mkdtemp(prefix="lp-shroud-")

from loadpath.domain import build_domain
from loadpath.schema import lint
from loadpath.shapes import GridSDF
from loadpath.shroud import expand_shroud, write_shroud
from loadpath.spec import study_from_dict

# a short link: a boss at each end, a channel between them that leaves
# through a port, a board to hide, and a mating part's room to stay out of
LINK = {
    "name": "shroud-link",
    "designSpace": {"type": "box", "min": [-12, -14, -9], "max": [72, 14, 9]},
    "preserve": [
        {"id": "a-boss", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 14, "r": 9}},
        {"id": "b-boss", "shape": {"type": "cylinder", "center": [60, 0, 0], "axis": "z", "length": 14, "r": 9}},
    ],
    "keepOut": [
        {"id": "a-bore", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 20, "r": 5}},
        {"id": "b-bore", "shape": {"type": "cylinder", "center": [60, 0, 0], "axis": "z", "length": 20, "r": 5}},
        {"id": "channel", "shape": {"type": "cylinder", "p0": [14, 0, 0], "p1": [46, 0, 0], "r": 3}},
        {"id": "port", "shape": {"type": "cylinder", "p0": [46, 0, 0], "p1": [46, 0, 12], "r": 3}},
        {"id": "board", "bore": False, "shape": {"type": "box", "min": [24, -9, -4], "max": [36, -5, 4]}},
        {"id": "neighbour", "bore": False, "shape": {"type": "box", "min": [26, 11, -10], "max": [34, 15, 10]}},
    ],
    "shroud": {"thickness": 2.0, "reach": 6, "smooth": 4, "conceal": ["channel", "board"],
               "seedCells": 3000, "seedIterations": 25, "rounds": 3},
    "supports": [{"region": "a-bore"}],
    "loads": [{"region": "b-bore", "force": [0, -30, 0]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.3},
    "minMember": 3.0,
    "resolution": {"pitch": 2.0},
}


def _expanded():
    if "_x" not in _CACHE:
        _CACHE["_x"] = expand_shroud(copy.deepcopy(LINK), str(ROOT))
    return _CACHE["_x"]


_CACHE = {}


def test_lints():
    assert not [i for i in lint(LINK, "study") if i["level"] == "error"], lint(LINK, "study")
    ex = json.loads((ROOT / "examples" / "shroud" / "arm-link.json").read_text())
    assert not [i for i in lint(ex, "study") if i["level"] == "error"]


def test_envelope_holds_what_it_should():
    x = _expanded()
    assert x["designSpace"]["type"] == "sdf" and x["skin"]["thickness"] == 2.0 and "shroud" not in x
    env = GridSDF.load(x["_shroud"]["path"])
    # interfaces and concealed regions inside, with the skin and gap clear of the board
    for p in ([0, 9 - 0.5, 0], [60, -8.5, 0], [30, 0, 0], [30, -7, 0]):
        assert env.sdf(np.array([p]))[0] < 0, p
    board_corner = np.array([[24, -9, -4]])
    assert env.sdf(board_corner)[0] <= -(2.0 + 1.0) + 0.25, env.sdf(board_corner)
    # the neighbour's room stays outside, the design space's walls bound it
    assert env.sdf(np.array([[30, 13, 0]]))[0] > 0
    lo, hi = np.array(LINK["designSpace"]["min"]), np.array(LINK["designSpace"]["max"])
    g = env.grid
    axes = [env.origin[i] + env.pitch * np.arange(g.shape[i]) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    inside = g <= 0
    assert np.all(P[inside] >= lo - 0.75) and np.all(P[inside] <= hi + 0.75)
    info = x["_shroud"]
    assert info["envelope"]["volume"] < info["envelope"]["designSpaceVolume"]


def test_shape_steps_only_improve():
    h = _expanded()["_shroud"]["envelope"]["refine"]["history"]
    js = [r["J"] for r in h]
    assert all(b < a for a, b in zip(js, js[1:])), js


def test_expanded_study_is_skin_and_rib():
    x = _expanded()
    st = study_from_dict(copy.deepcopy(x), str(ROOT))
    ids = [r.id for r in st.preserve]
    assert "skin" in ids
    d = build_domain(st)
    c = d.mesh.elem_centers()
    # the skin is preserved round the middle of the link, and the bores cut it
    skin = d.preserve & (np.abs(c[:, 0] - 30) < 3)
    assert skin.sum() > 10
    near_port = np.linalg.norm(c[:, [0, 1]] - [46, 0], axis=1) < 2.0
    assert not (d.preserve & near_port & (c[:, 2] > 5)).any(), "the port must open through the skin"
    # parsing the expanded study again reuses it (no second seed)
    t0 = time.time()
    expand_shroud(copy.deepcopy(LINK), str(ROOT))
    assert time.time() - t0 < 20


def test_refuses_skin_with_shroud():
    bad = {**copy.deepcopy(LINK), "skin": 2.0}
    try:
        expand_shroud(bad, str(ROOT))
    except ValueError as e:
        assert "skin" in str(e)
    else:
        raise AssertionError("shroud + skin should refuse")
    bad = copy.deepcopy(LINK)
    bad["shroud"]["conceal"] = ["nope"]
    try:
        expand_shroud(bad, str(ROOT))
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("an unknown conceal id should refuse")


def test_write_shroud():
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "link.json"
        sp.write_text(json.dumps(LINK))
        r = write_shroud(str(sp), str(Path(tmp) / "out"), base_dir=str(ROOT))
        assert Path(r["envelopeMesh"]).exists() and Path(r["study"]).exists()
        doc = json.loads(Path(r["study"]).read_text())
        assert doc["designSpace"]["shroud"] == doc["_shroud"]["key"]
        import trimesh
        m = trimesh.load(r["envelopeMesh"])
        assert m.is_watertight


def test_drone_canopy():
    """The drone canopy example: battery, VTX, camera and RX inside with the
    skin and gap clear of them, the camera window open through the skin and
    the cooling block still meaningful on the expanded study."""
    d = json.loads((ROOT / "examples" / "shroud" / "drone-canopy.json").read_text())
    assert not [i for i in lint(d, "study") if i["level"] == "error"]
    d["resolution"] = {"pitch": 2.0}
    d["shroud"].update({"seedCells": 4000, "seedIterations": 20, "rounds": 2})
    x = expand_shroud(d, str(ROOT / "examples" / "shroud"))
    env = GridSDF.load(x["_shroud"]["path"])
    t, gap = d["shroud"]["thickness"], d["shroud"]["gap"]
    boxes = {r["id"]: r["shape"] for r in d["keepOut"] if r["id"] in d["shroud"]["conceal"]}
    for rid, b in boxes.items():
        lo, hi = np.array(b["min"], float), np.array(b["max"], float)
        corners = np.array([[(lo, hi)[i >> k & 1][k] for k in range(3)] for i in range(8)])
        # the floor side sits on the frame; the rest keeps skin and gap clear
        top = corners[corners[:, 2] > lo[2] + 1e-9]
        sd = env.sdf(top)
        assert np.all(sd < 0), (rid, sd)
        assert np.all(sd <= -(t + gap) + 1.0), (rid, sd)
    st = study_from_dict(copy.deepcopy(x), str(ROOT / "examples" / "shroud"))
    dom = build_domain(st)
    c = dom.mesh.elem_centers()
    win = (np.linalg.norm(c[:, [1, 2]] - [0, 17.5], axis=1) < 5) & (c[:, 0] > 42)
    assert not (dom.preserve & win).any(), "the camera window must open through the skin"
    assert (st.raw or {}).get("cooling", {}).get("components"), "cooling must survive the expansion"


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)


def test_shroud_places_parts_with_no_drawn_outside():
    """shroud.place: parts laid out in the room (no exterior drawn), each
    the skin and its gap off the room's walls and clear of the others, then
    concealed; a part with watts joins the cooling block."""
    d = json.loads((ROOT / "examples" / "shroud" / "drone-canopy-placed.json").read_text())
    assert not [i for i in lint(d, "study") if i["level"] == "error"]
    d["resolution"] = {"pitch": 2.0}
    d["shroud"].update({"seedCells": 4000, "seedIterations": 20, "rounds": 0})
    x = expand_shroud(d, str(ROOT / "examples" / "shroud"))
    pl = x["_shroud"]["placement"]["components"]
    assert set(pl) == {"battery", "vtx", "rx"}
    t, gap = d["shroud"]["thickness"], d["shroud"]["gap"]
    for k, v in pl.items():
        assert v["roomToSkin"] >= -1e-6, (k, v["roomToSkin"])
        assert min(v["roomToParts"].values()) > 0, (k, v["roomToParts"])
    assert {"battery", "vtx", "rx", "camera"} <= set(x["_shroud"]["conceal"])
    assert any(c["region"] == "vtx" and c["watts"] == 4.0 for c in x["cooling"]["components"])
    env = GridSDF.load(x["_shroud"]["path"])
    for k, v in pl.items():
        lo, hi = (np.asarray(b, float) for b in v["bounds"])
        c = 0.5 * (lo + hi)
        assert env.sdf(c[None])[0] < -(t + gap), k
