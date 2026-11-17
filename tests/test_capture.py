"""
Capture (loadpath/capture.py): what holds a USB-C socket against the
plug's forces, decided by how the product goes together. A socket slid in
along its own axis has nothing behind it in the part it slid into; dropped
in across its axis and closed over, it is clamped. split (shells.py) uses
that to choose the cut and preserves the faces that bear on the socket.
About a minute.

Run: `python3 tests/test_capture.py`
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.capture import bearing_pad, choose_way_in, measure, plan, service_loads  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.shapes import Box  # noqa: E402

LOADS = service_loads("usb-c", out="+y", wide="+x")


def _row(p, name):
    return next(r for r in p["loads"] if r["name"] == name)


def test_loads():
    names = {r["name"]: r for r in LOADS}
    assert names["insert"]["force"] == 20.0 and np.allclose(names["insert"]["d"], [0, -1, 0])
    assert np.allclose(names["withdraw"]["d"], [0, 1, 0])
    # both ways across each side
    assert {"side-thin+", "side-thin-", "side-wide+", "side-wide-"} <= set(names)
    assert names["side-wide+"]["force"] == 20.0 and names["side-thin+"]["force"] == 8.0


def test_slide_in_has_nothing_behind():
    # the socket slid forward into its port (it leaves backward, -y): the
    # body it slid into cannot stand behind it, and nothing else is there
    p = plan(LOADS, "-y", [{"id": "body"}], "body")
    ins = _row(p, "insert")
    assert ins["ok"] is False and ins["by"] == "legs", ins
    assert "block the socket's way in" in ins["why"]
    # pulling out pushes it into the end of its slide: a seated face
    assert _row(p, "withdraw")["face"] == "seated"
    assert p["ok"] is False


def test_clamshell_holds_everything():
    # dropped into the body from below and a lid closed under it on screws
    parts = [{"id": "body"}, {"id": "lid", "after": True, "leave": "-z", "joints": "screws"}]
    p = plan(LOADS, "-z", parts, "body")
    assert p["ok"] is True, p
    assert _row(p, "insert")["by"] == "body" and _row(p, "insert")["face"] == "press"
    assert _row(p, "side-thin+")["by"] == "lid" and _row(p, "side-thin+")["face"] == "seated"
    assert _row(p, "side-thin-")["by"] == "body" and _row(p, "side-thin-")["face"] == "seated"
    assert p["style"].startswith("clamped between")
    # a lid held on by snaps keeps their play: the push into it goes to the legs
    parts[1].update(joints="snaps", clearance=0.2)
    p2 = plan(LOADS, "-z", parts, "body")
    assert _row(p2, "side-thin+")["ok"] is False and "0.20 mm of play" in _row(p2, "side-thin+")["why"]


def test_no_room_is_the_datasheet():
    # the socket's whole front is the plug's way: no face can take the
    # pull in any assembly; the legs' rating decides
    p = plan(LOADS, "-z", [{"id": "body", "room": {"withdraw": 0.0}}], "body")
    w = _row(p, "withdraw")
    assert w["ok"] is None and "retention" in w["why"]
    p2 = plan(LOADS, "-z", [{"id": "body", "room": {"withdraw": 0.0}}], "body", retention={"withdraw": 30})
    assert _row(p2, "withdraw")["ok"] is True


def test_choose_way_in():
    parts = [{"id": "body"}, {"id": "lid", "after": True, "leave": "-z", "joints": "screws"}]
    best = choose_way_in(LOADS, parts, "body", candidates=["+y", "+z"])
    # arriving along its own axis (+y) leaves the push-in on a press-fit lid
    # at best; dropped in (+z) the body itself stands behind it
    assert best[0]["socketIn"] == "+z", [(b["socketIn"], b["fits"]) for b in best]


def test_measure_and_pad():
    sock = Box([-4.5, 12.0, 7.6], [4.5, 19.6, 10.9])
    wall = Box([-10, 0.0, 0.0], [10, 11.95, 20.0])        # 0.05 mm behind the socket
    lid = Box([-10, 8.0, 11.2], [10, 22.0, 13.0])          # 0.3 mm over it
    r = measure(sock, {"body": wall, "lid": lid}, LOADS, "body", step=0.3)
    ins = next(x for x in r["loads"] if x["name"] == "insert")
    assert ins["by"] == "body" and abs(ins["gap"] - 0.05) < 0.03 and ins["ok"], ins
    up = next(x for x in r["loads"] if x["d"] == "+z")
    assert up["by"] == "lid" and abs(up["parts"]["lid"]["gap"] - 0.3) < 0.03 and not up["ok"], up
    pad = bearing_pad(sock, [0, -1, 0], 0.0, 1.0, step=0.1)
    assert pad is not None and abs(abs(pad.volume) - 9.0 * 3.3 * 1.0) < 3.0, abs(pad.volume)


def _product():
    """A 60 x 40 x 20 box with a USB-C socket on a board, its front in a
    port through the +y wall."""
    port = {"type": "box", "min": [-4.8, 18.2, 7.3], "max": [4.8, 21, 11.1]}
    return {"name": "porty",
            "designSpace": {"type": "subtract", "shapes": [{"type": "box", "min": [-30, -20, 0], "max": [30, 20, 20]},
                                                           port]},
            "skin": {"thickness": 1.5, "openings": [{"shape": port, "inward": "-y", "depth": 2.0}]},
            "material": "pa12", "process": {"type": "sls"}, "minMember": 1.0,
            "keepOut": [{"id": "board", "shape": {"type": "box", "min": [-20, -15, 6.0], "max": [20, 18.0, 7.6]},
                         "insertFrom": "+z", "bore": False},
                        {"id": "usb", "shape": {"type": "box", "min": [-4.47, 12.0, 7.6], "max": [4.47, 19.6, 10.86]},
                         "installedOn": "board", "bore": False,
                         "connector": {"type": "usb-c", "out": "+y", "wide": "+x"}}],
            "supports": [{"shape": {"type": "box", "min": [-31, -21, -1], "max": [31, 21, 0.3]}}],
            "loadCases": [{"name": "press", "loads": [{"shape": {"type": "box", "min": [-5, -5, 19.5],
                                                                   "max": [5, 5, 20.5]}, "force": [0, 0, -40]}]}],
            "goal": {"safetyFactor": 1.5}}


def test_split_chooses_a_cut_that_captures():
    from loadpath.shells import apply_split
    out = tempfile.mkdtemp()
    sp = {"normal": "+z", "screws": {"from": "-z", "size": "M2"}, "lip": False}
    # under the board: the board and socket go into the cover, sliding
    # forward into the port, so nothing can stand behind the socket
    flat = apply_split(_product(), {**sp, "at": 4.0}, out_dir=out, base_dir=out)["report"]
    cap = flat["capture"]["usb"]
    assert cap["ok"] is False and _row(cap, "insert")["by"] == "legs", cap
    assert any("solder joints" in w for w in flat["warnings"])
    # searched: a cut that clamps the socket wins over the shorter-seam one
    r = apply_split(_product(), sp, out_dir=out, base_dir=out)
    rep = r["report"]
    assert rep["cut"]["captureOk"] is True, rep["cut"]
    cap = rep["capture"]["usb"]
    assert _row(cap, "insert")["ok"] is True and _row(cap, "insert")["by"] in ("base", "cover")
    assert any(c.get("clamshell") == "usb" for c in rep["candidates"])
    # each face is preserved in its half with the plug's force on it and
    # its travel held
    faces = cap["faces"]
    assert any(f["load"] == "insert" for f in faces), faces
    for f in faces:
        h = r["studies"][f["half"]]
        assert f["id"] in {p["id"] for p in h["preserve"]}
        assert any(ld.get("region") == f["id"] for lc in h["loadCases"] for ld in lc["loads"])
        assert any(c["region"] == f["id"] and c["maxTravel"] == 0.1 for c in h["goal"]["clearances"])
    for h in r["studies"].values():
        errs = [x for x in lint(h, "study") if x["level"] == "error"]
        assert not errs, errs


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception:  # noqa: BLE001
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if fails else 0)
