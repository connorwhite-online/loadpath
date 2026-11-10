"""
Cables laid as elastic rods (loadpath.wires): the cable library, laying a
cable between two ports in free space and round an obstacle, the checks
(too short, bent past its radius, a flat asked to turn in its own plane),
a cable followed through a joint's motion, the brief, and the judge the
placement search asks.

    python tests/test_wires.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import wires as W                        # noqa: E402
from loadpath.rods import Rod, SdfField, GridField     # noqa: E402


def test_cable_library():
    w = W.cable("awg26-pvc")
    assert w.kind == "round" and 1.0 < w.od < 1.6 and w.EI1 > 0 and w.GJ > 0
    assert abs(w.rmin_static - 3 * w.od) < 1e-9 and abs(w.rmin_dynamic - 10 * w.od) < 1e-9
    # silicone is softer than PVC on the same conductor; a thicker wire is stiffer
    assert W.cable("awg26-silicone").EI1 < w.EI1 < W.cable("awg20-pvc").EI1
    # a loose bundle: n times one wire (they slip), and its rule goes by one wire
    b = W.cable({"type": "bundle", "wires": 3, "awg": 26, "insulation": "pvc"})
    assert abs(b.EI1 - 3 * w.EI1) < 1e-9 * w.EI1 and abs(b.rmin_static - w.rmin_static) < 1e-9 and b.od > w.od
    f = W.cable("ffc-10x0.5")
    assert f.kind == "flat" and f.EI2 > 1e3 * f.EI1 and abs(f.rmin_static - 10 * f.thickness) < 1e-9
    assert len(f.offsets()) >= 3 and f.max_twist is not None
    # a datasheet overrides the house rule
    d = W.cable({"type": "wire", "awg": 26, "insulation": "pvc", "minBendRadius": {"static": 7, "dynamic": 20}})
    assert d.rmin_static == 7 and d.rmin_dynamic == 20
    for bad in ("nope", {"type": "round", "od": 2.0}):
        try:
            W.cable(bad)
            assert False
        except ValueError:
            pass
    # measured stiffness: the droop of an overhang gives EI back
    EI = 60.0
    n = 40
    x = np.zeros((n + 1, 3))
    x[:, 0] = np.linspace(0, 60.0, n + 1)
    gpm = 8.0
    r = Rod(x, bend=EI, twist=EI / 1.3, weight=np.array([0, 0, -gpm * 9.81e-6]))
    r.clamp(0)
    assert r.solve()["converged"]
    got = W.stiffness_from_droop(60.0 - 0.5 * r.l[0], -r.x[-1, 2], gpm)
    assert abs(got / EI - 1) < 0.05, got
    print(f"cable library ok (droop gives EI {got:.1f} for {EI})")


def _port(at, d, **kw):
    return {"at": list(at), "direction": d, **kw}


def test_lay_free_and_short():
    cab = W.cable("awg24-pvc")
    lw = W.lay(cab, _port((0, 0, 0), "+x"), _port((40, 0, 0), "-x"), slack=0.1)
    rep = lw.report()
    assert rep["ok"], rep["problems"]
    assert abs(rep["length"] - 44.0) < 0.5 and rep["minBendRadius"] > rep["allowedBendRadius"]
    # 10% slack between facing ports bows out: nowhere near straight
    assert np.abs(lw.rod.x[:, 1:]).max() > 3.0
    short = W.lay(cab, _port((0, 0, 0), "+x"), _port((40, 0, 0), "-x"), length=37.0).report()
    assert not short["ok"] and any("too short" in p for p in short["problems"]), short["problems"]
    # two ports side by side, facing the same way, closer than the cable can turn round
    u = W.lay(cab, _port((0, 0, 0), "+x"), _port((0, 4.0, 0), "+x"), length=14.0).report()
    assert any("bends to" in p for p in u["problems"]), u
    print(f"lay ok: slack bow, too short ({short['maxTension']:.1f} N), a U of {u['minBendRadius']:.1f} mm "
          f"against {u['allowedBendRadius']:.1f} refused")


def test_lay_round_a_block():
    from loadpath.shapes import shape_from_spec
    block = shape_from_spec({"type": "box", "min": [14, -12, -12], "max": [26, 12, 12]})
    cab = W.cable("awg26-silicone")
    f = W.body_field(block, 0.5)
    room = W._Room.of([f], np.array([-10, -30, -30.0]), np.array([50, 30, 30.0]), 0.75)
    lw = W.lay(cab, _port((0, 0, 0), "+x"), _port((40, 0, 0), "-x"), [f], room=room, slack=0.1)
    assert lw is not None
    rep = lw.report()
    assert rep["ok"], rep["problems"]
    assert rep["length"] > 55.0                      # round it, not through it
    d = np.array([block.sdf(p[None])[0] for p in lw.rod.x])
    assert d.min() > cab.radius - 0.2, d.min()
    # the space it needs, as a solid to keep clear
    ch = W.channel_mesh([lw], clearance=0.5)
    assert ch.is_watertight and ch.volume > np.pi * cab.radius ** 2 * rep["length"]
    # walled in: no way
    wall = shape_from_spec({"type": "box", "min": [14, -40, -40], "max": [26, 40, 40]})
    fw = W.body_field(wall, 0.5)
    room2 = W._Room.of([fw], np.array([-10, -30, -30.0]), np.array([50, 30, 30.0]), 0.75)
    assert W.lay(cab, _port((0, 0, 0), "+x"), _port((40, 0, 0), "-x"), [fw], room=room2) is None
    print(f"lay round a block ok: {rep['length']:.1f} mm, {d.min():.2f} mm off it, tightest {rep['minBendRadius']:.1f} mm")


def test_flat_cable():
    cab = W.cable("ffc-8x0.5")
    # face to face, a gentle S in its easy direction
    ok = W.lay(cab, _port((0, 0, 0), "+x", up="+z"), _port((30, 0, 6), "-x", up="+z"), slack=0.06).report()
    assert ok["ok"], ok["problems"]
    # the far connector turned a quarter turn about the cable's run: it has to twist in 30 mm
    tw = W.lay(cab, _port((0, 0, 0), "+x", up="+z"), _port((30, 0, 0), "-x", up="+y"), slack=0.06).report()
    assert tw["maxTwist"] > 5 * max(ok["maxTwist"], 1.0)
    assert any("twists" in p for p in tw["problems"]), tw
    print(f"flat cable ok: S bend {ok['minBendRadius']:.1f} mm; a quarter turn in 30 mm is {tw['maxTwist']:.0f} deg/m, refused")


def test_motion():
    brief = {"wires": [{"id": "w", "cable": {"type": "wire", "awg": 24, "insulation": "silicone"},
                        "from": _port((-30, 0, 0), "+x"), "to": _port((30, 0, 0), "-x", body="arm"), "slack": 0.0}],
             "bodies": [{"id": "arm", "shape": {"type": "box", "min": [30, -4, -4], "max": [60, 4, 4]}, "moves": True}],
             "motion": {"type": "revolute", "axis": "y", "origin": [0, 0, 0], "range": [-60, 60], "steps": 8}}
    res = W.harness(brief)
    m = res["wires"]["w"]["motion"]
    assert m["ok"], m["problems"]
    # through the axis the cable just bends: tightest near the ends of travel, and about clip distance x 0.6
    assert 10.0 < m["minBendRadius"] < 40.0 and m["allowedBendRadius"] == round(W.cable("awg24-silicone").rmin_dynamic, 3)
    assert len(res["laid"][0].history) >= 9
    # the same cable held off the axis comes taut as the joint opens away from it
    brief["wires"][0]["from"]["at"] = [-30, 0, 14]
    brief["wires"][0]["to"]["at"] = [30, 0, 14]
    bad = W.harness(brief)["wires"]["w"]["motion"]
    assert not bad["ok"], bad
    with tempfile.TemporaryDirectory() as d:
        doc = W.write_harness(res, brief, d)
        assert os.path.exists(os.path.join(d, "wires.html")) and os.path.exists(doc["channelFiles"]["w"])
    print(f"motion ok: through the axis {m['minBendRadius']:.1f} mm (allowed {m['allowedBendRadius']:.1f}); "
          f"off it: {bad['problems'][0]}")


def test_placement_judge():
    comps = [{"id": "a", "ports": [{"id": "p", "at": [5, 0, 0], "direction": "+x"}]},
             {"id": "b", "ports": [{"id": "p", "at": [-5, 0, 0], "direction": "-x"}]}]
    wires = [{"id": "w", "from": "a.p", "to": "b.p", "cable": "awg26-silicone", "length": 30.0}]
    judge = W.placement_judge(wires, comps)
    h = 1.0
    room = np.full((60, 30, 30), 5.0)
    info = {"room": room, "origin": np.array([-30.0, -15, -15]), "pitch": h, "new": ["b"]}
    I = np.eye(3)
    # b not placed yet: nothing to say
    assert judge({**info, "transforms": {"a": (I, np.array([-15.0, 0, 0]))}, "new": ["a"]}) == 0.0
    near = judge({**info, "transforms": {"a": (I, np.array([-18.0, 0, 0])), "b": (I, np.array([18.0, 0, 0]))}})
    assert near is not None and near > 0
    # 46 mm between the ports and 30 mm of cable
    assert judge({**info, "transforms": {"a": (I, np.array([-28.0, 0, 0])), "b": (I, np.array([28.0, 0, 0]))}}) is None
    # back to back at 2 mm: the cable would have to fold
    Rz = np.diag([-1.0, -1.0, 1.0])
    fold = judge({**info, "transforms": {"a": (Rz, np.array([-6.0, 0, 0])), "b": (Rz, np.array([6.0, 0, 0]))}})
    assert fold is None or fold > near
    print(f"placement judge ok: {near:.1f} near, too far refused, folded {fold}")


def test_brief_schema():
    try:
        import jsonschema
    except ImportError:
        print("brief schema: jsonschema not installed, skipped")
        return
    from loadpath.schema import SCHEMAS
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "wires"))
    import joint
    import json
    with tempfile.TemporaryDirectory() as d:
        b = json.load(open(joint.write(d)))
        jsonschema.validate(b, SCHEMAS["wires"])
        import handheld
        e = json.load(open(handheld.write(d, lead=45.0)))
        jsonschema.validate(e, SCHEMAS["enclosure"])
    bad = {"wires": [{"cable": "awg26-pvc", "from": {"at": [0, 0, 0]}, "to": {"at": [1, 0, 0], "direction": "+x"}}]}
    try:
        jsonschema.validate(bad, SCHEMAS["wires"])
        assert False
    except jsonschema.ValidationError:
        pass
    print("brief schema ok")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all wires tests passed")
