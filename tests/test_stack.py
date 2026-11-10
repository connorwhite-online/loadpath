"""Board stacks on standoffs (stack.py): the gap's own air speed, the
resistance network, the search, and the drone example's leads."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "wires"))

from loadpath import stack as S


def _boards(watts=7.5):
    return [{"id": "fc", "size": [27, 27, 1.6], "mass": 0.0055, "watts": 0.8, "maxTemp": 85},
            {"id": "esc", "size": [27, 27, 1.6], "mass": 0.0075, "watts": watts, "maxTemp": 105}]


def test_gap_flow():
    # no stream, no flow; a wider slot loses less to friction, and never beats the stream
    assert S.gap_flow(0.0, 4.0, 27.0) == 0.0
    u = [S.gap_flow(8.0, g, 27.0) for g in (1.0, 2.0, 4.0, 8.0)]
    assert all(b > a for a, b in zip(u, u[1:])), u
    assert 0.0 < u[0] and u[-1] < 8.0
    # the entry and exit loss alone: dynamic pressure of the stream against K = 1.5
    assert abs(S.gap_flow(8.0, 50.0, 27.0) - 8.0 / np.sqrt(1.5)) < 0.4
    print(f"gap flow ok: {u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}, {u[3]:.2f} m/s in 1, 2, 4, 8 mm at 8 m/s")


def test_temperatures():
    b = _boards()
    kw = dict(base_gap=2.0, ambient=30.0)
    def T(boards, speed):
        t = S.temperatures(boards, [5.0], speed=speed, **kw)["temperatures"]
        return {x["id"]: v for x, v in zip(boards, t)}
    still, wash, fast = T(b, 0.0), T(b, 8.0), T(b, 16.0)
    # air helps, more air helps more, and nothing is cooler than the air it sits in
    assert still["esc"] > wash["esc"] > fast["esc"] > 30.0, (still, wash, fast)
    assert wash["fc"] > 30.0 and wash["esc"] > wash["fc"]
    # no heat: everything at ambient
    cold = T([{**x, "watts": 0.0} for x in b], 8.0)
    assert max(abs(v - 30.0) for v in cold.values()) < 0.11, cold
    # film properties barely move: twice the heat is close to twice the rise
    half = T(_boards(3.75), 8.0)
    r = (wash["esc"] - 30.0) / (half["esc"] - 30.0)
    assert 1.6 < r < 2.2, r
    print(f"temperatures ok: 7.5 W board {still['esc']:.0f} C in still air, {wash['esc']:.0f} at 8 m/s, {fast['esc']:.0f} at 16")


def test_stack_search_and_cables():
    import drone
    brief = drone.stack_brief(8.0)
    res = S.stack(brief)
    assert res["ok"], res.get("problems")
    best = res["best"]
    # both boards inside their limits and inside the room
    assert best["temperatures"]["esc"] <= 105 and best["temperatures"]["fc"] <= 85
    assert best["height"] <= brief["space"]["height"]
    # what the frame's study takes: one carried mass at the stack's centre of mass
    m = res["inertiaRelief"]["masses"][0]
    assert m["region"] == "stack-posts" and abs(m["mass"] - best["mass"]) < 1e-9
    assert m["at"] == best["centerOfMass"]
    assert 0.011 < m["mass"] < 0.02                # the boards, and standoffs worth a gram or so
    z = best["z"]
    cg = (0.0055 * (z["fc"] + 0.8) + 0.0075 * (z["esc"] + 0.8)) / 0.013
    assert abs(m["at"][2] - cg) < 1.0, (m, cg)
    assert res["envelope"]["max"][2] == best["height"]
    # the harness decides: a tighter stack scores better and is refused because its cable folds
    w = best["wires"]["signal"]
    assert w["ok"] and w["minBendRadius"] >= 0.98 * w["allowedBendRadius"]
    tighter = [r for r in res["rejected"] if r["height"] < best["height"]]
    assert tighter and all("signal" in " ".join(r["problems"]) for r in tighter), res["rejected"]
    # no cable: the lower, tighter stack wins
    free = S.stack({k: v for k, v in brief.items() if k != "wires"})
    assert free["best"]["height"] < best["height"]
    # still air: 7.5 W on a 27 mm board cannot be held, and the answer says so
    hot = S.stack({**{k: v for k, v in brief.items() if k != "wires"}, "airflow": {"speed": 0.0}})
    assert not hot["ok"] and hot["problems"]
    print(f"stack ok: {best['order']} gap {best['gaps'][0]:.1f} mm, esc {best['temperatures']['esc']:.0f} C, "
          f"CG z {m['at'][2]:.1f} mm; {len(tighter)} tighter stacks refused by the harness; "
          f"without it {free['best']['order']} at {free['best']['height']:.1f} mm")


def test_drone_leads():
    import drone
    from loadpath import wires as W
    st = {"z": {"fc": 2.0, "esc": 9.6}}
    with tempfile.TemporaryDirectory() as d:
        ok = W.harness(json.load(open(drone.wires_brief(d, st))), base_dir=d)
        short = W.harness(json.load(open(drone.wires_brief(d, st, lead_length=38.0, pack_z=22.0))), base_dir=d)
    assert ok["ok"], ok["problems"]
    lead = ok["wires"]["battery-lead"]
    # held to the fixed radius over a one-off slide, and the pads' load is a vector with a limit on it
    assert lead["motion"]["ok"] and lead["motion"]["minBendRadius"] >= 0.98 * lead["allowedBendRadius"]
    f = lead["ends"]["from"]
    assert len(f["force"]) == 3 and abs(f["pull"]) < 15.0 and f["moment"] < 150.0
    assert abs(np.hypot(f["pull"], f["side"]) - np.linalg.norm(f["force"])) < 1e-3
    # the clip on the arm carries a load of its own
    clips = ok["wires"]["motor-leads"]["clips"]
    assert len(clips) == 1 and 0.0 < np.linalg.norm(clips[0]["force"]) < 5.0, clips
    # the pack in line behind the stack on a short lead: taut when it slides
    assert not short["ok"] and any("taut" in p for p in short["problems"]), short["problems"]
    print(f"drone leads ok: battery lead {lead['motion']['minBendRadius']:.1f} mm against {lead['allowedBendRadius']:.1f}, "
          f"pads {f['force']}; arm clip {np.linalg.norm(clips[0]['force']):.2f} N; the short in-line lead comes taut")


def test_brief_schema():
    try:
        import jsonschema
    except ImportError:
        print("brief schema: jsonschema not installed, skipped")
        return
    import drone
    from loadpath.schema import SCHEMAS
    jsonschema.validate(drone.stack_brief(), SCHEMAS["stack"])
    with tempfile.TemporaryDirectory() as d:
        jsonschema.validate(json.load(open(drone.wires_brief(d, {"z": {"fc": 2.0, "esc": 9.6}}, strap=True))),
                            SCHEMAS["wires"])
    print("brief schema ok")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
