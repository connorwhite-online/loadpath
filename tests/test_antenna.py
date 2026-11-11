"""
A board antenna in the product it lives in (loadpath/antenna.py): the
radio-frequency material data and its sources, a mesh turned into cell
fractions, an inverted-F solved bare and inside a printed shell with a
battery, the sensitivity field against full solves, the dimensions of an
antenna solved from the fields, and what the placement search asks.

    python tests/test_antenna.py
"""
import copy
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import antenna as A                      # noqa: E402
from loadpath.schema import SCHEMAS, lint              # noqa: E402

# an inverted-F on a 32 x 56.6 mm, 1 mm FR-4 board (a 50 mm ground), its arm
# and tap as `tune` solved them for 2441.75 MHz
ARM, TAP = 22.467, 3.516


def ifa_brief(arm=ARM, tap=TAP):
    return A.board_brief(A.ifa(arm=arm, height=6.0, tap=tap), width=32.0, ground=50.0, sweep=(1800, 3100))


SHELL = [{"name": "shell", "material": "pa12", "box": [[-4, -4, -8], [36, 62, 7]]},
         {"name": "cavity", "material": "air", "box": [[-2, -2, -6], [34, 60, 5]]}]


def _z(out, f):
    return complex(np.interp(f, out["freqs"], out["z"].real), np.interp(f, out["freqs"], out["z"].imag))


def test_materials():
    for name, lo, hi in (("pla", 2.6, 3.1), ("petg", 2.5, 2.9), ("abs", 2.3, 2.8), ("tpu", 2.7, 3.3),
                         ("pa12", 2.4, 2.9), ("fr4", 3.7, 4.6)):
        eps, tan, note = A.rf_properties(name)
        assert lo <= eps <= hi and 0.0 < tan < 0.1 and len(note) > 20, (name, eps, tan)
    # PA12 has no GHz measurement: its card says it is an estimate
    assert "estimate" in A.rf_properties("pa12")[2]
    # a value given in the brief wins; a material with none on file is asked for, never assumed
    assert A.rf_properties({"base": "pa12", "permittivity": 2.75, "lossTangent": 0.015})[:2] == (2.75, 0.015)
    try:
        A.rf_properties("ti64")
        assert False
    except ValueError as exc:
        assert "permittivity" in str(exc)
    # part-filled plastic: between the Hashin-Shtrikman bounds, which are between air and solid
    for fill in (0.2, 0.5, 0.8):
        e, t, lo, hi = A.mixed(2.69, 0.007, fill)
        assert 1.0 < lo <= e <= hi < 2.69 and hi - lo < 0.1 and 0 < t < 0.007, (fill, e, lo, hi)
    assert A.mixed(2.69, 0.007, 1.0)[0] == 2.69
    # `fill` is the solid share of the volume, not the slicer's infill setting: Castro et al. 2026
    # measured PETG printed at 25, 50 and 75% infill as 1.71, 1.94 and 2.28, each above the mix at
    # that share (a printed sample's walls and skins are solid) and below solid
    for setting, measured in ((0.25, 1.71), (0.5, 1.94), (0.75, 2.28)):
        assert A.mixed(2.69, 0.007, setting)[3] < measured < 2.69, (setting, A.mixed(2.69, 0.007, setting))
    e, _, lo, hi = A.mixed(2.69, 0.007, 0.5)
    print("materials ok: petg at half solid %.2f (bounds %.2f to %.2f)" % (e, lo, hi))


def test_mesh_to_cells():
    import trimesh
    x = np.linspace(-6, 6, 25)
    s = trimesh.creation.icosphere(subdivisions=4, radius=4.0)
    f = A.mesh_fraction(np.asarray(s.vertices), np.asarray(s.faces), x, x, x)
    vol = float(f.sum() * 0.5 ** 3)
    assert abs(vol - s.volume) / s.volume < 0.02, (vol, s.volume)
    assert f.max() == 1.0 and 0.0 < f[f > 0].min() < 1.0            # part-filled cells at the surface
    # a 1.5 mm wall whose faces fall between grid planes keeps its volume
    v, fc = A._box_mesh([-5.2, -5.2, 0.3], [5.2, 5.2, 1.8])
    f = A.mesh_fraction(v, fc, x, x, x)
    assert abs(f.sum() * 0.125 - 10.4 * 10.4 * 1.5) < 0.03 * 162, f.sum() * 0.125
    print("mesh to cells ok: sphere volume within %.1f%%" % (100 * abs(vol - s.volume) / s.volume))


def test_brief_schema():
    b = ifa_brief()
    b["surroundings"] = SHELL + [{"name": "battery", "metal": True, "box": [[2, 5, -5.5], [30, 40, -1.5]]},
                                 {"name": "lead", "wire": [[10, 10, -2], [10, 30, -2]]},
                                 {"name": "foam", "material": {"base": "tpu", "fill": 0.4}, "box": [[0, 0, 2], [5, 5, 4]]}]
    assert not [i for i in lint(b, "antenna") if i["level"] == "error"], lint(b, "antenna")
    bad = copy.deepcopy(b)
    bad["feed"]["form"] = bad["feed"].pop("from")
    assert any("did you mean 'from'" in i["message"] for i in lint(bad, "antenna"))
    enc = SCHEMAS["enclosure"]["properties"]
    assert "antenna" in enc["components"]["items"]["properties"] and "rf" in enc["components"]["items"]["properties"]
    assert "antenna" in enc["placement"]["properties"]
    try:
        import jsonschema
    except ImportError:
        print("brief schema ok (lint only: no jsonschema)")
        return
    jsonschema.validate(json.loads(json.dumps(b)), SCHEMAS["antenna"])
    print("brief schema ok")


def test_antenna_in_a_shell():
    """The tuned inverted-F bare, then inside a 2 mm PA12 shell with a
    battery under its ground: the plastic pulls the resonance down, some
    of the power becomes heat in the wall, and energy is accounted for."""
    b = ifa_brief()
    b["surroundings"] = SHELL + [{"name": "battery", "metal": True, "box": [[2, 5, -5.5], [30, 40, -1.5]]}]
    out = tempfile.mkdtemp()
    r = A.run_brief(b, out)
    bare, inst, ch = r["bare"], r["installed"], r["change"]
    # bare: matched in the 2.4 GHz band, as tune left it
    assert 2400 < bare["resonance"]["frequency"] < 2483.5 and bare["inBand"]["worstS11"] < -15, bare["resonance"]
    eff = bare["efficiency"]
    assert all(0.90 < e < 0.97 for e in eff["radiation"]), eff          # the FR-4 takes about 6%
    assert all(abs(u) < 0.02 for u in eff["unaccounted"]), eff          # radiated + heat = accepted
    assert all(abs(1 - h - e) < 0.02 for h, e in zip(eff["heat"]["board"], eff["radiation"]))
    # installed: lower, worse matched at the band's top edge, and the wall takes its share
    assert -110 < ch["resonanceShift"] < -30, ch
    assert inst["inBand"]["worstS11"] > bare["inBand"]["worstS11"] + 3
    assert all(0.003 < h < 0.05 for h in inst["efficiency"]["heat"]["shell"]), inst["efficiency"]["heat"]
    assert all(abs(u) < 0.02 for u in inst["efficiency"]["unaccounted"])
    assert inst["materials"]["shell"]["permittivity"] == 2.6 and inst["materials"]["battery"] == {"metal": True}
    assert r["ok"] == (inst["inBand"]["worstS11"] <= -10.0)
    for f in ("antenna.json", "s11.csv", "sensitivity.npz"):
        assert os.path.exists(os.path.join(out, f)), f
    w = r["sensitivity"]["metalWithin"]
    # half of the port's sensitivity to metal lies within a few mm of the antenna's own copper; the
    # rest runs down the ground, which is the antenna's other half
    assert 0 < w["50%"] < 8 < w["90%"] < w["99%"], w
    # the battery pushed under the antenna itself: the port is shorted out
    b["surroundings"][-1]["box"][1][1] = 57
    bad, _, _ = A.simulate_brief(b)
    assert bad["inBand"]["worstS11"] > -3, bad["inBand"]
    print("antenna in a shell ok: %.0f MHz bare, %+.0f MHz in the shell (worst S11 %.1f -> %.1f dB), "
          "battery under the antenna %.1f dB" % (bare["resonance"]["frequency"], ch["resonanceShift"],
                                                 bare["inBand"]["worstS11"], inst["inBand"]["worstS11"],
                                                 bad["inBand"]["worstS11"]))


def test_sensitivity_against_solves():
    """The first-order change in port impedance, read from the bare
    board's fields, against solving the board again with the body there
    (on one grid, so the difference is the body)."""
    b = ifa_brief()
    bare, bout, sc = A.simulate_brief(b, installed=False, fields=True)
    sens = A.sensitivity(sc, bout)
    fc = float(bout["monitor"][1])

    def both(body, eps):
        (x0, y0, z0), (x1, y1, z1) = body["box"]
        g = [np.arange(lo + 0.125, hi, 0.25) for lo, hi in ((x0, x1), (y0, y1), (z0, z1))]
        P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        pred = A.predicted_change(sens, P, 0.25 ** 3, eps)[0]
        bb = {**b, "surroundings": [body]}
        ref = _z(A.simulate_brief(bb, installed="air")[1], fc)
        return _z(A.simulate_brief(bb)[1], fc) - ref, pred

    # a 1.5 mm PA12 wall 0.6 mm off the board's long edge, and a 3 mm metal cube 3 mm above the arm's end
    for tag, body, eps, tol in (
            ("PA12 wall", {"name": "wall", "material": "pa12", "box": [[32.6, -5, -3.9], [34.1, 62, 10.1]]}, 2.6, 0.25),
            ("metal cube", {"name": "cube", "metal": True, "box": [[21, 54, 4], [24, 57, 7]]}, None, 0.45)):
        solved, pred = both(body, eps)
        assert abs(solved) > 1.0, (tag, solved)                       # a change worth predicting
        assert abs(pred - solved) < tol * abs(solved), (tag, solved, pred)
        print("sensitivity ok: %s solved %.2f%+.2fj ohm, first order %.2f%+.2fj" % (
            tag, solved.real, solved.imag, pred.real, pred.imag))


def test_tune_solves_the_dimensions():
    """Arm length and tap position from a detuned start: no tuning rule,
    a root found on the solved reflection."""
    r = A.tune(lambda arm, tap: ifa_brief(arm, tap), {"arm": 25.0, "tap": 5.5},
               bounds={"arm": (10, 30), "tap": (2, 15)}, depth=-20.0, rounds=8)
    assert r["converged"] and r["solves"] <= 11, (r["solves"], r["history"])
    # it stops at the first dimensions under -20 dB at the band's centre, which are near (not at)
    # the ones a deeper solve ends on
    assert abs(r["arm"] - ARM) < 1.5 and abs(r["tap"] - TAP) < 1.5, (r["arm"], r["tap"])
    s = r["summary"]
    assert abs(s["resonance"]["frequency"] - 2441.75) < 40 and s["inBand"]["worstS11"] < -12, s["resonance"]
    assert r["history"][0]["s11AtTarget"] > r["s11AtTarget"] + 6, r["history"]     # the start was worse
    print("tune ok: arm %.2f, tap %.2f mm in %d solves (%.1f dB at the band centre)" % (
        r["arm"], r["tap"], r["solves"], r["s11AtTarget"]))


def test_placement_judge():
    """What placement asks of an antenna: the cost of each candidate
    position from the bare board's fields. The quick estimate (every
    position at once, by correlation) agrees with the point-by-point one,
    and a battery under the antenna costs far more than one under the
    ground."""
    from loadpath.shapes import Box
    from loadpath.spec import _skin
    ext = Box([-20, -50, -7], [20, 50, 7])
    _, skin = _skin({"thickness": 1.5}, [ext])
    comps = [{"id": "board", "antenna": ifa_brief(), "size": [32, 56.6, 1.0]},
             {"id": "battery", "size": [28, 34, 4]},
             {"id": "pad", "size": [10, 10, 2], "rf": "tpu"},
             {"id": "label", "size": [10, 10, 1], "rf": "ignore"}]
    a = A.EnclosureAntenna(comps[0], comps, skin, 1.5, "pa12")
    assert set(a.others) == {"battery", "pad"} and a.others["battery"]["eps"] is None
    assert a.others["battery"]["assumed"] and not a.others["pad"]["assumed"] and a.others["pad"]["eps"] == 3.0
    eye = np.eye(3)
    Tb = np.array([0.0, 15.0, 0.0])                # the board: its antenna end at y = +43
    under_ground, under_antenna = np.array([0.0, 0.0, -3.5]), np.array([0.0, 28.0, -3.5])
    cost = {}
    for tag, Tc in (("ground", under_ground), ("antenna", under_antenna)):
        tf = {"board": (eye, Tb), "battery": (eye, Tc)}
        p = a.predicted(tf)
        q_board = a.quick(["board"], eye, Tb[None], {"battery": tf["battery"]})[0] / a.weight
        q_batt = a.quick(["battery"], eye, Tc[None], {"board": tf["board"]})[0] / a.weight
        assert abs(q_batt - p["battery"]) < 0.35 * p["battery"] + 0.01, (tag, q_batt, p)
        assert abs(q_board - (p["battery"] + p["wall"])) < 0.35 * (p["battery"] + p["wall"]) + 0.01, (tag, q_board, p)
        cost[tag] = a.judge({"transforms": tf, "new": ["battery"]})
        assert abs(cost[tag] - a.weight * p["battery"]) < 1e-9
    assert cost["antenna"] > 5 * cost["ground"] and cost["antenna"] > 50, cost
    # many positions at once: the same numbers as one at a time
    T = np.array([under_ground, under_antenna, [0.0, 10.0, 3.0]])
    many = a.quick(["battery"], eye, T, {"board": (eye, Tb)})
    one = [a.quick(["battery"], eye, t[None], {"board": (eye, Tb)})[0] for t in T]
    assert np.allclose(many, one)
    # the wall: closer to the roof costs more
    lo = a.quick(["board"], eye, np.array([[0.0, 15.0, 0.0]]), {})[0]
    hi = a.quick(["board"], eye, np.array([[0.0, 15.0, 3.5]]), {})[0]
    assert hi > 1.5 * lo > 0, (lo, hi)
    # the chosen layout solved in full, as enclosure() does it: bare and installed on one grid, the
    # installed model written out as a brief of its own
    import trimesh
    from loadpath.placement import component_mesh
    from loadpath.shapes import trimesh_from_manifold

    def at(c, T):
        m = component_mesh(c).copy()
        m.apply_translation(T)
        M = np.eye(4)
        M[:3, 3] = T
        return {"matrix": M.tolist()}, m
    places = {"board": Tb, "battery": under_ground, "pad": np.array([0.0, -30.0, 0.0])}
    lay = {"components": {}, "meshes": {}}
    for c in comps[:3]:
        lay["components"][c["id"]], lay["meshes"][c["id"]] = at(c, places[c["id"]])
    out = tempfile.mkdtemp()
    r = a.check(lay, trimesh_from_manifold(skin.to_manifold()), out)
    assert 2400 < r["bare"]["resonance"]["frequency"] < 2483.5, r["bare"]["resonance"]
    assert -80 < r["change"]["resonanceShift"] < -5, r["change"]          # the box's walls and the battery
    assert r["takenAsMetal"] == ["battery"] and set(r["firstOrder"]) == {"battery", "pad", "wall"}
    assert r["installed"]["materials"]["pad"]["permittivity"] == 3.0
    assert r["ok"] == (r["installed"]["inBand"]["worstS11"] <= -10.0)
    again = json.loads(open(os.path.join(out, r["brief"])).read())
    assert not [i for i in lint(again, "antenna") if i["level"] == "error"], lint(again, "antenna")
    assert {b["name"] for b in again["surroundings"]} == {"shell", "battery", "pad"}
    assert all(os.path.exists(os.path.join(out, "antenna-board", b["mesh"])) for b in again["surroundings"])
    assert trimesh.load(os.path.join(out, "antenna-board", "shell.stl")).is_watertight
    assert os.path.exists(os.path.join(out, "antenna-board.csv"))
    print("installed check ok: %+.0f MHz in the box, worst S11 %.1f dB" % (
        r["change"]["resonanceShift"], r["installed"]["inBand"]["worstS11"]))
    # two judges as one: costs add, either may refuse, quick estimates add
    j = A.both_judges(lambda info: 1.0, lambda info: 2.0)
    assert j({}) == 3.0 and A.both_judges(None, j) is j
    assert A.both_judges(lambda info: None, lambda info: 2.0)({}) is None
    print("placement judge ok: battery under the ground %.1f, under the antenna %.1f (mm of layout cost)" % (
        cost["ground"], cost["antenna"]))


def test_frame_is_the_antenna():
    """No board antenna at all: a 40 x 100 x 8 mm product's metal frame, one
    plastic-filled split in its end, fed from the chassis plate inside
    (A.frame_brief; the split and feed positions are what `tune`
    solved, bare and inside 1.5 mm of TPU). Bare it covers the band; the
    TPU pulls it down by hundreds of MHz and takes a real share of the
    power; with the two positions solved inside the TPU it covers the band
    again, at the efficiency the TPU's loss leaves."""
    bare_dims, tpu_dims = {"split": 24.288, "tap": 2.212}, {"split": 30.432, "tap": 0.5}
    b = A.frame_brief(**bare_dims)
    assert not [m for m in lint(b, "antenna") if m["level"] == "error"]
    assert [m for m in lint({"band": b["band"], "feed": b["feed"]}, "antenna") if m["level"] == "error"]
    bare, _, _ = A.simulate_brief(b, installed=False)
    assert bare["resonance"]["insideBand"] and bare["inBand"]["worstS11"] < -12, bare["inBand"]
    assert min(bare["efficiency"]["total"]) > 0.9, bare["efficiency"]
    assert max(abs(v) for v in bare["efficiency"]["unaccounted"]) < 0.01
    worn, _, _ = A.simulate_brief(A.frame_brief(cover="tpu", **bare_dims))
    shift = worn["resonance"]["frequency"] - bare["resonance"]["frequency"]
    assert -520 < shift < -360, shift                        # -449 MHz
    assert worn["inBand"]["worstS11"] > -6, worn["inBand"]
    assert 0.07 < worn["efficiency"]["heat"]["cover"][1] < 0.12, worn["efficiency"]["heat"]
    tuned, _, _ = A.simulate_brief(A.frame_brief(cover="tpu", **tpu_dims))
    assert tuned["inBand"]["worstS11"] < -10 and tuned["bandwidth"]["-10dB"]["coversBand"], tuned["inBand"]
    assert 0.70 < min(tuned["efficiency"]["total"]) < 0.85, tuned["efficiency"]
    print("frame as the antenna ok: bare %.0f MHz, %.1f dB, %.0f%%; in TPU %+.0f MHz, %.1f dB; solved in TPU "
          "%.1f dB, %.0f%% (TPU takes %.0f%%)"
          % (bare["resonance"]["frequency"], bare["inBand"]["worstS11"], 100 * bare["efficiency"]["total"][1],
             shift, worn["inBand"]["worstS11"], tuned["inBand"]["worstS11"], 100 * tuned["efficiency"]["total"][1],
             100 * (tuned["efficiency"]["heat"]["cover"][1] + tuned["efficiency"]["heat"]["split"][1])))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and (len(sys.argv) < 2 or sys.argv[1] in name):
            fn()
    print("all antenna tests passed")
