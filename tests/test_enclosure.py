"""
A product from its outside in (loadpath/enclosure.py, loadpath/placement.py):
the components placed inside a fixed exterior by their rules, each with a
checked way in, mounts clear of every path, and a study that keeps the
exterior as its skin. A small synthetic box; a minute or two.

Run: `python3 tests/test_enclosure.py`
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

from loadpath.enclosure import enclosure, exterior_check, paths_into_skin  # noqa: E402
from loadpath.placement import insertion_path, place, rotations  # noqa: E402
from loadpath.schema import lint  # noqa: E402

_DIR = tempfile.mkdtemp()


def _stl(name, man):
    import trimesh
    m = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))
    path = os.path.join(_DIR, name)
    tm.export(path)
    return path


def product():
    """A 50 x 34 x 20 mm box, a port slot through its -y end and a door
    through its floor (each just through the 1 mm skin), a board with a
    connector, a battery and a door that closes the floor."""
    import manifold3d as m3d
    box = m3d.Manifold.cube((50, 34, 20)).translate((-25, -17, -10))
    port = m3d.Manifold.cube((10, 3.0, 4)).translate((-5, -19, -1))
    door = m3d.Manifold.cube((40, 26, 2.0)).translate((-20, -13, -11))
    board = m3d.Manifold.cube((30, 20, 1.6)).translate((-15, -10, -0.8))
    conn = m3d.Manifold.cube((8, 6, 3)).translate((-4, -10 - 2.0, 0.8))
    return {
        "name": "box-product",
        "exterior": _stl("exterior.stl", box - port - door), "skin": 1.0,
        "openings": [{"id": "port", "shape": _stl("port.stl", port), "inward": "+y"},
                     {"id": "door", "shape": _stl("door.stl", door), "inward": "+z"}],
        "components": [
            {"id": "board", "shape": _stl("board.stl", board), "mass": 0.006, "rotate": "z", "watts": 0.2},
            {"id": "conn", "shape": _stl("conn.stl", conn), "installedOn": "board",
             "alignTo": {"opening": "port", "face": "-y", "recess": 0.3}},
            {"id": "battery", "size": [30, 18, 5], "mass": 0.02, "against": "-z", "rotate": "z"},
            {"id": "lid", "shape": _stl("lid.stl", m3d.Manifold.cube((39.6, 25.6, 1.0)).translate(
                (-19.8, -12.8, -10))), "place": "given", "external": True, "mount": False}],
        "assembly": ["board", "battery", "lid"],
        "rest": "+z",
        "loads": {"drop": {"g": 100, "faces": ["-z", "+y"]}, "squeeze": {"axis": "x", "force": 50}},
        "material": "pa12", "minMember": 1.0, "thermal": {"convection": 8},
    }


def test_rotations():
    assert len(rotations("any")) == 24 and len(rotations("z")) == 4 and len(rotations("none")) == 1
    for R in rotations("any"):
        assert abs(np.linalg.det(R) - 1) < 1e-9 and np.allclose(R @ R.T, np.eye(3))


def test_insertion_path_through_an_opening():
    """A block under a slot in a lid comes out straight up through it; with
    the slot shut it has no way out."""
    from loadpath.shapes import Box, Union
    part = Box([-2, -2, 0], [2, 2, 3])
    lid = Union([Box([-10, -10, 5], [-2.6, 10, 6]), Box([2.6, -10, 5], [10, 10, 6])])
    r = insertion_path(part.sdf, part.bounds(), lid.sdf, ([-10, -10, -1], [10, 10, 6]))
    assert r is not None and len(r["legs"]) <= 2, r
    shut = Box([-10, -10, 4.5], [10, 10, 6])
    walls = Union([shut, Box([-10, -10, -2], [10, 10, -0.5]), Box([-10, -10, -2], [-8.5, 10, 6]),
                   Box([8.5, -10, -2], [10, 10, 6]), Box([-10, -10, -2], [10, -8.5, 6]),
                   Box([-10, 8.5, -2], [10, 10, 6])])
    assert insertion_path(part.sdf, part.bounds(), walls.sdf, ([-10, -10, -2], [10, 10, 6])) is None


def test_turn_legs():
    """A path's legs are slides and turns: the pose after them, the one
    move between two poses (a turn about the line that stays put), their
    JSON, and the points walked along them against a wall."""
    from loadpath.motion import Turn, _between, end_pose, legs_from_spec, legs_to_spec, path_check, rotation
    legs = [Turn(np.array([0.0, 1.0, 0.0]), np.array([10.0, 0.0, 2.0]), 20.0), (np.array([1.0, 0.0, 0.0]), 5.0)]
    R, t = end_pose(legs)
    assert np.allclose(R, rotation([0, 1, 0], 20.0))
    # the pivot stays where it is through the turn, then slides with the part
    assert np.allclose(R @ np.array([10.0, 0.0, 2.0]) + t, [15.0, 0.0, 2.0])
    back = _between((np.eye(3), np.zeros(3)), end_pose(legs[:1]))
    assert isinstance(back, Turn) and abs(back.angle - 20.0) < 1e-9 and abs(back.along) < 1e-9
    assert np.allclose(back.at[[0, 2]], [10.0, 2.0]) and np.allclose(back.axis, [0, 1, 0])
    spec = legs_to_spec(legs + [(np.array([0.0, 0.0, -1.0]), None)])
    assert spec == [{"turn": 20.0, "about": "+y", "at": [10.0, 0.0, 2.0]}, {"toward": "+x", "length": 5.0}, "-z"]
    again = legs_from_spec(spec)
    assert isinstance(again[0], Turn) and again[1][1] == 5.0 and again[2][1] is None
    assert legs_to_spec([(np.array([0.0, 0.0, -1.0]), None)]) == "-z"
    # a bar's end swings up into a ceiling 3 mm over it: 20 degrees about its other end lifts it 6.8 mm
    pts = np.array([[x, 0.0, 0.0] for x in np.linspace(0, 20, 41)])

    def ceiling(q):
        return q[:, 2] - 3.0, np.zeros(len(q), int)
    up = [Turn(np.array([0.0, 1.0, 0.0]), np.zeros(3), -20.0)]
    ok, worst, _, where, _ = path_check(pts, ceiling, up, [-1, -1, -1], [21, 1, 4], 0.1)
    assert not ok and where[2] > 3.0, (ok, worst, where)
    ok, worst, *_ = path_check(pts, ceiling, [Turn(np.array([0.0, 1.0, 0.0]), np.zeros(3), -8.0)],
                               [-1, -1, -1], [21, 1, 4], 0.1)
    assert ok and worst == 0.0, (ok, worst)
    # a cable followed out along the same legs (wires.py) ends in the same pose
    from loadpath.wires import _path_leg, rigid
    Rw, tw = rigid({"type": "path", "legs": [_path_leg(leg) for leg in legs]}, 1.0)
    assert np.allclose(Rw, R) and np.allclose(tw, t)


def _tray():
    """A board 40 mm long on the floor of a tray whose top is roofed over
    its +x end: the opening is 30 mm long."""
    from loadpath.shapes import Box, Union
    part = Box([-20, -10, 0], [20, 10, 2])
    walls = Union([Box([-24, -13, -3], [24, 13, -1.0]), Box([-24, -13, -3], [-22, 13, 14]),
                   Box([22, -13, -3], [24, 13, 14]), Box([-24, -13, -3], [24, -11.5, 14]),
                   Box([-24, 11.5, -3], [24, 13, 14]), Box([8, -13, 6], [24, 13, 14])])
    return part, walls, ([-24, -13, -3], [24, 13, 14])


def test_a_board_longer_than_its_opening_tilts_in():
    """No straight slides take the board out from under the roof; turned,
    it comes out, and the legs found are walked exactly. With tilting off
    it has no way in."""
    from loadpath.enclosure import _legs_of, _way_in
    from loadpath.motion import Turn
    from loadpath.placement import insertion_motion
    part, walls, box = _tray()
    assert insertion_path(part.sdf, part.bounds(), walls.sdf, box) is None
    r = insertion_motion(part.sdf, part.bounds(), walls.sdf, box)
    assert r is not None and 5.0 < r["tilt"] <= 45.0 and np.allclose(np.abs(r["axis"]), [0, 1, 0]), r and r["tilt"]
    assert insertion_motion(part.sdf, part.bounds(), walls.sdf, box, max_tilt=4.0) is None
    rec = _way_in(part.sdf, part.bounds(), None, walls.sdf, box, 0.25, None, {"max": 45.0})
    assert rec["insertFrom"] is not None and rec["how"] == "tilt" and rec["tilt"] > 5.0, rec
    legs = _legs_of(rec["insertFrom"])
    assert any(isinstance(leg, Turn) for leg in legs) and len(legs) <= 6, rec
    assert rec["worstPenetration"] <= 0.25
    assert _way_in(part.sdf, part.bounds(), None, walls.sdf, box, 0.25, None, None)["insertFrom"] is None
    # sealed, it has no way in however it turns
    from loadpath.shapes import Box, Union
    shut = Union([walls, Box([-24, -13, 12], [24, 13, 14])])
    assert insertion_motion(part.sdf, part.bounds(), shut.sdf, box) is None


def test_a_turn_sweeps_the_keep_out():
    """A keep-out whose way in turns keeps clear the space it turns
    through: the study lints, and the keep-out reaches where the part's
    far end swings and no further."""
    from loadpath.spec import study_from_dict
    study = {
        "designSpace": {"type": "box", "min": [-5, -6, -2], "max": [45, 6, 30]},
        "supports": [{"shape": {"type": "box", "min": [-6, -7, -3], "max": [46, 7, -1.99]}}],
        "loadCases": [{"name": "press", "loads": [
            {"shape": {"type": "box", "min": [-6, -7, 29.99], "max": [46, 7, 31]}, "force": [0, 0, -1]}]}],
        "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
        "goal": {"volumeFraction": 0.3},
        "keepOut": [{"id": "board", "bore": False, "shape": {"type": "box", "min": [0, -4, 0], "max": [40, 4, 2]},
                     "insertFrom": [{"turn": -30, "about": "y", "at": [0, 0, 1]},
                                    {"toward": [0.866025, 0, 0.5], "length": 6.0}, "+z"]}],
    }
    assert not [x for x in lint(study, "study") if x["level"] == "error"], lint(study, "study")
    ko = study_from_dict(study).keep_out[0].shape
    # the far end at 15 degrees: (38.6, 0, 11.4); at 30, slid 3 mm: (37.2, 0, 22.5); then on up
    inside = np.array([[20.0, 0, 1.0], [38.6, 0, 11.4], [37.2, 0, 22.5], [36.0, 0, 28.0]])
    outside = np.array([[38.0, 0, -1.0], [44.0, 0, 8.0], [20.0, 0, -1.5], [10.0, 5.5, 3.0]])
    assert (ko.sdf(inside) < 0).all(), ko.sdf(inside)
    assert (ko.sdf(outside) > 0).all(), ko.sdf(outside)
    man = ko.to_manifold()
    assert man.volume() > 40 * 8 * 2 * 3, man.volume()
    bad = dict(study, keepOut=[dict(study["keepOut"][0], insertFrom=[{"turn": 30, "about": "y"}, "+z"])])
    assert [x for x in lint(bad, "study") if x["level"] == "error"]


def test_assembly_steps_read_in_order():
    """The steps a person follows are the legs the part leaves by, turned
    round: held tilted outside, brought in, slid, laid flat."""
    from loadpath.motion import legs_from_spec, steps
    ins = [{"turn": 10.0, "about": "+y", "at": [-30.0, 0.0, 0.0]}, {"toward": "+x", "length": 8.0}, "-z"]
    out = steps(["board", "lid"], {"board": {"insertFrom": ins, "travelsWith": ["conn"]}},
                {"board": ([-27.0, -11.0, 0.0], [27.0, 11.0, 1.6])}, external=["lid"])
    assert [s["n"] for s in out] == [1, 2, 3, 4, 5] and [s["component"] for s in out] == ["board"] * 4 + ["lid"]
    text = [s["text"] for s in out]
    assert text[0].startswith("Hold board (with conn on it) tilted 10° about +y") and "-z side" in text[0], text
    assert text[1] == "Bring it in toward +z.", text
    assert text[2] == "Slide it 8 mm toward -x.", text
    assert text[3].startswith("Turn it 10° about a line along y through x = -30, z = 0") \
        and text[3].endswith("until it sits square: it is home."), text
    assert text[4] == "Fit lid from outside."
    assert len(legs_from_spec(ins)) == 3
    one = steps(["cell"], {"cell": {"insertFrom": "-z"}}, {"cell": ([0, 0, 0], [1, 1, 1])})
    assert [s["text"] for s in one] == ["Hold cell as it will sit, outside the -z side.",
                                        "Bring it in toward +z: it is home."], one


def test_a_product_that_only_goes_together_with_a_tilt():
    """The desk hub (evals/products.py): its board is longer than the door
    under it and its connector reaches out through a port, so no straight
    slides put it in. It is placed with a tilted way in that cuts nothing
    from the skin, and enclosure.json says how in numbered steps; with
    tilting off the board cannot be placed."""
    from loadpath.evals.products import write
    from loadpath.motion import Turn, legs_from_spec
    d = tempfile.mkdtemp()
    brief = json.load(open(write("desk-hub", d)))
    out = tempfile.mkdtemp()
    r = enclosure(brief, d, out)
    st, rep = r["study"], r["report"]
    assert not rep["lint"] and not rep["warnings"], (rep["lint"], rep["warnings"])
    way = rep["assembly"]["components"]["board"]
    assert way["how"] == "tilt" and 5.0 <= way["tilt"] <= 45.0, way
    legs = legs_from_spec(way["insertFrom"])
    assert any(isinstance(leg, Turn) for leg in legs) and len(legs) <= 6, way
    keep = {k["id"]: k for k in st["keepOut"]}
    assert keep["board"]["insertFrom"] == way["insertFrom"] == keep["connector"]["insertFrom"]
    steps = rep["assembly"]["steps"]
    assert [s["n"] for s in steps] == list(range(1, len(steps) + 1)) and steps[-1]["component"] == "foot", steps
    assert "tilted" in steps[0]["text"] and steps[len(legs)]["text"].endswith("it is home."), steps
    notch = paths_into_skin(st, out)
    assert notch and all(v <= 0.05 for v in notch.values()), notch
    json.dumps(rep, default=float)
    flat = enclosure(dict(brief, tilt=0), d, tempfile.mkdtemp())["report"]
    assert "board" not in flat["placement"]["components"] or \
        flat["assembly"]["components"]["board"]["insertFrom"] is None, flat["assembly"]
    assert any("board" in w for w in flat["warnings"]), flat["warnings"]


def test_placement_rules():
    """The connector ends up out through its port, short of the exterior by
    the recess, and the battery on the floor; every part keeps its margin."""
    from loadpath.handoff import _fine_mesh_shape
    from loadpath.shapes import shape_from_spec
    b = product()
    ext = _fine_mesh_shape({"type": "mesh", "path": b["exterior"]}, None)
    ops = [{"id": o["id"], "shape": shape_from_spec({"type": "mesh", "path": o["shape"]}), "inward": o["inward"],
            "depth": 1.2} for o in b["openings"]]
    comps = [dict(c, shape={"type": "mesh", "path": c["shape"]}) if isinstance(c.get("shape"), str) else c
             for c in b["components"]]
    r = place(ext, comps, skin=1.0, openings=ops)
    assert not r["blocked"], r["blocked"]
    c = r["best"]["components"]
    # the connector's face out in the port, past the inner wall (y = -16),
    # short of the exterior (y = -17) by at least the recess (the board's
    # edge stops it before the rim round the mouth)
    face = c["conn"]["bounds"][0][1]
    assert -17.0 + 0.3 - 1e-3 <= face <= -16.0, c["conn"]
    # the recess is the face's own (0.3 asked, not 0.3 plus the 0.5 margin);
    # here the skin lining the port's inner edge stops the board first
    assert 0.3 - 1e-3 <= c["conn"]["recess"] < 0.75, c["conn"]
    # the battery on the floor: the skin is 1 mm, the margin 0.5 mm
    assert c["battery"]["bounds"][0][2] < -10 + 1.0 + 0.5 + 0.6, c["battery"]
    for k in ("board", "battery"):
        assert c[k]["roomToSkin"] >= 0.5 - 0.02, (k, c[k])
        assert all(v >= 0.5 - 0.02 for v in c[k]["roomToParts"].values()), (k, c[k])


def test_enclosure_study():
    """The study: the exterior as skin, every placed part a keep-out with a
    checked way in, a mount for each massed part clear of the other paths,
    one drop case per face asked and the squeeze, heat from the board."""
    b = product()
    assert not [x for x in lint(b, "enclosure") if x["level"] == "error"], lint(b, "enclosure")
    out = tempfile.mkdtemp()
    r = enclosure(b, _DIR, out)
    st, rep = r["study"], r["report"]
    assert not rep["lint"], rep["lint"]
    assert not rep["warnings"], rep["warnings"]
    keep = {k["id"]: k for k in st["keepOut"]}
    assert keep["board"]["insertFrom"] is not None and keep["battery"]["insertFrom"] is not None
    assert keep["conn"]["insertFrom"] == keep["board"]["insertFrom"]          # it travels with the board
    assert "insertFrom" not in keep["lid"]
    assert {"mount-board", "mount-battery"} <= {p["id"] for p in st["preserve"]}
    assert [c["name"] for c in st["loadCases"]] == ["drop-z", "drop+y", "squeeze-x"]
    drop = st["loadCases"][0]["loads"]
    # the floor holds back exactly what the parts push on it
    total = sum(9.81 * 100 * ld["mass"] for ld in drop if "mass" in ld)
    floor = [ld for ld in drop if "shape" in ld][0]
    assert abs(floor["force"][2] - total) < 1e-3 * total, drop
    # and the landing's spin balances the moment: the support carries nothing
    from loadpath.shells import _load_force
    F = sum((_load_force(ld) for ld in drop), np.zeros(3))
    assert np.linalg.norm(F) < 1e-3 * total, F
    spin = [ld for ld in drop if "force" in ld and "region" in ld]
    assert spin and all("at" in ld for ld in spin), drop
    assert st["thermal"]["heat"] == [{"region": "board", "watts": 0.2}]
    assert st["skin"]["thickness"] == 1.0 and len(st["skin"]["openings"]) == 2
    assert os.path.exists(os.path.join(out, "cad", "battery.stl"))
    # no way in notches the skin: the keep-outs are cut from everything
    notch = paths_into_skin(st, out)
    assert notch and all(v <= 0.05 for v in notch.values()), notch
    json.dumps(rep, default=float)


def test_exterior_check():
    """The exterior read against itself is untouched; a part cut back from
    it is not."""
    import trimesh

    from loadpath.shapes import MeshShape
    b = product()
    m = trimesh.load(b["exterior"], force="mesh")
    ex = MeshShape(m)
    ok = exterior_check(m, ex, samples=8000)
    assert ok["untouched"] and ok["maxProud"] < 0.01, ok
    cut = m.copy()
    cut.apply_scale(0.97)
    bad = exterior_check(cut, ex, samples=8000)
    assert not bad["untouched"] and bad["coverage"] < 0.5, bad


def test_cooling_block_takes_the_parts_heat():
    """A cooling block in the brief gets each part's watts and maxTemp on
    its own keep-out, unless the block names that part already."""
    b = product()
    b["components"][2]["maxTemp"] = 45.0
    b["cooling"] = {"ambient": 30.0, "skin": {"contact": "held"},
                    "components": [{"region": "conn", "watts": 0.05}]}
    r = enclosure(b, _DIR, tempfile.mkdtemp())
    co = r["study"]["cooling"]
    by = {c["region"]: c for c in co["components"]}
    assert by["board"] == {"region": "board", "watts": 0.2}, by
    assert by["battery"] == {"region": "battery", "maxTemp": 45.0}, by
    assert by["conn"] == {"region": "conn", "watts": 0.05} and "lid" not in by, by
    assert co["ambient"] == 30.0


def test_rattle_holds_every_mounted_part_still():
    """loads.rattle: an isolate (upTo) feel target per massed, mounted
    part, from the hand on the skin to its mount; the study lints and the
    targets resolve on its grid."""
    from loadpath.domain import build_domain
    from loadpath.feel import feel_from_study
    from loadpath.spec import study_from_dict
    b = product()
    b["loads"]["rattle"] = {"band": 120, "g": 4}
    assert not [x for x in lint(b, "enclosure") if x["level"] == "error"], lint(b, "enclosure")
    out = tempfile.mkdtemp()
    r = enclosure(b, _DIR, out)
    st = r["study"]
    ts = {t["name"]: t for t in st["feel"]["targets"]}
    assert set(ts) == {"rattle-board", "rattle-battery"}, ts
    t = ts["rattle-battery"]
    assert t["kind"] == "isolate" and t["upTo"] == 120 and t["shockG"] == 4 and t["kg"] == 0.02, t
    assert t["mount"] == {"region": "mount-battery"} and t["hold"] == {"region": "skin"}
    assert not [x for x in lint(st, "study") if x["level"] == "error"], lint(st, "study")
    from loadpath.enclosure import _localize
    st = json.loads(json.dumps(st))
    _localize(st, out)
    s = study_from_dict(st, base_dir=out)
    got = feel_from_study(s, build_domain(s, 2.0))
    assert [g.kmin > 0 and g.kmax is None for g in got] == [True, True], got


def test_exterior_check_on_a_half():
    """A split shell's half is checked against the exterior inside its own
    region only: the other half's skin is not missing."""
    import trimesh

    from loadpath.shapes import Box, MeshShape
    b = product()
    m = trimesh.load(b["exterior"], force="mesh")
    half = trimesh.intersections.slice_mesh_plane(m, [0, 0, -1], [0, 0, 0], cap=True)
    ex = MeshShape(m)
    whole = exterior_check(half, ex, samples=8000)
    assert not whole["untouched"], whole
    mine = exterior_check(half, ex, samples=8000, region=Box([-40, -40, -20], [40, 40, 0]))
    assert mine["untouched"] and mine["maxProud"] < 0.01, mine


def test_eval_products_lint():
    """The enclosure eval products (loadpath/evals/products.py) write briefs
    that lint clean."""
    from loadpath.evals.products import PRODUCTS, write
    for name in PRODUCTS:
        path = write(name, os.path.join(tempfile.mkdtemp(), name))
        brief = json.load(open(path))
        errs = [x for x in lint(brief, "enclosure") if x["level"] == "error"]
        assert not errs, (name, errs)


def _rail_study(bore):
    from loadpath.spec import study_from_dict
    return study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [20, 10, 10]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [21, 11, 0.01]}}],
        "loadCases": [{"name": "press", "loads": [
            {"shape": {"type": "box", "min": [-1, -1, 9.99], "max": [21, 11, 11]}, "force": [0, 0, -1]}]}],
        "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
        "goal": {"safetyFactor": 2, "volumeFraction": 0.3},
        # a rail the designer drew, and a board's clear zone drawn over it
        "preserve": [{"id": "rail", "shape": {"type": "box", "min": [2, 2, 2], "max": [18, 4, 5]}}],
        "keepOut": [{"id": "board", "bore": bore, "shape": {"type": "box", "min": [1, 1, 3], "max": [19, 9, 8]}}],
    })


def test_a_clear_zone_does_not_cut_what_the_designer_drew():
    """A clearance keep-out ("bore": false) keeps the optimizer out; it
    does not delete the preserved rail under it, the way the touch
    microphone's board clear zone took 88% of its rails. A bore still cuts."""
    from loadpath.domain import build_domain
    clear, bore = _rail_study(False), _rail_study(True)
    assert not clear.keep_out[0].cuts and bore.keep_out[0].cuts
    kept = int(build_domain(clear, 1.0).preserve.sum())
    cut = int(build_domain(bore, 1.0).preserve.sum())
    # the bore takes the rail's top two thirds (16 x 2 x 2 cells); the
    # clear zone takes none of it (the rest are the load and support faces)
    assert kept - cut >= 60, (cut, kept)


def test_drawn_check_flags_a_missing_mount():
    """drawncheck reads each preserved region on the delivered mesh: the
    rail whole passes, the rail with its top cut away fails."""
    import trimesh

    from loadpath.drawncheck import drawn_check
    from loadpath.shapes import Box
    from loadpath.shapes import trimesh_from_manifold
    s = _rail_study(False)
    base = Box([0, 0, 0], [20, 10, 1.5]).to_manifold()
    whole = trimesh_from_manifold(base + Box([2, 2, 1], [18, 4, 5]).to_manifold())
    ok = drawn_check(s, whole)
    assert ok["ok"] and ok["regions"][0]["missingShare"] < 0.02, ok
    short = trimesh_from_manifold(base + Box([2, 2, 1], [18, 4, 3]).to_manifold())
    bad = drawn_check(s, short)
    assert not bad["ok"] and bad["regions"][0]["missingShare"] > 0.5, bad
    assert isinstance(short, trimesh.Trimesh)


def test_plug_loads_go_through_the_boards_mount():
    """loads.connector: a USB-C plug pushed in, pulled out and levered across
    both sides at 15 mm, into the board's mount with the moment of its lever,
    the hand on the face across from the port, and the board's travel
    capped in those cases."""
    from loadpath.connectors import connector_cases
    b = product()
    b["loads"] = {"connector": {"component": "conn"}}
    assert not [x for x in lint(b, "enclosure") if x["level"] == "error"], lint(b, "enclosure")
    out = tempfile.mkdtemp()
    st = enclosure(b, _DIR, out)["study"]
    cs = {c["name"]: c for c in st["loadCases"]}
    assert list(cs) == ["conn-insert", "conn-withdraw", "conn-side-thin", "conn-side-wide"], list(cs)
    ins = cs["conn-insert"]["loads"]
    assert ins[0]["region"] == "mount-board" and ins[0]["force"] == [0.0, 20.0, 0.0], ins
    assert ins[1]["force"] == [0.0, -20.0, 0.0] and ins[1]["shape"]["max"][1] > 16, ins   # the far (+y) face
    assert cs["conn-withdraw"]["loads"][0]["force"] == [0.0, -20.0, 0.0]
    # the plug's tip: 15 mm out of the socket's front edge (-y); wide is x (8 mm), thin z (3 mm)
    th, wd = cs["conn-side-thin"]["loads"][0], cs["conn-side-wide"]["loads"][0]
    mouth_y = ins[0]["at"][1]
    assert abs(th["at"][1] - (mouth_y - 15.0)) < 1e-6, (th, mouth_y)
    assert abs(abs(th["force"][2]) - 8.0) < 1e-6 and abs(abs(wd["force"][0]) - 20.0) < 1e-6, (th, wd)
    cl = st["goal"]["clearances"]
    assert cl == [{"name": "conn-shift", "region": "mount-board", "maxTravel": 0.1, "cases": list(cs)}], cl
    assert not [x for x in lint(st, "study") if x["level"] == "error"], lint(st, "study")
    # abuse: the wrenching moments as a force at the lever
    cc, _ = connector_cases("usb-c", region="r", mouth=[0, 0, 0], out=[0, -1, 0], wide=[1, 0, 0], abuse=True)
    w = {c["name"]: c["loads"][0]["force"] for c in cc}
    assert abs(np.linalg.norm(w["usb-c-wrench-wide"]) - 3500 / 15) < 1e-2, w
    assert [c.get("safetyFactor") for c in cc][-2:] == [1.0, 1.0], cc


def test_keep_out_insert_from_a_vector_is_one_straight_way_out():
    # shells.apply_split writes a tilted parting line's way out as a vector;
    # it once read as a path of three legs with no lengths and failed lint
    from loadpath.enclosure import _legs_of
    from loadpath.spec import _assembly_path
    from loadpath.shapes import Box
    import numpy as np

    lo, hi = np.array([-10.0, -10, -10]), np.array([10.0, 10, 10])
    sw = _assembly_path(Box((-1, -1, -1), (1, 1, 1)), [-0.258819, 0.0, 0.965926], (lo, hi), "k")
    a, b = sw.bounds()
    assert b[2] > 9.0 and a[0] < -2.0
    legs = _legs_of([-0.258819, 0.0, 0.965926])
    assert len(legs) == 1 and legs[0][1] is None


def _stadium_port_exterior(flare=False, pocket=True):
    """A 40 x 30 x 12 mm block with a 9.4 x 3.8 mm stadium port 4 mm into its
    +y face (a USB-C mouth), optionally with a 0.5 mm deep 10.4 x 4.8 mm
    lead-in at its mouth."""
    import manifold3d as m3d
    import trimesh

    from loadpath.shapes import MeshShape

    def stadium(w, h, depth):
        r = h / 2
        cs = (m3d.CrossSection.square([w - 2 * r, h]).translate([-(w - 2 * r) / 2, -r])
              + m3d.CrossSection.circle(r, 96).translate([-(w - 2 * r) / 2, 0])
              + m3d.CrossSection.circle(r, 96).translate([(w - 2 * r) / 2, 0]))
        # extruded on +z, turned so it runs in from the +y face along -y
        return m3d.Manifold.extrude(cs, depth).rotate([90, 0, 0]).translate([0, 15.001, 0])
    ext = m3d.Manifold.cube([40, 30, 12], center=True)
    if pocket:
        ext = ext - stadium(9.4, 3.8, 4.0)
        if flare:
            ext = ext - stadium(10.4, 4.8, 0.5)
    m = ext.to_mesh()
    return MeshShape(trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)))


def test_a_box_opening_keeps_the_drawn_ports_rounded_wall():
    """An opening drawn as a box round a drawn USB-C stadium opens the
    stadium only: the skin keeps the port's rounded wall (it used to be cut
    back to the box, a 12 x 6 mm square hole), the membrane across the
    port's inner end goes, and a lead-in flare at the mouth does not widen
    the cut through the walls below it."""
    from loadpath.shapes import trimesh_from_manifold
    from loadpath.sharp import exact_manifold
    from loadpath.spec import _skin
    box = {"type": "box", "min": [-6, 8, -3], "max": [6, 16, 3]}
    stadium_area = 9.4 * 3.8 - (4 - np.pi) * 1.9 ** 2
    for flare in (False, True):
        ds = _stadium_port_exterior(flare=flare)
        _, skin = _skin({"thickness": 1.0, "openings": [{"shape": box, "inward": "-y", "depth": 1.2}]}, [ds])
        m = trimesh_from_manifold(exact_manifold(skin))
        # the drawn wall just outside the stadium, at the mouth and deeper
        wall = np.array([[0, 13.0, 2.3], [5.1, 13.0, 0.0], [4.4, 13.0, 1.65], [0, 14.2, 2.3]])
        assert m.contains(wall).all(), (flare, m.contains(wall))
        # the membrane across the port's floor (y 11 to 10) is cut, in the stadium
        assert not m.contains(np.array([[0, 10.5, 0], [4.0, 10.5, 0.0]])).any()
        # and only in the stadium: beside it, at the floor, the skin stays
        assert m.contains(np.array([[0, 10.5, 2.4], [5.2, 10.5, 0.0]])).all()
        sec = m.section(plane_origin=[0, 12.0, 0], plane_normal=[0, 1, 0]).to_2D()[0]
        from shapely.geometry import Polygon
        # (the section also has the skin's inner cavity, 380 mm^2)
        holes = [h for p in sec.polygons_full for h in p.interiors if Polygon(h).area < 100]
        assert len(holes) == 1 and abs(Polygon(holes[0]).area - stadium_area) < 0.1, \
            [Polygon(h).area for h in holes]
    # nothing drawn there: the opening's own shape is the cut, as before
    ds = _stadium_port_exterior(pocket=False)
    _, skin = _skin({"thickness": 1.0, "openings": [{"shape": box, "inward": "-y", "depth": 1.2}]}, [ds])
    m = trimesh_from_manifold(exact_manifold(skin))
    assert not m.contains(np.array([[5.5, 14.5, 2.5]])).any()


def test_a_connector_is_centred_on_its_port_by_its_mating_face():
    """A socket aligned to a port is centred by its front (the stadium
    shell), not by its bounding box: a solder tab out to one side behind
    the front used to pull the shell 1.3 mm off the port's centre."""
    from loadpath.placement import _face_point
    shell = np.array([[x, y, z] for x in (-4.47, 4.47) for y in (0.0, 7.35) for z in (0.0, 3.26)])
    tab = np.array([[x, y, z] for x in (4.9, 7.1) for y in (5.0, 7.0) for z in (0.0, 1.0)])
    fp = _face_point(np.vstack([shell, tab]), np.array([0.0, -1.0, 0.0]))
    assert np.allclose(fp, [0.0, 0.0, 1.63]), fp


def test_a_cradle_follows_a_round_cell():
    """A cylinder cell lying on its side is cradled by a saddle that meets
    it along its curve, a layer of the cradle's thickness under its own
    underside and inside its own footprint; the bounding box's slab only
    touched it along its lowest line."""
    import trimesh

    from loadpath.enclosure import _conforming_cradle
    from loadpath.sharp import exact_manifold
    from loadpath.shapes import shape_from_spec, trimesh_from_manifold
    cell = trimesh.creation.cylinder(radius=9.0, height=65.0, sections=96)
    cell.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    ext = {"type": "box", "min": [-50, -30, -30], "max": [50, 30, 30]}
    spec = _conforming_cradle([cell], cell.bounds, np.array([0.0, 0.0, -1.0]), 1.2, ext,
                              os.path.join(_DIR, "cradle-cell.stl"))
    assert spec is not None
    m = trimesh_from_manifold(exact_manifold(shape_from_spec(spec)))
    zc = lambda y: -np.sqrt(81.0 - y * y)              # the cell's underside at y
    # under the cell's lowest line and up its curve, 0.6 mm under its skin
    assert m.contains([[0, 0, zc(0) - 0.6], [10, 6.0, zc(6.0) - 0.6], [-20, -7.0, zc(7.0) - 0.6]]).all()
    # not the flat slab: under the curve at the slab's depth, nothing
    assert not m.contains([[0, 6.0, -9.0 - 0.6], [0, -7.0, -9.0 - 0.6]]).any()
    # nothing past the footprint, nor in the cell
    assert not m.contains([[0, 9.6, -8.0], [0, 0, -8.5], [40.0, 0, -9.6]]).any()


def _body_study(cradle_top):
    """A board (body) resting on a drawn cradle whose top is at `cradle_top`
    (the board's underside is z = 0)."""
    from loadpath.spec import study_from_dict
    return study_from_dict({
        "name": "bodies", "material": "pa12",
        "designSpace": {"type": "box", "min": [-20, -20, -10], "max": [20, 20, 10]},
        "preserve": [{"id": "cradle", "shape": {"type": "box", "min": [-12, -8, -3], "max": [12, 8, cradle_top]}}],
        "keepOut": [{"id": "board", "bore": False, "body": True, "insertFrom": "+z",
                     "shape": {"type": "box", "min": [-10, -6, 0], "max": [10, 6, 1.6]}}],
        "supports": [{"shape": {"type": "box", "min": [-20, -20, -10], "max": [20, 20, -9]}}],
        "loads": [{"shape": {"type": "box", "min": [-20, -20, 9], "max": [20, 20, 10]}, "force": [0, 0, -1]}]})


def test_a_part_s_body_cuts_what_is_drawn_and_is_checked_by_depth():
    """Two solids cannot share space. A cradle drawn 0.4 mm into its board
    (as the enclosure's ray-sampled cradles and shell lips were, up to 1.5
    mm into a motor) is cut to the board's own surface when the study is
    read, and said so; any of the delivered part inside a body is measured
    by how deep, not by volume: a film 0.6 mm into a board edge is under
    1 mm^3 and was passed."""
    import manifold3d as m3d
    from loadpath.sharp import exact_manifold
    from loadpath.verify import body_intrusion
    st = _body_study(0.4)
    cradle, board = st.preserve[0], st.keep_out[0]
    assert board.body is not None and cradle.uncut is not None
    # the cradle as read no longer reaches into the board: it seats on it
    assert (exact_manifold(cradle.shape) ^ exact_manifold(board.body)).volume() < 1e-6
    # the cut is reported, with its depth
    rep = body_intrusion(st, exact_manifold(cradle.shape))
    assert rep["ok"] and rep["maxDepth"] <= 0.05, rep
    assert rep["drawnCut"] and abs(rep["drawnCut"][0]["depth"] - 0.4) < 0.02, rep["drawnCut"]
    # a cradle that only touches the board is left as drawn
    assert _body_study(0.0).preserve[0].uncut is None
    # a delivered part with a thin film 0.6 mm into the board's edge: 0.5
    # mm^3, under the old volume bar, fails on depth
    film = m3d.Manifold.cube((0.6, 4.0, 0.2)).translate((9.4, -2.0, 0.7))
    part = exact_manifold(cradle.shape) + film
    rep = body_intrusion(st, part)
    assert rep["volume"] < 1.0 and not rep["ok"], rep
    assert abs(rep["bodies"][0]["depth"] - 0.6) < 0.03 and rep["bodies"][0]["id"] == "board", rep


def test_the_enclosure_marks_every_component_a_body():
    """Every placed component's keep-out is its body, so its cradle is cut
    to it and the delivered part is checked against it."""
    out = tempfile.mkdtemp()
    r = enclosure(product(), _DIR, out)
    comps = {c["id"] for c in product()["components"]}
    keep = {k["id"]: k for k in r["study"]["keepOut"] if k["id"] in comps}
    assert keep and all(k.get("body") for k in keep.values()), keep


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
