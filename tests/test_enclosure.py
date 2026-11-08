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
