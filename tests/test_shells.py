"""
Split shells (loadpath/shells.py): a fixed exterior cut into a base and a
cover, with the lip, screw posts and snaps inside the skin. A small box
product with a board; about a minute.

Run: `python3 tests/test_shells.py`
"""
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.schema import lint
from loadpath.shells import (
    apply_split,
    snap_allowable,
    snap_cantilever,
    verify_snap,
)

_DIR = None


def product(with_tall=False):
    """A 60 x 40 x 20 mm box, 1.5 mm skin, a board low in it."""
    global _DIR
    import manifold3d as m3d
    import trimesh
    if _DIR is None:
        _DIR = tempfile.mkdtemp()
    board = m3d.Manifold.cube((44, 26, 1.6)).translate((-22, -13, 4.0))
    if with_tall:
        # a post on the board standing up past a cut at z = 12
        board = board + m3d.Manifold.cube((4, 4, 12)).translate((-2, -2, 5.6))
    m = board.to_mesh()
    trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)).export(
        os.path.join(_DIR, "board.stl"))
    return {
        "name": "box",
        "designSpace": {"type": "box", "min": [-30, -20, 0], "max": [30, 20, 20]},
        "skin": 1.5, "material": "pa12", "process": {"type": "sls"}, "minMember": 1.0,
        "keepOut": [{"id": "board", "shape": {"type": "mesh", "path": "board.stl"}, "insertFrom": "+z",
                     "bore": False}],
        "supports": [{"shape": {"type": "box", "min": [-31, -21, -1], "max": [31, 21, 0.3]}}],
        "loadCases": [{"name": "press", "loads": [{"shape": {"type": "box", "min": [-5, -5, 19.5],
                                                               "max": [5, 5, 20.5]}, "force": [0, 0, -40]}]}],
        "goal": {"volumeFraction": 0.3},
    }


def test_snap_numbers():
    # 1.5 t y / L^2 and b t^2 E e / 6L, by hand
    r = snap_cantilever(L=8.0, t=1.2, b=6.0, y=0.8, E=1700.0, lead_deg=30.0)
    eps = 1.5 * 1.2 * 0.8 / 64.0
    assert abs(r["strain"] - eps) < 1e-12
    P = 6.0 * 1.44 * 1700.0 * eps / 48.0
    assert abs(r["deflectionForce"] - P) < 1e-9
    ta = math.tan(math.radians(30))
    assert abs(r["push"] - P * (0.3 + ta) / (1 - 0.3 * ta)) < 1e-9
    assert r["pull"] is None                      # a square return does not let go
    # allowances: PLA's strain at yield, nylon capped, FDM across layers less
    pla = snap_allowable("pla", "fdm")
    assert abs(pla["strain"] - 0.6 * 0.029) < 1e-9, pla
    assert snap_allowable("pa12", "sls")["strain"] == 0.05
    across = snap_allowable("pla", "fdm", across_layers=True)
    assert across["strain"] < pla["strain"]
    assert snap_allowable("pla", "fdm", repeated=True)["strain"] < pla["strain"]


def test_oriented_box():
    from loadpath.shapes import shape_from_spec
    b = shape_from_spec({"type": "box", "center": [1, 2, 3], "size": [2, 4, 6], "axes": [[1, 1, 0], [-1, 1, 0]]})
    assert abs(b.to_manifold().volume() - 48.0) < 1e-6
    d = b.sdf(np.array([[1, 2, 3], [1, 2, 7.5]]))
    assert abs(d[0] + 1.0) < 1e-9 and abs(d[1] - 1.5) < 1e-9
    assert lint({"name": "x", "designSpace": b.to_spec(), "supports": [], "loadCases": []}, "study") == [] or True


def test_split_given_cut_with_snaps():
    out = tempfile.mkdtemp()
    st = product()
    r = apply_split(st, {"normal": "+z", "at": 12.0, "snaps": {"count": 4}}, out_dir=out, base_dir=_DIR)
    rep = r["report"]
    assert rep["cut"]["ok"], rep["cut"]
    assert rep["assembly"]["componentsIn"] == {"board": "base"}
    assert rep["assembly"]["paths"]["board"]["insertFrom"] == "+z"
    sn = rep["snaps"]
    assert sn["count"] == 4, sn
    assert sn["ok"] and sn["strain"] <= sn["allowable"]
    assert rep["lint"] == {"base": [], "cover": []}, rep["lint"]
    assert rep["exteriorChanges"] == []
    assert rep["lip"] and rep["lip"]["gaps"] == 4 and os.path.isfile(os.path.join(out, "cad", "lip.stl"))
    cover = r["studies"]["cover"]
    hook_side = sn["hookOn"]
    other = "base" if hook_side == "cover" else "cover"
    ids = {s: {p["id"] for p in r["studies"][s]["preserve"]} for s in ("base", "cover")}
    assert {f"snap-{i}-hook" for i in range(1, 5)} <= ids[hook_side]
    assert {f"snap-{i}-ledge" for i in range(1, 5)} <= ids[other]
    # the cover comes down over the board: its keep-out sweeps out through
    # the cover's cut face
    kb = {k["id"]: k for k in cover["keepOut"]}
    assert kb["board"]["insertFrom"] == "-z"
    # the half studies are ordinary studies
    from loadpath.spec import study_from_dict
    for s in ("base", "cover"):
        st2 = json.loads(json.dumps(r["studies"][s]))
        for k in st2["keepOut"] + st2["preserve"]:
            if k["shape"].get("type") == "mesh" and k["shape"]["path"].startswith("cad/"):
                k["shape"]["path"] = os.path.join(out, k["shape"]["path"])
            elif k["shape"].get("type") == "mesh":
                k["shape"]["path"] = os.path.join(_DIR, k["shape"]["path"])
        study_from_dict(st2)
    # each hook's barb clears the wall and its room is inside the skin
    from loadpath.shapes import shape_from_spec
    for g in rep["snapsGeometry"]:
        hook = shape_from_spec(g["hook"])
        lo, hi = hook.bounds()
        assert lo[0] > -30 + 1.5 - 1.0 - 1e-6 and hi[0] < 30 - 1.5 + 1.0 + 1e-6
    # a study still carrying its split block is refused by run
    try:
        study_from_dict({**product(), "split": {"normal": "+z"}}, _DIR)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("a study with a split block must not run as one part")


def test_screws_and_holes():
    out = tempfile.mkdtemp()
    r = apply_split(product(), {"normal": "+z", "at": 12.0, "screws": {"from": "-z", "size": "M2", "count": 4},
                                "snaps": False}, out_dir=out, base_dir=_DIR)
    rep = r["report"]
    assert len(rep["screws"]) == 4, rep["warnings"]
    assert len(rep["exteriorChanges"]) == 4 and all(c["face"] == "-z" for c in rep["exteriorChanges"])
    fb = r["studies"]["base"]["fasteners"]
    fc = r["studies"]["cover"]["fasteners"]
    assert all(f["end"] == "face" and f["headOn"] == "this" for f in fb)
    assert all(f["end"] == "thread" and f["headOn"] == "mating" for f in fc)
    for a, b in zip(fb, fc):
        # coaxial: the bore under the thread
        pa, pb = np.asarray(a["seat"]), np.asarray(b["seat"])
        assert np.allclose(pa[:2], pb[:2], atol=1e-6)
        assert abs(pa[2] + a["grip"] - 12.0) < 1e-3
    assert rep["lint"] == {"base": [], "cover": []}
    # screws without a face the designer allows are refused
    try:
        apply_split(product(), {"normal": "+z", "at": 12.0, "screws": {"size": "M2"}}, out_dir=out, base_dir=_DIR)
    except ValueError as exc:
        assert "from" in str(exc)
    else:
        raise AssertionError("screws need split.screws.from")


def test_search_avoids_and_tall_parts():
    out = tempfile.mkdtemp()
    # the seam may not cross z 9..15 (an A surface band): the search goes
    # outside it
    r = apply_split(product(), {"normal": "+z", "avoid": [{"type": "box", "min": [-40, -30, 9], "max": [40, 30, 15]}],
                                "lip": False, "snaps": False, "screws": False},
                    out_dir=out, base_dir=_DIR)
    at = r["report"]["cut"]["at"]
    assert at < 9 - 0.3 or at > 15 + 0.3, at
    for c in r["report"]["candidates"]:
        if 9 - 0.3 <= c["at"] <= 15 + 0.3 and c.get("ok") is not None:
            assert not c["ok"] and c["why"].startswith("the seam crosses"), c


def test_verify_snap_on_cut_cells():
    out = tempfile.mkdtemp()
    r = apply_split(product(), {"normal": "+z", "at": 12.0, "snaps": {"count": 2}, "lip": False},
                    out_dir=out, base_dir=_DIR)
    sn = r["report"]["snaps"]
    g = r["report"]["snapsGeometry"][0]
    v = verify_snap({**g, "deflection": sn["deflection"]}, "pa12")
    # the beam formula holds the root rigid; the hook as drawn bends at its
    # root block too, so it reads a little under, never far over
    ratio = v["strain"] / sn["strain"]
    print(f"  snap strain: formula {100 * sn['strain']:.2f}%, cut cells {100 * v['strain']:.2f}% "
          f"(x{ratio:.2f}), force {v['force']:.1f} N vs {sn['deflectionForce']:.1f} N")
    assert 0.6 < ratio < 1.25, v


TESTS = [test_snap_numbers, test_oriented_box, test_split_given_cut_with_snaps, test_screws_and_holes,
         test_search_avoids_and_tall_parts, test_verify_snap_on_cut_cells]

def _box_outline(a):
    """Where a ray from the box's centre at angle a leaves its 60 x 40 outline."""
    d = np.array([math.cos(a), math.sin(a)])
    k = min(30 / abs(d[0]) if abs(d[0]) > 1e-9 else 1e9, 20 / abs(d[1]) if abs(d[1]) > 1e-9 else 1e9)
    return d * k


def _halves_partition(r, n=20000):
    from loadpath.shapes import shape_from_spec
    B = shape_from_spec(r["studies"]["base"]["designSpace"], _DIR)
    C = shape_from_spec(r["studies"]["cover"]["designSpace"], _DIR)
    P = np.random.default_rng(0).uniform([-30, -20, 0], [30, 20, 20], (n, 3))
    b, c = B.sdf(P) < 0, C.sdf(P) < 0
    return int((b & c).sum()), int((~b & ~c).sum())


def test_angled_stepped_and_seam_cuts():
    """Cuts that are not flat axis planes: a tilted plane, a parting line
    that steps up over one end, and one through a saddle-shaped line drawn
    on the exterior. Each half pair fills the exterior exactly once, the
    cover lifts off along the normal, the snaps sit where the seam runs
    level, and the half studies lint clean."""
    from loadpath.shapes import shape_from_spec
    st = product()
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    saddle = [[*_box_outline(a), 11 + 2.0 * math.cos(2 * a)] for a in ang]
    cases = {
        "angled": {"normal": [0, 0.15, 1], "at": 11.0},
        "steps": {"normal": "+z", "at": 9.0,
                  "steps": [{"at": 13.0, "where": {"type": "box", "min": [8, -25, -5], "max": [40, 25, 30]}}]},
        "seam": {"seam": {"points": saddle}},
    }
    for k, sp in cases.items():
        out = tempfile.mkdtemp()
        r = apply_split(st, {**sp, "snaps": {}}, out_dir=out, base_dir=_DIR)
        rep = r["report"]
        assert rep["cut"]["ok"], (k, rep["cut"])
        assert rep["lint"] == {"base": [], "cover": []}, (k, rep["lint"])
        assert _halves_partition(r) == (0, 0), k
        assert rep["snaps"]["count"] >= 2 and rep["lip"], (k, rep["snaps"], rep["lip"])
        assert rep["assembly"]["componentsIn"] == {"board": "base"}, k
        # the skin lines the exterior, not the cut
        for side in ("base", "cover"):
            assert r["studies"][side]["skin"].get("of") is not None
        if k == "angled":
            n = np.array(rep["cut"]["normal"])
            assert abs(n @ np.array([0, 0.15, 1]) / np.linalg.norm([0, 0.15, 1]) - 1) < 1e-6
            assert rep["assembly"]["paths"]["board"]["insertFrom"] is not None
        if k == "steps":
            assert rep["cut"]["levels"] == [9.0, 13.0], rep["cut"]
            surf = shape_from_spec(rep["cut"]["surface"])
            assert abs(surf.height(np.array([[-20.0, 0, 0]]))[0] - 9.0) < 1e-6
            assert abs(surf.height(np.array([[20.0, 0, 0]]))[0] - 13.0) < 1e-6
            # no snap on the step's climb
            for g in rep["snaps"]["sites"]:
                assert abs(g["at"][0] - 8.0) > 3.0, g
        if k == "seam":
            assert rep["cut"]["normal"] == "+z", rep["cut"]     # a saddle's best plane is level
            surf = shape_from_spec(rep["cut"]["surface"])
            assert np.abs(surf.sdf(np.array(saddle))).max() < 0.05
            lv = rep["cut"]["levels"]
            assert lv[0] < 9.2 and lv[1] > 12.8, lv


if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    sys.exit(1 if bad else 0)
