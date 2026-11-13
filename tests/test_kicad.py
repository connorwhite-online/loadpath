"""
KiCad boards in (loadpath/kicad.py): a .kicad_pcb read into its outline,
stackup, part bodies and heights, mounting holes, connectors and heat, and
turned into enclosure components (held on screws through its holes, its
USB-C seated in a port with its plug loads) and a stack board.

The fixture (tests/data/kicad/sensor-board.kicad_pcb) is a 40 x 28 mm board
built from KiCad's own library footprints: four M2.5 mounting holes, a
GCT USB4085 USB-C receptacle over its -y edge, a QFN with Power and MaxTemp
fields, an SOT-223 regulator, 0805 capacitors (one on the back), a 6.3 x
7.7 mm electrolytic with a VRML model, a JST SH lead and a 2.54 mm header.

Run: `python3 tests/test_kicad.py`
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

from loadpath import kicad  # noqa: E402
from loadpath.schema import lint  # noqa: E402

BOARD = str(ROOT / "tests" / "data" / "kicad" / "sensor-board.kicad_pcb")
_DIR = tempfile.mkdtemp()


def _parts(b):
    return {p["ref"]: p for p in b["parts"]}


def test_outline_stackup_and_mass():
    b = kicad.read_board(BOARD)
    assert b["size"][:2] == [40.0, 28.0], b["size"]
    # the stackup's copper and core, not general.thickness
    assert abs(b["thickness"] - 1.58) < 1e-9 and b["thicknessFrom"] == "stackup"
    assert len(b["cutouts"]) == 1                       # the 3 x 6 mm slot
    area = abs(kicad._area(np.asarray(b["outline"]))) - abs(kicad._area(np.asarray(b["cutouts"][0])))
    corners = 40 * 28 - 4 * (4 - math.pi)               # four 2 mm round corners
    assert abs(area - (corners - 18.0)) < 0.3, area
    want = area * 1e-6 * (1.51e-3 * 1850 + 0.07e-3 * 8960 * 0.5)
    assert abs(b["mass"] - want) / want < 1e-3, (b["mass"], want)
    assert 5 < b["conductivity"] < 15                   # copper in parallel with FR-4


def test_holes():
    b = kicad.read_board(BOARD)
    hs = sorted((tuple(h["at"]), h["screw"], h["pad"]) for h in b["holes"])
    assert [h[1] for h in hs] == ["M2.5"] * 4, hs
    assert {(abs(x), abs(y)) for (x, y), _, _ in hs} == {(16.5, 10.5)}
    assert all(abs(h[2] - 5.4) < 1e-6 for h in hs)


def test_parts_heights_and_sides():
    P = _parts(kicad.read_board(BOARD))
    assert P["U1"]["heightFrom"].startswith("package") and P["U1"]["height"] == 1.0          # QFN
    assert P["U2"]["height"] == 1.8                                                          # SOT-223
    assert P["C1"]["height"] == 1.45                                                         # 0805 MLCC
    assert P["J3"]["height"] == 8.5                                                          # 2.54 header
    # the electrolytic's VRML model (0.1 inch units) is its body
    c4 = P["C4"]
    assert c4["heightFrom"] == "3D model" and abs(c4["height"] - 7.7) < 1e-3, c4["height"]
    h = c4["_hull"]
    assert np.allclose(h[:, :2].min(0) + 3.3, c4["at"], atol=1e-3) and np.allclose(h[:, :2].max(0) - 3.3, c4["at"],
                                                                                    atol=1e-3)
    # a back-side part hangs under the board
    assert P["C5"]["side"] == "bottom" and P["C5"]["z"] == [-1.45, 0.0]
    # through-hole leads stand out of the far face
    assert P["J3"]["leads"]["z"] == [-1.5, 0.0]
    # moulded QFN: 5 x 5 x 1 mm at 2.0 g/cm3
    assert 0.02e-3 < P["U1"]["mass"] < 0.08e-3


def test_connectors_face_out():
    P = _parts(kicad.read_board(BOARD))
    j1 = P["J1"]["connector"]
    assert j1["type"] == "usb-c" and j1["out"] == "-y" and j1["how"] == "board edge"
    # the receptacle overhangs the edge by 8.61 - 6.1 mm (its fab outline)
    assert abs(j1["mouth"][1] - (-14.0 - 2.51)) < 1e-6, j1
    assert P["J2"]["connector"]["out"] == "+y"
    assert P["J3"]["connector"]["out"] == "+z"           # a vertical header mates from above


def test_power_fields_and_overrides():
    b = kicad.read_board(BOARD, power={"U2": 0.4}, max_temp={"U2": 105}, heights={"J2": 3.1})
    P = _parts(b)
    assert P["U1"]["watts"] == 0.35 and P["U1"]["wattsFrom"] == "footprint Power field" and P["U1"]["maxTemp"] == 85
    assert P["U2"]["watts"] == 0.4 and P["U2"]["maxTemp"] == 105
    assert P["J2"]["height"] == 3.1 and P["J2"]["heightFrom"] == "brief"


def test_model_placement_rotated_and_flipped():
    """A model is placed as KiCad places it: turned with its footprint, and
    on the back flipped and hung under the board."""
    d = tempfile.mkdtemp()
    V = [(x, y, z) for x in (0, 4) for y in (0, 1) for z in (0, 2)]
    pts = ", ".join(f"{x / 2.54} {y / 2.54} {z / 2.54}" for x, y, z in V)
    faces = "0, 1, 3, 2, -1, 4, 6, 7, 5, -1, 0, 4, 5, 1, -1, 2, 3, 7, 6, -1, 0, 2, 6, 4, -1, 1, 5, 7, 3, -1"
    Path(d, "blk.wrl").write_text(f"#VRML V2.0 utf8\nShape {{ geometry IndexedFaceSet {{ coord Coordinate "
                                  f"{{ point [ {pts} ] }} coordIndex [ {faces} ] }} }}\n")
    fp = ('(footprint "x:Blk" (layer "{L}.Cu") (at 20 20 90) (property "Reference" "{R}") '
          '(fp_rect (start -1 -1) (end 1 1) (layer "{L}.Fab")) (pad "1" smd rect (at 0 0) (size 1 1) '
          '(layers "{L}.Cu")) (model "${{KIPRJMOD}}/blk.wrl"))')
    txt = ('(kicad_pcb (general (thickness 1.6)) (gr_rect (start 0 0) (end 40 40) (layer "Edge.Cuts")) '
           + fp.format(L="F", R="U1") + fp.format(L="B", R="U2") + ")")
    Path(d, "b.kicad_pcb").write_text(txt)
    P = _parts(kicad.read_board(os.path.join(d, "b.kicad_pcb")))
    top, bot = P["U1"]["_hull"], P["U2"]["_hull"]
    # turned 90 degrees counter-clockwise: x -> y, y -> -x
    assert np.allclose(top.min(0), [-1, 0, 1.6]) and np.allclose(top.max(0), [0, 4, 3.6]), (top.min(0), top.max(0))
    # on the back: half a turn about y then z (KiCad's 3D viewer: y and z
    # mirrored, as the flipped pads are), hung under the board, then turned
    assert np.allclose(bot.min(0), [0, 0, -2]) and np.allclose(bot.max(0), [1, 4, 0]), (bot.min(0), bot.max(0))


def _wrl(meshes):
    """VRML 2.0 (0.1 inch units) of trimesh solids, one Shape each."""
    out = ["#VRML V2.0 utf8"]
    for m in meshes:
        pts = ", ".join(f"{x / 2.54} {y / 2.54} {z / 2.54}" for x, y, z in m.vertices)
        idx = ", ".join(f"{a}, {b}, {c}, -1" for a, b, c in m.faces)
        out.append(f"Shape {{ geometry IndexedFaceSet {{ coord Coordinate {{ point [ {pts} ] }} "
                   f"coordIndex [ {idx} ] }} }}")
    return "\n".join(out) + "\n"


def test_a_sockets_body_is_its_model_not_the_models_hull():
    """A USB-C socket's body is its model's solids: the 8.94 x 3.26 mm
    stadium shell with its solder tabs and through pins apart, not the hull
    of all of them (a squared block over the tabs that every keep-out, way
    in and cutout then took). Mid-mount, the shell below the board's face
    in its slot is the body's too; only what is through the board goes."""
    import manifold3d as m3d
    import trimesh

    from loadpath.shapes import trimesh_from_manifold

    def stadium(w, h, L, z0):
        r = h / 2
        cs = (m3d.CrossSection.square([w - 2 * r, h]).translate([-(w - 2 * r) / 2, -r])
              + m3d.CrossSection.circle(r, 64).translate([-(w - 2 * r) / 2, 0])
              + m3d.CrossSection.circle(r, 64).translate([(w - 2 * r) / 2, 0]))
        # its axis along y (the plug's), wide along x, centred at z0 + h/2
        return trimesh_from_manifold(m3d.Manifold.extrude(cs, L).rotate([-90, 0, 0])
                                     .translate([0, 0, z0 + r]))

    def box(lo, hi):
        b = trimesh.creation.box(extents=np.subtract(hi, lo))
        b.apply_translation(0.5 * (np.add(lo, hi)))
        return b
    for mid in (False, True):
        d = tempfile.mkdtemp()
        z0 = -1.6 if mid else 0.0                          # mid-mount: shell 1.6 mm down in a slot
        shell = stadium(8.94, 3.26, 7.35, z0)
        tabs = [box((sx * 5.6, 1.0, 0.0), (sx * 4.9, 3.0, 1.2)) if sx > 0 else box((-5.6, 1.0, 0.0), (-4.9, 3.0, 1.2))
                for sx in (1, -1)]
        pins = [box((sx * 4.4 - 0.3, 5.0, -2.5), (sx * 4.4 + 0.3, 5.6, 0.2)) for sx in (1, -1)]
        Path(d, "usbc.wrl").write_text(_wrl([shell] + tabs + pins))
        fp = ('(footprint "Connector_USB:USB_C_Receptacle_X" (layer "F.Cu") (at 20 30) (property "Reference" "J1") '
              '(fp_rect (start -5.6 -7.35) (end 5.6 0) (layer "F.Fab")) (pad "1" smd rect (at 0 -6) (size 1 1) '
              '(layers "F.Cu")) (model "${KIPRJMOD}/usbc.wrl"))')
        slot = '(gr_rect (start 15 22) (end 25 31) (layer "Edge.Cuts"))' if mid else ""
        Path(d, "b.kicad_pcb").write_text('(kicad_pcb (general (thickness 1.6)) (gr_rect (start 0 0) (end 40 40) '
                                          '(layer "Edge.Cuts")) ' + slot + fp + ")")
        P = _parts(kicad.read_board(os.path.join(d, "b.kicad_pcb")))
        body = trimesh_from_manifold(kicad.part_manifold(P["J1"], leads=False))
        c = np.array(P["J1"]["at"])
        zc = 1.6 + z0 + 1.63                               # the shell's axis height on the board
        # the corner the hull filled, between the shell's round end and a tab:
        # outside the shell, so not in the body
        # (above the tab's top, 2.8 mm, under the hull's face from the shell to it)
        assert not body.contains([[c[0] + 5.0, c[1] + 2.0, 2.85 if mid else 3.0]])[0]
        # the shell is
        assert body.contains([[c[0], c[1] + 2.0, zc], [c[0] + 4.0, c[1] + 2.0, zc]]).all()
        # the hull's volume was the block over the tabs; the body is the shell and tabs
        v_shell = (8.94 * 3.26 - (4 - np.pi) * 1.63 ** 2) * 7.35
        assert body.volume < 1.1 * v_shell + 2 * 0.7 * 2.0 * 1.2, body.volume
        if mid:
            # the shell below the board's face, in the slot, is kept
            assert body.contains([[c[0], c[1] + 2.0, 0.3]])[0]
            assert body.bounds[0][2] < 0.01, body.bounds


def test_enclosure_components():
    spec = {"id": "main", "rotate": "z", "kicad": {"file": BOARD, "connectors": {"J1": {"opening": "usb"}},
                                                   "power": {"U2": 0.4}}}
    comps, loads, rep = kicad.enclosure_components(spec, None, _DIR)
    ids = [c["id"] for c in comps]
    # connectors and heat sources are components of their own on the board
    assert ids[0] == "main" and set(ids[1:]) == {"main-J1", "main-J2", "main-J3", "main-U1", "main-U2"}, ids
    by = {c["id"]: c for c in comps}
    assert by["main"]["mount"]["type"] == "holes" and len(by["main"]["mount"]["holes"]) == 4
    assert by["main"]["rotate"] == "z"
    assert by["main-J1"]["installedOn"] == "main"
    assert by["main-J1"]["alignTo"] == {"opening": "usb", "face": "-y", "recess": 0.3}
    assert by["main-U1"]["watts"] == 0.35 and by["main-U2"]["watts"] == 0.4
    assert loads == [{"component": "main-J1", "type": "usb-c"}]
    total = sum(c["mass"] for c in comps)
    b = kicad.read_board(BOARD)
    assert abs(total - (b["mass"] + sum(p["mass"] for p in b["parts"]))) < 2e-6
    for c in comps:
        assert os.path.isfile(c["shape"]["path"])
    # the brief lints
    brief = {"exterior": "x.stl", "components": [spec], "loads": {"drop": {"g": 100}}}
    assert not [x for x in lint(brief, "enclosure") if x["level"] == "error"], lint(brief, "enclosure")
    bad = {"exterior": "x.stl", "components": [{"id": "b", "kicad": {"file": BOARD, "powr": {"U1": 1}}}]}
    assert any("powr" in x["message"] or "powr" in x["path"] for x in lint(bad, "enclosure"))


def test_stack_board():
    from loadpath.stack import expand_boards
    sb = kicad.stack_board({"id": "s", "kicad": BOARD, "watts": 0.5})
    assert sb["size"] == [40.0, 28.0, 1.58]
    assert sb["above"] == 8.5 and sb["below"] == 1.5             # header on top, its leads under
    assert sb["watts"] == 0.5                                     # the entry wins
    assert {p["id"] for p in sb["ports"]} == {"J1", "J2", "J3"}
    br = expand_boards({"boards": [{"id": "s", "kicad": "sensor-board.kicad_pcb"}]}, str(Path(BOARD).parent))
    assert br["standoffs"]["count"] == 4 and br["boards"][0]["maxTemp"] == 85


def test_real_enclosure():
    """The board in a box with a USB-C slot in its -y end and a lid on
    top: placed with its receptacle 0.3 mm short of the outside, held on
    its four M2.5 holes, with the plug's loads through those screws."""
    import manifold3d as m3d
    import trimesh

    from loadpath.enclosure import enclosure

    def stl(name, man):
        m = man.to_mesh()
        p = os.path.join(_DIR, name)
        trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)).export(p)
        return p
    box = m3d.Manifold.cube((56, 42, 22)).translate((-28, -21, -11))
    port = m3d.Manifold.cube((11, 3.0, 5)).translate((-5.5, -23, -6))
    door = m3d.Manifold.cube((48, 34, 2.0)).translate((-24, -17, 10))
    lid = m3d.Manifold.cube((47.6, 33.6, 1.2)).translate((-23.8, -16.8, 9.8))
    brief = {"name": "kicad-box", "exterior": stl("ext.stl", box - port - door), "skin": 1.2,
             "openings": [{"id": "usb", "shape": stl("port.stl", port), "inward": "+y"},
                          {"id": "door", "shape": stl("door.stl", door), "inward": "-z"}],
             "components": [{"id": "board", "kicad": {"file": BOARD, "connectors": {"J1": {"opening": "usb"}}},
                             "rotate": "z", "margin": 0.3},
                            {"id": "lid", "shape": stl("lid.stl", lid), "place": "given", "external": True,
                             "mount": False}],
             "assembly": ["board", "lid"], "loads": {"drop": {"g": 200, "faces": ["-z"]}},
             "material": "pa12", "minMember": 1.2}
    assert not [x for x in lint(brief, "enclosure") if x["level"] == "error"]
    r = enclosure(brief, _DIR, os.path.join(_DIR, "enc"))
    rep, st = r["report"], r["study"]
    pl = rep["placement"]["components"]
    assert abs(pl["board-J1"]["recess"] - 0.3) < 0.1, pl["board-J1"]
    assert rep["mounts"]["board"]["type"] == "holes" and rep["mounts"]["board"]["side"] == "-z", rep["mounts"]
    assert {"board-J1-insert", "board-J1-withdraw", "board-J1-side-thin", "board-J1-side-wide"} <= set(
        rep["loadCases"]), rep["loadCases"]
    keep = {k["id"] for k in st["keepOut"]}
    assert {"mount-board-H1-screw", "mount-board-H1-head"} <= keep
    mount = next(p for p in st["preserve"] if p["id"] == "mount-board")
    assert mount["shape"]["type"] == "union" and len(mount["shape"]["shapes"]) == 4
    # the plug pushes on the screws' sleeves, not on a cradle
    ins = next(c for c in st["loadCases"] if c["name"] == "board-J1-insert")
    assert ins["loads"][0]["region"] == "mount-board"
    assert rep["boards"][0]["id"] == "board" and not rep["lint"], rep["lint"]
    with open(os.path.join(_DIR, "enc.json"), "w") as fh:
        json.dump(rep, fh, default=float)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
