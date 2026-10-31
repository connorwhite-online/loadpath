"""
Bench test for the wrist yoke Connor printed (GRIP generative/print/
grip-wrist-yoke-PETG-FDM-v3.stl, commit 6650c7b): two printed fixtures, the
predicted drop per kg, and the safe load.

The test: the base fixture stands in for the J4 servo horn and is held in a
vise by its long edge, so the yoke sticks out sideways. The dummy fixture
stands in for the J5 servo, screwed in with its eight case screws. A weight
hung from the dummy's eye bends the yoke the way the arm's reach does (a
moment about the J5 axis direction plus a force across it), and the eye's
drop is read with a dial indicator or a ruler.

    python examples/grip/yoke_test.py --yoke path/to/grip-wrist-yoke-PETG-FDM-v3.stl --out calibration/

Writes fixture-base.stl and fixture-dummy.stl (on the bed, in print
orientation), test-assembly.stl (the three in place, arm frame) and
yoke-test.json: predicted mm per kg at the eye for the whole assembly and
for the yoke alone on rigid fixtures, peak stress per kg, and a weight
that stays under a third of the PETG card's strength.

Frame: GRIP's arm frame (mm). The printed STL lies upside down on the bed;
REGISTER maps it back (found by correlating it with the study's preserved
bosses and keep-out screw holes; its screw holes land on the study's to
within a grid step, and its hub's underside sits at z 533.5, about 2.5 mm
below the study's hub: the horn-disc bug fixed in GRIP de445fd).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

# print STL -> arm frame: a half turn about x, then this shift
REGISTER = np.array([[1.0, 0, 0, -28.95], [0, -1.0, 0, -2.69], [0, 0, -1.0, 583.83], [0, 0, 0, 1.0]])

HUB_AXIS = (0.0, -25.0)                  # J4 axis (x, y)
HUB_FACE = 533.5                         # the yoke's flat underside at the hub
HORN_R, BOLT_R, PILOT_M25 = 11.0, 7.0, 2.1
# 25 mm thick and gripped 42 mm from the J4 axis: a 15 mm plate gripped at
# 55 mm turned ~1 mrad per kg, as much as the yoke itself
PLATE = {"x": (-62.0, 32.0), "y": (-55.0, 5.0), "z": (505.5, 530.5)}   # 1.3 mm under the yoke's legs
CLAMP_X = -42.0                          # vise jaws grip the plate from x = -62 to here
# J5 servo envelope (STS3250 body, 0.2-0.3 mm inside GRIP's keep-out box)
DUMMY = {"x": (-11.9, 11.9), "y": (-42.2, -8.4), "z": (538.9, 582.9)}
# Like the servo it replaces, the dummy touches the yoke only at its screw
# bosses: a pad round each screw reaches the boss face (as printed, short of
# it by PAD_CLEARANCE so it slides in; the screw pulls the boss onto it). A
# block bonded wherever it met the yoke stiffened the cradle ~3x.
PAD_R, PAD_CLEARANCE = 3.3, 0.15
PILOT_M2 = 1.6
FRONT_SCREWS = [(-8.35, 578.5), (8.35, 578.5), (-8.35, 543.3), (8.35, 543.3)]     # along y, at x, z
SIDE_SCREWS = [(-1, 577.5), (1, 577.5), (-1, 544.3), (1, 544.3)]                  # along x, side, z; y = -37
EYE = {"center": (0.0, -25.0, 590.0), "r": 3.0, "tab": (7.0, 3.0, 596.0)}         # hole axis along y
LOAD_DIR = np.array([-1.0, 0.0, 0.0])    # gravity in the arm frame during the test


def _box(lo, hi):
    from loadpath.calibration import _box as box
    return box(lo, hi)


def _cyl(p0, p1, r, n=48):
    import manifold3d as m3d
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    h = float(np.linalg.norm(d))
    c = m3d.Manifold.cylinder(h, r, r, n)
    z = d / h
    # rotate +z onto d
    v = np.cross([0, 0, 1.0], z)
    s, cth = np.linalg.norm(v), z[2]
    if s < 1e-9:
        R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = v / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * K + (1 - cth) * K @ K
    M = np.hstack([R, p0[:, None]])
    return c.transform(M.tolist())


def load_yoke(path):
    import trimesh
    m = trimesh.load(path)
    m.apply_transform(REGISTER)
    return m


def base_fixture():
    """Horn stand-in: a plate the vise grips, a 22 mm boss whose top meets
    the hub's underside, four M2.5 pilot holes on the horn's bolt circle."""
    x0, y0 = HUB_AXIS
    px, py, pz = PLATE["x"], PLATE["y"], PLATE["z"]
    base = _box([px[0], py[0], pz[0]], [px[1], py[1], pz[1]]) + _cyl([x0, y0, pz[1] - 1], [x0, y0, HUB_FACE], HORN_R)
    for i in range(4):
        a = np.pi / 4 + i * np.pi / 2
        x, y = x0 + BOLT_R * np.cos(a), y0 + BOLT_R * np.sin(a)
        base = base - _cyl([x, y, HUB_FACE - 14], [x, y, HUB_FACE + 1], PILOT_M25 / 2, 24)
    # a groove on the plate's face marks how far the vise jaws go
    base = base - _box([CLAMP_X - 0.6, py[0] - 1, pz[1] - 0.6], [CLAMP_X + 0.6, py[1] + 1, pz[1] + 1])
    return base


def dummy_fixture(yoke_manifold, clearance=PAD_CLEARANCE):
    """J5 stand-in: a block just inside the servo's envelope, pads at its
    eight case screws (M2 pilot holes through them) and an eye on top for
    the weight. clearance=0 gives the analysis model: pads on the bosses."""
    dx, dy, dz = DUMMY["x"], DUMMY["y"], DUMMY["z"]
    d = _box([dx[0], dy[0], dz[0]], [dx[1], dy[1], dz[1]])
    cx, cy, cz = EYE["center"]
    tx, ty, tz = EYE["tab"]
    tab = _box([cx - tx, cy - ty, dz[1] - 1], [cx + tx, cy + ty, tz])
    d = d + tab - _cyl([cx, cy - ty - 1, cz], [cx, cy + ty + 1, cz], EYE["r"])
    # pads reach 1.5 mm past the block, then the yoke (and the clearance)
    # is cut from them: what is left meets each boss face
    pads = []
    for x, z in FRONT_SCREWS:
        pads.append(_cyl([x, dy[1] - 2, z], [x, dy[1] + 1.5, z], PAD_R))
    for side, z in SIDE_SCREWS:
        xe = dx[1] if side > 0 else dx[0]
        pads.append(_cyl([xe - 2 * side, -37.0, z], [xe + 1.5 * side, -37.0, z], PAD_R))
    import manifold3d as m3d
    pad = m3d.Manifold.batch_boolean(pads, m3d.OpType.Add) - yoke_manifold
    if clearance > 0:
        # trim each pad's face back by cutting the yoke again, moved toward
        # the block along the front pads' axis and both side pads' axes
        for shift in ([0, -clearance, 0], [clearance, 0, 0], [-clearance, 0, 0]):
            pad = pad - yoke_manifold.translate(shift)
    d = d + pad
    for x, z in FRONT_SCREWS:
        d = d - _cyl([x, dy[1] + 3, z], [x, dy[1] - 9, z], PILOT_M2 / 2, 20)
    for side, z in SIDE_SCREWS:
        xe = dx[1] if side > 0 else dx[0]
        d = d - _cyl([xe + 3 * side, -37.0, z], [xe - 9 * side, -37.0, z], PILOT_M2 / 2, 20)
    return d - yoke_manifold


def _on_bed(man, R=None):
    """A fixture turned to print (R: 3x3, default as is) and set on z = 0."""
    if R is not None:
        man = man.transform(np.hstack([np.asarray(R, float), np.zeros((3, 1))]).tolist())
    lo = np.asarray(man.bounding_box()[:3])
    return man.translate((-lo).tolist())


def _study(material, supports, load, extent):
    from loadpath.spec import study_from_dict
    lo, hi = extent
    return study_from_dict({
        "designSpace": {"type": "box", "min": list(lo), "max": list(hi)},
        "supports": supports,
        "loadCases": [{"name": "1 kg on the eye", "loads": [load]}],
        "material": material, "goal": {"safetyFactor": 3.0},
    })


def predict(yoke, base, dummy, material="petg", max_cells=150_000, asm_cells=400_000):
    """Drop per kg at the eye for the assembly (fixtures as printed parts,
    joints bonded) and for the yoke alone on rigid fixtures."""
    from loadpath.calibration import G, part_prediction
    from loadpath.shapes import trimesh_from_manifold

    F = (G * LOAD_DIR).tolist()
    eye = np.array(EYE["center"])
    hook = eye + EYE["r"] * LOAD_DIR          # where the string bears
    asm = base + dummy + _manifold(yoke)
    asm_mesh = trimesh_from_manifold(asm)
    asm_mesh.merge_vertices()
    lo, hi = asm_mesh.bounds
    px, py, pz = PLATE["x"], PLATE["y"], PLATE["z"]
    clamp = {"shape": {"type": "box", "min": [px[0] - 1, py[0] - 1, pz[0] - 1], "max": [CLAMP_X, py[1] + 1, pz[1] + 1]}}
    eye_load = {"shape": {"type": "cylinder", "p0": [eye[0], eye[1] - 4, eye[2]], "p1": [eye[0], eye[1] + 4, eye[2]],
                          "r": EYE["r"] + 1.5}, "force": F}
    st = _study(material, [clamp], eye_load, (lo - 1, hi + 1))
    from loadpath.shapes import Box
    dx, dy, dz = DUMMY["x"], DUMMY["y"], DUMMY["z"]
    block = Box([dx[0] + 1, dy[0] + 1, dz[0] + 1], [dx[1] - 1, dy[1] - 1, dz[1] - 1])
    x0, y0 = HUB_AXIS
    probes = {"plate": Box([x0 - 15, y0 - 15, pz[0]], [x0 + 15, y0 + 15, pz[1]]),
              "hub": Box([x0 - 8, y0 - 8, HUB_FACE + 1], [x0 + 8, y0 + 8, HUB_FACE + 4])}
    out = {"assembly": part_prediction(st, asm_mesh, hook, rigid=block, probes=probes, max_cells=asm_cells)}
    a = out["assembly"]
    total = a["rotationMradPerKg"][1]
    # how the dummy's turn splits: plate, horn boss, then the yoke itself
    plate, hub = a["probeRotationMradPerKg"]["plate"][1], a["probeRotationMradPerKg"]["hub"][1]
    a["rotationShare"] = {"plate": round(plate / total, 3), "boss": round((hub - plate) / total, 3),
                          "yoke": round((total - hub) / total, 3)}

    # the yoke alone: hub held on the horn's face, the weight carried to the
    # J5 bosses as a force and its moment about their centroid
    hub = {"shape": {"type": "cylinder", "p0": [x0, y0, HUB_FACE - 0.5], "p1": [x0, y0, HUB_FACE + 1.5], "r": HORN_R}}
    bosses = [_boss(x, -6.2, z, "y") for x, z in FRONT_SCREWS] + \
             [_boss(14.15 * s, -37.0, z, "x") for s, z in SIDE_SCREWS]
    cen = np.mean([[b["p0"][0], b["p0"][1], b["p0"][2]] for b in bosses], axis=0)
    M = np.cross(hook - cen, np.asarray(F)).tolist()
    st2 = _study(material, [hub], {"shape": {"type": "union", "shapes": bosses}, "force": F, "moment": M},
                 (yoke.bounds[0] - 1, yoke.bounds[1] + 1))
    # the eye is not on the yoke: read the bosses' rigid motion there
    out["yokeAlone"] = part_prediction(st2, yoke, hook, rigid=st2.load_cases[0].loads[0].shape,
                                       max_cells=max_cells)
    for k in out:
        # a laser taped to the dummy, pointing along +z (away from the vise),
        # moves its spot on a wall by the rotation about y times the distance
        out[k]["laserSpotMmPerKgPerM"] = abs(out[k]["rotationMradPerKg"][1])
    return out, asm_mesh


def _boss(x, y, z, axis, r=3.5, half=2.2):
    d = np.array([0.0, half, 0.0]) if axis == "y" else np.array([half, 0.0, 0.0])
    p = np.array([x, y, z])
    return {"type": "cylinder", "p0": (p - d).tolist(), "p1": (p + d).tolist(), "r": r}


def _manifold(mesh):
    from loadpath.shapes import manifold_from_trimesh
    return manifold_from_trimesh(mesh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--yoke", required=True, help="grip-wrist-yoke-PETG-FDM-v3.stl as printed")
    ap.add_argument("--out", default="yoke-test")
    ap.add_argument("--material", default="petg")
    ap.add_argument("--max-cells", type=int, default=150_000, help="the yoke alone")
    ap.add_argument("--asm-cells", type=int, default=400_000, help="yoke + fixtures")
    a = ap.parse_args(argv)
    from loadpath.materials import material_from_spec
    from loadpath.shapes import trimesh_from_manifold

    os.makedirs(a.out, exist_ok=True)
    yoke = load_yoke(a.yoke)
    base = base_fixture()
    ym = _manifold(yoke)
    dummy = dummy_fixture(ym)                   # as printed
    dummy_fe = dummy_fixture(ym, clearance=0.0)   # screwed tight
    # both print as they sit in the arm frame: the base on its plate, boss
    # up; the dummy on its flat bottom, eye up (its pads stick out of the
    # sides and front, so none of them is on the bed)
    trimesh_from_manifold(_on_bed(base)).export(os.path.join(a.out, "fixture-base.stl"))
    trimesh_from_manifold(_on_bed(dummy)).export(os.path.join(a.out, "fixture-dummy.stl"))
    pred, asm = predict(yoke, base, dummy_fe, a.material, a.max_cells, a.asm_cells)
    asm.export(os.path.join(a.out, "test-assembly.stl"))
    mat = material_from_spec(a.material)
    # the yoke's own peak (the assembly's grid is coarser, and its peak can
    # sit in a fixture)
    peak = pred["yokeAlone"]["peakStressMPaPerKg"]
    pred["safeKg"] = round(mat.strength / 3.0 / peak, 2)
    pred["yokeMassG"] = round(float(yoke.volume) * mat.density / 1000, 2)
    # the entry calibration.write_kit / fit take as parts["wrist-yoke"]
    kgs = [k for k in (1, 2, 3, 4, 5) if k <= pred["safeKg"]]
    pred["calibrationPart"] = {"mmPerKg": pred["assembly"]["mmPerKg"],
                               "laserSpotMmPerKgPerM": pred["assembly"]["laserSpotMmPerKgPerM"],
                               "E": mat.E, "density": mat.density, "massG": pred["yokeMassG"], "loadsKg": kgs}
    pred["setup"] = {"loadDirection": LOAD_DIR.tolist(), "eye": EYE["center"], "clampToX": CLAMP_X,
                     "frame": "GRIP arm frame, mm", "register": REGISTER.tolist()}
    with open(os.path.join(a.out, "yoke-test.json"), "w") as fh:
        json.dump(pred, fh, indent=2, default=float)
    print(json.dumps(pred, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
