"""
The agent's step: from what the device does to the parts inside it.

The description (what a person would say about the device):

    A palm-sized Bluetooth camera shutter remote. You press the dished
    button on top to take a photo; it should click at a light, definite
    press there and not go off when it is squeezed elsewhere in a pocket.
    A CR2032 coin cell runs it; the battery door is on the underside and
    comes off with the board for a battery change. Soft-touch TPU, SLS.

From that and the exterior alone, this script decides what goes inside and
where, and writes the handoff brief:

- the **button** is the dish on top: found from the exterior's own top
  surface (where it sits deepest under the convex hull, a straight edge
  laid across the dish, and curves up like a bowl), centred on the circle
  its rim makes;
- the **door** goes where the underside is flattest, as large as the flat
  allows (a coin cell needs 21 mm across, plus its holder);
- the **board** rides on the door, half a millimetre in from its edges,
  so it goes in and out through the opening with it; the **switch** (a 6 x 6 mm
  tall tactile switch, 160 gf, 0.25 mm travel) sits on the board under the
  button, the **radio chip** beside it, and the **coin-cell holder** at the
  board's other end;
- the **press windows**: a click between 200 and 450 gf on the button
  (its middle and four points 5 mm out), and at least 1.2 kgf on the top
  clear of its rim, behind it and either side (no clicks in a pocket);
- the **skin**: 1.0 mm, SLS TPU's minimum wall. Thicker reads stiffer
  than the button's ceiling by the rim before anything is inside.

Every part is checked against the room inside the exterior once its
skin is in, and the script stops, saying which part does not fit
and by how much, when one does not. It reads nothing the description and
the exterior do not say.

    python layout.py exterior/exterior.stl --out .
"""
import argparse
import json
import os

import numpy as np

SKIN = 1.0          # SLS TPU's minimum wall: a 1.2 mm skin alone already clicks over 450 gf by the rim
SWITCH = {"body": (6.0, 6.0, 3.5), "stem_r": 1.75, "height": 9.5, "trigger_gf": 160, "travel": 0.25}
COIN_HOLDER = (22.0, 22.0, 4.2)     # a CR2032 holder (20 mm cell, 3.2 mm thick)
MODULE = (7.0, 7.0, 1.2)            # the BLE chip (a 6 x 6 mm QFN with its antenna keep-off)
DOOR_T, BOARD_T = 1.6, 1.0


def heightmaps(mesh, step=0.5):
    """Top and bottom surface heights over the exterior's footprint."""
    lo, hi = mesh.bounds
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    org = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, hi[2] + 5)])
    top = np.full(X.size, np.nan)
    bot = np.full(X.size, np.nan)
    for sign, arr in ((-1, top), (1, bot)):
        o = org.copy()
        if sign > 0:
            o[:, 2] = lo[2] - 5
        loc, ray, _ = mesh.ray.intersects_location(o, np.tile([0, 0, sign], (len(o), 1)), multiple_hits=False)
        arr[ray] = loc[:, 2]
    return xs, ys, top.reshape(X.shape), bot.reshape(X.shape)


def find_button(mesh, step=0.5):
    """The dish: where the top sits furthest under the exterior's convex
    hull (the hull spans the dish's rim like a straight edge laid across
    it). Its middle is the depth-weighted centre of the deepest hollow."""
    from scipy import ndimage
    xs, ys, top, _ = heightmaps(mesh, step)
    _, _, hull, _ = heightmaps(mesh.convex_hull, step)
    dip = np.where(np.isnan(top) | np.isnan(hull), 0.0, hull - top)
    dip[~ndimage.binary_erosion(~np.isnan(top), iterations=int(3 / step))] = 0
    # a hollow, not the shadow of a rim on a dome: the surface curves up
    # (a bowl) there
    t = np.where(np.isnan(top), np.nanmin(top), top)
    bowl = ndimage.gaussian_laplace(t, 3.0 / step) > 0
    lbl, _ = ndimage.label((dip > 0.3 * dip.max()) & bowl)
    k = np.bincount(lbl.ravel(), weights=dip.ravel())[1:].argmax() + 1
    m = lbl == k
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    w = dip[m]
    bx, by = float((X[m] * w).sum() / w.sum()), float((Y[m] * w).sum() / w.sum())
    # on a sloping top the hollow's deepest part sits downhill of the
    # middle the designer drew: the middle is the centre of the rim, the
    # ring of sharpest convex curvature round the hollow (a circle fit)
    L = ndimage.gaussian_laplace(t, 1.0 / step)
    near = (np.hypot(X - bx, Y - by) < 16) & ndimage.binary_erosion(~np.isnan(top), iterations=int(3 / step))
    rim = near & (L < 0.4 * L[near].min())
    r = None
    if rim.sum() > 30:
        x, y = X[rim], Y[rim]
        c = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), x ** 2 + y ** 2, rcond=None)[0]
        cx, cy = c[0] / 2, c[1] / 2
        rr = float(np.sqrt(max(c[2] + cx ** 2 + cy ** 2, 0)))
        if 3 < rr < 16 and np.hypot(cx - bx, cy - by) < rr:
            bx, by, r = float(cx), float(cy), rr
    return bx, by, float(dip[m].max()), r


def find_door(xs, ys, bot, need=24.0, flat=1.0):
    """The largest centred-on-the-flat rectangle of the underside within
    `flat` mm of its lowest point."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    z0 = np.nanmin(bot)
    ok = ~np.isnan(bot) & (bot <= z0 + flat)
    cx, cy = float(X[ok].mean()), float(Y[ok].mean())
    best = None
    for a in np.arange(need, 80, 1.0):
        for b in np.arange(need, 60, 1.0):
            box = (np.abs(X - cx) <= a / 2) & (np.abs(Y - cy) <= b / 2)
            if (box & ~ok).any():
                break
            if best is None or a * b > best[0] * best[1]:
                best = (a, b)
    if best is None:
        raise SystemExit(f"layout: the underside has no flat {need:g} x {need:g} mm for the battery door")
    return cx, cy, best[0], best[1], z0


def room(sdf, lo, hi, step=0.5):
    """The worst room (mm) round a box, inside the skin: > 0 fits."""
    axes = [np.arange(lo[i], hi[i] + 1e-9, min(step, max(hi[i] - lo[i], 1e-3))) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return float(-(sdf(P)).max() - SKIN)


def main():
    import trimesh
    from loadpath.shapes import MeshShape

    ap = argparse.ArgumentParser()
    ap.add_argument("exterior")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    ext = trimesh.load(a.exterior, force="mesh")
    sdf = MeshShape(ext).sdf
    xs, ys, top, bot = heightmaps(ext)
    bx, by, depth, rim_r = find_button(ext)
    dx, dy, da, db, z0 = find_door(xs, ys, bot)
    # the door sits in the opening, flush with the lowest point; parts
    # ride on it and must clear the skin round the opening's rim
    da, db = da - 0.4, db - 0.4
    zb = z0 + DOOR_T + BOARD_T
    parts = {}
    parts["door"] = ([dx - da / 2, dy - db / 2, z0], [dx + da / 2, dy + db / 2, z0 + DOOR_T])
    # the board: the door less half a millimetre a side; it rides in the
    # opening, which takes the skin away round it
    parts["board"] = ([dx - da / 2 + 0.5, dy - db / 2 + 0.5, z0 + DOOR_T], [dx + da / 2 - 0.5, dy + db / 2 - 0.5, zb])
    sx, sy, sz = SWITCH["body"]
    # the switch under the button, kept on the board
    bxs = float(np.clip(bx, parts["board"][0][0] + sx / 2, parts["board"][1][0] - sx / 2))
    bys = float(np.clip(by, parts["board"][0][1] + sy / 2, parts["board"][1][1] - sy / 2))
    parts["switch-body"] = ([bxs - sx / 2, bys - sy / 2, zb], [bxs + sx / 2, bys + sy / 2, zb + sz])
    # the coin cell and module where the board has length left, away from the switch
    cw, cd, ch = COIN_HOLDER
    side = -1.0 if bxs > dx else 1.0
    bl, bh = parts["board"]
    cx = (bh[0] - cw / 2 - 0.5) if side > 0 else (bl[0] + cw / 2 + 0.5)
    parts["coin-cell"] = ([cx - cw / 2, dy - cd / 2, zb], [cx + cw / 2, dy + cd / 2, zb + ch])
    # the radio beside the switch, across the board from it
    mw, md, mh = MODULE
    my = bys + sy / 2 + 1.0 + md / 2
    if my + md / 2 > bh[1]:
        my = bys - sy / 2 - 1.0 - md / 2
    parts["radio"] = ([bxs - mw / 2, my - md / 2, zb], [bxs + mw / 2, my + md / 2, zb + mh])
    checks = {}
    for k, (lo, hi) in parts.items():
        if k in ("door", "board"):
            continue
        r = room(sdf, np.asarray(lo), np.asarray(hi))
        checks[k] = round(r, 2)
        if r < 0.2:
            raise SystemExit(f"layout: {k} does not fit: {-r + 0.2:.2f} mm into the skin or past it "
                             f"(box {np.round(lo, 1).tolist()} - {np.round(hi, 1).tolist()})")
    stem_top = zb + SWITCH["height"]
    under = float(-sdf(np.array([[bxs, bys, stem_top]]))[0]) - SKIN
    if under < 1.0:
        raise SystemExit(f"layout: the switch's stem comes within {under:.2f} mm of the skin under the button")

    import manifold3d as m3d

    def box(lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))

    def save(name, man):
        mm = man.to_mesh()
        tm = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts))
        p = os.path.join(a.out, "parts", name + ".stl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tm.export(p)
        return os.path.relpath(p, a.out)
    sw = box(*parts["switch-body"]) + m3d.Manifold.cylinder(SWITCH["height"] - sz, SWITCH["stem_r"],
                                                            SWITCH["stem_r"], 48).translate((bxs, bys, zb + sz))
    files = {k: save(k, box(*v)) for k, v in parts.items() if k != "switch-body"}
    files["switch"] = save("switch", sw)
    # the opening: the door's footprint up through the skin
    files["opening"] = save("opening", box([dx - da / 2 - 0.2, dy - db / 2 - 0.2, z0 - 1.0],
                                           [dx + da / 2 + 0.2, dy + db / 2 + 0.2, z0 + DOOR_T + 0.05]))
    # off the button: on the top, clear of the rim, where a pocket or a
    # grip squeezes (behind it, and either side of it)
    R = (rim_r or 10.0) + 4.0
    off_button = []
    for x, y in ((bx - R - 4.0, by), (bx - R - 22.0, by), (bx, by + R), (bx, by - R)):
        # only where the top is still the top (not the rounded flank)
        hit = ext.ray.intersects_location([[x, y, ext.bounds[1][2] + 5]], [[0, 0, -1]], multiple_hits=False)[0]
        if len(hit) and hit[0][2] > ext.bounds[0][2] + 0.6 * ext.extents[2]:
            off_button.append((x, y))
    T = SWITCH["trigger_gf"]
    brief = {
        "name": "shutter-remote",
        "description": "Bluetooth shutter remote: dished button on top, coin cell, door underneath (layout.py)",
        "exterior": os.path.relpath(a.exterior, a.out), "skin": SKIN, "openings": [files["opening"]],
        "material": "tpu88a", "process": "sls", "minMember": 1.0, "restDepth": 1.5,
        # Quick mode's usual 6000 cells put this 1.0 mm skin on 1.8 mm cells,
        # where the design read the pocket press behind the button 26% stiff:
        # it met 1200 gf on the design and the delivered part buckled there
        # (680 gf, 7.9 mm of travel). On 1.3 mm cells, the full run's grid, the
        # optimizer holds that press itself (1243 gf delivered, no buckling).
        "expect": {"coarse": {"maxElements": 15000}},
        "components": [
            {"id": "door", "shape": files["door"], "external": True},
            {"id": "board", "shape": files["board"], "installedOn": "door"},
            {"id": "switch", "shape": files["switch"], "role": "switch", "installedOn": "board"},
            {"id": "coin-cell", "shape": files["coin-cell"], "installedOn": "board"},
            {"id": "radio", "shape": files["radio"], "installedOn": "board"}],
        "assembly": ["door"],
        "switch": {"unit": "gf", "trigger": T, "travel": SWITCH["travel"], "gap": 0.4,
                   "presses": [{"at": [round(bx, 2), round(by, 2)], "min": 200, "max": 450}] +
                   [{"at": [round(bx + 5 * np.cos(t), 2), round(by + 5 * np.sin(t), 2)], "min": 200, "max": 450}
                    for t in np.linspace(0, 2 * np.pi, 4, endpoint=False)] +
                   [{"at": [round(x, 1), round(y, 1)], "min": 1200}
                    for x, y in off_button]},
    }
    with open(os.path.join(a.out, "brief.json"), "w") as f:
        json.dump(brief, f, indent=1)
    lay = {"button": {"at": [round(bx, 2), round(by, 2)], "dishDepth": round(depth, 2),
                      "rimRadius": None if rim_r is None else round(rim_r, 2)},
           "door": {"center": [round(dx, 2), round(dy, 2)], "size": [round(da, 1), round(db, 1)], "z": round(z0, 2)},
           "room": checks, "stemToSkin": round(under, 2),
           "parts": {k: [np.round(v[0], 2).tolist(), np.round(v[1], 2).tolist()] for k, v in parts.items()}}
    with open(os.path.join(a.out, "layout.json"), "w") as f:
        json.dump(lay, f, indent=1)
    print(json.dumps(lay))


if __name__ == "__main__":
    main()
