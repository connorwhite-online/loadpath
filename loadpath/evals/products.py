"""
Generic products for the enclosure evals: a designer's exterior and a
plain-language parts list each, written out as a `loadpath enclosure` brief
(enclosure.py). None of them is a real product; each one stands for a kind
of thing a designer hands over:

- **remote**: a TV-style remote in rigid PA12 (SLS). A long thin bar with
  an IR window in its nose and a battery door underneath; the board is
  longer than any opening, so the shell splits into a top and a bottom
  (shells.py). Buttons pressed hard, drops on every face, a squeeze.
- **band-pod**: the pod of a fitness band, one piece of SLS TPU 88A with a
  window underneath that the rigid sensor back closes. Soft to pinch, and
  the battery cushioned in a drop (feel.py).
- **speaker**: a portable speaker in PETG (FDM), from a generated mesh
  (unit scale, Y-up, holes, a doubled shell, debris: the exterior intake
  wraps it). Everything goes in through the grille opening on top; the amp
  makes heat, so the interior is designed to carry it out (cooling.py).
- **tool-grip**: the body of a cordless rotary tool, a barrel and a
  handle, from a generated mesh, in PA12 (MJF). The motor's shaft out
  through the nose, the cell in the handle, a trigger switch under the
  finger; a clamshell split down the middle.
- **smart-button**: a smart-home button, one piece of SLS TPU 88A over a
  rigid back. A tactile switch under the middle of the dome; the dome
  clicks at a set force anywhere on it (switch.py through handoff).
- **desk-hub**: a desk USB hub in rigid PA12 (SLS), one piece with a
  service door underneath. The board is longer than the door and its
  connector has to reach out through a port in the end wall: there is no
  straight way in, only nose first at a tilt and then laid flat
  (motion.py).

    from loadpath.evals.products import write
    brief_path = write("remote", "out/remote")

Every exterior is the designer's: nothing here or downstream edits it.
"""
from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["PRODUCTS", "write"]


# ---------------------------------------------------------------- helpers

def _tm(man):
    import trimesh
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))


def _stl(man, out, name):
    _tm(man).export(os.path.join(out, name))
    return name


def _rounded_box(size, r, seg=40):
    import manifold3d as m3d
    sx, sy, sz = [s / 2 - r for s in size]
    return m3d.Manifold.batch_hull([m3d.Manifold.sphere(r, seg).translate((x, y, z))
                                    for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])


def _box(lo, hi):
    import manifold3d as m3d
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def _cyl(p0, axis, length, r, seg=48):
    """A cylinder from p0 along a signed axis ('+x', '-z', ...)."""
    import manifold3d as m3d
    c = m3d.Manifold.cylinder(length, r, r, seg)
    a = axis.lstrip("+-")
    sgn = -1.0 if axis.startswith("-") else 1.0
    if a == "x":
        c = c.rotate((0, 90 * sgn, 0))
    elif a == "y":
        c = c.rotate((-90 * sgn, 0, 0))
    elif sgn < 0:
        c = c.rotate((180, 0, 0))
    return c.translate(tuple(p0))


def _generated(clean, out, name, seed=0, holes=((0.0, 0.0, 0.0, 0.0),)):
    """The clean exterior as a generator would hand it back (see
    docs/tutorials/generated-exterior/generated_standin.py): open (holes
    torn at `holes`, each (x, y, z, r) snapped to the surface), a duplicate
    inner shell 0.6 mm in, a patch doubled proud of the surface, a patch of
    flipped triangles, vertex jitter, floating debris; unit scale, Y-up, as
    GLB. Returns (file name, the longest extent in mm), so the brief reads
    it back with {"path", "scale", "up": "+y", "place": "keep"}."""
    import trimesh
    m = _tm(clean)
    # a generator's mesh is dense and even, not a CAD tessellation
    m = m.subdivide_to_size(1.2, max_iter=8)
    rng = np.random.default_rng(seed)
    F = m.faces.copy()
    c = m.triangles_center
    keep = np.ones(len(F), bool)
    for x, y, z, r in holes:
        if r <= 0:
            continue
        at = m.nearest.on_surface([[x, y, z]])[0][0]
        keep &= np.linalg.norm(c - at, axis=1) > r
    k0 = int(rng.integers(len(F)))
    flip = np.linalg.norm(c - c[k0], axis=1) < 4.0
    F[flip] = F[flip][:, ::-1]
    parts = [trimesh.Trimesh(m.vertices.copy(), F[keep], process=False)]
    inner = m.copy()
    inner.vertices = inner.vertices - 0.6 * inner.vertex_normals
    parts.append(inner)
    k1 = int(rng.integers(len(F)))
    sel = np.linalg.norm(c - c[k1], axis=1) < 5.0
    patch = m.submesh([np.nonzero(sel)[0]], append=True)
    patch.vertices = patch.vertices + 0.3 * patch.vertex_normals * np.linspace(-1, 1, len(patch.vertices))[:, None]
    parts.append(patch)
    lo, hi = m.bounds
    for _ in range(4):
        s = trimesh.creation.icosphere(1, radius=float(rng.uniform(0.3, 0.8)))
        s.vertices += np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]), hi[2] + rng.uniform(2, 5)])
        parts.append(s)
    g = trimesh.util.concatenate(parts)
    g.vertices = g.vertices + rng.normal(0, 0.03, g.vertices.shape)
    L = float(np.ptp(m.vertices, axis=0).max())
    g.vertices = g.vertices / L
    # z up -> y up (glTF): the intake's up="+y" turns it back
    g.vertices = g.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T
    g.export(os.path.join(out, name))
    return name, L


# ---------------------------------------------------------------- products

def remote(out):
    """A TV-style remote, 150 x 44 x 22 mm, rigid PA12 (SLS)."""
    L, W, H = 150.0, 44.0, 22.0
    ext = _rounded_box((L, W, H), 7.0)
    # the IR window high in the nose (the cells go under the board); the
    # battery door under the back half
    win = _box((L / 2 - 4.0, -6.0, 3.5), (L / 2 + 2.0, 6.0, 7.5))
    door = _box((-62.0, -14.0, -H / 2 - 2.0), (-6.0, 14.0, -H / 2 + 1.2))
    f = {"exterior": _stl(ext - win - door, out, "exterior.stl"),
         "window": _stl(win, out, "ir-window.stl"), "door": _stl(door, out, "battery-door.stl"),
         "lid": _stl(_box((-61.8, -13.8, -H / 2), (-6.2, 13.8, -H / 2 + 1.2)), out, "door.stl")}
    # the board with its IR LED at one end and the button domes on top
    board = _box((-55.0, -17.0, -0.8), (55.0, 17.0, 0.8)) + _box((-20.0, -14.0, 0.8), (40.0, 14.0, 2.4))
    led = _cyl((55.0, 0.0, 1.6), "+x", 6.0, 2.5)
    f["board"] = _stl(board, out, "board.stl")
    f["led"] = _stl(led, out, "ir-led.stl")
    return {
        "name": "remote",
        "description": "A TV-style remote in rigid PA12 (SLS): an IR window in the nose, a battery door "
                       "underneath, a clamshell split. Parts: a 110 x 34 mm board with button domes and an IR LED "
                       "at its end (looking out the window), a two-AAA battery holder over the door. Drops on "
                       "every face, a hard squeeze, a thumb on the buttons.",
        "exterior": f["exterior"], "skin": 1.4,
        "openings": [{"id": "ir-window", "shape": f["window"], "inward": "-x"},
                     {"id": "battery-door", "shape": f["door"], "inward": "+z"}],
        "components": [
            {"id": "board", "shape": f["board"], "mass": 0.012, "rotate": "none", "against": "+z", "margin": 0.3},
            {"id": "led", "shape": f["led"], "installedOn": "board", "mass": 0.0005,
             "alignTo": {"opening": "ir-window", "face": "+x", "recess": 0.5}},
            # over the door (its face 1.7 mm in: the door's 1.2 mm and a gap)
            {"id": "batteries", "size": [50.0, 22.0, 10.5], "mass": 0.024, "rotate": "z",
             "alignTo": {"opening": "battery-door", "face": "-z", "recess": 1.7}},
            {"id": "door", "shape": f["lid"], "place": "given", "external": True, "mount": False}],
        "assembly": ["board", "batteries", "door"],
        "split": {"normal": "z", "search": "flat"},
        "rest": "-z",
        "loads": {"drop": {"g": 300, "faces": ["-z", "+z", "-x", "+x", "-y", "+y"]},
                  "squeeze": {"axis": "y", "force": 120, "radius": 10},
                  "press": [{"name": "thumb", "face": "+z", "at": [20.0, 0.0], "force": 30, "radius": 6}]},
        "material": "pa12", "process": "sls", "minMember": 1.2,
        "goal": {"safetyFactor": 1.5},
        "resolution": {"pitch": 0.7},
    }


def band_pod(out):
    """A fitness-band pod, 42 x 30 x 15 mm, SLS TPU 88A in one piece."""
    L, W, H = 42.0, 30.0, 15.0
    ext = _rounded_box((L, W, H), 6.0)
    win = _box((-15.0, -10.0, -H / 2 - 2.0), (15.0, 10.0, -H / 2 + 1.0))
    back = _box((-14.8, -9.8, -H / 2), (14.8, 9.8, -H / 2 + 1.0))
    f = {"exterior": _stl(ext - win, out, "exterior.stl"), "window": _stl(win, out, "sensor-window.stl"),
         "back": _stl(back, out, "sensor-back.stl")}
    return {
        "name": "band-pod",
        "description": "The pod of a fitness band, one piece of SLS TPU 88A; a window underneath that the "
                       "rigid sensor back closes. Parts: a 26 x 18 mm board, a 22 x 16 x 4 mm cell, a coin "
                       "vibration motor. Soft to pinch (10 N gives 0.2-0.8 mm), and the cell cushioned in a "
                       "1.2 m fall onto its face.",
        "exterior": f["exterior"], "skin": 1.0,
        "openings": [{"id": "sensor-window", "shape": f["window"], "inward": "+z"}],
        "components": [
            {"id": "board", "size": [26.0, 18.0, 1.2], "mass": 0.002, "against": "-z", "rotate": "z"},
            # 1.8 mm of room round the cell: the travel its cushioning
            # needs; held from the side (its top lands first, and it comes
            # in from below)
            {"id": "cell", "size": [22.0, 16.0, 4.0], "mass": 0.006, "against": "+z", "rotate": "z",
             "margin": 1.8, "mount": {"side": "+x"}},
            {"id": "motor", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 2.7, "r": 4.0},
             "mass": 0.001, "near": ["board"], "rotate": "none"},
            {"id": "back", "shape": f["back"], "place": "given", "external": True, "mount": False}],
        "assembly": ["cell", "motor", "board", "back"],
        "rest": "-z",
        "feel": {
            "targets": [
                {"kind": "squeeze", "name": "pinch", "direction": "-y", "force": 10, "travel": [0.2, 0.8],
                 "hold": {"face": "-y"}, "press": {"face": "+y"}},
                {"kind": "drop", "name": "face-drop", "lands": {"face": "+z"}, "mass": {"region": "mount-cell"},
                 "kg": 0.006, "height": 1.2, "maxG": 2000, "maxTravel": 1.6},
            ]},
        "material": "tpu88a", "process": "sls", "minMember": 1.0,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": 0.8},
    }


def speaker(out):
    """A portable speaker, 72 mm across and 96 mm tall, PETG (FDM), from a
    generated mesh."""
    import manifold3d as m3d
    R, H = 36.0, 96.0
    body = m3d.Manifold.batch_hull([
        m3d.Manifold.cylinder(H - 16.0, R, R, 72).translate((0, 0, -H / 2 + 8.0)),
        m3d.Manifold.cylinder(H, R - 8.0, R - 8.0, 72).translate((0, 0, -H / 2))])
    grille = m3d.Manifold.cylinder(4.0, 25.0, 25.0, 72).translate((0, 0, H / 2 - 2.5))
    port = _box((-6.0, -R - 2.0, -H / 2 + 12.0), (6.0, -R + 3.0, -H / 2 + 17.0))
    clean = body - grille - port
    ext, Ls = _generated(clean, out, "speaker-generated.glb", seed=3,
                         holes=((20.0, 25.0, 0.0, 2.5), (-30.0, -10.0, -20.0, 1.5)))
    cover = m3d.Manifold.cylinder(1.5, 24.8, 24.8, 72).translate((0, 0, H / 2 - 1.5))
    f = {"grille": _stl(grille, out, "grille.stl"), "port": _stl(port, out, "usb-port.stl"),
         "cover": _stl(cover, out, "grille-cover.stl")}
    driver = _cyl((0, 0, 0), "+z", 4.0, 22.0) + _cyl((0, 0, -14.0), "+z", 14.0, 12.0)
    amp = _box((-16.0, -13.0, -0.8), (16.0, 13.0, 0.8)) + _box((-6.0, -6.0, 0.8), (6.0, 6.0, 2.2))
    usbc = _box((-4.5, -13.0 - 2.6, 0.8), (4.5, -13.0 + 4.9, 4.0))
    f["driver"] = _stl(driver, out, "driver.stl")
    f["amp"] = _stl(amp, out, "amp.stl")
    f["usbc"] = _stl(usbc, out, "usbc.stl")
    return {
        "name": "speaker",
        "description": "A portable speaker in PETG (FDM) from a generated mesh: a 50 mm grille opening on top "
                       "(a cover closes it) and a USB-C port low on the back. Parts: a 44 mm driver facing up "
                       "out the grille, an 18650 cell, a 32 x 26 mm amp board losing 1 W (a class-D amp at "
                       "about 5 W out), with its USB-C receptacle in the port. Drops on the base and sides; the amp under 85 C and the shell "
                       "cool enough to hold.",
        "exterior": {"path": ext, "scale": Ls, "up": "+y", "place": "keep"}, "skin": 1.6,
        "openings": [{"id": "grille", "shape": f["grille"], "inward": "-z"},
                     {"id": "usb-port", "shape": f["port"], "inward": "+y"}],
        "components": [
            {"id": "driver", "shape": f["driver"], "mass": 0.055, "rotate": "none", "watts": 0.5,
             "alignTo": {"opening": "grille", "face": "+z", "recess": 1.0}},
            {"id": "cell", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 65.0, "r": 9.2},
             "mass": 0.046, "rotate": "none", "maxTemp": 45.0},
            {"id": "amp", "shape": f["amp"], "mass": 0.010, "watts": 1.0, "maxTemp": 85.0, "rotate": "x",
             "margin": 0.3},
            {"id": "usbc", "shape": f["usbc"], "installedOn": "amp", "mass": 0.001,
             "alignTo": {"opening": "usb-port", "face": "-y", "recess": 0.3}},
            {"id": "cover", "shape": f["cover"], "place": "given", "external": True, "mount": False}],
        "assembly": ["amp", "cell", "driver", "cover"],
        "rest": "-z",
        "loads": {"drop": {"g": 150, "faces": ["-z", "-x", "+x", "-y", "+y"]}},
        "cooling": {"ambient": 25.0, "skin": {"contact": "held"}},
        "material": "petg", "process": "fdm", "printDirection": "+z", "minMember": 1.2,
        "goal": {"safetyFactor": 1.5},
        "resolution": {"pitch": 1.0},
    }


def tool_grip(out):
    """A cordless rotary tool's body: a 34 mm barrel 130 mm long and a
    handle under it, PA12 (MJF), from a generated mesh."""
    import manifold3d as m3d
    Rb, Lb = 17.0, 130.0
    barrel = m3d.Manifold.batch_hull([m3d.Manifold.sphere(Rb, 48).translate((x, 0, 0))
                                      for x in (-Lb / 2 + Rb, Lb / 2 - Rb - 6.0)] +
                                     [_cyl((Lb / 2 - 8.0, 0, 0), "+x", 8.0, 10.0)])
    handle = m3d.Manifold.batch_hull([m3d.Manifold.sphere(15.0, 40).translate((-36.0, 0, -4.0)),
                                      m3d.Manifold.sphere(15.5, 40).translate((-39.0, 0, -84.0))])
    nose = _cyl((Lb / 2 - 3.0, 0, 0), "+x", 5.0, 5.0)
    plug = _box((-45.0, -6.0, -103.0), (-33.0, 6.0, -97.8))
    clean = (barrel + handle) - nose - plug
    ext, Ls = _generated(clean, out, "tool-generated.glb", seed=7,
                         holes=((0.0, 17.0, 0.0, 2.0), (-40.0, -14.0, -60.0, 3.0)))
    f = {"nose": _stl(nose, out, "nose.stl"), "plug": _stl(plug, out, "charge-port.stl")}
    motor = _cyl((-19.0, 0, 0), "+x", 38.0, 13.5) + _cyl((19.0, 0, 0), "+x", 12.0, 2.5)
    f["motor"] = _stl(motor, out, "motor.stl")
    return {
        "name": "tool-grip",
        "description": "The body of a cordless rotary tool in PA12 (MJF) from a generated mesh: a barrel with "
                       "the shaft out through its nose, a handle with a charge port at its foot, a clamshell "
                       "split. Parts: a 27 mm motor, an 18650 cell in the handle, a trigger switch where the "
                       "finger falls, a 40 x 16 mm controller board. Drops on every side, a hard grip.",
        "exterior": {"path": ext, "scale": Ls, "up": "+y", "place": "keep"}, "skin": 1.5,
        "openings": [{"id": "nose", "shape": f["nose"], "inward": "-x"},
                     {"id": "charge-port", "shape": f["plug"], "inward": "+z"}],
        "components": [
            {"id": "motor", "shape": f["motor"], "mass": 0.095, "rotate": "none",
             "alignTo": {"opening": "nose", "face": "+x", "recess": 0.5}},
            {"id": "cell", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 65.0, "r": 9.2},
             "mass": 0.046, "rotate": "none", "against": "-z"},
            # where the index finger falls: on the front of the handle
            # just under the barrel, pressed toward the palm
            {"id": "trigger", "size": [7.0, 8.0, 12.0], "mass": 0.004, "rotate": "none",
             "under": {"at": [-21.0, 0.0, -21.0], "direction": "-x", "gap": 1.0}},
            {"id": "board", "size": [40.0, 1.6, 16.0], "mass": 0.006, "rotate": "none", "near": ["motor", "cell"]}],
        "assembly": ["cell", "board", "trigger", "motor"],
        "split": {"normal": "y", "at": 0.0},
        "rest": "-y",
        "loads": {"drop": {"g": 200, "faces": ["-z", "-x", "+x", "-y", "+y"]},
                  "squeeze": {"axis": "y", "force": 250, "radius": 10, "at": [-37.5, 0.0, -55.0]}},
        "material": "pa12", "process": "mjf", "minMember": 1.2,
        "goal": {"safetyFactor": 1.5},
        "resolution": {"pitch": 1.0},
    }


def smart_button(out):
    """A smart-home button, 56 mm across and 17 mm tall, SLS TPU 88A in one
    piece over a rigid back."""
    import manifold3d as m3d
    R, H = 28.0, 17.0
    dome = m3d.Manifold.batch_hull([
        m3d.Manifold.cylinder(6.0, R, R, 96),
        m3d.Manifold.cylinder(H - 6.0, R - 2.0, R - 8.0, 96).translate((0, 0, 6.0))])
    win = m3d.Manifold.cylinder(3.0, 22.0, 22.0, 96).translate((0, 0, -2.0))
    back = m3d.Manifold.cylinder(2.0, 21.8, 21.8, 96)
    f = {"exterior": _stl(dome - win, out, "exterior.stl"), "window": _stl(win, out, "back-opening.stl"),
         "back": _stl(back, out, "back.stl")}
    # the switch sits on the back plate (a 6 x 6 mm tactile switch, 3.5 mm
    # tall with its stem), the coin cell beside it
    sw = _box((-3.0, -3.0, 2.0), (3.0, 3.0, 5.5)) + _cyl((0, 0, 5.5), "+z", 1.5, 1.7)
    f["switch"] = _stl(sw, out, "switch.stl")
    board = _box((-12.0, -12.0, 2.0), (12.0, 12.0, 3.2))
    f["board"] = _stl(board, out, "board.stl")
    return {
        "name": "smart-button",
        "description": "A smart-home button, one piece of SLS TPU 88A over a rigid back plate. Parts: a 6 x 6 mm "
                       "tactile switch on the back plate under the middle of the dome, a 24 mm board, a CR2032 "
                       "cell. The dome clicks at 150-400 gf at the middle and anywhere within 9 mm of it.",
        "exterior": f["exterior"], "skin": 1.0,
        "openings": [{"id": "back-opening", "shape": f["window"], "inward": "+z"}],
        "components": [
            {"id": "back", "shape": f["back"], "place": "given", "role": "seat", "external": True,
             "mount": False},
            {"id": "switch", "shape": f["switch"], "place": "given", "role": "switch", "installedOn": "back"},
            {"id": "board", "shape": f["board"], "place": "given", "mass": 0.003, "installedOn": "back"},
            {"id": "cell", "shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 3.2, "r": 10.0},
             "mass": 0.003, "against": "-z", "rotate": "none"}],
        "assembly": ["cell", "back"],
        "rest": "-z",
        "switch": {"unit": "gf", "trigger": 160, "travel": 0.25, "gap": 0.4,
                   "presses": [{"at": [0, 0], "min": 150, "max": 400},
                               {"at": [9, 0], "min": 150, "max": 400}, {"at": [-9, 0], "min": 150, "max": 400},
                               {"at": [0, 9], "min": 150, "max": 400}, {"at": [0, -9], "min": 150, "max": 400}]},
        "material": "tpu88a", "process": "sls", "minMember": 1.0,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": 1.0},
    }


def desk_hub(out):
    """A desk USB hub, 70 x 34 x 18 mm, rigid PA12 (SLS) in one piece. The
    board is longer than the service door under it, and its connector has
    to end up out through the port in the end wall: no straight way in.
    By hand it goes nose first into the port at a tilt, then is laid flat."""
    L, W, H = 70.0, 34.0, 18.0
    ext = _rounded_box((L, W, H), 3.0)
    port = _box((-L / 2 - 1.0, -6.5, -4.5), (-L / 2 + 2.0, 6.5, 1.5))
    door = _box((-18.0, -13.5, -H / 2 - 2.0), (31.0, 13.5, -H / 2 + 1.2))
    f = {"exterior": _stl(ext - port - door, out, "exterior.stl"), "port": _stl(port, out, "port.stl"),
         "door": _stl(door, out, "service-door.stl"),
         "foot": _stl(_box((-17.8, -13.3, -H / 2), (30.8, 13.3, -H / 2 + 1.2)), out, "foot.stl")}
    # the board, and the connector hung under its end (3 mm past the edge)
    f["board"] = _stl(_box((-27.0, -11.0, 0.0), (27.0, 11.0, 1.6)), out, "board.stl")
    f["connector"] = _stl(_box((-30.0, -5.0, -3.6), (-21.0, 5.0, 0.0)), out, "connector.stl")
    return {
        "name": "desk-hub",
        "description": "A desk USB hub in rigid PA12 (SLS), one piece: a port in the end wall and a service "
                       "door underneath that a foot plate closes. Parts: a 54 x 22 mm board with a connector "
                       "under its end, out through the port. The board is 5 mm longer than the door, so it "
                       "goes in nose first at a tilt and is laid flat. A drop on its base and a squeeze.",
        "exterior": f["exterior"], "skin": 1.2,
        "openings": [{"id": "port", "shape": f["port"], "inward": "+x"},
                     {"id": "service-door", "shape": f["door"], "inward": "+z"}],
        "components": [
            {"id": "board", "shape": f["board"], "mass": 0.014, "rotate": "none", "watts": 0.4},
            {"id": "connector", "shape": f["connector"], "installedOn": "board", "mass": 0.002,
             "alignTo": {"opening": "port", "face": "-x", "recess": 0.4}},
            {"id": "foot", "shape": f["foot"], "place": "given", "external": True, "mount": False}],
        "assembly": ["board", "foot"],
        "tilt": 45,
        # held by its top for the analysis: the door takes the middle of its base
        "rest": "+z",
        "loads": {"drop": {"g": 200, "faces": ["-z", "+z"]},
                  "squeeze": {"axis": "y", "force": 80, "radius": 8}},
        "material": "pa12", "process": "sls", "minMember": 1.2,
        "goal": {"safetyFactor": 1.5},
        "resolution": {"pitch": 0.8},
    }


PRODUCTS = {"remote": remote, "band-pod": band_pod, "speaker": speaker, "tool-grip": tool_grip,
            "smart-button": smart_button, "desk-hub": desk_hub}


def write(name: str, out: str) -> str:
    """Write product `name`'s exterior, parts and brief into `out`; the
    brief's path."""
    os.makedirs(out, exist_ok=True)
    brief = PRODUCTS[name](out)
    path = os.path.join(out, f"{name}.brief.json")
    with open(path, "w") as fh:
        json.dump(brief, fh, indent=1)
    return path
