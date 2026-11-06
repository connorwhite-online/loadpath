"""
Physical calibration: coupons to print, a measurement format, and a fit that
turns measured numbers into a calibrated material card.

Every number loadpath reports rests on a datasheet: Young's modulus,
strength, density, and the process's across-layer factors (spec.PROCESSES).
A printer, a filament and a slicer profile move all of them, so this closes
the loop with a handful of coupons that need no lab: a kitchen scale,
weights you can hang (water bottles), a ruler or a dial indicator, and a
luggage scale for the break tests.

    kit = coupon_kit("petg")            # meshes + predictions per coupon
    write_kit(kit, "calibration/")      # STLs, predictions.json, measurements template
    card = fit(json.load(open("measurements.json")))   # calibrated material card

The coupons (each STL sits on the bed in its print orientation):

  F-XY  cantilever printed flat: bending stiffness along the layers -> E
  F-Z   the same printed standing: bending across the layers -> E_z / E
  T-XY  dog-bone printed flat, pulled to break -> tensile strength
  T-Z   the same printed standing -> strength across layers / in-plane
        (the process's `interlayer`)
  L-GY  cantilever with a gyroid sheet core -> how a printed lattice compares
        with the homogenized model lattice.py uses (SHEETS)
  H-XY  a plate with a row of 12 numbered holes drawn 0.05 mm apart, printed
        flat (holes along the build), and a drill bit of known size: the
        first hole the bit's shank passes reads the printer's hole bias
        (fits.py) to +-0.025 mm with no calipers -> process.holeBias
  H-Z   the same ladder printed on its long edge (holes across the layers,
        where FDM hole tops sag) -> process.holeBiasAcross
  H-PEG pegs drawn at H-XY's first four sizes: the smallest hole each goes
        into reads the clearance a printed pin needs in a printed hole
  X-20, X-30 ... (with crush=True) gyroid crush blocks at two densities,
        squashed under growing weight -> the crush plateau impact.py
        predicts (CRUSH), its density exponent, and the densification
        strain. PA12 needs a 10:1 lever (a board and a bucket of water);
        TPU takes weights on a plate.

Every coupon's mass gives the as-printed density. A part test (a finished
part on a fixture with a known weight) compares measured deflection with
verify's prediction at the calibrated modulus: the model's own error
(voxel stiffness, bonded joints, slicer fill) left after the material is
right. Cards saved in loadpath/calibrations/ are usable by name as a
study's material (materials.material_from_spec), and carry their
process factors into the study (spec.study_from_dict).

The fit uses closed-form beam theory, not the voxel model, so the material
card is measured independently of the solver it then feeds.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .materials import Material, material_from_spec

__all__ = ["Coupon", "coupon_kit", "crush_coupons", "creep_coupons", "fit_creep", "hole_coupons", "fit_holes",
           "write_kit", "fit", "fit_crush", "measurement_template",
           "part_prediction", "rigid_motion", "CALIBRATION_DIR", "load_card", "list_cards", "crush_factor"]

G = 9.81                                  # N per kg
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations")

# cantilever: clamp block, then the beam; the load hangs from a hole near the
# tip, `span` mm from the block's face (the root)
_CANT = {"block": (30.0, 12.0, 10.0), "width": 12.0, "thickness": 5.0, "span": 80.0, "tip": 6.0, "hole": 4.5}
# dog-bone: tabs with an 8 mm pin hole, a smooth taper, a 2 x 3 mm gauge
_BONE = {"tab": (20.0, 18.0, 8.0), "taper": 16.0, "gauge": (2.0, 3.0, 24.0), "hole": 8.0}
# lattice cantilever: gyroid sheet core between a solid clamp block and a
# solid end that carries the hook hole
_LAT = {"block": 30.0, "width": 20.0, "height": 12.0, "core": 80.0, "end": 12.0, "cell": 10.0, "wall": 1.0,
        "hole": 4.5, "kind": "gyroid"}
_LOADS_KG = (0.2, 0.5, 1.0)
# crush blocks: n x n cells in plan, m cells tall, between solid plates;
# the cell is set so the thinnest wall (at the lower density) is the
# process minimum
_CRUSH = {"densities": (0.2, 0.3), "plan": 2, "tall": 2, "plate": 2.0, "kind": "gyroid", "lever": 15.0}
_CRUSH_ELASTIC = {"densities": (0.25, 0.4), "plan": 2, "tall": 2, "plate": 1.5, "kind": "gyroid", "lever": 1.0}
# hole-fit coupons: a plate with a row of holes drawn 0.05 mm apart round a
# gauge of known size (a drill bit's smooth shank), one printed flat (holes
# along the build) and one on its edge (holes across the layers), and a strip
# of pegs drawn at the plate's first sizes. Go/no-go: no calipers needed.
_HOLE = {"gauge": 5.0, "holes": 12, "step": 0.05, "below": 0.25, "pitch": 9.0, "width": 15.0,
         "thickness": 5.0, "margin": 6.0, "holeY": 10.0, "digit": 4.0, "seg": 0.8, "depth": 0.8,
         "digitY": 3.5, "chamfer": 0.4, "pegs": 4, "pegHeight": 8.0, "pegBase": 3.0}
# elastomer creep block: the low-density crush block, 3 cells tall, held at
# half its buckling load for a week (viscoelastic.py's creep law)
_CREEP = {"rho": 0.25, "tall": 3, "loadShare": 0.5, "readMinutes": (1, 10, 60, 600, 1440, 4320, 10080)}


@dataclass
class Coupon:
    id: str
    title: str
    mesh: object                         # trimesh, on the bed (z >= 0) in print orientation
    test: str                            # what to do with it, one paragraph
    measure: dict                        # template of what to write down
    predicted: dict                      # at the starting material card
    geometry: dict = field(default_factory=dict)

    def to_dict(self):
        return {"id": self.id, "title": self.title, "test": self.test, "measure": self.measure,
                "predicted": self.predicted, "geometry": self.geometry}


# ---------------------------------------------------------------- geometry

def _box(lo, hi):
    import manifold3d as m3d
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3d.Manifold.cube(list(hi - lo)).translate(list(lo))


def _cyl_z(x, y, z0, z1, r, n=48):
    import manifold3d as m3d
    return m3d.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def _to_trimesh(man):
    from .shapes import trimesh_from_manifold
    m = trimesh_from_manifold(man)
    m.merge_vertices()
    return m


def _rotate_upright(man):
    """Stand a part drawn along +x on its x = 0 end: x -> z."""
    # rotation about y by -90 degrees takes +x to +z and +z to -x
    man = man.rotate([0.0, -90.0, 0.0])
    lo = np.asarray(man.bounding_box()[:3])
    return man.translate(list(-lo))


def _cantilever_manifold(c=_CANT):
    bx, by, bz = c["block"]
    end = bx + c["span"] + c["tip"]
    body = _box([0, 0, 0], [bx, by, bz]) + _box([bx, 0, 0], [end, c["width"], c["thickness"]])
    return body - _cyl_z(bx + c["span"], c["width"] / 2, -1, c["thickness"] + 1, c["hole"] / 2)


def _bone_outline(b=_BONE, n=24):
    """Half-outline (u along the pull, v across) of the dog-bone, as a closed
    polygon, centred on v = 0, from u = 0."""
    tw, tl, _ = b["tab"]
    gw, _, gl = b["gauge"]
    tp = b["taper"]
    u_pts, v_pts = [0.0, tl], [tw / 2, tw / 2]
    s = np.linspace(0, 1, n)[1:]
    # cosine taper: zero slope at both ends, no sharp corner at the gauge
    for si in s:
        u_pts.append(tl + tp * si)
        v_pts.append(gw / 2 + (tw - gw) / 2 * 0.5 * (1 + np.cos(np.pi * si)))
    u_pts.append(tl + tp + gl)
    v_pts.append(gw / 2)
    for si in s:
        u_pts.append(tl + tp + gl + tp * si)
        v_pts.append(gw / 2 + (tw - gw) / 2 * 0.5 * (1 - np.cos(np.pi * si)))
    total = 2 * tl + 2 * tp + gl
    u_pts += [total]
    v_pts += [tw / 2]
    top = list(zip(u_pts, v_pts))
    bottom = [(u, -v) for u, v in reversed(top)]
    # counter-clockwise, as manifold3d wants a filled outline
    return list(reversed(top + bottom)), total


def _teardrop(r, n=40):
    """A hole that prints without support with its axis horizontal: a circle
    with a 45 degree roof pointing to +u (up, once the coupon stands)."""
    a = np.linspace(0.25 * np.pi, 1.75 * np.pi, n)
    pts = [(r * np.cos(t), r * np.sin(t)) for t in a]
    return [(r * np.sqrt(2.0), 0.0)] + pts


def _bone_manifold(b=_BONE):
    import manifold3d as m3d
    outline, total = _bone_outline(b)
    tw, tl, tt = b["tab"]
    gt = b["gauge"][1]
    plate = m3d.CrossSection([outline]).extrude(gt).translate([0, 0, (tt - gt) / 2])
    tabs = _box([0, -tw / 2, 0], [tl, tw / 2, tt]) + _box([total - tl, -tw / 2, 0], [total, tw / 2, tt])
    body = plate + tabs
    r = b["hole"] / 2
    for u0 in (tl / 2, total - tl / 2):
        drop = [(u0 + u, v) for u, v in _teardrop(r)]
        body = body - m3d.CrossSection([drop]).extrude(tt + 2).translate([0, 0, -1])
    lo = np.asarray(body.bounding_box()[:3])
    return body.translate(list(-lo)), total


def _lattice_sdf(c=_LAT):
    from .lattice import tpms

    L = c["block"] + c["core"] + c["end"]
    w, h = c["width"], c["height"]

    def box_sdf(p, lo, hi):
        lo, hi = np.asarray(lo), np.asarray(hi)
        ctr, half = (lo + hi) / 2, (hi - lo) / 2
        q = np.abs(p - ctr) - half
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)

    x0, x1 = c["block"], c["block"] + c["core"]
    hole = (x1 + c["end"] / 2, w / 2)

    def sdf(p):
        solid = np.minimum(box_sdf(p, [0, 0, 0], [x0 + 0.5, w, h]), box_sdf(p, [x1 - 0.5, 0, 0], [L, w, h]))
        r = np.hypot(p[:, 0] - hole[0], p[:, 1] - hole[1]) - c["hole"] / 2
        solid = np.maximum(solid, -r)
        sheet = np.abs(tpms(c["kind"], p, c["cell"])) - c["wall"] / 2
        core = np.maximum(sheet, box_sdf(p, [x0, 0, 0], [x1, w, h]))
        return np.minimum(solid, core)

    return sdf, (L, w, h), hole


def _lattice_mesh(c=_LAT, pitch=0.25):
    import trimesh
    from skimage.measure import marching_cubes

    sdf, (L, w, h), _ = _lattice_sdf(c)
    # grid offset off the faces: a sample exactly on a flat face (sdf = 0)
    # makes marching cubes leave a non-manifold seam
    pad = 2.137 * pitch
    xs = np.arange(-pad, L + pad + pitch / 2, pitch)
    ys = np.arange(-pad, w + pad + pitch / 2, pitch)
    zs = np.arange(-pad, h + pad + pitch / 2, pitch)
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    for i, x in enumerate(xs):
        p = np.column_stack([np.full(Y.size, x), Y.ravel(), Z.ravel()])
        vol[i] = sdf(p).reshape(Y.shape)
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(pitch,) * 3)
    verts += [xs[0], ys[0], zs[0]]
    m = trimesh.Trimesh(verts, faces, process=False)
    m.merge_vertices()
    if m.volume < 0:
        m.invert()
    # keep the one body (marching cubes can leave a crumb at a clipped cell)
    parts = m.split(only_watertight=True)
    if len(parts) > 1:
        m = max(parts, key=lambda q: q.volume)
    m.apply_translation(-m.bounds[0])
    return m


# ---------------------------------------------------------------- beam theory

def _cantilever_compliance(E, I, a, c=0.0, G_mod=None, area=None, kappa=5 / 6):
    """Tip deflection per newton (mm/N) at distance a + c from the root of a
    cantilever whose first `a` mm bend (E, I) and last `c` mm are rigid,
    plus Timoshenko shear over `a` when G and the area are given."""
    v = a ** 3 / 3 + c * a ** 2 / 2
    th = a ** 2 / 2 + c * a
    d = (v + th * c) / (E * I)
    if G_mod and area:
        d += a / (kappa * G_mod * area)
    return d


def _bone_area(b=_BONE):
    return b["gauge"][0] * b["gauge"][1]


def _lattice_props(mat, c=_LAT, rho=None, factor=1.0):
    from .lattice import relative_density, sheet_modulus
    if rho is None:
        rho = float(relative_density(c["kind"], c["wall"], c["cell"]))
    e = float(sheet_modulus(c["kind"], rho)) * mat.E * factor
    return rho, e


def _lattice_compliance(E_star, c=_LAT, nu_star=0.35):
    I = c["width"] * c["height"] ** 3 / 12
    return _cantilever_compliance(E_star, I, c["core"], c["end"] / 2, E_star / (2 * (1 + nu_star)),
                                  c["width"] * c["height"])


# ---------------------------------------------------------------- the kit

def coupon_kit(material="petg", process="fdm", loads_kg=_LOADS_KG, interlayer=None,
               stiffness_across=None, crush=False, holes=True, hole_gauge=None) -> list:
    """The coupons for one material, each with what to measure and what the
    starting card predicts. FDM by default; for powder or resin the same
    coupons read the (small) anisotropy of those processes. `holes` adds
    the go/no-go hole-fit coupons (hole_coupons), sized round a gauge of
    `hole_gauge` mm (default a 5 mm drill bit)."""
    from .am import _LAYER_STIFFNESS
    from .spec import PROCESSES

    mat = material_from_spec(material)
    inter = interlayer if interlayer is not None else PROCESSES[process]["interlayer"]
    ez = stiffness_across if stiffness_across is not None else _LAYER_STIFFNESS.get(process, (1.0, 1.0))[0]
    c = _CANT
    I = c["width"] * c["thickness"] ** 3 / 12
    sigma_per_kg = 6 * G * c["span"] / (c["width"] * c["thickness"] ** 2)
    kit = []

    cant = _cantilever_manifold()
    for cid, title, man, E, where in (
            ("F-XY", "cantilever, printed flat", cant, mat.E, "along the layers"),
            ("F-Z", "cantilever, printed standing", _rotate_upright(cant), mat.E * ez, "across the layers")):
        m = _to_trimesh(man)
        d = _cantilever_compliance(E, I, c["span"])
        kit.append(Coupon(
            cid, title, m,
            test=(f"Clamp the {c['block'][0]:.0f} mm block flat on a table, its end face flush with the table's "
                  f"edge, beam sticking out level. Hang each weight from the hole ({c['span']:.0f} mm from the "
                  "block) and read how far the hole drops, about 10 s after it settles (PETG creeps). Take the "
                  "weight off between readings and note the zero."),
            measure={"massG": None, "width": None, "thickness": None, "points": [[kg, None] for kg in loads_kg]},
            predicted={"massG": round(float(m.volume) * mat.density / 1000, 2),
                       "mmPerKg": round(d * G, 3),
                       "points": [[kg, round(d * G * kg, 2)] for kg in loads_kg],
                       "rootStressMPaPerKg": round(sigma_per_kg, 2), "E": round(E), "stressing": where},
            geometry={"span": c["span"], "width": c["width"], "thickness": c["thickness"], "I": I}))

    bone, total = _bone_manifold()
    A = _bone_area()
    for cid, title, man, s, where in (
            ("T-XY", "dog-bone, printed flat", bone, mat.strength, "along the layers"),
            ("T-Z", "dog-bone, printed standing", _rotate_upright(bone), mat.strength * inter,
             "across the layers")):
        m = _to_trimesh(man)
        kit.append(Coupon(
            cid, title, m,
            test=("Print two. Loop a strap or carabiner through each end. Anchor one end (a door handle, a "
                  "foot through the loop) and pull the other with a luggage scale on max-hold, slowly (about "
                  "5 s to break). Write down each peak, and where it broke."),
            measure={"massG": None, "width": None, "thickness": None, "breakKg": [None, None], "breakAt": None},
            predicted={"massG": round(float(m.volume) * mat.density / 1000, 2),
                       "breakKg": round(s * A / G, 1), "strength": round(s, 1), "stressing": where},
            geometry={"gaugeWidth": _BONE["gauge"][0], "gaugeThickness": _BONE["gauge"][1], "area": A,
                      "length": total}))

    lat = _lattice_mesh()
    rho, e_star = _lattice_props(mat)
    d = _lattice_compliance(e_star)
    lc = _LAT
    kit.append(Coupon(
        "L-GY", f"cantilever, {lc['kind']} core, printed flat", lat,
        test=("As F-XY: clamp the solid block flush with the table edge, hang the weights from the hole in "
              f"the solid end ({lc['core'] + lc['end'] / 2:.0f} mm from the block) and read the drop."),
        measure={"massG": None, "points": [[kg, None] for kg in loads_kg]},
        predicted={"massG": round(float(lat.volume) * mat.density / 1000, 2), "mmPerKg": round(d * G, 3),
                   "points": [[kg, round(d * G * kg, 2)] for kg in loads_kg],
                   "relativeDensity": round(rho, 3), "Estar": round(e_star, 1)},
        geometry={k: lc[k] for k in ("block", "width", "height", "core", "end", "cell", "wall", "kind")}
        | {"solidVolume": _lattice_solid_volume(lc), "coreVolume": lc["core"] * lc["width"] * lc["height"]}))
    if holes:
        kit.extend(hole_coupons(material, process, gauge=hole_gauge))
    if crush:
        kit.extend(crush_coupons(material, process))
        kit.extend(creep_coupons(material, process))
    return kit


# ---------------------------------------------------------------- hole fits

# seven-segment digits engraved beside each hole: segments a (top), b, c
# (right, top to bottom), d (bottom), e, f (left, bottom to top), g (middle)
_SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
             "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def _digits_cut(text, cx, cy, z_top, h=_HOLE):
    """Boxes to subtract from a face at z = z_top: `text` in seven-segment
    digits centred on (cx, cy), h["digit"] mm tall, read from +z."""
    H, s, dep = h["digit"], h["seg"], h["depth"]
    W = 0.55 * H
    gap = 0.6
    total = len(text) * W + (len(text) - 1) * gap
    cut = None
    for i, ch in enumerate(text):
        x0 = cx - total / 2 + i * (W + gap)
        y0 = cy - H / 2
        boxes = {"a": ([x0, y0 + H - s], [x0 + W, y0 + H]), "g": ([x0, y0 + (H - s) / 2], [x0 + W, y0 + (H + s) / 2]),
                 "d": ([x0, y0], [x0 + W, y0 + s]),
                 "b": ([x0 + W - s, y0 + H / 2], [x0 + W, y0 + H]), "c": ([x0 + W - s, y0], [x0 + W, y0 + H / 2]),
                 "f": ([x0, y0 + H / 2], [x0 + s, y0 + H]), "e": ([x0, y0], [x0 + s, y0 + H / 2])}
        for seg in _SEGMENTS[ch]:
            lo, hi = boxes[seg]
            b = _box([lo[0], lo[1], z_top - dep], [hi[0], hi[1], z_top + 1.0])
            cut = b if cut is None else cut + b
    return cut


def _hole_ladder(bias, gauge, h=_HOLE):
    """Drawn diameters, h["step"] apart, that put the gauge's go/no-go
    change near the middle of the row for a process whose holes print
    `bias` mm off (as built minus drawn)."""
    start = round((gauge - bias - h["below"]) / h["step"]) * h["step"]
    return [round(start + i * h["step"], 3) for i in range(h["holes"])]


def _hole_plate(drawn, chamfer=True, h=_HOLE):
    """The plate, flat on the bed: holes along z, numbered 1.. from -x,
    each with a small chamfer on the bed side (FDM's squashed first layer
    would otherwise close the hole's bottom edge)."""
    import manifold3d as m3d
    n = len(drawn)
    L = 2 * h["margin"] + (n - 1) * h["pitch"]
    T = h["thickness"]
    plate = _box([0, 0, 0], [L, h["width"], T])
    for i, d in enumerate(drawn):
        x = h["margin"] + i * h["pitch"]
        r = d / 2
        plate = plate - _cyl_z(x, h["holeY"], -1.0, T + 1.0, r, n=180)
        if chamfer and h["chamfer"] > 0:
            c = h["chamfer"]
            cone = m3d.Manifold.cylinder(c + 1.0, r + c + 1.0, r, 180).translate([x, h["holeY"], -1.0])
            plate = plate - cone
        plate = plate - _digits_cut(str(i + 1), x, h["digitY"], T, h)
    return plate, L


def _peg_strip(drawn, h=_HOLE):
    """Pegs standing on a numbered strip: the pegs print with their axis
    along the build, like the flat plate's holes."""
    n = len(drawn)
    L = 2 * h["margin"] + (n - 1) * h["pitch"]
    B = h["pegBase"]
    strip = _box([0, 0, 0], [L, h["width"], B])
    for i, d in enumerate(drawn):
        x = h["margin"] + i * h["pitch"]
        strip = strip + _cyl_z(x, h["holeY"], B - 0.01, B + h["pegHeight"], d / 2, n=180)
        strip = strip - _digits_cut(str(i + 1), x, h["digitY"], B, h)
    return strip, L


def _on_edge(man):
    """Stand the plate on its long edge: rotate about x so the holes run
    across the layers (along y)."""
    man = man.rotate([90.0, 0.0, 0.0])
    lo = np.asarray(man.bounding_box()[:3])
    return man.translate(list(-lo))


def _predicted_first(drawn, bias, gauge):
    """1-based number of the first hole the gauge passes, at `bias`."""
    for i, d in enumerate(drawn):
        if d + bias >= gauge - 1e-9:
            return i + 1
    return None


def hole_coupons(material="petg", process="fdm", gauge=None) -> list:
    """Go/no-go hole-fit coupons: H-XY (plate printed flat, holes along the
    build), H-Z (the plate on its edge, holes across the layers) and H-PEG
    (pegs drawn at H-XY's first sizes). The holes step 0.05 mm round a
    gauge of known diameter, the smooth shank of a drill bit by default,
    so the first hole it passes reads the printer's hole bias to +-0.025 mm
    with no calipers. The pegs in H-XY's holes read the clearance a printed
    pin needs in a printed hole. Bias goes to process.holeBias and
    holeBiasAcross (fits.py)."""
    from .fits import HOLES
    from .impact import ELASTOMERS

    mat = material_from_spec(material)
    h = _HOLE
    D = float(gauge if gauge is not None else h["gauge"])
    holes = HOLES.get(process, HOLES["fdm"])
    stretch = mat.name in ELASTOMERS
    go = ("Push the bit's smooth shank (the plain end, not the flutes) into each hole by hand from the numbered "
          "face. Go: it passes all the way through with finger pressure, no twisting or hammering. No-go: it "
          "stops, or needs forcing. Never force it: a forced hole is spoiled for the rest of the test. ")
    if stretch:
        go += ("TPU stretches, so count a hole as go only if the shank slides through without the hole "
               "visibly stretching or the coupon bending. ")
    go += ("Write true (go) or false (no-go) for each hole in order, 1 first. If you print two, give one "
           "list per copy.")
    out = []
    ladders = {}
    for cid, title, bias, where in (("H-XY", "hole ladder, printed flat", holes["bias"], "along the build"),
                                    ("H-Z", "hole ladder, printed on its long edge", holes["biasAcross"],
                                     "across the layers")):
        drawn = _hole_ladder(bias, D, h)
        ladders[cid] = drawn
        man, L = _hole_plate(drawn, chamfer=(cid == "H-XY"), h=h)
        if cid == "H-Z":
            man = _on_edge(man)
        m = _to_trimesh(man)
        first = _predicted_first(drawn, bias, D)
        out.append(Coupon(
            cid, title, m,
            test=(f"Needs a {D:g} mm drill bit (or any smooth rod of known size: a dowel pin, a bearing's "
                  "shaft; write its size in gaugeDiameter). " + go),
            measure={"gauge": f"{D:g} mm drill bit shank", "gaugeDiameter": D, "drawn": drawn,
                     "go": [None] * len(drawn), "smallestGo": None},
            predicted={"massG": round(float(m.volume) * mat.density / 1000, 2), "firstGo": first,
                       "bias": bias, "stressing": where,
                       "note": "design-guide bias (fits.HOLES), not this printer's"},
            geometry={"drawn": drawn, "gauge": D, "step": h["step"], "pitch": h["pitch"],
                      "thickness": h["thickness"], "length": L, "orientation": where}))
    pegs = ladders["H-XY"][:h["pegs"]]
    man, L = _peg_strip(pegs, h)
    m = _to_trimesh(man)
    # a peg prints about its drawn size (FDM perimeters squash out a little,
    # powder sinters on): assume 0 until measured
    hx = ladders["H-XY"]
    enters = [_predicted_first(hx, holes["bias"], p) for p in pegs]
    out.append(Coupon(
        "H-PEG", "pegs, printed standing", m,
        test=("After H-XY: for each peg, find the smallest H-XY hole it slides into by hand, all the way "
              "(as for the bit: no forcing, try from the numbered face). Write that hole's number per peg, "
              "or null if it fits none. This reads the clearance a printed pin needs in a printed hole."),
        measure={"drawn": pegs, "entersHole": [None] * len(pegs)},
        predicted={"massG": round(float(m.volume) * mat.density / 1000, 2), "entersHole": enters,
                   "note": "assumes pegs print at their drawn size and H-XY holes at the design-guide bias"},
        geometry={"drawn": pegs, "height": h["pegHeight"], "pitch": h["pitch"], "length": L,
                  "holes": "H-XY"}))
    return out


def _go_threshold(pairs):
    """The drawn diameter at which a hole starts to pass the gauge, from
    (drawn, went) pairs: the cut between no-go below and go above that
    misreads fewest holes (scatter leaves a mixed zone; the middle of the
    best cuts is taken). Returns (threshold, half-width of what it could
    be, a note) with threshold None when every hole read the same."""
    pairs = sorted((float(d), bool(g)) for d, g in pairs)
    ds = sorted({d for d, _ in pairs})
    if not ds:
        return None, None, "no readings"
    if all(g for _, g in pairs):
        return None, None, f"every hole passed: the gauge fits even {ds[0]:.2f} mm (the ladder is too big)"
    if not any(g for _, g in pairs):
        return None, None, f"no hole passed: the gauge is over {ds[-1]:.2f} mm as built (the ladder is too small)"
    cuts = [ds[0] - 1.0] + [0.5 * (a + b) for a, b in zip(ds, ds[1:])] + [ds[-1] + 1.0]
    errs = [sum((d < c and g) or (d > c and not g) for d, g in pairs) for c in cuts]
    best = min(errs)
    good = [c for c, e in zip(cuts, errs) if e == best]
    t = float(np.mean(good))
    # the cut is known to within the gap round it
    i = int(np.argmin([abs(c - t) for c in cuts]))
    lo = ds[i - 1] if i > 0 else ds[0]
    hi = ds[i] if i < len(ds) else ds[-1]
    half = 0.5 * (max(good) - min(good)) + 0.5 * (hi - lo)
    note = None
    if best:
        note = f"{best} hole(s) out of order: the printer's scatter is about the width of the mixed zone"
    return t, half, note


def _go_pairs(e):
    """(drawn, went) pairs from a filled-in H-XY / H-Z entry: `go` as one
    list or a list per copy, or just the smallest hole that passed."""
    drawn = e.get("drawn") or []
    rows = e.get("go")
    pairs = []
    if rows and isinstance(rows[0], list):
        lists = rows
    else:
        lists = [rows or []]
    for row in lists:
        for d, g in zip(drawn, row):
            if g is not None:
                pairs.append((d, bool(g)))
    if not pairs and e.get("smallestGo"):
        k = int(e["smallestGo"])
        pairs = [(d, i + 1 >= k) for i, d in enumerate(drawn)]
    return pairs


def fit_holes(coupons: dict) -> dict | None:
    """The printer's hole bias from filled-in H-XY / H-Z go/no-go readings
    (as built minus drawn diameter: the gauge's diameter minus the drawn
    size at which holes start to pass it), and from H-PEG the drawn
    clearance a printed peg needs to slide into a printed hole, and the
    pegs' own bias. None when no hole coupon was read."""
    out, ev = {}, {}
    for cid, key in (("H-XY", "bias"), ("H-Z", "biasAcross")):
        e = coupons.get(cid) or {}
        pairs = _go_pairs(e)
        if not pairs:
            continue
        D = float(e.get("gaugeDiameter") or _HOLE["gauge"])
        t, half, note = _go_threshold(pairs)
        row = {"gaugeDiameter": D, "readings": len(pairs)}
        if t is None:
            row["note"] = note
        else:
            row.update(threshold=round(t, 3), bias=round(D - t, 3), resolution=round(half, 3))
            if note:
                row["note"] = note
            out[key] = round(D - t, 3)
        ev[cid] = row
    e = coupons.get("H-PEG") or {}
    hx = (coupons.get("H-XY") or {}).get("drawn")
    cl = []
    if hx and e.get("drawn") and any(v is not None for v in e.get("entersHole") or []):
        pev = []
        for p, j in zip(e["drawn"], e.get("entersHole") or []):
            if j is None:
                continue
            j = int(j)
            if j <= 0 or j > len(hx):
                pev.append({"peg": p, "note": "fits no hole"})
                continue
            # it went into hole j and not j-1: the cut sits between them
            t = 0.5 * (hx[j - 2] + hx[j - 1]) if j > 1 else None
            if t is None:
                pev.append({"peg": p, "entersHole": j, "note": "entered the smallest hole: clearance at most "
                                                               f"{hx[0] - p:.2f} mm"})
                continue
            cl.append(t - p)
            pev.append({"peg": p, "entersHole": j, "clearance": round(t - p, 3)})
        ev["H-PEG"] = {"pegs": pev}
        if cl:
            c = float(np.mean(cl))
            out["printedSlipClearance"] = round(c, 3)
            if "bias" in out:
                # hole built = drawn + b_h; peg built = drawn + b_p; slip at
                # built hole = built peg: b_p = c + b_h
                out["pegBias"] = round(c + out["bias"], 3)
    if not ev:
        return None
    out["evidence"] = ev
    return out


def _crush_spec(mat, process):
    from .impact import ELASTOMERS
    from .lattice import wall_for_density
    from .spec import PROCESSES

    c = dict(_CRUSH_ELASTIC if mat.name in ELASTOMERS else _CRUSH)
    wall = float(PROCESSES[process]["minFeature"])
    c["cell"] = round(wall / float(wall_for_density(c["kind"], min(c["densities"]), 1.0)), 1)
    c["side"] = c["plan"] * c["cell"]
    c["core"] = c["tall"] * c["cell"]
    return c


def _crush_mesh(c, rho, pitch=None):
    from .fields import FieldGraph

    s, hcore, t = c["side"], c["core"], c["plate"]
    doc = {"nodes": {
        "core": {"op": "lattice", "type": c["kind"], "cell": c["cell"], "density": rho},
        "box": {"op": "shape", "shape": {"type": "box", "min": [0, 0, t], "max": [s, s, t + hcore]}},
        "plates": {"op": "union", "inputs": [
            {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [s, s, t + 0.3]}},
            {"op": "shape", "shape": {"type": "box", "min": [0, 0, t + hcore - 0.3], "max": [s, s, 2 * t + hcore]}}]},
        "part": {"op": "union", "inputs": ["plates", {"op": "intersect", "inputs": ["core", "box"]}]}},
        "output": "part"}
    return FieldGraph(doc).mesh(pitch=pitch or c["cell"] / 40)


def crush_coupons(material="pa12", process="sls") -> list:
    """Gyroid crush blocks at two densities for calibrating impact.py's crush
    law on a real printer: each block's predicted plateau force (and, for
    the lever test, the bucket load that starts it), its densification, and
    what to write down."""
    from .impact import crush_law, densification_strain, plateau_stress

    mat = material_from_spec(material)
    c = _crush_spec(mat, process)
    law, ref = crush_law(c["kind"], mat, calibrated=False)
    area = c["side"] ** 2
    lever = c["lever"]
    out = []
    for rho in c["densities"]:
        m = _crush_mesh(c, rho)
        f = float(plateau_stress(law, rho, ref)) * area
        cid = f"X-{round(rho * 100):d}"
        if lever > 1:
            test = (f"Lever test, {lever:.0f}:1. Use a stiff board about 1 m long (a 2x4). Hold one end down "
                    "under something that cannot lift (a strap round a heavy bench leg, the underside of a "
                    "bolted shelf). Stand the coupon on the floor under the board, its centre 1/"
                    f"{lever:.0f} of the way from that end to where the bucket hangs (6.7 cm for a 1 m "
                    "lever at 15:1). Hang a bucket at the far end and pour water in a litre at a time. After each "
                    "litre, weigh or note the bucket's total kg and measure how far the board's end has "
                    "dropped from where it started (a ruler against a wall mark). The block holds, then "
                    "squashes in one go when the bucket reaches the plateau load; keep adding until it "
                    "stops sinking. Write down the board's mass once (half of it counts at the bucket).")
        else:
            test = ("Weights on a plate. Stand the coupon on a table, lay a small flat board or a hardcover "
                    "book on it, and add weights (water bottles weighed on a kitchen scale) one at a time. "
                    "After each, note the total kg on top (board included) and the coupon's height "
                    "with calipers or a ruler. It squashes suddenly at the plateau and stops when the walls "
                    "touch; take it a bit past that. It springs back afterwards: measure that too.")
        out.append(Coupon(
            cid, f"{c['kind']} crush block, relative density {rho:.2f}", m, test=test,
            measure={"massG": None, "height": None, "leverRatio": lever, "boardKg": None if lever > 1 else 0.0,
                     "points": [[None, None]]},
            predicted={"massG": round(float(m.volume) * mat.density / 1000, 2),
                       "plateauN": round(f, 1), "plateauKgAtLoad": round(f / G / lever, 1),
                       "densificationStrain": round(float(densification_strain(rho)), 2),
                       "law": "elastic buckling" if law.endswith(":elastic") else "plastic collapse"},
            geometry={"rho": rho, "cell": c["cell"], "side": c["side"], "core": c["core"], "plate": c["plate"],
                      "kind": c["kind"], "lever": lever}))
    return out


def creep_coupons(material="tpu", process="fdm") -> list:
    """For materials viscoelastic.py models (TPU): one gyroid block held at
    half its buckling load for a week, its height read with calipers at
    growing intervals. The log-time slope of the deflection is the creep
    rate m that sets an isolator's long-term sag and its stiffening under
    vibration and impact. Empty for other materials."""
    from .impact import crush_law, plateau_stress
    from .lattice import sheet_modulus
    from .viscoelastic import creep_factor, creep_params

    mat = material_from_spec(material)
    if creep_params(mat) is None:
        return []
    c = dict(_crush_spec(mat, process))
    k = _CREEP
    c["tall"] = k["tall"]
    c["core"] = k["tall"] * c["cell"]
    rho = k["rho"]
    law, ref = crush_law(c["kind"], mat, calibrated=False)
    area = c["side"] ** 2
    f = k["loadShare"] * float(plateau_stress(law, rho, ref)) * area
    strain = f / area / (float(mat.E) * float(sheet_modulus(c["kind"], rho)))
    pred = {f"{m} min": round(float(c["core"] * strain * creep_factor(mat, 60.0 * m)), 3) for m in k["readMinutes"]}
    m = _crush_mesh(c, rho)
    test = (f"Creep under a dead weight. Measure the coupon's height with calipers first (unloaded). Then stand "
            f"it on a table, lay a small flat board on it and load it to {f / G:.2f} kg in all, board "
            "included (water bottles weighed on a kitchen scale), gently, in one go; start a timer. Read the "
            "height at the corners of the board (average two opposite corners) at about "
            + ", ".join(f"{x} min" for x in k["readMinutes"][:4]) + ", 1 day, 3 days and a week. Keep it "
            "somewhere the temperature stays steady (TPU creeps faster when warm) and note the room "
            "temperature. Write each reading as [minutes since loading, height mm].")
    return [Coupon(
        f"K-{round(rho * 100):d}", f"{c['kind']} creep block, relative density {rho:.2f}", m, test=test,
        measure={"massG": None, "heightUnloaded": None, "loadKg": round(f / G, 2), "temperatureC": None,
                 "points": [[None, None]]},
        predicted={"massG": round(float(m.volume) * mat.density / 1000, 2), "loadKg": round(f / G, 2),
                   "deflectionMm": pred, "creepRate": creep_params(mat)[0]},
        geometry={"rho": rho, "cell": c["cell"], "side": c["side"], "core": c["core"], "plate": c["plate"],
                  "kind": c["kind"]})]


def fit_creep(coupons: dict, mat, process: str = "fdm") -> dict | None:
    """The creep block from a filled-in K-* coupon: deflection against the
    log of time is a straight line d = a + b ln t (viscoelastic.py's law),
    so m = b / (a + b ln t_ref): the creep rate per e-fold of time relative
    to the deflection at the card's time scale. Also the modulus the
    coupon shows at t_ref, over the card's."""
    from .lattice import sheet_modulus
    from .viscoelastic import creep_params

    p = creep_params(mat)
    t_ref = p[1] if p else 10.0
    for cid, e in sorted(coupons.items()):
        if not cid.startswith("K-") or not e.get("heightUnloaded"):
            continue
        pts = [(float(t), float(h)) for t, h in (e.get("points") or []) if t is not None and h is not None]
        if len(pts) < 3:
            continue
        t = np.array([x for x, _ in pts]) * 60.0
        d = float(e["heightUnloaded"]) - np.array([h for _, h in pts])
        b, a = np.polyfit(np.log(t), d, 1)
        d_ref = a + b * np.log(t_ref)
        out = {"m": round(float(b / d_ref), 4), "tRef": t_ref,
               "coupon": {"id": cid, "points": len(pts), "deflectionAtTRef": round(float(d_ref), 3),
                          "temperatureC": e.get("temperatureC")}}
        geo = _crush_spec(mat, process)
        if e.get("loadKg") and d_ref > 0:
            core = _CREEP["tall"] * geo["cell"]
            e_meas = float(e["loadKg"]) * G / geo["side"] ** 2 / (d_ref / core)
            out["coupon"]["modulusFactor"] = round(e_meas / (float(mat.E) * float(sheet_modulus(geo["kind"],
                                                                                        _CREEP["rho"]))), 3)
        return out
    return None


def _lattice_solid_volume(c=_LAT):
    hole = np.pi * (c["hole"] / 2) ** 2 * c["height"]
    return (c["block"] + c["end"]) * c["width"] * c["height"] - hole


def measurement_template(kit, material="petg", process="fdm", parts=None) -> dict:
    """What to fill in after testing: one entry per coupon (and part test).
    Masses in grams, lengths in mm, loads in kg (what a kitchen scale reads).
    Points are [kg hung, mm the load point dropped]."""
    return {
        "material": material if isinstance(material, str) else material_from_spec(material).name,
        "process": process,
        "printer": None, "filament": None,
        "slicer": {"layer": None, "walls": None, "infill": "100%", "nozzleTemp": None},
        "notes": None,
        "coupons": {c.id: dict(c.measure) for c in kit},
        # a part test: drop at the hook in mm, or set laserDistanceM and give
        # the laser spot's travel on the wall in mm
        "parts": {k: {"massG": None, "laserDistanceM": None, "points": [[kg, None] for kg in v.get("loadsKg", [])]}
                  for k, v in (parts or {}).items()},
    }


def write_kit(kit, out_dir, material="petg", process="fdm", parts=None) -> dict:
    """Write each coupon's STL, predictions.json and measurements.json (the
    template to fill in). Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    mat = material_from_spec(material)
    paths = {}
    for c in kit:
        p = os.path.join(out_dir, f"coupon-{c.id}.stl")
        c.mesh.export(p)
        paths[c.id] = p
    pred = {"material": mat.to_dict(), "process": process,
            "coupons": {c.id: c.to_dict() for c in kit}, "parts": parts or {}}
    with open(os.path.join(out_dir, "predictions.json"), "w") as fh:
        json.dump(pred, fh, indent=2)
    tmpl = measurement_template(kit, mat.name, process, parts)
    with open(os.path.join(out_dir, "measurements.json"), "w") as fh:
        json.dump(tmpl, fh, indent=2)
    paths["predictions"] = os.path.join(out_dir, "predictions.json")
    paths["measurements"] = os.path.join(out_dir, "measurements.json")
    return paths


# ---------------------------------------------------------------- part tests

def part_prediction(study, mesh, hook, rigid=None, probes=None, max_cells=120_000) -> dict:
    """Deflection per kg at a hook point for a finished part under a hanging
    weight. `study` holds the part as in the test (its supports are the
    fixture, its one load case the weight, 1 kg as `force` [N]); `hook` is
    the point whose drop is read. With `rigid` (a shapes.Shape: a stiff
    block the weight hangs from, such as a dummy servo) the block's rigid
    motion is fitted to its nodes and read at the hook, and its rotation is
    reported too (for an optical lever: a laser on the block moves its spot
    by rotation x distance). `probes` ({name: Shape}) adds each region's
    fitted rotation, to see how much of the motion a fixture adds. Uses
    verify's grid and the study's material."""
    from .verify import verify_part

    rep = verify_part(study, mesh, max_cells=max_cells, refine=0, fields=True)
    f = rep["fields"]
    nodes = np.asarray(f["nodes"], float)
    u = np.asarray(f["cases"][0]["displacement"], float).reshape(-1, 3)
    load = study.load_cases[0].loads[0]
    kg = float(np.linalg.norm(load.force)) / G
    dirn = load.force / np.linalg.norm(load.force)
    hook = np.asarray(hook, float)
    out = {}
    if rigid is not None:
        sel = rigid.sdf(nodes) <= 0
        if sel.sum() < 4:
            raise ValueError("the rigid block holds fewer than 4 analysis nodes")
        t, w = rigid_motion(nodes[sel], u[sel])
        uh = t + np.cross(w, hook - nodes[sel].mean(axis=0))
        out["rotationMradPerKg"] = (w / kg * 1000).tolist()
        out["rigidNodes"] = int(sel.sum())
    else:
        d = np.linalg.norm(nodes - hook, axis=1)
        near = np.argsort(d)[:8]
        uh = u[near].mean(axis=0)
        out["hookNodeDistance"] = float(d[near].max())
    for name, shape in (probes or {}).items():
        sel = shape.sdf(nodes) <= 0
        if sel.sum() >= 4:
            out.setdefault("probeRotationMradPerKg", {})[name] = (rigid_motion(nodes[sel], u[sel])[1]
                                                                  / kg * 1000).tolist()
    c0 = rep["cases"][0]
    out.update({"mmPerKg": float(uh @ dirn) / kg, "E": study.material.E,
                "peakStressMPaPerKg": c0["peakStress"] / kg, "peakAt": c0["peakAt"],
                "massG": rep["mass"], "analysisPitch": rep["analysisPitch"]})
    return out


def rigid_motion(x, u):
    """Least-squares rigid motion (translation at the centroid, small
    rotation vector) of points x displaced by u."""
    c = x.mean(axis=0)
    r = x - c
    t = u.mean(axis=0)
    # u - t = w x r  ->  A w = b with A the cross-product matrices
    A = np.zeros((3 * len(r), 3))
    A[0::3, 1], A[0::3, 2] = r[:, 2], -r[:, 1]
    A[1::3, 0], A[1::3, 2] = -r[:, 2], r[:, 0]
    A[2::3, 0], A[2::3, 1] = r[:, 1], -r[:, 0]
    w = np.linalg.lstsq(A, (u - t).ravel(), rcond=None)[0]
    return t, w


# ---------------------------------------------------------------- the fit

def _slope(points):
    """mm per kg through [kg, mm] points: least squares with an offset when
    there are three or more (clamp slack, a misread zero), else through 0."""
    pts = np.asarray([p for p in points or [] if p and p[0] is not None and p[1] is not None], float)
    if len(pts) == 0:
        return None, None
    kg, mm = pts[:, 0], pts[:, 1]
    if len(pts) >= 3:
        A = np.column_stack([kg, np.ones_like(kg)])
        (s, b), *_ = np.linalg.lstsq(A, mm, rcond=None)
        resid = mm - (s * kg + b)
        r2 = 1 - (resid ** 2).sum() / max(((mm - mm.mean()) ** 2).sum(), 1e-12)
        return float(s), {"offsetMm": round(float(b), 3), "r2": round(float(r2), 4), "n": len(pts)}
    s = float((kg * mm).sum() / (kg * kg).sum())
    return s, {"n": len(pts)}


def _geom(entry, key, default):
    v = entry.get(key)
    return float(v) if v else float(default)


def fit(measurements: dict, predictions: dict | None = None, name: str | None = None) -> dict:
    """A calibrated material card from filled-in measurements (the
    write_kit template). Missing coupons leave the starting card's values
    in place and say so. `predictions` (predictions.json) supplies the part
    tests' predicted deflections, so the part's model factor can be
    reported. Returns the card: material (usable as a study's material),
    process factors, lattice and model factors, and the evidence."""
    from .am import _LAYER_STIFFNESS
    from .spec import PROCESSES

    base = material_from_spec(measurements.get("material", "petg"))
    proc = measurements.get("process", "fdm")
    co = measurements.get("coupons") or {}
    ev, kept = {}, []

    # modulus, along and across the layers
    c = _CANT
    E = {}
    for cid in ("F-XY", "F-Z"):
        e = co.get(cid) or {}
        s, info = _slope(e.get("points"))
        if s is None or s <= 0:
            continue
        b, h = _geom(e, "width", c["width"]), _geom(e, "thickness", c["thickness"])
        I = b * h ** 3 / 12
        E[cid] = _cantilever_compliance(1.0, I, c["span"]) * G / s
        ev[cid] = {"mmPerKg": round(s, 4), "E": round(E[cid]), **(info or {}), "I": round(I, 2)}
    E_xy = E.get("F-XY")
    if E_xy is None:
        E_xy = base.E
        kept.append("E (no F-XY points)")
    ez = None
    if "F-Z" in E and "F-XY" in E:
        ez = E["F-Z"] / E["F-XY"]

    # strength, along and across the layers
    S = {}
    for cid in ("T-XY", "T-Z"):
        e = co.get(cid) or {}
        if not any(k for k in np.atleast_1d(e.get("breakKg") or [])):
            continue
        w, t = _geom(e, "width", _BONE["gauge"][0]), _geom(e, "thickness", _BONE["gauge"][1])
        # one break, or a list from several copies (the mean is used)
        kgs = [float(k) for k in np.atleast_1d(e["breakKg"]) if k]
        S[cid] = float(np.mean(kgs)) * G / (w * t)
        ev[cid] = {"breakKg": e["breakKg"], "strength": round(S[cid], 1), "area": round(w * t, 2)}
        if len(kgs) > 1:
            ev[cid]["scatter"] = round(float(np.std(kgs, ddof=1) / np.mean(kgs)), 3)
        where = str(e.get("breakAt") or "")
        if where and not any(k in where.lower() for k in ("gauge", "middle", "centre", "center")):
            ev[cid]["warning"] = (f"broke at {where!r}, not in the gauge: the strength reads low; "
                                  "a stress raiser (seam, taper, hole) failed first")
    s_xy = S.get("T-XY")
    if s_xy is None:
        s_xy = base.strength
        kept.append("strength (no T-XY break)")
    inter = S["T-Z"] / s_xy if "T-Z" in S else None

    # as-printed density from the solid coupons' masses
    kit_vol = _coupon_volumes()
    dens = []
    for cid in ("F-XY", "F-Z", "T-XY", "T-Z"):
        mg = (co.get(cid) or {}).get("massG")
        if mg:
            dens.append(float(mg) / kit_vol[cid] * 1000)
            ev.setdefault(cid, {})["density"] = round(dens[-1], 3)
    rho_mat = float(np.median(dens)) if dens else base.density
    if not dens:
        kept.append("density (no coupon masses)")

    mat = Material(name or f"{base.name}-calibrated", round(float(E_xy), 0), base.nu,
                   round(float(s_xy), 1), round(rho_mat, 3),
                   f"calibrated from printed coupons ({measurements.get('printer') or 'printer not given'}, "
                   f"{measurements.get('filament') or base.name})", base.processes)

    # lattice: measured core modulus against the homogenized sheet model
    lat = None
    e = co.get("L-GY") or {}
    s, info = _slope(e.get("points"))
    if s and s > 0:
        lc = _LAT
        unit = _lattice_compliance(1.0)            # mm/N at E* = 1 MPa
        e_meas = unit * G / s
        rho_nom, e_nom = _lattice_props(mat)
        rho_built = rho_nom
        if e.get("massG"):
            core_mass = float(e["massG"]) - _lattice_solid_volume(lc) * rho_mat / 1000
            core_vol = lc["core"] * lc["width"] * lc["height"]
            rho_built = max(core_mass * 1000 / (rho_mat * core_vol), 1e-3)
        _, e_built = _lattice_props(mat, rho=rho_built)
        lat = {"kind": lc["kind"], "relativeDensityDrawn": round(rho_nom, 3),
               "relativeDensityPrinted": round(rho_built, 3),
               "Estar": round(e_meas, 1), "EstarModel": round(e_nom, 1),
               # model at the density that printed: wall-width error taken
               # out, what is left is the sheet model's own error
               "stiffnessFactor": round(e_meas / e_built, 3), **(info or {})}
        ev["L-GY"] = {"mmPerKg": round(s, 4), **lat}

    # part tests: measured against verify's prediction at the fitted modulus
    parts = {}
    pred_parts = (predictions or {}).get("parts") or {}
    for pid, e in (measurements.get("parts") or {}).items():
        s, info = _slope(e.get("points"))
        if not s or s <= 0:
            continue
        out = {"mmPerKg": round(s, 4), **(info or {})}
        pp = pred_parts.get(pid) or {}
        # read with a dial at the hook (mm), or a laser spot on a wall
        # laserDistanceM away (mm of spot travel)
        laser = e.get("laserDistanceM")
        per_kg = pp.get("laserSpotMmPerKgPerM") * float(laser) if laser and pp.get("laserSpotMmPerKgPerM") \
            else (None if laser else pp.get("mmPerKg"))
        if per_kg and pp.get("E"):
            pred = per_kg * pp["E"] / mat.E      # linear: deflection goes as 1/E
            out.update(predictedMmPerKg=round(pred, 4), stiffnessFactor=round(pred / s, 3))
            if laser:
                out["read"] = f"laser spot at {laser} m"
        if e.get("massG") and pp.get("massG"):
            # predicted at the starting card's density; rescale to the printed one
            pred_g = pp["massG"] * mat.density / float(pp.get("density") or base.density)
            out.update(predictedMassG=round(pred_g, 2), massRatio=round(float(e["massG"]) / pred_g, 3))
        parts[pid] = out

    process = {"type": proc}
    if inter is not None:
        process["interlayer"] = round(float(min(inter, 1.0)), 3)
    else:
        kept.append(f"interlayer (no T-Z break; process default {PROCESSES[proc]['interlayer']})")
    if ez is not None:
        process["stiffnessAcrossLayers"] = round(float(min(ez, 1.0)), 3)
    else:
        kept.append(f"stiffnessAcrossLayers (no F-Z points; default {_LAYER_STIFFNESS.get(proc, (1.0,))[0]})")
    holes = fit_holes(co)
    if holes is not None:
        if "bias" in holes:
            process["holeBias"] = holes["bias"]
        if "biasAcross" in holes:
            process["holeBiasAcross"] = holes["biasAcross"]
            if "bias" not in holes:
                process["holeBias"] = holes["biasAcross"]
                kept.append("hole bias along the build (no H-XY readings; H-Z's used for both)")
    if "holeBias" not in process and any(k.startswith("H-") for k in co):
        kept.append("hole bias (no H-XY or H-Z go/no-go readings; fits.py's design-guide default)")
    crush = fit_crush(co, mat, base, proc)
    creep = fit_creep(co, base, proc)
    if creep is None and any(k.startswith("K-") for k in co):
        kept.append("creep rate (no K- readings)")
    if crush is None and any(k.startswith("X-") for k in co):
        kept.append("crush plateau (no X- crush points)")
    sf = [p["stiffnessFactor"] for p in parts.values() if "stiffnessFactor" in p]
    card = {
        "name": mat.name,
        "basedOn": base.name,
        # pasteable as a study's custom material
        "material": {k: v for k, v in mat.to_dict().items() if k != "processes"},
        "process": process,
        "change": {"E": round(mat.E / base.E, 3), "strength": round(mat.strength / base.strength, 3),
                   "density": round(mat.density / base.density, 3)},
        "lattice": lat,
        "crush": crush,
        "creep": creep,
        "holes": holes,
        "parts": parts,
        # a part's measured stiffness over verify's, both at the calibrated
        # modulus: what the voxel model, bonded joints and slicer fill miss
        "modelFactor": {"stiffness": round(float(np.exp(np.mean(np.log(sf)))), 3)} if sf else None,
        "evidence": ev,
        "kept": kept,
        "measured": {k: measurements.get(k) for k in ("printer", "filament", "slicer", "notes")},
    }
    return card


def _coupon_volumes():
    """Volumes (mm^3) of the solid coupons, computed once."""
    global _VOLS
    try:
        return _VOLS
    except NameError:
        pass
    cant = _cantilever_manifold().volume()
    bone = _bone_manifold()[0].volume()
    _VOLS = {"F-XY": cant, "F-Z": cant, "T-XY": bone, "T-Z": bone}
    return _VOLS


def fit_crush(coupons: dict, mat, base, process: str) -> dict | None:
    """The crush block from filled-in X-* coupons: per coupon the plateau
    (the load at which it ran away, between the last load it held and the
    one that squashed it), the strain it ran to (densification) and the
    density it printed at; overall the factor on impact.py's plateau (the
    geometric mean of measured over predicted at the printed density) and,
    with two densities, the measured exponent."""
    from .impact import crush_law, plateau_stress

    spec = _crush_spec(base, process)
    law, _ = crush_law(spec["kind"], base, calibrated=False)       # the law goes with the base material
    ref = float(mat.E) if law.endswith(":elastic") else float(mat.strength)
    area, hcore, t = spec["side"] ** 2, spec["core"], spec["plate"]
    ev = {}
    for cid, e in sorted(coupons.items()):
        if not cid.startswith("X-"):
            continue
        pts = sorted((float(k), float(d)) for k, d in (e.get("points") or []) if k is not None and d is not None)
        if not pts:
            continue
        ratio = float(e.get("leverRatio") or 1.0)
        board = float(e.get("boardKg") or 0.0)
        force = np.array([(k + board / 2) * G * ratio for k, _ in pts])
        strain = np.array([d / ratio / hcore for _, d in pts])
        rho_draw = int(cid[2:]) / 100
        rho = rho_draw
        if e.get("massG"):
            core_g = float(e["massG"]) - 2 * t * area * mat.density / 1000
            rho = max(core_g * 1000 / (mat.density * area * hcore), 1e-3)
        out = {"relativeDensityDrawn": rho_draw, "relativeDensityPrinted": round(rho, 3)}
        jump = np.flatnonzero(strain > 0.08)
        if len(jump) == 0:
            out.update(plateauAtLeastN=round(float(force.max()), 1),
                       note="never squashed: the plateau is above the largest load")
            ev[cid] = out
            continue
        i = int(jump[0])
        pl = 0.5 * (force[i - 1] + force[i]) if i > 0 else force[i]
        model = float(plateau_stress(law, rho, ref)) * area
        out.update(plateauN=round(float(pl), 1), bracketN=[round(float(force[i - 1]), 1) if i > 0 else 0.0,
                                                          round(float(force[i]), 1)],
                   densificationStrain=round(float(strain[i:].max()), 3), predictedN=round(model, 1),
                   factor=round(float(pl / model), 3), plateauMPa=round(float(pl / area), 3))
        ev[cid] = out
    got = [v for v in ev.values() if "factor" in v]
    if not got:
        return {"coupons": ev} if ev else None
    crush = {"law": "elastic" if law.endswith(":elastic") else "plastic", "kind": spec["kind"],
             "factor": round(float(np.exp(np.mean(np.log([v["factor"] for v in got])))), 3),
             "densificationAlpha": round(float(np.mean([(1 - v["densificationStrain"]) / v["relativeDensityPrinted"]
                                                         for v in got])), 2),
             "coupons": ev}
    if len(got) >= 2:
        a, b = sorted(got, key=lambda v: v["relativeDensityPrinted"])[:: len(got) - 1]
        s1, s2 = a["plateauMPa"] / ref, b["plateauMPa"] / ref
        p = float(np.log(s2 / s1) / np.log(b["relativeDensityPrinted"] / a["relativeDensityPrinted"]))
        crush.update(p=round(p, 3), D=round(float(s1 / a["relativeDensityPrinted"] ** p), 4))
    return crush


def crush_factor(name: str, law: str) -> float:
    """The measured factor on impact.py's crush plateau from a saved card
    of this name (1 when there is none, or it was fitted for the other
    law)."""
    try:
        card = load_card(name) if name else None
    except Exception:  # noqa: BLE001
        card = None
    cr = (card or {}).get("crush") or {}
    return float(cr["factor"]) if cr.get("factor") and cr.get("law") == law else 1.0


# ---------------------------------------------------------------- cards

def list_cards() -> list:
    if not os.path.isdir(CALIBRATION_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CALIBRATION_DIR) if f.endswith(".json"))


def load_card(name: str) -> dict | None:
    """A saved calibration card by name (loadpath/calibrations/<name>.json),
    or None."""
    key = name.strip().lower()
    path = os.path.join(CALIBRATION_DIR, f"{key}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)
