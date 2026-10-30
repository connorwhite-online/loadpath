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

__all__ = ["Coupon", "coupon_kit", "write_kit", "fit", "measurement_template", "part_prediction",
           "rigid_motion", "CALIBRATION_DIR", "load_card", "list_cards"]

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
               stiffness_across=None) -> list:
    """The coupons for one material, each with what to measure and what the
    starting card predicts. FDM by default; for powder or resin the same
    coupons read the (small) anisotropy of those processes."""
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
    return kit


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
