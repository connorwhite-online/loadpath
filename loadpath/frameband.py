"""
A product's own side wall as its antenna: a metal band let into a fixed
exterior, flush with it, and the radio model of the band in that product.

Phones do this with their frames. Here the exterior is a designer's, drawn
whole and split into shells; the band is the outer `wall` of the side
between two heights, cut out of those shells, so its outer face is the
original surface and nothing outside changes. What the band needs that the
exterior does not say comes from physics, not from a drawing:

- **splits**: gaps right through the band (positions along its perimeter,
  as fractions from a start point), filled with the shell plastic. A split
  sets where a radiating arm ends; how long the arms are is a radio
  question, solved with `antenna.tune` or a sweep, not guessed here.
- **feed and ground ties**: where the radio's feed and the board's ground
  reach the band, as lugs off its inner face through the wall to the
  cavity (wires in the radio model). A tie shorts that point to ground, so
  ties set which stretch of band is a grounded arm and which a slot.
- **the air behind it**: plastic against the metal holds field and loses
  power in it (TPU's loss tangent is high), so a `behind`-deep air gap is
  cut into the shells behind the strip, ribbed every `rib_pitch` so the
  strip still bears on the shell, never through to the cavity (`keep` of
  wall stays).
- **sheet**: when the band is cut and bent from sheet instead of machined,
  the lugs become tabs off its edge folded inward, `holds` are tabs into
  the shell above, `ears` are tabs out to existing shell screws.

Everything behind the band is built as a few straight-walled layers: the
allowed planform (the shells' wall less `keep` from its inner face) is found
every 0.05 mm of height and heights that agree form one layer, so a layer
edge sits only where the shells' own wall changes (a lip, a shell's top).
Slicing at a fixed pitch instead leaves stepped slivers and loose pieces of
no volume along every curved wall. Every cut is cleaned of features
narrower than 0.6 mm (no knife-thin fins), and each part is checked to be
one watertight body.

`cut_band` makes the parts and a report (the exterior check against the
original surface included); `band_scene` makes the `antenna.Scene` of the
band in the product, given what is ground inside (boards, battery, shields)
and the skin's material, so `antenna.simulate` and `antenna.match` give the
radiated efficiency per band. `hand_grip` adds a hand holding it (tissue
values of CTIA 01.72), `usb_cable` a plugged-in cable's shield.
"""
from __future__ import annotations

import math

import numpy as np

# CTIA Certification, "Test Plan for Near-Field Phantoms" CTIA 01.72 v4.0.1,
# Table 2.2.4-1 (dry palm tissue, from measurements of five people's palms):
# frequency MHz -> (relative permittivity, conductivity S/m). Phantoms must be
# within +-15 % (permittivity) and +-25 % (conductivity) of these.
HAND_TISSUE = {300: (37.1, 0.36), 450: (33.9, 0.43), 835: (30.3, 0.59), 900: (30.0, 0.62),
               1450: (27.9, 0.85), 1575: (27.5, 0.90), 1800: (27.0, 0.99), 1900: (26.7, 1.04),
               1950: (26.6, 1.07), 2000: (26.5, 1.09), 2100: (26.3, 1.14), 2450: (25.7, 1.32),
               3000: (24.8, 1.61), 4000: (23.5, 2.18), 5000: (22.2, 2.84)}

# CTIA 01.72 v4.0.1 Table 2.2.3-1 (50th percentile hand, men and women averaged)
HAND_SIZE = {"hand_width": 85.0, "finger_pip_width": 21.0, "thumb_dip_width": 22.3}


def hand_tissue(f_mhz):
    """(permittivity, conductivity S/m) of hand tissue at f_mhz, interpolated
    in CTIA 01.72 Table 2.2.4-1."""
    f = np.array(sorted(HAND_TISSUE))
    e = np.array([HAND_TISSUE[k][0] for k in f]); s = np.array([HAND_TISSUE[k][1] for k in f])
    return float(np.interp(f_mhz, f, e)), float(np.interp(f_mhz, f, s))


# ---------------------------------------------------------------- geometry --

def _shapely():
    from shapely.geometry import Polygon, LineString, Point, box
    from shapely.ops import unary_union
    return Polygon, LineString, Point, box, unary_union


def section(mesh, z):
    """The mesh cut at height z, as shapely polygons (holes kept)."""
    Polygon, _, _, _, unary_union = _shapely()
    # a plane lying exactly in a face (a joint between shells, a floor) makes
    # the cut ill-posed: cut 1e-4 mm off it
    zs = mesh.vertices[:, 2]
    if np.any(np.abs(zs - z) < 1e-6):
        z = z + 1e-4
    s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if s is None:
        return Polygon()
    p, T = s.to_2D()
    out = []
    for g in p.polygons_full:
        tr = lambda c: (np.c_[np.asarray(c), np.zeros(len(c)), np.ones(len(c))] @ T.T)[:, :2]
        out.append(Polygon(tr(g.exterior.coords), [tr(h.coords) for h in g.interiors]).buffer(0))
    return unary_union(out)


def _sections(shells, z):
    _, _, _, _, unary_union = _shapely()
    return unary_union([section(m, z) for m in shells])


def envelope(shells, z, close=14.0):
    """The outline of the whole body at height z, openings closed."""
    Polygon, _, _, _, unary_union = _shapely()
    a = _sections(shells, z).buffer(close, join_style=1).buffer(-close, join_style=1)
    return unary_union([Polygon(g.exterior) for g in getattr(a, "geoms", [a])])


def clean(p, r=0.3):
    """Open then close by r: nothing narrower than 2r is left (no slivers,
    no knife-thin fins) and pieces under 0.5 mm2 go."""
    _, _, _, _, unary_union = _shapely()
    p = p.buffer(-r, join_style=1).buffer(2 * r, join_style=1).buffer(-r, join_style=1)
    return unary_union([g for g in getattr(p, "geoms", [p]) if g.area > 0.5])


def extrude(poly, z0, z1):
    import trimesh
    from trimesh.creation import extrude_polygon
    ms = []
    for p in getattr(poly, "geoms", [poly]):
        if p.area > 1e-6:
            m = extrude_polygon(p, z1 - z0)
            m.apply_translation([0, 0, z0])
            ms.append(m)
    if not ms:
        return None
    return ms[0] if len(ms) == 1 else trimesh.boolean.union(ms, engine="manifold")


def _bool(op, ms):
    import trimesh
    ms = [m for m in ms if m is not None]
    if op == "difference" and len(ms) > 2:          # a - b - c = a - (b | c)
        ms = [ms[0], trimesh.boolean.union(ms[1:], engine="manifold")]
    if op == "difference" and len(ms) == 1:
        return ms[0]
    return getattr(trimesh.boolean, op)(ms, engine="manifold")


class Layout:
    """The band's place on the exterior: `inside` is the outline the band
    sits in at every height z0..z1 (its outer face, openings closed),
    `core` everything the band is not (`wall` in from it), `line` the
    band's mid-line, anticlockwise from the point nearest -y on x = 0
    (or `start` [x, y])."""

    def __init__(self, shells, z=(3.5, 9.5), wall=1.5, start=None):
        z0, z1 = float(z[0]), float(z[1])
        self.z, self.wall = (z0, z1), float(wall)
        envs = [envelope(shells, v) for v in np.linspace(z0, z1, 13)]
        self.inside, self.outside = envs[0], envs[0]
        for e in envs[1:]:
            self.inside = self.inside.intersection(e)
            self.outside = self.outside.union(e)
        self.core = self.inside.buffer(-wall, join_style=1, resolution=16)
        mid = self.inside.buffer(-wall / 2, join_style=1).exterior
        c = np.asarray(mid.coords)
        if not mid.is_ccw:
            c = c[::-1]
        if start is None:
            k = int(np.argmin(np.abs(c[:-1, 0]) + (c[:-1, 1] - c[:, 1].min()) * 1e3))
        else:
            k = int(np.argmin(np.linalg.norm(c[:-1] - np.asarray(start, float), axis=1)))
        _, LineString, _, _, _ = _shapely()
        self.line = LineString(np.vstack([c[k:-1], c[:k + 1]]))
        self.length = self.line.length

    def at(self, s):
        """point, tangent and outward normal at arc length s"""
        s = s % self.length
        p = np.array(self.line.interpolate(s).coords[0])
        q = np.array(self.line.interpolate((s + 0.2) % self.length).coords[0])
        t = (q - p) / np.linalg.norm(q - p)
        return p, t, np.array([t[1], -t[0]])

    def quad(self, s, along, inward, outward):
        Polygon = _shapely()[0]
        p, t, n = self.at(s)
        a = 0.5 * along * t
        return Polygon([p - a - n * inward, p + a - n * inward, p + a + n * outward, p - a + n * outward])


def plan_stack(shells, z0, z1, keep, region, dz=0.05, tol=1.0):
    """What may be cut or filled behind the band, as [(polygon, za, zb)]
    straight-walled layers: at each height (every dz) the allowed planform
    is region & (the shells' wall less `keep` from its inner face);
    consecutive heights whose planforms agree to `tol` mm2 form one layer,
    which takes their intersection."""
    import shapely
    zs = np.arange(z0, z1 + 1e-9, dz)
    zs[-1] = z1
    grid = 1e-3                                   # snap to a micron grid: GEOS can stall on near-coincident edges

    def snap(g):
        g = shapely.set_precision(g, grid)
        return shapely.union_all([p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]) \
            if g.geom_type == "GeometryCollection" else g
    region = snap(region)
    ps = []
    for v in zs:
        w = _sections(shells, v).buffer(0.6, join_style=1).buffer(-0.6 - keep, join_style=1)
        ps.append(snap(shapely.intersection(region, snap(w), grid_size=grid)))
    out, start, acc = [], 0, ps[0]
    for i in range(1, len(zs) + 1):
        if i == len(zs) or ps[i].symmetric_difference(acc).area > tol:
            za = z0 if start == 0 else 0.5 * (zs[start - 1] + zs[start])
            zb = z1 if i == len(zs) else 0.5 * (zs[i - 1] + zs[i])
            p = clean(acc)
            if not p.is_empty and zb - za > 1e-6:
                out.append((p, float(za), float(zb)))
            if i < len(zs):
                start, acc = i, ps[i]
        else:
            acc = shapely.intersection(acc, ps[i], grid_size=grid)
    return out


def gap_ring(L, depth, splits, rib=1.2, rib_pitch=22.0, gap=1.5, keep_clear=()):
    """The air behind the strip: a `depth` ring inside its inner face, less
    a rib every `rib_pitch` (the strip bears on them) and solid at the splits
    and the keep-clear zones."""
    _, _, _, _, unary_union = _shapely()
    ring = L.core.difference(L.core.buffer(-depth, join_style=1))
    for k in keep_clear:
        ring = ring.difference(k)
    ribs = unary_union([L.quad(s, rib, L.wall / 2 + depth + 1.0, 0.0) for s in np.arange(rib_pitch / 2, L.length, rib_pitch)]
                       + [L.quad(s, gap + 4.0, L.wall / 2 + depth + 1.0, 0.0) for s in splits])
    return ring.difference(ribs).buffer(-0.05).buffer(0.05)


def opening_extent(shells, z, origin, direction, through=15.0, across=(-15.0, 15.0), step=0.05):
    """How wide an opening through the wall is (a USB mouth, a button): rays
    from `origin` [x, y] along `direction` [dx, dy], spread across it, at the
    band's heights; those that travel `through` mm past the first wall without
    a hit go through. Returns (lo, hi), offsets across the direction of the
    run of open rays through the centre (None if it is closed)."""
    import trimesh
    B = trimesh.util.concatenate(list(shells))
    d = np.array([direction[0], direction[1], 0.0], float); d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0], 0.0])
    u = np.arange(across[0], across[1] + 1e-9, step)
    lo, hi = [], []
    j0 = int(np.argmin(np.abs(u)))
    for zz in np.arange(z[0], z[1] + 1e-9, 0.25):
        o = np.array([origin[0], origin[1], zz]) + u[:, None] * n
        loc, ri, _ = B.ray.intersects_location(o, np.tile(d, (len(o), 1)), multiple_hits=False)
        hit = np.full(len(o), np.inf)
        hit[ri] = ((loc - o[ri]) @ d)
        first = np.min(hit[np.isfinite(hit)]) if np.isfinite(hit).any() else 0.0
        row = np.isfinite(hit) & (hit > first + through)
        if not row[j0]:
            continue
        a = b = j0
        while a > 0 and row[a - 1]:
            a -= 1
        while b < len(row) - 1 and row[b + 1]:
            b += 1
        lo.append(u[a]); hi.append(u[b])
    return (float(min(lo)), float(max(hi))) if lo else None


def ear(L, wall, xy, width=5.5):
    """A tab off the strip's edge out to a screw at xy (a hole for it is the
    screw's own: the shells' boss and counterbore are kept)."""
    from shapely.geometry import Point
    Polygon = _shapely()[0]
    s = L.line.project(Point(*xy))
    p, t, n = L.at(s)
    q = np.asarray(xy, float)
    a = 0.5 * width * t
    return Polygon([p - a, p + a, q + a, q - a]).union(Point(*q).buffer(width / 2)).buffer(0)


# ------------------------------------------------------------------ parts --

def cut_band(shells, z=(3.5, 9.5), wall=1.5, splits=(), ties=(), feed=None, gap=1.5, lug=(3.0, 1.5),
             reach=8.0, fill=0.25, behind=None, keep=0.8, rib=1.2, rib_pitch=22.0, sheet=None, holds=(),
             ears=(), keep_clear=(), start=None, check=True):
    """Cut a metal band out of a fixed exterior.

    `shells`: {name: trimesh} the product's shells as drawn (their union is
    the exterior). `z` the band's heights, `wall` its depth in from the
    surface. `splits`, `ties`, `feed`, `holds`: positions along the band as
    fractions of its perimeter. `ears`: [[x, y]] shell screws the band
    reaches out to (sheet only). `keep_clear`: shapely polygons in plan the
    band stays out of (an opening and the wall round it, see
    `opening_extent`). `fill`: the band is solid metal behind the original
    surface, `fill` short of the outline, where the shells have wall at
    every height (joint grooves closed). `behind`: depth of the air gap cut
    behind the strip. `sheet`: thickness when cut and bent from sheet.

    Returns {"parts": {name: trimesh}, "layout": Layout, "report": {...}}:
    shells keep their names, band segments are band-1.. (largest first)."""
    import trimesh
    Polygon, _, _, box, unary_union = _shapely()
    names = list(shells)
    meshes = [shells[k] for k in names]
    L = Layout(meshes, z, wall, start)
    z0, z1 = L.z
    zm = 0.5 * (z0 + z1)
    splits_s = [v * L.length for v in splits]
    ties_s = [v * L.length for v in ties]
    feed_s = None if feed is None else feed * L.length
    cuts = unary_union([L.quad(s, gap, wall + 1.0, 6.0) for s in splits_s] + list(keep_clear))
    ring = L.outside.buffer(3.0).difference(L.core).difference(cuts)
    zone = extrude(ring, z0, z1)
    band = _bool("intersection", [_bool("union", meshes), zone])        # its outer face is the original surface
    if fill:
        inner = L.inside.buffer(-fill, join_style=1).difference(L.core).difference(cuts)
        band = _bool("union", [band] + [extrude(p, a, b) for p, a, b in plan_stack(meshes, z0, z1, 0.0, inner)])
    cut = {k: _bool("difference", [m, zone]) for k, m in zip(names, meshes)}
    layers = []
    if behind:
        gr = gap_ring(L, behind, splits_s, rib, rib_pitch, gap, keep_clear)
        layers = plan_stack(meshes, z0, z1, keep, gr)
        rim = gr.intersection(L.inside.difference(L.core.buffer(-0.3)))
        airs = []
        for p, a, b in layers:
            q = clean(p.union(rim))
            # each pocket runs 0.5 mm on into the band's (already empty) zone: no paper-thin wall between
            q = q.union(q.buffer(0.5, join_style=2).difference(L.core.buffer(-0.01, join_style=1))).buffer(0)
            airs.append(extrude(q, a, b))
        cut = {k: _bool("difference", [m] + airs) for k, m in cut.items()}
    lugs = []
    for s in list(ties_s) + ([feed_s] if feed_s is not None else []):
        p, t, n = L.at(s)
        lugs.append(Polygon([p - 0.5 * lug[0] * t - n * (wall / 2 - 0.2), p + 0.5 * lug[0] * t - n * (wall / 2 - 0.2),
                             p + 0.5 * lug[0] * t - n * (wall / 2 + reach), p - 0.5 * lug[0] * t - n * (wall / 2 + reach)]))
    if sheet:
        za, zb = z0, z0 + sheet
        hq = []
        for v in holds:
            p, t, n = L.at(v * L.length)
            hq.append(Polygon([p - 0.5 * lug[0] * t - n * (wall / 2 - 0.2), p + 0.5 * lug[0] * t - n * (wall / 2 - 0.2),
                               p + 0.5 * lug[0] * t - n * (wall / 2 + 2.5), p - 0.5 * lug[0] * t - n * (wall / 2 + 2.5)]))
        eq = [ear(L, wall, xy) for xy in ears]
        lugm = [extrude(q, za, zb) for q in lugs + eq] + [extrude(q, z1 - sheet, z1) for q in hq]
        slot = ([extrude(Polygon(q.exterior).buffer(0.15), za - 0.15, zb + 0.15) for q in lugs + eq]
                + [extrude(q.buffer(0.15), z1 - sheet - 0.15, z1 + 0.15) for q in hq])
    else:
        lugm = [extrude(q, zm - lug[1] / 2, zm + lug[1] / 2) for q in lugs]
        slot = [extrude(q.buffer(0.15), zm - lug[1] / 2 - 0.15, zm + lug[1] / 2 + 0.15) for q in lugs]
    if lugm:
        band = _bool("union", [band] + lugm)
        cut = {k: _bool("difference", [m] + slot) for k, m in cut.items()}
    segs = sorted([m for m in band.split(only_watertight=True) if m.volume > 0.5], key=lambda m: -m.volume)
    parts, report = {}, {"perimeter": round(L.length, 2), "band_z": [z0, z1], "wall": wall,
                         "splits_mm": [round(v, 2) for v in splits_s], "ties_mm": [round(v, 2) for v in ties_s],
                         "feed_mm": None if feed_s is None else round(feed_s, 2),
                         "air_layers": [[round(a, 3), round(b, 3), round(p.area, 1)] for p, a, b in layers],
                         "parts": {}}
    for k, m in cut.items():
        bs = m.split(only_watertight=False)
        keep_ = [b for b in bs if abs(b.volume) > 1.0]                       # loose pieces of no volume go
        report.setdefault("dropped", {})[k] = len(bs) - len(keep_)
        parts[k] = trimesh.util.concatenate(keep_)
    for i, m in enumerate(segs):
        parts[f"band-{i + 1}"] = m
    for k, m in parts.items():
        report["parts"][k] = {"volume_mm3": round(float(m.volume), 1), "watertight": bool(m.is_watertight),
                              "bodies": len(m.split(only_watertight=False)),
                              "size": [round(float(v), 2) for v in m.extents]}
    report["ok"] = all(v["watertight"] and v["bodies"] == 1 for v in report["parts"].values())
    if check:
        from . import surfacecheck
        r = surfacecheck.compare_exterior(meshes, list(parts.values()))
        report["exterior"] = {"pass": bool(r["pass"]), "maxDeviation": r["maxDeviation"]}
    return {"parts": parts, "layout": L, "report": report}


# ------------------------------------------------------------------ radio --

def band_scene(layout, shells, skin, splits=(), ties=(), feed=None, grounds=(), boxes=(), gap=1.5,
               behind=None, keep=0.8, keep_clear=(), ears=(), screws=True, band=(699, 894), sweep=(600, 2700),
               fine=1.0, extra=()):
    """The band in its product as an `antenna.Scene`.

    `layout`/`shells` as from `cut_band`; `skin` the shells' material (a
    print-card name or {"permittivity", "lossTangent"}). `grounds`: [(outline
    [[x, y]], z_bottom, z_top)] boards whose ground plane the feed and ties
    reach (nearest one); `boxes`: [((lo), (hi))] other metal joined to ground
    (battery, shields, modules). Ties and feed are wires from the band's mid
    height to the nearest board's top. `ears` with `screws`: steel screws
    through both shells at each ear. `extra`: [(kind, geometry, dict)] more
    bodies for add_solid (a hand from `hand_grip`)."""
    from . import antenna as A
    from shapely.geometry import Polygon, Point
    from shapely.ops import nearest_points, unary_union
    L = layout
    z0, z1 = L.z
    zm = 0.5 * (z0 + z1)
    ring = lambda poly: np.asarray(poly.exterior.coords)[:-1].tolist()
    sc = A.Scene(band, sweep=sweep)
    e, td = A.rf_properties(skin)[:2]
    for k, m in shells.items():
        sc.add_solid(k, e, td, mesh=(np.asarray(m.vertices), np.asarray(m.faces)), fine=fine)
    splits_s = [v * L.length for v in splits]
    if behind:
        gr = gap_ring(L, behind, splits_s, gap=gap, keep_clear=keep_clear)
        rim = gr.intersection(L.inside.difference(L.core.buffer(-0.3)))
        for p, za, zb in plan_stack(list(shells.values()), z0, z1, keep, gr):
            p = clean(p.union(rim))
            for g in getattr(p, "geoms", [p]):
                if g.area > 0.5:
                    sc.add_solid("air", 1.0, 0.0, prism=([ring(g)] + [np.asarray(h.coords)[:-1].tolist() for h in g.interiors], za, zb),
                                 curved=True, fine=fine)
    for i, xy in enumerate(ears):
        q = Polygon(ear(L, L.wall, xy).exterior)
        sc.add_solid("band", metal=True, prism=([ring(q)], z0, z0 + 1.5), curved=True, fine=fine, rank=2)
        if screws:
            lo = min(float(m.bounds[0][2]) for m in shells.values())
            sc.add_solid("band", metal=True, box=((xy[0] - 0.8, xy[1] - 0.8, lo), (xy[0] + 0.8, xy[1] + 0.8, z1 + 1.5)))
    for outline, zb, zt in grounds:
        sc.add_solid("ground", metal=True, prism=([outline], zb, zt), curved=True, fine=fine, rank=2)
    for lo, hi in boxes:
        sc.add_solid("ground", metal=True, box=(lo, hi))
    for kind, geo, kw in extra:
        sc.add_solid(**{kind: geo}, **kw)
    cuts = unary_union([L.quad(s, gap, L.wall + 1.0, 6.0) for s in splits_s] + list(keep_clear))
    bandp = L.inside.difference(L.core).difference(cuts)
    for g in getattr(bandp, "geoms", [bandp]):
        sc.add_solid("band", metal=True, prism=([ring(g)] + [np.asarray(h.coords)[:-1].tolist() for h in g.interiors], z0, z1),
                     curved=True, fine=fine, rank=2)
    gpolys = [(Polygon(o).buffer(-1.0), zt) for o, _, zt in grounds]

    def reach(s):
        p, t, n = L.at(s)
        ax = int(np.argmax(np.abs(n)))
        return p, -np.eye(2)[ax] * np.sign(n[ax])

    def to_ground(q):
        if not gpolys:
            raise ValueError("ties and a feed need `grounds` (a board) to reach")
        poly, zt = min(gpolys, key=lambda g_: g_[0].distance(Point(*q)))
        r = np.array(nearest_points(poly, Point(*q))[0].coords[0])
        return [[*q, zm], [r[0], q[1], zm], [*r, zm], [*r, zt - 0.2]]

    for i, v in enumerate(ties):
        p, d = reach(v * L.length)
        sc.add_wire([[*p, zm]] + to_ground(p + d * (L.wall / 2 + 1.0)), f"tie{i}")
    if feed is not None:
        p, d = reach(feed * L.length)
        a = p + d * (L.wall / 2 + 1.0)
        g = a + d * 1.0
        sc.add_wire([[*p, zm], [*a, zm]], "feedpad")
        sc.add_wire(to_ground(g), "feedline")
        sc.set_port([*g, zm], [*a, zm])
    return sc


def hand_grip(shells, side="right", f_mhz=835.0, band_z=6.5, palm=(-62.0, 20.0), palm_t=25.0, fingers=(12.0, -8.0, -28.0, -48.0),
              thumb=-10.0, top=None, f_ref_mhz=None):
    """A hand holding the product in portrait, as tissue bodies for
    `band_scene(extra=...)`: a palm slab against the rear face (z < 0)
    spanning y `palm`, four fingers stood against one long side at y
    `fingers` and the thumb against the other at y `thumb`, all touching
    the skin. Sizes from CTIA 01.72 Table 2.2.3-1 (finger PIP width 21 mm,
    thumb DIP 22.3 mm, hand width 85 mm); tissue at f_mhz from Table
    2.2.4-1, its conductivity held at f_ref_mhz (the Scene's band centre,
    where the solver turns a loss tangent into conductivity; default
    f_mhz). This is a grip built to those sizes, not the CTIA phantom's
    CAD, so it compares designs; certification uses the phantom itself.
    side "right": fingers on -x, thumb on +x."""
    from shapely.geometry import LineString
    meshes = list(shells.values()) if isinstance(shells, dict) else list(shells)
    env = envelope(meshes, band_z)
    eps, sig = hand_tissue(f_mhz)
    tan = sig / (2 * math.pi * (f_ref_mhz or f_mhz) * 1e6 * 8.8541878128e-12 * eps)
    top = top if top is not None else max(float(m.bounds[1][2]) for m in meshes) - 6.0
    sgn = -1.0 if side == "right" else 1.0
    fd, td_ = HAND_SIZE["finger_pip_width"], HAND_SIZE["thumb_dip_width"]
    hw = HAND_SIZE["hand_width"] / 2

    def side_x(y, s):
        ln = LineString([(-200, y), (200, y)]).intersection(env)
        geoms = [ln] if ln.geom_type == "LineString" else list(ln.geoms)
        xs = np.concatenate([np.asarray(g.coords)[:, 0] for g in geoms])
        return xs.max() if s > 0 else xs.min()

    def disc(x, y, d):
        return [[x + d / 2 * math.cos(t), y + d / 2 * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]

    kw = {"name": "hand", "eps_r": eps, "tan_d": tan}
    out = [("box", ((-hw, palm[0], -palm_t - 0.3), (hw, palm[1], -0.3)), kw)]
    for y in fingers:
        x = side_x(y, sgn) + sgn * fd / 2
        out.append(("prism", (disc(x, y, fd), -palm_t, top), kw))
    x = side_x(thumb, -sgn) - sgn * td_ / 2
    out.append(("prism", (disc(x, thumb, td_), -palm_t, top), kw))
    return out
