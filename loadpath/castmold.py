"""
Molds and patterns for a cast part, as meshes that can be printed.

What gets made is whatever the process pulls the part out of, at the size
the physics says it has to be:

- **Slip casting** (`slip_mold`): a two-piece plaster mold. The cavity is
  the fired part scaled up by the drying and firing shrinkage
  (slipcast.shrinkage). The mold parts along whichever sideways direction
  leaves the least undercut, at the level along it that does (counted on
  the cavity's voxels: plaster that would have to pass through the part to
  come away). Its walls are as thick as the water of the planned number of
  casts needs (slipcast.mold_wetting). The pour opening continues up
  through the top as the reservoir. Each half comes with its *casting
  box*: the printable negative you pour plaster into to make that half
  (the half part standing on the parting face, walls round it, and the
  keys that register the halves).
- **Sand and permanent-mold casting** (`metal_pattern`): the pattern is
  the part scaled up by the alloy's contraction from the solidus to room
  temperature (casting.contraction), with the risers the solidification
  solve sized, a sprue of the flow area the pour needs and an ingate wide
  enough to enter under the critical velocity (casting.gating). It is cut
  at the parting plane into cope and drag halves on plates, and the mold
  halves themselves are written too (for a printed sand mold).
- **Investment and plaster-investment casting**: no parting, no draft; the
  pattern is burnt out. One printable tree: part, risers and sprue.

The sprue's and gate's *sizes* come from the flow; their shapes (a tapered
round sprue, a flat gate four times as wide as it is thick, hemispherical
keys) are the conventional ones and are stated as such in the result.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

__all__ = ["parting", "slip_mold", "cast_body", "metal_pattern", "write_meshes"]


def _box(lo, hi):
    import trimesh

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation(0.5 * (lo + hi))
    return b


def _bool(op, meshes):
    import trimesh

    meshes = [m for m in meshes if m is not None and len(m.faces)]
    if len(meshes) == 1:
        return meshes[0]
    fn = {"union": trimesh.boolean.union, "difference": trimesh.boolean.difference,
          "intersection": trimesh.boolean.intersection}[op]
    return fn(meshes, engine="manifold")


def _axis_vec(axis, sign=1):
    v = np.zeros(3)
    v[axis] = sign
    return v


def _cyl(p0, p1, r0, r1=None, sections=48):
    """A (tapered) cylinder from p0 to p1, radius r0 at p0 and r1 at p1."""
    import trimesh

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    h = float(np.linalg.norm(p1 - p0))
    r1 = r0 if r1 is None else r1
    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring0 = np.stack([r0 * np.cos(ang), r0 * np.sin(ang), np.zeros(sections)], 1)
    ring1 = np.stack([r1 * np.cos(ang), r1 * np.sin(ang), np.full(sections, h)], 1)
    v = np.vstack([ring0, ring1, [[0, 0, 0]], [[0, 0, h]]])
    f = []
    for i in range(sections):
        j = (i + 1) % sections
        f += [[i, j, sections + j], [i, sections + j, sections + i], [2 * sections, j, i],
              [2 * sections + 1, sections + i, sections + j]]
    m = trimesh.Trimesh(v, np.array(f), process=True)
    m.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], (p1 - p0) / h))
    m.apply_translation(p0)
    return m


def parting(solid, axes=(0, 1, 2)) -> dict:
    """Where a two-piece mold of this cavity parts with the least undercut.

    For a draw along an axis and a parting level p, the half on the far
    side pulls away along +axis: any mold cell there with cavity further
    along the same line would have to pass through the part. The same on
    the near side, the other way. Returns {"axis", "level" (cells from the
    grid's low face), "undercut" (cells), "share" (of the cavity)} for the
    axis and level with the fewest such cells; ties go to the level
    nearest the widest section."""
    solid = np.asarray(solid, bool)
    best = None
    for ax in axes:
        s = np.moveaxis(solid, ax, -1)
        n = s.shape[-1]
        # cavity cells at or beyond each position (from the high end), and at or before (from the low end)
        beyond = np.cumsum(s[..., ::-1], axis=-1)[..., ::-1] > 0
        before = np.cumsum(s, axis=-1) > 0
        up_block = (~s & beyond)                    # mold cells with cavity further along +axis
        dn_block = (~s & before)                    # mold cells with cavity further along -axis
        up_cnt = up_block.sum(axis=tuple(range(s.ndim - 1)))
        dn_cnt = dn_block.sum(axis=tuple(range(s.ndim - 1)))
        above = np.concatenate([np.cumsum(up_cnt[::-1])[::-1], [0]])     # blocked cells at index >= p
        below = np.concatenate([[0], np.cumsum(dn_cnt)])                 # blocked cells at index < p
        total = above + below                                            # for p in 0..n
        area = s.sum(axis=tuple(range(s.ndim - 1))).astype(float)
        wide = np.nonzero(area >= 0.98 * area.max())[0].mean()          # the middle of the widest run
        cand = np.nonzero(total == total.min())[0]
        p = int(cand[np.argmin(np.abs(cand - (wide + 0.5)))])
        row = {"axis": int(ax), "level": p, "undercut": int(total.min()),
               "share": float(total.min() / max(solid.sum(), 1))}
        if best is None or row["undercut"] < best["undercut"]:
            best = row
    return best


def _keys(face_lo, face_hi, axis, level, radius, inset):
    """Hemispherical registration keys at the corners of the parting face."""
    import trimesh

    lat = [d for d in range(3) if d != axis]
    out = []
    for a in (face_lo[lat[0]] + inset, face_hi[lat[0]] - inset):
        for b in (face_lo[lat[1]] + inset, face_hi[lat[1]] - inset):
            c = np.zeros(3)
            c[axis] = level
            c[lat[0]], c[lat[1]] = a, b
            s = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            s.apply_translation(c)
            out.append(s)
    return out


def cast_body(mesh, fields):
    """The slip-cast part as fired: the drawn outside with the inside the
    cast leaves (the drained core cut out of it, open at the pour opening).
    `fields` is slipcast_part's."""
    import trimesh
    from skimage import measure

    c = fields["cast"]
    if c.drain_time is None:
        return mesh.copy()
    p = fields["pitch"]
    org = np.asarray(fields["origin"], float)
    ax, sg = 2, 1
    for d in range(3):                                # the opening's side
        idx = np.argwhere(c.opening)
        if len(idx) and (idx[:, d].min() == idx[:, d].max()):
            ext = np.argwhere(c.cavity)[:, d]
            if idx[0, d] == ext.max():
                ax, sg = d, 1
            elif idx[0, d] == ext.min():
                ax, sg = d, -1
    # the cake's front where it stood when the mold was drained: the arrival time is smooth across
    # cells (the cells' fill shares are not), in sqrt(time), which is distance from the mold
    arr = np.where(np.isfinite(c.arrival), c.arrival, np.inf)
    # a cell's arrival is the front reaching its far face: half a cell past its centre
    depth = np.sqrt(np.minimum(arr, 4.0 * c.drain_time) / c.drain_time) - 1.0 - 0.5 * c.pitch / c.nominal
    core = np.where(c.cavity & ~c.trapped, depth, -1.0)
    core = 0.5 + 0.5 * np.clip(core, -1.0, 1.0)        # 1 in the drained core, 0.5 on the front
    pad = 4
    core = np.pad(core, pad)
    op = np.pad(c.opening, pad)
    # carry the core up through the opening so the cut opens the top
    col = np.moveaxis(core, ax, -1)
    opc = np.moveaxis(op & (core > 0.5), ax, -1)
    if sg < 0:
        col, opc = col[..., ::-1], opc[..., ::-1]
    top = opc.any(axis=-1)
    kmax = np.where(opc, np.arange(opc.shape[-1]), -1).max(axis=-1)
    for (i, j) in np.argwhere(top):
        col[i, j, kmax[i, j]:-1] = 1.0
    core = ndimage.gaussian_filter(core, 0.6)
    if core.max() < 0.5:
        return mesh.copy()
    v, f, _, _ = measure.marching_cubes(core, 0.5)
    cm = trimesh.Trimesh(org + (v - pad + 0.5) * p, f[:, ::-1], process=True)
    if cm.volume < 0:
        cm.invert()
    # the grid's steps (the mold face is cells) are not the cake's: take them out, volume kept
    trimesh.smoothing.filter_taubin(cm, lamb=0.5, nu=0.53, iterations=30)
    return _bool("difference", [mesh, cm])


def slip_mold(mesh, report, fields, up=(2, 1), casts: int = 5, min_wall: float = 20.0) -> dict:
    """A two-piece plaster mold for a slip-cast part and the printable boxes
    to cast its halves in. Returns {"meshes": {name: mesh}, "info": {...}}.

    mesh: the fired part's outside. report/fields: slipcast_part's.
    The mold wall is the larger of `min_wall` and the plaster the water of
    `casts` casts fills, doubled (report["mold"])."""
    scale = fields["scale"]
    ax_up, sg_up = up
    lat = [d for d in range(3) if d != ax_up]
    pt = parting(fields["cavity"], axes=lat)
    ax = pt["axis"]
    p = fields["pitch"]
    org = np.asarray(fields["origin"], float)
    part = mesh.copy()
    part.apply_scale(scale)
    level = (org[ax] + pt["level"] * p) * scale
    wet = report["mold"]["wetDepthPerCast"]
    wall = max(float(min_wall), 2.0 * wet * casts)
    lo, hi = part.bounds
    blo, bhi = lo - wall, hi + wall
    # the reservoir: the part's outline just under its top, carried up through the top of the mold
    top = hi[ax_up] if sg_up > 0 else lo[ax_up]
    spare_len = wall
    depth = min(0.04 * (hi[ax_up] - lo[ax_up]), 2.0)
    # the part's section just under its top, extruded up through the top of the mold
    band_lo, band_hi = blo.copy(), bhi.copy()
    if sg_up > 0:
        band_lo[ax_up], band_hi[ax_up] = top - depth, top + 1.0
    else:
        band_lo[ax_up], band_hi[ax_up] = top - 1.0, top + depth
    band = _bool("intersection", [part, _box(band_lo, band_hi)])
    spare = None
    if band is not None and len(band.faces):
        sec = band.section(plane_origin=_axis_vec(ax_up) * (top - sg_up * 0.5 * depth),
                           plane_normal=_axis_vec(ax_up))
        if sec is not None:
            import trimesh

            planar, to3d = sec.to_planar()
            polys = planar.polygons_full
            if len(polys):
                pieces = []
                for poly in polys:
                    ex = trimesh.creation.extrude_polygon(poly, spare_len + 1.0 + 0.5 * depth)
                    ex.apply_transform(to3d)
                    pieces.append(ex)
                spare = _bool("union", pieces)
                # extrude_polygon builds along the section's own normal: make sure it goes up
                c = spare.bounds.mean(0)[ax_up]
                if (c - top) * sg_up < 0:
                    spare.apply_translation(_axis_vec(ax_up, sg_up) * (spare_len + 1.0 + 0.5 * depth))
    cavity = _bool("union", [part, spare]) if spare is not None else part
    block = _box(blo, bhi)
    mold = _bool("difference", [block, cavity])
    a_lo, a_hi = blo.copy(), bhi.copy()
    a_lo[ax] = level
    b_lo, b_hi = blo.copy(), bhi.copy()
    b_hi[ax] = level
    key_r = min(0.3 * wall, 8.0)
    keys = _keys(blo, bhi, ax, level, key_r, 0.5 * wall)
    keys = [k for k in keys if not _touches(k, cavity)]
    half_a = _bool("difference", [_bool("intersection", [mold, _box(a_lo, a_hi)])] + keys)     # sockets
    half_b = _bool("union", [_bool("intersection", [mold, _box(b_lo, b_hi)])] + keys)          # keys
    # casting boxes: the half part on a floor at the parting face, walls round it; plaster poured in
    # makes the half. Box A makes half A (which has sockets), so box A carries key bumps; box B's
    # floor has the dimples that mold half B's keys.
    t = 2.4                                                     # printed wall of the box, mm
    meshes = {"mold-half-a": half_a, "mold-half-b": half_b}
    for name, side in (("a", 1), ("b", -1)):
        h_lo, h_hi = (a_lo, a_hi) if side > 0 else (b_lo, b_hi)
        outer_lo, outer_hi = h_lo - t, h_hi + t
        if side > 0:
            outer_hi[ax] = h_hi[ax]                             # open where the plaster goes in
            outer_lo[ax] = level - t
        else:
            outer_lo[ax] = h_lo[ax]
            outer_hi[ax] = level + t
        shell = _bool("difference", [_box(outer_lo, outer_hi), _box(h_lo, h_hi)])
        half_cav = _bool("intersection", [cavity, _box(h_lo, h_hi)])
        box = _bool("union", [shell, half_cav] + (keys if side > 0 else []))
        if side < 0 and keys:
            box = _bool("difference", [box] + keys)
        meshes[f"mold-box-{name}"] = box
    info = {"scale": scale, "partingAxis": "xyz"[ax], "partingLevel": level, "undercutShare": pt["share"],
            "wall": wall, "casts": casts, "keys": len(keys), "keyRadius": key_r,
            "plasterVolume": float(half_a.volume + half_b.volume) * 1e-6,           # litres
            "conventions": "hemispherical keys and a straight reservoir are conventional shapes; the scale, the "
                           "parting and the wall come from the physics"}
    if pt["share"] > 0.002:
        info["warning"] = (f"{100 * pt['share']:.1f}% of the cavity is undercut along {'xyz'[ax]} whatever the "
                           "parting level: this shape needs a third mold piece or a different split")
    return {"meshes": meshes, "info": info}


def _touches(a, b):
    try:
        m = _bool("intersection", [a, b])
        return m is not None and len(m.faces) > 0 and abs(m.volume) > 1e-6
    except Exception:  # noqa: BLE001
        return False


def metal_pattern(mesh, report, s, plate: float = 6.0) -> dict:
    """Pattern (and mold) meshes for a metal casting from `castability`'s
    report: the part at pattern scale with its risers, sprue and ingate.

    Sand and permanent mold: cope and drag pattern halves on plates, and
    the two mold halves. Investment and plaster: one tree to print and burn
    out. Returns {"meshes": {...}, "info": {...}}."""
    import trimesh

    k = report["contraction"]["patternScale"]
    ax, sg = s.up
    lat = [d for d in range(3) if d != ax]
    part = mesh.copy()
    part.apply_scale(k)
    pieces = [part]
    info = {"scale": k, "process": s.process}
    top_of_all = part.bounds[1][ax] if sg > 0 else part.bounds[0][ax]
    for r in report.get("risers", []):
        base = np.asarray(r["position"], float) * k
        d = r["diameter"] * k
        h = r["height"] * k
        p0 = base - _axis_vec(ax, sg) * 0.15 * d                 # sunk into the part so the union is clean
        p1 = base + _axis_vec(ax, sg) * h
        pieces.append(_cyl(p0, p1, 0.5 * d))
        top_of_all = max(top_of_all, p1[ax]) if sg > 0 else min(top_of_all, p1[ax])
    g = report.get("gating")
    gate_at = np.asarray(report["fill"]["gate"], float) * k
    lo, hi = part.bounds
    from .verify import voxelize

    two_part = s.process not in ("investment", "plaster")
    if two_part:
        # the metal enters at the parting plane: the gate is cut into the mold's parting face
        # (read on the part and its risers: the sprue is formed by a pin drawn out of the top of the cope)
        body = _bool("union", list(pieces))
        pitch = max(float(np.max(body.extents)) / 160.0, 0.4)
        solid, org = voxelize(body, pitch)
        pt = parting(solid, axes=(ax,))
        level = org[ax] + pt["level"] * pitch
        gate_at[ax] = level
    if g:
        # the sprue stands outside the part, on the side the gate point is nearest; the gate is a flat
        # bar from its foot to the part at the gate point
        c = 0.5 * (lo + hi)
        side = max(lat, key=lambda d: abs(gate_at[d] - c[d]) / max(hi[d] - lo[d], 1e-9))
        sgn = 1.0 if gate_at[side] >= c[side] else -1.0
        r_exit = 0.5 * g["sprueExitDiameter"]
        r_top = 0.5 * g["sprueTopDiameter"]
        gap = max(3.0 * r_top, 12.0)
        foot = gate_at.copy()
        foot[side] = (hi[side] if sgn > 0 else lo[side]) + sgn * gap
        area = g["ingateArea"]
        thick = math.sqrt(area / 4.0)
        width = 4.0 * thick
        foot[ax] = gate_at[ax]
        s_top = foot.copy()
        if sg > 0:
            s_top[ax] = max(top_of_all, foot[ax] + 10.0) + 5.0
        else:
            s_top[ax] = min(top_of_all, foot[ax] - 10.0) - 5.0
        pieces.append(_cyl(foot - _axis_vec(ax, sg) * thick, s_top, r_exit, r_top))
        other = [d for d in lat if d != side][0]
        b_lo, b_hi = gate_at.copy(), gate_at.copy()
        b_lo[side], b_hi[side] = sorted((gate_at[side] - sgn * 0.5 * thick, foot[side]))
        b_lo[other] -= 0.5 * width
        b_hi[other] += 0.5 * width
        if two_part:                                  # in the drag, its top on the parting face
            if sg > 0:
                b_lo[ax] -= thick
            else:
                b_hi[ax] += thick
        else:
            b_lo[ax] -= 0.5 * thick
            b_hi[ax] += 0.5 * thick
        pieces.append(_box(b_lo, b_hi))
        info["sprue"] = {"exitDiameter": 2 * r_exit, "topDiameter": 2 * r_top,
                         "height": abs(s_top[ax] - foot[ax]), "position": [float(v) for v in foot]}
        info["ingate"] = {"area": area, "width": width, "thickness": thick}
        info["conventions"] = ("round tapered sprue and a flat ingate 4:1 are conventional shapes; their areas "
                               "come from the pour (casting.gating)")
    tree = _bool("union", pieces)
    meshes = {"casting-tree": tree}
    if s.process in ("investment", "plaster"):
        info["note"] = "print the tree (castable resin, wax or PLA), invest it, burn it out, pour"
        return {"meshes": meshes, "info": info}
    # two-part mold along `up`: parting where the undercut is least
    tlo, thi = tree.bounds
    margin = max(25.0, 0.25 * float(np.max(thi - tlo)))
    blo, bhi = tlo - margin, thi + margin
    # the sprue and open risers come out of the top of the cope
    if sg > 0:
        bhi[ax] = thi[ax] - 1.0
    else:
        blo[ax] = tlo[ax] + 1.0
    block = _box(blo, bhi)
    mold = _bool("difference", [block, tree])
    up_lo, up_hi = blo.copy(), bhi.copy()
    dn_lo, dn_hi = blo.copy(), bhi.copy()
    up_lo[ax] = level
    dn_hi[ax] = level
    cope_box, drag_box = (_box(up_lo, up_hi), _box(dn_lo, dn_hi)) if sg > 0 else (_box(dn_lo, dn_hi),
                                                                                    _box(up_lo, up_hi))
    meshes["mold-cope"] = _bool("intersection", [mold, cope_box])
    meshes["mold-drag"] = _bool("intersection", [mold, drag_box])
    # pattern halves on plates
    for name, bx, side in (("pattern-cope", cope_box, sg), ("pattern-drag", drag_box, -sg)):
        half = _bool("intersection", [tree, bx])
        p_lo, p_hi = blo.copy(), bhi.copy()
        if side > 0:
            p_lo[ax], p_hi[ax] = level - plate, level
        else:
            p_lo[ax], p_hi[ax] = level, level + plate
        meshes[name] = _bool("union", [half, _box(p_lo, p_hi)]) if half is not None and len(half.faces) \
            else _box(p_lo, p_hi)
    info["partingLevel"] = float(level)
    info["partingAxis"] = "xyz"[ax]
    info["undercutShare"] = pt["share"]
    if pt["share"] > 0.002:
        info["warning"] = (f"{100 * pt['share']:.1f}% of the casting is undercut along the draw: it needs a core, "
                           "or `casting.draw` in the study that designs it")
    return {"meshes": meshes, "info": info}


def write_meshes(meshes: dict, out_dir: str) -> dict:
    """Write each mesh as <name>.stl under out_dir; returns {name: path}."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, m in meshes.items():
        if m is None or not len(m.faces):
            continue
        path = os.path.join(out_dir, f"{name}.stl")
        m.export(path)
        paths[name] = path
    return paths
