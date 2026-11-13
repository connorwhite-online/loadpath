"""
A product from its outside in: the designer's exterior is fixed, and
everything inside it is engineering.

An industrial designer draws the shape and the surfaces people see and
touch. What goes inside (a battery, a board and its connector, a motor, a
switch) and how it is held, how it survives a drop, how it feels in the
hand, is invisible, and it is what turns a rendering into a product that
can be built. `loadpath enclosure` does that part from one brief:

1. **placement** (placement.py): where each component sits inside the
   shell, unless the designer gave it: orientations and positions searched
   on the cavity, rules such as a connector out through its port, a
   battery on the floor, a switch under the spot a finger presses, a motor
   near the board, the centre of mass where it should be;
2. **assembly** (handoff.assembly_paths): the way each component goes in, in
   assembly order, through the openings the exterior has, with nothing in
   the way: each becomes a keep-out swept along that path, so no generated
   material blocks it;
3. **mounts**: a cradle under each component with a mass, preserved, so its
   inertia has somewhere to go;
4. **loads**: any mix of drops (each component's mass at the design
   deceleration, toward each face the product can land on, held off by the
   floor on that face), squeezes, presses and the study's own load cases;
   watts per component as heat; a `switch` block (a push pad's click force,
   via handoff) and a `feel` block (grip squeeze, drop cushioning, noise)
   pass through;
5. **the study**: an ordinary loadpath study, with the exterior kept as its
   `skin` (preserved, never edited) and the whole interior as design space.

`--run` then designs it (`run` for strength and stiffness; `switch` or
`feel` when those blocks are present, with the drop and handling cases
checked on the delivered part) and checks what matters for the designer
on the part as delivered, not the optimizer's grid:

- **the exterior is untouched**: the delivered surface sampled against the
  exterior (nothing proud of it, every bit of it still there, openings
  aside);
- the strength checks of `verify` (body-fitted cut cells or tets) per load
  case, and that no component's keep-out or path is intruded on.

Nothing here edits the exterior. A component with no place, or no way in,
is reported, with where it came closest, for the designer to resolve.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .handoff import _axis_name, _axis_vec, _fine_mesh_shape, _shape_spec

__all__ = ["enclosure", "run_enclosure", "exterior_check", "check_delivered", "design_and_check"]

_FACES = ["-z", "+z", "-x", "+x", "-y", "+y"]


def _inner_centre(exterior, lo, hi, n: int = 28):
    """The middle of the exterior's box, or where that is outside the
    exterior (a tool's barrel over its handle: the box's middle is in the
    air in front of the handle), the point inside nearest to it among the
    deeper half of the interior. Supports and squeezes are found from here
    by rays."""
    c = 0.5 * (lo + hi)
    if exterior.sdf(c[None])[0] < 0:
        return c
    ax = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    d = -exterior.sdf(g)
    if not (d > 0).any():
        return c
    deep = g[d >= 0.5 * d.max()]
    return deep[int(np.argmin(np.linalg.norm(deep - c, axis=1)))]


def _surface_point(exterior, centre, d, reach=None):
    """Where a ray from `centre` along d leaves the exterior."""
    lo, hi = exterior.bounds()
    reach = reach or float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
    t = np.arange(0.0, reach, 0.05)
    pts = centre[None, :] + t[:, None] * d[None, :]
    out = exterior.sdf(pts) > 0
    if not out.any():
        return centre + reach * d
    return pts[int(np.argmax(out))]


def _r(v, n=4):
    return np.round(np.asarray(v, float), n).tolist()


def _write_mesh(mesh, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)
    return path


def _cradle(bounds, side: np.ndarray, thickness: float, ext_spec) -> dict:
    """A slab under a component's face on `side`, clipped to the exterior."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    k = int(np.argmax(np.abs(side)))
    blo, bhi = lo.copy(), hi.copy()
    if side[k] < 0:
        blo[k], bhi[k] = lo[k] - thickness, lo[k] + 0.05
    else:
        blo[k], bhi[k] = hi[k] - 0.05, hi[k] + thickness
    return {"type": "intersect", "shapes": [{"type": "box", "min": _r(blo), "max": _r(bhi)}, ext_spec]}


def enclosure(brief: dict, base_dir: str | None, out_dir: str, on_progress=None) -> dict:
    """Brief in; a study, a placement and a report out (files under
    out_dir: cad/<component>.stl in product coordinates, the study). See the
    module docstring and `loadpath schema enclosure`."""
    from .placement import place
    from .shapes import MeshShape, shape_from_spec, trimesh_from_manifold
    from .spec import _skin

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    # a KiCad board stands for several components: the board with the
    # parts that stay on it, and its connectors and hot parts on it
    from .kicad import expand_brief
    b, boards = expand_brief(brief, base_dir, out_dir)
    cad = os.path.join(out_dir, "cad")
    ext_spec = _shape_spec(b["exterior"], base_dir)
    exterior = _fine_mesh_shape(ext_spec, None)
    lo, hi = exterior.bounds()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    centre = _inner_centre(exterior, lo, hi)
    skin_v = b.get("skin", 1.0)
    sk = dict(skin_v) if isinstance(skin_v, dict) else {"thickness": float(skin_v)}
    t_skin = float(sk["thickness"])
    min_member = float(b.get("minMember", max(t_skin, 1.0)))

    # openings: where the exterior opens into the interior (a port, an
    # underside a flap closes). No skin across their inner ends; a part may
    # reach into one (a connector's mouth).
    ops, op_place = [], []
    for j, o in enumerate(b.get("openings") or []):
        o = {"shape": o} if not isinstance(o, dict) else dict(o)
        oid = o.pop("id", None) or f"opening-{j}"
        o["shape"] = _shape_spec(o["shape"], base_dir)
        osh = shape_from_spec(o["shape"])
        olo, ohi = osh.bounds()
        if not o.get("inward"):
            v = centre - 0.5 * (np.asarray(olo) + np.asarray(ohi))
            o["inward"] = _axis_name(v)
        o.setdefault("depth", round(t_skin + 0.2, 3))
        ops.append(o)
        # a part aligned to the opening is centred on the port the exterior
        # draws there, not on the opening's box (spec.drawn_aperture)
        from .spec import _leg_direction, drawn_aperture
        ap = drawn_aperture(exterior, osh, _leg_direction(o["inward"], f"openings[{j}].inward"), float(o["depth"]))
        op_place.append({"id": oid, "shape": ap if ap is not None else osh, "inward": o["inward"],
                         "depth": float(o["depth"])})
    if ops:
        sk["openings"] = list(sk.get("openings") or []) + ops
    _, skin_shape = _skin(sk, [exterior])

    fixed, preserve = [], []
    for i, f in enumerate(b.get("fixed") or []):
        fid = f.get("id") or f"fixed-{i}"
        spec = _shape_spec(f["shape"], base_dir)
        fixed.append((fid, _fine_mesh_shape(spec, None)))
        preserve.append({"id": fid, "shape": spec})

    comps = []
    for i, c in enumerate(b.get("components") or []):
        c = dict(c)
        c["id"] = c.get("id") or f"component-{i}"
        if isinstance(c.get("shape"), str):
            c["shape"] = _shape_spec(c["shape"], base_dir)
        if c.get("antenna") and c.get("shape") is None and c.get("size") is None:
            # a board described by its antenna brief is a box that size
            ab = c["antenna"]["board"]
            c["size"] = [*(ab.get("size") or np.ptp(np.asarray(ab["outline"], float), axis=0).tolist()),
                         float(ab.get("thickness", 1.6))]
        comps.append(c)
    byid = {c["id"]: c for c in comps}

    # 1. placement
    pl_opts = dict(b.get("placement") or {})
    order = list(b.get("assembly") or [c["id"] for c in comps if not c.get("installedOn")])
    order += [c["id"] for c in comps if not c.get("installedOn") and c["id"] not in order]
    unknown = [i for i in order if i not in byid]
    if unknown:
        raise ValueError(f"enclosure.assembly names {unknown}, which are not components ({list(byid)})")
    skin_grid = _GridShape(skin_shape)
    fixed_grids = [(f, _GridShape(sh)) for f, sh in fixed]
    tol = float(b.get("pathTolerance", 0.25))

    def tilt_of(cid, quick=False):
        """How far this part may be turned on its way in (degrees; the
        brief's `tilt`, or the component's own): None for slides only."""
        v = byid[cid].get("tilt", b.get("tilt", 45.0))
        v = 0.0 if v is False else 45.0 if v is True else float(v)
        if v <= 0:
            return None
        # placement asks for every candidate position: a coarser grid there
        return {"max": v, **({"cells": 600_000, "minStep": 2.0, "tries": ((False, 0.0), (True, 0.0))} if quick else {})}
    rank = {gid: i for i, gid in enumerate(order)}

    def access(gid, sdf, bounds, placed, turning=False):
        """Can this part still get to this position, past the skin, the
        fixed geometry and the parts that go in before it?"""
        if byid[gid].get("external") or (byid[gid].get("fixed") and byid[gid].get("insertFrom")):
            return True
        earlier = [v[0] for k, v in placed.items() if rank.get(k, 1e9) < rank.get(gid, 1e9)]
        cuts = [sdf] + [v[0] for v in placed.values()]

        def obstacles(p):
            c = np.min([fn(p) for fn in cuts], axis=0)
            o = np.full(len(p), np.inf)
            for _, sh in [("skin", skin_grid)] + fixed_grids:
                o = np.minimum(o, np.maximum(sh.sdf(p), -c))
            for fn in earlier:
                o = np.minimum(o, fn(p))
            return o
        hint = byid[gid].get("insertFrom") if isinstance(byid[gid].get("insertFrom"), str) else None
        return _way_in(sdf, bounds, None, obstacles, (lo, hi), float(byid[gid].get("pathTolerance", tol)),
                       hint, tilt_of(gid, quick=True) if turning else None)["insertFrom"] is not None

    # a split exterior (shells.py) is assembled with its cover off: its own
    # pass finds the paths into the base and sets insertFrom
    split = b.get("split")
    judge = None
    if b.get("wires"):
        # what joins the parts decides the layout too: every candidate
        # position's cables are routed and laid (wires.py)
        from .wires import placement_judge
        wo = pl_opts.pop("wires", None) or {}
        judge = placement_judge(b["wires"], comps, weight=float(wo.get("weight", 0.15)),
                                physics=bool(wo.get("physics", True)))
    # an antenna on a board: the bare board's fields are solved once, and
    # every candidate layout is priced by what the wall and the other parts
    # would do to its port (antenna.py)
    antennas = []
    if any(c.get("antenna") for c in comps):
        from .antenna import both_judges, enclosure_antennas
        if on_progress:
            on_progress({"stage": "antenna", "message": "solving the bare board's fields"}, None)
        antennas, aj = enclosure_antennas(comps, skin_shape, t_skin, b.get("material"), pl_opts, base_dir,
                                          log=(lambda m: on_progress({"stage": "antenna", "message": m}, None))
                                          if on_progress else None)
        judge = both_judges(judge, aj)
    aware = pl_opts.get("assemblyAware", True) and not split

    def lay_out(turning):
        return place(exterior, comps, skin=t_skin, openings=op_place, fixed=fixed, rules=pl_opts, judge=judge,
                     pitch=pl_opts.get("pitch"), beam=int(pl_opts.get("beam", 3)),
                     margin=float(pl_opts.get("margin", 0.5)), on_progress=on_progress,
                     access=(lambda *a: access(*a, turning=turning)) if aware else None)

    # straight ways in first: a layout every part slides into is the one
    # to have. Only when some part has no place it can slide into are
    # positions it must be turned to reach let in
    pr = lay_out(False)
    if aware and (pr["best"] is None or pr["blocked"]) and any(tilt_of(c["id"]) for c in comps):
        if on_progress:
            on_progress({"stage": "placement", "message": "no layout by straight slides: letting parts turn"})
        again = lay_out(True)
        if again["best"] is not None and (pr["best"] is None or len(again["blocked"]) < len(pr["blocked"])):
            pr = again
    if pr["best"] is None:
        raise ValueError(f"enclosure: no layout fits every component; blocked: {pr['blocked']}")

    def root(cid):
        c = byid[cid]
        while c.get("installedOn"):
            c = byid[c["installedOn"]]
        return c["id"]

    rest = _axis_vec(b.get("rest", "-z"))

    # 2-3. per layout (best first): assembly paths, then mounts where they
    # block no path; the first layout every component can go into wins
    tried = []
    chosen = None
    a_order = [o for o in order if not byid[o].get("external")]
    for li, lay in enumerate(pr["layouts"][:2]):
        comp_shapes = {cid: _GridShape(MeshShape(m)) for cid, m in lay["meshes"].items()}
        if split:
            paths = {o: {"insertFrom": None, "how": "split: set by shells.apply_split"} for o in a_order
                     if not byid[o].get("installedOn")}
            mounts, mwarn = _mounts(comps, byid, root, lay, {}, comp_shapes, exterior, ext_spec, min_member, t_skin)
            chosen = (lay, mounts, paths, [], mwarn)
            break
        paths = _paths(a_order, byid, root, comp_shapes, lay, [("skin", skin_grid)] + fixed_grids, [], (lo, hi),
                       tol, on_progress, tilt_of)
        mounts, mwarn = _mounts(comps, byid, root, lay, paths, comp_shapes, exterior, ext_spec, min_member, t_skin)
        if mounts:
            # the cradles are printed: every path is checked again with them in
            mount_grids = [(f"mount-{k}", _GridShape(shape_from_spec(v["shape"]))) for k, v in mounts.items()]
            paths = _paths(a_order, byid, root, comp_shapes, lay,
                           [("skin", skin_grid)] + fixed_grids + mount_grids, [], (lo, hi), tol, on_progress,
                           tilt_of)
        blocked = [k for k, v in paths.items() if v.get("insertFrom") is None]
        tried.append({"layout": li, "cost": lay["cost"], "blocked": blocked})
        if chosen is None or len(blocked) < len(chosen[3]):
            chosen = (lay, mounts, paths, blocked, mwarn)
        if not blocked:
            break
    lay, mounts, paths, blocked, mwarn = chosen
    warnings = list(mwarn)
    for k in blocked:
        warnings.append(f"{k}: no clear way in, in up to four straight legs"
                        + (f" or turning up to {tilt_of(k)['max']:g}°" if tilt_of(k) else "")
                        + "; its keep-out covers where it sits "
                        "only. The exterior needs an opening (or a split), or the part moves")
    for k, v in pr["blocked"].items():
        warnings.append(f"{k}: no place found inside the exterior ({v['reason']})")

    # write the placed components
    comp_specs = {}
    for cid, m in lay["meshes"].items():
        p = _write_mesh(m, os.path.join(cad, f"{cid}.stl"))
        comp_specs[cid] = {"type": "mesh", "path": os.path.abspath(p)}

    def path_of(cid):
        return paths.get(root(cid), {}).get("insertFrom")

    keep = []
    for c in [c for c in comps if c["id"] in comp_specs]:
        k = {"id": c["id"], "bore": False, "shape": comp_specs[c["id"]]}
        ins = path_of(c["id"])
        if ins is not None and not c.get("external"):
            k["insertFrom"] = ins
        keep.append(k)
    keep += list(b.get("keepOut") or [])
    for k, v in mounts.items():
        preserve.append({"id": f"mount-{k}", "shape": v["shape"]})
        # a board on screws: the taps, shanks, heads and drivers kept clear
        keep += v.get("keepOut") or []
    # the cables: laid in this layout, followed along each part's way in,
    # and their channels kept clear of the interior
    wires_report = None
    if b.get("wires"):
        from .wires import enclosure_wires
        wb = {"skin": (skin_shape, None)}
        wb.update({f: (sh, None) for f, sh in fixed})
        wb.update({cid: (comp_shapes[cid], lay["meshes"][cid]) for cid in comp_specs})
        wb.update({f"mount-{k}": (shape_from_spec(v["shape"]), None) for k, v in mounts.items()})
        try:
            wb["skin"] = (skin_shape, trimesh_from_manifold(skin_shape.to_manifold()))
        except Exception:
            pass
        wr = enclosure_wires(b, comps, lay, paths, root, wb, order, (lo, hi), out_dir, on_progress=(
            (lambda ev: on_progress(ev)) if on_progress else None))
        keep += wr["keepOut"]
        warnings += wr["warnings"]
        wires_report = wr["report"]

    # the antenna as installed: the chosen layout solved in full
    antenna_report = None
    if antennas:
        if on_progress:
            on_progress({"stage": "antenna", "message": "solving the antenna in the chosen layout"}, None)
        skin_mesh = trimesh_from_manifold(skin_shape.to_manifold())
        antenna_report = {}
        for a in antennas:
            r = a.check(lay, skin_mesh, out_dir, base_dir)
            antenna_report[a.id] = r
            if not r["ok"]:
                warnings.append(
                    f"{a.id}: the antenna's reflection inside the band is {r['installed']['inBand']['worstS11']} dB "
                    f"as installed (limit {r['limit']} dB; {r['bare']['inBand']['worstS11']} dB on the bare "
                    f"board): its resonance moved {r['change']['resonanceShift']} MHz. Retune its copper in place "
                    f"(antenna.tune on {r.get('brief', 'the installed brief')}), or move what is near it "
                    "(antenna.firstOrder says which part or wall costs most)")
            if r["takenAsMetal"]:
                warnings.append(f"{a.id}: {', '.join(r['takenAsMetal'])} taken as metal for the antenna (give "
                                "each an `rf` material, or \"ignore\", if it is not)")

    # 4. supports and loads
    sp = _surface_point(exterior, centre, rest)
    r_sup = float(b.get("supportRadius", max(3.0, 2.0 * t_skin)))
    supports = b.get("supports") or [{"shape": {"type": "sphere", "center": _r(sp), "r": round(r_sup, 3)}}]
    cases = list(b.get("loadCases") or [])
    loads = b.get("loads") or {}

    def face_slab(f, depth):
        d = _axis_vec(f)
        k = int(np.argmax(np.abs(d)))
        slo, shi = lo - 1.0, hi + 1.0
        if d[k] < 0:
            slo[k], shi[k] = lo[k] - 0.5, lo[k] + depth
        else:
            slo[k], shi[k] = hi[k] - depth, hi[k] + 0.5
        return {"type": "box", "min": _r(slo), "max": _r(shi)}

    drop = loads.get("drop")
    if drop:
        g = float(drop.get("g", 50.0))
        faces = drop.get("faces") or _FACES
        massed = {k: v for k, v in mounts.items() if v["mass"] > 0}
        if not massed:
            warnings.append("loads.drop: no component has a mass with a mount; no drop cases written")
        for f in faces if massed else []:
            d = _axis_vec(f)
            ls, total, pts = [], 0.0, []
            for k, v in massed.items():
                gm = [m for m in byid if root(m) == k]
                w = np.array([float(byid[m].get("mass") or 0.0) for m in gm])
                cs = np.array([lay["components"][m]["centre"] for m in gm])
                at = (w[:, None] * cs).sum(axis=0) / max(w.sum(), 1e-12)
                ls.append({"region": f"mount-{k}", "mass": round(v["mass"], 6), "dynamic": g,
                           "gravity": f, "at": _r(at, 3)})
                pts.append((k, float(v["mass"]), np.asarray(at, float)))
                total += v["mass"] * 9.81 * g
            # the floor holds the shell back on that face
            slab = face_slab(f, t_skin + float(drop.get("contactDepth", 0.5)))
            ls.append({"shape": slab, "force": _r(-total * d, 3)})
            ls += _drop_spin(pts, -total * d, _contact_centre(exterior, slab), 9.81 * g * d)
            cases.append({"name": f"drop{f}", "loads": ls})
    for i, sq in enumerate(_listify(loads.get("squeeze"))):
        a = str(sq.get("axis", "x")).lstrip("+-")
        d = _axis_vec("+" + a)
        F = float(sq["force"])
        r = float(sq.get("radius", 8.0))
        c0 = np.asarray(sq.get("at", centre), float)
        p1, p2 = _surface_point(exterior, c0, d), _surface_point(exterior, c0, -d)
        cases.append({"name": sq.get("name", f"squeeze-{a}" + (f"-{i}" if i else "")), "loads": [
            {"shape": {"type": "sphere", "center": _r(p1), "r": r}, "force": _r(-F * d, 3)},
            {"shape": {"type": "sphere", "center": _r(p2), "r": r}, "force": _r(F * d, 3)}]})
    for i, pz in enumerate(_listify(loads.get("press"))):
        f = pz.get("face", "+z")
        n = _axis_vec(f)
        F = float(pz["force"])
        at = np.asarray(pz["at"], float)
        if at.size == 2:
            # a point on the face, given in the face's two other axes
            k = int(np.argmax(np.abs(n)))
            full = centre.copy()
            full[[j for j in range(3) if j != k]] = at
            at = full
        p = _surface_point(exterior, at, n) if exterior.sdf(at[None])[0] <= 0 else at
        cases.append({"name": pz.get("name", f"press-{i}"), "loads": [
            {"shape": {"type": "sphere", "center": _r(p), "r": float(pz.get("radius", 5.0))},
             "force": _r(-F * n, 3)},
            {"shape": face_slab(_axis_name(-n), t_skin + 0.5), "force": _r(F * n, 3)}]})
    # connectors: a plug pushed in, pulled out and levered sideways
    # (connectors.py), into the mount the socket's board sits on, the hand
    # holding the face across from the port
    shift_limits = []
    for cn in _listify(loads.get("connector")):
        from .connectors import connector_cases
        cid = cn["component"]
        if cid not in byid or cid not in lay["components"]:
            raise ValueError(f"loads.connector.component {cid!r}: not a placed component")
        if root(cid) not in mounts:
            warnings.append(f"loads.connector {cid}: its board ({root(cid)}) has no mount; the plug loads go "
                            "into the socket's own body, which the interior does not hold")
        region = f"mount-{root(cid)}" if root(cid) in mounts else None
        if region is None:
            continue
        if cn.get("out"):
            out = _axis_vec(cn["out"])
        else:
            al = byid[cid].get("alignTo") or {}
            o = next((o for o, p in zip(ops, op_place) if p["id"] == al.get("opening")), None)
            if o is None:
                raise ValueError(f"loads.connector {cid}: give `out` (the way the plug comes out), or align the "
                                 "component to an opening")
            out = -_axis_vec(o["inward"])
        blo, bhi = (np.asarray(v, float) for v in lay["components"][cid]["bounds"])
        c0 = 0.5 * (blo + bhi)
        k = int(np.argmax(np.abs(out)))
        mouth = c0.copy()
        mouth[k] = bhi[k] if out[k] > 0 else blo[k]
        ext = bhi - blo
        across = [j for j in range(3) if j != k]
        wide = np.eye(3)[max(across, key=lambda j: ext[j])]
        cc, clear = connector_cases(
            cn.get("type", "usb-c"), region=region, mouth=mouth, out=out, wide=wide,
            hold={"shape": face_slab(_axis_name(-out), t_skin + 0.5)}, name=cn.get("name", cid),
            insert=cn.get("insert"), withdraw=cn.get("withdraw"), side=cn.get("side"), lever=cn.get("lever"),
            abuse=bool(cn.get("abuse", False)), wrench=cn.get("wrench"), max_shift=cn.get("maxShift", 0.1))
        cases += cc
        if clear:
            shift_limits.append(clear)
    # rattle: every mounted component held still in the hand (feel.py
    # isolate with upTo): its resonance above the handling band, and a knock
    # at `g` moving it under maxTravel on its mount
    rattle = loads.get("rattle")
    rattle_targets = []
    if rattle is not None:
        want = rattle.get("components")
        hold = rattle.get("hold") or {"region": "skin"}
        for k, v in mounts.items():
            if v["mass"] <= 0 or (want and k not in want):
                continue
            rattle_targets.append({
                "kind": "isolate", "name": f"rattle-{k}", "hold": hold, "mount": {"region": f"mount-{k}"},
                "rigid": True, "kg": round(v["mass"], 6), "upTo": float(rattle.get("band", 150.0)),
                "maxAmplification": float(rattle.get("maxAmplification", 1.25)),
                "maxTravel": float(rattle.get("maxTravel", 0.1)), "shockG": float(rattle.get("g", 5.0))})
        if not rattle_targets:
            warnings.append("loads.rattle: no component has a mass with a mount; nothing to hold still")
    if not cases and not b.get("switch") and not b.get("feel") and not rattle_targets:
        raise ValueError("enclosure: nothing to design for; give loads (drop, squeeze, press, connector), loadCases, a "
                         "switch or a feel block")

    study = {"name": b.get("name", "enclosure"),
             "description": b.get("description", "the inside of a fixed exterior (loadpath enclosure)"),
             "designSpace": ext_spec, "skin": sk, "preserve": preserve, "keepOut": keep,
             "material": b.get("material", "pa12"), "minMember": min_member}
    if cases:
        study["supports"] = supports
        study["loadCases"] = cases
        study["goal"] = dict(b.get("goal") or {"safetyFactor": 2.0})
        if shift_limits:
            study["goal"]["clearances"] = list(study["goal"].get("clearances") or []) + shift_limits
    # heat: watts per component into the study's thermal block when the
    # brief has one (verify reports the temperatures); a cooling block
    # passes through untouched
    # (only with a thermal block: a coupled thermal solve costs several
    # times a structural one, for heat that rarely sizes a printed part)
    heat = [{"region": c["id"], "watts": float(c["watts"])} for c in comps if c.get("watts") and c["id"] in comp_specs]
    if b.get("thermal"):
        th = dict(b.get("thermal") or {})
        th["heat"] = list(th.get("heat") or []) + heat
        study["thermal"] = th
    for k in ("process", "resolution", "maxIterations", "printDirection", "drains", "cooling", "feel"):
        if b.get(k) is not None:
            study[k] = b[k]
    if rattle_targets:
        fb = dict(study.get("feel") or {})
        fb["targets"] = list(fb.get("targets") or []) + rattle_targets
        study["feel"] = fb
    if b.get("cooling") is not None:
        # the cooling block's components from the parts list: each part's
        # watts and maxTemp (and conductivity) on its own keep-out, unless
        # the block names that part itself
        co = dict(b["cooling"])
        named = {c.get("region") for c in co.get("components") or []}
        co["components"] = list(co.get("components") or []) + [
            {"region": c["id"], **{k: c[k] for k in ("watts", "maxTemp", "conductivity") if c.get(k) is not None}}
            for c in comps if c["id"] in comp_specs and c["id"] not in named and not c.get("external")
            and any(c.get(k) is not None for k in ("watts", "maxTemp", "conductivity"))]
        if not co["components"]:
            warnings.append("cooling: no component has watts or maxTemp; the cooling block has nothing to do")
        study["cooling"] = co

    # a push pad: handoff reads the plunger and its seat off the placed parts
    handoff_report = None
    if b.get("switch"):
        from .handoff import handoff as _handoff
        hb = {"name": study["name"], "exterior": ext_spec, "skin": sk, "material": study["material"],
              "minMember": min_member,
              "components": [{"id": c["id"], "shape": comp_specs[c["id"]],
                              **{k: c[k] for k in ("role", "installedOn", "external", "clearance") if k in c},
                              **({"insertFrom": path_of(c["id"]), "fixed": True}
                                 if path_of(c["id"]) is not None and not c.get("installedOn") else {})}
                             for c in comps if c["id"] in comp_specs],
              "assembly": order, "switch": b["switch"], "rest": _axis_name(rest),
              "fixed": [{"id": p["id"], "shape": p["shape"]} for p in preserve]}
        h = _handoff(hb, None)
        hs = h["study"]
        handoff_report = h["report"]
        study["switch"] = hs["switch"]
        have = {p["id"] for p in study["preserve"]}
        study["preserve"] += [p for p in hs["preserve"] if p["id"] not in have]
        have = {k["id"] for k in study["keepOut"]}
        study["keepOut"] += [k for k in hs["keepOut"] if k["id"] not in have]
        if not cases:
            study["supports"], study["loadCases"] = hs["supports"], hs["loadCases"]
        warnings += handoff_report.get("warnings", [])

    from .schema import lint
    issues = [x for x in lint(study, "study") if x["level"] == "error"]
    placement = {k: {kk: vv for kk, vv in v.items()} for k, v in lay["components"].items()}
    report = {"placement": {"components": placement, "order": pr["order"], "given": pr["given"],
                            "centerOfMass": lay["centerOfMass"], "mass": lay["mass"], "pitch": pr["pitch"],
                            "layoutsTried": tried, "blocked": pr["blocked"], "seconds": pr["seconds"]},
              "assembly": {"order": order, "components": {k: {kk: vv for kk, vv in v.items() if kk != "legs"}
                                                          for k, v in paths.items()},
                           "steps": _steps(order, byid, root, paths, lay)},
              "mounts": {k: {"side": v["side"], "massKg": round(v["mass"], 6),
                             **({kk: v[kk] for kk in ("type", "screws") if kk in v})} for k, v in mounts.items()},
              **({"boards": boards} if boards else {}),
              "loadCases": [c["name"] for c in cases], "supports": supports,
              **({"handoff": handoff_report} if handoff_report else {}),
              **({"wires": wires_report} if wires_report is not None else {}),
              **({"antenna": antenna_report} if antenna_report is not None else {}),
              "warnings": warnings, "lint": issues, "seconds": round(time.time() - t0, 1)}
    if boards:
        warnings += [f"{r['id']} (KiCad): {w}" for r in boards for w in r["warnings"]]
    return {"study": study, "report": report, "layout": lay, "brief": b}


class _GridShape:
    """A shape read off a precomputed distance grid (trilinear): the same
    answer as the shape to a fraction of the pitch, at one interpolation per
    point, for the many thousand points an assembly path is checked at."""

    kind = "grid"

    def __init__(self, shape, pitch: float = 0.3, pad: float = 2.0):
        from .shapes import Box
        lo, hi = shape.bounds()
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.o = self.lo - pad
        n = np.ceil((self.hi + pad - self.o) / pitch).astype(int) + 1
        axes = [self.o[i] + np.arange(n[i]) * pitch for i in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        self.g = shape.sdf(P).reshape(tuple(n))
        self.h = pitch
        self.box = Box(self.o, self.o + (n - 1) * pitch)
        self.shape = shape

    def sdf(self, p):
        from scipy import ndimage
        p = np.asarray(p, float).reshape(-1, 3)
        c = ((p - self.o) / self.h).T
        inside = np.all((c >= 0) & (c <= (np.asarray(self.g.shape)[:, None] - 1)), axis=0)
        out = np.empty(len(p))
        if inside.any():
            out[inside] = ndimage.map_coordinates(self.g, c[:, inside], order=1)
        if (~inside).any():
            q = np.maximum(np.maximum(self.lo - p[~inside], p[~inside] - self.hi), 0.0)
            out[~inside] = np.maximum(np.linalg.norm(q, axis=1), 1.0)
        return out

    def bounds(self):
        return self.lo.copy(), self.hi.copy()


def _surface_points(mesh, spacing: float = 0.5, cap: int = 2500):
    """What stands for a part on its way in: its surface, every `spacing` mm
    or so (a part cannot pass through a skin without its surface doing so),
    and its corners."""
    import trimesh
    n = int(min(cap, max(300, mesh.area / spacing ** 2)))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.vstack([pts, np.asarray(mesh.vertices)[:2000]])


def _group_sdf(ids, shapes):
    def f(p):
        out = None
        for i in ids:
            d = shapes[i].sdf(p)
            out = d if out is None else np.minimum(out, d)
        return out
    return f


def _paths(order, byid, root, comp_shapes, lay, fixed, placed0, exit_bounds, tol, on_progress=None,
           tilt_of=None) -> dict:
    """Each component's way in, in assembly order (placement.insertion_path
    on a grid, then checked exactly along its legs with
    handoff._path_check). A component travels with what is installed on
    it; the fixed geometry is there less every component's place; the parts
    already in are in the way."""

    groups = {}
    for cid in byid:
        if cid in comp_shapes:
            groups.setdefault(root(cid), []).append(cid)
    cut = _group_sdf(list(comp_shapes), comp_shapes)
    out, placed = {}, list(placed0)
    for cid in order:
        if byid[cid].get("installedOn") or cid not in groups:
            continue
        grp = groups[cid]
        if on_progress:
            on_progress({"stage": "assembly", "message": f"a way in for {cid}"}, None)
        fixed_fn = [s.sdf for _, s in fixed]
        placed_fn = [_group_sdf(groups[g], comp_shapes) for g in placed]

        def obstacles(p, fixed_fn=fixed_fn, placed_fn=placed_fn):
            c = cut(p)
            o = np.full(len(p), np.inf)
            for fn in fixed_fn:
                o = np.minimum(o, np.maximum(fn(p), -c))
            for fn in placed_fn:
                o = np.minimum(o, fn(p))
            return o
        b = [lay["components"][g]["bounds"] for g in grp]
        blo, bhi = np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)
        c = byid[cid]
        ctol = float(c.get("pathTolerance", tol))
        rec = {"travelsWith": grp[1:]}
        if c.get("fixed") and c.get("insertFrom") is not None:
            rec.update({"insertFrom": c["insertFrom"], "how": "as given"})
        else:
            hint = c.get("insertFrom") if isinstance(c.get("insertFrom"), str) else None
            pts = np.vstack([_surface_points(lay["meshes"][g]) for g in grp])
            rec.update(_way_in(_group_sdf(grp, comp_shapes), (blo, bhi), pts, obstacles, exit_bounds, ctol, hint,
                               tilt_of(cid) if tilt_of else None))
        out[cid] = rec
        placed.append(cid)
    return out


class _Pen:
    """handoff._Obstacles' interface over one obstacle distance function."""

    names = ["the skin, fixed geometry or a part already in"]

    def __init__(self, fn):
        self.fn = fn

    def penetration(self, p):
        return -self.fn(p), np.zeros(len(p), int)


def _way_in(part_sdf, bounds, pts, obstacles, exit_bounds, tol, hint=None, tilt=None) -> dict:
    """A part's way in: the fewest straight legs on the grid
    (placement.insertion_path), then checked exactly along those legs at
    the part's surface points (handoff._path_check). The grid can be half a
    cell out, so a path the exact check refuses is searched again with the
    grid's tolerance tightened. A part with no straight way in is tried
    turning (`tilt`: {max, cells, minStep}; _tilt_in)."""
    from .handoff import _path_check
    from .placement import insertion_path

    if pts is None:
        lo_, hi_ = bounds
        ax = [np.linspace(lo_[i], hi_[i], 12) for i in range(3)]
        g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        d = part_sdf(g)
        pts = g[(d <= 0) & (d > -1.5)]
        if len(pts) == 0:
            pts = g[d <= 0]
    pitch = 0.4
    last = None
    for k, gtol in enumerate((tol, tol - 0.3 * pitch, tol - 0.6 * pitch, tol - 0.9 * pitch)):
        r = insertion_path(part_sdf, bounds, obstacles, exit_bounds, pitch=pitch, tol=gtol, hint=hint)
        if r is None:
            break
        legs = [(d, n * r["pitch"]) for d, n in r["legs"] if n > 0]
        if not legs:
            legs = [(r["legs"][-1][0], None)]
        lg = [(d, L) for d, L in legs[:-1]] + [(legs[-1][0], None)]
        lg = _settle_legs(pts, obstacles, lg, exit_bounds, r["pitch"])
        ok, pen, _, where, dist = _path_check(pts, _Pen(obstacles), lg, exit_bounds[0], exit_bounds[1], tol)
        if len(lg) == 1:
            ins = _axis_name(lg[0][0])
        else:
            ins = [{"toward": _axis_name(d), "length": round(float(L), 2)} for d, L in lg[:-1]] + \
                [_axis_name(lg[-1][0])]
        rec = {"insertFrom": ins, "legs": len(lg), "worstPenetration": round(float(pen), 3)}
        if ok:
            return rec
        last = {"insertFrom": None, "closest": ins, "worstPenetration": round(float(pen), 3),
                "at": _r(where, 2), "after": round(dist, 2)}
    if tilt and float(tilt.get("max", 0)) > 0:
        rec = _tilt_in(part_sdf, bounds, pts, obstacles, exit_bounds, tol, tilt)
        if rec is not None:
            return rec
    return last or {"insertFrom": None}


def _tilt_in(part_sdf, bounds, pts, obstacles, exit_bounds, tol, tilt) -> dict | None:
    """A way in for a part that has to turn (a board longer than its
    opening: one end in first at a tilt, then laid flat). The route comes
    off a grid of positions and tilts (placement.insertion_motion) as a
    staircase a cell or a tilt step at a time; it is reduced to a few
    slides and pivots, each walked exactly at the part's surface points
    (motion.simplify). tilt: {max (degrees), cells, minStep, pitch, tries
    ((snug, room) per grid search)}."""
    from .placement import insertion_motion

    pen = _Pen(obstacles).penetration
    lo, hi = np.asarray(exit_bounds[0], float), np.asarray(exit_bounds[1], float)
    # with room first (every cell the part touches counts as the part,
    # grown 0.3 mm: the least tilt that is comfortable, on a route no thin
    # wall slips through and that cuts nothing as a keep-out), then with
    # half a cell, then snug (only the cells whose centres are in the part:
    # gaps under half a cell open up, and the exact check sorts out the rest)
    few = pts[::max(1, len(pts) // 500)]
    for snug, room in tilt.get("tries", ((False, 0.3), (False, 0.0), (True, 0.0))):
        r = insertion_motion(part_sdf, bounds, obstacles, exit_bounds, pitch=float(tilt.get("pitch", 0.5)),
                             tol=tol, max_tilt=float(tilt["max"]), cells=int(tilt.get("cells", 1_500_000)),
                             min_step=float(tilt.get("minStep", 1.5)), snug=snug, room=room)
        if r is None:
            continue
        rec = _tilt_legs(r, pts, few, pen, bounds, lo, hi, tol)
        if rec is not None:
            return rec
    return None


def _tilt_legs(r, pts, few, pen, bounds, lo, hi, tol) -> dict | None:
    """A grid route (placement.insertion_motion) as checked legs, or None."""
    from .motion import end_pose, legs_to_spec, max_tilt, path_check, rotation, settler, simplify

    c = r["centre"]
    way = []
    for deg, t in r["way"]:
        R = rotation(r["axis"], deg)
        way.append((R, c - R @ c + t))
    # the grid's poses are a cell out either way: each one a move ends on
    # is first walked to where the part is clear (a share of the points
    # finds the way, all of them decide)
    reach = float(np.linalg.norm(0.5 * (np.asarray(bounds[1], float) - np.asarray(bounds[0], float))))

    def depth(pose, P=few):
        return float(pen(P @ pose[0].T + pose[1])[0].max())

    # the obstacles are read off distance grids, good to a tenth of a
    # millimetre at a sharp rim: a path that clears them by that much cuts
    # nothing as a keep-out (the desk hub's connector, let within 0.03 mm
    # of the door's rim, took 0.1 mm^3 of it). Asked for first; then no
    # touch; then the path tolerance. The part may sit closer than that
    # where it is (in a cradle), so the move away from there is let be as
    # close as it starts
    home = depth(way[0], pts)
    for strict in dict.fromkeys((-0.1, min(tol, 0.05), tol)):
        nudge = settler(depth, r["axis"], c, reach, r["pitch"], strict)
        first = max(strict, min(tol, home + 0.05))

        def settle(pose, nudge=nudge, strict=strict):
            q = nudge(pose)
            return q if q is not None and depth(q, pts) <= strict else None

        def clear(leg, start, strict=strict, first=first):
            lim = first if start is way[0] else strict
            ok, worst = path_check(pts, pen, [leg], lo, hi, max(lim, 0.0), start=start)[:2]
            return ok and (lim >= 0 or _leg_depth(pts, pen, leg, lo, hi, start) <= lim)

        legs = simplify(way, clear, settle, centre=c, reach=reach)
        if not legs:
            continue
        if len(legs[-1]) == 2:
            # the last slide goes on out, if on out is clear
            before = end_pose(legs[:-1])
            if path_check(pts, pen, [(legs[-1][0], None)], lo, hi, max(strict, 0.0), start=before)[0]:
                legs[-1] = (legs[-1][0], None)
        ins = legs_to_spec(legs)
        ok, worst, _, where, dist = path_check(pts, pen, _legs_of(ins), lo, hi, tol)
        if ok:
            return {"insertFrom": ins, "legs": len(legs), "worstPenetration": round(float(worst), 3),
                    "tilt": round(max_tilt(legs), 1), "how": "tilt"}
    return None


def _settle_legs(pts, obstacles, lg, exit_bounds, pitch):
    """The grid's legs come in whole cells; each turning leg is slid within
    a cell either way to where the whole path keeps the most room (the
    least penetration, checked exactly). The path is written as the part's
    keep-out, which is cut from everything, the skin included: a leg that
    stops a hair short of an opening's rim would notch the rim on the way
    out."""
    from .handoff import _path_check

    if len(lg) < 2:
        return lg

    def worst(legs):
        return _path_check(pts, _Pen(obstacles), legs, exit_bounds[0], exit_bounds[1], np.inf, step=0.2)[1]

    best = list(lg)
    w = worst(best)
    deltas = np.linspace(-pitch, pitch, 9)
    for _ in range(2):
        for i in range(len(best) - 1):
            d, L = best[i]
            for dl in deltas:
                if dl == 0 or L + dl <= 0:
                    continue
                trial = best[:i] + [(d, L + dl)] + best[i + 1:]
                wt = worst(trial)
                if wt < w - 1e-6:
                    best, w = trial, wt
            # keep the turn mid-way between the lengths that stay this clear
            d, L = best[i]
            ok = [L + dl for dl in np.linspace(-pitch, pitch, 17)
                  if L + dl > 0 and worst(best[:i] + [(d, L + dl)] + best[i + 1:]) <= w + 1e-6]
            if ok:
                mid = best[:i] + [(d, 0.5 * (min(ok) + max(ok)))] + best[i + 1:]
                if worst(mid) <= w + 1e-6:
                    best = mid
    return best


def _leg_depth(pts, pen, leg, lo, hi, start) -> float:
    """The deepest the points get along one leg (negative: the least room)."""
    from .motion import poses
    return max(float(pen(pts @ R.T + t)[0].max()) for R, t, _ in poses([leg], pts, lo, hi, 0.25, start))


class _GroupShape:
    def __init__(self, ids, shapes):
        self.f = _group_sdf(ids, shapes)

    def sdf(self, p):
        return self.f(p)


def _swept_hits(pts, grp_sdf, legs, step: float = 0.5, reach: float = 30.0) -> bool:
    """Whether any point lies in the volume a part sweeps on its way out."""
    from .motion import poses
    legs = [(leg[0], 60.0) if len(leg) == 2 and leg[1] is None else leg for leg in legs]
    for R, t, _ in poses(legs, step=step, reach=reach):
        if (grp_sdf((pts - t) @ R) <= 0).any():
            return True
    return False


def _contact_centre(exterior, slab: dict):
    """Where the floor meets the part: the middle of the exterior's material
    inside a drop's floor slab."""
    lo, hi = np.asarray(slab["min"], float), np.asarray(slab["max"], float)
    n = np.clip(np.ceil((hi - lo) / 1.0).astype(int) + 1, 2, 160)
    P = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], n[i]) for i in range(3)], indexing="ij"),
                 -1).reshape(-1, 3)
    P = P[exterior.sdf(P) <= 0]
    return P.mean(axis=0) if len(P) else 0.5 * (lo + hi)


def _drop_spin(pts, floor_force, at_floor, accel) -> list:
    """Inertia relief for a drop: the floor's push on one face and the
    components' weight elsewhere balance as forces but not as moments (a
    tool landing on its handle's foot with its motor out front), and the
    part's one support then carries the whole twist as a load it does not
    see (the tool grip's base read 0.27 at its 3 mm support). The landing
    also spins the part: each component gets m (alpha x r) about their
    centre of mass, with alpha from the components' inertia, so the case
    balances in moment too and the support carries nothing. With a single
    massed component (or all on one line) the part cannot spin about that
    line; the least-squares alpha leaves that much to the support."""
    if not pts:
        return []
    m = np.array([p[1] for p in pts])
    r = np.array([p[2] for p in pts])
    cm = (m[:, None] * r).sum(axis=0) / m.sum()
    q = r - cm
    M = np.cross(np.asarray(at_floor, float) - cm, floor_force) + sum(
        np.cross(q[i], m[i] * accel) for i in range(len(m)))
    inertia = sum(m[i] * (np.dot(q[i], q[i]) * np.eye(3) - np.outer(q[i], q[i])) for i in range(len(m)))
    # the extra forces m (alpha x q) make a moment I alpha; it must cancel M.
    # Point masses alone may leave inertia singular (a single component):
    # the shell's own mass is not in it, so cap the spin at what the
    # components can take and say nothing more
    alpha = -np.linalg.lstsq(inertia, M, rcond=1e-9)[0]
    out = []
    for i, (k, mi, at) in enumerate(pts):
        e = mi * np.cross(alpha, q[i])
        if np.linalg.norm(e) > 1e-6:
            out.append({"region": f"mount-{k}", "force": _r(e, 3), "at": _r(at, 3)})
    return out


def _steps(order, byid, root, paths, lay) -> list:
    """The product's assembly as numbered steps a person can follow
    (motion.steps): each part's way in, in the assembly order."""
    from .motion import steps
    bounds = {}
    for cid in paths:
        grp = [k for k in lay["components"] if root(k) == cid]
        b = [lay["components"][g]["bounds"] for g in grp]
        if b:
            bounds[cid] = (np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0))
    ext = [c for c in order if byid[c].get("external")]
    # a split shell's parts go in with the cover off: shells.apply_split
    # finds those ways in, after this
    known = {k: v for k, v in paths.items() if not str(v.get("how", "")).startswith("split")}
    return steps([c for c in order if not byid[c].get("installedOn")], known, bounds, ext)


def _legs_of(ins):
    from .motion import legs_from_spec
    return legs_from_spec(ins)


def _first_way(legs, centre):
    """The side a part first moves toward as it leaves: its first slide's
    direction, or the way its centre swings in a first turn."""
    from .motion import Turn
    if not legs:
        return None
    leg = legs[0]
    if isinstance(leg, Turn):
        v = np.cross(np.radians(leg.angle) * np.asarray(leg.axis, float), np.asarray(centre, float) - leg.at)
        if np.linalg.norm(v) < 1e-6:
            return _first_way(legs[1:], centre)
        k = int(np.argmax(np.abs(v)))
        return ("+" if v[k] > 0 else "-") + "xyz"[k]
    d = leg[0]
    k = int(np.argmax(np.abs(d)))
    return ("+" if d[k] > 0 else "-") + "xyz"[k]


def _mounts(comps, byid, root, lay, paths, comp_shapes, exterior, ext_spec, min_member, skin=1.0):
    """A cradle under each component with a mass: on the side given, else
    the side it is pushed against (`against`, unless it leaves that way),
    else the side it comes to rest on as it goes in (opposite its first
    leg out), else its nearest wall; never in another part's way in. A
    board with mounting holes (`mount.type: holes`, as a KiCad board gets)
    is held on screws through them instead (_hole_mount)."""
    from .shapes import shape_from_spec
    groups = {}
    for cid in byid:
        if cid in comp_shapes:
            groups.setdefault(root(cid), []).append(cid)
    mounts, warn = {}, []
    for c in comps:
        if c.get("installedOn") or c["id"] not in groups:
            continue
        gmass = sum(float(byid[m].get("mass") or 0.0) for m in groups[c["id"]])
        mt = c.get("mount", "cradle" if gmass > 0 else None)
        if not mt:
            continue
        mt = {"type": mt} if isinstance(mt, str) else dict(mt)
        if mt.get("type") == "holes":
            hm, why = _hole_mount(c, mt, lay, groups, paths, comp_shapes, exterior, skin)
            if hm is not None:
                mounts[c["id"]] = {**hm, "mass": gmass}
                continue
            warn.append(f"{c['id']}: not held on its mounting holes ({why}); a cradle instead")
            mt = {k: v for k, v in mt.items() if k in ("side",)}
        elif mt.get("type", "cradle") != "cradle":
            raise ValueError(f"{c['id']}.mount.type: 'cradle' or 'holes' here (bosses and snaps between halves "
                             "come with a split block)")
        rb = lay["components"][c["id"]]["bounds"]
        legs = _legs_of(paths.get(c["id"], {}).get("insertFrom"))
        first = _first_way(legs, lay["components"][c["id"]]["centre"])
        prefs = []
        if mt.get("side"):
            prefs.append(mt["side"])
        if c.get("against") and c["against"] != first:
            prefs.append(c["against"])
        if first:
            prefs.append(_axis_name(-_axis_vec(first)))
        gaps = []
        for f in _FACES:
            d = _axis_vec(f)
            k = int(np.argmax(np.abs(d)))
            c3 = np.asarray(lay["components"][c["id"]]["centre"], float)
            c3[k] = rb[1][k] if d[k] > 0 else rb[0][k]
            gaps.append((float(np.linalg.norm(_surface_point(exterior, c3, d) - c3)), f))
        prefs += [f for _, f in sorted(gaps)]
        th = float(mt.get("thickness", min_member))
        chosen = None
        for side in dict.fromkeys(prefs):
            if side == first and not mt.get("side"):
                continue
            spec = _cradle(rb, _axis_vec(side), th, ext_spec)
            try:
                sh = shape_from_spec(spec)
                lo_, hi_ = sh.bounds()
            except ValueError:
                continue
            ax = [np.linspace(lo_[i], hi_[i], 7) for i in range(3)]
            P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
            P = P[sh.sdf(P) <= 0]
            if len(P) == 0:
                continue
            clash = None
            for o, rec in paths.items():
                if o == c["id"]:
                    continue
                lg = _legs_of(rec.get("insertFrom"))
                if lg and _swept_hits(P, _group_sdf(groups[o], comp_shapes), lg):
                    clash = o
                    break
            if clash is None:
                chosen = (side, spec)
                break
        if chosen is None:
            warn.append(f"{c['id']}: every side for its cradle is in another part's way in; no mount, so its mass "
                        "is not carried in the drop cases")
            continue
        mounts[c["id"]] = {"side": chosen[0], "mass": gmass, "shape": chosen[1]}
    return mounts, warn


def _hole_mount(c, mt, lay, groups, paths, comp_shapes, exterior, skin):
    """A board held by screws through its mounting holes: under each hole
    a threaded sleeve the board's pad bears on (as fasteners.py draws a
    thread: tap drill 0.84 d, wall max(1.5, 0.35 d), engagement 2 d unless
    `threadLength`), and kept clear of material the screw's shank through
    the board, its head (ISO 4762 head, d high) and the driver's line from
    the head to the skin. The sleeves go on the board's underside unless
    `side` (a product axis) is nearer its top, or the underside's sleeves
    would land in another part or its way in. The optimizer grows what
    carries them to the skin. Returns (mount, None) or (None, why)."""
    from .fasteners import METRIC

    rec = lay["components"][c["id"]]
    M = np.asarray(rec["matrix"], float)
    R, T = M[:3, :3], M[:3, 3]
    t = float(mt.get("thickness", 1.6))
    holes = [h for h in mt.get("holes") or [] if h.get("screw") in METRIC or h.get("diameter")]
    if not holes:
        return None, "no hole with a screw size"
    normals = [np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])]
    if mt.get("side") is not None:
        want = _axis_vec(mt["side"])
        normals.sort(key=lambda n: -float((R @ n) @ want))
    others = [g for g in groups if g != c["id"]]
    why = "every side"
    for nl in normals:
        d = R @ nl
        tubes, keep, clash = [], [], None
        for i, h in enumerate(holes):
            nom = float(h.get("diameter") or h["screw"][1:])
            _, head = METRIC.get(h.get("screw"), (1.1 * nom, 1.8 * nom))
            x, y = (list(h["at"]) + [0.0])[:2]
            P = R @ np.array([x, y, 0.0 if nl[2] < 0 else t]) + T           # the face the sleeve meets
            F = P - d * t                                                    # the face the head is on
            room = float(np.linalg.norm(_surface_point(exterior, P, d) - P))
            tl = min(float(mt.get("threadLength", 2.0 * nom)), max(room, 0.5))
            r_tap = 0.42 * nom
            r_sl = max(0.5 * nom + max(1.5, 0.35 * nom), 0.5 * float(h.get("pad") or 0.0))
            tubes.append({"type": "cylinder", "p0": _r(P), "p1": _r(P + d * tl), "r": round(r_sl, 4),
                          "rInner": round(r_tap, 4)})
            hid = h.get("ref") or str(i + 1)
            keep.append({"id": f"mount-{c['id']}-{hid}-screw", "bore": False, "shape": {
                "type": "cylinder", "p0": _r(F - d * 0.2), "p1": _r(P + d * (tl + 1.0)), "r": round(r_tap, 4)}})
            k_head = nom
            r_head = 0.5 * head + 0.5
            keep.append({"id": f"mount-{c['id']}-{hid}-head", "bore": False, "shape": {
                "type": "cylinder", "p0": _r(F + d * 0.05), "p1": _r(F - d * k_head), "r": round(r_head, 4)}})
            reach = float(np.linalg.norm(_surface_point(exterior, F, -d) - F)) - skin - k_head - 0.2
            if mt.get("driver", True) and reach > 0.5:
                keep.append({"id": f"mount-{c['id']}-{hid}-driver", "bore": False, "shape": {
                    "type": "cylinder", "p0": _r(F - d * k_head), "p1": _r(F - d * (k_head + reach)),
                    "r": round(r_head, 4)}})
        # the sleeves must not land in another part or in its way in
        spec = {"type": "union", "shapes": tubes}
        Pq = []
        for tb in tubes:
            p0, p1 = np.asarray(tb["p0"]), np.asarray(tb["p1"])
            u = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(d, u)
            for s_ in np.linspace(0.05, 0.95, 5):
                for rr in (0.5 * (tb["r"] + tb["rInner"]), 0.95 * tb["r"]):
                    for a_ in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
                        Pq.append(p0 + s_ * (p1 - p0) + rr * (np.cos(a_) * u + np.sin(a_) * v))
        Pq = np.asarray(Pq)
        for o in others:
            if any(comp_shapes[m].sdf(Pq).min() < -0.05 for m in groups[o] if m in comp_shapes):
                clash = f"its sleeves {'under' if nl[2] < 0 else 'over'} it land in {o}"
                break
            lg = _legs_of((paths.get(o) or {}).get("insertFrom"))
            if lg and _swept_hits(Pq, _group_sdf(groups[o], comp_shapes), lg):
                clash = f"its sleeves {'under' if nl[2] < 0 else 'over'} it are in {o}'s way in"
                break
        if clash is None:
            return {"type": "holes", "side": _axis_name(d), "shape": spec, "keepOut": keep,
                    "screws": [h.get("screw") or f"{h.get('diameter')} mm" for h in holes]}, None
        why = clash
    return None, why


def _listify(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def exterior_check(mesh, exterior, openings=(), tol: float = 0.05, samples: int = 40_000, region=None,
                   drawn=()) -> dict:
    """The delivered part against the designer's exterior: nothing proud of
    it, and all of it still there (bar the openings). Read on the meshes
    themselves: points sampled on each surface, distances exact (closest
    point on the triangles), not a grid. With `region` (a split shell's
    half), only the exterior inside it is expected, a skin clear of the
    cut. `drawn`: shapes the designer drew into the part (preserved), not
    read as proud where they stand past the exterior (the touch
    microphone's rail lips hang down into its underside opening)."""
    import trimesh

    from .shapes import shape_from_spec
    ext_mesh = exterior.mesh if hasattr(exterior, "mesh") else None
    if ext_mesh is None:
        raise ValueError("exterior_check needs the exterior as a mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=1)
    if drawn:
        mine = np.zeros(len(pts), bool)
        for d in drawn:
            mine |= d.sdf(pts) < max(tol, 0.3)     # a mesh shape's sdf is read on a grid
        pts = pts[~mine]
    proud = _signed_distance(ext_mesh, pts) if len(pts) else np.zeros(1)
    q, _ = trimesh.sample.sample_surface(ext_mesh, samples, seed=2)
    keep = np.ones(len(q), bool)
    for o in openings:
        sh = shape_from_spec(o["shape"]) if isinstance(o, dict) else o
        keep &= sh.sdf(q) > tol
    if region is not None:
        keep &= region.sdf(q) < -max(tol, 0.5)
    q = q[keep]
    try:
        import igl
    except ImportError:
        igl = None
    _, dist, _ = _closest(mesh, q, igl)
    missing = dist > tol
    return {"maxProud": round(float(max(proud.max(), 0.0)), 4), "proudShare": round(float((proud > tol).mean()), 5),
            "coverage": round(float(1.0 - missing.mean()), 5), "maxGap": round(float(dist.max()), 4),
            "tolerance": tol, "samples": int(len(q)),
            "untouched": bool(proud.max() <= tol and missing.mean() <= 1e-3)}


def _closest(mesh, q, igl=None):
    """(closest points, distances, face ids) on a mesh: libigl's AABB tree
    when installed, else trimesh's closest_point a few thousand points at a
    time (it holds every candidate triangle of every point at once: past
    12 GB for 40k points on a speaker's 340k-face wrap)."""
    if igl is not None:
        sq, fid, c = igl.point_mesh_squared_distance(q, np.asarray(mesh.vertices, float),
                                                     np.asarray(mesh.faces, np.int64))
        return np.asarray(c), np.sqrt(np.maximum(sq, 0.0)), np.asarray(fid, np.int64)
    from trimesh.proximity import closest_point
    parts = [closest_point(mesh, q[i:i + 2000]) for i in range(0, len(q), 2000)]
    return (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def _signed_distance(mesh, pts, chunk: int = 20_000) -> np.ndarray:
    """Exact distance from each point to the mesh, positive outside. The
    side from the closest face's normal where the point is clearly off it,
    from the winding number (libigl) or ray parity, a few points at a time,
    where it sits beside an edge. trimesh.proximity.signed_distance casts
    rays for every point at once and, without embree, held 12 GB for 40k
    points on a smart button's dome."""
    pts = np.asarray(pts, float)
    out = np.empty(len(pts))
    normals = mesh.face_normals
    try:
        import igl
    except ImportError:
        igl = None
    for s in range(0, len(pts), chunk):
        q = pts[s:s + chunk]
        c, d, tid = _closest(mesh, q, igl)
        off = q - c
        dot = np.einsum("ij,ij->i", off, normals[tid])
        sign = np.where(dot > 0, 1.0, -1.0)
        unclear = (np.abs(dot) < 0.7 * np.linalg.norm(off, axis=1)) & (d > 1e-9)
        if unclear.any():
            qu = q[unclear]
            if igl is not None:
                w = igl.fast_winding_number(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), qu)
                inside = np.abs(w) > 0.5
            else:
                inside = np.concatenate([mesh.contains(qu[i:i + 50]) for i in range(0, len(qu), 50)])
            sign[unclear] = np.where(inside, -1.0, 1.0)
        out[s:s + chunk] = sign * d
    return out


def _localize(study: dict, out_dir: str) -> None:
    """Mesh paths relative to the study's folder: the placed parts are in
    cad/ already; anything else is copied there."""
    import shutil
    root = os.path.realpath(out_dir)
    cad = os.path.join(root, "cad")

    def walk(v):
        if isinstance(v, dict):
            if v.get("type") == "mesh" and isinstance(v.get("path"), str):
                src = os.path.realpath(v["path"])
                if os.path.commonpath([root, src]) != root:
                    os.makedirs(cad, exist_ok=True)
                    dst = os.path.join(cad, os.path.basename(src))
                    if not os.path.exists(dst):
                        shutil.copyfile(src, dst)
                    src = dst
                v["path"] = os.path.relpath(src, root)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(study)


def run_enclosure(brief_path: str, out_dir: str, base_dir: str | None = None, on_progress=None) -> dict:
    """Read a brief, write <out>/<name>.study.json, enclosure.json and the
    placed components (cad/)."""
    brief = json.load(open(brief_path))
    base = base_dir or os.path.dirname(os.path.abspath(brief_path))
    r = enclosure(brief, base, out_dir, on_progress=on_progress)
    study = json.loads(json.dumps(r["study"]))
    _localize(study, out_dir)
    sp = os.path.join(out_dir, f"{study['name']}.study.json")
    json.dump(study, open(sp, "w"), indent=1)
    if brief.get("split"):
        # one study per half, the paths into the base with the cover off,
        # lips, bosses and snaps: shells.py
        try:
            from .shells import apply_split
        except ImportError:
            r["report"]["warnings"].append("split: this loadpath has no shells.py; the study is written whole")
        else:
            cids = {c.get("id"): c for c in r.get("brief", brief).get("components") or []}
            parts = [{"id": k["id"], "shape": k["shape"],
                      **({"installedOn": cids[k["id"]]["installedOn"]} if cids[k["id"]].get("installedOn") else {})}
                     for k in study["keepOut"] if k["id"] in cids and not cids[k["id"]].get("external")]
            sr = apply_split(study, brief["split"], parts, out_dir=out_dir, base_dir=out_dir)
            r["split"] = {}
            for half, hs in sr["studies"].items():
                hp = os.path.join(out_dir, f"{study['name']}-{half}.study.json")
                json.dump(hs, open(hp, "w"), indent=1)
                r["split"][half] = hp
            r["report"]["split"] = sr["report"]
    r["report"]["pathsIntoSkin"] = paths_into_skin(study, out_dir)
    bad = {k: v for k, v in r["report"]["pathsIntoSkin"].items() if v > 0.05}
    if bad:
        r["report"]["warnings"].append(
            f"ways in cut into the skin (mm^3, the keep-out wins): {bad}")
    json.dump(r["report"], open(os.path.join(out_dir, "enclosure.json"), "w"), indent=1, default=float)
    r["study"] = study
    r["studyPath"] = sp
    return r


def paths_into_skin(study: dict, base_dir: str) -> dict:
    """Volume (mm^3) each keep-out with a way in shares with the skin. A
    keep-out is cut from everything, so any overlap is a notch in the
    exterior's skin; the way in is chosen to leave none."""
    from .sharp import exact_manifold
    from .spec import study_from_dict

    study = json.loads(json.dumps(study))
    _localize(study, base_dir)
    st = study_from_dict(study, base_dir=base_dir)
    skin = [exact_manifold(r.shape) for r in st.preserve if r.id == "skin"]
    if not skin:
        return {}
    ins = {k["id"] for k in study.get("keepOut") or [] if k.get("insertFrom") is not None}
    return {r.id: round(float((exact_manifold(r.shape) ^ skin[0]).volume()), 3)
            for r in st.keep_out if r.id in ins}


def _delivered(design_dir: str, name: str):
    """The delivered STL of a design folder: `run`'s best outcome, or the
    switch/feel part (<name>.stl)."""
    rp = os.path.join(design_dir, "report.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        if r.get("best") is not None and r.get("outcomes"):
            return os.path.join(design_dir, os.path.basename(r["outcomes"][r["best"]]["stl"])), r
        # no passing outcome: `run` still delivers its closest part, and
        # that is what gets checked (and fails, with numbers) rather than
        # reporting nothing at all
        d = (r.get("deliver") or {}).get("stl")
        if d and os.path.exists(os.path.join(design_dir, os.path.basename(d))):
            return os.path.join(design_dir, os.path.basename(d)), r
    p = os.path.join(design_dir, f"{name}.stl")
    return (p if os.path.exists(p) else None), (json.load(open(rp)) if os.path.exists(rp) else None)


def check_delivered(study_path: str, design_dir: str, quick: bool = False, on_progress=None) -> dict:
    """The delivered part read as a designer and an engineer would: the
    exterior untouched (exterior_check, on the meshes), and every load case
    verified on the part itself (`verify`: cut cells, or tets where
    installed), with the components' keep-outs and paths clear."""
    import trimesh

    from . import quick as q
    from .spec import study_from_dict
    from .verify import verify_part

    raw = json.load(open(study_path))
    base = os.path.dirname(os.path.abspath(study_path))
    part, rep = _delivered(design_dir, raw["name"])
    if part is None or not os.path.exists(part):
        return {"ok": False, "error": f"no delivered part in {design_dir}"}
    mesh = trimesh.load(part, force="mesh")
    study = study_from_dict(raw, base_dir=base)
    from .shapes import shape_from_spec
    ds = raw.get("designSpace")
    region = None
    if isinstance(ds, dict) and ds.get("type") == "intersect" and len(ds.get("shapes") or []) == 2:
        # a split shell's half (shells.py): the exterior cut by the half's
        # region; the exterior is checked inside that region
        from .handoff import _fine_mesh_shape
        ext = _fine_mesh_shape(ds["shapes"][0], base)
        region = shape_from_spec(ds["shapes"][1], base)
    else:
        ext = study.design_space[0]
    openings = [o for o in ((raw.get("skin") or {}).get("openings") or []) if isinstance(raw.get("skin"), dict)]
    # the cuts the openings made, not their whole shapes: an opening drawn
    # as a box round a drawn port only opens the port (spec.skin_openings)
    from .spec import skin_openings
    ops = skin_openings(openings, ext, base)
    if on_progress:
        on_progress({"stage": "check", "message": "exterior against the delivered surface"}, None)
    ex = exterior_check(mesh, ext, ops, region=region)
    out = {"part": part, "exterior": ex, "volumeCm3": round(float(abs(mesh.volume)) / 1000.0, 3),
           "watertight": bool(mesh.is_watertight)}
    ok = ex["untouched"] and mesh.is_watertight
    if study.load_cases:
        if on_progress:
            on_progress({"stage": "check", "message": "load cases on the delivered part"}, None)
        cells = q.verify_cells(60_000) if quick else 120_000
        from .domain import choose_pitch
        # stress within a design cell of a load or support is left out of
        # the peak, as `run` and `loadpath verify` read it: the drop's
        # floor patch and a cradle's mass load are applied on a few cells,
        # and their edge stress grows without bound as the grid refines
        # (the tool grip's cover read 0.09 here and 5.5 in run's own check
        # of the same file, on the same solution)
        vr = verify_part(study, mesh, max_cells=cells, contact_band=choose_pitch(study))
        sf = study.goal.safety_factor if study.goal is not None else None
        out["verify"] = {k: vr[k] for k in ("safetyFactor", "keepOutIntrusion", "outsideDesignSpace",
                                            "preserveMissing", "bodies", "bucklingFactor", "thermal", "cases",
                                            "cutCells", "analysisPitch") if k in vr}
        smin = vr.get("safetyFactor")
        if sf is not None and smin is not None:
            ok = ok and float(smin) >= 0.98 * float(sf)
        if vr.get("keepOutIntrusion", 0.0) > 1.0:
            ok = False
    if rep is not None:
        out["designReport"] = {k: rep[k] for k in ("passes", "best", "summary") if k in rep}
        if "passes" in rep and any(rep.get(k) for k in ("feel", "presses", "cooling")):
            # a switch, feel or cooling design: its own targets on the
            # delivered part (press forces, feel windows, temperatures)
            ok = ok and bool(rep.get("passes"))
    out["ok"] = bool(ok)
    out["summary"] = (f"exterior {'untouched' if ex['untouched'] else 'CHANGED'} (max proud {ex['maxProud']} mm, "
                      f"coverage {100 * ex['coverage']:.2f}%)"
                      + (f"; min safety factor {out['verify'].get('safetyFactor')}"
                         if out.get("verify") else ""))
    return out


def design_and_check(r: dict, out_dir: str, pitches=None, iterations=None, time_limit: float | None = None,
                     quick: bool = False, nonlinear: bool | None = None, on_progress=None) -> dict:
    """Design what run_enclosure wrote and check it as delivered. `r` is
    run_enclosure's result. The design follows the study's blocks: `switch`
    (switch.py), `feel` (feel.py), `cooling` (cooling.py, designed together
    with any switch or feel block; its load cases checked on the part), else
    `run` for strength and stiffness. A split exterior is two studies, base
    and cover, each designed and checked on its own. Writes
    <out>/design[-half]/ and <out>/checks.json; returns {parts: {half:
    checks}, ok}."""
    from .__main__ import _jsonable, _write_report
    from .__main__ import main as cli
    from .spec import study_from_dict

    halves = dict(r.get("split") or {}) or {"whole": r["studyPath"]}
    parts = {}
    for half, sp in halves.items():
        raw = json.load(open(sp))
        design = os.path.join(out_dir, "design" if half == "whole" else f"design-{half}")
        t0 = time.time()
        code = 0
        try:
            if raw.get("cooling") or raw.get("feel") or raw.get("switch"):
                st = study_from_dict(raw, os.path.dirname(os.path.abspath(sp)))
                if raw.get("cooling"):
                    from .cooling import run_cooling
                    rep = run_cooling(st, design, pitches=pitches, iterations=iterations, on_progress=on_progress)
                elif raw.get("feel"):
                    from .feel import run_feel
                    rep = run_feel(st, design, pitches=pitches, iterations=iterations, on_progress=on_progress,
                                   nonlinear=nonlinear)
                else:
                    from .switch import run_switch
                    rep = run_switch(st, design, pitches=pitches, iterations=iterations, nonlinear=nonlinear,
                                     on_progress=on_progress)
                _write_report(design, {"ok": rep["passes"], **rep})
            else:
                argv = ["run", sp, "--out", design, "--viewer", "none"] + \
                    (["--time-limit", str(time_limit)] if time_limit else []) + (["--quick"] if quick else [])
                if len(halves) > 1:
                    # each half in a process of its own: what the base's run
                    # leaves allocated (solver caches, factor pools) stayed in
                    # the process, and the cover's check went over 15 GB on
                    # the tool grip
                    import subprocess
                    import sys
                    code = subprocess.run([sys.executable, "-m", "loadpath"] + argv,
                                          stdout=sys.stderr, check=False).returncode
                else:
                    code = cli(argv)
        except ValueError as exc:
            parts[half] = {"ok": False, "error": str(exc), "designSeconds": round(time.time() - t0, 1)}
            continue
        chk = check_delivered(sp, design, quick=quick, on_progress=on_progress)
        chk["designSeconds"] = round(time.time() - t0, 1)
        chk["ok"] = bool(chk["ok"] and code in (0, None))
        parts[half] = chk
    out = {"ok": all(p.get("ok") for p in parts.values()), "parts": parts}
    with open(os.path.join(out_dir, "checks.json"), "w") as fh:
        json.dump(_jsonable(out), fh, indent=1, default=float)
    return out
