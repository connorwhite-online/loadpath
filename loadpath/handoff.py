"""
The designer's handoff: the exterior surfacing and the component CAD in, a
push-pad study out.

An industrial designer hands over the exterior and the parts that go
inside: a board with its connectors, a switch, a flap that closes the
underside. What the part has to do is known too (where a press should
click and at what force). Writing the study by hand means reading the CAD
for numbers: where the switch's actuator is, how far the plunger sits over
it, which way each part goes in, what the part stands on. `handoff` reads
them off the geometry instead.

From the brief (`loadpath handoff brief.json`; schema `handoff`):

- **keep-outs**: every component, with the room to its parts measured per
  node along the press (`switch.clearance`, one entry per component under
  the skin, so a tall connector gets its own gap and the bare board its
  own);
- **insertFrom from the assembly order**: for each component in order, the
  way it can come in without passing through the skin, the designer's fixed
  geometry, or a component already in place. Each of the six axes is tried
  (the designer's hint first); if none is clear, a two-leg path (slide,
  then drop out: a board end that sits in a slot in the wall). Parts that
  are installed on another one (a switch on a flap, a connector on a board)
  travel with it. A component with no clear way in is reported as blocked,
  with where it hits: that is the designer's to fix;
- **supports**: what the part stands on (a slab at the far side of the
  exterior along `rest`, by default the press direction: the desk);
- **the plunger**: from the switch's geometry, the actuator's top and axis,
  a tip `gap` over it running up into the skin, preserved, and the column
  between kept clear; `bearsOn` the part the switch sits in, whose give is
  solved on its own (switch.part_compliance).

The result is an ordinary study (`switch` block), linted, plus a report of
how each value was found. Nothing in the exterior is changed.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

__all__ = ["handoff", "assembly_paths", "plunger_from_switch"]

_AX = {"x": 0, "y": 1, "z": 2}


def _axis_vec(side) -> np.ndarray:
    if isinstance(side, str):
        s = side.strip().lower()
        sg = -1.0 if s.startswith("-") else 1.0
        d = np.zeros(3)
        d[_AX[s.lstrip("+-")]] = sg
        return d
    d = np.asarray(side, float)
    return d / np.linalg.norm(d)


def _axis_name(d) -> str:
    k = int(np.argmax(np.abs(d)))
    return ("+" if d[k] > 0 else "-") + "xyz"[k]


def _fine_mesh_shape(spec, base_dir, cells: int = 320):
    """A shape; meshes with a finer distance grid than the study default
    (paths are checked to a fraction of a millimetre)."""
    from .shapes import MeshShape, shape_from_spec
    sh = shape_from_spec(spec, base_dir)
    if isinstance(sh, MeshShape):
        sh._GRID_CELLS = cells
    return sh


def _samples(shape, step: float = 0.4, cap: int = 8_000) -> np.ndarray:
    """Points that stand for a component: its inside on a grid, and for a
    mesh its vertices (a thin board between grid planes still shows)."""
    from .shapes import MeshShape
    lo, hi = shape.bounds()
    axes = [np.arange(lo[i] + 0.5 * step, hi[i], step) if hi[i] - lo[i] > step else np.array([0.5 * (lo[i] + hi[i])])
            for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    pts = g[shape.sdf(g) <= 0]
    if isinstance(shape, MeshShape):
        v = np.asarray(shape.mesh.vertices)
        # vertices pulled a hair inside, so a face resting on a face reads
        # as touching, not overlapping
        c = v.mean(axis=0)
        pts = np.vstack([pts, v + 0.02 * (c - v) / np.maximum(np.linalg.norm(c - v, axis=1, keepdims=True), 1e-9)])
    if len(pts) == 0:
        pts = np.array([0.5 * (lo + hi)])
    if len(pts) > cap:
        pts = pts[np.random.default_rng(0).choice(len(pts), cap, replace=False)]
    return pts


class _Obstacles:
    """What a moving component must not pass through. Fixed material (the
    skin, the designer's own geometry) is there less every component's
    place, because each component's keep-out cuts it out of the part: a
    board that sits in the skin's inner layer at the opening's rim is not
    blocked by it there, only by the skin it would pass through on the
    way. A component already in place blocks with all of itself."""

    def __init__(self, shapes: list, names: list, cut: list | None = None, n_fixed: int = 0):
        self.shapes, self.names, self.cut, self.n_fixed = shapes, names, cut or [], n_fixed

    def _cut_sdf(self, p):
        out = np.full(len(p), np.inf)
        for c in self.cut:
            out = np.minimum(out, c.sdf(p))
        return out

    def penetration(self, p):
        """Deepest penetration (mm, > 0 inside) into any obstacle, and which."""
        best = np.full(len(p), -np.inf)
        who = np.full(len(p), -1)
        cut = self._cut_sdf(p) if self.cut else None
        for k, s in enumerate(self.shapes):
            pen = -s.sdf(p)
            if cut is not None and k < self.n_fixed:
                pen = np.minimum(pen, cut)       # fixed material less the components' places
            m = pen > best
            best[m] = pen[m]
            who[m] = k
        return best, who


def _path_check(pts, obstacles: _Obstacles, legs, exit_lo, exit_hi, tol: float, step: float = 0.25):
    """Move the points along legs [(d, length or None)]; the last leg runs
    until every point is outside the exit box. Returns (clear, worst
    penetration, obstacle name, where (point), distance travelled)."""
    at = np.zeros(3)
    worst, where, who_w, dist = 0.0, None, None, 0.0
    corners = np.array([[x, y, z] for x in (exit_lo[0], exit_hi[0]) for y in (exit_lo[1], exit_hi[1])
                        for z in (exit_lo[2], exit_hi[2])])
    for d, length in legs:
        if length is None:
            # until the whole component is past the exterior's bounds
            length = float((corners @ d).max() - ((pts + at) @ d).min()) + 0.5
        n = max(int(np.ceil(length / step)), 1)
        for t in np.linspace(0.0, length, n + 1):
            q = pts + at + t * d
            pen, who = obstacles.penetration(q)
            j = int(np.argmax(pen))
            if pen[j] > worst:
                worst, where, who_w, dist = float(pen[j]), q[j], int(who[j]), float(np.linalg.norm(at) + t)
            if worst > tol:
                return False, worst, obstacles.names[who_w], where, dist
        at = at + length * d
    return True, worst, (obstacles.names[who_w] if who_w is not None else None), where, dist


def assembly_paths(components: list, order: list, fixed: list, exit_bounds, tol: float = 0.25,
                   max_slide: float = 12.0, cut: list | None = None) -> dict:
    """The way each component comes in, in assembly order.

    components: [{id, shape, installedOn?, insertFrom? (a hint, tried
    first, or fixed: true to take it as given)}]. fixed: [(name, shape)]
    that are there from the start (the skin, the designer's own internal
    geometry). A component travels with everything installed on it, and
    must not pass through a component already in place.

    Returns {id: {insertFrom, how, legs, worstPenetration, tried: [...]}};
    a component with no clear path has insertFrom None and blocked: where
    and what it hits on the best single-axis try."""
    byid = {c["id"]: c for c in components}
    kids = {}
    for c in components:
        if c.get("installedOn"):
            if c["installedOn"] not in byid:
                raise ValueError(f"component {c['id']!r} is installedOn {c['installedOn']!r}, which is not a component")
            kids.setdefault(c["installedOn"], []).append(c["id"])

    def group(cid):
        out = [cid]
        for k in kids.get(cid, []):
            out += group(k)
        return out

    pts_cache = {}

    def pts(cid):
        if cid not in pts_cache:
            # a caller may hand over its own points (surface samples of a placed part)
            given = byid[cid].get("points")
            pts_cache[cid] = np.asarray(given, float) if given is not None else _samples(byid[cid]["shape"])
        return pts_cache[cid]

    lo, hi = exit_bounds
    placed = []
    out = {}
    rests = [c["shape"] for c in components] + list(cut or [])
    for cid in order:
        c = byid[cid]
        if c.get("installedOn"):
            continue
        grp = group(cid)
        P = np.vstack([pts(g) for g in grp])
        obs = _Obstacles([s for _, s in fixed] + [byid[p]["shape"] for p in placed],
                         [n for n, _ in fixed] + list(placed), cut=rests, n_fixed=len(fixed))
        hint = c.get("insertFrom")
        ctol = float(c.get("pathTolerance", tol))
        tried = []
        found = None
        cands = []
        if hint is not None and not isinstance(hint, list):
            cands.append(_axis_vec(hint))
        for a in range(3):
            for sg in (-1.0, 1.0):
                d = np.zeros(3)
                d[a] = sg
                if not any(np.allclose(d, e) for e in cands):
                    cands.append(d)
        if hint is not None and c.get("fixed"):
            found = {"insertFrom": hint, "how": "as given", "legs": None}
        else:
            for d in cands:
                ok, pen, who, where, dist = _path_check(P, obs, [(d, None)], lo, hi, ctol)
                tried.append({"toward": _axis_name(d), "clear": ok, "penetration": round(pen, 3),
                              **({} if ok else {"hits": who, "at": np.round(where, 2).tolist(), "after": round(dist, 2)})})
                if ok:
                    found = {"insertFrom": _axis_name(d), "how": "one axis", "legs": [(_axis_name(d), None)],
                             "worstPenetration": round(pen, 3)}
                    break
            if found is None:
                # slide, then out: the shortest first leg that clears
                best = None
                for s in np.arange(0.5, max_slide + 1e-9, 0.5):
                    for d2 in cands:
                        for d1 in cands:
                            if abs(float(d1 @ d2)) > 0.5:
                                continue
                            ok, pen, who, where, dist = _path_check(P, obs, [(d1, float(s)), (d2, None)], lo, hi,
                                                                    ctol)
                            if ok:
                                best = (float(s), d1, d2, pen)
                                break
                        if best is not None:
                            break
                    if best is not None:
                        break
                if best is not None:
                    s, d1, d2, pen = best
                    found = {"insertFrom": [{"toward": _axis_name(d1), "length": round(s, 2)}, _axis_name(d2)],
                             "how": "two legs", "legs": [(_axis_name(d1), s), (_axis_name(d2), None)],
                             "worstPenetration": round(pen, 3)}
        rec = {"travelsWith": grp[1:], "tried": tried}
        if found is None:
            first = min(tried, key=lambda r: r["penetration"]) if tried else None
            rec.update({"insertFrom": None, "blocked": first})
        else:
            rec.update(found)
        out[cid] = rec
        placed += grp
    return out


def plunger_from_switch(switch_shape, press_dir: np.ndarray, gap: float, skin_shape, radius: float | None = None,
                        reach: float = 30.0) -> dict:
    """The actuator's top, axis and radius from the switch's geometry (the
    part of it furthest against the press), and a plunger tip `gap` over
    it, running up into the skin."""
    from .shapes import MeshShape
    up = -press_dir
    if isinstance(switch_shape, MeshShape):
        v = np.asarray(switch_shape.mesh.vertices)
    else:
        v = _samples(switch_shape, 0.1)
    h = v @ up
    top = float(h.max())
    cap = v[h >= top - 0.05]
    ax = cap.mean(axis=0)
    # the axis through the cap's centre; its radius from the cap's spread
    perp = cap - ax
    perp -= np.outer(perp @ up, up)
    r_act = float(np.linalg.norm(perp, axis=1).max())
    tip = ax + gap * up
    # up from the tip until the skin
    t = np.arange(0.0, reach, 0.05)
    pts = tip[None, :] + t[:, None] * up[None, :]
    inside = skin_shape.sdf(pts) <= 0
    if not inside.any():
        raise ValueError("handoff: going up from the switch's actuator, no skin within "
                         f"{reach:g} mm: is the press direction right?")
    t_skin = float(t[int(np.argmax(inside))])
    r = float(radius) if radius else max(r_act, 0.8)
    return {"actuatorTop": (ax).tolist(), "actuatorRadius": r_act, "tip": tip.tolist(), "radius": r,
            "skinAt": float(t_skin), "gap": float(gap)}


def _shape_spec(v, base_dir, reports: dict | None = None, key: str | None = None):
    """A component's geometry as a study shape: a shape dict, a path (STL/
    3MF/OBJ/GLB; STEP through cadimport; an .npz SDF grid), {path, size?,
    up?, ...} for the exterior intake, or in code anything
    `exterior.load_exterior` reads (an SDF function, a build123d object, a
    trimesh). Anything not already one clean closed mesh goes through the
    intake and is written beside its source as <stem>.handoff.stl (or into
    the working folder for an in-memory source); its report lands in
    `reports[key]`."""
    from .exterior import MESH_EXT, STEP_EXT, load_exterior
    opts = {}
    if isinstance(v, dict):
        if "type" in v:
            return v
        if "path" not in v:
            raise ValueError(f"a shape is a shape dict, a path, or {{path, size?, up?, ...}}: got {v!r}")
        opts = {k: x for k, x in v.items() if k != "path"}
        v = v["path"]
    if isinstance(v, (str, os.PathLike)):
        spec_path = os.fspath(v)
        bare = spec_path.partition(".py:")[0] + ".py" if ".py:" in spec_path else spec_path
        path = spec_path if os.path.isabs(bare) or base_dir is None else os.path.join(base_dir, spec_path)
        bare_path = path.partition(".py:")[0] + ".py" if ".py:" in path else path
        ext = os.path.splitext(bare_path)[1].lower()
        if ext == ".npz" and not opts:
            return {"type": "sdf", "path": os.path.abspath(path)}
        if ext in MESH_EXT and ext not in (".glb", ".gltf") and not opts:
            import trimesh
            m = trimesh.load(path, force="mesh", process=False)
            m.merge_vertices()
            if m.is_watertight and m.is_winding_consistent and len(m.split(only_watertight=False)) == 1:
                return {"type": "mesh", "path": os.path.abspath(path)}
        out = os.path.splitext(bare_path)[0] + ".handoff.stl"
        src = path
    else:
        if base_dir is None:
            base_dir = os.getcwd()
        out = os.path.join(base_dir, f"{key or 'shape'}.handoff.stl")
        src = v
        ext = ""
    if ext in STEP_EXT and not opts:
        from .cadimport import load_part
        part = load_part(src)
        part.mesh.export(out)
        if reports is not None and key:
            reports[key] = {"kind": "step"}
        return {"type": "mesh", "path": os.path.abspath(out)}
    ext_obj = load_exterior(src, **opts)
    ext_obj.mesh.export(out)
    if reports is not None and key:
        reports[key] = ext_obj.report
    return {"type": "mesh", "path": os.path.abspath(out)}


def handoff(brief: dict, base_dir: str | None = None) -> dict:
    """Brief in, {study, report} out. See the module docstring and
    `loadpath schema handoff`."""
    from .spec import _skin

    t0 = time.time()
    b = brief
    press = b.get("switch") or {}
    press_side = press.get("pressDirection", "-z")
    pdir = _axis_vec(press_side)
    rest = _axis_vec(b.get("rest", press_side))
    intake = {}
    ext_spec = _shape_spec(b["exterior"], base_dir, intake, "exterior")
    exterior = _fine_mesh_shape(ext_spec, None)
    skin_v = b.get("skin", 1.0)
    sk = dict(skin_v) if isinstance(skin_v, dict) else {"thickness": float(skin_v)}
    # openings in the exterior into the interior (an underside closed by a
    # flap, a port): the skin is not kept across their inner ends (see
    # spec._skin), swept a skin and 0.2 mm inward by default
    ops = []
    for o in b.get("openings") or []:
        o = {"shape": o} if not isinstance(o, dict) else dict(o)
        o["shape"] = _shape_spec(o["shape"], base_dir)
        o.setdefault("inward", _axis_name(-pdir))
        o.setdefault("depth", round(float(sk["thickness"]) + 0.2, 3))
        ops.append(o)
    if ops:
        sk["openings"] = list(sk.get("openings") or []) + ops
    skin_v = sk
    t_skin, skin_shape = _skin(sk, [exterior])
    fixed = []
    preserve = []
    for i, f in enumerate(b.get("fixed") or []):
        fid = f.get("id") or f"fixed-{i}"
        spec = _shape_spec(f["shape"], base_dir, intake, fid)
        fixed.append((fid, _fine_mesh_shape(spec, None)))
        preserve.append({"id": fid, "shape": spec})
    comps = []
    for i, c in enumerate(b.get("components") or []):
        cid = c.get("id") or f"component-{i}"
        spec = _shape_spec(c["shape"], base_dir, intake, cid)
        comps.append({**c, "id": cid, "spec": spec, "shape": _fine_mesh_shape(spec, None)})
    ids = [c["id"] for c in comps]
    order = b.get("assembly") or ids
    missing = [i for i in ids if i not in order and not next(c for c in comps if c["id"] == i).get("installedOn")]
    unknown = [i for i in order if i not in ids]
    if unknown:
        raise ValueError(f"handoff.assembly names {unknown}, which are not components ({ids})")
    order = list(order) + missing
    lo, hi = exterior.bounds()
    tol = float(b.get("pathTolerance", 0.25))
    paths = assembly_paths(comps, order, [("skin", skin_shape)] + fixed, (lo, hi), tol=tol,
                           max_slide=float(b.get("maxSlide", 12.0)))
    warnings = []
    # room inside the exterior once its skin is in: where it is too thin
    # for anything but skin is the designer's to know first
    from .exterior import capacity
    from .shapes import MeshShape
    if isinstance(exterior, MeshShape):
        cap = capacity(exterior.mesh, float(t_skin))
        intake.setdefault("exterior", {})["capacity"] = cap
        warnings += [f"exterior: {w}" for w in cap.pop("warnings", [])]
    for k, r in intake.items():
        warnings += [f"{k}: {w}" for w in r.get("warnings", [])]
    by = {c["id"]: c for c in comps}

    def path_of(cid):
        c = by[cid]
        while c.get("installedOn"):
            c = by[c["installedOn"]]
        return paths.get(c["id"], {}).get("insertFrom")

    keep = []
    for c in comps:
        ins = path_of(c["id"])
        if ins is None and not c.get("installedOn"):
            warnings.append(f"{c['id']}: no clear way in (best try: {paths[c['id']].get('blocked')}); its keep-out "
                            "covers where it sits only. The exterior or the fixed geometry has to change, or give "
                            "its path as insertFrom with fixed: true")
        k = {"id": c["id"], "bore": False, "shape": c["spec"]}
        if ins is not None:
            k["insertFrom"] = ins
        keep.append(k)

    # the switch and its plunger
    sw_id = press.get("component") or next((c["id"] for c in comps if c.get("role") == "switch"), None)
    if sw_id is None or sw_id not in by:
        raise ValueError("handoff.switch.component names the switch (or give one component role: switch)")
    gap = float(press.get("gap", 0.4))
    pl = plunger_from_switch(by[sw_id]["shape"], pdir, gap, skin_shape, press.get("plungerRadius"))
    up = -pdir
    tip = np.asarray(pl["tip"])
    p1 = tip + (pl["skinAt"] + 0.5 * t_skin) * up
    preserve.append({"id": "plunger", "shape": {"type": "cylinder", "p0": np.round(tip, 4).tolist(),
                                                "p1": np.round(p1, 4).tolist(), "r": round(pl["radius"], 3)}})
    # the column between the switch body's top and the tip stays clear,
    # wide enough for the plunger to come down on the stem off centre
    r_col = max(pl["radius"], pl["actuatorRadius"]) + float(press.get("columnClearance", 1.0))
    sw_lo, sw_hi = by[sw_id]["shape"].bounds()
    base_h = float(min(sw_lo @ up, sw_hi @ up))
    act = np.asarray(pl["actuatorTop"])
    c0 = act - (float(act @ up) - base_h) * up
    keep += list(b.get("keepOut") or [])     # the designer's own clear zones, as given
    keep.append({"id": "plunger-column", "bore": False,
                 "shape": {"type": "cylinder", "p0": np.round(c0, 4).tolist(), "p1": np.round(tip, 4).tolist(),
                           "r": round(r_col, 3)}})

    # supports: the slab the part stands on
    ra = int(np.argmax(np.abs(rest)))
    depth = float(b.get("restDepth", 0.3))
    slo, shi = lo - 1.0, hi + 1.0
    if rest[ra] < 0:
        slo[ra], shi[ra] = lo[ra] - 0.5, lo[ra] + depth
    else:
        slo[ra], shi[ra] = hi[ra] - depth, hi[ra] + 0.5
    desk = {"type": "box", "min": np.round(slo, 4).tolist(), "max": np.round(shi, 4).tolist()}
    supports = b.get("supports") or [{"shape": desk}]

    # what the switch bears on
    sw_block = {k: v for k, v in press.items()
                if k not in ("component", "gap", "plungerRadius", "columnClearance", "bearsOn", "trigger", "travel",
                             "clearance", "clearanceMargin", "clearanceRegion")}
    output = {"region": "plunger", "direction": _axis_name(pdir), "gap": gap,
              "trigger": press.get("trigger", 0.0), "travel": press.get("travel", 0.0)}
    bears = press.get("bearsOn", next((c["id"] for c in comps if c.get("role") == "seat"), None))
    if bears:
        if bears not in by:
            raise ValueError(f"handoff.switch.bearsOn: {bears!r} is not a component")
        foot = float(base_h)
        seat_lo, seat_hi = np.array(sw_lo, float), np.array(sw_hi, float)
        pa = int(np.argmax(np.abs(pdir)))
        seat_lo[pa], seat_hi[pa] = foot - 0.1, foot + 0.1
        output["bearsOn"] = {"part": {"shape": by[bears]["spec"],
                                      "seat": {"type": "box", "min": np.round(seat_lo, 4).tolist(),
                                               "max": np.round(seat_hi, 4).tolist()},
                                      "supports": [desk]}}
    sw_block["output"] = output
    # one clearance entry per component under the skin
    if press.get("clearance") is not None:
        sw_block["clearance"] = press["clearance"]
    else:
        margin = float(press.get("clearanceMargin", 0.2))
        # The skin's nodes by default: the touch surface coming down on a
        # part is the failure. Every node (clearanceRegion: null) also holds
        # interior material over a part to its room, which on the touch
        # microphone kept the first grid plane over the board (0.4 mm of
        # room) from carrying anything; keep such space clear with a
        # keepOut instead, as a designer would.
        creg = press.get("clearanceRegion", "skin")
        entries = [{"component": c["id"], "margin": margin, **({"region": creg} if creg else {})} for c in comps
                   if c["id"] != sw_id and c["id"] != bears and not c.get("external") and c.get("clearance", True)]
        if entries:
            sw_block["clearance"] = entries
    study = {"name": b.get("name", "handoff"),
             "description": b.get("description", "push pad from a designer's handoff (loadpath handoff)"),
             "designSpace": ext_spec, "skin": skin_v, "preserve": preserve, "keepOut": keep,
             "supports": supports, "material": b.get("material", "tpu88a"), "switch": sw_block}
    for k in ("minMember", "process", "goal", "resolution", "loadCases"):
        if b.get(k) is not None:
            study[k] = b[k]
    study.setdefault("goal", {"volumeFraction": 0.2})
    if not study.get("loadCases"):
        # a study needs a load case; the switch block sets the design, so
        # this one only records the click: the stem's reaction on the plunger
        unit = press.get("unit", "N")
        T = float(press.get("trigger", 0.0) or 1.0) * (0.00980665 if unit == "gf" else 1.0)
        study["loadCases"] = [{"name": "click", "loads": [{"region": "plunger",
                                                           "force": np.round(-T * pdir, 6).tolist()}]}]
    from .schema import lint
    issues = [x for x in lint(study, "study") if x["level"] == "error"]
    report = {"assembly": {"order": order, "components": {k: {kk: vv for kk, vv in v.items() if kk != "legs"}
                                                          for k, v in paths.items()},
                           "tolerance": tol},
              "plunger": {**pl, "top": np.round(p1, 4).tolist(), "columnRadius": r_col},
              "supports": supports, "bearsOn": bears, "warnings": warnings, "lint": issues,
              **({"intake": intake} if intake else {}),
              "seconds": round(time.time() - t0, 1)}
    return {"study": study, "report": report}


def _localize(study: dict, out_dir: str) -> None:
    """Copy every mesh the study names into <out>/cad/ and point the study
    at the copies: the study is self-contained, and `loadpath switch` keeps
    mesh paths inside the study's folder."""
    import shutil
    cad = os.path.join(out_dir, "cad")
    done = {}

    def walk(v):
        if isinstance(v, dict):
            if v.get("type") == "mesh" and isinstance(v.get("path"), str):
                src = os.path.abspath(v["path"])
                if src not in done:
                    os.makedirs(cad, exist_ok=True)
                    name = os.path.basename(src)
                    stem, ext = os.path.splitext(name)
                    k = 1
                    while name in done.values():
                        name = f"{stem}-{k}{ext}"
                        k += 1
                    shutil.copyfile(src, os.path.join(cad, name))
                    done[src] = name
                v["path"] = f"cad/{done[src]}"
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(study)


def run_handoff(brief_path: str, out_dir: str, base_dir: str | None = None) -> dict:
    """Read a brief, write <out>/<name>.study.json and handoff.json."""
    brief = json.load(open(brief_path))
    base = base_dir or os.path.dirname(os.path.abspath(brief_path))
    r = handoff(brief, base)
    os.makedirs(out_dir, exist_ok=True)
    r["study"] = json.loads(json.dumps(r["study"]))     # no shared dicts: each path is rewritten once
    _localize(r["study"], out_dir)
    sp = os.path.join(out_dir, f"{r['study']['name']}.study.json")
    json.dump(r["study"], open(sp, "w"), indent=1)
    json.dump(r["report"], open(os.path.join(out_dir, "handoff.json"), "w"), indent=1, default=float)
    r["studyPath"] = sp
    return r
