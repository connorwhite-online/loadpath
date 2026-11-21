"""
Freeform surfacing from a document, and the audit of any exterior's joins:
what `loadpath freeform`, `loadpath joins` and the MCP tools of the same
names run. The work is surfacing.py's (docs/FREEFORM.md); this is the way in
for an agent that writes JSON, not Python.

A document names its surfaces (a loft through sections, a patch through a
grid of points, a flat) and then lists steps, each the thing a surfacing
artist does by hand: cut an opening, fillet two crossing surfaces, blend
across a gap, cap an open edge, pull a surface by a handle. Edges are never
numbered: an open edge is named by a point near it. Every join made is
measured (gap mm, tangent degrees, curvature 1/mm) and the result is read
back as a whole by `surfacequality.joins`.
"""
import json
import os

import numpy as np


def _ring(d, n=96):
    """A closed section from {"centre", "radii": [a, b] | r, "x", "y",
    "exponent"}: a superellipse in the plane of x and y (exponent 2 an
    ellipse, more squarer)."""
    c = np.asarray(d["centre"], float)
    r = d["radii"]
    a, b = (float(r), float(r)) if np.isscalar(r) else map(float, r)
    x = np.asarray(d.get("x", [1.0, 0.0, 0.0]), float)
    y = np.asarray(d.get("y", [0.0, 1.0, 0.0]), float)
    x = x / np.linalg.norm(x)
    y = y - (y @ x) * x
    y = y / np.linalg.norm(y)
    e = 2.0 / float(d.get("exponent", 2.0))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + float(d.get("start", 0.0))
    ct, st = np.cos(t), np.sin(t)
    return c + (a * np.sign(ct) * np.abs(ct) ** e)[:, None] * x + (b * np.sign(st) * np.abs(st) ** e)[:, None] * y


def _loop(d, n=96):
    return _ring(d, n) if isinstance(d, dict) else np.asarray(d, float)


def _surface(d):
    from . import surfacing as S
    if "loft" in d:
        secs = [_loop(s) for s in d["loft"]]
        closed = bool(d.get("closed", True))
        return S.loft(secs, closed=closed, poles=tuple(d.get("poles", (24, max(4, min(16, len(secs) + 2))))),
                      fair=float(d.get("fair", 1e-6)), start=(lambda i: secs[i][0]) if closed else None)
    if "points" in d:
        P = np.asarray(d["points"], float)
        if P.ndim != 3 or P.shape[2] != 3:
            raise ValueError("a patch's points are a grid: rows of [x, y, z] points, every row as long")
        poles = tuple(d.get("poles", (min(P.shape[0], 12), min(P.shape[1], 12))))
        return S.fit_surface(P, poles, fair=float(d.get("fair", 1e-6)))
    if "flat" in d:
        f = d["flat"]
        c = np.asarray(f["centre"], float)
        x, y = np.asarray(f["x"], float), np.asarray(f["y"], float)
        x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
        a, b = (0.5 * float(v) for v in f["size"])
        g = np.linspace(-1, 1, 6)
        return S.fit_surface(np.stack([c + u * a * x + v * b * y for u in g for v in g]).reshape(6, 6, 3), (4, 4), fair=0.0)
    raise ValueError("a surface is {loft: [sections]}, {points: grid} or {flat: {centre, x, y, size}}")


def _open_edges(face):
    """A face's open edges as loops: each opening's edges together, each
    outer free edge alone, with points along each."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    from . import surfacing as S
    holes = S.hole_edges(face)
    in_hole = [e for h in holes for e in h]
    loops = [h for h in holes] + [[e] for e in S.boundary_edges(face) if not any(e.IsSame(x) for x in in_hole)]
    out = []
    for es in loops:
        pts = []
        for e in es:
            c = BRepAdaptor_Curve(e)
            for t in np.linspace(c.FirstParameter(), c.LastParameter(), 24):
                p = c.Value(float(t))
                pts.append([p.X(), p.Y(), p.Z()])
        out.append((es, np.asarray(pts)))
    return out


def _edge_near(face, point, what):
    loops = _open_edges(face)
    if not loops:
        raise ValueError(f"{what}: the surface has no open edge")
    p = np.asarray(point, float)
    return min(loops, key=lambda l: np.linalg.norm(l[1] - p, axis=1).min())[0]


def _uv(s, p):
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt
    pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, p)), s.geom)
    if pr.NbPoints() == 0:
        raise ValueError(f"edit: {list(p)} does not project onto the surface")
    return pr.LowerDistanceParameters()


def _num(d):
    def r(v):
        return round(float(v), 6) if isinstance(v, (int, float, np.floating)) else v
    return {k: ([r(x) for x in v] if isinstance(v, (list, tuple)) else r(v)) for k, v in d.items()
            if isinstance(v, (int, float, np.floating, str, list, tuple))}


def build(spec, out_dir="freeform", base_dir=None):
    """Build a freeform document (schema 'freeform'). Writes
    out_dir/<name>.step, .stl, -joins.png and .json; returns the report:
    {"ok", "closed", "watertight", "faces", "steps": what each step
    measured, "joins": every join graded, "worst": the grade counts}."""
    from . import surfacequality as Q
    from . import surfacing as S
    from .schema import lint
    if isinstance(spec, str):
        base_dir = base_dir or os.path.dirname(os.path.abspath(spec))
        with open(spec) as fh:
            spec = json.load(fh)
    issues = [i for i in lint(spec, "freeform") if i["level"] == "error"]
    if issues:
        return {"ok": False, "issues": issues}
    name = spec.get("name", "freeform")
    surf = {k: _surface(v) for k, v in spec["surfaces"].items()}       # Surfaces until a step cuts one into a face
    steps = []

    def face(k):
        if k not in surf:
            raise ValueError(f"no surface named {k!r} (have {sorted(surf)})")
        return surf[k].face() if isinstance(surf[k], S.Surface) else surf[k]

    def side(d, what):
        f = face(d["surface"])
        return f, _edge_near(f, d["edge"], what)

    for i, st in enumerate(spec.get("steps", [])):
        op, what = st["op"], f"steps[{i}] ({st['op']})"
        cont = st.get("continuity", "G2")
        if op == "cut":
            keep = st.get("keep", "outside")
            r = S.cut(face(st["surface"]), _loop(st["loop"]), keep)
            if keep == "both":
                surf[st["surface"]], surf[st["as"]] = r
            else:
                surf[st.get("as", st["surface"])] = r
            steps.append({"op": op})
        elif op == "fillet":
            a, b = st["surfaces"]
            keep = st.get("keep", [None, None])
            surf[a], bl, surf[b] = S.fillet(face(a), face(b), st["radius"], cont, st.get("lead", 1.0),
                                            tuple(None if k is None else np.asarray(k, float) for k in keep))
            surf[st.get("as", f"{a}-{b} fillet")] = bl
            steps.append({"op": op, **_num(bl.fit)})
        elif op == "blend":
            cont = tuple(cont) if isinstance(cont, list) else cont
            lead = st.get("lead", 1.0)
            bl = S.blend(side(st["from"], what), side(st["to"], what), cont, tuple(lead) if isinstance(lead, list) else lead)
            surf[st.get("as", f"blend {i}")] = bl
            steps.append({"op": op, **_num(bl.fit)})
        elif op == "cap":
            bnd = []
            for d in st["edges"]:
                f, es = side(d, what)
                bnd += [(e, f, d.get("continuity", cont)) for e in es]
            surf[st.get("as", f"cap {i}")], err = S.cap(bnd, lead=st.get("lead", 1.0))
            steps.append({"op": op, **_num({k: (list(err[k]) if k == "poles" else float(err[k]))
                                            for k in ("G0", "G1", "G2", "poles")})})
        elif op == "edit":
            s = surf.get(st["surface"])
            if not isinstance(s, S.Surface):
                raise ValueError(f"{what}: edit a surface before it is cut, filleted or capped")
            mv = [(*_uv(s, m["at"]), m["by"]) for m in st["moves"]]
            hold = [_uv(s, p) for p in st.get("hold", [])]
            surf[st["surface"]] = S.edit(s, mv, hold, int(st.get("freeze", 2)), st.get("energy", "bending"))
            steps.append({"op": op, **_num(surf[st["surface"]].fit)})
        else:
            raise ValueError(f"{what}: op is cut, fillet, blend, cap or edit")
    faces = [face(k) for k in surf]
    try:
        shape, closed = S.sew_solid(faces), True
    except RuntimeError:
        shape, closed = S.sew(faces), False
    m = S.mesh(shape, 0.01, 0.05)
    closed = closed and bool(m.is_watertight)
    if not closed:
        shape = S.sew(faces)
    out = os.path.join(base_dir or os.getcwd(), out_dir)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, name)
    S.write_step(shape, stem + ".step")
    m.export(stem + ".stl")
    js = Q.join_sheet(shape, stem + "-joins.png", mesh=m)
    worst = {}
    for j in js:
        worst[j["grade"]] = worst.get(j["grade"], 0) + 1
    rep = {"ok": bool(closed or not spec.get("closed", False)), "name": name, "closed": closed,
           "watertight": bool(m.is_watertight), "faces": len(faces), "names": list(surf), "steps": steps,
           "joins": [{k: j[k] for k in ("faces", "grade", "G0", "G1", "G2")} for j in js], "grades": worst,
           "step": stem + ".step", "stl": stem + ".stl", "sheet": stem + "-joins.png", "report": stem + ".json"}
    if spec.get("closed") and not closed:
        rep["error"] = "the faces do not close into a solid: an edge is left open (cap it, or blend it to its neighbour)"
    with open(stem + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    return rep


def audit(path, out=None, limit=12, min_radius=None):
    """Every join of an exterior, graded, with a zebra close-up of each
    (out, default beside the file as <name>-joins.png). A STEP file is read
    face by face (`surfacequality.joins`: crease | tangent | curvature at
    each shared edge, with the gap, the angle and the curvature jump). A
    mesh (STL, OBJ, 3MF, GLB) has no faces, so its creases and rounds are
    found and each round read across its width (`mesh_joins`: a constant
    radius is "tangent", one that eases in "curvature", too few facets to
    tell "unresolved"). Nothing is changed."""
    from . import surfacequality as Q
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path}")
    out = out or os.path.splitext(path)[0] + "-joins.png"
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        from .step import read_step
        js = Q.join_sheet(read_step(path), out, limit=limit)
        kind, rows = "brep", [{k: j[k] for k in ("faces", "grade", "G0", "G1", "G2", "centre")} for j in js]
    else:
        import trimesh
        m = trimesh.load(path, force="mesh")
        js = Q.mesh_join_sheet(m, out, limit=limit, min_radius=min_radius)
        kind = "mesh"
        rows = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in j.items() if k not in ("normal", "radius", "outward")}
                for j in js]
        for r in rows:
            r["centre"] = [round(float(x), 2) for x in r["centre"]]
    grades = {}
    for j in js:
        grades[j["grade"]] = grades.get(j["grade"], 0) + 1
    return {"ok": True, "kind": kind, "joins": rows, "grades": grades, "sheet": out}
