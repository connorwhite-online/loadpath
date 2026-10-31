"""
CAD in: a STEP (or STL/3MF/OBJ) part as a study's design space, with its
interfaces found for you.

Fusion's generative workspace asks you to click the faces a part bolts on
by, the bores its shafts run in and the faces it must not grow over. Most of
that is in the geometry already. `import_part` reads the part, heals it
(real exports are often open shells: five of GRIP's link STEPs are), and
finds:

- bores: concave cylinders, merged across split faces and stacked with
  their coaxial steps (a counterbore is one bore with two segments), with
  their axis, extent, through or blind, and "bolt" (through radius up to
  M6 clearance) or "bore" (a bearing seat, a shaft, a socket);
- flat faces the part can sit on: planar, exposed (nothing of the part in
  front of them), with the bolt holes that open onto each;
- interfaces: a face with its bolt pattern, a bolt pattern on its own, or
  a bore. These are where a part meets the rest of the machine.

`draft_study` turns them into a study an agent finishes: the part (or its
hull or box) as the design space, each interface's material preserved (a
ring round every bore and a pad under every bolted face, cut to the part),
every bore a keep-out, mating faces cleared when the space grows past the
part, and a suggested support and load: the strongest bolted interface
holds the part, the interface farthest from it takes a placeholder force
across the lever. The force is a placeholder: features.json lists every
interface so the real loads go on the right ones.

From a STEP file the bores and faces are exact (OpenCascade surface types);
from a mesh they are fitted (circles to smooth patches, coplanar facets).
STEP needs `pip install "loadpath[step]"` (cadquery-ocp, shared with
step.py's export).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Part", "load_part", "detect_features", "suggest_setup", "summarize", "draft_study", "import_part",
           "heal_mesh"]

STEP_EXT = (".step", ".stp")
# mm chord error of the STEP tessellation: a tenth of the finest grid the
# optimizer runs on, and coarse enough to tessellate a GRIP link in a second.
DEFAULT_TOLERANCE = 0.05
# mm: through radius up to an M6 clearance hole (3.2) counts as a bolt
BOLT_MAX_R = 3.6
# mm: a plain bore narrower than this (a cable pass, a pilot hole) is kept
# open but not ringed with preserved material unless it is loaded or holds:
# a ring there forces a strut out to it (GRIP's yoke grew one to a 4 mm pass)
SEAT_MIN_R = 5.0
# rad: a cylinder whose faces together cover less than this is a fillet or a
# slot end, not a hole
FULL_SPAN = 1.8 * math.pi
# a gap wider than this (mm) was a missing face, not a crack: say so
HEAL_WARN = 5.0


@dataclass
class Part:
    mesh: object                     # watertight trimesh
    path: str | None = None
    brep: object = None              # OpenCascade shape (STEP only)
    face_of: np.ndarray | None = None   # B-rep face index of each triangle (STEP only)
    healed: dict = field(default_factory=dict)


# ---- reading and healing -------------------------------------------------------------

def heal_mesh(mesh):
    """Close a nearly closed mesh: every boundary loop is fanned from its
    centroid. Cracks along unsewn edges and small missing faces close with a
    negligible change in volume; returns (mesh, report)."""
    import trimesh
    from collections import defaultdict

    if mesh.is_watertight:
        return mesh, {"loops": 0}
    e = mesh.edges
    es = np.sort(e, axis=1)
    _, inv, cnt = np.unique(es, axis=0, return_inverse=True, return_counts=True)
    b = e[cnt[inv.ravel()] == 1]
    nxt = defaultdict(list)
    for a, c in b:
        nxt[int(a)].append(int(c))
    used, loops = set(), []
    for a, c in b:
        a, c = int(a), int(c)
        if (a, c) in used:
            continue
        loop, cur = [a], (a, c)
        while cur not in used:
            used.add(cur)
            v = cur[1]
            if v == loop[0]:
                break
            loop.append(v)
            nx = [w for w in nxt[v] if (v, w) not in used]
            if not nx:
                loop = None
                break
            cur = (v, nx[0])
        if loop and len(loop) >= 3:
            loops.append(loop)
    V = [np.asarray(mesh.vertices, float)]
    F = [np.asarray(mesh.faces, np.int64)]
    n = len(mesh.vertices)
    widest = 0.0
    for L in loops:
        P = mesh.vertices[L]
        widest = max(widest, float(np.ptp(P, axis=0).max()))
        V.append(P.mean(0)[None])
        L = np.asarray(L)
        # the loop runs along the existing faces' half-edges; the fan runs back
        F.append(np.stack([np.roll(L, -1), L, np.full(len(L), n)], axis=1))
        n += 1
    out = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=True)
    out.merge_vertices()
    if not out.is_winding_consistent:
        trimesh.repair.fix_winding(out)
    if out.volume < 0:
        out.invert()
    rep = {"loops": len(loops), "widestGap": round(widest, 2),
           "volumeChange": round(float(out.volume - mesh.volume), 1), "watertight": bool(out.is_watertight)}
    if widest > HEAL_WARN:
        rep["warning"] = (f"a {widest:.1f} mm gap was closed with a flat fan: the file was missing faces, "
                          "check the part's shape there")
    return out, rep


def _tessellate(shape, tolerance):
    """Triangulate an OpenCascade shape, remembering each triangle's face."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, tolerance, False, 0.2, True)
    V, F, owner, faces = [], [], [], []
    off = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face(ex.Current())
        fi = len(faces)
        faces.append(f)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is not None:
            tr = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                V.append((p.X(), p.Y(), p.Z()))
            rev = f.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                F.append((off + a - 1, off + c - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + c - 1))
                owner.append(fi)
            off += tri.NbNodes()
        ex.Next()
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, np.int64), process=False)
    m.merge_vertices(digits_vertex=6)
    keep = m.nondegenerate_faces()
    owner = np.asarray(owner, np.int64)[keep]
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return m, owner, faces


def load_part(path, tolerance: float = DEFAULT_TOLERANCE) -> Part:
    """A part from a STEP, STL, 3MF or OBJ file, closed if it was not."""
    import trimesh

    ext = os.path.splitext(str(path))[1].lower()
    if ext in STEP_EXT:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing

        from .step import read_step
        shape = read_step(path)
        sew = BRepBuilderAPI_Sewing(1e-3)
        sew.Add(shape)
        sew.Perform()
        shape = sew.SewedShape()
        mesh, owner, faces = _tessellate(shape, tolerance)
        if len(mesh.faces) == 0:
            raise ValueError(f"{path}: the STEP file has no surfaces to mesh")
        n0 = len(mesh.faces)
        healed, rep = heal_mesh(mesh)
        # healing only appends fans: the original triangles keep their order
        # (merge_vertices does not reorder faces)
        owner = np.concatenate([owner, np.full(len(healed.faces) - n0, -1)]) if len(healed.faces) >= n0 else None
        part = Part(healed, str(path), shape, owner, rep)
        part.faces = faces
        return part
    mesh = trimesh.load(path, force="mesh")
    if mesh.volume < 0:
        mesh.invert()
    healed, rep = heal_mesh(mesh)
    return Part(healed, str(path), healed=rep)


# ---- cylinders and planes ------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _canon_axis(p, d):
    """A line as (point nearest the origin, direction with its largest
    component positive): equal lines get equal keys."""
    d = _unit(d)
    if d[np.argmax(np.abs(d))] < 0:
        d = -d
    p = np.asarray(p, float)
    return p - np.dot(p, d) * d, d


def _concave(mesh, pts, axis_p, axis_d, r):
    """True where the cylinder at these surface points is a hole: just
    inside the surface, towards the axis, there is no material."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    foot = axis_p + np.outer((pts - axis_p) @ axis_d, axis_d)
    radial = pts - foot
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    probe = pts - radial * min(0.3, 0.2 * r)
    return ~mesh.contains(probe)


def _brep_surfaces(part: Part):
    """Exact cylinders and planes of a STEP part, one record per face."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_REVERSED

    cyl, planes = [], []
    tris_of = {}
    if part.face_of is not None:
        for t, fi in enumerate(part.face_of):
            tris_of.setdefault(int(fi), []).append(t)
    for fi, f in enumerate(part.faces):
        a = BRepAdaptor_Surface(f)
        kind = a.GetType()
        tris = np.asarray(tris_of.get(fi, []), np.int64)
        if kind == GeomAbs_Cylinder:
            c = a.Cylinder()
            ax = c.Axis()
            p0 = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
            d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
            u0, u1 = a.FirstUParameter(), a.LastUParameter()
            v0, v1 = a.FirstVParameter(), a.LastVParameter()
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            pm = a.Value(um, vm)
            cyl.append({"r": float(c.Radius()), "p": p0, "d": d, "span": float(u1 - u0),
                        "ends": [p0 + v0 * d, p0 + v1 * d], "sample": np.array([pm.X(), pm.Y(), pm.Z()]),
                        "tris": tris})
        elif kind == GeomAbs_Plane:
            pl = a.Plane()
            n = np.array([pl.Axis().Direction().X(), pl.Axis().Direction().Y(), pl.Axis().Direction().Z()])
            if f.Orientation() == TopAbs_REVERSED:
                n = -n
            g = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, g)
            cm = g.CentreOfMass()
            planes.append({"n": n, "c": np.array([cm.X(), cm.Y(), cm.Z()]), "area": float(g.Mass()),
                           "tris": tris})
    return cyl, planes


def _mesh_surfaces(mesh):
    """Fitted cylinders and planes of a mesh. Cylinders: smooth patches
    (dihedral under 25 degrees) whose normals lie square to one axis and
    whose points fit a circle, several per patch when fillets join them.
    Planes: coplanar facets of three or more triangles that no cylinder
    claimed (a finely cut cylinder's strips are coplanar facets too)."""
    import trimesh

    planes, cyl = [], []
    adj = mesh.face_adjacency
    ok = mesh.face_adjacency_angles < math.radians(25)
    claimed = np.zeros(len(mesh.faces), bool)
    for comp in trimesh.graph.connected_components(adj[ok], nodes=np.arange(len(mesh.faces)), min_len=6):
        left = np.asarray(comp)
        for _ in range(4):
            if len(left) < 6:
                break
            fit = _fit_cylinder(mesh, left)
            if fit is None:
                break
            cyl.append(fit)
            claimed[fit["tris"]] = True
            left = left[~claimed[left]]
    for fac, n, area in zip(mesh.facets, mesh.facets_normal, mesh.facets_area):
        if len(fac) < 3 or claimed[fac].mean() > 0.5:
            continue
        w = mesh.area_faces[fac]
        planes.append({"n": np.asarray(n, float), "c": (mesh.triangles_center[fac] * w[:, None]).sum(0) / w.sum(),
                       "area": float(area), "tris": np.asarray(fac)})
    return cyl, planes


def _fit_cylinder(mesh, comp):
    """A cylinder fitted to a smooth patch of triangles, or None. Faces
    whose normals leave the axis's plane (a fillet rolling onto a face) are
    peeled off and the axis refitted; then points off the circle (the far
    side of a fillet, a flat beside it) are trimmed."""
    for _ in range(4):
        nrm = mesh.face_normals[comp]
        w = mesh.area_faces[comp]
        M = (nrm * w[:, None]).T @ nrm
        evals, evecs = np.linalg.eigh(M)
        d = evecs[:, 0]
        off = np.abs(nrm @ d) > 0.2
        if not off.any():
            break
        comp = comp[~off]
        if len(comp) < 6:
            return None
    if evals[0] > 0.02 * evals.sum() or evals[1] < 0.05 * evals.sum():
        return None                       # not square to one axis, or flat
    vid = np.unique(mesh.faces[comp])
    P = mesh.vertices[vid]
    e1 = _unit(np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0]))
    e2 = np.cross(d, e1)
    x, y = P @ e1, P @ e2
    inl = np.ones(len(P), bool)
    for _ in range(3):
        A = np.stack([x[inl], y[inl], np.ones(inl.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, x[inl] ** 2 + y[inl] ** 2, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
        if not 0 < r < 1e4:
            return None
        resid = np.abs(np.hypot(x - cx, y - cy) - r)
        inl = resid <= max(0.01 * r, 0.02)
        if inl.sum() < 6:
            return None
    if inl.sum() < 6:
        return None
    on = np.zeros(len(mesh.vertices), bool)
    on[vid[inl]] = True
    comp = comp[on[mesh.faces[comp]].all(1)]
    if len(comp) < 4 or (inl.mean() < 0.7 and len(comp) < 12):
        return None
    a = np.sort(np.arctan2(y[inl] - cy, x[inl] - cx))
    gaps = np.diff(np.concatenate([a, a[:1] + 2 * math.pi]))
    t = P[inl] @ d
    p0 = cx * e1 + cy * e2
    w = mesh.area_faces[comp]
    return {"r": float(r), "p": p0, "d": d, "span": float(2 * math.pi - gaps.max()),
            "ends": [p0 + t.min() * d, p0 + t.max() * d],
            "sample": mesh.triangles_center[comp[np.argmax(w)]], "tris": comp}


# ---- features -----------------------------------------------------------------------

def _group_cylinders(mesh, cyl, scale):
    """Merge cylinder faces into holes (and bosses): same line, same radius,
    then coaxial steps into one bore."""
    tol_p, tol_r = 1e-3 * scale + 0.01, 0.02
    groups = []
    for c in cyl:
        p, d = _canon_axis(c["p"], c["d"])
        for g in groups:
            if abs(abs(np.dot(g["d"], d)) - 1) < 1e-4 and np.linalg.norm(g["p"] - p) < tol_p \
                    and abs(g["r"] - c["r"]) < tol_r:
                g["faces"].append(c)
                break
        else:
            groups.append({"p": p, "d": d, "r": c["r"], "faces": [c]})
    holes, bosses = [], []
    for g in groups:
        span = sum(f["span"] for f in g["faces"])
        if span < FULL_SPAN:
            continue                          # a fillet, a slot end, a partial arc
        ts = np.concatenate([[(e - g["p"]) @ g["d"] for e in f["ends"]] for f in g["faces"]])
        g["t0"], g["t1"] = float(ts.min()), float(ts.max())
        if g["t1"] - g["t0"] < 1e-3:
            continue
        samples = np.array([f["sample"] for f in g["faces"]])
        hole = _concave(mesh, samples, g["p"], g["d"], g["r"])
        g["tris"] = np.concatenate([f["tris"] for f in g["faces"]])
        (holes if hole.mean() > 0.5 else bosses).append(g)
    # coaxial holes: one bore with steps
    bores = []
    for h in sorted(holes, key=lambda h: h["r"]):
        for b in bores:
            if abs(abs(np.dot(b["d"], h["d"])) - 1) < 1e-4 and np.linalg.norm(b["p"] - h["p"]) < tol_p:
                b["segments"].append(h)
                break
        else:
            bores.append({"p": h["p"], "d": h["d"], "segments": [h]})
    return bores, bosses


def _inside(mesh, pts):
    return mesh.contains(np.asarray(pts, float).reshape(-1, 3))


def _bore_record(mesh, b, i, scale):
    segs = sorted(b["segments"], key=lambda s: s["t0"])
    d, p = b["d"], b["p"]
    t0 = min(s["t0"] for s in segs)
    t1 = max(s["t1"] for s in segs)
    r_min = min(s["r"] for s in segs)
    eps = 0.5
    through = not bool(_inside(mesh, [p + (t0 - eps) * d, p + (t1 + eps) * d]).any())
    return {"id": f"bore-{i}", "kind": "bolt" if r_min <= BOLT_MAX_R else "bore",
            "axis": d.round(6).tolist(), "center": (p + 0.5 * (t0 + t1) * d).round(4).tolist(),
            "p0": (p + t0 * d).round(4).tolist(), "p1": (p + t1 * d).round(4).tolist(),
            "r": round(r_min, 4), "length": round(t1 - t0, 4), "through": through,
            "segments": [{"r": round(s["r"], 4), "p0": (p + s["t0"] * d).round(4).tolist(),
                          "p1": (p + s["t1"] * d).round(4).tolist()} for s in segs]}


def _exposed(mesh, plane, samples=12):
    """Share of the face that sees nothing of the part along its normal."""
    tris = plane["tris"]
    if len(tris) == 0:
        return 0.0
    rng = np.random.default_rng(0)
    pick = tris if len(tris) <= samples else rng.choice(tris, samples, replace=False, p=None)
    origins = mesh.triangles_center[pick] + plane["n"] * 1e-3
    dirs = np.repeat(plane["n"][None], len(origins), 0)
    hit = mesh.ray.intersects_any(origins, dirs)
    return float(1.0 - hit.mean())


def detect_features(part: Part) -> dict:
    """Bores, flat mounting faces and interfaces of a part (see the module
    docstring). Units mm."""
    mesh = part.mesh
    scale = float(np.max(mesh.extents))
    if part.brep is not None:
        cyl, planes = _brep_surfaces(part)
        source = "brep"
    else:
        cyl, planes = _mesh_surfaces(mesh)
        source = "mesh"
    raw_bores, bosses = _group_cylinders(mesh, cyl, scale)
    raw_bores.sort(key=lambda b: (-max(s["r"] for s in b["segments"]),
                                  tuple(np.round(b["p"] + b["segments"][0]["t0"] * b["d"], 3))))
    bores = [_bore_record(mesh, b, i, scale) for i, b in enumerate(raw_bores)]

    # mounting faces: exposed planes big enough to sit on
    total = float(mesh.area)
    # big enough to bolt to: 1% of the surface (a chamfer or a step is not
    # a mounting face; a plain L bracket read 14 of them at 0.2%)
    min_area = max(10.0, 0.01 * total)
    faces = []
    for pl in sorted(planes, key=lambda q: -q["area"]):
        if pl["area"] < min_area:
            continue
        exp = _exposed(mesh, pl)
        if exp < 0.6:
            continue
        n, c = _unit(pl["n"]), pl["c"]
        pts = mesh.vertices[np.unique(mesh.faces[pl["tris"]])] if len(pl["tris"]) else c[None]
        holes = []
        for b in bores:
            d = np.asarray(b["axis"])
            if abs(np.dot(d, n)) < 0.98:
                continue
            ends = [np.asarray(b["p0"]), np.asarray(b["p1"])]
            for s in b["segments"]:
                ends += [np.asarray(s["p0"]), np.asarray(s["p1"])]
            if min(abs(np.dot(e - c, n)) for e in ends) > 0.05:
                continue
            # where the axis meets the plane, the face lies within a wall's reach
            q = np.asarray(b["center"]) + d * np.dot(c - np.asarray(b["center"]), n) / np.dot(d, n)
            rmax = max(s["r"] for s in b["segments"])
            if np.min(np.linalg.norm(pts - q, axis=1)) <= rmax + 3.0:
                holes.append(b["id"])
        faces.append({"id": f"face-{len(faces)}", "normal": n.round(6).tolist(), "center": c.round(4).tolist(),
                      "area": round(pl["area"], 2), "exposed": round(exp, 2), "bores": holes,
                      "_tris": pl["tris"]})

    # interfaces: bolted faces, loose bolt patterns, bores
    by_id = {b["id"]: b for b in bores}
    used = set()
    interfaces = []
    for f in faces:
        bolts = [h for h in f["bores"] if by_id[h]["kind"] == "bolt" and h not in used]
        big = [h for h in f["bores"] if by_id[h]["kind"] == "bore"]
        if len(bolts) >= 2 or (bolts and f["area"] > 0.02 * total):
            used.update(bolts)
            interfaces.append({"id": f"{f['id']}-mount", "kind": "bolted-face", "face": f["id"],
                               "bores": bolts, "alsoOnFace": big})
    loose = [b for b in bores if b["kind"] == "bolt" and b["id"] not in used]
    while loose:
        b0 = loose.pop(0)
        d0 = np.asarray(b0["axis"])
        grp = [b0]
        for b in list(loose):
            if abs(abs(np.dot(np.asarray(b["axis"]), d0)) - 1) < 1e-3 and abs(b["r"] - b0["r"]) < 0.05 and \
                    abs(np.dot(np.asarray(b["p0"]) - np.asarray(b0["p0"]), d0)) < 0.5:
                grp.append(b)
                loose.remove(b)
        used.update(b["id"] for b in grp)
        interfaces.append({"id": f"{grp[0]['id']}-pattern" if len(grp) > 1 else grp[0]["id"],
                           "kind": "bolt-pattern" if len(grp) > 1 else "bolt", "bores": [b["id"] for b in grp]})
    for b in bores:
        if b["kind"] == "bore":
            interfaces.append({"id": b["id"], "kind": "bore", "bores": [b["id"]]})
    for it in interfaces:
        cs = np.array([by_id[h]["center"] for h in it["bores"]]) if it["bores"] else None
        if it["kind"] == "bolted-face":
            # centred on its bolts, which carry the load, not on the face
            fc = next(f for f in faces if f["id"] == it["face"])
            it["center"] = cs.mean(0).round(4).tolist()
            it["normal"] = fc["normal"]
        else:
            it["center"] = cs.mean(0).round(4).tolist()
            it["normal"] = by_id[it["bores"][0]]["axis"]
    out = {"source": source, "units": "mm", "bounds": np.asarray(mesh.bounds).round(4).tolist(),
           "volume": round(float(mesh.volume), 1), "area": round(total, 1),
           "bores": bores, "bosses": [{"axis": b["d"].round(6).tolist(), "r": round(b["r"], 4),
                                       "p0": (b["p"] + b["t0"] * b["d"]).round(4).tolist(),
                                       "p1": (b["p"] + b["t1"] * b["d"]).round(4).tolist()} for b in bosses],
           "faces": faces, "interfaces": interfaces}
    if part.healed.get("loops"):
        out["healed"] = part.healed
    return out


def _interface_size(it, feats):
    """How much an interface carries, roughly: bearing area of its bores
    (r x length, mm^2) plus a tenth of its face."""
    by_id = {b["id"]: b for b in feats["bores"]}
    s = sum(by_id[h]["r"] * by_id[h]["length"] for h in it["bores"])
    if it["kind"] == "bolted-face":
        s += 0.1 * next(f["area"] for f in feats["faces"] if f["id"] == it["face"])
    return s


def suggest_setup(feats: dict, force: float = 100.0) -> dict:
    """A support and a load to start from. A part is held by its bolts:
    the bolted face (or bolt pattern) with the most bolts holds it, the
    largest by area on a tie (any interface when nothing is bolted). The
    load goes where the most is carried farthest from that (size x lever:
    a bearing seat or a socket across the part), `force` N square to the
    lever, as close to straight down (-z) as that allows. A placeholder for
    the real loads, which only the machine knows."""
    its = feats["interfaces"]
    if not its:
        return {}
    bolted = [it for it in its if it["kind"] in ("bolted-face", "bolt-pattern")]
    pool = bolted or its
    hold = max(pool, key=lambda it: (len(it["bores"]) if bolted else 0, _interface_size(it, feats)))
    c0 = np.asarray(hold["center"])
    rest = [it for it in its if it is not hold]
    if not rest:
        return {"support": hold["id"]}
    load = max(rest, key=lambda it: _interface_size(it, feats) * np.linalg.norm(np.asarray(it["center"]) - c0))
    lever = np.asarray(load["center"]) - c0
    lever = lever / max(np.linalg.norm(lever), 1e-9)
    g = np.array([0.0, 0.0, -1.0])
    dirn = g - np.dot(g, lever) * lever
    if np.linalg.norm(dirn) < 0.3:
        a = np.asarray(load["normal"], float)
        dirn = np.cross(lever, a)
        if np.linalg.norm(dirn) < 1e-6:
            dirn = np.cross(lever, [1.0, 0.0, 0.0] if abs(lever[0]) < 0.9 else [0.0, 1.0, 0.0])
    dirn = dirn / np.linalg.norm(dirn)
    return {"support": hold["id"], "load": load["id"], "force": (force * dirn).round(3).tolist(),
            "lever": round(float(np.linalg.norm(np.asarray(load["center"]) - c0)), 2),
            "note": f"placeholder {force:g} N: replace with the part's real loads"}


# ---- the draft study ----------------------------------------------------------------

def _extrude(mesh, tris, normal, depth):
    """The triangles `tris` of `mesh` (a flat face) swept `depth` mm along
    `normal`, closed: a pad behind a face (depth < 0) or a clearance slab in
    front of it."""
    import trimesh

    F = mesh.faces[tris]
    vid, inv = np.unique(F, return_inverse=True)
    top = mesh.vertices[vid]
    bot = top + depth * np.asarray(normal)
    n = len(vid)
    f_top = inv.reshape(-1, 3)
    # boundary half-edges of the face
    e = np.concatenate([f_top[:, [0, 1]], f_top[:, [1, 2]], f_top[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, ix, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    be = e[cnt[ix.ravel()] == 1]
    sides = np.concatenate([np.stack([be[:, 1], be[:, 0], be[:, 0] + n], 1),
                            np.stack([be[:, 1], be[:, 0] + n, be[:, 1] + n], 1)])
    faces = np.concatenate([f_top, f_top[:, ::-1] + n, sides])
    m = trimesh.Trimesh(np.concatenate([top, bot]), faces, process=True)
    if m.volume < 0:
        m.invert()
    return m


def _tube(seg, wall, extend=0.0):
    from .shapes import Cylinder
    p0, p1 = np.asarray(seg["p0"], float), np.asarray(seg["p1"], float)
    a = _unit(p1 - p0)
    return Cylinder(p0 - extend * a, p1 + extend * a, seg["r"] + wall, 0.0)


def _load_pairs(loads):
    """[(interface id, force)] from a list of pairs or {id, force} dicts, or
    a dict {id: force}."""
    if not loads:
        return []
    if isinstance(loads, dict):
        return [(k, v) for k, v in loads.items()]
    out = []
    for x in loads:
        if isinstance(x, dict):
            out.append((x.get("interface") or x.get("region") or x.get("id"), x["force"]))
        else:
            out.append((x[0], x[1]))
    for k, f in out:
        if len(f) != 3:
            raise ValueError(f"load on {k!r}: force is [fx, fy, fz] in N")
    return out


def washer_r(r):
    """mm: the radius a bolt's head or washer bears on, for a clearance
    hole of radius r (ISO metric, fasteners.METRIC; about 1.9 r between)."""
    from .fasteners import METRIC
    best = min(METRIC.values(), key=lambda v: abs(v[0] / 2 - r))
    return best[1] / 2 if abs(best[0] / 2 - r) < 0.4 else 1.9 * r


def _bolt_ends(b, face_n=None, face_c=None):
    """(head end, far end, head-recess radius) of a bolt bore. The head sits
    in a counterbore when there is one (at its step); else on the end away
    from the face the part bolts against."""
    p0, p1 = np.asarray(b["p0"], float), np.asarray(b["p1"], float)
    segs = b["segments"]
    if len(segs) > 1:
        cb = max(segs, key=lambda s: s["r"])
        q0, q1 = np.asarray(cb["p0"], float), np.asarray(cb["p1"], float)
        # the counterbore's outer end is the bore's own end; its other end is the step
        if np.linalg.norm(q0 - p0) < np.linalg.norm(q1 - p1):
            return q1, p1, cb["r"]
        return q0, p0, cb["r"]
    if face_n is not None and abs(np.dot(p1 - face_c, face_n)) < abs(np.dot(p0 - face_c, face_n)):
        return p0, p1, b["r"]
    return p1, p0, b["r"]


def ring_wall(r):
    """mm of material kept round a bore of radius r: three FDM perimeters
    round a bolt, a little more round a bearing seat."""
    return float(np.clip(2.0 + 0.1 * r, 2.0, 5.0))


def _write_stl(mesh, path):
    mesh.export(path)
    return os.path.basename(path)


def draft_study(part: Part, feats: dict, out_dir: str, *, name: str | None = None, space: str = "part",
                material: str = "petg", force: float = 100.0, pad: float = 2.0, grow: float = 0.0,
                safety_factor: float = 2.0, process: str | None = None, support: str | None = None,
                loads=None, ring: float | None = None) -> dict:
    """Write a study for the part into `out_dir` (design space, preserved
    interfaces and clearance slabs as STL beside it) and return it.
    `support` (an interface id) and `loads` ([(interface id, [fx, fy, fz] N)],
    or a dict of them; each its own load case) replace the suggested setup;
    the suggestion stands for whichever is not given. `ring` (mm): every bore
    keeps an exact tube of that wall (a cylinder in the study, not cut to the
    part) instead of the default ring."""
    import trimesh

    from .shapes import Cylinder, manifold_from_trimesh, trimesh_from_manifold

    if space not in ("part", "hull", "box"):
        raise ValueError("space is 'part' (the part as drawn), 'hull' (its convex hull) or 'box'")
    os.makedirs(out_dir, exist_ok=True)
    mesh = part.mesh
    stem = name or os.path.splitext(os.path.basename(part.path or "part"))[0]
    by_id = {b["id"]: b for b in feats["bores"]}
    faces = {f["id"]: f for f in feats["faces"]}

    if space == "part":
        ds_mesh = mesh
    elif space == "hull":
        ds_mesh = mesh.convex_hull
    else:
        lo, hi = mesh.bounds
        ds_mesh = trimesh.creation.box(bounds=[lo - grow, hi + grow])
    if grow > 0 and space != "box":
        from .shapes import MeshShape, grow as grow_shape
        ds_spec_shape = grow_shape(MeshShape(ds_mesh), grow)
        ds_mesh = trimesh_from_manifold(ds_spec_shape.to_manifold())
    design = [{"type": "mesh", "path": _write_stl(ds_mesh, os.path.join(out_dir, f"{stem}-design-space.stl"))}]
    part_m = manifold_from_trimesh(mesh)

    preserve, keep_out = [], []
    lo, hi = np.asarray(ds_mesh.bounds[0]), np.asarray(ds_mesh.bounds[1])
    reach = float(np.linalg.norm(hi - lo))

    def bore_cyls(b, extend):
        segs = [dict(s) for s in b["segments"]]
        out = []
        for s in segs:
            p0, p1 = np.asarray(s["p0"]), np.asarray(s["p1"])
            a = _unit(p1 - p0)
            e0 = e1 = 0.5 if b["through"] else 0.0
            if extend and b["through"]:
                e0 = e1 = reach
            out.append({"type": "cylinder", "p0": (p0 - e0 * a).round(4).tolist(),
                        "p1": (p1 + e1 * a).round(4).tolist(), "r": s["r"]})
        return out

    grown = space != "part" or grow > 0
    setup = suggest_setup(feats, force)
    ids = {it["id"] for it in feats["interfaces"]}
    pairs = _load_pairs(loads)
    for k in [support] + [p[0] for p in pairs]:
        if k is not None and k not in ids:
            raise ValueError(f"no interface {k!r}; this part has {sorted(ids)} (see features.json)")
    used = {support or setup.get("support")} | ({p[0] for p in pairs} if pairs else {setup.get("load")})
    for it in feats["interfaces"]:
        bores = [by_id[h] for h in it["bores"]]
        # grown past the part, a through bore runs out of it: the bolt goes in,
        # the shaft goes on through
        cyls = [c for b in bores for c in bore_cyls(b, grown)]
        keep_out.append({"id": f"{it['id']}-hole" if it["kind"] == "bolted-face" or len(bores) > 1
                         else it["id"],
                         "shape": cyls[0] if len(cyls) == 1 else {"type": "union", "shapes": cyls}})
        if it["kind"] == "bore" and bores[0]["r"] < SEAT_MIN_R and it["id"] not in used:
            continue                  # a cable pass or a pilot hole: kept open, not kept solid
        # preserved: rings round the bores, a pad under a bolted face, cut to the part
        keep = []
        face_n = face_c = None
        if it["kind"] == "bolted-face":
            f = faces[it["face"]]
            face_n, face_c = np.asarray(f["normal"]), np.asarray(f["center"])
        for b in bores:
            if b["kind"] == "bolt":
                # a bolt needs a flat seat under its head and where it clamps,
                # not a boss: a washer-sized ring `pad` mm deep at each (the way
                # fasteners.py preserves a bolted joint)
                head, far, r_head = _bolt_ends(b, face_n, face_c)
                a = _unit(far - head)
                rw = washer_r(b["r"])
                keep.append(Cylinder(head - 0.5 * a, head + pad * a, rw, 0.0).to_manifold())
                keep.append(Cylinder(far - pad * a, far + 0.5 * a, rw, 0.0).to_manifold())
                if grown:
                    # the head and the driver come in from outside
                    keep_out.append({"id": f"{b['id']}-driver", "bore": False, "shape": {
                        "type": "cylinder", "p0": (head - reach * a).round(4).tolist(),
                        "p1": head.round(4).tolist(), "r": round(max(r_head, rw), 4)}})
                continue
            if ring:
                for s in b["segments"]:
                    p0, p1 = np.asarray(s["p0"]), np.asarray(s["p1"])
                    preserve.append({"id": f"{b['id']}-ring" + (f"-{len(preserve)}" if len(b["segments"]) > 1 else ""),
                                     "shape": {"type": "cylinder", "p0": p0.round(4).tolist(),
                                               "p1": p1.round(4).tolist(), "r": round(s["r"] + ring, 4),
                                               "rInner": s["r"]}})
                continue
            w = ring_wall(b["r"])
            for s in b["segments"]:
                # past the bore's ends, so no cap lies on the part's own face
                # (coplanar caps leave slivers that do not survive an STL)
                keep.append(_tube(s, w, extend=0.5).to_manifold())
        if it["kind"] == "bolted-face" and grown:
            # don't grow over the mating face: clear a slab in front of it
            slab = _extrude(mesh, f["_tris"], face_n, max(grow, 0.0) + reach)
            keep_out.append({"id": f"{it['id']}-clear", "bore": False,
                             "shape": {"type": "mesh", "path": _write_stl(
                                 slab, os.path.join(out_dir, f"{stem}-{it['id']}-clear.stl"))}})
        if not keep:
            continue
        acc = keep[0]
        for k in keep[1:]:
            acc = acc + k
        kept = trimesh_from_manifold(acc ^ part_m)
        if len(kept.faces) == 0:
            continue
        preserve.append({"id": f"{it['id']}-keep",
                         "shape": {"type": "mesh", "path": _write_stl(
                             kept, os.path.join(out_dir, f"{stem}-{it['id']}-keep.stl"))}})
    # every other bore stays a hole
    covered = {h for it in feats["interfaces"] for h in it["bores"]}
    for b in feats["bores"]:
        if b["id"] not in covered:
            cyls = bore_cyls(b, False)
            keep_out.append({"id": b["id"], "shape": cyls[0] if len(cyls) == 1 else
                             {"type": "union", "shapes": cyls}})

    study = {"name": stem,
             "description": f"Drafted by `loadpath import` from {os.path.basename(part.path or 'a mesh')}: "
                            f"{len(feats['bores'])} bores, {len(feats['faces'])} mounting faces, "
                            f"{len(feats['interfaces'])} interfaces. Check the loads before trusting a result.",
             "designSpace": design, "preserve": preserve, "keepOut": keep_out,
             "material": material, "goal": {"safetyFactor": safety_factor}}
    if process:
        study["process"] = process
    ko_of = {}
    for it in feats["interfaces"]:
        bores = it["bores"]
        ko_of[it["id"]] = f"{it['id']}-hole" if it["kind"] == "bolted-face" or len(bores) > 1 else it["id"]
    hold = support or setup.get("support")
    if hold:
        study["supports"] = [{"region": ko_of[hold]}]
    given = bool(pairs)
    if not pairs and setup.get("load"):
        pairs = [(setup["load"], setup["force"])]
    cases = []
    for k, (iid, fv) in enumerate(pairs):
        it = next(i for i in feats["interfaces"] if i["id"] == iid)
        ld = {"region": ko_of[iid], "force": [float(x) for x in fv]}
        if it["kind"] == "bore" and len(by_id[it["bores"][0]]["segments"]) == 1:
            ld["distribution"] = "bearing"
        cases.append({"name": f"load-{k + 1}" if given else "suggested", "loads": [ld]})
    if cases:
        study["loadCases"] = cases
    feats_out = {k: v for k, v in feats.items() if k != "faces"}
    feats_out["faces"] = [{k: v for k, v in f.items() if not k.startswith("_")} for f in feats["faces"]]
    feats_out["suggested"] = setup
    with open(os.path.join(out_dir, f"{stem}.study.json"), "w") as fh:
        json.dump(study, fh, indent=1)
    with open(os.path.join(out_dir, f"{stem}.features.json"), "w") as fh:
        json.dump(feats_out, fh, indent=1)
    return study


def summarize(feats: dict) -> dict:
    """The short read of a part's features: counts, and each interface in a
    line an agent can pick from."""
    by_id = {b["id"]: b for b in feats["bores"]}
    rows = []
    for it in feats["interfaces"]:
        bs = [by_id[h] for h in it["bores"]]
        if it["kind"] == "bore":
            b = bs[0]
            what = f"bore r {b['r']:g} mm x {b['length']:g} mm, {'through' if b['through'] else 'blind'}"
        else:
            rs = sorted({round(b["r"], 2) for b in bs})
            what = f"{len(bs)} bolt hole(s) r {'/'.join(f'{r:g}' for r in rs)} mm" + (
                f" on {it['face']}" if it["kind"] == "bolted-face" else "")
        rows.append({"id": it["id"], "kind": it["kind"], "what": what, "center": it["center"]})
    return {"bolts": sum(b["kind"] == "bolt" for b in feats["bores"]),
            "bores": sum(b["kind"] == "bore" for b in feats["bores"]),
            "boltedFaces": sum(it["kind"] == "bolted-face" for it in feats["interfaces"]),
            "mountingFaces": len(feats["faces"]), "interfaces": rows}


def import_part(path, out_dir: str | None = None, *, tolerance: float = DEFAULT_TOLERANCE, **kw) -> dict:
    """Read a part, find its features and (with `out_dir`) draft its study.
    Returns {"features", "study"?}."""
    part = load_part(path, tolerance)
    feats = detect_features(part)
    out = {"features": {k: v for k, v in feats.items() if k != "faces"}}
    out["features"]["faces"] = [{k: v for k, v in f.items() if not k.startswith("_")} for f in feats["faces"]]
    out["features"]["suggested"] = suggest_setup(feats, kw.get("force", 100.0))
    out["summary"] = summarize(out["features"])
    if out_dir is not None:
        out["study"] = draft_study(part, feats, out_dir, **kw)
    return out
