"""
Lattice parts as smooth, exact CAD (STEP), lattice and all.

Two routes (export_lattice_step picks one):

cavities - a TPMS sheet lattice sealed under a skin splits the part's inside
into cavities, the channels on each side of the lattice's walls. So

    part = envelope - cavities

where the envelope is the part with the lattice solid (step.export_step:
exact planes and cylinders, smooth patches elsewhere) and each cavity is
bounded by one smooth level set: the part's field with the lattice replaced
by one side of its wall, capped `inset` inside the envelope. That level set
is the wall's side, the skin's inner face and the fillets between them as
one surface with no crease (the sheet's two sides never meet), so it goes
straight to bicubic patches on the true field (ibrep.py), every patch whole:
no boolean and no trimmed face, the cavities are the solid's inner shells.
Drain bores are cut afterwards (one OpenCascade cut of plain cylinders).

void - anything else (strut cells, Voronoi foams, networks, several
lattices, lattices open at the outside): part = envelope - void, the void
one smooth level set of the part's own field carried on past the envelope
(void_family), meshed with patches half the thinnest strut. Sealed void is
inner shells; void that reaches the outside is cut by splitting the faces
where they cross and keeping pieces by the fields (_split_cut), not by
OpenCascade's cut, which crawled or failed on lattice blocks.
"""
from __future__ import annotations

import os
import time

import numpy as np

# ---- skinned lattices: the envelope less its cavities

def cavity_families(doc, g, name, base_dir=None, sources=None, meshes=None, inset=0.4, h=None, extra=None):
    """The part's cavities on each side of lattice `name`'s walls as smooth
    implicit families (negative = material), and the envelope graph (the
    part with the lattice solid).

    With the lattice replaced by one side of its wall (d - t/2, or -d - t/2),
    the part's field is the part with the other side's channels filled: its
    zero set is the outside, the bores and this side's cavities, wall,
    skin and the fillets between them one smooth surface (the sheet's two
    sides never meet, so no crease). Everything within `inset` of the
    envelope's surface is made material, which leaves the cavities alone."""
    from .fields import _lattice_variant_graph, expand_groups
    from .ibrep import Family
    from .struts import is_strut

    full = expand_groups(doc)
    spec = dict(full["nodes"][name])
    if is_strut(spec.get("type", "gyroid")) or spec.get("form", "sheet") != "sheet":
        raise ValueError(f"lattice {name!r}: cavity STEP is for TPMS sheet lattices")
    cell = spec.get("cell", 8.0)
    cmin = float(cell) if isinstance(cell, (int, float)) else 4.0
    others = [k for k, nd in g.nodes.items() if nd.op == "lattice" and k != name]
    env = _lattice_variant_graph(doc, g, (name,), others, base_dir, sources, meshes, extra=extra)
    ins = float(inset)
    fams = []
    for side in ("_plus", "_minus"):
        gs = _lattice_variant_graph(doc, g, (), others, base_dir, sources, meshes,
                                    extra={**(extra or {}), name: {**spec, "form": side}})

        def f(P, gs=gs):
            return np.minimum(gs.evaluate(P), -(env.evaluate(P) + ins))

        fams.append(Family(f"{name}{'+' if side == '_plus' else '-'}", f, float(h or cmin / 6.0)))
    return fams, env


def _components(Q):
    """Connected components of a quad mesh (quads sharing an edge)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    E = np.concatenate([np.sort(Q[:, [k, (k + 1) % 4]], axis=1) for k in range(4)])
    owner = np.tile(np.arange(len(Q)), 4)
    key = E[:, 0] * (int(Q.max()) + 1) + E[:, 1]
    o = np.argsort(key, kind="stable")
    k_, ow = key[o], owner[o]
    same = k_[1:] == k_[:-1]
    G = sp.coo_matrix((np.ones(same.sum()), (ow[1:][same], ow[:-1][same])), shape=(len(Q), len(Q)))
    return connected_components(G, directed=False)[1]


def _closed(Q):
    """True when every edge of quad mesh Q has two quads."""
    E = np.sort(np.concatenate([Q[:, [k, (k + 1) % 4]] for k in range(4)]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    return bool((cnt == 2).all())


def _pieces(fam, lo, hi, levels=2, small=12):
    """The family's cavities as (Sheet, quads, h) pieces, one per closed
    component. A component small enough (under `small` h across) whose
    patches fold (the surface turns within a patch: a sliver cavity, a
    tight corner) is meshed again on its own box at half the spacing, up to
    `levels` times; a component wholly inside such a box is replaced by
    what the finer mesh finds there."""
    from .ibrep import Family, Sheet, manifold_nets

    def build(f, a, b):
        V, Q, nm = manifold_nets(f, a, b)
        if not len(Q):
            return None, [], nm
        sh = Sheet(f, V, Q)
        fo = sh.folds()
        comp = _components(Q)
        out = []
        for c in np.unique(comp):
            qs = np.flatnonzero(comp == c)
            P = V[np.unique(Q[qs])]
            out.append({"sheet": sh, "quads": qs, "h": f.h, "lo": P.min(0), "hi": P.max(0),
                        "folds": int(fo[qs].sum()), "closed": _closed(Q[qs])})
        return sh, out, nm

    sh0, pieces, nonman = build(fam, lo - 2 * fam.h, hi + 2 * fam.h)
    sheets = [sh0]
    refined = 0
    for lev in range(levels):
        todo = [p for p in pieces if p["folds"] and p["closed"] and p["h"] == fam.h / 2 ** lev
                and (p["hi"] - p["lo"]).max() < small * fam.h]
        if not todo:
            break
        boxes = [[p["lo"] - 2 * p["h"], p["hi"] + 2 * p["h"]] for p in todo]
        merged = True
        while merged:                                   # overlapping boxes are one
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if (boxes[i][0] <= boxes[j][1]).all() and (boxes[j][0] <= boxes[i][1]).all():
                        boxes[i] = [np.minimum(boxes[i][0], boxes[j][0]), np.maximum(boxes[i][1], boxes[j][1])]
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        h2 = fam.h / 2 ** (lev + 1)
        f2 = Family(fam.name, fam.f, h2)
        for a, b in boxes:
            inside = [p for p in pieces if (p["lo"] >= a).all() and (p["hi"] <= b).all()]
            if not inside:
                continue
            sh2, new, nm = build(f2, a, b)
            new = [p for p in new if p["closed"] and (p["lo"] > a + h2).all() and (p["hi"] < b - h2).all()]
            if not new:
                continue                                # the finer mesh is no better: keep the coarse one
            pieces = [p for p in pieces if not any(p is x for x in inside)] + new
            if sh2 is not None:
                sheets.append(sh2)
            nonman += nm
            refined += 1
    return pieces, sheets, nonman, refined


def cavity_shells(fam, lo, hi, progress=None):
    """Closed shells of smooth patches, one per cavity of family `fam`
    (surface nets + ibrep patches; every patch whole, edges shared, so they
    sew exactly; folded small cavities meshed again finer). Returns
    (shells, report)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCP.Geom import Geom_BezierSurface
    from OCP.OCP.collections import Array2_gp_Pnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from .ibrep import DEG
    from .step import _pnt

    t0 = time.time()
    pieces, sheets, nonman, refined = _pieces(fam, np.asarray(lo, float), np.asarray(hi, float))
    n = DEG + 1
    shells, open_, dev = [], 0, []
    for p in pieces:
        sh, qs = p["sheet"], p["quads"]
        if not p["closed"]:
            open_ += 1
            continue
        dev.append(sh.deviation_of(qs))
        sew = BRepBuilderAPI_Sewing(1e-4)
        for q in qs:
            arr = Array2_gp_Pnt(1, n, 1, n)
            for a in range(n):
                for b in range(n):
                    arr.SetValue(a + 1, b + 1, _pnt(sh.C[q, a, b]))
            sew.Add(BRepBuilderAPI_MakeFace(Geom_BezierSurface(arr), 1e-7).Face())
        sew.Perform()
        if sew.NbFreeEdges():
            open_ += 1
            continue
        ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
        while ex.More():
            shells.append(TopoDS.Shell(ex.Current()))
            ex.Next()
    dev = np.concatenate(dev) if dev else np.zeros(1)
    rep = {"family": fam.name, "h": fam.h, "patches": int(sum(len(p["quads"]) for p in pieces)),
           "cavities": len(pieces), "openCavities": open_, "refinedBoxes": refined,
           "foldedPatches": int(sum(p["folds"] for p in pieces)), "nonManifoldEdges": int(nonman),
           "chordEdges": int(sum(s.wild_edges for s in sheets)), "coonsPatches": int(sum(s.wild_patches for s in sheets)),
           "patchDeviation": {"p99": round(float(np.percentile(dev, 99)), 4), "max": round(float(dev.max()), 4)},
           "seconds": round(time.time() - t0, 1)}
    return shells, rep


def _drain_bores(g):
    """The graph's drain bores (drains.py) as (p0, p1, r), and the graph
    edit that leaves them out (the `drained` node without its bores)."""
    bores = [k for k in g.nodes if k.startswith("drain-bore-")]
    nd = g.nodes.get("drained")
    if not bores or nd is None:
        return [], None
    cyl = [(np.asarray(g.nodes[k].spec["shape"]["p0"], float), np.asarray(g.nodes[k].spec["shape"]["p1"], float),
            float(g.nodes[k].spec["shape"]["r"])) for k in bores]
    keep = [x for x in nd.spec["inputs"] if x not in bores]
    return cyl, {"drained": {"op": "union", "inputs": keep}}


def _cut_bores(shape, bores):
    """shape less the bore cylinders (one OpenCascade cut)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.OCP.collections import List_TopoDS_Shape

    args, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
    args.Append(shape)
    for p0, p1, r in bores:
        ax = p1 - p0
        L = float(np.linalg.norm(ax))
        tools.Append(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, ax / L))),
                                              r, L).Shape())
    cut = BRepAlgoAPI_Cut()
    cut.SetArguments(args)
    cut.SetTools(tools)
    cut.SetRunParallel(True)
    cut.Build()
    if not cut.IsDone():
        raise RuntimeError("drain bore cut failed")
    return cut.Shape()


def export_cavity_step(doc, g, out_dir, name, base_dir=None, sources=None, meshes=None, inset=0.4, h=None,
                       surface="smooth", part_mesh=None, progress=None, faces=None, patches=None):
    """<name>.step for a part whose TPMS sheet lattice is covered by a skin:
    the part with the lattice solid (step.export_step: exact faces, smooth
    patches) less the cavities on both sides of the lattice's walls, each a
    closed shell of smooth patches on the true field (cavity_shells), then
    less the drain bores. Returns (report, shape)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from .step import (
        DEFAULT_FACES,
        DEFAULT_PATCHES,
        _volume,
        export_step,
        read_step,
        write_step,
    )

    t0 = time.time()
    lattices = [k for k, nd in g.nodes.items() if nd.op == "lattice"]
    if len(lattices) != 1:
        raise ValueError("cavity STEP takes a part with one lattice node")
    k = lattices[0]
    bores, extra = _drain_bores(g)
    fams, env = cavity_families(doc, g, k, base_dir, sources, meshes, inset, h, extra)
    em = env.mesh(progress=progress)
    st = export_step(em, os.path.join(out_dir, f"{name}-envelope.step"), faces=int(faces or DEFAULT_FACES),
                     name=name, surface=surface, patches=int(patches or DEFAULT_PATCHES))
    outer = st.pop("shape")
    lo, hi = np.asarray(em.bounds[0], float), np.asarray(em.bounds[1], float)
    shells, reps = [], []
    for f in fams:
        s, r = cavity_shells(f, lo, hi, progress)
        shells += s
        reps.append(r)
    ex = TopExp_Explorer(outer, TopAbs_SHELL)
    outs = []
    while ex.More():
        outs.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    mk = BRepBuilderAPI_MakeSolid(outs[0])
    for s in outs[1:]:
        mk.Add(s)
    islands, dropped, voids = [], [], []
    for s in shells:
        so = BRepBuilderAPI_MakeSolid(s).Solid()
        v = _volume(so)
        if abs(v) < 0.01:
            continue                                   # a speck where the surface grazes a grid node
        if v < 0:
            voids.append((s, v))                       # a cavity: faces point into it
        elif BRepCheck_Analyzer(so).IsValid():
            islands.append(so)                         # material loose inside a cavity (a cut-off wall)
        else:
            dropped.append(round(v, 2))
    for s, _ in voids:
        mk.Add(s)
    solid = mk.Solid()
    filled = []
    if not BRepCheck_Analyzer(solid, False).IsValid():
        # a cavity OpenCascade cannot place (its classifier miscounts on a
        # shell of a few strongly curved patches) is left solid
        keep = []
        for s, v in voids:
            m1 = BRepBuilderAPI_MakeSolid(outs[0])
            m1.Add(s)
            (keep if BRepCheck_Analyzer(m1.Solid(), False).IsValid() else filled).append((s, v))
        mk = BRepBuilderAPI_MakeSolid(outs[0])
        for s_ in outs[1:]:
            mk.Add(s_)
        for s, _ in keep:
            mk.Add(s)
        solid = mk.Solid()
        voids = keep
        filled = [round(-v, 2) for _, v in filled]
    n_cav = len(voids)
    if islands:
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound

        bb = BRep_Builder()
        comp = TopoDS_Compound()
        bb.MakeCompound(comp)
        for x in [solid] + islands:
            bb.Add(comp, x)
        solid = comp
    if bores:
        solid = _cut_bores(solid, bores)
    valid = bool(BRepCheck_Analyzer(solid).IsValid())
    rep = {"export": "envelope less cavities", "lattice": k, "inset": inset, "envelope": st,
           "cavities": reps, "cavityShells": n_cav, "looseBodies": len(islands),
           "droppedBodies": dropped, "filledCavities": filled, "drainBores": len(bores), "valid": valid, "volume": round(abs(_volume(solid)), 2),
           "envelopeVolume": round(abs(_volume(outer)), 2)}
    if part_mesh is not None:
        rep["meshVolume"] = round(float(abs(part_mesh.volume)), 2)
    rep.update(write_step(solid, os.path.join(out_dir, f"{name}.step"), name=name))
    t1 = time.time()
    back = read_step(rep["path"])
    rep["readBack"] = {"seconds": round(time.time() - t1, 1), "volume": round(abs(_volume(back)), 2),
                       "valid": bool(BRepCheck_Analyzer(back).IsValid())}
    rep["readBack"]["ok"] = rep["readBack"]["valid"] and abs(rep["readBack"]["volume"] - rep["volume"]) < 1e-3 * rep["volume"]
    rep["seconds"] = round(time.time() - t0, 1)
    return rep, solid


# ---- any lattice: the envelope less one void solid ------------------------

def _feature(g, names, P):
    """The finest lattice feature (mm) over points P: the smallest strut
    diameter or wall (5th percentile of the thickness node there) and the
    smallest cell."""
    from .struts import canonical, is_strut, strut_diameter_for_density

    feat, cells = [], []
    memo = {}
    for k in names:
        nd = g.nodes[k]
        s, r = nd.spec, nd.refs
        cell = np.broadcast_to(np.asarray(g._val(r["cell"], P, memo) if "cell" in r else s.get("cell", 8.0),
                                          float), (len(P),))
        if "thickness" in r:
            t = np.broadcast_to(np.asarray(g._val(r["thickness"], P, memo), float), (len(P),))
        elif "thickness" in s:
            t = np.full(len(P), float(s["thickness"]))
        elif "density" in r and is_strut(s.get("type", "gyroid")):
            rho = np.clip(np.broadcast_to(np.asarray(g._val(r["density"], P, memo), float), (len(P),)), 0.02, 0.9)
            t = strut_diameter_for_density(canonical(s["type"]), rho, cell, float(s.get("blend", 0.0) or 0.0))
        else:
            t = 0.12 * cell
        feat.append(float(np.percentile(t, 5)))
        cells.append(float(np.percentile(cell, 5)))
    return min(feat), min(cells)


def void_family(P_graph, E_graph, B_graph, lattices, h, cap=None, eta=None):
    """The part's void as a smooth implicit family (negative = material).
    Inside the envelope E (the part with its lattices solid) it is the
    part's own field. Outside, it carries on what the envelope cut: the
    raw lattice (struts and walls run on past E and cross its surface
    square instead of ending on it), except over the body (B: the part
    with its lattices gone; its value at the nearest point just inside E,
    Newton onto E's surface then eta in), which stays material so no void
    face lies on a skin. Capped `cap` outside E, the cap's crease rounded
    (all of that is outside, where the envelope cut takes it off). The
    void solid {f > 0} is what the envelope loses: sealed cavities inside
    it, open pores that reach through its surface."""
    from .fields import _smin
    from .ibrep import Family

    cap = float(cap if cap is not None else 2.0 * h)
    eta = float(eta if eta is not None else 0.25 * h)
    k = 0.5 * cap
    eps = 1e-4

    def vgE(Q):
        n = len(Q)
        X = np.concatenate([Q] + [Q + d for d in np.eye(3) * eps] + [Q - d for d in np.eye(3) * eps])
        v = E_graph.evaluate(X)
        gr = np.stack([(v[(1 + a) * n:(2 + a) * n] - v[(4 + a) * n:(5 + a) * n]) / (2 * eps) for a in range(3)], 1)
        return v[:n], gr

    def f(X):
        X = np.asarray(X, float)
        e = E_graph.evaluate(X)
        out = cap - e
        near = e < cap + k
        if not near.any():
            return out
        idx = np.flatnonzero(near)
        ei = e[idx]
        p = np.empty(len(idx))
        i_in = ei <= 0
        if i_in.any():
            p[i_in] = P_graph.evaluate(X[idx[i_in]])
        o = ~i_in
        if o.any():
            x = X[idx[o]]
            lat = np.min([P_graph.evaluate(x, name=n_) for n_ in lattices], axis=0)
            p[o] = lat
            if B_graph is not None:
                # over the body only where the lattice is open (or nearly):
                # elsewhere the strut already makes it material
                w = np.flatnonzero(lat > -h)
                q = x[w]
                live = np.arange(len(q))
                for _ in range(6):
                    v, gr = vgE(q[live])
                    q[live] -= ((v + eta) / np.maximum((gr * gr).sum(1), 1e-12))[:, None] * gr
                    live = live[np.abs(v + eta) > 1e-3 * h]
                    if not len(live):
                        break
                p[np.flatnonzero(o)[w]] = np.minimum(B_graph.evaluate(q), lat[w])
        out[idx] = _smin(p, cap - ei, k)
        return out

    return Family("void", f, float(h)), cap


def _shell_solid(sh, reverse=False):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.TopoDS import TopoDS

    s = TopoDS.Shell(sh.Reversed()) if reverse else sh
    return BRepBuilderAPI_MakeSolid(s).Solid()


BIG_SHELL = 8000      # faces: past this a whole-solid check is checked shell by shell


def _faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    ex = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def _valid(shape, geom=True):
    """BRepCheck validity. A solid with an inner shell of more than
    BIG_SHELL faces is checked shell by shell (each as a solid of its own):
    the whole-solid check classifies every shell against every other and
    went past 12 GB on an arm link's 38k-patch void."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shells = []
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if len(shells) < 2 or max(_faces(x) for x in shells) <= BIG_SHELL:
        return bool(BRepCheck_Analyzer(shape, geom).IsValid())
    return all(BRepCheck_Analyzer(_shell_solid(x), geom).IsValid() for x in shells)


def _bop(kind, args, tools, fuzzy=None):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.OCP.collections import List_TopoDS_Shape

    op = {"cut": BRepAlgoAPI_Cut, "common": BRepAlgoAPI_Common, "fuse": BRepAlgoAPI_Fuse}[kind]()
    a, t = List_TopoDS_Shape(), List_TopoDS_Shape()
    for x in args:
        a.Append(x)
    for x in tools:
        t.Append(x)
    op.SetArguments(a)
    op.SetTools(t)
    op.SetRunParallel(True)
    op.SetUseOBB(True)
    if fuzzy:
        op.SetFuzzyValue(fuzzy)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"OpenCascade {kind} failed")
    return op.Shape()


def _mesh_point(face):
    """(u, v) of a point well inside a face: the centroid of the biggest
    triangle of its own mesh (where PointInFace finds none: a sliver), or
    None."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location

    BRepMesh_IncrementalMesh(face, 0.01, False, 0.2, False)
    tri = BRep_Tool.Triangulation_s(face, TopLoc_Location())
    if tri is None or tri.NbTriangles() == 0 or not tri.HasUVNodes():
        return None
    best, out = -1.0, None
    for i in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(i).Get()
        pa, pb, pc = (tri.UVNode(k) for k in (a, b, c))
        area = abs((pb.X() - pa.X()) * (pc.Y() - pa.Y()) - (pc.X() - pa.X()) * (pb.Y() - pa.Y()))
        if area > best:
            best, out = area, ((pa.X() + pb.X() + pc.X()) / 3, (pa.Y() + pb.Y() + pc.Y()) / 3)
    return out


def _volume_of(shell):
    from .step import _volume

    return _volume(_shell_solid(shell))


def _solid_of(shell):
    """A shell as a solid facing out (whatever way it was sewn)."""
    from .step import _volume

    so = _shell_solid(shell)
    return so if _volume(so) > 0 else _shell_solid(shell, reverse=True)


def _inside_any(solids, P, tol=1e-6):
    """(n,) bool: P inside any of the solids (OpenCascade's classifier)."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN

    out = np.zeros(len(P), bool)
    for so in solids:
        cl = BRepClass3d_SolidClassifier(so)
        for i, p in enumerate(P):
            if not out[i]:
                cl.Perform(gp_Pnt(*map(float, p)), tol)
                out[i] = cl.State() == TopAbs_IN
    return out


def _faces_list(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    out, ex = [], TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face(ex.Current()))
        ex.Next()
    return out


def _flap_faces(sew, max_area):
    """Faces of a sewing with a free edge, under max_area (mm^2)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as Map,
    )
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    m = Map()
    TopExp.MapShapesAndAncestors_s(sew.SewedShape(), TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, sew.NbFreeEdges() + 1):
        e = sew.FreeEdge(i)
        if not m.Contains(e):
            continue
        for f in m.FindFromKey(e):
            f = TopoDS.Face(f)
            pr = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, pr)
            if pr.Mass() < max_area and not any(f.IsSame(x) for x in out):
                out.append(f)
    return out


def _split_cut(outer, voids, bore_solids, bores, env, fam, h):
    """The envelope (its shells `outer`) less the open void (`voids`: the
    void family's shells that reach out through it, as meshed: faces
    pointing away from the material) and the drain bores, without
    OpenCascade's cut: its general fuse splits every face where the shells
    cross (seconds), and each piece is kept or dropped from the fields
    instead of by classifying it against the solids (the cut crawled for
    many minutes on a lattice block's 9k patches doing that). A piece of
    the envelope stays where the part is solid just inside it and out of
    the bores; a piece of the void stays inside the envelope and out of
    the bores; a piece of a bore stays inside the envelope where the part
    round it is solid. The pieces share their edges, so they sew exactly.
    Returns (shells, report)."""
    from OCP.BOPTools import BOPTools_AlgoTools3D
    from OCP.BRepAlgoAPI import BRepAlgoAPI_BuilderAlgo
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
    from OCP.IntTools import IntTools_Context
    from OCP.OCP.collections import List_TopoDS_Shape
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    def faces_of(shape):
        out, ex = [], TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            out.append(TopoDS.Face(ex.Current()))
            ex.Next()
        return out

    t0 = time.time()
    groups = {"envelope": [], "void": [], "bore": []}
    args = List_TopoDS_Shape()
    for kind, shapes in (("envelope", outer), ("void", voids), ("bore", bore_solids)):
        for x in shapes:
            args.Append(x)
            groups[kind] += faces_of(x)
    gf = BRepAlgoAPI_BuilderAlgo()
    gf.SetArguments(args)
    gf.SetRunParallel(True)
    gf.Build()
    if not gf.IsDone():
        raise RuntimeError("OpenCascade could not split the envelope by the void")
    t_split = time.time() - t0
    ctx = IntTools_Context()
    pieces, pts, nrm, kinds = [], [], [], []
    nopoint = 0
    for kind, fs in groups.items():
        for f in fs:
            mod = gf.Modified(f)
            got = [TopoDS.Face(x) for x in mod] if mod.Size() else ([] if gf.IsDeleted(f) else [f])
            for p in got:
                P, uv = gp_Pnt(), gp_Pnt2d()
                if BOPTools_AlgoTools3D.PointInFace_s(p, P, uv, ctx) != 0:
                    got_uv = _mesh_point(p)       # its own triangulation's biggest triangle
                    if got_uv is None:
                        nopoint += 1
                        continue
                    uv = gp_Pnt2d(*got_uv)
                n, q = gp_Vec(), gp_Pnt()
                BRepGProp_Face(p).Normal(uv.X(), uv.Y(), q, n)
                if p.Orientation() == TopAbs_REVERSED:
                    n.Reverse()
                L = n.Magnitude()
                if L < 1e-12:
                    nopoint += 1
                    continue
                pieces.append(p)
                pts.append((q.X(), q.Y(), q.Z()))
                nrm.append((n.X() / L, n.Y() / L, n.Z() / L))
                kinds.append(kind)
    pts, nrm, kinds = np.asarray(pts, float).reshape(-1, 3), np.asarray(nrm, float).reshape(-1, 3), np.asarray(kinds)

    def in_bore(X):
        out = np.zeros(len(X), bool)
        for p0, p1, r in bores:
            ax = (p1 - p0) / np.linalg.norm(p1 - p0)
            t = (X - p0) @ ax
            d = np.linalg.norm(X - p0 - np.outer(t, ax), axis=1)
            out |= (t > 0) & (t < np.linalg.norm(p1 - p0)) & (d < r)
        return out

    eps = 0.25 * h
    near = 0.1 * h          # within this of the other surface, the B-rep decides, not the field
    keep = np.zeros(len(pieces), bool)
    flip = np.zeros(len(pieces), bool)
    e = np.flatnonzero(kinds == "envelope")
    if len(e):
        # solid just inside: the field says, unless the piece is a sliver
        # by a strut that grazes the face (the patches and the field part
        # by their deviation there): then the void solids themselves
        X = pts[e] - 1e-3 * h * nrm[e]
        fv = fam.f(X)
        keep[e] = (fv < 0) & ~in_bore(X)
        unsure = e[np.abs(fv) < near]
        if len(unsure):
            inside = _inside_any([_shell_solid(x, reverse=True) for x in voids if _volume_of(x) < 0],
                                 pts[unsure] - 1e-4 * nrm[unsure])
            keep[unsure] = ~inside & ~in_bore(pts[unsure])
    v = np.flatnonzero(kinds == "void")
    if len(v):
        ev = env.evaluate(pts[v])
        keep[v] = (ev < 0) & ~in_bore(pts[v])
        unsure = v[np.abs(ev) < near]
        if len(unsure):                     # by the envelope's own faces (patches off its field)
            keep[unsure] = _inside_any([_solid_of(x) for x in outer], pts[unsure]) & ~in_bore(pts[unsure])
    b = kinds == "bore"
    if b.any():                       # a bore's faces point out of its cylinder, into the part
        keep[b] = (env.evaluate(pts[b]) < 0) & (fam.f(pts[b] + eps * nrm[b]) < 0)
        flip[b] = True
    kept = [p.Reversed() if fl else p for p, k, fl in zip(pieces, keep, flip) if k]
    for tol in (1e-4, 2e-3 * h, 1e-2 * h):
        # a sliver kept that should not be (a flap: its free edges are its
        # own) is dropped if it is small; a sliver the classifiers dropped
        # that should be kept (a strut grazing a face makes pieces microns
        # across) leaves a hole that wide, so sewing a little looser closes
        # it. Flaps first: a loose tolerance can fold a face's wire on itself
        sew = BRepBuilderAPI_Sewing(tol)
        for f in kept:
            sew.Add(f)
        sew.Perform()
        flaps = 0
        for _ in range(3):
            if not sew.NbFreeEdges():
                break
            cut_faces = _flap_faces(sew, 0.25 * h * h)
            if not cut_faces:
                break
            flaps += len(cut_faces)
            keep_faces = [f for f in _faces_list(sew.SewedShape()) if not any(f.IsSame(c) for c in cut_faces)]
            sew = BRepBuilderAPI_Sewing(tol)
            for f in keep_faces:
                sew.Add(f)
            sew.Perform()
        if not sew.NbFreeEdges():
            break
    shells = []
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    return shells, {"pieces": len(pieces), "kept": int(keep.sum()), "noInteriorPoint": nopoint,
                    "freeEdges": int(sew.NbFreeEdges()), "sewTolerance": tol, "flapsDropped": flaps,
                    "splitSeconds": round(t_split, 1), "seconds": round(time.time() - t0, 1)}


def _bore_tools(bores):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    out = []
    for p0, p1, r in bores:
        ax = p1 - p0
        L = float(np.linalg.norm(ax))
        out.append(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, ax / L))),
                                            r, L).Shape())
    return out


def _void_shells(fam, lo, hi, E_graph):
    """Closed shells of the void family's patches, each with its signed
    volume and whether it reaches outside the envelope."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCP.Geom import Geom_BezierSurface
    from OCP.OCP.collections import Array2_gp_Pnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from .ibrep import DEG
    from .step import _pnt, _volume

    t0 = time.time()
    pieces, sheets, nonman, refined = _pieces(fam, np.asarray(lo, float), np.asarray(hi, float))
    n = DEG + 1
    out, open_, dev = [], 0, []
    for p in pieces:
        sh, qs = p["sheet"], p["quads"]
        if not p["closed"]:
            open_ += 1
            continue
        cen = sh.V[sh.Q[qs]].mean(axis=1)
        dev.append(sh.deviation_of(qs[E_graph.evaluate(cen) <= 0]))   # outside E is cut away
        sew = BRepBuilderAPI_Sewing(1e-4)
        for q in qs:
            arr = Array2_gp_Pnt(1, n, 1, n)
            for a in range(n):
                for b in range(n):
                    arr.SetValue(a + 1, b + 1, _pnt(sh.C[q, a, b]))
            sew.Add(BRepBuilderAPI_MakeFace(Geom_BezierSurface(arr), 1e-7).Face())
        sew.Perform()
        if sew.NbFreeEdges():
            open_ += 1
            continue
        V = sh.V[np.unique(sh.Q[qs])]
        crosses = bool((E_graph.evaluate(V) > 0).any())
        # the quads' own winding says which side is void (sewing orients a
        # shell after whichever face it starts from, and a few patches on a
        # rough field can face the wrong way): their signed volume decides
        T = np.concatenate([sh.Q[qs][:, [0, 1, 2]], sh.Q[qs][:, [0, 2, 3]]])
        X = sh.V
        qvol = float(np.einsum("ij,ij->i", X[T[:, 0]], np.cross(X[T[:, 1]], X[T[:, 2]])).sum() / 6.0)
        ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
        while ex.More():
            s = TopoDS.Shell(ex.Current())
            v = _volume(_shell_solid(s))
            if v * qvol < 0:
                s = TopoDS.Shell(s.Reversed())
                v = -v
            if abs(v) >= 0.01:                  # a speck where the surface grazes a grid node
                out.append({"shell": s, "volume": v, "crosses": crosses})
            ex.Next()
    dev = np.concatenate(dev) if dev else np.zeros(1)
    rep = {"h": fam.h, "patches": int(sum(len(p["quads"]) for p in pieces)),
           "components": len(pieces), "openComponents": open_, "refinedBoxes": refined,
           "foldedPatches": int(sum(p["folds"] for p in pieces)), "nonManifoldEdges": int(nonman),
           "chordEdges": int(sum(s.wild_edges for s in sheets)), "coonsPatches": int(sum(s.wild_patches for s in sheets)),
           "patchDeviation": {"p99": round(float(np.percentile(dev, 99)), 4), "max": round(float(dev.max()), 4)},
           "seconds": round(time.time() - t0, 1)}
    return out, rep


def _log(progress, msg):
    if progress:
        progress({"stage": "lattice-step", "message": msg})
    if os.environ.get("LOADPATH_VERBOSE"):
        print(f"[latticestep] {msg}", flush=True)


VOLUME_SLACK = 0.015   # solid this much heavier than the STL: void lost (pockets under the grid)


def _grid_shift(lo, hi, h):
    """Where the patch grid starts (a shift under one patch, per axis): no
    grid plane within a quarter patch of the part's bounding planes, where
    its flat faces usually are. A row of patch corners a few microns off a
    face leaves sliver pieces when the void is cut there (and OpenCascade's
    own cut crawled on them)."""
    out = np.zeros(3)
    for a in range(3):
        best = None
        for t in np.linspace(0.05, 0.95, 37):
            # grid planes sit at lo - t h + k h: their offsets from lo and hi
            d0 = t % 1.0
            d1 = ((hi[a] - lo[a]) / h + t) % 1.0
            m = min(d0, 1 - d0, d1, 1 - d1)
            if best is None or m > best[0] + 1e-9:
                best = (m, t)
        out[a] = best[1] * h
    return out


def export_void_step(doc, g, out_dir, name, base_dir=None, sources=None, meshes=None, h=None,
                     surface="smooth", part_mesh=None, progress=None, faces=None, patches=None):
    """<name>.step for a part with lattices of any kind (strut cells, Voronoi
    foams, TPMS sheets or networks; graded, conformal; under a skin or open
    at the outside): the envelope (the part with its lattices solid; exact
    faces, smooth patches) less the void (void_family: one smooth level set
    of the part's own field, node blends and fillets included, as bicubic
    patches on the true surface). Void that is sealed inside the part goes
    in as inner shells; void that reaches the outside (an open lattice) is
    cut away in one OpenCascade boolean, along with any drain bores.
    Returns (report, shape)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Compound

    from .step import (
        DEFAULT_FACES,
        DEFAULT_PATCHES,
        _volume,
        export_step,
        read_step,
        write_step,
    )

    t0 = time.time()
    lattices = [k for k, nd in g.nodes.items() if nd.op == "lattice"]
    if not lattices:
        raise ValueError("void STEP takes a part with lattice nodes")
    bores, extra = _drain_bores(g)
    env = _lattice_variant_graph_(doc, g, lattices, (), base_dir, sources, meshes, extra)
    part = _lattice_variant_graph_(doc, g, (), (), base_dir, sources, meshes, extra)
    body = _lattice_variant_graph_(doc, g, (), lattices, base_dir, sources, meshes, extra)
    em = env.mesh(progress=progress)
    st = export_step(em, os.path.join(out_dir, f"{name}-envelope.step"), faces=int(faces or DEFAULT_FACES),
                     name=name, surface=surface, patches=int(patches or DEFAULT_PATCHES))
    outer = st.pop("shape")
    lo, hi = np.asarray(em.bounds[0], float), np.asarray(em.bounds[1], float)
    feat, cell = _feature(g, lattices, em.sample(4000) if len(em.faces) else lo[None])
    h_given = h is not None
    if h is None:
        h = min(cell / 6.0, feat / 2.0)
    h = float(h)
    S = np.random.default_rng(0).uniform(lo, hi, (2000, 3))
    bv = body.evaluate(S)
    if (bv >= 100).all():
        body = None                                  # lattices alone: nothing to keep material outside
    fam, cap = void_family(part, env, body, lattices, h)
    # the grid off the part's own round numbers: a row of patch corners on a
    # face of the envelope (a box face through the cell nodes) makes the cut
    # meet patch edges lying in its plane, which OpenCascade crawls through
    ex = TopExp_Explorer(outer, TopAbs_SHELL)
    outs = []
    while ex.More():
        outs.append(TopoDS.Shell(ex.Current()))
        ex.Next()

    def envelope_with(voids):
        mk = BRepBuilderAPI_MakeSolid(outs[0])
        for s in outs[1:]:
            mk.Add(s)
        for s in voids:
            mk.Add(s)
        return mk.Solid()

    def assemble(shells):
        sealed = [x for x in shells if not x["crosses"] and x["volume"] < 0]
        loose = [x for x in shells if not x["crosses"] and x["volume"] > 0]
        pores = [x for x in shells if x["crosses"] and x["volume"] < 0]
        plugs = [x for x in shells if x["crosses"] and x["volume"] > 0]
        solid = envelope_with([x["shell"] for x in sealed])
        _log(progress, f"envelope with {len(sealed)} sealed voids; checking")
        filled = []
        if sealed and not _valid(solid, False):
            keep = []
            for x in sealed:
                (keep if _valid(envelope_with([x["shell"]]), False) else filled).append(x)
            solid = envelope_with([x["shell"] for x in keep])
            filled = [round(-x["volume"], 2) for x in filled]
            sealed = keep
        # open pores (void reaching out through the envelope) and any plugs (a
        # strut stub the envelope cut off, wrapped by a pore): the envelope split
        # by them and the pieces picked by the fields (_split_cut)
        split = None
        if pores or plugs:
            t1 = time.time()
            pieces, split = _split_cut(outs, [x["shell"] for x in pores + plugs], _bore_tools(bores), bores, env,
                                       fam, h)
            bodies = []
            for x in pieces:
                so = _shell_solid(x)
                v = _volume(so)
                if v < 0:
                    x = TopoDS.Shell(x.Reversed())
                bodies.append((abs(v), x))
            bodies.sort(key=lambda t: -t[0])
            mk = BRepBuilderAPI_MakeSolid(bodies[0][1])
            for x in sealed:
                mk.Add(x["shell"])
            solid = mk.Solid()
            if len(bodies) > 1:
                bb = BRep_Builder()
                comp = TopoDS_Compound()
                bb.MakeCompound(comp)
                bb.Add(comp, solid)
                for _, x in bodies[1:]:
                    bb.Add(comp, _shell_solid(x))
                solid = comp
            split["bodies"] = len(bodies)
            _log(progress, f"split {split['pieces']} face pieces, kept {split['kept']}, {len(bodies)} bodies: "
                           f"{time.time() - t1:.0f} s")
        elif bores:
            solid = _bop("cut", [solid], _bore_tools(bores))
        if loose:
            bb = BRep_Builder()
            comp = TopoDS_Compound()
            bb.MakeCompound(comp)
            for x in [solid] + [_shell_solid(x["shell"]) for x in loose]:
                bb.Add(comp, x)
            solid = comp
        _log(progress, "checking the solid")
        valid = _valid(solid)
        return solid, sealed, filled, split, pores, plugs, loose, valid

    tries = []
    for attempt in range(3):
        _log(progress, f"void patches at h {h:.3f} mm")
        j = _grid_shift(lo, hi, h)
        shells, vrep = _void_shells(fam, lo - cap - j, hi + cap, env)
        _log(progress, f"{vrep['patches']} patches, {len(shells)} shells, {vrep['openComponents']} open, "
                       f"{vrep['seconds']} s")
        last = h_given or attempt == 2
        if vrep["openComponents"] and not last:
            # a void surface that does not close (the grid could not tell two
            # sheets apart somewhere: a gap or a fold under the patch size) is
            # meshed again finer
            tries.append({"grid": round(h, 4), "openComponents": vrep["openComponents"]})
            h *= 0.7
            fam, cap = void_family(part, env, body, lattices, h)
            continue
        # (a face the split leaves invalid, two void sheets crossing the
        # envelope's face microns apart so its wire folds, is not retried: on
        # the open Voronoi block two finer grids failed the same way)
        solid, sealed, filled, split, pores, plugs, loose, valid = assemble(shells)
        # (nor are pockets thinner than the patch size, a wedge where a strut
        # meets a skin at a shallow angle, that fall between grid nodes: the
        # solid reads heavier than the STL and the report says so. Meshing
        # them finer made OpenCascade's solid check crash on the BCC test)
        break
    vrep["retries"] = tries
    _log(progress, f"valid {valid}; writing")
    rep = {"export": "envelope less void", "lattices": lattices, "grid": round(h, 4), "cap": round(cap, 4),
           "envelope": st, "void": vrep, "split": split, "sealedCavities": len(sealed), "openPores": len(pores),
           "plugs": len(plugs), "looseBodies": len(loose), "filledCavities": filled,
           "drainBores": len(bores), "valid": valid, "volume": round(abs(_volume(solid)), 2),
           "envelopeVolume": round(abs(_volume(outer)), 2)}
    if part_mesh is not None:
        rep["meshVolume"] = round(float(abs(part_mesh.volume)), 2)
    rep.update(write_step(solid, os.path.join(out_dir, f"{name}.step"), name=name))
    t1 = time.time()
    back = read_step(rep["path"])
    rep["readBack"] = {"seconds": round(time.time() - t1, 1), "volume": round(abs(_volume(back)), 2),
                       "valid": _valid(back)}
    rep["readBack"]["ok"] = rep["readBack"]["valid"] and abs(rep["readBack"]["volume"] - rep["volume"]) < 1e-3 * rep["volume"]
    _log(progress, "read back; distance to the STL")
    if part_mesh is not None:
        rep["stlDistance"] = stl_distance(solid, part_mesh)
    # complete: every void went in (none left open, filled or uncuttable)
    # and the solid no heavier than the STL by more than its facets explain
    heavy = part_mesh is not None and rep["volume"] > (1 + VOLUME_SLACK) * rep["meshVolume"]
    rep["complete"] = bool(not vrep["openComponents"] and not filled
                           and (split is None or not split["freeEdges"]) and not heavy)
    if not rep["complete"]:
        rep["warning"] = (f"the STEP is missing lattice void: {vrep['openComponents']} void surface(s) did not "
                          f"close, {len(filled)} sealed cavit(ies) left solid, "
                          f"{(split or {}).get('freeEdges', 0)} free edges where the open void was cut"
                          + (f", {100 * (rep['volume'] / rep['meshVolume'] - 1):.1f}% heavier than the STL "
                             "(pockets thinner than the grid)" if heavy else "")
                          + "; try a finer latticeGrid")
    if not valid:
        rep["warning"] = (rep.get("warning", "") + "; " if rep.get("warning") else "") + (
            "OpenCascade's check rejects the solid (an envelope face where two lattice surfaces cross it "
            "microns apart, typically); a CAD tool may need to heal it")
    rep["ok"] = bool(valid and rep["readBack"]["ok"] and rep["complete"])
    rep["seconds"] = round(time.time() - t0, 1)
    return rep, solid


def stl_distance(shape, mesh, n=40000, deflection=0.01):
    """How far a B-rep and the printed mesh are apart (mm), both ways:
    points on the B-rep's faces to the mesh, points on the mesh to the
    B-rep's tessellation (`deflection` mm). Mean, p99, max of each."""
    from .sharp import mesh_distance
    from .step import shape_to_trimesh

    t = shape_to_trimesh(shape, deflection=deflection, angle=0.1)
    a = mesh_distance(mesh, t.sample(n), cap=2.0)
    b = mesh_distance(t, mesh.sample(n), cap=2.0)

    def st(d):
        return {"mean": round(float(d.mean()), 4), "p99": round(float(np.percentile(d, 99)), 4),
                "max": round(float(d.max()), 4)}
    return {"stepToStl": st(a), "stlToStep": st(b), "tessellationDeflection": deflection}


def _lattice_variant_graph_(doc, g, solid, empty, base_dir, sources, meshes, extra):
    from .fields import _lattice_variant_graph

    return _lattice_variant_graph(doc, g, solid, empty, base_dir, sources, meshes, extra=extra)


def export_lattice_step(doc, g, out_dir, name, base_dir=None, sources=None, meshes=None, h=None,
                        surface="smooth", part_mesh=None, progress=None, faces=None, patches=None):
    """step lattice "exact": the cavity route (export_cavity_step: each side
    of the walls its own smooth level set, patches a sixth of a cell) for one
    TPMS sheet lattice sealed under a skin, since its patches need not
    resolve the wall; the void route (export_void_step: patches half the
    thinnest strut or wall) for everything else: strut cells, foams,
    networks, several lattices, lattices open at the outside."""
    from .struts import is_strut

    lattices = [k for k, nd in g.nodes.items() if nd.op == "lattice"]
    kw = {"base_dir": base_dir, "sources": sources, "meshes": meshes, "h": h, "surface": surface,
          "part_mesh": part_mesh, "progress": progress, "faces": faces, "patches": patches}
    route = "void"
    if len(lattices) == 1:
        s = g.nodes[lattices[0]].spec
        if not is_strut(s.get("type", "gyroid")) and s.get("form", "sheet") == "sheet":
            _, extra = _drain_bores(g)
            env = _lattice_variant_graph_(doc, g, lattices, (), base_dir, sources, meshes, extra)
            part = _lattice_variant_graph_(doc, g, (), (), base_dir, sources, meshes, extra)
            P = env.mesh(progress=progress).sample(20000)
            # sealed: the part's own surface is the envelope's everywhere
            # (no lattice opening onto it)
            if (part.evaluate(P) < 0.05).mean() > 0.999:
                route = "cavities"
    if route == "cavities":
        rep, shape = export_cavity_step(doc, g, out_dir, name, **kw)
        if part_mesh is not None:
            rep["stlDistance"] = stl_distance(shape, part_mesh)
        rep["complete"] = bool(not any(c["openCavities"] for c in rep["cavities"]) and not rep["filledCavities"])
        rep["ok"] = bool(rep["valid"] and rep["readBack"]["ok"] and rep["complete"])
    else:
        rep, shape = export_void_step(doc, g, out_dir, name, **kw)
    rep["route"] = route
    return rep, shape
