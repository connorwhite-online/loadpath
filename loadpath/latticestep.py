"""
A skinned TPMS lattice part as smooth, exact CAD (STEP), lattice and all.

A sheet lattice under a skin splits the part's inside into cavities, the
channels on each side of the lattice's walls. So

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
            if nm or not new:
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
