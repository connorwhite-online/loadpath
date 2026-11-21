"""
The checks every build gets, and the features a spec asserts.

A part that looks right can still be wrong in ways nobody sees until it is
printed: a port mouth that went square, a film of material into a board's
edge, a wall under the printer's minimum, a form a little off its
reference. Each of those is a measurement, so each is a check, run on the
delivered mesh on every build (build.py) or on any mesh against its spec
(`check_build`, `loadpath check-build`):

    closed          the mesh is a closed, valid solid
    one-piece       one body (specks dropped are reported)
    exterior        the outside is the exterior's own surface, outside the
                    declared cuts (six-view depth comparison, surfacecheck)
    open:<id>       a cut interface is empty: nothing of the part in it
    present:<id>    an added interface is all there
    clear:<id>      a component: nothing in its solid, its clearance kept
    way-in:<id>     its insertion path is free
    inside:<id>     it sits inside the exterior by its keepIn
    process         what the printing process cannot make (processcheck)
    dfm             the part against the print service's published limits
                    (dfm_reference: Materialize's minimum wall and build
                    volume for the process and material)
    mass            the assembly against targetMass
    assert ...      the spec's `assert` list
    reference ...   the spec's `references` (reference.py)

Asserted features (`assert` in the spec):

    {"type": "fillet", "interface": "port", "near": <shape>, "min": 1.0}
        every curved surface of the interface inside `near` has a radius
        of curvature of at least `min` (read from the B-rep kernel), that
        surface is on the delivered part, the delivered mesh's own bends
        there read at least that radius, and no edge of the delivered part
        inside `near` is sharper than `maxEdge` degrees (default 50). A
        mouth that went square fails all three.
    {"type": "wall", "min": 1.0, "near"?: <shape>}
    {"type": "clearance", "component": "board", "min": 0.15}
    {"type": "keepIn", "component": "board", "margin": 0.8}
    {"type": "surface", "interface": id, "near": shape, "tolerance"?: mm}
    {"type": "mass", "min"?: g, "max"?: g, "of"?: "assembly" | "part"}
    {"type": "open", "interface": id}, {"type": "present", "interface": id}
    {"type": "distance", "from": id, "to": id, "min"?: mm, "max"?: mm}

A check is {"id", "kind", "ok" (True, False, or None when it could not
run), "value", "limit", "detail"}. The build passes when no check is False.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["check_build", "ASSERT_TYPES"]

ASSERT_TYPES = ("surface", "fillet", "wall", "clearance", "keepIn", "mass", "open", "present", "distance")
ON = 0.03               # mm: a drawn point this close to the mesh is on it
INTRUDE = 0.05          # mm a part may reach into a component (mesh tolerance), as verify.body_intrusion


def _row(i, kind, ok, value=None, limit=None, detail=None, **more):
    r = {"id": i, "kind": kind, "ok": ok}
    if value is not None:
        r["value"] = value
    if limit is not None:
        r["limit"] = limit
    if detail:
        r["detail"] = detail
    r.update(more)
    return r


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _distance(mesh, P):
    """Unsigned distance from points to a mesh's surface."""
    P = np.asarray(P, float).reshape(-1, 3)
    if not len(P):
        return np.zeros(0)
    try:
        import igl

        V = np.asarray(mesh.vertices, float)
        F = np.asarray(mesh.faces, np.int64)
        return np.sqrt(np.maximum(igl.point_mesh_squared_distance(P, V, F)[0], 0.0))
    except ImportError:
        from trimesh.proximity import closest_point

        out = np.empty(len(P))
        for s in range(0, len(P), 5000):
            out[s:s + 5000] = closest_point(mesh, P[s:s + 5000])[1]
        return out


def _inside(mesh, P):
    P = np.asarray(P, float).reshape(-1, 3)
    if not len(P):
        return np.zeros(0, bool)
    try:
        import igl

        return igl.fast_winding_number(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), P) > 0.5
    except ImportError:
        return np.concatenate([mesh.contains(P[s:s + 2000]) for s in range(0, len(P), 2000)])


def _signed(mesh, P):
    """Signed distance to a closed mesh, exact (negative inside): a mesh
    shape's own sdf is read on a grid, a tenth of a millimetre coarse."""
    d = _distance(mesh, P)
    return np.where(_inside(mesh, P), -d, d)


def _vol(man):
    return 0.0 if man is None or man.is_empty() else float(man.volume())


def _near(a, base_dir):
    if a.get("near") is None:
        return None
    from .shapes import shape_from_spec

    return shape_from_spec(a["near"], base_dir)


def _no_slivers(mesh):
    """The mesh without the needle triangles exact booleans leave: their
    normals are noise, and an edge beside one reads any angle."""
    import trimesh

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    T = V[F]
    e = np.stack([np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 1], axis=1),
                  np.linalg.norm(T[:, 0] - T[:, 2], axis=1)], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    height = 2.0 * area / np.maximum(e.max(axis=1), 1e-12)
    good = (height > 0.004) & (height > 0.02 * e.max(axis=1))
    return trimesh.Trimesh(V, F[good], process=False)


# ---- asserted features ---------------------------------------------------------------

def fillet_check(spec, mesh, a, solid=None) -> dict:
    """An interface's rounds, on the kernel and on the delivered mesh."""
    from .brepio import mesh_edge_radii

    it = spec.interface(a["interface"])
    brep = it.brep
    near = _near(a, spec.base_dir)
    want = float(a["min"])
    max_edge = float(a.get("maxEdge", 50.0))
    name = a.get("id") or f"fillet:{it.id}"

    def in_region(P):
        return np.ones(len(P), bool) if near is None else near.sdf(P) <= 0.0

    # 1. the kernel: curved faces with surface in the region
    P, N, own = brep.samples(spacing=0.12)
    reg = in_region(P)
    kernel, faces_in = None, []
    rows = {f["index"]: f for f in brep.faces()}
    if brep.exact:
        for fi in np.unique(own[reg]):
            f = rows[int(fi)]
            if f["minRadius"] is not None:
                faces_in.append(int(fi))
                kernel = f["minRadius"] if kernel is None else min(kernel, f["minRadius"])
    # 2. the drawn surface there that has the part behind it is on the part
    sel = reg if not faces_in else reg & np.isin(own, faces_in)
    Ps, Ns = P[sel], N[sel]
    side = -1.0 if it.op == "add" else 1.0      # which way the part's material lies from the solid's surface
    probe = Ps + side * 0.08 * Ns
    behind = _inside(spec._solid_mesh, probe) if getattr(spec, "_solid_mesh", None) is not None else _inside(mesh, probe)
    d = _distance(mesh, Ps)
    # only where the drawn surface is inside the exterior: a tool runs on past it
    expect = (behind | (d <= ON)) & (_signed(spec.exterior_mesh(), Ps) <= 0.02)
    kept = float((d[expect] <= ON).mean()) if expect.any() else 0.0
    # 3. the delivered mesh's own reading: bends and sharp edges in the region
    # 3. the delivered mesh's own reading there: bends and sharp edges of
    # the feature itself, which are the edges between two faces that each
    # lie on the interface or on the exterior (a mouth's rim is where the
    # two meet; where a tunnel leaves the wall into the cavity is not it)
    clean = _no_slivers(mesh)
    adj = np.asarray(clean.face_adjacency)
    sharpest, measured = 0.0, None
    if len(adj):
        ang = np.degrees(np.asarray(clean.face_adjacency_angles))
        e = np.asarray(clean.face_adjacency_edges)
        V = np.asarray(clean.vertices)
        mid = V[e].mean(axis=1)
        cand = np.flatnonzero(in_region(mid) & (ang > 2.0))
        if len(cand):
            cand = cand[_distance(brep.mesh, mid[cand]) <= ON]
        if len(cand):
            cen = np.asarray(clean.triangles_center)
            fa, fb = adj[cand, 0], adj[cand, 1]

            def drawn(c):
                return (_distance(brep.mesh, c) <= ON), (_distance(spec.exterior_mesh(), c) <= ON)
            ta, xa = drawn(cen[fa])
            tb, xb = drawn(cen[fb])
            feature = (ta | xa) & (tb | xb) & (ta | tb)
            cand = cand[feature]
            both = (ta & tb)[feature]
        if len(cand):
            sharpest = float(ang[cand].max())
            length = np.linalg.norm(V[e[cand, 0]] - V[e[cand, 1]], axis=1)
            area = np.asarray(clean.area_faces)
            rad = (area[adj[cand, 0]] + area[adj[cand, 1]]) / np.maximum(length, 1e-12) / np.radians(ang[cand])
            bends = both & (ang[cand] <= max_edge) & (length > 0.02)
            if bends.any():
                order = np.argsort(rad[bends])
                w = np.cumsum(length[bends][order])
                measured = float(rad[bends][order][np.searchsorted(w, 0.1 * w[-1])])
    problems = []
    if not sel.any():
        problems.append("the interface has no surface inside `near`")
    if kernel is not None and kernel < want - 1e-6:
        problems.append(f"the drawn solid's tightest round there is {kernel:.3f} mm")
    if brep.exact and kernel is None and sel.any():
        problems.append("the drawn solid has no rounded surface there (flat faces only)")
    if kept < 0.9:
        problems.append(f"only {100 * kept:.0f}% of the drawn surface there is on the part")
    if measured is not None and measured < 0.8 * want:
        problems.append(f"the delivered mesh bends at {measured:.2f} mm there")
    if measured is None and sel.any():
        problems.append("the delivered part has no rounded surface there")
    if sharpest > max_edge:
        problems.append(f"the delivered part has a {sharpest:.0f} degree edge there")
    return _row(name, "fillet", not problems, value=_r(kernel if kernel is not None else measured), limit=want,
                detail="; ".join(problems) or None, interface=it.id, kernelRadius=_r(kernel),
                measuredRadius=_r(measured), keptShare=_r(kept), sharpestEdgeDeg=_r(sharpest, 1),
                maxEdgeDeg=max_edge)


def wall_check(spec, mesh, a) -> dict:
    """No wall under `min` that the process would lose (processcheck's
    reading: what a ball of that diameter cannot sweep, past an edge's
    rounding). Inside `near` only, when given."""
    from .processcheck import _choose_pitch, _grid, _walls

    want = float(a["min"])
    name = a.get("id") or f"wall>={want:g}"
    near = _near(a, spec.base_dir)
    p = _choose_pitch(np.asarray(mesh.extents, float), want)
    solid, origin = _grid(mesh, np.eye(3), None, p, pad=3)
    w = _walls(solid, p, origin, want)
    if w["ok"] is None:
        return _row(name, "wall", None, limit=want, detail=w.get("inconclusive"))
    bad = w["unformed"]
    if near is not None:
        bad = [u for u in bad if near.sdf(np.asarray([u["at"]], float))[0] <= 0.0]
    return _row(name, "wall", not bad, value=_r(w["unformedWallMm"] if bad else w["judgedWallMm"]), limit=want,
                detail=(f"{len(bad)} place(s) thinner than {want:g} mm would not form, the largest "
                        f"{bad[0]['volumeMm3']:.2f} mm^3 at {bad[0]['at']}") if bad else None,
                thinnestRidgeMm=_r(w.get("minWallMm")), unformed=bad[:5])


def clearance_check(spec, mesh, comp, want, name=None, part_man=None) -> dict:
    """Nothing of the part inside the component, and at least `want` mm
    between them everywhere."""
    from .shapes import manifold_from_trimesh

    name = name or f"clear:{comp.id}"
    if part_man is None:
        part_man = manifold_from_trimesh(mesh)
    body = comp.brep.manifold()
    inter = part_man ^ body
    depth = 0.0
    if _vol(inter) > 1e-6:
        # how deep the part reaches into the component: distance from the
        # overlap to the component's own surface
        from .shapes import trimesh_from_manifold

        ov = trimesh_from_manifold(inter)
        depth = float(_distance(comp.brep.mesh, ov.vertices).max()) if len(ov.vertices) else 0.0
    P, _, _ = comp.brep.samples(spacing=0.25)
    gap = float(_distance(mesh, P).min()) if len(P) else None
    if depth > INTRUDE:
        gap = 0.0
    ok = depth <= INTRUDE and (gap is None or gap >= want - 0.02)
    detail = None
    if depth > INTRUDE:
        detail = f"the part reaches {depth:.2f} mm into {comp.id} ({_vol(inter):.2f} mm^3)"
    elif not ok:
        detail = f"the part comes within {gap:.2f} mm of {comp.id}"
    return _row(name, "clearance", ok, value=_r(gap), limit=want, detail=detail, component=comp.id,
                intrusionDepth=_r(depth), intrusionVolume=_r(_vol(inter)))


def way_in_check(spec, mesh, comp, part_man) -> dict:
    """The component's path to its place is free of the part."""
    from .shapes import Sweep

    direction, length = comp.insert
    lo, hi = spec.exterior.bounds()
    if length is None:
        # out of the exterior's box, and a little
        blo, bhi = comp.brep.bounds()
        length = float(np.max(np.where(direction > 0, hi - blo, np.where(direction < 0, bhi - lo, 0.0)))) + 2.0
    try:
        swept = Sweep(comp.brep.as_shape(), direction, float(length)).to_manifold()
    except Exception as exc:  # noqa: BLE001 - reported, not guessed
        return _row(f"way-in:{comp.id}", "wayIn", None, detail=f"the sweep could not be built: {exc}")
    hit = _vol(part_man ^ swept)
    # the clearance cut round the part itself is exact, so a free path reads zero to rounding
    ok = hit <= 0.02
    return _row(f"way-in:{comp.id}", "wayIn", ok, value=_r(hit), limit=0.02, component=comp.id,
                detail=None if ok else f"{hit:.2f} mm^3 of the part stands in {comp.id}'s way along "
                                       f"{[round(float(x), 2) for x in direction]} for {length:.1f} mm",
                length=_r(length, 2))


def keep_in_check(spec, comp, margin, name=None) -> dict:
    P, _, _ = comp.brep.samples(spacing=0.3)
    P = np.vstack([P, np.asarray(comp.brep.mesh.vertices, float)])
    worst = float(spec.exterior.sdf(P).max())
    ok = worst <= -margin + 0.02
    return _row(name or f"inside:{comp.id}", "keepIn", ok, value=_r(-worst), limit=margin, component=comp.id,
                detail=None if ok else (f"{comp.id} comes within {-worst:.2f} mm of the exterior" if worst < 0
                                        else f"{comp.id} stands {worst:.2f} mm outside the exterior"))


SOUND_PLANES = 9        # sections per axis the self-crossing survey cuts
KEPT_SHARE = 0.98       # of a cut's drawn surface that must be on the part (drawncheck.MISSING_SHARE)


def sound_check(mesh) -> dict:
    """Watertight and one body is not sound: a surface can fold through
    itself and still close (hairline slits), and an STL's float32 weld can
    leave edges on four faces. Counts the edges that are not on exactly two
    faces as the file stores them, and cuts the part on planes across each
    axis: a sound solid's section is simple closed loops that do not cross."""
    import trimesh

    m = trimesh.Trimesh(np.asarray(mesh.vertices, np.float32).astype(float), np.asarray(mesh.faces), process=False)
    m.merge_vertices()
    f = np.asarray(m.faces)
    # faces the weld collapsed onto an edge or a point are gone from the surface
    m.update_faces((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2]))
    e = m.edges_sorted
    bad_edges, edge_at = 0, None
    if len(e):
        u, c = np.unique(e, axis=0, return_counts=True)
        bad_edges = int((c != 2).sum())
        if bad_edges:
            edge_at = [round(float(v), 2) for v in m.vertices[u[c != 2][0]].mean(axis=0)]
    lo, hi = m.bounds
    open_loops = crossings = 0
    where = None
    for ax in range(3):
        normal = np.eye(3)[ax]
        # off the round numbers faces sit on
        for t in (np.arange(SOUND_PLANES) + 0.5) / SOUND_PLANES + 0.0137:
            origin = lo + (hi - lo) * min(t, 0.999)
            try:
                sec = m.section(plane_origin=origin, plane_normal=normal)
            except Exception:  # noqa: BLE001 - a plane that cannot be cut is not a finding
                continue
            if sec is None:
                continue
            try:
                flat, T = sec.to_2D() if hasattr(sec, "to_2D") else sec.to_planar()
                polys = [q for q in flat.polygons_closed if q is not None]
                n_open = len(flat.paths) - len(polys)
            except Exception:  # noqa: BLE001
                continue
            hit = None
            if n_open > 0:
                open_loops += n_open
            for q in polys:
                if not q.is_valid:
                    crossings += 1
                    hit = q.representative_point()
            rings = [q.exterior for q in polys]
            for i in range(len(rings)):
                for j in range(i + 1, len(rings)):
                    if rings[i].crosses(rings[j]):
                        crossings += 1
                        hit = rings[i].intersection(rings[j]).representative_point()
            if hit is not None and where is None:
                w = np.asarray(T) @ np.array([hit.x, hit.y, 0.0, 1.0])
                where = [round(float(v), 2) for v in w[:3]]
    # every face against every face it does not share a vertex with (exact,
    # selfx): a fold too small or too oblique for nine planes to cut
    crossing_faces, face_at = 0, None
    if len(m.faces):
        from .crease import crossing_faces as _crossing

        x = _crossing(np.asarray(m.vertices, float), np.asarray(m.faces), np.arange(len(m.faces)))
        crossing_faces = int(len(x))
        if crossing_faces:
            face_at = [round(float(v), 2) for v in m.vertices[m.faces[x[0]]].mean(axis=0)]
    ok = bad_edges == 0 and crossings == 0 and open_loops == 0 and crossing_faces == 0
    detail = None
    if not ok:
        bits = []
        if bad_edges:
            bits.append(f"{bad_edges} edges are not on exactly two faces as an STL stores the part (first at "
                        f"{edge_at}: two drawn faces that coincide there kiss along an edge; move one off the other)")
        if crossings:
            bits.append(f"the surface crosses itself in {crossings} of {3 * SOUND_PLANES} sections"
                        + (f" (first near {where})" if where else ""))
        if open_loops:
            bits.append(f"{open_loops} section loops do not close")
        if crossing_faces:
            bits.append(f"{crossing_faces} faces cross another face (first near {face_at}): the surface folds "
                        "through itself, a slit or a doubled skin the voxel analysis reads as solid")
        detail = "; ".join(bits)
    return _row("sound", "sound", ok, value=bad_edges + crossings + open_loops + crossing_faces, limit=0,
                detail=detail, nonManifoldEdges=bad_edges, selfCrossings=crossings, openLoops=open_loops,
                crossingFaces=crossing_faces, **({"at": edge_at or where or face_at} if not ok else {}))


def _taken(spec, it, P):
    """Points other cuts or a component's pocket remove: nothing of `it` is
    expected on the part there."""
    out = np.zeros(len(P), bool)
    for o in spec.interfaces:
        if o is not it and o.op == "cut":
            out |= _inside(o.brep.mesh, P) | (_distance(o.brep.mesh, P) <= ON)
    for c in spec.components:
        out |= _inside(c.brep.mesh, P) | (_distance(c.brep.mesh, P) <= c.clearance + ON)
    return out


def drawn_surface_check(spec, mesh, it, near=None, name=None, tol=None, both=False) -> dict:
    """A cut's drawn surface, by region: every part of it that has the
    part's material behind it is on the delivered surface (the share kept),
    and how far the worst of it sits off. With `both` (and a `near`
    window), the delivered surface inside the window is also held to the
    drawn solids: nothing there that nobody drew. A region check fails
    outright when a feature is filled, patched over or rounded off, where
    a radius read off a mesh can drift."""
    import trimesh

    tol = ON if tol is None else float(tol)
    name = name or f"drawn:{it.id}"
    P, N, _ = it.brep.samples(spacing=0.2)
    if near is not None:
        keep = near.sdf(P) <= 0.0
        P, N = P[keep], N[keep]
    side = -1.0 if it.op == "add" else 1.0
    probe = P + side * 0.08 * N
    xm = spec.exterior_mesh()
    ext = _signed(xm, probe)
    if getattr(spec, "_solid_mesh", None) is not None and it.op == "cut":
        behind = _inside(spec._solid_mesh, probe)
    elif it.op == "cut":
        # no record of the body before its cuts: the wall the skin keeps is material for certain
        behind = (ext <= -0.02) & (ext >= -0.8 * spec.skin)
    else:
        # an add's surface shows where it stands in the cavity: off the wall the skin keeps
        behind = (ext <= -(max([spec.skin] + [z.thickness for z in spec.zones]) + it.blend + 0.3))
        probe = P
    expect = behind & (_signed(xm, P) <= 0.02) & ~_taken(spec, it, probe)
    d = _distance(mesh, P)
    n = int(expect.sum())
    if n == 0:
        return _row(name, "drawn", None, interface=it.id,
                    detail="none of the interface's surface is expected on the part" +
                           (" inside `near`" if near is not None else ""))
    on = d[expect] <= tol
    kept = float(on.mean())
    worst = float(d[expect].max())
    problems = []
    if kept < KEPT_SHARE:
        miss = P[expect][~on]
        problems.append(f"{100 * (1 - kept):.0f}% of {it.id}'s drawn surface is not on the part "
                        f"(up to {worst:.2f} mm off, round {[round(float(v), 1) for v in miss.mean(axis=0)]})")
    stray = None
    if both:
        if near is None:
            raise ValueError("a surface assertion needs `near`: the window the part is held to the drawing in")
        Q, _ = trimesh.sample.sample_surface(mesh, 200_000, seed=3)
        Q = Q[near.sdf(Q) <= 0.0]
        if len(Q):
            dq = np.minimum(_distance(spec.exterior_mesh(), Q), _distance(it.brep.mesh, Q))
            for o in spec.interfaces:
                if o is not it:
                    dq = np.minimum(dq, _distance(o.brep.mesh, Q))
            for c in spec.components:
                dq = np.minimum(dq, np.abs(_distance(c.brep.mesh, Q) - c.clearance))
            stray = float((dq > max(tol, 0.05)).mean())
            if stray > 1.0 - KEPT_SHARE:
                far = Q[dq > max(tol, 0.05)]
                problems.append(f"{100 * stray:.0f}% of the part's surface in the window is on nothing drawn "
                                f"(up to {float(dq.max()):.2f} mm off, round "
                                f"{[round(float(v), 1) for v in far.mean(axis=0)]})")
    return _row(name, "drawn", not problems, value=_r(kept), limit=KEPT_SHARE,
                detail="; ".join(problems) or None, interface=it.id, worstMm=_r(worst), samples=n,
                **({"strayShare": _r(stray)} if stray is not None else {}))


def _interface_checks(spec, part_man):
    rows = []
    ext = spec.exterior_manifold()
    cuts = [i.brep.manifold() for i in spec.interfaces if i.op == "cut"]
    from .build import _grown, _union

    removed = _union(cuts + [_grown(c.brep, c.clearance) for c in spec.components])
    for it in spec.interfaces:
        man = it.brep.manifold()
        v = max(_vol(man), 1e-9)
        if it.op == "cut":
            left = _vol(part_man ^ man)
            ok = left <= max(1e-4 * v, 0.005)
            rows.append(_row(f"open:{it.id}", "open", ok, value=_r(left), limit=_r(max(1e-4 * v, 0.005)),
                             interface=it.id,
                             detail=None if ok else f"{left:.2f} mm^3 of the part is inside the cut {it.id}"))
        else:
            should = man if it.proud else man ^ ext
            if not removed.is_empty():
                should = should - removed
            missing = _vol(should - part_man)
            lim = max(1e-3 * _vol(should), 0.005)
            ok = missing <= lim
            rows.append(_row(f"present:{it.id}", "present", ok, value=_r(missing), limit=_r(lim), interface=it.id,
                             detail=None if ok else f"{missing:.2f} mm^3 of {it.id} is missing from the part",
                             volume=_r(_vol(should), 2)))
    return rows


def _exterior_check(spec, mesh, out_dir, render):
    from .surfacecheck import compare_exterior

    boxes = []
    for it in spec.interfaces:
        if it.op == "cut" or it.proud:
            lo, hi = it.brep.bounds()
            boxes.append((lo - 0.1, hi + 0.1))
    for c in spec.components:
        lo, hi = c.brep.bounds()
        boxes.append((lo - c.clearance - 0.1, hi + c.clearance + 0.1))
    ext = spec.exterior_mesh()
    try:
        # the outside is the exterior's own triangles by construction: a
        # tenth of a millimetre a pixel finds anything that is not
        span = float(np.max(np.asarray(ext.extents, float)))
        res = compare_exterior(ext, mesh, tolerance=0.05, cuts=boxes, pixel=max(span / 500.0, 0.08))
    except Exception as exc:  # noqa: BLE001
        return _row("exterior", "exterior", None, detail=f"could not compare: {type(exc).__name__}: {exc}"), None
    image = None
    if out_dir and render:
        try:
            from .surfacecheck import ortho_sheet

            image = os.path.join(out_dir, "exterior.png")
            ortho_sheet(ext, mesh, image, tolerance=0.05, cuts=boxes, result=res,
                        title=f"{spec.name}: the exterior as drawn, and the part")
        except Exception:  # noqa: BLE001 - the numbers stand without the picture
            image = None
    in_cuts = sum(v["inCuts"] for v in res["views"].values())
    worst = max(res["views"].items(), key=lambda kv: kv[1]["maxDeviation"])
    return _row("exterior", "exterior", bool(res["pass"]), value=res["maxDeviation"], limit=0.05,
                detail=None if res["pass"] else (f"the outside is {res['maxDeviation']:.2f} mm off the exterior "
                                                 f"in the {worst[0]} view, outside every declared cut"),
                changedInsideCutsMm2=_r(in_cuts, 1)), image


def _process_checks(spec, mesh):
    rows = []
    proc = spec.process
    if proc is None:
        return rows
    from .dfm_reference import review

    floor = None if getattr(spec, "solid", False) else min([spec.skin] + [z.thickness for z in spec.zones])
    measured = None
    if proc in ("sls", "mjf", "sla", "dlp"):
        from .processcheck import check_process

        try:
            # a printer's or a service's own minimum wall, when the spec states it (process.minFeature)
            pd = spec.doc.get("process") if isinstance(spec.doc.get("process"), dict) else {}
            rules = {"minWall": float(pd["minFeature"])} if pd.get("minFeature") else None
            rep = check_process(mesh, proc, build=spec.build_direction, material=spec.material, slicer=False,
                                rules=rules)
        except Exception as exc:  # noqa: BLE001
            rows.append(_row("process", "process", None, detail=f"{type(exc).__name__}: {exc}"))
            rep = None
        if rep is not None:
            measured = rep["walls"].get("judgedWallMm")
            own = [w for w in rep["warnings"] if "Materialize" not in w]
            rows.append(_row("process", "process", bool(rep["ok"]) if rep["probesRan"] else None,
                             value=_r(measured), limit=rep["rules"]["minWall"],
                             detail=" | ".join(own) or None, process=proc,
                             trappedVolumeMm3=rep.get("trappedVolumeMm3"),
                             unformedVolumeMm3=rep["walls"].get("unformedVolumeMm3")))
    ref = review(proc, spec.material, extents=np.asarray(mesh.extents, float), wall_floor=floor,
                 measured_wall=measured)
    if ref is not None:
        rows.append(_row("dfm", "dfm", not ref["warnings"], value=_r(measured if measured is not None else floor),
                         limit=ref["minWall"], detail=" | ".join(ref["warnings"]) or ref.get("note"),
                         reference=ref.get("reference"), accessed=ref.get("accessed"),
                         materialId=(ref.get("material") or {}).get("id"),
                         fitsBuildVolume=ref.get("fitsBuildVolume")))
    return rows


def _assert_rows(spec, mesh, part_man, solid, mass):
    rows = []
    for j, a in enumerate(spec.asserts):
        kind = a["type"]
        name = a.get("id")
        try:
            if kind == "fillet":
                rows.append(fillet_check(spec, mesh, a, solid))
            elif kind == "surface":
                it = spec.interface(a["interface"])
                rows.append(drawn_surface_check(spec, mesh, it, near=_near(a, spec.base_dir),
                                                name=name or f"surface:{it.id}", tol=a.get("tolerance"), both=True))
            elif kind == "wall":
                rows.append(wall_check(spec, mesh, a))
            elif kind == "clearance":
                c = spec.component(a["component"])
                rows.append(clearance_check(spec, mesh, c, float(a["min"]), name or f"clearance:{c.id}>={a['min']:g}",
                                            part_man))
            elif kind == "keepIn":
                c = spec.component(a["component"])
                rows.append(keep_in_check(spec, c, float(a["margin"]), name or f"keepIn:{c.id}"))
            elif kind == "mass":
                of = a.get("of", "assembly")
                m = mass[of]
                lo, hi = a.get("min"), a.get("max")
                ok = (lo is None or m >= lo) and (hi is None or m <= hi)
                rows.append(_row(name or "mass-window", "mass", ok, value=_r(m), limit=[lo, hi],
                                 detail=None if ok else f"the {of} weighs {m:.2f} g"))
            elif kind in ("open", "present"):
                it = spec.interface(a["interface"])
                if (kind == "open") != (it.op == "cut"):
                    raise ValueError(f"{it.id} is a{'n add' if it.op == 'add' else ' cut'} interface: assert "
                                     f"{'present' if it.op == 'add' else 'open'}")
                # the default checks carry it; an assertion makes it explicit
                rows.append(_row(name or f"assert-{kind}:{it.id}", "assert", True,
                                 detail=f"held by the check {kind}:{it.id}"))
            elif kind == "distance":
                def solid_of(i):
                    for x in list(spec.interfaces) + list(spec.components):
                        if x.id == i:
                            return x.brep
                    raise ValueError(f"no interface or component {i!r}")
                A, B = solid_of(a["from"]), solid_of(a["to"])
                P, _, _ = A.samples(spacing=0.25)
                d = float(_distance(B.mesh, P).min())
                lo, hi = a.get("min"), a.get("max")
                ok = (lo is None or d >= lo - 1e-6) and (hi is None or d <= hi + 1e-6)
                rows.append(_row(name or f"distance:{a['from']}-{a['to']}", "distance", ok, value=_r(d),
                                 limit=[lo, hi], detail=None if ok else f"they are {d:.2f} mm apart"))
            else:
                raise ValueError(f"unknown assert type {kind!r} (one of {', '.join(ASSERT_TYPES)})")
        except (ValueError, KeyError) as exc:
            rows.append(_row(name or f"assert-{j + 1}", "assert", False, detail=f"{type(exc).__name__}: {exc}"))
    return rows


# ---- pictures ------------------------------------------------------------------------

def _views(spec, mesh, path):
    """Four fixed views of the part, with its components in place."""
    from .raster import Camera, render, shade_image
    from .reference import _sheet

    c = np.asarray(mesh.bounds, float).mean(axis=0)
    r = float(np.linalg.norm(mesh.extents))
    comps = [k.brep.mesh for k in spec.components]
    cols = [(214, 214, 214)] + [(60, 150, 105)] * len(comps)
    panels, titles = [], []
    for title, eye in (("front left", (-0.9, -1.4, 0.8)), ("back right", (0.9, 1.4, 0.8)),
                       ("underside", (-0.7, -1.2, -1.1)), ("top", None)):
        if eye is None:
            cam = Camera(direction="top", width=560, height=420)
        else:
            cam = Camera(position=c + 2.2 * r * np.asarray(eye) / np.linalg.norm(eye), look_at=c, fov=30,
                         width=560, height=420)
        panels.append(shade_image(render([mesh] + ([] if eye else []), cam), cols))
        titles.append(title)
    return _sheet(panels, titles, path, f"{spec.name}")


def _closeup(spec, mesh, part_man, target, title, path, comps_man=None):
    """One feature up close: from outside, and two sections through it,
    with the components in the sections."""
    from .raster import Camera, render, shade_image
    from .reference import _sheet
    from .shapes import trimesh_from_manifold

    lo, hi = target.bounds()
    # what of the feature is in or on the part: a cut tool runs on outside it
    elo, ehi = spec.exterior.bounds()
    lo, hi = np.maximum(lo, elo), np.minimum(hi, ehi)
    c = 0.5 * (lo + hi)
    size = float(np.linalg.norm(hi - lo))
    span = max(1.5 * size, 8.0)
    # the way out of the part from here: up the exterior's distance
    h = 0.5
    g = np.array([spec.exterior.sdf((c + h * e)[None])[0] - spec.exterior.sdf((c - h * e)[None])[0] for e in np.eye(3)])
    out = g / np.linalg.norm(g) if np.linalg.norm(g) > 1e-6 else np.array([0.0, -1.0, 0.3])
    panels, titles = [], []
    eye = out + (np.array([0.0, 0.0, 0.25]) if abs(out[2]) < 0.9 else np.array([0.2, -0.3, 0.0]))
    eye = eye / np.linalg.norm(eye)
    cam = Camera(position=c + 8.0 * span * eye, look_at=c,
                 fov=float(np.degrees(2 * np.arctan(0.5 * 1.25 * span / (8.0 * span)))), width=520, height=420)
    # only a feature that reaches the outside shows from there (a plunger under the top does not)
    try:
        Pt = target.samples(spacing=max(size / 30.0, 0.3))[0]
        shows = bool(len(Pt)) and float(_distance(spec.exterior_mesh(), Pt).min()) <= 0.3
    except Exception:  # noqa: BLE001
        shows = True
    if shows:
        panels.append(shade_image(render(mesh, cam)))
        titles.append("from outside")
    # two sections that hold the way out: the feature's profile along its own axis
    side = np.cross(out, [0.0, 0.0, 1.0]) if abs(out[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    other = np.cross(out, side)
    other = other / np.linalg.norm(other)
    for label, n in (("section, looking across", side), ("section, the other way", other)):
        half = part_man.trim_by_plane((float(n[0]), float(n[1]), float(n[2])), float(n @ c))
        if half.is_empty():
            continue
        hm = trimesh_from_manifold(half)
        meshes, cols = [hm], [(214, 214, 214)]
        if comps_man is not None and not comps_man.is_empty():
            ch = comps_man.trim_by_plane((float(n[0]), float(n[1]), float(n[2])), float(n @ c))
            if not ch.is_empty():
                meshes.append(trimesh_from_manifold(ch))
                cols.append((60, 150, 105))
        up = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else out
        cam = Camera(position=c - 200.0 * n, look_at=c, up=up, ortho=1.15 * span * 520 / 420, width=520, height=420)
        panels.append(shade_image(render(meshes, cam), cols))
        titles.append(label)
    return _sheet(panels, titles, path, title)


def check_build(spec, mesh, out_dir=None, solid=None, body=None, render=True, target=None) -> dict:
    """Every check for a delivered mesh against its build spec. `solid`:
    the part before its cuts (build.py has it; without it a fillet's kept
    surface is judged against the delivered part alone)."""
    import trimesh

    from .build import _union
    from .shapes import manifold_from_trimesh, trimesh_from_manifold

    rows, images = [], []
    rho = spec.material.density / 1000.0
    try:
        part_man = manifold_from_trimesh(mesh)
        closed = bool(mesh.is_watertight) and mesh.volume > 0
        rows.append(_row("closed", "closed", closed, detail=None if closed else "the mesh has open or doubled edges"))
    except ValueError as exc:
        rows.append(_row("closed", "closed", False, detail=str(exc)))
        return {"ok": False, "checks": rows, "images": images}
    n = len(part_man.decompose())
    rows.append(_row("one-piece", "bodies", n == 1, value=n, limit=1,
                     detail=None if n == 1 else f"the part is {n} separate bodies"))
    spec._solid_mesh = trimesh_from_manifold(solid) if solid is not None else None

    ext_row, img = _exterior_check(spec, mesh, out_dir, render)
    rows.append(ext_row)
    if img:
        images.append(img)
    rows.append(sound_check(mesh))
    rows += _interface_checks(spec, part_man)
    for it in spec.interfaces:
        if it.op == "cut":
            try:
                rows.append(drawn_surface_check(spec, mesh, it))
            except Exception as exc:  # noqa: BLE001 - reported as a check that could not run
                rows.append(_row(f"drawn:{it.id}", "drawn", None, detail=f"{type(exc).__name__}: {exc}"))
    for c in spec.components:
        rows.append(clearance_check(spec, mesh, c, c.clearance, part_man=part_man))
        if c.insert is not None:
            rows.append(way_in_check(spec, mesh, c, part_man))
        if c.keep_in is not None:
            rows.append(keep_in_check(spec, c, c.keep_in))
    rows += _process_checks(spec, mesh)

    carried = sum(c.mass for c in spec.components)
    mass = {"part": float(mesh.volume * rho), "assembly": float(mesh.volume * rho + carried)}
    if spec.target is not None:
        m = mass[spec.target["of"]]
        off = m - spec.target["grams"]
        ok = abs(off) <= spec.target["tolerance"] + 1e-9
        detail = None
        if not ok:
            detail = f"the {spec.target['of']} weighs {m:.2f} g, {abs(off):.2f} g {'over' if off > 0 else 'under'}"
            if target is not None and isinstance(target.get("reachable"), (list, tuple)):
                lo, hi = target["reachable"]
                if spec.target["grams"] > hi or spec.target["grams"] < lo:
                    detail += (f"; the walls named in targetMass.adjust reach {lo:.2f} to {hi:.2f} g: widen their "
                               "limits, add an interface that carries mass, or change the target")
        rows.append(_row("mass", "mass", ok, value=_r(m), limit=[_r(spec.target["grams"] - spec.target["tolerance"]),
                                                                  _r(spec.target["grams"] + spec.target["tolerance"])],
                         detail=detail, of=spec.target["of"]))
    rows += _assert_rows(spec, mesh, part_man, solid, mass)

    if spec.references:
        from .reference import score_references

        ref_dir = os.path.join(out_dir, "references") if out_dir else None
        scored = score_references(mesh, spec.references, spec.base_dir, ref_dir, exterior=spec.exterior_mesh())
        for ref, r in zip(spec.references, scored):
            name = f"reference:{r.get('reference')}"
            # a camera fitted to a sketch scores style and perspective as drift: shown, not gated
            gate = ref.get("gate", "fittedCamera" not in r) if isinstance(ref, dict) else True
            if not gate and "error" not in r:
                if not r["ok"]:
                    r["note"] = ("not gated (a fitted camera or gate: false): for a reviewer's eye, "
                                 "side by side in the sheet")
                    r["ok"] = None
            if "error" in r:
                rows.append(_row(name, "reference", False, detail=r["error"]))
                continue
            if r.get("image"):
                images.append(r["image"])
            if r["kind"] == "image":
                rows.append(_row(name, "reference", r["ok"], value=_r(r["match"]), limit=r["minMatch"],
                                 detail=None if r["ok"] is True else (f"{100 * r['match']:.0f}% of the part's outline lies on "
                                                              f"the reference's edges (95% within "
                                                              f"{r['outline95']:.2f} mm)"),
                                 **{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()
                                    if k in ("outlineMean", "outline95", "iou", "tolerance", "maskFrom",
                                             "fittedCamera", "note")}))
            else:
                rows.append(_row(name, "reference", r["ok"], value=_r(r["iou"]), limit=r["minIou"],
                                 detail=None if r["ok"] else "the part's silhouette or a section differs from the "
                                                             "reference mesh",
                                 views=[{k: (_r(v, 4) if isinstance(v, float) else v) for k, v in x.items()}
                                        for x in r["views"]],
                                 sections=r["sections"]))

    if out_dir and render:
        try:
            images.append(_views(spec, mesh, os.path.join(out_dir, "views.png")))
            comps = _union([c.brep.manifold() for c in spec.components])
            for it in spec.interfaces:
                images.append(_closeup(spec, mesh, part_man, it.brep, f"{it.id} ({it.op})",
                                       os.path.join(out_dir, "closeups", f"{it.id}.png"), comps))
            for c in spec.components:
                images.append(_closeup(spec, mesh, part_man, c.brep, f"{c.id} (component)",
                                       os.path.join(out_dir, "closeups", f"{c.id}.png"), comps))
        except Exception as exc:  # noqa: BLE001 - a picture that fails is reported, the checks stand
            rows.append(_row("images", "images", None, detail=f"{type(exc).__name__}: {exc}"))
    spec._solid_mesh = None
    return {"ok": not any(r["ok"] is False for r in rows), "checks": rows,
            "images": [os.path.relpath(i, out_dir) if out_dir else i for i in images]}
