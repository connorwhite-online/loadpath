"""
Mass fill: make a product as heavy as it can be (or as a target) without
changing how it presses, swings or assembles.

A desk product often needs weight (it should sit still and feel solid) while
its top must stay soft to press. Weight can only go where nothing else needs
the room. This module finds that room inside an assembled product and fills
it with the product's own material (or any other), bottom up, so the centre
of mass stays low:

- inside the exterior: under the skin, over the floor (rays up and down both
  hit the product), and not already solid;
- off every keep-out by its clearance (a board, a switch, wires), and, for a
  keep-out marked `noOverhang`, never over its footprint above its base, so
  the part still drops in from above;
- below the skin's travel envelope (switch_lattice.readback(..., travel_at))
  by `margin`: only room the skin never reaches when pressed;
- for a fill that rides on a moving part (a flap that opens on a hinge):
  its sweep about the hinge axis stays `clearance` off the fixed part.

Then: the room is opened to `minSection` (no feather edges), columns that
reach within `fuse` of the floor are run down into what they stand on so the
fill fuses to it, small islands are dropped, and the result is meshed and
smoothed. If a `target_g` is given and the room holds more, the fill keeps
the lowest voxels that reach it; if the room holds less, the report says the
most it can reach, so a designer knows before printing.

Run: `loadpath massfill brief.json --out fill.stl` (docs/STABILITY.md).
"""
import json
import os
import time

import numpy as np


def _load(p, base):
    import trimesh
    if hasattr(p, "vertices"):
        return p
    path = p if os.path.isabs(p) or base is None else os.path.join(base, p)
    return trimesh.load(path, force="mesh")


def _mf(t):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32),
                               tri_verts=np.asarray(t.faces, np.uint32)))


def _occ(mesh, xs, ys, zs):
    """Occupancy on the grid from manifold slices (even-odd fill of each section)."""
    import shapely
    m = _mf(mesh)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    G = np.zeros((len(xs), len(ys), len(zs)), bool)
    for k, z in enumerate(zs):
        g = np.zeros(X.shape, bool)
        for ring in m.slice(float(z)).to_polygons():
            if len(ring) >= 3:
                g ^= shapely.contains_xy(shapely.Polygon(ring), X, Y)
        G[:, :, k] = g
    return G


def _rot(axis_pt, axis_dir, deg):
    d = np.asarray(axis_dir, float); d /= np.linalg.norm(d)
    K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    a = np.radians(deg)
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return R, np.asarray(axis_pt, float)


def run_massfill(brief, base_dir=None, out=None):
    """Returns the report dict; writes the fill mesh to `out` (or brief["out"])."""
    import trimesh
    from scipy import ndimage
    from skimage import measure
    from .materials import material_from_spec
    t0 = time.time()
    if isinstance(brief, str):
        base_dir = base_dir or os.path.dirname(os.path.abspath(brief))
        with open(brief) as fh:
            brief = json.load(fh)
    H = float(brief.get("pitch", 0.25))
    ext = _load(brief["exterior"], base_dir)
    lo = ext.bounds[0] - 2 * H; hi = ext.bounds[1] + 2 * H
    xs, ys, zs = [np.arange(l + H / 2, h, H) for l, h in zip(lo, hi)]
    solid = _occ(ext, xs, ys, zs)
    above = np.flip(np.logical_or.accumulate(np.flip(solid, 2), 2), 2)
    below = np.logical_or.accumulate(solid, 2)
    room = above & below & ~solid
    room &= ndimage.distance_transform_edt(~solid) * H > float(brief.get("wallClearance", 0.2))
    rep = {"pitch": H, "cavity_cm3": round(room.sum() * H**3 / 1000, 3), "steps": []}
    cavity = above & below & ~solid
    Z = zs[None, None, :]
    if brief.get("floor") is not None:
        room &= Z - H / 2 >= float(brief["floor"]) - 1e-9
    for ko in brief.get("keepouts", []):
        g = _occ(_load(ko["mesh"], base_dir), xs, ys, zs)
        room &= ndimage.distance_transform_edt(~g) * H > float(ko.get("clearance", 0.3))
        if ko.get("noOverhang"):
            foot = g.any(2)
            foot = ndimage.binary_dilation(foot, iterations=max(1, int(np.ceil(float(ko.get("clearance", 0.3)) / H))))
            zb = zs[np.argmax(g.any((0, 1)))]
            room &= ~(foot[:, :, None] & (Z > zb))
        rep["steps"].append({"keepout": ko.get("id"), "room_cm3": round(room.sum() * H**3 / 1000, 3)})
    I = np.argwhere(room)
    P = np.stack([xs[I[:, 0]], ys[I[:, 1]], zs[I[:, 2]]], 1)
    tr = brief.get("travel")
    if tr:
        from scipy.spatial import cKDTree
        if "envelope" in tr:
            e = np.load(tr["envelope"] if os.path.isabs(tr["envelope"]) or base_dir is None
                        else os.path.join(base_dir, tr["envelope"]))
            N, T = e["nodes"], e["travel"]
        else:
            N, T = np.asarray(tr["nodes"], float), np.asarray(tr["travel"], float)
        keep = T > float(tr.get("minTravel", 0.1))           # the skin that moves
        N, T = N[keep], T[keep]
        kd = cKDTree(N[:, :2]); r = float(tr.get("radius", 0.6)); margin = float(tr.get("margin", 0.5))
        ok = np.ones(len(P), bool)
        for i, p in enumerate(P):
            idx = [j for j in kd.query_ball_point(p[:2], r) if N[j, 2] > p[2]]
            if idx:
                ok[i] = p[2] + H / 2 + margin < (N[idx, 2] - T[idx]).min()
        P, I = P[ok], I[ok]
        rep["steps"].append({"travel": "below the envelope", "room_cm3": round(len(P) * H**3 / 1000, 3)})
    mat = material_from_spec(brief.get("material", "tpu88a"))
    rho = float(mat.density)
    rep["material"] = mat.name; rep["density"] = rho
    # what the rules took out of the cavity (not the wall clearance): the fuse step may not refill it
    allowed = np.zeros(solid.shape, bool)
    Ia = np.round((P - lo) / H - 0.5).astype(int); allowed[tuple(Ia.T)] = True
    near_wall = cavity & (ndimage.distance_transform_edt(~solid) * H <= float(brief.get("wallClearance", 0.2)) + 1e-9)
    allowed |= near_wall
    if brief.get("floor") is not None:
        allowed &= Z - H / 2 >= float(brief["floor"]) - 1e-9

    def dist_at(D, Q):
        k = np.floor((Q - lo) / H).astype(int)
        inside = np.all((k >= 0) & (k < D.shape), 1)
        d = np.full(len(Q), 99.0); kk = k[inside]; d[inside] = D[kk[:, 0], kk[:, 1], kk[:, 2]]
        return d
    mv = brief.get("moving")
    fills = {}
    if mv:
        # two fills: one that rides with the moving part (its sweep clears the fixed part)
        # and one fused to the fixed part (clear of everything the moving part sweeps)
        angles = mv.get("angles", [0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90])
        clr = float(mv.get("clearance", 0.3))
        A_pt, A_dir = mv["axis"]["point"], mv["axis"]["direction"]
        S_fix = _occ(_load(mv["against"], base_dir), xs, ys, zs)
        dF = ndimage.distance_transform_edt(~S_fix) * H
        ok = np.ones(len(P), bool)
        for a in angles:
            R, A = _rot(A_pt, A_dir, a)
            ok &= dist_at(dF, (P - A) @ R.T + A) > clr
        Pm = P[ok]
        rep["steps"].append({"moving": "rides with it, swing-clear", "room_cm3": round(len(Pm) * H**3 / 1000, 3)})
        S_mov = np.zeros_like(solid)
        for p in mv.get("parts", []):
            S_mov |= _occ(_load(p, base_dir), xs, ys, zs)
        Im = np.round((Pm - lo) / H - 0.5).astype(int); S_mov[tuple(Im.T)] = True
        dM = ndimage.distance_transform_edt(~S_mov) * H
        Pr = P[~ok]; ok2 = np.ones(len(Pr), bool)
        for a in angles:
            R, A = _rot(A_pt, A_dir, a)
            ok2 &= dist_at(dM, (Pr - A) @ R + A) > clr          # R^T: where the moving part comes from
        Pf = Pr[ok2]
        aw = mv.get("fixedAwayFrom")
        if aw:
            # nor near where it is pressed: a thicker wall there is a stiffer press
            Q = np.asarray(aw["points"], float)[:, :2]
            dxy = np.sqrt(((Pf[:, None, :2] - Q[None]) ** 2).sum(2)).min(1)
            Pf = Pf[dxy > float(aw.get("radius", 6.0))]
        if mv.get("fixedBelow") is not None:
            # fill fused to the fixed part only this low: up the walls it stiffens the skin they hold
            Pf = Pf[Pf[:, 2] + H / 2 <= float(mv["fixedBelow"])]
        rep["steps"].append({"fixed": "fused to the fixed part, clear of the sweep", "room_cm3": round(len(Pf) * H**3 / 1000, 3)})
        fuse_to = {"moving": _occ(_load(mv["fuseTo"], base_dir), xs, ys, zs) if mv.get("fuseTo") else S_mov,
                   "fixed": S_fix}
        fills = {"moving": Pm, "fixed": Pf}
    else:
        fills = {"fill": P}; fuse_to = {"fill": solid}
    rep["room_cm3"] = round(sum(len(v) for v in fills.values()) * H**3 / 1000, 3)
    rep["maxFill_g"] = round(rep["room_cm3"] * rho, 2)
    tgt = brief.get("target_g")
    extra = float(brief.get("otherMass_g", 0.0))
    if tgt is not None:
        # keep the lowest voxels that reach the target, across the fills
        need = int(max(float(tgt) - extra, 0.0) * 1000 / rho / H**3)
        allz = np.concatenate([v[:, 2] for v in fills.values()])
        if need < len(allz):
            zcut = np.sort(allz)[need]
            fills = {k: v[v[:, 2] < zcut] for k, v in fills.items()}
    out = out or brief.get("out")
    import trimesh as _tm
    meshes = {}
    # exact clean-up after meshing: off every keep-out by its clearance, and each fill off
    # the other side of the hinge (the moving fill off the fixed part, the fixed fill off the
    # moving parts as they sit closed; its sweep was checked on the grid)
    cuts = [(_load(ko["mesh"], base_dir), float(ko.get("clearance", 0.3))) for ko in brief.get("keepouts", [])]
    for ko in brief.get("keepouts", []):
        if ko.get("noOverhang"):
            cuts.append((_prism_over(_load(ko["mesh"], base_dir), float(ko.get("clearance", 0.3)), hi[2] + 5), 0.0))
    side = {}
    if mv:
        side = {"moving": [(_load(mv["against"], base_dir), clr)],
                "fixed": [(_load(p, base_dir), clr) for p in mv.get("parts", [])]}
    for key, Pk in fills.items():
        G = np.zeros(solid.shape, bool)
        if len(Pk):
            Ik = np.round((Pk - lo) / H - 0.5).astype(int); G[tuple(Ik.T)] = True
        al = allowed.copy()
        if key == "moving":
            Ib = np.round((P[~ok] - lo) / H - 0.5).astype(int); al[tuple(Ib.T)] = False
        m = _mesh_fill(G, fuse_to[key], solid, brief, H, lo, al)
        if m is not None:
            m = _cut_clear(m, cuts + side.get(key, []))
        if m is not None and key == "fixed" and "moving" in meshes:
            # the exact sweep of the moving side as built (parts and its fill), every angle
            import manifold3d as mf
            mov = mf.Manifold.batch_boolean([_mf(_load(q, base_dir)) for q in mv.get("parts", [])]
                                            + [_mf(meshes["moving"])], mf.OpType.Add)
            sw = []
            for a in angles:
                R, A = _rot(A_pt, A_dir, a)
                Tm = np.hstack([R, (A - R @ A)[:, None]])
                sw.append(mov.transform(Tm.tolist()))
            sweep = mf.Manifold.batch_boolean(sw, mf.OpType.Add)
            m = _cut_clear(m, [(sweep, clr)])
        if m is not None:
            meshes[key] = m
    rep["fills"] = {}
    tot = 0.0
    for key, m in meshes.items():
        vol = abs(float(m.volume)); tot += vol
        row = {"cm3": round(vol / 1000, 3), "g": round(vol * rho / 1000, 2), "pieces": len(m.split(only_watertight=False)),
               "centreOfMass": np.round(m.center_mass, 2).tolist()}
        if out:
            path = out if len(meshes) == 1 and not mv else out.replace(".stl", f"-{key}.stl")
            m.export(path if os.path.isabs(path) or base_dir is None else os.path.join(base_dir, path))
            row["out"] = path
        rep["fills"][key] = row
    rep["fill_cm3"] = round(tot / 1000, 3); rep["fill_g"] = round(tot * rho / 1000, 2)
    rep["note_fuse"] = "each fill overlaps what it is fused to by about one voxel: union it with that part"
    if tgt is not None:
        rep["target_g"] = float(tgt)
        rep["reaches"] = bool(tot * rho / 1000 + extra >= float(tgt) - 0.05)
        if not rep["reaches"]:
            rep["note"] = (f"the room holds about {rep['maxFill_g']} g of {mat.name}; with the rest "
                           f"({extra} g) the product tops out near {round(rep['maxFill_g'] + extra, 1)} g")
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def _mesh_fill(G, fuse_into, solid, brief, H, lo, allowed=None):
    import trimesh
    from scipy import ndimage
    from skimage import measure
    ms = float(brief.get("minSection", 1.0))
    st = ndimage.iterate_structure(ndimage.generate_binary_structure(3, 1), max(1, int(round(ms / 2 / H))))
    G = ndimage.binary_opening(G, st)
    lab, n = ndimage.label(G)
    if not n:
        return None
    sz = ndimage.sum(G, lab, range(1, n + 1))
    G = np.isin(lab, 1 + np.where(sz * H**3 >= float(brief.get("minIsland_mm3", 60)))[0])
    if not G.any():
        return None
    # fuse: bridge the wall clearance to what it stands on and reach one voxel into it,
    # without touching anything else that is solid
    k = int(np.ceil(float(brief.get("wallClearance", 0.2)) / H)) + 2
    grow = ndimage.binary_dilation(G, iterations=k)
    near_f = ndimage.binary_dilation(fuse_into, iterations=k)
    other = solid & ~fuse_into
    far_other = ndimage.distance_transform_edt(~other) * H > float(brief.get("wallClearance", 0.2))
    add = grow & near_f & far_other
    if allowed is not None:
        add &= allowed | fuse_into
    G = G | add
    F = ndimage.gaussian_filter(G.astype(float), float(brief.get("smooth", 1.2)))
    v, f, _, _ = measure.marching_cubes(np.pad(F, 1), float(brief.get("iso", 0.58)))
    v = (v - 1) * H + lo
    m = trimesh.Trimesh(v, f[:, ::-1]); m.fix_normals()
    if m.volume < 0:
        m.invert()
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=20)
    parts = m.split(only_watertight=True)
    return trimesh.util.concatenate(parts) if len(parts) else None


_DIRS = [np.array(v, float) / np.linalg.norm(v) for v in
         [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 1), (1, 1, -1), (1, -1, 1),
          (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]]


def _grow(m, r):
    """A Minkowski grow by r, approximated by the union of 14 translated copies."""
    import manifold3d as mf
    return mf.Manifold.batch_boolean([m] + [m.translate((r * d).tolist()) for d in _DIRS], mf.OpType.Add)


def _prism_over(mesh, clearance, ztop):
    """The keep-out's plan (its convex hull, grown by its clearance), from its base up."""
    import manifold3d as mf
    from scipy.spatial import ConvexHull
    xy = np.asarray(mesh.vertices)[:, :2]
    ring = xy[ConvexHull(xy).vertices]
    cs = mf.CrossSection([ring.tolist()]).offset(clearance, mf.JoinType.Round)
    zb = float(mesh.bounds[0, 2])
    return mf.Manifold.extrude(cs, ztop - zb).translate([0, 0, zb])


def _cut_clear(mesh, cuts):
    import trimesh
    import manifold3d as mf
    b = _mf(mesh)
    for m, r in cuts:
        mm = m if isinstance(m, mf.Manifold) else _mf(m)
        b = b - (_grow(mm, r) if r > 0 else mm)
    g = b.to_mesh()
    t = trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts))
    t.merge_vertices()
    parts = [p for p in t.split(only_watertight=False) if abs(p.volume) > 1.0]
    return trimesh.util.concatenate(parts) if parts else None
