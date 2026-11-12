"""
Seats: finish the inside of a part around the components it holds so it
reads as designed and is strong where it should be, not as a stack of
clearance boxes.

Given the part (one or more meshes unioned), the components it seats
(keep-outs with a clearance), and the region that may change, it works on
a fine signed-distance grid:

1. fillets: a closing of radius `fillet` (mm) rounds every inside corner
   (where a wall or a block meets the floor it stands on), which is where
   a stiffener's root is strongest with material;
2. seats: each keep-out is cut back at its clearance by true distance, so
   a pocket round a box is the box grown by a ball: its inside corners
   are rounded by the clearance itself, and with `leadIn` (deg) the walls
   flare outward for `leadHeight` above the component's edge so it drops in by
   itself, then rise straight so a wall beside it still stops the skin;
   `noOverhang` keeps the plan clear above the component's base;
3. edges: an opening of radius `round` softens the outside edges (a
   pocket's rim, the top of a wall) without touching any seat face;
4. meshed at the grid's pitch (marching cubes on the signed distance) and
   unioned back onto the part outside the region, which is left exact.

Only the `region` changes (a box, and `zMin` above which the part may be
reworked), so the part's mating faces (a flap's bead and rebate, a hinge
strap) stay as drawn.
"""
import numpy as np


def _mf(t):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32),
                               tri_verts=np.asarray(t.faces, np.uint32)))


def _tm(m):
    import trimesh
    g = m.to_mesh()
    t = trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts))
    t.merge_vertices()
    return t


def _occ(m, xs, ys, zs):
    import shapely
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    G = np.zeros((len(xs), len(ys), len(zs)), bool)
    for k, z in enumerate(zs):
        g = np.zeros(X.shape, bool)
        for ring in m.slice(float(z)).to_polygons():
            if len(ring) >= 3:
                g ^= shapely.contains_xy(shapely.Polygon(ring), X, Y)
        G[:, :, k] = g
    return G


def finish_seats(part, keepouts, region, fillet=0.8, round_=0.4, pitch=0.1, keep_exact=None, smooth=2.0, taubin=10):
    """part: trimesh (or list); keepouts: [{"mesh", "clearance", "noOverhang"?, "leadIn"? deg,
    "leadFrom"? mm above the base where the flare starts, "leadHeight"? mm it flares over (1.0)}]; region: {"min": [x,y,z], "max": [x,y,z]}.
    keep_exact: meshes whose material must not be removed (unioned back exactly).
    Returns (trimesh, report)."""
    import trimesh
    import manifold3d as mf
    from scipy import ndimage
    from skimage import measure
    parts = part if isinstance(part, (list, tuple)) else [part]
    P = mf.Manifold.batch_boolean([_mf(p) for p in parts], mf.OpType.Add)
    H = float(pitch)
    lo = np.asarray(region["min"], float); hi = np.asarray(region["max"], float)
    pad = 3 * H + max(fillet, round_)
    glo, ghi = lo - pad, hi + pad
    xs, ys, zs = [np.arange(l + H / 2, h, H) for l, h in zip(glo, ghi)]
    S = _occ(P, xs, ys, zs)
    inside_region = np.zeros_like(S)
    i0 = [np.searchsorted(a, v) for a, v in zip((xs, ys, zs), lo)]
    i1 = [np.searchsorted(a, v) for a, v in zip((xs, ys, zs), hi)]
    inside_region[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = True
    S0 = S.copy()
    # 1 fillets: closing
    if fillet > 0:
        dil = ndimage.distance_transform_edt(~S) * H <= fillet
        S = ndimage.distance_transform_edt(dil) * H > fillet
    # 2 seats
    Z = zs[None, None, :]
    pocket = np.zeros_like(S)
    for ko in keepouts:
        km = _mf(ko["mesh"]) if not isinstance(ko["mesh"], mf.Manifold) else ko["mesh"]
        K = _occ(km, xs, ys, zs)
        clr = float(ko.get("clearance", 0.3))
        pocket |= ndimage.distance_transform_edt(~K) * H < clr
        if ko.get("noOverhang") or ko.get("leadIn"):
            foot = K.any(2)
            dxy = ndimage.distance_transform_edt(~foot) * H
            zb = float(ko["mesh"].bounds[0, 2]) if hasattr(ko["mesh"], "bounds") else float(km.bounding_box().min()[2])
            reach = np.full(Z.shape, clr, float)
            if ko.get("leadIn"):
                z0 = zb + float(ko.get("leadFrom", 0.0))
                # the flare is a lead-in, not a funnel: it opens over `leadHeight` mm and then
                # rises straight, so walls standing beside the part (a stop under the skin) stay
                reach = clr + np.clip(Z - z0, 0, float(ko.get("leadHeight", 1.0))) * np.tan(np.radians(float(ko["leadIn"])))
            if ko.get("noOverhang") or ko.get("leadIn"):
                pocket |= (dxy[:, :, None] < reach) & (Z > zb + (0 if ko.get("noOverhang") else float(ko.get("leadFrom", 0.0))))
    S &= ~pocket
    # 3 soften outside edges: opening, removing only
    if round_ > 0:
        er = ndimage.distance_transform_edt(S) * H > round_
        S = ndimage.distance_transform_edt(~er) * H <= round_
        S &= ~pocket
    # outside the region the part stays as it was
    S = np.where(inside_region, S, S0)
    # 4 mesh: signed distance, marching cubes
    sd = ndimage.distance_transform_edt(S) * H - ndimage.distance_transform_edt(~S) * H
    # the distance of a voxel set steps by a voxel; a blur of two voxels takes the terraces out
    # (the seats themselves are re-cut exactly below, so the blur cannot loosen a fit)
    sd = ndimage.gaussian_filter(sd.astype(np.float32), float(smooth))
    # the blur rounds a pocket's corners inward too: hold every pocket voxel empty again
    pk = pocket & inside_region
    if pk.any():
        sd = np.minimum(sd, (H / 2 - ndimage.distance_transform_edt(pk) * H).astype(np.float32))
    v, f, _, _ = measure.marching_cubes(np.pad(sd, 1, constant_values=-1.0), 0.0)
    v = (v - 1) * H + glo + H / 2
    m = trimesh.Trimesh(v, f[:, ::-1]); m.fix_normals()
    if taubin:
        trimesh.smoothing.filter_taubin(m, iterations=int(taubin))
    if m.volume < 0:
        m.invert()
    parts_ = m.split(only_watertight=True)
    m = trimesh.util.concatenate(parts_) if len(parts_) else m
    # the region from the smooth mesh, the rest exact
    big = 500.0
    box = mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist())
    # the exact part reaches 0.03 mm into the region so the two never meet edge to edge
    # on the region's faces (that leaves pinched, non-manifold seams once saved as STL)
    e = 0.03
    box_in = mf.Manifold.cube((hi - lo - 2 * e).tolist()).translate((lo + e).tolist())
    outer = mf.Manifold.cube([big] * 3).translate([-big / 2] * 3) - box_in
    res = (_mf(m) ^ box) + (P ^ outer)
    for ke in keep_exact or []:
        res = res + _mf(ke)
    # exact clean-up off the seats: the voxel seat is within a pitch of exact, and the blur
    # rounds a pocket's corners inward, so each keep-out grown by its clearance is cut again
    from .massfill import _prism_over
    for ko in keepouts:
        km = _mf(ko["mesh"]) if not isinstance(ko["mesh"], mf.Manifold) else ko["mesh"]
        clr = float(ko.get("clearance", 0.3))
        res = res - (km.minkowski_sum(mf.Manifold.sphere(clr, 24)) if clr > 0 else km)
        if ko.get("noOverhang") and hasattr(ko["mesh"], "vertices"):
            res = res - _prism_over(ko["mesh"], clr, float(hi[2]) + pad)
    out = _tm(res)
    pieces = [p for p in out.split(only_watertight=False) if abs(p.volume) > 0.5]
    out = trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    rep = {"pitch": H, "fillet": fillet, "round": round_, "volume_mm3": round(abs(float(out.volume)), 1),
           "before_mm3": round(P.volume(), 1), "pieces": len(pieces), "watertight": bool(out.is_watertight)}
    return out, rep
