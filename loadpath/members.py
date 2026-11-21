"""
Members: a topology optimizer's density rebuilt as clean, deliberate
structure.

A density field says where the load goes. Meshed as it stands it reads as
mush: lumps where members meet, thickness that wanders along a member, thin
webs half-formed between them. A designer redraws it as members: straight or
gently curving bars of one section each, meeting at nodes with smooth
fillets. This module does that redraw:

1. the solid (density over `threshold`), less what is not to be redrawn
   (`exclude`: the skin, the seats, anything kept as drawn), is thinned to
   its curve skeleton (scikit-image, Lee et al.);
2. the skeleton's voxels become a graph: nodes where three or more meet or a
   branch ends, branches between them; spurs shorter than their own
   thickness are pruned, and branches that end inside `exclude` are kept
   as roots (they carry the load into the skin or a seat);
3. each branch is smoothed and simplified (Ramer-Douglas-Peucker to `tol`
   mm), so a member is a few straight runs, not a staircase;
4. each branch gets one radius, the median of the solid's local half
   thickness along it, snapped to `quantum` and at least half `min_member`:
   members of a few consistent sizes;
5. the members are an exact distance field (capsules, Lipschitz 1) joined by
   a smooth minimum of radius `blend` (fillets at every node, and where a
   member roots into the excluded solid when that solid's field is given).

`overhang` reports each member's angle from the build plate: a member
flatter than 45 degrees (`angle`) is a bridge in FDM and a downskin in metal
powder bed, and is listed so it can be rerouted or the build turned.

    from loadpath.members import members_from_density
    M = members_from_density(rho, origin, pitch, exclude=keep_mask, min_member=2.0)
    M.sdf(P)                 # the members' field
    M.report                 # counts, sizes, overhangs
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Members", "Webs", "members_from_density", "rdp", "skeleton_graph", "webs_from_density"]


def _neighbours():
    o = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    return np.array(o)


def skeleton_graph(skel: np.ndarray):
    """Branches of a 1-voxel skeleton as index paths. Returns (branches, nodes):
    branches a list of (n, 3) int arrays from node to node (or end), nodes the
    voxels where three or more branches meet."""
    idx = np.argwhere(skel)
    if not len(idx):
        return [], np.zeros((0, 3), int)
    lut = {tuple(v): i for i, v in enumerate(idx)}
    nb = _neighbours()
    adj = [[] for _ in range(len(idx))]
    for i, v in enumerate(idx):
        for o in nb:
            j = lut.get((v[0] + o[0], v[1] + o[1], v[2] + o[2]))
            if j is not None:
                adj[i].append(j)
    deg = np.array([len(a) for a in adj])
    # a junction is a cluster of voxels of degree > 2; collapse it to one node
    junction = deg > 2
    seen_edge = set()
    branches = []
    for s in np.flatnonzero(deg != 2):
        for j in adj[s]:
            if (s, j) in seen_edge:
                continue
            path = [s, j]
            prev, cur = s, j
            while deg[cur] == 2:
                nxt = [a for a in adj[cur] if a != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
                if cur == s:
                    break
            seen_edge.add((path[-1], path[-2]))
            seen_edge.add((s, j))
            if len(path) == 2 and junction[path[0]] and junction[path[1]]:
                continue                       # two voxels of one junction cluster
            branches.append(idx[path])
    # loops with no node at all
    on = np.zeros(len(idx), bool)
    for b in branches:
        for v in b:
            on[lut[tuple(v)]] = True
    for s in np.flatnonzero(~on & (deg == 2)):
        if on[s]:
            continue
        path, prev, cur = [s], -1, s
        while True:
            on[cur] = True
            nxt = [a for a in adj[cur] if a != prev and not (a == s and len(path) < 3)]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
            if cur == s or on[cur] and cur != s:
                break
        branches.append(idx[path])
    return branches, idx[junction]


def rdp(P: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: the fewest points within tol of the polyline."""
    if len(P) < 3:
        return P
    a, b = P[0], P[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        d = np.linalg.norm(P - a, axis=1)
    else:
        d = np.linalg.norm(np.cross(P - a, ab / L), axis=1)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return np.array([a, b])
    return np.vstack([rdp(P[: i + 1], tol)[:-1], rdp(P[i:], tol)])


def _smooth(P, n=4):
    """Laplacian smoothing with the ends held."""
    P = P.astype(float).copy()
    for _ in range(n):
        if len(P) > 2:
            P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
    return P


def _seg_dist(X, A, B):
    """Distance from points X (m, 3) to segments A-B (k, 3): (m, k)."""
    ab = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    ax = X[:, None, :] - A[None]
    t = np.clip(np.einsum("mkj,kj->mk", ax, ab) / L2[None], 0.0, 1.0)
    q = A[None] + t[..., None] * ab[None]
    return np.linalg.norm(X[:, None, :] - q, axis=2)


def _smin_many(D, k):
    """Smooth minimum of the columns of D (quadratic polynomial, pairwise from the smallest)."""
    if D.shape[1] == 1 or k <= 0:
        return D.min(axis=1)
    D = np.sort(np.minimum(D, 1e6), axis=1)
    a = D[:, 0]
    for j in range(1, min(D.shape[1], 4)):
        b = D[:, j]
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        a = b * (1 - h) + a * h - k * h * (1 - h)
    return a


@dataclass
class Members:
    """Capsule members: segments A[i]-B[i] of radius r[i], joined with a smooth
    minimum of radius `blend`."""
    A: np.ndarray
    B: np.ndarray
    r: np.ndarray
    blend: float = 1.0
    branch: np.ndarray = None
    report: dict = field(default_factory=dict)

    def sdf(self, P, chunk=200_000):
        from scipy.spatial import cKDTree
        P = np.asarray(P, float)
        out = np.full(len(P), np.inf)
        if not len(self.A):
            return out
        mid = 0.5 * (self.A + self.B)
        half = 0.5 * np.linalg.norm(self.B - self.A, axis=1) + self.r
        reach = float(half.max()) + self.blend + 0.5
        tree = cKDTree(mid)
        for s in range(0, len(P), chunk):
            X = P[s:s + chunk]
            # the segments that can be nearest: a mid within reach of the point
            d0, _ = tree.query(X, k=1)
            near = d0 < reach + 6.0
            # far from every member: a lower bound on the distance (positive, finite)
            out[s:s + chunk] = d0 - reach
            if not near.any():
                continue
            Xn = X[near]
            lists = tree.query_ball_point(Xn, reach + self.blend)
            res = np.full(len(Xn), np.inf)
            # group by candidate count to vectorise
            order = np.argsort([len(l) for l in lists])
            for g0 in range(0, len(order), 4096):
                ids = order[g0:g0 + 4096]
                kk = max(1, max(len(lists[i]) for i in ids))
                C = np.zeros((len(ids), kk), int)
                M = np.zeros((len(ids), kk), bool)
                for a, i in enumerate(ids):
                    l = lists[i]
                    C[a, :len(l)] = l
                    M[a, :len(l)] = True
                Xi = Xn[ids]
                ab = self.B[C] - self.A[C]
                L2 = np.maximum(np.einsum("mkj,mkj->mk", ab, ab), 1e-12)
                ax = Xi[:, None, :] - self.A[C]
                t = np.clip(np.einsum("mkj,mkj->mk", ax, ab) / L2, 0.0, 1.0)
                q = self.A[C] + t[..., None] * ab
                D = np.linalg.norm(Xi[:, None, :] - q, axis=2) - self.r[C]
                D[~M] = np.inf
                if not M.any(axis=1).all():
                    D[~M.any(axis=1), 0] = 1e3
                res[ids] = _smin_many(D, self.blend)
            o = out[s:s + chunk]
            o[near] = res
        return out

    def overhang(self, build="+z", angle=45.0):
        """Each member's rise from the build plate, degrees; the ones flatter than `angle`."""
        b = _axis_vec(build)
        d = self.B - self.A
        L = np.linalg.norm(d, axis=1)
        rise = np.degrees(np.arcsin(np.clip(np.abs(d @ b) / np.maximum(L, 1e-9), 0, 1)))
        flat = rise < angle
        return {"members": len(L), "length mm": round(float(L.sum()), 1),
                f"flatter than {angle:g} deg": int(flat.sum()),
                "flat length mm": round(float(L[flat].sum()), 1),
                "flat fraction": round(float(L[flat].sum() / max(L.sum(), 1e-9)), 3),
                "flat": [[np.round(self.A[i], 1).tolist(), np.round(self.B[i], 1).tolist(), round(float(rise[i]), 1)]
                         for i in np.flatnonzero(flat)[:20]]}


def _axis_vec(s):
    if not isinstance(s, str):
        v = np.asarray(s, float)
        return v / np.linalg.norm(v)
    v = np.zeros(3)
    v["xyz".index(s[-1])] = -1.0 if s[0] == "-" else 1.0
    return v


def members_from_density(rho, origin, pitch, threshold=0.5, exclude=None, min_member=2.0, quantum=0.25,
                         tol=None, blend=None, min_length=None, max_radius=None, root_depth=None):
    """Clean members from a density grid (see the module docstring).

    rho: (nx, ny, nz) densities on nodes at origin + pitch * index. exclude: a
    bool grid of the same shape that is kept as drawn (the skin, the seats):
    not redrawn, but members that reach it end there as roots. Returns Members."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    rho = np.asarray(rho, float)
    origin = np.asarray(origin, float)
    solid = rho > threshold
    ex = np.zeros_like(solid) if exclude is None else np.asarray(exclude, bool)
    free = solid & ~ex
    # half thickness of the whole solid (so a member's root keeps its section into the skin)
    dt = ndi.distance_transform_edt(solid) * pitch
    sk = skeletonize(free).astype(bool)
    branches, _nodes = skeleton_graph(sk)
    r_min = 0.5 * min_member
    tol = tol if tol is not None else max(0.5 * pitch, 0.25 * min_member)
    min_length = min_length if min_length is not None else 2.0 * min_member
    blend = blend if blend is not None else 0.6 * min_member
    root_depth = root_depth if root_depth is not None else 1.5 * pitch
    exd = ndi.distance_transform_edt(~ex) * pitch if ex.any() else None
    A, Bv, R, Br = [], [], [], []
    kept, pruned, roots = 0, 0, 0
    deg = {}
    for b in branches:
        for v in (tuple(b[0]), tuple(b[-1])):
            deg[v] = deg.get(v, 0) + 1
    for bi, b in enumerate(branches):
        P = origin + pitch * b.astype(float)
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        rr = dt[tuple(b.T)]
        r = float(np.median(rr))
        r = max(r_min, quantum * round(r / quantum))
        if max_radius:
            r = min(r, max_radius)
        # an end touching the excluded solid is a root: run it into the skin or seat
        def at_root(v):
            # the skeleton stops about a radius short of where the member meets the kept solid
            return exd is not None and exd[tuple(v)] <= dt[tuple(v)] + 1.5 * pitch
        free_end = [deg.get(tuple(b[0]), 0) == 1 and not at_root(b[0]), deg.get(tuple(b[-1]), 0) == 1 and not at_root(b[-1])]
        if (free_end[0] or free_end[1]) and L < max(min_length, 2.0 * r):
            pruned += 1
            continue
        P = _smooth(P)
        for e, v in ((0, b[0]), (-1, b[-1])):
            if at_root(v) and len(P) > 1:
                # push the end on into the excluded solid by root_depth along the member
                d = P[e] - P[1 if e == 0 else -2]
                n = np.linalg.norm(d)
                if n > 1e-9:
                    P[e] = P[e] + d / n * (exd[tuple(v)] + root_depth)
                roots += 1
        Q = rdp(P, tol)
        if len(Q) < 2:
            continue
        A.append(Q[:-1])
        Bv.append(Q[1:])
        R.append(np.full(len(Q) - 1, r))
        Br.append(np.full(len(Q) - 1, bi))
        kept += 1
    if A:
        A, Bv, R, Br = np.vstack(A), np.vstack(Bv), np.concatenate(R), np.concatenate(Br)
    else:
        A = Bv = np.zeros((0, 3))
        R = Br = np.zeros(0)
    sizes = {}
    for r in np.round(2 * R, 2):
        sizes[f"{r:g}"] = sizes.get(f"{r:g}", 0) + 1
    rep = {"branches": len(branches), "members": kept, "pruned spurs": pruned, "roots": roots,
           "segments": len(A), "diameters mm (segments)": dict(sorted(sizes.items(), key=lambda t: float(t[0]))),
           "skeleton voxels": int(sk.sum()), "solid voxels redrawn": int(free.sum()),
           "settings": {"threshold": threshold, "minMember": min_member, "quantum": quantum, "tol": tol,
                        "blend": blend, "minLength": min_length}}
    return Members(A, Bv, R, blend=blend, branch=Br, report=rep)


# -- webs: the plate-like parts of a density, as flat ribs ---------------------

def _unused_ridge(dt, mask):
    """Voxels of the solid on its medial surface: dt a strict local maximum
    across the solid along at least one of the 13 lattice directions."""
    dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    out = np.zeros_like(mask)
    for o in dirs:
        a = np.roll(dt, o, axis=(0, 1, 2))
        b = np.roll(dt, tuple(-x for x in o), axis=(0, 1, 2))
        out |= (dt >= a) & (dt >= b) & ((dt > a) | (dt > b))
    return out & mask


@dataclass
class Webs:
    """Flat ribs: each a slab |n.(x - c)| <= t/2 inside a 2D outline in its plane
    (u, v axes, a signed distance raster `out2d` at `pitch2d` from `org2d`)."""
    planes: list
    report: dict = field(default_factory=dict)

    def sdf(self, P):
        from scipy.ndimage import map_coordinates
        P = np.asarray(P, float)
        out = np.full(len(P), 1e3)
        for w in self.planes:
            q = P - w["c"]
            h = np.abs(q @ w["n"]) - 0.5 * w["t"]
            u, v = q @ w["u"], q @ w["v"]
            iu = (u - w["org2d"][0]) / w["pitch2d"]
            iv = (v - w["org2d"][1]) / w["pitch2d"]
            o2 = map_coordinates(w["out2d"], np.vstack([iu, iv]), order=1, mode="constant", cval=1e3)
            # the slab and its outline, the edge rounded to half the thickness
            r = 0.5 * w["t"]
            a, b = h + r, o2 + r
            d = np.minimum(np.maximum(a, b), 0.0) + np.hypot(np.maximum(a, 0), np.maximum(b, 0)) - r
            out = np.minimum(out, d)
        return out

    def overhang(self, build="+z", angle=45.0):
        """A web prints unsupported when its plane stands at `angle` or more from the
        bed (its normal at least that far from the build direction's square)."""
        b = _axis_vec(build)
        rows = []
        for w in self.planes:
            tilt = np.degrees(np.arccos(np.clip(abs(float(w["n"] @ b)), 0, 1)))   # plane's angle from the bed
            rows.append(round(float(tilt), 1))
        flat = [i for i, t in enumerate(rows) if t < angle]
        return {"webs": len(rows), "plane angles from bed": rows, f"flatter than {angle:g} deg": len(flat),
                "flat area cm2": round(sum(self.planes[i]["area"] for i in flat) / 100.0, 1)}


def _local_shape(mask, pitch, radius):
    """Per voxel of mask: eigenvalues (ascending) and the eigenvector of the smallest, of the
    covariance of the solid within a box of half size `radius` round it (moments by box filters)."""
    from scipy import ndimage as ndi
    n = max(3, round(2 * radius / pitch) | 1)
    m = mask.astype(np.float32)
    I = np.indices(mask.shape, dtype=np.float32) * pitch
    f = lambda a: ndi.uniform_filter(a, n, mode="constant")
    w = f(m)
    w = np.maximum(w, 1e-6)
    mu = [f(m * I[i]) / w for i in range(3)]
    C = np.zeros(mask.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            c = f(m * I[i] * I[j]) / w - mu[i] * mu[j]
            C[..., i, j] = C[..., j, i] = c
    sel = np.argwhere(mask)
    ev, V = np.linalg.eigh(C[tuple(sel.T)].astype(float))
    return sel, ev, V[:, :, 0]


def webs_from_density(rho, origin, pitch, threshold=0.5, exclude=None, min_member=2.0, quantum=0.25,
                      planar=0.35, radius=None, flat_rms=None, min_area=None, merge_deg=15.0):
    """The plate-like parts of a density as flat ribs of one thickness each.

    Each solid voxel's neighbourhood (a box `radius` round it, default 2.5
    min_member) is a plate when it spreads in two directions and is thin in
    the third (PCA: middle spread at least `planar` of the largest, and the
    smallest under a quarter of the middle). Plate voxels are grown into
    patches of near-parallel normals (within `merge_deg`), each fitted with
    a plane; a patch within `flat_rms` of it becomes a web: the patch's
    outline in that plane (closed and opened so it reads as one clean
    shape), its thickness the patch's volume over its area, snapped to
    `quantum` and at least `min_member`. Returns (Webs, plate mask): the
    mask is the solid the webs stand for, for members_from_density to leave
    out."""
    from scipy import ndimage as ndi

    rho = np.asarray(rho, float)
    origin = np.asarray(origin, float)
    solid = rho > threshold
    ex = np.zeros_like(solid) if exclude is None else np.asarray(exclude, bool)
    free = solid & ~ex
    radius = radius if radius is not None else 2.5 * min_member
    flat_rms = flat_rms if flat_rms is not None else max(0.75 * pitch, 0.5 * min_member)
    min_area = min_area if min_area is not None else (3.0 * min_member) ** 2
    if free.sum() < 20:
        return Webs([], {"webs": 0}), np.zeros_like(solid)
    sel, ev, nrm = _local_shape(free, pitch, radius)
    ev = np.maximum(ev, 0)
    isplate = (ev[:, 1] >= planar * ev[:, 2]) & (ev[:, 0] < 0.25 * ev[:, 1])
    lut = -np.ones(solid.shape, int)
    lut[tuple(sel.T)] = np.arange(len(sel))
    lab = -np.ones(len(sel), int)
    cos_m = np.cos(np.radians(merge_deg))
    offs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
    nlab = 0
    shp = np.array(solid.shape)
    for s0 in np.flatnonzero(isplate):
        if lab[s0] >= 0:
            continue
        lab[s0] = nlab
        stack = [s0]
        acc = nrm[s0].copy()
        while stack:
            i = stack.pop()
            q = sel[i] + offs
            ok = np.all((q >= 0) & (q < shp), axis=1)
            for j in lut[tuple(q[ok].T)]:
                if j >= 0 and lab[j] < 0 and isplate[j]:
                    n0 = acc / np.linalg.norm(acc)
                    if abs(nrm[j] @ n0) >= cos_m:
                        lab[j] = nlab
                        acc += nrm[j] * np.sign(nrm[j] @ n0)
                        stack.append(j)
        nlab += 1
    planes, used = [], np.zeros(len(sel), bool)
    for L in range(nlab):
        ids = np.flatnonzero(lab == L)
        if len(ids) < 20:
            continue
        Q = origin + pitch * sel[ids]
        c = Q.mean(0)
        w, V = np.linalg.eigh((Q - c).T @ (Q - c))
        n = V[:, 0]
        u, v = V[:, 2], V[:, 1]
        uv = np.c_[(Q - c) @ u, (Q - c) @ v]
        p2 = pitch
        o2 = uv.min(0) - 3 * p2
        sz = np.ceil((uv.max(0) - o2) / p2).astype(int) + 4
        img = np.zeros(sz, bool)
        ij = np.round((uv - o2) / p2).astype(int)
        img[ij[:, 0], ij[:, 1]] = True
        img = ndi.binary_closing(img, iterations=2)
        img = ndi.binary_fill_holes(img)
        img = ndi.binary_opening(img, iterations=1)
        area = float(img.sum()) * p2 * p2
        if area < min_area:
            continue
        vol = len(ids) * pitch ** 3
        t = vol / area
        res = (Q - c) @ n
        if np.sqrt(np.mean(res ** 2)) > max(flat_rms, 0.4 * t):
            continue
        t = max(min_member, quantum * round(t / quantum))
        o2d = (ndi.distance_transform_edt(~img) - ndi.distance_transform_edt(img)) * p2
        planes.append({"c": c, "n": n, "u": u, "v": v, "t": t, "out2d": o2d, "org2d": o2, "pitch2d": p2,
                       "area": area, "voxels": len(ids)})
        used[ids] = True
    mask = np.zeros_like(solid)
    mask[tuple(sel[used].T)] = True
    sizes = {}
    for w in planes:
        sizes[f"{w['t']:g}"] = sizes.get(f"{w['t']:g}", 0) + 1
    rep = {"webs": len(planes), "thickness mm (webs)": dict(sorted(sizes.items(), key=lambda t: float(t[0]))),
           "area cm2": round(sum(w["area"] for w in planes) / 100.0, 1),
           "plate voxels": int(isplate.sum()), "of solid voxels": len(sel), "patches": int(nlab)}
    return Webs(planes, rep), mask
