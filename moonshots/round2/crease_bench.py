"""
Moonshot round 2: sharp creases between ORGANIC field surfaces.

Meshes test parts whose creases are the meeting of two smooth field
surfaces (no plain shape on either side, so sharp.py cannot help), before
(fields.field_mesh: marching cubes + relax + projection) and after
loadpath.crease.sharpen, and measures against the true crease:

  crease->mesh   distance from dense samples of the true intersection curve
                 to the mesh (max, RMS): sharp.edge_accuracy's idea, on a curve
  vertex->crease distance of the mesh's crease vertices from the true curve
                 (before: the ends of label-changing edges, the vertices
                 nearest the crease; after: the snapped / inserted vertices)
  dihedral       sharpest mesh edge within one pitch of each crease sample,
                 against the true angle between the two surfaces' normals
  validity       watertight + consistent winding, manifold3d accepts it,
                 folds (faces turned against the true normal), self-
                 intersections near the crease, slivers (min angle < 5 / 10 deg)
  runtime        marching cubes pipeline vs the sharpen pass

Run (one process, single-threaded BLAS):
  OMP_NUM_THREADS=1 ... python moonshots/round2/crease_bench.py [--quick] [--case NAME]
Writes moonshots/round2/results.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loadpath.crease import CreaseTree, _edges, meet, sharpen
from aware import finish  # noqa: E402
from loadpath.fields import FieldGraph  # noqa: E402

HERE = Path(__file__).resolve().parent


def sphere(cx, cy, cz, r):
    return {"op": "expr", "expr": f"sqrt((x-({cx}))**2 + (y-({cy}))**2 + (z-({cz}))**2) - {r}"}


def lens(phi_deg, R=10.0, op="intersect"):
    """Two radius-R spheres whose surfaces meet with `phi_deg` between their
    outward normals (intersect: a convex rim of interior angle 180 - phi;
    union: a concave notch)."""
    d = math.sqrt(2 * R * R - 2 * R * R * math.cos(math.radians(phi_deg)))
    nodes = {"a": sphere(-d / 2, 0, 0, R), "b": sphere(d / 2, 0.3, 0.2, R),
             "part": {"op": op, "inputs": ["a", "b"]}}
    lo = [-d / 2 - R - 1, -R - 1, -R - 1]
    hi = [d / 2 + R + 1, R + 1.3, R + 1.2]
    return {"nodes": nodes, "output": "part", "bounds": {"min": lo, "max": hi}}, {"R": R, "d": d}


def blob():
    """An organic body: three spheres, smooth-unioned, then offset."""
    return {
        "s1": sphere(-5, 0, 0, 7), "s2": sphere(5, 1, 0.5, 6), "s3": sphere(0, -2, 5, 5),
        "u1": {"op": "union", "inputs": ["s1", "s2"], "blend": 4},
        "u2": {"op": "union", "inputs": ["u1", "s3"], "blend": 3},
        "body": {"op": "offset", "input": "u2", "distance": -0.4},
    }


BLOB_BOUNDS = {"min": [-13, -8, -8], "max": [12, 8.5, 11]}


def cases():
    out = {}
    doc, meta = lens(90)
    out["lens90"] = (doc, meta, [0.8, 0.4, 0.2])
    doc, meta = lens(150)       # a 30 degree wedge
    out["lens30"] = (doc, meta, [0.8, 0.4, 0.2])
    doc, meta = lens(90, op="union")
    out["union90"] = (doc, meta, [0.8, 0.4, 0.2])
    # a gyroid clipped by an organic body (exposed lattice edges)
    n = blob()
    n["gyr"] = {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": 1.2}
    n["part"] = {"op": "intersect", "inputs": ["gyr", "body"]}
    out["gyroid_clip"] = ({"nodes": n, "output": "part", "bounds": BLOB_BOUNDS}, {}, [0.4, 0.25, 0.15])
    # skin + core: a 1.5 mm organic skin over a gyroid (walls meet the skin's inside)
    n = blob()
    n["gyr"] = {"op": "lattice", "type": "gyroid", "cell": 6, "thickness": 1.2}
    n["skin"] = {"op": "shell", "input": "body", "thickness": 1.5}
    n["core"] = {"op": "intersect", "inputs": ["gyr", "body"]}
    n["part"] = {"op": "union", "inputs": ["skin", "core"]}
    out["skin_core"] = ({"nodes": n, "output": "part", "bounds": BLOB_BOUNDS}, {}, [0.4, 0.25, 0.15])
    # an organic body minus another (a torus through it)
    n = blob()
    n["tor"] = {"op": "expr", "expr": "sqrt((sqrt((x-1)**2 + (z-1)**2) - 6)**2 + (y-0.5)**2) - 2.5"}
    n["part"] = {"op": "subtract", "inputs": ["body", "tor"]}
    out["minus"] = ({"nodes": n, "output": "part", "bounds": BLOB_BOUNDS}, {}, [0.8, 0.4, 0.2])
    return out


# ---- the true crease -----------------------------------------------------------------

def true_crease(tree, mesh_fine, pitch_ref, spacing):
    """Dense points on the true crease curves: seeds from the label-changing
    edges of a fine plain marching-cubes mesh (independent of sharpen),
    Gauss-Newton onto both leaves' zero levels to 1e-9 mm, kept only where
    the part's boundary really is (the full field is zero there), thinned to
    `spacing` and densified by projecting midpoints between neighbours."""
    from scipy.spatial import cKDTree

    V = np.asarray(mesh_fine.vertices)
    F = np.asarray(mesh_fine.faces)
    _, lab = tree.evaluate(V)
    E, _ = _edges(F)
    c = lab[E[:, 0]] != lab[E[:, 1]]
    E = E[c]
    seeds = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    pairs = np.sort(np.stack([lab[E[:, 0]], lab[E[:, 1]]], 1), axis=1)
    X, ok = meet(tree, seeds, pairs, pitch_ref, iters=30, tol=1e-9 / pitch_ref)
    f, _ = tree.evaluate(X)
    ok &= np.abs(f) < 1e-7
    X, pairs = X[ok], pairs[ok]
    for _ in range(2):
        # thin, then densify by neighbour midpoints of the same pair
        keep = _thin(X, spacing)
        X, pairs = X[keep], pairs[keep]
        t = cKDTree(X)
        d, j = t.query(X, k=3)
        mids, mp = [], []
        for col in (1, 2):
            sel = (d[:, col] < 4 * spacing) & (pairs[:, 0] == pairs[j[:, col], 0]) & (pairs[:, 1] == pairs[j[:, col], 1])
            mids.append(0.5 * (X[sel] + X[j[sel, col]]))
            mp.append(pairs[sel])
        M = np.concatenate(mids)
        MP = np.concatenate(mp)
        M, okm = meet(tree, M, MP, pitch_ref, iters=30, tol=1e-9 / pitch_ref)
        fm, _ = tree.evaluate(M)
        okm &= np.abs(fm) < 1e-7
        X = np.concatenate([X, M[okm]])
        pairs = np.concatenate([pairs, MP[okm]])
    keep = _thin(X, spacing / 2)
    return X[keep], pairs[keep]


def _thin(X, r):
    from scipy.spatial import cKDTree

    key = np.floor(X / r).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return np.sort(idx)


def true_angle(tree, X, pairs, h=1e-5):
    """Angle (deg) between the outward normals of the two surfaces at X."""
    n = []
    for col in (0, 1):
        g = np.empty((len(X), 3))
        for a in range(3):
            e = np.zeros(3)
            e[a] = h
            vp = tree.values(X + e)[pairs[:, col], np.arange(len(X))]
            vm = tree.values(X - e)[pairs[:, col], np.arange(len(X))]
            g[:, a] = (vp - vm) / (2 * h)
        n.append(g / np.linalg.norm(g, axis=1, keepdims=True))
    return np.degrees(np.arccos(np.clip((n[0] * n[1]).sum(1), -1, 1)))


# ---- validity ----------------------------------------------------------------------

def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-300)[:, None], 0.5 * a


def min_angles(V, F):
    T = V[F]
    ang = []
    for i in range(3):
        u = T[:, (i + 1) % 3] - T[:, i]
        w = T[:, (i + 2) % 3] - T[:, i]
        c = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300)
        ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.min(ang, axis=0)


def topology(V, F):
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)
    n = len(V) + 1
    directed = e[:, 0].astype(np.int64) * n + e[:, 1]
    und = np.sort(e, 1)
    ukey = und[:, 0].astype(np.int64) * n + und[:, 1]
    _, cnt = np.unique(ukey, return_counts=True)
    _, dcnt = np.unique(directed, return_counts=True)
    watertight = bool((cnt == 2).all())
    winding = bool((dcnt == 1).all())
    try:
        import manifold3d as m3d

        mm = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(F, np.uint32)))
        m3 = str(mm.status()).split(".")[-1]
    except Exception as exc:  # noqa: BLE001
        m3 = f"error: {exc}"
    return {"watertight": watertight, "winding": winding, "manifold3d": m3,
            "badEdges": int((cnt != 2).sum())}


def nonmanifold_vertices(V, F, which):
    """Vertices in `which` whose faces do not form one fan (a pinch)."""
    from collections import defaultdict

    vf = defaultdict(list)
    sel = np.zeros(len(V), bool)
    sel[which] = True
    for fi, f in enumerate(F):
        for k in range(3):
            if sel[f[k]]:
                vf[f[k]].append((f[(k + 1) % 3], f[(k + 2) % 3]))
    bad = 0
    for v, link in vf.items():
        nxt = {}
        for a, b in link:
            nxt.setdefault(a, []).append(b)
        if any(len(x) != 1 for x in nxt.values()):
            bad += 1
            continue
        start = link[0][0]
        cur, steps = start, 0
        while True:
            cur = nxt.get(cur, [None])[0]
            steps += 1
            if cur is None or cur == start or steps > len(link) + 1:
                break
        if cur != start or steps != len(link):
            bad += 1
    return bad


def self_intersections(V, F, faces, eps=1e-9, pairs_out=None):
    """Pairs of the `faces` (indices) and their neighbours that cross
    without sharing a vertex (edge-through-triangle test both ways)."""
    from scipy.spatial import cKDTree

    if not len(faces):
        return 0
    C = V[F].mean(1)
    L = np.linalg.norm(V[F] - C[:, None], axis=2).max()
    tree = cKDTree(C)
    pairs = tree.query_ball_point(C[faces], 2 * L + 1e-9)
    A, B = [], []
    fset = set(np.asarray(faces).tolist())
    for i, lst in zip(faces, pairs):
        for j in lst:
            if j != i and (j > i or j not in fset):
                A.append(i)
                B.append(j)
    A, B = np.asarray(A, np.int64), np.asarray(B, np.int64)
    if not len(A):
        return 0
    share = (F[A][:, :, None] == F[B][:, None, :]).any(axis=(1, 2))
    A, B = A[~share], B[~share]
    hit = np.zeros(len(A), bool)
    for X, Y in ((A, B), (B, A)):
        T = V[F[Y]]
        for k in range(3):
            p = V[F[X, k]]
            q = V[F[X, (k + 1) % 3]]
            hit |= _seg_tri(p, q, T, eps)
    if pairs_out is not None:
        pairs_out.append(np.stack([A[hit], B[hit]], 1))
    return int(hit.sum())


def _seg_tri(p, q, T, eps):
    d = q - p
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p - T[:, 0]
    u = f * (s * h).sum(1)
    qq = np.cross(s, e1)
    v = f * (d * qq).sum(1)
    t = f * (e2 * qq).sum(1)
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


# ---- the measurements ----------------------------------------------------------------

def measure(V, F, tree, X, pairs, phi_true, pitch, crease_v=None):
    import trimesh
    from scipy.spatial import cKDTree

    from loadpath.sharp import mesh_distance

    m = trimesh.Trimesh(V, F, process=False)
    d = mesh_distance(m, X, edge=max(pitch, 0.2))
    vals = tree.values(V)
    _, lab = tree.combine(vals)
    E, fe = _edges(F)
    if crease_v is None:
        c = lab[E[:, 0]] != lab[E[:, 1]]
        idx = np.unique(E[c].ravel())
        partner = np.empty(len(V), np.int64)
        partner[E[c, 0]] = lab[E[c, 1]]
        partner[E[c, 1]] = lab[E[c, 0]]
        which = np.stack([lab[idx], partner[idx]], 1)
    else:
        idx = np.flatnonzero(crease_v)
        # its two leaves: the two smallest |value|
        order = np.argsort(np.abs(vals[:, idx]), axis=0)[:2].T
        which = order
    Xv, okv = meet(tree, V[idx], which, pitch, iters=30, tol=1e-9 / pitch)
    dv = np.linalg.norm(Xv - V[idx], axis=1)[okv]
    # dihedral: sharpest edge near each crease sample
    N, area = face_normals(V, F)
    # faces of each edge
    ef = np.full((len(E), 2), -1, np.int64)
    flat = fe.ravel()
    fid = np.repeat(np.arange(len(F)), 3)
    o = np.argsort(flat, kind="stable")
    fs, fl = fid[o], flat[o]
    first = np.r_[True, fl[1:] != fl[:-1]]
    ef[fl[first], 0] = fs[first]
    ef[fl[~first], 1] = fs[~first]
    good = (ef >= 0).all(1)
    dih = np.zeros(len(E))
    dih[good] = np.degrees(np.arccos(np.clip((N[ef[good, 0]] * N[ef[good, 1]]).sum(1), -1, 1)))
    mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    et = cKDTree(mid)
    near = et.query_ball_point(X, pitch)
    phi_mesh = np.array([dih[n].max() if len(n) else 0.0 for n in near])
    # validity near the crease
    ct = cKDTree(X)
    dc, _ = ct.query(V[F].mean(1), distance_upper_bound=2 * pitch)
    region = np.flatnonzero(np.isfinite(dc))
    cen = V[F[region]].mean(1)
    cv, clab = tree.evaluate(cen)
    # true normal of the active leaf at the centroid
    gn = np.empty((len(region), 3))
    h = 1e-5
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        gn[:, a] = (tree.values(cen + e)[clab, np.arange(len(region))]
                    - tree.values(cen - e)[clab, np.arange(len(region))]) / (2 * h)
    folds = int(((N[region] * gn).sum(1) < 0).sum())
    ma = min_angles(V, F)
    topo = topology(V, F)
    rv = np.unique(F[region].ravel())
    out = {
        "faces": int(len(F)), "verts": int(len(V)),
        "creaseToMesh": {"max": float(d.max()), "rms": float(np.sqrt((d ** 2).mean())),
                         "p95": float(np.percentile(d, 95)), "maxPitch": float(d.max() / pitch),
                         "rmsPitch": float(np.sqrt((d ** 2).mean()) / pitch)},
        "vertexToCrease": {"n": int(len(dv)), "max": float(dv.max()) if len(dv) else None,
                           "rms": float(np.sqrt((dv ** 2).mean())) if len(dv) else None,
                           "unconverged": int((~okv).sum())},
        "dihedral": {"true": float(np.median(phi_true)), "median": float(np.median(phi_mesh)),
                     "p10": float(np.percentile(phi_mesh, 10)),
                     "medAbsErr": float(np.median(np.abs(phi_mesh - phi_true)))},
        "validity": {**topo, "folds": folds, "regionFaces": int(len(region)),
                     "nonmanifoldVerts": nonmanifold_vertices(V, F, rv),
                     "selfIntersections": self_intersections(V, F, region),
                     "minAngle": float(ma.min()), "slivers5": int((ma < 5).sum()), "slivers10": int((ma < 10).sum()),
                     "slivers5Region": int((ma[region] < 5).sum()), "degenerate": int((area < 1e-12).sum())},
    }
    return out


AWARE = False
REACH = 2.0


def run_case(name, doc, meta, pitches, snap=0.25, render=None):
    g = FieldGraph(doc)
    tree = CreaseTree.from_graph(g)
    print(f"[{name}] leaves: {[lf.name for lf in tree.leaves]}", flush=True)
    pref = min(pitches) / 1.5
    t = time.time()
    fine = g.mesh(pitch=pref, sharp=False)
    X, pairs = true_crease(tree, fine, pref, spacing=min(pitches) / 4)
    del fine
    phi = true_angle(tree, X, pairs)
    ref = {"points": int(len(X)), "refPitch": pref, "seconds": round(time.time() - t, 1)}
    if "R" in meta:
        # analytic check: the circle of two spheres (centres at -d/2,0,0 and d/2,.3,.2)
        R = meta["R"]
        c1 = np.array([-meta["d"] / 2, 0, 0])
        ref["sphereResidual"] = float(np.abs(np.linalg.norm(X - c1, axis=1) - R).max())
    print(f"[{name}] true crease: {ref}", flush=True)
    rows = []
    for p in pitches:
        t0 = time.time()
        mesh = g.mesh(pitch=p, sharp=False)
        t_mc = time.time() - t0
        V0, F0 = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        st = {}
        t1 = time.time()
        V1, F1, cr = sharpen(V0, F0, tree, p, snap=snap, reach=REACH, stats=st)
        t_sh = time.time() - t1
        # B: crease-aware projection + split + label-preserving relax, from raw MC
        before = measure(V0, F0, tree, X, pairs, phi, p)
        after = measure(V1, F1, tree, X, pairs, phi, p, crease_v=cr)
        t_raw = t_fin = 0.0
        stB, aware, V2, F2, cr2 = {}, None, V1, F1, cr
        if AWARE:
            t2 = time.time()
            raw = g.mesh(pitch=p, sharp=False, project=0)
            t_raw = time.time() - t2
            t3 = time.time()
            V2, F2, cr2 = finish(np.asarray(raw.vertices), np.asarray(raw.faces), tree, p, rounds=3, snap=snap,
                                 stats=stB)
            t_fin = time.time() - t3
            aware = measure(V2, F2, tree, X, pairs, phi, p, crease_v=cr2)
        row = {"case": name, "pitch": p, "meshSeconds": round(t_mc, 2), "sharpenSeconds": round(t_sh, 2),
               "rawMcSeconds": round(t_raw, 2), "awareSeconds": round(t_fin, 2),
               "stats": st, "statsAware": stB, "before": before, "after": after, "aware": aware}
        rows.append(row)
        for tag, r in (("before", before), ("sharpen", after), ("aware", aware)):
            if r is None:
                continue
            c, dd, v = r["creaseToMesh"], r["dihedral"], r["validity"]
            print(f"[{name}] {p} {tag:8s} c->m max {c['max']:.4f} rms {c['rms']:.4f} | dih {dd['median']:.1f}/{dd['true']:.1f}"
                  f" | wt {v['watertight']} m3d {v['manifold3d']} folds {v['folds']} selfX {v['selfIntersections']}"
                  f" sliv5 {v['slivers5Region']} minAng {v['minAngle']:.2f}", flush=True)
        print(f"      times: mesh {t_mc:.2f}s  sharpen {t_sh:.2f}s  rawMC {t_raw:.2f}s  aware {t_fin:.2f}s  {st} {stB}", flush=True)
        if render is not None and p in render:
            np.savez_compressed(HERE / f"mesh_{name}_{p}.npz", V0=V0, F0=F0, V1=V1, F1=F1, cr=cr, V2=V2, F2=F2, cr2=cr2, X=X)
    return {"case": name, "reference": ref, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--snap", type=float, default=0.25)
    ap.add_argument("--reach", type=float, default=2.0)
    ap.add_argument("--aware", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    global AWARE, REACH
    AWARE, REACH = a.aware, a.reach
    allc = cases()
    names = a.case or list(allc)
    res = []
    for n in names:
        doc, meta, pitches = allc[n]
        if a.quick:
            pitches = pitches[:1]
        res.append(run_case(n, doc, meta, pitches, snap=a.snap, render={pitches[0], pitches[1] if len(pitches) > 1 else pitches[0]}))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
