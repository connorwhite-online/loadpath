"""Physical validity of a printed part: one solid, every feature fused to what
it stands on, no slivers, no unfilleted re-entrant creases.

Every analysis in loadpath (FE, airflow, retention, clearance) assumes the
part is one piece of material and that what is drawn as a boss or a cradle is
held by the wall it stands on. A field that unions a feature placed on a plane
the wall no longer reaches, or that deletes the web a cradle hung from, still
marches into a watertight mesh: watertight is a property of each closed
surface, so seven loose bodies pass it. This module checks the things that
matter instead, and fails rather than warns.

- ``bodies`` / ``require_one_body``: the connected pieces of a mesh.
- ``attachment``: each declared Feature against what it attaches to, from the
  part's fields: how much of the feature's underside is fused (cover) and how
  much area it is fused over (contact). A ring that touches the wall over 86%
  of its circle and floats 1 mm off it over the rest is one body, but fails here.
- ``slivers``: gaps and walls under ``gap`` mm (two faces of the part facing each
  other across air, or across material, that close).
- ``creases``: re-entrant edges sharper than ``crease_angle`` (a root with no fillet).
- ``seated``: correct by construction, a feature swept onto its host so its
  root is fused wherever the host is within reach (union it with ridges.fillet_union).
- ``gate`` / ``export``: all of the above, the last step before a file leaves
  the tool. Writes section cuts through every feature root and a map of the
  bodies, each in its own colour, and a stamp beside the file. ``require_passed``
  is what renders and uploads call: no stamp, a stale stamp or a failed stamp
  raises.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

__all__ = ["IntegrityError", "Feature", "bodies", "require_one_body", "require_one_body_mask", "slivers",
           "creases", "attachment", "seated", "gate", "export", "require_passed", "stamp_path", "body_map",
           "GateReport"]

STAMP_VERSION = 1


class IntegrityError(RuntimeError):
    """The geometry is not a physically valid part. Nothing downstream should run on it."""


# -- bodies ----------------------------------------------------------------

def _tm(mesh):
    import trimesh
    if isinstance(mesh, (str, os.PathLike)):
        return trimesh.load(mesh, force="mesh", process=True)
    if isinstance(mesh, tuple):
        return trimesh.Trimesh(*mesh, process=True)
    return mesh


def _face_labels(m):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adj = m.face_adjacency
    n = len(m.faces)
    A = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    return connected_components(A, directed=False)


def bodies(mesh) -> list[dict]:
    """The mesh's connected pieces, largest first: volume (mm^3), bounds, faces.
    A shell's inner and outer surface are one piece only where material joins
    them; a hollow ball with no opening is two surfaces of one solid, so pieces
    that lie inside another piece's volume and face inward are counted with it."""
    m = _tm(mesh)
    nc, lab = _face_labels(m)
    out = []
    for c in range(nc):
        f = np.nonzero(lab == c)[0]
        sub = m.submesh([f], append=True, repair=False)
        out.append(dict(volume=float(sub.volume), bounds=sub.bounds.round(2).tolist(), faces=int(len(f)),
                        centre=sub.bounds.mean(0).round(2).tolist(), _faces=f))
    # a void's surface (negative volume, inside another piece) belongs to that piece
    solids = [b for b in out if b["volume"] > 0]
    voids = [b for b in out if b["volume"] <= 0]
    for v in voids:
        lo, hi = np.array(v["bounds"])
        host = [s for s in solids if np.all(np.array(s["bounds"][0]) <= lo) and np.all(np.array(s["bounds"][1]) >= hi)]
        if host:
            h = min(host, key=lambda s: s["volume"])
            h["volume"] += v["volume"]
            h["_faces"] = np.concatenate([h["_faces"], v["_faces"]])
            h["faces"] += v["faces"]
        else:
            solids.append(v)
    solids = _merge_overlapping(m, solids)
    solids.sort(key=lambda b: -abs(b["volume"]))
    return solids


def _merge_overlapping(m, parts):
    """Closed pieces that overlap are one solid (primitives concatenated into
    one file: a cylinder and the lip it passes through). A piece is merged
    into another when some of its vertices lie inside that piece."""
    if len(parts) < 2 or len(parts) > 400:
        return parts
    subs = [m.submesh([b["_faces"]], append=True, repair=False) for b in parts]
    parent = list(range(len(parts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(parts)):
        for j in range(len(parts)):
            if i == j or find(i) == find(j):
                continue
            a, b = subs[i], subs[j]
            if np.any(a.bounds[0] > b.bounds[1]) or np.any(b.bounds[0] > a.bounds[1]):
                continue
            if not b.is_watertight:
                continue
            v = a.vertices
            if len(v) > 4000:
                v = v[np.random.default_rng(0).choice(len(v), 4000, replace=False)]
            if b.contains(v).any():
                parent[find(i)] = find(j)
    groups = {}
    for i, b in enumerate(parts):
        groups.setdefault(find(i), []).append(b)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        f = np.concatenate([b["_faces"] for b in g])
        sub = m.submesh([f], append=True, repair=False)
        out.append(dict(volume=float(sum(b["volume"] for b in g)), bounds=sub.bounds.round(2).tolist(),
                        faces=int(len(f)), centre=sub.bounds.mean(0).round(2).tolist(), _faces=f, merged=len(g)))
    return out


COUNTED = None   # allow_bodies for a caller that counts the pieces itself and fails the part on them


def require_one_body(mesh, what: str = "part", allow: int | None = 1):
    """Raise IntegrityError unless the mesh is at most `allow` solid pieces
    (allow > 1 only for a file that is an assembly on purpose). `allow=COUNTED`
    is for a pipeline that counts the pieces itself and fails the part on them
    in its own report (the study's outcomes: report["bodies"], "the part came
    out in N pieces"); it never raises then, and nothing else may pass it."""
    if allow is COUNTED:
        return None
    b = bodies(mesh)
    if len(b) > allow:
        lines = [f"  {i}: {abs(x['volume']):.1f} mm^3 at {x['centre']}" for i, x in enumerate(b[:12])]
        raise IntegrityError(f"{what} is {len(b)} separate bodies, not one:\n" + "\n".join(lines)
                             + ("\n  ..." if len(b) > 12 else ""))
    return b


def require_one_body_mask(mask, what: str = "solid", min_cells: int = 1):
    """The voxel form: raise unless the True cells are one 6-connected piece
    (pieces under `min_cells` cells are counted too unless min_cells > 1)."""
    from scipy.ndimage import label
    lab, n = label(np.asarray(mask, bool))
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        big = int((sizes >= min_cells).sum())
        if big > 1 or (min_cells <= 1 and n > 1):
            raise IntegrityError(f"{what} is {n} separate pieces of material (cells per piece: "
                                 f"{sorted(sizes.tolist(), reverse=True)[:8]})")
    return n


# -- slivers and creases ------------------------------------------------------

def _rays(m):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector(m)
    except Exception:  # pragma: no cover - embree is an extra
        from trimesh.ray.ray_triangle import RayMeshIntersector
        return RayMeshIntersector(m)


def _clusters(m, sel, min_area, extra=None):
    """Face clusters of the selected faces (edge-adjacent), by area."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = np.nonzero(sel)[0]
    if not len(F):
        return []
    adj = m.face_adjacency
    k = sel[adj[:, 0]] & sel[adj[:, 1]]
    e = adj[k]
    n = len(m.faces)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    out = []
    c = m.triangles_center
    area = m.area_faces
    for L in np.unique(lab[F]):
        f = F[lab[F] == L]
        a = float(area[f].sum())
        if a < min_area:
            continue
        r = dict(area=round(a, 2), centre=c[f].mean(0).round(2).tolist(), lo=c[f].min(0).round(2).tolist(),
                 hi=c[f].max(0).round(2).tolist())
        if extra is not None:
            r.update(extra(f))
        out.append(r)
    out.sort(key=lambda r: -r["area"])
    return out


def slivers(mesh, gap: float = 0.4, min_area: float = 0.3) -> dict:
    """Air gaps and walls thinner than `gap` mm inside one part.

    A face's outward ray that meets a face looking back at it within `gap` is
    across an air gap (a slot, a crevice, a feature hovering over the wall); its
    inward ray leaving the material within `gap` is across a sliver of material.
    Clusters under `min_area` mm^2 are left out (a marching-cubes corner)."""
    m = _tm(mesh)
    c = m.triangles_center
    n = m.face_normals
    ri = _rays(m)
    res = {}
    for kind, sgn in (("air", 1.0), ("material", -1.0)):
        o = c + sgn * n * 1e-3
        d = sgn * n
        it, ir, loc = ri.intersects_id(o, d, multiple_hits=False, return_locations=True)
        dist = np.full(len(c), np.inf)
        dist[ir] = np.linalg.norm(loc - o[ir], axis=1)
        facing = np.zeros(len(c), bool)
        facing[ir] = (n[it] * n[ir]).sum(1) < -0.5
        sel = (dist < gap) & facing
        res[kind] = _clusters(m, sel, min_area, extra=lambda f: dict(
            gap_min=round(float(dist[f].min()), 3), gap_median=round(float(np.median(dist[f])), 3)))
    return res


def creases(mesh, angle: float = 50.0, min_length: float = 1.0) -> list[dict]:
    """Re-entrant edges where the surface turns more than `angle` degrees across
    one edge: an inside corner with no fillet (a fillet of radius r turns about
    edge/r per edge, so a 0.4 mm fillet at a 0.25 mm mesh is ~35 degrees).
    Chains of such edges shorter than `min_length` mm are left out."""
    m = _tm(mesh)
    ang = np.degrees(m.face_adjacency_angles)
    concave = ~m.face_adjacency_convex
    sel = concave & (ang > angle)
    if not sel.any():
        return []
    E = m.face_adjacency_edges[sel]
    V = m.vertices
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = len(V)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nv, nv))
    _, lab = connected_components(A, directed=False)
    el = lab[E[:, 0]]
    out = []
    for c in np.unique(el):
        k = el == c
        length = float(L[k].sum())
        if length < min_length:
            continue
        P = V[E[k].ravel()]
        out.append(dict(length=round(length, 2), centre=P.mean(0).round(2).tolist(), lo=P.min(0).round(2).tolist(),
                        hi=P.max(0).round(2).tolist(), angle_max=round(float(ang[sel][k].max()), 1)))
    out.sort(key=lambda r: -r["length"])
    return out


# -- features and what they attach to ------------------------------------------

@dataclass
class Feature:
    """A feature of a part, declared with what holds it.

    name: the key of its field in the part's fields (negative inside, as
    designed: before the part's cuts; the check intersects it with the part).
    attach: the key, or keys, of the field(s) it must be fused to (the skin, or
    another feature). Each connected piece of the feature is checked alone.
    min_contact: mm^2 of the piece's surface that must be fused into its host.
    min_cover: the share of its underside (its surface facing the host within
    `reach` mm) that must be fused rather than across a gap.
    reach: how far from the host a face of the feature counts as its underside;
    a piece with no face within reach is floating. A feature that spans or
    overhangs on purpose (a cradle on a web, a bridge) takes min_cover=0: only
    its contact counts.
    open_angle: an underside that opens from the fused root at this angle or
    steeper (degrees off the host) is an overhang, not a gap: a rib leaned 45
    degrees has air under it, but nothing hovers. Shallower is a crevice (a ring
    lifting off a curved face over its circumference opens at a few degrees, a
    shelf parallel to the wall at none) and counts against min_cover."""
    name: str
    attach: str | tuple = "skin"
    min_contact: float = 3.0
    min_cover: float = 0.97
    reach: float = 4.0
    open_angle: float = 30.0
    min_volume: float = 0.5
    note: str = ""


def _grid_points(lo, hi, p):
    axes = [np.arange(lo[i] + 0.5 * p, hi[i], p) for i in range(3)]
    return axes


def _eval(fields, P, chunk=400_000):
    out = {}
    for s in range(0, len(P), chunk):
        d = fields(P[s:s + chunk])
        for k, v in d.items():
            out.setdefault(k, []).append(np.asarray(v, float))
    return {k: np.concatenate(v) for k, v in out.items()}


def _host(vals, attach):
    keys = (attach,) if isinstance(attach, str) else tuple(attach)
    return np.minimum.reduce([vals[k] for k in keys])


def attachment(fields, features, box, solid: str = "solid", pitch: float = 0.25, coarse: float = 1.0,
               verbose: bool = False) -> list[dict]:
    """Check each piece of each feature against what it attaches to.

    fields(P) -> {key: values}: the part's fields at points P (n, 3), with
    `solid` the finished part and a key for every feature and host. box: (lo,
    hi) around the part. The pieces are found on a `coarse` grid; each piece's
    surface is then sampled at `pitch` near its host: area fused into the host
    (contact), and its underside (faces looking at the host, within reach)
    fused or across a gap (cover, max gap)."""
    from scipy.ndimage import label, binary_dilation
    lo, hi = (np.asarray(b, float) for b in box)
    ax = _grid_points(lo, hi, coarse)
    shape = tuple(len(a) for a in ax)
    names = [f.name for f in features]
    hosts = sorted({k for f in features for k in ((f.attach,) if isinstance(f.attach, str) else f.attach)})
    C = {k: np.empty(shape, np.float32) for k in set(names) | set(hosts) | {solid}}
    YZ = np.stack(np.meshgrid(ax[1], ax[2], indexing="ij"), -1).reshape(-1, 2)
    for i, x in enumerate(ax[0]):
        P = np.column_stack([np.full(len(YZ), x), YZ])
        v = _eval(fields, P)
        for k in C:
            C[k][i] = v[k].reshape(shape[1:])
    org = np.array([a[0] for a in ax])
    results = []
    sub = int(round(coarse / pitch))
    offs = (np.stack(np.meshgrid(*[np.arange(sub)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) * pitch - 0.5 * coarse
    for f in features:
        built = np.maximum(C[f.name], C[solid])
        lab, n = label(built < 0.9 * coarse, structure=np.ones((3, 3, 3)))
        if n == 0:
            results.append(dict(feature=f.name, piece=-1, passed=False, reasons=["absent: no material of it in the part"]))
            continue
        hostc = np.minimum.reduce([C[k] for k in ((f.attach,) if isinstance(f.attach, str) else f.attach)])
        for j in range(1, n + 1):
            pm = lab == j
            vol = float(pm.sum() * coarse ** 3)
            if vol < f.min_volume:
                continue
            idx = np.argwhere(pm)
            centre = (org + idx.mean(0) * coarse).round(2).tolist()
            # the band to sample: near this piece's surface and within reach of the host
            band = binary_dilation(pm, iterations=2) & (np.abs(built) < 2.0 * coarse + pitch) & (hostc < f.reach + 2 * coarse)
            cells = np.argwhere(band)
            if not len(cells):
                results.append(dict(feature=f.name, attach=f.attach, piece=j, centre=centre, volume=round(vol, 1), contact=0.0,
                                    cover=0.0, gap_max=None, passed=False,
                                    reasons=[f"floating: no face within {f.reach} mm of {f.attach}"]))
                continue
            P = (org + cells * coarse)[:, None, :] + offs[None]
            P = P.reshape(-1, 3)
            v = _eval(fields, P)
            fb = np.maximum(v[f.name], v[solid])
            h = _host(v, f.attach)
            shell = (fb > -pitch) & (fb <= 0.0)             # the piece's surface, one sample deep
            # only the samples nearest this piece (its neighbours of the same feature are their own pieces)
            P_s, fb_s, h_s, sv_s = P[shell], fb[shell], h[shell], v[solid][shell]
            near = np.zeros(len(P_s), bool)
            if len(P_s):
                ijk = np.clip(np.floor((P_s - org) / coarse + 0.5).astype(int), 0, np.array(shape) - 1)
                near = binary_dilation(pm, iterations=1)[tuple(ijk.T)]
            P_s, fb_s, h_s, sv_s = P_s[near], fb_s[near], h_s[near], sv_s[near]
            seat = f.attach if isinstance(f.attach, str) else f.attach[0]
            buried = sv_s < -0.5 * pitch                          # the feature's face is inside the part here
            # on or in the host: inside it, or within a sample of it with material between (half way
            # to the host is solid). Within a sample is not enough: a hairline gap one pitch wide
            # (a lip an edge round pulled 0.25 mm off its wall) read as fused
            direct = h_s <= 0.0
            near_ = (h_s > 0.0) & (h_s <= pitch) & ~buried
            if near_.any():
                Q = P_s[near_]
                e = 0.5 * pitch
                g = np.zeros_like(Q)
                for a_ in range(3):
                    d = np.zeros(3)
                    d[a_] = e
                    g[:, a_] = _host(_eval(fields, Q + d), f.attach) - _host(_eval(fields, Q - d), f.attach)
                n_ = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
                mid = _eval(fields, Q - 0.5 * h_s[near_, None] * n_)
                direct[np.nonzero(near_)[0]] = mid[solid] < 0.0
            fused = direct.copy()
            # a face buried in material that runs on to the host (a root fillet, a web): walk from it
            # toward the host; it is fused if the part stays solid all the way to the host
            walk = buried & ~direct & (h_s < f.reach)
            if walk.any():
                ix = np.nonzero(walk)[0]
                if len(ix) > 20000:
                    ix = np.sort(np.random.default_rng(0).choice(ix, 20000, replace=False))
                Q = P_s[ix]
                e = 0.5 * pitch
                g = np.zeros_like(Q)
                for a_ in range(3):
                    d = np.zeros(3)
                    d[a_] = e
                    g[:, a_] = _host(_eval(fields, Q + d), f.attach) - _host(_eval(fields, Q - d), f.attach)
                n_ = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
                ok = np.ones(len(Q), bool)
                done = np.zeros(len(Q), bool)
                for k in range(1, int(np.ceil(f.reach / (0.5 * pitch))) + 1):
                    live = ok & ~done
                    if not live.any():
                        break
                    v = _eval(fields, Q[live] - (k * 0.5 * pitch) * n_[live])
                    li = np.nonzero(live)[0]
                    reached = _host(v, f.attach) <= 0.0
                    solid_ = v[solid] < 0.0
                    done[li[reached & solid_]] = True
                    ok[li[~solid_ & ~reached]] = False
                fused[ix[done]] = True
            contact = float(fused.sum() * pitch ** 2)
            # its underside: faces on the part's surface that look at the seat host across air
            cand = ~fused & ~buried & (h_s > pitch) & (h_s < f.reach)
            gapfacing = np.zeros(len(P_s), bool)
            if cand.any():
                Q = P_s[cand]
                e = 0.5 * pitch
                g_f, g_h = np.zeros_like(Q), np.zeros_like(Q)
                hs = np.zeros(len(Q))
                for a_ in range(3):
                    d = np.zeros(3)
                    d[a_] = e
                    vp, vm = _eval(fields, Q + d), _eval(fields, Q - d)
                    g_f[:, a_] = np.maximum(vp[f.name], vp[solid]) - np.maximum(vm[f.name], vm[solid])
                    g_h[:, a_] = vp[seat] - vm[seat]
                    hs += 0.5 * (vp[seat] + vm[seat]) / 3.0
                nf = g_f / (np.linalg.norm(g_f, axis=1, keepdims=True) + 1e-12)
                nh = g_h / (np.linalg.norm(g_h, axis=1, keepdims=True) + 1e-12)
                looks = ((nf * nh).sum(1) < -0.5) & (hs < f.reach)   # the face looks at its seat
                # an overhang opens from the root at open_angle or steeper: its gap is at least
                # tan(open_angle) times its distance from the nearest fused face. A crevice is shallower
                if looks.any() and fused.any() and f.open_angle < 90.0:
                    from scipy.spatial import cKDTree
                    d_root, _ = cKDTree(P_s[fused]).query(Q[looks])
                    steep = hs[looks] >= np.tan(np.radians(f.open_angle)) * d_root - pitch
                    li = np.nonzero(looks)[0]
                    looks[li[steep]] = False
                gapfacing[np.nonzero(cand)[0]] = looks
            fused_under = fused.sum()
            gap_under = gapfacing.sum()
            under = fused_under + gap_under
            cover = float(fused_under / under) if under else 0.0
            gmax = float(h_s[gapfacing].max()) if gap_under else 0.0
            reasons = []
            if contact < f.min_contact:
                reasons.append(f"fused over {contact:.1f} mm^2 < {f.min_contact} to {f.attach}"
                               + (" (floating)" if contact == 0 else ""))
            if under and f.min_cover > 0 and cover < f.min_cover:
                reasons.append(f"{100 * (1 - cover):.0f}% of its underside is off {seat} (gap up to {gmax:.2f} mm)")
            if not under and contact == 0:
                reasons.append(f"floating: no face within {f.reach} mm of {f.attach}")
            gp = P_s[gapfacing]
            r = dict(feature=f.name, attach=f.attach, piece=j, centre=centre, volume=round(vol, 1),
                     contact=round(contact, 1), cover=round(cover, 3), gap_max=round(gmax, 2),
                     gap_area=round(float(gap_under * pitch ** 2), 1), passed=not reasons, reasons=reasons,
                     gap_at=gp[np.argmax(h_s[gapfacing])].round(2).tolist() if len(gp) else None,
                     root_at=P_s[fused][np.argmin(np.linalg.norm(P_s[fused] - P_s[fused].mean(0), axis=1))].round(2).tolist()
                     if fused.any() else None,
                     lo=(org + idx.min(0) * coarse).round(1).tolist(), hi=(org + idx.max(0) * coarse).round(1).tolist())
            if verbose:
                print(f"  {f.name}#{j}: {'ok' if r['passed'] else 'FAIL ' + '; '.join(reasons)}", flush=True)
            results.append(r)
    return results


# -- correct by construction ------------------------------------------------------

def seated(feature, host, direction, reach: float = 4.0, step: float = 0.1):
    """A feature carried onto its host: its own field, plus the material swept
    from it along `direction` (toward the host) for up to `reach` mm, kept only
    within reach of the host. Where the host is nearer than `reach` the
    feature's root is solid to it; union the result with the host by
    ridges.fillet_union for the root fillet. Placing a feature on a fixed plane
    and hoping the surface is there is what left rings 1-4 mm off a surfaced face.
    The sweep is sampled every `step` mm, so a face slanted to `direction`
    comes out scalloped that finely; for a prism along `direction`, lengthen
    the prism instead (exact)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    ts = np.arange(0.0, reach + 1e-9, step)

    def f(P):
        P = np.asarray(P, float)
        own = feature(P)
        sw = own.copy()
        for t in ts[1:]:
            sw = np.minimum(sw, feature(P - t * d))
        return np.minimum(own, np.maximum(sw, host(P) - reach))
    return f


# -- pictures ---------------------------------------------------------------

_COLOURS = np.array([[0.62, 0.63, 0.66], [0.90, 0.20, 0.15], [0.15, 0.45, 0.90], [0.10, 0.65, 0.30],
                     [0.95, 0.60, 0.10], [0.60, 0.25, 0.80], [0.05, 0.70, 0.75], [0.85, 0.30, 0.60],
                     [0.55, 0.45, 0.15], [0.30, 0.30, 0.30]])


def body_map(mesh, path, views=("+x", "-x", "+y", "-y", "+z", "-z"), px: float | None = None, title: str = ""):
    """Each body in its own colour (the largest grey, the rest red, blue, ...)
    from six sides, as one picture. Software z-buffer: no GL needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = _tm(mesh)
    bs = bodies(m)
    col = np.zeros(len(m.faces), int)
    for i, b in enumerate(bs):
        col[b["_faces"]] = min(i, len(_COLOURS) - 1)
    nrm = m.face_normals
    ext = m.bounds[1] - m.bounds[0]
    T = m.triangles
    if px is None:
        px = max(ext.max() / 900.0, 0.6 * float(np.median(m.edges_unique_length)))
    # each face splatted at its corners, edge midpoints and centre: about a pixel apart at this px
    bary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3]])
    pts = np.einsum("kj,fjd->fkd", bary, T).reshape(-1, 3)
    fid = np.repeat(np.arange(len(T)), len(bary))
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for ax, v in zip(axs.flat, views):
        a = "xyz".index(v[1])
        s = 1.0 if v[0] == "+" else -1.0
        u_, w_ = [i for i in range(3) if i != a]
        W, H = int(ext[u_] / px) + 3, int(ext[w_] / px) + 3
        iu = np.clip(((pts[:, u_] - m.bounds[0][u_]) / px).astype(int), 0, W - 1)
        iw = np.clip(((pts[:, w_] - m.bounds[0][w_]) / px).astype(int), 0, H - 1)
        near = s * pts[:, a]                          # larger is nearer the viewer
        pix = iw * W + iu
        o = np.lexsort((-near, pix))                  # per pixel, nearest first
        first = np.ones(len(o), bool)
        first[1:] = pix[o][1:] != pix[o][:-1]
        sel = o[first]
        shade = np.abs(nrm[fid[sel], a]) * 0.6 + 0.4
        img = np.ones((H * W, 3)) * 0.97
        img[pix[sel]] = _COLOURS[col[fid[sel]]] * shade[:, None]
        img = img.reshape(H, W, 3)[::-1]
        if s < 0:
            img = img[:, ::-1]                        # seen from the other side
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"from {v}", fontsize=9)
        ax.axis("off")
    names = ", ".join(f"{['grey', 'red', 'blue', 'green', 'orange', 'purple', 'teal', 'pink', 'olive', 'dark'][min(i, 9)]} "
                      f"{abs(b['volume']):.0f} mm3" for i, b in enumerate(bs[:10]))
    fig.suptitle(f"{title}  {len(bs)} bod{'y' if len(bs) == 1 else 'ies'}: {names}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return len(bs)


def _section_png(m, fields, r, path, solid, half=8.0, res=0.1):
    """Two cuts through a feature piece's root (or its gap): planes square to
    each other, both containing the host's normal there. Black: the part as
    meshed. Red: the feature's outline; blue: its host's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    at = np.array(r.get("gap_at") or r.get("root_at") or r["centre"], float)
    # host normal by central differences
    e = 0.2
    g = np.zeros(3)
    for a in range(3):
        d = np.zeros(3)
        d[a] = e
        vp, vm = fields(np.array([at + d])), fields(np.array([at - d]))
        g[a] = _host(vp, r["attach"])[0] - _host(vm, r["attach"])[0]
    nrm = g / (np.linalg.norm(g) + 1e-12)
    t1 = np.cross(nrm, [0, 0, 1.0]) if abs(nrm[2]) < 0.9 else np.cross(nrm, [1.0, 0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(nrm, t1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    u = np.arange(-half, half + 1e-9, res)
    UU, NN = np.meshgrid(u, u, indexing="xy")
    for ax, t in zip(axs, (t1, t2)):
        P = at + UU.ravel()[:, None] * t + NN.ravel()[:, None] * nrm
        v = _eval(fields, P)
        S = v[solid].reshape(UU.shape)
        Fv = np.maximum(v[r["feature"]], v[solid]).reshape(UU.shape)
        Hv = _host(v, r["attach"]).reshape(UU.shape)
        ax.contourf(UU, NN, S, levels=[-1e9, 0], colors=["#d9d9d9"])
        ax.contour(UU, NN, S, levels=[0], colors="k", linewidths=1.0)
        ax.contour(UU, NN, Fv, levels=[0], colors="#d62728", linewidths=1.2, linestyles="--")
        ax.contour(UU, NN, Hv, levels=[0], colors="#1f77b4", linewidths=1.0, linestyles=":")
        ax.set_aspect("equal")
        ax.set_xlabel("mm along the wall")
        ax.set_ylabel("mm off the wall")
        ax.set_title(f"plane through {at.round(1).tolist()}, along {t.round(2).tolist()}", fontsize=7)
    verdict = "PASS" if r["passed"] else "FAIL: " + "; ".join(r["reasons"])
    fig.suptitle(f"{r['feature']} #{r['piece']} on {r['attach']}: contact {r.get('contact')} mm2, cover "
                 f"{r.get('cover')}\n{verdict}\ngrey/black: the part   red dashed: the feature   blue dotted: what holds it",
                 fontsize=9, color="k" if r["passed"] else "#b00000")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _sheet(paths, out, cols=4, w=500):
    from PIL import Image
    if not paths:
        return
    ims = [Image.open(p) for p in paths]
    ims = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    h = max(im.height for im in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out)


# -- the gate ------------------------------------------------------------------

@dataclass
class GateReport:
    name: str
    passed: bool
    bodies: list
    slivers: dict
    creases: list
    attachment: list
    failures: list = field(default_factory=list)
    images: dict = field(default_factory=dict)

    def summary(self) -> str:
        att = [r for r in self.attachment if not r.get("passed", True)]
        s = [f"{self.name}: {'PASS' if self.passed else 'FAIL'}",
             f"  bodies {len(self.bodies)}",
             f"  features {len(self.attachment)} pieces, {len(att)} not fused",
             f"  air gaps < gap {len(self.slivers.get('air', []))}, thin walls {len(self.slivers.get('material', []))}",
             f"  unfilleted creases {len(self.creases)}"]
        s += ["  - " + f for f in self.failures[:40]]
        return "\n".join(s)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stamp_path(path) -> Path:
    return Path(str(path) + ".gate.json")


def gate(mesh, name: str, out_dir, *, fields=None, features=(), box=None, solid: str = "solid",
         pitch: float = 0.25, coarse: float = 1.0, gap: float = 0.4, sliver_area: float = 0.3,
         crease_angle: float = 50.0, crease_length: float = 1.0, allow_bodies: int = 1, images: bool = True,
         raise_on_fail: bool = True, waive: dict | None = None, verbose: bool = True) -> GateReport:
    """Everything a part must pass before it leaves the tool. Fails (raises
    IntegrityError) on: more than `allow_bodies` bodies; a declared feature piece
    not fused to what it attaches to; air gaps or walls under `gap` mm; re-entrant
    creases sharper than `crease_angle`. `waive`: {"creases": [centre, ...], ...}
    for an exception a person has accepted, by where it is (within 1 mm), and
    listed in the report. Writes out_dir/name-gate.json, name-bodies.png and a
    section cut through every feature piece's root (name-sections/, and a
    contact sheet name-sections.png, failures first)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = _tm(mesh)
    waive = waive or {}
    fails = []
    say = (lambda s: print(s, flush=True)) if verbose else (lambda s: None)
    say(f"gate {name}: bodies")
    bs = bodies(m)
    if len(bs) > allow_bodies:
        fails.append(f"{len(bs)} bodies: " + ", ".join(f"{abs(b['volume']):.0f} mm3 at {b['centre']}" for b in bs[1:8]))
    say(f"gate {name}: slivers")
    sl = slivers(m, gap, sliver_area)
    say(f"gate {name}: creases")
    cr = creases(m, crease_angle, crease_length)

    def waived(kind, r):
        return any(np.linalg.norm(np.asarray(r["centre"]) - np.asarray(w)) < 1.0 for w in waive.get(kind, ()))
    for kind in ("air", "material"):
        for r in sl[kind]:
            r["waived"] = waived(kind, r)
            if not r["waived"]:
                fails.append(f"{'air gap' if kind == 'air' else 'wall'} under {gap} mm ({r['gap_min']}) over {r['area']} mm2 at {r['centre']}")
    for r in cr:
        r["waived"] = waived("creases", r)
        if not r["waived"]:
            fails.append(f"unfilleted crease {r['length']} mm long ({r['angle_max']} deg) at {r['centre']}")
    att = []
    if fields is not None and features:
        say(f"gate {name}: attachment of {len(features)} features")
        bx = box if box is not None else (m.bounds[0] - 2.0, m.bounds[1] + 2.0)
        att = attachment(fields, features, bx, solid=solid, pitch=pitch, coarse=coarse, verbose=verbose)
        for r in att:
            if not r["passed"]:
                fails.append(f"{r['feature']} #{r['piece']} at {r.get('centre')}: " + "; ".join(r["reasons"]))
    imgs = {}
    clean_bodies = [{k: v for k, v in b.items() if not k.startswith("_")} for b in bs]
    # the verdict goes to disk before the pictures, so a picture that fails loses no analysis
    (out / f"{name}-gate.json").write_text(json.dumps(asdict(GateReport(name, not fails, clean_bodies, sl, cr, att, fails, imgs)),
                                                      indent=1, default=float))
    if images:
        say(f"gate {name}: pictures")
        bp = out / f"{name}-bodies.png"
        body_map(m, bp, title=name)
        imgs["bodies"] = str(bp)
        if att and fields is not None:
            sd = out / f"{name}-sections"
            sd.mkdir(exist_ok=True)
            paths = []
            for r in sorted(att, key=lambda r: (r["passed"], r["feature"], r.get("piece", 0))):
                if r.get("piece", -1) < 0:
                    continue
                p = sd / f"{r['feature']}-{r['piece']}.png"
                _section_png(m, fields, r, p, solid)
                paths.append(p)
            _sheet(paths, out / f"{name}-sections.png")
            imgs["sections"] = str(out / f"{name}-sections.png")
            imgs["section_dir"] = str(sd)
    rep = GateReport(name, not fails, clean_bodies, sl, cr, att, fails, imgs)
    (out / f"{name}-gate.json").write_text(json.dumps(asdict(rep), indent=1, default=float))
    say(rep.summary())
    if fails and raise_on_fail:
        raise IntegrityError(rep.summary())
    return rep


def export(mesh, path, name: str | None = None, out_dir=None, **kw) -> GateReport:
    """Write the mesh to `path` and gate it; the stamp beside it records the
    file's hash and the verdict. A failed part is still written (to look at)
    but stamped failed, and this raises: nothing downstream takes it."""
    m = _tm(mesh)
    path = Path(path)
    m.export(path)
    name = name or path.stem
    out_dir = out_dir or path.parent / "gate"
    try:
        rep = gate(m, name, out_dir, raise_on_fail=False, **kw)
    finally:
        pass
    stamp = dict(version=STAMP_VERSION, file=path.name, sha256=_sha(path), passed=rep.passed,
                 failures=rep.failures[:200], report=str(Path(out_dir) / f"{name}-gate.json"), images=rep.images)
    stamp_path(path).write_text(json.dumps(stamp, indent=1))
    if not rep.passed:
        raise IntegrityError(rep.summary())
    return rep


def require_passed(path):
    """What a render, an upload or a solver calls on a part file: raise unless
    it has a passing stamp for exactly these bytes."""
    path = Path(path)
    sp = stamp_path(path)
    if not sp.exists():
        raise IntegrityError(f"{path} has no integrity stamp: export it through loadpath.integrity.export")
    st = json.loads(sp.read_text())
    if st.get("sha256") != _sha(path):
        raise IntegrityError(f"{path} changed after it was gated: gate it again")
    if not st.get("passed"):
        raise IntegrityError(f"{path} failed its gate:\n  " + "\n  ".join(st.get("failures", [])[:20]))
    return st
