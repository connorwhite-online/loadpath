"""
Generated exteriors: a smooth skin the optimizer grows round its own load
paths (study block `shroud`).

`skin`, `loadpath enclosure` and `loadpath handoff` start from an exterior
someone drew: the skin lines it and only what is behind it is designed.
`shroud` is the other way round. There is no drawn exterior, only a design
space (the room the part may take up), the interfaces and what has to be
hidden inside: a cable or tendon channel, a board, a motor, a heat
exchanger's passages. The part's outside is designed with its inside:

1. **Seed.** The study without the shroud is optimized on a coarse grid at
   a low volume fraction (`seedFraction`, 0.15): the load paths between the
   interfaces, no skin.
2. **Envelope.** The seed's members, the preserved interfaces and every
   region in `conceal` (grown by the skin and `gap`, so the skin clears
   them) are wrapped in one smooth closed surface: a morphological closing
   of radius `reach` (gaps and notches narrower than about twice it are
   bridged; holes through the part wider than that stay open), cavities
   filled, then the signed distance smoothed over `smooth` mm, so the
   surface sweeps instead of following every member. What must be inside
   is blended back in with a smooth union, what must stay outside (a
   keep-out that is not concealed: a neighbour's swept volume) is blended
   out, and the result is clipped to the design space with a rounded edge.
3. **Skin and interior together.** The envelope becomes the study's design
   space, and `skin` (thickness `thickness`) lines it: the skin is
   preserved solid and is in every analysis, so it carries load as a
   monocoque, and the optimizer then designs the interior (ribs, posts,
   the members the skin still needs) with the skin there. The study's goal
   search sets how much interior is left: often little, because a closed
   skin round the load path is a stiff tube.

Keep-outs keep their meaning. A bore (`bore`: true, the default) cuts
through the skin wherever it crosses it: a bearing seat, a bolt, a cable
channel's exit. A keep-out with `bore: false` that is listed in `conceal`
(a board, a motor) ends up inside the skin, sealed from view, with the
skin `gap` clear of it. One that is not concealed stays outside the
envelope. A cable channel is a bore keep-out listed in `conceal`: it runs
inside the part and opens only where it leaves the design space or meets
an exit port (another bore keep-out).

Printability: the interior behind a closed skin is a sealed cavity. On
SLS, MJF, resin and LPBF the study's `drains` (on by default there) drill
escape holes through the calmest thin wall, and processcheck reads the
delivered part for trapped powder. A cavity that opens through a channel
exit or bore needs none.

A unibody is the default. A part that has to come apart (to put the
components in) goes through `loadpath split` on the expanded study
(`loadpath shroud study.json --out d` writes it): the generated exterior is
then an ordinary skinned exterior and the split search and joints apply.

The envelope is cached by the study's content (LOADPATH_CACHE, default
~/.cache/loadpath/shroud), so a re-parse does not re-run the seed.

Units are millimetres.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

__all__ = ["expand_shroud", "envelope_sdf", "write_shroud", "DEFAULTS"]

VERSION = 4
DEFAULTS = {
    "thickness": 2.0,        # mm of skin
    "reach": 12.0,           # mm, closing radius
    "smooth": 8.0,           # mm, Gaussian smoothing of the envelope's distance
    "gap": 1.0,              # mm the skin keeps off a concealed region
    "seedFraction": 0.15,
    "seedCells": 25_000,
    "seedIterations": 70,
    "rounds": 10,            # shape steps on the envelope from the skin's load share (0: the seed's hull)
    "step": 1.5,             # mm, the largest move of a shape step
    "soft": 0.0,             # mm: inflate and round the wrap by this much (a soft, full form; 0: as tight as the load allows)
}


def _cache_dir() -> Path:
    root = os.environ.get("LOADPATH_CACHE") or (Path.home() / ".cache" / "loadpath")
    p = Path(root) / "shroud"
    p.mkdir(parents=True, exist_ok=True)
    return p


def envelope_path(key: str) -> str:
    """The cached envelope a shroud key names (refuses anything but a key)."""
    import re
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{20}", key):
        raise ValueError("sdf shroud is a generated envelope's key (20 hex digits)")
    p = _cache_dir() / f"envelope-{key}.npz"
    if not p.exists():
        raise ValueError(f"generated envelope {key} is not in {_cache_dir()}: expand the study's shroud again "
                         "(`loadpath shroud`), or point designSpace at the envelope.npz it wrote")
    return str(p)


def _options(v) -> dict:
    if v is True:
        v = {}
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = {"thickness": v}
    if not isinstance(v, dict):
        raise ValueError("shroud is true, a skin thickness in mm, or {thickness, reach, smooth, conceal, ...}")
    o = {**DEFAULTS, **{k: v[k] for k in v if k in DEFAULTS}}
    for k in ("thickness", "reach", "smooth", "seedFraction"):
        if not isinstance(o[k], (int, float)) or o[k] <= 0:
            raise ValueError(f"shroud.{k} must be a positive number")
    if not 0 < o["seedFraction"] < 1:
        raise ValueError("shroud.seedFraction is a volume fraction between 0 and 1")
    if o["gap"] < 0:
        raise ValueError("shroud.gap is mm (>= 0)")
    if not isinstance(o["soft"], (int, float)) or o["soft"] < 0:
        raise ValueError("shroud.soft is mm (>= 0)")
    if float(o["soft"]) > 0 and "rounds" not in v:
        # a soft form is the designer's choice of fullness: shape steps
        # would pull it back in to the load paths
        o["rounds"] = 0
    o["conceal"] = list(v.get("conceal") or [])
    o["openings"] = list(v.get("openings") or [])
    o["pitch"] = v.get("pitch")
    return o


def _sdf_from_mask(mask: np.ndarray, pitch: float) -> np.ndarray:
    """Signed distance (mm, negative inside) of a boolean node mask, the
    surface half a pitch out from the last inside node."""
    if not mask.any():
        return np.full(mask.shape, 1e3, np.float32)
    if mask.all():
        return np.full(mask.shape, -1e3, np.float32)
    out = ndimage.distance_transform_edt(~mask) * pitch
    ins = ndimage.distance_transform_edt(mask) * pitch
    return (np.where(mask, -ins, out) + np.where(mask, 0.5, -0.5) * pitch).astype(np.float32)


def _smin(a, b, k):
    """Polynomial smooth minimum (union) with blend radius k."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


# A boss's root fillet sized by its stress concentration: a shoulder of
# radius r on a member of diameter d holds Kt near 1.2 at r/d = 0.3 and
# gains little past it (Peterson's charts, stepped bar in bending)
_FILLET_RATIO = 0.3


def _round_union(a, b, r):
    """Union with an exact circular fillet of radius r where the two
    surfaces meet (both near-distances)."""
    if r <= 0:
        return np.minimum(a, b)
    u = np.maximum(r - a, 0.0)
    v = np.maximum(r - b, 0.0)
    return (np.maximum(r, np.minimum(a, b)) - np.sqrt(u * u + v * v)).astype(np.float32)


def _join_parts(F, parts):
    """The envelope joined to every interface by its own root fillet, and to
    what it conceals by the envelope's blend."""
    for g, r in parts.get("fillets") or ():
        F = _round_union(F, g, r)
    if parts.get("conceal") is not None:
        F = _smin(F, parts["conceal"], parts["k"])
    return F


def _smax(a, b, k):
    return -_smin(-a, -b, k)


def _nodes(lo, hi, pitch):
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 2)
    axes = [lo[i] + pitch * np.arange(dims[i]) for i in range(3)]
    return dims, axes


def _contains(shapes, pts, dims):
    m = np.zeros(len(pts), bool)
    for s in shapes:
        m |= s.contains(pts)
    return m.reshape(dims)


def envelope_sdf(solid_fn, study, opts: dict, pitch: float, refine=None):
    """The envelope's signed distance on a node grid over the design space.

    solid_fn(points) -> bool: the seed's members. Returns (distance grid,
    origin, pitch, info)."""
    from .shapes import Union

    t, gap, R, sm = float(opts["thickness"]), float(opts["gap"]), float(opts["reach"]), float(opts["smooth"])
    lo, hi = study.bounds()
    pad = 2 * pitch
    lo, hi = lo - pad, hi + pad
    dims, axes = _nodes(lo, hi, pitch)
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    ds = study.design_space[0] if len(study.design_space) == 1 else Union(study.design_space)
    in_ds = ds.contains(P).reshape(dims)
    seed = solid_fn(P).reshape(dims) & in_ds
    pres = _contains([r.shape for r in study.preserve], P, dims)

    ids = {r.id: r for r in list(study.preserve) + list(study.keep_out)}
    for cid in opts["conceal"]:
        if cid not in ids:
            raise ValueError(f"shroud.conceal: unknown region {cid!r}; known: {sorted(ids)}")
    concealed = [ids[c].shape for c in opts["conceal"]]
    hidden = _contains(concealed, P, dims)
    # what the skin must stay clear of: concealed regions grown by skin + gap,
    # measured from the shapes themselves (an EDT of their nodes reads up to
    # a pitch short at a corner: a board's corner kept 0.3 mm of its 1 mm
    # gap on a 1.1 mm grid)
    need = pres.copy()
    need_d = None
    if concealed:
        dh = np.min([np.asarray(sh.sdf(P), float) for sh in concealed], axis=0).reshape(dims)
        # (half a pitch more: the skin's inner offset reads up to that short)
        need_d = (dh - (t + gap + 0.5 * pitch)).astype(np.float32)
        need |= need_d <= 0
    # keep-outs neither concealed nor bores stay outside the envelope
    outside = [r.shape for r in study.keep_out
               if r.id not in opts["conceal"] and not getattr(r, "bore", True)]
    excl = _contains(outside, P, dims)

    core = seed | need
    # closing: grow by the reach, then shrink back
    grown = ndimage.distance_transform_edt(~core) * pitch <= R
    closed = ndimage.distance_transform_edt(grown) * pitch > R
    closed |= core
    closed = ndimage.binary_fill_holes(closed)
    # keep the piece(s) holding the interfaces; drop seed crumbs
    lab, n = ndimage.label(closed)
    if n > 1:
        keep = np.unique(lab[need & closed])
        keep = keep[keep > 0]
        if keep.size:
            closed = np.isin(lab, keep)
    F = _sdf_from_mask(closed, pitch)
    soft = float(opts.get("soft") or 0.0)
    if soft > 0:
        # a soft form: the wrap grown by `soft` and smoothed over it as well,
        # so it swells round the parts like a drawn volume smoothed hard
        F = F - np.float32(soft)
    if sm + soft > 0:
        F = ndimage.gaussian_filter(F, sigma=(sm + soft) / pitch / 2.0, mode="nearest").astype(np.float32)
    k = 0.5 * sm
    # (the interfaces by their own distance too: from their grid nodes they
    # came out faceted, and every post read as a lumpy cylinder stuck on)
    # Each joins the skin with a root fillet of 0.3 x its own size (twice its
    # inscribed radius): a smooth-min of the envelope's blend moved the
    # surface k/4, and the posts read as stubs stuck on.
    fillets = []
    for r in study.preserve:
        g = np.asarray(r.shape.sdf(P), np.float32).reshape(dims)
        fillets.append((g, _FILLET_RATIO * 2.0 * max(-float(g.min()), pitch)))
    need_sdf = np.min([g for g, _ in fillets], axis=0) if fillets else None
    if need_d is not None:
        need_sdf = need_d if need_sdf is None else np.minimum(need_sdf, need_d)
    F = _join_parts(F, {"fillets": fillets, "conceal": need_d, "k": k})
    if excl.any():
        F = _smax(F, -_sdf_from_mask(excl, pitch), min(k, 1.0))
    # inside the design space, its edges rounded off a little
    F = _smax(F, _sdf_from_mask(in_ds, pitch), k)
    parts = {"need": need_sdf, "fillets": fillets, "conceal": need_d,
             "excl": -_sdf_from_mask(excl, pitch) if excl.any() else None,
             "ds": _sdf_from_mask(in_ds, pitch), "k": k}
    F = _hold_need(F, parts).astype(np.float32)
    if refine is not None and int(opts.get("rounds", 0)) > 0:
        F, rinfo = refine(F, lo, pitch, parts)
        # faired after the steps: each moved the surface by a velocity read
        # on the coarse analysis grid, and together they leave waves a few
        # cells long (the arm link's top read as rippled close up)
        F = ndimage.gaussian_filter(F, sigma=max(0.25 * sm, pitch) / pitch, mode="nearest")
        F = _blend(F, parts, min(k, 1.0))
    else:
        rinfo = None
    inside = F <= 0
    info = {"pitch": pitch, "nodes": [int(x) for x in dims],
            "volume": float(inside.sum() * pitch ** 3),
            "designSpaceVolume": float(in_ds.sum() * pitch ** 3),
            "concealed": list(opts["conceal"])}
    if rinfo is not None:
        info["refine"] = rinfo
    # what the envelope failed to hold: a concealed region the design space
    # leaves no room to wrap
    miss = hidden & ~inside
    if miss.any():
        info["warnings"] = [f"part of a concealed region ({miss.sum() * pitch ** 3:.0f} mm^3) lies outside the "
                            "design space; the skin cannot wrap it there"]
    # skin area (marching cubes on the level set) for a mass estimate
    try:
        from skimage import measure
        v, f, _, _ = measure.marching_cubes(F, 0.0, spacing=(pitch,) * 3)
        a = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        info["area"] = float(0.5 * np.linalg.norm(a, axis=1).sum())
    except Exception:                                     # noqa: BLE001 (skimage optional)
        g = np.array(np.gradient(F, pitch))
        band = np.abs(F) < 0.5 * pitch
        info["area"] = float(band.sum() * pitch ** 2)
    return F, lo, pitch, info


def _seed(study, opts, cache_file=None):
    """The coarse free design: (solid_fn, info). Cached in `cache_file`
    (the seed depends on the study and the seed options only, so the
    envelope's options can be tuned without re-running it)."""
    if cache_file is not None and Path(cache_file).exists():
        z = np.load(cache_file, allow_pickle=False)
        grid, origin, pitch = z["grid"], z["origin"], float(z["pitch"])
        info = json.loads(str(z["info"]))
        return _solid_fn(grid, origin, pitch), info
    from .domain import build_domain
    from .simp import SimpOptions, filter_radius, optimize
    from .study import _making

    d0 = build_domain(study)
    cells = d0.n_elem
    p = d0.pitch
    if cells > opts["seedCells"]:
        p *= (cells / opts["seedCells"]) ** (1 / 3)
    dom = build_domain(study, pitch=p) if p != d0.pitch else d0
    radius = filter_radius(dom, study.min_member, study.min_hole)
    t0 = time.time()
    making = {**_making(study), "escape": None}
    res = optimize(dom, study.elastic(design=True),
                   SimpOptions(volume_fraction=float(opts["seedFraction"]), filter_radius=radius,
                               max_iterations=int(opts["seedIterations"]), **making))
    grid = dom.full_grid(res.xphys).astype(np.float32)
    rho = study.material.density
    info = {"pitch": float(dom.pitch), "cells": int(dom.n_elem), "iterations": int(res.iterations),
            "seconds": round(time.time() - t0, 1), "volumeFraction": float(opts["seedFraction"]),
            "mass": float(res.xphys.sum() * dom.pitch ** 3 * rho / 1000.0)}
    if cache_file is not None:
        np.savez_compressed(cache_file, grid=grid, origin=np.asarray(dom.origin, float), pitch=float(dom.pitch),
                            info=json.dumps(info))
    return _solid_fn(grid, dom.origin, dom.pitch), info


def _solid_fn(grid, origin, pitch):
    def solid(pts):
        c = (pts - origin) / pitch - 0.5
        return ndimage.map_coordinates(grid, c.T, order=1, mode="constant", cval=0.0) > 0.5
    return solid


def _blend(F, parts, k_small):
    """Put back what the envelope must hold and keep out, and the design
    space's rounded edge, after a shape step."""
    F = _join_parts(F, parts)
    if parts["excl"] is not None:
        F = _smax(F, parts["excl"], k_small)
    return _hold_need(_smax(F, parts["ds"], parts["k"]), parts).astype(np.float32)


def _hold_need(F, parts):
    """What the envelope must hold, held exactly: the rounded edge where the
    envelope meets the design space's wall otherwise pulls in by up to a
    quarter of its blend, and a concealed board near that wall lost half a
    millimetre of its skin's gap. Clipped to the design space, which stays
    the harder limit."""
    if parts["need"] is None:
        return F
    return np.minimum(F, np.maximum(parts["need"], parts["ds"]))


def _refiner(study, opts, seed_pitch, seed_solid=None):
    """A shape step on the envelope from the skin's own load share, `rounds`
    times: the skin (and the preserved interfaces) alone is analysed, every
    skin cell's strain energy density w is read (each case normalized by
    its compliance, so cases weigh alike), and the envelope moves along its
    normal by step x clip(log4(w) - log4(w_q), -1, 1): out where the skin
    is working hardest, in where it idles. That is the boundary velocity of
    level-set shape optimization (compliance's shape derivative is the
    strain energy density on the boundary) against a skin-area price w_q, a
    quantile of the skin's share. Each round tries four prices and keeps
    the step that lowers compliance x skin volume most (stiffness per mass,
    independent of the skin's thickness); a round that lowers it at no
    price halves the step, and the refinement stops when a short step
    fails too. So the envelope only moves where the skin is better for it:
    a link tapers toward its lightly loaded end. Each step is smoothed and
    re-blended (interfaces and concealed regions stay inside, the design
    space outside)."""
    from .directions import collapse, expand
    from .domain import build_domain
    from .simp import E_MIN, PENALTY, _Analysis

    rounds = int(opts.get("rounds", 0))
    rp = float(opts.get("refinePitch") or min(max(0.75 * seed_pitch, 0.8), 1.6))
    t = float(opts["thickness"])

    def run(F, lo, pe, parts):
        dom = build_domain(study, pitch=rp)
        cen = dom.mesh.elem_centers()
        nu = study.elastic(design=True)
        fe = _Analysis(dom, nu, 1e-4)
        spreads = dom.spreads or [None] * len(dom.loads)
        weights = np.asarray(dom.case_weights, float)
        t_eff = max(t, 1.5 * dom.pitch)
        step = float(opts.get("step") or 1.5)
        hist = []

        def sample(G, pts):
            c = ((pts - lo) / pe).T
            return ndimage.map_coordinates(G, c, order=1, mode="nearest")

        nodes_shape = F.shape
        axes = [lo[i] + pe * np.arange(nodes_shape[i]) for i in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        cP = ((P - dom.origin) / dom.pitch - 0.5).T
        sig_v = max(float(opts["smooth"]) / pe / 2.0, 1.0)
        sig_f = max(float(opts["smooth"]) / pe / 4.0, 1.0)

        def evaluate(G, xe=None):
            if xe is None:
                Fc = sample(G, cen)
                skin = (Fc <= 0) & (Fc > -t_eff)
                xe = np.where(skin | dom.preserve, 1.0, 0.0)
            else:
                skin = xe > 0.5
            e = E_MIN + xe ** PENALTY * (1 - E_MIN)
            rhs, layout = expand(dom.loads, spreads, None)
            us = fe.solve(e, xe, [None] * len(rhs), rtol=1e-4, loads=rhs)
            w = np.zeros(len(xe))
            comp = 0.0
            for ci, wc in enumerate(collapse(rhs, layout, us, spreads)):
                ce = sum(m * fe.strain_energy(v) for m, v in wc.energy)
                comp += weights[ci] * wc.compliance
                w += weights[ci] * ce / max(wc.compliance, 1e-30)
            vol = float(skin.sum() * dom.pitch ** 3)
            return {"compliance": float(comp), "skinVolume": vol, "w": w, "skin": skin}

        def velocity(ev):
            skin = ev["skin"]
            ws = np.maximum(ev["w"][skin], 1e-30)
            v_cell = np.log(ws) / np.log(4.0)
            g = np.full(dom.dims, np.nan, np.float32)
            g[tuple(dom.mesh.elem_ijk[skin].T)] = v_cell
            idx = ndimage.distance_transform_edt(np.isnan(g), return_distances=False, return_indices=True)
            V = ndimage.map_coordinates(g[tuple(idx)], cP, order=1, mode="nearest").reshape(nodes_shape)
            return ndimage.gaussian_filter(V, sigma=sig_v, mode="nearest"), v_cell

        def stepped(G, V, v_cell, q, h):
            # out where the skin's load share is over the q quantile, in under it
            lam = float(np.quantile(v_cell, q))
            G = G - h * np.clip(V - lam, -1.0, 1.0)
            G = _sdf_from_mask(G <= 0, pe)
            G = ndimage.gaussian_filter(G, sigma=sig_f, mode="nearest")
            return _blend(G, parts, min(parts["k"], 1.0))

        # the stiffness to keep: the free seed's own (the skin replaces it),
        # times `stiffness` (1: as stiff as the seed)
        c_seed = None
        if seed_solid is not None:
            xs = np.where(seed_solid(cen) | dom.preserve, 1.0, 0.0)
            ev_s = evaluate(None, xs)
            c_seed = ev_s["compliance"]
        c_t = c_seed / float(opts.get("stiffness") or 1.0) if c_seed else None

        def score(ev):
            # skin volume, priced up steeply once it is less stiff than the
            # target (without a target: compliance x volume)
            if c_t is None:
                return ev["compliance"] * ev["skinVolume"]
            return ev["skinVolume"] * max(ev["compliance"] / c_t, 1.0) ** 4

        cur = evaluate(F)
        cur["J"] = score(cur)
        hist = [{"round": 0, **{k: cur[k] for k in ("J", "compliance", "skinVolume")}}]
        h = step
        for r in range(1, rounds + 1):
            V, v_cell = velocity(cur)
            best = None
            for q in (0.1, 0.3, 0.5, 0.7, 0.9):
                G = stepped(F, V, v_cell, q, h)
                ev = evaluate(G)
                ev["J"] = score(ev)
                if best is None or ev["J"] < best[1]["J"]:
                    best = (G, ev, q)
            if best[1]["J"] >= cur["J"] * (1 - 2e-3):
                if h > 0.3 * step:
                    h *= 0.5            # nothing helped at this step: try a shorter one
                    continue
                break
            F, cur = best[0], best[1]
            hist.append({"round": r, "quantile": best[2], "step": h,
                         **{k: cur[k] for k in ("J", "compliance", "skinVolume")}})
        return F, {"rounds": rounds, "pitch": float(dom.pitch), "step": step, "history": hist,
                   "seedCompliance": c_seed, "targetCompliance": c_t}

    return run


def _key(d: dict) -> str:
    blob = json.dumps({"v": VERSION, "d": d}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def expand_shroud(d: dict, base_dir: str | None = None, emit=None) -> dict:
    """The study with its `shroud` replaced by the generated exterior: the
    envelope (an .npz signed distance) as the design space and a `skin` of
    the shroud's thickness lining it. The original design space and the
    seed are recorded in d["_shroud"]. Returns d unchanged without a
    shroud (or when already expanded)."""
    v = d.get("shroud")
    if v in (None, False) or d.get("_shroud") is not None:
        return d
    if d.get("skin") is not None:
        raise ValueError("shroud generates the skin itself: drop `skin` (set shroud.thickness instead)")
    opts = _options(v)
    from .spec import study_from_dict

    seed_d = copy.deepcopy(d)
    seed_d.pop("shroud")
    seed_d.pop("lattice", None)
    seed_d.pop("split", None)
    # the seed is the load paths alone: blocks that design round a skin
    # (cooling, feel, switch) come back on the expanded study
    for k in ("cooling", "feel", "switch"):
        seed_d.pop(k, None)
    key = _key({"study": seed_d, "shroud": {**opts, "openings": None}, "base": str(base_dir or "")})
    cache = _cache_dir()
    npz = cache / f"envelope-{key}.npz"
    meta = cache / f"envelope-{key}.json"
    info = None
    if npz.exists() and meta.exists():
        info = json.loads(meta.read_text())
    else:
        study = study_from_dict(copy.deepcopy(seed_d), base_dir)
        if emit:
            emit({"stage": "shroud-seed"})
        skey = _key({"study": seed_d, "seed": [opts["seedFraction"], opts["seedCells"], opts["seedIterations"]],
                     "base": str(base_dir or "")})
        solid, seed_info = _seed(study, opts, cache / f"seed-{skey}.npz")
        pitch = float(opts["pitch"] or min(max(0.5 * seed_info["pitch"], 0.4), 0.5 * opts["thickness"] + 0.25))
        if emit:
            emit({"stage": "shroud-envelope", "pitch": pitch})
        refine = _refiner(study, opts, seed_info["pitch"], solid) if int(opts["rounds"]) > 0 else None
        F, origin, pitch, env = envelope_sdf(solid, study, opts, pitch, refine)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, distance=F, origin=np.asarray(origin, float), pitch=float(pitch), units="mm")
        rho = study.material.density
        info = {"seed": seed_info, "envelope": env,
                "skinMassEstimate": float(env["area"] * opts["thickness"] * rho / 1000.0)}
        meta.write_text(json.dumps(info, indent=1))
    out = copy.deepcopy(d)
    shroud = out.pop("shroud")
    out["_shroud"] = {**info, "key": key, "path": str(npz), "options": {k: opts[k] for k in DEFAULTS},
                      "conceal": opts["conceal"], "designSpace": d["designSpace"], "spec": shroud}
    out["designSpace"] = {"type": "sdf", "shroud": key}
    skin = {"thickness": float(opts["thickness"])}
    if opts["openings"]:
        skin["openings"] = opts["openings"]
    out["skin"] = skin
    return out


def write_shroud(study_path: str, out_dir: str, base_dir: str | None = None, emit=None) -> dict:
    """`loadpath shroud`: the study with its generated exterior, written to
    out_dir/<name>.study.json, with out_dir/envelope.npz (the envelope's
    signed distance, for SDF tools) and out_dir/envelope.stl (to look at,
    or to hand a designer). The written study names the envelope by its key
    ({"type": "sdf", "shroud": key}, in the cache) and keeps the original's
    relative paths, so run it with --base-dir at the original's folder; to
    move it, point designSpace at envelope.npz instead. It is an ordinary
    skinned study: run it, or give it a `split` block and `loadpath split`
    it into a base and a cover."""
    import shutil

    d = json.loads(Path(study_path).read_text())
    base = base_dir or str(Path(study_path).resolve().parent)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = expand_shroud(d, base, emit=emit)
    if x is d:
        raise ValueError("the study has no shroud block")
    src = Path(x["_shroud"]["path"])
    dst = out / "envelope.npz"
    shutil.copyfile(src, dst)
    z = np.load(dst)
    from skimage import measure
    import trimesh
    v, f, _, _ = measure.marching_cubes(z["distance"], 0.0, spacing=(float(z["pitch"]),) * 3)
    trimesh.Trimesh(v + z["origin"], f[:, ::-1]).export(out / "envelope.stl")
    name = d.get("name", Path(study_path).stem)
    sp = out / f"{name}.study.json"
    sp.write_text(json.dumps(x, indent=1))
    return {"study": str(sp), "envelope": str(dst), "envelopeMesh": str(out / "envelope.stl"),
            "baseDir": base, "shroud": {k: v for k, v in x["_shroud"].items() if k not in ("designSpace", "spec")}}
