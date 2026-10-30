"""
Drain and escape holes for field parts with sealed voids.

A closed skin over a lattice (or any hollow in a field graph) seals the
powder in (SLS, MJF, metal powder bed) or the uncured resin (SLA, DLP,
volumetric). `drain_graph` finds every sealed void of a FieldGraph's
output on a voxel grid of the field itself and opens each one:

- voids too small to hold a hole (under the escape hole's own ball by
  default) are filled solid;
- every other void gets holes half a millimetre over the process's
  minimum escape hole (SLS 4 mm, MJF 5, resin 3.5, volumetric 3; LPBF 4.2,
  an M5 tap drill, so it can be plugged), two per void by default:
  the calmest flat site first, lowest along the build, then the calmest one
  well away from it, highest along the build, so air or rinse flows
  through (and resin has its drain at the bottom and its vent at the top);
- a site must be just under the skin with a straight way out, on a face
  flat out past the boss (a hole on an edge loses half its boss and sits
  where the stress gathers), clear of anything named in `keepClear`;
- each hole gets a solid boss (a hole is a stress raiser, about 3x in a
  plate, and the lattice walls it cuts would otherwise be loose slivers),
  smooth-unioned in;
- then the grid is labelled again, and voids a boss walled off get the
  same treatment, for up to four rounds.

The calm is the graph's own stress: the first `result` node (von Mises),
or the node named by `stress`. Without one, sites are ranked by height
along the build alone.

The holes and bosses are ordinary graph nodes (cylinders), added to the
graph, so the drilled part meshes, verifies and saves like any other.
Verification on the drilled field: no sealed void left (the same grid
labelling), the powder path from every point of each void to its nearest
hole through the void (heatx's geodesic; nothing unreached), and
processcheck's escape check (enclosed voids, chambers that only open
through gaps narrower than the hole). The mesh run_field writes is checked
again for sealed shells.

Units are millimetres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

__all__ = ["DrainOptions", "drain_graph", "drains_wanted", "hole_defaults"]

# processes with something to drain, and the ones processcheck has no card for
_DRAINED = ("sls", "mjf", "sla", "dlp", "volumetric", "lpbf")
_EXTRA = {"volumetric": 3.0, "lpbf": 4.2}         # lpbf: an M5 tap drill, so a hole can be plugged
_TINY = 1.0                                       # mm^3: smaller is aliasing, not a void
_ROUNDS = 6                                       # relabellings; holes in the first four, then fill only
_DRILL_ROUNDS = 4
_MAX_HOLES = 6                                    # a void, with maxPath
_BUDGET = 16_000_000


@dataclass
class DrainOptions:
    diameter: float | None = None     # mm (default the process's escape hole + 0.5; LPBF 4.2, an M5 tap drill)
    count: int = 2                    # holes per void
    stress: str | None = "auto"       # node whose value ranks the sites: "auto" = the first result node, None = none
    boss: float | None = None         # mm of solid round each hole (default max(1.5 x min wall, 0.4 x radius))
    blend: float | None = None        # mm fillet where the boss meets the part (default the boss wall)
    max_depth: float = 6.0            # mm: sites at most this far under the outside
    fill_below: float | None = None   # mm^3: fill smaller voids solid (default the escape ball's volume)
    keep_clear: list = field(default_factory=list)   # solid nodes the bosses stay out of
    clearance: float = 1.0            # mm from those
    max_path: float | None = None     # mm: more holes (up to 6 a void) while powder is farther than this from one
    pitch: float | None = None        # mm grid (default 0.75 x min wall, at most diameter / 8)

    _KEYS = {"diameter": "diameter", "count": "count", "stress": "stress", "boss": "boss", "blend": "blend",
             "maxDepth": "max_depth", "fillBelow": "fill_below", "keepClear": "keep_clear",
             "clearance": "clearance", "pitch": "pitch",
             "maxPath": "max_path"}

    @classmethod
    def from_spec(cls, spec) -> "DrainOptions":
        o = cls()
        if spec in (None, True, "auto", "yes"):
            return o
        if not isinstance(spec, dict):
            raise ValueError("drains is 'auto', true, false or an object")
        for k, v in spec.items():
            if k == "enabled":
                continue
            if k not in cls._KEYS:
                raise ValueError(f"drains: unknown key {k!r} ({', '.join(cls._KEYS)})")
            setattr(o, cls._KEYS[k], v)
        if o.diameter is not None and not float(o.diameter) > 0:
            raise ValueError("drains.diameter is a hole diameter in mm, > 0")
        if int(o.count) < 1:
            raise ValueError("drains.count is at least 1")
        o.count = int(o.count)
        if isinstance(o.keep_clear, str):
            o.keep_clear = [o.keep_clear]
        return o


def drains_wanted(spec, process) -> bool:
    """`drains` in a field graph: 'auto' (default) drills for powder and
    resin processes and LPBF; true/an object always; false never."""
    if spec is False or spec == "no":
        return False
    if isinstance(spec, dict):
        return spec.get("enabled", True) is not False
    if spec is True or spec == "yes":
        return True
    return process is not None and process.kind in _DRAINED


def hole_defaults(process):
    """(drilled diameter mm, the process's minimum escape hole mm, holes per
    void) for a process (spec.Process or name). Powder and resin holes are
    drilled half a millimetre over the minimum: a hole exactly at it is a
    narrow exit to processcheck's escape ball on any grid, and to powder
    that has caked a little. LPBF drills an M5 tap size, to plug."""
    from .processcheck import RULES

    kind = str(getattr(process, "kind", process) or "").lower()
    esc = getattr(process, "escape_hole", None)
    if kind in RULES:
        least = float(esc or RULES[kind]["escapeHole"])
        return least + 0.5, least, int(RULES[kind]["escapeHoles"])
    least = float(esc or _EXTRA.get(kind, 4.0))
    return least + (0.0 if kind == "lpbf" and esc is None else 0.5), least, 2


def _as_process(process):
    if process is None or hasattr(process, "kind"):
        return process
    from .spec import _process

    return _process(process)


def _up(build, process):
    from .am import parse_build_direction

    b = build if build is not None else getattr(process, "build", None)
    if b is None or b == "auto":
        return np.array([0.0, 0.0, 1.0])
    b = parse_build_direction(b) if not (isinstance(b, tuple) and len(b) == 2) else b
    if len(b) == 2:
        u = np.zeros(3)
        u[b[0]] = b[1]
        return u
    return np.asarray(b, float)


def _sample(fn, lo, dims, p, progress=None):
    """fn at lo + ijk * p over the grid, float32."""
    F = np.empty(tuple(dims), np.float32)
    ys = lo[1] + np.arange(dims[1]) * p
    zs = lo[2] + np.arange(dims[2]) * p
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    per = max(1, 1_000_000 // (dims[1] * dims[2]))
    for i0 in range(0, dims[0], per):
        i1 = min(dims[0], i0 + per)
        xs = lo[0] + np.arange(i0, i1) * p
        P = np.stack([np.repeat(xs, Y.size), np.tile(Y.ravel(), len(xs)), np.tile(Z.ravel(), len(xs))], axis=1)
        F[i0:i1] = np.asarray(fn(P), np.float32).reshape(i1 - i0, dims[1], dims[2])
        if progress:
            progress({"stage": "drains", "done": i1, "of": int(dims[0])})
    return F


def _box(lo, p, dims, a, b, reach):
    """Grid block round segment a-b out to `reach`, and its points."""
    i0 = np.floor((np.minimum(a, b) - reach - lo) / p).astype(int).clip(0)
    i1 = (np.ceil((np.maximum(a, b) + reach - lo) / p).astype(int) + 1).clip(None, np.asarray(dims))
    sl = tuple(slice(x0, x1) for x0, x1 in zip(i0, i1))
    ii = np.stack(np.meshgrid(*[np.arange(x0, x1) for x0, x1 in zip(i0, i1)], indexing="ij"), -1)
    return sl, lo + ii.reshape(-1, 3) * p, ii.shape[:3]


def _mask_sdf(mask, p):
    """Signed distance (negative inside) of a voxel mask, boundary halfway
    between voxel centres."""
    return np.where(mask, -(ndimage.distance_transform_edt(mask) - 0.5) * p,
                    (ndimage.distance_transform_edt(~mask) - 0.5) * p).astype(np.float32)


def _labels(F, p):
    """The sealed voids, labelled (6-connected), how many, the outside (a
    mask) and the voids' sizes in voxels. What reaches the outside is
    worked out on the void shrunk by half a voxel, so a wall thinner than a
    voxel (where a lattice wall grazes the skin) still seals as it does in
    the mesh; the fringe of open space (three voxels of void out from it:
    a blend's field is not a true distance, so its fringe runs wider) is
    open too. Everything
    else that is not solid is sealed, down to the thinnest sliver."""
    core, n = ndimage.label(F >= 0.5 * p)
    border = set()
    for face in (core[0], core[-1], core[:, 0], core[:, -1], core[:, :, 0], core[:, :, -1]):
        border.update(np.unique(face).tolist())
    border.discard(0)
    outside = np.isin(core, list(border))
    del core
    void = F >= 0
    outside = ndimage.binary_dilation(outside, iterations=3, mask=void) | outside
    lbl, n = ndimage.label(void & ~outside)
    sizes = np.bincount(lbl.ravel(), minlength=n + 1)
    return lbl, n, outside, sizes


def drain_graph(g, process=None, opts: DrainOptions | None = None, build=None, min_wall=None, bounds=None,
                pitch=None, progress=None) -> dict:
    """Drill the sealed voids of `g`'s output (see the module docstring).
    Adds the fill, boss and bore nodes to `g` and points its output at the
    drilled part ('drained'); returns the report. `process` is a name, a
    study-style process object or a spec.Process; `build` the build
    direction (default the process's, else +z); `pitch` the grid (default
    drains.pitch, else 0.75 x the minimum wall, at most diameter / 8): give
    the mesh's pitch so the specks the mesh would see are seen here too."""
    from .fields import _smin
    from .heatx import _TAPS, _geodesic
    from .processcheck import _escapes
    from .shapes import Cylinder

    proc = _as_process(process)
    o = opts or DrainOptions()
    dia, least, per_void = hole_defaults(proc)
    if o.diameter is not None:
        dia = float(o.diameter)
    count = int(o.count) if o.count is not None else per_void
    r = 0.5 * dia
    min_wall = float(min_wall if min_wall is not None else getattr(proc, "min_feature", None) or 0.8)
    bw = float(o.boss) if o.boss is not None else max(1.5 * min_wall, 0.4 * r)
    k = float(o.blend) if o.blend is not None else bw
    fill_below = float(o.fill_below) if o.fill_below is not None else 4.0 / 3.0 * np.pi * r ** 3
    up = _up(build, proc)
    for name in o.keep_clear:
        if name not in g.nodes:
            raise ValueError(f"drains.keepClear: {name!r} is not a node")
    sname = o.stress
    if sname == "auto":
        sname = next((n for n, nd in g.nodes.items() if nd.op == "result"
                      and nd.spec.get("quantity", "vonMises") == "vonMises"), None)
    elif sname is not None and sname not in g.nodes:
        raise ValueError(f"drains.stress: {sname!r} is not a node")

    if bounds is None:
        if "bounds" in g.doc:
            bounds = (g.doc["bounds"]["min"], g.doc["bounds"]["max"])
        else:
            bounds = g.bounds()
        if bounds is None:
            raise ValueError("drains need the part's bounds: give the graph 'bounds'")
    blo, bhi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    p = float(o.pitch or pitch or min(0.75 * min_wall, dia / 8.0))
    pad = (r + bw + k) + 3 * p
    p = max(p, float((np.prod(bhi - blo + 2 * pad) / _BUDGET) ** (1 / 3)))
    pad = (r + bw + k) + 3 * p
    lo = blo - pad
    dims = np.ceil((bhi + pad - lo) / p).astype(int) + 1
    vox = p ** 3
    base = g.output
    F0 = _sample(g.field(base), lo, dims, p, progress)

    # the stress the sites are ranked by, and its scale (the part's own peak)
    def stress_at(P):
        return g.evaluate(P, sname) if sname else np.zeros(len(P))

    s_ref = 1.0
    if sname:
        ijk = np.argwhere(F0 < 0)
        if len(ijk) > 20_000:
            ijk = ijk[np.random.default_rng(0).choice(len(ijk), 20_000, replace=False)]
        s_ref = float(np.percentile(stress_at(lo + ijk * p), 99.5)) or 1.0

    lbl, n, outside0, sizes = _labels(F0, p)
    every = list(range(1, n + 1))
    sealed0 = [i for i in every if sizes[i] * vox >= _TINY]
    voids = []
    if sealed0:
        cent = ndimage.center_of_mass(np.ones_like(lbl, bool), lbl, sealed0)
        voids = [{"volume": round(float(sizes[i] * vox), 1), "center": np.round(lo + np.asarray(c) * p, 2).tolist(),
                  "id": i} for i, c in zip(sealed0, cent)]
    rep = {"process": getattr(proc, "kind", None), "diameter": dia, "minimumHole": least, "holesPerVoid": count, "boss": round(bw, 3),
           "pitch": round(p, 4), "stressNode": sname, "sealedBefore": len(voids),
           "sealedVolumeBefore": round(float(sum(v["volume"] for v in voids)), 1),
           # specks: sealed bits under a cubic millimetre (where a wall grazes the skin), filled
           "specks": len(every) - len(voids)}
    if not every:
        rep.update({"holes": [], "voids": [], "filledVolume": 0.0, "sealedAfter": 0, "sealedVolumeAfter": 0.0,
                    "ok": True})
        return rep

    holes, fill = [], np.zeros(tuple(dims), bool)
    lbl0 = lbl
    F = F0
    rounds = 0
    for rnd in range(_ROUNDS + 1):
        outside = outside0
        if rnd:
            lbl, n, outside, sizes = _labels(F, p)
        sealed = list(range(1, n + 1))
        if not sealed or rnd == _ROUNDS:
            break
        rounds = rnd + 1
        dist_out, near = ndimage.distance_transform_edt(~outside, return_indices=True)
        dist_out = (dist_out * p).astype(np.float32)
        objs = ndimage.find_objects(lbl)
        placed = 0
        for i in sorted(sealed, key=lambda i: -sizes[i]):
            vol = float(sizes[i] * vox)
            if vol < fill_below:
                fill |= lbl == i
                placed += 1
                continue
            if rnd >= _DRILL_ROUNDS:          # out of drilling rounds: reported as undrained
                continue
            new = _place(i, lbl, objs[i - 1], F, outside, dist_out, near, lo, p, r, bw, count, o, up,
                         stress_at, s_ref, holes, g)
            for h in new:
                h["void"] = int(lbl0[tuple(np.floor((h["site"] - lo) / p + 0.5).astype(int))])
            holes += new
            placed += len(new)
        if not placed:
            break
        F = _apply(F0, fill, holes, lo, p, dims, k, _smin, Cylinder)

    # what is left sealed, after everything
    sealed_after = list(range(1, n + 1))
    rep["rounds"] = rounds
    rep["sealedAfter"] = len(sealed_after)
    rep["sealedVolumeAfter"] = round(float(sum(sizes[i] for i in sealed_after) * vox), 1)
    if sealed_after:
        cent = ndimage.center_of_mass(np.ones_like(lbl, bool), lbl, sealed_after)
        rep["undrained"] = [{"volume": round(float(sizes[i] * vox), 1),
                             "center": np.round(lo + np.asarray(c) * p, 2).tolist()}
                            for i, c in zip(sealed_after, cent)][:10]
    rep["filledVolume"] = round(float(fill.sum() * vox), 1)

    # powder path: from every point of each original void, through it, to the nearest hole
    mask = (F >= 0) & ~outside0
    starts = []
    for h in holes:
        t = np.arange(0.0, h["depth"], 0.5 * p)
        ij = np.floor((h["site"] + np.outer(t, h["axis"]) - lo) / p + 0.5).astype(int)
        starts += [tuple(x) for x in ij if mask[tuple(x)]]
    geo = _geodesic(mask, starts, p) if starts else np.full(mask.shape, np.inf)
    for v in voids:
        m = (lbl0 == v["id"]) & mask
        reach = geo[m]
        fin = np.isfinite(reach)
        v["holes"] = sum(1 for h in holes if h.get("void") == v["id"])
        v["filled"] = bool(fill[lbl0 == v["id"]].all())
        v["farthestFromOpening"] = round(float(reach[fin].max()), 1) if fin.any() else None
        v["unreachedVolume"] = round(float((~fin).sum() * vox), 1)
    rep["voids"] = [{k_: v_ for k_, v_ in v.items() if k_ != "id"} for v in voids]
    rep["unreachedVolume"] = round(float(sum(v["unreachedVolume"] for v in voids)), 1)

    # processcheck's escape probe on the drilled field: 26-connected, with
    # the process's minimum escape ball
    esc = _escapes(F < 0, p, lo - 0.5 * p, min(least, dia), np.eye(3))
    rep["escape"] = {k_: esc[k_] for k_ in ("enclosedCount", "enclosedVolumeMm3", "narrowExitCount",
                                           "narrowExitVolumeMm3")}

    tap = min(_TAPS, key=lambda t: abs(t[0] - dia))
    out = []
    for j, h in enumerate(holes):
        e = {"at": np.round(h["surface"], 2).tolist(), "axis": np.round(h["axis"], 3).tolist(), "diameter": dia,
             "depth": round(float(h["depth"]), 2), "stress": round(float(h["stress"]), 4),
             "stressRatio": round(float(h["stress"] / s_ref), 3) if sname else None}
        if rep["process"] == "lpbf" and abs(tap[0] - dia) < 0.25:
            e["seal"] = (f"leave open, or tap {tap[1]} into the boss and plug it with a set screw and thread "
                         "sealant after de-powdering")
        out.append(e)
    rep["holes"] = out
    rep["ok"] = rep["sealedAfter"] == 0 and rep["unreachedVolume"] < 1.0

    # into the graph: fill, bosses (smooth union), bores
    cur = base
    if fill.any():
        sl, sdf = _fill_grid(fill, p)
        org = lo + np.array([s.start for s in sl]) * p
        g.sources["drain-fill"] = {"origin": org, "pitch": p, "values": sdf}
        hi_ = org + (np.array(sdf.shape) - 1) * p
        g._add("drain-fill", {"op": "grid", "source": "drain-fill", "bounds": {"min": org.tolist(),
                                                                               "max": hi_.tolist()}})
        cur = g._add("drain-filled", {"op": "union", "inputs": [cur, "drain-fill"]})
    if holes:
        bosses, bores = [], []
        for j, h in enumerate(holes):
            bosses.append(g._add(f"drain-boss-{j + 1}", {"op": "shape", "shape": _cyl(*h["boss"])}))
            bores.append(g._add(f"drain-bore-{j + 1}", {"op": "shape", "shape": _cyl(*h["bore"])}))
        cur = g._add("drain-bosses", {"op": "union", "inputs": [cur] + bosses, "blend": k})
        cur = g._add("drained", {"op": "subtract", "inputs": [cur] + bores})
    elif cur != base:
        cur = g._add("drained", {"op": "union", "inputs": [cur]})
    g.output = cur
    g._check_refs()
    rep["output"] = cur
    return rep


def _cyl(a, b, rad):
    return {"type": "cylinder", "p0": np.round(a, 4).tolist(), "p1": np.round(b, 4).tolist(), "r": round(float(rad), 4)}


def _fill_grid(mask, p):
    """The filled pockets as a local signed-distance grid, grown two voxels
    (the labelling's half voxel, and one more) into the walls round them:
    they are sealed, so that stays inside, and a fill that stopped short
    would leave a thin sealed shell of its own."""
    sl = ndimage.find_objects(mask.astype(np.int8))[0]
    sl = tuple(slice(max(s.start - 3, 0), min(s.stop + 3, n)) for s, n in zip(sl, mask.shape))
    return sl, _mask_sdf(mask[sl], p) - 2.0 * p


def _apply(F0, fill, holes, lo, p, dims, k, _smin, Cylinder):
    """F0 with the fill, bosses and bores applied, as the graph will do it."""
    F = F0.copy()
    if fill.any():
        sl, sdf = _fill_grid(fill, p)
        F[sl] = np.minimum(F[sl], sdf)
    for h in holes:
        a, b, rad = h["boss"]
        sl, P, shp = _box(lo, p, dims, a, b, rad + k + p)
        F[sl] = _smin(F[sl], Cylinder(a, b, rad).sdf(P).reshape(shp), k)
    for h in holes:
        a, b, rad = h["bore"]
        sl, P, shp = _box(lo, p, dims, a, b, rad + p)
        F[sl] = np.maximum(F[sl], -Cylinder(a, b, rad).sdf(P).reshape(shp))
    return F


def _place(i, lbl, sl, F, outside, dist_out, near, lo, p, r, bw, count, o, up, stress_at, s_ref, holes, g):
    """Up to `count` holes into void i: the calmest site lowest along the
    build, then the calmest well away from it highest along the build, then
    the calmest farthest from both."""
    off = np.array([s.start for s in sl])
    comp = lbl[sl] == i
    dsub = dist_out[sl]
    # sites on the void's own wall: the boss then ends just inside the
    # void, not as a plug half way across a channel
    wall = comp & ndimage.binary_dilation(F[sl] < 0)
    cand = wall & (dsub <= o.max_depth)
    if not cand.any():
        cand = comp & (dsub <= dsub[comp].min() + p)
    idx = np.argwhere(cand) + off
    if len(idx) > 40_000:
        idx = idx[np.sort(np.random.default_rng(0).choice(len(idx), 40_000, replace=False))]
    pts = lo + idx * p
    # the stress the hole sees: the worse of the site's and its mouth's
    # (the nearest outside; a bending part is calm at the core, not at the skin)
    mouth = lo + near[(slice(None),) + tuple(idx.T)].T * p
    sv = np.maximum(stress_at(pts), stress_at(mouth))
    bins = np.floor(np.maximum(sv, 0.0) / (0.05 * s_ref)) if s_ref > 0 else np.zeros(len(pts))
    # height in bands of a hole radius; inside a band, a shuffled order, so
    # the tries spread over the face instead of walking along one edge
    h = np.round(pts @ up / r)
    shuffle = np.random.default_rng(i).random(len(pts))
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    sep = max(4.0 * 2 * r, 0.5 * extent / max(1, count - 1))
    clear = 2.0 * (r + bw) + p           # bosses of two holes do not overlap
    picks = []
    tried = np.zeros(len(pts), bool)
    key = bins * 100_000 + h            # a (stress bin, height band) group
    for slot in range(count):
        if slot == 0:
            order = np.lexsort((shuffle, h, bins))
        elif slot == 1:
            order = np.lexsort((shuffle, -h, bins))
        else:
            dmin = np.min([np.linalg.norm(pts - q["site"], axis=1) for q in picks], axis=0)
            order = np.lexsort((shuffle, -np.round(dmin / (2 * r)), bins))
        # calm first (within 10% of the peak of the calmest site left), well
        # apart if it can be, else anywhere the bosses do not touch; only
        # then the rest of the void
        calm = bins <= bins[~tried].min() + 2 if (~tried).any() else bins >= 0
        for subset, apart in ((calm, sep), (calm, clear), (None, sep), (None, clear)):
            site = _first_site(order if subset is None else order[subset[order]], pts, idx, key, tried, picks,
                               holes, apart, clear, F, outside, dist_out, lo, p, r, bw, o, g)
            if site is not None:
                break
        if site is None:
            break
        site["stress"] = float(sv[site.pop("j")])
        picks.append(site)
    # powder farther than max_path from every hole (through the void): one
    # more hole, calm first, farthest along the void, until it is not
    while o.max_path and picks and len(picks) < _MAX_HOLES:
        from .heatx import _geodesic

        geo = _geodesic(comp, [tuple(np.floor((q["site"] - lo) / p + 0.5).astype(int) - off) for q in picks], p)
        reach = geo[comp & np.isfinite(geo)]
        if not reach.size or reach.max() <= o.max_path:
            break
        gc = geo[tuple((idx - off).T)]
        gc = np.where(np.isfinite(gc), gc, 0.0)
        order = np.lexsort((shuffle, -np.round(gc / (2 * r)), bins))
        far = gc > 0.5 * o.max_path
        calm = (bins <= bins[far & ~tried].min() + 2) & far if (far & ~tried).any() else far
        site = _first_site(order[calm[order]], pts, idx, key, tried, picks, holes, clear, clear, F, outside,
                           dist_out, lo, p, r, bw, o, g)
        if site is None:
            break
        site["stress"] = float(sv[site.pop("j")])
        picks.append(site)
    return picks


def _first_site(order, pts, idx, key, tried, picks, holes, apart, clear, F, outside, dist_out, lo, p, r, bw, o, g):
    """The first workable site in `order`, at most 40 tries per (stress bin,
    height band) group so a face that cannot take a hole (an edge, a
    keepClear) does not use up the search, 3000 in all."""
    per = {}
    n_try = 0
    for j in order:
        if tried[j] or per.get(key[j], 0) >= 40:
            continue
        q = pts[j]
        if any(np.linalg.norm(q - x["site"]) < apart for x in picks):
            continue
        if any(np.linalg.norm(q - x["site"]) < clear for x in holes):
            continue
        tried[j] = True
        per[key[j]] = per.get(key[j], 0) + 1
        site = _site(tuple(idx[j]), F, outside, dist_out, lo, p, r, bw, o, g)
        if site is not None:
            site["j"] = j
            return site
        n_try += 1
        if n_try >= 3000:
            return None
    return None


def _site(ijk, F, outside, dist_out, lo, p, r, bw, o, g):
    """A hole from void voxel ijk straight out, if there is one: the line
    leaves the part once within the skin's depth and stays out; the face
    round the mouth is flat out past the boss; the boss stays inside the
    part and clear of the keepClear nodes. Returns the hole (site, axis,
    surface point, depth, boss and bore cylinders) or None."""
    dims = np.asarray(F.shape)
    ijk = np.asarray(ijk)
    a0 = np.clip(ijk - 1, 0, dims - 1)
    a1 = np.clip(ijk + 1, 0, dims - 1)
    grad = np.array([dist_out[tuple(np.where(np.arange(3) == ax, a1, ijk))] -
                     dist_out[tuple(np.where(np.arange(3) == ax, a0, ijk))] for ax in range(3)], float)
    if np.linalg.norm(grad) < 1e-6:
        return None
    nrm = -grad / np.linalg.norm(grad)
    pt = lo + ijk * p
    depth = float(dist_out[tuple(ijk)])
    t = np.arange(0.0, 1.5 * depth + 2.0 * r + 3.0 * p, 0.5 * p)
    fi = (pt + np.outer(t, nrm) - lo) / p
    past = ((fi < 0) | (fi > dims - 1)).any(axis=1)        # off the grid is outside too
    fi = np.clip(fi, 0, dims - 1)
    ii = np.floor(fi + 0.5).astype(int)
    out = outside[tuple(ii.T)] | past
    if not out.any():
        return None
    e = int(np.argmax(out))
    if t[e] > 1.5 * depth + 2.0 * p or not out[e:].all():
        return None
    Fv = ndimage.map_coordinates(F, fi.T, order=1, mode="nearest")
    js = np.nonzero(Fv[:e + 1] < 0)[0]
    if not len(js) or js[-1] + 1 >= len(t):
        return None
    j = js[-1]
    t_surf = t[j] + (t[j + 1] - t[j]) * Fv[j] / (Fv[j] - Fv[j + 1])
    skin = t_surf - t[js[0]] + 0.5 * p
    surf = pt + nrm * t_surf
    cap = 0.5 * skin                     # the boss ends half way through the skin
    u = np.cross(nrm, [1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    circ = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    reach = r + bw
    ring = surf + reach * circ
    fv = ndimage.map_coordinates(F, ((ring - lo) / p).T, order=1, mode="nearest")
    # convex past the boss's cap pokes it out; concave is a bore, a seat or
    # a fillet: a functional face, or a corner the stress gathers in
    if fv.max() > 0.8 * cap or fv.min() < -0.5 * p:
        return None
    boss = (surf - nrm * cap, pt - nrm * bw, reach)
    bore = (pt - nrm * (bw + r), surf + nrm * (1.0 + p), r)
    # the boss's side and the bore's inner end stay inside the part
    axial = np.linspace(cap, t_surf + bw + r, 6)
    P = (surf[None, None, :] - axial[:, None, None] * nrm + reach * circ[None, :, :]).reshape(-1, 3)
    fp = (P - lo) / p
    if (fp < 0).any() or (fp > dims - 1).any():
        return None
    if outside[tuple(np.floor(fp + 0.5).astype(int).T)].any():
        return None
    if o.keep_clear:
        P2 = np.vstack([P, surf - np.outer(axial, nrm)])
        for name in o.keep_clear:
            if (g.evaluate(P2, name) < o.clearance).any():
                return None
    return {"site": pt, "axis": nrm, "surface": surf, "depth": t_surf, "boss": boss, "bore": bore}
