"""
Target-force push pads: design everything inside a fixed exterior so a
press anywhere on a touch area fires a switch at a chosen force, with the
least material.

The problem (a soft SLS TPU pad over a tactile switch, first posed for a
touch microphone): the exterior is industrial design and must not move, the
board and switch sit inside, and a plunger under the touch area clears the
switch stem by a gap. Pressing the pad bends the skin, the plunger crosses
the gap, then pushes the stem until the switch clicks at its trigger force.
How hard a press at point i has to be to get there depends on everything
in between: how thick the skin is, where it is tied to the walls, whether
spokes carry an edge press over to the plunger. That is a compliant
mechanism, and the skin is part of it, so only a thin cosmetic layer of the
exterior is kept (the study's `skin`) and the rest is design space.

Fire force, linear, per press point i (unit loads, mesh units mm and N):

    u_i     = K^-1 f_i           f_i: 1 N spread over a disc at press i
    lam     = K^-1 l             l:   the plunger face's mean travel along
                                      the stroke (a unit load spread on it)
    a_i     = l . u_i            plunger travel per N pressed at i
    c       = l . lam            plunger self-compliance (mm/N)
    F_i     = (g + s + T c) / a_i

with gap g, switch pre-travel s and trigger force T: at the click the
plunger has moved g + s while the stem pushes back with T (superposition of
the press and the reaction; by reciprocity the reaction lifts press point i
by a_i T). The press travel at the click is d_i = (f_i . u_i) F_i - a_i T:
the "mush" before the click.

Every quantity is a product of two solves, so every sensitivity is a pair
of solves already done: d(v . K^-1 w)/dx_e = -(K^-1 v)_e . dK_e (K^-1 w)_e.
One forward solve per press point plus one for the plunger per iteration.

The optimizer minimizes volume subject to one constraint per press point
(MMA; Svanberg 1987), on log F so a factor of two reads the same at 100 gf
and at 800 gf:

    minimize   V(x)
    subject to F_i <= F_max(p_i)      the ceiling, any gradient over the area
               F_i >= F_min(p_i)      the floor: a resting thumb does not fire it
               d_i <= maxTravel       crispness (optional)
               skin travel over a clearance region <= its gap (optional, KS
                                      over nodes and presses, one adjoint per press)
               mean deflection under each `hold` load case <= its limit
                                      (squeeze, drop: the rim keeps its shape)

Coarse to fine: `pitches` runs the same problem on finer and finer grids,
each starting from the last one's design. From simp.MG_MIN_CELLS cells the
solves go through the matrix-free multigrid (mg.py), which takes all the
right-hand sides with one hierarchy and runs on a GPU with LOADPATH_DEVICE.

This is a linear model: TPU skins stiffen as they bow, and the plunger
meets the stem in contact. `check_fire_nonlinear` re-reads each press with
large deflections, the stem's reaction applied at the click.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .domain import Domain, _region_nodes
from .fem import hex8_ke

__all__ = ["Press", "SwitchSpec", "SwitchResult", "switch_from_study", "fire_forces", "optimize_switch",
           "check_fire_nonlinear", "GF"]

GF = 0.00980665      # N per gram-force

_AXES = {"x": 0, "y": 1, "z": 2}


def _axis(v) -> tuple[int, int]:
    """'-z' -> (2, -1); '+x' or 'x' -> (0, 1)."""
    s = str(v).strip().lower()
    sign = -1 if s.startswith("-") else 1
    a = s.lstrip("+-")
    if a not in _AXES:
        raise ValueError(f"direction must be an axis like '-z' or '+x', got {v!r}")
    return _AXES[a], sign


@dataclass
class Press:
    at: np.ndarray                    # the point (2 or 3 coordinates; the press axis' coordinate is found)
    max: float | None = None          # N: fire at or below this
    min: float | None = None          # N: and not below this


@dataclass
class SwitchSpec:
    output_nodes: np.ndarray          # bool per node: the plunger face
    axis: int                         # stroke axis
    sign: int                         # stroke direction along it
    gap: float                        # mm, plunger to stem at rest
    trigger: float                    # N, the switch's operating force
    travel: float                     # mm, the switch's own travel to the click
    presses: list[Press]
    press_axis: int
    press_sign: int
    press_radius: float               # mm
    max_travel: float | None = None   # mm of press travel at the click
    clearance_nodes: np.ndarray | None = None
    clearance: float | None = None    # mm the clearance nodes may move along the press direction
    hold: list = field(default_factory=list)    # [(case index, max mean deflection mm)]
    # what the switch body bears on (see _bears_on): nodes in this mesh (the
    # stem's reaction pushes them along the stroke, and the plunger has to
    # close on them), and/or the compliance of a separate fixed part under
    # the switch, mm/N, whose give the plunger has to take up too
    base_nodes: np.ndarray | None = None
    base_compliance: float = 0.0
    base_info: dict = field(default_factory=dict)
    # per clearance node: the room it has (mm) and which entry it belongs to
    # (see _clearance); `clearance` stays the smallest of them
    clearance_gap: np.ndarray | None = None
    clearance_entry: np.ndarray | None = None
    clearance_entries: list = field(default_factory=list)

    @property
    def stroke(self) -> float:
        """How far the plunger closes on the switch's seat at the click: the
        gap, the switch's own travel, and the give of a separate part the
        switch bears on under the trigger force."""
        return self.gap + self.travel + self.trigger * self.base_compliance


@dataclass
class SwitchResult:
    x: np.ndarray
    xphys: np.ndarray
    domain: Domain
    forces: np.ndarray                # N to fire, per press (blueprint design, linear)
    travel: np.ndarray                # mm of press travel at the click, per press
    volume: float                     # solid share of the design cells
    iterations: int
    seconds: float
    history: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    floor: np.ndarray | None = None   # per element: a thin skin's density floor (see _thin_skin)


# ---------------------------------------------------------------------------
# problem set-up


def _press_nodes(domain: Domain, at, axis: int, sign: int, radius: float) -> np.ndarray:
    """Nodes a fingertip pressing along (axis, sign) at `at` touches: in each
    node column along the press axis within `radius` of the point, the
    outermost node (the first the finger meets). The nearest column if the
    disc is narrower than a cell."""
    mesh = domain.mesh
    xyz = mesh.node_xyz
    others = [i for i in range(3) if i != axis]
    at = np.asarray(at, float)
    p2 = at[others] if len(at) == 3 else at[:2]
    d2 = np.linalg.norm(xyz[:, others] - p2, axis=1)
    near = d2 <= max(radius, 1e-9) + 1e-9
    if not near.any():
        near = d2 <= d2.min() + 1e-6
    if len(at) == 3:
        # a 3D point: only nodes on its side of the part (a press on a
        # sunk area keeps off the rim behind it)
        pass
    idx = np.flatnonzero(near)
    ijk = mesh.node_ijk[idx]
    col = ijk[:, others[0]] * (int(ijk[:, others[1]].max()) + 2) + ijk[:, others[1]]
    # the finger comes from -sign along the axis: the outermost node is the
    # one with the largest coordinate against the press direction
    key = -sign * xyz[idx, axis]
    order = np.lexsort((-key, col))
    first = np.ones(len(order), bool)
    first[1:] = col[order][1:] != col[order][:-1]
    out = np.zeros(mesh.n_nodes if hasattr(mesh, "n_nodes") else len(xyz), bool)
    out[idx[order[first]]] = True
    return out


def _unit_load(nodes: np.ndarray, axis: int, sign: int, n_dof: int, fixed: np.ndarray) -> np.ndarray:
    f = np.zeros(n_dof)
    sel = np.flatnonzero(nodes)
    f[3 * sel + axis] = sign / max(len(sel), 1)
    f[fixed] = 0.0
    return f


def _force(v, unit: str) -> float | None:
    if v is None:
        return None
    return float(v) * (GF if unit == "gf" else 1.0)


def _press_list(block: dict, unit: str) -> list[Press]:
    """`presses` as given, or a `field`: a radial gradient sampled on rings.
    field = {center, radius, rings (default 2), perRing (default 6), max:
    [at the centre, at the rim], min}."""
    out = []
    for p in block.get("presses") or []:
        out.append(Press(np.asarray(p["at"], float), _force(p.get("max"), unit), _force(p.get("min"), unit)))
    fd = block.get("field")
    if fd:
        c = np.asarray(fd["center"], float)
        r = float(fd["radius"])
        rings = int(fd.get("rings", 2))
        per = int(fd.get("perRing", 6))
        mx = fd.get("max")
        mx = [mx, mx] if isinstance(mx, (int, float)) or mx is None else list(mx)
        mn = fd.get("min")
        mn = [mn, mn] if isinstance(mn, (int, float)) or mn is None else list(mn)

        def lerp(pair, t):
            if pair[0] is None:
                return None
            return _force(pair[0] + (pair[1] - pair[0]) * t, unit)

        out.append(Press(c.copy(), lerp(mx, 0.0), lerp(mn, 0.0)))
        for k in range(1, rings + 1):
            t = k / rings
            for j in range(per):
                ang = 2 * np.pi * (j + 0.5 * (k % 2)) / per
                q = c.copy()
                q[0] += r * t * np.cos(ang)
                q[1] += r * t * np.sin(ang)
                out.append(Press(q, lerp(mx, t), lerp(mn, t)))
    if not out:
        raise ValueError("switch needs presses (a list of {at, max, min}) or a field")
    return out


def _shape(spec, study):
    from .shapes import shape_from_spec
    return shape_from_spec(spec, getattr(study, "base_dir", None))


def _first_hit(shape, pts: np.ndarray, direction: np.ndarray, reach: float, step: float):
    """Distance along `direction` from each point to `shape` (first point
    with sdf <= 0), within `reach`; inf where the ray misses, 0 where the
    point is already inside. Marched at `step`, then bisected."""
    hit = np.full(len(pts), np.inf)
    d0 = shape.sdf(pts)
    hit[d0 <= 0] = 0.0
    todo = np.flatnonzero(d0 > 0)
    lo = np.zeros(len(pts))
    t = 0.0
    while len(todo) and t < reach:
        t1 = min(t + step, reach)
        d = shape.sdf(pts[todo] + t1 * direction)
        got = d <= 0
        idx = todo[got]
        lo[idx] = t
        hit[idx] = t1
        todo = todo[~got]
        t = t1
    idx = np.flatnonzero(np.isfinite(hit) & (hit > 0))
    a, b = lo[idx], hit[idx].copy()
    for _ in range(6):
        m = 0.5 * (a + b)
        inside = shape.sdf(pts[idx] + m[:, None] * direction) <= 0
        b = np.where(inside, m, b)
        a = np.where(inside, a, m)
    hit[idx] = b
    return hit


def _clearance(study, domain: Domain, clr, pax: int, psg: int):
    """The `clearance` block: one entry or a list. Each entry is either

    - {region | shape, gap}: those nodes may move `gap` mm along the press
      (the first form: one gap for the whole region), or
    - {component, margin, region?, shape?, reach?, id?}: a component under
      the skin (a keepOut id, its shape before any insertFrom sweep, or a
      shape of its own). Each node whose line along the press direction
      meets the component within `reach` (default 10 mm) gets its own room:
      the distance to it, less `margin` (default 0.1 mm). A region or
      shape limits which nodes count. Nodes that already rest on the
      component (room under margin + 0.05 mm: a rail on the board's edge, a
      boss under it) carry it by design and are left out.

    Returns (nodes, smallest gap, gap per node, entry per node, entries)."""
    if not clr:
        return None, None, None, None, []
    items = clr if isinstance(clr, list) else [clr]
    mesh = domain.mesh
    nn = len(mesh.node_xyz)
    gap = np.full(nn, np.inf)
    entry = np.full(nn, -1)
    entries = []
    dirv = np.zeros(3)
    dirv[pax] = psg
    raw_keep = {k.get("id"): k for k in (study.raw.get("keepOut") or [])}
    for k, it in enumerate(items):
        label = f"switch.clearance[{k}]" if isinstance(clr, list) else "switch.clearance"
        filt = None
        if it.get("region") or it.get("shape"):
            cs = _shape(it["shape"], study) if "shape" in it else None
            filt = _region_nodes(study, mesh, it.get("region"), cs, label)
        if "gap" in it:
            if filt is None:
                raise ValueError(f"{label}: a fixed gap needs a region or shape")
            g = np.where(filt, float(it["gap"]), np.inf)
            rec = {"id": it.get("id") or it.get("region") or f"clearance-{k}", "gap": float(it["gap"])}
        else:
            comp = it.get("component")
            if comp is None:
                raise ValueError(f"{label}: give a gap, or a component to measure the room to")
            if isinstance(comp, str):
                if comp not in raw_keep:
                    raise ValueError(f"{label}.component: no keepOut {comp!r}; keepOuts: {sorted(raw_keep)}")
                cshape = _shape(raw_keep[comp]["shape"], study)
                cid = it.get("id") or comp
            else:
                cshape = _shape(comp, study)
                cid = it.get("id") or f"component-{k}"
            margin = float(it.get("margin", 0.1))
            reach = float(it.get("reach", 10.0))
            cand = np.flatnonzero(filt if filt is not None else np.ones(nn, bool))
            cand = cand[~domain.fixed[3 * cand + pax]]
            room = _first_hit(cshape, mesh.node_xyz[cand], dirv, reach, min(0.1, mesh.pitch / 5))
            seen = np.isfinite(room)
            touching = seen & (room <= margin + 0.05)
            use = seen & ~touching
            g = np.full(nn, np.inf)
            g[cand[use]] = room[use] - margin
            rec = {"id": cid, "margin": margin, "skippedTouching": int(touching.sum())}
        nodes = np.isfinite(g)
        if not nodes.any():
            # nothing of the part within reach over it: nothing to hold
            # (a thinner skin read in the skin review can lift every node
            # out of reach)
            rec.update({"nodes": 0, "minGap": None})
            entries.append(rec)
            continue
        rec.update({"nodes": int(nodes.sum()), "minGap": float(g[nodes].min())})
        better = g < gap
        gap[better] = g[better]
        entry[better] = k
        entries.append(rec)
    nodes = np.isfinite(gap)
    if not nodes.any():
        return None, None, None, None, entries
    return nodes, float(gap[nodes].min()), np.where(nodes, gap, np.nan), entry, entries


_PART_CACHE: dict = {}


def part_compliance(study, part: dict, axis: int, sign: int, pitch: float | None = None) -> dict:
    """The give of a separate fixed part the switch bears on: mean travel
    along the stroke of its seat per newton on it, solved on the part
    alone as cut cells (switch_lattice.readback's method, so a 1.2 mm plate
    is 1.2 mm on any grid). part = {shape, seat (shape), supports: [shape,
    ...], material? (default the study's), pitch? (default 0.4 mm)}. The
    part is held at its supports only: anything else that couples it to
    the body (a snap bead, a press fit) is left out, which reads it as
    loose as it can be."""
    import json as _json

    from scipy import ndimage

    from .bucklingopt import _factor
    from .cutcell import CutGrid, scan_voxelize
    from .domain import build_domain
    from .fem import Stiffness
    from .spec import study_from_dict

    pitch = float(pitch or part.get("pitch", 0.4))
    key = (_json.dumps(part, sort_keys=True, default=str), axis, sign, pitch, getattr(study, "base_dir", None))
    if key in _PART_CACHE:
        return _PART_CACHE[key]
    t0 = time.time()
    mat = part.get("material") or study.raw.get("material")
    doc = {"name": "bears-on", "designSpace": part["shape"],
           "preserve": [{"id": "part", "shape": part["shape"]}],
           "supports": [{"shape": s} for s in part["supports"]],
           "loadCases": [{"name": "seat", "loads": [{"shape": part["seat"], "force": [0.0, 0.0, -1.0]}]}],
           "material": mat, "resolution": {"pitch": pitch}}
    mini = study_from_dict(doc, getattr(study, "base_dir", None))
    shape = mini.design_space[0]
    dom = build_domain(mini, pitch)
    seat_shape = _shape(part["seat"], mini)
    seat = _region_nodes(mini, dom.mesh, None, seat_shape, "bearsOn.part.seat")
    import trimesh
    man = shape.to_manifold().to_mesh()
    tm = trimesh.Trimesh(np.asarray(man.vert_properties)[:, :3], np.asarray(man.tri_verts), process=False)
    nu = mini.elastic(design=True)
    E = float(mini.material.E)
    n = 4
    dims = np.asarray(dom.dims)
    fine = scan_voxelize(tm, dom.pitch / n, dom.origin, dims * n)
    cg = CutGrid(fine, n, dom.origin, dom.pitch, nu)
    lut = np.full(tuple(dims), -1, np.int64)
    lut[tuple(dom.mesh.elem_ijk.T)] = np.arange(dom.n_elem)
    ijk = cg.mesh.elem_ijk
    inb = np.all(ijk < dims, axis=1)
    el = np.full(len(ijk), -1, np.int64)
    el[inb] = lut[tuple(ijk[inb].T)]
    keep = el >= 0
    g = np.zeros(tuple(dims), bool)
    g[tuple(ijk[keep].T)] = True
    lbl, _ = ndimage.label(g, structure=np.ones((3, 3, 3), bool))
    touch = dom.support_nodes[dom.mesh.enode].any(axis=1)
    held = np.unique(lbl[tuple(dom.mesh.elem_ijk[touch].T)])
    keep &= np.isin(lbl[tuple(np.minimum(ijk, dims - 1).T)], held[held > 0])
    cg = cg.restrict(keep)
    sub = cg.mesh
    nodes = dom.mesh.node_id(sub.node_ijk)
    dofs = (3 * nodes[:, None] + np.arange(3)).ravel()
    fixed = dom.fixed[dofs]
    free = np.flatnonzero(~fixed)
    k = cg.assemble(Stiffness(sub, nu, fixed), 1.0)
    present = np.zeros(len(dom.mesh.node_xyz), bool)
    present[nodes] = True
    seat &= present
    if not seat.any():
        raise ValueError("bearsOn.part.seat touches no material of the part")
    m = _unit_load(seat, axis, sign, dom.mesh.n_dof, dom.fixed)
    solve, release = _factor(k[free][:, free].tocsr())
    x = np.asarray(solve(m[dofs][free])).ravel() / E
    if release is not None:
        release()
    u = np.zeros(dom.mesh.n_dof)
    u[dofs[free]] = x
    c = float(m @ u)
    out = {"compliance": c, "pitch": dom.pitch, "cells": int(sub.n_elem), "seatNodes": int(seat.sum()),
           "material": mini.material.name if hasattr(mini.material, "name") else mat,
           "seconds": round(time.time() - t0, 1),
           "held": "at its supports only: a snap bead or press fit to the body is left out"}
    _PART_CACHE[key] = out
    return out


def _bears_on(study, domain: Domain, bo, axis: int, sign: int, out_nodes: np.ndarray):
    """`output.bearsOn`: what the switch body sits on, so its give counts.

    The stem pushes the plunger back with the trigger force, and the
    switch body pushes whatever it sits on down with the same force. Up to
    now that was rigid ground. bearsOn = {region | shape} names the seat in
    this mesh (a preserved flap, a boss): the closure that fires the switch
    is then the plunger's travel less the seat's, r = l - m in place of l in
    every formula (F = (g + s + T r.K^-1 r) / r.K^-1 f), and the reaction
    loads both. bearsOn = {part: {...}} is a separate fixed part (a flap that
    closes the underside and stands on the desk): its seat's compliance c_b
    comes from its own solve (part_compliance), and the plunger has to go
    T c_b further: g + s + T (c + c_b). bearsOn = {compliance: mm/N} takes a
    measured one. Any of these may be combined."""
    if not bo:
        return None, 0.0, {}
    info = {}
    nodes = None
    if bo.get("region") or bo.get("shape"):
        bs = _shape(bo["shape"], study) if "shape" in bo else None
        nodes = _region_nodes(study, domain.mesh, bo.get("region"), bs, "switch.output.bearsOn")
        if (nodes & out_nodes).any():
            raise ValueError("switch.output.bearsOn: the seat shares nodes with the plunger face")
        info["seatNodes"] = int(nodes.sum())
    c = 0.0
    if bo.get("part"):
        pc = part_compliance(study, bo["part"], axis, sign)
        c += pc["compliance"]
        info["part"] = pc
    if bo.get("compliance") is not None:
        c += float(bo["compliance"])
        info["measuredCompliance"] = float(bo["compliance"])
    info["compliance"] = c
    return nodes, c, info


def switch_from_study(study, domain: Domain) -> SwitchSpec:
    """The study's `switch` block on this domain's mesh."""
    from .domain import _region_nodes
    from .shapes import shape_from_spec

    b = study.raw.get("switch")
    if not b:
        raise ValueError("the study has no `switch` block")
    unit = b.get("unit", "N")
    o = b["output"]
    shape = shape_from_spec(o["shape"]) if "shape" in o else None
    out_nodes = _region_nodes(study, domain.mesh, o.get("region"), shape, "switch.output")
    axis, sign = _axis(o.get("direction", "-z"))
    pax, psg = _axis(b.get("pressDirection", o.get("direction", "-z")))
    presses = _press_list(b, unit)
    cn, cg, cgap, cent, centries = _clearance(study, domain, b.get("clearance"), pax, psg)
    hold = []
    for h in b.get("hold") or []:
        name = h["case"]
        if name not in domain.case_names:
            raise ValueError(f"switch.hold: unknown load case {name!r}; cases: {domain.case_names}")
        hold.append((domain.case_names.index(name), float(h["maxDeflection"])))
    trig = _force(o.get("trigger", 0.0), unit)
    bn, bc, binfo = _bears_on(study, domain, o.get("bearsOn"), axis, sign, out_nodes)
    return SwitchSpec(output_nodes=out_nodes, axis=axis, sign=sign, gap=float(o.get("gap", 0.0)),
                      trigger=trig, travel=float(o.get("travel", 0.0)),
                      presses=presses, press_axis=pax, press_sign=psg,
                      press_radius=float(b.get("pressRadius", 1.0)),
                      max_travel=b.get("maxTravel"), clearance_nodes=cn, clearance=cg, hold=hold,
                      base_nodes=bn, base_compliance=bc, base_info=binfo,
                      clearance_gap=cgap, clearance_entry=cent, clearance_entries=centries)


# ---------------------------------------------------------------------------
# solves


def _have_pardiso() -> bool:
    if os.environ.get("LOADPATH_NO_PARDISO"):
        return False
    try:
        import pypardiso  # noqa: F401
    except ImportError:
        return False
    return True


class _Solves:
    """K(E) u = f for several right-hand sides at once, at unit modulus, on
    the cells that carry material.

    A push pad optimum is mostly void: at 5% of the design cells the full
    system is a million-to-one stiffness ratio and preconditioned CG
    crawled (60 s an iteration on 17k cells). As in simp._Analysis, each
    solve keeps only the cells above ALIVE (plus every preserved cell) that
    connect to a support; void cells bordering them still get
    sensitivities through the nodes they share. Up to DIRECT_DOFS unknowns
    the reduced system is factored once (SuperLU) and every right-hand side
    is a back-substitution; above that it goes to the matrix-free
    multigrid (mg.py; one hierarchy for every right-hand side, a GPU with
    LOADPATH_DEVICE)."""

    ALIVE = 0.02
    DIRECT_DOFS = 600_000
    MMD_DOFS = 120_000
    # PARDISO up to DIRECT_DOFS: a grey 0.7 mm handheld interior (cooling
    # and a back-panel press, 469k dofs) took 94 s and 6.6 GB on the
    # single-precision fallback and 20 s and 4.8 GB on PARDISO; further on
    # into the run the fallback ran a 15 GB box out of memory
    PARDISO_DOFS = 600_000

    def __init__(self, domain: Domain, nu, rtol: float = 1e-8):
        from scipy import ndimage  # noqa: F401
        mesh = domain.mesh
        self.domain = domain
        self.nu = nu
        self.rtol = rtol
        self.support_touch = domain.support_nodes[mesh.enode].any(axis=1)
        self._lu = None
        self._release = None
        self.last_cells = mesh.n_elem
        self._mg = None
        self.prev = {}

    def _alive(self, xp):
        from scipy import ndimage
        d = self.domain
        mesh = d.mesh
        alive = (xp > self.ALIVE) | d.preserve
        g = np.zeros(d.dims, bool)
        ijk = mesh.elem_ijk
        g[tuple(ijk[alive].T)] = True
        lbl, _ = ndimage.label(g, structure=np.ones((3, 3, 3), bool))
        el = lbl[tuple(ijk.T)]
        keep = np.unique(el[alive & self.support_touch])
        return alive & np.isin(el, keep[keep > 0])

    def release(self):
        """Free the last PARDISO factor (MKL holds it outside Python: the
        skin review's dozens of refactors of the touch microphone ran a
        15 GB box out of memory before this)."""
        rel = getattr(self, "_release", None)
        self._release = None
        if rel is not None:
            rel()

    def __del__(self):
        try:
            self.release()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def factor(self, xp: np.ndarray):
        """Set up the reduced system for density xp (unit modulus)."""
        self.release()
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu

        from .fem import HexMesh, Stiffness
        from .simp import E_MIN, PENALTY
        d = self.domain
        mesh = d.mesh
        alive = self._alive(xp)
        el = np.flatnonzero(alive)
        self.last_cells = len(el)
        e = _modulus(xp[el], E_MIN, PENALTY)
        sub = HexMesh(mesh.elem_ijk[el], mesh.pitch, mesh.origin)
        sk = d.skin
        local = None
        if sk is not None:
            # skin cells: E_MIN plus their own matrices (_SkinCells)
            pos = np.full(mesh.n_elem, -1)
            pos[el] = np.arange(len(el))
            on = pos[sk.elem] >= 0
            local = pos[sk.elem[on]]
            t, scale, _, _ = sk.stiffness_scale(xp)
            mats = (t[on, None, None] * sk.ks[on] + scale[on, None, None] * sk.dk[on]) * (1 - E_MIN)
            e[local] = E_MIN
        nodes = mesh.node_id(sub.node_ijk)
        dofs = (3 * nodes[:, None] + np.arange(3)).ravel()
        fixed = d.fixed[dofs]
        free = np.flatnonzero(~fixed)
        self.dofs, self.free = dofs, free
        self.dofs_fixed_local = np.flatnonzero(fixed)
        if len(free) <= self.DIRECT_DOFS:
            st = Stiffness(sub, self.nu, fixed)
            k = st.assemble(e)
            if local is not None and len(local):
                data = np.zeros(st.nnz)
                st.add_elements(data, mats, local)
                k = (k + st.finish(data, diag=0.0)).tocsr()
            if d.springs is not None:
                k = (k + d.springs[dofs][:, dofs]).tocsr()
            if self._release is not None:
                self._release()
                self._release = None
            if local is not None and len(local):
                # cut skin cells: PARDISO (SuperLU's diagonal pivoting
                # fills in on cells the skin only partly holds)
                from .bucklingopt import _factor
                k = k.tocsr()
                if fixed.any():
                    keep = sp.diags((~fixed).astype(float))
                    k = (keep @ k @ keep + sp.diags(fixed.astype(float))).tocsr()
                solve, self._release = _factor(k, sub.node_ijk)
                self._lu = _Fn(solve)
                self._whole = True
            elif _have_pardiso() and len(free) <= self.PARDISO_DOFS:
                # PARDISO (bucklingopt._factor) wherever it is installed:
                # the microphone body's robust feel targets factor six
                # systems of ~90k dofs an iteration, 524 s with SuperLU's
                # minimum degree, 22 s with it
                from .bucklingopt import _factor
                solve, self._release = _factor(k[free][:, free].tocsr(), sub.node_ijk)
                self._lu = _Fn(solve)
                self._whole = False
            elif len(free) <= self.MMD_DOFS:
                a = k[free][:, free].tocsc()
                self._lu = splu(a, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
                self._whole = False
            else:
                # nested dissection and a single precision factor refined by
                # CG (nonlinear._MixedLU): minimum degree on the 0.7 mm
                # touch-mic pad's 200k dofs would not fit, and multigrid on
                # a mostly void design does not converge
                from .nonlinear import _MixedLU
                k = k.tocsr()
                if fixed.any():
                    keep = sp.diags((~fixed).astype(float))
                    k = (keep @ k @ keep + sp.diags(fixed.astype(float))).tocsr()
                self._lu = _MixedLU(k, sub.node_ijk)
                self._whole = True
            self._mg = None
        else:
            from .mg import GridElasticity
            full = np.zeros(mesh.n_elem)
            full[el] = _modulus(xp[el], E_MIN, PENALTY)      # 0 drops a cell outright
            # (shared-matrix multigrid: skin cells keep the E ~ phi^3 floor)
            if self._mg is None:
                self._mg = GridElasticity(mesh, self.nu, d.fixed, springs=d.springs, rtol=self.rtol, maxiter=2000)
            self._mg.set_moduli(full)
            self._lu = None

    def solve(self, fs: list, tag: str = "f") -> list:
        n = self.domain.mesh.n_dof
        if self._lu is not None:
            idx = self.dofs if self._whole else self.dofs[self.free]
            b = np.stack([np.asarray(f)[idx] for f in fs], axis=1)
            if self._whole:
                b[self.dofs_fixed_local] = 0.0
            x = self._lu.solve(b)
            if x.ndim == 1:
                x = x[:, None]
            out = []
            for j in range(len(fs)):
                u = np.zeros(n)
                u[idx] = x[:, j]
                out.append(u)
            return out
        out = self._mg.solve_many(fs, self.prev.get(tag))
        self.prev[tag] = [np.array(u) for u in out]
        return [np.asarray(u) for u in out]


class _Fn:
    def __init__(self, f):
        self.solve = f


def _pair(edof, ke, v, w):
    """Per element v_e . ke w_e."""
    return np.einsum("ij,ij->i", v[edof], w[edof] @ ke)


@dataclass
class _Problem:
    spec: SwitchSpec
    loads: list           # unit press loads
    l: np.ndarray         # unit plunger load
    weights: list         # press weights (f_i itself, for the press travel)
    hold: list            # [(force vector, |F|, limit)]
    clr_dofs: np.ndarray | None
    clr_gap: np.ndarray | None = None     # mm of room per clearance dof
    clr_entry: np.ndarray | None = None   # clearance entry per clearance dof


def _problem(domain: Domain, spec: SwitchSpec) -> _Problem:
    mesh = domain.mesh
    loads = []
    for p in spec.presses:
        nodes = _press_nodes(domain, p.at, spec.press_axis, spec.press_sign, spec.press_radius)
        loads.append(_unit_load(nodes, spec.press_axis, spec.press_sign, mesh.n_dof, domain.fixed))
    l = _unit_load(spec.output_nodes, spec.axis, spec.sign, mesh.n_dof, domain.fixed)
    if not np.abs(l).sum():
        raise ValueError("switch.output: every plunger node is held by a support")
    if spec.base_nodes is not None:
        # the switch's seat in this mesh: the closure is the plunger's
        # travel less the seat's, and the stem's reaction loads both
        m = _unit_load(spec.base_nodes, spec.axis, spec.sign, mesh.n_dof, domain.fixed)
        l = l - m
    hold = []
    for ci, lim in spec.hold:
        f = domain.loads[ci]
        # force-weighted mean travel along each nodal force: f . u / sum |f_node|
        # (a squeeze's two sides cancel in the net force)
        hold.append((f, float(np.linalg.norm(f.reshape(-1, 3), axis=1).sum()), lim))
    cd, cg, ce = None, None, None
    if spec.clearance_nodes is not None:
        nodes = np.flatnonzero(spec.clearance_nodes)
        cd = 3 * nodes + spec.press_axis
        keep = ~domain.fixed[cd]
        cd = cd[keep]
        if spec.clearance_gap is not None:
            cg = spec.clearance_gap[nodes[keep]]
            ce = spec.clearance_entry[nodes[keep]]
        else:
            cg = np.full(len(cd), float(spec.clearance))
            ce = np.zeros(len(cd), int)
    return _Problem(spec, loads, l, loads, hold, cd, cg, ce)


def _modulus(xp, E_MIN, PENALTY):
    return E_MIN + xp ** PENALTY * (1 - E_MIN)


def _dpair(domain: Domain, nu, xp, modulus: float, solves: _Solves):
    """d(v . K^-1 w) / dxp as a function of the solutions (v, w) =
    (K^-1 a, K^-1 b): -(v_e dK_e/dxp w_e), with K at modulus E (the stored
    solutions already carry 1/E). Skin cells read as cut cells
    (domain.skin, _SkinCells) have their own dK; the multigrid path (past
    DIRECT_DOFS) reads them by the floor."""
    from .simp import E_MIN, PENALTY
    mesh = domain.mesh
    edof = mesh.edof
    ke = hex8_ke(nu) * mesh.pitch
    dEdx = PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)   # unit modulus
    sk = domain.skin if solves._lu is not None else None
    if sk is not None:
        _, _, dtdx, dsdx = sk.stiffness_scale(xp)
        dsdx, dtdx = dsdx * (1 - E_MIN), dtdx * (1 - E_MIN)
        dEdx = dEdx.copy()
        dEdx[sk.elem] = 0.0
        under = dtdx != 0

    def dpair(v, w):
        g = -modulus * dEdx * _pair(edof, ke, v, w)
        if sk is not None:
            g[sk.elem] = -modulus * dsdx * sk.pair(edof, v, w)
            if under.any():
                de = edof[sk.elem[under]]
                g[sk.elem[under]] = -modulus * dtdx[under] * np.einsum("ei,eij,ej->e", v[de], sk.ks[under], w[de])
        return g
    return dpair


def _evaluate(domain: Domain, nu, prob: _Problem, xp: np.ndarray, modulus: float, solves: _Solves,
              grads: bool = True):
    """Fire forces and the constraint quantities for density xp (per
    element, preserve included), and their gradients wrt xp. Moduli in
    MPa: the solves run at unit modulus and displacements divide by E."""
    mesh = domain.mesh
    spec = prob.spec
    solves.factor(xp)
    rhs = list(prob.loads) + [prob.l] + [f for f, _, _ in prob.hold]
    sol = solves.solve(rhs, "f")
    nP = len(prob.loads)
    U = [np.asarray(u) / modulus for u in sol[:nP]]
    lam = np.asarray(sol[nP]) / modulus
    UH = [np.asarray(u) / modulus for u in sol[nP + 1:]]
    a = np.array([prob.l @ u for u in U])
    b = np.array([f @ u for f, u in zip(prob.loads, U)])
    c = float(prob.l @ lam)
    N = spec.stroke + spec.trigger * c
    a_safe = np.maximum(a, 1e-12)
    F = N / a_safe
    d = b * F - a * spec.trigger
    out = {"a": a, "b": b, "c": c, "F": F, "travel": d, "U": U, "lam": lam}
    hold_vals = [float(f @ u) / max(nf, 1e-30) for (f, nf, _), u in zip(prob.hold, UH)]
    out["hold"] = np.array(hold_vals)
    clr_q = None
    if prob.clr_dofs is not None:
        # displacement along the press direction at the clearance nodes,
        # under each press at its fire force (the switch's reaction would
        # lift them a little: conservative), or at its window's limit when
        # it fires beyond it: a press over a rigid part (a cell, a boss)
        # barely moves the plunger, so its fire force runs to millions of N,
        # and what the skin must clear is the hand at the window, not that
        cap = np.array([p.max if p.max is not None else (p.min if p.min is not None else np.inf)
                        for p in spec.presses], float)
        Fc = np.minimum(F, cap)
        out["clearanceForce"] = Fc
        clr_q = [spec.press_sign * u[prob.clr_dofs] * Fi for u, Fi in zip(U, Fc)]
        out["clearanceMax"] = float(max(q.max() for q in clr_q))
        out["clearance"] = clearance_rows(spec, prob, clr_q)
    if not grads:
        return out, None
    dpair = _dpair(domain, nu, xp, modulus, solves)

    dc = dpair(lam, lam)
    G = {"logF": [], "travel": [], "hold": [], "clearance": None}
    dF = []
    for i in range(nP):
        da = dpair(lam, U[i])
        dlogF = spec.trigger * dc / N - da / a_safe[i]
        G["logF"].append(dlogF)
        dF.append(F[i] * dlogF)
        if spec.max_travel is not None:
            db = dpair(U[i], U[i])
            G["travel"].append(db * F[i] + b[i] * dF[i] - da * spec.trigger)
    for (f, nf, _), u in zip(prob.hold, UH):
        G["hold"].append(dpair(u, u) / max(nf, 1e-30))
    if clr_q is not None:
        # KS over every press and node of q / gap
        rho = 40.0
        qs = np.concatenate([q / prob.clr_gap for q in clr_q])
        m = qs.max()
        w = np.exp(rho * (qs - m))
        ks = m + np.log(w.sum()) / rho
        w /= w.sum()
        out["clearanceKS"] = float(ks)
        # adjoint per press: K mu_i = sum_k w_ik F_i s e_k / gap
        n = len(prob.clr_dofs)
        adj = []
        for i in range(nP):
            r = np.zeros(mesh.n_dof)
            wi = w[i * n:(i + 1) * n]
            r[prob.clr_dofs] = wi * Fc[i] * spec.press_sign / prob.clr_gap
            adj.append(r)
        mus = solves.solve(adj, "clr")
        g = np.zeros(mesh.n_elem)
        for i in range(nP):
            wi = w[i * n:(i + 1) * n]
            mu = np.asarray(mus[i]) / modulus
            g += dpair(mu, U[i])
            zi = spec.press_sign * U[i][prob.clr_dofs] / prob.clr_gap
            if F[i] < cap[i]:       # capped: the force is fixed, only the skin's give moves
                g += float(wi @ zi) * dF[i]
        G["clearance"] = g
    return out, G


def clearance_rows(spec: SwitchSpec, prob: _Problem, clr_q: list) -> list[dict]:
    """Per clearance entry: the most any of its nodes moves toward what is
    under it at a press's fire force, its room there, and the worst share of
    room used (over 1: it touches)."""
    rows = []
    q = np.stack(clr_q)                     # presses x dofs
    ratio = q / prob.clr_gap
    names = spec.clearance_entries or [{"id": "clearance"}]
    for k, rec in enumerate(names):
        sel = prob.clr_entry == k
        if not sel.any():
            rows.append({"id": rec.get("id"), "nodes": 0, "ok": True})
            continue
        r = ratio[:, sel]
        i, j = np.unravel_index(int(np.argmax(r)), r.shape)
        rows.append({"id": rec.get("id"), "nodes": int(sel.sum()), "minGap": float(prob.clr_gap[sel].min()),
                     "maxTravel": float(q[:, sel].max()), "worstShare": float(r[i, j]),
                     "worstAtPress": int(i), "worstTravel": float(q[:, sel][i, j]),
                     "worstGap": float(prob.clr_gap[sel][j]), "ok": bool(r.max() <= 1.0)})
    return rows


def fire_forces(domain: Domain, nu, spec: SwitchSpec, xphys: np.ndarray, modulus: float) -> dict:
    """The linear fire force and press travel at the click for each press
    point, for a given density (1 = solid) on this domain."""
    prob = _problem(domain, spec)
    out, _ = _evaluate(domain, nu, prob, np.asarray(xphys, float), modulus, _Solves(domain, nu), grads=False)
    return {"forces": out["F"], "travel": out["travel"], "plungerCompliance": out["c"],
            "hold": out["hold"], **({"clearanceMax": out["clearanceMax"], "clearance": out["clearance"]}
                                    if "clearanceMax" in out else {})}


# ---------------------------------------------------------------------------
# optimizer


def _interp_design(old: Domain, x_old: np.ndarray, new: Domain) -> np.ndarray:
    """A design on `old`'s grid read at `new`'s cell centres (trilinear,
    cells outside the old mesh take their nearest old cell)."""
    from scipy import ndimage
    g = np.full(old.dims, np.nan)
    ijk = old.mesh.elem_ijk
    g[tuple(ijk.T)] = x_old
    miss = np.isnan(g)
    if miss.any():
        idx = ndimage.distance_transform_edt(miss, return_distances=False, return_indices=True)
        g = g[tuple(idx)]
    c = (new.mesh.elem_centers() - old.origin) / old.pitch - 0.5
    return ndimage.map_coordinates(g, c.T, order=1, mode="nearest")


@dataclass
class _SkinCells:
    """A thin skin's cells as cut cells: elem (element indices), phi (skin
    volume share), ks (24x24 per cell, unit modulus, side h: the skin's own
    stiffness, integrated over 4^3 sub-cells with the skin's share of each),
    dk = k_solid - ks (what filling the rest of the cell adds)."""
    elem: np.ndarray
    phi: np.ndarray
    ks: np.ndarray
    dk: np.ndarray
    # skin review: at x = phi, the derivative of thinning the skin (left)
    # rather than of adding structure over it (right, 0)
    left: bool = False

    def stiffness_scale(self, xp):
        """(t, s, dt/dxp, ds/dxp) per skin cell: K_e = t ks + s dk. Over the
        skin (x >= phi) t = 1 and s = extra^3, extra = (x - phi) / (1 - phi)
        the structure added over it (see _as_printed); under it (a thinner
        skin, skinreview._rescale) s = 0 and t = (x / phi)^3, bending as
        thickness cubed."""
        x = np.asarray(xp, float)[self.elem]
        phi = np.maximum(self.phi, 1e-9)
        span = np.maximum(1.0 - self.phi, 1e-9)
        ex = np.clip((x - self.phi) / span, 0.0, 1.0)
        r = np.clip(x / phi, 0.0, 1.0)
        under = x < self.phi if not self.left else x <= self.phi
        ds = np.where((x > self.phi) & (ex < 1.0), 3.0 * ex ** 2 / span, 0.0)
        dt = np.where(under, 3.0 * r ** 2 / phi, 0.0)
        return r ** 3, ex ** 3, dt, ds

    def pair(self, edof, v, w, which="dk"):
        """Per skin cell v_e . dk w_e (or ks)."""
        de = edof[self.elem]
        return np.einsum("ei,eij,ej->e", v[de], getattr(self, which), w[de])


CUT_ALPHA = 1e-4
SKIN_SUB = int(__import__("os").environ.get("LOADPATH_SKIN_SUB", 4))


def skin_model() -> str:
    """LOADPATH_SKIN_MODEL: "cut" (default) or "floor" (the density floor
    alone, for comparison)."""
    import os
    m = os.environ.get("LOADPATH_SKIN_MODEL", "cut").strip().lower() or "cut"
    if m not in ("cut", "floor"):
        raise ValueError(f"LOADPATH_SKIN_MODEL must be cut or floor, not {m!r}")
    return m


def _thin_skin(study, domain: Domain):
    """A skin thinner than two cells on this grid, as cut cells.

    The preserved skin is whole cells: at 1 mm cells a 0.7 mm skin is a
    1 mm (or 2 mm) one, and bending goes as thickness cubed, so the click
    came out three times too stiff on a coarse grid and nothing the
    optimizer added could bring it down. Here each cell the skin passes
    through gets its share of skin volume phi as a floor on its density,
    and the rest of the cell is design space.

    The stiffness of such a cell is the skin's own, integrated over 4^3
    sub-cells weighted by the skin's share of each (from its signed
    distance, so a skin's thickness is not rounded to a sub-cell), plus
    what the optimizer adds over it (_SkinCells). That knows where in the
    cell the skin sits. The earlier stand-in, E ~ phi^3 per cell, bent like
    the skin only when the skin sat in one cell layer; split across two it
    added two thin layers' bending instead of one solid layer's and read
    20-35% soft (skinreview.SPLIT_SKIN). LOADPATH_SKIN_MODEL=floor keeps
    that model. From two cells per skin thickness up the skin is plain
    preserved cells.

    Returns (domain with those cells as design and domain.skin set, floor
    per element)."""
    import dataclasses

    floor = np.zeros(domain.n_elem)
    thin = [(rid, t) for rid, t in study.shells if rid == "skin" and t < 2.0 * domain.pitch]
    if not thin:
        return domain, floor
    shape, _ = study.region_shape("skin")
    c = domain.mesh.elem_centers()
    n = SKIN_SUB
    sub = (np.arange(n) + 0.5) / n - 0.5
    offs = np.array([[a, b, d] for a in sub for b in sub for d in sub]) * domain.pitch
    # candidate cells: any of the skin within a cell's circumradius
    near = shape.sdf(c) < 0.87 * domain.pitch
    idx = np.flatnonzero(near)
    hs = domain.pitch / n
    ind = np.zeros((len(idx), len(offs)))
    for k, o in enumerate(offs):
        # the skin's share of each sub-cell, from its signed distance
        ind[:, k] = np.clip(0.5 - shape.sdf(c[idx] + o) / hs, 0.0, 1.0)
    phi = np.zeros(len(c))
    phi[idx] = ind.mean(axis=1)
    # other preserves (the plunger, bosses) stay whole cells
    others = [r.shape for r in study.preserve if r.id != "skin"]
    hard = np.zeros(len(c), bool)
    if others:
        from .domain import _fraction
        hard = _fraction(others, c, domain.pitch) >= 0.5
    skin = (phi > 1e-3) & ~hard
    floor[skin] = np.clip(phi[skin], 0.0, 1.0)
    pres = domain.preserve & ~skin
    cells = None
    if skin_model() == "cut":
        from .cutcell import MIN_FILL, SubcellRule
        # slivers stay on the floor law (as cutcell leaves them out): a cell
        # with a corner of skin has nodes it barely holds, and the factor
        # pivots off the diagonal and fills in
        el = np.flatnonzero(skin & (phi >= MIN_FILL))
        pos = np.full(len(c), -1)
        pos[idx] = np.arange(len(idx))
        nu = study.elastic(design=True)
        rule = SubcellRule(nu, n)
        k0 = hex8_ke(nu)
        ks = np.empty((len(el), 24, 24))
        for s in range(0, len(el), 4000):
            sl = slice(s, s + 4000)
            ks[sl] = rule.matrices(ind[pos[el[sl]]])[0]
        # a small share of a whole cell's stiffness under every cut cell
        # (finite cell alpha method) keeps the nodes a partial cell barely
        # holds conditioned; ~0.01-0.5% of the skin's own stiffness
        ks = ks + CUT_ALPHA * k0[None]
        cells = _SkinCells(el, floor[el], ks * domain.pitch, (k0[None] - ks) * domain.pitch)
    return dataclasses.replace(domain, preserve=pres, skin=cells), floor


RELAX_MARGIN = 0.02


def _relax_skin_limited(domain, nu, spec, floor, modulus):
    """(spec the optimizer works to, the ceilings it raised) for a ceiling
    the bare skin alone is over (skin, preserves, nothing designed).

    No interior can bring such a press under its ceiling. Kept as it is,
    the constraint stays violated all run and pulls the design: on the
    touch microphone four rim corners over 300 gf on the bare skin had the
    optimizer build a frame round the plunger. It brought the corners in by
    a few percent and put the centre over its own ceiling, 9 of 14 against
    the bare skin's 10. So the optimizer works to the bare skin's own force
    there, plus RELAX_MARGIN: no worse than the skin. The report still
    reads every press against its real window, and the skin review says
    what the designer could change."""
    import dataclasses
    bare = np.maximum(np.maximum(domain.preserve.astype(float), floor), 1e-9)
    f = fire_forces(domain, nu, spec, bare, modulus)["forces"]
    presses, relaxed = [], []
    for i, (p, fi) in enumerate(zip(spec.presses, f)):
        if p.max is not None and fi > p.max * (1 + RELAX_MARGIN):
            new = float(fi * (1 + RELAX_MARGIN))
            relaxed.append({"press": i, "at": p.at.tolist(), "ceilingGf": p.max / GF, "bareSkinGf": float(fi / GF),
                            "workedToGf": new / GF})
            p = dataclasses.replace(p, max=new)
        presses.append(p)
    if not relaxed:
        return spec, []
    return dataclasses.replace(spec, presses=presses), relaxed


def optimize_switch(study, pitches=None, iterations=None, x0: float = 0.5, beta_max: float = 16.0,
                    beta_every: int = 15, move: float = 0.1, on_iter=None, tol: float = 0.003,
                    providers=None, move_sharp: float | None = None) -> SwitchResult:
    """Minimize volume subject to the study's `switch` block (see the module
    docstring), on each pitch in turn (coarse to fine). Returns the last
    level's design, its fire forces and the per-level summaries.

    providers: more constraints on the same design, each a factory
    `make(study, domain, nu, modulus)` called once per grid that returns an
    object with `m` (how many constraints) and `evaluate(xp, ctx)` -> a
    list of (value, d value / d xp per element, name), feasible at value
    <= 0, or (value, d value / d xt, name, "xt") for a provider that
    projects the filtered design itself (ctx: xt, beta, floor, preserve). The
    feel targets (feel.py) come in this way. With providers the study may
    have no `switch` block at all.

    move_sharp: the move limit once the projection is at full sharpness
    (default: `move` throughout). At beta 16 a 0.1 step in x is nearly a
    whole cell switching, and the microphone body's run ended still
    swinging its drop stiffness 30% either side of the ceiling."""
    from .domain import build_domain
    from .mma import MMA
    from .simp import _dproject, _project, density_filter, filter_radius

    t0 = time.time()
    b = study.raw.get("switch") or {}
    pitches = list(pitches or b.get("pitches") or [study.pitch or None])
    iters = iterations or b.get("iterations") or [80] + [30] * (len(pitches) - 1)
    if isinstance(iters, int):
        iters = [iters] * len(pitches)
    nu = study.elastic(design=True)
    modulus = float(study.material.E)
    history, levels = [], []
    prev_dom, prev_x = None, None
    beta = 1.0
    it_total = 0
    relaxed = []
    for li, pitch in enumerate(pitches):
        domain, floor = _thin_skin(study, build_domain(study, pitch))
        spec = switch_from_study(study, domain) if b else None
        spec_target = spec
        if spec is not None:
            spec, relaxed = _relax_skin_limited(domain, nu, spec, floor, modulus)
        prob = _problem(domain, spec) if b else None
        provs = [make(study, domain, nu, modulus) for make in providers or []]
        mesh = domain.mesh
        n = mesh.n_elem
        pres = domain.preserve
        design = ~pres
        radius = filter_radius(domain, study.length_scale, study.min_hole)
        h, hs = density_filter(domain, radius)
        ht = h.T
        solves = _Solves(domain, nu)
        if prev_dom is None:
            x = np.full(n, float(x0))
        else:
            x = np.clip(_interp_design(prev_dom, prev_x, domain), 0.0, 1.0)
            beta = max(1.0, beta / 2)
        x[pres] = 1.0
        x = np.maximum(x, floor)
        m = sum(pv.m for pv in provs)
        if spec is not None:
            nP = len(spec.presses)
            m += sum(p.max is not None for p in spec.presses) + sum(p.min is not None for p in spec.presses) + \
                (nP if spec.max_travel is not None else 0) + len(prob.hold) + (1 if prob.clr_dofs is not None else 0)
        mma = MMA(int(design.sum()), max(m, 1), xmin=0.0, xmax=1.0, move=move)
        since = 0
        nd = float(design.sum())
        for it in range(1, int(iters[li]) + 1):
            it_total += 1
            xt = (h @ x) / hs
            xp = _project(xt, beta, 0.5)
            xp[pres] = 1.0
            dp = _dproject(xt, beta, 0.5)
            beta_seen = beta
            if move_sharp is not None:
                mma.move = move_sharp if beta >= beta_max else move
            held = floor > xp
            xp[held] = floor[held]
            dp = np.where(held, 0.0, dp)
            cons, dcons, names = [], [], []

            def add(val, grad, name, wrt="xp"):
                # grad is d val / d xp (the projected design), or with
                # wrt="xt" already d val / d xt (the filtered one: a provider
                # that projects xt its own way, feel.py's robust designs)
                g = ht @ (((grad * dp) if wrt == "xp" else grad) / hs)
                cons.append(val)
                dcons.append(g[design])
                names.append(name)

            F = np.zeros(0)
            if spec is not None:
                out, G = _evaluate(domain, nu, prob, xp, modulus, solves)
                F = out["F"]
                for i, p in enumerate(spec.presses):
                    if p.max is not None:
                        add(float(np.log(F[i] / p.max)), G["logF"][i], f"max{i}")
                    if p.min is not None:
                        add(float(np.log(p.min / F[i])), -G["logF"][i], f"min{i}")
                if spec.max_travel is not None:
                    for i in range(nP):
                        add(out["travel"][i] / spec.max_travel - 1.0, G["travel"][i] / spec.max_travel,
                            f"travel{i}")
                for k, (_, _, lim) in enumerate(prob.hold):
                    add(out["hold"][k] / lim - 1.0, G["hold"][k] / lim, f"hold{k}")
                if G["clearance"] is not None:
                    add(out["clearanceKS"] - 1.0, G["clearance"], "clearance")
            for pv in provs:
                ctx = {"xt": xt, "beta": beta, "floor": floor, "preserve": pres}
                for item in pv.evaluate(xp, ctx):
                    add(float(item[0]), *item[1:])
            vol = float(xp[design].sum()) / nd
            dvol = (ht @ (dp / hs))[design] / nd
            if not cons:
                cons, dcons = [vol - 1.0], [dvol * 0]
            x_seen = x.copy()
            xn = x.copy()
            xn[design] = mma.update(x[design], vol * 10.0, dvol * 10.0, np.array(cons), np.array(dcons))
            change = float(np.abs(xn - x).max())
            x = xn
            worst = float(max(cons)) if cons else 0.0
            worst_name = names[int(np.argmax(cons))] if names else None
            rec = {"level": li, "pitch": domain.pitch, "iteration": it, "volume": vol, "forces": F.tolist(),
                   "worstViolation": worst, "worst": worst_name, "beta": beta, "change": change, "seconds": round(time.time() - t0, 1)}
            history.append(rec)
            if on_iter is not None:
                on_iter(rec, xp)
            since += 1
            if beta < beta_max and (since >= beta_every or (change < 2 * tol and since >= 5)):
                beta, since = min(2 * beta, beta_max), 0
            elif beta >= beta_max and change < tol and worst < 0.01:
                break
        # The design read and returned is the last one the constraints were
        # evaluated on: at the iteration cap the step after it has not been
        # checked, and with thin suspensions at full sharpening one step can
        # cut a member (the microphone body's capsule mount came loose so).
        x, beta = x_seen, beta_seen
        xt = (h @ x) / hs
        xp = _project(xt, beta, 0.5)
        xp[pres] = 1.0
        xp = np.maximum(xp, floor)
        out = {"F": np.zeros(0), "travel": np.zeros(0)}
        if spec is not None:
            out, _ = _evaluate(domain, nu, prob, xp, modulus, solves, grads=False)
        levels.append({"pitch": domain.pitch, "cells": int(n), "iterations": it, "volume": float(xp[design].mean()),
                       "forces": out["F"].tolist(), "seconds": round(time.time() - t0, 1)})
        prev_dom, prev_x = domain, x
    if spec is None:
        return SwitchResult(x=x, xphys=xp, domain=prev_dom, floor=floor, forces=out["F"], travel=out["travel"],
                            volume=float(xp[~prev_dom.preserve].mean()), iterations=it_total,
                            seconds=time.time() - t0, history=history, levels=levels, report={})
    rep = {"presses": [{"at": p.at.tolist(), "max": p.max, "min": p.min, "force": float(f), "forceGf": float(f / GF),
                        "travel": float(d),
                        "ok": bool((p.max is None or f <= p.max * 1.02) and (p.min is None or f >= p.min * 0.98))}
                       for p, f, d in zip(spec_target.presses, out["F"], out["travel"])],
           "plungerCompliance": out["c"], "hold": out["hold"].tolist()}
    if relaxed:
        rep["relaxedCeilings"] = relaxed
    if "clearanceMax" in out:
        rep["clearanceMax"] = out["clearanceMax"]
        rep["clearance"] = out["clearance"]
    if spec.base_info:
        rep["bearsOn"] = {**spec.base_info, "giveAtTrigger": spec.trigger * spec.base_compliance}
    return SwitchResult(x=x, xphys=xp, domain=prev_dom, floor=floor, forces=out["F"], travel=out["travel"],
                        volume=float(xp[~prev_dom.preserve].mean()), iterations=it_total, seconds=time.time() - t0,
                        history=history, levels=levels, report=rep)


THIN_KEEP = 0.2


def check_fire_nonlinear(domain: Domain, nu, spec: SwitchSpec, xphys: np.ndarray, modulus: float,
                         presses=None, threshold: float = 0.5, steps: int = 8, material: str = "neo-hookean",
                         tol: float = 0.01, max_solves: int = 5, floor: np.ndarray | None = None) -> list[dict]:
    """Re-read the fire force of the solid cells (xphys > threshold) with
    large deflections: TPU skins stiffen as they bow, so the linear reading
    is low where the skin does the work.

    At the click the plunger has moved gap + travel along the stroke and
    the stem pushes back with the trigger force. For each press the solve
    applies P f_i - T l (the press, and the stem's reaction on the plunger
    face) and finds P where the plunger's travel reaches gap + travel, by
    secant steps from the linear answer on the same solid mesh. Elastic, so
    the final state does not depend on the order the two loads came in
    (the stem is not modelled as an obstacle on the way there; a snap
    before the click shows up as a non-converged step).

    floor: a thin skin's density floor (_thin_skin): those cells are kept
    below the threshold too, at modulus E x^3, so a coarse grid's skin
    bends as thin as it is.

    presses: indices to check (default all). Returns per press: linear and
    nonlinear fire force (N), their ratio, press travel at the click,
    largest strain, and with a clearance region its largest travel."""
    from .fem import Solver, Stiffness
    from .nonlinear import solid_submesh, solve_nonlinear

    prob = _problem(domain, spec)
    xphys = np.asarray(xphys, float)
    keep = xphys > threshold
    emod = np.full(len(xphys), float(modulus))
    if floor is not None:
        # a sliver of skin in a cell (phi under ~0.2, modulus under 1% of E)
        # adds nothing but a near-singular element to the Newton solve: the
        # first version kept them and the solve broke down (NaN) or crawled
        # Every floored cell is read at E x^3, like the linear reading: a
        # cell more than half skin is still that share of a cell, not a
        # solid one (thresholding it made the skin read 1.2-1.4x stiff on
        # the touch microphone before any large deflection)
        thin = (floor > 0) & (xphys >= THIN_KEEP)
        keep = (keep & ~(floor > 0)) | thin
        emod[thin] = modulus * np.minimum(xphys[thin], 1.0) ** 3
    target = spec.stroke
    # With cut-cell skin cells (domain.skin) the Newton solve still reads
    # them at E x^3 (it takes one modulus per cell): its large-deflection
    # ratio is applied to the cut-cell linear reading
    f_cut = None
    if floor is not None and domain.skin is not None:
        f_cut = fire_forces(domain, nu, spec, np.maximum(xphys, 1e-9), modulus)["forces"]
    clr = np.zeros(domain.mesh.n_dof)
    if prob.clr_dofs is not None:
        clr[prob.clr_dofs] = 1.0
    out = []
    for i in (range(len(prob.loads)) if presses is None else presses):
        sub, el, fx, (f, l, cm) = solid_submesh(domain.mesh, keep, domain.fixed, [prob.loads[i], prob.l, clr])
        if not np.abs(f).sum() or not np.abs(l).sum():
            out.append({"press": int(i), "error": "the press or the plunger is not on the solid part"})
            continue
        f = f / np.abs(f).sum()
        l = l / np.abs(l).sum()
        fx_flat = fx.ravel() if fx.ndim > 1 else fx
        mod_sub = emod[el]
        k = Stiffness(sub, nu, fx_flat).assemble(mod_sub)
        sv = Solver(sub, rtol=1e-8)
        try:
            u, lam = sv.solve_many(k, [f, l])
        except RuntimeError as exc:
            # a sliver held on only by floored skin cells: the linear
            # reading (cut cells) stands for this press
            out.append({"press": int(i), "at": spec.presses[i].at.tolist(),
                        "error": f"linear solve on the solid part failed: {exc}"})
            continue
        a, c = float(l @ u), float(l @ lam)
        f_lin = (target + spec.trigger * c) / max(a, 1e-12)

        def tip(P, start=None):
            load = P * f - spec.trigger * l
            if start is not None:
                # from the last converged press: a load 10-30% off takes a
                # step or two, not the whole ramp (elastic: the same answer)
                r = solve_nonlinear(sub, nu, mod_sub, fx_flat, load, material=material, steps=1, max_step=1.0,
                                    probes={"tip": l}, start=start)
                if r.converged:
                    return float(l @ r.u), r
            r = solve_nonlinear(sub, nu, mod_sub, fx_flat, load, material=material, steps=steps,
                                probes={"tip": l})
            return float(l @ r.u), r

        try:
            P0, (t0, r0) = f_lin, tip(f_lin)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            out.append({"press": int(i), "at": spec.presses[i].at.tolist(), "linear": f_lin,
                        "error": f"large-deflection solve failed: {exc}"})
            continue
        P1 = P0 * target / t0 if t0 > 0 else 1.5 * P0
        best = (P0, t0, r0)
        n_solves = 1
        while n_solves < max_solves:
            try:
                t1, r1 = tip(P1, start=best[2] if best[2].converged else None)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                break
            n_solves += 1
            best = (P1, t1, r1)
            if abs(t1 - target) <= tol * target or not r1.converged:
                break
            if t1 == t0:
                break
            P0, t0, P1 = P1, t1, P1 + (target - t1) * (P1 - P0) / (t1 - t0)
            P1 = max(P1, 0.2 * P0)
        P, t, r = best
        ratio = float(P / f_lin)
        rec = {"press": int(i), "at": spec.presses[i].at.tolist(), "linear": f_lin, "nonlinear": float(P),
               "nonlinearOverLinear": ratio, "plungerTravel": t, "converged": bool(r.converged),
               "pressTravel": float(f @ r.u), "maxStrain": float(r.strain.max()) if r.strain is not None else None,
               "solves": n_solves, "cells": int(sub.n_elem)}
        if f_cut is not None:
            rec.update({"linearFloor": f_lin, "linear": float(f_cut[i]), "nonlinear": ratio * float(f_cut[i]),
                        "skinModel": "cut cells; large-deflection ratio from the E x^3 floor"})
        if prob.clr_dofs is not None:
            rec["clearanceMax"] = float(spec.press_sign * (r.u[cm > 0]).max() if spec.press_sign > 0
                                        else (-r.u[cm > 0]).max())
        out.append(rec)
    return out


def _as_printed(xphys, floor, domain=None):
    """(field to mesh, black-and-white design to read) for a design whose
    thin-skin cells are floored at their skin share phi. In such a cell the
    skin itself is printed exactly (the delivered mesh joins the designer's
    skin), and only the material the optimizer added above phi is structure
    of its own: its share of the rest of the cell, (x - phi) / (1 - phi).
    Thresholding x itself instead made every cell more than half skin a
    whole solid cell: a 0.57 mm skin share read as 1 mm, and the as-printed
    fire forces 1.6-1.8x the design's on the touch microphone."""
    fl = floor > 0
    extra = np.clip(np.where(fl, (xphys - floor) / np.maximum(1.0 - floor, 1e-9), xphys), 0.0, 1.0)
    if domain is not None and fl.any():
        # A member ending on the skin ends in the cell next to it: the
        # thin-skin cell between carries it in the design (at x^3), but
        # meshed by its own share above phi it is empty, and since the mesh
        # is not run on into the skin the member stopped short of it. The
        # microphone body's capsule suspension came out a second body,
        # floating (2.5 N/mm read back for 50 designed). Each thin-skin cell
        # beside a solid interior cell is meshed solid, so the member runs
        # into the skin as it has to be printed.
        from scipy import ndimage
        g = np.zeros(domain.dims, bool)
        ijk = domain.mesh.elem_ijk
        g[tuple(ijk[(extra > 0.5) & ~fl & ~domain.preserve].T)] = True
        g = ndimage.binary_dilation(g, structure=ndimage.generate_binary_structure(3, 1))
        touch = fl & g[tuple(ijk.T)]
        extra[touch] = 1.0
    solid = np.where(extra > 0.5, 1.0, np.where(fl, floor, 0.0))
    return extra, solid


SKIN_LINING = True


def _skin_lining(study, domain, extra):
    """(extra, study) with the design's thickening of a thin skin moved out
    of the field and into the skin itself.

    Where the design adds a layer of cells right under the skin (and
    nothing behind it), the field meshes it as blocks one cell thick:
    smoothed, a skin 0.1-0.6 mm thicker that ripples cell by cell (the
    0.7 mm touch-mic part). Those cells become a thickness field instead,
    their solid share times the pitch averaged over a cell, and the skin's
    inner surface is moved inward by it as one smooth surface
    (shapes.Thickened). Members that run on into the skin keep their cells.
    Returns the inputs unchanged when the skin is not a plain shrunk shell."""
    import copy

    from scipy import ndimage

    from .shapes import Offset, Shell, Thickened, Union
    reg = next((r for r in study.preserve if r.id == "skin"), None)
    t = next((t for rid, t in study.shells if rid == "skin"), None)
    if not SKIN_LINING or reg is None or t is None or not isinstance(reg.shape, Shell):
        return extra, study
    inner = reg.shape.inner
    holes = []
    if isinstance(inner, Union) and isinstance(inner.shapes[0], Offset):
        inner, holes = inner.shapes[0], inner.shapes[1:]
    if not isinstance(inner, Offset):
        return extra, study
    h = domain.pitch
    c = domain.mesh.elem_centers()
    depth = -inner.base.sdf(c)                      # below the exterior, mm
    if hasattr(inner, "thickness_at"):               # a skin of zoned thickness
        t = inner.thickness_at(c)
    solid = (extra > 0.5) & ~domain.preserve
    band = solid & (depth > t - 0.5 * h) & (depth < t + 1.0 * h)
    if not band.any():
        return extra, study
    # a band cell with solid behind it (one cell further in) is a member's
    # foot, not a lining
    idx = np.flatnonzero(band)
    e = 1e-3
    grad = np.stack([(inner.base.sdf(c[idx] + e * d) - inner.base.sdf(c[idx] - e * d)) / (2 * e)
                     for d in np.eye(3)], 1)
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)
    behind = c[idx] - grad * h                       # one cell inward
    ijk = np.floor((behind - domain.origin) / h).astype(int)
    lut = -np.ones(domain.dims, int)
    lut[tuple(domain.mesh.elem_ijk.T)] = np.arange(domain.n_elem)
    ok = (ijk >= 0).all(1) & (ijk < np.array(domain.dims)).all(1)
    nb = np.full(len(idx), -1)
    nb[ok] = lut[tuple(ijk[ok].T)]
    foot = (nb >= 0) & solid[np.maximum(nb, 0)]
    lining = idx[~foot]
    if not len(lining):
        return extra, study
    g = np.zeros(domain.dims)
    g[tuple(domain.mesh.elem_ijk[lining].T)] = 1.0
    w = np.zeros(domain.dims)
    w[tuple(domain.mesh.elem_ijk[np.flatnonzero((depth > t - 0.5 * h) & (depth < t + 1.0 * h))].T)] = 1.0
    gs, ws = ndimage.gaussian_filter(g, 1.0), ndimage.gaussian_filter(w, 1.0)
    thick = np.where(ws > 1e-3, gs / np.maximum(ws, 1e-9), 0.0) * h
    thick[gs < 0.05] = 0.0

    def more(p):
        q = (np.asarray(p, float) - domain.origin) / h - 0.5
        return ndimage.map_coordinates(thick, q.T, order=1, mode="constant", cval=0.0)

    new_inner = Thickened(inner, more)
    if holes:
        new_inner = Union([new_inner] + holes)
    st = copy.copy(study)
    st.preserve = [dataclasses_replace(r, shape=Shell(r.shape.outer, new_inner, r.shape.clip)) if r.id == "skin"
                   else r for r in study.preserve]
    out = np.array(extra, float)
    out[lining] = 0.0
    return out, st


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def run_switch(study, out_dir: str, pitches=None, iterations=None, nonlinear: bool | None = None,
               nonlinear_presses=None, on_progress=None, providers=None, readback=None,
               mesh_options=None, optimize_options=None) -> dict:
    """Optimize, mesh and check a study's `switch` block. Writes
    <name>.stl (the delivered part: the exact exterior skin and preserves
    joined to the optimized interior) and returns the report: per press the
    target, the linear fire force on the design and, when nonlinear is
    true (the default), the large-deflection one on the solid part.

    providers: more constraints (see optimize_switch). readback:
    `fn(domain, design, solid, delivered, floor, mesh)` -> dict merged into
    the report, called with the design density, its black-and-white
    reading, the delivered mesh read back on this grid (None when the mesh
    is not watertight) and the mesh itself. mesh_options:
    reconstruct.ReconstructOptions for the delivered mesh. feel.run_feel uses both; with them the study may have no
    `switch` block."""
    import os

    from .reconstruct import reconstruct

    os.makedirs(out_dir, exist_ok=True)

    def note(rec, _x):
        if on_progress is not None:
            on_progress({"stage": "optimize", **rec})

    r = optimize_switch(study, pitches=pitches, iterations=iterations, on_iter=note, providers=providers,
                        **(optimize_options or {}))
    rep = {"name": study.name, "pitches": [lv["pitch"] for lv in r.levels], "levels": r.levels,
           "volume": r.volume, "iterations": r.iterations, **r.report}
    has_switch = bool(study.raw.get("switch"))
    rep.setdefault("presses", [])
    nu = study.elastic(design=True)
    modulus = float(study.material.E)
    spec = switch_from_study(study, r.domain) if has_switch else None
    # the same reading on the thresholded (as printed) design
    extra, solid = _as_printed(r.xphys, r.floor, r.domain)
    if has_switch:
        ff = fire_forces(r.domain, nu, spec, np.maximum(solid, 1e-9), modulus)
        for p, f in zip(rep["presses"], ff["forces"]):
            p["thresholdedForceGf"] = float(f / GF)
    if on_progress is not None:
        on_progress({"stage": "mesh"})
    np.save(os.path.join(out_dir, "xphys.npy"), r.xphys)
    mesh_field, mesh_study = _skin_lining(study, r.domain, extra)
    mesh, info = reconstruct(mesh_study, r.domain, mesh_field, mesh_options)
    path = os.path.join(out_dir, f"{study.name}.stl")
    mesh.export(path)
    rep["mesh"] = path
    rep["mass"] = float(mesh.volume * study.material.density / 1000.0)
    rep["watertight"] = bool(mesh.is_watertight)
    if mesh.is_watertight:
        # drawn: the delivered mesh's volume share per cell over the skin
        # floor (for a feel read-back)
        from .domain import _fraction
        from .shapes import MeshShape
        if on_progress is not None:
            on_progress({"stage": "verify", "check": "delivered mesh fire force"})
        drawn = _fraction([MeshShape(mesh)], r.domain.mesh.elem_centers(), r.domain.pitch)
        delivered = np.maximum(np.maximum(drawn, r.floor), 1e-9)
        if has_switch:
            # solved on the mesh itself as cut cells on this grid
            # (switch_lattice.readback): reading the volume share per cell
            # as a density read thin members as soft hinges and did not
            # converge with the grid
            from .switch_lattice import readback as cut_readback
            fd = cut_readback(study, mesh, pitch=r.domain.pitch)
            for p, fg in zip(rep["presses"], fd["forceGf"]):
                f = fg * GF
                p["deliveredForceGf"] = float(fg)
                p["deliveredOk"] = bool((p["max"] is None or f <= p["max"] * 1.02) and
                                        (p["min"] is None or f >= p["min"] * 0.98))
            rep["deliveredHold"] = [float(v) for v in fd["hold"]]
            rep["deliveredClearanceMax"] = fd.get("clearanceMax")
            if fd.get("clearance") is not None:
                rep["deliveredClearance"] = fd["clearance"]
            rep["deliveredReadback"] = {"method": "cut cells", "pitch": fd["pitch"], "seconds": fd["seconds"]}
    else:
        delivered = None
    if readback is not None:
        if on_progress is not None:
            on_progress({"stage": "verify", "check": "feel targets"})
        rep.update(readback(r.domain, r.xphys, np.maximum(solid, 1e-9), delivered, r.floor, mesh))
    from .skinreview import review_options, skin_review
    opts = review_options(study) if has_switch else None
    if opts is not None:
        # Where the fixed exterior, not the interior, keeps a press out of its
        # window, and what the designer could change. Feedback only: the
        # exterior is never edited.
        if on_progress is not None:
            on_progress({"stage": "verify", "check": "skin review"})
        dlv = [p["deliveredForceGf"] * GF for p in rep["presses"]] if "deliveredForceGf" in rep["presses"][0] \
            else None
        sr = skin_review(study, r.domain, spec, r.xphys, r.floor, nu, modulus, radii=opts["radii"], delivered=dlv)
        # The design grid's thin-skin model can read the skin soft, and a
        # coarse grid can meet a map a finer one does not: read the bare skin
        # and the delivered mesh again on cut cells.
        from .skinreview import finish_review
        bare_c = [row["forceGf"] * GF for row in sr["skinOnly"]]
        rep["skinReview"] = finish_review(study, sr, r.domain.pitch, r.floor, opts, bare_c,
                                          mesh if dlv else None, dlv, on_progress)
    rep["inWindow"] = sum(bool(p["ok"]) for p in rep["presses"])
    rep["deliveredInWindow"] = sum(bool(p.get("deliveredOk")) for p in rep["presses"])
    if nonlinear is None:
        nonlinear = bool((study.raw.get("switch") or {}).get("nonlinear", True))
    if nonlinear and has_switch:
        if on_progress is not None:
            on_progress({"stage": "verify", "check": "nonlinear fire force"})
        # Large-deflection solves cost ~90 s a press on 6k solid cells (four
        # cores); on the finest grid they took longer than the optimization.
        # The check runs on nonlinearPitch (default the first, coarsest
        # pitch), the design read onto it.
        from .domain import build_domain
        b = study.raw.get("switch") or {}
        p_nl = float(b.get("nonlinearPitch") or rep["pitches"][0])
        dom_nl, xp_nl, floor_nl, spec_nl = r.domain, r.xphys, r.floor, spec
        if abs(p_nl - r.domain.pitch) > 1e-9:
            dom_nl, floor_nl = _thin_skin(study, build_domain(study, p_nl))
            xp_nl = np.clip(_interp_design(r.domain, r.xphys, dom_nl), 0.0, 1.0)
            xp_nl[dom_nl.preserve] = 1.0
            xp_nl = np.maximum(xp_nl, floor_nl)
            spec_nl = switch_from_study(study, dom_nl)
        rep["nonlinearPitch"] = dom_nl.pitch
        if nonlinear_presses is None:
            # one press per distinct target (the centre, one per ring of a field)
            seen, nonlinear_presses = set(), []
            for i, p in enumerate(spec.presses):
                if (p.max, p.min) not in seen:
                    seen.add((p.max, p.min))
                    nonlinear_presses.append(i)
        # read the design as printed (as thresholdedForceGf): a floored
        # skin cell is its skin share phi plus whatever the optimizer added
        # above it, so thresholding x itself made every cell more than half
        # skin a whole solid cell, and the large-deflection forces 1.2-1.4x
        # too stiff before any large deflection (touch microphone)
        solid_nl = _as_printed(xp_nl, floor_nl)[1]
        nl = check_fire_nonlinear(dom_nl, nu, spec_nl, solid_nl, modulus, presses=nonlinear_presses,
                                  floor=floor_nl)
        for row in nl:
            if "nonlinear" in row:
                p = rep["presses"][row["press"]]
                p["nonlinearForceGf"] = row["nonlinear"] / GF
                p["nonlinearOk"] = bool((p["max"] is None or row["nonlinear"] <= p["max"] * 1.02) and
                                        (p["min"] is None or row["nonlinear"] >= p["min"] * 0.98))
        rep["nonlinear"] = nl
    rep["seconds"] = round(r.seconds, 1)
    rep["passes"] = all(p["ok"] for p in rep["presses"])
    return rep
