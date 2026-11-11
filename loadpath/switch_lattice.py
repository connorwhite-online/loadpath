"""
Push pads with a graded lattice inside the skin instead of ribs.

`switch.py` grows ribs: each cell is solid or empty, and the optimizer
places them. Here the interior is a lattice (a TPMS sheet or a strut cell
of one fixed size) whose wall thickness is graded cell by cell to hit each
press window. The question it answers for a designer: can one printable
lattice, varied only in how thick its walls are, give the same feel as
ribs, with less material and with powder that can get out?

The model is the same as switch.py's (fire force per press point from two
solves; see that module), with a lattice material in the design cells:

    rho   = rho_lo (rho_hi / rho_lo) ** y          y in [0, 1], filtered over
                                                   about half a lattice cell
    E/E_s = p^3 S(rho)                             S: the lattice's homogenized
                                                   modulus (lattice.sheet_modulus)

rho_lo is the lattice at the process's minimum wall (minMember), rho_hi
at the thickest wall allowed. With `mode: "fill"` the lattice fills the
whole interior (p = 1, every design cell at least rho_lo); with
`mode: "where-needed"` a second field p (filtered and projected like the
ribs' density) also decides where there is lattice at all.

In a thin-skin cell (switch._thin_skin: a 1 mm skin on 1 mm cells) the skin
takes its share phi of the cell and the lattice the rest:
E/E_s = phi^3 + (1 - phi) p^3 S(rho).

The homogenized model is good where the lattice is several cells deep.
Under a push pad's skin it is one or two cells deep, so the result is read
back on the printable mesh itself (`readback`): the exact lattice joined to
the designer's skin, voxelized at a third of a millimetre, and the same
fire forces solved on that. `correct` feeds the read-back's error per
press back into the windows and runs again: the homogenized model's
direction is right, its scale is not.
"""
from __future__ import annotations

import time
import types
from dataclasses import dataclass

import numpy as np

__all__ = ["LatticeOptions", "corrected_windows", "lattice_part", "optimize_switch_lattice", "path_clearance",
           "powder_escape", "readback", "run_switch_lattice"]


@dataclass
class LatticeOptions:
    type: str = "gyroid"
    cell: float = 6.0            # mm, lattice period
    wall_min: float | None = None   # mm; default the study's minMember
    wall_max: float | None = None   # mm; default 0.4 cell
    mode: str = "fill"           # "fill" | "where-needed"
    grade_radius: float | None = None   # mm, the density filter; default half a cell
    region_radius: float | None = None  # mm, where-needed's presence filter; default 0.75 cell

    @classmethod
    def from_block(cls, b: dict | None):
        b = dict(b or {})
        keys = {"type": "type", "cell": "cell", "wallMin": "wall_min", "wallMax": "wall_max", "mode": "mode",
                "gradeRadius": "grade_radius",
                "regionRadius": "region_radius"}
        bad = set(b) - set(keys)
        if bad:
            raise ValueError(f"switch.lattice: unknown keys {sorted(bad)}")
        o = cls(**{keys[k]: v for k, v in b.items()})
        if o.mode not in ("fill", "where-needed"):
            raise ValueError("switch.lattice.mode must be 'fill' or 'where-needed'")
        return o

    def resolved(self, study):
        import copy
        o = copy.copy(self)
        if o.wall_min is None:
            # a lattice wall is one printed feature: the process's minimum
            # feature, not the two-feature member an optimized solid keeps
            proc = getattr(study, "process", None)
            if proc is not None:
                o.wall_min = max(float(proc.min_feature), float((study.raw or {}).get("minMember") or 0.0))
            else:
                o.wall_min = float(study.min_member or 0.8)
        if o.wall_max is None:
            o.wall_max = 0.4 * o.cell
        if o.grade_radius is None:
            o.grade_radius = 0.5 * o.cell
        if o.region_radius is None:
            # a lattice region thinner than about a cell is not a lattice:
            # at the ribs' own filter (1.3 mm) the first runs grew 2 mm wide
            # "ribs" of 6 mm cells, which homogenize but cannot be printed as
            # a lattice. A cone filter of radius R keeps members over ~0.77 R.
            o.region_radius = 0.75 * o.cell
        return o


def _density(kind, wall, cell):
    from .lattice import relative_density
    return float(relative_density(kind, wall, cell))


def _wall(kind, rho, cell):
    """Sheet wall or strut diameter (mm) for relative density rho."""
    from .struts import is_strut, strut_diameter_for_density
    if is_strut(kind):
        return np.asarray(strut_diameter_for_density(kind, rho, cell), float)
    from .lattice import wall_for_density
    return np.asarray(wall_for_density(kind, rho, cell), float)


def _lattice_sdf(kind, P, cell, wall):
    """Signed distance (mm) of a sheet of thickness `wall` (or struts of
    that diameter) of period `cell`, per point."""
    from .struts import is_strut, strut_distance
    if is_strut(kind):
        U = np.mod(P / cell, 1.0)
        return strut_distance(kind, U, wall / cell) * cell
    from .lattice import tpms
    return np.abs(tpms(kind, P, cell)) - 0.5 * wall


def _modulus_curve(kind, rho_lo, rho_hi):
    """S(rho) = E*/E and dS/drho: a cubic in log-log fitted to the
    lattice's homogenized modulus over [rho_lo, rho_hi], so the slope the
    optimizer sees is the derivative of the curve it uses."""
    from .lattice import sheet_modulus
    r = np.geomspace(rho_lo * 0.98, rho_hi * 1.02, 64)
    s = np.asarray(sheet_modulus(kind, r), float)
    c = np.polyfit(np.log(r), np.log(np.maximum(s, 1e-12)), 3)
    dc = np.polyder(c)

    def S(rho):
        lr = np.log(rho)
        v = np.exp(np.polyval(c, lr))
        return v, v * np.polyval(dc, lr) / rho
    return S


@dataclass
class LatticeResult:
    domain: object
    floor: np.ndarray
    rho: np.ndarray       # per element, design cells (0 elsewhere)
    presence: np.ndarray  # per element (1 in fill mode)
    xphys: np.ndarray     # SIMP-equivalent density (E^(1/3)) the solves used
    forces: np.ndarray
    volume: float         # mm^3 of lattice in the interior (skin share excluded)
    history: list
    report: dict
    seconds: float
    opts: LatticeOptions


def optimize_switch_lattice(study, opts: LatticeOptions | None = None, pitch=None, iterations: int = 80,
                            windows=None, move: float = 0.1, beta_max: float = 16.0, beta_every: int = 15,
                            on_iter=None, tol: float = 0.003, probe=None) -> LatticeResult:
    """Minimize the lattice's volume subject to the switch block's windows,
    the lattice's wall thickness the design variable. `windows`: per press
    (min, max) in N to use instead of the study's (run_switch_lattice
    passes the read-back-corrected ones)."""
    from .domain import build_domain
    from .mma import MMA
    from .simp import E_MIN, PENALTY, _dproject, _project, density_filter, filter_radius
    from .switch import GF, _evaluate, _problem, _Solves, _thin_skin, switch_from_study

    t0 = time.time()
    b = study.raw.get("switch") or {}
    o = (opts or LatticeOptions.from_block(b.get("lattice"))).resolved(study)
    pitch = pitch or (b.get("pitches") or [study.pitch])[0]
    domain, floor = _thin_skin(study, build_domain(study, pitch))
    spec = switch_from_study(study, domain)
    prob = _problem(domain, spec)
    n = domain.n_elem
    pres = domain.preserve
    design = ~pres
    nd = int(design.sum())
    nu = study.elastic(design=True)
    modulus = float(study.material.E)
    rho_lo = _density(o.type, o.wall_min, o.cell)
    rho_hi = _density(o.type, o.wall_max, o.cell)
    if not rho_hi > rho_lo:
        raise ValueError(f"switch.lattice: a {o.wall_min} mm wall already gives the densest lattice allowed "
                         f"(cell {o.cell} mm); use a bigger cell or a thicker wallMax")
    S = _modulus_curve(o.type, rho_lo, rho_hi)
    span = np.log(rho_hi / rho_lo)
    hy, hys = density_filter(domain, max(o.grade_radius, 1.5 * domain.pitch))
    where = o.mode == "where-needed"
    if where:
        hz, hzs = density_filter(domain, max(o.region_radius,
                                             filter_radius(domain, study.length_scale, study.min_hole)))
    phi = floor
    lat_share = 1.0 - phi            # the cell's share left to the lattice
    cellv = domain.pitch ** 3
    win = windows or [(p.min, p.max) for p in spec.presses]
    nvar = nd * (2 if where else 1)
    m = sum(lo is not None for lo, _ in win) + sum(hi is not None for _, hi in win) + len(prob.hold) + \
        (1 if prob.clr_dofs is not None else 0)
    mma = MMA(nvar, max(m, 1), xmin=0.0, xmax=1.0, move=move)
    y = np.full(n, 0.3)
    z = np.full(n, 0.5)
    solves = _Solves(domain, nu)
    beta, since, history = 1.0, 0, []

    def fields(y, z, beta):
        yt = (hy @ y) / hys
        rho = rho_lo * np.exp(span * np.clip(yt, 0, 1))
        s, ds = S(rho)
        if where:
            zt = (hz @ z) / hzs
            p = _project(zt, beta, 0.5)
            dp = _dproject(zt, beta, 0.5)
        else:
            p, dp = np.ones(n), np.zeros(n)
        E = phi ** 3 + lat_share * p ** 3 * s
        E[pres] = 1.0
        xp = np.maximum(E, 1e-9) ** (1.0 / 3.0)
        dE_drho = lat_share * p ** 3 * ds
        dE_dp = lat_share * 3 * p ** 2 * s
        dE_drho[pres] = 0
        dE_dp[pres] = 0
        return rho, p, dp, xp, dE_drho, dE_dp

    def evaluate(y, z, beta):
        """(constraints, their gradients, volume, its gradient, solve
        output), gradients over the design variables [y | z]."""
        rho, p, dp, xp, dE_drho, dE_dp = fields(y, z, beta)
        out, G = _evaluate(domain, nu, prob, xp, modulus, solves)
        # G is d/dxp under SIMP E = E_MIN + xp^3 (1 - E_MIN): back to d/dE
        dEdx = PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)
        F = out["F"]

        def chain(g):
            gE = g / dEdx
            gy = hy.T @ ((gE * dE_drho * rho * span) / hys)
            parts = [gy[design]]
            if where:
                parts.append((hz.T @ ((gE * dE_dp * dp) / hzs))[design])
            return np.concatenate(parts)

        cons, dcons = [], []
        for i, (lo, hi) in enumerate(win):
            if hi is not None:
                cons.append(float(np.log(F[i] / hi)))
                dcons.append(chain(G["logF"][i]))
            if lo is not None:
                cons.append(float(np.log(lo / F[i])))
                dcons.append(chain(-G["logF"][i]))
        for k, (_, _, lim) in enumerate(prob.hold):
            cons.append(out["hold"][k] / lim - 1.0)
            dcons.append(chain(G["hold"][k] / lim))
        if G["clearance"] is not None:
            cons.append(out["clearanceKS"] - 1.0)
            dcons.append(chain(G["clearance"]))
        w = lat_share * cellv
        vol = float((w * p * rho)[design].sum())
        gv_y = hy.T @ ((w * p * rho * span) / hys)
        parts = [gv_y[design]]
        if where:
            parts.append((hz.T @ ((w * rho * dp) / hzs))[design])
        return np.array(cons), np.array(dcons), vol, np.concatenate(parts), out

    if probe is not None:
        # tests: the model itself, for finite-difference checks
        return types.SimpleNamespace(evaluate=evaluate, fields=fields, design=design, n=n, where=where)
    vscale = 10.0 / max(float((lat_share * cellv)[design].sum()) * rho_hi, 1e-9)
    it = 0
    for it in range(1, iterations + 1):
        cons, dcons, vol, dvol, out = evaluate(y, z, beta)
        F = out["F"]
        xv = np.concatenate([y[design]] + ([z[design]] if where else []))
        xn = mma.update(xv, vol * vscale, dvol * vscale, cons, dcons)
        change = float(np.abs(xn - xv).max())
        y[design] = xn[:nd]
        if where:
            z[design] = xn[nd:]
        worst = float(cons.max()) if len(cons) else 0.0
        rec = {"iteration": it, "volume": vol, "forcesGf": (F / GF).round(1).tolist(), "worstViolation": worst,
               "beta": beta, "change": change, "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        if on_iter is not None:
            on_iter(rec)
        since += 1
        if where and beta < beta_max and (since >= beta_every or (change < 2 * tol and since >= 5)):
            beta, since = min(2 * beta, beta_max), 0
        elif (not where or beta >= beta_max) and change < tol and worst < 0.01 and it > 10:
            break
    rho, p, dp, xp, _, _ = fields(y, z, beta)
    if where:
        p = (p > 0.5).astype(float)
        E = phi ** 3 + lat_share * p ** 3 * S(rho)[0]
        E[pres] = 1.0
        xp = np.maximum(E, 1e-9) ** (1.0 / 3.0)
    out, _ = _evaluate(domain, nu, prob, xp, modulus, solves, grads=False)
    rho = np.where(design, rho, 0.0)
    p = np.where(design, p, 0.0)
    vol = float((lat_share * cellv * p * rho)[design].sum())
    rep = {"presses": [{"at": pr.at.tolist(), "min": pr.min, "max": pr.max, "forceGf": float(f / GF),
                        "ok": _ok(f, pr.min, pr.max)} for pr, f in zip(spec.presses, out["F"])],
           "hold": out["hold"].tolist(), "clearanceMax": out.get("clearanceMax"), "latticeVolume": vol,
           "rhoRange": [rho_lo, rho_hi], "iterations": it}
    rep["inWindow"] = sum(r["ok"] for r in rep["presses"])
    return LatticeResult(domain, floor, rho, p, xp, out["F"], vol, history, rep, time.time() - t0, o)


def _ok(f, lo, hi, tol=0.02):
    return bool((hi is None or f <= hi * (1 + tol)) and (lo is None or f >= lo * (1 - tol)))


# ---------------------------------------------------------------------------
# the printable part


def _grid_field(domain, values, fill=0.0):
    g = np.full(domain.dims, fill, float)
    g[tuple(domain.mesh.elem_ijk.T)] = values
    return g


def _sample(domain, grid, pts, order=1):
    from scipy import ndimage
    c = (pts - domain.origin) / domain.pitch - 0.5
    return ndimage.map_coordinates(grid, c.T, order=order, mode="nearest")


def lattice_part(study, res: LatticeResult, step: float = 0.2, preserved=None):
    """The delivered part: the graded lattice (exact walls, from the TPMS
    level set at `step` mm) inside the design space, clear of every keep-out
    and its assembly path, joined to the designer's skin and preserves.
    Returns (trimesh, info)."""
    import trimesh
    from skimage import measure

    from .reconstruct import reconstruct
    from .shapes import manifold_from_trimesh

    o = res.opts
    dom = res.domain
    if preserved is None:
        preserved, _ = reconstruct(study, dom, np.zeros(dom.n_elem))
    lo, hi = study.bounds()
    lo = np.asarray(lo, float) - 2 * step
    hi = np.asarray(hi, float) + 2 * step
    dims = np.ceil((hi - lo) / step).astype(int) + 1
    axes = [lo[i] + step * np.arange(dims[i]) for i in range(3)]
    # where the lattice may go: inside the exterior, out of every keep-out
    # (their swept assembly paths included), and in a design cell that has
    # lattice; graded wall from the per-cell density
    rho_g = _grid_field(dom, np.where(res.presence > 0.5, res.rho, 0.0))
    pres_g = _grid_field(dom, (res.presence > 0.5).astype(float))
    F = np.empty(dims, np.float32)
    outer = study.design_space
    skin_t = max([t for rid, t in study.shells if rid == "skin"] or [o.wall_min])
    kos = [r.shape for r in study.keep_out]
    for k, z in enumerate(axes[2]):
        X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
        P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
        # half a skin inside the exterior: the lattice ends in the skin, and
        # the mesh shape's distance (a 0.37 mm grid) never lets it poke out
        region = np.min([s.sdf(P) for s in outer], axis=0) + 0.5 * skin_t
        for s in kos:
            region = np.maximum(region, -s.sdf(P))
        occ = _sample(dom, pres_g, P)
        region = np.maximum(region, (0.5 - occ) * dom.pitch)
        rho = np.maximum(_sample(dom, rho_g, P), 1e-4)
        wall = np.maximum(_wall(o.type, rho, o.cell), o.wall_min)
        d = _lattice_sdf(o.type, P, o.cell, wall)
        F[:, :, k] = np.maximum(d, region).reshape(X.shape)
    F = np.pad(F, 1, constant_values=1.0)
    if not (F < 1e-4).any():
        # the optimizer kept no lattice: the part is the skin and preserves
        return preserved.copy(), {"latticeOnlyVolume": 0.0, "preservedVolume": float(preserved.volume),
                                  "volume": float(preserved.volume), "watertight": bool(preserved.is_watertight),
                                  "triangles": len(preserved.faces)}
    # a hair off zero: the region's edge falls exactly on grid nodes, and
    # zero-area triangles there made non-manifold edges
    v, f, _, _ = measure.marching_cubes(F, 1e-4, spacing=(step, step, step))
    v += lo - step
    lat = trimesh.Trimesh(v, f, process=True)
    if lat.volume < 0:
        lat.invert()
    a = manifold_from_trimesh(preserved)
    b = manifold_from_trimesh(lat)
    u = (a + b).to_mesh()
    mesh = trimesh.Trimesh(u.vert_properties[:, :3], u.tri_verts, process=True)
    info = {"latticeOnlyVolume": float(lat.volume), "preservedVolume": float(preserved.volume),
            "volume": float(mesh.volume), "watertight": bool(mesh.is_watertight), "triangles": len(mesh.faces)}
    return mesh, info


def _fine_domain(study, pitch):
    """The study's domain at `pitch`, every design cell free (no thin-skin
    floor needed: a 1 mm skin is three cells)."""
    from .domain import build_domain
    return build_domain(study, pitch)


PRESS_FILL = 0.5


def readback(study, mesh, pitch: float = 0.5, n: int = 4, travel_at: float | None = None) -> dict:
    """Fire forces of a delivered mesh, solved on the mesh itself: cut
    cells (cutcell.py, the finite cell method) on the study's grid at
    `pitch`, each cell the surface cuts integrated over its solid part on
    an n^3 split, so a 1 mm wall is 1 mm whatever the grid. The press
    points, plunger, supports and clearance nodes are the study's own on
    that grid.

    Reading the cell's filled share as a density instead (E ~ x^3, or
    E ~ x) did not converge on thin members: a delivered rib design read
    stiffer at every refinement (centre 154 / 217 gf at 0.5 / 0.33 mm under
    the cube), because a half-filled cell across a 1 mm rib is a soft
    hinge whichever law it is given.

    travel_at (N): also the room the part sweeps inward when pressed hard,
    out["travelEnvelope"] = {"nodes": (k, 3) mm, "travel": (k,) mm}: at
    every node, the most it moves along the press direction over all the
    presses, each pressed at travel_at N or its fire force if higher. Past
    the click the switch is taken as a stop (its stem bottomed: the
    plunger's travel held at gap + travel, its seat's give counted), so a
    hard press bends the skin round the plunger rather than driving it on.
    Whatever sits inside under the skin (ballast, a board, a battery) needs
    this much room plus a margin. Linear: read it as an estimate past a
    few times the fire force."""
    from scipy import ndimage

    from .bucklingopt import _factor
    from .cutcell import CutGrid, scan_voxelize
    from .domain import build_domain
    from .fem import Stiffness
    from .switch import GF, _problem, switch_from_study

    t0 = time.time()
    nu = study.elastic(design=True)
    E = float(study.material.E)
    dom = build_domain(study, pitch)
    spec = switch_from_study(study, dom)
    prob = _problem(dom, spec)
    dims = np.asarray(dom.dims)
    fine = scan_voxelize(mesh, dom.pitch / n, dom.origin, dims * n)
    cg = CutGrid(fine, n, dom.origin, dom.pitch, nu)
    lut = np.full(tuple(dims), -1, np.int64)
    lut[tuple(dom.mesh.elem_ijk.T)] = np.arange(dom.n_elem)
    ijk = cg.mesh.elem_ijk
    inb = np.all(ijk < dims, axis=1)
    el = np.full(len(ijk), -1, np.int64)
    el[inb] = lut[tuple(ijk[inb].T)]
    keep = el >= 0
    # pieces no support holds would leave K singular
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
    if dom.springs is not None:
        k = (k + dom.springs[dofs][:, dofs]).tocsr()   # per unit modulus, as switch._Solves
    solve, release = _factor(k[free][:, free].tocsr())
    # The press points as the finger meets this part: the outermost node of
    # each column within the press radius, among the cells at least
    # PRESS_FILL full. The study grid's own press nodes sit on the design
    # space's outermost cells, which the part may barely touch: the first
    # read-backs pressed on near-empty slivers and lost the load.
    import types

    from .fem import HexMesh
    from .switch import _press_nodes, _unit_load
    solid_el = cg.fill >= PRESS_FILL
    pm = HexMesh(sub.elem_ijk[solid_el], dom.pitch, dom.origin)
    pid = dom.mesh.node_id(pm.node_ijk)
    loads = []
    for pr in spec.presses:
        hit = _press_nodes(types.SimpleNamespace(mesh=pm), pr.at, spec.press_axis, spec.press_sign,
                           spec.press_radius)
        sel = np.zeros(dom.mesh.n_nodes, bool)
        sel[pid[hit]] = True
        loads.append(_unit_load(sel, spec.press_axis, spec.press_sign, dom.mesh.n_dof, dom.fixed))
    present = np.zeros(dom.mesh.n_dof, bool)
    present[dofs] = True
    hold = []
    for f, nf, lim in prob.hold:
        # a hold case's load on the nodes this part has, its total kept
        f = np.where(present, f, 0.0)
        tot = float(np.linalg.norm(f.reshape(-1, 3), axis=1).sum())
        hold.append((f * (nf / tot if tot > 0 else 0.0), nf, lim))
    prob.loads, prob.hold = loads, hold
    rhs = list(prob.loads) + [prob.l] + [f for f, _, _ in prob.hold]
    B = np.stack([np.asarray(f)[dofs][free] for f in rhs], axis=1)
    X = np.asarray(solve(B)).reshape(B.shape) / E
    if release is not None:
        release()
    sol = []
    for j in range(X.shape[1]):
        u = np.zeros(dom.mesh.n_dof)
        u[dofs[free]] = X[:, j]
        sol.append(u)
    nP = len(prob.loads)
    U, lam, UH = sol[:nP], sol[nP], sol[nP + 1:]
    a = np.array([prob.l @ u for u in U])
    c = float(prob.l @ lam)
    F = (spec.stroke + spec.trigger * c) / np.maximum(a, 1e-12)
    if travel_at is not None:
        cb = float(spec.base_compliance)
        Fp = np.maximum(F, float(travel_at))
        R = np.maximum(Fp * a - (spec.gap + spec.travel), 0.0) / max(c + cb, 1e-30)
        ax = 3 * nodes + spec.press_axis
        env = np.zeros(len(nodes))
        for u, f_, r_ in zip(U, Fp, R):
            env = np.maximum(env, spec.press_sign * (f_ * u[ax] - r_ * lam[ax]))
        xyz = dom.origin + sub.node_ijk * dom.pitch
        travel_env = {"nodes": xyz, "travel": env, "force": float(travel_at),
                      "max": float(env.max()), "stopReaction": R.tolist()}
    out = {"pitch": dom.pitch, "cells": int(sub.n_elem), "cutCells": int(cg.cut.sum()),
           "forceGf": (F / GF).tolist(),
           "inWindow": int(sum(_ok(f, p.min, p.max) for f, p in zip(F, spec.presses))),
           "hold": [float(f @ u) / max(nf, 1e-30) for (f, nf, _), u in zip(prob.hold, UH)]}
    if prob.clr_dofs is not None:
        from .switch import clearance_rows
        clr_q = [spec.press_sign * u[prob.clr_dofs] * Fi for u, Fi in zip(U, F)]
        out["clearanceMax"] = float(max(q.max() for q in clr_q))
        out["clearance"] = clearance_rows(spec, prob, clr_q)
    if travel_at is not None:
        out["travelEnvelope"] = travel_env
    out["seconds"] = round(time.time() - t0, 1)
    return out


def powder_escape(study, mesh, step: float = 0.25, ball: float = 1.0, min_room: float = 1.0) -> dict:
    """Can unsintered powder leave the interior? Voids inside the part that
    no path of empty cells joins to the outside are sealed; rooms a ball of
    `ball` mm diameter fits in but can only leave through narrower gaps are
    trapped too (escape.trapped_voids). Also the narrowest passage
    on the way out from the deepest void (the largest ball that still rolls
    out of every room)."""
    from .escape import trapped_voids
    from .verify import voxelize

    solid, origin = voxelize(mesh, step)
    solid = np.pad(solid, 2)
    noise = max(1, round(min_room / step ** 3))

    def rooms(d):
        lbl, n = trapped_voids(solid, 0.5 * d / step)
        if not n:
            return 0, 0.0
        size = np.bincount(lbl.ravel())[1:]
        big = size >= noise
        return int(big.sum()), float(size[big].sum()) * step ** 3

    n, vol = rooms(ball)
    out = {"step": step, "ballMm": ball, "trappedRooms": n, "trappedVolumeMm3": vol}
    # the largest ball that still leaves every room it fits in
    best = 0.0
    for d in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        if rooms(d)[0]:
            break
        best = d
    out["largestFreeBallMm"] = best
    return out


def corrected_windows(windows, design_forces, read_forces):
    """Windows for the next round: each press's own window divided by how
    much stiffer the printable mesh read than the design model there
    (read / design). The ratio is the whole model error at that press, so
    it divides the study's window, not the last round's."""
    out = []
    for k, (lo, hi) in enumerate(windows):
        r = float(read_forces[k] / max(design_forces[k], 1e-12))
        out.append((None if lo is None else lo / r, None if hi is None else hi / r))
    return out


def run_switch_lattice(study, out_dir: str, opts: LatticeOptions | None = None, rounds: int | None = None,
                       readback_pitch: float | None = None, final_pitch: float | None = None,
                       iterations: int = 80, on_progress=None) -> dict:
    """Optimize the lattice, mesh it, read it back on the printable mesh,
    and (rounds > 1) correct the windows by the read-back and go again.
    Writes <name>-lattice.stl; returns the report (each round's design and
    read-back forces, the final part's volume, powder escape, the
    insertion paths' clearance)."""
    import os

    from .reconstruct import reconstruct
    from .switch import GF, switch_from_study

    os.makedirs(out_dir, exist_ok=True)
    b = study.raw.get("switch") or {}
    lb = b.get("lattice") or {}
    o = opts or LatticeOptions.from_block({k: v for k, v in lb.items()
                                           if k not in ("rounds", "readbackPitch", "finalPitch")})
    rounds = int(rounds or lb.get("rounds", 2))
    rp = float(readback_pitch or lb.get("readbackPitch", 0.5))
    fp = float(final_pitch or lb.get("finalPitch", rp))
    note = (lambda ev: on_progress(ev)) if on_progress else (lambda ev: None)
    win, hist, preserved = None, [], None
    best = None
    for k in range(rounds):
        res = optimize_switch_lattice(study, o, iterations=iterations, windows=win,
                                      on_iter=lambda rec, k=k: note({"stage": "optimize", "round": k, **rec}))
        if preserved is None:
            preserved, _ = reconstruct(study, res.domain, np.zeros(res.domain.n_elem))
        note({"stage": "mesh", "round": k})
        mesh, info = lattice_part(study, res, preserved=preserved)
        note({"stage": "readback", "round": k})
        rb = readback(study, mesh, rp)
        spec = switch_from_study(study, res.domain)
        rec = {"round": k, "windowsGf": [[None if lo is None else lo / GF, None if hi is None else hi / GF]
                                         for lo, hi in (win or [(p.min, p.max) for p in spec.presses])],
               "designForceGf": (res.forces / GF).tolist(), "designInWindow": res.report["inWindow"],
               "readback": rb, "volume": info["volume"], "latticeVolume": info["latticeOnlyVolume"]}
        hist.append(rec)
        if best is None or rb["inWindow"] > best[0]["readback"]["inWindow"]:
            best = (rec, mesh, res, info)
        windows = [(p.min, p.max) for p in spec.presses]
        win = corrected_windows(windows, res.forces, np.asarray(rb["forceGf"]) * GF)
    rec, mesh, res, info = best
    path = os.path.join(out_dir, f"{study.name}-lattice.stl")
    mesh.export(path)
    rep = {"name": study.name, "lattice": {"type": o.type, "cell": o.cell, "mode": o.mode},
           "rounds": hist, "chosenRound": rec["round"], "mesh": path, "volume": info["volume"],
           "watertight": info["watertight"], "mass": info["volume"] * float(study.material.density) / 1000.0}
    if abs(fp - rp) > 1e-9:
        note({"stage": "readback", "pitch": fp})
        rep["finalReadback"] = readback(study, mesh, fp)
    else:
        rep["finalReadback"] = rec["readback"]
    rep["inWindow"] = rep["finalReadback"]["inWindow"]
    note({"stage": "powder"})
    rep["powder"] = powder_escape(study, mesh)
    rep["paths"] = path_clearance(study, mesh, preserved)
    return rep


def path_clearance(study, mesh, preserved=None, samples: int = 200_000, seed: int = 0) -> dict:
    """Material inside each keep-out and its assembly path (mm^3, Monte
    Carlo over the keep-out's bounds, on a 0.2 mm voxel grid). With
    `preserved` (the designer's skin and kept geometry), only what the
    design added: the board rails and the base are meant to meet the board
    and the flap, and zero means the board, switch and plug go in past
    nothing new."""
    from .verify import voxelize
    out = {}
    step = 0.2
    lo_all, hi_all = (np.asarray(v, float) for v in mesh.bounds)
    box = (lo_all - step, hi_all + step)
    solid, origin = voxelize(mesh, step, box=box)
    keep = voxelize(preserved, step, box=box)[0] if preserved is not None else np.zeros_like(solid)
    added = solid & ~keep
    rng = np.random.default_rng(seed)
    for r in study.keep_out:
        lo, hi = (np.asarray(v, float) for v in r.shape.bounds())
        lo = np.maximum(lo, origin)
        hi = np.minimum(hi, origin + step * np.asarray(solid.shape))
        if np.any(hi <= lo):
            out[r.id] = 0.0
            continue
        P = lo + rng.random((samples, 3)) * (hi - lo)
        P = P[r.shape.contains(P)]
        if not len(P):
            out[r.id] = 0.0
            continue
        ijk = np.clip(((P - origin) / step).astype(int), 0, np.asarray(solid.shape) - 1)
        share = float(added[tuple(ijk.T)].mean())
        vol = float(np.prod(hi - lo)) * len(P) / samples
        out[r.id] = round(share * vol, 2)
    return out
