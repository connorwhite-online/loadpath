"""
Conjugate heat transfer on a voxel grid, for heat sinks and cold plates in a
duct: the flow from flow.py's lattice Boltzmann run, the heat by finite
volumes, solid and fluid in one linear system.

Topology optimization of heat sinks usually rates a design with a fixed
convection coefficient h on its surface (thermal.py does). That is fine for
choosing where metal goes in conduction, but it cannot say what a design
costs in pressure drop, or that the fluid warms as it passes the first fins,
or that a fine lattice chokes its own flow. To compare a solid topology
against a lattice fairly, both are rated here the same way:

- the flow through the design (`duct_flow`): flow.network_flow on the fluid
  voxels, from a plug at the duct's inlet face to a fixed-pressure outlet,
  time-averaged; the flow rate is searched until the pressure drop matches a
  target (`at_pressure_drop`), since a fair comparison of sinks holds the fan
  fixed, not the flow;
- the heat (`solve_heat`): steady conduction in the solid and the fluid,
  advection in the fluid (first-order upwind on the voxel faces, from the
  averaged velocity), the heat entering through given faces of the solid
  (a die under the base), the fluid entering at the inlet temperature.
  Walls not named are adiabatic. One sparse solve.

The result is the thermal resistance (the hottest point of the heated face
over the inlet temperature, per watt) and the mean one, with the energy
balance as a check (what the fluid carries out over what went in).

Units: mm for geometry, SI for properties, W, K.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["solve_heat", "duct_flow", "at_pressure_drop"]


def solve_heat(solid: np.ndarray, fluid: np.ndarray, velocity, h: float, k_solid: float, fluid_props,
               heat_faces: np.ndarray, power: float, flow_axis: int = 0, outlet=None, inlet=None,
               project: bool = False) -> dict:
    """Steady temperature rise over the inlet (K) in every solid and fluid voxel.

    solid, fluid: bool voxel masks (disjoint; everything else is an
    adiabatic wall). velocity: (3, *shape) m/s on the voxels (zero in the
    solid). heat_faces: bool mask of solid voxels whose face on the low side
    of axis 2 takes the heat (spread evenly by area), or a float field of
    weights (several dies at different powers), `power` W in all. The fluid
    enters through the voxels on the grid's low face along `flow_axis` at
    temperature 0 and leaves through the high one; or, with `inlet` and
    `outlet` = (mask, axis, sign), enters and leaves through the masked
    fluid voxels, inward (inlet) or outward (outlet) along `sign` x `axis`.
    project: first make the face fluxes conserve volume (`_project`); for a
    lattice Boltzmann flow through a fine network, without it the heat
    balance can be far off.

    Returns {"T": field (nan outside), "maxRise", "meanRise" (over the heated
    faces), "resistance" K/W (max), "meanResistance", "balance" (heat carried
    out over power)}."""
    shape = solid.shape
    hm = h * 1e-3
    live = solid | fluid
    idx = -np.ones(shape, np.int64)
    n = int(live.sum())
    idx[live] = np.arange(n)
    k = np.where(solid, k_solid, fluid_props.k)
    rc = fluid_props.rho * fluid_props.cp
    U = np.asarray(velocity, float)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    # the faces: conduction across every live pair, volume flux (m^3/s,
    # a to b positive) across fluid pairs from the averaged cell velocities
    fa, fb, flux = [], [], []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(0, -1)
        b[ax] = slice(1, None)
        a, b = tuple(a), tuple(b)
        both = live[a] & live[b]
        ia, ib = idx[a][both], idx[b][both]
        ka, kb = k[a][both], k[b][both]
        g = 2.0 * ka * kb / (ka + kb) * hm                    # W/K across the face
        rows += [ia, ib, ia, ib]
        cols += [ib, ia, ia, ib]
        vals += [-g, -g, g, g]
        ff = fluid[a] & fluid[b]
        fa.append(idx[a][ff])
        fb.append(idx[b][ff])
        flux.append(0.5 * (U[ax][a][ff] + U[ax][b][ff]) * hm * hm)
    fa, fb, flux = np.concatenate(fa), np.concatenate(fb), np.concatenate(flux)
    # inlet and outlet: (cells, volume flux in / out)
    if inlet is None:
        lo = [slice(None)] * 3
        lo[flow_axis] = 0
        lo = tuple(lo)
        ii = idx[lo][fluid[lo]]
        qi = np.maximum(U[flow_axis][lo][fluid[lo]] * hm * hm, 0.0)
    else:
        im, iax, isign = inlet
        im = np.asarray(im, bool) & fluid
        ii, qi = idx[im], np.maximum(isign * U[iax][im] * hm * hm, 0.0)
    if outlet is None:
        hi = [slice(None)] * 3
        hi[flow_axis] = -1
        hi = tuple(hi)
        io = idx[hi][fluid[hi]]
        qo = np.maximum(U[flow_axis][hi][fluid[hi]] * hm * hm, 0.0)
    else:
        om, oax, osign = outlet
        om = np.asarray(om, bool) & fluid
        io, qo = idx[om], np.maximum(osign * U[oax][om] * hm * hm, 0.0)
    if project:
        flux, qo = _project(n, fa, fb, flux, ii, qi, io, qo)
    F = rc * flux                                             # W/K
    fp, fm = np.maximum(F, 0.0), np.maximum(-F, 0.0)
    Fo = rc * qo
    # upwind advection, conservative: each cell gives out at its own T and
    # takes in F T_upwind. A cell's diagonal is its outflow; where (the LBM's
    # averaged velocity is weakly compressible) a cell's inflow beats it, the
    # matrix loses diagonal dominance and AIR can diverge, so the diagonal
    # is max(outflow, inflow): exact for a divergence-free flow, and with
    # `project` the flow is made divergence-free first.
    outs = np.zeros(n)
    ins = np.zeros(n)
    np.add.at(outs, fa, fp)
    np.add.at(outs, fb, fm)
    np.add.at(ins, fb, fp)
    np.add.at(ins, fa, fm)
    np.add.at(ins, ii, rc * qi)                               # inflow at temperature 0
    np.add.at(outs, io, Fo)
    rows += [fb, fa]
    cols += [fa, fb]
    vals += [-fp, -fm]
    diag = np.maximum(outs, ins)
    # heat in
    wq = np.where(solid, np.asarray(heat_faces, float), 0.0)
    q = wq > 0
    if not q.any():
        raise ValueError("no heated face on the solid")
    np.add.at(rhs, idx[q], power * wq[q] / wq[q].sum())
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    A = A + sp.diags(diag)
    t = _solve(A, rhs)
    T = np.full(shape, np.nan)
    T[live] = t
    carried = float((Fo * t[io]).sum())
    tq = T[q]
    return {"T": T, "maxRise": float(tq.max()), "meanRise": float(tq.mean()),
            "resistance": float(tq.max() / power), "meanResistance": float(tq.mean() / power),
            "balance": carried / power, "inletCells": int(ii.size)}


def _project(n, fa, fb, flux, ii, qi, io, qo):
    """The nearest face fluxes to `flux` that conserve volume in every cell
    (least squares, the inlet fluxes held, the outlet ones free): flux -
    grad phi with L phi = div, L the fluid cells' graph Laplacian (the
    outlets tied to phi = 0). The averaged LBM velocity, interpolated to
    faces, gains and loses volume cell by cell; advected heat with it does
    the same, and on a water cold plate that lost 99% of the heat."""
    import pyamg

    div = np.zeros(n)
    np.add.at(div, fa, flux)
    np.add.at(div, fb, -flux)
    np.add.at(div, io, qo)
    np.add.at(div, ii, -qi)
    used = np.zeros(n, bool)
    used[fa] = used[fb] = used[io] = used[ii] = True
    m = np.flatnonzero(used)
    loc = -np.ones(n, np.int64)
    loc[m] = np.arange(m.size)
    a, b = loc[fa], loc[fb]
    d = np.zeros(m.size)
    np.add.at(d, a, 1.0)
    np.add.at(d, b, 1.0)
    np.add.at(d, loc[io], 1.0)
    d += 1e-9                                                 # pockets cut off from the outlet
    Lm = sp.coo_matrix((np.concatenate([-np.ones(a.size), -np.ones(a.size)]),
                        (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(m.size, m.size)).tocsr()
    Lm = Lm + sp.diags(d)
    rhs = -div[m]
    ml = pyamg.smoothed_aggregation_solver(Lm)
    phi = ml.solve(rhs, tol=1e-12, accel="cg", maxiter=500)
    flux = flux - (phi[b] - phi[a])
    qo = qo + phi[loc[io]]
    return flux, qo


def _solve(A, b):
    """Advection-diffusion is nonsymmetric and, on a 0.5 mm air sink of ~300k
    cells, more than SuperLU's fill-in can take (not done in 20 minutes),
    while GMRES on the symmetric part's multigrid stalls (metal and air
    conductances 10^4 apart). Rows scaled by the diagonal, then pyamg's AIR
    (built for upwinded advection) under GMRES: ~20 s to 1e-9."""
    import pyamg

    A = A.tocsr()
    d = 1.0 / A.diagonal()
    As = (sp.diags(d) @ A).tocsr()
    bs = d * b
    for make in (lambda: pyamg.air_solver(As),
                 lambda: pyamg.smoothed_aggregation_solver(As, symmetry="nonsymmetric")):
        x = make().solve(bs, tol=1e-10, accel="gmres", maxiter=300)
        if np.linalg.norm(b - A @ x) <= 1e-7 * np.linalg.norm(b):
            return x
    raise RuntimeError("conjugate heat solve did not converge")


def duct_flow(fluid: np.ndarray, h: float, flow_lpm: float, medium="air", flow_axis: int = 0, **kw) -> dict:
    """flow.network_flow through a duct whose inlet is the grid's low face
    along `flow_axis` and outlet the high one (the fluid voxels there).
    Returns the network_flow report with "velocity" (3, *shape) m/s on the
    unpadded grid and "dp" Pa (inlet to outlet)."""
    from .flow import network_flow

    shape = np.asarray(fluid.shape)
    o = [a for a in range(3) if a != flow_axis]
    sel_in = np.take(fluid, 0, axis=flow_axis)
    ij = np.argwhere(sel_in)
    c = 0.5 * (ij.min(0) + ij.max(0)) * h
    rad = 0.5 * np.linalg.norm((ij.max(0) - ij.min(0) + 1) * h) + h

    def port(end):
        p = np.zeros(3)
        p[flow_axis] = 0.0 if end == 0 else (shape[flow_axis] - 1) * h
        p[o[0]], p[o[1]] = c
        ax = np.zeros(3)
        ax[flow_axis] = -1.0 if end == 0 else 1.0
        return {"point": p.tolist(), "axis": ax.tolist(), "radius": float(rad)}

    rep = network_flow(fluid, h, port(0), port(1), medium=medium, flow_lpm=flow_lpm, fields=True, **kw)
    fld = rep.pop("fields")
    rep["velocity"] = fld["velocity"][:, 1:-1, 1:-1, 1:-1]
    rep["pressure"] = fld["pressure"][1:-1, 1:-1, 1:-1]
    rep["dp"] = float(rep["totalPressureDrop"]) if "totalPressureDrop" in rep else _dp(rep, fluid, flow_axis)
    return rep


def _dp(rep, fluid, ax):
    p = rep["pressure"]
    a = np.take(p, 0, axis=ax)[np.take(fluid, 0, axis=ax)].mean()
    b = np.take(p, -1, axis=ax)[np.take(fluid, -1, axis=ax)].mean()
    return float(a - b)


def at_pressure_drop(fluid: np.ndarray, h: float, dp: float, guess_lpm: float, medium="air", flow_axis: int = 0,
                     rtol: float = 0.05, tries: int = 5, **kw) -> dict:
    """duct_flow at the flow rate whose pressure drop is `dp` Pa (within
    rtol): secant on log(flow) against log(dp) from `guess_lpm`."""
    q = float(guess_lpm)
    seen = []
    rep = None
    for _ in range(tries):
        rep = duct_flow(fluid, h, q, medium, flow_axis, **kw)
        rep["flowLpm"] = q
        seen.append((np.log(q), np.log(max(rep["dp"], 1e-9))))
        if abs(rep["dp"] / dp - 1.0) <= rtol:
            break
        if len(seen) == 1:
            slope = 1.6                                  # dp ~ Q^1.6 between laminar and turbulent
        else:
            (x0, y0), (x1, y1) = seen[-2], seen[-1]
            slope = (y1 - y0) / (x1 - x0) if abs(x1 - x0) > 1e-9 else 1.6
            slope = float(np.clip(slope, 1.0, 2.0))
        q = float(np.exp(seen[-1][0] + (np.log(dp) - seen[-1][1]) / slope))
    rep["searched"] = [(float(np.exp(a)), float(np.exp(b))) for a, b in seen]
    return rep
