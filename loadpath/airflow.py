"""
Air moving by itself: steady laminar natural convection (and, optionally, a
fan's push) with conjugate heat transfer, on a voxel grid.

convection.natural_h gives a surface's coefficient from plate correlations:
a vertical plate in open air, a horizontal one, a pair of isothermal plates
facing each other. A real enclosure's air does not know those cases. Air
drawn through a slotted grate, up a fin channel that is open on one side,
past a wall and out of the next slot up, is choked by the grate as much as
by the fins, warms as it rises, and is pulled by the whole column's
buoyancy, not by each face's. This module solves for that air:

- the flow: the steady incompressible Navier-Stokes equations with the
  Boussinesq buoyancy force rho0 beta (T - T_ambient) g, laminar, finite
  volumes on a staggered (MAC) grid: pressure and temperature at the cell
  centres, each velocity component on the faces normal to it. SIMPLEC
  pressure-velocity coupling (Van Doormaal & Raithby 1984), upwind
  convection with a deferred correction to central differences (second
  order where the flow is resolved), no-slip on every solid voxel;
- the heat: one energy equation over solid and fluid together (conduction
  in both at each cell's conductivity, advection in the fluid), so the
  wall temperature is an answer, not an input; heat sources per cell,
  fixed-temperature cells, isothermal or adiabatic walls;
- the boundaries of the box: `open` (an opening to still room air at the
  ambient temperature and zero total pressure: air comes in at the ambient
  temperature where it is drawn in and leaves where the plume takes it),
  `wall`, `symmetry`, `periodic`, or `inlet` (a fixed velocity in, for a
  fan or a duct). A `fan` zone inside the box adds a fixed pressure rise,
  or one read off a fan curve at the flow it passes.

Air's properties are taken at the film temperature (the mean of ambient and
the wetted walls), refreshed as the solution moves. Radiation is not in
here: add convection.radiation_h by each face's view (convection.openness),
as `wall_h` reports both separately.

Validation (tests/test_airflow.py, docs/AIRFLOW.md): de Vahl Davis' heated
square cavity (1983 benchmark: Nu 2.243 at Ra 1e4, 4.519 at 1e5), an
isothermal vertical plate in open air against Churchill & Chu's laminar
correlation, isothermal parallel-plate channels against Bar-Cohen &
Rohsenow (Elenbaas' data), and fully developed forced flow between plates
(f Re = 96).

Units: geometry in mm (the pitch), everything else SI (W, K, Pa, m/s).

    from loadpath import airflow
    r = airflow.solve(solid, pitch=0.5, k=k_cells, heat=watts_cells,
                      sides={"x": "periodic", "y-": "open", "y+": "wall", "z": "open"})
    r.t, r.wall_h(), r.balance
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .convection import air_properties

__all__ = ["solve", "Fan", "AirflowResult", "SIDES"]

G = 9.80665
# How far each pass's inner solves go. A pass is one step of an outer
# iteration that itself closes a few percent of what is left, so an inner
# solve taken past a thousandth of the pass's own change buys nothing: the
# fin stack in docs/AIRFLOW.md converges in the same 174 passes with these
# as with 1e-7 and 1e-6, to within 7e-4 K and 2e-5 m/s of that answer (the
# stop rule itself is 1e-3 K a pass), in two thirds of the time.
_T_RTOL = 1e-3          # energy: residual against the pass's starting residual
_P_TOL = 1e-3           # pressure correction: residual against the mass imbalance
_DIRECT_BELOW = 30000   # cells: under this a direct solve of the energy matrix is quicker than a hierarchy
SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")
_KINDS = ("open", "wall", "symmetry", "periodic", "inlet")


@dataclass
class Fan:
    """A fan inside the box: the cells it fills (bool mask) push air along
    `axis` (0, 1, 2) in the direction `sign`, with a pressure rise `dp` (Pa)
    across the zone, or `curve` = (dp at no flow Pa, flow at no pressure
    m^3/s), a straight fan line read at the flow the zone passes (scaled
    by `share`, the fraction of the fan's own face the box holds: a
    periodic slice holds a sliver of it)."""
    cells: np.ndarray
    axis: int = 2
    sign: int = 1
    dp: float | None = None
    curve: tuple | None = None
    share: float = 1.0


def _side(spec):
    if isinstance(spec, str):
        spec = {"kind": spec}
    spec = dict(spec)
    if spec.get("kind") not in _KINDS:
        raise ValueError(f"side kind must be one of {_KINDS}, got {spec.get('kind')!r}")
    return spec


def _sides(sides):
    out = {s: {"kind": "open"} for s in SIDES}
    for key, spec in (sides or {}).items():
        keys = [key] if key in SIDES else [key + "-", key + "+"] if key in "xyz" else None
        if keys is None:
            raise ValueError(f"unknown side {key!r}: use x-, x+, ..., or x, y, z for both")
        for k_ in keys:
            out[k_] = _side(spec)
    for a, ax in enumerate("xyz"):
        lo, hi = out[ax + "-"]["kind"] == "periodic", out[ax + "+"]["kind"] == "periodic"
        if lo != hi:
            raise ValueError(f"periodic on {ax} needs both {ax}- and {ax}+")
    return out


def _pad(arr, per):
    """One layer round every axis: wrapped on periodic axes, copied (zero
    gradient) on the others."""
    for ax in range(3):
        w = [(0, 0)] * 3
        w[ax] = (1, 1)
        arr = np.pad(arr, w, mode="wrap" if per[ax] else "edge")
    return arr


@dataclass
class AirflowResult:
    """t: C per cell (solid and fluid); u: the three face-velocity arrays
    (m/s; u[a] has one more layer along a than the cells unless a is
    periodic); p: Pa per cell (dynamic, over the ambient hydrostatic);
    velocity: (3, nx, ny, nz) cell-centred m/s; walls: every face between
    a solid and a fluid cell (see `wall_h`); balance: heat in and where it
    went."""
    t: np.ndarray
    u: list
    p: np.ndarray
    velocity: np.ndarray
    solid: np.ndarray
    pitch: float
    ambient: float
    walls: dict
    balance: dict
    iterations: int
    converged: bool
    history: list = field(default_factory=list)
    air: dict = field(default_factory=dict)
    seconds: float = 0.0

    def wall_h(self):
        """Per wetted face: the convective coefficient referred to the
        ambient, h = q / (A (T_wall - T_ambient)) in W/(m^2 K), with the
        face's centre (mm, from the grid's corner), outward normal (from the
        solid into the air), wall temperature (C), heat (W) and the solid
        cell it belongs to (flat index)."""
        w = self.walls
        dt = w["t_wall"] - self.ambient
        h = np.where(np.abs(dt) > 1e-6, w["q"] / (w["area"] * np.where(np.abs(dt) > 1e-6, dt, 1.0)), 0.0)
        return {"h": h, **w}

    def max_speed(self):
        return float(np.sqrt((self.velocity ** 2).sum(axis=0)).max())


def solve(solid, pitch: float, *, k=None, heat=None, ambient: float = 25.0, sides=None, gravity=(0.0, 0.0, -G),
          fixed=None, fan: Fan | None = None, film: float | None = None, max_iter: int = 3000, tol: float = 1e-4,
          relax=(0.6, 0.8, 1.0), deferred: float = 1.0, init: AirflowResult | None = None, report_every: int = 50,
          verbose: bool = True, pressure: float = 1.01325) -> AirflowResult:
    """Steady laminar buoyant flow and conjugate heat on a voxel box.

    solid: bool (nx, ny, nz), True where there is a solid (no flow); the rest
    is air. pitch: mm per voxel. k: solid conductivity, W/(m K), a number or
    one per cell (fluid cells take air's). heat: W per cell (a number array
    of the grid's shape), in solid or fluid. ambient: C, the room's air.
    sides: {side: kind or {"kind", ...}} for "x-", "x+", ... or "x" for
    both ends; kinds "open" (default), "wall" (no slip; "t": C for an
    isothermal wall, adiabatic without), "symmetry", "periodic" (both ends),
    "inlet" ("velocity": m/s into the box, "t": C, ambient by default).
    gravity: m/s^2 vector in grid axes (default -z). fixed: (mask, C) cells
    held at a temperature. fan: a Fan. film: C for air's properties, None to
    follow the walls. relax: under-relaxation of velocity, pressure and
    temperature. deferred: weight of the central-difference correction to
    upwind convection (0 pure upwind). init: a previous result on the same
    grid to start from.

    Converged when the mass residual is under `tol` of the largest face
    flow, the temperature moves less than 1e-3 K per pass and the heat
    balance closes within 0.5% (or, with no heat crossing an opening, the
    wall heat closes)."""
    t0 = time.time()
    solid = np.asarray(solid, bool)
    shape = solid.shape
    if solid.ndim != 3:
        raise ValueError("solid must be a 3D voxel mask")
    nx, ny, nz = shape
    N = solid.size
    fluid = ~solid
    h = float(pitch) * 1e-3
    A, V = h * h, h ** 3
    S = _sides(sides)
    per = [S["xyz"[a] + "-"]["kind"] == "periodic" for a in range(3)]
    gvec = np.asarray(gravity, float)
    amb = float(ambient)
    au, ap, at = relax

    if k is None:
        if solid.any():
            raise ValueError("give the solid's conductivity k (W/(m K), a number or one per cell)")
        k = 1.0
    ksol = np.broadcast_to(np.asarray(k, float), shape).astype(float)
    q_cell = np.zeros(shape) if heat is None else np.broadcast_to(np.asarray(heat, float), shape).astype(float)
    fix_mask = np.zeros(shape, bool)
    fix_t = np.zeros(shape)
    if fixed is not None:
        m, tv = fixed
        fix_mask = np.asarray(m, bool)
        fix_t = np.broadcast_to(np.asarray(tv, float), shape).astype(float)

    def props(tf):
        rho, cp, kair, mu = (float(v) for v in air_properties(tf, pressure))
        return {"rho": rho, "cp": cp, "k": kair, "mu": mu, "beta": 1.0 / (tf + 273.15), "film C": round(float(tf), 2)}

    # ------------------------------------------------------------------ faces
    cid = np.arange(N).reshape(shape)

    def comp_setup(a):
        """Face grid of velocity component a: its shape, which faces are
        unknowns, the fixed value of the rest, and the two cells each face
        separates (flat index, -1 outside the box)."""
        fs = list(shape)
        if not per[a]:
            fs[a] += 1
        fs = tuple(fs)
        idx = np.indices(fs)
        ia = idx[a]
        lo_i = ia - 1
        hi_i = ia.copy()
        if per[a]:
            lo_i = lo_i % shape[a]
        lo_ok = lo_i >= 0
        hi_ok = hi_i < shape[a]
        def cell_of(ii, ok):
            sl = [idx[0], idx[1], idx[2]]
            sl[a] = np.where(ok, ii, 0)
            return np.where(ok, cid[tuple(sl)], -1)
        lo_c, hi_c = cell_of(lo_i, lo_ok), cell_of(hi_i, hi_ok)
        flu = fluid.ravel()
        lo_f = np.where(lo_c >= 0, flu[np.maximum(lo_c, 0)], False)
        hi_f = np.where(hi_c >= 0, flu[np.maximum(hi_c, 0)], False)
        act = lo_f & hi_f
        val = np.zeros(fs)
        if not per[a]:
            for side, end, inner in ((S["xyz"[a] + "-"], 0, hi_f), (S["xyz"[a] + "+"], fs[a] - 1, lo_f)):
                sl = [slice(None)] * 3
                sl[a] = end
                sl = tuple(sl)
                if side["kind"] == "open":
                    act[sl] = inner[sl]
                elif side["kind"] == "inlet":
                    v = float(side.get("velocity", 0.0)) * (1.0 if end == 0 else -1.0)
                    val[sl] = np.where(inner[sl], v, 0.0)
        return {"shape": fs, "act": act, "val": val, "lo": lo_c, "hi": hi_c}

    comps = [comp_setup(a) for a in range(3)]

    # neighbours of each unknown face, per direction: kind 0 unknown, 1 fixed
    # value a full pitch away, 2 fixed value half a pitch away (a wall along
    # the face), 3 zero gradient
    for a, cp_ in enumerate(comps):
        fs = cp_["shape"]
        act = cp_["act"]
        n_unk = int(act.sum())
        num = -np.ones(fs, np.int64)
        num[act] = np.arange(n_unk)
        cp_["num"] = num
        cp_["n"] = n_unk
        pos = np.nonzero(act)
        cp_["pos"] = pos
        nb = []
        for b in range(3):
            for s in (-1, 1):
                q = [pos[0].copy(), pos[1].copy(), pos[2].copy()]
                q[b] = q[b] + s
                out = (q[b] < 0) | (q[b] >= fs[b])
                if per[b]:
                    q[b] = q[b] % fs[b]
                    out[:] = False
                qc = [np.clip(q[i], 0, fs[i] - 1) for i in range(3)]
                kind = np.zeros(len(pos[0]), np.int8)
                nidx = num[tuple(qc)].copy()
                nval = cp_["val"][tuple(qc)].copy()
                inactive = ~out & (nidx < 0)
                if b == a:
                    kind[inactive] = 1
                    kind[out] = 3                      # outward from an open boundary face
                else:
                    kind[inactive] = np.where(np.abs(nval[inactive]) > 0, 1, 2)
                    side = S["xyz"[b] + ("-" if s < 0 else "+")]["kind"]
                    kind[out] = 2 if side in ("wall", "inlet") else 3
                    nval[out] = 0.0
                nidx[kind != 0] = -1
                nb.append((b, s, kind, nidx, nval))
        cp_["nb"] = nb
        # where to read the convecting velocity on each of the CV's six faces:
        # flat indices into the padded component arrays (two points averaged)
        reads = []
        for b in range(3):
            pads = [tuple(np.asarray(comps[c]["shape"]) + 2) for c in range(3)]
            for s in (-1, 1):
                if b == a:
                    i1 = [pos[0] + 1, pos[1] + 1, pos[2] + 1]
                    i2 = [pos[0] + 1, pos[1] + 1, pos[2] + 1]
                    if s > 0:
                        i2[a] = i2[a] + 1
                    else:
                        i1[a] = i1[a] - 1
                else:
                    i1 = [pos[0] + 1, pos[1] + 1, pos[2] + 1]
                    i2 = [pos[0] + 1, pos[1] + 1, pos[2] + 1]
                    i1[a] = pos[a]                      # the cell below the face along a (padded: -1 + 1)
                    i2[a] = pos[a] + 1                  # the cell above
                    if s > 0:
                        i1[b] = i1[b] + 1
                        i2[b] = i2[b] + 1
                reads.append((b, np.ravel_multi_index(i1, pads[b]), np.ravel_multi_index(i2, pads[b])))
        cp_["reads"] = reads

    # --------------------------------------------------------------- the cells
    # cell-to-cell links for heat and pressure: (axis, flat lo cell, flat hi
    # cell, flat face index into comps[axis]); boundary faces separately
    links = []
    bfaces = []
    for a in range(3):
        cp_ = comps[a]
        lo, hi = cp_["lo"].ravel(), cp_["hi"].ravel()
        fidx = np.arange(lo.size)
        inner = (lo >= 0) & (hi >= 0)
        links.append((a, lo[inner], hi[inner], fidx[inner]))
        for end, sign in ((0, -1), (1, 1)):
            if per[a]:
                continue
            m = (hi >= 0) & (lo < 0) if end == 0 else (lo >= 0) & (hi < 0)
            cell = hi[m] if end == 0 else lo[m]
            bfaces.append((a, sign, cell, fidx[m], S["xyz"[a] + ("-" if end == 0 else "+")]))

    fl = fluid.ravel()
    # pressure unknowns: fluid cells
    pnum = -np.ones(N, np.int64)
    pnum[fl] = np.arange(int(fl.sum()))
    npn = int(fl.sum())

    # ------------------------------------------------------------ the state
    film_t = amb + 5.0 if film is None else float(film)
    air = props(film_t)
    if init is not None:
        U = [u.copy() for u in init.u]
        T = init.t.ravel().copy()
        P = init.p.ravel().copy()
    else:
        U = [cp_["val"].copy() for cp_ in comps]
        T = np.full(N, amb)
        T[fix_mask.ravel()] = fix_t.ravel()[fix_mask.ravel()]
        P = np.zeros(N)
    kcell = np.where(fl, air["k"], ksol.ravel())

    fan_faces = None
    if fan is not None:
        fm = np.asarray(fan.cells, bool).ravel()
        cp_ = comps[fan.axis]
        lo, hi = cp_["lo"].ravel(), cp_["hi"].ravel()
        inside = (lo >= 0) & (hi >= 0)
        inside[inside] = fm[lo[inside]] & fm[hi[inside]]
        fan_faces = np.nonzero(inside & (cp_["num"].ravel() >= 0))[0]
        if fan_faces.size == 0:
            raise ValueError("the fan zone holds no open face along its axis")
        layers = max(int(round(fm.sum() / max(np.count_nonzero(np.any(
            fm.reshape(shape), axis=fan.axis)), 1))), 1)

    hist = []
    solve_t = _TSolve()
    fan_dp = None
    converged = False
    ml_p = None
    it = 0
    d_store = [None, None, None]
    q_in_total = float(q_cell.sum())

    mres = np.inf
    dc_on = False              # deferred correction: on once the flow nearly conserves mass, and stays on
    for it in range(1, max_iter + 1):
        dc_on = dc_on or mres < 1e-2
        rho, mu = air["rho"], air["mu"]
        padded = [_pad(U[c], per) for c in range(3)]
        flat_pad = [p_.ravel() for p_ in padded]
        # ---- momentum, each component
        ustar = []
        for a in range(3):
            cp_ = comps[a]
            n = cp_["n"]
            if n == 0:
                ustar.append(U[a].copy())
                d_store[a] = np.zeros(0)
                continue
            pos = cp_["pos"]
            uf = U[a][pos]
            aP = np.zeros(n)
            src = np.zeros(n)
            rows, cols, vals = [], [], []
            ar = np.arange(n)
            dc = np.zeros(n)
            for (b, s, kind, nidx, nval), (_, r1, r2) in zip(cp_["nb"], cp_["reads"]):
                F = rho * A * 0.5 * (flat_pad[b][r1] + flat_pad[b][r2])      # +b direction, kg/s
                D = np.where(kind == 0, mu * A / h, np.where(kind == 1, mu * A / h,
                                                             np.where(kind == 2, 2.0 * mu * A / h, 0.0)))
                conv_in = np.maximum(-s * F, 0.0)
                a_nb = D + conv_in
                out_f = s * F
                # the neighbour's value
                phi_nb = nval.copy()
                phi_nb[kind == 0] = uf[nidx[kind == 0]]
                zg = kind == 3
                if b != a:
                    # across an open side, air coming in brings the room's
                    # (zero) velocity along the side, not the face's own
                    still = zg & (conv_in > 0)
                    zg = zg & ~still
                    phi_nb = np.where(still, 0.0, phi_nb)
                phi_nb = np.where(zg, uf, phi_nb)
                # deferred correction: central minus upwind face value, explicit
                if deferred and dc_on:
                    face_c = 0.5 * (uf + phi_nb)
                    face_u = np.where(out_f > 0, uf, phi_nb)
                    dc -= deferred * out_f * (face_c - face_u) * (kind != 2)
                # assemble
                a_keep = np.where(zg, 0.0, a_nb)
                aP += a_keep + out_f
                m0 = kind == 0
                rows.append(ar[m0])
                cols.append(nidx[m0])
                vals.append(-a_nb[m0])
                fixedn = (kind == 1) | (kind == 2)
                src[fixedn] += a_nb[fixedn] * nval[fixedn]
            # inflow across a zero-gradient side brings the face's own value:
            # it cancels on both sides; keep the diagonal no smaller than the
            # sum of its neighbours (a lagging mass imbalance must not flip it)
            sum_nb = np.zeros(n)
            for r_, v_ in zip(rows, vals):
                sum_nb += np.bincount(r_, -v_, minlength=sum_nb.size)
            aP = np.maximum(aP, sum_nb + 1e-30)
            aP = np.maximum(aP, mu * A / h * 1e-3)
            # pressure and buoyancy
            lo_c, hi_c = cp_["lo"][pos], cp_["hi"][pos]
            p_lo = np.where(lo_c >= 0, P[np.maximum(lo_c, 0)], 0.0)
            p_hi = np.where(hi_c >= 0, P[np.maximum(hi_c, 0)], 0.0)
            ob_lo = lo_c < 0
            ob_hi = hi_c < 0
            # an open face: zero static pressure outside where air leaves, zero
            # total pressure where it comes in (the room's still air speeds up
            # into the opening: p = -rho u^2 / 2, taken implicitly)
            p_lo[ob_lo] = 0.0
            p_hi[ob_hi] = 0.0
            inflow = (ob_lo & (uf > 0)) | (ob_hi & (uf < 0))
            aP = aP + np.where(inflow, 0.5 * rho * np.abs(uf) * A, 0.0)
            src += (p_lo - p_hi) * A
            t_lo = np.where(lo_c >= 0, T[np.maximum(lo_c, 0)], np.nan)
            t_hi = np.where(hi_c >= 0, T[np.maximum(hi_c, 0)], np.nan)
            tf = np.where(np.isnan(t_lo), t_hi, np.where(np.isnan(t_hi), t_lo, 0.5 * (t_lo + t_hi)))
            if gvec[a] != 0.0:
                src += -rho * air["beta"] * (tf - amb) * gvec[a] * V
            if fan is not None and a == fan.axis:
                ff = cp_["num"].ravel()[fan_faces]
                if fan.dp is not None:
                    dp = float(fan.dp) * min(1.0, it / 30.0)        # brought up over the first passes
                    air["fan dp Pa"] = round(dp, 3)
                else:
                    flow = float(np.sum(U[a].ravel()[fan_faces]) * A * fan.sign) / max(layers - 1, 1)
                    dp0, q0 = fan.curve
                    target = min(max(dp0 * (1.0 - flow / (fan.share * q0)), 0.0), dp0)
                    # read off the line explicitly, so moved toward it slowly:
                    # the flow answers a pressure change over many passes
                    # (from zero: the full shut-off head on still air throws the
                    # first passes far past the answer)
                    dp = 0.1 * target if fan_dp is None else fan_dp + 0.1 * (target - fan_dp)
                    fan_dp = dp
                    air["fan dp Pa"] = round(dp, 3)
                    air["fan flow m3/s"] = flow
                src[ff] += fan.sign * dp * A / max(layers - 1, 1)     # dp over the zone's faces in series
            src += dc
            # under-relaxation
            aPr = aP / au
            src += (1.0 - au) * aPr * uf
            M = sp.csr_matrix((np.concatenate(vals) if vals else np.zeros(0),
                               (np.concatenate(rows) if rows else np.zeros(0, int),
                                np.concatenate(cols) if cols else np.zeros(0, int))), shape=(n, n))
            M = M + sp.diags(aPr)
            Minv = sp.diags(1.0 / aPr)
            x, _ = spla.bicgstab(M, src, x0=uf, rtol=1e-4, maxiter=60, M=Minv)
            new = U[a].copy()
            new[pos] = x
            ustar.append(new)
            # SIMPLEC: d = A / (aP/alpha - sum a_nb)
            d_store[a] = A / np.maximum(aPr - sum_nb, 1e-30)
        # ---- pressure correction
        mass = np.zeros(N)                     # net outflow per cell, kg/s
        rows, cols, vals = [], [], []
        diag = np.zeros(N)
        for a in range(3):
            cp_ = comps[a]
            if cp_["n"] == 0:
                continue
            numf = cp_["num"].ravel()
            us = ustar[a].ravel()
            lo, hi = cp_["lo"].ravel(), cp_["hi"].ravel()
            # every face with flow: unknown or fixed (inlet)
            flowing = (numf >= 0) | (np.abs(cp_["val"].ravel()) > 0)
            fi = np.nonzero(flowing)[0]
            m_ = rho * A * us[fi]
            l_, h_ = lo[fi], hi[fi]
            mass += np.bincount(l_[l_ >= 0], m_[l_ >= 0], minlength=mass.size)
            mass += np.bincount(h_[h_ >= 0], -m_[h_ >= 0], minlength=mass.size)
            un = numf >= 0
            fu = np.nonzero(un)[0]
            c_ = rho * A * d_store[a][numf[fu]]
            l_, h_ = lo[fu], hi[fu]
            both = (l_ >= 0) & (h_ >= 0)
            rows += [l_[both], h_[both]]
            cols += [h_[both], l_[both]]
            vals += [-c_[both], -c_[both]]
            diag += np.bincount(l_[l_ >= 0], c_[l_ >= 0], minlength=diag.size)
            diag += np.bincount(h_[h_ >= 0], c_[h_ >= 0], minlength=diag.size)
        r_ = np.concatenate(rows)
        c_ = np.concatenate(cols)
        v_ = np.concatenate(vals)
        Mp = sp.csr_matrix((v_, (pnum[r_], pnum[c_])), shape=(npn, npn))
        dg = diag[fl]
        dg = np.where(dg > 0, dg, 1.0)
        Mp = (Mp + sp.diags(dg * (1.0 + 1e-9))).tocsr()
        bp = -mass[fl]
        if ml_p is None or it % 25 == 1:
            import pyamg
            ml_p = pyamg.smoothed_aggregation_solver(Mp, symmetry="symmetric", max_coarse=500)
        pp = ml_p.solve(bp, tol=_P_TOL, accel="cg", maxiter=150) if npn > 1 else bp / dg
        pcorr = np.zeros(N)
        pcorr[fl] = pp
        P = P + ap * pcorr
        # correct the face velocities
        mass_res = float(np.abs(mass[fl]).sum())
        u_prev = [u_.copy() for u_ in U]
        for a in range(3):
            cp_ = comps[a]
            if cp_["n"] == 0:
                U[a] = ustar[a]
                continue
            pos = cp_["pos"]
            lo_c, hi_c = cp_["lo"][pos], cp_["hi"][pos]
            pl = np.where(lo_c >= 0, pcorr[np.maximum(lo_c, 0)], 0.0)
            ph = np.where(hi_c >= 0, pcorr[np.maximum(hi_c, 0)], 0.0)
            new = ustar[a]
            new[pos] = new[pos] + d_store[a] * (pl - ph)
            U[a] = new
        # ---- energy over solid and fluid
        kcell = np.where(fl, air["k"], ksol.ravel())
        rc = air["rho"] * air["cp"]
        rows, cols, vals = [], [], []
        diagT = np.zeros(N)
        f_out = np.zeros(N)                                  # advection out of / into each cell, W/K
        f_in = np.zeros(N)
        bT = q_cell.ravel().copy()
        dcT = np.zeros(N)
        for a, lo, hi, fidx in links:
            g = A / (0.5 * h / kcell[lo] + 0.5 * h / kcell[hi])
            F = rc * A * U[a].ravel()[fidx]                 # lo -> hi, W/K
            fp, fm = np.maximum(F, 0.0), np.maximum(-F, 0.0)
            # cell lo: neighbour hi takes g + fm (inflow from hi), cell hi: g + fp
            rows += [lo, hi]
            cols += [hi, lo]
            vals += [-(g + fm), -(g + fp)]
            diagT += np.bincount(lo, g, minlength=diagT.size)
            diagT += np.bincount(hi, g, minlength=diagT.size)
            f_out += np.bincount(lo, fp, minlength=f_out.size)
            f_out += np.bincount(hi, fm, minlength=f_out.size)
            f_in += np.bincount(hi, fp, minlength=f_in.size)
            f_in += np.bincount(lo, fm, minlength=f_in.size)
            if deferred and dc_on:
                # only once the flow nearly conserves mass: the explicit
                # correction on a flow still far from it (a fan spinning up)
                # drives the temperatures unbounded
                tc = 0.5 * (T[lo] + T[hi])
                tu = np.where(F > 0, T[lo], T[hi])
                corr = deferred * F * (tc - tu)              # extra carried lo -> hi
                dcT += np.bincount(lo, -corr, minlength=dcT.size)
                dcT += np.bincount(hi, corr, minlength=dcT.size)
        for a, sign, cell, fidx, side in bfaces:
            kind = side["kind"]
            if kind in ("open", "inlet"):
                F = rc * A * U[a].ravel()[fidx] * sign      # outward, W/K
                t_in = float(side.get("t", amb)) if side.get("t") is not None else amb
                # leaving: at its own temperature; coming in (or still): the
                # room's air at the opening, half a pitch out
                # (a solid cell on an open side is adiabatic there: the room
                # is only reached through the air)
                inn = (F <= 0.0) & fl[cell]
                g = np.where(inn, A * kcell[cell] / (0.5 * h), 0.0)
                diagT += np.bincount(cell, g, minlength=diagT.size)
                bT += np.bincount(cell, (np.maximum(-F, 0.0) + g) * t_in, minlength=bT.size)
                f_out += np.bincount(cell, np.maximum(F, 0.0), minlength=f_out.size)
                f_in += np.bincount(cell, np.maximum(-F, 0.0), minlength=f_in.size)
            elif kind == "wall" and side.get("t") is not None:
                g = A * kcell[cell] / (0.5 * h)
                diagT += np.bincount(cell, g, minlength=diagT.size)
                bT += np.bincount(cell, g * float(side["t"]), minlength=bT.size)
        bT += dcT
        # upwind: a cell's diagonal is what flows in (what flows out leaves at
        # its own temperature, which is the same once the flow conserves
        # mass); while it does not yet, this keeps every row summing to zero,
        # so no cell is a sink or a source of heat by a mass imbalance
        diagT += f_in
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        fm_ = fix_mask.ravel()
        keep = ~fm_[rows]
        MT = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(N, N))
        dT = np.where(fm_, 1.0, diagT)
        bT = np.where(fm_, fix_t.ravel(), bT)
        dT = np.where(dT > 0, dT, 1e-12)
        MT = (MT + sp.diags(dT)).tocsr()
        Tn = solve_t(MT, bT, T)
        dTmax = float(np.abs(Tn - T).max())
        T = T + at * (Tn - T)
        # ---- bookkeeping
        if film is None and it % 10 == 0:
            wet = _wet_t(links, fl, T, kcell)
            if wet is not None:
                ft = 0.5 * (amb + wet)
                if abs(ft - air["film C"]) > 0.25:
                    air = props(ft)
        umax = max(max(float(np.abs(U[c]).max()) if U[c].size else 0.0 for c in range(3)), 1e-12)
        du = max(float(np.abs(U[c] - u_prev[c]).max()) if U[c].size else 0.0 for c in range(3)) / umax
        flow_scale = max(max((rho * A * np.abs(U[c]).max() if U[c].size else 0.0) for c in range(3)), 1e-30)
        mres = mass_res / flow_scale / max(npn, 1) ** 0.5
        q_fix = _fixed_heat(links, fix_mask.ravel(), T, kcell, A, h, bfaces)
        bal = _balance(bfaces, U, T, rc, A, amb, q_in_total + q_fix, S, kcell, h, fl)
        hist.append({"it": it, "mass": mres, "du": du, "dT": dTmax, "balance": bal["closure"], "umax": float(
            max(np.abs(U[c]).max() for c in range(3)))})
        if verbose and (it % report_every == 0 or it == 1):
            print(f"airflow {it}: mass {mres:.2e} du {du:.1e} dT {dTmax:.2e} K balance {bal['closure']:+.3%} "
                  + (f"fan {air.get('fan dp Pa', 0.0):.2f} Pa " if fan is not None else "")
                  + f"umax {hist[-1]['umax']:.4f} m/s T {T.min():.2f}..{T.max():.2f} C", flush=True)
        if it > 20 and mres < tol and du < tol and dTmax < 1e-3 and abs(bal["closure"]) < 5e-3:
            converged = True
            break
    solve_t.free()
    Tc = T.reshape(shape)
    vel = np.zeros((3,) + shape)
    for a in range(3):
        ua = U[a]
        if per[a]:
            vel[a] = 0.5 * (ua + np.roll(ua, -1, axis=a))
        else:
            sl1 = [slice(None)] * 3
            sl2 = [slice(None)] * 3
            sl1[a] = slice(0, -1)
            sl2[a] = slice(1, None)
            vel[a] = 0.5 * (ua[tuple(sl1)] + ua[tuple(sl2)])
    walls = _walls(links, fl, T, kcell, A, h, shape)
    q_fix = _fixed_heat(links, fix_mask.ravel(), T, kcell, A, h, bfaces)
    bal = _balance(bfaces, U, T, air["rho"] * air["cp"], A, amb, q_in_total + q_fix, S, kcell, h, fl)
    bal["from fixed cells W"] = q_fix
    if verbose:
        print(f"airflow done: {it} passes, converged {converged}, {time.time() - t0:.0f} s, "
              f"balance {bal['closure']:+.2%}", flush=True)
    return AirflowResult(t=Tc, u=U, p=P.reshape(shape), velocity=vel, solid=solid, pitch=float(pitch), ambient=amb,
                         walls=walls, balance=bal, iterations=it, converged=converged, history=hist, air=air,
                         seconds=round(time.time() - t0, 1))


class _TSolve:
    """The energy solve: one nonsymmetric system over solid and air.

    With upwind advection (the central part is deferred to the right-hand
    side) every row has a positive diagonal, non-positive neighbours and a
    row sum of zero or more: an M-matrix. Classical algebraic multigrid
    (Ruge & Stueben 1987) is built for exactly that. It coarsens along the
    strong couplings of the matrix itself, so aluminium beside air (a
    conductance contrast of about 8000) and the direction the air carries
    heat are both in the hierarchy, where a method built on the symmetric
    part or on geometry loses them. The hierarchy preconditions BiCGSTAB.

    The flow changes little from pass to pass, so a hierarchy serves the
    following passes too (184,000 cells, a fin stack in open air: 5
    iterations with the hierarchy of 60 passes before, as with a fresh one)
    and is rebuilt when the iterations it takes have grown by half.

    Against factorizing every pass (what this did before), on that box: 0.5 s
    a pass where PARDISO took 6.3 s on one core, the work grows with the
    cells and not their square, and it needs no PARDISO (there is none on
    Apple Silicon). The factorization (PARDISO on all the run's cores when
    pypardiso is installed, the `fast` extra; SuperLU otherwise) stays as
    the way out when the iteration does not converge."""

    def __init__(self):
        self.ml = None
        self.prec = None
        self.fresh_its = None
        self.pardiso = None
        self.analysed = False
        self.stats = {"amg setups": 0, "iterations": 0, "solves": 0, "factorizations": 0}

    def free(self):
        """Give MKL's factorization memory back (it is not freed with the
        object, and a few solves in one process otherwise pile up)."""
        self.ml = self.prec = None
        if self.pardiso:
            try:
                self.pardiso.free_memory(everything=True)
            except Exception:
                pass

    def _factor(self, M):
        M = M.tocsr()
        M.sort_indices()
        if self.pardiso is None and self.pardiso is not False:
            try:
                import pypardiso
                self.pardiso = pypardiso.PyPardisoSolver()
            except Exception:
                self.pardiso = False
        if self.pardiso:
            from .nonlinear import mkl_threads
            s = self.pardiso
            mkl_threads(s)
            s._check_A(M)
            z = np.zeros((M.shape[0], 1))
            if not self.analysed:
                s.set_phase(11)
                s._call_pardiso(M, z)
                self.analysed = True
            s.set_phase(22)
            s._call_pardiso(M, z)
            s.set_phase(33)
            return lambda r: s._call_pardiso(M, np.asarray(r, float).reshape(-1, 1)).ravel()
        lu = spla.splu(M.tocsc(), permc_spec="MMD_AT_PLUS_A")
        return lu.solve

    def _setup(self, M):
        import warnings

        import pyamg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ml = pyamg.ruge_stuben_solver(M, max_coarse=500)
        self.prec = self.ml.aspreconditioner(cycle="V")
        self.fresh_its = None
        self.stats["amg setups"] += 1

    def _krylov(self, M, b, x0, atol, maxiter):
        its = [0]

        def count(_):
            its[0] += 1
        x, info = spla.bicgstab(M, b, x0=x0, M=self.prec, rtol=0.0, atol=atol, maxiter=maxiter, callback=count)
        ok = info == 0 and bool(np.isfinite(x).all())
        return x, ok, its[0]

    def __call__(self, M, b, x0):
        n = M.shape[0]
        if n < _DIRECT_BELOW:
            return spla.spsolve(M.tocsc(), b)
        r0 = float(np.linalg.norm(b - M @ x0))
        if r0 == 0.0:
            return x0.copy()
        atol = _T_RTOL * r0 + 1e-12 * float(np.linalg.norm(b))       # the second term: rounding's floor
        self.stats["solves"] += 1
        for attempt in range(2):
            fresh = self.ml is None
            if fresh:
                self._setup(M)
            cap = 40 if self.fresh_its is None else max(int(1.5 * self.fresh_its) + 2, 8)
            x, ok, its = self._krylov(M, b, x0, atol, cap)
            self.stats["iterations"] += its
            if ok:
                if self.fresh_its is None:
                    self.fresh_its = its
                return x
            if fresh:
                break
            self.ml = None                      # an old hierarchy that has stopped serving: once more with a new one
        self.ml = None
        self.stats["factorizations"] += 1
        return self._factor(M)(b)


def _wet_t(links, fl, T, kcell):
    ts = []
    for a, lo, hi, fidx in links:
        m = fl[lo] != fl[hi]
        if m.any():
            l_, h_ = lo[m], hi[m]
            ts.append((kcell[l_] * T[l_] + kcell[h_] * T[h_]) / (kcell[l_] + kcell[h_]))
    if not ts:
        return None
    return float(np.concatenate(ts).mean())


def _walls(links, fl, T, kcell, A, h, shape):
    out = {k: [] for k in ("solid_cell", "fluid_cell", "axis", "normal_sign", "q", "t_wall", "center")}
    for a, lo, hi, fidx in links:
        m = fl[lo] != fl[hi]
        if not m.any():
            continue
        l_, h_ = lo[m], hi[m]
        s_is_lo = ~fl[l_]
        sc = np.where(s_is_lo, l_, h_)
        fc = np.where(s_is_lo, h_, l_)
        g = A / (0.5 * h / kcell[sc] + 0.5 * h / kcell[fc])
        q = g * (T[sc] - T[fc])
        tw = (kcell[sc] * T[sc] + kcell[fc] * T[fc]) / (kcell[sc] + kcell[fc])
        ijk = np.array(np.unravel_index(sc, shape)).T.astype(float) + 0.5
        sgn = np.where(s_is_lo, 1.0, -1.0)
        ijk[:, a] += 0.5 * sgn
        out["solid_cell"].append(sc)
        out["fluid_cell"].append(fc)
        out["axis"].append(np.full(sc.size, a))
        out["normal_sign"].append(sgn)
        out["q"].append(q)
        out["t_wall"].append(tw)
        out["center"].append(ijk * h * 1e3)
    res = {k: (np.concatenate(v) if v else np.zeros((0, 3) if k == "center" else 0)) for k, v in out.items()}
    res["area"] = A
    return res


def _fixed_heat(links, fixm, T, kcell, A, h, bfaces):
    """Heat the fixed-temperature cells give to the rest (by conduction:
    they are solids, or still air held at a temperature)."""
    if not fixm.any():
        return 0.0
    q = 0.0
    for a, lo, hi, fidx in links:
        m = fixm[lo] != fixm[hi]
        if m.any():
            l_, h_ = lo[m], hi[m]
            g = A / (0.5 * h / kcell[l_] + 0.5 * h / kcell[h_])
            q += float(np.sum(np.where(fixm[l_], 1.0, -1.0) * g * (T[l_] - T[h_])))
    return q


def _balance(bfaces, U, T, rc, A, amb, q_in, S, kcell, h, fl):
    """Heat into the box (sources) against what leaves: carried out of
    openings over the ambient, through isothermal walls."""
    carried = 0.0
    walls = 0.0
    for a, sign, cell, fidx, side in bfaces:
        kind = side["kind"]
        if kind in ("open", "inlet"):
            F = rc * A * U[a].ravel()[fidx] * sign
            t_in = float(side["t"]) if side.get("t") is not None else amb
            g = np.where((F <= 0.0) & fl[cell], A * kcell[cell] / (0.5 * h), 0.0)
            carried += float(np.sum(np.where(F > 0, F * (T[cell] - amb), F * (t_in - amb)) + g * (T[cell] - t_in)))
        elif kind == "wall" and side.get("t") is not None:
            g = A * kcell[cell] / (0.5 * h)
            walls += float(np.sum(g * (T[cell] - float(side["t"]))))
    out = carried + walls
    ref = max(abs(q_in), abs(walls), abs(carried), 1e-12)
    closure = (out - q_in) / ref
    return {"heat in W": q_in, "carried out W": carried, "through walls W": walls, "closure": closure}
