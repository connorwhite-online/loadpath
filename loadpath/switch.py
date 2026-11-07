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

import time
from dataclasses import dataclass, field

import numpy as np

from .domain import Domain
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
    clr = b.get("clearance")
    cn, cg = None, None
    if clr:
        cs = shape_from_spec(clr["shape"]) if "shape" in clr else None
        cn = _region_nodes(study, domain.mesh, clr.get("region"), cs, "switch.clearance")
        cg = float(clr["gap"])
    hold = []
    for h in b.get("hold") or []:
        name = h["case"]
        if name not in domain.case_names:
            raise ValueError(f"switch.hold: unknown load case {name!r}; cases: {domain.case_names}")
        hold.append((domain.case_names.index(name), float(h["maxDeflection"])))
    return SwitchSpec(output_nodes=out_nodes, axis=axis, sign=sign, gap=float(o.get("gap", 0.0)),
                      trigger=_force(o.get("trigger", 0.0), unit), travel=float(o.get("travel", 0.0)),
                      presses=presses, press_axis=pax, press_sign=psg,
                      press_radius=float(b.get("pressRadius", 1.0)),
                      max_travel=b.get("maxTravel"), clearance_nodes=cn, clearance=cg, hold=hold)


# ---------------------------------------------------------------------------
# solves


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
    DIRECT_DOFS = 120_000

    def __init__(self, domain: Domain, nu, rtol: float = 1e-8):
        from scipy import ndimage  # noqa: F401
        mesh = domain.mesh
        self.domain = domain
        self.nu = nu
        self.rtol = rtol
        self.support_touch = domain.support_nodes[mesh.enode].any(axis=1)
        self._lu = None
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

    def factor(self, xp: np.ndarray):
        """Set up the reduced system for density xp (unit modulus)."""
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
        nodes = mesh.node_id(sub.node_ijk)
        dofs = (3 * nodes[:, None] + np.arange(3)).ravel()
        fixed = d.fixed[dofs]
        free = np.flatnonzero(~fixed)
        self.dofs, self.free = dofs, free
        if len(free) <= self.DIRECT_DOFS:
            k = Stiffness(sub, self.nu, fixed).assemble(e)
            if d.springs is not None:
                k = (k + d.springs[dofs][:, dofs]).tocsr()
            a = k[free][:, free].tocsc()
            self._lu = splu(a, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
            self._mg = None
        else:
            from .mg import GridElasticity
            full = np.zeros(mesh.n_elem)
            full[el] = e      # 0 drops a cell outright
            if self._mg is None:
                self._mg = GridElasticity(mesh, self.nu, d.fixed, springs=d.springs, rtol=self.rtol, maxiter=2000)
            self._mg.set_moduli(full)
            self._lu = None

    def solve(self, fs: list, tag: str = "f") -> list:
        n = self.domain.mesh.n_dof
        if self._lu is not None:
            b = np.stack([np.asarray(f)[self.dofs][self.free] for f in fs], axis=1)
            x = self._lu.solve(b)
            out = []
            for j in range(len(fs)):
                u = np.zeros(n)
                u[self.dofs[self.free]] = x[:, j]
                out.append(u)
            return out
        out = self._mg.solve_many(fs, self.prev.get(tag))
        self.prev[tag] = [np.array(u) for u in out]
        return [np.asarray(u) for u in out]


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


def _problem(domain: Domain, spec: SwitchSpec) -> _Problem:
    mesh = domain.mesh
    loads = []
    for p in spec.presses:
        nodes = _press_nodes(domain, p.at, spec.press_axis, spec.press_sign, spec.press_radius)
        loads.append(_unit_load(nodes, spec.press_axis, spec.press_sign, mesh.n_dof, domain.fixed))
    l = _unit_load(spec.output_nodes, spec.axis, spec.sign, mesh.n_dof, domain.fixed)
    if not np.abs(l).sum():
        raise ValueError("switch.output: every plunger node is held by a support")
    hold = []
    for ci, lim in spec.hold:
        f = domain.loads[ci]
        # force-weighted mean travel along each nodal force: f . u / sum |f_node|
        # (a squeeze's two sides cancel in the net force)
        hold.append((f, float(np.linalg.norm(f.reshape(-1, 3), axis=1).sum()), lim))
    cd = None
    if spec.clearance_nodes is not None:
        cd = 3 * np.flatnonzero(spec.clearance_nodes) + spec.press_axis
        cd = cd[~domain.fixed[cd]]
    return _Problem(spec, loads, l, loads, hold, cd)


def _modulus(xp, E_MIN, PENALTY):
    return E_MIN + xp ** PENALTY * (1 - E_MIN)


def _evaluate(domain: Domain, nu, prob: _Problem, xp: np.ndarray, modulus: float, solves: _Solves,
              grads: bool = True):
    """Fire forces and the constraint quantities for density xp (per
    element, preserve included), and their gradients wrt xp. Moduli in
    MPa: the solves run at unit modulus and displacements divide by E."""
    from .simp import E_MIN, PENALTY
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
    N = spec.gap + spec.travel + spec.trigger * c
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
        # lift them a little: conservative)
        clr_q = [spec.press_sign * u[prob.clr_dofs] * Fi for u, Fi in zip(U, F)]
        out["clearanceMax"] = float(max(q.max() for q in clr_q))
    if not grads:
        return out, None
    edof = mesh.edof
    ke = hex8_ke(nu) * mesh.pitch
    dEdx = PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)   # unit modulus

    # d(v . K^-1 w) / dxp = -(K^-1 v)_e ke (K^-1 w)_e dE/dxp, with K at
    # modulus E: the stored solutions already carry 1/E, and dK = E dE ke
    def dpair(v, w):
        return -modulus * dEdx * _pair(edof, ke, v, w)

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
        qs = np.concatenate(clr_q) / spec.clearance
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
            r[prob.clr_dofs] = wi * F[i] * spec.press_sign / spec.clearance
            adj.append(r)
        mus = solves.solve(adj, "clr")
        g = np.zeros(mesh.n_elem)
        for i in range(nP):
            wi = w[i * n:(i + 1) * n]
            mu = np.asarray(mus[i]) / modulus
            g += dpair(mu, U[i])
            zi = spec.press_sign * U[i][prob.clr_dofs] / spec.clearance
            g += float(wi @ zi) * dF[i]
        G["clearance"] = g
    return out, G


def fire_forces(domain: Domain, nu, spec: SwitchSpec, xphys: np.ndarray, modulus: float) -> dict:
    """The linear fire force and press travel at the click for each press
    point, for a given density (1 = solid) on this domain."""
    prob = _problem(domain, spec)
    out, _ = _evaluate(domain, nu, prob, np.asarray(xphys, float), modulus, _Solves(domain, nu), grads=False)
    return {"forces": out["F"], "travel": out["travel"], "plungerCompliance": out["c"],
            "hold": out["hold"], **({"clearanceMax": out["clearanceMax"]} if "clearanceMax" in out else {})}


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


def _thin_skin(study, domain: Domain):
    """A skin thinner than two cells on this grid, as a density floor.

    The preserved skin is whole cells: at 1 mm cells a 0.7 mm skin is a
    1 mm (or 2 mm) one, and bending goes as thickness cubed, so the click
    came out three times too stiff on a coarse grid and nothing the
    optimizer added could bring it down. Here each cell the skin passes
    through gets its share of skin volume phi (4^3 samples) as a floor on
    its density, and the rest of the cell is design space. With SIMP's
    E ~ x^3 a layer of cells at phi bends like the true skin (phi h = t in
    one cell: E h^3 phi^3 = E t^3); membrane stiffness reads low, by phi^2.
    A coarse-grid stand-in: from two cells per skin thickness up the skin
    is plain preserved cells.

    Returns (domain with those cells as design, floor per element)."""
    import dataclasses

    floor = np.zeros(domain.n_elem)
    thin = [(rid, t) for rid, t in study.shells if rid == "skin" and t < 2.0 * domain.pitch]
    if not thin:
        return domain, floor
    shape, _ = study.region_shape("skin")
    c = domain.mesh.elem_centers()
    sub = (np.arange(4) + 0.5) / 4 - 0.5
    offs = np.array([[a, b, d] for a in sub for b in sub for d in sub]) * domain.pitch
    phi = np.zeros(len(c))
    for o in offs:
        phi += shape.contains(c + o)
    phi /= len(offs)
    # other preserves (the plunger, bosses) stay whole cells
    others = [r.shape for r in study.preserve if r.id != "skin"]
    hard = np.zeros(len(c), bool)
    if others:
        from .domain import _fraction
        hard = _fraction(others, c, domain.pitch) >= 0.5
    skin = (phi > 0) & ~hard
    floor[skin] = np.clip(phi[skin], 0.0, 1.0)
    pres = domain.preserve & ~skin
    return dataclasses.replace(domain, preserve=pres), floor


def optimize_switch(study, pitches=None, iterations=None, x0: float = 0.5, beta_max: float = 16.0,
                    beta_every: int = 15, move: float = 0.1, on_iter=None, tol: float = 0.003) -> SwitchResult:
    """Minimize volume subject to the study's `switch` block (see the module
    docstring), on each pitch in turn (coarse to fine). Returns the last
    level's design, its fire forces and the per-level summaries."""
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
    for li, pitch in enumerate(pitches):
        domain, floor = _thin_skin(study, build_domain(study, pitch))
        spec = switch_from_study(study, domain)
        prob = _problem(domain, spec)
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
        nP = len(spec.presses)
        m = sum(p.max is not None for p in spec.presses) + sum(p.min is not None for p in spec.presses) + \
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
            held = floor > xp
            xp[held] = floor[held]
            dp = np.where(held, 0.0, dp)
            out, G = _evaluate(domain, nu, prob, xp, modulus, solves)
            F = out["F"]
            cons, dcons, names = [], [], []

            def add(val, grad, name):
                g = ht @ ((grad * dp) / hs)
                cons.append(val)
                dcons.append(g[design])
                names.append(name)

            for i, p in enumerate(spec.presses):
                if p.max is not None:
                    add(float(np.log(F[i] / p.max)), G["logF"][i], f"max{i}")
                if p.min is not None:
                    add(float(np.log(p.min / F[i])), -G["logF"][i], f"min{i}")
            if spec.max_travel is not None:
                for i in range(nP):
                    add(out["travel"][i] / spec.max_travel - 1.0, G["travel"][i] / spec.max_travel, f"travel{i}")
            for k, (_, _, lim) in enumerate(prob.hold):
                add(out["hold"][k] / lim - 1.0, G["hold"][k] / lim, f"hold{k}")
            if G["clearance"] is not None:
                add(out["clearanceKS"] - 1.0, G["clearance"], "clearance")
            vol = float(xp[design].sum()) / nd
            dvol = (ht @ (dp / hs))[design] / nd
            if not cons:
                cons, dcons = [vol - 1.0], [dvol * 0]
            xn = x.copy()
            xn[design] = mma.update(x[design], vol * 10.0, dvol * 10.0, np.array(cons), np.array(dcons))
            change = float(np.abs(xn - x).max())
            x = xn
            worst = float(max(cons)) if cons else 0.0
            rec = {"level": li, "pitch": domain.pitch, "iteration": it, "volume": vol, "forces": F.tolist(),
                   "worstViolation": worst, "beta": beta, "change": change, "seconds": round(time.time() - t0, 1)}
            history.append(rec)
            if on_iter is not None:
                on_iter(rec, xp)
            since += 1
            if beta < beta_max and (since >= beta_every or (change < 2 * tol and since >= 5)):
                beta, since = min(2 * beta, beta_max), 0
            elif beta >= beta_max and change < tol and worst < 0.01:
                break
        xt = (h @ x) / hs
        xp = _project(xt, beta, 0.5)
        xp[pres] = 1.0
        xp = np.maximum(xp, floor)
        out, _ = _evaluate(domain, nu, prob, xp, modulus, solves, grads=False)
        levels.append({"pitch": domain.pitch, "cells": int(n), "iterations": it, "volume": float(xp[design].mean()),
                       "forces": out["F"].tolist(), "seconds": round(time.time() - t0, 1)})
        prev_dom, prev_x = domain, x
    rep = {"presses": [{"at": p.at.tolist(), "max": p.max, "min": p.min, "force": float(f), "forceGf": float(f / GF),
                        "travel": float(d),
                        "ok": bool((p.max is None or f <= p.max * 1.02) and (p.min is None or f >= p.min * 0.98))}
                       for p, f, d in zip(spec.presses, out["F"], out["travel"])],
           "plungerCompliance": out["c"], "hold": out["hold"].tolist()}
    if "clearanceMax" in out:
        rep["clearanceMax"] = out["clearanceMax"]
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
        thin = (floor > 0) & ~keep & (xphys >= THIN_KEEP)
        keep |= thin
        emod[thin] = modulus * xphys[thin] ** 3
    target = spec.gap + spec.travel
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
        u, lam = sv.solve_many(k, [f, l])
        a, c = float(l @ u), float(l @ lam)
        f_lin = (target + spec.trigger * c) / max(a, 1e-12)

        def tip(P):
            r = solve_nonlinear(sub, nu, mod_sub, fx_flat, P * f - spec.trigger * l, material=material, steps=steps,
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
                t1, r1 = tip(P1)
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
        rec = {"press": int(i), "at": spec.presses[i].at.tolist(), "linear": f_lin, "nonlinear": float(P),
               "nonlinearOverLinear": float(P / f_lin), "plungerTravel": t, "converged": bool(r.converged),
               "pressTravel": float(f @ r.u), "maxStrain": float(r.strain.max()) if r.strain is not None else None,
               "solves": n_solves, "cells": int(sub.n_elem)}
        if prob.clr_dofs is not None:
            rec["clearanceMax"] = float(spec.press_sign * (r.u[cm > 0]).max() if spec.press_sign > 0
                                        else (-r.u[cm > 0]).max())
        out.append(rec)
    return out


def _as_printed(xphys, floor):
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
    solid = np.where(extra > 0.5, 1.0, np.where(fl, floor, 0.0))
    return extra, solid


def run_switch(study, out_dir: str, pitches=None, iterations=None, nonlinear: bool | None = None,
               nonlinear_presses=None, on_progress=None) -> dict:
    """Optimize, mesh and check a study's `switch` block. Writes
    <name>.stl (the delivered part: the exact exterior skin and preserves
    joined to the optimized interior) and returns the report: per press the
    target, the linear fire force on the design and, when nonlinear is
    true, the large-deflection one on the solid part (experimental: on a
    thin-skin design the Newton solve crawls and does not converge yet)."""
    import os

    from .reconstruct import reconstruct

    os.makedirs(out_dir, exist_ok=True)

    def note(rec, _x):
        if on_progress is not None:
            on_progress({"stage": "optimize", **rec})

    r = optimize_switch(study, pitches=pitches, iterations=iterations, on_iter=note)
    rep = {"name": study.name, "pitches": [lv["pitch"] for lv in r.levels], "levels": r.levels,
           "volume": r.volume, "iterations": r.iterations, **r.report}
    nu = study.elastic(design=True)
    modulus = float(study.material.E)
    spec = switch_from_study(study, r.domain)
    # the same reading on the thresholded (as printed) design
    extra, solid = _as_printed(r.xphys, r.floor)
    ff = fire_forces(r.domain, nu, spec, np.maximum(solid, 1e-9), modulus)
    for p, f in zip(rep["presses"], ff["forces"]):
        p["thresholdedForceGf"] = float(f / GF)
    if on_progress is not None:
        on_progress({"stage": "mesh"})
    mesh, info = reconstruct(study, r.domain, extra)
    path = os.path.join(out_dir, f"{study.name}.stl")
    mesh.export(path)
    rep["mesh"] = path
    rep["mass"] = float(mesh.volume * study.material.density / 1000.0)
    rep["watertight"] = bool(mesh.is_watertight)
    if mesh.is_watertight:
        # The delivered mesh read back the way a drawn part is: its volume
        # share in each cell (the skin floor under it), the same fire forces.
        # Smoothing and the exact joins move material by a fraction of a
        # cell, and on 1 mm cells that is a few percent of force.
        from .domain import _fraction
        from .shapes import MeshShape
        if on_progress is not None:
            on_progress({"stage": "verify", "check": "delivered mesh fire force"})
        drawn = _fraction([MeshShape(mesh)], r.domain.mesh.elem_centers(), r.domain.pitch)
        fd = fire_forces(r.domain, nu, spec, np.maximum(np.maximum(drawn, r.floor), 1e-9), modulus)
        for p, f in zip(rep["presses"], fd["forces"]):
            p["deliveredForceGf"] = float(f / GF)
            p["deliveredOk"] = bool((p["max"] is None or f <= p["max"] * 1.02) and
                                    (p["min"] is None or f >= p["min"] * 0.98))
        rep["deliveredHold"] = [float(v) for v in fd["hold"]]
        rep["deliveredClearanceMax"] = fd.get("clearanceMax")
    rep["inWindow"] = sum(bool(p["ok"]) for p in rep["presses"])
    rep["deliveredInWindow"] = sum(bool(p.get("deliveredOk")) for p in rep["presses"])
    if nonlinear is None:
        nonlinear = bool((study.raw.get("switch") or {}).get("nonlinear", False))
    np.save(os.path.join(out_dir, "xphys.npy"), r.xphys)
    if nonlinear:
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
        nl = check_fire_nonlinear(dom_nl, nu, spec_nl, xp_nl, modulus, presses=nonlinear_presses, floor=floor_nl)
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
