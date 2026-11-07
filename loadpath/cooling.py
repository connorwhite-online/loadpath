"""
Cooling: design the inside of a fixed exterior around the heat its
electronics make.

An industrial designer hands over the exterior; the parts inside are known
(a processor, a radio, a charger chip, a battery) and each one makes heat
and has a temperature it must stay under. The exterior has its own limit:
the surface a hand holds must not get hotter than is comfortable or safe to
touch. Loadpath grows the inside (ribs and pads that conduct heat from the
hot parts out to the skin, spread it so no patch of the skin gets too hot,
and nothing more), with the least material, together with whatever
structural targets the same study carries (a click force, a grip, a drop:
switch.py, feel.py).

The model, per grid cell, inside the exterior:

- the printed material (the designed interior and the skin) conducts at its
  card's conductivity (thermal.CONDUCTIVITY), k(x) = k_air + x^3 (k - k_air),
  so a void cell is air, not nothing: a polymer conducts only 6-8 times
  better than air, and the air in the gaps carries a real share of the heat;
- the enclosed air conducts at `air.conductivity` (default K_AIR). In a
  hand-held product the gaps are a few mm, and a Rayleigh number across
  a 5 mm gap at a 10 K difference is about 100, far under the ~1700 where
  natural convection starts in an enclosed layer: the air inside is close
  to still, and conduction is what moves heat through it;
- each component is a block of its own conductivity (default
  COMPONENT_K: a package on a copper-filled board spreads heat well in
  plane) and gives off its `watts` over its volume; components listed with
  no watts (a battery, a board) only conduct;
- the outside of the exterior loses heat to the room by natural convection
  and radiation together, h (default H_OUT: about 5 W/(m^2 K) of each for
  a matte surface a few tens of K over room temperature);
- vents, if the exterior has them (`vents`: openings the designer drew,
  each a shape on the skin; loadpath never cuts the exterior), draw room
  air through by buoyancy when one is higher than another: the stack flow
  Q = Cd A sqrt(2 g H dT / T) carries rho c_p Q dT out, with dT the mean
  rise of the air inside. This is a well-mixed estimate (the air inside is
  taken as one temperature for the flow, and the heat it carries comes off
  every air cell in proportion to its air): good for a yes or no and a
  size, not for where the draft goes.

The limits are temperature limits: each component's hottest point under its
`maxTemp`, and the hottest point of the touched skin under `skin.maxTemp`
(or a contact class, TOUCH_LIMITS). Both are read as a p-norm of the node
temperatures (p = P_NORM) scaled to the true maximum of the last solve (Le
et al. 2010), so the gradient is smooth and the limit is the hot spot,
not an average. They go to switch.optimize_switch as constraints (min
volume, MMA, thin-skin cut cells, coarse to fine), so one study can carry a
click force, a grip feel and its cooling together. Each costs one adjoint
solve per iteration on a scalar system (CG with AMG), cheap next to the
elastic solves.

Two things the limits trade against each other: material from a hot chip
straight to the nearest wall cools the chip and makes a hot spot on the
wall over it; spread over more of the inside of the skin, it cools the chip
a little less and keeps every patch of the skin cooler. The optimizer finds
the balance the two limits ask for.

After the run the design is read again black and white, and the delivered
mesh on two finer grids (the delivered mesh's own fill, solid where a cell
is at least half inside it), so a reading is quoted only where the two
grids agree.

Not modelled: contact resistance between a component and a printed pad
resting on it (a thermal pad or paste is assumed), forced air (a fan),
the hand itself as a heat sink, and sunlight. The structure's own
softening with temperature (thermal.retention) is in thermal.py's
`thermal` block, not here.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .domain import Domain
from .fem import HexMesh
from .thermal import _FACE_M, hex8_kt

__all__ = ["CoolingProblem", "CoolingModel", "cooling_readings", "run_cooling", "TOUCH_LIMITS", "K_AIR", "H_OUT",
           "COMPONENT_K", "stack_conductance"]

K_AIR = 0.03            # W/(m K): air near 40 C is 0.027; a little for what convection there is
H_OUT = 10.0            # W/(m^2 K): natural convection ~5 + radiation ~5-6 (emissivity 0.9, 30-50 C)
COMPONENT_K = 20.0      # W/(m K): a package on a copper-filled board, in plane
P_NORM = 12.0
CD = 0.6                # discharge coefficient of a sharp-edged vent
RHO_CP_AIR = 1.16 * 1007.0    # J/(m^3 K) at ~30 C
G = 9.81
# Nodes up to which the system is factored (PARDISO); a finer read-back
# (millions of nodes) is solved with CG and AMG instead, in a fraction of
# the memory.
DIRECT_MAX = 600_000

try:
    import pypardiso  # noqa: F401
    _HAS_PARDISO = True
except ImportError:          # CG with AMG (thermal.ScalarSolver)
    _HAS_PARDISO = False

# Hottest a surface may be where it is touched, C, for plastic surfaces,
# after IEC 62368-1 Table 38 (TS1); metal limits are lower. Check the
# edition the product is certified to.
TOUCH_LIMITS = {
    "continuous": 43.0,   # held or touched for more than 8 h
    "held": 48.0,         # held for 10 min to 8 h (a handheld product in use)
    "brief": 60.0,        # touched for 1 to 10 min
}


# ---------------------------------------------------------------- the spec


@dataclass
class Source:
    id: str
    watts: float
    max_t: float | None
    k: float
    shape: object
    cells: np.ndarray = None          # thermal cell indices
    nodes: np.ndarray = None          # bool per thermal node


@dataclass
class Vent:
    id: str
    shape: object
    area: float | None                # mm^2; None: from the faces it covers
    faces: np.ndarray = None


@dataclass
class CoolingSpec:
    ambient: float
    h_out: float
    k_air: float
    k_material: float
    sources: list
    skin_max: float | None
    skin_shape: object                # where the touch limit applies (None: everywhere)
    vents: list
    up: np.ndarray                    # unit vector against gravity
    raw: dict = field(default_factory=dict)


def _num(v, label, positive=True):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or (positive and v <= 0):
        raise ValueError(f"{label} must be a {'positive ' if positive else ''}number")
    return float(v)


def _component_shape(study, item, label):
    """A component's own shape: an inline `shape`, or the keep-out (or
    preserve) `region` it is, as drawn (a keep-out with an assembly path is
    swept along it in the study; the part itself is only where it sits)."""
    from .spec import shape_from_spec
    if "shape" in item:
        return shape_from_spec(item["shape"], study.base_dir)
    if "region" not in item and "id" not in item:
        raise ValueError(f"{label} needs a `region` (a keepOut id) or a `shape`")
    rid = str(item.get("region") or item.get("id"))
    for key in ("keepOut", "preserve"):
        for r in study.raw.get(key) or []:
            if isinstance(r, dict) and str(r.get("id")) == rid and "shape" in r:
                return shape_from_spec(r["shape"], study.base_dir)
    return study.region_shape(rid)[0]


def cooling_from_study(study) -> CoolingSpec:
    from .spec import shape_from_spec
    from .thermal import conductivity
    cb = study.raw.get("cooling")
    if not isinstance(cb, dict):
        raise ValueError("the study needs a `cooling` block")
    ambient = float(cb.get("ambient", 25.0))
    k_mat = cb.get("conductivity")
    k_mat = _num(k_mat, "cooling.conductivity") if k_mat is not None else conductivity(study.material)
    air = cb.get("air") or {}
    k_air = _num(air.get("conductivity", K_AIR), "cooling.air.conductivity")
    sources = []
    for i, c in enumerate(cb.get("components") or []):
        label = f"cooling.components[{i}]"
        if not isinstance(c, dict):
            raise ValueError(f"{label} is {{region | shape, watts?, maxTemp?, conductivity?}}")
        w = float(c.get("watts", 0.0))
        if w < 0:
            raise ValueError(f"{label}.watts is the heat it gives off, W (>= 0)")
        mt = c.get("maxTemp")
        if mt is not None:
            mt = float(mt)
            if mt <= ambient:
                raise ValueError(f"{label}.maxTemp {mt:g} C is not over the ambient {ambient:g} C")
        sources.append(Source(str(c.get("region") or c.get("id") or f"component-{i}"), w, mt,
                              _num(c.get("conductivity", COMPONENT_K), f"{label}.conductivity"),
                              _component_shape(study, c, label)))
    if not sources or not any(s.watts > 0 for s in sources):
        raise ValueError("cooling.components: at least one component with watts > 0")
    sk = cb.get("skin") or {}
    smax = sk.get("maxTemp", sk.get("contact"))
    if isinstance(smax, str):
        if smax not in TOUCH_LIMITS:
            raise ValueError(f"cooling.skin.contact is one of {sorted(TOUCH_LIMITS)} or a maxTemp in C")
        smax = TOUCH_LIMITS[smax]
    elif smax is not None:
        smax = float(smax)
        if smax <= ambient:
            raise ValueError(f"cooling.skin.maxTemp {smax:g} C is not over the ambient {ambient:g} C")
    h = _num(sk.get("h", cb.get("h", H_OUT)), "cooling.skin.h")
    skin_shape = shape_from_spec(sk["shape"], study.base_dir) if sk.get("shape") else None
    vents = []
    for i, v in enumerate(cb.get("vents") or []):
        if not isinstance(v, dict) or "shape" not in v:
            raise ValueError(f"cooling.vents[{i}] is {{shape, area?}}: an opening the exterior already has")
        vents.append(Vent(str(v.get("id") or f"vent-{i}"), shape_from_spec(v["shape"], study.base_dir),
                          _num(v["area"], f"cooling.vents[{i}].area") if v.get("area") is not None else None))
    from .spec import _leg_direction
    up = -_leg_direction(cb.get("gravity", "-z"), "cooling.gravity")
    if not any(s.max_t is not None for s in sources) and smax is None:
        raise ValueError("cooling: nothing to hold: give a component maxTemp or skin.maxTemp")
    return CoolingSpec(ambient, h, k_air, k_mat, sources, smax, skin_shape, vents, up, cb)


def stack_conductance(vents_area_height, d_t: float, ambient: float) -> tuple[float, float]:
    """(W/K carried out, m^3/s) by buoyant flow through vents at different
    heights: [(area mm^2, height mm)], the inside air d_t K over the room.
    The lowest vents are the inlet and the highest the outlet (split at the
    mean height), in series: A = 1 / sqrt(1/A_in^2 + 1/A_out^2)."""
    if len(vents_area_height) < 2 or d_t <= 0:
        return 0.0, 0.0
    hs = np.array([h for _, h in vents_area_height], float)
    a = np.array([a for a, _ in vents_area_height], float) * 1e-6
    mid = 0.5 * (hs.min() + hs.max())
    lo, hi = hs <= mid, hs > mid
    if not lo.any() or not hi.any():
        return 0.0, 0.0
    a_in, a_out = a[lo].sum(), a[hi].sum()
    h_in = float((a[lo] * hs[lo]).sum() / a_in)
    h_out = float((a[hi] * hs[hi]).sum() / a_out)
    stack = (h_out - h_in) * 1e-3
    a_eff = 1.0 / np.sqrt(1.0 / a_in ** 2 + 1.0 / a_out ** 2)
    q = CD * a_eff * np.sqrt(2 * G * stack * d_t / (ambient + 273.15))
    return float(RHO_CP_AIR * q), float(q)


# --------------------------------------------------------------- the model


class CoolingModel:
    """The conduction problem on one grid: every cell inside the exterior
    (the designed cells of `domain`, the components, the air in the
    keep-outs), with the mechanical design's densities mapped in."""

    def __init__(self, study, domain: Domain, spec: CoolingSpec | None = None, floor=None):
        from .domain import _fraction
        from .thermal import Faces, ScalarSolver, thermal_bc
        self.spec = spec = spec or cooling_from_study(study)
        self.domain = domain
        pitch, dims, origin = domain.pitch, tuple(domain.dims), np.asarray(domain.origin, float)
        ii, jj, kk = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
        ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
        centers = origin + (ijk + 0.5) * pitch
        inside = _fraction(study.design_space, centers, pitch) >= 0.5
        flat = np.ravel_multi_index(tuple(domain.mesh.elem_ijk.T), dims)
        inside[flat] = True
        for s in spec.sources:
            inside |= s.shape.sdf(centers) <= 0
        mesh = HexMesh(ijk[inside], pitch, origin)
        self.mesh = mesh
        lut = np.full(int(np.prod(dims)), -1, np.int64)
        lut[np.flatnonzero(inside)] = np.arange(mesh.n_elem)
        self.mech = lut[flat]                           # thermal cell of each mechanical cell
        is_mech = np.zeros(mesh.n_elem, bool)
        is_mech[self.mech] = True
        self.is_mech = is_mech
        c = mesh.elem_centers()
        k0 = np.full(mesh.n_elem, spec.k_air)
        src = np.zeros(mesh.n_elem, bool)
        heat = []
        for s in spec.sources:
            sd = s.shape.sdf(c)
            cells = np.flatnonzero((sd <= 0) & ~is_mech)
            if not len(cells):
                # thinner than a cell: the cells it passes through (a chip
                # 1.2 mm thick on a 2 mm grid), none of them designed
                cells = np.flatnonzero((sd <= 0.5 * pitch) & ~is_mech)
            if not len(cells):
                raise ValueError(f"cooling component {s.id!r} covers no cell at {pitch:g} mm: refine the grid "
                                 "or give it its own shape")
            s.cells = cells
            k0[cells] = s.k
            src[cells] = True
            nodes = np.zeros(mesh.n_nodes, bool)
            nodes[mesh.enode[cells].ravel()] = True
            s.nodes = nodes
            if s.watts > 0:
                m = np.zeros(mesh.n_elem, bool)
                m[cells] = True
                heat.append((m, s.watts, "volume"))
        self.k0 = k0
        self.src = src
        faces = Faces(mesh)
        self.faces = faces
        self.bc = thermal_bc(mesh, ambient=spec.ambient, heat=heat, faces=faces)
        self.heat = self.bc.heat
        self._kt = hex8_kt() * pitch
        # the outside's convection, h (T - ambient) on every outside face: a
        # constant matrix (the outside is the exterior, whatever the design)
        bf = faces.nodes[faces.boundary]
        w = (spec.h_out * 1e-6 * faces.area) * _FACE_M.ravel()
        self._kc = sp.csr_matrix((np.tile(w, len(bf)), (np.repeat(bf, 4, axis=1).ravel(),
                                                       np.tile(bf, (1, 4)).ravel())),
                                 shape=(mesh.n_nodes, mesh.n_nodes))
        self.solver = ScalarSolver(rtol=1e-9)
        self._ones = np.ones(mesh.n_elem)
        # air share of each cell (1 - x on designed cells, set per design)
        self.air0 = np.where(src | is_mech, 0.0, 1.0)
        # a thin skin's share of its cells (switch._thin_skin): it conducts
        # along itself as that share of a cell, not as its cube
        if floor is None and domain.skin is not None:
            floor = np.zeros(domain.n_elem)
            floor[domain.skin.elem] = domain.skin.phi
        self.floor = np.zeros(domain.n_elem) if floor is None else np.clip(np.asarray(floor, float), 0.0, 0.999)
        # touched surface: the outside faces of the exterior's own cells (not
        # the air or a component showing through an opening)
        bnd = np.flatnonzero(faces.boundary)
        sel = bnd[is_mech[faces.a[bnd]]]
        if spec.skin_shape is not None:
            sel = sel[spec.skin_shape.sdf(faces.center[sel]) <= 0.5 * pitch]
        self.skin_nodes = np.zeros(mesh.n_nodes, bool)
        self.skin_nodes[faces.nodes[sel].ravel()] = True
        if spec.skin_max is not None and not self.skin_nodes.any():
            raise ValueError("cooling.skin: no outside skin faces found (is there a `skin` on the study?)")
        # vents: their outside faces, area and height
        self.vent_ah = []
        for v in spec.vents:
            sel_v = bnd[v.shape.sdf(faces.center[bnd]) <= 0.5 * pitch]
            v.faces = sel_v
            area = v.area if v.area is not None else len(sel_v) * pitch ** 2
            if not len(sel_v) and v.area is None:
                raise ValueError(f"cooling vent {v.id!r} touches no outside face of the grid")
            if len(sel_v):
                pos = faces.center[sel_v].mean(axis=0)
            else:
                lo, hi = v.shape.bounds()
                pos = 0.5 * (np.asarray(lo) + np.asarray(hi))
            self.vent_ah.append((float(area), float(np.asarray(pos) @ spec.up)))
        self.g_vent = 0.0
        self.vent_flow = 0.0
        self._theta = None

    # ---- fields
    def conductivity(self, xp):
        """W/(m K) per thermal cell for mechanical densities xp."""
        k = self.k0.copy()
        k[self.mech] = self.spec.k_air + self._share(xp)[0] * (self.spec.k_material - self.spec.k_air)
        return k

    def _share(self, xp):
        """(conducting share of each mechanical cell, its derivative): x^3,
        or on a thin-skin cell the skin's share phi plus the cube of what
        the design adds over it."""
        x = np.clip(np.asarray(xp, float), 0.0, 1.0)
        phi = self.floor
        y = np.clip((x - phi) / (1.0 - phi), 0.0, 1.0)
        s = phi + (1.0 - phi) * y ** 3
        ds = np.where(x > phi, 3.0 * y ** 2, 0.0)
        return s, ds

    def air(self, xp):
        a = self.air0.copy()
        a[self.mech] = 1.0 - np.clip(np.asarray(xp, float), 0.0, 1.0)
        a[self.mech[self.domain.preserve]] = 0.0
        return a

    def _vent_sink(self, air):
        """W/K per node from the stack flow, spread over the air by volume."""
        v_air = float(air.sum())
        if self.g_vent <= 0 or v_air <= 0:
            return None, 0.0
        per = self.g_vent * air / v_air
        return np.bincount(self.mesh.enode.ravel(), weights=np.repeat(per / 8.0, 8),
                           minlength=self.mesh.n_nodes), v_air

    def solve(self, xp, vent_rounds: int = 1):
        """Node temperature rise over ambient for mechanical densities xp.
        With vents, the stack flow is taken from the last solve's air
        temperature (vent_rounds re-solves it to a fixed point). The system
        is factored once (PARDISO when installed) and the factor kept for
        the adjoints (`adjoint`)."""
        k = self.conductivity(xp) * 1e-3
        air = self.air(xp)
        for r in range(max(1, vent_rounds if self.vent_ah else 1)):
            mat, f = self.assemble(k), self.heat
            sink, _ = self._vent_sink(air)
            if sink is not None:
                mat = mat + sp.diags(sink)
            mat = mat.tocsr()
            self._factorize(mat)
            theta = self.adjoint(f) if self._direct is not None else self.solver.solve(mat, f, x0=self._theta)
            self._theta = theta
            self._update_vents(theta, air)
        self._mat = mat
        return theta

    _direct = None
    _pardiso = None
    CHUNK = 200_000

    def assemble(self, k):
        """Conduction matrix for per-cell k (W/(mm K)) plus the outside's
        convection, built a chunk of cells at a time: thermal.Conduction's
        one-shot pattern took 5 GB at 0.9 M cells and could not read a
        handheld's delivered mesh at 0.33 mm."""
        n = self.mesh.n_nodes
        en = self.mesh.enode
        kt = self._kt.ravel()
        mat = self._kc.copy()
        for s in range(0, len(en), self.CHUNK):
            e = en[s:s + self.CHUNK].astype(np.int32)
            mat = mat + sp.csr_matrix(((k[s:s + self.CHUNK, None] * kt[None]).ravel(),
                                       (np.repeat(e, 8, axis=1).ravel(), np.tile(e, (1, 8)).ravel())),
                                      shape=(n, n))
        mat.sum_duplicates()
        mat.sort_indices()
        return mat

    def _factorize(self, mat):
        self.release()
        if not _HAS_PARDISO or mat.shape[0] > DIRECT_MAX:
            return
        from .bucklingopt import _factor
        self._direct, self._pardiso = _factor(mat)

    def adjoint(self, rhs):
        """Solve with the last system (symmetric: the adjoint is the same)."""
        if self._direct is not None:
            return np.asarray(self._direct(np.asarray(rhs, float)), float).reshape(np.shape(rhs))
        return self.solver.solve(self._mat, rhs)

    def release(self):
        if self._pardiso is not None:
            self._pardiso()       # bucklingopt._factor's release: MKL holds the factor outside Python
        self._direct, self._pardiso = None, None

    def air_rise(self, theta, air):
        te = theta[self.mesh.enode].mean(axis=1)
        return float((te * air).sum() / max(air.sum(), 1e-12))

    def _update_vents(self, theta, air):
        if not self.vent_ah:
            return
        dt = self.air_rise(theta, air)
        self.g_vent, self.vent_flow = stack_conductance(self.vent_ah, dt, self.spec.ambient)

    # ---- gradients
    def gradient(self, xp, theta, lam):
        """d(J)/d xp for an objective J with adjoint lam (K lam = dJ/dtheta):
        - lam . dK/dxp theta (the vent conductance held)."""
        en = self.mesh.enode[self.mech]
        e = np.einsum("ij,jk,ik->i", lam[en], self._kt, theta[en])
        g = -self._share(xp)[1] * (self.spec.k_material - self.spec.k_air) * 1e-3 * e
        if self.g_vent > 0:
            air = self.air(xp)
            v_air = float(air.sum())
            en = self.mesh.enode[self.mech]
            lt = (lam[en] * theta[en]).sum(axis=1) / 8.0
            # d air / d x = -1: K loses g/v_air per node-eighth
            g += self.g_vent / v_air * lt
            g[self.domain.preserve] = 0.0
        return g


class _Limit:
    def __init__(self, name, nodes, limit_rise):
        self.name, self.nodes, self.limit = name, np.flatnonzero(nodes), float(limit_rise)
        self.scale = 1.0

    def value(self, theta, p=P_NORM):
        t = np.maximum(theta[self.nodes], 1e-12)
        tmax = float(t.max())
        tn = t / tmax
        pn = tmax * float(np.mean(tn ** p)) ** (1.0 / p)
        d = np.zeros(len(theta))
        d[self.nodes] = (tn ** (p - 1)) * float(np.mean(tn ** p)) ** (1.0 / p - 1.0) / len(t)
        return pn, d, tmax


class CoolingProblem:
    """The cooling limits on one grid, as constraints for
    switch.optimize_switch (see `providers` there)."""

    def __init__(self, study, domain: Domain, nu=None, modulus=None, floor=None):
        self.model = CoolingModel(study, domain, floor=floor)
        spec = self.model.spec
        self.limits = []
        for s in spec.sources:
            if s.max_t is not None:
                self.limits.append(_Limit(f"cool:{s.id}", s.nodes, s.max_t - spec.ambient))
        if spec.skin_max is not None:
            self.limits.append(_Limit("cool:skin", self.model.skin_nodes, spec.skin_max - spec.ambient))
        self.m = len(self.limits)

    def evaluate(self, xp, ctx=None):
        mdl = self.model
        theta = mdl.solve(xp)
        out = []
        for lim in self.limits:
            pn, d, tmax = lim.value(theta)
            # adaptive scale: the p-norm times the last true max / p-norm
            lim.scale = tmax / max(pn, 1e-12)
            lam = mdl.adjoint(d)
            g = mdl.gradient(xp, theta, lam)
            out.append((lim.scale * pn / lim.limit - 1.0, lim.scale * g / lim.limit, lim.name))
        mdl.release()
        return out


# ------------------------------------------------------------- readings


def cooling_readings(study, domain: Domain, xp, floor=None, vent_rounds: int = 8) -> dict:
    """Temperatures of a design: each component's hottest point and its
    limit, the touched skin's hottest and mean, where the heat leaves (the
    skin by convection, vents by stack flow) and the air's mean rise."""
    mdl = CoolingModel(study, domain, floor=floor)
    spec = mdl.spec
    theta = mdl.solve(xp, vent_rounds=vent_rounds)
    t = theta + spec.ambient
    rows = []
    for s in spec.sources:
        tm = float(t[s.nodes].max())
        rows.append({"component": s.id, "watts": s.watts, "maxTemp": s.max_t, "temperature": round(tm, 2),
                     "mean": round(float(t[s.nodes].mean()), 2),
                     "ok": bool(s.max_t is None or tm <= s.max_t + 0.25)})
    f = mdl.faces
    bnd = np.flatnonzero(f.boundary)
    th_face = theta[f.nodes[bnd]].mean(axis=1)
    q_skin = float(np.sum(spec.h_out * 1e-6 * f.area * th_face))
    air = mdl.air(xp)
    rep = {"components": rows, "ambient": spec.ambient, "watts": float(sum(s.watts for s in spec.sources)),
           "heatOut": {"skin": round(q_skin, 4)}, "airRise": round(mdl.air_rise(theta, air), 2),
           "cells": int(mdl.mesh.n_elem), "pitch": float(domain.pitch)}
    if mdl.vent_ah:
        rep["heatOut"]["vents"] = round(mdl.g_vent * mdl.air_rise(theta, air), 4)
        rep["ventFlow"] = {"litresPerMinute": round(mdl.vent_flow * 6e4, 3),
                           "conductance": round(mdl.g_vent, 5),
                           "vents": [{"id": v.id, "area": a, "height": h} for v, (a, h) in zip(spec.vents, mdl.vent_ah)]}
    if mdl.skin_nodes.any():
        ts = t[mdl.skin_nodes]
        hot = int(np.flatnonzero(mdl.skin_nodes)[int(np.argmax(ts))])
        rep["skin"] = {"maxTemp": spec.skin_max, "temperature": round(float(ts.max()), 2),
                       "mean": round(float(ts.mean()), 2), "hotSpot": mdl.mesh.node_xyz[hot].round(2).tolist(),
                       "ok": bool(spec.skin_max is None or ts.max() <= spec.skin_max + 0.25)}
    rep["ok"] = all(r["ok"] for r in rows) and rep.get("skin", {}).get("ok", True)
    mdl.release()
    rep["_theta"] = t
    rep["_model"] = mdl
    return rep


def _public(rep):
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def delivered_readings(study, mesh, pitches) -> list[dict]:
    """The delivered mesh's temperatures on finer grids: each cell solid
    where at least half of it is inside the mesh (the skin and preserves
    as the study keeps them)."""
    from .domain import _fraction, build_domain
    from .shapes import MeshShape
    from .switch import _thin_skin
    out = []
    ms = MeshShape(mesh)
    for p in pitches:
        dom, fl = _thin_skin(study, build_domain(study, p))
        drawn = _fraction([ms], dom.mesh.elem_centers(), dom.pitch)
        x = np.where(drawn >= 0.5, 1.0, 0.0)
        x[dom.preserve] = 1.0
        # a thin skin the grid cannot resolve reads at its own share
        x = np.maximum(x, fl)
        out.append(_public(cooling_readings(study, dom, x, floor=fl)))
    return out


# ----------------------------------------------------------------- the run


def run_cooling(study, out_dir: str, pitches=None, iterations=None, on_progress=None,
                verify_pitches=None) -> dict:
    """Optimize, mesh and check a study's `cooling` block (with its
    `switch` and `feel` blocks too, if it has them). Writes <name>.stl
    and returns the report: each limit read on the design, on the
    black-and-white design and on the delivered mesh at two finer grids."""
    from .switch import run_switch

    cb = study.raw.get("cooling") or {}
    has_feel = bool(study.raw.get("feel"))
    cooling_from_study(study)       # errors before any solve

    def make(st, domain, nu, modulus):
        return CoolingProblem(st, domain, nu, modulus)

    providers = [make]
    feel_readback = None
    mesh_opts = None
    opt_opts = {}
    if has_feel:
        from .feel import FeelProblem
        providers.append(lambda st, domain, nu, modulus: FeelProblem(st, domain, nu, modulus))
        fb = study.raw.get("feel") or {}
        from .reconstruct import ReconstructOptions
        mesh_opts = ReconstructOptions(axial_debias=bool(fb.get("keepThinMembers", True)))
        opt_opts["move_sharp"] = float(fb.get("moveSharp", 0.03))

    def readback(domain, design, solid, delivered, floor, mesh):
        out = {}
        if on_progress is not None:
            on_progress({"stage": "verify", "check": "temperatures of the design"})
        rd = cooling_readings(study, domain, design, floor=floor)
        out["cooling"] = {"design": _public(rd),
                          "thresholded": _public(cooling_readings(study, domain, solid, floor=floor))}
        if mesh is not None:
            vps = verify_pitches or cb.get("verifyPitches") or [0.7 * domain.pitch, 0.5 * domain.pitch]
            if on_progress is not None:
                on_progress({"stage": "verify", "check": f"temperatures of the delivered mesh at {vps} mm"})
            dl = delivered_readings(study, mesh, vps)
            out["cooling"]["delivered"] = dl
            last = dl[-1]
            out["cooling"]["deliveredOk"] = bool(last["ok"])
            if len(dl) > 1:
                out["cooling"]["gridChange"] = _grid_change(dl[-2], dl[-1])
        out["cooling"]["ok"] = bool(rd["ok"])
        if has_feel:
            from .feel import FeelProblem, feel_readings
            nu = study.elastic(design=True)
            prob = FeelProblem(study, domain, nu, float(study.material.E))
            rows = feel_readings(prob, design)
            for row, rs in zip(rows, feel_readings(prob, solid)):
                row["thresholded"] = rs["directions"]
            out["feel"] = rows
            out["feelOk"] = all(r["ok"] for r in rows)
        return out

    if not study.raw.get("switch"):
        pitches = pitches or cb.get("pitches")
        iterations = iterations or cb.get("iterations")
    rep = run_switch(study, out_dir, pitches=pitches, iterations=iterations, on_progress=on_progress,
                     providers=providers, readback=readback, mesh_options=mesh_opts, optimize_options=opt_opts)
    c = rep.get("cooling") or {}
    rep["passes"] = bool(c.get("ok") and c.get("deliveredOk", True) and rep.get("feelOk", True)
                         and (rep["passes"] if rep.get("presses") else True))
    return rep


def _grid_change(a, b) -> dict:
    """Largest change in any reported temperature rise between two grids."""
    amb = a["ambient"]
    out = {}
    for ra, rb in zip(a["components"], b["components"]):
        out[ra["component"]] = round(abs(rb["temperature"] - ra["temperature"]) / max(rb["temperature"] - amb, 1e-9), 4)
    if "skin" in a and "skin" in b:
        out["skin"] = round(abs(b["skin"]["temperature"] - a["skin"]["temperature"])
                            / max(b["skin"]["temperature"] - amb, 1e-9), 4)
    return out
