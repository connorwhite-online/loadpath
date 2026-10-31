"""
Steady heat conduction on the voxel grid, and topology optimization with it:
conductors and heat sinks on their own, and parts that have to carry load
and get rid of heat at once (a NEMA 17 bracket next to a 60 C motor).

The analysis is the scalar twin of fem.py: 8-node hexahedra, all the same
cube, one element conduction matrix scaled per cell by its conductivity.
Boundary conditions are the three a printed part sees:

- fixed temperature (Dirichlet) on nodes: a sink bolted to a frame, a motor
  face held at its case temperature;
- heat input in W, spread over a region's nodes or cells: a die, a coil, a
  contact patch;
- convection h (T - T_fluid) on cell faces, with a per-face coefficient and
  fluid temperature. The faces that convect are the part's own surface, so
  under optimization they move with the design: every face between two
  cells convects in proportion to how different their densities are, and a
  face on the grid's outside in proportion to its cell's density. A crisp
  solid part therefore convects exactly on its skin, and the optimizer
  sees what adding a fin buys (Yoon & Kim's smeared-interface idea, with a
  smooth |.| so the gradient exists).

Temperatures are solved relative to `ambient`, so ambient-temperature sinks
and fluids add nothing to the right-hand side.

Optimization is SIMP on conductivity, k(x) = kmin + x^3 (1 - kmin), with the
density filter, Heaviside continuation and robust erosion of simp.py, and
MMA (mma.py) because with convection the sensitivities change sign: a fin
in the wrong place hides surface that was convecting. Objectives are the
mean temperature rise of a region (default: power-weighted over the heat
inputs, i.e. thermal compliance divided by the power), so the adjoint is
one extra solve, or none when it equals the state.

Thermo-mechanical coupling (`optimize_thermomechanical`) is one-way:
temperature softens the polymer. Each cell's modulus is derated by the
material's retention curve at its temperature (PETG keeps about 60% of its
stiffness at 65 C), the design's own temperature field decides where, and
the compliance gradient carries the thermal adjoint, so the optimizer knows
that material near the motor is both hot and soft, and that material which
conducts heat into the structure softens it. A weighted temperature term
can be added to the objective as well.

Thermal expansion (`alpha=` on optimize_thermomechanical, and
`thermoelastic` to rate a design): each cell wants to grow by
alpha (T - t_ref), fem.thermal_load turns what stops it into nodal forces,
and the stress comes off the elastic part of the strain. The objective is
then twice the elastic strain energy, which is the compliance without
expansion: f.u would count the work of free growth and pay the optimizer
to cut whatever hot material moves most. The load depends on density and
on the design's temperature, so the gradient takes a mechanical adjoint
besides the thermal one. Checked against a bar held between rollers
(sigma = -E alpha dT exactly) and Timoshenko's bimetal strip (within 2.5%
at 4 cells per layer); bench/thermal_expansion.py runs the motor mount.

Units: mm, W, K (or C) throughout; conductivity is given in W/(m K) and
convection in W/(m^2 K) at the edges, as datasheets quote them, and
converted here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from .fem import CORNERS, HexMesh
from .material_library import LIBRARY as _LIBRARY
from .mma import MMA

__all__ = [
    "CONDUCTIVITY", "MODULUS_RETENTION", "conductivity", "retention", "hex8_kt", "ThermalGrid",
    "thermal_grid", "grid_from_domain", "Faces", "Conduction", "ScalarSolver", "ThermalBC",
    "Convection", "nodes_in", "cells_in", "thermal_bc", "solve_temperature", "ThermalResult",
    "ThermalOptions", "ThermalOptResult", "optimize_thermal", "optimize_thermomechanical",
    "EXPANSION", "expansion", "thermoelastic",
]

# W/(m K), dense printed parts, in plane, from the material library's cards
# (material_library.py cites each: Tychanicz-Kwiecien et al. 2025 for PLA
# and PETG, the AddUp sheets for AlSi10Mg and 316L, estimates otherwise).
# FDM conducts 10-30% worse across layers.
CONDUCTIVITY = {k: float(c["values"]["conductivity"][0]) for k, c in _LIBRARY.items()
                if "conductivity" in c["values"]}

# Fraction of room-temperature modulus kept at temperature (C), piecewise
# linear. Read off typical DMA storage-modulus curves and heat deflection
# temperatures; approximate, and the cliff at the glass transition is where
# they disagree most. Above the last point the last value holds.
MODULUS_RETENTION = {
    "pla": [(23, 1.0), (45, 0.92), (55, 0.6), (60, 0.2), (65, 0.04)],
    "petg": [(23, 1.0), (40, 0.93), (55, 0.8), (65, 0.6), (75, 0.25), (85, 0.05)],
    "abs": [(23, 1.0), (60, 0.85), (80, 0.65), (95, 0.25), (105, 0.04)],
    "asa": [(23, 1.0), (60, 0.85), (80, 0.65), (95, 0.25), (105, 0.04)],
    "tpu": [(23, 1.0), (40, 0.85), (60, 0.7), (80, 0.55), (100, 0.4)],
    "pa-cf": [(23, 1.0), (50, 0.8), (80, 0.55), (120, 0.35), (150, 0.2)],
    "pa12": [(23, 1.0), (45, 0.75), (80, 0.45), (120, 0.25), (160, 0.1)],
    "pa11": [(23, 1.0), (45, 0.75), (80, 0.45), (120, 0.25), (160, 0.1)],
    "resin": [(23, 1.0), (40, 0.8), (50, 0.5), (60, 0.2), (70, 0.05)],
    "resin-tough": [(23, 1.0), (40, 0.75), (50, 0.45), (60, 0.2), (70, 0.05)],
    "alsi10mg": [(23, 1.0), (150, 0.92), (200, 0.85), (300, 0.6)],
    "ti64": [(23, 1.0), (300, 0.85), (500, 0.75)],
    "316l": [(23, 1.0), (300, 0.9), (500, 0.82)],
}

# Linear thermal expansion, 1/K, near room temperature (datasheets; FDM
# parts in plane, PA-CF along the fibres it is laid with). Printed polymers
# grow 5-10x more than aluminium, which is what loads a plastic part bolted
# to a metal frame when the motor beside it warms up.
EXPANSION = {
    "pla": 68e-6, "petg": 68e-6, "abs": 90e-6, "asa": 95e-6, "tpu": 150e-6, "pa-cf": 30e-6, "pa12": 109e-6,
    "pa11": 100e-6, "resin": 80e-6, "resin-tough": 90e-6, "alsi10mg": 21e-6, "ti64": 8.6e-6, "316l": 16e-6,
}

K_MIN = 1e-3          # void conductivity, relative to the material's
PENALTY = 3.0
_SMOOTH = 0.05        # smoothing of |rho_i - rho_j| on interface faces


def _material_key(material) -> str:
    name = getattr(material, "name", material)
    from .materials import _ALIASES
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def conductivity(material) -> float:
    """W/(m K) for a material name (or Material), from CONDUCTIVITY."""
    key = _material_key(material)
    if key not in CONDUCTIVITY:
        raise ValueError(f"no conductivity for {material!r}; known: {sorted(CONDUCTIVITY)} (or pass k)")
    return CONDUCTIVITY[key]


def expansion(material) -> float:
    """1/K for a material name (or Material), from EXPANSION."""
    key = _material_key(material)
    if key not in EXPANSION:
        raise ValueError(f"no expansion coefficient for {material!r}; known: {sorted(EXPANSION)} (or pass a number)")
    return EXPANSION[key]


def retention(material, t: np.ndarray):
    """(modulus retention, its derivative per K) at temperatures t (C)."""
    pts = MODULUS_RETENTION[_material_key(material)] if not isinstance(material, (list, tuple)) else material
    tt = np.array([p[0] for p in pts], float)
    gg = np.array([p[1] for p in pts], float)
    t = np.asarray(t, float)
    g = np.interp(t, tt, gg)
    seg = np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2)
    slope = (gg[seg + 1] - gg[seg]) / (tt[seg + 1] - tt[seg])
    dg = np.where((t > tt[0]) & (t < tt[-1]), slope, 0.0)
    return g, dg


def hex8_kt() -> np.ndarray:
    """Conduction matrix (8x8) of a unit cube with k = 1, 2x2x2 Gauss. For side
    h and conductivity k the element matrix is k * h * this."""
    s = 2.0 * CORNERS - 1.0
    g = 1.0 / np.sqrt(3.0)
    kt = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                dn = np.empty((8, 3))
                dn[:, 0] = 0.25 * s[:, 0] * (1 + eta * s[:, 1]) * (1 + zeta * s[:, 2])
                dn[:, 1] = 0.25 * s[:, 1] * (1 + xi * s[:, 0]) * (1 + zeta * s[:, 2])
                dn[:, 2] = 0.25 * s[:, 2] * (1 + xi * s[:, 0]) * (1 + eta * s[:, 1])
                kt += dn @ dn.T * 0.125
    return 0.5 * (kt + kt.T)


# Consistent face matrix of a bilinear square of area 1, corners in cyclic
# order: int N_i N_j dA.
_FACE_M = np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], float) / 36.0


def _face_corners(axis: int, side: int) -> list:
    b, c = [a for a in range(3) if a != axis]
    out = []
    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
        out.append((side << axis) | (u << b) | (v << c))
    return out


# ---------------------------------------------------------------- the grid


@dataclass
class ThermalGrid:
    """What the thermal solver and optimizer need of a voxel domain. Duck-
    types simp.Domain for simp.density_filter (mesh, pitch, dims)."""
    pitch: float
    origin: np.ndarray
    dims: tuple
    mesh: HexMesh
    preserve: np.ndarray            # bool per element
    mirror: dict = field(default_factory=dict)

    @property
    def n_elem(self):
        return self.mesh.n_elem

    def full_grid(self, values, fill=0.0):
        g = np.full(self.dims, fill, float)
        g[tuple(self.mesh.elem_ijk.T)] = values
        return g


def thermal_grid(design_space, pitch: float, preserve=(), keep_out=(), symmetry=()) -> ThermalGrid:
    """Voxelize shapes (lists of shapes.Shape) the way domain.build_domain
    does, without the supports a mechanical domain needs."""
    from .domain import _fraction

    design_space = list(design_space) if isinstance(design_space, (list, tuple)) else [design_space]
    lo = np.min([s.bounds()[0] for s in design_space], axis=0)
    hi = np.max([s.bounds()[1] for s in design_space], axis=0)
    dims = np.maximum(np.ceil((hi - lo) / pitch - 1e-9).astype(int), 1)
    origin = 0.5 * (lo + hi) - 0.5 * dims * pitch
    ii, jj, kk = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    centers = origin + (ijk + 0.5) * pitch
    design_f = _fraction(design_space, centers, pitch)
    pres_f = _fraction(list(preserve), centers, pitch)
    keep_f = _fraction(list(keep_out), centers, pitch)
    pres = (pres_f >= 0.5) & (keep_f < 0.5)
    active = pres | ((design_f >= 0.5) & (keep_f == 0))
    if not active.any():
        raise ValueError("the design space is empty after removing keep-outs")
    mesh = HexMesh(ijk[active], pitch, origin)
    mirror = {}
    for axis in symmetry:
        a = "xyz".index(axis)
        m_ijk = mesh.elem_ijk.copy()
        m_ijk[:, a] = dims[a] - 1 - m_ijk[:, a]
        lut = np.full(dims, -1, np.int64)
        lut[tuple(mesh.elem_ijk.T)] = np.arange(mesh.n_elem)
        mirror[axis] = lut[tuple(m_ijk.T)]
    return ThermalGrid(float(pitch), origin, tuple(int(d) for d in dims), mesh, pres[active], mirror)


def grid_from_domain(domain) -> ThermalGrid:
    """The thermal view of a mechanical simp.Domain: same cells, same nodes."""
    return ThermalGrid(domain.pitch, domain.origin, domain.dims, domain.mesh, domain.preserve.copy(),
                       dict(domain.mirror))


def nodes_in(mesh: HexMesh, shape, tol: float | None = None) -> np.ndarray:
    """Bool per node: inside `shape` or within `tol` (default half a cell)."""
    tol = 0.5 * mesh.pitch if tol is None else tol
    return shape.sdf(mesh.node_xyz) <= tol


def cells_in(mesh: HexMesh, shape) -> np.ndarray:
    return shape.sdf(mesh.elem_centers()) <= 0


# ------------------------------------------------------------- the faces


class Faces:
    """Every cell face that can convect: faces between two active cells
    (`b` >= 0) and faces on the outside of the active set (`b` == -1).

    a, b: the cells on either side (normal points from a to b); nodes: (nf, 4)
    in cyclic order; center, normal: (nf, 3)."""

    def __init__(self, mesh: HexMesh):
        ijk = mesh.elem_ijk
        dims = mesh.node_dims - 1
        lut = np.full(tuple(dims), -1, np.int64)
        lut[tuple(ijk.T)] = np.arange(mesh.n_elem)
        a_l, b_l, n_l, c_l, nrm_l = [], [], [], [], []
        for axis in range(3):
            for side in (0, 1):
                nb = ijk.copy()
                nb[:, axis] += 2 * side - 1
                ok = (nb[:, axis] >= 0) & (nb[:, axis] < dims[axis])
                j = np.full(mesh.n_elem, -1, np.int64)
                j[ok] = lut[tuple(nb[ok].T)]
                # internal faces once (from the lower cell), boundary faces always
                take = (j < 0) | ((j >= 0) & (side == 1))
                ii = np.nonzero(take)[0]
                corners = _face_corners(axis, side)
                a_l.append(ii)
                b_l.append(j[ii])
                n_l.append(mesh.enode[ii][:, corners])
                cen = mesh.origin + (ijk[ii] + 0.5) * mesh.pitch
                cen[:, axis] += (side - 0.5) * mesh.pitch
                c_l.append(cen)
                nrm = np.zeros((len(ii), 3))
                nrm[:, axis] = 2 * side - 1
                nrm_l.append(nrm)
        self.a = np.concatenate(a_l)
        self.b = np.concatenate(b_l)
        self.nodes = np.concatenate(n_l)
        self.center = np.concatenate(c_l)
        self.normal = np.concatenate(nrm_l)
        self.area = mesh.pitch ** 2
        self.boundary = self.b < 0

    def __len__(self):
        return len(self.a)

    def exposure(self, rho: np.ndarray):
        """Per face: how much of it is solid-against-void (0..1) at densities
        rho, and its derivatives with respect to rho_a and rho_b."""
        ra = rho[self.a]
        rb = np.where(self.boundary, 0.0, rho[np.maximum(self.b, 0)])
        d = ra - rb
        e = _SMOOTH
        norm = np.sqrt(1 + e * e) - e
        r = np.sqrt(d * d + e * e)
        s = (r - e) / norm
        ds = d / r / norm
        s = np.where(self.boundary, ra, s)
        ds_a = np.where(self.boundary, 1.0, ds)
        ds_b = np.where(self.boundary, 0.0, -ds)
        return s, ds_a, ds_b


@dataclass
class Convection:
    """Convection to a fluid at `fluid_t` with coefficient `h` W/(m^2 K) on the
    faces `where` selects: None (all), a Shape (face centers within half a cell
    of it), or a callable (centers, normals) -> bool. `boundary` / `internal`
    restrict it to the grid's outside or to faces between cells (the surface
    the design grows)."""
    h: float
    fluid_t: float | None = None     # None: ambient
    where: object = None
    boundary: bool = True
    internal: bool = True


@dataclass
class ThermalBC:
    ambient: float                     # C; temperatures are solved relative to it
    fixed: np.ndarray                  # bool per node
    fixed_t: np.ndarray                # C per node (used where fixed)
    heat: np.ndarray                   # W per node
    face_h: np.ndarray                 # W/(mm^2 K) per face (0: none)
    face_t: np.ndarray                 # fluid C per face
    faces: Faces
    heat_nodes: np.ndarray             # bool per node: where heat goes in

    @property
    def power(self):
        return float(self.heat.sum())


def thermal_bc(mesh: HexMesh, *, ambient: float = 25.0, temperatures=(), heat=(), convection=(),
               faces: Faces | None = None) -> ThermalBC:
    """Assemble boundary conditions.

    temperatures: [(Shape or node mask, C)]; heat: [(Shape / node mask, W)] or
    [(Shape / cell mask, W, "volume")] spread over the region's cells by
    volume; convection: [Convection]."""
    faces = faces or Faces(mesh)
    nn = mesh.n_nodes
    fixed = np.zeros(nn, bool)
    fixed_t = np.full(nn, float(ambient))
    for region, t in temperatures:
        m = region if isinstance(region, np.ndarray) else nodes_in(mesh, region)
        if not m.any():
            raise ValueError("a fixed temperature touches no node of the grid")
        fixed |= m
        fixed_t[m] = float(t)
    q = np.zeros(nn)
    heat_nodes = np.zeros(nn, bool)
    for item in heat:
        region, watts = item[0], float(item[1])
        if len(item) > 2 and item[2] == "volume":
            cm = region if isinstance(region, np.ndarray) else cells_in(mesh, region)
            if not cm.any():
                raise ValueError("a volume heat source covers no cell of the grid")
            per = watts / cm.sum() / 8.0
            np.add.at(q, mesh.enode[cm].ravel(), per)
            heat_nodes[mesh.enode[cm].ravel()] = True
        else:
            m = region if isinstance(region, np.ndarray) else nodes_in(mesh, region)
            if not m.any():
                raise ValueError("a heat input touches no node of the grid")
            # weighted by the cells each node touches, so a flux over a face
            # is spread as the face's area is (corners a quarter, edges a half)
            w = np.bincount(mesh.enode.ravel(), minlength=nn).astype(float) * m
            q += watts * w / w.sum()
            heat_nodes |= m
    face_h = np.zeros(len(faces))
    face_t = np.full(len(faces), float(ambient))
    for cv in convection:
        sel = np.ones(len(faces), bool)
        if cv.where is not None:
            if callable(cv.where) and not hasattr(cv.where, "sdf"):
                sel = np.asarray(cv.where(faces.center, faces.normal), bool)
            else:
                sel = cv.where.sdf(faces.center) <= 0.5 * mesh.pitch
        if not cv.boundary:
            sel &= ~faces.boundary
        if not cv.internal:
            sel &= faces.boundary
        face_h[sel] = float(cv.h) * 1e-6
        face_t[sel] = float(ambient if cv.fluid_t is None else cv.fluid_t)
    return ThermalBC(float(ambient), fixed, fixed_t, q, face_h, face_t, faces, heat_nodes)


# ---------------------------------------------------------- the assembly


class Conduction:
    """CSR layout for a HexMesh's scalar conduction matrix. `assemble(k, rho)`
    takes per-element conductivity k (W/(mm K)) and densities rho (which set
    the convecting surface) and returns (matrix with Dirichlet rows, rhs).

    `porous`: while a design is still grey, a uniform field has no interfaces
    and so no convecting surface, and the only faces that convect are on the
    grid's outside: the optimizer then pushes everything there (a ring of pins
    round the edge of a heat sink). A grey cell is treated as a porous medium
    instead, with 4 rho (1 - rho) * `porous` faces' worth of wetted area,
    which is gone once the design is crisp. Set by the optimizer; 0 for
    analysis."""

    def __init__(self, mesh: HexMesh, bc: ThermalBC, porous: float = 0.0):
        self.mesh = mesh
        self.bc = bc
        self.kt = hex8_kt() * mesh.pitch
        nn = mesh.n_nodes
        en = mesh.enode
        rows = np.repeat(en, 8, axis=1).ravel()
        cols = np.tile(en, (1, 8)).ravel()
        keys = rows * nn + cols
        uniq, self._inv = np.unique(keys, return_inverse=True)
        self._inv = self._inv.astype(np.int64)
        self.indices = (uniq % nn).astype(np.int32)
        r = uniq // nn
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=nn))]).astype(np.int32)
        self.nnz = len(uniq)
        self._kflat = self.kt.ravel()
        # face node pairs -> slots (always within one element, so present)
        faces = bc.faces
        fn = faces.nodes
        conv = np.nonzero(bc.face_h > 0)[0]
        self.conv = conv
        frow = np.repeat(fn[conv], 4, axis=1).ravel()
        fcol = np.tile(fn[conv], (1, 4)).ravel()
        self._fslot = np.searchsorted(uniq, frow * nn + fcol)
        self._fm = _FACE_M.ravel() * faces.area
        # porous convection: per cell, the strongest internal convection of
        # its faces (lumped onto its corners' diagonal)
        inner = conv[~faces.boundary[conv]]
        self.cell_h = np.zeros(mesh.n_elem)
        self.cell_dt = np.zeros(mesh.n_elem)
        for side in (faces.a[inner], faces.b[inner]):
            take = bc.face_h[inner] > self.cell_h[side]
            self.cell_h[side[take]] = bc.face_h[inner][take]
            self.cell_dt[side[take]] = bc.face_t[inner][take] - bc.ambient
        self._dslot = np.searchsorted(uniq, en.ravel() * nn + en.ravel()).reshape(-1, 8)
        self.porous = float(porous)
        fixed = bc.fixed
        self._keep = ~(fixed[r] | fixed[self.indices])
        self._diag_fixed = np.nonzero((r == self.indices) & fixed[r])[0]
        self.theta_fixed = np.where(fixed, bc.fixed_t - bc.ambient, 0.0)

    def _porous(self, rho):
        a = self.porous * self.mesh.pitch ** 2
        return a * 4 * rho * (1 - rho), a * 4 * (1 - 2 * rho)

    def _data(self, k, rho):
        data = np.bincount(self._inv, weights=(np.asarray(k, float)[:, None] * self._kflat).ravel(),
                           minlength=self.nnz)
        if len(self.conv):
            s = self.bc.faces.exposure(rho)[0]
            w = (self.bc.face_h[self.conv] * s[self.conv])[:, None] * self._fm
            data += np.bincount(self._fslot, weights=w.ravel(), minlength=self.nnz)
        if self.porous > 0:
            sv = self._porous(rho)[0] * self.cell_h / 8.0
            data += np.bincount(self._dslot.ravel(), weights=np.repeat(sv, 8), minlength=self.nnz)
        return data

    def load(self, rho):
        """Right-hand side before Dirichlet lifting: heat in plus convection
        from fluids that are not at ambient."""
        f = self.bc.heat.copy()
        c = self.conv
        if len(c):
            s = self.bc.faces.exposure(rho)[0]
            dt = self.bc.face_t[c] - self.bc.ambient
            w = self.bc.face_h[c] * s[c] * dt * self.bc.faces.area / 4.0
            if np.any(w):
                np.add.at(f, self.bc.faces.nodes[c].ravel(), np.repeat(w, 4))
        if self.porous > 0 and np.any(self.cell_dt):
            w = self._porous(rho)[0] * self.cell_h * self.cell_dt / 8.0
            np.add.at(f, self.mesh.enode.ravel(), np.repeat(w, 8))
        return f

    def has_sink(self, rho):
        if self.bc.fixed.any():
            return True
        s = self.bc.faces.exposure(rho)[0]
        return bool(np.any(self.bc.face_h * s > 0)) or bool(self.porous > 0 and np.any(self.cell_h > 0))

    def assemble(self, k, rho):
        nn = self.mesh.n_nodes
        data = self._data(k, rho)
        f = self.load(rho)
        if self.bc.fixed.any():
            full = sp.csr_matrix((data, self.indices, self.indptr), shape=(nn, nn))
            f = f - full @ self.theta_fixed
            # Identity rows scaled like the rest of the matrix (k h is ~1e-4
            # W/K for a polymer at 1 mm): unscaled, the fixed values dominate
            # the residual norm and CG stops before the free rows converge.
            dval = float(np.mean(full.diagonal()))
            f[self.bc.fixed] = dval * self.theta_fixed[self.bc.fixed]
            data = data * self._keep
            data[self._diag_fixed] = dval
        mat = sp.csr_matrix((data, self.indices, self.indptr), shape=(nn, nn))
        mat.has_sorted_indices = True
        return mat, f

    def residual(self, k, rho, theta):
        """K theta - f without Dirichlet rows: at fixed nodes, minus the heat
        they inject."""
        nn = self.mesh.n_nodes
        full = sp.csr_matrix((self._data(k, rho), self.indices, self.indptr), shape=(nn, nn))
        return full @ theta - self.load(rho)

    def elem_energy(self, lam, theta):
        """Per element lam_e^T Kt theta_e at unit conductivity."""
        en = self.mesh.enode
        return np.einsum("ij,jk,ik->i", lam[en], self.kt, theta[en])

    def convection_gradient(self, rho, lam, theta):
        """Per element d(lam^T R)/d rho from the convection terms."""
        g = np.zeros(len(rho))
        c = self.conv
        if len(c):
            faces = self.bc.faces
            _, ds_a, ds_b = faces.exposure(rho)
            fn = faces.nodes[c]
            a = faces.area
            fe = np.einsum("ij,jk,ik->i", lam[fn], _FACE_M * a, theta[fn])
            fe -= lam[fn].sum(axis=1) * (self.bc.face_t[c] - self.bc.ambient) * a / 4.0
            fe *= self.bc.face_h[c]
            g += np.bincount(faces.a[c], weights=fe * ds_a[c], minlength=len(rho))
            inner = faces.b[c] >= 0
            g += np.bincount(faces.b[c][inner], weights=(fe * ds_b[c])[inner], minlength=len(rho))
        if self.porous > 0:
            en = self.mesh.enode
            dsv = self._porous(rho)[1]
            pe = (lam[en] * theta[en]).sum(axis=1) - lam[en].sum(axis=1) * self.cell_dt
            g += dsv * self.cell_h / 8.0 * pe
        return g


class ScalarSolver:
    """CG + smoothed-aggregation AMG for a scalar diffusion matrix, keeping
    the hierarchy across solves while it still works (as fem.Solver)."""

    def __init__(self, rtol=1e-8, maxiter=2000, rebuild_after_iters=40):
        self.rtol, self.maxiter, self.rebuild_after_iters = rtol, maxiter, rebuild_after_iters
        self._ml = None
        self.last_iters = 0

    def invalidate(self):
        self._ml = None

    def _precond(self, mat):
        if self._ml is None:
            from .fem import amg_solver
            self._ml = amg_solver(mat, symmetry="hermitian", max_coarse=400, coarse_solver="splu")
        return self._ml.aspreconditioner(cycle="V")

    def solve(self, mat, f, x0=None):
        count = [0]

        def cb(_):
            count[0] += 1

        # the system is scaled by the conductivity, which can be tiny (W/mm K)
        scale = float(np.abs(f).max()) or 1.0
        t, info = cg(mat, f / scale, x0=None if x0 is None else x0 / scale, rtol=self.rtol,
                     maxiter=self.maxiter, M=self._precond(mat), callback=cb)
        if info != 0:
            self._ml = None
            count[0] = 0
            t, info = cg(mat, f / scale, x0=t, rtol=self.rtol, maxiter=self.maxiter,
                         M=self._precond(mat), callback=cb)
        self.last_iters = count[0]
        if count[0] > self.rebuild_after_iters:
            self._ml = None
        if info != 0:
            raise RuntimeError(f"thermal solve did not converge ({count[0]} iterations); is any heat "
                               "path to a fixed temperature or a convecting face?")
        return t * scale


@dataclass
class ThermalResult:
    temperature: np.ndarray          # C per node
    elem_temperature: np.ndarray     # C per element (mean of its nodes)
    max_t: float
    heat_in: float                   # W
    heat_out_fixed: float            # W leaving through fixed temperatures (negative: entering)
    heat_out_convection: float       # W leaving by convection
    resistance: float                # K/W: mean rise over the heat inputs per W (nan without heat)

    def mean_t(self, nodes):
        return float(self.temperature[nodes].mean())


def solve_temperature(mesh: HexMesh, k, bc: ThermalBC, density=None, solver: ScalarSolver | None = None,
                      conduction: Conduction | None = None) -> ThermalResult:
    """Steady temperatures on `mesh`. k: conductivity in W/(m K), scalar or per
    element; density (per element, default solid) scales it SIMP-wise and
    sets which faces convect."""
    rho = np.ones(mesh.n_elem) if density is None else np.asarray(density, float)
    kk = np.broadcast_to(np.asarray(k, float) * 1e-3, (mesh.n_elem,))
    ke = kk * (K_MIN + rho ** PENALTY * (1 - K_MIN))
    cond = conduction or Conduction(mesh, bc)
    if not cond.has_sink(rho):
        raise ValueError("no heat path out: add a fixed temperature or convection")
    mat, f = cond.assemble(ke, rho)
    theta = (solver or ScalarSolver()).solve(mat, f)
    t = theta + bc.ambient
    # heat balance: reaction at the fixed nodes, convection integral
    out_fixed = -float(cond.residual(ke, rho, theta)[bc.fixed].sum())
    c = cond.conv
    s = bc.faces.exposure(rho)[0]
    th_face = theta[bc.faces.nodes[c]].mean(axis=1)
    out_conv = float(np.sum(bc.face_h[c] * s[c] * bc.faces.area * (th_face - (bc.face_t[c] - bc.ambient))))
    q = bc.heat
    res = float((q @ theta) / q.sum() ** 2) if q.sum() > 0 else float("nan")
    return ThermalResult(t, t[mesh.enode].mean(axis=1), float(t.max()), float(q.sum()), out_fixed, out_conv, res)


# --------------------------------------------------------- optimization


@dataclass
class ThermalOptions:
    volume_fraction: float
    filter_radius: float               # mm (simp.filter_radius gives one from a min member)
    k: float = 0.2                     # W/(m K) of the solid
    max_iterations: int = 100
    beta_max: float = 16.0
    beta_every: int = 12
    eta: float = 0.5
    # Robust erosion as simp.SimpOptions (physics on the eroded design, the
    # opening delivered). Off by default here: on a 40 mm convective sink it
    # left the die 35-75% hotter than the same run without it (36 vs 24 K,
    # 37 vs 21 K with porous grey), since the optimizer is paid for surface
    # and erosion takes surface away; the filter still sets the member size.
    # Use 0.15 for load-bearing parts.
    erode: float = 0.0
    move: float = 0.1
    tol: float = 0.01
    # objective: mean temperature rise over these nodes, or None for the
    # power-weighted rise over the heat inputs (thermal compliance / power)
    target: np.ndarray | None = None
    # pin cells touching fixed temperatures and heat inputs solid
    anchor: bool = True
    x0: np.ndarray | None = None
    # Axis index the part is extruded along (2.5D: a spreader cut from plate),
    # or None. Also turns off the filter's boundary padding, which would
    # otherwise thin a plate a few cells thick to nothing.
    extrusion: int | None = None
    # grey cells convect as a porous medium with this many faces' worth of
    # wetted area at rho = 0.5 (see Conduction); 0 turns it off
    # (on the same sink: 21.0 K with it, 23.7 K without; a hand-laid forest
    # of 2 mm pins at the same mass reads 23.0 K)
    porous: float = 1.0


@dataclass
class ThermalOptResult:
    x: np.ndarray
    xphys: np.ndarray                  # delivered design: 0/1, connected to a heat path
    objective: float                   # mean temperature rise, K (thermal), or the weighted sum
    temperature: ThermalResult         # of the delivered design
    volume_fraction: float
    iterations: int
    converged: bool
    seconds: float
    compliance: float = float("nan")   # thermo-mechanical: weighted compliance at temperature, N mm
    case_compliance: list = field(default_factory=list)
    history: list = field(default_factory=list)


class _Thermal:
    """State, objective and gradient (with respect to the eroded density) of
    the thermal problem inside an optimization loop."""

    def __init__(self, grid: ThermalGrid, bc: ThermalBC, k: float, target, porous: float = 0.0):
        self.grid, self.bc = grid, bc
        self.mesh = grid.mesh
        self.k = float(k) * 1e-3
        self.cond = Conduction(grid.mesh, bc, porous)
        self.solver = ScalarSolver(rtol=1e-7)
        if target is None:
            if bc.power <= 0:
                raise ValueError("no heat input: pass a target region to measure")
            self.l = bc.heat / bc.power
        else:
            tg = np.asarray(target, bool)
            self.l = tg / tg.sum()
        self.l = np.where(bc.fixed, 0.0, self.l)
        # self-adjoint when the objective weights are the loads and nothing
        # is lifted: then lambda = theta / power
        self.self_adjoint = target is None and not np.any(self.cond.theta_fixed) and \
            not np.any((bc.face_h > 0) & (bc.face_t != bc.ambient)) and not np.any(self.cond.cell_dt)
        self.theta = None
        self.lam = None

    def solve(self, rho):
        ke = self.k * (K_MIN + rho ** PENALTY * (1 - K_MIN))
        if not self.cond.has_sink(rho):
            raise ValueError("no heat path out: add a fixed temperature or convection")
        mat, f = self.cond.assemble(ke, rho)
        self.theta = self.solver.solve(mat, f, self.theta)
        self._mat = mat
        return self.theta

    def objective(self):
        return float(self.l @ self.theta)

    def adjoint(self, rhs, x0=None):
        rhs = np.where(self.bc.fixed, 0.0, rhs)
        return self.solver.solve(self._mat, rhs, x0)

    def gradient(self, rho, lam):
        """d(lam-weighted residual)/d rho: -lam^T dR/drho, i.e. the gradient of
        a functional whose adjoint is lam."""
        dk = self.k * PENALTY * rho ** (PENALTY - 1) * (1 - K_MIN)
        g = -dk * self.cond.elem_energy(lam, self.theta)
        return g - self.cond.convection_gradient(rho, lam, self.theta)

    def objective_gradient(self, rho):
        if self.self_adjoint:
            lam = self.theta / self.bc.power
        else:
            self.lam = self.adjoint(self.l, self.lam)
            lam = self.lam
        return self.gradient(rho, lam)


def _anchor_cells(grid, bc):
    """Cells touching a fixed temperature or a heat input, pinned solid when
    that is a small patch (a sink pad, a die): the boundary-padded filter
    would otherwise make them too expensive to fill. A distributed source
    (heat generated everywhere) is left to the optimizer."""
    out = np.zeros(grid.mesh.n_elem, bool)
    for nodes in (bc.fixed, bc.heat_nodes):
        touch = nodes[grid.mesh.enode].any(axis=1)
        if touch.sum() <= 0.03 * grid.mesh.n_elem:
            out |= touch
    return out


class _Design:
    """Filter, projection, erosion and symmetry shared by both optimizers."""

    def __init__(self, grid, opts, pres):
        from .simp import density_filter
        self.h, self.hs = density_filter(grid, opts.filter_radius, pad=opts.extrusion is None)
        self.ht = self.h.T.tocsr()
        self.opts = opts
        self.pres = pres
        self.design = ~pres
        self.mirror = grid.mirror
        self.columns = None
        if opts.extrusion is not None:
            others = [i for i in range(3) if i != opts.extrusion]
            ij = grid.mesh.elem_ijk[:, others]
            _, self.columns = np.unique(ij[:, 0] * (grid.dims[others[1]] + 1) + ij[:, 1], return_inverse=True)

    def tie(self, v):
        """Symmetry and extrusion applied to a design-variable field."""
        from .simp import _extrude, _symmetrize
        if self.mirror:
            v = _symmetrize(v, self.mirror)
        return _extrude(v, self.columns, self.design)

    def fields(self, x, beta):
        from .simp import _dproject, _project
        o = self.opts
        xt = (self.h @ x) / self.hs
        xp = _project(xt, beta, o.eta)
        xe = _project(xt, beta, o.eta + o.erode)
        dpb = _dproject(xt, beta, o.eta)
        dpe = _dproject(xt, beta, o.eta + o.erode)
        for v in (xp, xe):
            v[self.pres] = 1.0
        for v in (dpb, dpe):
            v[self.pres] = 0.0
        return xp, xe, dpb, dpe

    def back(self, g, dp):
        return self.tie(self.ht @ (g * dp / self.hs))

    def deliver(self, x, beta):
        from .simp import _project
        o = self.opts
        xt = (self.h @ x) / self.hs
        if o.erode > 0:
            xe = _project(xt, beta, o.eta + o.erode)
            xe[self.pres] = 1.0
            xp = _project((self.h @ xe) / self.hs, beta, o.eta - o.erode)
        else:
            xp = _project(xt, beta, o.eta)
        xp[self.pres] = 1.0
        return xp


def _crisp_connected(grid, xp, bc, pres):
    """The delivered part: cells above half density, minus pieces that touch
    no heat input, fixed temperature or preserved cell. Convective designs
    keep grey shells the projection never quite clears (40% of the non-void
    cells on the 40 mm sink at beta 16), and floating specks; analysed crisp
    and connected, the same sinks read 1-2 K different and lighter."""
    from scipy import ndimage
    mesh = grid.mesh
    solid = (xp > 0.5) | pres
    lbl, _ = ndimage.label(grid.full_grid(solid) > 0.5, structure=np.ones((3, 3, 3), bool))
    el = lbl[tuple(mesh.elem_ijk.T)]
    anchor = solid & ((bc.fixed | bc.heat_nodes)[mesh.enode].any(axis=1) | pres)
    keep = np.isin(el, np.unique(el[anchor]))
    return (solid & keep & (el > 0)).astype(float)


def _mma_loop(grid, opts, pres, evaluate, on_iter=None, should_stop=None):
    """Volume-constrained MMA with projection continuation. evaluate(xe) ->
    (objective, d objective / d xe, record)."""
    t0 = time.time()
    n = grid.mesh.n_elem
    design = ~pres
    free = np.flatnonzero(design)
    if len(free) == 0:
        raise ValueError("nothing to optimize: every cell is preserved")
    dz = _Design(grid, opts, pres)
    x = np.full(n, opts.volume_fraction)
    if opts.x0 is not None:
        x = np.clip(np.asarray(opts.x0, float), 0.0, 1.0)
    x[pres] = 1.0
    x = dz.tie(x)
    mma = MMA(len(free), 1, move=opts.move)
    beta, since_beta = 1.0, 0
    f_ref = None
    history = []
    converged = False
    it = 0
    for it in range(1, opts.max_iterations + 1):
        if should_stop is not None and should_stop():
            break
        xp, xe, dpb, dpe = dz.fields(x, beta)
        obj, g_xe, rec = evaluate(xe)
        if f_ref is None:
            f_ref = abs(obj) or 1.0
        f0 = 10.0 * obj / f_ref
        df0 = dz.back(10.0 * g_xe / f_ref, dpe)
        vol = float(xp[design].mean())
        con = vol / opts.volume_fraction - 1.0
        dcon = dz.back(design / design.sum() / opts.volume_fraction, dpb)
        xn = mma.update(x[free], f0, df0[free], np.array([con]), dcon[free][None, :])
        x_new = x.copy()
        x_new[free] = xn
        x_new = dz.tie(x_new)
        change = float(np.abs(x_new - x)[design].max())
        x = x_new
        since_beta += 1
        rec = {"iteration": it, "objective": obj, "volume": vol, "beta": beta, "change": change,
               "seconds": time.time() - t0, **rec}
        history.append(rec)
        if on_iter is not None:
            on_iter(rec, xp)
        if beta < opts.beta_max and (since_beta >= opts.beta_every or change < 2 * opts.tol):
            beta = min(2 * beta, opts.beta_max)
            since_beta = 0
        elif beta >= opts.beta_max and since_beta >= 5 and con < 0.02:
            if change < opts.tol or (len(history) > 6 and
                                     abs(history[-1]["objective"] - history[-7]["objective"])
                                     < 2e-3 * abs(history[-7]["objective"])):
                converged = True
                break
    return x, dz.deliver(x, beta), it, converged, history, time.time() - t0


def optimize_thermal(grid: ThermalGrid, bc: ThermalBC, opts: ThermalOptions, on_iter=None,
                     should_stop=None) -> ThermalOptResult:
    """Lay out `volume_fraction` of the design cells to minimize the mean
    temperature rise (of opts.target, or over the heat inputs)."""
    pres = grid.preserve.copy()
    if opts.anchor:
        pres |= _anchor_cells(grid, bc)
    th = _Thermal(grid, bc, opts.k, opts.target, opts.porous)

    def evaluate(xe):
        th.solve(xe)
        return th.objective(), th.objective_gradient(xe), {"cgIterations": th.solver.last_iters}

    x, xp, it, conv, hist, secs = _mma_loop(grid, opts, pres, evaluate, on_iter, should_stop)
    xp = _crisp_connected(grid, xp, bc, pres)
    res = solve_temperature(grid.mesh, opts.k, bc, density=xp)
    obj = float(th.l @ (res.temperature - bc.ambient))
    return ThermalOptResult(x=x, xphys=xp, objective=obj, temperature=res,
                            volume_fraction=float(xp[~pres].mean()), iterations=it, converged=conv,
                            seconds=secs, history=hist)


def optimize_thermomechanical(domain, nu: float, bc: ThermalBC, opts: ThermalOptions, material="petg",
                              thermal_weight: float = 0.0, on_iter=None, should_stop=None, alpha=None,
                              t_ref=None, modulus=None) -> ThermalOptResult:
    """Minimize the weighted compliance of a mechanical simp.Domain whose
    moduli are derated by temperature (`material`'s MODULUS_RETENTION, or a
    list of (C, retention) points), plus `thermal_weight` times the mean
    temperature rise (normalized by its starting value), for a volume.

    The temperature field is the design's own: solved on the same cells with
    `bc` every iteration, and the compliance gradient includes its adjoint.

    `alpha` adds thermal expansion: True for the material's EXPANSION, a
    number (1/K), or one per cell (a metal insert in a polymer part: the
    mismatch warps it). Each cell then wants to grow by alpha (T - t_ref)
    (t_ref, the stress-free temperature, defaults to bc.ambient), and
    wherever the supports, the loads' fixtures or its neighbours stop it,
    that is a load. It is applied in every load case, scales with the
    cell's (derated) stiffness, so it depends on the design, and its
    gradient (through density and through temperature) is in the
    sensitivities. `modulus` (MPa) defaults to the material's E."""
    from .simp import E_MIN

    grid = grid_from_domain(domain)
    mesh = domain.mesh
    pres = grid.preserve.copy()
    if opts.anchor:
        pres |= _anchor_cells(grid, bc)
    evaluate, fe, th = _thermomech_evaluator(domain, nu, bc, opts, material, thermal_weight, alpha=alpha,
                                             t_ref=t_ref, modulus=modulus)
    weights = np.asarray(domain.case_weights, float)

    x, xp, it, conv, hist, secs = _mma_loop(grid, opts, pres, evaluate, on_iter, should_stop)
    xp = _crisp_connected(grid, xp, bc, pres | (domain.support_nodes | domain.load_nodes)[mesh.enode].any(axis=1))
    res = solve_temperature(mesh, opts.k, bc, density=xp)
    ev = thermoelastic(domain, nu, xp, res.elem_temperature, material, alpha=alpha,
                       t_ref=bc.ambient if t_ref is None else t_ref, modulus=modulus, fe=fe)
    comp = float(np.dot(weights, ev["caseCompliance"]))
    tobj = float(th.l @ (res.temperature - bc.ambient))
    out = ThermalOptResult(x=x, xphys=xp, objective=tobj, temperature=res,
                           volume_fraction=float(xp[~pres].mean()), iterations=it, converged=conv,
                           seconds=secs, compliance=comp, case_compliance=ev["caseCompliance"], history=hist)
    out.thermoelastic = ev
    return out


def _alpha_cells(alpha, material, n):
    if alpha is None or alpha is False:
        return None
    a = expansion(material) if alpha is True else alpha
    return np.asarray(a, float) * np.ones(n)


def _modulus(material, modulus):
    if modulus is not None:
        return float(modulus)
    from .materials import material_from_spec

    return float(material_from_spec(material).E)


def thermoelastic(domain, nu, xp, elem_temperature, material="petg", alpha=True, t_ref=25.0, modulus=None,
                  fe=None, rtol=1e-6):
    """Solve a design (densities xp) at a temperature field (per cell, C):
    the load cases plus the thermal expansion load (alpha as in
    optimize_thermomechanical; None for none), moduli derated by
    temperature. Returns per-case displacements (mm), compliance (N mm),
    twice the elastic strain energy (N mm: the optimizer's
    objective with expansion, the compliance without), von Mises stress per cell (MPa, the expansion strain taken off) and the
    largest displacement."""
    from .fem import elastic_c, hex8_b_center, thermal_load, von_mises
    from .simp import E_MIN, _Analysis

    mesh = domain.mesh
    fe = fe or _Analysis(domain, nu, rtol)
    E0 = _modulus(material, modulus)
    te = np.asarray(elem_temperature, float)
    g, _ = retention(material, te)
    e = (E_MIN + xp ** 3 * (1 - E_MIN)) * g
    a = _alpha_cells(alpha, material, mesh.n_elem)
    loads = list(domain.loads)
    if a is not None:
        # in fe's units (stiffness at E = 1, so u scaled by E0): E0 E_e eps
        eps = a * (te - t_ref)
        fth = thermal_load(mesh, nu, E0 * xp ** 3 * (1 - E_MIN) * g, eps, domain.fixed)
        loads = [f + fth for f in loads] if loads else [fth]
    us = fe.solve(e, xp, [None] * len(loads), rtol=rtol, loads=loads)
    b = hex8_b_center() / mesh.pitch
    c = elastic_c(1.0, nu)
    vm = np.zeros(mesh.n_elem)
    for u in us:
        strain = u[mesh.edof] @ b.T / E0
        if a is not None:
            strain[:, :3] -= eps[:, None]
        sig = (strain @ c.T) * (E0 * e)[:, None]
        vm = np.maximum(vm, von_mises(sig))
    # twice the elastic strain energy, exactly: u.(f_m - f_th) + sum E eps^2 m^T C m V
    energy = [float(u @ f) / E0 for f, u in zip(loads, us)]
    if a is not None:
        cm = float(np.array([1, 1, 1, 0, 0, 0]) @ c @ np.array([1, 1, 1, 0, 0, 0])) * mesh.pitch ** 3
        self_e = float(np.sum(E0 * xp ** 3 * (1 - E_MIN) * g * eps * eps)) * cm
        energy = [float(u @ (f - 2 * fth)) / E0 + self_e for f, u in zip(loads, us)]
    solid = xp > 0.5
    disp = [np.linalg.norm(u.reshape(-1, 3), axis=1) / E0 for u in us]
    return {"displacement": [u / E0 for u in us], "caseCompliance": [float(f @ u) / E0 for f, u in zip(loads, us)],
            "strainEnergy2": energy, "vonMises": vm, "maxVonMises": float(vm[solid].max()) if solid.any() else 0.0,
            "maxDisplacement": float(max(d.max() for d in disp)), "thermal": a is not None}


def _thermomech_evaluator(domain, nu, bc, opts, material, thermal_weight, rtol=1e-4, alpha=None, t_ref=None,
                          modulus=None):
    """evaluate(xe) -> (objective, gradient w.r.t. xe, record) for
    optimize_thermomechanical; normalized by the first evaluation."""
    from .fem import elastic_c, hex8_fth
    from .simp import ALIVE, E_MIN, _Analysis

    mesh = domain.mesh
    fe = _Analysis(domain, nu, rtol)
    th = _Thermal(grid_from_domain(domain), bc, opts.k, opts.target, opts.porous)
    weights = np.asarray(domain.case_weights, float)
    a = _alpha_cells(alpha, material, mesh.n_elem)
    tr = bc.ambient if t_ref is None else float(t_ref)
    E0 = _modulus(material, modulus) if a is not None else 1.0
    vth = hex8_fth(nu) * mesh.pitch ** 2
    # m^T C0 m V: a free-growth strain's energy density at E = 1, times the cell
    cm = float(np.array([1, 1, 1, 0, 0, 0]) @ elastic_c(1.0, nu) @ np.array([1, 1, 1, 0, 0, 0])) * mesh.pitch ** 3
    ncase = max(1, len(domain.loads)) if a is not None else len(domain.loads)
    state = {"us": [None] * ncase}
    refs = {}

    def evaluate(xe):
        theta = th.solve(xe)
        te = (theta + bc.ambient)[mesh.enode].mean(axis=1)
        g, dg = retention(material, te)
        estiff = E_MIN + xe ** 3 * (1 - E_MIN)
        loads = list(domain.loads)
        if a is not None:
            # expansion load per cell: E0 p(x) g(T) alpha (T - t_ref) vth,
            # p = x^3 without the void floor, and none from cells the
            # solve leaves out (their share is ~x^3, nothing)
            live = xe > ALIVE
            pth = np.where(live, xe ** 3 * (1 - E_MIN), 0.0)
            eps = a * (te - tr)
            w = E0 * pth * g * eps
            fth = np.bincount(mesh.edof.ravel(), weights=(w[:, None] * vth[None, :]).ravel(), minlength=mesh.n_dof)
            fth[domain.fixed] = 0.0
            loads = [f + fth for f in loads] if loads else [fth]
        us = fe.solve(estiff * g, xe, state["us"], rtol=rtol, loads=loads)
        state["us"] = us
        cw = weights if len(weights) == len(loads) else np.ones(len(loads))
        if a is None:
            comp = 0.0
            w_e = np.zeros(mesh.n_elem)           # sum_c w_c u_e^T K0 u_e
            for ci, f in enumerate(loads):
                comp += cw[ci] * float(f @ us[ci])
                w_e += cw[ci] * fe.strain_energy(us[ci])
            refs.setdefault("c", comp)
            # explicit part, at fixed temperature
            g_xe = -(3 * xe ** 2 * (1 - E_MIN)) * g * w_e / refs["c"]
            # through temperature: dC/dT_e = -E_e g'(T_e) w_e, a node holds 1/8
            # of each of its cells' temperature
            dc_dte = -estiff * dg * w_e / refs["c"]
        else:
            # With expansion the objective is twice the elastic strain energy,
            # sum_e E_e (eps - eps_th)^T C0 (eps - eps_th) V, not f.u: free
            # growth does work on its load without stressing anything, and
            # f.u would pay the optimizer to cut the hot material that moves
            # most rather than the material that is fought over. Without
            # expansion it is the compliance. In fe's units:
            # J = u.f_m - u.f_th + E0^2 sum_e p_e g_e eps_e^2 cm V, adjoint
            # K lam = f_m - f_th.
            # (loads already carry f_th: f_m - f_th = f - 2 f_th)
            lams = fe.resolve([f - 2 * fth for f in loads], state.get("lams"))
            state["lams"] = lams
            comp = 0.0
            m_e = np.zeros(mesh.n_elem)           # sum_c w_c lam_e^T K0 u_e
            q_e = np.zeros(mesh.n_elem)           # sum_c w_c (lam_e - u_e) . vth
            self_e = E0 * E0 * eps * eps * cm
            for ci, f in enumerate(loads):
                u, lam = us[ci], lams[ci]
                comp += cw[ci] * (float(u @ (f - 2 * fth)) + float(np.sum(pth * g * self_e)))
                m_e += cw[ci] * 0.25 * (fe.strain_energy(lam + u) - fe.strain_energy(lam - u))
                q_e += cw[ci] * ((lam - u)[mesh.edof] @ vth)
            refs.setdefault("c", comp)
            dp = np.where(live, 3 * xe ** 2 * (1 - E_MIN), 0.0)
            sw = cw.sum()
            g_xe = (-(3 * xe ** 2 * (1 - E_MIN)) * g * m_e + E0 * dp * g * eps * q_e
                    + sw * dp * g * self_e) / refs["c"]
            dc_dte = (-estiff * dg * m_e + E0 * pth * (dg * eps + g * a) * q_e
                      + sw * E0 * E0 * pth * cm * (dg * eps * eps + 2 * g * eps * a)) / refs["c"]
        rhs = np.bincount(mesh.enode.ravel(), weights=np.repeat(dc_dte / 8.0, 8), minlength=mesh.n_nodes)
        obj = comp / refs["c"]
        if thermal_weight > 0:
            tj = th.objective()
            refs.setdefault("t", abs(tj) or 1.0)
            obj += thermal_weight * tj / refs["t"]
            rhs = rhs + thermal_weight * th.l / refs["t"]
        if np.any(rhs):
            th.lam = th.adjoint(rhs, th.lam)
            g_xe = g_xe + th.gradient(xe, th.lam)
        rec = {"compliance" if a is None else "strainEnergy2": comp / E0, "maxT": float(theta.max() + bc.ambient),
               "minRetention": float(g[xe > 0.5].min()) if np.any(xe > 0.5) else 1.0}
        return obj, g_xe, rec

    return evaluate, fe, th
