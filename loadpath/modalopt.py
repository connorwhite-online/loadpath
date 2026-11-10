"""
Natural frequencies as constraints in the optimizer.

The search could only read a part's modes on the delivered mesh (verify's
modal check): the optimizer itself saw the vibration block as carried masses
shaken along x, y and z (domain._inertial_cases), and for a free part
(inertiaRelief: a drone in flight) not at all. A frame whose canopy rocked
on its four posts at 153 Hz against a 546 Hz floor was found out only after
it was meshed.

ModalModel computes the density design's own lowest modes on the optimizer's
grid, with everything the modal check counts: what the part carries
(vibration.masses, inertiaRelief.masses, rigid bodies with their inertia), a
free part's modes free-free with the six rigid ones left out, pins as their
springs. Each mode's eigenvalue has an exact gradient,

    d(w^2) / dx = phi^T (dK/dx - w^2 dM/dx) phi        (phi^T M phi = 1)

and the block's limits become constraints g <= 0 on the frequencies:

- minFrequency: every mode at or above the floor, f_min / f_i - 1 <= 0;
- excitations: no mode inside a band (modal.band_keepout). A mode at or
  above the band's bottom is driven out over its top, f_top / f_i - 1 <= 0;
  a mode under the bottom is left alone. Nothing is ever driven down or
  held down: the softest way to lower a mode is to cut the part in two (an
  optimizer asked to lower one does), and a grey design early in a run has
  every mode low, which says nothing about where they will end. A mode that
  climbs into a band as the design forms is pushed on through it. A band
  that reaches down to the floor raises the floor to its top.

Modes cross and pair up (a symmetric frame's sway comes in twos), so each
limit is a smooth maximum (KS) over the modes it counts, not the lowest one
alone. Cells at low density get stiffness x^p and, below 0.1, mass x^6
(Du and Olhoff 2007): mass that fades faster than stiffness, or the empty
region's own slow modes would be the lowest ones read.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

from .fem import HexMesh

__all__ = ["ModalModel", "modal_model"]

_CONN26 = np.ones((3, 3, 3), bool)
MASS_CUT = 0.1          # below this density a cell's mass fades as x^6
KS_RHO = 30.0


def _mass_interp(x):
    c = MASS_CUT ** -5
    return np.where(x >= MASS_CUT, x, c * x ** 6), np.where(x >= MASS_CUT, 1.0, 6 * c * x ** 5)


def _ks(v, rho=KS_RHO):
    """Smooth maximum of v (> 0) and its weights (sum to 1)."""
    v = np.asarray(v, float)
    vmax = float(v.max())
    w = np.exp(rho * (v / vmax - 1.0))
    return vmax * (1.0 + np.log(w.sum()) / rho), w / w.sum()


def modal_model(study, domain):
    """The study's vibration block as optimizer constraints, or None: no
    block, inOptimizer false, or no floor and no band to hold."""
    vib = getattr(study, "vibration", None)
    if vib is None or not vib.in_optimizer:
        return None
    from .modal import band_keepout, loss_factor

    eta = loss_factor(study.material)
    bands = []
    for ex in vib.excitations:
        k = band_keepout(ex[1], ex[2], eta, ex[3])
        if k is not None:
            bands.append((ex[0], k[0], k[1]))
    if vib.min_frequency is None and not bands:
        return None
    free = bool(study.inertia_relief)
    if not free and not domain.fixed.any() and domain.springs is None:
        return None
    return ModalModel(study, domain, vib.min_frequency, bands)


class ModalModel:
    """Lowest natural frequencies of a density design on `domain`'s grid and
    the vibration limits as constraints with gradients (see the module)."""

    def __init__(self, study, domain, floor=None, bands=(), n_modes: int = 8, alive: float = 1e-3):
        from .domain import _region_nodes, carried_bodies
        from .fem import rigid_body_modes

        self.domain = domain
        mesh = domain.mesh
        mat = study.material
        self.nu = float(mat.nu)
        self.modulus = float(mat.E)
        self.density = float(mat.density)
        self.free = bool(study.inertia_relief)
        self.floor = None if floor is None else float(floor)
        self.bands = [(str(n), float(lo), float(hi)) for n, lo, hi in bands]
        self.n_modes = int(n_modes)
        self.alive = alive
        vib = study.vibration
        # what it carries, as the modal check counts it (verify.modal_check)
        self.masses = []
        for i, (region, shape, kg) in enumerate(vib.masses):
            nodes = np.flatnonzero(_region_nodes(study, mesh, region, shape, f"vibration.masses[{i}]"))
            self.masses.append((nodes, float(kg)))
        self.blocks = []
        if self.free:
            named = {r for r, _, _ in vib.masses if r is not None}
            ids = {id(sh) for _, sh, _ in vib.masses}
            for i, (region, shape, kg) in enumerate(getattr(study, "inertia_masses", []) or []):
                if (region is not None and region in named) or id(shape) in ids:
                    continue
                nodes = np.flatnonzero(_region_nodes(study, mesh, region, shape, f"inertiaRelief.masses[{i}]"))
                self.masses.append((nodes, float(kg)))
            if getattr(study, "inertia_bodies", None):
                to_t = self.density * 1e-6 * 1e-3                    # mm^3 of material -> tonnes
                for dofs, dmat, mc in carried_bodies(study, mesh, rigid_body_modes(mesh.node_xyz)):
                    self.blocks.append((np.asarray(dofs), dmat @ (mc * to_t) @ dmat.T))
        carried = np.zeros(mesh.n_nodes, bool)
        for nodes, _ in self.masses:
            carried[nodes] = True
        for dofs, _ in self.blocks:
            carried[dofs // 3] = True
        self.anchor = domain.preserve | carried[mesh.enode].any(axis=1)
        self.support_touch = domain.support_nodes[mesh.enode].any(axis=1)
        self.last = {}

    def slots(self) -> list:
        """Every constraint name evaluate can return (the optimizer's fixed slots)."""
        out = ["minFrequency"]
        for name, _, _ in self.bands:
            out.append(f"band.{name}")
        return out

    def _live(self, xe):
        """The cells the modes are read on: the body the supports hold, or a
        free part's largest body (loose crumbs would add zero modes)."""
        mesh = self.domain.mesh
        alive = (xe > self.alive) | self.anchor
        g = np.zeros(self.domain.dims, bool)
        ijk = tuple(mesh.elem_ijk.T)
        g[ijk] = alive
        lbl, nb = ndimage.label(g, structure=_CONN26)
        el = lbl[ijk]
        if self.free:
            sizes = np.bincount(el[alive], minlength=nb + 1)
            sizes[0] = 0
            return alive & (el == int(np.argmax(sizes)))
        keep_l = np.unique(el[alive & self.support_touch])
        return alive & np.isin(el, keep_l[keep_l > 0])

    def modes(self, xe, p: float):
        """(frequencies Hz, d(w^2)/dxe per mode (k, n_elem)) of the design."""
        from .modal import modal_analysis
        from .simp import E_MIN

        d = self.domain
        mesh = d.mesh
        el = np.flatnonzero(self._live(xe))
        sub = HexMesh(mesh.elem_ijk[el], mesh.pitch, mesh.origin)
        fn = mesh.node_id(sub.node_ijk)                              # sub node -> full node
        back = np.full(mesh.n_nodes, -1, np.int64)
        back[fn] = np.arange(len(fn))
        fd = (3 * fn[:, None] + np.arange(3)).ravel()
        x = xe[el]
        e = self.modulus * (E_MIN + x ** p * (1 - E_MIN))
        de = self.modulus * p * x ** (p - 1) * (1 - E_MIN)
        mi, dmi = _mass_interp(x)
        masses = []
        for nodes, kg in self.masses:
            sn = back[nodes]
            sn = sn[sn >= 0]
            if len(sn):
                masses.append((sn, kg))
        blocks = []
        for dofs, blk in self.blocks:
            sd = 3 * back[dofs // 3] + dofs % 3
            if (back[dofs // 3] >= 0).all():
                blocks.append((sd, blk))
        springs = None
        if not self.free and d.springs is not None:
            springs = self.modulus * d.springs[fd][:, fd]
        nm = min(self.n_modes, max(1, sub.n_dof - 8))
        res = modal_analysis(sub, e, self.density * mi, self.nu, d.fixed[fd], nm, masses, springs=springs,
                             free=self.free, mass_blocks=blocks)
        lam = (2 * np.pi * np.asarray(res.frequencies, float)) ** 2
        from .fem import Stiffness

        ke = Stiffness(sub, self.nu, np.zeros(sub.n_dof, bool)).ke
        dm = self.density * 1e-9 * mesh.pitch ** 3 / 8.0 * dmi           # t per node of the cell, per unit x
        grads = np.zeros((len(lam), mesh.n_elem))
        for i in range(len(lam)):
            pe = res.modes[:, i][sub.edof]
            grads[i, el] = de * np.einsum("mi,ij,mj->m", pe, ke, pe) - lam[i] * dm * (pe * pe).sum(axis=1)
        return np.asarray(res.frequencies, float), lam, grads

    def evaluate(self, xe: np.ndarray, p: float):
        """[(name, g, dg/dxe)]: each limit as g <= 0."""
        t0 = time.time()
        f, lam, dlam = self.modes(xe, p)
        ok = lam > 1e-9
        f, lam, dlam = f[ok], lam[ok], dlam[ok]
        out = []
        self.last = {"frequencies": [round(float(v), 2) for v in f]}
        if not len(f):
            return out

        def above(name, f_req, idx):
            # every mode in idx at or over f_req
            v = f_req / f[idx]
            ks, w = _ks(v)
            g = ((w * (-0.5 * v / lam[idx]))[:, None] * dlam[idx]).sum(axis=0)
            out.append((name, ks - 1.0, g))

        floor = self.floor or 0.0
        every = np.arange(len(f))
        for name, lo, hi in sorted(self.bands, key=lambda b: b[1]):
            if hi <= floor:
                continue
            if lo <= floor:
                floor = hi                                   # nothing may stay under it
                continue
            idx = every[f >= lo]
            if len(idx):
                above(f"band.{name}", hi, idx)
        if floor > 0:
            above("minFrequency", floor, every)
        self.last.update(floor=floor, seconds=round(time.time() - t0, 2))
        return out
