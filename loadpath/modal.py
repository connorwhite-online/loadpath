"""
Natural frequencies, mode shapes and damping, on the same voxel hex mesh.

A part that shakes (a motor mount, an arm link carrying a gripper, a sensor
bracket on a printer gantry) fails by resonance long before its static
safety factor says it should: if a mode sits near the excitation the
response is the static one times 1/(2 zeta). This module finds the modes:

- `modal_analysis`: the generalized eigenproblem K phi = omega^2 M phi with
  the consistent-free lumped mass (each cell's mass split over its 8 nodes),
  per-cell modulus and density, so a homogenized graded lattice (modulus
  E C rho^n, density rho_s rho) is analysed exactly like a solid. Point
  masses (a motor, a gripper) attach to nodes. Returns frequencies (Hz),
  modes, and each mode's effective mass fraction per axis (how much of
  the part a base shake in that direction moves).
- `transmissibility`: base-to-part amplification of a single mode at a
  forcing frequency, with modal damping zeta = eta / 2 from the material's
  loss factor (LOSS_FACTOR).

On damping: a part made of one linear viscoelastic material has the modal
loss factor of that material whatever its shape (every mode stores its
energy in the same material), so a lattice of PA12 damps like solid PA12.
Lattices raise damping only through something else: a second, lossy
material (a TPU lattice, a TPU-filled core), friction, or unfused powder
left in the cells (particle damping, large but unmodelled here). What a
lattice does change is where the modes are: E/rho drops as rho^(n-1), so a
lattice core lowers frequencies, which is what an isolator wants and a
stiff bracket does not.

Units: mm, N, MPa, g/cm^3 for density (converted to tonne/mm^3, which with
N and mm gives seconds), kg for point masses, Hz out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .fem import HexMesh, Stiffness

__all__ = ["ModalResult", "modal_analysis", "transmissibility", "LOSS_FACTOR", "loss_factor", "vibration_report"]

# Room-temperature loss factor (tan delta) near 10-100 Hz, typical values
# from DMA data sheets and literature; rough (+-50%), and strongly
# temperature dependent near Tg.
LOSS_FACTOR = {
    "pla": 0.015, "petg": 0.02, "abs": 0.02, "asa": 0.02, "pc": 0.012, "pa12": 0.04, "pa11": 0.05,
    "pa12-gb": 0.03, "pa12-cf": 0.03, "tpu": 0.12, "resin-rigid": 0.03, "resin-tough": 0.05,
    "alsi10mg": 0.002, "ti64": 0.002, "316l": 0.002, "aluminum": 0.001, "steel": 0.001,
}


def loss_factor(material) -> float:
    name = getattr(material, "name", str(material)).lower()
    return LOSS_FACTOR.get(name, 0.02)


@dataclass
class ModalResult:
    frequencies: np.ndarray          # Hz, ascending
    modes: np.ndarray                # (n_dof, k), mass-normalized
    mass: float                      # kg, the part plus point masses
    participation: np.ndarray        # (k, 3) effective mass fraction per axis

    def to_dict(self, eta: float | None = None) -> dict:
        out = {"frequencies": [round(float(f), 2) for f in self.frequencies], "mass": round(self.mass, 5),
               "massFraction": [[round(float(x), 3) for x in row] for row in self.participation]}
        if eta is not None:
            out["lossFactor"] = eta
            out["dampingRatio"] = round(eta / 2.0, 4)
            out["resonantAmplification"] = round(1.0 / eta, 1)
        return out


def _lumped_mass(mesh: HexMesh, density):
    rho = np.broadcast_to(np.asarray(density, float), (mesh.n_elem,)) * 1e-9     # g/cm^3 -> t/mm^3
    m_node = np.zeros(mesh.n_nodes)
    np.add.at(m_node, mesh.enode, (rho * mesh.pitch ** 3 / 8.0)[:, None])
    return m_node


def modal_analysis(mesh: HexMesh, modulus, density, nu: float, fixed: np.ndarray, n_modes: int = 6,
                   point_masses: list | None = None) -> ModalResult:
    """Lowest `n_modes` natural frequencies (Hz) of a voxel model.

    modulus: E per element (MPa) or scalar; density: g/cm^3 per element or
    scalar; fixed: bool per dof (clamped). point_masses: [(node_ids, kg)],
    each spread evenly over its nodes (all three directions)."""
    from scipy.sparse.linalg import eigsh

    fixed = np.asarray(fixed, bool)
    st = Stiffness(mesh, nu, fixed)
    k = st.assemble(np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,)))
    m_node = _lumped_mass(mesh, density)
    for nodes, kg in point_masses or []:
        nodes = np.asarray(nodes, np.int64)
        m_node[nodes] += kg * 1e-3 / len(nodes)
    m = np.repeat(m_node, 3)
    free = np.flatnonzero(~fixed)
    kf = k[free][:, free].tocsc()
    mf = m[free]
    nm = min(n_modes, len(free) - 2)
    if len(free) <= 120_000:
        w, v = eigsh(kf, k=nm, M=sp.diags(mf).tocsc(), sigma=0.0, which="LM")
    else:
        # big models: LOBPCG on the mass-scaled operator with an AMG preconditioner
        from scipy.sparse.linalg import lobpcg

        from .fem import amg_solver

        ml = amg_solver(kf.tocsr())
        x0 = np.random.default_rng(0).standard_normal((len(free), nm + 4))
        w, v = lobpcg(kf, x0, B=sp.diags(mf), M=ml.aspreconditioner(), largest=False, tol=1e-6, maxiter=400)
        w, v = w[:nm], v[:, :nm]
    order = np.argsort(w)
    w, v = np.maximum(w[order], 0.0), v[:, order]
    # mass-normalize (phi^T M phi = 1)
    v = v / np.sqrt(np.einsum("ik,i,ik->k", v, mf, v))[None, :]
    modes = np.zeros((mesh.n_dof, nm))
    modes[free] = v
    total = m_node.sum()
    part = np.zeros((nm, 3))
    for a in range(3):
        r = np.zeros(mesh.n_dof)
        r[a::3] = 1.0
        g = (v * (mf * r[free])[:, None]).sum(axis=0)          # participation factor phi^T M r
        part[:, a] = g ** 2 / total
    return ModalResult(np.sqrt(w) / (2 * np.pi), modes, float(total * 1e3), part)


def transmissibility(f_forcing, f_natural, eta: float):
    """Base-excitation transmissibility |X / Y| of one mode with structural
    damping (loss factor eta): 1 / eta at resonance, below 1 above
    sqrt(2) f_n (isolation)."""
    r2 = (np.asarray(f_forcing, float) / f_natural) ** 2
    return np.sqrt((1 + eta ** 2) / ((1 - r2) ** 2 + eta ** 2))


def vibration_report(graph, spec: dict, material) -> dict:
    """The field graph's `vibration` block: modes of the `envelope` filled
    with the `lattice` at `density` (homogenized: E C rho^n, rho_s rho),
    clamped inside the `fixed` solid, with point masses."""
    from .impact import sample_core
    from .lattice import sheet_modulus

    kind = spec.get("lattice", "gyroid")
    inside, rho, lo, h = sample_core(graph, spec["envelope"], spec.get("density"), float(spec.get("pitch", 1.0)))
    ijk = np.argwhere(inside)
    mesh = HexMesh(ijk, h, lo)
    r = rho[inside]
    mod = float(material.E) * sheet_modulus(kind, r)
    dens = float(material.density) * r

    def nodes_in(node):
        return np.flatnonzero(graph.evaluate(mesh.node_xyz, node) <= 0.5 * h)

    fixed = np.zeros(mesh.n_dof, bool)
    fn = nodes_in(spec["fixed"])
    if len(fn) == 0:
        raise ValueError(f"vibration: no nodes inside the fixed node {spec['fixed']!r}")
    for a in range(3):
        fixed[3 * fn + a] = True
    pm = []
    for p in spec.get("pointMasses", []):
        nn = nodes_in(p["region"])
        if len(nn) == 0:
            raise ValueError(f"vibration: point mass region {p['region']!r} has no nodes")
        pm.append((nn, float(p["mass"])))
    res = modal_analysis(mesh, mod, dens, float(material.nu), fixed, int(spec.get("modes", 6)), pm)
    eta = float(spec.get("lossFactor", loss_factor(material)))
    out = {"lattice": kind, "cells": int(mesh.n_elem), **res.to_dict(eta)}
    f1 = float(res.frequencies[0])
    if spec.get("minFrequency") is not None:
        out["minFrequency"] = float(spec["minFrequency"])
        out["ok"] = bool(f1 >= float(spec["minFrequency"]))
    if spec.get("excitation") is not None:
        fx = float(spec["excitation"])
        out["excitation"] = fx
        out["transmissibility"] = round(float(max(transmissibility(fx, f, eta) for f in res.frequencies)), 2)
    return out
