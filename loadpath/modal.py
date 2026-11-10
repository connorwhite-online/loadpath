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

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .fem import HexMesh, Stiffness

__all__ = ["magnification", "band_keepout", "band_response", "ModalResult", "modal_analysis", "transmissibility", "base_transmissibility", "LOSS_FACTOR", "loss_factor", "vibration_report"]

# Room-temperature loss factor (tan delta) near 10-100 Hz, typical values
# from DMA data sheets and literature; rough (+-50%), and strongly
# temperature dependent near Tg.
LOSS_FACTOR = {
    "pla": 0.015, "petg": 0.02, "abs": 0.02, "asa": 0.02, "pc": 0.012, "pa12": 0.04, "pa11": 0.05,
    "pa12-gb": 0.03, "pa12-cf": 0.03, "tpu": 0.12, "resin-rigid": 0.03, "resin-tough": 0.05,
    "alsi10mg": 0.002, "ti64": 0.002, "316l": 0.002, "aluminum": 0.001, "steel": 0.001,
}


# Up to this many free dofs the eigensolve factors the reduced K with
# scipy's own shift-invert; above it with bucklingopt._factor, which is
# ~8x faster already at 25k dofs (a 8,000-cell bar: 21.9 s -> 2.7 s, same
# frequencies) and 16x at 60k
DIRECT_EIG_DOFS = 5_000


def loss_factor(material) -> float:
    name = getattr(material, "name", str(material)).lower()
    if name in LOSS_FACTOR:
        return LOSS_FACTOR[name]
    # grades of a family (tpu88a, tpu95a) damp like the family
    for fam in sorted(LOSS_FACTOR, key=len, reverse=True):
        if name.startswith(fam):
            return LOSS_FACTOR[fam]
    return 0.02


@dataclass
class ModalResult:
    frequencies: np.ndarray          # Hz, ascending
    modes: np.ndarray                # (n_dof, k), mass-normalized
    mass: float                      # kg, the part plus point masses
    participation: np.ndarray        # (k, 3) effective mass fraction per axis
    mass_vector: np.ndarray | None = None    # (n_dof,) lumped mass per dof, tonnes

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
                   point_masses: list | None = None, springs=None, free: bool = False,
                   mass_blocks: list | None = None) -> ModalResult:
    """Lowest `n_modes` natural frequencies (Hz) of a voxel model.

    modulus: E per element (MPa) or scalar; density: g/cm^3 per element or
    scalar; fixed: bool per dof (clamped). point_masses: [(node_ids, kg)],
    each spread evenly over its nodes (all three directions). springs: an
    (n_dof x n_dof) sparse stiffness added to K (pin supports).
    free=True: nothing holds it (a drone in flight); `fixed` is ignored, K
    is shifted by s M to factor it, and the six rigid-body modes (at zero)
    are left out of the result: its first frequency is the first flexible
    mode. mass_blocks: [(dofs, dense block, t)] added to M (a carried
    rigid body on a patch of nodes, domain.carried_bodies)."""
    from scipy.sparse.linalg import eigsh

    fixed = np.zeros(mesh.n_dof, bool) if free else np.asarray(fixed, bool)
    st = Stiffness(mesh, nu, fixed)
    k = st.assemble(np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,)))
    if springs is not None:
        # pin supports' penalty springs (n_dof x n_dof, already in MPa x mm)
        k = (k + springs).tocsr()
    m_node = _lumped_mass(mesh, density)
    for nodes, kg in point_masses or []:
        nodes = np.asarray(nodes, np.int64)
        m_node[nodes] += kg * 1e-3 / len(nodes)
    m = np.repeat(m_node, 3)
    mmat = sp.diags(m).tocsr()
    if mass_blocks:
        rows, cols, vals = [], [], []
        for dofs, blk in mass_blocks:
            dofs = np.asarray(dofs, np.int64)
            rows.append(np.repeat(dofs, len(dofs)))
            cols.append(np.tile(dofs, len(dofs)))
            vals.append(np.asarray(blk, float).ravel())
        mmat = (mmat + sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                     shape=mmat.shape)).tocsr()
    free_dofs = free
    free = np.flatnonzero(~fixed)
    mf = m[free]
    mff = mmat[free][:, free].tocsc()
    rigid = 6 if free_dofs else 0
    shift = 0.0
    if free_dofs:
        # K + s M is positive definite with the same modes, each at w^2 + s;
        # s at (2 pi 10 Hz)^2 (N/mm per tonne is 1/s^2)
        shift = (2 * np.pi * 10.0) ** 2
        k = (k + mmat * shift).tocsr()
    kf = k[free][:, free].tocsc()
    nm = min(n_modes + rigid, len(free) - 2)
    if len(free) <= DIRECT_EIG_DOFS:
        w, v = eigsh(kf, k=nm, M=mff, sigma=0.0, which="LM")
    elif os.environ.get("LOADPATH_MODAL_LOBPCG") != "1":
        # big models: shift-invert Lanczos with the factorization the push
        # pad and buckling solves use (bucklingopt._factor: PARDISO, or a
        # single precision LU in nested-dissection order refined by CG), on
        # the whole system with the held dofs as identity rows, restricted to
        # the free ones (zero mass on the held rows sent ARPACK to zeros).
        # LOBPCG with AMG (below) took over ten minutes on a 150k-dof TPU
        # suspension and warned of ill-conditioning all the way.
        from scipy.sparse.linalg import LinearOperator

        from .bucklingopt import _factor
        solve, free_mem = _factor(k.tocsr(), mesh.node_ijk)

        def inv(b):
            # K_ff^-1 b: the whole system with zero load on the held dofs
            full = np.zeros(mesh.n_dof)
            full[free] = np.asarray(b).ravel()
            return np.array(solve(full), dtype=float).ravel()[free]

        op = LinearOperator(kf.shape, matvec=inv, dtype=float)
        try:
            w, v = eigsh(kf, k=nm, M=mff, sigma=0.0, which="LM", OPinv=op)
        finally:
            if free_mem is not None:
                free_mem()
    else:
        # big models: LOBPCG on the mass-scaled operator with an AMG preconditioner
        from scipy.sparse.linalg import lobpcg

        from .fem import amg_solver

        ml = amg_solver(kf.tocsr())
        x0 = np.random.default_rng(0).standard_normal((len(free), nm + 4))
        w, v = lobpcg(kf, x0, B=mff, M=ml.aspreconditioner(), largest=False, tol=1e-6, maxiter=400)
        w, v = w[:nm], v[:, :nm]
    order = np.argsort(w)
    w, v = w[order] - shift, v[:, order]
    if rigid:
        w, v = w[rigid:], v[:, rigid:]
        nm -= rigid
    w = np.maximum(w, 0.0)
    # mass-normalize (phi^T M phi = 1)
    mv = mff @ v
    v = v / np.sqrt(np.einsum("ik,ik->k", v, mv))[None, :]
    mv = mff @ v
    modes = np.zeros((mesh.n_dof, nm))
    modes[free] = v
    total = m_node.sum()
    if mass_blocks:
        # a rigid body's mass is its translational block's: ones^T M ones / 3
        one = np.ones(mesh.n_dof)
        total = float(one @ (mmat @ one)) / 3.0
    part = np.zeros((nm, 3))
    for a in range(3):
        r = np.zeros(mesh.n_dof)
        r[a::3] = 1.0
        g = mv.T @ r[free]                                      # participation factor phi^T M r
        part[:, a] = g ** 2 / total
    return ModalResult(np.sqrt(w) / (2 * np.pi), modes, float(total * 1e3), part, m)


def transmissibility(f_forcing, f_natural, eta: float):
    """Base-excitation transmissibility |X / Y| of one mode with structural
    damping (loss factor eta): 1 / eta at resonance, below 1 above
    sqrt(2) f_n (isolation)."""
    r2 = (np.asarray(f_forcing, float) / f_natural) ** 2
    return np.sqrt((1 + eta ** 2) / ((1 - r2) ** 2 + eta ** 2))


def magnification(f_forcing, f_natural, eta: float):
    """|X / X_static| of one mode driven by a force at f_forcing, with
    structural damping (loss factor eta): 1 at rest, 1 / eta at resonance,
    falling as (f_n / f)^2 far above it."""
    r2 = (np.asarray(f_forcing, float) / f_natural) ** 2
    return 1.0 / np.sqrt((1 - r2) ** 2 + eta ** 2)


def band_keepout(lo: float, hi: float, eta: float, max_amplification: float | None = None):
    """The natural frequencies (f_lo, f_hi) a forcing that sweeps lo..hi Hz
    must not find. With no limit on magnification that is the band itself: a
    mode inside it is at resonance at some speed. With one, the band widens
    to where a mode at its edge is magnified exactly that much: from
    1 / ((1 - r^2)^2 + eta^2) = A^2, r^2 = 1 -+ sqrt(1 / A^2 - eta^2), so a
    mode above the band must clear hi / r_below and one below it must stay
    under lo / r_above. None when resonance itself is within the limit
    (A >= 1 / eta: a rubber's own damping holds it)."""
    if max_amplification is None:
        return float(lo), float(hi)
    d = 1.0 / max_amplification ** 2 - eta ** 2
    if d <= 0:
        return None
    d = np.sqrt(d)
    return float(lo / np.sqrt(1.0 + d)), float(hi / np.sqrt(max(1.0 - d, 1e-12)))


def band_response(frequencies, excitations, eta: float) -> list[dict]:
    """Each excitation (name, lo, hi, max amplification or None) against the
    part's modes: the modes inside the band, the worst magnification any
    mode gives the forcing anywhere in the band (1 / eta inside it; at the
    nearer edge outside), the frequencies kept clear (band_keepout), whether
    the computed modes reach the top of that range (`covered`), and `margin`
    for the search: the lowest offending mode has to climb past the range's
    top, and with carried masses the material that takes goes as the
    frequency squared (as the floor's margin does)."""
    f = np.asarray(frequencies, float)
    out = []
    for name, lo, hi, amp in excitations:
        keep = band_keepout(lo, hi, eta, amp)
        row = {"name": name, "band": [lo, hi]}
        if amp is not None:
            row["maxAmplification"] = amp
        # each mode's worst magnification over the band
        edge = np.clip(f, lo, hi)                    # the forcing frequency nearest the mode
        mag = magnification(edge, np.maximum(f, 1e-9), eta) if len(f) else np.zeros(0)
        inside = (f >= lo) & (f <= hi)
        row["modesInBand"] = [{"mode": int(i) + 1, "frequency": round(float(f[i]), 2)} for i in np.flatnonzero(inside)]
        if len(f):
            w = int(np.argmax(mag))
            row["amplification"] = round(float(mag[w]), 2)
            row["worstMode"] = {"mode": w + 1, "frequency": round(float(f[w]), 2)}
        if keep is None:
            row.update(ok=True, covered=True, margin=float("inf"),
                       note=f"resonance itself magnifies {1.0 / eta:.1f}x, within the limit")
            out.append(row)
            continue
        row["keepClear"] = [round(keep[0], 2), round(keep[1], 2)]
        bad = (f > keep[0]) & (f < keep[1])
        row["covered"] = bool(len(f) and f[-1] >= keep[1])
        row["ok"] = bool(not bad.any() and row["covered"])
        if bad.any():
            row["margin"] = float((f[bad].min() / keep[1]) ** 2)
        elif not row["covered"]:
            # every computed mode is under the range and there are more above
            # the last of them: not shown clear
            row["margin"] = float((f[-1] / keep[1]) ** 2) if len(f) else 0.0
        else:
            row["margin"] = float("inf")
        out.append(row)
    return out


def base_transmissibility(res: ModalResult, dofs, axis: int, freqs, eta: float, fixed=None,
                          stiffness_scale=None) -> np.ndarray:
    """|absolute motion / base motion| at the given dofs (averaged, as complex
    numbers) for a base shaken along `axis` (0, 1, 2) at each frequency, by
    modal superposition with structural damping eta. The base is the
    clamped dofs; `fixed` (bool per dof) leaves their mass out of the
    participation factors. Truncated to the computed modes, so it holds up
    to well below the highest of them. stiffness_scale (one per frequency)
    multiplies every mode's stiffness at that frequency: a viscoelastic
    mount that stiffens with frequency (viscoelastic.dynamic_factor over its
    value where the modes were computed)."""
    m = res.mass_vector.copy()
    if fixed is not None:
        m[np.asarray(fixed, bool)] = 0.0
    r = np.zeros(len(m))
    r[axis::3] = 1.0
    gam = res.modes.T @ (m * r)                                  # participation factors
    phi = res.modes[np.asarray(dofs, np.int64)].mean(axis=0)      # (k,)
    wr2 = (2 * np.pi * res.frequencies) ** 2
    w2 = (2 * np.pi * np.asarray(freqs, float))[:, None] ** 2
    if stiffness_scale is not None:
        wr2 = wr2[None, :] * np.asarray(stiffness_scale, float)[:, None]
    drive = 1.0 if np.all(np.asarray(dofs) % 3 == axis) else 0.0
    wr2 = np.broadcast_to(wr2, (w2.shape[0], len(gam)))
    rel = (phi * gam)[None, :] * w2 / (wr2 - w2 + 1j * eta * wr2)
    return np.abs(drive + rel.sum(axis=1))


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
