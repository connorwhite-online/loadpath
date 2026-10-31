"""
Lattice vibration isolators: elastomer (TPU) lattice feet under a payload,
tuned to a natural frequency, levelled under an off-centre load, and
checked against shock.

An isolator is a spring chosen soft enough that the payload's mounted
frequency f_n sits well below the excitation: above sqrt(2) f_n the base
motion is attenuated. Four identical feet under a payload whose centre of
gravity is off the middle of the footprint do three things wrong:

- the payload sits tilted (the feet near the CG squash more);
- bounce and rocking couple, so a vertical shake also rocks the payload and
  the mount resonates at several frequencies instead of one, pushing the
  start of isolation up;
- the most loaded foot reaches the end of its linear range first, so the
  mount takes less shock before it starts to buckle.

The classic fix is the "elastic centre" rule: give each foot a stiffness
so that the stiffness-weighted centre of the feet sits under the CG
(sum k_i = K, sum k_i x_i = K x_cg). Then gravity squashes every foot by
the same amount (level), vertical bounce is a pure translation, and every
foot carries load in proportion to its stiffness. With a lattice the
stiffness of each foot is a density (k = E C rho^n A / H), so a graded
set of feet does this at no cost in parts or footprint.

- `foot_stiffness`, `density_for_stiffness`: homogenized axial and lateral
  stiffness of a block foot.
- `rigid_body`: mass, CG and inertia of a set of point masses.
- `mount_modes`: the six rigid-body modes of a payload on springs.
- `tune_mount`: per-foot stiffness for a target bounce frequency with the
  elastic centre under the CG (bounded by the printable density range).
- `shock_margin`: the acceleration (in g, on top of gravity) at which the
  first foot reaches its elastic buckling stress.

Voxel FE of the full mount (modal_analysis, base_transmissibility) checks
all of it: see examples/isolator/mount.py and docs/IMPACT.md.

Units: mm, N, MPa, kg; stiffness N/mm.
"""
from __future__ import annotations

import numpy as np

from .lattice import SHEETS, sheet_modulus

__all__ = ["foot_stiffness", "density_for_stiffness", "rigid_body", "mount_modes", "tune_mount",
           "shock_margin", "G"]

G = 9810.0      # mm/s^2


def foot_stiffness(kind: str, rho, E: float, width: float, height: float, nu_eff: float = 0.3):
    """Axial and lateral stiffness (N/mm) of a square `width` x `width` foot
    `height` tall, homogenized lattice of `kind` at density rho, glued at
    both ends. Lateral: shear and guided bending in series."""
    e = E * sheet_modulus(kind, rho)
    a = width * width
    kz = e * a / height
    g = e / (2 * (1 + nu_eff))
    k_shear = g * a / height
    k_bend = 12 * e * (width ** 4 / 12) / height ** 3
    return kz, 1.0 / (1.0 / k_shear + 1.0 / k_bend)


def density_for_stiffness(kind: str, k, E: float, width: float, height: float):
    """Density giving axial stiffness k (N/mm); inverse of foot_stiffness."""
    s = SHEETS[kind]
    rel = np.asarray(k, float) * height / (E * width * width)
    return (rel / s["C"]) ** (1.0 / s["n"])


def rigid_body(points, masses):
    """Total mass (kg), CG (mm) and inertia tensor about the CG (kg mm^2)
    of point masses."""
    p = np.asarray(points, float).reshape(-1, 3)
    m = np.asarray(masses, float).ravel()
    tot = float(m.sum())
    cg = (p * m[:, None]).sum(axis=0) / tot
    r = p - cg
    inertia = (m[:, None, None] * ((r * r).sum(axis=1)[:, None, None] * np.eye(3)[None]
                                   - r[:, :, None] * r[:, None, :])).sum(axis=0)
    return tot, cg, inertia


def _skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def mount_modes(mass: float, cg, inertia, feet, kz, ks):
    """Rigid payload on spring feet. feet: (n, 3) top-centre of each foot
    (where it meets the payload); kz, ks: axial and lateral stiffness per
    foot (N/mm). Returns frequencies (Hz, ascending), mode vectors
    (6, 6: translation mm, rotation rad, about the CG) and each mode's
    effective mass fraction per translation axis."""
    from scipy.linalg import eigh

    feet = np.asarray(feet, float)
    kz = np.broadcast_to(np.asarray(kz, float), (len(feet),))
    ks = np.broadcast_to(np.asarray(ks, float), (len(feet),))
    # SI inside: translations in m, rotations in rad, stiffness N/m, inertia kg m^2
    kmat = np.zeros((6, 6))
    for x, a, b in zip(feet, kz, ks):
        bm = np.hstack([np.eye(3), -_skew((x - np.asarray(cg, float)) * 1e-3)])     # u = t + theta x r
        kmat += bm.T @ (np.diag([b, b, a]) * 1e3) @ bm
    mmat = np.zeros((6, 6))
    mmat[:3, :3] = np.eye(3) * mass
    mmat[3:, 3:] = np.asarray(inertia, float) * 1e-6
    w2, v = eigh(kmat, mmat)
    gam = v[:3, :].T * mass               # phi^T M r for unit translations (v is M-normalized)
    return np.sqrt(np.maximum(w2, 0)) / (2 * np.pi), v, gam ** 2 / mass


def tune_mount(cg_xy, feet_xy, k_total: float, k_min=None, k_max=None):
    """Per-foot axial stiffness with sum k = k_total and the stiffness
    centre under the CG (sum k x = k_total cg), closest to equal feet (least
    squares), within [k_min, k_max] per foot. Returns (k, residual): the
    residual is the CG offset (mm) left over when the bounds bind."""
    from scipy.optimize import lsq_linear

    p = np.asarray(feet_xy, float)[:, :2]
    n = len(p)
    a = np.vstack([np.ones(n), p[:, 0], p[:, 1]])
    b = k_total * np.array([1.0, cg_xy[0], cg_xy[1]])
    k0 = np.full(n, k_total / n)
    # the constraints, weighted heavily, plus a small pull toward equal feet
    w = 1e3
    span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
    aa = np.vstack([w * a / np.array([[1.0], [span], [span]]), np.eye(n)])
    bb = np.concatenate([w * b / np.array([1.0, span, span]), k0])
    lo = np.full(n, 0.0 if k_min is None else k_min)
    hi = np.full(n, np.inf if k_max is None else k_max)
    k = lsq_linear(aa, bb, bounds=(lo, hi)).x
    centre = (k[:, None] * p).sum(axis=0) / k.sum()
    return k, float(np.hypot(*(centre - np.asarray(cg_xy, float)[:2])))


def shock_margin(kind: str, rho, E: float, width: float, load, creep: float = 1.0, rate: float = 1.0):
    """Acceleration (in g, on top of the 1 g static load) at which the most
    loaded foot buckles. load: static force per foot (N). Buckling of thin
    elastic walls is set by strain, so the margin is a strain budget: the
    buckling strain (0.05 E rho^2 / E_foot, the elastic crush law in
    impact.py) less the static strain, which grows by `creep`
    (viscoelastic.creep_factor) over the years under load, with the shock's
    extra strain taken at the stiffer modulus of a fast event (`rate`,
    viscoelastic.event_factor of the pulse length)."""
    from .impact import plateau_stress

    fb = plateau_stress(f"{kind}:elastic", np.asarray(rho, float), E) * width * width
    return float(((fb / np.asarray(load, float)) - creep).min() * rate)
