"""
Anisotropic lattice homogenization: the full elasticity tensor of a unit
cell and a directional wall-stress surface, and both rotated into the frame
the cell sits in.

lattice.SHEETS / STRUTS read every cell as an isotropic material, and for a
strut cell the conservative one: the modulus is the lesser of its [100] and
[111] moduli, the wall stress the worst of a [100] pull, a [111] pull and a
shear. An octet is 1.5x stiffer along its struts than that, a simple cubic
cell 10x and more on its axes, a BCCz 2-3x along z; a part whose lattice is
loaded the way its cells are strong carries mass it does not need. The
scalar reading also scales the wall stress by the macro von Mises stress,
which is zero under a hydrostatic load the walls feel in full.

Here each cell is solved six times on a periodic voxel grid (a unit macro
strain per Voigt component, the nodes wrapping across the cell's faces):

- the mean stresses give the 6x6 effective stiffness C* (over the solid's
  E, engineering shear strains), symmetrized over the cell's own symmetry;
- the six nodal stress fields superpose to the walls' stress under any
  macro stress sigma: sigma -> C*^-1 sigma -> sum of the fields. Sampled
  over a few hundred directions (axes, diagonals, shears, hydrostatic and
  random), the walls' peak (99.5th percentile of nodal von Mises, as the
  scalar fits read it) is fitted by a quadratic form, peak^2 = sigma' P
  sigma, an anisotropic yield surface in the spirit of Hill's, then scaled
  up until it lies on or above every sample, so it never under-reads one.

Tables over relative density per cell type live in data/lattice_tensors.json
(bench/lattice_tensors.py writes them); tensors_at reads them at any density
(log-log on the leading entry, the shape interpolated in log rho, up to the
solid at rho = 1).

Frames: a cell's tensors are in its own axes; homogenized.py rotates them by
the lattice coordinates' local Jacobian (polar factor), so a cylindrical,
conformal or rotated lattice is read the way it actually lies.
"""
from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["unit_cell_tensors", "tensors_at", "rotate_stiffness", "rotate_stress_form", "stress_rotation",
           "strain_rotation", "has_tensors", "directional_modulus", "stress_surface", "VM_FORM", "TABLE_PATH"]

TABLE_PATH = Path(__file__).resolve().parent / "data" / "lattice_tensors.json"
_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))

# von Mises squared as a quadratic form of Voigt stress (tensor shears)
VM_FORM = np.array([[1.0, -0.5, -0.5, 0, 0, 0],
                    [-0.5, 1.0, -0.5, 0, 0, 0],
                    [-0.5, -0.5, 1.0, 0, 0, 0],
                    [0, 0, 0, 3.0, 0, 0],
                    [0, 0, 0, 0, 3.0, 0],
                    [0, 0, 0, 0, 0, 3.0]])


# ---- rotations ----------------------------------------------------------------------

def _basis(shear_half: bool) -> np.ndarray:
    """(6, 3, 3): the tensor of each Voigt unit vector; engineering shear
    strain puts 1/2 on each off-diagonal entry, stress 1."""
    e = np.zeros((6, 3, 3))
    for a, (i, j) in enumerate(_VOIGT):
        v = 0.5 if (shear_half and i != j) else 1.0
        e[a, i, j] = v
        e[a, j, i] = v
    return e


def _rotation_matrix(rot, shear_half: bool):
    """(..., 6, 6) T with v_local = T v_global for Voigt vectors, where
    `rot` (..., 3, 3) has the local axes as its columns (in global
    components): tensor_local = rot' tensor rot."""
    rot = np.asarray(rot, float)
    e = _basis(shear_half)
    loc = np.einsum("...pi,apq,...qj->...aij", rot, e, rot)
    ii = np.array([i for i, _ in _VOIGT])
    jj = np.array([j for _, j in _VOIGT])
    t = loc[..., ii, jj]                              # (..., a, b): component b of rotated unit a
    if shear_half:
        t = t * np.array([1, 1, 1, 2, 2, 2], float)  # engineering shear read back
    return np.swapaxes(t, -1, -2)


def stress_rotation(rot):
    return _rotation_matrix(rot, False)


def strain_rotation(rot):
    return _rotation_matrix(rot, True)


def rotate_stiffness(c_local, rot):
    """C in global axes for a cell whose local axes are the columns of
    `rot`: C_g = T_e' C_l T_e (strain energy is frame free)."""
    t = strain_rotation(rot)
    return np.einsum("...ab,...bc,...cd->...ad", np.swapaxes(t, -1, -2), c_local, t)


def rotate_stress_form(p_local, rot):
    """A quadratic form of stress (P) in global axes: P_g = T_s' P_l T_s."""
    t = stress_rotation(rot)
    return np.einsum("...ab,...bc,...cd->...ad", np.swapaxes(t, -1, -2), p_local, t)


@lru_cache(maxsize=None)
def _group(kind_sym: str) -> np.ndarray:
    """Proper rotations of the cube (24) or those keeping z (8)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            r = np.zeros((3, 3))
            for i, p in enumerate(perm):
                r[p, i] = signs[i]
            if np.linalg.det(r) < 0:
                continue
            if kind_sym == "tetragonal" and abs(r[2, 2]) != 1:
                continue
            if kind_sym == "orthorhombic" and not np.allclose(np.abs(r), np.eye(3)):
                continue
            out.append(r)
    return np.array(out)


def symmetry_of(kind: str) -> str:
    """cubic, tetragonal (z stands apart) or none (a random foam draw)."""
    from .struts import canonical

    k = canonical(kind)
    if k in ("bccz", "fccz"):
        return "tetragonal"
    if k in ("voronoi", "delaunay") or k.startswith("custom-"):
        return "none"
    return "cubic"


def _symmetrize(m, sym: str, rotate):
    m = 0.5 * (m + m.T)
    if sym == "none":
        return m
    g = _group(sym)
    return np.mean(rotate(np.broadcast_to(m, (len(g), 6, 6)), g), axis=0)


# ---- unit cell ----------------------------------------------------------------------

def _directions(seed=0, n_random=600):
    """Unit macro stresses (Voigt, Frobenius norm 1): axes, face and cube
    diagonals in tension and compression, pure shears, hydrostatic, random."""
    out = []
    dirs = [np.array(v, float) for v in itertools.product((-1, 0, 1), repeat=3) if any(v)]
    for d in dirs:
        d = d / np.linalg.norm(d)
        s = np.outer(d, d)
        out.append(np.array([s[i, j] for i, j in _VOIGT]))
    for a in range(3, 6):
        v = np.zeros(6)
        v[a] = 1.0
        out += [v, -v]
    for sgn in (1, -1):
        out.append(sgn * np.array([1, 1, 1, 0, 0, 0], float))
    rng = np.random.default_rng(seed)
    out += list(rng.normal(size=(n_random, 6)))
    out = np.array(out)
    frob = np.sqrt((out[:, :3] ** 2).sum(1) + 2 * (out[:, 3:] ** 2).sum(1))
    return out / frob[:, None]


def unit_cell_tensors(kind: str, wall_over_cell: float, n: int = 40, nu: float = 0.35, blend: float = 0.0,
                      percentile: float = 99.5, aspect=None, _keep: bool = False) -> dict:
    """Full anisotropic homogenization of one unit cell (module docstring)
    on a periodic voxel grid, n voxels along its shortest side. `wall_over_cell`:
    sheet wall or strut diameter over the cell; `blend` the strut node
    fillet over the cell; `aspect` (per axis) a stretched cell, its box
    `aspect` cells along each axis, as the field graph's `aspect` key lays
    it out. Returns rho, C (6x6 over E), the wall-stress surface (P, Q4,
    floor), held-out checks of it, the sampled directions' peaks, and the
    scalar readings (E100, E111, G) for comparison with the isotropic
    tables."""
    import pyamg
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg

    from .fem import CORNERS, _b_matrix, elastic_c, hex8_b, hex8_ke, von_mises
    from .lattice import tpms
    from .struts import is_strut, unit_cell_distance

    a = np.ones(3) if aspect is None else np.asarray(aspect, float)
    stretched = not np.allclose(a, 1.0)
    h = float(a.min()) / n
    dims = np.maximum(np.round(a / h).astype(int), 1)
    a = dims * h                                       # the box the grid holds exactly
    axes = [(np.arange(d) + 0.5) * h for d in dims]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    U = g / a
    if is_strut(kind):
        solid = unit_cell_distance(kind, U, blend, aspect=tuple(a) if stretched else None) < 0.5 * wall_over_cell
    else:
        solid = np.abs(tpms(kind, U, 1.0) * float(a.min())) < 0.5 * wall_over_cell
    eijk = np.argwhere(solid.reshape(tuple(dims)))
    ne = len(eijk)
    corner = (eijk[:, None, :] + CORNERS[None]) % dims
    used, inv = np.unique((corner[..., 0] * dims[1] + corner[..., 1]) * dims[2] + corner[..., 2],
                          return_inverse=True)
    enode = inv.reshape(ne, 8)
    nn = len(used)
    edof = (3 * enode[:, :, None] + np.arange(3)).reshape(ne, 24)
    ke = hex8_ke(nu) * h
    k = sp.csr_matrix((np.tile(ke.ravel(), ne), (np.repeat(edof, 24, axis=1).ravel(),
                                                 np.tile(edof, (1, 24)).ravel())), shape=(3 * nn, 3 * nn))
    free = np.ones(3 * nn, bool)
    free[:3] = False
    kf = k[free][:, free].tocsr()
    ml = pyamg.smoothed_aggregation_solver(kf, symmetry="hermitian", max_coarse=400, coarse_solver="splu")
    xc = (eijk[:, None, :] + CORNERS[None]) * h
    c0 = elastic_c(1.0, nu)
    bc = _b_matrix(0, 0, 0) / h
    bcorner = [hex8_b(x, y, z, nu) / h for (x, y, z) in 2.0 * CORNERS - 1.0]
    count = np.bincount(enode.ravel(), minlength=nn)[:, None]
    C = np.zeros((6, 6))
    L = np.zeros((nn, 6, 6), np.float32)            # nodal stress per unit macro strain
    for comp in range(6):
        e = np.zeros(6)
        e[comp] = 1.0
        eps = np.array([[e[0], e[3] / 2, e[5] / 2], [e[3] / 2, e[1], e[4] / 2], [e[5] / 2, e[4] / 2, e[2]]])
        u0 = (xc @ eps.T).reshape(ne, 24)
        f = np.zeros(3 * nn)
        np.add.at(f, edof, -(u0 @ ke.T))
        uf, info = cg(kf, f[free], rtol=1e-9, maxiter=3000, M=ml.aspreconditioner())
        if info != 0:
            raise RuntimeError("unit cell solve did not converge")
        u = np.zeros(3 * nn)
        u[free] = uf
        ue = u[edof] + u0
        C[:, comp] = ((ue @ bc.T) @ c0.T).sum(axis=0) * h ** 3 / float(np.prod(a))
        acc = np.zeros((nn, 6))
        for kk in range(8):
            np.add.at(acc, enode[:, kk], (ue @ bcorner[kk].T) @ c0.T)
        L[:, :, comp] = acc / count
    sym = symmetry_of(kind)
    if stretched and sym != "none":
        # a cell stretched along its axes keeps only the half turns about them
        sym = "orthorhombic"
    C = _symmetrize(C, sym, rotate_stiffness)
    S = np.linalg.inv(C)

    def peaks_on(Ls, dirs, rank_shift=0):
        out = np.empty(len(dirs))
        q = percentile / 100.0 * (nn - 1) - rank_shift      # the order statistic, numpy's "linear"
        for i in range(0, len(dirs), 16):
            eps = dirs[i:i + 16] @ S.T                    # (k, 6) macro strains
            st = np.einsum("nab,kb->kna", Ls, eps.astype(np.float32))
            for k, s_ in enumerate(st):
                v = von_mises(s_)
                lo = int(np.floor(q))
                part = np.partition(v, [lo, min(lo + 1, len(v) - 1)])
                out[i + k] = part[lo] + (q - lo) * (part[min(lo + 1, len(v) - 1)] - part[lo])
        return out

    # The peak is an order statistic near the top: only nodes whose stress
    # could reach it matter. A node's von Mises per unit macro stress is at
    # most the root of the largest eigenvalue of M' V M (M its stress per
    # macro stress), so nodes whose bound is under a threshold every
    # direction's peak clears can be dropped, the rank shifted by as many.
    M = np.einsum("nab,bc->nac", L, S.astype(np.float32))
    bound = np.sqrt(np.linalg.eigvalsh(np.einsum("nba,bc,ncd->nad", M, VM_FORM.astype(np.float32), M))[:, -1])
    ref = _directions(n_random=0)
    thr = 0.5 * float(peaks_on(L, ref).min())
    hot = bound >= thr
    Lh = L[hot]
    shift = int(nn - hot.sum())

    def peaks_of(dirs):
        out = peaks_on(Lh, dirs, shift)
        bad = out < thr
        if bad.any():
            out[bad] = peaks_on(L, dirs[bad])
        return out

    base = _directions()
    peaks = peaks_of(base)
    # the cell's symmetry images of every sample read the same peak
    if sym == "none":
        dirs, pk = base, peaks
    else:
        t = stress_rotation(_group(sym))                  # (g, 6, 6)
        dirs = np.einsum("gab,kb->gka", t, base).reshape(-1, 6)
        pk = np.tile(peaks, len(t))
    P, quad_worst = _fit_quadratic(dirs, pk, sym)
    Q4, q4_scale = _fit_quartic(dirs, pk)
    floor = 1.0 / quad_worst
    # held-out directions, solved directly: the model against the cell
    hold = _directions(seed=11, n_random=200)[-200:]
    ph = peaks_of(hold)
    ratio = stress_surface(hold, P, Q4, floor) / ph
    ratio_quad = np.sqrt(np.einsum("ka,ab,kb->k", hold, P, hold)) / ph
    e100 = 1.0 / S[0, 0]
    d111 = np.full(3, 1 / np.sqrt(3.0))
    return {"kind": kind, "d": float(wall_over_cell), "n": n, "nu": nu, "rho": float(solid.mean()),
            "aspect": a.tolist() if stretched else None,
            "C": C, "P": P, "Q4": Q4, "floor": floor, "symmetry": sym,
            "holdout": {"min": float(ratio.min()), "mean": float(ratio.mean()), "max": float(ratio.max())},
            "holdoutQuadratic": {"min": float(ratio_quad.min()), "mean": float(ratio_quad.mean()),
                                 "max": float(ratio_quad.max())},
            "E100": float(e100), "E111": float(directional_modulus(C, d111)),
            "G": float(C[3, 3]), "directions": base, "peaks": peaks, **({"L": L, "S": S} if _keep else {})}


def _fit_quadratic(dirs, pk, sym):
    """peak^2 = s' P s by least squares on the relative error (a cell's
    strong directions count as much as its weak ones), positive definite,
    scaled to lie on or above every sample. Returns (P, worst over-read)."""
    iu = np.triu_indices(6)
    w = np.where(iu[0] == iu[1], 1.0, 2.0)
    A = dirs[:, iu[0]] * dirs[:, iu[1]] * w
    wt = 1.0 / pk ** 2
    p, *_ = np.linalg.lstsq(A * wt[:, None], pk ** 2 * wt, rcond=None)
    P = np.zeros((6, 6))
    P[iu] = p
    P = P + np.triu(P, 1).T
    P = _symmetrize(P, sym, rotate_stress_form)
    wv, vv = np.linalg.eigh(P)
    P = (vv * np.maximum(wv, 1e-6 * wv.max())) @ vv.T
    fit = np.sqrt(np.einsum("ka,ab,kb->k", dirs, P, dirs))
    scale = float(np.max(pk / fit))
    return P * scale ** 2, float(np.max(fit * scale / pk))


MONOMIALS4 = np.array(list(itertools.combinations_with_replacement(range(6), 4)))   # (126, 4)


def quartic_features(s):
    """(k, 126) the degree-4 monomials of Voigt stresses s (k, 6)."""
    s = np.asarray(s, float)
    return s[:, MONOMIALS4[:, 0]] * s[:, MONOMIALS4[:, 1]] * s[:, MONOMIALS4[:, 2]] * s[:, MONOMIALS4[:, 3]]


def square_of_quadratic(P) -> np.ndarray:
    """The quartic coefficients (126,) of (s' P s)^2."""
    coef = {}
    for a in range(6):
        for b in range(6):
            for c in range(6):
                for d in range(6):
                    key = tuple(sorted((a, b, c, d)))
                    coef[key] = coef.get(key, 0.0) + P[a, b] * P[c, d]
    return np.array([coef[tuple(m)] for m in MONOMIALS4])


def _fit_quartic(dirs, pk):
    """peak^4 = a quartic polynomial of the macro stress (126 terms), least
    squares on the relative error, scaled onto or above every sample."""
    A = quartic_features(dirs)
    wt = 1.0 / pk ** 4
    q, *_ = np.linalg.lstsq(A * wt[:, None], pk ** 4 * wt, rcond=None)
    fit = np.maximum(A @ q, 1e-30) ** 0.25
    scale = float(np.max(pk / fit))
    return q * scale ** 4, scale


def stress_surface(s, P, Q4, floor):
    """Peak wall stress for macro stresses s (k, 6) in the cell's axes: the
    quartic surface, never under `floor` times the quadratic one (positive
    definite everywhere, so a direction the quartic dips in between samples
    still reads). P, Q4, floor may be per row ((k, 6, 6), (k, 126), (k,))."""
    s = np.asarray(s, float)
    quad = np.sqrt(np.maximum(np.einsum("ka,ab,kb->k", s, P, s) if np.ndim(P) == 2
                              else np.einsum("ka,kab,kb->k", s, P, s), 0.0))
    f = quartic_features(s)
    q4 = (f @ Q4) if np.ndim(Q4) == 1 else np.einsum("km,km->k", f, Q4)
    return np.maximum(np.maximum(q4, 0.0) ** 0.25, np.asarray(floor) * quad)


def directional_modulus(C, d) -> float:
    """Young's modulus along unit direction d of stiffness C."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    s = np.outer(d, d)
    sig = np.array([s[i, j] for i, j in _VOIGT])
    return float(1.0 / (sig @ np.linalg.solve(C, sig)))


# ---- tables -------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _tables() -> dict:
    import os

    path = Path(os.environ.get("LOADPATH_LATTICE_TABLES") or TABLE_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _key(kind: str) -> str:
    from .struts import canonical, is_strut

    return canonical(kind) if is_strut(kind) else kind


def _aspect_key(aspect):
    if aspect is None:
        return None
    a = tuple(round(float(x), 3) for x in aspect)
    return None if np.allclose(a, 1.0) else a


def has_tensors(kind: str, aspect=None) -> bool:
    """Tables exist (cubic cells), or can be built (a stretched strut or
    sheet cell; not a Voronoi foam)."""
    from .struts import canonical

    if _aspect_key(aspect) is not None:
        return canonical(kind) not in ("voronoi", "delaunay") and _key(kind) in _tables().get("kinds", {})
    return _key(kind) in _tables().get("kinds", {})


# stretched cells: a short ladder solved on first use (about a minute and a
# half a cell type and aspect on 4 cores), cached on disk
ASPECT_RHO = (0.06, 0.12, 0.22, 0.36, 0.55, 0.8)
ASPECT_N = 32


def _cache_dir() -> Path:
    import os

    return Path(os.environ.get("LOADPATH_CACHE") or Path.home() / ".cache" / "loadpath") / "lattice-tensors"


@lru_cache(maxsize=None)
def aspect_table(kind: str, aspect: tuple) -> list:
    """Rows (rho, d, C, P, Q4, floor) of a stretched cell, from the disk
    cache or solved now (unit_cell_tensors with `aspect`). `d` is the wall
    or strut diameter over the cell: a stretched cell's density at a given
    wall is not the cubic cell's, so it is read from these rows too."""
    from .lattice import SHEETS, wall_for_density
    from .struts import strut_diameter_for_density

    k = _key(kind)
    path = _cache_dir() / f"{k}-{aspect[0]:g}x{aspect[1]:g}x{aspect[2]:g}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    rows = []
    for rho in ASPECT_RHO:
        d = float(wall_for_density(k, rho, 1.0)) if k in SHEETS else float(strut_diameter_for_density(k, rho, 1.0))
        r = unit_cell_tensors(k, d, n=ASPECT_N, aspect=aspect)
        rows.append({"rho": r["rho"], "d": d, "C": r["C"].tolist(), "P": r["P"].tolist(),
                     "Q4": r["Q4"].tolist(), "floor": r["floor"], "holdout": r["holdout"]})
    rows.sort(key=lambda x: x["rho"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    except OSError:
        pass
    return rows


def aspect_density(kind: str, aspect, d_over_cell):
    """Relative density of a stretched cell at wall (or strut diameter)
    over cell `d_over_cell`, from its rows (log-log, to the solid)."""
    rows = aspect_table(kind, _aspect_key(aspect))
    d = np.array([r["d"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    x = np.log(np.clip(np.asarray(d_over_cell, float), 1e-4, None))
    lx, ly = np.log(d), np.log(rho)
    slope = (ly[1] - ly[0]) / (lx[1] - lx[0])
    hi = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    out = np.where(x < lx[0], ly[0] + slope * (x - lx[0]),
                   np.where(x > lx[-1], ly[-1] + hi * (x - lx[-1]), np.interp(x, lx, ly)))
    return np.clip(np.exp(out), 1e-3, 1.0)


@lru_cache(maxsize=None)
def _rows(kind: str, aspect=None):
    t = aspect_table(kind, aspect) if aspect is not None else _tables()["kinds"][_key(kind)]
    nu = float(_tables().get("nu", 0.35))
    from .fem import elastic_c

    rho = np.array([r["rho"] for r in t] + [1.0])
    C = np.array([np.array(r["C"]) for r in t] + [elastic_c(1.0, nu)])
    P = np.array([np.array(r["P"]) for r in t] + [VM_FORM])
    Q = np.array([np.array(r["Q4"]) for r in t] + [square_of_quadratic(VM_FORM)])
    F = np.array([r["floor"] for r in t] + [1.0])
    return rho, C, P, Q, F


def _interp_at(rho, xs):
    """(log-log leading-entry interpolant, row j, weight t) at rho: under
    the first row the leading entry runs on with the first two rows' slope
    and the shape is held; between rows the shape is linear in log rho."""
    lx = np.log(xs)
    lr = np.log(np.clip(rho, 1e-3, 1.0))
    j = np.clip(np.searchsorted(lx, lr) - 1, 0, len(lx) - 2)
    t = np.clip((lr - lx[j]) / (lx[j + 1] - lx[j]), 0.0, 1.0)

    def lead(v):
        lv = np.log(v)
        slope = (lv[1] - lv[0]) / (lx[1] - lx[0])
        return np.exp(np.where(lr < lx[0], lv[0] + slope * (lr - lx[0]), np.interp(lr, lx, lv)))

    return lead, j, t


def _interp(rho, xs, M, norm):
    lead, j, t = _interp_at(rho, xs)
    shape = M / norm.reshape((-1,) + (1,) * (M.ndim - 1))
    tt = t.reshape((-1,) + (1,) * (M.ndim - 1))
    return lead(norm).reshape((-1,) + (1,) * (M.ndim - 1)) * (shape[j] * (1 - tt) + shape[j + 1] * tt)


def tensors_at(kind: str, rho, aspect=None):
    """(C, P, Q4, floor) of a cell type at relative densities rho, in the
    cell's own axes: C (k, 6, 6) over the solid's E (engineering shears);
    the wall-stress surface (stress_surface) as its quadratic form P (k, 6,
    6), quartic Q4 (k, 126) and floor (k,), for the macro stress the lattice
    carries (Voigt, tensor shears). `aspect`: a stretched cell (its own
    tables, aspect_table)."""
    if not has_tensors(kind, aspect):
        raise ValueError(f"no anisotropic tensors for lattice {kind!r}; run bench/lattice_tensors.py")
    rho = np.atleast_1d(np.asarray(rho, float))
    xs, C, P, Q, F = _rows(kind, _aspect_key(aspect))
    _, j, t = _interp_at(rho, xs)
    c, pp, q = _interp(rho, xs, C, C[:, 0, 0]), _interp(rho, xs, P, P[:, 0, 0]), _interp(rho, xs, Q, P[:, 0, 0] ** 2)
    low = rho < xs[0]
    if low.any():
        # under the table each eigen-mode runs on with its own power law
        # (matrix log linear in log rho, through the first two rows): a strut
        # cell's bending modes (shear for the cubic) fall about as rho^2.3
        # while its stretching modes fall as rho, so holding the shape of the
        # first row read the shear 3x too stiff at rho 0.02
        u = (np.log(np.clip(rho[low], 1e-3, 1.0)) - np.log(xs[0])) / (np.log(xs[1]) - np.log(xs[0]))
        c_low, p_low = _spd_powerlaw(C[0], C[1], u), _spd_powerlaw(P[0], P[1], u)
        q[low] = q[low] * (p_low[:, 0, 0] / pp[low][:, 0, 0])[:, None] ** 2
        c[low], pp[low] = c_low, p_low
    return c, pp, q, F[j] * (1 - t) + F[j + 1] * t


def _spd_logm(A):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return (V * np.log(np.maximum(w, 1e-300 + 1e-12 * w.max()))) @ V.T


def _spd_powerlaw(A0, A1, u):
    """expm(log A0 + u (log A1 - log A0)) for each u: symmetric positive
    definite, each eigen-direction on its own power law."""
    L0, dL = _spd_logm(A0), _spd_logm(A1) - _spd_logm(A0)
    out = np.empty((len(u),) + A0.shape)
    for i, ui in enumerate(np.atleast_1d(u)):
        w, V = np.linalg.eigh(L0 + ui * dL)
        out[i] = (V * np.exp(w)) @ V.T
    return out
