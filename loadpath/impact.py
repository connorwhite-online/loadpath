"""
Crush and impact: what a lattice is for when stiffness per gram is not the goal.

A solid part stops a falling or swinging mass like a spring: the force rises
with the deflection until the part has stored the whole kinetic energy, and
a stiff part (which is what topology optimization makes) has to push back
very hard to do that in a fraction of a millimetre. Then it hands the energy
back as a rebound. A cellular core crushes at a nearly constant stress (the
plateau) over most of its height, so it absorbs the same energy at a force
the designer picks, and keeps it (the walls yield or buckle). That is the
job of a crash box, a hard stop, a bumper or a drop pad.

This module predicts that behaviour for TPMS sheet lattices graded by a
field, and grades one for a given impact:

- `collapse_strength`: plastic collapse stress of one periodic unit cell
  under uniaxial compression, from the elastic compensation method
  (Mackenzie & Boyle 1993): repeated linear solves in which every cell wall
  above the yield stress is softened in proportion, so the load spreads
  the way a plastic mechanism spreads it. The lower bound P sigma_y /
  max sigma it gives rises with each pass to near the limit load. The fits
  in CRUSH (sigma_pl / sigma_y = D rho^p) come from it.
- `crush_curve`: force against crush distance for a part along one axis,
  as a stack of thin layers in series. Each layer's crush force is the sum
  of its cells' plateau stress times their area (cells in parallel), and
  the weakest layer crushes first, through its densification strain
  (Gibson & Ashby: eps_D = 1 - 1.4 rho), before the next one starts. This
  is the standard model for graded foams; it ignores lateral load spreading
  between layers, which `limit_load` checks in 3D.
- `impact_response`: where on that curve a mass m at speed v stops: crush
  distance, peak force, peak deceleration, and whether the core bottoms out.
- `grade_for_impact`: the density per layer that crushes every layer at the
  same force (sigma_pl(rho(z)) A(z) = F), the lowest such force that stops
  the mass within the core, which is the flattest, gentlest absorber that
  fits the envelope. A tapered envelope makes a uniform core's force rise
  as the crush front reaches wider layers; grading takes that out.
- `solid_impact`: the same impact on a solid part from its linear stiffness
  and peak stress (what a topology-optimized part does: elastic, so the
  peak force is sqrt(2 E k), and it rebounds).
- `limit_load`: the elastic compensation method on a 3D voxel model with
  per-cell modulus and strength (a homogenized graded lattice), for the
  first collapse load without the layer model's assumptions.

Assumptions worth knowing. The plateau is taken equal to the plastic
collapse stress (right for ductile polymers such as PA12, PA11, PETG; PLA
crushes by fracture and is ragged; TPU buckles elastically and recovers,
which this does not model). Densification uses Gibson & Ashby's foam
constant 1.4, not a TPMS measurement. Rate effects are ignored (polymers
are 10-30% stronger at impact rates, which raises the force a little and
is on the safe side for stroke, not for force). Treat predictions as
+-25% until a printed crush coupon calibrates D (see CRUSH).

Units: mm, N, MPa; energy in N mm (= mJ), mass in kg, speed in m/s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import relative_density, sheet_modulus, tpms

__all__ = ["CRUSH", "CRUSH_DATA", "crush_law", "ELASTOMERS", "HYSTERESIS", "collapse_strength", "plateau_stress", "density_for_plateau", "densification_strain",
           "CrushCurve", "crush_curve", "Columns", "contact_columns", "contact_curve", "grade_for_contact", "series_impact", "layer_profile", "impact_response", "grade_for_impact", "solid_impact",
           "limit_load", "foam_stress", "face_load", "grade_3d", "save_density_grid", "impact_report", "sample_core", "DENSIFICATION_ALPHA"]

DENSIFICATION_ALPHA = 1.4    # Gibson & Ashby, open-cell foams: eps_D = 1 - 1.4 rho

# Plastic collapse under uniaxial compression, sigma_pl / sigma_y = D rho^p,
# from `collapse_strength` (elastic compensation lower bound, 32^3 voxels
# per cell, nu 0.39, 15 passes) at rho = 0.11 .. 0.51, fit residuals under
# 6%. The collapse stress is 1.4-2x the first-yield stress (the walls share
# load once the hottest ones yield). Exponents near 1.1-1.4 say the sheets
# carry load mostly by stretching, which is why they beat strut lattices
# (bending, p ~ 1.5-2) per gram. Per gram, diamond is the strongest.
# Perfect cells: no buckling, no print defects, quasi-static; a crush
# coupon is what pins D down for a real machine (docs/IMPACT.md).
CRUSH = {
    "gyroid": {"D": 0.633, "p": 1.191},
    "diamond": {"D": 0.675, "p": 1.122},
    "schwarz-p": {"D": 0.795, "p": 1.380},
}

# Elastomers (TPU) do not yield: their sheets buckle elastically, the core
# crushes at the buckling stress and springs back. Plateau sigma_el / E =
# D rho^p, Gibson & Ashby's elastic-collapse law for elastomeric foams
# (0.05 rho^2). Our own check, linear buckling of a 2x2x2 gyroid block
# between glued plates (nonlinear.linear_buckling), gives 0.0043 at rho 0.3
# with 24 voxels per cell against the law's 0.0045, but is not converged
# (0.0144 at 16 voxels: thin walls lock), so the law stands until a
# coupon says otherwise. Same law for every sheet type (no per-type data).
for _k in ("gyroid", "diamond", "schwarz-p"):
    CRUSH[f"{_k}:elastic"] = {"D": 0.05, "p": 2.0}
ELASTOMERS = {"tpu"}
# fraction of the absorbed energy an elastomer core dissipates (the rest
# comes back as rebound): large-strain compression hysteresis of TPU
# foams and lattices, roughly 0.2-0.4; rough
HYSTERESIS = {"tpu": 0.3}

# (rho, sigma_pl / sigma_y) behind the fits, for tests and re-fitting
CRUSH_DATA = {
    "gyroid": [(0.123, 0.0542), (0.206, 0.0927), (0.303, 0.1479), (0.408, 0.2121), (0.509, 0.2987)],
    "diamond": [(0.111, 0.0580), (0.199, 0.1098), (0.289, 0.1656), (0.412, 0.2509), (0.512, 0.3202)],
    "schwarz-p": [(0.113, 0.0397), (0.198, 0.0826), (0.307, 0.1568), (0.399, 0.2267), (0.498, 0.3024)],
}


# ---- unit cell ---------------------------------------------------------------------

def collapse_strength(kind: str, wall_over_cell: float, n: int = 32, nu: float = 0.39, iters: int = 15,
                      axis: int = 2) -> dict:
    """Plastic collapse stress (over the solid's yield stress) of a TPMS sheet
    unit cell under uniaxial macro compression along `axis`, lateral faces
    free on average, by the elastic compensation method on an n^3 periodic
    voxel cell.

    Each pass solves the cell for a unit macro stress, takes the lower bound
    1 / max(von Mises) (the macro stress at which the most loaded wall
    yields, with the moduli of this pass) and softens every element by
    sigma_ref / sigma_e, sigma_ref the mean wall stress, so overloaded walls
    shed load to idle ones as they would after yielding. Returns rho, the
    first-yield stress (pass 1, the elastic answer), the collapse estimate
    (best lower bound over the passes) and the history."""
    import pyamg
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg

    from .fem import CORNERS, _b_matrix, elastic_c, hex8_ke, von_mises

    h = 1.0 / n
    c = (np.arange(n) + 0.5) * h
    g = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    solid = np.abs(tpms(kind, g, 1.0)) < 0.5 * wall_over_cell
    eijk = np.argwhere(solid.reshape(n, n, n))
    ne = len(eijk)
    corner = (eijk[:, None, :] + CORNERS[None]) % n
    _, inv = np.unique((corner[..., 0] * n + corner[..., 1]) * n + corner[..., 2], return_inverse=True)
    enode = inv.reshape(ne, 8)
    nn = int(enode.max()) + 1
    edof = (3 * enode[:, :, None] + np.arange(3)).reshape(ne, 24)
    ke = hex8_ke(nu) * h
    rows = np.repeat(edof, 24, axis=1).ravel()
    cols = np.tile(edof, (1, 24)).ravel()
    free = np.ones(3 * nn, bool)
    free[:3] = False
    xc = (eijk[:, None, :] + CORNERS[None]) * h
    c0 = elastic_c(1.0, nu)
    bc = _b_matrix(0, 0, 0) / h
    scale = np.ones(ne)

    def solve(eps_list):
        k = sp.csr_matrix((np.repeat(scale, 576) * np.tile(ke.ravel(), ne), (rows, cols)), shape=(3 * nn, 3 * nn))
        kf = k[free][:, free]
        ml = pyamg.smoothed_aggregation_solver(kf, symmetry="hermitian", max_coarse=400, coarse_solver="splu")
        out = []
        for e in eps_list:
            eps = np.array([[e[0], e[3] / 2, e[5] / 2], [e[3] / 2, e[1], e[4] / 2], [e[5] / 2, e[4] / 2, e[2]]])
            u0 = (xc @ eps.T).reshape(ne, 24)
            f = np.zeros(3 * nn)
            np.add.at(f, edof, -scale[:, None] * (u0 @ ke.T))
            uf, info = cg(kf, f[free], rtol=1e-8, maxiter=3000, M=ml.aspreconditioner())
            if info != 0:
                raise RuntimeError("unit cell solve did not converge")
            u = np.zeros(3 * nn)
            u[free] = uf
            sig = ((u[edof] + u0) @ bc.T) @ c0.T * scale[:, None]     # element-center stress
            out.append((sig.sum(axis=0) / n ** 3, sig))
        return out

    unit = np.eye(6)
    hist = []
    best = 0.0
    first = None
    for _ in range(iters):
        res = solve([unit[0], unit[1], unit[2]])
        cm = np.stack([r[0][:3] for r in res], axis=1)            # 3x3 normal block of the macro stiffness
        # macro strains giving unit uniaxial stress along `axis`
        target = np.zeros(3)
        target[axis] = 1.0
        e = np.linalg.solve(cm, target)
        sig = sum(e[i] * res[i][1] for i in range(3))
        vm = von_mises(sig)
        lb = 1.0 / float(vm.max())
        if first is None:
            first = lb
        best = max(best, lb)
        hist.append(lb)
        ref = float(vm.mean())
        scale = scale * np.clip(ref / np.maximum(vm, 1e-12), 0.05, 20.0)
        scale = np.maximum(scale, 1e-6)
    return {"kind": kind, "rho": float(solid.mean()), "firstYield": float(first), "collapse": float(best),
            "history": [float(x) for x in hist]}


# ---- homogenized crush laws ------------------------------------------------------------

def _crush_fit(kind):
    fit = CRUSH.get(kind)
    if fit is None:
        raise ValueError(f"lattice type must be one of {sorted(CRUSH)}, got {kind!r}")
    return fit


def _base(kind: str) -> str:
    return kind.split(":")[0]


def crush_law(kind: str, material, calibrated: bool = True) -> tuple:
    """(law key, reference stress) for a lattice of `material`: plastic
    collapse over the yield stress for polymers and metals that yield, and
    elastic buckling over the modulus ("<kind>:elastic") for elastomers,
    which buckle, crush and spring back. A calibration card of that name
    with a crush block (calibration.fit on crush coupons) scales the
    reference by its measured factor."""
    name = getattr(material, "name", "")
    elastic = name in ELASTOMERS or str(getattr(material, "note", "")).startswith("elastomer")
    if not elastic:
        try:
            from .calibration import load_card

            card = load_card(name) or {}
            elastic = card.get("basedOn") in ELASTOMERS
        except Exception:  # noqa: BLE001 - a missing or broken card is no card
            card = {}
    law, ref = (f"{kind}:elastic", float(material.E)) if elastic else (kind, float(material.strength))
    if calibrated:
        from .calibration import crush_factor

        ref *= crush_factor(name, "elastic" if elastic else "plastic")
    return law, ref


def plateau_stress(kind: str, rho, sigma_y: float):
    """Crush plateau stress (MPa) of a `kind` sheet at relative density rho;
    solid (rho >= 1) is sigma_y. For an elastic law ("gyroid:elastic")
    sigma_y is the solid's modulus (see crush_law)."""
    f = _crush_fit(kind)
    rho = np.asarray(rho, float)
    return np.where(rho >= 1.0, sigma_y, sigma_y * f["D"] * np.clip(rho, 1e-3, 1.0) ** f["p"])


def density_for_plateau(kind: str, stress, sigma_y: float):
    """The relative density whose plateau is `stress` (MPa)."""
    f = _crush_fit(kind)
    return (np.asarray(stress, float) / (sigma_y * f["D"])) ** (1.0 / f["p"])


def densification_strain(rho, alpha: float = DENSIFICATION_ALPHA):
    """Nominal strain at which the crushed walls lock up (Gibson & Ashby)."""
    return np.clip(1.0 - alpha * np.asarray(rho, float), 0.0, 1.0)


# ---- the layer model ----------------------------------------------------------------------

@dataclass
class CrushCurve:
    """Force (N) against crush distance (mm), piecewise linear, plus the
    layers it came from. `stroke` is the distance at which every layer has
    densified (the core bottoms out); `energy(d)` the work absorbed to d."""
    d: np.ndarray
    f: np.ndarray
    stroke: float
    layer_force: np.ndarray
    layer_area: np.ndarray
    layer_rho: np.ndarray
    layer_stroke: np.ndarray

    def energy(self, d=None):
        e = np.concatenate([[0.0], np.cumsum(0.5 * (self.f[1:] + self.f[:-1]) * np.diff(self.d))])
        return e if d is None else np.interp(d, self.d, e)

    @property
    def capacity(self):
        return float(self.energy()[-1])

    def to_dict(self, points=60):
        dd = np.linspace(0.0, float(self.d[-1]), points)
        return {"stroke": round(self.stroke, 3), "capacity": round(self.capacity, 1),
                "distance": [round(float(x), 3) for x in dd],
                "force": [round(float(x), 1) for x in np.interp(dd, self.d, self.f)]}


def layer_profile(inside, rho, pitch: float, axis: int = 2):
    """Per layer along `axis` of a voxel grid: area (mm^2) and area-weighted
    mean relative density of the cells inside the part. inside: bool grid;
    rho: relative density per voxel (same shape, or a scalar)."""
    inside = np.asarray(inside, bool)
    rho = np.broadcast_to(np.asarray(rho, float), inside.shape)
    other = tuple(i for i in range(3) if i != axis)
    n = inside.sum(axis=other)
    area = n * pitch ** 2
    mean = np.where(n > 0, (rho * inside).sum(axis=other) / np.maximum(n, 1), 0.0)
    return area, mean


def crush_curve(kind: str, area, rho, h: float, sigma_y: float, modulus: float,
                alpha: float = DENSIFICATION_ALPHA, layer_force=None) -> CrushCurve:
    """Force-crush curve of a stack of layers of thickness h (mm) with the
    given area and density (one entry per layer; rho may be a scalar). The
    layers act in series: all carry the same force, the elastic part is the
    series compliance, and the layer with the lowest crush force collapses
    first and runs through its densification strain before the force rises
    to the next. `layer_force` (N per layer) overrides sigma_pl(rho) A when
    a layer's cells are not uniform (pass the sum over its cells)."""
    area = np.asarray(area, float)
    rho = np.broadcast_to(np.asarray(rho, float), area.shape).copy()
    keep = area > 0
    area, rho = area[keep], rho[keep]
    fl = plateau_stress(kind, rho, sigma_y) * area if layer_force is None else np.asarray(layer_force, float)[keep]
    stroke = densification_strain(rho, alpha) * h
    k_layer = modulus * sheet_modulus(_base(kind), rho) * area / h
    comp = float(np.sum(1.0 / np.maximum(k_layer, 1e-12)))      # series compliance, mm/N
    order = np.argsort(fl, kind="stable")
    d, f = [0.0], [0.0]
    crushed = 0.0
    for i in order:
        # elastic ramp to this layer's force (the uncrushed layers still spring)
        d.append(crushed + comp * fl[i])
        f.append(fl[i])
        crushed += stroke[i]
        d.append(crushed + comp * fl[i])
        f.append(fl[i])
    return CrushCurve(np.array(d), np.array(f), float(crushed + comp * fl.max()), fl, area, rho, stroke)


# ---- contact: a curved face against a wall, a pole or a ball -------------------------------

@dataclass
class Columns:
    """A core as columns along the hit (contact_columns): per distinct
    column its cells' relative density from the struck face inwards (padded
    with nan), how far behind the first touch its face starts (`depth`,
    mm, the striker's own shape included), and the area it stands for."""
    rho: np.ndarray            # (n, cells)
    depth: np.ndarray          # (n,)
    area: np.ndarray           # (n,) mm^2
    pitch: float


def contact_columns(inside, rho, pitch: float, axis: int = 2, side: str = "+", striker: dict | None = None,
                    lo=None, field=None) -> Columns:
    """The voxel core cut into columns along `axis`, read from the struck
    side ("+": the striker comes from the high end of the axis). A column is
    the first unbroken run of cells behind its own face: what lies between
    the striker and whatever backs the core there (the anvil). `striker`:
    None or {"shape": "flat"} for a wall; {"shape": "sphere", "radius"} or
    {"shape": "cylinder", "radius", "along": axis} (a ball, a pole or a
    rounded edge), centred where the core is touched first, or `at`
    [.., ..] in the two other axes' coordinates (`lo`: the grid's corner).
    `field`: the envelope's own field at the cell centres (negative
    inside): each column's face is then put where the field crosses zero
    between its first cell and the one outside it, not on the cell's edge
    (a round face read on cell edges touches in steps, a whole ring of
    columns at a time). Columns with the same cells and depth are merged."""
    inside = np.asarray(inside, bool)
    rho = np.broadcast_to(np.asarray(rho, float), inside.shape)
    other = [i for i in range(3) if i != axis]
    ins = np.moveaxis(inside, axis, 2)
    rr = np.moveaxis(rho, axis, 2)
    if side == "+":
        ins, rr = ins[:, :, ::-1], rr[:, :, ::-1]
    elif side != "-":
        raise ValueError("striker.side is '+' or '-': the end of the axis the hit comes from")
    nz = ins.shape[2]
    has = ins.any(axis=2)
    if not has.any():
        raise ValueError("the envelope holds no cell on this grid")
    k0 = np.argmax(ins, axis=2)
    # the first unbroken run: its length is where the column first leaves the core
    after = ~ins & (np.arange(nz)[None, None, :] >= k0[:, :, None])
    k1 = np.where(after.any(axis=2), np.argmax(after, axis=2), nz)
    face = k0 * pitch
    if field is not None:
        ff = np.moveaxis(np.asarray(field, float), axis, 2)
        if side == "+":
            ff = ff[:, :, ::-1]
        ia_, ib_ = np.nonzero(has & (k0 > 0))
        f_in = ff[ia_, ib_, k0[ia_, ib_]]
        f_out = ff[ia_, ib_, k0[ia_, ib_] - 1]
        # from the first cell's centre out to the zero crossing, less the half cell already counted
        t = np.clip(f_in / np.minimum(f_in - f_out, -1e-12), 0.0, 1.0)
        face = face.astype(float)
        face[ia_, ib_] += (0.5 - t) * pitch
    depth = face - face[has].min()
    st = striker or {}
    shape = st.get("shape", "flat")
    if shape != "flat":
        r = float(st.get("radius", 0.0))
        if r <= 0:
            raise ValueError("striker.radius (mm) is needed for a sphere or a cylinder")
        ua = (np.arange(ins.shape[0]) + 0.5) * pitch
        ub = (np.arange(ins.shape[1]) + 0.5) * pitch
        ga, gb = np.meshgrid(ua, ub, indexing="ij")
        if st.get("at") is not None:
            base = np.zeros(3) if lo is None else np.asarray(lo, float)
            ca, cb = float(st["at"][0]) - base[other[0]], float(st["at"][1]) - base[other[1]]
        else:
            first = has & (depth <= depth[has].min() + 1e-9)
            ca, cb = float(ga[first].mean()), float(gb[first].mean())
        if shape == "sphere":
            d2 = (ga - ca) ** 2 + (gb - cb) ** 2
        elif shape == "cylinder":
            along = "xyz".index(st.get("along", "xyz"[other[1]]))
            if along == axis:
                raise ValueError("striker.along is the cylinder's axis, across the hit: not the crush axis")
            d2 = (ga - ca) ** 2 if along == other[1] else (gb - cb) ** 2
        else:
            raise ValueError("striker.shape is flat, sphere or cylinder")
        # how far the striker's face has fallen away from its tip there (never touching past its radius)
        gap = np.where(d2 < r * r, r - np.sqrt(np.maximum(r * r - d2, 0.0)), np.inf)
        depth = depth + gap
        depth = depth - depth[has].min()
    has &= np.isfinite(depth)
    ia, ib = np.nonzero(has)
    n = (k1 - k0)[ia, ib]
    cells = np.full((len(ia), int(n.max())), np.nan)
    for j in range(cells.shape[1]):
        m = n > j
        cells[m, j] = rr[ia[m], ib[m], k0[ia[m], ib[m]] + j]
    key = np.column_stack([np.round(depth[ia, ib], 6), np.round(np.nan_to_num(cells, nan=-1.0), 5)])
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return Columns(cells[idx], depth[ia, ib][idx], cnt * pitch ** 2, float(pitch))


def contact_curve(kind: str, cols: Columns, sigma_y: float, modulus: float,
                  alpha: float = DENSIFICATION_ALPHA, points: int = 160, rho=None) -> CrushCurve:
    """Force against the striker's travel for a core it meets on a curved
    face, or with a curved face of its own (a round tyre against a wall, a
    pad against a pole): columns in parallel, cells in series in each.

    A crushing foam does not spread load sideways (its plastic Poisson's
    ratio is about nil), so each column along the hit is squeezed by what
    the striker has travelled past its own face, w = d - depth, and no more:
    material the striker has not reached carries nothing. Within a column
    every cell carries the column's stress; the cell with the lowest plateau
    crushes first, through its densification strain, then the next (the
    layer model, per column), with the elastic give of the whole column in
    series. Past its last cell's stroke a column is solid: its stress climbs
    with the solid's modulus over the solid height left. The force is the
    sum over the columns touched. For a flat-ended prism this is the layer
    model exactly; for a round face the force grows with the contact patch,
    sigma_pl A(d), where the layer model ran the crush front ahead of the
    striker by 1 / eps_D and read the force that much early.

    `rho` replaces the columns' own densities (same shape). `stroke` is
    where the first column has every cell locked; the curve runs on past it
    (that column a solid, the rest still crushing) to where it could give no
    more. layer_area holds the contact area and layer_stroke the share of it
    locked, at each d."""
    r = cols.rho if rho is None else np.asarray(rho, float)
    live = np.isfinite(r)
    rz = np.where(live, r, 1.0)
    h = cols.pitch
    sig = np.where(live, plateau_stress(kind, rz, sigma_y), np.inf)
    stroke = np.where(live, densification_strain(rz, alpha) * h, 0.0)
    comp = np.where(live, h / np.maximum(modulus * sheet_modulus(_base(kind), rz), 1e-12), 0.0).sum(axis=1)
    solid = np.where(live, h - stroke, 0.0).sum(axis=1)                  # height once every cell has locked
    length = live.sum(axis=1) * h
    order = np.argsort(sig, axis=1, kind="stable")
    sig = np.take_along_axis(sig, order, axis=1)
    stroke = np.take_along_axis(stroke, order, axis=1)
    before = np.cumsum(stroke, axis=1) - stroke                         # crushed before each cell starts
    total = stroke.sum(axis=1)
    top = np.where(np.isfinite(sig), sig, -np.inf).max(axis=1)
    # the striker stops at the first column squeezed down to its solid
    d_end = float(np.min(cols.depth + total + comp * top + 0.98 * np.maximum(solid - comp * top, 0.0)))
    d_end = min(d_end, float(np.min(cols.depth + 0.999 * length)))
    dd = np.unique(np.concatenate([np.linspace(0.0, d_end, points), np.minimum(cols.depth, d_end)]))
    f = np.zeros(len(dd))
    touch = np.zeros(len(dd))
    dense = np.zeros(len(dd))
    for i, d in enumerate(dd):
        w = d - cols.depth
        on = w > 0
        if not on.any():
            continue
        ww = w[on]
        cand = np.minimum(sig[on], (ww[:, None] - before[on]) / np.maximum(comp[on], 1e-12)[:, None])
        st = np.maximum(cand.max(axis=1), 0.0)
        over = np.maximum(ww - total[on] - comp[on] * top[on], 0.0)
        st = st + modulus * over / np.maximum(solid[on], 1e-9)
        f[i] = float(st @ cols.area[on])
        touch[i] = float(cols.area[on].sum())
        dense[i] = float(cols.area[on][over > 0].sum()) / max(touch[i], 1e-12)
    lock = float(np.min(cols.depth + total + comp * top))             # the first column down to solid
    return CrushCurve(dd, f, min(lock, d_end), f, touch, np.zeros(len(dd)), dense)


def grade_for_contact(kind: str, layers: Columns, n_layers: int, sigma_y: float, modulus: float, energy: float,
                      force_limit: float | None = None, rho_min: float = 0.12, rho_max: float = 0.5,
                      reserve: float = 1.25, alpha: float = DENSIFICATION_ALPHA, steps: int = 9,
                      candidates=None) -> dict:
    """Density per layer along the hit for the gentlest stop of `energy`
    (N mm) on a curved contact (contact_curve). `layers`: the core's columns
    with each cell's layer index (from the struck end) in place of its
    density. With a contact patch that grows as the striker comes in, no
    grading makes the force flat (a column's weakest cell crushes first,
    so its stress only ever rises), and the layer rule (every layer at the
    same force) does not apply; what is left to choose is how soft the
    core is and how that changes from the struck face to the back. Every
    straight-line grade between rho_min and rho_max (`steps` each way; the
    uniform cores among them, then a finer pass round the best) is tried,
    with any `candidates` (density tables per layer, e.g. the layer rule's
    own, which is the answer when the contact is flat), and the one with the lowest peak
    force that stops `reserve` x the energy before its first column locks
    solid is kept. `ok` is false if none does, or its peak passes
    `force_limit`."""
    idx = np.where(np.isfinite(layers.rho), layers.rho, 0).astype(int)
    live = np.isfinite(layers.rho)
    used = np.unique(idx[live])
    z0, z1 = float(used.min()), float(max(used.max(), used.min() + 1))
    best, best_any = None, None

    def try_(table):
        nonlocal best, best_any
        table = np.clip(np.asarray(table, float), rho_min, rho_max)
        cv = contact_curve(kind, layers, sigma_y, modulus, alpha, rho=np.where(live, table[idx], np.nan))
        resp = impact_response(cv, energy)
        if resp["peakForce"] is None:
            return
        row = (resp["peakForce"], table, cv, resp)
        if best_any is None or row[0] < best_any[0]:
            best_any = row
        if cv.energy(cv.stroke) >= reserve * energy and (best is None or row[0] < best[0]):
            best = row

    def line(ra, rb):
        return ra + (rb - ra) * (np.arange(n_layers) - z0) / (z1 - z0)

    for t in candidates or ():
        try_(t)
    grid = np.linspace(rho_min, rho_max, steps)
    for ra in grid:
        for rb in grid:
            try_(line(ra, rb))
    # a second, finer pass round the best line found
    seed = best or best_any
    if seed is not None:
        t = seed[1]
        a0, b0 = float(t[int(z0)]), float(t[min(int(z1), n_layers - 1)])
        step = (rho_max - rho_min) / max(steps - 1, 1)
        for ra in np.linspace(a0 - step, a0 + step, 9):
            for rb in np.linspace(b0 - step, b0 + step, 9):
                try_(line(ra, rb))
    ok = best is not None
    peak, table, cv, resp = best if ok else (best_any or (None, np.full(n_layers, rho_max), None, None))
    if cv is None:
        cv = contact_curve(kind, layers, sigma_y, modulus, alpha, rho=np.where(live, rho_max, np.nan))
        resp = impact_response(cv, energy)
    if force_limit is not None and (resp["peakForce"] is None or resp["peakForce"] > force_limit):
        ok = False
    w = np.array([float((layers.area[:, None] * ((idx == k) & live)).sum()) for k in range(n_layers)])
    return {"ok": ok, "force": float(peak) if peak is not None else None, "rho": table, "curve": cv,
            "response": resp, "meanRho": float(np.sum(table * w) / max(np.sum(w), 1e-12)), "layerArea": w}


def series_impact(energy: float, compliance: float, distance=None, force=None, points: int = 4000) -> dict:
    """A crusher and the structure behind it as springs in series, stopping
    `energy` (N mm).

    A bumper's force-crush curve is measured against a rigid backing. On a
    part that gives, the same force also bends the part: the striker's
    travel is the crush plus the part's own deflection at the patch,
    compliance x F (mm/N: the part's f.u / F^2 under the hit, with inertia
    relief for a free body, so its own mass and what it carries take
    part). Quasi-static, the energy at crush d is

        E(d) = integral of F dd  +  1/2 compliance F(d)^2

    and the hit stops where that reaches `energy`. The peak force is the
    highest the curve reads on the way. A stiff part leaves the bumper its
    rigid-backing answer; a soft one takes most of the energy itself at a
    force far below the bumper's plateau (a 5.5 N/mm ring stops 6.2 J at
    261 N, never reaching an 800 N crush force).

    `distance`, `force`: the crusher's curve (mm, N; CrushCurve.d, .f), or
    None for a hit with nothing in between: the part alone, F = sqrt(2 E /
    compliance). Past the curve's end the crusher is solid and the part
    takes what is left as a spring (`bottomedOut`)."""
    c = max(float(compliance), 0.0)
    if distance is None or force is None or len(distance) < 2:
        if c <= 0:
            raise ValueError("a rigid part hit with no crusher has no finite force: give the crusher's curve")
        f = float(np.sqrt(2.0 * energy / c))
        return {"peakForce": f, "crush": 0.0, "deflection": c * f, "crusherEnergy": 0.0,
                "structureEnergy": float(energy), "structureShare": 1.0, "bottomedOut": False}
    d = np.asarray(distance, float)
    fc = np.maximum(np.asarray(force, float), 0.0)
    dd = np.unique(np.concatenate([np.linspace(d[0], d[-1], points), d]))
    ff = np.interp(dd, d, fc)
    ec = np.concatenate([[0.0], np.cumsum(0.5 * (ff[1:] + ff[:-1]) * np.diff(dd))])
    tot = ec + 0.5 * c * ff ** 2
    over = np.flatnonzero(tot >= energy)
    if len(over) == 0:
        # the crusher is used up: what is left goes into the part as a spring
        rest = float(energy - ec[-1])
        if c <= 0:
            return {"peakForce": None, "crush": float(dd[-1]), "deflection": 0.0, "crusherEnergy": float(ec[-1]),
                    "structureEnergy": 0.0, "structureShare": 0.0, "bottomedOut": True,
                    "residualEnergy": rest}
        f = float(max(np.sqrt(2.0 * rest / c), ff.max()))
        return {"peakForce": f, "crush": float(dd[-1]), "deflection": c * f, "crusherEnergy": float(ec[-1]),
                "structureEnergy": rest, "structureShare": rest / float(energy), "bottomedOut": True}
    k = int(over[0])
    if k == 0:
        t, dk, fk, ek = 0.0, float(dd[0]), float(ff[0]), 0.0
    else:
        t = float((energy - tot[k - 1]) / max(tot[k] - tot[k - 1], 1e-300))
        dk = float(dd[k - 1] + t * (dd[k] - dd[k - 1]))
        fk = float(ff[k - 1] + t * (ff[k] - ff[k - 1]))
        ek = float(ec[k - 1] + t * (ec[k] - ec[k - 1]))
    peak = float(max(ff[:k].max(initial=0.0), fk))
    es = float(energy - ek)
    return {"peakForce": peak, "crush": dk, "deflection": c * peak, "crusherEnergy": ek, "structureEnergy": es,
            "structureShare": es / float(energy) if energy > 0 else 0.0, "bottomedOut": False}


def release_energy(cap: float, compliance: float, distance=None, force=None, points: int = 4000) -> float:
    """What a crusher in series with the part has taken (N mm) when the force through them first reaches
    `cap`: the crush integral to that point plus the part's spring energy 1/2 compliance cap^2.

    A link that lets go at `cap` (a fuse, a slip clutch) stops the hit only if this is at least the hit's
    energy. Otherwise it breaks with the rest still moving, and that rest arrives at the next contact
    (energy is not lost by breaking a pin: a brittle shear absorbs only a few mJ)."""
    c = max(float(compliance), 0.0)
    e = 0.5 * c * float(cap) ** 2
    if distance is None or force is None or len(distance) < 2:
        return e
    d = np.asarray(distance, float)
    fc = np.maximum(np.asarray(force, float), 0.0)
    dd = np.unique(np.concatenate([np.linspace(d[0], d[-1], points), d]))
    ff = np.interp(dd, d, fc)
    ec = np.concatenate([[0.0], np.cumsum(0.5 * (ff[1:] + ff[:-1]) * np.diff(dd))])
    over = np.flatnonzero(ff >= cap)
    if len(over) == 0:
        return float(ec[-1]) + e          # the crusher never reaches the cap before it is used up
    k = int(over[0])
    if k == 0:
        return e
    t = (cap - ff[k - 1]) / max(ff[k] - ff[k - 1], 1e-300)
    return float(ec[k - 1] + t * (ec[k] - ec[k - 1])) + e


def impact_response(curve: CrushCurve, energy: float, mass: float | None = None) -> dict:
    """Where a mass carrying `energy` (N mm) stops on the curve: crush
    distance, peak force and (with `mass`, kg) peak deceleration in g. If
    the core cannot take it all before densifying, `bottomedOut` is true and
    `residualEnergy` is what is left for the densified core and whatever
    is behind it (the force then is far above the plateau)."""
    e = curve.energy()
    out = {"energy": round(float(energy), 1), "capacity": round(float(e[-1]), 1)}
    if energy > e[-1]:
        out.update(bottomedOut=True, residualEnergy=round(float(energy - e[-1]), 1),
                   crush=round(float(curve.d[-1]), 3), peakForce=None, usedStroke=1.0)
        return out
    # f is piecewise linear in d, so within a segment the energy is
    # quadratic in d: solve it there (a straight line between the corners
    # reads a stiff core that stops on its first ramp far too low)
    k = int(np.clip(np.searchsorted(e, energy, side="right") - 1, 0, len(e) - 2))
    d0, d1 = float(curve.d[k]), float(curve.d[k + 1])
    f0, f1 = float(curve.f[k]), float(curve.f[k + 1])
    rest = max(float(energy - e[k]), 0.0)
    s = (f1 - f0) / max(d1 - d0, 1e-12)
    if abs(s) < 1e-12:
        x = rest / max(f0, 1e-12)
    else:
        x = (-f0 + np.sqrt(max(f0 * f0 + 2 * s * rest, 0.0))) / s
    d = float(min(d0 + max(x, 0.0), d1))
    j = int(np.searchsorted(curve.d, d, side="right"))
    fpk = float(max(curve.f[:j].max(initial=0.0), np.interp(d, curve.d, curve.f)))
    out.update(bottomedOut=False, crush=round(d, 3), peakForce=round(fpk, 1),
               usedStroke=round(d / max(curve.stroke, 1e-9), 3),
               # energy over (peak force x crush): 1 is an ideal square pulse
               efficiency=round(float(energy / max(fpk * d, 1e-12)), 3))
    if mass:
        out["peakG"] = round(fpk / (mass * 9.81), 1)
    return out


def grade_for_impact(kind: str, area, h: float, sigma_y: float, modulus: float, energy: float,
                     force_limit: float | None = None, rho_min: float = 0.12, rho_max: float = 0.5,
                     reserve: float = 1.25, alpha: float = DENSIFICATION_ALPHA) -> dict:
    """Density per layer for the gentlest stop of `energy` (N mm) in this
    envelope: every layer crushes at the same force F (rho_i from
    sigma_pl(rho_i) A_i = F, clipped to the printable range), and F is the
    lowest for which the core can absorb `reserve` x the energy before it
    densifies. Returns the force, the densities, and the curve, with
    `ok` false if no F up to `force_limit` does it."""
    area = np.asarray(area, float)
    a = area[area > 0]
    lo = float(plateau_stress(kind, rho_min, sigma_y) * a.min())
    hi = float(plateau_stress(kind, rho_max, sigma_y) * a.max())

    def at(force):
        rho = np.clip(density_for_plateau(kind, force / a, sigma_y), rho_min, rho_max)
        return rho, crush_curve(kind, a, rho, h, sigma_y, modulus, alpha)

    # the target force with the lowest peak (layers clipped at rho_min crush
    # above it, so the lowest target is not always the gentlest) among those
    # that keep `reserve` x the energy in hand before densifying
    best, peak = None, np.inf
    for force in np.geomspace(lo, hi, 300):
        rho, cv = at(force)
        if cv.capacity < reserve * energy:
            continue
        p = impact_response(cv, energy)["peakForce"]
        if p < peak:
            best, peak = (force, rho, cv), p
    ok = best is not None
    if best is None:                               # take the most capacity there is
        best = max(((f, *at(f)) for f in np.geomspace(lo, hi, 200)), key=lambda t: t[2].capacity)
    force, rho, cv = best
    resp = impact_response(cv, energy)
    if force_limit is not None and (resp["peakForce"] is None or resp["peakForce"] > force_limit):
        ok = False
    full = np.zeros_like(area)
    full[area > 0] = rho
    return {"ok": ok, "force": float(force), "rho": full, "curve": cv, "response": resp,
            "meanRho": float(np.sum(rho * a) / np.sum(a))}


def solid_impact(stiffness: float, stress_per_newton: float, sigma_y: float, energy: float,
                 force_limit: float | None = None, mass: float | None = None) -> dict:
    """The same impact on a solid (elastic) part with stiffness k (N/mm) at the
    impact point and peak von Mises stress s (MPa per N of load): it stops
    the mass at F = sqrt(2 E k) after F/k mm, stores (does not dissipate) the
    energy, and yields if s F > sigma_y. Also the most energy it can take
    under the force limit, F_lim^2 / 2k."""
    fpk = float(np.sqrt(2.0 * energy * stiffness))
    out = {"stiffness": round(float(stiffness), 1), "peakForce": round(fpk, 1),
           "crush": round(fpk / stiffness, 4), "peakStress": round(stress_per_newton * fpk, 1),
           "yields": bool(stress_per_newton * fpk > sigma_y),
           "energyAtYield": round(float((sigma_y / stress_per_newton) ** 2 / (2 * stiffness)), 1),
           "rebound": "elastic: the stored energy comes back"}
    if force_limit is not None:
        out["energyAtForceLimit"] = round(float(force_limit ** 2 / (2 * stiffness)), 2)
    if mass:
        out["peakG"] = round(fpk / (mass * 9.81), 1)
    return out


# ---- 3D check -------------------------------------------------------------------------------

def face_load(mesh, axis: int, total: float, top: bool = True) -> np.ndarray:
    """A force `total` (N) along `axis` spread evenly over the mesh's
    highest (or lowest) face normal to that axis, each node taking its
    share of the face cells around it (a uniform pressure, consistently)."""
    from .fem import CORNERS

    k = mesh.elem_ijk[:, axis]
    edge = k.max() if top else k.min()
    corners = np.flatnonzero(CORNERS[:, axis] == (1 if top else 0))
    elems = np.flatnonzero(k == edge)
    f = np.zeros(mesh.n_dof)
    np.add.at(f, 3 * mesh.enode[np.ix_(elems, corners)].ravel() + axis, total / (4.0 * len(elems)))
    return f


FOAM_ALPHA = 2.12   # Deshpande & Fleck (2000): crushable foam with zero plastic Poisson ratio


def foam_stress(stress, alpha: float = FOAM_ALPHA):
    """Deshpande-Fleck effective stress of a crushable cellular solid:
    sqrt((vm^2 + alpha^2 sm^2) / (1 + (alpha/3)^2)), sm the mean stress.
    Equals the stress in uniaxial compression; unlike von Mises it counts
    pressure, so a laterally confined lattice still crushes (alpha 0 is
    von Mises, a solid)."""
    from .fem import von_mises

    vm = von_mises(stress)
    sm = stress[:, :3].mean(axis=1)
    return np.sqrt((vm ** 2 + alpha ** 2 * sm ** 2) / (1 + (alpha / 3) ** 2))


def limit_load(mesh, modulus, strength, nu: float, fixed, force, iters: int = 15,
               alpha: float = FOAM_ALPHA) -> dict:
    """Collapse load factor of a voxel model (per-element modulus and
    strength, MPa; for a homogenized lattice the cell's plateau stress) under
    `force` (N per dof), by the elastic compensation method: lower bound
    min over elements of strength / effective stress per pass, elements
    softened by sigma_ref / sigma_e between passes. The effective stress is
    `foam_stress` (alpha 0 for a solid). lambda < 1: it collapses below the
    applied load. Also returns the elastic stiffness along the load
    (|f|^2 / f.u) of the first pass."""
    from scipy.sparse.linalg import cg

    from .fem import Stiffness, amg_solver

    mod = np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,)).copy()
    strength = np.broadcast_to(np.asarray(strength, float), (mesh.n_elem,))
    st = Stiffness(mesh, nu, np.asarray(fixed, bool))
    f = np.where(st.fixed, 0.0, np.asarray(force, float))
    scale = mod.copy()
    hist, best, k0, upper = [], 0.0, None, np.inf
    u = None
    for _ in range(iters):
        k = st.assemble(scale)
        ml = amg_solver(k, symmetry="hermitian", max_coarse=500)
        u, info = cg(k, f, x0=u, rtol=1e-7, maxiter=1500, M=ml.aspreconditioner())
        if k0 is None:
            k0 = float(f @ f / max(f @ u, 1e-30))
        se = foam_stress(st.stress(u, scale), alpha)
        util = se / strength
        lam = 1.0 / float(util.max())
        best = max(best, lam)
        hist.append(lam)
        # Mackenzie & Boyle's upper bound from the same field (kinematic: the
        # strain field is compatible), sum(s sigma / E) / sum(sigma^2 / E)
        ub = float(np.sum(strength * se / scale) / max(np.sum(se ** 2 / scale), 1e-300))
        upper = min(upper, ub)
        ref = float(np.mean(util))
        # softening only, at most 1000x, which keeps the solves well conditioned
        scale = np.clip(scale * np.clip(ref / np.maximum(util, 1e-12), 0.05, 20.0), 1e-3 * mod, mod)
    return {"loadFactor": best, "upperBound": upper, "firstYieldFactor": hist[0], "history": hist,
            "stiffness": k0}


# ---- field graphs -------------------------------------------------------------------------

def sample_core(graph, envelope: str, density=None, pitch: float = 0.5, bounds=None, with_field: bool = False):
    """Voxel samples of a field graph's lattice core: (inside, rho, lo, pitch)
    with `inside` the cells whose centres are inside the `envelope` solid
    and rho the `density` field there (a number, a node name, or None for
    solid)."""
    b = graph.bounds(envelope) if bounds is None else bounds
    if b is None:
        raise ValueError(f"node {envelope!r} has no bounds; give the analysis 'bounds'")
    lo, hi = (np.asarray(x, float) for x in b)
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    axes = [lo[i] + (np.arange(dims[i]) + 0.5) * pitch for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    phi = np.asarray(graph.evaluate(P, envelope), float).reshape(tuple(dims))
    inside = phi < 0
    if density is None:
        rho = np.ones(inside.shape)
    elif isinstance(density, (int, float)):
        rho = np.full(inside.shape, float(density))
    else:
        rho = np.broadcast_to(np.asarray(graph.evaluate(P, density), float), (len(P),)).reshape(tuple(dims))
    if with_field:
        return inside, np.clip(rho, 0.0, 1.0), lo, pitch, phi
    return inside, np.clip(rho, 0.0, 1.0), lo, pitch


def _energy(spec):
    if spec.get("energy") is not None:
        return 1000.0 * float(spec["energy"])                    # J -> N mm
    return 1000.0 * 0.5 * float(spec["mass"]) * float(spec["velocity"]) ** 2


def impact_report(graph, spec: dict, material) -> dict:
    """The field graph's `impact` block: crush the `envelope` filled with the
    `lattice` at `density` along `axis`, stop the impact on it, and (with
    `grade`) work out the flattest graded core for the same envelope as a
    ready remap table for the density node."""
    kind = spec.get("lattice", "gyroid")
    axis = "xyz".index(spec.get("axis", "z"))
    pitch = float(spec.get("pitch", 0.5))
    inside, rho, lo, h, phi = sample_core(graph, spec["envelope"], spec.get("density"), pitch, with_field=True)
    law, sy = crush_law(kind, material)
    E = float(material.E)
    other = tuple(i for i in range(3) if i != axis)
    area, mean = layer_profile(inside, rho, h, axis)
    lf = (plateau_stress(law, rho, sy) * inside).sum(axis=other) * h ** 2
    energy = _energy(spec)
    mass = spec.get("mass")
    flim = spec.get("forceLimit")
    striker = spec.get("striker")
    cols = None
    if striker is not None:
        # what it hits and from which side: the core as columns under the
        # contact (contact_curve), not as layers that each carry the whole load
        if not isinstance(striker, dict):
            raise ValueError("impact.striker is {side: '+' | '-', shape: flat | sphere | cylinder, radius, along, at}")
        side = striker.get("side", "+")
        cols = contact_columns(inside, rho, h, axis, side, striker, lo, field=phi)

    def curve_at(sy_, e_, fac=1.0):
        if cols is not None:
            return contact_curve(law, cols, sy_, e_)
        return crush_curve(law, area, mean, h, sy_, e_, layer_force=lf * fac)

    cv = curve_at(sy, E)
    resp = impact_response(cv, energy, mass)
    rate = None
    if law.endswith(":elastic") and mass and spec.get("rate", True):
        # an elastomer is stiffer in a fast knock than in the card's test: the
        # crush time (constant-force estimate 2 d / v) sets the factor
        from .viscoelastic import creep_params, event_factor

        if creep_params(material) is not None:
            v = np.sqrt(2.0 * energy * 1e-3 / float(mass))                     # m/s
            fac = 1.0
            for _ in range(4):
                t = 2.0 * float(resp["crush"]) * 1e-3 / v
                fac = float(event_factor(material, t))
                cv = curve_at(sy * fac, E * fac, fac)
                resp = impact_response(cv, energy, mass)
            sy, E = sy * fac, E * fac
            rate = {"eventSeconds": round(t, 4), "stiffening": round(fac, 3)}
    core_mass = float((rho * inside).sum() * h ** 3 * material.density / 1000.0)
    out = {"lattice": kind, "law": "elastic buckling" if law.endswith(":elastic") else "plastic collapse",
           "axis": "xyz"[axis], "energy": round(energy, 1), "coreMass": round(core_mass, 2),
           "meanDensity": round(float((rho * inside).sum() / max(inside.sum(), 1)), 3),
           "response": resp, "curve": cv.to_dict()}
    if rate:
        out["rate"] = rate
    if cols is not None:
        d_stop = float(resp["crush"])
        out["contact"] = {"model": "columns", "side": striker.get("side", "+"), "striker": striker.get("shape", "flat"),
                          "area": round(float(np.interp(d_stop, cv.d, cv.layer_area)), 1),
                          "lockedShare": round(float(np.interp(d_stop, cv.d, cv.layer_stroke)), 3),
                          "firstLockAt": round(float(cv.stroke), 3)}
    if law.endswith(":elastic"):
        # an elastomer core springs back: only its hysteresis is dissipated
        hy = HYSTERESIS.get(material.name, 0.3)
        out["dissipated"] = hy
        out["reboundEnergy"] = round((1 - hy) * energy, 1)
    if flim is not None:
        out["forceLimit"] = float(flim)
        out["ok"] = bool(not resp["bottomedOut"] and resp["peakForce"] <= flim)
    if spec.get("grade"):
        rho_min = float(spec.get("rhoMin", 0.12))
        if spec.get("cell") is not None:
            # the thinnest printable wall sets the lowest density at this cell
            rho_min = max(rho_min, float(relative_density(kind, float(spec.get("minWall", 0.8)),
                                                          float(spec["cell"]))))
        z = lo[axis] + (np.arange(len(area)) + 0.5) * h
        keep = area > 0
        if cols is not None:
            # layer index of every cell, from the struck end, in place of its density
            n_l = inside.shape[axis]
            shp = [1, 1, 1]
            shp[axis] = n_l
            k_idx = np.arange(n_l)[::-1] if striker.get("side", "+") == "+" else np.arange(n_l)
            lay = contact_columns(inside, np.broadcast_to(k_idx.reshape(shp).astype(float), inside.shape), h, axis,
                                  striker.get("side", "+"), striker, lo, field=phi)
            g0 = grade_for_impact(law, area, h, sy, E, energy, flim, rho_min, float(spec.get("rhoMax", 0.5)))
            g = grade_for_contact(law, lay, n_l, sy, E, energy, flim, rho_min, float(spec.get("rhoMax", 0.5)),
                                  candidates=[g0["rho"][k_idx]])
            g["rho"] = g["rho"][k_idx]                    # back to the grid's own order along the axis
        else:
            g = grade_for_impact(law, area, h, sy, E, energy, flim, rho_min, float(spec.get("rhoMax", 0.5)))
        zs, rs = z[keep], g["rho"][keep]
        # thin the table to every ~1 mm (the field is linear in between)
        step = max(1, int(round(1.0 / h)))
        idx = np.unique(np.r_[np.arange(0, len(zs), step), len(zs) - 1])
        gm = float((g["rho"] * area).sum() * h * material.density / 1000.0)
        # the fair comparison: a uniform core of the same mass
        if cols is not None:
            cu = contact_curve(law, cols, sy, E, rho=np.where(np.isfinite(cols.rho), g["meanRho"], np.nan))
        else:
            cu = crush_curve(law, area, g["meanRho"], h, sy, E)
        out["uniformSameMass"] = {"density": round(g["meanRho"], 4), "response": impact_response(cu, energy, mass),
                                  "curve": cu.to_dict()}
        out["graded"] = {"ok": g["ok"], "plateauForce": None if g["force"] is None else round(g["force"], 1),
                         "coreMass": round(gm, 2),
                         "rhoMin": round(rho_min, 4),
                         "response": impact_response(g["curve"], energy, mass), "curve": g["curve"].to_dict(),
                         "density": {"op": "remap", "input": {"op": "coordinate", "axis": "xyz"[axis]},
                                     "from": [round(float(x), 3) for x in zs[idx]],
                                     "to": [round(float(x), 4) for x in rs[idx]]}}
    return out


def grade_3d(inside, lo, h: float, kind: str, material, force: float, rho0=None, rho_min: float = 0.12,
             rho_max: float = 0.5, axis: int = 2, iters: int = 16, alpha: float = FOAM_ALPHA,
             keep_layers: bool = True, voids: bool = True, void_below: float = 0.5) -> dict:
    """Density per cell, not just per layer, for a core that crushes at
    `force` (N, pressed evenly on the top face along `axis`, foot held).

    A layer-only grading (grade_for_impact) gives every layer the right
    total strength, but in a taper the load from a narrow layer stays in a
    column and reaches the middle of the wider layers below before their
    rims: the middle crushes early and the rims idle. Here each pass solves
    the homogenized core elastically and, within each layer, shares the
    layer's crush force out in proportion to where the (crushable-foam)
    stress goes, so each layer still crushes at `force` (`keep_layers`).
    Cells that would need less than `void_below` x rho_min become empty
    (`voids`): the idle rim is removed and its material moves to the loaded
    middle, which is why the result is lighter as well. Damped, as the load
    path moves with the densities.

    inside: bool voxel grid (cell centres at lo + (ijk + 0.5) h); rho0: the
    starting density per voxel (default rho_min). Returns the density grid
    (0 = empty; outside cells carry the nearest inside value so a field
    graph can sample it anywhere), the worst utilization per pass and the
    core mass. In a graph, a grid node of it drives the lattice density and
    an expr `0.1 - density` intersected with the envelope cuts the empty
    cells (see examples/impact/end_stop.py)."""
    from scipy import ndimage
    from scipy.sparse.linalg import cg

    from .fem import HexMesh, Stiffness, amg_solver
    from .lattice import sheet_modulus

    kind, sy = crush_law(_base(kind), material)
    E = float(material.E)
    ijk = np.argwhere(inside)
    mesh = HexMesh(ijk, h, lo)
    rho = (np.full(len(ijk), rho_min) if rho0 is None
           else np.clip(np.broadcast_to(rho0, inside.shape)[inside], rho_min, rho_max)).astype(float)
    x = mesh.node_xyz[:, axis]
    fixed = np.zeros(mesh.n_dof, bool)
    foot = np.flatnonzero(x < x.min() + 1e-6)
    fixed[3 * foot + axis] = True
    others = [a for a in range(3) if a != axis]
    fixed[3 * foot[0] + others[0]] = fixed[3 * foot[0] + others[1]] = True     # hold it still sideways
    fixed[3 * foot[-1] + others[1]] = True
    f = face_load(mesh, axis, -force)
    st = Stiffness(mesh, float(material.nu), fixed)
    f = np.where(st.fixed, 0.0, f)
    u, hist = None, []
    for _ in range(iters):
        mod = np.where(rho > 0, E * sheet_modulus(_base(kind), np.maximum(rho, 1e-3)), 1e-6 * E)
        k = st.assemble(mod)
        u, _info = cg(k, f, x0=u, rtol=1e-7, maxiter=1500,
                      M=amg_solver(k, symmetry="hermitian", max_coarse=500).aspreconditioner())
        se = foam_stress(st.stress(u, mod), alpha)
        util = np.where(rho > 0, se / plateau_stress(kind, np.maximum(rho, 1e-3), sy), 0.0)
        hist.append(float(util.max()))
        if keep_layers:
            # within each layer, share the layer's crush force out in proportion
            # to where the stress goes: scale c per layer so sum sigma_pl A = force
            lay = ijk[:, axis]
            nl = lay.max() + 1
            lo_c, hi_c = np.full(nl, 1e-3), np.full(nl, 1e3)
            for _b in range(40):
                c = np.sqrt(lo_c * hi_c)
                t = np.clip(density_for_plateau(kind, c[lay] * se, sy), rho_min, rho_max)
                if voids:
                    t = np.where(density_for_plateau(kind, c[lay] * se, sy) < void_below * rho_min, 0.0, t)
                tot = np.bincount(lay, plateau_stress(kind, t, sy) * h ** 2, nl)
                over = tot > force
                hi_c = np.where(over, c, hi_c)
                lo_c = np.where(over, lo_c, c)
            target = t
        else:
            target = np.clip(density_for_plateau(kind, se, sy), rho_min, rho_max)
        rho = np.where(target == 0.0, 0.0, np.sqrt(np.maximum(rho, rho_min) * np.maximum(target, rho_min)))
    grid = np.zeros(inside.shape)
    grid[inside] = rho
    idx = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    return {"rho": grid[tuple(idx)], "inside": inside, "history": hist, "utilization": util,
            "mass": float(rho.sum() * h ** 3 * material.density / 1000.0)}


def save_density_grid(path: str, rho, lo, h: float) -> None:
    """A density grid for a field graph's `grid` node: values at the voxel
    centres lo + (ijk + 0.5) h."""
    np.savez_compressed(path, values=np.asarray(rho, np.float32), origin=np.asarray(lo, float) + 0.5 * h,
                        pitch=float(h))
