"""
Strut (beam) lattices as distance fields: the periodic truss cells nTop and
every AM lattice library ship (simple cubic, BCC, BCCz, FCC, FCCz, octet,
Kelvin, diamond), and stochastic Voronoi foams.

A strut lattice is a set of segments; the solid is every point within
`thickness / 2` of one, so its field is the exact distance to the nearest
segment less the radius (Lipschitz 1, which the narrow-band mesher and the
Newton projection both like). The radius can vary per point, so a strut
thickens where a field says, and `blend` rounds every node with a smooth
minimum (nTop's node fillets): thin struts break at their nodes first.

Periodic cells are evaluated in cell units (period 1). A point is folded
into the cell's fundamental region first, by the mirrors and axis swaps the
cell is invariant under (found numerically, so a new cell needs no hand
work), and only the few segments that can be nearest to that region are
tested. For the cubic cells that is 1/48 of the cell and a handful of
segments; BCCz and FCCz keep z special (no swaps with z), diamond has no
mirrors through its nodes, so it tests more.

Voronoi foams are not periodic: seeds are scattered over a box at a mean
spacing that may vary (a cell field: denser seeds where the cell is small),
relaxed a few steps so the cells are even, and the Voronoi edges are the
struts. A periodic Voronoi unit cell (seeds tiled 3x3x3) stands in for the
foam in homogenization.

Strut directions matter for FDM: a strut is self-supporting when it rises
at least `angle` (45 degrees) from the bed; flatter ones are bridges.
`overhang_report` says how much strut length bridges, for a cell in any
orientation, or through any coordinate map (graded or conformal frames).
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

__all__ = ["STRUT_KINDS", "is_strut", "cell_segments", "strut_distance", "strut_density_table",
           "strut_relative_density", "strut_diameter_for_density", "VoronoiStruts", "overhang_report",
           "unit_cell_distance", "best_orientation", "lattice_printability"]

_CHUNK = 65_536


def _fcc():
    return [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]


def _nodes_pairs(nodes, length, tol=1e-6):
    """Every pair of the periodic node set (`nodes` in [0,1)^3, tiled) at
    `length` apart, as segments with the first end in the home cell."""
    base = np.asarray(nodes, float)
    offs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    allp = (base[None] + offs[:, None]).reshape(-1, 3)
    seg = []
    for a in base:
        d = np.linalg.norm(allp - a, axis=1)
        for b in allp[np.abs(d - length) < tol]:
            seg.append((a, b))
    return seg


def _axis_edges(axis):
    """Cube edges along one axis (period 1: one per cell)."""
    e = np.zeros(3)
    e["xyz".index(axis)] = 1.0
    return [(np.zeros(3), e)]


def _kelvin():
    # bitruncated cubic honeycomb: truncated octahedra on a BCC lattice,
    # vertices the permutations of (0, +-1/4, +-1/2) round each centre
    v = set()
    for p in set(itertools.permutations((0, 1, 2))):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                q = np.zeros(3)
                q[p[1]] = 0.25 * s1
                q[p[2]] = 0.5 * s2
                for c in ((0, 0, 0), (0.5, 0.5, 0.5)):
                    w = tuple(np.round((q + c) % 1.0, 6))
                    v.add(w)
    return _nodes_pairs(sorted(v), np.sqrt(2) / 4)


def _diamond():
    nodes = _fcc() + [tuple(np.add(p, 0.25)) for p in _fcc()]
    return _nodes_pairs(nodes, np.sqrt(3) / 4)


# segments of each unit cell, period 1
_CELLS = {
    "cubic": lambda: _axis_edges("x") + _axis_edges("y") + _axis_edges("z"),
    "bcc": lambda: _nodes_pairs([(0, 0, 0), (0.5, 0.5, 0.5)], np.sqrt(3) / 2),
    "bccz": lambda: _nodes_pairs([(0, 0, 0), (0.5, 0.5, 0.5)], np.sqrt(3) / 2) + _axis_edges("z"),
    "fcc": lambda: [s for s in _nodes_pairs(_fcc(), np.sqrt(2) / 2)
                    if np.count_nonzero(np.abs(np.asarray(s[0]) % 1.0) < 1e-9) >= 2
                    or np.count_nonzero(np.abs(np.asarray(s[1]) % 1.0) < 1e-9) >= 2],
    "fccz": lambda: [s for s in _nodes_pairs(_fcc(), np.sqrt(2) / 2)
                     if np.count_nonzero(np.abs(np.asarray(s[0]) % 1.0) < 1e-9) >= 2
                     or np.count_nonzero(np.abs(np.asarray(s[1]) % 1.0) < 1e-9) >= 2] + _axis_edges("z"),
    "octet": lambda: _nodes_pairs(_fcc(), np.sqrt(2) / 2),
    "kelvin": _kelvin,
    "diamond-strut": _diamond,
}
STRUT_KINDS = tuple(_CELLS) + ("voronoi",)
_ALIASES = {"simple-cubic": "cubic", "bcc-z": "bccz", "fcc-z": "fccz", "octet-truss": "octet",
            "tetrakaidecahedron": "kelvin", "diamond-beam": "diamond-strut", "strut-diamond": "diamond-strut"}


def canonical(kind):
    k = str(kind).lower()
    return _ALIASES.get(k, k)


def is_strut(kind) -> bool:
    return canonical(kind) in STRUT_KINDS


def _dedupe(segs):
    """Unique segments (either direction)."""
    out, seen = [], set()
    for a, b in segs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        key = tuple(np.round(np.concatenate([a, b]), 6)), tuple(np.round(np.concatenate([b, a]), 6))
        if key[0] in seen or key[1] in seen:
            continue
        seen.add(key[0])
        out.append((a, b))
    return out


@lru_cache(maxsize=None)
def cell_segments(kind) -> np.ndarray:
    """(S, 2, 3) the unit cell's struts, each once (period 1), so one cell's
    worth of strut length; translate by whole cells for the rest."""
    kind = canonical(kind)
    if kind == "voronoi":
        return _periodic_voronoi(8, 0)
    segs = _CELLS[kind]()
    # one representative per periodic class: shift each so its midpoint is in [0,1)
    reps = []
    for a, b in segs:
        m = 0.5 * (np.asarray(a) + np.asarray(b))
        sh = np.floor(m + 1e-9)
        reps.append((np.asarray(a) - sh, np.asarray(b) - sh))
    return np.asarray(_dedupe(reps), float)


def _seg_dist(P, A, B):
    """(N, S) distance from points to segments."""
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-18)
    AP = P[:, None, :] - A[None]
    t = np.clip(np.einsum("nsj,sj->ns", AP, AB) / L2, 0.0, 1.0)
    D = AP - t[..., None] * AB[None]
    return np.sqrt(np.einsum("nsj,nsj->ns", D, D))


def _all_translates(segs, reach=1):
    offs = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), float)
    return (segs[None, :, :, :] + offs[:, None, None, :]).reshape(-1, 2, 3)


@lru_cache(maxsize=None)
def _symmetry(kind):
    """(mirror, swaps) the cell is invariant under, about its corner node:
    mirror x -> -x on every axis (and through the cell's 1/2 planes, which
    follows with period 1), and which axis swaps (all, xy only, none)."""
    segs = _all_translates(cell_segments(kind), 2)
    rng = np.random.default_rng(1)
    P = rng.uniform(-0.5, 0.5, (600, 3))

    def field(Q):
        return _seg_dist(Q, segs[:, 0], segs[:, 1]).min(axis=1)

    f0 = field(P)
    mirror = all(np.allclose(field(P * np.where(np.arange(3) == a, -1, 1)), f0, atol=1e-9) for a in range(3))
    swap_xy = np.allclose(field(P[:, [1, 0, 2]]), f0, atol=1e-9)
    swap_xz = np.allclose(field(P[:, [2, 1, 0]]), f0, atol=1e-9)
    return mirror, ("all" if swap_xy and swap_xz else "xy" if swap_xy else "none")


@lru_cache(maxsize=None)
def _candidates(kind, reach=0.32, aspect=None):
    """Segments that can be nearest to a point of the folded region (within
    `reach` of it, cell units: a strut up to 0.6 cell across is right).
    With a stretched cell (`aspect`, per axis) distances are measured in
    the stretched space and only the mirrors fold (no axis swaps)."""
    mirror, swaps = _symmetry(kind)
    if aspect is not None:
        swaps = "none"
    segs = _all_translates(cell_segments(kind), 2)
    n = 18
    c = (np.arange(n) + 0.5) / n
    if mirror:
        g = np.stack(np.meshgrid(*[0.5 * c] * 3, indexing="ij"), -1).reshape(-1, 3)
        if swaps == "all":
            g = g[(g[:, 0] <= g[:, 1] + 1e-9) & (g[:, 1] <= g[:, 2] + 1e-9)]
        elif swaps == "xy":
            g = g[g[:, 0] <= g[:, 1] + 1e-9]
        h = 0.5 / n
    else:
        g = np.stack(np.meshgrid(*[c - 0.5] * 3, indexing="ij"), -1).reshape(-1, 3)
        h = 1.0 / n
    near = np.zeros(len(segs), bool)
    best = np.full(len(g), np.inf)
    if aspect is not None:
        a = np.asarray(aspect, float)
        segs, g, h = segs * a, g * a, h * float(a.max())
    for i in range(0, len(segs), 64):
        d = _seg_dist(g, segs[i:i + 64, 0], segs[i:i + 64, 1])
        best = np.minimum(best, d.min(axis=1))
    for i in range(0, len(segs), 64):
        d = _seg_dist(g, segs[i:i + 64, 0], segs[i:i + 64, 1])
        # nearest to some sample by less than the cell's reach (plus the
        # sample spacing), or within reach of the strut surface at all
        near[i:i + 64] = (d <= np.minimum(best[:, None] + 2 * h * np.sqrt(3), reach + h * np.sqrt(3))).any(axis=0)
    if aspect is not None:
        segs = segs / np.asarray(aspect, float)
    return segs[near], mirror, swaps


def _fold(U, mirror, swaps):
    f = U - np.round(U)
    if not mirror:
        return f
    f = np.abs(f)
    if swaps == "all":
        f = np.sort(f, axis=1)
    elif swaps == "xy":
        f[:, :2] = np.sort(f[:, :2], axis=1)
    return f


def _smin_many(D, k):
    """Smooth minimum over the columns of D, polynomial, radius k (scalar
    or per row): sharp mins where k is 0."""
    D = np.sort(D, axis=1)
    out = D[:, 0].copy()
    k = np.broadcast_to(np.asarray(k, float), (len(D),))
    if not np.any(k > 0):
        return out
    kk = np.maximum(k, 1e-12)
    for j in range(1, min(D.shape[1], 6)):
        b = D[:, j]
        h = np.clip(0.5 + 0.5 * (b - out) / kk, 0.0, 1.0)
        s = b * (1 - h) + out * h - kk * h * (1 - h)
        out = np.where(k > 0, s, out)
    return out


def unit_cell_distance(kind, U, blend=0.0, aspect=None):
    """Distance (cell units) from lattice coordinates U (period 1) to the
    nearest strut centreline of a periodic cell; `blend` (cell units, per
    point or scalar) smooth-unions the struts at the nodes. `aspect` (per
    axis) stretches the cell: U is in stretched cells, the distance in
    units of the unstretched cell."""
    kind = canonical(kind)
    a = None
    if aspect is not None and not np.allclose(aspect, 1.0):
        a = tuple(float(x) for x in aspect)
    segs, mirror, swaps = _candidates(kind, 0.32 * (max(a) / min(a) if a else 1.0), a)
    U = np.asarray(U, float)
    out = np.empty(len(U))
    blend = np.broadcast_to(np.asarray(blend, float), (len(U),))
    sc = np.asarray(a if a else (1.0, 1.0, 1.0))
    A, B = segs[:, 0] * sc, segs[:, 1] * sc
    step = max(1, _CHUNK * 8 // max(len(segs), 1))
    for i in range(0, len(U), step):
        f = _fold(U[i:i + step], mirror, swaps) * sc
        D = _seg_dist(f, A, B)
        bl = blend[i:i + step]
        out[i:i + step] = _smin_many(D, bl) if np.any(bl > 0) else D.min(axis=1)
    return out


def strut_distance(kind, U, diameter_over_cell, blend=0.0):
    """Signed distance (cell units) of the strut solid: centreline distance
    less the radius."""
    return unit_cell_distance(kind, U, blend) - 0.5 * np.asarray(diameter_over_cell, float)


# ---- density ------------------------------------------------------------------------

@lru_cache(maxsize=None)
def strut_density_table(kind, blend=0.0):
    """(diameter/cell, relative density) on one cell, sampled at 2^20
    scrambled Sobol points (a regular grid aliases with axis-aligned struts:
    12% light on thin cubic rods)."""
    from scipy.stats import qmc

    g = qmc.Sobol(3, scramble=True, seed=7).random_base2(20)
    d = np.sort(unit_cell_distance(kind, g, blend))
    t = np.linspace(0.0, 0.8, 161)
    rho = np.searchsorted(d, 0.5 * t) / d.size
    return t, rho


def _blend_key(blend):
    return round(float(blend), 2)


# the periodic box of a cell, in cells: a Voronoi foam's `cell` is its seed
# spacing, and its periodic stand-in holds 8 seeds a box, 2 spacings across
_BOX = {"voronoi": 2.0}


def strut_relative_density(kind, diameter, cell, blend=0.0):
    kind = canonical(kind)
    box = _BOX.get(kind, 1.0)
    t, rho = strut_density_table(kind, _blend_key(blend / box))
    return np.interp(np.asarray(diameter, float) / (box * np.asarray(cell, float)), t, rho)


def strut_diameter_for_density(kind, rho, cell, blend=0.0):
    kind = canonical(kind)
    box = _BOX.get(kind, 1.0)
    t, r = strut_density_table(kind, _blend_key(blend / box))
    # the table is non-decreasing; interp wants strictly increasing x
    r, idx = np.unique(r, return_index=True)
    return np.interp(np.asarray(rho, float), r, t[idx]) * box * np.asarray(cell, float)


# ---- Voronoi foams ------------------------------------------------------------------

def _voronoi_edges(seeds, keep_lo, keep_hi):
    from scipy.spatial import Voronoi

    vor = Voronoi(seeds)
    V = vor.vertices
    edges = set()
    for ridge in vor.ridge_vertices:
        if -1 in ridge or len(ridge) < 2:
            continue
        for a, b in zip(ridge, ridge[1:] + ridge[:1]):
            edges.add((min(a, b), max(a, b)))
    E = np.array(sorted(edges), int).reshape(-1, 2)
    if not len(E):
        return np.zeros((0, 2, 3))
    S = V[E]
    m = S.mean(axis=1)
    inside = np.all((m >= keep_lo) & (m <= keep_hi), axis=1)
    return S[inside]


@lru_cache(maxsize=None)
def _periodic_voronoi(n_seeds, seed):
    """A periodic Voronoi cell: `n_seeds` relaxed seeds in the unit box,
    tiled 3x3x3; the edges whose midpoints are in the home box."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, (n_seeds, 3))
    offs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    for _ in range(30):                       # periodic repulsion: even cells
        allp = (s[None] + offs[:, None]).reshape(-1, 3)
        d = s[:, None, :] - allp[None]
        r = np.linalg.norm(d, axis=2)
        r[r < 1e-9] = np.inf
        h = n_seeds ** (-1 / 3)
        w = np.clip(1.0 - r / h, 0, None) ** 2
        s = (s + 0.1 * h * (w[..., None] * d / r[..., None]).sum(axis=1)) % 1.0
    allp = (s[None] + offs[:, None]).reshape(-1, 3)
    return _voronoi_edges(allp, np.zeros(3), np.ones(3) - 1e-12)


class VoronoiStruts:
    """A stochastic Voronoi strut foam over a box: seeds `cell` mm apart on
    average (a callable gives a per-point spacing), relaxed so cells come out
    even, Voronoi edges as struts. `distance(P)` is the exact distance to the
    nearest edge (edges are split so a k-nearest query on piece midpoints
    finds it)."""

    def __init__(self, lo, hi, cell, seed=0, relax=8):
        from scipy.spatial import cKDTree

        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        cell_fn = cell if callable(cell) else (lambda P, c=float(cell): np.full(len(P), c))
        rng = np.random.default_rng(seed)
        pad = 2.0 * float(np.max(cell_fn(np.array([0.5 * (lo + hi)]))))
        # rejection sampling at the finest spacing, density ~ 1 / cell^3
        probe = rng.uniform(lo - pad, hi + pad, (20_000, 3))
        cmin = float(np.min(cell_fn(probe)))
        vol = float(np.prod(hi - lo + 2 * pad))
        n_try = int(min(4e6, vol / cmin ** 3 * 1.0))
        C = rng.uniform(lo - pad, hi + pad, (n_try, 3))
        c = np.asarray(cell_fn(C), float)
        S = C[rng.uniform(0, 1, n_try) < (cmin / c) ** 3]
        if len(S) < 8:
            raise ValueError("voronoi lattice: the box holds fewer than 8 cells; make the cell smaller")
        for _ in range(relax):                # push neighbours apart by the local spacing
            tree = cKDTree(S)
            h = np.asarray(cell_fn(S), float)
            d, j = tree.query(S, k=min(9, len(S)))
            d, j = d[:, 1:], j[:, 1:]
            vec = S[:, None, :] - S[j]
            w = np.clip(1.0 - d / h[:, None], 0, None) ** 2
            S = S + 0.15 * h[:, None] * (w[..., None] * vec / np.maximum(d, 1e-9)[..., None]).sum(axis=1)
        self.seeds = S
        segs = _voronoi_edges(S, lo - 0.5 * pad, hi + 0.5 * pad)
        # split long edges so each piece is short against the spacing
        L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
        piece = 0.2 * cmin
        k = np.maximum(1, np.ceil(L / piece).astype(int))
        A, B = [], []
        for n in np.unique(k):
            sel = segs[k == n]
            t = np.linspace(0, 1, n + 1)
            pa = sel[:, None, 0] + (sel[:, None, 1] - sel[:, None, 0]) * t[None, :-1, None]
            pb = sel[:, None, 0] + (sel[:, None, 1] - sel[:, None, 0]) * t[None, 1:, None]
            A.append(pa.reshape(-1, 3))
            B.append(pb.reshape(-1, 3))
        self.edges = segs
        self.A, self.B = np.concatenate(A), np.concatenate(B)
        self.tree = cKDTree(0.5 * (self.A + self.B))
        self.piece = piece
        self.cell_min = cmin

    def distance(self, P, blend=0.0, k=12):
        P = np.asarray(P, float)
        out = np.empty(len(P))
        blend = np.broadcast_to(np.asarray(blend, float), (len(P),))
        for i in range(0, len(P), _CHUNK):
            Q = P[i:i + _CHUNK]
            _, j = self.tree.query(Q, k=k)
            A, B = self.A[j], self.B[j]
            AB = B - A
            AP = Q[:, None] - A
            t = np.clip(np.einsum("nkj,nkj->nk", AP, AB) / np.maximum(np.einsum("nkj,nkj->nk", AB, AB), 1e-18),
                        0, 1)
            D = np.linalg.norm(AP - t[..., None] * AB, axis=2)
            bl = blend[i:i + _CHUNK]
            out[i:i + _CHUNK] = _smin_many(D, bl) if np.any(bl > 0) else D.min(axis=1)
        return out


# ---- printability -------------------------------------------------------------------

def _rotation(build):
    b = np.asarray(build, float)
    return b / np.linalg.norm(b)


def overhang_report(kind, cell, diameter=None, build=(0, 0, 1), angle=45.0, jacobian=None,
                    max_bridge=None) -> dict:
    """How a strut cell prints on FDM: each strut's rise from the bed
    (degrees), the strut length below `angle` (bridging, unsupported), and
    the longest such bridge in mm. `jacobian` (M, 3, 3) are dx/du of the
    lattice's coordinate map sampled through the part (graded and conformal
    frames bend struts; None is the plain cell, du = dx / cell): every
    strut is mapped through each sample, so the fractions are over the part,
    not the cell. `max_bridge` (mm) is what the printer bridges cleanly;
    flat struts longer than that are flagged."""
    kind = canonical(kind)
    segs = cell_segments(kind)
    du = segs[:, 1] - segs[:, 0]
    up = _rotation(build)
    if jacobian is None:
        J = np.eye(3)[None] * float(np.mean(cell))
    else:
        J = np.asarray(jacobian, float)
    dx = np.einsum("mij,sj->msi", J, du)             # (M, S, 3) mm
    L = np.linalg.norm(dx, axis=2)
    rise = np.degrees(np.arcsin(np.clip(np.abs(dx @ up) / np.maximum(L, 1e-12), 0, 1)))
    flat = rise < angle - 1e-6
    total = float(L.sum())
    rep = {"kind": kind, "angle": float(angle), "build": up.tolist(),
           "strutsPerCell": int(len(segs)),
           "minRise": float(rise.min()), "unsupportedFraction": float(L[flat].sum() / max(total, 1e-12)),
           "horizontalFraction": float(L[rise < 5.0].sum() / max(total, 1e-12)),
           "longestBridge": float(L[flat].max()) if flat.any() else 0.0,
           "selfSupporting": bool(not flat.any())}
    if diameter is not None:
        rep["diameter"] = [float(np.min(diameter)), float(np.max(diameter))]
    if max_bridge is not None:
        rep["maxBridge"] = float(max_bridge)
        rep["bridgesTooLong"] = bool(rep["longestBridge"] > max_bridge)
    return rep


def best_orientation(kind, angle=45.0, n=600, seed=0):
    """The cell rotation (3x3, rows the cell axes in part coordinates) that
    leaves the least strut length below `angle` with build +z: random
    rotations plus the cube's special axes. Returns (R, report)."""
    from scipy.spatial.transform import Rotation

    rots = list(Rotation.random(n, random_state=seed).as_matrix())
    # z along [001], [011], [111] and their spins
    for ax in ((0, 0, 1), (0, 1, 1), (1, 1, 1)):
        a = np.asarray(ax, float) / np.linalg.norm(ax)
        r0 = Rotation.align_vectors([[0, 0, 1]], [a])[0]
        for spin in np.linspace(0, np.pi / 2, 7):
            rots.append((Rotation.from_rotvec([0, 0, spin]) * r0).as_matrix())
    best = None
    for R in rots:
        rep = overhang_report(kind, 1.0, jacobian=R[None], angle=angle)
        key = (rep["unsupportedFraction"], -rep["minRise"])
        if best is None or key < best[0]:
            best = (key, R, rep)
    return best[1], best[2]


def lattice_printability(graph, lattice, solid=None, build=(0, 0, 1), angle=45.0, samples=300, seed=0,
                         max_bridge=None) -> dict:
    """overhang_report for a FieldGraph's strut lattice node as it sits in
    the part: the coordinate map's Jacobian is sampled at points inside
    `solid` (default the graph's output), so graded, stretched and
    conformal cells are judged as they print. Voronoi foams are judged on
    their actual edges inside the solid."""
    nd = graph.nodes[lattice]
    kind = canonical(nd.spec.get("type", "gyroid"))
    solid = solid or graph.output
    b = graph.bounds(solid)
    if b is None:
        raise ValueError(f"{solid!r} is unbounded")
    lo, hi = b
    rng = np.random.default_rng(seed)
    P = rng.uniform(lo, hi, (samples * 20, 3))
    P = P[graph.evaluate(P, solid) < 0][:samples]
    if not len(P):
        raise ValueError(f"no points inside {solid!r} to sample the lattice at")
    up = _rotation(build)
    if kind == "voronoi":
        vor = graph._resource(nd, lambda: graph._voronoi(nd))
        E = vor.edges
        off = np.asarray(nd.spec.get("origin", (0, 0, 0)), float)
        mid = E.mean(axis=1) + off
        E = E[graph.evaluate(mid, solid) < 0]
        dx = E[:, 1] - E[:, 0]
        L = np.linalg.norm(dx, axis=1)
        rise = np.degrees(np.arcsin(np.clip(np.abs(dx @ up) / np.maximum(L, 1e-12), 0, 1)))
        flat = rise < angle - 1e-6
        rep = {"kind": kind, "angle": float(angle), "build": up.tolist(), "struts": int(len(E)),
               "minRise": float(rise.min()) if len(E) else 90.0,
               "unsupportedFraction": float(L[flat].sum() / max(L.sum(), 1e-12)),
               "horizontalFraction": float(L[rise < 5.0].sum() / max(L.sum(), 1e-12)),
               "longestBridge": float(L[flat].max()) if flat.any() else 0.0, "selfSupporting": bool(not flat.any())}
    else:
        eps = 1e-3
        U0, cell = graph.lattice_coords(nd, P)
        Jinv = np.empty((len(P), 3, 3))
        for a in range(3):
            dP = np.zeros(3)
            dP[a] = eps
            U1, _ = graph.lattice_coords(nd, P + dP)
            du = U1 - U0
            du -= np.round(du)              # phases wrap
            Jinv[:, :, a] = du / eps        # du/dx
        ok = np.abs(np.linalg.det(Jinv)) > 1e-9
        J = np.zeros_like(Jinv)
        J[ok] = np.linalg.inv(Jinv[ok])     # dx/du
        # a conformal map's phase defects (and the medial axis, where the
        # layer coordinate folds) have no cell to speak of: leave them out
        span = np.linalg.norm(J, axis=1) / cell[:, None]
        ok &= np.all((span > 0.3) & (span < 3.0), axis=1)
        if not ok.any():
            raise ValueError(f"lattice {lattice!r}: no regular cell among the samples")
        rep = overhang_report(kind, float(np.mean(cell)), jacobian=J[ok], build=build, angle=angle)
        rep["irregularSamples"] = int((~ok).sum())
        rep["cell"] = [float(cell.min()), float(cell.max())]
    if max_bridge is not None:
        rep["maxBridge"] = float(max_bridge)
        rep["bridgesTooLong"] = bool(rep["longestBridge"] > max_bridge)
    rep["samples"] = int(len(P))
    return rep
