"""
Thin pinned walls on the design grid, from their signed distance.

A wall the study pins (a skin, a duct's or plenum's `wall` round a
keep-out) carries load by bending, and bending goes as thickness cubed. On
whole cells a wall rounds to a cell: a 1.6 mm wall on 1.1 mm cells is one
cell (1.1 mm, a third as stiff) or two (2.2 mm, 2.6x as stiff), and the
design behind it is answered for the wrong wall. `loadpath run` used to buy
the wall's thickness back with the whole grid: two cells through the
thinnest wall (domain.wall_pitch), which on a fan duct's 1.6 mm wall is
0.8 mm cells everywhere, up to four times the study's cell budget, for a
wall that is a few percent of the part. The search's every iteration paid
for it.

Here the grid stays at the study's own budget and the wall is put into it
the way the verification's cut cells put a finished part in (cutcell.py):
each cell the wall crosses takes the wall's own stiffness, integrated over
4^3 sub-cells weighted by the wall's share of each (from its signed
distance, so the thickness is not rounded to a sub-cell either), with the
incompatible-mode element that carries linear bending in one layer. The
switch, feel and cooling solves model a thin skin the same way
(switch._thin_skin); measured on four parts in docs/GENERAL-DESIGN.md, two
cells through a wall read to the scatter of six, and the sub-cell rule is
what holds below that.

How the wall enters the optimizer (the way bought stock does, stock.py):

- its stiffness is a fixed operator, sum over its cells of K_wall (plus a
  small share of a whole cell's, the finite-cell alpha that keeps nodes a
  sliver barely holds conditioned), added to every solve as a pin's
  springs are;
- the rest of each wall cell is still the optimizer's: its density there
  is the share of the rest it fills, at (1 - phi) of a cell's stiffness
  (`mesh.stock_scale`) and (1 - phi) of a cell's volume in the budget
  (`domain.volume_weights`). A rib's root and a wall's fillet live in that
  rest;
- a wall cell that touches a keep-out (a duct's bore) or lies outside the
  design space is not designed (`domain.wall_void`): its rest is air;
- the wall's weight is the wall's (`mesh.stock_density`), for self-weight
  and a free part's inertia.

The delivered part takes the wall exactly (reconstruct unions preserved
shapes), and the verification reads it on its own cut cells as before.

The constrained method reads a wall cell's stress as the verification
reads a cut cell (constrained._Averaged): the wall's own stress from its
cut-cell kinematics at its solid sub-cell centres, the quarter points
carrying only the design in the rest. LOADPATH_WALL_CONSTRAINED=0 keeps the
refined whole-cell grid for it. A pressure on a
keep-out acts on the nodes lining it, the wall cells' among them. A bought
member's cell stays the member's. Not for studies with a thermal block:
its expansion load and derating are assembled per whole cell, and the
wall's operator would neither expand nor soften. LOADPATH_WALL_MODEL=cells
restores the refined grid everywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# sub-cells per side the wall's share is integrated over (as switch.SKIN_SUB)
SUB = 4
# a wall thinner than this many cells is cut cells (as switch._thin_skin;
# from two up the wall is plain preserved cells, which round by at most a
# quarter of the wall)
CELLS = float(os.environ.get("LOADPATH_WALL_CELLS", 4.0))
# cells through the thinnest wall a run's grid is refined to (domain.wall_pitch):
# a cut cell bends like the wall from about one cell through it (a cantilever
# plate read 1.11x as stiff as at three cells, 0.46 cells through 1.9x), where
# whole cells needed two and still rounded the wall by up to a quarter
MIN_CELLS = 1.0
# the finite-cell alpha: a share of a whole cell's stiffness under every wall
# cell (switch.CUT_ALPHA)
ALPHA = 1e-4


# The constrained method on cut walls. Read at the quarter points with the
# whole-cell element (as first wired), a ribbed lid's best passing part was
# 22.4 g (tet safety factor 2.33 for a limit of 2) where the search on the
# same walls delivered 20.7 g; read at the wall's own sub-cells, 19.9 g (tet
# 2.14). LOADPATH_WALL_CONSTRAINED=0 turns it off.
CONSTRAINED = os.environ.get("LOADPATH_WALL_CONSTRAINED", "1").strip().lower() not in ("0", "false", "no", "off")


def model() -> str:
    """LOADPATH_WALL_MODEL: "cut" (default) or "cells" (refine the grid to
    two cells through the thinnest wall, as before)."""
    m = os.environ.get("LOADPATH_WALL_MODEL", "cut").strip().lower() or "cut"
    if m not in ("cut", "cells"):
        raise ValueError(f"LOADPATH_WALL_MODEL must be cut or cells, not {m!r}")
    return m


def applies(study) -> bool:
    """Whether `loadpath run` models this study's thin pinned walls as cut
    cells on its own grid (instead of refining the grid to them)."""
    if model() != "cut" or not study.shells:
        return False
    if study.goal.method == "constrained" and not CONSTRAINED:
        return False
    # a thermal block's expansion load and derating are assembled per whole
    # cell (thermal.py); the wall's own operator would neither expand nor soften
    return getattr(study, "thermal", None) is None


@dataclass
class WallCells:
    """The thin walls on one grid (indices into the grid's full cell list)."""
    idx: np.ndarray        # cells the walls cross (phi >= MIN_FILL)
    ind: np.ndarray        # (len(idx), SUB^3) the wall's share of each sub-cell
    phi: np.ndarray        # (len(idx),) the wall's share of each cell
    ids: list              # the walls' region ids


def thin_shells(study, pitch: float) -> list:
    """(id, thickness) of the pinned walls under CELLS cells on this grid."""
    return [(rid, float(t)) for rid, t in study.shells if float(t) < CELLS * pitch]


def wall_cells(study, centers: np.ndarray, pitch: float, hard: np.ndarray, cut=()) -> WallCells | None:
    """The cells the study's thin walls cross, and the wall's share of each
    of their sub-cells from the walls' signed distance. `hard`: cells other
    drawn geometry fills (bosses, pads), which stay whole preserved cells.
    `cut`: shapes that cut the wall (bores through it)."""
    from .cutcell import MIN_FILL

    thin = thin_shells(study, pitch)
    if not thin:
        return None
    shapes = [study.region_shape(rid)[0] for rid, _ in thin]

    def sdf(p):
        d = shapes[0].sdf(p)
        for s in shapes[1:]:
            d = np.minimum(d, s.sdf(p))
        for s in cut:
            d = np.maximum(d, -s.sdf(p))
        return d

    # candidates: any of the wall within a cell's circumradius
    idx = np.flatnonzero((sdf(centers) < 0.87 * pitch) & ~hard)
    if not len(idx):
        return None
    sub = (np.arange(SUB) + 0.5) / SUB - 0.5
    offs = np.array([[a, b, c] for a in sub for b in sub for c in sub]) * pitch
    hs = pitch / SUB
    ind = np.empty((len(idx), len(offs)))
    for k, o in enumerate(offs):
        ind[:, k] = np.clip(0.5 - sdf(centers[idx] + o) / hs, 0.0, 1.0)
    phi = ind.mean(axis=1)
    on = phi >= MIN_FILL
    return WallCells(idx[on], ind[on], phi[on], [rid for rid, _ in thin])


def operator(walls: WallCells, mesh, elem: np.ndarray, nu, fixed: np.ndarray | None = None):
    """The walls' stiffness on `mesh` at unit modulus (the springs' unit):
    sum over wall elements `elem` (mesh element indices, in walls' order) of
    the cut cell's matrix, plus ALPHA of a whole cell's. Held dofs' rows and
    columns cleared."""
    import scipy.sparse as sp

    from .cutcell import SubcellRule
    from .fem import hex8_ke

    rule = SubcellRule(nu, SUB)
    k0 = hex8_ke(nu)
    h = mesh.pitch
    ed = mesh.edof[elem]
    rows, cols, vals = [], [], []
    for s in range(0, len(elem), 4000):
        sl = slice(s, s + 4000)
        ks = (rule.matrices(walls.ind[sl])[0] + ALPHA * k0[None]) * h
        e = ed[sl]
        rows.append(np.repeat(e, 24, axis=1).ravel())
        cols.append(np.tile(e, (1, 24)).ravel())
        vals.append(ks.ravel())
    k = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(mesh.n_dof, mesh.n_dof))
    if fixed is not None and np.any(fixed):
        free = sp.diags((~np.asarray(fixed, bool)).astype(float))
        k = (free @ k @ free).tocsr()
    k.eliminate_zeros()
    return k
