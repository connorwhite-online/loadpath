"""
The benchmark problems: the textbook topology-optimization cases, each
defined once on a unit voxel grid so the same boundary conditions drive the
published reference code (top88, top3d, run in Octave), loadpath, and the
evaluator that scores every design.

Grid convention is loadpath's: cell (i, j, k) along x, y, z with z up, unit
cells, unit Young's modulus, nu = 0.3. A plane (2D) problem has one cell in y
and is solved in plane stress by the reference and the evaluator; loadpath
optimizes it as a one-cell slab extruded along y.

Boundary conditions are node sets in grid coordinates:
    fixed: [(nodes (N, 3) int, "xz")]          components held at zero
    loads: [(nodes (N, 3) int, "z", value)]    force per node
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Problem", "PROBLEMS", "get"]


def _nodes(i, j, k):
    """Every node (i, j, k) of the given index ranges (ints or iterables)."""
    ii, jj, kk = np.meshgrid(np.atleast_1d(i), np.atleast_1d(j), np.atleast_1d(k), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.int64)


@dataclass
class Problem:
    name: str
    title: str
    source: str
    dims: tuple                 # cells (nx, ny, nz)
    volfrac: float              # of the whole box, as top88/top3d count it
    rmin: float                 # filter radius, cells
    fixed: list
    loads: list
    references: list            # reference solver ids (reference.SOLVERS)
    plane: bool = False
    void: np.ndarray | None = None    # bool (nx, ny, nz): passive void cells
    # The reference model is the half of a symmetric part cut at x = 0 (the
    # half MBB beam). loadpath models the whole part with mirror symmetry, as
    # a study would; `full` is that problem and the delivered design's
    # x >= nx half is the one scored.
    full: "Problem | None" = None
    symmetry: list = field(default_factory=list)
    # Also score peak von Mises and run loadpath's stress-constrained
    # minimum-mass formulation against the reference's peak.
    stress: bool = False
    notes: str = ""

    @property
    def active(self) -> np.ndarray:
        a = np.ones(self.dims, bool)
        if self.void is not None:
            a &= ~self.void
        return a

    @property
    def active_volfrac(self) -> float:
        """The same amount of material as a share of the cells that exist."""
        return self.volfrac * float(np.prod(self.dims)) / float(self.active.sum())


def mbb_2d(nx=150, nz=50, volfrac=0.5, rmin=2.0):
    """Half MBB beam exactly as top88 ships it: symmetry line on the left,
    roller at the bottom right, unit load down at the top left."""
    half = Problem(
        name="mbb-2d", title=f"MBB beam, 2D half model ({nx}x{nz})",
        source="Andreassen et al. 2011, 'Efficient topology optimization in MATLAB using 88 lines of code' "
               "(top88.m, default load case)",
        dims=(nx, 1, nz), volfrac=volfrac, rmin=rmin, plane=True,
        fixed=[(_nodes(0, 0, range(nz + 1)), "x"), (_nodes(nx, 0, 0), "z")],
        loads=[(_nodes(0, 0, nz), "z", -1.0)],
        references=["top88-ft1", "top88-ft2"])
    # whole beam: x in [0, 2nx], the half above sits at x >= nx
    half.full = Problem(
        name="mbb-2d-full", title="MBB beam, whole", source=half.source,
        dims=(2 * nx, 1, nz), volfrac=volfrac, rmin=rmin, plane=True,
        fixed=[(_nodes(0, 0, 0), "xz"), (_nodes(2 * nx, 0, 0), "z")],
        loads=[(_nodes(nx, 0, nz), "z", -2.0)],
        references=[], symmetry=["x"])
    return half


def lbracket_2d(n=100, cut=0.6, volfrac_active=0.4, rmin=2.0):
    """L-bracket: the 2D stress benchmark (Le et al. 2010 and most stress
    papers since). Clamped along the top edge, loaded down on the top of the
    free arm's tip, the re-entrant corner is where compliance designs
    concentrate stress."""
    c = int(round(n * (1 - cut)))                 # arm width, cells
    void = np.zeros((n, 1, n), bool)
    void[c:, :, c:] = True
    tip = 6
    active = n * n - (n - c) ** 2
    return Problem(
        name="lbracket-2d", title=f"L-bracket, 2D ({n}x{n}, {n - c}x{n - c} cut out)",
        source="Le, Norato, Bruns, Ha & Tortorelli 2010, 'Stress-based topology optimization for continua' "
               "(geometry); reference design from top88.m with passive voids (Andreassen et al. 2011, sec. 5.2)",
        dims=(n, 1, n), volfrac=volfrac_active * active / (n * n), rmin=rmin, plane=True, void=void,
        fixed=[(_nodes(range(c + 1), 0, n), "xz")],
        loads=[(_nodes(n, 0, range(c - tip + 1, c + 1)), "z", -1.0 / tip)],
        references=["top88-ft2"], stress=True)


def cantilever_3d(nx=60, ny=4, nz=20, volfrac=0.3, rmin=1.5):
    """top3d's default problem: clamped at x = 0, a unit load per node down
    along the bottom edge of the free end."""
    return Problem(
        name="cantilever-3d", title=f"Cantilever beam, 3D ({nx}x{nz}x{ny})",
        source="Liu & Tovar 2014, 'An efficient 3D topology optimization code written in Matlab' "
               "(top3d.m, default load case, top3d(60,20,4,0.3,3,1.5))",
        dims=(nx, ny, nz), volfrac=volfrac, rmin=rmin,
        fixed=[(_nodes(0, range(ny + 1), range(nz + 1)), "xyz")],
        loads=[(_nodes(nx, range(ny + 1), 0), "z", -1.0)],
        references=["top3d"])


def bridge_3d(nx=48, ny=12, nz=16, volfrac=0.15, rmin=1.5):
    """A deck bridge: pinned along the bottom edges at both ends, a uniform
    load over the whole top face."""
    top = _nodes(range(nx + 1), range(ny + 1), nz)
    return Problem(
        name="bridge-3d", title=f"Bridge, 3D ({nx}x{ny}x{nz}, deck load)",
        source="Classic 3D bridge benchmark (e.g. Aage et al. 2017 giga-voxel bridge, coarse); "
               "reference design from top3d.m with these supports and loads",
        dims=(nx, ny, nz), volfrac=volfrac, rmin=rmin,
        fixed=[(_nodes([0, nx], range(ny + 1), 0), "xyz")],
        loads=[(top, "z", -100.0 / len(top))],
        references=["top3d"], symmetry=["x", "y"])


PROBLEMS = {p.name: p for p in (mbb_2d(), lbracket_2d(), cantilever_3d(), bridge_3d())}


def get(name: str) -> Problem:
    return PROBLEMS[name]
