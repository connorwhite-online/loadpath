"""
A part is one piece: the state that makes it so, as a requirement the
optimizer holds.

Every load case is solved with every support held at once. Two halves that
each stand on their own screws then carry their own loads to their own
supports, and nothing in those cases asks one half to hold the other: the
compliance design keeps them apart, however much material it is given (a
VESA plate as two straps at 30, 39, 52 and 70%; a Raspberry Pi wall mount as
two brackets, its board's load split evenly onto each pair of standoffs).

What does ask it is a state the load cases leave out: the part on its own,
handled and installed. It is carried by one of its interfaces (the hand, the
first screw driven) while the part itself locates every other one (the next
screw's hole over its hole in the wall, the standoffs under the board's
holes). In that state a part in two pieces drops its second piece, and a
part whose pieces are joined by a grey film locates nothing.

So the requirement: held at one support group and free at the others, each
other group pushed along each axis by a unit force, the part's compliance is
at most that of the least the process makes between them, a round bar of
the minimum member running straight from the held group to each of the
others (axial L / EA plus two directions of cantilever bending L^3 / 3EI,
summed over the groups). Any part joined by real material meets it by a
wide margin; one in pieces misses it by the void's stiffness ratio (E_MIN,
a factor of about a million). The magnitude cancels: the bound and the
compliance are both read under the same unit force.

The optimizer (simp.py) holds it as the other limits it carries as
penalties: on the logarithm of the compliance's excess over the bound, so
its pull is the same relative size whether the part is in pieces or just
over, with a weight raised while the design is over and kept, not reset,
once it is under. A design that meets it is not touched at all (no extra
material for parts that are one piece on their own); a run only pays the
second solve when it is switched on (study.py: when a run comes out in
pieces on separate supports, or the study says `"onePiece": true`).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage

_CONN26 = np.ones((3, 3, 3), bool)

# The penalty's weight while the design is over the bound: started at this
# (the pull of the logarithm, relative to the compliance's own) and raised
# by RAISE an iteration up to CAP. Raising keeps a design that hovers just
# over the bound from settling there.
START = 0.5
RAISE = 1.25
CAP = 20.0


def support_groups(domain) -> list[np.ndarray]:
    """The node sets that hold the part, one per separate contact (a screw's
    bore, a seat face, a pin): what holds it in any dof, by a pin's spring
    or as a seat, split into pieces that touch on the node grid. Largest
    first."""
    mesh = domain.mesh
    held = np.zeros(mesh.n_nodes, bool)
    held |= np.asarray(domain.fixed).reshape(-1, 3).any(axis=1)
    pins = getattr(domain, "pin_springs", None)
    if pins is not None:
        rows = np.unique(pins.tocoo().row) // 3
        held[rows] = True
    for seat in getattr(domain, "seats", None) or []:
        held[np.asarray(seat[0], int)] = True
    if not held.any():
        return []
    ijk = mesh.node_ijk
    dims = tuple(int(v) for v in ijk.max(axis=0) + 1)
    g = np.zeros(dims, bool)
    g[tuple(ijk[held].T)] = True
    lbl, n = ndimage.label(g, structure=_CONN26)
    on = lbl[tuple(ijk.T)]
    groups = [np.flatnonzero(on == k) for k in range(1, n + 1)]
    return sorted(groups, key=len, reverse=True)


def split_supports(domain, xphys: np.ndarray, groups: list | None = None) -> int:
    """How many separate pieces of the design (solid at 0.5, faces, edges
    and corners joining) the support groups stand on: 1 for a part in one
    piece, 0 when no piece holds a support."""
    groups = support_groups(domain) if groups is None else groups
    if len(groups) < 2:
        return min(len(groups), 1)
    mesh = domain.mesh
    g = np.zeros(domain.dims, bool)
    ijk = tuple(mesh.elem_ijk.T)
    g[ijk] = np.asarray(xphys) > 0.5
    lbl, _ = ndimage.label(g, structure=_CONN26)
    el = lbl[ijk]
    seen = set()
    for nodes in groups:
        on = np.zeros(mesh.n_nodes, bool)
        on[nodes] = True
        cells = on[mesh.enode].any(axis=1)
        seen |= set(int(v) for v in np.unique(el[cells]) if v > 0)
    return len(seen)


@dataclasses.dataclass
class OnePiece:
    """The handled state on a domain: `domain` holds the part at `held` and
    pushes each other group along x, y and z (three cases); `bound` is the
    compliance of the minimum-member bars (unit modulus, the optimizer's
    units)."""
    domain: object
    bound: float
    groups: list
    held: np.ndarray


def handled(domain, min_member: float) -> OnePiece | None:
    """The handled state for `domain` (see the module's note), or None when
    the part stands on one support group (nothing to locate)."""
    groups = support_groups(domain)
    if len(groups) < 2:
        return None
    mesh = domain.mesh
    held, others = groups[0], groups[1:]
    fixed = np.zeros(mesh.n_dof, bool)
    fixed[(3 * held[:, None] + np.arange(3)).ravel()] = True
    loads = []
    for a in range(3):
        f = np.zeros(mesh.n_dof)
        for nodes in others:
            f[3 * nodes + a] += 1.0 / len(nodes)
        loads.append(f)
    on_held = np.zeros(mesh.n_nodes, bool)
    on_held[held] = True
    on_others = np.zeros(mesh.n_nodes, bool)
    for nodes in others:
        on_others[nodes] = True
    # what holds the part is gone in this state; a thin wall's or a bought
    # member's stiffness (the rest of `springs`) is still the part's own
    springs = domain.springs
    pins = getattr(domain, "pin_springs", None)
    if springs is not None and pins is not None:
        springs = (springs - pins).tocsr()
        springs.eliminate_zeros()
        if springs.nnz == 0:
            springs = None
    hd = dataclasses.replace(
        domain, fixed=fixed, loads=loads, case_names=["handled-x", "handled-y", "handled-z"],
        case_weights=[1.0, 1.0, 1.0], support_nodes=on_held, load_nodes=on_others, springs=springs,
        pin_springs=None, seats=[], gravity=None, spreads=[], thermal=None, warp=None, raw_loads=None,
        carried=None, bodies=[], inertial_cases=[], impacts=[], modal=None, blockage=None, feeding=None,
        center_of_mass=None)
    # a round bar of the minimum member from the held group to each other one
    d = float(min_member)
    area = np.pi * d * d / 4.0
    inertia = np.pi * d ** 4 / 64.0
    c_held = mesh.node_xyz[held].mean(axis=0)
    bound = 0.0
    for nodes in others:
        length = max(float(np.linalg.norm(mesh.node_xyz[nodes].mean(axis=0) - c_held)), d)
        bound += length / area + 2.0 * length ** 3 / (3.0 * inertia)
    return OnePiece(hd, bound, groups, held)
