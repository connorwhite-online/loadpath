"""
Pressure on the walls of a cavity: a boosted intake plenum, a coolant
jacket, a pneumatic chamber.

A pressure load names a keep-out (the fluid's volume) and a gauge pressure
in MPa (1 bar = 0.1 MPa). It pushes on every face of material that lines
that keep-out, along the face's normal, out of the fluid.

On the voxel grid the lined surface is the set of element faces between
material and cavity cells, a staircase. That is exact where it matters: a
uniform pressure on a closed staircase sums to zero force and zero moment,
like on the true surface (both are p times the integral of n dA, and of
x cross n dA, over a closed surface), and each face's share of p A sits at
its centre. So a sealed vessel's load is self-equilibrating to rounding,
and a study whose only loads are pressures needs no supports: the solver
holds it by six statically determinate constraints at three nodes on the
wall (`stabilize`), which take no reaction from a balanced load and only
remove the rigid-body motion.

Cavity cells are the empty cells within half a cell of the keep-out, plus
every empty pocket the material seals off that reaches into the keep-out.
The first rule finds the wet surface of an open duct (whose cavity runs out
of the grid at its ends); the second keeps a sealed cavity's surface closed
where a cell only grazes the shape, so its load balances exactly.

The wall is fixed geometry (spec.py pins a `wall` round every pressurized
keep-out), so the load does not depend on the design and the optimizer's
sensitivities need no pressure term: the pressure boundary of a housing is
its flow path, which aerodynamics sets, not the optimizer.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .fem import CORNERS, HexMesh

__all__ = ["cavity_faces", "pressure_force", "stabilize"]

_DIRS = [(a, s) for a in range(3) for s in (-1, 1)]
# Tolerance for "within half a cell": cells partly inside a keep-out are
# left out of the design (domain.build_domain), and those have a sampling
# point inside, so their centres are within 0.43 cells of the shape.
_REACH = 0.5


def _occupancy(mesh: HexMesh):
    ijk = mesh.elem_ijk
    lo = ijk.min(axis=0) - 1
    dims = ijk.max(axis=0) - lo + 2
    occ = np.zeros(dims, bool)
    occ[tuple((ijk - lo).T)] = True
    return occ, lo


def _cavity_cells(mesh: HexMesh, shape, occ, lo):
    """Bool grid (occupancy's frame): the empty cells the fluid fills."""
    pitch, origin = mesh.pitch, mesh.origin
    dims = np.asarray(occ.shape)
    slo, shi = shape.bounds()
    a = np.clip(np.floor((slo - origin) / pitch).astype(int) - lo - 1, 0, dims)
    b = np.clip(np.ceil((shi - origin) / pitch).astype(int) - lo + 2, 0, dims)
    cav = np.zeros(occ.shape, bool)
    inside = np.zeros(occ.shape, bool)
    if np.any(b <= a):
        return cav
    box = tuple(slice(a[i], b[i]) for i in range(3))
    idx = np.argwhere(~occ[box])
    if len(idx) == 0:
        return cav
    centers = origin + (idx + a + lo + 0.5) * pitch
    d = shape.sdf(centers)
    near = idx[d < _REACH * pitch] + a
    cav[tuple(near.T)] = True
    deep = idx[d < 0.0] + a
    inside[tuple(deep.T)] = True
    # sealed pockets: empty components that never reach the grid's border
    lbl, n = ndimage.label(~occ)
    if n:
        border = np.unique(np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(),
                                           lbl[:, -1].ravel(), lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()]))
        seeds = np.unique(lbl[inside])
        sealed = np.setdiff1d(seeds[seeds > 0], border)
        if len(sealed):
            cav |= np.isin(lbl, sealed)
    return cav


def cavity_faces(mesh: HexMesh, shape):
    """Wetted element faces: list of (element indices, axis, sign), the
    cavity lying on the `sign` side of each element along `axis`."""
    occ, lo = _occupancy(mesh)
    cav = _cavity_cells(mesh, shape, occ, lo)
    out = []
    rel = mesh.elem_ijk - lo
    for a, s in _DIRS:
        nb = rel.copy()
        nb[:, a] += s
        hit = np.nonzero(cav[tuple(nb.T)])[0]
        if len(hit):
            out.append((hit, a, s))
    return out


def pressure_force(mesh: HexMesh, shape, pressure: float):
    """Nodal forces (per dof) of `pressure` (MPa) in the cavity `shape` on
    the material lining it, and the bool mask of the nodes it acts on."""
    f = np.zeros((mesh.n_nodes, 3))
    nodes = np.zeros(mesh.n_nodes, bool)
    share = pressure * mesh.pitch ** 2 / 4.0
    for elems, a, s in cavity_faces(mesh, shape):
        corners = np.nonzero(CORNERS[:, a] == (1 if s > 0 else 0))[0]
        nid = mesh.enode[np.ix_(elems, corners)].ravel()
        # out of the fluid: against the direction the cavity lies in
        np.add.at(f[:, a], nid, -s * share)
        nodes[nid] = True
    return f.ravel(), nodes


def stabilize(mesh: HexMesh, nodes: np.ndarray):
    """Six statically determinate constraints on three of `nodes` (a 3-2-1
    support): the node lowest in x held in x, y and z; the node highest in x
    held across x; a third, farthest from the line between them, held in the
    one direction that stops rotation about that line. A self-equilibrating
    load takes no reaction from them. Returns (fixed dofs, support nodes)."""
    idx = np.nonzero(nodes)[0]
    if len(idx) < 3:
        raise ValueError("a pressure-only study needs a wetted wall to hold it by; the cavity touches no material")
    xyz = mesh.node_xyz[idx]
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    ia, ib = order[0], order[-1]
    pa, pb = xyz[ia], xyz[ib]
    ab = pb - pa
    if np.linalg.norm(ab) <= 0:
        raise ValueError("pressure-only study: the wetted wall is a single point")
    lever = np.cross(ab / np.linalg.norm(ab), xyz - pa)
    ic = int(np.argmax(np.linalg.norm(lever, axis=1)))
    fixed = np.zeros(mesh.n_dof, bool)
    na, nb, nc = idx[ia], idx[ib], idx[ic]
    fixed[3 * na:3 * na + 3] = True
    across = [i for i in range(3) if i != int(np.argmax(np.abs(ab)))]
    for i in across:
        fixed[3 * nb + i] = True
    # the one direction at C that, with the five above, leaves no rigid
    # motion: try them in order of C's lever about A-B
    ac = xyz[ic] - pa

    def rigid_rank(k):
        rows = []
        for n, p in ((na, pa), (nb, pb), (nc, xyz[ic])):
            for i in range(3):
                if fixed[3 * n + i] or (n == nc and i == k):
                    r = p - pa
                    t = np.zeros(6)
                    t[i] = 1.0
                    # rotation w about A moves p by w x r: d(u_i)/d(w)
                    e = np.zeros(3)
                    e[i] = 1.0
                    t[3:] = np.cross(r, e)
                    rows.append(t)
        return np.linalg.matrix_rank(np.array(rows), tol=1e-9 * max(1.0, np.abs(ac).max() ** 2))

    for k in np.argsort(-np.abs(np.cross(ab, ac))):
        if rigid_rank(int(k)) == 6:
            fixed[3 * nc + int(k)] = True
            break
    else:
        raise ValueError("pressure-only study: the wetted wall is too flat to hold without supports; add one")
    sup = np.zeros(mesh.n_nodes, bool)
    sup[[na, nb, nc]] = True
    return fixed, sup
