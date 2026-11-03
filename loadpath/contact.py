"""
Contact for the large-deflection solver: the part pressing on rigid objects
and on itself.

A finger closing on a pen, a snap fit riding over its catch, a compliant
gripper whose jaws meet, a flexure that bottoms out on a stop, a curling
finger whose segments fold into each other: all of them stop being "a part
under a load" the moment two surfaces touch. nonlinear.solve_nonlinear takes
these as `contact=[...]`, next to its tendons, and each one adds a force and
a load stiffness to every Newton iteration:

- `RigidObstacle`: a rigid plane, sphere, cylinder or box the part may not
  enter. It can stay put (a palm, a table, the object being gripped) or move
  with the load factor (an indenter pushed `move` mm into the part, the
  catch a snap fit rides over). Frictionless, penalty on the part's surface
  nodes, with the obstacle's exact signed distance, normal and curvature
  (the tangent is the exact derivative of the contact force).
- `SelfContact`: the part's own surface faces pressing on each other.
  Every surface node is checked against every surface face nearby that
  is not its own neighbour (two-pass node-to-face, so neither side is
  favoured), frictionless penalty along the face's normal.

Penalty: a node pressed d into a surface pushes back k d with
k = stiffness * E * pitch (N/mm; E of the cells round it), eased in
quadratically over the first `smooth` * pitch of penetration so Newton sees
a continuous tangent as nodes come into contact. The default stiffness 20 leaves a
penetration of about F / (20 E h) under a node force F, i.e. the surface
behaves as if a skin 1/20 of a cell thick were slightly soft: well under
the voxel geometry's own error.

Validated in tests/test_contact.py against closed forms: a cantilever
landing on a rigid support (propped-cantilever reaction), two fingers of one
part pressed together (both tips meet, the reaction splits as beam theory
says), a cantilever pressed onto a frictionless floor through large
deflection (the elastica with the floor reaction, tip sliding along it), and
an indenter pushed into a cantilever.

Units: mm, N, MPa.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_TAPER = 0.2
_TAU = 0.05       # cells: how sharply a node's contact favours the nearest face behind it
__all__ = ["RigidObstacle", "SelfContact", "boundary_faces", "obstacle_from_spec"]

# hex8 corner order (fem.CORNERS: x fastest, then y, then z) and each face's
# four corners in a cycle, with its outward axis and sign
_FACES = [((0, 2, 6, 4), 0, -1.0), ((1, 3, 7, 5), 0, 1.0),
          ((0, 1, 5, 4), 1, -1.0), ((2, 3, 7, 6), 1, 1.0),
          ((0, 1, 3, 2), 2, -1.0), ((4, 5, 7, 6), 2, 1.0)]


def _penalty(g, k, s):
    """Contact pressure law p(g) >= 0 (force along the outward normal) and
    dp/dg: 0 for g >= 0 (no force before touching), k g^2 / (2 s) on
    (-s, 0), k (-g - s / 2) below -s."""
    p = np.zeros_like(g)
    dp = np.zeros_like(g)
    lin = g <= -s
    p[lin] = k[lin] * (-g[lin] - 0.5 * s)
    dp[lin] = -k[lin]
    q = (g > -s) & (g < 0)
    p[q] = k[q] * g[q] ** 2 / (2 * s)
    dp[q] = k[q] * g[q] / s
    return p, dp


def _bilinear(xi, eta):
    """Weights of a face's corners (cycle 0-1-2-3) at (xi, eta)."""
    return 0.25 * np.stack([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], 1)


def boundary_faces(mesh):
    """The mesh's surface faces: (faces (m, 4) node ids in a cycle, their
    element, outward axis, outward sign)."""
    ijk = mesh.elem_ijk
    lut = {tuple(c): i for i, c in enumerate(ijk.tolist())}
    faces, owner, axes, signs = [], [], [], []
    for corners, ax, sg in _FACES:
        nb = ijk.copy()
        nb[:, ax] += int(sg)
        has = np.fromiter((tuple(c) in lut for c in nb.tolist()), bool, len(nb))
        el = np.flatnonzero(~has)
        faces.append(mesh.enode[el][:, list(corners)])
        owner.append(el)
        axes.append(np.full(len(el), ax))
        signs.append(np.full(len(el), sg))
    return np.concatenate(faces), np.concatenate(owner), np.concatenate(axes), np.concatenate(signs)


def _node_modulus(mesh, modulus):
    """E at each node: the largest of its cells'."""
    mod = np.broadcast_to(np.asarray(modulus, float), (mesh.n_elem,))
    out = np.zeros(mesh.n_nodes)
    np.maximum.at(out, mesh.enode.ravel(), np.repeat(mod, 8))
    return out


class _Base:
    """Shared bookkeeping: surface nodes, their penalty, per-step results."""

    def __init__(self, stiffness, smooth):
        self.stiffness, self.smooth = float(stiffness), float(smooth)

    def _setup_common(self, mesh, modulus):
        self.mesh = mesh
        self.faces, self.owner, self.axes, self.signs = boundary_faces(mesh)
        self.snodes = np.unique(self.faces.ravel())
        self.kpen = self.stiffness * _node_modulus(mesh, modulus) * mesh.pitch
        self.s = self.smooth * mesh.pitch
        self.info = {}


class RigidObstacle(_Base):
    """A rigid object the part's surface may not enter.

    shape: "plane" (point, normal: the outward side is free), "sphere"
    (center, radius), "cylinder" (center, axis, radius: infinitely long),
    or "box" (min, max). `inside: True` turns sphere/cylinder/box into a
    cavity the part must stay inside (a socket, a tube, a mold).
    move: mm the object travels at full load (scaled by the load factor):
    an indenter or a moving jaw. stiffness, smooth: the penalty (module
    docstring).

    After each step `info` holds the force the object takes from the part
    (N, a 3-vector: on a fixed object, the reaction it must hold), how many
    nodes touch it, and the deepest penetration (mm)."""

    def __init__(self, shape: str, *, point=None, normal=None, center=None, radius=None, axis=None,
                 lo=None, hi=None, inside: bool = False, move=None, stiffness: float = 20.0,
                 smooth: float = 0.02, name: str | None = None):
        super().__init__(stiffness, smooth)
        if shape not in ("plane", "sphere", "cylinder", "box"):
            raise ValueError(f"RigidObstacle: shape must be plane, sphere, cylinder or box, not {shape!r}")
        self.shape, self.inside, self.name = shape, bool(inside), name
        self.move = np.zeros(3) if move is None else np.asarray(move, float)
        if shape == "plane":
            if point is None or normal is None:
                raise ValueError("RigidObstacle plane: needs point and normal")
            self.point = np.asarray(point, float)
            n = np.asarray(normal, float)
            self.normal = n / np.linalg.norm(n)
        elif shape in ("sphere", "cylinder"):
            if center is None or radius is None:
                raise ValueError(f"RigidObstacle {shape}: needs center and radius")
            self.center, self.radius = np.asarray(center, float), float(radius)
            if shape == "cylinder":
                a = np.eye(3)["xyz".index(axis)] if isinstance(axis, str) else np.asarray(
                    (0, 1, 0) if axis is None else axis, float)
                self.axis = a / np.linalg.norm(a)
        else:
            if lo is None or hi is None:
                raise ValueError("RigidObstacle box: needs min and max corners")
            self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def sdf(self, x, scale=1.0):
        """Signed distance (positive outside the object, where the part may
        be), its gradient (the outward normal) and Hessian, at points x."""
        off = scale * self.move
        n = len(x)
        H = np.zeros((n, 3, 3))
        if self.shape == "plane":
            g = (x - self.point - off) @ self.normal
            return g, np.broadcast_to(self.normal, (n, 3)).copy(), H
        if self.shape == "box":
            c = 0.5 * (self.lo + self.hi) + off
            h = 0.5 * (self.hi - self.lo)
            q = x - c
            d = np.abs(q) - h
            out = np.maximum(d, 0.0)
            od = np.linalg.norm(out, axis=1)
            outside = od > 0
            g = np.where(outside, od, d.max(axis=1))
            grad = np.zeros((n, 3))
            grad[outside] = np.sign(q[outside]) * out[outside] / od[outside, None]
            ins = ~outside
            if ins.any():
                k = d[ins].argmax(axis=1)
                grad[np.flatnonzero(ins), k] = np.sign(q[ins, k])
            # flat faces; the edges' curvature is left out (a tangent approximation)
            return self._flip(g, grad, H)
        r = x - self.center - off
        if self.shape == "cylinder":
            r = r - np.outer(r @ self.axis, self.axis)
            P = np.eye(3) - np.outer(self.axis, self.axis)
        else:
            P = np.eye(3)
        dist = np.maximum(np.linalg.norm(r, axis=1), 1e-12)
        nrm = r / dist[:, None]
        g = dist - self.radius
        H = (P[None] - np.einsum("ni,nj->nij", nrm, nrm)) / dist[:, None, None]
        return self._flip(g, nrm, H)

    def _flip(self, g, grad, H):
        if self.inside:
            return -g, -grad, -H
        return g, grad, H

    def _setup(self, mesh, modulus=1.0):
        self._setup_common(mesh, modulus)
        self._dofs = 3 * self.snodes[:, None] + np.arange(3)[None, :]

    def _load(self, u, scale):
        """Force on the part per dof, the load stiffness (-d force / du), and
        the step's report."""
        x = self.mesh.node_xyz[self.snodes] + u[self._dofs]
        g, nrm, H = self.sdf(x, scale)
        k = self.kpen[self.snodes]
        p, dp = _penalty(g, k, self.s)
        act = np.flatnonzero(p > 0)
        n = u.shape[0]
        f = np.zeros(n)
        if len(act) == 0:
            self.info = {"force": [0.0, 0.0, 0.0], "nodes": 0, "penetration": 0.0}
            return f, sp.csr_matrix((n, n)), self.info
        fa = p[act, None] * nrm[act]
        np.add.at(f, self._dofs[act].ravel(), fa.ravel())
        # f = p(g(x)) n(x): df/dx = p' n n^T + p H
        ke = -(dp[act, None, None] * np.einsum("ni,nj->nij", nrm[act], nrm[act]) + p[act, None, None] * H[act])
        rows = np.repeat(self._dofs[act], 3, axis=1).ravel()
        cols = np.tile(self._dofs[act], (1, 3)).ravel()
        kl = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n, n))
        self.info = {"force": (-fa.sum(axis=0)).tolist(), "nodes": int((g[act] < 0).sum()),
                     "penetration": float(max(0.0, -g[act].min()))}
        return f, kl, self.info


class SelfContact(_Base):
    """The part's surface pressing on itself (fingers of one part meeting, a
    curling finger folding onto its own segments, a flexure closing its
    gap).

    Each surface node is tested against the surface faces within reach that
    are not its own neighbourhood (within `rings` cells of any of the face's
    nodes through the material, which also leaves out the far side of a
    thin wall and the same surface bending round a tight curl), whose
    outward normals oppose its own, and onto which it projects. A node
    pressed into a face pushes back along the face's normal (penalty as in
    the module docstring), the face's corners taking the reaction by their
    bilinear weights at the projection. Every surface node is a "slave" and
    every surface face a "master", so contact is symmetric between the two
    sides.

    The tangent holds the face normal and the projection point fixed
    within an iteration (the classic node-to-surface linearization without
    the normal's rotation): exact for flat sliding, a slightly slower Newton
    where faces rotate quickly under contact.

    `info` after each step: the total normal force (N) carried across
    the contacts, the nodes in contact, the deepest penetration."""

    def __init__(self, stiffness: float = 20.0, smooth: float = 0.02, reach: float = 0.6, rings: int = 2):
        super().__init__(stiffness, smooth)
        self.reach, self.rings = float(reach), int(rings)

    def _setup(self, mesh, modulus=1.0, fixed=None):
        from scipy.spatial import cKDTree  # noqa: F401 - checked here so a missing scipy fails early

        self._setup_common(mesh, modulus)
        # A clamped node never moves: against it the other side is the one
        # that slides, so it is only a master (as a rigid obstacle's
        # surface is), which keeps the exact flat-face linearization there
        self._slaves = self.snodes
        if fixed is not None:
            held = np.asarray(fixed, bool).reshape(-1, 3).all(axis=1)
            self._slaves = self.snodes[~held[self.snodes]]
        # 1-ring of each node: the nodes it shares a cell with
        ne = mesh.n_elem
        inc = sp.csr_matrix((np.ones(ne * 8), (mesh.enode.ravel(), np.repeat(np.arange(ne), 8))),
                            shape=(mesh.n_nodes, ne))
        adj = (inc @ inc.T).tocsr()
        adj.data[:] = 1.0
        ring = adj
        for _ in range(self.rings - 1):
            ring = (ring @ adj).tocsr()
            ring.data[:] = 1.0
        ring = ring.tocoo()
        self._ring = np.sort(ring.row.astype(np.int64) * mesh.n_nodes + ring.col)
        # surface-node index of every mesh node, for the node normals
        self._sidx = np.full(mesh.n_nodes, -1)
        self._sidx[self.snodes] = np.arange(len(self.snodes))

    def _pairs(self, x):
        """Candidate (slave node, face) pairs whose positions are close."""
        from scipy.spatial import cKDTree

        fx = x[self.faces]                                   # (m, 4, 3)
        cen = fx.mean(axis=1)
        h = self.mesh.pitch
        tree = cKDTree(cen)
        # a node above a face lies within half a diagonal of its centre
        # (the face can stretch: allow some)
        lists = tree.query_ball_point(x[self._slaves], r=(0.71 + self.reach) * h)
        cnt = np.fromiter((len(a) for a in lists), int, len(lists))
        if cnt.sum() == 0:
            return np.zeros(0, int), np.zeros(0, int), fx
        si = np.repeat(self._slaves, cnt)
        fi = np.fromiter((j for a in lists for j in a), int, int(cnt.sum()))
        # not the face's own neighbourhood
        near = np.zeros(len(si), bool)
        nn = self.mesh.n_nodes
        for c in range(4):
            key = si.astype(np.int64) * nn + self.faces[fi, c]
            pos = np.minimum(np.searchsorted(self._ring, key), len(self._ring) - 1)
            near |= self._ring[pos] == key
        keep = ~near
        return si[keep], fi[keep], fx

    def _load(self, u, scale):
        mesh = self.mesh
        n = u.shape[0]
        x = mesh.node_xyz + u.reshape(-1, 3)
        si, fi, fx = self._pairs(x)
        f = np.zeros(n)
        empty = {"force": 0.0, "nodes": 0, "penetration": 0.0}
        if len(si) == 0:
            self.info = empty
            return f, sp.csr_matrix((n, n)), self.info
        # deformed face frames: normal from the diagonals, outward
        q = fx[fi]
        d1 = q[:, 2] - q[:, 0]
        d2 = q[:, 3] - q[:, 1]
        nf = np.cross(d1, d2)
        nf /= np.linalg.norm(nf, axis=1)[:, None]
        # the diagonals' order follows the face cycle; make it outward by
        # the reference outward normal carried by the deformation (sign
        # against the face's own cell centre)
        cell_c = x[mesh.enode[self.owner[fi]]].mean(axis=1)
        cen = q.mean(axis=1)
        flip = np.einsum("ni,ni->n", nf, cen - cell_c) < 0
        nf[flip] *= -1
        # node normals: mean of the outward normals of its own surface faces
        allq = fx
        an = np.cross(allq[:, 2] - allq[:, 0], allq[:, 3] - allq[:, 1])
        acen = allq.mean(axis=1)
        acell = x[mesh.enode[self.owner]].mean(axis=1)
        an *= np.where(np.einsum("ni,ni->n", an, acen - acell) < 0, -1.0, 1.0)[:, None]
        nn = np.zeros((len(self.snodes), 3))
        for c in range(4):
            np.add.at(nn, self._sidx[self.faces[:, c]], an)
        nn /= np.maximum(np.linalg.norm(nn, axis=1), 1e-30)[:, None]
        # facing each other: the force eases in as the node's normal turns
        # from square to the face (cos -1 .. -0.3) to edge-on (0), so a
        # node rounding a corner does not switch in one jump
        facing = np.clip(-np.einsum("ni,ni->n", nn[self._sidx[si]], nf) / 0.3, 0.0, 1.0)
        opposed = facing > 0
        # projection onto the face (parallelogram coordinates from its centre)
        xs = x[si]
        e1 = 0.25 * ((q[:, 1] + q[:, 2]) - (q[:, 0] + q[:, 3]))    # centre to the 1-2 edge
        e2 = 0.25 * ((q[:, 2] + q[:, 3]) - (q[:, 0] + q[:, 1]))
        rel = xs - cen
        a11, a12, a22 = np.einsum("ni,ni->n", e1, e1), np.einsum("ni,ni->n", e1, e2), np.einsum("ni,ni->n", e2, e2)
        b1, b2 = np.einsum("ni,ni->n", rel, e1), np.einsum("ni,ni->n", rel, e2)
        det = a11 * a22 - a12 ** 2
        xi = (a22 * b1 - a12 * b2) / det
        eta = (a11 * b2 - a12 * b1) / det
        # the gap from the bilinear surface point under the node: its
        # derivative is then the corner weights along the normal (the
        # normal's own rotation drops out to first order)
        xc, ec = np.clip(xi, -1, 1), np.clip(eta, -1, 1)
        w = _bilinear(xc, ec)
        g = np.einsum("ni,ni->n", xs - np.einsum("nc,nci->ni", w, q), nf)
        # a node may slide off a face's free edge: its force tapers to zero
        # over the last TAPER of a half-face past it, so contact does not
        # switch on and off in one jump there
        reach = np.maximum(np.abs(xi), np.abs(eta))
        ok = opposed & (reach < 1.0 + _TAPER) & (g > -0.5 * mesh.pitch) & (g < 0)
        if not ok.any():
            self.info = empty
            return f, sp.csr_matrix((n, n)), self.info
        si, fi, g, nf, xi, eta, reach, facing = (a[ok] for a in (si, fi, g, nf, xi, eta, reach, facing))
        # A node behind several faces (an edge of a flat surface, or a
        # corner it is pushed into) shares its contact among them, weighted
        # towards the nearest surface behind it (the one it has just
        # crossed, i.e. the shortest way back out): omega = exp((g - g_max) /
        # tau), smooth in the positions, so a node rounding a corner turns
        # its push round instead of switching faces in one jump. On a flat
        # surface the faces either side of an edge agree on the gap and
        # share it evenly. Each weight also carries the edge taper and the
        # facing ease-in; the shares sum to at most 1.
        xi, eta = np.clip(xi, -1, 1), np.clip(eta, -1, 1)
        t = np.clip((reach - 1.0) / _TAPER, 0.0, 1.0)
        om = (1 - t * t * (3 - 2 * t)) * facing * facing * (3 - 2 * facing)
        gmax = np.full(mesh.n_nodes, -np.inf)
        np.maximum.at(gmax, si, g)
        om = om * np.exp((g - gmax[si]) / (_TAU * mesh.pitch))
        tot = np.zeros(mesh.n_nodes)
        np.add.at(tot, si, om)
        om = om / np.maximum(tot[si], 1.0)
        k = np.minimum(self.kpen[si], self.kpen[self.faces[fi]].max(axis=1)) * om
        p, dp = _penalty(g, k, self.s)
        act = p > 0
        si, fi, g, nf, xi, eta, p, dp = (a[act] for a in (si, fi, g, nf, xi, eta, p, dp))
        if len(si) == 0:
            self.info = empty
            return f, sp.csr_matrix((n, n)), self.info
        w = _bilinear(xi, eta)
        nodes = np.concatenate([si[:, None], self.faces[fi]], axis=1)          # (a, 5)
        coef = np.concatenate([np.ones((len(si), 1)), -w], axis=1)             # dg/dx per node
        dofs = (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(len(si), 15)
        v = (coef[:, :, None] * nf[:, None, :]).reshape(len(si), 15)          # dg/du
        np.add.at(f, dofs.ravel(), (p[:, None] * v).ravel())
        ke = -dp[:, None, None] * np.einsum("ai,aj->aij", v, v)
        rows = np.repeat(dofs, 15, axis=1).ravel()
        cols = np.tile(dofs, (1, 15)).ravel()
        kl = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n, n))
        # every pair passes its force from face to node: the sum is what
        # the two sides press on each other with
        self.info = {"force": float(p.sum()), "nodes": int(len(np.unique(si[g < 0]))),
                     "penetration": float(max(0.0, -g.min()))}
        return f, kl, self.info


def obstacle_from_spec(spec: dict) -> RigidObstacle:
    """A RigidObstacle from its JSON form (the field graph's
    largeDeflection.contact.obstacles items)."""
    kind = spec.get("type")
    kw = {k: spec[k] for k in ("point", "normal", "center", "radius", "axis", "inside", "move", "stiffness",
                               "smooth", "name") if k in spec}
    if kind == "box":
        kw["lo"], kw["hi"] = spec["min"], spec["max"]
    return RigidObstacle(kind, **kw)
