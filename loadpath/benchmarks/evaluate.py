"""
One evaluator for every design, whoever made it.

Reference codes report the compliance of a grey, penalized density field;
loadpath delivers an almost black-and-white one. Every design is scored here
the same way, on the problem's own boundary conditions with void at 1e-9:
its penalized (p = 3) compliance as delivered, and its compliance once
thresholded to solid/void at the target volume (keeping the densest cells).
The headline `compliance` is the lower of the two, the most generous reading
of either kind of design.

Plane problems use the 4-node plane-stress element of top88 (so a reference
design scores what top88 itself would compute); 3D ones use loadpath's 8-node
hexahedron, the same element as top3d.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .problems import Problem

__all__ = ["Evaluator", "binarize"]

E_VOID = 1e-9
NU = 0.3
_AX = {"x": 0, "y": 1, "z": 2}


def _direct(k, f):
    """Sparse direct solve: CHOLMOD when scikit-sparse is installed (what
    MATLAB's backslash uses for these matrices; much faster than SuperLU in
    3D), SuperLU otherwise."""
    try:
        from sksparse.cholmod import cho_solve
    except ImportError:
        return spla.spsolve(k.tocsc(), f)
    return cho_solve(k.tocsc(), f)


def binarize(x: np.ndarray, active: np.ndarray, volfrac: float | None) -> np.ndarray:
    """Solid where x is highest, `volfrac` of the active cells; None
    thresholds at 0.5 instead."""
    out = np.zeros_like(x, dtype=float)
    xa = x[active]
    if volfrac is None:
        out[active] = (xa > 0.5).astype(float)
        return out
    k = int(round(volfrac * xa.size))
    order = np.argsort(-xa, kind="stable")
    b = np.zeros(xa.size)
    b[order[:k]] = 1.0
    out[active] = b
    return out


def _q4_ke(nu=NU):
    a11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], float)
    a12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], float)
    b11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], float)
    b12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], float)
    return 1 / (1 - nu ** 2) / 24 * (np.block([[a11, a12], [a12.T, a11]]) + nu * np.block([[b11, b12], [b12.T, b11]]))


class Evaluator:
    def __init__(self, p: Problem):
        self.p = p
        nx, ny, nz = p.dims
        if p.plane:
            # top88 numbering in grid terms: node (i, k) -> i*(nz+1) + (nz-k),
            # dofs 2n (x), 2n+1 (z up); element nodes BL, BR, TR, TL
            def node(i, k):
                return i * (nz + 1) + (nz - k)
            ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
            ii, kk = ii.ravel(), kk.ravel()                     # element order: i major, k minor
            nodes = np.stack([node(ii, kk), node(ii + 1, kk), node(ii + 1, kk + 1), node(ii, kk + 1)], axis=1)
            self.edof = np.stack([2 * nodes, 2 * nodes + 1], axis=2).reshape(-1, 8)
            self.ke = _q4_ke()
            self.n_dof = 2 * (nx + 1) * (nz + 1)
            self.cell = (ii, np.zeros_like(ii), kk)
            comp = {"x": 0, "z": 1}
            dof = lambda n, c: 2 * node(n[:, 0], n[:, 2]) + comp[c]
        else:
            from ..fem import HexMesh, hex8_ke
            ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
            ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
            mesh = HexMesh(ijk, 1.0, np.zeros(3))
            self.edof = mesh.edof
            self.ke = hex8_ke(NU)
            self.n_dof = mesh.n_dof
            self.cell = tuple(ijk.T)
            dof = lambda n, c: 3 * mesh.node_id(n) + _AX[c]
        fixed = np.zeros(self.n_dof, bool)
        for nodes, comps in p.fixed:
            for c in comps:
                fixed[dof(nodes, c)] = True
        f = np.zeros(self.n_dof)
        for nodes, c, v in p.loads:
            np.add.at(f, dof(nodes, c), v)
        self.fixed, self.f = fixed, f
        self.free = np.flatnonzero(~fixed)
        ne = len(self.edof)
        self._rows = np.repeat(self.edof, self.edof.shape[1], axis=1).ravel()
        self._cols = np.tile(self.edof, (1, self.edof.shape[1])).ravel()
        self.active = p.active[self.cell]

    def cells(self, grid: np.ndarray) -> np.ndarray:
        """Grid (nx, ny, nz) -> this evaluator's element order."""
        return np.asarray(grid, float)[self.cell]

    def solve(self, e: np.ndarray) -> np.ndarray:
        data = (e[:, None, None] * self.ke[None]).ravel()
        k = sp.csr_matrix((data, (self._rows, self._cols)), shape=(self.n_dof, self.n_dof))
        kf = k[self.free][:, self.free]
        u = np.zeros(self.n_dof)
        u[self.free] = _direct(kf, self.f[self.free])
        return u

    def compliance(self, grid: np.ndarray, penal: float = 3.0) -> float:
        x = self.cells(grid)
        u = self.solve(E_VOID + x ** penal * (1 - E_VOID))
        return float(self.f @ u)

    def von_mises(self, grid: np.ndarray) -> np.ndarray:
        """Peak-relevant von Mises per element of a black-and-white design
        (solid cells only; void reads 0). Plane problems only."""
        if not self.p.plane:
            raise NotImplementedError("von Mises scoring is for plane problems")
        x = self.cells(grid)
        u = self.solve(E_VOID + x ** 3 * (1 - E_VOID))
        b = 0.5 * np.array([[-1, 0, 1, 0, 1, 0, -1, 0],
                            [0, -1, 0, -1, 0, 1, 0, 1],
                            [-1, -1, -1, 1, 1, 1, 1, -1]], float)
        d = 1 / (1 - NU ** 2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
        s = u[self.edof] @ (d @ b).T
        vm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 - s[:, 0] * s[:, 1] + 3 * s[:, 2] ** 2)
        return np.where(x > 0.5, vm, 0.0)

    def load_cells(self, layers: int = 2) -> np.ndarray:
        """Elements within `layers` cells of a loaded node, which every
        stress score leaves out (point loads make a singular stress there)."""
        nodes = np.concatenate([n for n, _, _ in self.p.loads])
        ijk = np.stack(self.cell, axis=1)
        near = np.zeros(len(ijk), bool)
        for n in nodes:
            near |= np.all(np.abs(ijk + 0.5 - n) <= layers, axis=1)
        return near

    def score(self, grid: np.ndarray, target: float) -> dict:
        """Everything the report compares, for one design grid."""
        x = self.cells(grid)
        act = self.active
        full = np.zeros(self.p.dims)
        out = {"volume": float(x[act].mean()), "greyness": float((4 * x * (1 - x))[act].mean()),
               "compliance_grey": self.compliance(grid)}
        for key, vf in (("matched", target), ("threshold", None)):
            b = binarize(x, act, vf)
            full[self.cell] = b
            out[f"compliance_{key}"] = self.compliance(full)
            out[f"volume_{key}"] = float(b[act].mean())
            if key == "matched":
                bin_matched = full.copy()
        # the most generous reading of each design: its own penalized
        # compliance, or black and white at the target volume, whichever is
        # lower (thresholding a grey design can cut its thin members off)
        out["compliance"] = min(out["compliance_grey"], out["compliance_matched"])
        if self.p.stress:
            vm = self.von_mises(bin_matched)
            vm[self.load_cells()] = 0.0
            out["peak_von_mises"] = float(vm.max())
        return out
