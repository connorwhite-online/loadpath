"""
Patch tests for the voxel hex8 element, plain vs incompatible modes
(fem.hex8_element), on grid-aligned strips 1, 2, 3, 5 cells deep:

- cantilever under a tip shear (root clamped): nodal surface stress over the
  middle of the span against 6 P (L - x) / (w t^2), and the tip deflection
  against Timoshenko's P L^3 / 3EI + P L / (k G A);
- pure bending (a couple on the end face, the root held only on its
  symmetry plane): stress against M y / I, end rotation against M L / EI;
- tension: stress against P / A, extension against P L / EA.

Run: python moonshots/im-hex8/patch_table.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy.sparse.linalg import spsolve

from loadpath.fem import HexMesh, Stiffness, von_mises
from loadpath.verify import nodal_stress

E, NU = 1000.0, 0.3


def strip(t, w=2, slender=10):
    L = max(slender * t, 20)
    ijk = np.array([(i, j, k) for i in range(L) for j in range(t) for k in range(w)])
    return HexMesh(ijk, 1.0, np.zeros(3)), L


def face_loads(mesh, x0, traction):
    """Consistent nodal forces of a traction (function of y, z -> 3-vector)
    on the x = x0 face, 2x2 Gauss per cell face."""
    f = np.zeros(mesh.n_dof)
    xyz = mesh.node_xyz
    on = np.isclose(xyz[:, 0], x0)
    idx = {tuple(np.round(p[1:], 6)): i for i, p in zip(np.nonzero(on)[0], xyz[on])}
    g = 0.5 / np.sqrt(3.0)
    ys = sorted({k[0] for k in idx})
    zs = sorted({k[1] for k in idx})
    for y0 in ys[:-1]:
        for z0 in zs[:-1]:
            for a in (0.5 - g, 0.5 + g):
                for b in (0.5 - g, 0.5 + g):
                    tr = np.asarray(traction(y0 + a, z0 + b), float) * 0.25
                    for cy, ny in ((0, 1 - a), (1, a)):
                        for cz, nz in ((0, 1 - b), (1, b)):
                            n = idx[(round(y0 + cy, 6), round(z0 + cz, 6))]
                            f[3 * n:3 * n + 3] += ny * nz * tr
    return f


def solve(mesh, fixed, f, element):
    st = Stiffness(mesh, NU, fixed, element=element)
    k = st.assemble(np.full(mesh.n_elem, E))
    free = ~fixed
    u = np.zeros(mesh.n_dof)
    u[free] = spsolve(k[free][:, free].tocsc(), f[free])
    return st, u


def cantilever(t, element, w=2):
    mesh, L = strip(t, w)
    xyz = mesh.node_xyz
    fixed = np.repeat(np.isclose(xyz[:, 0], 0.0), 3)
    P = 1.0
    A, I = w * t, w * t ** 3 / 12.0
    yc = t / 2.0
    # parabolic shear on the tip face, resultant P along -y
    f = face_loads(mesh, L, lambda y, z: (0.0, -P * 6.0 * (yc ** 2 - (y - yc) ** 2) / (t ** 3 * w), 0.0))
    st, u = solve(mesh, fixed, f, element)
    vm = von_mises(nodal_stress(st, u, E))
    x = xyz[:, 0]
    win = (x >= 0.3 * L) & (x <= 0.7 * L)
    ratio = float(np.max(vm[win] / (6.0 * P * (L - x[win]) / (w * t ** 2))))
    tip = np.isclose(x, L)
    defl = float(-u.reshape(-1, 3)[tip, 1].mean())
    G = E / (2 * (1 + NU))
    ref = P * L ** 3 / (3 * E * I) + P * L / (5.0 / 6.0 * G * A)
    return ratio, defl / ref


def pure_bending(t, element, w=2):
    mesh, L = strip(t, w)
    xyz = mesh.node_xyz
    yc = t / 2.0
    fixed = np.zeros(mesh.n_dof, bool)
    root = np.isclose(xyz[:, 0], 0.0)
    fixed[3 * np.nonzero(root)[0]] = True                         # symmetry plane: ux = 0
    n0 = np.nonzero(root & np.isclose(xyz[:, 1], 0.0) & np.isclose(xyz[:, 2], 0.0))[0][0]
    n1 = np.nonzero(root & np.isclose(xyz[:, 1], 0.0) & np.isclose(xyz[:, 2], w))[0][0]
    fixed[3 * n0 + 1] = fixed[3 * n0 + 2] = True
    fixed[3 * n1 + 1] = True
    k = 1.0                                                       # sigma_x = k (y - yc)
    f = face_loads(mesh, L, lambda y, z: (k * (y - yc), 0.0, 0.0))
    st, u = solve(mesh, fixed, f, element)
    s = nodal_stress(st, u, E)
    top = np.isclose(xyz[:, 1], t) & (xyz[:, 0] > 0.2 * L) & (xyz[:, 0] < 0.8 * L)
    sx = float(s[top, 0].mean() / (k * yc))
    M = k * w * t ** 3 / 12.0
    I = w * t ** 3 / 12.0
    end = np.isclose(xyz[:, 0], L)
    ux = u.reshape(-1, 3)[end, 0]
    slope = np.polyfit(xyz[end, 1] - yc, ux, 1)[0]
    return sx, float(slope / (M * L / (E * I)))


def tension(t, element, w=2):
    mesh, L = strip(t, w)
    xyz = mesh.node_xyz
    fixed = np.zeros(mesh.n_dof, bool)
    root = np.isclose(xyz[:, 0], 0.0)
    fixed[3 * np.nonzero(root)[0]] = True
    n0 = np.nonzero(root & np.isclose(xyz[:, 1], 0.0) & np.isclose(xyz[:, 2], 0.0))[0][0]
    n1 = np.nonzero(root & np.isclose(xyz[:, 1], 0.0) & np.isclose(xyz[:, 2], w))[0][0]
    fixed[3 * n0 + 1] = fixed[3 * n0 + 2] = True
    fixed[3 * n1 + 1] = True
    P = 1.0
    A = w * t
    f = face_loads(mesh, L, lambda y, z: (P / A, 0.0, 0.0))
    st, u = solve(mesh, fixed, f, element)
    vm = von_mises(nodal_stress(st, u, E))
    end = np.isclose(xyz[:, 0], L)
    return float(vm.max() / (P / A)), float(u.reshape(-1, 3)[end, 0].mean() / (P * L / (E * A)))


def main():
    print("| t (cells) | element | cantilever stress | cantilever tip defl | pure-bending stress | pure-bending rotation | tension stress | tension extension |")
    print("|---|---|---|---|---|---|---|---|")
    for t in (1, 2, 3, 5):
        for el in ("plain", "im"):
            c = cantilever(t, el)
            b = pure_bending(t, el)
            te = tension(t, el)
            print(f"| {t} | {el} | {c[0]:.3f} | {c[1]:.3f} | {b[0]:.3f} | {b[1]:.3f} | {te[0]:.3f} | {te[1]:.3f} |", flush=True)


if __name__ == "__main__":
    main()
