"""
A second opinion on a delivered part, from an independent solver.

verify.py analyses the part on the same kind of voxel grid the optimizer
uses. That is fast and consistent, and it shares the optimizer's blind
spots: a staircase boundary, stress that depends on where the grid falls on
a fillet (GRIP's turret read SF 2.41 at 150k cells and 1.92 at 120k). This
module re-meshes the delivered surface into curved-edge (quadratic)
tetrahedra that follow it exactly, applies the study's supports and loads,
and solves linear elasticity with a separate code path:

- mesh: gmsh (optional dependency; GPL, called as a tool, not linked into
  loadpath's own code), 10-node tetrahedra sized from the part's thinnest
  member;
- solve: scikit-fem (BSD), P2 elasticity, direct or AMG-preconditioned CG;
- or, with `backend="ccx"` and CalculiX's `ccx` on PATH, the same mesh and
  boundary conditions written as an Abaqus-style deck and solved by
  CalculiX, the open-source reference for this kind of check.

The two analyses agreeing is the evidence; disagreeing is a finding.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .spec import Study
from .verify import STRESS_ALLOWANCE

__all__ = ["crosscheck"]


def _tet_mesh(mesh, size, order=1):
    """Tetrahedra filling a closed triangle mesh: (points, tets), 4 or 10 nodes."""
    import gmsh

    with tempfile.TemporaryDirectory() as tmp:
        stl = os.path.join(tmp, "part.stl")
        mesh.export(stl)
        gmsh.initialize(["-noenv"])
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.merge(stl)
            gmsh.model.mesh.removeDuplicateNodes()   # STL repeats every vertex per triangle
            # keep the triangles as given: they already are the exact surface,
            # and the volume mesh grows inward from them
            surfs = gmsh.model.getEntities(2)
            loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.25)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)           # Delaunay (HXT refuses a discrete STL surface)
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)     # mid-nodes on the straight edge: the surface is facets
            gmsh.model.mesh.generate(3)
            tags, coords, _ = gmsh.model.mesh.getNodes()
            pts = coords.reshape(-1, 3)
            index = {int(t): i for i, t in enumerate(tags)}
            types, _, nodes = gmsh.model.mesh.getElements(3)
            n_per, code = (4, 4) if order == 1 else (10, 11)
            k = list(types).index(code)
            tets = np.array([index[int(t)] for t in nodes[k]]).reshape(-1, n_per)
            used = np.unique(tets)
            remap = -np.ones(len(pts), int)
            remap[used] = np.arange(len(used))
            pts, tets = pts[used], remap[tets]
        finally:
            gmsh.finalize()
    return pts, tets


def _region_nodes(study, shape, pts, tol):
    return np.nonzero(shape.sdf(pts) <= tol)[0]


def _load_vectors(study, pts, surf, tol):
    """Per load case: nodal force array (n, 3), and the load node indices."""
    cases = []
    for case in study.load_cases:
        f = np.zeros((len(pts), 3))
        nodes_all = []
        for ld in case.loads:
            shape = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
            nodes = np.intersect1d(_region_nodes(study, shape, pts, tol), surf)
            if len(nodes) == 0:
                raise ValueError(f"load in case {case.name!r} touches no surface node of the part")
            nodes_all.append(nodes)
            if np.any(ld.force):
                f[nodes] += np.asarray(ld.force, float) / len(nodes)
            if ld.moment is not None and np.any(ld.moment):
                # a couple as a rigid rotation's forces about the nodes' centroid
                r = pts[nodes] - pts[nodes].mean(axis=0)
                m = np.asarray(ld.moment, float)
                j = np.einsum("ij,ij->", r, r) * np.eye(3) - r.T @ r
                w = np.linalg.solve(j + 1e-9 * np.trace(j) * np.eye(3), m)
                f[nodes] += np.cross(w, r)
        cases.append((case.name, f, np.unique(np.concatenate(nodes_all))))
    return cases


def _solve_skfem(pts, tets, E, nu, study, fixed_shapes, tol, loads):
    """Quadratic (P2) elasticity on the linear tet mesh: skfem adds the
    edge nodes. Supports clamp every dof on the boundary facets inside a
    support region; loads act at the region's surface vertices."""
    from skfem import Basis, ElementTetP2, ElementVector, MeshTet, asm, condense, solve
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    m = MeshTet(pts.T.copy(), tets.T.copy())
    ib = Basis(m, ElementVector(ElementTetP2()))
    k = asm(linear_elasticity(*lame_parameters(E, nu)), ib)

    def in_support(x):
        x = np.asarray(x).reshape(3, -1).T
        return np.min([sh.sdf(x) for sh in fixed_shapes], axis=0) <= tol
    fixed = ib.get_dofs(lambda x: in_support(x)).all()
    if len(fixed) == 0:
        raise ValueError("no boundary facet of the part lies in any support region")
    nd = ib.nodal_dofs                     # (3, n_vertices)
    us = []
    for name, f, _ in loads:
        rhs = np.zeros(ib.N)
        for c in range(3):
            rhs[nd[c]] = f[:, c]
        u = solve(*condense(k, rhs, D=fixed), solver=_amg_cg if ib.N > 150_000 else None)
        us.append((u, np.stack([u[nd[0]], u[nd[1]], u[nd[2]]], axis=1)))
    return m, ib, us


def _amg_cg(a, b, **_):
    import pyamg
    import scipy.sparse.linalg as spla
    ml = pyamg.smoothed_aggregation_solver(a.tocsr(), coarse_solver="splu")
    x, _info = spla.cg(a, b, M=ml.aspreconditioner(), rtol=1e-8, maxiter=3000)
    return x


def _nodal_von_mises(m, u, E, nu):
    """Von Mises at each vertex: from the P2 field's gradient at the
    element's own vertices, averaged over the elements sharing it."""
    from skfem import Basis, ElementTetP2, ElementVector
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    ref = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    b = Basis(m, ElementVector(ElementTetP2()), quadrature=(ref, np.full(4, 1 / 24)))
    grad = b.interpolate(u).grad            # (3, 3, n_elem, 4)
    eps = 0.5 * (grad + np.swapaxes(grad, 0, 1))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    s = 2 * mu * eps
    for i in range(3):
        s[i, i] = s[i, i] + lam * tr
    vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                 + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2))   # (n_elem, 4)
    n = m.p.shape[1]
    acc = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(acc, m.t.T.ravel(), vm.ravel())
    np.add.at(cnt, m.t.T.ravel(), 1)
    return acc / np.maximum(cnt, 1)


def crosscheck(study: Study, mesh, size: float | None = None, contact_band: float | None = None,
               backend: str = "skfem") -> dict:
    """Independent analysis of `mesh` (trimesh) under the study's cases.
    Returns {safetyFactor, cases: [{name, peakStress, peakAt, deflection}],
    nodes, elements, seconds, backend}."""
    t0 = time.time()
    mat = study.material
    size = size or max(0.75, 0.5 * (study.min_member or 3.0))
    band = contact_band if contact_band is not None else 2.0 * size
    pts, tets = _tet_mesh(mesh, size, order=2 if backend == "ccx" else 1)
    from scipy.spatial import cKDTree
    # surface nodes: on the input surface (to a hair)
    surf = np.nonzero(np.abs(mesh.nearest.signed_distance(pts)) < 0.05 * size)[0]
    tol = 0.6 * size
    fixed_shapes = [sup.shape if sup.shape is not None else study.region_shape(sup.region)[0]
                    for sup in study.supports]
    fixed_nodes = np.unique(np.concatenate([np.intersect1d(_region_nodes(study, sh, pts, tol), surf)
                                            for sh in fixed_shapes]))
    if len(fixed_nodes) == 0:
        raise ValueError("no surface node of the part lies in any support region")
    loads = _load_vectors(study, pts, surf, tol)

    if backend == "ccx":
        cases = _solve_ccx(study, pts, tets, fixed_nodes, loads)
    else:
        m, ib, us = _solve_skfem(pts, tets, mat.E, mat.nu, study, fixed_shapes, tol, loads)
        contact = np.unique(np.concatenate([fixed_nodes] + [l[2] for l in loads]))
        far = cKDTree(pts[contact]).query(pts, distance_upper_bound=band)[0] == np.inf
        cases = []
        for (name, f, lnodes), (u_full, u) in zip(loads, us):
            vm = _nodal_von_mises(m, u_full, mat.E, mat.nu)
            i = int(np.argmax(np.where(far, vm, -1)))
            cases.append({"name": name, "peakStress": float(vm[i]), "peakAt": pts[i].tolist(),
                          "deflection": float(np.linalg.norm(u[lnodes].mean(axis=0))),
                          "maxDisplacement": float(np.linalg.norm(u, axis=1).max())})
    peak = max(c["peakStress"] for c in cases)
    return {"backend": backend, "safetyFactor": mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else float("inf"),
            "safetyFactorNoAllowance": mat.strength / peak if peak > 0 else float("inf"),
            "cases": cases, "nodes": int(len(pts)), "elements": int(len(tets)), "elementSize": size,
            "contactBand": band, "seconds": time.time() - t0}


def _solve_ccx(study, pts, tets, fixed_nodes, loads):
    """Write an Abaqus-style deck and run CalculiX (ccx on PATH)."""
    exe = shutil.which("ccx")
    if exe is None:
        raise RuntimeError("CalculiX (ccx) is not on PATH; install it (Debian: apt install calculix-ccx) "
                           "or use backend='skfem'")
    mat = study.material
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, f, lnodes in loads:
            deck = os.path.join(tmp, "job")
            with open(deck + ".inp", "w") as fh:
                fh.write("*NODE\n")
                for i, p in enumerate(pts):
                    fh.write(f"{i + 1},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")
                fh.write("*ELEMENT,TYPE=C3D10,ELSET=PART\n")
                for j, t in enumerate(tets):
                    # gmsh and CalculiX share the 10-node ordering up to the last two mid-edge nodes
                    tt = t[[0, 1, 2, 3, 4, 5, 6, 7, 9, 8]] + 1
                    fh.write(f"{j + 1}," + ",".join(str(v) for v in tt) + "\n")
                fh.write("*NSET,NSET=FIX\n")
                for i in fixed_nodes:
                    fh.write(f"{i + 1}\n")
                fh.write(f"*MATERIAL,NAME=M\n*ELASTIC\n{mat.E},{mat.nu}\n*SOLID SECTION,ELSET=PART,MATERIAL=M\n")
                fh.write("*BOUNDARY\nFIX,1,3\n*STEP\n*STATIC\n*CLOAD\n")
                for i in np.nonzero(np.abs(f).sum(axis=1) > 0)[0]:
                    for c in range(3):
                        if f[i, c]:
                            fh.write(f"{i + 1},{c + 1},{f[i, c]:.9g}\n")
                fh.write("*NODE FILE\nU\n*EL FILE\nS\n*END STEP\n")
            subprocess.run([exe, "job"], cwd=tmp, check=True, capture_output=True, timeout=3600)
            u, vm = _read_frd(deck + ".frd", len(pts))
            i = int(np.argmax(vm))
            out.append({"name": name, "peakStress": float(vm[i]), "peakAt": pts[i].tolist(),
                        "deflection": float(np.linalg.norm(u[lnodes].mean(axis=0))),
                        "maxDisplacement": float(np.linalg.norm(u, axis=1).max())})
    return out


def _read_frd(path, n):
    """Displacements and nodal von Mises from a CalculiX .frd result file."""
    u = np.zeros((n, 3))
    s = np.zeros((n, 6))
    block = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(" -4"):
                block = line.split()[1]
            elif line.startswith(" -1") and block in ("DISP", "STRESS"):
                i = int(line[3:13]) - 1
                vals = [float(line[13 + 12 * k:25 + 12 * k]) for k in range((len(line.rstrip()) - 13) // 12)]
                if block == "DISP":
                    u[i] = vals[:3]
                else:
                    s[i] = vals[:6]
            elif line.startswith(" -3"):
                block = None
    sx, sy, sz, sxy, syz, szx = s.T
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + szx ** 2))
    return u, vm
