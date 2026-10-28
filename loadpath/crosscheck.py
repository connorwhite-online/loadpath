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
- solve: quadratic (P2) tetrahedra assembled here, vectorized, solved by CG
  with a p-multigrid preconditioner (pyamg on the linear problem);
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


def _tet_mesh(mesh, size, order=1, size_at=None):
    """Tetrahedra filling a closed triangle mesh: (points, tets), 4 or 10
    nodes. Boolean output carries near-coincident sliver facets that gmsh's
    boundary recovery rejects ("overlapping facets", "two facets
    intersect"); which tolerance clears them varies by part, so on failure
    the surface is cleaned with manifold's simplify at growing tolerances
    (0.05 mm moved nothing visible on the GRIP wrist yoke) and tried again."""
    from .shapes import manifold_from_trimesh, trimesh_from_manifold

    # (Decimating a fine STL to the element size first halved the time and
    # under-read the yoke's peak by a third: the surface resolution is part
    # of the answer. Keep it.)
    last = None
    for eps in (0.0, 0.03, 0.05, 0.08, 0.12):
        m = mesh if eps == 0.0 else trimesh_from_manifold(manifold_from_trimesh(mesh).simplify(eps))
        for algo in (1, 10):
            try:
                return _tet_mesh_once(m, size, order, algo, size_at)
            except Exception as ex:  # noqa: BLE001 - gmsh raises bare Exception
                last = ex
    raise RuntimeError(f"could not tetrahedralize the part: {last}")


def _tet_mesh_once(mesh, size, order, algo, size_at=None):
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
            if size_at is not None:
                # element size as a function of position (adaptive refinement)
                gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: min(lc, size_at(x, y, z)))
            gmsh.option.setNumber("Mesh.Algorithm3D", algo)        # 1 Delaunay, 10 HXT
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


# 4-point Gauss rule on a tetrahedron (exact for degree 2: a P2 stiffness
# integrand is degree 2), in barycentric coordinates.
_QA, _QB = 0.5854101966249685, 0.1381966011250105
_QUAD = np.array([[_QA, _QB, _QB, _QB], [_QB, _QA, _QB, _QB], [_QB, _QB, _QA, _QB], [_QB, _QB, _QB, _QA]])
_EDGES = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])


def _p2_mesh(pts, tets):
    """Quadratic tets on the linear mesh: edge mid-nodes appended after the
    vertices. Returns (nodes, conn10) with conn10 = 4 vertices + 6 edges in
    _EDGES order."""
    e = np.sort(tets[:, _EDGES].reshape(-1, 2), axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    mids = 0.5 * (pts[uniq[:, 0]] + pts[uniq[:, 1]])
    conn = np.hstack([tets, len(pts) + inv.reshape(-1, 6)])
    return np.vstack([pts, mids]), conn, uniq


def _grads(nodes, conn):
    """Per element: volume and the constant gradients of the 4 barycentrics."""
    x = nodes[conn[:, :4]]                         # (ne, 4, 3)
    j = (x[:, 1:] - x[:, :1]).transpose(0, 2, 1)   # columns = edge vectors
    jinv = np.linalg.inv(j)                        # rows: grad L1..L3
    g = np.empty((len(conn), 4, 3))
    g[:, 1:] = jinv
    g[:, 0] = -jinv.sum(axis=1)
    vol = np.abs(np.linalg.det(j)) / 6.0
    return vol, g


def _shape_grads(g, lam):
    """Gradients of the 10 P2 shape functions at barycentric point `lam`."""
    ne = len(g)
    d = np.empty((ne, 10, 3))
    for i in range(4):
        d[:, i] = (4.0 * lam[i] - 1.0) * g[:, i]
    for k, (a, b) in enumerate(_EDGES):
        d[:, 4 + k] = 4.0 * (lam[b] * g[:, a] + lam[a] * g[:, b])
    return d


def _b_matrices(d):
    """Strain-displacement matrices (ne, 6, 30) from shape gradients (ne, 10, 3);
    dofs are node-interleaved (u0x, u0y, u0z, u1x, ...)."""
    ne = len(d)
    b = np.zeros((ne, 6, 30))
    for n in range(10):
        dx, dy, dz = d[:, n, 0], d[:, n, 1], d[:, n, 2]
        c = 3 * n
        b[:, 0, c], b[:, 1, c + 1], b[:, 2, c + 2] = dx, dy, dz
        b[:, 3, c], b[:, 3, c + 1] = dy, dx
        b[:, 4, c + 1], b[:, 4, c + 2] = dz, dy
        b[:, 5, c], b[:, 5, c + 2] = dz, dx
    return b


def _elastic(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] += 2 * mu
    c[np.arange(3, 6), np.arange(3, 6)] = mu
    return c


def _assemble(nodes, conn, E, nu, chunk=20_000):
    """Global stiffness, vectorized over elements in chunks (skfem's
    assembly took 117 of 644 s on the wrist yoke)."""
    import scipy.sparse as sp
    c = _elastic(E, nu)
    n_dof = 3 * len(nodes)
    rows, cols, vals = [], [], []
    for s0 in range(0, len(conn), chunk):
        cc = conn[s0:s0 + chunk]
        vol, g = _grads(nodes, cc)
        ke = np.zeros((len(cc), 30, 30))
        for q in _QUAD:
            b = _b_matrices(_shape_grads(g, q))
            ke += np.einsum("eki,kl,elj->eij", b, c, b, optimize=True) * (vol / 4.0)[:, None, None]
        dof = (3 * cc[:, :, None] + np.arange(3)).reshape(len(cc), 30)
        rows.append(np.repeat(dof, 30, axis=1).ravel())
        cols.append(np.tile(dof, (1, 30)).ravel())
        vals.append(ke.ravel())
    k = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_dof, n_dof))
    return k.tocsr()


def _solve(nodes, conn, E, nu, fixed_nodes, loads, n_vertices, edges):
    """P2 elasticity by CG with a p-multigrid preconditioner: symmetric
    Gauss-Seidel on the quadratic system, and a coarse correction on the
    linear (vertex-only) problem, Galerkin-projected and solved by node-
    blocked smoothed-aggregation AMG with the six rigid-body modes.
    Aggregation AMG coarsens the quadratic system's edge nodes badly: on the
    wrist yoke it took 166 iterations directly and 82 this way (22 s vs 38 s
    per case)."""
    import pyamg
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from pyamg.relaxation.relaxation import gauss_seidel

    k = _assemble(nodes, conn, E, nu)
    n = len(nodes)
    fixed = np.zeros(n, bool)
    fixed[fixed_nodes] = True
    free_n = np.nonzero(~fixed)[0]
    free = (3 * free_n[:, None] + np.arange(3)).ravel()
    kff = k[free][:, free].tocsr().astype(np.float64)
    # P1 -> P2 prolongation: vertices copy, edge nodes average their ends
    nv = n_vertices
    rows, cols, vals = [], [], []
    for c in range(3):
        rows.append(3 * np.arange(nv) + c)
        cols.append(3 * np.arange(nv) + c)
        vals.append(np.ones(nv))
        for e in range(2):
            rows.append(3 * (nv + np.arange(len(edges))) + c)
            cols.append(3 * edges[:, e] + c)
            vals.append(np.full(len(edges), 0.5))
    p = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * nv))[free]
    used = np.asarray(p.sum(axis=0)).ravel() > 0
    p = p[:, used].tocsr()
    a1 = (p.T @ kff @ p).tocsr()
    cn = np.nonzero(used)[0]
    blocked = a1.shape[0] % 3 == 0 and np.all(np.bincount(cn // 3, minlength=nv)[np.unique(cn // 3)] == 3)
    ml = pyamg.smoothed_aggregation_solver(a1.tobsr(blocksize=(3, 3)) if blocked else a1,
                                           B=_rigid_modes(nodes[cn // 3], cn % 3), coarse_solver="splu",
                                           max_coarse=3000)

    def precond(r):
        r = np.asarray(r, dtype=np.float64).ravel()
        x = np.zeros_like(r)
        gauss_seidel(kff, x, r, iterations=2, sweep="symmetric")
        x += p @ ml.solve(p.T @ (r - kff @ x), maxiter=1, cycle="V", tol=1e-12)
        gauss_seidel(kff, x, r, iterations=2, sweep="symmetric")
        return x
    m = spla.LinearOperator(kff.shape, matvec=precond)

    def one(f):
        fn = np.zeros((n, 3))
        fn[: len(f)] = f
        u = np.zeros(3 * n)
        # 1e-7 is six digits of stress; 1e-9 cost a third more iterations
        u[free], _info = spla.cg(kff, fn.ravel()[free], M=m, rtol=1e-7, maxiter=2000)
        return u.reshape(n, 3)
    # cases in sequence: Gauss-Seidel holds the GIL, so threads only contend
    return [one(f) for _n, f, _ in loads]


def _rigid_modes(xyz, comp):
    """The six rigid-body modes of the free dofs, the near-null space AMG
    needs to coarsen elasticity well."""
    n = len(comp)
    b = np.zeros((n, 6))
    for c in range(3):
        b[comp == c, c] = 1.0
    x, y, z = xyz.T
    # rotations about x, y, z: (0,-z,y), (z,0,-x), (-y,x,0)
    b[comp == 1, 3], b[comp == 2, 3] = -z[comp == 1], y[comp == 2]
    b[comp == 0, 4], b[comp == 2, 4] = z[comp == 0], -x[comp == 2]
    b[comp == 0, 5], b[comp == 1, 5] = -y[comp == 0], x[comp == 1]
    return b


def _nodal_von_mises(nodes, conn, u, E, nu, n_vertices):
    """Von Mises at each vertex: the P2 field's stress at each element's own
    vertices, averaged over the elements sharing it."""
    c = _elastic(E, nu)
    vol, g = _grads(nodes, conn)
    ue = u.reshape(-1, 3)[conn].reshape(len(conn), 30)
    acc = np.zeros(n_vertices)
    cnt = np.zeros(n_vertices)
    for i in range(4):
        lam = np.zeros(4)
        lam[i] = 1.0
        b = _b_matrices(_shape_grads(g, lam))
        sig = np.einsum("kl,eli,ei->ek", c, b, ue, optimize=True)
        sx, sy, sz, txy, tyz, tzx = sig.T
        vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
        np.add.at(acc, conn[:, i], vm)
        np.add.at(cnt, conn[:, i], 1)
    return acc / np.maximum(cnt, 1)


def crosscheck(study: Study, mesh, size: float | None = None, contact_band: float | None = None,
               backend: str = "native", adaptive: bool = False) -> dict:
    """Independent analysis of `mesh` (trimesh) under the study's cases.
    Returns {safetyFactor, cases: [{name, peakStress, peakAt, deflection}],
    nodes, elements, seconds, backend}."""
    t0 = time.time()
    mat = study.material
    size = size or max(0.75, 0.5 * (study.min_member or 3.0))
    band = contact_band if contact_band is not None else 2.0 * size
    if adaptive and backend != "ccx":
        # (Off by default: the volume mesh grows from the STL's own surface
        # triangles, so on the wrist yoke this cut elements by only 12% and
        # the extra pass cost more than it saved.)
        # Two passes: the whole part at twice the element size finds where
        # stress concentrates; the answer comes from a mesh that is fine
        # (`size`) within a few elements of those places and coarse
        # elsewhere. Peaks sit at a handful of fillets, so a uniformly fine
        # mesh spends most of its elements where nothing happens.
        coarse = crosscheck(study, mesh, size=2.0 * size, contact_band=band, backend=backend, adaptive=False)
        hot = np.array([h for c in coarse["cases"] for h in c["hotspots"]])
        radius = 6.0 * size
        from scipy.spatial import cKDTree as _T
        tree = _T(hot)

        def size_at(x, y, z):
            d = tree.query([x, y, z])[0]
            return size if d < radius else size + (2.5 * size - size) * min(1.0, (d - radius) / radius)
        pts, tets = _tet_mesh(mesh, 2.5 * size, size_at=size_at)
    else:
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
        nodes, conn, edges = _p2_mesh(pts, tets)
        # an edge node is held when both its ends are
        held = np.zeros(len(pts), bool)
        held[fixed_nodes] = True
        fixed_all = np.concatenate([fixed_nodes, len(pts) + np.nonzero(held[edges[:, 0]] & held[edges[:, 1]])[0]])
        us = _solve(nodes, conn, mat.E, mat.nu, fixed_all, loads, len(pts), edges)
        contact = np.unique(np.concatenate([fixed_nodes] + [l[2] for l in loads]))
        far = cKDTree(pts[contact]).query(pts, distance_upper_bound=band)[0] == np.inf
        cases = []
        for (name, f, lnodes), u in zip(loads, us):
            vm = _nodal_von_mises(nodes, conn, u.ravel(), mat.E, mat.nu, len(pts))
            i = int(np.argmax(np.where(far, vm, -1)))
            # where the stress is within half of the peak: the next pass refines there
            order = np.argsort(-np.where(far, vm, -1))[:200]
            hot = pts[order[vm[order] > 0.5 * vm[i]]]
            cases.append({"name": name, "peakStress": float(vm[i]), "peakAt": pts[i].tolist(),
                          "hotspots": hot.tolist(),
                          "deflection": float(np.linalg.norm(u[lnodes].mean(axis=0))),
                          "maxDisplacement": float(np.linalg.norm(u, axis=1).max())})
    peak = max(c["peakStress"] for c in cases)
    return {"backend": backend, "safetyFactor": mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else float("inf"),
            "safetyFactorNoAllowance": mat.strength / peak if peak > 0 else float("inf"),
            "cases": cases, "nodes": int(len(pts)), "elements": int(len(tets)), "elementSize": size,
            "contactBand": band, "adaptive": bool(adaptive and backend != "ccx"), "seconds": time.time() - t0}


def _solve_ccx(study, pts, tets, fixed_nodes, loads):
    """Write an Abaqus-style deck and run CalculiX (ccx on PATH)."""
    exe = shutil.which("ccx")
    if exe is None:
        raise RuntimeError("CalculiX (ccx) is not on PATH; install it (Debian: apt install calculix-ccx) "
                           "or use the default backend")
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
