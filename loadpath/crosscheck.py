"""
A second opinion on a delivered part, from an independent solver.

verify.py analyses the part on the same kind of voxel grid the optimizer
uses. That is fast and consistent, and it shares the optimizer's blind
spots: a staircase boundary, stress that depends on where the grid falls on
a fillet (the robot arm's turret read SF 2.41 at 150k cells and 1.92 at 120k). This
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

__all__ = ["crosscheck", "element_size", "unsupported"]


def _tet_mesh(mesh, size, order=1, size_at=None, mesher=None):
    """Tetrahedra filling a closed triangle mesh: (points, tets), 4 or 10
    nodes.

    fTetWild (pytetwild) by default when installed. Delivered parts come out
    of the booleans with needle triangles and near-touching facets (4k of
    60k triangles on the motor-mount eval). Constrained meshers must keep
    those triangles, so they leave flat slivers along the surface: gmsh's
    Delaunay ground for 20 minutes on that part and failed, HXT met it in
    2 s with 11k interior slivers that CG could not converge on. fTetWild
    meshes within a small envelope of the surface instead (0.1% of the
    part's diagonal, under 0.1 mm here) and came out with no element below
    a fifth of the regular tet's quality, in about a minute.

    gmsh otherwise: on failure the surface is cleaned with manifold's
    simplify at growing tolerances (0.05 mm moved nothing visible on the robot-arm
    wrist yoke) and tried again."""
    mesher = mesher or ("tetwild" if _has_tetwild() and size_at is None and order == 1 else "gmsh")
    if mesher == "tetwild":
        return _tetwild(mesh, size)
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


def element_size(study: Study) -> float:
    """The default tetrahedron edge, mm: half the thinnest member."""
    return max(0.75, 0.5 * (study.min_member or 3.0))


# Memory of a native P2 solve, measured on the robot arm's upper arm (180 cm3):
# about 10 V / h^3 tetrahedra at edge h, and 7 kB each with the matrix
# assembled in place (520k tets at 1.5 mm: 3.8 GB peak, all three cases).
# It was 20 kB a tet while assembly summed chunk matrices, held a second
# free-dof copy and formed P^T K whole: the upper arm was coarsened to
# 1.8 mm to fit a 15 GB box.
_TETS_PER_VOLUME = 10.0
_BYTES_PER_TET = 7_500


def memory_budget() -> float:
    """Bytes a tet solve may use: LOADPATH_TET_MEMORY (GB), else 40% of
    the memory free now (in a container, its limit less its use), at most 8 GB."""
    env = os.environ.get("LOADPATH_TET_MEMORY")
    if env:
        return float(env) * 1e9
    from .study import _available_bytes   # MemAvailable, and the container's free share

    avail = _available_bytes() or 8e9
    return min(8e9, 0.4 * avail)


def fit_size(volume: float, size: float, budget: float | None = None) -> float:
    """`size`, coarsened just enough that the solve fits `budget` bytes."""
    budget = memory_budget() if budget is None else budget
    need = (_TETS_PER_VOLUME * abs(volume) * _BYTES_PER_TET / max(budget, 1e8)) ** (1.0 / 3.0)
    return max(size, need)


def unsupported(study: Study) -> str | None:
    """Why the cross-check cannot stand in for verify.py on this study, or
    None when it can: it solves linear elasticity under the study's
    supports, forces, moments and bearing loads, and nothing else. A layered
    process is covered: the stiffness is the voxel verify's (study.elastic:
    transversely isotropic about the build direction) and the safety factor
    takes the same across-layer bond check (process.interlayer)."""
    if not study.supports:
        return "no supports (a self-equilibrated part)"
    if study.gravity is not None:
        return "gravity loads"
    if any(ld.pressure for c in study.load_cases for ld in c.loads):
        return "pressure loads"
    return mesher_missing()


def mesher_missing() -> str | None:
    """Why no tetrahedral mesh can be made here, or None when one can."""
    if not _has_tetwild():
        try:
            import gmsh  # noqa: F401  (availability check)
        except (ImportError, OSError):
            # OSError: the wheel is there but its shared libraries are not
            # (libGLU on a headless box), which failed every SF study
            return "no tetrahedral mesher (pip install loadpath[crosscheck])"
    return None


def _has_tetwild():
    try:
        import pytetwild  # noqa: F401  (availability check)
    except ImportError:
        return False
    return True


# fTetWild's surface envelope, as a fraction of the bounding-box diagonal
TETWILD_EPSILON = 1e-3


def _tet_cache_dir():
    """Where volume meshes are kept, by surface: LOADPATH_TET_CACHE (a
    directory, or "0" for none), else ~/.cache/loadpath/tets."""
    env = os.environ.get("LOADPATH_TET_CACHE")
    if env == "0":
        return None
    return env or os.path.join(os.path.expanduser("~"), ".cache", "loadpath", "tets")


# fTetWild meshes the part's whole bounding box at the target edge and keeps
# the inside: on the robot arm's upper arm (13% of its box) it optimized ~2M tets to
# keep 237k, 250 s of a 260 s mesh. Its global edge is set this many times
# coarser, and a sizing field asks for `size` inside the part and within
# 1.5 grid steps of its surface: the same mesh in the part (228k tets, mean
# edge 1.94 vs 1.91 mm, worst-element energy 17 vs 19) in 87 s.
#
# The sizing grid's spacing (times `size`), how many of its steps from the
# surface stay at `size` (sqrt(3) covers every corner of a grid cube the
# surface passes through), and how fast the edge grows beyond that (mm per
# mm). At 4 steps, 1.5 of them and 0.5 mm/mm, a band 9 mm thick round the
# robot-arm wrist yoke stayed fine, and in its open frame that was most of
# the box: fTetWild's last pass refined to 908k tets to keep 55k. At 2
# steps, 1.75 and 1.0 the band is 5 mm and the mesh takes 47 s, not 64 s,
# with the same tets inside the part (55k).
_COARSE_OUTSIDE = 8.0
_BG_SPACING = 2.0
_BG_NEAR = 1.75
_BG_GROWTH = 1.0


def _inside_grid(tri, axes):
    """Which points of the grid axes[0] x axes[1] x axes[2] are inside the
    closed triangle soup `tri` (n, 3, 3): one ray along +x per (y, z)
    column, its crossings found in 2D, parity counted per grid x. (trimesh's
    mesh.contains, without embree, ran past 14 GB on 7k points of the robot-arm
    wrist link and took the box down; this is a few seconds.) Columns are
    nudged off the grid by a fraction of a micron, so a ray never grazes an
    edge exactly."""
    ax, ay, az = (np.asarray(a, float) for a in axes)
    out = np.zeros((len(ax), len(ay), len(az)), bool)
    ylo, yhi = tri[:, :, 1].min(1), tri[:, :, 1].max(1)
    zlo, zhi = tri[:, :, 2].min(1), tri[:, :, 2].max(1)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    for j, y in enumerate(ay + 1.37e-4):
        rows = np.flatnonzero((ylo <= y) & (yhi >= y))
        if not len(rows):
            continue
        for k, z in enumerate(az + 2.91e-4):
            t = rows[(zlo[rows] <= z) & (zhi[rows] >= z)]
            if not len(t):
                continue
            # barycentrics of (y, z) in each triangle's y-z projection
            ab, ac = b[t] - a[t], c[t] - a[t]
            det = ab[:, 1] * ac[:, 2] - ab[:, 2] * ac[:, 1]
            ok = np.abs(det) > 1e-15
            det = np.where(ok, det, 1.0)
            py, pz = y - a[t, 1], z - a[t, 2]
            u = (py * ac[:, 2] - pz * ac[:, 1]) / det
            w = (ab[:, 1] * pz - ab[:, 2] * py) / det
            hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
            if not hit.any():
                continue
            xs = np.sort((a[t, 0] + u * ab[:, 0] + w * ac[:, 0])[hit])
            # crossings beyond each grid x: odd = inside
            out[:, j, k] = (len(xs) - np.searchsorted(xs, ax, side="right")) % 2 == 1
    return out


def _outside_coarse(mesh, v, size):
    """fTetWild's background sizing field: (points, tets, edge lengths) on
    a grid 4 elements apart, `size` inside the part and near it, growing
    with distance outside (to _COARSE_OUTSIDE times). Beyond the grid,
    fTetWild uses its (coarse) global edge. None if the part's inside test
    fails (a mesh that is not watertight)."""
    from scipy.spatial import cKDTree

    h = _BG_SPACING * size
    lo, hi = v.min(0) - 2.0 * h, v.max(0) + 2.0 * h
    axes = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
    n = [len(a) for a in axes]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    if not getattr(mesh, "is_watertight", True):
        return None
    inside = _inside_grid(np.asarray(mesh.triangles, float), axes).ravel()
    surf = np.vstack([v, np.asarray(mesh.triangles_center)])
    d = np.where(inside, 0.0, cKDTree(surf).query(G)[0])
    # every grid cell the surface passes through, and its neighbours, stays
    # at `size` at all 8 corners, so no element at the surface is coarser
    near = _BG_NEAR * h
    val = np.minimum(size + _BG_GROWTH * np.maximum(d - near, 0.0), _COARSE_OUTSIDE * size)
    # each grid cube as six tetrahedra around its main diagonal (Kuhn)
    i, j, k = np.meshgrid(*(np.arange(m - 1) for m in n), indexing="ij")
    base = (i * n[1] + j) * n[2] + k
    base = base.ravel()
    off = np.array([[a * n[1] * n[2] + b * n[2] + c for a, b, c in corner] for corner in (
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)))])
    tets = (base[:, None, None] + off[None]).reshape(-1, 4).astype(np.int32)
    x = G[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return np.ascontiguousarray(G, np.float64), tets, np.ascontiguousarray(val, np.float64)


def _tetwild(mesh, size):
    """fTetWild's mesh of `mesh`, the same one every time for the same
    surface and settings.

    fTetWild is not reproducible: the same surface comes back with a
    different mesh on every call (4,904 vs 4,920 vertices on a small lug with
    one thread, optimization and simplification off, address randomization
    off; its triangle insertion already differs), and the cross-check's
    peak stress moved 3-8% with it. None of that is reachable from Python,
    so the mesh is kept by a hash of the surface and the settings, and the
    same part is analysed on the same mesh, and reads the same, every time
    after the first."""
    import hashlib

    import pytetwild

    # copies: the binding refuses a read-only array ("incompatible function
    # arguments"), and ascontiguousarray hands a read-only mesh's own back,
    # which failed every tet confirm, the study falling back to voxels
    v = np.array(mesh.vertices, np.float64, order="C", copy=True)
    f = np.array(mesh.faces, np.int32, order="C", copy=True)
    opts = dict(edge_length_abs=float(size), epsilon=TETWILD_EPSILON,
                # its defaults (energy 10, 80 passes) took twice as long for
                # a handful fewer elements under a tenth of regular quality
                stop_energy=50.0, num_opt_iter=20)
    sizing = _outside_coarse(mesh, v, size) if os.environ.get("LOADPATH_TET_SIZING", "1") != "0" else None
    if sizing is not None:
        opts["edge_length_abs"] = float(size) * _COARSE_OUTSIDE
    key = hashlib.sha256(v.tobytes() + f.tobytes() + repr(sorted(opts.items())).encode()
                         + (b"sizing-v1" if sizing is not None else b"")
                         + str(getattr(pytetwild, "__version__", "")).encode()).hexdigest()[:24]
    cache = _tet_cache_dir()
    path = os.path.join(cache, key + ".npz") if cache else None
    if path and os.path.exists(path):
        try:
            with np.load(path) as z:
                return z["pts"].astype(float), z["tets"].astype(np.int64)
        except (OSError, ValueError, KeyError):
            pass
    bg = dict(zip(("bg_vertices", "bg_tets", "bg_values"), sizing)) if sizing is not None else {}
    got = _tetwild_guarded(v, f, opts, bg)
    if got is None and bg:
        # fTetWild went past its memory limit with the sizing field: mesh
        # at the plain edge everywhere (0.55 GB on the robot-arm wrist link, 0.94
        # GB on the upper arm). The result is kept under this key, so the
        # surface goes straight to it next time.
        opts["edge_length_abs"] = float(size)
        got = _tetwild_guarded(v, f, opts, {})
    if got is None:
        raise MemoryError(f"fTetWild needed more than {_mesher_budget() / 1e9:.1f} GB "
                          "(LOADPATH_TET_MESH_MEMORY sets the limit, in GB)")
    pts, tets = got
    pts, tets = np.asarray(pts, float), np.asarray(tets, np.int64)
    x = pts[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    if path:
        try:
            os.makedirs(cache, exist_ok=True)
            tmp_path = path + f".{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp_path, pts=pts, tets=tets.astype(np.int32))
            os.replace(tmp_path, path)
        except OSError:
            pass
    return pts, tets


def _mesher_budget() -> float:
    """Bytes fTetWild may use before it is stopped: LOADPATH_TET_MESH_MEMORY
    (GB), else 60% of the memory free now, at least 2 GB."""
    env = os.environ.get("LOADPATH_TET_MESH_MEMORY")
    if env:
        return float(env) * 1e9
    from .study import _available_bytes

    return max(2e9, 0.6 * (_available_bytes() or 8e9))


def _rss(pid: int) -> float | None:
    """Resident bytes of process `pid` (Linux /proc, else ps), or None."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5, check=False)
        return float(out.stdout.strip()) * 1024.0 if out.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _tetwild_main(work: str) -> None:
    """Child process entry: mesh `work`/in.npz (v, f, bg_*) with the
    options in `work`/opts.json, write `work`/out.npz (pts, tets)."""
    import json

    import pytetwild

    with np.load(os.path.join(work, "in.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    with open(os.path.join(work, "opts.json")) as fh:
        opts = json.load(fh)
    v, f = arrays.pop("v"), arrays.pop("f")
    # fTetWild leaves files (__tracked_surface.stl, _skipped.*) in the
    # working directory: run it in the scratch one
    os.chdir(work)
    pts, tets = pytetwild.tetrahedralize(v, f, num_threads=os.cpu_count() or 1, **opts, **arrays)
    np.savez(os.path.join(work, "out.tmp.npz"), pts=np.asarray(pts, float), tets=np.asarray(tets, np.int64))
    os.replace(os.path.join(work, "out.tmp.npz"), os.path.join(work, "out.npz"))


def _tetwild_guarded(v, f, opts, bg):
    """fTetWild in a child process under a memory limit (_mesher_budget):
    (pts, tets), or None when it went past the limit and was stopped. A
    mesher that runs away in-process takes the whole study (and on a
    shared box, everything else) down with it; in a child it costs one
    retry. (A plain subprocess, not
    multiprocessing: spawning re-runs the caller's script without a
    __main__ guard.)"""
    import json
    import sys

    budget = _mesher_budget()
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([pkg_parent] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    with tempfile.TemporaryDirectory() as work:
        np.savez(os.path.join(work, "in.npz"), v=v, f=f, **bg)
        with open(os.path.join(work, "opts.json"), "w") as fh:
            json.dump(opts, fh)
        code = ("import sys; from loadpath.crosscheck import _tetwild_main; "
                "_tetwild_main(sys.argv[1])")
        with open(os.path.join(work, "log.txt"), "wb") as log:
            proc = subprocess.Popen([sys.executable, "-c", code, work], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        proc.wait(0.5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    r = _rss(proc.pid)
                    if r is not None and r > budget:
                        return None
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        out = os.path.join(work, "out.npz")
        if proc.returncode != 0 or not os.path.exists(out):
            with open(os.path.join(work, "log.txt"), "rb") as fh:
                err = fh.read().decode(errors="replace").strip().splitlines()[-3:]
            raise RuntimeError(f"fTetWild exited with code {proc.returncode}: {' '.join(err)}")
        with np.load(out) as z:
            return z["pts"], z["tets"]


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


def _boundary_nodes(tets):
    """Vertices on the mesh's boundary: of the faces only one tet owns.
    (A distance to the input surface took 7 s on the motor-mount and misses
    nodes fTetWild leaves inside its envelope.)"""
    f = np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3), axis=1)
    uniq, count = np.unique(f, axis=0, return_counts=True)
    return np.unique(uniq[count == 1])


def _region_nodes(study, shape, pts, tol):
    return np.nonzero(shape.sdf(pts) <= tol)[0]


def _load_vectors(study, pts, surf, tol):
    """Per load case: nodal force array (n, 3), and the load node indices."""
    cases = []
    for case in study.load_cases:
        f = np.zeros((len(pts), 3))
        nodes_all = []
        for ld in case.loads:
            if ld.pressure:
                raise ValueError("the tetrahedral cross-check does not apply pressure loads yet")
            shape = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
            nodes = np.intersect1d(_region_nodes(study, shape, pts, tol), surf)
            if len(nodes) == 0:
                raise ValueError(f"load in case {case.name!r} touches no surface node of the part")
            nodes_all.append(nodes)
            force = np.asarray(ld.force, float)
            if ld.distribution == "bearing":
                # as domain.py: the shaft pushes on the half of the bore facing
                # the force, as a cosine over it
                from .domain import _cylinder_of

                p0, ax = _cylinder_of(study, ld.region, ld.shape, f"load in case {case.name!r}")
                axial = (force @ ax) * ax
                radial = force - axial
                if np.any(radial):
                    d = pts[nodes] - p0
                    rad = d - (d @ ax)[:, None] * ax
                    rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
                    w = np.maximum(rad @ (radial / np.linalg.norm(radial)), 0.0)
                    if w.sum() > 0:
                        f[nodes] += (w / w.sum())[:, None] * radial[None]
                        force = axial
            if np.any(force):
                f[nodes] += force / len(nodes)
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
    """6x6 elasticity (engineering shear, the order of fem._VOIGT, which
    _b_matrices shares): isotropic for a Poisson's ratio, transversely
    isotropic about the build direction for a fem.Layered, exactly the
    matrix the voxel verify solves with."""
    from .fem import elastic_c

    return elastic_c(E, nu)


def _node_pattern(conn, n, chunk=200_000):
    """Which nodes share an element: (indptr, indices) of the node graph,
    each row sorted: the pattern of E^T E for the element-node incidence E,
    summed over chunks of elements (a sort of every element's 100 node
    pairs took 73 s on the upper arm; this, a few)."""
    import scipy.sparse as sp

    g = None
    m = conn.shape[1]
    for s0 in range(0, len(conn), chunk):
        cc = conn[s0:s0 + chunk]
        e = sp.csr_matrix((np.ones(cc.size, np.float32), cc.ravel(), np.arange(0, cc.size + 1, m)),
                          shape=(len(cc), n))
        part = (e.T.tocsr() @ e).tocsr()
        part.data[:] = 1.0
        g = part if g is None else ((g + part).tocsr())
        if g is not part:
            g.data[:] = 1.0
    g.sort_indices()
    return g.indptr.astype(np.int64), g.indices.astype(np.int32)


def _scatter(cc, ke, nptr, nind, data):
    """Add element matrices `ke` (ne, 3m, 3m) on nodes `cc` (ne, m) into
    `data` of the dof CSR expanded from the node graph (nptr, nind): row
    3i+c starts at 9 nptr[i] + 3 deg(i) c, and holds node i's deg(i)
    neighbours x 3 components."""
    ne, m = cc.shape
    for e in range(ne):
        for a in range(m):
            i = cc[e, a]
            lo, hi = nptr[i], nptr[i + 1]
            deg = hi - lo
            for b in range(m):
                j = cc[e, b]
                l, h = lo, hi
                while l < h:            # nind[lo:hi] is sorted
                    mid = (l + h) >> 1
                    if nind[mid] < j:
                        l = mid + 1
                    else:
                        h = mid
                off = 3 * (l - lo)
                for ci in range(3):
                    base = 9 * lo + 3 * deg * ci + off
                    for cj in range(3):
                        data[base + cj] += ke[e, 3 * a + ci, 3 * b + cj]


def _eliminate(indptr, indices, data, fixed):
    """Zero the rows and columns of held dofs in a CSR, unit diagonal on
    theirs (in place, without a row index per entry)."""
    for i in range(len(indptr) - 1):
        fi = fixed[i]
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            if fi or fixed[j]:
                data[jj] = 1.0 if (fi and j == i) else 0.0


try:
    import numba as _nb

    _scatter = _nb.njit(cache=True, nogil=True)(_scatter)
    _eliminate = _nb.njit(cache=True, nogil=True)(_eliminate)
except ImportError:  # pragma: no cover - numba is a dependency
    pass


def _assemble(nodes, conn, E, nu, chunk=20_000, extra=()):
    """Global stiffness, vectorized over elements in chunks (skfem's
    assembly took 117 of 644 s on the wrist yoke), added straight into the
    final CSR: its pattern comes first from the node graph, and each
    chunk's element matrices are scattered into it. (Summing per-chunk COO
    matrices held three to four copies of the matrix at once: 2.07 GB to
    assemble a 0.48 GB matrix on the wrist link, past 15 GB on the upper
    arm.) `extra`: (nodes (k,), blocks (k, 3, 3)) added on the diagonal
    node blocks (pins)."""
    import scipy.sparse as sp

    c = _elastic(E, nu)
    n = len(nodes)
    nptr, nind = _node_pattern(conn, n)
    deg = np.diff(nptr)
    # dof rows: node i's three rows, each 3 deg(i) long
    row_len = np.repeat(3 * deg, 3)
    indptr = np.zeros(3 * n + 1, np.int64)
    np.cumsum(row_len, out=indptr[1:])
    cols3 = (3 * nind[:, None].astype(np.int64) + np.arange(3)).ravel()       # per node row, 3 deg long
    indices = np.empty(indptr[-1], np.int64 if indptr[-1] > np.iinfo(np.int32).max else np.int32)
    for ci in range(3):
        # row 3i+ci = node i's expanded columns; written per node in one go
        pos = (9 * nptr[:-1] + 3 * deg * ci)
        src = 3 * nptr[:-1]
        idx = np.repeat(pos - src, 3 * deg) + np.arange(len(cols3))
        indices[idx] = cols3
    del cols3
    data = np.zeros(len(indices))
    for s0 in range(0, len(conn), chunk):
        cc = conn[s0:s0 + chunk]
        vol, g = _grads(nodes, cc)
        ke = np.zeros((len(cc), 30, 30))
        for q in _QUAD:
            b = _b_matrices(_shape_grads(g, q))
            ke += np.einsum("eki,kl,elj->eij", b, c, b, optimize=True) * (vol / 4.0)[:, None, None]
        _scatter(np.ascontiguousarray(cc, np.int64), ke, nptr, nind, data)
    for idx, blocks in extra:
        idx = np.asarray(idx, np.int64)
        _scatter(idx[:, None], np.ascontiguousarray(blocks, np.float64), nptr, nind, data)
    if indices.dtype == np.int32 and indptr[-1] <= np.iinfo(np.int32).max:
        indptr = indptr.astype(np.int32)
    k = sp.csr_matrix((data, indices, indptr), shape=(3 * n, 3 * n))
    k.has_sorted_indices = True
    return k


def _solve(nodes, conn, E, nu, fixed_dof, springs, loads, n_vertices, edges):
    """P2 elasticity by CG with a p-multigrid preconditioner: symmetric
    Gauss-Seidel on the quadratic system, and a coarse correction on the
    linear (vertex-only) problem, Galerkin-projected and solved by node-
    blocked smoothed-aggregation AMG with the six rigid-body modes.
    Aggregation AMG coarsens the quadratic system's edge nodes badly: on the
    wrist yoke it took 166 iterations directly and 82 this way (22 s vs 38 s
    per case). `fixed_dof` marks held dofs; `springs` (or None; _pin_springs)
    is added to the stiffness (pins)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from pyamg.relaxation.relaxation import gauss_seidel

    from .fem import amg_solver

    k = _assemble(nodes, conn, E, nu, extra=springs or ())
    n = len(nodes)
    # held dofs eliminated in place, not by slicing out the free block (a
    # second copy of the matrix): their rows and columns zeroed and a unit
    # diagonal. With zero load there, CG and the smoother keep them at zero,
    # and every other dof sees exactly the free block.
    kff = k
    del k
    _eliminate(kff.indptr, kff.indices, kff.data, fixed_dof)
    free_mask = ~fixed_dof
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
    p = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * nv))
    p = (sp.diags(free_mask.astype(float)) @ p).tocsr()
    p.eliminate_zeros()
    used = np.asarray(p.sum(axis=0)).ravel() > 0
    p = p[:, used].tocsr()
    # the Galerkin product by blocks of rows: P^T K whole is about as big as
    # K itself (2.5 GB more on the upper arm, the solve's peak)
    a1 = None
    step = max(1, -(-kff.shape[0] // 16))
    for r0 in range(0, kff.shape[0], step):
        part = (p[r0:r0 + step].T @ (kff[r0:r0 + step] @ p)).tocsr()
        a1 = part if a1 is None else a1 + part
    a1 = a1.tocsr()
    cn = np.nonzero(used)[0]
    blocked = a1.shape[0] % 3 == 0 and np.all(np.bincount(cn // 3, minlength=nv)[np.unique(cn // 3)] == 3)
    # seeded (fem.amg_solver): pyamg draws its setup from np.random
    ml = amg_solver(a1.tobsr(blocksize=(3, 3)) if blocked else a1,
                    B=_rigid_modes(nodes[cn // 3], cn % 3), coarse_solver="splu", max_coarse=3000)

    def precond(r):
        r = np.asarray(r, dtype=np.float64).ravel()
        x = np.zeros_like(r)
        # one symmetric sweep each side: 106 iterations at 0.17 s beat two
        # sweeps' 82 at 0.28 s (wrist yoke)
        gauss_seidel(kff, x, r, iterations=1, sweep="symmetric")
        x += p @ ml.solve(p.T @ (r - kff @ x), maxiter=1, cycle="V", tol=1e-12)
        gauss_seidel(kff, x, r, iterations=1, sweep="symmetric")
        return x
    m = spla.LinearOperator(kff.shape, matvec=precond)

    def one(f):
        fn = np.zeros((n, 3))
        fn[: len(f)] = f
        # 1e-7 is six digits of stress; 1e-9 cost a third more iterations
        u, _info = spla.cg(kff, np.where(free_mask, fn.ravel(), 0.0), M=m, rtol=1e-7, maxiter=2000)
        u[fixed_dof] = 0.0
        return u.reshape(n, 3)
    fs = [f for _n, f, _ in loads]
    from . import amgfast

    if len(fs) < 2 or not amgfast.installed():
        # pyamg's own Gauss-Seidel holds the GIL: threads would only contend
        return [one(f) for f in fs]
    # cases on threads (amgfast's smoother releases the GIL), each its own
    # CG, so the same numbers; one cycle first factors the coarsest level
    # once (pyamg does it lazily, and each thread would make its own)
    precond(np.zeros(kff.shape[0]))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(min(len(fs), 4)) as ex:
        return list(ex.map(one, fs))


# a pin's stiffness per node, per unit modulus times element size, radially
# and along the axis: stiff against the elements' own (about E h), not so
# stiff that CG notices
_PIN_STIFFNESS = 1e3


def _pin_springs(blocks, e_h, n):
    """Pin stiffness as (nodes, 3x3 blocks) for _assemble's `extra`, or None."""
    if not blocks:
        return None
    return [(idx, b * (_PIN_STIFFNESS * e_h)) for idx, b in blocks]


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


def _nodal_von_mises(nodes, conn, u, E, nu, n_vertices, normal=None, chunk=50_000):
    """Von Mises at each vertex: the P2 field's stress at each element's own
    vertices, averaged over the elements sharing it. With `normal` (the
    build direction, unit), also the normal stress across the layers there,
    (vm, sn)."""
    c = _elastic(E, nu)
    acc = np.zeros(n_vertices)
    acc_n = np.zeros(n_vertices)
    cnt = np.zeros(n_vertices)
    # in chunks: the strain matrices of every element at once are 1.4 kB
    # each (0.7 GB on the upper arm)
    for s0 in range(0, len(conn), chunk):
        cc = conn[s0:s0 + chunk]
        _vol, g = _grads(nodes, cc)
        ue = u.reshape(-1, 3)[cc].reshape(len(cc), 30)
        for i in range(4):
            lam = np.zeros(4)
            lam[i] = 1.0
            b = _b_matrices(_shape_grads(g, lam))
            sig = np.einsum("kl,eli,ei->ek", c, b, ue, optimize=True)
            sx, sy, sz, txy, tyz, tzx = sig.T
            vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
            np.add.at(acc, cc[:, i], vm)
            if normal is not None:
                n = normal
                # n . sigma . n, as verify.py reads it (shear terms counted twice)
                sn = (sx * n[0] ** 2 + sy * n[1] ** 2 + sz * n[2] ** 2
                      + 2.0 * (txy * n[0] * n[1] + tyz * n[1] * n[2] + tzx * n[2] * n[0]))
                np.add.at(acc_n, cc[:, i], sn)
            np.add.at(cnt, cc[:, i], 1)
    vm = acc / np.maximum(cnt, 1)
    if normal is None:
        return vm
    return vm, acc_n / np.maximum(cnt, 1)


def crosscheck(study: Study, mesh, size: float | None = None, contact_band: float | None = None,
               backend: str = "native", adaptive: bool = False, mesher: str | None = None) -> dict:
    """Independent analysis of `mesh` (trimesh) under the study's cases.
    Returns {safetyFactor, cases: [{name, peakStress, peakAt, deflection}],
    nodes, elements, seconds, timings, backend, mesher}.

    backend: "native" (P2 here), "ccx" (CalculiX), or "both" on the same
    mesh: then the native result carries `calculix` (CalculiX's) and
    `agreement` (CalculiX / native per case). Supports hold the axes the
    study gives them (`fix`), and pins hold radially and axially.
    mesher: "tetwild" (default when pytetwild is installed) or "gmsh"."""
    t0 = time.time()
    mat = study.material
    # the voxel verify's elasticity and bond check (verify.py), so both read
    # a layered part the same way
    el = study.elastic(mat)
    proc = study.process
    normal, interlayer = None, 1.0
    if proc is not None and proc.build not in (None, "auto") and proc.interlayer < 1.0:
        from .am import build_vector

        normal = build_vector(proc.build)
        normal = normal / np.linalg.norm(normal)
        interlayer = float(proc.interlayer)
    asked = size or element_size(study)
    # an explicit size is the caller's; the default is coarsened to fit memory
    size = asked if size else fit_size(float(mesh.volume), asked)
    band = contact_band if contact_band is not None else 2.0 * size
    if adaptive:
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
        pts, tets = _tet_mesh(mesh, size, mesher=mesher)
    timings = {"mesh": time.time() - t0}
    from scipy.spatial import cKDTree
    surf = _boundary_nodes(tets)
    tol = 0.6 * size
    loads = _load_vectors(study, pts, surf, tol)

    t1 = time.time()
    nodes, conn, edges = _p2_mesh(pts, tets)
    # supports on the quadratic mesh: an edge node is held when both its
    # ends are; per axis as the study says (a sliding support holds one)
    fixed_dof = np.zeros(3 * len(nodes), bool)
    supports, pin_blocks, fixed_nodes = [], [], []
    for i, sup in enumerate(study.supports):
        sh = sup.shape if sup.shape is not None else study.region_shape(sup.region)[0]
        vn = np.intersect1d(_region_nodes(study, sh, pts, tol), surf)
        if len(vn) == 0:
            raise ValueError(f"supports[{i}] touches no surface node of the part")
        on = np.zeros(len(pts), bool)
        on[vn] = True
        held = np.concatenate([vn, len(pts) + np.nonzero(on[edges[:, 0]] & on[edges[:, 1]])[0]])
        fixed_nodes.append(vn)
        for a in range(3):
            if sup.fix[a]:
                fixed_dof[3 * held + a] = True
        pin = None
        if sup.pin:
            # held radially and along the axis, free to turn about it
            pin = (np.asarray(sh.p0, float), np.asarray(sh.axis, float) / np.linalg.norm(sh.axis))
            d = nodes[held] - pin[0]
            rad = d - (d @ pin[1])[:, None] * pin[1]
            rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
            pin_blocks.append((held, rad[:, :, None] * rad[:, None, :] + np.outer(pin[1], pin[1])[None]))
        supports.append((held, sup.fix, pin))
    fixed_nodes = np.unique(np.concatenate(fixed_nodes))
    contact = np.unique(np.concatenate([fixed_nodes] + [l[2] for l in loads]))
    far = cKDTree(pts[contact]).query(pts, distance_upper_bound=band)[0] == np.inf

    def summarize(results):
        cases = []
        sfs, sfs_raw = [], []
        for (name, f, lnodes), res in zip(loads, results):
            u, vm = res[0], res[1]
            sn = res[2] if len(res) > 2 else None
            i = int(np.argmax(np.where(far, vm, -1)))
            # where the stress is within half of the peak: the next pass refines there
            order = np.argsort(-np.where(far, vm, -1))[:200]
            hot = pts[order[vm[order] > 0.5 * vm[i]]]
            peak = float(vm[i])
            raw = mat.strength / peak if peak > 0 else float("inf")
            row = {}
            if sn is not None:
                # layers part along their bond: tension across them against
                # interlayer x strength, beside von Mises against strength
                # (verify.py's check, the same allowance)
                tens = np.where(far, np.maximum(sn, 0.0), 0.0)
                j = int(np.argmax(tens))
                inter = float(tens[j])
                raw_i = interlayer * mat.strength / inter if inter > 0 else float("inf")
                row = {"interlayerStress": inter, "interlayerAt": pts[j].tolist(),
                       "interlayerSafetyFactor": raw_i / STRESS_ALLOWANCE}
                raw = min(raw, raw_i)
            sfs_raw.append(raw)
            sfs.append(raw / STRESS_ALLOWANCE)
            cases.append({"name": name, "peakStress": peak, "peakAt": pts[i].tolist(),
                          "hotspots": hot.tolist(),
                          "deflection": float(np.linalg.norm(u[lnodes].mean(axis=0))),
                          "maxDisplacement": float(np.linalg.norm(u, axis=1).max()),
                          "safetyFactor": raw / STRESS_ALLOWANCE, **row})
        return {"safetyFactor": min(sfs), "safetyFactorNoAllowance": min(sfs_raw), "cases": cases}

    out = {}
    if backend in ("native", "both"):
        us = _solve(nodes, conn, mat.E, el, fixed_dof, _pin_springs(pin_blocks, mat.E * size, len(nodes)),
                    loads, len(pts), edges)
        if normal is None:
            out["native"] = summarize([(u, _nodal_von_mises(nodes, conn, u.ravel(), mat.E, el, len(pts)))
                                       for u in us])
        else:
            out["native"] = summarize([(u, *_nodal_von_mises(nodes, conn, u.ravel(), mat.E, el, len(pts), normal))
                                       for u in us])
        timings["native"] = time.time() - t1
    if backend in ("ccx", "both"):
        # the same quadratic mesh, held nodes and nodal forces: only the
        # solver and its stress recovery differ (a layered material as
        # engineering constants in the frame of its layers)
        t2 = time.time()
        out["ccx"] = summarize(_solve_ccx(mat, nodes, conn, supports, loads, len(pts), el, normal))
        timings["ccx"] = time.time() - t2
    if not out:
        raise ValueError(f"backend must be 'native', 'ccx' or 'both', not {backend!r}")
    result = dict(out.get("native") or out["ccx"])
    if len(out) == 2:
        # both solvers on one mesh: the native answer, CalculiX's beside it,
        # and how far apart they are (ratio CalculiX / native per case)
        result["calculix"] = out["ccx"]
        result["agreement"] = [{"name": a["name"], "peakStress": b["peakStress"] / a["peakStress"],
                                "deflection": b["deflection"] / a["deflection"]}
                               for a, b in zip(out["native"]["cases"], out["ccx"]["cases"])]
    return result | {"timings": timings, "backend": backend, "nodes": int(len(pts)), "elements": int(len(tets)),
                     "elementSize": size, "contactBand": band, "adaptive": bool(adaptive),
                     **({"coarsenedFrom": asked} if size > asked else {}),
                     "mesher": mesher or ("tetwild" if _has_tetwild() and not adaptive else "gmsh"),
                     "seconds": time.time() - t0}


def _ccx_material(mat, el) -> str:
    """The *MATERIAL block: isotropic, or for a fem.Layered the same
    transversely isotropic constants as engineering constants in the frame
    of the layers (1, 2 in the layer, 3 along the build direction), with an
    *ORIENTATION putting axis 3 on the layer normal; the section names it.
    fem.Layered: E3 = ez E, nu12 = nu13 = nu23 = nu (so nu31 = nu ez),
    G12 = E / 2(1 + nu), G13 = G23 = gz G12."""
    from .fem import Layered

    if not isinstance(el, Layered):
        return (f"*MATERIAL,NAME=M\n*ELASTIC\n{mat.E:.9g},{float(el):.9g}\n"
                "*SOLID SECTION,ELSET=PART,MATERIAL=M\n")
    e, nu = mat.E, el.nu
    g = e / (2.0 * (1.0 + nu))
    z = np.asarray(el.normal, float)
    # the local frame fem.Layered.c rotates from
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    consts = (e, e, el.ez * e, nu, nu, nu, g, el.gz * g, el.gz * g)
    return ("*MATERIAL,NAME=M\n*ELASTIC,TYPE=ENGINEERING CONSTANTS\n"
            + ",".join(f"{v:.9g}" for v in consts[:8]) + f"\n{consts[8]:.9g}\n"
            "*ORIENTATION,NAME=LAYERS,SYSTEM=RECTANGULAR\n" + ",".join(f"{v:.9g}" for v in (*x, *y)) + "\n"
            "*SOLID SECTION,ELSET=PART,MATERIAL=M,ORIENTATION=LAYERS\n")


def _solve_ccx(mat, nodes, conn, supports, loads, n_vertices, el=None, normal=None):
    """The quadratic system solved by CalculiX (ccx on PATH): one Abaqus-
    style deck, one static step per load case. Returns [(u, von Mises at
    the vertices)] per case, as the native solve does, and with `normal`
    (a layered study's build direction) the normal stress across the
    layers too, (u, vm, sn). CalculiX recovers stress at its integration
    points and extrapolates to the nodes, a different route from
    _nodal_von_mises; its nodal stresses are in the global frame."""
    exe = shutil.which("ccx")
    if exe is None:
        raise RuntimeError("CalculiX (ccx) is not on PATH; install it (Debian: apt install calculix-ccx) "
                           "or use the default backend")
    solver = _CCX_SOLVER or ("SPOOLES" if 3 * len(nodes) <= _CCX_DIRECT_MAX_DOF else "ITERATIVE CHOLESKY")
    with tempfile.TemporaryDirectory() as tmp:
        deck = os.path.join(tmp, "job")
        with open(deck + ".inp", "w") as fh:
            fh.write("*NODE,NSET=NALL\n")
            np.savetxt(fh, np.column_stack([np.arange(1, len(nodes) + 1), nodes]), fmt="%d,%.9g,%.9g,%.9g")
            # _EDGES is CalculiX's C3D10 mid-node order (12, 23, 31, 14, 24, 34)
            fh.write("*ELEMENT,TYPE=C3D10,ELSET=PART\n")
            np.savetxt(fh, np.column_stack([np.arange(1, len(conn) + 1), conn + 1]), fmt="%d", delimiter=",")
            bc = []
            for k, (held, fix, pin) in enumerate(supports):
                fh.write(f"*NSET,NSET=S{k}\n")
                np.savetxt(fh, held + 1, fmt="%d")
                if pin is not None:
                    # the pin exactly: cylindrical axes about it (radial,
                    # tangential, axial), radial and axial held
                    a, b = pin[0], pin[0] + pin[1]
                    fh.write(f"*TRANSFORM,NSET=S{k},TYPE=C\n" + ",".join(f"{v:.9g}" for v in (*a, *b)) + "\n")
                    bc += [f"S{k},1,1", f"S{k},3,3"]
                bc += [f"S{k},{a + 1},{a + 1}" for a in range(3) if fix[a]]
            fh.write(_ccx_material(mat, mat.nu if el is None else el) + "*BOUNDARY\n" + "\n".join(bc) + "\n")
            for _name, f, _l in loads:
                fh.write(f"*STEP\n*STATIC,SOLVER={solver}\n*CLOAD,OP=NEW\n")
                i, c = np.nonzero(f)
                np.savetxt(fh, np.column_stack([i + 1, c + 1, f[i, c]]), fmt="%d,%d,%.12g")
                fh.write("*NODE FILE\nU\n*NODE FILE\nS\n*END STEP\n")
        # One thread unless LOADPATH_CCX_THREADS asks for more: the packaged
        # CalculiX (2.21, SPOOLES) on several threads returned a different
        # answer run to run on the same deck (a propped cantilever's
        # deflection 0.0975 / 0.0963 / 0.0943 mm, its peak stress 1.23 or
        # 2.07 MPa; 2 runs in 6 off), and single-threaded the same every
        # time (8 of 8). The cross-check is a second opinion: it has to be
        # repeatable before it is fast.
        env = dict(os.environ)
        threads = os.environ.get("LOADPATH_CCX_THREADS", "1")
        env["OMP_NUM_THREADS"] = threads
        env["CCX_NPROC_EQUATION_SOLVER"] = threads
        env["CCX_NPROC_STIFFNESS"] = threads
        env["CCX_NPROC_RESULTS"] = threads
        run = subprocess.run([exe, "job"], cwd=tmp, capture_output=True, text=True, timeout=7200, env=env)
        if run.returncode != 0 or not os.path.exists(deck + ".frd"):
            raise RuntimeError(f"CalculiX failed: {(run.stdout + run.stderr)[-2000:]}")
        steps = _read_frd(deck + ".frd", len(nodes))
    if len(steps) != len(loads):
        raise RuntimeError(f"CalculiX returned {len(steps)} steps for {len(loads)} load cases")
    if normal is None:
        return [(u, vm[:n_vertices]) for u, vm, _s in steps]
    n = np.asarray(normal, float)
    out = []
    for u, vm, sig in steps:
        sx, sy, sz, sxy, syz, szx = sig[:n_vertices].T
        sn = (sx * n[0] ** 2 + sy * n[1] ** 2 + sz * n[2] ** 2
              + 2.0 * (sxy * n[0] * n[1] + syz * n[1] * n[2] + szx * n[2] * n[0]))
        out.append((u, vm[:n_vertices], sn))
    return out


# CalculiX's solver: its direct SPOOLES factorization up to this many
# unknowns (110 s on the motor-mount's 350k, where its preconditioned CG
# took 410 s), the CG above (memory stays linear in the mesh)
_CCX_DIRECT_MAX_DOF = 1_000_000
_CCX_SOLVER = None


def _read_frd(path, n):
    """Per step: displacements (n, 3), nodal von Mises (n,) and the nodal
    stress (n, 6: xx, yy, zz, xy, yz, zx) from a CalculiX .frd result file
    (fixed-width records)."""
    steps, u, s, block = [], None, None, None
    with open(path) as fh:
        for line in fh:
            if line.startswith(" -4"):
                block = line.split()[1]
                if block == "DISP":
                    u, s = np.zeros((n, 3)), np.zeros((n, 6))
                    steps.append((u, s))
            elif line.startswith(" -1") and block in ("DISP", "STRESS"):
                i = int(line[3:13]) - 1
                vals = [float(line[13 + 12 * k:25 + 12 * k]) for k in range((len(line.rstrip()) - 13) // 12)]
                if block == "DISP":
                    u[i] = vals[:3]
                else:
                    s[i] = vals[:6]
            elif line.startswith(" -3"):
                block = None
    out = []
    for u, s in steps:
        sx, sy, sz, sxy, syz, szx = s.T
        out.append((u, np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                               + 3 * (sxy ** 2 + syz ** 2 + szx ** 2)), s))
    return out
