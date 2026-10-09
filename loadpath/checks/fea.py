"""
fea_probe: coarse voxel linear-elastic critic for load-bearing parts. Seconds,
not minutes; advises, never generates — geometry always comes from the model,
this just says "the root of that cantilever is the hotspot, thicken it".

The mesh is voxelized on a regular grid and every solid voxel becomes an
8-node hexahedral element (full 2x2x2 Gauss integration, one shared stiffness
matrix since all elements are identical cubes). E=1.0, nu=0.3 — RELATIVE
results only: use displacements/stresses to compare variants of the same part
under the same loads, never as absolute MPa. Real materials, units, and safety
factors are out of scope by design.
"""
import numpy as np
from scipy import ndimage
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from sdf_kit import _solid_voxels

__all__ = ["fea_probe"]

_E, _NU = 1.0, 0.3
_MAX_SOLID = 200_000      # element-count refusal — keep the critic seconds-scale
_CG_MAXITER = 5000
_CG_RTOL = 1e-6
_HOTSPOTS = 5

# local corner order for the hex (matches the B-matrix DOF layout below)
_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
])
_CONN26 = np.ones((3, 3, 3), bool)


def _elastic_c(e, nu):
    """Isotropic elasticity matrix (6x6, engineering shear strains)."""
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] = lam + 2 * mu
    c[np.arange(3, 6), np.arange(3, 6)] = mu
    return c


def _b_matrix(xi, eta, zeta, h):
    """Strain-displacement matrix (6x24) at a natural-coordinate point, for a
    cube element of side h (Jacobian is diagonal h/2)."""
    s = 2.0 * _CORNERS - 1.0  # corner signs
    scale = 2.0 / h           # dN/dx = dN/dxi * 2/h
    dn = np.empty((8, 3))
    dn[:, 0] = 0.125 * s[:, 0] * (1 + eta * s[:, 1]) * (1 + zeta * s[:, 2]) * scale
    dn[:, 1] = 0.125 * s[:, 1] * (1 + xi * s[:, 0]) * (1 + zeta * s[:, 2]) * scale
    dn[:, 2] = 0.125 * s[:, 2] * (1 + xi * s[:, 0]) * (1 + eta * s[:, 1]) * scale
    b = np.zeros((6, 24))
    for i in range(8):
        dx, dy, dz = dn[i]
        col = 3 * i
        b[0, col] = dx
        b[1, col + 1] = dy
        b[2, col + 2] = dz
        b[3, col], b[3, col + 1] = dy, dx
        b[4, col + 1], b[4, col + 2] = dz, dy
        b[5, col], b[5, col + 2] = dz, dx
    return b


def _hex_ke(h, c):
    """Element stiffness (24x24) for a cube of side h, 2x2x2 Gauss."""
    g = 1.0 / np.sqrt(3.0)
    det_j = (h / 2.0) ** 3
    ke = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                b = _b_matrix(xi, eta, zeta, h)
                ke += b.T @ c @ b * det_j
    return ke


def _nodes_in_spheres(coords, spheres):
    """Boolean mask of node coords within any {"point", "r"} sphere."""
    hit = np.zeros(len(coords), bool)
    for s in spheres:
        pt = np.asarray(s["point"], float)
        hit |= np.linalg.norm(coords - pt, axis=1) <= float(s["r"])
    return hit


def _fail(out, msg):
    out["error"] = msg
    return out


def fea_probe(mesh, loads, supports, resolution=32):
    """Coarse voxel FEA. loads: [{"point", "r", "force": [fx,fy,fz]}] — each
    force split evenly over the nodes its sphere catches. supports:
    [{"point", "r"}] — nodes inside are fully fixed.

    Returns {"ok", "maxDisplacement", "compliance", "hotspots": top-5
    [{"point", "vonMises"}], "resolution", "error"}. Comparative numbers only
    (E=1): valid for "variant B is stiffer than variant A" and "the hotspot is
    here", meaningless as absolute stress. Refuses (ok=False + error) on >200k
    solid voxels, disconnected loads, or CG non-convergence.
    """
    out = {"ok": False, "maxDisplacement": 0.0, "compliance": 0.0,
           "hotspots": [], "resolution": int(resolution), "error": None}
    try:
        pitch = float(np.max(mesh.extents)) / float(resolution)
        solid, origin = _solid_voxels(mesh, pitch, pad=1)
        n_solid = int(solid.sum())
        if n_solid == 0:
            return _fail(out, "voxelization produced no solid elements")
        if n_solid > _MAX_SOLID:
            return _fail(out, f"{n_solid} solid voxels exceeds the "
                              f"{_MAX_SOLID} element budget — lower resolution")

        eidx = np.argwhere(solid)                     # (ne, 3) element cells
        dims = np.asarray(solid.shape) + 1            # node grid dims
        # node (i,j,k) sits at the voxel-corner lattice; voxel centers are at
        # origin + idx*pitch, so corners are offset by -pitch/2.
        node0 = origin - 0.5 * pitch

        def nid(multi):  # (…,3) node multi-index -> flat node id
            return (multi[..., 0] * dims[1] + multi[..., 1]) * dims[2] + multi[..., 2]

        # drop solid islands with no support: they make K singular, and any
        # load on them is a modeling error worth surfacing, not solving.
        comp_lbl, _ = ndimage.label(solid, structure=_CONN26)
        all_nodes_multi = np.stack(np.meshgrid(*[np.arange(d) for d in dims],
                                               indexing="ij"), axis=-1)
        all_coords = node0 + all_nodes_multi.reshape(-1, 3) * pitch
        sup_node = _nodes_in_spheres(all_coords, supports).reshape(*dims)
        if not sup_node.any():
            return _fail(out, "no nodes inside any support sphere")
        enodes_multi = eidx[:, None, :] + _CORNERS[None, :, :]   # (ne, 8, 3)
        e_supported = sup_node[tuple(np.moveaxis(enodes_multi, -1, 0))].any(axis=1)
        good_comps = np.unique(comp_lbl[tuple(eidx[e_supported].T)]) \
            if e_supported.any() else np.array([], int)
        keep = np.isin(comp_lbl[tuple(eidx.T)], good_comps)
        if not keep.any():
            return _fail(out, "no solid material connected to the supports")
        eidx = eidx[keep]
        enodes_multi = enodes_multi[keep]

        # compact node numbering over used nodes only
        enodes_flat = nid(enodes_multi)                          # (ne, 8)
        used, enodes = np.unique(enodes_flat, return_inverse=True)
        enodes = enodes.reshape(enodes_flat.shape)
        coords = node0 + np.stack(np.unravel_index(used, dims), axis=1) * pitch
        n_node = len(used)
        edofs = (3 * enodes[:, :, None] + np.arange(3)).reshape(-1, 24)

        # loads and supports on the compacted nodes
        fixed = np.repeat(_nodes_in_spheres(coords, supports), 3)
        f = np.zeros(3 * n_node)
        for load in loads:
            hit = _nodes_in_spheres(coords, [load])
            k = int(hit.sum())
            if k == 0:
                return _fail(out, f"load at {load['point']} catches no nodes "
                                  "connected to the supports")
            f[np.repeat(hit, 3)] += np.tile(np.asarray(load["force"], float) / k, k)

        c = _elastic_c(_E, _NU)
        ke = _hex_ke(pitch, c)
        n_dof = 3 * n_node
        # chunked assembly: 576 triplets per element adds up fast at the top
        # of the element budget, so build and fold in ~2M-triplet slabs.
        k_mat = sp.csr_matrix((n_dof, n_dof))
        chunk = 4000
        for i in range(0, len(edofs), chunk):
            ed = edofs[i:i + chunk]
            rows = np.repeat(ed, 24, axis=1).ravel()
            cols = np.tile(ed, (1, 24)).ravel()
            data = np.tile(ke.ravel(), len(ed))
            k_mat += sp.coo_matrix((data, (rows, cols)),
                                   shape=(n_dof, n_dof)).tocsr()

        free = ~fixed
        kff = k_mat[free][:, free]
        diag = kff.diagonal()
        diag[diag <= 0] = 1.0  # paranoia; pruning should preclude zero rows
        precond = sp.diags(1.0 / diag)  # Jacobi
        u_f, info = cg(kff, f[free], rtol=_CG_RTOL, maxiter=_CG_MAXITER, M=precond)
        if info != 0:
            return _fail(out, f"CG failed to converge (info={info}) — the "
                              "structure may be near-mechanism at this resolution")

        u = np.zeros(n_dof)
        u[free] = u_f
        out["maxDisplacement"] = float(
            np.linalg.norm(u.reshape(-1, 3), axis=1).max())
        out["compliance"] = float(f @ u)

        # von Mises from the strain at each element center
        b0 = _b_matrix(0.0, 0.0, 0.0, pitch)
        stress = (u[edofs] @ b0.T) @ c.T                         # (ne, 6)
        sx, sy, sz, txy, tyz, tzx = stress.T
        vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                     + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))
        top = np.argsort(vm)[::-1][:_HOTSPOTS]
        centers = origin + eidx * pitch                          # voxel centers
        out["hotspots"] = [
            {"point": [float(x) for x in centers[i]], "vonMises": float(vm[i])}
            for i in top
        ]
        out["ok"] = True
        return out
    except Exception as exc:  # critic must never take the run down with it
        return _fail(out, f"{type(exc).__name__}: {exc}")
