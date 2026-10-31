"""
Build-process simulation: the residual stress and distortion a part picks
up while it is printed, and what to do about it.

One mechanism covers the three processes this handles. Each layer goes
down hot and stress-free, bonds to the (already distorted) part below, and
then shrinks as it cools; what it is bonded to stops part of that shrink,
so the layer is left in tension, the part below it bends, and when the part
comes off the plate or bed it springs into a new shape. The solver adds the
part a layer at a time on the voxel grid fem.py uses, gives each new layer
an *inherent strain* (the shrink it would take if it were free), solves for
the displacement increment with only the cells printed so far, and carries
every cell's stress forward. A new layer is stress-free in the shape it is
laid into: its bottom nodes sit where the layer below has moved to, its top
is the nominal plane (the recoater or nozzle levels it). Then the part is
released (cut off the plate, or peeled off the bed) and springs back.

What sets the inherent strain:

  lpbf  Each melt track shrinks from the solidus and is held by the solid
        metal around it, so it yields; what it keeps is roughly the yield
        strain. eps = -k * yield / E in the layer plane (and `kz` times that
        along the build), with k fitted to a published distortion benchmark
        (validated in bench/buildsim_validate.py and docs/BUILDSIM.md). The
        commercial tools (Ansys Additive's assumed-uniform-strain mode,
        Netfabb, Simufact) do the same with a strain scaling factor they
        calibrate per machine and material.
  fdm   A road freezes at its glass transition and cools to the temperature
        the part sits at during the print, set by the bed and the chamber:
        T(z) = T_chamber + (T_bed - T_chamber) exp(-z / lambda). Polymer
        relaxes while warm, so only `relax` of that shrink is kept. After
        the print the whole part cools on the bed to room temperature
        (stage two), then comes off. The bed holds the first layer until
        the peel traction beats the adhesion, so the report gives the peel
        stress at the bed: that, not the final warp, is what lifts corners.
  sls   The part sinters in a hot powder cake and cools as a whole, so there
        is no layer stepping. The bulk shrink (3-4%) is taken out by the
        printer's scale factor; what warps a part is the difference between
        thin walls, which cool fast, and thick sections, which crystallize
        more and shrink more. eps = -ds * min(depth / depth_sat, 1),
        depth = distance to the nearest surface. That model is a heuristic
        (stated in the report), sized to the 0.2-0.5% differential shrink
        nylon sintering reports, not fitted to a measured part.

Layers are lumped: one analysis layer is `layer` mm (default one voxel) of
real layers, which is how every inherent-strain code runs a part in finite
time. Lumping is not free: a thick slab laid at once on a compliant part
bends it less than the same slab laid in thin layers (on the free strip of
tests/test_buildsim.py, 0.5 mm lumps give 72% of the thin-layer curvature,
progressive_beam). On a part held by a thick plate the effect is smaller;
the LPBF factor was fitted at the lumping the validation runs, and
docs/BUILDSIM.md gives the pitch study.

What it gives back (BuildResult): the displacement field before and after
release (best-fit rigid motion removed after a full release, as a CMM
reading of a free part would), the residual stress, the peak von Mises
against the material's yield, the upward lift at the build plane during
the build (an LPBF recoater-contact indicator), the FDM bed peel stress,
and a sampler of the displacement anywhere, which `compensate` uses to
pre-distort exported geometry so the print comes out nominal.

`optimize_warp` puts the as-built distortion into topology optimization:
weighted compliance plus distortion, with the gradient of the layer-by-
layer solve by adjoint (two solves per analysis layer and iteration on a
factorization shared between them).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .fem import HexMesh, Stiffness, _b_matrix, amg_solver, elastic_c, hex8_b_center, rigid_body_modes, \
    von_mises

__all__ = ["BUILD", "BuildSetup", "BuildResult", "inherent_strain", "simulate_build", "simulate_part",
           "compensate", "compensate_mesh", "to_build_frame", "progressive_beam", "optimize_warp",
           "WarpOptions", "WarpResult"]


# Process parameters by material. Every number carries its source; the
# fitted ones say what they were fitted to. Values a datasheet does not
# give are marked "estimate" with why.
BUILD = {
    # LPBF: k is the share of the yield strain a layer keeps (fit: see
    # LPBF_K below); kz the build-direction strain relative to in-plane.
    "lpbf": {"k": None, "kz": 1.0},
    # FDM: tg is where the road stops flowing (DSC Tg); relax the share of
    # the thermal shrink below Tg the polymer keeps as stress (the rest
    # creeps out while the part is warm). bed/chamber: typical settings.
    "fdm": {
        "abs": {"tg": (101.1, "polylite-abs DSC"), "bed": 100.0, "chamber": 35.0},
        "asa": {"tg": (97.8, "polylite-asa DSC"), "bed": 100.0, "chamber": 35.0},
        "petg": {"tg": (80.0, "estimate: PETG copolyester Tg 78-81 C (Eastman Eastar 6763, DSC)"),
                 "bed": 80.0, "chamber": 30.0},
        "pla": {"tg": (60.0, "estimate: PLA Tg 55-65 C"), "bed": 60.0, "chamber": 30.0},
        "pa-cf": {"tg": (70.0, "bambu-paht-cf DSC"), "bed": 90.0, "chamber": 40.0},
    },
    # SLS: ds the extra linear shrink of a thick section over a thin wall;
    # depth_sat the depth (mm) past which a section counts as thick.
    "sls": {"ds": 0.003, "depth_sat": 4.0},
}

# LPBF strain factor k (eps = -k * yield / E), fitted to the NIST AM-Bench
# 2018 bridges at 0.5 mm voxels, one analysis layer per voxel, a 12.7 mm
# plate (bench/buildsim_validate.py; docs/BUILDSIM.md). The yield-strain
# scaling does not carry across alloy classes: with IN625's k, 15-5 PH
# came out 60% high (its martensite forms on cooling and expands, which
# undoes part of the shrink), so 15-5 PH has its own fit. Alloys without a
# fit use IN625's and the report says so.
LPBF_K = 0.909
LPBF_K_FIT = {"in625": 0.909, "15-5ph": 0.569}
# FDM retained share of the sub-Tg shrink, and the bed's hold (N/mm^3 of
# elastic foundation): fitted together on the nine 20 mm-wide ABS bars of
# Armillotta et al. 2018 (bench/buildsim_fdm.py; rms 0.15 mm, mean 22%).
# The decay length (mm) of the bed's heat up the part is an estimate.
FDM_RELAX = 0.83
FDM_BED = 3.0
FDM_DECAY = 6.0
ROOM = 23.0


@dataclass
class BuildSetup:
    """How the part is made. Lengths mm, temperatures C, strains signed
    (negative = shrink). Everything not given comes from BUILD and the
    material library."""
    process: str = "lpbf"
    material: str = "ti64"
    layer: float | None = None          # mm per analysis layer (default: one voxel)
    plate: float = 10.0                 # mm of build plate under the part (lpbf); 0 = rigid plate
    plate_margin: float = 2.0           # mm the plate extends past the part's footprint
    plate_fixed: bool = True            # plate bolted down along its underside; False: free (3-2-1)
    strain: tuple | None = None         # inherent strain override (exx, eyy, ezz), build frame
    scale: float = 1.0                  # multiplies the inherent strain (a calibration)
    k: float | None = None              # lpbf: share of the yield strain kept (default LPBF_K)
    kz: float | None = None             # lpbf: build-direction strain / in-plane
    bed: float | None = None            # fdm: bed temperature
    chamber: float | None = None        # fdm: chamber / ambient air during the print
    room: float = ROOM
    relax: float | None = None          # fdm: share of the sub-Tg shrink kept
    decay: float | None = None          # fdm: mm the bed's heat reaches up the part
    ds: float | None = None             # sls: differential shrink, thick vs thin
    depth_sat: float | None = None      # sls: mm
    release: str = "full"               # "full" (off the plate, free), "none" (stay on), "cut"
    cut: tuple | None = None            # release="cut": ((x0,y0,z0),(x1,y1,z1)) box of cells removed
    E: float | None = None              # MPa override
    nu: float | None = None
    yield_strength: float | None = None  # MPa override (lpbf strain, stress check)
    alpha: float | None = None          # 1/K override (fdm)
    # fdm: how the bed holds the first layer, N/mm^3 (stiffness per unit
    # area of an elastic foundation in z and in plane); None = FDM_BED
    # (fitted), inf = rigid (glued), 0 = not at all (curls freely)
    bed_stiffness: float | None = None
    recoater_gap: float = 0.05          # mm: lift at the build plane that can touch the recoater


@dataclass
class BuildResult:
    mesh: HexMesh
    part: np.ndarray                    # bool per element: the printed part (not plate)
    layer_of: np.ndarray                # analysis layer per element (-1 = plate)
    u_built: np.ndarray                 # (n_nodes, 3) mm, at the end of the build, still on the plate/bed
    u: np.ndarray                       # (n_nodes, 3) mm, after release (rigid motion removed if free)
    stress_built: np.ndarray            # (ne, 6) MPa
    stress: np.ndarray                  # (ne, 6) MPa after release
    strain: np.ndarray                  # (ne, 6) inherent strain applied
    report: dict = field(default_factory=dict)
    node_active: np.ndarray | None = None  # bool per node: belongs to the released part

    def sampler(self, released=True):
        return _NodeField(self.mesh, self.u if released else self.u_built,
                          self.node_active if released else None)

    def displacement_at(self, points, released=True):
        """Displacement (n, 3) at world points, trilinear in the cell the
        point is in; points outside the part take the nearest node's."""
        return self.sampler(released)(points)


# ------------------------------------------------------------ the strain


def _material(name):
    from .materials import material_from_spec
    return material_from_spec(name)


def _lib(name, key, default=None):
    from .material_library import value
    return value(name, key, default)


def inherent_strain(setup: BuildSetup, z=None, depth=None) -> tuple[np.ndarray, dict]:
    """Per-cell inherent strain (n, 6) for heights `z` (mm above the plate,
    fdm) or depths below the surface (mm, sls), and a dict of what set it.
    lpbf is uniform: one row broadcast."""
    proc = setup.process.lower()
    mat = _material(setup.material)
    e = float(setup.E or mat.E)
    n = 1 if z is None and depth is None else len(z if z is not None else depth)
    info = {"process": proc, "material": setup.material, "E": e}
    if setup.strain is not None:
        s = np.asarray(setup.strain, float) * setup.scale
        out = np.zeros((n, 6))
        out[:, :3] = s
        info.update({"model": "given", "strain": s.tolist()})
        return out, info
    if proc == "lpbf":
        sy = float(setup.yield_strength or mat.strength)
        name = str(setup.material).lower()
        k = LPBF_K_FIT.get(name, LPBF_K) if setup.k is None else float(setup.k)
        info["kSource"] = ("given" if setup.k is not None else
                           "fitted to AM-Bench 2018 (this alloy)" if name in LPBF_K_FIT else
                           "AM-Bench 2018 IN625 fit, not calibrated for this alloy: could be off 2x (calibrate with a "
                           "cantilever cut test and set buildSim.scale)")
        kz = BUILD["lpbf"]["kz"] if setup.kz is None else float(setup.kz)
        ep = -k * sy / e * setup.scale
        out = np.zeros((n, 6))
        out[:, :2] = ep
        out[:, 2] = kz * ep
        info.update({"model": "yield-strain", "k": k, "kz": kz, "yield": sy, "strain": [ep, ep, kz * ep]})
        return out, info
    if proc == "fdm":
        p = _fdm_params(setup)
        tz = _fdm_temperature(p, np.zeros(n) if z is None else np.asarray(z, float))
        ep = -p["relax"] * p["alpha"] * np.maximum(p["tg"] - tz, 0.0) * setup.scale
        out = np.zeros((n, 6))
        out[:, :3] = ep[:, None]
        info.update({"model": "glass-transition shrink", **{k: v for k, v in p.items() if k != "tgSource"},
                     "tgSource": p["tgSource"]})
        return out, info
    if proc in ("sls", "mjf"):
        ds = BUILD["sls"]["ds"] if setup.ds is None else float(setup.ds)
        dsat = BUILD["sls"]["depth_sat"] if setup.depth_sat is None else float(setup.depth_sat)
        d = np.zeros(n) if depth is None else np.asarray(depth, float)
        ep = -ds * np.minimum(d / dsat, 1.0) * setup.scale
        out = np.zeros((n, 6))
        out[:, :3] = ep[:, None]
        info.update({"model": "differential cooling shrink (heuristic)", "ds": ds, "depthSat": dsat})
        return out, info
    raise ValueError(f"build simulation: process must be lpbf, fdm or sls, got {setup.process!r}")


def _fdm_params(setup):
    from .thermal import expansion
    name = str(setup.material).lower()
    row = BUILD["fdm"].get(name)
    tg = _lib(name, "tg")
    src = "material library DSC"
    if row is not None and tg is None:
        tg, src = row["tg"]
    if tg is None:
        raise ValueError(f"fdm build simulation: no glass transition for {name!r}; give BuildSetup.strain")
    alpha = float(setup.alpha) if setup.alpha is not None else float(expansion(name))
    bed = setup.bed if setup.bed is not None else (row or {}).get("bed", 60.0)
    ch = setup.chamber if setup.chamber is not None else (row or {}).get("chamber", 30.0)
    return {"tg": float(tg), "tgSource": src, "alpha": alpha, "bed": float(bed), "chamber": float(ch),
            "room": float(setup.room), "relax": FDM_RELAX if setup.relax is None else float(setup.relax),
            "decay": FDM_DECAY if setup.decay is None else float(setup.decay)}


def _fdm_temperature(p, z):
    """Where each height sits during the print: bed temperature at the bed,
    the chamber's far up, never above Tg (a layer below Tg is solid)."""
    t = p["chamber"] + (p["bed"] - p["chamber"]) * np.exp(-np.maximum(z, 0.0) / p["decay"])
    return np.minimum(t, p["tg"])


# ------------------------------------------------------------ element kernels


def _eig_unit(nu) -> np.ndarray:
    """(24, 6): integral of B^T C over the unit cube at E = 1. An element's
    load from an inherent strain eps (6,) is E h^2 this @ eps."""
    c = elastic_c(1.0, nu)
    g = 1.0 / np.sqrt(3.0)
    out = np.zeros((24, 6))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                out += _b_matrix(xi, eta, zeta).T @ c * 0.125
    return out


class _Kernel:
    def __init__(self, mesh: HexMesh, nu, fixed, springs=None):
        self.mesh = mesh
        self.springs = springs if springs is not None and np.any(springs) else None
        self.nu = nu
        self.st = Stiffness(mesh, nu, fixed)
        self.g = _eig_unit(nu) * mesh.pitch ** 2
        self.bc = hex8_b_center() / mesh.pitch
        self.c1 = elastic_c(1.0, nu)

    def k(self, scale):
        """Stiffness at per-element moduli, plus the bed's springs."""
        k = self.st.assemble(scale)
        return k if self.springs is None else k + sp.diags(self.springs * ~self.st.fixed)

    def load(self, e, eps, elems):
        """Global load of inherent strains `eps` (len(elems), 6) at moduli e."""
        fe = (e[elems, None] * eps) @ self.g.T
        return np.bincount(self.mesh.edof[elems].ravel(), weights=fe.ravel(), minlength=self.mesh.n_dof)

    def internal(self, e, ue, eps, elems):
        """Global internal force of elements with displacement-since-
        activation ue (len, 24) and inherent strain eps."""
        fe = e[elems, None] * (ue @ self.st.ke.T - eps @ self.g.T)
        return np.bincount(self.mesh.edof[elems].ravel(), weights=fe.ravel(), minlength=self.mesh.n_dof)

    def stress(self, e, ue, eps):
        return ((ue @ self.bc.T - eps) @ self.c1.T) * e[:, None]


def _solve(k, f, xyz, d=None, direct=90_000, rtol=1e-7):
    """Sparse SPD solve: a direct factorization for small systems, CG on
    smoothed-aggregation AMG (rigid-body near null space of the nodes
    `xyz`, restricted to the dofs `d` the system keeps) beyond."""
    from scipy.sparse.linalg import cg, splu
    n = k.shape[0]
    if n <= direct:
        return splu(k.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(f)
    b = rigid_body_modes(xyz)
    if d is not None:
        b = b[d]
    ml = amg_solver(k.tocsr(), B=b, symmetry="hermitian", strength="symmetric", max_coarse=400,
                    coarse_solver="splu")
    u, info = cg(k, f, rtol=rtol, maxiter=2000, M=ml.aspreconditioner(cycle="V"))
    if info != 0:
        raise RuntimeError("build simulation: linear solve did not converge (is every piece attached?)")
    return u


def _dofs(nodes, fixed):
    d = (3 * np.flatnonzero(nodes)[:, None] + np.arange(3)).ravel()
    return d[~fixed[d]]


def _three_two_one(mesh, nodes):
    """Statically determinate support of a free part on its own nodes:
    one node held in x, y, z, a second (far along x) in y and z, a third
    (far along y) in z. A self-equilibrated load leaves no reaction."""
    idx = np.flatnonzero(nodes)
    xyz = mesh.node_xyz[idx]
    lo = xyz.min(axis=0)
    a = idx[np.argmin(np.linalg.norm(xyz - lo, axis=1))]
    pa = mesh.node_xyz[a]
    b = idx[np.argmax(xyz[:, 0] - 1e-3 * np.abs(xyz[:, 1] - pa[1]) - 1e-3 * np.abs(xyz[:, 2] - pa[2]))]
    pb = mesh.node_xyz[b]
    ab = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-12)
    off = np.linalg.norm(np.cross(xyz - pa, ab), axis=1) - 1e-3 * np.abs(xyz[:, 2] - pa[2])
    c = idx[np.argmax(off)]
    fixed = np.zeros(mesh.n_dof, bool)
    fixed[3 * a:3 * a + 3] = True
    fixed[3 * b + 1:3 * b + 3] = True
    fixed[3 * c + 2] = True
    return fixed


def _remove_rigid(xyz, u):
    """u minus its least-squares rigid motion (a best-fit alignment)."""
    b = rigid_body_modes(xyz)
    coef, *_ = np.linalg.lstsq(b, u.ravel(), rcond=None)
    return (u.ravel() - b @ coef).reshape(-1, 3)


# ------------------------------------------------------------ the solve


def _grid_mesh(solid, pitch, origin, setup):
    """HexMesh of the part's cells plus (lpbf, plate > 0) a plate under its
    footprint. Returns mesh, part mask, layer per element, fixed dofs and
    each element's height above the plate."""
    solid = np.asarray(solid, bool)
    ijk = np.argwhere(solid)
    if len(ijk) == 0:
        raise ValueError("build simulation: the part is empty")
    k0 = ijk[:, 2].min()
    plate_cells = 0
    if setup.process.lower() == "lpbf" and setup.plate > 0:
        plate_cells = max(1, int(round(setup.plate / pitch)))
    margin = int(round(setup.plate_margin / pitch)) if plate_cells else 0
    shift = np.array([margin, margin, plate_cells - k0])
    part_ijk = ijk + shift
    if plate_cells:
        lo = ijk[:, :2].min(axis=0)
        hi = ijk[:, :2].max(axis=0) + 2 * margin
        pi, pj, pk = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), np.arange(plate_cells),
                                 indexing="ij")
        plate_ijk = np.stack([pi.ravel(), pj.ravel(), pk.ravel()], axis=1)
        all_ijk = np.concatenate([plate_ijk, part_ijk])
        part = np.concatenate([np.zeros(len(plate_ijk), bool), np.ones(len(part_ijk), bool)])
    else:
        all_ijk = part_ijk
        part = np.ones(len(part_ijk), bool)
    org = np.asarray(origin, float) - shift * pitch
    mesh = HexMesh(all_ijk, pitch, org)
    lump = max(1, int(round((setup.layer or pitch) / pitch)))
    layer_of = np.where(part, (mesh.elem_ijk[:, 2] - plate_cells) // lump, -1)
    # the plate (or, without one, the bed) is held at its underside
    bottom = mesh.node_ijk[:, 2] == 0
    fixed = np.repeat(bottom, 3)
    if plate_cells and not setup.plate_fixed:
        on = np.zeros(mesh.n_nodes, bool)
        on[mesh.enode[~part].ravel()] = True
        fixed = _three_two_one(mesh, on)
    springs = np.zeros(mesh.n_dof)
    kb = setup.bed_stiffness
    if kb is None and setup.process.lower() == "fdm":
        kb = FDM_BED
    if not plate_cells and kb is not None and np.isfinite(kb):
        # an elastic bed: springs on the underside's nodes (a quarter cell
        # of area per touching cell), 3-2-1 against rigid motion
        fixed = _three_two_one(mesh, bottom)
        area = np.zeros(mesh.n_nodes)
        under = mesh.elem_ijk[:, 2] == 0
        for c in range(4):
            np.add.at(area, mesh.enode[under, c], 0.25 * pitch ** 2)
        springs = np.repeat(area * float(kb), 3)
    z_above = (mesh.elem_ijk[:, 2] - plate_cells + 0.5) * pitch
    return mesh, part, layer_of, fixed, z_above, plate_cells, springs


def simulate_build(solid, pitch, origin=(0.0, 0.0, 0.0), setup: BuildSetup | None = None, on_layer=None):
    """Simulate printing a voxel part (bool grid, build direction +z, the
    lowest cells sit on the plate or bed) and releasing it.

    solid: bool (nx, ny, nz); pitch mm; origin: world position of the grid's
    min corner. Returns a BuildResult; its mesh is the part's cells (plus a
    plate, lpbf), in the same world frame."""
    setup = setup or BuildSetup()
    t0 = time.time()
    proc = setup.process.lower()
    mat = _material(setup.material)
    e0 = float(setup.E or mat.E)
    nu = float(setup.nu if setup.nu is not None else mat.nu)
    solid, floating = _grounded(solid, proc)
    mesh, part, layer_of, fixed, z_above, plate_cells, springs = _grid_mesh(solid, pitch, origin, setup)
    ne = mesh.n_elem
    kern = _Kernel(mesh, nu, fixed, springs)
    e = np.full(ne, e0)
    depth = None
    if proc in ("sls", "mjf"):
        from scipy import ndimage
        g = np.zeros(tuple(mesh.elem_ijk.max(axis=0) + 3), bool)
        g[tuple((mesh.elem_ijk + 1).T)] = True
        depth = ndimage.distance_transform_edt(g)[tuple((mesh.elem_ijk + 1).T)] * pitch - 0.5 * pitch
    eps, info = inherent_strain(setup, z=z_above if proc == "fdm" else None, depth=depth)
    eps = np.broadcast_to(eps, (ne, 6)).copy()
    eps[~part] = 0.0
    u = np.zeros(mesh.n_dof)
    ue = np.zeros((ne, 24))              # each element's displacement since it was laid
    lift = []
    xyz = mesh.node_xyz
    layered = proc in ("lpbf", "fdm")
    if layered:
        n_layers = int(layer_of.max()) + 1
        active = ~part
        node_on = np.zeros(mesh.n_nodes, bool)
        if active.any():
            node_on[mesh.enode[active].ravel()] = True
        for li in range(n_layers):
            new = part & (layer_of == li)
            if not new.any():
                continue
            active = active | new
            node_on[mesh.enode[new].ravel()] = True
            k = kern.k(e * active)
            d = _dofs(node_on, fixed)
            f = kern.load(e, eps[new], np.flatnonzero(new))
            du = np.zeros(mesh.n_dof)
            du[d] = _solve(k[d][:, d], f[d], xyz, d)
            u += du
            act = np.flatnonzero(active)
            ue[act] += du[mesh.edof[act]]
            # lift at the build plane (the top of the cells just laid)
            top_z = mesh.elem_ijk[new, 2].max() + 1
            top = node_on & (mesh.node_ijk[:, 2] == top_z)
            lz = u[3 * np.flatnonzero(top) + 2]
            lift.append(float(lz.max()) if len(lz) else 0.0)
            if on_layer is not None:
                on_layer(li, n_layers)
    else:
        # sls/mjf: the part cools as a whole, free in the powder
        node_on = np.zeros(mesh.n_nodes, bool)
        node_on[mesh.enode.ravel()] = True
        n_layers = 0
    report = {"process": proc, "material": setup.material, "strain": info, "pitch": float(pitch),
              "cells": int(part.sum()), "analysisLayers": n_layers,
              "layerMm": float((setup.layer or pitch)) if layered else None}

    if proc == "fdm":
        # stage two: the finished part cools on the bed to room temperature
        p = _fdm_params(setup)
        tz = _fdm_temperature(p, z_above)
        eps2 = np.zeros((ne, 6))
        eps2[:, :3] = (-p["alpha"] * (tz - p["room"]) * setup.scale)[:, None]
        report["peel"] = _bed_peel(mesh, kern, e, ue, eps, part, fixed, u)
        k = kern.k(e)
        d = _dofs(node_on, fixed)
        f = kern.load(e, eps2, np.arange(ne))
        du = np.zeros(mesh.n_dof)
        du[d] = _solve(k[d][:, d], f[d], xyz, d)
        u += du
        ue += du[mesh.edof]
        eps = eps + eps2
        report["peelCooled"] = _bed_peel(mesh, kern, e, ue, eps, part, fixed, u)

    u_built = u.reshape(-1, 3).copy()
    stress_built = kern.stress(e, ue, eps)
    if proc in ("sls", "mjf"):
        release = "full"
    else:
        release = setup.release
    keep = part.copy() if release in ("full", "none") else part & ~_in_box(mesh.elem_centers(), setup.cut)
    if release == "none":
        u_rel, stress = u_built.copy(), stress_built.copy()
        node_act = node_on.copy()
    else:
        if release == "cut" and setup.cut is None:
            raise ValueError("release='cut' needs BuildSetup.cut, the box of cells the cut removes")
        if release == "cut" and plate_cells:
            keep = keep | ~part          # the plate stays, the cut slab goes
        node_act = np.zeros(mesh.n_nodes, bool)
        node_act[mesh.enode[keep].ravel()] = True
        comps = _components(mesh, keep & part) if release == "full" else []
        if release == "full":
            # each piece the release frees gets its own 3-2-1
            fx = np.zeros(mesh.n_dof, bool)
            for c in comps:
                fx |= _three_two_one(mesh, c)
        else:
            fx = fixed
        kel = np.flatnonzero(keep)
        r = kern.internal(e, ue[kel], eps[kel], kel)
        # assembled without Dirichlet rows: the 3-2-1 set is not `fixed`,
        # and the submatrix over the free dofs is all the solve needs
        k = _assemble_free(kern, e * keep)
        d = _dofs(node_act, fx)
        du = np.zeros(mesh.n_dof)
        du[d] = _solve(k[d][:, d], -r[d], xyz, d)
        u_tot = u + du
        ue[kel] += du[mesh.edof[kel]]
        stress = kern.stress(e, ue, eps)
        stress[~keep] = 0.0
        u_rel = u_tot.reshape(-1, 3).copy()
        u_rel[~node_act] = 0.0
        for c in comps:
            ids = np.flatnonzero(c)
            u_rel[ids] = _remove_rigid(xyz[ids], u_rel[ids])
    vm_b = von_mises(stress_built[part])
    vm = von_mises(stress[keep])
    sy = float(setup.yield_strength or mat.strength)
    pn = np.flatnonzero(node_act)
    dmag = np.linalg.norm(u_rel[pn], axis=1)
    report.update({
        "release": release,
        "floatingCells": floating,
        "distortion": {"max": float(dmag.max()), "rms": float(np.sqrt((dmag ** 2).mean())),
                       "maxZ": float(np.abs(u_rel[pn, 2]).max()),
                       "at": [float(c) for c in xyz[pn[np.argmax(dmag)]]]},
        "builtDistortionMax": float(np.linalg.norm(u_built[mesh.enode[part].ravel()], axis=1).max()),
        "residualStress": {"vmMaxBuilt": float(vm_b.max()), "vmP95Built": float(np.percentile(vm_b, 95)),
                           "vmMax": float(vm.max()), "vmP95": float(np.percentile(vm, 95)),
                           "yield": sy, "overYieldShare": float((vm > sy).mean())},
        "seconds": round(time.time() - t0, 2),
    })
    if proc == "lpbf" and lift:
        report["recoater"] = {"maxLift": float(max(lift)), "gap": setup.recoater_gap,
                              "layer": int(np.argmax(lift)), "risk": bool(max(lift) > setup.recoater_gap)}
    report["warnings"] = _warnings(report, setup)
    res = BuildResult(mesh=mesh, part=part, layer_of=layer_of, u_built=u_built, u=u_rel,
                      stress_built=stress_built, stress=stress, strain=eps, report=report, node_active=node_act)
    return res


def _grounded(solid, proc):
    """The part's cells that reach the plate or bed through faces (lpbf,
    fdm), and how many were dropped: a piece printed in mid-air has
    nothing to hold it, in the machine or in the solve. Powder-bed polymer
    keeps every piece (the cake holds them)."""
    from scipy import ndimage
    solid = np.asarray(solid, bool)
    if proc in ("sls", "mjf") or not solid.any():
        return solid, 0
    lbl, _ = ndimage.label(solid)
    k0 = np.argwhere(solid)[:, 2].min()
    keep = np.isin(lbl, np.unique(lbl[:, :, k0][solid[:, :, k0]]))
    return solid & keep, int((solid & ~keep).sum())


def _components(mesh, cells):
    """Node masks of the face-connected pieces of `cells` (bool per element)."""
    from scipy import ndimage
    g = np.zeros(tuple(mesh.elem_ijk.max(axis=0) + 1), int)
    idx = np.flatnonzero(cells)
    g[tuple(mesh.elem_ijk[idx].T)] = 1
    lbl, n = ndimage.label(g)
    el = lbl[tuple(mesh.elem_ijk[idx].T)]
    out = []
    for i in range(1, n + 1):
        on = np.zeros(mesh.n_nodes, bool)
        on[mesh.enode[idx[el == i]].ravel()] = True
        out.append(on)
    return out


def _assemble_free(kern, scale):
    """Stiffness without Dirichlet rows (a free part's)."""
    st = kern.st
    data = np.zeros(st.nnz)
    s = np.asarray(scale, float)[:, None]
    for slots, blk in st._pairs:
        data[slots] += s * blk
    return sp.csr_matrix((data, st.indices, st.indptr), shape=(kern.mesh.n_dof, kern.mesh.n_dof))


def _in_box(pts, box):
    if box is None:
        return np.zeros(len(pts), bool)
    lo, hi = (np.asarray(b, float) for b in box)
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def _bed_peel(mesh, kern, e, ue, eps, part, fixed, u_now=None):
    """FDM: the bed's pull on the part (MPa, positive = the part trying to
    lift off) per bed node, from the reactions of the fixed underside."""
    el = np.flatnonzero(part)
    r = kern.internal(e, ue[el], eps[el], el)
    bed = np.flatnonzero(mesh.node_ijk[:, 2] == 0)
    fz = r[3 * bed + 2]               # internal force; the bed's reaction is -fz
    if kern.springs is not None:
        # an elastic bed: its pull is the spring's stretch (the 3-2-1
        # nodes that hold the rigid motion carry no bed)
        fz = -kern.springs[3 * bed + 2] * u_now[3 * bed + 2]
    # tributary area: a quarter cell per touching cell
    touch = np.zeros(mesh.n_nodes)
    bottom = el[mesh.elem_ijk[el, 2] == 0]
    for c in range(4):                # the four bottom corners
        np.add.at(touch, mesh.enode[bottom, c], 0.25)
    area = np.maximum(touch[bed], 1e-9) * mesh.pitch ** 2
    peel = -fz / area                 # reaction pulling down (on a part that wants up) -> positive
    pos = peel > 0
    lift = float((peel[pos] * area[pos]).sum())
    # the max sits at an edge, where a bonded joint's peel stress is
    # singular (it grows as the voxels shrink); the mean over the area in
    # tension is the steadier number
    return {"maxPeelStress": float(peel.max()), "meanPeelStress": lift / max(float(area[pos].sum()), 1e-12),
            "peelArea": float(area[pos].sum()),
            "at": [float(c) for c in mesh.node_xyz[bed[np.argmax(peel)]]],
            "netLiftForce": lift}


def _warnings(rep, setup):
    w = []
    d = rep["distortion"]
    if rep.get("recoater", {}).get("risk"):
        w.append(f"build: the build plane lifts {rep['recoater']['maxLift']:.3f} mm "
                 f"(> {setup.recoater_gap} mm) at analysis layer {rep['recoater']['layer']}: recoater contact "
                 "risk; support the overhang or change the orientation")
    rs = rep["residualStress"]
    if rs["vmMaxBuilt"] > rs["yield"]:
        w.append(f"build: residual stress on the plate reaches {rs['vmMaxBuilt']:.0f} MPa "
                 f"(yield {rs['yield']:.0f}); expect local yielding and cracking risk at sharp corners "
                 "(a linear model overstates the peak where it passes yield)")
    if rep["process"] == "fdm":
        pk = rep.get("peel", {}).get("meanPeelStress", 0.0)
        if pk > 0.5:
            w.append(f"build: the part pulls on the bed at {pk:.2f} MPa during the print: corners may lift "
                     "(PEI/glue hold roughly 0.3-1 MPa); add brim/mouse ears or a hotter chamber")
    if rep.get("floatingCells"):
        w.append(f"build: {rep['floatingCells']} cells are not connected to the plate or bed and were left out")
    if d["max"] > 0.2:
        w.append(f"build: distortion after release up to {d['max']:.2f} mm; consider compensate()")
    return w


# --------------------------------------------------------- mesh entry points


def to_build_frame(points, build):
    """Rotation (3x3) taking a build direction ('+z', (axis, sign) or a
    vector) to +z, and the points rotated by it."""
    from .am import build_vector, parse_build_direction
    v = build_vector(parse_build_direction(build) if isinstance(build, str) else build)
    z = np.array([0.0, 0.0, 1.0])
    c = float(v @ z)
    if c > 1 - 1e-12:
        r = np.eye(3)
    elif c < -1 + 1e-12:
        r = np.diag([1.0, -1.0, -1.0])
    else:
        ax = np.cross(v, z)
        s = np.linalg.norm(ax)
        ax /= s
        kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        r = np.eye(3) + s * kx + (1 - c) * kx @ kx
    pts = None if points is None else np.asarray(points, float) @ r.T
    return r, pts


def simulate_part(mesh, setup: BuildSetup | None = None, pitch=None, build="+z", cells=60_000):
    """Simulate a triangle mesh (trimesh) printed along `build`. Voxelizes
    at `pitch` (default: about `cells` cells in the part). The BuildResult
    is in the build frame; result.report["rotation"] is the world->build
    rotation."""
    import trimesh
    setup = setup or BuildSetup()
    from .voxels import mesh_solid
    r, v = to_build_frame(mesh.vertices, build)
    m = trimesh.Trimesh(v, mesh.faces, process=False)
    if pitch is None:
        pitch = float((abs(m.volume) / cells) ** (1 / 3))
    solid, origin, _ = mesh_solid(m, pitch, pad=1)
    res = simulate_build(solid, pitch, origin, setup)
    res.report["rotation"] = r.tolist()
    return res


class _NodeField:
    """Trilinear sampler of a nodal field on the voxel grid; nodes outside
    the part (or not in `active`) are filled from their nearest part node."""

    def __init__(self, mesh: HexMesh, u, active=None):
        from scipy import ndimage
        dims = tuple(mesh.node_dims)
        on = np.zeros(dims, bool)
        ok = np.ones(mesh.n_nodes, bool) if active is None else np.asarray(active, bool)
        ijk = mesh.node_ijk[ok]
        on[tuple(ijk.T)] = True
        grid = np.zeros(dims + (3,))
        grid[tuple(ijk.T)] = np.asarray(u)[ok]
        _, ind = ndimage.distance_transform_edt(~on, return_indices=True)
        self.grid = grid[tuple(ind)]
        self.origin = mesh.origin
        self.pitch = mesh.pitch

    def __call__(self, pts):
        from scipy.ndimage import map_coordinates
        q = (np.asarray(pts, float) - self.origin) / self.pitch
        return np.stack([map_coordinates(self.grid[..., c], q.T, order=1, mode="nearest") for c in range(3)],
                        axis=1)


# --------------------------------------------------------- compensation


def compensate(points, sample, factor=1.0):
    """Pre-distort `points` against a displacement sampler: the inverse
    of x -> x + d(x), by fixed point (y = x - factor * d(y), a few rounds),
    so that printing y lands on x."""
    x = np.asarray(points, float)
    y = x.copy()
    for _ in range(4):
        y = x - factor * sample(y)
    return y


def compensate_mesh(mesh, setup: BuildSetup, pitch=None, build="+z", iterations=2, cells=60_000, factor=1.0):
    """Iterative pre-distortion of a triangle mesh: simulate, move every
    vertex against its predicted displacement, simulate the compensated
    shape, correct by what is left, and so on. Returns (compensated
    trimesh, report with the residual distortion after each round; round 0
    is the uncompensated part)."""
    import trimesh
    r, v_nom = to_build_frame(mesh.vertices, build)
    nominal = trimesh.Trimesh(v_nom, mesh.faces, process=False)
    if pitch is None:
        pitch = float((abs(nominal.volume) / cells) ** (1 / 3))
    from .voxels import mesh_solid
    cur = v_nom.copy()
    rounds = []
    for it in range(iterations + 1):
        m = trimesh.Trimesh(cur, mesh.faces, process=False)
        solid, origin, _ = mesh_solid(m, pitch, pad=1)
        res = simulate_build(solid, pitch, origin, setup)
        printed = cur + res.displacement_at(cur)
        if res.report["release"] == "full":
            printed = _align(printed, v_nom)
        err = np.linalg.norm(printed - v_nom, axis=1)
        rounds.append({"round": it, "maxDeviation": float(err.max()), "rmsDeviation": float(np.sqrt((err ** 2).mean())),
                       "simDistortionMax": res.report["distortion"]["max"]})
        if it == iterations:
            break
        cur = cur - factor * (printed - v_nom)
    out = trimesh.Trimesh(cur @ r, mesh.faces, process=False)  # back to the world frame
    return out, {"rounds": rounds, "pitch": pitch, "process": setup.process, "material": setup.material}


def _align(p, q):
    """p rigidly moved onto q (Kabsch, least squares)."""
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    h = (p - pc).T @ (q - qc)
    uu, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ uu.T))
    rot = vt.T @ np.diag([1, 1, d]) @ uu.T
    return (p - pc) @ rot.T + qc


# --------------------------------------------------------- analytic check


def progressive_beam(e, h_sub, layers, eps):
    """Curvature of a free beam (unit width, modulus e, uniaxial) after
    depositing `layers` (thicknesses) on a substrate of thickness h_sub,
    each layer stress-free when laid and then given misfit strain `eps`
    (Tsui & Clyne 1997, progressive deposition, one material). Returns
    (curvature 1/mm, positive = concave up, per-layer stress list)."""
    zs = [0.0, h_sub]
    kappa = 0.0
    for t in layers:
        z0, z1 = zs[-1], zs[-1] + t
        a = z1 - 0.0            # current composite spans [0, z1]
        sz = 0.5 * z1 ** 2
        szz = z1 ** 3 / 3.0
        # new layer's misfit force and moment
        fn = e * eps * t
        mn = e * eps * 0.5 * (z1 ** 2 - z0 ** 2)
        m = np.array([[a, sz], [sz, szz]]) * e
        d0, dk = np.linalg.solve(m, [fn, mn])
        kappa += dk
        zs.append(z1)
    return kappa


# --------------------------------------------------------- optimization


@dataclass
class WarpOptions:
    volume_fraction: float
    filter_radius: float
    process: BuildSetup = field(default_factory=BuildSetup)
    warp_weight: float = 1.0            # weight of normalized distortion against normalized compliance
    layer: float | None = None          # mm per analysis layer (default 4 cells)
    max_iterations: int = 80
    beta_max: float = 16.0
    beta_every: int = 12
    eta: float = 0.5
    erode: float = 0.15
    move: float = 0.1
    tol: float = 0.01
    x0: np.ndarray | None = None
    extrusion: int | None = None
    target: np.ndarray | None = None    # bool per node whose distortion counts (default every node)


@dataclass
class WarpResult:
    x: np.ndarray
    xphys: np.ndarray
    compliance: float
    distortion: float                   # mean squared as-built displacement over the target, mm^2
    volume_fraction: float
    iterations: int
    converged: bool
    seconds: float
    history: list = field(default_factory=list)


def _warp_problem(grid, supports, force, opts: WarpOptions):
    """evaluate(xe) -> (objective, gradient, record) of optimize_warp, on
    the eroded density xe."""
    from .simp import E_MIN
    from scipy.sparse.linalg import splu

    mesh = grid.mesh
    setup = opts.process
    mat = _material(setup.material)
    e0 = float(setup.E or mat.E)
    nu = float(setup.nu if setup.nu is not None else mat.nu)
    p = 3.0
    ne = mesh.n_elem
    z_lo = mesh.elem_ijk[:, 2].min()
    plate_fixed = np.repeat(mesh.node_ijk[:, 2] == z_lo, 3)
    kern_b = _Kernel(mesh, nu, plate_fixed)
    st_s = Stiffness(mesh, nu, np.asarray(supports, bool))
    z_above = (mesh.elem_ijk[:, 2] - z_lo + 0.5) * mesh.pitch
    eps, _ = inherent_strain(setup, z=z_above if setup.process == "fdm" else None)
    eps = np.broadcast_to(eps, (ne, 6)).copy()
    lump = max(1, int(round((opts.layer or 4 * mesh.pitch) / mesh.pitch)))
    layer_of = (mesh.elem_ijk[:, 2] - z_lo) // lump
    n_layers = int(layer_of.max()) + 1
    target = np.ones(mesh.n_nodes, bool) if opts.target is None else np.asarray(opts.target, bool)
    wdof = np.repeat(target, 3).astype(float) / max(target.sum(), 1)
    f_s = np.asarray(force, float)
    ke = kern_b.st.ke
    g = kern_b.g
    ref = {}

    def evaluate(xe):
        e = e0 * (E_MIN + (1 - E_MIN) * xe ** p)
        de = e0 * (1 - E_MIN) * p * xe ** (p - 1)
        # service compliance
        us = _solve(st_s.assemble(e), f_s, mesh.node_xyz)
        comp = float(f_s @ us)
        dc = -de * st_s.strain_energy(us)
        # build: forward pass, keeping each layer's factorization
        u = np.zeros(mesh.n_dof)
        steps = []
        node_on = np.zeros(mesh.n_nodes, bool)
        active = np.zeros(ne, bool)
        for li in range(n_layers):
            new = layer_of == li
            active |= new
            node_on[mesh.enode[new].ravel()] = True
            d = _dofs(node_on, plate_fixed)
            lu = splu(kern_b.st.assemble(e * active)[d][:, d].tocsc(), permc_spec="MMD_AT_PLUS_A")
            el = np.flatnonzero(new)
            f = kern_b.load(e, eps[el], el)
            du = np.zeros(mesh.n_dof)
            du[d] = lu.solve(f[d])
            u += du
            steps.append((d, lu, el, du, active.copy()))
        dist = float((wdof * u * u).sum())
        # adjoint: J = u^T W u with u the sum of the layer increments, so
        # dJ/d du_k = 2 W u on step k's dofs; K_k lam_k = that, and
        # dJ/de = sum_k lam_k^T (df_k/de - dK_k/de du_k)
        gu = 2.0 * wdof * u
        dd = np.zeros(ne)
        for d, lu, el, du, act in steps:
            lam = np.zeros(mesh.n_dof)
            lam[d] = lu.solve(gu[d])
            dd[el] += np.einsum("ij,ij->i", lam[mesh.edof[el]], eps[el] @ g.T)
            ia = np.flatnonzero(act)
            dd[ia] -= np.einsum("ij,jk,ik->i", lam[mesh.edof[ia]], ke, du[mesh.edof[ia]])
        dd *= de
        if not ref:
            ref["c"], ref["d"] = comp, max(dist, 1e-30)
        obj = comp / ref["c"] + opts.warp_weight * dist / ref["d"]
        grad = dc / ref["c"] + opts.warp_weight * dd / ref["d"]
        return obj, grad, {"compliance": comp, "distortion": dist}

    return evaluate


def optimize_warp(grid, supports, force, opts: WarpOptions, on_iter=None, should_stop=None):
    """Topology optimization against service compliance and build
    distortion. `grid`: a thermal.ThermalGrid (the design cells, build
    direction +z, the lowest layer on the plate, held at its nodes there);
    `supports`: bool per dof fixed in service; `force`: service load
    (n_dof). Objective: c / c0 + warp_weight * D / D0, D the mean squared
    displacement of the target nodes at the end of the layer-by-layer
    build (inherent strain from opts.process, lumped to opts.layer), c0
    and D0 the first iterate's. warp_weight 0 is plain compliance."""
    from types import SimpleNamespace

    from .thermal import _mma_loop

    evaluate = _warp_problem(grid, supports, force, opts)
    o = SimpleNamespace(volume_fraction=opts.volume_fraction, filter_radius=opts.filter_radius,
                        max_iterations=opts.max_iterations, beta_max=opts.beta_max, beta_every=opts.beta_every,
                        eta=opts.eta, erode=opts.erode, move=opts.move, tol=opts.tol, x0=opts.x0,
                        extrusion=opts.extrusion)
    pres = grid.preserve.copy()
    x, xp, it, conv, hist, secs = _mma_loop(grid, o, pres, evaluate, on_iter, should_stop)
    last = hist[-1] if hist else {}
    return WarpResult(x=x, xphys=xp, compliance=float(last.get("compliance", np.nan)),
                      distortion=float(last.get("distortion", np.nan)),
                      volume_fraction=float(xp[~pres].mean()), iterations=it, converged=conv, seconds=secs,
                      history=hist)
