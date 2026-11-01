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
    # FDM: tg is where an amorphous road stops flowing (DSC Tg); tc, for a
    # semi-crystalline one (PP, nylon), where it crystallizes on cooling
    # and turns solid, far above its Tg. relax is the share of the thermal
    # shrink below that the polymer keeps as stress (the rest creeps out
    # while the part is warm). bed/chamber: typical settings.
    "fdm": {
        "abs": {"tg": (101.1, "polylite-abs DSC"), "bed": 100.0, "chamber": 35.0},
        "asa": {"tg": (97.8, "polylite-asa DSC"), "bed": 100.0, "chamber": 35.0},
        "petg": {"tg": (80.0, "estimate: PETG copolyester Tg 78-81 C (Eastman Eastar 6763, DSC)"),
                 "bed": 80.0, "chamber": 30.0},
        "pla": {"tg": (60.0, "estimate: PLA Tg 55-65 C"), "bed": 60.0, "chamber": 30.0},
        "pa-cf": {"tg": (70.0, "bambu-paht-cf DSC"), "bed": 90.0, "chamber": 40.0},
        "pp": {"tc": (115.0, "estimate: isotactic PP crystallizes at 110-120 C on cooling (DSC, 10 K/min)"),
               "bed": 100.0, "chamber": 25.0},
        "pa12": {"tc": (150.0, "estimate: PA12 crystallizes at 145-155 C on cooling (DSC, 10 K/min)"),
                 "bed": 90.0, "chamber": 40.0},
    },
    # SLS: ds the extra linear shrink of a thick section over a thin wall;
    # depth_sat the depth (mm) past which a section counts as thick.
    "sls": {"ds": 0.003, "depth_sat": 4.0},
}

# LPBF strain factor k (eps = -k * yield / E), fitted to the NIST AM-Bench
# 2018 bridges at 0.5 mm voxels, one analysis layer per voxel, the 12.7 mm
# plate reduced to 2 voxel layers (plate_cells; on the whole plate the fit
# was 0.909) (bench/buildsim_validate.py; docs/BUILDSIM.md). The yield-strain
# scaling does not carry across alloy classes: with IN625's k, 15-5 PH
# came out 60% high (its martensite forms on cooling and expands, which
# undoes part of the shrink), so 15-5 PH has its own fit. Alloys without a
# fit use IN625's and the report says so.
LPBF_K = 0.839
LPBF_K_FIT = {"in625": 0.839, "15-5ph": 0.522}
# FDM retained share of the sub-Tg shrink, and the bed's hold (N/mm^3 of
# elastic foundation): fitted together on the nine 20 mm-wide ABS bars of
# Armillotta et al. 2018 (bench/buildsim_fdm.py; rms 0.16 mm, mean 24%;
# the bed pinned only in plane, so corners can lift). The decay length
# (mm) of the bed's heat up the part is an estimate. FDM_ADHESION: MPa of
# bond tension before the first layer lets go; off by default, since
# letting go made the Armillotta fit worse (it flattens the tall bars and
# leaves the thin ones), see docs/BUILDSIM.md.
FDM_RELAX = 0.895
FDM_BED = 4.0
FDM_DECAY = 6.0
FDM_ADHESION = float("inf")
# FDM: the printed layer (mm) builds are extrapolated to when their analysis
# layers are thicker (layer_extrapolation); 0.2 mm is the common setting.
FDM_PRINT_LAYER = 0.2
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
    # at most this many cells of plate; a thinner model plate is made as
    # compliant as the real one through the thickness (modulus scaled by
    # the thickness ratio), for a plate held along its underside only.
    # None = model the whole plate. On AM-Bench at 0.5 mm voxels, 2 cells
    # ran in 65 s against 181 s and rose 8% more; LPBF_K is fitted with 2.
    plate_cells: int | None = 2
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
    # fdm: peel strength of the bed bond, MPa: where the bed's pull on the
    # first layer passes it, that spot lets go for good (corner lift).
    # None = FDM_ADHESION, inf = never
    bed_adhesion: float | None = None
    recoater_gap: float = 0.05          # mm: lift at the build plane that can touch the recoater
    # fdm: the printed layer, mm (None: FDM_PRINT_LAYER; 0: no extrapolation)
    print_layer: float | None = None


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
        info.update({"model": "solidification shrink", **{k: v for k, v in p.items() if k != "tgSource"},
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
    """The FDM shrink model's numbers. 'tg' is the temperature the road
    turns solid at: Tg for an amorphous polymer, the crystallization
    temperature for a semi-crystalline one (the BUILD row's 'tc')."""
    from .thermal import expansion
    name = str(setup.material).lower()
    row = BUILD["fdm"].get(name)
    if row is not None and "tc" in row:
        tg, src = row["tc"]
        kind = "semi-crystalline: solid below crystallization"
    else:
        kind = "amorphous: solid below Tg"
        tg = _lib(name, "tg")
        src = "material library DSC"
        if row is not None and tg is None:
            tg, src = row["tg"]
    if tg is None:
        raise ValueError(f"fdm build simulation: no glass transition or crystallization temperature for {name!r}; "
                         "give BuildSetup.strain")
    alpha = float(setup.alpha) if setup.alpha is not None else float(expansion(name))
    bed = setup.bed if setup.bed is not None else (row or {}).get("bed", 60.0)
    ch = setup.chamber if setup.chamber is not None else (row or {}).get("chamber", 30.0)
    return {"tg": float(tg), "tgSource": src, "solidifies": kind, "alpha": alpha, "bed": float(bed),
            "chamber": float(ch), "room": float(setup.room),
            "relax": FDM_RELAX if setup.relax is None else float(setup.relax),
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


_DIRECT = 90_000      # dofs: a direct factorization up to here


def _solve(k, f, xyz, d=None, direct=_DIRECT, rtol=1e-7):
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


def _in_plane_pins(mesh, nodes):
    """Pins for a part on an elastic bed: the springs carry z, so only the
    in-plane drift is held, at the node nearest the footprint's centre
    (x, y) and one far from it along x (y). Pinning z at corners, as a
    3-2-1 does, would hold down the corners that lift."""
    idx = np.flatnonzero(nodes)
    xyz = mesh.node_xyz[idx]
    a = idx[np.argmin(np.linalg.norm(xyz[:, :2] - xyz[:, :2].mean(axis=0), axis=1))]
    pa = mesh.node_xyz[a]
    b = idx[np.argmax(np.abs(xyz[:, 0] - pa[0]) - 1e-3 * np.abs(xyz[:, 1] - pa[1]))]
    fixed = np.zeros(mesh.n_dof, bool)
    fixed[3 * a:3 * a + 2] = True
    fixed[3 * b + 1] = True
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
        if setup.plate_cells is not None and setup.plate_fixed:
            # the scaled modulus keeps a held plate's through-thickness
            # compliance (E / h); a free plate bends (E h^3), so it stays whole
            plate_cells = min(plate_cells, max(1, int(setup.plate_cells)))
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
        fixed = _in_plane_pins(mesh, bottom)
        area = np.zeros(mesh.n_nodes)
        under = mesh.elem_ijk[:, 2] == 0
        for c in range(4):
            np.add.at(area, mesh.enode[under, c], 0.25 * pitch ** 2)
        springs = np.repeat(area * float(kb), 3)
    z_above = (mesh.elem_ijk[:, 2] - plate_cells + 0.5) * pitch
    return mesh, part, layer_of, fixed, z_above, plate_cells, springs


def layer_extrapolation(t, setup) -> float:
    """k for extrapolating an FDM build at analysis layers t mm to the
    printed layer: u = (1 + k) u(t) - k u(2t), k = log2(t / printed).

    FDM warp grows as the analysis layers thin and does not settle: a
    60 x 4 x 3 mm nylon bar on the bed lifts 0.30, 0.46, 0.62, 0.80 mm at
    1, 0.5, 0.25, 0.125 mm layers, whatever the voxel size (0.125-0.5 mm
    voxels agree within 1%), about +0.16 mm per halving. Each layer goes
    down stress-free on a part already curled by the ones below, so thin
    layers keep adding curl. The model was fitted at the printed layers
    (Armillotta, 0.254 mm), so that is where it must be read: the gain per
    halving between t and 2t, carried log-linearly down to the printed
    layer, lands within 4% of the direct thin-layer runs (0.61 against
    0.62, and 0.92 against 0.96 on a 5.5 mm bar)."""
    if setup is None or setup.process.lower() != "fdm":
        return 0.0
    tp = FDM_PRINT_LAYER if setup.print_layer is None else float(setup.print_layer)
    if tp <= 0 or t <= 1.05 * tp:
        return 0.0
    return float(np.log2(t / tp))


def simulate_build(solid, pitch, origin=(0.0, 0.0, 0.0), setup: BuildSetup | None = None, on_layer=None):
    """Simulate printing a voxel part (bool grid, build direction +z, the
    lowest cells sit on the plate or bed) and releasing it.

    solid: bool (nx, ny, nz); pitch mm; origin: world position of the grid's
    min corner. Returns a BuildResult; its mesh is the part's cells (plus a
    plate, lpbf), in the same world frame. FDM analysis layers thicker than
    the printed layer are extrapolated to it (layer_extrapolation): the
    build runs twice, at its layers and at twice them."""
    setup = setup or BuildSetup()
    t = float(setup.layer or pitch)
    kx = layer_extrapolation(t, setup)
    r1 = _simulate_build(solid, pitch, origin, setup, on_layer)
    if not kx:
        return r1
    from dataclasses import replace
    r2 = _simulate_build(solid, pitch, origin, replace(setup, layer=2 * t))
    lin = (lambda a, b: (1.0 + kx) * a - kx * b)
    res = replace(r1, u_built=lin(r1.u_built, r2.u_built), u=lin(r1.u, r2.u),
                  stress_built=lin(r1.stress_built, r2.stress_built), stress=lin(r1.stress, r2.stress))
    rep = res.report
    m, part = res.mesh, res.part
    pn = np.flatnonzero(res.node_active)
    dmag = np.linalg.norm(res.u[pn], axis=1)
    rep["distortion"] = {"max": float(dmag.max()), "rms": float(np.sqrt((dmag ** 2).mean())),
                         "maxZ": float(np.abs(res.u[pn, 2]).max()),
                         "at": [float(c) for c in m.node_xyz[pn[np.argmax(dmag)]]]}
    rep["builtDistortionMax"] = float(np.linalg.norm(res.u_built[m.enode[part].ravel()], axis=1).max())
    keep = np.abs(res.stress).sum(axis=1) > 0
    vm_b, vm = von_mises(res.stress_built[part]), von_mises(res.stress[keep & part] if (keep & part).any()
                                                            else res.stress[part])
    rs = rep["residualStress"]
    rs.update(vmMaxBuilt=float(vm_b.max()), vmP95Built=float(np.percentile(vm_b, 95)), vmMax=float(vm.max()),
              vmP95=float(np.percentile(vm, 95)), overYieldShare=float((vm > rs["yield"]).mean()))
    rep["layerExtrapolation"] = {"analysisLayer": t, "printedLayer": t / 2 ** kx, "k": kx,
                                 "distortionMaxUnextrapolated": r1.report["distortion"]["max"],
                                 "note": "linear in log(layer) from builds at the analysis layer and twice it"}
    rep["warnings"] = _warnings(rep, setup)
    return res


def _simulate_build(solid, pitch, origin=(0.0, 0.0, 0.0), setup: BuildSetup | None = None, on_layer=None):
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
    if plate_cells and plate_cells * pitch < setup.plate - 1e-9:
        e[~part] *= plate_cells * pitch / setup.plate
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
        layer_of = _joined_layers(mesh, part, layer_of)
        n_layers = int(layer_of.max()) + 1
        active = ~part
        node_on = np.zeros(mesh.n_nodes, bool)
        if active.any():
            node_on[mesh.enode[active].ravel()] = True
        # Big parts solve on mg.py's matrix-free multigrid (cells not yet
        # printed get modulus 0, which drops them; its setup per layer is a
        # reshape, where pyamg's was 60% of the time on AM-Bench); small
        # ones factor each layer exactly.
        big = mesh.n_dof > _DIRECT
        detached = 0
        solver = None
        if big:
            from .mg import GridElasticity
            solver = GridElasticity(mesh, nu, fixed, springs=sp.diags(springs) if np.any(springs) else None,
                                    rtol=1e-7, maxiter=3000, device="cpu")
        for li in range(n_layers):
            new = part & (layer_of == li)
            if not new.any():
                continue
            active = active | new
            node_on[mesh.enode[new].ravel()] = True
            f = kern.load(e, eps[new], np.flatnonzero(new))
            du = np.zeros(mesh.n_dof)
            if big:
                f[fixed] = 0.0
                solver.set_moduli(e * active)
                du = solver.solve_many([f])[0]
                du[~np.repeat(node_on, 3)] = 0.0      # unprinted nodes go down at nominal
            else:
                k = kern.k(e * active)
                d = _dofs(node_on, fixed)
                du[d] = _solve(k[d][:, d], f[d], xyz, d)
            u += du
            act = np.flatnonzero(active)
            ue[act] += du[mesh.edof[act]]
            lost = _detach(kern, setup, u, ue, act, lambda g: _resolve(kern, e * active, node_on, fixed, xyz, g))
            if lost and big:
                solver = GridElasticity(mesh, nu, fixed, springs=sp.diags(kern.springs), rtol=1e-7, maxiter=3000,
                                        device="cpu")
            detached += lost
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
        detached += _detach(kern, setup, u, ue, np.arange(ne), lambda g: _resolve(kern, e, node_on, fixed, xyz, g))
        eps = eps + eps2
        report["detachedArea"] = float(detached)
        report["bedFailed"] = bool(getattr(kern, "bed_failed", False))
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


def _joined_layers(mesh, part, layer_of):
    """The analysis layer at which each part cell joins the plate or bed
    through faces. A cell laid in mid-air (the low tip of a drooping,
    pre-deformed arm; anything printed on supports) has nothing to hold it
    in the solve until the layer that bonds it to the part: it goes in then,
    stress-free in its place, as if its supports had held it."""
    from scipy import ndimage
    ijk = mesh.elem_ijk
    g = np.full(tuple(ijk.max(axis=0) + 1), np.iinfo(np.int64).max, np.int64)
    g[tuple(ijk.T)] = layer_of                       # the plate is -1
    out = layer_of.copy()
    pending = part.copy()
    for li in range(int(layer_of.max()) + 1):
        lbl, _ = ndimage.label(g <= li)
        roots = np.unique(lbl[:, :, 0])
        on = np.isin(lbl[tuple(ijk.T)], roots[roots > 0])
        out[pending & on] = li
        pending &= ~on
        if not pending.any():
            break
    return out


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


def _detach(kern, setup, u, ue, act, solve):
    """FDM bed bond failure: every bonded bed node whose bond pulls harder
    than the adhesion (tension kb * u_z, MPa) lets go, its springs removed,
    and the part re-balances; repeat until none fails. `solve(load)` solves
    the current stiffness (it sees the springs as they are now). Updates u
    and ue in place; returns the bed area let go (mm^2)."""
    if kern.springs is None or setup.process.lower() != "fdm":
        return 0
    adh = FDM_ADHESION if setup.bed_adhesion is None else float(setup.bed_adhesion)
    if not np.isfinite(adh):
        return 0
    kb = float(setup.bed_stiffness if setup.bed_stiffness is not None else FDM_BED)
    mesh = kern.mesh
    bed = np.flatnonzero(mesh.node_ijk[:, 2] == 0)
    lost = 0.0
    for _ in range(60):
        bonded = kern.springs[3 * bed + 2] > 0
        fail = bed[bonded & (kb * u[3 * bed + 2] > adh)]
        if len(fail) == 0:
            break
        if bonded.sum() - len(fail) < max(4, 0.02 * len(bed)):
            # the whole footprint lets go: the part pops off the bed (a
            # failed print); keep the last bonds so the solve stays posed
            kern.bed_failed = True
            break
        lost += float(kern.springs[3 * fail + 2].sum()) / kb      # their bed area, mm^2
        dof = (3 * fail[:, None] + np.arange(3)).ravel()
        load = np.zeros(mesh.n_dof)
        load[dof] = kern.springs[dof] * u[dof]      # what the bond held, now let go
        kern.springs[dof] = 0.0
        du = solve(load)
        u += du
        ue[act] += du[mesh.edof[act]]
    return lost


def _resolve(kern, scale, node_on, fixed, xyz, load):
    """One solve of the current stiffness (springs as they stand) on the
    printed nodes: direct when small, multigrid when big."""
    mesh = kern.mesh
    du = np.zeros(mesh.n_dof)
    if mesh.n_dof > _DIRECT:
        from .mg import GridElasticity
        sv = GridElasticity(mesh, kern.nu, fixed, springs=sp.diags(kern.springs) if kern.springs is not None else None,
                            rtol=1e-7, maxiter=3000, device="cpu")
        sv.set_moduli(scale)
        g = load.copy()
        g[fixed] = 0.0
        du = sv.solve_many([g])[0]
        du[~np.repeat(node_on, 3)] = 0.0
        return du
    k = kern.k(scale)
    d = _dofs(node_on, fixed)
    du[d] = _solve(k[d][:, d], load[d], xyz, d)
    return du


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


def refine_mesh(mesh, size):
    """The mesh with every edge longer than `size` mm split, so a
    displacement sampled at its vertices can bend it (a CAD export's long
    triangles move only at their corners)."""
    import trimesh
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=float(size), max_iter=20)
    return trimesh.Trimesh(v, f, process=True)


def printed_shape(mesh, res: BuildResult, build="+z"):
    """The mesh as printed: every vertex moved by the simulated
    displacement (released), in the mesh's own frame."""
    import trimesh
    r = np.asarray(res.report.get("rotation", np.eye(3)), float)
    v = np.asarray(mesh.vertices, float) @ r.T
    return trimesh.Trimesh((v + res.displacement_at(v)) @ r, mesh.faces, process=False)


def deviation(printed, nominal):
    """How far a printed shape is from the design, as a scan comparison
    reads it: `printed` best-fit onto `nominal` (rigid ICP on the
    surface), then the distance between the two surfaces both ways.
    Returns ({"max", "rms", "p95"} mm, the aligned printed mesh)."""
    import trimesh
    m, _, _ = trimesh.registration.icp(np.asarray(printed.vertices), nominal, max_iterations=40, scale=False)
    aligned = printed.copy()
    aligned.apply_transform(m)
    _, d1, _ = trimesh.proximity.closest_point(nominal, aligned.vertices)
    _, d2, _ = trimesh.proximity.closest_point(aligned, nominal.vertices)
    d = np.concatenate([d1, d2])
    return {"max": float(d.max()), "rms": float(np.sqrt((d ** 2).mean())),
            "p95": float(np.percentile(d, 95))}, aligned


def compensate_mesh(mesh, setup: BuildSetup, pitch=None, build="+z", iterations=2, cells=60_000, factor=1.0,
                    refine=True, first: BuildResult | None = None):
    """Iterative pre-distortion of a triangle mesh: simulate, move every
    vertex against its predicted displacement, simulate the compensated
    shape, correct by what is left, and so on. Returns (compensated
    trimesh, report with the residual distortion after each round; round 0
    is the uncompensated part). `refine` first splits edges longer than
    the voxel pitch, so long flat triangles can take the shape's curl.
    `first`: simulate_part's result for this mesh at this pitch and build,
    if already run (round 0 reuses it)."""
    import trimesh
    if pitch is None:
        pitch = float((abs(mesh.volume) / cells) ** (1 / 3))
    if refine:
        mesh = refine_mesh(mesh, pitch)
    r, v_nom = to_build_frame(mesh.vertices, build)
    nominal = trimesh.Trimesh(v_nom, mesh.faces, process=False)
    from .voxels import mesh_solid
    cur = v_nom.copy()
    rounds = []
    for it in range(iterations + 1):
        if it == 0 and first is not None:
            res = first
        else:
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

    mesh = grid.mesh
    setup = opts.process
    mat = _material(setup.material)
    e0 = float(setup.E or mat.E)
    nu = float(setup.nu if setup.nu is not None else mat.nu)
    p = 3.0
    warp = WarpTerm(mesh, setup, build=(2, 1), layer=opts.layer, target=opts.target, nu=nu)
    st_s = Stiffness(mesh, nu, np.asarray(supports, bool))
    f_s = np.asarray(force, float)
    ref = {}

    def evaluate(xe):
        e = e0 * (E_MIN + (1 - E_MIN) * xe ** p)
        de = e0 * (1 - E_MIN) * p * xe ** (p - 1)
        # service compliance
        us = _solve(st_s.assemble(e), f_s, mesh.node_xyz)
        comp = float(f_s @ us)
        dc = -de * st_s.strain_energy(us)
        dist, dd = warp.evaluate(e, de)
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


# --------------------------------------------------------- the warp term


class WarpTerm:
    """The as-built distortion of a density design and its gradient: the
    layer-by-layer build of buildsim on a fixed voxel mesh, every cell
    present at modulus e (SIMP), the lowest layer's underside held by the
    plate. D = mean over the target nodes of |u|^2, u the displacement at
    the end of the build (still on the plate). Shared by optimize_warp and
    the study optimizer (simp, buildSim.warpWeight).

    build: (axis, sign), the direction layers stack (a grid axis; a tilted
    build has no layers on this grid); layer: mm per analysis layer
    (default 4 cells)."""

    DIRECT = 90_000          # dofs: factor each layer once for the forward and adjoint solves

    def __init__(self, mesh: HexMesh, setup: BuildSetup, build=(2, 1), layer=None, target=None, nu=None):
        if build is None or len(build) != 2:
            raise ValueError("the warp term needs a build direction along a grid axis ('+z', '-y', ...); "
                             "a tilted build direction has no layers on the voxel grid")
        axis, sign = int(build[0]), int(build[1])
        self.mesh = mesh
        self.axis = axis
        mat = _material(setup.material)
        nu = float(nu if nu is not None else (setup.nu if setup.nu is not None else mat.nu))
        hgt = sign * mesh.elem_ijk[:, axis]
        hgt = hgt - hgt.min()
        hn = sign * mesh.node_ijk[:, axis]
        self.sign = sign
        self.hgt = hgt
        self.hn = hn - hn.min()
        self.fdm = setup.process.lower() == "fdm"
        inplane = [a for a in range(3) if a != axis]
        springs = None
        if self.fdm:
            # FDM: the elastic bed of simulate_build (springs on the first
            # layer's underside, a quarter cell of area per touching cell),
            # pinned only against drift in its plane, so corners can lift
            kb = float(setup.bed_stiffness if setup.bed_stiffness is not None else FDM_BED)
            area = np.zeros(mesh.n_nodes)
            under = hgt == 0
            for c in range(8):
                nd = mesh.enode[under, c]
                on = self.hn[nd] == 0
                np.add.at(area, nd[on], 0.25 * mesh.pitch ** 2)
            springs = np.repeat(area * kb, 3)
            bottom = np.flatnonzero(self.hn == 0)
            pc = mesh.node_xyz[bottom][:, inplane]
            a = bottom[np.argmin(np.linalg.norm(pc - pc.mean(axis=0), axis=1))]
            b = bottom[np.argmax(np.abs(pc[:, 0] - mesh.node_xyz[a, inplane[0]]))]
            self.fixed = np.zeros(mesh.n_dof, bool)
            self.fixed[3 * a + inplane[0]] = self.fixed[3 * a + inplane[1]] = True
            self.fixed[3 * b + inplane[1]] = True
        else:
            self.fixed = np.repeat(hn == hn.min(), 3)
        self.kern = _Kernel(mesh, nu, self.fixed, springs)
        # the distortion measure: |u| for a plate-held metal build; along
        # the build only for FDM, where the in-plane shrink is uniform (the
        # slicer scales it out) and the warp is the lift off the bed
        self.mask = np.ones(3)
        if self.fdm:
            self.mask = np.zeros(3)
            self.mask[axis] = 1.0
        self.cool = None
        if self.fdm:
            p = _fdm_params(setup)
            tz = _fdm_temperature(p, (hgt + 0.5) * mesh.pitch)
            self.cool = np.zeros((mesh.n_elem, 6))
            self.cool[:, :3] = (-p["alpha"] * (tz - p["room"]) * setup.scale)[:, None]
        eps, self.info = inherent_strain(setup, z=(hgt + 0.5) * mesh.pitch if setup.process == "fdm" else None)
        eps = np.broadcast_to(eps, (mesh.n_elem, 6))
        # build frame (in plane, in plane, along the build) -> grid axes
        world = np.zeros((mesh.n_elem, 6))
        world[:, inplane[0]] = eps[:, 0]
        world[:, inplane[1]] = eps[:, 1]
        world[:, axis] = eps[:, 2]
        self.eps = world
        # default lumps: 4 cells for metal; 1 for FDM, whose warp builds up
        # layer by layer on a bed that gives (4-cell lumps read it 2-3x low)
        lump = max(1, int(round((layer or (1 if setup.process.lower() == "fdm" else 4) * mesh.pitch) / mesh.pitch)))
        self.layer_of = hgt // lump
        self.n_layers = int(self.layer_of.max()) + 1
        # FDM: lumps thicker than the printed layer read the warp low (it
        # grows as the layers thin); extrapolate (layer_extrapolation)
        self.extrapolate = 0.0
        if self.fdm:
            self.extrapolate = layer_extrapolation(lump * mesh.pitch, setup)
            self.layer_of2 = hgt // (2 * lump)
        target = np.ones(mesh.n_nodes, bool) if target is None else np.asarray(target, bool)
        self.wdof = np.repeat(target, 3).astype(float) / max(int(target.sum()), 1)
        self.last_u = None

    def evaluate(self, e, de=None):
        """(D, dD/dx) for moduli e (per cell) and their derivative de with
        respect to the design density (None: D only)."""
        vals, grad = self.measures(e, de, lambda v: {"mean": 1.0})
        return vals["mean"], grad

    def _forward(self, e, layer_of, keep):
        """One build at the given layer lumps: the end displacement, each
        step's (dofs, solve, cells, du, active, eps) when keep, and the
        build-plane rises (step, dofs, rise)."""
        from scipy.sparse.linalg import splu
        mesh, kern = self.mesh, self.kern
        ne = mesh.n_elem
        u = np.zeros(mesh.n_dof)
        steps, lifts = [], []
        node_on = np.zeros(mesh.n_nodes, bool)
        active = np.zeros(ne, bool)

        def step(scale, f):
            d = _dofs(node_on, self.fixed)
            k = kern.k(scale)[d][:, d]
            du = np.zeros(mesh.n_dof)
            if len(d) <= self.DIRECT:
                lu = splu(k.tocsc(), permc_spec="MMD_AT_PLUS_A")
                du[d] = lu.solve(f[d])
                return d, lu.solve, du
            du[d] = _solve(k, f[d], mesh.node_xyz, d)
            return d, (lambda kk, dd: (lambda r: _solve(kk, r, mesh.node_xyz, dd)))(k, d), du

        for li in range(int(layer_of.max()) + 1):
            new = layer_of == li
            if not new.any():
                continue
            active |= new
            node_on[mesh.enode[new].ravel()] = True
            el = np.flatnonzero(new)
            d, solve, du = step(e * active, kern.load(e, self.eps[el], el))
            u += du
            plane = np.flatnonzero(node_on & (self.hn == self.hgt[el].max() + 1))
            ldof = 3 * plane + self.axis
            lifts.append((len(steps), ldof, self.sign * u[ldof]))
            steps.append((d, solve, el, du, np.flatnonzero(active), self.eps) if keep else None)
        if self.cool is not None:
            # FDM: the finished part cools on the bed to room temperature
            el = np.arange(ne)
            d, solve, du = step(e, kern.load(e, self.cool, el))
            u += du
            steps.append((d, solve, el, du, el, self.cool) if keep else None)
        return u, steps, lifts

    def measures(self, e, de=None, weights=None):
        """The as-built measures of moduli e: "mean" (D), "max" (the largest
        |u| at the end of the build, mm; along the build only for FDM) and
        "lift" (the largest rise along the build at the build plane, over
        the layers as each goes down: the recoater's clearance, mm). With
        de, also the gradient with respect to the design density of
        sum(w[m] * measure m), w = weights(vals) (a callable, so weights can
        depend on this design's values). The maxima are steered through
        16-norms rescaled to the true maximum at this design (adaptive
        normalization), so the gradient is that of the maximum to first
        order. FDM builds at lumps thicker than the printed layer are
        extrapolated to it from a second build at twice the lumps (see
        layer_extrapolation)."""
        mesh, kern = self.mesh, self.kern
        ne = mesh.n_elem
        keep = de is not None
        u1, steps, lifts = self._forward(e, self.layer_of, keep)
        kx = self.extrapolate
        if kx:
            u2, steps2, _ = self._forward(e, self.layer_of2, keep)
            u = (1.0 + kx) * u1 - kx * u2
        else:
            u = u1
        self.last_u = u
        uu = u.reshape(-1, 3) * self.mask
        un = np.linalg.norm(uu, axis=1)
        allrise = np.concatenate([r for _, _, r in lifts]) if lifts else np.zeros(1)
        vals = {"mean": float((self.wdof * u * u).sum()), "max": float(un.max()),
                "lift": float(max(allrise.max(), 0.0))}
        # the 16-norms the gradient steers (checks)
        vals["_maxP"] = float(np.sum(un ** 16) ** (1 / 16))
        vals["_liftP"] = float(np.sum(np.maximum(allrise, 0.0) ** 16) ** (1 / 16))
        if de is None:
            return vals, None
        w = weights(vals) if callable(weights) else (weights or {"mean": 1.0})
        q = 16.0
        # dJ/du at the end of the build, and per step (the lift)
        gu = 2.0 * w.get("mean", 0.0) * self.wdof * u
        if w.get("max") and vals["max"] > 0:
            r = un / vals["max"]
            c = r ** (q - 1) / (r ** q).sum()
            gu += w["max"] * ((c / np.maximum(un, 1e-30))[:, None] * uu).ravel()
        gstep = {}
        if w.get("lift") and vals["lift"] > 0:
            tot = float(np.sum((np.maximum(allrise, 0.0) / vals["lift"]) ** q))
            for si, ldof, rs in lifts:
                g = np.zeros(mesh.n_dof)
                g[ldof] = w["lift"] * self.sign * (np.maximum(rs, 0.0) / vals["lift"]) ** (q - 1) / tot
                gstep[si] = g
        # adjoint: J depends on the final u = sum of the layer increments
        # and on each step's u_k (the lift), so dJ/d du_j = gu + the sum of
        # the step terms from j on; K_j lam_j = that, and
        # dJ/de = sum_j lam_j^T (df_j/de - dK_j/de du_j). Extrapolated, u is
        # (1 + k) u1 - k u2, so each build's adjoint takes its share.
        dd = np.zeros(ne)
        ke, g = kern.st.ke, kern.g

        def adjoint(steps, scale, with_lift):
            acc = np.zeros(mesh.n_dof)
            for si in range(len(steps) - 1, -1, -1):
                d, solve, el, du, ia, eps = steps[si]
                if with_lift and si in gstep:
                    acc = acc + gstep[si]
                rhs = scale * gu + acc
                lam = np.zeros(mesh.n_dof)
                lam[d] = solve(rhs[d])
                dd[el] += np.einsum("ij,ij->i", lam[mesh.edof[el]], eps[el] @ g.T)
                dd[ia] -= np.einsum("ij,jk,ik->i", lam[mesh.edof[ia]], ke, du[mesh.edof[ia]])

        adjoint(steps, 1.0 + kx, True)
        if kx:
            adjoint(steps2, -kx, False)
        return vals, dd * de


class CoarseWarp:
    """A WarpTerm on a grid `factor` times coarser than the optimizer's:
    each coarse cell's modulus is the mean of its fine cells' (absent ones
    count as void), and the gradient goes back to each fine cell as its
    share. factor 2 has 8x fewer cells and half the layers, so the build
    solves in the optimizer loop cost a small fraction of the fine ones."""

    def __init__(self, mesh: HexMesh, setup: BuildSetup, factor=2, build=(2, 1), layer=None, nu=None):
        f = int(factor)
        cijk = mesh.elem_ijk // f
        uniq, self.parent = np.unique(cijk, axis=0, return_inverse=True)
        self.parent = self.parent.ravel()
        self.n = len(uniq)
        self.share = 1.0 / f ** 3
        self.mesh = HexMesh(uniq, mesh.pitch * f, mesh.origin)
        self.fine_mesh = mesh
        self._fine = None
        self.ratio = {"mean": 1.0, "max": 1.0, "lift": 1.0}
        self.layer = layer or (f if setup.process.lower() == "fdm" else 4) * mesh.pitch
        self.setup, self.build = setup, build
        self.term = WarpTerm(self.mesh, setup, build=build, layer=self.layer, nu=nu)
        self.factor = f

    def _coarse(self, e):
        return np.bincount(self.parent, weights=e, minlength=self.n) * self.share

    def fine(self):
        """The same build model on the full grid."""
        if self._fine is None:
            self._fine = WarpTerm(self.fine_mesh, self.setup, build=self.build,
                                  layer=self.layer,
                                  nu=self.term.kern.nu)
        return self._fine

    def measures(self, e, de=None, weights=None, refresh=False):
        """As WarpTerm.measures, on the coarse grid, with each measure scaled
        by its full-grid / coarse-grid ratio (refresh: re-measure that ratio
        on this design, one forward build on the full grid). Averaging the
        moduli makes thin members softer and stiffer by turns than they are,
        so the coarse grid alone misreads a design far from where the ratio
        was taken."""
        ec = np.maximum(self._coarse(e), 1e-9)
        if refresh:
            vf, _ = self.fine().measures(e)
            vc, _ = self.term.measures(ec)
            self.ratio = {k: (vf[k] / vc[k] if vc[k] > 1e-12 else 1.0) for k in ("mean", "max", "lift")}
        r = self.ratio

        def scaled(v):
            return {**v, **{k: v[k] * r[k] for k in r}}

        if de is None:
            v, _ = self.term.measures(ec)
            return scaled(v), None

        def wc(v):
            w = weights(scaled(v)) if callable(weights) else (weights or {"mean": 1.0})
            return {k: w[k] * r.get(k, 1.0) for k in w}

        vals, gc = self.term.measures(ec, np.ones(self.n), wc)
        return scaled(vals), gc[self.parent] * self.share * de

    def evaluate(self, e, de=None):
        vals, g = self.measures(e, de, lambda v: {"mean": 1.0})
        return vals["mean"], g
