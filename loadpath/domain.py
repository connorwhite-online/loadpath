"""
Turn a study's shapes into the voxel problem the solver works on: which
cells exist, which are pinned solid (preserve), and which nodes carry the
supports and loads.

Classification is by supersampling each cell at 8 interior points, so a
cell's membership is its covered fraction rather than whether its center
happens to fall inside:

- preserve wins over design space, and is pinned to full density;
- a keep-out removes any design cell it touches at all (the final part must
  clear it), but only removes a preserve cell when it covers most of it, so
  a thin bearing boss around a bore keeps its wall in the analysis;
- islands of material that cannot reach a support are dropped: they carry
  no load and would make the stiffness matrix singular.

The exact keep-out and preserve geometry is restored after optimization by
mesh booleans, so the voxel classification only has to be right about load
paths, not about surfaces.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import os

import numpy as np
from scipy import ndimage

from .fem import HexMesh
from .shapes import Shape
from .spec import Study

__all__ = ["Domain", "build_domain", "choose_pitch"]

# Sub-cell samples sit a hair inside the quarter points: a face that lies on a
# quarter plane (a part some odd half cells long) is then counted the same way
# at both ends, not by rounding, and a mirror-symmetric part stays one.
_Q = 0.25 * (1.0 - 1e-6)
_SUB = np.array([[sx, sy, sz] for sx in (-_Q, _Q) for sy in (-_Q, _Q) for sz in (-_Q, _Q)])
_CONN26 = np.ones((3, 3, 3), bool)


@dataclass
class Domain:
    pitch: float
    origin: np.ndarray            # min corner of cell (0, 0, 0), mm
    dims: tuple                   # cells per axis
    mesh: HexMesh
    preserve: np.ndarray          # bool per active element: pinned solid
    fixed: np.ndarray             # bool per dof
    loads: list                   # per load case: force vector per dof (N)
    case_names: list
    case_weights: list
    support_nodes: np.ndarray     # bool per node
    load_nodes: np.ndarray        # bool per node (any case)
    mirror: dict = field(default_factory=dict)   # axis -> element index of the mirror cell (-1 none)
    # per element, or None: how full a cell pinned by a loaded surface is
    # (0-1, 0.5 on the pinned strap's back; nan where nothing says), for the
    # mesh's sub-cell surface (simp.boundary_field)
    level_hint: np.ndarray | None = None
    warnings: list = field(default_factory=list)
    # Pin supports as penalty springs at unit modulus (scale by E), or None.
    springs: object = None
    # supports that only push (BoundaryConditions.seats, pushseat.py)
    seats: list = field(default_factory=list)
    # Self-weight: N per mm^3 of solid, or None (see BoundaryConditions).
    gravity: np.ndarray | None = None
    # per case, directions.Spread or None (see BoundaryConditions)
    spreads: list = field(default_factory=list)
    # bearing loads through a rigid ring (insert.Insert; BoundaryConditions.inserts)
    inserts: list = field(default_factory=list)
    # The study's thermal block on this mesh (thermal.StudyHeat), or None.
    thermal: object = None
    # buildSim warp design (WarpSpec: warpWeight and/or limits), or None.
    warp: object = None
    # A thin skin's cells as cut cells (switch._SkinCells), or None.
    skin: object = None
    # indices of the stiffness-only inertial cases (vibration masses shaken
    # along x, y, z; see _inertial_cases): in the compliance objective, not
    # in stress, deflection or buckling limits
    inertial_cases: list = field(default_factory=list)
    # massProperties: each cell's cost in the volume budget, mean 1 over the
    # design cells ((1 - w) + w r^2 / mean r^2 about the axis), or None
    volume_weights: np.ndarray | None = None
    # aero.rotors: the drag of the design in each rotor's stream as a cost in
    # the material budget (aero.BlockageCost), or None
    blockage: object = None
    # castfeed.FeedTerm: the pour as the optimizer sees it (casting.feed), or None
    feeding: object = None
    # the vibration block as optimizer constraints (modalopt.ModalModel), or None
    modal: object = None
    # impact load cases: [(case index, spec.LoadCase)], and the material's modulus and density for them
    impacts: list = field(default_factory=list)
    impact_material: tuple = (1.0, 1.0)
    impact_study: object = None
    # massProperties.centerOfMass: (target (3,) with nan on a free axis, tolerance mm), or None
    center_of_mass: tuple | None = None
    # inertiaRelief: the cases' loads as written (BoundaryConditions.raw_loads)
    # so the optimizer rebalances them against the design's own mass
    raw_loads: list | None = None
    # inertiaRelief.masses per node (carried_volume), or None
    carried: np.ndarray | None = None
    # inertiaRelief.masses with a centre: rigid bodies (carried_bodies)
    bodies: list = field(default_factory=list)
    # design cells that join drawn cells through faces where the drawn cells
    # meet only at an edge or corner (drawn_joints): never pruned
    joints: np.ndarray | None = None
    # thin pinned walls as cut cells (thinwall.py): the cells they cross
    # (their stiffness is in `springs`), and those of them the optimizer may
    # not fill (a keep-out touches them, or they are outside the design space)
    walls: np.ndarray | None = None
    wall_void: np.ndarray | None = None
    # the pins' own springs, the part of `springs` that holds the part (the
    # rest is a thin wall's or a bought member's stiffness), or None
    pin_springs: object = None

    @property
    def n_elem(self):
        return self.mesh.n_elem

    def full_grid(self, values: np.ndarray, fill=0.0) -> np.ndarray:
        g = np.full(self.dims, fill, float)
        ijk = self.mesh.elem_ijk
        g[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values
        return g


def _fraction(shapes: list[Shape], centers: np.ndarray, pitch: float) -> np.ndarray:
    if not shapes:
        return np.zeros(len(centers))
    hit = np.zeros((len(centers), len(_SUB)), bool)
    for k, off in enumerate(_SUB):
        pts = centers + off * pitch
        for s in shapes:
            hit[:, k] |= s.contains(pts)
    return hit.mean(axis=1)


def cell_cap(study: Study) -> int:
    """The most cells a study's grid may have: maxElements, or up to 4x it
    for the constrained method, whose members need 2.5 cells across (see
    choose_pitch)."""
    if study.goal.method == "constrained" and study.min_member:
        return 4 * int(study.max_elements)
    return int(study.max_elements)


def choose_pitch(study: Study) -> float:
    """Cell size: explicit, else the size that puts roughly maxElements cells
    in the design space (estimated from a coarse occupancy sample)."""
    if study.pitch:
        return float(study.pitch)
    lo, hi = study.bounds()
    vol = float(np.prod(hi - lo))
    # Coarse 24^3 sample of what fraction of the box is actually design space.
    n = 24
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(pts), bool)
    for s in study.design_space + [r.shape for r in study.preserve]:
        inside |= s.contains(pts)
    frac = max(float(inside.mean()), 0.05)
    pitch = (vol * frac / study.max_elements) ** (1.0 / 3.0)
    budget = pitch
    if study.min_member:
        # No finer than two cells per filter radius (1.3 x minMember, see
        # simp.filter_radius): the filter sets the feature scale, so finer
        # cells only cost time. On the arm link this cut the cells from 35k to
        # 18k and the study from ~255s to ~155s, and the design came out
        # stiffer (0.58mm vs 1.06mm deflection at the same 30% volume): the
        # finer grid had settled in a worse local optimum.
        pitch = max(pitch, 1.3 * float(study.min_member) / 2.0)
        if study.goal.method == "constrained":
            # The stress-constrained method sits its members right at the
            # minimum, and at two cells per filter radius that is 1.5 cells:
            # the wall bracket's 4 mm members on a 2.6 mm grid meshed three
            # times floppier than their voxels, its first run verified at SF
            # 0.1 with the optimizer reading the limit, and four targets
            # 2.45-2.6 ended at 55-75 g. With the member 2.5 cells across
            # (1.6 mm) the optimizer's reading holds on the mesh (first run
            # SF 1.7) and the four land at 29.2-30.1 g, in about 4x the time.
            # Up to 4x maxElements cells for it, no more.
            pitch = min(pitch, max(float(study.min_member) / 2.5, budget / 4.0 ** (1.0 / 3.0)))
    return pitch


def wall_pitch(study: Study) -> float | None:
    """The cell a pinned wall asks of a plain run, or None when the study's
    own grid already has it (or sets its pitch itself).

    A wall the study pins (a skin, a duct's or plenum's wall) carries by
    bending. On whole cells it needs two cells across, and still rounds by
    up to half a cell (the check warning below); as cut cells
    (thinwall.py, where it applies) one cell across bends like the wall,
    so the grid is refined only where the wall is thinner than a cell
    (eight times fewer cells than two through it). The wall the study drew sets the cell, as
    the member does for the constrained method, up to the same 4x
    maxElements cells. Every packaged study with such a wall set its pitch
    by hand to get this. For `loadpath run` and `check` only: the switch,
    feel and cooling solves model a thin skin as cut cells on their own
    grids (switch._thin_skin)."""
    if study.pitch or not study.shells:
        return None
    from . import thinwall
    cut = thinwall.applies(study)
    d = build_domain(study, cut_walls=cut)
    thin = min(float(t) for _, t in study.shells)
    # (cells counted on the study's own grid, not estimated: the estimate
    # runs a few percent over and left a 1.6 mm skin at 1.95 cells)
    most = d.pitch * (d.n_elem / (4.0 * study.max_elements)) ** (1.0 / 3.0)
    # a wall as cut cells on the grid (thinwall.py) needs only a cell through it
    want = max(thin / (thinwall.MIN_CELLS if cut else 2.0), most)
    return want if want < d.pitch - 1e-9 else None


DRAWN_BEARS = os.environ.get("LOADPATH_DRAWN_BEARS", "1") != "0"


def _sdf_normal(shape, pts, h):
    """Unit gradient of a shape's signed distance at `pts` (central differences)."""
    g = np.zeros_like(pts)
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        g[:, a] = shape.sdf(pts + e) - shape.sdf(pts - e)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-12)


def _min_shape_sdf(shapes, pts, mask):
    out = np.full(len(pts), np.inf)
    idx = np.flatnonzero(mask)
    for sh in shapes:
        out[idx] = np.minimum(out[idx], sh.sdf(pts[idx]))
    return out


def _region_nodes(study: Study, mesh: HexMesh, region: str | None, shape: Shape | None, label: str,
                  tol_floor: float = 0.0):
    if shape is None:
        shape, kind = study.region_shape(region)
    else:
        kind = "inline"
    # A keep-out is a hole: the nodes that load or hold it are the ones
    # lining it, up to one cell away. Solid regions take nodes inside or on
    # their surface.
    # `tol_floor`: a verification grid finer than the optimization's still
    # reaches as far as the optimization did. On the printed hook the design
    # grid's support took the node plane 0.75 mm off the wall, so the part
    # was (correctly) designed to start there, and a 0.6 mm verification
    # grid then found nothing at the wall to hold.
    # Only as a fallback: widening it everywhere clamped more of every bolt
    # ring in verification and read the robot-arm upper arm at SF 10 instead of 5.
    tol = mesh.pitch * (1.0 if kind == "keepOut" else 0.5)
    d = shape.sdf(mesh.node_xyz)
    nodes = d <= tol
    if kind != "keepOut" and (d <= 1e-9).sum() >= 4:
        # A solid region with nodes inside it holds or loads those, not a
        # plane half a cell outside it too. The extra plane let the printed
        # hook's optimizer hang the part off nodes 0.75 mm short of the wall:
        # it was designed not to touch the wall it is screwed to.
        nodes = d <= 1e-9
    if kind == "keepOut" and nodes.any() and DRAWN_BEARS and study.preserve:
        # A pin or bolt bears on the material it passes through. Where the
        # study draws that material (a boss round the bore), the drawn
        # shape is what carries it, and the keep-out running on past the
        # boss is clearance for the pin, not a surface it presses: loaded
        # there too, the optimizer had to line that clearance a member thick
        # and the part grew a collar over the boss's end face (a seat).
        on = nodes & (_min_shape_sdf([r.shape for r in study.preserve], mesh.node_xyz, nodes) <= 0.5 * tol)
        if on.sum() >= 4:
            nodes = on
    if not nodes.any() and tol_floor > tol:
        nodes = d <= tol_floor
    if not nodes.any():
        name = region or "inline shape"
        raise ValueError(f"{label} on {name!r} touches no material: it lies outside the design space, "
                         "or every cell around it was removed. Move it onto the part.")
    return nodes


# Pin springs, relative to a cell's own stiffness: stiff enough to hold the
# pin to well under a percent of the part's deflection, soft enough not to
# wreck the solver's conditioning.
_PIN_STIFFNESS = 100.0
# Supports and loads touching at most this share of the cells are pinned
# solid (see build_domain).
_ANCHOR_MAX_FRACTION = 0.03


@dataclass
class BoundaryConditions:
    fixed: np.ndarray
    springs: object                # scipy CSR at unit modulus, or None
    support_nodes: np.ndarray
    loads: list                    # per case, force per dof (N)
    case_load_nodes: list          # per case, bool per node
    load_nodes: np.ndarray
    gravity: np.ndarray | None     # N per mm^3 of solid material
    regions: list = field(default_factory=list)   # node mask of every support and load region
    # per case, bool per node: the walls a pressure acts on (pressure.py).
    # Not contact: a pressure is a real distributed load, and the peak stress
    # of a pressure vessel is on its wetted wall.
    case_pressure_nodes: list = field(default_factory=list)
    # per case, |net force| / sum |nodal force| of its pressures (0 when sealed)
    pressure_imbalance: list = field(default_factory=list)
    # per case, directions.Spread for a force of unknown direction, or None;
    # `loads` then holds the force as written
    spreads: list = field(default_factory=list)
    # inertiaRelief: the loads as written (before the part's own inertia
    # balances them) and the rigid-body modes, for rebalancing on a design
    raw_loads: list | None = None
    # whether `springs` holds the part anywhere (a pin support); a bought member's stiffness is in
    # `springs` too (stock.py) and holds nothing
    pinned: bool = False
    # seats (supports[].push, seats.py): (node indices, the unit push, each
    # node's share of the face). Their nodes are free in `fixed`; a solve
    # that honours them adds their forces (seats.solve)
    seats: list = field(default_factory=list)
    # the pins' own springs (the part of `springs` that holds it), or None
    pin_springs: object = None
    # bearing loads as a rigid ring in the bore (insert.py): a check that
    # solves them replaces the fixed pressure in `loads` with the ring's
    inserts: list = field(default_factory=list)
    # per case, bool per node: the bores a ring bears on (inserts) and no
    # other load of the case touches. The ring's contact is solved, so the
    # stress at the bore is the part's and is read (a pin-loaded hole's net
    # section and bearing face are where it breaks), not left out as the
    # band round an idealized load is.
    case_ring_nodes: list = field(default_factory=list)


def node_mass(mesh: HexMesh, density) -> np.ndarray:
    """Lumped volume per node (mm^3) of cells at `density` (0-1 per cell or
    a scalar): what inertia relief weighs each node's acceleration by."""
    m = np.zeros(mesh.n_nodes)
    dens = np.broadcast_to(np.asarray(density, float), (mesh.n_elem,))
    extra = getattr(mesh, "stock_density", None)
    if extra is not None:
        # bought members in the mesh (stock.py): their own density, less the printed material's they displace
        dens = dens + extra
    np.add.at(m, mesh.enode, (dens * mesh.pitch ** 3 / 8.0)[:, None])
    return m


def inertia_balance(f: np.ndarray, m_node: np.ndarray, phi: np.ndarray, bodies: list | None = None):
    """Inertia relief: the load less the d'Alembert force of the rigid
    acceleration it gives the free part, f - M Phi a with a = (Phi^T M
    Phi)^-1 Phi^T f, so it has no net force or moment and needs no
    reaction (Nastran's SUPORT / nTop's static analysis with inertia
    relief). `bodies` (carried_bodies): rigid bodies carried on the part,
    each adding its 6x6 mass to Phi^T M Phi and its inertia wrench -Mc a
    through its attachment nodes. Returns (balanced load, Phi a per dof)."""
    m = np.repeat(m_node, 3)
    mphi = phi * m[:, None]
    mrb = phi.T @ mphi
    for _, _, mc in bodies or ():
        mrb = mrb + mc
    a = np.linalg.solve(mrb, phi.T @ f)
    acc = phi @ a
    fb = f - m * acc
    for dofs, dmat, mc in bodies or ():
        fb[dofs] -= dmat @ (mc @ a)
    return fb, acc


def rigid_free(u: np.ndarray, m_node: np.ndarray, phi: np.ndarray, bodies: list | None = None):
    """u less the rigid motion that makes it orthogonal to every inertia
    force a rigid acceleration raises ((M Phi + sum D Mc)^T u_r = 0), so the
    change in the acceleration drops out of the relieved load's
    sensitivity."""
    m = np.repeat(m_node, 3)
    mphi = phi * m[:, None]
    lhs = phi.T @ mphi
    rhs = mphi.T @ u
    for dofs, dmat, mc in bodies or ():
        lhs = lhs + mc
        rhs = rhs + mc.T @ (dmat.T @ u[dofs])
    return u - phi @ np.linalg.solve(lhs, rhs)


def _rbm(xyz):
    from .fem import rigid_body_modes

    return rigid_body_modes(xyz)


def _skew(d):
    return np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])


def carried_bodies(study: Study, mesh: HexMesh, phi: np.ndarray, tol_floor: float = 0.0) -> list:
    """inertiaRelief.masses given with their centre (`at`): each a rigid
    body on its region's nodes, [(dofs, D, Mc)]. Mc is its 6x6 mass in the
    rigid-mode coordinates of `phi` (fem.rigid_body_modes: translations,
    then rotations about z, x, y through the nodes' centroid), in mm^3 of
    the study's material so it adds to node_mass; D (3k x 6) spreads a
    wrench over its nodes as a rigid patch would (Phi_r (Phi_r^T Phi_r)^+).
    A battery strapped 30 mm above its posts pulls on them with the moment
    of its inertia, which mass spread on the posts alone leaves out."""
    masses = getattr(study, "inertia_bodies", None) or []
    if not masses:
        return []
    rho = float(study.material.density) * 1e-6
    c0 = mesh.node_xyz.mean(axis=0)
    # a -> (t, w): w = (a4, a5, a3)
    g = np.zeros((6, 6))
    g[0, 0] = g[1, 1] = g[2, 2] = 1.0
    g[3, 4] = g[4, 5] = g[5, 3] = 1.0
    out = []
    for i, (region, shape, kg, at, inertia) in enumerate(masses):
        nodes = np.flatnonzero(_region_nodes(study, mesh, region, shape, f"inertiaRelief.masses[{i}]",
                                             tol_floor))
        dofs = (3 * nodes[:, None] + np.arange(3)[None, :]).ravel()
        pr = phi[dofs]
        dmat = pr @ np.linalg.pinv(pr.T @ pr)
        mv = kg / rho
        d = np.asarray(at, float) - c0
        dt = _skew(d)
        j = np.diag(np.asarray(inertia, float)) if np.ndim(inertia) == 1 else np.asarray(inertia, float)
        mtw = np.block([[mv * np.eye(3), -mv * dt], [mv * dt, -mv * dt @ dt + j / rho]])
        out.append((dofs, dmat, g.T @ mtw @ g))
    return out


def carried_volume(study: Study, mesh: HexMesh, tol_floor: float = 0.0) -> np.ndarray | None:
    """inertiaRelief.masses on a mesh's nodes, as the volume of the study's
    material that weighs the same (mm^3 per node, node_mass's unit), each
    spread evenly over its region's nodes; None when it carries nothing.
    A drone's motors and battery are most of its mass: balanced by the
    frame's own inertia alone, a thrust case pulls on the frame's far
    corners instead of on the battery in its middle."""
    masses = getattr(study, "inertia_masses", None) or []
    if not masses:
        return None
    rho = float(study.material.density) * 1e-6           # kg per mm^3
    out = np.zeros(mesh.n_nodes)
    for i, (region, shape, kg) in enumerate(masses):
        nodes = np.flatnonzero(_region_nodes(study, mesh, region, shape, f"inertiaRelief.masses[{i}]",
                                             tol_floor))
        out[nodes] += kg / rho / len(nodes)
    return out


def _inertia_relief(study: Study, mesh: HexMesh, loads: list, fixed: np.ndarray, tol_floor: float = 0.0):
    """Every case balanced by the part's own inertia, the mesh taken as
    solid, plus what it carries (inertiaRelief.masses); the 3-2-1 support
    then takes nothing."""
    from .fem import rigid_body_modes

    phi = rigid_body_modes(mesh.node_xyz)
    m = node_mass(mesh, 1.0)
    carried = carried_volume(study, mesh, tol_floor)
    if carried is not None:
        m = m + carried
    bodies = carried_bodies(study, mesh, phi, tol_floor)
    out = []
    for f in loads:
        fb, _ = inertia_balance(f, m, phi, bodies)
        fb[fixed] = 0.0
        out.append(fb)
    return out


def impact_energy(study: Study, case, f_raw: np.ndarray, acc: np.ndarray | None):
    """An impact case's (energy N mm, mass kg it is the kinetic energy of or
    None). With a velocity and no mass on a free part, the mass is what the
    body presents at the patch along the hit: 1 / m = f^ . Phi (Phi^T M
    Phi)^-1 Phi^T f^, the acceleration a unit force gives the patch (`acc`:
    inertia_balance's Phi a for `f_raw`, masses in mm^3 of the material). A
    hit through the centre of mass stops the whole mass; a corner hit spins
    the body and stops less."""
    ic = case.impact
    if ic.energy is not None:
        return float(ic.energy), None
    mass = ic.mass
    if mass is None:
        if acc is None:
            raise ValueError(f"load case {case.name!r}: impact.mass is needed (no inertiaRelief to take it from)")
        fn2 = float(np.sum(f_raw.reshape(-1, 3).sum(axis=0) ** 2))
        mass = float(study.material.density) * 1e-6 * fn2 / max(float(f_raw @ acc), 1e-300)
    return 1000.0 * 0.5 * mass * ic.velocity ** 2, float(mass)


def patch_acceleration(study: Study, mesh: HexMesh, f_raw: np.ndarray, density=1.0, tol_floor: float = 0.0):
    """Phi a for `f_raw` on a free part (see impact_energy): the mesh at
    `density` plus what it carries."""
    from .fem import rigid_body_modes

    phi = rigid_body_modes(mesh.node_xyz)
    m = node_mass(mesh, density)
    carried = carried_volume(study, mesh, tol_floor)
    if carried is not None:
        m = m + carried
    return inertia_balance(f_raw, m, phi, carried_bodies(study, mesh, phi, tol_floor))[1]


# a bore's seat in sectors round its axis: each sector pushes along its middle's
# outward normal, within cos(15 deg) = 0.97 of the wall's own at its edges
RADIAL_SECTORS = 12


def radial_seats(pts, p0, ax, sectors: int = RADIAL_SECTORS) -> list:
    """A tube, shaft or pin in a bore (supports[].push = "radial") as
    seats: [(indices into pts, outward unit normal)], one per sector of
    the bore wall round the axis through p0. What is in the bore pushes out
    on the wall where it bears and nowhere pulls; held "fix" the wall is
    glued to it all round, and a part that wraps the tube less than half
    way round passes on that glue (held-out batch 3: a tube tee's rail
    socket came out a saddle under the rail)."""
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    d = np.asarray(pts, float) - np.asarray(p0, float)
    rad = d - (d @ ax)[:, None] * ax
    a = np.cross(ax, [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    th = np.arctan2(rad @ b, rad @ a)
    k = np.floor((th + np.pi) / (2 * np.pi / sectors)).astype(int) % sectors
    out = []
    for j in range(sectors):
        sub = np.nonzero(k == j)[0]
        if len(sub) == 0:
            continue
        mid = -np.pi + (j + 0.5) * 2 * np.pi / sectors
        out.append((sub, np.cos(mid) * a + np.sin(mid) * b))
    return out


# a socket's seats (push "normal"): wall nodes grouped by their outward
# normal, within 15 degrees of their group's mean (as a bore's sectors are)
WALL_SEAT_COS = float(np.cos(np.radians(15.0)))


def wall_normals(shape, pts, h: float = 0.02) -> np.ndarray:
    """Unit outward normals of `shape`'s wall at points on or near it: the
    gradient of its signed distance, by central differences."""
    pts = np.asarray(pts, float)
    g = np.empty_like(pts)
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        g[:, a] = (np.asarray(shape.sdf(pts + e)) - np.asarray(shape.sdf(pts - e))) / (2 * h)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-12)


def wall_seats(pts, shape, cos_min: float = WALL_SEAT_COS) -> list:
    """Whatever fills a keep-out (a nut in a wrench's hex, a D-shaft in a
    hub, a square leg in a socket, a shaft in a bore) as seats
    (supports[].push = "normal"): [(indices into pts, outward unit
    normal)], the wall's nodes grouped by the direction their wall faces.
    The body inside pushes out on the wall where it bears and nowhere
    pulls; a torque on a hex is carried by its flats bearing, as a
    wrench's is. Held "fix", the wall is glued to it all round, and a part
    that grips a nut on two flats passes on that glue (held-out batch 4:
    a box-end wrench came out a fork on one side of its nut)."""
    nrm = wall_normals(shape, pts)
    centres = []
    for v in nrm:
        if not centres or max(float(c @ v) for c in centres) < cos_min:
            centres.append(v.copy())
    c = np.asarray(centres)
    for _ in range(2):                               # settle: assign to the nearest, re-centre
        k = np.argmax(nrm @ c.T, axis=1)
        c = np.asarray([nrm[k == j].sum(axis=0) if np.any(k == j) else c[j] for j in range(len(c))])
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    k = np.argmax(nrm @ c.T, axis=1)
    return [(np.nonzero(k == j)[0], c[j]) for j in range(len(c)) if np.any(k == j)]


def seat_groups(push, pts, shape, label="supports"):
    """A string push's seats: "radial" round a cylinder's axis, "normal"
    along any keep-out wall's normal."""
    if push == "normal":
        return wall_seats(pts, shape)
    if not hasattr(shape, "axis"):
        raise ValueError(f"{label} needs a cylinder region (a bore keep-out or a cylinder) for its axis")
    return radial_seats(pts, shape.p0, shape.axis)


def _cylinder_of(study: Study, region, shape, label):
    sh = shape if shape is not None else study.region_shape(region)[0]
    if not hasattr(sh, "axis"):
        raise ValueError(f"{label} needs a cylinder region (a bore keep-out or a cylinder) for its axis")
    return sh.p0, sh.axis


def _wall_normals(shape: Shape, pts: np.ndarray) -> np.ndarray:
    """Unit normals of the keep-out's surface at `pts` (nodes lining it, on
    the material side): the gradient of its distance, pointing out of the
    hole into the wall."""
    pts = np.asarray(pts, float)
    lo, hi = (np.asarray(b, float) for b in shape.bounds())
    h = 1e-4 * max(float(np.max(hi - lo)), 1.0)
    g = np.empty_like(pts)
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        g[:, a] = shape.sdf(pts + e) - shape.sdf(pts - e)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def bearing_forces(shape: Shape, pts: np.ndarray, share: np.ndarray, force: np.ndarray) -> np.ndarray | None:
    """Nodal forces (n, 3) of a body seated in the bore `shape` (a bearing's
    outer ring, a shaft, a pin) pushed with `force`, or None when no wall
    faces the radial part of it.

    A seated body reaches the wall through contact, and contact carries
    pressure square to the surface it touches (friction aside), so each
    wall takes p n, never a share of the force along the force's own
    direction. Radially that is the cosine law: p = p0 max(n . d, 0) on the
    bore's half facing the push. A bore's end face (a bearing's shoulder,
    a blind bore's floor) faces along the axis, so it takes none of the
    radial load: the 608 pillow block's seat put 31% of its side load on
    the shoulder face, sideways, beside the shaft's clearance hole, and
    the solid block peaked there. The axial part bears on the faces that
    face it (the shoulder a bearing is pushed into); with none (a pin
    through an open bore) it is the friction along the bore, spread on it
    as before."""
    pts = np.asarray(pts, float)
    share = np.asarray(share, float)
    force = np.asarray(force, float)
    ax = getattr(shape, "axis", None)
    out = np.zeros((len(pts), 3))
    if ax is None or not len(pts):
        out += force[None] * share[:, None]
        return out
    m = _wall_normals(shape, pts)
    axial = (force @ ax) * ax
    radial = force - axial
    size = float(np.linalg.norm(radial))
    if size > 0:
        d = radial / size
        w = np.maximum(m @ d, 0.0) * share
        if w.sum() <= 0:
            return None
        t = w[:, None] * m
        t *= size / float((t @ d).sum())
        # what the nodes' sampling of the circle leaves over (a grid's bore
        # is not exactly symmetric about the force): spread as the pressure
        out += t + (w / w.sum())[:, None] * (radial - t.sum(axis=0))[None]
    size = float(np.linalg.norm(axial))
    if size > 0:
        d = axial / size
        # faces that face the push: within 45 degrees of the axis
        w = np.where(m @ d > np.sqrt(0.5), share, 0.0)
        if w.sum() > 0:
            out += (w / w.sum())[:, None] * axial[None]
        else:
            out += share[:, None] * axial[None]
    return out


def _wetted(study: Study, mesh: HexMesh) -> dict:
    """Per pressurized keep-out: (nodal force per unit pressure, node mask)."""
    from .pressure import pressure_force

    out = {}
    for case in study.load_cases:
        for ld in case.loads:
            if ld.pressure and ld.region not in out:
                out[ld.region] = pressure_force(mesh, study.region_shape(ld.region)[0], 1.0)
    return out


# A pressure-only study is held by a 3-2-1 support, which is right only for a
# balanced load: a sealed cavity's balances to rounding, an open duct's does
# not (the thrust on its missing end caps).
_IMBALANCE_MAX = 0.02


class _HeldPreload(ValueError):
    """The fasteners' preload case lands only on supported nodes."""


def boundary_conditions(study: Study, mesh: HexMesh, check_loads=True, tol_floor: float = 0.0) -> BoundaryConditions:
    """Supports, pins and loads on a mesh's nodes. Shared by the optimizer
    and the verification, so both analyze exactly the same problem."""
    xyz = mesh.node_xyz
    support_nodes = np.zeros(mesh.n_nodes, bool)
    fixed = np.zeros(mesh.n_dof, bool)
    spring_blocks = []
    regions = []
    seats = []
    wet = _wetted(study, mesh)
    for rid, (_, nodes) in wet.items():
        if not nodes.any():
            raise ValueError(f"pressure in {rid!r} pushes on no material: the keep-out is not lined by the part")
    if study.self_equilibrated:
        from .pressure import stabilize

        held = np.any([n for _, n in wet.values()], axis=0) if wet else np.zeros(mesh.n_nodes, bool)
        if study.inertia_relief:
            # free in space: three nodes of the part hold it still, on what
            # is drawn (preserve) when there is some. On the design space's
            # far corners, which the optimizer empties, the hold sat on void
            # cells and the solve could not converge (a drone frame at 30%)
            held = np.zeros(mesh.n_nodes, bool)
            for r in study.preserve:
                held |= r.shape.sdf(mesh.node_xyz) <= 1e-9
            if held.sum() < 3 or np.ptp(mesh.node_xyz[held], axis=0).max() < 0.25 * np.ptp(mesh.node_xyz,
                                                                                          axis=0).max():
                held = np.ones(mesh.n_nodes, bool)
        fixed, support_nodes = stabilize(mesh, held)
    for i, s in enumerate(study.supports):
        label = f"supports[{i}]"
        nodes = _region_nodes(study, mesh, s.region, s.shape, label, tol_floor)
        regions.append(nodes)
        support_nodes |= nodes
        idx = np.nonzero(nodes)[0]
        if s.push is not None:
            share = np.bincount(mesh.enode.ravel(), minlength=mesh.n_nodes)[idx].astype(float)
            if isinstance(s.push, str):
                sh = s.shape if s.shape is not None else study.region_shape(s.region)[0]
                for sub, n in seat_groups(s.push, xyz[idx], sh, label):
                    seats.append((idx[sub], n, share[sub] / share[sub].sum(), getattr(s, "friction", None),
                                  None))
                continue
            seats.append((idx, np.asarray(s.push, float), share / share.sum(), getattr(s, "friction", None),
                          getattr(s, "preload", None)))
            continue
        for a in range(3):
            if s.fix[a]:
                fixed[3 * idx + a] = True
        if s.pin:
            p0, ax = _cylinder_of(study, s.region, s.shape, label)
            d = xyz[idx] - p0
            rad = d - (d @ ax)[:, None] * ax
            rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
            blocks = rad[:, :, None] * rad[:, None, :] + np.outer(ax, ax)[None]
            spring_blocks.append((idx, blocks * _PIN_STIFFNESS * mesh.pitch))
    if study.supports and all(s.pin for s in study.supports) and not study.self_equilibrated:
        lines = [_cylinder_of(study, s.region, s.shape, f"supports[{i}]") for i, s in enumerate(study.supports)]
        p0, ax = lines[0]
        coaxial = all(abs(abs(float(a @ ax)) - 1) < 1e-3 and
                      np.linalg.norm((q - p0) - ((q - p0) @ ax) * ax) < mesh.pitch for q, a in lines[1:])
        if coaxial:
            # a brief-eval bell crank pinned at its pivot alone read SF 0.0004:
            # nothing holds it from turning, so the solve is singular
            raise ValueError("every support is a pin on one axis, so nothing stops the part turning about it and "
                             "the stiffness is singular (safety factors come out near zero). Hold what really "
                             "stops it turning: fix the link, arm or stop that reacts the load (\"fix\": \"all\" "
                             "on that bore), or fix the pivot itself with \"fix\": \"all\"")
    springs = None
    if spring_blocks:
        import scipy.sparse as sp
        rows, cols, vals = [], [], []
        for idx, blocks in spring_blocks:
            for i in range(3):
                for j in range(3):
                    rows.append(3 * idx + i)
                    cols.append(3 * idx + j)
                    vals.append(blocks[:, i, j])
        rows, cols, vals = (np.concatenate(v) for v in (rows, cols, vals))
        keep = ~(fixed[rows] | fixed[cols])
        springs = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(mesh.n_dof, mesh.n_dof))
    pinned = springs is not None
    pin_springs = springs
    if getattr(study, "stock", None):
        # bought members: their own stiffness on their cells, less the printed material's the grid puts
        # there, added to every solve the way the pins' springs are (stock.stock_operator)
        from .stock import stock_operator
        k_stock = stock_operator(study, mesh, fixed)
        if k_stock is not None:
            springs = k_stock if springs is None else (springs + k_stock).tocsr()

    load_nodes = np.zeros(mesh.n_nodes, bool)
    # A force "over a region" is a uniform traction on it (or a uniform body
    # force through it), so each node takes the share of the surface that is
    # its own: on a loaded face a corner node stands for a quarter of an
    # interior node's area and an edge node for half. Equal shares per node
    # put 4x the traction on the corners and 2x on the edges of every
    # loaded face. The share is the count of cells the node touches.
    tributary = np.bincount(mesh.enode.ravel(), minlength=mesh.n_nodes).astype(float)
    loads, case_nodes, case_wet, imbalance, spreads = [], [], [], [], []
    case_rings = []
    from .insert import applies as _insert_applies, plan as plan_insert, plan_spread

    inserts, insert_applies = [], _insert_applies(study.material)
    for ci, case in enumerate(study.load_cases):
        f = np.zeros(mesh.n_dof)
        spread = None
        cn = np.zeros(mesh.n_nodes, bool)
        cw = np.zeros(mesh.n_nodes, bool)
        ringed = np.zeros(mesh.n_nodes, bool)
        plain = np.zeros(mesh.n_nodes, bool)
        fp = np.zeros(mesh.n_dof)
        for li, ld in enumerate(case.loads):
            label = f"loadCases[{ci}].loads[{li}]"
            if ld.pressure:
                unit, nodes = wet[ld.region]
                fp += ld.pressure * unit
                cw |= nodes
                continue
            nodes = _region_nodes(study, mesh, ld.region, ld.shape, label, tol_floor)
            regions.append(nodes)
            idx = np.nonzero(nodes)[0]
            cn[idx] = True
            pts = xyz[idx]
            share = tributary[idx] / tributary[idx].sum()
            force = ld.force
            if ld.direction is not None:
                # the uncertain force, spread evenly, one vector per axis
                size = float(np.linalg.norm(force))
                basis = []
                for a in range(3):
                    b = np.zeros(mesh.n_dof)
                    b.reshape(-1, 3)[idx, a] = size * share
                    basis.append(b)
                cos_half = -1.0 if ld.direction == "any" else float(np.cos(np.radians(ld.cone)))
                spread = (basis, force / size, cos_half)
                if ld.distribution == "bearing" and insert_applies:
                    # through the ring, solved per direction (insert.RingedSpread)
                    _cylinder_of(study, ld.region, ld.shape, label)
                    sh = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
                    ring = plan_spread(ci, sh, pts, idx, share, size, force / size)
                    if ring is not None:
                        inserts.append(ring)
                force = np.zeros(3)
            if ld.distribution == "bearing" and np.any(force):
                # A shaft pushes on the half of the bore that faces the force,
                # as contact pressure square to the wall (bearing_forces).
                _cylinder_of(study, ld.region, ld.shape, label)
                sh = ld.shape if ld.shape is not None else study.region_shape(ld.region)[0]
                fb = bearing_forces(sh, pts, share, force)
                if fb is None:
                    raise ValueError(f"{label}: no material on the loaded side of the bore")
                f.reshape(-1, 3)[idx] += fb
                if insert_applies:
                    radial = force - (force @ sh.axis) * sh.axis
                    ring = plan_insert(ci, sh, pts, idx, share, force,
                                       bearing_forces(sh, pts, share, radial) if np.any(radial) else 0 * pts)
                    if ring is not None:
                        inserts.append(ring)
                    # read at the bore only when the ring carries all of it:
                    # a thrust bears on the shoulder as a fixed pressure, a
                    # moment as fixed tractions
                    whole = ring is not None and abs(force @ sh.axis) <= 1e-9 * np.linalg.norm(force) \
                        and not np.any(ld.moment)
                    (ringed if whole else plain)[idx] = True
                else:
                    plain[idx] = True
                force = np.zeros(3)
            else:
                plain[idx] = True
            if np.any(force):
                f.reshape(-1, 3)[idx] += force[None] * share[:, None]
            if np.any(ld.moment):
                # Tractions that carry the moment and nothing else: the rigid
                # rotation field w x r about the region's centroid, with w
                # from its polar inertia (each node at its share of the area).
                r = pts - share @ pts
                inertia = (share * (r * r).sum(axis=1)).sum() * np.eye(3) - (r * share[:, None]).T @ r
                omega = np.linalg.lstsq(inertia, ld.moment, rcond=None)[0]
                f.reshape(-1, 3)[idx] += share[:, None] * np.cross(omega, r)
        if cw.any():
            fv = fp.reshape(-1, 3)
            imb = float(np.linalg.norm(fv.sum(axis=0)) / max(np.abs(fv).sum(), 1e-300))
            if study.self_equilibrated and not study.inertia_relief and imb > _IMBALANCE_MAX and check_loads:
                raise ValueError(
                    f"load case {case.name!r}: the pressure does not balance on itself (net force "
                    f"{np.linalg.norm(fv.sum(axis=0)):.1f} N): the cavity is open, so its ends' thrust has "
                    "to go somewhere. Add supports where the part is held (a flange, a clamp).")
            imbalance.append(imb)
            f += fp
        else:
            imbalance.append(0.0)
        applied = bool(np.any(f))
        f[fixed] = 0.0
        if spread is not None:
            from .directions import Spread

            for b in spread[0]:
                b[fixed] = 0.0
            spread = Spread(f.copy(), spread[0], spread[1], spread[2])
            f = spread.load(spread.nominal)
        spreads.append(spread)
        clamp = case.name == "preload" and any(len(s_) > 4 and (s_[4] or 0) > 0 for s_ in seats)
        if check_loads and not np.any(f) and study.gravity is None and not study.inertia_relief and not clamp:
            if not applied and spread is None:
                raise ValueError(f"load case {case.name!r} adds up to nothing at pitch {mesh.pitch:.3g} mm: its "
                                 "forces cancel on the same grid nodes (a fastener's seat and face rings closer "
                                 "than a cell, or equal and opposite loads on one region). Lengthen the grip, "
                                 "use a finer pitch, or drop the case")
            if case.name == "preload" and study.joints:
                raise _HeldPreload(case.name)
            raise ValueError(f"load case {case.name!r} acts only on fixed nodes; "
                             "a load applied where the part is held does nothing: load a region the supports "
                             "do not cover")
        scale = float(getattr(study, "impact_scale", {}).get(case.name, 1.0))
        if scale != 1.0:
            # an impact's peak scales the whole case: its rings' loads with it
            if spread is not None:
                spread = Spread(spread.base * scale, [b * scale for b in spread.basis], spread.nominal, spread.cos_half)
                spreads[-1] = spread
            for ring in inserts:
                if ring.case == ci:
                    ring.force = ring.force * scale
                    ring.fixed = ring.fixed * scale
                    if ring.thrust is not None:
                        ring.thrust = tuple(t * scale for t in ring.thrust)
        loads.append(f * scale)
        case_nodes.append(cn)
        case_wet.append(cw)
        case_rings.append(ringed & ~plain)
        load_nodes |= cn

    gravity = None
    if study.gravity is not None:
        # g/cm^3 -> kg/mm^3 is 1e-6; times m/s^2 gives N per mm^3.
        gravity = study.material.density * 1e-6 * study.gravity
    raw = None
    if study.inertia_relief:
        raw = [f.copy() for f in loads]
        loads = _inertia_relief(study, mesh, loads, fixed, tol_floor)
    if seats and check_loads and not pinned and not study.inertia_relief:
        _seated_statics(study, xyz, fixed, seats, loads, spreads)
    elif check_loads and not seats and springs is None and study.supports and not study.self_equilibrated \
            and not study.inertia_relief:
        _held_statics(xyz, fixed)
    if study.inertia_relief:
        inserts = []    # the loads were rebalanced on the part's inertia: the ring's are not
        case_rings = [np.zeros(mesh.n_nodes, bool) for _ in case_rings]
    return BoundaryConditions(fixed, springs, support_nodes, loads, case_nodes, load_nodes, gravity, regions,
                              case_wet, imbalance, spreads, raw, pinned=pinned, seats=seats,
                              pin_springs=pin_springs, inserts=inserts, case_ring_nodes=case_rings)


def _free_motions(xyz, fixed):
    """The rigid motions the fixed dofs leave free: (6 x k orthonormal
    basis of the free wrench space, in force and moment-over-size units,
    or None when they hold every one; the projector onto it; the scale)."""
    c = xyz.mean(axis=0)
    r = xyz - c
    size = max(float(np.ptp(xyz, axis=0).max()), 1e-9)
    sc = np.array([1, 1, 1, 1 / size, 1 / size, 1 / size])
    cols = []
    for a in range(3):
        nodes = np.nonzero(fixed[a::3])[0]
        if len(nodes):
            e = np.zeros((len(nodes), 3))
            e[:, a] = 1.0
            cols.append(np.concatenate([e, np.cross(r[nodes], e)], axis=1))
    q = np.zeros((6, 0))
    if cols:
        w = np.concatenate(cols, axis=0)
        u, sv, _ = np.linalg.svd((w * sc).T, full_matrices=False)
        q = u[:, sv > 1e-9 * max(sv.max(), 1e-300)]
    proj = np.eye(6) - q @ q.T
    return proj, sc, r


def _motion_words(v) -> str:
    """A free rigid motion (6-vector: force part, moment part) in words."""
    fpart, mpart = v[:3], v[3:]
    if np.linalg.norm(fpart) >= np.linalg.norm(mpart):
        d = fpart / np.linalg.norm(fpart)
        ax = int(np.argmax(np.abs(d)))
        return f"slide along {'xyz'[ax]}" if abs(d[ax]) > 0.9 else \
            f"slide along [{', '.join(f'{x:.2f}' for x in d)}]"
    d = mpart / np.linalg.norm(mpart)
    ax = int(np.argmax(np.abs(d)))
    return f"turn about {'xyz'[ax]}" if abs(d[ax]) > 0.9 else \
        f"turn about [{', '.join(f'{x:.2f}' for x in d)}]"


def _held_statics(xyz, fixed) -> None:
    """Refuse supports that leave the part a rigid motion. With fixed
    supports alone (no seats, pins or springs) the stiffness is singular
    when the held dofs span fewer than the six rigid motions: three
    supports fixed along z alone leave it free to slide in x and y and turn
    about z, every node of a line fixed leaves it free to turn about that
    line. Solves of it either fail or wander off, and the run stalls on a
    deflection of 1e13 mm instead of saying why; read on the rigid part it
    is one 6x6 projector, so `check` says it in well under a second."""
    proj, _, _ = _free_motions(xyz, fixed)
    if np.abs(proj).max() < 1e-9:
        return
    w, v = np.linalg.eigh(proj)
    free = [v[:, i] for i in range(6) if w[i] > 0.5]
    raise ValueError(
        "the supports leave the part free to " + " and ".join(_motion_words(m) for m in free) +
        ": nothing holds it that way, so its stiffness is singular and no load case can be solved. Hold "
        "it where it is really held that way (supports[].fix on a face or bore that stops it, a seat with "
        "supports[].push, or fasteners with their bores as supports)")


def _seated_statics(study, xyz, fixed, seats, loads, spreads=None) -> None:
    """Refuse a load case the supports cannot hold as a rigid body. A seat
    (supports[].push) only pushes: a case whose force and moment no sum of
    the fixed supports' reactions and the seats' pushes balances lifts the
    part off, and every solve of it flies off (pushseat.resolve). Read on
    the rigid part, it is six numbers per case, so `check` says so in a
    second and names the case and the way it goes, where the run used to
    spend its time and then call it a limit other than strength (held-out
    batch 3: a pillow block lifted by its shaft, a Pi mount pulled off its
    wall, both held by screws fixed across their axis only)."""
    from scipy.optimize import nnls

    proj, sc, r = _free_motions(xyz, fixed)

    def wrench(nodes, dirs):
        return np.concatenate([dirs, np.cross(r[nodes], dirs)], axis=1)

    if np.abs(proj).max() < 1e-9:
        return                                        # the fixed supports hold every rigid motion
    seat_cols, fric_cols, fric_mu = [], [], []
    # friction across a seat (pushseat.py) as eight forces round each node, bounded together by
    # mu times its push: the octagon round the friction disc, so a case that holds is never refused
    ring = 2 * np.pi * np.arange(8) / 8
    widen = 1.0 / np.cos(np.pi / 8)
    for s in seats:
        idx, n = np.asarray(s[0]), np.asarray(s[1], float)
        mu = float(s[3] or 0.0) if len(s) > 3 else 0.0
        most = 100 if mu > 0 else 400
        if len(idx) > most:
            idx = idx[np.linspace(0, len(idx) - 1, most).astype(int)]
        if mu > 0:
            from .pushseat import _across

            a, b_ = _across(n)
            for k, t in enumerate(ring):
                fric_cols.append(wrench(idx, np.repeat((np.cos(t) * a + np.sin(t) * b_)[None], len(idx), axis=0)))
            fric_mu.append((len(np.concatenate(seat_cols, axis=0)) if seat_cols else 0, len(idx), mu * widen))
        seat_cols.append(wrench(idx, np.repeat(n[None], len(idx), axis=0)))
    sw = (proj @ (np.concatenate(seat_cols, axis=0) * sc).T)
    sf = (proj @ (np.concatenate(fric_cols, axis=0) * sc).T) if fric_cols else None

    def balance(b):
        """(seat pushes, what is left unbalanced) for the wrench b."""
        if sf is None:
            lam, _ = nnls(sw, b, maxiter=50 * sw.shape[1])
            return b - sw @ lam
        import scipy.sparse as sp_
        from scipy.optimize import linprog

        nl, nf = sw.shape[1], sf.shape[1]
        # columns: pushes, friction (8 per node, seat by seat, direction-major), slack +, slack -
        a_eq = np.concatenate([sw, sf, np.eye(6), -np.eye(6)], axis=1)
        rows, cols, vals = [], [], []
        f0, r = 0, 0
        for start, count, m in fric_mu:
            for j in range(count):
                for k in range(8):
                    rows.append(r)
                    cols.append(nl + f0 + k * count + j)
                    vals.append(1.0)
                rows.append(r)
                cols.append(start + j)
                vals.append(-m)
                r += 1
            f0 += 8 * count
        a_ub = sp_.csr_matrix((vals, (rows, cols)), shape=(r, nl + nf + 12))
        cost = np.concatenate([np.zeros(nl + nf), np.ones(12)])
        res = linprog(cost, A_ub=a_ub, b_ub=np.zeros(r), A_eq=a_eq, b_eq=b, bounds=(0, None), method="highs")
        if res.status != 0:
            lam, _ = nnls(sw, b, maxiter=50 * sw.shape[1])
            return b - sw @ lam
        x = res.x
        return b - sw @ x[:nl] - sf @ x[nl:nl + nf]
    for ci, (case, f) in enumerate(zip(study.load_cases, loads)):
        if spreads and ci < len(spreads) and spreads[ci] is not None:
            continue                                  # a force of unknown direction: pushseat.SeatedSpread
        fn = f.reshape(-1, 3)
        on = np.abs(fn).sum(axis=1) > 0
        if not on.any():
            continue
        wl = np.concatenate([fn[on].sum(axis=0), np.cross(r[on], fn[on]).sum(axis=0)]) * sc
        b = -(proj @ wl)
        total = float(np.abs(fn).sum())
        if np.linalg.norm(b) < 1e-6 * total:
            continue
        rest = balance(b)
        if np.linalg.norm(rest) <= 1e-3 * max(np.linalg.norm(wl), 1e-12):
            continue
        left = -rest                                 # what nothing balances: where the part goes
        fpart, mpart = left[:3], left[3:]
        if np.linalg.norm(fpart) >= np.linalg.norm(mpart):
            d = fpart / np.linalg.norm(fpart)
            ax = int(np.argmax(np.abs(d)))
            way = f"along {'+' if d[ax] > 0 else '-'}{'xyz'[ax]}" if abs(d[ax]) > 0.9 else \
                f"along [{', '.join(f'{v:.2f}' for v in d)}]"
        else:
            d = mpart / np.linalg.norm(mpart)
            ax = int(np.argmax(np.abs(d)))
            way = (f"turning about {'xyz'[ax]}" if abs(d[ax]) > 0.9 else
                   f"turning about [{', '.join(f'{v:.2f}' for v in d)}]")
        raise ValueError(
            f"load case {case.name!r} lifts the part off its seat and nothing else holds it {way}: "
            "a seat (supports[].push) only pushes" + (", and holds across only by its friction" if sf is not None
                                                      else " (give it friction to hold across)") +
            ". Hold it where it is really held that way: write the screws "
            "as `fasteners` with their bores as supports (a screw's head then holds the part along it), "
            "fix a bore or face along that axis, or add a seat on the face the load presses it against")


def _near_cells(mesh: HexMesh, nodes: np.ndarray, layers: int = 2) -> np.ndarray:
    """Cells within `layers` cells of the given nodes: the band round
    supports and loads that verification leaves out of the peak stress."""
    near = np.asarray(nodes, bool).copy()
    cells = np.zeros(mesh.n_elem, bool)
    for _ in range(layers):
        cells = near[mesh.enode].any(axis=1)
        near[mesh.enode[cells].ravel()] = True
    return cells


def gravity_force(mesh: HexMesh, density: np.ndarray, gravity: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Nodal self-weight for per-element solid fraction `density`."""
    density = np.asarray(density, float)
    extra = getattr(mesh, "stock_density", None)
    if extra is not None:
        density = density + extra         # bought members weigh what they weigh (stock.py)
    per_node = (density * mesh.pitch ** 3 / 8.0)[:, None] * gravity[None, :]
    f = np.zeros(mesh.n_dof)
    for a in range(3):
        f += np.bincount(mesh.edof[:, a::3].ravel(), weights=np.repeat(per_node[:, a], 8),
                         minlength=mesh.n_dof)
    f[fixed] = 0.0
    return f


def build_domain(study: Study, pitch: float | None = None, cut_walls: bool = False) -> Domain:
    """The design grid. `cut_walls`: a pinned wall under two cells thick on
    it enters as cut cells (thinwall.py) where thinwall.applies; `loadpath
    run`'s optimizer asks for that, the other solves keep whole cells (or
    their own thin skin, switch._thin_skin)."""
    if pitch or study.pitch:
        return _build_domain(study, float(pitch) if pitch else float(study.pitch), cut_walls)
    # maxElements is a cap: the pitch is estimated from a coarse occupancy
    # sample, which came out 2-5% over in the brief evals (20988 cells for
    # 20000); grow the cells until the grid fits
    # (cell_cap: the constrained method's finer grid is allowed past it; the
    # cap had undone that, and the arm link's first constrained run went back
    # to verifying 1.32 mm against the optimizer's 0.48 mm)
    p = choose_pitch(study)
    cap = cell_cap(study)
    for _ in range(4):
        d = _build_domain(study, p, cut_walls)
        if d.n_elem <= cap:
            return d
        p *= (d.n_elem / cap) ** (1.0 / 3.0) * 1.003
    return d


def _face_bridge(preserve, pres_f, dims, open_cells=None):
    """Drawn geometry joined through faces. A wall thinner than a cell and
    curved across the grid (a 1.2 mm duct at 1.5 mm pitch) comes out as a
    staircase of cells touching only along edges or at corners, each a hinge:
    with a keep-out on both sides (the prop's air inside, the bumper outside)
    nothing else joins them, the stiffness matrix is singular and the solve
    never converged (a drone frame, 96 such pieces). Each edge or corner
    contact between preserved cells gets the cells that join it through faces,
    the ones most inside the drawn shape: the wall is there, the grid only
    lost it. Where a cell the optimizer may fill (open_cells) already joins
    them, nothing is added: the design decides that joint, and at a near-void
    density it still carries a little stiffness."""
    g = preserve.reshape(dims).copy()
    o = (np.zeros(dims, bool) if open_cells is None else open_cells.reshape(dims)) | g
    if not g.any():
        return preserve
    f = pres_f.reshape(dims)
    n = tuple(int(x) for x in dims)

    def shifted(d):
        # shifted(d)[c] = g[c + d], False off the grid
        out = np.zeros(n, bool)
        src = tuple(slice(max(0, di), n[k] + min(0, di)) for k, di in enumerate(d))
        dst = tuple(slice(max(0, -di), n[k] + min(0, -di)) for k, di in enumerate(d))
        out[dst] = g[src]
        return out

    def shifted_o(d):
        out = np.zeros(n, bool)
        src = tuple(slice(max(0, di), n[k] + min(0, di)) for k, di in enumerate(d))
        dst = tuple(slice(max(0, -di), n[k] + min(0, -di)) for k, di in enumerate(d))
        out[dst] = o[src]
        return out

    for _ in range(3):
        added = False
        # edge contacts: cells c and c+a+b, both a, b unit steps on two axes
        for ax, bx in ((0, 1), (0, 2), (1, 2)):
            for sb in (1, -1):
                a = [0, 0, 0]; a[ax] = 1
                b = [0, 0, 0]; b[bx] = sb
                ab = [a[k] + b[k] for k in range(3)]
                diag = g & shifted(ab) & ~shifted_o(a) & ~shifted_o(b)
                if not diag.any():
                    continue
                # candidates: c+a (shifted by a) and c+b; take the one more inside
                ia = np.argwhere(diag)
                pa = ia + a
                pb = ia + b
                fa = f[tuple(pa.T)]
                fb = f[tuple(pb.T)]
                pick = np.where((fa >= fb)[:, None], pa, pb)
                g[tuple(pick.T)] = True
                o[tuple(pick.T)] = True
                added = True
        # corner contacts: c and c+(1,+-1,+-1) with no face path through the
        # six cells between: join through the two most inside
        for sy in (1, -1):
            for sz in (1, -1):
                d = (1, sy, sz)
                corner = g & shifted(d)
                if not corner.any():
                    continue
                ic = np.argwhere(corner)
                paths = [((1, 0, 0), (1, sy, 0)), ((1, 0, 0), (1, 0, sz)), ((0, sy, 0), (1, sy, 0)),
                         ((0, sy, 0), (0, sy, sz)), ((0, 0, sz), (1, 0, sz)), ((0, 0, sz), (0, sy, sz))]
                ok = np.zeros(len(ic), bool)
                for p1, p2 in paths:
                    ok |= o[tuple((ic + p1).T)] & o[tuple((ic + p2).T)]
                if ok.all():
                    continue
                ic = ic[~ok]
                best = np.full(len(ic), -1.0)
                choice = np.zeros((len(ic), 2, 3), int)
                for p1, p2 in paths:
                    q1, q2 = ic + p1, ic + p2
                    sc = f[tuple(q1.T)] + f[tuple(q2.T)] + 2.0 * (o[tuple(q1.T)] | o[tuple(q2.T)])
                    better = sc > best
                    best = np.where(better, sc, best)
                    choice[better, 0] = q1[better]
                    choice[better, 1] = q2[better]
                for q in (choice[:, 0], choice[:, 1]):
                    g[tuple(q.T)] = True
                    o[tuple(q.T)] = True
                added = True
        if not added:
            break
    return g.reshape(-1)


def drawn_joints(pres_grid, elem_ijk):
    """Cells (elem_ijk order) with drawn cells on two or more faces: where
    a staircase of drawn cells (a thin curved wall) meets only at edges, these
    are what join it through faces (see _face_bridge). The optimizer may keep
    or empty them; once kept they must not be pruned as carrying nothing (a
    duct that carries only its own inertia), or the wall falls apart into
    hinged pieces and the solve cannot converge."""
    p = np.pad(pres_grid, 1)
    i, j, k = (elem_ijk + 1).T
    n = np.zeros(len(elem_ijk), int)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        n += p[i + d[0], j + d[1], k + d[2]]
    return n >= 2


def _sharp_inside_edges(study, pitch: float) -> list:
    """The edges of keep-out boxes the part is formed round: (keep-out id,
    axis the edge runs along, its two ends), each stretch of an edge with
    material round it.

    A box drawn for a clearance (a clamp's throat, a slot for a mating
    part, a pocket for a board) has square edges, and where the design
    space wraps one (material against both faces that meet at it and in
    the quarter between them) the part gets a sharp inside corner. Stress
    concentrates there whatever the part's shape is elsewhere: a C-clamp's
    solid envelope read a safety factor of 1.37 at its throat's corner,
    1.73 with a 4 mm radius there and 2.36 with 8 mm on a 25 mm spine,
    and no search inside the envelope could do better than the block.
    Read on the shapes; the grid only sets how much material counts as
    being there (a cell either side).

    Not a warning of `check`: a pocket for a square motor has such edges
    and carries nothing round them (14 of 55 packaged studies have one, and
    their parts pass). It is what the solid block's reading is held against
    when its peak sits on one (study.bound_note)."""
    from .shapes import Box, Union

    def boxes(shape):
        if isinstance(shape, Box):
            return [shape]
        if isinstance(shape, Union):
            return [b for sh in shape.shapes for b in boxes(sh)]
        return []

    design = list(study.design_space) + [r.shape for r in study.preserve]
    keep = [r.shape for r in study.keep_out]
    if not design or not keep:
        return []

    def solid(pts):
        inside = np.min([sh.sdf(pts) for sh in design], axis=0) < 0
        return inside & (np.min([sh.sdf(pts) for sh in keep], axis=0) > 0)

    d = float(pitch)
    out = []
    for r in study.keep_out:
        for b in boxes(r.shape):
            for ax in range(3):
                u, v = [a for a in range(3) if a != ax]
                n = max(int(np.ceil((b.hi[ax] - b.lo[ax]) / (0.5 * d))), 2)
                step = (b.hi[ax] - b.lo[ax]) / n
                t = b.lo[ax] + (np.arange(n) + 0.5) * step
                for su in (-1, 1):
                    for sv in (-1, 1):
                        e = np.zeros((n, 3))
                        e[:, ax] = t
                        e[:, u] = b.hi[u] if su > 0 else b.lo[u]
                        e[:, v] = b.hi[v] if sv > 0 else b.lo[v]
                        nu, nv = np.zeros(3), np.zeros(3)
                        nu[u], nv[v] = su * d, sv * d
                        on = solid(e + nu) & solid(e + nv) & solid(e + nu + nv)
                        if on.sum() * step < 2.0 * d:
                            continue
                        i = np.flatnonzero(on)
                        p0, p1 = e[i[0]].copy(), e[i[-1]].copy()
                        p0[ax] -= 0.5 * step
                        p1[ax] += 0.5 * step
                        out.append((r.id, "xyz"[ax], p0, p1))
    return out


def _overlap_share(shape, cutters, pitch: float, cap: int = 40_000):
    """(share, wall, shrink): the share of `shape`'s volume that lies inside
    any of `cutters`, the thickest wall of what they leave (mm), and how far
    what is left falls short of the shape's own extent along its worst axis
    (mm), read on the shapes themselves at points a quarter cell apart
    ((None, 0, 0) when a shape gives no distance)."""
    try:
        lo, hi = (np.asarray(v, float) for v in shape.bounds())
        ext = np.maximum(hi - lo, 1e-9)
        step = max(0.25 * pitch, float(np.prod(ext) / cap) ** (1.0 / 3.0))
        axes = [lo[a] + step * (np.arange(max(int(np.ceil(ext[a] / step)), 1)) + 0.5) for a in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        d_in = -shape.sdf(pts)
        pts, d_in = pts[d_in > 0.0], d_in[d_in > 0.0]
        if not len(pts):
            return None, 0.0, 0.0
        d_out = np.full(len(pts), np.inf)
        for c in cutters:
            d_out = np.minimum(d_out, c.sdf(pts))
        left = d_out > 0.0
        # the thickest wall of what is left: twice the deepest point's
        # distance to the nearer of the shape's own surface and a cutter's
        wall = 2.0 * float(np.minimum(d_in[left], d_out[left]).max()) + step if left.any() else 0.0
        if left.any():
            shrink = float(((pts.max(axis=0) - pts.min(axis=0))
                            - (pts[left].max(axis=0) - pts[left].min(axis=0))).max())
        else:
            shrink = float(ext.max())
        return float(1.0 - left.mean()), wall, shrink
    except Exception:  # noqa: BLE001 - a shape with no distance field: fall back to the cells
        return None, 0.0, 0.0


# what a mirror plane must not have to pass over, for the test below to be
# the whole story: each of these puts something in the design problem that
# the load vectors, supports and pinned cells do not show
_SYMMETRY_BLIND = ("thermal", "access", "pattern", "stock", "multi_material", "vibration", "mass_properties",
                   "aero", "foundry", "cast_feed", "build_sim")


def _found_symmetry(study: Study, mesh: HexMesh, dims, pres_el, bc, loads, weights, spreads) -> list:
    """The mirror planes (through the design space's centre, 'x', 'y', 'z')
    the problem itself has and the study did not declare.

    A design problem that is its own mirror image has a mirror-symmetric
    optimum: the mirror image of any design carries the mirrored loads
    exactly as well, and the loads are the same set. The optimizer run
    without the constraint still drifts off it (round-off picks a side where
    two layouts tie, and every later iteration builds on the pick), so a
    fork's two arms or a plate's four straps come out different for no
    reason a load gives. That reads as a mistake, and it is one: the part
    holds a preference the problem does not have. So the symmetry found
    here is held, as a declared one is.

    Found on the grid, where the problem is finally stated: every cell has
    its mirror cell, the same cells are pinned solid, the same freedoms are
    held at mirror nodes, and every load case's mirror image is a load case
    of the same weight (itself, usually). A force of unknown direction is
    its own image when its nodes are; a cone's axis must lie in the plane.
    A push given one way only (a sideways load along +x) is not symmetric
    and is not taken to be: say `"direction": "any"`, add the opposite
    case, or declare `symmetry`. `"symmetry": false` turns the search off.
    Self-weight must lie in the plane, and so must the build direction of
    a part grown to print without supports (an overhang is not its own
    mirror image across the bed)."""
    raw = getattr(study, "raw", None) or {}
    study.symmetry_missed = {}
    if raw.get("symmetry") is False or os.environ.get("LOADPATH_AUTO_SYMMETRY", "1") == "0":
        return []
    if any(getattr(study, a, None) for a in _SYMMETRY_BLIND) or bc.springs is not None or bc.raw_loads:
        return []
    if getattr(study, "inertia_masses", None) or getattr(study, "inertia_bodies", None):
        return []
    dims = tuple(int(d) for d in dims)
    lut = np.full(dims, -1, np.int64)
    lut[tuple(mesh.elem_ijk.T)] = np.arange(mesh.n_elem)
    ndims = tuple(d + 1 for d in dims)
    nlut = np.full(ndims, -1, np.int64)
    nlut[tuple(mesh.node_ijk.T)] = np.arange(mesh.n_nodes)
    fixed = np.asarray(bc.fixed, bool).reshape(-1, 3)
    out = []
    why = {}
    study.symmetry_missed = why
    for a, axis in enumerate("xyz"):
        if axis in study.symmetry:
            continue
        if study.gravity is not None and abs(float(study.gravity[a])) > 1e-9 * float(np.linalg.norm(study.gravity)):
            why[axis] = "gravity"
            continue
        proc = getattr(study, "process", None)
        builds = [getattr(study, "print_direction", None)]
        if proc is not None and proc.overhangs:
            # (layers alone are their own mirror image across the bed; a
            # part grown to stand without supports is not)
            builds.append(proc.build)
        if any(_along(b, a) for b in builds):
            why[axis] = "build direction"
            continue
        m_ijk = mesh.elem_ijk.copy()
        m_ijk[:, a] = dims[a] - 1 - m_ijk[:, a]
        em = lut[tuple(m_ijk.T)]
        if (em < 0).any() or not np.array_equal(pres_el[em], pres_el):
            why[axis] = "shape"
            continue
        n_ijk = mesh.node_ijk.copy()
        n_ijk[:, a] = dims[a] - n_ijk[:, a]
        nm = nlut[tuple(n_ijk.T)]
        if (nm < 0).any() or not np.array_equal(fixed[nm], fixed):
            why[axis] = "supports"
            continue
        sign = np.ones(3)
        sign[a] = -1.0

        def image(f):
            return (np.asarray(f, float).reshape(-1, 3)[nm] * sign).ravel()

        def same(u, v):
            scale = max(float(np.abs(u).max(initial=0.0)), float(np.abs(v).max(initial=0.0)))
            return scale == 0.0 or float(np.abs(u - v).max()) <= 1e-6 * scale

        ok = True
        for i, f in enumerate(loads):
            sp = spreads[i] if i < len(spreads) else None
            hit = False
            for j, g in enumerate(loads):
                if abs(float(weights[i]) - float(weights[j])) > 1e-12:
                    continue
                sq = spreads[j] if j < len(spreads) else None
                if (sp is None) != (sq is None):
                    continue
                if sp is None:
                    hit = same(image(f), np.asarray(g, float))
                else:
                    hit = (sp.cos_half == sq.cos_half and same(image(sp.base), sq.base)
                           and all(same(image(sp.basis[k]) * sign[k], sq.basis[k]) for k in range(3))
                           and (sp.cos_half <= -1.0 or (abs(float(sp.nominal[a])) < 1e-9
                                                         and same(np.asarray(sp.nominal), np.asarray(sq.nominal)))))
                if hit:
                    break
            if not hit:
                ok = False
                why[axis] = "load case %d" % i
                break
        if ok:
            out.append(axis)
    return out


def _region_kind(study: Study, ri: int):
    """"keepOut" when bc.regions[ri] lines a keep-out (a support's or a
    load's region), else None."""
    if ri < len(study.supports):
        sup = study.supports[ri]
        if sup.shape is not None or sup.region is None:
            return None
        try:
            return study.region_shape(sup.region)[1]
        except ValueError:
            return None
    return "keepOut" if _region_shape_of(study, ri) is not None else None


def _region_shape_of(study: Study, ri: int):
    """The keep-out a load region lines (bc.regions[ri]: supports first, then
    each case's loads in order, pressures left out), or None when the region
    is a solid or an inline shape (its nodes are in it, not round it)."""
    k = len(study.supports)
    for case in study.load_cases:
        for ld in case.loads:
            if ld.pressure:
                continue
            if k == ri:
                if ld.shape is not None or ld.region is None:
                    return None
                try:
                    shape, kind = study.region_shape(ld.region)
                except ValueError:
                    return None
                return shape if kind == "keepOut" else None
            k += 1
    return None


def _along(build, a: int) -> bool:
    """Whether a build direction ((axis, sign), a vector, or "auto"/None) has a part along axis a."""
    if build is None or isinstance(build, str):
        return False
    try:
        if len(build) == 2 and not hasattr(build[0], "__len__") and int(build[0]) == build[0] and abs(build[1]) == 1:
            return int(build[0]) == a
        v = np.asarray(build, float)
        return abs(float(v[a])) > 1e-9 * float(np.linalg.norm(v))
    except (TypeError, ValueError):
        return True


def _build_domain(study: Study, pitch: float, cut_walls: bool = False) -> Domain:
    lo, hi = study.bounds()
    center = 0.5 * (lo + hi)
    dims = np.maximum(np.ceil((hi - lo) / pitch - 1e-9).astype(int), 1)
    # Centered grid, so a symmetry plane through the design space center is
    # also a plane between cells and every cell has an exact mirror. (Starting
    # the grid on the design space's minimum face along non-symmetric axes
    # was tried for the printed hook's half-cell overhang at its wall; it
    # moved every case's search path and lost 3 of 8 evals. Not worth it.)
    origin = center - 0.5 * dims * pitch
    ii, jj, kk = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    centers = origin + (ijk + 0.5) * pitch

    design_f = _fraction(study.design_space, centers, pitch)
    # thin pinned walls as cut cells (thinwall.py): their cells are not whole
    # preserved cells, so they are left out of the drawn geometry's share
    thin_ids = set()
    if cut_walls:
        from . import thinwall
        if thinwall.applies(study):
            thin_ids = {rid for rid, _ in thinwall.thin_shells(study, pitch)}
    pres_f = _fraction([r.shape for r in study.preserve if r.id not in thin_ids], centers, pitch)
    keep_f = _fraction([r.shape for r in study.keep_out], centers, pitch)
    # Only a keep-out that cuts (a bore, a way in, cutsPreserve) takes
    # preserved geometry out; a clearance volume keeps the optimizer's
    # material out and leaves what the designer drew. The skin is cut by
    # bores alone (spec.cuts_of).
    cutting = [r.shape for r in study.keep_out if getattr(r, "cuts", True)]
    cut_f = keep_f if len(cutting) == len(study.keep_out) else _fraction(cutting, centers, pitch)
    preserve = (pres_f >= 0.5) & (cut_f < 0.5)
    skin = [r.shape for r in study.preserve if r.id == "skin" and r.id not in thin_ids]
    if skin and any(not getattr(r, "bore", True) for r in study.keep_out if getattr(r, "cuts", True)):
        bores = [r.shape for r in study.keep_out if getattr(r, "cuts", True) and getattr(r, "bore", True)]
        bore_f = _fraction(bores, centers, pitch) if bores else np.zeros(len(centers))
        preserve |= (_fraction(skin, centers, pitch) >= 0.5) & (bore_f < 0.5)
    preserve = _face_bridge(preserve, pres_f, dims, (design_f >= 0.5) & (keep_f == 0))
    design = (design_f >= 0.5) & ~preserve & (keep_f == 0)
    wall_cells, wall, wall_open = None, None, None
    if thin_ids:
        # the cells a thin wall crosses: in the analysis with the wall's own
        # cut-cell stiffness; the rest of each is designed where the design
        # space has it and no keep-out touches it, and air otherwise
        from .thinwall import wall_cells as _wall_cells
        bores = [r.shape for r in study.keep_out if getattr(r, "cuts", True) and getattr(r, "bore", True)]
        wall_cells = _wall_cells(study, centers, pitch, preserve, cut=bores)
        if wall_cells is not None:
            wall = np.zeros(len(centers), bool)
            wall[wall_cells.idx] = True
            wall_open = wall & (design_f >= 0.5) & (keep_f == 0)
            design |= wall_open
    stock_cell = None
    if getattr(study, "stock", None):
        # Bought members (stock.py) are bodies in the analysis: every cell one touches is in the mesh,
        # pinned, and carries the member's own stiffness there (boundary_conditions adds it). A cell its
        # wall shares with a sleeve is the member's; the sleeve's part of it is kept as printed material.
        from .stock import _shares
        stock_f = np.zeros(len(centers))
        free = np.zeros(len(centers), bool)
        joined = np.zeros(len(centers), bool)
        body_f = np.zeros(len(centers))
        for m in study.stock:
            f = _shares(m.fe_shape(), centers, pitch)
            stock_f = np.maximum(stock_f, f)
            on = f > 0
            body_f[on] = np.maximum(body_f[on], _shares(m.body(), centers[on], pitch, floor=False))
            s_c = (centers - m.p0) @ m.axis
            for j in m.joints:
                joined |= (f > 0) & (s_c >= j.span[0] - 0.5 * pitch) & (s_c <= j.span[1] + 0.5 * pitch)
        free = (stock_f > 0) & ~joined
        stock_cell = stock_f >= 1.0 / 64.0
        preserve &= ~stock_cell
        design &= ~stock_cell
        if free.any():
            # where a member is not joined nothing printed may touch it: a cell of air round it on this
            # grid, whatever clearance the keep-out draws
            fg = ndimage.binary_dilation((free & stock_cell).reshape(dims), structure=_CONN26).ravel()
            design &= ~fg
    if wall is not None and stock_cell is not None:
        # a bought member's cell is the member's (stock.py), wall or not
        keep_w = ~stock_cell[wall_cells.idx]
        wall_cells = dataclasses.replace(wall_cells, idx=wall_cells.idx[keep_w], ind=wall_cells.ind[keep_w],
                                         phi=wall_cells.phi[keep_w])
        wall &= ~stock_cell
        wall_open &= ~stock_cell
    active = preserve | design
    if stock_cell is not None:
        active |= stock_cell
    if wall is not None:
        active |= wall
    if not active.any():
        raise ValueError("the design space is empty after removing keep-outs")

    warnings = []
    # A keep-out drawn round a preserved part (a board's clear zone round the
    # rails that hold it) takes the part out of the analysis without a word:
    # the touch microphone's rails were read as missing except at the wall
    # they hang from, so its board looked held at one end only.
    lost_cells = (pres_f >= 0.5) & ~preserve
    if stock_cell is not None:
        lost_cells &= ~stock_cell          # a sleeve's cells its member shares are the member's, not lost
    if lost_cells.any() and cutting:
        on = np.flatnonzero(pres_f >= 0.5)
        for r in study.preserve:
            mine = on[_fraction([r.shape], centers[on], pitch) >= 0.5]
            gone = mine[lost_cells[mine]]
            if len(mine) and len(gone) >= 4 and len(gone) > 0.25 * len(mine):
                cut_by = [k for k in study.keep_out if getattr(k, "cuts", True)
                          and _fraction([k.shape], centers[gone], pitch).mean() >= 0.25]
                if cut_by and all(getattr(k, "cuts_declared", False) for k in cut_by):
                    continue   # the study says these keep-outs cut what is drawn
                # Counted on the shapes, not the cells. A cell the two share
                # goes to the keep-out, so on a coarse grid a ring that only
                # touches a clearance loses a layer of cells to it: every
                # fastener's own seat ring read "43% inside" its own head
                # path, which starts at the face the ring ends on, and a
                # hinge knuckle with its pin through it "31%" (20% by
                # volume). The warning is for a keep-out that really takes
                # the drawn part away.
                # And a hole through a boss is the bore doing its job (the
                # packaged arm link's bearing takes 48% of its shoulder boss
                # and warned): what matters is what the cut leaves. A hole
                # through a part leaves it reaching as far as it was drawn,
                # with a wall round the hole; a clear zone over one end of a
                # rail leaves a stub. Not warned when what is left still
                # spans the drawn part (to within a cell, which is all the
                # grid can tell) and its thickest wall is one the process
                # can make. No share of the volume decides it: a bearing
                # takes half its boss and a clearance box that takes half a
                # rail is the mistake this warning is for.
                share, wall, shrink = _overlap_share(r.shape, [k.shape for k in cut_by], pitch)
                if share is not None:
                    floor = study.process.min_feature if study.process is not None else 0.0
                    if share < 1.0 and shrink <= pitch and wall >= floor:
                        continue
                pct = 100 * (share if share is not None else len(gone) / len(mine))
                by = [k.id for k in cut_by]
                warnings.append(
                    f"preserve {r.id!r}: {pct:.0f}% of it is inside keep-out "
                    f"{', '.join(map(repr, by)) or 'regions'} and is left out of the part; a clear zone drawn round "
                    "a preserved part takes the part away with it. Subtract the preserve from the keep-out "
                    "({\"type\": \"subtract\", \"shapes\": [keep-out, preserve]}), or give a clearance keep-out "
                    "\"bore\": false (it then never cuts what is drawn), if it is meant to stay.")
    for rid, t in study.shells:
        if rid in thin_ids:
            from .thinwall import MIN_CELLS
            if t < MIN_CELLS * pitch:
                warnings.append(
                    f"{rid} is {t:g}mm thick, {t / pitch:.1f} cells: the optimizer reads a wall under a cell thick "
                    "too stiff in bending (1.4x at two thirds of a cell, 1.9x at under half). Set resolution.pitch "
                    f"to {t:.2g}mm or less, or raise resolution.maxElements.")
            continue
        if t < 2.0 * pitch:
            warnings.append(
                f"{rid} is {t:g}mm thick, {t / pitch:.1f} cells: a wall under two cells thick is too stiff in "
                f"bending on this grid (and under one is not there at all). Set resolution.pitch to {t / 2:.2g}mm "
                "or less, or raise resolution.maxElements.")
    # Only for a minMember the study asked for: a process's own floor is what
    # the printer can make, not a member the design needs, and at a first-look
    # resolution it would warn on every powder or FDM study.
    if study.min_member is not None and study.min_member < 1.5 * pitch and study.raw.get("minMember") is not None:
        warnings.append(
            f"minMember {study.min_member:g}mm is under 1.5 cells of this {pitch:.2f}mm grid ({1.5 * pitch:.2f}mm); "
            "members that thin cannot be resolved. Raise resolution.maxElements or set resolution.pitch "
            "(raising minMember coarsens the default pitch, which follows it).")

    if study.max_member is not None and study.max_member < 3.0 * pitch:
        warnings.append(
            f"maxMember {study.max_member:g}mm is under 3 cells ({3 * pitch:.2f}mm); the thickness limit is too "
            "coarse to hold on this grid. Raise resolution.maxElements or set resolution.pitch.")
    access = getattr(study, "access", None)
    if access is not None and access.tool_radius and access.tool_radius < pitch:
        warnings.append(
            f"milling.toolDiameter {2 * access.tool_radius:g}mm is under two cells ({2 * pitch:.2f}mm); slots "
            "that narrow are left to the grid, which is coarser.")

    # Support nodes on a provisional mesh, then drop islands that do not
    # reach one, then rebuild the mesh over what remains.
    mesh = HexMesh(ijk[active], pitch, origin)
    support_nodes = np.zeros(mesh.n_nodes, bool)
    for i, s in enumerate(study.supports):
        support_nodes |= _region_nodes(study, mesh, s.region, s.shape, f"supports[{i}]")
    if study.self_equilibrated:
        # held by its own wetted walls (pressure.stabilize)
        for _, nodes in _wetted(study, mesh).values():
            support_nodes |= nodes
    grid = np.zeros(dims, bool)
    grid[tuple(ijk[active].T)] = True
    lbl, _ = ndimage.label(grid, structure=_CONN26)
    el_lbl = lbl[tuple(mesh.elem_ijk.T)]
    touches = np.zeros(mesh.n_elem, bool)
    touches |= support_nodes[mesh.enode].any(axis=1)
    keep_lbls = np.unique(el_lbl[touches])
    keep = np.isin(el_lbl, keep_lbls)
    if not study.supports and ((study.raw or {}).get("feel") or (study.raw or {}).get("cooling")):
        # a feel study holds the part where each target says (feel.py); a
        # cooling study has no holds (cooling.py)
        keep[:] = True
    elif study.inertia_relief:
        # a free part is its largest body: islands (a fastener's ring cut off
        # by its own head path on a coarse grid, a thin wall the grid loses)
        # float, and the 3-2-1 hold cannot stop them, so the solve never
        # converged (a drone frame on a 3 mm coarse start had 88 of them)
        sizes = np.bincount(el_lbl)
        sizes[0] = 0
        keep = el_lbl == int(np.argmax(sizes))
    else:
        # Space that is in pieces gives a part in pieces, whatever is asked of
        # it: a saddle clamp whose envelope stopped below the top of its pipe
        # was two feet with nothing over the pipe, each on its own screw, and
        # read a safety factor of 5 against the pipe pulling off. Said here,
        # before an optimization is spent on it.
        if len(keep_lbls) > 1:
            cen = mesh.elem_centers()
            sizes = [(int((el_lbl == k).sum()), k) for k in keep_lbls]
            sizes.sort(reverse=True)
            boxes = []
            for n_k, k in sizes[:4]:
                c = cen[el_lbl == k]
                blo, bhi = c.min(axis=0) - 0.5 * pitch, c.max(axis=0) + 0.5 * pitch
                boxes.append(f"{n_k} cells in [{', '.join(f'{v:.4g}' for v in blo)}] to "
                             f"[{', '.join(f'{v:.4g}' for v in bhi)}]")
            warnings.append(
                f"the space the part may take is in {len(keep_lbls)} separate pieces on this {pitch:.2f} mm grid, "
                f"each with its own support ({'; '.join(boxes)}{'; ...' if len(sizes) > 4 else ''}): nothing can "
                "join them, so the part will come out in pieces and no load passes from one to the other. If "
                "it is one part, the design space is missing the way between them (an envelope that stops "
                "short of going over or round a keep-out, a keep-out drawn right across it); a wall thinner "
                "than a cell also reads as a gap, which resolution.pitch or maxElements closes.")
    if (~keep).any():
        dropped = int((~keep).sum())
        if preserve[active][~keep].any():
            warnings.append("some preserved geometry is not connected to any support and was left out of the "
                            "analysis; it will still be in the final part but carries no load")
        warnings.append(f"{dropped} cells not connected to a support were removed")
        act_idx = np.nonzero(active)[0]
        active[act_idx[~keep]] = False
        mesh = HexMesh(ijk[active], pitch, origin)

    if getattr(study, "print_direction", None) is not None:
        proc = getattr(study, "process", None)
        warnings.extend(_unprintable_preserve(study.print_direction, dims, active, preserve, pitch, origin,
                                              process=getattr(proc, "kind", None),
                                              bridge_span=getattr(proc, "bridge_span", None)))
    pres_el = preserve[active]
    joints = drawn_joints(preserve.reshape(dims), mesh.elem_ijk) & ~pres_el
    stock_el = None
    if stock_cell is not None:
        stock_el = stock_cell[active]
        # the printed share of a member's cell: what of it is a sleeve (or other drawn geometry)
        mesh.stock_print = np.where(stock_el, np.minimum(pres_f[active], 1.0 - body_f[active]), 0.0)
        mesh.stock_elastic = study.elastic(design=True)
        pres_el = pres_el | stock_el
    try:
        bc = boundary_conditions(study, mesh)
    except _HeldPreload:
        # Supports that hold a fastener's rings along the screw (a bolt hole
        # fixed in every direction, a bearing face under a thin flange) take
        # the clamp force themselves, so the preload case would load nothing.
        # Drop it and say how to keep it, rather than refuse a study `check`
        # passed and the user never wrote that case for.
        study.load_cases = [c for c in study.load_cases if c.name != "preload"]
        warnings.append("the fasteners' preload case was dropped: the supports hold its seat and face rings along "
                        "the screw, so it would load nothing. To design for the clamp, hold the bolt holes across "
                        "the screw only (fix 'xy' for a screw along z) or support elsewhere")
        bc = boundary_conditions(study, mesh)
    wall_el = wall_void = wall_k = None
    if wall_cells is not None:
        from .thinwall import operator as _wall_operator
        pos = np.full(len(centers), -1)
        pos[np.flatnonzero(active)] = np.arange(mesh.n_elem)
        el = pos[wall_cells.idx]
        on = el >= 0          # (cells left out as unconnected)
        wall_el = np.zeros(mesh.n_elem, bool)
        wall_el[el[on]] = True
        phi = np.zeros(mesh.n_elem)
        phi[el[on]] = wall_cells.phi[on]
        wall_void = wall_el & ~wall_open[active] & ~pres_el
        # the wall's share of each cell is the wall's: the grid's own element
        # carries the rest (scaled, as a bought member's cell is), the wall's
        # weight is its own (thinwall.py)
        prev = getattr(mesh, "stock_scale", None)       # bought members' cells (stock.py), none of them walls
        mesh.stock_scale = np.where(wall_el, np.maximum(1.0 - phi, 1e-4), 1.0 if prev is None else prev)
        prev = getattr(mesh, "stock_density", None)
        mesh.stock_density = phi if prev is None else prev + phi
        mesh.wall_share = phi
        sub = dataclasses.replace(wall_cells, ind=wall_cells.ind[on], phi=wall_cells.phi[on])
        # the wall's own sub-cells, for reading its stress (constrained._Averaged)
        mesh.wall_sub = (el[on], sub.ind)
        wall_k = _wall_operator(sub, mesh, el[on], study.elastic(design=True), bc.fixed)
        bc.springs = wall_k if bc.springs is None else (bc.springs + wall_k).tocsr()
        if bc.raw_loads:
            # a free part's inertia was weighed before the wall's own weight was on the mesh
            bc = dataclasses.replace(boundary_conditions(study, mesh), springs=bc.springs)
    support_nodes, fixed, load_nodes = bc.support_nodes, bc.fixed, bc.load_nodes
    loads = list(bc.loads)
    names = [c.name for c in study.load_cases]
    weights = [c.weight for c in study.load_cases]
    spreads = list(bc.spreads) if bc.spreads else []
    inertial = []
    for name, f in _inertial_cases(study, mesh, fixed, loads):
        inertial.append(len(loads))
        loads.append(f)
        names.append(name)
        weights.append(1.0)
        if spreads:
            spreads.append(None)

    # A small load or support (a bolt hole, a pin, a corner support) needs
    # material to act through: the cells touching it are pinned solid, as if
    # preserved, or the boundary-padded filter (simp.density_filter) makes
    # them too expensive to fill and the support is optimized away (the
    # textbook MBB beam's corners). A large one (a whole wall face, a shelf
    # surface) is left to the optimizer: pinning it forced a full skin over
    # the face, which read as part of the design not being optimized at all.
    # A face is told apart by its shape as well as its share of the cells: a
    # region at most two cells thick that spans more than two filter
    # diameters both ways is a face however small the part is around it. By
    # share alone the printed hook's wall face (1.6%) and the wall bracket's
    # shelf (2.5%) were pinned one cell thick: the optimizer never had to
    # thicken that plate, and it was delivered 1.5 and 1.9 mm thick against a
    # 2.4 and 3.0 mm minimum member. A supporting face is left to the
    # optimizer, like any large one. A loaded face is pinned at the minimum
    # member's thickness: its load is spread over every node of it, so the
    # part has to cover it anyway, and left unpinned the optimizer could
    # empty it (the fastener bracket's shelf, which the part then no longer
    # reached). The MBB beam's load line and the motor mount's pads are
    # under two diameters and stay pinned as they were.
    radius = 1.3 * float(study.min_member) if study.min_member else 1.5 * pitch
    active_grid = np.zeros(dims, bool)
    active_grid[tuple(mesh.elem_ijk.T)] = True
    # the nodes that carry force in some case (an unknown direction's along any axis)
    level_hint = np.full(mesh.n_elem, np.nan)
    forced = np.zeros(mesh.n_nodes, bool)
    for i, f in enumerate(loads):
        forced |= np.abs(np.asarray(f).reshape(-1, 3)).max(axis=1) > 0
        sp = spreads[i] if i < len(spreads) else None
        if sp is not None:
            for b in sp.basis:
                forced |= np.abs(np.asarray(b).reshape(-1, 3)).max(axis=1) > 0
    if not forced.any():
        forced = None
    def strap(touch, shape, thick):
        """The cells within `thick` of a loaded keep-out's surface, behind the
        cells `touch` that carry it (and the level the mesh is cut at there)."""
        # by distance from the loaded surface itself, not by layers
        # of cells: the strap's back is then a surface (level_hint,
        # which the mesh is cut from), not the grid's staircase
        reach = int(np.ceil(thick / pitch)) + 1
        g = np.zeros(dims, bool)
        g[tuple(mesh.elem_ijk[touch].T)] = True
        g = ndimage.binary_dilation(g, structure=ndimage.generate_binary_structure(3, 3),
                                    iterations=reach) & active_grid
        near = np.flatnonzero(g[tuple(mesh.elem_ijk.T)])
        dist = np.abs(shape.sdf(mesh.elem_centers()[near]))
        lv = np.clip(0.5 + (thick - dist) / pitch, 0.0, 1.0)
        # ... and only behind the nodes that carry force: within a
        # cell and a half of a cell that touches one
        gt = np.zeros(dims, bool)
        gt[tuple(mesh.elem_ijk[touch].T)] = True
        # (measured from the cell's foot on the loaded surface, not from the
        # cell: measured from the cell, the strap ran on a member's thickness
        # past the loaded surface's end, along the pin, and stood over a
        # drawn boss's end face as a collar)
        cen = mesh.elem_centers()[near]
        foot = cen - shape.sdf(cen)[:, None] * _sdf_normal(shape, cen, 0.05 * pitch)
        fijk = np.clip(np.floor((foot - mesh.origin) / pitch).astype(int), 0, np.array(dims) - 1)
        side = ndimage.distance_transform_edt(~gt)[tuple(fijk.T)] <= 1.75
        lv = np.where(side, lv, np.nan)
        keep = np.isfinite(lv) & (np.isnan(level_hint[near]) | (lv > np.nan_to_num(level_hint[near], nan=0.0)))
        level_hint[near[keep]] = lv[keep]
        touch = np.zeros(mesh.n_elem, bool)
        touch[near[np.nan_to_num(lv, nan=0.0) >= 0.5]] = True
        return touch

    drawn_held = np.zeros(mesh.n_nodes, bool)
    drawn_held[mesh.enode[pres_el].ravel()] = True
    for ri, nodes in enumerate(bc.regions):
        touch = nodes[mesh.enode].any(axis=1)
        if not touch.any():
            continue
        if DRAWN_BEARS and not (nodes & ~drawn_held).any() and _region_kind(study, ri) == "keepOut":
            # a pin bearing on drawn material only (a boss round the bore):
            # the drawn cells are pinned already and carry it; a strap round
            # it only ran on past the boss (_region_nodes)
            continue
        ext = np.ptp(mesh.elem_ijk[touch], axis=0) + 1
        thin = int(np.argmin(ext))
        face = ext[thin] <= 2 and np.sort(ext)[1] * pitch > 4.0 * radius
        # A loaded face is pinned however much of the part it is: a deck
        # under a spread load. Capped like the rest, the 3D bridge benchmark
        # (a load over its whole top, 6% of the cells) left deck nodes on
        # void and its compliance went up a million-fold.
        if touch.sum() > _ANCHOR_MAX_FRACTION * mesh.n_elem and not (face and ri >= len(study.supports)):
            if ri < len(study.supports) or forced is None:
                continue
            # A loaded region that is neither small nor a flat face: a shaft
            # bearing on a saddle, a tool in a clip's jaws, which on a small
            # part line more than that share of its cells (surface grows as
            # the square of a part's size, cells as the cube). A force acts
            # on material or not at all: left to the optimizer, the nodes
            # that carry it sat on void once the projection sharpened (the
            # eroded design's edge ends short of every boundary), the
            # compliance it reads went up a hundredfold between its first
            # and last stage, and what it handed on was a design of that
            # reading: ragged walls and spikes reaching for loaded nodes (a
            # rod holder at 40,000 cells, 2026-11-21). So the cells under
            # the nodes that carry force in some case are pinned, to the
            # minimum member's thickness as a loaded face is. Only those: a
            # bearing presses on the half of the bore it pushes, and the
            # other half is the optimizer's.
            # (a node already on pinned material, a drawn solid's own, has
            # what it needs: nothing is added round a preserved region)
            held = np.zeros(mesh.n_nodes, bool)
            held[mesh.enode[pres_el].ravel()] = True
            touch = (nodes & forced & ~held)[mesh.enode].any(axis=1)
            if not touch.any():
                continue
            thick = max(float(study.min_member or 0.0), pitch)
            shape = _region_shape_of(study, ri)
            if shape is not None:
                touch = strap(touch, shape, thick)
            else:
                layers = int(np.ceil(thick / pitch))
                if layers > 1:
                    g = np.zeros(dims, bool)
                    g[tuple(mesh.elem_ijk[touch].T)] = True
                    g = ndimage.binary_dilation(g, structure=ndimage.generate_binary_structure(3, 1),
                                                iterations=layers - 1) & active_grid
                    touch = g[tuple(mesh.elem_ijk.T)]
            pres_el = pres_el | touch
            continue
        if face:
            if ri < len(study.supports):
                continue
            layers = int(np.ceil(float(study.min_member or 0.0) / pitch))
            if layers > ext[thin]:
                g = np.zeros(dims, bool)
                g[tuple(mesh.elem_ijk[touch].T)] = True
                step = np.zeros((3, 3, 3), bool)
                step[1, 1, 1] = True
                step[tuple(np.where(np.arange(3) == thin, [0, 0, 0], 1))] = True
                step[tuple(np.where(np.arange(3) == thin, [2, 2, 2], 1))] = True
                g = ndimage.binary_dilation(g, structure=step, iterations=layers - int(ext[thin])) & active_grid
                touch = g[tuple(mesh.elem_ijk.T)]
        elif ri >= len(study.supports) and float(study.min_member or 0.0) > pitch:
            # A small loaded region lining a keep-out (a bore on its pin):
            # one cell of pinned wall is thinner than anything the mesh
            # keeps, so the analysis had a ring round the pin and the part
            # did not (a fork's axle bores came out open under the axle).
            # A wall that carries the load is a member: pinned to the
            # minimum member's thickness, as the larger regions above are.
            shape = _region_shape_of(study, ri)
            if shape is not None:
                touch = strap(touch, shape, float(study.min_member))
        pres_el = pres_el | touch

    found = _found_symmetry(study, mesh, dims, pres_el, bc, loads, weights, spreads)
    mirror = {}
    for axis in list(study.symmetry) + found:
        a = "xyz".index(axis)
        m_ijk = mesh.elem_ijk.copy()
        m_ijk[:, a] = dims[a] - 1 - m_ijk[:, a]
        lut = np.full(dims, -1, np.int64)
        lut[tuple(mesh.elem_ijk.T)] = np.arange(mesh.n_elem)
        mirror[axis] = lut[tuple(m_ijk.T)]
        unmatched = float((mirror[axis] < 0).mean())
        if unmatched > 0.02:
            warnings.append(f"symmetry '{axis}': {unmatched:.0%} of cells have no mirror; the design space, "
                            "preserve or keep-outs are not symmetric about that plane")

    from .thermal import study_heat

    if found:
        study.found_symmetry = list(found)
    hint = level_hint if np.isfinite(level_hint).any() else None
    dom = Domain(pitch=pitch, origin=origin, dims=tuple(int(d) for d in dims), mesh=mesh,
                  preserve=pres_el, fixed=fixed, loads=loads, case_names=names, case_weights=weights,
                  support_nodes=support_nodes, load_nodes=load_nodes, mirror=mirror, warnings=warnings,
                  springs=bc.springs, seats=bc.seats, gravity=bc.gravity, spreads=spreads, inertial_cases=inertial,
                  inserts=list(getattr(bc, "inserts", None) or []),
                  volume_weights=_wall_weights(_volume_weights(study, mesh, pres_el), mesh, wall_el),
                  center_of_mass=(None if getattr(getattr(study, "mass_properties", None), "com_target", None) is None
                                  else (study.mass_properties.com_target, study.mass_properties.com_tolerance)),
                  raw_loads=list(bc.raw_loads) if bc.raw_loads else None,
                  carried=carried_volume(study, mesh) if bc.raw_loads else None,
                  bodies=carried_bodies(study, mesh, _rbm(mesh.node_xyz)) if bc.raw_loads else [],
                  thermal=study_heat(study, mesh, study.elastic(design=True), fixed,
                                     near=_near_cells(mesh, support_nodes | load_nodes)),
                  warp=_warp_term(study, mesh), joints=joints, level_hint=hint,
                  pin_springs=getattr(bc, "pin_springs", None))
    aero = getattr(study, "aero", None)
    if aero is not None and aero.in_optimizer:
        from .aero import BlockageCost

        dom.blockage = BlockageCost(dom, aero.rotors, float(study.material.density), aero.grams_per_newton)
    from .castfeed import study_feed

    dom.feeding = study_feed(study, mesh, dom.dims, pitch)
    from .modalopt import modal_model

    dom.modal = modal_model(study, dom)
    dom.impacts = [(ci, c) for ci, c in enumerate(study.load_cases) if getattr(c, "impact", None) is not None]
    dom.impact_material = (float(study.material.E), float(study.material.density))
    dom.impact_study = study
    if wall_el is not None:
        # thin walls as cut cells (thinwall.py): their cells, and those of them the optimizer may not fill
        dom.walls = wall_el
        dom.wall_void = wall_void & ~dom.preserve
    if stock_el is not None:
        # bought members' cells, and those of them on stretches that are not joined (reconstruct leaves
        # these out of the printed part)
        dom.stock = stock_el
        dom.stock_free = free[active] & stock_el
    return dom


def _wall_weights(vw, mesh: HexMesh, wall_el):
    """Volume weights with a thin wall's cells costing the rest of the cell
    the wall leaves (thinwall.py): the optimizer's density there is the
    share of that rest it fills."""
    if wall_el is None:
        return vw
    share = getattr(mesh, "wall_share", None)
    w = np.where(wall_el, 1.0 - share, 1.0)
    return w if vw is None else vw * w


def _volume_weights(study: Study, mesh: HexMesh, preserve: np.ndarray):
    """massProperties.inertiaWeight w: a cell's cost (1 - w) + w r^2 / mean
    r^2, r its centre's distance from the axis, the mean over the design
    cells, so the budget the search sets keeps its meaning (a fraction of
    the design space) and a cell on the axis costs 1 - w of one far out."""
    mp = getattr(study, "mass_properties", None)
    if mp is None or not mp.weight or mp.through is None:
        return None
    c = mesh.origin + (mesh.elem_ijk + 0.5) * mesh.pitch - mp.through[None, :]
    r2 = np.einsum("ij,ij->i", c, c) - (c @ mp.direction) ** 2
    design = ~np.asarray(preserve, bool)
    mean = float(r2[design].mean()) if design.any() else float(r2.mean())
    return (1.0 - mp.weight) + mp.weight * r2 / max(mean, 1e-12)


def _inertial_cases(study: Study, mesh: HexMesh, fixed: np.ndarray, loads: list) -> list:
    """The vibration block's carried masses shaken along x, y and z, as
    stiffness-only load cases: (name, force per dof). A part that is stiff
    against its masses' inertia has a high first frequency (the Rayleigh
    quotient's numerator is the strain energy of the mode the masses
    drive), so these steer the compliance design towards it; the delivered
    part's frequency is then read by a modal check (verify._modal) and the
    search holds it. The acceleration is scaled so the masses' total force
    matches the largest real load case: the objective weighs them alongside
    the real cases rather than letting either swamp the other."""
    vib = getattr(study, "vibration", None)
    if vib is None or not vib.in_optimizer or not vib.masses or study.inertia_relief:
        # a free part shaken as a whole only translates: these cases say
        # nothing about it. Its modes are held directly (modalopt.py).
        return []
    total = sum(kg for _, _, kg in vib.masses)
    real = max((float(np.linalg.norm(f.reshape(-1, 3).sum(axis=0))) for f in loads), default=0.0)
    scale = real / total if real > 0 else 9.81
    out = []
    for a in range(3):
        f = np.zeros(mesh.n_dof)
        for i, (region, shape, kg) in enumerate(vib.masses):
            nodes = np.flatnonzero(_region_nodes(study, mesh, region, shape, f"vibration.masses[{i}]"))
            f.reshape(-1, 3)[nodes, a] += kg * scale / len(nodes)
        f[fixed] = 0.0
        if np.any(f):
            out.append((f"inertia-{'xyz'[a]}", f))
    return out


def _warp_term(study, mesh):
    """buildSim.warpWeight > 0 or a warp limit (maxDistortion,
    maxRecoaterLift): the as-built build model the optimizer carries
    (buildsim.WarpTerm, on a coarser grid when the mesh is big) and what it
    does with it (WarpSpec)."""
    bs = getattr(study, "build_sim", None)
    if not bs or not (bs.get("warpWeight") or bs.get("maxDistortion") or bs.get("maxRecoaterLift")):
        return None
    if bs["setup"].process.lower() == "fdm" and not bs.get("warpWeight") and not bs.get("warpInOptimizer"):
        # FDM limits are checked on the delivered part (study.py); pushing
        # them through the optimizer cost the load path and did not lower
        # the warp (docs/BUILDSIM.md)
        return None
    from .buildsim import CoarseWarp, WarpTerm
    nu = float(study.elastic(design=True))
    f = bs.get("warpCoarsen") or (2 if mesh.n_elem >= 4000 else 1)
    layer = bs.get("warpLayer")
    if f > 1:
        term = CoarseWarp(mesh, bs["setup"], f, build=study.process.build, layer=layer, nu=nu)
    else:
        term = WarpTerm(mesh, bs["setup"], build=study.process.build, layer=layer, nu=nu)
    return WarpSpec(term, float(bs.get("warpWeight") or 0.0), bs.get("maxDistortion"), bs.get("maxRecoaterLift"),
                    f)


def _reach_bridged(env, direction, pitch, process, bridge_span=None):
    """Cells of `env` (bool, grid shape) a print along `direction` (axis,
    sign) reaches from the bed: each layer stands on the 45-degree cross of
    the layer below, and what is left of a layer prints if it is a bridge
    or a ledge by printcheck's reading (spans no longer than the bridge
    span, landing on reached cells of the same layer at both ends)."""
    from scipy import ndimage

    from .printcheck import _CROSS, _EIGHT, _LEDGE, _spans, BRIDGE_SPAN
    from .edtops import edt

    axis, sign = direction
    g = np.moveaxis(np.asarray(env, bool), axis, 0)
    g = g if sign > 0 else g[::-1]
    span_mm = float(bridge_span if bridge_span is not None else BRIDGE_SPAN.get(process, 15.0))
    max_steps = int(np.ceil(span_mm / pitch)) + 2
    ledge_cells = _LEDGE / pitch
    out = np.zeros_like(g)
    out[0] = g[0]
    for k in range(1, g.shape[0]):
        lay = g[k] & ndimage.binary_dilation(out[k - 1], _CROSS)
        free = g[k] & ~lay
        if free.any() and lay.any():
            lbl, n = ndimage.label(free, _EIGHT)
            dist = None
            for r in range(1, n + 1):
                reg = lbl == r
                if not (ndimage.binary_dilation(reg, _EIGHT) & lay).any():
                    continue
                if dist is None:
                    dist = edt(~lay)
                if dist[reg].max() <= ledge_cells + 1.0:
                    lay = lay | reg
                    continue
                frac, span = _spans(reg, lay, pitch, max_steps)
                if frac >= 0.97 and span <= span_mm:
                    lay = lay | reg
        out[k] = lay
    out = out if sign > 0 else out[::-1]
    return np.moveaxis(out, 0, axis)


@dataclass
class WarpSpec:
    term: object                  # buildsim.WarpTerm or CoarseWarp: measures(e, de, weights)
    weight: float                 # buildSim.warpWeight
    max_distortion: float | None  # mm limits, held as constraints
    max_lift: float | None
    coarsen: int = 1


def _unprintable_preserve(direction, dims, active, preserve, pitch, origin, process=None,
                          bridge_span=None) -> list:
    """Drawn geometry the print cannot reach from its bed. The overhang rule
    holds preserved cells solid whatever is under them (they are the
    user's), so a feature hanging over air in the build direction is never
    the optimizer's problem and only shows as support on the finished part;
    and a design space whose floor stands off the bed (a hinge knuckle below
    a clamp's split face set the bed, 7 mm under everything else) grows
    nothing at all (an iPhone mount read safety factor 0.003 for it). Here
    the whole envelope is printed solid along the build direction with no
    cell held: whatever cannot be reached at 45 degrees even then is said,
    before an optimization is spent on it.

    A printer also bridges: a cell the 45-degree reach misses but that a
    straight span of at most the process's bridge span holds at both ends
    (the roof of a hole lying across the layers, a 4.5 mm screw hole in a
    bracket printed on its side) prints as a slicer prints it, and is read
    as printcheck.check_print reads the finished part (a narrow ledge off
    a wall too). Without that, every horizontal hole's roof was called
    preserved geometry needing support."""
    from .am import AMFilter, direction_label

    try:
        env = active.reshape(dims).astype(float)
        printed = AMFilter(tuple(dims), direction).forward(env).reshape(-1)
    except Exception:  # noqa: BLE001 - a check, never a reason not to build
        return []
    cant = active & (printed < 0.5)
    if (cant & preserve).any() and process is not None and len(direction) == 2:
        try:
            cant = active & ~_reach_bridged(active.reshape(dims), direction, pitch, process, bridge_span).reshape(-1)
        except Exception:  # noqa: BLE001 - a check, never a reason not to build
            pass
    pres_cant = cant & preserve
    n_act = int(active.sum())
    if not pres_cant.any() or n_act == 0:
        return []
    cell = pitch ** 3
    idx = np.argwhere(pres_cant.reshape(dims))
    lo = np.asarray(origin, float) + idx.min(axis=0) * pitch
    hi = np.asarray(origin, float) + (idx.max(axis=0) + 1) * pitch
    share = cant.sum() / n_act
    try:
        lab = direction_label(direction)
    except Exception:  # noqa: BLE001
        lab = str(direction)
    msg = (f"printed along {lab}, {pres_cant.sum() * cell:.0f} mm^3 of preserved geometry (in "
           f"[{', '.join(f'{v:.4g}' for v in lo)}] to [{', '.join(f'{v:.4g}' for v in hi)}]) cannot be reached "
           "from the bed at 45 degrees even with the whole design space solid: it will print on supports. "
           "Draw it to stand on the bed or on itself, give the design space room under it, or choose another "
           "build direction")
    if share > 0.25:
        msg += (f"; {100 * share:.0f}% of the design space cannot be reached either, so the overhang rule will "
                "leave the part disconnected (the bed is the lowest cell of the domain: does a preserved feature "
                "hang below the rest of the part?)")
    return [msg]
