"""
Cooling: design the inside of a fixed exterior around the heat its
electronics make.

An industrial designer hands over the exterior; the parts inside are known
(a processor, a radio, a charger chip, a battery) and each one makes heat
and has a temperature it must stay under. The exterior has its own limit:
the surface a hand holds must not get hotter than is comfortable or safe to
touch. Loadpath grows the inside (ribs and pads that conduct heat from the
hot parts out to the skin, spread it so no patch of the skin gets too hot,
and nothing more), with the least material, together with whatever
structural targets the same study carries (a click force, a grip, a drop:
switch.py, feel.py).

The model, per grid cell, inside the exterior:

- the printed material (the designed interior and the skin) conducts at its
  card's conductivity (thermal.CONDUCTIVITY), k(x) = k_air + x^3 (k - k_air),
  so a void cell is air, not nothing: a polymer conducts only 6-8 times
  better than air, and the air in the gaps carries a real share of the heat;
- the enclosed air conducts at `air.conductivity` (default K_AIR). In a
  hand-held product the gaps are a few mm, and a Rayleigh number across
  a 5 mm gap at a 10 K difference is about 100, far under the ~1700 where
  natural convection starts in an enclosed layer: the air inside is close
  to still, and conduction is what moves heat through it;
- each component is a block of its own conductivity (default
  COMPONENT_K: a package on a copper-filled board spreads heat well in
  plane) and gives off its `watts` over its volume; components listed with
  no watts (a battery, a board) only conduct;
- the outside of the exterior loses heat to the room by natural convection
  and radiation together, h (default H_OUT: about 5 W/(m^2 K) of each for
  a matte surface a few tens of K over room temperature);
- vents, if the exterior has them (`vents`: openings the designer drew,
  each a shape on the skin; loadpath never cuts the exterior), draw room
  air through by buoyancy when one is higher than another: the stack flow
  Q = Cd A sqrt(2 g H dT / T) carries rho c_p Q dT out, with dT the mean
  rise of the air inside. This is a well-mixed estimate (the air inside is
  taken as one temperature for the flow, and the heat it carries comes off
  every air cell in proportion to its air): good for a yes or no and a
  size, not for where the draft goes.

The limits are temperature limits: each component's hottest point under its
`maxTemp`, and the hottest point of the touched skin under `skin.maxTemp`
(or a contact class, TOUCH_LIMITS). Both are read as a p-norm of the node
temperatures (p = P_NORM) scaled to the true maximum of the last solve (Le
et al. 2010), so the gradient is smooth and the limit is the hot spot,
not an average. They go to switch.optimize_switch as constraints (min
volume, MMA, thin-skin cut cells, coarse to fine), so one study can carry a
click force, a grip feel and its cooling together. Each costs one adjoint
solve per iteration on a scalar system (CG with AMG), cheap next to the
elastic solves.

Two things the limits trade against each other: material from a hot chip
straight to the nearest wall cools the chip and makes a hot spot on the
wall over it; spread over more of the inside of the skin, it cools the chip
a little less and keeps every patch of the skin cooler. The optimizer finds
the balance the two limits ask for.

After the run the design is read again black and white, and the delivered
mesh on two finer grids (the delivered mesh's own fill, solid where a cell
is at least half inside it), so a reading is quoted only where the two
grids agree.

Forced air over the outside (flight, a rotor's wash, a fan blowing on it)
is `skin.airflow: {speed, length}`: h from the flat plate's correlation
(forced_h). Not modelled: contact resistance between a component and a printed pad
resting on it (a thermal pad or paste is assumed), forced air through the inside,
the hand itself as a heat sink, and sunlight. The structure's own
softening with temperature (thermal.retention) is in thermal.py's
`thermal` block, not here.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .domain import Domain
from .fem import HexMesh
from .thermal import _FACE_M, hex8_kt

__all__ = ["forced_h", "CoolingProblem", "CoolingModel", "cooling_readings", "run_cooling", "TOUCH_LIMITS", "K_AIR", "H_OUT",
           "COMPONENT_K", "stack_conductance"]

K_AIR = 0.03            # W/(m K): air near 40 C is 0.027; a little for what convection there is
H_OUT = 10.0            # W/(m^2 K): natural convection ~5 + radiation ~5-6 (emissivity 0.9, 30-50 C)
COMPONENT_K = 20.0
# J/(mm^3 K) a component holds per K and volume when it gives no
# heatCapacity: rho c_p of silicon 1.66, FR-4 1.6, a lithium pouch about 2.0
# (2300 kg/m^3 x 900 J/(kg K)), copper 3.4 (MJ/(m^3 K))
VOL_CAP = 1.8e-3
RHO_CP_AIR_VOL = 1.2e-6       # J/(mm^3 K), still air      # W/(m K): a package on a copper-filled board, in plane
P_NORM = 12.0
CD = 0.6                # discharge coefficient of a sharp-edged vent
RHO_CP_AIR = 1.16 * 1007.0    # J/(m^3 K) at ~30 C
G = 9.81
# Nodes up to which the system is factored (PARDISO); a finer read-back
# (millions of nodes) is solved with CG and AMG instead, in a fraction of
# the memory.
DIRECT_MAX = 600_000

try:
    import pypardiso  # noqa: F401
    _HAS_PARDISO = True
except ImportError:          # CG with AMG (thermal.ScalarSolver)
    _HAS_PARDISO = False

# Hottest a surface may be where it is touched, C, for plastic surfaces,
# after IEC 62368-1 Table 38 (TS1); metal limits are lower. Check the
# edition the product is certified to.
TOUCH_LIMITS = {
    "continuous": 43.0,   # held or touched for more than 8 h
    "held": 48.0,         # held for 10 min to 8 h (a handheld product in use)
    "brief": 60.0,        # touched for 1 to 10 min
}


# ---------------------------------------------------------------- the spec


@dataclass
class Source:
    id: str
    watts: float
    max_t: float | None
    k: float
    shape: object
    capacity: float | None = None     # J/K (None: VOL_CAP times its volume)
    cells: np.ndarray = None          # thermal cell indices
    nodes: np.ndarray = None          # bool per thermal node


@dataclass
class Vent:
    id: str
    shape: object
    area: float | None                # mm^2; None: from the faces it covers
    faces: np.ndarray = None


@dataclass
class CoolingSpec:
    ambient: float
    h_out: float
    k_air: float
    k_material: float
    sources: list
    skin_max: float | None
    skin_shape: object                # where the touch limit applies (None: everywhere)
    vents: list
    up: np.ndarray                    # unit vector against gravity
    raw: dict = field(default_factory=dict)
    cycle: "Cycle | None" = None


@dataclass
class Cycle:
    """Power over time: phases of (seconds, steps, watts per source), from
    `start` ("first": the steady state of the first phase's power;
    "ambient": everything at room temperature), or with `repeat` the
    repeating state, run `cycles` times from the steady state at the mean
    power and read on the last."""
    phases: list                      # [(seconds, steps, np.array watts per source)]
    start: str = "first"
    repeat: bool = False
    cycles: int = 3

    def steps(self):
        """[(dt, watts)] for every step of the run, and the first step
        the limits read (the last cycle's, with repeat)."""
        one = [(sec / n, w) for sec, n, w in self.phases for _ in range(n)]
        if not self.repeat:
            return one, 0
        return one * self.cycles, len(one) * (self.cycles - 1)

    def start_watts(self):
        if self.repeat:
            t = sum(sec for sec, _, _ in self.phases)
            return sum(sec * w for sec, _, w in self.phases) / t
        return None if self.start == "ambient" else self.phases[0][2]


def _num(v, label, positive=True):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or (positive and v <= 0):
        raise ValueError(f"{label} must be a {'positive ' if positive else ''}number")
    return float(v)


def _component_shape(study, item, label):
    """A component's own shape: an inline `shape`, or the keep-out (or
    preserve) `region` it is, as drawn (a keep-out with an assembly path is
    swept along it in the study; the part itself is only where it sits)."""
    from .spec import shape_from_spec
    if "shape" in item:
        return shape_from_spec(item["shape"], study.base_dir)
    if "region" not in item and "id" not in item:
        raise ValueError(f"{label} needs a `region` (a keepOut id) or a `shape`")
    rid = str(item.get("region") or item.get("id"))
    for key in ("keepOut", "preserve"):
        for r in study.raw.get(key) or []:
            if isinstance(r, dict) and str(r.get("id")) == rid and "shape" in r:
                return shape_from_spec(r["shape"], study.base_dir)
    return study.region_shape(rid)[0]


def cooling_from_study(study) -> CoolingSpec:
    from .spec import shape_from_spec
    from .thermal import conductivity
    cb = study.raw.get("cooling")
    if not isinstance(cb, dict):
        raise ValueError("the study needs a `cooling` block")
    ambient = float(cb.get("ambient", 25.0))
    k_mat = cb.get("conductivity")
    k_mat = _num(k_mat, "cooling.conductivity") if k_mat is not None else conductivity(study.material)
    air = cb.get("air") or {}
    k_air = _num(air.get("conductivity", K_AIR), "cooling.air.conductivity")
    sources = []
    for i, c in enumerate(cb.get("components") or []):
        label = f"cooling.components[{i}]"
        if not isinstance(c, dict):
            raise ValueError(f"{label} is {{region | shape, watts?, maxTemp?, conductivity?}}")
        w = float(c.get("watts", 0.0))
        if w < 0:
            raise ValueError(f"{label}.watts is the heat it gives off, W (>= 0)")
        mt = c.get("maxTemp")
        if mt is not None:
            mt = float(mt)
            if mt <= ambient:
                raise ValueError(f"{label}.maxTemp {mt:g} C is not over the ambient {ambient:g} C")
        hc = c.get("heatCapacity")
        sources.append(Source(str(c.get("region") or c.get("id") or f"component-{i}"), w, mt,
                              _num(c.get("conductivity", COMPONENT_K), f"{label}.conductivity"),
                              _component_shape(study, c, label),
                              _num(hc, f"{label}.heatCapacity") if hc is not None else None))
    if not sources or not any(s.watts > 0 for s in sources):
        raise ValueError("cooling.components: at least one component with watts > 0")
    sk = cb.get("skin") or {}
    smax = sk.get("maxTemp", sk.get("contact"))
    if isinstance(smax, str):
        if smax not in TOUCH_LIMITS:
            raise ValueError(f"cooling.skin.contact is one of {sorted(TOUCH_LIMITS)} or a maxTemp in C")
        smax = TOUCH_LIMITS[smax]
    elif smax is not None:
        smax = float(smax)
        if smax <= ambient:
            raise ValueError(f"cooling.skin.maxTemp {smax:g} C is not over the ambient {ambient:g} C")
    h = _num(sk.get("h", cb.get("h", H_OUT)), "cooling.skin.h")
    if sk.get("airflow") is not None:
        if sk.get("h") is not None or cb.get("h") is not None:
            raise ValueError("cooling.skin: give h or airflow, not both (airflow works h out)")
        af = sk["airflow"]
        if not isinstance(af, dict) or "speed" not in af or "length" not in af:
            raise ValueError("cooling.skin.airflow is {speed (m/s), length (mm, the run of the surface along the "
                             "flow)}: the stream over the outside (flight speed, a fan's or a rotor's wash)")
        h = forced_h(_num(af["speed"], "cooling.skin.airflow.speed"), _num(af["length"], "cooling.skin.airflow.length"),
                     ambient)["h"]
    skin_shape = shape_from_spec(sk["shape"], study.base_dir) if sk.get("shape") else None
    vents = []
    for i, v in enumerate(cb.get("vents") or []):
        if not isinstance(v, dict) or "shape" not in v:
            raise ValueError(f"cooling.vents[{i}] is {{shape, area?}}: an opening the exterior already has")
        vents.append(Vent(str(v.get("id") or f"vent-{i}"), shape_from_spec(v["shape"], study.base_dir),
                          _num(v["area"], f"cooling.vents[{i}].area") if v.get("area") is not None else None))
    from .spec import _leg_direction
    up = -_leg_direction(cb.get("gravity", "-z"), "cooling.gravity")
    if not any(s.max_t is not None for s in sources) and smax is None:
        raise ValueError("cooling: nothing to hold: give a component maxTemp or skin.maxTemp")
    cyc = _cycle_from(cb.get("cycle"), sources) if cb.get("cycle") is not None else None
    return CoolingSpec(ambient, h, k_air, k_mat, sources, smax, skin_shape, vents, up, cb, cyc)


def forced_h(speed: float, length: float, ambient: float = 25.0, rise: float = 20.0, emissivity: float = 0.9) -> dict:
    """Heat transfer off an outside surface in a stream (W/(m^2 K)): air
    at `speed` m/s along a run of `length` mm, the surface `rise` K over
    `ambient` C. Forced convection is the flat plate's mean (laminar Nu =
    0.664 Re^1/2 Pr^1/3; mixed with a turbulent run past Re 5e5, Nu =
    (0.037 Re^0.8 - 871) Pr^1/3), combined with the natural convection
    still there at low speed as (Nu_f^3 + Nu_n^3)^1/3 (Churchill), plus
    radiation 4 e sigma T^3. A canopy 60 mm long at 8 m/s: about 46 + 6,
    five times still air's 10."""
    t_film = ambient + 0.5 * rise + 273.15
    k = 0.0241 * (t_film / 273.15) ** 0.81                     # W/(m K), air
    nu = 1.33e-5 * (t_film / 273.15) ** 1.75                   # m^2/s
    pr = 0.71
    ell = max(float(length), 1e-3) * 1e-3
    re = max(float(speed), 0.0) * ell / nu
    if re < 5e5:
        nu_f = 0.664 * re ** 0.5 * pr ** (1 / 3)
    else:
        nu_f = (0.037 * re ** 0.8 - 871.0) * pr ** (1 / 3)
    # natural convection off the same run (vertical plate, Churchill-Chu laminar)
    g, beta = 9.81, 1.0 / t_film
    ra = g * beta * max(rise, 0.1) * ell ** 3 / (nu * nu / pr)
    nu_n = 0.68 + 0.670 * ra ** 0.25 / (1 + (0.492 / pr) ** (9 / 16)) ** (4 / 9)
    h_conv = (nu_f ** 3 + nu_n ** 3) ** (1 / 3) * k / ell
    ts, ta = t_film + 0.5 * rise, ambient + 273.15
    h_rad = emissivity * 5.670e-8 * (ts ** 2 + ta ** 2) * (ts + ta)
    return {"h": h_conv + h_rad, "convection": h_conv, "radiation": h_rad, "re": re, "forced": nu_f * k / ell,
            "natural": nu_n * k / ell}


def _cycle_from(cb, sources) -> Cycle:
    if not isinstance(cb, dict) or not isinstance(cb.get("phases"), list) or not cb["phases"]:
        raise ValueError("cooling.cycle is {phases: [{seconds, watts: {component: W}, steps?}], repeat?, start?}")
    ids = [s.id for s in sources]
    steps = int(cb.get("steps", 12))
    phases = []
    for i, ph in enumerate(cb["phases"]):
        label = f"cooling.cycle.phases[{i}]"
        if not isinstance(ph, dict):
            raise ValueError(f"{label} is {{seconds, watts: {{component: W}}, steps?}}")
        sec = _num(ph.get("seconds"), f"{label}.seconds")
        w = np.array([s.watts for s in sources], float)
        for cid, v in (ph.get("watts") or {}).items():
            if cid not in ids:
                raise ValueError(f"{label}.watts: no cooling component {cid!r} (have {ids})")
            w[ids.index(cid)] = _num(v, f"{label}.watts.{cid}", positive=False)
        if (w < 0).any():
            raise ValueError(f"{label}.watts must be >= 0")
        n = int(ph.get("steps", steps))
        if n < 1:
            raise ValueError(f"{label}.steps must be >= 1")
        phases.append((sec, n, w))
    start = str(cb.get("start", "first"))
    if start not in ("first", "ambient"):
        raise ValueError("cooling.cycle.start is 'first' (the first phase's steady state) or 'ambient'")
    cycles = int(cb.get("cycles", 3))
    if cycles < 1:
        raise ValueError("cooling.cycle.cycles must be >= 1")
    return Cycle(phases, start, bool(cb.get("repeat", False)), cycles)


def stack_conductance(vents_area_height, d_t: float, ambient: float) -> tuple[float, float]:
    """(W/K carried out, m^3/s) by buoyant flow through vents at different
    heights: [(area mm^2, height mm)], the inside air d_t K over the room.
    The lowest vents are the inlet and the highest the outlet (split at the
    mean height), in series: A = 1 / sqrt(1/A_in^2 + 1/A_out^2)."""
    if len(vents_area_height) < 2 or d_t <= 0:
        return 0.0, 0.0
    hs = np.array([h for _, h in vents_area_height], float)
    a = np.array([a for a, _ in vents_area_height], float) * 1e-6
    mid = 0.5 * (hs.min() + hs.max())
    lo, hi = hs <= mid, hs > mid
    if not lo.any() or not hi.any():
        return 0.0, 0.0
    a_in, a_out = a[lo].sum(), a[hi].sum()
    h_in = float((a[lo] * hs[lo]).sum() / a_in)
    h_out = float((a[hi] * hs[hi]).sum() / a_out)
    stack = (h_out - h_in) * 1e-3
    a_eff = 1.0 / np.sqrt(1.0 / a_in ** 2 + 1.0 / a_out ** 2)
    q = CD * a_eff * np.sqrt(2 * G * stack * d_t / (ambient + 273.15))
    return float(RHO_CP_AIR * q), float(q)


# --------------------------------------------------------------- the model


def _true_area_weights(study, centers, pitch):
    """Per outside grid face, the share of its area the exterior's own
    surface has there: 1 / (|nx| + |ny| + |nz|) with n the exterior's unit
    normal (from its distance field), so the faces of a stepped surface add
    up to the area of the surface they follow. 1 on faces that line up
    with the grid, 1/sqrt(3) at most on a surface at 45 degrees to all
    three axes."""
    shapes = list(study.design_space or [])
    if not len(centers) or not shapes:
        return np.ones(len(centers))

    def sdf(p):
        return np.min([sh.sdf(p) for sh in shapes], axis=0)
    e = 0.25 * pitch
    g = np.stack([(sdf(centers + e * d) - sdf(centers - e * d)) / (2 * e) for d in np.eye(3)], axis=1)
    norm = np.linalg.norm(g, axis=1)
    ok = np.isfinite(norm) & (norm > 0.2)
    n = np.where(ok[:, None], g / np.where(ok, norm, 1.0)[:, None], 0.0)
    l1 = np.abs(n).sum(axis=1)
    return np.where(ok, 1.0 / np.clip(l1, 1.0, np.sqrt(3.0)), 1.0)


def _gap_between_sources(sources, c, k, pitch: float, k_air: float) -> None:
    """Air between two components, under a cell wide. Two components 0.4 mm
    apart (the speaker's amp and its cell) fill face-adjacent cells on most
    grids, and those cells conducted at the components' own 20 W/(m K)
    straight across what is air: the amp's heat ran into the cell on the
    1.05 and 0.6 mm grids and not on the 0.75 and 0.5 mm ones. Each
    component's cells within a cell of another conduct in series with half
    the air gap between the two shapes, so the pair reads the gap's
    resistance on any grid. Components that touch keep their contact."""
    if len(sources) < 2:
        return
    for s in sources:
        for t in sources:
            if t is s:
                continue
            near = t.shape.sdf(c[s.cells]) < pitch
            if not near.any():
                continue
            g = _shape_gap(s.shape, t.shape, pitch)
            if g <= 0.02:
                continue
            half = min(0.5 * g, pitch)
            sel = s.cells[near]
            k[sel] = np.minimum(k[sel], pitch / (half / k_air + max(pitch - half, 0.0) / s.k))


def _shape_gap(a, b, pitch: float) -> float:
    """The narrowest gap between two shapes where they come within a cell
    of each other: a's surface sampled at a quarter cell (0.25 mm at most)
    over the box where the two are near."""
    lo_a, hi_a = a.bounds()
    lo_b, hi_b = b.bounds()
    lo = np.maximum(lo_a, lo_b - pitch) - 1e-6
    hi = np.minimum(hi_a, hi_b + pitch) + 1e-6
    if np.any(hi < lo):
        return float("inf")
    h = min(0.25, 0.25 * pitch)
    n = np.clip(np.ceil((hi - lo) / h).astype(int) + 1, 2, 400)
    best = float("inf")
    xs = [np.linspace(lo[i], hi[i], n[i]) for i in range(3)]
    for x in xs[0]:
        P = np.stack(np.meshgrid([x], xs[1], xs[2], indexing="ij"), -1).reshape(-1, 3)
        on = np.abs(a.sdf(P)) <= 0.6 * h
        if on.any():
            best = min(best, float(b.sdf(P[on]).min()))
    return best


class CoolingModel:
    """The conduction problem on one grid: every cell inside the exterior
    (the designed cells of `domain`, the components, the air in the
    keep-outs), with the mechanical design's densities mapped in."""

    def __init__(self, study, domain: Domain, spec: CoolingSpec | None = None, floor=None):
        from .domain import _fraction
        from .thermal import Faces, ScalarSolver, thermal_bc
        self.spec = spec = spec or cooling_from_study(study)
        self.domain = domain
        pitch, dims, origin = domain.pitch, tuple(domain.dims), np.asarray(domain.origin, float)
        ii, jj, kk = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
        ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
        centers = origin + (ijk + 0.5) * pitch
        inside = _fraction(study.design_space, centers, pitch) >= 0.5
        flat = np.ravel_multi_index(tuple(domain.mesh.elem_ijk.T), dims)
        inside[flat] = True
        for s in spec.sources:
            inside |= s.shape.sdf(centers) <= 0
        mesh = HexMesh(ijk[inside], pitch, origin)
        self.mesh = mesh
        lut = np.full(int(np.prod(dims)), -1, np.int64)
        lut[np.flatnonzero(inside)] = np.arange(mesh.n_elem)
        self.mech = lut[flat]                           # thermal cell of each mechanical cell
        is_mech = np.zeros(mesh.n_elem, bool)
        is_mech[self.mech] = True
        self.is_mech = is_mech
        c = mesh.elem_centers()
        k0 = np.full(mesh.n_elem, spec.k_air)
        src = np.zeros(mesh.n_elem, bool)
        heat = []
        for s in spec.sources:
            sd = s.shape.sdf(c)
            cells = np.flatnonzero((sd <= 0) & ~is_mech)
            if not len(cells):
                # thinner than a cell: the cells it passes through (a chip
                # 1.2 mm thick on a 2 mm grid), none of them designed
                cells = np.flatnonzero((sd <= 0.5 * pitch) & ~is_mech)
            if not len(cells):
                raise ValueError(f"cooling component {s.id!r} covers no cell at {pitch:g} mm: refine the grid "
                                 "or give it its own shape")
            s.cells = cells
            k0[cells] = s.k
            src[cells] = True
            nodes = np.zeros(mesh.n_nodes, bool)
            nodes[mesh.enode[cells].ravel()] = True
            s.nodes = nodes
            if s.watts > 0:
                m = np.zeros(mesh.n_elem, bool)
                m[cells] = True
                heat.append((m, s.watts, "volume"))
        _gap_between_sources(spec.sources, c, k0, pitch, spec.k_air)
        # a component is read on its own nodes: the nodes it shares with
        # another component's cells sit on the gap between them, and read
        # the neighbour's heat (the speaker's cell read 49 C at a node
        # 0.4 mm off the amp while its body was at 34 C)
        for s in spec.sources:
            others = np.zeros(mesh.n_nodes, bool)
            for t in spec.sources:
                if t is not s:
                    others[mesh.enode[t.cells].ravel()] = True
            own = s.nodes & ~others
            if own.any():
                s.nodes = own
        self.k0 = k0
        self.src = src
        # heat capacity per thermal cell, J/K (designed cells set per design)
        vol = pitch ** 3
        c0 = np.full(mesh.n_elem, RHO_CP_AIR_VOL * vol)
        for s in spec.sources:
            c0[s.cells] = (s.capacity / len(s.cells)) if s.capacity is not None else VOL_CAP * vol
        self.c0 = c0
        from .thermal import specific_heat
        self.c_mat = float(study.material.density) * 1e-6 * specific_heat(study.material) * vol
        # node load per watt of each source (its heat spread over its cells)
        self.src_load = np.stack([np.bincount(mesh.enode[s.cells].ravel(),
                                              weights=np.full(8 * len(s.cells), 1.0 / (8 * len(s.cells))),
                                              minlength=mesh.n_nodes) for s in spec.sources], axis=1)
        faces = Faces(mesh)
        self.faces = faces
        self.bc = thermal_bc(mesh, ambient=spec.ambient, heat=heat, faces=faces)
        self.heat = self.bc.heat
        self._kt = hex8_kt() * pitch
        # the outside's convection, h (T - ambient) on every outside face: a (the outside is the exterior, whatever the design)
        # constant matrix (the outside is the exterior, whatever the design).
        # Each face convects over the true surface it stands for: a stepped
        # grid surface has |nx| + |ny| + |nz| times the area of the surface
        # it follows (1.5x on a sphere, which read its outside a third cool)
        bf = faces.nodes[faces.boundary]
        self.out_w = _true_area_weights(study, faces.center[faces.boundary], pitch)
        w = (spec.h_out * 1e-6 * faces.area) * self.out_w[:, None] * _FACE_M.ravel()[None]
        self._kc = sp.csr_matrix((w.ravel(), (np.repeat(bf, 4, axis=1).ravel(),
                                              np.tile(bf, (1, 4)).ravel())),
                                 shape=(mesh.n_nodes, mesh.n_nodes))
        self.solver = ScalarSolver(rtol=1e-9)
        self._ones = np.ones(mesh.n_elem)
        # air share of each cell (1 - x on designed cells, set per design)
        self.air0 = np.where(src | is_mech, 0.0, 1.0)
        # a thin skin's share of its cells (switch._thin_skin): it conducts
        # along itself as that share of a cell, not as its cube
        if floor is None and domain.skin is not None:
            floor = np.zeros(domain.n_elem)
            floor[domain.skin.elem] = domain.skin.phi
        self.floor = np.zeros(domain.n_elem) if floor is None else np.clip(np.asarray(floor, float), 0.0, 0.999)
        # touched surface: the outside faces of the exterior's own cells (not
        # the air or a component showing through an opening)
        bnd = np.flatnonzero(faces.boundary)
        sel = bnd[is_mech[faces.a[bnd]]]
        if spec.skin_shape is not None:
            sel = sel[spec.skin_shape.sdf(faces.center[sel]) <= 0.5 * pitch]
        self.skin_nodes = np.zeros(mesh.n_nodes, bool)
        self.skin_nodes[faces.nodes[sel].ravel()] = True
        if spec.skin_max is not None and not self.skin_nodes.any():
            raise ValueError("cooling.skin: no outside skin faces found (is there a `skin` on the study?)")
        # vents: their outside faces, area and height
        self.vent_ah = []
        for v in spec.vents:
            sel_v = bnd[v.shape.sdf(faces.center[bnd]) <= 0.5 * pitch]
            v.faces = sel_v
            wv = self.out_w[np.searchsorted(bnd, sel_v)]
            area = v.area if v.area is not None else float(wv.sum()) * pitch ** 2
            if not len(sel_v) and v.area is None:
                raise ValueError(f"cooling vent {v.id!r} touches no outside face of the grid")
            if len(sel_v):
                pos = faces.center[sel_v].mean(axis=0)
            else:
                lo, hi = v.shape.bounds()
                pos = 0.5 * (np.asarray(lo) + np.asarray(hi))
            self.vent_ah.append((float(area), float(np.asarray(pos) @ spec.up)))
        self.g_vent = 0.0
        self.vent_flow = 0.0
        self._theta = None

    # ---- fields
    def conductivity(self, xp):
        """W/(m K) per thermal cell for mechanical densities xp."""
        k = self.k0.copy()
        k[self.mech] = self.spec.k_air + self._share(xp)[0] * (self.spec.k_material - self.spec.k_air)
        return k

    def _share(self, xp):
        """(conducting share of each mechanical cell, its derivative): x^3,
        or on a thin-skin cell the skin's share phi plus the cube of what
        the design adds over it."""
        x = np.clip(np.asarray(xp, float), 0.0, 1.0)
        phi = self.floor
        y = np.clip((x - phi) / (1.0 - phi), 0.0, 1.0)
        s = phi + (1.0 - phi) * y ** 3
        ds = np.where(x > phi, 3.0 * y ** 2, 0.0)
        return s, ds

    def air(self, xp):
        a = self.air0.copy()
        a[self.mech] = 1.0 - np.clip(np.asarray(xp, float), 0.0, 1.0)
        a[self.mech[self.domain.preserve]] = 0.0
        return a

    def _vent_sink(self, air):
        """W/K per node from the stack flow, spread over the air by volume."""
        v_air = float(air.sum())
        if self.g_vent <= 0 or v_air <= 0:
            return None, 0.0
        per = self.g_vent * air / v_air
        return np.bincount(self.mesh.enode.ravel(), weights=np.repeat(per / 8.0, 8),
                           minlength=self.mesh.n_nodes), v_air

    def solve(self, xp, vent_rounds: int = 1):
        """Node temperature rise over ambient for mechanical densities xp.
        With vents, the stack flow is taken from the last solve's air
        temperature (vent_rounds re-solves it to a fixed point). The system
        is factored once (PARDISO when installed) and the factor kept for
        the adjoints (`adjoint`)."""
        k = self.conductivity(xp) * 1e-3
        air = self.air(xp)
        for r in range(max(1, vent_rounds if self.vent_ah else 1)):
            mat, f = self.assemble(k), self.heat
            sink, _ = self._vent_sink(air)
            if sink is not None:
                mat = mat + sp.diags(sink)
            mat = mat.tocsr()
            self._factorize(mat)
            theta = self.adjoint(f) if self._direct is not None else self.solver.solve(mat, f, x0=self._theta)
            self._theta = theta
            self._update_vents(theta, air)
        self._mat = mat
        return theta

    _direct = None
    _pardiso = None
    CHUNK = 200_000

    def assemble(self, k):
        """Conduction matrix for per-cell k (W/(mm K)) plus the outside's
        convection, built a chunk of cells at a time: thermal.Conduction's
        one-shot pattern took 5 GB at 0.9 M cells and could not read a
        handheld's delivered mesh at 0.33 mm."""
        n = self.mesh.n_nodes
        en = self.mesh.enode
        kt = self._kt.ravel()
        mat = self._kc.copy()
        for s in range(0, len(en), self.CHUNK):
            e = en[s:s + self.CHUNK].astype(np.int32)
            mat = mat + sp.csr_matrix(((k[s:s + self.CHUNK, None] * kt[None]).ravel(),
                                       (np.repeat(e, 8, axis=1).ravel(), np.tile(e, (1, 8)).ravel())),
                                      shape=(n, n))
        mat.sum_duplicates()
        mat.sort_indices()
        return mat

    def _factorize(self, mat):
        self.release()
        if not _HAS_PARDISO or mat.shape[0] > DIRECT_MAX:
            return
        from .bucklingopt import _factor
        self._direct, self._pardiso = _factor(mat)

    def adjoint(self, rhs):
        """Solve with the last system (symmetric: the adjoint is the same)."""
        if self._direct is not None:
            return np.asarray(self._direct(np.asarray(rhs, float)), float).reshape(np.shape(rhs))
        return self.solver.solve(self._mat, rhs)

    def release(self):
        if self._pardiso is not None:
            self._pardiso()       # bucklingopt._factor's release: MKL holds the factor outside Python
        self._direct, self._pardiso = None, None

    def air_rise(self, theta, air):
        te = theta[self.mesh.enode].mean(axis=1)
        return float((te * air).sum() / max(air.sum(), 1e-12))

    def _update_vents(self, theta, air):
        if not self.vent_ah:
            return
        dt = self.air_rise(theta, air)
        self.g_vent, self.vent_flow = stack_conductance(self.vent_ah, dt, self.spec.ambient)

    # ---- gradients
    def gradient(self, xp, theta, lam):
        """d(J)/d xp for an objective J with adjoint lam (K lam = dJ/dtheta):
        - lam . dK/dxp theta (the vent conductance held)."""
        en = self.mesh.enode[self.mech]
        e = np.einsum("ij,jk,ik->i", lam[en], self._kt, theta[en])
        g = -self._share(xp)[1] * (self.spec.k_material - self.spec.k_air) * 1e-3 * e
        if self.g_vent > 0:
            air = self.air(xp)
            v_air = float(air.sum())
            en = self.mesh.enode[self.mech]
            lt = (lam[en] * theta[en]).sum(axis=1) / 8.0
            # d air / d x = -1: K loses g/v_air per node-eighth
            g += self.g_vent / v_air * lt
            g[self.domain.preserve] = 0.0
        return g

    # ---- heat over time
    def capacity(self, xp):
        """Lumped heat capacity per node, J/K: a designed cell holds its
        density's share of the material's rho c_p (the rest is air)."""
        c = self.c0.copy()
        x = np.clip(np.asarray(xp, float), 0.0, 1.0)
        c[self.mech] = RHO_CP_AIR_VOL * self.mesh.pitch ** 3 + x * (self.c_mat - RHO_CP_AIR_VOL * self.mesh.pitch ** 3)
        return np.bincount(self.mesh.enode.ravel(), weights=np.repeat(c / 8.0, 8), minlength=self.mesh.n_nodes)

    def history(self, xp, vent_rounds: int = 1):
        """Node temperature rise at every step of the study's cycle
        (backward Euler on the conduction system with a lumped capacity:
        (K + C/dt) theta_n = C/dt theta_n-1 + f_n), as a (steps + 1, nodes)
        array with the start state first, and the first step the limits
        read. One factor per distinct step length, kept for the adjoint
        (`history_gradient`)."""
        cyc = self.spec.cycle
        steps, first = cyc.steps()
        sw = cyc.start_watts()
        # the steady start (and the vents' flow, frozen from it)
        self.heat_saved = self.heat
        if sw is not None:
            self.heat = self.src_load @ sw
        theta0 = self.solve(xp, vent_rounds=vent_rounds) if sw is not None else np.zeros(self.mesh.n_nodes)
        if sw is None:
            self.solve(xp, vent_rounds=1)         # K (and the vents at no rise: none)
        self.heat = self.heat_saved
        self._steady = (self._direct, self._pardiso) if sw is not None else None
        if sw is None:
            self.release()
        self._direct, self._pardiso = None, None
        kmat = self._mat
        cap = self.capacity(xp)
        self._cap = cap
        self._steps = steps
        self._start_steady = sw is not None
        self._factors = {}
        th = [theta0]
        for dt, w in steps:
            solve = self._step_factor(kmat, cap, dt)
            th.append(solve(cap / dt * th[-1] + self.src_load @ w))
        return np.array(th), first + 1

    def _step_factor(self, kmat, cap, dt):
        key = float(dt)
        if key not in self._factors:
            a = (kmat + sp.diags(cap / dt)).tocsr()
            if _HAS_PARDISO and a.shape[0] <= DIRECT_MAX:
                from .bucklingopt import _factor
                f, rel = _factor(a)
                self._factors[key] = (lambda b, f=f: np.asarray(f(np.asarray(b, float)), float).reshape(np.shape(b)),
                                      rel)
            else:
                self._factors[key] = (lambda b, a=a: self._cg_block(a, b), None)
        return self._factors[key][0]

    def _cg_block(self, a, b):
        b = np.asarray(b, float)
        if b.ndim == 1:
            return self.solver.solve(a, b)
        return np.stack([self.solver.solve(a, b[:, j]) for j in range(b.shape[1])], axis=1)

    def history_gradient(self, xp, th, dj):
        """d J / d xp for objectives J of the history th: dj is (steps + 1,
        nodes, m), dJ_j/dtheta_n. The adjoint runs back from the last step
        on the same factors: (K + C/dt_n) lam_n = dJ/dtheta_n + C/dt_n+1
        lam_n+1, and K lam_0 = dJ/dtheta_0 + C/dt_1 lam_1 from a steady
        start. Returns (xp size, m)."""
        steps, cap = self._steps, self._cap
        n_steps = len(steps)
        m = dj.shape[2]
        g = np.zeros((len(xp), m))
        en = self.mesh.enode[self.mech]
        dce = (self.c_mat - RHO_CP_AIR_VOL * self.mesh.pitch ** 3) / 8.0
        lam_next = None
        for n in range(n_steps, 0, -1):
            dt = steps[n - 1][0]
            rhs = dj[n].copy()
            if lam_next is not None:
                rhs += (cap / steps[n][0])[:, None] * lam_next
            lam = self._factors[float(dt)][0](rhs)
            lam = lam.reshape(-1, m)
            dth = (th[n] - th[n - 1]) / dt
            for j in range(m):
                g[:, j] += self.gradient(xp, th[n], lam[:, j])
                g[:, j] -= dce * (lam[en, j] * dth[en]).sum(axis=1)
            lam_next = lam
        if self._start_steady:
            rhs = dj[0] + (cap / steps[0][0])[:, None] * lam_next
            direct, _ = self._steady
            lam0 = np.asarray(direct(rhs), float).reshape(-1, m) if direct is not None else \
                self._cg_block(self._mat, rhs).reshape(-1, m)
            for j in range(m):
                g[:, j] += self.gradient(xp, th[0], lam0[:, j])
        g[self.domain.preserve] = 0.0
        return g

    def release_history(self):
        for _, rel in getattr(self, "_factors", {}).values():
            if rel is not None:
                rel()
        self._factors = {}
        st = getattr(self, "_steady", None)
        if st is not None and st[1] is not None:
            st[1]()
        self._steady = None


class _Limit:
    def __init__(self, name, nodes, limit_rise):
        self.name, self.nodes, self.limit = name, np.flatnonzero(nodes), float(limit_rise)
        self.scale = 1.0

    def value(self, theta, p=P_NORM):
        t = np.maximum(theta[self.nodes], 1e-12)
        tmax = float(t.max())
        tn = t / tmax
        pn = tmax * float(np.mean(tn ** p)) ** (1.0 / p)
        d = np.zeros(len(theta))
        d[self.nodes] = (tn ** (p - 1)) * float(np.mean(tn ** p)) ** (1.0 / p - 1.0) / len(t)
        return pn, d, tmax

    def value_history(self, th, first, p=P_NORM):
        """The same p-norm over every node and every step from `first`:
        the hot spot at its hottest moment. d is (steps + 1, nodes)."""
        t = np.maximum(th[first:, self.nodes], 1e-12)
        tmax = float(t.max())
        tn = t / tmax
        mp = float(np.mean(tn ** p))
        pn = tmax * mp ** (1.0 / p)
        d = np.zeros(th.shape)
        d[first:, self.nodes] = (tn ** (p - 1)) * mp ** (1.0 / p - 1.0) / t.size
        return pn, d, tmax


class CoolingProblem:
    """The cooling limits on one grid, as constraints for
    switch.optimize_switch (see `providers` there)."""

    # `robust`: the limits hold on the design eroded and dilated by this
    # much of the projection too (simp's robust formulation, as feel.py),
    # for designs whose grey cells the delivered mesh reads differently.
    # Off by default: on the handheld it cost 8% more material and closed
    # only half of a 1.1 K delivered skin overshoot, which came from the
    # mesher closing an air gap (see run_cooling), not from grey cells
    ROBUST = 0.15

    def __init__(self, study, domain: Domain, nu=None, modulus=None, floor=None):
        self.model = CoolingModel(study, domain, floor=floor)
        spec = self.model.spec
        self.limits = []
        for s in spec.sources:
            if s.max_t is not None:
                self.limits.append(_Limit(f"cool:{s.id}", s.nodes, s.max_t - spec.ambient))
        if spec.skin_max is not None:
            self.limits.append(_Limit("cool:skin", self.model.skin_nodes, spec.skin_max - spec.ambient))
        # `margin`: K under each limit the design aims for (the readings
        # keep the true limits): the design grid's own error, read off the
        # delivered part's finer grids
        margin = float(spec.raw.get("margin", 0.0))
        for lim in self.limits:
            if margin >= lim.limit:
                raise ValueError(f"cooling.margin {margin:g} K is more than {lim.name}'s whole rise over ambient")
            lim.limit -= margin
        rb = spec.raw.get("robust", False)
        self.robust = float(self.ROBUST if rb is True else (rb or 0.0))
        self.sides = (("eroded", 0.5 + self.robust), ("dilated", 0.5 - self.robust)) if self.robust > 0 else \
            (("", None),)
        self.m = len(self.limits) * len(self.sides)

    def evaluate(self, xp, ctx=None):
        from .simp import _dproject, _project
        out = []
        for side, eta in self.sides:
            if eta is None or ctx is None:
                x, d, wrt = np.asarray(xp, float), None, None
            else:
                x = _project(ctx["xt"], ctx["beta"], eta)
                d = _dproject(ctx["xt"], ctx["beta"], eta)
                x[ctx["preserve"]] = 1.0
                d[ctx["preserve"]] = 0.0
                held = ctx["floor"] > x
                x[held] = ctx["floor"][held]
                d[held] = 0.0
                wrt = "xt"
            for val, g, name in self._evaluate_one(x, side):
                tag = f"{name}:{side}" if side else name
                out.append((val, g, tag) if wrt is None else (val, g * d, tag, wrt))
        return out

    def _evaluate_one(self, xp, side):
        mdl = self.model
        if mdl.spec.cycle is not None:
            th, first = mdl.history(xp)
            vals, ds, scales = [], [], []
            for lim in self.limits:
                pn, d, tmax = lim.value_history(th, first)
                vals.append(pn)
                ds.append(d)
                scales.append(tmax / max(pn, 1e-12))
            g = mdl.history_gradient(xp, th, np.stack(ds, axis=2))
            mdl.release_history()
            return [(sc * pn / lim.limit - 1.0, sc * g[:, j] / lim.limit, lim.name)
                    for j, (lim, pn, sc) in enumerate(zip(self.limits, vals, scales))]
        theta = mdl.solve(xp)
        out = []
        for lim in self.limits:
            pn, d, tmax = lim.value(theta)
            # adaptive scale: the p-norm times the last true max / p-norm
            sc = tmax / max(pn, 1e-12)
            lam = mdl.adjoint(d)
            g = mdl.gradient(xp, theta, lam)
            out.append((sc * pn / lim.limit - 1.0, sc * g / lim.limit, lim.name))
        mdl.release()
        return out


# ------------------------------------------------------------- readings


def cooling_readings(study, domain: Domain, xp, floor=None, vent_rounds: int = 8, fill=None) -> dict:
    """Temperatures of a design: each component's hottest point and its
    limit, the touched skin's hottest and mean, where the heat leaves (the
    skin by convection, vents by stack flow) and the air's mean rise.

    fill: model -> a mask over the model's thermal cells that are solid
    material whatever the design says (the delivered part's own fill,
    including cells the design grid kept clear)."""
    mdl = CoolingModel(study, domain, floor=floor)
    spec = mdl.spec
    if fill is not None:
        m = np.asarray(fill(mdl), bool)
        free = m & ~mdl.src & ~mdl.is_mech
        mdl.k0[free] = spec.k_material
        xp = np.array(xp, float)
        xp[m[mdl.mech]] = 1.0
    times = None
    if spec.cycle is not None:
        # with a cycle: every node's hottest moment (from the limits' first
        # step on), and the heat out at the end of the run
        th, first = mdl.history(xp, vent_rounds=vent_rounds)
        mdl.release_history()
        clock = np.concatenate([[0.0], np.cumsum([dt for dt, _ in mdl._steps])])
        clock = clock - clock[first - 1]
        hist = th[first:]
        theta = hist.max(axis=0)
        end = th[-1]
        times = (hist, clock[first:])
    else:
        theta = mdl.solve(xp, vent_rounds=vent_rounds)
        end = theta
    t = theta + spec.ambient
    rows = []
    for s in spec.sources:
        tm = float(t[s.nodes].max())
        row = {"component": s.id, "watts": s.watts, "maxTemp": s.max_t, "temperature": round(tm, 2),
               "mean": round(float(t[s.nodes].mean()), 2),
               "ok": bool(s.max_t is None or tm <= s.max_t + 0.25)}
        if times is not None:
            row["peakAt"] = round(float(times[1][int(np.argmax(times[0][:, s.nodes].max(axis=1)))]), 1)
        rows.append(row)
    f = mdl.faces
    bnd = np.flatnonzero(f.boundary)
    th_face = end[f.nodes[bnd]].mean(axis=1)
    q_skin = float(np.sum(spec.h_out * 1e-6 * f.area * mdl.out_w * th_face))
    air = mdl.air(xp)
    rep = {"components": rows, "ambient": spec.ambient, "watts": float(sum(s.watts for s in spec.sources)),
           "heatOut": {"skin": round(q_skin, 4)}, "airRise": round(mdl.air_rise(theta, air), 2),
           "cells": int(mdl.mesh.n_elem), "pitch": float(domain.pitch)}
    if mdl.vent_ah:
        rep["heatOut"]["vents"] = round(mdl.g_vent * mdl.air_rise(theta, air), 4)
        rep["ventFlow"] = {"litresPerMinute": round(mdl.vent_flow * 6e4, 3),
                           "conductance": round(mdl.g_vent, 5),
                           "vents": [{"id": v.id, "area": a, "height": h} for v, (a, h) in zip(spec.vents, mdl.vent_ah)]}
    if mdl.skin_nodes.any():
        ts = t[mdl.skin_nodes]
        hot = int(np.flatnonzero(mdl.skin_nodes)[int(np.argmax(ts))])
        rep["skin"] = {"maxTemp": spec.skin_max, "temperature": round(float(ts.max()), 2),
                       "mean": round(float(ts.mean()), 2), "hotSpot": mdl.mesh.node_xyz[hot].round(2).tolist(),
                       "ok": bool(spec.skin_max is None or ts.max() <= spec.skin_max + 0.25)}
        if times is not None:
            rep["skin"]["peakAt"] = round(float(times[1][int(np.argmax(times[0][:, mdl.skin_nodes].max(axis=1)))]), 1)
    if times is not None:
        rep["cycle"] = {"seconds": round(float(times[1][-1]), 1), "steps": int(len(times[1]) - 1),
                        "note": "temperatures are each point's hottest over the cycle; heatOut at its end"}
    rep["ok"] = all(r["ok"] for r in rows) and rep.get("skin", {}).get("ok", True)
    mdl.release()
    rep["_theta"] = t
    rep["_model"] = mdl
    return rep


def _public(rep):
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def delivered_readings(study, mesh, pitches) -> list[dict]:
    """The delivered mesh's temperatures on finer grids: each cell solid
    where at least half of it is inside the mesh (the skin and preserves
    as the study keeps them)."""
    from .domain import _fraction, build_domain
    from .shapes import MeshShape
    from .switch import _thin_skin
    out = []
    ms = MeshShape(mesh)
    gap = float((study.raw.get("cooling") or {}).get("padGap", PAD_GAP))
    for p in pitches:
        dom, fl = _thin_skin(study, build_domain(study, p))
        drawn = _fraction([ms], dom.mesh.elem_centers(), dom.pitch)
        x = np.where(drawn >= 0.5, 1.0, 0.0)
        x[dom.preserve] = 1.0
        # a thin skin the grid cannot resolve reads at its own share
        x = np.maximum(x, fl)
        out.append(_public(cooling_readings(study, dom, x, floor=fl,
                                            fill=lambda mdl: _delivered_fill(mdl, ms, x, gap))))
    return out


# a gap this thin between a component and the plastic beside it is taken as
# filled by a thermal pad (or paste), as the model assumes everywhere else
PAD_GAP = 1.0


def _delivered_fill(mdl, ms, x, gap: float):
    """The delivered part on the thermal grid, read the same way on every
    grid. Two things a design-grid view gets wrong by where the grid falls:

    - Cells the design grid keeps clear around a keep-out (any cell a
      keep-out reaches into) are air in the design, but the delivered mesh
      may fill part of them: they are read off the mesh like any other.
    - A gap of up to `gap` mm between a component and the plastic beside it
      is the thermal pad the model assumes, filled.

    The speaker's amp board sits 0.3-0.6 mm off its pads: the 0.75 and
    0.5 mm grids read a whole cell of air around it (amp 66 C) and the 1.05
    and 0.6 mm grids read contact (57 C)."""
    from scipy import ndimage
    from .domain import _fraction
    mesh = mdl.mesh
    c = mesh.elem_centers()
    solid = np.zeros(mesh.n_elem, bool)
    solid[mdl.mech] = x >= 0.5
    rest = ~mdl.is_mech & ~mdl.src
    if rest.any():
        solid[rest] = _fraction([ms], c[rest], mesh.pitch) >= 0.5
    if gap > 0 and mdl.spec.sources:
        d = np.full(mesh.n_elem, np.inf)
        for s in mdl.spec.sources:
            d = np.minimum(d, s.shape.sdf(c))
        band_e = (d > 0) & (d <= gap) & ~mdl.src
        if band_e.any():
            ijk = mesh.elem_ijk
            lo = ijk.min(axis=0)
            dims = tuple(ijk.max(axis=0) - lo + 1)
            at = tuple((ijk - lo).T)
            grown = np.zeros(dims, bool)
            grown[at] = solid
            band = np.zeros(dims, bool)
            band[at] = band_e
            # grow the plastic toward the component, at most the gap deep:
            # the band holds only cells that close to a component
            for _ in range(max(1, int(np.ceil(gap / mesh.pitch)))):
                grown |= ndimage.binary_dilation(grown) & band
            solid = grown[at]
    return solid


# ----------------------------------------------------------------- the run


def run_cooling(study, out_dir: str, pitches=None, iterations=None, on_progress=None,
                verify_pitches=None) -> dict:
    """Optimize, mesh and check a study's `cooling` block (with its
    `switch` and `feel` blocks too, if it has them). Writes <name>.stl
    and returns the report: each limit read on the design, on the
    black-and-white design and on the delivered mesh at two finer grids."""
    from .switch import run_switch

    cb = study.raw.get("cooling") or {}
    has_feel = bool(study.raw.get("feel"))
    cooling_from_study(study)       # errors before any solve

    def make(st, domain, nu, modulus):
        return CoolingProblem(st, domain, nu, modulus)

    providers = [make]
    feel_readback = None
    from .reconstruct import ReconstructOptions
    # no closing: it fills slots under twice its radius (1 mm here) as
    # unprintable, but an air gap of 1-2 mm prints (MJF, SLS, FDM) and is
    # what keeps a pad's heat off the skin. With it, the handheld's mesh
    # bridged a 1.4 mm gap between a pad under the board and the back
    # skin: 2.1 g more and the back 0.5 K over its limit
    mesh_opts = ReconstructOptions(close=0.0)
    opt_opts = {}
    if has_feel:
        from .feel import FeelProblem
        providers.append(lambda st, domain, nu, modulus: FeelProblem(st, domain, nu, modulus))
        fb = study.raw.get("feel") or {}
        mesh_opts = ReconstructOptions(close=0.0, axial_debias=bool(fb.get("keepThinMembers", True)))
        opt_opts["move_sharp"] = float(fb.get("moveSharp", 0.03))

    def readback(domain, design, solid, delivered, floor, mesh):
        out = {}
        if on_progress is not None:
            on_progress({"stage": "verify", "check": "temperatures of the design"})
        rd = cooling_readings(study, domain, design, floor=floor)
        out["cooling"] = {"design": _public(rd),
                          "thresholded": _public(cooling_readings(study, domain, solid, floor=floor))}
        if mesh is not None:
            vps = verify_pitches or cb.get("verifyPitches") or [0.7 * domain.pitch, 0.5 * domain.pitch]
            if on_progress is not None:
                on_progress({"stage": "verify", "check": f"temperatures of the delivered mesh at {vps} mm"})
            dl = delivered_readings(study, mesh, vps)
            out["cooling"]["delivered"] = dl
            last = dl[-1]
            out["cooling"]["deliveredOk"] = bool(last["ok"])
            if len(dl) > 1:
                out["cooling"]["gridChange"] = _grid_change(dl[-2], dl[-1])
        out["cooling"]["ok"] = bool(rd["ok"])
        if has_feel:
            from .feel import FeelProblem, feel_readings
            nu = study.elastic(design=True)
            prob = FeelProblem(study, domain, nu, float(study.material.E))
            rows = feel_readings(prob, design)
            for row, rs in zip(rows, feel_readings(prob, solid)):
                row["thresholded"] = rs["directions"]
            out["feel"] = rows
            out["feelOk"] = all(r["ok"] for r in rows)
        return out

    if not study.raw.get("switch"):
        pitches = pitches or cb.get("pitches")
        iterations = iterations or cb.get("iterations")
    rep = run_switch(study, out_dir, pitches=pitches, iterations=iterations, on_progress=on_progress,
                     providers=providers, readback=readback, mesh_options=mesh_opts, optimize_options=opt_opts)
    c = rep.get("cooling") or {}
    rep["passes"] = bool(c.get("ok") and c.get("deliveredOk", True) and rep.get("feelOk", True)
                         and (rep["passes"] if rep.get("presses") else True))
    return rep


def _grid_change(a, b) -> dict:
    """Largest change in any reported temperature rise between two grids."""
    amb = a["ambient"]
    out = {}
    for ra, rb in zip(a["components"], b["components"]):
        out[ra["component"]] = round(abs(rb["temperature"] - ra["temperature"]) / max(rb["temperature"] - amb, 1e-9), 4)
    if "skin" in a and "skin" in b:
        out["skin"] = round(abs(b["skin"]["temperature"] - a["skin"]["temperature"])
                            / max(b["skin"]["temperature"] - amb, 1e-9), 4)
    return out
