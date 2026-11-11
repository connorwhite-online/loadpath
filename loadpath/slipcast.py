"""
Slip casting: a ceramic part made by pouring slip into a plaster mold.

The designer draws the outside of the fired part. Everything else follows
from the process, and this module works it out from the physics of each
step, with the slip's and the plaster's measured properties as the inputs.

**The wall builds by filtration** (`cast_rate`, `cast`). Plaster is a
network of fine pores; its capillary suction (0.1 to 0.2 MPa) draws water
out of the slip, and the particles left behind pack against the mold wall
as a cake. The water has to pass through the cake already there (Darcy
flow, permeability k_c) and then through the wetted plaster (k_m), so the
wall slows as it thickens. One-dimensionally, with n = phi_cake/phi_slip - 1
the volume of water removed per volume of cake laid down:

    L^2 = A t,   A = 2 dP / (mu n (1/k_c + n / (eps_m k_m)))

(Aksay & Schilling's colloidal filtration with the mold's resistance in
series, Tiller & Tsai 1986). That is the potter's rule that twice the time
does not give twice the wall. In a real shape the pressure in the cake
obeys Laplace's equation with the front moving at the local flux, a Stefan
problem; `cast` solves it on the voxel grid of the mold cavity (with the
same compiled kernel as the metal solidification solve, casting.py: water
pressure for temperature, water removed for latent heat), so the wall
comes out thicker in the cavity's inside corners, where two walls draw
water from the same slip, and thinner over a ridge of the mold. Draining
at time t leaves every cell the front has reached: the cast body. A neck
that closes before the drain traps slip behind it, and a solid casting
whose supply closes off starves; both are found from the same field.

**The mold drinks a known amount** (`mold_wetting`): the water removed
wets n L / eps_m of plaster behind each wall, which sets how thick the
mold must be and how many casts it takes before it needs drying.

**It shrinks twice** (`shrinkage`). Drying: water leaves until the
particles touch; the volume lost is the difference between the cake's
packing as cast and bone dry. Firing: mass is conserved (less the loss on
ignition) while the bulk density rises to the fired value. Both are cube
roots of density ratios, no rule of thumb. The mold cavity is the fired
part scaled up by 1 / ((1 - s_dry)(1 - s_fire)).

**It sags in the kiln** (`sag`). At peak temperature the body is a very
viscous solid and creeps under its own weight. The pyroplastic index the
tile industry measures (a bar on two supports through the firing) is,
by viscous beam theory, a direct reading of the creep the body sees over
its firing: PI = 4 h^2 S / (3 L^4) = (5/24) rho g Phi, with Phi the
integral of dt / viscosity. A viscous body under a fixed load deforms like
an elastic one of modulus 1 / Phi, so the kiln sag of any shape is one
linear solve under self-weight on the fired geometry.

**It is brittle** (`failure_probability`): strength is a distribution,
not a number. Weibull's weakest-link integral over the stressed volume
gives the probability a part breaks under a stress field, for the green
(unfired) body when it is lifted from the mold and for the fired one.

Bodies: `BODIES` (porcelain, stoneware, earthenware, alumina). Values are
typical of each class and say where they come from; a real slip is
characterized by its specific gravity, a timed cast (the wall after a
known time gives A) and fired test bars, and `body({...})` takes those.

Not modelled: particle segregation and orientation in the cake (which
make shrinkage slightly anisotropic), drying stress and drying cracks,
glaze fit, and warp from uneven drying.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage, optimize

__all__ = ["Body", "Plaster", "BODIES", "PLASTER", "SOURCES", "body", "cast_rate", "wall_time", "wall_thickness",
           "mold_wetting", "shrinkage", "cast", "SlipCast", "sag", "failure_probability", "effective_volume",
           "slipcast_part"]

G0 = 9.81
WATER_VISCOSITY = 1.0e-3      # Pa s at 20 C

SOURCES = {
    "aksay-schilling": "Aksay & Schilling 1984, Mechanics of colloidal filtration, Advances in Ceramics 9; "
                       "Tiller & Tsai 1986, Theory of filtration of ceramics: I, slip casting, J. Am. Ceram. Soc. "
                       "69:882 (cake and mold resistances in series)",
    "suction": "plaster mold capillary suction 0.1-0.2 MPa (as quoted in slip casting course notes, "
               "https://metalurji.mu.edu.tr/Newfiles/192/Content/Exp_%204%20Manual%202025.pdf)",
    "digitalfire": "Digitalfire reference, slip casting and casting rate (https://digitalfire.com/glossary/439): "
                   "3 mm in about 15 minutes for a casting slip, 20-30 minutes for ball clay and talc slips",
    "pyroplastic": "pyroplastic index of porcelain tile bodies, PI = 4 h^2 S / (3 L^4): Italian and Spanish "
                   "commercial bodies average 3.9e-5 /cm (2.0-6.2), Brazilian 6.7e-5 /cm (3.7-14.9) "
                   "(Qualicer, https://www.qualicer.org/recopilatorio/ponencias/pdfs/0632400e.pdf)",
    "handbook": "typical values for the class of body (Reed, Principles of Ceramics Processing; Rahaman, "
                "Ceramic Processing and Sintering); not re-read for this card",
    "estimate": "estimate: stated in the note",
}


@dataclass(frozen=True)
class Plaster:
    """A plaster mold: capillary suction (Pa), permeability (m2) and the
    pore fraction water can fill."""
    suction: float = 0.15e6
    permeability: float = 1.0e-14
    porosity: float = 0.45
    note: str = ("pottery plaster at 70 parts water to 100 plaster: suction 0.15 MPa (suction), permeability "
                 "1e-14 m2 and 45% open pores (handbook)")


PLASTER = Plaster()


@dataclass(frozen=True)
class Body:
    """A slip and the body it makes. Fractions are of volume unless said.

    solids           solids in the slip (from its specific gravity)
    cake_packing     solids in the wall as cast
    dry_packing      solids bone dry (the particles touching)
    particle_density kg/m3
    cake_permeability m2, of the cast wall to water
    loss_on_ignition mass fraction lost in the firing (bound water, organics)
    fired_density    bulk, kg/m3
    green_strength   MPa, modulus of rupture bone dry; leather_strength as demolded
    fired_strength   MPa, modulus of rupture; with weibull (m) and test_volume (mm3,
                     the effective volume of the bars it was measured on)
    sag_index        pyroplastic index over its firing, 1/cm
    """
    name: str
    description: str
    solids: float
    cake_packing: float
    dry_packing: float
    particle_density: float
    cake_permeability: float
    loss_on_ignition: float
    fired_density: float
    green_strength: float
    leather_strength: float
    fired_strength: float
    weibull: float
    fired_modulus: float          # MPa
    sag_index: float
    green_weibull: float = 6.0
    test_volume: float = 20.0
    poisson: float = 0.22
    notes: dict = field(default_factory=dict)

    @property
    def filtrate_ratio(self):
        """n: water removed per volume of cake formed."""
        return self.cake_packing / self.solids - 1.0

    @property
    def specific_gravity(self):
        return 1.0 + self.solids * (self.particle_density / 1000.0 - 1.0)

    def to_dict(self):
        return {"name": self.name, "description": self.description, "solids": self.solids,
                "specificGravity": self.specific_gravity, "cakePacking": self.cake_packing,
                "dryPacking": self.dry_packing, "particleDensity": self.particle_density,
                "cakePermeability": self.cake_permeability, "lossOnIgnition": self.loss_on_ignition,
                "firedDensity": self.fired_density, "greenStrength": self.green_strength,
                "leatherStrength": self.leather_strength, "firedStrength": self.fired_strength,
                "weibullModulus": self.weibull, "firedModulus": self.fired_modulus, "sagIndex": self.sag_index,
                "sources": dict(self.notes)}


def _permeability_from_rate(rate_mm2_min, solids, cake_packing, plaster: Plaster = PLASTER):
    """The cake permeability a timed cast implies (inverse of cast_rate)."""
    a = rate_mm2_min * 1e-6 / 60.0
    n = cake_packing / solids - 1.0
    inv = 2.0 * plaster.suction / (WATER_VISCOSITY * n * a) - n / (plaster.porosity * plaster.permeability)
    if inv <= 0:
        raise ValueError("that casting rate is faster than the plaster alone allows")
    return 1.0 / inv


def kozeny_carman(diameter_um: float, packing: float) -> float:
    """Permeability (m2) of a packed bed of near-spherical particles of
    surface-mean diameter d: k = d^2 eps^3 / (180 (1 - eps)^2)."""
    d = diameter_um * 1e-6
    eps = 1.0 - packing
    return d * d * eps ** 3 / (180.0 * (1.0 - eps) ** 2)


_CLAY_NOTE = {"solids": "handbook (slip at specific gravity 1.75-1.80)",
              "cake_packing, dry_packing": "handbook (about 20% water by weight as cast; dry bulk density "
                                           "1.65-1.75 g/cm3)",
              "fired_strength, weibull, fired_modulus, green_strength": "handbook",
              "leather_strength": "estimate: a tenth to a fifth of the dry strength at demolding moisture"}

BODIES = {
    "porcelain": Body(
        "porcelain", "Vitreous casting porcelain (kaolin, feldspar, silica, some ball clay), cone 6 to 10",
        solids=0.48, cake_packing=0.605, dry_packing=0.645, particle_density=2600.0,
        cake_permeability=_permeability_from_rate(0.6, 0.48, 0.605), loss_on_ignition=0.065,
        fired_density=2380.0, green_strength=3.0, leather_strength=0.4, fired_strength=70.0, weibull=10.0,
        fired_modulus=70000.0, sag_index=3.9e-5,
        notes=dict(_CLAY_NOTE, cake_permeability="digitalfire (3 mm in 15 minutes: 0.6 mm2/min)",
                   sag_index="pyroplastic (Italian and Spanish porcelain bodies, mean)",
                   fired_density="handbook (vitreous porcelain 2.35-2.45 g/cm3)")),
    "stoneware": Body(
        "stoneware", "Casting stoneware (ball clay, kaolin, feldspar, silica), cone 6",
        solids=0.48, cake_packing=0.60, dry_packing=0.645, particle_density=2620.0,
        cake_permeability=_permeability_from_rate(0.4, 0.48, 0.60), loss_on_ignition=0.07,
        fired_density=2250.0, green_strength=4.0, leather_strength=0.5, fired_strength=50.0, weibull=9.0,
        fired_modulus=55000.0, sag_index=2.0e-5,
        notes=dict(_CLAY_NOTE, cake_permeability="digitalfire (ball clay slips: 3 mm in 20-30 minutes)",
                   sag_index="estimate: half a porcelain's, with less glass at maturity",
                   fired_density="handbook (2.2-2.3 g/cm3, a few percent porosity)")),
    "earthenware": Body(
        "earthenware", "Low-fire talc and ball clay casting body, cone 06 to 04 (porous when fired)",
        solids=0.47, cake_packing=0.59, dry_packing=0.635, particle_density=2650.0,
        cake_permeability=_permeability_from_rate(0.35, 0.47, 0.59), loss_on_ignition=0.08,
        fired_density=1700.0, green_strength=4.0, leather_strength=0.5, fired_strength=25.0, weibull=8.0,
        fired_modulus=25000.0, sag_index=0.2e-5,
        notes=dict(_CLAY_NOTE, cake_permeability="digitalfire (ball clay and talc slips: 20-30 minutes)",
                   sag_index="estimate: almost no glass at cone 04; a twentieth of a porcelain's",
                   fired_density="handbook (talc bodies fire with under 1% shrinkage and 15-20% absorption)")),
    "alumina": Body(
        "alumina", "Deflocculated submicron alumina slip (99.5% Al2O3), sintered at 1550 to 1650 C",
        solids=0.50, cake_packing=0.62, dry_packing=0.62, particle_density=3980.0,
        cake_permeability=kozeny_carman(0.25, 0.62), loss_on_ignition=0.005,
        fired_density=3900.0, green_strength=2.0, leather_strength=0.5, fired_strength=330.0, weibull=10.0,
        fired_modulus=370000.0, sag_index=0.1e-5, poisson=0.22,
        notes={"solids, cake_packing, fired_density, fired_strength, fired_modulus": "handbook",
               "dry_packing": "handbook (no plastic clay: the cake does not shrink on drying)",
               "cake_permeability": "estimate: Kozeny-Carman at a 0.25 um surface-mean diameter",
               "sag_index": "estimate: solid-state sintering, no glass; set from a fired test bar if it matters",
               "green_strength, leather_strength": "estimate: binderless cast alumina"}),
}


def body(name) -> Body:
    """A card by name, a dict of overrides on a named card ({"base":
    "porcelain", "specificGravity": 1.78, "castingRate": 0.5, ...}), or a Body.

    castingRate (mm2/min, the wall squared over the time of a timed cast in
    a dry plaster mold) sets the cake permeability; particleSize (um) sets
    it by Kozeny-Carman instead."""
    if isinstance(name, Body):
        return name
    if isinstance(name, dict):
        b = body(name.get("base", "porcelain"))
        keys = {"solids": "solids", "cakePacking": "cake_packing", "dryPacking": "dry_packing",
                "particleDensity": "particle_density", "cakePermeability": "cake_permeability",
                "lossOnIgnition": "loss_on_ignition", "firedDensity": "fired_density",
                "greenStrength": "green_strength", "leatherStrength": "leather_strength",
                "firedStrength": "fired_strength", "weibullModulus": "weibull", "firedModulus": "fired_modulus",
                "sagIndex": "sag_index", "testVolume": "test_volume"}
        over = {}
        for k, v in name.items():
            if k in ("base", "name", "specificGravity", "castingRate", "particleSize"):
                continue
            if k not in keys:
                raise ValueError(f"slip body: unknown property '{k}' (one of "
                                 f"{sorted(list(keys) + ['specificGravity', 'castingRate', 'particleSize'])})")
            over[keys[k]] = float(v)
        b = replace(b, name=str(name.get("name", b.name)), **over)
        if "specificGravity" in name:
            b = replace(b, solids=(float(name["specificGravity"]) - 1.0) / (b.particle_density / 1000.0 - 1.0))
        if "castingRate" in name:
            b = replace(b, cake_permeability=_permeability_from_rate(float(name["castingRate"]), b.solids,
                                                                     b.cake_packing))
        elif "particleSize" in name:
            b = replace(b, cake_permeability=kozeny_carman(float(name["particleSize"]), b.cake_packing))
        if not 0.05 < b.solids < b.cake_packing < 0.9:
            raise ValueError("slip body: solids must be under cakePacking (the cake is denser than the slip)")
        return b
    key = str(name).lower()
    if key not in BODIES:
        raise ValueError(f"slip body '{name}': one of {sorted(BODIES)}, or a dict of properties on a base")
    return BODIES[key]


# --- filtration ---------------------------------------------------------------

def cast_rate(b: Body, plaster: Plaster = PLASTER, pressure: float = 0.0, viscosity: float = WATER_VISCOSITY):
    """A in L^2 = A t (m2/s): cake and wetted plaster in series under the
    plaster's suction plus any applied pressure (pressure casting, Pa)."""
    n = b.filtrate_ratio
    return 2.0 * (plaster.suction + pressure) / (
        viscosity * n * (1.0 / b.cake_permeability + n / (plaster.porosity * plaster.permeability)))


def mold_share(b: Body, plaster: Plaster = PLASTER) -> float:
    """Share of the resistance to the water that is the plaster's."""
    n = b.filtrate_ratio
    rm = n / (plaster.porosity * plaster.permeability)
    return rm / (rm + 1.0 / b.cake_permeability)


def wall_time(thickness_mm: float, b: Body, plaster: Plaster = PLASTER, pressure: float = 0.0) -> float:
    """Seconds in the mold for a flat wall this thick (as cast)."""
    return (thickness_mm * 1e-3) ** 2 / cast_rate(b, plaster, pressure)


def wall_thickness(seconds: float, b: Body, plaster: Plaster = PLASTER, pressure: float = 0.0) -> float:
    """Flat wall (mm, as cast) after this long in the mold."""
    return math.sqrt(cast_rate(b, plaster, pressure) * seconds) * 1e3


def mold_wetting(thickness_mm: float, b: Body, plaster: Plaster = PLASTER) -> dict:
    """What casting a wall this thick puts into the plaster: litres of
    water per square metre of mold face, and the depth of plaster it fills."""
    water = b.filtrate_ratio * thickness_mm          # mm of water = L/m2
    return {"waterPerArea": water, "wetDepth": water / plaster.porosity}


def shrinkage(b: Body) -> dict:
    """Linear shrinkage from the mold to the kiln, by conservation of the
    solids: drying takes the cake's packing to the dry packing; firing takes
    the dry bulk density, less the loss on ignition, to the fired density."""
    dry = 1.0 - (b.cake_packing / b.dry_packing) ** (1.0 / 3.0)
    rho_dry = b.dry_packing * b.particle_density
    fire = 1.0 - (rho_dry * (1.0 - b.loss_on_ignition) / b.fired_density) ** (1.0 / 3.0)
    fire = max(fire, 0.0)
    total = 1.0 - (1.0 - dry) * (1.0 - fire)
    return {"drying": dry, "firing": fire, "total": total, "moldScale": 1.0 / (1.0 - total),
            "greenDensity": rho_dry, "firedDensity": b.fired_density,
            "firedPorosity": max(0.0, 1.0 - b.fired_density / (b.particle_density * (1.0 - 0.0)))}


# --- the cast in a real cavity ---------------------------------------------------

@dataclass
class SlipCast:
    """Result of `cast`. Grids are the cavity's shape.

    arrival   seconds at which the cake front reaches each cavity cell
              (NaN where it has not by the end of the solve)
    body      cells that are cast wall at the drain time (all of the cavity
              for a solid cast)
    fill      the same as a share per cell: 1 in the wall, the part-filled
              share of the cell the front stands in, 0 in the drained core
    trapped   cells of slip cut off from the opening when the mold is drained
    starved   (solid casting) cells whose slip supply closed off before they filled
    """
    cavity: np.ndarray
    pitch: float
    arrival: np.ndarray
    body: np.ndarray
    fill: np.ndarray
    trapped: np.ndarray
    starved: np.ndarray
    opening: np.ndarray
    drain_time: float | None
    close_time: float | None
    nominal: float
    thickness: dict
    seconds: float


def _stefan_lambda2(ste: float) -> float:
    """lambda^2 of the one-phase Stefan problem, lambda exp(lambda^2) erf(lambda) = Ste / sqrt(pi)."""
    f = lambda x: x * math.exp(x * x) * math.erf(x) - ste / math.sqrt(math.pi)  # noqa: E731
    lam = optimize.brentq(f, 1e-9, 3.0)
    return lam * lam


def _fed(arrival, source, levels: int = 48):
    """For each cell, was it still joined to `source` through unfilled
    cells at the moment the front reached it? (False: its supply had
    already closed off.) Checked at `levels` times through the cast."""
    fin = np.isfinite(arrival)
    out = np.ones(arrival.shape, bool)
    if not fin.any():
        return out
    ts = np.quantile(arrival[fin], np.linspace(0.0, 1.0, levels + 1)[:-1])
    inside = fin | np.isnan(arrival)
    struct = ndimage.generate_binary_structure(3, 1)
    for lo, hi in zip(ts, list(ts[1:]) + [np.inf]):
        open_ = np.where(fin, arrival >= lo, False) | (~fin & inside & np.isnan(arrival))
        lab, n = ndimage.label(open_, structure=struct)
        good = np.unique(lab[source & open_])
        good = good[good > 0]
        band = fin & (arrival >= lo) & (arrival < hi)
        out[band] = np.isin(lab[band], good)
    return out


def cast(cavity, pitch: float, b: Body, plaster: Plaster = PLASTER, *, wall: float | None = None,
         time: float | None = None, solid: bool = False, up=(2, 1), opening=None, pressure: float = 0.0,
         stefan: float = 0.04) -> SlipCast:
    """Cast the cavity: the cake front's arrival time at every cell.

    cavity   bool grid of the mold cavity at `pitch` mm (mold scale)
    wall     drain when a flat wall would be this thick (mm), or
    time     drain after this many seconds; or solid=True to cast it solid
    opening  bool grid of cavity cells at the pour opening (no mold wall on
             their `up` side, and where a solid cast is topped up from);
             None: the cavity's top layer
    The pressure field in the cake is the Stefan problem's quasi-steady
    limit. It is solved as a Stefan problem with a small artificial storage
    (Stefan number `stefan`), and the times are corrected by the exact
    one-dimensional ratio between the two, so a flat wall comes out on
    L^2 = A t exactly; the plaster's own resistance scales every time by the
    same series factor as in one dimension."""
    import time as _time

    from . import casting as C

    t_wall = _time.time()
    cavity = np.asarray(cavity, bool)
    ax, sg = up
    n = b.filtrate_ratio
    dP = plaster.suction + pressure
    A = cast_rate(b, plaster, pressure)
    A_cake = 2.0 * dP * b.cake_permeability / (WATER_VISCOSITY * n)       # the cake alone
    if solid:
        t_drain = None
    elif time is not None:
        t_drain = float(time)
    elif wall is not None:
        t_drain = wall_time(wall, b, plaster, pressure)
    else:
        raise ValueError("cast: give wall (mm), time (s) or solid=True")
    # the cavity in a one-cell frame; the cells beside it are plaster face (held at the suction),
    # except above the opening, where there is slip, not mold
    cav = np.pad(cavity, 1)
    if opening is None:
        h = np.moveaxis(np.indices(cavity.shape)[ax], 0, 0) * sg
        top = h[cavity].max()
        opening = cavity & (h >= top)
    opening = np.asarray(opening, bool)
    op = np.pad(opening, 1)
    mold_face = ndimage.binary_dilation(cav) & ~cav
    above = np.roll(op, sg, axis=ax) & ~cav
    mold_face &= ~above
    # pressure (Pa) for temperature: slip at 0, plaster face at -dP; water removed per cake volume for
    # latent heat; k/mu for conductivity; a small storage c = stefan n / dP
    pseudo = C.CastAlloy("slip", "filtration", liquidus=0.0, solidus=-0.02, latent=n, density=1.0,
                         specific_heat=stefan * n / dP, conductivity=b.cake_permeability / WATER_VISCOSITY,
                         viscosity=1.0, surface_tension=1.0, shrinkage=0.0, cte=0.0,
                         fraction=((0.0, 0.0), (-0.02, 1.0)))
    none = C.MoldMaterial("none", "", 1.0, ((0.0, 1e-30), (1.0, 1e-30)), ((0.0, 1.0), (1.0, 1.0)))
    s = C.CastSetup(alloy=pseudo, mold=none, process="slip", pour_temperature=0.0, mold_temperature=0.0,
                    ambient=-dP, htc=1e30, shell=None, up=up)
    half = float(ndimage.distance_transform_edt(cav).max()) * pitch * 1e-3
    t_close = half * half / A_cake                                         # a slab that thick, in solver time
    corr = 2.0 * _stefan_lambda2(stefan) / stefan                          # exact 1-D: quasi-steady / Stefan
    series = A_cake / A                                                    # plaster resistance in series
    scale = series / corr                                                  # solver time -> real time
    # a drain cast is solved on until the front is a cell and a half past the flat wall, so the
    # cell the front stands in at the drain has an arrival time to interpolate with
    w_flat = None if t_drain is None else math.sqrt(A * t_drain) * 1e3
    stop = None if t_drain is None else t_drain * ((w_flat + 1.6 * pitch) / w_flat) ** 2 / scale
    r = C.solidify(cav, pitch, s, pad=0, ambient=mold_face, ambient_htc=1e30,
                   time_scale=(t_close if stop is None else min(stop, t_close)), max_time=stop, records=300)
    core = tuple(slice(1, -1) for _ in range(3))
    arrival = r.time[core] * scale
    arrival = np.where(cavity, arrival, np.nan)
    reached = np.isfinite(arrival)
    struct = ndimage.generate_binary_structure(3, 1)
    trapped = np.zeros(cavity.shape, bool)
    starved = np.zeros(cavity.shape, bool)
    fill = cavity.astype(float)
    if t_drain is None:
        body_ = cavity.copy()
        starved = cavity & ~_fed(np.where(cavity, np.where(reached, arrival, np.inf), np.nan), opening) & reached
        close = float(np.nanmax(arrival)) if reached.any() else None
    else:
        body_ = cavity & reached & (arrival <= t_drain)
        liquid = cavity & ~body_
        lab, nl = ndimage.label(liquid, structure=struct)
        good = np.unique(lab[opening & liquid])
        trapped = liquid & ~np.isin(lab, good[good > 0])
        close = None
        # the cell the front stands in: filled by the share of its own filling time that has passed
        # (it began when its last-filled neighbour was complete)
        prev = ndimage.maximum_filter(np.where(body_, arrival, -np.inf), footprint=struct, mode="constant",
                                      cval=-np.inf)
        front = liquid & np.isfinite(prev)
        nxt = np.where(reached, arrival, np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(front, np.clip((t_drain - prev) / (nxt - prev), 0.0, 1.0), 0.0)
        share = np.where(front & ~np.isfinite(nxt), 0.5, share)
        fill = np.where(body_, 1.0, share)
    # wall thickness as cast, read at the mold face: full cells in to the front, plus the front cell's share
    thick = {}
    if t_drain is not None and (cavity & ~body_).any():
        depth, idx = ndimage.distance_transform_edt(np.pad(body_ | ~cavity, 1, constant_values=True),
                                                    return_indices=True)
        depth = depth[1:-1, 1:-1, 1:-1]
        near = tuple(i[1:-1, 1:-1, 1:-1] - 1 for i in idx)                 # the nearest cell that is not wall
        face = body_ & ndimage.binary_dilation(np.pad(~cavity, 1, constant_values=True))[1:-1, 1:-1, 1:-1]
        face &= ~opening
        if face.any():
            nc = tuple(np.clip(n_[face], 0, sh - 1) for n_, sh in zip(near, cavity.shape))
            w = (depth[face] + fill[nc]) * pitch
            thick = {"min": float(w.min()), "p5": float(np.percentile(w, 5)), "mean": float(w.mean()),
                     "p95": float(np.percentile(w, 95)), "max": float(w.max())}
    nominal = wall_thickness(t_drain, b, plaster, pressure) if t_drain is not None else float("nan")
    return SlipCast(cavity=cavity, pitch=pitch, arrival=arrival, body=body_, fill=fill, trapped=trapped,
                    starved=starved, opening=opening, drain_time=t_drain, close_time=close, nominal=nominal, thickness=thick,
                    seconds=_time.time() - t_wall)


# --- kiln sag and strength --------------------------------------------------------

def creep_compliance(b: Body) -> float:
    """Phi = integral of dt / (uniaxial viscosity) over the firing, 1/Pa,
    from the pyroplastic index: a bar of thickness h on a span L sags
    S = (5/32) rho g L^4 Phi / h^2 (the elastic beam with E -> 1/Phi), and
    PI = 4 h^2 S / (3 L^4), so Phi = 24 PI / (5 rho g)."""
    return 24.0 * (b.sag_index * 100.0) / (5.0 * b.fired_density * G0)


def _self_weight(solid, pitch, weight, modulus, nu, up, support=None):
    """Displacement (n_nodes, 3) and element stress of a voxel body under
    its own weight (N/mm3), resting on `support` (bool grid of cells whose
    lower face sits on the setter; None: the lowest layer of the body)."""
    from .domain import gravity_force
    from .fem import HexMesh, Solver, Stiffness

    ax, sg = up
    ijk = np.argwhere(solid)
    mesh = HexMesh(ijk, pitch, np.zeros(3))
    h = ijk[:, ax] * sg
    if support is None:
        support = np.zeros(solid.shape, bool)
        support[tuple(ijk[h == h.min()].T)] = True
    sup = np.argwhere(support & solid)
    # the nodes on the lower face of each supported cell
    low = 0 if sg > 0 else 1
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    corners = corners[corners[:, ax] == low]
    nodes = np.unique(mesh.node_id((sup[:, None, :] + corners[None, :, :]).reshape(-1, 3)))
    nodes = nodes[nodes >= 0]
    fixed = [3 * nodes + ax]
    # hold the slide and spin in the setter's plane at the support's centroid (the part is free to
    # shrink across the setter): fix the node nearest the middle in both, and one more in one
    lat = [d for d in range(3) if d != ax]
    xyz = mesh.node_xyz[nodes]
    c = xyz.mean(0)
    n0 = nodes[np.argmin(((xyz - c) ** 2).sum(1))]
    far = nodes[np.argmax(np.abs(xyz[:, lat[0]] - mesh.node_xyz[n0, lat[0]]))]
    fixed += [np.array([3 * n0 + lat[0], 3 * n0 + lat[1], 3 * far + lat[1]])]
    fixed_ = np.zeros(mesh.n_dof, bool)
    fixed_[np.concatenate(fixed)] = True
    fixed = fixed_
    st = Stiffness(mesh, nu, fixed)
    g = np.zeros(3)
    g[ax] = -sg
    f = gravity_force(mesh, np.full(mesh.n_elem, weight), g, fixed)
    k = st.assemble(np.full(mesh.n_elem, modulus))
    u = Solver(mesh, rtol=1e-7, maxiter=4000).solve(k, f)
    return mesh, u.reshape(-1, 3), st.stress(u, modulus)


def _principal_max(stress):
    s = np.zeros((len(stress), 3, 3))
    s[:, 0, 0], s[:, 1, 1], s[:, 2, 2] = stress[:, 0], stress[:, 1], stress[:, 2]
    s[:, 1, 2] = s[:, 2, 1] = stress[:, 3]
    s[:, 0, 2] = s[:, 2, 0] = stress[:, 4]
    s[:, 0, 1] = s[:, 1, 0] = stress[:, 5]
    return np.linalg.eigvalsh(s)[:, -1]


def sag(solid, pitch: float, b: Body, up=(2, 1), support=None) -> dict:
    """Kiln sag of the fired shape (bool grid at `pitch` mm, fired size)
    under its own weight: a viscous body under a steady load deforms like
    an elastic one of modulus 1 / Phi (`creep_compliance`), with the
    Poisson's ratio of a sintering compact. Returns the peak sag (mm), the
    node displacements and the mesh."""
    phi = creep_compliance(b)                     # 1/Pa
    modulus = 1.0 / phi * 1e-6                    # MPa
    weight = b.fired_density * G0 * 1e-9          # N/mm3
    mesh, u, _ = _self_weight(solid, pitch, weight, modulus, 0.35, up, support)
    mag = np.linalg.norm(u, axis=1)
    ax, sg = up
    return {"max": float(mag.max()), "maxDown": float((-u[:, ax] * sg).max()), "displacement": u, "mesh": mesh,
            "effectiveModulus": modulus, "creepCompliance": phi}


def effective_volume(volume: float, m: float, loading: str = "3pt") -> float:
    """Effective (uniformly stressed) volume of a bend bar: V / (2 (m+1)^2)
    in three-point bending, V (m+2) / (4 (m+1)^2) in four-point quarter-span."""
    if loading == "3pt":
        return volume / (2.0 * (m + 1.0) ** 2)
    return volume * (m + 2.0) / (4.0 * (m + 1.0) ** 2)


def failure_probability(principal, volumes, strength: float, m: float, test_volume: float) -> float:
    """Weibull's weakest link: P = 1 - exp(-sum (sigma_1 / sigma_0)^m V / V_0)
    over the cells in tension, sigma_0 the characteristic strength measured
    on an effective volume V_0 (mm3). `strength` is the mean modulus of
    rupture; the characteristic strength is mean / Gamma(1 + 1/m)."""
    s0 = strength / math.gamma(1.0 + 1.0 / m)
    s = np.maximum(np.asarray(principal, float), 0.0)
    risk = float(((s / s0) ** m * np.asarray(volumes, float) / test_volume).sum())
    return 1.0 - math.exp(-risk)


# --- the report --------------------------------------------------------------------

def slipcast_part(mesh, body_name="porcelain", *, wall: float | None = 4.0, solid: bool = False,
                  pitch: float | None = None, cells: int = 150_000, up="+z", plaster: Plaster = PLASTER,
                  casts: int = 1, check_sag: bool = True, pressure: float = 0.0):
    """Slip cast a part: `mesh` is the fired part's outside (mm), `wall` its
    fired wall thickness (drain casting) or solid=True.

    Returns (report, fields): the cast time, the wall the real shape gets
    (corners thicker), slip trapped behind necks, the water the mold takes
    and the plaster that needs, both shrinkages and the mold scale, the
    kiln sag and the green and fired self-weight checks."""
    from .casting import _direction, grid_pitch
    from .verify import voxelize

    b = body(body_name)
    upd = _direction(up)
    sh = shrinkage(b)
    scale = sh["moldScale"]
    p = float(pitch) if pitch else grid_pitch(mesh, cells)
    if not solid and wall is not None and p > wall / 2.5:
        p = wall / 2.5                                  # at least two and a half cells through the wall
    cavity, org = voxelize(mesh, p)
    p_mold = p * scale                                  # the same grid at mold size
    wall_cast = None if solid or wall is None else wall * scale
    c = cast(cavity, p_mold, b, plaster, wall=wall_cast, solid=solid, up=upd, pressure=pressure)
    v_cell = p ** 3
    warnings = []
    report = {"body": b.to_dict(), "plaster": {"suction": plaster.suction, "permeability": plaster.permeability,
                                               "porosity": plaster.porosity},
              "pitch": p,
              "shrinkage": {k: sh[k] for k in ("drying", "firing", "total", "moldScale")},
              "casting": {"rate": cast_rate(b, plaster, pressure) * 1e6 * 60.0,        # mm2/min
                          "moldResistanceShare": mold_share(b, plaster),
                          "solid": bool(solid)}}
    rc = report["casting"]
    if solid:
        rc["time"] = c.close_time
        rc["starvedVolume"] = float(c.starved.sum() * v_cell)
        slip_volume = float(cavity.sum() * v_cell * scale ** 3 * b.cake_packing / b.solids)
        fired = cavity
        if c.starved.sum() > 0.002 * cavity.sum():
            warnings.append(f"{100 * c.starved.sum() / cavity.sum():.1f}% of the part fills after its supply of "
                            "slip has closed off (a centre void): thin the section or cast it hollow")
        wet = mold_wetting(2.0 * float(ndimage.distance_transform_edt(cavity).max()) * p_mold / 2.0, b, plaster)
    else:
        rc["time"] = c.drain_time
        rc["wallAsCast"] = wall_cast
        rc["wallFired"] = wall
        rc["wall"] = {k: v / scale for k, v in c.thickness.items()}       # fired mm
        rc["trappedVolume"] = float(c.trapped.sum() * v_cell)
        slip_volume = float(cavity.sum() * v_cell * scale ** 3)
        fired = c.body | c.trapped
        if c.trapped.sum() > 0.002 * cavity.sum():
            warnings.append(f"{c.trapped.sum() * v_cell:.0f} mm^3 of slip is cut off from the opening when the "
                            "mold is drained (a neck closes first): it stays in the part")
        if c.thickness and c.thickness["max"] > 1.6 * c.nominal:
            warnings.append(f"the wall reaches {c.thickness['max'] / scale:.1f} mm in the corners against "
                            f"{wall:.1f} mm on the flats: thick corners dry and fire slower than the wall")
        wet = mold_wetting(wall_cast, b, plaster)
    rc["slipVolume"] = slip_volume * 1e-6                                               # litres
    rc["slipMass"] = slip_volume * 1e-9 * b.specific_gravity * 1000.0                   # kg
    v_fired = float((c.fill.sum() + c.trapped.sum()) * v_cell) if not solid else float(fired.sum() * v_cell)
    fired_mass = v_fired * 1e-9 * b.fired_density * 1e3                                 # g
    report["part"] = {"firedVolume": v_fired, "firedMass": fired_mass,
                      "greenMass": fired_mass / (1.0 - b.loss_on_ignition)}
    report["mold"] = {"scale": scale, "waterPerCast": wet["waterPerArea"], "wetDepthPerCast": wet["wetDepth"],
                      "casts": int(casts), "minWall": float(casts) * wet["wetDepth"] * 2.0,
                      "note": "minWall: twice the plaster the water of that many casts fills, so the wetted "
                              "zone stays inside the mold and the suction holds"}
    fields = {"cast": c, "cavity": cavity, "origin": org, "pitch": p, "fired": fired, "scale": scale}
    if check_sag and fired.any():
        try:
            sg_ = sag(fired, p, b, upd)
            weight_f = b.fired_density * G0 * 1e-9
            mesh_f, u, stress = sg_["mesh"], sg_["displacement"], None
            # stresses under self-weight do not depend on the modulus: one more solve at the fired
            # stiffness reads them (and the same field scales to the green part)
            _, _, stress = _self_weight(fired, p, weight_f, b.fired_modulus, b.poisson, upd)
            s1 = _principal_max(stress)
            vol = np.full(len(s1), v_cell)
            # green, as demolded: the same shape at mold size, wet density; stress scales with weight x size
            rho_wet = b.cake_packing * b.particle_density + (1 - b.cake_packing) * 1000.0
            k_green = (rho_wet / b.fired_density) * scale
            report["kiln"] = {"sag": sg_["max"], "sagShare": sg_["max"] / float(np.ptp(np.argwhere(fired), 0).max()
                                                                                  * p + p),
                              "effectiveModulus": sg_["effectiveModulus"], "sagIndex": b.sag_index}
            report["strength"] = {
                "firedSelfWeightStress": float(s1.max()),
                "firedSelfWeightFailure": failure_probability(s1, vol, b.fired_strength, b.weibull, b.test_volume),
                "greenSelfWeightStress": float(s1.max() * k_green),
                "greenDemoldFailure": failure_probability(s1 * k_green, vol * scale ** 3, b.leather_strength,
                                                          b.green_weibull, b.test_volume),
                "firedStrength": b.fired_strength, "leatherStrength": b.leather_strength}
            fields["sag"] = sg_
            if report["kiln"]["sagShare"] > 0.01:
                warnings.append(f"the part sags {sg_['max']:.1f} mm in the kiln "
                                f"({100 * report['kiln']['sagShare']:.1f}% of its size): support it on a setter "
                                "or thicken the spans")
            if report["strength"]["greenDemoldFailure"] > 0.05:
                warnings.append(f"{100 * report['strength']['greenDemoldFailure']:.0f}% chance the green part "
                                "breaks under its own weight when it comes out of the mold")
        except Exception as e:  # noqa: BLE001 - a body in pieces has no single self-weight solve
            warnings.append(f"kiln sag not computed: {e}")
    report["warnings"] = warnings
    # thick corners are a note on drying; everything else is a part that does not come out right
    report["ok"] = not [w for w in warnings if not w.startswith("the wall reaches")]
    return report, fields
