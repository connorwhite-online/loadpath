"""
Metal casting: whether a part will cast sound, and what it takes to make it.

The `casting` study field already keeps undercuts out along the mold's draw
and builds draft into the delivered surface (manufacture.py). That is the
mold's geometry. This module is the metal: how the part freezes, where the
last liquid ends up, whether anything feeds it, how thin a wall the metal
will run into, and how much smaller the casting comes out than the pattern.
Each answer comes from the physics of the process, with the alloy's and the
mold's measured properties as the only inputs.

**Solidification** (`solidify`). Transient heat conduction with latent heat
on a voxel grid of the casting *and its mold*, by the enthalpy method: each
cell carries its enthalpy, the temperature and the solid fraction follow
from the alloy's enthalpy curve, and heat moves between neighbours through
face conductances (the two half-cells in series, plus the contact
resistance 1/h where metal meets mold). The mold is solved, not replaced by
a boundary condition: a sand mold saturates in a re-entrant corner and
draws heat faster round an outside one, which is what makes hot spots and
is exactly what a flat-wall rule (Chvorinov's) leaves out. On a flat plate
the solver gives Chvorinov's rule back, with the mold constant the
properties imply (tests/test_casting.py checks both against the analytic
solution). Explicit time stepping in a compiled kernel: the method conserves
energy exactly and takes an isothermal eutectic without special handling.

**Feeding and shrinkage** (inside `solidify`). Metal shrinks as it freezes
(`shrinkage`, a few percent by volume). While liquid is connected, that
volume is made up from the highest free liquid surface of the connected
pool, which sinks: a riser's pipe. Once a pool is cut off from any riser by
solid, its own top empties instead: a shrinkage cavity inside the part. The
solver follows this by mass conservation at every recorded step: the liquid
regions (cells not yet solid, connected), the volume they lost since the
last step, taken from the highest drainable cells of the same region
(`feedLimit`: liquid moves in bulk until the dendrites lock up, at a solid
fraction that is an alloy property). What is left is a void map: the pipe
in each riser, and any cavity in the part with its volume and position.
A riser is big enough when the part has no void, and that is how risers are
sized here (`design_risers`): by solving the casting with the riser on it.

**Microporosity** (`pore_fraction`). Late in freezing the liquid has to be
drawn through the dendrite network, and the pressure it loses on the way
(Darcy flow through a mush whose permeability follows the arm spacing) can
fall to where a pore opens. Carlson & Beckermann (2009) integrate that
pressure drop over the mushy zone: the dimensionless Niyama number
Ny* = C_lambda G / Tdot^(5/6) * sqrt(dP_cr / (mu beta dT_f)) gives the
liquid fraction at which the pore forms, and the pore volume is the
shrinkage of what was left. G and Tdot are read from the solve at
solidus + 10% of the freezing range, as in the paper.

**Filling** (`fluidity_length`, `fill`). A stream running into a cold mold
stops when its tip has lost its superheat and enough latent heat to lock
(the coherency fraction). The time that takes in a channel of a given
thickness is the same conduction problem, solved in closed form for a mold
wall with a contact resistance; the stream's speed comes from the metal
head less the surface-tension back pressure 2 gamma / t and wall friction.
Length = speed x time (Flemings). `fill` integrates the share of that
length spent along the cheapest path from the gate to every cell, through
the wall thickness the path actually runs in: over 1, the cell misruns.
The thinnest wall the head can enter at all is the capillary limit,
t = 2 gamma / (rho g h).

**Contraction** (`contraction`): the solid casting shrinks from the solidus
to room temperature by the integral of its expansion coefficient. The
pattern (or printed mold) is made larger by that.

**Gating** (`gating`): the pour must finish inside the fluidity of the
thinnest wall; the sprue is sized from that flow at the speed the head
gives (and tapered so the falling stream stays full), and the ingates are
wide enough that metal enters under Campbell's critical velocity, where
surface tension can still hold the front together and the oxide skin is
not folded in.

Properties: `ALLOYS` and `MOLDS`, every value with its source (`SOURCES`).
The mold's conductivity and heat capacity rise with temperature; the solve
uses their means over the range the mold actually spans (ambient to the
metal's liquidus), which is the exact steady-conduction average.

Not modelled: mold filling as a flow solve (fill is the thermal budget along
a path, not a free-surface simulation), gas porosity, hot tearing, residual
stress, and the graphite expansion that lets cast irons feed themselves.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage, optimize, special

try:
    import numba as nb
except ImportError:  # pragma: no cover - numba is a dependency; the numpy path without it
    nb = None

__all__ = ["CastAlloy", "MoldMaterial", "CastSetup", "Solidification", "ALLOYS", "MOLDS", "PROCESSES", "SOURCES",
           "alloy", "setup", "solidify", "plate_freeze_time", "chvorinov_constant", "schwarz_front", "niyama_star", "pore_fraction",
           "fluidity_length", "capillary_wall", "min_wall", "local_thickness", "fill", "contraction", "gating",
           "design_risers", "castability", "cast_part"]

G0 = 9.81
P_ATM = 101325.0
SIGMA_SB = 5.670374e-8

SOURCES = {
    "carlson-2009": "Carlson & Beckermann 2009, Prediction of shrinkage pore volume fraction using a dimensionless "
                    "Niyama criterion, Metall. Mater. Trans. A 40A:163-175, Tables II and III "
                    "(https://beckermann.lab.uiowa.edu/sites/beckermann.lab.uiowa.edu/files/2023-10/NiyamaCarlson.pdf)",
    "ou-2002": "Ou, Carlson, Hardin & Beckermann 2002, Development of new feeding-distance rules using casting "
               "simulation: Part II, Metall. Mater. Trans. B 33B:741-755 "
               "(https://beckermann.lab.uiowa.edu/sites/beckermann.lab.uiowa.edu/files/2023-10/FeedingPart2.pdf)",
    "mills-2002": "Mills 2002, Recommended Values of Thermophysical Properties for Selected Commercial Alloys, "
                  "Woodhead (values as commonly quoted from it; not re-read for this card)",
    "asm-15": "ASM Handbook vol. 15, Casting (typical values; not re-read for this card)",
    "poirier-geiger": "Poirier & Geiger, Transport Phenomena in Materials Processing, TMS 1994: silica sand "
                      "k 0.61 W/(m K), 1500 kg/m3, 1160 J/(kg K) for aluminium castings",
    "sand-conductivity": "Krajewski et al., thermal conductivity of bentonite-bonded silica molding sand by the "
                         "casting method, Archives of Foundry Engineering "
                         "(https://bibliotekanauki.pl/articles/380747.pdf)",
    "flemings-1974": "Flemings 1974, Solidification Processing, McGraw-Hill: fluidity as speed x time to the "
                     "coherency fraction at the stream tip",
    "campbell": "Campbell 2015, Complete Casting Handbook, 2nd ed., Butterworth-Heinemann: critical ingate "
                "velocity near 0.5 m/s from the balance of inertia and surface tension; feeding rules",
    "copper-org": "Copper Development Association alloy data sheet (copper.org), C87300 silicon bronze",
    "estimate": "estimate: stated in the note",
}


# --- alloys -----------------------------------------------------------------

@dataclass(frozen=True)
class CastAlloy:
    """A casting alloy's thermophysical card. SI units, temperatures in C.

    fraction: the solid fraction against temperature, as [(T, fs)] from the
    liquidus (fs 0) down to the solidus (fs 1); a vertical run is a eutectic.
    shrinkage: beta = (rho_s - rho_l) / rho_l over freezing.
    feed_limit: solid fraction past which liquid no longer moves in bulk.
    coherency: solid fraction at which a flowing stream's tip locks.
    sdas: C_lambda in lambda_2 = C_lambda Tdot^(-1/3), m (K/s)^(1/3).
    cte: mean linear expansion from room temperature to the solidus, 1/K.
    pore_fit: Carlson & Beckermann's Table III rows (Ny* upper bound, A, B, C, D):
    pores % = min(beta', -A log10 Ny* + B + C Ny*^-D)."""
    name: str
    description: str
    liquidus: float
    solidus: float
    latent: float
    density: float            # liquid, at the liquidus, kg/m3
    specific_heat: float      # J/(kg K), mean over the freezing range
    conductivity: float       # W/(m K), solid near the solidus
    viscosity: float          # Pa s
    surface_tension: float    # N/m
    shrinkage: float
    cte: float
    fraction: tuple
    feed_limit: float = 0.7
    coherency: float = 0.3
    sdas: float = 40e-6
    superheat: float = 80.0
    pore_fit: tuple = ()
    notes: dict = field(default_factory=dict)

    def curve(self):
        """(T ascending, fs) nodes with every run made strictly monotone
        (a eutectic gets 0.02 K, so the enthalpy curve can be inverted)."""
        pts = sorted(self.fraction, key=lambda p: (-p[0], p[1]))
        T = np.array([p[0] for p in pts], float)
        f = np.array([p[1] for p in pts], float)
        for i in range(1, len(T)):
            if T[i] > T[i - 1] - 0.02:
                T[i] = T[i - 1] - 0.02
        return T[::-1].copy(), f[::-1].copy()

    def solid_fraction(self, T):
        Tn, fn = self.curve()
        return np.interp(T, Tn, fn, left=1.0, right=0.0)

    def enthalpy_table(self):
        """Volumetric enthalpy (J/m3, zero at 0 C fully solid) against
        temperature: nodes (H ascending, T, fs)."""
        Tn, fn = self.curve()
        T = np.concatenate([[-273.0], Tn, [Tn[-1] + 3000.0]])
        f = np.concatenate([[1.0], fn, [0.0]])
        H = self.density * (self.specific_heat * T + self.latent * (1.0 - f))
        return H, T, f

    def beta_prime(self):
        return self.shrinkage / (1.0 + self.shrinkage)

    def to_dict(self):
        return {"name": self.name, "description": self.description, "liquidus": self.liquidus,
                "solidus": self.solidus, "latentHeat": self.latent, "density": self.density,
                "specificHeat": self.specific_heat, "conductivity": self.conductivity,
                "viscosity": self.viscosity, "surfaceTension": self.surface_tension, "shrinkage": self.shrinkage,
                "cte": self.cte, "feedLimit": self.feed_limit, "coherency": self.coherency,
                "sdasConstant": self.sdas, "superheat": self.superheat, "sources": dict(self.notes)}


def _scheil(t_liq, t_melt, k, t_eut, n=8):
    """Scheil solid fraction of a binary with partition coefficient k, from
    the liquidus down to the eutectic: fs = 1 - ((Tm - T)/(Tm - TL))^(1/(k-1))."""
    T = np.linspace(t_liq, t_eut, n)
    fs = 1.0 - ((t_melt - T) / (t_melt - t_liq)) ** (1.0 / (k - 1.0))
    return [(float(a), float(b)) for a, b in zip(T, fs)]


ALLOYS = {
    "a356": CastAlloy(
        "a356", "A356 (Al-7Si-0.3Mg): sand, permanent mold and investment",
        liquidus=616.0, solidus=550.7, latent=425e3, density=2400.0, specific_heat=1120.0, conductivity=150.0,
        viscosity=1.58e-3, surface_tension=0.87, shrinkage=0.0589, cte=25.0e-6,
        # Scheil for Al-7Si (k_Si 0.13, Tm 660 C) to the Al-Si eutectic at 577 C, where about half is
        # still liquid; the eutectic freezes over a few kelvin, the last phases (Mg2Si, Fe) down to the
        # solidus of Carlson's JMatPro curve
        fraction=tuple(_scheil(616.0, 660.0, 0.13, 577.0) + [(574.0, 0.93), (550.7, 1.0)]),
        feed_limit=0.7, coherency=0.3, sdas=40.9e-6, superheat=90.0,
        pore_fit=((1.43, 2.068, 3.160, 0.0, 0.0), (18.0, 0.0, 0.0, 4.024, 0.9786),
                  (math.inf, 0.0, 0.0, 7.771, 1.206)),
        notes={"liquidus, solidus (liquidus - freezing range), viscosity, shrinkage, sdas, pore_fit": "carlson-2009",
               "latent, density, specific_heat, conductivity, surface_tension": "mills-2002",
               "cte": "asm-15 (mean 20 C to solidus; gives the 1.3% patternmaker's contraction)",
               "fraction": "estimate: Scheil with k 0.13 to the 577 C eutectic",
               "feed_limit, coherency": "campbell (mass feeding ends near 0.7 solid; flow stops near 0.3)"}),
    "a380": CastAlloy(
        "a380", "A380 (Al-8.5Si-3.5Cu): high-pressure die casting",
        liquidus=595.0, solidus=540.0, latent=389e3, density=2500.0, specific_heat=1100.0, conductivity=110.0,
        viscosity=1.6e-3, surface_tension=0.85, shrinkage=0.055, cte=24.5e-6,
        fraction=tuple(_scheil(595.0, 660.0, 0.13, 568.0) + [(562.0, 0.90), (540.0, 1.0)]),
        feed_limit=0.7, coherency=0.3, sdas=40.9e-6, superheat=65.0,
        notes={"liquidus, solidus, latent, conductivity": "asm-15",
               "density, specific_heat, viscosity, surface_tension, shrinkage": "estimate: as A356, less silicon "
               "eutectic, more copper",
               "sdas": "carlson-2009 (A356's constant)",
               "fraction": "estimate: Scheil to the Al-Si-Cu eutectic valley near 568 C"}),
    "wcb": CastAlloy(
        "wcb", "Plain carbon steel (ASTM A216 WCB, AISI 1025): sand and investment",
        liquidus=1509.0, solidus=1438.2, latent=270e3, density=7000.0, specific_heat=750.0, conductivity=32.0,
        viscosity=5.63e-3, surface_tension=1.7, shrinkage=0.0449, cte=14.5e-6,
        # the paper's WCB curve (from the IDS solidification code) has a long tail: the last few percent
        # of liquid freeze over a third of the range. Its pore-fraction fit encodes that curve
        # (Ny*^2 = I(g_l), eq. 9), and below 30% liquid it can be read back out of the fit; above,
        # interpolated to the liquidus.
        fraction=tuple((1438.2 + 70.8 * th, 1.0 - gl) for gl, th in
                       ((1.0, 1.0), (0.7, 0.95), (0.5, 0.87), (0.3, 0.689), (0.2, 0.509), (0.1, 0.341),
                        (0.05, 0.244), (0.02, 0.159), (0.01, 0.114), (0.0, 0.0))),
        feed_limit=0.7, coherency=0.3, sdas=144e-6, superheat=60.0,
        pore_fit=((28.2, 1.654, 3.052, 0.0, 0.0), (math.inf, 0.0, 0.0, 43.05, 1.254)),
        notes={"liquidus, solidus, viscosity, shrinkage, sdas, pore_fit": "carlson-2009",
               "latent, density, specific_heat, conductivity, surface_tension": "mills-2002",
               "cte": "asm-15 (mean 20 C to solidus; gives the 2% patternmaker's contraction)",
               "fraction": "carlson-2009 (recovered from the Table III fit below 30% liquid)"}),
    "silicon-bronze": CastAlloy(
        "silicon-bronze", "Silicon bronze (C87300, Everdur): art and marine castings, sand and investment",
        liquidus=916.0, solidus=821.0, latent=205e3, density=7700.0, specific_heat=440.0, conductivity=60.0,
        viscosity=4.0e-3, surface_tension=1.2, shrinkage=0.045, cte=19.0e-6,
        fraction=((916.0, 0.0), (821.0, 1.0)),
        feed_limit=0.7, coherency=0.3, sdas=60e-6, superheat=150.0,
        notes={"liquidus, solidus": "copper-org",
               "latent, viscosity, surface_tension, shrinkage": "estimate: pure copper's (205 kJ/kg, 4 mPa s, "
               "1.3 N/m less the silicon, 4.5%)",
               "density, specific_heat, conductivity": "estimate: copper alloys near 800 C",
               "cte": "copper-org (18 um/m K to 300 C, extended to the solidus; 1.5% contraction)",
               "sdas": "estimate: between the aluminium and steel constants",
               "fraction": "estimate: linear between liquidus and solidus"}),
}


def alloy(name) -> CastAlloy:
    """A card by name, or a dict of overrides on a named card
    ({"base": "a356", "liquidus": 612, ...}), or a CastAlloy."""
    if isinstance(name, CastAlloy):
        return name
    if isinstance(name, dict):
        base = alloy(name.get("base", "a356"))
        keys = {"liquidus": "liquidus", "solidus": "solidus", "latentHeat": "latent", "density": "density",
                "specificHeat": "specific_heat", "conductivity": "conductivity", "viscosity": "viscosity",
                "surfaceTension": "surface_tension", "shrinkage": "shrinkage", "cte": "cte",
                "feedLimit": "feed_limit", "coherency": "coherency", "sdasConstant": "sdas",
                "superheat": "superheat"}
        over = {}
        for k, v in name.items():
            if k in ("base", "name", "fraction"):
                continue
            if k not in keys:
                raise ValueError(f"casting alloy: unknown property '{k}' (one of {sorted(keys)})")
            over[keys[k]] = float(v)
        if "fraction" in name:
            over["fraction"] = tuple((float(a), float(b)) for a, b in name["fraction"])
            over["pore_fit"] = ()
        elif {"liquidus", "solidus"} & set(over):
            over["fraction"] = ((over.get("liquidus", base.liquidus), 0.0), (over.get("solidus", base.solidus), 1.0))
            over["pore_fit"] = ()
        return replace(base, name=str(name.get("name", base.name)), **over)
    key = str(name).lower().replace("_", "-")
    key = {"a356-t6": "a356", "a356.0": "a356", "aluminium": "a356", "aluminum": "a356", "a380.0": "a380",
           "steel": "wcb", "carbon-steel": "wcb", "1025": "wcb", "bronze": "silicon-bronze",
           "c87300": "silicon-bronze", "everdur": "silicon-bronze"}.get(key, key)
    if key not in ALLOYS:
        raise ValueError(f"casting alloy '{name}': one of {sorted(ALLOYS)}, or a dict of properties on a base")
    return ALLOYS[key]


# --- molds ------------------------------------------------------------------

@dataclass(frozen=True)
class MoldMaterial:
    """conductivity and specific_heat are [(T C, value)] tables; the solve
    uses their means over the temperatures the mold spans."""
    name: str
    description: str
    density: float
    conductivity: tuple
    specific_heat: tuple
    source: str = ""

    def mean(self, t0: float, t1: float):
        """(k, c) averaged over [t0, t1]: the Kirchhoff mean, which gives
        the exact steady flux for a conductivity that varies with temperature."""
        lo, hi = min(t0, t1), max(t0, t1)
        if hi - lo < 1.0:
            hi = lo + 1.0
        T = np.linspace(lo, hi, 65)
        out = []
        for tab in (self.conductivity, self.specific_heat):
            x = np.array([p[0] for p in tab], float)
            y = np.array([p[1] for p in tab], float)
            v = np.interp(T, x, y)
            out.append(float(((v[:-1] + v[1:]) * 0.5).mean()))
        return out[0], out[1]


MOLDS = {
    "silica-sand": MoldMaterial(
        "silica-sand", "Silica sand, clay or chemically bonded (green sand, furan no-bake), and binder-jet "
        "printed sand", 1500.0,
        conductivity=((25.0, 0.70), (700.0, 0.78), (1000.0, 1.05), (1300.0, 1.30), (1500.0, 1.50)),
        specific_heat=((25.0, 800.0), (600.0, 1120.0), (1500.0, 1250.0)),
        source="sand-conductivity: 0.7-0.8 W/(m K) from 100 to 700 C, rising to 1.3 above as radiation between "
               "grains takes over; mean over 25-616 C is 0.74 W/(m K), effusivity 1050 (poirier-geiger: 1030)"),
    "investment-shell": MoldMaterial(
        "investment-shell", "Ceramic investment shell (fused silica and aluminosilicate stucco)", 1900.0,
        conductivity=((25.0, 0.6), (1000.0, 1.0), (1500.0, 1.4)),
        specific_heat=((25.0, 750.0), (1000.0, 1150.0), (1500.0, 1200.0)),
        source="estimate: published shell conductivities run 0.5-1.5 W/(m K) with porosity and temperature"),
    "plaster-investment": MoldMaterial(
        "plaster-investment", "Gypsum-bonded block investment (flask casting from a printed or wax pattern), "
        "after burnout", 1150.0,
        conductivity=((25.0, 0.35), (700.0, 0.50), (1100.0, 0.60)),
        specific_heat=((25.0, 850.0), (700.0, 1050.0), (1100.0, 1100.0)),
        source="estimate: burnt-out gypsum-silica investment, about half the effusivity of sand"),
    "die-steel": MoldMaterial(
        "die-steel", "H13 tool steel die or permanent mold", 7700.0,
        conductivity=((25.0, 25.0), (600.0, 29.0)),
        specific_heat=((25.0, 460.0), (600.0, 600.0)),
        source="asm-15"),
}

# How each process meets the metal: its mold, the contact conductance h
# (W/m2 K), the mold's starting temperature, how thick the mold is (None:
# thick enough that its outside never matters) and the gate pressure a
# machine adds to the metal head (Pa).
PROCESSES = {
    "sand": {"mold": "silica-sand", "htc": 1000.0, "moldTemperature": 25.0, "shell": None, "pressure": 0.0,
             "pattern": "reusable",
             "note": "h 1000 W/(m2 K) as carlson-2009's sand molds; it matters little, the sand is the resistance"},
    "investment": {"mold": "investment-shell", "htc": 1000.0, "moldTemperature": None, "shell": 7.0,
                   "pressure": 0.0, "pattern": "expendable",
                   "note": "shell preheated: 0.55 x liquidus for aluminium, 0.65 for steel and bronze"},
    "plaster": {"mold": "plaster-investment", "htc": 1000.0, "moldTemperature": None, "shell": None,
                "pressure": 0.0, "pattern": "expendable",
                "note": "flask poured at burnout cool-down temperature"},
    "permanent": {"mold": "die-steel", "htc": 2500.0, "moldTemperature": 300.0, "shell": None, "pressure": 0.0,
                  "pattern": "die",
                  "note": "h 2500 W/(m2 K) as carlson-2009's coated steel mold"},
    "die": {"mold": "die-steel", "htc": 20000.0, "moldTemperature": 200.0, "shell": None, "pressure": 40e6,
            "pattern": "die",
            "note": "h 2e4 W/(m2 K): the intensified metal is pressed onto the die; 40 MPa at the gate"},
}


@dataclass
class CastSetup:
    alloy: CastAlloy
    mold: MoldMaterial
    process: str = "sand"
    pour_temperature: float = 0.0
    mold_temperature: float = 25.0
    ambient: float = 25.0
    htc: float = 1000.0
    shell: float | None = None          # mm of mold round the metal; None: thick
    head: float = 100.0                 # mm of metal above the top of the part (the sprue)
    pressure: float = 0.0               # Pa added at the gate by a machine
    up: tuple = (2, 1)                  # (axis, sign): which way is up in the mold

    def mold_k_c(self):
        return self.mold.mean(self.mold_temperature, self.alloy.liquidus)

    def effusivity(self):
        k, c = self.mold_k_c()
        return math.sqrt(k * self.mold.density * c)

    def to_dict(self):
        k, c = self.mold_k_c()
        return {"process": self.process, "alloy": self.alloy.name, "mold": self.mold.name,
                "pourTemperature": self.pour_temperature, "moldTemperature": self.mold_temperature,
                "interfaceConductance": self.htc, "shell": self.shell, "head": self.head,
                "pressure": self.pressure, "up": ("+" if self.up[1] > 0 else "-") + "xyz"[self.up[0]],
                "moldConductivity": k, "moldSpecificHeat": c, "moldDensity": self.mold.density}


def _direction(v, default=(2, 1)):
    if v is None:
        return default
    if isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], int):
        return (int(v[0]), 1 if v[1] > 0 else -1)
    s = str(v).strip().lower()
    sign = -1 if s.startswith("-") else 1
    ax = s.lstrip("+-")
    if ax not in ("x", "y", "z"):
        raise ValueError(f"direction '{v}': one of +x, -x, +y, -y, +z, -z")
    return ("xyz".index(ax), sign)


def setup(process: str = "sand", alloy_name="a356", *, mold=None, pour_temperature=None, mold_temperature=None,
          htc=None, shell="default", head=None, pressure=None, up=None, ambient=25.0) -> CastSetup:
    """A process with its usual mold and contact conductance; anything given overrides."""
    if process not in PROCESSES:
        raise ValueError(f"casting process '{process}': one of {sorted(PROCESSES)}")
    p = PROCESSES[process]
    a = alloy(alloy_name)
    m = MOLDS[mold or p["mold"]] if not isinstance(mold, MoldMaterial) else mold
    t_mold = mold_temperature
    if t_mold is None:
        t_mold = p["moldTemperature"]
    if t_mold is None:
        t_mold = round((0.55 if a.liquidus < 700 else 0.65) * a.liquidus)
    return CastSetup(alloy=a, mold=m, process=process,
                     pour_temperature=float(pour_temperature if pour_temperature is not None
                                            else a.liquidus + a.superheat),
                     mold_temperature=float(t_mold), ambient=float(ambient),
                     htc=float(htc if htc is not None else p["htc"]),
                     shell=(p["shell"] if shell == "default" else shell),
                     head=float(head if head is not None else 100.0),
                     pressure=float(pressure if pressure is not None else p["pressure"]),
                     up=_direction(up))


# --- closed-form conduction into a mold wall --------------------------------

def _wall_heat(tau, effusivity, htc, dT):
    """Heat (J/m2) a wall at dT above a semi-infinite mold gives up in time
    tau through a contact conductance h: the classical convective-surface
    solution, Q = dT e^2/h [exp(n^2) erfc(n) - 1 + 2n/sqrt(pi)], n = h sqrt(tau)/e."""
    n = htc * np.sqrt(tau) / effusivity
    return dT * effusivity ** 2 / htc * (special.erfcx(n) - 1.0 + 2.0 * n / math.sqrt(math.pi))


def _wall_time(q, effusivity, htc, dT):
    """Time for the wall to give up q J/m2 (the inverse of _wall_heat)."""
    if q <= 0:
        return 0.0
    hi = max((q / (htc * dT)) * 4.0, (q * math.sqrt(math.pi) / (2 * effusivity * dT)) ** 2 * 4.0, 1e-9)
    f = lambda t: _wall_heat(t, effusivity, htc, dT) - q  # noqa: E731
    while f(hi) < 0:
        hi *= 4.0
    return float(optimize.brentq(f, 0.0, hi, xtol=1e-12, rtol=1e-10))


def _release(a: CastAlloy, t_from: float, fraction: float = 1.0):
    """Heat a volume of alloy gives up from `t_from` to solid fraction
    `fraction` (J/m3), and the mean temperature it gives it up at (weighted
    by the heat: a eutectic alloy releases most of it at the eutectic)."""
    H, T, F = a.enthalpy_table()
    h1 = float(np.interp(t_from, T, H))
    h0 = float(np.interp(fraction, F[1:-1][::-1], H[1:-1][::-1]))
    hs = np.linspace(h0, h1, 257)
    return h1 - h0, float(np.interp(hs, H, T).mean())


def plate_freeze_time(thickness_mm: float, s: CastSetup, fraction: float = 1.0) -> float:
    """Seconds for a wide plate to reach solid fraction `fraction`, by a
    heat balance on the half plate against the closed-form mold wall: the
    plate gives up its superheat and that share of its latent heat, the
    mold surface sits at the heat-weighted mean temperature of the release.
    This is Chvorinov's rule with the contact resistance kept; the metal's
    own gradient is neglected (small against a sand mold's, not against a
    die's), so it is an estimate: `solidify` is the solve."""
    q, t_mean = _release(s.alloy, s.pour_temperature, fraction)
    return _wall_time(q * 0.5e-3 * thickness_mm, s.effusivity(), s.htc, t_mean - s.mold_temperature)


def chvorinov_constant(s: CastSetup) -> float:
    """B in t = B (V/A)^2, s/m2, for a mold that is all the resistance
    (h -> infinity): B = pi/4 (q / (T_f - T_0))^2 / (k rho c)_mold, q the
    heat released per volume and T_f the mean temperature it is released at."""
    q, t_mean = _release(s.alloy, s.pour_temperature)
    return math.pi / 4.0 * (q / (t_mean - s.mold_temperature)) ** 2 / s.effusivity() ** 2


def schwarz_front(a: CastAlloy, s: CastSetup, melting: float) -> float:
    """phi in S(t) = 2 phi sqrt(alpha t): Schwarz's exact solution for a
    pure metal poured at its melting point against a semi-infinite mold in
    perfect contact, phi exp(phi^2) (M + erf phi) = c (Tm - T0) / (L sqrt(pi)),
    M the ratio of the metal's effusivity to the mold's."""
    m = math.sqrt(a.conductivity * a.density * a.specific_heat) / s.effusivity()
    rhs = a.specific_heat * (melting - s.mold_temperature) / (a.latent * math.sqrt(math.pi))
    return float(optimize.brentq(lambda p: p * math.exp(p * p) * (m + math.erf(p)) - rhs, 1e-9, 5.0))


# --- the solidification solve ----------------------------------------------

MOLD, METAL, AMBIENT = 0, 1, 2


def _numba_ok():
    return nb is not None and os.environ.get("LOADPATH_CASTING", "").lower() != "numpy"


def _interp_table(h, tabH, tabT):
    return np.interp(h, tabH, tabT)


def _fast_numpy(H, T, Tsum, cells, nbr, g, ismetal, tabH, tabT, inv_rc, dt_dx, nsub):
    ok = nbr >= 0
    j = np.where(ok, nbr, 0)
    for _ in range(nsub):
        t = T[cells]
        q = (np.where(ok, g, 0.0) * (T[j] - t[:, None])).sum(1)
        H[cells] += dt_dx * q
        Tsum += t
        h = H[cells]
        T[cells] = np.where(ismetal, np.interp(h, tabH, tabT), h * inv_rc)


def _slow_numpy(H, T, Tstar, cells, nbr, g, inv_rc, dt_dx):
    ok = nbr >= 0
    j = np.where(ok, nbr, 0)
    q = (np.where(ok, g, 0.0) * (Tstar[j] - T[cells][:, None])).sum(1)
    H[cells] += dt_dx * q
    T[cells] = H[cells] * inv_rc


if nb is not None:

    @nb.njit(cache=True, parallel=True)
    def _fast_numba(H, T, Tsum, cells, nbr, g, ismetal, tabH, tabT, inv_rc, dt_dx, nsub):  # pragma: no cover
        n = cells.shape[0]
        m = tabH.shape[0]
        for _ in range(nsub):
            for c in nb.prange(n):
                i = cells[c]
                t = T[i]
                q = 0.0
                for f in range(6):
                    j = nbr[c, f]
                    if j >= 0:
                        q += g[c, f] * (T[j] - t)
                H[i] += dt_dx * q
                Tsum[c] += t
            for c in nb.prange(n):
                i = cells[c]
                h = H[i]
                if ismetal[c]:
                    lo = 0
                    hi = m - 1
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if tabH[mid] <= h:
                            lo = mid
                        else:
                            hi = mid
                    T[i] = tabT[lo] + (h - tabH[lo]) / (tabH[hi] - tabH[lo]) * (tabT[hi] - tabT[lo])
                else:
                    T[i] = h * inv_rc

    @nb.njit(cache=True, parallel=True)
    def _slow_numba(H, T, Tstar, cells, nbr, g, inv_rc, dt_dx):  # pragma: no cover
        n = cells.shape[0]
        for c in nb.prange(n):
            i = cells[c]
            t = T[i]
            q = 0.0
            for f in range(6):
                j = nbr[c, f]
                if j >= 0:
                    q += g[c, f] * (Tstar[j] - t)
            H[i] += dt_dx * q
        for c in nb.prange(n):
            i = cells[c]
            T[i] = H[i] * inv_rc


def _neighbours(cells, shape, gx, gy, gz):
    """For a list of flat cell indices: the six face neighbours' flat
    indices (-1 off the grid) and the face conductances to them."""
    ijk = np.stack(np.unravel_index(cells, shape), axis=1)
    strides = (shape[1] * shape[2], shape[2], 1)
    nbr = np.full((len(cells), 6), -1, np.int64)
    g = np.zeros((len(cells), 6))
    for d, gg in enumerate((gx, gy, gz)):
        lo_ok = ijk[:, d] > 0
        hi_ok = ijk[:, d] < shape[d] - 1
        nbr[lo_ok, 2 * d] = cells[lo_ok] - strides[d]
        nbr[hi_ok, 2 * d + 1] = cells[hi_ok] + strides[d]
        lo_idx = ijk[lo_ok].copy()
        lo_idx[:, d] -= 1
        g[lo_ok, 2 * d] = gg[tuple(lo_idx.T)]
        g[hi_ok, 2 * d + 1] = gg[tuple(ijk[hi_ok].T)]
    dead = g <= 0
    nbr[dead] = -1
    return nbr, g


@dataclass
class Solidification:
    """What `solidify` returns. Fields are on the grid it was given (the
    metal array's shape), NaN outside the metal.

    time            seconds to fully solid, per cell
    feed_time       seconds to the feed limit (bulk liquid gone)
    gradient        K/mm at solidus + 10% of the freezing range
    cooling_rate    K/s there
    niyama          gradient / sqrt(cooling rate), (K s)^0.5 / mm
    void            share of the cell emptied by shrinkage (pipe, cavities)
    pores           microporosity share from the dimensionless Niyama number
    sdas            secondary dendrite arm spacing, um
    """
    metal: np.ndarray
    pitch: float
    setup: CastSetup
    time: np.ndarray
    feed_time: np.ndarray
    gradient: np.ndarray
    cooling_rate: np.ndarray
    niyama: np.ndarray
    void: np.ndarray
    pores: np.ndarray
    sdas: np.ndarray
    total_time: float
    steps: int
    seconds: float
    energy_error: float
    complete: bool
    feeders: np.ndarray | None = None
    history: list = field(default_factory=list)

    @property
    def part(self):
        return self.metal if self.feeders is None else self.metal & ~self.feeders

    def cavities(self, min_share: float = 0.5, min_volume: float | None = None, where=None):
        """Shrinkage cavities: connected cells emptied by more than
        `min_share`, each {volume mm3, position (grid index, float),
        cells, lastToFreeze s}. `where` limits them to a mask (the part)."""
        m = (np.nan_to_num(self.void) >= min_share)
        if where is not None:
            m &= where
        lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
        out = []
        v = self.pitch ** 3
        for i in range(1, n + 1):
            sel = lab == i
            vol = float(self.void[sel].sum() * v)
            if min_volume is not None and vol < min_volume:
                continue
            idx = np.argwhere(sel)
            w = self.void[sel]
            out.append({"volume": vol, "position": (idx * w[:, None]).sum(0) / w.sum(), "cells": int(sel.sum()),
                        "lastToFreeze": float(np.nanmax(self.time[sel]))})
        return sorted(out, key=lambda c: -c["volume"])


def _faces(kind, k_cell, dx, htc, h_amb):
    """Face conductances (W/m2 K) along each axis: half cells in series,
    the contact resistance where metal meets mold, the surface coefficient
    where anything meets the ambient."""
    out = []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(None, -1)
        b[ax] = slice(1, None)
        a, b = tuple(a), tuple(b)
        ka, kb = kind[a], kind[b]
        r = np.zeros(ka.shape)
        for kk, kc in ((ka, k_cell[a]), (kb, k_cell[b])):
            r += np.where(kk == AMBIENT, 0.0, 0.5 * dx / np.maximum(kc, 1e-300))
        r += np.where(((ka == METAL) & (kb == MOLD)) | ((ka == MOLD) & (kb == METAL)), 1.0 / htc, 0.0)
        amb = (ka == AMBIENT) ^ (kb == AMBIENT)
        r += np.where(amb, 1.0 / h_amb, 0.0)
        g = np.where((ka == AMBIENT) & (kb == AMBIENT), 0.0, 1.0 / np.maximum(r, 1e-30))
        out.append(np.ascontiguousarray(g))
    return out


def _inscribed(metal, insulated=()):
    """Largest inscribed radius of the metal, in cells; a symmetry face of
    the grid mirrors the metal instead of bounding it."""
    names = ["x-", "x+", "y-", "y+", "z-", "z+"]
    n = int(max(metal.shape))
    m = metal
    for d in range(3):
        lo, hi = names[2 * d] in insulated, names[2 * d + 1] in insulated
        w = [(0, 0)] * 3
        w[d] = (n if lo else 1, n if hi else 1)
        m = np.pad(m, w, mode="constant") if not (lo or hi) else np.concatenate(
            [np.repeat(np.take(m, [0], axis=d), n, axis=d) if lo else np.zeros_like(np.take(m, [0], axis=d)),
             m,
             np.repeat(np.take(m, [-1], axis=d), n, axis=d) if hi else np.zeros_like(np.take(m, [0], axis=d))],
            axis=d)
    return float(ndimage.distance_transform_edt(m).max())


def _pad_cells(metal, pitch, s: CastSetup, max_cells, insulated=()):
    """Mold cells to add round the metal: three thermal penetration depths
    of the mold over the freezing time of the thickest section (so the
    outside of the grid never feels the casting), or the shell's thickness."""
    if s.shell is not None:
        return max(1, int(round(s.shell / pitch)))
    half = _inscribed(metal, insulated) * pitch            # mm, largest inscribed radius
    t = plate_freeze_time(2.0 * half, s)
    k, c = s.mold_k_c()
    depth = 3.0 * math.sqrt(k / (s.mold.density * c) * t) * 1e3     # mm
    n = int(math.ceil(depth / pitch))
    # no deeper than keeps the grid in budget
    while n > 3 and np.prod(np.asarray(metal.shape) + 2 * n) > max_cells:
        n -= 1
    return max(3, n)


def solidify(metal, pitch: float, s: CastSetup, *, feeders=None, open_top=None, pad=None, insulated=(),
             max_cells: int = 3_000_000, records: int = 400, max_time: float | None = None,
             on_progress=None, history: bool = False, ambient_htc: float | None = None, ambient=None,
             time_scale: float | None = None) -> Solidification:
    """Freeze the metal in its mold.

    metal      bool grid of metal cells (casting, risers, gates), pitch mm
    feeders    bool grid, the subset that is riser (reported apart)
    open_top   bool grid of metal cells whose `up` face is open to the air
               (an open riser's top): the column above them is ambient
    pad        mold cells added on each side, an int or six ints
               (x-, x+, y-, y+, z-, z+); None: sized from the physics
    insulated  faces of the grid that are symmetry planes ("x-", "z+", ...):
               no mold is added there and no heat crosses
    ambient    bool grid of cells held at the ambient temperature (air, or
               any fixed-temperature boundary), coupled through `ambient_htc`
    time_scale seconds the freezing is expected to take (sets the recording
               interval and the time limit); None: from the thickest section
    """
    t_wall = time.time()
    metal = np.asarray(metal, bool)
    if not metal.any():
        raise ValueError("solidify: no metal cells")
    a = s.alloy
    dx = pitch * 1e-3
    names = ["x-", "x+", "y-", "y+", "z-", "z+"]
    if pad is None:
        n = _pad_cells(metal, pitch, s, max_cells, insulated)
        pads = [n] * 6
    elif isinstance(pad, int):
        pads = [pad] * 6
    else:
        pads = [int(p) for p in pad]
    for f in insulated:
        pads[names.index(f)] = 0
    pw = [(pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5])]
    kind = np.pad(metal.astype(np.int8), pw)                 # METAL = 1, rest MOLD = 0
    core = tuple(slice(p[0], p[0] + n) for p, n in zip(pw, metal.shape))
    ax, sg = s.up
    if ambient is not None:
        kind[np.pad(np.asarray(ambient, bool), pw) & (kind != METAL)] = AMBIENT
    if s.shell is not None:
        # beyond the shell is air; a face the grid ends on (a symmetry plane) is not a surface
        edt = ndimage.distance_transform_edt(kind != METAL)
        kind[(kind == MOLD) & (edt > s.shell / pitch + 0.5)] = AMBIENT
    if open_top is not None and np.asarray(open_top).any():
        ot = np.pad(np.asarray(open_top, bool), pw)
        cols = np.moveaxis(ot, ax, -1)
        kcol = np.moveaxis(kind, ax, -1)
        if sg < 0:
            cols, kcol = cols[..., ::-1], kcol[..., ::-1]
        above = (np.cumsum(cols, axis=-1) - cols) > 0          # strictly above an open-top cell
        kcol[above & (kcol != METAL)] = AMBIENT
    k_mold, c_mold = s.mold_k_c()
    k_cell = np.where(kind == METAL, a.conductivity, k_mold)
    # air over an open surface or round a shell: natural convection and radiation, linearized at the
    # mean of the liquidus and the ambient (the shell's or the riser top's range of surface temperature)
    tm = 0.5 * (a.liquidus + s.ambient) + 273.15
    ta = s.ambient + 273.15
    h_amb = 10.0 + 0.8 * SIGMA_SB * (tm * tm + ta * ta) * (tm + ta)
    if ambient_htc is not None:
        h_amb = float(ambient_htc)
    gx, gy, gz = _faces(kind, k_cell, dx, s.htc, h_amb)
    tabH, tabT, tabF = a.enthalpy_table()
    rc_metal = a.density * a.specific_heat
    rc_mold = s.mold.density * c_mold
    shape = kind.shape
    kflat = kind.ravel()
    # Two rates. The metal (and the mold cells touching it) take the small explicit step its
    # conductivity needs; the rest of the mold, a hundred times less diffusive, takes one step for
    # every `sub` of those, against the fast cells' time-averaged temperature, so the heat that
    # crosses between the two sets is counted once and the same on both sides.
    fast3 = (kind == METAL) | ((kind == MOLD) & ndimage.binary_dilation(kind == METAL))
    fcells = np.nonzero(fast3.ravel())[0]
    scells = np.nonzero(((kind == MOLD) & ~fast3).ravel())[0]
    fnbr, fg = _neighbours(fcells, shape, gx, gy, gz)
    snbr, sg_ = _neighbours(scells, shape, gx, gy, gz)
    f_metal = kflat[fcells] == METAL
    f_rc = np.where(f_metal, rc_metal, rc_mold)
    dt = 0.9 * float((f_rc * dx / np.maximum(fg.sum(1), 1e-30)).min())
    if len(scells):
        dt_slow = 0.9 * float((rc_mold * dx / np.maximum(sg_.sum(1), 1e-30)).min())
        sub = int(max(1, min(math.floor(dt_slow / dt), 200)))
    else:
        sub = 1
    T3 = np.where(kind == METAL, s.pour_temperature, np.where(kind == MOLD, s.mold_temperature, s.ambient))
    T = np.ascontiguousarray(T3, float).ravel()
    H = np.where(kflat == METAL, np.interp(T, tabT, tabH), rc_mold * T)
    Tsum = np.zeros(len(fcells))

    mi = np.nonzero(kflat == METAL)[0]                         # flat indices of metal cells
    nm = len(mi)
    strides = (shape[1] * shape[2], shape[2], 1)
    ijk = np.stack(np.unravel_index(mi, shape), axis=1)
    # neighbours of each metal cell that are metal too (for the gradient)
    nbr = []
    for d in range(3):
        lo_ok = ijk[:, d] > 0
        hi_ok = ijk[:, d] < shape[d] - 1
        lo = np.where(lo_ok, mi - strides[d], mi)
        hi = np.where(hi_ok, mi + strides[d], mi)
        lo_ok &= kflat[lo] == METAL
        hi_ok &= kflat[hi] == METAL
        nbr.append((np.where(lo_ok, lo, mi), np.where(hi_ok, hi, mi), lo_ok.astype(float) + hi_ok.astype(float)))
    height = ijk[:, ax] * sg

    dT_f = a.liquidus - a.solidus
    t_eval = a.solidus + 0.10 * dT_f
    t_hi, t_lo = a.solidus + 0.15 * dT_f, a.solidus + 0.05 * dT_f
    h_feed = float(np.interp(a.feed_limit, tabF[1:-1][::-1], tabH[1:-1][::-1]))   # enthalpy at the feed limit
    h_solid = float(np.interp(a.solidus, tabT, tabH))
    nan = np.full(nm, np.nan)
    t_solid, t_feed, t_e, t_h, t_l, grad = (nan.copy() for _ in range(6))
    void = np.zeros(nm)
    dispersed = np.zeros(nm)
    Hm_prev = H[mi].copy()
    Tm_prev = T[mi].copy()
    fs_book = np.interp(Hm_prev, tabH, tabF)
    beta_p = a.beta_prime()

    # how long: the thickest section as a plate; slow steps per record from that
    t_guess = time_scale if time_scale is not None else plate_freeze_time(
        2.0 * _inscribed(metal, insulated) * pitch, s)
    t_guess = max(t_guess, 20 * dt * sub)
    chunk = int(max(1, round(t_guess / (dt * sub) / records)))
    limit = max_time if max_time is not None else 40.0 * t_guess
    use_nb = _numba_ok()
    fast = _fast_numba if use_nb else _fast_numpy
    slow = _slow_numba if use_nb else _slow_numpy
    live = kflat != AMBIENT
    e0 = float(H[live].sum())
    lost = 0.0
    now = 0.0
    steps = 0
    hist = []
    amb_faces = bool((kflat == AMBIENT).any())
    struct = ndimage.generate_binary_structure(3, 1)
    liquidish = np.zeros(shape, bool)

    def crossing(prev, cur, level, t0, t1, store):
        hit = np.isnan(store) & (cur <= level)
        if hit.any():
            w = np.clip((prev[hit] - level) / np.maximum(prev[hit] - cur[hit], 1e-12), 0.0, 1.0)
            store[hit] = t0 + w * (t1 - t0)
        return hit

    while True:
        for _ in range(chunk):
            Tsum[:] = 0.0
            fast(H, T, Tsum, fcells, fnbr, fg, f_metal, tabH, tabT, 1.0 / rc_mold, dt / dx, sub)
            if len(scells):
                Tstar = T.copy()
                Tstar[fcells] = Tsum / sub
                slow(H, T, Tstar, scells, snbr, sg_, 1.0 / rc_mold, dt * sub / dx)
        t0, now = now, now + chunk * sub * dt
        steps += chunk * sub
        Hm = H[mi]
        Tm = T[mi]
        crossing(Hm_prev, Hm, h_feed, t0, now, t_feed)
        crossing(Hm_prev, Hm, h_solid, t0, now, t_solid)
        crossing(Tm_prev, Tm, t_hi, t0, now, t_h)
        crossing(Tm_prev, Tm, t_lo, t0, now, t_l)
        hit = crossing(Tm_prev, Tm, t_eval, t0, now, t_e)
        if hit.any():
            g2 = np.zeros(int(hit.sum()))
            for lo, hi, cnt in nbr:
                g2 += ((T[hi[hit]] - T[lo[hit]]) / np.maximum(cnt[hit], 1.0)) ** 2
            grad[hit] = np.sqrt(g2) / pitch                      # K/mm
        fs = np.interp(Hm, tabH, tabF)

        # --- shrinkage bookkeeping: what froze since the last record is made up from the highest
        # drainable liquid of the region it froze in
        dfs = np.maximum(fs - fs_book, 0.0)
        nlab = 0
        if beta_p > 0:
            liquidish.ravel()[mi] = fs_book < 1.0
            lab3, nlab = ndimage.label(liquidish, structure=struct)
            lab = lab3.ravel()[mi]
        if nlab:
            need = np.bincount(lab, weights=dfs, minlength=nlab + 1) * beta_p      # in cell volumes
            order = np.lexsort((-height, lab))                                    # by region, highest first
            lab_o = lab[order]
            starts = np.searchsorted(lab_o, np.arange(1, nlab + 2))
            for r in np.nonzero(need[1:] > 0)[0] + 1:
                sel = order[starts[r - 1]:starts[r]]
                cap = np.where(fs[sel] < a.feed_limit, np.maximum(1.0 - fs[sel] - void[sel], 0.0), 0.0)
                tot = cap.sum()
                take = min(need[r], tot)
                if take > 0:
                    cum = np.cumsum(cap)
                    full = cum <= take
                    void[sel[full]] += cap[full]
                    j = int(full.sum())
                    if j < len(sel):
                        void[sel[j]] += take - (cum[j - 1] if j else 0.0)
                rest = need[r] - take
                if rest > 1e-12:
                    # no free liquid left in the region: the shrinkage stays where it happens
                    d = dfs[sel]
                    if d.sum() > 0:
                        dispersed[sel] += rest * d / d.sum()
        fs_book = fs
        if history:
            hist.append((now, float((1.0 - fs).mean())))
        Hm_prev, Tm_prev = Hm.copy(), Tm.copy()
        if on_progress is not None:
            on_progress({"stage": "solidify", "time": now, "liquid": float((1.0 - fs).mean())})
        if not np.isnan(t_solid).any() and not np.isnan(t_l).any():
            complete = True
            break
        if now >= limit:
            complete = False
            break
    e1 = float(H[live].sum())
    if not amb_faces:
        lost = abs(e1 - e0) / max(abs(e0), 1.0)

    def field3(v):
        out = np.full(shape, np.nan)
        out.ravel()[mi] = v
        return out[core]

    rate = (t_hi - t_lo) / np.maximum(t_l - t_h, 1e-9)           # K/s through the evaluation point
    with np.errstate(invalid="ignore", divide="ignore"):
        ny = grad / np.sqrt(rate)
    pores = pore_fraction(niyama_star(grad, rate, a, s), a) if a.shrinkage > 0 else np.zeros(nm)
    sdas = a.sdas * np.maximum(rate, 1e-9) ** (-1.0 / 3.0) * 1e6
    return Solidification(metal=metal, pitch=pitch, setup=s, time=field3(t_solid), feed_time=field3(t_feed),
                          gradient=field3(grad), cooling_rate=field3(rate), niyama=field3(ny),
                          void=field3(np.minimum(void + dispersed, 1.0)), pores=field3(pores), sdas=field3(sdas),
                          total_time=now, steps=steps, seconds=time.time() - t_wall, energy_error=lost,
                          complete=complete, feeders=None if feeders is None else np.asarray(feeders, bool),
                          history=hist)


# --- microporosity: the dimensionless Niyama criterion -----------------------

def niyama_star(gradient, cooling_rate, a: CastAlloy, s: CastSetup | None = None, pressure: float | None = None):
    """Ny* = C_lambda G Tdot^(-5/6) sqrt(dP_cr / (mu beta dT_f)) (Carlson &
    Beckermann 2009, eq. 12). gradient K/mm, cooling_rate K/s. dP_cr is the
    pressure on the liquid at the liquidus (atmosphere plus the metal head
    plus any machine pressure) with zero capillary pressure, the paper's base case."""
    p = pressure
    if p is None:
        p = P_ATM
        if s is not None:
            p += a.density * G0 * s.head * 1e-3 + s.pressure
    g = np.asarray(gradient, float) * 1e3
    r = np.maximum(np.asarray(cooling_rate, float), 1e-12)
    return a.sdas * g * r ** (-5.0 / 6.0) * math.sqrt(p / (a.viscosity * a.shrinkage * (a.liquidus - a.solidus)))


def _pore_integral(a: CastAlloy, n: int = 4000):
    """I(g_l,cr) = int_{g_l,cr}^1 180 (1 - g)^2 / g^2 (d theta / d g) dg on
    the alloy's own fraction curve (eq. 9): (g_l nodes ascending, I)."""
    Tn, fn = a.curve()
    theta = (Tn - a.solidus) / (a.liquidus - a.solidus)
    gl = 1.0 - fn                                   # ascending with T
    g = np.linspace(1e-4, 1.0, n)
    th = np.interp(g, gl, theta)
    dth = np.gradient(th, g)
    integrand = 180.0 * (1.0 - g) ** 2 / g ** 2 * dth
    # integrate from each g up to 1
    seg = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(g)
    I = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return g, I


def pore_fraction(ny_star, a: CastAlloy, fit: bool = True):
    """Shrinkage pore volume fraction from Ny*: the liquid fraction at which
    the pressure drop through the mush reaches dP_cr (Ny*^2 = I(g_l,cr)),
    times beta' (eq. 13). Uses the paper's fitted curve where the alloy has
    one (`fit`), the integral over the card's fraction curve otherwise."""
    x = np.asarray(ny_star, float)
    bp = a.beta_prime()
    if fit and a.pore_fit:
        out = np.zeros(x.shape)
        xs = np.maximum(x, 1e-9)
        done = np.zeros(x.shape, bool)
        for top, A, B, C, D in a.pore_fit:
            sel = ~done & (xs <= top)
            val = -A * np.log10(xs[sel]) + B if A or B else C * xs[sel] ** (-D)
            out[sel] = val
            done |= sel
        out = np.clip(out / 100.0, 0.0, bp)
    else:
        g, I = _pore_integral(a)
        gcr = np.interp(np.minimum(x, 1e30) ** 2, I[::-1], g[::-1], left=1.0, right=g[0])
        out = bp * np.where(x ** 2 >= I[0], 0.0, gcr)
    return np.where(np.isfinite(x), out, np.nan)


# --- filling ----------------------------------------------------------------

def capillary_wall(a: CastAlloy, head_mm: float, pressure: float = 0.0) -> float:
    """The thinnest wall (mm) the metal enters at all: the head must beat
    the surface-tension pressure of a non-wetting front between two walls,
    2 gamma / t."""
    p = a.density * G0 * head_mm * 1e-3 + pressure
    return 2.0 * a.surface_tension / max(p, 1e-9) * 1e3


def _stream_speed(a: CastAlloy, thickness, length, head, pressure):
    """Speed (m/s) of metal running in a plate channel `thickness` thick
    after `length` of it, under the head less the capillary back pressure,
    with wall friction (laminar 96/Re, turbulent Blasius) and one velocity
    head lost at entry."""
    dp = a.density * G0 * head + pressure - 2.0 * a.surface_tension / thickness
    if dp <= 0:
        return 0.0
    dh = 2.0 * thickness                    # hydraulic diameter of a wide slot
    v = math.sqrt(2.0 * dp / a.density)
    for _ in range(40):
        re = max(a.density * v * dh / a.viscosity, 1.0)
        f = 96.0 / re if re < 2300 else 0.316 / re ** 0.25
        v_new = math.sqrt(2.0 * dp / (a.density * (1.5 + f * length / dh)))
        if abs(v_new - v) < 1e-6 * v:
            break
        v = 0.5 * (v + v_new)
    return v


def _tip_time(a: CastAlloy, thickness, s: CastSetup):
    """Seconds for the tip of a stream in a channel `thickness` (m) thick to
    lose its superheat and the coherency share of its latent heat through
    both faces. The tip always runs over mold that has not been heated yet,
    so the mold wall under it stays at its starting temperature and the
    contact conductance h is the whole resistance (Flemings): the flux is
    h (T - T0), not the falling flux a standing casting sees."""
    q, t_mean = _release(a, s.pour_temperature, a.coherency)
    return q * 0.5 * thickness / (s.htc * (t_mean - s.mold_temperature))


def fluidity_length(thickness_mm: float, s: CastSetup, head_mm: float | None = None) -> float:
    """How far (mm) the alloy runs in a plate channel of this thickness
    before it stops: speed x the time the tip takes to reach the coherency
    fraction, with the speed falling as the channel behind it lengthens."""
    a = s.alloy
    t = thickness_mm * 1e-3
    head = (s.head if head_mm is None else head_mm) * 1e-3
    tau = _tip_time(a, t, s)
    if _stream_speed(a, t, 0.0, head, s.pressure) <= 0:
        return 0.0

    # the front at x moves at the speed friction over the length behind it allows
    def reach(tt):
        x = 0.0
        n = 60
        for _ in range(n):
            x += _stream_speed(a, t, x, head, s.pressure) * tt / n
        return x
    return reach(tau) * 1e3


def min_wall(flow_length_mm: float, s: CastSetup, head_mm: float | None = None) -> float:
    """The thinnest wall (mm) that fills over `flow_length_mm` from the gate."""
    lo = capillary_wall(s.alloy, s.head if head_mm is None else head_mm, s.pressure)
    f = lambda t: fluidity_length(t, s, head_mm) - flow_length_mm  # noqa: E731
    hi = max(lo * 1.05, 0.05)
    if f(hi) >= 0:
        return hi
    top = hi
    while f(top) < 0:
        top *= 1.6
        if top > 500:
            return float("inf")
    return float(optimize.brentq(f, hi, top, xtol=1e-3))


def local_thickness(solid, pitch: float):
    """Wall thickness at every solid cell (mm): the diameter of the largest
    ball that fits in the solid and covers the cell."""
    solid = np.asarray(solid, bool)
    edt = ndimage.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1, 1:-1]
    out = np.zeros(solid.shape)
    radii = np.unique(np.ceil(edt[solid] * 2.0) / 2.0)[::-1]
    for r in radii:
        seeds = edt >= r - 0.25
        if not seeds.any():
            continue
        reach = ndimage.distance_transform_edt(~seeds) < r
        new = reach & solid & (out == 0)
        out[new] = 2.0 * r - 1.0
    out[solid & (out == 0)] = 1.0
    return np.where(solid, out * pitch, 0.0)


def fill(metal, pitch: float, s: CastSetup, gate=None, thickness=None) -> dict:
    """Will the metal reach everywhere? Each cell's `index` is the share of
    the fluidity length spent on the cheapest path from the gate, through
    the wall thickness the path runs in (over 1: a misrun).

    gate: a grid index (i, j, k), or None to take the surface cell that
    leaves the smallest worst index (the best place to gate from, among
    cells spread over the surface)."""
    from skimage.graph import MCP_Geometric

    metal = np.asarray(metal, bool)
    th = local_thickness(metal, pitch) if thickness is None else thickness
    vals = np.unique(np.round(th[metal] / pitch * 2.0) / 2.0) * pitch
    table = np.array([max(fluidity_length(v, s), 1e-6) for v in vals])
    lf = np.interp(th, vals, table)
    cost = np.where(metal, pitch / np.maximum(lf, 1e-6), np.inf)     # share of fluidity per cell crossed

    def run(g):
        m = MCP_Geometric(cost, fully_connected=True)
        c, _ = m.find_costs([tuple(int(v) for v in g)])
        return np.where(metal, c, np.nan)

    if gate is None:
        surf = metal & ~ndimage.binary_erosion(metal)
        cand = _spread(np.argwhere(surf), 16)
        best = None
        for g in cand:
            idx = run(g)
            worst = float(np.nanmax(np.where(np.isfinite(idx), idx, np.nan)))
            if best is None or worst < best[0]:
                best = (worst, g, idx)
        _, gate, idx = best
    else:
        gate = np.asarray(gate, int)
        if not metal[tuple(gate)]:
            pts = np.argwhere(metal)
            gate = pts[np.argmin(((pts - gate) ** 2).sum(1))]
        idx = run(gate)
    reach = np.isfinite(idx)
    mis = metal & (~reach | (np.nan_to_num(idx, nan=np.inf) > 1.0))
    from .verify import _thickness

    thin = _thickness(metal, pitch)[0]          # on the medial ridge: a ball does not reach into corners
    thin = float(th[metal].min()) if thin is None else float(thin)
    return {"gate": [int(v) for v in gate], "index": idx, "thickness": th,
            "maxIndex": float(np.nanmax(np.where(reach, idx, np.nan))),
            "misrunVolume": float(mis.sum() * pitch ** 3),
            "misrunShare": float(mis.sum() / max(metal.sum(), 1)),
            "thinnestWall": thin, "capillaryWall": capillary_wall(s.alloy, s.head, s.pressure),
            "fluidityAtThinnest": fluidity_length(thin, s)}


def _spread(points, n):
    """n points spread over a set (farthest-point sampling)."""
    points = np.asarray(points)
    if len(points) <= n:
        return points
    chosen = [int(np.argmin(points.sum(1)))]
    d = ((points - points[chosen[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, ((points - points[i]) ** 2).sum(1))
    return points[chosen]


# --- contraction and gating --------------------------------------------------

def contraction(a: CastAlloy, room: float = 20.0) -> float:
    """Linear contraction of the solid casting from the solidus to room
    temperature (a fraction): the pattern is made larger by 1 / (1 - this)."""
    return a.cte * (a.solidus - room)


def gating(volume_mm3: float, s: CastSetup, thinnest_wall_mm: float, drop_mm: float | None = None) -> dict:
    """Pour time, sprue and ingate sizes for a gravity pour.

    The pour must finish before the thinnest wall's stream would stop
    (its tip time); the flow that takes fills the sprue at the speed the
    head gives, sqrt(2 g h), so the sprue's exit area follows, and its top
    is wider by sqrt(h_exit / h_top) so the falling stream stays full
    (continuity). The ingates pass the same flow under Campbell's critical
    velocity (`critical_velocity`, about 0.5 m/s for aluminium): the speed
    at which the front's inertia equals what surface tension can hold."""
    a = s.alloy
    head = (s.head if drop_mm is None else drop_mm) * 1e-3
    tau = _tip_time(a, thinnest_wall_mm * 1e-3, s)
    v_crit = critical_velocity(a)
    q = volume_mm3 * 1e-9 / max(tau, 1e-6)                       # m3/s
    v_exit = math.sqrt(2.0 * G0 * head)
    a_exit = q / v_exit
    top_head = 0.25 * head                                       # the pouring basin's depth over the sprue
    a_top = a_exit * math.sqrt(head / top_head)
    a_gate = q / v_crit
    d = lambda area: 2.0 * math.sqrt(area / math.pi) * 1e3       # noqa: E731
    return {"pourTime": tau, "flowRate": q * 1e6,                 # s, cm3/s
            "sprueExitDiameter": d(a_exit), "sprueTopDiameter": d(a_top), "sprueHeight": head * 1e3,
            "ingateArea": a_gate * 1e6, "criticalVelocity": v_crit, "sprueExitVelocity": v_exit}


def critical_velocity(a: CastAlloy) -> float:
    """Campbell's critical ingate velocity (m/s): a rising front breaks up
    when its velocity head exceeds the height surface tension can hold as a
    sessile drop, h = 2 sqrt(gamma / (rho g)); v = sqrt(2 g h)."""
    return math.sqrt(2.0 * G0 * 2.0 * math.sqrt(a.surface_tension / (a.density * G0)))


# --- risers -------------------------------------------------------------------

def _stamp(part, risers, pitch, up):
    """The part's grid grown to hold its risers: (metal, feeders, open_top,
    offset of the part's cell (0,0,0) in the new grid). A riser is
    {"base": (i, j, k) the part cell it stands on, "diameter", "height"} (mm),
    a cylinder along `up` standing on that cell's top face."""
    ax, sg = up
    lat = [d for d in range(3) if d != ax]
    rmax = max([0.5 * r["diameter"] for r in risers], default=0.0)
    hmax = max([r["height"] for r in risers], default=0.0)
    rp = int(math.ceil(rmax / pitch)) + 1 if risers else 0
    hp = int(math.ceil(hmax / pitch)) + 1 if risers else 0
    pw = [(0, 0)] * 3
    for d in lat:
        pw[d] = (rp, rp)
    pw[ax] = (0, hp) if sg > 0 else (hp, 0)
    metal = np.pad(np.asarray(part, bool), pw)
    off = np.array([p[0] for p in pw])
    feeders = np.zeros(metal.shape, bool)
    open_top = np.zeros(metal.shape, bool)
    idx = np.indices(metal.shape)
    for r in risers:
        b = np.asarray(r["base"], int) + off
        d2 = sum((idx[d] - b[d]) ** 2 for d in lat) * pitch ** 2
        h = (idx[ax] - b[ax]) * sg * pitch                     # height of the cell centre above the base cell
        n = max(1, int(round(r["height"] / pitch)))
        cyl = (d2 <= (0.5 * r["diameter"]) ** 2) & (h > 0) & (h <= n * pitch + 1e-9)
        new = cyl & ~metal
        feeders |= new
        metal |= new
        open_top |= cyl & (np.abs(h - n * pitch) < 1e-6)
    return metal, feeders, open_top, off


def design_risers(part, pitch: float, s: CastSetup, *, max_risers: int = 4, tolerance: float = 0.001,
                  max_solves: int = 14, trim: bool = True, on_progress=None, **solve_kw):
    """Risers that leave the part free of shrinkage cavities, sized by
    solving the casting with them on it.

    Each round freezes the part with its risers so far. If shrinkage is left
    in the part (more than `tolerance` of its volume), the largest cavity's
    thermal centre is found by climbing the freezing-time field from it; a
    riser goes on the part's upper surface over that point (first guess:
    1.2 times the hot spot's freezing modulus, read from the solve), or the
    riser already there grows by a fifth. When the part is sound each riser
    is tried one size smaller (`trim`).

    Returns {"risers": [...], "result": Solidification (with risers),
    "bare": Solidification (without), "sound": bool, "offset": the part's
    origin cell in the result's grid}."""
    part = np.asarray(part, bool)
    ax, sg = s.up
    lat = [d for d in range(3) if d != ax]
    v_part = float(part.sum())
    risers = []
    B = chvorinov_constant(s)
    solves = 0

    def run(rs):
        nonlocal solves
        solves += 1
        metal, feed, top, off = _stamp(part, rs, pitch, s.up)
        r = solidify(metal, pitch, s, feeders=feed if rs else None, open_top=top if rs else None, **solve_kw)
        core = tuple(slice(o, o + n) for o, n in zip(off, part.shape))
        pv = np.nan_to_num(r.void[core]) * part
        return r, off, pv

    def unsound(pv):
        return float(pv.sum()) / v_part

    bare, off0, pv = run([])
    res, off = bare, off0
    time_bare = bare.time
    while unsound(pv) > tolerance and solves < max_solves:
        lab, n = ndimage.label(pv > 0.2, structure=np.ones((3, 3, 3)))
        if n == 0:
            lab, n = ndimage.label(pv > 0.02, structure=np.ones((3, 3, 3)))
        if n == 0:
            break
        vols = ndimage.sum(pv, lab, np.arange(1, n + 1))
        order = np.argsort(-vols)
        placed = False
        for c in order:
            cells = np.argwhere(lab == c + 1)
            core = tuple(slice(o, o + m) for o, m in zip(off, part.shape))
            tfield = np.where(part, np.nan_to_num(res.time[core], nan=-1.0), -1.0)
            h = cells[np.argmax(tfield[tuple(cells.T)])]
            # climb to the local maximum of the freezing time: the thermal centre feeding that cavity
            for _ in range(10 * max(part.shape)):
                lo = np.maximum(h - 1, 0)
                hi = np.minimum(h + 2, part.shape)
                box = tfield[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                k = np.array(np.unravel_index(np.argmax(box), box.shape)) + lo
                if tfield[tuple(k)] <= tfield[tuple(h)]:
                    break
                h = k
            col = [slice(None)] * 3
            for d in lat:
                col[d] = int(h[d])
            column = part[tuple(col)]
            tops = np.nonzero(column)[0]
            base = h.copy()
            base[ax] = int(tops.max() if sg > 0 else tops.min())
            near = None
            for r in risers:
                if sum((r["base"][d] - base[d]) ** 2 for d in lat) * pitch ** 2 <= (0.75 * r["diameter"]) ** 2:
                    near = r
            if near is not None:
                if near.get("grown", 0) >= 5:
                    continue
                near["diameter"] *= 1.2
                near["height"] = near["diameter"]
                near["grown"] = near.get("grown", 0) + 1
                placed = True
                break
            if len(risers) >= max_risers:
                continue
            t_hot = float(np.nanmax(np.where(part, np.nan_to_num(time_bare, nan=0.0), 0.0)[tuple(cells.T)]))
            t_hot = max(t_hot, float(tfield[tuple(h)]))
            d0 = max(6.0 * math.sqrt(t_hot / B) * 1e3, 3.0 * pitch)
            risers.append({"base": [int(v) for v in base], "diameter": d0, "height": d0, "hotSpot": [int(v) for v in h]})
            placed = True
            break
        if not placed:
            break
        res, off, pv = run(risers)
        if on_progress is not None:
            on_progress({"stage": "risers", "risers": len(risers), "unsound": unsound(pv)})
    sound = unsound(pv) <= tolerance
    if sound and trim:
        for r in risers:
            while solves < max_solves and r["diameter"] * 0.87 >= 3.0 * pitch:
                keep = (r["diameter"], r["height"])
                r["diameter"] *= 0.87
                r["height"] = r["diameter"]
                res2, off2, pv2 = run(risers)
                if unsound(pv2) <= tolerance:
                    res, off, pv = res2, off2, pv2
                else:
                    r["diameter"], r["height"] = keep
                    break
        # the result on hand may be a rejected trial's: solve the accepted set once more if so
        metal, feed, top, off = _stamp(part, risers, pitch, s.up)
        if res.metal.shape != metal.shape or (res.metal != metal).any():
            res, off, pv = run(risers)
    for r in risers:
        r.pop("grown", None)
        r["volume"] = math.pi / 4.0 * r["diameter"] ** 2 * r["height"]
    return {"risers": risers, "result": res, "bare": bare, "sound": bool(sound), "offset": off,
            "partVoid": pv, "solves": solves}


# --- the report -----------------------------------------------------------------

def _mm(idx, origin, pitch):
    return [float(v) for v in (np.asarray(origin, float) + (np.asarray(idx, float) + 0.5) * pitch)]


def castability(part, pitch: float, s: CastSetup, *, origin=(0.0, 0.0, 0.0), gate=None, risers="auto",
                max_risers: int = 4, tolerance: float = 0.001, on_progress=None, **solve_kw) -> dict:
    """Everything this module knows about casting a part, as a report.

    part: bool grid at `pitch` mm, cell (0,0,0)'s min corner at `origin`.
    risers: "auto" designs them (design_risers), None leaves the part bare,
    or a list of {"position": [x, y, z] mm, "diameter", "height"}.
    Returns (report dict, fields dict) - the fields are the grids behind it
    (result, bare, fill index, thickness) for rendering and mold making."""
    part = np.asarray(part, bool)
    a = s.alloy
    v_cell = pitch ** 3
    v_part = float(part.sum() * v_cell)
    origin = np.asarray(origin, float)
    warnings = []
    if risers == "auto" and s.process == "die":
        risers = None                         # a die casting is fed through the gate under pressure
    if risers == "auto":
        d = design_risers(part, pitch, s, max_risers=max_risers, tolerance=tolerance, on_progress=on_progress,
                          **solve_kw)
        res, bare, off, rs = d["result"], d["bare"], d["offset"], d["risers"]
    else:
        rs = []
        for r in risers or []:
            b = np.round((np.asarray(r["position"], float) - origin) / pitch - 0.5).astype(int)
            b = np.clip(b, 0, np.asarray(part.shape) - 1)
            if not part[tuple(b)]:
                pts = np.argwhere(part)
                b = pts[np.argmin(((pts - b) ** 2).sum(1))]
            rs.append({"base": [int(v) for v in b], "diameter": float(r["diameter"]),
                       "height": float(r.get("height", r["diameter"])),
                       "volume": math.pi / 4 * r["diameter"] ** 2 * r.get("height", r["diameter"])})
        bare = solidify(part, pitch, s, **solve_kw)
        if rs:
            metal, feed, top, off = _stamp(part, rs, pitch, s.up)
            res = solidify(metal, pitch, s, feeders=feed, open_top=top, **solve_kw)
        else:
            res, off = bare, np.zeros(3, int)
    core = tuple(slice(o, o + n) for o, n in zip(off, part.shape))
    crop = lambda f: np.where(part, f[core], np.nan)  # noqa: E731
    void = np.nan_to_num(crop(res.void))
    pores = crop(res.pores)
    tsol = crop(res.time)
    ny = crop(res.niyama)
    sdas = crop(res.sdas)

    def spots(result, where, offset):
        out = []
        for c in result.cavities(min_share=0.2, where=where)[:6]:
            if c["volume"] < max(0.0002 * v_part, 2 * v_cell * a.beta_prime()):
                continue
            pos = _mm(c["position"] - offset, origin, pitch)
            out.append({"position": pos, "volume": c["volume"], "freezesAt": c["lastToFreeze"],
                        "modulus": math.sqrt(c["lastToFreeze"] / chvorinov_constant(s)) * 1e3})
        return out

    part_ext = np.zeros(res.metal.shape, bool)
    part_ext[core] = part
    hot_bare = spots(bare, part, np.zeros(3))
    left = spots(res, part_ext, off)
    shrink_bare = float(np.nan_to_num(bare.void)[part].sum() * v_cell)
    shrink_left = float(void.sum() * v_cell)
    report = {"setup": s.to_dict(), "volume": v_part, "mass": v_part * 1e-9 * a.density * (1 + a.shrinkage) * 1e3,
              "pitch": pitch,
              "solidification": {"time": float(np.nanmax(tsol)), "timeWithRisers": float(res.total_time),
                                 "solveSeconds": float(res.seconds + (bare.seconds if bare is not res else 0.0)),
                                 "energyError": float(res.energy_error), "complete": bool(res.complete)},
              "shrinkage": {"total": v_part * a.beta_prime(), "inPartWithoutRisers": shrink_bare,
                            "inPart": shrink_left, "inPartShare": shrink_left / v_part,
                            "hotSpots": hot_bare, "cavities": left,
                            "sound": bool(shrink_left <= tolerance * v_part)}}
    if rs:
        feed_vol = sum(r["volume"] for r in rs)
        fz = res.feeders
        pipe = []
        for r in rs:
            b = np.asarray(r["base"]) + off
            ax, sg = s.up
            lat = [d for d in range(3) if d != ax]
            idx = np.indices(res.metal.shape)
            near = fz & (sum((idx[d] - b[d]) ** 2 for d in lat) * pitch ** 2 <= (0.5 * r["diameter"] + pitch) ** 2)
            emptied = near & (np.nan_to_num(res.void) > 0.5)
            if emptied.any():
                hh = (idx[ax][emptied] - b[ax]) * sg * pitch
                depth = r["height"] - float(hh.min()) + pitch
            else:
                depth = 0.0
            pipe.append(depth)
        report["risers"] = [{"position": _mm(np.asarray(r["base"]) + (np.eye(3)[s.up[0]] * 0.5 * s.up[1]), origin,
                                             pitch),
                             "diameter": r["diameter"], "height": r["height"], "volume": r["volume"],
                             "pipeDepth": p, "freezesAt": float(np.nanmax(np.where(fz, res.time, np.nan)))}
                            for r, p in zip(rs, pipe)]
        report["yield"] = v_part / (v_part + feed_vol)
    finite = np.isfinite(pores) & part
    report["microporosity"] = {
        "max": float(np.nanmax(pores)) if finite.any() else 0.0,
        "mean": float(np.nanmean(pores[finite])) if finite.any() else 0.0,
        "volumeOver0.5pct": float((pores[finite] > 0.005).sum() * v_cell),
        "shareOver0.5pct": float((pores[finite] > 0.005).mean()) if finite.any() else 0.0,
        "niyamaMin": float(np.nanmin(ny)) if np.isfinite(ny).any() else None}
    report["dendriteArmSpacing"] = {"mean": float(np.nanmean(sdas)), "p95": float(np.nanpercentile(sdas, 95)),
                                    "min": float(np.nanmin(sdas))}
    g = None if gate is None else np.round((np.asarray(gate, float) - origin) / pitch - 0.5).astype(int)
    f = fill(part, pitch, s, gate=None if g is None else np.clip(g, 0, np.asarray(part.shape) - 1))
    far = float(np.nanmax(np.where(np.isfinite(f["index"]), f["index"], np.nan)))
    flow = _flow_length(part, f["gate"], pitch)
    report["fill"] = {"gate": _mm(f["gate"], origin, pitch), "gateChosen": gate is None,
                      "thinnestWall": f["thinnestWall"], "capillaryWall": f["capillaryWall"],
                      "fluidityAtThinnestWall": f["fluidityAtThinnest"], "flowLength": flow,
                      "minWallAtFlowLength": min_wall(flow, s), "maxIndex": far,
                      "misrunVolume": f["misrunVolume"], "misrunShare": f["misrunShare"]}
    report["contraction"] = {"linear": contraction(a), "patternScale": 1.0 / (1.0 - contraction(a))}
    if s.process != "die":
        report["gating"] = gating(v_part + sum(r["volume"] for r in rs), s, max(f["thinnestWall"], 0.5 * pitch))
    if not report["shrinkage"]["sound"]:
        warnings.append(f"{shrink_left:.0f} mm^3 of shrinkage is left in the part "
                        f"({100 * shrink_left / v_part:.2f}% of its volume) in {len(left)} cavities"
                        + ("" if rs else "; no risers"))
    if f["misrunShare"] > 0.002:
        warnings.append(f"{100 * f['misrunShare']:.1f}% of the part is past the metal's fluidity from the gate "
                        f"(thinnest wall {f['thinnestWall']:.1f} mm; {report['fill']['minWallAtFlowLength']:.1f} mm "
                        f"fills over the {flow:.0f} mm flow length)")
    if f["thinnestWall"] < f["capillaryWall"]:
        warnings.append(f"the thinnest wall ({f['thinnestWall']:.2f} mm) is under the {f['capillaryWall']:.2f} mm "
                        "the metal head can enter against surface tension")
    if report["microporosity"]["shareOver0.5pct"] > 0.02:
        warnings.append(f"{100 * report['microporosity']['shareOver0.5pct']:.0f}% of the part is predicted over "
                        "0.5% microporosity (low thermal gradient late in freezing)")
    if not res.complete:
        warnings.append("the solidification solve stopped at its time limit before the metal was all solid")
    if pitch > 0.5 * f["thinnestWall"] + 1e-9:
        warnings.append(f"the thinnest wall ({f['thinnestWall']:.1f} mm) is under two cells of the {pitch:.2f} mm "
                        "grid: its freezing time is resolved coarsely")
    report["warnings"] = warnings
    report["ok"] = bool(report["shrinkage"]["sound"] and f["misrunShare"] <= 0.002
                        and f["thinnestWall"] >= f["capillaryWall"])
    fields = {"result": res, "bare": bare, "offset": off, "fill": f, "void": void, "pores": pores, "time": tsol,
              "risers": rs, "part": part, "origin": origin, "pitch": pitch}
    return report, fields


def _flow_length(metal, gate, pitch):
    """Longest path (mm) the metal runs through the part from the gate."""
    from skimage.graph import MCP_Geometric

    cost = np.where(metal, 1.0, np.inf)
    c, _ = MCP_Geometric(cost, fully_connected=True).find_costs([tuple(int(v) for v in gate)])
    c = c[metal]
    return float(c[np.isfinite(c)].max() * pitch) if np.isfinite(c).any() else 0.0


def grid_pitch(mesh, cells: int = 60_000, min_pitch: float = 0.25) -> float:
    """A pitch that puts about `cells` cells in the mesh's bounding box."""
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], float)
    return float(max((float(np.prod(ext)) / cells) ** (1.0 / 3.0), min_pitch))


def cast_part(mesh, process: str = "sand", alloy_name="a356", *, pitch: float | None = None, cells: int = 60_000,
              gate=None, risers="auto", on_progress=None, **setup_kw):
    """`castability` of a mesh (mm): voxelizes it and runs the report.
    setup_kw go to `setup` (pour_temperature, mold_temperature, head, up...)."""
    from .verify import voxelize

    s = setup(process, alloy_name, **setup_kw)
    p = float(pitch) if pitch else grid_pitch(mesh, cells)
    solid, org = voxelize(mesh, p)
    report, fields = castability(solid, p, s, origin=org, gate=gate, risers=risers, on_progress=on_progress)
    report["meshVolume"] = float(mesh.volume)
    return report, fields
