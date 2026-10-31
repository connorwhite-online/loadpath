"""
Durability: fatigue under repeated loads, creep and stress relaxation under
sustained ones, for printed parts (mostly plastics).

The static check (verify.py) asks whether a part survives its load once. A
robot arm's printed links see the same servo load hundreds of thousands of
times, and its bolted bosses hold a clamp load for years, sometimes warm.
Plastics are poor at both: fatigue strength at a million cycles is a small
fraction of the tensile strength, and a polymer under a steady load keeps
straining (creep) or, held at fixed strain by a bolt, loses its stress
(relaxation). This module turns the verified stresses into those readings.

A load case opts in with `cycles` (how many times it is applied over the
part's life: fatigue) and/or `hours` (how long it is held: creep), and
`temperature` (C) if it differs from the study's `durability.temperature`.
Fasteners' preload cases are held for `durability.preloadHours`.

Fatigue. Ezeh and Susmel (2018, 2019) re-analysed every published fatigue
dataset on FDM PLA with their own tests across raster angles and load
ratios R = -1..0.3 and found that (a) fatigue strength scales with the
static tensile strength, (b) the raster angle matters little for parts
printed flat, and (c) Goodman's mean-stress correction is unconservative
for printed PLA, while the maximum stress in the cycle captures the mean
stress effect on its own. Their design curve (survival probability > 90%,
100% infill) is

    sigma_max(N) = 0.10 * UTS * (N / 2e6) ** (-1 / 5.5)

so a load case's life is N = 2e6 * (0.10 UTS / sigma_max) ** 5.5, and
several cyclic cases add up by Miner's rule. The fatigue safety factor is
the factor on every cyclic load that brings the Miner sum to 1: D ** (-1/k).
(The peaks of different cases can sit in different places; summing them is
conservative.) Across layers the same curve is scaled by the process's
bond ratio (process.interlayer) and read against the tension normal to the
layers, as the static bond check does.

No comparable design curve was found for printed PETG, ABS, ASA or TPU, so
those use the PLA curve normalized by their own strength (the
strength-proportional result above), flagged "assumed". SLS/MJF PA12/PA11
use Van Hooreweder and Kruth (CIRP Annals 2014): every specimen survived
18 MPa amplitude at 1e6 cycles, R = -1 (UTS ~48 MPa); taken as 0.30 UTS at
2e6 cycles with a slope of 10 after a scatter allowance. Metals get rough
as-built-surface values (machining or HIP raises them 2-3x).

Creep. Strain under a held stress grows as a power of time (Findley):
eps(t) = eps0 * (1 + phi(t)), phi(t) = phi_1000 * (t / 1000 h) ** n, with
phi_1000 the creep-to-elastic strain ratio after 1000 h at 23 C, read off
typical design-data creep moduli (E_creep(1000 h) / E ~ 0.65-0.75 for
amorphous FDM plastics, ~0.5 for nylons and resins). Warmer, a polymer
creeps faster as it softens: phi is divided by the modulus retention r(T)
(thermal.MODULUS_RETENTION) and the elastic strain too, so a held
deflection grows by (1 + phi / r) / r. That is a coarse stand-in for
time-temperature superposition and is least trustworthy near the glass
transition, where both readings run away together, as real parts do.

Creep rupture: a held stress breaks a polymer eventually. Its long-term
strength falls roughly linearly in log time from the short test (~6 min):
S_r(t) = UTS * r_s(T) * max(0.25, 1 - 0.10 * log10(t / 0.1 h)), which is
~60% of UTS at 1000 h and ~40% at ten years, typical of amorphous
thermoplastics. Strength retention with temperature is taken as the
modulus retention: conservative below Tg (strength falls less than
modulus there), right at it.

Bolted joints: a screw clamping a plastic part holds it at nearly fixed
strain, so the clamp load relaxes with the relaxation modulus: preload
retention ~ r(T) / (1 + phi(t, T)). Below about half, the joint will need
a compression limiter (a metal sleeve the screw clamps on), a heat-set
insert clamping metal to metal, a spring washer, or retightening.

Thermal cycling: a study whose thermal block has a duty cycle (thermal.
Cycle, with `cycles`) heats up and cools down that many times. verify_part
solves the cycle's hot and cold moments and, per node, takes the larger
von Mises stress over strength retention (so it reads against room-
temperature strength) as the cycle's maximum; where the stress barely moves
over the cycle (R above ~0.3, where the max-stress reading stops holding
and the load is closer to creep), the range over 0.7 counts instead. The
worst load case's peak of that, times `cycles`, joins the in-plane Miner
sum. An estimate on top of an estimate: no printed-polymer thermal-fatigue
data were found, so it is the mechanical design curve at the thermal
stress.

Everything here is an estimate to compare designs and catch parts that will
not last, not an allowable: printed fatigue scatters by a factor of 3 in
life even within one study, creep data for printed parts is sparse, and
both depend on infill, moisture (nylons), frequency (self-heating above
~5 Hz) and print settings. Test the part that matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DurabilityData", "DURABILITY", "Durability", "durability_from_spec", "durability_data",
           "strength_retention", "fatigue_strength", "cycles_to_failure", "creep_ratio", "rupture_strength",
           "assess"]

N_REF = 2e6           # cycles at which fatigue strength is quoted
T_REF = 23.0          # C
RUPTURE_T0 = 0.1      # h: the duration of a short-term tensile test
RUPTURE_FLOOR = 0.25  # long-term strength never read below this share of UTS

_EZEH = "Ezeh & Susmel, Int J Fatigue 2019 (FDM PLA design curve, Ps > 90%)"


@dataclass(frozen=True)
class DurabilityData:
    fatigue_ratio: float        # max stress / strength survived for N_REF cycles
    fatigue_slope: float        # k: life goes as stress ** -k
    creep_ratio: float          # creep / elastic strain after 1000 h at 23 C (phi_1000)
    creep_exponent: float       # n in phi ~ t ** n
    rupture_per_decade: float   # long-term strength lost per decade of time, share of UTS
    source: str
    quality: str                # "measured", "assumed" or "rough"

    def to_dict(self):
        return {"fatigueStrength": self.fatigue_ratio, "fatigueSlope": self.fatigue_slope,
                "creepRatio": self.creep_ratio, "creepExponent": self.creep_exponent,
                "rupturePerDecade": self.rupture_per_decade, "source": self.source, "quality": self.quality}


_PLA_CURVE = (0.10, 5.5)
DURABILITY = {
    "pla": DurabilityData(*_PLA_CURVE, 0.30, 0.20, 0.10, _EZEH, "measured"),
    "petg": DurabilityData(*_PLA_CURVE, 0.40, 0.20, 0.10,
                           "PLA design curve scaled by strength (no printed PETG design curve found); "
                           "creep from copolyester creep-modulus data", "assumed"),
    "abs": DurabilityData(*_PLA_CURVE, 0.45, 0.20, 0.10,
                          "PLA design curve scaled by strength; creep from ABS creep-modulus data", "assumed"),
    "asa": DurabilityData(*_PLA_CURVE, 0.45, 0.20, 0.10,
                          "PLA design curve scaled by strength; creep as ABS", "assumed"),
    "tpu": DurabilityData(0.15, 8.0, 1.00, 0.15, 0.10,
                          "engineering estimate: elastomer fatigue is strain-governed and no printed-TPU "
                          "S-N data was found; verify with nonlinear analysis and a test", "rough"),
    "pa-cf": DurabilityData(0.15, 8.0, 0.50, 0.20, 0.10,
                            "estimate between FDM PLA and filled nylon; moisture lowers both", "rough"),
    "pa12": DurabilityData(0.30, 10.0, 0.80, 0.20, 0.10,
                           "Van Hooreweder & Kruth, CIRP Annals 2014 (SLS PA12, 18 MPa at 1e6, R=-1); "
                           "creep from PA12 creep-modulus data, dry", "measured"),
    "pa11": DurabilityData(0.30, 10.0, 0.80, 0.20, 0.10, "as PA12", "assumed"),
    "resin": DurabilityData(0.08, 5.0, 1.00, 0.25, 0.12,
                            "estimate: standard SLA resins are brittle and creep at room temperature", "rough"),
    "resin-tough": DurabilityData(0.10, 6.0, 1.20, 0.25, 0.12, "estimate, as standard resin", "rough"),
    # metals, against `strength` (yield): as-built surfaces, no creep at these temperatures
    "alsi10mg": DurabilityData(0.30, 6.0, 0.0, 0.2, 0.0,
                               "rough: LPBF AlSi10Mg as built, ~60-70 MPa at 1e7 R=-1", "rough"),
    "ti64": DurabilityData(0.28, 6.0, 0.0, 0.2, 0.0, "rough: LPBF Ti-6Al-4V as built", "rough"),
    "316l": DurabilityData(0.40, 8.0, 0.0, 0.2, 0.0, "rough: LPBF 316L as built", "rough"),
}
_GENERIC = DurabilityData(*_PLA_CURVE, 0.50, 0.20, 0.10,
                          "no data for this material: generic FDM thermoplastic (PLA curve, ABS-like creep)",
                          "assumed")


@dataclass
class Durability:
    temperature: float = T_REF
    safety_factor: float = 1.5
    # True: the safety-factor search sizes the part for fatigue and creep too
    constrain: bool = False
    preload_hours: float = 8760.0
    overrides: dict = field(default_factory=dict)


_OVERRIDES = {"fatigueStrength": "fatigue_ratio", "fatigueSlope": "fatigue_slope",
              "creepRatio": "creep_ratio", "creepExponent": "creep_exponent"}


def durability_from_spec(spec) -> Durability | None:
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("durability must be an object: {temperature, safetyFactor, constrain, preloadHours}")
    t = float(spec.get("temperature", T_REF))
    if not -60.0 <= t <= 400.0:
        raise ValueError("durability.temperature is the service temperature in C")
    sf = float(spec.get("safetyFactor", 1.5))
    if sf <= 0:
        raise ValueError("durability.safetyFactor must be positive")
    ph = float(spec.get("preloadHours", 8760.0))
    if ph <= 0:
        raise ValueError("durability.preloadHours must be positive (h)")
    over = {}
    for key, attr in _OVERRIDES.items():
        if spec.get(key) is not None:
            v = float(spec[key])
            if v < 0 or (key in ("fatigueStrength", "fatigueSlope", "creepExponent") and v <= 0):
                raise ValueError(f"durability.{key} must be positive")
            over[attr] = v
    return Durability(t, sf, bool(spec.get("constrain", False)), ph, over)


def _key(material) -> str:
    from .thermal import _material_key

    return _material_key(material)


def durability_data(material, overrides: dict | None = None) -> DurabilityData:
    import dataclasses

    base = DURABILITY.get(_key(material), _GENERIC)
    if overrides:
        base = dataclasses.replace(base, **overrides, source=base.source + " (overridden in the study)",
                                   quality="study")
    return base


def strength_retention(material, t: float) -> float | None:
    """Share of room-temperature modulus (and, as a proxy, strength) kept at
    t (C), or None when the material has no table."""
    from .thermal import MODULUS_RETENTION, retention

    if _key(material) not in MODULUS_RETENTION:
        return None
    return float(retention(material, np.array([t]))[0][0])


def fatigue_strength(data: DurabilityData, strength: float, cycles: float) -> float:
    """Max stress survived for `cycles` (MPa), never above `strength`."""
    s = data.fatigue_ratio * strength * (max(cycles, 1.0) / N_REF) ** (-1.0 / data.fatigue_slope)
    return float(min(s, strength))


def cycles_to_failure(data: DurabilityData, strength: float, sigma_max: float) -> float:
    if sigma_max <= 0:
        return float("inf")
    if sigma_max >= strength:
        return 1.0
    return float(N_REF * (data.fatigue_ratio * strength / sigma_max) ** data.fatigue_slope)


def creep_ratio(data: DurabilityData, hours: float, retention: float = 1.0) -> float:
    """phi: creep strain over elastic strain after `hours`, at a temperature
    where the modulus is `retention` of its room value."""
    if data.creep_ratio <= 0 or hours <= 0:
        return 0.0
    return float(data.creep_ratio * (hours / 1000.0) ** data.creep_exponent / max(retention, 0.02))


def rupture_strength(data: DurabilityData, strength: float, hours: float) -> float:
    """Stress held for `hours` before creep rupture, MPa (strength already at temperature)."""
    if hours <= RUPTURE_T0 or data.rupture_per_decade <= 0:
        return float(strength)
    f = max(RUPTURE_FLOOR, 1.0 - data.rupture_per_decade * np.log10(hours / RUPTURE_T0))
    return float(strength * f)


def _thermal_cycles(study):
    th = getattr(study, "thermal", None)
    return th.cycle.cycles if th is not None and th.cycle is not None and th.cycle.cycles else None


def active(study) -> bool:
    return study.durability is not None or any(c.cycles or c.hours for c in study.load_cases) \
        or _thermal_cycles(study) is not None


def _hdt(mat):
    """The library card's heat deflection temperature at 0.45 MPa, if any."""
    from .material_library import value
    return value(_key(mat), "hdt")


def assess(study, cases: list, allowance: float = 1.0) -> dict:
    """The durability block of a verification report. `cases` are the
    report's per-case dicts (peakStress, interlayerStress, deflection) in
    load-case order; stresses are multiplied by `allowance` (the same
    discretization allowance the static safety factor uses)."""
    dur = study.durability or Durability()
    mat = study.material
    data = durability_data(mat, dur.overrides)
    proc = study.process
    inter = proc.interlayer if proc is not None and proc.build not in (None, "auto") else None
    out = {"temperature": dur.temperature, "target": dur.safety_factor, "constrain": dur.constrain,
           "material": mat.name, "data": data.to_dict(), "cases": [], "warnings": []}

    def ret(t):
        r = strength_retention(mat, t)
        if r is None:
            if abs(t - T_REF) > 5:
                out["warnings"].append(f"no temperature data for {mat.name}: read at room-temperature properties")
            return 1.0
        return r

    damage = {"inPlane": 0.0, "acrossLayers": 0.0}
    creep_sf = np.inf
    for lc, c in zip(study.load_cases, cases):
        hours = lc.hours
        if hours is None and lc.name == "preload" and study.joints and study.durability is not None:
            hours = dur.preload_hours
        if not lc.cycles and not hours:
            continue
        t = lc.temperature if lc.temperature is not None else dur.temperature
        # a study with heat has solved the temperature at the peak itself
        r = c["strengthRetention"] if c.get("strengthRetention") is not None else ret(t)
        if c.get("peakTemperature") is not None:
            t = c["peakTemperature"]
        hdt = _hdt(mat)
        if hdt is not None and t >= hdt:
            msg = (f"{lc.name}: {t:g} C is at or above {mat.name}'s heat deflection temperature ({hdt:g} C at "
                   "0.45 MPa, material_library.py): the part will sag; use a material rated hotter")
            if msg not in out["warnings"]:
                out["warnings"].append(msg)
        s_t = mat.strength * r
        peak = c["peakStress"] * allowance
        tens = (c.get("interlayerStress") or 0.0) * allowance if inter is not None else 0.0
        row = {"name": lc.name, "temperature": t, "strengthAtTemperature": s_t}
        if lc.cycles:
            life = cycles_to_failure(data, s_t, peak)
            row.update(cycles=lc.cycles, fatigueStrength=fatigue_strength(data, s_t, lc.cycles), life=life,
                       damage=lc.cycles / life)
            damage["inPlane"] += lc.cycles / life
            if tens > 0:
                life_i = cycles_to_failure(data, inter * s_t, tens)
                row.update(interlayerLife=life_i, interlayerDamage=lc.cycles / life_i)
                damage["acrossLayers"] += lc.cycles / life_i
        if hours:
            phi = creep_ratio(data, hours, r)
            s_r = rupture_strength(data, s_t, hours)
            sf = s_r / peak if peak > 0 else np.inf
            if tens > 0:
                sf = min(sf, inter * s_r / tens)
            creep_sf = min(creep_sf, sf)
            row.update(hours=hours, creepRatio=phi, creepDeflection=c["deflection"] * (1.0 + phi) / r,
                       ruptureStrength=s_r, creepSafetyFactor=sf)
            if sf < 2.0 and data.creep_ratio > 0:
                out["warnings"].append(
                    f"{lc.name}: held at {1 / sf:.0%} of its {hours:g} h rupture strength; creep there is "
                    "nonlinear and faster than the estimate")
        out["cases"].append(row)

    n_th = _thermal_cycles(study)
    if n_th is not None and any("cycleStress" in c for c in cases):
        # heating up and cooling down (thermal.Cycle): the worst load case's
        # equivalent cycle stress (verify_part, over strength retention, so
        # read against room-temperature strength), n_th times
        worst = max((c for c in cases if "cycleStress" in c), key=lambda c: c["cycleStress"])
        sig = worst["cycleStress"] * allowance
        life = cycles_to_failure(data, mat.strength, sig)
        damage["inPlane"] += n_th / life
        out["thermalCycling"] = {"cycles": n_th, "case": worst["name"], "stress": sig,
                                 "stressRange": worst["cycleStressRange"] * allowance, "life": life,
                                 "damage": n_th / life,
                                 "fatigueStrength": fatigue_strength(data, mat.strength, n_th)}

    k = data.fatigue_slope
    fat_sf = np.inf
    for where, d in damage.items():
        if d > 0:
            fat_sf = min(fat_sf, d ** (-1.0 / k))
    out["fatigueDamage"] = max(damage.values())
    out["fatigueSafetyFactor"] = fat_sf if np.isfinite(fat_sf) else None
    out["creepSafetyFactor"] = creep_sf if np.isfinite(creep_sf) else None
    sfs = [v for v in (fat_sf, creep_sf) if np.isfinite(v)]
    out["safetyFactor"] = min(sfs) if sfs else None
    if damage["acrossLayers"] > damage["inPlane"] and damage["acrossLayers"] > 0:
        out["fatigueLimitedBy"] = "layer bond"
    elif damage["inPlane"] > 0:
        out["fatigueLimitedBy"] = "in-plane"

    if study.joints and study.durability is not None:
        t = dur.temperature
        r = ret(t)
        phi = creep_ratio(data, dur.preload_hours, r)
        s_r = rupture_strength(data, mat.strength * r, dur.preload_hours)
        joints, relaxed = [], []
        for j in study.joints:
            bearing = j["preload"] / j["area"] if j["area"] > 0 else 0.0
            keep = r / (1.0 + phi)
            row = {"id": j["id"], "preload": j["preload"], "bearingStress": bearing,
                   "preloadRetention": keep, "bearingSafetyFactor": s_r / bearing if bearing > 0 else None,
                   "hours": dur.preload_hours, "temperature": t}
            joints.append(row)
            if keep < 0.5 and data.creep_ratio > 0:
                relaxed.append(j["id"])
            if bearing > 0 and s_r / bearing < 1.0:
                out["warnings"].append(f"fastener {j['id']}: the head's bearing stress {bearing:.1f} MPa is above "
                                       f"the {dur.preload_hours:g} h creep strength {s_r:.1f} MPa; use a washer "
                                       "or lower the preload")
        if relaxed:
            # the same for every screw in one material: one line, not one per screw
            keep = r / (1.0 + phi)
            who = f"fastener {relaxed[0]}" if len(relaxed) == 1 else f"fasteners {', '.join(relaxed)}"
            out["warnings"].append(
                f"{who}: about {keep:.0%} of the clamp load is left after {dur.preload_hours:g} h "
                f"at {t:g} C; clamp on a metal compression limiter or insert, or add a spring washer")
        out["joints"] = joints

    if data.quality in ("assumed", "rough"):
        out["warnings"].append(f"{mat.name} fatigue/creep data is {data.quality}: {data.source}")
    return out
