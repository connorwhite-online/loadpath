"""
Material properties for the verification pass.

The optimizer itself does not need any of this: for a linear elastic
compliance problem the optimal layout is independent of Young's modulus, so
the solver runs on E = 1. Material only enters when we turn an outcome into
numbers a person acts on (stress against strength, deflection in mm, mass in
grams).

Values are typical datasheet numbers for printed parts, not certified
allowables. Printed plastics are anisotropic (weaker across layers, often by
half for FDM) and the analysis is linear, so a safety factor computed from
these is an estimate to compare designs and to catch an undersized part, not
a sign-off.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Material", "MATERIALS", "material_from_spec"]


@dataclass(frozen=True)
class Material:
    name: str
    E: float          # Young's modulus, MPa
    nu: float         # Poisson's ratio
    strength: float   # MPa: yield for metals, tensile strength for polymers
    density: float    # g/cm^3
    note: str = ""

    def to_dict(self):
        return {"name": self.name, "E": self.E, "nu": self.nu,
                "strength": self.strength, "density": self.density, "note": self.note}


MATERIALS = {m.name: m for m in [
    Material("pla", 3500, 0.36, 50, 1.24, "FDM, in-plane"),
    Material("petg", 2000, 0.38, 45, 1.27, "FDM, in-plane"),
    Material("abs", 2100, 0.35, 35, 1.04, "FDM, in-plane"),
    Material("asa", 2000, 0.35, 40, 1.07, "FDM, in-plane"),
    Material("pa-cf", 4500, 0.35, 70, 1.15, "FDM carbon-filled nylon, in-plane"),
    Material("pa12", 1800, 0.39, 48, 1.01, "SLS/MJF"),
    Material("pa11", 1600, 0.39, 48, 1.05, "SLS/MJF"),
    Material("resin", 2500, 0.38, 50, 1.18, "SLA standard"),
    Material("resin-tough", 1600, 0.40, 40, 1.15, "SLA tough"),
    Material("alsi10mg", 70000, 0.33, 230, 2.67, "LPBF aluminium, as built"),
    Material("ti64", 110000, 0.34, 900, 4.43, "LPBF titanium grade 5"),
    Material("316l", 190000, 0.30, 450, 7.98, "LPBF stainless"),
]}

_ALIASES = {
    "nylon": "pa12", "pa": "pa12", "pa 12": "pa12", "pa-12": "pa12",
    "aluminium": "alsi10mg", "aluminum": "alsi10mg", "titanium": "ti64",
    "stainless": "316l", "steel": "316l", "carbon nylon": "pa-cf",
}


def material_from_spec(spec) -> Material:
    """A name from MATERIALS (or an alias), or a custom
    {"name"?, "E", "nu", "strength", "density"} object."""
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        key = _ALIASES.get(key, key)
        if key not in MATERIALS:
            raise ValueError(f"unknown material {spec!r}; known: {sorted(MATERIALS)} "
                             "or pass {E, nu, strength, density}")
        return MATERIALS[key]
    if isinstance(spec, dict):
        try:
            m = Material(str(spec.get("name", "custom")), float(spec["E"]), float(spec.get("nu", 0.35)),
                         float(spec["strength"]), float(spec["density"]), str(spec.get("note", "")))
        except KeyError as exc:
            raise ValueError(f"custom material is missing {exc.args[0]!r} (needs E, strength, density)") from None
        if m.E <= 0 or m.strength <= 0 or m.density <= 0 or not (0 <= m.nu < 0.5):
            raise ValueError("custom material needs E, strength, density > 0 and 0 <= nu < 0.5")
        return m
    raise ValueError("material must be a name or {E, nu, strength, density}")
