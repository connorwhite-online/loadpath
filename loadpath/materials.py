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
    # the processes (spec.PROCESSES) it is printed by; empty = any
    processes: tuple = ()

    def to_dict(self):
        return {"name": self.name, "E": self.E, "nu": self.nu,
                "strength": self.strength, "density": self.density, "note": self.note,
                "processes": list(self.processes)}


_FDM, _POWDER, _RESIN, _METAL = ("fdm",), ("sls", "mjf"), ("sla", "dlp"), ("lpbf",)

MATERIALS = {m.name: m for m in [
    Material("pla", 3500, 0.36, 50, 1.24, "FDM, in-plane", _FDM),
    Material("petg", 2000, 0.38, 45, 1.27, "FDM, in-plane", _FDM),
    Material("abs", 2100, 0.35, 35, 1.04, "FDM, in-plane", _FDM),
    Material("asa", 2000, 0.35, 40, 1.07, "FDM, in-plane", _FDM),
    # 95A: linear analysis holds only to a few percent strain; flexures
    # want verify_nonlinear (nonlinear.py)
    Material("tpu", 26, 0.45, 26, 1.21, "FDM TPU 95A, small strains only", _FDM),
    Material("pa-cf", 4500, 0.35, 70, 1.15, "FDM carbon-filled nylon, in-plane", _FDM),
    # powder bed: datasheet XY values (HP 3D HR PA 12 / PA 11 / PA 12 GB,
    # BASF Ultrasint TPU 88A); Z is within ~5-10% (spec.PROCESSES interlayer)
    Material("pa12", 1800, 0.39, 48, 1.01, "SLS/MJF nylon 12", _POWDER + _FDM),
    Material("pa11", 1800, 0.39, 52, 1.05, "SLS/MJF nylon 11: tougher than PA12 (~50% elongation)", _POWDER),
    Material("pa12-gb", 2500, 0.37, 30, 1.30, "SLS/MJF glass-bead filled PA12: stiff, stable, brittle", _POWDER),
    Material("tpu88a", 85, 0.45, 9, 1.10, "SLS/MJF TPU 88A: a linear analysis is a rough estimate past a few "
                                           "percent strain; run verify_nonlinear", _POWDER),
    # resin, post-cured (Formlabs datasheets)
    Material("resin", 2500, 0.38, 50, 1.18, "SLA/DLP standard", _RESIN),
    Material("resin-tough", 1600, 0.40, 40, 1.15, "SLA/DLP tough (ABS-like)", _RESIN),
    Material("resin-rigid", 10000, 0.33, 65, 1.65, "SLA/DLP glass-filled rigid (Rigid 10K class): brittle", _RESIN),
    Material("alsi10mg", 70000, 0.33, 230, 2.67, "LPBF aluminium, as built", _METAL),
    Material("ti64", 110000, 0.34, 900, 4.43, "LPBF titanium grade 5", _METAL),
    Material("316l", 190000, 0.30, 450, 7.98, "LPBF stainless", _METAL),
]}

_ALIASES = {
    "nylon": "pa12", "pa": "pa12", "pa 12": "pa12", "pa-12": "pa12",
    "aluminium": "alsi10mg", "aluminum": "alsi10mg", "titanium": "ti64",
    "stainless": "316l", "steel": "316l", "carbon nylon": "pa-cf",
    "tpu95a": "tpu", "tpu 95a": "tpu", "flex": "tpu",
    "pa 11": "pa11", "pa-11": "pa11", "pa12gb": "pa12-gb", "glass nylon": "pa12-gb",
    "tpu 88a": "tpu88a", "tpu-88a": "tpu88a", "sla": "resin", "standard resin": "resin", "tough resin": "resin-tough",
    "rigid resin": "resin-rigid",
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
