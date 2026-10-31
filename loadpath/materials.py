"""
Material properties for the verification pass.

The optimizer itself does not need any of this: for a linear elastic
compliance problem the optimal layout is independent of Young's modulus, so
the solver runs on E = 1. Material only enters when we turn an outcome into
numbers a person acts on (stress against strength, deflection in mm, mass in
grams).

Values come from named datasheets of printed parts, each value cited in
material_library.py (`loadpath materials` lists them); they are typical
values, not certified allowables. Printed plastics are anisotropic (weaker across layers, often by
half for FDM) and the analysis is linear, so a safety factor computed from
these is an estimate to compare designs and to catch an undersized part, not
a sign-off.
"""
from __future__ import annotations

from dataclasses import dataclass

from .material_library import LIBRARY

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
    # across layers, when the datasheet measured it (MPa); None = in-plane
    # values with the process's own factors (spec.PROCESSES)
    E_z: float | None = None
    strength_z: float | None = None
    # the process the datasheet's specimens were made by: E_z and
    # strength_z apply to a study printed that way
    measured: str = ""

    def to_dict(self):
        out = {"name": self.name, "E": self.E, "nu": self.nu,
               "strength": self.strength, "density": self.density, "note": self.note,
               "processes": list(self.processes)}
        if self.E_z is not None:
            out["EZ"] = self.E_z
        if self.strength_z is not None:
            out["strengthZ"] = self.strength_z
        if self.measured:
            out["measured"] = self.measured
        return out

    def across_layers(self):
        """(strength ratio, modulus ratio) across layers from the datasheet,
        each capped at 1, or None where it wasn't measured."""
        s = min(1.0, self.strength_z / self.strength) if self.strength_z else None
        e = min(1.0, self.E_z / self.E) if self.E_z else None
        return s, e


def _from_card(name, c):
    v = c["values"]
    get = (lambda k: float(v[k][0]) if k in v else None)
    return Material(name, get("E"), get("nu"), get("strength"), get("density"), c["title"],
                    c["processes"], E_z=get("EZ"), strength_z=get("strengthZ"),
                    measured=c.get("measured", c["processes"][0]))


# every value sourced: material_library.py (`loadpath materials <name>`)
MATERIALS = {k: _from_card(k, c) for k, c in LIBRARY.items()}

_ALIASES = {
    "nylon": "pa12", "pa": "pa12", "pa 12": "pa12", "pa-12": "pa12",
    "aluminium": "alsi10mg", "aluminum": "alsi10mg", "titanium": "ti64",
    "stainless": "316l", "steel": "316l", "carbon nylon": "pa-cf",
    "tpu95a": "tpu", "tpu 95a": "tpu", "flex": "tpu",
    "pa 11": "pa11", "pa-11": "pa11", "pa12gb": "pa12-gb", "glass nylon": "pa12-gb",
    "pa12gf": "pa12-gf", "pa 3200 gf": "pa12-gf", "pa3200gf": "pa12-gf",
    "tpu 88a": "tpu88a", "tpu-88a": "tpu88a", "sla": "resin", "standard resin": "resin", "tough resin": "resin-tough",
    "rigid resin": "resin-rigid",
    "verowhite": "vero", "vero white": "vero", "agilus": "agilus30", "agilus 30": "agilus30",
}


def material_from_spec(spec) -> Material:
    """A name from MATERIALS (or an alias), a calibrated card's name, or a
    custom {"name"?, "E", "nu", "strength", "density"} object; a custom
    object with "base": <name> starts from that material and changes only
    the keys it gives. A calibrated card (calibration.py, measured on your
    printer) named like a library material replaces it."""
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        from .calibration import list_cards, load_card
        cal = _calibrated(key, load_card)
        if cal is not None:
            return cal
        key = _ALIASES.get(key, key)
        if key in MATERIALS:
            return MATERIALS[key]
        raise ValueError(f"unknown material {spec!r}; known: {sorted(MATERIALS)}"
                         + (f", calibrated: {list_cards()}" if list_cards() else "")
                         + " or pass {E, nu, strength, density}")
    if isinstance(spec, dict):
        base = material_from_spec(spec["base"]) if spec.get("base") is not None else None
        d = dict(base.to_dict()) if base is not None else {}
        d.update({k: v for k, v in spec.items() if k != "base"})
        if base is not None and "name" not in spec:
            d["name"] = f"{base.name}-custom"
        if base is not None and "note" not in spec:
            d["note"] = f"{base.name} with {', '.join(sorted(k for k in spec if k not in ('base', 'name')))} changed"
        try:
            m = Material(str(d.get("name", "custom")), float(d["E"]), float(d.get("nu", 0.35)),
                         float(d["strength"]), float(d["density"]), str(d.get("note", "")),
                         tuple(d.get("processes") or ()),
                         E_z=float(d["EZ"]) if d.get("EZ") is not None else None,
                         strength_z=float(d["strengthZ"]) if d.get("strengthZ") is not None else None,
                         measured=str(d.get("measured") or ""))
        except KeyError as exc:
            raise ValueError(f"custom material is missing {exc.args[0]!r} (needs E, strength, density, "
                             "or a base material)") from None
        if m.E <= 0 or m.strength <= 0 or m.density <= 0 or not (0 <= m.nu < 0.5):
            raise ValueError("custom material needs E, strength, density > 0 and 0 <= nu < 0.5")
        if (m.E_z is not None and m.E_z <= 0) or (m.strength_z is not None and m.strength_z <= 0):
            raise ValueError("custom material EZ and strengthZ must be > 0")
        return m
    raise ValueError("material must be a name or {E, nu, strength, density}")


def _calibrated(key, load_card):
    """The Material of a calibrated card named `key`, or None. Its
    across-layer factors travel in the card's process block (spec.py)."""
    card = load_card(key)
    if card is None:
        return None
    m = card["material"]
    base = MATERIALS.get(_ALIASES.get(card.get("basedOn", ""), card.get("basedOn", "")))
    return Material(card["name"], float(m["E"]), float(m["nu"]), float(m["strength"]),
                    float(m["density"]), str(m.get("note", "")),
                    tuple(m.get("processes") or (base.processes if base else ())))
