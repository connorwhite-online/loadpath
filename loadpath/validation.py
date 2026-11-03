"""
Published experiments loadpath's models are checked against.

`data/published_experiments.json` holds printed lattice specimens from the
literature whose geometry is fully specified (cell type, cell size, wall or
strut, specimen size) with their measured compression modulus and strength
and the solid material's properties, each with its source. This module
turns a specimen into the inputs loadpath's checks take:

- `table_prediction`: the periodic unit cell's reading (lattice_tensors),
  modulus along the build axis and the stress at which the first wall
  yields;
- `specimen_model`: the specimen as a field graph and study (lattice cut
  to the cube, bonded 2 mm platens, the top one pushed down and held
  sideways), for `homogenized_check` or a resolved voxel solve
  (bench/published_experiments.py runs both).

Results and the comparison with the measurements: docs/VALIDATION.md and
bench/results/published-experiments.json.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent / "data" / "published_experiments.json"

__all__ = ["load_experiments", "measured", "lattice_thickness", "geometry_key", "specimen_model", "process_kind",
           "table_prediction"]


@lru_cache(maxsize=1)
def _load() -> str:
    return DATA_PATH.read_text()


def load_experiments() -> dict:
    """The published specimens, materials and sources (a fresh copy)."""
    return json.loads(_load())


def measured(spec: dict) -> dict:
    """Measured modulus and strength (MPa) of a specimen, with standard
    deviations where the paper gives them. A source that published only a
    power-law fit over its measured relative density is read at the
    specimen's measured density (sd None)."""
    m = spec["measured"]
    if "fit" in m:
        rho = float(m["rho"])
        e = m["fit"]["E"][0] * rho ** m["fit"]["E"][1]
        s = m["fit"]["strength"][0] * rho ** m["fit"]["strength"][1]
        return {"E": e, "Esd": None, "strength": s, "strengthSd": None, "strengthKind": m["strengthKind"],
                "rho": rho}
    return {"E": m["E"][0], "Esd": m["E"][1], "strength": m["strength"][0], "strengthSd": m["strength"][1],
            "strengthKind": m["strengthKind"], "rho": m.get("rho")}


def lattice_thickness(spec: dict) -> float:
    """Wall (sheet) or strut diameter (mm) of the specimen's lattice: the
    stated one, or the one that gives the paper's relative density when it
    states that (the papers make sheets by thickening the level set, not
    the distance, so matching the density is the fairer reading)."""
    from .lattice import wall_for_density
    from .struts import is_strut, strut_diameter_for_density

    lat = spec["lattice"]
    kind, cell = lat["type"], float(lat["cell"])
    if is_strut(kind):
        if "diameter" in lat:
            return float(lat["diameter"])
        return float(strut_diameter_for_density(kind, float(lat["rho"]), cell))
    if "rho" in lat:
        return float(wall_for_density(kind, float(lat["rho"]), cell))
    return float(lat["wall"])


def _rho(spec: dict) -> float:
    from .lattice import relative_density

    lat = spec["lattice"]
    return float(relative_density(lat["type"], lattice_thickness(spec), float(lat["cell"])))


def geometry_key(spec: dict) -> str:
    lat = spec["lattice"]
    size = "x".join(f"{float(s):g}" for s in spec["size"])
    return f"{lat['type']}-c{float(lat['cell']):g}-t{lattice_thickness(spec):.4f}-{size}"


def table_prediction(spec: dict, mat: dict) -> dict:
    """The infinite lattice's reading: modulus along z (MPa) and the
    uniaxial z stress (MPa) at which its peak wall stress reaches the
    solid's strength."""
    from .lattice_tensors import directional_modulus, stress_surface, tensors_at

    rho = _rho(spec)
    c, p, q, f = tensors_at(spec["lattice"]["type"], rho)
    e = directional_modulus(c[0], [0.0, 0.0, 1.0]) * float(mat["E"])
    k = float(stress_surface(np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), p[0], q[0], f[0])[0])
    return {"rho": rho, "thickness": lattice_thickness(spec), "E": e, "firstYield": float(mat["strength"]) / k,
            "stressFactor": k}


def specimen_model(spec: dict, mat: dict, plate: float = 2.0, stress: float = 1.0):
    """(field graph, study dict, loaded area mm^2, lattice height mm) of a
    compression specimen: the lattice (origin at the cube's corner) cut to
    the cube, `plate` mm solid platens bonded below and above. The bottom
    platen is held; the top one carries `stress` MPa down and is held
    sideways, as a rough platen would."""
    lat = spec["lattice"]
    sx, sy, sz = (float(s) for s in spec["size"])
    nodes = {
        "lat": {"op": "lattice", "type": lat["type"], "cell": float(lat["cell"]),
                "thickness": lattice_thickness(spec), "origin": [0.0, 0.0, 0.0]},
        "box": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [sx, sy, sz]}},
        "fill": {"op": "intersect", "inputs": ["box", "lat"]},
        "plates": {"op": "shape", "shape": {"type": "union", "shapes": [
            {"type": "box", "min": [0, 0, -plate], "max": [sx, sy, 0]},
            {"type": "box", "min": [0, 0, sz], "max": [sx, sy, sz + plate]}]}},
        "part": {"op": "union", "inputs": ["fill", "plates"]},
    }
    doc = {"name": spec["id"], "nodes": nodes, "output": "part"}
    area = sx * sy
    study = {
        "name": spec["id"],
        "designSpace": [{"type": "box", "min": [0, 0, -plate], "max": [sx, sy, sz + plate]}],
        "preserve": [{"id": "bottom", "shape": {"type": "box", "min": [0, 0, -plate], "max": [sx, sy, 0]}},
                     {"id": "top", "shape": {"type": "box", "min": [0, 0, sz], "max": [sx, sy, sz + plate]}}],
        "supports": [{"region": "bottom"}, {"region": "top", "fix": [True, True, False]}],
        "loadCases": [{"name": "compress", "loads": [{"region": "top", "force": [0, 0, -stress * area]}]}],
        "material": {"E": float(mat["E"]), "nu": float(mat["nu"]), "strength": float(mat["strength"]),
                     "density": float(mat["density"])},
        "goal": {"safetyFactor": 1.0},
    }
    return doc, study, area, sz


def process_kind(process: str) -> str:
    """loadpath's process type for a specimen's process label ("SLS PA12",
    "FDM ABS 0.25 mm", "LCD resin"): LCD is whole-layer vat exposure, dlp."""
    word = str(process).split()[0].lower()
    return {"lcd": "dlp"}.get(word, word)
