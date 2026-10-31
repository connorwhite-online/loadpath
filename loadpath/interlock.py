"""
Mechanical interlocks at the interface between two printed materials.

A flat butt joint between two materials holds only by adhesion, and some
pairs barely adhere: PETG/TPU 1.5 MPa against PETG's 47, PLA on PETG almost
nothing across layers (interface_library.py). Multi-material printers get
round that by interlocking the materials where they meet: Cura's and Bambu
Studio's "interlocking beams", after Kuipers et al.'s interlaced
topologically interlocking lattice (ITIL, Additive Manufacturing 2022,
doi:10.1016/j.addma.2021.102495).

The pattern here is that lattice. In a band of `depth` mm straddling the
interface, space is cut into beams `beamWidth` wide and `layers` print
layers tall. Layers alternate between beams along one in-plane axis and
beams along the other, and the beams alternate material. Each material
forms a woodpile connected to its own side, and the two woodpiles
interpenetrate: neither can be pulled out of the other without breaking
beams. Every beam is a whole number of extrusion widths and layers, so a
slicer prints it as solid lines.

What the interlock is credited with (`interlocked_card`), for a pair whose
flat bond is S (tensile, and shear the same way):

    S_eff = min(gain * S, cap)

* `gain` is the measured ratio of interlocked to flat strength, scaled by
  depth up to 2 mm. Frenkel et al. 2022 (Biomimetics 7:170,
  doi:10.3390/biomimetics7040170), PETG/TPU on one FDM head: an
  interdigitated interface with 2 mm overlap raised UTS by 202% over the
  butt joint (x3.0). A 2026 PLA/TPU study (Int. J. Adv. Manuf. Technol.,
  doi:10.1007/s00170-026-17902-3): 45 deg interlocking, 2 mm deep, 5.91 MPa,
  against ~1.5 MPa for a flat PLA/TPU butt (interface_library's card), so
  x3.8. The default gain of 2.5 is below both.
* `cap` is what the lattice itself can carry. In the woodpile, a plane
  across the band in a print layer cuts each material over at least a
  quarter of its area, so in-layer the lattice carries at most a quarter
  of the weaker material's strength. Between two layers, the plane is a
  quarter one material on itself, a quarter the other on itself (at the
  materials' across-layer strengths) and half the bond:
  0.25 * min(S_z) + 0.5 * S_bond,z. Shear caps at 0.6 of those.

Verification judges the smooth interface (what the optimizer designed)
against S_eff, except within `avoid` of the part's outer surface, where
no beams are put (they would leave slivers, as Cura's boundary avoidance
knows) and the bond is judged flat. The delivered bodies carry the lattice. A voxel model at
analysis pitch cannot resolve 0.8 mm beams, so the lattice is credited as
a homogenized bond and not analysed beam by beam. The default gain sits
below both sources because of that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

GAIN = 2.5              # interlocked / flat bond strength at full depth (see module docstring)
FULL_DEPTH = 2.0        # mm: the depth the gains were measured at
SHEAR_CAP = 0.6
SOURCES = [
    "Frenkel et al. 2022, Biomimetics 7:170, doi:10.3390/biomimetics7040170 (PETG/TPU, 2 mm overlap, +202% UTS)",
    "Int. J. Adv. Manuf. Technol. 2026, doi:10.1007/s00170-026-17902-3 (PLA/TPU, 45 deg interlock 2 mm, 5.91 MPa)",
    "Kuipers et al. 2022, Additive Manufacturing, doi:10.1016/j.addma.2021.102495 (ITIL lattice)",
]


@dataclass
class Interlock:
    pattern: str = "beams"
    depth: float = 2.0          # mm, total, straddling the interface
    beam_width: float = 0.8     # mm in plane (two 0.4 mm extrusions)
    layer_height: float = 0.4   # mm per beam layer (two 0.2 mm layers)
    gain: float | None = None   # None: GAIN scaled by depth
    pairs: list | None = None   # ["petg+tpu", ...]; None: every pair
    # mm: no beams this close to the part's outer surface (slivers); the
    # bond there is flat and verification judges it so. None: a beam width
    # plus half the design grid's cell (set by multimat)
    avoid: float | None = None

    def factor(self) -> float:
        g = GAIN if self.gain is None else float(self.gain)
        if self.gain is not None:
            return g
        return 1.0 + (g - 1.0) * min(1.0, self.depth / FULL_DEPTH)

    def applies(self, a: str, b: str) -> bool:
        if not self.pairs:
            return True
        key = "+".join(sorted((a, b)))
        return any("+".join(sorted(s.strip() for s in p.split("+"))) == key for p in self.pairs)

    def to_dict(self):
        return {"pattern": self.pattern, "depth": self.depth, "beamWidth": self.beam_width,
                "layerHeight": self.layer_height, "gain": self.factor(), "pairs": self.pairs}


def parse_interlock(d, process=None) -> Interlock | None:
    """The `interlock` value of a multiMaterial block: true, false, or
    {depth, beamWidth, layers | layerHeight, gain, pairs}."""
    if d is None or d is False:
        return None
    nozzle, layer = 0.4, 0.2
    if process is not None:
        layer = float(getattr(process, "layer_height", None) or layer)
        nozzle = float(getattr(process, "nozzle", None) or nozzle)
    il = Interlock(beam_width=2 * nozzle, layer_height=2 * layer)
    if d is True:
        return il
    if not isinstance(d, dict):
        raise ValueError("multiMaterial.interlock is true or {depth, beamWidth, layers, gain, pairs}")
    if d.get("pattern", "beams") != "beams":
        raise ValueError("multiMaterial.interlock.pattern: only 'beams' (an interlaced woodpile)")
    if "depth" in d:
        il.depth = float(d["depth"])
    if "beamWidth" in d:
        il.beam_width = float(d["beamWidth"])
    if "layers" in d:
        il.layer_height = int(d["layers"]) * layer
    if "layerHeight" in d:
        il.layer_height = float(d["layerHeight"])
    if "gain" in d:
        il.gain = float(d["gain"])
        if il.gain < 1.0:
            raise ValueError("multiMaterial.interlock.gain is interlocked / flat strength, >= 1")
    if "pairs" in d:
        il.pairs = [str(p) for p in d["pairs"]]
    for k, v in (("depth", il.depth), ("beamWidth", il.beam_width), ("layerHeight", il.layer_height)):
        if v <= 0:
            raise ValueError(f"multiMaterial.interlock.{k} must be > 0 (mm)")
    if il.depth < 2 * il.beam_width:
        raise ValueError(f"multiMaterial.interlock.depth {il.depth} mm holds less than two beams of "
                         f"{il.beam_width} mm: nothing interlocks")
    return il


def interlocked_card(card: dict, a, b, il: Interlock) -> dict:
    """The bond card of materials a, b (Material) with the interlock credited."""
    g = il.factor()
    weak = min(a.strength, b.strength)
    weak_z = min(a.strength_z or a.strength, b.strength_z or b.strength)
    cap = 0.25 * weak
    cap_z = 0.25 * weak_z + 0.5 * card["tensileZ"]
    out = dict(card)
    out["tensile"] = max(card["tensile"], min(g * card["tensile"], cap))
    out["shear"] = max(card["shear"], min(g * card["shear"], SHEAR_CAP * cap))
    out["tensileZ"] = max(card["tensileZ"], min(g * card["tensileZ"], cap_z))
    out["shearZ"] = max(card["shearZ"], min(g * card["shearZ"], SHEAR_CAP * cap_z))
    out["flat"] = {k: card[k] for k in ("tensile", "shear", "tensileZ", "shearZ")}
    out["interlock"] = {**il.to_dict(), "cap": cap, "capZ": cap_z, "sources": SOURCES}
    return out


def beam_pattern(xyz: np.ndarray, il: Interlock, build_axis: int = 2, phase=None) -> np.ndarray:
    """True where the first material of the pair goes, for points (..., 3):
    layer k (along the build axis) holds beams along in-plane axis u when
    k is even, along v when odd; beams alternate material across the other
    in-plane axis."""
    u, v = [a for a in range(3) if a != build_axis]
    p = xyz if phase is None else xyz - np.asarray(phase, float)
    k = np.floor(p[..., build_axis] / il.layer_height).astype(np.int64)
    iu = np.floor(p[..., u] / il.beam_width).astype(np.int64)
    iv = np.floor(p[..., v] / il.beam_width).astype(np.int64)
    # even layers: beams run along u, material set by the v index; odd: along v, set by u
    return np.where(k % 2 == 0, iv % 2 == 0, iu % 2 == 0)


def interlocked_indicator(inside: np.ndarray, spacing, origin, il: Interlock, build_axis: int = 2,
                          phase=None, mask=None) -> np.ndarray:
    """A material's region on a fine grid (bool, `inside`) with the beam
    lattice put in a band of il.depth round its boundary. `mask` (bool)
    limits the band (where the other side is a material the interlock
    applies to). Grid point (i, j, k) is at origin + index * spacing."""
    spacing = np.asarray(spacing, float)
    d_in = ndimage.distance_transform_edt(inside, sampling=spacing)
    d_out = ndimage.distance_transform_edt(~inside, sampling=spacing)
    sd = np.where(inside, d_in, -d_out)
    band = np.abs(sd) < 0.5 * il.depth
    if mask is not None:
        band &= mask
    if not band.any():
        return inside
    idx = np.argwhere(band)
    xyz = np.asarray(origin, float) + idx * spacing
    out = inside.copy()
    out[tuple(idx.T)] = beam_pattern(xyz, il, build_axis, phase)
    return out
