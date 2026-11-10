"""
Heat leaving a surface in still air: natural convection and radiation,
from the surface's own temperature, direction and surroundings.

A fixed coefficient ("8 W/(m^2 K), still air") is a guess at the answer to
a problem whose inputs are known. In still air a surface loses heat two
ways, and both depend on what the fixed number ignores:

- Buoyancy. Air warmed by the surface rises and is replaced, faster the
  hotter the surface is, so h grows with the temperature difference (as
  its 1/4 power while the boundary layer is laminar), is larger on a short
  surface than a tall one, and depends on which way the surface faces: a
  hot face looking up sheds plumes, a hot face looking down traps a
  stagnant layer and loses about half as much. Standard plate correlations
  (Churchill & Chu 1975 for vertical plates; McAdams' for horizontal ones,
  as in Incropera et al., Fundamentals of Heat and Mass Transfer, ch. 9)
  with air's properties at the film temperature.
- Radiation. Every surface radiates to the room; linearized, h_r =
  eps sigma (Ts^2 + Ta^2)(Ts + Ta), about 5-6 W/(m^2 K) for a printed
  polymer (emissivity ~0.9) near room temperature and as much as the
  convection. Bare aluminium (emissivity ~0.1) loses almost none of it,
  which is why a polymer and a metal part need different numbers.

Both are choked between facing walls. Two fins closer than their boundary
layers share the same air, and the correlation for isothermal parallel
plates (Bar-Cohen & Rohsenow 1984, after Elenbaas) falls as the gap closes;
a surface that looks at another surface of the same part radiates to it,
not to the room, so its radiation counts by its view of the surroundings.
A constant h credits a deep narrow slot with as much cooling per area as an
open face, so anything sized on it grows fine, close fins that do nothing.

`natural_h` is the coefficient per face; `openness` reads each boundary
face's gap and view of the room from a voxel grid; `NaturalConvection`
holds a thermal block's settings. thermal.StudyHeat re-solves with the
coefficients of its own temperatures until they agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NaturalConvection", "air_properties", "natural_h", "radiation_h", "openness", "EMISSIVITY"]

SIGMA = 5.670374e-8     # W/(m^2 K^4)
G = 9.80665             # m/s^2

# total hemispherical emissivity near room temperature, by what the surface is
EMISSIVITY = {"polymer": 0.90, "metal": 0.25, "anodized": 0.85}


def air_properties(t_c, pressure: float = 1.01325):
    """(rho, cp, k, mu) of dry air at t_c (C, array) and `pressure` (bar):
    heatx_thermal.fluid_at's model, vectorized."""
    tk = np.asarray(t_c, float) + 273.15
    rho = pressure * 1e5 / (287.05 * tk)
    mu = 1.716e-5 * (tk / 273.15) ** 1.5 * (273.15 + 110.4) / (tk + 110.4)
    k = 0.0241 * (tk / 273.15) ** 1.5 * (273.15 + 194.0) / (tk + 194.0)
    cp = 1002.5 + 2.75e-4 * (tk - 200.0) ** 2
    return rho, cp, k, mu


def radiation_h(t_surface, t_ambient, emissivity: float = 0.9, view=1.0):
    """Linearized radiation coefficient, W/(m^2 K), to surroundings at the
    ambient temperature, seen through a view factor `view`."""
    ts = np.asarray(t_surface, float) + 273.15
    ta = float(t_ambient) + 273.15
    return emissivity * SIGMA * (ts * ts + ta * ta) * (ts + ta) * np.asarray(view, float)


def natural_h(t_surface, t_ambient: float, facing_up, height: float, width: float, gap=None, view=1.0,
              emissivity: float = 0.9, pressure: float = 1.01325):
    """W/(m^2 K) per face: natural convection plus radiation.

    t_surface: C per face; facing_up: the face normal's component along up
    (1 a face looking up, 0 a vertical one, -1 looking down); height: the
    part's extent along up (mm), the length a vertical boundary layer
    grows over; width: its plan area over its plan perimeter (mm), the
    length for horizontal faces; gap: mm to the surface each face looks at
    (inf or None: open); view: each face's view factor to the room."""
    ts = np.asarray(t_surface, float)
    ta = float(t_ambient)
    cu = np.clip(np.asarray(facing_up, float), -1.0, 1.0)
    dt = np.maximum(np.abs(ts - ta), 0.05)
    tf = 0.5 * (ts + ta)
    rho, cp, k, mu = air_properties(tf, pressure)
    nu = mu / rho
    alpha = k / (rho * cp)
    pr = nu / alpha
    gb = G / (tf + 273.15) * dt / (nu * alpha)       # Ra / L^3
    lv = max(float(height), 1.0) * 1e-3
    lh = max(float(width), 1.0) * 1e-3
    ra_v = gb * lv ** 3
    h_vert = k / lv * (0.825 + 0.387 * ra_v ** (1.0 / 6.0) / (1.0 + (0.492 / pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    ra_h = gb * lh ** 3
    h_up = k / lh * np.where(ra_h < 1e7, 0.54 * ra_h ** 0.25, 0.15 * ra_h ** (1.0 / 3.0))
    h_down = k / lh * 0.27 * ra_h ** 0.25
    # a cold face looking down behaves as a hot one looking up
    f = np.where(ts >= ta, cu, -cu)
    w_up, w_dn = np.maximum(f, 0.0) ** 2, np.maximum(-f, 0.0) ** 2
    h = w_up * h_up + w_dn * h_down + (1.0 - w_up - w_dn) * h_vert
    if gap is not None:
        s = np.asarray(gap, float) * 1e-3
        close = np.isfinite(s) & (s > 0)
        if np.any(close):
            sc = np.where(close, s, 1.0)
            x = gb * sc ** 4 / lv                     # Ra_s * s / L
            nu_s = (576.0 / x ** 2 + 2.873 / np.sqrt(x)) ** -0.5
            h = np.where(close, np.minimum(h, nu_s * k / sc), h)
    return h + radiation_h(ts, ta, emissivity, view)


# rays over the hemisphere a face looks into, as (along the normal, along
# tangent 1, along tangent 2), with weights proportional to cos(theta) x
# the solid angle each stands for: the view factor's own weighting
_RAYS = np.array([(1.0, 0.0, 0.0)]
                 + [(np.cos(np.radians(40)), np.sin(np.radians(40)) * np.cos(a), np.sin(np.radians(40)) * np.sin(a))
                    for a in np.radians([0, 90, 180, 270])]
                 + [(np.cos(np.radians(68)), np.sin(np.radians(68)) * np.cos(a), np.sin(np.radians(68)) * np.sin(a))
                    for a in np.radians([45, 135, 225, 315])])
# rings at 0-20, 20-55 and 55-90 degrees: integral of cos sin over each
_RING = np.array([np.sin(np.radians(20)) ** 2,
                  np.sin(np.radians(55)) ** 2 - np.sin(np.radians(20)) ** 2,
                  1.0 - np.sin(np.radians(55)) ** 2])
_WEIGHT = np.concatenate([[_RING[0]], np.full(4, _RING[1] / 4), np.full(4, _RING[2] / 4)])


def openness(solid: np.ndarray, cell: np.ndarray, normal: np.ndarray, pitch: float, reach: int | None = None):
    """What each outward face of a voxel solid looks at.

    solid: bool grid; cell: (n, 3) integer indices of the solid cell each
    face belongs to; normal: (n, 3) axis unit normals. Returns (gap, view):
    the distance in mm along the normal to the next solid (inf when the
    face looks out of the grid) and the face's view factor to the
    surroundings (the cosine-weighted share of nine rays that leave the
    grid without meeting the solid)."""
    dims = np.asarray(solid.shape)
    cell = np.asarray(cell, np.int64)
    normal = np.asarray(normal, float)
    n = len(cell)
    reach = int(reach or dims.max())
    axis = np.argmax(np.abs(normal), axis=1)
    sign = np.sign(normal[np.arange(n), axis])
    t1, t2 = (axis + 1) % 3, (axis + 2) % 3
    start = cell + 0.5
    gap = np.full(n, np.inf)
    view = np.zeros(n)
    rows = np.arange(n)
    for ri, (rn, ra, rb) in enumerate(_RAYS):
        d = np.zeros((n, 3))
        d[rows, axis] = rn * sign
        d[rows, t1] = ra
        d[rows, t2] = rb
        d /= np.abs(d).max(axis=1, keepdims=True)          # one cell a step along its longest axis
        step_len = np.linalg.norm(d, axis=1)
        pos = start + (0.5 + 1e-6) * normal                  # on the face
        alive = np.ones(n, bool)
        hit_at = np.full(n, np.inf)
        for k in range(1, reach + 1):
            idx = np.nonzero(alive)[0]
            if not len(idx):
                break
            p = pos[idx] + d[idx] * (k - 0.5)
            ijk = np.floor(p).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < dims), axis=1)
            alive[idx[~inside]] = False                      # left the grid: sees the room
            ii = idx[inside]
            blocked = solid[tuple(ijk[inside].T)]
            hit = ii[blocked]
            hit_at[hit] = (k - 0.5) * step_len[hit]
            alive[hit] = False
        view += _WEIGHT[ri] * ~np.isfinite(hit_at)
        if ri == 0:
            gap = hit_at * pitch
    return gap, view


@dataclass
class NaturalConvection:
    """A thermal block's `"convection": "natural"`: still air around the
    part. emissivity: the surface's (None: by the material, polymer 0.9,
    bare metal 0.25); up: the direction opposing gravity."""
    emissivity: float | None = None
    up: tuple = (0.0, 0.0, 1.0)
    pressure: float = 1.01325

    def emissivity_for(self, material) -> float:
        if self.emissivity is not None:
            return float(self.emissivity)
        metal = getattr(material, "E", 0.0) > 20_000 or getattr(material, "density", 0.0) > 2.5
        return EMISSIVITY["metal" if metal else "polymer"]
