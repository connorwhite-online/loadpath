"""
Wires, cables and flat flexes inside a product: can this stack be wired?

Placement (placement.py) decides where the parts go; whether that layout can
be built is often decided by what joins them. A lead has a length and a
bend radius, a flat flex bends one way only and has to arrive face up, a
connector points where it points, and the cable has to get there past
everything else, be plugged in while the part is still outside, and then
stow as the part goes home. Across a joint it has to do that at every
angle of the joint, thousands of times.

This module answers those questions with physics (rods.py: a Cosserat rod
with contact) rather than a straight-line length:

- `cable(spec)`: a cable's mechanics from its construction: stranded wire
  by gauge and insulation, bundles, jacketed cables, coax, FFC and FPC
  flats; bending stiffness about both section axes, twist stiffness, mass,
  bend radius rules (static and flexing), resistance. Every number can be
  overridden from a datasheet or a measurement (`stiffness_from_droop`: a
  ruler and a table edge).
- `lay(...)`: one cable between two ports. A clearance-aware shortest path
  through the free space gives the way round the obstacles (which side of
  the battery it passes); the rod is started along it, then let go and
  brought to its real length, and settles where its stiffness and the
  walls put it. Out come its shape, tightest bend radius, twist, the force
  and moment it puts on each connector, how hard it leans on what it
  touches, and whether it is simply too short.
- `sweep(...)`: the same cable followed quasi-statically while one end (and
  the bodies it is on) moves: a part going in along its assembly path with
  its lead already plugged in, or a joint turning through its range. The
  bend radius is then held to the cable's flexing rule, and the swing in
  curvature at every station (the strain range its conductors see each
  cycle) is reported.
- `harness(brief)`: all of a product's cables from one brief (`loadpath
  wires`, `loadpath schema wires`), with an HTML viewer and, for each
  cable, a channel: the tube it occupies (over the whole motion), as a
  keep-out the interior is designed round.
- `placement_judge(...)`: the hook placement.place calls for each
  candidate position, so a layout whose cables cannot be laid is dropped
  while there are others, and among those that can the shorter, easier
  harness wins.

Limits, said plainly: contact is frictionless (a cable settles to its
lowest energy; tape, ties and friction that hold a real one elsewhere are
`via` clips you place); cables do not push the parts (parts are rigid and
fixed); two cables are kept apart by treating the ones already laid as
obstacles, in the order given; stranded conductors are modelled by their
slipped-strand stiffness, good to tens of percent unless you measure it.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .rods import GridField, MovingField, Rod, _mat_quat, frame_quat, parallel_frames, qmul, quat_matrix

__all__ = ["cable", "Cable", "lay", "sweep", "harness", "run_harness", "placement_judge", "channel_mesh",
           "stiffness_from_droop", "PolylineField", "body_field"]

_AX = {"x": 0, "y": 1, "z": 2}
_E_CU = 110e3            # MPa, annealed copper
_RHO_CU = 8.96e-3        # g/mm^3
_RES_CU = 1.72e-5        # ohm mm
# insulation and film: E (MPa), density (g/mm^3), rated temperature (C)
_INS = {"pvc": (25.0, 1.35e-3, 80.0), "pvc105": (25.0, 1.35e-3, 105.0), "silicone": (6.0, 1.15e-3, 150.0),
        "ptfe": (500.0, 2.2e-3, 200.0), "fep": (480.0, 2.15e-3, 200.0), "tpe": (30.0, 1.1e-3, 105.0),
        "xlpe": (150.0, 0.95e-3, 125.0), "pur": (40.0, 1.2e-3, 90.0), "polyimide": (2500.0, 1.42e-3, 200.0),
        "pet": (4000.0, 1.38e-3, 105.0)}


def _vec(v):
    if isinstance(v, str):
        s = v.strip().lower()
        d = np.zeros(3)
        d[_AX[s.lstrip("+-")]] = -1.0 if s.startswith("-") else 1.0
        return d
    d = np.asarray(v, float)
    return d / np.linalg.norm(d)


@dataclass
class Cable:
    """A cable's mechanics. kind 'round' (od) or 'flat' (width, thickness).
    EI in N mm^2: EI1 bends a flat through its thickness (the easy way)."""
    name: str
    kind: str
    od: float
    width: float
    thickness: float
    EI1: float
    EI2: float
    GJ: float
    grams_per_m: float
    rmin_static: float
    rmin_dynamic: float
    ohm_per_m: float | None = None
    max_temp: float | None = None
    fibre: float = 0.0          # distance from a conductor's own neutral axis to its surface, mm
    max_twist: float | None = None      # deg per m, fixed (a flat: a quarter turn in ten widths)
    max_twist_dynamic: float | None = None
    notes: list = field(default_factory=list)

    @property
    def radius(self):
        return 0.5 * (self.od if self.kind == "round" else self.thickness)

    @property
    def half_width(self):
        return 0.5 * (self.od if self.kind == "round" else self.width)

    def offsets(self):
        if self.kind == "round":
            return np.zeros((1, 3))
        w = 0.5 * self.width - self.radius
        return np.array([[u * w, 0.0, 0.0] for u in (-1.0, -0.5, 0.0, 0.5, 1.0)])

    def to_dict(self):
        d = {"name": self.name, "type": self.kind, "bendStiffness": [round(self.EI1, 4), round(self.EI2, 4)],
             "twistStiffness": round(self.GJ, 4), "gramsPerMeter": round(self.grams_per_m, 3),
             "minBendRadius": {"static": round(self.rmin_static, 3), "dynamic": round(self.rmin_dynamic, 3)}}
        if self.kind == "round":
            d["od"] = round(self.od, 3)
        else:
            d["width"], d["thickness"] = round(self.width, 3), round(self.thickness, 3)
        if self.ohm_per_m is not None:
            d["ohmPerMeter"] = round(self.ohm_per_m, 5)
        if self.max_temp is not None:
            d["maxTemp"] = self.max_temp
        if self.notes:
            d["notes"] = list(self.notes)
        return d


def _awg_d(awg: float) -> float:
    return 0.127 * 92.0 ** ((36.0 - float(awg)) / 39.0)


def _wire(spec: dict):
    """One insulated stranded (or solid) wire: (od, EI, GJ, g/m, ohm/m,
    strand diameter, rated temperature)."""
    ins = str(spec.get("insulation", "pvc")).lower()
    if ins not in _INS:
        raise ValueError(f"insulation {ins!r} is not one of {sorted(_INS)}")
    Ei, rho_i, tmax = _INS[ins]
    if spec.get("awg") is not None:
        d = _awg_d(spec["awg"])
    elif spec.get("conductorArea") is not None:
        d = float(np.sqrt(4 * float(spec["conductorArea"]) / np.pi))
    else:
        raise ValueError("a wire needs awg or conductorArea (mm^2)")
    A = np.pi * d * d / 4
    if spec.get("solid"):
        ns = 1
    elif spec.get("strands") is not None:
        ns = int(spec["strands"])
    elif ins == "silicone":
        ns = max(7, int(round(A / (np.pi * 0.04 ** 2))))        # fine 0.08 mm strands
    else:
        ns = 7 if d < 0.7 else 19
    ds = d / np.sqrt(ns)
    dc = d if ns == 1 else 1.15 * d                               # the stranded bundle's own diameter
    wall = float(spec.get("wall", 0.25 + 0.25 * dc if ins != "silicone" else 0.3 + 0.35 * dc))
    od = float(spec.get("od", dc + 2 * wall))
    k = float(spec.get("strandCoupling", 1.0))
    # strands slip on each other in any bend a product asks of a wire: each
    # bends about its own axis (the lower bound; a bonded bundle is far stiffer)
    EI = k * ns * _E_CU * np.pi * ds ** 4 / 64 + Ei * np.pi * (od ** 4 - dc ** 4) / 64
    GJ = k * ns * (_E_CU / 2.6) * np.pi * ds ** 4 / 32 + (Ei / 3.0) * np.pi * (od ** 4 - dc ** 4) / 32
    gpm = (A * _RHO_CU + np.pi * (od * od - dc * dc) / 4 * rho_i) * 1000.0
    return od, EI, GJ, gpm, _RES_CU / A * 1000.0, ds, tmax


def cable(spec) -> Cable:
    """A Cable from a spec: a name ('awg26-pvc', 'awg22-silicone',
    'ffc-10x0.5', 'fpc-6x0.11', 'usb2', 'coax-1.13') or a dict:

    {type: 'wire', awg | conductorArea, insulation, strands?, od?}
    {type: 'bundle', wires: n, awg, insulation}: n wires side by side in a
        loose sleeve or lacing (they slip on each other)
    {type: 'cable', wires: n, awg, insulation, jacket, jacketWall, od?}
    {type: 'coax', od, core (mm), dielectric: 'fep', jacket: 'fep'}
    {type: 'ffc', conductors, pitch, thickness?}: flat flexible cable
    {type: 'fpc', width, thickness, copper?, copperFill?}: a flex circuit
    {type: 'round' | 'flat', od | width + thickness, bendStiffness, ...}

    and any of: bendStiffness (N mm^2, or [easy, edgewise] for a flat),
    twistStiffness, minBendRadius (mm, or {static, dynamic}), gramsPerMeter,
    maxTemp. The bend radius rules are house rules (round wire 3x its
    diameter fixed, 10x flexing; coax 5x and 10x; flats 10x their thickness
    fixed, 100x flexing): a cable's datasheet overrides them."""
    if isinstance(spec, Cable):
        return spec
    if isinstance(spec, str):
        spec = _named(spec)
    s = dict(spec)
    kind = str(s.get("type", "wire")).lower()
    notes = []
    fibre = 0.0
    ohm = tmax = None
    if kind in ("wire", "bundle", "cable"):
        od1, EI, GJ, gpm, ohm, ds, tmax = _wire(s)
        n = int(s.get("wires", 1)) if kind != "wire" else 1
        fibre = 0.5 * ds
        if kind == "wire":
            od = od1
        elif kind == "bundle":
            od = float(s.get("od", od1 * (1.0 if n == 1 else 1.15 * np.sqrt(n))))
            EI, GJ, gpm = n * EI, n * GJ, n * gpm
        else:
            jk = str(s.get("jacket", "pvc")).lower()
            Ej, rho_j, tj = _INS[jk]
            core = od1 * (1.0 if n == 1 else 1.15 * np.sqrt(n))
            wall = float(s.get("jacketWall", 0.5))
            od = float(s.get("od", core + 2 * wall))
            Ij = np.pi * (od ** 4 - (od - 2 * wall) ** 4) / 64
            EI, GJ = n * EI + Ej * Ij, n * GJ + (Ej / 3.0) * 2 * Ij
            gpm = n * gpm + np.pi * (od ** 2 - (od - 2 * wall) ** 2) / 4 * rho_j * 1000.0
            if s.get("shield"):
                # a braid of 0.1 mm strands under the jacket, each bending by itself
                nb = int(np.pi * (od - 2 * wall) / 0.1 * 0.85)
                EI += nb * _E_CU * np.pi * 0.1 ** 4 / 64
                gpm += nb * np.pi * 0.05 ** 2 * _RHO_CU * 1000.0
            tmax = min(tmax, tj)
        # a loose bundle's wires each bend about their own axis: the rule goes by one wire
        rule = od1 if kind == "bundle" else od
        c = Cable(s.get("name", kind), "round", od, od, od, EI, EI, GJ, gpm, 3.0 * rule, 10.0 * rule, ohm, tmax, fibre)
        notes.append("stiffness from slipped strands and the insulation tube: an estimate; measure it "
                     "(stiffness_from_droop) where a design is close")
    elif kind == "coax":
        od = float(s["od"])
        core = float(s.get("core", 0.25 * od))
        Ed, rho_d, td = _INS[str(s.get("dielectric", "fep")).lower()]
        I = np.pi * od ** 4 / 64
        EI = _E_CU * np.pi * core ** 4 / 64 + Ed * (I - np.pi * core ** 4 / 64) * 0.6
        nb = int(np.pi * 0.7 * od / 0.05 * 0.85)
        EI += nb * _E_CU * np.pi * 0.05 ** 4 / 64
        gpm = (np.pi * core ** 2 / 4 * _RHO_CU + np.pi * (od ** 2 - core ** 2) / 4 * rho_d) * 1000.0
        c = Cable(s.get("name", "coax"), "round", od, od, od, EI, EI, EI / 1.3, gpm, 5.0 * od, 10.0 * od,
                  _RES_CU / (np.pi * core ** 2 / 4) * 1000.0, td, 0.5 * core)
    elif kind in ("ffc", "fpc", "flat"):
        if kind == "ffc":
            nc = int(s["conductors"])
            pitch = float(s.get("pitch", 0.5))
            w = float(s.get("width", (nc + 1) * pitch))
            t = float(s.get("thickness", 0.3))
            tcu = float(s.get("copper", 0.05 if pitch <= 0.5 else 0.1))
            fill = float(s.get("copperFill", min(0.7, nc * 0.6 * pitch / w)))
            Ef, rho_f, tmax = _INS["pet"]
            Acu = nc * 0.6 * pitch * tcu
        else:
            w, t = float(s["width"]), float(s["thickness"])
            tcu = float(s.get("copper", 0.018 if kind == "fpc" else 0.0))
            fill = float(s.get("copperFill", 0.5 if kind == "fpc" else 0.0))
            Ef, rho_f, tmax = _INS[str(s.get("film", "polyimide")).lower()]
            Acu = fill * w * tcu
        # film either side of a copper layer on the mid-plane
        EI1 = Ef * w * (t ** 3 - fill * tcu ** 3) / 12 + _E_CU * fill * w * tcu ** 3 / 12
        EA_w = Ef * (t - fill * tcu) + _E_CU * fill * tcu                  # per width
        EI2 = EA_w * w ** 3 / 12
        GJ = (Ef / 2.8) * w * t ** 3 / 3
        gpm = (w * (t - fill * tcu) * rho_f + fill * w * tcu * _RHO_CU) * 1000.0
        ohm = _RES_CU / Acu * 1000.0 if Acu > 0 else None
        c = Cable(s.get("name", kind), "flat", t, w, t, EI1, EI2, GJ, gpm, 10.0 * t, 100.0 * t, ohm, tmax,
                  0.5 * max(tcu, 0.2 * t))
    elif kind == "round":
        od = float(s["od"])
        if s.get("bendStiffness") is None:
            raise ValueError("a 'round' cable needs od and bendStiffness (N mm^2); or give its construction "
                             "(type 'wire', 'bundle', 'cable', 'coax')")
        c = Cable(s.get("name", "round"), "round", od, od, od, 0.0, 0.0, 0.0, float(s.get("gramsPerMeter", 0.0)),
                  3.0 * od, 10.0 * od, None, None, 0.25 * od)
    else:
        raise ValueError(f"cable type {kind!r} is not wire, bundle, cable, coax, ffc, fpc, round or flat")
    if s.get("bendStiffness") is not None:
        b = np.atleast_1d(np.asarray(s["bendStiffness"], float))
        c.EI1, c.EI2 = float(b[0]), float(b[-1] if len(b) > 1 else (b[0] if c.kind == "round" else c.EI2))
        if s.get("twistStiffness") is None and c.kind == "round":
            c.GJ = c.EI1 / 1.3
        c.notes = [n for n in notes if not n.startswith("stiffness from")]
    else:
        c.notes = notes
    if s.get("twistStiffness") is not None:
        c.GJ = float(s["twistStiffness"])
    mb = s.get("minBendRadius")
    if mb is not None:
        if isinstance(mb, dict):
            c.rmin_static = float(mb.get("static", c.rmin_static))
            c.rmin_dynamic = float(mb.get("dynamic", max(c.rmin_dynamic, c.rmin_static)))
        else:
            c.rmin_static = float(mb)
            c.rmin_dynamic = max(float(mb), c.rmin_dynamic) if s.get("type") in ("wire", "bundle") else float(mb)
    else:
        c.notes.append("bend radius from house rules, not a datasheet")
    if c.kind == "flat":
        c.max_twist = 90.0 / (10.0 * c.width) * 1000.0
        c.max_twist_dynamic = c.max_twist / 3.0
    else:
        c.max_twist_dynamic = 360.0           # a round cable flexing in torsion: a turn per metre
    mt = s.get("maxTwist")
    if mt is not None:
        if isinstance(mt, dict):
            c.max_twist = mt.get("static", c.max_twist)
            c.max_twist_dynamic = mt.get("dynamic", c.max_twist_dynamic)
        else:
            c.max_twist = c.max_twist_dynamic = float(mt)
    if s.get("gramsPerMeter") is not None:
        c.grams_per_m = float(s["gramsPerMeter"])
    if s.get("maxTemp") is not None:
        c.max_temp = float(s["maxTemp"])
    if c.EI1 <= 0 or c.GJ <= 0:
        raise ValueError(f"cable {c.name!r} has no stiffness")
    if isinstance(spec, dict) and spec.get("name") is None and isinstance(s.get("_name"), str):
        c.name = s["_name"]
    return c


def _named(name: str) -> dict:
    n = name.strip().lower()
    out = None
    if n.startswith("awg"):
        g, _, ins = n[3:].partition("-")
        out = {"type": "wire", "awg": float(g), "insulation": ins or "pvc"}
    elif n.startswith(("ffc-", "fpc-")):
        a, _, b = n[4:].partition("x")
        out = {"type": "ffc", "conductors": int(a), "pitch": float(b or 0.5)} if n.startswith("ffc") else \
            {"type": "fpc", "width": float(a), "thickness": float(b or 0.11)}
    elif n in ("usb2", "usb"):
        out = {"type": "cable", "wires": 4, "awg": 28, "insulation": "pvc", "jacket": "pvc", "jacketWall": 0.6,
               "shield": True}
    elif n.startswith("coax-"):
        od = {"rg174": 2.8, "rg178": 1.8, "rg316": 2.5}.get(n[5:])
        od = float(n[5:]) if od is None else od
        out = {"type": "coax", "od": od, "core": 0.17 * od}
    if out is None:
        raise ValueError(f"cable {name!r}: use awg<N>-<insulation>, ffc-<conductors>x<pitch>, "
                         "fpc-<width>x<thickness>, usb2, coax-<od>, or a dict (`loadpath schema wires`)")
    out["_name"] = name
    return out


def stiffness_from_droop(overhang: float, droop: float, grams_per_m: float) -> float:
    """Bending stiffness (N mm^2) from the simplest test there is: hold the
    cable flat on a table with `overhang` mm past the edge, measure how far
    its tip hangs below the table (`droop` mm). Solved on the elastica, so a
    floppy cable that hangs most of the way down reads right too."""
    w = grams_per_m * 1e-6 * 9.81            # N/mm
    if not 0 < droop < overhang:
        raise ValueError("droop must be between 0 and the overhang")

    def tip(EI):
        n = 40
        x = np.linspace(0, overhang, n + 1)[:, None] * np.array([1.0, 0, 0])
        rod = Rod(x, bend=EI, twist=EI, weight=[0, 0, -w])
        rod.clamp(0)
        rod.solve(stable=False)
        return -rod.x[-1, 2]
    lo = w * overhang ** 4 / (8 * droop) * 0.05
    hi = w * overhang ** 4 / (8 * droop) * 1.2
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if tip(mid) > droop:
            lo = mid
        else:
            hi = mid
    # the clamp holds half an edge: the free length is that much shorter
    return float(np.sqrt(lo * hi)) * (1 - 0.5 / 40) ** -4


# --- obstacles -----------------------------------------------------------------

class PolylineField:
    """A cable already laid, as an obstacle: distance to its centreline
    less its radius."""

    def __init__(self, points, radius: float):
        self.p = np.asarray(points, float)
        self.r = float(radius)

    def eval(self, q):
        q = np.asarray(q, float)
        a, b = self.p[:-1], self.p[1:]
        ab = b - a
        t = np.einsum("nmk,mk->nm", q[:, None, :] - a[None], ab) / np.maximum(np.einsum("mk,mk->m", ab, ab), 1e-18)
        c = a[None] + np.clip(t, 0, 1)[:, :, None] * ab[None]
        d = q[:, None, :] - c
        dist = np.linalg.norm(d, axis=2)
        i = np.argmin(dist, axis=1)
        dm = dist[np.arange(len(q)), i]
        g = d[np.arange(len(q)), i] / np.maximum(dm, 1e-12)[:, None]
        return dm - self.r, g


def body_field(shape, pitch: float = 0.4, inside: bool = False, skin: float = 0.0, pad: float = 4.0):
    """A rigid body as an obstacle field (rods.GridField). inside=True: the
    shape is a cavity the cable must stay in (an exterior, less its skin)."""
    lo, hi = shape.bounds()
    if inside:
        return GridField.sample(lambda p: -(shape.sdf(p) + skin), lo, hi, pitch, pad=pad)
    return GridField.sample(shape.sdf, lo, hi, pitch, pad=pad)


class _Room:
    """Free room (mm to the nearest obstacle) on a grid, for path finding."""

    def __init__(self, values, origin, pitch):
        self.v = np.asarray(values, np.float32)
        self.o = np.asarray(origin, float)
        self.h = float(pitch)

    @classmethod
    def of(cls, fields, lo, hi, pitch: float, cells: float = 2.5e5):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        pitch = max(float(pitch), float((np.prod(np.maximum(hi - lo, pitch)) / cells) ** (1 / 3)))
        n = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 2)
        ax = [lo[i] + np.arange(n[i]) * pitch for i in range(3)]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        v = np.full(len(P), 1e3)
        for f in fields:
            for i in range(0, len(P), 200_000):
                q = P[i:i + 200_000]
                v[i:i + 200_000] = np.minimum(v[i:i + 200_000], f.reach(q) if hasattr(f, "reach") else f.eval(q)[0])
        return cls(v.reshape(tuple(n)), lo, pitch)

    def cell(self, p):
        return tuple(np.clip(np.round((np.asarray(p, float) - self.o) / self.h).astype(int), 0,
                             np.asarray(self.v.shape) - 1))

    def at(self, p):
        from scipy import ndimage
        c = ((np.asarray(p, float).reshape(-1, 3) - self.o) / self.h).T
        return ndimage.map_coordinates(self.v, c, order=1, mode="nearest")

    def point(self, c):
        return self.o + np.asarray(c, float) * self.h

    def as_field(self):
        return GridField(self.v.astype(float), self.o, self.h)


def _free_cell(room: _Room, p, need: float, reach: int = 4):
    c = np.asarray(room.cell(p))
    if room.v[tuple(c)] >= need:
        return tuple(c)
    best = None
    n = np.asarray(room.v.shape)
    r = reach
    lo, hi = np.maximum(c - r, 0), np.minimum(c + r + 1, n)
    sub = room.v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    idx = np.argwhere(sub >= need)
    if len(idx):
        idx = idx + lo
        best = tuple(idx[np.argmin(np.linalg.norm(idx - c, axis=1))])
    return best


def route(room: _Room, a, a_dir, b, b_dir, radius: float, launch: float, avoid=None,
          prefer: float | None = None) -> np.ndarray | None:
    """The clearance-aware shortest way from port a to port b through the
    free room: leaves each port along its direction for `launch` mm, then
    Dijkstra on the grid (26 neighbours), keeping off the walls where it
    can; pulled taut and rounded. avoid: an earlier route to stay away
    from (for a second way round). None when there is no way."""
    from skimage.graph import MCP_Geometric

    a, b = np.asarray(a, float), np.asarray(b, float)
    need = radius
    ends = []
    for p, d in ((a, a_dir), (b, b_dir)):
        got = None
        for f in (1.0, 0.7, 0.4, 0.15):
            c = _free_cell(room, p + f * launch * np.asarray(d, float), need, 3)
            if c is not None:
                got = c
                break
        if got is None:
            return None
        ends.append(got)
    v = room.v
    c0 = max(3.0 * radius, 2.0 * room.h, prefer or 0.0)
    cost = 1.0 + 2.0 * np.clip((c0 - v) / c0, 0.0, 1.0)
    cost[v < need] = -1.0
    if avoid is not None:
        ax = [room.o[i] + np.arange(v.shape[i]) * room.h for i in range(3)]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        d = PolylineField(avoid, 0.0).eval(P)[0].reshape(v.shape)
        near = d < max(6.0 * radius, 4.0)
        cost = np.where((cost > 0) & near, cost * 6.0, cost)
    cost[ends[0]] = abs(cost[ends[0]])
    cost[ends[1]] = abs(cost[ends[1]])
    mcp = MCP_Geometric(cost, fully_connected=True)
    costs, _ = mcp.find_costs([ends[0]], [ends[1]])
    if not np.isfinite(costs[ends[1]]):
        return None
    cells = np.asarray(mcp.traceback(ends[1]), float)
    pts = np.vstack([a[None], room.point(cells), b[None]])
    # pull taut: drop any point whose neighbours see each other with room to spare
    keep = [0]
    i = 0
    while i < len(pts) - 1:
        j = min(len(pts) - 1, i + 40)
        while j > i + 1:
            m = max(2, int(np.linalg.norm(pts[j] - pts[i]) / (0.5 * room.h)))
            seg = pts[i][None] + np.linspace(0, 1, m)[1:-1, None] * (pts[j] - pts[i])[None]
            lim = need if (i == 0 or j == len(pts) - 1) else max(need, 0.6 * c0)
            if len(seg) == 0 or room.at(seg).min() >= lim:
                break
            j -= 1
        keep.append(j)
        i = j
    return pts[keep]


def _resample(pts, n):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n + 1)
    return np.column_stack([np.interp(u, s, pts[:, k]) for k in range(3)]), float(s[-1])


def _round(pts, room: _Room, need: float, passes: int = 30):
    """Round a polyline's corners (Laplacian smoothing), not into a wall."""
    p = pts.copy()
    for _ in range(passes):
        q = p.copy()
        q[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
        ok = room.at(q[1:-1]) >= np.minimum(need, room.at(p[1:-1]))
        p[1:-1] = np.where(ok[:, None], q[1:-1], p[1:-1])
    return p


# --- one cable -----------------------------------------------------------------

class Laid:
    """A cable laid between two ports: the rod, and how to read it."""

    def __init__(self, cab: Cable, rod: Rod, ends, wid=None):
        self.cable, self.rod, self.ends, self.id = cab, rod, ends, wid
        self.status = {}
        self.route_length = None
        self.history = []
        self.ref = [(rod.x[0].copy(), rod.q[0].copy()), (rod.x[-1].copy(), rod.q[-1].copy())]
        self.current = None

    def pose(self, end: int, R, t):
        """Carry an end by a rigid motion (R, t) from where it was laid."""
        p0, q0 = self.ref[0 if end == 0 else 1]
        self.rod.set_end(end, R @ p0 + t, qmul(_mat_quat(np.asarray(R, float)), q0))

    def snapshot(self):
        r = self.rod
        k = r.curvature()
        strain = float(np.max(np.linalg.norm(np.diff(r.x, axis=0), axis=1) / r.l - 1.0))
        return {"points": r.x.copy(), "quats": r.q.copy(), "curvature": k,
                "radius": r.bend_radius(), "contact": r.contact_state(), "stretch": strain,
                "ends": self.end_loads(), "clips": self.clip_loads()}

    def end_loads(self):
        """On each connector: the force the cable puts on it (N, world axes)
        and, from that, its pull along the connector's axis, side load and
        moment (N mm)."""
        out = []
        for e in (0, -1):
            F, Mo = self.rod.end_load(e)
            # the way the cable leaves the connector, where the connector is now
            d = quat_matrix(self.rod.q[e:e + 1] if e == 0 else self.rod.q[-1:])[0][:, 2] * (1.0 if e == 0 else -1.0)
            pull = float(-F @ d)
            out.append({"force": -F, "pull": pull, "side": float(np.linalg.norm(-F - pull * d)),
                        "moment": float(np.linalg.norm(Mo))})
        return out

    def clip_loads(self):
        """The force (N) the cable puts on each clip (`via`)."""
        holds = getattr(self, "holds", [])
        if not holds:
            return []
        _, g, _ = self.rod.assemble(hessian=False)
        return [-g[6 * i:6 * i + 3] for i, _, _ in holds]

    def report(self, dynamic: bool = False, port_limits=None) -> dict:
        """What the cable is asked to do, against what it allows."""
        c, r = self.cable, self.rod
        k = r.curvature()
        rad = r.bend_radius()
        allow = c.rmin_dynamic if dynamic else c.rmin_static
        i = int(np.argmin(rad))
        nf, m = r.internal()
        R = quat_matrix(r.q)
        tension = np.einsum("ni,ni->n", nf, R[:, :, 2])
        strain = float(np.max(np.linalg.norm(np.diff(r.x, axis=0), axis=1) / r.l - 1.0))
        cs = r.contact_state()
        # on the connector: the cable pulls it along its own way out; `force` is
        # that load in world axes, to put on the part that holds the connector
        ends = [{"pull": round(e["pull"], 4), "side": round(e["side"], 4), "moment": round(e["moment"], 3),
                 "force": np.round(e["force"], 4).tolist()} for e in self.end_loads()]
        s = np.concatenate([[0], np.cumsum(r.l)])
        out = {"cable": c.name, "length": round(float(r.l.sum()), 2),
               "minBendRadius": round(float(rad[i]), 3), "allowedBendRadius": round(allow, 3),
               "bendAt": round(float(s[i + 1]), 2),
               "maxTwist": round(float(np.degrees(np.abs(k[:, 2]).max()) * 1000.0), 1),        # deg per m
               "maxTension": round(float(tension.max()), 4),
               "stretch": round(strain, 6),
               "penetration": round(float(cs["penetration"].max()), 4),
               "contactForce": round(float(cs["force"].sum()), 4),
               "clearance": round(float(min(cs["gap"].min(), 1e3)), 3),
               "ends": {"from": ends[0], "to": ends[1]},
               "converged": bool(self.status.get("converged", False)),
               "stable": bool(self.status.get("stable", False))}
        if self.route_length is not None:
            out["routeLength"] = round(self.route_length, 2)
        clips = self.clip_loads()
        if clips:
            out["clips"] = [{"at": np.round(r.x[i], 3).tolist(), "force": np.round(f, 4).tolist(),
                             "newtons": round(float(np.linalg.norm(f)), 4)}
                            for (i, _, _), f in zip(self.holds, clips)]
        problems = []
        if not out["converged"]:
            problems.append("no equilibrium found: the cable does not settle in this space")
        if rad[i] < allow * 0.98:
            problems.append(f"bends to {rad[i]:.2f} mm radius at {s[i + 1]:.0f} mm along; "
                            f"{'flexing' if dynamic else 'fixed'} it allows {allow:.2f} mm")
        if strain > 3e-3:
            problems.append(f"too short: pulled taut ({tension.max():.1f} N)")
        if c.max_twist is not None and out["maxTwist"] > c.max_twist * 1.02:
            problems.append(f"twists {out['maxTwist']:.0f} deg/m; it allows {c.max_twist:.0f} "
                            "(a flat cable asked to turn a corner in its own plane, or to roll over too soon)")
        if cs["penetration"].max() > max(0.1, 0.25 * c.radius):
            problems.append(f"pinched: {cs['penetration'].max():.2f} mm into what it touches")
        for name, e, lim in (("from", ends[0], (port_limits or [None, None])[0]),
                             ("to", ends[1], (port_limits or [None, None])[1])):
            lim = lim or {}
            for key, lab, unit in (("maxPull", "pull", "N"), ("maxSide", "side", "N"), ("maxMoment", "moment", "N mm")):
                if lim.get(key) is not None and e[lab] > float(lim[key]):
                    problems.append(f"{lab} on the '{name}' connector {e[lab]:.2f} {unit}; it allows {lim[key]}")
        if self.current:
            I = float(self.current)
            L = float(r.l.sum()) / 1000.0
            el = {"amps": I}
            if c.ohm_per_m:
                wm = I * I * c.ohm_per_m
                # one wire in still air: h ~ 10 W/m2K over its surface (an estimate)
                rise = wm / (10.0 * np.pi * (c.od if c.kind == "round" else 2 * c.width / np.pi) * 1e-3)
                el.update({"ohms": round(c.ohm_per_m * L, 5), "voltageDrop": round(I * c.ohm_per_m * L, 4),
                           "watts": round(wm * L, 4), "riseEstimateK": round(float(rise), 1)})
                if c.max_temp is not None and 40.0 + rise > c.max_temp:
                    problems.append(f"{I} A heats it about {rise:.0f} K over the air round it; rated {c.max_temp} C")
            out["electrical"] = el
        out["problems"] = problems
        out["ok"] = not problems
        return out


def _end(e) -> dict:
    d = _vec(e["direction"])
    out = {"at": np.asarray(e["at"], float), "direction": d, "relief": float(e.get("relief", 0.0)),
           "body": e.get("body"), "limits": {k: e[k] for k in ("maxPull", "maxSide", "maxMoment") if k in e}}
    if e.get("up") is not None:
        out["up"] = _vec(e["up"])
    return out


def _segments(cab: Cable, L: float, n=None) -> int:
    if n:
        return int(n)
    h = float(np.clip(min(cab.rmin_static / 2.5, L / 30.0), 0.5, 3.0))
    return int(np.clip(np.ceil(L / h), 12, 220))


def lay(cab, a: dict, b: dict, fields=(), *, room: _Room | None = None, length: float | None = None,
        slack: float = 0.08, gravity=None, clearance: float = 0.0, segments: int | None = None, roll=None,
        avoid=None, path=None, wid=None, via=None, min_length: float | None = None,
        tol: float = 1e-6) -> Laid | None:
    """Lay one cable from port a to port b ({at, direction, up?, relief?};
    `direction` is the way the cable leaves the port) among obstacle
    `fields`. See the module docstring. length: the cable's cut length (mm,
    port to port); None: the route's length plus `slack` (and at least
    min_length). roll: whether the far end's roll about its own axis is
    held (default: yes for a flat, no for a round wire). Returns a Laid, or
    None when there is no way from a to b at all."""
    cab = cable(cab)
    a, b = (_end(a) if "limits" not in a else a), (_end(b) if "limits" not in b else b)
    fields = list(fields)
    r = cab.radius + clearance
    if path is None:
        if room is None:
            lo = np.minimum(a["at"], b["at"])
            hi = np.maximum(a["at"], b["at"])
            padv = max(15.0, 0.6 * float(np.linalg.norm(b["at"] - a["at"])), (length or 0.0) * 0.5)
            room = _Room.of(fields, lo - padv, hi + padv, float(np.clip(cab.radius, 0.6, 1.2)))
        launch = max(1.5 * room.h, cab.rmin_static * 0.8, a["relief"], b["relief"])
        path = route(room, a["at"], a["direction"], b["at"], b["direction"], r, launch, avoid=avoid,
                     prefer=cab.half_width + clearance)
        if path is None:
            return None
        fine, _ = _resample(path, max(8, int(_resample(path, 4)[1] / max(0.5 * room.h, 0.3))))
        path = _round(fine, room, r)
    pts, Lp = _resample(np.asarray(path, float), 8)
    L = float(length) if length is not None else max(Lp * (1.0 + slack), min_length or 0.0)
    n = _segments(cab, L, segments)
    x, Lp = _resample(np.asarray(path, float), n)
    # frames: the port's own at a; carried along; the roll at b spread as twist
    d1a = np.cross(a["up"], a["direction"]) if a.get("up") is not None else None
    q = parallel_frames(x, d1a)
    qa = frame_quat(a["direction"], d1a if d1a is not None else quat_matrix(q[:1])[0][:, 0])
    held_roll = (cab.kind == "flat") if roll is None else bool(roll)
    if b.get("up") is not None:
        qb = frame_quat(-b["direction"], np.cross(b["up"], -b["direction"]))
        Rl, Rb = quat_matrix(q[-1:])[0], quat_matrix(qb[None])[0]
        ang = float(np.arctan2(Rb[:, 0] @ Rl[:, 1], Rb[:, 0] @ Rl[:, 0]))
        q = parallel_frames(x, d1a, twist=ang)
    else:
        qb = frame_quat(-b["direction"], quat_matrix(q[-1:])[0][:, 0])
        held_roll = False if roll is None else held_roll
    g = None
    if gravity is not None:
        g = _vec(gravity) * cab.grams_per_m * 1e-6 * 9.81
    rod = Rod(x, q, bend=(cab.EI1, cab.EI2), twist=cab.GJ, lengths=Lp / n, weight=g, radius=cab.radius,
              offsets=cab.offsets())
    rod.clearance = clearance
    rod.obstacles = fields
    rod.set_end(0, a["at"], qa)
    rod.set_end(-1, b["at"], qb)
    rod.clamp(0).clamp(-1, roll=held_roll)
    # the first of the cable at each end is in its connector: it does not rub on
    # the part it is plugged into (the report's pinching is the cable's, not the plug's)
    arc = (np.arange(n) + 0.5) * (L / n)
    rod.exempt = (arc < max(a["relief"], 1.5 * cab.radius, 1.0)) | (L - arc < max(b["relief"], 1.5 * cab.radius, 1.0))
    holds = []
    for v in via or []:
        # a clip: the node nearest its arc length (or its point) held there
        if v.get("along") is not None:
            i = int(np.clip(round(float(v["along"]) / (L / n)), 1, n - 1))
        else:
            i = 1 + int(np.argmin(np.linalg.norm(rod.x[1:-1] - np.asarray(v["at"], float), axis=1)))
        rod.x[i] = np.asarray(v["at"], float)
        rod.hold(i)
        holds.append((i, np.asarray(v["at"], float), bool(v.get("moves"))))
    w = Laid(cab, rod, [a, b], wid)
    w.holds = holds
    w.route_length = Lp
    w.path = np.asarray(path, float)
    st = rod.solve(max_iter=400, tol=tol)
    if not st["converged"]:
        # a rough start (a corner the path cut too close): let it settle soft, then stiffen
        k = rod._contact_k()
        for f in (0.01, 0.1, 1.0):
            rod.contact_k = k * f
            st = rod.solve(max_iter=400, tol=tol)
        rod.contact_k = None
    if abs(L - Lp) > 1e-6 * L:
        steps = max(2, int(np.ceil(abs(L - Lp) / (0.03 * L))))
        dr = rod.drive(lambda s: rod.l.__setitem__(slice(None), ((1 - s) * Lp + s * L) / n), steps=steps,
                       max_iter=300, tol=tol, min_step=1 / 32)
        st = {**rod.solve(max_iter=300, tol=tol), "lengthReached": dr["reached"]} if dr["converged"] else \
            {"converged": False, "stable": False, "lengthReached": dr["reached"]}
        if not dr["converged"]:
            rod.l[:] = L / n
            st2 = rod.solve(max_iter=600, tol=tol)
            if st2["converged"]:
                st = st2
    w.status = st
    w.ref = [(rod.x[0].copy(), rod.q[0].copy()), (rod.x[-1].copy(), rod.q[-1].copy())]
    return w


# --- motion ----------------------------------------------------------------------

def rigid(motion: dict, s: float):
    """(R, t) of a motion at s in [0, 1]: {type: 'revolute', axis, origin,
    range: [deg, deg]}, {type: 'prismatic', axis, range: [mm, mm]}, or
    {type: 'path', legs: [{toward, length}]} (a part leaving the way it
    went in; s = 0 is home)."""
    kind = motion.get("type", "revolute")
    if kind == "revolute":
        lo, hi = motion.get("range", [-90, 90])
        ang = np.radians(lo + s * (hi - lo))
        ax = _vec(motion.get("axis", "z"))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        o = np.asarray(motion.get("origin", [0, 0, 0]), float)
        return R, o - R @ o
    if kind == "prismatic":
        lo, hi = motion.get("range", [0, 10])
        return np.eye(3), _vec(motion.get("axis", "z")) * (lo + s * (hi - lo))
    if kind == "path":
        legs = [(_vec(l["toward"]), float(l["length"])) for l in motion["legs"]]
        tot = sum(L for _, L in legs)
        d = s * tot
        t = np.zeros(3)
        for v, L in legs:
            t = t + v * min(max(d, 0.0), L)
            d -= L
        return np.eye(3), t
    raise ValueError(f"motion type {kind!r} is not revolute, prismatic or path")


def sweep(wires: list, motion: dict, steps: int = 24, moving=(), start: float | None = None, on_step=None) -> dict:
    """Follow laid cables while the moving bodies (rods.MovingField, with
    every cable end whose port is on one: end['moves']) go through a
    motion. The cables start where they were laid (motion at `start`, by
    default where its range passes zero, else its low end), and are walked
    to one end of the range, back through the start to the other, and
    home: the equilibria are followed, not jumped to. Every step is kept
    in each Laid's history."""
    kind = motion.get("type", "revolute")
    if start is None:
        if kind in ("revolute", "prismatic"):
            lo, hi = motion.get("range", [-90, 90])
            start = float(np.clip((0.0 - lo) / (hi - lo), 0.0, 1.0)) if hi != lo else 0.0
        else:
            start = 0.0
    R0, t0 = rigid(motion, start)

    def rel(s):
        # the motion from the pose the cables were laid in
        R, t = rigid(motion, s)
        Rr = R @ R0.T
        return Rr, t - Rr @ t0
    n_up = max(1, int(round(steps * (1 - start))))
    n_dn = max(1, int(round(steps * start)))
    seq = []
    if start < 1.0:
        seq += list(np.linspace(start, 1.0, n_up + 1)[1:])
    if start > 0.0:
        seq += list(np.linspace(1.0 if start < 1.0 else start, 0.0, (n_up if start < 1.0 else 0) + n_dn + 1)[1:])
        seq += list(np.linspace(0.0, start, n_dn + 1)[1:])
    else:
        seq += list(np.linspace(1.0, 0.0, n_up + 1)[1:])
    for w in wires:
        w.history = [{"s": start, **w.snapshot(), "converged": True}]
        w.failed_at = None
        w.snaps = []
    cur = start
    for s2 in seq:
        for w in wires:
            if w.failed_at is not None:
                continue
            s1 = cur

            def set_to(u, w=w, s1=s1, s2=s2):
                R, t = rel(s1 + u * (s2 - s1))
                for mf in moving:
                    mf.R, mf.t = R, t
                for e in (0, -1):
                    if w.ends[0 if e == 0 else 1].get("moves"):
                        w.pose(e, R, t)
                for node, p0, mv in getattr(w, "holds", []):
                    if mv:
                        w.rod.x[node] = R @ p0 + t
            nsub = max(1, int(np.ceil(_travel(w, rel(s1), rel(s2)) / (4.0 * w.rod.l.mean()))))
            dr = w.rod.drive(set_to, steps=nsub, max_iter=500, tol=1e-6)
            ok = bool(dr["converged"])
            if not ok:
                # no equilibrium next to the last one: the cable lets go and
                # finds another (it snaps through). Carry on from there
                set_to(1.0)
                st = w.rod.solve(max_iter=1500, tol=1e-6)
                ok = bool(st["converged"])
                if ok:
                    w.snaps.append(float(s1 + dr["reached"] * (s2 - s1)))
            snap = {"s": float(s2), **w.snapshot(), "converged": ok}
            w.history.append(snap)
            if not ok:
                w.failed_at = float(s1 + dr["reached"] * (s2 - s1))
        cur = s2
        if on_step:
            on_step(cur)
    R, t = rel(cur)
    for mf in moving:
        mf.R, mf.t = R, t
    return {"start": start, "sequence": [float(v) for v in seq]}


def _travel(w, T1, T2):
    d = 0.0
    for e in (0, 1):
        if w.ends[e].get("moves"):
            p0 = w.ref[e][0]
            d = max(d, float(np.linalg.norm((T2[0] @ p0 + T2[1]) - (T1[0] @ p0 + T1[1]))))
    return d


def motion_report(w: Laid, limits=None) -> dict:
    """A swept cable against its flexing rules: the tightest radius over the
    motion, the swing in curvature at each station (and the strain range
    that puts on a conductor), the twist, the worst connector loads."""
    c = w.cable
    H = [h for h in w.history if h["converged"]]
    K = np.array([h["curvature"] for h in H])                  # (steps, nodes, 3)
    rad = np.array([h["radius"] for h in H])
    kb = K[:, :, :2]
    swing = np.linalg.norm(kb.max(axis=0) - kb.min(axis=0), axis=1)        # 1/mm
    tw = np.degrees(K[:, :, 2]) * 1000.0
    pen = max(float(h["contact"]["penetration"].max()) for h in H)
    stretch = np.array([h["stretch"] for h in H])
    i = np.unravel_index(int(np.argmin(rad)), rad.shape)
    flexes = bool(swing.max() * c.rmin_dynamic > 0.05) and bool(getattr(w, "flexing", True))
    allow = c.rmin_dynamic if flexes else c.rmin_static
    out = {"steps": len(H), "minBendRadius": round(float(rad.min()), 3), "allowedBendRadius": round(allow, 3),
           "at": {"s": round(float(H[i[0]]["s"]), 4)},
           "curvatureSwing": round(float(swing.max()), 5),
           "strainRange": round(float(swing.max() * c.fibre), 6),
           "twistRange": round(float((tw.max(axis=0) - tw.min(axis=0)).max()), 1),
           "maxTwist": round(float(np.abs(tw).max()), 1),
           "stretch": round(float(stretch.max()), 6),
           "penetration": round(pen, 4), "flexes": flexes, "snaps": [round(v, 3) for v in getattr(w, "snaps", [])]}
    names = ("from", "to")
    out["ends"] = {}
    for k, nm in enumerate(names):
        E = [h["ends"][k] for h in H if h.get("ends")]
        if E:
            j = int(np.argmax([np.linalg.norm(e["force"]) for e in E]))
            out["ends"][nm] = {"pull": round(max(e["pull"] for e in E), 4), "side": round(max(e["side"] for e in E), 4),
                               "moment": round(max(e["moment"] for e in E), 3),
                               "force": np.round(E[j]["force"], 4).tolist()}
    C = [h["clips"] for h in H if h.get("clips")]
    if C:
        out["clips"] = [{"newtons": round(float(max(np.linalg.norm(c[i]) for c in C)), 4),
                         "force": np.round(max((c[i] for c in C), key=np.linalg.norm), 4).tolist()}
                        for i in range(len(C[0]))]
    problems = []
    for nm, lim in zip(names, limits or (None, None)):
        e = out["ends"].get(nm)
        for key, lab, unit in (("maxPull", "pull", "N"), ("maxSide", "side", "N"), ("maxMoment", "moment", "N mm")):
            if e and lim and lim.get(key) is not None and e[lab] > float(lim[key]):
                problems.append(f"{lab} on the '{nm}' connector reaches {e[lab]:.2f} {unit} during the motion; "
                                f"it allows {lim[key]}")
    fa = getattr(w, "failed_at", None)
    if fa is not None:
        problems.append(f"the cable cannot follow the motion past s = {fa:.3f}: it snags or runs out of length")
    if stretch.max() > 3e-3:
        j = int(np.argmax(stretch))
        problems.append(f"comes taut during the motion (s = {H[j]['s']:.2f}): too short, or on the outside of the bend")
    if out["snaps"]:
        problems.append(f"snaps through during the motion (s = {out['snaps'][0]:.2f}): slack that flips from one "
                        "side to the other each cycle")
    if rad.min() < allow * 0.98:
        problems.append(f"bends to {rad.min():.2f} mm radius during the motion; "
                        f"{'flexing' if flexes else 'fixed'} it allows {allow:.2f} mm")
    if pen > max(0.1, 0.25 * c.radius):
        problems.append(f"pinched during the motion: {pen:.2f} mm into what it touches")
    lim = c.max_twist_dynamic if flexes else c.max_twist
    if lim is not None and out["maxTwist"] > lim * 1.02:
        problems.append(f"twists {out['maxTwist']:.0f} deg/m during the motion; it allows {lim:.0f}")
    out["problems"] = problems
    out["ok"] = not problems
    return out


# --- channels -----------------------------------------------------------------

def channel_mesh(wires_or_points, radius: float | None = None, clearance: float = 0.5, tol: float = 0.15):
    """The space cables occupy, as one closed mesh (trimesh): a tube round
    each centreline (every step of a Laid's history when it has one), grown
    by `clearance`. A flat cable gets a tube of its half width. For the
    interior's keep-out, and for a shroud to hide."""
    import manifold3d as m3d
    import trimesh

    lines = []
    for w in wires_or_points if isinstance(wires_or_points, (list, tuple)) else [wires_or_points]:
        if isinstance(w, Laid):
            r = w.cable.half_width + clearance
            snaps = [h["points"] for h in w.history] or [w.rod.x]
            if len(snaps) > 12:
                snaps = [snaps[i] for i in np.unique(np.linspace(0, len(snaps) - 1, 12).astype(int))]
            lines += [(s, r) for s in snaps]
        else:
            lines.append((np.asarray(w, float), float(radius) + clearance))
    parts = []
    for pts, r in lines:
        pts = _simplify(pts, tol)
        ball = m3d.Manifold.sphere(r, 16)
        balls = [ball.translate(tuple(p)) for p in pts]
        parts += [m3d.Manifold.batch_hull([balls[i], balls[i + 1]]) for i in range(len(pts) - 1)]
    man = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))


def _simplify(pts, tol):
    """Douglas-Peucker."""
    pts = np.asarray(pts, float)
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ab = pts[j] - pts[i]
        L = np.linalg.norm(ab)
        q = pts[i + 1:j] - pts[i]
        d = np.linalg.norm(q - np.outer(q @ ab / max(L * L, 1e-18), ab), axis=1) if L > 1e-12 else np.linalg.norm(q, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


# --- a whole harness ------------------------------------------------------------

def _body_mesh(shape):
    from .shapes import trimesh_from_manifold
    m = getattr(shape, "mesh", None)
    if m is not None:
        return m
    return trimesh_from_manifold(shape.to_manifold())


def _lay_best(cab, a, b, fields, room, spec, gravity, clearance, routes: int):
    """A cable by its brief: its cut length, or the shortest of a few
    slacks that lays cleanly; the best of up to `routes` ways round."""
    length = spec.get("length")
    slacks = [None] if length is not None else \
        list(dict.fromkeys([float(spec.get("slack", 0.08)), 0.2, 0.35]))
    best = None
    avoid = None
    tried = 0
    for _ in range(max(1, int(routes))):
        got = None
        for sl in slacks:
            w = lay(cab, a, b, fields, room=room, length=length, slack=sl if sl is not None else 0.0,
                    gravity=gravity, clearance=clearance, roll=spec.get("roll"), avoid=avoid, wid=spec.get("id"),
                    via=spec.get("via"), min_length=spec.get("minLength"), segments=spec.get("segments"))
            if w is None:
                break
            tried += 1
            w.current = spec.get("current")
            rep = w.report(port_limits=[a["limits"], b["limits"]])
            key = (len(rep["problems"]), rep["length"])
            if got is None or key < got[0]:
                got = (key, w, rep)
            if rep["ok"]:
                break
        if got is None:
            break
        if best is None or got[0] < best[0]:
            best = got
        avoid = got[1].path
    return (None, None, tried) if best is None else (best[1], best[2], tried)


def harness(brief: dict, base_dir=None, on_progress=None, fields: dict | None = None,
            meshes: dict | None = None) -> dict:
    """Every cable of a brief, laid and checked. See `loadpath schema wires`.

    brief: {bodies: [{id, shape, inside?, skin?, moves?}], wires: [{id,
    cable, from, to, length | slack, current?, via?, routes?}], motion?,
    gravity?, pitch?, clearance?}. A wire's `from` and `to` are ports: {at,
    direction (the way the cable leaves), up? (a flat's face normal), body?
    (it moves with that body), relief?, maxPull?, maxSide?, maxMoment?}.
    fields / meshes: bodies already sampled ({id: field}) or meshed, to
    skip their shapes (the enclosure passes its own).

    Returns {wires: {id: report}, ok, problems, seconds, laid: [Laid]}."""
    from .shapes import shape_from_spec

    t0 = time.time()
    pitch = float(brief.get("pitch", 0.4))
    clearance = float(brief.get("clearance", 0.0))
    gravity = brief.get("gravity")
    motion = brief.get("motion")
    bodies = {}
    for i, b in enumerate(brief.get("bodies") or []):
        bid = b.get("id") or f"body-{i}"
        f = (fields or {}).get(bid)
        sh = None
        if f is None or (meshes is not None and bid not in meshes):
            spec = b["shape"]
            if isinstance(spec, str):
                spec = {"type": "mesh", "path": spec}
            sh = shape_from_spec(spec, base_dir)
        if f is None:
            if on_progress:
                on_progress({"stage": "wires", "message": f"sampling {bid}"})
            f = body_field(sh, pitch, inside=bool(b.get("inside")), skin=float(b.get("skin", 0.0)),
                           pad=4.0 + clearance + max(cable(w["cable"]).radius for w in brief.get("wires") or [{}]))
        moves = bool(b.get("moves")) and motion is not None
        bodies[bid] = {"field": MovingField(f) if moves else f, "moves": moves, "inside": bool(b.get("inside")),
                       "shape": sh, "spec": b}
    static = [b["field"] for b in bodies.values()]
    moving = [b["field"] for b in bodies.values() if b["moves"]]
    wires = brief.get("wires") or []
    if not wires:
        raise ValueError("wires: the brief lists no wires")
    ends = []
    for w in wires:
        pair = []
        for key in ("from", "to"):
            e = _end(w[key])
            if e["body"] is not None and e["body"] not in bodies:
                raise ValueError(f"wire {w.get('id')!r}: {key}.body {e['body']!r} is not a body ({list(bodies)})")
            e["moves"] = bool(e["body"] is not None and bodies[e["body"]]["moves"])
            pair.append(e)
        for v in w.get("via") or []:
            if v.get("body") is not None and v["body"] not in bodies:
                raise ValueError(f"wire {w.get('id')!r}: via.body {v['body']!r} is not a body ({list(bodies)})")
            v["moves"] = bool(v.get("body") is not None and bodies[v["body"]]["moves"])
        ends.append(pair)
    # one room grid for all the routes
    pts = np.array([e["at"] for pr in ends for e in pr])
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    Lmax = max([float(w.get("length") or 0.0) for w in wires] + [span])
    padv = max(12.0, 0.5 * Lmax)
    lo, hi = pts.min(axis=0) - padv, pts.max(axis=0) + padv
    for b in bodies.values():
        if b["inside"] and b["shape"] is not None:
            blo, bhi = b["shape"].bounds()
            lo, hi = np.maximum(lo, np.asarray(blo) - 1.0), np.minimum(hi, np.asarray(bhi) + 1.0)
    rmin = min(cable(w["cable"]).radius for w in wires)
    room = _Room.of(static, lo, hi, float(np.clip(rmin, 0.5, 1.0)))
    out, laid, problems = {}, [], []
    extra = []
    for w, (a, b) in zip(wires, ends):
        wid = w.get("id") or f"wire-{len(out)}"
        cab = cable(w["cable"])
        if on_progress:
            on_progress({"stage": "wires", "message": f"laying {wid} ({cab.name})"})
        lw, rep, tried = _lay_best(cab, a, b, static + extra, room, {**w, "id": wid}, gravity, clearance,
                                   int(w.get("routes", brief.get("routes", 1))))
        if lw is None:
            rep = {"cable": cab.name, "ok": False, "converged": False,
                   "problems": ["no way from one port to the other through the free space"]}
            out[wid] = rep
            problems.append(f"{wid}: {rep['problems'][0]}")
            continue
        rep["cableSpec"] = cab.to_dict()
        rep["tried"] = tried
        out[wid] = rep
        laid.append(lw)
        if motion is None and brief.get("wiresCollide", True):
            extra.append(PolylineField(lw.rod.x, cab.half_width))
    if motion is not None and laid:
        if on_progress:
            on_progress({"stage": "wires", "message": f"following the motion ({motion.get('type', 'revolute')})"})
        for lw in laid:
            lw.flexing = bool(motion.get("flexing", True))
        info = sweep(laid, motion, steps=int(motion.get("steps", 24)), moving=moving)
        for lw in laid:
            mr = motion_report(lw, limits=[e["limits"] for e in lw.ends])
            out[lw.id]["motion"] = mr
            if not mr["ok"]:
                out[lw.id]["ok"] = False
        out_motion = {**{k: v for k, v in motion.items()}, "start": info["start"]}
    for wid, rep in out.items():
        for p in rep.get("problems", []) + (rep.get("motion") or {}).get("problems", []):
            if f"{wid}: {p}" not in problems:
                problems.append(f"{wid}: {p}")
    res = {"name": brief.get("name", "harness"), "wires": out, "ok": not problems, "problems": problems,
           "seconds": round(time.time() - t0, 1), "laid": laid, "bodies": bodies}
    if motion is not None and laid:
        res["motion"] = out_motion
    return res


def run_harness(brief_path: str, out_dir: str, base_dir=None, on_progress=None, viewer: bool = True) -> dict:
    """`loadpath wires brief.json --out dir`: lay and check the brief's
    cables; write wires.json (the report, with each cable's centreline),
    channels/<id>.stl (the space each cable needs, over its whole motion)
    and wires.html (a viewer)."""
    with open(brief_path) as fh:
        brief = json.load(fh)
    base_dir = base_dir or os.path.dirname(os.path.abspath(brief_path))
    os.makedirs(out_dir, exist_ok=True)
    res = harness(brief, base_dir, on_progress)
    return write_harness(res, brief, out_dir, viewer=viewer)


def write_harness(res: dict, brief: dict, out_dir: str, viewer: bool = True, meshes: dict | None = None) -> dict:
    ch = brief.get("channels")
    ch = {} if ch is None or ch is True else ch
    files = {}
    if ch is not False:
        os.makedirs(os.path.join(out_dir, "channels"), exist_ok=True)
        for lw in res["laid"]:
            m = channel_mesh([lw], clearance=float(ch.get("clearance", 0.5)))
            pth = os.path.join(out_dir, "channels", f"{lw.id}.stl")
            m.export(pth)
            files[lw.id] = pth
            res["wires"][lw.id]["channel"] = {"path": os.path.relpath(pth, out_dir), "volume": round(float(m.volume), 1)}
    doc = {k: v for k, v in res.items() if k not in ("laid", "bodies")}
    for lw in res["laid"]:
        doc["wires"][lw.id]["points"] = np.round(lw.rod.x, 3).tolist()
    with open(os.path.join(out_dir, "wires.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    if viewer:
        if meshes is None:
            meshes = {}
            for bid, b in res["bodies"].items():
                if b["shape"] is not None:
                    try:
                        meshes[bid] = _body_mesh(b["shape"])
                    except Exception:
                        pass
        harness_viewer(res, meshes, os.path.join(out_dir, "wires.html"))
        doc["viewer"] = "wires.html"
    doc["channelFiles"] = files
    return doc


# --- viewer -----------------------------------------------------------------------

def _b64(a, dtype):
    import base64
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


def _lite(mesh, faces: int = 30_000):
    if len(mesh.faces) <= faces:
        return mesh
    try:
        import fast_simplification
        v, f = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces),
                                            target_count=faces)
        import trimesh
        return trimesh.Trimesh(v, f, process=False)
    except Exception:
        return mesh


def harness_viewer(res: dict, meshes: dict, path: str, title: str | None = None) -> str:
    """A self-contained page (three.js): the bodies, each cable as a tube
    or ribbon coloured by how close it is to its bend radius, the motion
    on a slider."""
    bodies = []
    for bid, m in (meshes or {}).items():
        b = (res.get("bodies") or {}).get(bid, {})
        m = _lite(m)
        bodies.append({"id": bid, "moves": bool(b.get("moves")), "inside": bool(b.get("inside")),
                       "color": (b.get("spec") or {}).get("color"),
                       "v": _b64(m.vertices, np.float32), "f": _b64(m.faces, np.uint32)})
    wires = []
    steps = None
    for lw in res["laid"]:
        c = lw.cable
        H = lw.history or [{"s": 0.0, **lw.snapshot(), "converged": True}]
        steps = [h["s"] for h in H] if steps is None else steps
        rep = res["wires"][lw.id]
        allow = (rep.get("motion") or rep)["allowedBendRadius"]
        frames = []
        for h in H:
            R = quat_matrix(h["quats"])
            # per node: the mean of its edges' width axis
            d1 = np.vstack([R[:1, :, 0], 0.5 * (R[:-1, :, 0] + R[1:, :, 0]), R[-1:, :, 0]])
            rad = np.concatenate([[np.inf], h["radius"], [np.inf]])
            sev = np.clip(allow / np.maximum(rad, 1e-9), 0, 2)
            frames.append({"p": _b64(h["points"], np.float32), "d": _b64(d1, np.float32), "s": _b64(sev, np.float32)})
        wires.append({"id": lw.id, "cable": c.name, "kind": c.kind, "r": c.radius, "w": c.half_width,
                      "n": len(lw.rod.x), "frames": frames, "ok": bool(rep["ok"]),
                      "report": {k: v for k, v in rep.items() if k not in ("cableSpec", "points")}})
    data = {"title": title or res.get("name", "harness"), "bodies": bodies, "wires": wires, "steps": steps or [0.0],
            "motion": res.get("motion"), "ok": bool(res["ok"]), "problems": res["problems"]}
    if res.get("motion"):
        mo = res["motion"]
        R0, t0 = rigid(mo, mo.get("start", 0.0))
        poses = []
        for sv in data["steps"]:
            R, t = rigid(mo, sv)
            M = np.eye(4)
            M[:3, :3] = R @ R0.T
            M[:3, 3] = t - M[:3, :3] @ t0
            poses.append(np.round(M, 6).ravel().tolist())
        data["poses"] = poses
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wires_viewer.html")) as fh:
        html = fh.read()
    html = html.replace("/*DATA*/null", json.dumps(data))
    with open(path, "w") as fh:
        fh.write(html)
    return path


# --- inside an enclosure: placement, assembly, channels -----------------------------

def _port_world(spec, byid, tf):
    """A wire end of an enclosure brief as a world port: 'component.port'
    through that component's transform (R, T), or a port given in product
    coordinates. None when its component is not placed yet."""
    if isinstance(spec, dict):
        return {**_end(spec), "component": None}
    cid, _, pid = spec.partition(".")
    if cid not in byid:
        raise ValueError(f"wire end {spec!r}: no component {cid!r}")
    port = next((p for p in byid[cid].get("ports") or [] if p.get("id") == pid), None)
    if port is None:
        raise ValueError(f"wire end {spec!r}: component {cid!r} has no port {pid!r}")
    if cid not in tf:
        return None
    R, T = tf[cid]
    e = {"at": (R @ np.asarray(port["at"], float) + T).tolist(), "direction": (R @ _vec(port["direction"])).tolist(),
         **{k: port[k] for k in ("relief", "maxPull", "maxSide", "maxMoment") if k in port}}
    if port.get("up") is not None:
        e["up"] = (R @ _vec(port["up"])).tolist()
    return {**_end(e), "component": cid}


def placement_judge(wires: list, components: list, weight: float = 0.15, physics: bool = True, on_progress=None):
    """The judge placement.place asks about each candidate position (its
    `judge`): every cable both of whose ends are now placed is routed
    through the room left and, with physics, laid as a rod. A cable with
    no way through, too short for the way there is, or that cannot be laid
    within its bend radius refuses the position (None); otherwise the
    position costs `weight` per mm of cable, and more as its tightest bend
    nears the limit."""
    byid = {c["id"]: c for c in components}
    cabs = [cable(w["cable"]) for w in wires]
    cache = {}

    def judge(info):
        tf, new = info["transforms"], set(info["new"])
        room = None
        total = 0.0
        for w, cab in zip(wires, cabs):
            a, b = _port_world(w["from"], byid, tf), _port_world(w["to"], byid, tf)
            if a is None or b is None or not ({a["component"], b["component"]} & new):
                continue
            L = w.get("length")
            if L is not None and np.linalg.norm(a["at"] - b["at"]) > float(L):
                return None
            key = (w.get("id"), tuple(np.round(np.concatenate([a["at"], a["direction"], b["at"], b["direction"]]), 2)))
            if key not in cache:
                if room is None:
                    room = _Room(info["room"], info["origin"], info["pitch"])
                cache[key] = _judge_one(cab, a, b, room, w, physics)
            if cache[key] is None:
                return None
            total += weight * cache[key]
        return total

    def quick(members, R, T, tf):
        """The same question asked cheaply of many positions at once (T:
        (n, 3) translations of the members' frames at rotation R): each
        cable's straight run costs its length; one longer than the cable
        is refused (inf)."""
        cost = np.zeros(len(T))
        mem = set(members)
        for w in wires:
            ends = []
            for spec in (w["from"], w["to"]):
                if isinstance(spec, dict):
                    ends.append(np.asarray(spec["at"], float))
                    continue
                cid, _, pid = spec.partition(".")
                port = next((p for p in (byid.get(cid) or {}).get("ports") or [] if p.get("id") == pid), None)
                if port is None:
                    ends.append(None)
                elif cid in mem:
                    ends.append(T + R @ np.asarray(port["at"], float))
                elif cid in tf:
                    ends.append(tf[cid][0] @ np.asarray(port["at"], float) + tf[cid][1])
                else:
                    ends.append(None)
            if ends[0] is None or ends[1] is None or (ends[0].ndim == 1 and ends[1].ndim == 1):
                continue
            d = np.linalg.norm(ends[0] - ends[1], axis=-1)
            cost += weight * d
            if w.get("length") is not None:
                cost[d > float(w["length"])] = np.inf
        return cost
    judge.quick = quick
    return judge


def _judge_one(cab, a, b, room, w, physics):
    r = cab.radius
    launch = max(1.5 * room.h, 0.8 * cab.rmin_static)
    path = route(room, a["at"], a["direction"], b["at"], b["direction"], r, launch, prefer=cab.half_width)
    if path is None:
        return None
    Lr = _resample(path, 4)[1]
    L = w.get("length")
    if L is not None and 0.9 * Lr > float(L):
        return None
    if not physics:
        return float(L or Lr)
    fine, _ = _resample(path, max(8, int(Lr / max(0.5 * room.h, 0.3))))
    lw = lay(cab, a, b, [room.as_field()], path=_round(fine, room, r), length=L, slack=float(w.get("slack", 0.08)),
             roll=w.get("roll"), segments=int(np.clip(Lr / max(cab.rmin_static / 2.0, 1.0), 16, 60)), tol=1e-5)
    rep = lw.report()
    hard = [p for p in rep["problems"] if not p.startswith("pinched")]     # the coarse grid over-reads pinching
    if hard:
        return None
    return rep["length"] * (1.0 + 2.0 * max(0.0, rep["allowedBendRadius"] / max(rep["minBendRadius"], 1e-6) - 0.6))


def assembly_follow(lw: Laid, legs: list, moving: list, end: int, travel: float, steps: int = 16) -> dict:
    """A cable already plugged in at both ends, followed as the part at
    `end` backs out the way it went in (legs: [(direction, mm)]), until it
    is `travel` mm out or the cable comes taut, then home again. Says how
    far out the part can be with the cable on, and whether the cable
    stows again without a kink."""
    motion = {"type": "path", "legs": [{"toward": np.asarray(d, float).tolist(), "length": float(L)} for d, L in legs]}
    total = sum(float(L) for _, L in legs)
    travel = min(float(travel), total)
    rod = lw.rod
    lw.history = [{"s": 0.0, **lw.snapshot(), "converged": True}]

    def to(sv):
        R, t = rigid(motion, sv)
        for mf in moving:
            mf.R, mf.t = R, t
        lw.pose(end, R, t)
    ds = travel / total / max(steps, 1)
    s, reached, why = 0.0, 0.0, None
    home = (rod.x.copy(), rod.q.copy())
    while s < travel / total - 1e-9:
        s2 = min(travel / total, s + ds)
        dr = rod.drive(lambda u, s=s, s2=s2: to(s + u * (s2 - s)), steps=max(1, int(np.ceil(ds * total / (4 * rod.l.mean())))),
                       max_iter=400, tol=1e-6)
        snap = lw.snapshot()
        if not dr["converged"]:
            why = "snags"
            break
        if snap["stretch"] > 3e-3:
            why = "taut"
            break
        if snap["radius"].min() < 0.98 * lw.cable.rmin_static:
            why = "bends past its radius"
            break
        s = s2
        reached = s * total
        lw.history.append({"s": float(s), **snap, "converged": True})
    out_state = (rod.x.copy(), rod.q.copy())
    # home again: the state the product is built in
    back = rod.drive(lambda u, s=s: to(s * (1 - u)), steps=max(2, len(lw.history) - 1), max_iter=400, tol=1e-6)
    if not back["converged"]:
        to(0.0)
        rod.x, rod.q = home
        rod.solve(max_iter=800, tol=1e-6)
    to(0.0)
    del out_state
    return {"reach": round(float(reached), 2), "travel": round(float(travel), 2), "stoppedBy": why,
            "returns": bool(back["converged"])}


def enclosure_wires(brief: dict, comps: list, lay_out: dict, paths: dict, root, bodies: dict, order: list,
                    exit_bounds, out_dir: str, on_progress=None) -> dict:
    """An enclosure's cables in its chosen layout (enclosure.py): each laid
    among the skin, the fixed geometry, the placed components and their
    mounts; a cable between two parts is then followed as the later part
    backs out along its assembly path, so it is known that the connector
    can be mated with the part still in the hand, and that the cable stows
    as the part goes home. Each cable's channel (over that motion) comes
    back as a keep-out. bodies: {id: (shape with .sdf and .bounds, mesh or
    None)}."""
    from .enclosure import _legs_of

    byid = {c["id"]: c for c in comps}
    tf = {cid: (np.asarray(rec["matrix"])[:3, :3], np.asarray(rec["matrix"])[:3, 3])
          for cid, rec in lay_out["components"].items()}
    pitch = float((brief.get("wiresOptions") or {}).get("pitch", 0.4))
    fields, meshes = {}, {}
    for bid, (sh, mesh) in bodies.items():
        if on_progress:
            on_progress({"stage": "wires", "message": f"sampling {bid}"})
        fields[bid] = MovingField(body_field(sh, pitch))
        if mesh is None:
            try:
                mesh = _body_mesh(sh)
            except Exception:
                mesh = None
        if mesh is not None:
            meshes[bid] = mesh
    rank = {cid: i for i, cid in enumerate(order)}
    wires = brief["wires"]
    ends = [(_port_world(w["from"], byid, tf), _port_world(w["to"], byid, tf)) for w in wires]
    pts = np.array([e["at"] for pr in ends for e in pr if e is not None])
    lo, hi = np.asarray(exit_bounds[0], float) - 1.0, np.asarray(exit_bounds[1], float) + 1.0
    room = _Room.of(list(fields.values()), lo, hi, float(np.clip(min(cable(w["cable"]).radius for w in wires), 0.5, 1.0)))
    report, laid, keep, warnings, extra = {}, [], [], [], []
    for i, (w, (a, b)) in enumerate(zip(wires, ends)):
        wid = w.get("id") or f"wire-{i}"
        cab = cable(w["cable"])
        if a is None or b is None:
            report[wid] = {"ok": False, "problems": ["an end's component has no place"]}
            warnings.append(f"wire {wid}: an end's component has no place")
            continue
        if on_progress:
            on_progress({"stage": "wires", "message": f"laying {wid} ({cab.name})"})
        static = list(fields.values()) + extra
        # the later part of the two is the one the cable follows in
        roots = [root(e["component"]) if e["component"] else None for e in (a, b)]
        mover = None
        if w.get("plugged", "before") == "before" and roots[0] != roots[1]:
            cand = [(rank.get(r, -1), k) for k, r in enumerate(roots) if r is not None and r in paths
                    and paths[r].get("insertFrom") is not None and not byid[r].get("external")]
            if cand:
                mover = max(cand)[1]
        spec = {**w, "id": wid}
        need = float(w.get("reach", 12.0))
        rep = lw = None
        follow = None
        for attempt in range(3):
            lw, rep, tried = _lay_best(cab, a, b, static, room, spec, None, 0.0, int(w.get("routes", 2)))
            if lw is None:
                break
            if mover is None:
                break
            r = roots[mover]
            legs = _legs_of(paths[r]["insertFrom"])
            grp = [c for c in byid if root(c) == r]
            blo = np.min([lay_out["components"][c]["bounds"][0] for c in grp if c in lay_out["components"]], axis=0)
            bhi = np.max([lay_out["components"][c]["bounds"][1] for c in grp if c in lay_out["components"]], axis=0)
            full, off = [], np.zeros(3)
            for d, L in legs:
                if L is None:
                    k = int(np.argmax(np.abs(d)))
                    L = (hi[k] - (blo[k] + off[k])) if d[k] > 0 else ((bhi[k] + off[k]) - lo[k])
                full.append((d, max(float(L), 1.0)))
                off = off + d * full[-1][1]
            # on the way in, the parts that go in later are not there yet, nor are the other cables
            later = {c for c in byid if rank.get(root(c), -1) > rank.get(r, -1)}
            moving = [fields[c] for c in grp if c in fields]
            lw.rod.obstacles = [f for bid, f in fields.items() if bid not in later]
            follow = assembly_follow(lw, full, moving, 0 if mover == 0 else -1, sum(L for _, L in full))
            lw.rod.obstacles = static
            st = lw.rod.solve(max_iter=600, tol=1e-6)
            lw.status = {**lw.status, **st}
            rep = lw.report(port_limits=[a["limits"], b["limits"]])
            follow["need"] = need
            if follow["reach"] >= min(need, follow["travel"]) - 1e-6 or w.get("length") is not None:
                break
            # too short to plug in with the part in the hand: more cable, and see that it still stows
            spec = {**spec, "minLength": rep["length"] + 1.25 * (need - follow["reach"])}
        if lw is None:
            report[wid] = {"cable": cab.name, "ok": False,
                           "problems": ["no way from one port to the other through the free space"]}
            warnings.append(f"wire {wid}: no way from one port to the other")
            continue
        rep["cableSpec"] = cab.to_dict()
        if follow is not None:
            rep["assembly"] = {"follows": roots[mover], **follow}
            if follow["reach"] < min(need, follow["travel"]) - 1e-6:
                rep["problems"].append(
                    f"with {roots[mover]} {follow['reach']:.0f} mm out the cable {follow['stoppedBy'] or 'stops'}: "
                    f"its connector has to be mated with the part at least {need:.0f} mm out")
                rep["ok"] = False
            if not follow["returns"]:
                rep["problems"].append(f"does not stow as {roots[mover]} goes home")
                rep["ok"] = False
        report[wid] = rep
        laid.append(lw)
        extra.append(PolylineField(lw.rod.x, cab.half_width))
        for p in rep["problems"]:
            warnings.append(f"wire {wid}: {p}")
    res = {"name": brief.get("name", "enclosure") + " wires", "wires": report, "ok": not warnings,
           "problems": warnings, "laid": laid,
           "bodies": {bid: {"moves": False, "inside": False, "spec": {}} for bid in meshes}}
    ch = float((brief.get("wiresOptions") or {}).get("channelClearance", 0.6))
    os.makedirs(os.path.join(out_dir, "cad"), exist_ok=True)
    for lw in laid:
        m = channel_mesh([lw], clearance=ch)
        pth = os.path.abspath(os.path.join(out_dir, "cad", f"wire-{lw.id}.stl"))
        m.export(pth)
        keep.append({"id": f"wire-{lw.id}", "bore": False, "shape": {"type": "mesh", "path": pth}})
        report[lw.id]["channel"] = {"keepOut": f"wire-{lw.id}", "volume": round(float(m.volume), 1)}
        report[lw.id]["points"] = np.round(lw.rod.x, 3).tolist()
        lw.history = []                       # the viewer shows the product as built
    # the same harness as a brief of its own (`loadpath wires`): to try another
    # cable, length or connector direction without placing the parts again
    try:
        os.makedirs(os.path.join(out_dir, "wires"), exist_ok=True)
        wb = {"name": res["name"], "pitch": pitch, "bodies": [], "wires": []}
        for bid, m in meshes.items():
            m.export(os.path.join(out_dir, "wires", f"{bid}.stl"))
            wb["bodies"].append({"id": bid, "shape": f"wires/{bid}.stl"})
        for i, (w, (a, b)) in enumerate(zip(wires, ends)):
            if a is None or b is None:
                continue
            ww = {k: v for k, v in w.items() if k not in ("from", "to", "reach", "plugged")}
            ww["id"] = w.get("id") or f"wire-{i}"
            for key, e in (("from", a), ("to", b)):
                ww[key] = {"at": np.round(e["at"], 4).tolist(), "direction": np.round(e["direction"], 6).tolist(),
                           **({"up": np.round(e["up"], 6).tolist()} if e.get("up") is not None else {}),
                           **({"relief": e["relief"]} if e.get("relief") else {}), **e["limits"]}
            if w.get("length") is None and ww["id"] in report and report[ww["id"]].get("length"):
                ww["length"] = report[ww["id"]]["length"]
            wb["wires"].append(ww)
        with open(os.path.join(out_dir, "wires.brief.json"), "w") as fh:
            json.dump(wb, fh, indent=1)
    except Exception as exc:
        warnings.append(f"wires brief not written: {exc}")
    try:
        harness_viewer(res, meshes, os.path.join(out_dir, "wires.html"))
    except Exception as exc:                  # the viewer is a convenience
        warnings.append(f"wires viewer not written: {exc}")
    return {"report": report, "keepOut": keep, "warnings": warnings, "laid": laid}
