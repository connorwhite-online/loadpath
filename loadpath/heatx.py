"""
Two-fluid heat exchangers from a sheet TPMS, in any core shape.

A sheet gyroid (or diamond, Schwarz P) splits space into two interleaved
channel networks that never meet: one on each side of the sheet. Give each
network its own fluid and the sheet is the wall heat crosses, with a very
large area per volume and no straight path for either flow to short-circuit.
The work is in the headers: each fluid has to reach its own network and only
that one.

The core is any closed shape: a box, a cylinder, a sphere, a union, or a
mesh (anything with a signed distance). Its surface is split into zones,
two per fluid (in and out), by the direction the surface faces, optionally
within a window along an axis:

- "cross": fluid A enters through the faces looking along -axes[0] and
  leaves through those looking along +axes[0]; fluid B the same along
  axes[1]. On a box these are its faces; on a cylinder with axes "zx", A's
  zones are the end caps and B's the two sides.
- "counter": A through the ends along axes[0]; B through the side (facing
  -axes[1]) near A's outlet and out through the side (facing +axes[1]) near
  A's inlet, so the two run against each other, the arrangement of a
  printed shell-and-tube.
- or the zones themselves, as a list.

Zones are computed on the surface, with each surface point's lateral
distance to its zone's edge (ℓ, measured along the surface), and carried to
every point in space through its nearest surface point. Then:

- each fluid's network is sealed within `cap` of the surface everywhere
  outside its own zones, so the other fluid's plenum only sees its own
  channels' mouths;
- each zone gets a plenum: the space over the zone out to `header` from the
  surface, inset `cap` from the zone's edge, whose edges are round vaults
  (a half circle from the core to the lid), so it has no corners or chamfers
  inside or out;
- what the build needs is shaped only where the chosen process demonstrably
  needs it (selfsupport.py): nothing for powder-bed polymer (SLS, MJF), whose
  powder carries every layer; for FDM, resin and powder-bed metal, a round
  span builds up to a size (about 8 mm in metal, 21 mm in FDM), and only a
  vault or level bore wider than that runs on along its tangents where it
  turns 45 degrees and closes over a round tip that size: the lowest top
  the process builds, tangent all round, never a point;
- a zone that faces up or down (along `build`, for a process that needs
  shaping) gets a vertical cone at 55 degrees instead, over a level circle
  inside the zone (so a tilted top works too), rising (or falling) to its
  port: an upward port is the top of a roof and a downward one the foot the
  part stands on;
- sideways zones keep only surface facing at most 20 degrees above level
  (a roof over more would lean over its fluid), their ports run level, and
  in FDM (`skirt`, FDM only) a 45 degree gusset holds each tube up and
  everything that overhangs the outside gets a 45 degree fill to the bed;
- each port is a tube on the zone's centre (its point farthest from the
  edge);
- with `closure` (default, with a build axis), the channels never end on a
  flat ceiling: under core surface that looks up they close over a gentle
  ramp into peaked roofs, and over a plenum below the core they flare open,
  so what hangs between the mouths leans (`report["internalCeilings"]`
  measures what is left; in powder-bed metal, `process: "lpbf"`, nothing
  inside can be supported, so it should be near zero);
- with `blend` (default a quarter cell), each plenum joins its channels by
  a smooth union, so every channel mouth opens as a rounded
  bellmouth instead of the sheet's square-cut end, never closer than a
  wall to the other fluid.

Built implicitly on one grid (distances in mm, negative = solid), meshed by
marching cubes, then checked on that same grid: the fluid spaces are labeled,
each port must reach its partner and no port of the other fluid, the
thinnest walls are measured (between the fluids and to the outside), and
the overhangs a printer would have to bridge are counted.

    hx = HeatExchangerSpec(core=Cylinder((0, 0, 0), (0, 0, 60), 20), axes=(2, 0), layout="counter")
    mesh, report = design_heat_exchanger(hx)

Units are millimetres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .lattice import tpms
from .reconstruct import _smooth_union
from .selfsupport import arch_sdf, crown_height, crown_radius, needs_shaping
from .shapes import Cylinder, Shape, shape_from_spec

__all__ = ["HeatExchangerSpec", "Zone", "design_heat_exchanger", "grid_plan", "heat_exchanger_from_spec",
           "zones_for"]

_SQRT2 = np.sqrt(2.0)
# the design's peak memory: about this many bytes per grid point, plus the interpreter.
# The design grid is float32 and the checks keep no full float64 fields: a 216 x 90 x 61 mm
# intercooler core at 0.5 mm (37M points) peaked at 9.8 GB, 5M points at 1.6 GB (it was
# 350-400 B a point while the grid was float64, which put that core out of a 15 GB machine)
_BYTES_PER_POINT = 270.0
_BASE_BYTES = 0.25e9
# report warnings, per process: an internal ceiling area no edge holds up (mm^2, faces
# flatter than the critical angle farther than half the arch chord the process closes
# from any edge: _ceiling_spans) and trapped open space (mm^3) past which powder-bed
# metal needs a look
_LPBF_FLAT_CEILING = 300.0
_LPBF_TRAPPED = 20.0
# the cone roofs over zones that face up or down lean at 55 degrees from level, a margin
# on the 45 degrees FDM and powder-bed metal build unsupported
_LEAN = 55.0
# a sideways zone keeps the surface facing at most 20 degrees above level
_MAX_UP = np.sin(np.radians(20.0))
_LOFT_SMOOTH = 1.5                 # mm, how far a lofted header's facets and rims are rounded


@dataclass
class Zone:
    """Where one fluid enters or leaves the core: the surface facing within
    acos(min_cos) of `direction`, optionally only where the position along
    `window[0]` (a unit vector) lies between the fractions window[1] and
    window[2] of the core's extent along it. Where zones overlap, each
    surface point goes to the zone it faces most."""
    fluid: int
    end: str                        # "in" or "out"
    direction: np.ndarray
    min_cos: float = 0.5
    window: tuple | None = None


@dataclass
class HeatExchangerSpec:
    core: Shape
    cell: float = 10.0              # mm per TPMS period
    wall: float = 0.8               # mm, the sheet between the fluids
    kind: str = "gyroid"
    shell: float = 1.6              # mm, outer walls
    header: float = 6.0             # mm, plenum depth over the surface
    cap: float | None = None        # mm, sealing depth under the other fluid's zones (default: wall + 1 mm)
    blend: float | None = None      # mm, smooth-union radius from plenum into its channels (default cell / 4; 0: sharp)
    exterior_blend: float = 0.0     # mm, fillet radius where the outside of headers, ports and core meet (0: sharp)
    headers: str = "flat"           # "flat": a plenum of even depth over its zone; "loft": a smooth funnel from the zone to its port
    header_ribs: int = 0            # retired (ignored, with a warning): hand-placed fins no load sized
    port_diameter: float = 8.0      # mm; or one per fluid, (A, B): an air side wants a far bigger port
    port_length: float = 8.0
    axes: tuple = (0, 1)            # cross: A along axes[0], B along axes[1]; counter: A along axes[0], B's side ports along axes[1]
    layout: str = "cross"
    zones: list | None = None       # explicit zones, instead of the layout's
    build: int | None = 2           # build axis for self-supporting roofs (None: no roof shaping)
    skirt: bool = True              # with build: fill under the outside at 45 degrees to the bed
    process: str = "fdm"            # selfsupport.SELF_SUPPORT: fdm, lpbf (powder-bed metal), sls, mjf, sla, dlp
    split: float = 0.5              # fluid A's share of the void (the sheet moves toward B)
    closure: bool = True            # with build: channels close gently under surfaces that look up, and flare open over plenums below
    cleanouts: int = 0              # powder clean-out ports per fluid, at the points farthest from its ports (plugged after)
    cleanout_diameter: float = 4.2  # mm: an M5 tap drill
    phase: tuple = (0.0, 0.0, 0.0)  # mm: shifts the TPMS against the core's low corner (where its faces cut the channels)
    step: float | None = None       # grid, mm (default wall / 3)
    max_points: int = 16_000_000
    names: tuple = ("A", "B")
    extra: dict = field(default_factory=dict)


def port_diameter(hx, fluid: int) -> float:
    """The port bore of fluid 0 or 1 (mm)."""
    d = hx.port_diameter
    return float(d[fluid]) if isinstance(d, (tuple, list)) else float(d)


def _unit(v):
    if isinstance(v, str):
        sign = -1.0 if v.startswith("-") else 1.0
        e = np.zeros(3)
        e["xyz".index(v.lstrip("+-"))] = sign
        return e
    if isinstance(v, (int, np.integer)):
        return np.eye(3)[int(v)]
    a = np.asarray(v, float).reshape(3)
    return a / np.linalg.norm(a)


def zones_for(hx: HeatExchangerSpec) -> list:
    """The surface zones: hx.zones as given, else the layout's."""
    if hx.zones:
        for f in (0, 1):
            ends = sorted(z.end for z in hx.zones if z.fluid == f)
            if ends != ["in", "out"]:
                raise ValueError(f"fluid {hx.names[f]} needs one \"in\" and one \"out\" zone, got {ends}")
        return hx.zones
    ea, eb = _unit(hx.axes[0]), _unit(hx.axes[1])
    if hx.layout == "cross":
        return [Zone(0, "in", -ea), Zone(0, "out", ea), Zone(1, "in", -eb), Zone(1, "out", eb)]
    if hx.layout == "counter":
        # A along ea through the ends; B in on one side near A's outlet and
        # out on the other side near A's inlet
        return [Zone(0, "in", -ea, 0.7), Zone(0, "out", ea, 0.7),
                Zone(1, "in", -eb, 0.4, (ea, 0.55, 1.0)), Zone(1, "out", eb, 0.4, (ea, 0.0, 0.45))]
    raise ValueError(f"heatExchanger.layout must be \"cross\" or \"counter\", got {hx.layout!r}")


def heat_exchanger_from_spec(spec: dict, base_dir=None) -> HeatExchangerSpec:
    core = spec.get("core")
    if not isinstance(core, dict):
        raise ValueError("heatExchanger.core is a shape: a box, cylinder, sphere, union or mesh, "
                         "e.g. {\"type\": \"cylinder\", \"center\": [0, 0, 30], \"axis\": \"z\", \"length\": 60, \"r\": 20}")
    hx = HeatExchangerSpec(core=shape_from_spec(core, base_dir))
    for key, attr in (("cell", "cell"), ("wall", "wall"), ("shell", "shell"), ("header", "header"),
                      ("cap", "cap"), ("blend", "blend"), ("portLength", "port_length"),
                      ("exteriorBlend", "exterior_blend")):
        if key in spec:
            setattr(hx, attr, float(spec[key]))
    if "portDiameter" in spec:
        pd = spec["portDiameter"]
        if isinstance(pd, dict):
            names = tuple(str(n) for n in spec.get("fluids", hx.names))
            if set(pd) != set(names):
                raise ValueError(f"heatExchanger.portDiameter per fluid names each of {list(names)}, e.g. "
                                 f"{{\"{names[0]}\": 8, \"{names[1]}\": 14}}")
            hx.port_diameter = tuple(float(pd[n]) for n in names)
        else:
            hx.port_diameter = float(pd)
    if "phase" in spec:
        ph = spec["phase"]
        if not (isinstance(ph, (list, tuple)) and len(ph) == 3):
            raise ValueError("heatExchanger.phase is a shift of the TPMS in mm, [x, y, z]")
        hx.phase = tuple(float(v) for v in ph)
    if "headers" in spec:
        if spec["headers"] not in ("flat", "loft"):
            raise ValueError("heatExchanger.headers is \"flat\" (even depth over the zone) or \"loft\" (a funnel to the port)")
        hx.headers = str(spec["headers"])
    if "headerRibs" in spec:
        if int(spec["headerRibs"]) < 0:
            raise ValueError("headerRibs must be 0 or more")
        # retired: the fins were placed and sized by hand, not by any load
        # (heatx.py, "The outside"); kept readable so older specs still parse
        hx.header_ribs = int(spec["headerRibs"])
    if "type" in spec:
        hx.kind = str(spec["type"])
    if "layout" in spec:
        hx.layout = str(spec["layout"])
    if "axes" in spec:
        hx.axes = tuple("xyz".index(a) for a in spec["axes"])
        if len(set(hx.axes)) != 2:
            raise ValueError("heatExchanger.axes names two different axes, e.g. \"xy\"")
    if "fluids" in spec:
        hx.names = tuple(str(n) for n in spec["fluids"])
        if len(hx.names) != 2 or hx.names[0] == hx.names[1]:
            raise ValueError("heatExchanger.fluids names two different fluids, e.g. [\"oil\", \"water\"]")
    if "build" in spec:
        b = spec["build"]
        if b not in (None, False, "none", "x", "y", "z"):
            raise ValueError("heatExchanger.build is the axis pointing up while printing: \"x\", \"y\" or \"z\" "
                             "(or null for no print shaping); turn the core to print it another way up")
        hx.build = None if b in (None, False, "none") else "xyz".index(b)
    if "process" in spec:
        from .selfsupport import SELF_SUPPORT

        hx.process = str(spec["process"])
        if hx.process not in SELF_SUPPORT:
            raise ValueError(f"heatExchanger.process is one of {sorted(SELF_SUPPORT)} (lpbf: laser powder-bed "
                             "metal, SLM / DMLS)")
        if hx.process != "fdm":
            # only FDM cannot grow an outside overhang off removable supports (powder holds it in
            # SLS and MJF; supports outside a metal or resin part come off): a skirt is only weight
            hx.skirt = False
    if "skirt" in spec:
        hx.skirt = bool(spec["skirt"])
    if "closure" in spec:
        hx.closure = bool(spec["closure"])
    if "split" in spec:
        hx.split = float(spec["split"])
        if not 0.05 < hx.split < 0.95:
            raise ValueError("heatExchanger.split is the first fluid's share of the void, between 0.05 and 0.95")
    if "cleanouts" in spec:
        hx.cleanouts = int(spec["cleanouts"])
    if "maxPoints" in spec:
        hx.max_points = int(spec["maxPoints"])
    if "cleanoutDiameter" in spec:
        hx.cleanout_diameter = float(spec["cleanoutDiameter"])
    if "zones" in spec:
        zs = []
        for z in spec["zones"]:
            try:
                fl = hx.names.index(z["fluid"]) if isinstance(z["fluid"], str) else int(z["fluid"])
                win = z.get("window")
                if win is not None:
                    win = (_unit(win["axis"]), float(win.get("from", 0.0)), float(win.get("to", 1.0)))
                zs.append(Zone(fl, str(z["end"]), _unit(z["direction"]), float(z.get("minCos", 0.5)), win))
            except (KeyError, ValueError) as e:
                raise ValueError(f"heatExchanger.zones entries are {{\"fluid\": \"A\", \"end\": \"in\", "
                                 f"\"direction\": \"-z\", \"minCos\"?: 0.5, \"window\"?: {{\"axis\": \"z\", "
                                 f"\"from\": 0, \"to\": 0.4}}}}: {e}") from None
        hx.zones = zs
    if hx.cell < 3.0 * hx.wall:
        raise ValueError("heatExchanger.cell must be at least 3 x wall, or the channels close")
    zones_for(hx)                                  # checks explicit zones
    return hx


def _grid(lo, hi, step, dtype=float):
    dims = np.maximum(np.ceil((hi - lo) / step).astype(int), 2)
    return _grid_points(lo, dims, step, dtype), dims


def _grid_points(lo, dims, step, dtype=float):
    """The points of a grid already sized (dims), in _grid's order: rebuilding
    it from lo + dims * step can round to one more cell. The design grid
    asks for float32: a car-sized exchanger at wall / 2 is 40M points, and
    float32 still resolves a 300 mm part to 0.03 um."""
    dims = [int(n) for n in dims]
    out = np.empty((int(np.prod(dims)), 3), dtype)
    view = out.reshape(*dims, 3)
    for a in range(3):
        ax = (lo[a] + (np.arange(dims[a]) + 0.5) * step).astype(dtype)
        view[..., a] = ax.reshape([-1 if b == a else 1 for b in range(3)])
    return out


def _surface_zones(hx, zones, pitch, lo, hi, up=None):
    """On a coarse grid: label the core's surface by zone and give every grid
    point its nearest surface point's lateral distance ℓ to each zone's edge
    (mm, positive inside the zone). Returns (ℓ grids, grid origin, pitch,
    per-zone info). With `up`, a zone that faces sideways keeps only the
    surface facing level or down: a plenum over a surface that faces up
    would have a roof leaning over its fluid, and its top edge a ceiling."""
    pts, dims = _grid(lo, hi, pitch)
    d = hx.core.sdf(pts).reshape(dims)
    n = np.stack(np.gradient(d, pitch), -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    surf = np.abs(d) < 0.75 * pitch
    # the core's distance far from its surface, from its inside/outside on this grid: a mesh
    # core's own field is only good near its surface
    inside = d < 0
    far = (ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * pitch
    far -= 0.5 * pitch * np.sign(far)
    # the direction off the core from every grid point (the gradient of the field the design
    # grid uses: the core's own near its surface, the far field beyond)
    g_far = np.stack(np.gradient(np.where(np.abs(far) > 3.0 * pitch, far, d), pitch), -1)
    g_far /= np.maximum(np.linalg.norm(g_far, axis=-1, keepdims=True), 1e-9)
    sp, sn = pts.reshape(*dims, 3)[surf], n[surf]
    clo, chi = hx.core.bounds()
    score = np.full((len(sp), len(zones)), -np.inf)
    for k, z in enumerate(zones):
        s = sn @ z.direction
        ok = s >= z.min_cos
        if up is not None and abs(float(z.direction @ up)) <= 0.6:
            ok &= sn @ up <= _MAX_UP
        if z.window is not None:
            ax, t0, t1 = z.window
            # the core's extent along the window axis, from its bounding box corners
            corners = np.array([[a, b, c] for a in (clo[0], chi[0]) for b in (clo[1], chi[1])
                                for c in (clo[2], chi[2])]) @ ax
            proj_lo, proj_hi = corners.min(), corners.max()
            t = (sp @ ax - proj_lo) / (proj_hi - proj_lo)
            ok &= (t >= t0) & (t <= t1)
        score[ok, k] = s[ok]
    best = np.argmax(score, axis=1)
    assigned = np.isfinite(score[np.arange(len(sp)), best])
    lab = np.full(dims, -1)
    lab[surf] = np.where(assigned, best, -1)
    # nearest surface point of every grid point
    near = ndimage.distance_transform_edt(~surf, return_distances=False, return_indices=True)
    ell, slopes, info = [], [], []
    for k, z in enumerate(zones):
        zk = lab == k
        if not zk.any():
            raise ValueError(f"zone {k} ({z.end} of fluid {hx.names[z.fluid]}, facing "
                             f"{np.round(z.direction, 2).tolist()}) covers none of the core's surface; "
                             "check axes or the zone's direction and window")
        inside = ndimage.distance_transform_edt(~(surf & ~zk)) * pitch
        outside = ndimage.distance_transform_edt(~zk) * pitch
        ls = np.where(zk, inside - 0.5 * pitch, -(outside - 0.5 * pitch))
        lk = ls[near[0], near[1], near[2]]
        lk = ndimage.gaussian_filter(lk, 1.0)
        ell.append(lk)
        if up is not None:
            # how much this zone's plenum edge is a top edge (1: the edge runs level over the
            # plenum, its vault a ceiling; 0: a side or the bottom)
            ge = np.stack(np.gradient(lk, pitch), -1)
            slopes.append(np.clip(-(ge @ up) / np.maximum(np.linalg.norm(ge, axis=-1), 1e-9), 0.0, 1.0))
        # the zone's centre: its surface point farthest from the edge
        # (the middle of its ridge, so a rectangle's port sits on its centre)
        vals = np.where(zk, inside, -1.0)
        top = vals >= vals.max() - pitch
        c = (lo + (np.argwhere(top) + 0.5) * pitch).mean(axis=0)
        for _ in range(3):                         # onto the surface
            g = np.stack([hx.core.sdf((c + e * 1e-3)[None])[0] - hx.core.sdf((c - e * 1e-3)[None])[0]
                          for e in np.eye(3)]) / 2e-3
            nc = g / np.linalg.norm(g)
            c = c - hx.core.sdf(c[None])[0] * nc
        mean_n = n[zk].sum(axis=0)
        mean_n /= np.linalg.norm(mean_n)
        # the core's normal at the nearest point well inside the zone (two grid steps from its
        # edge: an edge's own normal is the corner's mean): the zone's frame off its edges
        held = zk & (inside >= 2.0 * pitch)
        idx = ndimage.distance_transform_edt(~(held if held.any() else zk), return_distances=False, return_indices=True)
        zone_n = n[idx[0], idx[1], idx[2]]
        del idx, held
        # cos t: how square a grid point sits over that normal (1 over the zone; less round
        # its convex edges), from the same far field the design grid uses; 1 wherever the
        # nearest surface point lies a step inside the zone, so its face keeps its own frame
        cos_t = np.clip(np.einsum("...i,...i->...", g_far, zone_n), 0.0, 1.0)
        held_w = np.clip(lk / pitch, 0.0, 1.0)
        cos_t = (held_w + (1.0 - held_w) * cos_t).astype(np.float32)
        del held_w
        del zone_n
        info.append({"center": c, "normal": nc, "meanNormal": mean_n, "ellMax": float(vals[top].mean() - 0.5 * pitch),
                     "cosT": cos_t,
                     "surface": (sp, sn), "patch": (pts.reshape(*dims, 3)[zk], inside[zk] - 0.5 * pitch)})
    return ell, slopes, far, lo, pitch, info


def design_heat_exchanger(hx: HeatExchangerSpec, fluid_meshes: bool = False, grids: bool = False):
    """Returns (trimesh.Trimesh, report), and with `fluid_meshes` a third
    item: each fluid's space as a mesh, by fluid name (for renders and
    flow simulation). With `grids`, a last item: each fluid's space on the
    design grid ({"masks": (A, B) bool arrays, "lo", "step", "dims"}), for
    the whole-exchanger simulation (porous.py)."""
    import trimesh
    from skimage.measure import marching_cubes

    zones = zones_for(hx)
    blend = hx.cell * 0.25 if hx.blend is None else float(hx.blend)
    cap = hx.cap if hx.cap is not None else hx.wall + 1.0
    clo, chi = (np.asarray(b, float) for b in hx.core.bounds())
    step = hx.step or hx.wall / 3.0

    # zones on a coarse grid that reaches as far as any roof could
    span = float((chi - clo).max())
    reach_c = max(hx.header, 0.5 * span) + hx.shell + hx.port_length + 6.0
    pitch = max(2.0 * step, 0.4, (float(np.prod(chi - clo + 2 * reach_c)) / 6e6) ** (1.0 / 3.0))
    up = _shaping_up(hx)
    ell_grids, top_grids, far, c_lo, pitch, zinfo = _surface_zones(hx, zones, pitch, clo - reach_c, chi + reach_c, up)

    # roofs: a zone facing up along `build` gets a cone at 55 degrees rising to its port, a
    # zone facing down the same cone upside down, standing on its port
    warnings = []
    tan_lean = np.tan(np.radians(_LEAN))
    for k, (z, zi) in enumerate(zip(zones, zinfo)):
        zi["portDiameter"] = port_diameter(hx, z.fluid)
        r_port = 0.5 * zi["portDiameter"]
        pyramid = up is not None and abs(float(zi["meanNormal"] @ up)) > 0.6
        if pyramid:
            s = np.sign(float(zi["meanNormal"] @ up))
            # a level circle inside the zone, under a vertical cone: on a tilted top the
            # circle's rim stays inside the zone, and the cone's walls stay at 55 degrees
            cos_tilt = min(abs(float(zi["normal"] @ up)), 1.0)
            tan_tilt = np.sqrt(max(1.0 - cos_tilt ** 2, 0.0)) / max(cos_tilt, 1e-6)
            rad = (zi["ellMax"] - cap) * cos_tilt
            if rad < r_port + 1.0:
                raise ValueError(f"zone {hx.names[z.fluid]}-{z.end} is too small for a "
                                 f"{zi['portDiameter']:g} mm port under a 55 degree roof; use a smaller port")
            h = rad * tan_lean
            # the cone's walls reach down (toward the core) to the lowest surface under them
            # that faces the zone's way, so they always close on the core
            sp, sn = zi["surface"]
            rel = sp - zi["center"]
            hs = s * (rel @ up)
            under = (np.linalg.norm(rel - np.outer(hs * s, up), axis=1) <= rad + hx.shell + pitch) & \
                (s * (sn @ up) > -0.2) & (hs > -rad * max(tan_tilt, 1.0) - 2 * hx.shell - span)
            low = min(float(hs[under].min()) if under.any() else 0.0, -rad * tan_tilt) - hx.shell - 1.0
            zi.update(sign=s, radius=rad, low=low)
            a = s * up
            p0 = zi["center"] + a * 1.0
            zi["port"] = (p0, zi["center"] + a * (h + hx.shell + hx.port_length))
            zi["exit"] = zi["center"] + a * h
        else:
            h = hx.header
            # the plenum's edges are vaults of radius h / 2 (round over the zone's edge, from the
            # core to the lid): at the lid it spans the zone less h / 2 all round
            if zi["ellMax"] - cap - 0.5 * h < r_port:
                warnings.append(f"{hx.names[z.fluid]}-{z.end}: the {zi['portDiameter']:g} mm port is wider than "
                                f"the plenum at its lid ({2 * (zi['ellMax'] - cap - 0.5 * h):.1f} mm); "
                                "use a smaller port or a larger zone")
            a = zi["normal"]
            p0 = zi["center"] + a * 0.5 * h
            if up is not None:
                # sideways ports run level, for printing and plumbing alike
                a = a - (a @ up) * up
                a /= np.linalg.norm(a)
            reach_out = 0.5 * h / max(float(a @ zi["normal"]), 0.3)
            zi["port"] = (p0, p0 + a * (reach_out + hx.shell + hx.port_length))
            zi["exit"] = p0 + a * reach_out
        zi["roof"] = "pyramid" if pyramid else "vault"
        # how far the bore rises over its axis: round, unless it runs level and the process
        # cannot build a round top that size (selfsupport: then tangent flanks to a round tip)
        level = up is not None and abs(float(a @ up)) <= 0.7
        zi["crown"] = crown_height(r_port, hx.process) if level else r_port
        zi["height"] = float(h)
        zi["axis"] = a
        if not pyramid:
            # a top edge's vault spans the plenum's depth h: a half circle where the process
            # builds a round span that wide, else tangent flanks to a round tip
            zi["vaultRise"] = crown_height(0.5 * h, hx.process) / (0.5 * h) if up is not None else 1.0
            if level and zi["crown"] > r_port + 1e-9:
                # a crowned bore leaves through the lid under the top edge's vault: were its
                # crown to reach into the vault, the two crowns would meet in a ceiling wider
                # than either, which the process cannot close. Lowered just enough, if need be
                pp, depth = zi["patch"]
                top = float(((pp[depth >= 0.0] - zi["center"]) @ up).max())
                bottom = float(((pp[depth >= 0.0] - zi["center"]) @ up).min())
                spring = top - cap - 0.5 * h * zi["vaultRise"]
                drop = min(spring - zi["crown"] - hx.wall, 0.0)
                if drop < 0.0:
                    if drop - r_port < bottom + cap:
                        warnings.append(f"{hx.names[z.fluid]}-{z.end}: the {zi['portDiameter']:g} mm level port's "
                                        f"crown ({zi['crown']:.1f} mm over its axis, for {hx.process}) does not fit "
                                        "under the plenum's top vault: use a smaller port, a taller zone, or a "
                                        "build that leaves the port tilted")
                    shift = up * max(drop, bottom + cap + r_port)
                    zi["port"] = (zi["port"][0] + shift, zi["port"][1] + shift)
                    zi["exit"] = zi["exit"] + shift
                    zi["portDrop"] = float(-np.linalg.norm(shift))

    reach = max(zi["height"] for zi in zinfo) + hx.shell + hx.port_length + 4.0
    # off the round numbers a core's faces sit on, so no face lands on a sample plane
    lo, hi = clo - reach - 0.137 * step, chi + reach
    step = max(step, (float(np.prod(hi - lo)) / hx.max_points) ** (1.0 / 3.0))
    if step > hx.wall / 2.0:
        raise ValueError(_too_coarse(hx, float(np.prod(hi - lo)), step))
    pts, dims = _grid(lo, hi, step, np.float32)
    # the lowest point of the part, where a port's gusset may stand
    bed = None
    if up is not None:
        bed = min([float(clo @ up) - hx.shell] +
                  [float((zi["center"] + zi["normal"] * (zi["height"] + hx.shell)) @ up) for zi in zinfo] +
                  [float(zi["port"][1] @ up) for zi in zinfo])

    coords = ((pts - np.asarray(c_lo, np.float32)) / np.float32(pitch) - np.float32(0.5)).T
    ell = [ndimage.map_coordinates(g, coords, order=1, mode="nearest", output=np.float32) for g in ell_grids]
    d = np.asarray(hx.core.sdf(pts), np.float32)
    d_far = ndimage.map_coordinates(far, coords, order=1, mode="nearest", output=np.float32)
    d = np.where(np.abs(d_far) > 3.0 * pitch, d_far, d)
    del d_far
    g = np.asarray(tpms(hx.kind, pts - (clo + np.asarray(hx.phase, float)).astype(np.float32), hx.cell), np.float32)
    sheet_c = 0.0
    if abs(hx.split - 0.5) > 1e-9:
        from .heatx_thermal import sheet_offset

        # A is where g > offset + half: move the sheet so A gets `split` of the void
        sheet_c = sheet_offset(lambda q: tpms(hx.kind, q, 1.0), hx.wall / hx.cell, hx.split) * hx.cell
        g -= sheet_c
    half = 0.5 * hx.wall
    fluid = []
    body = [d - hx.shell]
    shape = _closure(hx, zones, ell, d, dims, step, up, cap) if up is not None and hx.closure else None
    for f in (0, 1):
        ks = [k for k, z in enumerate(zones) if z.fluid == f]
        side = g - half if f == 0 else -g - half          # > 0 inside this fluid's network
        cap_f = cap
        if shape is not None:
            side = side - shape["close"][f]
            cap_f = cap + shape["deep"][f]
        ell_f = np.max([ell[k] for k in ks], axis=0)
        # open in the core, but sealed within `cap` of the surface outside this fluid's zones
        void = np.maximum(-side, np.maximum(d, np.minimum(cap_f + d, -ell_f)))
        net, void = void, np.full_like(void, np.inf)       # plenums and bores gather in `void`
        for k in ks:
            zi = zinfo[k]
            if zi["roof"] == "pyramid":
                # under a vertical cone over a level circle, in the build frame
                hgt = zi["sign"] * ((pts - zi["center"]) @ up)
                rh = np.linalg.norm((pts - zi["center"]) - np.outer(hgt * zi["sign"], up), axis=1)
                cone = (hgt - zi["height"] + rh * tan_lean) / np.hypot(1.0, tan_lean)
                column = rh - zi["radius"]
                floor = zi["low"] - hgt                   # never past the zone's own side of the core
                inzone = cap - ell[k]                     # and never past the zone's own edge
                void = np.minimum(void, np.maximum(np.maximum(np.maximum(-d, floor), inzone),
                                                   np.maximum(cone, column)))
                body.append(np.maximum(np.maximum(cone, column) - hx.shell, floor))
                del hgt, rh, cone, column, floor, inzone
            elif hx.headers == "loft":
                # a funnel lofted from the zone (held `cap` inside its edge) to the port's root:
                # the convex hull of the two, so the outside runs in one smooth sweep from the
                # core to the port instead of a flat lid with walls round it
                pad = hx.shell + hx.exterior_blend + 2.0 * step + 6.0
                hull = _planes_sdf(_loft_planes(zi, cap), pts, pad)
                # smoothed: the hull's facets and its sharp rims round over (planes stay planes)
                hull = ndimage.gaussian_filter(hull.reshape(dims), _LOFT_SMOOTH / step).ravel()
                void = np.minimum(void, np.maximum(hull, -d))
                body.append(hull - hx.shell)
                del hull
            else:
                # a plenum of even depth whose edges are round vaults from the core to the lid,
                # a half circle over the zone's edge; over a top edge the vault rises to a round
                # tip on tangent flanks where the process cannot build that round span (selfsupport)
                rise = 1.0
                if up is not None:
                    top = ndimage.map_coordinates(top_grids[k], coords, order=1, mode="nearest", output=np.float32)
                    rise = 1.0 + (zi["vaultRise"] - 1.0) * top
                    del top
                # (d, ell) are the distance off the core and the lateral distance of the nearest
                # surface point: over the zone a plane's own coordinates, but around a convex edge
                # of the core polar ones (every point round the edge has the edge's ell), which
                # would sweep the vault's section round the edge into a bulge under the part. In
                # the zone's own frame a point off the edge is d cos t out along its normal and
                # d sin t past its edge, t the angle between grad d and the zone's normal at its
                # nearest point well inside the zone (on a curved zone, its own local normal)
                cos_t = ndimage.map_coordinates(zi["cosT"], coords, order=1, mode="nearest", output=np.float32)
                out = d > 0.0
                d_z = np.where(out, d * cos_t, d)
                ell_z = np.where(out, ell[k] - d * np.sqrt(1.0 - cos_t * cos_t), ell[k])
                del cos_t, out
                vault = _vault(d_z, ell_z, cap, 0.5 * zi["height"], rise, crown_radius(hx.process))
                del d_z, ell_z
                void = np.minimum(void, vault)
                body.append(vault - hx.shell)
                del vault
            # evaluated only near the port: its fields are far positive elsewhere
            sel = _port_box(hx, zi, pts, up, bed)
            bore, tube = _port(hx, zi, pts[sel], up, bed)
            # (a port's root stays sharp: rounded, it would leave a ceiling under the plenum's roof)
            void[sel] = np.minimum(void[sel], bore)
            xb = hx.exterior_blend
            body[-1][sel] = _smooth_union(body[-1][sel], tube, xb) if xb > 0 else np.minimum(body[-1][sel], tube)
        fluid.append((net, void))
        del net, void
    if shape is not None:
        # flare each fluid's channels open toward a plenum below the core, never
        # within a wall (and two grid steps) of the other fluid's space
        for f in (0, 1):
            pot = shape["flare"][f]
            if not pot.any():
                continue
            other = (fluid[1 - f][0] < 0).reshape(dims)
            room = (ndimage.distance_transform_edt(~other).astype(np.float32).ravel() * np.float32(step)
                    if other.any() else np.full(len(d), np.inf, np.float32))
            del other
            fl = pot * np.clip((room - hx.wall - 2.0 * step) / step, 0.0, 1.0)
            del room
            side = (g - half if f == 0 else -g - half) - shape["close"][f] + fl
            ell_f = np.max([ell[k] for k, z in enumerate(zones) if z.fluid == f], axis=0)
            flared = np.maximum(-side, np.maximum(d, np.minimum(cap + shape["deep"][f] + d, -ell_f)))
            fluid[f] = (np.minimum(fluid[f][0], flared), fluid[f][1])
            del side, flared, fl
    sharp = [np.minimum(net, void) for net, void in fluid]
    if blend > 0:
        # the plenum flows into its channels through rounded mouths (a bellmouth on every
        # opening) rather than over the sheet's square-cut ends: the rounding only cuts into
        # the core (the plenum's own walls keep their shapes), may eat the sheet's end, but
        # never comes within a wall of the other fluid
        out = []
        for f, (net, void) in enumerate(fluid):
            other = sharp[1 - f] if f == 0 else out[0]
            # (a step over the wall: the fields are near-distances, the check is on the grid)
            rounded = np.maximum(np.maximum(_smooth_union(net, void, blend), d), hx.wall + step - other)
            out.append(np.minimum(sharp[f], rounded))
        fluid = out
    else:
        fluid = sharp
    del sharp
    if hx.exterior_blend > 0:
        # the outside blends where core, headers and ports meet: the fillets only add
        # material outside the sharp union, so the fluid spaces are untouched
        d_body = body[0]
        for b in body[1:]:
            d_body = _smooth_union(d_body, b, hx.exterior_blend)
    else:
        d_body = np.min(body, axis=0)
    if hx.headers == "loft":
        # one smooth skin: the creases where lofts, core and ports meet round over too
        # (a smoothed skin thins over convex spots: the shell is put back wherever the sharp
        # body had it, within a shell of either fluid)
        keep = np.where(d_body <= 0.0, np.maximum(d_body, np.minimum(fluid[0], fluid[1]) - hx.shell), np.inf)
        d_body = np.minimum(ndimage.gaussian_filter(d_body.reshape(dims), _LOFT_SMOOTH / step).ravel(), keep)
        del keep
        # and each port's tube left as drawn: a true cylinder, cut square at its end
        for zi in zinfo:
            sel = _port_box(hx, zi, pts, up, bed)
            p0, p1 = zi["port"]
            ax = (p1 - p0) / np.linalg.norm(p1 - p0)
            q = pts[sel]
            tube = Cylinder(p0, p1, 0.5 * zi["portDiameter"] + hx.shell).sdf(q)
            d_body[sel] = np.minimum(np.maximum(d_body[sel], (q - p1) @ ax), tube)
            del q, tube
    del body, ell, coords
    # No wall thicker than it has to be. The outside is a blend of the core's,
    # headers' and ports' own outsides, and the blend and the smoothing fill
    # every concave junction between them (where a plenum meets the core's
    # corner, or two plenums meet) with solid metal: up to 6 mm where the
    # walls are 1.6, weight that holds no pressure and moves no heat. The
    # outside is held to what needs a wall: the fluid spaces grown by the
    # shell, and the core grown by the shell over its seal; their union
    # blended over `exteriorBlend`, so a junction keeps its fillet and
    # nothing more.
    xb = hx.exterior_blend
    walls = [fluid[0] - np.float32(hx.shell + step), fluid[1] - np.float32(hx.shell + step), d - np.float32(hx.shell)]
    need = walls[0]
    for w in walls[1:]:
        need = _smooth_union(need, w, xb) if xb > 0 else np.minimum(need, w)
    d_body = np.maximum(d_body, need)
    del walls, need
    grid = np.maximum(d_body, -np.minimum(fluid[0], fluid[1])).reshape(dims)
    if up is not None and hx.skirt:
        grid = _skirt(grid, d_body.reshape(dims), int(np.argmax(np.abs(up))), step)

    cleanouts, powder = [], None
    if hx.cleanouts > 0 or hx.process == "lpbf":
        _, masks0 = _check(hx, zones, zinfo, grid, g - half, d_body, d, dims, step, lo, cap)
        grid, cleanouts, powder = _cleanouts(hx, zones, zinfo, grid, masks0, d_body, d, lo, step, dims)
        del masks0

    # no vertex on (or within rounding of) a grid point: two edges' vertices there would merge
    # and leave the mesh non-manifold (a round vault's crown can sit a hair off a sample plane)
    eps = np.float32(1e-3 * step)
    near0 = np.abs(grid) < eps
    grid[near0] = np.where(grid[near0] < 0, -eps, eps)
    del near0
    verts, faces, _, _ = marching_cubes(grid, 0.0, spacing=(step, step, step), allow_degenerate=False)
    mesh = trimesh.Trimesh(verts + lo + 0.5 * step, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()

    report, masks = _check(hx, zones, zinfo, grid, (side_a := g - half), d_body, d, dims, step, lo, cap,
                           mouths=[(c["plug"]["a"], c["plug"]["b"], c["diameter"] / 2) for c in cleanouts])
    if powder is not None:
        report["powder"] = powder
    if cleanouts:
        report["cleanouts"] = cleanouts
    report.update(_overhangs(hx, mesh, up))
    if up is not None:
        report["internalCeilings"] = _internal_ceilings(hx, mesh, masks, lo, step, up, clo, cap, sheet_c)
    report.update({"kind": hx.kind, "cell": hx.cell, "wall": hx.wall, "step": step, "cap": cap, "blend": blend,
                   "solidVolume": float(mesh.volume), "watertight": bool(mesh.is_watertight),
                   "bounds": [mesh.bounds[0].round(2).tolist(), mesh.bounds[1].round(2).tolist()],
                   "zones": {f"{hx.names[z.fluid]}-{z.end}": {
                       "roof": zi["roof"], "center": np.round(zi["center"], 2).tolist(), "plenumHeight": round(zi["height"], 2),
                       "port": [np.round(zi["port"][1], 2).tolist(), np.round(zi["axis"], 3).tolist()],
                       "portDiameter": zi["portDiameter"],
                       "portCrown": round(float(zi["crown"]), 2), "vaultRise": round(float(zi.get("vaultRise", 1.0)), 3),
                       "portDrop": round(float(zi.get("portDrop", 0.0)), 2)}
                       for z, zi in zip(zones, zinfo)},
                   "warnings": warnings})
    if hx.header_ribs:
        warnings.append("headerRibs is retired and ignored: the fins were placed and sized by hand, not by a load; "
                        "the plenum walls are the shell everywhere")
    warnings += _process_warnings(hx, report)
    # the design's full-grid fields are done with: free them before the fluid spaces
    del side_a, g, d, pts, fluid, shape
    extra = ({"masks": masks, "lo": lo, "step": step, "dims": tuple(dims)},) if grids else ()
    if not fluid_meshes:
        return (mesh, report, *extra)
    # Each space is the part's own void (grid > 0, inside the body), so its
    # walls come from the same smooth field as the part; the voxel label
    # only says which fluid, by a distance 1.5 voxels past its mask (inside
    # metal there, where the field governs).
    spaces = {}
    inside = -d_body.reshape(dims)
    for name, m in zip(hx.names, masks):
        if m.any():
            near = np.float32(1.5 * step) - ndimage.distance_transform_edt(~m).astype(np.float32) * np.float32(step)
            f = np.minimum(np.minimum(grid, inside), near).astype(np.float32)
            del near
            f[f == 0.0] = -1e-6
            v, fc, _, _ = marching_cubes(np.pad(f, 1, constant_values=-1.0), 0.0, spacing=(step, step, step),
                                         allow_degenerate=False)
            fm = trimesh.Trimesh(v + lo - 0.5 * step, fc[:, ::-1], process=True)
            if fm.volume < 0:
                fm.invert()
            spaces[name] = fm
    return (mesh, report, spaces, *extra)


def _shaping_up(hx):
    """The build direction the design shapes for (roofs, closure, bore crowns),
    or None: no build axis, or a process that builds anything unsupported
    (SLS, MJF: the powder carries every layer)."""
    if hx.build is None or not needs_shaping(hx.process):
        return None
    return _unit(hx.build)


def _vault(d, ell, cap, R, rise=1.0, rho=np.inf):
    """A plenum of depth 2R over its zone whose edges are round vaults: in the
    plane across an edge (x = d - R from the plenum's middle, y = how far past
    the vault's spring line toward the zone's edge), a half circle of radius R,
    or, rising R * rise over the spring line, tangent flanks to a round tip of
    radius rho (selfsupport.arch_sdf). The spring line sits R * rise in from
    the zone's edge less `cap`, so the vault's top never leaves the zone.
    Near-distance (mm), < 0 inside."""
    x = d - np.float32(R)
    H = np.float32(R) * rise
    y = np.float32(cap) + H - ell
    slab = np.abs(x) - np.float32(R)
    arch = arch_sdf(x, np.maximum(y, 0.0), R, H, rho)
    return np.where(y > 0.0, arch, slab).astype(np.float32, copy=False)


def _too_coarse(hx, volume, step):
    need = volume / (0.5 * hx.wall) ** 3
    return (f"the heat exchanger needs a {step:.2f} mm grid for its size within its point cap "
            f"({hx.max_points / 1e6:.1f}M points), too coarse for {hx.wall:g} mm walls (at most wall / 2 = "
            f"{0.5 * hx.wall:.2f} mm). Shrink the core, thicken the wall, or raise the cap (`loadpath heatx "
            f"--max-points {int(np.ceil(need / 1e6)) * 1_000_000}`: about "
            f"{(_BASE_BYTES + _BYTES_PER_POINT * need) / 1e9:.1f} GB at its peak)")


def _reach_estimate(hx, zones=None):
    """How far past the core the design grid reaches (mm), without the zone
    stage: a sideways zone's plenum is `header` deep, an up or down one a
    55 degree cone over (at most) the core's half-width."""
    zones = zones if zones is not None else zones_for(hx)
    clo, chi = (np.asarray(b, float) for b in hx.core.bounds())
    ext = chi - clo
    cap = hx.cap if hx.cap is not None else hx.wall + 1.0
    up = _shaping_up(hx)
    heights = [hx.header]
    for z in zones:
        if up is not None and abs(float(_unit(z.direction) @ up)) > 0.6:
            across = [ext[i] for i in range(3) if abs(up[i]) < 0.5] or [ext.min()]
            heights.append(max(0.5 * min(across) - cap, 0.0) * np.tan(np.radians(_LEAN)))
    return max(heights) + hx.shell + hx.port_length + 4.0


def grid_plan(hx) -> dict:
    """The design grid before designing (milliseconds): its step (mm), point
    count and estimated peak memory, and whether the point cap
    (hx.max_points) leaves a grid fine enough for the wall (at most
    wall / 2). The roofs' heights are estimated (a cone over the core's
    half-width), so the real grid is no larger."""
    clo, chi = (np.asarray(b, float) for b in hx.core.bounds())
    reach = _reach_estimate(hx)
    ext = chi - clo + 2.0 * reach
    vol = float(np.prod(ext))
    step = max(hx.step or hx.wall / 3.0, (vol / hx.max_points) ** (1.0 / 3.0))
    points = float(np.prod(np.maximum(np.ceil(ext / step), 2)))
    out = {"step": round(step, 4), "points": int(points), "maxPoints": int(hx.max_points),
           "bounds": [np.round(clo - reach, 2).tolist(), np.round(chi + reach, 2).tolist()],
           "peakMemoryGB": round((_BASE_BYTES + _BYTES_PER_POINT * points) / 1e9, 2),
           "fits": bool(step <= hx.wall / 2.0 + 1e-12)}
    if not out["fits"]:
        out["error"] = _too_coarse(hx, vol, step)
        out["maxPointsNeeded"] = int(np.ceil(vol / (0.5 * hx.wall) ** 3))
        out["memoryNeededGB"] = round((_BASE_BYTES + _BYTES_PER_POINT * out["maxPointsNeeded"]) / 1e9, 1)
    return out


def _process_warnings(hx, report) -> list[str]:
    """What the report shows past the process's limits (sizing.PROCESS_LIMITS):
    the outer wall away from the ports' mouths, and in powder-bed metal,
    internal ceilings and powder no opening reaches."""
    from .sizing import PROCESS_LIMITS

    lim = PROCESS_LIMITS.get(hx.process, PROCESS_LIMITS["fdm"])
    out = []
    w = report.get("minOuterWall")
    if w is not None and w < lim["minWall"]:
        out.append(f"the outer wall is {w:.2f} mm at {report.get('minOuterWallAt')} (away from the ports' mouths), "
                   f"under {hx.process}'s {lim['minWall']:g} mm minimum: raise shell (now {hx.shell:g} mm)")
    if hx.process == "lpbf":
        ic = report.get("internalCeilings") or {}
        if (ic.get("unbuildableArea") or 0.0) > _LPBF_FLAT_CEILING:
            at = (ic.get("unbuildableAt") or [{}])[0]
            out.append(f"{ic['unbuildableArea']:.0f} mm^2 of internal ceiling lies farther than half the "
                       f"{ic.get('buildableChord')} mm chord lpbf closes over an arch from any edge that holds it "
                       f"(widest {ic.get('widestSpan')} mm, near {at.get('from')} to {at.get('to')}), over "
                       f"{_LPBF_FLAT_CEILING:g} mm^2: lpbf cannot support anything inside. Keep closure on, try "
                       "another build axis, a smaller cell or phase")
        tv = report.get("trappedVolume") or 0.0
        if tv > _LPBF_TRAPPED:
            lt = report.get("largestTrapped") or {}
            out.append(f"{tv:.1f} mm^3 of open space no port reaches ({report.get('trappedPockets')} pockets, the "
                       f"largest {lt.get('volume', 0):.1f} mm^3 from {lt.get('from')} to {lt.get('to')}), over "
                       f"{_LPBF_TRAPPED:g} mm^3: powder stays trapped there")
        for n, p in (report.get("powder") or {}).items():
            if (p.get("unreached") or 0.0) > 0:
                out.append(f"{n}: {p['unreached']:.1f} mm^3 of its space no opening reaches: powder cannot get out")
    return out


# A surface within this many degrees of level is a ceiling over the channels
# that end under it; the sheet closes toward it (or flares open over a plenum
# below) by _CLOSE_SLOPE mm of half-thickness per mm, as in exchanger_core.
_CLOSE_FACE = 40.0
_CLOSE_SLOPE = 1.0


def _closure(hx, zones, ell, d, dims, step, up, cap):
    """How each fluid's network is reshaped for printing, per fluid:
    "close" (mm the fluid's side of the sheet is pushed back: under core
    surface that looks up and is not this fluid's zone, both networks close
    over 0.45 cell / slope before the sealing depth `cap`, so every channel
    ends in a peaked roof, not on a flat skin); "deep" (mm added to `cap`:
    under the other fluid's zone where it looks down, this fluid is sealed
    deeper, to make room); "flare" (mm the fluid may open toward its own zone
    where that looks down, a plenum under the core, so what hangs between the
    mouths leans instead of ending flat; applied only a wall clear of the
    other fluid). Closing and sealing only add metal, so they can never open
    a leak."""
    grad = np.gradient(d.reshape(dims), float(step))
    nu = sum(float(up[i]) * grad[i] for i in range(3) if abs(up[i]) > 1e-9).ravel().astype(np.float32, copy=False)
    del grad
    c = np.cos(np.radians(_CLOSE_FACE))
    w_up = np.clip((nu - c) / 0.1, 0.0, 1.0)
    w_dn = np.clip((-nu - c) / 0.1, 0.0, 1.0)
    del nu
    reach = 0.45 * hx.cell / _CLOSE_SLOPE
    close = _CLOSE_SLOPE * np.clip(reach + cap + d, 0.0, None) * w_up
    flare_ramp = _CLOSE_SLOPE * np.clip(reach + d, 0.0, None) * w_dn
    out = {"close": [], "deep": [None, None], "flare": []}
    for f in (0, 1):
        own = np.max([ell[k] for k, z in enumerate(zones) if z.fluid == f], axis=0)
        inzone = np.clip(own / (2.0 * step) + 0.5, 0.0, 1.0)
        out["close"].append((close * (1.0 - inzone)).astype(np.float32, copy=False))
        out["flare"].append((flare_ramp * inzone).astype(np.float32, copy=False))
        out["deep"][1 - f] = ((reach + hx.wall) * w_dn * inzone).astype(np.float32, copy=False)
    return out


def _internal_ceilings(hx, mesh, masks, lo, step, up, clo, cap, sheet_c=0.0):
    """Faces over either fluid's space leaning flatter than 45 degrees from
    level (and those within 20 degrees: flat), leaving out the sheet itself,
    which prints unsupported. Inside, no support can be
    removed: in powder-bed metal these must be near zero."""
    c = mesh.triangles_center
    n = mesh.vertex_normals[mesh.faces].mean(axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    ijk = np.floor((c + n * step - lo) / step).astype(int)
    dims = np.asarray(masks[0].shape)
    ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
    wet = np.zeros(len(c), bool)
    i = ijk[ok]
    wet[ok] = masks[0][i[:, 0], i[:, 1], i[:, 2]] | masks[1][i[:, 0], i[:, 1], i[:, 2]]
    # the sheet as the TPMS shapes it (not where a closure has moved it)
    inside = hx.core.sdf(c) < -step
    sheet = np.abs(np.abs(tpms(hx.kind, c - clo - np.asarray(hx.phase, float), hx.cell) - sheet_c) - 0.5 * hx.wall) < 1.5 * step
    wet_all = wet.copy()
    wet &= ~(inside & sheet)
    down = -(n @ up)
    a = mesh.area_faces
    over = wet & (down > np.cos(np.radians(45.0)))
    flat = wet & (down > np.cos(np.radians(20.0)))
    out = {"overhangArea": float(a[over].sum()), "flatArea": float(a[flat].sum())}
    if flat.any():
        out["worstAt"] = _densest(c[flat], a[flat], hx.cell)
    # an arch closes its crown: a round top D_max across is flatter than any angle t over a
    # chord D_max sin t, so the ceiling is checked at a few angles up to the critical one,
    # each against its own chord (a flat cap fails at the shallow ones, a round crown passes)
    crit = _critical_angle(hx)
    # (the sheet counts here: it only holds a ceiling up where it is itself steep enough)
    spans = [_ceiling_spans(hx, mesh, wet_all & (down > np.cos(np.radians(t))), t)
             for t in sorted({min(5.0, crit), min(15.0, crit), min(30.0, crit), crit}) if t > 0]
    if spans:
        out.update(max(spans, key=lambda r: (r["unbuildableArea"], r["ceilingAngle"])))
    return out


def _critical_angle(hx):
    from .selfsupport import SELF_SUPPORT

    return float(SELF_SUPPORT[hx.process]["criticalAngle"]) if needs_shaping(hx.process) else 0.0


def _ceiling_spans(hx, mesh, over, angle):
    """How far each point of ceiling flatter than `angle` from level lies
    from the edge of its patch (where the surface turns steeper, or the sheet
    holds it up), against half the chord the process closes over a round arch
    at that angle (selfsupport: round limit x sin(angle); 5.7 mm at 45
    degrees in powder-bed metal, 2.1 mm at 15), or the flat span it bridges
    outright if that is more (printcheck.BRIDGE_SPAN, 2 mm). A round bore's or vault's crown is such a patch and
    builds while it is that narrow; a layer farther than half that chord from
    any edge has nothing to grow from. "unbuildableArea" (mm^2: faces past
    half the chord, half a mesh edge of slack), "widestSpan" (mm, twice the
    farthest), "unbuildableAt" (the worst patches)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from .selfsupport import round_limit

    from .printcheck import BRIDGE_SPAN

    # never less than the flat span the process bridges outright (printcheck: 2 mm in metal)
    chord = max(round_limit(hx.process) * np.sin(np.radians(angle)), BRIDGE_SPAN.get(hx.process, 0.0))
    out = {"widestSpan": 0.0, "unbuildableArea": 0.0, "buildableChord": round(float(chord), 2),
           "ceilingAngle": float(angle)}
    if not over.any():
        return out
    faces = mesh.faces[over]
    verts = np.unique(faces)
    vid = np.full(len(mesh.vertices), -1)
    vid[verts] = np.arange(len(verts))
    f = vid[faces]
    on_other = np.zeros(len(mesh.vertices), bool)
    on_other[np.unique(mesh.faces[~over])] = True
    edge = on_other[verts]
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(verts), len(verts)))
    _, comp = connected_components(g, directed=False)
    xyz = mesh.vertices[verts]
    dist = np.zeros(len(verts))
    order = np.argsort(comp, kind="stable")
    cuts = np.flatnonzero(np.diff(comp[order])) + 1
    for idx in np.split(order, cuts):
        b = idx[edge[idx]]
        if len(b) == 0:
            dist[idx] = np.inf
            continue
        dist[idx] = cKDTree(xyz[b]).query(xyz[idx])[0]
    fd = dist[f].mean(axis=1)
    # half a mesh edge of slack: marching cubes moves a surface about that much
    tol = 0.5 * float(np.mean(mesh.edges_unique_length)) if len(mesh.edges_unique) else 0.0
    wide = fd > 0.5 * chord + tol
    area = mesh.area_faces[over]
    out["widestSpan"] = round(float(min(2.0 * dist.max(), 1e9)), 2)
    out["unbuildableArea"] = float(area[wide].sum())
    if wide.any():
        c = mesh.triangles_center[over]
        fc = comp[f[:, 0]]
        worst = []
        for k in np.unique(fc[wide]):
            s_ = wide & (fc == k)
            worst.append({"area": round(float(area[s_].sum()), 1), "span": round(float(2.0 * fd[fc == k].max()), 2),
                          "from": np.round(c[s_].min(axis=0), 1).tolist(),
                          "to": np.round(c[s_].max(axis=0), 1).tolist()})
        out["unbuildableAt"] = sorted(worst, key=lambda w: -w["area"])[:6]
    return out


def _densest(points, weights, size):
    """The centre of the size-mm cube holding the most weight (where the
    worst of a scattered set of faces is)."""
    key = np.floor(points / size).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    best = int(np.argmax(np.bincount(inv.ravel(), weights=weights)))
    sel = inv.ravel() == best
    return np.round((points[sel] * weights[sel, None]).sum(0) / weights[sel].sum(), 1).tolist()


# a clean-out's plug seat: the tap drill a clean-out diameter takes (mm -> thread)
_TAPS = ((2.5, "M3x0.5"), (3.3, "M4x0.7"), (4.2, "M5x0.8"), (5.0, "M6x1"), (6.8, "M8x1.25"), (8.5, "M10x1.5"))


def _central_gradient(field, ijk, step):
    """The gradient of a grid field at one voxel (central differences, one-sided
    at the grid's edge): np.gradient at that point, without the whole field's."""
    g = np.zeros(3)
    for a in range(3):
        lo_, hi_ = list(ijk), list(ijk)
        lo_[a] = max(ijk[a] - 1, 0)
        hi_[a] = min(ijk[a] + 1, field.shape[a] - 1)
        if hi_[a] > lo_[a]:
            g[a] = (float(field[tuple(hi_)]) - float(field[tuple(lo_)])) / ((hi_[a] - lo_[a]) * step)
    return g


def _geodesic(mask, starts, step):
    """Distance (mm) from `starts` (voxel indices) through `mask`, inf where
    unreached."""
    from skimage.graph import MCP_Geometric

    cost = np.where(mask, 1.0, np.inf)
    starts = [tuple(s) for s in starts if mask[tuple(s)]]
    if not starts:
        return np.full(mask.shape, np.inf)
    cum, _ = MCP_Geometric(cost, fully_connected=True).find_costs(starts)
    return cum * step


def _cleanouts(hx, zones, zinfo, grid, masks, d_body, d_core, lo, step, dims):
    """Powder clean-out ports: for each fluid, `hx.cleanouts` times, a bore
    from the point of its space farthest (through the space) from every
    opening, straight out through the seal and the outer wall, on a boss long
    enough to tap for a plug (1.5 diameters of thread). The other fluid's
    channels within a wall of the bore are filled (only metal is added, so
    no leak can open; the bore never crosses a plenum). Also reports, per
    fluid, the longest path powder travels to an opening (`powder`).
    Returns (grid, cleanouts, powder)."""
    r = 0.5 * hx.cleanout_diameter
    solid = grid < 0
    outside = ~solid & (d_body.reshape(dims) >= 0)
    dist_out = ndimage.distance_transform_edt(~outside).astype(np.float32) * np.float32(step)
    core = d_core.reshape(dims) < 0
    body = d_body.reshape(dims) < 0
    del outside
    out, powder = [], {}
    pts_all = None
    boss_len = 3.0 * r
    boss_wall = max(hx.shell, 0.6 * r)
    reach_in = hx.shell + (hx.cap if hx.cap is not None else hx.wall + 1.0) + hx.cell
    for f, m in enumerate(masks):
        if not m.any():
            continue
        starts = []
        for z, zi in zip(zones, zinfo):
            if z.fluid != f:
                continue
            p0, p1 = zi["port"]
            ax = (p1 - p0) / np.linalg.norm(p1 - p0)
            q = p1 - ax * (0.5 * hx.port_length)
            starts.append(np.clip(np.floor((q - lo) / step).astype(int), 0, np.asarray(dims) - 1))
        other = masks[1 - f]
        ok = m & core & (dist_out <= reach_in)
        for _ in range(hx.cleanouts):
            geo = _geodesic(m, starts, step)
            placed = False
            cand = np.where(ok & np.isfinite(geo), geo, -1.0)
            del geo
            while not placed:
                i = int(np.argmax(cand))
                if cand.flat[i] <= 0:
                    break
                ijk = np.asarray(np.unravel_index(i, dims))
                ok[tuple(ijk)] = False
                path = float(cand.flat[i])
                cand.flat[i] = -1.0
                n = -_central_gradient(dist_out, ijk, step)
                if np.linalg.norm(n) < 1e-6:
                    continue
                n /= np.linalg.norm(n)
                site = lo + (ijk + 0.5) * step
                depth = float(dist_out[tuple(ijk)])
                # out of the core the bore and its boss meet only metal and then
                # air: no plenum, no port, and the straight line does reach out
                along = np.arange(0.0, depth + 3.0 * r + 1.0, 0.5 * step)
                idx = np.floor((site + np.outer(along, n) - lo) / step).astype(int)
                if (idx < 0).any() or (idx >= np.asarray(dims)).any():
                    continue
                ix, iy, iz = idx.T
                if ((m | other)[ix, iy, iz] & ~core[ix, iy, iz]).any():
                    continue
                if (body[ix, iy, iz] & (along > depth + 2.0 * step)).any():
                    continue
                placed = True
            if not placed:
                break
            b = site + n * (depth + boss_len)
            if pts_all is None:
                pts_all = _grid_points(lo, dims, step, np.float32)
            box_lo = np.minimum(site, b) - r - boss_wall - hx.wall - 2 * step
            box_hi = np.maximum(site, b) + r + boss_wall + hx.wall + 2 * step
            sel = np.all((pts_all >= box_lo) & (pts_all <= box_hi), axis=1)
            q = pts_all[sel]
            g_ = grid.reshape(-1)
            # the other fluid kept a wall (and two grid steps) clear of the bore
            collar = Cylinder(site - n * (r + hx.wall), b, r + hx.wall + 2 * step).sdf(q)
            theirs = other.reshape(-1)[sel]
            g_sel = g_[sel]
            g_sel[theirs] = np.minimum(g_sel[theirs], collar[theirs])
            boss = Cylinder(site + n * max(depth - hx.shell, step), b, r + boss_wall).sdf(q)
            bore = Cylinder(site - n * step, b + n * 1.0, r).sdf(q)
            g_[sel] = np.maximum(np.minimum(g_sel, boss), -bore)
            grid = g_.reshape(dims)
            starts.append(ijk)
            tap = min(_TAPS, key=lambda t: abs(t[0] - hx.cleanout_diameter))[1]
            plug_a = site + n * max(depth - hx.shell, step)
            out.append({"fluid": hx.names[f], "at": np.round(site, 2).tolist(), "axis": np.round(n, 3).tolist(),
                        "diameter": hx.cleanout_diameter, "pathBefore": round(path, 1),
                        "plug": {"a": np.round(plug_a, 3).tolist(), "b": np.round(b + n * 1.0, 3).tolist(),
                                 "r": r, "ends": "flat"},
                        "seal": f"tap {tap} through the boss; after de-powdering, a set screw with thread "
                                "sealant, or weld it shut"})
            near = np.zeros(dims, bool)
            near[tuple(ijk)] = True
            ok &= ndimage.distance_transform_edt(~near) * step > 4 * r      # not beside this one
        geo = _geodesic(m, starts, step)
        reach = geo[m & np.isfinite(geo)]
        powder[hx.names[f]] = {"farthestFromOpening": round(float(reach.max()), 1) if reach.size else None,
                               "unreached": float((m & ~np.isfinite(geo)).sum()) * step ** 3}
    return grid, out, powder


def _skirt(grid, envelope, axis, step):
    """Fill under the part's outside at 45 degrees, down to the bed: every
    layer of the envelope is carried down one layer at a time, eroded by a
    voxel, so whatever overhangs the outside (a rim, a horizontal cylinder's
    underside) gets a chamfer or keel under it. Only outside the envelope:
    plenums and channels keep their own shapes."""
    m = np.moveaxis(envelope < 0, axis, -1)
    k = m.copy()
    cross = ndimage.generate_binary_structure(2, 1)
    for i in range(k.shape[-1] - 2, -1, -1):
        k[..., i] |= ndimage.binary_erosion(k[..., i + 1], cross)
    filled = np.nonzero(m.any(axis=(0, 1)))[0]
    k[..., :filled[0]] = False                     # nothing below the bed
    sk = np.moveaxis(k & ~m, -1, axis)
    if not sk.any():
        return grid
    # its signed distance, on the box around it
    span = [np.nonzero(sk.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
    box = tuple(slice(max(n.min() - 3, 0), n.max() + 4) for n in span)
    sub = sk[box]
    d = (ndimage.distance_transform_edt(~sub) - ndimage.distance_transform_edt(sub)) * step
    d += np.where(sub, 0.5 * step, -0.5 * step)
    out = grid.copy()
    out[box] = np.minimum(out[box], d)
    return out


def _loft_planes(zi, cap, n_patch=700, n_rim=64):
    """A lofted header (scipy ConvexHull; face planes n.x + d <= 0 inside): the convex hull of
    its zone's surface, `cap` inside the zone's edge, and the port's root circle."""
    from scipy.spatial import ConvexHull

    pp, depth = zi["patch"]
    pp = pp[depth >= cap]
    if len(pp) < 8:
        raise ValueError("a lofted header's zone is too small for its cap; use flat headers or a smaller cap")
    if len(pp) > n_patch:
        pp = pp[np.random.default_rng(0).choice(len(pp), n_patch, replace=False)]
    a = zi["axis"]
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False)
    r = 0.5 * zi["portDiameter"]
    rim = zi["exit"] + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    return ConvexHull(np.vstack([pp, rim]))


def _planes_sdf(hull, pts, pad, soft=0.0, chunk=32768):
    """max over a convex hull's face planes: the exact distance inside, a close lower
    bound outside; evaluated only within `pad` of the hull's box (`pad` beyond it)."""
    out = np.full(len(pts), np.float32(pad), np.float32)
    eq, hv = hull.equations, hull.points[hull.vertices]
    lo, hi = hv.min(axis=0) - pad, hv.max(axis=0) + pad
    sel = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
    n, c = eq[:, :3].astype(np.float32).T, eq[:, 3].astype(np.float32)
    for i in range(0, len(sel), chunk):
        j = sel[i:i + chunk]
        v = pts[j] @ n + c
        m = v.max(axis=1)
        if soft > 0:
            # a soft maximum: the hull's facet edges round over at about `soft` mm
            m = m + np.float32(soft) * np.log(np.exp((v - m[:, None]) / np.float32(soft)).sum(axis=1))
        out[j] = np.minimum(m, pad)
    return out


def _port_box(hx, zi, pts, up, bed):
    """The grid points near a port, its gusset included."""
    p0, p1 = zi["port"]
    ax = (p1 - p0) / np.linalg.norm(p1 - p0)
    ends = np.array([p0, p1 + ax * 3.0])
    pad = max(0.5 * zi["portDiameter"], zi.get("crown", 0.0)) + hx.shell + 1.0
    blo, bhi = ends.min(axis=0) - pad, ends.max(axis=0) + pad
    if up is not None and abs(float(ax @ up)) <= 0.7:
        k = int(np.argmax(np.abs(up)))
        blo[k] = min(blo[k], bed - 1.0)
    return np.all((pts >= blo) & (pts <= bhi), axis=1)


def _crowned(u, l, r, h, rho):
    """Across a bore: its circle of radius r, rising h over its axis on tangent
    flanks to a round tip of radius rho (selfsupport.arch_sdf). Near-distance."""
    if h <= r + 1e-9:
        return np.sqrt(u * u + l * l) - np.float32(r)
    return arch_sdf(l, u, r, h, rho)


def _port(hx, zi, pts, up, bed):
    """A port's bore (void) and tube (solid) fields. The bore is round. One
    that runs level, wider than the process builds round (selfsupport), keeps
    its round bottom and sides and rises zi["crown"] over its axis on tangent
    flanks to a round tip the process closes: smooth all round, never a point. With a skirt (FDM:
    an outside overhang cannot grow off removable supports), a gusset under
    the tube rises at 45 degrees from the plenum wall."""
    r = 0.5 * zi["portDiameter"]
    ro = r + hx.shell
    p0, p1 = zi["port"]
    length = float(np.linalg.norm(p1 - p0))
    ax = (p1 - p0) / length
    bore = Cylinder(p0, p1 + ax * 2.0, r).sdf(pts)
    tube = Cylinder(p0, p1, ro).sdf(pts)
    if up is None or abs(float(ax @ up)) > 0.7:
        return bore, tube
    v = up - (up @ ax) * ax                        # up, across the port
    v /= np.linalg.norm(v)
    w = np.cross(ax, v)
    q = pts - p0
    a, u, l = q @ ax, q @ v, np.abs(q @ w)
    del q
    hc = float(zi.get("crown", r))
    if hc > r + 1e-9:
        rho = crown_radius(hx.process)
        bore = np.maximum(_crowned(u, l, r, hc, rho), np.maximum(-a, a - length - 2.0))
        # the tube follows the crowned bore, a shell over it; it starts a shell back toward the core,
        # so where the crown rises past the plenum's vault the bore's root is closed, not open
        back = hx.shell + 1.0
        tube = np.minimum(tube, np.maximum(_crowned(u, l, ro, hc + hx.shell, rho + hx.shell), np.maximum(-a - back, a - length)))
    if hx.skirt:
        # gusset: under the tube, its underside rising 45 degrees from the wall to the tube's end
        under = (a - length - u - ro) / _SQRT2         # < 0 above the 45 degree plane
        gus = np.maximum(np.maximum(l - ro, u), np.maximum(np.maximum(-a, a - length), under))
        if bed is not None:
            gus = np.maximum(gus, bed - (pts @ up))
        tube = np.minimum(tube, gus)
    return bore, tube


def _check(hx, zones, zinfo, grid, side_a, d_body, d_core, dims, step, lo, cap, mouths=()):
    """Isolation, walls and capacity, measured on the grid the part was
    meshed from. `mouths` (a, b, r): clean-out bores, left out of the outer
    wall like the ports' mouths."""
    solid = grid < 0
    inside_body = (d_body < 0).reshape(dims)
    open_ = ~solid & inside_body
    # 26-connected: fluids that touch only at a voxel's edge or corner count as touching
    lbl, _ = ndimage.label(open_, np.ones((3, 3, 3), bool))
    names = hx.names
    ports = {}
    for k, (z, zi) in enumerate(zip(zones, zinfo)):
        # a point in the bore just inside the tube's mouth
        p0, p1 = zi["port"]
        ax = (p1 - p0) / np.linalg.norm(p1 - p0)
        q = p1 - ax * (0.5 * hx.port_length)
        ijk = np.clip(np.floor((q - lo) / step).astype(int), 0, np.asarray(dims) - 1)
        ports[k] = (z.fluid, int(lbl[tuple(ijk)]))
    report = {"ports": {f"{names[z.fluid]}-{z.end}": ports[k][1] for k, z in enumerate(zones)}}
    a_lbls = {v for f, v in ports.values() if f == 0}
    b_lbls = {v for f, v in ports.values() if f == 1}
    report["isolated"] = bool(0 not in a_lbls | b_lbls and not (a_lbls & b_lbls))
    report["connected"] = {names[0]: len(a_lbls) == 1 and 0 not in a_lbls,
                           names[1]: len(b_lbls) == 1 and 0 not in b_lbls}
    a = np.isin(lbl, list(a_lbls - {0}))
    b = np.isin(lbl, list(b_lbls - {0}))
    cell_vol = step ** 3
    # open space inside the part that no port reaches: dead pockets (they trap
    # powder or resin, and in FDM they are just weight)
    trapped = open_ & ~a & ~b
    report["trappedVolume"] = float(trapped.sum()) * cell_vol
    if trapped.any():
        tl, n_t = ndimage.label(trapped)
        sizes = np.bincount(tl.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        report["trappedPockets"] = int(n_t)
        where = np.argwhere(tl == big)
        report["largestTrapped"] = {"volume": float(sizes[big - 1]) * cell_vol,
                                    "from": np.round(lo + where.min(axis=0) * step, 1).tolist(),
                                    "to": np.round(lo + (where.max(axis=0) + 1) * step, 1).tolist()}
        del tl
    if a.any() and b.any():
        # thinnest wall between the fluids: distance from B's space to A's
        dist_to_a = ndimage.distance_transform_edt(~a).astype(np.float32)
        dist_to_a[~b] = np.inf
        i = int(np.argmin(dist_to_a))
        report["minWallBetweenFluids"] = float(dist_to_a.flat[i]) * step
        report["minWallBetweenFluidsAt"] = np.round(lo + (np.asarray(np.unravel_index(i, dims)) + 0.5) * step, 1).tolist()
        del dist_to_a
    # thinnest outer wall: from either fluid to the outside, away from the port mouths
    outside = ~solid & ~inside_body               # air (the skirt is solid, though outside the body)
    dist_out = ndimage.distance_transform_edt(~outside).astype(np.float32) * np.float32(step)
    near_mouth = np.zeros(int(np.prod(dims)), bool)
    pts = _grid_points(lo, dims, step, np.float32)
    for zi in zinfo:
        p0, p1 = zi["port"]
        ax = (p1 - p0) / np.linalg.norm(p1 - p0)
        # a level port's crowned bore rises zi["crown"] over its axis (_port): its mouth
        # reaches that far, so the check leaves out that far too
        r_m = max(0.5 * zi["portDiameter"], zi.get("crown", 0.0))
        sel = _port_box(hx, zi, pts, None, None)
        near_mouth[sel] |= Cylinder(zi["exit"], p1 + ax * 3.0, r_m + hx.shell + 2 * step).sdf(pts[sel]) < 0
    for a_, b_, r_ in mouths:
        a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
        ax_ = (b_ - a_) / np.linalg.norm(b_ - a_)
        a0, rr = a_ - ax_ * (hx.shell + 2 * step), r_ + hx.shell + 2 * step
        box = np.all((pts >= np.minimum(a0, b_) - rr) & (pts <= np.maximum(a0, b_) + rr), axis=1)
        near_mouth[box] |= Cylinder(a0, b_, rr).sdf(pts[box]) < 0
    del pts
    near_mouth = near_mouth.reshape(dims)
    wet = (a | b) & ~near_mouth
    report["minOuterWall"] = float(dist_out[wet].min()) if wet.any() else None
    if wet.any():
        i = np.argmin(np.where(wet, dist_out, np.inf))
        report["minOuterWallAt"] = np.round(lo + (np.asarray(np.unravel_index(i, dims)) + 0.5) * step, 1).tolist()
    del dist_out
    core = (d_core < 0).reshape(dims)
    side_a = (side_a > 0).reshape(dims)
    for f, m in ((0, a), (1, b)):
        vol = float(m.sum()) * cell_vol
        core_vol = float((m & core).sum()) * cell_vol
        # how much of this fluid's side of the sheet in the core its ports reach
        # (a flare can carry one fluid across the sheet's mid-surface: what it
        # takes there counts as its own side, and not the other's)
        own = (core & ~solid & (side_a if f == 0 else ~side_a) & ~(b if f == 0 else a)) | (m & core)
        reached = core_vol / max(float(own.sum()) * cell_vol, 1e-9)
        # wetted area in the core: faces between this fluid and solid
        area = 0.0
        mf = m & core
        for ax in range(3):
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[ax], s1[ax] = slice(0, -1), slice(1, None)
            area += float(((mf[tuple(s0)] & solid[tuple(s1)] & core[tuple(s1)]) |
                           (mf[tuple(s1)] & solid[tuple(s0)] & core[tuple(s0)])).sum()) * step * step
        # a staircase over-reads a smooth surface's area by about 3/2
        area /= 1.5
        report[names[f]] = {"volume": vol, "coreVolume": core_vol, "coreWettedArea": area,
                            "coreReached": round(reached, 4),
                            "hydraulicDiameter": 4.0 * core_vol / area if area > 0 else None}
    core_total = float(core.sum()) * cell_vol
    report["coreVolume"] = core_total
    report["areaDensity"] = report[names[0]]["coreWettedArea"] / core_total * 1000.0   # m^2/m^3 = mm^2/mm^3 * 1000
    return report, (a, b)


def _overhangs(hx, mesh, up):
    """Downward faces steeper than 45 degrees outside the core (plenums,
    roofs, ports), off the bed: what FDM would have to bridge or support.
    The core's own faces are left out: a sheet TPMS prints unsupported."""
    if up is None:
        return {}
    c = mesh.triangles_center
    # vertex-averaged normals: marching cubes' facets scatter a few degrees either way
    nz = mesh.vertex_normals[mesh.faces].mean(axis=1) @ up
    z = c @ up
    bed = z.min()
    out = (hx.core.sdf(c) > 0.3) & (z > bed + 0.5)
    hang = out & (nz < -np.cos(np.radians(45.0)) - 0.02)
    return {"overhangArea": float(mesh.area_faces[hang].sum()),
            "bedArea": float(mesh.area_faces[(z <= bed + 0.5) & (nz < -0.9)].sum())}
