"""
Two-fluid heat exchangers from a sheet TPMS.

A sheet gyroid (or diamond, Schwarz P) splits space into two interleaved
channel networks that never meet: one on each side of the sheet. Give each
network its own fluid and the sheet is the wall heat crosses, with a very
large area per volume and no straight path for either flow to short-circuit.
The work is in the headers: each fluid has to reach its own network and only
that one.

The core is a box. Fluid A runs along one axis and fluid B along another
(cross flow). On A's two faces, B's channels are sealed off over a cap
depth, so only A's channels open into the plenum chamber there; on B's faces
the same the other way round. Each plenum is walled in and fed by a port
tube on its outer face. The plenums are inset from the core's edges by the
cap depth, so an A plenum never meets a B plenum or a B channel.

Built implicitly on one grid (distances in mm, negative = solid), meshed by
marching cubes, then checked on that same grid: the fluid spaces are labeled,
each port must reach its partner and no port of the other fluid, and the
thinnest wall between the two fluids is measured.

    hx = HeatExchangerSpec(core=Box((0, 0, 0), (40, 40, 40)), cell=10, wall=0.8)
    mesh, report = design_heat_exchanger(hx)

Units are millimetres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .lattice import tpms
from .shapes import Box, Cylinder

__all__ = ["HeatExchangerSpec", "design_heat_exchanger", "heat_exchanger_from_spec"]


@dataclass
class HeatExchangerSpec:
    core: Box
    cell: float = 10.0              # mm per TPMS period
    wall: float = 0.8               # mm, the sheet between the fluids
    kind: str = "gyroid"
    shell: float = 1.6              # mm, outer walls
    header: float = 6.0             # mm, plenum depth beyond the core
    cap: float | None = None        # mm, sealing depth at the other fluid's faces (default: wall + 1 mm)
    port_diameter: float = 8.0
    port_length: float = 8.0
    axes: tuple = (0, 1)            # fluid A flows along axes[0], fluid B along axes[1]
    step: float | None = None       # grid, mm (default wall / 3)
    max_points: int = 16_000_000
    names: tuple = ("A", "B")
    extra: dict = field(default_factory=dict)


def heat_exchanger_from_spec(spec: dict) -> HeatExchangerSpec:
    core = spec.get("core")
    if not isinstance(core, dict) or core.get("type", "box") != "box":
        raise ValueError("heatExchanger.core must be a box: {\"type\": \"box\", \"min\": [..], \"max\": [..]}")
    hx = HeatExchangerSpec(core=Box(core["min"], core["max"]))
    for key, attr in (("cell", "cell"), ("wall", "wall"), ("shell", "shell"), ("header", "header"),
                      ("cap", "cap"), ("portDiameter", "port_diameter"), ("portLength", "port_length")):
        if key in spec:
            setattr(hx, attr, float(spec[key]))
    if "type" in spec:
        hx.kind = str(spec["type"])
    if "axes" in spec:
        hx.axes = tuple("xyz".index(a) for a in spec["axes"])
    if "fluids" in spec:
        hx.names = tuple(spec["fluids"])
    if len(set(hx.axes)) != 2:
        raise ValueError("heatExchanger.axes names two different axes, e.g. \"xy\"")
    if hx.cell < 3.0 * hx.wall:
        raise ValueError("heatExchanger.cell must be at least 3 x wall, or the channels close")
    if hx.port_diameter > min(hx.core.hi - hx.core.lo) - 2 * hx.shell:
        raise ValueError("heatExchanger.portDiameter does not fit on the core's faces")
    return hx


def _box_sdf(lo, hi, pts):
    return Box(lo, hi).sdf(pts)


def _layout(hx: HeatExchangerSpec):
    """Plenum boxes and port cylinders: per fluid, (plenums, ports)."""
    lo, hi = hx.core.lo, hx.core.hi
    cap = hx.cap if hx.cap is not None else hx.wall + 1.0
    out = []
    for f, a in enumerate(hx.axes):
        plenums, ports = [], []
        for sign, face in ((-1, lo[a]), (1, hi[a])):
            plo, phi = lo.copy() + cap, hi.copy() - cap
            if sign < 0:
                plo[a], phi[a] = face - hx.header, face
            else:
                plo[a], phi[a] = face, face + hx.header
            plenums.append((plo, phi))
            c = 0.5 * (plo + phi)
            p0 = c.copy()
            p0[a] = face + sign * hx.header
            p1 = p0.copy()
            p1[a] += sign * (hx.shell + hx.port_length)
            ports.append((p0, p1))
        out.append((plenums, ports))
    return out, cap


def design_heat_exchanger(hx: HeatExchangerSpec, fluid_meshes: bool = False):
    """Returns (trimesh.Trimesh, report), and with `fluid_meshes` a third
    item: each fluid's space as a mesh, by fluid name (for renders and
    flow simulation)."""
    import trimesh
    from skimage.measure import marching_cubes

    layout, cap = _layout(hx)
    r_port = 0.5 * hx.port_diameter
    reach = hx.header + hx.shell + hx.port_length + 4.0     # past the ports' mouths, so the mesh closes
    lo = hx.core.lo - reach
    hi = hx.core.hi + reach
    step = hx.step or hx.wall / 3.0
    step = max(step, (float(np.prod(hi - lo)) / hx.max_points) ** (1.0 / 3.0))
    if step > hx.wall / 2.0:
        raise ValueError(f"the heat exchanger needs a {step:.2f} mm grid for its size, too coarse for "
                         f"{hx.wall:g} mm walls; shrink the core or raise max_points")
    dims = np.maximum(np.ceil((hi - lo) / step).astype(int), 2)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)
    pts = lo + (idx + 0.5) * step

    # the TPMS's two sides: > 0 is fluid A's network, < 0 fluid B's
    g = tpms(hx.kind, pts - hx.core.lo, hx.cell)
    d_core = hx.core.sdf(pts)
    half = 0.5 * hx.wall
    fluid = []
    body = [d_core - hx.shell]
    for f, (plenums, ports) in enumerate(layout):
        a_other = hx.axes[1 - f]
        side = g - half if f == 0 else -g - half          # > 0 inside this fluid's network
        # sealed within `cap` of the other fluid's faces
        dist_other = np.minimum(pts[:, a_other] - hx.core.lo[a_other], hx.core.hi[a_other] - pts[:, a_other])
        net = np.maximum(-side, np.maximum(d_core, cap - dist_other))   # < 0: open channel
        void = net
        for plo, phi in plenums:
            void = np.minimum(void, _box_sdf(plo, phi, pts))
            body.append(_box_sdf(plo - hx.shell, phi + hx.shell, pts))
        for p0, p1 in ports:
            # the bore runs from inside the plenum out past the tube's end
            ax = (p1 - p0) / np.linalg.norm(p1 - p0)
            bore = Cylinder(p0 - ax * 1.0, p1 + ax * 2.0, r_port)
            void = np.minimum(void, bore.sdf(pts))
            body.append(Cylinder(p0 - ax * hx.shell, p1, r_port + hx.shell).sdf(pts))
        fluid.append(void)
    d_body = np.min(body, axis=0)
    d = np.maximum(d_body, -np.minimum(fluid[0], fluid[1]))
    grid = d.reshape(dims)

    verts, faces, _, _ = marching_cubes(grid, 0.0, spacing=(step, step, step))
    mesh = trimesh.Trimesh(verts + lo + 0.5 * step, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()

    report, masks = _check(hx, layout, grid, fluid, d_body, dims, step, lo, pts, r_port)
    report.update({"kind": hx.kind, "cell": hx.cell, "wall": hx.wall, "step": step,
                   "solidVolume": float(mesh.volume), "watertight": bool(mesh.is_watertight),
                   "bounds": [mesh.bounds[0].round(2).tolist(), mesh.bounds[1].round(2).tolist()]})
    if not fluid_meshes:
        return mesh, report
    spaces = {}
    for name, m in zip(hx.names, masks):
        if m.any():
            v, fc, _, _ = marching_cubes(np.pad(m, 1).astype(float), 0.5, spacing=(step, step, step))
            fm = trimesh.Trimesh(v + lo - 0.5 * step, fc[:, ::-1], process=True)
            if fm.volume < 0:
                fm.invert()
            spaces[name] = fm
    return mesh, report, spaces


def _check(hx, layout, grid, fluid, d_body, dims, step, lo, pts, r_port):
    """Isolation and capacity, measured on the grid the part was meshed from."""
    solid = grid < 0
    inside_body = (d_body < 0).reshape(dims)
    open_ = ~solid & inside_body
    lbl, _ = ndimage.label(open_)
    names = hx.names
    ports = {}
    for f, (_, port_list) in enumerate(layout):
        for k, (p0, p1) in enumerate(port_list):
            # a point in the bore just inside the tube's mouth
            ax = (p1 - p0) / np.linalg.norm(p1 - p0)
            q = p1 - ax * (0.5 * hx.port_length)
            ijk = np.clip(np.floor((q - lo) / step).astype(int), 0, np.asarray(dims) - 1)
            ports[(f, k)] = int(lbl[tuple(ijk)])
    report = {"ports": {f"{names[f]}{'-in' if k == 0 else '-out'}": v for (f, k), v in ports.items()}}
    a_lbls = {ports[(0, 0)], ports[(0, 1)]}
    b_lbls = {ports[(1, 0)], ports[(1, 1)]}
    report["isolated"] = bool(0 not in a_lbls | b_lbls and not (a_lbls & b_lbls))
    report["connected"] = {names[0]: len(a_lbls) == 1 and 0 not in a_lbls,
                           names[1]: len(b_lbls) == 1 and 0 not in b_lbls}
    a = np.isin(lbl, list(a_lbls - {0}))
    b = np.isin(lbl, list(b_lbls - {0}))
    if a.any() and b.any():
        # thinnest wall between the fluids: distance from B's space to A's
        dist_to_a = ndimage.distance_transform_edt(~a) * step
        report["minWallBetweenFluids"] = float(dist_to_a[b].min())
    cell_vol = step ** 3
    core = (hx.core.sdf(pts) < 0).reshape(dims)
    for f, m in ((0, a), (1, b)):
        vol = float(m.sum()) * cell_vol
        core_vol = float((m & core).sum()) * cell_vol
        # wetted area in the core: faces between this fluid and solid
        area = 0.0
        for ax in range(3):
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[ax], s1[ax] = slice(0, -1), slice(1, None)
            mf, ms, cf = m & core, solid, core
            area += float(((mf[tuple(s0)] & ms[tuple(s1)] & cf[tuple(s1)]) |
                           (mf[tuple(s1)] & ms[tuple(s0)] & cf[tuple(s0)])).sum()) * step * step
        # a staircase over-reads a smooth surface's area by about 3/2
        area /= 1.5
        report[names[f]] = {"volume": vol, "coreVolume": core_vol, "coreWettedArea": area,
                            "hydraulicDiameter": 4.0 * core_vol / area if area > 0 else None}
    core_total = float(np.prod(hx.core.hi - hx.core.lo))
    report["areaDensity"] = report[names[0]]["coreWettedArea"] / core_total * 1000.0   # m^2/m^3 = mm^2/mm^3 * 1000
    return report, (a, b)
