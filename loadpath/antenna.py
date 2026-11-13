"""An antenna where it will live: on its board, inside the product.

    loadpath antenna brief.json --out out

A board antenna is tuned on a bare reference board in free space. Inside a
product it sits a few millimetres from a plastic wall, a battery, a motor,
a cable, and every one of them moves its resonance and takes some of its
power. This module builds the electromagnetic model of that situation from
the same parts loadpath already places (the board's copper, the shell's
mesh and material, the components and wires) and solves Maxwell's equations
for it (rf.py). Nothing is tuned by rule: the plastic's permittivity and
loss, the copper pattern and the metal near it are the model, and the
resonance, the match and the efficiency are what the fields do.

What it reports, for the bare board and for the board as installed:

* the reflection S11 across a sweep, the resonance, the worst S11 and the
  mismatch loss inside the band, the -10 dB and -6 dB bandwidths, and the
  input impedance (what a matching network has to work with);
* radiation efficiency (radiated / accepted) and total efficiency
  (radiated / available), with the heat split by material, so a lossy wall
  shows up as the watts it takes;
* the shift between the two, in MHz and as a share of the bandwidth;
* a sensitivity field around the antenna: how much a small piece of
  dielectric at each point would change the input impedance, from the
  unperturbed fields by reciprocity (first-order exact). It is the
  antenna's keep-out as physics gives it, and what placement ranks
  positions by.

docs/ANTENNA.md has the method, what it was checked against and its limits.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import rf
from .rf import C0, EPS0, njit

# ------------------------------------------------------------- materials --

# Radio-frequency properties not on the print cards: the board laminate.
# Each: (relative permittivity, loss tangent, source).
RF_ONLY = {
    "fr4": (4.04, 0.021,
            "Isola 370HR datasheet, Dk 4.04 / Df 0.021 at 2 GHz (a standard high-Tg FR-4; "
            "FR-4 grades run 3.7 to 4.6 with resin content: give the laminate's own values "
            "when the board house states them)"),
    "air": (1.0, 0.0, "exact"),
}


def rf_properties(material):
    """(eps_r, tan_d, note) of a material: a name on the print library's
    cards or in RF_ONLY, or a dict {"permittivity": .., "lossTangent": ..}
    (with an optional "base" name). A material with no radio-frequency
    value raises: it is asked for, never assumed."""
    if isinstance(material, dict):
        base = material.get("base") or material.get("name")
        if "permittivity" in material:
            return (float(material["permittivity"]), float(material.get("lossTangent", 0.0)),
                    material.get("source", "given in the brief"))
        material = base
    name = str(material).lower()
    if name in RF_ONLY:
        return RF_ONLY[name]
    from . import material_library as ml
    eps = ml.value(name, "permittivity")
    if eps is None:
        raise ValueError(
            f"no radio-frequency permittivity on file for '{material}': give "
            '{"base": "%s", "permittivity": ..., "lossTangent": ...} (measured, or from the '
            "material's datasheet)" % material)
    c = ml.card(name)["values"]
    src = {}
    for key in ("permittivity", "lossTangent"):
        v = c.get(key)
        if v:
            s = ml.SOURCES.get(v[1])
            src.setdefault((s.title if s else v[1]) + (f" ({v[2]})" if v[2] else ""), []).append(key)
    return (float(eps), float(ml.value(name, "lossTangent", 0.0)),
            "; ".join(f"{' and '.join(k)}: {t}" for t, k in src.items()))


def mixed(eps_r, tan_d, fill):
    """Effective permittivity of a material that fills only the fraction
    `fill` of its volume (infill, a lattice), the rest air. Returns
    (eps_r, tan_d, eps_low, eps_high).

    The Hashin-Shtrikman bounds are the tightest limits on the effective
    permittivity of any isotropic two-phase mix at a given fraction
    (Hashin & Shtrikman, J. Appl. Phys. 33, 1962): the upper one is solid
    with isolated air pockets, the lower one isolated solid in air. A
    printed lattice has both phases connected and lies between; for
    plastics (permittivity 2.5 to 3.5 against 1) the two are within a few
    percent of each other, so the value used is their mean and the bounds
    are returned as its uncertainty. Loss follows from the same formula
    with complex permittivity."""
    f = float(fill)
    if f >= 1.0:
        return eps_r, tan_d, eps_r, eps_r
    e1 = eps_r * (1 - 1j * tan_d)

    def mg(em, ei, fi):       # inclusions ei at fraction fi in matrix em
        return em + 3 * fi * em * (ei - em) / (ei + 2 * em - fi * (ei - em))
    hi = mg(e1, 1.0, 1 - f)
    lo = mg(1.0, e1, f)
    e = 0.5 * (hi + lo)
    return float(e.real), float(-e.imag / e.real), float(lo.real), float(hi.real)


# ------------------------------------------------------------ voxelizing --

@njit(cache=True)
def _hits(tri, xs, ys, col, zz):
    """Where vertical lines through the sample points (xs[i], ys[j]) cross
    the triangles. Fills col (i * ny + j) and zz while they have room;
    returns the number of crossings."""
    n, ny = 0, ys.shape[0]
    for t in range(tri.shape[0]):
        ax, ay, az = tri[t, 0, 0], tri[t, 0, 1], tri[t, 0, 2]
        bx, by, bz = tri[t, 1, 0], tri[t, 1, 1], tri[t, 1, 2]
        cx, cy, cz = tri[t, 2, 0], tri[t, 2, 1], tri[t, 2, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-300:
            continue
        i0 = np.searchsorted(xs, min(ax, bx, cx))
        i1 = np.searchsorted(xs, max(ax, bx, cx))
        j0 = np.searchsorted(ys, min(ay, by, cy))
        j1 = np.searchsorted(ys, max(ay, by, cy))
        for i in range(i0, i1):
            px = xs[i]
            for j in range(j0, j1):
                py = ys[j]
                l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
                l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
                l3 = 1.0 - l1 - l2
                if l1 >= 0 and l2 >= 0 and l3 >= 0:
                    if n < col.shape[0]:
                        col[n] = i * ny + j
                        zz[n] = l1 * az + l2 * bz + l3 * cz
                    n += 1
    return n


@njit(cache=True)
def _fill(col, zz, z, out, ny):
    """Inside lengths per cell: consecutive crossing pairs of a column are
    an entry and an exit; the overlap of each pair with each cell's z range,
    as a fraction of the cell's height, is added to out[i, j, k]."""
    nz = z.shape[0] - 1
    n, a = col.shape[0], 0
    while a < n:
        b = a
        while b < n and col[b] == col[a]:
            b += 1
        i, j = col[a] // ny, col[a] % ny
        q = a
        while q + 1 < b:
            z0, z1 = zz[q], zz[q + 1]
            k = max(np.searchsorted(z, z0) - 1, 0)
            while k < nz and z[k] < z1:
                lo = max(z0, z[k])
                hi = min(z1, z[k + 1])
                if hi > lo:
                    out[i, j, k] += (hi - lo) / (z[k + 1] - z[k])
                k += 1
            q += 2
        a = b


def mesh_fraction(vertices, faces, x, y, z, sub=3):
    """The fraction of each grid cell inside a closed mesh, on a grid whose
    lines are x, y, z: exact along z (entry and exit heights of vertical
    lines), sampled sub x sub lines per cell across."""
    tri = np.asarray(vertices, float)[np.asarray(faces)]

    def samples(a, shift):
        d = np.diff(a) / sub
        return (a[:-1, None] + (np.arange(sub)[None, :] + 0.5 + shift) * d[:, None]).ravel()
    # off the cell's symmetry lines, so a sample never lands on a mesh edge
    # that runs along a grid line or a cell diagonal
    xs, ys = samples(np.asarray(x, float), 0.0137), samples(np.asarray(y, float), -0.0291)
    n = _hits(tri, xs, ys, np.zeros(0, np.int64), np.zeros(0))
    col, zz = np.zeros(n, np.int64), np.zeros(n)
    _hits(tri, xs, ys, col, zz)
    order = np.lexsort((zz, col))
    out = np.zeros((len(xs), len(ys), len(z) - 1), np.float32)
    _fill(col[order], zz[order], np.asarray(z, float), out, len(ys))
    nx, ny = len(x) - 1, len(y) - 1
    return out.reshape(nx, sub, ny, sub, -1).mean(axis=(1, 3))


def _in_rings(px, py, rings):
    """Even-odd point-in-polygon for arrays of points; rings is a list of
    closed loops [[x, y], ...]."""
    inside = np.zeros(px.shape, bool)
    for ring in rings:
        r = np.asarray(ring, float)
        x0, y0 = r[:, 0], r[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        for a, b, c, d in zip(x0, y0, x1, y1):
            if b == d:
                continue
            cross = ((b > py) != (d > py)) & (px < (c - a) * (py - b) / (d - b) + a)
            inside ^= cross
    return inside


def _box_mesh(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


# ----------------------------------------------------------------- scene --

class Scene:
    """Everything the fields see, in the board's frame (mm): copper sheets
    in planes of constant z, dielectric and metal bodies, wires, one port.
    """

    def __init__(self, band, sweep=None):
        self.band = (float(band[0]) * 1e6, float(band[1]) * 1e6)      # Hz
        fc = 0.5 * (self.band[0] + self.band[1])
        self.sweep = ((float(sweep[0]) * 1e6, float(sweep[1]) * 1e6) if sweep
                      else (0.75 * fc, 1.25 * fc))
        self.sheets, self.solids, self.wires, self.port = [], [], [], None
        self.lines = [[], [], []]          # extra grid lines asked for (coordinate, spacing)

    # copper in a plane
    def add_sheet(self, z, rings, name="copper"):
        """Copper with no thickness in the plane z: `rings` is one closed
        outline [[x, y], ...] or a list of them (even-odd: a ring inside a
        ring is a hole)."""
        rings = [rings] if np.ndim(rings[0][0]) == 0 else rings
        self.sheets.append({"z": float(z), "rings": [np.asarray(r, float) for r in rings], "name": name})

    def add_solid(self, name, eps_r=None, tan_d=0.0, metal=False, box=None, mesh=None, prism=None,
                  fine=None, rank=1, curved=False):
        """A body: `box` ((x0, y0, z0), (x1, y1, z1)), `prism` (rings, z0,
        z1: outlines in x, y extruded between two heights) or `mesh`
        (vertices, faces). Dielectric with eps_r and tan_d, or metal. Bodies
        added later overwrite earlier ones where they overlap. `fine` asks
        for a grid spacing inside it. A box's or prism's faces become grid
        lines, so it is represented exactly; a mesh is sampled. Where two
        bodies' faces are closer than half the copper's cell, the one of
        higher `rank` keeps its line (the board's own parts are 2, the
        copper 3) and the other's face moves to it. `curved` is for a
        prism whose outline is a curve of many points: its vertices are
        not made grid lines (only its two heights are), and a cell belongs
        to it when its centre is inside, so resolve it with `fine`."""
        if sum(v is not None for v in (box, mesh, prism)) != 1:
            raise ValueError("a solid is a box, a prism or a mesh")
        if box is not None:
            (x0, y0, z0), (x1, y1, z1) = box
            prism = ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], z0, z1)
        if prism is not None:
            rings = prism[0]
            rings = [rings] if np.ndim(rings[0][0]) == 0 else rings
            prism = ([np.asarray(r, float) for r in rings], float(min(prism[1:])), float(max(prism[1:])))
        else:
            mesh = (np.asarray(mesh[0], float), np.asarray(mesh[1]))
        self.solids.append({"name": name, "eps_r": eps_r, "tan_d": tan_d, "metal": bool(metal),
                            "prism": prism, "mesh": mesh, "fine": fine, "rank": int(rank),
                            "curved": bool(curved)})

    @staticmethod
    def _extent(s):
        if s["prism"] is not None:
            r = np.vstack(s["prism"][0])
            return np.r_[r.min(0), s["prism"][1]], np.r_[r.max(0), s["prism"][2]]
        return s["mesh"][0].min(0), s["mesh"][0].max(0)

    def add_wire(self, points, name="wire"):
        """A thin conductor along a polyline. On the grid it follows cell
        edges, and a line of edges acts as a round wire of radius about
        0.23 of the cell across it (Makinen, Juntunen & Kivikoski, IEEE
        Trans. MTT 50, 2002), so the grid near a wire is set by its gauge
        where that matters."""
        self.wires.append({"points": np.asarray(points, float), "name": name})

    def set_port(self, p0, p1, width=0.0, across=None, resistance=50.0):
        """The feed: a source of `resistance` ohms between points p0 and p1
        (mm), which differ along one axis. `width` spreads it over a strip
        `across` another axis (0, 1 or 2), as under a feed trace."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = np.abs(p1 - p0)
        axis = int(np.argmax(d))
        if np.count_nonzero(d > 1e-9) != 1:
            raise ValueError("a port runs along one axis: its two ends differ in one coordinate")
        self.port = {"p0": p0, "p1": p1, "axis": axis, "width": float(width),
                     "across": across, "resistance": float(resistance)}

    # -- the grid -----------------------------------------------------------
    def bounds(self):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for s in self.sheets:
            for r in s["rings"]:
                lo[:2] = np.minimum(lo[:2], r.min(0))
                hi[:2] = np.maximum(hi[:2], r.max(0))
            lo[2], hi[2] = min(lo[2], s["z"]), max(hi[2], s["z"])
        for s in self.solids:
            b = self._extent(s)
            lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
        for w in self.wires:
            lo, hi = np.minimum(lo, w["points"].min(0)), np.maximum(hi, w["points"].max(0))
        return lo, hi

    def grid(self, fine=0.5, body=1.0, coarse=None, pad=None, npml=8, ratio=1.4):
        """Grid lines (mm). `fine` is the spacing at copper edges and the
        port, `body` inside dielectric and metal bodies, `coarse` in open
        air (default a twentieth of the shortest wavelength in the sweep),
        `pad` the air between the model and the absorbing layer (default a
        quarter of the longest wavelength in the band)."""
        lam_min = C0 / self.sweep[1] * 1e3
        coarse = coarse or lam_min / 20.0
        pad = pad if pad is not None else C0 / self.band[0] * 1e3 / 4.0
        lo, hi = self.bounds()
        fixed = [list(self.lines[a]) for a in range(3)]
        for s in self.sheets:
            for r in s["rings"]:
                for a in (0, 1):
                    fixed[a] += [(v, fine, 3) for v in np.unique(np.round(r[:, a], 6))]
            fixed[2].append((s["z"], fine, 3))
        for s in self.solids:
            b = self._extent(s)
            d = s["fine"] or body
            eps = 1.0 if s["metal"] else float(s["eps_r"])
            d = min(d, lam_min / 20.0 / math.sqrt(eps))
            for a in range(3):
                # lines through the body at its own spacing: there for resolution only
                n = max(1, int(math.ceil((b[1][a] - b[0][a]) / d - 1e-9)))
                fixed[a] += [(v, d, 0) for v in np.linspace(b[0][a], b[1][a], n + 1)]
            if s["prism"] is not None:
                # a prism's own faces are lines; the board's before any other body's
                for r in ([] if s.get("curved") else s["prism"][0]):
                    for a in (0, 1):
                        fixed[a] += [(v, d, s["rank"]) for v in np.unique(np.round(r[:, a], 6))]
                fixed[2] += [(s["prism"][1], d, s["rank"]), (s["prism"][2], d, s["rank"])]
        for w in self.wires:
            for a in range(3):
                fixed[a] += [(v, body, 1) for v in w["points"][:, a]]
        if self.port:
            p = self.port
            for a in range(3):
                fixed[a] += [(p["p0"][a], fine, 3), (p["p1"][a], fine, 3)]
            if p["width"] and p["across"] is not None:
                c = p["p0"][p["across"]]
                fixed[p["across"]] += [(c - p["width"] / 2, fine, 3), (c + p["width"] / 2, fine, 3)]
        out = []
        for a in range(3):
            g = rf.graded_lines(fixed[a], coarse, lo[a] - pad, hi[a] + pad, ratio=ratio,
                                dmin=fine / 2.0)
            g = np.r_[g[0] - (g[1] - g[0]) * np.arange(npml, 0, -1), g,
                      g[-1] + (g[-1] - g[-2]) * np.arange(1, npml + 1)]
            out.append(g)
        return out

    # -- the model ----------------------------------------------------------
    def build(self, fine=0.5, body=1.0, coarse=None, pad=None, npml=8, sub=3, ratio=1.4):
        x, y, z = self.grid(fine, body, coarse, pad, npml, ratio)
        nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
        eps = np.ones((nx, ny, nz), np.float32)
        inv = np.ones((nx, ny, nz), np.float32)          # mean of 1/eps_r in each cell
        lossy = np.zeros((nx, ny, nz), np.float32)       # eps_r * tan_d
        label = np.zeros((nx, ny, nz), np.int16)         # which body owns each cell
        metal = np.zeros((nx, ny, nz), bool)
        xc, yc, zc = (0.5 * (a[1:] + a[:-1]) for a in (x, y, z))
        self.names = ["air"]
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        for s in self.solids:
            if s["prism"] is not None:
                rings, z0, z1 = s["prism"]
                # a prism's faces are grid lines: a cell is inside or out
                f = (_in_rings(X, Y, rings)[:, :, None] & ((zc > z0) & (zc < z1))[None, None, :])
                f = f.astype(np.float32)
            else:
                f = mesh_fraction(s["mesh"][0], s["mesh"][1], x, y, z, sub=sub)
            if s["name"] not in self.names:
                self.names.append(s["name"])
            idx = self.names.index(s["name"])
            if s["metal"]:
                metal |= f > 0.5
                label[f > 0.5] = idx
                continue
            metal &= ~(f > 0.5)
            eps = eps * (1 - f) + f * np.float32(s["eps_r"])
            inv = inv * (1 - f) + f / np.float32(s["eps_r"])
            lossy = lossy * (1 - f) + f * np.float32(s["eps_r"] * s["tan_d"])
            label[f > 0.5] = idx
        fc = 0.5 * (self.band[0] + self.band[1])
        sim = rf.Sim(x * 1e-3, y * 1e-3, z * 1e-3, eps_r=eps, tan_d=lossy / eps, f0=fc, pml=npml,
                     inv_eps_r=inv)
        sim.pec_cells(metal)
        for s in self.sheets:
            k = int(np.argmin(np.abs(z - s["z"])))
            sim.pec_sheet(2, k, _in_rings(X, Y, s["rings"]))
        for w in self.wires:
            self._wire(sim, w["points"], (x, y, z))
        if self.port is None:
            raise ValueError("the scene has no port: set_port, or a feed in the brief")
        p = self.port
        ax = p["axis"]
        g = (x, y, z)
        near = lambda a, v: int(np.argmin(np.abs(g[a] - v)))
        i0, i1 = sorted((near(ax, p["p0"][ax]), near(ax, p["p1"][ax])))
        if i1 == i0:
            raise ValueError("the port's two ends fall on the same grid line")
        base = [near(a, p["p0"][a]) for a in range(3)]
        lines = []
        if p["width"] and p["across"] is not None:
            a = p["across"]
            c = p["p0"][a]
            js = [j for j in range(len(g[a])) if c - p["width"] / 2 - 1e-6 <= g[a][j] <= c + p["width"] / 2 + 1e-6]
        else:
            a, js = None, [None]
        for j in js:
            ln = []
            for i in range(i0, i1):
                e = list(base)
                e[ax] = i
                if a is not None:
                    e[a] = j
                ln.append(tuple(e))
            lines.append(ln)
        port = sim.add_port(ax, lines, p["resistance"])
        # the sign that makes "p1 above p0" positive
        port.sign = 1.0 if p["p1"][ax] > p["p0"][ax] else -1.0
        self.sim, self.label, self.grid_mm, self.metal = sim, label, (x, y, z), metal
        return sim

    @staticmethod
    def _wire(sim, pts, g):
        pec = (sim.pec_x, sim.pec_y, sim.pec_z)
        node = [np.array([int(np.argmin(np.abs(g[a] - p[a]))) for a in range(3)]) for p in pts]
        for n0, n1 in zip(node[:-1], node[1:]):
            cur = n0.copy()
            # walk to the next vertex, always stepping along the axis with
            # the most left to go: a staircase that stays near the segment
            while np.any(cur != n1):
                rem = n1 - cur
                frac = [abs(rem[a]) / max(1, abs(n1[a] - n0[a])) for a in range(3)]
                a = int(np.argmax(frac))
                step = 1 if rem[a] > 0 else -1
                e = cur.copy()
                if step < 0:
                    e[a] -= 1
                pec[a][tuple(e)] = True
                cur[a] += step


# -------------------------------------------------------------- analysis --

def _crossings(f, y, level):
    """Frequencies where y crosses `level` (linear interpolation)."""
    out = []
    for i in range(len(f) - 1):
        a, b = y[i] - level, y[i + 1] - level
        if a == 0 or (a < 0) != (b < 0):
            out.append(f[i] + (f[i + 1] - f[i]) * (a / (a - b) if a != b else 0.0))
    return out


def summarize(scene, out, n_sweep=None):
    """The numbers a designer reads, from a finished run."""
    f, s11 = out["freqs"], out["s11"]
    db = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
    lo, hi = scene.band
    inb = (f >= lo) & (f <= hi)
    fr, depth = rf.resonance(f, s11)
    res = {
        "band": [lo / 1e6, hi / 1e6],
        "resonance": {"frequency": round(fr / 1e6, 2), "s11": round(depth, 2),
                      "insideBand": bool(lo <= fr <= hi)},
        "inBand": {
            "worstS11": round(float(db[inb].max()), 2),
            "bestS11": round(float(db[inb].min()), 2),
            "worstMismatchLoss": round(float(-10 * np.log10(1 - np.abs(s11[inb]).max() ** 2)), 3),
        },
        "bandwidth": {},
        "impedance": {},
    }
    for name, level in (("-10dB", -10.0), ("-6dB", -6.0)):
        if depth < level:
            c = _crossings(f, db, level)
            below = [v for v in c if v <= fr]
            above = [v for v in c if v >= fr]
            if below and above:
                a, b = max(below), min(above)
                res["bandwidth"][name] = {"from": round(a / 1e6, 1), "to": round(b / 1e6, 1),
                                          "width": round((b - a) / 1e6, 1),
                                          "coversBand": bool(a <= lo and b >= hi)}
            else:
                res["bandwidth"][name] = {"note": "wider than the sweep on one side",
                                          "coversBand": bool((not below or max(below) <= lo)
                                                             and (not above or min(above) >= hi))}
        else:
            res["bandwidth"][name] = None
    for label_, fq in (("bandLow", lo), ("bandCenter", 0.5 * (lo + hi)), ("bandHigh", hi)):
        zr = np.interp(fq, f, out["z"].real)
        zi = np.interp(fq, f, out["z"].imag)
        res["impedance"][label_] = [round(float(zr), 1), round(float(zi), 1)]
    mon = out["monitor"]
    eff = {"at": [round(v / 1e6, 1) for v in mon],
           "radiation": [round(float(v), 4) for v in out["radiationEfficiency"]],
           "total": [round(float(v), 4) for v in out["totalEfficiency"]],
           "total_dB": [round(float(10 * np.log10(max(v, 1e-9))), 2) for v in out["totalEfficiency"]]}
    sim = scene.sim
    if "dissipated" in out:
        (i0, i1), (j0, j1), (k0, k1) = sim.heat_box
        lab = scene.label[i0:i1, j0:j1, k0:k1]
        heat = {}
        for idx, name in enumerate(scene.names):
            m = lab == idx
            if idx and m.any():
                share = sim.heat[:, m].sum(axis=1) / out["accepted"]
                if share.max() > 5e-5:
                    heat[name] = [round(float(v), 4) for v in share]
        eff["heat"] = heat
        eff["unaccounted"] = [round(float(1 - (r + d) / a), 4) for r, d, a in
                              zip(out["radiated"], out["dissipated"], out["accepted"])]
    else:
        eff["unaccounted"] = [round(float(1 - r / a), 4) for r, a in zip(out["radiated"], out["accepted"])]
    res["efficiency"] = eff
    res["run"] = {"cells": int(np.prod(sim.n)), "smallestCell": round(float(min(d.min() for d in sim.d)) * 1e3, 3),
                  "steps": int(out["steps"]), "rungDown": bool(out["rungDown"]),
                  "seconds": round(out.get("seconds", 0.0), 1)}
    if not out["rungDown"]:
        res["run"]["note"] = ("the port had not rung down when the run stopped (residual "
                              f"{out['residual']:.1e} of its peak): spectra carry a ripple")
    return res


# conductivity of metals the scene's conductors are made of (S/m, 20 C)
METAL_CONDUCTIVITY = {
    "copper": (5.80e7, "IACS: 100 % = 58.0 MS/m (annealed copper standard)"),
    "aluminium 5052": (2.03e7, "35 % IACS, 5052-H32 (ASM Handbook vol. 2)"),
    "aluminium 6061": (2.50e7, "43 % IACS, 6061-T6 (ASM Handbook vol. 2)"),
    "stainless 304": (1.39e6, "resistivity 72 uohm cm (ASM Handbook vol. 1)"),
}


def metal_box(scene, margin=2):
    """Cell index ranges round every metal cell of a built scene (pass to
    `Sim.run(field_box=...)` so `conductor_loss` has the fields there)."""
    nz = np.nonzero(scene.metal)
    return [(max(int(a.min()) - margin, 0), min(int(a.max()) + 1 + margin, n)) for a, n in zip(nz, scene.metal.shape)]


def conductor_loss(scene, conductivity, default=None):
    """Power lost in the conductors' own resistance, per monitor frequency
    (W, for the run's drive), and its share of the accepted power: the
    standard perturbation for a good conductor (its skin depth, microns, is
    resolved by no grid): the surface resistance Rs = sqrt(pi f mu0 / sigma)
    times the tangential magnetic field on each metal face,
    P = 1/2 Rs |H_t|^2 dA (Harrington, Time-Harmonic Electromagnetic Fields,
    sec. 2-9). The field is the one in the open cell beside each face (half a
    cell off the surface).

    `conductivity`: {body name: S/m or a METAL_CONDUCTIVITY name}; bodies
    not named take `default` (none: left out). Needs a run with
    `field_box` covering the metal (`metal_box`). Metal cells only: wires
    (staircased edges) and zero-thickness sheets carry no area here and are
    left out, which the result lists."""
    sim = scene.sim
    if getattr(sim, "H_cells", None) is None:
        raise ValueError("run with field_box=metal_box(scene) first: the loss needs the magnetic field at the metal")
    sig = {}
    for k, v in dict(conductivity).items():
        sig[k] = METAL_CONDUCTIVITY[v][0] if isinstance(v, str) else float(v)
    (i0, i1), (j0, j1), (k0, k1) = sim.field_box
    metal = scene.metal[i0:i1, j0:j1, k0:k1]
    lab = scene.label[i0:i1, j0:j1, k0:k1]
    H = sim.H_cells                                   # [3] x (nmon, ni, nj, nk), A/m
    d = [sim.d[0][i0:i1], sim.d[1][j0:j1], sim.d[2][k0:k1]]
    mon = np.asarray(sim.monitor, float)
    rs_of = {}
    for idx, name in enumerate(scene.names):
        s_ = sig.get(name, default)
        if s_:
            rs_of[idx] = np.sqrt(np.pi * mon * 4e-7 * np.pi / s_)
    loss = np.zeros(len(mon))
    by = {}
    for ax in range(3):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
        ma, mb = metal[tuple(sl_a)], metal[tuple(sl_b)]
        for face_metal, open_side, mlab_sl, osl in ((ma & ~mb, sl_b, sl_a, sl_b), (mb & ~ma, sl_a, sl_b, sl_a)):
            if not face_metal.any():
                continue
            ml = lab[tuple(mlab_sl)][face_metal]
            ht2 = sum(np.abs(H[q][(slice(None),) + tuple(osl)][:, face_metal]) ** 2 for q in (u, v))
            shp = [1, 1, 1]; shp[u] = -1
            shq = [1, 1, 1]; shq[v] = -1
            area = np.broadcast_to(d[u].reshape(shp) * d[v].reshape(shq), face_metal.shape)[face_metal]
            for idx, rs in rs_of.items():
                m = ml == idx
                if m.any():
                    p = 0.5 * rs * (ht2[:, m] * area[m]).sum(axis=1)
                    loss += p
                    by[scene.names[idx]] = by.get(scene.names[idx], 0.0) + p
    return {"monitor": mon, "loss": loss, "byBody": by,
            "notIncluded": "wires and zero-thickness sheets (no surface in the grid); bodies without a conductivity"}


def simulate(scene, fine=0.5, body=1.0, n_freq=401, log=None, **kw):
    """Build and run a scene. Returns (summary, raw output)."""
    build_kw = {k: kw.pop(k) for k in ("coarse", "pad", "npml", "sub", "ratio") if k in kw}
    t0 = time.time()
    sim = scene.build(fine=fine, body=body, **build_kw)
    f = np.linspace(scene.sweep[0], scene.sweep[1], n_freq)
    lo, hi = scene.band
    out = sim.run(f, monitor=[lo, 0.5 * (lo + hi), hi],
                  bandwidth=max(0.6 * sim.f0, 2.4 * max(abs(f - sim.f0).max(), 1.0)), log=log, **kw)
    out["seconds"] = time.time() - t0
    return summarize(scene, out), out


def compare(installed, bare):
    """What the product did to the antenna: `installed` and `bare` are
    summaries from `simulate`."""
    fi, fb = installed["resonance"]["frequency"], bare["resonance"]["frequency"]
    out = {"resonanceShift": round(fi - fb, 2),
           "resonanceShiftPercent": round(100 * (fi - fb) / fb, 2)}
    bw = (bare["bandwidth"].get("-10dB") or {}).get("width")
    if bw:
        out["shiftOverBandwidth"] = round(abs(fi - fb) / bw, 3)
    ei, eb = installed["efficiency"]["total"], bare["efficiency"]["total"]
    out["totalEfficiencyChange_dB"] = [round(float(10 * np.log10(max(a, 1e-9) / max(b, 1e-9))), 2)
                                       for a, b in zip(ei, eb)]
    out["worstS11"] = {"bare": bare["inBand"]["worstS11"], "installed": installed["inBand"]["worstS11"]}
    return out


# --------------------------------------------------------- published designs --

def _rect(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def ifa_ti_dn007():
    """Texas Instruments' 2.4 GHz inverted-F board antenna (design note
    DN007, SWRU120D, Table 1-1 and Figure 1-1), as copper outlines in mm.

    x runs along the arm from the antenna's closed end, y is the distance
    from the ground plane's edge. Every dimension the table gives is used
    as given (H1..H9, W1, W2, L1..L8). Three the table leaves out are read
    off the figure, which is drawn to scale (the tabled dimensions measure
    within 2% on it): the two short legs to ground (x 6.84..8.18 and
    9.02..10.18) and the right edge of the feed's gap in the ground
    (x 12.08).

    Returns {"copper": [rings], "feed": (x, y) of the feed trace's centre
    at the ground edge, "feedWidth", "groundGap": (x0, x1), "size"}."""
    H1, H2, H3, H4, H5, H6, H7, H8 = 5.70, 0.74, 1.29, 2.21, 0.66, 1.21, 0.80, 1.80
    W1, W2 = 1.21, 0.46
    L1, L2, L3, L4, L5, L6, L7, L8 = 25.58, 16.40, 2.18, 4.80, 1.00, 1.00, 3.20, 0.45
    top = H1 + W1                               # the arm's outer edge above the ground edge
    stub0 = L3 + L4 + L5                        # the stub from the arm down to the feed pad
    stub1 = L1 - L2
    pad0, pad1 = stub0 - L5, stub0 - L5 + L7    # the feed pad
    pad_top = top - W1 - H8
    pad_bot = pad_top - H6
    bar_top = pad_bot - H5                      # the return bar along the ground
    assert abs(bar_top - (H2 + H3)) < 0.02 and abs(pad1 - (stub1 + L6)) < 0.02
    f0, f1 = pad1 + L8, pad1 + L8 + W2          # the feed trace
    copper = [
        _rect(0, L1, top - W1, top),            # the arm
        _rect(0, stub1, top - H4, top),         # its thick closed end
        _rect(stub0, stub1, pad_top, top),      # stub to the feed pad
        _rect(0, L3, H2, top),                  # down the closed end
        _rect(0, pad1, H2, H2 + H3),            # the return bar
        _rect(6.84, 8.18, 0, H2),               # two legs to ground (from the figure)
        _rect(9.02, pad1, 0, H2),
        _rect(pad0, pad1, pad_bot, pad_top),    # feed pad
        # the feed: a 45 degree jog off the pad's corner, then straight to the ground edge
        [[pad1, pad_bot + 0.52], [f1, pad_bot - 0.39], [f1, 0], [f0, 0], [f0, pad_bot - 0.58],
         [pad1 - 0.13, pad_bot], [pad1, pad_bot]],
    ]
    return {"copper": copper, "feed": (0.5 * (f0 + f1), 0.0), "feedWidth": W2,
            "groundGap": (pad1, 12.08), "size": (L1, top), "edgeMargin": 0.61,
            "source": "Texas Instruments DN007 / SWRU120D, 2.4-GHz Inverted F Antenna, Table 1-1"}


def ifa(arm=22.0, height=6.0, tap=5.0, trace=1.0, gap=0.5):
    """A plain inverted-F in copper outlines (mm), x along the arm from the
    shorted end, y up from the ground edge: a leg to ground at the end, the
    arm parallel to the ground edge at `height`, and a feed leg `tap` along
    the arm that stops `gap` short of the ground, where the port sits. The
    arm's length sets the resonance and the tap the resistance there; `tune`
    solves both on the board they will be built on."""
    copper = [_rect(0, trace, 0, height), _rect(0, arm, height - trace, height),
              _rect(tap, tap + trace, gap, height - trace)]
    return {"copper": copper, "feed": (tap + trace / 2, 0.0), "feedWidth": trace, "feedGap": gap,
            "size": (arm, height), "edgeMargin": 0.6}


def board_brief(design, width, ground, thickness=1.0, laminate="fr4", side=1.0, band=(2400, 2483.5),
                sweep=None, lead=1.0):
    """A brief (`loadpath schema antenna`) for a design from `ifa` or
    `ifa_ti_dn007` across the end of a board `width` wide with `ground` mm
    of ground plane behind it (copper on both faces, stitched). The board's
    corner is the origin, its bottom face z = 0; the antenna is on the top
    face at the +y end, its shorted end `side` mm from the board's edge
    (None: centred)."""
    L, H = design["size"]
    x0 = (width - L) / 2.0 if side is None else float(side)
    if x0 < 0 or x0 + L > width + 1e-9:
        raise ValueError(f"the antenna is {L:.1f} mm long and the board {width:.1f} mm wide: it does not fit "
                         "across the end")
    y0 = float(ground)
    mv = lambda ring: [[round(x + x0, 4), round(y + y0, 4)] for x, y in ring]
    fx, w = design["feed"][0] + x0, design["feedWidth"]
    board = {"size": [width, round(y0 + H + design["edgeMargin"], 4)], "thickness": thickness,
             "material": laminate, "top": [mv(r) for r in design["copper"]]}
    if "groundGap" in design:
        # the feed is a trace over the bottom ground through a gap in the
        # top ground; the port is `lead` mm inside the ground edge
        g0, g1 = design["groundGap"][0] + x0, design["groundGap"][1] + x0
        board["ground"] = [_rect(0, g0, 0, y0), _rect(g1, width, 0, y0), _rect(g0, g1, 0, y0 - lead - 0.5)]
        board["bottom"] = [_rect(g0, g1, y0 - lead - 0.5, y0)]
        board["top"].append(_rect(fx - w / 2, fx + w / 2, y0 - lead, y0))
        feed = {"from": [fx, y0 - lead, 0.0], "to": [fx, y0 - lead, thickness], "width": w, "across": "x"}
    else:
        board["ground"] = [_rect(0, width, 0, y0)]
        feed = {"from": [fx, y0, thickness], "to": [fx, y0 + design["feedGap"], thickness],
                "width": w, "across": "x"}
    out = {"band": list(band), "board": board, "feed": feed}
    if sweep:
        out["sweep"] = list(sweep)
    return out


def frame_brief(size=(40.0, 100.0, 8.0), wall=1.5, clearance=8.0, split=24.0, tap=2.5, gap=1.5,
                plate=(3.0, 4.0), tie=True, fill="pa12", cover=None, cover_thickness=1.5,
                band=(2400, 2483.5), sweep=None):
    """A brief for a product whose metal frame is the antenna: a band of
    metal `wall` thick round a body `size` (x, y, z; corner at the origin),
    a metal plate inside it (the chassis: board ground, battery tray)
    joined to the band along its sides and stopping `clearance` short of
    the +y end, one split `gap` wide in the +y edge starting `split` from
    x = 0, filled with `fill`, and the radio between the plate's edge and
    the band `tap` beyond the split. The stretch of band from the split round to
    where the plate meets it is the radiator; `tie` joins the band to the
    plate just beyond the split so the rest of the edge is ground. Nothing
    here says how long that stretch should be: `tune` solves `split` and
    `tap` from the fields. `cover` is a material the frame is sandwiched
    in (`cover_thickness` over every outer face, and in the split), given
    as surroundings so the bare frame and the covered one can be compared.
    `tap` is measured from the split's far edge, so the feed cannot land on
    the split whatever the two are moved to."""
    W, L, H = (float(v) for v in size)
    tap = split + gap + max(float(tap), 0.0)
    if not (wall + 3.0 <= split and tap <= W - wall):
        raise ValueError(f"split at {split:.1f} mm and feed at {tap:.1f} mm do not fit the {W:.0f} mm edge")
    z0, z1 = plate
    zm = z1
    rad = [
        {"name": "frame", "metal": True, "prism": {
            "outline": [_rect(0, W, 0, L), _rect(wall, W - wall, wall, L - wall)], "from": 0.0, "to": H}},
        {"name": "chassis", "metal": True, "box": [[wall, wall, z0], [W - wall, L - clearance, z1]]}]
    if tie:
        rad.append({"name": "tie", "metal": True,
                    "box": [[split - 3.0, L - clearance, z0], [split - 1.0, L - wall, z1]]})
    rad += [{"name": "split", "material": cover or fill, "box": [[split, L - wall, 0.0], [split + gap, L, H]]},
            {"name": "feed", "wire": [[tap, L - wall, zm], [tap, L - clearance + 1.0, zm]]}]
    out = {"band": list(band), "radiator": rad,
           "feed": {"from": [tap, L - clearance, zm], "to": [tap, L - clearance + 1.0, zm]}}
    if cover:
        c = float(cover_thickness)
        out["surroundings"] = [
            {"name": "cover", "material": cover, "box": [[-c, -c, -c], [W + c, L + c, H + c]]},
            {"name": "inside", "material": {"permittivity": 1.0, "lossTangent": 0.0},
             "box": [[wall, wall, 0.0], [W - wall, L - wall, H]]}]
    if sweep:
        out["sweep"] = list(sweep)
    return out


def bench_monopole(height=28.0, plate=60.0, thickness=1.6, rod=1.3, cap=None):
    """The bench test of docs/ANTENNA.md as a brief: a round rod `rod` mm
    across standing `height` mm on the centre pin of a coax jack through a
    square metal plate, and optionally a printed cap set over it
    ({"inner", "tall", "wall", "material"}: an open-bottomed square cup
    standing on the plate, its inside `inner` wide and `tall` high).

    The rod is solved as a square one of the same capacitance per length
    (side = rod / 1.18: a square conductor acts as a round one of radius
    0.59 of its side). Both caps' outlines are always in the grid (as air
    when absent), so a cap's shift is read on one grid."""
    c, w, t = plate / 2.0, rod / 1.18, thickness
    def cup(inner, tall, wall, material, name):
        o = inner / 2.0 + wall
        return [{"name": name, "material": material, "box": [[c - o, c - o, t], [c + o, c + o, t + tall + wall]]},
                {"name": name + "-inside", "material": "air",
                 "box": [[c - inner / 2.0, c - inner / 2.0, t], [c + inner / 2.0, c + inner / 2.0, t + tall]]}]
    sur = []
    for name, (inner, tall, wall) in BENCH_CAPS.items():
        sur += cup(inner, tall, wall, "air", f"grid-{name}")
    if cap:
        sur += cup(cap["inner"], cap["tall"], cap["wall"], cap["material"], "cap")
    sur.append({"name": "rod", "metal": True, "resolution": 0.3,
                "box": [[c - w / 2, c - w / 2, t + 0.5], [c + w / 2, c + w / 2, t + height]]})
    return {"name": "bench-monopole", "band": [2400, 2483.5], "sweep": [1500, 3400],
            "board": {"size": [plate, plate], "thickness": t, "material": "fr4",
                      "ground": [[[0, 0], [plate, 0], [plate, plate], [0, plate]]]},
            "feed": {"from": [c, c, t], "to": [c, c, t + 0.5], "width": w, "across": "x"},
            "surroundings": sur}


def bench_blade(height=28.0, width=8.0, sheet=0.5, plate=60.0, thickness=1.6, pin=1.3, sleeve=None,
                wall=2.0, margin=3.0):
    """The second bench test of docs/ANTENNA.md: metal sandwiched in
    plastic. A flat strip of sheet metal `width` x `sheet` standing
    `height` mm on the centre pin of a coax jack through a square metal
    plate (the strip is the brief's `radiator`), and optionally a printed
    sleeve of `sleeve` (a material) slid over it: `wall` mm of plastic on
    each face of the strip, `margin` mm past its edges and its top. The
    sleeve's outline is always in the grid (as air when absent), so its
    shift is read on one grid."""
    c, t, w = plate / 2.0, thickness, pin / 1.18
    top = t + height
    block = [[c - width / 2 - margin, c - sheet / 2 - wall, t], [c + width / 2 + margin, c + sheet / 2 + wall,
                                                                top + margin]]
    return {"name": "bench-blade", "band": [2400, 2483.5], "sweep": [1500, 3400],
            "board": {"size": [plate, plate], "thickness": t, "material": "fr4",
                      "ground": [[[0, 0], [plate, 0], [plate, plate], [0, plate]]]},
            "radiator": [
                {"name": "pin", "metal": True, "resolution": 0.3,
                 "box": [[c - w / 2, c - w / 2, t + 0.5], [c + w / 2, c + w / 2, t + 2.0]]},
                {"name": "blade", "metal": True, "resolution": 0.5,
                 "box": [[c - width / 2, c - sheet / 2, t + 2.0], [c + width / 2, c + sheet / 2, top]]}],
            "feed": {"from": [c, c, t], "to": [c, c, t + 0.5], "width": w, "across": "x"},
            "surroundings": [{"name": "sleeve", "material": sleeve or "air", "box": block}]}


# the two printed caps of the bench test: inside width, inside height, wall (mm)
BENCH_CAPS = {"a": (20.0, 32.0, 2.0), "b": (6.0, 30.0, 4.0)}


def reactance_zero(out):
    """The first frequency (Hz) at which the port's reactance crosses zero
    going up: the resonance a network analyser's Smith chart shows."""
    f, x = out["freqs"], out["z"].imag
    k = np.where(np.diff(np.sign(x)) > 0)[0]
    if not len(k):
        return None
    i = int(k[0])
    return float(f[i] - x[i] * (f[i + 1] - f[i]) / (x[i + 1] - x[i]))


# ----------------------------------------------------------------- briefs --

def _material(spec):
    """(eps_r, tan_d, metal, note) of a surrounding body's material entry."""
    if spec is None or spec == "metal" or (isinstance(spec, dict) and spec.get("metal")):
        return None, 0.0, True, "metal (a perfect conductor)"
    fill = spec.get("fill") if isinstance(spec, dict) else None
    eps, tan, note = rf_properties(spec)
    if fill is not None and float(fill) < 1.0:
        eps, tan, lo, hi = mixed(eps, tan, float(fill))
        note += f"; {float(fill):.0%} solid: permittivity {eps:.2f} (bounds {lo:.2f} to {hi:.2f})"
    return eps, tan, False, note


def scene_from_brief(brief, base_dir=None, surroundings=True):
    """The Scene of an antenna brief. `surroundings=False` leaves out
    everything but the board; "air" keeps every body's place in the grid
    but makes it air: the bare board on the very grid the installed board
    is solved on, so the difference between the two is the surroundings
    and not the grid (a change of grid alone moves a 2.4 GHz resonance by
    a few tenths of a percent)."""
    ghost = surroundings == "air"
    b = brief.get("board")
    if b is None and not brief.get("radiator"):
        raise ValueError("an antenna brief needs a board, a radiator (metal of the product itself) or both")
    sc = Scene(brief["band"], brief.get("sweep"))
    sc.materials = {}
    # the surroundings first: the board and its copper sit in them
    for i, s in enumerate(brief.get("surroundings") or [] if surroundings else []):
        name = s.get("name") or f"body{i}"
        if "wire" in s:
            if not ghost:
                sc.add_wire(s["wire"], name)
            continue
        eps, tan, metal, note = _material(s.get("material", "metal") if not s.get("metal") else "metal")
        if ghost:
            eps, tan, metal = 1.0, 0.0, False
        else:
            sc.materials[name] = ({"metal": True} if metal else
                                  {"permittivity": round(eps, 3), "lossTangent": round(tan, 4), "source": note})
        kw = {"fine": s.get("resolution")}
        if "box" in s:
            sc.add_solid(name, eps, tan, metal, box=s["box"], **kw)
        elif "prism" in s:
            sc.add_solid(name, eps, tan, metal, prism=(s["prism"]["outline"], s["prism"]["from"], s["prism"]["to"]), **kw)
        else:
            m = s["mesh"]
            if isinstance(m, (str, Path)):
                import trimesh
                path = Path(m) if Path(m).is_absolute() or base_dir is None else Path(base_dir) / m
                tm = trimesh.load(path, force="mesh")
                m = (np.asarray(tm.vertices), np.asarray(tm.faces))
            v = np.asarray(m[0], float)
            if s.get("matrix") is not None:
                M = np.asarray(s["matrix"], float)
                v = v @ M[:3, :3].T + M[:3, 3]
            sc.add_solid(name, eps, tan, metal, mesh=(v, m[1]), **kw)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    if b is not None:
        t = float(b.get("thickness", 1.6))
        outline = b.get("outline") or _rect(0, b["size"][0], 0, b["size"][1])
        eps, tan, note = rf_properties(b.get("material", "fr4"))
        sc.materials["board"] = {"permittivity": eps, "lossTangent": tan, "source": note}
        sc.add_solid("board", eps, tan, prism=(outline, 0.0, t), rank=2)
        for ring in b.get("ground") or []:
            sc.add_solid("ground", metal=True, prism=(ring, 0.0, t), rank=2)
        for ring in b.get("top") or []:
            sc.add_sheet(t, ring, "top copper")
        for ring in b.get("bottom") or []:
            sc.add_sheet(0.0, ring, "bottom copper")
        for v in b.get("vias") or []:
            sc.add_wire([[v[0], v[1], 0.0], [v[0], v[1], t]], "via")
        ring = np.asarray(outline, float)
        lo, hi = np.r_[ring.min(0), 0.0], np.r_[ring.max(0), t]
    # the product's own metal that the radio drives (a frame, a plate, a
    # foil) and what is built into it (the plastic in a split): part of the
    # antenna, so there in the bare solve too
    for i, s in enumerate(brief.get("radiator") or []):
        name = s.get("name") or f"radiator{i}"
        if "wire" in s:
            sc.add_wire(s["wire"], name)
            pts = np.asarray(s["wire"], float)
            lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
            continue
        eps, tan, metal, note = _material(s.get("material", "metal") if not s.get("metal") else "metal")
        sc.materials[name] = ({"metal": True} if metal else
                              {"permittivity": round(eps, 3), "lossTangent": round(tan, 4), "source": note})
        if "box" in s:
            sc.add_solid(name, eps, tan, metal, box=s["box"], fine=s.get("resolution"), rank=2)
        elif "prism" in s:
            sc.add_solid(name, eps, tan, metal, prism=(s["prism"]["outline"], s["prism"]["from"],
                                                       s["prism"]["to"]), fine=s.get("resolution"), rank=2)
        else:
            raise ValueError(f"radiator body {name!r}: give a box, a prism or a wire (drawn shapes whose "
                             "faces become grid lines)")
        e = Scene._extent(sc.solids[-1])
        lo, hi = np.minimum(lo, e[0]), np.maximum(hi, e[1])
    sc.antenna_box = (lo, hi)
    f = brief["feed"]
    across = f.get("across")
    sc.set_port(f["from"], f["to"], width=float(f.get("width", 0.0)),
                across={"x": 0, "y": 1, "z": 2}[across] if across else None,
                resistance=float(f.get("impedance", 50.0)))
    return sc


def sensitivity(scene, out, at=1):
    """How much the port's impedance changes per cubic millimetre of
    material put at each point near the antenna, from the fields of the run
    as it stands (monitor frequency index `at`, the band's centre).

    Reciprocity gives the first-order change exactly (Harrington,
    Time-Harmonic Electromagnetic Fields, sec. 7-4): a small volume V whose
    permittivity rises by d_eps changes the impedance seen by a current I
    at the port by
        dZ = -(j w eps0 d_eps V) E.E / I^2,
    with E the unperturbed field there (the plain product, not the
    magnitude: phase matters). A small conducting body is the same with its
    electric and magnetic dipoles: for a compact one (a sphere)
        dZ = -(j w V) (3 eps0 E.E + 1.5 mu0 H.H) / I^2.
    Returns {"x", "y", "z" (cell centres, mm), "dielectric" (ohm per mm3
    per unit of d_eps), "metal" (ohm per mm3 of compact conductor), "z0",
    "impedance"}."""
    sim = scene.sim
    (i0, i1), (j0, j1), (k0, k1) = sim.field_box
    w = 2 * math.pi * out["monitor"][at]
    I2 = out["current"][at] ** 2
    ee = sum(e[at].astype(np.complex128) ** 2 for e in sim.E_cells)
    hh = sum(h[at].astype(np.complex128) ** 2 for h in sim.H_cells)
    x, y, z = scene.grid_mm
    c = lambda a, s0, s1: 0.5 * (a[s0:s1] + a[s0 + 1:s1 + 1])
    zin = out["z"][int(np.argmin(np.abs(out["freqs"] - out["monitor"][at])))]
    return {"x": c(x, i0, i1), "y": c(y, j0, j1), "z": c(z, k0, k1),
            "edges": (x[i0:i1 + 1], y[j0:j1 + 1], z[k0:k1 + 1]),
            "dielectric": -1j * w * EPS0 * ee / I2 * 1e-9,
            "metal": -1j * w * (3 * EPS0 * ee + 1.5 * rf.MU0 * hh) / I2 * 1e-9,
            "impedance": complex(zin), "z0": scene.port["resistance"],
            "frequency": float(out["monitor"][at])}


def predicted_change(sens, points, volumes, eps_r=None):
    """First-order change of the port's impedance and reflection from
    material at `points` ((n, 3) mm, in the board's frame) with `volumes`
    mm3 each: a dielectric of permittivity `eps_r`, or compact metal when
    eps_r is None. Returns (dZ, |change of reflection|).

    The field inside a dielectric body is lower than the field it was put
    in by its depolarization: none for a field along a thin wall, a factor
    3 / (eps_r + 2) for a compact body (a sphere, Clausius-Mossotti). The
    compact factor is used; the full solve of the layout is the check.
    Against full solves this is within 10 to 20% for PA12 walls and a
    whole shell, and about a third low for a small metal body
    (tests/test_antenna.py)."""
    pts = np.atleast_2d(points)
    ex, ey, ez = sens["edges"]
    idx = [np.searchsorted(e, pts[:, a]) - 1 for a, e in enumerate((ex, ey, ez))]
    ok = np.ones(len(pts), bool)
    for a, e in enumerate((ex, ey, ez)):
        ok &= (idx[a] >= 0) & (idx[a] < len(e) - 1)
    field = sens["metal"] if eps_r is None else sens["dielectric"] * (3.0 * (eps_r - 1.0) / (eps_r + 2.0))
    dz = np.sum(field[idx[0][ok], idx[1][ok], idx[2][ok]] * np.broadcast_to(volumes, (len(pts),))[ok])
    z, z0 = sens["impedance"], sens["z0"]
    return complex(dz), float(abs(2 * z0 * dz / (z + z0) ** 2))


def _field_box(scene, sim, reach, lo=None, hi=None):
    """Cell ranges covering the board (or the whole model) and `reach` mm
    round it."""
    if lo is None:
        lo, hi = scene.bounds()
    box = []
    for a, g in enumerate(scene.grid_mm):
        m = 8 + 4          # inside the absorbing layer and the flux box
        i0 = max(int(np.searchsorted(g, lo[a] - reach)) - 1, m)
        i1 = min(int(np.searchsorted(g, hi[a] + reach)), len(g) - 1 - m)
        box.append((i0, i1))
    return box


def simulate_brief(brief, base_dir=None, installed=True, fields=False, log=None):
    """Run a brief's bare board (installed=False), the bare board on the
    installed grid (installed="air") or the board with its surroundings.
    Returns (summary, raw output, scene)."""
    res = brief.get("resolution") or {}
    sc = scene_from_brief(brief, base_dir, surroundings=installed)
    kw = {}
    if fields:
        sc.build(fine=float(res.get("copper", 0.5)), body=float(res.get("body", 1.0)))
        reach = float(res.get("reach", C0 / sc.band[0] * 1e3 / 8.0))
        kw["field_box"] = _field_box(sc, sc.sim, reach, *sc.antenna_box)
    summary, out = simulate(sc, fine=float(res.get("copper", 0.5)), body=float(res.get("body", 1.0)),
                            log=log, **kw)
    summary["materials"] = sc.materials
    return summary, out, sc


def tune(make, start, target=None, steps=None, bounds=None, depth=-20.0, rounds=10, installed=True,
         base_dir=None, log=None):
    """Solve an antenna's own dimensions on the board, and in the
    surroundings, it will work in: find the dimensions at which the port's
    reflection at the target frequency is zero.

    make(**dims) returns a brief; `start` is {name: mm} for the dimensions
    to solve (two of them: the reflection is a complex number, so two
    dimensions can bring it to zero, e.g. an inverted-F's arm length and
    tap position). No tuning rule is assumed: the reflection is read from
    the solved fields, its dependence on each dimension is measured by
    moving that dimension `steps[name]` mm (default 1) and solving again,
    and the root is found by Broyden's method (a secant method in two
    unknowns), each step a new field solution. Stops when the reflection at
    the target is under `depth` dB. Returns {dims, "history", "summary",
    "brief", "converged"}."""
    names = list(start)
    x = np.array([float(start[k]) for k in names])
    steps = steps or {}
    bounds = bounds or {}
    hist = []

    def reflect(xv):
        dims = {k: float(v) for k, v in zip(names, xv)}
        brief = make(**dims)
        ft = (target or 0.5 * (brief["band"][0] + brief["band"][1])) * 1e6
        summary, out, _ = simulate_brief(brief, base_dir, installed=installed)
        g = complex(np.interp(ft, out["freqs"], out["s11"].real), np.interp(ft, out["freqs"], out["s11"].imag))
        hist.append({**{k: round(v, 3) for k, v in dims.items()},
                     "s11AtTarget": round(20 * math.log10(max(abs(g), 1e-6)), 1),
                     "resonance": summary["resonance"]["frequency"]})
        if log:
            log(f"solve {len(hist)}: {hist[-1]}")
        return np.array([g.real, g.imag]), summary, brief

    def clip(xv):
        for i, k in enumerate(names):
            if k in bounds:
                xv[i] = min(max(xv[i], bounds[k][0]), bounds[k][1])
        return xv

    r, summary, brief = reflect(x)
    best = (np.linalg.norm(r), x.copy(), summary, brief)
    # how the reflection moves with each dimension, by moving it
    J = np.zeros((2, len(names)))
    for i, k in enumerate(names):
        h = float(steps.get(k, 1.0))
        xi = x.copy()
        xi[i] += h
        if k in bounds and xi[i] > bounds[k][1]:
            xi[i], h = x[i] - h, -h
        ri, si, bi = reflect(xi)
        J[:, i] = (ri - r) / h
        if np.linalg.norm(ri) < best[0]:
            best = (np.linalg.norm(ri), xi.copy(), si, bi)
    for _ in range(rounds):
        if 20 * math.log10(max(best[0], 1e-6)) < depth:
            break
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        # no step further than a quarter of the dimension: the reflection
        # is not linear that far
        dx *= min(1.0, *(0.25 * abs(x[i]) / max(abs(dx[i]), 1e-9) for i in range(len(names))))
        x_new = clip(x + dx)
        dx = x_new - x
        if np.linalg.norm(dx) < 1e-3:
            break
        r_new, s_new, b_new = reflect(x_new)
        J += np.outer(r_new - r - J @ dx, dx) / (dx @ dx)
        x, r = x_new, r_new
        if np.linalg.norm(r) < best[0]:
            best = (np.linalg.norm(r), x.copy(), s_new, b_new)
    norm, xb, summary, brief = best
    return {**{k: round(float(v), 3) for k, v in zip(names, xb)}, "history": hist, "summary": summary,
            "brief": brief, "s11AtTarget": round(20 * math.log10(max(norm, 1e-6)), 1),
            "converged": bool(20 * math.log10(max(norm, 1e-6)) < depth), "solves": len(hist)}


_E24 = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6,
                 6.2, 6.8, 7.5, 8.2, 9.1])


def match(freqs, z, bands, z0=50.0, parts=2, q_inductor=40.0, q_capacitor=150.0, efficiency=None, network=None):
    """The lumped network between a 50 ohm radio and an antenna whose
    impedance `z` (complex, at `freqs` in Hz) has been solved: every ladder
    of up to `parts` standard-value inductors and capacitors (series or
    shunt, in either order) is tried, and the one kept delivers the most
    power to the antenna at its worst frequency inside `bands` ([[lo, hi],
    ...] MHz: one fixed network for all of them). The parts lose power
    (inductor and capacitor quality factors of ordinary 0402 parts), and
    that loss is in the number. Nothing here changes the antenna: a
    network trades reflection for loss and bandwidth, and what it cannot
    recover (the Bode-Fano limit of a small radiator) shows as a worst
    case that stays low whatever the parts.

    Returns {"network": [{"place", "part", "value"}...] from the antenna
    toward the radio, "delivered": the fraction of the radio's power that
    reaches the antenna at each band's worst frequency, "worst", "s11"
    (dB, at `freqs`), and with `efficiency` (radiation efficiency at
    `freqs`) "radiated": delivered x radiation efficiency.

    `network` (a previous result's "network") skips the search and reads
    that network on this impedance: how a product matched in free space
    does once a hand or a cable has moved its antenna."""
    freqs, z = np.asarray(freqs, float), np.asarray(z, complex)
    sel = np.zeros(len(freqs), bool)
    for lo, hi in bands:
        sel |= (freqs >= lo * 1e6) & (freqs <= hi * 1e6)
    if not sel.any():
        raise ValueError("no solved frequency inside the bands")
    w = 2 * math.pi * freqs
    ind = np.concatenate([_E24 * d for d in (1e-9, 1e-8)])[:40]            # 1 to 47 nH
    ind = ind[ind <= 47e-9]
    cap = np.concatenate([_E24 * d for d in (1e-13, 1e-12, 1e-11)])
    cap = cap[(cap >= 0.2e-12) & (cap <= 33e-12)]
    kinds = [("series", "L"), ("series", "C"), ("shunt", "L"), ("shunt", "C")]

    def imp(part, values, fi):
        v = np.asarray(values, float)[:, None]
        if part == "L":
            x = w[None, fi] * v
            return x / q_inductor + 1j * x
        x = 1.0 / (w[None, fi] * v)
        return x / q_capacitor - 1j * x

    def through(place, part, values, zl, pl, fi):
        """Add one part toward the radio. zl: impedance seen so far (n, f);
        pl: fraction of the power entering there that reaches the antenna."""
        ze = imp(part, values, fi)                      # (m, f)
        zl, pl = zl[:, None, :], pl[:, None, :]
        if place == "series":
            zin = zl + ze[None]
            frac = zl.real / np.maximum(zin.real, 1e-12)
        else:
            y = 1.0 / zl + 1.0 / ze[None]
            zin = 1.0 / y
            frac = (1.0 / zl).real / np.maximum(y.real, 1e-18)
        return zin.reshape(-1, len(fi)), (pl * frac).reshape(-1, len(fi))

    def gain(zin, pl):
        g = (zin - z0) / (zin + z0)
        return (1 - np.abs(g) ** 2) * pl, g

    # the search runs on the in-band frequencies only (thinned to a few
    # dozen), one ladder shape at a time, so three parts fit in memory;
    # the network kept is then evaluated at every solved frequency
    idx = np.flatnonzero(sel)
    if len(idx) > 32:
        idx = idx[np.unique(np.linspace(0, len(idx) - 1, 32).round().astype(int))]
    full = np.arange(len(freqs))
    best = [None]

    def ladder(net, fi):
        zin, pl = z[None, fi], np.ones((1, len(fi)))
        for place, part, values in net:
            zin, pl = through(place, part, values, zin, pl, fi)
        return zin, pl

    def visit(net, zin, pl):
        g, _ = gain(zin, pl)
        worst = g.min(axis=1)
        k = int(np.argmax(worst))
        # a part more has to earn its place
        if best[0] is None or worst[k] > best[0][0] + (0.005 if len(net) > len(best[0][1]) else 0.0):
            vals, kk = [], k
            for _, _, values in reversed(net):          # earlier parts vary slowest
                vals.append(float(values[kk % len(values)]))
                kk //= len(values)
            best[0] = (float(worst[k]), [(pl_, pt, v) for (pl_, pt, _), v in zip(net, reversed(vals))])
        if len(net) == parts:
            return
        for place, part in kinds:
            if net and net[-1][0] == place:
                continue                                # two in a row of one kind are one part
            values = ind if part == "L" else cap
            z2, p2 = through(place, part, values, zin, pl, idx)
            visit(net + [(place, part, values)], z2, p2)

    if network is not None:
        net = [(n["place"], n["part"], n["value"] * (1e-9 if n["part"] == "L" else 1e-12)) for n in network]
    else:
        visit([], z[None, idx], np.ones((1, len(idx))))
        net = best[0][1]
    zin, pl = ladder([(pl_, pt, [v]) for pl_, pt, v in net], full)
    g, refl = gain(zin, pl)
    g, refl = g[0], refl[0]
    worst = float(g[sel].min())
    out = {"network": [{"place": pl_, "part": pt, "value": (round(v * 1e9, 2) if pt == "L" else round(v * 1e12, 2)),
                        "unit": "nH" if pt == "L" else "pF"} for pl_, pt, v in net],
           "worst": round(worst, 4), "worst_dB": round(10 * math.log10(max(worst, 1e-9)), 2),
           "delivered": [round(float(g[(freqs >= lo * 1e6) & (freqs <= hi * 1e6)].min()), 4) for lo, hi in bands],
           "gain": g, "s11": 20 * np.log10(np.maximum(np.abs(refl), 1e-6))}
    if efficiency is not None:
        out["radiated"] = g * np.asarray(efficiency, float)
    return out


def run_brief(brief, out_dir, base_dir=None, log=None):
    """`loadpath antenna`: the bare board, the board as installed, the
    difference, and the sensitivity field. Writes antenna.json, s11.csv and
    sensitivity.npz under out_dir; returns the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    has = bool(brief.get("surroundings"))
    # the bare board on the grid the installed one is solved on
    bare, bout, bsc = simulate_brief(brief, base_dir, installed="air" if has else False, fields=True, log=log)
    report = {"name": brief.get("name"), "bare": bare}
    cols = [bout["freqs"] / 1e6, 20 * np.log10(np.maximum(np.abs(bout["s11"]), 1e-6)),
            bout["z"].real, bout["z"].imag]
    head = "MHz,bare_s11_dB,bare_R,bare_X"
    sens = sensitivity(bsc, bout)
    np.savez_compressed(out_dir / "sensitivity.npz", x=sens["x"], y=sens["y"], z=sens["z"],
                        dielectric=sens["dielectric"].astype(np.complex64),
                        metal=sens["metal"].astype(np.complex64))
    report["sensitivity"] = _sens_summary(sens, bsc)
    if has:
        inst, iout, isc = simulate_brief(brief, base_dir, installed=True, log=log)
        report["installed"] = inst
        report["change"] = compare(inst, bare)
        cols += [20 * np.log10(np.maximum(np.abs(iout["s11"]), 1e-6)), iout["z"].real, iout["z"].imag]
        head += ",installed_s11_dB,installed_R,installed_X"
    np.savetxt(out_dir / "s11.csv", np.c_[tuple(cols)], delimiter=",", header=head, comments="", fmt="%.4f")
    final = report.get("installed") or bare
    limit = float(brief.get("limit", -10.0))
    report["ok"] = bool(final["inBand"]["worstS11"] <= limit)
    report["limit"] = limit
    report["seconds"] = round(time.time() - t0, 1)
    (out_dir / "antenna.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def _sens_summary(sens, scene):
    """Where the sensitivity is: the distance from the copper within which
    half, 90% and 99% of it lies (for metal), so "how far is far enough"
    has a number behind it."""
    x, y, z = np.meshgrid(sens["x"], sens["y"], sens["z"], indexing="ij")
    ex, ey, ez = sens["edges"]
    vol = np.diff(ex)[:, None, None] * np.diff(ey)[None, :, None] * np.diff(ez)[None, None, :]
    # distance to the antenna's own copper (the top sheets), by bounding box
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    if scene.sheets:
        zt = max(s["z"] for s in scene.sheets)
        for s in scene.sheets:
            if s["z"] == zt:
                for r in s["rings"]:
                    lo[:2], hi[:2] = np.minimum(lo[:2], r.min(0)), np.maximum(hi[:2], r.max(0))
        lo[2] = hi[2] = zt
    else:
        # no board copper: the product's own metal is the antenna
        lo, hi = (np.array(v, float) for v in scene.antenna_box)
    d = np.sqrt(sum(np.maximum(np.maximum(l - c, c - h), 0) ** 2 for c, l, h in zip((x, y, z), lo, hi)))
    wgt = np.abs(sens["metal"]) * vol
    order = np.argsort(d.ravel())
    cum = np.cumsum(wgt.ravel()[order])
    cum /= cum[-1]
    dd = d.ravel()[order]
    return {"frequency": round(sens["frequency"] / 1e6, 1),
            "metalWithin": {f"{int(q * 100)}%": round(float(dd[np.searchsorted(cum, q)]), 1)
                            for q in (0.5, 0.9, 0.99)},
            "note": "share of the port's first-order sensitivity to a small conductor that lies within "
                    "each distance (mm) of the antenna's copper; file sensitivity.npz has the field"}


# ------------------------------------------------------- inside a product --

def _inside_points(mesh, pitch=1.5):
    """Sample points inside a closed mesh on a grid of `pitch` mm, and the
    volume each stands for."""
    lo, hi = mesh.bounds
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    g = [lo[a] + (np.arange(n[a]) + 0.5) * (hi[a] - lo[a]) / n[a] for a in range(3)]
    edges = [np.r_[lo[a], lo[a] + np.arange(1, n[a] + 1) * (hi[a] - lo[a]) / n[a]] for a in range(3)]
    f = mesh_fraction(np.asarray(mesh.vertices), np.asarray(mesh.faces), *edges, sub=2)
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    vol = np.prod((hi - lo) / n) * f.ravel()
    keep = vol > 0
    return P[keep], vol[keep]


class EnclosureAntenna:
    """An antenna on a component of an enclosure brief: the bare board is
    solved once, with its sensitivity field; `judge` then prices any layout
    by what the other parts and the wall would do to the port, to first
    order, and `check` solves the chosen layout in full."""

    def __init__(self, comp, comps, wall_shape, skin, material, base_dir=None, weight=200.0, log=None):
        from .placement import component_mesh
        self.id = comp["id"]
        self.brief = {k: v for k, v in comp["antenna"].items() if k not in ("origin", "surroundings")}
        b = self.brief["board"]
        size = np.array([*(b.get("size") or np.ptp(np.asarray(b["outline"], float), axis=0)),
                         float(b.get("thickness", 1.6))])
        # the board brief's origin (its corner, bottom face) in the component's own frame
        self.origin = np.asarray(comp["antenna"].get("origin", -size / 2.0), float)
        self.weight, self.log = float(weight), log
        self.wall_shape, self.skin = wall_shape, float(skin)        # the skin as a shape (its sdf)
        self.wall = _material(material)
        self.material = material
        self.others = {}
        for c in comps:
            if c["id"] == self.id or c.get("rf") == "ignore":
                continue
            # what it is to a radio wave: stated (rf), else a closure of the shell (external: a lid, a
            # door) is the shell's material, and anything inside is metal until the brief says otherwise
            spec = c.get("rf", material if c.get("external") else "metal")
            eps, tan, metal, note = _material(spec)
            pts, vol = _inside_points(component_mesh(c, base_dir))
            self.others[c["id"]] = {"points": pts, "volumes": vol, "eps": None if metal else eps,
                                    "spec": spec,
                                    "assumed": "rf" not in c and not c.get("external"),
                                    "rigid": c.get("installedOn") == self.id}
        t0 = time.time()
        self.bare, out, sc = simulate_brief(self.brief, base_dir, installed=False, fields=True, log=log)
        self.sens = sensitivity(sc, out)
        self.bare_out = out
        ex, ey, ez = self.sens["edges"]
        # wall samples: the field box on a 1.5 mm grid, in the board's frame
        g = [np.arange(e[0] + 0.75, e[-1], self._PITCH) for e in (ex, ey, ez)]
        self.wall_pts = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        self.seconds = time.time() - t0
        self._grids, self._wall_occ = {}, None

    def _to_board(self, R, T, world):
        return (world - T) @ R - self.origin

    def predicted(self, tf):
        """{component or "wall": |change of reflection|} for a layout
        ({id: (R, T)} of each component's own frame)."""
        if self.id not in tf:
            return {}
        R, T = (np.asarray(v, float) for v in tf[self.id])
        out = {}
        for cid, o in self.others.items():
            if cid not in tf or o["rigid"]:
                continue
            Rc, Tc = (np.asarray(v, float) for v in tf[cid])
            pts = self._to_board(R, T, o["points"] @ Rc.T + Tc)
            out[cid] = predicted_change(self.sens, pts, o["volumes"], o["eps"])[1]
        if not self.wall[2]:
            world = (self.wall_pts + self.origin) @ R.T + T
            vol = self._wall_volume(self.wall_shape.sdf(world))
            m = vol > 0
            out["wall"] = predicted_change(self.sens, self.wall_pts[m], vol[m], self.wall[0])[1]
        return out

    # -- every candidate position at once, for placement's first ranking --

    _PITCH = 1.5

    @staticmethod
    def _binned(pts, values, volumes, pitch):
        """Sum `values * volumes` at points into cells of `pitch`: (grid of
        sums, grid of volumes, origin of cell [0, 0, 0]'s centre)."""
        lo = np.floor(pts.min(0) / pitch).astype(int)
        ijk = np.floor(pts / pitch).astype(int) - lo
        n = ijk.max(0) + 1
        flat = np.ravel_multi_index(ijk.T, n)
        w = np.broadcast_to(volumes, (len(pts),))
        vol = np.bincount(flat, w, minlength=int(np.prod(n))).reshape(n)
        if values is None:
            return None, vol, (lo + 0.5) * pitch
        tot = (np.bincount(flat, values.real * w, minlength=int(np.prod(n)))
               + 1j * np.bincount(flat, values.imag * w, minlength=int(np.prod(n)))).reshape(n)
        return tot, vol, (lo + 0.5) * pitch

    def _field_grid(self, R, T, metal):
        """The sensitivity (ohm per mm3) averaged over cells of _PITCH, in
        a frame turned by R and moved by T from the board component's own."""
        key = (metal, tuple(np.round(R, 6).ravel()), tuple(np.round(T, 4)))
        if key not in self._grids:
            if len(self._grids) > 64:
                self._grids.clear()
            s = self.sens
            ex, ey, ez = s["edges"]
            c = np.stack(np.meshgrid(s["x"], s["y"], s["z"], indexing="ij"), -1).reshape(-1, 3)
            v = (np.diff(ex)[:, None, None] * np.diff(ey)[None, :, None] * np.diff(ez)[None, None, :]).ravel()
            f = (s["metal"] if metal else s["dielectric"]).ravel()
            tot, vol, o = self._binned((c + self.origin) @ R.T + T, f, v, self._PITCH)
            dens = np.where(vol > 0, tot / np.maximum(vol, 1e-12), 0.0)
            self._grids[key] = (dens, o)
        return self._grids[key]

    def _wall_volume(self, d):
        """mm3 of wall in a cell of _PITCH whose centre is `d` from the
        skin's nearest face (negative inside): the share of the cell
        between the wall's two faces, for a wall `skin` thick."""
        p = self._PITCH
        return np.clip(0.5 - d / p, 0.0, min(1.0, self.skin / p)) * p ** 3

    def _wall_grid(self):
        """mm3 of wall in each cell of _PITCH over the skin."""
        if self._wall_occ is None:
            p = self._PITCH
            lo, hi = (np.asarray(v, float) for v in self.wall_shape.bounds())
            g = [np.arange(np.floor(lo[a] / p), np.ceil(hi[a] / p)) * p + 0.5 * p for a in range(3)]
            P = np.stack(np.meshgrid(*g, indexing="ij"), -1)
            d = self.wall_shape.sdf(P.reshape(-1, 3)).reshape(P.shape[:3])
            self._wall_occ = (self._wall_volume(d), np.array([g[0][0], g[1][0], g[2][0]]))
        return self._wall_occ

    def _correlate(self, A, a0, B, b0, T):
        """sum over u of B(u) A(u + T) at each row of T (grids of _PITCH,
        cell-centre origins a0 and b0), trilinear between lattice shifts."""
        from scipy.ndimage import map_coordinates
        from scipy.signal import fftconvolve
        C = fftconvolve(A, B[::-1, ::-1, ::-1], mode="full")
        idx = (np.atleast_2d(T) - (a0 - b0)) / self._PITCH + (np.array(B.shape) - 1)
        re = map_coordinates(C.real, idx.T, order=1, mode="constant", cval=0.0)
        im = map_coordinates(np.ascontiguousarray(C.imag), idx.T, order=1, mode="constant", cval=0.0) \
            if np.iscomplexobj(C) else 0.0
        return re + 1j * im

    def _gamma(self, dz):
        z, z0 = self.sens["impedance"], self.sens["z0"]
        return np.abs(2 * z0 * dz / (z + z0) ** 2)

    def quick(self, members, R, T, placed):
        """placement.place's first ranking: the first-order cost of every
        candidate position T ((n, 3), the moving parts' own frame at
        rotation R) at once, by correlating what moves with what stays."""
        T = np.atleast_2d(T)
        cost = np.zeros(len(T))
        zero = np.zeros(3)
        if self.id in members:
            # the board moves: the wall and the parts already in place stay
            if not self.wall[2]:
                W, w0 = self._wall_grid()
                Fd, f0 = self._field_grid(R, zero, False)
                k = 3.0 * (self.wall[0] - 1.0) / (self.wall[0] + 2.0)
                cost += self._gamma(k * self._correlate(W, w0, Fd, f0, T))
            for cid, o in self.others.items():
                if cid in members or o["rigid"] or cid not in placed:
                    continue
                Rc, Tc = (np.asarray(v, float) for v in placed[cid])
                _, B, b0 = self._binned(o["points"] @ Rc.T + Tc, None, o["volumes"], self._PITCH)
                if o["eps"] is None:
                    F, f0 = self._field_grid(R, zero, True)
                    k = 1.0
                else:
                    F, f0 = self._field_grid(R, zero, False)
                    k = 3.0 * (o["eps"] - 1.0) / (o["eps"] + 2.0)
                cost += self._gamma(k * self._correlate(B, b0, F, f0, T))
        elif self.id in placed:
            # another part moves past the board where it already is
            Ra, Ta = (np.asarray(v, float) for v in placed[self.id])
            for cid in members:
                o = self.others.get(cid)
                if o is None or o["rigid"]:
                    continue
                F, f0 = self._field_grid(Ra, Ta, o["eps"] is None)
                k = 1.0 if o["eps"] is None else 3.0 * (o["eps"] - 1.0) / (o["eps"] + 2.0)
                _, B, b0 = self._binned(o["points"] @ R.T, None, o["volumes"], self._PITCH)
                cost += self._gamma(k * self._correlate(F, f0, B, b0, T))
        return self.weight * cost

    def judge(self, info):
        """placement.place's judge: the cost of this position."""
        new = set(info["new"])
        pred = self.predicted(info["transforms"])
        if self.id not in new:
            pred = {k: v for k, v in pred.items() if k in new}
        return self.weight * float(sum(pred.values()))

    def check(self, lay, skin_mesh, out_dir, base_dir=None):
        """Solve the chosen layout in full: the shell, every other part
        (and what each is taken to be), the board. Returns the report."""
        M = np.asarray(lay["components"][self.id]["matrix"], float)
        R, T = M[:3, :3], M[:3, 3]
        sur, assumed = [], []
        v = self._to_board(R, T, np.asarray(skin_mesh.vertices))
        wall = self.material if not self.wall[2] else "metal"
        sur.append({"name": "shell", "material": wall, "mesh": (v, np.asarray(skin_mesh.faces))})
        for cid, m in lay["meshes"].items():
            if cid == self.id or cid not in self.others:
                continue
            o = self.others[cid]
            body = {"name": cid, "mesh": (self._to_board(R, T, np.asarray(m.vertices)), np.asarray(m.faces))}
            if o["eps"] is None:
                body["metal"] = True
            else:
                body["material"] = o["spec"]
            if o["assumed"]:
                assumed.append(cid)
            sur.append(body)
        brief = {**self.brief, "surroundings": sur}
        # the bare board again, on the grid of the installed solve: their difference is the product
        bare, bout, _ = simulate_brief(brief, base_dir, installed="air", log=self.log)
        inst, iout, isc = simulate_brief(brief, base_dir, installed=True, log=self.log)
        limit = float(self.brief.get("limit", -10.0))
        tf = {cid: (np.asarray(c["matrix"])[:3, :3], np.asarray(c["matrix"])[:3, 3])
              for cid, c in lay["components"].items()}
        rep = {"bare": bare, "installed": inst, "change": compare(inst, bare),
               "firstOrder": {k: round(v, 4) for k, v in self.predicted(tf).items()},
               "limit": limit, "ok": bool(inst["inBand"]["worstS11"] <= limit),
               "takenAsMetal": assumed,
               "seconds": round(self.seconds + bare["run"]["seconds"] + inst["run"]["seconds"], 1)}
        if out_dir:
            # the installed model as a brief of its own (bodies as STL, in the board's frame):
            # `loadpath antenna` solves it again, antenna.tune retunes the copper in it
            import trimesh
            folder = Path(out_dir) / f"antenna-{self.id}"
            folder.mkdir(parents=True, exist_ok=True)
            bodies = []
            for body in sur:
                trimesh.Trimesh(*body["mesh"], process=False).export(folder / f"{body['name']}.stl")
                bodies.append({**{k: v for k, v in body.items() if k != "mesh"}, "mesh": f"{body['name']}.stl"})
            (folder / "brief.json").write_text(json.dumps({**self.brief, "name": self.id, "surroundings": bodies},
                                                          indent=1, default=float))
            rep["brief"] = f"antenna-{self.id}/brief.json"
            f = iout["freqs"]
            b = bout
            np.savetxt(Path(out_dir) / f"antenna-{self.id}.csv",
                       np.c_[f / 1e6, 20 * np.log10(np.maximum(np.abs(b["s11"]), 1e-6)),
                             20 * np.log10(np.maximum(np.abs(iout["s11"]), 1e-6))],
                       delimiter=",", header="MHz,bare_s11_dB,installed_s11_dB", comments="", fmt="%.3f")
        return rep


def enclosure_antennas(comps, wall_shape, skin, material, placement_opts, base_dir=None, log=None):
    """The antennas of an enclosure brief ([EnclosureAntenna]) and the
    judge placement asks (None when there are none)."""
    ants = []
    opts = placement_opts.pop("antenna", None) or {}
    for c in comps:
        if c.get("antenna"):
            a = EnclosureAntenna(c, comps, wall_shape, skin, material, base_dir,
                                 weight=float(opts.get("weight", 200.0)), log=log)
            ants.append(a)
    if not ants:
        return [], None
    judge = lambda info: float(sum(a.judge(info) for a in ants))      # noqa: E731
    judge.quick = lambda members, R, T, placed: sum(a.quick(members, R, T, placed) for a in ants)
    return ants, judge


def both_judges(first, second):
    """Two placement judges as one: their costs add, either may refuse,
    and so do their quick estimates."""
    if first is None or second is None:
        return first or second

    def judge(info):
        a = first(info)
        if a is None:
            return None
        b = second(info)
        return None if b is None else a + b
    quicks = [q for q in (getattr(first, "quick", None), getattr(second, "quick", None)) if q is not None]
    if quicks:
        judge.quick = lambda *a: sum(q(*a) for q in quicks)
    return judge


ETA0 = 376.730313668


def pattern(sim, out, up=(0.0, 0.0, 1.0), n_theta=37, n_phi=72, sky=10.0):
    """Radiation pattern after a run (`sim.far_field` on the flux box), read
    the way an over-the-air test reports it, at each monitor frequency.

    up: the direction the product points at the sky in use (scene
    coordinates); sky: the elevation above the horizon (degrees) a GPS
    receiver needs satellites at. Returns per frequency: "directivity"
    (dBi, peak), "peakGain" (realized: of the power available from the
    radio, so it carries the match and every loss), "trp" (total radiated
    over available, dB: what a chamber's TRP divided by the conducted
    power is), "farOverFlux" (the far field's power over the box's flux:
    1 when the box is clean), "upperHemisphere" (share of radiated power
    above the horizon), "skyGain" and "skyRhcp" (mean realized gain, total
    and right-hand circular, above `sky` degrees elevation, dBi), and the
    gridded realized gain "gain" (linear) on "theta", "phi" (radians,
    scene frame)."""
    th = np.linspace(0.0, np.pi, n_theta)
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    ff = sim.far_field(T, P)
    # solid angle of each sample: trapezoid in theta (half at the poles)
    wt = np.full(n_theta, th[1] - th[0])
    wt[[0, -1]] *= 0.5
    dO = (np.sin(T) * wt[:, None]) * (2 * np.pi / n_phi)
    r = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    upv = np.asarray(up, float) / np.linalg.norm(up)
    elev = np.degrees(np.arcsin(np.clip(r @ upv, -1, 1)))
    db = lambda v: float(10 * np.log10(max(float(v), 1e-12)))
    res = []
    for i in range(len(ff["f"])):
        U = ff["U"][i]
        prad = float((U * dO).sum())
        pav = float(out["available"][i])
        G = 4 * np.pi * U / pav
        # right-hand circular for a wave leaving along r (IEEE, e^{jwt})
        er = (ff["E_theta"][i] - 1j * ff["E_phi"][i]) / math.sqrt(2)
        Gr = 4 * np.pi * (np.abs(er) ** 2 / (2 * ETA0)) / pav
        skym = elev >= sky
        res.append({"f": float(ff["f"][i]), "directivity": db(4 * np.pi * U.max() / prad),
                    "peakGain": db(G.max()), "trp": db(prad / pav),
                    "farOverFlux": round(prad / float(out["radiated"][i]), 4),
                    "upperHemisphere": round(float((U * dO * (0.5 + 0.5 * np.sign(np.round(elev, 6)))).sum()) / prad, 3),
                    "skyGain": db((G * dO)[skym].sum() / dO[skym].sum()),
                    "skyRhcp": db((Gr * dO)[skym].sum() / dO[skym].sum()),
                    "gain": G})
    return {"theta": T, "phi": P, "elevation": elev, "byFrequency": res}
