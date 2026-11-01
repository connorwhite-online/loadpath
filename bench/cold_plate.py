"""
Do lattices beat pin fins and channels in a liquid cold plate?

The case where a lattice should have its best chance: water, three dies at
different powers, and a U-turn flow path (inlet and outlet on the same side),
so straight channels can't just run from end to end and the heat isn't where
the flow naturally goes.

The plate: a 24 x 24 mm AlSi10Mg (LPBF) footprint, a 1.5 mm base, a 2.5 mm
cavity over it closed by an adiabatic lid, 5 x 2.5 mm inlet and outlet
channels 5 mm long on the x = 0 side (inlet at y 2-7, outlet at y 17-22;
the wall between them on that side is solid). Dies under the base (5 x 5 mm):
25 W at (17, 6), 25 W at (17, 18), 15 W at (8, 12), 65 W in all. Every
design holds the same metal fraction of the cavity, connected to the base,
and is rated by conjugate.py at the same pressure drop: lattice Boltzmann
water flow (flow.network_flow, port to port), then solid and water heat in
one finite-volume solve. Score: the hottest die's peak rise over the inlet
water (K), lower is better.

Designs (each family's one parameter swept on the coarse grid):
- pin fins: round, staggered, full cavity height, `pitch` mm,
- channels: walls along the streamlines of the empty cavity's potential
  flow (curved, following the U), `n` channels carrying equal flow,
- uniform gyroid network (solid side), `cell` mm,
- gyroid graded over the dies: denser where the heat goes in.

Run: `python bench/cold_plate.py --pitch 0.25 --out <json>` (resumes per
design; `--only name,name` to pick).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy import ndimage

from lattice_heatsink import gyroid_value, keep_connected, match  # noqa: E402
from loadpath.conjugate import solve_heat  # noqa: E402
from loadpath.heatx_thermal import FLUIDS  # noqa: E402
from loadpath.thermal import CONDUCTIVITY  # noqa: E402

L, W = 24.0, 24.0
BASE, GAP = 1.5, 2.5
ENTRY = 5.0
IN_Y, OUT_Y = (2.0, 7.0), (17.0, 22.0)
DIES = [((17.0, 6.0), 5.0, 25.0), ((17.0, 18.0), 5.0, 25.0), ((8.0, 12.0), 5.0, 15.0)]
VF = 0.3
MIN_GAP = 0.5      # mm, the narrowest channel a wall may leave
MIN_WALL = 0.5     # mm, the thinnest channel wall
METAL = "alsi10mg"


class Plate:
    def __init__(self, pitch):
        self.h = h = pitch
        self.nx = int(round((ENTRY + L) / h))
        self.ny = int(round(W / h))
        self.nz = int(round((BASE + GAP) / h))
        x = (np.arange(self.nx) + 0.5) * h - ENTRY
        y = (np.arange(self.ny) + 0.5) * h
        z = (np.arange(self.nz) + 0.5) * h
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")
        under = self.X >= 0
        cav = self.Z >= BASE
        self.base = under & ~cav
        self.space = under & cav
        inl = (self.Y > IN_Y[0]) & (self.Y < IN_Y[1])
        outl = (self.Y > OUT_Y[0]) & (self.Y < OUT_Y[1])
        self.entry_in = (self.X < 0) & cav & inl
        self.entry_out = (self.X < 0) & cav & outl
        self.flowable = self.space | self.entry_in | self.entry_out
        # heat in through the base's bottom layer, each die's power over its cells
        self.heat = np.zeros(self.X.shape)
        self.die_masks = []
        for (cx, cy), a, w in DIES:
            m = self.base & (self.Z < h) & (np.abs(self.X - cx) <= a / 2) & (np.abs(self.Y - cy) <= a / 2)
            self.heat[m] = w / m.sum()
            self.die_masks.append(m)
        self.power = float(sum(d[2] for d in DIES))
        first = np.arange(self.nx)[:, None, None] == 0
        self.inlet = (self.entry_in & first, 0, 1)
        self.outlet = (self.entry_out & first, 0, -1)

    def masks(self, fins):
        solid = self.base | (fins & self.space)
        fluid = self.flowable & ~solid
        return solid, fluid

    def vf(self, fins):
        return float((fins & self.space).sum() / self.space.sum())

    def ports(self):
        zc = BASE + GAP / 2
        r = 0.5 * np.hypot(IN_Y[1] - IN_Y[0], GAP) + 0.02
        mk = lambda y: {"point": [-ENTRY, 0.5 * (y[0] + y[1]), zc], "axis": [-1.0, 0.0, 0.0], "radius": float(r)}
        # network_flow's grid origin: the centre of voxel (0, 0, 0)
        return mk(IN_Y), mk(OUT_Y), [self.X[0, 0, 0], self.Y[0, 0, 0], self.Z[0, 0, 0]]


# ---------------------------------------------------------------- designs

def _dither(p: Plate, f, scale):
    """A periodic design sampled on a commensurate grid has only a few
    distinct values, so its metal jumps in steps of up to 5% as the level
    moves; a tiny fixed noise breaks the ties (it only decides which of the
    voxels on the same iso-level go first), so every design hits vf."""
    return f + scale * 1e-4 * np.random.default_rng(0).standard_normal(f.shape)


def pins(p: Plate, pitch, vf=VF):
    """Staggered round pins (rows across the plate every `pitch`, every other
    row shifted half a pitch), full height, diameter matched to vf."""
    xs = np.arange(pitch / 2, L, pitch)
    d2 = np.full(p.X.shape[:2], np.inf)
    X2, Y2 = p.X[:, :, 0], p.Y[:, :, 0]
    for i, xc in enumerate(xs):
        ys = np.arange(pitch / 2 + (pitch / 2 if i % 2 else 0.0), W, pitch)
        for yc in ys:
            d2 = np.minimum(d2, (X2 - xc) ** 2 + (Y2 - yc) ** 2)
    dist = _dither(p, np.broadcast_to(np.sqrt(d2)[:, :, None], p.X.shape), pitch)
    return match(p, lambda r: (dist <= r) & p.space, vf, 0.0, pitch / 2)


def stream_function(p: Plate):
    """2D stream function of potential flow through the empty cavity
    (x >= 0): 0 on the inner wall (the x = 0 side between the ports), 1 on
    the outer walls, no flux condition across the ports. Its level sets are
    the streamlines; walls along them split the flow evenly."""
    nx, ny = int(round(L / p.h)), p.ny
    x = (np.arange(nx) + 0.5) * p.h
    y = (np.arange(ny) + 0.5) * p.h
    n = nx * ny
    idx = np.arange(n).reshape(nx, ny)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    rhs = np.zeros(n)
    for ax in range(2):
        a = [slice(None)] * 2
        b = [slice(None)] * 2
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        ia, ib = idx[tuple(a)].ravel(), idx[tuple(b)].ravel()
        rows += [ia, ib]
        cols += [ib, ia]
        vals += [-np.ones(ia.size), -np.ones(ia.size)]
        np.add.at(diag, ia, 1.0)
        np.add.at(diag, ib, 1.0)

    def wall(cells, value):
        np.add.at(diag, cells, 2.0)              # half a cell to the wall
        np.add.at(rhs, cells, 2.0 * value)

    left = idx[0]
    inner = (y > IN_Y[1]) & (y < OUT_Y[0])
    outer = (y < IN_Y[0]) | (y > OUT_Y[1])
    wall(left[inner], 0.0)
    wall(left[outer], 1.0)
    wall(idx[-1], 1.0)
    wall(idx[:, 0], 1.0)
    wall(idx[:, -1], 1.0)
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    A = A + sp.diags(diag)
    psi = spl.spsolve(A.tocsc(), rhs).reshape(nx, ny)
    return x, y, psi


def channels(p: Plate, n, vf=VF):
    """Walls along n - 1 streamlines of the empty cavity's potential flow:
    curved channels that follow the U, evenly spaced across the plate's
    middle (the levels are psi at even steps along y = W/2, from the inner
    wall to the far one). Constant wall thickness, matched to vf; distance
    to a streamline from |psi - level| / |grad psi|. Where streamlines
    bunch up (at the ports, round the inner corners) a wall stops if its
    channels would be narrower than MIN_GAP, leaving the ports a plenum."""
    x, y, psi = stream_function(p)
    gx, gy = np.gradient(psi, p.h)
    g = np.maximum(np.hypot(gx, gy), 1e-12)
    mid = psi[:, int(np.argmin(np.abs(y - W / 2)))]
    levels = np.interp(np.arange(1, n) * L / n, x, mid)
    d = np.full(psi.shape, np.inf)
    gap = np.zeros(psi.shape)                  # psi between this wall and its neighbours
    edges = np.concatenate([[0.0], levels, [1.0]])
    for i, lv in enumerate(levels):
        di = np.abs(psi - lv) / g
        near = di < d
        d = np.where(near, di, d)
        gap = np.where(near, min(edges[i + 1] - edges[i], edges[i + 2] - edges[i + 1]), gap)
    width = gap / g                            # mm, the narrower adjacent channel's local pitch
    nx0 = p.nx - psi.shape[0]
    pad = lambda a, fill: np.concatenate([np.full((nx0, a.shape[1]), fill), a])[:, :, None]
    dist, width = pad(d, np.inf), pad(width, 0.0)
    # each wall takes the fraction f of its local channel pitch (thinner
    # where the channels narrow), down to MIN_WALL, and stops where the
    # channel left would be under MIN_GAP
    return match(p, lambda f: (dist <= 0.5 * f * width) & (f * width >= MIN_WALL) & ((1 - f) * width >= MIN_GAP)
                 & p.space, vf, 0.0, 0.95)


def gyroid(p: Plate, cell, vf=VF):
    g = _dither(p, gyroid_value(p, cell), 1.0)
    return match(p, lambda t: (g <= t) & p.space, vf, -1.5, 1.5)


def gyroid_graded(p: Plate, cell, field, vf=VF, rho_min=0.15, rho_max=0.55):
    """Network gyroid whose local solid fraction follows `field`, clipped to
    [rho_min, rho_max], scaled for the mean vf (the level for a fraction rho
    read off the uniform gyroid's value distribution)."""
    g = _dither(p, gyroid_value(p, cell), 1.0)
    vals = np.sort(g[p.space])
    f = np.clip(field, 0, None) / max(float(field[p.space].mean()), 1e-30)

    def build(c):
        rho = np.clip(c * f, rho_min, rho_max)
        return (g <= vals[np.clip((rho * (len(vals) - 1)).astype(int), 0, len(vals) - 1)]) & p.space

    return match(p, build, vf, 0.0, 5.0)


def die_field(p: Plate, sigma=4.0, floor=0.3):
    """Denser over the dies: each die's power per area spread by a Gaussian
    of `sigma` mm, plus a floor so the lattice holds together elsewhere."""
    f = np.zeros(p.X.shape)
    for (cx, cy), a, w in DIES:
        f += (w / a ** 2) * np.exp(-((p.X - cx) ** 2 + (p.Y - cy) ** 2) / (2 * sigma ** 2))
    return f / f[p.space].max() + floor


# ---------------------------------------------------------------- rating

LBM = {"cs": 0.17}


def flow(p: Plate, fluid, q_lpm):
    from loadpath.flow import network_flow

    pin, pout, origin = p.ports()
    kw = dict(medium="water", flow_lpm=q_lpm, fields=True, origin=origin)
    for extra in ({}, {"cs": 0.25}):
        rep = network_flow(fluid, p.h, pin, pout, **{**kw, **LBM, **extra})
        fld = rep.pop("fields")
        u = fld["velocity"][:, 1:-1, 1:-1, 1:-1]
        if np.isfinite(u).all() and 0 < rep["pressureDrop"] < 1e7 and rep.get("maxMach", 0) < 0.5:
            rep["velocity"] = u
            return rep
    raise RuntimeError(f"lattice Boltzmann diverged at {q_lpm:.3f} L/min")


def rate(p: Plate, fins, dp, guess, name):
    """Flow runs bracketing `dp` (Pa), each rated; the hottest die's rise at
    exactly dp read off the log-log line through the two nearest."""
    solid, fluid = p.masks(fins)
    t0 = time.time()
    pts = []
    q = guess
    for _ in range(4):
        fl = flow(p, fluid, q)
        ht = solve_heat(solid, fluid, fl["velocity"], p.h, CONDUCTIVITY[METAL], FLUIDS["water"], p.heat, p.power,
                        inlet=p.inlet, outlet=p.outlet, project=True)
        T = ht["T"]
        dies = [float(np.nanmax(T[m])) for m in p.die_masks]
        pts.append((q, fl["pressureDrop"], max(dies), dies, ht["balance"], fl.get("converged")))
        print(json.dumps({"design": name, "flowLpm": q, "dp": fl["pressureDrop"], "maxRise": max(dies),
                          "dies": dies, "balance": ht["balance"], "converged": fl.get("converged"),
                          "lbmSeconds": fl.get("seconds")}), flush=True)
        lo = [x for x in pts if x[1] <= dp]
        hi = [x for x in pts if x[1] > dp]
        if lo and hi:
            break
        slope = 1.3
        if len(pts) > 1:
            (q0, d0, *_), (q1, d1, *_) = pts[-2], pts[-1]
            slope = float(np.clip(np.log(d1 / d0) / np.log(q1 / q0), 1.0, 2.0))
        aim = dp * (1.15 if not hi else 1 / 1.15)
        q = q * (aim / fl["pressureDrop"]) ** (1.0 / slope)
    a = max((x for x in pts if x[1] <= dp), key=lambda x: x[1], default=None)
    b = min((x for x in pts if x[1] > dp), key=lambda x: x[1], default=None)
    if a is None or b is None:
        a, b = sorted(pts, key=lambda x: abs(np.log(x[1] / dp)))[:2]
    w = np.log(dp / a[1]) / np.log(b[1] / a[1])
    lerp = lambda u, v: float(np.exp((1 - w) * np.log(u) + w * np.log(v)))
    out = {"design": name, "vf": p.vf(fins), "dp": dp, "flowLpm": lerp(a[0], b[0]), "maxRise": lerp(a[2], b[2]),
           "dies": [lerp(u, v) for u, v in zip(a[3], b[3])], "balance": [x[4] for x in pts],
           "runs": [{"flowLpm": x[0], "dp": x[1], "maxRise": x[2], "converged": x[5]} for x in pts],
           "seconds": time.time() - t0}
    print(json.dumps(out), flush=True)
    return out


def build(p: Plate, name):
    kind, _, arg = name.partition("-")
    if kind == "pins":
        return pins(p, float(arg))
    if kind == "channels":
        return channels(p, int(arg))
    if kind == "gyroid":
        return gyroid(p, float(arg))
    if kind == "graded":
        cell, _, sig = arg.partition("-s")
        return gyroid_graded(p, float(cell), die_field(p, float(sig or 4.0)))
    raise ValueError(name)


DEFAULT = "pins-1.5,pins-2,pins-3,channels-6,channels-10,channels-14,gyroid-2,gyroid-3,gyroid-4,graded-3-s4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=0.25)
    ap.add_argument("--dp", type=float, default=2000.0)
    ap.add_argument("--only", default=DEFAULT)
    ap.add_argument("--guess", type=float, default=0.5)
    ap.add_argument("--out", default="bench/cold_plate.json")
    a = ap.parse_args()
    p = Plate(a.pitch)
    done = {}
    if Path(a.out).exists():
        done = {r["design"]: r for r in json.loads(Path(a.out).read_text())["results"]}
    results = list(done.values())
    guess = a.guess
    for name in [n for n in a.only.split(",") if n]:
        if name in done:
            continue
        fins = keep_connected(p, build(p, name))
        print(json.dumps({"design": name, "vf": p.vf(fins)}), flush=True)
        out = rate(p, fins, a.dp, guess, name)
        guess = out["flowLpm"]
        results.append(out)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"pitch": a.pitch, "dp": a.dp, "vf": VF, "power": p.power,
                                           "dies": DIES, "results": results}, indent=1))
        np.save(Path(a.out).with_suffix(f".{name}.npy"), np.packbits(fins))


if __name__ == "__main__":
    main()
