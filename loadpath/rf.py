"""Maxwell's equations in the time domain on a graded Yee grid (FDTD).

The solver under `loadpath antenna` (antenna.py, docs/ANTENNA.md): fields
around a conductor pattern, in and near plastic, metal and board laminate,
from a pulse at a lumped port. One run gives the port's reflection and
impedance across a band, the power radiated through a closed surface, the
power lost in each lossy material, and the far field.

    sim = Sim(x, y, z, eps_r=..., tan_d=..., f0=2.45e9)   # nodes in metres
    sim.pec_x[...] = True                                 # conductor edges
    port = sim.add_port(axis=2, edges=[(i, j, k)], resistance=50.0)
    out = sim.run(freqs)

What is solved, and how each piece is checked (tests/test_rf.py):

* Yee's staggered leapfrog (Yee 1966; Taflove & Hagness, Computational
  Electrodynamics, 3rd ed., ch. 3) on a rectilinear grid whose spacing can
  change along each axis, so cells are small at a trace and large in the air
  round it. Second-order where the spacing is even. Checked on the modes of
  a closed metal box, empty and part-filled with dielectric (exact).
* Conductors are perfect: the electric field along a conductor edge is held
  at zero. A sheet (a trace, a ground pour) is a set of edges in one grid
  plane with no thickness. Copper's loss is added after the run from the
  surface current (`conductor_loss`), the standard perturbation for a good
  conductor whose skin depth (1.3 um at 2.4 GHz) no grid resolves.
* A lossy dielectric carries conductivity sigma = 2 pi f0 eps0 eps_r tan_d,
  exact at f0 and close across an antenna's band.
* The open boundary is a convolutional perfectly matched layer (Roden &
  Gedney, Microwave Opt. Technol. Lett. 27, 2000), complex-frequency
  shifted so stored (evanescent) near field is absorbed too.
* The port is a resistive voltage source on grid edges (Piket-May, Taflove
  & Baron, IEEE Trans. MTT 42, 1994). Its voltage and the source's give the
  reflection exactly as a network analyser defines it: S11 = 2V/Vs - 1.
* Radiated power is the Poynting flux through a closed box at each
  frequency; accepted power comes from the port. In a lossless model the
  two agree (the energy check), and the gap between them in a lossy one is
  the heat, which `run` also integrates cell by cell per material.
"""
from __future__ import annotations

import math

import numpy as np

try:  # compiled kernels; the numpy fallback is the same arithmetic, slower
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (len(a) == 1 and callable(a[0])) else a[0]
    prange = range

C0 = 299792458.0
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C0 * C0)
ETA0 = math.sqrt(MU0 / EPS0)


# ---------------------------------------------------------------- kernels --

@njit(parallel=True, fastmath=True, cache=True)
def _step_h(Ex, Ey, Ez, Hx, Hy, Hz, cx, cy, cz):
    nx, ny, nz = Hz.shape[0], Hz.shape[1], Hx.shape[2]
    for i in prange(nx + 1):
        for j in range(ny):
            for k in range(nz):
                Hx[i, j, k] -= (cy[j] * (Ez[i, j + 1, k] - Ez[i, j, k])
                                - cz[k] * (Ey[i, j, k + 1] - Ey[i, j, k]))
    for i in prange(nx):
        for j in range(ny + 1):
            for k in range(nz):
                Hy[i, j, k] -= (cz[k] * (Ex[i, j, k + 1] - Ex[i, j, k])
                                - cx[i] * (Ez[i + 1, j, k] - Ez[i, j, k]))
        for j in range(ny):
            for k in range(nz + 1):
                Hz[i, j, k] -= (cx[i] * (Ey[i + 1, j, k] - Ey[i, j, k])
                                - cy[j] * (Ex[i, j + 1, k] - Ex[i, j, k]))


@njit(parallel=True, fastmath=True, cache=True)
def _step_e(Ex, Ey, Ez, Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz, ix, iy, iz):
    nx, ny, nz = Hz.shape[0], Hz.shape[1], Hx.shape[2]
    for i in prange(nx):
        for j in range(1, ny):
            for k in range(1, nz):
                Ex[i, j, k] = Cax[i, j, k] * Ex[i, j, k] + Cbx[i, j, k] * (
                    (Hz[i, j, k] - Hz[i, j - 1, k]) * iy[j]
                    - (Hy[i, j, k] - Hy[i, j, k - 1]) * iz[k])
    for i in prange(1, nx):
        for j in range(ny):
            for k in range(1, nz):
                Ey[i, j, k] = Cay[i, j, k] * Ey[i, j, k] + Cby[i, j, k] * (
                    (Hx[i, j, k] - Hx[i, j, k - 1]) * iz[k]
                    - (Hz[i, j, k] - Hz[i - 1, j, k]) * ix[i])
        for j in range(1, ny):
            for k in range(nz):
                Ez[i, j, k] = Caz[i, j, k] * Ez[i, j, k] + Cbz[i, j, k] * (
                    (Hy[i, j, k] - Hy[i - 1, j, k]) * ix[i]
                    - (Hx[i, j, k] - Hx[i, j - 1, k]) * iy[j])


@njit(parallel=True, fastmath=True, cache=True)
def _pml_e(Ex, Ey, Ez, Hx, Hy, Hz, Cbx, Cby, Cbz,
           pyx, pzx, pxy, pzy, pxz, pyz, ex, ey, ez, bx, by, bz, cx, cy, cz, ix, iy, iz):
    """The layer's terms in the E update: for each derivative across a
    layer, the convolution psi of that derivative (p<component><axis>),
    added with the curl's sign. e* are the node indices inside the layers."""
    nx, ny, nz = Hz.shape[0], Hz.shape[1], Hx.shape[2]
    for m in prange(ex.shape[0]):
        i = ex[m]
        for j in range(ny):
            for k in range(nz + 1):
                p = bx[m] * pyx[m, j, k] + cx[m] * (Hz[i, j, k] - Hz[i - 1, j, k]) * ix[i]
                pyx[m, j, k] = p
                Ey[i, j, k] -= Cby[i, j, k] * p
        for j in range(ny + 1):
            for k in range(nz):
                p = bx[m] * pzx[m, j, k] + cx[m] * (Hy[i, j, k] - Hy[i - 1, j, k]) * ix[i]
                pzx[m, j, k] = p
                Ez[i, j, k] += Cbz[i, j, k] * p
    for i in prange(nx + 1):
        for m in range(ey.shape[0]):
            j = ey[m]
            if i < nx:
                for k in range(nz + 1):
                    p = by[m] * pxy[i, m, k] + cy[m] * (Hz[i, j, k] - Hz[i, j - 1, k]) * iy[j]
                    pxy[i, m, k] = p
                    Ex[i, j, k] += Cbx[i, j, k] * p
            for k in range(nz):
                p = by[m] * pzy[i, m, k] + cy[m] * (Hx[i, j, k] - Hx[i, j - 1, k]) * iy[j]
                pzy[i, m, k] = p
                Ez[i, j, k] -= Cbz[i, j, k] * p
        for m in range(ez.shape[0]):
            k = ez[m]
            if i < nx:
                for j in range(ny + 1):
                    p = bz[m] * pxz[i, j, m] + cz[m] * (Hy[i, j, k] - Hy[i, j, k - 1]) * iz[k]
                    pxz[i, j, m] = p
                    Ex[i, j, k] -= Cbx[i, j, k] * p
            for j in range(ny):
                p = bz[m] * pyz[i, j, m] + cz[m] * (Hx[i, j, k] - Hx[i, j, k - 1]) * iz[k]
                pyz[i, j, m] = p
                Ey[i, j, k] += Cby[i, j, k] * p


@njit(parallel=True, fastmath=True, cache=True)
def _pml_h(Ex, Ey, Ez, Hx, Hy, Hz,
           pyx, pzx, pxy, pzy, pxz, pyz, hx, hy, hz, bx, by, bz, cx, cy, cz, ix, iy, iz, dtm):
    """The layer's terms in the H update; h* are the cells inside the
    layers, i* the inverse cell widths."""
    nx, ny, nz = Hz.shape[0], Hz.shape[1], Hx.shape[2]
    for m in prange(hx.shape[0]):
        i = hx[m]
        for j in range(ny + 1):
            for k in range(nz):
                p = bx[m] * pyx[m, j, k] + cx[m] * (Ez[i + 1, j, k] - Ez[i, j, k]) * ix[i]
                pyx[m, j, k] = p
                Hy[i, j, k] += dtm * p
        for j in range(ny):
            for k in range(nz + 1):
                p = bx[m] * pzx[m, j, k] + cx[m] * (Ey[i + 1, j, k] - Ey[i, j, k]) * ix[i]
                pzx[m, j, k] = p
                Hz[i, j, k] -= dtm * p
    for i in prange(nx + 1):
        for m in range(hy.shape[0]):
            j = hy[m]
            for k in range(nz):
                p = by[m] * pxy[i, m, k] + cy[m] * (Ez[i, j + 1, k] - Ez[i, j, k]) * iy[j]
                pxy[i, m, k] = p
                Hx[i, j, k] -= dtm * p
            if i < nx:
                for k in range(nz + 1):
                    p = by[m] * pzy[i, m, k] + cy[m] * (Ex[i, j + 1, k] - Ex[i, j, k]) * iy[j]
                    pzy[i, m, k] = p
                    Hz[i, j, k] += dtm * p
        for m in range(hz.shape[0]):
            k = hz[m]
            for j in range(ny):
                p = bz[m] * pxz[i, j, m] + cz[m] * (Ey[i, j, k + 1] - Ey[i, j, k]) * iz[k]
                pxz[i, j, m] = p
                Hx[i, j, k] += dtm * p
            if i < nx:
                for j in range(ny + 1):
                    p = bz[m] * pyz[i, j, m] + cz[m] * (Ex[i, j, k + 1] - Ex[i, j, k]) * iz[k]
                    pyz[i, j, m] = p
                    Hy[i, j, k] -= dtm * p


# ------------------------------------------------------------------- grid --

def graded_lines(fixed, dmax, lo=None, hi=None, ratio=1.3, dmin=None):
    """Grid lines along one axis: every coordinate in `fixed` is a line (a
    trace edge, a board face), neighbours are never further apart than
    `dmax`, and spacing changes by about `ratio` at most from one cell to
    the next, so a fine region grades out into a coarse one without the
    reflection a sudden jump makes. `fixed` may hold (coordinate, spacing)
    pairs to ask for a local spacing, or (coordinate, spacing, rank): of
    lines closer than `dmin` (default a thousandth of dmax) only one is
    kept, the one of highest rank (the antenna's own copper before a body
    near it before a line that is only there for resolution), and of equal
    ranks the lower coordinate."""
    import bisect
    pts = []
    for f in fixed:
        if np.ndim(f):
            pts.append((float(f[0]), float(f[1]), int(f[2]) if len(f) > 2 else 1))
        else:
            pts.append((float(f), float(dmax), 1))
    for e in (lo, hi):
        if e is not None:
            pts.append((float(e), float(dmax), 0))
    tol = dmin if dmin else 1e-3 * dmax
    # lines of a higher rank are placed first and never move: one of a lower
    # rank that falls within `tol` of a line already placed becomes that line
    # (and hands it its spacing if finer)
    xs, d = [], []
    for p, q, _ in sorted(pts, key=lambda t: (-t[2], t[0])):
        i = bisect.bisect_left(xs, p)
        near = [j for j in (i - 1, i) if 0 <= j < len(xs) and abs(xs[j] - p) < tol]
        if near:
            j = min(near, key=lambda j: abs(xs[j] - p))
            d[j] = min(d[j], q)
        else:
            xs.insert(i, p)
            d.insert(i, min(q, dmax))
    xs, d = np.array(xs), np.array(d)
    # a spacing asked at one line limits its neighbours through the ratio
    g = ratio - 1.0
    for i in range(1, len(xs)):
        d[i] = min(d[i], d[i - 1] + g * (xs[i] - xs[i - 1]))
    for i in range(len(xs) - 2, -1, -1):
        d[i] = min(d[i], d[i + 1] + g * (xs[i + 1] - xs[i]))
    out = [xs[0]]
    for i in range(len(xs) - 1):
        L = xs[i + 1] - xs[i]
        # the spacing wanted along the interval, and lines at equal steps of
        # its integral: as few cells as the three limits allow, none a sliver
        s = np.linspace(0.0, L, 400)
        h = np.minimum(np.minimum(d[i] + g * s, d[i + 1] + g * (L - s)), dmax)
        cum = np.r_[0.0, np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(s))]
        n = max(1, int(math.ceil(cum[-1] - 0.05)))
        inner = np.interp(np.arange(1, n) * cum[-1] / n, cum, s)
        out.extend((xs[i] + inner).tolist())
        out.append(xs[i + 1])
    return np.array(out)


class Port:
    """A resistive voltage source on grid edges. `lines` is a list of edge
    chains (each a list of (i, j, k) along `axis`): the chains are in
    parallel, the edges of a chain in series."""

    def __init__(self, axis, lines, resistance):
        self.axis, self.lines, self.resistance = axis, lines, float(resistance)
        self.v = []        # port voltage at half steps (V)
        self.vs = []       # source voltage at half steps (V)


class Sim:
    """One FDTD model. Coordinates are node positions in metres; `eps_r` and
    `tan_d` are per cell (nx, ny, nz). `pml` is the layer's thickness in
    cells per face (x-, x+, y-, y+, z-, z+); 0 leaves a metal wall there."""

    def __init__(self, x, y, z, eps_r=None, tan_d=None, sigma=None, f0=2.45e9, pml=8,
                 courant=0.98, inv_eps_r=None):
        self.x, self.y, self.z = (np.asarray(a, float) for a in (x, y, z))
        self.d = [np.diff(a) for a in (self.x, self.y, self.z)]
        self.n = tuple(len(d) for d in self.d)
        nx, ny, nz = self.n
        self.f0 = float(f0)
        self.pml = tuple(int(p) for p in (pml if np.ndim(pml) else (pml,) * 6))
        self.eps_r = np.ones(self.n, np.float32) if eps_r is None else np.asarray(eps_r, np.float32)
        # the mean of 1/eps_r in each cell, where a cell holds two materials
        # (a surface cutting through it): with it the field across the
        # surface sees the two in series, the field along it in parallel
        self.inv_eps_r = None if inv_eps_r is None else np.asarray(inv_eps_r, np.float32)
        sig = np.zeros(self.n, np.float32) if sigma is None else np.asarray(sigma, np.float32).copy()
        if tan_d is not None:
            sig = sig + (2 * math.pi * self.f0 * EPS0) * self.eps_r * np.asarray(tan_d, np.float32)
        self.sigma = sig
        self.dt = courant / (C0 * math.sqrt(sum(1.0 / d.min() ** 2 for d in self.d)))
        f32 = np.float32
        self.Ex = np.zeros((nx, ny + 1, nz + 1), f32)
        self.Ey = np.zeros((nx + 1, ny, nz + 1), f32)
        self.Ez = np.zeros((nx + 1, ny + 1, nz), f32)
        self.Hx = np.zeros((nx + 1, ny, nz), f32)
        self.Hy = np.zeros((nx, ny + 1, nz), f32)
        self.Hz = np.zeros((nx, ny, nz + 1), f32)
        self.pec_x = np.zeros(self.Ex.shape, bool)
        self.pec_y = np.zeros(self.Ey.shape, bool)
        self.pec_z = np.zeros(self.Ez.shape, bool)
        self.ports = []
        self.steps = 0
        self._built = False

    # -- geometry helpers ---------------------------------------------------
    def pec_cells(self, mask):
        """Every edge of every cell in `mask` (nx, ny, nz) is conductor."""
        m = np.asarray(mask, bool)
        p = np.pad(m, 1)
        self.pec_x |= p[1:-1, :-1, :-1] | p[1:-1, 1:, :-1] | p[1:-1, :-1, 1:] | p[1:-1, 1:, 1:]
        self.pec_y |= p[:-1, 1:-1, :-1] | p[1:, 1:-1, :-1] | p[:-1, 1:-1, 1:] | p[1:, 1:-1, 1:]
        self.pec_z |= p[:-1, :-1, 1:-1] | p[1:, :-1, 1:-1] | p[:-1, 1:, 1:-1] | p[1:, 1:, 1:-1]

    def pec_sheet(self, axis, index, mask):
        """A conductor sheet with no thickness in the node plane `index`
        normal to `axis`. `mask` marks the sheet's faces (cells of that
        plane): the edges round each marked face are conductor."""
        m = np.asarray(mask, bool)
        p = np.pad(m, 1)
        a = p[1:-1, :-1] | p[1:-1, 1:]      # edges along the plane's first axis
        b = p[:-1, 1:-1] | p[1:, 1:-1]      # along its second
        if axis == 2:
            self.pec_x[:, :, index] |= a
            self.pec_y[:, :, index] |= b
        elif axis == 1:
            self.pec_x[:, index, :] |= a
            self.pec_z[:, index, :] |= b
        else:
            self.pec_y[index, :, :] |= a
            self.pec_z[index, :, :] |= b

    def add_port(self, axis, edges, resistance=50.0):
        """A port across `edges`: a list of (i, j, k) edges along `axis` in
        series, or a list of such lists side by side (in parallel, as under
        the width of a strip). Positive voltage is the far end of the edge
        above its near end."""
        lines = edges if isinstance(edges[0][0], (tuple, list)) else [edges]
        p = Port(axis, [list(map(tuple, ln)) for ln in lines], resistance)
        self.ports.append(p)
        self._built = False
        return p

    # -- coefficients -------------------------------------------------------
    def _edge_avg(self, a, axis):
        """Cell values to the edges along `axis`: the mean over the four
        cells round each edge, weighted by their share of the edge's dual
        face (exact for the field along an interface)."""
        o = [ax for ax in range(3) if ax != axis]
        w = np.ones(self.n)
        for ax in o:
            sh = [1, 1, 1]
            sh[ax] = -1
            w = w * self.d[ax].reshape(sh)
        num, den = a * w, w
        for ax in o:
            pad = [(0, 0)] * 3
            pad[ax] = (1, 1)
            num, den = np.pad(num, pad), np.pad(den, pad)
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
            num = num[tuple(sl0)] + num[tuple(sl1)]
            den = den[tuple(sl0)] + den[tuple(sl1)]
        return num / den

    def _dual(self, ax):
        d = self.d[ax]
        return np.r_[d[0], 0.5 * (d[:-1] + d[1:]), d[-1]]

    def build(self):
        dt, f32 = self.dt, np.float32
        self.cx, self.cy, self.cz = ((dt / (MU0 * d)).astype(f32) for d in self.d)
        self.dual = [self._dual(ax) for ax in range(3)]
        self.ix, self.iy, self.iz = ((1.0 / d).astype(f32) for d in self.dual)
        self._eps_edge, self.Ca, self.Cb = [], [], []
        if self.inv_eps_r is not None:
            centres = [0.5 * (a[1:] + a[:-1]) for a in (self.x, self.y, self.z)]
            grad = [g ** 2 for g in np.gradient(self.eps_r.astype(float), *centres)]
        for ax, pec in enumerate((self.pec_x, self.pec_y, self.pec_z)):
            eps = self._edge_avg(self.eps_r.astype(float), ax)
            if self.inv_eps_r is not None:
                # Along a surface between two dielectrics the tangential
                # field is continuous (the materials act in parallel: mean
                # of eps); across it the flux is (in series: mean of 1/eps).
                # n2 is the squared component of the surface normal along
                # this edge, from the gradient of eps (Farjadpour et al.,
                # Opt. Lett. 31, 2006, diagonal form).
                tot = self._edge_avg(grad[0] + grad[1] + grad[2], ax)
                n2 = np.where(tot > 0, self._edge_avg(grad[ax], ax) / np.maximum(tot, 1e-30), 0.0)
                inv = self._edge_avg(self.inv_eps_r.astype(float), ax)
                eps = 1.0 / (n2 * inv + (1 - n2) / eps)
            eps = EPS0 * eps
            sig = self._edge_avg(self.sigma.astype(float), ax)
            self._eps_edge.append(eps)
            beta = sig * dt / (2 * eps)
            ca, cb = (1 - beta) / (1 + beta), (dt / eps) / (1 + beta)
            self.Ca.append(ca)
            self.Cb.append(cb)
        # ports: a series resistor and source on each edge
        for p in self.ports:
            o = [ax for ax in range(3) if ax != p.axis]
            n_par, n_ser = len(p.lines), len(p.lines[0])
            r_edge = p.resistance * n_par / n_ser
            p._cs, p._len, p._idx = [], [], []
            for ln in p.lines:
                for e in ln:
                    L = self.d[p.axis][e[p.axis]]
                    A = self.dual[o[0]][e[o[0]]] * self.dual[o[1]][e[o[1]]]
                    eps = self._eps_edge[p.axis][e]
                    beta = dt * L / (2 * r_edge * A * eps)
                    self.Ca[p.axis][e] = (1 - beta) / (1 + beta)
                    self.Cb[p.axis][e] = (dt / eps) / (1 + beta)
                    p._cs.append(dt / (eps * r_edge * A * (1 + beta)) / n_ser)
                    p._len.append(L)
                    p._idx.append(e)
            p._ii = tuple(np.array([e[a] for e in p._idx]) for a in range(3))
            p._cs = np.array(p._cs, f32)
            p._len = np.array(p._len)
            p._npar = n_par
        for ax, pec in enumerate((self.pec_x, self.pec_y, self.pec_z)):
            pec = pec.copy()
            for p in self.ports:
                if p.axis == ax:
                    pec[p._ii] = False
            # the outer wall: tangential E stays zero there
            for o in range(3):
                if o != ax:
                    sl = [slice(None)] * 3
                    sl[o] = [0, -1]
                    pec[tuple(sl)] = True
            self.Ca[ax][pec] = 0.0
            self.Cb[ax][pec] = 0.0
            self.Ca[ax] = np.ascontiguousarray(self.Ca[ax], f32)
            self.Cb[ax] = np.ascontiguousarray(self.Cb[ax], f32)
        self._build_pml()
        self._built = True

    def _build_pml(self):
        """CPML profiles (Roden & Gedney 2000): polynomial conductivity,
        kappa 1, and a frequency shift alpha that falls from the inner face
        to the wall so the layer also damps stored near field."""
        dt, f32 = self.dt, np.float32
        nx, ny, nz = self.n
        order, a_max = 3, 2 * math.pi * self.f0 * EPS0 * 0.25
        idx_e, be, ce, idx_h, bh, ch = ([] for _ in range(6))
        for ax in range(3):
            n = self.n[ax]
            ie, b_e, c_e, ih, b_h, c_h = ([] for _ in range(6))
            for side, npml in ((0, self.pml[2 * ax]), (1, self.pml[2 * ax + 1])):
                if not npml:
                    continue
                dcell = self.d[ax][0] if side == 0 else self.d[ax][-1]
                s_max = 0.8 * (order + 1) / (ETA0 * dcell)

                def coef(depth):        # depth 0 at the inner face, 1 at the wall
                    s = s_max * depth ** order
                    a = a_max * (1 - depth)
                    b = math.exp(-(s + a) * dt / EPS0)
                    return b, s / (s + a) * (b - 1)
                base = npml if side == 0 else n - npml      # the inner face's node
                for m in range(1, npml):                    # E nodes inside the layer
                    b, c = coef(m / npml)
                    ie.append(base - m if side == 0 else base + m)
                    b_e.append(b)
                    c_e.append(c)
                for m in range(npml):                       # H cells inside the layer
                    b, c = coef((m + 0.5) / npml)
                    ih.append(base - 1 - m if side == 0 else base + m)
                    b_h.append(b)
                    c_h.append(c)
            idx_e.append(np.array(ie, np.int64))
            idx_h.append(np.array(ih, np.int64))
            for lst, v in ((be, b_e), (ce, c_e), (bh, b_h), (ch, c_h)):
                lst.append(np.array(v, f32))
        (mx, my, mz), (hx, hy, hz) = (len(a) for a in idx_e), (len(a) for a in idx_h)
        z = lambda *sh: np.zeros(sh, f32)
        self._pe = (z(mx, ny, nz + 1), z(mx, ny + 1, nz), z(nx, my, nz + 1), z(nx + 1, my, nz),
                    z(nx, ny + 1, mz), z(nx + 1, ny, mz), *idx_e, *be, *ce, self.ix, self.iy, self.iz)
        inv = tuple((1.0 / d).astype(f32) for d in self.d)
        self._ph = (z(hx, ny + 1, nz), z(hx, ny, nz + 1), z(nx + 1, hy, nz), z(nx, hy, nz + 1),
                    z(nx + 1, ny, hz), z(nx, ny + 1, hz), *idx_h, *bh, *ch, *inv, f32(dt / MU0))

    # -- stepping -----------------------------------------------------------
    def step(self, n, source=None):
        """Advance n steps. `source(t)` is the source voltage of port 0 (and
        `source` may be a list, one per port)."""
        if not self._built:
            self.build()
        E = (self.Ex, self.Ey, self.Ez)
        srcs = source if isinstance(source, (list, tuple)) else [source] * len(self.ports)
        for _ in range(n):
            _step_h(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz, self.cx, self.cy, self.cz)
            _pml_h(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz, *self._ph)
            t_half = (self.steps + 0.5) * self.dt
            old = [E[p.axis][p._ii].astype(float) for p in self.ports]
            _step_e(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz,
                    self.Ca[0], self.Cb[0], self.Ca[1], self.Cb[1], self.Ca[2], self.Cb[2],
                    self.ix, self.iy, self.iz)
            _pml_e(self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz,
                   self.Cb[0], self.Cb[1], self.Cb[2], *self._pe)
            for p, src, e0 in zip(self.ports, srcs, old):
                vs = float(src(t_half)) if src is not None else 0.0
                if vs:
                    E[p.axis][p._ii] -= p._cs * np.float32(vs)
                e1 = E[p.axis][p._ii].astype(float)
                p.v.append(-float(np.sum(0.5 * (e0 + e1) * p._len)) / p._npar)
                p.vs.append(vs)
            self.steps += 1

    # -- monitors -----------------------------------------------------------
    def _box(self, margin):
        """Node index ranges of the flux box: `margin` cells inside the layer."""
        r = []
        for ax in range(3):
            lo = self.pml[2 * ax] + margin
            hi = self.n[ax] - self.pml[2 * ax + 1] - margin
            r.append((lo, hi))
        return r

    def _faces(self, box):
        """Tangential E and H at the face centres of the box, both at the
        face plane: ((axis, side), E_u, E_v, H_u, H_v) with (axis, u, v)
        cyclic."""
        E = (self.Ex, self.Ey, self.Ez)
        H = (self.Hx, self.Hy, self.Hz)
        out = []
        for ax in range(3):
            u, v = (ax + 1) % 3, (ax + 2) % 3
            for side in (0, 1):
                i0 = box[ax][side]

                def avg(a, axis_in_face):
                    # node values to face centres along one in-face axis
                    return 0.5 * (a[:-1] + a[1:]) if axis_in_face == 0 else 0.5 * (a[:, :-1] + a[:, 1:])

                def get(F, comp, at):
                    sl = [None] * 3
                    sl[ax] = at
                    for w in (u, v):
                        sl[w] = slice(box[w][0], box[w][1] + 1)
                    # staggered along its own axis: cells, not nodes
                    sl[comp] = slice(box[comp][0], box[comp][1])
                    a = F[comp][tuple(sl)]
                    # order the two in-face axes as (u, v)
                    if u > v:
                        a = a.T
                    return a
                # E_u is staggered along u (cell-centred there), nodes along v
                Eu = avg(get(E, u, i0), 1)
                Ev = avg(get(E, v, i0), 0)
                # H_u sits half a cell off the plane on both sides, at nodes
                # along u and cells along v; H_v the reverse
                d0, d1 = self.d[ax][i0 - 1], self.d[ax][i0]
                w0, w1 = d1 / (d0 + d1), d0 / (d0 + d1)

                def geth(comp):
                    sl = [None] * 3
                    other = v if comp == u else u
                    sl[comp] = slice(box[comp][0], box[comp][1] + 1)
                    sl[other] = slice(box[other][0], box[other][1])
                    acc = 0
                    for at, w in ((i0 - 1, w0), (i0, w1)):
                        sl[ax] = at
                        a = H[comp][tuple(sl)]
                        acc = acc + w * (a.T if u > v else a)
                    return acc
                Hu = avg(geth(u), 0)
                Hv = avg(geth(v), 1)
                out.append(((ax, side), Eu, Ev, Hu, Hv))
        return out

    def run(self, freqs, source=None, bandwidth=None, max_time=None, decay=1e-4,
            margin=3, every=None, loss=True, min_time=None, log=None, monitor=None,
            field_box=None):
        """Drive port 0 with a pulse and run until the port has rung down.

        freqs: frequencies (Hz) of the port's reflection and impedance (any
        number: they come from the port's record after the run).
        monitor: frequencies of the powers (radiated, lost) and the far
        field, accumulated during the run; default five across `freqs`.

        field_box: cell index ranges ((i0, i1), (j0, j1), (k0, k1)) to keep
        the fields in: afterwards `self.E_cells` and `self.H_cells` hold the
        complex cell-centred fields there at each monitor frequency (for
        the sensitivity of the port to material placed there).

        The pulse is a Gaussian-modulated sine centred on f0 that covers
        `bandwidth` (default: wide enough for `freqs`, at least 60% of f0).
        Stops when the port voltage has stayed under `decay` of its peak for
        five periods, or at `max_time` seconds (then `rungDown` is False and
        the spectra carry a truncation ripple: the result says so).
        """
        if not self._built:
            self.build()
        freqs = np.atleast_1d(np.asarray(freqs, float))
        mon = (np.linspace(freqs.min(), freqs.max(), min(5, len(freqs))) if monitor is None
               else np.atleast_1d(np.asarray(monitor, float)))
        f0 = self.f0
        bw = bandwidth or max(0.6 * f0, 2.4 * max(abs(freqs - f0).max(), 1.0))
        tau = math.sqrt(2.0) * 2.146 / (math.pi * bw)      # -20 dB at f0 +/- bw/2
        t0 = 4.5 * tau
        if source is None:
            def source(t):
                return math.exp(-((t - t0) / tau) ** 2) * math.sin(2 * math.pi * f0 * (t - t0))
        dt = self.dt
        # the running transforms sample ten times per period of the highest
        # frequency the pulse carries any energy at
        every = every or max(1, int(1.0 / ((f0 + bw) * dt) / 10))
        max_time = max_time or 400.0 / f0
        min_time = min_time or (2 * t0 + 6.0 / f0)
        box = self._box(margin)
        w = 2 * math.pi * mon
        acc = None
        lossy = loss and bool(np.any(self.sigma > 0))
        lb = None
        if lossy:
            lb = [(int(a.min()), int(a.max()) + 1) for a in np.nonzero(self.sigma > 0)]
        if field_box is not None:
            fb = [(int(a), int(b)) for a, b in field_box]
            lb = fb if lb is None else [(min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(lb, fb)]
        if lb is not None:
            shape = (len(mon),) + tuple(b[1] - b[0] for b in lb)
            eacc = [np.zeros(shape, np.complex64) for _ in range(3)]
            hacc = [np.zeros(shape, np.complex64) for _ in range(3)] if field_box is not None else None
        port = self.ports[0]
        peak, quiet_since = 0.0, None
        while True:
            self.step(every, source)
            tE = self.steps * dt
            pe = (np.exp(-1j * w * tE) * (every * dt)).astype(np.complex64)
            ph = (np.exp(-1j * w * (tE - 0.5 * dt)) * (every * dt)).astype(np.complex64)
            faces = self._faces(box)
            if acc is None:
                acc = [[np.zeros((len(mon),) + f[1].shape, np.complex64) for _ in range(4)]
                       for f in faces]
            for a, f in zip(acc, faces):
                for q in range(4):
                    a[q] += (pe if q < 2 else ph)[:, None, None] * f[1 + q]
            if lb is not None:
                (i0, i1), (j0, j1), (k0, k1) = lb
                ex = self.Ex[i0:i1, j0:j1 + 1, k0:k1 + 1]
                ey = self.Ey[i0:i1 + 1, j0:j1, k0:k1 + 1]
                ez = self.Ez[i0:i1 + 1, j0:j1 + 1, k0:k1]
                cells = (0.25 * (ex[:, :-1, :-1] + ex[:, 1:, :-1] + ex[:, :-1, 1:] + ex[:, 1:, 1:]),
                         0.25 * (ey[:-1, :, :-1] + ey[1:, :, :-1] + ey[:-1, :, 1:] + ey[1:, :, 1:]),
                         0.25 * (ez[:-1, :-1] + ez[1:, :-1] + ez[:-1, 1:] + ez[1:, 1:]))
                for q in range(3):
                    eacc[q] += pe[:, None, None, None] * cells[q]
                if hacc is not None:
                    hx = self.Hx[i0:i1 + 1, j0:j1, k0:k1]
                    hy = self.Hy[i0:i1, j0:j1 + 1, k0:k1]
                    hz = self.Hz[i0:i1, j0:j1, k0:k1 + 1]
                    for q, c in enumerate((0.5 * (hx[:-1] + hx[1:]), 0.5 * (hy[:, :-1] + hy[:, 1:]),
                                           0.5 * (hz[:, :, :-1] + hz[:, :, 1:]))):
                        hacc[q] += ph[:, None, None, None] * c
            v = np.abs(port.v[-every:]).max()
            peak = max(peak, v)
            if log and (self.steps // every) % 200 == 0:
                log(f"t = {tE * 1e9:.2f} ns, port at {v / max(peak, 1e-30):.1e} of its peak")
            if tE > min_time:
                if v < decay * peak:
                    quiet_since = quiet_since if quiet_since is not None else tE
                    if tE - quiet_since > 5.0 / f0:
                        rung = True
                        break
                else:
                    quiet_since = None
            if tE >= max_time:
                rung = False
                break
        out = self.port_spectra(freqs)
        pm = self.port_spectra(mon)
        out.update(rungDown=rung, time=self.steps * dt, steps=self.steps,
                   residual=float(v / max(peak, 1e-30)), monitor=mon,
                   accepted=pm["accepted"], available=pm["available"])
        # radiated power through the box
        rad = np.zeros(len(mon))
        self.near = []
        for a, ((ax, side), *_f) in zip(acc, faces):
            u, v_ = (ax + 1) % 3, (ax + 2) % 3
            dA = (self.d[u][box[u][0]:box[u][1]][:, None]
                  * self.d[v_][box[v_][0]:box[v_][1]][None, :])
            S = a[0] * np.conj(a[3]) - a[1] * np.conj(a[2])
            rad += (1 if side else -1) * 0.5 * np.real(S * dA).sum(axis=(1, 2))
            self.near.append(((ax, side), a, dA))
        self.box, self.monitor = box, mon
        out["radiated"] = rad
        out["radiationEfficiency"] = rad / out["accepted"]
        out["totalEfficiency"] = rad / out["available"]
        out["current"] = pm["current"]
        if lb is not None:
            self.E_cells, self.H_cells, self.field_box = eacc, hacc, lb
            (i0, i1), (j0, j1), (k0, k1) = lb
            vol = (self.d[0][i0:i1, None, None] * self.d[1][None, j0:j1, None]
                   * self.d[2][None, None, k0:k1])
            heat = 0.5 * self.sigma[i0:i1, j0:j1, k0:k1] * vol * sum(np.abs(e) ** 2 for e in eacc)
            self.heat, self.heat_box = heat, lb       # W per cell (per monitor frequency)
            out["dissipated"] = heat.sum(axis=(1, 2, 3))
        return out

    def port_spectra(self, freqs, port=0):
        """Reflection, impedance and powers of a port at `freqs` from its
        voltage record."""
        p = self.ports[port]
        freqs = np.atleast_1d(np.asarray(freqs, float))
        t = (np.arange(len(p.v)) + 0.5) * self.dt
        ker = np.exp(-2j * math.pi * freqs[:, None] * t[None, :]) * self.dt
        V = ker @ np.asarray(p.v)
        Vs = ker @ np.asarray(p.vs)
        R = p.resistance
        s11 = 2 * V / Vs - 1
        cur = (Vs - V) / R
        return {
            "freqs": freqs, "s11": s11, "z": V / cur, "current": cur,
            "accepted": 0.5 * np.real(V * np.conj(cur)),
            "available": np.abs(Vs) ** 2 / (8 * R),
        }


def resonance(freqs, s11):
    """The frequency of the deepest reflection minimum, refined by a
    parabola through the three samples round it, and its depth in dB."""
    m = np.abs(np.asarray(s11))
    i = int(np.argmin(m))
    f = np.asarray(freqs, float)
    if 0 < i < len(m) - 1:
        y0, y1, y2 = m[i - 1], m[i], m[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        fr = f[i] + off * (f[i + 1] - f[i])
        depth = y1 - 0.25 * (y0 - y2) * off
    else:
        fr, depth = f[i], m[i]
    return float(fr), float(20 * math.log10(max(depth, 1e-6)))
