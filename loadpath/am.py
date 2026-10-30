"""
Overhang control for additive manufacturing: Langelaar's AM filter
(Langelaar 2017, "An additive manufacturing filter for topology
optimization of print-ready designs", SMO 55).

The part is printed layer by layer along a build direction. A cell can only
be printed if the layer below supports it, so its "printed" density is the
smaller of its own density and the strongest support beneath it:

    printed_i = smin(x_i, smax(printed_j for j in S_i))

where S_i is the cell directly below and its four neighbors in that layer.
That stencil allows overhangs up to 45 degrees, the usual self-supporting
limit for FDM and for metal powder-bed printing. The first layer sits on the
build plate. Running the optimizer on printed densities means material that
would need support simply earns nothing, so the design grows its own
supports: Y-shapes instead of flat ceilings.

smax/smin are smooth so gradients exist; the backward pass carries
sensitivities down through the layers (each layer's support depends on
every layer under it).

Any build direction works, not only the six grid axes. For a tilted one
(`TiltedAMFilter`) the support stencil is every neighbor among the 26 that
lies within 45 degrees of straight down, and the "layers" are bands of the
height along the build direction narrow enough that a cell's supports are
always in an earlier band. Along a grid axis that stencil is exactly
Langelaar's five cells.

The module also carries what a layered process does to the material
(`layered_elasticity`): FDM parts are slightly softer and much weaker across
layers than along them. Verification always solves with the layered
stiffness and checks the bond. The optimizer can design with it too
(process.anisotropic: true), or with the across-layer terms cut to the
bond's strength ratio ("bond"), a surrogate meant to stop compliance
minimization leaning on material loaded across its layers. Measured on the
motor mount and the printed hook at fixed volume fraction (2026-10-29),
neither beat the isotropic design reliably: compliance cannot tell tension
across layers (which breaks bonds) from compression (which doesn't), and
the layout shift moved the von Mises peak more than it saved the bond. So
the default designs isotropic; the bond enters through the safety-factor
search (verified margins include it) and the choice of build direction
(orient.py).
"""
from __future__ import annotations

import numpy as np

__all__ = ["AMFilter", "TiltedAMFilter", "parse_build_direction", "build_vector", "layered_elasticity",
           "direction_label"]

_P = 40.0
_XI0 = 0.5
_EPS = 1e-4
_N_SUPPORT = 5
_Q = _P + np.log(_N_SUPPORT) / np.log(_XI0)


def parse_build_direction(spec):
    """'+z' -> (axis 2, +1). Layers stack along the direction; the build
    plate is at the far end opposite it. A vector [x, y, z] gives a tilted
    build: returned as a unit 3-tuple, or as (axis, sign) when it lies on a
    grid axis."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 3:
        v = np.asarray(spec, float)
        nrm = np.linalg.norm(v)
        if not np.isfinite(nrm) or nrm == 0:
            raise ValueError("printDirection vector must be nonzero")
        v = v / nrm
        ax = int(np.argmax(np.abs(v)))
        if abs(v[ax]) > 1 - 1e-9:
            return ax, (1 if v[ax] > 0 else -1)
        return tuple(float(c) for c in v)
    if isinstance(spec, tuple) and len(spec) == 2:
        return int(spec[0]), int(spec[1])
    s = str(spec).strip().lower()
    if len(s) == 1:
        s = "+" + s
    if len(s) != 2 or s[0] not in "+-" or s[1] not in "xyz":
        raise ValueError("printDirection must be one of +x, -x, +y, -y, +z, -z (the direction layers stack), "
                         "or a vector [x, y, z]")
    return "xyz".index(s[1]), (1 if s[0] == "+" else -1)


def build_vector(direction) -> np.ndarray:
    """Unit vector of a build direction in either form."""
    if len(direction) == 2:
        v = np.zeros(3)
        v[int(direction[0])] = float(direction[1])
        return v
    v = np.asarray(direction, float)
    return v / np.linalg.norm(v)


def direction_label(direction) -> str:
    if direction is None:
        return "none"
    if len(direction) == 2:
        return ("+" if direction[1] > 0 else "-") + "xyz"[direction[0]]
    return "[" + ", ".join(f"{c:.3f}" for c in direction) + "]"


# Layered stiffness, as fractions of the in-plane values. Printed at full
# density FDM loses little stiffness across layers (published tensile
# moduli across layers sit at 80-95% of in-plane for PLA/PETG/ABS) but the
# bond carries 50-70% of the strength, and less in shear.
_LAYER_STIFFNESS = {"fdm": (0.85, 0.75), "sls": (0.95, 0.9), "mjf": (0.97, 0.95), "lpbf": (0.97, 0.95)}


def layered_elasticity(nu: float, process, design: bool = False):
    """The elasticity a solver should use for a material made by `process`
    (spec.Process): a plain Poisson's ratio when the process has no layers
    worth modeling, else a fem.Layered about the build direction.

    design=True is the "bond" surrogate: across-layer modulus and shear
    capped at the bond's strength ratio (see the module note for how it
    fared)."""
    if process is None or process.build is None or process.build == "auto":
        return nu
    ez, gz = _LAYER_STIFFNESS.get(process.kind, (1.0, 1.0))
    ez = getattr(process, "ez", None) or ez
    gz = getattr(process, "gz", None) or gz
    if design:
        ez = min(ez, process.interlayer)
        gz = min(gz, process.interlayer)
    if ez >= 1.0 and gz >= 1.0:
        return nu
    from .fem import Layered
    return Layered(nu, ez, gz, build_vector(process.build))


def _smin(x, xi):
    d = x - xi
    r = np.sqrt(d * d + _EPS)
    val = 0.5 * (x + xi - r + np.sqrt(_EPS))
    dx = 0.5 * (1.0 - d / r)
    dxi = 0.5 * (1.0 + d / r)
    return val, dx, dxi


def _neighbors(layer):
    """The 5 support cells under each cell, as shifted copies of the layer
    below (outside the grid counts as empty)."""
    out = [layer]
    for ax in (0, 1):
        for sh in (1, -1):
            s = np.roll(layer, sh, axis=ax)
            idx = [slice(None), slice(None)]
            idx[ax] = slice(0, 1) if sh == 1 else slice(-1, None)
            s[tuple(idx)] = 0.0
            out.append(s)
    return out


def _unshift(grads):
    """Adjoint of _neighbors: route each shifted copy's gradient back to the
    cell it came from."""
    total = grads[0].copy()
    k = 1
    for ax in (0, 1):
        for sh in (1, -1):
            g = grads[k].copy()
            idx = [slice(None), slice(None)]
            idx[ax] = slice(0, 1) if sh == 1 else slice(-1, None)
            g[tuple(idx)] = 0.0
            total += np.roll(g, -sh, axis=ax)
            k += 1
    return total


class AMFilter:
    def __new__(cls, dims, direction, pinned=None):
        if cls is AMFilter and len(direction) == 3:
            return TiltedAMFilter(dims, direction, pinned)
        return super().__new__(cls)

    def __init__(self, dims, direction: tuple[int, int], pinned: np.ndarray | None = None):
        self.dims = tuple(dims)
        self.axis, self.sign = direction
        # Pinned cells (preserved geometry) print as solid regardless: they
        # are the user's, and they support what is above them.
        self.pinned = None if pinned is None else self._to_layers(pinned.astype(float))
        self._cache = None

    def _to_layers(self, g):
        """Move the build axis first, ordered from the build plate up."""
        g = np.moveaxis(g, self.axis, 0)
        return g if self.sign > 0 else g[::-1]

    def _from_layers(self, g):
        g = g if self.sign > 0 else g[::-1]
        return np.moveaxis(g, 0, self.axis)

    def forward(self, field: np.ndarray) -> np.ndarray:
        x = self._to_layers(np.asarray(field, float))
        n = x.shape[0]
        printed = np.empty_like(x)
        cache = []
        for k in range(n):
            if k == 0:
                xi = np.ones_like(x[0])
                parts, ssum = None, None
            else:
                parts = _neighbors(printed[k - 1])
                ssum = sum(np.maximum(p, 0.0) ** _P for p in parts)
                xi = np.maximum(ssum, 1e-300) ** (1.0 / _Q)
            val, dx, dxi = _smin(x[k], xi)
            if self.pinned is not None:
                pin = self.pinned[k] > 0.5
                val = np.where(pin, 1.0, val)
                dx = np.where(pin, 0.0, dx)
                dxi = np.where(pin, 0.0, dxi)
            printed[k] = val
            cache.append((parts, ssum, dx, dxi))
        self._cache = cache
        return self._from_layers(printed)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Gradient wrt the input field of the last forward(), given the
        gradient wrt its output."""
        g = self._to_layers(np.asarray(grad, float)).copy()
        n = g.shape[0]
        out = np.zeros_like(g)
        carry = np.zeros_like(g[0])
        for k in range(n - 1, -1, -1):
            parts, ssum, dx, dxi = self._cache[k]
            total = g[k] + carry
            out[k] = total * dx
            carry = np.zeros_like(g[0])
            if k > 0:
                # d xi / d support_j = (P/Q) s_j^(P-1) * ssum^(1/Q - 1)
                common = total * dxi * (_P / _Q) * np.maximum(ssum, 1e-300) ** (1.0 / _Q - 1.0)
                grads = [common * np.maximum(p, 0.0) ** (_P - 1.0) for p in parts]
                carry = _unshift(grads)
        return self._from_layers(out)


class TiltedAMFilter:
    """The AM filter for a build direction off the grid axes (see the module
    note). The build plate is the plane through the lowest cell of the grid
    along the direction unless `plate` (a bool grid) says which cells sit on
    it: a support-free tilted print has to grow from wherever it touches."""

    def __init__(self, dims, direction, pinned: np.ndarray | None = None, plate: np.ndarray | None = None):
        self.dims = tuple(int(d) for d in dims)
        up = build_vector(direction)
        offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                         if (i, j, k) != (0, 0, 0)], float)
        proj = offs @ up
        cosang = -proj / np.linalg.norm(offs, axis=1)
        sel = cosang >= np.cos(np.radians(45.0)) - 1e-9
        self.offsets = offs[sel].astype(np.int64)
        n_sup = len(self.offsets)
        self._q = _P + np.log(n_sup) / np.log(_XI0)
        delta = float((-proj[sel]).min())
        ijk = np.indices(self.dims).reshape(3, -1).T
        h = ijk @ up
        level = np.floor((h - h.min()) / delta + 1e-9).astype(np.int64)
        n = len(ijk)
        self._nbr = []
        for o in self.offsets:
            q = ijk + o
            ok = np.all((q >= 0) & (q < np.array(self.dims)), axis=1)
            flat = np.full(n, -1, np.int64)
            flat[ok] = np.ravel_multi_index(q[ok].T, self.dims)
            self._nbr.append(flat)
        order = np.argsort(level, kind="stable")
        bounds = np.searchsorted(level[order], np.arange(level.max() + 2))
        self._groups = [order[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)
                        if bounds[k + 1] > bounds[k]]
        if plate is None:
            plate = level == 0
        else:
            plate = np.asarray(plate, bool).ravel()
        self._plate = plate
        self._pinned = None if pinned is None else np.asarray(pinned, bool).ravel()
        self._cache = None

    def forward(self, field: np.ndarray) -> np.ndarray:
        x = np.asarray(field, float).ravel()
        printed = np.zeros_like(x)
        padded = np.zeros(len(x) + 1)           # slot -1 reads as empty
        cache = []
        for idx in self._groups:
            nb = [f[idx] for f in self._nbr]
            parts = [padded[b] for b in nb]
            ssum = sum(np.maximum(p, 0.0) ** _P for p in parts)
            xi = np.maximum(ssum, 1e-300) ** (1.0 / self._q)
            on_plate = self._plate[idx]
            xi = np.where(on_plate, 1.0, xi)
            val, dx, dxi = _smin(x[idx], xi)
            dxi = np.where(on_plate, 0.0, dxi)
            if self._pinned is not None:
                pin = self._pinned[idx]
                val = np.where(pin, 1.0, val)
                dx = np.where(pin, 0.0, dx)
                dxi = np.where(pin, 0.0, dxi)
            printed[idx] = val
            padded[idx] = val
            cache.append((nb, parts, ssum, dx, dxi))
        self._cache = cache
        return printed.reshape(self.dims)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        gp = np.zeros(int(np.prod(self.dims)) + 1)
        gp[:-1] = np.asarray(grad, float).ravel()
        out = np.zeros(len(gp) - 1)
        for idx, (nb, parts, ssum, dx, dxi) in zip(reversed(self._groups), reversed(self._cache)):
            total = gp[idx]
            out[idx] = total * dx
            common = total * dxi * (_P / self._q) * np.maximum(ssum, 1e-300) ** (1.0 / self._q - 1.0)
            for b, p in zip(nb, parts):
                gp[b] += common * np.maximum(p, 0.0) ** (_P - 1.0)    # one offset: distinct targets
        return out.reshape(self.dims)
