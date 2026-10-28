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
"""
from __future__ import annotations

import numpy as np

__all__ = ["AMFilter", "parse_build_direction"]

_P = 40.0
_XI0 = 0.5
_EPS = 1e-4
_N_SUPPORT = 5
_Q = _P + np.log(_N_SUPPORT) / np.log(_XI0)


def parse_build_direction(spec) -> tuple[int, int] | None:
    """'+z' -> (axis 2, +1). Layers stack along the direction; the build
    plate is at the far end opposite it."""
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if len(s) == 1:
        s = "+" + s
    if len(s) != 2 or s[0] not in "+-" or s[1] not in "xyz":
        raise ValueError("printDirection must be one of +x, -x, +y, -y, +z, -z (the direction layers stack)")
    return "xyz".index(s[1]), (1 if s[0] == "+" else -1)


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
