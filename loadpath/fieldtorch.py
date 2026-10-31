"""
Field graphs evaluated with torch: on a GPU (CUDA or Metal) when there is
one, the same code on the CPU otherwise.

Meshing a field graph is almost all field evaluation (marching cubes is a
few percent): every op of every node over millions of sample points, a
dozen elementwise passes each. That is what a GPU does best, and the
editor already shows it, raymarching the same graphs in a fragment shader.
This module runs FieldGraph's ops as torch tensor ops instead of numpy:
constant, coordinate, the analytic shapes (box, cylinder, tube, sphere and
their unions), remap (curves and tables), expr, the smooth booleans, shell,
offset, transform, TPMS lattices (cartesian, with the graded cell's warp
map sampled trilinearly on the device) and sampled fields on a regular
grid (a study's result, an imported grid, an image).

What it does not port runs as before: such a node (an exact distance to a
mesh, a distance to points, a cylindrical lattice, a cross-faded grading,
an offset or mesh shape) is evaluated by the numpy graph on the CPU, with
everything under it, and its values copied over. So every graph runs, and
the answer is the numpy one: in float64 to ~1e-7 (sampled fields are
stored as float32, which numpy interpolates in float32); in float32, the
default on a GPU, where float64 is slow or missing, to ~5e-4 mm on the
graded arm link (180 mm across, the lattice's phase is where the digits
go), a fraction of a percent of any pitch it is meshed at.

Selection: LOADPATH_FIELD_DEVICE=auto (default: CUDA, then Metal; numpy
when there is neither), cuda, mps, torch (torch on the CPU) or numpy;
LOADPATH_FIELD_DTYPE=float32|float64 overrides the precision (float64 on
the CPU, float32 on a GPU by default).
"""
from __future__ import annotations

import ast
import math
import os

import numpy as np

__all__ = ["field_device", "TorchField"]


def field_device(name: str | None = None):
    """The torch device field graphs evaluate on, or None for numpy."""
    name = (name if name is not None else os.environ.get("LOADPATH_FIELD_DEVICE", "auto")).strip().lower()
    if name in ("", "cpu", "none", "off", "numpy"):
        return None
    try:
        import torch
    except ImportError:
        if name == "auto":
            return None
        raise RuntimeError(f"LOADPATH_FIELD_DEVICE={name} needs torch (pip install 'loadpath[gpu]')")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return None
    if name == "torch":
        return torch.device("cpu")
    return torch.device(name)


class TorchField:
    """`TorchField(graph, device)(P)`: graph.evaluate(P, name) with torch.
    P and the result are numpy (float64); in between, tensors on `device`."""

    def __init__(self, graph, device=None, dtype=None):
        import torch

        self.torch = torch
        self.g = graph
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if dtype is None:
            want = os.environ.get("LOADPATH_FIELD_DTYPE", "")
            dtype = {"float32": torch.float32, "float64": torch.float64}.get(
                want, torch.float64 if self.device.type == "cpu" else torch.float32)
        self.dtype = dtype
        self._exprs, self._grids = {}, {}
        self.fallbacks = set()          # nodes that ran on numpy, for the curious
        # On a GPU the mesher calls serially in batches this big (enough
        # to amortize kernel launches; a deep graph's ~40 intermediates
        # still fit). On the CPU, None: the mesher's thread pool beats
        # torch's own threads on one big batch (10 s vs 19 s, arm link).
        b = int(os.environ.get("LOADPATH_FIELD_BATCH", 0))
        self.batch = b or (None if self.device.type == "cpu" else 4_000_000)

    # ---- entry ----

    def __call__(self, P, name=None):
        torch = self.torch
        P = np.asarray(P, float).reshape(-1, 3)
        Pt = torch.as_tensor(P, dtype=self.dtype, device=self.device)
        memo = {"np": (P, {}), "P": Pt}
        out = self._eval(name or self.g.output, Pt, memo)
        if not torch.is_tensor(out):
            return np.full(len(P), float(out))
        return np.broadcast_to(out.to(torch.float64).cpu().numpy(), (len(P),)).astype(float)

    # ---- helpers ----

    def _t(self, x):
        torch = self.torch
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(np.asarray(x, float), dtype=self.dtype, device=self.device)

    def _val(self, r, P, memo):
        if isinstance(r, float):
            return r
        return self._eval(r, P, memo)

    def _eval(self, name, P, memo):
        key = (name, id(P))
        if key in memo:
            return memo[key][1]
        try:
            out = self._compute(self.g.nodes[name], P, memo)
        except _NumpyOnly:
            out = self._numpy(name, P, memo)
        memo[key] = (P, out)
        return out

    def _numpy(self, name, P, memo):
        """The node (and what it uses) by the numpy graph, copied over."""
        self.fallbacks.add(name)
        if P is memo["P"]:
            Pn, nmemo = memo["np"]
        else:                      # a transformed frame: its own points
            Pn, nmemo = P.to(self.torch.float64).cpu().numpy(), {}
        out = self.g._eval(name, Pn, nmemo)
        return out if isinstance(out, float) else self._t(np.asarray(out, float))

    def _interp(self, x, xs, ys):
        """np.interp on tensors: piecewise linear, clamped at the ends."""
        torch = self.torch
        xs, ys = self._t(xs).contiguous(), self._t(ys)
        x = self._t(x)
        i = torch.clamp(torch.searchsorted(xs, x.contiguous()), 1, len(xs) - 1)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        t = torch.clamp((x - x0) / (x1 - x0), 0.0, 1.0)
        return y0 + (y1 - y0) * t

    def _trilinear(self, key, grid, idx):
        """map_coordinates(grid, idx, order=1, mode='nearest') on the device:
        `idx` (N, 3) fractional grid indices."""
        torch = self.torch
        if key not in self._grids:
            G = self._t(np.asarray(grid, float))
            shape = list(G.shape)
            self._grids[key] = (G.reshape(-1), torch.as_tensor(shape, device=self.device),
                                (shape[1] * shape[2], shape[2]))
        G, n, (sx, sy) = self._grids[key]
        i = torch.minimum(torch.clamp(idx, min=0.0), (n - 1).to(idx.dtype))
        i0 = torch.minimum(torch.floor(i).long(), torch.clamp(n - 2, min=0))
        f = i - i0.to(i.dtype)
        step = torch.minimum(i0 + 1, n - 1) - i0          # 0 on a one-sample axis
        base = i0[:, 0] * sx + i0[:, 1] * sy + i0[:, 2]
        dx, dy, dz = step[:, 0] * sx, step[:, 1] * sy, step[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        c00 = torch.lerp(G[base], G[base + dz], fz)
        c01 = torch.lerp(G[base + dy], G[base + dy + dz], fz)
        c10 = torch.lerp(G[base + dx], G[base + dx + dz], fz)
        c11 = torch.lerp(G[base + dx + dy], G[base + dx + dy + dz], fz)
        return torch.lerp(torch.lerp(c00, c01, fy), torch.lerp(c10, c11, fy), fx)

    def _smin(self, a, b, k):
        torch = self.torch
        a, b = self._t(a), self._t(b)
        if isinstance(k, float) and k <= 1e-9:
            return torch.minimum(a, b)
        k = self._t(k)
        hard = k <= 1e-9
        kk = torch.where(hard, torch.ones_like(k), k)
        h = torch.clamp(0.5 + 0.5 * (b - a) / kk, 0.0, 1.0)
        s = b * (1 - h) + a * h - kk * h * (1 - h)
        return torch.where(hard, torch.minimum(a, b), s)

    # ---- ops ----

    def _compute(self, nd, P, memo):
        torch = self.torch
        s, r, op = nd.spec, nd.refs, nd.op
        v = lambda key, default=None: self._val(r[key], P, memo) if key in r else default  # noqa: E731
        if op == "constant":
            return float(s["value"])
        if op == "coordinate":
            ax = s.get("axis", "z")
            if isinstance(ax, str):
                return P[:, "xyz".index(ax)]
            a = np.asarray(ax, float)
            a = a / np.linalg.norm(a)
            return (P - self._t(s.get("origin", (0, 0, 0)))) @ self._t(a)
        if op == "shape":
            from .shapes import shape_from_spec

            if s["shape"].get("type") == "mesh":
                raise _NumpyOnly
            shp = self.g._resource(nd, lambda: shape_from_spec(s["shape"], self.g.base_dir))
            return self._shape(shp, P)
        if op in ("result", "grid", "image"):
            smp = self.g._resource(nd, lambda: self.g._samples(nd))
            if smp.tree is not None:
                raise _NumpyOnly
            idx = (P - self._t(smp.origin)) / self._t(smp.pitch)
            return self._trilinear(("samples", nd.name, id(smp)), smp.grid, idx)
        if op == "remap":
            return self._remap(v("input"), s, nd.name)
        if op == "expr":
            fn = self._exprs.get(nd.name)
            if fn is None:
                from .fields import _compile_expr

                fn = self._exprs[nd.name] = _compile_expr(s["expr"], self._lib())[1]
            env = {"x": P[:, 0], "y": P[:, 1], "z": P[:, 2]}
            for k, ref in r["vars"].items():
                env[k] = self._val(ref, P, memo)
            out = fn(env)
            return out if torch.is_tensor(out) else float(out)
        if op in ("union", "intersect", "subtract"):
            vals = [self._val(x, P, memo) for x in r["inputs"]]
            k = v("blend", 0.0)
            out = self._t(vals[0])
            for b in vals[1:]:
                if op == "union":
                    out = self._smin(out, b, k)
                elif op == "intersect":
                    out = -self._smin(-out, -self._t(b), k)
                else:
                    out = -self._smin(-out, b, k)
            return out
        if op == "shell":
            f, t = self._t(v("input")), v("thickness")
            side = s.get("side", "inside")
            if side == "inside":
                return torch.maximum(f, -(f + t))
            if side == "outside":
                return torch.maximum(-f, f - t)
            if side == "center":
                return torch.abs(f) - 0.5 * t
            raise _NumpyOnly
        if op == "offset":
            return v("input") - v("distance")
        if op == "lattice":
            return self._lattice(nd, P, memo)
        if op == "transform":
            return self._transform(nd, P, memo)
        if op == "distance":
            return self._distance(nd, P, memo)
        raise _NumpyOnly           # distance to points or a shape, mesh: exact geometry stays on the CPU

    def _distance(self, nd, P, memo):
        from .shapes import shape_from_spec

        torch = self.torch
        s, r = nd.spec, nd.refs
        if "input" in r:
            f = self._t(self._val(r["input"], P, memo))
            return f if s.get("signed") else torch.abs(f)
        if "segment" in s:
            a, b = (np.asarray(x, float) for x in s["segment"])
            ab = b - a
            t = torch.clamp(((P - self._t(a)) @ self._t(ab)) / max(float(ab @ ab), 1e-300), 0.0, 1.0)
            return torch.linalg.norm(P - (self._t(a) + t[:, None] * self._t(ab)), dim=1)
        if "plane" in s:
            n = np.asarray(s["plane"]["normal"], float)
            return torch.abs((P - self._t(s["plane"]["point"])) @ self._t(n / np.linalg.norm(n)))
        if "shape" in s:
            shp = self.g._resource(nd, lambda: shape_from_spec(s["shape"], self.g.base_dir))
            return torch.abs(self._shape(shp, P))
        raise _NumpyOnly             # points (a k-d tree)

    def _shape(self, shp, P):
        from .shapes import Box, Cylinder, Sphere, Union

        torch = self.torch
        if isinstance(shp, Box):
            c, h = self._t(0.5 * (shp.lo + shp.hi)), self._t(0.5 * (shp.hi - shp.lo))
            q = torch.abs(P - c) - h
            return torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)
        if isinstance(shp, Cylinder):
            d = P - self._t(shp.p0)
            ax = self._t(shp.axis)
            t = d @ ax
            radial = torch.linalg.norm(d - t[:, None] * ax, dim=-1)
            half = 0.5 * shp.length
            if shp.r_inner > 0:
                rad_q = torch.abs(radial - 0.5 * (shp.r + shp.r_inner)) - 0.5 * (shp.r - shp.r_inner)
            else:
                rad_q = radial - shp.r
            q = torch.stack([rad_q, torch.abs(t - half) - half], dim=-1)
            return torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)
        if isinstance(shp, Sphere):
            return torch.linalg.norm(P - self._t(shp.center), dim=-1) - shp.r
        if isinstance(shp, Union):
            out = None
            for x in shp.shapes:
                d = self._shape(x, P)
                out = d if out is None else torch.minimum(out, d)
            return out
        raise _NumpyOnly

    def _remap(self, x, s, name):
        torch = self.torch
        x = self._t(x)
        if len(s.get("from", ())) > 2 or len(s.get("to", ())) > 2:
            return self._interp(x, np.asarray(s["from"], float), np.asarray(s["to"], float))
        a, b = (float(u) for u in s.get("from", (0.0, 1.0)))
        c, d = (float(u) for u in s.get("to", (0.0, 1.0)))
        if a == b:
            raise _NumpyOnly           # the numpy path names the mistake
        u = (x - a) / (b - a)
        if s.get("clamp", True):
            u = torch.clamp(u, 0.0, 1.0)
        curve = s.get("curve", "linear")
        if isinstance(curve, (list, tuple)):
            pts = np.asarray(curve, float)
            return c + (d - c) * self._interp(torch.clamp(u, float(pts[0, 0]), float(pts[-1, 0])), pts[:, 0],
                                              pts[:, 1])
        if curve == "smoothstep":
            uc = torch.clamp(u, 0.0, 1.0)
            u = uc * uc * (3 - 2 * uc)
        elif curve == "smootherstep":
            uc = torch.clamp(u, 0.0, 1.0)
            u = uc ** 3 * (uc * (6 * uc - 15) + 10)
        elif curve == "sqrt":
            u = torch.sqrt(torch.clamp(u, min=0.0))
        elif curve == "square":
            u = u * u
        elif curve != "linear":
            raise _NumpyOnly
        return c + (d - c) * u

    def _tpms(self, kind, Q):
        """lattice.tpms on tensors (period 1 in Q's units)."""
        torch = self.torch
        k = 2.0 * math.pi
        x, y, z = (k * Q[:, i] for i in range(3))
        sx, cx, sy, cy, sz, cz = torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y), torch.sin(z), torch.cos(z)
        if kind == "gyroid":
            f = sx * cy + sy * cz + sz * cx
            g = (cx * cy - sz * sx, cy * cz - sx * sy, cz * cx - sy * sz)
        elif kind == "diamond":
            f = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
            g = (cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz,
                 sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz,
                 sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz)
        else:                                 # schwarz-p
            f = cx + cy + cz
            g = (-sx, -sy, -sz)
        gn = torch.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) * k
        return f / torch.clamp(gn, min=0.3 * k)

    def _lattice(self, nd, P, memo):
        from .fields import _TPMS_KIND

        torch = self.torch
        s, r = nd.spec, nd.refs
        kind = _TPMS_KIND.get(s.get("type", "gyroid"))
        if kind is None or s.get("frame") or "aspect" in s:
            raise _NumpyOnly
        cell = self._val(r["cell"], P, memo) if "cell" in r else 8.0
        Q = P - self._t(s["origin"]) if "origin" in s else P
        if isinstance(r.get("cell"), str):
            if s.get("grading", "warp") != "warp":
                raise _NumpyOnly
            warp = self.g._resource(nd, lambda: self.g._warp(nd))
            cell = self._t(cell)
            idx = (Q - self._t(warp.lo)) / warp.h
            n = self._t(np.asarray(warp.dims, float) - 1)
            cl = torch.minimum(torch.clamp(idx, min=0.0), n)
            U = torch.stack([self._trilinear(("warp", nd.name, id(warp), i), u, cl) for i, u in enumerate(warp.u)],
                            dim=1)
            U = U + (idx - cl) * warp.h / cell[:, None]
            d = self._tpms(kind, U) * cell
        else:
            c = float(cell) if not torch.is_tensor(cell) else None
            if c is None:
                raise _NumpyOnly
            d = self._tpms(kind, Q / c) * c
        if "offset" in r:
            d = d - self._val(r["offset"], P, memo)
        if "thickness" in r:
            t = self._val(r["thickness"], P, memo)
        elif "density" in r:
            from .lattice import _density_table

            rho = torch.clamp(self._t(self._val(r["density"], P, memo)), 0.05, 0.95)
            tt, rr = _density_table(kind)
            t = self._interp(rho, rr, tt) * self._t(cell)
        else:
            t = 1.0
        form = s.get("form", "sheet")
        if form == "sheet":
            return torch.abs(d) - 0.5 * t
        if form == "network":
            return d - 0.5 * t
        raise _NumpyOnly

    def _transform(self, nd, P, memo):
        s = nd.spec
        Q = P
        if "translate" in s:
            Q = Q - self._t(s["translate"])
        if "rotate" in s:
            rot = s["rotate"]
            ax = rot.get("axis", "z")
            a = np.asarray({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax] if isinstance(ax, str) else ax,
                           float)
            a = a / np.linalg.norm(a)
            c = self._t(rot.get("center", (0, 0, 0)))
            th = -math.radians(float(rot["degrees"]))
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
            Q = (Q - c) @ self._t(R.T) + c
        scale = float(s.get("scale", 1.0))
        if scale != 1.0:
            Q = Q / scale
        return self._val(nd.refs["input"], Q, memo) * scale

    def _lib(self):
        """The expr functions on tensors (fields._compile_expr's `lib`)."""
        torch = self.torch
        T = self._t

        def red(fn, args):
            out = T(args[0])
            for a in args[1:]:
                out = fn(out, T(a))
            return out

        def smoothstep(e0, e1, x):
            e0, e1, x = T(e0), T(e1), T(x)
            d = torch.where(e1 == e0, torch.full_like(e1, 1e-12), e1 - e0)
            u = torch.clamp((x - e0) / d, 0.0, 1.0)
            return u * u * (3.0 - 2.0 * u)

        tiny = 1e-300 if self.dtype == torch.float64 else 1e-30
        funcs = {
            "abs": lambda a: torch.abs(T(a)), "sqrt": lambda a: torch.sqrt(torch.clamp(T(a), min=0.0)),
            "exp": lambda a: torch.exp(T(a)), "log": lambda a: torch.log(torch.clamp(T(a), min=tiny)),
            "sin": lambda a: torch.sin(T(a)), "cos": lambda a: torch.cos(T(a)), "tan": lambda a: torch.tan(T(a)),
            "floor": lambda a: torch.floor(T(a)), "ceil": lambda a: torch.ceil(T(a)),
            "sign": lambda a: torch.sign(T(a)),
            "min": lambda *a: red(torch.minimum, a), "max": lambda *a: red(torch.maximum, a),
            "clamp": lambda x, lo, hi: torch.minimum(torch.maximum(T(x), T(lo)), T(hi)),
            "lerp": lambda a, b, t: a + (b - a) * t,
            "smoothstep": smoothstep,
            "step": lambda edge, x: (T(x) >= T(edge)).to(self.dtype),
            "where": lambda c, a, b: torch.where(T(c) > 0, T(a), T(b)),
        }
        binops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                  ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: T(a) ** b if torch.is_tensor(a) or
                  torch.is_tensor(b) else a ** b,
                  ast.Mod: lambda a, b: torch.remainder(T(a), T(b))}
        cmpops = {ast.Lt: lambda a, b: T(a) < T(b), ast.LtE: lambda a, b: T(a) <= T(b),
                  ast.Gt: lambda a, b: T(a) > T(b), ast.GtE: lambda a, b: T(a) >= T(b)}
        return {"funcs": funcs, "binops": binops, "cmpops": cmpops,
                "where": lambda c, a, b: torch.where(T(c) > 0, T(a), T(b)),
                "float": lambda a: a.to(self.dtype)}


class _NumpyOnly(Exception):
    """This node runs on the numpy graph (see the module docstring)."""
