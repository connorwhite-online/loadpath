"""
Field graphs compiled to one numba kernel: every node of the graph, point by
point, in a single pass.

The numpy graph (FieldGraph._compute) runs each node as a handful of
whole-array passes, so a deep graph (the graded arm link: ~25 nodes, a
dozen elementwise ops each) streams hundreds of temporaries of a million
doubles through memory per chunk, and peval's four threads mostly wait on
memory bandwidth (they scaled ~2.4x on 4 cores). Here the graph becomes
Python source for one loop over the points, each node a few scalar lines,
compiled by numba (nogil, so peval's threads still split the points). The
source depends only on the graph's structure; every number (box corners,
remap ranges, blend radii) is read from a parameter array, so a tune that
moves parameters reuses the compiled kernel, and kernels are cached on disk
across processes (~/.cache/loadpath/fieldjit, LOADPATH_FIELDJIT_CACHE).

Compiled: constant, coordinate, the analytic shapes (box, cylinder, tube,
sphere, their unions, intersections, differences and shells), sampled
fields on a regular grid (a study's result, grid, image), remap (curves,
drawn curves, tables), expr, the smooth booleans, shell, offset, distance
(to a field, segment, plane or shape), transform, and TPMS lattices in the
cartesian frame (constant cell or the graded cell's warp, orient). Anything
else (an exact mesh distance, distance to points, a strut, conformal or
zoned lattice, a cross-faded grading, density-driven walls) stays numpy:
the node and what is under it are evaluated by the numpy graph first and
read by the kernel as an input array.

Same answer as numpy: every op is written in numpy's order of operations
(no fast-math, no fused multiply-add), the trilinear lookups as
scipy.ndimage.map_coordinates(order=1) sums them, so the values match bit
for bit (the arm link meshes to the same vertices, in double precision)
except where numpy hands a 3-vector dot product to BLAS (a cylinder on a
slanted axis, an orient or rotate, a coordinate along a vector) or uses its
own SIMD pow / exp / log (smootherstep, expr): there to ~1e-15 relative.
In single-precision mode (lattice.single_precision, which the mesher's grid
sampling and first relax rounds use) the lattice trig is libm's float32
instead of numpy's SIMD float32, which agree to ~1 ulp of float32 (~2e-6
mm on a 7 mm cell, the noise that mode accepts anyway): the meshes come out
with the same triangles, vertices apart by up to ~1e-3 mm where the relax
carries that noise along.

LOADPATH_FIELDJIT=0 (or numpy) turns it off; without numba it is off.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import os
import sys
import threading

import numpy as np

try:
    import numba as nb

    from . import meshops as _meshops  # noqa: F401 - caps numba's threads (LOADPATH_THREADS)
except ImportError:  # pragma: no cover
    nb = None

__all__ = ["available", "compile_graph", "CompiledField"]

_KINDS = {"gyroid": 0, "diamond": 1, "schwarz-p": 2}


def available():
    if nb is None:
        return False
    return os.environ.get("LOADPATH_FIELDJIT", "1").strip().lower() not in ("0", "numpy", "off", "false", "no")


# ---- scalar helpers, numpy's semantics -----------------------------------------------

if nb is not None:
    _jit = nb.njit(cache=True, nogil=True, error_model="numpy")

    @_jit
    def _mx(a, b):
        # np.maximum: NaN wins
        if a != a:
            return a
        return a if a >= b else b

    @_jit
    def _mn(a, b):
        if a != a:
            return a
        return a if a <= b else b

    @_jit
    def _clip(x, lo, hi):
        return _mn(_mx(x, lo), hi)

    @_jit
    def _smin(a, b, k):
        # fields._smin, one radius per point (0 = hard)
        if k <= 1e-9:
            return _mn(a, b)
        h = _clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return b * (1 - h) + a * h - k * h * (1 - h)

    @_jit
    def _sign(a):
        if a > 0:
            return 1.0
        if a < 0:
            return -1.0
        return a          # 0, -0 and NaN as np.sign

    @_jit
    def _interp(x, xs, ys, i0, n):
        # np.interp(x, xs[i0:i0+n], ys[i0:i0+n]) (numpy's compiled_base.c)
        if x != x:
            return x
        if x < xs[i0]:
            return ys[i0]
        last = i0 + n - 1
        if x > xs[last]:
            return ys[last]
        if x == xs[last]:
            return ys[last]
        lo = i0
        hi = last
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        j = lo
        if x == xs[j]:
            return ys[j]
        slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
        r = slope * (x - xs[j]) + ys[j]
        if r != r:
            r = slope * (x - xs[j + 1]) + ys[j + 1]
            if r != r and ys[j] == ys[j + 1]:
                r = ys[j]
        return r

    @_jit
    def _tri(G, x, y, z):
        # map_coordinates(G, idx, order=1, mode="nearest") in float64: the
        # eight corners summed a-outer, c-inner, each value * wa * wb * wc
        n0, n1, n2 = G.shape
        fx = math.floor(x)
        fy = math.floor(y)
        fz = math.floor(z)
        tx = x - fx
        ty = y - fy
        tz = z - fz
        i = int(fx)
        j = int(fy)
        k = int(fz)
        t = 0.0
        for a in range(2):
            ia = min(max(i + a, 0), n0 - 1)
            wa = 1.0 - tx if a == 0 else tx
            for b in range(2):
                jb = min(max(j + b, 0), n1 - 1)
                wb = 1.0 - ty if b == 0 else ty
                for c in range(2):
                    kc = min(max(k + c, 0), n2 - 1)
                    wc = 1.0 - tz if c == 0 else tz
                    t += np.float64(G[ia, jb, kc]) * wa * wb * wc
        return t

    @_jit
    def _tri32(G, x, y, z):
        # a float32 grid: map_coordinates hands back float32
        return np.float64(np.float32(_tri(G, x, y, z)))

    @_jit
    def _lerp_index(x, n):
        i0 = int(np.floor(x))
        if i0 > n - 2:
            i0 = n - 2 if n > 1 else 0
        w = x - i0
        i1 = i0 + 1 if n > 1 else 0
        return i0, i1, w

    @_jit
    def _warp1(u, a0, a1, wa, b0, b1, wb, c0, c1, wc):
        v00 = u[a0, b0, c0] * (1 - wc) + u[a0, b0, c1] * wc
        v01 = u[a0, b1, c0] * (1 - wc) + u[a0, b1, c1] * wc
        v10 = u[a1, b0, c0] * (1 - wc) + u[a1, b0, c1] * wc
        v11 = u[a1, b1, c0] * (1 - wc) + u[a1, b1, c1] * wc
        v0 = v00 * (1 - wb) + v01 * wb
        v1 = v10 * (1 - wb) + v11 * wb
        return v0 * (1 - wa) + v1 * wa

    @_jit
    def _warp(x, y, z, lo0, lo1, lo2, h, u0, u1, u2, c):
        # fieldops._warp for one point: the graded cell's lattice coordinates
        n0, n1, n2 = u0.shape
        ix = (x - lo0) / h
        iy = (y - lo1) / h
        iz = (z - lo2) / h
        cx = min(max(ix, 0.0), n0 - 1.0)
        cy = min(max(iy, 0.0), n1 - 1.0)
        cz = min(max(iz, 0.0), n2 - 1.0)
        a0, a1, wa = _lerp_index(cx, n0)
        b0, b1, wb = _lerp_index(cy, n1)
        c0, c1, wc = _lerp_index(cz, n2)
        U0 = _warp1(u0, a0, a1, wa, b0, b1, wb, c0, c1, wc) + (ix - cx) * h / c
        U1 = _warp1(u1, a0, a1, wa, b0, b1, wb, c0, c1, wc) + (iy - cy) * h / c
        U2 = _warp1(u2, a0, a1, wa, b0, b1, wb, c0, c1, wc) + (iz - cz) * h / c
        return U0, U1, U2

    @_jit
    def _tpms64(kind, px, py, pz, k):
        # fieldops._tpms for one point (= lattice.tpms in double)
        x = k * px
        y = k * py
        z = k * pz
        if kind == 2:
            f = np.cos(x) + np.cos(y) + np.cos(z)
            g0 = -np.sin(x) * k
            g1 = -np.sin(y) * k
            g2 = -np.sin(z) * k
            gn = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
            return f / max(gn, 0.3 * k)
        sx, cx, sy, cy, sz, cz = np.sin(x), np.cos(x), np.sin(y), np.cos(y), np.sin(z), np.cos(z)
        if kind == 0:
            f = sx * cy + sy * cz + sz * cx
            g0 = cx * cy - sz * sx
            g1 = cy * cz - sx * sy
            g2 = cz * cx - sy * sz
        else:
            f = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
            g0 = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
            g1 = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
            g2 = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
        gn = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2) * k
        return f / max(gn, 0.3 * k)

    @_jit
    def _tpms32(kind, px, py, pz, cell):
        # lattice.tpms under single_precision: phase reduced in double, trig in float32
        two_pi = np.float32(2.0 * np.pi)
        inv = 1.0 / cell
        u = px * inv
        u -= np.floor(u)
        x = np.float32(u) * two_pi
        u = py * inv
        u -= np.floor(u)
        y = np.float32(u) * two_pi
        u = pz * inv
        u -= np.floor(u)
        z = np.float32(u) * two_pi
        k = np.float32(2.0 * np.pi / cell)
        sx, cx, sy, cy, sz, cz = np.sin(x), np.cos(x), np.sin(y), np.cos(y), np.sin(z), np.cos(z)
        if kind == 0:
            f = sx * cy + sy * cz + sz * cx
            g0 = cx * cy - sz * sx
            g1 = cy * cz - sx * sy
            g2 = cz * cx - sy * sz
        else:
            f = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
            g0 = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
            g1 = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
            g2 = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
        gn = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2) * k
        lim = np.float32(0.3) * k
        return np.float64(f / (gn if gn >= lim else lim))

    @_jit
    def _tpms(kind, single, px, py, pz, cell):
        """lattice.tpms(kind, p, cell) for one point, as the numpy graph
        takes it (single precision inside lattice.single_precision)."""
        if single and kind != 2:
            return _tpms32(kind, px, py, pz, cell)
        return _tpms64(kind, px, py, pz, 2.0 * np.pi / cell)


# ---- code generation ---------------------------------------------------------------

class _Opaque(Exception):
    """The node runs on the numpy graph."""


_EXPR_FUNCS = {"abs", "sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceil", "sign", "min", "max",
               "clamp", "lerp", "smoothstep", "step", "where"}


class _Gen:
    def __init__(self, g):
        self.g = g
        self.C = []                  # parameters
        self.tables = []             # remap tables / drawn curves: concatenated
        self.lines = []
        self.vars = {}               # (node, frame) -> variable
        self.opaque = []             # root-frame nodes the numpy graph evaluates
        self.grids = []              # (array, dtype tag)
        self.grid_ids = {}
        self.frames = [("px", "py", "pz")]
        self.n = 0
        self._ok = {}

    # -- small builders --

    def c(self, v):
        self.C.append(float(v))
        return f"C[{len(self.C) - 1}]"

    def t(self, expr):
        name = f"t{self.n}"
        self.n += 1
        self.lines.append(f"{name} = {expr}")
        return name

    def grid(self, key, arr):
        if key not in self.grid_ids:
            a = np.ascontiguousarray(arr)
            if a.dtype not in (np.float32, np.float64):
                a = a.astype(np.float64)
            self.grid_ids[key] = len(self.grids)
            self.grids.append(a)
        return f"G{self.grid_ids[key]}"

    def table(self, xs, ys):
        i0 = len(self.tables)
        self.tables += [(float(a), float(b)) for a, b in zip(xs, ys)]
        return self.c(i0), len(xs)

    # -- which nodes compile --

    def supported(self, name, frame=0):
        """True when the node itself compiles (its inputs may still be opaque,
        in the root frame); a transform also needs everything under it."""
        key = (name, frame)
        if key in self._ok:
            return self._ok[key]
        self._ok[key] = False          # cycles are refused by the graph anyway
        nd = self.g.nodes[name]
        ok = self._local_ok(nd)
        if ok and nd.op == "transform":
            ok = self._all_ok(nd.refs["input"])
        self._ok[key] = ok
        return ok

    def _all_ok(self, ref, seen=None):
        if not isinstance(ref, str):
            return True
        seen = seen if seen is not None else set()
        if ref in seen:
            return True
        seen.add(ref)
        nd = self.g.nodes[ref]
        if not self._local_ok(nd):
            return False
        return all(self._all_ok(d, seen) for d in self.g.deps[ref])

    def _local_ok(self, nd):
        from .fields import _TPMS_KIND

        s, r, op = nd.spec, nd.refs, nd.op
        try:
            if op in ("constant", "remap", "union", "intersect", "subtract", "offset", "transform"):
                if op == "remap":
                    self._remap_kind(s)
                return True
            if op == "coordinate":
                return True
            if op == "shell":
                return s.get("side", "inside") in ("inside", "outside", "center")
            if op == "expr":
                self._expr_check(s["expr"])
                return True
            if op == "shape":
                if s["shape"].get("type") == "mesh":
                    return False
                return self._shape_ok(self._shape(nd))
            if op == "distance":
                if "input" in r or "segment" in s or "plane" in s:
                    return True
                if "shape" in s:
                    return self._shape_ok(self._shape(nd))
                return False
            if op in ("result", "grid", "image"):
                smp = self.g._resource(nd, lambda: self.g._samples(nd))
                return getattr(smp, "tree", 1) is None and np.ndim(getattr(smp, "grid", None)) == 3
            if op == "lattice":
                from .struts import is_strut

                kind = s.get("type", "gyroid")
                if _TPMS_KIND.get(kind) is None or is_strut(kind):
                    return False
                if isinstance(s.get("orient"), dict) and "zones" in s["orient"]:
                    return False
                if s.get("frame") or "aspect" in s or "density" in r:
                    return False
                if s.get("form", "sheet") not in ("sheet", "network", "_plus", "_minus"):
                    return False
                graded = isinstance(r.get("cell"), str)
                if graded and s.get("grading", "warp") != "warp":
                    return False
                from .fields import lattice_orientation

                lattice_orientation(nd)      # raises on what numpy refuses
                return True
        except Exception:
            return False
        return False

    def _shape(self, nd):
        from .shapes import shape_from_spec

        return self.g._resource(nd, lambda: shape_from_spec(nd.spec["shape"], self.g.base_dir))

    def _shape_ok(self, shp):
        from .shapes import Boolean, Box, Cylinder, Shell, Sphere, Union

        if isinstance(shp, (Box, Cylinder, Sphere)):
            return True
        if isinstance(shp, Union):
            return all(self._shape_ok(x) for x in shp.shapes)
        if isinstance(shp, Boolean):
            return all(self._shape_ok(x) for x in shp.parts)
        if type(shp) is Shell:
            return self._shape_ok(shp.outer) and self._shape_ok(shp.inner) and (
                shp.clip is None or self._shape_ok(shp.clip))
        return False

    @staticmethod
    def _remap_kind(s):
        if len(s.get("from", ())) > 2 or len(s.get("to", ())) > 2:
            xs, ys = np.asarray(s["from"], float), np.asarray(s["to"], float)
            if len(xs) != len(ys) or np.any(np.diff(xs) <= 0):
                raise ValueError
            return "table"
        a, b = (float(u) for u in s.get("from", (0.0, 1.0)))
        if a == b:
            raise ValueError
        curve = s.get("curve", "linear")
        if isinstance(curve, (list, tuple)):
            pts = np.asarray(curve, float)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or np.any(np.diff(pts[:, 0]) <= 0):
                raise ValueError
            return "points"
        if curve not in ("linear", "smoothstep", "smootherstep", "sqrt", "square"):
            raise ValueError
        return curve

    # -- emission --

    def val(self, ref, frame):
        if isinstance(ref, float):
            return self.c(ref)
        return self.node(ref, frame)

    def node(self, name, frame=0):
        key = (name, frame)
        if key in self.vars:
            return self.vars[key]
        if not self.supported(name, frame):
            if frame != 0:
                raise AssertionError("an opaque node inside a transform")   # _all_ok keeps this out
            j = len(self.opaque)
            self.opaque.append(name)
            v = self.t(f"X[{j}, i]")
        else:
            v = self._emit(self.g.nodes[name], frame)
        self.vars[key] = v
        return v

    def _emit(self, nd, f):
        s, r, op = nd.spec, nd.refs, nd.op
        x, y, z = self.frames[f]
        v = lambda key: self.val(r[key], f)  # noqa: E731
        if op == "constant":
            return self.c(s["value"])
        if op == "coordinate":
            ax = s.get("axis", "z")
            if isinstance(ax, str):
                return (x, y, z)["xyz".index(ax)]
            a = np.asarray(ax, float)
            a = a / np.linalg.norm(a)
            o = np.asarray(s.get("origin", (0, 0, 0)), float)
            return self.t(f"({x} - {self.c(o[0])}) * {self.c(a[0])} + ({y} - {self.c(o[1])}) * {self.c(a[1])}"
                          f" + ({z} - {self.c(o[2])}) * {self.c(a[2])}")
        if op == "shape":
            return self._sdf(self._shape(nd), f)
        if op in ("result", "grid", "image"):
            smp = self.g._resource(nd, lambda: self.g._samples(nd))
            G = self.grid(("samples", nd.name), smp.grid)
            o, p = smp.origin, smp.pitch
            fn = "_tri32" if smp.grid.dtype == np.float32 else "_tri"
            return self.t(f"{fn}({G}, ({x} - {self.c(o[0])}) / {self.c(p[0])}, ({y} - {self.c(o[1])}) / "
                          f"{self.c(p[1])}, ({z} - {self.c(o[2])}) / {self.c(p[2])})")
        if op == "remap":
            return self._remap(v("input"), s)
        if op == "expr":
            env = {"x": x, "y": y, "z": z}
            for k, ref in r["vars"].items():
                env[k] = self.val(ref, f)
            return self.t(self._expr(s["expr"], env))
        if op in ("union", "intersect", "subtract"):
            vals = [self.val(a, f) for a in r["inputs"]]
            k = v("blend") if "blend" in r else "0.0"
            out = vals[0]
            for b in vals[1:]:
                if op == "union":
                    out = self.t(f"_smin({out}, {b}, {k})")
                elif op == "intersect":
                    out = self.t(f"-_smin(-{out}, -{b}, {k})")
                else:
                    out = self.t(f"-_smin(-{out}, {b}, {k})")
            return out
        if op == "shell":
            a, t = v("input"), v("thickness")
            side = s.get("side", "inside")
            if side == "inside":
                return self.t(f"_mx({a}, -({a} + {t}))")
            if side == "outside":
                return self.t(f"_mx(-{a}, {a} - {t})")
            return self.t(f"abs({a}) - 0.5 * {t}")
        if op == "offset":
            return self.t(f"{v('input')} - {v('distance')}")
        if op == "distance":
            return self._distance(nd, f)
        if op == "lattice":
            return self._lattice(nd, f)
        if op == "transform":
            return self._transform(nd, f)
        raise AssertionError(op)

    def _remap(self, xv, s):
        kind = self._remap_kind(s)
        if kind == "table":
            i0, n = self.table(s["from"], s["to"])
            return self.t(f"_interp({xv}, TX, TY, int({i0}), {n})")
        a, b = (float(u) for u in s.get("from", (0.0, 1.0)))
        c, d = (float(u) for u in s.get("to", (0.0, 1.0)))
        u = self.t(f"({xv} - {self.c(a)}) / {self.c(b - a)}")
        if s.get("clamp", True):
            u = self.t(f"_clip({u}, 0.0, 1.0)")
        cc, dc = self.c(c), self.c(d - c)
        if kind == "points":
            pts = np.asarray(s["curve"], float)
            i0, n = self.table(pts[:, 0], pts[:, 1])
            w = self.t(f"_interp(_clip({u}, {self.c(pts[0, 0])}, {self.c(pts[-1, 0])}), TX, TY, int({i0}), {n})")
            return self.t(f"{cc} + {dc} * {w}")
        if kind == "smoothstep":
            uc = self.t(f"_clip({u}, 0.0, 1.0)")
            u = self.t(f"{uc} * {uc} * (3 - 2 * {uc})")
        elif kind == "smootherstep":
            uc = self.t(f"_clip({u}, 0.0, 1.0)")
            u = self.t(f"{uc} ** 3.0 * ({uc} * (6 * {uc} - 15) + 10)")
        elif kind == "sqrt":
            u = self.t(f"np.sqrt(_mx({u}, 0.0))")
        elif kind == "square":
            u = self.t(f"{u} * {u}")
        return self.t(f"{cc} + {dc} * {u}")

    def _sdf(self, shp, f):
        from .shapes import Boolean, Box, Cylinder, Shell, Sphere, Union

        x, y, z = self.frames[f]
        if isinstance(shp, Box):
            c, h = 0.5 * (shp.lo + shp.hi), 0.5 * (shp.hi - shp.lo)
            q = [self.t(f"abs({p} - {self.c(c[i])}) - {self.c(h[i])}") for i, p in enumerate((x, y, z))]
            a, b, d = (self.t(f"_mx({qi}, 0.0)") for qi in q)
            return self.t(f"np.sqrt({a} * {a} + {b} * {b} + {d} * {d}) + _mn(_mx(_mx({q[0]}, {q[1]}), {q[2]}), 0.0)")
        if isinstance(shp, Sphere):
            ce = shp.center
            d = [self.t(f"{p} - {self.c(ce[i])}") for i, p in enumerate((x, y, z))]
            return self.t(f"np.sqrt({d[0]} * {d[0]} + {d[1]} * {d[1]} + {d[2]} * {d[2]}) - {self.c(shp.r)}")
        if isinstance(shp, Cylinder):
            P = (x, y, z)
            ax = shp.axis
            k = int(np.argmax(np.abs(ax)))
            if abs(ax[k]) == 1.0:
                i, j = [w for w in range(3) if w != k]
                t = self.t(f"({P[k]} - {self.c(shp.p0[k])}) * {self.c(ax[k])}")
                u = self.t(f"{P[i]} - {self.c(shp.p0[i])}")
                w = self.t(f"{P[j]} - {self.c(shp.p0[j])}")
                radial = self.t(f"np.sqrt({u} * {u} + {w} * {w})")
            else:
                d = [self.t(f"{P[a]} - {self.c(shp.p0[a])}") for a in range(3)]
                t = self.t(f"{d[0]} * {self.c(ax[0])} + {d[1]} * {self.c(ax[1])} + {d[2]} * {self.c(ax[2])}")
                radial = self.t(f"np.sqrt(_mx({d[0]} * {d[0]} + {d[1]} * {d[1]} + {d[2]} * {d[2]} - {t} * {t}, 0.0))")
            half = 0.5 * shp.length
            if shp.r_inner > 0:
                mid = 0.5 * (shp.r + shp.r_inner)
                rq = self.t(f"abs({radial} - {self.c(mid)}) - {self.c(0.5 * (shp.r - shp.r_inner))}")
            else:
                rq = self.t(f"{radial} - {self.c(shp.r)}")
            hc = self.c(half)
            aq = self.t(f"abs({t} - {hc}) - {hc}")
            a, b = self.t(f"_mx({rq}, 0.0)"), self.t(f"_mx({aq}, 0.0)")
            return self.t(f"np.sqrt({a} * {a} + {b} * {b}) + _mn(_mx({rq}, {aq}), 0.0)")
        if isinstance(shp, Union):
            out = self._sdf(shp.shapes[0], f)
            for sub in shp.shapes[1:]:
                out = self.t(f"_mn({out}, {self._sdf(sub, f)})")
            return out
        if isinstance(shp, Boolean):
            out = self._sdf(shp.parts[0], f)
            for sub in shp.parts[1:]:
                b = self._sdf(sub, f)
                out = self.t(f"_mx({out}, {b})" if shp.kind == "intersect" else f"_mx({out}, -{b})")
            return out
        if isinstance(shp, Shell):
            d = self.t(f"_mx({self._sdf(shp.outer, f)}, -{self._sdf(shp.inner, f)})")
            return d if shp.clip is None else self.t(f"_mx({d}, {self._sdf(shp.clip, f)})")
        raise AssertionError(type(shp))

    def _distance(self, nd, f):
        s, r = nd.spec, nd.refs
        x, y, z = self.frames[f]
        if "input" in r:
            a = self.val(r["input"], f)
            return a if s.get("signed") else self.t(f"abs({a})")
        if "segment" in s:
            a, b = (np.asarray(q, float) for q in s["segment"])
            ab = b - a
            d = [self.t(f"{p} - {self.c(a[i])}") for i, p in enumerate((x, y, z))]
            den = self.c(max(float(ab @ ab), 1e-300))
            t = self.t(f"_clip(({d[0]} * {self.c(ab[0])} + {d[1]} * {self.c(ab[1])} + {d[2]} * {self.c(ab[2])})"
                       f" / {den}, 0.0, 1.0)")
            e = [self.t(f"{p} - ({self.c(a[i])} + {t} * {self.c(ab[i])})") for i, p in enumerate((x, y, z))]
            return self.t(f"np.sqrt({e[0]} * {e[0]} + {e[1]} * {e[1]} + {e[2]} * {e[2]})")
        if "plane" in s:
            n = np.asarray(s["plane"]["normal"], float)
            n = n / np.linalg.norm(n)
            o = np.asarray(s["plane"]["point"], float)
            return self.t(f"abs(({x} - {self.c(o[0])}) * {self.c(n[0])} + ({y} - {self.c(o[1])}) * {self.c(n[1])}"
                          f" + ({z} - {self.c(o[2])}) * {self.c(n[2])})")
        return self.t(f"abs({self._sdf(self._shape(nd), f)})")

    def _lattice(self, nd, f):
        from .fields import _TPMS_KIND, lattice_orientation

        s, r = nd.spec, nd.refs
        x, y, z = self.frames[f]
        kind = _KINDS[_TPMS_KIND[s.get("type", "gyroid")]]
        cell = self.val(r["cell"], f) if "cell" in r else self.c(8.0)
        q = [x, y, z]
        if "origin" in s:
            o = np.asarray(s["origin"], float)
            q = [self.t(f"{p} - {self.c(o[i])}") for i, p in enumerate(q)]
        R = lattice_orientation(nd)
        if R is not None:
            q = [self.t(" + ".join(f"{q[a]} * {self.c(R[a, b])}" for a in range(3))) for b in range(3)]
        if isinstance(r.get("cell"), str):
            warp = self.g._resource(nd, lambda: self.g._warp(nd))
            us = warp._grids()
            G = [self.grid(("warp", nd.name, i), us[i]) for i in range(3)]
            u = [f"w{self.n}_{i}" for i in range(3)]
            self.n += 1
            self.lines.append(f"{u[0]}, {u[1]}, {u[2]} = _warp({q[0]}, {q[1]}, {q[2]}, {self.c(warp.lo[0])}, "
                              f"{self.c(warp.lo[1])}, {self.c(warp.lo[2])}, {self.c(warp.h)}, {G[0]}, {G[1]}, {G[2]}, "
                              f"{cell})")
            d = self.t(f"_tpms({kind}, single, {u[0]}, {u[1]}, {u[2]}, 1.0) * {cell}")
        else:
            d = self.t(f"_tpms({kind}, single, {q[0]}, {q[1]}, {q[2]}, {cell})")
        if "offset" in r:
            d = self.t(f"{d} - {self.val(r['offset'], f)}")
        t = self.val(r["thickness"], f) if "thickness" in r else "1.0"
        form = s.get("form", "sheet")
        if form == "sheet":
            return self.t(f"abs({d}) - 0.5 * {t}")
        if form in ("network", "_plus"):
            return self.t(f"{d} - 0.5 * {t}")
        return self.t(f"-{d} - 0.5 * {t}")

    def _transform(self, nd, f):
        s = nd.spec
        q = list(self.frames[f])
        if "translate" in s:
            tr = np.asarray(s["translate"], float)
            q = [self.t(f"{p} - {self.c(tr[i])}") for i, p in enumerate(q)]
        if "rotate" in s:
            rot = s["rotate"]
            ax = rot.get("axis", "z")
            a = np.asarray({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax] if isinstance(ax, str) else ax,
                           float)
            a = a / np.linalg.norm(a)
            c = np.asarray(rot.get("center", (0, 0, 0)), float)
            th = -math.radians(float(rot["degrees"]))
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
            d = [self.t(f"{p} - {self.c(c[i])}") for i, p in enumerate(q)]
            q = [self.t(" + ".join(f"{d[a]} * {self.c(R[b, a])}" for a in range(3)) + f" + {self.c(c[b])}")
                 for b in range(3)]
        scale = float(s.get("scale", 1.0))
        if scale != 1.0:
            sc = self.c(scale)
            q = [self.t(f"{p} / {sc}") for p in q]
        self.frames.append(tuple(q))
        inner = self.node(nd.refs["input"], len(self.frames) - 1)
        return self.t(f"{inner} * {self.c(scale)}")

    # -- expr --

    def _expr_check(self, text):
        self._expr(text, None)

    def _expr(self, text, env):
        tree = ast.parse(text, mode="eval")
        ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**", ast.Mod: "%"}
        cmps = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}

        def b(n):
            if isinstance(n, ast.Expression):
                return b(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                return repr(float(n.value))
            if isinstance(n, ast.Name):
                if n.id == "pi":
                    return repr(math.pi)
                if n.id == "e":
                    return repr(math.e)
                if env is None:
                    return "0.0"
                return env[n.id]
            if isinstance(n, ast.BinOp) and type(n.op) in ops:
                return f"({b(n.left)} {ops[type(n.op)]} {b(n.right)})"
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                return f"(-{b(n.operand)})" if isinstance(n.op, ast.USub) else b(n.operand)
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in cmps:
                return f"(1.0 if {b(n.left)} {cmps[type(n.ops[0])]} {b(n.comparators[0])} else 0.0)"
            if isinstance(n, ast.IfExp):
                return f"({b(n.body)} if {b(n.test)} > 0 else {b(n.orelse)})"
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _EXPR_FUNCS \
                    and not n.keywords:
                a = [b(q) for q in n.args]
                fn = n.func.id
                one = {"abs": "abs({})", "sqrt": "np.sqrt(_mx({}, 0.0))", "exp": "np.exp({})",
                       "log": "np.log(_mx({}, 1e-300))", "sin": "np.sin({})", "cos": "np.cos({})",
                       "tan": "np.tan({})", "floor": "np.floor({})", "ceil": "np.ceil({})", "sign": "_sign({})"}
                if fn in one:
                    if len(a) != 1:
                        raise ValueError
                    return one[fn].format(a[0])
                if fn in ("min", "max"):
                    if not a:
                        raise ValueError
                    out = a[0]
                    for q in a[1:]:
                        out = f"{'_mn' if fn == 'min' else '_mx'}({out}, {q})"
                    return out
                if fn == "clamp" and len(a) == 3:
                    return f"_clip({a[0]}, {a[1]}, {a[2]})"
                if fn == "lerp" and len(a) == 3:
                    return f"({a[0]} + ({a[1]} - {a[0]}) * {a[2]})"
                if fn == "smoothstep" and len(a) == 3:
                    u = f"_clip(({a[2]} - {a[0]}) / (1e-12 if {a[1]} == {a[0]} else {a[1]} - {a[0]}), 0.0, 1.0)"
                    return f"_sstep({u})"
                if fn == "step" and len(a) == 2:
                    return f"(1.0 if {a[1]} >= {a[0]} else 0.0)"
                if fn == "where" and len(a) == 3:
                    return f"({a[1]} if {a[0]} > 0 else {a[2]})"
            raise ValueError(f"expr {text!r} does not compile")

        return b(tree)

    # -- the kernel --

    def source(self, out):
        args = ", ".join(["P", "C", "TX", "TY", "X", "single", "out"] + [f"G{i}" for i in range(len(self.grids))])
        body = "\n".join("        " + ln for ln in self.lines)
        return (
            "import math\nimport numpy as np\nimport numba as nb\n"
            "from loadpath.fieldjit import _mx, _mn, _clip, _smin, _sign, _interp, _tri, _tri32, _warp, _tpms, _sstep\n\n"
            "@nb.njit(cache=True, nogil=True, error_model='numpy')\n"
            f"def kernel({args}):\n"
            "    for i in range(P.shape[0]):\n"
            "        px = P[i, 0]\n        py = P[i, 1]\n        pz = P[i, 2]\n"
            f"{body}\n"
            f"        out[i] = {out}\n")


if nb is not None:
    @_jit
    def _sstep(u):
        return u * u * (3.0 - 2.0 * u)


# ---- loading kernels ----------------------------------------------------------------

_KERNELS = {}
_LOCK = threading.Lock()


def _cache_dir():
    d = os.environ.get("LOADPATH_FIELDJIT_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "loadpath",
                                                                  "fieldjit")
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".w")
        with open(probe, "w"):
            pass
        return d
    except OSError:
        return None


def _kernel(src):
    """The compiled kernel for this source (in-process, then on disk)."""
    key = hashlib.sha1(src.encode()).hexdigest()[:20]
    with _LOCK:
        if key in _KERNELS:
            return _KERNELS[key]
        d = _cache_dir()
        name = f"loadpath_fieldjit_{key}"
        if d is not None:
            path = os.path.join(d, name + ".py")
            if not os.path.exists(path):
                tmp = path + f".{os.getpid()}.tmp"
                with open(tmp, "w") as fh:
                    fh.write(src)
                os.replace(tmp, path)
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            k = mod.kernel
        else:                                     # nowhere to cache: compile in memory
            ns = {}
            exec(compile(src.replace("cache=True", "cache=False"), name, "exec"), ns)
            k = ns["kernel"]
        _KERNELS[key] = k
        return k


class CompiledField:
    """graph.evaluate(P, name) as one compiled pass (plus the numpy graph for
    the nodes that do not compile, see `opaque`)."""

    def __init__(self, g, name):
        gen = _Gen(g)
        out = gen.node(name)
        self.g = g
        self.name = name
        self.opaque = gen.opaque
        self.C = np.asarray(gen.C, np.float64)
        tb = np.asarray(gen.tables, np.float64).reshape(-1, 2)
        self.TX, self.TY = np.ascontiguousarray(tb[:, 0]), np.ascontiguousarray(tb[:, 1])
        self.grids = gen.grids
        self.source = gen.source(out)
        self.kernel = _kernel(self.source)
        self.nodes = len(gen.vars)

    def __call__(self, P):
        from .lattice import _SINGLE

        P = np.ascontiguousarray(np.asarray(P, np.float64).reshape(-1, 3))
        n = len(P)
        X = np.empty((len(self.opaque), n))
        memo = {}
        for j, name in enumerate(self.opaque):
            X[j] = np.broadcast_to(np.asarray(self.g._eval(name, P, memo), float), (n,))
        out = np.empty(n)
        if n:
            self.kernel(P, self.C, self.TX, self.TY, X, bool(_SINGLE[0]), out, *self.grids)
        return out


def compile_graph(g, name):
    """CompiledField for graph g's node `name`, or None when it would only
    wrap the numpy graph (the node itself does not compile) or compiling
    fails (the reason is kept in g._jit_errors)."""
    gen = _Gen(g)
    if not gen.supported(name):
        return None
    try:
        return CompiledField(g, name)
    except Exception as exc:  # pragma: no cover - a bug here must not stop a mesh
        g._jit_errors = getattr(g, "_jit_errors", []) + [f"{name}: {type(exc).__name__}: {exc}"]
        return None
