"""
Manufacturing constraints for parts that are not printed, and the size and
pattern controls that go with them: the rest of what Fusion 360's
generative design offers beside additive.

- **Milling** (`"milling"`): every void must be reachable by a cutter coming
  straight in along one of the access directions. 3-axis mills from one
  side (+z by default), 2.5-axis does the same with flat pocket floors at a
  fixed step-down, and 5-axis is approximated as 3+2 positional machining
  from the six axis directions. `toolDiameter` keeps every slot and inside
  corner at least as wide as the cutter.
- **Casting / molding** (`"casting"`): no undercuts along the draw
  direction. A two-sided mold (the default) pulls halves both ways along the
  draw, so every line parallel to it crosses the part once; a one-sided mold
  pulls one way off a flat parting plane.
- **Circular pattern** (`"pattern"`): N identical sectors round an axis (a
  turret, a hub, a wheel), as Fusion's circular symmetry.
- **Maximum member** (`"maxMember"`): no member thicker than this, the
  counterpart of minMember.

Access is one operator for milling and casting. A cell is blocked from
direction d when anything solid lies between it and the outside along d;
it must then be solid itself, because neither a cutter nor a mold half can
get there. With several access directions a cell has to be blocked from
all of them:

    accessible_i = min over d of  max(x_j : j on the ray from i along d)

which is a cumulative max per direction (exact, with the gradient routed to
the cell that set the max) and a min across directions. It runs on the
projected densities like the AM filter (am.py), so material that would
leave an undercut is credited as filled, and the optimizer stops putting it
there. Casting two-sided along z is access from +z and -z; 3-axis milling
from the top is access from +z, the same as a one-sided mold.

The cutter's radius and the 2.5-axis floors are applied exactly to the
delivered design (`machinable`): the void a flat end mill of that radius can
sweep from each side, with each pocket's depth rounded up to the step-down,
and everything else in the design space filled. It only adds material, so
it can only lower stress; the final analysis runs on the filled part.

The maximum member is the local volume constraint of Wu, Aage, Westermann &
Sigmund (2018, "Infill optimization for additive manufacturing", IEEE TVCG
24): the solid share of every ball round a design cell stays under ALPHA. A
slab thicker than the limit fills the ball past it. The ball is sized so a
flat slab exactly maxMember thick reads ALPHA (a ball of radius R cut by a
centred slab of half-thickness h holds 1.5 u - 0.5 u^3 of solid, u = h/R);
a round strut of the same diameter reads less, so for struts the limit is
looser. Instead of one aggregated constraint each violating ball adds a
penalty to the optimality-criteria update, as a cost on top of volume,
whose weight grows while any ball is over: thick members hollow out or split
in two, and the volume they free goes where the stiffness is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

__all__ = ["Access", "Pattern", "AccessFilter", "MaxMember", "parse_milling", "parse_casting",
           "parse_pattern", "tie_groups", "machinable", "member_thickness", "report", "AXIS_DIRECTIONS"]

AXIS_DIRECTIONS = [(a, s) for a in range(3) for s in (1, -1)]
# Solid share of a ball allowed by maxMember.
ALPHA = 0.85
# Ball radius / maxMember for which a slab maxMember thick reads ALPHA.
_BALL = 0.754


def _label(d):
    return ("+" if d[1] > 0 else "-") + "xyz"[d[0]]


def _axis_direction(v, field_name):
    s = str(v).strip().lower()
    if len(s) == 1:
        s = "+" + s
    if len(s) != 2 or s[0] not in "+-" or s[1] not in "xyz":
        raise ValueError(f"{field_name} entries are axis directions like '+z' or '-x' "
                         f"(tilted directions are not supported yet), got {v!r}")
    return "xyz".index(s[1]), (1 if s[0] == "+" else -1)


@dataclass
class Access:
    """Which way a cutter or mold half reaches the part."""
    kind: str                         # "milling" | "casting"
    directions: list                  # (axis, sign): the side the cutter comes from / a mold half pulls toward
    axes: float | None = None         # milling: 2.5, 3 or 5
    tool_radius: float | None = None  # mm
    step: float | None = None         # mm between 2.5-axis pocket floors
    draw: tuple | None = None         # casting

    def summary(self) -> dict:
        out = {"kind": self.kind, "directions": [_label(d) for d in self.directions]}
        if self.kind == "milling":
            out["axes"] = self.axes
            if self.tool_radius:
                out["toolDiameter"] = 2.0 * self.tool_radius
            if self.step:
                out["stepDown"] = self.step
        else:
            out["draw"] = _label(self.draw)
            out["sides"] = len(self.directions)
        return out


@dataclass
class Pattern:
    """N identical sectors round an axis through `center`."""
    axis: int
    count: int
    center: np.ndarray | None = None

    def summary(self) -> dict:
        out = {"type": "circular", "axis": "xyz"[self.axis], "count": self.count}
        if self.center is not None:
            out["center"] = [float(c) for c in self.center]
        return out


def parse_milling(v) -> Access | None:
    """`"milling": {"axes": 3, "directions": ["+z"], "toolDiameter": 6, "stepDown": 2}`."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("milling is an object: {axes: 2.5 | 3 | 5, directions: ['+z', ...], toolDiameter: mm}")
    axes = float(v.get("axes", 3))
    if axes not in (2.5, 3.0, 5.0):
        raise ValueError("milling.axes is 2.5, 3 or 5")
    dirs = v.get("directions")
    if dirs is None:
        dirs = [_label(d) for d in AXIS_DIRECTIONS] if axes == 5 else ["+z"]
    if isinstance(dirs, str):
        dirs = [dirs]
    if not dirs:
        raise ValueError("milling.directions needs at least one direction the cutter comes from")
    parsed = []
    for d in dirs:
        p = _axis_direction(d, "milling.directions")
        if p not in parsed:
            parsed.append(p)
    tool = v.get("toolDiameter")
    if tool is not None and not (isinstance(tool, (int, float)) and tool > 0):
        raise ValueError("milling.toolDiameter is the smallest cutter's diameter in mm (> 0)")
    step = v.get("stepDown")
    if step is not None and not (isinstance(step, (int, float)) and step > 0):
        raise ValueError("milling.stepDown is the spacing of 2.5-axis pocket floors in mm (> 0)")
    if step is not None and axes != 2.5:
        raise ValueError("milling.stepDown is for 2.5-axis milling (axes: 2.5)")
    if axes == 2.5 and step is None:
        step = 2.0
    return Access("milling", parsed, axes=axes, tool_radius=0.5 * float(tool) if tool else None,
                  step=float(step) if step else None)


def parse_casting(v) -> Access | None:
    """`"casting": {"draw": "+z", "sides": 2}`; a string is the draw direction."""
    if v is None:
        return None
    if isinstance(v, str):
        v = {"draw": v}
    if not isinstance(v, dict) or "draw" not in v:
        raise ValueError("casting is {draw: '+z', sides: 1 | 2}: the direction the mold opens")
    draw = _axis_direction(v["draw"], "casting.draw")
    sides = v.get("sides", 2)
    if sides not in (1, 2):
        raise ValueError("casting.sides is 2 (mold halves pull both ways along the draw) or 1 (one half pulls "
                         "toward draw off a flat parting plane)")
    dirs = [draw] if sides == 1 else [draw, (draw[0], -draw[1])]
    return Access("casting", dirs, draw=draw)


def parse_pattern(v) -> Pattern | None:
    """`"pattern": {"type": "circular", "axis": "z", "count": 6, "center": [x, y, z]}`."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("pattern is {type: 'circular', axis: 'z', count: N, center: [x, y, z]}")
    if v.get("type", "circular") != "circular":
        raise ValueError("pattern.type must be 'circular'")
    a = str(v.get("axis", "z")).strip().lower().lstrip("+-")
    if a not in ("x", "y", "z"):
        raise ValueError("pattern.axis is 'x', 'y' or 'z'")
    n = v.get("count")
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("pattern.count is the number of identical sectors (an integer, at least 2)")
    c = v.get("center")
    if c is not None:
        c = np.asarray(c, float)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            raise ValueError("pattern.center is a point [x, y, z] on the axis")
    return Pattern("xyz".index(a), n, c)


# --- access ---------------------------------------------------------------

def _cummax(f, axis, sign, track):
    """Max over each cell and everything beyond it toward the `sign` end of
    `axis` (where the cutter comes from), and the axis coordinate of the
    cell that set it."""
    g = np.moveaxis(f, axis, -1)
    if sign > 0:
        g = g[..., ::-1]
    m = np.maximum.accumulate(g, axis=-1)
    idx = None
    if track:
        n = g.shape[-1]
        pos = np.where(g >= m, np.arange(n), 0)
        idx = np.maximum.accumulate(pos, axis=-1)
        if sign > 0:
            idx = (n - 1 - idx)[..., ::-1]
        idx = np.moveaxis(idx, -1, axis)
    if sign > 0:
        m = m[..., ::-1]
    return np.moveaxis(m, -1, axis), idx


class AccessFilter:
    """Same interface as am.AMFilter: forward/backward on full grids."""

    def __init__(self, dims, directions, track: bool = True):
        self.dims = tuple(int(d) for d in dims)
        self.directions = list(directions)
        self.track = track
        self._state = None

    def forward(self, field: np.ndarray) -> np.ndarray:
        vals, idxs = [], []
        for a, s in self.directions:
            m, idx = _cummax(field, a, s, self.track)
            vals.append(m)
            idxs.append(idx)
        if len(vals) == 1:
            out, choice = vals[0], None
        else:
            stack = np.stack(vals)
            choice = np.argmin(stack, axis=0)
            out = np.take_along_axis(stack, choice[None], axis=0)[0]
        self._state = (idxs, choice) if self.track else None
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("AccessFilter.backward needs a tracked forward first")
        idxs, choice = self._state
        flat = np.arange(int(np.prod(self.dims))).reshape(self.dims)
        strides = [self.dims[1] * self.dims[2], self.dims[2], 1]
        coords = np.indices(self.dims)
        out = np.zeros(flat.size)
        for k, (a, _) in enumerate(self.directions):
            g = grad if choice is None else np.where(choice == k, grad, 0.0)
            target = flat + (idxs[k] - coords[a]) * strides[a]
            out += np.bincount(target.ravel(), weights=g.ravel(), minlength=flat.size)
        return out.reshape(self.dims)


def _disk(r_cells: float, axis: int) -> np.ndarray:
    k = int(np.floor(r_cells + 1e-9))
    i, j = np.mgrid[-k:k + 1, -k:k + 1]
    d = (i * i + j * j) <= r_cells * r_cells + 1e-9
    return np.expand_dims(d, axis)


def machinable(solid: np.ndarray, access: Access, pitch: float) -> np.ndarray:
    """The void a cutter (or mold half) can reach: exact, on a bool grid.
    Returns the bool grid of cells it cannot reach; with a tool radius the
    cutter is a flat end mill of that radius entering along each direction,
    and with a step-down each pocket's depth is rounded up to it (floors on
    a grid of levels from the stock face)."""
    r = (access.tool_radius or 0.0) / pitch
    pad = int(np.ceil(r)) + 1
    s = np.pad(solid, pad, constant_values=False)
    reach_any = np.zeros_like(s)
    step = (access.step / pitch) if access.step else None
    for a, sign in access.directions:
        void = ~s
        disk = _disk(r, a) if r >= 1.0 else None
        centre_ok = ndimage.binary_erosion(void, structure=disk, border_value=1) if disk is not None else void
        g = np.moveaxis(centre_ok, a, -1)
        if sign > 0:
            g = g[..., ::-1]
        reach = np.logical_and.accumulate(g, axis=-1)
        if step:
            # depth reached inside the stock, rounded up to a floor level
            # (so less is cut); a through-cut stays through
            n = reach.shape[-1]
            depth = reach.sum(axis=-1)
            level = np.floor(np.floor(np.maximum(depth - pad, 0) / step + 1e-9) * step + 1e-9).astype(int) + pad
            depth = np.where((depth >= n) | (depth < pad), depth, level)
            reach = np.arange(n) < depth[..., None]
        if sign > 0:
            reach = reach[..., ::-1]
        reach = np.moveaxis(reach, -1, a)
        cut = ndimage.binary_dilation(reach, structure=disk) if disk is not None else reach
        reach_any |= cut
    blocked = ~reach_any
    return blocked[pad:-pad, pad:-pad, pad:-pad]


# --- pattern and ties -----------------------------------------------------

def tie_groups(domain, extrusion: int | None = None, pattern: Pattern | None = None):
    """Label per element: cells with one label share one design value (the
    columns of an extruded part, the sectors of a circular pattern, or
    both). None when nothing is tied."""
    mesh = domain.mesh
    if extrusion is None and pattern is None:
        return None
    if pattern is None:
        a = extrusion
        others = [i for i in range(3) if i != a]
        ij = mesh.elem_ijk[:, others]
        _, columns = np.unique(ij[:, 0] * (domain.dims[others[1]] + 1) + ij[:, 1], return_inverse=True)
        return columns
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = mesh.n_elem
    rows, cols = [], []
    if extrusion is not None:
        cols_ = tie_groups(domain, extrusion, None)
        _, first = np.unique(cols_, return_index=True)
        rows.append(np.arange(n))
        cols.append(first[cols_])
    lut = np.full(domain.dims, -1, np.int64)
    lut[tuple(mesh.elem_ijk.T)] = np.arange(n)
    pts = domain.origin + (mesh.elem_ijk + 0.5) * domain.pitch
    c = pattern.center if pattern.center is not None else domain.origin + 0.5 * np.asarray(domain.dims) * domain.pitch
    a = pattern.axis
    u_ax, v_ax = (a + 1) % 3, (a + 2) % 3
    q = pts - c
    theta = np.mod(np.arctan2(q[:, v_ax], q[:, u_ax]), 2 * np.pi)
    sector = 2 * np.pi / pattern.count
    k = np.floor(theta / sector + 1e-9)
    back = -k * sector
    cu, sv = np.cos(back), np.sin(back)
    p0 = pts.copy()
    p0[:, u_ax] = c[u_ax] + cu * q[:, u_ax] - sv * q[:, v_ax]
    p0[:, v_ax] = c[v_ax] + sv * q[:, u_ax] + cu * q[:, v_ax]
    ijk0 = np.floor((p0 - domain.origin) / domain.pitch).astype(int)
    ok = np.all((ijk0 >= 0) & (ijk0 < np.asarray(domain.dims)), axis=1)
    rep = np.full(n, -1)
    rep[ok] = lut[tuple(ijk0[ok].T)]
    has = rep >= 0
    rows.append(np.arange(n)[has])
    cols.append(rep[has])
    r = np.concatenate(rows)
    cc = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, cc)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


# --- maximum member -------------------------------------------------------

class MaxMember:
    """Local volume penalty (module docstring). Call with the projected
    densities; returns the excess over ALPHA (as a share of 1 - ALPHA) at
    the worst ball and the 95th percentile over the solid, and the penalty
    gradient per element."""

    def __init__(self, domain, max_member: float, design: np.ndarray):
        from scipy.signal import fftconvolve

        self._conv = lambda f: fftconvolve(f, self.kernel, mode="same")
        self.domain = domain
        r = _BALL * float(max_member) / domain.pitch
        k = int(np.ceil(r))
        i, j, l = np.mgrid[-k:k + 1, -k:k + 1, -k:k + 1]
        self.kernel = ((i * i + j * j + l * l) <= r * r).astype(float)
        self.ijk = tuple(domain.mesh.elem_ijk.T)
        self.design = design
        # normalized by the whole ball: the outside and keep-outs count as
        # void, so a flange along a face reads its own thickness (normalized
        # over the cells that exist, it read as if mirrored: twice as thick)
        self.norm = float(self.kernel.sum())

    def local(self, xp: np.ndarray) -> np.ndarray:
        f = np.zeros(self.domain.dims)
        f[self.ijk] = np.where(self.design, xp, 0.0)   # preserved bosses are not members
        return np.clip(self._conv(f)[self.ijk], 0.0, None) / self.norm

    def __call__(self, xp: np.ndarray):
        """(worst excess, 95th percentile excess over the solid, gradient)."""
        rho = self.local(xp)
        v = np.where(self.design, np.maximum(rho - ALPHA, 0.0) / (1.0 - ALPHA), 0.0)
        solid = self.design & (xp > 0.5)
        over = float(np.percentile(v[solid], 95)) if solid.any() else 0.0
        g = np.zeros(self.domain.dims)
        g[self.ijk] = 2.0 * v / (1.0 - ALPHA) / self.norm
        grad = np.clip(self._conv(g)[self.ijk], 0.0, None) * self.design
        return float(v.max()), over, grad


def member_thickness(solid: np.ndarray, preserved: np.ndarray, pitch: float) -> float:
    """Thickest member (mm) of a bool grid, from the distance transform:
    twice the largest inscribed ball in the solid that keeps clear of
    preserved geometry (a boss is the user's, not a member)."""
    if not solid.any():
        return 0.0
    d = ndimage.distance_transform_edt(np.pad(solid, 1)) [1:-1, 1:-1, 1:-1]
    dp = ndimage.distance_transform_edt(~preserved) if preserved.any() else np.full(solid.shape, np.inf)
    ok = solid & ~preserved & (d <= dp)
    if not ok.any():
        return 0.0
    return float(max(2.0 * d[ok].max() - 1.0, 1.0) * pitch)


# --- report ---------------------------------------------------------------

def report(study, domain, mesh) -> dict:
    """What the delivered mesh does against the study's manufacturing
    constraints, measured on its voxelization at the design grid."""
    from .verify import voxelize

    out = {}
    access, pattern, mx = study.access, study.pattern, study.max_member
    if access is None and pattern is None and mx is None:
        return out
    lo = domain.origin
    hi = lo + np.asarray(domain.dims) * domain.pitch
    solid, _ = voxelize(mesh, domain.pitch, box=(lo, hi))
    v = domain.pitch ** 3
    if access is not None:
        rough = Access(access.kind, access.directions)   # exact access, no cutter radius or floors
        blocked = machinable(solid, rough, domain.pitch)
        hidden = blocked & ~solid
        keep = np.zeros(domain.dims, bool)
        if study.keep_out:
            ctr = (domain.origin + (np.indices(domain.dims).reshape(3, -1).T + 0.5) * domain.pitch)
            for r in study.keep_out:
                keep |= r.shape.contains(ctr).reshape(domain.dims)
            # the cells its surface cuts through are its shadow too
            keep = ndimage.binary_dilation(keep, iterations=1)
        out["access"] = access.summary() | {
            "undercutVolume": float(hidden.sum() * v),
            "undercutInKeepOuts": float((hidden & keep).sum() * v),
        }
    if pattern is not None:
        out["pattern"] = pattern.summary()
    if mx is not None:
        pres = domain.full_grid(domain.preserve.astype(float)) > 0.5
        out["maxMember"] = {"limit": mx, "measured": member_thickness(solid, pres, domain.pitch)}
    return out
