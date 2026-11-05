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
it can only lower stress; the final analysis runs on the filled part. The
same fill is also held as passive solid inside the optimization once the
projection is sharp (simp, SimpOptions.tool_fill), so its volume is paid
for there instead of appearing at delivery.

Draft (`casting.draft`, degrees) is built into the delivered surface:
reconstruct takes the drafted closure of the meshed field
(`drafted_distance`, every wall leaning in going away from the parting
plane) and leans the design space's own walls in the same way, leaving
the user's interfaces (bores, bosses, mounting faces) exact. On the
optimizer's grid a degree or two moves a wall well under a cell, so it
is not modelled in the loop; it only adds material (except along the
design space's walls), and the delivered part is analysed with it.
`draft_report` reads the draft of every facet against the mold half
that releases it.

`machining_report` gives a milled part's tool reach (void the smallest
cutter cannot reach, tool stickout over diameter) and a rough time and
cost: roughing with a 10 mm cutter, rest machining with the smallest,
finishing passes over the machined area, setups from the directions that
cut, at per-family rates (MACHINING) a study can override.

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

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["Access", "Pattern", "AccessFilter", "MaxMember", "parse_milling", "parse_casting",
           "parse_pattern", "tie_groups", "machinable", "member_thickness", "report", "AXIS_DIRECTIONS",
           "drafted_distance", "parting_index", "draft_report", "machining_report", "MACHINING"]

AXIS_DIRECTIONS = [(a, s) for a in range(3) for s in (1, -1)]
# Solid share of a ball allowed by maxMember.
ALPHA = 0.85
# Ball radius / maxMember for which a slab maxMember thick reads ALPHA.
_BALL = 0.754

# Machining cost model (machining_report), per material family: roughing
# metal removal rate with a 10 mm end mill (cm^3/min, a mid-size VMC at
# conservative job-shop parameters), finishing feed (mm/min) and stock
# price ($/kg, bar or plate). Every figure can be overridden per study
# (milling.rates). Rough on purpose: the point is ranking designs and
# showing where the minutes go, not a quote.
MACHINING = {
    "plastic":   {"roughingRate": 30.0, "finishFeed": 2000.0, "stockPerKg": 15.0},
    "aluminium": {"roughingRate": 25.0, "finishFeed": 1500.0, "stockPerKg": 10.0},
    "titanium":  {"roughingRate": 2.5, "finishFeed": 300.0, "stockPerKg": 45.0},
    "steel":     {"roughingRate": 6.0, "finishFeed": 600.0, "stockPerKg": 4.0},
    "stainless": {"roughingRate": 4.0, "finishFeed": 400.0, "stockPerKg": 9.0},
    "nickel":    {"roughingRate": 1.5, "finishFeed": 200.0, "stockPerKg": 60.0},
}
# machine $/h by axes, $ per setup (fixture, probe, first-article check)
MACHINE_RATE = {2.5: 65.0, 3.0: 75.0, 5.0: 120.0}
PER_SETUP = 40.0
MACHINING_KEYS = {"roughingRate", "finishFeed", "stockPerKg", "machinePerHour", "perSetup", "stepover",
                  "stockMargin"}
# tool stickout / diameter past which a cut needs a long-reach tool
STICKOUT_WARN = 4.0
# draft read on mesh facets: tolerance (deg) for facet noise, and the
# extra draft reconstruct builds in so the meshed walls clear the request
DRAFT_TOL = 0.5
DRAFT_MARGIN = 1.0
# ...at mesh cells up to this size (mm); coarser cells read walls shallower,
# so the margin grows in proportion (the coarse eval tier's 0.97 mm cells
# left the cast arm link 373 mm^2 under draft at 1 deg, against 300)
DRAFT_MARGIN_STEP = 0.55


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
    draft: float = 0.0                # casting: degrees of draft on every wall the mold pulls off
    parting: float | None = None      # casting: parting plane, mm along the draw axis (None: chosen)
    rates: dict | None = None         # milling: cost-model overrides (MACHINING)

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
            if self.draft:
                out["draft"] = self.draft
            if self.parting is not None:
                out["parting"] = self.parting
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
    rates = v.get("rates")
    if rates is not None:
        if not isinstance(rates, dict) or any(k not in MACHINING_KEYS for k in rates):
            raise ValueError(f"milling.rates overrides the cost model: keys {sorted(MACHINING_KEYS)}")
        for k, x in rates.items():
            if not isinstance(x, (int, float)) or isinstance(x, bool) or x < 0:
                raise ValueError(f"milling.rates.{k} is a number >= 0")
    return Access("milling", parsed, axes=axes, tool_radius=0.5 * float(tool) if tool else None,
                  step=float(step) if step else None, rates=dict(rates) if rates else None)


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
    draft = v.get("draft", 0.0)
    if not isinstance(draft, (int, float)) or isinstance(draft, bool) or not 0.0 <= draft <= 15.0:
        raise ValueError("casting.draft is the draft angle in degrees (0-15; 1-3 is usual for die casting "
                         "and injection molding, 3-5 for sand)")
    parting = v.get("parting")
    if parting is not None and (not isinstance(parting, (int, float)) or isinstance(parting, bool)):
        raise ValueError("casting.parting is the parting plane's coordinate along the draw axis, mm")
    if parting is not None and sides == 1:
        raise ValueError("casting.parting is for two-sided molds; a one-sided mold parts at the part's far face")
    dirs = [draw] if sides == 1 else [draw, (draw[0], -draw[1])]
    return Access("casting", dirs, draw=draw, draft=float(draft),
                  parting=None if parting is None else float(parting))


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
        if access.kind == "casting":
            out["draft"] = draft_report(study, mesh, access, domain.pitch,
                                        center=lo + 0.5 * np.asarray(domain.dims) * domain.pitch)
        else:
            out["machining"] = machining_report(study, mesh, solid, access, domain.pitch, keep, domain.origin)
    if pattern is not None:
        out["pattern"] = pattern.summary()
    if mx is not None:
        pres = domain.full_grid(domain.preserve.astype(float)) > 0.5
        out["maxMember"] = {"limit": mx, "measured": member_thickness(solid, pres, domain.pitch)}
    warn = list((out.get("machining") or {}).get("warnings", []))
    d = out.get("draft")
    if d and d["required"] > 0 and d["underDraftedArea"] > 0.02 * max(d["wallArea"], 1.0):
        warn.append(f"{d['underDraftedArea']:.0f} mm^2 of wall ({d['underDraftedShare']:.0%}) has less than "
                    f"{d['required']:g} deg of draft")
    if warn:
        out["warnings"] = warn
    return out


# --- draft ----------------------------------------------------------------

def parting_index(solid: np.ndarray, axis: int) -> int:
    """The layer index along `axis` where a two-sided mold parts: the
    widest cross-section (the mold halves must both pull away from it),
    the one nearest the middle of the part among near-ties."""
    area = solid.sum(axis=tuple(i for i in range(3) if i != axis)).astype(float)
    if area.max() <= 0:
        return solid.shape[axis] // 2
    idx = np.nonzero(area > 0)[0]
    mid = 0.5 * (idx[0] + idx[-1])
    cand = np.nonzero(area >= 0.98 * area.max())[0]
    return int(cand[np.argmin(np.abs(cand - mid))])


def drafted_distance(d: np.ndarray, axis: int, sign: int, sides: int, parting: int | None, step: float,
                     angle: float, grow: bool = True) -> np.ndarray:
    """Signed distance (negative inside) of the drafted closure of a part:
    every wall leaning in by `angle` degrees as it runs away from the
    parting plane, so each mold half pulls clear. A point is solid if it
    lies within (h tan angle) sideways of solid h further from the parting
    plane: G(z) = min(d(z), G(z + step) - step tan angle), a running
    minimum outward from the parting plane on each side (one side for a
    one-sided mold, parting at its far face). Exact on the grid and in
    sub-cell steps; it only adds material.

    grow=False gives the largest drafted subset instead (a design space's
    walls leaning in: G(z) = max(d(z), G(z - step) + step tan angle), a
    running maximum outward from the parting plane): it only removes."""
    t = np.tan(np.radians(angle)) * step
    g = np.moveaxis(d, axis, 0).copy()
    n = g.shape[0]

    def run(outward):
        # `outward` runs away from the parting plane
        if grow:
            for a, b in zip(outward[::-1][:-1], outward[::-1][1:]):
                g[b] = np.minimum(g[b], g[a] - t)   # b one layer nearer the parting plane than a
        else:
            for a, b in zip(outward[:-1], outward[1:]):
                g[b] = np.maximum(g[b], g[a] + t)   # b one layer further out than a

    if sides == 1:
        # mold half pulls toward +sign: walls lean in going that way
        run(list(range(n)) if sign > 0 else list(range(n - 1, -1, -1)))
    else:
        p = n // 2 if parting is None else int(np.clip(parting, 0, n - 1))
        run(list(range(p, n)))
        run(list(range(p - 1, -1, -1)) if p > 0 else [])
        if grow and p > 0:
            # one section either side of the parting plane: no ledge there
            # for the surface smoothing to lean the wrong way
            g[p] = g[p - 1] = np.minimum(g[p], g[p - 1])
    return np.moveaxis(g, 0, axis)


def interface_shapes(study) -> list:
    """The user's interfaces: preserved and keep-out shapes, and the faces
    the part is held and loaded at. Bores, bosses and mounting faces are
    machined after casting or drafted in CAD, so draft leaves them alone."""
    out = [r.shape for r in study.preserve] + [r.shape for r in study.keep_out]
    for sup in getattr(study, "supports", ()):
        try:
            out.append(sup.shape if sup.shape is not None else study.region_shape(sup.region)[0])
        except (ValueError, AttributeError):
            pass
    for case in getattr(study, "load_cases", ()):
        for ld in case.loads:
            try:
                out.append(ld.shape if ld.shape is not None else study.region_shape(ld.region)[0])
            except (ValueError, AttributeError):
                pass
    return out


def _interface_mask(study, pts, tol, keep_tol=None):
    """Points on the interfaces; within `keep_tol` of a keep-out (the
    meshed field fades out over half a cell round each one before the
    exact cut, so that band is the keep-out's own wall too)."""
    near = np.zeros(len(pts), bool)
    keeps = {id(r.shape) for r in study.keep_out}
    for sh in interface_shapes(study):
        t = keep_tol if keep_tol is not None and id(sh) in keeps else tol
        near |= np.abs(sh.sdf(pts)) < t
    return near


def symmetric_parting(study, access: Access, center) -> float | None:
    """A two-sided mold on a part symmetric about a plane across the draw
    parts on that plane (study symmetry planes run through the design
    space's centre `center`)."""
    if access.parting is not None:
        return access.parting
    if center is None or len(access.directions) != 2:
        return None
    a = access.draw[0]
    if "xyz"[a] in [str(x).lower() for x in (getattr(study, "symmetry", None) or [])]:
        return float(np.asarray(center, float)[a])
    return None


def draft_report(study, mesh, access: Access, pitch: float, center=None) -> dict:
    """Draft angle of every facet against the mold half that releases it.
    A facet on the far side of the parting plane from a half's pull must
    face that way by at least the draft: n . pull >= sin(draft). Facets on
    preserved or keep-out surfaces are counted apart (interfaces), and a
    band one cell wide round the parting plane is the parting line."""
    a, sgn = access.draw
    d = np.zeros(3)
    d[a] = sgn
    n = mesh.face_normals
    area = mesh.area_faces
    c = mesh.triangles_center
    h = c[:, a]
    sides = len(access.directions)
    meta = getattr(mesh, "metadata", None) or {}
    if sides == 1:
        parting = float(mesh.bounds[0 if sgn > 0 else 1][a])
        pull = np.tile(d, (len(n), 1))
    else:
        parting = meta.get("parting")
        if parting is None:
            parting = symmetric_parting(study, access, center)
        if parting is None:
            from .verify import voxelize
            solid, org = voxelize(mesh, pitch)
            parting = float(org[a] + (parting_index(solid, a) + 0.5) * pitch)
        parting = float(parting)
        up = (h - parting) * sgn >= 0
        pull = np.where(up[:, None], d, -d)
    s = np.einsum("ij,ij->i", n, pull)
    angle = np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
    iface = _interface_mask(study, c, max(0.05, 0.1 * pitch), 0.6 * pitch)
    band = np.abs(h - parting) < 0.5 * pitch if sides == 2 else np.zeros(len(n), bool)
    if sides == 1:
        # the face the part sits on at the parting plane
        band |= (np.abs(h - parting) < 0.5 * pitch) & (s < -0.9)
    need = float(access.draft)
    low = angle < need - DRAFT_TOL
    walls = ~iface & ~band
    bad = low & walls
    out = {"required": need, "parting": parting,
           "underDraftedArea": float(area[bad].sum()),
           "undercutFacetArea": float(area[walls & (s < -0.05)].sum()),
           "interfaceUnderDraftedArea": float(area[low & iface].sum()),
           "wallArea": float(area[walls & (np.abs(s) < 0.9)].sum())}
    sw = walls & (np.abs(s) < 0.9)
    if sw.any():
        order = np.argsort(angle[sw])
        cum = np.cumsum(area[sw][order]) / area[sw].sum()
        out["p1Draft"] = float(angle[sw][order][np.searchsorted(cum, 0.01)])
        out["p5Draft"] = float(angle[sw][order][np.searchsorted(cum, 0.05)])
    out["underDraftedShare"] = out["underDraftedArea"] / out["wallArea"] if out["wallArea"] > 0 else 0.0
    return out


# --- machining ------------------------------------------------------------

def _family(material) -> str:
    name = (getattr(material, "name", "") or "").lower()
    rho = float(getattr(material, "density", 2.7))
    if rho < 2.0:
        return "plastic"
    if rho < 3.3:
        return "aluminium"
    if rho < 5.0:
        return "titanium"
    if rho > 8.2 or any(k in name for k in ("inconel", "625", "718", "nickel")):
        return "nickel"
    if any(k in name for k in ("316", "304", "stainless", "15-5", "17-4")):
        return "stainless"
    return "steel"


def _reach_by_direction(solid: np.ndarray, access: Access, pitch: float, radius: float):
    """Per access direction, the void a flat end mill of `radius` sweeps
    from that side (bool grids, unpadded), and the tool-centre depth from
    the stock face reached in each cell (cells)."""
    r = radius / pitch
    pad = int(np.ceil(r)) + 1
    s = np.pad(solid, pad, constant_values=False)
    out = []
    for a, sign in access.directions:
        disk = _disk(r, a) if r >= 1.0 else None
        void = ~s
        ok = ndimage.binary_erosion(void, structure=disk, border_value=1) if disk is not None else void
        g = np.moveaxis(ok, a, -1)
        if sign > 0:
            g = g[..., ::-1]
        reach = np.logical_and.accumulate(g, axis=-1)
        if sign > 0:
            reach = reach[..., ::-1]
        reach = np.moveaxis(reach, -1, a)
        cut = ndimage.binary_dilation(reach, structure=disk) if disk is not None else reach
        out.append(cut[pad:-pad, pad:-pad, pad:-pad])
    return out


def machining_report(study, mesh, solid: np.ndarray, access: Access, pitch: float, keep: np.ndarray,
                     origin) -> dict:
    """Tool reach and a machining time and cost estimate for a milled part
    on its voxel grid `solid` (the part's bounding box plus a cell all
    round). Stock: the bounding box plus `stockMargin` (1 mm) a side.
    Roughing clears the stock with a 10 mm end mill at the family's rate;
    rest machining clears what only the smallest cutter reaches at its
    rate (scaled with diameter squared, as both cut depth and width are);
    finishing passes over every machined surface at a stepover of a tenth
    of the cutter. Setups: the access directions that cut a meaningful
    share; 3+2 five-axis reaches five sides per fixture."""
    mat = study.material
    fam = _family(mat)
    rates = dict(MACHINING[fam])
    rates["machinePerHour"] = MACHINE_RATE.get(float(access.axes or 3.0), 75.0)
    rates["perSetup"] = PER_SETUP
    tool_d = 2.0 * access.tool_radius if access.tool_radius else 6.0
    rates["stepover"] = 0.1 * tool_d
    rates["stockMargin"] = 1.0
    rates.update(access.rates or {})
    v = pitch ** 3
    bounds = np.asarray(mesh.bounds, float)
    size = bounds[1] - bounds[0] + 2.0 * rates["stockMargin"]
    stock_vol = float(np.prod(size))              # mm^3
    part_vol = float(mesh.volume)
    removed = max(stock_vol - part_vol, 0.0)
    out = {"family": fam, "stock": [float(x) for x in size],
           "stockMass": stock_vol * mat.density / 1000.0, "removedVolume": removed}

    # what each direction can cut with the roughing tool and the smallest
    rough_r = max(5.0, access.tool_radius or 0.0)
    fine_r = access.tool_radius or 0.0
    void = ~solid
    by_dir = _reach_by_direction(solid, access, pitch, fine_r)
    cut_fine = np.zeros_like(solid)
    for c in by_dir:
        cut_fine |= c
    cut_rough = np.zeros_like(solid)
    if rough_r > fine_r:
        for c in _reach_by_direction(solid, access, pitch, rough_r):
            cut_rough |= c
    else:
        cut_rough = cut_fine
    # the stock: cells inside the part's bounding box (outside it is air)
    ctr = [np.asarray(origin, float)[i] + (np.arange(solid.shape[i]) + 0.5) * pitch for i in range(3)]
    inb = [(c > bounds[0][i]) & (c < bounds[1][i]) for i, c in enumerate(ctr)]
    stock = inb[0][:, None, None] & inb[1][None, :, None] & inb[2][None, None, :]
    unreach = void & ~cut_fine & ~keep & stock
    out["toolDiameter"] = 2.0 * fine_r if fine_r else None
    out["toolUnreachableVolume"] = float(unreach.sum() * v)
    # setups: greedy, the direction cutting most of what is left first
    left = void & cut_fine
    total = max(int(left.sum()), 1)
    used = []
    rest = left.copy()
    for _ in range(len(by_dir)):
        gains = [int((c & rest).sum()) for c in by_dir]
        k = int(np.argmax(gains))
        if gains[k] < 0.01 * total:
            break
        used.append(_label(access.directions[k]))
        rest &= ~by_dir[k]
    setups = max(1, int(np.ceil(len(used) / 5.0))) if access.axes == 5 else max(1, len(used))
    out["directionsUsed"] = used
    out["setups"] = setups
    # stickout: tool-centre depth below the stock face each cut needs,
    # the shallower side when two directions reach a cell
    need = np.full(solid.shape, np.inf)
    for (a, sign), c in zip(access.directions, by_dir):
        n_ = solid.shape[a]
        idx = np.arange(n_)
        # the stock face along this direction: the part's own extent
        prof = solid.any(axis=tuple(i for i in range(3) if i != a))
        nz = np.nonzero(prof)[0]
        if not len(nz):
            continue
        face = nz[-1] + 1 if sign > 0 else nz[0]
        depth = (face - idx) if sign > 0 else (idx - face + 1)
        shape = [1, 1, 1]
        shape[a] = n_
        dgrid = np.broadcast_to(np.clip(depth, 0, None).reshape(shape), solid.shape).astype(float)
        need = np.where(c, np.minimum(need, dgrid), need)
    cut_cells = void & cut_fine & np.isfinite(need) & stock
    stick = float(need[cut_cells].max() * pitch) if cut_cells.any() else 0.0
    out["maxStickout"] = stick
    out["maxStickoutToDiameter"] = stick / tool_d

    # time
    rr = rates["roughingRate"]
    rough_mm3 = removed - float((cut_fine & ~cut_rough & void).sum() * v)
    rest_mm3 = float((cut_fine & ~cut_rough & void).sum() * v)
    fine_rate = rr * float(np.clip((tool_d / 10.0) ** 2, 0.05, 2.0))
    rough_min = max(rough_mm3, 0.0) / 1000.0 / rr
    rest_min = rest_mm3 / 1000.0 / fine_rate if rest_mm3 > 0 else 0.0
    # machined surface: every facet not lying on the stock's faces
    c = mesh.triangles_center
    on_face = np.zeros(len(c), bool)
    for ax in range(3):
        on_face |= (np.abs(c[:, ax] - bounds[0][ax]) < 1e-3) | (np.abs(c[:, ax] - bounds[1][ax]) < 1e-3)
    area = float(mesh.area_faces[~on_face].sum())
    finish_min = area / (rates["stepover"] * rates["finishFeed"])
    minutes = rough_min + rest_min + finish_min
    out["machinedArea"] = area
    out["minutes"] = {"roughing": rough_min, "rest": rest_min, "finishing": finish_min, "total": minutes}
    cost_machine = minutes / 60.0 * rates["machinePerHour"]
    cost_setup = setups * rates["perSetup"]
    cost_stock = out["stockMass"] / 1000.0 * rates["stockPerKg"]
    out["cost"] = {"machine": cost_machine, "setups": cost_setup, "stock": cost_stock,
                   "total": cost_machine + cost_setup + cost_stock, "currency": "USD"}
    out["rates"] = rates
    warn = []
    if out["maxStickoutToDiameter"] > STICKOUT_WARN:
        warn.append(f"a cut needs {stick:.0f} mm of tool stickout, {out['maxStickoutToDiameter']:.1f}x the "
                    f"{tool_d:g} mm cutter: past {STICKOUT_WARN:g}x it needs a long-reach tool and chatters")
    if out["toolUnreachableVolume"] > 10 * v:
        warn.append(f"{out['toolUnreachableVolume']:.0f} mm^3 of void the cutter cannot reach from "
                    f"{', '.join(_label(d) for d in access.directions)}")
    if warn:
        out["warnings"] = warn
    return out
