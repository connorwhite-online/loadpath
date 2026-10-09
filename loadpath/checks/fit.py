"""
Component-fit verifier (MTR-204): does the built enclosure actually FIT the
component the brief promised it would?

`check_fit(mesh, spec)` voxelizes the part (reusing sdf_kit's occupancy
helper), solves where the component actually sits inside the interior air,
and verifies three machine-checkable contracts per component:

  fit_cavity  — a placement exists where the component envelope (PCB footprint
                + clearance, board top to tallest part + clearance) is fully in
                air, and that placement is walled on all four sides (it's a
                cavity, not empty space beside the part).
  fit_bosses  — at the solved placement, every mounting hole in the pattern
                has a pilot bore (air) and boss/standoff material around it
                just below the board seating plane.
  fit_cutout  — each verified port has a through-opening of (roughly) the
                cutout size at the expected edge/offset/height.

Placement solve: the envelope box is slid over the air grid (uniform_filter =
"is every voxel under the box air?") for all four flat rotations; among
feasible poses we prefer the one matching the most mounting holes, then the
lowest seat. The cavity walls therefore PIN the board — bosses drilled 4 mm
off pattern cannot be rescued by sliding the board 4 mm, because the envelope
no longer fits.

Resolution honesty (same posture as networks.py): verified at voxel pitch p
means "no interference thicker than ~p" — sub-voxel interference can slip
through the occupancy grid. The report carries the pitch used; below-board
pin clearance is NOT yet asserted (bosses/standoffs already imply seating
height; a dedicated pins check is future work).

Spec shape (built by lib/cad/dimension-check.ts buildFitChecksFromTargets):
  {"components": [{
      "id": "pico", "label": "Raspberry Pi Pico",
      "boardMm": [L, W, T], "clearanceMm": c, "aboveMm": a, "belowMm": b,
      "holes": {"positions": [[x, y], ...], "diaMm": d} | null,
      "ports": [{"id": "micro-usb", "edge": "-x", "offsetMm": o,
                 "heightMm": h, "wMm": w, "hMm": hh},
                {"id": "screen-window", "face": "+z", "xMm": x, "yMm": y,
                 "wMm": w, "hMm": hh}, ...]
  }]}

Port kinds: lateral ports carry edge/offset/height and are checked with
outward rays through the wall; `face: "+z"` ports (screen windows, lenses,
shafts, MTR-203) carry a center in the component frame and are checked with
upward rays from the component top through the lid.

Component frame: origin at board center, +x along length, +y along width,
z up; port heights above the PCB TOP. Result ids: "fit:<id>:cavity",
"fit:<id>:bosses", "fit:<id>:port:<portId>" — matched by the TS side against
the DimensionTargets that requested them.
"""
import numpy as np
from scipy import ndimage

from sdf_kit import _solid_voxels

__all__ = ["check_fit"]

# Feasibility: every sampled envelope voxel must be air.
_AIR_FRACTION_OK = 0.999
# A lateral face counts as "walled" when at least this fraction of its
# outward rays hit solid before leaving the grid (port cutouts legitimately
# open part of a wall).
_WALL_MIN_BLOCKED = 0.35
# Boss ring: sample points on a circle this far outside the hole radius...
_RING_EXTRA_MM = 1.0
# ...and require at least this many of the 8 ring points to be solid.
_RING_POINTS = 8
_RING_MIN_SOLID = 6
# Cap on candidate placements scored per rotation (subsampled by z-order).
_MAX_CANDIDATES = 4000
# Voxel budget — pitch grows until the padded grid fits.
_MAX_VOXELS = 4.5e6


def _rot_xy(pts, rot):
    """Rotate (n, 2) component-frame XY points by rot degrees (multiple of 90)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    if rot == 0:
        return pts
    if rot == 90:
        return np.column_stack([-pts[:, 1], pts[:, 0]])
    if rot == 180:
        return -pts
    return np.column_stack([pts[:, 1], -pts[:, 0]])  # 270


def _pitch_for(mesh, comps):
    """Voxel pitch: fine enough to resolve pilot bores, coarse enough to fit
    the voxel budget."""
    ext = np.asarray(mesh.extents, float)
    pitch = float(max(ext)) / 110.0
    pitch = min(max(pitch, 0.35), 0.8)
    dias = [
        float(c["holes"]["diaMm"])
        for c in comps
        if c.get("holes") and c["holes"].get("diaMm")
    ]
    if dias:
        pitch = min(pitch, min(dias) * 0.35)
    while np.prod(np.ceil(ext / pitch) + 6) > _MAX_VOXELS:
        pitch *= 1.25
    return float(pitch)


class _AirGrid:
    """Boolean air occupancy over the mesh bbox (padded). Points outside the
    grid are air by definition (outside the part's bounding box)."""

    def __init__(self, mesh, pitch):
        solid, origin = _solid_voxels(mesh, pitch, pad=3)
        self.air = ~solid
        self.origin = np.asarray(origin, float)
        self.pitch = float(pitch)

    def is_air(self, pts):
        """(n, 3) world points -> (n,) bool."""
        pts = np.atleast_2d(np.asarray(pts, float))
        idx = np.rint((pts - self.origin) / self.pitch).astype(int)
        shape = np.asarray(self.air.shape)
        inside = np.all((idx >= 0) & (idx < shape), axis=1)
        out = np.ones(len(idx), bool)
        ii = idx[inside]
        if len(ii):
            out[inside] = self.air[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def ray_clear(self, start, direction, step=None):
        """True when marching from `start` along `direction` exits the grid
        without ever touching solid — i.e. an opening goes all the way out."""
        step = step or self.pitch * 0.9
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        span = float(np.linalg.norm(np.asarray(self.air.shape) * self.pitch))
        n = int(span / step) + 2
        ts = self.pitch * 0.6 + step * np.arange(n)
        pts = np.asarray(start, float)[None, :] + ts[:, None] * d[None, :]
        return bool(np.all(self.is_air(pts)))


def _feasible_centers(grid, box_mm):
    """All voxel centers where an axis-aligned box of `box_mm` is entirely
    air. Returns (centers_world (n,3), best_air_fraction)."""
    airf = grid.air.astype(np.float32)
    # Erode the sampled box by ~1 voxel per axis: surface voxelization already
    # dilates the solid by ~half a pitch, and we must not double-count it.
    size = np.maximum(np.rint(np.asarray(box_mm, float) / grid.pitch) - 1, 1).astype(int)
    if np.any(size > np.asarray(grid.air.shape)):
        return np.empty((0, 3)), 0.0
    frac = ndimage.uniform_filter(airf, size=tuple(size), mode="constant", cval=0.0)
    half = size // 2
    valid = np.zeros_like(grid.air)
    sx, sy, sz = grid.air.shape
    valid[half[0]:sx - half[0], half[1]:sy - half[1], half[2]:sz - half[2]] = True
    if not valid.any():
        return np.empty((0, 3)), 0.0
    best = float(frac[valid].max())
    feas = (frac >= _AIR_FRACTION_OK) & valid
    idxs = np.argwhere(feas)
    centers = grid.origin + idxs * grid.pitch
    return centers, best


def _boss_matches(grid, comp, centers, rot, env_z):
    """(n_candidates,) matched-hole counts at each candidate box center.
    A hole matches when its pilot is air and a ring around it is solid just
    below the board seating plane (= envelope bottom)."""
    holes = comp.get("holes")
    if not holes or not holes.get("positions"):
        return np.zeros(len(centers), int)
    pos = _rot_xy(holes["positions"], rot)  # (H, 2) world-frame offsets
    r_ring = float(holes["diaMm"]) / 2.0 + _RING_EXTRA_MM
    ang = np.linspace(0, 2 * np.pi, _RING_POINTS, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)]) * r_ring  # (P, 2)
    depth1 = max(1.5 * grid.pitch, 1.0)
    depth2 = depth1 + 1.2

    matched = np.zeros(len(centers), int)
    zb = centers[:, 2] - env_z / 2.0  # board bottom = envelope bottom
    for h in range(len(pos)):
        cxy = centers[:, :2] + pos[h][None, :]  # (n, 2)
        ok_h = np.ones(len(centers), bool)
        # Pilot bore: air at the hole axis just below the seat (either depth).
        pilot_ok = np.zeros(len(centers), bool)
        for d in (depth1, depth2):
            pts = np.column_stack([cxy, zb - d])
            pilot_ok |= grid.is_air(pts)
        ok_h &= pilot_ok
        # Boss material: ring of solid around the bore (either depth).
        ring_ok = np.zeros(len(centers), bool)
        for d in (depth1, depth2):
            solid_count = np.zeros(len(centers), int)
            for p in range(_RING_POINTS):
                pts = np.column_stack([cxy + ring[p][None, :], zb - d])
                solid_count += (~grid.is_air(pts)).astype(int)
            ring_ok |= solid_count >= _RING_MIN_SOLID
        ok_h &= ring_ok
        matched += ok_h.astype(int)
    return matched


def _walls_blocked(grid, center, box_mm):
    """Per-face blocked fraction for the four lateral faces of the box at
    `center`. Returns a dict face -> fraction of outward rays that hit solid."""
    bx, by, bz = box_mm
    zs = [center[2] - bz * 0.25, center[2] + bz * 0.25]
    out = {}
    for face, (axis, sign) in {
        "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1),
    }.items():
        tangent = 1 - axis
        half_t = (by if axis == 0 else bx) / 2.0 - grid.pitch
        offs = np.linspace(-half_t, half_t, 5)
        n_hit = 0
        n_all = 0
        d = np.zeros(3)
        d[axis] = sign
        start_axis = center[axis] + sign * ((bx if axis == 0 else by) / 2.0)
        for z in zs:
            for t in offs:
                p = np.array(center, float)
                p[axis] = start_axis
                p[tangent] = center[tangent] + t
                p[2] = z
                n_all += 1
                if not grid.ray_clear(p, d):
                    n_hit += 1
        out[face] = n_hit / max(n_all, 1)
    return out


def _port_open(grid, comp, center, rot, env_z, port):
    """Is there a through-opening for `port` at the solved placement?
    Five rays (center, ±width, ±height inside the cutout) must all exit the
    grid without touching solid. Lateral ports shoot outward through the
    wall; `face: "+z"` ports (screen windows) shoot upward through the lid."""
    L, W, T = [float(v) for v in comp["boardMm"]]
    zb = center[2] - env_z / 2.0
    if port.get("face") == "+z":
        # Top-face windows are usually cut at exactly the recommended size,
        # and window-bearing components often have no bosses to pin the solved
        # seat — allow 1 mm of seat slack on top of the voxel margin so a
        # correct window over a slightly-slid seat still verifies.
        margin = max(grid.pitch, 0.8) + 1.0
        half_w = max(float(port["wMm"]) / 2.0 - margin, 0.0)
        half_h = max(float(port["hMm"]) / 2.0 - margin, 0.0)
        p_b = np.array([float(port.get("xMm") or 0.0), float(port.get("yMm") or 0.0)])
        u_w = _rot_xy(np.array([1.0, 0.0]), rot)[0]
        u_h = _rot_xy(np.array([0.0, 1.0]), rot)[0]
        c_w = center[:2] + _rot_xy(p_b, rot)[0]
        z0 = zb + T + grid.pitch
        direction = np.array([0.0, 0.0, 1.0])
        for dw, dh in ((0, 0), (half_w, 0), (-half_w, 0), (0, half_h), (0, -half_h)):
            start = np.array([
                c_w[0] + u_w[0] * dw + u_h[0] * dh,
                c_w[1] + u_w[1] * dw + u_h[1] * dh,
                z0,
            ])
            if not grid.ray_clear(start, direction):
                return False
        return True
    edge = port["edge"]
    off = float(port.get("offsetMm") or 0.0)
    if edge in ("+x", "-x"):
        p_b = np.array([(L / 2.0) * (1 if edge == "+x" else -1), off])
        n_b = np.array([1.0 if edge == "+x" else -1.0, 0.0])
        t_b = np.array([0.0, 1.0])
    else:
        p_b = np.array([off, (W / 2.0) * (1 if edge == "+y" else -1)])
        n_b = np.array([0.0, 1.0 if edge == "+y" else -1.0])
        t_b = np.array([1.0, 0.0])
    p_w = center[:2] + _rot_xy(p_b, rot)[0]
    n_w = _rot_xy(n_b, rot)[0]
    t_w = _rot_xy(t_b, rot)[0]
    z_c = zb + T + float(port.get("heightMm") or 0.0)
    margin = max(grid.pitch, 0.8)
    half_w = max(float(port["wMm"]) / 2.0 - margin, 0.0)
    half_h = max(float(port["hMm"]) / 2.0 - margin, 0.0)
    direction = np.array([n_w[0], n_w[1], 0.0])
    for dw, dh in ((0, 0), (half_w, 0), (-half_w, 0), (0, half_h), (0, -half_h)):
        start = np.array([
            p_w[0] + t_w[0] * dw,
            p_w[1] + t_w[1] * dw,
            z_c + dh,
        ])
        if not grid.ray_clear(start, direction):
            return False
    return True


def _check_component(grid, comp):
    """All fit results for one component spec. Never raises past the caller's
    failure isolation — plain dict results only."""
    cid = str(comp.get("id") or "component")
    label = str(comp.get("label") or cid)
    L, W, T = [float(v) for v in comp["boardMm"]]
    c = float(comp.get("clearanceMm") or 1.0)
    above = float(comp.get("aboveMm") or 0.0)
    env = np.array([L + 2 * c, W + 2 * c, T + above + c])
    holes = comp.get("holes")
    ports = comp.get("ports") or []
    n_holes = len(holes["positions"]) if holes and holes.get("positions") else 0

    results = []

    # ---- Placement solve over the four flat rotations ------------------
    # The envelope box is 180°-symmetric, so rotations share feasible sets in
    # pairs: {0, 180} use env as-is, {90, 270} use the XY-swapped box.
    env_swapped = np.array([env[1], env[0], env[2]])
    feas = {0: _feasible_centers(grid, env), 90: _feasible_centers(grid, env_swapped)}
    best_frac = max(feas[0][1], feas[90][1])
    best = None  # ((matched, ports_open, -z), center, rot, box)
    for rot in (0, 90, 180, 270):
        box = env if rot % 180 == 0 else env_swapped
        centers = feas[rot % 180][0]
        if len(centers) == 0:
            continue
        if len(centers) > _MAX_CANDIDATES:
            order = np.argsort(centers[:, 2], kind="stable")
            centers = centers[order][:: max(1, len(centers) // _MAX_CANDIDATES)]
        matched = _boss_matches(grid, comp, centers, rot, env[2])
        # Best candidate for this rotation: most holes matched, then lowest.
        order = np.lexsort((centers[:, 2], -matched))
        cand = centers[order[0]]
        m = int(matched[order[0]])
        ports_open = sum(
            1 for p in ports if _port_open(grid, comp, cand, rot, env[2], p)
        )
        key = (m, ports_open, -cand[2])
        if best is None or key > best[0]:
            best = (key, cand, rot, box)

    if best is None:
        results.append({
            "id": f"fit:{cid}:cavity",
            "kind": "fit_cavity",
            "ok": False,
            "got": round(best_frac, 3),
            "note": (
                f"{label} envelope {env[0]:.1f} x {env[1]:.1f} x {env[2]:.1f} mm "
                f"(board + clearance) has NO placement fully inside interior air — "
                f"best candidate still overlaps solid (max air fraction "
                f"{best_frac:.0%}). Enlarge the cavity so the board plus "
                f"clearance actually fits."
            ),
        })
        if n_holes:
            results.append({
                "id": f"fit:{cid}:bosses",
                "kind": "fit_bosses",
                "ok": None,
                "note": "not evaluated — no feasible cavity placement",
            })
        for p in ports:
            results.append({
                "id": f"fit:{cid}:port:{p['id']}",
                "kind": "fit_cutout",
                "ok": None,
                "note": "not evaluated — no feasible cavity placement",
            })
        return results

    _key, center, rot, box = best

    # ---- Cavity: feasible + actually enclosed ---------------------------
    walls = _walls_blocked(grid, center, box)
    open_faces = [f for f, blocked in walls.items() if blocked < _WALL_MIN_BLOCKED]
    cavity_ok = len(open_faces) == 0
    note = (
        f"{label} seats at ({center[0]:.1f}, {center[1]:.1f}), board bottom "
        f"z={center[2] - env[2] / 2:.1f}, rotated {rot} deg (verified at voxel "
        f"pitch {grid.pitch:.2f} mm)"
    )
    if not cavity_ok:
        note = (
            f"{label} fits in the air volume but is not enclosed: side(s) "
            f"{', '.join(sorted(open_faces))} of the solved seat have no wall "
            f"(blocked fractions {', '.join(f'{f}={walls[f]:.0%}' for f in sorted(walls))}). "
            f"Walls must surround the component seat."
        )
    results.append({
        "id": f"fit:{cid}:cavity",
        "kind": "fit_cavity",
        "ok": bool(cavity_ok),
        "got": 1.0 if cavity_ok else 0.0,
        "note": note,
    })

    # ---- Bosses ----------------------------------------------------------
    if n_holes:
        matched = int(_boss_matches(grid, comp, center[None, :], rot, env[2])[0])
        ok = matched == n_holes
        if ok:
            bnote = f"all {n_holes} mounting holes have a pilot bore and boss material at the seat"
        else:
            bnote = (
                f"only {matched}/{n_holes} mounting holes align with boss/standoff "
                f"material + pilot bore at the solved seat (pattern: "
                f"{holes['positions']}, D{holes['diaMm']} mm, board frame from "
                f"center, rotated {rot} deg). Bosses must sit exactly under the "
                f"hole pattern with a pilot hole (~{float(holes['diaMm']) - 0.4:.1f} mm) each."
            )
        results.append({
            "id": f"fit:{cid}:bosses",
            "kind": "fit_bosses",
            "ok": bool(ok),
            "got": matched,
            "note": bnote,
        })

    # ---- Ports -----------------------------------------------------------
    for p in ports:
        ok = _port_open(grid, comp, center, rot, env[2], p)
        if p.get("face") == "+z":
            where_ok = f"{p['id']} opening verified through the top face"
            where_bad = (
                f"no through-opening for {p['id']} ({p['wMm']} x {p['hMm']} mm) "
                f"above the component: window center at ({p.get('xMm', 0)}, "
                f"{p.get('yMm', 0)}) in the board frame. The lid there is "
                f"solid, or the window is misplaced / too small."
            )
        else:
            where_ok = f"{p['id']} opening verified through the {p['edge']} wall"
            where_bad = (
                f"no through-opening for {p['id']} ({p['wMm']} x {p['hMm']} mm) "
                f"where the connector sits: board edge {p['edge']}, "
                f"{p['offsetMm']} mm along the edge, center "
                f"{p['heightMm']} mm above the board top. The wall there is "
                f"solid, or the cutout is on the wrong face / misplaced / too small."
            )
        results.append({
            "id": f"fit:{cid}:port:{p['id']}",
            "kind": "fit_cutout",
            "ok": bool(ok),
            "got": 1.0 if ok else 0.0,
            "note": where_ok if ok else where_bad,
        })

    return results


def check_fit(mesh, spec):
    """Verify component fit of `mesh` against `spec` (see module docstring).

    Returns {"pitchMm": p, "results": [{id, kind, ok, got?, note}, ...]}.
    `ok: null` = that check could not be evaluated (reported honest-not-run
    upstream). Raises only on a completely unusable spec/mesh — the caller
    (_run_checks) converts that into {"error": ...}.
    """
    comps = list(spec.get("components") or [])
    if not comps:
        return {"pitchMm": None, "results": []}
    if not bool(mesh.is_watertight):
        # Occupancy classification needs a closed surface; report honestly.
        return {
            "pitchMm": None,
            "results": [],
            "error": "fit checks need a watertight mesh; run them after repair",
        }
    pitch = _pitch_for(mesh, comps)
    grid = _AirGrid(mesh, pitch)
    results = []
    for comp in comps:
        try:
            results.extend(_check_component(grid, comp))
        except Exception as err:  # noqa: BLE001 — isolate per component
            cid = str(comp.get("id") or "component")
            results.append({
                "id": f"fit:{cid}:cavity",
                "kind": "fit_cavity",
                "ok": None,
                "note": f"fit check failed to run: {err}",
            })
    return {"pitchMm": pitch, "results": results}
