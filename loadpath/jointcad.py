"""
Exact B-rep solids for the joints joints.py sizes: threads, hinges, snaps.

Each function takes a design dict (joints.thread / hinge / snap) and
returns OpenCascade solids through build123d, drawn in the joint's own
frame (the axis along +z for threads and hinges, the beam along +x for
snaps) and placed with `at` (a build123d Location, or (origin, axis)).
The solids are what `loadpath build` adds and cuts exactly, and what a
mating part (a screw, a pin) is exported as.

Threads are a helical sweep of the drawn tooth (joints.thread_profile)
with its binormal held along the axis, so every turn is the same exact
tooth: no faceting, no stacked rings. The nut's thread is a cut tool: the
screw's tooth grown by the gap all round, swept on the same helix, with a
45 degree lead-in at each end so the first turn starts clean. A screw
printed lying gets a flat at its minor radius to stand on.

Hinges: the knuckles of either member (add), the bore through them (cut:
the pin's press fit in the outer knuckles, its running gap in the middle
ones), the room each member's knuckles turn in (cut from the other), the
pin. Printed in place, the pin is the outer member's, and the middle
knuckles' bore has a 45 degree teardrop roof so it prints unsupported.

Snaps: the beam with its barb (add), the slot it bends into (cut), and
the window or ledge on the mating part the barb catches (cut).

Needs build123d (pip install build123d).
"""
from __future__ import annotations

import math

import numpy as np

FLAT_DEPTH = 0.3     # mm a lying screw's flat cuts under its minor radius (joints.FLAT_DEPTH)

__all__ = ["hinge_solids", "nut_tool", "place", "screw_solid", "snap_solids", "thread_solids"]


def _bd():
    import build123d as bd

    return bd


def _one(x):
    """build123d booleans return a ShapeList when the result has pieces:
    one piece as itself, several as a compound."""
    bd = _bd()
    if isinstance(x, (list, tuple)) or type(x).__name__ == "ShapeList":
        xs = list(x)
        return xs[0] if len(xs) == 1 else bd.Compound(xs)
    return x


def _u(a, b):
    return _one(a.fuse(b))


def _c(a, b):
    return _one(a.cut(b))


def _i(a, b):
    return _one(a.intersect(b))


def place(solid, at=None):
    """Move a solid drawn in the joint's frame to `at`: a build123d
    Location, a (origin, axis) pair (the frame's +z onto axis), or None."""
    bd = _bd()
    if at is None:
        return solid
    if isinstance(at, bd.Location):
        return solid.moved(at)
    origin, axis = at
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    pl = bd.Plane(origin=tuple(map(float, origin)), x_dir=tuple(map(float, x)), z_dir=tuple(map(float, z)))
    return solid.moved(bd.Location(pl))


# ---------------------------------------------------------------------------
# threads


def _tooth(prof, grow=0.0, under=0.3):
    """The tooth outline in (r, z), closed below its root by `under` mm so
    it overlaps the core it is fused to. grow: the nut's tool, every face
    moved out by `grow` (the flanks along their normals, mitred at the
    crest), standing on the nut's minor radius."""
    R, rmin = prof["screwMajor"], prof["screwMinor"]
    c = prof["crest"]
    bl, bb = (math.radians(a) for a in prof["flanks"])
    tl, tb = math.tan(bl), math.tan(bb)
    g = float(grow)
    # z along the axis (toward the screw's tip), r out; the loaded flank faces -z (the head)
    zl = lambda r: -c / 2 - (R - r) * tl - g / math.cos(bl)
    zb = lambda r: c / 2 + (R - r) * tb + g / math.cos(bb)
    r_lo, r_hi = rmin + g, R + g
    # the flanks run on into the core, so they cross its surface rather than
    # meet it along an edge (a tangent seam the kernel's fuse leaves folded),
    # but not so far that one turn's root reaches the next's
    u = min(under, 0.4 * (prof["pitch"] - (zb(r_lo) - zl(r_lo))) / (tl + tb))
    r0 = r_lo - u
    return [(r0, zl(r0)), (r_hi, zl(r_hi)), (r_hi, zb(r_hi)), (r0, zb(r0))]


def _sweep(pts, pitch, z0, length, r_path, left=False):
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.gp import gp_Dir

    bd = _bd()
    w = bd.Wire.make_polygon([bd.Vector(r, 0, z + z0) for r, z in pts], close=True)
    helix = bd.Helix(pitch=pitch, height=length, radius=r_path, center=(0, 0, z0), lefthand=left)
    ps = BRepOffsetAPI_MakePipeShell(bd.Wire([helix]).wrapped)
    ps.SetMode(gp_Dir(0, 0, 1))                 # binormal along the axis: every turn the same tooth
    ps.Add(w.wrapped, False, False)
    ps.Build()
    ps.MakeSolid()
    return bd.Solid(ps.Shape())


def _revolved(rz):
    """A solid of revolution about z of the (r, z) outline."""
    bd = _bd()
    w = bd.Wire.make_polygon([bd.Vector(r, 0, z) for r, z in rz], close=True)
    return bd.Solid.revolve(bd.Face(w), 360, bd.Axis.Z)


# OpenCascade's fuse of a helical sweep and a cylinder drops one operand
# outright when the cylinder's seam lies on the sweep's start (2026-11-24:
# a 10 mm thread came back as its core alone, a 20 mm one as its teeth
# alone). Turning the round solids about their own axis moves the seam and
# changes nothing else, so each union is tried at these seam angles and
# kept at the first that adds up.
SEAMS = (101.0, 37.0, 233.0, 167.0, 0.0)


def _sane_union(u, *parts) -> bool:
    if u is None:
        return False
    v = float(u.volume)
    vs = [float(p.volume) for p in parts if p is not None]
    return v > max(vs) * (1 + 1e-6) + 1e-6 and v < sum(vs) * (1 + 1e-6) + 1e-6


def _threaded(rounds, tooth, extra=(), what="thread"):
    """core U (tooth ^ envelope) U extra..., with the round solids (core,
    envelope, extras: callables of the seam angle) turned until every
    union adds up."""
    bd = _bd()
    for ang in SEAMS:
        core, env = (f().rotate(bd.Axis.Z, ang) for f in rounds)
        x = _i(tooth, env)
        out = _u(core, x)
        ok = _sane_union(out, core, x)
        for f in extra:
            if not ok:
                break
            e = f().rotate(bd.Axis.Z, ang)
            nxt = _u(out, e)
            ok = _sane_union(nxt, out, e)
            out = nxt
        if ok:
            return out
    raise RuntimeError(f"the {what}'s solid did not fuse cleanly at any seam angle; "
                       "try a slightly different pitch or diameter")


def screw_solid(design: dict, length: float, tip: bool = True, flat: bool | None = None, left: bool = False):
    """The screw's threaded shank from z = 0 (the head's face) to z =
    length: core, swept tooth and a 45 degree chamfer at the tip (to the
    minor radius). flat: a flat FLAT_DEPTH under the minor radius along -y
    to print it lying on (default: when the design was checked lying)."""
    bd = _bd()
    prof = design["profile"]
    P = prof["pitch"]
    R, rmin = prof["screwMajor"], prof["screwMinor"]
    pts = _tooth(prof)
    th = _sweep(pts, P, -P, length + 2 * P, pts[0][0], left)
    # trim: the length, its tip chamfered 45 degrees down to the minor radius
    e = R + 1.0
    tipz = length - (e - rmin) if tip else length
    out = _threaded((lambda: bd.Solid.make_cylinder(rmin + 0.02, length),
                     lambda: _revolved([(0.0, 0.0), (e, 0.0), (e, tipz), (rmin if tip else e, length),
                                        (0.0, length)])), th, what="screw")
    if flat is None:
        b = (design.get("inputs") or {}).get("screwBuild")
        ax = (design.get("inputs") or {}).get("axis") or [0, 0, 1]
        flat = b is not None and abs(float(np.dot(np.asarray(b, float) / np.linalg.norm(b),
                                                    np.asarray(ax, float) / np.linalg.norm(ax)))) < 0.3
    if flat:
        # the flat cuts into the core, never tangent to it (a tangent cut leaves the kernel a sliver)
        yf = rmin - FLAT_DEPTH
        cut = _c(out, bd.Solid.make_box(4 * e, 2 * e, length + 2).moved(bd.Location((-2 * e, -2 * e - yf, -1))))
        if cut is None or not (0.5 * out.volume < cut.volume < out.volume):
            raise RuntimeError("the screw's flat did not cut cleanly")
        out = cut
    return out


def nut_tool(design: dict, length: float, lead_in: bool = True, left: bool = False, overrun: float = 1.0):
    """The space the nut's thread is cut with, from z = -overrun to z =
    length + overrun: the screw's tooth grown by the gap, swept, over a
    core at the nut's minor radius, with a 45 degree countersink at each
    end from just outside the nut's major radius down to its minor."""
    bd = _bd()
    prof = design["profile"]
    P, g = prof["pitch"], prof["gap"]
    Rn, rn = prof["nutMajor"], prof["nutMinor"]
    z0, L = -overrun, length + 2 * overrun
    pts = _tooth(prof, grow=g)
    # the same helix as the screw's (its phase set at z = 0): start a whole number of turns down
    k = math.ceil(overrun / P) + 1
    th = _sweep(pts, P, -k * P, L + (k + 1) * P, pts[0][0], left)
    r_face = Rn + 0.3
    depth = r_face - rn
    extra = []
    if lead_in:
        extra = [lambda: bd.Solid.make_cone(r_face + overrun, rn, depth + overrun, bd.Plane.XY.offset(-overrun)),
                 lambda: bd.Solid.make_cone(rn, r_face + overrun, depth + overrun,
                                            bd.Plane.XY.offset(length - depth))]
    return _threaded((lambda: bd.Solid.make_cylinder(rn, L, bd.Plane.XY.offset(z0)),
                      lambda: bd.Solid.make_cylinder(Rn + 1.0, L, bd.Plane.XY.offset(z0))), th, extra, what="nut")


def thread_solids(design: dict, length: float | None = None, at=None) -> dict:
    """{'screw': the threaded shank, 'nut': the nut's cut tool} for a
    thread design, `length` mm of thread (default the design's engagement),
    placed at `at`."""
    L = float(length or design["engage"])
    return {"screw": place(screw_solid(design, L), at), "nut": place(nut_tool(design, L), at)}


# ---------------------------------------------------------------------------
# hinges


def _teardrop(r, length, z0=0.0):
    """A bore of radius r along z with a 45 degree roof along +y."""
    bd = _bd()
    cyl = bd.Solid.make_cylinder(r, length, bd.Plane.XY.offset(z0))
    s = r / math.sqrt(2.0)
    with bd.BuildSketch(bd.Plane.XY.offset(z0)) as sk:
        bd.Polygon((-s, s), (0.0, r * math.sqrt(2.0)), (s, s), (0.0, 0.0), align=None)
    roof = bd.extrude(sk.sketch, amount=length)
    return _u(cyl, roof)


def hinge_solids(design: dict, at=None, start: float | None = None, up=(0, 1, 0)) -> dict:
    """The hinge along +z from `start` (default centred on z = 0):

    - outer / inner: the knuckles of the outer member (the two ends and
      every other one) and the middle member, as cylinders of the design's
      outer diameter;
    - outer_room / inner_room: the room each member's knuckles turn in,
      to cut from the other member's body (knuckle + gap all round);
    - outer_bore / inner_bore: the bores (pinned: the press fit in the
      outer knuckles, the running gap in the middle ones; printed in
      place: none in the outer, a teardrop bore with its roof up `up` in
      the middle ones);
    - pin: the pin (pinned: its own part, chamfered; printed in place: the
      length through the middle knuckles, to add to the outer member)."""
    bd = _bd()
    n = int(design["knuckles"])
    W, d = float(design["width"]), float(design["d"])
    ga = float(design["axialGap"])
    t_in, t_out = float(design["knuckle"]), float(design["outerKnuckle"])
    Ro = 0.5 * float(design["outer"])
    z = -0.5 * W if start is None else float(start)
    segs = []
    for i in range(n):
        t = t_out if i % 2 == 0 else t_in
        segs.append((i % 2 == 0, z, t))
        z += t + ga

    def cyl(r, z0, t):
        return bd.Solid.make_cylinder(r, t, bd.Plane.XY.offset(z0))

    def union(xs):
        out = xs[0]
        for x in xs[1:]:
            out = _u(out, x)
        return out

    outer = union([cyl(Ro, z0, t) for o, z0, t in segs if o])
    inner = union([cyl(Ro, z0, t) for o, z0, t in segs if not o])
    room_o = union([cyl(Ro + ga, z0 - ga, t + 2 * ga) for o, z0, t in segs if o])
    room_i = union([cyl(Ro + ga, z0 - ga, t + 2 * ga) for o, z0, t in segs if not o])
    z_lo, z_hi = segs[0][1], segs[-1][1] + segs[-1][2]
    rb = 0.5 * float(design["bore"])
    if design["kind"] == "print-in-place":
        # the roof points along `up` rotated into the frame's +y
        inner_bore = _teardrop(rb, z_hi - z_lo + 2.0, z_lo - 1.0)
        ang = math.degrees(math.atan2(-float(up[0]), float(up[1]))) if up is not None else 0.0
        inner_bore = inner_bore.rotate(bd.Axis.Z, ang)
        outer_bore = None
        pin = cyl(0.5 * d, z_lo + 0.01, z_hi - z_lo - 0.02)
    else:
        inner_bore = cyl(rb, z_lo - 1.0, z_hi - z_lo + 2.0)
        outer_bore = cyl(0.5 * float(design["press"]), z_lo - 1.0, z_hi - z_lo + 2.0)
        L = z_hi - z_lo - 0.2
        pin = bd.Solid.make_cylinder(0.5 * d, L, bd.Plane.XY.offset(z_lo + 0.1))
        ch = min(0.5, 0.15 * d)
        pin = bd.chamfer(pin.edges(), ch)
    out = {"outer": outer, "inner": inner, "outer_room": room_o, "inner_room": room_i,
           "outer_bore": outer_bore, "inner_bore": inner_bore, "pin": pin}
    return {k: (None if v is None else place(v, at)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# snaps


def snap_solids(design: dict, at=None, gap: float = 0.3) -> dict:
    """A cantilever snap along +x from its root at x = 0, bending along
    +z (the barb proud on +z), width along y:

    - beam: the beam and its barb (lead face at the design's lead angle,
      return face at its return angle);
    - slot: the room the beam needs to bend (cut from its own part
      round the beam, `gap` each side, its deflection under it);
    - catch: the window in the mating part the barb drops into (cut)."""
    bd = _bd()
    L, t, b, y = (float(design[k]) for k in ("L", "t", "b", "undercut"))
    lead, ret = math.radians(float(design["lead"])), math.radians(float(design["return"]))
    beam = bd.Solid.make_box(L, b, t, bd.Plane.XY.offset(0).shift_origin((0, 0, 0))).moved(
        bd.Location((0, -b / 2, -t)))
    run_lead = y / math.tan(lead)
    run_ret = y / math.tan(ret) if ret < math.radians(89.9) else 0.0
    x0 = L - run_lead - run_ret - 0.5
    prof = [(x0, 0.0), (x0 + run_ret, y), (x0 + run_ret + 0.5, y), (L, 0.0)]
    with bd.BuildSketch(bd.Plane.XZ) as sk:
        bd.Polygon(*prof, align=None)
    barb = bd.extrude(sk.sketch, amount=b / 2, both=True)
    beam = _u(beam, barb)
    slot = bd.Solid.make_box(L + gap, b + 2 * gap, t + y + 2 * gap).moved(
        bd.Location((0, -b / 2 - gap, -t - y - gap)))
    catch = bd.Solid.make_box(run_ret + 0.5 + 2 * gap + run_lead, b + 2 * gap, y + gap).moved(
        bd.Location((x0 - gap, -b / 2 - gap, 0.0)))
    out = {"beam": beam, "slot": slot, "catch": catch}
    return {k: place(v, at) for k, v in out.items()}


# ---------------------------------------------------------------------------
# a spec's joint entry, drawn


def _frame(origin, z, x=None):
    """A build123d Location with +z along `z` (and +x along `x`)."""
    bd = _bd()
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    if x is None:
        ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(ref, z)
    x = np.asarray(x, float)
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    return bd.Location(bd.Plane(origin=tuple(map(float, origin)), x_dir=tuple(map(float, x)),
                                z_dir=tuple(map(float, z))))


def joint_interfaces(entry: dict, design: dict, build=None) -> tuple[list, dict]:
    """The interfaces a `joints` entry puts on the part being built, and
    the mating solids to export: ([(suffix, op, solid, wall, blend,
    proud)], {name: solid}).

    entry["member"] says which side this part is: a thread's "nut" or
    "screw", a hinge's "outer" or "inner", a snap's "hook" or "catch"."""
    kind = design["type"]
    at = entry.get("at", [0.0, 0.0, 0.0])
    proud = bool(entry.get("proud", False))
    blend = float(entry.get("blend", 1.0))
    member = entry.get("member")
    if kind == "thread":
        member = member or "nut"
        loc = _frame(at, entry.get("axis", [0, 0, 1]))
        L = float(entry.get("length") or design["engage"])
        screw = screw_solid(design, L)
        if member == "nut":
            tool = nut_tool(design, L)
            wall = max(0.0, 0.5 * float(design["boss"]) - float(design["profile"]["nutMajor"]))
            return [("thread", "cut", tool.moved(loc), wall, blend, False)], {"screw": screw.moved(loc)}
        if member == "screw":
            return [("thread", "add", screw.moved(loc), 0.0, blend, proud)], {"nut-tool": nut_tool(design, L).moved(loc)}
        raise ValueError(f"joint {entry.get('id')}: a thread's member is 'nut' or 'screw', got {member!r}")
    if kind == "hinge":
        member = member or "outer"
        axis = entry.get("axis", [1, 0, 0])
        up = None
        if build is not None:
            # the teardrop's roof up the build, in the hinge's frame
            ax = np.asarray(axis, float) / np.linalg.norm(axis)
            b = np.asarray(build, float)
            b = b - (b @ ax) * ax
            up = b if np.linalg.norm(b) > 1e-6 else None
        loc = _frame(at, axis, up)
        # the frame's +x is the build's up: the teardrop's roof points there
        hs = hinge_solids(design, up=(1.0, 0.0, 0.0) if up is not None else (0.0, 1.0, 0.0))
        mv = {k: (None if v is None else v.moved(loc)) for k, v in hs.items()}
        if member == "outer":
            if design["kind"] == "print-in-place":
                # the pin is this member's, through the other's room (cuts come after adds)
                return [("knuckles", "add", mv["outer"], 0.0, blend, proud),
                        ("pin", "add", mv["pin"], 0.0, 0.0, True),
                        ("room", "cut", _c(mv["inner_room"], mv["pin"]), 0.0, 0.0, False)], {}
            out = [("knuckles", "add", mv["outer"], 0.0, blend, proud), ("room", "cut", mv["inner_room"], 0.0, 0.0, False)]
            out.append(("bore", "cut", mv["outer_bore"], 0.0, 0.0, False))
            return out, {"pin": hs["pin"].moved(loc)}
        if member == "inner":
            return ([("knuckles", "add", mv["inner"], 0.0, blend, proud),
                     ("room", "cut", mv["outer_room"], 0.0, 0.0, False),
                     ("bore", "cut", mv["inner_bore"], 0.0, 0.0, False)],
                    {} if design["kind"] == "print-in-place" else {"pin": hs["pin"].moved(loc)})
        raise ValueError(f"joint {entry.get('id')}: a hinge's member is 'outer' or 'inner', got {member!r}")
    if kind == "snap":
        member = member or "hook"
        loc = _frame(at, entry.get("deflectDir", [1, 0, 0]), entry.get("lengthDir", [0, 0, 1]))
        # the snap is drawn with its length along +x and its bend along +z
        ss = snap_solids(design)
        mv = {k: v.moved(loc) for k, v in ss.items()}
        if member == "hook":
            return [("slot", "cut", _c(mv["slot"], mv["beam"]), 0.0, 0.0, False),
                    ("beam", "add", mv["beam"], 0.0, blend, proud)], {}
        if member == "catch":
            return [("catch", "cut", mv["catch"], 0.0, 0.0, False)], {}
        raise ValueError(f"joint {entry.get('id')}: a snap's member is 'hook' or 'catch', got {member!r}")
    raise ValueError(f"joint type {kind!r}")
