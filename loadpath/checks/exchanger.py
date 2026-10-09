"""
Dual-fluid TPMS heat-exchanger generator: a rectangular sheet-TPMS core (two
isolated fluid labyrinths sharing every wall — maximum exchange area per
volume) wrapped in a solid skin, with per-fluid plenum chambers and port
bores. The manifold problem this solves: every planar cut of a TPMS exposes
BOTH labyrinths, so each fluid's plenum face must seal the other fluid's
openings. `dual_sheet` does that by ramp-swallowing the other labyrinth into
solid near the face (over ~half a cell), which keeps the isolation guarantee:
the solid always contains the midsurface, so the circuits can never connect.

Flow arrangements (set purely by port placement — the labyrinths themselves
interpenetrate everywhere):
  "cross":   fluid A along X (plenums on the two X faces), fluid B along Y.
  "counter": both fluids along X; each X face splits into an upper (A, z>0)
             and lower (B, z<0) plenum with a solid divider between them.

Verification: `check_networks(mesh, ports)` (networks.py) flood-fills the
voids and proves the two circuits are isolated at voxel resolution. Scripts
run by the sidecar can assign the returned ports to `fluid_ports` and the
check runs automatically on the produced mesh.

Print notes: the off-fluid's channel stubs dead-end near each sealed face and
will hold powder/resin after printing; drain holes are deliberately NOT
provided (each one would be a leak), so prefer a process you can flush through
the ports — SLS/MJF powder or low-viscosity resin — and plan a flush cycle. Wall thickness: the thinnest point of a TPMS sheet
runs ~12% under nominal — keep `wall` >= 1.15x the process minimum
(>= ~0.5mm metal LPBF, >= ~1.0mm polymer).
"""
import numpy as np

from itertools import permutations

from sdf_kit import (
    box, capsule, dual_sheet, mask, seal_ramp, smin, subtract, union,
)

__all__ = ["exchanger_core"]


def _require(cond, msg):
    """Self-explanatory refusals the repair loop can act on (to_mesh style)."""
    if not cond:
        raise ValueError(msg)


def exchanger_core(size, cell=8.0, wall=1.0, kind="gyroid", flow="cross",
                   skin=None, plenum=None, port_r=None, seal_len=None,
                   margin=None, hose=None, blend=None, a_face=None,
                   b_face=None):
    """Build a complete dual-fluid TPMS heat-exchanger block.

    size:     (sx, sy, sz) mm — the TPMS core box, centered at the origin.
              The overall part grows beyond this by plenum + skin on the
              plenum faces and by skin elsewhere.
    cell:     TPMS lattice period, mm. 6-12mm prints and de-powders reliably.
    wall:     separator sheet thickness, mm (scalar or callable(P) for graded).
    kind:     "gyroid" (default; best all-round) | "schwarz_p" | "diamond".
    flow:     "cross" | "counter" (see module docstring).
    skin:     outer solid skin thickness, mm. Default max(wall, 1.2).
    plenum:   plenum chamber depth, mm. Default = cell.
    port_r:   port bore radius, mm. Default sized to the plenum window.
              Also accepts a dict for asymmetric ports — keys are port names
              ("a_in", "a_out", "b_in", "b_out") or fluid prefixes ("a",
              "b"); missing keys get the auto default. E.g.
              {"a_in": 6.0, "a_out": 5.0, "b": 3.0}.
    seal_len: depth over which the off-fluid labyrinth seals shut at each
              plenum face, mm. Default max(0.35*cell, 2.5*wall) — deep
              enough to close the labyrinth throat and stay several walls
              thick under pressure, without the old cell/2 default's
              visibly massive end caps (dead volume, dead weight).
    margin:   solid margin separating plenums from each other / the skin
              edges, mm. Default max(wall, skin).
    blend:    smooth-union radius (mm) where hood manifolds meet the body —
              the exterior fillet that makes the loft read as one form
              instead of a pipe glued to a box. Default adapts per hood
              (~1/4 of its hose radius, floored at 2mm) so big hoods get a
              proportionate fillet; a number applies uniformly; 0 disables.
              Exterior-only: channels stay sharp, so it can only ADD
              material and can never open a leak.
    a_face:   (w, h) mm — the DIMENSIONS of the two faces fluid A's ports
              must sit on (order-insensitive), straight from the spec.
              Validated against the axis layout; a mismatched `size`
              ordering is refused with the corrected ordering in the
              message. The cheap insurance against the recurring bug of
              hot ports landing on the wrong faces.
    b_face:   same for fluid B.
    hose:     external hood-loft hose manifolds, keyed like port_r. Each
              value is the hose INNER diameter (mm) or a dict
              {"d": mm, "neck": mm, "barb": mm}. A hooded port opens its
              ENTIRE rectangular plenum window through the skin (no small
              bore — the header collects the whole core face) and lofts
              that rectangle down to the round hose bore with a cubic-eased
              (smoothstep = cubic-Bezier basis) cross-section morph: zero
              slope leaving the face and zero slope arriving at the hose
              stub, so the area contraction is gradual — the implicit
              equivalent of a Bezier-swept loft, not a cone. Then a ringed
              barb run for the hose to clamp onto. Wall = skin throughout.
              Ports without an entry stay plain bores. E.g.
              hose={"a_in": 63.0, "a_out": 50.0, "b": 12.7}.
              Prefer this over hand-rolled manifolds: the axis/sign/face
              math comes from the port layout, which kills the classic bug
              of placing the "-X" manifold at +X and boring a leak path
              straight through the core.

    Returns (field, ports, plugs, bounds):
      field(P) -> (N,) sdf for `to_mesh(field, *bounds, pitch<=wall/3)`,
      ports    -> [{"name": "a_in"|"a_out"|"b_in"|"b_out", "point", "r"}, ...]
                  probe spheres for check_networks,
      plugs    -> [{"a", "b", "r"}, ...] — the port bores, for check_networks
                  to virtually refill (open ports otherwise read as a leak
                  through the outside air),
      bounds   -> (lo, hi) meshing bounds with clearance.

    Sidecar scripts get automatic isolation verification by assigning:
        field, ports, plugs, bounds = exchanger_core(...)
        result = to_mesh(field, *bounds, pitch=...)
        fluid_ports, fluid_plugs = ports, plugs
    """
    sx, sy, sz = (float(v) for v in size)
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    cell = float(cell)
    skin = max(float(wall), 1.2) if skin is None else float(skin)
    plenum = cell if plenum is None else float(plenum)
    wall_ref = float(wall) if not callable(wall) else 1.0
    # Seal caps: deep enough to close the off-fluid's labyrinth throat
    # (~1/3 cell) and stay a few walls thick under pressure — the old
    # cell/2 default read as a massive solid block at each sealed face.
    seal_len = (max(0.35 * cell, 2.5 * wall_ref)
                if seal_len is None else float(seal_len))
    m = max(wall_ref, skin) if margin is None else float(margin)
    # Hood-to-body fillet. None -> adaptive per hood (~1/4 of its hose
    # radius, floored at 2mm): a fixed skin-scale radius read as a hard edge
    # on big hoods — a 63mm hood on a 270mm part needs ~8mm of blend to look
    # lofted, while a 12.7mm stub wants ~2-3mm. An explicit number applies
    # uniformly; 0 disables.
    blend_explicit = blend is not None
    blend = 0.0 if blend is None else float(blend)

    # Orientation contract: when the caller states which faces each fluid's
    # ports live on (dimensions straight from the spec), refuse a `size`
    # ordering that puts them elsewhere — with the corrected ordering in the
    # message. Fluid A's plenums sit on the ±X faces (dims sy x sz); B's on
    # ±Y (cross: dims sx x sz) or also ±X (counter).
    if a_face is not None or b_face is not None:
        dims = (sx, sy, sz)

        def _matches(order):
            s = [dims[i] for i in order]
            a_dims = sorted((s[1], s[2]))
            b_dims = sorted((s[1], s[2]) if flow == "counter" else (s[0], s[2]))
            ok_a = a_face is None or all(
                abs(x - y) < 1e-6
                for x, y in zip(a_dims, sorted(float(v) for v in a_face)))
            ok_b = b_face is None or all(
                abs(x - y) < 1e-6
                for x, y in zip(b_dims, sorted(float(v) for v in b_face)))
            return ok_a and ok_b

        if not _matches((0, 1, 2)):
            fix = next(
                (tuple(dims[i] for i in order)
                 for order in permutations(range(3)) if _matches(order)),
                None,
            )
            cur_a = f"{sy:g}x{sz:g}"
            _require(False, (
                f"size {sx:g}x{sy:g}x{sz:g} puts fluid A's ports on the "
                f"{cur_a} faces, but a_face={a_face} was requested"
                + (f" — reorder size to "
                   f"{fix[0]:g}x{fix[1]:g}x{fix[2]:g} (A's plenums sit on "
                   f"the ±X faces, B's on ±Y)" if fix else
                   " — no ordering of these dimensions matches; re-check "
                   "the face sizes against the core dimensions")))

    _require(flow in ("cross", "counter"),
             f'flow must be "cross" or "counter", got {flow!r}')
    _require(min(sx, sy, sz) >= 1.5 * cell,
             f"core {sx:g}x{sy:g}x{sz:g}mm is under 1.5 cells ({cell:g}mm) on "
             f"its smallest side — no room for labyrinths; shrink `cell` or "
             f"grow the core")
    _require(plenum >= 2.0, f"plenum depth {plenum:g}mm is too shallow to "
             f"act as a header — use >= 2mm")

    # Plenum cavity windows (per flow). Each plenum entry carries its cavity
    # box, the port-bore axis, and the bore's inner mouth (where the tube
    # through the skin begins — also where the verification plug starts).
    if flow == "cross":
        _require(hy - m > 1.0 and hz - m > 1.0 and hx - m > 1.0,
                 f"margin {m:g}mm leaves no plenum window on a "
                 f"{sx:g}x{sy:g}x{sz:g}mm core")
        outer = (hx + plenum + skin, hy + plenum + skin, hz + skin)
        wy, wz = hy - m, hz - m  # A-window half-extents
        wxb, wzb = hx - m, hz - m  # B-window half-extents
        # (name, plenum center, plenum half, probe point, window min half-dim)
        # Probes sit at the diagonal corner of the window, OFF the bore axis,
        # so the verification plug (a sphere cap reaching in from the bore's
        # inner mouth) can never swallow them.
        layout = [
            ("a_in", (hx + 0.5 * plenum, 0.0, 0.0), (0.5 * plenum, wy, wz),
             (hx + 0.5 * plenum, 0.5 * wy, 0.5 * wz), min(wy, wz)),
            ("a_out", (-hx - 0.5 * plenum, 0.0, 0.0), (0.5 * plenum, wy, wz),
             (-hx - 0.5 * plenum, 0.5 * wy, 0.5 * wz), min(wy, wz)),
            ("b_in", (0.0, hy + 0.5 * plenum, 0.0), (wxb, 0.5 * plenum, wzb),
             (0.5 * wxb, hy + 0.5 * plenum, 0.5 * wzb), min(wxb, wzb)),
            ("b_out", (0.0, -hy - 0.5 * plenum, 0.0), (wxb, 0.5 * plenum, wzb),
             (0.5 * wxb, -hy - 0.5 * plenum, 0.5 * wzb), min(wxb, wzb)),
        ]
    else:  # counter
        _require(hz - 2.0 * m > 1.0 and hy - m > 1.0,
                 f"margin {m:g}mm leaves no split plenum window on a "
                 f"{sx:g}x{sy:g}x{sz:g}mm core")
        outer = (hx + plenum + skin, hy + skin, hz + skin)
        zc = 0.5 * hz  # centers of the upper (A) / lower (B) half-windows
        zh = 0.5 * (hz - 2.0 * m)  # their half-heights (divider = 2m at z=0)
        wy = hy - m
        # Counterflow: A runs +X -> -X while B runs -X -> +X.
        layout = [
            ("a_in", (hx + 0.5 * plenum, 0.0, zc), (0.5 * plenum, wy, zh),
             (hx + 0.5 * plenum, 0.5 * wy, zc + 0.5 * zh), min(wy, zh)),
            ("a_out", (-hx - 0.5 * plenum, 0.0, zc), (0.5 * plenum, wy, zh),
             (-hx - 0.5 * plenum, 0.5 * wy, zc + 0.5 * zh), min(wy, zh)),
            ("b_in", (-hx - 0.5 * plenum, 0.0, -zc), (0.5 * plenum, wy, zh),
             (-hx - 0.5 * plenum, 0.5 * wy, -zc - 0.5 * zh), min(wy, zh)),
            ("b_out", (hx + 0.5 * plenum, 0.0, -zc), (0.5 * plenum, wy, zh),
             (hx + 0.5 * plenum, 0.5 * wy, -zc - 0.5 * zh), min(wy, zh)),
        ]

    plenum_boxes = [(center, half) for _, center, half, _, _ in layout]

    # Geometry each port needs, independent of port_r: bore axis, the tube's
    # inner mouth (on the plenum's outer wall), the off-axis probe, and the
    # probe->mouth clearance that bounds how far a verification plug may
    # reach back into the plenum without swallowing the probe.
    resolved = []
    min_budget = np.inf
    for name, center, half, probe, wmin in layout:
        c = np.asarray(center, float)
        axis = np.array([np.sign(c[0]), 0.0, 0.0]) if abs(c[0]) > hx else \
            np.array([0.0, np.sign(c[1]), 0.0])
        mouth = c + axis * 0.5 * plenum
        probe_r = max(0.6, min(2.5, 0.3 * plenum, 0.45 * wmin))
        gap = float(np.linalg.norm(np.asarray(probe, float) - mouth))
        budget = gap - probe_r - 1.2
        resolved.append((name, c, axis, mouth, probe, probe_r, wmin, budget))
        min_budget = min(min_budget, budget)

    # External hood-loft manifolds (opt-in via `hose`). All axis/sign/face
    # math derives from the port layout — the hand-rolled version of this
    # once placed an "out" manifold on the wrong face and bored a leak path
    # through the core. Wall thickness through the hood is constant (= skin)
    # because the outer shell is the channel morph offset outward by skin.
    # Profile constants (_RING_*) live at module level with the _hood_*
    # helpers.
    barbs = []
    if hose:
        _require(isinstance(hose, dict),
                 'hose must be a dict like {"a_in": 63.0, "b": 12.7}')
        # Window half-extents per port (cross-section of its plenum window,
        # perpendicular to its axis) — the hood's rectangular end.
        window_of = {}
        for lname, center, half, _probe, _wmin in layout:
            cvec = np.asarray(center, float)
            li = int(np.argmax(np.abs(cvec) - np.asarray([hx, hy, hz])))
            cross = [j for j in range(3) if j != li]
            window_of[lname] = (half[cross[0]], half[cross[1]])
        for name, c, axis, mouth, probe, probe_r, _wmin, _budget in resolved:
            spec = hose.get(name, hose.get(name.split("_")[0]))
            if spec is None:
                continue
            if isinstance(spec, dict):
                d = float(spec.get("d", 0.0))
                neck = spec.get("neck")
                barb_len = spec.get("barb")
            else:
                d, neck, barb_len = float(spec), None, None
            _require(d > 2.0,
                     f'hose["{name}"] diameter {d:g}mm is not a usable hose '
                     f"bore — pass the hose INNER diameter in mm")
            r1 = 0.5 * d
            i = int(np.argmax(np.abs(axis)))
            sign = float(np.sign(axis[i]))
            face = sign * outer[i]
            cross = [j for j in range(3) if j != i]
            win = window_of[name]
            # Neck long enough for a gentle, printable contraction from the
            # window's larger half-extent down to the hose bore — the eased
            # profile spends its steepest slope mid-neck, so give it room.
            neck = (max(8.0, 1.1 * max(max(win) - r1, abs(r1)))
                    if neck is None else float(neck))
            barb_len = 14.0 if barb_len is None else float(barb_len)
            _require(barb_len >= 6.0,
                     f'hose["{name}"] barb run {barb_len:g}mm is too short '
                     f"for a hose clamp — use >= 6mm")
            barbs.append({
                "name": name, "i": i, "sign": sign, "face": face,
                "c": c, "win": win, "cross": cross,
                "r1": r1, "neck": neck, "barb": barb_len,
            })
        # Two barbs sharing a face (counterflow splits each X face between
        # the fluids) must not overlap — merged necks would join the two
        # circuits OUTSIDE the core, a leak no seal can prevent.
        for ai in range(len(barbs)):
            for bi in range(ai + 1, len(barbs)):
                A, B = barbs[ai], barbs[bi]
                if A["i"] != B["i"] or A["sign"] != B["sign"]:
                    continue
                dist = float(np.linalg.norm(A["c"] - B["c"]))
                need = (A["r1"] + skin + _RING_AMP) + (B["r1"] + skin + _RING_AMP)
                _require(
                    dist >= need + 1.0,
                    f'hose barbs "{A["name"]}" and "{B["name"]}" share a face '
                    f"and would merge ({dist:.1f}mm apart, need >= "
                    f"{need + 1.0:.1f}mm) — merged manifolds join the two "
                    f"circuits outside the core; shrink the hose diameters or "
                    f"grow the core")
    hooded = {b["name"] for b in barbs}

    max_r = 0.8 * min(w for _, _, _, _, w in layout)
    auto_r = max(0.8, min(0.6 * plenum, 0.6 * max_r, min_budget))
    if isinstance(port_r, dict):
        # Asymmetric ports: per-name (or per-fluid-prefix) radii, validated
        # against each port's own window and probe budget. Hooded ports have
        # no bore at all (their whole window opens into the hood), so any
        # port_r entry for them is ignored rather than validated against a
        # bore that will never exist.
        per_port_r = {}
        for name, _c, _axis, _mouth, _probe, _pr, wmin_p, budget_p in resolved:
            if name in hooded:
                per_port_r[name] = auto_r
                continue
            r = port_r.get(name, port_r.get(name.split("_")[0]))
            if r is None:
                r = auto_r
            else:
                r = float(r)
                _require(
                    r <= 0.8 * wmin_p,
                    f'port_r["{name}"] {r:g}mm does not fit its plenum '
                    f"window — use <= {0.8 * wmin_p:.1f}mm (a bore can never "
                    f"exceed the face's short side; for bigger hose sizes "
                    f"add an external lofted manifold that necks OUTSIDE "
                    f"the core)")
                _require(
                    r <= budget_p + 0.2,
                    f'port_r["{name}"] {r:g}mm leaves no room to verify '
                    f"isolation (the bore plug would swallow the port "
                    f"probe) — use <= {max(budget_p, 0.8):.1f}mm or a "
                    f"deeper plenum")
            per_port_r[name] = r
    elif port_r is None:
        per_port_r = {r[0]: auto_r for r in resolved}
    else:
        port_r = float(port_r)
        _require(port_r <= max_r,
                 f"port_r {port_r:g}mm does not fit the plenum window — "
                 f"use <= {max_r:.1f}mm")
        _require(port_r <= min_budget + 0.2,
                 f"port_r {port_r:g}mm leaves no room to verify isolation (the "
                 f"bore plug would swallow a port probe) — use <= "
                 f"{max(min_budget, 0.8):.1f}mm or a deeper plenum")
        per_port_r = {r[0]: port_r for r in resolved}


    # Port bores run outward along each plenum's axis, from mid-plenum through
    # the outer skin (ends past both surfaces so the round caps never show).
    # HOODED ports get no bore: their whole plenum window opens through the
    # skin into the hood instead, so a bore (and its capsule plug) would be
    # redundant geometry. Verification plugs: capsule per plain bore; box per
    # hood (covering the hood's full envelope from mid-skin to past the tip —
    # a capsule can't cap a rectangular window without bulging into the
    # plenum and swallowing the probes).
    ports, bores, plugs = [], [], []
    for name, c, axis, mouth, probe, probe_r, _wmin, _budget in resolved:
        ports.append({"name": name, "point": list(probe), "r": probe_r})
        if name in hooded:
            continue
        r = per_port_r[name]
        exit_pt = mouth + axis * (skin + 2.0)
        bores.append((c, exit_pt, r))
        plugs.append({"a": mouth.tolist(), "b": exit_pt.tolist(), "r": r})
    for b in barbs:
        i, sign = b["i"], b["sign"]
        half_len = 0.5 * (skin * 0.5 + b["neck"] + b["barb"] + 2.0)
        center = b["c"].copy()
        # Box spans from mid-skin (sealing inside the window tube, clear of
        # the plenum probes) out past the barb tip.
        center[i] = b["face"] - sign * skin * 0.5 + sign * half_len
        h = [0.0, 0.0, 0.0]
        h[i] = half_len
        cross_r = max(max(b["win"]), b["r1"]) + skin + _RING_AMP
        for j in b["cross"]:
            h[j] = cross_r
        plugs.append({"c": center.tolist(), "h": h})

    core_half = (hx, hy, hz)

    def field(P):
        P = np.asarray(P, float)
        if flow == "cross":
            # B's labyrinth seals shut near A's X faces and vice versa.
            s_b = seal_ramp(hx - np.abs(P[:, 0]), seal_len)
            s_a = seal_ramp(hy - np.abs(P[:, 1]), seal_len)
        else:
            # Near the X faces: B seals in the upper half (A's plenums), A in
            # the lower; both blend at z=0 into the solid divider band.
            face = seal_ramp(hx - np.abs(P[:, 0]), seal_len)
            s_b = face * seal_ramp(-P[:, 2], 0.5 * cell)
            s_a = face * seal_ramp(P[:, 2], 0.5 * cell)
        core = mask(
            dual_sheet(P, cell, wall, kind, seal_a=s_a, seal_b=s_b),
            box(P, (0.0, 0.0, 0.0), core_half),
        )
        # Everything between the core box and the outer box is solid until a
        # plenum or bore carves it (Kim & Yoo-style implicit booleans) — the
        # skin, plenum side walls, and corner blocks all come from this ring.
        ring = subtract(box(P, (0.0, 0.0, 0.0), outer),
                        box(P, (0.0, 0.0, 0.0), core_half))
        solid = union(core, ring)
        for center, half in plenum_boxes:
            solid = subtract(solid, box(P, center, half))
        for a, b, r in bores:
            solid = subtract(solid, capsule(P, a, b, r))
        # Hoods join with a smooth union (exterior fillet) so the loft reads
        # as one continuous form, not a pipe glued onto a box. Only ever ADDS
        # material around the junction, so it cannot open a leak; the channel
        # subtraction below stays sharp for exactly that reason.
        for b in barbs:
            hood = _barb_solid(P, b, skin)
            k = (blend if blend_explicit
                 else max(2.0, skin, 0.25 * b["r1"]))
            solid = smin(solid, hood, k) if k > 0 else union(solid, hood)
        for b in barbs:
            solid = subtract(solid, _barb_channel(P, b, skin))
        return solid

    pad = 1.0
    lo = [-(o + pad) for o in outer]
    hi = [o + pad for o in outer]
    for b in barbs:
        i, sign = b["i"], b["sign"]
        reach = abs(b["face"]) + b["neck"] + b["barb"] + 2.0 + pad
        if sign > 0:
            hi[i] = max(hi[i], reach)
        else:
            lo[i] = min(lo[i], -reach)
        r_out = b["r1"] + skin + _RING_AMP + pad
        for j in range(3):
            if j == i:
                continue
            hi[j] = max(hi[j], float(b["c"][j]) + r_out)
            lo[j] = min(lo[j], float(b["c"][j]) - r_out)
    bounds = (tuple(lo), tuple(hi))
    return field, ports, plugs, bounds


# Hood/barb geometry constants shared by the profile helpers below (defined
# here, not inside exchanger_core, so the helpers stay module-level and
# testable).
_RING_SPACING = 4.0
_RING_AMP = 1.0
_TIP_LEAD = 2.0


def _hood_frame(P, b):
    """(u, dy, dz, rr) for a hood: u = mm outward from its skin face along
    its port axis; (dy, dz) = cross offsets from the port centerline in the
    window's own axes; rr = radial distance from that centerline."""
    i, sign = b["i"], b["sign"]
    u = sign * (P[:, i] - b["face"])
    j, k = b["cross"]
    dy = P[:, j] - b["c"][j]
    dz = P[:, k] - b["c"][k]
    rr = np.sqrt(dy * dy + dz * dz)
    return u, dy, dz, rr


def _hood_section(u, dy, dz, rr, b):
    """Cross-section field of the hood CHANNEL at each point: the plenum
    window rectangle at the face morphing to the hose circle at the neck's
    end, blended with a cubic ease (smoothstep — the cubic-Bezier basis, so
    the section is tangent to the window at u=0 and tangent to the straight
    hose stub at u=neck: a swept loft, not a cone). Negative inside the
    channel. Inside the face (u<=0) it stays the pure window rectangle, so
    the subtraction just opens the window through the skin — it can never
    cut new paths into the labyrinths."""
    wy, wz = b["win"]
    t = np.clip(u / max(b["neck"], 1e-6), 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep: cubic ease, C1 at both ends
    d_rect = np.maximum(np.abs(dy) - wy, np.abs(dz) - wz)
    d_circ = rr - b["r1"]
    return (1.0 - s) * d_rect + s * d_circ


def _barb_solid(P, b, skin):
    """Outer solid of one hood manifold: the channel morph offset outward by
    `skin` (constant wall) through the neck, then a ringed hose-barb run
    with a chamfered lead-in tip."""
    u, dy, dz, rr = _hood_frame(P, b)
    neck, barb_len = b["neck"], b["barb"]
    shell = _hood_section(u, dy, dz, rr, b) - skin
    ub = u - neck
    in_barb = (u > neck) & (u <= neck + barb_len)
    # Sawtooth rings on the straight run: the hose slides on over the ramps
    # and grips against the sharp faces.
    ring = _RING_AMP * ((ub % _RING_SPACING) / _RING_SPACING)
    R1 = b["r1"] + skin
    shell = np.where(in_barb, rr - (R1 + ring), shell)
    shell = np.where(u > neck + barb_len - _TIP_LEAD, rr - (R1 - 0.6), shell)
    total = neck + barb_len
    return np.maximum(shell, np.maximum(u - total, -u))


def _barb_channel(P, b, skin):
    """Flow channel through one hood: the full plenum window just inside the
    face (opening the whole core face into the header), the eased rect-to-
    round morph over the neck, then the straight hose bore through the barb
    run and out past the tip."""
    u, dy, dz, rr = _hood_frame(P, b)
    d = _hood_section(u, dy, dz, rr, b)
    total = b["neck"] + b["barb"]
    return np.maximum(
        d, np.maximum(u - (total + 2.0), -(u + skin + 2.0))
    )
