"""
A real board in: a KiCad `.kicad_pcb` read into what an enclosure, a
stack or any study needs to know about it.

A board is not a box. Its outline is whatever the designer cut (Edge.Cuts,
with any slots and cut-outs), it is as thick as its stackup, its parts
stand up from both faces to their own heights, it is held by screws
through its mounting holes, its connectors face out over particular edges
and get pushed on, and some of its parts make heat. All of that is in the
board file, or in the 3D models it points to, so none of it should be
typed in again or guessed as a box:

    python -m loadpath kicad board.kicad_pcb --out out/ [--models DIR]

or, in an enclosure brief, a component that is the board:

    {"id": "board", "kicad": {"file": "board.kicad_pcb",
                               "connectors": {"J1": {"opening": "usb-port"}},
                               "power": {"U3": 0.6}, "maxTemp": {"U3": 85}},
     "rotate": "z", "margin": 0.3}

and in a stack brief, a board: `{"id": "fc", "kicad": "fc.kicad_pcb"}`.

What is read, and where each number comes from (every part says, in
`heightFrom`, `massFrom`, `wattsFrom`):

- **outline**: Edge.Cuts lines, arcs, circles, rectangles and polygons
  (on the board or inside footprints), chained into closed loops; the
  largest is the board, loops inside it are cut-outs.
- **thickness**: the stackup's layers when there is one, else
  `general.thickness`. Copper from the stackup (else 35 um per copper
  layer), FR-4 for the rest: the board's mass (FR-4 1850 kg/m3, copper
  8960 kg/m3 at `copperFill` of each layer, default 0.5) and its in-plane
  conductivity (copper 390 W/(m K) and FR-4 0.8 in parallel by thickness).
- **parts**: each footprint's body is its 3D model when the model file can
  be found (STEP through the `step` extra, or VRML), placed the way KiCad
  places it (the model's offset, scale and rotation, the footprint's
  position, rotation and side; a back-side model mirrored as the 3D viewer
  does); the convex hull of the model is the part's
  body. Without a model the body is the footprint's fab outline (the body
  as drawn; else its courtyard, else its pads) stood up to the part's height: a `Height`
  field on the footprint, else the package's standard maximum seated
  height read off its library name (JEDEC outlines, chip sizes, headers,
  common connectors; `_HEIGHTS` lists them with their sources), else
  `defaultHeight` (3 mm), flagged as assumed. Through-hole leads stand
  `leadProtrusion` (1.5 mm; IPC-A-610 allows up to 2.5) out of the other
  face.
- **mass**: a footprint's `Mass` field (g), else its body's volume at its
  family's effective density (moulded ICs 2.0 g/cm3, ceramic capacitors
  5.5, resistors 3.5, ...: estimates, flagged so).
- **mounting holes**: MountingHole footprints (and lone large drills): the
  centre, the drill, the copper pad the screw head and the standoff bear
  on, and the screw (from the name, else the largest ISO size that passes
  the drill). In an enclosure the board is held on them: a threaded
  sleeve under each hole and the screw's shank, head and driver kept
  clear (enclosure `mount.type: holes`), not a cradle under its outline.
- **connectors**: Connector and TerminalBlock footprints (and J/P
  references). Which way each faces: out over the board edge it overhangs
  or sits within `edgeReach` (2 mm) of, else up off its face for vertical
  ones, else from its contacts toward the far side of its body (flagged
  as a guess). USB-C, USB-A and micro-B are typed, so their plug loads
  (connectors.py) are written when the connector is aligned to an opening.
- **heat**: a board file has no power numbers. A footprint field `Power`
  (W or mW; also `Pd`, `Dissipation`, `Watts`) and `MaxTemp` (C), or the
  brief's `power` and `maxTemp` maps by reference, make a part a heat
  source; nothing is assumed, and a board with none says so.

Frame: millimetres, the origin at the centre of the outline's bounding
box, +x right and +y up as seen from the top (KiCad's y points down, so it
is flipped), z = 0 the board's underside, z = thickness its top face.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

__all__ = ["parse_sexpr", "read_board", "enclosure_components", "stack_board", "write_board"]

# --- the file -----------------------------------------------------------------


def parse_sexpr(text: str):
    """A KiCad s-expression as nested lists of strings."""
    tok = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')
    stack, cur = [], []
    for m in tok.finditer(text):
        t = m.group(0)
        if t == "(":
            stack.append(cur)
            cur = []
        elif t == ")":
            done = cur
            cur = stack.pop()
            cur.append(done)
        elif t[0] == '"':
            cur.append(t[1:-1].replace('\\"', '"').replace("\\\\", "\\"))
        else:
            cur.append(t)
    if stack:
        raise ValueError("kicad: unbalanced parentheses")
    if len(cur) != 1:
        raise ValueError("kicad: expected one top-level expression")
    return cur[0]


def _all(node, key):
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == key]


def _one(node, key):
    for c in node[1:]:
        if isinstance(c, list) and c and c[0] == key:
            return c
    return None


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _xy(node, key):
    c = _one(node, key)
    return None if c is None else np.array([_f(c[1]), _f(c[2])])


def _layer(node):
    c = _one(node, "layer")
    return c[1] if c is not None and len(c) > 1 else None


def _flag(node, key):
    """KiCad 6+ `(key yes)`, or a bare `key` atom (KiCad 5)."""
    c = _one(node, key)
    if c is not None:
        return len(c) < 2 or c[1] in ("yes", "true")
    return key in [x for x in node[1:] if isinstance(x, str)]


# --- geometry in KiCad's own coordinates (y down) -------------------------------

def _arc3(a, m, b, seg_deg=10.0):
    """Points along the circle through a, m, b from a to b."""
    ax, ay = a
    bx, by = b
    mx, my = m
    d = 2.0 * (ax * (my - by) + mx * (by - ay) + bx * (ay - my))
    if abs(d) < 1e-12:
        return np.array([a, b])
    ux = ((ax * ax + ay * ay) * (my - by) + (mx * mx + my * my) * (by - ay) + (bx * bx + by * by) * (ay - my)) / d
    uy = ((ax * ax + ay * ay) * (bx - mx) + (mx * mx + my * my) * (ax - bx) + (bx * bx + by * by) * (mx - ax)) / d
    c = np.array([ux, uy])
    r = float(np.linalg.norm(np.asarray(a) - c))
    t0 = math.atan2(ay - uy, ax - ux)
    tm = math.atan2(my - uy, mx - ux)
    t1 = math.atan2(by - uy, bx - ux)

    def ccw(s, e):
        return (e - s) % (2 * math.pi)
    sweep = ccw(t0, t1)
    if ccw(t0, tm) > sweep:          # the mid point is on the other way round
        sweep = sweep - 2 * math.pi
    n = max(2, int(abs(math.degrees(sweep)) / seg_deg) + 1)
    ts = t0 + np.linspace(0.0, sweep, n)
    pts = c + r * np.stack([np.cos(ts), np.sin(ts)], -1)
    pts[0], pts[-1] = a, b
    return pts


def _circle(c, r, seg_deg=10.0):
    n = max(12, int(360 / seg_deg))
    ts = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.asarray(c) + r * np.stack([np.cos(ts), np.sin(ts)], -1)


def _graphic(node):
    """A drawn item as ("open", points) or ("closed", points), KiCad coords
    in its own frame (the board's, or a footprint's)."""
    kind = node[0].split("_", 1)[-1]
    if kind == "line":
        return "open", np.array([_xy(node, "start"), _xy(node, "end")])
    if kind == "arc":
        s, e, m = _xy(node, "start"), _xy(node, "end"), _xy(node, "mid")
        if m is not None:
            return "open", _arc3(s, m, e)
        # KiCad 5: start is the centre, end the arc's first point, angle
        ang = _f((_one(node, "angle") or [0, 0])[1])
        c, p0 = s, e
        r = float(np.linalg.norm(p0 - c))
        t0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
        ts = t0 + np.radians(np.linspace(0.0, ang, max(2, int(abs(ang) / 10) + 1)))
        return "open", c + r * np.stack([np.cos(ts), np.sin(ts)], -1)
    if kind == "circle":
        c, e = _xy(node, "center"), _xy(node, "end")
        return "closed", _circle(c, float(np.linalg.norm(e - c)))
    if kind == "rect":
        s, e = _xy(node, "start"), _xy(node, "end")
        return "closed", np.array([[s[0], s[1]], [e[0], s[1]], [e[0], e[1]], [s[0], e[1]]])
    if kind in ("poly", "polygon"):
        pts = _one(node, "pts")
        out = []
        for p in (pts or [])[1:]:
            if p[0] == "xy":
                out.append([_f(p[1]), _f(p[2])])
            elif p[0] == "arc":
                a, m, b = _xy(p, "start"), _xy(p, "mid"), _xy(p, "end")
                out += _arc3(a, m, b).tolist()
        return "closed", np.array(out)
    return None, None


def _loops(items, tol=0.02):
    """Closed loops from drawn items: closed ones as they are, open ones
    chained end to end. Returns (loops, stray open chains)."""
    loops, opens = [], []
    for k, p in items:
        if k == "closed" and len(p) >= 3:
            loops.append(np.asarray(p, float))
        elif k == "open" and len(p) >= 2:
            opens.append([np.asarray(q, float) for q in p])
    stray = []
    while opens:
        chain = opens.pop()
        grown = True
        while grown and np.linalg.norm(chain[0] - chain[-1]) > tol:
            grown = False
            for i, o in enumerate(opens):
                if np.linalg.norm(o[0] - chain[-1]) <= tol:
                    chain += o[1:]
                elif np.linalg.norm(o[-1] - chain[-1]) <= tol:
                    chain += o[::-1][1:]
                elif np.linalg.norm(o[-1] - chain[0]) <= tol:
                    chain = o[:-1] + chain
                elif np.linalg.norm(o[0] - chain[0]) <= tol:
                    chain = o[::-1][:-1] + chain
                else:
                    continue
                opens.pop(i)
                grown = True
                break
        if len(chain) >= 4 and np.linalg.norm(chain[0] - chain[-1]) <= tol:
            loops.append(np.array(chain[:-1]))
        else:
            stray.append(np.array(chain))
    return loops, stray


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(pts, poly):
    """Even-odd point in polygon, vectorised over pts."""
    pts = np.atleast_2d(pts)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    a, b = poly, np.roll(poly, -1, axis=0)
    cond = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return (np.sum(cond & (x < xi), axis=1) % 2) == 1


def _seg_dist(pts, poly):
    """Distance of each point to a closed polygon's boundary, and the index
    of the nearest edge."""
    pts = np.atleast_2d(pts)
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    q = a + t[..., None] * ab
    d = np.linalg.norm(pts[:, None, :] - q, axis=-1)
    j = np.argmin(d, axis=1)
    return d[np.arange(len(pts)), j], j


def _hull2(p):
    """Convex hull of 2D points (monotone chain), counter-clockwise."""
    p = np.unique(np.round(np.asarray(p, float), 6), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


# --- packages: heights and densities ------------------------------------------

# Maximum seated heights (mm) by library name, first match wins. Sources:
# JEDEC outlines (MS-012 SOIC 1.75, MO-153 TSSOP 1.20, MO-187 MSOP 1.10,
# MO-150 SSOP 2.00, MO-220 QFN 1.00, MS-026 LQFP 1.60 / TQFP 1.20, MS-022
# PQFP 3.40 (body + standoff), MS-018 PLCC 4.57, MS-001 DIP 5.33, TO-236
# SOT-23 1.45, SC-70 1.10, TO-243 SOT-89 1.60, TO-261 SOT-223 1.80, TO-252
# DPAK 2.39, TO-263 D2PAK 4.83), MLCC maximum thicknesses (EIA 0201 0.33,
# 0402 0.55, 0603 0.90, 0805 1.45, 1206 1.80, 1210 2.70, 1812 2.80) and
# thick-film resistors (0402 0.40 ... 2512 0.70), DO-214 (SMA 2.44, SMB
# 2.62, SMC 2.62), SOD-123 1.35, SOD-323 1.10, 2.54 mm headers and sockets
# (2.5 mm insulator + 6 mm pin = 8.5), USB-C top-mount receptacles 3.3
# (GCT USB4085/4105 3.2-3.31), micro-B 2.8 (Molex 47346 2.75), USB-A
# horizontal 7.0, JST SH side entry 2.95 / top entry 4.25, JST PH top entry
# 6.0, JST XH top entry 7.0, ESP32 WROOM modules 3.2 (datasheets 3.1-3.2),
# 3225/2520/2016 crystals 1.0. A footprint's own `Height` field, a 3D model,
# or the brief's `heights` map wins over all of these.
_CHIP_C = {"0201": 0.33, "0402": 0.55, "0603": 0.90, "0805": 1.45, "1206": 1.80, "1210": 2.70,
           "1812": 2.80, "2220": 2.80, "2010": 2.5, "2512": 2.8}
_CHIP_R = {"0201": 0.26, "0402": 0.40, "0603": 0.50, "0805": 0.60, "1206": 0.65, "1210": 0.65,
           "2010": 0.70, "2512": 0.70, "1812": 0.70}
_CHIP_L = {"0402": 0.55, "0603": 0.95, "0805": 1.25, "1206": 1.30, "1210": 2.0}
_HEIGHTS = [
    (r"^MountingHole|^Fiducial|^TestPoint|^SolderJumper|^NetTie|^Jumper:SolderJumper|^Symbol|^Logo"
     r"|^kibuzzard|^MountingEquipment", 0.0, "no body (hole, fiducial, test point, jumper)"),
    (r"(^|[_:])(SSOP|SOIC|SOP|TSSOP|MSOP|QFN|DFN|LQFP)[-_].*?_\d+(?:\.\d+)?x\d+(?:\.\d+)?x(\d+(?:\.\d+)?)mm",
     "name", "the package's height in its name"),
    (r"(^|[_:])CP_Elec_\d+(?:\.\d+)?x(\d+(?:\.\d+)?)", "name", "can diameter x height in its name"),
    (r"_Axial_.*?_D(\d+(?:\.\d+)?)mm.*_Horizontal", "name", "an axial body laid flat: its diameter, in its name"),
    (r"(^|[_:])D_DO-35_.*Horizontal", 2.28, "DO-35 (DO-204AH) body diameter max, laid flat"),
    (r"(^|[_:])D_DO-41_.*Horizontal", 2.72, "DO-41 (DO-204AL) body diameter max, laid flat"),
    (r"(^|[_:])D_DO-34_.*Horizontal", 1.90, "DO-34 (DO-204AG) body diameter max, laid flat"),
    (r"(^|[_:])LED_D5\.0mm", 8.70, "5 mm (T-1 3/4) LED, seated on its flange"),
    (r"(^|[_:])LED_D3\.0mm", 5.30, "3 mm (T-1) LED, seated on its flange"),
    (r"(^|[_:])TO-92", 5.33, "TO-92 (TO-226AA) body height; leads left long stand it higher"),
    (r"(^|[_:])(TSSOP|HTSSOP)[-_]", 1.20, "JEDEC MO-153 TSSOP max seated height"),
    (r"(^|[_:])(MSOP|VSSOP|HVSSOP)[-_]", 1.10, "JEDEC MO-187 MSOP max seated height"),
    (r"(^|[_:])(SSOP|QSOP)[-_]", 2.00, "JEDEC MO-150 SSOP max seated height"),
    (r"(^|[_:])(SOIC|SOP|HSOP)[-_]", 1.75, "JEDEC MS-012 SOIC max seated height"),
    (r"(^|[_:])(W|U|X)(QFN|DFN)[-_]", 0.80, "JEDEC MO-220 thin QFN/DFN max height"),
    (r"(^|[_:])(V?QFN|DFN|TDFN|VDFN|HVQFN)[-_]", 1.00, "JEDEC MO-220 QFN/DFN max height"),
    (r"(^|[_:])LQFP[-_]", 1.60, "JEDEC MS-026 LQFP max seated height"),
    (r"(^|[_:])TQFP[-_]", 1.20, "JEDEC MS-026 TQFP max seated height"),
    (r"(^|[_:])PQFP[-_]", 3.40, "JEDEC MS-022 PQFP max seated height"),
    (r"(^|[_:])PLCC", 4.57, "JEDEC MS-018 PLCC max seated height"),
    (r"(^|[_:])DIP-\d+_W[\d.]+mm(?!.*Socket)", 5.33, "JEDEC MS-001 DIP max seated height"),
    (r"(^|[_:])(SOT-23|SOT23|TSOT-23)", 1.45, "JEDEC TO-236 SOT-23 max height"),
    (r"(^|[_:])(SOT-323|SOT-363|SC-70)", 1.10, "JEDEC SC-70 / SOT-323 max height"),
    (r"(^|[_:])SOT-89", 1.60, "JEDEC TO-243 SOT-89 max height"),
    (r"(^|[_:])SOT-223", 1.80, "JEDEC TO-261 SOT-223 max height"),
    (r"(^|[_:])TO-252", 2.39, "JEDEC TO-252 DPAK max height"),
    (r"(^|[_:])TO-263", 4.83, "JEDEC TO-263 D2PAK max height"),
    (r"(^|[_:])TO-220-\d+_Horizontal", 4.83, "TO-220 laid flat: tab and body"),
    (r"(^|[_:])D_SMA\b|(^|[_:])D_SMA[_-]", 2.44, "DO-214AC (SMA) max height"),
    (r"(^|[_:])D_SM[BC][_-]?", 2.62, "DO-214AA/AB (SMB/SMC) max height"),
    (r"(^|[_:])D_SOD-123", 1.35, "SOD-123 max height"),
    (r"(^|[_:])D_SOD-323", 1.10, "SOD-323 max height"),
    (r"(^|[_:])D_SOD-523", 0.70, "SOD-523 max height"),
    (r"(^|[_:])LED_0603", 0.80, "0603 chip LED"),
    (r"(^|[_:])LED_0805", 1.10, "0805 chip LED"),
    (r"(^|[_:])C_(\d{4})_\d{4}Metric", "chipC", "MLCC maximum thickness for its size"),
    (r"(^|[_:])R_(\d{4})_\d{4}Metric", "chipR", "thick-film chip resistor thickness for its size"),
    (r"(^|[_:])L_(\d{4})_\d{4}Metric", "chipL", "multilayer chip inductor thickness for its size"),
    (r"(^|[_:])(PinHeader|PinSocket)_\d+x\d+_P2\.54mm_Vertical", 8.50,
     "2.54 mm header/socket: 2.5 mm insulator + 6 mm pin"),
    (r"(^|[_:])(PinHeader|PinSocket)_\d+x\d+_P2\.54mm_Horizontal", 2.54 + 0.2,
     "2.54 mm right-angle header: one pitch high"),
    (r"(^|[_:])USB_C_Receptacle", 3.30, "USB-C top-mount receptacle (GCT USB4085/4105: 3.2-3.31 mm)"),
    (r"(^|[_:])USB_Micro-B", 2.80, "micro-B receptacle (Molex 47346: 2.75 mm)"),
    (r"(^|[_:])USB_A_.*Stacked", 15.6, "stacked USB-A receptacle"),
    (r"(^|[_:])USB_A_", 7.00, "USB-A horizontal receptacle"),
    (r"(^|[_:])JST_SH_.*Horizontal", 2.95, "JST SH side entry (SM..B-SRSS-TB)"),
    (r"(^|[_:])JST_SH_.*Vertical", 4.25, "JST SH top entry (BM..B-SRSS-TB)"),
    (r"(^|[_:])JST_PH_.*Vertical", 6.00, "JST PH top entry (B..B-PH-K)"),
    (r"(^|[_:])JST_XH_.*Vertical", 7.00, "JST XH top entry (B..B-XH-A)"),
    (r"ESP32-(S3-)?WROOM", 3.20, "ESP32 WROOM module (datasheets 3.1-3.2 mm)"),
    (r"(^|[_:])Crystal_SMD_(3225|2520|2016)", 1.00, "3225/2520/2016 SMD crystal"),
]

# effective density of a part's body envelope, g/cm3 (an estimate: a
# moulded package with its lead frame, a ceramic capacitor with its
# terminations, a connector's shell and insulator round air)
_DENSITY = [("ic", 2.0), ("mlcc", 5.5), ("resistor", 3.5), ("inductor", 4.5), ("electrolytic", 1.25),
            ("connector", 1.2), ("module", 2.0), ("crystal", 2.5), ("part", 1.5)]
_DENS = dict(_DENSITY)


def _family(fp: str, ref: str) -> str:
    n = fp.split(":")[-1]
    lib = fp.split(":")[0] if ":" in fp else ""
    if re.match(r"(Connector|TerminalBlock)", lib) or re.match(r"(USB|JST|PinHeader|PinSocket|Molex|Hirose|XT\d|"
                                                                  r"DSUB|TerminalBlock|Altech|BarrelJack)", n):
        return "connector"
    if re.match(r"Package_", lib) or re.match(r"(DIP|SOIC|SSOP|TSSOP|QFN|DFN|LQFP|TQFP|PQFP|PLCC|BGA|SOT|TO-)", n):
        return "ic"
    if re.match(r"C_\d{4}", n):
        return "mlcc"
    if re.match(r"R_(\d{4}|Array|Axial)", n):
        return "resistor"
    if re.match(r"L_", n):
        return "inductor"
    if re.match(r"CP_", n):
        return "electrolytic"
    if re.match(r"(RF_Module|Module)", lib) or "WROOM" in n or "Module" in n:
        return "module"
    if re.match(r"Crystal", n):
        return "crystal"
    if re.match(r"^(J|P|CN)\d", ref):
        return "connector"
    if re.match(r"(Package_|SOIC|SSOP|TSSOP|QFN|DFN|LQFP|TQFP|SOT|TO-|DIP|BGA|PLCC|PQFP)", lib + ":" + n):
        return "ic"
    return "part"


def _name_height(fp: str):
    n = fp.split(":")[-1]
    for pat, h, why in _HEIGHTS:
        m = re.search(pat, n)
        if not m:
            continue
        if h == "name":
            return float(m.groups()[-1]), why
        if h in ("chipC", "chipR", "chipL"):
            tab = {"chipC": _CHIP_C, "chipR": _CHIP_R, "chipL": _CHIP_L}[h]
            code = m.group(2)
            if code in tab:
                return tab[code], why
            continue
        return float(h), why
    return None, None


def _connector_type(fp: str):
    n = fp.split(":")[-1]
    if re.search(r"USB_C_Receptacle", n):
        return "usb-c"
    if re.search(r"USB_Micro-B", n):
        return "usb-micro-b"
    if re.search(r"USB_A_", n) and "Plug" not in n:
        return "usb-a"
    return None


def _quantity(v, unit):
    """'0.6', '0.6 W', '600mW', '85C', '7.7 mm' -> float in the base unit."""
    if v is None:
        return None
    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z°µu]*)", str(v))
    if not m:
        return None
    x, u = float(m.group(1)), m.group(2).lower()
    if unit == "W" and u in ("mw",):
        x *= 1e-3
    if unit == "g" and u in ("mg",):
        x *= 1e-3
    if unit == "g" and u in ("kg",):
        x *= 1e3
    if unit == "mm" and u in ("um", "µm"):
        x *= 1e-3
    return x


# --- 3D models --------------------------------------------------------------------

def _read_wrl(path):
    """Vertices and triangles of a VRML 2.0 file as KiCad writes them
    (IndexedFaceSets, units of 0.1 inch)."""
    txt = open(path, errors="ignore").read()
    V, F = [], []
    for m in re.finditer(r"point\s*\[([^\]]*)\]\s*\}?\s*(?:.|\n)*?coordIndex\s*\[([^\]]*)\]", txt):
        pts = np.array([float(x) for x in re.split(r"[\s,]+", m.group(1).strip()) if x]).reshape(-1, 3)
        idx = [int(x) for x in re.split(r"[\s,]+", m.group(2).strip()) if x]
        base = sum(len(v) for v in V)
        face = []
        for i in idx:
            if i < 0:
                for k in range(1, len(face) - 1):
                    F.append([base + face[0], base + face[k], base + face[k + 1]])
                face = []
            else:
                face.append(i)
        V.append(pts)
    if not V:
        return None
    return np.concatenate(V) * 2.54, np.array(F, int)


_MODEL_CACHE: dict = {}


def _model_points(path):
    """A model's vertices in mm, or None when it cannot be read."""
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    pts = None
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wrl":
            r = _read_wrl(path)
            pts = None if r is None else r[0]
        elif ext in (".step", ".stp"):
            from .cadimport import load_part
            pts = np.asarray(load_part(path).mesh.vertices, float)
    except Exception:
        pts = None
    _MODEL_CACHE[path] = pts
    return pts


@__import__("functools").lru_cache(maxsize=1)
def _install_model_dirs():
    """Where a KiCad install keeps its 3D library (macOS, Linux, Windows)."""
    import glob
    pats = ["/Applications/KiCad/KiCad.app/Contents/SharedSupport/3dmodels", "/usr/share/kicad/3dmodels",
            "/usr/local/share/kicad/3dmodels", os.path.expanduser("~/.local/share/kicad/*/3dmodels"),
            "C:/Program Files/KiCad/*/share/kicad/3dmodels"]
    out = []
    for p in pats:
        out += sorted(glob.glob(p), reverse=True)
    return tuple(out)


def _resolve_model(raw, board_dir, dirs):
    """The file a footprint's model line points to, if it is here: the
    environment's KiCad model folders, the board's folder (KIPRJMOD), any
    `models` folders given, then a KiCad install's own library; .wrl and
    .step are tried for each other (KiCad's library ships both)."""
    cand = []
    s = raw
    for var in re.findall(r"\$\{([^}]+)\}", raw):
        val = board_dir if var == "KIPRJMOD" else os.environ.get(var)
        if val:
            s = s.replace("${" + var + "}", val)
    cand.append(s)
    rel = re.sub(r"^\$\{[^}]+\}[/\\]?", "", raw)
    for d in list(dirs or []) + [board_dir] + list(_install_model_dirs()):
        cand.append(os.path.join(d, rel))
        cand.append(os.path.join(d, os.path.basename(rel)))
    out = []
    for c in cand:
        stem, _ = os.path.splitext(c)
        for e in ("", ".step", ".stp", ".wrl"):
            p = stem + e if e else c
            if p not in out:
                out.append(p)
    for p in out:
        if os.path.isfile(p):
            return p
    return None


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _model_transform(node):
    """(scale, rotation, offset) of a footprint's model line: KiCad scales,
    then turns by -x, -y, -z degrees (in that order), then offsets (mm; inches
    in KiCad 5's `at`)."""
    def vec(key, default):
        c = _one(node, key)
        if c is None:
            return None
        x = _one(c, "xyz")
        return np.array([_f(x[1]), _f(x[2]), _f(x[3])]) if x else np.array(default, float)
    off = vec("offset", [0, 0, 0])
    if off is None:
        at = vec("at", [0, 0, 0])
        off = (at if at is not None else np.zeros(3)) * 25.4
    sc = vec("scale", [1, 1, 1])
    sc = np.ones(3) if sc is None else sc
    ro = vec("rotate", [0, 0, 0])
    ro = np.zeros(3) if ro is None else ro
    R = _rot(2, -ro[2]) @ _rot(1, -ro[1]) @ _rot(0, -ro[0])
    return sc, R, off


# --- the board ---------------------------------------------------------------------

def read_board(path, *, models=None, heights=None, power=None, max_temp=None, masses=None,
               default_height: float = 3.0, lead_protrusion: float = 1.5, edge_reach: float = 2.0,
               copper_fill: float = 0.5, connectors=None, use_models: bool = True) -> dict:
    """Read a .kicad_pcb. See the module docstring. heights, power, max_temp,
    masses: {reference: value} (mm, W, C, g) over what the file says;
    connectors: {reference: {out}} to say which way one faces; models:
    folders to look for 3D models in.

    Returns {file, thickness, size [x, y, t], outline [[x, y]], cutouts,
    origin (the KiCad point at the frame's origin), copper, mass (kg, the
    bare board), conductivity (in-plane, W/(m K)), parts [...], holes [...],
    warnings}, in the frame of the module docstring."""
    path = str(path)
    root = parse_sexpr(open(path, encoding="utf-8", errors="ignore").read())
    if not root or root[0] not in ("kicad_pcb",):
        raise ValueError(f"{path}: not a KiCad board (.kicad_pcb)")
    bdir = os.path.dirname(os.path.abspath(path))
    heights, power, max_temp, masses = (dict(x or {}) for x in (heights, power, max_temp, masses))
    conn_over = dict(connectors or {})
    warnings = []

    # thickness and copper: the stackup when there is one
    gen = _one(root, "general")
    t_gen = _f((_one(gen, "thickness") or [0, 1.6])[1], 1.6) if gen else 1.6
    setup = _one(root, "setup")
    stk = _one(setup, "stackup") if setup else None
    cu_t, diel_t, n_cu = [], 0.0, 0
    if stk:
        for ly in _all(stk, "layer"):
            ty = (_one(ly, "type") or [0, ""])[1]
            th = _f((_one(ly, "thickness") or [0, 0])[1])
            if ty == "copper":
                cu_t.append(th)
            elif ty in ("core", "prepreg") or "dielectric" in str(ly[1]):
                diel_t += th
    layers = _one(root, "layers")
    if layers is not None:
        n_cu = sum(1 for ly in layers[1:] if isinstance(ly, list) and len(ly) > 1 and str(ly[1]).endswith(".Cu"))
    if not cu_t:
        cu_t = [0.035] * max(n_cu, 2)
    t_stack = sum(cu_t) + diel_t if diel_t > 0 else 0.0
    t = t_stack if t_stack > 0.2 else t_gen
    thick_from = "stackup" if t_stack > 0.2 else "general.thickness"

    # the outline: Edge.Cuts on the board and inside footprints
    items = []
    fps = _all(root, "footprint") + _all(root, "module")
    for g in root[1:]:
        if isinstance(g, list) and g and g[0].startswith("gr_") and _layer(g) == "Edge.Cuts":
            items.append(_graphic(g))
    for fp in fps:
        at = _one(fp, "at")
        X0, Y0, A = _f(at[1]), _f(at[2]), _f(at[3]) if len(at) > 3 else 0.0
        for g in fp[1:]:
            if isinstance(g, list) and g and g[0].startswith("fp_") and _layer(g) == "Edge.Cuts":
                k, p = _graphic(g)
                if k:
                    items.append((k, _fp_to_kicad(p, X0, Y0, A)))
    loops, stray = _loops([i for i in items if i[0]])
    if not loops:
        raise ValueError(f"{path}: no closed board outline on Edge.Cuts"
                         + (f" ({len(stray)} open chains: a gap in the outline)" if stray else ""))
    if stray:
        warnings.append(f"{len(stray)} Edge.Cuts chains do not close; left out of the outline")
    loops.sort(key=lambda p: -abs(_area(p)))
    outer_k = loops[0]
    lo, hi = outer_k.min(axis=0), outer_k.max(axis=0)
    ck = 0.5 * (lo + hi)

    def to_board(p):
        p = np.atleast_2d(np.asarray(p, float))
        return np.stack([p[:, 0] - ck[0], -(p[:, 1] - ck[1])], -1)
    outer = to_board(outer_k)
    if _area(outer) < 0:
        outer = outer[::-1]
    cut = []
    for lp in loops[1:]:
        b = to_board(lp)
        if _inside(b[:1], outer)[0]:
            cut.append(b[::-1] if _area(b) > 0 else b)
    size = [float(hi[0] - lo[0]), float(hi[1] - lo[1]), round(float(t), 4)]
    area_m2 = (abs(_area(outer)) - sum(abs(_area(c)) for c in cut)) * 1e-6

    # the bare board's mass and its in-plane conductivity
    t_cu = sum(cu_t)
    t_fr4 = max(t - t_cu, 0.0)
    m_board = area_m2 * (t_fr4 * 1e-3 * 1850.0 + t_cu * 1e-3 * 8960.0 * copper_fill)
    k_plane = (390.0 * t_cu * copper_fill + 0.8 * t_fr4) / max(t, 1e-9)

    parts, holes = [], []
    for fp in fps:
        parts_or_hole = _footprint(fp, to_board, t, bdir, models, heights, power, max_temp, masses,
                                   default_height, lead_protrusion, use_models, warnings)
        if parts_or_hole is None:
            continue
        kind, rec = parts_or_hole
        if kind == "hole":
            holes.append(rec)
        else:
            parts.append(rec)
    # connectors: which way each faces
    for p in parts:
        if p["family"] != "connector":
            continue
        _facing(p, outer, t, edge_reach, conn_over.get(p["ref"]) or {})
        if p["connector"].get("how") == "guessed":
            warnings.append(f"{p['ref']} ({p['footprint']}): not at a board edge; facing {p['connector']['out']} "
                            "guessed from its contacts and body (give connectors.{ref}.out)")
        elif p["connector"].get("out") is None:
            warnings.append(f"{p['ref']}: which way it faces is unknown (give connectors.{p['ref']}.out)")
    # a screw needs its head's seat and its sleeve's ring clear of parts
    from .fasteners import METRIC
    for h in holes:
        if not h["screw"]:
            continue
        nom = float(h["screw"][1:])
        r_head = 0.5 * METRIC[h["screw"]][1] + 0.5
        r_sleeve = max(0.5 * nom + max(1.5, 0.35 * nom), 0.5 * h["pad"])
        c = np.asarray(h["at"], float)
        for p in parts:
            plan = np.asarray(p["plan"], float)
            d = 0.0 if _inside(c[None], plan)[0] else float(_seg_dist(c[None], plan)[0][0])
            if d < max(r_head, r_sleeve) - 1e-6:
                warnings.append(f"{h['ref']} ({h['screw']}): {p['ref']} is {d:.2f} mm from its centre, inside the "
                                f"screw head's or the sleeve's {max(r_head, r_sleeve):.2f} mm")
    assumed = [p["ref"] for p in parts if p["heightFrom"] == "assumed"]
    if assumed:
        warnings.append(f"{len(assumed)} parts have no model, no Height field and no known package; "
                        f"{default_height:g} mm assumed (give heights): {', '.join(assumed[:20])}"
                        + (" ..." if len(assumed) > 20 else ""))
    if not any(p.get("watts") for p in parts):
        warnings.append("no part has a power number (footprint field Power, or the brief's power map): "
                        "the board makes no heat here")
    return {"file": os.path.abspath(path), "thickness": round(float(t), 4), "thicknessFrom": thick_from,
            "size": [round(s, 4) for s in size], "outline": np.round(outer, 4).tolist(),
            "cutouts": [np.round(c, 4).tolist() for c in cut], "origin": np.round(ck, 4).tolist(),
            "copper": {"layers": len(cu_t), "thickness": round(t_cu, 4), "fill": copper_fill},
            "mass": round(m_board, 6), "conductivity": round(k_plane, 2), "parts": parts, "holes": holes,
            "warnings": warnings}


def _fp_to_kicad(p, X0, Y0, A):
    """Footprint-local KiCad points (y down) to the board's KiCad coords."""
    p = np.atleast_2d(np.asarray(p, float))
    a = math.radians(A)
    c, s = math.cos(a), math.sin(a)
    # in y-up terms the footprint turns counter-clockwise by A
    xu, yu = p[:, 0], -p[:, 1]
    xr, yr = c * xu - s * yu, s * xu + c * yu
    return np.stack([X0 + xr, Y0 - yr], -1)


def _props(fp):
    out = {}
    for pr in _all(fp, "property"):
        if len(pr) > 2:
            out[str(pr[1])] = pr[2]
    for tx in _all(fp, "fp_text"):              # KiCad 5
        if len(tx) > 2 and tx[1] in ("reference", "value"):
            out["Reference" if tx[1] == "reference" else "Value"] = tx[2]
    return out


def _footprint(fp, to_board, t, bdir, models, heights, power, max_temp, masses, default_height,
               lead_protrusion, use_models, warnings):
    name = str(fp[1])
    lname = name.split(":")[-1]
    at = _one(fp, "at")
    X0, Y0, A = _f(at[1]), _f(at[2]), _f(at[3]) if len(at) > 3 else 0.0
    side = "bottom" if str(_layer(fp) or "F.Cu").startswith("B.") else "top"
    pr = _props(fp)
    ref = str(pr.get("Reference", "?"))
    attr = _one(fp, "attr")
    attrs = [str(a) for a in (attr or [])[1:]]
    if "board_only" in attrs and not re.match(r"MountingHole", lname):
        return None

    def kb(p):
        return to_board(_fp_to_kicad(p, X0, Y0, A))
    pads = []
    for pd in _all(fp, "pad"):
        pat = _one(pd, "at")
        sz = _one(pd, "size")
        dr = _one(pd, "drill")
        drill = None
        if dr is not None:
            nums = [_f(x) for x in dr[1:] if not isinstance(x, list) and x != "oval"]
            drill = max(nums) if nums else None
        pads.append({"num": str(pd[1]), "type": str(pd[2]), "at": np.array([_f(pat[1]), _f(pat[2])]),
                     "size": np.array([_f(sz[1]), _f(sz[2])]) if sz else np.zeros(2), "drill": drill})
    tht = any(p["type"] == "thru_hole" for p in pads)

    # a mounting hole: a MountingHole footprint, or a lone big drill on an H ref
    drills = [p for p in pads if p["drill"] and p["type"] in ("thru_hole", "np_thru_hole")]
    if re.match(r"MountingHole", lname) or (re.match(r"^(H|MH)\d", ref) and drills and
                                            max(p["drill"] for p in drills) >= 1.5):
        if not drills:
            return None
        big = max(drills, key=lambda p: p["drill"])
        c = kb(big["at"])[0]
        pad = max(float(max(p["size"])) for p in drills if p["type"] == "thru_hole") if any(
            p["type"] == "thru_hole" for p in drills) else 0.0
        from .fasteners import METRIC
        m = re.search(r"_M(\d+(?:\.\d+)?)", lname)
        screw = f"M{m.group(1)}" if m and f"M{m.group(1)}" in METRIC else None
        if screw is None:
            fits = [k for k in METRIC if float(k[1:]) < big["drill"] - 0.1]
            screw = max(fits, key=lambda k: float(k[1:])) if fits else None
        return "hole", {"ref": ref, "at": np.round(c, 4).tolist(), "drill": round(big["drill"], 3),
                        "pad": round(pad, 3), "screw": screw, "plated": big["type"] == "thru_hole"}

    # the body's plan: courtyard, else fab outline, else pads
    plan, plan_from = None, None
    # the fab outline is the body as drawn; the courtyard adds the assembly
    # clearance round it (0.25-0.5 mm), so it is only the fallback
    for lyr, label in (("Fab", "fab outline"), ("CrtYd", "courtyard")):
        its = [_graphic(g) for g in fp[1:] if isinstance(g, list) and g and g[0].startswith("fp_")
               and str(_layer(g) or "").endswith(lyr)]
        its = [i for i in its if i[0]]
        if not its:
            continue
        lp, stray = _loops(its)
        pts = np.concatenate([p for _, p in its])
        if lp and not stray:
            big = max(lp, key=lambda q: abs(_area(q)))
            cand = big if len(lp) == 1 else _hull2(np.concatenate(lp))
        else:
            cand = _hull2(pts)
        if len(cand) < 3 or abs(_area(cand)) < 0.05:
            continue                        # a lone line on that layer: not an outline
        plan, plan_from = cand, label
        break
    if plan is None and pads:
        P = np.concatenate([[p["at"] - p["size"] / 2, p["at"] + p["size"] / 2] for p in pads])
        plan = _hull2(np.array([[P[:, 0].min(), P[:, 1].min()], [P[:, 0].max(), P[:, 1].min()],
                                [P[:, 0].max(), P[:, 1].max()], [P[:, 0].min(), P[:, 1].max()]]) +
                      np.array([[-.25, -.25], [.25, -.25], [.25, .25], [-.25, .25]]))
        plan_from = "pads"
    if plan is None:
        return None
    plan_b = kb(plan)
    if _area(plan_b) < 0:
        plan_b = plan_b[::-1]

    # height: the brief, the footprint's field, its 3D model, its package
    height, h_from, hull = None, None, None
    model_file = None
    if ref in heights:
        height, h_from = float(heights[ref]), "brief"
    elif pr.get("Height") is not None and _quantity(pr["Height"], "mm") is not None:
        height, h_from = _quantity(pr["Height"], "mm"), "footprint Height field"
    if use_models:
        Ms = [m for m in _all(fp, "model") if not _flag(m, "hide")]
        P3 = []
        for mnode in Ms:
            fpath = _resolve_model(str(mnode[1]), bdir, models)
            if fpath is None:
                continue
            pts = _model_points(fpath)
            if pts is None or len(pts) == 0:
                continue
            sc, R, off = _model_transform(mnode)
            q = (R @ (pts * sc).T).T + off
            if side == "bottom":
                # KiCad's 3D viewer turns a flipped footprint's model half a
                # turn about y and then about z: (x, y, z) -> (x, -y, -z),
                # the mirror its pads and outlines were stored with
                q = q * np.array([1.0, -1.0, -1.0])
            a = math.radians(A)
            Rz = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            xy = (Rz @ q[:, :2].T).T
            c0 = to_board(np.array([[X0, Y0]]))[0]
            z = q[:, 2] + (t if side == "top" else 0.0)
            P3.append(np.column_stack([xy + c0, z]))
            model_file = fpath
        if P3:
            hull = np.concatenate(P3)
            # the body is what stands off its face: pins through the board
            # are the leads' (below), and the hull of the whole model would
            # stretch the body's front down to them
            keep_ = hull[:, 2] >= t - 0.05 if side == "top" else hull[:, 2] <= 0.05
            if keep_.sum() >= 4:
                hull = hull[keep_]
                hull[:, 2] = np.clip(hull[:, 2], t, None) if side == "top" else np.clip(hull[:, 2], None, 0.0)
            mh = float(hull[:, 2].max() - t) if side == "top" else float(-hull[:, 2].min())
            if height is None:
                height, h_from = round(mh, 3), "3D model"
    if height is None:
        h, why = _name_height(name)
        if h is not None:
            height, h_from = h, f"package: {why}"
    if height is None:
        height, h_from = float(default_height), "assumed"
    if height <= 0 and hull is None:
        return None
    fam = _family(name, ref)

    # body volume, mass
    a_plan = abs(_area(plan_b))
    vol = a_plan * height                                   # mm3
    if hull is not None:
        try:
            import trimesh
            vol = float(trimesh.convex.convex_hull(hull).volume)
        except Exception:
            pass
    if ref in masses:
        mass_g, m_from = float(masses[ref]), "brief"
    elif _quantity(pr.get("Mass"), "g") is not None:
        mass_g, m_from = _quantity(pr["Mass"], "g"), "footprint Mass field"
    else:
        mass_g = vol * 1e-3 * _DENS.get(fam, 1.5)
        m_from = f"estimate: body volume x {_DENS.get(fam, 1.5):g} g/cm3 ({fam})"
    watts, w_from = None, None
    if ref in power:
        watts, w_from = float(power[ref]), "brief"
    else:
        for k in ("Power", "Pd", "Dissipation", "Watts", "Heat"):
            if pr.get(k) is not None and _quantity(pr[k], "W") is not None:
                watts, w_from = _quantity(pr[k], "W"), f"footprint {k} field"
                break
    tmax = None
    if ref in max_temp:
        tmax = float(max_temp[ref])
    else:
        for k in ("MaxTemp", "Tmax", "TMax", "MaxTemperature"):
            if pr.get(k) is not None and _quantity(pr[k], "C") is not None:
                tmax = _quantity(pr[k], "C")
                break
    pads_b = [{"num": p["num"], "type": p["type"], "at": np.round(kb(p["at"])[0], 4).tolist()} for p in pads]
    rec = {"ref": ref, "value": str(pr.get("Value", "")), "footprint": name, "side": side, "family": fam,
           "at": np.round(to_board(np.array([[X0, Y0]]))[0], 4).tolist(), "angle": A,
           "plan": np.round(plan_b, 4).tolist(), "planFrom": plan_from,
           "height": round(float(height), 4), "heightFrom": h_from,
           "z": ([round(t, 4), round(t + height, 4)] if side == "top" else [round(-height, 4), 0.0]),
           "mass": round(mass_g * 1e-3, 7), "massFrom": m_from, "throughHole": tht, "pads": pads_b}
    if hull is not None:
        rec["model"] = model_file
        rec["_hull"] = hull
    if watts:
        rec["watts"], rec["wattsFrom"] = round(float(watts), 5), w_from
    if tmax is not None:
        rec["maxTemp"] = tmax
    if tht and lead_protrusion > 0:
        P = np.array([p["at"] for p in pads_b if p["type"] == "thru_hole"])
        rec["leads"] = {"plan": np.round(_hull2(np.vstack([P + d for d in ([-.6, -.6], [.6, -.6], [.6, .6],
                                                                              [-.6, .6])])), 4).tolist(),
                        "z": [-lead_protrusion, 0.0] if side == "top" else [t, t + lead_protrusion]}
    if fam == "connector":
        rec["connector"] = {"type": _connector_type(name)}
    return "part", rec


def _snap(v, deg=12.0):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) >= math.cos(math.radians(deg)):
        e = np.zeros(len(v))
        e[k] = np.sign(v[k])
        return e
    return v


def _axis_str(v):
    v = np.asarray(v, float)
    k = int(np.argmax(np.abs(v)))
    if abs(abs(v[k]) - 1.0) < 1e-9:
        return ("+" if v[k] > 0 else "-") + "xyz"[k]
    return np.round(v, 4).tolist()


def _facing(p, outer, t, reach, over):
    """Which way a connector faces, and where its mouth is."""
    plan = np.asarray(p["plan"], float)
    z0, z1 = p["z"]
    c = p.setdefault("connector", {})
    out, how = None, None
    if over.get("out") is not None:
        from .handoff import _axis_vec
        out, how = np.asarray(_axis_vec(over["out"]), float), "brief"
    else:
        d, j = _seg_dist(plan, outer)
        outside = ~_inside(plan, outer)
        if re.search(r"Vertical", p["footprint"]):
            # mates from above, wherever it sits
            out, how = np.array([0.0, 0.0, 1.0 if p["side"] == "top" else -1.0]), "vertical: off its face"
        elif outside.any() or d.min() <= reach:
            # the edge it overhangs or sits at: the outward normal there
            k = int(np.argmax(np.where(outside, 1e9, 0) - d))
            e = int(j[k])
            a, b = outer[e], outer[(e + 1) % len(outer)]
            tng = (b - a) / max(np.linalg.norm(b - a), 1e-12)
            n = np.array([tng[1], -tng[0]])          # outward for a counter-clockwise outline
            out, how = np.array([*_snap(n), 0.0]), "board edge"
        else:
            sig = [q["at"] for q in p["pads"] if re.match(r"^\d+$", q["num"])]
            if sig:
                v = plan.mean(axis=0) - np.mean(sig, axis=0)
                if np.linalg.norm(v) > 0.2:
                    out, how = np.array([*_snap(v, 45.0), 0.0]), "guessed"
    if out is None:
        c.update({"out": None, "how": "unknown"})
        return
    # the mouth: the body's furthest point that way, at its mid height
    if abs(out[2]) > 0.5:
        m = np.array([*plan.mean(axis=0), z1 if out[2] > 0 else z0])
    else:
        s = plan @ out[:2]
        far = plan[s >= s.max() - 1e-6]
        m = np.array([*far.mean(axis=0), 0.5 * (z0 + z1)])
    c.update({"out": _axis_str(out), "mouth": np.round(m, 4).tolist(), "how": how})


# --- bodies as meshes ----------------------------------------------------------------

def _prism(poly, z0, z1):
    import manifold3d as m3d
    poly = np.asarray(poly, float)
    if _area(poly) < 0:
        poly = poly[::-1]
    cs = m3d.CrossSection([poly.tolist()], m3d.FillRule.Positive)
    return m3d.Manifold.extrude(cs, float(z1 - z0)).translate((0.0, 0.0, float(z0)))


def part_manifold(p, leads=True):
    """A part's body (and its leads through the board) as a manifold."""
    import manifold3d as m3d
    if p.get("_hull") is not None:
        body = m3d.Manifold.hull_points(np.asarray(p["_hull"], float).tolist())
    else:
        body = _prism(p["plan"], *p["z"])
    if leads and p.get("leads"):
        body = body + _prism(p["leads"]["plan"], *p["leads"]["z"])
    return body


def board_manifold(board, with_parts=True, skip=()):
    """The board (outline less cut-outs and holes) and, unless with_parts
    is False, every part not in `skip`, as one manifold."""
    import manifold3d as m3d
    t = board["thickness"]
    loops = [np.asarray(board["outline"], float)] + [np.asarray(c, float)[::-1] if _area(np.asarray(c)) < 0
                                                    else np.asarray(c, float)[::-1] for c in board["cutouts"]]
    loops = [loops[0] if _area(loops[0]) > 0 else loops[0][::-1]] + [
        (c if _area(c) < 0 else c[::-1]) for c in loops[1:]]
    cs = m3d.CrossSection([lp.tolist() for lp in loops], m3d.FillRule.NonZero)
    for h in board["holes"]:
        cs = cs - m3d.CrossSection.circle(0.5 * h["drill"], 32).translate(tuple(h["at"]))
    man = m3d.Manifold.extrude(cs, float(t))
    if with_parts:
        # a part made a component of its own leaves its leads with the board
        bodies = [part_manifold(p) for p in board["parts"] if p["ref"] not in skip] + [
            _prism(p["leads"]["plan"], *p["leads"]["z"]) for p in board["parts"] if p["ref"] in skip and p.get("leads")]
        bodies = [m for m in bodies if m.volume() > 0]
        if bodies:
            man = m3d.Manifold.batch_boolean([man] + bodies, m3d.OpType.Add)
    return man


def _write_stl(man, path):
    import trimesh
    m = man.to_mesh()
    trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)).export(path)
    return path


def _public(board):
    """The board record without private arrays."""
    out = dict(board)
    out["parts"] = [{k: v for k, v in p.items() if not k.startswith("_")} for p in board["parts"]]
    return out


def write_board(board, out_dir, name="board"):
    """board.json and the board with its parts as an STL (board frame)."""
    os.makedirs(out_dir, exist_ok=True)
    _write_stl(board_manifold(board), os.path.join(out_dir, f"{name}.stl"))
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(_public(board), fh, indent=1)
    return os.path.join(out_dir, f"{name}.json")


# --- into an enclosure ----------------------------------------------------------------

_COMPONENT_KEYS = ("rotate", "against", "alignTo", "under", "near", "margin", "place", "mount", "role",
                   "external", "insertFrom", "fixed", "clearance", "tilt", "pathTolerance", "ports", "maxTemp",
                   "installedOn")


def _kicad_opts(spec):
    k = spec.get("kicad")
    return {"file": k} if isinstance(k, str) else dict(k or {})


def load_board(spec, base_dir=None):
    """read_board from a brief's `kicad` entry (a path, or {file, models,
    heights, power, maxTemp, masses, defaultHeight, leadProtrusion,
    edgeReach, copperFill, connectors, useModels})."""
    o = _kicad_opts(spec)
    f = o.get("file")
    if not f:
        raise ValueError("kicad: give the board file (`kicad: \"board.kicad_pcb\"` or {file})")
    if base_dir and not os.path.isabs(f):
        f = os.path.join(base_dir, f)
    mdirs = [m if os.path.isabs(m) or not base_dir else os.path.join(base_dir, m) for m in
             (o.get("models") or [])] if not isinstance(o.get("models"), str) else [o["models"]]
    conns = {r: v for r, v in (o.get("connectors") or {}).items()}
    return read_board(f, models=mdirs, heights=o.get("heights"), power=o.get("power"),
                      max_temp=o.get("maxTemp"), masses=o.get("masses"),
                      default_height=float(o.get("defaultHeight", 3.0)),
                      lead_protrusion=float(o.get("leadProtrusion", 1.5)),
                      edge_reach=float(o.get("edgeReach", 2.0)), copper_fill=float(o.get("copperFill", 0.5)),
                      connectors=conns, use_models=bool(o.get("useModels", True)))


def enclosure_components(spec: dict, base_dir: str | None, out_dir: str):
    """An enclosure component with a `kicad` entry, as the components it
    stands for: the board (outline, thickness, the parts that stay on it,
    its mass and in-plane conductivity, held on its mounting holes), and
    each connector and each part that makes heat as its own component
    installed on it (so a connector can be aligned to an opening and its
    plug loads go through the board's mount, and heat sits where the part
    is). Returns (components, connector loads, report)."""
    o = _kicad_opts(spec)
    board = load_board(spec, base_dir)
    bid = spec.get("id") or "board"
    cmap = o.get("connectors") or {}
    unknown = [r for r in list(cmap) + list(o.get("power") or {}) + list(o.get("heights") or {})
               if r not in {p["ref"] for p in board["parts"]}]
    if unknown:
        raise ValueError(f"{bid}.kicad: no part {sorted(set(unknown))} on {os.path.basename(board['file'])}")
    sep_refs = set(o.get("separate") or [])
    own = [p for p in board["parts"] if p["family"] == "connector" or p.get("watts") or p["ref"] in sep_refs]
    kdir = os.path.join(out_dir, "cad", "kicad")
    os.makedirs(kdir, exist_ok=True)
    skip = {p["ref"] for p in own}
    bpath = _write_stl(board_manifold(board, skip=skip), os.path.join(kdir, f"{bid}.stl"))
    m_rest = board["mass"] + sum(p["mass"] for p in board["parts"] if p["ref"] not in skip)
    comp = {k: v for k, v in spec.items() if k in _COMPONENT_KEYS}
    comp.update({"id": bid, "shape": {"type": "mesh", "path": os.path.abspath(bpath)}, "mass": round(m_rest, 6),
                 "conductivity": board["conductivity"]})
    w_rest = sum(p.get("watts") or 0 for p in board["parts"] if p["ref"] not in skip)
    if w_rest:
        comp["watts"] = round(w_rest, 5)
    if "mount" not in spec and board["holes"]:
        comp["mount"] = {"type": "holes", "holes": [{"at": [h["at"][0], h["at"][1]], "screw": h["screw"],
                                                     "pad": h["pad"], "drill": h["drill"], "ref": h["ref"]}
                                                    for h in board["holes"] if h["screw"]],
                         "thickness": board["thickness"]}
    comps, loads = [comp], []
    for p in own:
        cid = f"{bid}-{p['ref']}"
        ppath = _write_stl(part_manifold(p, leads=False), os.path.join(kdir, f"{cid}.stl"))
        c = {"id": cid, "shape": {"type": "mesh", "path": os.path.abspath(ppath)}, "installedOn": bid,
             "mass": p["mass"]}
        if p.get("watts"):
            c["watts"] = p["watts"]
        if p.get("maxTemp") is not None:
            c["maxTemp"] = p["maxTemp"]
        cn = p.get("connector") or {}
        m = cmap.get(p["ref"]) or {}
        if cn.get("out") is not None:
            c["ports"] = [{"id": "mate", "at": cn["mouth"], "direction": cn["out"]}]
        if m.get("opening"):
            if cn.get("out") is None or not isinstance(cn["out"], str):
                raise ValueError(f"{bid}.kicad.connectors.{p['ref']}: aligned to {m['opening']!r}, but which way it "
                                 "faces is not along an axis of the board; give out")
            c["alignTo"] = {"opening": m["opening"], "face": cn["out"], "recess": float(m.get("recess", 0.3))}
            typ = m.get("type") or cn.get("type")
            if typ and m.get("plugLoads", True):
                ld = {"component": cid, "type": typ}
                for k in ("insert", "withdraw", "side", "lever", "abuse", "wrench", "maxShift"):
                    if k in m:
                        ld[k] = m[k]
                loads.append(ld)
        comps.append(c)
    report = {"id": bid, "file": board["file"], "size": board["size"], "thicknessFrom": board["thicknessFrom"],
              "mass": round(sum(c.get("mass") or 0 for c in comps), 6),
              "watts": round(sum(c.get("watts") or 0 for c in comps), 5),
              "holes": board["holes"], "components": [c["id"] for c in comps],
              "parts": [{k: p[k] for k in ("ref", "footprint", "side", "height", "heightFrom", "mass", "massFrom")
                         if k in p} | ({"connector": p["connector"]} if p.get("connector") else {})
                        | ({"watts": p["watts"]} if p.get("watts") else {}) for p in board["parts"]],
              "warnings": board["warnings"]}
    with open(os.path.join(kdir, f"{bid}.json"), "w") as fh:
        json.dump(_public(board), fh, indent=1)
    return comps, loads, report


def expand_brief(brief: dict, base_dir: str | None, out_dir: str):
    """An enclosure brief with every `kicad` component expanded (see
    enclosure_components) and their plug loads added to loads.connector
    (one already given for the same component wins). Returns (brief, the
    boards' reports)."""
    if not any(isinstance(c, dict) and c.get("kicad") for c in brief.get("components") or []):
        return brief, []
    b = dict(brief)
    comps, reports, plug = [], [], []
    for c in b["components"]:
        if isinstance(c, dict) and c.get("kicad"):
            cs, ls, rep = enclosure_components(c, base_dir, out_dir)
            comps += cs
            plug += ls
            reports.append(rep)
        else:
            comps.append(c)
    b["components"] = comps
    if plug:
        loads = dict(b.get("loads") or {})
        have = loads.get("connector")
        have = [] if have is None else (list(have) if isinstance(have, list) else [have])
        named = {x.get("component") for x in have}
        loads["connector"] = have + [x for x in plug if x["component"] not in named]
        b["loads"] = loads
    return b, reports


# --- into a stack -----------------------------------------------------------------------

def stack_board(spec: dict, base_dir: str | None = None) -> dict:
    """A stack brief's board with a `kicad` entry, filled from the board:
    size [x, y, thickness] (the outline's bounding box), mass (board and
    parts), watts, maxTemp (the lowest any part allows), above/below (the
    tallest part on each face) and ports (each connector's mouth, the way
    it faces). What the entry already says wins."""
    board = load_board(spec, base_dir)
    t = board["thickness"]
    parts = board["parts"]
    above = max([p["z"][1] - t for p in parts if p["side"] == "top"] +
                [p["leads"]["z"][1] - t for p in parts if p.get("leads") and p["side"] == "bottom"] + [0.0])
    below = max([-p["z"][0] for p in parts if p["side"] == "bottom"] +
                [-p["leads"]["z"][0] for p in parts if p.get("leads") and p["side"] == "top"] + [0.0])
    w = sum(p.get("watts") or 0 for p in parts)
    tm = [p["maxTemp"] for p in parts if p.get("maxTemp") is not None]
    ports = [{"id": p["ref"], "at": p["connector"]["mouth"], "direction": p["connector"]["out"]}
             for p in parts if (p.get("connector") or {}).get("out") is not None]
    out = {"id": spec.get("id"), "size": board["size"],
           "mass": round(board["mass"] + sum(p["mass"] for p in parts), 6),
           "above": round(above, 3), "below": round(below, 3)}
    if w:
        out["watts"] = round(w, 5)
    if tm:
        out["maxTemp"] = float(min(tm))
    if ports:
        out["ports"] = ports
    out.update({k: v for k, v in spec.items() if k != "kicad"})
    out["_kicad"] = {"file": board["file"], "holes": board["holes"], "warnings": board["warnings"]}
    return out
