"""
Plug loads: what a user does to a product's port.

A USB-C socket is pushed on every time a cable goes in, pulled on every
time it comes out, and levered sideways whenever someone trips on the cable
or uses the product while it charges. The socket is soldered to a board, the
board sits on mounts, and the mounts are part of the printed interior: every
one of those forces ends up in the part we design. These are load cases any
study (or an enclosure brief's `loads.connector`) can declare by naming the
connector, instead of a designer looking the numbers up.

Each connector type has

- **insert**: N to push the plug home (the design value is the top of the
  spec's range: the stiffest new socket);
- **withdraw**: N to pull it out (again the top of the range);
- **side**: sideways forces the plug must survive *while still working*,
  applied `lever` mm out from the socket's front edge: the plug levered
  across its thin side and across its wide side (USB-C's 4-axis continuity
  test). These are everyday abuse, so they are designed to the study's full
  safety factor;
- **wrench** (optional, `abuse`): the moment at which the plug itself is
  allowed to break or come out (USB-C's wrenching strength test). Past this
  the cable is the weak link; the product should hold up to it, so these
  cases carry their own safety factor of 1 (`abuse_factor`), not the
  goal's. Off by default: several times the side load.

The forces go into `region`, the thing the socket is fixed to (a board's
mount, or the socket's own preserved body), with `at` set to where the plug
pushes, so the lever arm's moment reaches the mounts too. The reaction is
the hand holding the product (`hold`, optional; the study's supports
otherwise). A travel limit on `region` in these cases (`maxShift`, mm)
keeps the board from moving enough to flex the socket's solder joints.

Numbers: USB-C from vendor qualification reports that cite the USB-IF
Type-C specification and EIA-364-13 (insertion 5-20 N, extraction 8-20 N
new; 4-axis continuity 8 N up/down and 20 N left/right at 15 mm; wrenching
2.0 N·m up/down, 3.5 N·m left/right). USB 2.0 Standard-A and Micro-B from
the USB 2.0 specification's mechanical limits (35 N insertion max, 10 N
extraction min; no side test, so a 10 N side load at 15 mm is assumed).
Treat them as typical: the connector's own datasheet wins, and any field can
be overridden.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CONNECTORS", "connector_cases"]

# N, N, [(across, N)], mm, {across: N·mm}
CONNECTORS = {
    "usb-c": {"insert": 20.0, "withdraw": 20.0, "side": {"thin": 8.0, "wide": 20.0}, "lever": 15.0,
              "wrench": {"thin": 2000.0, "wide": 3500.0},
              "source": "USB Type-C (vendor reports citing USB-IF / EIA-364-13)"},
    "usb-a": {"insert": 35.0, "withdraw": 35.0, "side": {"thin": 10.0, "wide": 10.0}, "lever": 15.0,
              "source": "USB 2.0 Standard-A (insertion max; extraction taken equal; side assumed)"},
    "usb-micro-b": {"insert": 35.0, "withdraw": 35.0, "side": {"thin": 10.0, "wide": 10.0}, "lever": 15.0,
                    "source": "USB 2.0 Micro-B (insertion max; extraction taken equal; side assumed)"},
}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _r(v, n=4):
    return np.round(np.asarray(v, float), n).tolist()


def connector_cases(kind: str = "usb-c", *, region, mouth, out, wide=None, hold=None, name: str | None = None,
                    insert: float | None = None, withdraw: float | None = None, side: dict | None = None,
                    lever: float | None = None, abuse: bool = False, wrench: dict | None = None, abuse_factor: float = 1.0,
                    max_shift: float | None = None):
    """Load cases for a plug in a socket.

    region: what the socket is fixed to, a region id (a board's mount, the
    socket's body) or a load target ({region} or {shape}: where the board
    bears on the part); mouth: the socket's front edge, where the plug enters (mm);
    out: the direction the plug comes out; wide: the socket's wide
    direction across the plug (default the axis across `out` nearest +x,
    else +y); hold: a load target ({region} or {shape}) that takes the
    reaction, or None for the study's supports.

    Returns (cases, clearance): the load cases, named <name>-insert,
    -withdraw, -side-thin, -side-wide (and -wrench-thin, -wrench-wide with
    abuse), and a goal clearance on `region` in those cases when max_shift
    is given (else None)."""
    base = dict(CONNECTORS.get(kind) or {})
    if not base and None in (insert, withdraw):
        raise ValueError(f"connector type {kind!r}: not one of {sorted(CONNECTORS)}; give insert and withdraw")
    name = name or kind
    out = _unit(out)
    mouth = np.asarray(mouth, float)
    if wide is None:
        for ax in (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]):
            w = ax - (ax @ out) * out
            if np.linalg.norm(w) > 0.5:
                wide = w
                break
    wide = _unit(np.asarray(wide, float) - (np.asarray(wide, float) @ out) * out)
    thin = np.cross(out, wide)
    F_in = float(insert if insert is not None else base["insert"])
    F_out = float(withdraw if withdraw is not None else base["withdraw"])
    sides = dict(base.get("side") or {})
    sides.update(side or {})
    L = float(lever if lever is not None else base.get("lever", 15.0))
    tip = mouth + L * out
    target = {"region": region} if isinstance(region, str) else dict(region)

    def case(nm, F, at):
        ls = [{**target, "force": _r(F, 3), "at": _r(at)}]
        if hold is not None:
            ls.append({**hold, "force": _r(-F, 3)})
        return {"name": f"{name}-{nm}", "loads": ls}

    cases = [case("insert", -F_in * out, mouth), case("withdraw", F_out * out, mouth)]
    across = {"thin": thin, "wide": wide}
    for k in ("thin", "wide"):
        if sides.get(k):
            cases.append(case(f"side-{k}", float(sides[k]) * across[k], tip))
    if abuse:
        wr = dict(base.get("wrench") or {})
        wr.update(wrench or {})
        if not wr:
            raise ValueError(f"connector type {kind!r} has no wrenching strength; give wrench {{thin, wide}} N·mm")
        for k in ("thin", "wide"):
            if wr.get(k):
                c = case(f"wrench-{k}", float(wr[k]) / L * across[k], tip)
                # the plug is allowed to give here; the product only has to
                # outlast it (a case safety factor of 1, not the goal's)
                c["safetyFactor"] = float(abuse_factor)
                cases.append(c)
    clearance = None
    if max_shift:
        clearance = {"name": f"{name}-shift", **target, "maxTravel": float(max_shift),
                     "cases": [c["name"] for c in cases]}
    return cases, clearance
