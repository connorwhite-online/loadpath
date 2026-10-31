"""
Fasteners as interfaces, not bosses.

A bolted joint needs very little of the part to be fixed: a flat ring for
the head or washer to bear on, a flat ring where it clamps against the
mating part (or a sleeve of material for the thread, an insert or a nut), a
clear hole, and a clear path for the head and the driver. Drawing a solid
boss round all of that (the old habit) preserves material the joint does not
need; drawing only the hole loses the clamp. This expands a `fasteners`
entry into exactly those preserve and keep-out regions, plus a self-balanced
preload case that squeezes the seat towards the far end, so the optimizer
grows its own compression cone between them instead of inheriting a solid
cylinder.

    "fasteners": [{
      "id": "top", "size": "M5",
      "seat": [6, 0, -20],          # centre of the face the head bears on
      "direction": "-x",            # from the seat into the part, the way the screw goes
      "grip": 6,                    # mm from the seat to the far end
      "end": "face",                # "face" (clamps on a mating part), "thread", or "none"
      "headOn": "this",             # "mating": the head is on the mating part, which bears on
                                    # this part at the seat (no head or driver path to keep clear)
      "threadLength": 8,            # end = "thread": engagement length past the grip
      "insert": 6.4,                # optional: heat-set insert or nut OD; else a tapped hole
      "preload": 600                # N; default from the seat's bearing area
    }]

A screw driven from the mating part into this one is `"headOn": "mating"`,
`"end": "thread"`: the seat is then where the mating part clamps down, and
only the bore and the tap are kept clear.

Regions it adds, by id: `<id>-bore`, `<id>-head` (unless the head is on
the other part; and `<id>-tap` for a thread) as keep-outs; `<id>-seat` and `<id>-face` or `<id>-thread` as
preserves. Supports and loads can name any of them. Hold a bore across
the screw only (`"fix": "yz"` for a screw along x): fixed along it too,
the bore would hold the preload ring and the preload would do nothing.
"""
from __future__ import annotations

import copy

import numpy as np

__all__ = ["expand_fasteners", "METRIC"]

# ISO metric: (clearance hole, pan/socket head or washer diameter), mm
METRIC = {"M2": (2.4, 4.5), "M2.5": (2.9, 5.5), "M3": (3.4, 6.5), "M4": (4.5, 8.5),
          "M5": (5.5, 10.0), "M6": (6.6, 12.0), "M8": (9.0, 16.0), "M10": (11.0, 20.0)}
_AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def _direction(v, label):
    if isinstance(v, str):
        s = v.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        ax = s.lstrip("+-")
        if ax not in _AXES:
            raise ValueError(f"{label}.direction must be like '-x' or a vector, got {v!r}")
        return sign * np.asarray(_AXES[ax], float)
    a = np.asarray(v, float).reshape(-1)
    if a.shape != (3,) or not np.linalg.norm(a) > 0:
        raise ValueError(f"{label}.direction must be like '-x' or a non-zero vector")
    return a / np.linalg.norm(a)


def _tube(p0, p1, r, r_inner=0.0):
    out = {"type": "cylinder", "p0": np.round(p0, 4).tolist(), "p1": np.round(p1, 4).tolist(), "r": float(r)}
    if r_inner:
        out["rInner"] = float(r_inner)
    return out


def expand_fasteners(d: dict) -> dict:
    """The study with each `fasteners` entry turned into regions and a
    preload case. Returns d itself when there are none."""
    fs = d.get("fasteners")
    if not fs:
        return d
    from .materials import material_from_spec

    d = copy.deepcopy(d)
    preserve = list(d.get("preserve") or [])
    keep_out = list(d.get("keepOut") or [])
    mat_in = d.get("material", "pla")
    mat = material_from_spec(mat_in[0] if isinstance(mat_in, list) else mat_in)
    preload_loads = []
    joints = []
    for i, f in enumerate(fs):
        label = f"fasteners[{i}]"
        fid = f.get("id") or f"fastener-{i}"
        size = f.get("size")
        if size is not None and size not in METRIC:
            raise ValueError(f"{label}.size must be one of {sorted(METRIC)} (or give diameter and head)")
        hole, head = METRIC[size] if size else (None, None)
        hole = float(f.get("hole", hole or 0.0)) or (1.1 * float(f["diameter"]) if "diameter" in f else None)
        head = float(f.get("head", head or 0.0)) or None
        if not hole or not head:
            raise ValueError(f"{label} needs a size (e.g. 'M5') or diameter and head (mm)")
        nominal = float(f.get("diameter", hole / 1.1))
        if "seat" not in f or "direction" not in f or "grip" not in f:
            raise ValueError(f"{label} needs seat (point), direction (into the part) and grip (mm)")
        seat = np.asarray(f["seat"], float)
        a = _direction(f["direction"], label)
        grip = float(f["grip"])
        end_kind = str(f.get("end", "face"))
        if end_kind not in ("face", "thread", "none"):
            raise ValueError(f"{label}.end must be 'face', 'thread' or 'none'")
        head_on = str(f.get("headOn", "this"))
        if head_on not in ("this", "mating"):
            raise ValueError(f"{label}.headOn must be 'this' or 'mating'")
        # the rings: a little wider than the head, a couple of layers thick;
        # the thread sleeve's wall about a third of the screw's diameter.
        # (The optimizer adds what the loads need around them.)
        wall = float(f.get("wall", max(1.0, 0.15 * nominal)))
        ring = float(f.get("ring", max(2.0, 0.4 * nominal)))
        sleeve = float(f.get("sleeveWall", max(1.5, 0.35 * nominal)))
        driver = float(f.get("driver", 60.0))
        end = seat + a * grip
        r_ring = 0.5 * head + wall
        keep_out.append({"id": f"{fid}-bore", "shape": _tube(seat - a * 1.0, end + a * 2.0, 0.5 * hole)})
        if head_on == "this":
            # as wide as the seat ring: a head path narrower than the ring
            # leaves a sub-millimetre lip of ring standing round it
            keep_out.append({"id": f"{fid}-head", "shape": _tube(seat, seat - a * driver, max(0.5 * head + 0.5, r_ring))})
        preserve.append({"id": f"{fid}-seat", "shape": _tube(seat, seat + a * ring, r_ring, 0.5 * hole)})
        far = None
        if end_kind == "face":
            far = f"{fid}-face"
            preserve.append({"id": far, "shape": _tube(end - a * ring, end, r_ring, 0.5 * hole)})
        elif end_kind == "thread":
            tl = float(f.get("threadLength", 2.0 * nominal))
            insert = f.get("insert")
            r_hole = 0.5 * float(insert) if insert else 0.42 * nominal     # insert bore, or tap drill
            r_sleeve = (0.5 * float(insert) if insert else 0.5 * nominal) + sleeve
            far = f"{fid}-thread"
            keep_out.append({"id": f"{fid}-tap", "shape": _tube(end - a * 0.5, end + a * (tl + 1.0), r_hole)})
            preserve.append({"id": far, "shape": _tube(end, end + a * tl, r_sleeve, r_hole)})
        if far is not None:
            if "preload" in f:
                p = float(f["preload"])
            else:
                # a quarter of what the seat can bear: plastics creep under
                # sustained bearing stress
                area = 0.25 * np.pi * (head ** 2 - hole ** 2)
                p = 0.25 * mat.strength * area
                # ...and no more than a class 8.8 screw is tightened to (about
                # 60% of its proof load on the stress area): a titanium seat
                # read 10 kN for an M4, three times what the screw carries
                screw = 0.6 * 600.0 * 0.25 * np.pi * (0.83 * nominal) ** 2
                p = min(p, screw)
            if p > 0:
                joints.append({"id": fid, "preload": p, "area": 0.25 * np.pi * (head ** 2 - hole ** 2)})
                preload_loads.append({"region": f"{fid}-seat", "force": (a * p).tolist()})
                preload_loads.append({"region": far, "force": (-a * p).tolist()})
    d["preserve"] = preserve
    d["keepOut"] = keep_out
    if preload_loads:
        cases = list(d.get("loadCases") or [])
        cases.append({"name": "preload", "loads": preload_loads})
        d["loadCases"] = cases
    d.pop("fasteners")
    # for the durability check's preload relaxation (spec pops it)
    d["_joints"] = joints
    return d
