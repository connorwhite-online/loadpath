"""
A stack of boards on standoffs: which order, how far apart.

Electronics often go in as a stack: a flight controller over a speed
controller, a radio over a carrier board, held on four standoffs. The
choices are few (which board is where, how tall each gap) and each one
moves three things at once: how hot the boards run, where the stack's
mass sits, and whether the cables between the boards can be laid. This
module searches them.

    python -m loadpath stack brief.json --out out/      (`loadpath schema stack`)

    {"boards": [{"id": "esc", "size": [27, 27, 1.6], "mass": 0.0075, "watts": 7.5, "maxTemp": 105,
                 "above": 2.5, "below": 1.5},
                {"id": "fc", "size": [27, 27, 1.6], "mass": 0.0055, "watts": 0.8, "maxTemp": 85,
                 "above": 3.0, "below": 1.2}],
     "space": {"height": 22, "base": 0},
     "standoffs": {"pattern": 20, "diameter": 4.0, "material": "nylon"},
     "airflow": {"speed": 6, "direction": "+x"}, "ambient": 30,
     "goal": {"centerOfMass": "low", "weights": {"temperature": 1, "centerOfMass": 0.3, "height": 0.1}},
     "wires": [{"id": "esc-signal", "from": "esc.sig", "to": "fc.esc", "cable": "..."}]}

Heat (per candidate, a resistance network solved to steady state):

- Each board is one temperature. Each gap between boards is a channel
  between parallel plates with its own air node.
- With `airflow`, the stream's dynamic pressure drives air through each
  gap against the channel's entry, exit and wall friction losses, so a
  narrow gap gets little air (the speed is solved, not assumed). The
  faces give heat to that air by the developing-flow Nusselt number for
  parallel plates (Stephan), and the air carries it out at m cp: a gap
  can be starved by its width or by its air warming up.
- In still air a gap between horizontal boards conducts (Nu = 1) from
  face to face; open outer faces lose heat as a flat plate (cooling.
  forced_h: forced and natural mixed, plus radiation).
- Facing boards exchange radiation; standoffs conduct between them.

It ranks layouts and says which will not do; it is not a board-level
thermal analysis (one temperature per board, no hot spots: a part's
`maxTemp` should be the board temperature its hottest part allows).

The result's `inertiaRelief` entry is the stack as one carried mass at
its centre of mass, for the frame's study; `envelope` is its keep-out.
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

_SIGMA = 5.670e-8
_STANDOFF = {"nylon": (0.25, 1150.0), "aluminium": (150.0, 2700.0), "aluminum": (150.0, 2700.0),
             "brass": (110.0, 8500.0), "steel": (16.0, 7900.0), "tpu": (0.2, 1200.0)}     # W/(m K), kg/m3
_AX = {"x": 0, "y": 1, "z": 2}


def _air(t_c: float):
    t = t_c + 273.15
    k = 0.0241 * (t / 273.15) ** 0.81
    nu = 1.33e-5 * (t / 273.15) ** 1.75
    rho = 1.293 * 273.15 / t
    return k, nu, rho, 1006.0, 0.71


def gap_flow(speed: float, gap: float, length: float, t_c: float = 30.0) -> float:
    """Air speed (m/s) through a gap `gap` mm high and `length` mm long,
    between plates facing a stream of `speed` m/s: the stream's dynamic
    pressure against an entry and exit loss (K = 1.5) and laminar wall
    friction (f = 96 / Re on the hydraulic diameter 2 x gap)."""
    if speed <= 0 or gap <= 0:
        return 0.0
    _, nu, _, _, _ = _air(t_c)
    dh, L = 2.0 * gap * 1e-3, length * 1e-3
    # U^2 = u^2 (K + f L / dh), f = 96 nu / (u dh)  ->  K u^2 + (96 nu L / dh^2) u - U^2 = 0
    K, b = 1.5, 96.0 * nu * L / dh ** 2
    return float((-b + np.sqrt(b * b + 4 * K * speed ** 2)) / (2 * K))


def _channel_h(u: float, gap: float, length: float, t_c: float):
    """(h W/(m2 K) from a face to the gap's air, m cp per metre of width W/(K m))."""
    k, nu, rho, cp, pr = _air(t_c)
    dh, L = 2.0 * gap * 1e-3, length * 1e-3
    if u <= 1e-6:
        return k / (0.5 * gap * 1e-3), 0.0                    # conduction to the mid-gap
    re = u * dh / nu
    xs = max(L / (dh * re * pr), 1e-6)
    nu_m = 7.55 + 0.024 * xs ** -1.14 / (1.0 + 0.0358 * pr ** 0.17 * xs ** -0.64)
    return nu_m * k / dh, rho * u * gap * 1e-3 * cp


def temperatures(boards: list, gaps: list, *, base_gap: float, speed: float = 0.0, ambient: float = 25.0,
                 standoff: dict | None = None, flow_axis: int = 0, open_top: bool = True, open_bottom: bool = False,
                 emissivity: float = 0.9) -> dict:
    """Steady board temperatures (C) of a stack, bottom board first.
    boards: [{size [x, y, t] mm, watts}], gaps: mm between consecutive
    boards (len(boards) - 1), base_gap: mm under the bottom board (to the
    plate it stands on, taken as adiabatic unless open_bottom)."""
    from .cooling import forced_h

    n = len(boards)
    A = [b["size"][0] * b["size"][1] * 1e-6 for b in boards]
    Lf = [b["size"][flow_axis] for b in boards]
    Wd = [b["size"][1 - flow_axis] * 1e-3 for b in boards]
    q = np.array([float(b.get("watts") or 0.0) for b in boards])
    so = standoff or {}
    k_so, _ = _STANDOFF.get(str(so.get("material", "nylon")).lower(), _STANDOFF["nylon"])
    a_so = int(so.get("count", 4)) * np.pi * (0.5 * float(so.get("diameter", 4.0)) * 1e-3) ** 2 * \
        float(so.get("fill", 0.6))
    T = np.full(n, ambient + 20.0)
    info = {}
    for _ in range(30):
        # unknowns: boards 0..n-1, then one air node per gap (the base gap first)
        m = n + n
        G = np.zeros((m, m))
        rhs = np.zeros(m)

        def link(i, j, g):
            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g

        def to_ambient(i, g):
            G[i, i] += g
            rhs[i] += g * ambient
        rhs[:n] += q
        speeds = []
        for gi in range(n):                    # gap gi is under board gi
            gap = base_gap if gi == 0 else gaps[gi - 1]
            faces = [gi] if gi == 0 else [gi - 1, gi]
            tm = float(np.mean([T[f] for f in faces]))
            a = min(A[f] for f in faces)
            Lc, Wc = min(Lf[f] for f in faces), min(Wd[f] for f in faces)
            u = gap_flow(speed, gap, Lc, 0.5 * (tm + ambient)) if (gi > 0 or gap > 0.5) else 0.0
            speeds.append(u)
            h, mcp = _channel_h(u, max(gap, 0.05), Lc, 0.5 * (tm + ambient))
            air = n + gi
            for f in faces:
                link(f, air, h * a)
            if mcp > 0:
                to_ambient(air, 2.0 * mcp * Wc)            # mean air temperature: half the outlet rise
            else:
                G[air, air] += 1e-9                        # still: the air node only joins the faces
            if gi > 0:
                tk = 0.5 * (T[gi - 1] + T[gi]) + 273.15
                e = 1.0 / (2.0 / emissivity - 1.0)
                link(gi - 1, gi, e * _SIGMA * 4 * tk ** 3 * a)                     # radiation, face to face
                link(gi - 1, gi, k_so * a_so / max(gap * 1e-3, 1e-4))              # the standoffs
        for i, is_open in ((n - 1, open_top), (0, open_bottom)):
            if is_open:
                rise = max(T[i] - ambient, 1.0)
                hh = forced_h(speed, Lf[i], ambient, rise, emissivity)["h"]
                if speed <= 0 and i == 0:
                    hh *= 0.5                                                      # a hot face looking down
                to_ambient(i, hh * A[i])
        # every board's edges, and a floor so a sealed stack still has an answer
        for i in range(n):
            per = 2 * (boards[i]["size"][0] + boards[i]["size"][1]) * boards[i]["size"][2] * 1e-6
            to_ambient(i, 10.0 * per)
        Tn = np.linalg.solve(G, rhs)[:n]
        done = np.abs(Tn - T).max() < 0.05
        T = 0.5 * T + 0.5 * Tn
        info = {"gapSpeeds": [round(float(v), 2) for v in speeds]}
        if done:
            break
    return {"temperatures": [round(float(t), 1) for t in T], **info}


def _axis(v, default=0):
    if v is None:
        return default
    return _AX[str(v).strip().lstrip("+-").lower()]


def expand_boards(brief: dict, base_dir: str | None = None) -> dict:
    """The brief with each board that names a KiCad file (`kicad`) filled
    from it: size, mass, watts, maxTemp, above, below, ports
    (kicad.stack_board), and standoffs.count from its mounting holes when
    the brief does not say."""
    if not any(b.get("kicad") for b in brief.get("boards") or []):
        return brief
    from .kicad import stack_board
    out = dict(brief)
    boards, notes = [], []
    for b in brief["boards"]:
        if b.get("kicad"):
            nb = stack_board(b, base_dir)
            info = nb.pop("_kicad")
            notes.append({"id": b["id"], "file": info["file"], "holes": len(info["holes"]),
                          "warnings": info["warnings"]})
            boards.append(nb)
        else:
            boards.append(b)
    out["boards"] = boards
    so = dict(out.get("standoffs") or {})
    if "count" not in so:
        n = {x["holes"] for x in notes if x["holes"]}
        if len(n) == 1:
            so["count"] = n.pop()
            out["standoffs"] = so
    out["_kicad"] = notes
    return out


def stack(brief: dict, on_progress=None, base_dir: str | None = None) -> dict:
    """Search a board stack's order and gaps. See the module docstring and
    `loadpath schema stack`. Returns {best, alternatives, rejected (better
    on paper, with why not), asGiven, inertiaRelief, envelope, tried,
    seconds}. A board may be a KiCad file (expand_boards)."""
    t0 = time.time()
    brief = expand_boards(brief, base_dir)
    boards = [dict(b) for b in brief["boards"]]
    ids = [b["id"] for b in boards]
    if len(set(ids)) != len(ids):
        raise ValueError("stack: board ids must be unique")
    if len(boards) > 6:
        raise ValueError("stack: at most 6 boards (the orders are searched exhaustively)")
    sp = brief.get("space") or {}
    Hmax = float(sp.get("height", 1e9))
    base = float(sp.get("base", 0.0))
    centre = np.asarray(sp.get("center", sp.get("centre", [0.0, 0.0])), float)
    so = dict(brief.get("standoffs") or {})
    k_so, rho_so = _STANDOFF.get(str(so.get("material", "nylon")).lower(), _STANDOFF["nylon"])
    d_so = float(so.get("diameter", 4.0))
    count = int(so.get("count", 4))
    base_gap = float(so.get("baseGap", sp.get("baseGap", 2.0)))
    clear = float(brief.get("clearance", 0.5))
    step = float(brief.get("gapStep", 1.0))
    af = brief.get("airflow") or {}
    speed = float(af.get("speed", 0.0))
    flow_axis = _axis(af.get("direction"), 0)
    if flow_axis == 2:
        raise ValueError("stack: airflow.direction is along the boards (x or y); through the stack is not modelled")
    ambient = float(brief.get("ambient", 25.0))
    goal = brief.get("goal") or {}
    W = {"temperature": 1.0, "centerOfMass": 0.3, "height": 0.1, **(goal.get("weights") or {})}
    com_goal = goal.get("centerOfMass", "low")
    wires = brief.get("wires") or []
    byid = {b["id"]: b for b in boards}
    for b in boards:
        if b.get("position") not in (None, "top", "bottom"):
            raise ValueError(f"stack: board {b['id']!r} position is 'top' or 'bottom'")

    def layout(order, gaps):
        """z of each board's underside, the stack's top, mass and centre of mass."""
        z, zs = base + base_gap, []
        for i, bid in enumerate(order):
            zs.append(z)
            z += byid[bid]["size"][2] + (gaps[i] if i < len(gaps) else 0.0)
        top = zs[-1] + byid[order[-1]]["size"][2] + float(byid[order[-1]].get("above", 0.0))
        m_so = count * np.pi * (0.5 * d_so * 1e-3) ** 2 * rho_so * float(so.get("fill", 0.6)) * \
            (zs[-1] - base) * 1e-3
        ms = [float(byid[b].get("mass", 0.0)) for b in order]
        zc = [zs[i] + 0.5 * byid[b]["size"][2] for i, b in enumerate(order)]
        M = sum(ms) + m_so
        zcm = (sum(m * z for m, z in zip(ms, zc)) + m_so * 0.5 * (zs[-1] + base)) / max(M, 1e-12)
        return zs, top, M, zcm

    def evaluate(order, gaps):
        zs, top, M, zcm = layout(order, gaps)
        th = temperatures([byid[b] for b in order], list(gaps), base_gap=base_gap, speed=speed, ambient=ambient,
                          standoff=so, flow_axis=flow_axis, open_top=bool(sp.get("openTop", True)),
                          open_bottom=bool(sp.get("openBottom", False)))
        T = th["temperatures"]
        used, hot = 0.0, []
        for bid, t in zip(order, T):
            lim = byid[bid].get("maxTemp")
            if lim is not None:
                u = (t - ambient) / max(float(lim) - ambient, 1.0)
                used = max(used, u)
                if t > float(lim):
                    hot.append(f"{bid} runs at {t:.0f} C; it allows {float(lim):.0f}")
            else:
                used = max(used, (t - ambient) / 60.0)
        span = max(Hmax - base, 1.0) if Hmax < 1e8 else max(top - base, 1.0)
        if com_goal == "low":
            cm = (zcm - base) / span
        elif com_goal == "high":
            cm = 1.0 - (zcm - base) / span
        else:
            cm = abs(zcm - float(com_goal)) / span
        cost = W["temperature"] * used + W["centerOfMass"] * cm + W["height"] * (top - base) / span
        return {"order": list(order), "gaps": [round(float(g), 2) for g in gaps],
                "z": {b: round(float(z), 2) for b, z in zip(order, zs)}, "height": round(float(top - base), 2),
                "mass": round(float(M), 5), "centerOfMass": [round(float(centre[0]), 2), round(float(centre[1]), 2),
                                                             round(float(zcm), 2)],
                "temperatures": {b: t for b, t in zip(order, T)}, "gapAirSpeed": th["gapSpeeds"][1:],
                "budgetUsed": round(float(used), 3), "cost": round(float(cost), 4), "problems": hot,
                "fits": bool(top <= Hmax + 1e-9)}

    def min_gap(lower, upper):
        return float(byid[lower].get("above", 0.0)) + float(byid[upper].get("below", 0.0)) + clear

    cands = []
    for order in itertools.permutations(ids):
        if any(byid[b].get("position") == "bottom" and i != 0 or
               byid[b].get("position") == "top" and i != len(order) - 1 for i, b in enumerate(order)):
            continue
        mins = [min_gap(order[i], order[i + 1]) for i in range(len(order) - 1)]
        _, top0, _, _ = layout(order, mins)
        room = Hmax - top0 if Hmax < 1e8 else float(brief.get("maxExtraGap", 10.0)) * max(len(mins), 1)
        if room < -1e-9:
            cands.append((order, mins, None))
            continue
        nmax = int(np.floor(min(room, float(brief.get("maxExtraGap", 10.0)) * max(len(mins), 1)) / step + 1e-9))
        per = range(0, min(nmax, int(float(brief.get("maxExtraGap", 10.0)) / step)) + 1)
        for extra in itertools.product(per, repeat=len(mins)):
            if sum(extra) > nmax:
                continue
            cands.append((order, [m + e * step for m, e in zip(mins, extra)], True))
    if on_progress:
        on_progress({"stage": "stack", "message": f"{len(cands)} stacks to try"})
    scored = []
    unfit = []
    for order, gaps, ok in cands:
        r = evaluate(order, gaps)
        if ok is None or not r["fits"]:
            r["problems"] = r["problems"] + [f"{r['height']:.1f} mm tall at its tightest; the space is "
                                             f"{Hmax - base:.1f}"]
            unfit.append(r)
            continue
        scored.append(r)
    scored.sort(key=lambda r: (bool(r["problems"]), r["cost"]))
    # the cables: laid for the best few, best first, until enough stacks pass
    keep, rejected, laid_best = [], [], None
    want = int(brief.get("alternatives", 4)) + 1
    lays = 0
    for r in scored:
        if len(keep) >= want or lays >= int(brief.get("maxLays", 60)):
            break
        if r["problems"]:
            rejected.append(r)
            continue
        if wires:
            wr, laid = _stack_wires(wires, byid, r, centre, on_progress)
            lays += 1
            r["wires"] = wr
            for wid, rep in wr.items():
                r["problems"] += [f"{wid}: {p}" for p in rep.get("problems", [])]
            if r["problems"]:
                rejected.append(r)
                continue
            if laid_best is None:
                laid_best = laid
        keep.append(r)
    if not keep:
        keep = rejected[:1]
        rejected = rejected[1:]
    keep.sort(key=lambda r: (bool(r["problems"]), r["cost"]))
    best = keep[0] if keep else (unfit[0] if unfit else None)
    given = None
    try:
        g0 = [min_gap(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
        given = evaluate(ids, g0)
    except Exception:
        pass
    out = {"name": brief.get("name", "stack"), "ok": bool(best and not best["problems"]), "best": best,
           "alternatives": keep[1:], "rejected": rejected[:6], "asGiven": given, "tried": len(cands), "seconds": round(time.time() - t0, 1),
           "airflow": {"speed": speed, "direction": "xyz"[flow_axis]}, "ambient": ambient}
    if best:
        sx = max(b["size"][0] for b in boards)
        sy = max(b["size"][1] for b in boards)
        out["envelope"] = {"type": "box", "min": [float(centre[0] - sx / 2 - 1), float(centre[1] - sy / 2 - 1), base],
                           "max": [float(centre[0] + sx / 2 + 1), float(centre[1] + sy / 2 + 1),
                                   base + best["height"]]}
        mount = brief.get("mount")
        entry = {"mass": best["mass"], "at": best["centerOfMass"]}
        if isinstance(mount, str):
            entry["region"] = mount
        elif isinstance(mount, dict):
            entry["shape"] = mount
        out["inertiaRelief"] = {"masses": [entry]}
        if not best["problems"] and laid_best is not None:
            out["laid"] = laid_best
    if not out["ok"] and best:
        out["problems"] = best["problems"]
    if brief.get("_kicad"):
        out["boards"] = {x["id"]: {**{k: v for k, v in byid[x["id"]].items() if k != "ports"},
                                   "file": x["file"], "warnings": x["warnings"]} for x in brief["_kicad"]}
    return out


def _stack_wires(wires, byid, r, centre, on_progress=None):
    """Lay each cable of a candidate stack as a rod among its boards. A
    port is {id, at [x, y, z], direction} in its board's frame: the origin
    at the board's centre, z = 0 its underside."""
    from . import wires as W
    from .shapes import shape_from_spec

    fields = []
    for bid, z in r["z"].items():
        sx, sy, t = byid[bid]["size"]
        box = {"type": "box", "min": [centre[0] - sx / 2, centre[1] - sy / 2, z],
               "max": [centre[0] + sx / 2, centre[1] + sy / 2, z + t]}
        fields.append(W.body_field(shape_from_spec(box), 0.4))

    def port(spec):
        if isinstance(spec, dict):
            return spec
        bid, _, pid = spec.partition(".")
        if bid not in byid:
            raise ValueError(f"stack wire end {spec!r}: no board {bid!r}")
        p = next((p for p in byid[bid].get("ports") or [] if p.get("id") == pid), None)
        if p is None:
            raise ValueError(f"stack wire end {spec!r}: board {bid!r} has no port {pid!r}")
        at = np.asarray(p["at"], float) + np.array([centre[0], centre[1], r["z"][bid]])
        return {**{k: v for k, v in p.items() if k != "id"}, "at": at.tolist()}
    out, laid = {}, []
    pts = []
    ends = []
    for w in wires:
        a, b = W._end(port(w["from"])), W._end(port(w["to"]))
        ends.append((a, b))
        pts += [a["at"], b["at"]]
    pts = np.array(pts)
    room = W._Room.of(fields, pts.min(axis=0) - 15.0, pts.max(axis=0) + 15.0, 0.5)
    extra = []
    for i, (w, (a, b)) in enumerate(zip(wires, ends)):
        wid = w.get("id") or f"wire-{i}"
        cab = W.cable(w["cable"])
        lw, rep, _ = W._lay_best(cab, a, b, fields + extra, room, {**w, "id": wid}, None, 0.0, int(w.get("routes", 1)))
        if lw is None:
            out[wid] = {"cable": cab.name, "ok": False, "problems": ["no way from one port to the other"]}
            continue
        out[wid] = {k: v for k, v in rep.items() if k not in ("cableSpec",)}
        laid.append(lw)
        extra.append(W.PolylineField(lw.rod.x, cab.half_width))
    return out, laid


def run_stack(brief_path: str, out_dir: str, on_progress=None, base_dir: str | None = None) -> dict:
    """`loadpath stack brief.json --out dir`: writes stack.json (and
    wires.html when the brief has cables)."""
    with open(brief_path) as fh:
        brief = json.load(fh)
    os.makedirs(out_dir, exist_ok=True)
    brief = expand_boards(brief, base_dir or os.path.dirname(os.path.abspath(brief_path)))
    res = stack(brief, on_progress)
    laid = res.pop("laid", None)
    if laid and res.get("best"):
        from . import wires as W
        from .shapes import shape_from_spec, trimesh_from_manifold
        meshes = {}
        byid = {b["id"]: b for b in brief["boards"]}
        c = (brief.get("space") or {}).get("center", [0.0, 0.0])
        for bid, z in res["best"]["z"].items():
            sx, sy, t = byid[bid]["size"]
            sh = shape_from_spec({"type": "box", "min": [c[0] - sx / 2, c[1] - sy / 2, z],
                                  "max": [c[0] + sx / 2, c[1] + sy / 2, z + t]})
            meshes[bid] = trimesh_from_manifold(sh.to_manifold())
        hv = {"name": res["name"], "wires": res["best"]["wires"], "ok": res["ok"], "problems": [], "laid": laid,
              "bodies": {b: {"moves": False, "inside": False, "spec": {}} for b in meshes}}
        try:
            W.harness_viewer(hv, meshes, os.path.join(out_dir, "wires.html"))
            res["viewer"] = "wires.html"
        except Exception as exc:
            res.setdefault("warnings", []).append(f"viewer not written: {exc}")
    with open(os.path.join(out_dir, "stack.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res
