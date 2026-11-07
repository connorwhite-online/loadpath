"""
Feel targets: design the inside of a fixed exterior for how the product
feels in the hand, with the least material.

An industrial designer hands over the surfacing; the parts that sit inside
(a board, a battery, a microphone capsule) are known, and so is how the
product should feel. The push pad's click force is one such target
(switch.py). This module adds three more, each a stiffness window between
where the product is held and where the load comes in:

- `squeeze`: how much a grip gives. A hand squeezing the body with
  `force` N should close it by `travel` [min, max] mm: soft-touch, but not
  mushy, and not creaky-hard. Held on one side (`hold`), pressed on the
  other (`press`) along `direction`.

- `drop`: cushioning a heavy or fragile inside part (`mass`, `kg`) when the
  product falls `height` m and lands on `lands` (an exterior face, or a
  patch of it). The structure between the landing patch and the part is a
  spring k: with the fall's energy E = m g h, the peak force is
  sqrt(2 E k) and the part's travel sqrt(2 E / k). So a peak deceleration
  under `maxG` caps k from above, and a travel under `maxTravel` (before
  the part hits something) bounds it from below:

      2 E / maxTravel^2  <=  k  <=  (maxG m g)^2 / (2 E)

  which needs maxTravel >= 2 h / maxG: a 1 m fall at 1500 g needs at least
  1.3 mm of travel whatever the structure is. A linear spring is the worst
  shape a cushion can have (a constant-force crush would halve the travel,
  impact.py); the shell's own mass is left out (the part's deceleration is
  what is asked for).

- `isolate`: keeping handling vibration away from a part (a microphone
  capsule on its mount, `kg`) while the hand (`hold`) shakes the body. The
  mount and the structure to the hand are a suspension of natural
  frequency f_n = sqrt(k / m) / 2 pi per direction, and with structural
  damping (the material's loss factor eta, modal.LOSS_FACTOR)

      T(r) = sqrt((1 + eta^2) / ((1 - r^2)^2 + eta^2)),  r = f / f_n

  There are two ways to keep a band of handling noise off the part. A soft
  mount (`band`): from `band` Hz up the vibration reaches the part at no
  more than `maxTransmissibility`, which caps f_n, so k. A stiff mount
  (`upTo`): the resonance sits over the handling band, so nothing up to
  `upTo` Hz is amplified more than `maxAmplification` (default 1.25),
  which sets a floor on f_n. A part hard-mounted to the exterior (a board
  on rails moulded into the base) can only be the second. A floor from
  `maxTravel` under a `shockG` knock (default 1: the part's own weight)
  keeps it from rattling.

All three are compliance windows c = f . K^-1 f (f: 1 N spread over the
loaded nodes), each with its own holds, so each costs one solve per
direction per iteration, and its sensitivity is the solve itself:
dc/dx_e = -u_e . dK_e u_e. They are written on log c, like the push pad's
log F, and go to the same optimizer (switch.optimize_switch: minimum
volume, MMA, thin-skin floors, coarse to fine) as extra constraints, so one
study can carry a click force and a grip feel together.

After the run, each target is read again on the black-and-white design
and on the delivered mesh, and an `isolate` target is checked with the
modes of the part with the capsule as a point mass (modal.py): the
transmissibility at the band from modal superposition, the shell and the
structure's own mass included.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .domain import Domain
from .fem import hex8_ke
from .switch import _AXES, _axis, _pair, _Solves

__all__ = ["FeelTarget", "feel_from_study", "feel_readings", "run_feel", "drop_window", "isolation_window",
           "G0"]

G0 = 9.80665          # m/s^2


# ---------------------------------------------------------------------------
# windows from the product's numbers


def drop_window(kg: float, height: float, max_g: float | None = None, max_travel: float | None = None):
    """(k_min, k_max) N/mm for a part of `kg` on an elastic cushion, dropped
    `height` m: peak deceleration under max_g (in g) and travel under
    max_travel (mm). Either may be None."""
    energy = kg * G0 * height * 1000.0              # N mm
    kmax = (max_g * kg * G0) ** 2 / (2.0 * energy) if max_g else None
    kmin = 2.0 * energy / max_travel ** 2 if max_travel else None
    return kmin, kmax


def drop_response(k: float, kg: float, height: float) -> dict:
    """Peak force, deceleration and travel of `kg` dropped `height` m onto a
    linear spring of k N/mm."""
    energy = kg * G0 * height * 1000.0
    force = float(np.sqrt(2.0 * energy * k))
    return {"peakForce": force, "peakG": force / (kg * G0), "travel": float(np.sqrt(2.0 * energy / k)),
            "seconds": float(np.pi * np.sqrt(kg / (k * 1000.0)))}


def isolation_ratio(t_max: float, eta: float) -> float:
    """The lowest r = f / f_n at which a structurally damped mode transmits
    at most t_max (< 1)."""
    if not 0 < t_max < 1:
        raise ValueError("maxTransmissibility must be between 0 and 1 (isolation means less than 1)")
    disc = (1.0 + eta ** 2) / t_max ** 2 - eta ** 2
    return float(np.sqrt(1.0 + np.sqrt(disc)))


def resonance_ratio(max_amp: float, eta: float) -> float:
    """The highest r = f / f_n (under 1) at which a structurally damped mode
    amplifies at most max_amp (> 1): below it the mount moves with the
    hand."""
    if max_amp <= 1:
        raise ValueError("maxAmplification must be over 1")
    disc = (1.0 + eta ** 2) / max_amp ** 2 - eta ** 2
    if disc <= 0:
        return 1.0
    return float(np.sqrt(max(1.0 - np.sqrt(disc), 1e-12)))


def isolation_window(kg: float, band: float | None, max_t: float, eta: float, max_travel: float | None = None,
                     shock_g: float = 1.0, up_to: float | None = None, max_amp: float = 1.25):
    """(k_min, k_max) N/mm for a part of `kg`: isolated from `band` Hz up at
    transmissibility max_t (a soft mount: a ceiling), and/or with its
    resonance kept over a handling band that reaches `up_to` Hz, amplifying
    it at most max_amp (a stiff mount: a floor); travelling under
    max_travel mm at shock_g (a floor)."""
    kmax = kmin = None
    if band:
        fn = band / isolation_ratio(max_t, eta)
        kmax = kg * (2 * np.pi * fn) ** 2 / 1000.0
    if up_to:
        fn = up_to / resonance_ratio(max_amp, eta)
        kmin = kg * (2 * np.pi * fn) ** 2 / 1000.0
    if max_travel:
        kt = kg * G0 * shock_g / max_travel
        kmin = kt if kmin is None else max(kmin, kt)
    return kmin, kmax


# ---------------------------------------------------------------------------
# targets on a mesh


@dataclass
class FeelTarget:
    kind: str                       # squeeze | drop | isolate
    name: str
    ground: np.ndarray              # bool per node: held (all three directions)
    load: np.ndarray                # bool per node: where the 1 N goes
    directions: list                # [(axis, sign)]
    kmin: float | None              # N/mm
    kmax: float | None
    params: dict = field(default_factory=dict)
    key: str = ""                   # the hold's identity: targets that share it share a factorization


def _face_nodes(domain: Domain, axis: int, sign: int, within=None) -> np.ndarray:
    """The outermost node of every node column along `axis` on its `sign`
    side (the part's exterior facing that way), optionally only inside
    `within` (a shape)."""
    mesh = domain.mesh
    xyz = mesh.node_xyz
    cand = np.ones(len(xyz), bool)
    if within is not None:
        cand = within.sdf(xyz) <= 0.5 * mesh.pitch
    idx = np.flatnonzero(cand)
    if not len(idx):
        raise ValueError("face: the shape it is limited to touches no node of the part")
    others = [i for i in range(3) if i != axis]
    ijk = mesh.node_ijk[idx]
    col = ijk[:, others[0]] * (int(mesh.node_ijk[:, others[1]].max()) + 2) + ijk[:, others[1]]
    order = np.lexsort((-sign * xyz[idx, axis], col))
    first = np.ones(len(order), bool)
    first[1:] = col[order][1:] != col[order][:-1]
    out = np.zeros(len(xyz), bool)
    out[idx[order[first]]] = True
    return out


def _where(study, domain: Domain, spec, label: str) -> np.ndarray:
    """Nodes of a location: {"region": id}, {"shape": {...}}, or {"face":
    "-z"} (the exterior facing -z). A region or a face with a "shape" too is
    only the part of it inside the shape."""
    from .domain import _region_nodes
    from .shapes import shape_from_spec
    if isinstance(spec, str):
        spec = {"region": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"{label}: a location is {{region}}, {{shape}} or {{face}}")
    shape = shape_from_spec(spec["shape"], getattr(study, "base_dir", None)) if "shape" in spec else None
    if "face" in spec:
        ax, sg = _axis(spec["face"])
        nodes = _face_nodes(domain, ax, sg, shape)
    elif "region" in spec:
        # a region, optionally only where it is inside the shape (the
        # skin where the hand is)
        nodes = _region_nodes(study, domain.mesh, spec["region"], None, label)
        if shape is not None:
            nodes &= shape.sdf(domain.mesh.node_xyz) <= 0.5 * domain.mesh.pitch
    elif shape is not None:
        nodes = _region_nodes(study, domain.mesh, None, shape, label)
    else:
        raise ValueError(f"{label}: give a region, a shape or a face")
    if not nodes.any():
        raise ValueError(f"{label}: no node of the part there")
    return nodes


def _rng(v):
    """[min, max] or {min, max} or a single max -> (min, max)."""
    if v is None:
        return None, None
    if isinstance(v, dict):
        return v.get("min"), v.get("max")
    if isinstance(v, (list, tuple)):
        return (v[0], v[1]) if len(v) == 2 else (None, v[0])
    return None, float(v)


def feel_from_study(study, domain: Domain) -> list[FeelTarget]:
    """The study's `feel.targets` on this domain's mesh."""
    import json

    from .modal import loss_factor
    b = study.raw.get("feel") or {}
    out = []
    eta = float(b.get("lossFactor") or loss_factor(study.material))
    for i, t in enumerate(b.get("targets") or []):
        kind = t.get("kind")
        name = t.get("name") or f"{kind}-{i}"
        lab = f"feel.targets[{i}]"
        if kind == "squeeze":
            hold_spec, load_spec = t["hold"], t["press"]
            dirs = [_axis(t["direction"])]
            if "stiffness" in t:
                kmin, kmax = _rng(t["stiffness"])     # N/mm, [min, max]
            else:
                force = float(t["force"])
                lo, hi = _rng(t["travel"])
                kmin = force / hi if hi else None
                kmax = force / lo if lo else None
            params = {k: t[k] for k in ("force", "travel", "stiffness") if k in t}
        elif kind == "drop":
            hold_spec, load_spec = t["lands"], t["mass"]
            if "direction" in t:
                d = _axis(t["direction"])
            elif isinstance(t["lands"], dict) and "face" in t["lands"]:
                d = _axis(t["lands"]["face"])     # it falls toward the face it lands on
            else:
                raise ValueError(f"{lab}: a drop that does not land on a face needs a direction")
            dirs = [d]
            kmin, kmax = drop_window(float(t["kg"]), float(t["height"]), t.get("maxG"), t.get("maxTravel"))
            params = {k: t[k] for k in ("kg", "height", "maxG", "maxTravel") if k in t}
        elif kind == "isolate":
            hold_spec, load_spec = t["hold"], t["mount"]
            dirs = [(_AXES[a.strip("+-").lower()], 1) for a in t.get("directions", ["x", "y", "z"])]
            if not (t.get("band") or t.get("upTo")):
                raise ValueError(f"{lab}: isolate needs a band (a soft mount: isolate from band Hz up) or upTo "
                                 "(a stiff one: keep the resonance over a handling band up to upTo Hz)")
            mt = float(t.get("maxTransmissibility", 0.5))
            ma = float(t.get("maxAmplification", 1.25))
            kmin, kmax = isolation_window(float(t["kg"]), t.get("band"), mt, eta, t.get("maxTravel"),
                                          float(t.get("shockG", 1.0)), t.get("upTo"), ma)
            params = {"kg": float(t["kg"]), "lossFactor": eta,
                      **{k: t[k] for k in ("band", "upTo", "maxTravel", "shockG") if k in t}}
            if t.get("band"):
                params["maxTransmissibility"] = mt
                params["maxFrequency"] = float(t["band"]) / isolation_ratio(mt, eta)
            if t.get("upTo"):
                params["maxAmplification"] = ma
                params["minFrequency"] = float(t["upTo"]) / resonance_ratio(ma, eta)
        else:
            raise ValueError(f"{lab}: kind must be squeeze, drop or isolate, got {kind!r}")
        if kmin is not None and kmax is not None and kmin > kmax:
            why = ""
            if kind == "drop":
                why = (f": a {t['height']} m fall at under {t.get('maxG')} g needs at least "
                       f"{2000 * float(t['height']) / float(t['maxG']):.2f} mm of travel, whatever is inside")
            raise ValueError(f"{lab} ({name}) asks for a stiffness of at least {kmin:.3g} N/mm and at most "
                             f"{kmax:.3g} N/mm{why}")
        ground = _where(study, domain, hold_spec, f"{lab}.{'lands' if kind == 'drop' else 'hold'}")
        load = _where(study, domain, load_spec, f"{lab}.{ {'squeeze': 'press', 'drop': 'mass'}.get(kind, 'mount')}")
        load &= ~ground
        if not load.any():
            raise ValueError(f"{lab}: every loaded node is also held")
        out.append(FeelTarget(kind, name, ground, load, dirs, kmin, kmax, params,
                              key=json.dumps(hold_spec, sort_keys=True)))
    return out


def _unit(nodes, axis, sign, n_dof):
    f = np.zeros(n_dof)
    sel = np.flatnonzero(nodes)
    f[3 * sel + axis] = sign / len(sel)
    return f


class _HeldSolves(_Solves):
    """_Solves on one target's holds. Every loaded node has to be in the
    reduced system: one cut off from the hold would read as not moving at
    all (infinitely stiff), so then the whole mesh is solved, void cells at
    E_MIN, and the compliance comes out huge, as it is."""

    def __init__(self, domain, nu, need):
        super().__init__(domain, nu)
        self.need = need

    def _alive(self, xp):
        alive = super()._alive(xp)
        cov = np.zeros(self.domain.mesh.n_nodes, bool)
        cov[self.domain.mesh.enode[alive].ravel()] = True
        if not cov[self.need].all():
            return np.ones(len(xp), bool)
        return alive


@dataclass
class _Group:
    domain: Domain
    solves: _HeldSolves
    probes: list           # [(target index, direction index, f)]


class FeelProblem:
    """The feel targets on one grid, as constraints for
    switch.optimize_switch (see `providers` there)."""

    ROBUST = 0.2

    def __init__(self, study, domain: Domain, nu, modulus: float):
        self.targets = feel_from_study(study, domain)
        self.robust = bool((study.raw.get("feel") or {}).get("robust", True))
        self.domain, self.nu, self.modulus = domain, nu, modulus
        mesh = domain.mesh
        groups: dict[str, _Group] = {}
        for ti, t in enumerate(self.targets):
            if t.key not in groups:
                fixed = np.repeat(t.ground, 3)
                dom = dataclasses.replace(domain, fixed=fixed, support_nodes=t.ground.copy(), springs=None)
                need = np.zeros(mesh.n_nodes, bool)
                groups[t.key] = _Group(dom, None, [])
                groups[t.key].need = need
            g = groups[t.key]
            g.need |= t.load
            for di, (ax, sg) in enumerate(t.directions):
                g.probes.append((ti, di, _unit(t.load, ax, sg, mesh.n_dof)))
        for g in groups.values():
            g.solves = _HeldSolves(g.domain, nu, g.need)
        self.groups = list(groups.values())
        self.m = sum((t.kmin is not None) + (t.kmax is not None) for t in self.targets for _ in t.directions)

    def compliances(self, xp, grads: bool = True):
        """{(target, direction): (c mm/N, dc/dxp or None)}."""
        from .simp import E_MIN, PENALTY
        mesh = self.domain.mesh
        ke = hex8_ke(self.nu) * mesh.pitch
        dEdx = PENALTY * xp ** (PENALTY - 1) * (1 - E_MIN)
        out = {}
        for g in self.groups:
            g.solves.factor(xp)
            sol = g.solves.solve([f for _, _, f in g.probes], "feel")
            for (ti, di, f), u in zip(g.probes, sol):
                u = np.asarray(u) / self.modulus
                c = float(f @ u)
                dc = -self.modulus * dEdx * _pair(mesh.edof, ke, u, u) if grads else None
                out[(ti, di)] = (c, dc)
        return out

    def evaluate(self, xp, ctx=None):
        """Constraints for the optimizer. With ctx (the filtered design xt
        and the sharpness beta) and `robust`, each floor (k >= kmin) is read
        on the eroded design (projected at 0.5 + ROBUST) and each ceiling
        (k <= kmax) on the dilated one (0.5 - ROBUST), so the blueprint
        between them meets both once it is printed black and white. Read on
        the blueprint alone, the optimizer met soft windows with grey
        cells (the microphone body's capsule suspension: design 8.7 N/mm,
        thresholded 0.5): grey is cheap softness that printing does not
        have, and it thresholds away to nothing."""
        if ctx is None or not self.robust:
            res = []
            for (ti, di), (c, dc) in self.compliances(xp).items():
                res += self._cons(ti, di, c, dc, True, True)
            return res
        from .simp import _dproject, _project
        xt, beta = ctx["xt"], ctx["beta"]
        res = []
        for side, eta in (("min", 0.5 + self.ROBUST), ("max", 0.5 - self.ROBUST)):
            if not any(getattr(t, "k" + side) is not None for t in self.targets):
                continue
            x = _project(xt, beta, eta)
            d = _dproject(xt, beta, eta)
            x[ctx["preserve"]] = 1.0
            d[ctx["preserve"]] = 0.0
            held = ctx["floor"] > x
            x[held] = ctx["floor"][held]
            d[held] = 0.0
            for (ti, di), (c, dc) in self.compliances(x).items():
                for val, g, name in self._cons(ti, di, c, dc, side == "min", side == "max"):
                    res.append((val, g * d, name, "xt"))
        return res

    def _cons(self, ti, di, c, dc, lo, hi):
        t = self.targets[ti]
        c = max(c, 1e-30)
        out = []
        if lo and t.kmin is not None:       # k >= kmin: c <= 1 / kmin
            out.append((np.log(c * t.kmin), dc / c, f"{t.name}:{di}:min"))
        if hi and t.kmax is not None:       # k <= kmax: c >= 1 / kmax
            out.append((-np.log(c * t.kmax), -dc / c, f"{t.name}:{di}:max"))
        return out


def _ok(k, t: FeelTarget, tol: float = 0.02) -> bool:
    return bool((t.kmin is None or k >= t.kmin * (1 - tol)) and (t.kmax is None or k <= t.kmax * (1 + tol)))


def feel_readings(prob: FeelProblem, xp: np.ndarray) -> list[dict]:
    """Per target and direction: stiffness, its window, and what it means
    (travel at the squeeze force; peak g and travel in the drop; natural
    frequency and transmissibility at the band)."""
    from .modal import transmissibility
    comp = prob.compliances(np.asarray(xp, float), grads=False)
    rows = []
    for ti, t in enumerate(prob.targets):
        dirs = []
        for di, (ax, sg) in enumerate(t.directions):
            c = comp[(ti, di)][0]
            k = 1.0 / max(c, 1e-30)
            r = {"direction": ("-" if sg < 0 else "+") + "xyz"[ax], "stiffness": k, "ok": _ok(k, t)}
            if t.kind == "squeeze" and "force" in t.params:
                r["travel"] = float(t.params["force"]) * c
            elif t.kind == "drop":
                r.update(drop_response(k, float(t.params["kg"]), float(t.params["height"])))
            elif t.kind == "isolate":
                fn = float(np.sqrt(k * 1000.0 / t.params["kg"]) / (2 * np.pi))
                r["frequency"] = fn
                if "band" in t.params:
                    r["transmissibility"] = float(transmissibility(t.params["band"], fn, t.params["lossFactor"]))
                if "upTo" in t.params:
                    r["amplification"] = float(transmissibility(t.params["upTo"], fn, t.params["lossFactor"]))
                if "maxTravel" in t.params:
                    r["travel"] = t.params["kg"] * G0 * float(t.params.get("shockG", 1.0)) / k
            dirs.append(r)
        rows.append({"name": t.name, "kind": t.kind, "minStiffness": t.kmin, "maxStiffness": t.kmax,
                     "params": t.params, "directions": dirs, "ok": all(r["ok"] for r in dirs)})
    return rows


def check_isolation_modal(study, prob: FeelProblem, ti: int, xp: np.ndarray, n_modes: int = 12,
                          keep_min: float = 0.2) -> dict:
    """The isolate target `ti` read with the modes of the part: the capsule
    a point mass on its mount, the hand's nodes the shaken base, cells at
    or over keep_min kept at modulus E x^3 and density rho x (the same
    reading the static target uses, mass included). Transmissibility to
    the mount at the band per direction, by modal superposition."""
    from .modal import base_transmissibility, modal_analysis
    from .nonlinear import solid_submesh
    t = prob.targets[ti]
    mesh = prob.domain.mesh
    xp = np.asarray(xp, float)
    # (not every preserved cell: one the mesh leaves empty, where a keep-out
    # cuts the preserve, came in at a modulus of zero and the eigensolve
    # returned nothing but zero-frequency modes)
    keep = xp >= keep_min
    # Only cells joined to the hand through faces: a piece hanging off a
    # single node or edge of another (common in a part read back from a
    # mesh) is a hinge with no stiffness.
    from scipy import ndimage
    ijk = mesh.elem_ijk
    g = np.zeros(prob.domain.dims, bool)
    g[tuple(ijk[keep].T)] = True
    lbl, _ = ndimage.label(g, structure=ndimage.generate_binary_structure(3, 1))
    el_lbl = lbl[tuple(ijk.T)]
    held = np.unique(el_lbl[keep & t.ground[mesh.enode].any(axis=1)])
    keep &= np.isin(el_lbl, held[held > 0])
    fixed = np.repeat(t.ground, 3)
    mvec = np.repeat(t.load.astype(float), 3)
    sub, el, fx, (mv,) = solid_submesh(mesh, keep, fixed, [mvec])
    nodes = np.flatnonzero(mv.reshape(-1, 3)[:, 0] > 0)
    if not len(nodes):
        return {"name": t.name, "error": "the mount is not connected to the hand through the part"}
    E = float(study.material.E) * xp[el] ** 3
    rho = float(study.material.density) * xp[el]
    res = modal_analysis(sub, E, rho, prob.nu, fx, n_modes=n_modes, point_masses=[(nodes, t.params["kg"])])
    eta = t.params["lossFactor"]
    rows = []
    for ax, _ in t.directions:
        dofs = 3 * nodes + ax
        # the mode that moves the part most along this axis
        phi = np.abs(res.modes[dofs].mean(axis=0)) * np.sqrt(t.params["kg"] * 1e-3)
        j = int(np.argmax(phi))
        row = {"direction": "xyz"[ax], "partMode": float(res.frequencies[j]), "ok": True}
        if "band" in t.params:
            tr = float(base_transmissibility(res, dofs, ax, [t.params["band"]], eta, fixed=fx)[0])
            row["transmissibility"] = tr
            row["ok"] &= bool(tr <= t.params["maxTransmissibility"] * 1.05)
        if "upTo" in t.params:
            # the worst amplification over the handling band (up to upTo)
            fs = np.linspace(0.05, 1.0, 40) * t.params["upTo"]
            amp = float(base_transmissibility(res, dofs, ax, fs, eta, fixed=fx).max())
            row["amplification"] = amp
            row["ok"] &= bool(amp <= t.params["maxAmplification"] * 1.05)
        rows.append(row)
    return {"name": t.name, "frequencies": [round(float(f), 1) for f in res.frequencies], "directions": rows,
            "ok": all(r["ok"] for r in rows), "cells": int(sub.n_elem)}


def run_feel(study, out_dir: str, pitches=None, iterations=None, on_progress=None, modal: bool = True,
             nonlinear: bool | None = None) -> dict:
    """Optimize, mesh and check a study's `feel` block (with its `switch`
    block too, if it has one). Writes <name>.stl and returns the report:
    per target its window and its reading on the design, on the
    black-and-white design and on the delivered mesh, and for an isolate
    target the modal check on the delivered mesh."""
    from .switch import run_switch

    fb = study.raw.get("feel") or {}

    def make(st, domain, nu, modulus):
        return FeelProblem(st, domain, nu, modulus)

    def readback(domain, design, solid, delivered, floor, mesh):
        from .domain import _fraction, build_domain
        from .shapes import MeshShape
        from .switch import _thin_skin
        nu = study.elastic(design=True)
        prob = FeelProblem(study, domain, nu, float(study.material.E))
        rows = feel_readings(prob, design)
        for row, rs in zip(rows, feel_readings(prob, solid)):
            row["thresholded"] = rs["directions"]
        out = {}
        x_modal, prob_modal = solid, prob
        if delivered is not None:
            # The delivered mesh on a grid of half the pitch: a member one
            # design cell thick is a few cells partly filled on the design
            # grid, and those read at x^3 put the microphone body's capsule
            # suspension at a fifth of what the same mesh reads at 0.5 mm.
            vp = float(fb.get("verifyPitch") or 0.5 * domain.pitch)
            if on_progress is not None:
                on_progress({"stage": "verify", "check": f"feel targets on the delivered mesh at {vp:g} mm"})
            fine, ffloor = _thin_skin(study, build_domain(study, vp))
            drawn = _fraction([MeshShape(mesh)], fine.mesh.elem_centers(), fine.pitch)
            x_modal = np.maximum(np.maximum(drawn, ffloor), 1e-9)
            prob_modal = FeelProblem(study, fine, nu, float(study.material.E))
            for row, rd in zip(rows, feel_readings(prob_modal, x_modal)):
                row["delivered"] = rd["directions"]
                row["deliveredOk"] = rd["ok"]
            out["verifyPitch"] = fine.pitch
        if modal:
            for ti, t in enumerate(prob_modal.targets):
                if t.kind == "isolate":
                    if on_progress is not None:
                        on_progress({"stage": "verify", "check": f"modes: {t.name}"})
                    rows[ti]["modal"] = check_isolation_modal(study, prob_modal, ti, x_modal)
        return {**out, "feel": rows, "feelOk": all(r["ok"] for r in rows),
                "feelDeliveredOk": all(r.get("deliveredOk", False) for r in rows)}

    if not study.raw.get("switch"):
        pitches = pitches or fb.get("pitches")
        iterations = iterations or fb.get("iterations")
    # Feel windows are often soft, and soft inside a small exterior means
    # members about one cell thick: mesh them without the thinning that
    # smoothing gives a member that thin (reconstruct's axial debias; the
    # capsule suspension read 46 N/mm against a 50 N/mm design with it,
    # 2 without).
    from .reconstruct import ReconstructOptions
    mesh_opts = ReconstructOptions(axial_debias=bool(fb.get("keepThinMembers", True)))
    rep = run_switch(study, out_dir, pitches=pitches, iterations=iterations, on_progress=on_progress,
                     nonlinear=nonlinear, providers=[make], readback=readback, mesh_options=mesh_opts,
                     optimize_options={"move_sharp": float(fb.get("moveSharp", 0.03))})
    rep["passes"] = bool(rep["feelOk"] and (rep["passes"] if rep.get("presses") else True))
    return rep
