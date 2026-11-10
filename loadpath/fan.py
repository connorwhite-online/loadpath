"""
Axial fan rotors: blade design, performance and a printable rotor.

A fan pushing air through a heat exchanger, a filter or a duct works against
the system's resistance, and what it delivers is where its pressure-flow
curve crosses the system's. This module designs the blades for that crossing
and predicts the whole curve, by the blade-element method fan and compressor
design has used since the 1950s:

- the rotor is cut into stream tubes at radii from hub to tip; in each the
  blade row is a cascade that turns the relative flow, and the Euler equation
  gives the work, rho * U * c_u2 (no swirl coming in);
- simple radial equilibrium (dp/dr = rho c_u^2 / r) sets how the axial
  velocity leaving the rotor varies with radius, with continuity holding the
  flow;
- blade angles from the flow angles by Carter's deviation rule (circular-arc
  camber), solidity from Lieblein's diffusion factor (D = 0.45 at design,
  stall flagged past 0.6), profile loss from Lieblein's equivalent diffusion
  ratio (Aungier's fit of the momentum thickness), and annulus and secondary
  loss from Howell's drag coefficients;
- low Reynolds number (a printed 120 mm fan runs at 3-8 x 10^4 on the chord)
  raises the profile loss by (2e5/Re)^0.4 and the deviation by up to 1.5 deg,
  where the cascade data stop.

The tip is one of three: a `band` (a ring on the blade tips that spins with
them: no tip gap, a little skin friction on its outside), a `shroud` (a
stationary ring with a tip gap: pressure falls by 2 gap / span) or `open`
(no ring: Prandtl's tip-loss factor on the work, as for a propeller).

What it is and is not: a correlation model, as a fan designer's first pass is.
It is not CFD, its loss and deviation correlations come from cascade tests at
higher Reynolds number than a small fan sees, and it does not predict the
stalled part of the curve (it flags it). Expect the pressure at a given flow
within 15-25 %. Test a printed rotor before relying on the last few percent.

Units: lengths in mm, flow in m^3/s, pressure in Pa, power in W, speed in rpm.

    from loadpath import fan
    air = fan.air(35.0)
    rotor = fan.design(0.04, 150.0, 6400, r_tip=57.3, r_hub=23.5, blades=9, air=air)
    fan.performance(rotor, 0.04, air=air)["dp_ts"]
    best, rep = fan.optimize(lambda q: 9e4 * q * q, r_tip=57.3, r_hub=23.5, power=6.0, rpm=(4000, 6400), air=air)
    fan.rotor_mesh(best).export("rotor.stl")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

R_AIR = 287.05


def air(t_c: float = 20.0, p: float = 101325.0) -> dict:
    """Dry air at t_c (deg C): density (kg/m^3) and kinematic viscosity (m^2/s, Sutherland)."""
    t = t_c + 273.15
    rho = p / (R_AIR * t)
    mu = 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4)
    return {"rho": rho, "nu": mu / rho, "t": t_c}


@dataclass
class Rotor:
    """Blade geometry at stations from hub to tip. Angles in radians from the axis; lengths in mm."""
    r: np.ndarray
    chord: np.ndarray
    kappa1: np.ndarray            # blade angle at the leading edge
    kappa2: np.ndarray            # blade angle at the trailing edge
    thickness: np.ndarray         # maximum thickness
    te: float                     # trailing-edge thickness
    blades: int
    r_hub: float
    r_tip: float
    rpm: float
    tip: str = "band"             # band | shroud | open
    gap: float = 0.0              # tip gap for a shroud (mm)
    band: tuple = (1.6, 18.0)     # band thickness and height (mm)
    exit: str = "sre"             # what's downstream: sre (a free duct) | uniform (a resistance close behind)
    design_point: dict = field(default_factory=dict)

    @property
    def camber(self):
        return self.kappa1 - self.kappa2

    @property
    def stagger(self):
        return 0.5 * (self.kappa1 + self.kappa2)

    @property
    def solidity(self):
        return self.blades * self.chord / (2 * np.pi * self.r)

    @property
    def axial_depth(self):
        """Axial length of each section (mm): chord times cos(stagger), plus the thickness's share."""
        return self.chord * np.cos(self.stagger) + self.thickness * np.sin(self.stagger)

    def to_dict(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        d["kappa1_deg"] = np.degrees(self.kappa1).tolist()
        d["kappa2_deg"] = np.degrees(self.kappa2).tolist()
        d["stagger_deg"] = np.degrees(self.stagger).tolist()
        d["solidity"] = self.solidity.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        keys = ("r", "chord", "kappa1", "kappa2", "thickness")
        kw = {k: np.asarray(d[k], float) for k in keys}
        rest = {k: d[k] for k in ("te", "blades", "r_hub", "r_tip", "rpm", "tip", "gap", "band", "exit", "design_point") if k in d}
        return cls(**kw, **rest)


# ----------------------------------------------------------------- correlations

def _re_loss(re):
    return np.where(re < 2e5, (2e5 / np.maximum(re, 1e3)) ** 0.4, 1.0)


def _re_deviation(re):
    return np.radians(1.5) * np.clip(np.log10(2e5 / np.maximum(re, 1e3)), 0.0, 1.0)


def _carter_m(kappa2):
    # circular-arc camber line, maximum camber at mid-chord (Carter 1950): m = 0.23 (2a/c)^2 + kappa2 / 500 (deg)
    return 0.23 + np.degrees(np.abs(kappa2)) / 500.0


def _losses(b1, b2, sigma, inc, re, pitch, span):
    """Loss coefficients on the inlet relative dynamic pressure: profile (Lieblein's equivalent diffusion, Aungier's
    momentum-thickness fit, low-Re factor) and endwall (Howell: annulus drag 0.02 s/H, secondary 0.018 C_L^2).
    Returns (profile, endwall, Deq, C_L), each per station."""
    ideg = np.degrees(inc)
    a = np.where(ideg > 0, 0.0117, 0.0070)
    tb1, tb2 = np.tan(b1), np.tan(b2)
    deq = (np.cos(b2) / np.cos(b1)) * (1.12 + a * np.abs(ideg) ** 1.43 + 0.61 * np.cos(b1) ** 2 / sigma * (tb1 - tb2))
    theta = 0.004 * (1 + 3.1 * (deq - 1) ** 2 + 0.4 * (deq - 1) ** 8) * _re_loss(re)
    w_p = 2 * theta * (sigma / np.cos(b2)) * (np.cos(b1) / np.cos(b2)) ** 2
    bm = np.arctan(0.5 * (tb1 + tb2))
    cl = 2 * np.cos(bm) / sigma * (tb1 - tb2)
    cd = 0.020 * pitch / span + 0.018 * cl ** 2
    w_e = sigma * cd * np.cos(b1) ** 2 / np.cos(bm) ** 3
    return np.minimum(w_p, 1.5), np.minimum(w_e, 1.5), deq, cl


def _row_loss(b1, b2, sig, inc, re, pitch, span, w1, rho, cx2, r):
    """Total-pressure loss (Pa) per station: its own profile loss, plus the endwall loss as the passage's flow-weighted
    average (endwall loss lives in the hub and tip boundary layers and the secondary flow that carries it across the
    span: it lowers the average, it does not shape the radial balance)."""
    wp, we, deq, cl = _losses(b1, b2, sig, inc, re, pitch, span)
    qd = 0.5 * rho * w1 * w1
    dq = cx2 * 2 * np.pi * r
    ew = _trapz(we * qd * dq, r) / _trapz(dq, r)
    return wp * qd + ew, wp, we, deq, cl


def _prandtl(r, r_tip, blades, b_m):
    phi = np.maximum(0.5 * np.pi - b_m, 1e-3)              # flow angle to the rotor plane
    f = blades / 2.0 * (r_tip - r) / (r * np.sin(phi))
    return (2 / np.pi) * np.arccos(np.clip(np.exp(-np.maximum(f, 0.0)), 0.0, 1.0))


def _tip_factor(rotor_tip, gap, r, blades, b_m):
    span = r[-1] - r[0]
    if rotor_tip == "open":
        return _prandtl(r, r[-1], blades, b_m)
    if rotor_tip == "shroud":
        return np.full_like(r, 1 - 2 * gap * 1e-3 / span)
    return np.ones_like(r)


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def _sre(r, p02, cu2, rho, q):
    """Axial velocity leaving the rotor by simple radial equilibrium, holding the flow q. r in m. Returns (cx, reversed)."""
    g = np.gradient(p02, r) / rho - (cu2 / r) * np.gradient(r * cu2, r)
    e = _cumtrapz(g, r)                                    # 0.5 cx^2 - 0.5 cx_hub^2

    def flow(c0):
        return _trapz(np.sqrt(np.maximum(c0 * c0 + 2 * e, 0.0)) * 2 * np.pi * r, r)

    lo = math.sqrt(max(0.0, -2 * e.min()))
    if flow(lo) >= q:                                      # even the least velocity at one end carries too much
        cx = np.sqrt(np.maximum(lo * lo + 2 * e, 0.0))
        return cx * q / max(flow(lo), 1e-12), True
    hi = lo + 2 * q / (np.pi * (r[-1] ** 2 - r[0] ** 2)) + 1.0
    while flow(hi) < q:
        hi *= 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if flow(mid) < q else (lo, mid)
    return np.sqrt(np.maximum(hi * hi + 2 * e, 0.0)), False


def _band_power(rotor, om, air):
    # skin friction on the outside of a spinning tip band, from a turbulent flat plate on its height
    if rotor.tip != "band":
        return 0.0
    t, h = rotor.band
    rb, hb = (rotor.r_tip + t) * 1e-3, h * 1e-3
    u = om * rb
    cf = 0.074 * max(u * hb / air["nu"], 5e3) ** -0.2
    return cf * 0.5 * air["rho"] * u * u * (2 * np.pi * rb * hb) * rb * om


# ----------------------------------------------------------------- performance

def performance(rotor: Rotor, q: float, rpm: float | None = None, air: dict | None = None, iters: int = 60,
                exit: str | None = None) -> dict:
    """The rotor at flow q (m^3/s) and speed rpm: total and total-to-static pressure rise (Pa, flow-averaged; the exit's
    kinetic energy counts as lost for dp_ts, as on a datasheet's static curve), shaft power (W), efficiencies, and the
    per-station flow.

    `exit` (default the rotor's): "sre", a free duct downstream, where the axial velocity settles by radial
    equilibrium; "uniform", a resistance close behind (a heat exchanger a few mm off, a filter), whose channels hold
    the axial velocity even across the span and carry the radial pressure difference themselves.
    `stalled` when a station in the outer 80 % of the span passes D = 0.6 or Deq = 2.2 (rotating stall starts at the
    tip); `hub_stall` when the inner 20 % does, or the flow reverses there."""
    air = air or globals()["air"]()
    rho, nu = air["rho"], air["nu"]
    rpm = rpm or rotor.rpm
    exit = exit or rotor.exit
    om = rpm * 2 * np.pi / 60
    r = rotor.r * 1e-3
    c = rotor.chord * 1e-3
    u = om * r
    area = np.pi * (r[-1] ** 2 - r[0] ** 2)
    cx1 = q / area
    b1 = np.arctan(u / cx1)
    inc = b1 - rotor.kappa1
    w1 = np.hypot(cx1, u)
    re = w1 * c / nu
    sig = rotor.solidity
    dev0 = _carter_m(rotor.kappa2) * np.abs(rotor.camber) / np.sqrt(sig) + _re_deviation(re)
    b2 = rotor.kappa2 + dev0 + 0.10 * inc                  # the relative flow leaves at the blade's angle plus deviation
    pitch, span = 2 * np.pi * r / rotor.blades, r[-1] - r[0]

    def row(cx2):
        cu2 = u - cx2 * np.tan(b2)
        loss, wp, we, deq, cl = _row_loss(b1, b2, sig, inc, re, pitch, span, w1, rho, cx2, r)
        work = rho * u * cu2 * _tip_factor(rotor.tip, rotor.gap, r, rotor.blades, 0.5 * (b1 + b2))
        return cu2, work - loss, wp, we, deq, cl

    cx2 = np.full_like(r, cx1)
    rev = False
    if exit == "sre":
        for _ in range(iters):
            cu2, p02, *_ = row(cx2)
            new, rev = _sre(r, p02, cu2, rho, q)
            cx2 = 0.5 * cx2 + 0.5 * np.maximum(new, 0.05 * cx1)
    cu2, p02, wp, we, deq, cl = row(cx2)
    dq = cx2 * 2 * np.pi * r
    qx = _trapz(dq, r)
    dp_t = _trapz(p02 * dq, r) / qx
    ke = _trapz(0.5 * rho * (cx2 ** 2 + cu2 ** 2) * dq, r) / qx
    torque = _trapz(rho * cx2 * cu2 * r * 2 * np.pi * r, r)
    power = om * torque + _band_power(rotor, om, air)
    w2 = np.hypot(cx2, u - cu2)
    d = 1 - w2 / w1 + cu2 / (2 * sig * w1)
    outer = rotor.r >= rotor.r_hub + 0.2 * (rotor.r_tip - rotor.r_hub)
    stalled = bool(d[outer].max() > 0.6 or deq[outer].max() > 2.2)
    hub_stall = bool(rev or d[~outer].max() > 0.6)
    dp_ts = dp_t - ke
    return {"q": q, "rpm": rpm, "exit": exit, "dp_t": dp_t, "dp_ts": dp_ts, "power": power, "torque": torque,
            "eff_t": q * dp_t / power if power > 0 else 0.0, "eff_ts": q * dp_ts / power if power > 0 else 0.0,
            "thrust": dp_ts * area, "stalled": stalled, "hub_stall": hub_stall, "reversed": rev, "D_max": float(d.max()),
            "stations": {"r": rotor.r.tolist(), "cx1": cx1, "cx2": cx2.tolist(), "cu2": cu2.tolist(),
                         "beta1_deg": np.degrees(b1).tolist(), "beta2_deg": np.degrees(b2).tolist(),
                         "incidence_deg": np.degrees(inc).tolist(), "D": d.tolist(), "Deq": deq.tolist(),
                         "CL": cl.tolist(), "loss_profile": wp.tolist(), "loss_endwall": we.tolist(),
                         "Re": re.tolist(), "dp_t": p02.tolist()}}


def curve(rotor: Rotor, rpm: float | None = None, air: dict | None = None, qs=None) -> list:
    """Performance over a range of flows (default 5 % to 160 % of the design flow)."""
    q0 = rotor.design_point.get("q", 0.04)
    qs = np.linspace(0.05, 1.6, 32) * q0 if qs is None else qs
    return [performance(rotor, float(q), rpm, air) for q in qs]


def stall_flow(rotor: Rotor, rpm=None, air=None, q_hi=None) -> float:
    """The largest flow at which the model flags stall, searching down from q_hi (m^3/s)."""
    q_hi = q_hi or 1.5 * rotor.design_point.get("q", 0.04)
    for q in np.linspace(q_hi, 0.03 * q_hi, 40):
        if performance(rotor, float(q), rpm, air, iters=30)["stalled"]:
            return float(q)
    return 0.0


def operating_point(rotor: Rotor, system, rpm=None, air=None, q_max=None) -> dict | None:
    """Where the rotor's dp_ts(q) meets system(q) (the static pressure the system needs at fan flow q, Pa)."""
    qd = rotor.design_point.get("q", 0.04)
    q_hi = q_max or 3.0 * qd
    f = lambda q: performance(rotor, q, rpm, air, iters=30)["dp_ts"] - system(q)
    qs = np.geomspace(0.2 * qd, q_hi, 24)                  # the first crossing from above, scanning up in flow
    fs = [f(float(q)) for q in qs]
    k = next((i for i in range(len(qs) - 1) if fs[i] > 0 >= fs[i + 1]), None)
    if k is None:
        return None
    lo, hi = float(qs[k]), float(qs[k + 1])
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return performance(rotor, 0.5 * (lo + hi), rpm, air)


# ----------------------------------------------------------------- design

def design(q: float, dp: float, rpm: float, r_tip: float, r_hub: float, blades: int, *, work_exp: float = 0.0,
           diffusion: float = 0.45, air: dict | None = None, tip: str = "band", gap: float = 0.0,
           band: tuple = (1.6, 18.0), n: int = 15, incidence: float = 0.0, t_ratio: float = 0.07,
           t_min: float = 1.4, te: float = 0.8, sigma_range: tuple = (0.3, 2.2), static: bool = True,
           exit: str = "sre") -> Rotor:
    """Blades for dp (Pa, total-to-static if `static`, else total) at flow q (m^3/s) and rpm.

    The work varies with radius as (r / r_tip)^work_exp: 0 is a free vortex (uniform work, the only loading a free
    duct leaves evenly spread at a low flow coefficient), 1-2 put it outboard where the blade speed is and spare the
    hub, which suits `exit="uniform"` (a resistance close behind holds the flow even). Solidity from the diffusion
    factor; incidence (deg) onto the camber line at the leading edge. Thickness t_ratio of the chord but never under
    t_min, trailing edge te (mm): printable sections. Returns a Rotor; `design_point` records the aim, the diffusion
    it hit, and whether it converged."""
    air = air or globals()["air"]()
    rho, nu = air["rho"], air["nu"]
    om = rpm * 2 * np.pi / 60
    rr = np.linspace(r_hub, r_tip, n)
    r = rr * 1e-3
    u = om * r
    area = np.pi * (r[-1] ** 2 - r[0] ** 2)
    cx1 = q / area
    b1 = np.arctan(u / cx1)
    w1 = np.hypot(cx1, u)
    shape = (r / r[-1]) ** work_exp
    shape = shape / np.average(shape, weights=r)
    pitch, span = 2 * np.pi * r / blades, r[-1] - r[0]
    loss = np.zeros_like(r)
    cx2 = np.full_like(r, cx1)
    sig = np.full_like(r, 1.0)
    b2 = b1.copy()
    re = w1 * sig * pitch / nu
    w0 = dp + 0.5 * rho * cx1 ** 2
    rev, got, converged = False, 0.0, False
    tf = np.maximum(_tip_factor(tip, gap, r, blades, b1), 0.3)   # an open tip's last few mm can't be made to work
    for _ in range(80):
        ideal = w0 * shape                                 # Euler work, rho U cu2, per station
        cu2 = ideal / (rho * u * tf)
        if exit == "sre":
            cx2, rev = _sre(r, ideal - loss, cu2, rho, q)
            cx2 = np.maximum(cx2, 0.05 * cx1)
        w2 = np.hypot(cx2, u - cu2)
        den = diffusion - 1 + w2 / w1
        sig = np.clip(np.where(den > 0.02, cu2 / (2 * w1 * np.maximum(den, 0.02)), sigma_range[1]), *sigma_range)
        b2 = np.arctan((u - cu2) / cx2)
        re = w1 * sig * pitch / nu
        loss, *_ = _row_loss(b1, b2, sig, np.radians(incidence), re, pitch, span, w1, rho, cx2, r)
        dq = cx2 * 2 * np.pi * r
        dp_t = _trapz((ideal - loss) * dq, r) / q
        ke = _trapz(0.5 * rho * (cx2 ** 2 + cu2 ** 2) * dq, r) / q
        got = dp_t - ke if static else dp_t
        if abs(got / dp - 1) < 1e-4:
            converged = True
            break
        w0 *= 1.5 if got <= 0 else float(np.clip((dp / got) ** 0.8, 0.67, 1.5))
        if w0 > 50 * dp:                                   # the blades can't reach dp at this flow
            break
    w2 = np.hypot(cx2, u - cu2)
    d = 1 - w2 / w1 + cu2 / (2 * sig * w1)
    k1 = b1 - np.radians(incidence)
    k2 = b2.copy()
    for _ in range(40):                                    # Carter's rule: deviation grows with camber
        theta = np.clip(k1 - k2, -0.5, 1.4)
        k2 = 0.5 * k2 + 0.5 * (b2 - (_carter_m(k2) * np.abs(theta) / np.sqrt(sig) + _re_deviation(re)))
    chord = sig * pitch * 1e3
    rotor = Rotor(r=rr, chord=chord, kappa1=k1, kappa2=k2, thickness=np.maximum(t_ratio * chord, t_min), te=te,
                  blades=blades, r_hub=r_hub, r_tip=r_tip, rpm=rpm, tip=tip, gap=gap, band=band, exit=exit)
    rotor.design_point = {"q": q, "dp": dp, "static": static, "rpm": rpm, "work_exp": work_exp, "diffusion": diffusion,
                          "D_max": float(d.max()), "sigma_clipped": bool(np.any(sig >= sigma_range[1] - 1e-9)),
                          "converged": converged, "reversed": bool(rev), "achieved": float(got),
                          "flow_coefficient": float(cx1 / u[-1]),
                          "pressure_coefficient": float(dp / (0.5 * rho * u[-1] ** 2))}
    return rotor


def optimize(system, *, r_tip: float, r_hub: float, power: float | None = None, rpm=6000.0, blades=range(5, 14),
             axial: float | None = None, air: dict | None = None, tip: str = "band", gap: float = 0.0,
             band: tuple = (1.6, 18.0), stall_margin: float = 0.15, t_min: float = 1.4, te: float = 0.8,
             starts: int = 3, seed: int = 0, objective=None, exit: str = "sre") -> tuple:
    """Blades that move the most air through `system` (q -> static Pa the system needs at fan flow q).

    Within: shaft power at the operating point no more than `power` (W); rpm fixed or a (lo, hi) range; every
    section's axial depth no more than `axial` (mm); unstalled with the stall flow at least `stall_margin` below the
    operating flow; design diffusion factor 0.30-0.55. Searches blade count and, for each, the design flow, pressure,
    work exponent, diffusion and speed (Nelder-Mead from several starts). `objective(op) -> float` to maximize
    instead of the operating flow. Returns (rotor, report)."""
    from scipy.optimize import minimize

    air = air or globals()["air"]()
    rng = np.random.default_rng(seed)
    lo_rpm, hi_rpm = (rpm, rpm) if np.isscalar(rpm) else rpm
    obj = objective or (lambda op: op["q"])
    best = (None, None, -np.inf)
    tried = []
    area = np.pi * ((r_tip * 1e-3) ** 2 - (r_hub * 1e-3) ** 2)
    u_tip = hi_rpm * 2 * np.pi / 60 * r_tip * 1e-3
    q_ref = 0.10 * u_tip * area                             # a flow coefficient of 0.10 to start from
    dp_ref = max(system(q_ref), 10.0)

    def build(x, z):
        qd, dpd, wexp, diff, sp = x
        sp = lo_rpm + (hi_rpm - lo_rpm) * float(np.clip(sp, 0, 1))
        return design(q_ref * math.exp(qd), dp_ref * math.exp(dpd), sp, r_tip, r_hub, z, work_exp=float(np.clip(wexp, 0, 3)),
                      diffusion=float(np.clip(diff, 0.30, 0.55)), air=air, tip=tip, gap=gap, band=band, t_min=t_min, te=te,
                      exit=exit)

    def score(x, z):
        try:
            rot = build(x, z)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return 1e6, None, None
        pen = 0.0
        if axial is not None:
            pen += 50 * max(0.0, float(rot.axial_depth.max()) - axial) / axial
        if rot.design_point["sigma_clipped"]:
            pen += 0.2
        dpt = rot.design_point
        if not dpt["converged"] or dpt["reversed"]:
            return 1e2 + abs(math.log(max(dpt["achieved"], 1e-3) / dpt["dp"])) + pen, None, None
        op = operating_point(rot, system, air=air)
        if op is None:
            return 50.0 + pen, None, None
        if op["stalled"]:
            return 10.0 + 10 * op["D_max"] + pen, rot, op
        if op["hub_stall"]:
            pen += 0.3
        if power is not None:
            pen += 20 * max(0.0, op["power"] - power) / power
        qs = stall_flow(rot, air=air, q_hi=op["q"])
        pen += 10 * max(0.0, qs - (1 - stall_margin) * op["q"]) / op["q"]
        return -obj(op) / q_ref + pen, rot, op

    # start where the system takes about a third of the shaft power as air power: a fan's design point sits near there
    q_t = q_ref
    if power is not None:
        lo_q, hi_q = 1e-5, 1.0
        for _ in range(60):
            q_t = math.sqrt(lo_q * hi_q)
            lo_q, hi_q = (q_t, hi_q) if q_t * system(q_t) < 0.35 * power else (lo_q, q_t)
    x_t = np.array([math.log(q_t / q_ref), math.log(max(system(q_t), 1.0) * 1.15 / dp_ref), 1.4, 0.5, 0.6])
    best_x = None

    def polish(x0, z, iters):
        res = minimize(lambda x: score(x, z)[0], x0, method="Nelder-Mead",
                       options={"maxiter": iters, "xatol": 1e-3, "fatol": 1e-4, "initial_simplex": _simplex(x0)})
        return res.x, score(res.x, z)

    def _simplex(x0):
        steps = np.array([0.15, 0.2, 0.4, 0.05, 0.25])
        return np.vstack([x0] + [x0 + np.eye(5)[i] * steps[i] for i in range(5)])

    for z in blades:
        for k in range(starts):
            if k == 0:
                x0 = x_t.copy()
            elif k == 1 and best_x is not None:
                x0 = best_x.copy()                         # the best so far, at this blade count
            else:
                x0 = x_t + rng.normal(0, [0.15, 0.2, 0.4, 0.04, 0.25])
            if hi_rpm == lo_rpm:
                x0[4] = 1.0
            x, (f, rot, op) = polish(x0, z, 160)
            tried.append({"blades": z, "score": float(f), "q": None if op is None else op["q"],
                          "power": None if op is None else op["power"]})
            if rot is not None and op is not None and not op["stalled"] and -f > best[2]:
                best = (rot, op, -f)
                best_x, best_z = x, z
    if best_x is not None:                                 # one more polish of the winner
        x, (f, rot, op) = polish(best_x, best_z, 240)
        if rot is not None and op is not None and not op["stalled"] and -f > best[2]:
            best = (rot, op, -f)
    rot, op, _ = best
    if rot is None:
        raise RuntimeError("no unstalled design meets the system within the limits")
    report = {"operating": op, "stall_flow": stall_flow(rot, air=air, q_hi=op["q"]), "tried": tried,
              "limits": {"power": power, "rpm": [lo_rpm, hi_rpm], "axial": axial, "stall_margin": stall_margin},
              "axial_depth_max": float(rot.axial_depth.max())}
    return rot, report


# ----------------------------------------------------------------- strength

def strength(rotor: Rotor, rpm: float | None = None, op: dict | None = None, density: float = 1010.0,
             yield_mpa: float = 48.0, retention: float = 1.0, hub_floor: float = 0.0) -> dict:
    """Stresses at speed (MPa) for a printed rotor: the band's hoop stress (rho w^2 r^2), and at each blade's root the
    pull of its own spin plus its share of the band, and bending from its share of the torque and thrust (a cantilever
    from the hub: conservative for a banded rotor, whose tips are held). Safety factor on yield x retention."""
    rpm = rpm or rotor.rpm
    om = rpm * 2 * np.pi / 60
    secs = [_section(rotor, i) for i in range(len(rotor.r))]
    areas = np.array([_poly_area(s) for s in secs]) * 1e-6                     # m^2
    r = rotor.r * 1e-3
    pull = density * om ** 2 * _trapz(areas * r, r)                              # N, the blade's own spin
    out = {"rpm": rpm}
    if rotor.tip == "band":
        t, h = rotor.band
        rb = (rotor.r_tip + 0.5 * t) * 1e-3
        mb = density * 2 * np.pi * rb * t * h * 1e-6
        out["band_hoop"] = density * (om * rb) ** 2 / 1e6
        pull += mb * om ** 2 * rb / rotor.blades
    root = secs[0]
    a_root = areas[0]
    ix, iy, cmax = _poly_bending(root)
    span = (rotor.r_tip - rotor.r_hub) * 1e-3
    if op is not None:
        ft = op["torque"] / (rotor.blades * (rotor.r_hub + 0.7 * (rotor.r_tip - rotor.r_hub)) * 1e-3)
        fa = op["thrust"] / rotor.blades
    else:
        ft = fa = 0.0
    m = math.hypot(ft, fa) * 0.6 * span                                         # resultant at 60 % span from the root
    i_min = min(ix, iy) * 1e-12
    sig = pull / a_root / 1e6 + (m * cmax * 1e-3 / i_min) / 1e6
    allow = yield_mpa * retention
    out.update({"blade_pull_N": pull, "root_area_mm2": a_root * 1e6, "root_bending_Nm": m, "root_stress": sig,
                "allowable": allow, "safety_factor": allow / max(sig, out.get("band_hoop", 0.0), 1e-9),
                "rotor_mass_g": (density * _trapz(areas, r) * rotor.blades + (mb if rotor.tip == "band" else 0.0)) * 1e3})
    return out


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_bending(p):
    """Second moments about the centroid's principal axes and the largest distance to the fibre (mm units)."""
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = 0.5 * cr.sum()
    cx, cy = ((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)
    x, y, x1, y1 = x - cx, y - cy, x1 - cx, y1 - cy
    cr = x * y1 - x1 * y
    ixx = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12
    iyy = ((x * x + x * x1 + x1 * x1) * cr).sum() / 12
    ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24
    avg, dif = 0.5 * (ixx + iyy), math.hypot(0.5 * (ixx - iyy), ixy)
    i1, i2 = abs(avg + dif), abs(avg - dif)
    return i1, i2, float(np.hypot(x, y).max())


# ----------------------------------------------------------------- geometry

def _section(rotor: Rotor, i: int, n: int = 40, chord=None, kappa1=None, kappa2=None, thick=None):
    """The blade section at station i as a closed loop (mm) in (a, z): a along the direction of rotation on the
    cylinder at that radius, z along the axis toward the inlet. Leading edge at the top; circular-arc camber line
    with NACA four-digit thickness, opened to the trailing-edge thickness. Centroid at the origin."""
    c = rotor.chord[i] if chord is None else chord
    k1 = rotor.kappa1[i] if kappa1 is None else kappa1
    k2 = rotor.kappa2[i] if kappa2 is None else kappa2
    t = rotor.thickness[i] if thick is None else thick
    s = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))           # cosine spacing: fine at both edges
    th = k1 - k2
    length = c * (0.5 * th / math.sin(0.5 * th)) if abs(th) > 1e-6 else c
    kap = k1 - th * s                                         # tangent angle runs linearly along a circular arc
    # relative flow runs down (-z) and against the rotation (-a): the chord runs from the leading edge that way
    ta, tz = -np.sin(kap), -np.cos(kap)
    a = np.concatenate([[0.0], np.cumsum(0.5 * (ta[1:] + ta[:-1]) * np.diff(s) * length)])
    z = np.concatenate([[0.0], np.cumsum(0.5 * (tz[1:] + tz[:-1]) * np.diff(s) * length)])
    yt = 5 * t * (0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * s ** 2 + 0.2843 * s ** 3 - 0.1015 * s ** 4)
    yt = yt + s * (0.5 * rotor.te - yt[-1])                   # open the trailing edge to te
    na, nz = np.cos(kap), -np.sin(kap)                         # normal to the camber line (the pressure side's way)
    up = np.c_[a + yt * na, z + yt * nz]
    lo = np.c_[a - yt * na, z - yt * nz]
    loop = np.vstack([up, lo[::-1][:-1]])                     # leading edge shared, trailing edge open to te
    loop -= _centroid(loop)
    return loop


def _centroid(p):
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = 0.5 * cr.sum()
    return np.array([((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)])


def blade_mesh(rotor: Rotor, z_mid: float = 0.0, root_in: float = 1.5, tip_in: float = 0.8, flare: float = 0.6,
               flare_len: float = 4.0, spanwise: int = 40, chordwise: int = 40, rotation: str = "ccw"):
    """One blade as a closed mesh (mm), its sections stacked on their centroids along the +x axis, reaching root_in
    into the hub and tip_in into the band. Thickness swells by `flare` where it meets the hub (and the band) over
    `flare_len`, for a filleted root. `rotation`: the way it turns seen from the inlet (+z)."""
    import trimesh

    rs = np.linspace(rotor.r_hub - root_in, rotor.r_tip + (tip_in if rotor.tip == "band" else 0.0), spanwise)
    rc = np.clip(rs, rotor.r[0], rotor.r[-1])
    interp = lambda v: np.interp(rc, rotor.r, v)
    ch, k1, k2, tk = interp(rotor.chord), interp(rotor.kappa1), interp(rotor.kappa2), interp(rotor.thickness)
    swell = 1 + flare * np.exp(-np.maximum(rs - rotor.r_hub, 0) / flare_len)
    if rotor.tip == "band":
        swell += flare * np.exp(-np.maximum(rotor.r_tip - rs, 0) / flare_len)
    rings = []
    for j, r in enumerate(rs):
        loop = _section(rotor, 0, chordwise, chord=ch[j], kappa1=k1[j], kappa2=k2[j], thick=tk[j] * swell[j])
        ang = loop[:, 0] / r * (1 if rotation == "ccw" else -1)
        rings.append(np.c_[r * np.cos(ang), r * np.sin(ang), loop[:, 1] + z_mid])
    m = len(rings[0])
    v = np.vstack(rings)
    f = []
    for j in range(len(rings) - 1):
        for k in range(m):
            a, b = j * m + k, j * m + (k + 1) % m
            c, d = a + m, b + m
            f += [[a, b, d], [a, d, c]]
    # end caps as a strip across the section's thickness, station by station along the chord: a fan from one centre
    # would cut the chord of the arc a tip section wraps round the band, and leave a pocket under it
    up = list(range(chordwise))
    lo = [0] + [chordwise + (chordwise - 1 - k) for k in range(1, chordwise)]
    cap = []
    for k in range(chordwise - 1):
        if k == 0:
            cap.append([up[0], up[1], lo[1]])
        else:
            cap += [[up[k], up[k + 1], lo[k + 1]], [up[k], lo[k + 1], lo[k]]]
    cap = np.array(cap)
    last = (len(rings) - 1) * m
    f = np.vstack([np.array(f), cap[:, ::-1], cap + last])
    mesh = trimesh.Trimesh(v, f, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _revolve(profile, sections=180):
    """A solid of revolution about z from a closed (r, z) polygon (mm)."""
    import trimesh
    p = np.asarray(profile, float)
    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    m = len(p)
    v = np.vstack([np.c_[p[:, 0] * np.cos(a), p[:, 0] * np.sin(a), p[:, 1]] for a in ang])
    f = []
    for i in range(sections):
        j = (i + 1) % sections
        for k in range(m):
            a, b = i * m + k, i * m + (k + 1) % m
            c, d = j * m + k, j * m + (k + 1) % m
            f += [[a, c, d], [a, d, b]]
    mesh = trimesh.Trimesh(v, np.array(f), process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def band_profile(rotor: Rotor, z0: float, z1: float, lip: float = 3.0, flare: float = 1.2, cap: float = 1.2):
    """(r, z) outline of the tip band from z0 (outlet) to z1 (inlet). Over its top `lip` mm the inside flares out by
    `flare` into a short bellmouth for the air coming in, the outside steps out to keep the rounded top `cap` thick."""
    t = rotor.band[0]
    ri, ro = rotor.r_tip, rotor.r_tip + t
    r_top = max(ro, ri + flare + cap)
    zc = z1 - 0.5 * cap
    arc = [(0.5 * (ri + flare + r_top) + 0.5 * cap * math.cos(a), zc + 0.5 * cap * math.sin(a)) for a in np.linspace(0, np.pi, 9)]
    bell = [(ri + flare * (1 - math.cos(p)), z1 - lip + (zc - z1 + lip) * math.sin(p)) for p in np.linspace(0.5 * np.pi, 0, 9)]
    return [(ri, z0 + 0.5), (ri + 0.5, z0), (ro - 0.5, z0), (ro, z0 + 0.5), (ro, z1 - lip)] + arc + bell[1:-1]


def rotor_mesh(rotor: Rotor, z0: float = 0.0, z1: float | None = None, hub: list | None = None,
               rotation: str = "ccw", lip: float = 3.0):
    """The whole rotor (mm): blades, the band (if the tip is a band) from z0 to z1, and a hub, a closed (r, z)
    profile revolved (default: a plain ring r_hub - 2.5 to r_hub over the band's height). Blades centred at mid
    height. Unioned with manifold3d, watertight."""
    import manifold3d as m3d
    import trimesh

    h = rotor.band[1] if z1 is None else z1 - z0
    z1 = z0 + h
    zm = 0.5 * (z0 + z1)
    parts = []
    one = blade_mesh(rotor, z_mid=zm, rotation=rotation)
    for k in range(rotor.blades):
        a = 2 * np.pi * k / rotor.blades
        parts.append(one.copy().apply_transform(trimesh.transformations.rotation_matrix(a, (0, 0, 1))))
    hub = hub or [(rotor.r_hub - 2.5, z0), (rotor.r_hub, z0), (rotor.r_hub, z1), (rotor.r_hub - 2.5, z1)]
    parts.append(_revolve(hub))
    if rotor.tip == "band":
        parts.append(_revolve(band_profile(rotor, z0, z1, lip)))

    def man(t):
        return m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))

    u = m3d.Manifold.batch_boolean([man(p) for p in parts], m3d.OpType.Add)
    o = u.to_mesh()
    return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
