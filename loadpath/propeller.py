"""
Propellers and rotors: blade design, performance, the motor that turns them,
and a printable blade that holds its shape at speed.

A drone's rotor is not a fan in a duct working against a resistance (fan.py):
it is a few slender blades at low solidity throwing air down into a free wake,
and what a builder asks of it is thrust per watt at hover, enough thrust at
full throttle, a blade that does not fly apart or flex out of its design, and
a stiffness that keeps its own frequencies off the rotor's. This module answers
those from the physics:

- **Blade-element momentum** (Glauert), solved on the inflow angle per
  annulus by bracketing (Ning's residual, rewritten so the static case,
  V = 0, is as well posed as the advancing one), with Prandtl's tip and hub
  loss. A rotor in a duct (aero.duct_performance) carries only the rotor's
  share of the thrust, at the duct's disk velocity, with no tip loss.
- **Section polars** from the section actually drawn: NeuralFoil (a network
  trained on XFOIL's viscous panel method with e^N transition, pip
  `loadpath[aero]`) at each station's own Reynolds number, which on a small
  prop is 2-8 x 10^4, where laminar separation decides lift and drag. Without
  it, a thin-airfoil estimate (2 pi lift slope cut by the boundary layer,
  flat-plate friction), flagged in the report.
- **Rotational stall delay** (Snel 1994): on a spinning blade the Coriolis and
  centrifugal pumping of the boundary layer keep the inboard sections
  attached past their 2D stall, Cl_3D = Cl_2D + 3 (c/r)^2 (Cl_lin - Cl_2D).
- **The motor** by its three constants (Kv, winding resistance, no-load
  current) behind the battery's own resistance: rpm, current and electrical
  power at any throttle, so 'thrust per watt' is per watt drawn from the pack
  and 'full throttle' is where the motor's torque curve meets the prop's.
- **The blade as a rotating beam** (flap, lag, torsion): centrifugal
  stiffening, the thrust's flap bending, the centrifugal twisting moment that
  turns a spinning blade toward flat pitch (the 'propeller moment'), the
  aerodynamic pitching moment, so the pitch the air sees is the drawn pitch
  less its elastic twist, solved together with the air loads. The rotating
  frequencies (Southwell) are checked against 1x and 2x rotor speed and the
  blade-pass frequency, and the root's stress against the material's
  strength and fatigue.
- **Design**: chord, pitch, camber and thickness along the span, and the
  blade count, for the least electrical power at hover subject to the
  thrust asked at full throttle, strength, printable thickness, tip Mach
  number and frequency placement. A printable mesh (fan.rotor_mesh: the
  same section that was analysed) for each hand of rotation.

What it is and is not: blade-element momentum is the method propeller
designers use for a first and second pass; it treats each annulus apart,
assumes the wake is axisymmetric, and knows nothing of the frame below the
rotor (aero.py carries that). Validated here against published static tests
of 3D-printed propellers with 2, 3 and 4 blades (UIUC propeller database,
Deters, Ananda & Selig 2014; tests/test_propeller.py).

Units: lengths in mm, speed in rpm, thrust N, torque N m, power W.

    from loadpath import propeller as pr
    blade = pr.Blade.uniform(r_tip=44.45, r_hub=6.0, blades=3, chord=9.0, pitch_in=3.0)
    pr.performance(blade, rpm=20000)["thrust"]
    motor = pr.Motor(kv=2450, rm=0.12, i0=0.6)
    pr.hover(blade, motor, thrust=0.9, volts=15.2)
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

RHO = 1.20
NU = 1.5e-5
A_SOUND = 343.0


def air(t_c: float = 20.0, p: float = 101325.0) -> dict:
    """Dry air at t_c (deg C): density (kg/m^3), kinematic viscosity (m^2/s) and speed of sound (m/s)."""
    t = t_c + 273.15
    rho = p / (287.05 * t)
    mu = 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4)
    return {"rho": rho, "nu": mu / rho, "a": math.sqrt(1.4 * 287.05 * t), "t": t_c}


# ----------------------------------------------------------------- geometry

@dataclass
class Blade:
    """A blade at stations from hub to tip (mm, radians).

    pitch: the chord line's angle to the rotor plane. camber: the circular-arc
    camber line's total turning (leading edge to trailing edge), its height
    is chord x tan(camber / 4) / 2. t_ratio: maximum thickness over chord
    (NACA four-digit distribution). te: trailing-edge thickness (mm), the
    printer's floor. airfoil: optional unit-chord coordinates (Selig order,
    trailing edge, upper surface, leading edge, lower surface) to use in
    place of the circular-arc section, its thickness scaled to t_ratio
    station by station (for checking against a published propeller)."""
    r: np.ndarray
    chord: np.ndarray
    pitch: np.ndarray
    camber: np.ndarray
    t_ratio: np.ndarray
    blades: int
    r_hub: float
    r_tip: float
    te: float = 0.3
    airfoil: np.ndarray | None = None
    note: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in ("r", "chord", "pitch", "camber", "t_ratio"):
            setattr(self, k, np.asarray(getattr(self, k), float) * np.ones(len(np.atleast_1d(self.r))))

    @property
    def diameter(self):
        return 2 * self.r_tip

    @property
    def solidity(self):
        return self.blades * self.chord / (2 * np.pi * self.r)

    def thickness(self):
        return self.t_ratio * self.chord

    def pitch_length(self, at: float = 0.75):
        """Geometric pitch (mm) at r / R = at: the advance per turn of the chord line, 2 pi r tan(pitch)."""
        rr = at * self.r_tip
        return 2 * np.pi * rr * math.tan(float(np.interp(rr, self.r, self.pitch)))

    @classmethod
    def uniform(cls, r_tip, r_hub, blades, chord, pitch_in=None, pitch_deg=None, camber_deg=6.0, t_ratio=0.08,
                stations=24, te=0.3):
        """A constant-chord blade of constant geometric pitch (pitch_in inches, as a prop is labelled) or a constant
        blade angle (pitch_deg)."""
        r = _stations(r_hub, r_tip, stations)
        if pitch_in is not None:
            pitch = np.arctan(pitch_in * 25.4 / (2 * np.pi * r))
        else:
            pitch = np.radians(pitch_deg) * np.ones_like(r)
        return cls(r, chord, pitch, np.radians(camber_deg), t_ratio, blades, r_hub, r_tip, te)

    def to_dict(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()}
        d["pitch_deg"] = np.degrees(self.pitch).tolist()
        d["camber_deg"] = np.degrees(self.camber).tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        kw = {k: d[k] for k in ("r", "chord", "pitch", "camber", "t_ratio", "blades", "r_hub", "r_tip") }
        for k in ("te", "note"):
            if k in d:
                kw[k] = d[k]
        if d.get("airfoil") is not None:
            kw["airfoil"] = np.asarray(d["airfoil"], float)
        return cls(**kw)


def _stations(r_hub, r_tip, n):
    """Cosine-clustered toward the tip, where the load falls to zero fastest."""
    s = np.sin(np.linspace(0, 0.5 * np.pi, n))
    return r_hub + (r_tip - r_hub) * (0.02 + 0.975 * s)


def section_coords(camber: float, t_ratio: float, te_ratio: float = 0.0, n: int = 61, airfoil=None) -> np.ndarray:
    """A unit-chord section, Selig order (TE, upper, LE, lower, TE): a circular-arc camber line turning `camber`
    radians with NACA four-digit thickness t_ratio, its trailing edge opened to te_ratio (te / chord). With
    `airfoil` (Selig-order coordinates), that section's camber line with its thickness scaled to t_ratio."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    if airfoil is not None:
        a = np.asarray(airfoil, float)
        i = int(np.argmin(a[:, 0]))
        up, lo = a[: i + 1][::-1], a[i:]
        yu = np.interp(x, up[:, 0], up[:, 1])
        yl = np.interp(x, lo[:, 0], lo[:, 1])
        yc, yt = 0.5 * (yu + yl), 0.5 * (yu - yl)
        yt = yt * t_ratio / max(2 * yt.max(), 1e-9)
    else:
        h = 0.5 * math.tan(camber / 4)                       # arc height over chord
        if abs(h) > 1e-9:
            rad = (0.25 + h * h) / (2 * h)
            yc = np.sqrt(np.maximum(rad * rad - (x - 0.5) ** 2, 0)) - (rad - h)
        else:
            yc = np.zeros_like(x)
        yt = 5 * t_ratio * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    yt = yt + x * (0.5 * te_ratio - yt[-1])
    upper = np.c_[x, yc + yt][::-1]
    lower = np.c_[x, yc - yt][1:]
    return np.vstack([upper, lower])


# ----------------------------------------------------------------- polars

ALPHA = np.radians(np.linspace(-20, 30, 101))


class Polars:
    """Lift, drag and moment of each station's section against angle of attack, tabulated at a few Reynolds numbers
    per station (re: one per station, or (stations, k) for k levels) and read at each station's own Reynolds number
    by interpolation in log Re. Built once per blade; `source` says which model made them."""

    def __init__(self, blade: Blade, re: np.ndarray, n_crit: float = 9.0, model: str = "auto", stall_delay=3.0):
        self.alpha = ALPHA
        re = np.asarray(re, float)
        self.re = re[:, None] if re.ndim == 1 else re                 # (stations, levels)
        tabs = [_section_polars(blade, self.re[:, j], n_crit, model) for j in range(self.re.shape[1])]
        self.source = tabs[0][3]
        cl = np.stack([t[0] for t in tabs], 1)                         # (stations, levels, alpha)
        self.cl2d = cl
        if stall_delay:
            cl = np.stack([_snel(cl[:, j], self.alpha, blade.chord / blade.r, stall_delay) for j in range(cl.shape[1])], 1)
        self.cl = cl
        self.cd = np.stack([t[1] for t in tabs], 1)
        self.cm = np.stack([t[2] for t in tabs], 1)
        self._re_now = None

    def set_re(self, re):
        """Read the tables at these Reynolds numbers (one per station) from now on."""
        lr = np.log(self.re)
        x = np.log(np.asarray(re, float))
        n, k = self.re.shape
        if k == 1:
            self._w = (np.zeros(n, int), np.zeros(n))
        else:
            j = np.array([np.clip(np.searchsorted(lr[i], x[i]) - 1, 0, k - 2) for i in range(n)])
            i = np.arange(n)
            w = np.clip((x - lr[i, j]) / (lr[i, j + 1] - lr[i, j]), 0.0, 1.0)
            self._w = (j, w)

    def at(self, alpha):
        """(cl, cd, cm) per station at angles alpha: shape (stations,) or (..., stations)."""
        a = np.clip(np.asarray(alpha, float), self.alpha[0], self.alpha[-1])
        ja = np.clip(np.searchsorted(self.alpha, a) - 1, 0, len(self.alpha) - 2)
        wa = (a - self.alpha[ja]) / (self.alpha[ja + 1] - self.alpha[ja])
        n = a.shape[-1]
        i = np.broadcast_to(np.arange(n), a.shape)
        jr, wr = self._w if getattr(self, "_w", None) is not None else (np.zeros(n, int), np.zeros(n))
        jr2 = np.minimum(jr + 1, self.re.shape[1] - 1)
        jr, jr2, wr = (np.broadcast_to(v, a.shape) for v in (jr, jr2, wr))

        def f(t):
            lo = t[i, jr, ja] * (1 - wa) + t[i, jr, ja + 1] * wa
            hi = t[i, jr2, ja] * (1 - wa) + t[i, jr2, ja + 1] * wa
            return lo * (1 - wr) + hi * wr
        return f(self.cl), f(self.cd), f(self.cm)


def polars_for(blade: Blade, rpm_range=(5000, 40000), air=None, levels: int = 3, **kw) -> Polars:
    """Polars tabulated over the Reynolds numbers this blade sees between rpm_range[0] and rpm_range[1]."""
    air = air or {"rho": RHO, "nu": NU, "a": A_SOUND}
    rpms = np.geomspace(rpm_range[0], rpm_range[1], levels)
    re = np.stack([(rp * 2 * np.pi / 60) * blade.r * 1e-3 * blade.chord * 1e-3 / air["nu"] for rp in rpms], 1)
    # the relative speed is above r omega by the inflow; take 1.05 x
    return Polars(blade, 1.05 * re, kw.get("n_crit", mack_n_crit(TURBULENCE)), kw.get("model", "auto"),
                  kw.get("stall_delay", 3.0))


_NF_CACHE: dict = {}


def _have_neuralfoil():
    try:
        import neuralfoil  # noqa: F401
        return True
    except Exception:
        return False


def _kulfan(coords):
    import aerosandbox as asb
    return asb.Airfoil(coordinates=coords).to_kulfan_airfoil().kulfan_parameters


def _section_polars(blade, re, n_crit, model):
    n = len(blade.r)
    deg = np.degrees(ALPHA)
    use_nf = model in ("auto", "neuralfoil") and _have_neuralfoil()
    if model == "neuralfoil" and not use_nf:
        raise ImportError("model='neuralfoil' needs `pip install loadpath[aero]`")
    cl = np.zeros((n, len(ALPHA)))
    cd = np.zeros_like(cl)
    cm = np.zeros_like(cl)
    if use_nf:
        import neuralfoil as nf
        for i in range(n):
            te_ratio = min(blade.te / blade.chord[i], 0.5 * blade.t_ratio[i])
            key = (round(float(blade.camber[i]), 4), round(float(blade.t_ratio[i]), 4), round(te_ratio, 4),
                   None if blade.airfoil is None else hash(np.asarray(blade.airfoil).tobytes()))
            k = _NF_CACHE.get(key)
            if k is None:
                k = _kulfan(section_coords(blade.camber[i], blade.t_ratio[i], te_ratio, airfoil=blade.airfoil))
                _NF_CACHE[key] = k
            o = nf.get_aero_from_kulfan_parameters(k, deg, float(re[i]), n_crit=n_crit, model_size="xlarge")
            cl[i], cd[i], cm[i] = o["CL"], o["CD"], o["CM"]
        return cl, cd, cm, "neuralfoil"
    for i in range(n):
        cl[i], cd[i], cm[i] = _thin_polar(blade.camber[i], blade.t_ratio[i], float(re[i]))
    return cl, cd, cm, "thin-airfoil estimate"


def _thin_polar(camber, t_ratio, re):
    """Thin-airfoil lift (slope 2 pi cut to 0.85 of it by the boundary layer at low Re, zero-lift angle of a circular
    arc = -camber / 4), stall at the laminar-separation lift ~ 0.9 + 4 f/c falling to a flat plate's 2 sin a cos a,
    drag from both sides' friction (laminar 1.328 / sqrt(Re) below Re 5e5) with form factor (1 + 2 t + 60 t^4) plus
    the lift-dependent laminar-bubble drag. An estimate to run without NeuralFoil, not a polar to trust."""
    a = ALPHA
    a0 = -camber / 4
    lin = 0.85 * 2 * np.pi * (a - a0)
    clmax = 0.9 + 4 * 0.5 * math.tan(camber / 4)
    plate = 2 * np.sin(a) * np.cos(a)
    s = 1 / (1 + np.exp(-(np.abs(lin) - clmax) / 0.08))
    cl = (1 - s) * lin + s * plate
    cf = 1.328 / math.sqrt(max(re, 1e3))
    cd0 = 2 * cf * (1 + 2 * t_ratio + 60 * t_ratio ** 4)
    cd = cd0 + 0.02 * (cl - 0.5) ** 2 + s * 1.2 * np.sin(np.abs(a)) ** 2
    cm = -np.pi / 4 * 4 * 0.5 * math.tan(camber / 4) * np.ones_like(a)
    return cl, cd, cm


def _snel(cl, alpha, c_over_r, a_snel):
    """Snel's rotational stall delay on each station: the gap between the attached-flow line and the 2D lift is closed
    by a (c / r)^2 (a = 3, Snel, Houwink & Bosschers 1994), above the zero-lift angle only."""
    out = cl.copy()
    j0, j1 = np.searchsorted(alpha, np.radians(-2.0)), np.searchsorted(alpha, np.radians(4.0))
    for i in range(cl.shape[0]):
        slope = (cl[i, j1] - cl[i, j0]) / (alpha[j1] - alpha[j0])
        if slope <= 0:
            continue
        a0 = alpha[j0] - cl[i, j0] / slope
        lin = slope * (alpha - a0)
        f = min(a_snel * float(c_over_r[i]) ** 2, 1.0)
        out[i] = np.where(alpha > a0, cl[i] + f * np.clip(lin - cl[i], 0, None), cl[i])
    return out


# ----------------------------------------------------------------- blade-element momentum

def _loss(blade, r, phi, tip=True):
    s = np.maximum(np.sin(phi), 1e-6)
    f = np.ones_like(r)
    if tip:
        ft = blade.blades / 2 * (blade.r_tip - r) / (r * s)
        f = f * 2 / np.pi * np.arccos(np.clip(np.exp(-np.maximum(ft, 0)), 0, 1))
    fh = blade.blades / 2 * (r - blade.r_hub) / (r * s)
    f = f * 2 / np.pi * np.arccos(np.clip(np.exp(-np.maximum(fh, 0)), 0, 1))
    return np.maximum(f, 1e-4)


def performance(blade: Blade, rpm: float, speed: float = 0.0, air: dict | None = None, polars: Polars | None = None,
                twist: np.ndarray | None = None, duct: dict | None = None, n_crit: float = 9.0, model: str = "auto",
                stall_delay: float = 3.0) -> dict:
    """The rotor at `rpm` in an axial stream `speed` (m/s, 0 for hover): thrust (N), torque (N m), shaft power (W),
    CT = T / (rho n^2 D^4), CP = P / (rho n^3 D^5), figure of merit, and per station the inflow angle, angle of
    attack, lift and drag coefficients, Reynolds and Mach numbers and the thrust and torque per mm.

    `twist`: elastic twist per station (radians, + toward more pitch), from the blade's own deflection.
    `duct`: a rotor inside a duct, {"exitAreaRatio": sigma} (aero.duct_performance): the annulus momentum is the
    duct's (rotor pressure jump 0.5 rho (w / sigma)^2 at disk velocity w), with no tip loss; static only."""
    air = air or {"rho": RHO, "nu": NU, "a": A_SOUND}
    rho, nu = air["rho"], air["nu"]
    om = rpm * 2 * np.pi / 60
    r = blade.r * 1e-3
    c = blade.chord * 1e-3
    R = blade.r_tip * 1e-3
    beta = blade.pitch + (0 if twist is None else twist)
    B = blade.blades
    V = float(speed)
    if polars is None:
        re0 = om * r * c / nu
        polars = Polars(blade, re0, n_crit, model, stall_delay)
    polars.set_re(np.maximum(1.05 * om * r * c / nu, 1e3))
    sigma_e = None if duct is None else float(duct.get("exitAreaRatio", 1.0))
    if sigma_e is not None and V != 0:
        raise ValueError("a ducted rotor is modelled static (speed 0) here")

    def resid(phi):
        a = beta - phi
        cl, cd, _ = polars.at(a)
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl * np.sin(phi) + cd * np.cos(phi)
        sp, cp = np.sin(phi), np.cos(phi)
        if sigma_e is None:
            F = _loss(blade, blade.r, phi)
            k = B * c * cn / (8 * np.pi * r * F * sp * sp)          # w / (V + w)
            kp = B * c * ct / (8 * np.pi * r * F * sp * cp)          # a' / (1 - a')
            ap = kp / (1 + kp)
            res = om * r * (1 - ap) * np.tan(phi) * (1 - k) - V
        else:
            # duct: dT_rotor = 0.5 rho (w / sigma)^2 2 pi r dr and the element's 0.5 rho W^2 B c cn dr, W = w / sin(phi)
            k = B * c * cn * sigma_e ** 2 / (2 * np.pi * r * sp * sp)
            kp = B * c * ct / (8 * np.pi * r * sp * cp)
            ap = kp / (1 + kp)
            res = om * r * (1 - ap) * np.tan(phi) * (1 - k)
        return res, cl, cd, cn, ct, ap

    # bracket: scan down from the least angle of attack (largest phi) to the first sign change, the attached root
    top = np.minimum(beta + np.radians(25), 0.5 * np.pi - 1e-3)
    grid = np.linspace(1e-3, 1.0, 240)[:, None] * top[None, :]
    rg = resid(grid)[0]
    lo = np.empty(len(r))
    hi = np.empty(len(r))
    found = np.zeros(len(r), bool)
    for i in range(len(r)):
        idx = np.nonzero((rg[:-1, i] < 0) & (rg[1:, i] >= 0))[0]
        j = idx[-1] if len(idx) else int(np.argmin(np.abs(rg[:, i])))
        found[i] = len(idx) > 0
        lo[i], hi[i] = grid[j, i], grid[min(j + 1, len(grid) - 1), i]
    for _ in range(48):
        m = 0.5 * (lo + hi)
        neg = resid(m)[0] < 0
        lo, hi = np.where(neg, m, lo), np.where(neg, hi, m)
    phi = 0.5 * (lo + hi)
    _, cl, cd, cn, ct, ap = resid(phi)
    vt = om * r * (1 - ap)
    vax = vt * np.tan(phi)
    W = np.hypot(vax, vt)
    q = 0.5 * rho * W * W * c * B
    dT = q * cn                                             # N per m of span, all blades
    dQ = q * ct * r                                         # N m per m
    T = float(np.trapezoid(dT, r))
    Q = float(np.trapezoid(dQ, r))
    P = Q * om
    n = rpm / 60
    D = 2 * R
    ctc = T / (rho * n * n * D ** 4) if n > 0 else 0.0
    cpc = P / (rho * n ** 3 * D ** 5) if n > 0 else 0.0
    disk = np.pi * (R * R - (blade.r_hub * 1e-3) ** 2)
    fm = (max(T, 0) ** 1.5 / math.sqrt(2 * rho * disk)) / P if (V == 0 and P > 0 and sigma_e is None) else None
    re = W * c / nu
    return {"rpm": rpm, "speed": V, "thrust": T, "torque": Q, "power": P, "CT": ctc, "CP": cpc,
            "J": V / (n * D) if n > 0 else None, "eta": T * V / P if (V > 0 and P > 0) else None,
            "figureOfMerit": fm, "tipMach": float(om * R / air.get("a", A_SOUND)),
            "polars": polars.source, "converged": bool(found.all()),
            "stations": {"r": blade.r.tolist(), "phi": phi.tolist(), "alpha": (beta - phi).tolist(),
                         "cl": cl.tolist(), "cd": cd.tolist(), "re": re.tolist(), "W": W.tolist(),
                         "dT": (dT * 1e-3).tolist(), "dQ": (dQ * 1e-3).tolist(), "inflow": vax.tolist(),
                         "stalled": (cl < polars.at(beta - phi)[0] * 0).tolist()}}


def mack_n_crit(turbulence: float) -> float:
    """e^N transition's critical amplification from the free-stream turbulence intensity (Mack 1977):
    N = -8.43 - 2.4 ln(Tu). A rotor hovering in a room breathes its own recirculated wake (Tu ~ 1 %, N ~ 2.6);
    a quiet wind tunnel is N 9."""
    return float(np.clip(-8.43 - 2.4 * math.log(max(turbulence, 1e-5)), 0.5, 12.0))


TURBULENCE = 0.01


# ----------------------------------------------------------------- motor and battery

@dataclass
class Motor:
    """A brushless outrunner by its three constants: kv (rpm per volt), rm (winding resistance, ohm), i0 (no-load
    current, A, taken as constant: the iron and bearing loss it stands for grows with speed, so quote it near the
    working speed). max_current (A): the datasheet's burst rating, a limit not a model. rotor_inertia: the bell's
    own (kg m^2), which the rotor's adds to in the response time."""
    kv: float
    rm: float
    i0: float
    max_current: float = 30.0
    mass: float = 0.020
    rotor_inertia: float = 1.0e-6      # kg m^2: the spinning bell and magnets (an 1804's ~9 g at ~10.5 mm)

    @property
    def k_rad(self):                        # rad/s per volt
        return self.kv * 2 * np.pi / 60

    def at(self, volts: float, torque: float) -> dict:
        """Steady speed, current and electrical power at the motor's terminals for a shaft torque (N m)."""
        i = torque * self.k_rad + self.i0
        om = (volts - i * self.rm) * self.k_rad
        return {"current": i, "rpm": om * 60 / (2 * np.pi), "electrical": volts * i,
                "shaft": torque * om, "efficiency": torque * om / max(volts * i, 1e-9)}


@dataclass
class Battery:
    """cells in series at `cell_volts` under load, internal resistance per cell (ohm), and the ESC's on-resistance
    and loss in series with each motor."""
    cells: int
    cell_volts: float = 3.8
    r_cell: float = 0.012
    r_esc: float = 0.004

    @property
    def volts(self):
        return self.cells * self.cell_volts

    @property
    def r_pack(self):
        return self.cells * self.r_cell


class RotorMap:
    """Thrust and torque against rpm, from the blade-element model at a handful of speeds (each with its own
    Reynolds number), interpolated in log space: the motor and throttle solves then cost nothing."""

    def __init__(self, blade: Blade, rpm_range=(3000, 40000), n: int = 9, air=None, duct=None, twist_fn=None, **kw):
        self.blade = blade
        self.rpms = np.geomspace(*rpm_range, n)
        self.points = []
        for rpm in self.rpms:
            tw = twist_fn(rpm) if twist_fn else None
            self.points.append(performance(blade, rpm, air=air, duct=duct, twist=tw, **kw))
        self.T = np.array([max(p["thrust"], 1e-9) for p in self.points])
        self.Q = np.array([max(p["torque"], 1e-12) for p in self.points])
        self.lr = np.log(self.rpms)

    def thrust(self, rpm):
        return float(np.exp(np.interp(math.log(rpm), self.lr, np.log(self.T))))

    def torque(self, rpm):
        return float(np.exp(np.interp(math.log(rpm), self.lr, np.log(self.Q))))

    def rpm_for_thrust(self, thrust):
        return float(np.exp(np.interp(math.log(thrust), np.log(self.T), self.lr)))


def drive(rmap: RotorMap, motor: Motor, battery: Battery, throttle: float = 1.0, motors: int = 4) -> dict:
    """The rotor on its motor at a throttle (the ESC's duty, 0-1), every motor drawing the same: the speed where the
    motor's torque, at the pack's sagged voltage less its own winding and ESC drop, meets the rotor's. Thrust (N),
    rpm, current (A per motor), electrical power (W per motor, at the pack), shaft power, efficiency."""
    lo, hi = rmap.rpms[0], rmap.rpms[-1]
    for _ in range(60):
        rpm = 0.5 * (lo + hi)
        q = rmap.torque(rpm)
        i = q * motor.k_rad + motor.i0
        v_bus = battery.volts - motors * i * battery.r_pack
        v_m = throttle * v_bus - i * battery.r_esc
        rpm_m = (v_m - i * motor.rm) * motor.k_rad * 60 / (2 * np.pi)
        lo, hi = (rpm, hi) if rpm_m > rpm else (lo, rpm)
    q = rmap.torque(rpm)
    i = q * motor.k_rad + motor.i0
    v_bus = battery.volts - motors * i * battery.r_pack
    p_el = v_bus * i * throttle                       # the pack's watts per motor (an ideal ESC passes duty x current)
    shaft = q * rpm * 2 * np.pi / 60
    return {"throttle": throttle, "rpm": rpm, "thrust": rmap.thrust(rpm), "torque": q, "current": i,
            "packCurrent": i * throttle, "busVolts": v_bus, "electrical": p_el,
            "shaft": shaft, "efficiency": shaft / max(p_el, 1e-9),
            "gramsPerWatt": rmap.thrust(rpm) / 9.81 * 1000 / max(p_el, 1e-9),
            "overCurrent": bool(i > motor.max_current)}


def hover(rmap: RotorMap, motor: Motor, battery: Battery, thrust: float, motors: int = 4) -> dict:
    """The throttle, rpm and electrical power at which each rotor gives `thrust` (N); None if even full throttle
    can't."""
    full = drive(rmap, motor, battery, 1.0, motors)
    if full["thrust"] < thrust:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(50):
        t = 0.5 * (lo + hi)
        lo, hi = (t, hi) if drive(rmap, motor, battery, t, motors)["thrust"] < thrust else (lo, t)
    out = drive(rmap, motor, battery, hi, motors)
    out["fullThrottle"] = full
    return out


# ----------------------------------------------------------------- the blade as a rotating beam

def section_properties(blade: Blade, n: int = 61) -> dict:
    """Per station (SI): area A (m^2), the second moments about the section's own chordwise and normal centroidal
    axes (I_chord: bending out of the chord plane, the weak one; I_normal: edgewise), the product, the torsion
    constant J (thin-strip sum of t^3 / 3 along the chord, exact as t / c -> 0), and the chordwise position of the
    centroid from the leading edge (the elastic axis taken there)."""
    out = {k: np.zeros(len(blade.r)) for k in ("A", "I_chord", "I_normal", "I_cn", "J", "x_c", "y_max", "x_span")}
    for i in range(len(blade.r)):
        c = blade.chord[i] * 1e-3
        te_ratio = min(blade.te / blade.chord[i], 0.5 * blade.t_ratio[i])
        xy = section_coords(blade.camber[i], blade.t_ratio[i], te_ratio, n, airfoil=blade.airfoil) * c
        x, y = xy[:, 0], xy[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        a = 0.5 * cr.sum()
        cx, cy = ((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)
        u, v, u1, v1 = x - cx, y - cy, x1 - cx, y1 - cy
        cr = u * v1 - u1 * v
        sg = 1.0 if a > 0 else -1.0
        iyy = sg * ((v * v + v * v1 + v1 * v1) * cr).sum() / 12          # int v^2: about the chordwise axis
        ixx = sg * ((u * u + u * u1 + u1 * u1) * cr).sum() / 12          # int u^2: about the normal axis
        ixy = sg * ((u * v1 + 2 * u * v + 2 * u1 * v1 + u1 * v) * cr).sum() / 24
        # thickness along the chord for the torsion constant
        m = len(x) // 2
        xs = np.linspace(0, c, 80)
        up, lo = xy[: m + 1][::-1], xy[m:]
        t = np.interp(xs, up[:, 0], up[:, 1]) - np.interp(xs, lo[:, 0], lo[:, 1])
        out["A"][i] = abs(a)
        out["I_chord"][i] = iyy
        out["I_normal"][i] = ixx
        out["I_cn"][i] = ixy
        out["J"][i] = np.trapezoid(np.maximum(t, 0) ** 3, xs) / 3
        out["x_c"][i] = cx
        out["y_max"][i] = float(np.max(np.abs(v)))
        out["x_span"][i] = float(np.max(np.abs(u)))
    return out


def _beam(r, EI, N, q, rho_a=None, om=0.0, modes=0, inplane=False):
    """A rotating cantilever, clamped at r[0], on stations r (m): (EI w'')'' - (N w')' [- rho A om^2 w in plane] = q.
    Finite differences on a uniform grid. Returns (w, EI w'' moment, frequencies in Hz if modes)."""
    n = 80
    x = np.linspace(r[0], r[-1], n)
    h = x[1] - x[0]
    ei = np.interp(x, r, EI)
    nn = np.interp(x, r, N)
    qq = np.interp(x, r, q)
    # unknowns w_1..w_{n-1} (w_0 = 0), ghost w_{-1} = w_1 (slope 0 at the root), free end: M = 0, V = 0
    K = np.zeros((n, n))
    # moment operator D2 at interior nodes, assembled as D2^T EI D2 (energy form), plus N D1^T D1
    D2 = np.zeros((n, n))
    for i in range(1, n - 1):
        D2[i, i - 1], D2[i, i], D2[i, i + 1] = 1 / h ** 2, -2 / h ** 2, 1 / h ** 2
    D2[0, 0], D2[0, 1] = -2 / h ** 2, 2 / h ** 2                      # root with the ghost node: w'' = 2 (w1 - w0) / h^2
    wts = np.full(n, h)
    wts[0] = wts[-1] = 0.5 * h
    wts[-1] = 0.0                                                    # free tip carries no moment
    K += D2.T @ np.diag(wts * ei) @ D2
    D1 = np.zeros((n - 1, n))
    for i in range(n - 1):
        D1[i, i], D1[i, i + 1] = -1 / h, 1 / h
    nm = 0.5 * (nn[1:] + nn[:-1])
    K += D1.T @ np.diag(h * nm) @ D1
    mass = None
    if rho_a is not None:
        ma = np.interp(x, r, rho_a) * wts.clip(min=0.5 * h)
        mass = np.diag(ma)
        if inplane:
            K -= om * om * mass
    f = qq * np.full(n, h)
    f[0] *= 0.5
    f[-1] *= 0.5
    keep = np.arange(1, n)
    w = np.zeros(n)
    w[keep] = np.linalg.solve(K[np.ix_(keep, keep)], f[keep])
    mom = ei * (D2 @ w)
    mom[-1] = 0.0
    freqs = None
    if modes and mass is not None:
        import scipy.linalg as sl
        ev = sl.eigh(K[np.ix_(keep, keep)], mass[np.ix_(keep, keep)], eigvals_only=True, subset_by_index=[0, modes - 1])
        freqs = np.sqrt(np.maximum(ev, 0)) / (2 * np.pi)
    return np.interp(r, x, w), np.interp(r, x, mom), freqs


def structure(blade: Blade, op: dict, material="pa12", rpm: float | None = None, modes: int = 2) -> dict:
    """The blade at the operating point `op` (performance()): its out-of-plane (flap) and in-plane (lag) bending
    under the thrust and drag-torque loads with centrifugal stiffening, the elastic twist from the aerodynamic
    pitching moment, the lift's offset from the elastic axis and the centrifugal twisting moment, the root stress and
    the rotating natural frequencies. SI inside; mm, MPa, degrees and Hz out."""
    from .materials import material_from_spec
    mat = material_from_spec(material)
    E = mat.E * 1e6
    G = E / (2 * (1 + mat.nu))
    rho_m = mat.density * 1000
    rpm = rpm or op["rpm"]
    om = rpm * 2 * np.pi / 60
    r = blade.r * 1e-3
    sp = section_properties(blade)
    th = blade.pitch
    st = op["stations"]
    dT = np.asarray(st["dT"]) * 1e3 / blade.blades                 # N/m per blade, out of plane
    dQ = np.asarray(st["dQ"]) * 1e3 / blade.blades                 # N m / m per blade
    fq = dQ / r                                                      # N/m in plane, against the rotation
    # stiffness about the rotor's axes: the section's axes turned by the pitch
    ic, inn = sp["I_chord"], sp["I_normal"]
    ei_flap = E * (ic * np.cos(th) ** 2 + inn * np.sin(th) ** 2)
    ei_lag = E * (ic * np.sin(th) ** 2 + inn * np.cos(th) ** 2)
    rho_a = rho_m * sp["A"]
    cf = np.array([np.trapezoid((rho_a * om * om * r)[i:], r[i:]) for i in range(len(r))])   # N, centrifugal pull
    wf, mf, ff = _beam(r, ei_flap, cf, dT, rho_a, om, modes)
    wl, ml, fl = _beam(r, ei_lag, cf, fq, rho_a, om, modes, inplane=True)
    # torsion: aerodynamic moment about the elastic axis (centroid) + centrifugal twisting moment
    W = np.asarray(st["W"])
    c = blade.chord * 1e-3
    cl = np.asarray(st["cl"])
    qd = 0.5 * 1.2 * W * W
    cm_air = performance_cm(blade, op)
    m_aero = qd * c * c * cm_air - qd * c * cl * (sp["x_c"] - 0.25 * c)    # + nose up (more pitch)
    ip = rho_m * (inn - ic)                                          # kg m per m: the CTM's inertia difference
    m_ctm = -ip * om * om * np.sin(th) * np.cos(th)                  # toward flat pitch
    mt = m_aero + m_ctm
    Mt = np.array([np.trapezoid(mt[i:], r[i:]) for i in range(len(r))])
    gj = G * sp["J"]
    # tennis-racket stiffening of torsion: d/dr(GJ phi') - Ip om^2 cos(2 th) phi = -mt, solved as a beam-like system
    twist = _torsion(r, gj, ip * om * om * np.cos(2 * th), mt)
    # root stress: pull + both bending moments resolved onto the section's own axes
    i0 = 0
    mflap, mlag = mf[i0], ml[i0]
    m_c = mflap * np.cos(th[i0]) + mlag * np.sin(th[i0])            # about the chordwise axis
    m_n = -mflap * np.sin(th[i0]) + mlag * np.cos(th[i0])           # about the normal axis
    s_pull = cf[i0] / sp["A"][i0]
    s_bend = abs(m_c) * sp["y_max"][i0] / ic[i0] + abs(m_n) * sp["x_span"][i0] / inn[i0]
    sig = (s_pull + s_bend) / 1e6
    f1 = om / (2 * np.pi)
    out = {"material": mat.name, "rpm": rpm, "rotorHz": f1,
           "tipFlap_mm": float(wf[-1] * 1e3), "tipLag_mm": float(wl[-1] * 1e3),
           "coning_deg": float(np.degrees(np.arctan(wf[-1] / (r[-1] - r[0])))),
           "tipTwist_deg": float(np.degrees(twist[-1])), "twist": twist.tolist(),
           "twist_deg_075": float(np.degrees(np.interp(0.75 * r[-1], r, twist))),
           "rootPull_N": float(cf[i0]), "rootStress": float(sig), "rootStressPull": float(s_pull / 1e6),
           "rootStressBending": float(s_bend / 1e6), "strength": mat.strength,
           "safetyFactor": float(mat.strength / max(sig, 1e-9)),
           "flapHz": None if ff is None else [float(v) for v in ff],
           "lagHz": None if fl is None else [float(v) for v in fl],
           "bladeMass_g": float(np.trapezoid(rho_a, r) * 1e3)}
    if ff is not None:
        harm = [k * f1 for k in (1, 2, 3, blade.blades)]
        near = [min(abs(f / h - 1) for h in harm) for f in list(ff) + list(fl)]
        out["frequencyMargin"] = float(min(near))
    return out


def performance_cm(blade, op):
    """The sections' quarter-chord pitching moment at the operating point's angles of attack."""
    pol = op.get("_polars")
    if pol is None:
        return np.zeros(len(blade.r))
    return pol.at(np.asarray(op["stations"]["alpha"]))[2]


def _torsion(r, gj, kt, mt):
    """GJ phi'' (as d/dr(GJ phi')) - kt phi = -mt, phi(root) = 0, phi'(tip) = 0. Finite differences."""
    n = 80
    x = np.linspace(r[0], r[-1], n)
    h = x[1] - x[0]
    g = np.interp(x, r, gj)
    k = np.interp(x, r, kt)
    m = np.interp(x, r, mt)
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = 1.0
    for i in range(1, n):
        gm = 0.5 * (g[i] + g[i - 1])
        gp = 0.5 * (g[i] + g[min(i + 1, n - 1)])
        if i < n - 1:
            A[i, i - 1] = gm / h ** 2
            A[i, i + 1] = gp / h ** 2
            A[i, i] = -(gm + gp) / h ** 2 - k[i]
        else:                                    # free tip: ghost node mirrors, half cell
            A[i, i - 1] = 2 * gm / h ** 2
            A[i, i] = -2 * gm / h ** 2 - k[i]
        b[i] = -m[i]
    phi = np.linalg.solve(A, b)
    return np.interp(r, x, phi)


def aeroelastic(blade: Blade, rpm: float, material="pa12", speed: float = 0.0, air=None, duct=None, iters: int = 6,
                **kw) -> tuple:
    """Performance and structure solved together: the air loads twist and bend the blade, the twist changes the
    pitch the air sees, until the twist settles. Returns (performance, structure)."""
    air = air or {"rho": RHO, "nu": NU, "a": A_SOUND}
    om = rpm * 2 * np.pi / 60
    pol = kw.pop("polars", None)
    if pol is None:
        re0 = om * blade.r * 1e-3 * blade.chord * 1e-3 / air["nu"]
        pol = Polars(blade, re0, kw.pop("n_crit", mack_n_crit(TURBULENCE)), kw.pop("model", "auto"),
                     kw.pop("stall_delay", 3.0))
    twist = np.zeros(len(blade.r))
    for _ in range(iters):
        op = performance(blade, rpm, speed, air, polars=pol, twist=twist, duct=duct)
        op["_polars"] = pol
        st = structure(blade, op, material, modes=2)
        twist = 0.5 * (twist + np.asarray(st["twist"]))       # fixed count, no early stop: smooth for the optimizer
    op = performance(blade, rpm, speed, air, polars=pol, twist=twist, duct=duct)
    op["_polars"] = pol
    st = structure(blade, op, material, modes=2)
    op.pop("_polars")
    return op, st


# ----------------------------------------------------------------- the rotor on its motor, flexing

def _at_rpm(blade, rpm, material, air, duct, iters, polars_cache, kw):
    key = round(float(rpm), 1)
    if key not in polars_cache:
        polars_cache[key] = aeroelastic(blade, rpm, material, air=air, duct=duct, iters=iters, **dict(kw))
    return polars_cache[key]


def evaluate(blade: Blade, motor: Motor, battery: Battery, hover_thrust: float, material="pa12", motors: int = 4,
             air=None, duct=None, iters: int = 5, **kw) -> dict:
    """The flexing blade on its motor and pack: the hover point (rpm where the deformed blade gives hover_thrust,
    electrical watts drawn from the pack, throttle) and the full-throttle point (rpm where the motor's torque at the
    sagged pack voltage meets the deformed blade's), each with its performance and structure."""
    cache: dict = {}
    if "polars" not in kw:
        kw = {**kw, "polars": polars_for(blade, (4000, 50000), air)}

    def thrust_at(rpm):
        return _at_rpm(blade, rpm, material, air, duct, iters, cache, kw)[0]["thrust"]

    # hover: secant on log rpm against log thrust (thrust ~ rpm^2)
    r0 = 15000.0
    t0 = thrust_at(r0)
    r1 = r0 * math.sqrt(hover_thrust / max(t0, 1e-6))
    for _ in range(8):
        r1 = float(np.clip(r1, 2000, 80000))
        t1 = thrust_at(r1)
        if abs(t1 / hover_thrust - 1) < 1e-4:
            break
        slope = (math.log(max(t1, 1e-9)) - math.log(max(t0, 1e-9))) / (math.log(r1) - math.log(r0)) if r1 != r0 else 2.0
        slope = slope if slope > 0.5 else 2.0
        r0, t0 = r1, t1
        r1 = r1 * math.exp((math.log(hover_thrust) - math.log(max(t1, 1e-9))) / slope)
    op_h, st_h = _at_rpm(blade, r1, material, air, duct, iters, cache, kw)
    h = _electrical(op_h, motor, battery, motors)

    # full throttle: rpm where the motor at duty 1 turns the blade
    def gap(rpm):
        op = _at_rpm(blade, rpm, material, air, duct, iters, cache, kw)[0]
        e = _electrical(op, motor, battery, motors)
        return e["throttle"] - 1.0

    a, ga = r1, gap(r1)
    b = r1 * math.sqrt(1.0 / max(ga + 1.0, 0.05))
    gb = gap(b)
    for _ in range(10):
        if abs(gb) < 1e-4 or b == a:
            break
        a, ga, b = b, gb, float(np.clip(b - gb * (b - a) / (gb - ga), 0.5 * b, 2.0 * b))
        gb = gap(b)
    lo = b
    op_f, st_f = _at_rpm(blade, lo, material, air, duct, iters, cache, kw)
    f = _electrical(op_f, motor, battery, motors)
    resp = response(blade, op_h, motor, battery, material)
    return {"hover": {**h, **resp, "rpm": op_h["rpm"], "thrust": op_h["thrust"], "figureOfMerit": op_h["figureOfMerit"],
                      "tipMach": op_h["tipMach"], "performance": op_h, "structure": st_h,
                      "gramsPerWatt": op_h["thrust"] / 9.81 * 1000 / h["electrical"]},
            "full": {**f, "rpm": op_f["rpm"], "thrust": op_f["thrust"], "tipMach": op_f["tipMach"],
                     "performance": op_f, "structure": st_f},
            "converged": bool(op_h["converged"] and op_f["converged"])}


def inertia(blade: Blade, material="pa12", hub_height: float = 6.0, bore: float = 5.0) -> float:
    """The rotor's polar moment of inertia (kg m^2): its blades from their sections, plus the hub ring."""
    from .materials import material_from_spec
    rho = material_from_spec(material).density * 1000
    r = blade.r * 1e-3
    a = section_properties(blade)["A"]
    ib = blade.blades * rho * np.trapezoid(a * r * r, r)
    ro, ri, h = blade.r_hub * 1e-3, 0.5 * bore * 1e-3, hub_height * 1e-3
    ih = rho * np.pi * (ro ** 2 - ri ** 2) * h * 0.5 * (ro ** 2 + ri ** 2)
    return float(ib + ih)


def response(blade: Blade, op: dict, motor: Motor, battery: Battery, material="pa12") -> dict:
    """How fast the rotor follows a throttle change about this point: the small-signal time constant
    tau = I / (dQ_prop/dw + 1 / (k^2 R)), the rotor's inertia against its own aerodynamic damping (Q ~ w^2, so
    2 Q / w) and the motor's back-EMF damping through the winding and ESC resistance. A flight controller's
    attitude loop can't be faster than its motors: tau sets the bandwidth, 1 / (2 pi tau)."""
    I = inertia(blade, material) + motor.rotor_inertia
    om = op["rpm"] * 2 * np.pi / 60
    k = motor.k_rad
    damp = 2 * max(op["torque"], 0) / max(om, 1e-6) + 1 / (k * k * (motor.rm + battery.r_esc))
    tau = I / damp
    return {"inertia_kgm2": I, "responseTime_ms": tau * 1e3, "bandwidthHz": 1 / (2 * np.pi * tau)}


def _electrical(op, motor, battery, motors):
    """The pack's watts for one rotor at this rpm and torque: winding and ESC drops on the motor's current, the
    duty that leaves the motor the voltage it needs, and this motor's share of the pack's own loss."""
    k = motor.k_rad
    q = max(op["torque"], 0.0)
    om = op["rpm"] * 2 * np.pi / 60
    i = q * k + motor.i0
    v_need = om / k + i * (motor.rm + battery.r_esc)
    # bus sags with the pack current motors x duty x i; duty = v_need / v_bus
    v_bus = battery.volts
    for _ in range(20):
        duty = v_need / max(v_bus, 1e-6)
        v_bus = battery.volts - motors * duty * i * battery.r_pack
    duty = v_need / max(v_bus, 1e-6)
    p_el = duty * i * v_bus + (duty * i) ** 2 * battery.r_pack       # what the cells give per motor
    return {"throttle": duty, "current": i, "busVolts": v_bus, "electrical": p_el, "shaft": q * om,
            "efficiency": q * om / max(p_el, 1e-9), "overCurrent": bool(i > motor.max_current)}


# ----------------------------------------------------------------- design

KNOTS = np.array([0.0, 1 / 3, 2 / 3, 1.0])


def _spline(s, knots, vals):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(knots, vals)(np.clip(s, 0, 1))


def tip_closure(r, r_tip, c_tip):
    """How the planform closes at the tip: a blade cut off square at full chord ends in a flat face whose two corners
    each shed a vortex; closed over the last half chord as a quarter ellipse, the chord (and with it the circulation)
    falls to zero at the tip, the loading the tip-loss factor already says the air takes there. A factor on chord."""
    l = max(0.5 * c_tip, 1e-6)
    u = np.clip((np.asarray(r, float) - (r_tip - l)) / l, 0.0, 1.0)
    return np.sqrt(np.maximum(1.0 - u * u, 0.0))


def blade_from_params(x, blades, r_tip, r_hub, stations=24, te=0.3, t_min=0.8, hub_height=None):
    """Design variables -> Blade. x: chord at 4 knots (mm), geometric pitch at 3 knots (mm per turn), camber at root
    and tip (deg), thickness ratio at root and tip. Thickness is held to t_min mm (the process's thinnest
    reliable wall) wherever the ratio would draw it thinner. With `hub_height` (mm, the motor's shaft length a
    prop can take), the root knot's chord is held to what stands inside that hub (root_chord_limit): the blade
    necks down to a shank there, and the chord spline carries it out to the blade's working chord."""
    x = np.asarray(x, float)
    bl = _blade_from_params(x, blades, r_tip, r_hub, stations, te, t_min)
    if hub_height:
        # the root's footprint on the hub (its section grown by a t_min fillet, with half t_min of hub beyond it,
        # hub_height()) must stand inside the hub: the footprint's depth goes about as the chord, so scale it in
        x = x.copy()
        for _ in range(4):
            need = _hub_depth(bl, t_min, 0.5 * t_min)
            if need <= hub_height + 1e-3:
                break
            room = hub_height - 3 * t_min
            x[0] = max(1.0, x[0] * max(room, 0.1) / max(need - 3 * t_min, 1e-3) * 0.995)
            bl = _blade_from_params(x, blades, r_tip, r_hub, stations, te, t_min)
        bl.note["hubHeight"] = float(hub_height)
    return bl


def _blade_from_params(x, blades, r_tip, r_hub, stations, te, t_min):
    r = _stations(r_hub, r_tip, stations)
    s = (r - r_hub) / (r_tip - r_hub)
    chord = np.maximum(_spline(s, KNOTS, x[0:4]), 1.0)
    chord = chord * tip_closure(r, r_tip, float(chord[-1]))
    chord = np.maximum(chord, 1.0)
    pitch_len = _spline(s, np.array([0.0, 0.5, 1.0]), x[4:7])
    pitch = np.arctan(pitch_len / (2 * np.pi * r))
    # neighbouring blades can't overlap: each section's span round the circle, chord x cos(pitch), stays within 80%
    # of its share of that radius's circumference (it's the hub's at the root)
    chord = np.minimum(chord, 0.8 * 2 * np.pi * r / (int(blades) * np.maximum(np.cos(pitch), 0.05)))
    camber = np.radians(np.interp(s, [0, 1], x[7:9]))
    tr = np.interp(s, [0, 1], x[9:11])
    tr = np.minimum(np.maximum(tr, t_min / chord), 0.3)
    return Blade(r, chord, pitch, camber, tr, int(blades), r_hub, r_tip, te,
                 note={"params": [float(v) for v in x], "t_min": t_min})


def ideal_start(blades, r_tip, r_hub, hover_thrust, rpm, cl=0.8, alpha_deg=4.0, air=None):
    """A first blade from momentum theory: the even induced velocity of the least induced power (w from
    T = 2 rho A w^2), chord from the annulus balance B c cl = 8 pi r F sin^2(phi) at the lift coefficient asked for,
    pitch = inflow angle + alpha. Returns design variables for blade_from_params."""
    air = air or {"rho": RHO}
    R, rh = r_tip * 1e-3, r_hub * 1e-3
    A = np.pi * (R * R - rh * rh)
    w = math.sqrt(hover_thrust / (2 * air["rho"] * A))
    om = rpm * 2 * np.pi / 60
    out_c, out_p = [], []
    for sk in KNOTS:
        rr = rh + sk * (R - rh)
        rr = min(max(rr, rh + 0.03 * (R - rh)), R * 0.97)
        phi = math.atan(w / (om * rr))
        f = 2 / np.pi * math.acos(math.exp(-blades / 2 * (R - rr) / (rr * math.sin(phi))))
        c = 8 * np.pi * rr * f * math.sin(phi) ** 2 / (blades * cl)
        out_c.append(c * 1e3)
    for sk in (0.0, 0.5, 1.0):
        rr = rh + sk * (R - rh)
        rr = max(rr, rh + 0.25 * (R - rh))
        phi = math.atan(w / (om * rr))
        out_p.append(2 * np.pi * rr * math.tan(phi + math.radians(alpha_deg)) * 1e3)
    return np.array(out_c + out_p + [5.0, 4.0, 0.12, 0.08])


def design(motor: Motor, battery: Battery, hover_thrust: float, max_thrust: float, r_tip: float, r_hub: float,
           blades=(2, 3, 4), material="pa12", t_min: float = 0.8, te: float = 0.3, sf_min: float = 2.0,
           tip_mach: float = 0.6, frequency_margin: float = 0.08, motors: int = 4, air=None, duct=None,
           response_time: float | None = None, start=None, maxiter: int = 40, log=None,
           hub_height: float | None = None) -> dict:
    """The blade that draws the least power from the pack at hover while still giving max_thrust (N per rotor) at
    full throttle, every operating point solved on the flexing blade. Constraints: the root's safety factor at
    full throttle (sf_min, on the material's strength), tip Mach number at full throttle, the blade's flap and lag
    frequencies at least `frequency_margin` off 1x, 2x, 3x and blade-pass at hover, thickness >= t_min mm, and
    optionally the motor-and-rotor response time at hover (ms, `response_time`: what the attitude loop needs).
    `hub_height` (mm): the hub the motor's shaft takes; the root necks down to stand inside it
    (blade_from_params), and the root's stress constraint is then the shank's.
    One design per blade count; returns {"best", "designs": [...]} with each one's blade and evaluation."""
    from scipy.optimize import minimize
    out = []
    for B in blades:
        x0 = np.asarray(start, float) if start is not None else ideal_start(B, r_tip, r_hub, hover_thrust, 14000.0)
        lo = np.array([2.0] * 4 + [5.0] * 3 + [0.0, 0.0, 0.05, 0.04])
        # chord: blade-element momentum treats each blade apart; past a local solidity of ~0.3 the blades work as a
        # cascade it doesn't model, so the chord is held inside that (at 0.4 R and outboard)
        cmax = 0.3 * 2 * np.pi * (r_hub + np.array([0.4, 0.4, 2 / 3, 1.0]) * (r_tip - r_hub)) / B
        hi = np.array(list(np.minimum(cmax, 0.6 * r_tip)) + [8 * r_tip] * 3 + [30.0, 30.0, 0.25, 0.16])
        x0 = np.clip(x0, lo, hi)
        scale = hi - lo
        memo: dict = {}

        def ev(z):
            key = tuple(np.round(z, 10))
            if key not in memo:
                x = lo + z * scale
                bl = blade_from_params(x, B, r_tip, r_hub, te=te, t_min=t_min, hub_height=hub_height)
                try:
                    e = evaluate(bl, motor, battery, hover_thrust, material, motors, air, duct)
                except Exception as exc:                                # a geometry the model can't solve
                    e = {"error": str(exc)}
                memo[key] = (bl, e)
            return memo[key]

        def obj(z):
            bl, e = ev(z)
            return (e["hover"]["electrical"] / 10.0) if "hover" in e else 1e3

        def cons(z):
            bl, e = ev(z)
            if "hover" not in e:
                return -np.ones(5)
            f, h = e["full"], e["hover"]
            return np.array([f["thrust"] / max_thrust - 1.0,
                             f["structure"]["safetyFactor"] / sf_min - 1.0,
                             1.0 - f["tipMach"] / tip_mach,
                             h["structure"].get("frequencyMargin", 1.0) / frequency_margin - 1.0,
                             1.0 - h["responseTime_ms"] / response_time if response_time else 1.0])

        z0 = (x0 - lo) / scale
        res = minimize(obj, z0, method="SLSQP", bounds=[(0, 1)] * len(z0),
                       constraints=[{"type": "ineq", "fun": cons}],
                       options={"maxiter": maxiter, "ftol": 1e-5, "eps": 2e-3})
        bl, e = ev(res.x)
        c = cons(res.x)
        rec = {"blades": B, "blade": bl, "evaluation": e, "feasible": bool(np.all(c > -1e-3)), "constraints": c.tolist(),
               "iterations": int(res.nit), "evaluations": len(memo), "message": str(res.message)}
        if log:
            log(f"{B} blades: hover {e['hover']['electrical']:.2f} W, full {e['full']['thrust']:.2f} N, "
                f"feasible {rec['feasible']}, {len(memo)} evaluations")
        out.append(rec)
    ok = [d for d in out if d["feasible"]] or out
    best = min(ok, key=lambda d: d["evaluation"]["hover"]["electrical"] if "hover" in d["evaluation"] else 1e9)
    return {"best": best, "designs": out}


# ----------------------------------------------------------------- printable rotor

def to_fan_rotor(blade: Blade):
    """The same blade as a fan.Rotor (angles from the axis), whose section drawing (circular-arc camber line, NACA
    four-digit thickness, trailing edge opened to te) is the section this module analysed."""
    from .fan import Rotor
    stagger = 0.5 * np.pi - blade.pitch
    return Rotor(r=blade.r.copy(), chord=blade.chord.copy(), kappa1=stagger + 0.5 * blade.camber,
                 kappa2=stagger - 0.5 * blade.camber, thickness=blade.thickness(), te=blade.te,
                 blades=blade.blades, r_hub=blade.r_hub, r_tip=blade.r_tip, rpm=0.0, tip="open")


def root_fillet(blade: Blade, root_stress: float, strength: float, sf: float = 2.0, t_min: float = 0.8) -> dict:
    """The radius of the fillet where the blade meets the hub, from the stress it has to carry. A shoulder fillet of
    radius r on a section of thickness t raises the bending stress by Kt = 1 + 0.2 (r / t)^-0.55 (a fit to Peterson's
    stepped flat bar in bending at a large step, within 0.1 of his chart for r / t 0.05-0.3, above it beyond). The
    radius is the smallest that keeps Kt x the root's nominal stress (structure()['rootStress'], at full throttle)
    within strength / sf, and never under the process's thinnest wall nor over three root thicknesses."""
    t = float(blade.thickness()[0])
    kt_allow = strength / max(sf * root_stress, 1e-9)
    if kt_allow <= 1.0:
        r = 3.0 * t
    else:
        r = t * (0.2 / (kt_allow - 1.0)) ** (1 / 0.55)
    r = float(np.clip(r, t_min, 3.0 * t))
    return {"radius": r, "rootThickness": t, "Kt": 1 + 0.2 * (r / t) ** -0.55, "KtAllowed": kt_allow}


def _fillet_growth(d, rf):
    """Extra width (both sides together) of a circular fillet of radius rf at distance d from the surface it meets."""
    d = np.asarray(d, float)
    g = 2 * (rf - np.sqrt(np.maximum(rf * rf - (rf - np.clip(d, 0, rf)) ** 2, 0.0)))
    return np.where(d <= 0, 2 * rf, g)


def blade_solid(blade: Blade, fillet: float = 0.0, z_mid: float = 0.0, root_in: float = 1.5, rotation: str = "ccw",
                spanwise: int = 60, chordwise: int = 48):
    """One blade as a closed mesh (mm) along +x, its sections (the analysed ones: fan._section's circular-arc camber
    line and NACA thickness) stacked on their centroids at z_mid, reaching root_in into the hub. Over the last
    `fillet` mm before the hub (r_hub) the section's thickness and chord both grow by a circular fillet's profile,
    so the union with the hub's cylinder (on which every section lies at its own radius) meets it with that radius
    all round the root."""
    import trimesh
    from .fan import _section
    rot = to_fan_rotor(blade)
    rs = np.concatenate([np.linspace(blade.r_hub - root_in, blade.r_hub + max(fillet, 0.5), 12)[:-1],
                         np.linspace(blade.r_hub + max(fillet, 0.5), blade.r_tip, spanwise)])
    rc = np.clip(rs, blade.r[0], blade.r[-1])
    it = lambda v: np.interp(rc, blade.r, v)
    ch, k1, k2, tk = it(rot.chord), it(rot.kappa1), it(rot.kappa2), it(rot.thickness)
    grow = _fillet_growth(rs - blade.r_hub, fillet) if fillet > 0 else np.zeros_like(rs)
    rings = []
    for j, r in enumerate(rs):
        loop = _section(rot, 0, chordwise, chord=ch[j] + grow[j], kappa1=k1[j], kappa2=k2[j], thick=tk[j] + grow[j])
        ang = loop[:, 0] / r * (1 if rotation == "ccw" else -1)
        rings.append(np.c_[r * np.cos(ang), r * np.sin(ang), loop[:, 1] + z_mid])
    m = len(rings[0])
    v = np.vstack(rings)
    f = []
    for j in range(len(rings) - 1):
        for k in range(m):
            a, b = j * m + k, j * m + (k + 1) % m
            f += [[a, b, b + m], [a, b + m, a + m]]
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
    mesh = trimesh.Trimesh(v, np.vstack([np.array(f), cap[:, ::-1], cap + last]), process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def hub_height(blade: Blade, fillet: float = 0.0, wall: float = 1.0) -> float:
    """The hub's height the root needs (_hub_depth)."""
    return _hub_depth(blade, fillet, wall)


def _hub_depth(blade: Blade, fillet: float = 0.0, wall: float = 1.0) -> float:
    """The hub's height: the root section's own axial depth (fillet included), with `wall` above and below, so the
    root lands whole on the hub's cylinder and its fillet doesn't run off the hub's top or bottom edge."""
    from .fan import _section
    rot = to_fan_rotor(blade)
    loop = _section(rot, 0, 48, chord=rot.chord[0] + 2 * fillet, thick=rot.thickness[0] + 2 * fillet)
    return float(np.ptp(loop[:, 1]) + 2 * wall)


def rotor_mesh(blade: Blade, bore: float = 5.0, height: float | None = None, rotation: str = "ccw",
               fillet: float | None = None, wall: float = 1.0):
    """The printable rotor (mm): its blades on a hub ring of radius r_hub bored for the motor's shaft, the hub as
    tall as the root needs (hub_height; or `height`), each root filleted onto it (`fillet` mm; default 0.6 x the
    root's thickness, Kt about 1.25: pass root_fillet()['radius'] to size it from the stress). `rotation`: the way
    it turns seen from above (the inlet side); a quad needs two of each. Watertight (manifold3d union)."""
    import manifold3d as m3d
    import trimesh
    from .fan import _revolve
    rf = 0.6 * float(blade.thickness()[0]) if fillet is None else float(fillet)
    h = height or hub_height(blade, rf, wall)
    one = blade_solid(blade, rf, z_mid=0.5 * h, rotation=rotation)
    parts = [one.copy().apply_transform(trimesh.transformations.rotation_matrix(2 * np.pi * k / blade.blades, (0, 0, 1)))
             for k in range(blade.blades)]
    parts.append(_revolve([(0.5 * bore, 0.0), (blade.r_hub, 0.0), (blade.r_hub, h), (0.5 * bore, h)]))

    def man(t):
        return m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))

    u = m3d.Manifold.batch_boolean([man(p) for p in parts], m3d.OpType.Add)
    o = u.to_mesh()
    out = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
    out.metadata.update({"hubHeight": h, "fillet": rf, "bore": bore})
    return out


def solid_check(blade: Blade, op: dict, material="pa12", mesh=None, bore: float = 5.0, size: float | None = None,
                clamp_height: float | None = None) -> dict:
    """The printed rotor as a 3D solid (quadratic tetrahedra) under its spin and its air loads at the operating
    point `op` (performance() on this blade, e.g. evaluate()['full']['performance']): every node carries its share
    of the centrifugal body force rho omega^2 r, and each blade's surface carries its stations' thrust and drag
    forces over the band of radius each station stands for. Held at the bore (the shaft and its nut clamp the hub's
    faces within clamp_height of the bore, default the bore radius out). Returns the peak equivalent stress on the
    blades and its radius, the safety factor on the material's strength, and the beam model's root stress beside it.
    The check the strip beam can't make: the root's fillet, the hub, the section's real shape."""
    from .crosscheck import _p2_mesh, _read_stress, _solve, _tet_mesh, reading_radius
    from .materials import material_from_spec
    mat = material_from_spec(material)
    mesh = mesh if mesh is not None else rotor_mesh(blade, bore=bore)
    t_root = float(blade.thickness()[0])
    size = size or max(0.35 * t_root, 0.25)
    pts, tets = _tet_mesh(mesh, size)
    nodes, conn, edges = _p2_mesh(pts, tets)
    nv = len(pts)
    om = op["rpm"] * 2 * np.pi / 60
    rho = mat.density * 1e-9                        # kg/mm^3
    # lumped vertex volumes (a quarter of each tet to each vertex)
    x = pts[tets]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6
    vv = np.zeros(nv)
    np.add.at(vv, tets.ravel(), np.repeat(vol / 4, 4))
    f = np.zeros((nv, 3))
    rxy = np.linalg.norm(pts[:, :2], axis=1)
    # centrifugal: kg mm^-3 x mm^3 x s^-2 x mm = kg mm s^-2 = 1e-3 N
    f[:, :2] += (rho * vv * om * om)[:, None] * pts[:, :2] * 1e-3
    # air loads: per station band, spread over that band's surface vertices of each blade by area
    import trimesh
    surf = trimesh.Trimesh(pts, _boundary(tets), process=False)
    area = np.zeros(nv)
    np.add.at(area, surf.faces.ravel(), np.repeat(surf.area_faces / 3, 3))
    st = op["stations"]
    r_st = np.asarray(st["r"])
    edges_r = np.concatenate([[blade.r_hub], 0.5 * (r_st[1:] + r_st[:-1]), [blade.r_tip + 1]])
    dT = np.asarray(st["dT"]) / blade.blades          # N/mm per blade
    dQ = np.asarray(st["dQ"]) / blade.blades          # N m / mm per blade
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    on_blade = area > 0
    for k in range(blade.blades):
        a0 = 2 * np.pi * k / blade.blades
        near = np.abs(np.angle(np.exp(1j * (ang - a0)))) < np.pi / blade.blades
        for i in range(len(r_st)):
            sel = on_blade & near & (rxy >= edges_r[i]) & (rxy < edges_r[i + 1]) & (rxy > blade.r_hub + 0.2)
            if not sel.any():
                continue
            w = area[sel] / area[sel].sum()
            span = edges_r[i + 1] - edges_r[i]
            fz = dT[i] * span                                         # thrust, +z (the inlet side)
            ft = dQ[i] * span / (r_st[i] * 1e-3)                      # drag, against the rotation (ccw from +z)
            tang = np.c_[-pts[sel, 1], pts[sel, 0]] / np.maximum(rxy[sel, None], 1e-9)
            f[sel, 2] += fz * w
            f[sel, :2] -= ft * w[:, None] * tang
    ch = clamp_height if clamp_height is not None else 0.5 * bore
    hold = (rxy <= 0.5 * bore + 0.05) | ((rxy <= 0.5 * bore + ch) & ((pts[:, 2] < pts[:, 2].min() + 0.05) |
                                                                    (pts[:, 2] > pts[:, 2].max() - 0.05)))
    fixed = np.zeros((len(nodes), 3), bool)
    fixed[:nv][hold] = True
    E = mat.E
    u = _solve(nodes, conn, E, mat.nu, fixed.ravel(), None, [("spin+air", f, None)], nv, edges)[0]
    # von Mises of the stress averaged over the material's length, and over no more than 4/9 of the blade's
    # thickness (crosscheck.reading_radius): the root's peak as the material answers to it, on any mesh
    radius = reading_radius(mat, size, float(np.min(blade.thickness())))[0]
    vm = _read_stress(nodes, conn, [u], E, mat.nu, nv, radius)[0][1]
    away = rxy > blade.r_hub + 0.05                   # the blades and their roots (the hub's bore is held)
    i = int(np.argmax(np.where(away, vm, 0)))
    tip = pts[:, 0].argmax()
    return {"peakStress": float(vm[i]), "peakRadius": float(rxy[i]), "safetyFactor": float(mat.strength / max(vm[i], 1e-9)),
            "hubPeak": float(vm[~away].max()) if (~away).any() else 0.0, "elements": int(len(tets)), "size": size, "averagedOver_mm": float(radius),
            "tipLift_mm": float(u[tip, 2]), "rpm": op["rpm"], "material": mat.name}


def _boundary(tets):
    f = np.sort(np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    u, c = np.unique(f, axis=0, return_counts=True)
    return u[c == 1]


# ----------------------------------------------------------------- validation

def validate(sample: int = 3, **kw) -> dict:
    """The model against published static tests of 3D-printed propellers (loadpath/data/propellers_uiuc.json:
    NR640, 5 and 9 in, 2, 3 and 4 blades, two pitches; UIUC propeller database). For each propeller, `sample`
    speeds from its run: predicted and measured CT and CP. Returns rows and the mean and RMS of the ratio - 1."""
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).parent / "data" / "propellers_uiuc.json").read_text())
    cy = np.asarray(d["clarky"])
    th = np.asarray(d["thickness"])
    rows = []
    for p in d["propellers"]:
        g = np.asarray(p["geometry"])
        R = p["diameter_in"] * 25.4 / 2
        rr = np.linspace(0.16, 0.995, 30)
        bl = Blade(rr * R, np.interp(rr, g[:, 0], g[:, 1]) * R, np.radians(np.interp(rr, g[:, 0], g[:, 2]) + p["pitchOffset_deg"]),
                   0.0, np.minimum(np.interp(rr, th[:, 0], th[:, 1]), 0.3), p["blades"], 0.15 * R, R, te=0.13, airfoil=cy)
        st = np.asarray(p["static"])
        for row in st[np.linspace(0, len(st) - 1, sample).round().astype(int)]:
            o = performance(bl, float(row[0]), n_crit=kw.get("n_crit", mack_n_crit(TURBULENCE)),
                            stall_delay=kw.get("stall_delay", 3.0), model=kw.get("model", "auto"))
            rows.append({"propeller": p["name"], "blades": p["blades"], "diameter_in": p["diameter_in"],
                         "rpm": float(row[0]), "CT": o["CT"], "CTmeasured": float(row[1]), "CP": o["CP"],
                         "CPmeasured": float(row[2]), "polars": o["polars"]})
    e = np.array([[r["CT"] / r["CTmeasured"] - 1, r["CP"] / r["CPmeasured"] - 1] for r in rows])
    return {"rows": rows, "CTmean": float(e[:, 0].mean()), "CTrms": float(np.sqrt((e[:, 0] ** 2).mean())),
            "CPmean": float(e[:, 1].mean()), "CPrms": float(np.sqrt((e[:, 1] ** 2).mean())), "source": d["source"]}
