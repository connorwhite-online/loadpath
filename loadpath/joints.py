"""
Printed joints sized from physics: coarse threads, hinges and snap fits.

A printed part is often held to its neighbour by a joint printed with it:
a thumbscrew in a printed nut, a hinge pin through printed knuckles, a
snap. Each fails in a few known ways, and each way is a stress on a
surface whose direction against the layers decides what the print can
take. This sizes the joint from the load it carries and checks every way
it can fail, with the process's tolerances and the layers in the sums:

**Threads** (`thread`). A coarse trapezoidal thread with chosen flank
angles (60 degrees included, ISO's form; 90 degrees, every flank
self-supporting on FDM; or a buttress with a steep loaded flank and a
45 degree back). What fails:

- the screw's core in tension and the twist of tightening;
- the teeth of either member, bent and sheared at their roots, and
  stripped along the engagement cylinder; the first turns carry more
  than their share (elastic load sharing along the engagement: screw core
  and nut wall as bars joined by the tooth springs, Sopwith's model), so
  the check uses the peak turn, not the mean;
- the flanks in bearing;
- the nut's wall burst by the flanks' wedge (a thick cylinder under the
  flank's radial push), so the boss round a nut is sized, not guessed;
- the clamp lost to creep: a plastic part held at a fixed squeeze relaxes
  (durability.clamp_retention), so the thread must start with clamp /
  retention, and the hand must be able to turn that in, and the thread
  must survive the most a hand can turn on that knob.

The radial gap between the members comes from the process (fits.HOLES:
the bias that closes a printed hole and the band each face scatters in),
and the thread's depth must keep at least half of itself at the worst
case, which sets the finest pitch a process can make.

**Hinges** (`hinge`). A pin through alternating knuckles, pinned (a
printed or bought pin, pressed into the outer knuckles and running in the
middle ones) or printed in place (the pin is part of the outer knuckles,
the gap is the process's, the bore's roof is a 45 degree teardrop so it
prints without support). What fails: the pin in double shear and in
bending (a clevis pin: load spread over each knuckle), the knuckles in
bearing, shear-out and net tension.

**Snaps** (`snap`). A cantilever with a barb (shells.snap_cantilever):
root strain against the allowance (shells.snap_allowable, measured by the
snap coupons when a card has them), the push to close it, what it holds
(the return face), the beam in tension and the ledge in shear.

Every stress is judged against the strength in its own direction: a
printed part is weaker across its layers (Material.strength_z, else the
process's interlayer share), and a shear on a plane that holds layers is
the bond's shear. `_utilization` resolves a stress state onto the layers
(normal across them, shear along them, and the rest in plane) and adds
the three quadratically (a Tsai-Hill reading for a layered solid).

Every function returns a design dict: the sizes, the drawn geometry the
B-rep builder needs (jointcad.py), `checks` (one row per failure mode:
stress, allowable, safety factor, the basis) and `assumptions` (every
number that is not a measured property of the material).

    from loadpath.joints import thread, hinge, snap
    t = thread(clamp=400, material="petg", process="fdm", knob=24)
    t["d"], t["pitch"], t["engage"], t["boss"], t["pass"]
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "HAND",
    "SAFETY",
    "check_joint",
    "design_joint",
    "hinge",
    "process_gap",
    "snap",
    "thread",
    "thread_profile",
]

SAFETY = 2.0            # target safety factor on strength (static, short term), the library's default
# An over-tightening is a one-off overload (the most a strong hand turns, on
# a thread at its smoothest), judged for yield, not for life: a lower
# target, as for any abuse load. A hand's snug turn is a third or less of
# it, so the service clamp passes SAFETY whenever this passes.
ABUSE = 1.5
RUN = 0.05              # mm of running clearance left at the worst case of the printed gap
FRICTION = 0.3          # plastic on plastic, dry (PLA/PETG/PA on themselves: 0.25-0.4 in published tables)
FRICTION_MIN = 0.2      # the low end of the same (smooth, worn in): the most a hand's torque clamps, and loosening
# The teeth's compliance, both members together (bending, shear, root
# rotation and the radial give of a tooth, Yamamoto's terms): relative flank
# displacement = C * (line load on the turn, N per mm of circumference) / E.
# C = 13 reproduces the classic result for a standard steel nut (0.8 d
# engaged, ~5.5 turns: the first turn carries about a third of the load,
# Sopwith 1948), the one calibration the sharing model needs; the peak
# share moves only as the square root of C.
TOOTH_COMPLIANCE = 13.0
# What a hand turns on a knob: the tangential force at the rim. Published
# maximum finger-twisting torques on 25 mm knobs run about 1-2.5 N m
# (Imrhan & Loo 1988 and later knob studies): 80-200 N at the rim. Snug is
# a firm turn, about half a typical user's most; the most is a strong
# user's. Rounded; not measured.
HAND = {"snug": 60.0, "max": 160.0}
# The most a hand pushes a snap shut with the thumb, N (comfortable push).
HAND_PUSH = 50.0
FDM_LAYER = 0.2         # mm, the default FDM layer (buildsim.FDM_PRINT_LAYER)
FLAT_DEPTH = 0.3        # mm a screw printed lying is flattened under its minor radius to stand on


# ---------------------------------------------------------------------------
# materials in a direction


def _material(material):
    from .materials import material_from_spec

    return material_from_spec(material[0] if isinstance(material, list) else material)


def _strengths(mat, process: str | None) -> dict:
    """In-plane and across-layer strength (MPa) and the bond's shear, for
    the process. A card's strength_z applies when it was measured on this
    process (or the process is unknown); otherwise the process's interlayer
    share. Unlayered processes and metals' own anisotropy are left out."""
    from .spec import PROCESSES

    s = float(mat.strength)
    f = float(PROCESSES.get(process, {}).get("interlayer", 1.0)) if process else 1.0
    sz = None
    if getattr(mat, "strength_z", None) and getattr(mat, "measured", None) in (None, process):
        sz = float(mat.strength_z)
    if sz is None:
        sz = s * f
    basis = (f"across layers {sz:g} MPa ({'the card' if getattr(mat, 'strength_z', None) and sz == mat.strength_z else f'x{f:g} interlayer'})"
             if sz < s else "no layer knock-down")
    return {"inPlane": s, "across": sz, "bondShear": sz / math.sqrt(3.0), "basis": basis}


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _utilization(sig: np.ndarray, st: dict, build=None) -> float:
    """Stress over strength (1.0 = failure) of a stress tensor `sig`
    (MPa, 3x3) in a layered part built along `build` (None: isotropic).

    The traction on the layer plane splits into the normal across the
    layers (judged against the across strength) and the shear along them
    (the bond's shear); the stress left in the plane of the layers is
    judged by von Mises against the in-plane strength. The three add
    quadratically."""
    sig = np.asarray(sig, float)
    s = st["inPlane"]
    if build is None or st["across"] >= s:
        dev = sig - np.trace(sig) / 3.0 * np.eye(3)
        return float(math.sqrt(1.5 * np.sum(dev * dev)) / s)
    b = _unit(build)
    t = sig @ b
    sn = float(b @ t)
    tau = float(np.linalg.norm(t - sn * b))
    P = np.eye(3) - np.outer(b, b)
    ip = P @ sig @ P
    dev = ip - np.trace(ip) / 3.0 * np.eye(3)
    vm = math.sqrt(1.5 * np.sum(dev * dev))
    # compression across layers does not open the bond: judged in plane
    across = max(sn, 0.0) / st["across"]
    comp = min(sn, 0.0) / s
    return float(math.sqrt((vm / s) ** 2 + across ** 2 + comp ** 2 + (tau / st["bondShear"]) ** 2))


def _cyl_tensor(srr=0.0, stt=0.0, szz=0.0, trz=0.0, ttz=0.0, trt=0.0, theta=0.0, axis=(0, 0, 1)):
    """A stress given in cylindrical components (r, theta, z) at angle
    theta round `axis`, as a Cartesian tensor."""
    z = _unit(axis)
    x = _unit(np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0]))
    y = np.cross(z, x)
    er = math.cos(theta) * x + math.sin(theta) * y
    et = -math.sin(theta) * x + math.cos(theta) * y
    E = np.stack([er, et, z], axis=1)
    S = np.array([[srr, trt, trz], [trt, stt, ttz], [trz, ttz, szz]], float)
    return E @ S @ E.T


def _worst_round(st, build, axis, **comp) -> float:
    """The largest utilization of a cylindrical stress state round the
    axis (it differs where the layers cut the cylinder differently)."""
    return max(_utilization(_cyl_tensor(theta=th, axis=axis, **comp), st, build)
               for th in np.linspace(0.0, math.pi, 13))


def _rupture(material, hours: float | None) -> float:
    """The share of short-term strength held for `hours` before creep
    rupture (durability.rupture_strength); 1.0 for short loads and metals."""
    if not hours:
        return 1.0
    from .durability import durability_data, rupture_strength

    return float(rupture_strength(durability_data(material), 1.0, float(hours)))


def _retention(material, durability=None) -> float:
    from .durability import clamp_retention

    return float(clamp_retention(material, durability))


def _row(id_, stress, allow, util=None, basis="", target=SAFETY, what=None, unit="MPa"):
    """One check: stress and allowable (or a utilization), safety factor
    against the target."""
    if util is None:
        util = (stress / allow) if allow else float("inf")
    sf = (1.0 / util) if util > 0 else float("inf")
    return {"id": id_, "what": what or id_, "stress": round(float(stress), 3), "allowable": round(float(allow), 3),
            "unit": unit, "sf": round(sf, 2), "target": target, "pass": bool(sf >= target - 1e-9), "basis": basis}


# ---------------------------------------------------------------------------
# the process: gaps and the finest feature


def _holes(process: str | None, holes: dict | None = None) -> dict | None:
    from .fits import HOLES

    if holes is not None:
        return holes
    return dict(HOLES[process]) if process in HOLES else None


def _along(axis, build) -> float:
    """|cos| of the angle between a joint's axis and the build direction;
    0.5 when the build is not known (between the two)."""
    if build is None or axis is None:
        return 0.5
    return abs(float(_unit(axis) @ _unit(build)))


def process_gap(process: str | None, d: float, along: float = 0.5, holes: dict | None = None,
                printed_shaft: bool = True) -> dict:
    """The radial gap (mm) a printed shaft needs in a printed (or drawn)
    bore of diameter d to turn at the worst case of the process: the
    bore closes by its bias (fits.HOLES: along or across the layers by
    `along`, |cos| of the axis to the build), the shaft grows by its own
    (pegBias on a calibrated card, else drawn size: calibration.H-PEG's
    assumption), and each scatters over its band; RSS of the two bands,
    plus RUN of running clearance. Returns {gap, bias, band, basis}."""
    from .fits import _bias

    h = _holes(process, holes)
    if h is None:
        return {"gap": RUN + 0.05, "bias": 0.0, "band": 0.0,
                "basis": "no process: a 0.1 mm gap, tolerances not checked"}
    build = None if along == 0.5 else np.array([0.0, 0.0, 1.0])
    axis = np.array([math.sqrt(max(0.0, 1 - along * along)), 0.0, along])
    bias, tol, where = _bias(h, axis, build, d)
    peg = float(h.get("pegBias", 0.0)) if printed_shaft else 0.0
    band = math.sqrt(2.0) * 0.5 * tol if printed_shaft else 0.5 * tol
    gap = 0.5 * abs(bias) + 0.5 * max(peg, 0.0) + band + RUN
    return {"gap": float(gap), "bias": float(bias), "band": float(band), "where": where,
            "basis": (f"bore closes {abs(bias):.2f} mm on the diameter ({where} the layers"
                      f"{', calibrated' if h.get('calibrated') else ', fits.HOLES design-guide figure'}), "
                      f"faces scatter +-{0.5 * tol:.3f} mm (RSS of {'both members' if printed_shaft else 'the bore'}), "
                      f"{RUN} mm left to run")}


def _min_feature(process: str | None) -> float:
    from .spec import PROCESSES

    try:
        from .dfm_reference import PROCESSES as DFM

        md = DFM.get(process, {}).get("minDetail")
    except Exception:  # noqa: BLE001
        md = None
    mf = PROCESSES.get(process, {}).get("minFeature") if process else None
    vals = [v for v in (md, mf) if v]
    return float(max(vals)) if vals else 0.5


def _self_support(process: str | None) -> float | None:
    """The steepest overhang (degrees from horizontal a downward face can
    lean) the process prints without support; None: no limit (powder)."""
    if process in ("sls", "mjf"):
        return None
    if process in ("fdm", "sla", "dlp", "lpbf", "jetting"):
        return 45.0
    return 45.0


# ---------------------------------------------------------------------------
# threads


FORMS = {
    "iso": (30.0, 30.0),       # 60 degrees included: ISO's form, least radial push
    "print": (45.0, 45.0),     # 90 degrees included: every flank self-supporting at 45
    "buttress": (10.0, 45.0),  # steep loaded flank (little radial push), 45 degree back
}


def thread_profile(d: float, pitch: float, flanks=(30.0, 30.0), gap: float = 0.25) -> dict:
    """The drawn tooth of a printed thread pair, in (axial, radial) mm.

    The screw's tooth: a crest flat P/8 at the major radius d/2, flanks at
    `flanks` (loaded, back) degrees off the radial plane, and a root flat
    wide enough that the nut's crest (the screw's tooth grown by `gap`
    all round) keeps a P/8 flat too. The nut's thread is the screw's grown
    by the gap, so every face of the pair stands `gap` apart when centred.

    Returns radii (screw major/minor, nut major/minor), depths (the screw's
    tooth, the radial overlap H1), and the widths the stress sums use."""
    bl, bb = (math.radians(float(a)) for a in flanks)
    tl, tb = math.tan(bl), math.tan(bb)
    P = float(pitch)
    c = P / 8.0
    a = P / 8.0 + gap * ((1 - math.sin(bl)) / math.cos(bl) + (1 - math.sin(bb)) / math.cos(bb))
    h = (P - c - a) / (tl + tb)
    if h <= gap:
        raise ValueError(f"pitch {P} with a {gap:.2f} mm gap leaves no thread (depth {h:.2f} mm)")
    R = d / 2.0
    rmin = R - h
    H1 = h - gap
    grow = gap * (1 / math.cos(bl) + 1 / math.cos(bb))

    def w_screw(r):
        return c + (R - r) * (tl + tb)

    def w_nut(r):
        return P - w_screw(r) - grow

    return {"d": d, "pitch": P, "flanks": [float(flanks[0]), float(flanks[1])], "gap": gap,
            "crest": c, "rootFlat": a, "depth": h, "overlap": H1,
            "screwMajor": R, "screwMinor": rmin, "nutMajor": R + gap, "nutMinor": rmin + gap,
            "pitchRadius": R - 0.5 * h, "screwTooth": w_screw, "nutTooth": w_nut,
            "screwRootWidth": w_screw(rmin), "nutRootWidth": w_nut(R + gap)}


def _share_peak(L, P, prof, E, A_screw, A_nut) -> tuple[float, float]:
    """(peak turn's load over the mean turn's, lambda L): the elastic
    load sharing along an engagement L where the screw is in tension and
    the nut in compression (Sopwith): q ~ cosh(lambda x), peak/mean =
    lambda L / tanh(lambda L)."""
    rp = prof["pitchRadius"]
    k = 2.0 * math.pi * rp * E / (TOOTH_COMPLIANCE * P)          # (N/mm per mm) per mm of slip
    lam = math.sqrt(k * (1.0 / (E * A_screw) + 1.0 / (E * A_nut)))
    x = lam * L
    return (x / math.tanh(x) if x > 1e-6 else 1.0), x


def _torque_factor(prof, mu, collar_r, mu_c=None) -> dict:
    """K (N mm of torque per N of axial force) to tighten, and the lead
    and friction angles. Square-thread lift with the loaded flank's
    wedge (mu / cos beta) plus the collar's friction."""
    rp = prof["pitchRadius"]
    lam = math.atan(prof["pitch"] / (2 * math.pi * rp))
    bl = math.radians(prof["flanks"][0])
    mup = mu / math.cos(bl)
    K = rp * (math.tan(lam) + mup) / (1 - mup * math.tan(lam)) + (mu if mu_c is None else mu_c) * collar_r
    return {"K": K, "lead": math.degrees(lam), "muFlank": mup}


def _thread_rows(prof, L, F, Fsustained, mat, st, boss_r, screw_build, nut_build, axis, rupture, mu, collar_r,
                 screw="printed", target=SAFETY) -> list:
    """Every failure mode of the thread pair at axial force F (N)."""
    P = prof["pitch"]
    E = float(mat.E)
    rows = []
    rmin, R, rs = prof["screwMinor"], prof["screwMajor"], prof["nutMinor"]
    A_core = math.pi * rmin ** 2
    Ro = boss_r if boss_r else R + max(0.5 * R, 2.0)
    A_nut = math.pi * max(Ro ** 2 - prof["nutMajor"] ** 2, 1e-6)
    peak, lamL = _share_peak(L, P, prof, E, A_core, A_nut)
    n = L / P
    Q = F / n * peak                                  # N on the most loaded turn
    tf = _torque_factor(prof, mu, collar_r)
    share = f"peak turn {peak:.2f}x the mean of {n:.1f} turns (elastic sharing, lambda L {lamL:.2f})"
    if screw == "printed" and screw_build is not None and _along(axis, screw_build) < 0.3:
        share += (f"; screw printed lying on a flat {FLAT_DEPTH} mm under its minor radius "
                  f"({100 * (1 - math.acos((rmin - FLAT_DEPTH) / R) / math.pi):.0f}% of each turn left)")

    # the screw's core: tension and the twist of tightening (thread friction only)
    if screw == "printed":
        Tth = F * (tf["K"] - mu * collar_r)
        sig = F / A_core
        tau = 2.0 * Tth / (math.pi * rmin ** 3)
        u = _worst_round(st, screw_build, axis, szz=sig, ttz=tau)
        rows.append(_row("screw-core", sig, sig / u if u else 0, util=u, target=target,
                         what="the screw's core in tension and tightening twist",
                         basis=f"F/A at the minor diameter {2 * rmin:.2f} mm, twist {tau:.1f} MPa; "
                               + ("printed standing: tension across the layers" if _along(axis, screw_build) > 0.7
                                  else "printed lying: tension along the layers" if screw_build is not None else "orientation unknown")))
        if Fsustained:
            us = _worst_round(st, screw_build, axis, szz=Fsustained / A_core) / rupture
            rows.append(_row("screw-creep", Fsustained / A_core, Fsustained / A_core / us if us else 0, util=us,
                             target=target, what="the screw's core held at the clamp for its life (creep rupture)",
                             basis=f"rupture strength {rupture:.2f}x short-term over the clamp's hours"))

    # a screw printed lying prints on a flat at its minor radius: the teeth
    # over that arc are gone, and the rest carry the turn's load
    flat = 1.0
    if screw == "printed" and screw_build is not None and _along(axis, screw_build) < 0.3:
        flat = 1.0 - math.acos((rmin - FLAT_DEPTH) / R) / math.pi
    # the teeth at their roots: bending (axial stress) and shear, each member on its own layers
    for who, wroot, r_root, arm, build in (
            ("screw", prof["screwRootWidth"], rmin, 0.5 * (rs + R) - rmin, screw_build),
            ("nut", prof["nutRootWidth"], prof["nutMajor"], prof["nutMajor"] - 0.5 * (rs + R), nut_build)):
        if who == "screw" and screw != "printed":
            continue
        q = Q / (2.0 * math.pi * r_root * flat)         # N per mm of circumference
        sb = 6.0 * q * arm / wroot ** 2
        ts = 1.5 * q / wroot
        u = _worst_round(st, build, axis, szz=sb, trz=ts)
        rows.append(_row(f"{who}-teeth", sb, sb / u if u else 0, util=u, target=target,
                         what=f"the {who}'s teeth bent and sheared at the root",
                         basis=f"{share}; root {wroot:.2f} mm wide, load {arm:.2f} mm out; "
                               + ("axial stress across the layers" if _along(axis, build) > 0.7 else "in the layers' plane"
                                  if build is not None else "orientation unknown")))
    # stripping along the engagement cylinders
    for who, r, w, build in (("screw", rs, prof["screwTooth"](rs), screw_build),
                             ("nut", R, prof["nutTooth"](R), nut_build)):
        if who == "screw" and screw != "printed":
            continue
        tau = Q / (2.0 * math.pi * r * w * flat)
        u = _worst_round(st, build, axis, trz=tau)
        rows.append(_row(f"{who}-strip", tau, tau / u if u else 0, util=u, target=target,
                         what=f"the {who}'s thread stripped along the engagement",
                         basis=f"shear on the cylinder r {r:.2f} mm, {w:.2f} mm of tooth a turn; {share}"))
    # flank bearing
    Ab = math.pi * (R ** 2 - rs ** 2)
    pb = Q / Ab
    rows.append(_row("flanks", pb, st["inPlane"] * rupture, target=target, what="the flanks in bearing",
                     basis=f"peak turn on its projected flank {Ab:.1f} mm2, compressive strength ~ tensile, "
                           f"x{rupture:.2f} held"))
    # the nut's wall burst by the flanks' wedge: thick cylinder at the bore
    bl = math.radians(prof["flanks"][0])
    rb = prof["nutMajor"]
    p = Q * math.tan(bl) / (2.0 * math.pi * prof["pitchRadius"] * P)
    if Ro > rb:
        st_h = p * (Ro ** 2 + rb ** 2) / (Ro ** 2 - rb ** 2)
        sz_n = -F / A_nut
        u = _worst_round(st, nut_build, axis, srr=-p, stt=st_h, szz=sz_n)
        rows.append(_row("nut-wall", st_h, st_h / u if u else 0, util=u, target=target,
                         what="the nut's wall burst by the flanks' wedge",
                         basis=f"radial push {math.tan(bl):.2f} Q a turn on the bore, wall to r {Ro:.2f} mm "
                               f"({'measured on the part' if boss_r else 'assumed: give boss'})"))
    return rows


def _nut_wall_needed(prof, L, F, mat, st, nut_build, axis, target) -> float:
    """The smallest outer radius round the nut that holds the burst at the target."""
    lo, hi = prof["nutMajor"] + 0.2, prof["nutMajor"] * 4.0 + 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        rows = _thread_rows(prof, L, F, None, mat, st, mid, None, nut_build, axis, 1.0, FRICTION,
                            prof["screwMajor"], screw="bought", target=target)
        ok = all(r["pass"] for r in rows if r["id"] == "nut-wall")
        if ok:
            hi = mid
        else:
            lo = mid
    return hi


def thread(clamp: float | None = None, load: float = 0.0, material="petg", process: str | None = "fdm",
           d: float | None = None, pitch: float | None = None, engage: float | None = None,
           form: str | tuple = "auto", knob: float | None = None, torque: float | None = None,
           boss: float | None = None, axis=(0, 0, 1), screw_build=None, nut_build=None,
           screw: str = "printed", collar: float | None = None, durability=None, hours: float | None = 8760.0,
           target: float = SAFETY, holes: dict | None = None, mu: float = FRICTION,
           clamp_material=None, slip: float | None = None, gap: float | None = None) -> dict:
    """Size (or check) a coarse printed thread pair.

    clamp: N the joint must still clamp with at the end of its life (a
        friction grip, a seal). The thread starts at clamp / retention
        (creep relaxation).
    load: N of external tension the joint carries on top.
    d, pitch, engage: fix any of them; the others are searched (the
        smallest that passes, by the screw's volume).
    form: "iso", "print", "buttress", (loaded, back) flank angles, or
        "auto" (by the members' print orientation: the flanks that face
        down while printing kept within the process's self-support).
    knob: the knob's diameter (mm) when it is turned by hand: the snug and
        most torque follow (HAND). "auto": the knob is sized too, the
        smallest (in 2 mm steps) whose snug turn gives the fresh clamp.
        torque: N mm, when a tool sets it.
    clamp_material: what else is squeezed in the clamp (a TPU liner, a
        gasket): the clamp relaxes as the most creeping of the stack does.
    slip: a knob that slips past this multiple of the snug torque (a
        torque-limiting clutch): the most a hand can then put in. Without
        it a strong hand on a smooth thread clamps several times the snug
        clamp, and that over-tightening sizes the joint.
    boss: the nut's outer diameter (mm). None: the wall it needs is sized.
    axis, screw_build, nut_build: the thread's axis and each member's
        build direction, in one frame (None: orientation unknown).
    screw: "printed" or "bought" (a metal screw in a printed nut).
    collar: the mean radius the head bears on (mm; default 0.65 d: a
        bearing ring from the clearance hole to about 1.5 d across).
    hours: how long the clamp is held (creep rupture of the core).
    gap: the radial gap as drawn (default: the process's, process_gap).
    """
    mat = _material(material)
    st = _strengths(mat, process)
    axis = _unit(axis)
    ret = _retention(material, durability) if clamp else 1.0
    if clamp and clamp_material is not None:
        ret = min(ret, _retention(clamp_material, durability))
    rup = _rupture(material, hours)
    assumptions = [f"friction {mu} plastic on plastic (dry, published range 0.25-0.4); loosening judged at "
                   f"{FRICTION_MIN}",
                   f"tooth compliance {TOOTH_COMPLIANCE:g} (calibrated to a steel nut's first turn carrying a third) for the turns' load sharing"]
    auto_knob = isinstance(knob, str) and knob == "auto"
    if knob:
        assumptions.append(f"hand torque: snug {HAND['snug']:.0f} N and most {HAND['max']:.0f} N at the knob's rim "
                           "(published knob-torque ranges, rounded)")

    al_s, al_n = _along(axis, screw_build), _along(axis, nut_build)
    sup = _self_support(process)
    if isinstance(form, str) and form != "auto":
        forms = [(form, FORMS[form])]
    elif isinstance(form, (list, tuple)):
        forms = [("custom", (float(form[0]), float(form[1])))]
    else:
        # the flanks of a member printed with its axis along the build face
        # down at their angle off the radial plane: past the process's
        # self-support they droop. ISO where nothing prints standing.
        standing = sup is not None and max(al_s if screw == "printed" else 0.0, al_n) > 0.7
        forms = [("print", FORMS["print"]), ("buttress", FORMS["buttress"])] if standing else [("iso", FORMS["iso"])]

    Fneed = (clamp or 0.0) / ret

    def design(dd, P, L, fname, fl):
        # the nut is the hole: its orientation sets how far it closes
        g = process_gap(process, dd, al_n, holes, printed_shaft=(screw == "printed"))
        if gap is not None:
            g = {**g, "gap": float(gap), "basis": f"{gap} mm as drawn; " + g["basis"]}
        try:
            prof = thread_profile(dd, P, fl, g["gap"])
        except ValueError:
            return None
        cr = collar if collar else 0.65 * dd
        tf = _torque_factor(prof, mu, cr)
        tf_lo = _torque_factor(prof, FRICTION_MIN, cr)
        kn = knob
        if auto_knob:
            kn = max(16.0, 2.0 * math.ceil(tf["K"] * Fneed / HAND["snug"])) if clamp else 2.0 * dd
            kn = max(kn, dd + 6.0)
        if torque:
            T_snug, T_max = float(torque), 1.25 * float(torque)
        elif kn:
            T_snug, T_max = HAND["snug"] * kn / 2.0, HAND["max"] * kn / 2.0
            if slip:
                T_max = min(T_max, float(slip) * T_snug)
        else:
            T_snug = T_max = None
        F0 = Fneed if T_snug is None else max(Fneed, T_snug / tf["K"])
        # the most the thread sees: the hand's most on a lubricated thread, or 1.25x the set preload
        Fmax = (T_max / tf_lo["K"]) if T_max else 1.25 * max(Fneed, 1e-9)
        Fmax = max(Fmax, Fneed) + load
        hand = T_max is not None and not torque
        tgt = min(target, ABUSE) if hand else target
        rows = _thread_rows(prof, L, Fmax, Fneed if clamp else None, mat, st, (boss / 2.0) if boss else None,
                            screw_build, nut_build, axis, rup, mu, cr, screw=screw, target=tgt)
        for r in rows:
            if r["id"] == "screw-creep":
                r["target"] = target
                r["pass"] = bool(r["sf"] >= target - 1e-9)
        # tolerance: half the overlap kept at the worst case (each face off by its band)
        band = g["band"]
        keep = (prof["overlap"] - 2.0 * band) / prof["overlap"]
        rows.append({"id": "engagement", "what": "the thread's depth left at the worst-case tolerance",
                     "stress": round(keep, 3), "allowable": 0.5, "unit": "share", "sf": round(keep / 0.5, 2),
                     "target": 1.0, "pass": keep >= 0.5,
                     "basis": f"overlap {prof['overlap']:.2f} mm less both faces' band ({band:.3f} mm each)"})
        mf = _min_feature(process)
        wt = min(prof["screwRootWidth"], prof["nutRootWidth"])
        rows.append({"id": "tooth-print", "what": "the narrowest tooth against the process's finest detail",
                     "stress": round(wt, 3), "allowable": mf, "unit": "mm", "sf": round(wt / mf, 2), "target": 1.0,
                     "pass": wt >= mf, "basis": "root width of the narrower member's tooth"})
        if process == "fdm" and max(al_s, al_n) > 0.7:
            rows.append({"id": "layers", "what": "the crest flat in layers (a member printed standing)",
                         "stress": round(prof["crest"], 3), "allowable": FDM_LAYER, "unit": "mm",
                         "sf": round(prof["crest"] / FDM_LAYER, 2), "target": 1.0,
                         "pass": prof["crest"] >= FDM_LAYER, "basis": f"crest P/8 against a {FDM_LAYER} mm layer"})
        if T_snug is not None and clamp:
            rows.append({"id": "torque", "what": "the clamp a snug turn gives against the clamp needed fresh",
                         "stress": round(T_snug / tf["K"], 1), "allowable": round(Fneed, 1), "unit": "N",
                         "sf": round(T_snug / tf["K"] / Fneed, 2) if Fneed else None, "target": 1.0,
                         "pass": T_snug / tf["K"] >= Fneed - 1e-9,
                         "basis": f"K {tf['K']:.2f} N mm per N (lead {tf['lead']:.1f} deg, flank friction "
                                  f"{tf['muFlank']:.2f}, collar r {cr:.1f}); clamp {clamp:.0f} N / retention {ret:.2f}"})
        if sup is not None:
            for who, al in (("screw", al_s if screw == "printed" else 0.0), ("nut", al_n)):
                if al > 0.7:
                    # one flank of each member faces down while it prints standing
                    worst = min(fl)
                    rows.append({"id": f"{who}-overhang", "what": f"the {who}'s down-facing flank while printing",
                                 "stress": round(90.0 - worst, 1), "allowable": 90.0 - sup, "unit": "deg off vertical",
                                 "sf": None, "target": None, "pass": worst >= sup - 1e-9,
                                 "basis": f"flanks {fl[0]:g}/{fl[1]:g} deg off the radial plane, printed standing; "
                                          f"self-supports to {sup:g} deg"})
        lock = math.tan(math.radians(tf["lead"])) < tf_lo["muFlank"]
        ok = all(r["pass"] for r in rows)
        strong = all(r["pass"] for r in rows if r["id"] != "torque")
        vol = math.pi * (dd / 2) ** 2 * L
        return {"type": "thread", "form": fname, "d": dd, "pitch": P, "engage": L, "flanks": list(fl),
                "gap": round(g["gap"], 3), "gapBasis": g["basis"], "profile": {k: (round(v, 4) if isinstance(v, float) else v)
                                                                             for k, v in prof.items() if not callable(v)},
                "boss": boss, "bossSized": not bool(boss), "_F": Fmax, "_prof": prof, "_strong": strong, "_tgt": tgt,
                "preload": {"clamp": clamp, "retention": round(ret, 3), "fresh": round(Fneed, 1),
                            "snug": round(F0, 1), "most": round(Fmax, 1),
                            "judgedAt": ("the most a hand turns on a smooth thread (abuse, target "
                                         f"{tgt:g})" if hand else f"1.25x the set clamp (target {tgt:g})")},
                "knob": kn, "torque": {"K": round(tf["K"], 3), "snug": T_snug, "most": T_max,
                           "needed": round(tf["K"] * Fneed, 1) if clamp else None,
                           "knobNeeded": round(2.0 * tf["K"] * Fneed / HAND["snug"], 1) if clamp else None,
                           "lead": round(tf["lead"], 2), "selfLocking": bool(lock)},
                "checks": rows, "pass": ok, "_vol": vol,
                "minSF": min((r["sf"] for r in rows if r.get("sf") is not None and r.get("unit") == "MPa"),
                             default=None)}

    ds = [float(d)] if d else [5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 30.0]
    best = None
    tried = 0
    for dd in ds:
        Ps = [float(pitch)] if pitch else [p for p in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0) if p <= dd / 3.0]
        Ls = [float(engage)] if engage else [m * dd for m in (0.8, 1.0, 1.5, 2.0)]
        for P in Ps:
            for L in Ls:
                for fname, fl in forms:
                    r = design(dd, P, L, fname, fl)
                    tried += 1
                    if r is None:
                        continue
                    # the knob's torque is the user's to change: strength decides the size
                    key = (not r["_strong"], not r["pass"], r["_vol"] if r["_strong"] else -(r["minSF"] or 0),
                           -(r["minSF"] or 0))
                    if best is None or key < best[0]:
                        best = (key, r)
        if best and best[1]["pass"] and not d:
            break
    if best is None:
        raise ValueError("no thread could be drawn for these sizes and this process's gap")
    out = best[1]
    if not boss:
        need = _nut_wall_needed(out["_prof"], out["engage"], out["_F"], mat, st, nut_build, axis, out["_tgt"])
        out["boss"] = round(2.0 * need, 2)
    for k in ("_vol", "_F", "_prof", "_strong", "_tgt"):
        out.pop(k, None)
    out["material"] = mat.name
    out["process"] = process
    out["strengths"] = st
    out["assumptions"] = assumptions
    out["searched"] = tried
    out["inputs"] = {"clamp": clamp, "load": load, "knob": knob, "torque": torque, "screw": screw,
                     "axis": axis.tolist(), "screwBuild": None if screw_build is None else list(map(float, screw_build)),
                     "nutBuild": None if nut_build is None else list(map(float, nut_build)), "hours": hours,
                     "target": target}
    return out


# ---------------------------------------------------------------------------
# hinges


def hinge(load: float, material="petg", process: str | None = "fdm", kind: str = "pinned",
          d: float | None = None, width: float | None = None, knuckles: int = 3, ligament: float | None = None,
          gap: float | None = None, pin=None, load_dir=(0, 0, 1), axis=(1, 0, 0), build=None, pin_build=None,
          hours: float | None = None, target: float = SAFETY, holes: dict | None = None) -> dict:
    """Size (or check) a hinge: a pin through `knuckles` alternating
    knuckles (odd: the outer pair and every other one on one member).

    load: N across the pin (the hinge's reaction), along `load_dir`.
    kind: "pinned" (a separate pin: printed, or `pin` a bought one's
        material, e.g. "304") or "print-in-place" (the pin is one member's,
        printed through the other's bore with the process's gap).
    d: pin diameter; width: the hinge's length along the pin; ligament:
        the knuckle's wall round the bore. Any left out are sized.
    axis, build, pin_build: the pin's axis and the build directions of the
        knuckles' part and of a printed pin (None: unknown).
    hours: how long the load is held (creep rupture), None for short loads.
    """
    if knuckles < 2:
        raise ValueError("a hinge has at least two knuckles (one on each member)")
    mat = _material(material)
    st = _strengths(mat, process)
    if kind == "print-in-place":
        pmat, pst, pproc = mat, st, process
        pin_build = build
    elif pin is not None:
        pmat = _material(pin)
        pproc = None if float(pmat.density) > 2.0 else process
        pst = _strengths(pmat, pproc)
    else:
        pmat, pst, pproc = mat, st, process
    axis = _unit(axis)
    ld = _unit(load_dir)
    ld = _unit(ld - (ld @ axis) * axis) if np.linalg.norm(ld - (ld @ axis) * axis) > 1e-6 else _unit(np.cross(axis, [0, 0, 1]))
    rup = _rupture(material, hours)
    rup_pin = _rupture(pin if pin is not None and kind != "print-in-place" else material, hours)
    al = _along(axis, build)
    sup = _self_support(process)
    n_in = knuckles // 2                     # the middle member's knuckles
    n_out = knuckles - n_in
    assumptions = ["clevis pin: each knuckle's share spread evenly over its width",
                   "lug shear-out on two planes from the bore to the edge; net tension with no Kt (ductile)"]

    teardrop = kind == "print-in-place" and sup is not None and al < 0.3

    def rows_for(dd, W, lig):
        g = gap if gap is not None else process_gap(process, dd, al, holes,
                                                    printed_shaft=(kind == "print-in-place" or pin is None))["gap"]
        if kind == "pinned":
            # the pin presses into the outer knuckles and runs in the inner ones
            g_run = g
        else:
            g_run = g
        e_ax = g if kind == "print-in-place" else max(0.2, 0.5 * g)   # axial gap between knuckles
        # each knuckle as wide as its share of the load: the middle member's
        # carry F / n_in each, the outer member's F / n_out (a clevis: the
        # middle knuckle twice an outer one)
        Weff = W - (knuckles - 1) * e_ax
        t_in, t_out = Weff / (2.0 * n_in), Weff / (2.0 * n_out)
        t = min(t_in, t_out)
        if t <= 0.5:
            return None
        rows = []
        F = float(load)
        # pin: each middle knuckle's load leaves it through the two planes beside it
        planes = knuckles - 1
        Vp = F / n_in / 2.0
        A = math.pi * dd ** 2 / 4.0
        tau = 4.0 / 3.0 * Vp / A
        S = np.outer(axis, ld) + np.outer(ld, axis)
        u = _utilization(tau * S, pst, pin_build) / rup_pin
        rows.append(_row("pin-shear", tau, tau / u if u else 0, util=u, target=target, what="the pin in shear",
                         basis=f"{planes} shear planes, peak 4/3 V/A; "
                               + ("load along the layers' bond" if pin_build is not None and _along(ld, pin_build) > 0.7
                                  or pin_build is not None and _along(axis, pin_build) > 0.7 else
                                  "bought pin" if pin is not None and kind != "print-in-place" else "shear in the layers' plane"
                                  if pin_build is not None else "orientation unknown")))
        # pin bending: a clevis round each middle knuckle (F / n_in spread over
        # it, half of it taken out over each neighbour's half width)
        Fi = F / n_in
        M = Fi / 8.0 * (2.0 * t_out + 4.0 * e_ax + t_in)
        sb = 32.0 * M / (math.pi * dd ** 3)
        u = _utilization(sb * np.outer(axis, axis), pst, pin_build) / rup_pin
        rows.append(_row("pin-bending", sb, sb / u if u else 0, util=u, target=target, what="the pin in bending",
                         basis=f"clevis moment F/8 (2a + 4g + b) = {M:.0f} N mm; "
                               + ("printed standing: bending across the layers" if pin_build is not None and _along(axis, pin_build) > 0.7
                                  else "along the pin's layers" if pin_build is not None else "")))
        # knuckles: bearing, shear-out, net tension; the inner ones carry F over n_in, the outer F over n_out
        for who, nk, t in (("inner", n_in, t_in), ("outer", n_out, t_out)):
            Fk = F / nk
            pb = Fk / (dd * t)
            rows.append(_row(f"{who}-bearing", pb, st["inPlane"] * rup, target=target,
                             what=f"the {who} knuckles in bearing on the pin",
                             basis=f"{Fk:.0f} N on {dd:.2f} x {t:.2f} mm"))
            R = 0.5 * dd + g_run + lig
            ts = Fk / (2.0 * t * max(R - 0.5 * dd - g_run, 1e-3))
            # shear-out: planes along the load, normal across the pin axis and the load
            nrm = _unit(np.cross(axis, ld))
            S2 = np.outer(nrm, ld) + np.outer(ld, nrm)
            u = _utilization(ts * S2, st, build) / rup
            rows.append(_row(f"{who}-shearout", ts, ts / u if u else 0, util=u, target=target,
                             what=f"the {who} knuckles torn out ahead of the pin",
                             basis=f"two planes {lig:.2f} mm long, {t:.2f} mm wide"))
            sn = Fk / (2.0 * lig * t)
            u = _utilization(sn * np.outer(ld, ld), st, build) / rup
            rows.append(_row(f"{who}-tension", sn, sn / u if u else 0, util=u, target=target,
                             what=f"the {who} knuckles in tension beside the bore",
                             basis=f"two ligaments {lig:.2f} x {t:.2f} mm"))
        mf = _min_feature(process)
        rows.append({"id": "ligament-print", "what": "the knuckle's wall against the process's finest detail",
                     "stress": round(lig, 3), "allowable": mf, "unit": "mm", "sf": round(lig / mf, 2), "target": 1.0,
                     "pass": lig >= mf, "basis": "bore to the knuckle's edge"})
        if teardrop:
            rb = 0.5 * dd + g_run
            roof = (rb + lig) - rb * math.sqrt(2.0)
            rows.append({"id": "bore-roof", "what": "the knuckle's wall over the bore's 45 degree teardrop roof",
                         "stress": round(roof, 3), "allowable": mf, "unit": "mm", "sf": round(roof / mf, 2),
                         "target": 1.0, "pass": roof >= mf - 1e-9,
                         "basis": "pin across the build: the bore's top is drawn as a teardrop so it prints "
                                  "unsupported; the pin's underside bridges between its outer knuckles"})
            from .printcheck import BRIDGE_SPAN

            span = t_in + 2 * e_ax
            lim = BRIDGE_SPAN.get(process)
            if lim:
                rows.append({"id": "pin-bridge", "what": "the pin's underside bridged across the middle knuckle",
                             "stress": round(span, 2), "allowable": lim, "unit": "mm", "sf": round(lim / span, 2),
                             "target": 1.0, "pass": span <= lim, "basis": "printcheck.BRIDGE_SPAN"})
        return {"rows": rows, "gap": g, "axialGap": e_ax, "knuckle": t_in, "outerKnuckle": t_out}

    ds = [float(d)] if d else [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    Ws = [float(width)] if width else None
    best = None
    for dd in ds:
        # a teardrop's roof reaches sqrt(2) of the bore's radius: the wall over it must still print
        lo = _min_feature(process) + ((math.sqrt(2.0) - 1.0) * 0.5 * (dd + 0.6) if teardrop else 0.0)
        ligs = [float(ligament)] if ligament else sorted({max(lo, m * dd) for m in (0.4, 0.5, 0.75, 1.0)})
        for lig in ligs:
            for W in (Ws or [m * dd for m in (2.0, 3.0, 4.0, 6.0, 8.0)]):
                r = rows_for(dd, W, lig)
                if r is None:
                    continue
                ok = all(x["pass"] for x in r["rows"])
                msf = min((x["sf"] for x in r["rows"] if x.get("unit") == "MPa"), default=0)
                vol = (dd / 2 + lig) ** 2 * W
                key = (not ok, vol if ok else -msf)
                if best is None or key < best[0]:
                    best = (key, dd, W, lig, r, ok, msf)
        if best and best[5] and not d:
            break
    if best is None:
        raise ValueError("no hinge fits: widen it or allow a thicker pin")
    _, dd, W, lig, r, ok, msf = best
    g = r["gap"]
    out = {"type": "hinge", "kind": kind, "d": dd, "width": W, "knuckles": knuckles, "knuckle": round(r["knuckle"], 3), "outerKnuckle": round(r["outerKnuckle"], 3),
           "ligament": lig, "gap": round(g, 3), "axialGap": round(r["axialGap"], 3),
           "bore": round(dd + 2 * g, 3), "outer": round(dd + 2 * g + 2 * lig, 3),
           "checks": r["rows"], "pass": ok, "minSF": msf, "material": mat.name, "process": process,
           "pin": (pin if pin is not None and kind != "print-in-place" else "printed"), "assumptions": assumptions,
           "inputs": {"load": load, "loadDir": ld.tolist(), "axis": axis.tolist(),
                      "build": None if build is None else list(map(float, build)), "hours": hours, "target": target}}
    if kind == "pinned":
        # the pin is pressed into the outer knuckles: drawn under the pin by the process's bias + interference
        from .fits import _bias, fit_window

        h = _holes(process, holes)
        win = fit_window("press", dd, "plastic")
        bias = _bias(h, axis, None if build is None else _unit(build), dd)[0] if h else 0.0
        out["press"] = round(win["target"] - bias, 3)
        out["pressBasis"] = f"{win['standard']} target {win['target']:.3f} mm, drawn {-bias:+.2f} for the process's bias"
    return out


# ---------------------------------------------------------------------------
# snaps


def snap(hold: float, material="petg", process: str | None = "fdm", undercut: float | None = None,
         t: float | None = None, L: float | None = None, b: float | None = None, lead: float = 30.0,
         ret: float = 90.0, mu: float = FRICTION, length_dir=(0, 0, 1), deflect_dir=(1, 0, 0), build=None,
         repeated: bool = False, target: float = SAFETY, holes: dict | None = None) -> dict:
    """Size (or check) a cantilever snap that must hold `hold` N.

    The undercut is what the barb must catch past the process's worst
    case (tolerance), the strain at the root sets the length for a
    thickness, and the width is what holds `hold` through the return face
    (or, with a 90 degree return, the beam's tension and the ledge's
    shear). length_dir / deflect_dir: the beam's length and the way it
    bends; build the part's build direction."""
    from .shells import snap_allowable, snap_cantilever

    mat = _material(material)
    st = _strengths(mat, process)
    across = build is not None and _along(length_dir, build) > 0.7
    allow = snap_allowable(material, process, across_layers=across, repeated=repeated)
    eps = float(allow["strain"])
    h = _holes(process, holes)
    band = 0.0
    if h:
        from .fits import _bias

        band = math.sqrt(2.0) * 0.5 * _bias(h, _unit(deflect_dir), None if build is None else _unit(build), 5.0)[1]
    catch = max(0.35, _min_feature(process) * 0.5)
    y = float(undercut) if undercut else catch + 2.0 * band
    mf = _min_feature(process)
    best = None
    for tt in ([float(t)] if t else [max(mf, x) for x in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)]):
        LL = float(L) if L else math.ceil(math.sqrt(1.5 * tt * y / eps) * 2.0) / 2.0
        for bb in ([float(b)] if b else [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]):
            sc = snap_cantilever(LL, tt, bb, y, float(mat.E), lead, ret, mu)
            rows = [{"id": "strain", "what": "the root's strain as the barb passes", "stress": round(sc["strain"], 5),
                     "allowable": round(eps, 5), "unit": "strain", "sf": round(eps / sc["strain"], 2), "target": 1.0,
                     "pass": sc["strain"] <= eps + 1e-9, "basis": allow["basis"]}]
            pull = sc["pull"]
            # holding: the return face lets go at `pull`; a 90 degree face holds until something breaks
            sig = hold / (bb * tt)
            u = _utilization(sig * np.outer(_unit(length_dir), _unit(length_dir)), st, build)
            rows.append(_row("beam-tension", sig, sig / u if u else 0, util=u, target=target,
                             what="the beam pulled by what it holds",
                             basis="across the layers" if across else "along the layers" if build is not None else ""))
            ledge = max(y, 1e-3)
            tau = hold / (bb * max(tt, ledge))
            dl = _unit(length_dir)
            ddf = _unit(deflect_dir)
            u = _utilization(tau * (np.outer(dl, ddf) + np.outer(ddf, dl)), st, build)
            rows.append(_row("barb-shear", tau, tau / u if u else 0, util=u, target=target,
                             what="the barb sheared off by what it holds",
                             basis=f"{bb:.1f} x {max(tt, ledge):.2f} mm"))
            if pull is not None:
                rows.append({"id": "hold", "what": "the pull the return face holds before it lets go",
                             "stress": round(pull, 2), "allowable": hold, "unit": "N", "sf": round(pull / hold, 2),
                             "target": 1.0, "pass": pull >= hold, "basis": f"return face {ret:g} deg, friction {mu}"})
            push = sc["push"] or 0.0
            rows.append({"id": "push", "what": "the push to close it", "stress": round(push, 2), "allowable": HAND_PUSH,
                         "unit": "N", "sf": round(HAND_PUSH / push, 2) if push else None, "target": 1.0,
                         "pass": push <= HAND_PUSH, "basis": f"lead face {lead:g} deg; a thumb's comfortable push"})
            ok = all(r["pass"] for r in rows)
            vol = LL * tt * bb
            key = (not ok, vol)
            if best is None or key < best[0]:
                best = (key, tt, LL, bb, sc, rows, ok)
    _, tt, LL, bb, sc, rows, ok = best
    return {"type": "snap", "t": tt, "L": LL, "b": bb, "undercut": round(y, 3), "lead": lead, "return": ret,
            "strain": round(sc["strain"], 5), "allowable": round(eps, 5), "push": sc["push"], "pull": sc["pull"],
            "checks": rows, "pass": ok, "material": mat.name, "process": process,
            "assumptions": [f"catch {catch:.2f} mm past the worst case (band {band:.3f} mm a face)",
                            f"push limit {HAND_PUSH:.0f} N (a thumb, comfortable)"],
            "inputs": {"hold": hold, "lengthDir": list(map(float, length_dir)),
                       "deflectDir": list(map(float, deflect_dir)),
                       "build": None if build is None else list(map(float, build)), "repeated": repeated}}


# ---------------------------------------------------------------------------
# from a spec entry


def design_joint(j: dict, material=None, process=None, build=None) -> dict:
    """Size the joint a `joints` entry describes (schema $.joints): the
    keys of thread/hinge/snap by their camelCase names. material/process/
    build: the part's, when the entry leaves them out."""
    kind = j.get("type")
    mat = j.get("material", material) or "petg"
    proc = j.get("process", process)
    if isinstance(proc, dict):
        build = proc.get("buildDirection", build)
        proc = proc.get("type")
    b = j.get("build", build)
    if kind == "thread":
        return thread(clamp=j.get("clamp"), load=float(j.get("load", 0.0)), material=mat, process=proc,
                      d=j.get("d"), pitch=j.get("pitch"), engage=j.get("engage"), form=j.get("form", "auto"),
                      knob=j.get("knob"), torque=j.get("torque"), boss=j.get("boss"), axis=j.get("axis", (0, 0, 1)),
                      screw_build=j.get("screwBuild"), nut_build=b, screw=j.get("screw", "printed"),
                      collar=j.get("collar"), hours=j.get("hours", 8760.0), target=float(j.get("safety", SAFETY)),
                      clamp_material=j.get("clampMaterial"), slip=j.get("slip"), gap=j.get("gap"))
    if kind == "hinge":
        return hinge(load=float(j["load"]), material=mat, process=proc, kind=j.get("kind", "pinned"),
                     d=j.get("d"), width=j.get("width"), knuckles=int(j.get("knuckles", 3)),
                     ligament=j.get("ligament"), gap=j.get("gap"), pin=j.get("pin"),
                     load_dir=j.get("loadDir", (0, 0, 1)), axis=j.get("axis", (1, 0, 0)), build=b,
                     pin_build=j.get("pinBuild"), hours=j.get("hours"), target=float(j.get("safety", SAFETY)))
    if kind == "snap":
        return snap(hold=float(j["hold"]), material=mat, process=proc, undercut=j.get("undercut"), t=j.get("t"),
                    L=j.get("L"), b=j.get("b"), lead=float(j.get("lead", 30.0)), ret=float(j.get("return", 90.0)),
                    length_dir=j.get("lengthDir", (0, 0, 1)), deflect_dir=j.get("deflectDir", (1, 0, 0)), build=b,
                    repeated=bool(j.get("repeated", False)), target=float(j.get("safety", SAFETY)))
    raise ValueError(f"joint type is 'thread', 'hinge' or 'snap', got {kind!r}")


def check_joint(j: dict, measured: dict, material=None, process=None, build=None) -> dict:
    """Re-run a joint's checks at the geometry measured on a delivered
    part: `measured` overrides the entry's sizes (a thread's `boss` read
    as the wall round the nut, a hinge's `ligament` read round its bore).
    The sizes themselves are fixed to the design's, so nothing is
    re-searched."""
    des = design_joint(j, material, process, build)
    fixed = dict(j)
    if des["type"] == "thread":
        fixed.update(d=des["d"], pitch=des["pitch"], engage=des["engage"], form=des["flanks"])
        if fixed.get("knob") == "auto":
            fixed["knob"] = des["knob"]
    elif des["type"] == "hinge":
        fixed.update(d=des["d"], width=des["width"], ligament=des["ligament"])
    else:
        fixed.update(t=des["t"], L=des["L"], b=des["b"], undercut=des["undercut"])
    fixed.update({k: v for k, v in measured.items() if v is not None})
    return design_joint(fixed, material, process, build)
