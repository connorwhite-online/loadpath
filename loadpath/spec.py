"""
The study: everything a topology optimization needs, as one JSON document.

Units are fixed so nobody has to guess: millimetres, newtons, megapascals
(N/mm^2), grams per cubic centimetre. A study an agent writes looks like:

    {
      "name": "arm-link",
      "designSpace": [{"type": "box", "min": [0, -20, -12], "max": [160, 20, 12]}],
      "preserve": [
        {"id": "shoulder", "shape": {"type": "cylinder", "center": [0, 0, 0],
                                     "axis": "z", "length": 24, "r": 16}}
      ],
      "keepOut": [
        {"id": "shoulder-bore", "shape": {"type": "cylinder", "center": [0, 0, 0],
                                          "axis": "z", "length": 30, "r": 11}}
      ],
      "supports": [{"region": "shoulder-bore"}],
      "loadCases": [
        {"name": "payload", "loads": [{"region": "wrist-bore", "force": [0, 0, -40]}]}
      ],
      "material": "pa12",
      "goal": {"volumeFraction": 0.25},
      "minMember": 3.0,
      "symmetry": ["z"],
      "resolution": {"maxElements": 80000}
    }

A load or support names a region: a preserve or keep-out id, or an inline
shape. Loading a keep-out (a bore) means "through whatever material lines it",
which is how a bolt or a shaft actually pushes on a part.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .am import parse_build_direction
from .materials import Material, material_from_spec
from .shapes import Shape, shape_from_spec

__all__ = ["Region", "Load", "LoadCase", "Support", "Goal", "Study", "study_from_dict"]

DEFAULT_MAX_ELEMENTS = 60_000
# mm: the member size the filter holds a study to when it names neither a
# minMember nor a process: about three FDM perimeters, and under FDM's own
# floor (2 x its 0.8 mm minimum feature) on purpose. At 1.6 the filter grew
# on grids coarser than a millimetre too (the mbb-beam eval, 1.29 mm cells,
# went from passing at SF 1.52 to a search that stalled at 1.49); at 1.4 it
# only acts below ~1.2 mm cells, where the sub-millimetre skins come from.
DEFAULT_MIN_MEMBER = 1.4
# Hard ceiling regardless of what a study asks for. From 100k cells the
# optimizer is matrix-free multigrid (mg.py), ~1.5 KB per cell: an 845k-cell
# study ran end to end in 20 min and 1.6 GB on a 4-core box. Past a few
# million the meshing and verification steps, not the solver, set the limit.
MAX_ELEMENTS_CEILING = 4_000_000


@dataclass
class Clearance:
    """goal.clearances[i]: no node of the region (or inline shape) moves
    more than `max_travel` mm, along `direction` (a unit vector: travel
    toward what is there) or in any direction (None), in `cases` (names;
    empty: every case)."""
    region: str | None
    shape: object
    max_travel: float
    direction: np.ndarray | None = None
    cases: list = field(default_factory=list)
    name: str = ""


@dataclass
class Region:
    id: str
    shape: Shape
    # keepOut only: false for a clearance volume (a drive, a spring's
    # swept space) that may end against material, unlike a bore a screw or
    # pin has to pass through (verify's closed-bore check skips it)
    bore: bool = True
    # keepOut only: mm of material kept round it, pinned solid (a duct's or
    # a plenum's wall; see pressure.py). Set for every pressurized keep-out.
    wall: float | None = None
    # keepOut only: whether it cuts preserved geometry. A bore does (a hole
    # through a boss), and so does a part's way in (insertFrom: a capsule's
    # path through its cup's floor); a clearance zone does not: it keeps the
    # optimizer's material out, never what the designer drew (the touch
    # microphone's board clear zone deleted 88% of its rails). Only a bore
    # cuts the skin: the exterior is the designer's, and a way in through
    # it is one of the skin's openings (the same part's plug path cut a
    # square hole where its USB-C port is a stadium). `cutsPreserve` sets
    # it either way; see cuts_of.
    cuts: bool = True


@dataclass
class Load:
    region: str | None
    shape: Shape | None
    force: np.ndarray  # N, total over the region
    # N·mm about the region's centroid (a motor's torque on a hub).
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # "uniform", or "bearing": a shaft pushing on one side of a bore, with
    # the force spread as a cosine over the half facing it.
    distribution: str = "uniform"
    # MPa gauge in the keep-out `region` names, on the material lining it
    # (pressure.py); 0 = no pressure. Negative is a vacuum.
    pressure: float = 0.0
    # The force's direction is not known: None (as written), "any" (its size
    # in every direction) or "cone" (within `cone` degrees of it). The
    # optimizer and verification design for the worst one (directions.py).
    direction: str | None = None
    cone: float | None = None


@dataclass
class LoadCase:
    name: str
    loads: list[Load]
    weight: float = 1.0
    # durability (durability.py): times applied over the part's life
    # (fatigue), hours held (creep), and the temperature then (C)
    cycles: float | None = None
    hours: float | None = None
    temperature: float | None = None
    # this case's own safety factor (a crash that may use most of the
    # strength, beside service loads held to the goal's): None takes the goal's
    safety_factor: float | None = None
    # an impact (ImpactCase): the loads give where and which way it hits,
    # their size comes from the energy, the crusher and the part's own give
    impact: object = None


@dataclass
class ImpactCase:
    """A load case that is a hit, not a force: `energy` (N mm) to stop, or
    `velocity` (m/s) with `mass` (kg; None on a free part: the mass the
    body presents at the patch along the hit, from its own inertia), through
    a crusher whose force-crush curve is `distance` (mm), `force` (N) (None:
    nothing in between). impact.series_impact puts the curve in series with
    the part's compliance at the patch."""
    energy: float | None = None
    velocity: float | None = None
    mass: float | None = None
    distance: np.ndarray | None = None
    force: np.ndarray | None = None


@dataclass
class Support:
    region: str | None
    shape: Shape | None
    fix: tuple[bool, bool, bool] = (True, True, True)
    # A pin through a bore: held radially and axially, free to rotate about
    # the bore's axis (a bolt or a shaft in a plain hole).
    pin: bool = False
    # the most this support may carry (N, the resultant of its reaction in
    # any case): a bolt's or a bearing's rating. Reported for every support
    # (verify.support_reactions); with a cap it is held like the safety factor.
    max_force: float | None = None
    max_moment: float | None = None


@dataclass
class Vibration:
    """The study's `vibration` block: the delivered part's first natural
    frequency (clamped at the supports, carrying `masses`) must reach
    `min_frequency` Hz (None: no floor), and no mode may sit where an
    excitation drives it. `masses`: [(region id or None, shape or None, kg)].
    `excitations`: [(name, lo Hz, hi Hz, max amplification or None)], each a
    forcing that sweeps a band (a rotor's once-per-rev and blade-pass lines
    over the throttle range); see modal.band_response.
    `in_optimizer`: the floor and the bands are held in the optimizer on the
    design's own modes (modalopt.ModalModel), and for a held part the masses
    shaken along x, y and z become extra, stiffness-only load cases
    (domain.inertial_cases)."""
    min_frequency: float | None
    masses: list = field(default_factory=list)
    modes: int = 3
    in_optimizer: bool = True
    excitations: list = field(default_factory=list)


@dataclass
class Rotor:
    """A rotor or fan whose stream the part sits in (aero.RotorStream): the
    disk's `center` (mm), unit `axis` pointing the way the air goes,
    `radius` and `hub` (mm), `thrust` (N) at the operating point the cost is
    weighed at, drag coefficient `cd` of what the stream meets, air density
    `rho` (kg/m^3), `ducted`, and how far along the axis the stream is
    followed (`reach` mm, None: three radii)."""
    name: str
    center: np.ndarray
    axis: np.ndarray
    radius: float
    thrust: float
    hub: float = 0.0
    cd: float = 1.2
    rho: float = 1.2
    ducted: bool = False
    reach: float | None = None


@dataclass
class Aero:
    """The study's `aero` block: rotors whose streams the part sits in.
    `in_optimizer`: the drag is a cost in the optimizer's material budget
    (aero.BlockageCost); `grams_per_newton`: what a newton of drag is worth
    in carried mass (1000 / 9.81: a craft that hovers on these rotors)."""
    rotors: list = field(default_factory=list)
    in_optimizer: bool = True
    grams_per_newton: float = 1000.0 / 9.81


def _impact_case(b, ci, loads, free):
    lab = f"loadCases[{ci}].impact"
    if not isinstance(b, dict):
        raise ValueError(f"{lab} is {{energy | velocity (+ mass), crusher: {{distance: [...], force: [...]}}}}")
    if any(ld.direction is not None for ld in loads):
        raise ValueError(f"{lab}: the hit has a known direction (its loads' forces); give each direction as a case")
    energy, vel, mass = b.get("energy"), b.get("velocity"), b.get("mass")
    for key, v in (("energy", energy), ("velocity", vel), ("mass", mass)):
        if v is not None and (not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0):
            raise ValueError(f"{lab}.{key} must be a positive number (J, m/s, kg)")
    if (energy is None) == (vel is None):
        raise ValueError(f"{lab} needs energy (J), or velocity (m/s) with mass (kg), not both")
    if vel is not None and mass is None and not free:
        raise ValueError(f"{lab}.mass (kg) is needed with velocity: only a free part (inertiaRelief) has a mass "
                         "of its own to present at the patch")
    dist = force = None
    cr = b.get("crusher")
    if cr is not None:
        if not isinstance(cr, dict) or cr.get("distance") is None or cr.get("force") is None:
            raise ValueError(f"{lab}.crusher is a force-crush curve {{distance: [mm...], force: [N...]}} (an impact "
                             "report's curve)")
        dist, force = np.asarray(cr["distance"], float).ravel(), np.asarray(cr["force"], float).ravel()
        if len(dist) != len(force) or len(dist) < 2 or np.any(np.diff(dist) <= 0) or np.any(force < 0) \
                or not np.all(np.isfinite(dist)) or not np.all(np.isfinite(force)):
            raise ValueError(f"{lab}.crusher: distance must rise, force be >= 0, the two the same length (>= 2)")
    return ImpactCase(None if energy is None else 1000.0 * float(energy), None if vel is None else float(vel),
                      None if mass is None else float(mass), dist, force)


def _aero(v):
    if v is None:
        return None
    if not isinstance(v, dict) or not isinstance(v.get("rotors"), list) or not v["rotors"]:
        raise ValueError("aero is {rotors: [{center, axis, radius, thrust, ...}]}: the rotors or fans whose "
                         "streams the part sits in")
    from .shapes import _vec

    def num(x, label, positive=True, default=None):
        if x is None and default is not None:
            return default
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or (positive and x <= 0) \
                or x < 0:
            raise ValueError(f"{label} must be a {'positive ' if positive else ''}number")
        return float(x)

    rotors = []
    for i, r in enumerate(v["rotors"]):
        label = f"aero.rotors[{i}]"
        if not isinstance(r, dict):
            raise ValueError(f"{label} is an object: {{center, axis, radius, thrust, hub, ducted, cd}}")
        axis = np.asarray(_axis_vector(r.get("axis"), f"{label}.axis"), float)
        rotors.append(Rotor(
            str(r.get("name") or f"rotor {i + 1}"),
            np.asarray(_vec(r.get("center"), name=f"{label}.center"), float), axis,
            num(r.get("radius"), f"{label}.radius (mm)"), num(r.get("thrust"), f"{label}.thrust (N)"),
            num(r.get("hub", 0.0), f"{label}.hub (mm)", positive=False),
            num(r.get("cd"), f"{label}.cd", default=1.2), num(r.get("airDensity"), f"{label}.airDensity", default=1.2),
            bool(r.get("ducted", False)),
            None if r.get("reach") is None else num(r["reach"], f"{label}.reach (mm)")))
        if rotors[-1].hub >= rotors[-1].radius:
            raise ValueError(f"{label}.hub must be smaller than the radius")
    gpn = v.get("gramsPerNewton")
    return Aero(rotors, bool(v.get("inOptimizer", True)),
                1000.0 / 9.81 if gpn is None else num(gpn, "aero.gramsPerNewton"))


def _axis_vector(a, label):
    """A direction as 'x', '-z' or [x, y, z], as a unit vector."""
    if isinstance(a, str) and a.lstrip("+-") in ("x", "y", "z") and len(a) <= 2:
        out = np.zeros(3)
        out["xyz".index(a.lstrip("+-"))] = -1.0 if a.startswith("-") else 1.0
        return out
    try:
        out = np.asarray(a, float)
    except (TypeError, ValueError):
        out = None
    if out is None or out.shape != (3,) or not np.all(np.isfinite(out)) or np.linalg.norm(out) < 1e-9:
        raise ValueError(f"{label} is the way the air goes: 'x', '-z' or a vector [x, y, z]")
    return out / np.linalg.norm(out)


@dataclass
class MassProperties:
    """The study's `massProperties` block. `through` and `direction`: an
    axis the part turns about (a robot link's joint); `max_inertia`
    (kg mm^2) a cap on the delivered part's moment of inertia about it,
    held by the search; `weight` (0-1) how much of each cell's cost in the
    optimizer is its inertia about the axis rather than its volume
    (domain.volume_weights)."""
    through: np.ndarray | None = None
    direction: np.ndarray | None = None
    max_inertia: float | None = None
    weight: float = 0.0
    # centerOfMass: target (3,) with nan on a free axis, and the tolerance (mm)
    com_target: np.ndarray | None = None
    com_tolerance: float = 1.0


@dataclass
class Goal:
    # Exactly one of these drives the optimizer.
    volume_fraction: float | None = None
    # "As light as possible while every load case keeps this safety factor"
    # and/or "deflects no more than this (mm) at its loads": the study
    # searches over volume fractions, verifying each outcome. A robot arm is
    # usually stiffness-limited long before it is strength-limited.
    safety_factor: float | None = None
    max_deflection: float | None = None
    # "search": compliance designs at a sequence of volume fractions until
    # one passes; "constrained": minimize mass with the limits as MMA
    # constraints, corrected against the verified mesh (constrained.py);
    # "auto": the search, then the constrained method too when the search's
    # part idles (stress utilization under 0.15, not deflection-bound),
    # keeping the lighter (the default; study._run_auto).
    method: str = "auto"
    # How the search's pick is confirmed before delivery: "voxel" re-verifies
    # it on a finer voxel grid; "tet" on a body-fitted tetrahedral mesh
    # (crosscheck.py), whose reading also calibrates the search's voxel
    # stress so it can step down to the real target; "auto" is "tet" when
    # the cross-check is installed and covers the study, else "voxel".
    confirm: str = "auto"
    # Extra fixed fractions to return alongside, for a trade-off gallery.
    outcomes: list[float] = field(default_factory=list)
    # Which passing outcome is delivered: the lightest ("mass"), or for FDM
    # the one a slicer prints fastest ("printTime") or with the least
    # filament, support included ("filament"); see slicer.py.
    minimize: str = "mass"
    # Search this many times from different starting fractions (the goal's
    # volumeFraction times RESTART_STARTS) and deliver the lightest confirmed
    # part. One search settles in the local optimum its start leads to: the
    # same study started at 0.8x and 1.25x delivered 68.7-80.6 g (wall
    # bracket), 140.7-150.1 g (the robot-arm turret). For a final production part;
    # N times the time. See docs/study-format.md.
    restarts: int = 1
    # A safety factor holds against buckling too: the part's linear
    # buckling load factor (verify._buckling) must reach it, as the stress
    # safety factor must. False reports the factor without failing on it.
    buckling: bool = True
    # Travel limits away from the loads: [Clearance]. maxDeflection reads
    # the mean displacement of the loaded nodes, which a load spread over a
    # stiff rim hides: a fan plate read 0.8 mm there while its deck's middle
    # dropped 3.9. A clearance caps the most any node in a region moves (or
    # moves along a direction), "keep this face within d of where it is".
    clearances: list = field(default_factory=list)


# How the part will be made. Each process sets what the design must respect:
# whether layers make it weaker across them (FDM bonds layer to layer at a
# fraction of the in-plane strength), whether overhangs need control, and
# the smallest feature it holds, and the smallest hole that stays open (unfused
# powder sinters shut in a narrow SLS/MJF hole, resin cures shut in a narrow
# SLA one, a nozzle bridges over a narrow FDM one). Values are typical
# published ranges, chosen conservative; override any of them in the study.
# minFeature is never under Materialize's published minimum wall for the
# process (dfm_reference.PROCESSES: FDM 1.0, SLS/MJF 0.8, SLA/DLP 0.6,
# PolyJet 1.0, LPBF metal 1.0 mm), the DFM reference when a part is printed;
# minHole is loadpath's own (Materialize publishes none). Volumetric is not
# offered on Materialize.
PROCESSES = {
    "fdm": {"interlayer": 0.6, "minFeature": 1.0, "minHole": 2.0, "overhangs": True,
            "note": "filament: layer bonds ~50-70% of in-plane strength; self-supporting at 45 degrees"},
    "sls": {"interlayer": 0.9, "minFeature": 0.8, "minHole": 2.0, "overhangs": False,
            "note": "powder bed (PA12, PA11): nearly isotropic, no supports; leave powder escape holes in closed cavities"},
    "mjf": {"interlayer": 0.95, "minFeature": 0.8, "minHole": 2.0, "overhangs": False,
            "note": "HP Multi Jet Fusion: as SLS, slightly more isotropic"},
    "sla": {"interlayer": 1.0, "minFeature": 0.6, "minHole": 1.0, "overhangs": False,
            "note": "resin vat: isotropic once cured; supports are added by the slicer"},
    "dlp": {"interlayer": 1.0, "minFeature": 0.6, "minHole": 1.0, "overhangs": False,
            "note": "resin vat, whole-layer exposure (DLP/LCD): as SLA; large cross-sections peel hard"},
    "volumetric": {"interlayer": 1.0, "minFeature": 0.3, "minHole": 0.6, "overhangs": False,
                   "note": "volumetric / tomographic resin printing: no layers, no supports"},
    "jetting": {"interlayer": 0.95, "minFeature": 1.0, "minHole": 1.0, "overhangs": False,
                "note": "material jetting (PolyJet): photopolymer droplets cured layer by layer, several materials "
                        "mixed voxel by voxel; gel support everywhere, so no overhang limit; nearly isotropic"},
    "lpbf": {"interlayer": 0.85, "minFeature": 1.0, "minHole": 1.0, "overhangs": True,
             "note": "laser powder-bed fusion metal (AlSi10Mg, Ti64, 316L): strength across layers ~85-90% "
                     "as built; overhangs past 45 degrees need supports, which cannot come out of closed "
                     "channels; bridges only ~2 mm; enclosed powder must drain"},
}


@dataclass
class Process:
    kind: str
    # (axis, sign) or a unit vector the layers stack along, if layered;
    # "auto" until orient.choose_build_direction picks one
    build: Any
    interlayer: float                  # strength across layers / in-plane strength
    min_feature: float                 # mm
    overhangs: bool                    # design self-supporting along `build`
    # stiffness across layers / in-plane (modulus, shear); None = the
    # process default in am._LAYER_STIFFNESS
    ez: float | None = None
    gz: float | None = None
    # what the optimizer designs with: False = isotropic (verification
    # always uses the layered stiffness), True = the physical layered
    # stiffness, "bond" = across-layer terms cut to the bond's strength
    # ratio (am.layered_elasticity). Neither layered option beat isotropic
    # reliably on the motor mount and hook (2026-10-29), so off by default.
    anisotropic: Any = False
    # mm: the smallest powder escape or resin drain hole the part's cavities
    # may rely on (processcheck.RULES default)
    escape_hole: float | None = None
    min_hole: float | None = None      # mm
    # mm: longest unsupported bridge the printer manages (printcheck.BRIDGE_SPAN default)
    bridge_span: float | None = None
    # lattice stiffness as printed over designed: None = the published-test
    # fit (lattice_print.print_factor), a number, or "designed" (1)
    lattice_stiffness: Any = None
    # the layer bond's shear strength / in-plane tensile strength; None =
    # interlayer / sqrt(3) (failure.py)
    interlayer_shear: float | None = None


def _process(spec) -> Process | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    kind = str(spec.get("type", "")).lower()
    if kind not in PROCESSES:
        raise ValueError(f"process.type must be one of {sorted(PROCESSES)}, got {kind!r}")
    base = PROCESSES[kind]
    from .am import parse_build_direction
    raw_build = spec.get("buildDirection", "+z" if kind in ("fdm", "lpbf") else None)
    build = "auto" if isinstance(raw_build, str) and raw_build.strip().lower() == "auto" \
        else parse_build_direction(raw_build)
    inter = float(spec.get("interlayer", base["interlayer"]))
    if not 0.0 < inter <= 1.0:
        raise ValueError("process.interlayer is a fraction of the in-plane strength, in (0, 1]")
    ish = spec.get("interlayerShear")
    if ish is not None and not 0.0 < float(ish) <= 1.0:
        raise ValueError("process.interlayerShear is the layer bond's shear strength as a fraction of the "
                         "in-plane tensile strength, in (0, 1]")
    stiff = {}
    for key, name in (("ez", "stiffnessAcrossLayers"), ("gz", "shearAcrossLayers")):
        if spec.get(name) is not None:
            v = float(spec[name])
            if not 0.0 < v <= 1.0:
                raise ValueError(f"process.{name} is a fraction of the in-plane value, in (0, 1]")
            stiff[key] = v
    esc = spec.get("escapeHole")
    if esc is not None and float(esc) <= 0:
        raise ValueError("process.escapeHole is a hole diameter in mm, > 0")
    span = spec.get("bridgeSpan")
    if span is not None and float(span) <= 0:
        raise ValueError("process.bridgeSpan is the longest bridge the printer prints cleanly, in mm, > 0")
    lat_k = spec.get("latticeStiffness")
    if lat_k is not None:
        if isinstance(lat_k, str):
            if lat_k.strip().lower() not in ("designed", "as-printed"):
                raise ValueError('process.latticeStiffness is a factor in (0, 1.5], "as-printed" or "designed"')
            lat_k = None if lat_k.strip().lower() == "as-printed" else "designed"
        elif not 0.0 < float(lat_k) <= 1.5:
            raise ValueError("process.latticeStiffness is the printed lattice's stiffness over the designed "
                             "geometry's, in (0, 1.5]")
        else:
            lat_k = float(lat_k)
    return Process(kind, build, inter, float(spec.get("minFeature", base["minFeature"])),
                   bool(spec.get("overhangs", base["overhangs"])),
                   anisotropic=_anisotropic(spec.get("anisotropic", False)),
                   escape_hole=float(esc) if esc is not None else None,
                   min_hole=float(spec.get("minHole", base["minHole"])),
                   bridge_span=float(span) if span is not None else None, lattice_stiffness=lat_k,
                   interlayer_shear=float(ish) if ish is not None else None, **stiff)


def _drains(spec, process):
    """`drains`: "auto" (default; powder and resin processes and LPBF), true,
    false, or an object of escape.EscapeOptions keys. Resolved to an
    EscapeOptions, or None when off."""
    from .drains import drains_wanted
    from .escape import EscapeOptions

    if spec in ("auto", None):
        return EscapeOptions.for_process(process) if drains_wanted("auto", process) else None
    if not drains_wanted(spec, process):
        return None
    return EscapeOptions.from_spec(spec if isinstance(spec, dict) else {}, process)


def _card_process(process, mat, spec):
    """A library card that measured across-layer strength and stiffness on
    the study's process (material_library.py) sets the process's factors,
    unless the study sets its own; a calibrated card then overrides both.
    A card's minimum wall (a thin flexible TPU skin) raises the process's,
    unless the study sets process.minFeature."""
    if process is None:
        return process
    spec = spec if isinstance(spec, dict) else {}
    mw = getattr(mat, "min_wall", None)
    if mw is not None and "minFeature" not in spec and mw > process.min_feature:
        process.min_feature = float(mw)
    if not mat.measured or mat.measured != process.kind:
        return process
    inter, ez = mat.across_layers()
    if inter is not None and "interlayer" not in spec:
        process.interlayer = inter
    if ez is not None and "stiffnessAcrossLayers" not in spec:
        process.ez = ez
    return process


def _calibrated_process(process, mat_in, spec):
    """A calibrated card (calibration.py) measured its printer's across-layer
    strength and stiffness: use them unless the study sets its own."""
    if process is None or not isinstance(mat_in, str):
        return process
    from .calibration import load_card
    card = load_card(mat_in)
    cp = (card or {}).get("process") or {}
    if cp.get("type") != process.kind:
        return process
    spec = spec or {}
    if "interlayer" not in spec and cp.get("interlayer"):
        process.interlayer = float(cp["interlayer"])
    if "stiffnessAcrossLayers" not in spec and cp.get("stiffnessAcrossLayers"):
        process.ez = float(cp["stiffnessAcrossLayers"])
    # the gyroid coupon measured this printer's lattice against the model
    lat = (card or {}).get("lattice") or {}
    if "latticeStiffness" not in spec and lat.get("stiffnessFactor"):
        process.lattice_stiffness = float(lat["stiffnessFactor"])
    return process


_BUILD_SIM_KEYS = {"layer": "layer", "plate": "plate", "scale": "scale", "release": "release",
                   "bedTemperature": "bed", "chamberTemperature": "chamber", "bedStiffness": "bed_stiffness",
                   "relax": "relax", "recoaterGap": "recoater_gap", "plateCells": "plate_cells",
                   "bedAdhesion": "bed_adhesion", "printLayer": "print_layer", "brim": "brim"}


def _build_sim(spec, process, material):
    """`buildSim`: true or options -> {"setup": buildsim.BuildSetup,
    "compensate": bool, "cells": int}; None when absent or false."""
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    kind = process.kind if process is not None else None
    if kind not in ("lpbf", "fdm", "sls", "mjf"):
        raise ValueError("buildSim simulates lpbf, fdm, sls or mjf builds: give process.type")
    from .buildsim import BuildSetup
    kw = {v: spec[k] for k, v in _BUILD_SIM_KEYS.items() if k in spec}
    if "strain" in spec:
        kw["strain"] = tuple(float(x) for x in spec["strain"])
    if kw.get("release", "full") not in ("full", "none"):
        raise ValueError("buildSim.release is 'full' (off the plate) or 'none' (as built, on it)")
    mat = material if isinstance(material, str) else "pla"
    setup = BuildSetup(process="sls" if kind == "mjf" else kind, material=mat, **kw)
    warp = float(spec.get("warpWeight", 0.0))
    if warp < 0:
        raise ValueError("buildSim.warpWeight is a weight >= 0 (0: verify only)")
    lim_d = float(spec["maxDistortion"]) if spec.get("maxDistortion") is not None else None
    lim_l = float(spec["maxRecoaterLift"]) if spec.get("maxRecoaterLift") is not None else None
    for name, v in (("maxDistortion", lim_d), ("maxRecoaterLift", lim_l)):
        if v is not None and v <= 0:
            raise ValueError(f"buildSim.{name} is a limit in mm > 0")
    if warp > 0 or lim_d is not None or lim_l is not None:
        what = "warpWeight" if warp > 0 else "maxDistortion / maxRecoaterLift"
        if kind not in ("lpbf", "fdm"):
            raise ValueError(f"buildSim.{what} designs against a layer-by-layer build: process lpbf or fdm")
        if process.build == "auto" or process.build is None or len(process.build) != 2:
            raise ValueError(f"buildSim.{what} needs process.buildDirection along a grid axis "
                             "('+z', '-y', ...), not 'auto' or a tilted vector")
    coarsen = spec.get("warpCoarsen")
    ps = spec.get("printSetup")
    if ps is True:
        ps = {}
    if ps:
        if kind != "fdm":
            raise ValueError("buildSim.printSetup searches FDM print setups: process fdm")
        ps = {"materials": ps.get("materials"), "builds": ps.get("orientations"), "beds": ps.get("bedTemperatures"),
              "brims": tuple(float(b) for b in ps.get("brims", (0.0, 5.0))), "max_bed": float(ps.get("maxBed", 110.0)),
              "reoptimize": bool(ps.get("reoptimize", True))}
    return {"printSetup": ps or None, "setup": setup, "compensate": bool(spec.get("compensate", False)),
            "cells": int(spec.get("cells", 60_000)), "iterations": int(spec.get("iterations", 2)),
            "warpWeight": warp, "warpLayer": float(spec["warpLayer"]) if spec.get("warpLayer") else None,
            "maxDistortion": lim_d, "maxRecoaterLift": lim_l,
            "warpCoarsen": int(coarsen) if coarsen is not None else None,
            "warpInOptimizer": bool(spec.get("warpInOptimizer", kind != "fdm"))}


def _slicer(spec, process, base_dir, goal):
    """slicer.SlicerOptions for FDM studies (None otherwise)."""
    from .slicer import SlicerOptions

    if spec is not None and (process is None or process.kind != "fdm"):
        raise ValueError("slicer is for FDM studies: give process {\"type\": \"fdm\"}")
    if goal.minimize != "mass" and (process is None or process.kind != "fdm"):
        raise ValueError(f"goal.minimize {goal.minimize!r} slices the part: it needs process fdm")
    if process is None or process.kind != "fdm":
        return None
    opts = SlicerOptions.from_spec(spec, base_dir)
    if goal.minimize != "mass" and opts.enabled is False:
        raise ValueError(f"goal.minimize {goal.minimize!r} needs the slicer: drop slicer.enabled false")
    if opts.grade is not None:
        from .infill import GradeOptions

        GradeOptions.from_spec(opts.grade, goal.safety_factor)     # validate now, not after the solve
    return opts


def _anisotropic(v):
    if v in (True, False, "bond"):
        return v
    if v in ("physical", "true"):
        return True
    raise ValueError("process.anisotropic is false, true (physical layered stiffness) or \"bond\"")


@dataclass
class Thermal:
    """A study's `thermal` block (see thermal.study_heat): temperatures in C,
    heat in W, convection in W/(m^2 K); each target is (region id, shape)."""
    ambient: float = 25.0
    stress_free: float | None = None          # None: ambient
    temperatures: list = field(default_factory=list)   # (region, shape, C)
    heat: list = field(default_factory=list)           # (region, shape, W, "surface" | "volume")
    convection: list = field(default_factory=list)     # (region, shape, h, fluid C | None)
    conductivity: float | None = None         # W/(m K); None: the material card's
    expansion: Any = True                     # True (card), 1/K, or False
    soften: bool = True
    specific_heat: float | None = None        # J/(kg K); None: the material card's
    cycle: Any = None                         # thermal.Cycle: heat switched on and off, or None (steady)
    # convection.NaturalConvection: still air, the coefficient solved per
    # face from its own temperature (thermal.StudyHeat); None: as listed
    natural: Any = None

    @property
    def t_ref(self):
        return self.ambient if self.stress_free is None else self.stress_free


# W/(m^2 K): where a still-air solve starts before it has temperatures to
# read its own coefficient from (convection.py)
STILL_AIR = 8.0


def _thermal(t, base_dir) -> Thermal | None:
    if t is None:
        return None
    if not isinstance(t, dict):
        raise ValueError("thermal must be an object: {ambient, temperatures, heat, convection, ...}")

    def num(v, label, positive=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or (positive and v <= 0):
            raise ValueError(f"{label} must be a {'positive ' if positive else ''}number")
        return float(v)

    out = Thermal(ambient=num(t.get("ambient", 25.0), "thermal.ambient"))
    if t.get("stressFree") is not None:
        out.stress_free = num(t["stressFree"], "thermal.stressFree")
    for i, it in enumerate(t.get("temperatures") or []):
        label = f"thermal.temperatures[{i}]"
        region, shape = _target(it, label, base_dir)
        out.temperatures.append((region, shape, num(it.get("t"), f"{label}.t (C)")))
    for i, it in enumerate(t.get("heat") or []):
        label = f"thermal.heat[{i}]"
        region, shape = _target(it, label, base_dir)
        spread = it.get("spread", "surface")
        if spread not in ("surface", "volume"):
            raise ValueError(f"{label}.spread must be 'surface' or 'volume'")
        out.heat.append((region, shape, num(it.get("watts"), f"{label}.watts", True), spread))
    conv = t.get("convection", "natural")
    if isinstance(conv, str) or (isinstance(conv, dict) and conv.get("type") == "natural"):
        # still air: the coefficient is not an input but a result of the
        # surface's temperature, direction and surroundings
        from .am import build_vector, parse_build_direction
        from .convection import NaturalConvection

        if isinstance(conv, str) and conv.strip().lower() != "natural":
            raise ValueError('thermal.convection is "natural", a coefficient in W/(m² K) or a list of '
                             "{h, region|shape, fluid}")
        opts = conv if isinstance(conv, dict) else {}
        bad = set(opts) - {"type", "emissivity", "up", "pressure"}
        if bad:
            raise ValueError(f"thermal.convection (natural): unknown key(s) {sorted(bad)}; it takes emissivity, "
                             "up and pressure")
        em = opts.get("emissivity")
        if em is not None and not 0.0 < num(em, "thermal.convection.emissivity") <= 1.0:
            raise ValueError("thermal.convection.emissivity is in (0, 1]")
        try:
            up = build_vector(parse_build_direction(opts.get("up", "+z")))
        except (ValueError, TypeError):
            raise ValueError("thermal.convection.up is the direction opposing gravity: '+z', '-y', ... or "
                             "[x, y, z]") from None
        out.natural = NaturalConvection(None if em is None else float(em),
                                        tuple(float(v) for v in up / np.linalg.norm(up)),
                                        num(opts.get("pressure", 1.01325), "thermal.convection.pressure (bar)", True))
        conv = STILL_AIR
    if isinstance(conv, (int, float)) and not isinstance(conv, bool):
        conv = [{"h": conv}]
    if not isinstance(conv, list):
        raise ValueError('thermal.convection is "natural", a coefficient in W/(m² K) or a list of '
                         "{h, region|shape, fluid}")
    for i, it in enumerate(conv):
        label = f"thermal.convection[{i}]"
        h = num(it.get("h"), f"{label}.h (W/(m² K))")
        if h < 0:
            raise ValueError(f"{label}.h must be >= 0")
        region, shape = _target(it, label, base_dir) if ("region" in it or "shape" in it) else (None, None)
        fluid = num(it["fluid"], f"{label}.fluid (C)") if it.get("fluid") is not None else None
        out.convection.append((region, shape, h, fluid))
    if t.get("conductivity") is not None:
        out.conductivity = num(t["conductivity"], "thermal.conductivity", True)
    e = t.get("expansion", True)
    if not isinstance(e, bool):
        e = num(e, "thermal.expansion (1/K, or true/false)")
        if e < 0:
            raise ValueError("thermal.expansion must be >= 0")
    out.expansion = e
    out.soften = bool(t.get("soften", True))
    if t.get("specificHeat") is not None:
        out.specific_heat = num(t["specificHeat"], "thermal.specificHeat (J/(kg K))", True)
    cy = t.get("cycle")
    if cy is not None:
        from .thermal import Cycle

        if not isinstance(cy, dict):
            raise ValueError("thermal.cycle is {period, on, cycles}: seconds per cycle, seconds the heat is on")
        period = num(cy.get("period"), "thermal.cycle.period (s)", True)
        on = num(cy.get("on"), "thermal.cycle.on (s the heat is on each period)", True)
        if on >= period:
            raise ValueError("thermal.cycle.on must be shorter than the period (for heat on all the time, "
                             "leave cycle out)")
        if not out.heat:
            raise ValueError("thermal.cycle switches the heat inputs on and off: give thermal.heat")
        n = cy.get("cycles")
        if n is not None:
            n = num(n, "thermal.cycle.cycles (over the part's life)", True)
        steps = cy.get("steps", 48)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 8:
            raise ValueError("thermal.cycle.steps is a whole number of time steps per period, at least 8")
        fw = cy.get("fatigueWeight", 0.0)
        fw = num(fw, "thermal.cycle.fatigueWeight")
        if fw < 0:
            raise ValueError("thermal.cycle.fatigueWeight must be >= 0")
        if fw > 0 and out.expansion is False:
            raise ValueError("thermal.cycle.fatigueWeight needs expansion: without it the swing carries no stress")
        fsf = cy.get("fatigueSafetyFactor")
        if fsf is not None:
            fsf = num(fsf, "thermal.cycle.fatigueSafetyFactor", True)
            if n is None:
                raise ValueError("thermal.cycle.fatigueSafetyFactor needs cycles: the fatigue strength it holds "
                                 "the swing under is the one at that many cycles")
            if out.expansion is False:
                raise ValueError("thermal.cycle.fatigueSafetyFactor needs expansion: without it the swing "
                                 "carries no stress")
            if fw > 0:
                raise ValueError("thermal.cycle: give fatigueWeight (objective) or fatigueSafetyFactor "
                                 "(constraint), not both")
        out.cycle = Cycle(period, on, n, steps, fw, fsf)
    if not out.temperatures and not any(h > 0 for *_, h, _ in out.convection):
        raise ValueError("thermal needs a way for heat out: a fixed temperature (temperatures) or convection")
    # a part that just runs warm (inside a 55 C enclosure): ambient alone,
    # every cell at it, softened for it; with no heat and no fixed
    # temperature it is that or nothing
    if not out.temperatures and not out.heat and out.t_ref == out.ambient \
            and (not out.soften or abs(out.ambient - 25.0) < 1.0):
        raise ValueError("thermal has nothing to do: no heat input, no fixed temperature, the stress-free "
                         "temperature is the ambient, and the ambient is room temperature (for a part that "
                         "runs warm, set ambient to its service temperature)")
    return out


@dataclass
class Study:
    name: str
    design_space: list[Shape]
    preserve: list[Region]
    keep_out: list[Region]
    supports: list[Support]
    load_cases: list[LoadCase]
    material: Material
    goal: Goal
    min_member: float | None
    symmetry: list[str]
    pitch: float | None
    max_elements: int
    max_iterations: int = 120
    # m/s², applied to the part's own mass in every load case.
    gravity: np.ndarray | None = None
    # Build direction for overhang control, as (axis, sign); see am.py.
    print_direction: Any = None
    # Heat in the part (spec.Thermal), applied in every load case, or None.
    thermal: Any = None
    # 2.5D manufacturing: the part is a profile extruded along this axis
    # (0, 1, 2), i.e. cut from plate by waterjet, laser or a 2.5-axis mill.
    extrusion: int | None = None
    # Milling or casting access (manufacture.Access), a circular pattern
    # (manufacture.Pattern), and the thickest member allowed (mm).
    access: Any = None
    pattern: Any = None
    max_member: float | None = None
    # Every material to try (a list in the study); `material` is the first.
    materials: list = field(default_factory=list)
    # How it is made (spec.PROCESSES): layer weakness, overhangs, min feature.
    process: Process | None = None
    # (region id, thickness mm) of every wall and skin pinned solid
    shells: list = field(default_factory=list)
    # fatigue / creep settings (durability.Durability), and each fastener's
    # preload and head bearing area for the relaxation check
    durability: Any = None
    joints: list = field(default_factory=list)
    # FDM: slicer.SlicerOptions (print time, filament, support per outcome;
    # graded infill); None for other processes
    slicer: Any = None
    # Smallest hole (mm) the optimizer may leave, or None: see simp.SimpOptions.dilate.
    min_hole: float | None = None
    # Escape holes into sealed voids: "auto" (powder and resin processes),
    # True, False, or an escape.EscapeOptions; see escape.py.
    drains: Any = "auto"
    # multimat.MultiMaterial: the optimizer places several materials
    multi_material: Any = None
    # Build-process simulation of the delivered part (buildsim.py): a
    # buildsim.BuildSetup plus {"compensate": bool, "cells": int}, or None
    build_sim: Any = None
    # fits.py rows: each fitted bore (bearing seat, pin, bolt hole) with the
    # hole drawn for the process and the window it must print to
    fits: list = field(default_factory=list)
    # natural frequency floor and carried masses (Vibration), or None
    vibration: Any = None
    # an axis to keep the inertia down about (MassProperties), or None
    mass_properties: Any = None
    # rotors whose streams the part sits in (Aero), or None
    aero: Any = None
    # inertiaRelief.masses: what the free part carries, [(region id or None,
    # shape or None, kg)]. A drone frame's motors and battery outweigh it,
    # so they, not the frame, take most of the thrust's inertia.
    inertia_masses: list = field(default_factory=list)
    # ... and those given with their centre (`at`): rigid bodies, [(region,
    # shape, kg, at (3,), inertia about the centre (3,) or (3, 3) kg mm^2)]
    inertia_bodies: list = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    # where the study's relative mesh paths start (blocks read later:
    # switch.output.bearsOn, switch.clearance, skin.openings)
    base_dir: str | None = None

    @property
    def length_scale(self) -> float:
        """The member size the density filter enforces (mm): minMember, else
        DEFAULT_MIN_MEMBER. A grid fine enough to resolve a sub-millimetre
        sheet would otherwise build one (the 845k-cell arm link with no
        minMember came out a 1 mm skin sealed round its cable channel)."""
        return float(self.min_member) if self.min_member else DEFAULT_MIN_MEMBER

    @property
    def inertia_relief(self) -> bool:
        """No supports: the part floats free and its own inertia balances
        the loads (study `inertiaRelief`)."""
        return bool((self.raw or {}).get("inertiaRelief"))

    # impact cases' loads scaled to the coupled peak force while a part is
    # verified ({case name: factor on the loads as written}; verify_part)
    impact_scale: dict = field(default_factory=dict)

    def case_factor(self, name: str) -> float:
        """How much a case's safety factor counts for against the goal's:
        goal / the case's own (loadCases[].safetyFactor), 1 without one. A
        case's reading times this is what the search compares to the goal
        (a crash at SF 1.3 with its own 1.25 counts as 2.08 against a goal
        of 2). Cases a load of unknown plane split (`case@angle`) take the
        written case's."""
        g = self.goal.safety_factor
        base = str(name).split("@")[0]
        for c in self.load_cases:
            if (c.name == name or c.name == base) and c.safety_factor and g:
                return float(g) / float(c.safety_factor)
        return 1.0

    @property
    def carried_mass(self) -> float:
        """kg the free part carries (inertiaRelief.masses)."""
        return float(sum(kg for _, _, kg in self.inertia_masses) + sum(b[2] for b in self.inertia_bodies))

    @property
    def self_equilibrated(self) -> bool:
        """No supports: every load is a pressure, held by pressure.stabilize."""
        return not self.supports and not (self.raw or {}).get("feel") and not (self.raw or {}).get("cooling")

    def elastic(self, material: Material | None = None, design: bool = False):
        """What the solver takes for `nu`: the material's Poisson's ratio, or
        a fem.Layered when the process builds in layers (am.layered_elasticity).
        design=True is the optimizer's surrogate stiffness."""
        from .am import layered_elasticity

        mat = material or self.material
        if self.process is None or (design and not self.process.anisotropic):
            return mat.nu
        return layered_elasticity(mat.nu, self.process, design=design and self.process.anisotropic == "bond")

    def region_shape(self, region_id: str) -> tuple[Shape, str]:
        for r in self.preserve:
            if r.id == region_id:
                return r.shape, "preserve"
        for r in self.keep_out:
            if r.id == region_id:
                return r.shape, "keepOut"
        known = [r.id for r in self.preserve] + [r.id for r in self.keep_out]
        raise ValueError(f"unknown region {region_id!r}; known regions: {known}")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        los, his = [], []
        for s in self.design_space + [r.shape for r in self.preserve]:
            lo, hi = s.bounds()
            los.append(lo)
            his.append(hi)
        return np.min(los, axis=0), np.max(his, axis=0)


def cuts_of(study, region) -> list:
    """The keep-outs that cut preserved `region` out of the part: every
    cutting keep-out (Region.cuts), but for the skin only the bores."""
    return [k for k in study.keep_out if getattr(k, "cuts", True)
            and (region.id != "skin" or getattr(k, "bore", True))]


def _regions(items, label, base_dir, bounds=None):
    out, seen = [], set()
    for i, item in enumerate(items or []):
        if not isinstance(item, dict) or "shape" not in item:
            raise ValueError(f"{label}[{i}] must be {{'id', 'shape'}}")
        rid = str(item.get("id") or f"{label}-{i}")
        if rid in seen:
            raise ValueError(f"duplicate {label} id {rid!r}")
        seen.add(rid)
        wall = item.get("wall")
        if wall is not None and not (isinstance(wall, (int, float)) and wall > 0):
            raise ValueError(f"{label}[{i}].wall is a thickness in mm (> 0)")
        shape = shape_from_spec(item["shape"], base_dir)
        if item.get("insertFrom") is not None:
            if label != "keepOut":
                raise ValueError(f"{label}[{i}].insertFrom: only a keep-out has an assembly path")
            if bounds is None:
                raise ValueError(f"{label}[{i}].insertFrom needs the design space")
            shape = _assembly_path(shape, item["insertFrom"], bounds, f"{label}[{i}].insertFrom")
        bore = bool(item.get("bore", True)) and item.get("insertFrom") is None
        cuts = bool(item.get("bore", True)) or item.get("insertFrom") is not None
        out.append(Region(rid, shape, bore, float(wall) if wall is not None else None,
                          bool(item.get("cutsPreserve", cuts))))
    return out


def _assembly_path(shape, side, bounds, label):
    """A keep-out swept out of the design space toward `side` (an axis like
    '-z', or a vector): the way the component (or a tool) comes in.

    Or a path of legs, as the component leaves its place: [{toward, length},
    ..., {toward}], each leg an axis or vector, every one but the last with
    a length (mm); the last goes on out of the design space unless it has a
    length too (an opening's mouth, swept a skin's depth inward). A board whose
    end sits in a slot in the wall slides back out of it, then drops out
    through the opening: [{"toward": "-y", "length": 3.5}, "-z"]."""
    from .shapes import TurnSweep, Union, posed, sweep

    # a vector ([x, y, z], as shells.apply_split writes for a tilted
    # parting line) is one straight way out, not a path of three legs
    is_vec = (isinstance(side, (list, tuple)) and len(side) == 3
              and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in side))
    if isinstance(side, list) and not is_vec:
        if not side:
            raise ValueError(f"{label}: an empty path")
        parts, cur, at, R = [], shape, np.zeros(3), np.eye(3)
        for j, leg in enumerate(side):
            leg = {"toward": leg} if not isinstance(leg, dict) else leg
            last = j == len(side) - 1
            if "turn" in leg:
                # a turn about a line: {turn: degrees, about: axis, at: [x, y, z], along?: mm}
                if not isinstance(leg["turn"], (int, float)) or not leg["turn"] or abs(leg["turn"]) > 180:
                    raise ValueError(f"{label}[{j}].turn: degrees about the axis (not 0, at most 180 either way)")
                if leg.get("about") is None or leg.get("at") is None:
                    raise ValueError(f"{label}[{j}]: a turn needs about (an axis) and at (a point on its line)")
                a = _leg_direction(leg["about"], f"{label}[{j}].about")
                p0 = np.asarray(leg["at"], float)
                if p0.shape != (3,):
                    raise ValueError(f"{label}[{j}].at: a point [x, y, z] on the turn's axis")
                ts = TurnSweep(cur, a, p0, float(leg["turn"]), float(leg.get("along") or 0.0))
                parts.append(ts)
                Q, t = ts.pose(1.0)
                R, at = Q @ R, Q @ at + t
                cur = posed(shape, R, at)
                continue
            if "toward" not in leg:
                raise ValueError(f"{label}[{j}] is {{toward, length}}, an axis, or a turn {{turn, about, at}}")
            if last and leg.get("length") is None:
                parts.append(_assembly_path(cur, leg["toward"], bounds, f"{label}[{j}]"))
                break
            if not isinstance(leg.get("length"), (int, float)) or leg["length"] <= 0:
                raise ValueError(f"{label}[{j}].length: mm along this leg (> 0); only the last leg may have none")
            d = _leg_direction(leg["toward"], f"{label}[{j}]")
            parts.append(sweep(cur, d, float(leg["length"])))
            at = at + float(leg["length"]) * d
            cur = posed(shape, R, at)
        return Union(parts)
    d = _leg_direction(side, label)
    lo, hi = bounds
    slo, shi = shape.bounds()

    def corners(a, b):
        return np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])

    length = float((corners(lo, hi) @ d).max() - (corners(slo, shi) @ d).min()) + 1.0
    return sweep(shape, d, max(length, 0.0))


def _leg_direction(side, label):
    if isinstance(side, str):
        key = side.strip().lower()
        key = key if key[:1] in "+-" else "+" + key
        if key not in _FACES:
            raise ValueError(f"{label} is a side like '-z' or a vector, got {side!r}")
        a, sg = _FACES[key]
        d = np.zeros(3)
        d[a] = sg
    else:
        d = np.asarray(side, float)
        if d.shape != (3,) or not np.linalg.norm(d) > 0:
            raise ValueError(f"{label} is a side like '-z' or a nonzero vector [x, y, z]")
        d = d / np.linalg.norm(d)
    return d


_FACES = {f"{s}{a}": (i, 1 if s == "+" else -1) for i, a in enumerate("xyz") for s in "+-"}


def _skin(v, design, base_dir=None):
    """Top-level `skin`: a thickness (mm) or {thickness, open: ['-z', ..],
    openings: [{shape, inward?, depth?}], of?}. Returns (thickness, the
    skin shape). `of` (a shape, or a list as designSpace) is the exterior
    the skin lines when the design space is only part of it: a split
    half's skin follows the whole exterior and stops at the cut.

    `openings` are where the exterior opens into the interior (an
    underside a flap closes, a port). The exterior's solid ends at an
    opening's inner end, so the skin would line that end too: a membrane a
    skin thick across the cavity behind it (the touch microphone's first
    study had one, 0.4 mm thick, across its board bay). The skin is not
    kept inside an opening, swept `depth` mm `inward` when given."""
    from .shapes import Shell, Union, shrink, sweep

    if isinstance(v, (int, float)):
        v = {"thickness": v}
    if not isinstance(v, dict):
        raise ValueError("skin is a thickness in mm, or {thickness, open: ['-z', ...]}")
    t = v.get("thickness")
    if not isinstance(t, (int, float)) or t <= 0:
        raise ValueError("skin.thickness is the wall in mm (> 0)")
    faces = []
    for f in v.get("open") or []:
        key = str(f).strip().lower()
        key = key if key[:1] in "+-" else "+" + key
        if key not in _FACES:
            raise ValueError(f"skin.open entries are design-space faces like '-z' or '+x', got {f!r}")
        faces.append(_FACES[key])
    ds = design[0] if len(design) == 1 else Union(design)
    part = ds
    if v.get("of") is not None:
        # the design space is part of a larger exterior (a split half): the
        # skin lines that exterior and stops where the design space does,
        # so it is not drawn across the cut
        of = v["of"]
        of_shapes = [shape_from_spec(x, base_dir) for x in (of if isinstance(of, list) else [of])]
        ds = of_shapes[0] if len(of_shapes) == 1 else Union(of_shapes)
    inner = shrink(ds, float(t), faces)
    zones = []
    for j, z in enumerate(v.get("zones") or []):
        if not isinstance(z, dict) or "shape" not in z or not isinstance(z.get("thickness"), (int, float)) \
                or z["thickness"] <= 0:
            raise ValueError(f"skin.zones[{j}] is {{shape, thickness (mm > 0), blend?}}")
        zones.append((shape_from_spec(z["shape"], base_dir), float(z["thickness"]), float(z.get("blend", 2.0))))
    if zones:
        inner = _zoned(ds, float(t), faces, zones)
        # the thinnest skin is the one the grid has to resolve
        t = min([float(t)] + [z[1] for z in zones])
    holes = []
    for j, o in enumerate(v.get("openings") or []):
        if not isinstance(o, dict) or "shape" not in o:
            raise ValueError(f"skin.openings[{j}] is {{shape, inward?, depth?}}")
        sh = shape_from_spec(o["shape"], base_dir)
        if o.get("depth") is not None:
            sh = sweep(sh, _leg_direction(o.get("inward", "+z"), f"skin.openings[{j}].inward"), float(o["depth"]))
        holes.append(sh)
    if holes:
        inner = Union([inner] + holes)
    if part is not ds:
        from .shapes import Boolean
        return float(t), Boolean("intersect", [Shell(ds, inner), part])
    return float(t), Shell(ds, inner)


def _zoned(ds, t, faces, zones):
    """The inner surface of a skin `t` mm thick except in `zones`
    [(shape, thickness, blend)], where it is that zone's thickness, ramped
    smoothly over `blend` mm across the zone's edge. The exterior is not
    touched: a Thickened offset of the thinnest skin, so the lining and the
    cut-cell skin see one smooth inner surface."""
    from .shapes import MeshShape, Offset, Thickened

    t_min = min([t] + [z[1] for z in zones])
    t_max = max([t] + [z[1] for z in zones])
    base = Offset(ds, -t_min, faces)
    if isinstance(ds, MeshShape):
        ds.exact_band(t_max + 0.5)

    def extra(p):
        te = np.full(len(p), t)
        for shape, tz, blend in zones:
            d = shape.sdf(p)
            w = np.clip(0.5 - d / blend, 0.0, 1.0) if blend > 0 else (d <= 0).astype(float)
            w = w * w * (3 - 2 * w)
            te += (tz - t) * w
        return np.clip(te, t_min, t_max) - t_min

    return Thickened(base, extra)


def _target(item, label, base_dir):
    if "region" in item:
        return str(item["region"]), None
    if "shape" in item:
        return None, shape_from_spec(item["shape"], base_dir)
    raise ValueError(f"{label} needs a 'region' id or an inline 'shape'")


def _mass_properties(v, goal):
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("massProperties is an object: {axis: {through, direction}, maxInertia, inertiaWeight}")
    from .am import build_vector
    from .shapes import _vec

    through = direction = None
    if v.get("axis") is not None:
        ax = v["axis"]
        if not isinstance(ax, dict) or "through" not in ax or "direction" not in ax:
            raise ValueError("massProperties.axis is {through: [x, y, z], direction: 'z' or [x, y, z]}: the axis "
                             "the part turns about")
        through = np.asarray(_vec(ax["through"], name="massProperties.axis.through"), float)
        dv = ax["direction"]
        dv = build_vector(parse_build_direction(dv)) if isinstance(dv, str) else dv
        direction = np.asarray(dv, float).reshape(-1)
        if direction.shape != (3,) or not np.linalg.norm(direction) > 0:
            raise ValueError("massProperties.axis.direction is an axis like 'z' or a vector")
        direction = direction / np.linalg.norm(direction)
    mi = v.get("maxInertia")
    if mi is not None:
        if through is None:
            raise ValueError("massProperties.maxInertia is about the axis: give massProperties.axis")
        if not isinstance(mi, (int, float)) or mi <= 0:
            raise ValueError("massProperties.maxInertia is kg mm^2 (> 0)")
        if goal.safety_factor is None and goal.max_deflection is None:
            raise ValueError("massProperties.maxInertia is held by the search: give the goal a safetyFactor or "
                             "maxDeflection too")
    w = v.get("inertiaWeight", 0.5 if through is not None else 0.0)
    if not isinstance(w, (int, float)) or not 0 <= w <= 1:
        raise ValueError("massProperties.inertiaWeight is 0-1: the share of a cell's cost that is its inertia")
    if w > 0 and through is None:
        raise ValueError("massProperties.inertiaWeight needs massProperties.axis")
    com_t, com_tol = None, 1.0
    if v.get("centerOfMass") is not None:
        cm = v["centerOfMass"]
        t = cm.get("target") if isinstance(cm, dict) else None
        if not isinstance(t, (list, tuple)) or len(t) != 3 \
                or not all(x is None or isinstance(x, (int, float)) for x in t) or all(x is None for x in t):
            raise ValueError("massProperties.centerOfMass is {target: [x, y, z] (mm, null for a free axis), "
                             "tolerance}")
        com_t = np.array([np.nan if x is None else float(x) for x in t])
        com_tol = cm.get("tolerance", 1.0)
        if not isinstance(com_tol, (int, float)) or com_tol <= 0:
            raise ValueError("massProperties.centerOfMass.tolerance is mm (> 0)")
        com_tol = float(com_tol)
    return MassProperties(through, direction, float(mi) if mi is not None else None, float(w), com_t, com_tol)


def _clearances(items, base_dir):
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("goal.clearances is a list of {region or shape, maxTravel, direction?, cases?}")
    out = []
    for i, it in enumerate(items):
        label = f"goal.clearances[{i}]"
        if not isinstance(it, dict):
            raise ValueError(f"{label} is an object: {{region or shape, maxTravel}}")
        mt = it.get("maxTravel")
        if not isinstance(mt, (int, float)) or isinstance(mt, bool) or mt <= 0:
            raise ValueError(f"{label}.maxTravel is how far (mm, > 0) the region may move")
        region, shape = _target(it, label, base_dir)
        dv = it.get("direction")
        direction = None
        if dv is not None:
            if isinstance(dv, str):
                from .am import build_vector

                dv = build_vector(parse_build_direction(dv))
            direction = np.asarray(dv, float).reshape(-1)
            if direction.shape != (3,) or not np.isfinite(direction).all() or not np.linalg.norm(direction) > 0:
                raise ValueError(f"{label}.direction is an axis like '-z' or a vector [x, y, z]")
            direction = direction / np.linalg.norm(direction)
        cases = it.get("cases") or []
        if isinstance(cases, str):
            cases = [cases]
        out.append(Clearance(region, shape, float(mt), direction, [str(c) for c in cases],
                             str(it.get("name") or region or f"clearance {i + 1}")))
    return out


def _vibration(v, goal, supports, base_dir, free=False):
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("vibration must be an object: {minFrequency, excitations, masses, modes}")
    fmin = v.get("minFrequency")
    if fmin is not None and (isinstance(fmin, bool) or not isinstance(fmin, (int, float)) or fmin <= 0):
        raise ValueError("vibration.minFrequency is the Hz the part's first natural frequency must reach")
    excitations = []
    for i, ex in enumerate(v.get("excitations") or []):
        label = f"vibration.excitations[{i}]"
        band = ex.get("band") if isinstance(ex, dict) else None
        if (not isinstance(band, (list, tuple)) or len(band) != 2
                or any(isinstance(b, bool) or not isinstance(b, (int, float)) for b in band)
                or not 0 < band[0] < band[1]):
            raise ValueError(f"{label}.band is [lo, hi] in Hz, the range the forcing sweeps (0 < lo < hi)")
        amp = ex.get("maxAmplification")
        if amp is not None and (isinstance(amp, bool) or not isinstance(amp, (int, float)) or amp <= 1):
            raise ValueError(f"{label}.maxAmplification is the most a mode may magnify this forcing over its "
                             "static response (> 1)")
        excitations.append((str(ex.get("name") or f"excitation {i + 1}"), float(band[0]), float(band[1]),
                            None if amp is None else float(amp)))
    if fmin is None and not excitations:
        raise ValueError("vibration needs minFrequency (the Hz the first natural frequency must reach) or "
                         "excitations (bands of forcing no mode may sit in)")
    if not supports and not free:
        raise ValueError("vibration needs supports (natural frequencies of the part clamped where it is held), "
                         "or inertiaRelief for a part that flies free (its free-free modes)")
    if goal.safety_factor is None and goal.max_deflection is None:
        raise ValueError("vibration is held by the search for the lightest passing part: give the "
                         "goal a safetyFactor or maxDeflection too")
    masses = []
    for i, m in enumerate(v.get("masses") or []):
        label = f"vibration.masses[{i}]"
        if not isinstance(m, dict) or not isinstance(m.get("mass"), (int, float)) or m["mass"] <= 0:
            raise ValueError(f"{label}.mass is the carried mass in kg (> 0)")
        region, shape = _target(m, label, base_dir)
        masses.append((region, shape, float(m["mass"])))
    modes = int(v.get("modes", 3))
    if not 1 <= modes <= 20:
        raise ValueError("vibration.modes is how many modes to report (1-20)")
    return Vibration(None if fmin is None else float(fmin), masses, modes, bool(v.get("inOptimizer", True)),
                     excitations)


def _extrusion_axis(v):
    if v is None:
        return None
    a = str(v).strip().lower().lstrip("+-")
    if a not in ("x", "y", "z"):
        raise ValueError("extrusion must be 'x', 'y' or 'z' (the axis the profile is extruded along)")
    return "xyz".index(a)


def _mass_load(ld, f, m, region, shape, label, preserve, keep_out):
    """A carried mass as the force and moment it puts on its region: weight
    x dynamic factor along gravity, and the moment of that force about the
    region's centre when the mass sits elsewhere ('at')."""
    from .am import build_vector

    mass, dyn = ld["mass"], ld.get("dynamic", 1.0)
    if not isinstance(mass, (int, float)) or mass <= 0:
        raise ValueError(f"{label}.mass is the carried mass in kg (> 0)")
    if not isinstance(dyn, (int, float)) or dyn < 1:
        raise ValueError(f"{label}.dynamic multiplies the weight: 1 static, 2 a moving joint, 3 a crash")
    try:
        g = build_vector(parse_build_direction(ld.get("gravity", "-z")))
    except (ValueError, TypeError, KeyError):
        raise ValueError(f"{label}.gravity is the way the mass pulls: '-z', '+y', ... or [x, y, z]") from None
    w = 9.81 * float(mass) * float(dyn) * np.asarray(g, float) / np.linalg.norm(g)
    return f + w, m


def _at_moment(ld, f, m, region, shape, label, preserve, keep_out):
    """A force (or carried mass) applied away from its region ('at'): the
    moment of the whole force about the region's centre. A belt pulling on
    a pulley in front of a motor face, a gripper's mass 80 mm out."""
    from .shapes import _vec

    if ld.get("direction") is not None or ld.get("cone") is not None:
        raise ValueError(f"{label}: a force 'at' an offset makes a moment that would have to turn with an "
                         "unknown direction; give each direction you need as a load case of its own (or use an "
                         "in-plane direction such as \"xz\", which turns the force round 'at'; it takes a plain force, so write a mass as kg x 9.81 x the dynamic factor in N)")
    if shape is None:
        regs = {r.id: r.shape for r in list(preserve) + list(keep_out)}
        shape = regs.get(region)
    if shape is None:
        raise ValueError(f"{label}: 'at' needs the region's shape to measure the lever from")
    lo, hi = shape.bounds()
    lever = _vec(ld["at"], name=f"{label}.at") - 0.5 * (np.asarray(lo) + np.asarray(hi))
    return m + np.cross(lever, f)


_PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
_PLANAR_STEPS = 8          # 22.5 degrees apart: the worst direction is within cos(11.25) = 98%


def _expand_planar(d: dict) -> dict:
    """A load whose direction is any in a plane ("direction": "xz") becomes
    eight plain cases round that plane. Von Mises stress does not care about
    the force's sign, so half a turn covers it. Brief-eval agents wrote these
    cases by hand (a pillow block's radial load, a hook's swinging strap): a
    3D 'any' or a cone also pushes out of plane, which a thin part cannot
    carry and need not."""
    cases = d.get("loadCases")
    if not isinstance(cases, list) or not any(
            isinstance(ld, dict) and ld.get("direction") in _PLANES
            for c in cases if isinstance(c, dict) for ld in (c.get("loads") or [])):
        return d
    out = []
    for ci, c in enumerate(cases):
        loads = c.get("loads") or [] if isinstance(c, dict) else []
        planar = [i for i, ld in enumerate(loads) if isinstance(ld, dict) and ld.get("direction") in _PLANES]
        if not planar:
            out.append(c)
            continue
        if len(planar) > 1:
            raise ValueError(f"loadCases[{ci}]: at most one load per case may have an in-plane direction; "
                             "put another in a load case of its own")
        li = planar[0]
        ld = loads[li]
        if ld.get("cone") is not None or ld.get("mass") is not None or ld.get("pressure") is not None:
            raise ValueError(f"loadCases[{ci}].loads[{li}]: an in-plane direction goes with a force, not a cone, "
                             "mass or pressure. For a swinging payload, write the force out: mass (kg) x 9.81 x the dynamic "
                             "factor, in N, e.g. 0.6 kg at 2 g is \"force\": [11.8, 0, 0]")
        a, b = _PLANES[ld["direction"]]
        f = np.asarray(ld.get("force") or [0, 0, 0], float)
        size = float(np.linalg.norm(f))
        if size <= 0:
            raise ValueError(f"loadCases[{ci}].loads[{li}]: direction {ld['direction']!r} needs a force "
                             "(its size, and the first direction to try)")
        start = float(np.arctan2(f[b], f[a])) if abs(f[a]) + abs(f[b]) > 1e-9 * size else 0.0
        normal = np.zeros(3)
        normal[3 - a - b] = 1.0 if (a, b) != (0, 2) else -1.0   # right-handed: a x b
        m0 = np.asarray(ld.get("moment") or [0, 0, 0], float)
        name = c.get("name") or f"case{ci}"
        # Half a circle covers it only when the stress under -F is the stress
        # under F: the case's one load, spread evenly, with no moment and an
        # isotropic, tension-blind check. A bearing load presses on one side
        # of the bore, other loads add, a moment does not turn round, and a
        # layered process fails on interlayer tension: the full circle then.
        full = (len(loads) > 1 or ld.get("distribution") == "bearing" or ld.get("moment") is not None
                or d.get("process") is not None or d.get("printDirection") is not None)
        steps, span = (2 * _PLANAR_STEPS, 2 * np.pi) if full else (_PLANAR_STEPS, np.pi)
        for k in range(steps):
            t = span * k / steps
            fk = np.zeros(3)
            fk[a], fk[b] = size * np.cos(start + t), size * np.sin(start + t)
            # the moment turns with the force about the plane's normal
            cth, sth = np.cos(t), np.sin(t)
            mk = m0 * cth + np.cross(normal, m0) * sth + normal * (normal @ m0) * (1 - cth)
            new = {k2: v for k2, v in ld.items() if k2 != "direction"}
            new["force"] = fk.round(9).tolist()
            if ld.get("moment") is not None:
                new["moment"] = mk.round(9).tolist()
            out.append({**c, "name": f"{name}@{180.0 * k / _PLANAR_STEPS:g}",
                        "loads": [new if i == li else x for i, x in enumerate(loads)]})
    return {**d, "loadCases": out}


def study_from_dict(d: dict, base_dir: str | None = None) -> Study:
    """Validate and parse a study. Every error names the field to fix."""
    if not isinstance(d, dict):
        raise ValueError("study must be a JSON object")
    from . import quick
    from .fasteners import expand_fasteners
    d = quick.study(d)
    d = expand_fasteners(d)
    d = _expand_planar(d)
    from .fits import expand_fits
    d = expand_fits(d)
    if d.get("shroud") not in (None, False):
        # a generated exterior (shroud.py): the envelope becomes the design
        # space and its skin is preserved, so the rest parses as skin-and-rib
        from .shroud import expand_shroud
        d = expand_shroud(d, base_dir)
    # kept in raw: a study re-parsed from its raw (orientation, explore)
    # is not compensated twice
    fits = d.get("_fits") or []
    joints = d.pop("_joints", []) if isinstance(d.get("_joints"), list) else []
    if d.get("split") is not None:
        raise ValueError("this study has a split block: it is two parts. `loadpath split study.json --out dir` "
                         "writes a study per half (MCP loadpath_split); run those")
    ds = d.get("designSpace")
    if isinstance(ds, dict):
        ds = [ds]
    if not ds:
        raise ValueError("designSpace is required: one shape or a list of shapes")
    design_space = [shape_from_spec(s, base_dir) for s in ds]
    preserve = _regions(d.get("preserve"), "preserve", base_dir)
    ds_lo = np.min([s.bounds()[0] for s in design_space], axis=0)
    ds_hi = np.max([s.bounds()[1] for s in design_space], axis=0)
    keep_out = _regions(d.get("keepOut"), "keepOut", base_dir, bounds=(ds_lo, ds_hi))
    clash = {r.id for r in preserve} & {r.id for r in keep_out}
    if clash:
        raise ValueError(f"region ids must be unique across preserve and keepOut: {sorted(clash)}")

    supports = []
    for i, s in enumerate(d.get("supports") or []):
        region, shape = _target(s, f"supports[{i}]", base_dir)
        fix = s.get("fix", "all")
        pin = fix == "pin"
        if pin:
            if region is None and not hasattr(shape, "axis"):
                raise ValueError(f"supports[{i}]: a pin needs a cylinder (a bore keep-out or a cylinder shape)")
            fix_t = (False, False, False)
        elif fix == "all":
            fix_t = (True, True, True)
        else:
            fix_t = tuple(a in fix for a in "xyz") if isinstance(fix, str) else tuple(bool(v) for v in fix)
            if len(fix_t) != 3 or not any(fix_t):
                raise ValueError(f"supports[{i}].fix must be 'all', 'pin', axes like 'xz', or 3 booleans")
        caps = []
        for key in ("maxForce", "maxMoment"):
            v = s.get(key)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
                raise ValueError(f"supports[{i}].{key} must be a positive number ({'N' if key == 'maxForce' else 'N·mm'})")
            caps.append(None if v is None else float(v))
        if any(c is not None for c in caps) and not (d.get("goal") or {}).get("safetyFactor") \
                and (d.get("goal") or {}).get("maxDeflection") is None:
            raise ValueError(f"supports[{i}]: maxForce / maxMoment are held by the search for the lightest part, "
                             "which needs goal.safetyFactor or maxDeflection")
        supports.append(Support(region, shape, fix_t, pin, *caps))

    cases_in = d.get("loadCases")
    if cases_in is None and "loads" in d:
        cases_in = [{"name": "default", "loads": d["loads"]}]
    cases = []
    for ci, c in enumerate(cases_in or []):
        loads = []
        for li, ld in enumerate(c.get("loads") or []):
            region, shape = _target(ld, f"loadCases[{ci}].loads[{li}]", base_dir)
            label = f"loadCases[{ci}].loads[{li}]"
            p = ld.get("pressure", 0.0)
            if not isinstance(p, (int, float)) or not np.isfinite(p):
                raise ValueError(f"{label}.pressure is a gauge pressure in MPa (1 bar = 0.1 MPa)")
            if p:
                if region is None:
                    raise ValueError(f"{label}: a pressure acts in a keepOut (the fluid's volume); "
                                     "name it with 'region'")
                if any(k in ld for k in ("force", "moment", "torque", "distribution")):
                    raise ValueError(f"{label}: a pressure load takes no force, moment or distribution; "
                                     "put those in a load of their own")
                loads.append(Load(region, None, np.zeros(3), pressure=float(p)))
                continue
            f = np.asarray(ld.get("force", [0, 0, 0]), float).reshape(-1)
            m = np.asarray(ld.get("moment", ld.get("torque", [0, 0, 0])), float).reshape(-1)
            if f.shape != (3,) or m.shape != (3,) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(m)):
                raise ValueError(f"{label}: force [fx, fy, fz] (N) and moment [mx, my, mz] (N·mm) must be 3 numbers")
            if ld.get("mass") is not None:
                f, m = _mass_load(ld, f, m, region, shape, label, preserve, keep_out)
            if ld.get("at") is not None:
                m = _at_moment(ld, f, m, region, shape, label, preserve, keep_out)
            if not np.any(f) and not np.any(m):
                raise ValueError(f"{label} needs a nonzero force (N), moment (N·mm), mass (kg) or pressure (MPa)")
            dist = str(ld.get("distribution", "uniform"))
            if dist not in ("uniform", "bearing"):
                raise ValueError(f"{label}.distribution must be 'uniform' or 'bearing'")
            if dist == "bearing" and not np.any(f):
                raise ValueError(f"{label}: a bearing load needs a force")
            spread, cone = None, None
            if ld.get("direction") is not None:
                if ld["direction"] != "any":
                    raise ValueError(f"{label}.direction must be 'any' (the force's size, in every direction) "
                                     "or a plane, 'xy', 'xz' or 'yz' (any direction in it)")
                spread = "any"
            if ld.get("cone") is not None:
                cone = ld["cone"]
                if spread or not isinstance(cone, (int, float)) or not 0 < cone <= 180:
                    raise ValueError(f"{label}.cone is a half-angle in degrees (0-180] round the force's "
                                     "direction, and cannot go with direction 'any'")
                spread, cone = ("any", None) if cone >= 180 else ("cone", float(cone))
            if spread and not np.any(f):
                raise ValueError(f"{label}: direction 'any' or a cone needs a force (its size and, for a cone, "
                                 "its axis)")
            loads.append(Load(region, shape, f, m, dist, direction=spread, cone=cone))
        if not loads:
            raise ValueError(f"loadCases[{ci}] has no loads")
        if sum(ld.direction is not None for ld in loads) > 1:
            raise ValueError(f"loadCases[{ci}]: at most one load per case may have an unknown direction "
                             "(direction 'any' or a cone); put another in a load case of its own")
        life = {}
        for key in ("cycles", "hours"):
            if c.get(key) is not None:
                v = c[key]
                if not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                    raise ValueError(f"loadCases[{ci}].{key} must be a positive number "
                                     + ("(load cycles over the part's life)" if key == "cycles" else
                                        "(hours the load is held over the part's life)"))
                life[key] = float(v)
        if c.get("temperature") is not None:
            life["temperature"] = float(c["temperature"])
        if c.get("safetyFactor") is not None:
            v = c["safetyFactor"]
            if not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"loadCases[{ci}].safetyFactor is this case's own factor (> 0), in place of the "
                                 "goal's: a rare crash held to 1.25 beside service loads at 2")
            life["safety_factor"] = float(v)
        if c.get("impact") is not None:
            life["impact"] = _impact_case(c["impact"], ci, loads, bool(d.get("inertiaRelief")))
        cases.append(LoadCase(str(c.get("name") or f"case-{ci}"), loads, float(c.get("weight", 1.0)), **life))
    # A `feel` block brings its own loads and holds (feel.py), and a
    # `cooling` block its own heat (cooling.py): a study that only asks for
    # those needs neither load cases nor supports.
    feel_only = bool(d.get("feel") or d.get("cooling"))
    if not cases and not feel_only:
        raise ValueError("at least one load case is required (loadCases: [{name, loads}])")
    pressurized = {ld.region for c in cases for ld in c.loads if ld.pressure}
    ko_ids = {r.id for r in keep_out}
    for rid in sorted(pressurized - ko_ids):
        raise ValueError(f"pressure on {rid!r}: a pressure acts in a keepOut (the fluid's volume); "
                         f"keepOut ids: {sorted(ko_ids)}")
    relief = d.get("inertiaRelief")
    inertia_masses, inertia_bodies = [], []
    if isinstance(relief, dict):
        unknown = set(relief) - {"masses"}
        if unknown:
            raise ValueError(f"inertiaRelief takes masses (what the free part carries); unknown: {sorted(unknown)}")
        for i, m in enumerate(relief.get("masses") or []):
            label = f"inertiaRelief.masses[{i}]"
            if not isinstance(m, dict) or not isinstance(m.get("mass"), (int, float)) or m["mass"] <= 0:
                raise ValueError(f"{label}.mass is the carried mass in kg (> 0): a motor, a battery, a camera")
            region, shape = _target(m, label, base_dir)
            if m.get("at") is None:
                if m.get("inertia") is not None:
                    raise ValueError(f"{label}.inertia needs `at`, the carried body's centre of mass")
                inertia_masses.append((region, shape, float(m["mass"])))
                continue
            at = np.asarray(m["at"], float).reshape(-1)
            if at.shape != (3,) or not np.all(np.isfinite(at)):
                raise ValueError(f"{label}.at is the carried body's centre of mass [x, y, z] (mm)")
            inertia = np.zeros(3)
            if m.get("inertia") is not None:
                inertia = np.asarray(m["inertia"], float)
                if inertia.shape not in ((3,), (3, 3)) or np.any(np.diag(inertia) < 0 if inertia.ndim == 2 else
                                                                   inertia < 0):
                    raise ValueError(f"{label}.inertia is the body's moments of inertia about its centre, "
                                     "[Ixx, Iyy, Izz] or a 3x3 (kg mm^2)")
            elif m.get("size") is not None:
                a, b, c = (float(v) for v in m["size"])
                kg = float(m["mass"])
                inertia = kg / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
            inertia_bodies.append((region, shape, float(m["mass"]), at, inertia))
        relief = True
    elif relief is not None and not isinstance(relief, bool):
        raise ValueError("inertiaRelief is true, false or {masses: [{region, mass}]}")
    if relief:
        if supports:
            raise ValueError("inertiaRelief is for a part nothing holds (a drone frame, a thrown or dropped part, a "
                             "satellite bracket): its own inertia balances the loads. Drop the supports, or drop "
                             "inertiaRelief")
        if d.get("gravity") is not None:
            raise ValueError("inertiaRelief with gravity: a free part only falls under its own weight, which "
                             "stresses nothing; give the accelerations as loads (a carried mass, a thrust)")
        if feel_only:
            raise ValueError("inertiaRelief does not go with feel or cooling blocks yet")
        if any(ld.direction is not None for c in cases for ld in c.loads):
            raise ValueError("inertiaRelief takes forces of a known direction: give each direction as a load case")
        if (d.get("goal") or {}).get("method") == "constrained":
            raise ValueError("inertiaRelief runs with the search (goal.method search or auto)")
    if any(c.impact is not None for c in cases) and (d.get("goal") or {}).get("method") == "constrained":
        raise ValueError("impact load cases run with the search (goal.method search or auto): the force changes "
                         "with the design")
    if not supports and not feel_only and not relief and not all(ld.pressure for c in cases for ld in c.loads):
        raise ValueError("at least one support is required (where the part is held); only a study whose "
                         "every load is a pressure may leave supports out, or one with inertiaRelief: true (a "
                         "free part whose own inertia balances its loads)")
    gravity = d.get("gravity")
    if gravity is not None and not supports:
        raise ValueError("gravity needs supports: a part's own weight has to be held somewhere")
    if gravity is not None:
        gravity = np.asarray(gravity, float).reshape(-1)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must be an acceleration vector in m/s², e.g. [0, 0, -9.81]")
    thermal = _thermal(d.get("thermal"), base_dir)

    g = d.get("goal") or {"volumeFraction": 0.3}
    goal = Goal(
        volume_fraction=g.get("volumeFraction"),
        safety_factor=g.get("safetyFactor"),
        max_deflection=g.get("maxDeflection"),
        method=str(g.get("method", "auto")),
        confirm=str(g.get("confirm", "auto")),
        outcomes=[float(v) for v in g.get("outcomes") or []],
        minimize=str(g.get("minimize", "mass")),
        restarts=int(g.get("restarts", 1)),
        buckling=bool(g.get("buckling", True)),
        clearances=_clearances(g.get("clearances"), base_dir),
    )
    if goal.restarts < 1:
        raise ValueError("goal.restarts must be 1 or more")
    if goal.minimize not in ("mass", "printTime", "filament"):
        raise ValueError("goal.minimize must be 'mass', 'printTime' or 'filament'")
    if goal.volume_fraction is None and goal.safety_factor is None and goal.max_deflection is None:
        raise ValueError("goal needs volumeFraction (0..1), safetyFactor (>1) and/or maxDeflection (mm)")
    if goal.method not in ("search", "constrained", "auto"):
        raise ValueError("goal.method must be 'search', 'constrained' or 'auto'")
    if goal.confirm not in ("auto", "voxel", "tet"):
        raise ValueError("goal.confirm must be 'auto', 'voxel' or 'tet'")
    if goal.max_deflection is not None and goal.max_deflection <= 0:
        raise ValueError("goal.maxDeflection must be a positive distance in mm")
    if goal.volume_fraction is not None and not (0.02 <= goal.volume_fraction <= 0.95):
        raise ValueError("goal.volumeFraction must be between 0.02 and 0.95")
    if goal.safety_factor is not None and goal.safety_factor <= 0:
        raise ValueError("goal.safetyFactor must be positive")
    for v in goal.outcomes:
        if not (0.02 <= v <= 0.95):
            raise ValueError("goal.outcomes values must be between 0.02 and 0.95")
    if goal.clearances and goal.safety_factor is None and goal.max_deflection is None:
        raise ValueError("goal.clearances are held by the search for the lightest passing part: give the goal a "
                         "safetyFactor or maxDeflection too")
    names = {c.get("name") for c in d.get("loadCases") or [] if isinstance(c, dict)}
    for i, c in enumerate(goal.clearances):
        for n in c.cases:
            if n not in names:
                raise ValueError(f"goal.clearances[{i}].cases names {n!r}, which is not a load case")

    mat_in = d.get("material", "pla")
    materials = [material_from_spec(m) for m in mat_in] if isinstance(mat_in, list) else [material_from_spec(mat_in)]
    if not materials:
        raise ValueError("material list is empty")

    sym = [str(a).lower() for a in d.get("symmetry") or []]
    if any(a not in ("x", "y", "z") for a in sym):
        raise ValueError("symmetry entries must be 'x', 'y' or 'z' (planes through the design space center)")

    res = d.get("resolution") or {}
    pitch = res.get("pitch")
    max_el = int(res.get("maxElements") or DEFAULT_MAX_ELEMENTS)
    if max_el > MAX_ELEMENTS_CEILING:
        raise ValueError(f"resolution.maxElements is capped at {MAX_ELEMENTS_CEILING}")
    if pitch is not None and float(pitch) <= 0:
        raise ValueError("resolution.pitch must be positive (mm)")
    ps = res.get("pitchStarts", "auto")
    if ps != "auto" and (isinstance(ps, bool) or not isinstance(ps, int) or not 1 <= ps <= 5):
        raise ValueError("resolution.pitchStarts must be 'auto' or an integer 1-5")

    min_member = d.get("minMember")
    process = _process(d.get("process"))
    process = _card_process(process, materials[0], d.get("process"))
    process = _calibrated_process(process, mat_in, d.get("process"))
    print_direction = parse_build_direction(d.get("printDirection"))
    if process is not None:
        # the process sets what a study left open: overhang control along the
        # build direction, and a minimum member it can actually make
        if print_direction is None and process.overhangs and process.build not in (None, "auto"):
            print_direction = process.build
        floor = 2.0 * process.min_feature
        min_member = max(float(min_member), floor) if min_member is not None else floor
    min_hole = d.get("minHole")
    if min_hole is not None and not (isinstance(min_hole, (int, float)) and min_hole > 0):
        raise ValueError("minHole is the narrowest hole or gap allowed, in mm (> 0)")
    if min_hole is None and process is not None:
        min_hole = process.min_hole
    # Walls round keep-outs and the design space's skin are preserved
    # material: fixed, pinned solid, with only what is behind them designed
    # (skin-and-rib). Every pressurized keep-out has one: the pressure needs
    # a surface to push on that the optimizer cannot open.
    from .shapes import Shell, Union, grow

    shells = []
    clip = design_space[0] if len(design_space) == 1 else Union(design_space)
    taken = {r.id for r in preserve} | {r.id for r in keep_out}
    for r in keep_out:
        t = r.wall
        if t is None and r.id in pressurized:
            if min_member is None:
                raise ValueError(f"keepOut {r.id!r} is pressurized: give it a 'wall' (mm), the thickness of "
                                 "the skin the pressure acts on, or set minMember")
            t = r.wall = float(min_member)
        if t is None:
            continue
        rid = f"{r.id}-wall"
        if rid in taken:
            raise ValueError(f"region id {rid!r} is taken; it names keepOut {r.id!r}'s wall")
        preserve.append(Region(rid, Shell(grow(r.shape, t), r.shape, clip)))
        shells.append((rid, t))
        if "bore" not in next((k for k in d.get("keepOut") or [] if k.get("id") == r.id), {}):
            # a pressure vessel's cavity is closed on purpose
            r.bore = r.bore and r.id not in pressurized
    if d.get("skin") is not None:
        t, shape = _skin(d["skin"], design_space, base_dir)
        if "skin" in taken:
            raise ValueError("region id 'skin' is taken; it names the design space's skin")
        preserve.append(Region("skin", shape))
        shells.append(("skin", t))
    from .durability import durability_from_spec
    durability = durability_from_spec(d.get("durability"))
    from .manufacture import parse_casting, parse_milling, parse_pattern

    if d.get("milling") is not None and d.get("casting") is not None:
        raise ValueError("a study is milled or cast, not both: give milling or casting")
    access = parse_milling(d.get("milling")) or parse_casting(d.get("casting"))
    if access is not None and print_direction is not None:
        raise ValueError(f"{access.kind} and overhang control (printDirection, or a process that prints with "
                         "overhang limits) do not go together: drop one")
    max_member = d.get("maxMember")
    if max_member is not None:
        if not (isinstance(max_member, (int, float)) and max_member > 0):
            raise ValueError("maxMember is the thickest member allowed, in mm (> 0)")
        if min_member is not None and max_member < 2.0 * float(min_member):
            raise ValueError(f"maxMember ({max_member:g}mm) must be at least twice minMember "
                             f"({float(min_member):g}mm) to leave the optimizer room")
    vibration = _vibration(d.get("vibration"), goal, supports, base_dir, free=bool(relief))
    mass_properties = _mass_properties(d.get("massProperties"), goal)
    st = Study(
        name=str(d.get("name") or "study"),
        design_space=design_space,
        preserve=preserve,
        keep_out=keep_out,
        supports=supports,
        load_cases=cases,
        material=materials[0],
        materials=materials,
        goal=goal,
        min_member=float(min_member) if min_member is not None else None,
        symmetry=sym,
        pitch=float(pitch) if pitch is not None else None,
        max_elements=max_el,
        max_iterations=int(d.get("maxIterations", 120)),
        gravity=gravity,
        thermal=thermal,
        print_direction=print_direction,
        extrusion=_extrusion_axis(d.get("extrusion")),
        access=access,
        pattern=parse_pattern(d.get("pattern")),
        max_member=float(max_member) if max_member is not None else None,
        process=process,
        shells=shells,
        durability=durability,
        joints=joints,
        slicer=_slicer(d.get("slicer"), process, base_dir, goal),
        min_hole=float(min_hole) if min_hole is not None else None,
        drains=_drains(d.get("drains", "auto"), process),
        build_sim=_build_sim(d.get("buildSim"), process, d.get("material")),
        fits=fits,
        vibration=vibration,
        mass_properties=mass_properties,
        aero=_aero(d.get("aero")),
        inertia_masses=inertia_masses,
        inertia_bodies=inertia_bodies,
        raw=d,
    )
    if d.get("multiMaterial") is not None:
        from .multimat import parse_multimaterial

        st.multi_material = parse_multimaterial(d["multiMaterial"], st, base_dir)
        if "material" not in d:
            st.material = st.materials[0] = st.multi_material.materials[0]
    # mesh paths in blocks read later (switch.output.bearsOn, switch.clearance)
    st.base_dir = base_dir
    for i, c in enumerate(goal.clearances):
        if c.region is not None:
            try:
                st.region_shape(c.region)
            except ValueError as exc:
                raise ValueError(f"goal.clearances[{i}]: {exc}") from None
    return st
