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


@dataclass
class Support:
    region: str | None
    shape: Shape | None
    fix: tuple[bool, bool, bool] = (True, True, True)
    # A pin through a bore: held radially and axially, free to rotate about
    # the bore's axis (a bolt or a shaft in a plain hole).
    pin: bool = False


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
    # constraints, corrected against the verified mesh (constrained.py).
    method: str = "search"
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


# How the part will be made. Each process sets what the design must respect:
# whether layers make it weaker across them (FDM bonds layer to layer at a
# fraction of the in-plane strength), whether overhangs need control, and
# the smallest feature it holds, and the smallest hole that stays open (unfused
# powder sinters shut in a narrow SLS/MJF hole, resin cures shut in a narrow
# SLA one, a nozzle bridges over a narrow FDM one). Values are typical
# published ranges, chosen conservative; override any of them in the study.
PROCESSES = {
    "fdm": {"interlayer": 0.6, "minFeature": 0.8, "minHole": 2.0, "overhangs": True,
            "note": "filament: layer bonds ~50-70% of in-plane strength; self-supporting at 45 degrees"},
    "sls": {"interlayer": 0.9, "minFeature": 0.8, "minHole": 2.0, "overhangs": False,
            "note": "powder bed (PA12, PA11): nearly isotropic, no supports; leave powder escape holes in closed cavities"},
    "mjf": {"interlayer": 0.95, "minFeature": 0.8, "minHole": 2.0, "overhangs": False,
            "note": "HP Multi Jet Fusion: as SLS, slightly more isotropic"},
    "sla": {"interlayer": 1.0, "minFeature": 0.5, "minHole": 1.0, "overhangs": False,
            "note": "resin vat: isotropic once cured; supports are added by the slicer"},
    "dlp": {"interlayer": 1.0, "minFeature": 0.5, "minHole": 1.0, "overhangs": False,
            "note": "resin vat, whole-layer exposure (DLP/LCD): as SLA; large cross-sections peel hard"},
    "volumetric": {"interlayer": 1.0, "minFeature": 0.3, "minHole": 0.6, "overhangs": False,
                   "note": "volumetric / tomographic resin printing: no layers, no supports"},
    "lpbf": {"interlayer": 0.85, "minFeature": 0.4, "minHole": 1.0, "overhangs": True,
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
    return Process(kind, build, inter, float(spec.get("minFeature", base["minFeature"])),
                   bool(spec.get("overhangs", base["overhangs"])),
                   anisotropic=_anisotropic(spec.get("anisotropic", False)),
                   escape_hole=float(esc) if esc is not None else None,
                   min_hole=float(spec.get("minHole", base["minHole"])), **stiff)


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
    unless the study sets its own; a calibrated card then overrides both."""
    if process is None or not mat.measured or mat.measured != process.kind:
        return process
    spec = spec if isinstance(spec, dict) else {}
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
    return process


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
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def length_scale(self) -> float:
        """The member size the density filter enforces (mm): minMember, else
        DEFAULT_MIN_MEMBER. A grid fine enough to resolve a sub-millimetre
        sheet would otherwise build one (the 845k-cell arm link with no
        minMember came out a 1 mm skin sealed round its cable channel)."""
        return float(self.min_member) if self.min_member else DEFAULT_MIN_MEMBER

    @property
    def self_equilibrated(self) -> bool:
        """No supports: every load is a pressure, held by pressure.stabilize."""
        return not self.supports

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


def _regions(items, label, base_dir):
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
        out.append(Region(rid, shape_from_spec(item["shape"], base_dir), bool(item.get("bore", True)),
                          float(wall) if wall is not None else None))
    return out


_FACES = {f"{s}{a}": (i, 1 if s == "+" else -1) for i, a in enumerate("xyz") for s in "+-"}


def _skin(v, design):
    """Top-level `skin`: a thickness (mm) or {thickness, open: ['-z', ..]}.
    Returns (thickness, the skin shape)."""
    from .shapes import Shell, Union, shrink

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
    return float(t), Shell(ds, shrink(ds, float(t), faces))


def _target(item, label, base_dir):
    if "region" in item:
        return str(item["region"]), None
    if "shape" in item:
        return None, shape_from_spec(item["shape"], base_dir)
    raise ValueError(f"{label} needs a 'region' id or an inline 'shape'")


def _extrusion_axis(v):
    if v is None:
        return None
    a = str(v).strip().lower().lstrip("+-")
    if a not in ("x", "y", "z"):
        raise ValueError("extrusion must be 'x', 'y' or 'z' (the axis the profile is extruded along)")
    return "xyz".index(a)


def study_from_dict(d: dict, base_dir: str | None = None) -> Study:
    """Validate and parse a study. Every error names the field to fix."""
    if not isinstance(d, dict):
        raise ValueError("study must be a JSON object")
    from .fasteners import expand_fasteners
    d = expand_fasteners(d)
    joints = d.pop("_joints", []) if isinstance(d.get("_joints"), list) else []
    ds = d.get("designSpace")
    if isinstance(ds, dict):
        ds = [ds]
    if not ds:
        raise ValueError("designSpace is required: one shape or a list of shapes")
    design_space = [shape_from_spec(s, base_dir) for s in ds]
    preserve = _regions(d.get("preserve"), "preserve", base_dir)
    keep_out = _regions(d.get("keepOut"), "keepOut", base_dir)
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
        supports.append(Support(region, shape, fix_t, pin))

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
            if not np.any(f) and not np.any(m):
                raise ValueError(f"{label} needs a nonzero force (N), moment (N·mm) or pressure (MPa)")
            dist = str(ld.get("distribution", "uniform"))
            if dist not in ("uniform", "bearing"):
                raise ValueError(f"{label}.distribution must be 'uniform' or 'bearing'")
            if dist == "bearing" and not np.any(f):
                raise ValueError(f"{label}: a bearing load needs a force")
            spread, cone = None, None
            if ld.get("direction") is not None:
                if ld["direction"] != "any":
                    raise ValueError(f"{label}.direction must be 'any' (the force's size, in every direction)")
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
        cases.append(LoadCase(str(c.get("name") or f"case-{ci}"), loads, float(c.get("weight", 1.0)), **life))
    if not cases:
        raise ValueError("at least one load case is required (loadCases: [{name, loads}])")
    pressurized = {ld.region for c in cases for ld in c.loads if ld.pressure}
    ko_ids = {r.id for r in keep_out}
    for rid in sorted(pressurized - ko_ids):
        raise ValueError(f"pressure on {rid!r}: a pressure acts in a keepOut (the fluid's volume); "
                         f"keepOut ids: {sorted(ko_ids)}")
    if not supports and not all(ld.pressure for c in cases for ld in c.loads):
        raise ValueError("at least one support is required (where the part is held); only a study whose "
                         "every load is a pressure may leave supports out")
    gravity = d.get("gravity")
    if gravity is not None and not supports:
        raise ValueError("gravity needs supports: a part's own weight has to be held somewhere")
    if gravity is not None:
        gravity = np.asarray(gravity, float).reshape(-1)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must be an acceleration vector in m/s², e.g. [0, 0, -9.81]")

    g = d.get("goal") or {"volumeFraction": 0.3}
    goal = Goal(
        volume_fraction=g.get("volumeFraction"),
        safety_factor=g.get("safetyFactor"),
        max_deflection=g.get("maxDeflection"),
        method=str(g.get("method", "search")),
        confirm=str(g.get("confirm", "auto")),
        outcomes=[float(v) for v in g.get("outcomes") or []],
        minimize=str(g.get("minimize", "mass")),
    )
    if goal.minimize not in ("mass", "printTime", "filament"):
        raise ValueError("goal.minimize must be 'mass', 'printTime' or 'filament'")
    if goal.volume_fraction is None and goal.safety_factor is None and goal.max_deflection is None:
        raise ValueError("goal needs volumeFraction (0..1), safetyFactor (>1) and/or maxDeflection (mm)")
    if goal.method not in ("search", "constrained"):
        raise ValueError("goal.method must be 'search' or 'constrained'")
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
        t, shape = _skin(d["skin"], design_space)
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
    return Study(
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
        raw=d,
    )
