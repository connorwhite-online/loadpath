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
# Hard ceiling regardless of what a study asks for: memory and minutes both
# grow faster than linearly past this on a modest CPU budget.
MAX_ELEMENTS_CEILING = 400_000


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


@dataclass
class LoadCase:
    name: str
    loads: list[Load]
    weight: float = 1.0


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
    # Extra fixed fractions to return alongside, for a trade-off gallery.
    outcomes: list[float] = field(default_factory=list)


# How the part will be made. Each process sets what the design must respect:
# whether layers make it weaker across them (FDM bonds layer to layer at a
# fraction of the in-plane strength), whether overhangs need control, and
# the smallest feature it holds. Values are typical published ranges, chosen
# conservative; override any of them in the study.
PROCESSES = {
    "fdm": {"interlayer": 0.6, "minFeature": 0.8, "overhangs": True,
            "note": "filament: layer bonds ~50-70% of in-plane strength; self-supporting at 45 degrees"},
    "sls": {"interlayer": 0.9, "minFeature": 0.8, "overhangs": False,
            "note": "powder bed (PA12, PA11): nearly isotropic, no supports; leave powder escape holes in closed cavities"},
    "mjf": {"interlayer": 0.95, "minFeature": 0.8, "overhangs": False,
            "note": "HP Multi Jet Fusion: as SLS, slightly more isotropic"},
    "sla": {"interlayer": 1.0, "minFeature": 0.5, "overhangs": False,
            "note": "resin vat: isotropic once cured; supports are added by the slicer"},
    "volumetric": {"interlayer": 1.0, "minFeature": 0.3, "overhangs": False,
                   "note": "volumetric / tomographic resin printing: no layers, no supports"},
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
    raw_build = spec.get("buildDirection", "+z" if kind == "fdm" else None)
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
    return Process(kind, build, inter, float(spec.get("minFeature", base["minFeature"])),
                   bool(spec.get("overhangs", base["overhangs"])),
                   anisotropic=_anisotropic(spec.get("anisotropic", False)), **stiff)


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
    # Every material to try (a list in the study); `material` is the first.
    materials: list = field(default_factory=list)
    # How it is made (spec.PROCESSES): layer weakness, overhangs, min feature.
    process: Process | None = None
    # (region id, thickness mm) of every wall and skin pinned solid
    shells: list = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

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
            loads.append(Load(region, shape, f, m, dist))
        if not loads:
            raise ValueError(f"loadCases[{ci}] has no loads")
        cases.append(LoadCase(str(c.get("name") or f"case-{ci}"), loads, float(c.get("weight", 1.0))))
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
        outcomes=[float(v) for v in g.get("outcomes") or []],
    )
    if goal.volume_fraction is None and goal.safety_factor is None and goal.max_deflection is None:
        raise ValueError("goal needs volumeFraction (0..1), safetyFactor (>1) and/or maxDeflection (mm)")
    if goal.method not in ("search", "constrained"):
        raise ValueError("goal.method must be 'search' or 'constrained'")
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
    print_direction = parse_build_direction(d.get("printDirection"))
    if process is not None:
        # the process sets what a study left open: overhang control along the
        # build direction, and a minimum member it can actually make
        if print_direction is None and process.overhangs and process.build not in (None, "auto"):
            print_direction = process.build
        floor = 2.0 * process.min_feature
        min_member = max(float(min_member), floor) if min_member is not None else floor
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
        process=process,
        shells=shells,
        raw=d,
    )
