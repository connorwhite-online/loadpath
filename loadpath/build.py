"""
Build a part from its spec: drawn interfaces, a body between them, checked.

`loadpath run` grows structure for known loads. Many product parts are not
decided by loads: an enclosure is its exterior, the solids it has to mate
with, a wall that prints, and a weight. `build` makes that part in one
run, and checks it the same way every time:

- the **exterior** (the study's `designSpace`) is the designer's and stays
  exact: the part's outside *is* the exterior's own surface, never a
  re-meshed copy of it;
- **interfaces** are exact solids (`{"type": "brep"}` shapes drawn in
  build123d, STEP, or any loadpath shape). `add` ones are material (a
  boss, a rail, a cradle); `cut` ones are space (a port, a pocket, an
  opening), optionally with a `wall` of material kept round them. They are
  the source of truth: added and cut with exact booleans after everything
  else, so no field ever redraws them;
- **components** are the bought parts, as solids. Each is cut out with its
  `clearance`, and checked: nothing of the part inside it, the clearance
  kept, its way in (`insertFrom`) free, inside the exterior by `keepIn`;
- the **body** between them is a distance field: the skin (the exterior
  shrunk by a wall that can vary by zone), rounded into every `add`
  interface and every cut's wall with a fillet of `blend` mm. The field
  only describes the cavity's surface, strictly inside the exterior and
  inside each interface it meets by a hair, so the exact booleans that
  follow cross it cleanly: the seam between a drawn solid and the body is
  a fillet, not a row of patches;
- **targetMass**: the walls named in `adjust` are thickened (or thinned)
  between their limits until the part, with its components, weighs the
  target;
- then the checks (buildcheck.py): closed and one piece, the exterior
  untouched outside the declared cuts, every interface present or open,
  every component clear, the process's limits (Materialize's published
  minimums are the reference for printed parts), the mass, the asserted
  features (`"fillet"`: a port mouth's radius, read from the kernel and
  again from the delivered mesh), and the references (reference.py).

    loadpath build spec.json --out out/

writes out/part.stl, out/checks.json (every check, pass or fail, with its
number), out/report.json, and images: views.png, a close-up sheet per
interface, a side-by-side per reference. The exit code is 0 only when
every check passes.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .brepio import Brep, load_brep

__all__ = ["BuildSpec", "Component", "Interface", "build_spec", "run_build", "has_build"]

INSET = 0.05            # mm the cavity's surface sits inside an added solid, so their surfaces cross
DEFAULT_BLEND = 0.5     # mm fillet where the body meets a drawn solid


@dataclass
class Interface:
    id: str
    op: str                      # "add" | "cut"
    brep: Brep
    blend: float = DEFAULT_BLEND
    wall: float = 0.0            # cut: mm of material kept round it
    proud: bool = False          # add: may stand outside the exterior
    spec: dict = field(default_factory=dict)


@dataclass
class Component:
    id: str
    brep: Brep
    clearance: float = 0.0
    mass: float = 0.0            # g
    keep_in: float | None = None
    insert: tuple | None = None  # (unit vector the part leaves along, length mm)
    spec: dict = field(default_factory=dict)


@dataclass
class Zone:
    id: str
    shape: object
    thickness: float
    blend: float
    lo: float | None = None      # targetMass limits, when this zone is adjusted
    hi: float | None = None


@dataclass
class BuildSpec:
    name: str
    exterior: object             # Shape
    skin: float
    zones: list
    interfaces: list
    components: list
    material: object
    process: str | None
    target: dict | None
    asserts: list
    references: list
    doc: dict
    base_dir: str | None
    skin_lo: float | None = None
    skin_hi: float | None = None
    build_direction: object = None
    solid: bool = False                # skin "solid": the designSpace is the part, no cavity
    _ext_man: object = None
    _ext_mesh: object = None

    def exterior_manifold(self):
        if self._ext_man is None:
            self._ext_man = self.exterior.to_manifold()
        return self._ext_man

    def exterior_mesh(self):
        if self._ext_mesh is None:
            from .shapes import trimesh_from_manifold

            self._ext_mesh = trimesh_from_manifold(self.exterior_manifold())
        return self._ext_mesh

    def interface(self, name):
        for it in self.interfaces:
            if it.id == name:
                return it
        raise ValueError(f"no interface {name!r} (have: {', '.join(i.id for i in self.interfaces) or 'none'})")

    def component(self, name):
        for c in self.components:
            if c.id == name:
                return c
        raise ValueError(f"no component {name!r} (have: {', '.join(c.id for c in self.components) or 'none'})")

    def adjustable(self) -> bool:
        return self.skin_hi is not None or any(z.hi is not None for z in self.zones)


def has_build(doc: dict) -> bool:
    """Whether a study document carries what `build` reads."""
    return any(doc.get(k) for k in ("interfaces", "components", "targetMass", "assert", "references"))


_AXES = {"x": (1.0, 0, 0), "y": (0, 1.0, 0), "z": (0, 0, 1.0)}


def _direction(v, label):
    if isinstance(v, str):
        s = v.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        a = s.lstrip("+-")
        if a not in _AXES:
            raise ValueError(f"{label} is a direction like '-z' or a vector, got {v!r}")
        return sign * np.asarray(_AXES[a])
    d = np.asarray(v, float)
    if d.shape != (3,) or np.linalg.norm(d) < 1e-9:
        raise ValueError(f"{label} is a direction like '-z' or a vector [x, y, z]")
    return d / np.linalg.norm(d)


def _brep_of(shape_spec, base_dir, label):
    if shape_spec is None:
        raise ValueError(f"{label} needs a shape")
    if isinstance(shape_spec, dict) and shape_spec.get("type") == "brep":
        src = shape_spec.get("path") or shape_spec.get("source")
        if not src:
            raise ValueError(f"{label}: a brep shape needs 'path' ('file.py:name', a STEP file or a mesh file)")
        return load_brep(src, base_dir, tolerance=float(shape_spec.get("tolerance", 0.005)))
    if isinstance(shape_spec, dict) and shape_spec.get("type") == "mesh" and shape_spec.get("path") \
            and str(shape_spec["path"]).lower().endswith((".step", ".stp")):
        return load_brep(shape_spec["path"], base_dir)
    return load_brep(shape_spec, base_dir)


def build_spec(doc: dict, base_dir: str | None = None) -> BuildSpec:
    """Read a study document's build blocks. Raises ValueError with a
    message an agent can act on."""
    from .materials import material_from_spec
    from .shapes import Union, shape_from_spec

    if not isinstance(doc, dict):
        raise ValueError("a build spec is a JSON object (a study with interfaces, components, targetMass, assert)")
    ds = doc.get("designSpace")
    if ds is None:
        raise ValueError("designSpace is the exterior the part is built inside (a mesh, STEP, brep or sdf shape)")
    shapes = [shape_from_spec(x, base_dir) for x in (ds if isinstance(ds, list) else [ds])]
    exterior = shapes[0] if len(shapes) == 1 else Union(shapes)

    skin = doc.get("skin")
    if skin is None:
        raise ValueError("skin is the wall the body keeps of the exterior: a thickness in mm, "
                         "{thickness, zones: [{id, shape, thickness, blend}]}, or \"solid\" for a part "
                         "drawn whole (a bracket, a mount: the designSpace is the part)")
    solid = isinstance(skin, str) and skin.strip().lower() == "solid"
    if isinstance(skin, str) and not solid:
        raise ValueError(f"skin is a thickness in mm, an object, or \"solid\"; got {skin!r}")
    if solid:
        # no cavity: the walls fill the exterior (assemble skips the field)
        lo, hi = exterior.bounds()
        skin = {"thickness": float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))}
    if isinstance(skin, (int, float)):
        skin = {"thickness": skin}
    t0 = skin.get("thickness")
    if not isinstance(t0, (int, float)) or t0 <= 0:
        raise ValueError("skin.thickness is the wall in mm (> 0)")
    zones = []
    for j, z in enumerate(skin.get("zones") or []):
        if not isinstance(z, dict) or "shape" not in z or not isinstance(z.get("thickness"), (int, float)):
            raise ValueError(f"skin.zones[{j}] is {{id?, shape, thickness, blend?}}")
        zones.append(Zone(str(z.get("id", f"zone-{j + 1}")), shape_from_spec(z["shape"], base_dir),
                          float(z["thickness"]), float(z.get("blend", 2.0))))

    seen = set()

    def ident(d, label):
        i = d.get("id")
        if not isinstance(i, str) or not i:
            raise ValueError(f"{label} needs an id")
        if i in seen:
            raise ValueError(f"{label}: id {i!r} is used twice (interfaces and components share one namespace)")
        seen.add(i)
        return i

    interfaces = []
    for j, it in enumerate(doc.get("interfaces") or []):
        label = f"interfaces[{j}]"
        if not isinstance(it, dict):
            raise ValueError(f"{label} is {{id, shape, op: 'add' | 'cut', blend?, wall?}}")
        i = ident(it, label)
        op = it.get("op")
        if op not in ("add", "cut"):
            raise ValueError(f"{label} ({i}): op is 'add' (material) or 'cut' (space)")
        if it.get("wall") and op != "cut":
            raise ValueError(f"{label} ({i}): wall is the material kept round a cut; an 'add' interface is material")
        interfaces.append(Interface(i, op, _brep_of(it.get("shape"), base_dir, f"{label} ({i})"),
                                    blend=float(it.get("blend", DEFAULT_BLEND)), wall=float(it.get("wall", 0.0)),
                                    proud=bool(it.get("proud", False)), spec=it))
    for j, op in enumerate(skin.get("openings") or []):
        # a skin opening is a cut through the skin: the same thing here
        i = f"opening-{j + 1}"
        seen.add(i)
        interfaces.append(Interface(i, "cut", _brep_of(op.get("shape"), base_dir, f"skin.openings[{j}]"),
                                    blend=0.0, spec=op))

    components = []
    for j, c in enumerate(doc.get("components") or []):
        label = f"components[{j}]"
        if not isinstance(c, dict):
            raise ValueError(f"{label} is {{id, shape, clearance?, mass?, keepIn?, insertFrom?}}")
        i = ident(c, label)
        ins = None
        if c.get("insertFrom") is not None:
            v = c["insertFrom"]
            if isinstance(v, dict):
                ins = (_direction(v.get("toward"), f"{label}.insertFrom.toward"), float(v.get("length", 0)) or None)
            else:
                ins = (_direction(v, f"{label}.insertFrom"), None)
        components.append(Component(i, _brep_of(c.get("shape"), base_dir, f"{label} ({i})"),
                                    clearance=float(c.get("clearance", 0.0)), mass=float(c.get("mass", 0.0)),
                                    keep_in=None if c.get("keepIn") is None else float(c["keepIn"]), insert=ins,
                                    spec=c))

    mat = doc.get("material")
    if isinstance(mat, list):
        mat = mat[0]
    if mat is None:
        raise ValueError("material is needed (the part's mass and its process's limits come from it)")
    material = material_from_spec(mat)
    proc = doc.get("process")
    build_dir = None
    if isinstance(proc, dict):
        build_dir = proc.get("buildDirection")
        proc = proc.get("type")
    if proc is None and getattr(material, "processes", None):
        proc = material.processes[0]
    if proc == "none":
        proc = None

    spec = BuildSpec(name=str(doc.get("name", "part")), exterior=exterior, skin=float(t0), zones=zones,
                     interfaces=interfaces, components=components, material=material, process=proc,
                     target=None, asserts=list(doc.get("assert") or []), references=list(doc.get("references") or []),
                     doc=doc, base_dir=base_dir, build_direction=build_dir, solid=solid)
    if solid and spec.adjustable():
        raise ValueError("a solid part has no wall to move: targetMass.adjust needs a skin")

    tm = doc.get("targetMass")
    if tm is not None:
        if isinstance(tm, (int, float)):
            tm = {"grams": tm}
        g = tm.get("grams")
        if not isinstance(g, (int, float)) or g <= 0:
            raise ValueError("targetMass.grams is the mass to reach (g > 0)")
        tol = float(tm.get("tolerance", max(0.02 * g, 0.05)))
        of = tm.get("of", "assembly")
        if of not in ("assembly", "part"):
            raise ValueError("targetMass.of is 'assembly' (the part with its components' masses) or 'part'")
        adjust = tm.get("adjust")
        if adjust is None:
            # nothing named: the whole skin, from the wall given up to five of it
            adjust = [{"skin": True}]
        for j, a in enumerate(adjust):
            if a.get("skin"):
                spec.skin_lo = float(a.get("min", spec.skin))
                spec.skin_hi = float(a.get("max", 5.0 * spec.skin))
                if spec.skin_hi < spec.skin_lo:
                    raise ValueError(f"targetMass.adjust[{j}]: max is under min")
                continue
            zid = a.get("zone")
            match = [z for z in zones if z.id == zid]
            if not match:
                raise ValueError(f"targetMass.adjust[{j}] names {{skin: true}} or a zone id "
                                 f"(have: {', '.join(z.id for z in zones) or 'no zones'})")
            z = match[0]
            z.lo = float(a.get("min", z.thickness))
            z.hi = float(a.get("max", 5.0 * z.thickness))
            if z.hi < z.lo:
                raise ValueError(f"targetMass.adjust[{j}]: max is under min")
        spec.target = {"grams": float(g), "tolerance": tol, "of": of}

    ids = {i.id for i in interfaces} | {c.id for c in components}
    for j, a in enumerate(spec.asserts):
        if not isinstance(a, dict) or "type" not in a:
            raise ValueError(f"assert[{j}] is {{type, ...}} (surface, fillet, wall, clearance, keepIn, mass, open, present, distance)")
        for key in ("interface", "component"):
            if a.get(key) is not None and a[key] not in ids:
                raise ValueError(f"assert[{j}] names {key} {a[key]!r}, which the spec does not have")
    return spec


# ---- the body's field ----------------------------------------------------------------

def _round_union(a, b, r):
    """min(a, b) with the corner between them filled by a fillet of radius r."""
    if r <= 0:
        return np.minimum(a, b)
    u = np.maximum(r - a, 0.0)
    v = np.maximum(r - b, 0.0)
    return np.maximum(r, np.minimum(a, b)) - np.sqrt(u * u + v * v)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class BodyField:
    """The fields a build is made of, at one setting `s` in [0, 1] of the
    adjustable walls (0: their minimum, 1: their maximum).

    `cavity(P)` is negative inside the void the body leaves (what is cut
    from the exterior); `part(P)` is negative inside the finished part
    (the same construction with the cuts, for volumes without a mesh)."""

    chunk = None

    def __init__(self, spec: BuildSpec, s: float | None = None):
        from .shapes import MeshShape

        self.spec = spec
        self.s = s
        self.ext = spec.exterior
        deepest = max([spec.skin, spec.skin_hi or 0.0] + [max(z.thickness, z.hi or 0.0) for z in spec.zones])
        if isinstance(self.ext, MeshShape):
            self.ext.exact_band(deepest + 1.0)
        self.adds = [(i.brep.as_shape(), i.blend, i.proud) for i in spec.interfaces if i.op == "add"]
        # a cut's wall is kept round what of it is inside the exterior: a
        # tool runs on outside the part (a block to round a mouth against),
        # and a wall round that would be a second skin a hair off the first
        self.walls = []
        for i in spec.interfaces:
            if i.op == "cut" and i.wall > 0:
                from .shapes import trimesh_from_manifold

                inside = i.brep.manifold() ^ spec.exterior_manifold()
                if inside.is_empty():
                    continue
                m = trimesh_from_manifold(inside)
                m.merge_vertices()
                self.walls.append((MeshShape(m), i.wall, i.blend))
        self.cuts = [i.brep.as_shape() for i in spec.interfaces if i.op == "cut"]
        self.comps = [(c.brep.as_shape(), c.clearance) for c in spec.components]

    def at(self, s):
        out = BodyField.__new__(BodyField)
        out.__dict__.update(self.__dict__)
        out.s = s
        return out

    def thickness(self, P):
        sp, s = self.spec, self.s
        t0 = sp.skin
        if s is not None and sp.skin_hi is not None:
            t0 = sp.skin_lo + s * (sp.skin_hi - sp.skin_lo)
        t = np.full(len(P), float(t0))
        for z in sp.zones:
            tz = z.thickness
            if s is not None and z.hi is not None:
                tz = z.lo + s * (z.hi - z.lo)
            w = _smooth(0.5 - z.shape.sdf(P) / max(z.blend, 1e-6))
            t = t + w * (tz - t)
        return t

    def _solid(self, P, inset):
        """The material inside the exterior, exterior not yet applied."""
        e = self.ext.sdf(P)
        sol = -(e + self.thickness(P))
        for sh, blend, proud in self.adds:
            sol = _round_union(sol, sh.sdf(P) + inset, blend)
        for sh, wall, blend in self.walls:
            sol = _round_union(sol, sh.sdf(P) - wall, blend)
        return sol, e

    def cavity(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        sol, _ = self._solid(P, INSET)
        return -sol

    __call__ = cavity

    def part(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        sol, e = self._solid(P, 0.0)
        d = np.maximum(e, sol)
        for sh, _, proud in self.adds:
            if proud:
                d = np.minimum(d, sh.sdf(P))
        for sh in self.cuts:
            d = np.maximum(d, -sh.sdf(P))
        for sh, c in self.comps:
            d = np.maximum(d, -(sh.sdf(P) - c))
        return d


class _Grid:
    """The part's volume at any wall setting, from one sampling of every
    field on a grid: the mass solve's inner loop is then arithmetic."""

    def __init__(self, body: BodyField, pitch: float):
        sp = body.spec
        lo, hi = sp.exterior.bounds()
        for sh, _, proud in body.adds:
            if proud:
                a, b = sh.bounds()
                lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        self.pitch = float(pitch)
        ax = [np.arange(lo[i] - pitch, hi[i] + 1.5 * pitch, pitch) + 0.5 * pitch for i in range(3)]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        self.e = body.ext.sdf(P)
        near = self.e < pitch
        self.P = P[near]
        self.e = self.e[near]
        self.zone_w = [_smooth(0.5 - z.shape.sdf(self.P) / max(z.blend, 1e-6)) for z in sp.zones]
        self.adds = [(sh.sdf(self.P), blend, proud) for sh, blend, proud in body.adds]
        self.walls = [(sh.sdf(self.P) - wall, blend) for sh, wall, blend in body.walls]
        cut = np.full(len(self.P), np.inf)
        for sh in body.cuts:
            cut = np.minimum(cut, sh.sdf(self.P))
        for sh, c in body.comps:
            cut = np.minimum(cut, sh.sdf(self.P) - c)
        self.cut = cut
        self.spec = sp

    def volume(self, s) -> float:
        sp = self.spec
        t0 = sp.skin if sp.skin_hi is None else sp.skin_lo + s * (sp.skin_hi - sp.skin_lo)
        t = np.full(len(self.P), float(t0))
        for z, w in zip(sp.zones, self.zone_w):
            tz = z.thickness if z.hi is None else z.lo + s * (z.hi - z.lo)
            t = t + w * (tz - t)
        sol = -(self.e + t)
        for a, blend, _ in self.adds:
            sol = _round_union(sol, a, blend)
        for a, blend in self.walls:
            sol = _round_union(sol, a, blend)
        d = np.maximum(self.e, sol)
        for a, _, proud in self.adds:
            if proud:
                d = np.minimum(d, a)
        d = np.maximum(d, -self.cut)
        return float(np.clip(0.5 - d / self.pitch, 0.0, 1.0).sum() * self.pitch ** 3)


def _grown(brep: Brep, r: float):
    """A component's solid grown by its clearance, as a manifold."""
    import manifold3d as m3d

    if r <= 1e-9:
        return brep.manifold()
    cache = brep.__dict__.setdefault("_grown", {})
    key = round(float(r), 6)
    if key not in cache:
        man = None
        if len(brep.mesh.faces) <= 4000:
            try:
                man = brep.manifold().minkowski_sum(m3d.Manifold.sphere(r, 24))
            except Exception:  # noqa: BLE001 - fall back to the offset surface
                man = None
        if man is None or man.is_empty():
            from .shapes import Offset

            man = Offset(brep.as_shape(), r).to_manifold()
        cache[key] = man
    return cache[key]


def _union(mans):
    import manifold3d as m3d

    mans = [m for m in mans if m is not None and not m.is_empty()]
    if not mans:
        return m3d.Manifold()
    return mans[0] if len(mans) == 1 else m3d.Manifold.batch_boolean(mans, m3d.OpType.Add)


def assemble(spec: BuildSpec, body: BodyField, pitch: float, progress=None):
    """Mesh the cavity and do the exact booleans. Returns (part manifold,
    info, the solid before the cuts)."""
    from .fields import field_mesh
    from .shapes import manifold_from_trimesh

    lo, hi = spec.exterior.bounds()
    t = time.time()
    info = {"pitch": round(float(pitch), 4)}
    try:
        if spec.solid:
            raise ValueError("no inside: a solid part")
        cav = field_mesh(body, lo, hi, pitch=pitch, project=3, progress=progress)
        cav_man = manifold_from_trimesh(cav)
        info["cavityVolume"] = round(float(cav.volume), 2)
    except ValueError as exc:
        if "no inside" not in str(exc):
            raise
        cav_man = None                      # the walls fill the exterior: a solid part
        info["cavityVolume"] = 0.0
    info["meshSeconds"] = round(time.time() - t, 1)
    ext = spec.exterior_manifold()
    adds = _union([i.brep.manifold() for i in spec.interfaces if i.op == "add" and not i.proud])
    solid = ext
    if cav_man is not None:
        void = cav_man - adds if not adds.is_empty() else cav_man
        solid = ext - void
    proud = _union([i.brep.manifold() for i in spec.interfaces if i.op == "add" and i.proud])
    if not proud.is_empty():
        solid = solid + proud
    cuts = _union([i.brep.manifold() for i in spec.interfaces if i.op == "cut"]
                  + [_grown(c.brep, c.clearance) for c in spec.components])
    part = solid - cuts if not cuts.is_empty() else solid
    # specks the cuts leave are not the part. decompose() splits by surface,
    # so an enclosed void comes out as its own piece of negative volume: a
    # void is kept (it is the part's, and the process check says if it traps
    # powder), only small solid pieces are dropped
    pieces = part.decompose()
    solids = sorted([q for q in pieces if q.volume() > 0], key=lambda q: -q.volume())
    voids = [q for q in pieces if q.volume() <= 0]
    if not solids:
        raise ValueError("nothing is left of the part after its cuts")
    keep = [q for q in solids if q.volume() >= 0.002 * solids[0].volume()]
    info["bodies"] = len(keep)
    info["enclosedVoids"] = len(voids)
    info["droppedSpecks"] = len(solids) - len(keep)
    info["droppedVolume"] = round(float(sum(q.volume() for q in solids[len(keep):])), 3)
    if len(keep) < len(solids):
        part = part - _union(solids[len(keep):])
    return part, info, solid


def _pitch(spec: BuildSpec, quick: bool, pitch=None):
    if pitch:
        return float(pitch)
    lo, hi = spec.exterior.bounds()
    thin = min([spec.skin] + [z.thickness for z in spec.zones])
    if spec.solid:
        thin = 2.0                       # nothing is sampled but the checks' grids; keep the pitch modest
    p = thin / (2.5 if quick else 4.0)
    # and no coarser than the smallest fillet can be drawn on
    blends = [i.blend for i in spec.interfaces if i.blend > 0]
    if blends and not quick:
        p = min(p, max(min(blends) / 2.0, 0.12))
    cells = float(np.prod((hi - lo) / p))
    budget = 6e6 if quick else 6e7
    if cells > budget:
        p *= (cells / budget) ** (1 / 3)
    return float(np.clip(p, 0.1, 2.0))


def solve_mass(spec: BuildSpec, body: BodyField, pitch: float, progress=None) -> dict:
    """The wall setting that reaches the target mass, on the sampled
    fields. Returns {"s", "predicted" (g of the part), "reachable": [lo, hi]}."""
    grid = _Grid(body, max(pitch, min(0.5, 2 * pitch)))
    rho = spec.material.density / 1000.0        # g / mm^3
    carried = sum(c.mass for c in spec.components) if spec.target["of"] == "assembly" else 0.0
    want = spec.target["grams"] - carried
    m0, m1 = grid.volume(0.0) * rho, grid.volume(1.0) * rho
    out = {"reachable": [round(m0 + carried, 3), round(m1 + carried, 3)], "components": round(carried, 3),
           "grid": grid}
    if want <= m0:
        out.update(s=0.0, predicted=m0)
        return out
    if want >= m1:
        out.update(s=1.0, predicted=m1)
        return out
    a, b = 0.0, 1.0
    for _ in range(40):
        s = 0.5 * (a + b)
        m = grid.volume(s) * rho
        if m < want:
            a = s
        else:
            b = s
        if b - a < 1e-5:
            break
    s = 0.5 * (a + b)
    out.update(s=s, predicted=grid.volume(s) * rho)
    return out


def _plain(o):
    """numpy scalars and arrays as plain JSON values."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def run_build(doc: dict, base_dir: str | None = None, out: str | None = None, quick: bool = False,
              pitch: float | None = None, render: bool = True, progress=None) -> dict:
    """Build the part a study document describes and check it. Returns the
    report (also written to out/report.json, with out/part.stl and
    out/checks.json)."""
    from .buildcheck import check_build
    from .shapes import trimesh_from_manifold

    t0 = time.time()
    spec = build_spec(doc, base_dir)
    p = _pitch(spec, quick, pitch)
    body = BodyField(spec, None)
    rho = spec.material.density / 1000.0
    carried = sum(c.mass for c in spec.components)
    report = {"name": spec.name, "material": spec.material.name, "process": spec.process, "pitch": round(p, 4)}

    mass_info = None
    if spec.target is not None:
        if progress:
            progress({"stage": "mass"})
        sol = solve_mass(spec, body, p)
        grid = sol.pop("grid")
        s = sol["s"]
        body = body.at(s)
        part, info, solid = assemble(spec, body, p, progress)
        mesh = trimesh_from_manifold(part)
        got = mesh.volume * rho
        want = spec.target["grams"] - (carried if spec.target["of"] == "assembly" else 0.0)
        steps = [{"s": round(s, 5), "grams": round(got, 3)}]
        # the sampled volume is a percent or so off the meshed one: step the
        # setting on the meshed masses (the sampled curve's slope first, a
        # secant through the two nearest meshes after)
        tried = [(s, got)]
        for _ in range(4):
            if abs(got - want) <= 0.5 * spec.target["tolerance"]:
                break
            if (s <= 0.0 and got > want) or (s >= 1.0 and got < want):
                break                       # at a limit and still off: the target is out of reach
            if len(tried) == 1:
                h = 0.02
                slope = (grid.volume(min(s + h, 1.0)) - grid.volume(max(s - h, 0.0))) * rho \
                    / (min(s + h, 1.0) - max(s - h, 0.0))
            else:
                (sa, ma), (sb, mb) = sorted(tried, key=lambda t: abs(t[1] - want))[:2]
                slope = (mb - ma) / (sb - sa) if abs(sb - sa) > 1e-9 else 0.0
            if slope <= 1e-9:
                break
            s2 = float(np.clip(s + (want - got) / slope, 0.0, 1.0))
            if abs(s2 - s) < 1e-6:
                break
            s = s2
            body = body.at(s)
            part, info, solid = assemble(spec, body, p, progress)
            mesh = trimesh_from_manifold(part)
            got = mesh.volume * rho
            tried.append((s, got))
            steps.append({"s": round(s, 5), "grams": round(got, 3)})
        walls = {}
        if spec.skin_hi is not None:
            walls["skin"] = round(spec.skin_lo + s * (spec.skin_hi - spec.skin_lo), 3)
        for z in spec.zones:
            if z.hi is not None:
                walls[z.id] = round(z.lo + s * (z.hi - z.lo), 3)
        mass_info = {"target": spec.target["grams"], "tolerance": spec.target["tolerance"], "of": spec.target["of"],
                     "components": round(carried, 3), "reachable": sol["reachable"], "setting": round(s, 5),
                     "walls": walls, "steps": steps}
    else:
        part, info, solid = assemble(spec, body, p, progress)
        mesh = trimesh_from_manifold(part)

    mesh.merge_vertices()
    # triangles the weld collapsed onto an edge (a few microns across) are not surface: a
    # print service's checker reads each as an open edge
    f = np.asarray(mesh.faces)
    flat = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    if flat.any():
        mesh.update_faces(~flat)
        mesh.remove_unreferenced_vertices()
        info["collapsedTriangles"] = int(flat.sum())
    report["build"] = info
    report["mass"] = {"part": round(float(mesh.volume * rho), 3), "components": round(carried, 3),
                      "assembly": round(float(mesh.volume * rho + carried), 3)}
    if mass_info is not None:
        report["targetMass"] = mass_info
    report["volume"] = round(float(mesh.volume), 2)
    if out is not None:
        os.makedirs(out, exist_ok=True)
        mesh.export(os.path.join(out, "part.stl"))
        report["part"] = "part.stl"
    if progress:
        progress({"stage": "checks"})
    checks = check_build(spec, mesh, out_dir=out, solid=solid, body=body, render=render, target=mass_info)
    report["checks"] = checks["checks"]
    report["ok"] = checks["ok"]
    report["failed"] = [c["id"] for c in checks["checks"] if c["ok"] is False]
    report["images"] = checks.get("images", [])
    report["seconds"] = round(time.time() - t0, 1)
    report = _plain(report)
    if out is not None:
        with open(os.path.join(out, "checks.json"), "w") as f:
            json.dump({"ok": report["ok"], "checks": report["checks"]}, f, indent=1)
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
    return report


def check_part(doc: dict, part, base_dir: str | None = None, out: str | None = None, render: bool = True) -> dict:
    """Check a mesh that already exists (a path, or a trimesh) against a
    spec: the same checks a build ends with, on a part made any other way
    or edited by hand since."""
    import trimesh

    from .buildcheck import check_build

    t0 = time.time()
    spec = build_spec(doc, base_dir)
    mesh = part if isinstance(part, trimesh.Trimesh) else trimesh.load(part, force="mesh")
    mesh.merge_vertices()
    rho = spec.material.density / 1000.0
    carried = sum(c.mass for c in spec.components)
    target = None
    if out is not None:
        os.makedirs(out, exist_ok=True)
    checks = check_build(spec, mesh, out_dir=out, solid=None, body=None, render=render, target=target)
    report = _plain({"ok": checks["ok"], "name": spec.name,
                     "mass": {"part": round(float(mesh.volume * rho), 3), "components": round(carried, 3),
                              "assembly": round(float(mesh.volume * rho + carried), 3)},
                     "checks": checks["checks"],
                     "failed": [c["id"] for c in checks["checks"] if c["ok"] is False],
                     "images": checks.get("images", []), "seconds": round(time.time() - t0, 1)})
    if out is not None:
        with open(os.path.join(out, "checks.json"), "w") as f:
            json.dump({"ok": report["ok"], "checks": report["checks"]}, f, indent=1)
    return report
