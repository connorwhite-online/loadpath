"""
Stress-graded FDM infill: the slicer grades the part, loadpath says where.

A slicer already knows how to print a part sparse inside and solid at the
skin. What it does not know is where the part can afford to be sparse, so
people print everything at 15-20% and hope, or at 100% and wait. This sizes
a ladder of (infill density, perimeter count) levels region by region from
the part's stress, then hands the regions to the slicer as modifier volumes
in a 3MF (threemf.write_3mf): no lattice mesh, no million triangles, and the
slicer's own toolpaths.

The sizing is the homogenized fully-stressed design lattice.py uses for TPMS
cores, with the slicer's construction in place of a sheet lattice:

- each analysis cell is part skin, part infill. The skin is the perimeters
  (count x line width) and is solid; the rest is gyroid infill at the
  level's density rho, a sheet of modulus E C rho^n whose walls carry
  A rho^-m times the macro stress (lattice.SHEETS: PrusaSlicer's gyroid is
  the same surface drawn one extrusion wide), at a knockdown for walls that
  are a single line with a seam every layer;
- a voxel FEA of the graded part gives each cell's strain; each cell takes
  the lowest level whose skin and infill both stay under strength / (SF x
  verify's 1.15 allowance), holding the cell's macro stress (fully stressed
  design), and the analysis repeats as the load moves onto what got stiffer;
- material round preserved shapes, keep-outs, loads and supports (a
  `collar`) stays solid: a bolt boss is not where to save filament.

The verified safety factor of the solid part is scaled by how much the
graded part reads below the solid one on the same grid and loads, so the
grading carries the delivered part's refined verification (submodels,
interlayer check) instead of the coarser analysis grid's absolute numbers.
Deflection is scaled the same way and the whole ladder floor steps up while
it breaks `goal.maxDeflection`.

The infill constants are estimates: sheet-gyroid unit-cell FEA, and a 0.8
knockdown for single-line walls. Printed coupons (the calibration loop) are
what will tighten them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .lattice import SHEETS, _block_mean, _collar_distance, _signed_distance
from .spec import Study

__all__ = ["GradeOptions", "grade_infill", "DEFAULT_LEVELS"]

# (infill %, perimeters) from sparsest to densest; solid (100%) is always
# the level above the last. One perimeter count throughout by default: a
# modifier that changes the perimeters makes PrusaSlicer split the layer
# there and run perimeters round the cut, inside the part (the shelf
# bracket's external perimeters went 6 -> 16 cm^3 and the print took 45 min
# longer than with the same infill at 3 walls throughout). Changing only the
# density merges the regions' perimeters.
DEFAULT_LEVELS = ((10, 3), (15, 3), (25, 3), (40, 3), (60, 3))


@dataclass
class GradeOptions:
    levels: tuple = DEFAULT_LEVELS
    line_width: float = 0.45          # mm per perimeter (the profile's perimeter width)
    safety_factor: float | None = None   # default: the study goal's, else 2
    collar: float = 2.0               # mm of solid round preserves, keep-outs, loads, supports
    knockdown: float = 0.8            # infill wall strength / solid strength
    analysis_cells: int = 25_000
    iterations: int = 12
    modifier_faces: int = 20_000      # per modifier mesh, after decimation

    @classmethod
    def from_spec(cls, spec, safety_factor: float | None = None) -> "GradeOptions":
        spec = dict(spec or {})
        o = cls()
        if "levels" in spec:
            lv = []
            for item in spec["levels"]:
                if isinstance(item, dict):
                    d, p = item.get("infill"), item.get("perimeters", 3)
                else:
                    d, p = item
                d, p = float(d), int(p)
                if not 0.0 < d < 100.0 or p < 1:
                    raise ValueError("slicer.grade.levels are [infill % (0..100, exclusive), perimeters >= 1]")
                lv.append((d, p))
            if not lv:
                raise ValueError("slicer.grade.levels is empty")
            lv.sort()
            o.levels = tuple(lv)
        for key, attr in (("lineWidth", "line_width"), ("collar", "collar"), ("knockdown", "knockdown"),
                          ("analysisCells", "analysis_cells")):
            if key in spec:
                setattr(o, attr, type(getattr(o, attr))(spec[key]))
        if not 0.0 < o.knockdown <= 1.0:
            raise ValueError("slicer.grade.knockdown is a fraction of the solid strength, in (0, 1]")
        o.safety_factor = float(spec.get("safetyFactor") or safety_factor or 2.0)
        return o


def _level_props(o: GradeOptions, kind: str = "gyroid"):
    """Per level (the solid one last): relative density, skin thickness mm,
    sheet modulus factor and infill wall-stress factor."""
    s = SHEETS[kind]
    rho = np.array([d / 100.0 for d, _ in o.levels] + [1.0])
    skin = np.array([p * o.line_width for _, p in o.levels] + [o.levels[-1][1] * o.line_width])
    emod = np.where(rho >= 1.0, 1.0, s["C"] * rho ** s["n"])
    wall = np.where(rho >= 1.0, 1.0, s["A"] * rho ** -s["m"])
    return rho, skin, emod, wall


def grade_infill(study: Study, mesh, verified: dict | None = None, opts: GradeOptions | None = None,
                 on_progress=None) -> dict:
    """Size per-region infill for `mesh` (the delivered part). `verified` is
    its verification report (safety factor and per-case deflection to
    scale). Returns {"objectSettings", "modifiers" (meshes and settings for
    threemf.write_3mf), "levels", "safetyFactor", "predicted", ...}."""
    from scipy.spatial import cKDTree

    from .domain import boundary_conditions, gravity_force
    from .fem import HexMesh, Solver, Stiffness, von_mises
    from .verify import STRESS_ALLOWANCE, voxelize

    o = opts or GradeOptions(safety_factor=study.goal.safety_factor or 2.0)
    mat = study.material
    rho_l, skin_l, emod_l, wall_l = _level_props(o)
    n_lv = len(rho_l)
    solid_lv = n_lv - 1

    vol = float(mesh.volume)
    pitch = (vol / o.analysis_cells) ** (1.0 / 3.0)
    r = 3
    step = pitch / r
    lo = np.asarray(mesh.bounds[0], float) - pitch
    hi = np.asarray(mesh.bounds[1], float) + pitch
    dims = tuple(int(np.ceil((hi[i] - lo[i]) / pitch)) * r for i in range(3))
    solid, origin = voxelize(mesh, step, box=(lo, lo + np.asarray(dims) * step))
    if solid.shape != dims:
        out = np.zeros(dims, bool)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(solid.shape, dims))
        out[sl] = solid[sl]
        solid = out
    depth = -_signed_distance(solid, step)          # mm inside the surface, > 0 inside
    f_in = _block_mean(solid.astype(np.float32), r)
    # share of each analysis cell's material that is skin at each level
    fs_lv = []
    for t in skin_l:
        fs_lv.append(_block_mean((solid & (depth < t)).astype(np.float32), r) / np.maximum(f_in, 1e-6))
    active = f_in >= 0.5
    lbl, _ = ndimage.label(active)
    ijk = np.argwhere(active)
    hm = HexMesh(ijk, pitch, lo)
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    sup = np.unique(lbl[tuple(ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(ijk.T)], sup)
    if not keep.all():
        ijk = ijk[keep]
        hm = HexMesh(ijk, pitch, lo)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    fs = np.stack([np.clip(f[tuple(ijk.T)], 0.0, 1.0) for f in fs_lv], axis=1)    # (ne, levels)
    fs[:, solid_lv] = 1.0
    centers = hm.elem_centers()
    collar = np.zeros(hm.n_elem, bool)
    if o.collar > 0:
        collar = _collar_distance(study, centers, o.collar) < 0
    # the stiffness of each cell at each level, as a share of solid
    e_lv = fs + (1.0 - fs) * emod_l[None, :]

    stiff = Stiffness(hm, mat.nu, bc.fixed)
    solver = Solver(hm, rtol=1e-6, maxiter=3000)
    contact = bc.support_nodes.copy()
    for ln in bc.case_load_nodes:
        contact |= ln
    band = 1.5 * pitch
    near = cKDTree(hm.node_xyz[contact]).query(centers, k=1, distance_upper_bound=band)[0] < np.inf
    fg = gravity_force(hm, np.ones(hm.n_elem), bc.gravity, bc.fixed) if bc.gravity is not None else 0.0
    loads = [bc.loads[ci] + fg for ci in range(len(study.load_cases))]
    allow0 = mat.strength / (o.safety_factor * STRESS_ALLOWANCE)
    rows = np.arange(hm.n_elem)

    last = [None]

    def analyze(lev):
        e = e_lv[rows, lev]
        k = stiff.assemble(mat.E * e)
        if bc.springs is not None:
            k = (k + mat.E * bc.springs).tocsr()
        solver.invalidate()          # a new stiffness contrast wants a new hierarchy
        us = solver.solve_many(k, loads, last[0])
        last[0] = us
        se = np.max([von_mises(stiff.stress(u, mat.E)) for u in us], axis=0)
        defl = []
        for ci, u in enumerate(us):
            ln = bc.case_load_nodes[ci]
            d = u.reshape(-1, 3)
            defl.append(float(np.linalg.norm(d[ln].mean(axis=0))) if ln.any() else
                        float(np.linalg.norm(d, axis=1).max()))
        # the skin carries the strain at full modulus; the infill's walls
        # carry A rho^-m times its macro stress (E* / E of the strain stress)
        skin_s = np.where(fs[rows, lev] > 0, se, 0.0)
        inf_s = np.where(fs[rows, lev] < 1, wall_l[lev] * emod_l[lev] * se / o.knockdown, 0.0)
        peak = float(np.max(np.where(near, 0.0, np.maximum(skin_s, inf_s))))
        return se, defl, peak

    def report_sf(peak):
        return mat.strength / (STRESS_ALLOWANCE * peak) if peak > 0 else float("inf")

    # the solid part on the same grid: what the graded one is compared with
    se_solid, defl_solid, peak_solid = analyze(np.full(hm.n_elem, solid_lv))
    sf_solid = report_sf(peak_solid)
    ver_sf = (verified or {}).get("safetyFactor")
    ver_defl = [c.get("deflection") for c in (verified or {}).get("cases", [])]
    target_defl = study.goal.max_deflection

    def on_verified_scale(sf_pred):
        # the graded part's safety factor on the verified part's scale
        return ver_sf * sf_pred / sf_solid if ver_sf is not None else sf_pred

    # Start from the solid part's strain and let each cell find its level;
    # the allowable then moves (geometrically damped) until the graded part
    # reads the target, as the load shifts onto what stayed stiff. Each
    # cell's macro stress is held while it picks a level (fully stressed
    # design, as lattice.py sizes TPMS): at a fixed strain a denser gyroid's
    # walls carry more, never less, so a strain-held pick only ever chose
    # the sparsest level or solid.
    lev = np.full(hm.n_elem, solid_lv)
    se = se_solid
    floor = 0
    history = []
    allow = allow0 * (sf_solid / ver_sf) if ver_sf is not None else allow0
    grid = np.zeros(active.shape)
    best = None             # (fill, levels, se, defl, peak) of the sparsest passing pass
    short = 1.0
    prev = lev
    for it in range(o.iterations):
        sig = (se * e_lv[rows, lev])[:, None]          # macro stress, MPa
        skin_s = sig / e_lv                              # strain-equivalent at each level
        inf_s = wall_l[None, :] * emod_l[None, :] * skin_s / o.knockdown
        ok = (np.where(fs < 1, inf_s, 0.0) <= allow) & (np.where(fs > 0, skin_s, 0.0) <= allow)
        ok[:, solid_lv] = True
        need = np.argmax(ok, axis=1)
        # a concentration spreads over a cell before the slicer's regions can
        # follow it
        grid[:] = 0
        grid[tuple(ijk.T)] = need
        need = ndimage.maximum_filter(grid, size=3)[tuple(ijk.T)].astype(int)
        # hysteresis, or levels flicker: after a pass under the target only
        # raise levels; after one that held it, drop at most one
        if it == 0:
            new = need
        elif short < 1.0:
            new = np.maximum(need, lev)
        else:
            new = np.maximum(need, lev - 1)
        new = np.maximum(new, floor)
        new[collar] = solid_lv
        se, defl, peak = analyze(new)
        lev = new
        sf_pred = report_sf(peak)
        short = on_verified_scale(sf_pred) / o.safety_factor
        ratio_d = [d / max(ds, 1e-12) for d, ds in zip(defl, defl_solid)]
        stiff_ok = not (target_defl and ver_defl) or \
            max(v * rr for v, rr in zip(ver_defl, ratio_d) if v is not None) <= target_defl
        fill = float(np.mean(fs[rows, lev] + (1 - fs[rows, lev]) * rho_l[lev]))
        history.append({"iteration": it, "predictedSF": on_verified_scale(sf_pred), "floor": floor,
                        "fill": fill, "solidCells": float(np.mean(lev == solid_lv))})
        if on_progress is not None:
            on_progress({"stage": "grade", **history[-1]}, None)
        if short >= 1.0 and stiff_ok and (best is None or fill < best[0]):
            best = (fill, lev.copy(), se, defl, peak)
        if not stiff_ok:
            if floor >= solid_lv:
                break
            floor += 1
            continue
        if 1.0 <= short < 1.06:
            break
        allow *= float(np.clip(short, 0.8, 0.99)) if short < 1.0 else float(np.clip(short, 1.0, 2.0)) ** 0.75
        prev = lev
    if best is None:
        # nothing graded held the target: solid, as verified
        lev = np.full(hm.n_elem, solid_lv)
        se, defl, peak = se_solid, defl_solid, peak_solid
    else:
        _, lev, se, defl, peak = best
    # reported on the verified part's scale
    knock = sf_solid / report_sf(peak) if peak > 0 else 1.0
    sf = ver_sf / knock if ver_sf is not None else report_sf(peak)
    ratio_d = [d / max(ds, 1e-12) for d, ds in zip(defl, defl_solid)]

    cell_vol = pitch ** 3 * f_in[tuple(ijk.T)]
    fill = fs[rows, lev] + (1.0 - fs[rows, lev]) * rho_l[lev]
    est_volume = float((cell_vol * fill).sum())
    lv_grid = np.full(active.shape, -1, int)
    lv_grid[tuple(ijk.T)] = lev
    base = int(lev[~collar].min()) if (~collar).any() else solid_lv
    modifiers = _modifier_meshes(lv_grid, lo, pitch, base, n_lv, o)
    usage = {(_label(o, i)): float(cell_vol[lev == i].sum() / cell_vol.sum()) for i in range(n_lv)
             if (lev == i).any()}
    obj = _settings(o, base)
    skin_t = skin_l[base]
    obj.update(top_solid_min_thickness=f"{skin_t:.2f}", bottom_solid_min_thickness=f"{skin_t:.2f}")
    return {
        "levels": [{"infill": float(rho_l[i] * 100), "perimeters": (o.levels[i][1] if i < solid_lv else None),
                    "label": _label(o, i)} for i in range(n_lv)],
        "base": _label(o, base),
        "volumeShare": usage,
        "objectSettings": obj,
        "modifiers": modifiers,
        "safetyFactor": sf,
        "targetSafetyFactor": o.safety_factor,
        "solidSafetyFactor": ver_sf,
        "knockdown": knock,
        "deflectionRatio": ratio_d,
        "deflection": [v * rr for v, rr in zip(ver_defl, ratio_d)] if ver_defl else None,
        "estimatedVolume": est_volume,
        "estimatedMass": est_volume * mat.density / 1000.0,
        "analysisPitch": pitch,
        "analysisCells": int(hm.n_elem),
        "predicted": {"solid": sf_solid, "graded": report_sf(peak)},
        "history": history,
        "assumptions": {"pattern": "gyroid", "sheet": SHEETS["gyroid"], "knockdown": o.knockdown,
                        "lineWidth": o.line_width, "collar": o.collar},
    }


def _label(o: GradeOptions, i: int) -> str:
    if i >= len(o.levels):
        return "solid"
    d, p = o.levels[i]
    return f"{d:g}% x{p}"


def _settings(o: GradeOptions, i: int, walls: bool = True) -> dict:
    """PrusaSlicer keys for level i; `walls` False leaves the perimeters to
    the object (a modifier that keeps the object's count)."""
    if i >= len(o.levels):
        out = {"fill_density": "100%", "fill_pattern": "rectilinear", "perimeters": str(o.levels[-1][1])}
    else:
        d, p = o.levels[i]
        out = {"fill_density": f"{d:g}%", "fill_pattern": "gyroid", "perimeters": str(p)}
    if not walls:
        out.pop("perimeters")
    return out


def _modifier_meshes(lv_grid, lo, pitch, base, n_lv, o: GradeOptions) -> list:
    """One closed mesh per level above the base, around every cell at that
    level or above (nested: later, denser modifiers win in the slicer), a
    cell larger all round so a region is never short of what it needs."""
    import trimesh
    from skimage.measure import marching_cubes

    out = []
    for i in range(base + 1, n_lv):
        m = lv_grid >= i
        if not (lv_grid == i).any():
            continue            # the same region as the next level up
        m = ndimage.binary_dilation(m, iterations=1)
        f = ndimage.gaussian_filter(np.pad(m, 2).astype(np.float32), 0.7)
        verts, faces, _, _ = marching_cubes(f, 0.5, spacing=(pitch,) * 3)
        verts = verts + lo - 2.0 * pitch + 0.5 * pitch
        tm = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
        if len(tm.faces) > o.modifier_faces:
            try:
                import fast_simplification

                v2, f2 = fast_simplification.simplify(tm.vertices, tm.faces, 1.0 - o.modifier_faces / len(tm.faces))
                tm = trimesh.Trimesh(v2, f2, process=True)
            except ImportError:
                pass
        if tm.volume < 0:
            tm.invert()
        walls = o.levels[min(i, len(o.levels) - 1)][1] != o.levels[min(base, len(o.levels) - 1)][1]
        out.append({"name": f"loadpath {_label(o, i)}", "mesh": tm, "settings": _settings(o, i, walls)})
    return out
