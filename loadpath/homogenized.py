"""
FE of a field graph's part with its lattice homogenized.

A lattice's walls are a fraction of a millimetre, so a voxel model of the
real geometry needs cells a few tenths of a millimetre across and millions
of them; tuning a lattice by meshing and analyzing every candidate that way
is out of reach. Instead each analysis cell (a millimetre or so) is the
material it holds on average, as lattice.py does for its own infill:

- the graph is evaluated twice per sample point, once with the lattice
  node filled solid and once with it empty, so each cell knows how much of
  it is solid skin or boss (`fs`) and how much is lattice (`fc`);
- the lattice's own fields (wall or density, cell) at the cell's centre give
  its relative density rho (lattice.relative_density);
- by default (lattice_tensors.py) the lattice is its unit cell's full 6x6
  stiffness at rho, rotated into the frame the cells lie in, and its wall
  stress comes from the cell's directional stress surface under the macro
  stress the lattice carries; the cell's elasticity is E (fs C_solid +
  fc C_lattice), each element its own;
- with anisotropic=False, the scalar fits: modulus E C rho^n, peak wall
  stress A rho^-m times the stress it carries on average (lattice.SHEETS),
  the cell's modulus E (fs + fc C rho^n);
- lattice cut open by a face with no skin over it keeps only part of its
  stiffness within a cell of that face (FREE_SURFACE_KNOCKDOWN).

Safety factor and deflection are read as verify_part reads them: peaks
outside a contact band round supports and loads, the stress allowance
included, deflection the mean displacement of each case's loaded nodes.
Sheet lattices (lattice.SHEETS) and strut cells (lattice.STRUTS); a
stretched `aspect` cell is read from its own tensors, solved on first use
and cached (lattice_tensors.aspect_table). Validation against
the resolved lattice: bench/lattice_aniso_validate.py, docs/fields.md.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["homogenized_check", "lattice_nodes"]

_SOLID, _EMPTY = -1.0e3, 1.0e3


def lattice_nodes(graph, node=None):
    """Names of the lattice nodes a FieldGraph's `node` (default its
    output) depends on, inline nodes included."""
    todo, seen, out = [node or graph.output], set(), []
    while todo:
        n = todo.pop()
        if n in seen or n not in graph.nodes:
            continue
        seen.add(n)
        if graph.nodes[n].op == "lattice":
            out.append(n)
        todo += list(graph.deps.get(n, ()))
    return out


def _variant(graph, doc, base_dir, lattice, value):
    """The same graph with `lattice` a constant: solid (negative) or empty.
    It shares the original's built resources (meshes, analyses), so a
    `result` node behind the lattice's wall field is not run again."""
    from .fields import FieldGraph

    g = FieldGraph(doc, base_dir=base_dir, sources=graph.sources)
    g._built, g.analyses = graph._built, graph.analyses
    nd = g.nodes[lattice]
    o = nd.spec.get("orient")
    if value == _EMPTY and isinstance(o, dict) and o.get("zones") and o.get("seam"):
        # the lattice taken out leaves its zones' seam plates: solid, like a skin
        nd.spec = {**nd.spec, "_seamOnly": True}
        return g
    nd.op, nd.spec, nd.refs = "constant", {"op": "constant", "value": value}, {}
    g.deps[lattice] = []
    return g


# the least share of a cell the part must fill for the cell to be analyzed
FILL_MIN = 0.1

# lattice stiffness lost at a face the lattice is cut open by (no skin):
# the cell right at the face keeps 1 - this of its unit-cell stiffness,
# rising linearly to all of it one lattice cell in (free_surface_exposure)
FREE_SURFACE_KNOCKDOWN = 0.5

# where a lattice roots into a skin or solid, each strut or sheet pulls on
# the skin over a small patch, and the resolved skin peaks there above the
# smooth stress the homogenized skin carries: a skinned tall-BCC beam read
# 0.87-0.90x of the resolved peak, 1.01-1.05x with this (bench/lattice_aniso_validate.py,
# --band 12). Analysis cells holding
# both skin and lattice read this many times their stress
SKIN_JUNCTION = 1.25

# the contact band is measured to analysis cell centres, a resolved check's
# to its nodes: a cell straddling the band's edge was left out whole, up to
# half a cell more than the resolved check leaves out, and right there (where
# a load spreads from the block it comes in through) the stress climbs. The
# band shrinks by this many cells so straddling cells count: a tall-BCC
# beam's pull read 0.88x of the resolved peak at the band's edge, 0.99x with
# it; the other skinned beams move up 1-5%
BAND_EDGE = 0.5


def homogenized_check(doc, study, node=None, base_dir=None, cells: int = 30_000, samples: int = 3,
                      fields: bool = False, graph=None, solid_ok: bool = False,
                      contact_band: float | None = None, anisotropic: bool | None = None,
                      free_surface: float | None = None, skin_junction: float | None = None,
                      sources: dict | None = None) -> dict:
    """mass, safety factor (worst case) and deflection of the graph's part
    under `study` with its lattice homogenized (module docstring). Returns
    a verify_part-like dict: mass, volume, safetyFactor, cases (name,
    safetyFactor, deflection, peakStress, peakAt, inLattice), bodies,
    analysisPitch, analysisCells, latticeFraction. `graph` is a built
    FieldGraph of `doc` to evaluate (and reuse the resources of).
    `contact_band` (mm): stress this close to a support or load is left out
    of the peak; default the study's optimization cell (domain.choose_pitch),
    as the study runner's own check does, and never under 1.5 analysis cells.
    A band that shrinks with the grid lets a bolt-hole singularity govern a
    fine confirm: a GRIP forearm read SF 4.7 on 60k cells and 1.3 on 2M,
    1 mm from a J3 bolt the part is clamped by.

    `anisotropic` (default: on wherever lattice_tensors has the cell's
    tables; LOADPATH_LATTICE_ANISO=0 turns it off): each lattice cell is its
    unit cell's full 6x6 stiffness, rotated into the frame the lattice lies
    in, and its wall stress is read from the directional stress surface
    (lattice_tensors) instead of the scalar fits' worst direction. False
    gives the isotropic reading. The report says which (`latticeModel`).
    `skin_junction` (default SKIN_JUNCTION) scales the stress in cells
    where the lattice meets a skin or solid. `sources`: in-memory fields
    for the graph's `source` nodes (a sized density grid, lattice_grade)."""
    import time

    from scipy.spatial import cKDTree

    from .domain import boundary_conditions
    from .fem import HexMesh, Solver, Stiffness, von_mises
    from .simp import _use_mg
    from .fields import FieldGraph, expand_groups
    from .lattice import lattice_fit, relative_density, sheet_modulus, sheet_stress_factor
    from .struts import is_strut
    from .verify import STRESS_ALLOWANCE

    t0 = time.time()
    if anisotropic is None:
        import os

        anisotropic = os.environ.get("LOADPATH_LATTICE_ANISO", "1").strip() not in ("0", "false", "off")
    from .lattice_tensors import aspect_density, has_tensors

    aspect = None
    full = expand_groups(doc)
    name = node or full.get("output") or list(full["nodes"])[-1]
    full = {**full, "output": name}
    g = graph or FieldGraph(full, base_dir=base_dir, sources=sources)
    lats = lattice_nodes(g, name)
    if not lats and not solid_ok:
        raise ValueError(f"{name!r} has no lattice in it: analyze it as a solid (verify_part), "
                         "or pass solid_ok to read it on the same fill-weighted grid")
    if not lats:
        # no lattice: the same grid, every cell solid by its fill. A thin
        # skin reads here without a mesh fine enough to resolve it, and
        # against a lattice part on equal terms
        lat, kind, lrefs, g_solid, g_empty = None, None, {}, g, g
    elif len(lats) > 1:
        raise ValueError(f"homogenized check handles one lattice node, {name!r} uses {', '.join(lats)}")
    else:
        lat = lats[0]
        lnode = g.nodes[lat]
        lspec, lrefs = lnode.spec, lnode.refs
        kind = lspec.get("type", "gyroid")
        if not is_strut(kind) and lspec.get("form", "sheet") != "sheet":
            raise ValueError(f"lattice {lat!r}: the homogenized fits are for sheet lattices, not network")
        lattice_fit(kind)                    # refuse a cell with no fit before any work
        if "aspect" in lspec and not np.allclose(lspec["aspect"], 1.0):
            aspect = tuple(float(x) for x in lspec["aspect"])
            if not (anisotropic and has_tensors(kind, aspect)):
                raise ValueError(f"lattice {lat!r}: a stretched 'aspect' cell is read from its own tensors "
                                 "(anisotropic on); the scalar fits are for cubic cells")
        g_solid = _variant(g, full, base_dir, lat, _SOLID)
        g_empty = _variant(g, full, base_dir, lat, _EMPTY)
    b = g_solid.bounds(name)
    if b is None:
        raise ValueError(f"{name!r} is unbounded: intersect its lattice with a solid")
    lo, hi = b
    mat = study.material

    # analysis grid: the envelope's box at about `cells` cells of the envelope
    span = hi - lo
    pitch = float((np.prod(span) * 0.6 / cells) ** (1.0 / 3.0))
    dims = np.maximum(np.ceil(span / pitch).astype(int), 1)
    origin = lo - 0.5 * (dims * pitch - span)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)
    sub = (np.arange(samples) + 0.5) / samples
    offs = np.stack(np.meshgrid(sub, sub, sub, indexing="ij"), -1).reshape(-1, 3)

    f_full = occupancy_at(g_solid, name, origin, ijk, offs, pitch)
    # every cell the part fills a tenth of or more, each as stiff and as
    # heavy as its share: dropping the half-filled cells at a face (as a
    # 0.5 threshold did) peeled up to half a cell off every skin, a quarter
    # of a 1 mm skin on a 0.5 mm grid, and which half depended on how the
    # grid fell
    active = f_full >= FILL_MIN
    f_skin = np.zeros(len(ijk))
    f_skin[active] = (f_full[active] if lat is None
                      else occupancy_at(g_empty, name, origin, ijk[active], offs, pitch))
    grid = np.zeros(tuple(dims), bool)
    grid[tuple(ijk[active].T)] = True
    lbl, _ = ndimage.label(grid)
    e_ijk = ijk[active]
    fs = np.clip(f_skin[active] / np.maximum(f_full[active], 1e-9), 0.0, 1.0)
    fc = 1.0 - fs
    fill = np.clip(f_full[active], 0.0, 1.0)
    centers = origin + (e_ijk + 0.5) * pitch
    # the lattice's density where it is
    if lat is None:
        rho = np.ones(len(centers))
    elif aspect is not None:
        # a stretched cell's density at a wall is its own (aspect_table)
        from .lattice import wall_for_density
        from .struts import strut_diameter_for_density

        cell = np.broadcast_to(_field(g, lrefs.get("cell", 8.0), centers), (len(centers),))
        if "density" in lrefs:
            dens = np.broadcast_to(_field(g, lrefs["density"], centers), (len(centers),))
            dens = np.clip(dens, 0.02 if is_strut(kind) else 0.05, 0.95)
            wall = (strut_diameter_for_density(kind, dens, cell) if is_strut(kind)
                    else wall_for_density(kind, dens, cell))
        else:
            wall = np.broadcast_to(_field(g, lrefs.get("thickness", 1.0), centers), (len(centers),))
        rho = aspect_density(kind, aspect, wall / cell)
    elif "density" in lrefs:
        rho = _field(g, lrefs["density"], centers)
    else:
        wall = np.broadcast_to(_field(g, lrefs.get("thickness", 1.0), centers), (len(centers),))
        cell = np.broadcast_to(_field(g, lrefs.get("cell", 8.0), centers), (len(centers),))
        blend = lrefs.get("blend", 0.0)
        blend = float(np.mean(_field(g, blend, centers))) if not isinstance(blend, float) else blend
        rho = relative_density(kind, wall / cell, 1.0, blend=blend / float(np.mean(cell)))
    rho = np.clip(np.broadcast_to(np.asarray(rho, float), (len(centers),)), 1e-3, 1.0)

    hm = HexMesh(e_ijk, pitch, origin)
    bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    sup = np.unique(lbl[tuple(e_ijk[bc.support_nodes[hm.enode].any(axis=1)].T)])
    keep = np.isin(lbl[tuple(e_ijk.T)], sup)
    bodies = int(len(np.unique(lbl[grid])))
    if not keep.all():
        e_ijk, fs, fc, rho, centers, fill = e_ijk[keep], fs[keep], fc[keep], rho[keep], centers[keep], fill[keep]
        hm = HexMesh(e_ijk, pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    # lattice cut open by the part's face with no skin over it: within a
    # cell of that face its struts and sheets end free, and carry less than
    # the periodic unit cell says
    if lat is None:
        exposed = np.zeros(len(e_ijk))
    else:
        cell_mm = np.broadcast_to(np.asarray(_field(g, lrefs.get("cell", 8.0), centers), float), (len(centers),))
        exposed = free_surface_exposure(e_ijk, dims, fs, fc, pitch, cell_mm)
    ec = np.ones(len(rho)) if lat is None else sheet_modulus(kind, rho)
    if free_surface is None:
        free_surface = FREE_SURFACE_KNOCKDOWN
    # stiffness only: the stress read below takes the cell's full tensor
    # times the (larger) strain the softer cell takes, so it errs high
    knock = 1.0 - float(free_surface) * exposed
    e_cell = mat.E * fill * (fs + fc * ec * knock)
    rhs = [bc.loads[ci] for ci in range(len(study.load_cases))]
    nu = study.elastic(mat)
    aniso = bool(anisotropic) and lat is not None and has_tensors(kind, aspect)
    if aniso:
        from .fem import elastic_c
        from .lattice_tensors import rotate_stiffness, stress_rotation, stress_surface, tensors_at

        c_lat, p_lat, q_lat, f_lat = tensors_at(kind, rho, aspect)
        rot = lattice_frames(g, lat, centers, pitch, start=name)
        # stiffness into the part's axes; stresses go into the cell's
        # (stress_rotation) for its wall-stress surface
        t_rot = None
        if rot is not None:
            c_lat, t_rot = rotate_stiffness(c_lat, rot), stress_rotation(rot)
        c_cell = (mat.E * fill)[:, None, None] * (fs[:, None, None] * elastic_c(1.0, nu)[None]
                                                    + (fc * knock)[:, None, None] * c_lat)
        us = solve_anisotropic(hm, nu, c_cell, bc, mat.E, rhs, proxy=e_cell)
    elif _use_mg(hm.n_elem):
        # a fine grid (a thin skin's re-check runs to a million cells and
        # more): matrix-free multigrid, as verify_part, a fraction of the
        # assembled matrix's memory (13 GB at 1M cells)
        from .mg import GridElasticity

        stiff = GridElasticity(hm, study.elastic(mat), bc.fixed,
                               springs=mat.E * bc.springs if bc.springs is not None else None,
                               rtol=1e-6, maxiter=3000)
        stiff.set_moduli(np.asarray(e_cell, float))
        us = stiff.solve_many(rhs)
    else:
        stiff = Stiffness(hm, study.elastic(mat), bc.fixed)
        k = stiff.assemble(e_cell)
        if bc.springs is not None:
            k = (k + mat.E * bc.springs).tocsr()
        us = Solver(hm, rtol=1e-6, maxiter=3000).solve_many(k, rhs)
    if contact_band is None:
        from .domain import choose_pitch
        contact_band = choose_pitch(study)
    band = max(1.5 * pitch, float(contact_band) - BAND_EDGE * pitch)
    wall_k = ec if lat is None else ec * sheet_stress_factor(kind, rho)
    vol_cell = pitch ** 3
    volume = float((fill * (fs + fc * rho) * vol_cell).sum())
    cases, worst = [], np.inf
    macro, bodies_mask = [], []
    junction_k = float(SKIN_JUNCTION if skin_junction is None else skin_junction)
    junction = (fs > 0.05) & (fc > 0.05)
    for ci, case in enumerate(study.load_cases):
        u = us[ci]
        if aniso:
            strain = _center_strain(hm, u)
            se = von_mises(strain @ (mat.E * elastic_c(1.0, nu)).T)
            sig_lat = mat.E * np.einsum("eab,eb->ea", c_lat, strain)
            if fields:
                macro.append(sig_lat)           # global axes
            if t_rot is not None:
                sig_lat = np.einsum("eab,eb->ea", t_rot, sig_lat)
            s_lat = np.empty(len(sig_lat))
            for a in range(0, len(sig_lat), 100_000):
                sl = slice(a, a + 100_000)
                s_lat[sl] = stress_surface(sig_lat[sl], p_lat[sl], q_lat[sl], f_lat[sl])
        else:
            se = von_mises(_center_stress(hm, nu, u, mat.E))
            s_lat = se * wall_k
        s_solid = np.where(fs > 0.05, se, 0.0)
        s_wall = np.where(fc > 0.05, s_lat, 0.0)
        s = np.maximum(s_solid, s_wall)
        if lat is not None:
            s = np.where(junction, s * junction_k, s)
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        near = cKDTree(hm.node_xyz[contact]).query(centers, k=1, distance_upper_bound=band)[0] < np.inf
        body = ~near if (~near).any() else np.ones(len(s), bool)
        if fields:
            bodies_mask.append(body)
        i = int(np.argmax(np.where(body, s, -1.0)))
        peak = float(s[i])
        sf = mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else np.inf
        disp = u.reshape(-1, 3)
        ln = bc.case_load_nodes[ci]
        defl = float(np.linalg.norm(disp[ln].mean(axis=0))) if ln.any() else 0.0
        worst = min(worst, sf)
        cases.append({"name": case.name, "safetyFactor": sf, "deflection": defl, "peakStress": peak,
                      "compliance": float(rhs[ci] @ u),
                      "peakAt": centers[i].tolist(), "inLattice": bool(s_wall[i] >= s_solid[i]),
                      "maxDisplacement": float(np.linalg.norm(disp, axis=1).max())})
    rep = {"mass": volume * mat.density / 1000.0, "volume": volume, "safetyFactor": worst, "cases": cases,
           "bodies": bodies, "analysisPitch": pitch, "analysisCells": int(hm.n_elem),
           "latticeFraction": float((fc * vol_cell).sum() / max(len(fc) * vol_cell, 1e-12)),
           "latticeDensity": [float(rho[fc > 0.05].min()), float(rho[fc > 0.05].max())] if (fc > 0.05).any()
           else None, "lattice": lat, "stressAllowance": STRESS_ALLOWANCE, "seconds": time.time() - t0,
           "latticeModel": None if lat is None else ("anisotropic" if aniso else "isotropic"),
           # share of the lattice's volume within half a cell of a face it is
           # cut open by (no skin): homogenization reads it too stiff there
           "latticeExposed": float((fc * fill * (exposed > 0.5)).sum() / max((fc * fill).sum(), 1e-12))
           if lat is not None else None}
    turned_exposed = None if lat is None else _turned_exposed(rot if aniso else None, fc * fill, exposed)
    rep["latticeTurnedExposed"] = turned_exposed
    zo = (lspec.get("orient") if lat is not None else None)
    zoned = isinstance(zo, dict) and len(zo.get("zones") or []) > 1
    if zoned and not zo.get("seam"):
        rep.setdefault("warnings", []).append(
            "zoned lattice without seam plates (orient.seam): where thin cells of two directions meet the "
            "check reads the seam too stiff (a skinless graded zoned beam: peak 2.4x higher, 1.4x softer at full "
            "resolution, the peak on the seam). Add orient.seam (1 mm) or confirm with a resolved check.")
    elif zoned:
        rep.setdefault("warnings", []).append(
            "zoned lattice with seam plates: the peak reads on the safe side but the stiffness about 20-25% "
            "high (a skinless graded zoned beam at full resolution); leave margin on deflection.")
    elif turned_exposed is not None and turned_exposed > TURNED_EXPOSED_WARN:
        rep.setdefault("warnings", []).append(
            f"{100 * turned_exposed:.0f}% of the lattice is turned cells cut open at a free face (no skin): "
            "the free-surface knockdown reads square-cut cells 1.5-2.8x conservative and bias-cut (turned) ones "
            "about right to 2.5x too stiff, so a turned design compared with the cells as laid looks better than "
            "it is (skinless bars read 7-23% lighter turned were 1.3-3.2x weaker at full resolution, "
            "bench/results/lattice-orient-confirm.json). Add a skin or confirm with resolved.resolved_check "
            "(`loadpath resolved-check`).")
    if fields:
        rep["fields"] = {"centers": centers, "rho": rho, "fs": fs, "fc": fc, "fill": fill,
                         "knock": knock if lat is not None else None,
                         # anisotropic: per case the lattice's macro stress (global
                         # axes) and the cells the peak is read over; the cell frames
                         "latticeStress": macro or None, "body": bodies_mask,
                         "frames": rot if aniso else None, "aspect": aspect, "kind": kind}
    return rep


def free_surface_exposure(e_ijk, dims, fs, fc, pitch, cell_mm):
    """Per analysis cell, 1 - (distance to the part's outside) / (lattice
    cell), clipped to [0, 1], where the outside is nearer than any solid
    (skin) cell; 0 elsewhere. A lattice under a skin is not exposed."""
    from scipy import ndimage

    pad = 1
    shape = tuple(int(d) + 2 * pad for d in dims)
    inside = np.zeros(shape, bool)
    idx = tuple((e_ijk + pad).T)
    inside[idx] = True
    skin = np.zeros(shape, bool)
    skin[idx] = fs > 0.5
    d_out = ndimage.distance_transform_edt(inside)[idx] * pitch - 0.5 * pitch
    d_skin = ndimage.distance_transform_edt(~skin)[idx] * pitch if skin.any() else np.full(len(e_ijk), np.inf)
    e = np.clip(1.0 - d_out / np.maximum(cell_mm, 1e-9), 0.0, 1.0)
    return np.where((fc > 0.05) & (d_out < d_skin), e, 0.0)


def _center_strain(hm, u):
    """Element-center strain (ne, 6), engineering shears (the bubbles'
    strain is zero there)."""
    from .fem import hex8_b_center

    return u[hm.edof] @ (hex8_b_center() / hm.pitch).T


# share of the lattice (by volume) that is turned more than 5 degrees off
# its laid axes AND within half a cell of a free face it is cut open by.
# Above this the result carries a warning: the free-surface knockdown was
# fitted on cells cut square, and turned cells cut on the bias at a bare
# face read far too stiff (bench: skinless graded deep beam, resolved check)
TURNED_EXPOSED_WARN = 0.1


def _turned_exposed(rot, weight, exposed):
    if rot is None:
        return 0.0
    R = np.asarray(rot, float)
    if R.ndim == 2:
        R = np.broadcast_to(R, (len(weight), 3, 3))
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    w = np.asarray(weight, float)
    hit = (ang > 5.0) & (np.asarray(exposed) > 0.5)
    return float((w * hit).sum() / max(w.sum(), 1e-12))


def lattice_frames(g, lat, centers, pitch, start=None):
    """(n, 3, 3) the lattice's local cell axes at `centers` (columns, in
    global components): the polar factor of the lattice coordinates'
    Jacobian, by central differences. None where the cell axes are the
    global ones (a cartesian, ungraded lattice, or a Voronoi foam, whose
    tables are its own periodic draw in global axes)."""
    nd = g.nodes[lat]
    s = nd.spec
    from .struts import canonical

    moves = _path_transforms(g, start or g.output, lat)
    rotated = any("rotate" in g.nodes[t].spec for t in moves)
    if isinstance(s.get("orient"), dict) and "zones" in s["orient"]:
        # turned zone by zone: the polar factor of the zones' rotations
        # weighted where the cell sits (a transform above moves the points)
        from .fields import orient_weights, zone_rotations

        P = centers
        for t in moves:
            P = g.transform_points(g.nodes[t], P)
        if rotated:
            raise ValueError(f"lattice {lat!r}: a zoned orient under a rotating transform node is not supported; "
                             "fold the rotation into the zones")
        m = np.einsum("nk,kij->nij", orient_weights(nd, P), zone_rotations(nd))
        w, _, vt = np.linalg.svd(m)
        return w @ vt
    frame = (s.get("frame") or {}).get("type", "cartesian")
    graded = isinstance(nd.refs.get("cell"), str) and s.get("grading", "warp") == "warp"
    if not rotated and (canonical(s.get("type", "gyroid")) == "voronoi"
                        or (frame == "cartesian" and not graded and "orient" not in s)):
        return None

    def coords(P):
        for t in moves:
            P = g.transform_points(g.nodes[t], P)
        if canonical(s.get("type", "gyroid")) == "voronoi":
            return P
        return g.lattice_coords(nd, P)[0]

    h = 1e-3 * pitch
    jac = np.empty((len(centers), 3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        up = coords(centers + d)
        um = coords(centers - d)
        du = up - um
        du -= np.round(du)                    # a frame's seam jumps by whole cells
        jac[:, :, i] = du / (2 * h)
    w, _, vt = np.linalg.svd(jac)
    q = w @ vt                                # rows: the cell's axes
    return np.swapaxes(q, 1, 2)


def _path_transforms(g, start, target):
    """Transform nodes on a path from `start` down to `target`, outermost
    first (each moves the points the lattice is sampled at)."""
    def walk(n, seen):
        if n == target:
            return []
        if n in seen or n not in g.nodes:
            return None
        seen.add(n)
        for d in g.deps.get(n, ()):
            rest = walk(d, seen)
            if rest is not None:
                return ([n] if g.nodes[n].op == "transform" else []) + rest
        return None

    return walk(start, set()) or []


def solve_anisotropic(hm, nu, c_cell, bc, modulus, rhs, proxy, rtol=1e-6, maxiter=3000):
    """Displacements for per-element elasticity matrices `c_cell` (ne, 6, 6,
    moduli included). Below simp.MG_MIN_CELLS: the assembled matrix, each
    element its own condensed hex8 (fem.hex8_ke_batch), CG + AMG. From
    there: the operator matrix-free (each element's C and bubble
    condensation kept, in single precision) inside CG preconditioned by the
    geometric multigrid of the isotropic stand-in with moduli `proxy`."""
    from .fem import Solver, Stiffness, hex8_ke_batch
    from .simp import _use_mg

    springs = modulus * bc.springs if bc.springs is not None else None
    ne = len(c_cell)
    chunk = 20_000
    if not _use_mg(ne):
        stiff = Stiffness(hm, nu, bc.fixed)
        data = np.zeros(stiff.nnz)
        for a in range(0, ne, chunk):
            idx = np.arange(a, min(a + chunk, ne))
            ke, _ = hex8_ke_batch(c_cell[idx])
            stiff.add_elements(data, ke * hm.pitch, idx)
        k = stiff.finish(data)
        if springs is not None:
            k = (k + springs).tocsr()
        return Solver(hm, rtol=rtol, maxiter=maxiter).solve_many(k, rhs)
    return _AnisoMG(hm, nu, c_cell, bc, springs, proxy, rtol, maxiter).solve_many(rhs)


class _AnisoMG:
    """Matrix-free anisotropic operator + isotropic multigrid preconditioner."""

    CHUNK = 40_000

    def __init__(self, hm, nu, c_cell, bc, springs, proxy, rtol, maxiter):
        from .fem import _gauss_b, hex8_ke_batch
        from .mg import GridElasticity

        self.hm, self.rtol, self.maxiter = hm, rtol, maxiter
        self.free = ~np.asarray(bc.fixed, bool)
        self.springs = springs.tocsr() if springs is not None else None
        b, ba = _gauss_b()
        self.b = (b.reshape(48, 24) / hm.pitch).astype(np.float32)
        self.ba = np.zeros((48, 9), np.float32)
        for gi in range(8):
            self.ba[6 * gi:6 * gi + 6] = ba[gi] / hm.pitch
        self.w = hm.pitch ** 3 / 8.0
        ne = len(c_cell)
        self.c = c_cell.astype(np.float32)
        self.r = np.empty((ne, 9, 24), np.float32)
        for a in range(0, ne, 20_000):
            _, r = hex8_ke_batch(c_cell[a:a + 20_000])
            self.r[a:a + 20_000] = 0.0 if r is None else r
        self.mg = GridElasticity(hm, nu, bc.fixed, springs=springs, rtol=rtol, maxiter=maxiter)
        self.mg.set_moduli(np.asarray(proxy, float))
        self.last_iters = 0

    def apply(self, u):
        hm = self.hm
        out = np.zeros(hm.n_dof)
        u = np.where(self.free, u, 0.0)
        for a in range(0, hm.n_elem, self.CHUNK):
            sl = slice(a, a + self.CHUNK)
            ed = hm.edof[sl]
            ue = u[ed]
            alpha = np.einsum("eij,ej->ei", self.r[sl], ue)
            eps = (ue @ self.b.T + alpha @ self.ba.T).reshape(-1, 8, 6)
            sig = np.einsum("eij,egj->egi", self.c[sl], eps).reshape(-1, 48)
            fe = (sig @ self.b) * self.w
            out += np.bincount(ed.ravel(), weights=fe.astype(np.float64).ravel(), minlength=hm.n_dof)
        if self.springs is not None:
            out += self.springs @ u
        return np.where(self.free, out, 0.0)

    def precondition(self, r):
        g = self.mg.to_grid(r) * self.mg.levels[0].mask
        return np.where(self.free, self.mg.from_grid(self.mg.precondition(g) * self.mg.levels[0].mask), 0.0)

    def solve_many(self, fs):
        out = []
        for f in fs:
            b = np.where(self.free, f, 0.0)
            x = np.zeros_like(b)
            bn = float(np.linalg.norm(b))
            if bn == 0:
                out.append(x)
                continue
            r = b.copy()
            z = self.precondition(r)
            p = z.copy()
            rz = float(r @ z)
            ok = False
            for it in range(1, self.maxiter + 1):
                ap = self.apply(p)
                alpha = rz / float(p @ ap)
                x += alpha * p
                r -= alpha * ap
                if np.linalg.norm(r) <= self.rtol * bn:
                    ok = True
                    break
                z = self.precondition(r)
                rz_new = float(r @ z)
                p = z + (rz_new / rz) * p
                rz = rz_new
            self.last_iters = it
            if not ok:
                raise RuntimeError(f"anisotropic lattice solve did not converge ({it} iterations)")
            out.append(x)
        return out


def _center_stress(hm, nu, u, modulus):
    """Element-center stress (Stiffness.stress without the matrix layout)."""
    from .fem import elastic_c, hex8_b_center

    b = hex8_b_center() / hm.pitch
    strain = u[hm.edof] @ b.T
    return (strain @ elastic_c(1.0, nu).T) * modulus


def occupancy_at(graph, name, origin, ijk, offs, pitch):
    """Solid share of each cell from its sub-samples, each counted by how
    deep it sits (the signed distance over the sample spacing, a linear ramp
    across it) rather than in or out: a flat face that lands on a row of
    samples would otherwise drop them all, and a thin wall's reading jumped
    by a third from one grid to the next (a 1 mm skin on a 0.5 mm grid lost
    13% of its mass)."""
    occ = np.empty(len(ijk))
    n = round(len(offs) ** (1.0 / 3.0))
    h = pitch / max(n, 1)
    step = max(1, 400_000 // len(offs))
    for i in range(0, len(ijk), step):
        P = (origin + (ijk[i:i + step, None, :] + offs[None]) * pitch).reshape(-1, 3)
        d = np.asarray(graph.evaluate(P, name), float)
        occ[i:i + step] = np.clip(0.5 - d / h, 0.0, 1.0).reshape(-1, len(offs)).mean(axis=1)
    return occ


def _field(g, ref, P):
    """A lattice param at points: a number or the node it names."""
    return float(ref) if isinstance(ref, (int, float)) else np.asarray(g.evaluate(P, ref), float)
