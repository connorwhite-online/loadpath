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
  its relative density rho (lattice.relative_density), and the unit-cell
  fits give its modulus E C rho^n and its peak wall stress A rho^-m times
  the stress it carries on average (lattice.SHEETS);
- the cell's modulus is E (fs + fc C rho^n); the solid's stress is the
  strain times E, the lattice's wall stress is that times C rho^n A rho^-m.

Safety factor and deflection are read as verify_part reads them: peaks
outside a contact band round supports and loads, the stress allowance
included, deflection the mean displacement of each case's loaded nodes.
Sheet lattices (lattice.SHEETS) and strut cells (lattice.STRUTS, the
conservative isotropic reading of an anisotropic cell); a stretched
`aspect` cell has no fit and is refused. The cell's frame (graded,
cylindrical, conformal) only moves cells around: the fits are per cell.
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

    g = FieldGraph(doc, base_dir=base_dir)
    g._built, g.analyses = graph._built, graph.analyses
    nd = g.nodes[lattice]
    nd.op, nd.spec, nd.refs = "constant", {"op": "constant", "value": value}, {}
    g.deps[lattice] = []
    return g


def homogenized_check(doc, study, node=None, base_dir=None, cells: int = 30_000, samples: int = 3,
                      fields: bool = False, graph=None) -> dict:
    """mass, safety factor (worst case) and deflection of the graph's part
    under `study` with its lattice homogenized (module docstring). Returns
    a verify_part-like dict: mass, volume, safetyFactor, cases (name,
    safetyFactor, deflection, peakStress, peakAt, inLattice), bodies,
    analysisPitch, analysisCells, latticeFraction. `graph` is a built
    FieldGraph of `doc` to evaluate (and reuse the resources of)."""
    import time

    from scipy.spatial import cKDTree

    from .domain import boundary_conditions
    from .fem import HexMesh, Solver, Stiffness, von_mises
    from .fields import FieldGraph, expand_groups
    from .lattice import lattice_fit, relative_density, sheet_modulus, sheet_stress_factor
    from .struts import is_strut
    from .verify import STRESS_ALLOWANCE

    t0 = time.time()
    full = expand_groups(doc)
    name = node or full.get("output") or list(full["nodes"])[-1]
    full = {**full, "output": name}
    g = graph or FieldGraph(full, base_dir=base_dir)
    lats = lattice_nodes(g, name)
    if not lats:
        raise ValueError(f"{name!r} has no lattice in it: analyze it as a solid (verify_part)")
    if len(lats) > 1:
        raise ValueError(f"homogenized check handles one lattice node, {name!r} uses {', '.join(lats)}")
    lat = lats[0]
    lnode = g.nodes[lat]
    lspec, lrefs = lnode.spec, lnode.refs
    kind = lspec.get("type", "gyroid")
    if not is_strut(kind) and lspec.get("form", "sheet") != "sheet":
        raise ValueError(f"lattice {lat!r}: the homogenized fits are for sheet lattices, not network")
    lattice_fit(kind)                    # refuse a cell with no fit before any work
    if "aspect" in lspec:
        raise ValueError(f"lattice {lat!r}: the homogenized fits are for cubic cells, not a stretched 'aspect' "
                         "cell; check it resolved (verify_part on the mesh)")
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
    active = f_full >= 0.5
    f_skin = np.zeros(len(ijk))
    f_skin[active] = occupancy_at(g_empty, name, origin, ijk[active], offs, pitch)
    grid = np.zeros(tuple(dims), bool)
    grid[tuple(ijk[active].T)] = True
    lbl, _ = ndimage.label(grid)
    e_ijk = ijk[active]
    fs = np.clip(f_skin[active] / np.maximum(f_full[active], 1e-9), 0.0, 1.0)
    fc = 1.0 - fs
    centers = origin + (e_ijk + 0.5) * pitch
    # the lattice's density where it is
    if "density" in lrefs:
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
        e_ijk, fs, fc, rho, centers = e_ijk[keep], fs[keep], fc[keep], rho[keep], centers[keep]
        hm = HexMesh(e_ijk, pitch, origin)
        bc = boundary_conditions(study, hm, check_loads=False, tol_floor=0.5 * pitch)
    ec = sheet_modulus(kind, rho)
    e_cell = mat.E * (fs + fc * ec)
    stiff = Stiffness(hm, study.elastic(mat), bc.fixed)
    k = stiff.assemble(e_cell)
    if bc.springs is not None:
        k = (k + mat.E * bc.springs).tocsr()
    us = Solver(hm, rtol=1e-6, maxiter=3000).solve_many(k, [bc.loads[ci] for ci in range(len(study.load_cases))])
    band = 1.5 * pitch
    wall_k = ec * sheet_stress_factor(kind, rho)
    vol_cell = pitch ** 3
    volume = float(((fs + fc * rho) * vol_cell).sum())
    cases, worst = [], np.inf
    for ci, case in enumerate(study.load_cases):
        u = us[ci]
        se = von_mises(stiff.stress(u, mat.E))
        s_solid = np.where(fs > 0.05, se, 0.0)
        s_wall = np.where(fc > 0.05, se * wall_k, 0.0)
        s = np.maximum(s_solid, s_wall)
        contact = bc.support_nodes | bc.case_load_nodes[ci]
        near = cKDTree(hm.node_xyz[contact]).query(centers, k=1, distance_upper_bound=band)[0] < np.inf
        body = ~near if (~near).any() else np.ones(len(s), bool)
        i = int(np.argmax(np.where(body, s, -1.0)))
        peak = float(s[i])
        sf = mat.strength / (peak * STRESS_ALLOWANCE) if peak > 0 else np.inf
        disp = u.reshape(-1, 3)
        ln = bc.case_load_nodes[ci]
        defl = float(np.linalg.norm(disp[ln].mean(axis=0))) if ln.any() else 0.0
        worst = min(worst, sf)
        cases.append({"name": case.name, "safetyFactor": sf, "deflection": defl, "peakStress": peak,
                      "peakAt": centers[i].tolist(), "inLattice": bool(s_wall[i] >= s_solid[i]),
                      "maxDisplacement": float(np.linalg.norm(disp, axis=1).max())})
    rep = {"mass": volume * mat.density / 1000.0, "volume": volume, "safetyFactor": worst, "cases": cases,
           "bodies": bodies, "analysisPitch": pitch, "analysisCells": int(hm.n_elem),
           "latticeFraction": float((fc * vol_cell).sum() / max(len(fc) * vol_cell, 1e-12)),
           "latticeDensity": [float(rho[fc > 0.05].min()), float(rho[fc > 0.05].max())] if (fc > 0.05).any()
           else None, "lattice": lat, "stressAllowance": STRESS_ALLOWANCE, "seconds": time.time() - t0}
    if fields:
        rep["fields"] = {"centers": centers, "rho": rho, "fs": fs, "fc": fc}
    return rep


def occupancy_at(graph, name, origin, ijk, offs, pitch):
    occ = np.empty(len(ijk))
    step = max(1, 400_000 // len(offs))
    for i in range(0, len(ijk), step):
        P = (origin + (ijk[i:i + step, None, :] + offs[None]) * pitch).reshape(-1, 3)
        occ[i:i + step] = (graph.evaluate(P, name) < 0).reshape(-1, len(offs)).mean(axis=1)
    return occ


def _field(g, ref, P):
    """A lattice param at points: a number or the node it names."""
    return float(ref) if isinstance(ref, (int, float)) else np.asarray(g.evaluate(P, ref), float)
