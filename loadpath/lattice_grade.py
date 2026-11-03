"""
Lattice walls graded to the stress, for field-graph lattices, together
with the cells' direction.

`grade_lattice` replaces a lattice node's constant `thickness` with a
`density` field (a grid node) sized by fully stressed iteration on the
homogenized check (homogenized.py, anisotropic tensors):

1. analyze the part with the current density; every lattice cell's macro
   stress per load case comes back in global axes;
2. holding that stress, each cell takes the least density (of 28
   candidates between the wall limits, log-interpolated) whose wall stress
   surface, in the cell's own frame, stays under the allowable in every
   case; cells where lattice meets a skin carry the skin-junction factor,
   as the check does;
3. the requirement spreads over a quarter cell (a peak spreads over a cell
   before the lattice can follow it), moves at most halving or doubling a
   pass, and the allowable is tightened or eased by how far the analysis
   lands from the target, as lattice.lattice_infill's sizing does.

With `orient` the cells are turned too: "uniform" one direction for the
whole lattice, an integer k for k zones (stress_zones). The direction is
chosen for the least material the sized lattice needs, not for the peak:
holding each cell's macro stress, every candidate rotation's per-cell need
is a few matrix products away (the quartic features of the turned stress
dotted with every candidate density's surface), so hundreds of rotations
per zone are scored by the mass they would leave. Sizing and turning
alternate for `rounds` rounds.

The density is a grid (`op: grid`, `source` while sizing; written to an
.npz the result's graph points at), so `loadpath field` meshes the graded
lattice as sized. Stretched (`aspect`) cells and graded (warp) cells are not
graded here yet.
"""
from __future__ import annotations

import copy
import time

import numpy as np

__all__ = ["grade_lattice", "need_density"]

SOURCE = "lattice-density"
N_CAND = 28


def _cand_tensors(kind, rho_min, rho_max, aspect=None):
    from .lattice_tensors import tensors_at

    cand = np.geomspace(rho_min, rho_max, N_CAND)
    C, P, Q, F = tensors_at(kind, cand, aspect)
    return cand, P, Q, F


def _surfaces(sc, P, Q, F):
    """Wall stress surface of stresses sc (n, 6), in the cells' axes, for
    every candidate density's (P, Q, F): (n, J)."""
    from .lattice_tensors import quartic_features

    f = quartic_features(sc)
    q4 = f @ Q.T
    quad = np.sqrt(np.maximum(np.einsum("na,jab,nb->nj", sc, P, sc), 0.0))
    return np.maximum(np.maximum(q4, 0.0) ** 0.25, F[None, :] * quad)


def need_density(worst, cand, allow, over):
    """Least candidate density with worst (n, J) <= allow, log-interpolated
    between candidates; `over` where none is."""
    ok = worst <= allow
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), len(cand))
    need = np.full(len(worst), float(over))
    at = first < len(cand)
    need[at] = cand[first[at]]
    mid = at & (first > 0)
    j = first[mid]
    w0 = np.maximum(worst[mid, j - 1], 1e-30)
    w1 = np.maximum(worst[mid, j], 1e-30)
    t = np.clip((np.log(w0) - np.log(allow)) / np.maximum(np.log(w0) - np.log(w1), 1e-12), 0.0, 1.0)
    need[mid] = np.exp(np.log(cand[j - 1]) + t * (np.log(cand[j]) - np.log(cand[j - 1])))
    return need


def _cell_stress(fl, mask, frames=None):
    """Per case the lattice macro stress of the masked cells in their own
    axes: frames (n, 3, 3) columns the cell axes, or the check's own."""
    from .lattice_tensors import stress_rotation

    fr = fl["frames"] if frames is None else frames
    out = []
    for sig in fl["latticeStress"]:
        s = sig[mask]
        if fr is not None:
            s = np.einsum("eab,eb->ea", stress_rotation(fr[mask] if fr.ndim == 3 else fr), s)
        out.append(s)
    return out


def _grid_of(centers, values, origin, pitch):
    """A dense grid (origin at the first cell centre) of per-cell values,
    cells with none filled from their nearest neighbour."""
    from scipy import ndimage

    ijk = np.round((centers - origin) / pitch).astype(int)
    dims = ijk.max(axis=0) + 1
    grid = np.zeros(tuple(dims))
    have = np.zeros(tuple(dims), bool)
    grid[tuple(ijk.T)] = values
    have[tuple(ijk.T)] = True
    if not have.all():
        idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
        grid = grid[tuple(idx)]
    return grid


def grade_lattice(doc, study, target_sf=None, wall_min=None, wall_max=None, cells: int = 15_000,
                  iterations: int = 12, orient=None, rounds: int = 2, base_dir=None, node=None,
                  progress=None, n_rot: int = 800, seam: float = 1.0, confirm: str = "auto") -> dict:
    """Grade the graph's lattice wall to the stress (module docstring).
    Returns {"doc" (density as a grid node reading the in-memory source),
    "density": {"origin", "pitch", "values"}, "best" (the lightest analyzed
    design that met the target), "history", "orient", "seconds"}. `seam`:
    zones only, mm of solid plate between zones (orient.seam; 0 for none).
    `confirm`: with "auto" (default) a lattice cut open at bare faces
    (latticeExposed over lattice_orient.EXPOSED_LIMIT) is graded but not
    turned, and "guard" says why (lattice_orient.optimize_orientation);
    "none" turns it on the homogenized reading. Check a turned graded
    design yourself with resolved.resolved_check."""
    from scipy import ndimage

    from .fields import FieldGraph, expand_groups
    from .homogenized import SKIN_JUNCTION, homogenized_check, lattice_nodes
    from .lattice import relative_density
    from .lattice_orient import (_cell_group, _rotvec_deg, _smallest, candidate_rotations,
                                 dominant_directions, stress_zones)
    from .lattice_tensors import has_tensors, stress_rotation, tensors_at
    from .struts import is_strut
    from .verify import STRESS_ALLOWANCE

    t0 = time.time()
    say = progress or (lambda *_: None)
    full = expand_groups(doc)
    name = node or full.get("output") or list(full["nodes"])[-1]
    g = FieldGraph({**full, "output": name}, base_dir=base_dir)
    lats = lattice_nodes(g, name)
    if len(lats) != 1:
        raise ValueError(f"grading needs one lattice node in {name!r} (found {len(lats)})")
    lat = lats[0]
    spec = full["nodes"][lat]
    kind = spec.get("type", "gyroid")
    if "aspect" in spec:
        raise ValueError(f"lattice {lat!r}: grading a stretched (aspect) cell is not supported yet")
    if not isinstance(spec.get("cell", 8.0), (int, float)):
        raise ValueError(f"lattice {lat!r}: grading needs a constant cell")
    if not has_tensors(kind):
        raise ValueError(f"lattice {lat!r} ({kind}) has no cell tensors to grade by")
    cell = float(spec.get("cell", 8.0))
    target = float(target_sf or (study.goal.safety_factor if getattr(study, "goal", None) else 2.0) or 2.0)
    proc = getattr(study, "process", None)
    wmin = float(wall_min if wall_min is not None else max(getattr(proc, "min_feature", 0.0) or 0.0, 0.6))
    lo_lim = 0.02 if is_strut(kind) else 0.05
    rho_min = float(np.clip(relative_density(kind, wmin, cell), lo_lim, 0.95))
    rho_max = float(np.clip(relative_density(kind, wall_max, cell), rho_min, 0.95)) if wall_max else 0.6
    cand, Pc, Qc, Fc = _cand_tensors(kind, rho_min, rho_max)
    allow = study.material.strength / (target * STRESS_ALLOWANCE)
    group = _cell_group(kind)

    work = copy.deepcopy(full)
    ls = work["nodes"][lat]
    rho0 = float(np.clip(relative_density(kind, float(ls.get("thickness", 1.0)), cell), rho_min, rho_max)) \
        if not isinstance(ls.get("thickness"), str) else 0.5 * (rho_min + rho_max)
    ls.pop("thickness", None)
    dens = f"{lat}__density"
    ls["density"] = dens
    work["nodes"][dens] = {"op": "grid", "source": SOURCE}
    if isinstance(orient, (int, np.integer)) and orient > 1 or orient == "uniform":
        ls.pop("orient", None)

    def analyze(values, origin, pitch, fields=True):
        src = {SOURCE: {"origin": origin, "pitch": pitch, "values": values}}
        return homogenized_check(copy.deepcopy(work), study, node=name, base_dir=base_dir, cells=cells,
                                 fields=fields, sources=src)

    # the grid: the check's own, from one analysis at the uniform start
    probe = {SOURCE: {"origin": np.zeros(3), "pitch": 1e3, "values": np.full((2, 2, 2), rho0)}}
    rep = homogenized_check(copy.deepcopy(work), study, node=name, base_dir=base_dir, cells=cells, fields=True,
                            sources=probe)
    fl = rep["fields"]
    pitch = float(rep["analysisPitch"])
    origin = fl["centers"].min(axis=0)
    dims = np.round((fl["centers"].max(axis=0) - origin) / pitch).astype(int) + 1
    G = np.full(tuple(dims), rho0)            # the density grid (cell centres)
    history, best = [], None
    grow = max(1, int(round(0.25 * cell / pitch)))

    def ijk_of(c):
        return np.clip(np.round((c - origin) / pitch).astype(int), 0, dims - 1)

    def size(G, fl, allow):
        """The grid's next density from the analysis `fl` of it."""
        c = fl["centers"]
        ijk = ijk_of(c)
        rho = G[tuple(ijk.T)]
        latc = fl["fc"] > 0.05
        junction = (fl["fs"] > 0.05) & latc
        worst = np.zeros((len(c), len(cand)))
        for sc in _cell_stress(fl, np.ones(len(c), bool)):
            worst = np.maximum(worst, _surfaces(sc, Pc, Qc, Fc))
        worst = np.where(junction[:, None], worst * SKIN_JUNCTION, worst)
        need = np.where(latc, need_density(worst, cand, allow, rho_max), 0.0)
        g3 = np.zeros(tuple(dims))
        g3[tuple(ijk.T)] = need
        g3 = ndimage.maximum_filter(g3, size=2 * grow + 1)
        need = np.where(latc, g3[tuple(ijk.T)], rho)
        new = G.copy()
        new[tuple(ijk.T)] = np.clip(np.clip(need, 0.5 * rho, 2.0 * rho), rho_min, rho_max)
        return new

    def run_sizing(G, fl, allow, tag):
        nonlocal best
        for it in range(iterations):
            new = size(G, fl, allow)
            rep = analyze(new, origin, pitch)
            fl = rep["fields"]
            sf = float(rep["safetyFactor"])
            latc = fl["fc"] > 0.05
            r_now = new[tuple(ijk_of(fl["centers"][latc]).T)]
            row = {"stage": tag, "iteration": it, "mass": rep["mass"], "safetyFactor": sf,
                   "densityRange": [float(r_now.min()), float(r_now.max())] if len(r_now) else None,
                   "latticeTurnedExposed": rep.get("latticeTurnedExposed")}
            if rep.get("warnings"):
                row["warnings"] = rep["warnings"]
            history.append(row)
            say({"stage": "lattice-grade", **row})
            if sf >= target * 0.995 and (best is None or rep["mass"] < best["mass"]):
                best = {**row, "values": new.copy(), "orient": copy.deepcopy(work["nodes"][lat].get("orient"))}
            short = sf / target
            if it >= 1:
                allow *= float(np.clip(short, 0.85, 1.15)) ** 0.5
            converged = np.allclose(new, G, rtol=0.02, atol=0.003) and abs(short - 1.0) < 0.03
            G = new
            if converged:
                break
        return G, fl, allow

    G, fl, allow = run_sizing(G, fl, allow, "size")
    guard = None
    if orient:
        from .lattice_orient import _guard

        if confirm not in ("auto", "none"):
            raise ValueError("grade_lattice confirm is 'auto' or 'none' (check a turned design with resolved_check)")
        exposed = float(rep.get("latticeExposed") or 0.0)
        guard = _guard(confirm, exposed, True, None, None, study, "safetyFactor", name, base_dir)
        if guard is not None:
            say({"stage": "lattice-grade-guard", "reason": guard["reason"]})
            orient = None
    if orient:
        # zones start from the best single direction: one uniform round first
        k_z = 1 if orient == "uniform" else int(orient)
        plan = [1] * rounds if k_z == 1 else [1] + [k_z] * rounds
        current = [np.eye(3)]
        at = None
        for rnd, k in enumerate(plan):
            centers = fl["centers"]
            latc = fl["fc"] > 0.05
            w = fl["fc"] * fl["fill"] * pitch ** 3
            rho = G[tuple(ijk_of(centers).T)]
            if k > 1:
                from .fields import zone_weights

                if at is None:
                    _, at = stress_zones(centers[latc], [s[latc] for s in fl["latticeStress"]], w[latc], k,
                                         spatial=2.0)
                    current = [current[0]] * k
                own = np.argmax(zone_weights(at, cell, centers), axis=1)
            else:
                own = np.zeros(len(centers), int)
            rho_mid = float(np.median(rho[latc]))
            C = tensors_at(kind, [rho_mid])[0][0]
            rots = []
            for i in range(k):
                m = (own == i) & latc
                dirs, _ = dominant_directions([s[m] for s in fl["latticeStress"]], w[m])
                # the zone's current turn first (cost[0]): it is kept unless beaten
                R = np.concatenate([[current[i]], candidate_rotations(C, dirs, n_random=n_rot, seed=17 * rnd + i)])
                sig = [s[m] for s in fl["latticeStress"]]
                junction = (fl["fs"][m] > 0.05)
                vol = w[m]
                cost = np.empty(len(R))
                for j, r in enumerate(R):
                    t = stress_rotation(r)
                    worst = np.zeros((m.sum(), len(cand)))
                    for s in sig:
                        worst = np.maximum(worst, _surfaces(s @ t.T, Pc, Qc, Fc))
                    worst = np.where(junction[:, None], worst * SKIN_JUNCTION, worst)
                    need = need_density(worst, cand, allow, 2.0 * rho_max)
                    cost[j] = float((vol * need).sum())
                rots.append(_smallest(R[int(np.argmin(cost))], group))
                say({"stage": "lattice-grade-orient", "zone": i, "screened": len(R),
                     "massNeeded": float(cost.min()), "asLaid": float(cost[0])})
            current = rots
            if k > 1:
                work["nodes"][lat]["orient"] = {"zones": [{"at": [round(float(v), 3) for v in a],
                                                           "rotate": [round(float(v), 3) for v in _rotvec_deg(r)]}
                                                          for a, r in zip(at, rots)], "blend": cell}
                if seam:
                    work["nodes"][lat]["orient"]["seam"] = float(seam)
            else:
                work["nodes"][lat]["orient"] = [round(float(v), 3) for v in _rotvec_deg(rots[0])]
            # re-read the stresses in the turned lattice before sizing on
            rep = analyze(G, origin, pitch)
            fl = rep["fields"]
            G, fl, allow = run_sizing(G, fl, allow, f"size-after-orient-{rnd}")
    out = copy.deepcopy(work)
    if best is not None and best.get("orient") is not None:
        out["nodes"][lat]["orient"] = best["orient"]
    elif best is not None:
        out["nodes"][lat].pop("orient", None)
    return {"doc": out, "lattice": lat, "densityNode": dens, "source": SOURCE,
            "density": None if best is None else {"origin": origin.tolist(), "pitch": pitch,
                                                  "values": best["values"]},
            "best": None if best is None else {k: v for k, v in best.items() if k != "values"},
            "target": target, "densityLimits": [rho_min, rho_max], "history": history, "guard": guard,
            "seconds": time.time() - t0}
