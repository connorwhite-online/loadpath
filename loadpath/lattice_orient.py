"""
Lattice orientation as a design variable.

A strut or sheet cell is stiff and strong along some directions and soft
along others (a simple cubic cell carries a pull along its rods at about
ten times the stiffness it has along the cube diagonal; BCCz carries its
vertical struts' axis). homogenized_check reads a lattice from its full
tensors (lattice_tensors), so it can tell how the cell lies relative to the
load. `optimize_orientation` turns the cell (the lattice node's `orient`,
which rotates the cells and leaves the fields grading them where they are)
for the best worst-case safety factor, or the least compliance:

1. One homogenized analysis with the cell as it is gives every lattice
   cell's macro stress per load case.
2. Screen: holding those macro stresses (the assumption fully stressed
   sizing makes), each candidate rotation's peak wall stress (the stress
   surface, as a p-norm over the cells the peak is read over) and
   complementary energy are a rotation of tensors away, so thousands of
   rotations cost a few seconds: rotations that put each of the cell's
   stiffest directions on the dominant stress direction, spun about it,
   and a random spread over all rotations.
3. The best few go through a full homogenized analysis each (the load
   moves onto what got stiffer), then Nelder-Mead on the rotation vector
   refines the best of those, again on full analyses.

The orientation is uniform over the lattice. It wins most where one stress
direction dominates (a beam's skinless core, a strut, a bracket arm): see
docs/fields.md. A gyroid is close to isotropic, so it gains little.
"""
from __future__ import annotations

import copy
import time

import numpy as np

__all__ = ["optimize_orientation", "screen_rotations", "dominant_directions"]

# directions the cell's stiff axes are looked for along (cube axes, face
# and body diagonals; with a stretched cell these are in its own axes)
_LATTICE_DIRS = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1),
                          (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
                          (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)], float)
_LATTICE_DIRS /= np.linalg.norm(_LATTICE_DIRS, axis=1, keepdims=True)

P_NORM = 8.0          # the screen's soft maximum of the wall stress


def _voigt_tensor(s):
    """(n, 6) Voigt stresses (xx, yy, zz, yz, xz, xy) -> (n, 3, 3)."""
    t = np.empty((len(s), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    t[:, 1, 2] = t[:, 2, 1] = s[:, 3]
    t[:, 0, 2] = t[:, 2, 0] = s[:, 4]
    t[:, 0, 1] = t[:, 1, 0] = s[:, 5]
    return t


def dominant_directions(stresses, weights):
    """Principal axes of sum_e w_e sigma_e . sigma_e over cells and cases,
    strongest first (columns): the directions the lattice's stress mostly
    runs along, whatever its sign."""
    m = np.zeros((3, 3))
    for s in stresses:
        t = _voigt_tensor(s)
        m += np.einsum("e,eij,ejk->ik", weights, t, t)
    w, v = np.linalg.eigh(m)
    return v[:, ::-1], w[::-1]


def _align(a, b):
    """A rotation taking unit vector a onto unit vector b."""
    from scipy.spatial.transform import Rotation

    c = np.cross(a, b)
    s, d = np.linalg.norm(c), float(np.dot(a, b))
    if s < 1e-12:
        if d > 0:
            return np.eye(3)
        p = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
        return Rotation.from_rotvec(np.pi * p / np.linalg.norm(p)).as_matrix()
    return Rotation.from_rotvec(c / s * np.arctan2(s, d)).as_matrix()


def candidate_rotations(C, dirs, n_spin=12, n_random=1500, seed=0):
    """Rotations (k, 3, 3), columns the cell's axes in global components:
    the identity; each of the cell's stiffest lattice directions (within
    15% of the stiffest) onto the dominant stress direction, spun about it
    in n_spin steps; random ones."""
    from scipy.spatial.transform import Rotation

    from .lattice_tensors import directional_modulus

    e = np.array([directional_modulus(C, d) for d in _LATTICE_DIRS])
    stiff = _LATTICE_DIRS[e >= 0.85 * e.max()]
    out = [np.eye(3)]
    for d in stiff:
        base = _align(d, dirs[:, 0])
        for k in range(n_spin):
            spin = Rotation.from_rotvec(dirs[:, 0] * np.pi * k / n_spin).as_matrix()
            out.append(spin @ base)
    out += list(Rotation.random(n_random, random_state=seed).as_matrix())
    return np.array(out)


def screen_rotations(R, fields, aspect=None):
    """Score rotations holding each lattice cell's macro stress. Returns
    (peak, energy): per rotation the p-norm of the wall stress over the
    cells the peak is read over (worst case) and the complementary energy
    summed over cases, both relative to the cells' tensors in their current
    frames turned by R (the cell axes are frames @ R)."""
    from .lattice_tensors import stress_rotation, stress_surface, tensors_at

    rho, fc, fill = fields["rho"], fields["fc"], fields["fill"]
    lat = fc > 0.05
    C, P, Q, F = tensors_at(fields["kind"], rho[lat], aspect)
    S = np.linalg.inv(C)
    w = (fc * fill)[lat]
    frames = fields["frames"]
    f0 = np.broadcast_to(np.eye(3), (lat.sum(), 3, 3)) if frames is None else frames[lat]
    peak = np.zeros(len(R))
    energy = np.zeros(len(R))
    for ci, sig in enumerate(fields["latticeStress"]):
        sig = sig[lat]
        body = fields["body"][ci][lat]
        for k, r in enumerate(R):
            t = stress_rotation(np.einsum("eij,jk->eik", f0, r))
            sc = np.einsum("eab,eb->ea", t, sig)
            s = stress_surface(sc[body], P[body], Q[body], F[body])
            m = s.max() if len(s) else 0.0
            pn = m * (np.sum(w[body] * (s / max(m, 1e-30)) ** P_NORM) / max(w[body].sum(), 1e-30)) ** (1 / P_NORM)
            peak[k] = max(peak[k], pn)
            energy[k] += float(np.einsum("e,ea,eab,eb->", w, sc, S, sc))
    return peak, energy


def _cell_group(kind, aspect=None):
    from .lattice_tensors import _group, symmetry_of

    sym = "orthorhombic" if aspect is not None and not np.allclose(aspect, aspect[0]) else symmetry_of(kind)
    return _group(sym) if sym != "none" else np.eye(3)[None]


def _angle_mod(a, b, group):
    """Least rotation angle between cell frames a and b, the cell's
    symmetry rotations (in its own axes) taken as no turn."""
    tr = np.einsum("ji,jk,gki->g", a, b, group)
    return float(np.arccos(np.clip((tr.max() - 1.0) / 2.0, -1.0, 1.0)))


def _smallest(r, group):
    """The frame r @ g (g a symmetry rotation) nearest the identity."""
    tr = np.einsum("ij,gji->g", r, group)
    return r @ group[int(np.argmax(tr))]


def _with_orient(doc, lat, rv):
    d = copy.deepcopy(doc)
    nodes = d["nodes"]
    spec = nodes[lat]
    if np.allclose(rv, 0.0):
        spec.pop("orient", None)
    else:
        spec["orient"] = [round(float(x), 3) for x in rv]
    return d


def _rotvec_deg(R):
    from scipy.spatial.transform import Rotation

    return np.degrees(Rotation.from_matrix(R).as_rotvec())


def optimize_orientation(doc, study, node=None, base_dir=None, cells: int = 15_000, objective: str = "safetyFactor",
                         full_checks: int = 4, refine: int = 24, progress=None) -> dict:
    """Turn the graph's lattice (its `orient`) for the best worst-case
    safety factor (objective "safetyFactor") or the least total compliance
    ("stiffness"), on homogenized analyses of `cells` cells (module
    docstring). Returns {"doc" (the graph with the best orient written),
    "orient" (degrees, rotation vector), "start", "best" (homogenized
    reports, trimmed), "screened", "evaluations", "history", "seconds"}.
    The lattice must be cartesian with a constant cell (orient's limits);
    an orient it already has is the starting point."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    from .fields import FieldGraph, expand_groups
    from .homogenized import homogenized_check, lattice_nodes
    from .lattice_tensors import has_tensors, tensors_at

    if objective not in ("safetyFactor", "stiffness"):
        raise ValueError("objective is 'safetyFactor' or 'stiffness'")
    t0 = time.time()
    full = expand_groups(doc)
    name = node or full.get("output") or list(full["nodes"])[-1]
    g = FieldGraph({**full, "output": name}, base_dir=base_dir)
    lats = lattice_nodes(g, name)
    if len(lats) != 1:
        raise ValueError(f"orientation needs one lattice node in {name!r} (found {len(lats)})")
    lat = lats[0]
    spec = full["nodes"][lat]
    kind = spec.get("type", "gyroid")
    aspect = tuple(spec["aspect"]) if "aspect" in spec else None
    if not has_tensors(kind, aspect):
        raise ValueError(f"lattice {lat!r} ({kind}) has no cell tensors to orient by")
    say = progress or (lambda *_: None)
    r_start = (Rotation.from_rotvec(np.radians(np.asarray(spec["orient"], float))).as_matrix()
               if isinstance(spec.get("orient"), list) else np.eye(3))
    if isinstance(spec.get("orient"), dict):
        from .fields import lattice_orientation

        r_start = lattice_orientation(g.nodes[lat])
        r_start = np.eye(3) if r_start is None else r_start
    base = _with_orient(full, lat, np.zeros(3))
    history = []

    def score(rep):
        if objective == "safetyFactor":
            return -float(rep["safetyFactor"])
        return float(sum(c["compliance"] for c in rep["cases"]))

    def trim(rep, rv):
        return {"orient": [float(x) for x in rv], "mass": rep["mass"], "safetyFactor": rep["safetyFactor"],
                "compliance": float(sum(c["compliance"] for c in rep["cases"])),
                "cases": [{k: c[k] for k in ("name", "safetyFactor", "deflection", "compliance")}
                          for c in rep["cases"]]}

    cache = {}

    def check(rv):
        key = tuple(np.round(rv, 2))
        if key not in cache:
            rep = homogenized_check(_with_orient(base, lat, rv), study, node=name, base_dir=base_dir, cells=cells)
            cache[key] = (score(rep), trim(rep, rv))
            history.append(cache[key][1])
            say({"stage": "lattice-orient", "orient": np.round(rv, 1).tolist(),
                 "safetyFactor": rep["safetyFactor"], "compliance": history[-1]["compliance"]})
        return cache[key]

    # the cell as it lies (no orient) for the macro stresses and frames
    ref = homogenized_check(base, study, node=name, base_dir=base_dir, cells=cells, fields=True)
    fl = ref["fields"]
    if fl["latticeStress"] is None:
        raise ValueError("the anisotropic reading is off (LOADPATH_LATTICE_ANISO=0): orientation needs it")
    lat_cells = fl["fc"] > 0.05
    dirs, strength = dominant_directions([s[lat_cells] for s in fl["latticeStress"]],
                                         (fl["fc"] * fl["fill"])[lat_cells])
    rho_mid = float(np.median(fl["rho"][lat_cells]))
    C = tensors_at(kind, [rho_mid], aspect)[0][0]
    # candidates are the cell's rotation from its unrotated lie; the stiff
    # directions are in the cell's axes, the stress directions in global
    f0 = fl["frames"]
    f_mean = np.eye(3) if f0 is None else f0[lat_cells].mean(axis=0)
    u, _, vt = np.linalg.svd(f_mean)
    f_mean = u @ vt
    R = candidate_rotations(C, f_mean.T @ dirs)
    R = np.concatenate([[r_start], R])
    peak, energy = screen_rotations(R, fl, aspect)
    rank = np.argsort(peak if objective == "safetyFactor" else energy)
    group = _cell_group(kind, aspect)
    picks = [np.eye(3)]                  # the unrotated cell is checked anyway
    for k in rank:
        # one of each set the cell's symmetry makes the same, written as
        # its smallest turn
        if min(_angle_mod(R[k], p, group) for p in picks) < np.radians(8.0):
            continue
        picks.append(_smallest(R[k], group))
        if len(picks) > full_checks:
            break
    picks = [_rotvec_deg(p) for p in picks[1:]]
    start_rv = _rotvec_deg(r_start)
    start = check(start_rv)
    check(np.zeros(3))
    for rv in picks:
        check(rv)
    best_key = min(cache, key=lambda k: cache[k][0])
    x0 = np.array(best_key, float)
    if refine > 0:
        simplex = [x0] + [x0 + 12.0 * e for e in np.eye(3)]
        minimize(lambda x: check(x)[0], x0, method="Nelder-Mead",
                 options={"initial_simplex": simplex, "maxfev": refine, "xatol": 1.0, "fatol": 1e-3})
    best_key = min(cache, key=lambda k: cache[k][0])
    best = cache[best_key][1]
    out_doc = _with_orient(copy.deepcopy(doc) if "nodes" in doc and lat in doc["nodes"] else full, lat,
                           np.array(best_key, float))
    return {"doc": out_doc, "orient": [float(x) for x in best_key], "lattice": lat, "objective": objective,
            "start": start[1], "unrotated": cache[tuple(np.round(np.zeros(3), 2))][1], "best": best,
            "screened": int(len(R)), "dominantStress": dirs[:, 0].tolist(),
            "stressAnisotropy": float(strength[0] / max(strength.sum(), 1e-30)),
            "evaluations": len(history), "history": history, "seconds": time.time() - t0}
