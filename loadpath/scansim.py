"""
A scanner in software: scans of known geometry with the errors a real
scanner makes, for validating what scan.py measures (and for tests).

A view is a sensor position. From each, points are cast onto the surface
(only those it can see: the first hit along the ray), at the spacing the
device's depth map gives at that range, and moved along the ray by:

- random range noise, sigma_noise per point (optionally correlated over a
  patch, as a phone's fused depth is: `corr` mm);
- a systematic range bias drawn from N(0, sigma_sys), one for the whole
  scan (bias="common", the default: a calibration offset, a laser's
  subsurface scatter in plastic) or one per view ("view"); every point it
  applies to moves together, which no averaging removes;
- a scale error for the whole scan, (1 + eps), eps from N(0, scale).

Gaussian splats are made from the noisy points: a flat Gaussian per point
(tangent scales the spacing, the normal scale its thickness), random
opacity, a share of floaters, all at an arbitrary scale (a splat has no
metres), written in the 3D Gaussian splatting PLY layout.

    from loadpath.scansim import simulate, bar_and_light, write_ply
    mesh, truth = bar_and_light()
    pts, nrm, meta = simulate(mesh, views=6, spacing=1.5, sigma_noise=0.3, sigma_sys=0.1)
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["simulate", "ring_views", "bar_and_light", "write_ply", "write_splat_ply", "DEVICE_PROFILES"]

# what each device's simulated scan looks like: point spacing at the working
# distance (mm), random noise (mm), patch correlation length (mm), working
# distance (mm). The systematic sigma and scale come from scan.DEVICES, so a
# validation scan is made by the same numbers the uncertainty assumes.
DEVICE_PROFILES = {
    # ARKit depth 256 x 192 over a ~60 degree field at 0.35 m: ~1.6 mm spacing;
    # fused depth is smoothed: correlated over ~10 mm
    "iphone-lidar": {"spacing": 2.0, "noise": 1.5, "corr": 10.0, "distance": 350.0},
    "artec-eva": {"spacing": 0.5, "noise": 0.05, "corr": 0.0, "distance": 600.0},
    "leica-rtc360": {"spacing": 1.5, "noise": 0.4, "corr": 0.0, "distance": 3000.0},
}


def ring_views(center, radius, n=6, elevations=(25.0, -20.0)):
    """Sensor positions on rings round a point: n per elevation (degrees)."""
    c = np.asarray(center, float)
    out = []
    for k, el in enumerate(elevations):
        for i in range(n):
            az = 2 * math.pi * (i + 0.5 * k) / n
            e = math.radians(el)
            out.append(c + radius * np.array([math.cos(az) * math.cos(e), math.sin(az) * math.cos(e), math.sin(e)]))
    return np.array(out)


def simulate(mesh, views, *, spacing: float, sigma_noise: float, sigma_sys: float = 0.0, scale: float = 0.0,
             corr: float = 0.0, seed: int = 0, max_angle: float = 80.0, bias: str = "common"):
    """Points and outward normals as a scanner at `views` would measure them
    (see the module docstring). Returns (points, normals, meta) with meta's
    true per-view biases and scale error."""
    import trimesh
    rng = np.random.default_rng(seed)
    views = np.asarray(views, float)
    # dense surface samples at the spacing, then keep what each view sees
    n_s = int(mesh.area / (spacing * spacing) * 1.0)
    S, fi = trimesh.sample.sample_surface_even(mesh, n_s, radius=0.5 * spacing, seed=int(rng.integers(1 << 30)))
    Ns = mesh.face_normals[fi]
    eps = rng.normal(0.0, scale) if scale else 0.0
    # "common": one range bias for every view (a calibration offset, the
    # laser's subsurface scatter in plastic: the same wherever the sensor
    # is), the case no averaging helps; "view": an independent one per view
    if bias not in ("common", "view"):
        raise ValueError("bias is 'common' or 'view'")
    if not sigma_sys:
        biases = np.zeros(len(views))
    elif bias == "common":
        biases = np.full(len(views), rng.normal(0.0, sigma_sys))
    else:
        biases = rng.normal(0.0, sigma_sys, len(views))
    # a correlated noise field: random values on a coarse lattice, trilinear in between
    if corr > 0:
        lo, hi = mesh.bounds
        nl = np.ceil((hi - lo) / corr).astype(int) + 2
        lattice = rng.normal(0.0, 1.0, nl)
        # trilinear of unit-variance values has variance < 1: rescale empirically
        from scipy.ndimage import map_coordinates

        def field(X):
            g = ((X - lo) / corr).T
            return map_coordinates(lattice, g, order=1, mode="nearest")
        probe = field(lo + rng.random((4000, 3)) * (hi - lo))
        fscale = 1.0 / max(float(probe.std()), 1e-9)
    P, Nout, V = [], [], []
    for vi, s in enumerate(views):
        d = s - S
        dist = np.linalg.norm(d, axis=1)
        u = d / dist[:, None]
        facing = np.einsum("ij,ij->i", u, Ns) > math.cos(math.radians(max_angle))
        idx = np.flatnonzero(facing)
        if len(idx) == 0:
            continue
        origins = np.repeat(s[None], len(idx), 0)
        dirs = -u[idx]
        locs, ray_i, tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        hit_d = np.full(len(idx), np.inf)
        hit_d[ray_i] = np.linalg.norm(locs - s, axis=1)
        seen = idx[np.abs(hit_d - dist[idx]) < 0.25 * spacing]
        # subsample: each view sees the surface at its own spacing (not all of the dense set)
        seen = seen[rng.random(len(seen)) < 1.0 / max(1.0, len(views) / 3.0)]
        ray = -u[seen]
        r_noise = rng.normal(0.0, sigma_noise, len(seen))
        if corr > 0:
            r_noise = 0.5 * r_noise + math.sqrt(0.75) * sigma_noise * fscale * field(S[seen])
        dr = r_noise + biases[vi]                 # a range error moves the point along its ray
        X = S[seen] + ray * dr[:, None]
        P.append(X)
        Nout.append(Ns[seen])
        V.append(np.repeat(s[None], len(seen), 0))
    P = np.concatenate(P)
    c = P.mean(0)
    P = c + (P - c) * (1.0 + eps)
    return P, np.concatenate(Nout), {"biases": biases.tolist(), "scaleError": float(eps),
                                     "viewpoints": np.concatenate(V), "center": c.tolist()}


def bar_and_light(bar_d: float = 31.8, bar_len: float = 220.0, light=(60.0, 34.0, 30.0), gap: float = 18.0,
                  insert_d: float = 6.0, card: bool = False):
    """A handlebar (a tube along x) and a bike light hanging in front of
    it below the bar, its bottom face with a threaded-insert bore: the scene
    the scan tutorial builds a mount around. card: an ID-1 card (85.60 x
    53.98 x 0.76 mm) propped beside the light, for a scan scaled by it.
    Returns (mesh, truth)."""
    import trimesh
    from .shapes import manifold_from_trimesh, trimesh_from_manifold
    bar = trimesh.creation.cylinder(radius=bar_d / 2, height=bar_len, sections=256)
    bar.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    lx, ly, lz = light
    # in front of the bar (+y), its top a gap below the bar's bottom
    cy = bar_d / 2 + 6.0 + ly / 2
    top = -bar_d / 2 - gap
    box = trimesh.creation.box(extents=light)
    box.apply_translation([0, cy, top - lz / 2])
    hole = trimesh.creation.cylinder(radius=insert_d / 2, height=16.0, sections=128)
    hole.apply_translation([0, cy, top])
    lm = manifold_from_trimesh(box) - manifold_from_trimesh(hole)
    body = trimesh_from_manifold(lm)
    parts = [bar, body]
    card_truth = None
    if card:
        c = trimesh.creation.box(extents=[53.98, 85.60, 0.76])
        c.apply_transform(trimesh.transformations.rotation_matrix(math.radians(35.0), [0, 1, 0]))
        c.apply_translation([80.0, cy, top - 10.0])
        parts.append(c)
        card_truth = {"size": [85.60, 53.98, 0.76], "center": [80.0, cy, top - 10.0]}
    scene = trimesh.util.concatenate(parts)
    truth = {"bar": {"diameter": bar_d, "axis": [1, 0, 0], "point": [0, 0, 0]},
             "lightTop": {"normal": [0, 0, 1], "z": top},
             "lightFront": {"normal": [0, 1, 0], "y": cy + ly / 2},
             "insert": {"diameter": insert_d, "axis": [0, 0, 1], "point": [0, cy, top]},
             "light": {"size": list(light), "center": [0, cy, top - lz / 2]}}
    if card_truth:
        truth["card"] = card_truth
    return scene, truth


def write_ply(path, points, normals=None):
    """A binary point-cloud PLY (x y z [nx ny nz], float32)."""
    P = np.asarray(points, np.float32)
    cols = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    data = [P[:, 0], P[:, 1], P[:, 2]]
    if normals is not None:
        Nn = np.asarray(normals, np.float32)
        cols += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        data += [Nn[:, 0], Nn[:, 1], Nn[:, 2]]
    arr = np.empty(len(P), dtype=cols)
    for (k, _), v in zip(cols, data):
        arr[k] = v
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(P) + \
        "".join(f"property float {k}\n" for k, _ in cols) + "end_header\n"
    with open(path, "wb") as fh:
        fh.write(hdr.encode())
        fh.write(arr.tobytes())
    return path


def write_splat_ply(path, points, normals, spacing, thickness, *, scale_factor=1.0, floaters=0.02, seed=0):
    """Gaussians from scan points, as a splat trainer would leave them: flat
    discs (tangent scales ~ the spacing, normal scale `thickness`), opacity
    0.3-1, plus a share of large faint floaters; coordinates times
    scale_factor (a splat's arbitrary units)."""
    rng = np.random.default_rng(seed)
    P = np.asarray(points, float)
    Nn = np.asarray(normals, float)
    n = len(P)
    nf = int(floaters * n)
    lo, hi = P.min(0), P.max(0)
    Pf = lo + rng.random((nf, 3)) * (hi - lo) * 1.3 - 0.15 * (hi - lo)
    # rotation taking z to the normal; the disc's tangent axes random about it
    def quat_from_z(nv):
        z = np.array([0, 0, 1.0])
        v = np.cross(z, nv)
        c = nv @ z
        if np.linalg.norm(v) < 1e-9:
            return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0, 1.0, 0, 0])
        s = math.sqrt((1 + c) * 2)
        return np.array([s / 2, v[0] / s, v[1] / s, v[2] / s])
    Q = np.array([quat_from_z(v) for v in Nn])
    sc = np.stack([np.full(n, spacing), np.full(n, spacing), np.full(n, thickness)], 1) * rng.uniform(0.7, 1.3, (n, 3))
    op = rng.uniform(0.3, 1.0, n)
    Qf = rng.normal(size=(nf, 4))
    scf = np.full((nf, 3), 6 * spacing)
    opf = rng.uniform(0.01, 0.15, nf)
    allP = np.concatenate([P, Pf]) * scale_factor
    allQ = np.concatenate([Q, Qf])
    allS = np.concatenate([sc, scf]) * scale_factor
    allO = np.concatenate([op, opf])
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(len(allP), dtype=[(k, "<f4") for k in names])
    arr["x"], arr["y"], arr["z"] = allP.T
    arr["opacity"] = np.log(allO / (1 - allO))
    for i in range(3):
        arr[f"scale_{i}"] = np.log(allS[:, i])
        arr[f"f_dc_{i}"] = 0.5
    for i in range(4):
        arr[f"rot_{i}"] = allQ[:, i]
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(allP) + \
        "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with open(path, "wb") as fh:
        fh.write(hdr.encode())
        fh.write(arr.tobytes())
    return path
