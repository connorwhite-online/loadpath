"""
A stand-in for an AI-generated exterior (Tripo, Meshy, ...): a palm-sized
Bluetooth shutter remote, a pebble with a dished press button, grip ridges
along its sides and a lanyard loop, written out the way a generator hands
it over:

- unit scale (longest side 1.0, not 72 mm) and Y-up (glTF), as GLB;
- open: three holes torn in the surface (the largest 6 mm across);
- a duplicate inner shell 0.6 mm in and a patch doubled 0.3 mm proud of the surface
  (self-intersecting);
- a patch of flipped triangles; jitter on every vertex;
- floating debris near the surface.

The clean shape (`pebble_sdf`) is kept so the wrap can be checked against
the truth: `python generated_standin.py --out generated.glb`.
"""
import argparse

import numpy as np

L, W, H = 72.0, 46.0, 22.0       # mm


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def pebble_sdf(P):
    """Signed distance (mm, negative inside), +z up, resting on z = 0."""
    P = np.asarray(P, float)
    x, y, z = P[:, 0], P[:, 1], P[:, 2] - H / 2
    # a superellipsoid-ish pebble, flatter on the bottom
    zz = np.where(z < 0, z * 1.25, z)
    n = 2.6
    r = (np.abs(x / (L / 2)) ** n + np.abs(y / (W / 2)) ** n) ** (1 / n)
    d = (np.sqrt(r ** 2 + (zz / (H / 2)) ** 2) - 1.0) * min(W, H) / 2 * 0.95
    # grip ridges along both long sides
    side = np.clip((np.abs(y) - (W / 2 - 6)) / 6, 0, 1) * np.clip(1 - np.abs(x) / 26, 0, 1)
    d += 0.35 * side * np.sin(x * 2 * np.pi / 3.2)
    # the dished button: a shallow spherical dent on top, ahead of centre
    bx, R = 12.0, 40.0
    dish = np.sqrt((x - bx) ** 2 + y ** 2 + (z - (H / 2 + R - 1.2)) ** 2) - R
    d = np.maximum(d, -dish)
    # a raised rim around the dish, 0.6 mm
    ring = np.abs(np.sqrt((x - bx) ** 2 + y ** 2) - 11.5) - 0.7
    top = z - (H / 2 - 0.2)
    d = _smin(d, np.maximum(ring, np.maximum(top - 0.6, -top - 3)), 0.6)
    # a lanyard loop off the back end
    q = np.column_stack([x + L / 2 + 1.0, y, z - 2.0])
    loop = np.sqrt((np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - 4.2) ** 2 + q[:, 2] ** 2) - 1.4
    loop = np.maximum(loop, x + L / 2 - 1.5)
    return _smin(d, loop, 1.0)


BOUNDS = (np.array([-L / 2 - 8, -W / 2 - 2, -1.0]), np.array([L / 2 + 2, W / 2 + 2, H + 2.5]))


def clean_mesh(pitch=0.3):
    from loadpath.exterior import mesh_from_sdf
    return mesh_from_sdf(pebble_sdf, BOUNDS, pitch)[0]


def break_it(m, seed=0):
    """What a generator does to it."""
    import trimesh
    rng = np.random.default_rng(seed)
    V = m.vertices.copy()
    F = m.faces.copy()
    c = m.triangles_center
    keep = np.ones(len(F), bool)
    # three holes: under the lanyard, on a flank, on the bottom
    for at, r in (((-20, 25, 11), 3.0), ((5, -10, 0), 2.0), ((30, 5, 25), 1.2)):
        at = m.nearest.on_surface([at])[0][0]
        keep &= np.linalg.norm(c - at, axis=1) > r
    flip = np.linalg.norm(c - np.array([-15, -5, 21.5]), axis=1) < 4
    F2 = F.copy()
    F2[flip] = F2[flip][:, ::-1]
    parts = [trimesh.Trimesh(V, F2[keep], process=False)]
    # a duplicate inner shell 0.6 mm in (a generator's double wall)
    inner = m.copy()
    inner.vertices = inner.vertices - 0.6 * inner.vertex_normals
    parts.append(inner)
    # a patch doubled 0.3 mm proud: crosses the surface at its edge
    sel = np.linalg.norm(c - np.array([25, 18, 12]), axis=1) < 5
    patch = m.submesh([np.nonzero(sel)[0]], append=True)
    patch.vertices = patch.vertices + 0.3 * patch.vertex_normals * np.linspace(-1, 1, len(patch.vertices))[:, None]
    parts.append(patch)
    # debris
    for k in range(4):
        s = trimesh.creation.icosphere(1, radius=rng.uniform(0.3, 0.8))
        s.vertices += np.array([rng.uniform(-30, 30), rng.uniform(-28, 28), rng.uniform(25, 30)])
        parts.append(s)
    g = trimesh.util.concatenate(parts)
    g.vertices = g.vertices + rng.normal(0, 0.03, g.vertices.shape)
    # unit scale, Y-up (z -> y), centred on the origin
    g.vertices -= g.bounds.mean(0)
    g.vertices /= float(g.extents.max())
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)     # z up -> y up
    g.vertices = g.vertices @ R.T
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="generated.glb")
    ap.add_argument("--clean", help="also write the clean shape (for the check)")
    a = ap.parse_args()
    m = clean_mesh()
    if a.clean:
        m.export(a.clean)
    g = break_it(m)
    g.export(a.out)
    print(f"{a.out}: {len(g.faces)} faces, extents {np.round(g.extents, 3).tolist()}")
