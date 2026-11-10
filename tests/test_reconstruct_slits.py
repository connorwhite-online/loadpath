"""reconstruct() and hairline slits: where a design space's face lies on the
fine grid the field sits on the level there, marching cubes leaves
zero-area triangles, and the surface smoothing must not fold the part
back through itself (the fan bracket's slits, loadpath 9815ad1).

Run: python -m pytest -q tests/test_reconstruct_slits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath.domain import build_domain
from loadpath.reconstruct import reconstruct
from loadpath.spec import study_from_dict


def _step():
    # The fan bracket's reach in small: a slab with a tower standing on it,
    # one union (the slab's top is an internal face under the tower and a
    # real one beside it), and a member that runs down the tower and out
    # along the slab under its top. At pitch 2 the top (z 13) lies on the
    # fine grid's nodes, as the bracket's z 12 did.
    def box(lo, hi):
        return {"type": "box", "min": lo, "max": hi}
    spec = {
        "name": "step",
        "designSpace": [{"type": "union", "shapes": [box([-10, -34, 0], [10, 10, 13]), box([-10, -10, 0], [10, 10, 20])]}],
        "supports": [{"shape": box([-11, -35, -1], [11, -28, 0.01])}],
        "loadCases": [{"name": "a", "loads": [{"shape": box([-6, -6, 19.99], [6, 6, 21]), "force": [0, 0, -100]}]}],
        "material": "pa12", "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 2.0}, "minMember": 4.0}
    s = study_from_dict(spec)
    d = build_domain(s)
    c = d.origin + (d.mesh.elem_ijk + 0.5) * d.pitch
    x = np.zeros(len(c))
    x[(np.abs(c[:, 0]) < 5) & (np.abs(c[:, 1]) < 5)] = 1                   # column in the tower
    x[(np.abs(c[:, 0]) < 5) & (c[:, 1] < 0) & (c[:, 2] > 4)] = 1           # beam under the slab's top
    x[(np.abs(c[:, 0]) < 5) & (c[:, 1] < -28) & (c[:, 2] < 6)] = 1         # foot on the support
    return s, d, x


def _self_crossings(mesh, step=0.5, tol=1e-5):
    """Points where a mesh's surface passes through itself. A closed surface
    that does not cuts every plane in loops that never cross, so: sections
    every `step` mm along each axis, and the pairs of segments in each whose
    ends lie more than `tol` mm either side of the other's line (segments
    that meet end to end, or run along one flat face, do not count)."""
    from trimesh.intersections import mesh_plane

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    lo, hi = mesh.bounds
    out = []
    for a in range(3):
        n, oth = np.eye(3)[a], [i for i in range(3) if i != a]
        for at in np.arange(lo[a] + 0.123, hi[a], step):
            s = mesh_plane(mesh, n, n * at)
            if len(s) < 2:
                continue
            p, r = s[:, 0][:, oth], (s[:, 1] - s[:, 0])[:, oth]
            L = np.maximum(np.linalg.norm(r, axis=1), 1e-12)
            q, rr = p[None] - p[:, None], cross(r[:, None], r[None])          # p_j - p_i, r_i x r_j
            a0 = cross(r[:, None], q) / L[:, None]                             # j's ends from i's line
            a1 = a0 + rr / L[:, None]
            b0 = cross(r[None], q) / L[None]                                   # i's ends from j's line
            b1 = b0 + rr / L[None]
            hit = (a0 * a1 < 0) & (np.minimum(abs(a0), abs(a1)) > tol) & \
                  (b0 * b1 < 0) & (np.minimum(abs(b0), abs(b1)) > tol)
            for i, j in zip(*np.nonzero(np.triu(hit, 1))):
                out.append(np.insert(p[j] + a0[i, j] / (a0[i, j] - a1[i, j]) * r[j], a, at))
    return np.array(out).reshape(-1, 3)


def test_no_slit_where_a_member_meets_a_union_face_on_the_fine_grid(monkeypatch):
    # The fan bracket (MJF PA12) came out of 9815ad1 with hairline slits,
    # 0.06-0.2 mm across, through its arms at the reach's top face (z 12)
    # and the upper arms' faces: the field sat on the level along them,
    # marching cubes left zero-area triangles, and trimesh's smoothing
    # operator pulled their vertices 30-150 mm toward the world origin,
    # folding the surface through solid material. Watertight, one body, and
    # a tet check read 90-260 MPa cracks. The on-level guard now moves such
    # nodes off the level; the smoothing must hold even when it does not.
    import skimage.measure
    s, d, x = _step()
    ref, info = reconstruct(s, d, x)
    assert info["watertight"] and info["bodies"] == 1, info
    p = _self_crossings(ref)
    assert len(p) == 0, (len(p), np.round(p[:5], 1).tolist())
    # the field as it reached marching cubes before the guard: on the level
    # at the top face's nodes (here it threw vertices 29 mm, crossed itself
    # in 362 places and lost 12% of the volume)
    real, put = skimage.measure.marching_cubes, []

    def on_level(vol, level, **k):
        back = vol == level - 1e-3              # what the guard moved off the level
        put.append(int(back.sum()))
        return real(np.where(back, level, vol), level, **k)
    monkeypatch.setattr(skimage.measure, "marching_cubes", on_level)
    mesh, info = reconstruct(s, d, x)
    assert put and put[0] > 20, put
    p = _self_crossings(mesh)
    assert len(p) == 0, (len(p), np.round(p[:5], 1).tolist())
    assert abs(mesh.volume - ref.volume) < 0.01 * ref.volume, (mesh.volume, ref.volume)


def test_smoothing_operator_is_trimeshs_on_a_clean_mesh_and_averages_on_a_degenerate_one():
    # no change where there was nothing wrong: every eval part meshes as before
    import loadpath.reconstruct as R
    m = trimesh.creation.icosphere(2)
    assert abs(R._umbrella(m) - trimesh.smoothing.laplacian_calculation(m).tocsr()).max() == 0.0
    # a face with a repeated vertex, as marching cubes leaves one on the level
    bad = trimesh.Trimesh(m.vertices, np.vstack([m.faces, [[0, 0, 1]]]), process=False)
    assert np.allclose(np.asarray(R._umbrella(bad).sum(axis=1)).ravel(), 1.0)
