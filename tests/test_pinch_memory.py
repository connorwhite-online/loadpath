"""Pinch filling's STL edge count and its incremental rounds, and verify's
submodel memory gate (the full-tier enclosure cases ran out of memory in
both)."""
import threading
import time

import numpy as np
import pytest

m3d = pytest.importorskip("manifold3d")
trimesh = pytest.importorskip("trimesh")

from loadpath import reconstruct as R  # noqa: E402
from loadpath.verify import _MemoryGate  # noqa: E402


def _kissing(count=12):
    """Pairs of cubes that touch along an edge, apart from each other."""
    parts = []
    for i in range(count):
        x = np.array([10.0 * (i % 4), 10.0 * (i // 4), 0.0])
        parts += [m3d.Manifold.cube((2, 2, 2)).translate(tuple(x)),
                  m3d.Manifold.cube((2, 2, 2)).translate(tuple(x + [2, 2, 0]))]
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


def _tangled(seed=1, count=10):
    """Cubes that overlap and kiss each other and a sphere at random: balls
    leave contacts of their own, so pinch filling takes several rounds."""
    rng = np.random.default_rng(seed)
    parts = [m3d.Manifold.sphere(10, 32)]
    for _ in range(count):
        x = rng.uniform(-14, 14, 3)
        parts += [m3d.Manifold.cube((2, 2, 2)).translate(tuple(x + o)) for o in ([0, 0, 0], [2, 2, 0], [2, -2, 2])]
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


def _trimesh_bad(man):
    m = R.trimesh_from_manifold(man)
    m.vertices = m.vertices.astype(np.float32).astype(float)
    m.merge_vertices()
    u, c = np.unique(m.edges_sorted, axis=0, return_counts=True)
    bad = u[(c > 2) | (c == 1)]
    return {tuple(x) for x in np.round(m.vertices[bad].mean(axis=1), 6)}


def test_bad_edges_match_an_stl_readers_weld():
    part = _kissing()
    mg = part.to_mesh64()
    bad, wv = R._stl_bad_edges(np.asarray(mg.vert_properties)[:, :3], np.asarray(mg.tri_verts, np.int64))
    assert len(bad) > 0
    assert {tuple(x) for x in np.round(wv[bad].mean(axis=1), 6)} == _trimesh_bad(part)
    assert not R._stl_watertight(R.trimesh_from_manifold(part))
    assert R._stl_watertight(trimesh.creation.box((1, 2, 3)))


def test_incremental_rounds_find_every_contact_the_whole_mesh_has(monkeypatch):
    """Each round after the first reads only near the last round's balls; it
    must find the contacts a pass over the whole part finds."""
    calls = []
    real = R._round_contacts

    def spy(verts, faces, last=None, reach=0.0, ends=None):
        out = real(verts, faces, last, reach, ends)
        calls.append((verts, faces, last, out))
        return out
    monkeypatch.setattr(R, "_round_contacts", spy)
    R._fill_pinches(_tangled(), rounds=4)
    later = [c for c in calls if c[2] is not None]
    assert later
    for verts, faces, _, (bad, wv) in later:
        full, fw = real(verts, faces)
        mids = lambda b, w: {tuple(x) for x in np.round(w[b].mean(axis=1), 6)}  # noqa: E731
        assert mids(bad, wv) == mids(full, fw)


def test_pinches_filled_read_watertight():
    part = _kissing()
    assert not R._stl_watertight(R.trimesh_from_manifold(part))
    filled, n = R._fill_pinches(part)
    assert n >= 12
    assert len(filled.decompose()) == 12     # each pair one body
    assert R._stl_watertight(R.trimesh_from_manifold(filled))


def test_memory_gate_holds_back_what_does_not_fit():
    gate = _MemoryGate(10.0)
    live, most = [0], [0]
    lock = threading.Lock()

    def job(need):
        got = gate.take(need)
        with lock:
            live[0] += 1
            most[0] = max(most[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        gate.give(got)

    ts = [threading.Thread(target=job, args=(6.0,)) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert most[0] == 1          # two 6s never fit in 10
    # a lone job bigger than the budget still runs
    gate.give(gate.take(50.0))
    assert gate.held == 0.0


def test_material_peak_matches_averaging_every_node(monkeypatch):
    """The pruned, blocked search finds the same highest ball average as
    averaging every node does (a sharp spike reads low, a broad rise high)."""
    from scipy.spatial import cKDTree

    from loadpath import verify
    from loadpath.failure import equivalent_stress

    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 10, (6000, 3))
    w = rng.uniform(0.5, 1.5, len(xs))
    broad = 20.0 * np.exp(-np.sum((xs - [3, 3, 3]) ** 2, axis=1) / 4.0)
    spike = 60.0 * np.exp(-np.sum((xs - [7, 7, 7]) ** 2, axis=1) / 0.01)
    sig = np.zeros((len(xs), 6))
    sig[:, 0] = broad + spike + rng.normal(0, 0.5, len(xs))
    nodes = xs[::3]
    trusted = np.ones(len(nodes), bool)
    vm = equivalent_stress(sig[::3])
    monkeypatch.setattr(verify, "_AVG_BLOCK", 50)
    i, val = verify._material_peak(nodes, trusted, vm, xs, w, sig, 0.86)
    nb = cKDTree(xs).query_ball_point(nodes, 0.86)
    every = np.array([equivalent_stress((w[j, None] * sig[j]).sum(0, keepdims=True) / w[j].sum())[0] for j in nb])
    assert abs(val - every.max()) < 1e-9
    assert np.linalg.norm(nodes[i] - [3, 3, 3]) < 1.5      # the broad rise, not the spike
