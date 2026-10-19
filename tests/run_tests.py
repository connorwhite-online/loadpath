"""
Plain-python test runner for sdf_kit v2 + networks + fea (no pytest — the
sidecar image stays lean). Run: `python3 cad-runner/tests/run_tests.py`.
Covers the MTR-177 / MTR-179 contracts: TPMS thickness means real mm, the
dual-network verifier catches both isolation and leaks, from_mesh bridges
B-rep meshes into the SDF world, to_mesh refuses oversized grids helpfully,
and fea_probe localizes a cantilever's root as the hotspot.
"""
import sys
import time
import traceback
import os
from pathlib import Path

# app.py fails closed at import without a secret (arbitrary code execution is
# not safe to expose); the assembly-promotion unit tests import it directly.
os.environ.setdefault("CAD_RUNNER_ALLOW_NO_AUTH", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh
import trimesh.sample

from sdf_kit import (
    box, capsule, dual_sheet, from_mesh, gyroid, mask, mesh_cyl, mesh_rod,
    largest_body, mesh_subtract, offset_field, pocket, seal_ramp, shell_field, smax, smin, sphere,
    sq_prism, spline_tube, subtract, superellipsoid, tapered_capsule,
    to_mesh, tpms_dist, translate, union,
)
from exchanger import exchanger_core
from networks import check_networks
from fea import fea_probe

THICKNESS_MEASUREMENT = {}


# ---- helpers -----------------------------------------------------------------
def first_hit_distances(mesh, origins, dirs):
    """Moller-Trumbore first-hit distance per ray (pure numpy — the container
    has no rtree, so trimesh's ray casting is unavailable)."""
    tri = mesh.triangles
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    out = []
    for o, d in zip(origins, dirs):
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = o - v0
        u = np.einsum("ij,ij->i", tvec, p) * inv
        q = np.cross(tvec, e1)
        v = np.einsum("j,ij->i", d, q) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        out.append(t[hit].min() if hit.any() else np.nan)
    return np.asarray(out)


def jacketed_gyroid_field(P):
    """30mm cube: gyroid sheet core (cell 10, t 1.2) sealed inside a 2mm-wall
    jacket — the heat-exchanger-core shape. Two fully enclosed void networks."""
    b = box(P, (15, 15, 15), (15, 15, 15))
    jacket = shell_field(b, 2.0)
    core = mask(gyroid(P, 10.0, 1.2), offset_field(b, -1.0))
    return union(jacket, core)


def pick_ports():
    """Derive port positions from the analytic field sign: sample the interior
    and keep points deep inside each network (normalized field > +2mm /
    < -2mm clearance), so ports never straddle the sheet."""
    ax = np.arange(7.0, 23.5, 1.0)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    d = gyroid(pts, 10.0, 0.0, solid=True)  # t=0 solid form = normalized field
    plus, minus = pts[d > 2.0], pts[d < -2.0]
    assert len(plus) > 1 and len(minus) > 1, "field sampling found no deep voids"

    def pair(cands):
        far = np.argmax(np.linalg.norm(cands - cands[0], axis=1))
        return cands[0], cands[far]

    (a_in, a_out), (b_in, b_out) = pair(plus), pair(minus)
    return [
        {"name": "a_in", "point": a_in.tolist(), "r": 1.2},
        {"name": "a_out", "point": a_out.tolist(), "r": 1.2},
        {"name": "b_in", "point": b_in.tolist(), "r": 1.2},
        {"name": "b_out", "point": b_out.tolist(), "r": 1.2},
    ]


# ---- tests ---------------------------------------------------------------
def test_gyroid_thickness_is_real_mm():
    """Sheet thickness parameter must mean physical mm within 15%, measured on
    the actual meshed geometry by casting rays through the wall."""
    def sheet(Q):
        return mask(gyroid(Q, 10.0, 1.2), box(Q, (15, 15, 15), (6, 6, 6)))

    m = to_mesh(sheet, (8, 8, 8), (22, 22, 22), pitch=0.25)
    assert m.is_watertight, "measurement sheet not watertight"
    pts, fid = trimesh.sample.sample_surface(m, 200, seed=0)
    dirs = -m.face_normals[fid]
    dists = first_hit_distances(m, pts + dirs * 1e-3, dirs) + 1e-3
    measured = float(np.median(dists[np.isfinite(dists)]))
    THICKNESS_MEASUREMENT["target"] = 1.2
    THICKNESS_MEASUREMENT["measured"] = measured
    err = abs(measured - 1.2) / 1.2
    THICKNESS_MEASUREMENT["error"] = err
    assert err < 0.15, f"sheet thickness {measured:.3f}mm vs 1.2mm ({err:.0%} off)"


def test_gyroid_two_isolated_networks():
    """The flagship invariant: a sealed gyroid-sheet core has exactly two void
    networks, each owning its declared port pair, with a real wall between."""
    m = to_mesh(jacketed_gyroid_field, (-2, -2, -2), (32, 32, 32), pitch=0.4)
    assert m.is_watertight, "jacketed gyroid mesh not watertight"

    report = check_networks(m, pick_ports())
    assert report["isolated"] is True, f"expected isolation, got {report}"
    assert report["components"] == 2, f"expected 2 networks, got {report}"
    ports = report["ports"]
    assert ports["a_in"] == ports["a_out"], f"circuit a split: {ports}"
    assert ports["b_in"] == ports["b_out"], f"circuit b split: {ports}"
    assert ports["a_in"] != ports["b_in"], f"circuits share a void: {ports}"
    mw = report["minWallVoxels"]
    assert mw is not None and 1 <= mw <= 8, \
        f"minWallVoxels {mw} implausible for a 1.2mm sheet at pitch {report['pitch']:.3f}"


def test_dual_sheet_defaults_match_sheet_gyroid():
    """With no seals, dual_sheet IS the sheet form of gyroid — same solid."""
    rng = np.random.default_rng(11)
    P = rng.uniform(-15, 15, (5000, 3))
    got = dual_sheet(P, 10.0, 1.2)
    want = gyroid(P, 10.0, 1.2)
    assert np.allclose(got, want, atol=1e-9), "unsealed dual_sheet != sheet gyroid"


def test_dual_sheet_seal_swallows_one_labyrinth():
    """seal_b=1 must solidify fluid B's labyrinth while leaving A's open —
    the manifold-capping primitive. And the midsurface must stay solid under
    ANY seal combination (the isolation guarantee)."""
    rng = np.random.default_rng(12)
    P = rng.uniform(-15, 15, (20000, 3))
    d = tpms_dist(P, 10.0)
    sealed_b = dual_sheet(P, 10.0, 1.2, seal_b=1.0)
    deep_a, deep_b = d > 2.0, d < -2.0
    assert np.all(sealed_b[deep_b] < 0), "seal_b left fluid B voids open"
    assert np.all(sealed_b[deep_a] > 0), "seal_b swallowed fluid A too"
    sealed_a = dual_sheet(P, 10.0, 1.2, seal_a=1.0)
    assert np.all(sealed_a[deep_a] < 0), "seal_a left fluid A voids open"
    assert np.all(sealed_a[deep_b] > 0), "seal_a swallowed fluid B too"
    near_mid = np.abs(d) < 0.4  # inside a 1.2mm wall regardless of seals
    for s in (sealed_a, sealed_b, dual_sheet(P, 10.0, 1.2)):
        assert np.all(s[near_mid] < 0), "midsurface not solid — leak possible"


def test_seal_ramp_is_a_clamped_smoothstep():
    r = seal_ramp(np.array([-3.0, 0.0, 1.0, 2.0, 5.0]), 2.0)
    assert r[0] == 1.0 and r[1] == 1.0, "sealed side must be exactly 1"
    assert abs(r[2] - 0.5) < 1e-9, "midpoint must be 0.5"
    assert r[3] == 0.0 and r[4] == 0.0, "open side must be exactly 0"
    assert np.all(np.diff(r) <= 0), "ramp must be monotone"


_CROSS_CORE = {}


def _cross_core():
    """Memoized 20mm cross-flow exchanger build shared by several tests."""
    if not _CROSS_CORE:
        field, ports, plugs, bounds = exchanger_core(
            (20, 20, 20), cell=8.0, wall=1.2)
        mesh = to_mesh(field, *bounds, pitch=0.4)
        _CROSS_CORE.update(mesh=mesh, ports=ports, plugs=plugs)
    return _CROSS_CORE


def _assert_isolated(report):
    assert report["isolated"] is True, f"expected isolation, got {report}"
    assert report["components"] == 2, f"expected 2 circuits, got {report}"
    ports = report["ports"]
    assert ports["a_in"] == ports["a_out"], f"circuit a split: {ports}"
    assert ports["b_in"] == ports["b_out"], f"circuit b split: {ports}"
    assert ports["a_in"] != ports["b_in"], f"circuits share a void: {ports}"
    mw = report["minWallVoxels"]
    assert mw is not None and mw >= 1, f"no measurable wall: {report}"


def test_exchanger_core_cross_isolated():
    """The deliverable invariant: a manifolded cross-flow exchanger meshes
    watertight and its two circuits verify isolated (bores plugged)."""
    built = _cross_core()
    assert built["mesh"].is_watertight, "cross exchanger not watertight"
    report = check_networks(built["mesh"], built["ports"], plugs=built["plugs"])
    _assert_isolated(report)


def test_exchanger_core_open_ports_need_plugs():
    """Without plugs the open ports connect both circuits through outside
    air — the check must (correctly) refuse to call that isolated. This
    documents WHY the plugs exist; a false 'isolated' here would mean the
    verifier can no longer see leaks at all."""
    built = _cross_core()
    report = check_networks(built["mesh"], built["ports"])
    assert report["isolated"] is False, (
        f"open-port part reported isolated without plugs: {report}")


def test_exchanger_core_counter_isolated():
    """Counterflow variant: split plenums on the X faces, solid divider."""
    field, ports, plugs, bounds = exchanger_core(
        (20, 20, 24), cell=8.0, wall=1.2, flow="counter")
    mesh = to_mesh(field, *bounds, pitch=0.4)
    assert mesh.is_watertight, "counter exchanger not watertight"
    report = check_networks(mesh, ports, plugs=plugs)
    _assert_isolated(report)


def test_exchanger_core_refusals():
    """Undersized cores and oversized ports must refuse with messages the
    repair loop can act on."""
    try:
        exchanger_core((8, 8, 8), cell=8.0, wall=1.0)
    except ValueError as e:
        assert "1.5 cells" in str(e), f"unhelpful refusal: {e}"
    else:
        raise AssertionError("sub-1.5-cell core was accepted")
    try:
        exchanger_core((20, 20, 20), cell=8.0, wall=1.0, port_r=50.0)
    except ValueError as e:
        assert "port_r" in str(e), f"unhelpful refusal: {e}"
    else:
        raise AssertionError("oversized port_r was accepted")
    # Per-port dict: the oversized entry is named in the refusal.
    try:
        exchanger_core((20, 20, 20), cell=8.0, wall=1.0,
                       port_r={"a_in": 50.0})
    except ValueError as e:
        assert 'port_r["a_in"]' in str(e), f"unhelpful refusal: {e}"
    else:
        raise AssertionError("oversized per-port radius was accepted")


def test_exchanger_core_asymmetric_ports_isolated():
    """Per-port radii (the asymmetric inlet/outlet case that pushed a build
    off the paved path): distinct bores land in the plugs, and the circuits
    still verify isolated."""
    field, ports, plugs, bounds = exchanger_core(
        (24, 24, 20), cell=8.0, wall=1.2,
        port_r={"a_in": 4.0, "a_out": 3.0, "b": 2.0})
    by_name = {}
    for port, plug in zip(ports, plugs):
        by_name[port["name"]] = plug["r"]
    assert by_name["a_in"] == 4.0 and by_name["a_out"] == 3.0, by_name
    assert by_name["b_in"] == 2.0 and by_name["b_out"] == 2.0, by_name
    mesh = to_mesh(field, *bounds, pitch=0.4)
    assert mesh.is_watertight, "asymmetric-port exchanger not watertight"
    report = check_networks(mesh, ports, plugs=plugs, min_feature=1.2)
    _assert_isolated(report)


def test_exchanger_core_hose_hoods_isolated():
    """Integrated hood-loft hose manifolds (the feature that replaces
    hand-rolled manifolds, whose sign bugs bored leak paths through cores):
    each hooded port opens its FULL plenum window and lofts rect-to-round
    (cubic-eased) down to the hose bore. The mesh stays one watertight body,
    the circuits verify isolated (box plugs capping the windows), and the
    bounds cover the barb runs."""
    field, ports, plugs, bounds = exchanger_core(
        (24, 24, 20), cell=8.0, wall=1.2,
        hose={"a_in": 12.0, "a_out": 10.0, "b": 6.0})
    # Hooded ports have no bore: one BOX plug per port, no capsules.
    assert len(plugs) == 4, f"expected one box plug per port: {len(plugs)}"
    assert all("c" in p and "h" in p for p in plugs), f"not boxes: {plugs}"
    lo, hi = bounds
    # Hoods protrude along X (fluid A) and Y (fluid B) — well past the core.
    assert hi[0] > 35 and hi[1] > 35, f"bounds ignore the hoods: {bounds}"
    mesh = to_mesh(field, *bounds, pitch=0.4)
    assert mesh.is_watertight, "hooded exchanger not watertight"
    assert len(mesh.split(only_watertight=False)) == 1, "hoods disconnected"
    report = check_networks(mesh, ports, plugs=plugs, min_feature=1.2)
    _assert_isolated(report)


def test_exchanger_core_counter_hoods_isolated():
    """Counterflow with hoods: two hooded ports share each X face (split
    windows) — the hoods must coexist without merging channels."""
    field, ports, plugs, bounds = exchanger_core(
        (20, 20, 28), cell=8.0, wall=1.2, flow="counter",
        hose={"a": 5.0, "b": 5.0})
    mesh = to_mesh(field, *bounds, pitch=0.4)
    assert mesh.is_watertight, "counter hooded exchanger not watertight"
    report = check_networks(mesh, ports, plugs=plugs, min_feature=1.2)
    _assert_isolated(report)


def test_exchanger_core_face_spec_orientation():
    """Orientation contract (a_face/b_face): the user's own exchanger spec
    ("hot ports on the 150x60 faces of a 150x60x80 core") has now shipped
    wrong twice because the model ordered `size` by habit. With the face
    dims declared, a wrong ordering must refuse AND name the right one; the
    right ordering must pass validation."""
    try:
        exchanger_core((150, 60, 80), cell=10.0, wall=1.2,
                       a_face=(150, 60), b_face=(80, 60))
    except ValueError as e:
        msg = str(e)
        assert "reorder size to 80x150x60" in msg, f"no corrected order: {e}"
    else:
        raise AssertionError("wrong size ordering was accepted")
    # The corrected ordering builds without complaint (validation only —
    # no meshing needed here; the hood tests cover geometry).
    field, ports, plugs, bounds = exchanger_core(
        (80, 150, 60), cell=10.0, wall=1.2,
        a_face=(150, 60), b_face=(80, 60))
    assert len(ports) == 4


def test_exchanger_core_hose_overlap_refused():
    """Counterflow puts two ports on each X face; hoses fat enough to merge
    there would join the circuits OUTSIDE the core — must refuse, naming
    both ports."""
    try:
        exchanger_core((20, 20, 24), cell=8.0, wall=1.2, flow="counter",
                       hose={"a": 18.0, "b": 18.0})
    except ValueError as e:
        msg = str(e)
        assert "share a face" in msg and "a_in" in msg, f"unhelpful: {e}"
    else:
        raise AssertionError("overlapping barbs were accepted")


def test_check_networks_min_feature_refines_pitch():
    """On a part large enough that the default extent/64 probe would be
    coarser than the walls, declaring min_feature must pull the pitch down
    to <= min_feature/2 — the fix for sound 1.2mm walls reading as phantom
    leaks under a 2.66mm probe on a 150mm-class exchanger."""
    def hollow(P):
        return shell_field(box(P, (0, 0, 0), (50, 50, 50)), 6.0)

    m = to_mesh(hollow, (-53, -53, -53), (53, 53, 53), pitch=1.0)
    ports = [
        {"name": "a_in", "point": [-20, 0, 0], "r": 4.0},
        {"name": "b_in", "point": [20, 0, 0], "r": 4.0},
    ]
    # Default probe on a ~106mm part: 106/64 ≈ 1.66mm.
    coarse = check_networks(m, ports)
    assert coarse["pitch"] > 1.5, f"expected coarse default, got {coarse}"
    fine = check_networks(m, ports, min_feature=3.0)
    assert fine["pitch"] <= 1.5 + 1e-9, f"min_feature ignored: {fine}"
    assert "inconclusive" not in fine, f"should be conclusive: {fine}"


def test_check_networks_inconclusive_over_voxel_budget():
    """When the voxel budget cannot afford a min_feature-resolving pitch,
    the check must decline the verdict (inconclusive) rather than render a
    phantom-leak report — 'not verified' is honest, 'leaked' would be noise."""
    import networks as networks_mod

    built = _cross_core()
    saved = networks_mod._VOXEL_BUDGET
    networks_mod._VOXEL_BUDGET = 5_000  # force the coarsening loop
    try:
        report = check_networks(
            built["mesh"], built["ports"], plugs=built["plugs"],
            min_feature=1.2)
    finally:
        networks_mod._VOXEL_BUDGET = saved
    assert report.get("inconclusive") is True, f"expected inconclusive: {report}"
    assert "isolated" not in report, f"inconclusive must carry no verdict: {report}"
    assert "UNVERIFIED" in report.get("reason", ""), f"reason unclear: {report}"


def test_from_mesh_matches_analytic_box():
    """The B-rep bridge: from_mesh of a box must reproduce the analytic box
    SDF within ~pitch at sampled points."""
    pitch = 1.0
    bmesh = trimesh.creation.box(extents=(20, 16, 12))
    f = from_mesh(bmesh, pitch=pitch)
    rng = np.random.default_rng(7)
    P = rng.uniform(-1, 1, size=(500, 3)) * np.array([12.0, 10.0, 8.0])
    got = f(P)
    want = box(P, (0, 0, 0), (10, 8, 6))
    worst = float(np.max(np.abs(got - want)))
    assert worst <= 1.2 * pitch, f"from_mesh error {worst:.2f}mm > ~pitch ({pitch}mm)"
    # signs must never disagree beyond the surface band
    off = np.abs(want) > pitch
    assert np.all(np.sign(got[off]) == np.sign(want[off])), "inside/outside flipped"


def test_from_mesh_blends_watertight():
    """A from_mesh box smin-blended with a capsule must mesh watertight — the
    whole point of the bridge is composing B-rep bodies into implicit ones."""
    f = from_mesh(trimesh.creation.box(extents=(20, 16, 12)), pitch=1.0)

    def blended(P):
        return smin(f(P), capsule(P, (8, 0, 0), (20, 0, 0), 4.0), 3.0)

    m = to_mesh(blended, (-14, -12, -10), (26, 12, 10), pitch=0.5)
    assert m.is_watertight, "blended from_mesh body not watertight"
    assert m.volume > 20 * 16 * 12 * 0.8, "blend lost the box body"


def test_to_mesh_budget_names_a_pitch():
    """Oversized grids must be refused before allocation, with an error naming
    a pitch that fits — a failure the repair loop can act on."""
    try:
        to_mesh(lambda P: sphere(P, (0, 0, 0), 10), (0, 0, 0), (400, 400, 400), pitch=0.2)
    except ValueError as e:
        msg = str(e)
        assert "pitch >=" in msg, f"error does not name a pitch: {msg}"
        import sdf_kit as sdf_kit_mod

        suggested = float(msg.rsplit("pitch >=", 1)[1].strip())
        n = [int(np.ceil(400 / suggested)) + 1 for _ in range(3)]
        assert n[0] * n[1] * n[2] <= sdf_kit_mod._CELL_BUDGET, \
            f"suggested pitch {suggested} still busts the budget"
        return
    raise AssertionError("to_mesh accepted an 8e9-cell grid")


def test_check_networks_negative_single_cavity():
    """A single open cavity with two differently-prefixed ports is NOT
    isolated — both circuits share one void."""
    def hollow(P):
        return shell_field(box(P, (0, 0, 0), (10, 10, 10)), 2.0)

    m = to_mesh(hollow, (-13, -13, -13), (13, 13, 13), pitch=0.4)
    assert m.is_watertight
    report = check_networks(m, [
        {"name": "a_in", "point": [-5, 0, 0], "r": 2.0},
        {"name": "b_in", "point": [5, 0, 0], "r": 2.0},
    ])
    assert report["components"] == 1, f"expected one shared void, got {report}"
    assert report["ports"]["a_in"] == report["ports"]["b_in"], f"bad mapping: {report}"
    assert report["isolated"] is False, f"leak not flagged: {report}"


def _beam(length):
    move = trimesh.transformations.translation_matrix([length / 2, 4, 4])
    return trimesh.creation.box(extents=(length, 8, 8), transform=move)


def test_fea_cantilever():
    """Cantilever sanity: fixed root, tip load down — hotspot at the root
    third, nonzero deflection, and a longer beam deflects more."""
    supports = [{"point": [0, 4, 4], "r": 6.0}]
    t0 = time.time()
    r40 = fea_probe(_beam(40), [{"point": [40, 4, 4], "r": 3.0, "force": [0, 0, -1.0]}],
                    supports, resolution=32)
    elapsed = time.time() - t0
    assert r40["ok"], f"fea_probe failed: {r40['error']}"
    assert r40["maxDisplacement"] > 0, "beam did not deflect"
    assert r40["hotspots"], "no hotspots returned"
    hot_x = r40["hotspots"][0]["point"][0]
    assert hot_x < 15.0, f"top hotspot at x={hot_x:.1f}, expected in the root third"
    assert elapsed < 10.0, f"resolution-32 probe took {elapsed:.1f}s (budget ~10s)"

    r60 = fea_probe(_beam(60), [{"point": [60, 4, 4], "r": 3.0, "force": [0, 0, -1.0]}],
                    supports, resolution=32)
    assert r60["ok"], f"60mm probe failed: {r60['error']}"
    assert r60["maxDisplacement"] > r40["maxDisplacement"], (
        f"longer beam must deflect more: 60mm={r60['maxDisplacement']:.4g} "
        f"vs 40mm={r40['maxDisplacement']:.4g}")
    print(f"    [fea] 40mm: disp={r40['maxDisplacement']:.4g} "
          f"compliance={r40['compliance']:.4g} hotspot x={hot_x:.1f} ({elapsed:.1f}s); "
          f"60mm disp={r60['maxDisplacement']:.4g}")


def test_fea_refusals():
    """Guards: missing supports and disconnected loads refuse cleanly."""
    beam = _beam(40)
    r = fea_probe(beam, [{"point": [40, 4, 4], "r": 3.0, "force": [0, 0, -1.0]}],
                  [{"point": [500, 500, 500], "r": 1.0}], resolution=32)
    assert not r["ok"] and r["error"], "missing supports must refuse"
    r = fea_probe(beam, [{"point": [500, 500, 500], "r": 1.0, "force": [0, 0, -1.0]}],
                  [{"point": [0, 4, 4], "r": 6.0}], resolution=32)
    assert not r["ok"] and r["error"], "disconnected load must refuse"


def test_transforms_move_primitives():
    """translate/rotate_z warp P, so the primitive lands where advertised."""
    P = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    d = sphere(translate(P, (10, 0, 0)), (0, 0, 0), 2.0)
    assert d[0] < 0 < d[1], "translate did not move the sphere to x=+10"
    from sdf_kit import rotate_z
    Q = np.array([[0.0, 10.0, 0.0]])
    d = sphere(rotate_z(Q, 90.0), (10, 0, 0), 2.0)
    assert d[0] < 0, "rotate_z(90) did not carry (10,0,0) onto (0,10,0)"


def test_split_shell_printed_fit():
    from sdf_kit import box, smin, offset_field, subtract, split_shell, to_mesh
    import numpy as np

    def cavity(P):
        return smin(box(P, (0, 0, 8.0), (31, 21, 6)),
                    box(P, (0, 0, 20.5), (25, 15, 6.5)), 14.0)

    def body(P):
        return np.maximum(subtract(offset_field(cavity(P), 2.4), cavity(P)),
                          -P[:, 2])

    fit_gap = 0.25
    bottom, top = split_shell(body, cavity, 16.0, fit_gap=fit_gap)
    mb = to_mesh(bottom, (-40, -30, -3), (40, 30, 34), pitch=0.55)
    mt = to_mesh(top, (-40, -30, -3), (40, 30, 34), pitch=0.55)
    assert mb.is_watertight and mt.is_watertight
    # printed-fit clearance on the lip faces + no interpenetration anywhere
    rng = np.random.default_rng(3)
    pts = rng.uniform([-38, -28, 16.4], [38, 28, 18.6], (40000, 3))
    c = cavity(pts)
    lip_pts = pts[np.abs(c - 1.1) < 0.08][:800]
    gap = top(lip_pts)
    assert np.median(gap) > fit_gap * 0.8, f"median lip clearance {np.median(gap):.2f}"
    both = (bottom(pts) < 0) & (top(pts) < 0)
    assert both.sum() == 0, "halves interpenetrate"
    # union coverage: every solidly-interior body point belongs to one half,
    # EXCEPT inside the designed lip-clearance band (the fit_gap ring between
    # lip and recess is in neither half — that gap IS the printed fit).
    z_split, lip_h, lip_t = 16.0, 3.0, 1.1
    vol = rng.uniform([-38, -28, -1], [38, 28, 33], (60000, 3))
    interior = body(vol) < -0.2
    p_in = vol[interior]
    covered = (bottom(p_in) < 0) | (top(p_in) < 0)
    missed = p_in[~covered]
    assert len(missed) < 0.05 * len(p_in), f"{len(missed)}/{len(p_in)} interior points uncovered"
    if len(missed):
        cz = missed[:, 2]
        cc = cavity(missed)
        in_band = (
            (cz > z_split - 1.2) & (cz < z_split + lip_h + fit_gap + 0.2)
            & (cc > -0.1) & (cc < lip_t + fit_gap + 0.1)
        )
        assert in_band.all(), "uncovered material outside the lip-clearance band"


def test_promote_clean_base_lid_compound():
    """MTR-213: a single `result` mesh that splits into two comparable,
    individually-watertight bodies promotes to a base/lid assembly ordered
    low->high z — instead of hard-failing the fragment gate (which collapsed
    enclosures to one shell)."""
    import app
    base = trimesh.creation.box(extents=(40, 30, 16)).apply_translation((0, 0, 8))
    lid = trimesh.creation.box(extents=(40, 30, 8)).apply_translation((0, 0, 22))
    compound = trimesh.util.concatenate([base, lid])
    assert len(compound.split(only_watertight=False)) == 2, "fixture must be 2 bodies"
    promoted = app._promote_disconnected_bodies(compound)
    assert promoted is not None, "clean base+lid compound must promote"
    assert [n for n, _ in promoted] == ["base", "lid"], [n for n, _ in promoted]
    zb, zl = float(promoted[0][1].centroid[2]), float(promoted[1][1].centroid[2])
    assert zb < zl, f"base must sit below lid ({zb:.1f} vs {zl:.1f})"


def test_promote_thin_lid_compound():
    """A thin lid (~6% of base volume) must still promote — the old 15% gate
    left stacked compounds in the preview with no parts[] split."""
    import app
    base = trimesh.creation.box(extents=(40, 30, 20)).apply_translation((0, 0, 10))
    lid = trimesh.creation.box(extents=(40, 30, 1.2)).apply_translation((0, 0, 21))
    compound = trimesh.util.concatenate([base, lid])
    promoted = app._promote_disconnected_bodies(compound)
    assert promoted is not None, "thin lid must promote as a real part"
    assert [n for n, _ in promoted] == ["base", "lid"]


def test_promote_rejects_genuine_debris():
    """The conservative gate: a real base+lid trailing a loose chip (an
    un-unioned nub) is NOT promoted — the fragment error must stand so the model
    fixes it. A lone solid + chip is likewise no assembly. But a sub-mm^3
    tessellation sliver alongside the parts IS tolerated."""
    import app
    base = trimesh.creation.box(extents=(40, 30, 16)).apply_translation((0, 0, 8))
    lid = trimesh.creation.box(extents=(40, 30, 8)).apply_translation((0, 0, 22))
    chip = trimesh.creation.box(extents=(5, 5, 5)).apply_translation((80, 80, 80))
    assert app._promote_disconnected_bodies(
        trimesh.util.concatenate([base, lid, chip])
    ) is None, "a loose chip beside the parts must block promotion"
    solid = trimesh.creation.box(extents=(40, 30, 20))
    assert app._promote_disconnected_bodies(
        trimesh.util.concatenate([solid, chip])
    ) is None, "a solid + chip is not a 2-part assembly"
    sliver = trimesh.creation.box(extents=(0.5, 0.5, 0.5)).apply_translation((80, 80, 80))
    ok = app._promote_disconnected_bodies(trimesh.util.concatenate([base, lid, sliver]))
    assert ok is not None and [n for n, _ in ok] == ["base", "lid"], \
        "a sub-mm^3 sliver must be tolerated alongside the real parts"


def test_combinators_accept_field_functions():
    """Generated code passes field FUNCTIONS to offset_field & co. as often as
    arrays (the prompt calls the field `f`). Before _lift, that failed as
    "unsupported operand type(s) for -: 'function' and 'float'" on every
    attempt of a local SDF knob build. A function in gives a function out that
    matches the array path exactly."""
    P = np.random.default_rng(0).uniform(-20, 20, (500, 3))
    f = lambda Q: sphere(Q, (0, 0, 0), 10)
    g = lambda Q: box(Q, (5, 0, 0), (6, 6, 6))
    arr = offset_field(f(P), 2)
    fn = offset_field(f, 2)
    assert callable(fn) and np.allclose(fn(P), arr)
    assert np.allclose(smin(f, g, 3)(P), smin(f(P), g(P), 3))
    assert np.allclose(smax(f, g(P), 1)(P), smax(f(P), g(P), 1)), "mixed"
    assert np.allclose(shell_field(f, 2)(P), shell_field(f(P), 2))
    assert np.allclose(mask(f, g)(P), mask(f(P), g(P)))
    # arrays in -> array out: existing code is untouched
    assert isinstance(offset_field(f(P), 2), np.ndarray)
    m = to_mesh(offset_field(f, 1.0), (-13,) * 3, (13,) * 3, 0.8)
    assert m.is_watertight


def test_mesh_rod_cuts_an_exact_radial_hole():
    """The only exact cut used to be the vertical mesh_cyl, so a knob's radial
    set-screw hole came out as a square slot. mesh_rod runs on any axis."""
    rod = mesh_rod((0, 0, 5), (20, 0, 5), 1.7)
    assert rod.is_watertight
    assert abs(rod.volume - np.pi * 1.7 ** 2 * 20) / rod.volume < 0.01
    body = mesh_cyl(0, 0, 15, 0, 18)
    cut = mesh_subtract(body, rod)
    assert cut.is_watertight
    removed = body.volume - cut.volume
    assert abs(removed - np.pi * 1.7 ** 2 * 15) / removed < 0.02, removed
    try:
        mesh_rod((1, 1, 1), (1, 1, 1), 2)
        raise AssertionError("zero-length rod accepted")
    except ValueError:
        pass


def _wall_along(inner, outer, direction, z=0.0):
    """Wall thickness where the ray from the origin along `direction` crosses
    both surfaces. Along a symmetry axis the ray is the surface normal, so
    this is the true wall."""
    def hit(f):
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2
            v = f(np.array([[mid * direction[0], mid * direction[1], z]]))[0]
            lo, hi = (mid, hi) if v < 0 else (lo, mid)
        return lo
    return hit(outer) - hit(inner)


def test_superellipse_offsets_give_uniform_walls():
    """sq_prism used to scale its field by min(a, b), which is only a distance
    along the short axis: a 2mm offset of a 27x14 prism came out 3.9mm at the
    ends. Now gradient-normalized, both axes get the wall that was asked."""
    cav = lambda P: sq_prism(P, 27.25, 14.0, 5.5, -10, 10)
    out = lambda P: offset_field(cav(P), 2.0)
    for d in ((1, 0), (0, 1)):
        w = _wall_along(cav, out, d)
        assert abs(w - 2.0) < 0.05, (d, w)
    peb = lambda P: superellipsoid(P, (0, 0, 0), (36, 20, 9.5), 3.0, 3.5)
    shell = lambda P: offset_field(peb(P), 2.2)
    for d in ((1, 0), (0, 1)):
        w = _wall_along(peb, shell, d)
        assert abs(w - 2.2) < 0.05, (d, w)
    m = to_mesh(peb, (-38, -22, -11), (38, 22, 11), 0.8)
    assert m.is_watertight


def test_tapered_capsule_is_an_exact_distance():
    """The organic strut: equal radii must reduce to capsule, and the field
    must agree with the meshed surface (it is an exact SDF, not a proxy)."""
    a, b = (0, 0, 0), (40, 0, 0)
    P = np.random.default_rng(3).uniform([-10, -10, -10], [50, 10, 10], (400, 3))
    assert np.allclose(tapered_capsule(P, a, b, 4, 4), capsule(P, a, b, 4))
    m = to_mesh(lambda Q: tapered_capsule(Q, a, b, 6, 3), (-8, -8, -8), (45, 8, 8), 0.3)
    assert m.is_watertight
    _, dist, _ = trimesh.proximity.closest_point(m, P)
    err = np.max(np.abs(np.abs(tapered_capsule(P, a, b, 6, 3)) - dist))
    assert err < 0.05, err


def test_spline_tube_sweeps_smoothly_through_its_points():
    """A spline sweep, not a polyline: one watertight body passing through
    every control point at that point's radius, with no elbow at interior
    control points (the surface normal turns gradually along the sweep)."""
    pts = [(0, 0, 0), (10, 20, 5), (30, 30, 5), (45, 35, 15)]
    radii = [5, 4, 3.5, 3]
    f = lambda P: spline_tube(P, pts, radii)
    m = to_mesh(f, (-8, -8, -8), (52, 44, 24), 0.4)
    assert m.is_watertight and len(m.split(only_watertight=False)) == 1
    for p, r in zip(pts, radii):
        # a point one radius away, perpendicular-ish (straight up), is on the surface
        v = f(np.array([[p[0], p[1], p[2] + r]]))[0]
        assert abs(v) < 0.35, (p, v)
    try:
        spline_tube(np.zeros((1, 3)), pts, radii[:2])
        raise AssertionError("mismatched radii accepted")
    except ValueError:
        pass


def test_pocket_is_open_with_a_rounded_floor():
    """pocket() exists so a container can't come out closed: cut from a
    block it opens through the top, keeps its floor, rounds the floor edge,
    and three of them leave two walls between them."""
    blk = lambda P: box(P, (0, 0, 20), (60, 25, 20))
    pk = lambda P, x: pocket(P, (x, 0), (16, 20), 4.0, r=5)
    cut = lambda P: subtract(subtract(subtract(blk(P), pk(P, -40)), pk(P, 0)), pk(P, 40))
    m = to_mesh(cut, (-63, -28, -2), (63, 28, 43), 0.6)
    assert m.is_watertight and len(m.split(only_watertight=False)) == 1
    fill = m.volume / (120 * 50 * 40)
    assert fill < 0.55, fill
    q = lambda *p: cut(np.array([p], float))[0]
    assert q(0, 0, 39) > 0      # open at the rim
    assert q(0, 0, 2) < 0       # floor kept
    assert q(-20, 0, 30) < 0    # divider between pockets
    # rounded floor: the corner of the pocket floor stays solid
    assert q(14.5, 18.5, 5) < 0 and pk(np.array([[0, 0, 5.0]]), 0)[0] < 0


def test_star_import_exports_every_primitive():
    """Generated programs only see sdf_kit through `from sdf_kit import *`,
    so a public function missing from __all__ passes every direct-import test
    here and is a NameError in production. pocket() shipped that way once."""
    import inspect
    import sdf_kit
    public = {
        n for n, f in inspect.getmembers(sdf_kit, inspect.isfunction)
        if not n.startswith("_") and f.__module__ == "sdf_kit"
    }
    missing = public - set(sdf_kit.__all__)
    assert not missing, sorted(missing)


def test_opening_check_tells_open_from_closed():
    """A container reads as one only if it opens from above. An open tray,
    a sealed hollow (mostly empty, so a fill ratio passes it) and a solid
    block must land on opposite sides of the blockout threshold (0.2)."""
    from opening import check_opening
    blk = lambda P: box(P, (0, 0, 20), (60, 25, 20))
    pk = lambda P, x: pocket(P, (x, 0), (16, 20), 4.0, r=5)
    tray = lambda P: subtract(subtract(subtract(blk(P), pk(P, -40)), pk(P, 0)), pk(P, 40))
    inner = lambda P: box(P, (0, 0, 20), (56, 21, 16))
    hollow = lambda P: subtract(blk(P), inner(P))
    grid = ((-63, -28, -2), (63, 28, 43), 0.8)
    o_tray = check_opening(to_mesh(tray, *grid))["openFraction"]
    o_hollow = check_opening(to_mesh(hollow, *grid))["openFraction"]
    o_solid = check_opening(to_mesh(blk, *grid))["openFraction"]
    assert o_tray > 0.4, o_tray
    assert o_hollow < 0.05 and o_solid < 0.05, (o_hollow, o_solid)


def test_largest_body_drops_specks_not_parts():
    """Concept blockouts drop pinhead islands; a real second body stays, so
    the one-body gate still sees it."""
    body = lambda P: box(P, (0, 0, 0), (20, 20, 10))
    speck = lambda P: sphere(P, (40, 0, 0), 1.5)
    part = lambda P: box(P, (45, 0, 0), (8, 8, 8))
    grid = ((-22, -22, -12), (55, 22, 12), 0.5)
    m = to_mesh(lambda P: union(body(P), speck(P)), *grid)
    assert len(m.split(only_watertight=False)) == 2
    kept = largest_body(m)
    assert len(kept.split(only_watertight=False)) == 1 and kept.volume > 15_000
    m2 = to_mesh(lambda P: union(body(P), part(P)), *grid)
    assert largest_body(m2) is m2


TESTS = [
    test_largest_body_drops_specks_not_parts,
    test_opening_check_tells_open_from_closed,
    test_star_import_exports_every_primitive,
    test_pocket_is_open_with_a_rounded_floor,
    test_spline_tube_sweeps_smoothly_through_its_points,
    test_tapered_capsule_is_an_exact_distance,
    test_superellipse_offsets_give_uniform_walls,
    test_combinators_accept_field_functions,
    test_mesh_rod_cuts_an_exact_radial_hole,
    test_gyroid_thickness_is_real_mm,
    test_gyroid_two_isolated_networks,
    test_dual_sheet_defaults_match_sheet_gyroid,
    test_dual_sheet_seal_swallows_one_labyrinth,
    test_seal_ramp_is_a_clamped_smoothstep,
    test_exchanger_core_cross_isolated,
    test_exchanger_core_open_ports_need_plugs,
    test_exchanger_core_counter_isolated,
    test_exchanger_core_refusals,
    test_exchanger_core_asymmetric_ports_isolated,
    test_exchanger_core_hose_hoods_isolated,
    test_exchanger_core_counter_hoods_isolated,
    test_exchanger_core_face_spec_orientation,
    test_exchanger_core_hose_overlap_refused,
    test_check_networks_min_feature_refines_pitch,
    test_check_networks_inconclusive_over_voxel_budget,
    test_from_mesh_matches_analytic_box,
    test_from_mesh_blends_watertight,
    test_to_mesh_budget_names_a_pitch,
    test_check_networks_negative_single_cavity,
    test_fea_cantilever,
    test_fea_refusals,
    test_transforms_move_primitives,
    test_split_shell_printed_fit,
    test_promote_clean_base_lid_compound,
    test_promote_thin_lid_compound,
    test_promote_rejects_genuine_debris,
]


def main():
    failures = 0
    for t in TESTS:
        start = time.time()
        try:
            t()
            print(f"PASS  {t.__name__}  ({time.time() - start:.1f}s)")
        except Exception:
            failures += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    if THICKNESS_MEASUREMENT:
        print(f"\ngyroid sheet thickness: target {THICKNESS_MEASUREMENT['target']}mm, "
              f"measured {THICKNESS_MEASUREMENT['measured']:.3f}mm "
              f"({THICKNESS_MEASUREMENT['error']:.1%} error)")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
