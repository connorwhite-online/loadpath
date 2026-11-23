"""Freeform surfacing (surfacing.py), its quality checks (surfacequality.py)
and sketch matching (sketchmatch.py)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import sketchmatch as SM            # noqa: E402
from loadpath import surfacequality as Q          # noqa: E402
from loadpath import surfacing as S               # noqa: E402


def _tube(top_bulge=0.6):
    """An open elliptical tube along z (periodic in u), bulging a little."""
    u = np.linspace(0, 1, 60, endpoint=False)
    v = np.linspace(0, 1, 30)
    th = 2 * np.pi * u
    R = 20 + 5 * np.sin(np.pi * v * top_bulge)
    X = np.stack([np.c_[R[j] * np.cos(th), 0.7 * R[j] * np.sin(th), np.full(60, 100 * v[j])] for j in range(30)], 1)
    return S.fit_surface(X, (20, 10), uperiodic=True, fair=1e-6)


def _ends(face):
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    def mid(e):
        c = BRepAdaptor_Curve(e)
        return c.Value(0.5 * (c.FirstParameter() + c.LastParameter())).Z()
    be = S.boundary_edges(face)
    return [e for e in be if mid(e) > 50], [e for e in be if mid(e) < 50]


def test_basis_matches_opencascade():
    """The fast clamped basis is the one OpenCascade evaluates, values and slopes."""
    t = np.linspace(0, 1, 41)
    for n in (4, 9, 15):
        assert np.abs(S.basis(n, 3, False, t) - S._basis_occ(n, 3, False, t)).max() < 1e-12
        assert np.abs(S.basis(n, 3, False, t, 1) - S._basis_occ(n, 3, False, t, 1)).max() < 1e-9
    P = S.basis(8, 3, True, t)
    assert np.allclose(P.sum(1), 1.0)
    for per in (False, True):
        tt = t[:-1] if per else t
        for deg in (3, 5):
            for d in (0, 1, 2, 3):
                ref = S._basis_occ(11, deg, per, tt, d)
                assert np.abs(S.basis(11, deg, per, tt, d) - ref).max() < 1e-9 * max(1.0, np.abs(ref).max()), (per, deg, d)
    # the energy's exact integrals: x(u, v) = u^2 has x_uu = 2 over the unit
    # square (bending energy 4) and no third derivative (jerk 0)
    nu = 9
    w = np.linspace(0, 1, 30)
    X = np.stack([w[:, None] ** 2 + 0 * w[None, :], 0 * w[:, None] + w[None, :], np.zeros((30, 30))], -1)
    z = S.fit_surface(X, (nu, nu), fair=0.0).poles[..., 0].ravel()
    assert abs(z @ (S._energy(nu, nu, 3, 3, False, False, "bending") @ z) - 4.0) < 1e-6
    assert abs(z @ (S._energy(nu, nu, 3, 3, False, False, "jerk") @ z)) < 1e-6
    print("basis ok")


def test_curves():
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    pts = np.c_[30 * np.cos(th), 20 * np.sin(th), 0 * th]
    c = S.fit_curve(pts, 40, closed=True, fair=0.0)
    assert c.fit_error < 0.002 and c.periodic
    # the curvature of an ellipse peaks at a / b^2 at the ends of its long axis
    assert abs(c.curvature(400).max() - 30 / 400) < 0.004
    # pencil wobble is smoothed, not followed
    rng = np.random.default_rng(0)
    noisy = pts + rng.normal(0, 0.3, pts.shape) * [1, 1, 0]
    s = S.fit_curve(noisy, 24, closed=True).sample(400)
    ang = np.arctan2(s[:, 1] / 20, s[:, 0] / 30)
    assert np.linalg.norm(s[:, :2] - np.c_[30 * np.cos(ang), 20 * np.sin(ang)], axis=1).max() < 0.25
    sp = S.spline(np.c_[np.arange(6.0), np.arange(6.0) ** 2, np.zeros(6)])
    assert np.allclose(sp(np.array([0.0, 1.0])), [[0, 0, 0], [5, 25, 0]], atol=1e-9)
    print("curves ok")


def test_fit_and_fair_patch():
    s = _tube()
    assert s.fit["maxDeviation"] < 0.1
    # a fair patch: flat plateau at 12 inside r 10, the body (0) beyond r 28,
    # a soft sweep between with no overshoot past either
    xs = np.linspace(-40, 40, 41)
    Z, X = np.meshgrid(xs, xs)
    r = np.hypot(X, Z)
    on, far = r < 10, r > 28
    p = S.fair_patch((-40, 40), (-40, 40), np.c_[X[on], np.full(on.sum(), 12.0), Z[on]], poles=(24, 24), fair=1.0,
                     soft=np.c_[X[far], np.zeros(far.sum()), Z[far], np.ones(far.sum())])
    h = p.height(np.linspace(0, 32, 33), np.zeros(33))
    assert abs(h[0] - 12) < 0.05 and abs(h[5] - 12) < 0.05 and abs(h[-1]) < 0.3
    assert np.all(np.diff(h[10:30]) < 0.05), "the sweep falls steadily from the plateau to the body"
    # its height function agrees with the OpenCascade surface it builds
    P = p(np.array([0.3, 0.7]), np.array([0.5]))[:, 0]
    assert np.allclose(p.height(P[:, 0], P[:, 2]), P[:, 1], atol=1e-6)
    f = S.height_sdf(p)
    assert f(np.array([[0.0, 10.0, 0.0]]))[0] < 0 < f(np.array([[0.0, 14.0, 0.0]]))[0]
    print("fit and fair patch ok")


def test_loft_with_rails():
    """Sections warped to meet a rail: the surface passes through the rail."""
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    secs = [np.c_[20 * np.cos(th), 15 * np.sin(th), np.full(120, z)] for z in (0.0, 30.0, 60.0, 90.0)]
    zr = np.linspace(0, 90, 50)
    rail = np.c_[20 + 6 * np.sin(np.pi * zr / 90), np.zeros(50), zr]   # the +x silhouette bulges out 6 mm
    s = S.loft(secs, rails=[rail], n=96, poles=(24, 12), start=lambda i: secs[i][0])
    m = S.mesh(s, 0.02)
    import trimesh
    _, d, _ = trimesh.proximity.closest_point(m, rail[5:-5])
    assert d.max() < 0.25, d.max()           # the B-spline fit to the Gordon surface
    assert s.fit["networkMismatch"] < 1e-6
    plain = S.loft(secs, n=96, poles=(24, 12), start=lambda i: secs[i][0])
    assert np.abs(plain(np.array([0.0]), np.array([0.5]))[0, 0, 0] - 20.0) < 0.3
    print("loft ok")


def test_fill_solid_continuity_sdf():
    s = _tube()
    f = s.face()
    tops, bots = _ends(f)
    assert len(tops) == 1 and len(bots) == 1
    sharp, e0 = S.fill([(e, f, "G0") for e in tops])
    smooth, e1 = S.fill([(e, f, "G1") for e in tops])
    base, _ = S.fill([(e, f, "G0") for e in bots])
    r0, _ = Q.edge_continuity(S.sew_solid([f, sharp, base]))
    r1, _ = Q.edge_continuity(S.sew_solid([f, smooth, base]))
    top0 = [r for r in r0 if r["point"][2] > 50][0]
    top1 = [r for r in r1 if r["point"][2] > 50][0]
    assert top0["G1"] > 45.0, top0          # a flat-ish cap on a tube: a corner
    assert top1["G1"] < 0.5, top1           # held tangent
    assert e1["G1"] < 0.01
    # a G2 boundary is built G2 (the variational cap), not quietly G1
    dome, e2 = S.fill([(e, f, "G2") for e in tops])
    assert e2["G0"] < 5e-4 and e2["G1"] < np.radians(0.1) and e2["G2"] < 5e-4, e2
    r2, _ = Q.edge_continuity(S.sew_solid([f, dome, base]))
    top2 = [r for r in r2 if r["point"][2] > 50]
    assert max(r["G1"] for r in top2) < 0.1 and max(r["G2"] for r in top2) < 1e-3, top2
    md = S.mesh(dome, 0.05)
    assert 110.0 < md.bounds[1][2] < 125.0, "the tube's end closes as a dome, not a flat lid"
    sol = S.sew_solid([f, sharp, base])
    m = S.mesh(sol, 0.05)
    assert m.is_watertight
    d = S.sdf(sol, pitch=1.0)
    v = d(np.array([[0.0, 0.0, 50.0], [40.0, 0.0, 50.0], [0.0, 0.0, -6.0]]))
    assert v[0] < -10 and v[1] > 10 and v[2] > 0
    with tempfile.TemporaryDirectory() as tmp:
        p = S.write_step(sol, os.path.join(tmp, "tube.step"))
        assert os.path.getsize(p) > 5000
    sc = Q.surface_curvature(s, (16, 16))
    assert 5.0 < sc["minRadius"] < 25.0
    print("fill, solid, continuity, sdf ok")


def _sphere_patch(R=50.0):
    x = np.linspace(-30, 30, 61)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return S.fit_surface(np.stack([X, Y, np.sqrt(R * R - X * X - Y * Y)], -1), (24, 24), fair=1e-9)


def _on_sphere(xy, R=50.0):
    return np.c_[xy, np.sqrt(R * R - (xy ** 2).sum(1))]


def test_jerk_plateau():
    """Least change of curvature: the sweep leaves a plateau with its
    curvature growing from zero, a fifth as fast as the thin plate's."""
    g = np.linspace(-30, 30, 41)
    X, Z = np.meshgrid(g, g, indexing="ij")
    r = np.hypot(X, Z)
    pins = np.c_[X[r <= 8], np.full((r <= 8).sum(), 4.0), Z[r <= 8]]
    soft = np.c_[X[r >= 22], np.zeros((r >= 22).sum()), Z[r >= 22], np.full((r >= 22).sum(), 10.0)]
    x = np.linspace(0, 30, 601)
    out = {}
    for en, fair in (("bending", 1e-2), ("jerk", 1.0)):
        p = S.fair_patch((-30, 30), (-30, 30), pins, poles=(36, 36), soft=soft, energy=en, fair=fair)
        h = p.height(x, 0 * x)
        k = np.gradient(np.gradient(h, x), x)
        out[en] = (np.abs(k).max(), np.abs(np.gradient(k, x)).max(), h)
        assert abs(h[0] - 4.0) < 1e-3 and abs(h[-1]) < 0.1 and h.min() > -0.1
    assert out["jerk"][0] < 0.8 * out["bending"][0], "a lower peak curvature"
    assert out["jerk"][1] < 0.35 * out["bending"][1], "and curvature that changes far more slowly"
    print("jerk plateau ok", out["bending"][:2], out["jerk"][:2])


def test_cut_and_g2_cap_follow_a_sphere():
    """An opening cut into a sphere and capped: the G2 cap carries the
    sphere's curvature across the edge and so follows the sphere; the G1
    cap leaves a flat spot."""
    R = 50.0
    s = _sphere_patch(R)
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    loop = _on_sphere(np.c_[12 * np.cos(th), 9 * np.sin(th) + 2], R)
    outer, inner = S.cut(s, loop, keep="both")
    holes = S.hole_edges(outer)
    assert len(holes) == 1 and len(holes[0]) == 1
    area_cut = S.mesh(inner, 0.02).area
    assert abs(S.mesh(outer, 0.02).area + area_cut - S.mesh(s, 0.02).area) < 0.05
    dev = {}
    for c in ("G1", "G2"):
        fc, err = S.cap([(e, outer, c) for e in holes[0]], poles=(16, 16))
        assert err["G0"] < 1e-4 and err["G1"] < 0.01, err
        m = S.mesh(fc, 0.02)
        assert abs(m.area - area_cut) < 0.5
        dev[c] = float(np.abs(np.linalg.norm(m.vertices, axis=1) - R).max())
        if c == "G2":
            assert err["G2"] < 2e-4, err
            rows, summ = Q.edge_continuity(S.sew([outer, fc]))
            assert summ["edges"] >= 1 and summ["worstG1"] < 0.01 and summ["worstG2"] < 5e-4, summ
    assert dev["G2"] < 0.005 and dev["G1"] > 5 * dev["G2"], dev
    print("cut and cap ok", dev)


def test_blend_g2():
    """A boss blended into an opening: curvature carried across both joins."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    s = _sphere_patch()
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    outer = S.cut(s, _on_sphere(np.c_[14 * np.cos(th) + 3, 11 * np.sin(th) + 2]))
    hole = S.hole_edges(outer)[0][0]
    u, v = np.linspace(0, 1, 48, endpoint=False), np.linspace(0, 1, 8)
    T = np.stack([np.c_[3 + 6 * np.cos(2 * np.pi * u) + 0.8 * vv, 2 + 6 * np.sin(2 * np.pi * u), np.full(48, 56 + 8 * vv)]
                  for vv in v], 1)
    tube = S.fit_surface(T, (16, 5), uperiodic=True, fair=1e-9).face()
    bot = min(S.boundary_edges(tube), key=lambda e: BRepAdaptor_Curve(e).Value(0.3).Z())
    worst = {}
    for c in ("G1", "G2"):
        b = S.blend((outer, hole), (tube, bot), c)
        assert b.fit["G0"] < 5e-4, b.fit
        rows, summ = Q.edge_continuity(S.sew([outer, b, tube]))
        assert summ["edges"] >= 2 and summ["worstG1"] < 0.01, summ
        worst[c] = summ["worstG2"]
    assert worst["G2"] < 2e-4 and worst["G1"] > 0.01, worst       # the sphere's curvature is 0.02 /mm
    # joins: every pair of faces graded, with a close-up of each
    g1 = Q.joins(S.sew([outer, S.blend((outer, hole), (tube, bot), "G1"), tube]))
    assert len(g1) == 2 and g1[0]["grade"] == "tangent" and g1[0]["G2"] > 0.01, g1
    with tempfile.TemporaryDirectory() as tmp:
        g2 = Q.join_sheet(S.sew([outer, b, tube]), os.path.join(tmp, "joins.png"))
        assert os.path.getsize(os.path.join(tmp, "joins.png")) > 20000
    assert len(g2) == 2 and all(j["grade"] == "curvature" for j in g2) and all(3.0 < j["radius"] < 20.0 for j in g2), g2
    # the lead: how far each face's shape carries into the blend
    near, far = S.blend((outer, hole), (tube, bot), lead=0.6), S.blend((outer, hole), (tube, bot), lead=(1.0, 1.5))
    assert near.fit["lead"][0] < 0.7 * b.fit["lead"][0] and far.fit["lead"][1] > 1.4 * b.fit["lead"][1]
    mid = lambda q: q(np.array([0.25]), np.array([0.5]))[0, 0]      # noqa: E731
    assert np.linalg.norm(mid(near) - mid(far)) > 0.3
    print("blend ok", worst)


def test_fillet_two_crossing_faces():
    """A post standing through a sphere: both cut back 3 mm from the other
    and joined G2, in place of the sharp intersection."""
    s = _sphere_patch()
    u, v = np.linspace(0, 1, 48, endpoint=False), np.linspace(0, 1, 8)
    T = np.stack([np.c_[3 + 6 * np.cos(2 * np.pi * u), 2 + 6 * np.sin(2 * np.pi * u), np.full(48, 44 + 16 * vv)]
                  for vv in v], 1)
    post = S.fit_surface(T, (16, 5), uperiodic=True, fair=1e-9)
    a, b, c = S.fillet(s, post, 3.0, keep=(None, (3.0, 2.0, 60.0)))
    assert b.fit["G0"] < 5e-4, b.fit
    assert S._area(a) < S._area(s.face()) - 150 and S.mesh(c, 0.05).bounds[0][2] > 51.0     # a hole in a; c above the skin
    rows, summ = Q.edge_continuity(S.sew([a, b, c]))
    assert summ["edges"] >= 2 and summ["worstG0"] < 1e-3 and summ["worstG1"] < 0.1 and summ["worstG2"] < 5e-4, summ
    # the new edge of the sphere lies 3 mm from the post's surface (a cylinder of radius 6 about (3, 2))
    kept, edges = S.setback(s, post, 3.0)
    fr = S._edge_frames(kept, edges, np.linspace(0, 1, 40, endpoint=False))
    assert np.abs(np.hypot(fr["P"][:, 0] - 3, fr["P"][:, 1] - 2) - 9.0).max() < 0.01
    print("fillet ok", summ)


def test_edit_stays_fair():
    g = np.linspace(-30, 30, 31)
    X, Z = np.meshgrid(g, g, indexing="ij")
    s = S.fit_surface(np.stack([X, 0.004 * (X ** 2 + Z ** 2), Z], -1), (20, 20))
    e = S.edit(s, [(0.5, 0.5, (0, 3, 0))], hold=[(0.25, 0.25)], freeze=3)
    at = lambda q, a, b: q(np.array([a]), np.array([b]))[0, 0]      # noqa: E731
    assert np.allclose(at(e, 0.5, 0.5) - at(s, 0.5, 0.5), (0, 3, 0), atol=1e-9)
    assert np.allclose(at(e, 0.25, 0.25), at(s, 0.25, 0.25), atol=1e-9)
    assert np.abs(e.poles[:3] - s.poles[:3]).max() == 0 and np.abs(e.poles[:, -3:] - s.poles[:, -3:]).max() == 0
    # a swell, no ripple: the change falls away steadily from the handle
    d = (e(np.linspace(0.5, 1, 30), np.array([0.5])) - s(np.linspace(0.5, 1, 30), np.array([0.5])))[:, 0, 1]
    assert np.all(np.diff(d) < 1e-9) and d[-1] == 0.0 and d.min() > -1e-9
    assert Q.surface_curvature(e, (30, 30))["minRadius"] > 15.0
    print("edit ok")


def test_mesh_fairness():
    import trimesh
    sp = trimesh.creation.icosphere(5, radius=20)
    c = Q.mesh_curvature(sp)
    assert np.allclose(np.percentile(c["k1"], [1, 99]), 0.05, atol=0.002)
    r = Q.fairness(sp, min_radius=3)
    assert r["score"]["wobbleArea"] == 0 and r["score"]["tightArea"] == 0
    V = sp.vertices.copy()
    V *= (1 + 0.03 * np.exp(-(np.linalg.norm(V - [0, 0, 20], axis=1) / 4) ** 2))[:, None]
    lump = trimesh.Trimesh(V, sp.faces)
    r = Q.fairness(lump, min_radius=3)
    assert len(r["wobble"]) == 1 and np.linalg.norm(np.array(r["wobble"][0]["at"]) - [0, 0, 20.6]) < 2
    # leave the lump out on purpose: nothing found
    r = Q.fairness(lump, min_radius=3, ignore=lambda P: P[:, 2] > 15)
    assert not r["wobble"]
    from loadpath.raster import Camera
    with tempfile.TemporaryDirectory() as tmp:
        cam = Camera(direction="front", width=200, height=200)
        z = Q.zebra_image(lump, cam)
        assert z.shape == (200, 200, 3) and z.std() > 20
        Q.quality_sheet([sp, lump], os.path.join(tmp, "q.png"), [cam], titles=["sphere", "lump"])
        assert os.path.getsize(os.path.join(tmp, "q.png")) > 10_000
    print("mesh fairness ok")


def _drawn(outline_mm, scale, size=(400, 520), wobble=0.6, seed=0):
    """A pencil-ish drawing of an outline (k, 2 mm): grey paper, a dot grid,
    a dark wobbly stroke, at `scale` px per mm."""
    from skimage.draw import line
    rng = np.random.default_rng(seed)
    H, W = size
    img = np.full((H, W, 3), 225.0)
    img[::20, ::20] = 150.0
    P = outline_mm * scale
    P = P - P.mean(0) + [W / 2, H / 2]
    P += rng.normal(0, wobble, P.shape)
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        rr, cc = line(int(H - a[1]), int(a[0]), int(H - b[1]), int(b[0]))
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        for dr in (-1, 0, 1):
            img[np.clip(rr[ok] + dr, 0, H - 1), cc[ok]] = 60.0
    return img.astype(np.uint8)


def test_sketch_match():
    """A drawn T traced, registered at an unknown scale, and measured: the
    shape it was drawn from matches closely, a fattened one shows where."""
    import trimesh
    head = trimesh.creation.box(extents=(120, 30, 40))
    head.apply_translation((0, 0, 20))
    stem = trimesh.creation.box(extents=(40, 30, 90))
    stem.apply_translation((0, 0, -45))
    T = trimesh.boolean.union([head, stem]) if trimesh.boolean.engines_available else trimesh.util.concatenate([head, stem])
    sil = SM.model_silhouette(T, "back")
    img = _drawn(sil["contour"], 2.5)
    m = SM.match_view(T, "back", img, name="T")
    assert abs(m["scale"] - 1 / 2.5) < 0.01, m["scale"]
    assert m["mean"] < 1.5 and m["iou"] > 0.95, SM.summary(m)
    fat = stem.copy()
    fat.apply_scale((1.4, 1.0, 1.0))
    T2 = trimesh.boolean.union([head, fat]) if trimesh.boolean.engines_available else trimesh.util.concatenate([head, fat])
    m2 = SM.match_view(T2, "back", img, name="fat stem", scale=m["scale"])
    side = [s for s in m2["segments"] if s["at"][1] < -20]
    assert max(s["mean"] for s in side) > 4.0, m2["segments"]
    rail = SM.outline_curve(m)
    pts = rail.sample(200)
    assert np.abs(pts[:, 1]).max() < 1e-6        # in the view's plane
    with tempfile.TemporaryDirectory() as tmp:
        SM.match_sheet([m, m2], os.path.join(tmp, "s.png"))
        assert os.path.getsize(os.path.join(tmp, "s.png")) > 10_000
    print("sketch match ok")


def _corner(kind, n, R=3.0):
    """An extruded 90 degree corner, flat - round - flat, as a strip mesh:
    an arc, or a round whose curvature grows from zero and falls back."""
    import trimesh
    s = np.linspace(0, 1, 4001)
    th = (np.pi / 2) * (s if kind == "arc" else s - np.sin(2 * np.pi * s) / (2 * np.pi))
    L = R * np.pi / 2 * (1 if kind == "arc" else 2)
    x, z = np.cumsum(np.cos(th)) * L / 4000, -np.cumsum(np.sin(th)) * L / 4000
    i = np.linspace(0, 4000, n + 1).astype(int)
    prof = np.r_[[[-15, 0]], np.c_[x[i], z[i]], [[x[-1], z[-1] - 15]]]
    ys = np.linspace(0, 40, 9)
    V = np.array([[px, y, pz] for px, pz in prof for y in ys])
    m = len(ys)
    F = [t for i in range(len(prof) - 1) for j in range(m - 1)
         for t in ([i * m + j, i * m + m + j, i * m + j + 1], [i * m + j + 1, i * m + m + j, i * m + m + j + 1])]
    return trimesh.Trimesh(V, F, process=False)


def test_mesh_joins_tell_an_arc_from_a_round_that_eases_in():
    import trimesh
    # a sharp box: one crease, every edge of it
    js = Q.mesh_joins(trimesh.creation.box(extents=(40, 30, 12)))
    assert [j["grade"] for j in js] == ["crease"] and abs(js[0]["angle"] - 90) < 1e-6 and abs(js[0]["length"] - 328) < 1e-6
    for n in (8, 16, 32):
        arc, = Q.mesh_joins(_corner("arc", n), min_radius=20)
        assert arc["grade"] == "tangent" and abs(arc["roundRadius"] - 3.0) < 0.05 and arc["rim"] > 0.95, arc
        eased, = Q.mesh_joins(_corner("eased", n), min_radius=20)
        assert eased["grade"] == "curvature" and eased["rim"] < 0.3, eased
    coarse, = Q.mesh_joins(_corner("arc", 4), min_radius=20)        # three facets across: the mesh cannot say
    assert coarse["grade"] == "unresolved", coarse
    with tempfile.TemporaryDirectory() as tmp:
        Q.mesh_join_sheet(_corner("arc", 16), os.path.join(tmp, "j.png"), min_radius=20)
        assert os.path.getsize(os.path.join(tmp, "j.png")) > 10_000
    print("mesh joins ok")


def test_freeform_document_builds_a_closed_g2_solid_and_audits():
    import json

    from loadpath import freeform, mcp_server
    from loadpath.schema import lint
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(root, "examples", "freeform", "pod.json")) as fh:
        assert not lint(json.load(fh), "freeform")
    doc = {"name": "capsule", "closed": True,
           "surfaces": {"tube": {"loft": [{"centre": [0, 0, z], "radii": [r, 0.8 * r]}
                                          for z, r in ((0, 8), (12, 10), (26, 10), (38, 7))], "poles": [16, 8]}},
           "steps": [{"op": "edit", "surface": "tube", "moves": [{"at": [10, 0, 19], "by": [1.5, 0, 0]}]},
                     {"op": "cap", "edges": [{"surface": "tube", "edge": [8, 0, 0]}], "as": "foot"},
                     {"op": "cap", "edges": [{"surface": "tube", "edge": [7, 0, 38]}], "as": "head"}]}
    assert not lint(doc, "freeform")
    bad = lint({"surfaces": {"a": {"loft": [{"centre": [0, 0, 0]}]}}, "steps": [{"op": "weld"}]}, "freeform")
    assert len([i for i in bad if i["level"] == "error"]) >= 2, bad
    with tempfile.TemporaryDirectory() as tmp:
        r = mcp_server.guarded(mcp_server.freeform)(doc, "out", tmp)
        assert r["ok"] and r["closed"] and r["watertight"] and r["faces"] == 3, r
        assert r["grades"] == {"curvature": 2}, r["joins"]
        assert r["steps"][0]["handleError"] < 1e-6 and all(s["G0"] < 5e-4 and s["G2"] < 2e-3 for s in r["steps"][1:]), r["steps"]
        for k in ("step", "stl", "sheet", "report"):
            assert os.path.getsize(r[k]) > 300, k
        a = mcp_server.guarded(mcp_server.check_joins)("out/capsule.step", base_dir=tmp)       # the STEP read back
        assert a["ok"] and a["kind"] == "brep" and a["grades"] == {"curvature": 2}, a
        m = mcp_server.guarded(mcp_server.check_joins)("out/capsule.stl", base_dir=tmp)
        assert m["ok"] and m["kind"] == "mesh" and "crease" not in m["grades"], m["grades"]
        # an edge left open: built and written, but not ok when the document says closed
        doc["steps"].pop()
        r = freeform.build(doc, "open", tmp)
        assert not r["ok"] and not r["closed"] and "open" in r["error"], r
        e = mcp_server.guarded(mcp_server.check_joins)("nope.step", base_dir=tmp)
        assert e["ok"] is False and "no such file" in e["error"], e
    print("freeform document ok")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all surfacing tests passed")


def test_sdf_far_band_and_cubic():
    """A blend reads the field far from the skin: with a wide band and cubic
    interpolation it is the true distance there and smooth between nodes;
    the default (3 mm band, trilinear) steps."""
    import trimesh
    ball = trimesh.creation.icosphere(subdivisions=5, radius=20.0)
    P = np.column_stack([np.linspace(30.0, 31.0, 41), np.full(41, 3.1), np.full(41, 2.3)])
    true = trimesh.proximity.closest_point(ball, P)[1]                # outside: distance to the mesh
    coarse = S.sdf(ball, pitch=0.5, bounds=((-36.0,) * 3, (36.0,) * 3))(P)
    box = ((-36.0, -36.0, -36.0), (36.0, 36.0, 36.0))                 # the grid must reach the blend too
    good = S.sdf(ball, pitch=0.5, bounds=box, band=15.0, interp="cubic")(P)
    assert np.abs(good - true).max() < 0.02
    # second differences: the cubic field's curvature along the line is smooth
    d2 = lambda v: np.abs(np.diff(v, 2)).max()                         # noqa: E731
    assert d2(good) < 5 * d2(true) + 1e-5, (d2(good), d2(true))
    assert np.abs(coarse - true).max() > 3 * np.abs(good - true).max()


def test_basis_cache_keeps_no_large_evaluations():
    from loadpath import surfacing as SF
    SF._BASIS_CACHE.clear()
    t = np.linspace(0.0, 1.0, 200_000)
    SF.basis(12, 3, False, t)
    SF.basis(12, 3, False, t[:50])
    assert len(SF._BASIS_CACHE) == 1


def test_soft_subtract_rounds_the_rim_and_cuts_a_proud_face_flat():
    from loadpath.sdf_kit import box, soft_subtract, subtract
    plate = lambda P: box(P, (0, 0, -2), (10, 10, 2))           # top face z = 0
    hole = lambda P: box(P, (0, 0, 5), (3, 3, 5))               # a pocket from z = 0 up, 6 x 6
    rim = 1.0
    # just outside the pocket's corner on the plate's top: solid with a hard cut, rounded away with a soft one
    P = np.array([[3.1, 3.1, -0.05]])
    assert subtract(plate(P), hole(P))[0] < 0
    assert soft_subtract(plate(P), hole(P), rim)[0] > 0
    # a pad meant to land on the pocket's floor (z = 0): flush, it is shaved by rim / 4
    flush = lambda P: np.minimum(plate(P), box(P, (0, 0, -0.5), (2, 2, 0.5)))
    top = lambda f: max(z for z in np.arange(-1, 1, 0.005) if f(np.array([[0, 0, z]]))[0] < 0)
    assert abs(top(lambda P: soft_subtract(flush(P), hole(P), rim)) + rim / 4) < 0.02
    # drawn `rim` proud, the pocket cuts it flat where it should be
    proud = lambda P: np.minimum(plate(P), box(P, (0, 0, -0.5 + rim / 2), (2, 2, 0.5 + rim / 2)))
    assert abs(top(lambda P: soft_subtract(proud(P), hole(P), rim))) < 0.01
