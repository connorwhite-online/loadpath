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
        assert np.abs(S.basis(n, 3, False, t) - S._basis_fast(n, 3, t)).max() < 1e-12
        assert np.abs(S.basis(n, 3, False, t, 1) - S._basis_fast(n, 3, t, 1)).max() < 1e-9
    P = S.basis(8, 3, True, t)
    assert np.allclose(P.sum(1), 1.0)
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
    try:
        S.fill([(e, f, "G2") for e in tops])
        raise AssertionError("a G2 boundary should be refused, not built G1")
    except ValueError:
        pass
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all surfacing tests passed")
