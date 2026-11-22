"""ridges.py: printed ridges (web + teardrop bulb), lofted, windowed, blended."""
import numpy as np

from loadpath.ridges import Ridges, profile, section


def _grid(p=0.1):
    u = np.arange(-6, 6, p) + p / 2
    v = np.arange(-1, 14, p) + p / 2
    return np.meshgrid(u, v, indexing="ij")


def test_profile_is_self_supporting():
    U, V = _grid()
    s = profile(U, V, 1.6, 10.0, 2.2, chin=45.0) <= 0
    # walking up, the solid's width never grows by more than one step per step (45 degrees)
    width = s.sum(axis=0) * 0.1
    grow = np.diff(width)
    assert grow.max() <= 2 * 0.1 + 1e-6 + 0.2       # two sides, a cell of rounding
    # the web is t wide low down and the bulb 2R wide near the top
    assert abs(width[np.searchsorted(V[0], 2.0)] - 1.6) < 0.21
    assert abs(width.max() - 4.4) < 0.21
    assert V[0][np.nonzero(width)[0].max()] < 10.0 + 0.1


def test_bulb_beats_a_wall_of_the_same_height_and_mass():
    r = section("ridge", t=1.6, H=10.0, R=2.2)
    tw = (r["area mm2"] - 2.0 * 11.0) / 10.0
    w = section("wall", t=tw, H=10.0)
    assert abs(w["area mm2"] - r["area mm2"]) < 0.5
    assert r["I mm4"] > 1.1 * w["I mm4"]
    assert min(r["Z tip mm3"], r["Z skin mm3"]) > 1.15 * min(w["Z tip mm3"], w["Z skin mm3"])


def test_ridges_loft_window_and_blend():
    L = [np.array([[-30, 0, 0], [30, 0, 0]], float), np.array([[0, -30, 0], [0, 30, 0]], float)]
    W = [np.linspace(0, 1, 2), np.zeros(2)]
    # lofted height: the x line grows with its weight
    r = Ridges(L, "+z", weight=W, t=(1.6, 2.4), H=(4.0, 12.0), R=(1.0, 2.2))
    tip = lambda r, x: max(v for v in np.arange(0, 20, 0.1) if r(np.array([[x, 0, 0]]), np.array([v]))[0] <= 0)
    assert tip(r, -25) < tip(r, 25) - 4
    # the crossing is blended: just off the corner between the two lines is inside
    assert r(np.array([[1.25, 1.25, 0]]), np.array([1.0]))[0] < 0 < min(r(np.array([[1.25, 5.0, 0]]), np.array([1.0]))[0], 0.3)
    assert r(np.array([[4.0, 4.0, 0]]), np.array([1.0]))[0] > 0
    # a deep web (height from the room) that is lightly loaded has diamond windows; loaded, none
    r = Ridges(L, "+z", weight=W, H=lambda P: np.full(len(P), 16.0))
    xs = np.arange(-28, 28, 0.25)
    mid = r(np.column_stack([xs, np.full_like(xs, 0.0), np.zeros_like(xs)]), np.full(len(xs), 7.0))
    light, heavy = mid[xs < -12], mid[xs > 20]
    assert (light > 0).any() and (light <= 0).any()
    assert (heavy <= 0).all()


def _supported(S):
    """Every solid cell (rows bottom to top) rests on a solid cell below it or one diagonal from it."""
    below = np.zeros_like(S)
    below[:, 1:] = S[:, :-1]
    below[1:, 1:] |= S[:-1, :-1]
    below[:-1, 1:] |= S[1:, :-1]
    return ~(S & ~below)[:, 1:]


def test_wall_ridges_print_on_a_side_wall_and_stand_up_on_the_floor():
    from loadpath.ridges import WallRidges

    L = [np.array([[0, -20, 0], [0, 20, 0]], float)]
    p = 0.25
    x = np.arange(0, 16, p) + p / 2
    z = np.arange(-8, 16, p) + p / 2
    X, Z = np.meshgrid(x, z, indexing="ij")
    P = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    # a side wall at x = 0 (the part at x > 0), printed up +z: a 45 degree shelf, reaching 10 mm out
    r = WallRidges(L, "+z", lambda Q: np.tile([1.0, 0, 0], (len(Q), 1)), H=(10, 10), t=(2, 2), R=(2, 2))
    S = (r(P) <= 0).reshape(X.shape)
    S[0] = True                                         # the wall
    ok = _supported(S)
    assert ok.mean() > 0.995, np.argwhere(~ok)[:5]
    assert 9.0 < x[S[1:].any(axis=1).nonzero()[0].max() + 1] < 10.6
    # the floor (normal +z): upright, 10 mm tall, a web 2 mm wide
    r = WallRidges([np.array([[-20, 0, 0], [20, 0, 0]], float)], "+z", lambda Q: np.tile([0, 0, 1.0], (len(Q), 1)),
                   H=(10, 10), t=(2, 2), R=(2, 2))
    Y, Z2 = np.meshgrid(np.arange(-6, 6, p) + p / 2, z, indexing="ij")
    S = (r(np.column_stack([np.zeros(Y.size), Y.ravel(), Z2.ravel()])) <= 0).reshape(Y.shape)
    assert abs(Z2[S].max() - 10.0) < 0.3
    assert abs(S[:, np.searchsorted(z, 3.0)].sum() * p - 2.0) < 0.3


def test_thin_keeps_half_of_a_parallel_family_and_every_crossing_line():
    from loadpath.ridges import thin

    xs = [np.array([[x, -40, 0], [x, 40, 0]], float) for x in np.arange(-33, 34, 11)]
    ys = [np.array([[-40, 40, 0], [40, 40, 0]], float)]
    _kept, _, idx = thin([*(np.linspace(a, b, 40) for a, b in xs), np.linspace(*ys[0], 40)], spacing=22.0)
    assert len(idx) == 5 and 7 in idx


def test_tidy_smooths_a_wandering_line_and_drops_short_ones():
    from loadpath.ridges import tidy

    x = np.arange(0, 80, 1.0)
    wiggle = np.column_stack([x, 0.6 * (-1.0) ** np.arange(len(x)), np.zeros_like(x)])
    short = np.array([[0, 10, 0], [20, 10, 0]], float)
    L, W = tidy([wiggle, short], [np.ones(len(x)), np.ones(2)])
    assert len(L) == 1 and len(W) == 1
    assert np.abs(L[0][5:-5, 1]).max() < 0.1


def _floor_ridges(L, **kw):
    from loadpath.ridges import WallRidges
    up = lambda Q: np.tile([0, 0, 1.0], (len(Q), 1))
    return WallRidges(L, "+z", up, H=(8, 8), t=(2, 2), R=(1.5, 1.5), **kw)


def _top(r, x, y):
    zs = np.arange(0, 14, 0.1)
    inside = r(np.column_stack([np.full_like(zs, x), np.full_like(zs, y), zs])) <= 0
    return zs[inside].max() if inside.any() else 0.0


def test_free_ends_taper_and_never_run_past_the_line():
    L = [np.array([[-30, 0, 0], [30, 0, 0]], float)]
    r = _floor_ridges(L, taper=12.0)
    assert abs(_top(r, 0, 0) - 8.0) < 0.3                    # full height mid-line
    assert _top(r, 26, 0) < 0.5 * _top(r, 0, 0)               # tapering toward the free end
    assert _top(r, 31, 0) == 0.0 and _top(r, 45, 0) == 0.0    # nothing past the end
    # a joined end keeps its section to the end
    r = _floor_ridges(L, taper=12.0, ends=[(False, True)])
    assert abs(_top(r, 29, 0) - 8.0) < 0.5 and _top(r, 31, 0) == 0.0


def test_room_running_out_tapers_instead_of_slicing():
    L = [np.array([[-40, 0, 0], [40, 0, 0]], float)]
    room = lambda C: np.where(np.abs(C[:, 0]) < 6, 1.0, 20.0)          # an obstacle over the middle
    r = _floor_ridges(L, room=room, taper=10.0)
    assert _top(r, 0, 0) == 0.0 and abs(_top(r, -30, 0) - 8.0) < 0.3
    tops = [_top(r, x, 0) for x in np.arange(-20, -6, 1.0)]
    assert np.all(np.diff(tops) <= 0.11)                      # it runs down to the gap, no cliff
    assert max(np.abs(np.diff(tops))) < 2.0


def test_network_joins_near_ends_cuts_stubs_and_leaves_free_ends():
    from loadpath.ridges import network
    host = np.linspace([-40, 0, 0], [40, 0, 0], 81)
    near = np.linspace([0, 30, 0], [0, 6, 0], 25)              # stops 6 mm short of the host
    stub = np.linspace([20, 30, 0], [20, -8, 0], 39)           # crosses it and runs on 8 mm
    far = np.linspace([-20, 40, 0], [-20, 25, 0], 16)          # nowhere near anything
    L, _, ends = network([host, near, stub, far], snap=10.0, stub=12.0)
    assert ends[1][1] and np.linalg.norm(L[1][-1] - [0, 0, 0]) < 1.0
    assert ends[2][1] and L[2][-1][1] > -1.5
    assert ends[3] == (False, False)
    assert ends[0] == (False, False)


def test_crossing_tops_meet_flush():
    L = [np.array([[-30, 0, 0], [30, 0, 0]], float), np.array([[0, -30, 0], [0, 30, 0]], float)]
    W = [np.ones(2), np.zeros(2)]
    from loadpath.ridges import WallRidges
    up = lambda Q: np.tile([0, 0, 1.0], (len(Q), 1))
    r = WallRidges(L, "+z", up, weight=W, H=(4, 10), t=(2, 2), R=(1.5, 1.5), match=6.0)
    # the low ridge rises to the tall one's height at the node, and is back down away from it
    assert abs(_top(r, 0, 2.5) - _top(r, 0, 0)) < 0.6      # the node itself is 0.4 up: the blend
    assert _top(r, 0, 20) < 6.0


def test_lines_are_faired_and_laid_on_the_skin():
    # a traced line wandering 1.5 mm off a floor (z = 0) and jittering across it
    from loadpath.ridges import WallRidges
    x = np.arange(-40, 41, 1.0)
    rng = np.random.default_rng(0)
    pl = np.column_stack([x, 0.4 * rng.standard_normal(len(x)), 1.5 + 0.3 * rng.standard_normal(len(x))])
    up = lambda P: np.tile([0.0, 0.0, 1.0], (len(P), 1))
    r = WallRidges([pl], "+z", up, H=(8.0, 8.0), taper=10.0, skin=lambda P: -P[:, 2])
    assert np.abs(r.A[:, 2]).max() < 1e-9                       # on the skin
    wig = np.abs(np.diff(r.A[:, 1], 2)).max()
    assert wig < 0.1 * np.abs(np.diff(pl[:, 1], 2)).max()       # the jitter faired out
    assert abs(r.A[0, 0] - pl[0, 0]) < 0.5 and abs(r.A[0, 1] - pl[0, 1]) < 0.05   # its ends held


def test_inside_a_ridge_is_negative_not_a_flat_zero():
    # a field flat at 0 through the web (a clamp) has no surface to find: marching cubes stair-steps it
    L = [np.array([[-30, 0, 0], [30, 0, 0]], float)]
    r = _floor_ridges(L, taper=12.0)
    f = r(np.array([[0.0, 0.0, z] for z in (1.0, 3.0, 5.0)]))
    assert (f < -0.5).all()


def test_network_merges_an_end_running_alongside_another_line():
    from loadpath.ridges import network
    a = np.column_stack([np.zeros(61), np.arange(0, 61.0), np.zeros(61)])          # up the y axis
    # comes in from the right and turns to run 3 mm beside it for its last 15 mm
    b = np.vstack([np.column_stack([np.arange(40, 3, -1.0), np.full(37, 20.0), np.zeros(37)]),
                   np.column_stack([np.full(15, 3.0), np.arange(21, 36.0), np.zeros(15)])])
    L, _, ends = network([a, b], merge=6.0)
    assert ends[1][1]
    d = np.min(np.linalg.norm(L[1][:, None] - a[None], axis=2), axis=1)
    assert (d[-5:] < 6.0).any() and np.sum((d > 1.0) & (d < 4.0)) < 6      # no long run alongside


def test_anchor_network_is_planar_and_ties_anchors():
    import trimesh
    from loadpath.ridges import anchor_network
    m = trimesh.creation.box((100.0, 100.0, 2.0)).subdivide().subdivide().subdivide().subdivide()
    top = m.vertices[:, 2] > 0.5
    keep = top[m.faces].all(axis=1)
    sub = m.submesh([np.where(keep)[0]], append=True)
    A = np.array([[-30, -30, 1], [30, -30, 1], [30, 30, 1], [-30, 30, 1], [0, 0, 1]], float)
    lines, kept = anchor_network(sub.vertices, sub.faces, A, k=4, max_length=100, clear=5.0, min_angle=30.0)
    assert len(kept) >= 6                          # the square's sides and spokes to its middle
    assert (0, 2) not in kept and (1, 3) not in kept   # its diagonals would cross the spokes
    for (i, j), P in zip(kept, lines):
        assert np.linalg.norm(P[0] - A[i]) < 4 and np.linalg.norm(P[-1] - A[j]) < 4


def test_anchor_network_mirror_is_symmetric():
    import trimesh
    from loadpath.ridges import anchor_network
    m = trimesh.creation.box((100.0, 100.0, 2.0)).subdivide().subdivide().subdivide().subdivide()
    keep = (m.vertices[:, 2] > 0.5)[m.faces].all(axis=1)
    sub = m.submesh([np.where(keep)[0]], append=True)
    A = np.array([[-30, -30, 1], [30, -30, 1], [30, 30, 1], [-30, 30, 1], [0, 0, 1], [-35, 5, 1], [35, 5, 1]], float)
    lines, kept = anchor_network(sub.vertices, sub.faces, A, k=4, max_length=100, clear=5.0, min_angle=30.0,
                                 mirror=(-1, 1, 1))
    pts = np.vstack(lines)
    from scipy.spatial import cKDTree
    assert cKDTree(pts).query(pts * [-1, 1, 1])[0].max() < 1e-6


def test_fillet_union_is_curvature_continuous_and_as_deep_as_its_radius():
    from loadpath.ridges import fillet_union
    # a floor (y < 0 solid) and a wall (x < 0 solid): the fillet fills the corner's bisector
    # r (sqrt 2 - 1) deep, as a circular fillet of radius r would
    r = 3.0
    s = np.linspace(0.0, 4.0, 4001)
    x = s / np.sqrt(2.0)                             # along the bisector, both fields x = y = s / sqrt 2 off the faces
    f = fillet_union(x, x, r)
    depth = s[np.argmin(np.abs(f))]
    assert abs(depth - r * (np.sqrt(2.0) - 1.0)) < 0.02
    # along the floor, the second difference never jumps (no crease at the fillet's edge)
    y = 0.5
    xs = np.linspace(-1.0, 12.0, 13001)
    g = fillet_union(np.full_like(xs, y), -xs, r)
    d2 = np.diff(g, 2) / (xs[1] - xs[0]) ** 2
    assert np.abs(np.diff(d2)).max() < 0.05


def test_wall_ridges_root_grows_with_height():
    from loadpath.ridges import WallRidges
    line = np.c_[np.linspace(0, 80, 81), np.zeros(81), np.zeros(81)]
    up = lambda P: np.tile([0.0, 0.0, 1.0], (len(P), 1))
    lo = WallRidges([line], "+z", up, H=(4.0, 4.0), taper=10.0, root=(0.3, 1.0, 3.0))
    hi = WallRidges([line], "+z", up, H=(9.0, 9.0), taper=10.0, root=(0.3, 1.0, 3.0))
    mid = np.array([[40.0, 3.0, 0.5]])
    assert abs(lo.root_radius(mid)[0] - 1.2) < 0.1 and abs(hi.root_radius(mid)[0] - 2.7) < 0.1
    end = np.array([[0.5, 3.0, 0.5]])                   # the tapered end: back to the least radius
    assert hi.root_radius(end)[0] < 1.3
    far = np.array([[40.0, 40.0, 0.5]])
    assert abs(hi.root_radius(far)[0] - 1.0) < 1e-6
