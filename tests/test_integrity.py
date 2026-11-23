"""loadpath.integrity: a part is one solid and its features are fused to what holds them.

The cases are the failures that reached a printed enclosure's export (2026-11-23):
a cradle left hanging when the web that held it was removed, a ring placed on a
fixed plane behind a surfaced face (fused over part of its circle, 1 mm off the
rest), a lip pulled 0.2 mm off its wall by an edge round, and roots with no fillet."""

import json

import numpy as np
import pytest

from loadpath import integrity as I
from loadpath.exterior import mesh_from_sdf
from loadpath.ridges import fillet_union

P = 0.25


def box(c, h):
    c, h = np.asarray(c, float), np.asarray(h, float)

    def f(X):
        q = np.abs(np.asarray(X, float) - c) - h
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return f


def cyl_z(c, r, z0, z1):
    def f(X):
        X = np.asarray(X, float)
        d = np.hypot(X[:, 0] - c[0], X[:, 1] - c[1]) - r
        return np.maximum(d, np.maximum(z0 - X[:, 2], X[:, 2] - z1))
    return f


def smax_c2(a, b, k):
    return -fillet_union(-a, -b, k / (6.0 * (1.0 - 2 ** -0.5)))


def part(fields, lo, hi, pitch=P):
    return mesh_from_sdf(lambda X: fields(X)["solid"], (lo, hi), pitch)[0]


# a 2 mm wall, its top face at z = 0
skin = box((0, 0, -1), (12, 12, 1))


def boss_fields(lift=0.0, fillet=None):
    boss = cyl_z((0, 0), 2.5, lift, 6.0)

    def fields(X):
        s, b = skin(X), boss(X)
        solid = np.minimum(s, b) if fillet is None else fillet_union(s, b, fillet)
        return {"solid": solid, "skin": s, "boss": b}
    return fields


LO, HI = (-14, -14, -4), (14, 14, 8)


def test_a_fused_boss_is_one_body_and_attached():
    f = boss_fields(fillet=0.8)
    m = part(f, LO, HI)
    assert len(I.bodies(m)) == 1
    r = I.attachment(f, [I.Feature("boss")], (LO, HI), pitch=P)
    assert len(r) == 1 and r[0]["passed"], r
    assert r[0]["contact"] > 10.0


def test_a_boss_hovering_off_the_wall_is_two_bodies_and_not_attached():
    f = boss_fields(lift=1.0)
    m = part(f, LO, HI)
    assert len(I.bodies(m)) == 2
    with pytest.raises(I.IntegrityError, match="2 separate bodies"):
        I.require_one_body(m)
    r = I.attachment(f, [I.Feature("boss")], (LO, HI), pitch=P)
    assert not r[0]["passed"] and r[0]["contact"] == 0.0
    assert r[0]["gap_max"] == pytest.approx(1.0, abs=0.3)


def test_a_cradle_whose_web_was_removed_fails_the_gate(tmp_path):
    """The web held the cradle; delete the web and the cradle floats 3 mm off."""
    cradle = box((0, 0, 4.5), (6, 1.0, 1.5))
    web = box((0, 0, 1.5), (1.0, 0.8, 1.6))

    def fields_with(webbed):
        def f(X):
            s, c, w = skin(X), cradle(X), web(X)
            held = fillet_union(c, w, 0.6) if webbed else c
            return {"solid": fillet_union(s, held, 0.6) if webbed else np.minimum(s, c), "skin": s, "cradle": held}
        return f
    feats = [I.Feature("cradle", min_contact=2.0, min_cover=0.0)]   # it spans: only its web is on the wall
    ok = I.gate(part(fields_with(True), LO, HI), "webbed", tmp_path, fields=fields_with(True), features=feats,
                box=(LO, HI), images=False, crease_angle=60.0, verbose=False)
    assert ok.passed, ok.failures
    with pytest.raises(I.IntegrityError, match="bodies"):
        I.gate(part(fields_with(False), LO, HI), "orphan", tmp_path, fields=fields_with(False), features=feats,
               box=(LO, HI), images=False, verbose=False)


def curved_face(X):
    """A wall whose inside face bulges toward +z by up to ~1.5 mm (a surfaced face)."""
    X = np.asarray(X, float)
    top = 1.5 * np.exp(-((X[:, 0] - 6.0) ** 2) / 20.0)
    return np.maximum(X[:, 2] - top, -(X[:, 2] - top + 2.0)) * 0.95


def ring(z0, z1):
    def f(X):
        X = np.asarray(X, float)
        r = np.hypot(X[:, 0], X[:, 1])
        return np.maximum(np.abs(r - 5.0) - 0.8, np.maximum(z0 - X[:, 2], X[:, 2] - z1))
    return f


def test_a_ring_on_a_fixed_plane_under_a_surfaced_face_is_one_body_but_not_attached():
    rg = ring(-4.0, -2.0)                        # its top on the plane z = -2, where a flat face's inside was

    def f(X):
        s, g = curved_face(X), rg(X)
        return {"solid": np.minimum(s, g), "skin": s, "ring": g}
    lo, hi = (-9, -9, -6), (9, 9, 3)
    m = part(f, lo, hi)
    assert len(I.bodies(m)) == 1                  # touching on one side: a body count alone passes it
    r = I.attachment(f, [I.Feature("ring")], (lo, hi), pitch=P)
    assert len(r) == 1 and not r[0]["passed"], r
    assert r[0]["cover"] < 0.9 and r[0]["gap_max"] > 0.5


def test_seated_carries_the_ring_onto_the_face_with_a_fused_filleted_root():
    rg = I.seated(ring(-4.0, -2.0), curved_face, direction=(0, 0, 1), reach=4.0)

    def f(X):
        s, g = curved_face(X), rg(X)
        return {"solid": fillet_union(s, g, 0.8), "skin": s, "ring": g}
    lo, hi = (-9, -9, -6), (9, 9, 3)
    m = part(f, lo, hi)
    assert len(I.bodies(m)) == 1
    r = I.attachment(f, [I.Feature("ring")], (lo, hi), pitch=P)
    assert r[0]["passed"], r
    assert not I.creases(m, 50.0, 1.0)


def test_a_lip_pulled_off_its_wall_by_an_edge_round_is_caught():
    """A lip that only abuts its wall (shares a face with it) and is then
    rounded on every edge: the round takes k/6 off the shared face too."""
    wall = box((0, 0, -1), (10, 10, 1))

    def lip_r(X):
        X = np.asarray(X, float)
        a = np.maximum(np.abs(X[:, 0]) - 8, np.abs(X[:, 1]) - 0.6)
        b = np.abs(X[:, 2] - 2.0) - 2.0
        return smax_c2(a, b, 1.5)                  # every edge rounded, the bottom face pulled up ~0.25

    def f(X):
        s, l_ = wall(X), lip_r(X)
        return {"solid": np.minimum(s, l_), "skin": s, "lip": l_}
    lo, hi = (-12, -12, -4), (12, 12, 6)
    m = part(f, lo, hi)
    r = I.attachment(f, [I.Feature("lip")], (lo, hi), pitch=P)
    assert not r[0]["passed"], r


def test_slivers_find_a_narrow_slot_and_creases_an_unfilleted_root():
    slot = box((0, 0, 3), (6, 0.1, 2.5))
    b = box((0, 0, 3), (4, 3, 3))

    def solid(X):
        return np.maximum(np.minimum(skin(X), b(X)), -slot(X))
    m = mesh_from_sdf(solid, (LO, HI), 0.1)[0]
    s = I.slivers(m, 0.4, 0.3)
    assert s["air"], "the 0.2 mm slot"
    cr = I.creases(mesh_from_sdf(lambda X: np.minimum(skin(X), b(X)), (LO, HI), P)[0])
    assert cr and cr[0]["length"] > 10.0, "the box's root on the wall, square"
    filleted = mesh_from_sdf(lambda X: fillet_union(skin(X), b(X), 1.0), (LO, HI), P)[0]
    assert not I.creases(filleted)


def test_export_stamps_and_require_passed(tmp_path):
    f = boss_fields(fillet=0.8)
    m = part(f, LO, HI)
    p = tmp_path / "good.stl"
    rep = I.export(m, p, fields=f, features=[I.Feature("boss")], box=(LO, HI), verbose=False)
    assert rep.passed and (tmp_path / "gate" / "good-bodies.png").exists()
    assert (tmp_path / "gate" / "good-sections.png").exists()
    assert I.require_passed(p)["passed"]
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(I.IntegrityError, match="changed"):
        I.require_passed(p)
    bad = tmp_path / "bad.stl"
    g = boss_fields(lift=1.0)
    with pytest.raises(I.IntegrityError):
        I.export(part(g, LO, HI), bad, fields=g, features=[I.Feature("boss")], box=(LO, HI), verbose=False)
    st = json.loads(I.stamp_path(bad).read_text())
    assert st["passed"] is False
    with pytest.raises(I.IntegrityError, match="failed"):
        I.require_passed(bad)
    with pytest.raises(I.IntegrityError, match="no integrity stamp"):
        I.require_passed(tmp_path / "nothing.stl")


def test_require_one_body_mask():
    a = np.zeros((10, 10, 10), bool)
    a[1:4, 1:4, 1:4] = True
    assert I.require_one_body_mask(a) == 1
    a[6:8, 6:8, 6:8] = True
    with pytest.raises(I.IntegrityError, match="2 separate pieces"):
        I.require_one_body_mask(a)


def test_solvers_refuse_a_part_in_pieces():
    """FE, slicing, retention, clearance and airflow do not run on a part that is not one solid."""
    from loadpath import airflow, printcheck, retention
    from loadpath.verify import surface_thickness
    g = boss_fields(lift=1.0)
    m = part(g, LO, HI, pitch=0.5)
    with pytest.raises(I.IntegrityError):
        printcheck.check_print(m, (2, 1))
    with pytest.raises(I.IntegrityError):
        retention.body_from_mesh("front", m)
    with pytest.raises(I.IntegrityError):
        surface_thickness(m)
    solid = np.zeros((8, 8, 8), bool)
    solid[1:3, 1:3, 1:3] = True
    solid[5:7, 5:7, 5:7] = True
    with pytest.raises(I.IntegrityError):
        airflow.solve(solid, 1.0, k=1.0, parts={"half": solid}, verbose=False, max_iter=1)


def test_a_boss_joined_by_its_root_fillet_alone_is_attached_and_without_it_is_not():
    """The boss stops 0.5 mm short of the wall; a 1.5 mm root fillet bridges it (fused), a plain union does not."""
    for r, ok in ((1.5, True), (None, False)):
        f = boss_fields(lift=0.5, fillet=r)
        res = I.attachment(f, [I.Feature("boss")], (LO, HI), pitch=P)
        assert res[0]["passed"] is ok, res
