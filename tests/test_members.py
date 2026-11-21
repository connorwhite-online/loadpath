"""members.py: a density redrawn as clean members."""
import numpy as np

from loadpath.members import members_from_density, rdp


def _cap(X, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((X - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(X - (a + t[..., None] * ab), axis=-1) - r


def _y(p=0.5):
    ax = np.arange(-20, 20, p)
    X = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
    d = np.minimum.reduce([_cap(X, (0, 0, 0), (0, 0, 18), 2.0), _cap(X, (0, 0, 0), (15, 0, -15), 2.0),
                           _cap(X, (0, 0, 0), (-15, 0, -15), 1.5)])
    return X, d, (-20, -20, -20), p


def test_a_y_becomes_three_members_of_their_own_sizes():
    X, d, o, p = _y()
    M = members_from_density((d < 0).astype(float), o, p, min_member=2.0)
    assert M.report["members"] == 3, M.report
    assert set(M.report["diameters mm (segments)"]) <= {"3", "3.5", "4", "4.5"}, M.report
    # the rebuilt field is close to the original surface, and every member rises >= 45 degrees
    P = X.reshape(-1, 3)[::37]
    dd = d.reshape(-1)[::37]
    on = np.abs(dd) < 0.25
    assert np.isfinite(M.sdf(P)).all()
    assert np.abs(M.sdf(P[on])).mean() < 0.6
    assert M.overhang("+z")["flatter than 45 deg"] == 0


def test_a_member_rooted_in_a_kept_plate_ends_in_it():
    p = 0.5
    ax = np.arange(-15, 15, p)
    X = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
    plate = np.abs(X[..., 2] + 10.0) - 1.0
    bar = _cap(X, (0, 0, -10), (0, 0, 10), 1.5)
    solid = np.minimum(plate, bar) < 0
    M = members_from_density(solid.astype(float), (-15, -15, -15), p, exclude=plate < 0, min_member=2.0)
    assert M.report["members"] == 1 and M.report["roots"] == 1, M.report
    z = np.concatenate([M.A[:, 2], M.B[:, 2]])
    assert z.min() < -10.0 + 1.0              # it runs into the plate


def test_rdp_keeps_a_corner():
    P = np.array([[0, 0, 0], [1, 0.01, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0]], float)
    Q = rdp(P, 0.1)
    assert len(Q) == 3 and np.allclose(Q[1], [2, 0, 0])
