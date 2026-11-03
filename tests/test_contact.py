"""
Contact in the large-deflection solver (loadpath.contact) against closed
forms: the contact tangent against finite differences, a cantilever landing
on a rigid support (propped cantilever), two fingers of one part pressed
together (self contact: the reaction splits as beam theory says), an
indenter pushed into a cantilever (a moving obstacle), and a cantilever
pressed through large deflection onto a frictionless floor (the elastica
with the floor's reaction, its tip sliding along the floor).
Run: `python tests/test_contact.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from loadpath.contact import RigidObstacle, SelfContact
from loadpath.fem import HexMesh
from loadpath.nonlinear import solve_nonlinear

E, NU = 1.0, 0.3
L, T, B, H = 40.0, 2.0, 2.0, 0.5       # 80 x 4 x 4 cells
I = B * T ** 3 / 12
K_TIP = 3 * E * I / L ** 3


def bars(*y0s):
    """Cantilevers along x clamped at x = 0, one per y offset (in cells)."""
    nx, ny, nz = round(L / H), round(T / H), round(B / H)
    ijk = np.concatenate([np.stack(np.meshgrid(np.arange(nx), np.arange(ny) + j0, np.arange(nz), indexing="ij"),
                                   -1).reshape(-1, 3) for j0 in y0s])
    m = HexMesh(ijk, H, np.zeros(3))
    fixed = np.repeat(m.node_xyz[:, 0] < 1e-9, 3)
    return m, fixed


def at(m, x=None, y=None):
    sel = np.ones(m.n_nodes, bool)
    if x is not None:
        sel &= np.abs(m.node_xyz[:, 0] - x) < 1e-9
    if y is not None:
        sel &= np.abs(m.node_xyz[:, 1] - y) < 1e-9
    return np.flatnonzero(sel)


def force_on(m, nodes, fy):
    f = np.zeros(m.n_dof)
    f[3 * nodes + 1] = fy / len(nodes)
    return f


def test_rigid_tangent_matches_finite_differences():
    m, _ = bars(0)
    rng = np.random.default_rng(4)
    for ob in (RigidObstacle("sphere", center=(20, -1.9, 1), radius=2.0),
               RigidObstacle("cylinder", center=(20, -1.8, 0), radius=2.0, axis="z"),
               RigidObstacle("plane", point=(0, -0.05, 0), normal=(0.001, 1, 0.02)),
               RigidObstacle("box", lo=(10, -5, -1), hi=(30, -0.05, 3))):
        ob._setup(m, E)
        u = np.zeros(m.n_dof)
        u[1::3] = -0.15 * np.sin(m.node_xyz[:, 0] / L * np.pi)
        u += rng.normal(size=m.n_dof) * 0.01
        f0, k0, info = ob._load(u, 1.0)
        assert info["nodes"] > 0, ob.shape
        du = rng.normal(size=m.n_dof) * 1e-7
        f1, _, _ = ob._load(u + du, 1.0)
        # load stiffness is -df/du
        assert np.linalg.norm((f0 - f1) - k0 @ du) < 1e-3 * np.linalg.norm(f1 - f0), ob.shape


def test_self_contact_tangent_matches_finite_differences_on_flat_faces():
    # two bars overlapping by a sliver: flat faces, normals fixed, so the
    # node-to-face linearization is exact there
    m, _ = bars(0, 5)
    sc = SelfContact()
    sc._setup(m, E)
    u = np.zeros(m.n_dof)
    up = m.node_xyz[:, 1] > 2.1
    u[1::3] = np.where(up, -0.504, 0.0)
    f0, k0, info = sc._load(u, 1.0)
    assert info["nodes"] > 50, info
    rng = np.random.default_rng(1)
    du = np.zeros(m.n_dof)
    du[1::3] = rng.normal(size=m.n_nodes) * 1e-7
    f1, _, _ = sc._load(u + du, 1.0)
    # across the faces (the frozen-normal linearization's exact part)
    dy = (f0 - f1)[1::3] - (k0 @ du)[1::3]
    assert np.linalg.norm(dy) < 1e-3 * np.linalg.norm((f1 - f0)[1::3])
    # equal and opposite: no net force from contact inside one part
    assert abs(f0[1::3].sum()) < 1e-9 * np.abs(f0).sum()


def test_cantilever_lands_on_a_rigid_support():
    # tip load P, a rigid roller (cylinder across the beam) a gap g under
    # mid-span: once the beam touches it, a propped cantilever.
    m, fixed = bars(0)
    a, g, P = L / 2, 0.2, 8e-5
    free_at_a = P * a ** 2 * (3 * L - a) / (6 * E * I)
    R = (free_at_a - g) * 3 * E * I / a ** 3
    tip_expect = P * L ** 3 / (3 * E * I) - R * a ** 2 * (3 * L - a) / (6 * E * I)
    roller = RigidObstacle("cylinder", center=(a, -g - 2.0, B / 2), radius=2.0, axis="z")
    tip = at(m, x=L)
    r = solve_nonlinear(m, NU, E, fixed, force_on(m, tip, -P), contact=[roller], steps=4)
    assert r.converged, r.message
    got_R = -r.history[-1]["contact0"]["force"][1]
    got_tip = -r.u[3 * tip + 1].mean()
    assert abs(got_R / R - 1) < 0.03, (got_R, R)
    assert abs(got_tip / tip_expect - 1) < 0.03, (got_tip, tip_expect)
    # it would have gone much further without the roller
    assert got_tip < 0.75 * P * L ** 3 / (3 * E * I)


def test_two_fingers_of_one_part_press_together():
    # two parallel cantilevers in one mesh, a gap of one cell between them;
    # the top one is pushed down at its tip onto the other. Both tips move
    # together after contact: (P - R) / k - R / k = g, R = (P - k g) / 2.
    m, fixed = bars(0, 5)
    g, P = H, 1e-4
    R = (P - K_TIP * g) / 2
    top_tip = at(m, x=L, y=T + H + T)
    r = solve_nonlinear(m, NU, E, fixed, force_on(m, top_tip, -P), contact=[SelfContact()], steps=4)
    assert r.converged, r.message
    c = r.history[-1]["contact0"]
    assert abs(c["force"] / R - 1) < 0.04, (c, R)
    low_tip = at(m, x=L, y=0.0)
    assert abs(-r.u[3 * low_tip + 1].mean() / (R / K_TIP) - 1) < 0.04
    assert abs(-r.u[3 * top_tip + 1].mean() / ((P - R) / K_TIP) - 1) < 0.04
    # without contact the top finger passes straight through the other
    r0 = solve_nonlinear(m, NU, E, fixed, force_on(m, top_tip, -P), steps=4)
    assert -r0.u[3 * top_tip + 1].mean() > 1.5 * g


def test_indenter_pushed_into_a_cantilever():
    # a rigid ball driven 1 mm down onto the top face near the tip: the
    # force it takes is the beam's stiffness there times the deflection
    m, fixed = bars(0)
    a = L - 1.0
    ball = RigidObstacle("sphere", center=(a, T + 2.0 + 0.1, B / 2), radius=2.0, move=(0, -1.0, 0))
    r = solve_nonlinear(m, NU, E, fixed, None, contact=[ball], steps=4)
    assert r.converged, r.message
    sec = at(m, x=a, y=T)
    d = -r.u[3 * sec + 1].mean()
    assert abs(d - 0.9) < 0.02, d                     # it follows the ball (0.1 mm gap first)
    F = r.history[-1]["contact0"]["force"][1]           # the part pushes the ball up
    expect = 3 * E * I * d / a ** 3
    assert abs(F / expect - 1) < 0.03, (F, expect)


def _elastica(F, M0):
    """Cantilever, tip force F (down) and tip moment M0 (bending it down):
    EI theta'' = -F cos(theta), theta(0) = 0, EI theta'(L) = M0. Returns the
    tip's x, y (y down) and rotation."""
    s = np.linspace(0, L, 401)

    def f(s, y):
        return np.vstack([y[1], -F / (E * I) * np.cos(y[0]), np.cos(y[0]), np.sin(y[0])])

    def bc(a, b):
        return np.array([a[0], b[1] - M0 / (E * I), a[2], a[3]])

    guess = np.vstack([0.5 * s / L, np.full_like(s, 0.5 / L), s, 0.25 * s ** 2 / L])
    sol = solve_bvp(f, bc, s, guess, tol=1e-9, max_nodes=200000)
    th, _, x, y = sol.sol(L)
    return x, y, th


def test_cantilever_pressed_onto_a_floor_through_large_deflection():
    # a dead tip load that would bend the tip 0.59 L down; a frictionless
    # floor 0.35 L under the axis stops it. The floor pushes straight up
    # at the tip's bottom edge, which slides along it as the beam curls;
    # the beam is the elastica under P - R with the small extra moment of
    # R acting T/2 sin(theta) behind the tip.
    m, fixed = bars(0)
    P = 3.0 * E * I / L ** 2
    drop = 0.35 * L                       # floor under the beam axis
    floor = RigidObstacle("plane", point=(0, T / 2 - drop, 0), normal=(0, 1, 0))

    def miss(R):
        x, y, th = _elastica(P - R, 0.0)
        for _ in range(3):                # the moment depends on the tip's rotation
            x, y, th = _elastica(P - R, R * T / 2 * np.sin(th))
        return y + T / 2 * np.cos(th) - drop, x, th

    R = brentq(lambda r: miss(r)[0], 0.0, 0.99 * P)
    _, x_tip, th_tip = miss(R)
    tip = at(m, x=L)
    r = solve_nonlinear(m, NU, E, fixed, force_on(m, tip, -P), contact=[floor], steps=10)
    assert r.converged, r.message
    got_R = -r.history[-1]["contact0"]["force"][1]
    assert abs(got_R / R - 1) < 0.03, (got_R, R)
    pos = (m.node_xyz + r.u.reshape(-1, 3))[tip].mean(axis=0)
    assert abs(pos[0] / x_tip - 1) < 0.01, (pos[0], x_tip)       # how far it slid back
    d = pos - (m.node_xyz + r.u.reshape(-1, 3))[at(m, x=L - H)].mean(axis=0)
    assert abs(np.arctan2(-d[1], d[0]) / th_tip - 1) < 0.05, (np.arctan2(-d[1], d[0]), th_tip)
    assert r.history[-1]["contact0"]["penetration"] < 0.01


def test_field_graph_tendon_finger_grips_a_pen():
    # the largeDeflection block: a TPU finger curled by a servo-wound tendon
    # through guides along its top, closing on a rigid pen, self contact on
    from loadpath.fields import FieldGraph
    from loadpath.materials import material_from_spec
    from loadpath.nonlinear import large_deflection_report
    from loadpath.schema import lint

    doc = {"nodes": {"finger": {"op": "shape", "shape": {"type": "box", "min": [0, 0, 0], "max": [40, 4, 4]}},
                     "base": {"op": "shape", "shape": {"type": "box", "min": [-1, -1, -1], "max": [0.5, 5, 5]}}},
           "output": "finger", "material": "tpu",
           "largeDeflection": {
               "fixed": "base", "pitch": 1.0, "steps": 8,
               "tendons": [{"path": [[-5, 3.5, 2]] + [{"at": [x, 3.5, 2], "radius": 0.8} for x in range(5, 41, 5)],
                            "stroke": 4.5, "stiffness": 40}],
               "contact": {"obstacles": [{"type": "cylinder", "name": "pen", "center": [18, 12, 2], "radius": 5,
                                          "axis": "z"}], "self": True},
               "probes": [{"name": "tipUp", "at": [40, 2, 2], "radius": 1.5, "direction": [0, 1, 0]}]}}
    assert lint(doc, "field") == [], lint(doc, "field")
    r = large_deflection_report(FieldGraph(doc), doc["largeDeflection"], material_from_spec("tpu"))
    assert r["converged"], r.get("message")
    pen, me = r["contact"]
    assert pen["name"] == "pen" and pen["nodes"] > 0 and pen["penetration"] < 0.02
    grip = np.linalg.norm(pen["force"])
    assert 1.0 < grip < 5.0, grip               # about 2 N on the pen
    assert me["nodes"] == 0                      # the finger does not touch itself here
    t = r["tendons"][0]
    assert 3.0 < t["travel"] < 4.5 and 0 < t["tension"] <= 40 * 4.5
    assert r["probes"]["tipUp"] > 10


TESTS = [
    test_rigid_tangent_matches_finite_differences,
    test_self_contact_tangent_matches_finite_differences_on_flat_faces,
    test_cantilever_lands_on_a_rigid_support,
    test_two_fingers_of_one_part_press_together,
    test_indenter_pushed_into_a_cantilever,
    test_cantilever_pressed_onto_a_floor_through_large_deflection,
    test_field_graph_tendon_finger_grips_a_pen,
]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
