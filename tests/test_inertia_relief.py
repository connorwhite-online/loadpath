"""
Inertia relief: a part nothing holds (a drone arm, a dropped or thrown
part), its loads balanced by its own inertia, analysed and optimized with
no supports.

Run: `python tests/test_inertia_relief.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.domain import build_domain, inertia_balance, node_mass
from loadpath.fem import rigid_body_modes
from loadpath.simp import SimpOptions, filter_radius, optimize
from loadpath.spec import study_from_dict
from loadpath.verify import verify_part

TIP = {"type": "box", "min": [99.0, -1, -1], "max": [101, 11, 11]}
FREE_BAR = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [100, 10, 10]},
    "loadCases": [{"name": "tip", "loads": [{"shape": TIP, "force": [0, 0, -10]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
    "goal": {"safetyFactor": 2},
    "inertiaRelief": True,
    "resolution": {"pitch": 2.0},
}


def test_balanced_load_has_no_net_force_or_moment():
    s = study_from_dict(FREE_BAR)
    d = build_domain(s)
    f = d.loads[0]
    xyz = d.mesh.node_xyz
    fv = f.reshape(-1, 3)
    # (all but the inertia of the six held dofs: 0.02%)
    assert np.abs(fv.sum(axis=0)).max() < 1e-3 * 10
    assert np.abs(np.cross(xyz - xyz.mean(axis=0), fv).sum(axis=0)).max() < 1e-3 * 10 * 100
    # the 3-2-1 hold takes (next to) nothing
    assert np.abs(d.raw_loads[0]).sum() > 9.0
    fb, acc = inertia_balance(d.raw_loads[0], node_mass(d.mesh, 1.0), rigid_body_modes(xyz))
    assert np.allclose(fb[~d.fixed], f[~d.fixed])


def test_free_bar_bends_as_beam_theory_says():
    # a free-free bar pushed at one end: the inertia it takes on bends it
    # most at two thirds of the way from the push's far end, M = 4 F L / 27
    s = study_from_dict(FREE_BAR)
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    rep = verify_part(s, bar, max_cells=40_000, refine=0)
    sigma = (4 * 10 * 100 / 27) * 5 / (10 * 10 ** 3 / 12)          # 0.889 MPa
    peak = 50.0 / rep["safetyFactor"]
    assert abs(peak - sigma) / sigma < 0.15, (peak, sigma)
    # held at the root instead it would read 6.7x that
    held = study_from_dict({**{k: v for k, v in FREE_BAR.items() if k != "inertiaRelief"},
                            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}]})
    rep2 = verify_part(held, bar, max_cells=40_000, refine=0)
    assert rep2["safetyFactor"] < 0.3 * rep["safetyFactor"]


def test_optimizer_designs_a_free_part():
    s = study_from_dict({**FREE_BAR, "designSpace": {"type": "box", "min": [0, 0, -10], "max": [100, 10, 20]}})
    d = build_domain(s)
    r = optimize(d, 0.3, SimpOptions(volume_fraction=0.3, filter_radius=filter_radius(d, None), max_iterations=40))
    assert np.isfinite(r.compliance) and r.history[-1]["compliance"] < r.history[0]["compliance"]
    assert abs(r.xphys[~d.preserve].mean() - 0.3) < 0.02
    # its mass sits where the bending is: more material in the middle third
    # than at the free end it is pushed from the far side of
    x = (d.mesh.elem_ijk[:, 0] + 0.5) * d.pitch
    mid = r.xphys[(x > 50) & (x < 80)].mean()
    far = r.xphys[x < 10].mean()
    assert mid > far, (mid, far)


END_A = {"type": "box", "min": [-1, -1, -1], "max": [2.01, 11, 11]}
END_B = {"type": "box", "min": [97.99, -1, -1], "max": [101, 11, 11]}
MID = {"type": "box", "min": [49, -1, 9.99], "max": [51, 11, 11]}


def test_carried_masses_take_the_inertia():
    # a dumbbell: 1 kg at each end of a 12 g bar, pushed down in the middle.
    # The ends' inertia reacts the push, so the bar bends as if simply
    # supported there, M = F/2 x 49 mm; balanced by the bar's own mass alone
    # the middle would read far less
    s = study_from_dict({**FREE_BAR, "loadCases": [{"name": "mid", "loads": [{"shape": MID, "force": [0, 0, -10]}]}],
                         "inertiaRelief": {"masses": [{"shape": END_A, "mass": 1.0}, {"shape": END_B, "mass": 1.0}]}})
    assert abs(s.carried_mass - 2.0) < 1e-12
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    rep = verify_part(s, bar, max_cells=40_000, refine=0)
    sigma = 5.0 * 49.0 * 5 / (10 * 10 ** 3 / 12)                     # 1.47 MPa
    peak = 50.0 / rep["safetyFactor"]
    assert abs(peak - sigma) / sigma < 0.15, (peak, sigma)
    bare = study_from_dict({**FREE_BAR, "loadCases": s.raw["loadCases"]})
    rep2 = verify_part(bare, bar, max_cells=40_000, refine=0)
    assert 50.0 / rep2["safetyFactor"] < 0.6 * sigma
    # the optimizer's rebalancing on a design sees them too
    d = build_domain(s)
    assert d.carried is not None and abs(d.carried.sum() * 1.2e-6 - 2.0) < 1e-9


def test_free_free_modes():
    # a free-free bar's first bending mode, (4.730^2 / 2 pi L^2) sqrt(EI / rho A)
    from loadpath.verify import modal_check
    s = study_from_dict({**FREE_BAR, "vibration": {"minFrequency": 100, "modes": 2}})
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    out = modal_check(s, bar)
    ei, rho_a, length = 2e9 * (0.01 ** 4 / 12), 1200 * 1e-4, 0.1
    f1 = 4.730 ** 2 / (2 * np.pi * length ** 2) * np.sqrt(ei / rho_a)          # 1327 Hz
    assert out["free"] and abs(out["firstFrequency"] - f1) / f1 < 0.08, (out["firstFrequency"], f1)
    # 1 kg on each end pulls it far down: a carried mass is in the modes too
    sm = study_from_dict({**FREE_BAR, "vibration": {"minFrequency": 100},
                          "inertiaRelief": {"masses": [{"shape": END_A, "mass": 1.0}, {"shape": END_B, "mass": 1.0}]}})
    out2 = modal_check(sm, bar)
    assert out2["carried"] == 2.0 and out2["firstFrequency"] < 0.2 * out["firstFrequency"]
    # 50 g on each end as rigid bodies: about the lumped reading; lifted 40 mm
    # off the bar they rock on it, far lower
    freq = []
    for h in (5, 45):
        ends = [{"shape": END_A, "mass": 0.05, "at": [1, 5, h], "size": [2, 10, 10]},
                {"shape": END_B, "mass": 0.05, "at": [99, 5, h], "size": [2, 10, 10]}]
        freq.append(modal_check(study_from_dict({**FREE_BAR, "vibration": {"minFrequency": 100},
                                                 "inertiaRelief": {"masses": ends}}), bar)["firstFrequency"])
    lumped = modal_check(study_from_dict({**FREE_BAR, "vibration": {"minFrequency": 100}, "inertiaRelief": {
        "masses": [{"shape": END_A, "mass": 0.05}, {"shape": END_B, "mass": 0.05}]}}), bar)["firstFrequency"]
    assert abs(freq[0] - lumped) / lumped < 0.1 and freq[1] < 0.4 * freq[0], (freq, lumped)


def test_carried_body_off_its_mounts():
    # the same dumbbell with its end masses as rigid bodies centred on their
    # ends reads the same bending; lifted 40 mm above the bar, a push along
    # the bar makes each end's inertia twist its mount by m a h
    from loadpath.domain import carried_bodies, inertia_balance, node_mass
    from loadpath.fem import rigid_body_modes
    ends = [{"shape": END_A, "mass": 1.0, "at": [1, 5, 5]}, {"shape": END_B, "mass": 1.0, "at": [99, 5, 5]}]
    s = study_from_dict({**FREE_BAR, "loadCases": [{"name": "mid", "loads": [{"shape": MID, "force": [0, 0, -10]}]}],
                         "inertiaRelief": {"masses": ends}})
    assert abs(s.carried_mass - 2.0) < 1e-12 and len(s.inertia_bodies) == 2
    bar = trimesh.creation.box(extents=[100, 10, 10])
    bar.apply_translation([50, 5, 5])
    rep = verify_part(s, bar, max_cells=40_000, refine=0)
    sigma = 5.0 * 49.0 * 5 / (10 * 10 ** 3 / 12)
    assert abs(50.0 / rep["safetyFactor"] - sigma) / sigma < 0.15
    lifted = [{**e, "at": [e["at"][0], 5, 45]} for e in ends]
    push = {"name": "push", "loads": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]},
                                       "force": [10, 0, 0]}]}
    s2 = study_from_dict({**FREE_BAR, "loadCases": [push], "inertiaRelief": {"masses": lifted}})
    d = build_domain(s2)
    f = d.raw_loads[0]
    xyz = d.mesh.node_xyz
    phi = rigid_body_modes(xyz)
    m = node_mass(d.mesh, 1.0)
    bodies = carried_bodies(s2, d.mesh, phi)
    fb, _ = inertia_balance(f, m, phi, bodies)
    # the frame's part and the bodies' wrenches: nothing left over
    mrb = phi.T @ (phi * np.repeat(m, 3)[:, None]) + sum(b[2] for b in bodies)
    a = np.linalg.solve(mrb, phi.T @ f)
    tot = fb.reshape(-1, 3).sum(axis=0)
    assert np.abs(tot).max() < 1e-8, tot
    mom = np.cross(xyz - xyz.mean(axis=0), fb.reshape(-1, 3)).sum(axis=0)
    assert np.abs(mom).max() < 1e-6, mom
    # what end A's mounts take: about its own nodes, a moment of size m a h
    da, ma = bodies[0][0], bodies[0][1]
    fa = np.zeros_like(f)
    fa[da] = -(ma @ (bodies[0][2] @ a))
    pa = xyz[da[::3] // 3]
    moment_a = np.cross(pa - pa.mean(axis=0), fa[da].reshape(-1, 3)).sum(axis=0)
    acc_x = 10.0 / (2.0 + 0.012)
    assert abs(abs(moment_a[1]) - 1.0 * acc_x * 40.0) / (acc_x * 40.0) < 0.05, moment_a
    assert d.bodies and len(d.bodies) == 2


def test_inertia_relief_spec_errors():
    with pytest.raises(ValueError, match="Drop the supports"):
        study_from_dict({**FREE_BAR, "supports": [{"shape": TIP}]})
    with pytest.raises(ValueError, match="falls"):
        study_from_dict({**FREE_BAR, "gravity": [0, 0, -9.81]})
    with pytest.raises(ValueError, match="inertiaRelief"):
        study_from_dict({k: v for k, v in FREE_BAR.items() if k != "inertiaRelief"})
    with pytest.raises(ValueError, match="carried mass"):
        study_from_dict({**FREE_BAR, "inertiaRelief": {"masses": [{"shape": TIP, "mass": 0}]}})
    with pytest.raises(ValueError, match="unknown"):
        study_from_dict({**FREE_BAR, "inertiaRelief": {"mass": 1}})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
