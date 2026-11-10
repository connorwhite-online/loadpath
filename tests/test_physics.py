"""
Tests for the physics audit (docs/PHYSICS.md): what breaks a printed part
(failure.py), loads nobody timed (durability.envelope), a force as a
uniform traction, fluids at their own state, and still-air convection
solved from the surface. Small fixtures, about two minutes.
Run: `python3 tests/test_physics.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.spec import study_from_dict  # noqa: E402


def _box(lo, hi):
    b = trimesh.creation.box(extents=np.subtract(hi, lo))
    b.apply_translation(np.add(lo, hi) / 2)
    return b


# ------------------------------------------------------------- failure.py


def test_equivalent_stress_is_tension_compression_aware():
    from loadpath.failure import equivalent_stress
    from loadpath.fem import von_mises

    s = np.array([[10.0, 0, 0, 0, 0, 0], [-10.0, 0, 0, 0, 0, 0], [5.0, 5, 5, 0, 0, 0], [3.0, -2, 1, 4, -1, 2]])
    # at a ratio of 1 it is von Mises
    assert np.allclose(equivalent_stress(s, 1.0), von_mises(s))
    e = equivalent_stress(s, 1.25)
    assert abs(e[0] - 10.0) < 1e-12                 # uniaxial tension: the stress itself
    assert abs(e[1] - 10.0 / 1.25) < 1e-12          # uniaxial compression: over the ratio
    # hydrostatic tension is not free: 3 p (m - 1) / m, where von Mises reads 0
    assert abs(e[2] - 3 * 5.0 * 0.25 / 1.25) < 1e-12 and von_mises(s)[2] < 1e-12
    # scaling the stress scales the measure (it is read against a strength)
    assert np.allclose(equivalent_stress(2.5 * s, 1.25), 2.5 * e)


def test_bond_stress_counts_tension_and_shear_not_compression():
    from loadpath.failure import bond_stress

    n = [0, 0, 1]
    s = np.array([[0, 0, 6.0, 0, 0, 0], [0, 0, -6.0, 0, 0, 0], [0, 0, 0, 0, 4.0, 0], [0, 0, 0, 0, 0, 4.0],
                  [9.0, 9, 0, 5, 0, 0], [0, 0, 3.0, 0, 4.0, 0]])
    eq, tens, tau = bond_stress(s, n)
    assert np.allclose(tens, [6, 0, 0, 0, 0, 3]) and np.allclose(tau, [0, 0, 4, 4, 0, 4])
    r3 = np.sqrt(3.0)
    assert np.allclose(eq, [6, 0, 4 * r3, 4 * r3, 0, np.hypot(3, 4 * r3)])
    # in-plane stress and in-plane shear do not load the bond; a tilted
    # normal reads the same state in its own frame
    c = np.sqrt(0.5)
    eq2, tens2, tau2 = bond_stress(np.array([[0, 0, 6.0, 0, 0, 0]]), [c, 0, c])
    assert abs(tens2[0] - 3.0) < 1e-12 and abs(tau2[0] - 3.0) < 1e-12
    # a measured bond shear strength replaces the default ratio
    assert np.allclose(bond_stress(s[2:3], n, shear_ratio=2.0)[0], 8.0)


def test_materials_and_process_carry_the_ratios():
    from loadpath.failure import bond_shear_ratio, compression_ratio, criterion
    from loadpath.materials import material_from_spec

    assert compression_ratio(material_from_spec("petg"))[0] == 1.2
    assert compression_ratio(material_from_spec("pa12"))[0] == 1.15
    assert compression_ratio(material_from_spec("alsi10mg"))[0] == 1.0
    assert compression_ratio(material_from_spec("tpu"))[0] == 1.0
    m = material_from_spec({"base": "petg", "compressionRatio": 1.4})
    assert compression_ratio(m) == (1.4, "given with the material", "study")
    assert compression_ratio(material_from_spec({"E": 70000, "strength": 240, "density": 2.7}))[0] == 1.0
    try:
        material_from_spec({"base": "petg", "compressionRatio": 0.8})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a compression ratio under 1")
    base = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [10, 10, 40]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [11, 11, 0.1]}}],
            "loadCases": [{"name": "pull", "loads": [{"shape": {"type": "box", "min": [-1, -1, 39.9],
                                                                "max": [11, 11, 41]}, "force": [0, 0, 500]}]}],
            "material": "petg", "goal": {"volumeFraction": 0.5}}
    st = study_from_dict({**base, "process": {"type": "fdm", "interlayer": 0.5, "interlayerShear": 0.25}})
    assert abs(bond_shear_ratio(st.process) - 2.0) < 1e-12
    st = study_from_dict({**base, "process": {"type": "fdm", "interlayer": 0.5}})
    assert abs(bond_shear_ratio(st.process) - np.sqrt(3)) < 1e-12
    c = criterion(st.material, st.process)
    assert c["criterion"] == "pressure-modified von Mises" and c["bond"]["shearQuality"] == "assumed"
    from loadpath.schema import lint
    assert not lint({**base, "process": {"type": "fdm", "interlayerShear": 0.3},
                     "material": {"base": "petg", "compressionRatio": 1.3}})


def _boss(load, build="+z", **extra):
    return study_from_dict({
        "designSpace": {"type": "cylinder", "p0": [0, 0, 0], "p1": [0, 0, 40], "r": 8},
        "supports": [{"shape": {"type": "box", "min": [-9, -9, -1], "max": [9, 9, 0.1]}}],
        "loadCases": [{"name": "c", "loads": [{"shape": {"type": "box", "min": [-9, -9, 39.9], "max": [9, 9, 41]},
                                               **load}]}],
        "material": "petg", "goal": {"volumeFraction": 0.5},
        **({"process": {"type": "fdm", "buildDirection": build}} if build else {}), **extra})


def _boss_mesh():
    m = trimesh.creation.cylinder(radius=8, height=40, sections=96)
    m.apply_translation([0, 0, 20])
    return m


def test_twisted_standing_boss_is_limited_by_bond_shear():
    """Torsion about the build axis loads every layer bond in shear and
    none in tension: the tension-only check passed it at the in-plane
    strength."""
    from loadpath.verify import STRESS_ALLOWANCE, verify_part

    st = _boss({"force": [0, 0, 0], "moment": [0, 0, 3000]})
    rep = verify_part(st, _boss_mesh(), max_cells=20_000, contact_band=4.0, refine=0)
    c = rep["cases"][0]
    tau = 3000 * 8 / (np.pi * 8 ** 4 / 2)          # T r / J at the surface, MPa
    assert abs(c["interlayerShear"] - tau) < 0.15 * tau, (c["interlayerShear"], tau)
    assert c["interlayerTension"] < 0.1 * c["interlayerShear"]
    assert abs(c["interlayerStress"] - np.sqrt(3) * c["interlayerShear"]) < 0.05 * c["interlayerStress"]
    assert c["limitedBy"] == "layer bond"
    want = st.process.interlayer * st.material.strength / (c["interlayerStress"] * STRESS_ALLOWANCE)
    assert abs(rep["safetyFactor"] - want) < 1e-9
    # pure shear is not compression-relieved or tension-penalized much: the
    # in-plane reading stays near von Mises / sqrt(m)
    assert rep["failure"]["bond"]["shearQuality"] == "assumed"


def test_pushed_boss_is_stronger_than_pulled():
    """The same force pulling and pushing: the layers part in tension, and
    a polymer squeezed is stronger than one pulled."""
    from loadpath.verify import verify_part

    mesh = _boss_mesh()
    pull = verify_part(_boss({"force": [0, 0, 800]}), mesh, max_cells=20_000, contact_band=4.0, refine=0)
    push = verify_part(_boss({"force": [0, 0, -800]}), mesh, max_cells=20_000, contact_band=4.0, refine=0)
    assert pull["cases"][0]["limitedBy"] == "layer bond" and push["cases"][0]["limitedBy"] == "in-plane"
    assert push["safetyFactor"] > 2.0 * pull["safetyFactor"]
    # compression reads 1 / 1.2 of the tension's equivalent stress
    r = push["cases"][0]["peakStress"] / pull["cases"][0]["peakStress"]
    assert 0.75 < r < 0.9, r


def test_force_over_a_face_is_a_uniform_traction():
    """A bar pulled on its end face: uniform stress F / A everywhere, also
    by the loaded face's edges, where equal shares per node put 2-4x the
    traction."""
    from loadpath.domain import boundary_conditions
    from loadpath.fem import HexMesh
    from loadpath.verify import verify_part

    st = study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [10, 10, 40]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [11, 11, 0.1]}, "fix": [False, False, True]},
                     {"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.1, 0.1, 0.1]}},
                     {"shape": {"type": "box", "min": [9.9, -1, -1], "max": [11, 0.1, 0.1]}, "fix": [False, True, True]}],
        "loadCases": [{"name": "pull", "loads": [{"shape": {"type": "box", "min": [-1, -1, 39.9],
                                                            "max": [11, 11, 41]}, "force": [0, 0, 500],
                                                  "moment": [0, 0, 400]}]}],
        "material": "alsi10mg", "goal": {"volumeFraction": 0.5}})
    ijk = np.argwhere(np.ones((5, 5, 20), bool))
    hm = HexMesh(ijk, 2.0, np.zeros(3))
    bc = boundary_conditions(st, hm)
    f = bc.loads[0].reshape(-1, 3)
    top = np.abs(hm.node_xyz[:, 2] - 40.0) < 1e-9
    assert abs(f[:, 2].sum() - 500.0) < 1e-9 and np.abs(f[~top]).max() < 1e-12
    xy = hm.node_xyz[top][:, :2]
    on_x, on_y = (xy[:, 0] < 1e-9) | (xy[:, 0] > 10 - 1e-9), (xy[:, 1] < 1e-9) | (xy[:, 1] > 10 - 1e-9)
    fz = f[top][:, 2]
    inner, edge, corner = fz[~on_x & ~on_y], fz[on_x ^ on_y], fz[on_x & on_y]
    assert np.allclose(inner, inner[0]) and np.allclose(edge, inner[0] / 2) and np.allclose(corner, inner[0] / 4)
    # the couple is the moment asked for and carries no net force
    r = hm.node_xyz - np.array([5.0, 5.0, 40.0])
    mz = (r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]).sum()
    assert abs(mz - 400.0) < 1e-6 and np.abs(f[:, :2].sum(axis=0)).max() < 1e-9

    st2 = study_from_dict({**st.raw, "loadCases": [{"name": "pull", "loads": [
        {"shape": {"type": "box", "min": [-1, -1, 39.9], "max": [11, 11, 41]}, "force": [0, 0, 500]}]}]})
    rep = verify_part(st2, _box((0, 0, 0), (10, 10, 40)), max_cells=8000, refine=0, cut_cells=False)
    c = rep["cases"][0]
    # sliding supports and a uniform traction: one stress (F over the
    # voxelized section) to the last node of the loaded face
    assert 4.4 < c["contactPeakStress"] < 5.2, c["contactPeakStress"]
    assert c["contactPeakStress"] < 1.01 * c["globalPeakStress"], (c["contactPeakStress"], c["globalPeakStress"])


# ------------------------------------------------- loads nobody timed


def test_hold_life_and_temperature_ceiling_invert_the_laws():
    from loadpath.durability import (durability_data, hold_life, rupture_strength, strength_retention,
                                     temperature_ceiling)

    d = durability_data("petg")
    for u in (0.9, 0.6, 0.4):
        h = hold_life(d, u)
        assert abs(rupture_strength(d, 47.0, h) / 47.0 - u) < 1e-9
    assert hold_life(d, 1.2) == 0.0 and hold_life(d, 0.2) is None
    assert hold_life(durability_data("alsi10mg"), 0.9) is None
    for u in (0.9, 0.7, 0.3):
        t = temperature_ceiling("petg", u)
        assert abs(strength_retention("petg", t) - u) < 1e-9
    assert temperature_ceiling("petg", 0.01) == 85.0      # under the curve's last point: its temperature


def test_envelope_reads_what_an_untimed_load_allows():
    from loadpath.delivery import attention
    from loadpath.verify import STRESS_ALLOWANCE, verify_part

    mesh = _box((0, 0, 0), (10, 10, 40))
    base = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [10, 10, 40]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [11, 11, 0.1]}}],
            "material": "petg", "goal": {"volumeFraction": 0.5}}
    load = {"shape": {"type": "box", "min": [-1, -1, 39.9], "max": [11, 11, 41]}, "force": [0, 0, 1500]}
    rep = verify_part(study_from_dict({**base, "loadCases": [{"name": "pull", "loads": [load]}]}), mesh,
                      max_cells=6000, refine=0)
    env = rep["serviceEnvelope"]
    row = env["cases"][0]
    u = rep["cases"][0]["peakStress"] * STRESS_ALLOWANCE / 47.0
    assert abs(row["utilization"] - u) < 1e-9 and 0.3 < u < 0.6
    # a part that passes its static check at SF ~2.5 holds for years but
    # only for hundreds to thousands of load cycles (the PETG curve is assumed)
    assert rep["safetyFactor"] > 1.6
    assert env["heldHours"] > 5000 and 100 < env["cycles"] < 20000
    assert 55 < env["maxTemperature"] < 80
    assert env["undeclared"] == ["pull"] and "no duration" in env["note"]
    assert any("no duration" in a for a in attention({**rep, "passes": True}, {}))
    # a case that says how long and how often is not listed as untimed
    rep2 = verify_part(study_from_dict({**base, "loadCases": [{"name": "pull", "loads": [load], "cycles": 1000,
                                                                "hours": 100}]}), mesh, max_cells=6000, refine=0)
    assert rep2["serviceEnvelope"]["undeclared"] == [] and "note" not in rep2["serviceEnvelope"]
    assert rep2["serviceEnvelope"]["cases"][0]["declared"] == {"hours": 100.0, "cycles": 1000.0}


def test_no_build_direction_still_reads_the_layer_bonds():
    """PETG with no process is still printed in layers: the report says
    what the bonds would hold for layers stacked along each axis."""
    from loadpath.verify import verify_part

    mesh = _boss_mesh()
    rep = verify_part(_boss({"force": [0, 0, 800]}, build=None), mesh, max_cells=20_000, contact_band=4.0, refine=0)
    lay = rep["failure"]["layers"]
    assert lay["checked"] is False and lay["process"] == "fdm"
    sf = {a: v["safetyFactor"] for a, v in lay["ifBuiltAlong"].items()}
    # pulled along z: layers stacked along z part, the others hold
    assert sf["z"] < 0.6 * rep["safetyFactor"] and sf["x"] > 2 * sf["z"] and sf["y"] > 2 * sf["z"]
    assert "along z" in lay["note"]
    # the declared build direction reads the same bond
    dec = verify_part(_boss({"force": [0, 0, 800]}, build="+z"), mesh, max_cells=20_000, contact_band=4.0, refine=0)
    assert "layers" not in dec["failure"]
    assert abs(dec["safetyFactor"] - sf["z"]) < 0.12 * sf["z"], (dec["safetyFactor"], sf["z"])


# ----------------------------------------------------------- fluids


def test_fluids_follow_temperature_and_pressure():
    from loadpath.heatx_thermal import FLUIDS, duty_fluid, fluid_at

    a = fluid_at("air", 25.0)
    assert abs(a.rho - 1.184) < 0.005 and abs(a.mu - 1.84e-5) < 2e-7 and abs(a.k - 0.0261) < 4e-4
    hot = fluid_at("air", 150.0, 2.5)
    assert abs(hot.rho - 2.5e5 / (287.05 * 423.15)) < 1e-9 and abs(hot.mu - 2.38e-5) < 3e-7
    assert abs(hot.cp - 1017) < 6 and abs(hot.k - 0.0352) < 6e-4
    # water: table values at 25, 60 and 90 C
    for t, rho, mu, k in ((25, 997.0, 0.890e-3, 0.607), (60, 983.2, 0.467e-3, 0.654), (90, 965.3, 0.315e-3, 0.675)):
        w = fluid_at("water", t)
        assert abs(w.rho / rho - 1) < 0.002 and abs(w.mu / mu - 1) < 0.03 and abs(w.k / k - 1) < 0.01, (t, w)
    g = fluid_at("glycol50", 90.0)
    assert g.mu < 0.3 * fluid_at("glycol50", 20.0).mu
    assert fluid_at("oil", 60.0) is None
    # a duty's fluid: at its inlet state when it has one; as listed without
    assert duty_fluid({"fluid": "water"}) is FLUIDS["water"]
    assert duty_fluid({"fluid": "water", "tIn": 60}).mu < 0.6 * FLUIDS["water"].mu
    assert abs(duty_fluid({"fluid": "air", "tIn": 20, "pressure": 2.0}).rho / fluid_at("air", 20).rho - 2 / 1.01325) < 1e-9
    own = {"rho": 1.8, "cp": 1013, "k": 0.031, "mu": 2.2e-5}
    assert duty_fluid({"fluid": own, "tIn": 120}).rho == 1.8      # properties given: used as given
    assert duty_fluid({"fluid": "oil", "tIn": 90}) is FLUIDS["oil"]


def test_rating_uses_each_fluid_at_its_mean_temperature():
    from loadpath.heatx_thermal import FLUIDS, fluid_at, rate

    geo = ("gyroid", 8.0, 0.6, 250 * 120 * 100.0, [100.0, 120.0])
    duty = {"air": {"fluid": "air", "kgPerS": 0.08, "tIn": 150, "pressure": 2.5},
            "coolant": {"fluid": "glycol50", "lPerMin": 20, "tIn": 45}, "wall": "alsi10mg"}
    r = rate(*geo, duty, port_diameter=(60.0, 19.0))
    p = r["fluids"]["air"]["properties"]
    mean = 0.5 * (150 + r["tOut"]["air"])
    assert abs(p["rho"] - fluid_at("air", mean, 2.5).rho) < 2e-3 * p["rho"]
    # a volume flow is a mass flow at the inlet's density
    assert abs(r["fluids"]["coolant"]["massFlow"] - 20 / 60000 * fluid_at("glycol50", 45).rho) < 1e-9
    room = rate(*geo, {"air": {"fluid": FLUIDS["air"], "kgPerS": 0.08, "tIn": 150},
                       "coolant": {"fluid": FLUIDS["glycol50"], "lPerMin": 20, "tIn": 45}, "wall": "alsi10mg"},
                port_diameter=(60.0, 19.0))
    # boost air is denser: the same mass flow moves slower and loses far less pressure
    assert r["fluids"]["air"]["pressureDrop"] < 0.65 * room["fluids"]["air"]["pressureDrop"]
    # warm glycol is thinner: higher Re
    assert r["fluids"]["coolant"]["re"] > 1.3 * room["fluids"]["coolant"]["re"]
    # no temperatures: the room-temperature table, as before
    cold = rate(*geo, {"air": {"fluid": "air", "kgPerS": 0.08}, "coolant": {"fluid": "glycol50", "lPerMin": 20},
                       "wall": "alsi10mg"}, port_diameter=(60.0, 19.0))
    assert abs(cold["fluids"]["air"]["re"] - room["fluids"]["air"]["re"]) < 1e-9


# ------------------------------------------------------- convection


def test_natural_h_matches_plate_correlations():
    from loadpath.convection import air_properties, natural_h, radiation_h

    # vertical 50 mm plate 30 K over a 25 C room: Churchill-Chu by hand
    rho, cp, k, mu = air_properties(40.0)
    nu, al = mu / rho, k / (rho * cp)
    ra = 9.80665 / 313.15 * 30 * 0.05 ** 3 / (nu * al)
    nus = (0.825 + 0.387 * ra ** (1 / 6) / (1 + (0.492 / (nu / al)) ** (9 / 16)) ** (8 / 27)) ** 2
    h = natural_h(np.array([55.0]), 25.0, 0.0, 50.0, 12.0, emissivity=0.0)[0]
    assert abs(h - nus * k / 0.05) < 1e-9 and 5.5 < h < 7.5
    # radiation from a polymer near room temperature is about as much again
    hr = radiation_h(55.0, 25.0, 0.9)
    assert 5.5 < hr < 7.0
    assert abs(natural_h(np.array([55.0]), 25.0, 0.0, 50.0, 12.0, emissivity=0.9)[0] - h - hr) < 1e-9
    # hotter surfaces shed more; a hot face up sheds more than one facing down
    hs = natural_h(np.array([30.0, 55.0, 95.0]), 25.0, 0.0, 50.0, 12.0, emissivity=0.0)
    assert hs[0] < hs[1] < hs[2]
    up, down = (natural_h(np.array([55.0]), 25.0, f, 50.0, 12.0, emissivity=0.0)[0] for f in (1.0, -1.0))
    assert up > 1.7 * down
    # a cold face looking down is a hot face looking up
    assert abs(natural_h(np.array([5.0]), 25.0, -1.0, 50.0, 12.0, emissivity=0.0)[0]
               / natural_h(np.array([45.0]), 25.0, 1.0, 50.0, 12.0, emissivity=0.0)[0] - 1) < 0.1
    # closing a gap chokes it; a wide one is an open plate
    gaps = natural_h(np.full(4, 55.0), 25.0, 0.0, 50.0, 12.0, gap=np.array([1.0, 3.0, 8.0, 60.0]), emissivity=0.0)
    assert gaps[0] < 0.02 * h and gaps[0] < gaps[1] < gaps[2] <= gaps[3] and abs(gaps[3] - h) < 1e-9


def test_openness_reads_gaps_and_views():
    from loadpath.convection import openness

    solid = np.zeros((12, 12, 12), bool)
    solid[:, :, 0] = True            # a base plate
    solid[2, :, :] = True            # two fins 3 cells apart
    solid[6, :, :] = True
    cells = np.array([[2, 6, 6], [6, 6, 6], [6, 6, 6], [8, 6, 0]])
    normals = np.array([[1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    gap, view = openness(solid, cells, normals, 2.0)
    assert abs(gap[0] - 2.0 * 3) < 2.0 and abs(gap[1] - 2.0 * 3) < 2.0     # fin to fin
    assert np.isinf(gap[2]) and np.isinf(gap[3])                         # the outer fin face, the open plate top
    assert view[0] < 0.5 and view[1] < 0.5 and view[2] > 0.75 and view[3] > 0.6
    assert np.all((view >= 0) & (view <= 1.0 + 1e-12))


def test_still_air_finds_the_fin_spacing_a_fixed_coefficient_cannot():
    """More fins always help at a fixed h; in still air, fins closer than
    their boundary layers stop helping."""
    from loadpath.verify import verify_part

    def sink(n):
        parts = [_box([0, 0, 0], [40, 40, 4])]
        step = (40 - 1.5) / (n - 1)
        parts += [_box([i * step, 0, 3.99], [i * step + 1.5, 40, 24]) for i in range(n)]
        return trimesh.boolean.union(parts, engine="manifold")

    def study(conv):
        th = {"heat": [{"shape": {"type": "box", "min": [10, 10, -1], "max": [30, 30, 0.6]}, "watts": 1.5}],
              "expansion": False}
        if conv is not None:
            th["convection"] = conv
        return study_from_dict({
            "designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 40, 24]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [2, 41, 0.1]}}],
            "loadCases": [{"name": "c", "loads": [{"shape": {"type": "box", "min": [38, -1, -1], "max": [41, 41, 0.1]},
                                                   "force": [0, 0, -1]}]}],
            "material": "alsi10mg", "goal": {"volumeFraction": 0.5}, "thermal": th})

    def tmax(mesh, conv):
        return verify_part(study(conv), mesh, max_cells=40_000, refine=0)["thermal"]

    few, many = sink(6), sink(14)
    assert tmax(many, 8)["maxTemperature"] < tmax(few, 8)["maxTemperature"] - 3
    a, b = tmax(few, None), tmax(many, None)        # no convection named: still air
    assert a["convection"]["model"] == "natural" and a["convection"]["emissivity"] == 0.25
    assert b["maxTemperature"] > a["maxTemperature"] + 2, (a["maxTemperature"], b["maxTemperature"])
    assert b["convection"]["h"]["mean"] < 0.6 * a["convection"]["h"]["mean"]
    assert b["convection"]["meanView"] < a["convection"]["meanView"]
    # the study fields: a string, an object, a number (no still-air model), and bad ones refused
    assert study("natural").thermal.natural is not None and study(8).thermal.natural is None
    nat = study({"type": "natural", "emissivity": 0.85, "up": "+y"}).thermal.natural
    assert nat.emissivity == 0.85 and nat.up == (0.0, 1.0, 0.0)
    for bad in ("forced", {"type": "natural", "emissivity": 1.5}, {"type": "natural", "fan": 1}):
        try:
            study(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


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
