"""
Tests for the physical calibration loop: coupon meshes, beam-theory
predictions, the fit recovering a known card from synthetic measurements,
and a saved card working as a study's material with its process factors.
Run: `python3 tests/test_calibration.py`.
"""
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from loadpath import calibration as cal  # noqa: E402
from loadpath.materials import material_from_spec  # noqa: E402

_KIT = None


def _kit():
    global _KIT
    if _KIT is None:
        _KIT = cal.coupon_kit("petg")
    return _KIT


def test_kit_meshes_are_printable():
    kit = _kit()
    assert [c.id for c in kit] == ["F-XY", "F-Z", "T-XY", "T-Z", "L-GY"]
    for c in kit:
        assert c.mesh.is_watertight, c.id
        assert len(c.mesh.split()) == 1, c.id
        assert abs(c.mesh.bounds[0]).max() < 1e-6, f"{c.id} not on the bed"
        assert c.mesh.bounds[1].max() <= 180, f"{c.id} does not fit a 180 mm bed"
    by = {c.id: c for c in kit}
    # standing coupons stand: tallest along z
    for cid in ("F-Z", "T-Z"):
        ext = by[cid].mesh.extents
        assert ext[2] == ext.max(), cid
    # the flat and standing coupons are the same solid
    assert abs(by["F-XY"].mesh.volume - by["F-Z"].mesh.volume) < 1e-3 * by["F-XY"].mesh.volume


def test_predictions_follow_beam_theory():
    by = {c.id: c for c in _kit()}
    petg = material_from_spec("petg")
    g = by["F-XY"].geometry
    want = 9.81 * g["span"] ** 3 / (3 * petg.E * g["I"])
    assert abs(by["F-XY"].predicted["mmPerKg"] - want) < 1e-3
    # across the layers is softer, and weaker by the process's interlayer
    assert by["F-Z"].predicted["mmPerKg"] > by["F-XY"].predicted["mmPerKg"]
    assert abs(by["T-Z"].predicted["breakKg"] / by["T-XY"].predicted["breakKg"] - 0.6) < 0.01
    # the lattice's mass is its drawn relative density's
    lat = by["L-GY"]
    rho = lat.predicted["relativeDensity"]
    solid = lat.geometry["solidVolume"]
    est = (solid + rho * lat.geometry["coreVolume"]) * petg.density / 1000
    assert abs(est - lat.predicted["massG"]) / est < 0.08, (est, lat.predicted["massG"])


def _synthetic(E_scale=0.8, s_scale=1.1, inter=0.5, ez=0.9, dens=1.25, lat_factor=0.7, noise=0.0):
    """Measurements a printer whose PETG is E x0.8, strength x1.1, bonds at
    0.5 and E_z at 0.9 would give."""
    rng = np.random.default_rng(1)
    by = {c.id: c for c in _kit()}
    petg = material_from_spec("petg")
    E = petg.E * E_scale
    vols = cal._coupon_volumes()
    m = cal.measurement_template(_kit(), "petg", "fdm", parts={"bracket": {"loadsKg": [0.5, 1, 2]}})
    m["printer"], m["filament"] = "test printer", "test PETG"
    co = m["coupons"]
    for cid, e in (("F-XY", E), ("F-Z", E * ez)):
        mmkg = by[cid].predicted["mmPerKg"] * by[cid].predicted["E"] / e
        co[cid]["points"] = [[kg, mmkg * kg + 0.05 + noise * rng.normal()] for kg, _ in co[cid]["points"]]
        co[cid]["massG"] = vols[cid] * dens / 1000
    for cid, s in (("T-XY", petg.strength * s_scale), ("T-Z", petg.strength * s_scale * inter)):
        co[cid]["breakKg"] = s * cal._bone_area() / 9.81
        co[cid]["breakAt"] = "gauge"
        co[cid]["massG"] = vols[cid] * dens / 1000
    # lattice printed at the drawn density, stiffness lat_factor x the model's
    mat = material_from_spec({"E": E, "strength": 50, "density": dens})
    rho, e_star = cal._lattice_props(mat, factor=lat_factor)
    mmkg = cal._lattice_compliance(e_star) * 9.81
    co["L-GY"]["points"] = [[kg, mmkg * kg] for kg, _ in co["L-GY"]["points"]]
    co["L-GY"]["massG"] = (cal._lattice_solid_volume() + rho * cal._LAT["core"] * cal._LAT["width"]
                           * cal._LAT["height"]) * dens / 1000
    # a part 25% softer than its prediction (made at the starting E)
    pred = {"parts": {"bracket": {"mmPerKg": 0.4, "E": petg.E, "massG": 30.0}}}
    m["parts"]["bracket"]["points"] = [[kg, 0.4 * petg.E / E / 0.75 * kg] for kg in (0.5, 1, 2)]
    return m, pred


def test_fit_recovers_a_known_card():
    m, pred = _synthetic()
    card = cal.fit(m, pred, name="petg-test")
    mat = card["material"]
    petg = material_from_spec("petg")
    assert abs(mat["E"] / (0.8 * petg.E) - 1) < 0.01, mat
    assert abs(mat["strength"] / (1.1 * petg.strength) - 1) < 0.01, mat
    assert abs(mat["density"] - 1.25) < 0.005, mat
    assert abs(card["process"]["interlayer"] - 0.5) < 0.005, card["process"]
    assert abs(card["process"]["stiffnessAcrossLayers"] - 0.9) < 0.01, card["process"]
    assert abs(card["lattice"]["stiffnessFactor"] - 0.7) < 0.03, card["lattice"]
    assert abs(card["modelFactor"]["stiffness"] - 0.75) < 0.01, card["modelFactor"]
    assert card["kept"] == []
    assert abs(card["evidence"]["F-XY"]["offsetMm"] - 0.05) < 1e-6
    # the same part read with a laser 3 m away
    pred["parts"]["bracket"]["laserSpotMmPerKgPerM"] = 2.0
    m["parts"]["bracket"]["laserDistanceM"] = 3.0
    m["parts"]["bracket"]["points"] = [[kg, 6.0 / 0.8 / 0.6 * kg] for kg in (0.5, 1, 2)]
    card = cal.fit(m, pred)
    assert abs(card["parts"]["bracket"]["stiffnessFactor"] - 0.6) < 0.01, card["parts"]
    # the card's material is a valid study material as it stands
    material_from_spec(card["material"])


def test_fit_keeps_defaults_and_flags_bad_breaks():
    m, _ = _synthetic()
    m["coupons"]["T-Z"]["breakKg"] = None
    m["coupons"]["T-XY"]["breakAt"] = "at the hole"
    m["coupons"]["F-Z"]["points"] = [[1.0, None]]
    card = cal.fit(m)
    assert "interlayer" not in card["process"]
    assert any("interlayer" in k for k in card["kept"])
    assert any("stiffnessAcrossLayers" in k for k in card["kept"])
    assert "warning" in card["evidence"]["T-XY"]


def test_saved_card_is_a_study_material():
    from loadpath.spec import study_from_dict
    m, pred = _synthetic()
    card = cal.fit(m, pred, name="petg-unit-test")
    with tempfile.TemporaryDirectory() as d:
        old = cal.CALIBRATION_DIR
        cal.CALIBRATION_DIR = d
        try:
            with open(os.path.join(d, "petg-unit-test.json"), "w") as fh:
                json.dump(card, fh)
            base = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 10, 10]},
                    "supports": [{"shape": {"type": "box", "min": [0, 0, 0], "max": [1, 10, 10]}}],
                    "loads": [{"shape": {"type": "box", "min": [39, 0, 0], "max": [40, 10, 10]},
                               "force": [0, 0, -10]}],
                    "material": "petg-unit-test", "process": {"type": "fdm"}}
            st = study_from_dict(base)
            assert st.material.E == card["material"]["E"]
            assert st.material.processes == material_from_spec("petg").processes
            assert abs(st.process.interlayer - card["process"]["interlayer"]) < 1e-9
            assert abs(st.process.ez - card["process"]["stiffnessAcrossLayers"]) < 1e-9
            # a study's own process value wins
            st2 = study_from_dict({**base, "process": {"type": "fdm", "interlayer": 0.7}})
            assert st2.process.interlayer == 0.7
            assert "petg-unit-test" in cal.list_cards()
        finally:
            cal.CALIBRATION_DIR = old


def test_write_kit_and_cli_fit():
    from loadpath.__main__ import main
    with tempfile.TemporaryDirectory() as d:
        paths = cal.write_kit(_kit(), d)
        for cid in ("F-XY", "F-Z", "T-XY", "T-Z", "L-GY"):
            assert os.path.getsize(paths[cid]) > 1000
        tmpl = json.load(open(paths["measurements"]))
        assert set(tmpl["coupons"]) == {"F-XY", "F-Z", "T-XY", "T-Z", "L-GY"}
        m, _ = _synthetic()
        mp = os.path.join(d, "filled.json")
        json.dump(m, open(mp, "w"))
        rc = main(["calibrate", "fit", mp, "--predictions", paths["predictions"], "--name", "cli-card",
                   "--out", d])
        assert rc == 0
        card = json.load(open(os.path.join(d, "cli-card.json")))
        assert abs(card["material"]["E"] / (0.8 * material_from_spec("petg").E) - 1) < 0.01


def _crush_points(f_plateau, eps_d, ratio, board, core, steps):
    """A load-controlled crush: the block holds below the plateau, runs to
    densification at it. Points are [kg in the bucket, mm the load point
    dropped]."""
    pts = []
    for kg in steps:
        f = (kg + board / 2) * 9.81 * ratio
        strain = 0.01 * f / f_plateau if f < f_plateau else eps_d
        pts.append([kg, round(strain * core * ratio, 2)])
    return pts


def test_crush_fit_recovers_a_known_plateau():
    from loadpath.calibration import _crush_spec, fit
    from loadpath.impact import crush_law, plateau_stress
    from loadpath.materials import material_from_spec

    for material, process in (("pa12", "sls"), ("tpu", "fdm")):
        mat = material_from_spec(material)
        c = _crush_spec(mat, process)
        law, ref = crush_law(c["kind"], mat, calibrated=False)
        area = c["side"] ** 2
        coupons = {}
        for rho, true in zip(c["densities"], (0.8, 0.8 * (c["densities"][1] / c["densities"][0]) ** 0.2)):
            f = true * float(plateau_stress(law, rho, ref)) * area
            ratio, board = c["lever"], (2.0 if c["lever"] > 1 else 0.0)
            kg_pl = f / 9.81 / ratio - board / 2
            steps = np.linspace(0.3 * kg_pl, 1.3 * kg_pl, 41)
            coupons[f"X-{round(rho * 100)}"] = {"leverRatio": ratio, "boardKg": board,
                                                 "points": _crush_points(f, 0.55, ratio, board, c["core"], steps)}
        card = fit({"material": material, "process": process, "coupons": coupons})
        cr = card["crush"]
        assert cr["law"] == ("elastic" if material == "tpu" else "plastic")
        x = cr["coupons"][f"X-{round(c['densities'][0] * 100)}"]
        assert abs(x["factor"] / 0.8 - 1) < 0.03, x
        assert abs(x["densificationStrain"] - 0.55) < 1e-6
        # the exponent: the model's plus the 0.2 put in
        from loadpath.impact import CRUSH
        assert abs(cr["p"] - (CRUSH[law]["p"] + 0.2)) < 0.15, cr


def test_crush_card_scales_impact():
    import tempfile

    from loadpath import calibration
    from loadpath.impact import crush_law
    from loadpath.materials import material_from_spec

    with tempfile.TemporaryDirectory() as d:
        old = calibration.CALIBRATION_DIR
        calibration.CALIBRATION_DIR = d
        try:
            json.dump({"name": "pa12-test", "basedOn": "pa12", "crush": {"law": "plastic", "factor": 0.7}},
                      open(os.path.join(d, "pa12-test.json"), "w"))
            mat = material_from_spec({"name": "pa12-test", "E": 1800, "nu": 0.39, "strength": 48, "density": 1.0})
            assert crush_law("gyroid", mat) == ("gyroid", 48 * 0.7)
            assert crush_law("gyroid", mat, calibrated=False) == ("gyroid", 48.0)
        finally:
            calibration.CALIBRATION_DIR = old


def test_creep_fit_recovers_the_rate_and_the_card_uses_it():
    import tempfile

    from loadpath import calibration
    from loadpath.calibration import creep_coupons, fit
    from loadpath.materials import material_from_spec
    from loadpath.viscoelastic import creep_factor, creep_params

    tpu = material_from_spec("tpu")
    c = creep_coupons("tpu", "fdm")[0]
    assert c.id == "K-25" and creep_coupons("pa12", "sls") == []
    # synthetic week: twice the default creep rate, deflection 0.5 mm at 10 s
    m_true, d10, h0 = 0.09, 0.5, 31.8
    pts = [[mins, round(h0 - d10 * (1 + m_true * np.log(60.0 * mins / 10.0)), 3)]
           for mins in (1, 10, 60, 600, 1440, 4320, 10080)]
    with tempfile.TemporaryDirectory() as d:
        old = calibration.CALIBRATION_DIR
        calibration.CALIBRATION_DIR = d
        try:
            card = fit({"material": "tpu", "process": "fdm",
                        "coupons": {"K-25": {"heightUnloaded": h0, "loadKg": c.measure["loadKg"], "points": pts}}},
                       name="tpu-test")
            assert abs(card["creep"]["m"] - m_true) < 0.03 * m_true, card["creep"]
            json.dump(card, open(os.path.join(d, "tpu-test.json"), "w"))
            mat = material_from_spec({**card["material"], "name": "tpu-test"})
            assert abs(creep_params(mat)[0] - card["creep"]["m"]) < 1e-9
            assert float(creep_factor(mat, 3600.0)) > float(creep_factor(tpu, 3600.0))
        finally:
            calibration.CALIBRATION_DIR = old


def test_rigid_motion_fit():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 20, (60, 3))
    t, w = np.array([0.1, -0.2, 0.05]), np.array([2e-3, -1e-3, 5e-4])
    u = t + np.cross(w, x - x.mean(axis=0))
    t2, w2 = cal.rigid_motion(x, u)
    assert np.allclose(t, t2) and np.allclose(w, w2)


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{'all %d passed' % len(tests) if not failed else '%d of %d failed' % (failed, len(tests))}")
    sys.exit(1 if failed else 0)
