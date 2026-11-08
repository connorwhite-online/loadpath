"""
Touch zones (loadpath/touchzones.py, feel `touch` targets): fingertip
stiffness per zone of a fixed exterior, contrast between zones, and the
skin under a pliable zone chosen down to the process's minimum wall.

Run: `python3 tests/test_touch_zones.py`
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import feel, touchzones  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.switch import _thin_skin  # noqa: E402

PAD = {"type": "cylinder", "p0": [0, 0, 5], "p1": [0, 0, 9], "r": 7}


def lid(**over):
    """A 32 x 24 x 8 mm TPU lid: a 1.5 mm skin open underneath, held on its
    bottom edge; a pliable pad in the middle of the top, a firm rim."""
    rim = {"type": "subtract", "shapes": [{"type": "box", "min": [-20, -20, 6], "max": [20, 20, 9]},
                                          {"type": "cylinder", "p0": [0, 0, 5], "p1": [0, 0, 9], "r": 9}]}
    doc = {
        "name": "lid",
        "designSpace": {"type": "box", "min": [-16, -12, 0], "max": [16, 12, 8]},
        "skin": {"thickness": 1.5, "open": ["-z"]},
        "material": "tpu88a",
        "process": {"type": "sls"},
        "minMember": 2.0,
        "goal": {"volumeFraction": 0.3},
        "resolution": {"pitch": 1.0},
        "feel": {"targets": [
            {"kind": "touch", "name": "pad", "zone": PAD, "feel": "pliable", "spacing": 5},
            {"kind": "touch", "name": "rim", "zone": rim, "feel": "firm", "spacing": 8}],
            "contrast": [{"softer": "pad", "firmer": "rim", "ratio": 3}],
            "iterations": 30},
    }
    doc["feel"].update(over)
    return doc


def test_lints_and_windows():
    assert not lint(lid(), "study"), lint(lid(), "study")
    bad = lid(targets=[{"kind": "touch", "name": "pad", "zone": PAD, "feel": "squishy"}])
    assert lint(bad, "study")
    assert touchzones.touch_window({"feel": "pliable"}, "t") == touchzones.TOUCH_FEELS["pliable"]
    # at least 5 N to push it 0.25 mm: a floor of 20 N/mm, no ceiling
    assert touchzones.touch_window({"force": 5, "travel": {"max": 0.25}}, "t") == (20.0, None)
    # explicit numbers override the preset's side they give
    assert touchzones.touch_window({"feel": "pliable", "stiffness": {"max": 2.5}}, "t") == (1.0, 2.5)


def test_presses_are_the_same_on_every_grid():
    """Points come from the shapes: a coarse-to-fine run and the read-back
    on the delivered mesh read the same presses."""
    st = study_from_dict(lid())
    names = []
    for pitch in (1.0, 0.5):
        ts = feel.feel_from_study(st, build_domain(st, pitch))
        names.append([t.name for t in ts])
        pad = [t for t in ts if t.params["zone"] == "pad"]
        # inside the 7 mm disc by half a fingertip: the centre and a ring of 4
        assert len(pad) == 5, [t.name for t in pad]
        xyz = build_domain(st, pitch).mesh.node_xyz
        for t in pad:
            assert np.allclose(xyz[t.load, 2], 8.0)            # on the top, where the finger is
            assert t.ground.any() and np.allclose(xyz[t.ground, 2], 0.0)
    assert names[0] == names[1]


def test_contrast_sensitivities_match_finite_differences():
    st = study_from_dict(lid())
    dom, floor = _thin_skin(st, build_domain(st, 1.0))
    prob = feel.FeelProblem(st, dom, st.elastic(design=True), float(st.material.E))
    assert len(prob.contrasts) == 1
    rng = np.random.default_rng(7)
    x = np.clip(0.3 + 0.5 * rng.random(dom.n_elem), 0, 1)
    x[dom.preserve] = 1.0
    x = np.maximum(x, floor)
    base = prob.evaluate(x)
    assert len(base) == prob.m
    con = [i for i, b in enumerate(base) if b[2].startswith("contrast")]
    assert len(con) == 1
    v, g, _ = base[con[0]]
    for e in rng.choice(np.flatnonzero(~dom.preserve & (floor == 0)), 3, replace=False):
        h = 1e-6
        xp = x.copy()
        xp[e] += h
        fd = (prob.evaluate(xp)[con[0]][0] - v) / h
        assert abs(fd - g[e]) <= 1e-3 * max(abs(fd), 1e-3 * np.abs(g).max()), (e, fd, g[e])
    # the robust form: one contrast too, on the filtered design
    rob = prob.evaluate(None, {"xt": x, "beta": 4.0, "floor": floor, "preserve": dom.preserve})
    assert len(rob) == prob.m and sum(r[2].startswith("contrast") for r in rob) == 1


def test_auto_skin_thins_a_pliable_zone_and_flags_the_minimum_wall():
    # a 1.5 mm skin is too stiff for a 1-4 N/mm pad: thinned, never under
    # the TPU card's 1.0 mm minimum wall (Materialize lists 0.8; the card
    # keeps a margin)
    st = study_from_dict(lid(targets=[{"kind": "touch", "name": "pad", "zone": PAD, "feel": "pliable",
                                       "skinThickness": "auto"}]))
    k_bare = touchzones._bare_stiffness(st, "pad", 1.0).max()
    st2, rows = touchzones.plan_touch_skins(st, 1.0)
    (r,) = rows
    assert st.process.min_feature == 1.0
    assert 1.0 <= r["thickness"] < 1.5, r
    assert st2.raw["skin"]["zones"][0]["thickness"] == r["thickness"]
    k_new = touchzones._bare_stiffness(st2, "pad", 1.0).max()
    assert k_new < k_bare and (r.get("belowMinWall") or k_new <= touchzones.AUTO_MARGIN * 4.0 * 1.001), (k_new, r)
    # a ceiling the thinnest printable skin cannot get under: flagged with
    # the thickness it would need, and the optimizer works to the skin
    st = study_from_dict(lid(targets=[{"kind": "touch", "name": "pad", "zone": PAD, "stiffness": [0.1, 0.4],
                                       "skinThickness": "auto"}], contrast=[]))
    st3, (r,) = touchzones.plan_touch_skins(st, 1.0)
    assert r["thickness"] == 1.0 and r["belowMinWall"] and r["needs"] < 1.0, r
    assert "minimum wall" in r["note"] and r["workingCeiling"] > 0.4
    ts = feel.feel_from_study(st3, build_domain(st3, 1.0))
    assert all(t.kmax == 0.4 and t.params["workingMax"] == r["workingCeiling"] for t in ts)
    # a thickness given under the minimum wall is flagged too
    st = study_from_dict(lid(targets=[{"kind": "touch", "name": "pad", "zone": PAD, "feel": "pliable",
                                       "skinThickness": 0.8}], contrast=[]))
    _, (r,) = touchzones.plan_touch_skins(st, 1.0)
    assert r["belowMinWall"] and r["thickness"] == 0.8
    # a study that sets its own minimum wall is taken at its word
    d = lid(targets=[{"kind": "touch", "name": "pad", "zone": PAD, "feel": "pliable", "skinThickness": 0.8}],
            contrast=[])
    d["process"] = {"type": "sls", "minFeature": 0.8}
    _, (r,) = touchzones.plan_touch_skins(study_from_dict(d), 1.0)
    assert not r.get("belowMinWall"), r


def test_run_meets_zones_and_contrast():
    """Coarse and short: the pad stays pliable, the rim gets firm, the
    contrast holds on the design."""
    doc = lid(robust=False)
    pad, rim = doc["feel"]["targets"]
    del pad["feel"]
    pad.update(stiffness=[2, 14], skinThickness="auto")
    rim.update(stiffness={"min": 80})       # the bare skin's rim is 46-50: ribs
    st = study_from_dict(doc)
    with tempfile.TemporaryDirectory() as out:
        rep = feel.run_feel(st, out, pitches=[1.0], iterations=[40], modal=False)
    assert 1.0 <= rep["touchSkin"][0]["thickness"] < 1.5, rep["touchSkin"]
    zs = rep["touchZones"]["design"]
    assert zs["zones"]["pad"]["max"] <= 14 * 1.02, zs
    assert zs["zones"]["rim"]["min"] >= 80 * 0.98, zs
    assert zs["contrast"][0]["ok"], zs
    # read on the delivered mesh too, held on the walls' bottom edge (not
    # under the pad: the outermost node of a column through the cavity)
    dz = rep["touchZones"]["delivered"]["zones"]
    assert dz["pad"]["max"] < 30 and dz["rim"]["min"] > 40, dz


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
