"""
Reaction forces: what each support carries, reported on every part and held
to a cap (a bolt's or bearing's rating) like the safety factor.

Run: `python tests/test_reactions.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.spec import study_from_dict
from loadpath.study import _margin
from loadpath.verify import verify_part

L = 100.0
BEAM = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [L, 10, 10]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [2, 11, 0.01]}, "fix": "z"},
                 {"shape": {"type": "box", "min": [L - 2, -1, -1], "max": [L + 1, 11, 0.01]}}],
    "loadCases": [{"name": "quarter", "loads": [{"shape": {"type": "box", "min": [23, -1, 9.99], "max": [27, 11, 11]},
                                                 "force": [0, 0, -100]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
    "goal": {"safetyFactor": 2},
    "resolution": {"pitch": 2.0},
}


def _bar():
    bar = trimesh.creation.box(extents=[L, 10, 10])
    bar.apply_translation([L / 2, 5, 5])
    return bar


@pytest.mark.parametrize("cut", [False, True])
def test_simply_supported_beam_splits_as_statics_says(cut):
    rep = verify_part(study_from_dict(BEAM), _bar(), max_cells=20_000, refine=0, cut_cells=cut)
    r = rep["reactions"]
    assert [x["support"] for x in r] == ["supports[0]", "supports[1]"]
    f0, f1 = (np.array(x["cases"][0]["force"]) for x in r)
    # the supports' forces sum to the load, and split by the lever rule
    # (supports centred at 1 and 99 mm, the load at 25)
    assert np.allclose(f0 + f1, [0, 0, -100], atol=1e-3)
    assert abs(-f0[2] - 100 * 74 / 98) < 2.0 and abs(-f1[2] - 100 * 24 / 98) < 2.0
    assert abs(r[0]["cases"][0]["share"] + r[1]["cases"][0]["share"] - 1) < 1e-9


def test_pins_and_moments():
    # a cantilever held at its root carries the tip load and its moment
    s = study_from_dict({**BEAM, "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
                         "loadCases": [{"name": "tip", "loads": [{"shape": {"type": "box", "min": [99, -1, -1],
                                                                            "max": [101, 11, 11]},
                                                                  "force": [0, 0, -10]}]}]})
    r = verify_part(s, _bar(), max_cells=20_000, refine=0)["reactions"][0]
    assert np.allclose(r["cases"][0]["force"], [0, 0, -10], atol=1e-3)
    # the part pushes down at the tip, 100 mm out: about y, -10 x 100 the other way round
    assert abs(r["maxMoment"] - 10 * L) / (10 * L) < 0.03, r["maxMoment"]
    # a pin through a lug's bore reads its spring force, all of the load
    lug = {"designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
           "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                               "length": 12, "r": 4}}],
           "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}},
                        {"region": "pin", "fix": "pin"}],
           "loadCases": [{"name": "push", "loads": [{"shape": {"type": "box", "min": [20, 14.99, -5], "max": [30, 16, 5]},
                                                     "force": [0, -50, 0]}]}],
           "material": "pa12", "goal": {"safetyFactor": 2}, "resolution": {"pitch": 1.5}}
    body = trimesh.creation.box(extents=[60, 30, 8])
    body.apply_translation([30, 0, 0])
    hole = trimesh.creation.cylinder(radius=4, height=12, sections=48)
    hole.apply_translation([48, 0, 0])
    part = body.difference(hole)
    rr = verify_part(study_from_dict(lug), part, max_cells=30_000, refine=0)["reactions"]
    tot = sum(np.array(x["cases"][0]["force"]) for x in rr)
    assert np.allclose(tot, [0, -50, 0], atol=0.05), tot
    assert rr[1]["maxForce"] > 2.0          # the pin takes a real share


def test_cap_is_a_margin_and_a_warning():
    capped = {**BEAM, "supports": [{**BEAM["supports"][0], "maxForce": 50.0}, BEAM["supports"][1]]}
    s = study_from_dict(capped)
    rep = verify_part(s, _bar(), max_cells=20_000, refine=0)
    r0 = rep["reactions"][0]
    assert r0["forceLimit"] == 50.0 and r0["maxForce"] > 70
    assert any("over its limit" in w for w in rep["warnings"])
    m = _margin(s, {**rep, "safetyFactor": 100.0})
    assert abs(m - 50.0 / r0["maxForce"]) < 1e-9
    with pytest.raises(ValueError, match="positive"):
        study_from_dict({**BEAM, "supports": [{**BEAM["supports"][0], "maxForce": -1}]})
    with pytest.raises(ValueError, match="safetyFactor or maxDeflection"):
        study_from_dict({**capped, "goal": {"volumeFraction": 0.3}})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
