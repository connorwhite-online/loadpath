"""
External flow round a part: drag on a sphere against Schiller-Naumann, the
stream turned to any direction, and the surface forces as a study load case.

Run: `python tests/test_aero.py` (or pytest). About two minutes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.aero import external_flow
from loadpath.spec import study_from_dict

D = 20.0
NU_AIR = 1.85e-5 / 1.18


@pytest.fixture(scope="module")
def sphere_y():
    # Re 100 on the diameter, the stream along +y
    s = trimesh.creation.icosphere(subdivisions=4, radius=D / 2)
    s.apply_translation([5.0, 7.0, -3.0])
    v = 100 * NU_AIR / (D * 1e-3)
    return external_flow(s, v, (0, 1, 0), cells=10, upstream=2.0, downstream=4.0, side=2.0,
                         flow_throughs=4.0, average=2.5), v


def test_sphere_drag_reads_near_and_above_schiller_naumann(sphere_y):
    r, _v = sphere_y
    ref = 24 / 100 * (1 + 0.15 * 100 ** 0.687)                  # 1.09
    assert 1.0 <= r["dragCoefficient"] / ref < 1.35, r["dragCoefficient"]
    # the force is along the stream, in the part's frame
    F = np.array(r["force"])
    assert F[1] > 0 and abs(F[0]) < 0.05 * F[1] and abs(F[2]) < 0.05 * F[1]
    assert abs(r["reynolds"] - 100) < 1 and r["reScale"] == 1.0
    # the moment about the centre of a sphere is near zero
    assert np.linalg.norm(r["moment"]) < 0.1 * F[1] * D


def test_surface_forces_go_into_a_study(sphere_y, tmp_path):
    r, _v = sphere_y
    lc = r["load_case"]
    tot = sum(np.array(ld["force"]) for ld in lc["loads"])
    assert np.allclose(tot, r["wallForceTotal"], rtol=1e-6) and abs(tot[1] / r["force"][1] - 1) < 0.01
    # upstream patches are pushed downstream, so every patch box sits on the part
    study = study_from_dict({
        "designSpace": {"type": "box", "min": [-6, -4, -14], "max": [16, 18, 8]},
        "supports": [{"shape": {"type": "box", "min": [-7, -5, -15], "max": [-5.9, 19, 9]}}],
        "loadCases": [lc], "material": "pa12", "goal": {"safetyFactor": 2}})
    assert study.load_cases[0].name.startswith("flow-")
    (tmp_path / "case.json").write_text(json.dumps(lc))


def test_sharp_edged_cube_stays_stable():
    # a cube face-on, the case that rang up and blew before the sponge layers
    m = trimesh.creation.box((20, 20, 20))
    r = external_flow(m, 15.0, (-1, 0, 0), cells=8, upstream=2.0, downstream=4.0, side=2.0,
                      flow_throughs=3.0, average=1.5)
    assert r["reScale"] < 1 and "note" in r
    assert 0.8 < r["dragCoefficient"] < 1.6, r["dragCoefficient"]
    assert r["force"][0] < 0 and abs(r["force"][1]) < 0.05 * abs(r["force"][0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
