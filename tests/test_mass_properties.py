"""
Mass properties: mass, centre of mass and inertia on every report, and a
part designed to turn about an axis (massProperties.axis, inertiaWeight,
maxInertia).

Run: `python tests/test_mass_properties.py` (or pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import trimesh

from loadpath.domain import build_domain
from loadpath.simp import SimpOptions, filter_radius, optimize
from loadpath.spec import study_from_dict
from loadpath.study import _margin
from loadpath.verify import mass_properties

# A link that turns about the z axis through its root, carrying a load at
# its tip: material near the axis turns cheaply, far out it costs inertia.
LINK = {
    "designSpace": {"type": "box", "min": [0, -20, -4], "max": [80, 20, 4]},
    "supports": [{"shape": {"type": "box", "min": [-1, -21, -5], "max": [0.01, 21, 5]}}],
    "loadCases": [{"name": "tip", "loads": [
        {"shape": {"type": "box", "min": [77, -3, -5], "max": [81, 3, 5]}, "force": [0, -50, 0]}]}],
    "material": "pa12",
    "goal": {"safetyFactor": 2},
    "resolution": {"pitch": 2.0},
}
AXIS = {"through": [0, 0, 0], "direction": "z"}


def test_mass_properties_of_a_box():
    box = trimesh.creation.box(extents=[40, 20, 10])
    box.apply_translation([30, 0, 0])
    mp = mass_properties(box, 2.0, study_from_dict({**LINK, "massProperties": {"axis": AXIS}}).mass_properties)
    m = 40 * 20 * 10 * 2e-6                                      # kg
    assert abs(mp["mass"] - m * 1e3) < 1e-6
    assert np.allclose(mp["centerOfMass"], [30, 0, 0], atol=1e-6)
    izz = m * (40 ** 2 + 20 ** 2) / 12
    assert abs(mp["inertia"][2][2] - izz) / izz < 1e-5
    # about the z axis through the origin: parallel axis, 30 mm out
    assert abs(mp["axisInertia"] - (izz + m * 30 ** 2)) / izz < 1e-4


def test_inertia_weights_and_spec_errors():
    d = build_domain(study_from_dict({**LINK, "massProperties": {"axis": AXIS, "inertiaWeight": 0.8}}))
    w = d.volume_weights
    assert abs(w[~d.preserve].mean() - 1.0) < 1e-9
    assert w.min() >= 0.2 - 1e-9                           # a cell on the axis costs 1 - w
    assert build_domain(study_from_dict({**LINK, "massProperties": {}})).volume_weights is None
    with pytest.raises(ValueError, match="axis"):
        study_from_dict({**LINK, "massProperties": {"maxInertia": 5}})
    with pytest.raises(ValueError, match="0-1"):
        study_from_dict({**LINK, "massProperties": {"axis": AXIS, "inertiaWeight": 2}})
    # the cap enters the search's margin linearly
    s = study_from_dict({**LINK, "massProperties": {"axis": AXIS, "maxInertia": 10.0}})
    assert abs(_margin(s, {"safetyFactor": 6.0, "cases": [],
                           "massProperties": {"axisInertia": 20.0, "maxInertia": 10.0}}) - 0.5) < 1e-9


def _axis_inertia(d, xphys):
    c = d.mesh.origin + (d.mesh.elem_ijk + 0.5) * d.mesh.pitch
    return float((xphys * (c[:, 0] ** 2 + c[:, 1] ** 2)).sum())


def test_inertia_weight_pulls_material_towards_the_axis():
    # the plain design at 30% against the inertia-weighted one (w 0.5) at a
    # weighted 25%: the same material and stiffness, less inertia (measured
    # 0.295 / 0.295 of the space, compliance 29.8k / 30.0k, inertia -10%)
    out = {}
    for w, vf in ((0.0, 0.3), (0.5, 0.25)):
        s = study_from_dict({**LINK, "massProperties": {"axis": AXIS, "inertiaWeight": w}})
        d = build_domain(s)
        r = optimize(d, 0.39, SimpOptions(volume_fraction=vf, filter_radius=filter_radius(d, None),
                                          max_iterations=50))
        out[w] = (_axis_inertia(d, r.xphys), r.compliance, float(r.xphys[~d.preserve].mean()))
    (i0, c0, v0), (i1, c1, v1) = out[0.0], out[0.5]
    assert abs(v1 - v0) < 0.02 * v0 + 0.005, (v0, v1)
    assert c1 < 1.05 * c0, (c0, c1)
    assert i1 < 0.94 * i0, (i0, i1)


def test_center_of_mass_target_is_held():
    # the link's own design sits on y = 0; asked for y = 6 +- 1 the
    # optimizer moves its material over (at a stiffness cost)
    s = study_from_dict({**LINK, "massProperties": {"centerOfMass": {"target": [None, 6, None], "tolerance": 1}}})
    assert np.isnan(s.mass_properties.com_target[0]) and s.mass_properties.com_target[1] == 6
    d = build_domain(s)
    r = optimize(d, 0.39, SimpOptions(volume_fraction=0.3, filter_radius=filter_radius(d, None),
                                      max_iterations=50))
    c = d.mesh.origin + (d.mesh.elem_ijk + 0.5) * d.mesh.pitch
    com = (r.xphys @ c) / r.xphys.sum()
    assert abs(com[1] - 6) < 1.0, com
    assert r.history[-1]["centerOfMass"][1] == pytest.approx(com[1], abs=0.6)
    # the report checks it, per held axis
    box = trimesh.creation.box(extents=[40, 20, 10])
    box.apply_translation([30, 2, 0])
    mp = mass_properties(box, 1.0, s.mass_properties)
    assert mp["centerOfMassOffset"] == [0.0, -4.0, 0.0] and mp["centerOfMassOk"] is False
    assert mp["centerOfMassTarget"] == [None, 6.0, None]
    with pytest.raises(ValueError, match="centerOfMass"):
        study_from_dict({**LINK, "massProperties": {"centerOfMass": {"target": [None, None, None]}}})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
