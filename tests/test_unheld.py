"""
Loads that cancel and supports that hold nothing (2026-11-24).

The held-out junction-box cover stalled in its optimize stage reading a
3e13 mm deflection on its fasteners' preload case: an equal and opposite
pair of ring forces, net force 1.8e-15 N, and the constrained method read
deflection as compliance over the net force. The deflection at the loads is
the work they do per unit of force, C / sum |f_n|, which is C / |F| for
loads that all point one way and stays the squeeze or the twist for loads
that cancel. Supports that leave a rigid motion free make every solve
singular: refused at the domain, naming the motion, in well under a second.

Run: `pytest tests/test_unheld.py` (about a minute).
"""
import time

import numpy as np
import pytest

from loadpath.constrained import Limits, optimize_constrained
from loadpath.domain import build_domain
from loadpath.simp import SimpOptions, filter_radius
from loadpath.spec import study_from_dict
from loadpath.tasks import check_study

PLATE = {
    "name": "plate",
    "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
    "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                         "length": 8, "r": 9}}],
    "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                        "length": 12, "r": 4}}],
    "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
    "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]},
                  # a pair that cancels, as a fastener's preload: the eye squeezed across
                  {"name": "squeeze", "loads": [
                      {"shape": {"type": "box", "min": [38.5, -9, -5], "max": [40.5, 9, 5]}, "force": [100, 0, 0]},
                      {"shape": {"type": "box", "min": [55.5, -9, -5], "max": [57.5, 9, 5]}, "force": [-100, 0, 0]}]},
                  # a pure torque on the eye: a couple, no net force
                  {"name": "twist", "loads": [{"region": "eye", "moment": [0, 0, 400]}]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.35},
    "resolution": {"pitch": 2.0},
}


def test_loads_that_cancel_read_their_squeeze_not_a_blow_up():
    s = study_from_dict(PLATE)
    d = build_domain(s)
    for name in ("squeeze", "twist"):
        f = d.loads[d.case_names.index(name)].reshape(-1, 3)
        assert np.linalg.norm(f.sum(axis=0)) < 1e-9 * np.abs(f).sum(), name
    res = optimize_constrained(d, s.material.nu,
                               SimpOptions(volume_fraction=0.5, filter_radius=filter_radius(d, None),
                                           max_iterations=3),
                               Limits(None, 0.3, s.material.E))
    first = res.history[0]
    size = 60.0
    for name in ("pull", "squeeze", "twist"):
        v = first[f"deflection.{name}"]
        assert np.isfinite(v) and 0 < v < 0.05 * size, (name, v)
    # loads that all point one way read C / |F|, as before
    f = d.loads[0]
    assert abs(np.linalg.norm(f.reshape(-1, 3), axis=1).sum() - 80.0) < 1e-9


@pytest.mark.parametrize("fix, words", [("z", "slide along x"), ("xy", "slide along z")])
def test_supports_that_leave_a_rigid_motion_free_are_refused_at_once(fix, words):
    spec = dict(PLATE, supports=[{"shape": PLATE["supports"][0]["shape"], "fix": fix}])
    t0 = time.time()
    with pytest.raises(ValueError) as exc:
        build_domain(study_from_dict(spec))
    assert "free to" in str(exc.value) and words in str(exc.value) and "singular" in str(exc.value), exc.value
    assert time.time() - t0 < 30
    c = check_study(spec)
    assert not c["ok"] and "free to" in c["error"], c


def test_a_face_fixed_in_every_direction_holds_the_part():
    build_domain(study_from_dict(PLATE))


if __name__ == "__main__":
    test_loads_that_cancel_read_their_squeeze_not_a_blow_up()
    test_supports_that_leave_a_rigid_motion_free_are_refused_at_once("z", "slide along x")
    test_supports_that_leave_a_rigid_motion_free_are_refused_at_once("xy", "slide along z")
    test_a_face_fixed_in_every_direction_holds_the_part()
    print("ok")
