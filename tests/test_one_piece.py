"""
A part is one piece (onepiece.py, 2026-11-24): two halves that each stand on
their own supports and carry their own load to them come out as two parts
under the load cases alone, because every case holds every support at once
and nothing asks one half to hold the other. The handled state (held at one
support group, the part locating the others) is what makes it one part, and
the optimizer holds it when asked (SimpOptions.one_piece).

The part here is a ceiling plate for two lamps, each hook nearer its own
screw: a development part, not one of the held-out set.

Run: `python3 tests/test_one_piece.py` (about a minute).
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath.domain import build_domain  # noqa: E402
from loadpath.onepiece import handled, split_supports, support_groups  # noqa: E402
from loadpath.simp import SimpOptions, filter_radius, optimize  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def _cyl(c, r, length):
    return {"type": "cylinder", "center": c, "axis": "z", "length": length, "r": r}


TWIN = {
    "name": "twin-lamp-plate",
    "designSpace": [{"type": "box", "min": [-50, -15, 0], "max": [50, 15, 8]}],
    "preserve": [{"id": "sL", "shape": _cyl([-42, 0, 4], 5, 8)}, {"id": "sR", "shape": _cyl([42, 0, 4], 5, 8)},
                 {"id": "hL", "shape": _cyl([-30, 0, 4], 4, 8)}, {"id": "hR", "shape": _cyl([30, 0, 4], 4, 8)}],
    "supports": [{"shape": _cyl([-42, 0, 0.5], 5, 1.0), "fix": "xyz"},
                 {"shape": _cyl([42, 0, 0.5], 5, 1.0), "fix": "xyz"}],
    "loadCases": [{"name": "hang", "loads": [{"region": "hL", "force": [0, 0, 40]},
                                             {"region": "hR", "force": [0, 0, 40]}]},
                  {"name": "swing", "loads": [{"region": "hL", "force": [0, 8, 20]},
                                              {"region": "hR", "force": [0, 8, 20]}]}],
    "material": "petg", "goal": {"volumeFraction": 0.12}, "minMember": 3.0,
}


def _domain(raw=TWIN):
    return build_domain(study_from_dict(copy.deepcopy(raw), base_dir="."), pitch=2.0)


def _run(d, one_piece):
    return optimize(d, 0.36, SimpOptions(volume_fraction=0.12, filter_radius=filter_radius(d, 3.0),
                                         max_iterations=120, one_piece=one_piece))


def test_the_supports_are_two_groups_and_the_handled_state_holds_one():
    d = _domain()
    groups = support_groups(d)
    assert len(groups) == 2
    hp = handled(d, 3.0)
    held = hp.domain.fixed.reshape(-1, 3)
    assert held[groups[0]].all() and not held[groups[1]].any(), "held at one group, free at the other"
    for a, f in enumerate(hp.domain.loads):
        net = f.reshape(-1, 3).sum(axis=0)
        assert abs(net[a] - 1.0) < 1e-9 and abs(net).sum() - 1.0 < 1e-9, "a unit push along each axis"
    # a 3 mm bar 84 mm long: L / EA + 2 L^3 / 3EI at unit modulus
    L = 84.0
    want = L / (np.pi * 9 / 4) + 2 * L ** 3 / (3 * np.pi * 81 / 64)
    assert abs(hp.bound / want - 1.0) < 0.05, (hp.bound, want)


def test_a_seat_the_part_presses_on_is_not_an_interface_it_locates():
    # the plate pressed up against the ceiling over its whole back as well:
    # in the handled state the ceiling is not there, so the screws are
    # still two groups to join
    raw = copy.deepcopy(TWIN)
    raw["supports"] = raw["supports"] + [
        {"shape": {"type": "box", "min": [-51, -16, -0.5], "max": [51, 16, 0.3]}, "push": "+z"}]
    d = _domain(raw)
    assert d.seats and len(support_groups(d)) == 2


def test_one_support_group_has_nothing_to_locate():
    raw = copy.deepcopy(TWIN)
    raw["supports"] = [{"shape": {"type": "box", "min": [-51, -16, -1], "max": [51, 16, 0.5]}, "fix": "xyz"}]
    assert handled(_domain(raw), 3.0) is None


def test_halves_on_their_own_screws_come_out_as_one_part_when_held():
    d = _domain()
    loose = _run(d, None)
    assert split_supports(d, loose.xphys) == 2, "the gap this closes: two halves, each on its own screw"
    held = _run(d, 3.0)
    assert split_supports(d, held.xphys) == 1, "held in one piece"
    rec = held.history[-1]
    assert rec["onePiece"] <= 1.0, rec
    assert abs(rec["volume"] - 0.12) < 0.005
    # the tie costs the cases next to nothing (here the joined part is the stiffer)
    assert held.compliance <= 1.05 * loose.compliance, (held.compliance, loose.compliance)


TESTS = [test_the_supports_are_two_groups_and_the_handled_state_holds_one,
         test_a_seat_the_part_presses_on_is_not_an_interface_it_locates, test_one_support_group_has_nothing_to_locate,
         test_halves_on_their_own_screws_come_out_as_one_part_when_held]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
