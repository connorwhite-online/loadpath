"""
Tests for the length-scale controls and escape holes: the default member
size, the minimum hole (closing at delivery), and escape.py's drilling into
voids the optimizer sealed. Small grids only.

Run: `python3 tests/test_escape.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.domain import build_domain
from loadpath.escape import EscapeOptions, drill, sealed_voids
from loadpath.schema import lint
from loadpath.simp import SimpOptions, filter_radius, optimize
from loadpath.spec import DEFAULT_MIN_MEMBER, study_from_dict

# a square tube's design box round a keep-out channel, held at one end and
# loaded at the other: the stiff answer is a closed tube, sealed at its ends
TUBE = {
    "designSpace": [{"type": "box", "min": [0, -15, -15], "max": [100, 15, 15]}],
    "keepOut": [{"id": "ch", "shape": {"type": "cylinder", "p0": [15, 0, 0], "p1": [85, 0, 0], "r": 5}}],
    "preserve": [{"id": "end", "shape": {"type": "box", "min": [0, -15, -15], "max": [3, 15, 15]}},
                 {"id": "tip", "shape": {"type": "box", "min": [97, -15, -15], "max": [100, 15, 15]}}],
    "supports": [{"region": "end"}],
    "loads": [{"region": "tip", "force": [0, -10, 0]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.3},
    "symmetry": ["z"],
    "resolution": {"pitch": 1.0},
}


def _shell(d, t=3.0):
    c = d.mesh.elem_centers()
    return ((np.abs(c[:, 1]) > 15 - t) | (np.abs(c[:, 2]) > 15 - t) | (c[:, 0] < 2 + t) | (c[:, 0] > 98 - t))


def _sealed_after(d, xp):
    g = np.zeros(tuple(np.asarray(d.dims) + 2), bool)
    g[tuple((d.mesh.elem_ijk + 1).T)] = xp > 0.5
    return sealed_voids(g)[1]


def test_spec_defaults():
    s = study_from_dict(TUBE)
    assert s.min_member is None and s.length_scale == DEFAULT_MIN_MEMBER
    assert s.min_hole is None and s.drains is None
    s = study_from_dict({**TUBE, "process": "sls"})
    assert s.min_member == 1.6 and s.min_hole == 2.0
    assert s.drains.diameter == 6.0 and s.drains.count == 2 and s.drains.max_path == 60.0
    assert study_from_dict({**TUBE, "process": "mjf"}).drains.diameter == 7.5
    assert study_from_dict({**TUBE, "process": "fdm"}).drains is None
    assert study_from_dict({**TUBE, "process": "sls", "drains": False}).drains is None
    s = study_from_dict({**TUBE, "minHole": 3, "drains": {"diameter": 8, "maxPath": None}})
    assert s.min_hole == 3.0 and s.drains.diameter == 8.0 and s.drains.max_path is None
    d = build_domain(s)
    assert abs(filter_radius(d, s.length_scale, s.min_hole) - 3.9) < 1e-9
    for bad in ({"minHole": 0}, {"drains": {"diameter": -1}}, {"drains": {"size": 3}}):
        try:
            study_from_dict({**TUBE, **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")
    assert not lint({**TUBE, "minHole": 2, "drains": {"count": 3, "maxPath": 50}}, "study"), "schema rejects"


def test_drill_opens_a_sealed_tube():
    s = study_from_dict({**TUBE, "process": "sls"})
    d = build_domain(s)
    xp = _shell(d).astype(float)
    assert _sealed_after(d, xp) == 1
    rep = drill(d, xp, np.ones(len(xp)), s.drains)
    assert rep["sealed"] == 1 and rep["unopened"] == 0 and len(rep["holes"]) >= 2
    assert not (rep["void"] & d.preserve).any(), "a hole cut preserved geometry"
    assert rep["maxPath"] is not None and rep["maxPath"] <= 60.0
    for h in rep["holes"]:
        assert h["wallMm"] <= 3.5, h               # through the 3 mm wall, not along it
    opened = xp.copy()
    opened[rep["void"]] = 0.0
    assert _sealed_after(d, opened) == 0
    # holes come in mirror pairs across the z plane (or sit on it)
    zs = sorted(round(h["site"][2], 3) for h in rep["holes"])
    assert all(abs(z + w) < 1e-6 for z, w in zip(zs, zs[::-1])), zs


def test_long_void_gets_more_holes():
    s = study_from_dict({**TUBE, "process": "sls"})
    d = build_domain(s)
    xp = _shell(d).astype(float)
    few = drill(d, xp, None, EscapeOptions(diameter=6.0, count=1, max_path=None))
    near = drill(d, xp, None, EscapeOptions(diameter=6.0, count=1, max_path=25.0))
    assert len(near["holes"]) > len(few["holes"])
    assert near["maxPath"] <= 25.0 + 2.0 * d.pitch, near["maxPath"]


def test_box_with_pinholes_still_traps_powder():
    s = study_from_dict({**TUBE, "process": "sls"})
    d = build_domain(s)
    c = d.mesh.elem_centers()
    xp = _shell(d).astype(float)
    pin = (np.abs(c[:, 0] - 50) < 1.0) & (np.abs(c[:, 2]) < 1.0) & (c[:, 1] > 10)   # a 2 mm hole on top
    xp[pin] = 0.0
    assert _sealed_after(d, xp) == 0                 # not sealed...
    rep = drill(d, xp, None, s.drains)
    assert rep["sealed"] == 1 and rep["holes"], rep  # ...but a 6 mm ball cannot get out


def test_deep_open_tube_gets_holes():
    s = study_from_dict({**TUBE, "process": "sls"})
    d = build_domain(s)
    c = d.mesh.elem_centers()
    xp = _shell(d).astype(float)
    xp[c[:, 0] > 94] = 0.0                              # open at the tip end only
    xp[d.preserve & (c[:, 0] > 94)] = 1.0
    xp[(c[:, 0] > 94) & (np.abs(c[:, 1]) < 9) & (np.abs(c[:, 2]) < 9)] = 0.0
    rep = drill(d, xp, None, s.drains)
    assert rep["sealed"] == 0 and rep["deep"] >= 1 and rep["holes"], rep
    assert rep["maxPath"] <= 60.0, rep["maxPath"]


def test_small_void_is_filled():
    s = study_from_dict({**TUBE, "keepOut": [], "process": "sls"})
    d = build_domain(s)
    c = d.mesh.elem_centers()
    xp = np.ones(len(c))
    pocket = (np.abs(c[:, 0] - 50) < 2) & (np.abs(c[:, 1]) < 2) & (np.abs(c[:, 2]) < 2)
    xp[pocket] = 0.0
    rep = drill(d, xp, None, s.drains)
    assert rep["sealed"] == 1 and not rep["holes"] and rep["fill"].sum() == pocket.sum()


def test_closing_only_adds_and_escape_runs():
    base = {**TUBE, "resolution": {"pitch": 2.5}}
    s = study_from_dict(base)
    d = build_domain(s)
    r = filter_radius(d, s.length_scale, 2.0)
    kw = dict(volume_fraction=0.25, filter_radius=r, max_iterations=30)
    plain = optimize(d, 0.39, SimpOptions(**kw))
    closed = optimize(d, 0.39, SimpOptions(dilate=0.15, **kw))
    assert (closed.xphys >= plain.xphys - 1e-9).all()
    assert closed.volume_fraction >= plain.volume_fraction
    esc = optimize(d, 0.39, SimpOptions(escape=EscapeOptions(iterations=8), **kw))
    assert esc.escape is not None and esc.escape["left"] == 0, esc.escape
    if esc.escape["holes"]:
        assert esc.escape["rounds"] >= 1
    assert _sealed_after(d, esc.xphys) == 0


TESTS = [test_spec_defaults, test_drill_opens_a_sealed_tube, test_long_void_gets_more_holes,
         test_box_with_pinholes_still_traps_powder, test_deep_open_tube_gets_holes,
         test_small_void_is_filled, test_closing_only_adds_and_escape_runs]


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
