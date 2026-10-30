"""
Tests for the manufacturing constraints (manufacture.py): milling and
casting access, the cutter radius and 2.5-axis floors, circular patterns and
the maximum member. Small grids only.

Run: `python3 tests/test_manufacture.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loadpath.domain import build_domain
from loadpath.manufacture import (AXIS_DIRECTIONS, Access, AccessFilter, MaxMember, machinable,
                                  member_thickness, tie_groups)
from loadpath.simp import SimpOptions, filter_radius, optimize
from loadpath.spec import study_from_dict

# A short cantilever, loaded down at the tip: the free design is an I-beam
# like truss with closed holes through y, which casting along z has to fill.
BEAM = {
    "designSpace": {"type": "box", "min": [0, -8, -8], "max": [48, 8, 8]},
    "supports": [{"shape": {"type": "box", "min": [-1, -9, -9], "max": [0.01, 9, 9]}}],
    "loads": [{"shape": {"type": "box", "min": [46, -2, -9], "max": [48.1, 2, 9]}, "force": [0, 0, -100]}],
    "material": "pa12",
    "goal": {"volumeFraction": 0.3},
    "minMember": 2.0,
    "resolution": {"pitch": 2.0},
}


def _run(extra, iters=40):
    s = study_from_dict({**BEAM, **extra})
    d = build_domain(s)
    res = optimize(d, 0.39, SimpOptions(volume_fraction=0.3, filter_radius=filter_radius(d, s.min_member),
                                        max_iterations=iters, access=s.access, pattern=s.pattern,
                                        max_member=s.max_member))
    return s, d, res


def test_access_filter_gradient():
    rng = np.random.default_rng(0)
    dims = (5, 6, 7)
    f = rng.random(dims)
    for dirs in ([(2, 1)], [(2, 1), (2, -1)], AXIS_DIRECTIONS):
        a = AccessFilter(dims, dirs)
        out = a.forward(f)
        assert (out >= f - 1e-12).all()   # access only ever fills
        w = rng.random(dims)
        g = a.backward(w)
        for _ in range(10):
            i = tuple(int(rng.integers(0, n)) for n in dims)
            fp = f.copy()
            fp[i] += 1e-6
            num = ((a.forward(fp) * w).sum() - (out * w).sum()) / 1e-6
            a.forward(f)
            assert abs(num - g[i]) < 1e-5, (dirs, i, num, g[i])


def test_machinable_access_tool_and_floors():
    s = np.ones((10, 10, 10), bool)
    s[3:7, 3:7, 2:5] = False          # sealed void
    s[3:7, 3:7, 7:] = False           # pocket open to +z
    top = Access("milling", [(2, 1)], axes=3)
    b = machinable(s, top, 1.0)
    assert b[3:7, 3:7, 2:5].all() and not b[3:7, 3:7, 7:].any()
    s2 = np.ones((10, 10, 10), bool)
    s2[3:7, 3:7, :3] = False          # pocket open to -z
    assert machinable(s2, top, 1.0)[3:7, 3:7, :3].all()
    two = Access("casting", [(2, 1), (2, -1)])
    assert not machinable(s2, two, 1.0)[3:7, 3:7, :3].any()
    # a 4 mm cutter cannot enter a 2 mm slot, and does a 5 mm one
    cutter = Access("milling", [(2, 1)], axes=3, tool_radius=2.0)
    s3 = np.ones((14, 14, 6), bool)
    s3[5:7, :, 3:] = False
    assert machinable(s3, cutter, 1.0)[5:7, 4:10, 3:].all()
    s4 = np.ones((14, 14, 6), bool)
    s4[4:9, :, 3:] = False
    assert not machinable(s4, cutter, 1.0)[4:9, 4:10, 3:].any()
    # 2.5-axis: a 5 mm pocket comes out 4 mm deep on 2 mm floors
    s5 = np.ones((10, 10, 10), bool)
    s5[2:8, 2:8, 5:] = False
    b5 = machinable(s5, Access("milling", [(2, 1)], axes=2.5, step=2.0), 1.0)
    assert (~b5[4, 4]).sum() == 4


def test_spec_and_errors():
    s = study_from_dict({**BEAM, "milling": {"axes": 5, "toolDiameter": 6}})
    assert len(s.access.directions) == 6 and s.access.tool_radius == 3.0
    s = study_from_dict({**BEAM, "casting": "+y"})
    assert s.access.directions == [(1, 1), (1, -1)]
    s = study_from_dict({**BEAM, "milling": {"axes": 2.5}})
    assert s.access.step == 2.0
    for bad, word in (({"milling": {"axes": 4}}, "axes"),
                      ({"casting": {"draw": "+z", "sides": 3}}, "sides"),
                      ({"casting": "+z", "milling": {}}, "not both"),
                      ({"casting": "+z", "printDirection": "+z"}, "overhang"),
                      ({"maxMember": 3.0}, "twice minMember"),
                      ({"pattern": {"axis": "z", "count": 1}}, "count"),
                      ({"milling": {"directions": [[0, 0.5, 1]]}}, "axis directions")):
        try:
            study_from_dict({**BEAM, **bad})
        except ValueError as e:
            assert word in str(e), (bad, e)
        else:
            raise AssertionError(f"{bad} was accepted")


def test_casting_leaves_no_undercut():
    _, d, free = _run({})
    _, dc, cast = _run({"casting": {"draw": "+z", "sides": 2}})
    zz = Access("casting", [(2, 1), (2, -1)])
    g_free = d.full_grid(free.xphys) > 0.5
    g_cast = dc.full_grid(cast.xphys) > 0.5
    assert (machinable(g_free, zz, d.pitch) & ~g_free).sum() > 0   # the free design has holes through y
    assert (machinable(g_cast, zz, dc.pitch) & ~g_cast).sum() == 0
    # and still a working beam (on this coarse grid the free run can land
    # in a worse layout than the constrained one, so no ordering)
    assert cast.compliance < 3.0 * free.compliance


def test_milling_with_cutter_from_top():
    _, d, res = _run({"milling": {"axes": 3, "directions": ["+z"], "toolDiameter": 4}})
    g = d.full_grid(res.xphys) > 0.5
    exact = Access("milling", [(2, 1)], axes=3, tool_radius=2.0)
    assert (machinable(g, exact, d.pitch) & ~g).sum() == 0


def test_circular_pattern_ties_sectors():
    disk = {"designSpace": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 4, "r": 20},
            "supports": [{"shape": {"type": "cylinder", "center": [0, 0, 0], "axis": "z", "length": 6, "r": 5}}],
            "loads": [{"shape": {"type": "box", "min": [17, -3, -3], "max": [21, 3, 3]}, "force": [0, 30, 0]}],
            "material": "pa12", "goal": {"volumeFraction": 0.3}, "resolution": {"pitch": 1.0},
            "pattern": {"type": "circular", "axis": "z", "count": 4}}
    s = study_from_dict(disk)
    d = build_domain(s)
    lab = tie_groups(d, None, s.pattern)
    # about a quarter as many groups as cells, and 90 degree turns map onto each other
    groups = len(np.unique(lab))
    assert 0.2 * d.n_elem < groups < 0.35 * d.n_elem, (groups, d.n_elem)
    res = optimize(d, 0.39, SimpOptions(volume_fraction=0.3, filter_radius=filter_radius(d, None),
                                        max_iterations=25, pattern=s.pattern))
    g = d.full_grid(res.x)
    rot = np.rot90(g, 1, axes=(0, 1))
    act = d.full_grid(np.ones(d.n_elem)) > 0
    both = act & np.rot90(act, 1, axes=(0, 1)) & ~(d.full_grid(d.preserve.astype(float)) > 0.5)
    assert np.abs(g - rot)[both].mean() < 0.05


def test_max_member_thins_the_beam():
    # half full, the free beam's flanges run 8 mm thick
    out = []
    for extra in ({}, {"maxMember": 5.0}):
        s = study_from_dict({**BEAM, "resolution": {"pitch": 1.5}, **extra})
        d = build_domain(s)
        res = optimize(d, 0.39, SimpOptions(volume_fraction=0.5, filter_radius=filter_radius(d, 2.0),
                                            max_iterations=80, max_member=s.max_member))
        pres = d.full_grid(d.preserve.astype(float)) > 0.5
        mm = MaxMember(d, 5.0, ~d.preserve)
        out.append((member_thickness(d.full_grid(res.xphys) > 0.5, pres, d.pitch), mm(res.xphys)[1], res.compliance))
    (t_free, o_free, c_free), (t_lim, o_lim, c_lim) = out
    assert o_lim < 0.5 * o_free, out
    assert t_lim < t_free, out
    assert c_lim < 1.3 * c_free, out


TESTS = [
    test_access_filter_gradient,
    test_machinable_access_tool_and_floors,
    test_spec_and_errors,
    test_casting_leaves_no_undercut,
    test_milling_with_cutter_from_top,
    test_circular_pattern_ties_sectors,
    test_max_member_thins_the_beam,
]


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
