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
from loadpath.manufacture import (AXIS_DIRECTIONS, Access, AccessFilter, MaxMember, draft_report,
                                  drafted_distance, machinable, machining_report, member_thickness,
                                  parse_casting, tie_groups)
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
                      ({"casting": {"draw": "+z", "draft": 20}}, "draft"),
                      ({"casting": {"draw": "+z", "sides": 1, "parting": 0}}, "parting"),
                      ({"milling": {"rates": {"speed": 3}}}, "rates"),
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


def test_drafted_distance_leans_walls_in():
    # a cylinder r 8, 16 tall: 3 deg of draft off the mid-plane widens it by
    # h tan 3 at the parting plane and not at all at the end faces
    step = 0.5
    z = np.arange(-20, 21) * step
    x = np.arange(-30, 31) * step
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    d = np.maximum(np.sqrt(X ** 2 + Y ** 2) - 8, np.abs(Z) - 8)
    t = np.tan(np.radians(3.0))
    dd = drafted_distance(d, 2, 1, 2, 20, step, 3.0)
    assert (dd <= d + 1e-12).all()                       # only adds
    assert abs(dd[46, 30, 20] + 8 * t) < 0.02            # at r = 8 on the parting plane
    assert abs(dd[46, 30, 36]) < 1e-9                    # at the end face
    one = drafted_distance(d, 2, 1, 1, None, step, 3.0)  # one half pulling +z: widest at the bottom
    assert abs(one[46, 30, 4] + 16 * t) < 0.03


class _Bare:
    preserve = ()
    keep_out = ()


def test_draft_report_reads_facets():
    import trimesh

    box = trimesh.creation.box((20, 20, 10))
    cast = parse_casting({"draw": "+z", "draft": 2.0})
    r = draft_report(_Bare, box, cast, 1.0)
    assert r["underDraftedShare"] > 0.95, r              # straight walls: no draft
    # a frustum leaning in 4 deg both ways from the mid-plane passes
    t = np.tan(np.radians(4.0))
    ring = []
    for zz, rr in ((-5, 10 - 5 * t), (0, 10.0), (5, 10 - 5 * t)):
        for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            ring.append([rr * np.cos(a), rr * np.sin(a), zz])
    hull = trimesh.convex.convex_hull(np.array(ring))
    r = draft_report(_Bare, hull, parse_casting({"draw": "+z", "draft": 2.0, "parting": 0.0}), 1.0)
    assert r["underDraftedArea"] < 1e-6 and r["p1Draft"] > 3.5, r
    r = draft_report(_Bare, hull, parse_casting({"draw": "+z", "draft": 6.0, "parting": 0.0}), 1.0)
    assert r["underDraftedShare"] > 0.95, r


def test_draft_applied_on_delivery():
    from loadpath.reconstruct import reconstruct

    out = []
    for draft in (0.0, 3.0):
        s, d, res = _run({"casting": {"draw": "+z", "sides": 2, "draft": draft}}, iters=30)
        mesh, info = reconstruct(s, d, res.xphys)
        r = draft_report(s, mesh, parse_casting({"draw": "+z", "draft": 3.0}), d.pitch)
        out.append((r["underDraftedShare"], float(mesh.volume)))
    (bare, v0), (drafted, v1) = out
    assert bare > 0.3 and drafted < 0.1 * bare, out
    # draft only adds: 4 deg (3 asked + the build margin) on 2-3 mm members up to 12 mm off
    # the parting plane is a lot of them
    assert v0 < v1 < 1.35 * v0, out


def test_machining_report_reach_and_time():
    import trimesh
    from types import SimpleNamespace

    from loadpath.materials import MATERIALS
    from loadpath.verify import voxelize

    # a 40 x 40 x 20 block with a 20 mm pocket and a 3 mm slot from the top
    block = trimesh.creation.box((40, 40, 20))
    pocket = trimesh.creation.box((20, 20, 10), transform=trimesh.transformations.translation_matrix([0, 0, 5.01]))
    slot = trimesh.creation.box((3, 30, 10), transform=trimesh.transformations.translation_matrix([15, 0, 5.01]))
    part = block.difference(pocket).difference(slot)
    st = SimpleNamespace(material=MATERIALS["alsi10mg"], preserve=(), keep_out=())
    solid, _ = voxelize(part, 1.0, box=(np.full(3, -21.0), np.full(3, 21.0)))
    keep = np.zeros(solid.shape, bool)
    six = Access("milling", [(2, 1)], axes=3, tool_radius=3.0)
    r = machining_report(st, part, solid, six, 1.0, keep, np.full(3, -21.0))
    assert r["family"] == "aluminium" and r["setups"] == 1 and r["directionsUsed"] == ["+z"]
    # the 3 mm slot (900 mm^3) is too narrow for 6 mm, and the pocket keeps r 3 corners
    assert 900 < r["toolUnreachableVolume"] < 1250, r
    assert r["minutes"]["total"] > 0 and r["cost"]["total"] > r["cost"]["setups"]
    two = Access("milling", [(2, 1)], axes=3, tool_radius=1.0)
    r2 = machining_report(st, part, solid, two, 1.0, keep, np.full(3, -21.0))
    assert r2["toolUnreachableVolume"] < 0.2 * r["toolUnreachableVolume"], r2
    assert r2["minutes"]["finishing"] > r["minutes"]["finishing"]   # a smaller cutter takes longer passes



def test_unmachinable_voids_filled_on_delivery():
    """A pocket opening downward cannot be cut from +z; delivery fills it
    (and the sealed sliver round it), leaving the through hole a keep-out."""
    import manifold3d as m3d
    import trimesh

    from loadpath.manufacture import report
    from loadpath.reconstruct import _fill_unmachinable
    from loadpath.shapes import manifold_from_trimesh, trimesh_from_manifold
    from loadpath.sharp import exact_manifold

    s = study_from_dict({**BEAM, "designSpace": {"type": "box", "min": [0, 0, 0], "max": [30, 20, 10]},
                         "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 21, 11]}}],
                         "loads": [{"shape": {"type": "box", "min": [28, -1, -1], "max": [31, 21, 11]},
                                    "force": [0, 0, -100]}],
                         "milling": {"axes": 3, "directions": ["+z"], "toolDiameter": 2}})
    d = build_domain(s)
    block = trimesh.creation.box(extents=[30, 20, 10]); block.apply_translation([15, 10, 5])
    pocket = trimesh.creation.box(extents=[8, 6, 5.2]); pocket.apply_translation([15, 10, 2.5])
    part = manifold_from_trimesh(block) - manifold_from_trimesh(pocket)
    before = report(s, d, trimesh_from_manifold(part))["machining"]["toolUnreachableVolume"]
    env = m3d.Manifold.batch_boolean([exact_manifold(sh) for sh in s.design_space], m3d.OpType.Add)
    out, filled = _fill_unmachinable(part, s, d, env)
    mesh = trimesh_from_manifold(out)
    after = report(s, d, mesh)["machining"]["toolUnreachableVolume"]
    assert before > 150 and filled > 150, (before, filled)
    assert after == 0.0, after
    assert len(out.decompose()) == 1 and mesh.is_watertight
    assert mesh.volume <= 30 * 20 * 10 + 1e-6

TESTS = [
    test_access_filter_gradient,
    test_machinable_access_tool_and_floors,
    test_spec_and_errors,
    test_casting_leaves_no_undercut,
    test_milling_with_cutter_from_top,
    test_circular_pattern_ties_sectors,
    test_max_member_thins_the_beam,
    test_drafted_distance_leans_walls_in,
    test_draft_report_reads_facets,
    test_draft_applied_on_delivery,
    test_machining_report_reach_and_time,
    test_unmachinable_voids_filled_on_delivery,
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
