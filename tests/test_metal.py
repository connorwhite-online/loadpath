"""Printing heat exchangers in powder-bed metal: plug shapes, internal
ceilings along a build direction, roofs and clean-out ports."""
import numpy as np

from loadpath.checks.networks import _refill_plugs


def _plug_grid():
    pitch = 0.5
    origin = np.zeros(3)
    solid = np.zeros((80, 80, 80), bool)
    return solid, origin, pitch


def test_flat_plug_stops_square_at_its_inner_end():
    # a 47 mm port's plug: a round end would reach r = 23.5 mm past the mouth
    a, b, r = [20.0, 20.0, 10.0], [20.0, 20.0, 30.0], 8.0
    round_ = _plug_grid()[0]
    _refill_plugs(round_, np.zeros(3), 0.5, [{"a": a, "b": b, "r": r}])
    flat = _plug_grid()[0]
    _refill_plugs(flat, np.zeros(3), 0.5, [{"a": a, "b": b, "r": r, "ends": "flat"}])
    k = np.arange(80) * 0.5
    # nothing of the flat plug below its inner end; the round one bulges r + 1.5 voxels past it
    assert not flat[:, :, k < 10.0 - 1e-9].any()
    assert round_[:, :, k < 10.0 - 4.0].any()
    # the full bore section is still filled, out past the outer end
    ii = int(20.0 / 0.5)
    assert flat[ii, ii, int(10.0 / 0.5)] and flat[ii, ii, int(30.5 / 0.5)]
    assert flat[ii + int(r / 0.5), ii, int(20.0 / 0.5)]


def _block(build=None, **kw):
    from loadpath.exchanger import exchanger_core

    return exchanger_core((40.0, 32.0, 24.0), cell=8.0, wall=1.0, build=build, **kw)


def test_internal_ceilings_see_a_box_plenum_roof_and_not_the_outside():
    import trimesh

    from loadpath.cavities import internal_ceilings, rank_build_directions

    # a closed 20 mm cube cavity in a 30 mm block: its 400 mm^2 roof is the only ceiling inside
    outer = trimesh.creation.box((30, 30, 30))
    inner = trimesh.creation.box((20, 20, 20))
    inner.invert()
    part = trimesh.util.concatenate([outer, inner])
    r = internal_ceilings(part, up="+z", pitch=0.5)
    assert 380 < r["internalFlatArea"] < 420, r
    # along the cube diagonal every face leans 54.7 degrees
    best = rank_build_directions(part, pitch=0.5)[0]
    assert best["internalOverhangArea"] < 1.0 and len(best["label"]) > 3


def test_exchanger_core_build_roofs_its_ceilings_and_stays_isolated():
    from loadpath.cavities import cavity_surface
    from loadpath.checks.networks import check_networks
    from loadpath.sdf_kit import to_mesh

    flat = {}
    for build in (None, "+z", "+y"):
        field, ports, plugs, bounds = _block(build)
        s = cavity_surface(field, bounds, pitch=0.4, plugs=plugs, exclude=field.exclude)
        flat[build] = s.ceilings(build or "+z")["internalFlatArea"]
        if build is not None:
            mesh = to_mesh(field, *bounds, pitch=0.4)
            net = check_networks(mesh, ports, plugs=plugs, min_feature=1.0)
            assert net["isolated"], (build, net)
    assert flat["+z"] < 0.35 * flat[None], flat
    assert flat["+y"] < 0.35 * flat[None], flat


def test_exchanger_core_build_auto_is_refused_with_the_way_to_choose():
    import pytest

    with pytest.raises(ValueError, match="rank_build_directions"):
        _block("auto")


def test_lpbf_process():
    from loadpath.spec import _process

    p = _process("lpbf")
    assert p.overhangs and p.min_feature == 0.4 and p.build is not None


def test_heatx_closure_cuts_internal_ceilings_and_keeps_the_fluids_apart():
    from loadpath.heatx import design_heat_exchanger, heat_exchanger_from_spec

    base = {"core": {"type": "box", "min": [0, 0, 0], "max": [30, 30, 30]}, "layout": "cross", "axes": "xy",
            "cell": 8, "wall": 1.0, "header": 4, "portDiameter": 6, "portLength": 4, "process": "lpbf"}
    flat = {}
    for closure in (False, True):
        hx = heat_exchanger_from_spec({**base, "closure": closure})
        hx.step = 0.45
        _, rep = design_heat_exchanger(hx)
        assert rep["isolated"] and all(rep["connected"].values()), rep["ports"]
        assert rep["minWallBetweenFluids"] > 0.9
        flat[closure] = rep["internalCeilings"]["flatArea"]
    assert not hx.skirt                       # lpbf: outside supports are cut off anyway
    assert flat[True] < 0.4 * flat[False], flat
