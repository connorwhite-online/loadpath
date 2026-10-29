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
