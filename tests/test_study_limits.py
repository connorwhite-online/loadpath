"""
Limits and grids a study is checked against (bugs found on a fan-plate
study): goal.clearances caps travel away from the loads, the analysis grid
is never coarser than the design grid, the confirm grid never coarser than
the analysis grid, thin preserved decks are flagged, and a search whose
picks all fail their confirm delivers nothing as passing.

Run: `python tests/test_study_limits.py` (or pytest).
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

# a cantilever held at x=0, pushed down at its tip: the tip travels most,
# the root least
BAR = {
    "designSpace": {"type": "box", "min": [0, 0, 0], "max": [100, 10, 10]},
    "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.01, 11, 11]}}],
    "loadCases": [{"name": "tip", "loads": [
        {"shape": {"type": "box", "min": [99.0, -1, -1], "max": [101, 11, 11]}, "force": [0, 0, -10]}]}],
    "material": {"E": 2000, "nu": 0.3, "strength": 50, "density": 1.2},
    "goal": {"safetyFactor": 2},
    "resolution": {"pitch": 2.0},
}
MID = {"type": "box", "min": [45, -1, -1], "max": [55, 11, 11]}


def _bar():
    return trimesh.creation.box(extents=[100, 10, 10]).apply_translation([50, 5, 5])


def test_clearance_reads_travel_away_from_the_loads():
    s = study_from_dict({**BAR, "goal": {"safetyFactor": 2, "clearances": [
        {"shape": MID, "maxTravel": 0.05, "name": "middle"},
        {"shape": MID, "maxTravel": 0.05, "direction": "+z", "name": "up"}]}})
    rep = verify_part(s, _bar(), refine=0)
    c = rep["cases"][0]
    mid, up = c["clearances"]
    # the middle moves down, under the tip's deflection
    assert 0 < mid["travel"] < c["deflection"], (mid, c["deflection"])
    assert up["travel"] == 0.0   # nothing moves up
    assert abs(mid["at"][0] - 55) < 2.5   # its far end travels most
    m = _margin(s, rep)
    assert m <= 0.05 / mid["travel"] + 1e-12


def test_clearance_spec_errors():
    with pytest.raises(ValueError, match="safetyFactor or maxDeflection"):
        study_from_dict({**BAR, "goal": {"volumeFraction": 0.3, "clearances": [{"shape": MID, "maxTravel": 1}]}})
    with pytest.raises(ValueError, match="maxTravel"):
        study_from_dict({**BAR, "goal": {"safetyFactor": 2, "clearances": [{"shape": MID}]}})
    with pytest.raises(ValueError, match="not a load case"):
        study_from_dict({**BAR, "goal": {"safetyFactor": 2, "clearances": [
            {"shape": MID, "maxTravel": 1, "cases": ["drop"]}]}})
    with pytest.raises(ValueError, match="unknown region"):
        study_from_dict({**BAR, "goal": {"safetyFactor": 2, "clearances": [{"region": "deck", "maxTravel": 1}]}})


def test_analysis_grid_is_no_coarser_than_the_design_grid():
    s = study_from_dict(BAR)
    # 2000 cells over 10 cm^3 is a 1.71 mm grid; the design's is 1.5
    loose = verify_part(s, _bar(), max_cells=2000, refine=0)
    held = verify_part(s, _bar(), max_cells=2000, refine=0, max_pitch=1.5)
    assert loose["analysisPitch"] > 1.6
    assert held["analysisPitch"] <= 1.5 + 1e-9
    # past the stretch it can't reach the design grid: it says so
    far = verify_part(s, _bar(), max_cells=2000, refine=0, max_pitch=0.8)
    assert far["analysisPitch"] > 0.8 and any("coarser than the 0.80 mm grid" in w for w in far["warnings"])


def test_thin_preserved_deck_is_flagged():
    deck = {"id": "deck", "shape": {"type": "box", "min": [0, 0, 7], "max": [100, 10, 10]}}
    s = study_from_dict({**BAR, "preserve": [deck], "goal": {"maxDeflection": 1.0}})
    rep = verify_part(s, _bar(), refine=0, max_cells=8000)   # 2.5 mm cells: the 3 mm deck is ~1.2
    assert rep["thinPreserves"][0]["id"] == "deck", rep.get("thinPreserves")
    assert abs(rep["thinPreserves"][0]["thickness"] - 3.0) < 0.6
    assert any("preserve 'deck'" in w for w in rep["warnings"])
    # a stress-only goal is not about travel: no flag
    assert "thinPreserves" not in verify_part(study_from_dict({**BAR, "preserve": [deck]}), _bar(), refine=0,
                                              max_cells=8000)


def test_no_fraction_passes_when_every_confirm_fails():
    # every pick passes its analysis grid and fails the finer confirm: the
    # step up used to clamp at the 0.7 ceiling, re-run the same design and
    # deliver its last, unconfirmed copy as passing
    import loadpath.study as ST

    real = ST.verify_part

    def harsh(study, mesh, **kw):
        rep = real(study, mesh, **kw)
        if kw.get("min_pitch") is not None and kw.get("max_cells", 0) >= 100_000:   # the confirm
            rep["safetyFactor"] *= 0.05
            for c in rep["cases"]:
                c["safetyFactor"] = c.get("safetyFactor", 1.0) * 0.05
        return rep

    light = {**BAR, "loadCases": [{"name": "tip", "loads": [{**BAR["loadCases"][0]["loads"][0], "force": [0, 0, -2]}]}]}
    s = study_from_dict({**light, "goal": {"safetyFactor": 2, "confirm": "voxel", "method": "search",
                                           "volumeFraction": 0.5}, "maxIterations": 25})
    ST.verify_part = harsh
    old = ST._confirm_cells
    ST._confirm_cells = lambda st=None: 150_000
    try:
        res = ST._run_single(s)
    finally:
        ST.verify_part = real
        ST._confirm_cells = old
    assert res.best is None, [(o.volume_fraction, o.passes) for o in res.outcomes]
    assert not any(o.passes for o in res.outcomes)
    assert any("no volume fraction passes" in w for w in res.warnings), res.warnings
    # the ceiling's design ran once, not over and over
    top = [o for o in res.outcomes if o.volume_fraction >= ST._VF_MAX - 1e-9]
    assert len(top) <= 1, [o.volume_fraction for o in res.outcomes]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
