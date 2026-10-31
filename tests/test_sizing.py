"""Exchanger sizing in an envelope (sizing.py): the search respects the
envelope, the budget and the process limits, and its specs design."""
import pytest

from loadpath.sizing import PROCESS_LIMITS, size_exchanger
from loadpath.tasks import size_heat_exchanger

DUTY = {"A": {"fluid": "water", "lPerMin": 1.0, "tIn": 80}, "B": {"fluid": "water", "lPerMin": 1.0, "tIn": 20},
        "wall": "alsi10mg"}
ENV = [80, 70, 40]


def test_sized_candidates_fit_the_envelope_and_the_budget():
    budget = {"A": 3000, "B": 3000}
    res = size_exchanger(DUTY, ENV, process="lpbf", max_pressure_drop=budget, cells=[5.0, 6.0, 8.0],
                         walls=[0.5, 0.8], fractions=(0.6, 1.0))
    assert res["feasible"] > 0 and res["best"] is not None
    lim = PROCESS_LIMITS["lpbf"]
    for c in res["candidates"]:
        s = c["spec"]
        allow = s["header"] + s["shell"] + s["portLength"]
        core = sorted(c["core"][:2])
        assert core[0] + 2 * allow <= sorted(ENV[:2])[0] + 1e-6 and core[1] + 2 * allow <= sorted(ENV[:2])[1] + 1e-6
        assert c["core"][2] + 2 * s["shell"] <= ENV[2] + 1e-6
        assert c["wall"] >= lim["minWall"]
        assert all(c["pressureDrop"][n] <= budget[n] for n in budget)
        assert 0 < c["effectiveness"] < 1
    # sorted lightest first
    v = [c["solidVolume"] for c in res["candidates"]]
    assert v == sorted(v)


def test_effectiveness_objective_and_floor():
    kw = dict(process="lpbf", cells=[5.0, 8.0], walls=[0.5], fractions=(0.6, 1.0))
    best = size_exchanger(DUTY, ENV, objective="effectiveness", **kw)["best"]
    light = size_exchanger(DUTY, ENV, objective="mass", **kw)["best"]
    assert best["effectiveness"] >= light["effectiveness"]
    floor = size_exchanger(DUTY, ENV, objective="mass", min_effectiveness=0.9 * best["effectiveness"], **kw)
    assert all(c["effectiveness"] >= 0.9 * best["effectiveness"] for c in floor["candidates"])


def test_fdm_limits_the_wall():
    res = size_exchanger(DUTY, ENV, process="fdm", cells=[6.0, 8.0], fractions=(1.0,))
    assert all(c["wall"] >= PROCESS_LIMITS["fdm"]["minWall"] for c in res["candidates"])
    with pytest.raises(ValueError):
        size_exchanger(DUTY, ENV, process="cnc")


def test_heatx_size_takes_an_envelope_spec():
    spec = {"name": "box", "envelope": ENV, "duty": DUTY, "layout": "cross",
            "sizing": {"process": "lpbf", "maxPressureDrop": {"A": 5000, "B": 5000}}}
    res = size_heat_exchanger(spec)
    assert res["best"] and res["best"]["spec"]["name"].startswith("box-")
    assert res["best"]["spec"]["process"] == "lpbf"


def test_verify_simulates_a_cross_candidate_whole():
    from loadpath.sizing import verify

    res = size_exchanger(DUTY, ENV, process="lpbf", layout="cross", cells=[8.0], walls=[0.8], splits=(0.5,),
                         fractions=(1.0,), max_pressure_drop={"A": 5000, "B": 5000}, objective="effectiveness")
    v = verify(res["best"], voxel=3.0, method="box")
    assert v["method"] == "box"
    assert abs(v["simulated"]["effectiveness"] / v["sized"]["effectiveness"] - 1) < 0.15
