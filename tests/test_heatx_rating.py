"""Rating and sizing heat exchangers: the Nu band, the pressure-drop fit's
range, metal-light sizing and an offset sheet (split)."""
import numpy as np

from loadpath.heatx_thermal import Fluid, rate, size_core, tpms_geometry

AIR = Fluid("charge air", 1.75, 1015.0, 0.032, 2.25e-5)
COOL = Fluid("coolant", 1040.0, 3500.0, 0.42, 1.2e-3)
DUTY = {"air": {"fluid": AIR, "kgPerS": 0.143, "tIn": 137.0}, "cool": {"fluid": COOL, "kgPerS": 0.33, "tIn": 55.0},
        "wall": "alsi10mg"}
CORE = 216.0 * 90.0 * 61.0


def test_design_nu_is_the_local_fit_and_the_band_spans_cordisco():
    r = rate("gyroid", 10.0, 1.0, 40e3, [40, 40], {"a": {"fluid": "air", "kgPerS": 0.004, "tIn": 80},
                                                    "b": {"fluid": "water", "lPerMin": 1, "tIn": 20}})
    for s in r["fluids"].values():
        f = s["nuFits"]
        assert s["nu"] == f["reynolds2023"]
        assert s["nuSpread"] == [min(f.values()), max(f.values())]
        assert f["cordisco2025"] < f["reynolds2023"]
    lo, hi = r["effectivenessRange"]
    assert lo < r["effectiveness"] <= hi + 1e-12


def test_pressure_drop_flag_covers_gyroid_sheets_to_re_1500():
    r = rate("gyroid", 10.0, 1.0, 40e3, [40, 40], {"a": {"fluid": "air", "kgPerS": 0.002, "tIn": 80},
                                                    "b": {"fluid": "water", "lPerMin": 2, "tIn": 20}})
    a = r["fluids"]["a"]
    assert 100 < a["re"] < 1500 and not any("pressure drop" in f for f in a["flags"])


def test_split_moves_the_sheet_toward_b():
    g = tpms_geometry("gyroid", 10.0, 1.5, 0.66)
    a, b = g["sides"]
    assert abs(a["phi"] / (a["phi"] + b["phi"]) - 0.66) < 0.01
    # the 1KZ cooler's LBM cell at 10/1.5 measured Dh 5.13 (air) and 3.15 mm (coolant)
    assert abs(a["dh"] - 5.13) < 0.3 and abs(b["dh"] - 3.15) < 0.2
    assert g["offset"] < 0


def test_size_core_least_metal_meets_the_limits():
    lim = {"air": 12000, "cool": 30000}
    res = size_core("gyroid", CORE, [90.0, 432.0], DUTY, "counter", 60.0, min_wall=1.0, max_pressure_drop=lim,
                    cells=[8, 9, 10, 12], walls=[1.0, 1.5], split=[0.5, 0.66], objective="metal",
                    min_effectiveness=0.6)
    c = res["candidates"]
    assert c and all(x["effectiveness"] >= 0.6 and x["pressureDrop"]["air"] <= 12000 for x in c)
    assert [x["solidVolume"] for x in c] == sorted(x["solidVolume"] for x in c)
    assert c[0]["mass"] == c[0]["solidVolume"] * 2.67 / 1000
