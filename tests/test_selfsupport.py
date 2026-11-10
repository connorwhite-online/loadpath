"""Round plenums and ports: shaped for the build only where the process needs it."""
import math

import numpy as np
import pytest

from loadpath.selfsupport import SELF_SUPPORT, crown_height, needs_shaping, round_limit


def test_round_until_the_process_limit_then_a_smooth_egg():
    assert crown_height(3.0, "lpbf") == 3.0                      # a 6 mm hole builds round in metal
    assert math.isclose(crown_height(4.0, "lpbf"), 4.0)          # 8 mm: the limit, still a circle
    assert crown_height(4.01, "lpbf") >= 4.01                    # just past it: never under the circle
    h = crown_height(6.0, "lpbf")
    assert 6.0 < h < 6.0 * math.sqrt(2.0) + 6.0
    # the egg's own crown flatter than 45 degrees is the chord the 8 mm circle closes
    r, a = 6.0, math.radians(45.0)
    x = r * r * math.tan(a) / math.sqrt(h * h + r * r * math.tan(a) ** 2)
    assert math.isclose(2.0 * x, 0.9 * 8.0 * math.sin(a), rel_tol=1e-9)     # DESIGN_MARGIN
    for p in ("sls", "mjf"):
        assert not needs_shaping(p) and round_limit(p) == math.inf and crown_height(30.0, p) == 30.0
    assert round_limit("fdm") > 20.0 and needs_shaping("fdm")
    assert set(SELF_SUPPORT) >= {"fdm", "lpbf", "sls", "mjf", "sla", "dlp"}
    with pytest.raises(ValueError):
        crown_height(3.0, "casting")


BASE = {"core": {"type": "box", "min": [0, 0, 0], "max": [30, 30, 24]}, "layout": "cross", "axes": "xy",
        "cell": 8, "wall": 1.0, "header": 6, "portDiameter": 6, "portLength": 4}


def _design(**kw):
    from loadpath.heatx import design_heat_exchanger, heat_exchanger_from_spec

    hx = heat_exchanger_from_spec({**BASE, **kw})
    hx.step = 0.45
    return design_heat_exchanger(hx)


def test_powder_polymer_exchanger_is_round_everywhere():
    mesh, rep = _design(process="sls", portDiameter=12)
    assert rep["isolated"] and all(rep["connected"].values()) and mesh.is_watertight
    for z in rep["zones"].values():
        assert z["roof"] == "vault" and z["portCrown"] == 6.0 and z["vaultRise"] == 1.0
    assert "internalCeilings" not in rep                 # nothing is shaped for a build


def test_metal_ports_round_below_the_limit_and_plenum_vaults_round():
    mesh, rep = _design(process="lpbf")
    assert rep["isolated"] and all(rep["connected"].values()) and mesh.is_watertight
    for z in rep["zones"].values():
        # a 6 mm port and a 6 mm deep plenum both build round in powder-bed metal
        assert z["portCrown"] == 3.0 and z["vaultRise"] == 1.0
    # the round crowns are near-level, but every point of them is within half the arch chord
    # powder-bed metal closes of an edge that holds it
    ic = rep["internalCeilings"]
    assert ic["flatArea"] > 0 and ic["unbuildableArea"] < 20.0, ic
    assert ic["buildableChord"] == 5.66


def test_metal_deep_plenum_vault_rises_only_on_its_top_edge():
    _, rep = _design(process="lpbf", header=12)
    assert rep["isolated"] and all(rep["connected"].values())
    assert all(z["vaultRise"] > 1.0 for z in rep["zones"].values())
    _, rep = _design(process="fdm", header=12)
    assert all(z["vaultRise"] == 1.0 for z in rep["zones"].values())   # 12 mm FDM bridges


def test_exchanger_core_bore_is_round_when_the_process_builds_it():
    from loadpath.exchanger import exchanger_core

    for process, big in (("lpbf", True), ("sls", False)):
        _, _, plugs, _ = exchanger_core((30, 30, 24), cell=8, wall=1.0, port_r=6.0, build="+z", process=process)
        rs = sorted({round(p["r"], 3) for p in plugs})
        assert (max(rs) > 6.0) == big, (process, rs)
    _, _, plugs, _ = exchanger_core((30, 30, 24), cell=8, wall=1.0, port_r=3.0, build="+z")
    assert all(np.isclose(p["r"], 3.0) for p in plugs)
