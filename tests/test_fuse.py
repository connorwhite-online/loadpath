"""
Designed weak links (loadpath.fuse): break force against the closed forms, sizing that holds service on the
weakest print, the window's width, crash cases capped through a study, and the cap reaching the impact
model's peak force.

Run: `python tests/test_fuse.py` (or pytest). A few seconds.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadpath import fuse  # noqa: E402
from loadpath.materials import material_from_spec  # noqa: E402


def test_break_force_closed_forms():
    s = material_from_spec("pa12").strength
    # single-shear pin: tau_u pi d^2 / 4, tau_u = s / sqrt 3; double shear doubles it, two pins double again
    f1 = fuse.break_force("shear-pin", 3.0, "pa12", "sls")
    assert abs(f1 / (s / math.sqrt(3) * math.pi * 9 / 4) - 1) < 1e-9
    assert abs(fuse.break_force("shear-pin", 3.0, "pa12", "sls", planes=2, count=2) / f1 - 4) < 1e-9
    # tension link: s w t; across layers the process's bond factor (SLS 0.9)
    assert abs(fuse.break_force("tension-link", (4.0, 2.0), "pa12", "sls") - s * 8) < 1e-9
    assert abs(fuse.break_force("tension-link", (4.0, 2.0), "pa12", "sls", across_layers=True) / (s * 8) - 0.9) < 1e-9


def test_sizing_holds_service_and_reports_window():
    r = fuse.size_fuse("shear-pin", 40.0, "pa12", "sls", service_sf=2.0)
    # the weakest print holds 2x the service load, and the size is the smallest that does (to the 0.05 step)
    assert r["breakLow"] >= 80.0 - 1e-6
    smaller = fuse.break_force("shear-pin", r["diameter"] - fuse.STEP, "pa12", "sls") * (1 - 2 * 0.05)
    assert smaller < 80.0
    # window: (1 + 2 cov) x rate / (1 - 2 cov), SLS cov 0.05, semicrystalline rate 1.3
    assert abs(r["window"] - 1.1 * 1.3 / 0.9) < 1e-3
    assert r["sizedBy"] == "service"
    # filament scatters more: a wider window for the same job
    assert fuse.size_fuse("shear-pin", 40.0, "petg", "fdm")["window"] > r["window"]
    # a tiny load leaves the pin at the printable minimum, and says so
    tiny = fuse.size_fuse("shear-pin", 0.5, "pa12", "sls")
    assert tiny["sizedBy"] == "print" and abs(tiny["diameter"] - 1.6) < 1e-9 and "note" in tiny
    # protecting a part that takes less than the window's top is reported as not protecting it
    bad = fuse.size_fuse("shear-pin", 40.0, "pa12", "sls", protect_force=100.0)
    assert bad["protects"] is False and "why" in bad
    assert fuse.size_fuse("shear-pin", 40.0, "pa12", "sls", protect_force=200.0)["protects"] is True


def _study():
    return {
        "name": "fuse-bracket", "material": "pa12", "process": {"type": "sls"},
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 20, 10]},
        "preserve": [{"id": "wall", "shape": {"type": "box", "min": [0, 0, 0], "max": [2, 20, 10]}},
                     {"id": "lug", "shape": {"type": "box", "min": [34, 6, 0], "max": [40, 14, 10]}}],
        "supports": [{"region": "wall"}],
        "loadCases": [
            {"name": "shelf", "loads": [{"region": "lug", "force": [0, 0, -30]}]},
            {"name": "knock", "loads": [{"region": "lug", "force": [-1, 0, 0]}],
             "impact": {"energy": 2.0, "mass": 0.5}},
        ],
        "fuses": [{"id": "pin", "type": "shear-pin", "region": "lug", "axis": [0, 1, 0]}],
        "goal": {"safetyFactor": 2.0},
        "resolution": {"pitch": 2.0},
    }


def test_study_caps_crash_cases():
    from loadpath.spec import study_from_dict

    d = fuse.expand_fuses(_study())
    row = d["_fuses"][0]
    assert row["serviceForce"] == 30.0 and row["breakLow"] >= 60.0 - 1e-6
    knock = [c for c in d["loadCases"] if c["name"] == "knock"][0]
    # the hit runs across the pin: capped at the window's top
    assert abs(knock["impact"]["maxForce"] - row["breakHigh"]) < 1e-3
    assert fuse.expand_fuses(d) is d                       # a re-parse does not size twice
    st = study_from_dict(_study())
    case = [c for c in st.load_cases if c.name == "knock"][0]
    assert abs(case.impact.max_force - row["breakHigh"]) < 1e-3
    assert st.raw["_fuses"][0]["id"] == "pin"


def test_cap_reaches_the_peak_force():
    from loadpath.impact import series_impact

    # a stiff part with no crusher: an unbounded spike, held to the fuse's force by the cap
    r = series_impact(2000.0, 1e-4, None, None)
    assert r["peakForce"] > 1000
    cap = fuse.size_fuse("shear-pin", 30.0, "pa12", "sls")["breakHigh"]
    assert cap < r["peakForce"]


def test_pin_mesh_watertight():
    m = fuse.fuse_pin_mesh(2.0, 8.0)
    assert m.is_watertight and abs(m.bounds[1][2] - (8.0 + 1.2)) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
