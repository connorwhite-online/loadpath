"""Desk stability (loadpath/stability.py): statics against hand calculations."""
import math

import numpy as np
import pytest

from loadpath.stability import G, load_margin, min_mass, run_stability, support_polygon

SQ = {"points": [[-50, -50], [50, -50], [50, 50], [-50, 50]]}


def _feet():
    return support_polygon(SQ)


def test_tip_at_the_hand_calculation():
    # 1 kg, centre 50 mm up over the middle of a 100 mm square: a sideways
    # push at 50 mm tips it at F h = W * 50, F = W
    poly, C, A, z0 = _feet()
    com = np.array([0, 0, 50.0])
    W = 1.0 * G
    r = load_margin(1.0, com, poly, C, A, z0, 2.0, [0, 0, 50], [0.999 * W, 0, 0])
    assert 0 < r["tip"] < 0.01
    r = load_margin(1.0, com, poly, C, A, z0, 2.0, [0, 0, 50], [1.01 * W, 0, 0])
    assert r["tip"] < 0
    m = min_mass(com, poly, C, A, z0, 2.0, [([0, 0, 50], [W, 0, 0], 0.0)])
    assert m == pytest.approx(1.0, rel=1e-3)


def test_slide_needs_force_over_mu_g():
    poly, C, A, z0 = _feet()
    com = np.array([0, 0, 1.0])
    m = min_mass(com, poly, C, A, z0, 0.5, [([0, 0, 0], [1.0, 0, 0], 0.0)])
    assert m == pytest.approx(1.0 / (0.5 * G), rel=1e-3)


def test_a_push_straight_down_inside_never_moves_it():
    poly, C, A, z0 = _feet()
    m = min_mass(np.array([0, 0, 5.0]), poly, C, A, z0, 0.5, [([20, 10, 10], [0, 0, -50.0], 0.0)])
    assert m == 0.0


def test_a_push_down_outside_the_feet_needs_a_counterweight():
    # 1 N down 10 mm outside the edge, centre 50 mm inside it: W 50 >= 1 * 10
    poly, C, A, z0 = _feet()
    m = min_mass(np.array([0, 0, 5.0]), poly, C, A, z0, 0.5, [([60, 0, 10], [0, 0, -1.0], 0.0)])
    assert m == pytest.approx(10.0 / 50.0 / G, rel=1e-3)


def test_twist_needs_mu_n_r_eff():
    # four pads at r = 50 sqrt 2: a pure twist T slips at mu W r
    poly, C, A, z0 = _feet()
    r = 50 * math.sqrt(2)
    m = min_mass(np.array([0, 0, 5.0]), poly, C, A, z0, 0.5, [([0, 0, 0], [0, 0, 0], 1000.0)])
    assert m == pytest.approx(1000.0 / (0.5 * G * r), rel=1e-3)


def test_brief_end_to_end():
    rep = run_stability({
        "parts": [{"id": "block", "mass": 0.05, "at": [0, 0, 6]}],
        "feet": SQ, "friction": 0.5, "envelope_cm3": 30,
        "presses": {"points": [[0, 0, 12], [45, 45, 12]], "force": 3.0, "cone": 20},
        "cables": [{"id": "usb", "at": [0, 50, 6], "direction": [0, 1, -0.2], "force": 0.5, "lever": 0.3,
                    "plug": "usb-c"}]})
    assert rep["mass_g"] == pytest.approx(50.0)
    assert rep["apparentDensity"] == pytest.approx(50 / 30, rel=1e-3)
    tug = rep["cases"][1]["tug"]
    # 0.5 N along a cable leaning 11 degrees down: slide needs H / (mu (W + V))
    d = np.array([0, 1, -0.2]); d /= np.linalg.norm(d)
    H, V = 0.5 * d[1], -0.5 * d[2]
    assert tug["minMass_g"] == pytest.approx(1000 * (H / 0.5 - V) / G, rel=2e-3)
    assert rep["cases"][1]["plugIn"]["slide"] < 0       # 20 N: hold it
    assert "summary" in rep and isinstance(rep["holds"], bool)


def test_cli(tmp_path):
    import json, subprocess, sys
    brief = {"parts": [{"id": "block", "mass": 0.05, "at": [0, 0, 5]}],
             "feet": {"points": [[-20, -20], [20, -20], [20, 20], [-20, 20]]},
             "friction": 0.5,
             "cables": [{"id": "usb", "at": [0, 20, 5], "direction": [0, 1, 0], "force": 0.1}]}
    p = tmp_path / "b.json"; p.write_text(json.dumps(brief))
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, "-m", "loadpath", "stability", str(p), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["mass_g"] == 50.0
