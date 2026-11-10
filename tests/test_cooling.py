"""
Cooling inside a fixed exterior (loadpath/cooling.py): component heat,
skin touch limits, vents, designed together with feel targets. Small
problems; under a minute.

Run: `python3 tests/test_cooling.py`
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import cooling  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.switch import _thin_skin  # noqa: E402


def box(**over):
    """A 40 x 24 x 12 mm PA12 box, 1.2 mm skin, a board with a 0.5 W chip."""
    doc = {
        "name": "box",
        "designSpace": {"type": "box", "min": [-20, -12, 0], "max": [20, 12, 12]},
        "skin": {"thickness": 1.2},
        "keepOut": [
            {"id": "board", "bore": False, "shape": {"type": "box", "min": [-16, -9, 2], "max": [16, 9, 3.6]}},
            {"id": "chip", "bore": False, "shape": {"type": "box", "min": [-10, -4, 3.6], "max": [-2, 4, 4.8]}}],
        "material": "pa12", "minMember": 2.0, "resolution": {"pitch": 2.0},
        "cooling": {"ambient": 25, "components": [
            {"region": "chip", "watts": 0.5, "maxTemp": 60},
            {"region": "board", "conductivity": 20}],
            "skin": {"contact": "held"}, "iterations": 30},
    }
    doc["cooling"].update(over)
    return doc


def _dom(doc, pitch=None):
    st = study_from_dict(doc)
    dom, fl = _thin_skin(st, build_domain(st, pitch or doc["resolution"]["pitch"]))
    return st, dom, fl


def test_lints_clean_without_supports_or_loads():
    assert not lint(box(), "study"), lint(box(), "study")
    assert lint(box(components=[{"region": "chip", "watt": 1}]), "study")
    bad = box(skin={"contact": "warm"})
    assert lint(bad, "study")
    try:
        study_from_dict(box(components=[{"region": "chip", "watts": 1, "maxTemp": 20}]))
        cooling.cooling_from_study(study_from_dict(box(components=[{"region": "chip", "watts": 1,
                                                                   "maxTemp": 20}])))
        raise AssertionError("a limit under the ambient should be refused")
    except ValueError as exc:
        assert "ambient" in str(exc)


def test_heat_balance_and_uniform_skin():
    # all heat leaves through the outside: sum h A theta over the faces is
    # the power, whatever is inside
    st, dom, fl = _dom(box(), 1.0)
    for x in (np.maximum(dom.preserve.astype(float), fl), np.ones(dom.n_elem)):
        r = cooling.cooling_readings(st, dom, np.maximum(x, 1e-9), floor=fl)
        assert abs(r["heatOut"]["skin"] - 0.5) < 1e-4, r["heatOut"]
        # every temperature over ambient, the chip the hottest point
        t = r["_theta"]
        assert t.min() >= 25.0 - 1e-6
        assert abs(r["components"][0]["temperature"] - t.max()) < 0.01
    # an exterior of area A at uniform temperature rise W / (h A): a part
    # filled with a much better conductor is nearly isothermal
    doc = box()
    doc["cooling"]["conductivity"] = 5000.0
    st, dom, fl = _dom(doc, 1.0)
    r = cooling.cooling_readings(st, dom, np.ones(dom.n_elem), floor=fl)
    area = 2 * (40 * 24 + 40 * 12 + 24 * 12) * 1e-6
    expect = 25 + 0.5 / (cooling.H_OUT * area)
    assert abs(r["skin"]["mean"] - expect) < 0.03 * (expect - 25), (r["skin"], expect)


def test_sphere_in_a_shell_matches_the_exact_answer():
    # a 1 W sphere (r 4 mm, near-isothermal) in a PA 12 ball (r 20 mm)
    # losing heat at h: rise = W/(4 pi k)(1/a - 1/b) + W/(4 pi b^2 h). The
    # grid's stepped outside has 1.5x the ball's area; each face convects
    # over the true surface it stands for, or the outside reads 34% cool
    a, b, w, k = 4.0, 20.0, 1.0, 0.25
    doc = {"name": "ball", "designSpace": {"type": "sphere", "center": [0, 0, 0], "r": b},
           "keepOut": [{"id": "src", "bore": False, "shape": {"type": "sphere", "center": [0, 0, 0], "r": a}}],
           "material": "pa12", "minMember": 2.0, "resolution": {"pitch": 2.0},
           "cooling": {"ambient": 25, "conductivity": k,
                       "components": [{"region": "src", "watts": w, "conductivity": 2000, "maxTemp": 500}]}}
    st = study_from_dict(doc)
    dom = build_domain(st, 2.0)
    r = cooling.cooling_readings(st, dom, np.ones(dom.n_elem))
    h = cooling.H_OUT * 1e-6
    skin = w / (4 * np.pi * b ** 2 * h)
    src = w / (4 * np.pi * k * 1e-3) * (1 / a - 1 / b) + skin
    assert abs(r["skin"]["mean"] - 25 - skin) < 0.03 * skin, (r["skin"], skin)
    assert abs(r["components"][0]["temperature"] - 25 - src) < 0.08 * src, (r["components"], src)
    assert abs(r["heatOut"]["skin"] - w) < 1e-3


def test_cycle_lumped_heat_up_matches_the_exact_answer():
    # a near-isothermal box heated from room temperature: W/(hA)(1 - e^-t/tau),
    # tau = C/(hA); backward Euler is first order, 1% at 40 steps here
    doc = box()
    doc["cooling"]["conductivity"] = 5000.0
    doc["cooling"]["cycle"] = {"start": "ambient", "phases": [{"seconds": 600, "steps": 40}]}
    st, dom, fl = _dom(doc, 2.0)
    x = np.ones(dom.n_elem)
    mdl = cooling.CoolingModel(st, dom, floor=fl)
    th, first = mdl.history(x)
    mdl.release_history()
    cap = mdl.capacity(x).sum()
    ha = cooling.H_OUT * 1e-6 * float(mdl.out_w.sum()) * 4.0
    tau = cap / ha
    exact = 0.5 / ha * (1 - np.exp(-600 / tau))
    assert abs(th[-1].mean() - exact) < 0.015 * exact, (th[-1].mean(), exact, tau)
    r = cooling.cooling_readings(st, dom, x, floor=fl)
    assert r["cycle"]["seconds"] == 600 and r["components"][0]["peakAt"] == 600


def test_cycle_sensitivities_match_finite_differences():
    # a burst after idling: the limits read the hottest moment; the adjoint
    # runs back through every step and the steady start
    doc = box()
    doc["cooling"]["cycle"] = {"phases": [{"seconds": 300, "watts": {"chip": 0.1}, "steps": 3},
                                          {"seconds": 120, "watts": {"chip": 1.5}, "steps": 4},
                                          {"seconds": 60, "watts": {"chip": 0.1}, "steps": 2}]}
    st, dom, fl = _dom(doc, 2.0)
    prob = cooling.CoolingProblem(st, dom, floor=fl)
    rng = np.random.default_rng(2)
    x = np.clip(0.3 + 0.4 * rng.random(dom.n_elem), 0, 1)
    x[dom.preserve] = 1.0
    x = np.maximum(x, fl)
    mdl = prob.model
    cand = np.flatnonzero(~dom.preserve & (x > fl + 0.05))

    def pnorms(xx):
        th, first = mdl.history(xx)
        out = [lim.value_history(th, first) for lim in prob.limits]
        return th, first, out
    th, first, base = pnorms(x)
    g = mdl.history_gradient(x, th, np.stack([d for _, d, _ in base], axis=2))
    mdl.release_history()
    for e in cand[[3, len(cand) // 2, -3]]:
        h = 1e-4
        xp, xm = x.copy(), x.copy()
        xp[e] += h
        xm[e] -= h
        vp = [v for v, _, _ in pnorms(xp)[2]]
        mdl.release_history()
        vm = [v for v, _, _ in pnorms(xm)[2]]
        mdl.release_history()
        for j, lim in enumerate(prob.limits):
            fd = (vp[j] - vm[j]) / (2 * h)
            assert abs(fd - g[e, j]) <= 1e-3 * abs(g[e, j]) + 1e-12, (lim.name, e, g[e, j], fd)


def test_sensitivities_match_finite_differences():
    st, dom, fl = _dom(box(), 1.0)
    prob = cooling.CoolingProblem(st, dom, floor=fl)
    mdl = prob.model
    rng = np.random.default_rng(1)
    x = np.clip(0.3 + 0.4 * rng.random(dom.n_elem), 0, 1)
    x[dom.preserve] = 1.0
    x = np.maximum(x, fl)
    cand = np.flatnonzero(~dom.preserve & (x > fl + 0.05))
    for lim in prob.limits:
        th = mdl.solve(x)
        v, d, _ = lim.value(th)
        g = mdl.gradient(x, th, mdl.adjoint(d))
        for e in cand[[5, len(cand) // 2, -5]]:
            h = 1e-4
            xp, xm = x.copy(), x.copy()
            xp[e] += h
            xm[e] -= h
            fd = (lim.value(mdl.solve(xp))[0] - lim.value(mdl.solve(xm))[0]) / (2 * h)
            assert abs(fd - g[e]) <= 1e-3 * abs(g[e]) + 1e-12, (lim.name, e, g[e], fd)
    mdl.release()


def test_stack_flow():
    # 1 cm^2 in and out, 100 mm apart, 10 K: Cd A_eff sqrt(2 g H dT / T)
    g, q = cooling.stack_conductance([(100.0, 0.0), (100.0, 100.0)], 10.0, 25.0)
    a = 1e-4 / np.sqrt(2)
    q0 = cooling.CD * a * np.sqrt(2 * 9.81 * 0.1 * 10 / 298.15)
    assert abs(q - q0) < 1e-12 and abs(g - cooling.RHO_CP_AIR * q0) < 1e-9
    assert cooling.stack_conductance([(100.0, 0.0), (100.0, 0.0)], 10.0, 25.0) == (0.0, 0.0)
    # vents in the box's ends, held upright: cooler, and the heat still balances
    doc = box(gravity="-y", vents=[{"shape": {"type": "box", "min": [-8, -13, 2], "max": [8, -11, 10]}},
                                   {"shape": {"type": "box", "min": [-8, 11, 2], "max": [8, 13, 10]}}])
    st, dom, fl = _dom(doc, 1.0)
    x = np.maximum(np.maximum(dom.preserve.astype(float), fl), 1e-9)
    vented = cooling.cooling_readings(st, dom, x, floor=fl)
    shut = cooling.cooling_readings(*_dom(box(), 1.0)[:2], x, floor=fl)
    out = vented["heatOut"]
    assert abs(out["skin"] + out["vents"] - 0.5) < 2e-3 * 0.5, out
    assert out["vents"] > 0 and vented["components"][0]["temperature"] < shut["components"][0]["temperature"]


def test_run_meets_the_limits_with_a_grip():
    # cooling and a squeeze target designed together
    doc = box(components=[{"region": "chip", "watts": 0.5, "maxTemp": 50},
                          {"region": "board", "conductivity": 20}])
    doc["feel"] = {"targets": [{"kind": "squeeze", "name": "grip", "hold": {"face": "-y"},
                                "press": {"face": "+y"}, "direction": "-y", "force": 10,
                                "travel": {"max": 0.05}}]}
    st = study_from_dict(doc)
    assert not lint(doc, "study"), lint(doc, "study")
    with tempfile.TemporaryDirectory() as d:
        rep = cooling.run_cooling(st, d, pitches=[2.0], iterations=[30], verify_pitches=[1.0])
    c = rep["cooling"]
    assert c["ok"], c["design"]
    assert rep["feelOk"], rep["feel"]
    assert 0.0 < rep["volume"] < 0.9, rep["volume"]
    assert rep["watertight"] and len(c["delivered"]) == 1
    # the bare skin misses the chip's limit: the interior is doing the work
    st2, dom, fl = _dom(doc, 2.0)
    bare = cooling.cooling_readings(st2, dom, np.maximum(np.maximum(dom.preserve.astype(float), fl), 1e-9),
                                    floor=fl)
    assert not bare["components"][0]["ok"], bare["components"]


def test_closed_exterior_keeps_its_cavity():
    # a liner grown under the whole skin (the spreader a cooling design
    # makes) encloses the interior by itself: the delivered mesh must keep
    # that cavity (it was meshed solid, the cavity's surface dropped as a
    # body of negative volume)
    from scipy import ndimage

    from loadpath.reconstruct import reconstruct
    from loadpath.switch import _as_printed
    doc = box()
    doc["designSpace"] = {"type": "sphere", "center": [0, 0, 6], "r": 14}
    doc["keepOut"] = [{"id": "chip", "bore": False, "shape": {"type": "box", "min": [-4, -4, 2], "max": [4, 4, 4]}}]
    doc["cooling"]["components"] = [{"region": "chip", "watts": 0.5, "maxTemp": 60}]
    st, dom, fl = _dom(doc, 1.0)
    g = np.zeros(dom.dims, bool)
    ijk = dom.mesh.elem_ijk
    g[tuple(ijk[dom.preserve | (fl > 0)].T)] = True
    near = ndimage.binary_dilation(g, iterations=2)[tuple(ijk.T)]
    x = np.maximum(np.where(near, 1.0, 0.0), np.maximum(dom.preserve.astype(float), fl))
    ex, _ = _as_printed(x, fl, dom)
    mesh, _ = reconstruct(st, dom, ex)
    full = 4 / 3 * np.pi * 14 ** 3
    # (meshed solid it was 11.4 of 11.5 cm^3; with its cavity 6.9)
    assert mesh.is_watertight and mesh.volume < 0.75 * full, (mesh.volume, full)


def test_airflow_sets_h_from_the_flat_plate():
    # still air reads about H_OUT; a 60 mm canopy at 8 m/s, Re 2.9e4, reads
    # the laminar plate's 0.664 Re^1/2 Pr^1/3 k / L plus radiation
    still = cooling.forced_h(0.0, 60.0)
    assert abs(still["h"] - cooling.H_OUT) < 0.25 * cooling.H_OUT, still
    f = cooling.forced_h(8.0, 60.0)
    nu = 0.664 * f["re"] ** 0.5 * 0.71 ** (1 / 3)
    assert abs(f["forced"] - nu * 0.0263 / 0.06) < 0.05 * f["forced"], f
    assert 4.0 < f["h"] / cooling.H_OUT < 6.0
    st = study_from_dict(box(skin={"contact": "held", "airflow": {"speed": 8.0, "length": 40.0}}))
    spec = cooling.cooling_from_study(st)
    assert abs(spec.h_out - cooling.forced_h(8.0, 40.0)["h"]) < 1e-9
    assert not lint(box(skin={"airflow": {"speed": 8.0, "length": 40.0}}), "study")
    try:
        cooling.cooling_from_study(study_from_dict(box(skin={"h": 20, "airflow": {"speed": 8, "length": 40}})))
        raise AssertionError("h and airflow together should be refused")
    except ValueError as exc:
        assert "airflow" in str(exc)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


def test_air_between_two_components_on_any_grid():
    # the chip lifted 0.4 mm off the board: 0.4 mm of air over 8 x 8 mm is
    # 240 K/W, so the chip runs far hotter than the board. On a 1 mm grid
    # their cells sit face to face, and they once conducted at the board's
    # own 20 W/(m K) straight across the gap (chip 63 C, board 63 C)
    doc = box()
    doc["keepOut"][1]["shape"] = {"type": "box", "min": [-10, -4, 4.0], "max": [-2, 4, 5.2]}
    doc["cooling"]["components"][1]["maxTemp"] = 200
    chips = []
    for p in (1.0, 0.6):
        st, dom, fl = _dom(doc, p)
        x = np.maximum(dom.preserve.astype(float), fl)
        r = cooling.cooling_readings(st, dom, np.maximum(x, 1e-9), floor=fl)
        chip, board = (c["temperature"] for c in r["components"])
        assert chip > board + 40.0, r["components"]
        chips.append(chip)
    assert abs(chips[0] - chips[1]) < 0.25 * (chips[1] - 25.0), chips


def test_delivered_read_takes_a_pad_gap_the_same_on_any_grid():
    # a chip in a printed pocket 0.4 mm proud all round: without the pad
    # the gap is air on the 0.7 mm grid and contact on the 1.0 and 0.5 mm
    # ones (80 / 146 / 83 C). Read as the thermal pad the model assumes,
    # every grid agrees
    import manifold3d as m3d
    import trimesh
    doc = box(components=[{"region": "chip", "watts": 0.5, "maxTemp": 60}])
    doc["keepOut"] = [doc["keepOut"][1]]
    st = study_from_dict(doc)
    blk = m3d.Manifold.cube((40, 24, 12)).translate((-20, -12, 0))
    pocket = m3d.Manifold.cube((8.8, 8.8, 2.0)).translate((-10.4, -4.4, 3.2))
    m = (blk - pocket).to_mesh()
    mesh = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))
    t = [r["components"][0]["temperature"] for r in cooling.delivered_readings(st, mesh, [1.0, 0.7, 0.5])]
    assert max(t) - min(t) < 0.15 * (min(t) - 25.0), t
