"""Multi-material topology optimization (multimat.py): elastics, gradients,
parsing, body split and verification on small grids."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import multimat as M  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.fem import Solver, Stiffness, hex8_ke  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402

BEAM = {
    "name": "beam",
    "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [40, 10, 8]}],
    "preserve": [{"id": "wall", "shape": {"type": "box", "min": [0, 0, 0], "max": [2, 10, 8]}},
                 {"id": "tip", "shape": {"type": "box", "min": [37, 2, 0], "max": [40, 8, 8]}}],
    "supports": [{"region": "wall"}],
    "loadCases": [{"name": "load", "loads": [{"region": "tip", "force": [0, -50, 0]}]},
                  {"name": "knock", "loads": [{"region": "tip", "force": [-50, 0, 0]}]}],
    "process": "fdm",
    "goal": {"volumeFraction": 0.5},
    "multiMaterial": {"materials": ["petg", "tpu"],
                      "requirements": [{"case": "knock", "knock": {"energy": 0.05, "maxForce": 300}}]},
    "minMember": 2.5,
    "resolution": {"maxElements": 2500},
    "maxIterations": 60,
}


def _study(**over):
    d = json.loads(json.dumps(BEAM))
    d["multiMaterial"].update(over)
    return study_from_dict(d)


def test_lame_split_matches_hex8():
    kl, km = M._ke_parts()
    for E, nu in ((1500.0, 0.38), (67.0, 0.45)):
        lam, mu = M.lame(E, nu)
        assert np.allclose(lam * kl + mu * km, E * hex8_ke(nu), rtol=1e-10, atol=1e-8)


def test_lame_assembly_matches_fem():
    st = _study()
    dom = build_domain(st)
    lam, mu = M.lame(1500.0, 0.38)
    a = M.LameStiffness(dom.mesh, dom.fixed).assemble(np.full(dom.n_elem, lam), np.full(dom.n_elem, mu))
    b = Stiffness(dom.mesh, 0.38, dom.fixed).assemble(np.full(dom.n_elem, 1500.0))
    assert abs(a - b).max() < 1e-6 * abs(b).max()


def test_weights_sum_to_one_and_gradient():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.05, 0.95, (2, 7))
    w, dw = M._weights(y)
    assert np.allclose(w.sum(axis=0), 1.0)
    h = 1e-7
    for j in range(2):
        yp = y.copy()
        yp[j] += h
        assert np.allclose((M._weights(yp)[0] - w) / h, dw[:, j], atol=1e-5)
    for k in range(3):
        wk, _ = M._weights(M._y_for(k, 3)[:, None])
        assert wk[k, 0] == 1.0


def test_energy_constraint_gradient():
    st = _study()
    mm, dom = st.multi_material, build_domain(st)
    n = dom.n_elem
    rng = np.random.default_rng(0)
    Es = np.array([m.E for m in mm.materials])
    lam_i, mu_i = M.lame(Es, np.array([m.nu for m in mm.materials]))
    resil = np.array([m.strength ** 2 / (2 * m.E) for m in mm.materials])
    stiff = M.LameStiffness(dom.mesh, dom.fixed)
    solver = Solver(dom.mesh, rtol=1e-12, maxiter=5000)
    lmin, mmin = M.lame(1e-6 * Es.max(), 0.3)
    f = dom.loads[1]

    def g_of(xb, yb):
        w, dw = M._weights(yb[None])
        wp, xpp = w ** 3, xb ** 3
        lam, mu = lmin + xpp * (wp.T @ lam_i), mmin + xpp * (wp.T @ mu_i)
        k = stiff.assemble(lam, mu)
        u = solver.solve_many(k, [f])[0]
        return M._energy_constraint(stiff, solver, k, f, u, float(f @ u), 50.0, lam, mu, xb, w,
                                    3 * xb ** 2 * (wp.T @ lam_i), 3 * xb ** 2 * (wp.T @ mu_i), xpp, 3 * w ** 2,
                                    dw, lam_i, mu_i, resil, 4.0, 0.4 * n)

    xb, yb = rng.uniform(0.3, 1, n), rng.uniform(0.1, 0.9, n)
    g, gx, gy, _ = g_of(xb, yb)
    for e in rng.choice(n, 3, replace=False):
        h = 1e-5
        a = xb.copy()
        a[e] += h
        assert (g_of(a, yb)[0] - g) / h == pytest.approx(gx[e], rel=1e-3, abs=1e-9)
        b = yb.copy()
        b[e] += h
        assert (g_of(xb, b)[0] - g) / h == pytest.approx(gy[0, e], rel=1e-3, abs=1e-9)


def test_parse_errors():
    with pytest.raises(ValueError, match="two to four"):
        _study(materials=["petg"])
    with pytest.raises(ValueError, match="not one of"):
        _study(maxFraction={"pla": 0.2})
    with pytest.raises(ValueError, match="names case"):
        _study(requirements=[{"case": "nope", "knock": {"energy": 1, "maxForce": 10}}])


def test_interface_card_override_and_library():
    st = _study(interfaces={"tpu+petg": {"tensile": 3.0, "shear": 2.0}})
    c = M.interface_card(st.multi_material, 0, 1)
    assert c["tensile"] == 3.0 and c["source"] == "study"
    c = M.interface_card(_study().multi_material, 0, 1, "fdm")
    assert 0 < c["tensile"] < 47.0


def test_traction_util():
    card = {"tensile": 2.0, "shear": 1.0, "tensileZ": 1.0, "shearZ": 0.5}
    s = np.array([[2.0, 0, 0, 0, 0, 0], [-5.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]])
    n = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    u = M._traction_util(s, n, card, build_axis=2)
    assert np.allclose(u, [1.0, 0.0, 1.0])
    # across layers the bond is weaker
    s = np.array([[0, 0, 1.0, 0, 0, 0]])
    assert M._traction_util(s, np.array([[0, 0, 1.0]]), card, build_axis=2)[0] == pytest.approx(1.0)


def test_assign_and_short_run_split_verify(tmp_path):
    d = json.loads(json.dumps(BEAM))
    d["multiMaterial"]["assign"] = [{"region": "tip", "material": "tpu"}]
    # the knock is a separate test: on this 2-cell-filter grid it can leave the
    # tip on a hinge the delivered mesh does not keep
    d["multiMaterial"]["requirements"] = []
    st = study_from_dict(d)
    r = M.run_multimaterial(st, max_runs=1)
    rep = r.report
    names = {b["material"] for b in rep["bodies"]}
    assert names == {"petg", "tpu"}
    assert all(b["watertight"] for b in rep["bodies"])
    assert rep["interfaceArea"]["petg+tpu"] > 0
    assert rep["safetyFactor"] > 0 and rep["interfaceSafetyFactor"] > 0
    files = r.write(tmp_path)
    assert os.path.exists(files["petg"]) and os.path.exists(files["tpu"])
    assert json.load(open(files["report"]))["bodies"]


def test_knock_requirement_is_met_in_the_optimizer():
    from loadpath.simp import filter_radius

    st = _study()
    dom = build_domain(st)
    res = M.optimize_multimaterial(st, dom, st.multi_material, 0.5, filter_radius(dom, st.length_scale),
                                   max_iterations=40)
    knock = res.requirement_values["knock"]
    assert knock["peakForce"] <= 300 * 1.05
    assert np.isfinite(knock["energyUtilization"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
