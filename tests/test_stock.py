"""
Bought stock members in a study (loadpath/stock.py): a carbon tube, rod or
plate as a body in the analysis with its own material, the printed part
designed round it, and the joint checked from what the solved assembly puts
on the member. Each test reads a number statics or a handbook gives.

Run: `python -m pytest tests/test_stock.py`.
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath import stock
from loadpath.spec import study_from_dict

L = 90.0          # the tube's length; its first 26 mm sit in the hub
TIP = {"type": "cylinder", "p0": [L + 1, 0, 0], "p1": [L + 4, 0, 0], "r": 5, "rInner": 4}


def _study(joint=None, loads=None, material="cfrp-pultruded", **extra):
    d = {"name": "socket",
         "designSpace": [{"type": "box", "min": [-6, -12, -12], "max": [30, 12, 12]}],
         "preserve": [{"id": "base", "shape": {"type": "box", "min": [-6, -12, -12], "max": [-3, 12, 12]}}],
         "supports": [{"region": "base"}],
         "stock": [{"id": "arm", "type": "tube", "material": material, "od": 10, "wall": 1,
                    "from": [4, 0, 0], "to": [4 + L, 0, 0], "joint": joint or {"type": "bond"}}],
         "loadCases": loads or [{"name": "pull", "loads": [{"shape": TIP, "force": [100, 0, 0]}]},
                                {"name": "bend", "loads": [{"shape": TIP, "force": [0, 0, -10]}]}],
         "material": "pa12", "goal": {"safetyFactor": 2.0}, "resolution": {"pitch": 1.5}}
    d.update(extra)
    return study_from_dict(d)


def _block(study):
    """The printed part with nothing optimized away: the design space less
    the keep-outs (the member's seat among them)."""
    import manifold3d as m3d

    from loadpath.shapes import trimesh_from_manifold
    from loadpath.sharp import exact_manifold

    add = m3d.Manifold.batch_boolean([exact_manifold(s) for s in study.design_space], m3d.OpType.Add)
    cut = m3d.Manifold.batch_boolean([exact_manifold(r.shape) for r in study.keep_out], m3d.OpType.Add)
    return trimesh_from_manifold(add - cut)


def _report(study, cells=12_000):
    from loadpath.verify import _verify_part

    return _verify_part(study, _block(study), max_cells=cells, contact_band=1.5)


def _case(rows, name):
    return next(c for c in rows if c["name"] == name)


def test_laminate_constants():
    ud = stock.laminate(stock.LAMINAE["carbon-epoxy-ud"], [(0, 1.0)])
    lam = stock.LAMINAE["carbon-epoxy-ud"]
    assert abs(ud.E1 - stock._val(lam, "E1")) < 1e-6 * ud.E1 and abs(ud.E2 - stock._val(lam, "E2")) < 1e-6 * ud.E2
    # a quasi-isotropic layup is isotropic in its plane: E1 = E2 and G = E / 2(1 + nu)
    qi = stock.stock_material("cfrp-quasi-isotropic")
    assert abs(qi.E1 - qi.E2) < 1e-3 * qi.E1
    assert abs(qi.G12 - qi.E1 / (2 * (1 + qi.nu12))) < 1e-3 * qi.G12
    # more fibre round the tube: softer along it, stiffer in shear
    a, b = stock.stock_material("cfrp-pultruded"), stock.stock_material("cfrp-roll-wrapped")
    assert a.E1 > b.E1 and b.E2 > a.E2
    # the stiffness matrix every card hands the solver is positive definite
    for k in stock.STOCK:
        assert np.linalg.eigvalsh(stock.stock_material(k).stiffness()).min() > 0


def test_wall_buckling_matches_the_classical_cylinder():
    st = _study(material="6061-t6")
    m = st.stock[0]
    t, r = m.wall, m.ro - 0.5 * m.wall
    e, nu = m.material.E1, m.material.nu12
    classical = e * t / (r * np.sqrt(3 * (1 - nu ** 2)))
    knock = 1 - 0.901 * (1 - np.exp(-np.sqrt(r / t) / 16))
    assert abs(stock._wall_buckling(m) / (classical * knock) - 1) < 0.03


def test_expansion_regions_and_reparse():
    st = _study()
    m = st.stock[0]
    assert [j.kind for j in m.joints] == ["bond"] and abs(m.joints[0].span[0]) < 1e-6
    assert abs(m.joints[0].span[1] - 26.0) < 0.6          # to the mouth of the design space
    assert {"arm", "arm-clear-0"} <= {r.id for r in st.keep_out}
    assert "arm-sleeve-0" in {r.id for r in st.preserve}
    seat = next(r for r in st.keep_out if r.id == "arm").shape
    # the bond line is in the seat (and the process's hole compensation on top, fits.py)
    assert 5 + m.joints[0].gap - 1e-9 <= seat.r < 5 + m.joints[0].gap + 0.2
    again = study_from_dict(copy.deepcopy(st.raw))
    assert [r.id for r in again.keep_out] == [r.id for r in st.keep_out]
    assert abs(next(r for r in again.keep_out if r.id == "arm").shape.r - seat.r) < 1e-9
    assert again.stock[0].joints[0].span == m.joints[0].span


def test_member_stiffness_is_the_tube_s():
    """A tube cantilevered from a clamped block: the grid's member bends,
    stretches and twists as the tube's own section and moduli say."""
    from loadpath.domain import build_domain
    from loadpath.fem import Solver, Stiffness

    st = _study(loads=[{"name": "bend", "loads": [{"shape": TIP, "force": [0, 0, -10]}]},
                       {"name": "pull", "loads": [{"shape": TIP, "force": [10, 0, 0]}]}])
    dom = build_domain(st)
    mesh = dom.mesh
    k = (Stiffness(mesh, st.elastic(design=True), dom.fixed).assemble(np.ones(mesh.n_elem)) + dom.springs).tocsr()
    us = Solver(mesh, rtol=1e-8, maxiter=5000).solve_many(k, dom.loads)
    m = st.stock[0]
    sec, mat = m.section(), m.material
    p = stock.stock_cells(st, mesh).per[0]
    free = p["nodes"][(p["s"] > 34) & (p["s"] < L - 6)]
    x = mesh.node_xyz[free, 0]
    xu = np.unique(np.round(x, 6))
    e_print = st.material.E
    uz = np.array([us[0].reshape(-1, 3)[free, 2][np.isclose(x, v)].mean() for v in xu]) / e_print
    ux = np.array([us[1].reshape(-1, 3)[free, 0][np.isclose(x, v)].mean() for v in xu]) / e_print
    ei = 10.0 / abs(6 * np.polyfit(xu, uz, 3)[0])          # u''' = P / EI
    ea = 10.0 / np.polyfit(xu, ux, 1)[0]
    assert abs(ei / (mat.E1 * sec["I2"]) - 1) < 0.12
    assert abs(ea / (mat.E1 * sec["A"]) - 1) < 0.08


def test_bonded_socket_carries_what_statics_says():
    st = _study()
    rep = _report(st)
    sk = rep["stock"]
    assert rep["bodies"] == 1 and abs(rep["stockMass"] - st.stock[0].mass()) < 1e-9
    mem, joint = sk["members"][0], sk["joints"][0]
    sec = st.stock[0].section()
    pull = _case(mem["cases"], "pull")
    assert abs(pull["axialForce"] - 100) < 3 and abs(pull["tensileStress"] - 100 / sec["A"]) < 0.25
    # the bond takes the whole pull, on the member's sides alone
    jp = _case(joint["cases"], "pull")
    assert abs(jp["axialForce"] + 100) < 4               # what the print puts on the member: against the pull
    area = np.pi * 10 * (joint["span"][1] - joint["span"][0])
    assert abs(jp["meanShear"] - 100 / area) < 0.1 * 100 / area
    assert jp["limitedBy"].startswith("bond") and 0 < jp["safetyFactor"] < joint["adhesive"]["strength"] / jp["meanShear"]
    # bending: the member's worst section is at the mouth or inside it, the joint holds the tip load and
    # its moment about the joint's middle
    bend = _case(mem["cases"], "bend")
    arm = (4 + L + 2.5) - 30.0
    assert bend["bendingMoment"] <= 10 * arm * 1.05
    jb = _case(joint["cases"], "bend")
    mid = 4 + 0.5 * (joint["span"][0] + joint["span"][1])
    assert abs(jb["shearForce"] - 10) < 1.0
    assert abs(jb["bendingMoment"] - 10 * (4 + L + 2.5 - mid)) < 0.08 * 10 * (4 + L + 2.5 - mid)
    # the supports carry the applied load, members and all
    r = rep["reactions"][0]
    assert abs(_case(r["cases"], "pull")["magnitude"] - 100) < 1.0
    assert sk["margin"] is not None and "endFace" in joint


def test_clamp_slips_at_friction_times_grip():
    st = _study(joint={"type": "clamp", "clampForce": 500.0, "friction": 0.2},
                loads=[{"name": "pull", "loads": [{"shape": TIP, "force": [60, 0, 0]}]}])
    joint = _report(st)["stock"]["joints"][0]
    c = joint["cases"][0]
    assert abs(c["slipForce"] - 60) < 3
    assert abs(c["safetyFactor"] - 0.2 * 500 / c["slipForce"]) < 1e-3 and c["limitedBy"] == "clamp slip"


def test_pin_bears_on_both_walls():
    st = _study(joint={"type": "pin", "diameter": 3.0, "at": 14.0},
                loads=[{"name": "pull", "loads": [{"shape": TIP, "force": [100, 0, 0]}]}])
    assert "arm-pin-0" in {r.id for r in st.keep_out}
    c = _report(st)["stock"]["joints"][0]["cases"][0]
    assert abs(c["bearingStress"] - 0.5 * abs(c["axialForce"]) / (3.0 * 1.0)) < 1e-2
    assert abs(c["axialForce"] + 100) < 5 and c["edgeDistance"] == 14.0


def test_margin_counts_the_joint():
    from loadpath.study import _margin

    st = _study()
    rep = {"safetyFactor": 10.0, "cases": [], "stock": {"margin": 0.6}}
    assert _margin(st, rep) == 0.6


def test_bad_stock_is_reported_plainly():
    for bad, word in (({"od": 10, "wall": 6}, "wall"), ({"material": "unobtainium"}, "material"),
                      ({"joint": {"type": "clamp"}}, "clamp"), ({"joint": {"type": "weld"}}, "type")):
        d = {"designSpace": [{"type": "box", "min": [0, -10, -10], "max": [30, 10, 10]}],
             "stock": [{"id": "t", "type": "tube", "od": 10, "wall": 1, "from": [5, 0, 0], "to": [80, 0, 0], **bad}],
             "supports": [{"shape": {"type": "box", "min": [-1, -10, -10], "max": [0.5, 10, 10]}}],
             "loadCases": [{"name": "a", "loads": [{"region": "t", "force": [0, 0, -5]}]}], "material": "pa12",
             "goal": {"volumeFraction": 0.4}}
        try:
            study_from_dict(d)
        except ValueError as exc:
            assert word in str(exc), (word, str(exc))
        else:
            raise AssertionError(f"accepted a bad {word}")


def test_span_holds_on_a_member_of_any_length():
    """A default span is rounded for the study's raw: it must still lie on a
    member whose length does not round (a diagonal tube, 43.5436 mm)."""
    d = {"designSpace": [{"type": "box", "min": [-20, -20, -10], "max": [60, 60, 10]}],
         "stock": [{"id": "t", "type": "tube", "od": 8, "wall": 1, "from": [0.0, 0.0, 0.0], "to": [30.79, 30.79, 0.0]}],
         "supports": [{"shape": {"type": "box", "min": [-20, -20, -10], "max": [-18, 60, 10]}}],
         "loadCases": [{"name": "a", "loads": [{"shape": {"type": "box", "min": [58, -20, -10], "max": [60, 60, 10]},
                                                "force": [0, 0, -5]}]}],
         "material": "pa12", "goal": {"volumeFraction": 0.4}}
    st = study_from_dict(d)
    m = st.stock[0]
    assert m.joints[0].span[1] <= m.length
    again = study_from_dict(copy.deepcopy(st.raw))
    assert again.stock[0].joints[0].span == m.joints[0].span


def test_member_cells_do_not_take_the_plastic_s_expansion():
    """A heated part: the expansion load is the printed material's, so a cell
    that is all member carries none of it (carbon barely grows along its
    fibres; given PA12's growth a tube read 200-300 N in its joints under a
    6 N load, the same in every case)."""
    from loadpath.domain import build_domain
    from loadpath.thermal import StudyHeat

    st = _study(thermal={"ambient": 25.0, "convection": 20.0, "temperatures": [{"region": "base", "t": 70.0}]})
    dom = build_domain(st)
    mesh = dom.mesh
    heat = StudyHeat(st, mesh, st.elastic(st.material), dom.fixed, modulus=st.material.E)
    heat.temperature()
    assert heat.expands
    scale = mesh.stock_scale
    w = heat.load(np.ones(mesh.n_elem), st.material.E)
    only = scale < 1e-3                                   # cells that are member alone
    assert only.any()
    nodes = np.unique(mesh.enode[only])
    other = np.unique(mesh.enode[~only])
    inner = np.setdiff1d(nodes, other)                    # nodes no printed cell touches
    wn = np.abs(w.reshape(-1, 3))
    assert len(inner) and wn[inner].max() < 1e-3 * wn.max()
