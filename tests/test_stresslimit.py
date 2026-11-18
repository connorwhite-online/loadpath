"""stresslimit.StressLimit: the load cases' stress limit on a design made
for something else (cooling, feel, a switch)."""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)


def _provider():
    from loadpath.domain import build_domain
    from loadpath.spec import study_from_dict
    from loadpath.stresslimit import StressLimit

    st = study_from_dict(json.load(open(os.path.join(HERE, "data", "two-case-bracket.json"))),
                         os.path.join(HERE, "data"))
    st.goal.safety_factor = st.goal.safety_factor or 2.0
    dom = build_domain(st, 4.0)
    return StressLimit(st, dom, st.elastic(design=True), float(st.material.E)), dom


def test_stress_limit_gradient_matches_finite_differences():
    p, dom = _provider()
    rng = np.random.default_rng(0)
    x = rng.uniform(0.4, 1.0, dom.mesh.n_elem)
    x[dom.preserve] = 1.0
    r0 = p.evaluate(x)
    assert len(r0) == len(dom.case_names) == p.m
    p.freeze = True
    d = rng.normal(size=x.size)
    d[dom.preserve] = 0.0
    h = 1e-5
    rp, rm = p.evaluate(x + h * d), p.evaluate(x - h * d)
    for (_, g, _), (vp, _, _), (vm, _, _) in zip(r0, rp, rm):
        fd = (vp - vm) / (2 * h)
        assert abs(fd - float(g @ d)) <= 1e-5 * max(abs(fd), 1e-3)


def test_a_loose_piece_reads_over_the_limit():
    """Material taken out of the load path reads far over the limit."""
    p, dom = _provider()
    full = np.ones(dom.mesh.n_elem)
    thin = np.full(dom.mesh.n_elem, 0.05)
    thin[dom.preserve] = 1.0
    a = max(v for v, _, _ in p.evaluate(full))
    p.it = 0
    b = max(v for v, _, _ in p.evaluate(thin))
    assert b > a
