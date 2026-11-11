"""
Slip casting (loadpath/slipcast.py): the cake against the filtration law
L^2 = A t, shrinkage against conservation of the solids, the kiln sag solve
against the beam the pyroplastic index is measured on, Weibull's weakest
link against its closed forms, and a cup cast, drained, fired and given a
plaster mold.

    python tests/test_slipcast.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import slipcast as S                     # noqa: E402


def test_filtration_law():
    b = S.body("porcelain")
    # the card's rate is a timed cast: 3 mm in 15 min in a dry plaster mold
    assert abs(S.cast_rate(b) * 6e7 - 0.6) < 1e-6
    assert abs(S.wall_time(3.0, b) - 900.0) < 1.0 and abs(S.wall_thickness(900.0, b) - 3.0) < 1e-3
    assert abs(S.wall_time(6.0, b) / S.wall_time(3.0, b) - 4.0) < 1e-9          # twice the wall, four times as long
    # the cake is the resistance, not the plaster; pressure on the slip adds to the plaster's suction
    assert S.mold_share(b) < 0.01
    k = S.cast_rate(b, pressure=1.5e6) / S.cast_rate(b)
    assert abs(k - (S.PLASTER.suction + 1.5e6) / S.PLASTER.suction) < 1e-6
    # Kozeny-Carman: a finer powder casts slower, as the square of its size
    assert abs(S.kozeny_carman(0.5, 0.6) / S.kozeny_carman(0.25, 0.6) - 4.0) < 1e-9


def test_cake_grows_as_root_time():
    # a slab cavity between two mold faces, solved on the grid: each cell's cake arrives at x^2 / A
    b = S.body("porcelain")
    pitch = 0.25
    cav = np.ones((12, 36, 36), bool)
    c = S.cast(cav, pitch, b, solid=True, opening=np.zeros_like(cav))
    got = c.arrival[:6, 18, 18]
    x = (np.arange(6) + 1.0) * pitch * 1e-3
    exact = x ** 2 / S.cast_rate(b)
    assert np.all(np.abs(got[2:5] / exact[2:5] - 1) < 0.10), got / exact      # past the first cells' rounding
    assert abs(c.close_time / ((1.5e-3) ** 2 / S.cast_rate(b)) - 1) < 0.08, c.close_time


def test_shrinkage_conserves_the_solids():
    for name, lo, hi in (("porcelain", 0.12, 0.17), ("stoneware", 0.10, 0.15), ("earthenware", 0.03, 0.08),
                         ("alumina", 0.12, 0.18)):
        b = S.body(name)
        sh = S.shrinkage(b)
        assert lo < sh["total"] < hi, (name, sh)
        assert abs(sh["moldScale"] * (1 - sh["total"]) - 1) < 1e-12
        # the solids in a unit of cake are the solids in what it fires to
        v = (1 - sh["total"]) ** 3
        assert abs(b.cake_packing * b.particle_density * (1 - b.loss_on_ignition) - v * b.fired_density) \
            < 1e-6 * b.fired_density, name
    assert S.shrinkage(S.body("porcelain"))["drying"] < 0.04          # most of it is the firing


def test_kiln_sag_against_the_beam():
    # a bar held at its middle, both ends overhanging: tip sag 1.5 rho g l^4 Phi / h^2
    b = S.body("porcelain")
    p, nz = 1.25, 4
    nx = 81
    bar = np.ones((nx, 4, nz), bool)
    sup = np.zeros_like(bar)
    sup[nx // 2, :, 0] = True
    r = S.sag(bar, p, b, (2, 1), sup)
    arm, h = (nx // 2) * p * 1e-3, nz * p * 1e-3
    beam = 1.5 * b.fired_density * S.G0 * arm ** 4 * S.creep_compliance(b) / h ** 2 * 1e3
    assert abs(r["maxDown"] / beam - 1) < 0.10, (r["maxDown"], beam)
    # the compliance is the pyroplastic index's own test: a 10 cm span of 5 mm bar sags 3 L^4 PI / (4 h^2)
    span, hh = 0.10, 0.005
    s = 5.0 / 32.0 * b.fired_density * S.G0 * span ** 4 * S.creep_compliance(b) / hh ** 2
    assert abs(s * 100 - 3 * 10.0 ** 4 * b.sag_index / (4 * 0.5 ** 2)) < 1e-9


def test_weakest_link():
    m, v0 = 10.0, 100.0
    s0 = 10.0 / math.gamma(1 + 1 / m)
    # the test volume at its characteristic strength fails 1 - 1/e of the time
    assert abs(S.failure_probability([s0], [v0], 10.0, m, v0) - (1 - math.exp(-1))) < 1e-12
    # twice the volume at the same stress: the survival squares
    p1 = S.failure_probability([6.0], [v0], 10.0, m, v0)
    p2 = S.failure_probability([6.0], [2 * v0], 10.0, m, v0)
    assert abs((1 - p2) - (1 - p1) ** 2) < 1e-12
    assert S.failure_probability([-5.0], [v0], 10.0, m, v0) == 0.0             # compression breaks nothing
    assert abs(S.effective_volume(1000.0, 10.0) - 1000.0 / 242.0) < 1e-9


def _cup():
    import trimesh

    prof = [(0, 0), (26, 0), (26, 3), (24, 3), (24, 6), (30, 10), (38, 90), (0, 90)]
    return trimesh.creation.revolve(np.array(prof, float), sections=64)


def test_cup_cast_drained_and_molded():
    from loadpath import castmold

    cup = _cup()
    rep, f = S.slipcast_part(cup, "porcelain", wall=3.0, pitch=1.2)
    rc = rep["casting"]
    scale = rep["mold"]["scale"]
    assert rep["ok"], rep["warnings"]
    # the time in the mold is the flat wall's, at mold size
    assert abs(rc["time"] / S.wall_time(3.0 * scale, S.body("porcelain")) - 1) < 0.15, rc["time"]
    assert abs(rc["wall"]["mean"] / 3.0 - 1) < 0.2, rc["wall"]
    assert rc["wall"]["max"] > 1.3 * 3.0                       # the inside corner at the foot builds from two faces
    assert rc["trappedVolume"] == 0.0
    assert 0.0 < rep["kiln"]["sagShare"] < 0.01 and rep["strength"]["greenDemoldFailure"] < 1e-3
    body = castmold.cast_body(cup, f)
    assert body.is_watertight and abs(body.volume / rep["part"]["firedVolume"] - 1) < 0.15
    made = castmold.slip_mold(cup, rep, f)
    for name in ("mold-half-a", "mold-half-b", "mold-box-a", "mold-box-b"):
        assert made["meshes"][name].is_watertight and made["meshes"][name].volume > 0, name
    info = made["info"]
    assert info["undercutShare"] < 0.002 and info["partingAxis"] in "xy", info      # splits along the cup's axis
    assert abs(info["scale"] - scale) < 1e-9
    # the two halves and the cavity fill the block: plaster walls at least what five casts of water need
    a, bm = made["meshes"]["mold-half-a"], made["meshes"]["mold-half-b"]
    assert a.volume + bm.volume > 2.0 * cup.volume * scale ** 3


TESTS = [test_filtration_law, test_cake_grows_as_root_time, test_shrinkage_conserves_the_solids,
         test_kiln_sag_against_the_beam, test_weakest_link, test_cup_cast_drained_and_molded]

if __name__ == "__main__":
    import time

    for t in TESTS:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
    print(f"\nall {len(TESTS)} passed")
