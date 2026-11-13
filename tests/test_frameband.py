"""
A product's side wall as its antenna (loadpath/frameband.py): a metal band
cut out of a fixed two-shell exterior keeps that exterior, comes out as
clean single bodies (no stepped slivers, no loose pieces), the air behind
it stays inside the wall, an opening is measured from the shells, the hand
tissue values are CTIA's, and the band in its product builds as a radio
scene. Conductor loss (antenna.conductor_loss) against a textbook monopole.

    python tests/test_frameband.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import frameband as F                    # noqa: E402


def _shells():
    """A 50 x 100 mm rounded body 14 mm tall, 2.6 mm walls, split at z 7
    into a rear tray and a front cover, a 9 x 3.5 mm slot through the -y end
    at z 7 (a USB mouth)."""
    import trimesh
    from shapely.geometry import box
    from trimesh.creation import extrude_polygon
    out = box(-25, -50, 25, 50).buffer(-6).buffer(6, join_style=1)
    inn = out.buffer(-2.6, join_style=1)

    def solid(poly, z0, z1):
        m = extrude_polygon(poly, z1 - z0)
        m.apply_translation([0, 0, z0])
        return m
    body = trimesh.boolean.difference([solid(out, 0, 14), solid(inn, 2.6, 11.4)], engine="manifold")
    slot = trimesh.creation.box((9.0, 8.0, 3.5), trimesh.transformations.translation_matrix([0, -49, 7]))
    body = trimesh.boolean.difference([body, slot], engine="manifold")
    rear = trimesh.boolean.intersection([body, solid(box(-60, -60, 60, 60), -1, 7)], engine="manifold")
    front = trimesh.boolean.intersection([body, solid(box(-60, -60, 60, 60), 7, 15)], engine="manifold")
    return {"rear": rear, "front": front}


def test_band_keeps_the_exterior():
    shells = _shells()
    ext = F.opening_extent(list(shells.values()), (4.0, 10.0), [0, -60], [0, 1])
    assert ext is not None and 3.5 < -ext[0] < 5.5 and 3.5 < ext[1] < 5.5, ext     # the slot is 9 mm wide
    from shapely.geometry import box
    keep = [box(ext[0] - 3.0, -70, ext[1] + 3.0, -40)]
    r = F.cut_band(shells, z=(4.0, 10.0), wall=1.2, splits=[0.3, 0.7], ties=[0.1], feed=0.2, behind=0.8,
                   keep=0.3, keep_clear=keep)
    rep = r["report"]
    assert rep["ok"], rep["parts"]
    assert rep["exterior"]["pass"] and rep["exterior"]["maxDeviation"] <= 0.05, rep["exterior"]
    bands = [k for k in r["parts"] if k.startswith("band-")]
    assert len(bands) == 3, bands                       # two splits and the keep-clear zone: three pieces
    # the air is behind the strip, inside the wall: no shell lost more than the ring could hold
    vol0 = sum(m.volume for m in shells.values())
    vol1 = sum(r["parts"][k].volume for k in r["parts"])
    assert abs(vol1 - vol0) < 0.1 * vol0, (vol0, vol1)
    assert rep["air_layers"] and all(b > a for a, b, _ in rep["air_layers"])
    # every section of each shell through the band is free of slivers (pieces of no area)
    for k, zs in (("rear", (4.5, 6.0)), ("front", (8.0, 9.5))):
        for z in zs:
            p = F.section(r["parts"][k], z)
            assert not p.is_empty, (k, z)
            assert not any(g.area < 0.05 for g in getattr(p, "geoms", [p])), (k, z)
    return r


def test_hand_tissue_is_ctia():
    assert F.hand_tissue(835) == (30.3, 0.59)
    e, s = F.hand_tissue(2450)
    assert (e, s) == (25.7, 1.32)
    e, s = F.hand_tissue(1700)                           # between the 1575 and 1800 rows
    assert 27.0 < e < 27.5 and 0.90 < s < 0.99


def test_band_scene_builds(r=None):
    r = r or test_band_keeps_the_exterior()
    shells = {k: r["parts"][k] for k in ("rear", "front")}
    board = [[-18, -40], [18, -40], [18, 40], [-18, 40]]
    hand = F.hand_grip(shells, f_mhz=835, fingers=(25.0, 5.0, -15.0, -35.0), palm=(-50, 20), f_ref_mhz=796.5)
    sc = F.band_scene(r["layout"], shells, "tpu", splits=[0.3, 0.7], ties=[0.1], feed=0.2,
                      grounds=[(board, 2.0, 2.8)], boxes=[((-15, -30, 2.8), (15, 0, 7.0))], behind=0.8, keep=0.3,
                      extra=hand)
    names = {s["name"] for s in sc.solids}
    assert {"band", "ground", "hand", "rear", "front"} <= names, names
    assert sc.port is not None and len(sc.wires) == 3
    hs = [s for s in sc.solids if s["name"] == "hand"]
    eps, tan = hs[0]["eps_r"], hs[0]["tan_d"]
    sigma = 2 * np.pi * 796.5e6 * 8.8541878128e-12 * eps * tan      # the conductivity the solver will use
    assert abs(sigma - 0.59) < 1e-3, sigma


def test_fixed_network_reads_detuning():
    """A network found for one impedance, read on another: on its own
    impedance it gives what the search gave; on a detuned one it delivers
    less than a network matched to that one."""
    from loadpath import antenna as A
    f = np.linspace(700e6, 900e6, 41)
    z0 = 12 + 1j * (f / 1e6 - 800) * 0.4 - 35j
    z1 = 18 + 1j * (f / 1e6 - 760) * 0.4 - 35j        # what a hand does: more loss, lower resonance
    m = A.match(f, z0, [[750, 850]], parts=2)
    same = A.match(f, z0, [[750, 850]], network=m["network"])
    assert abs(same["worst"] - m["worst"]) < 0.01, (same["worst"], m["worst"])
    moved = A.match(f, z1, [[750, 850]], network=m["network"])
    again = A.match(f, z1, [[750, 850]], parts=2)
    assert moved["worst"] < again["worst"], (moved["worst"], again["worst"])


def test_conductor_loss_monopole():
    """A 29 mm square-rod monopole on a 60 mm plate, both of a poor
    conductor (1e5 S/m) so the loss is large: the loss resistance at
    resonance against the textbook one (rod with the quarter-wave cosine
    current, plus the plate's radial spread)."""
    from loadpath import antenna as A
    P, T, H, W, sig = 60.0, 1.0, 28.0, 2.0, 1.0e5
    c = P / 2
    sc = A.Scene((2400, 2500), sweep=(1500, 3500))
    sc.add_solid("plate", metal=True, box=((0, 0, -T), (P, P, 0)))
    sc.add_solid("rod", metal=True, box=((c - W / 2, c - W / 2, 1.0), (c + W / 2, c + W / 2, 1.0 + H)), fine=0.5)
    sc.set_port([c, c, 0.0], [c, c, 1.0])
    sim = sc.build(fine=0.5, body=1.0)
    out = sim.run(np.linspace(1.5e9, 3.5e9, 101), monitor=[2.45e9], field_box=A.metal_box(sc))
    cl = A.conductor_loss(sc, {"plate": sig, "rod": sig})
    R = 2 * cl["loss"][0] / abs(out["current"][0]) ** 2
    f = 2.45e9
    Rs = np.sqrt(np.pi * f * 4e-7 * np.pi / sig)
    k, h = 2 * np.pi * f / 2.998e8, (H + 1.0) * 1e-3
    rod = Rs / (4 * W * 1e-3) * (h / 2 + np.sin(2 * k * h) / (4 * k))
    plate = Rs / (2 * np.pi) * np.log((P / 2 * 1.13) / (0.59 * W))
    assert abs(R / (rod + plate) - 1) < 0.15, (R, rod + plate)


if __name__ == "__main__":
    r = test_band_keeps_the_exterior()
    print("band:", r["report"]["parts"], r["report"]["exterior"])
    test_hand_tissue_is_ctia()
    test_band_scene_builds(r)
    test_fixed_network_reads_detuning()
    test_conductor_loss_monopole()
    print("ok")
