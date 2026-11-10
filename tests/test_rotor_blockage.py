"""Structure in a rotor's stream (aero.rotor_blockage): drag of what the
disk's air meets, from momentum theory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from loadpath.aero import rotor_blockage


def _bar(z, length=40.0, width=4.0, x0=5.0):
    b = trimesh.creation.box(extents=[length, width, 4.0])
    b.apply_translation([x0 + length / 2, 0, z])
    return b


def test_bar_under_a_prop_drags_as_momentum_theory_says():
    # a 4 x 40 mm bar 10 mm under a 45 mm disk at 0.78 N, hub 10 mm
    r = rotor_blockage(_bar(-12.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0)
    v_i = np.sqrt(0.78 / (2 * 1.2 * np.pi * 0.045 ** 2))
    v = v_i * (1 + 10 / np.hypot(45, 10))
    expect = 1.2 * 0.5 * 1.2 * v * v * (35 * 4) * 1e-6
    assert abs(r["downstream"]["blockedArea"] - 140) < 8, r
    assert abs(r["drag"] - expect) < 0.08 * expect, (r, expect)
    assert r["upstream"]["blockedArea"] == 0
    # far behind it meets about twice the speed, four times the pressure;
    # ahead of the disk much less than at it
    far = rotor_blockage(_bar(-200.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0, reach=300)
    near = rotor_blockage(_bar(-2.1), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0)
    assert 3.0 < far["drag"] / near["drag"] < 4.2, (far["drag"], near["drag"])
    ahead = rotor_blockage(_bar(30.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0)
    assert ahead["upstream"]["drag"] < 0.3 * near["drag"]
    # in a duct the stream leaves at sqrt(2) v_i: twice the open disk's pressure
    duct = rotor_blockage(_bar(-2.1), [0, 0, 0], [0, 0, -1], 45.0, 0.78, hub=10.0, ducted=True)
    assert 1.7 < duct["drag"] / near["drag"] < 2.2


def test_only_the_nearest_material_counts_and_off_disk_is_free():
    two = trimesh.util.concatenate([_bar(-12.0), _bar(-30.0)])
    one = rotor_blockage(_bar(-12.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78)
    both = rotor_blockage(two, [0, 0, 0], [0, 0, -1], 45.0, 0.78)
    assert abs(both["drag"] - one["drag"]) < 1e-9
    off = rotor_blockage(_bar(-12.0, x0=60.0), [0, 0, 0], [0, 0, -1], 45.0, 0.78)
    assert off["drag"] == 0.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
