"""Interlocks at material interfaces (interlock.py): credited bond, the
beam lattice, and interlocked bodies from multimat.split_bodies."""
import json
import os
import sys

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import interlock as IL  # noqa: E402
from loadpath import multimat as M  # noqa: E402
from loadpath.materials import material_from_spec  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402

from test_multimat import BEAM  # noqa: E402


def test_card_gain_and_caps():
    petg, tpu = material_from_spec("petg"), material_from_spec("tpu")
    flat = {"tensile": 1.48, "shear": 0.89, "tensileZ": 1.48, "shearZ": 0.89}
    c = IL.interlocked_card(flat, petg, tpu, IL.Interlock())
    assert c["tensile"] == pytest.approx(2.5 * 1.48)
    # between layers: a quarter TPU on TPU (6.4 across layers) and half bond
    assert c["tensileZ"] == pytest.approx(min(2.5 * 1.48, 0.25 * 6.4 + 0.5 * 1.48))
    assert c["flat"]["tensile"] == 1.48
    # shallow: the gain scales with depth
    c1 = IL.interlocked_card(flat, petg, tpu, IL.Interlock(depth=1.6))
    assert 1.48 < c1["tensile"] < c["tensile"]
    # a strong bond is not cut by the cap
    strong = {"tensile": 9.0, "shear": 5.0, "tensileZ": 9.0, "shearZ": 5.0}
    assert IL.interlocked_card(strong, petg, tpu, IL.Interlock())["tensile"] == 9.0


def test_parse():
    assert IL.parse_interlock(True).beam_width == pytest.approx(0.8)
    assert IL.parse_interlock(False) is None
    assert IL.parse_interlock({"layers": 3}).layer_height == pytest.approx(0.6)
    with pytest.raises(ValueError, match="gain"):
        IL.parse_interlock({"gain": 0.5})
    with pytest.raises(ValueError, match="two beams"):
        IL.parse_interlock({"depth": 1.0})


def test_beam_lattice_interpenetrates():
    """Each material's beams form one connected woodpile through the cube."""
    il = IL.Interlock()
    s = np.array([0.2, 0.2, 0.1])
    n = (40, 40, 40)
    xyz = np.stack(np.meshgrid(*[(np.arange(k) + 0.5) * s[a] for a, k in enumerate(n)], indexing="ij"), -1)
    a = IL.beam_pattern(xyz, il, 2)
    assert 0.45 < a.mean() < 0.55
    for part in (a, ~a):
        _, ncc = ndimage.label(part)      # face connectivity
        assert ncc == 1


def test_interlocked_split():
    d = json.loads(json.dumps(BEAM))
    d["multiMaterial"]["assign"] = [{"region": "tip", "material": "tpu"}]
    d["multiMaterial"]["requirements"] = []
    d["multiMaterial"]["interlock"] = {"depth": 2.0}
    st = study_from_dict(d)
    mm = st.multi_material
    assert M.interface_card(mm, 0, 1, st.process)["tensile"] > M.interface_card(mm, 0, 1, st.process,
                                                                                  interlock=False)["tensile"]
    r = M.run_multimaterial(st, max_runs=1)
    assert r.report["interlock"]["gain"] == pytest.approx(2.5)
    bodies = dict((mm.materials[k].name, m) for k, m in r.bodies)
    assert all(m.is_watertight for m in bodies.values())
    # the TPU tip reaches into the PETG past where the smooth interface sits
    flat = M.split_bodies(r.domain, M.material_labels(r.domain, r.optimum.xphys, r.optimum.w),
                          bodies["petg"].union(bodies["tpu"]), mm.names)
    tpu_flat = [m for k, m in flat if k == 1][0]
    # the lattice moves material between bodies, never adds or loses any
    total = sum(m.volume for m in bodies.values())
    assert sum(m.volume for _, m in flat) == pytest.approx(total, rel=0.01)
    # (1.26 at this grid; 1.3 held only while _region_mesh shrank the flat
    # body by stretching boundaries off the grid corner)
    assert bodies["tpu"].area > 1.2 * tpu_flat.area


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
