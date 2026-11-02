"""amgfast: the GIL-free Gauss-Seidel gives pyamg's numbers bit for bit."""
import numpy as np
import pytest

pyamg = pytest.importorskip("pyamg")
numba = pytest.importorskip("numba")

import scipy.sparse as sp
from pyamg.relaxation import relaxation as rel

from loadpath import amgfast


def _mats():
    poisson = pyamg.gallery.poisson((14, 12, 9), format="csr")
    elastic, _ = pyamg.gallery.linear_elasticity((14, 14), format="bsr")
    r = sp.random(900, 900, density=0.01, random_state=1)
    blocks = ((r + r.T) + sp.eye(900) * 5).tobsr(blocksize=(3, 3))
    return [poisson, elastic, blocks]


def _sweeps(mats):
    rng = np.random.default_rng(0)
    out = []
    for m in mats:
        for sweep in ("forward", "backward", "symmetric"):
            x = rng.standard_normal(m.shape[0])
            b = rng.standard_normal(m.shape[0])
            rel.gauss_seidel(m, x, b, iterations=2, sweep=sweep)
            out.append(x)
            y = rng.standard_normal(m.shape[0])
            bs = m.blocksize[0] if sp.issparse(m) and m.format == "bsr" else 2
            rel.block_gauss_seidel(m, y, b, iterations=2, sweep=sweep, blocksize=bs)
            out.append(y)
    return out


def test_same_bits_as_pyamg(monkeypatch):
    mats = _mats()
    core = rel.amg_core
    raw = core._core if isinstance(core, amgfast._Core) else core
    monkeypatch.setattr(rel, "amg_core", raw)
    ref = _sweeps(mats)
    monkeypatch.setattr(rel, "amg_core", amgfast._Core(raw))
    new = _sweeps(mats)
    for a, b in zip(ref, new):
        np.testing.assert_array_equal(a, b)


def test_install_is_idempotent():
    assert amgfast.install()
    core = rel.amg_core
    assert amgfast.install()
    assert rel.amg_core is core and isinstance(core, amgfast._Core)
    assert amgfast.installed()
