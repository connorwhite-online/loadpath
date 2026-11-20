"""A verify submodel's box stays a submodel on a fine grid. Its half-width
is eight global cells but at least 6 mm, solved 2.5x finer; when a thin
skin pins the global grid to 0.21 mm cells (verify._SHELL_CELLS), that floor
made each box 0.7-0.8M dofs (2 min of AMG-CG, ~3.6 GB, three to six a load
case): the puck enclosure's six-case drop verify ran past 2 h at 11 GB.
Boxes now shrink toward eight global cells to fit verify._SUB_MAX_DOFS;
on an ordinary grid they are the size they always were."""
import numpy as np

from loadpath import verify as V
from loadpath.fem import HexMesh


def _block(side_mm, pitch):
    n = int(round(side_mm / pitch))
    ijk = np.argwhere(np.ones((n, n, n), bool))
    return HexMesh(ijk, pitch, np.zeros(3))


def test_fine_grid_box_fits_the_budget():
    hm = _block(14.0, 0.25)              # 56^3 cells, a solid block
    center = np.full(3, 7.0)
    half = max(8.0 * hm.pitch, 6.0)      # as _verify_part sets it
    sub = hm.pitch / 2.5
    before = V._submodel_dofs(hm, center, half, sub)
    assert before > 20 * V._SUB_MAX_DOFS     # the box as it was: millions of dofs
    h = V._submodel_half(hm, center, half, sub)
    assert 8.0 * hm.pitch <= h < half
    assert V._submodel_dofs(hm, center, h, sub) <= V._SUB_MAX_DOFS


def test_ordinary_grid_box_is_unchanged():
    hm = _block(20.0, 1.0)
    center = np.full(3, 10.0)
    half = max(8.0 * hm.pitch, 6.0)
    assert V._submodel_half(hm, center, half, hm.pitch / 2.5) == half


def test_submodel_peak_builds_the_bounded_box(monkeypatch):
    hm = _block(14.0, 0.25)
    seen = []
    monkeypatch.setattr(V, "_submodel_box", lambda study, mesh, hm_, u, c, half, pitch, *a, **k:
                        seen.append(V._submodel_dofs(hm_, c, half, pitch)))
    gate = V._MemoryGate(None)
    for g in (None, gate):
        V._submodel_peak(None, None, hm, None, np.full(3, 7.0), 6.0, hm.pitch / 2.5, None, np.zeros((0, 3)),
                         0.0, gate=g)
    assert len(seen) == 2 and max(seen) <= V._SUB_MAX_DOFS
    assert gate.held == 0.0
