"""Multi-material briefs (loadpath/evals/brief_kinds/multimat.py): each gold
reference passes its own scorer, and deliveries that miss the point (one
material only, the soft pad missing, a rigid bridge through an isolator)
fail."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath.evals import briefs
from loadpath.evals.brief_kinds import multimat as M

IDS = ["grip-pad-finger-mm", "isolated-camera-mount-mm", "polyjet-fingertip-mm"]
CELLS = 20_000      # coarse: the golds keep their margins here too


def _deliver(brief, bodies, folder, **kw):
    study = briefs.gold_study(brief)
    M.write_delivery(bodies, str(folder), process=study.process, direction=study.print_direction, **kw)
    return str(folder)


@pytest.mark.parametrize("bid", IDS)
def test_gold_passes_its_scorer(bid):
    b = briefs.load(bid)
    assert b["kind"] == "multimat"
    res = M.gold(b, max_cells=CELLS)
    assert res["ok"], res.get("error")
    s = res["score"]
    assert s["safetyFactor"] >= b["pass"]["safetyFactor"]
    assert s["interfaceSafetyFactor"] >= b["pass"]["interfaceSafetyFactor"]
    assert s["mass"] <= b["pass"]["maxMass"]


def test_prompt_asks_for_material_bodies():
    text = briefs.prompt(briefs.load(IDS[0]))
    assert "part.3mf" in text and "part-<material>.stl" in text and "part.stl     the part" not in text


def test_one_material_fails(tmp_path):
    b = briefs.load("grip-pad-finger-mm")
    ref = M.reference_bodies(b)
    row = M.score(b, _deliver(b, [("pla", m) for _, m in ref], tmp_path), max_cells=CELLS)
    assert not row["pass"]
    txt = " ".join(row["failures"])
    assert "zone 'pad'" in txt and "1 material" in txt


def test_missing_pad_fails(tmp_path):
    b = briefs.load("grip-pad-finger-mm")
    ref = [(k, m) for k, m in M.reference_bodies(b) if k != "tpu"]
    row = M.score(b, _deliver(b, ref, tmp_path), max_cells=CELLS)
    assert not row["pass"]
    txt = " ".join(row["failures"])
    assert "interfaces missing" in txt and "zone 'pad'" in txt


def test_rigid_bridge_through_isolator_fails(tmp_path):
    b = briefs.load("isolated-camera-mount-mm")
    row = M.score(b, _deliver(b, [("petg", m) for _, m in M.reference_bodies(b)], tmp_path), max_cells=CELLS)
    assert not row["pass"]
    assert row["isolated"]["isolation"] is False
    assert any("rigid path" in f for f in row["failures"])


def test_stls_by_material_and_bambu_3mf_read_back(tmp_path):
    b = briefs.load("polyjet-fingertip-mm")
    ref = M.reference_bodies(b)
    stl = tmp_path / "stl"
    stl.mkdir()
    for k, m in ref:
        m.export(str(stl / f"part-{k}.stl"))
    a, src, _ = M.read_delivery(str(stl))
    bam, _, _ = M.read_delivery(_deliver(b, ref, tmp_path / "bambu", slicer="bambu"))
    for got in (a, bam):
        vols = {k: round(m.volume, 1) for k, m in got}
        assert vols == {k: round(m.volume, 1) for k, m in ref}
    assert src == "stl"
    assert M._stem_material("gripper-finger-mm-pa-cf") == "pa-cf"
    assert M._stem_material("finger_tpu95a_T2") == "tpu"


def test_soft_skin_in_rigid_material_fails_through_score_dir(tmp_path):
    b = briefs.load("polyjet-fingertip-mm")
    folder = _deliver(b, [("vero", m) for _, m in M.reference_bodies(b)], tmp_path)
    row = briefs.score_dir(b, folder)
    assert not row["pass"] and any("zone 'skin'" in f for f in row["failures"])
    assert not briefs.score_dir(b, str(tmp_path / "nothing-here"))["pass"]


def test_interlock_reads_from_geometry():
    import numpy as np

    flat = np.zeros((40, 40, 40), np.int64)
    flat[:, :, 20:] = 1
    lattice = flat.copy()
    for z in range(16, 24):      # 0.8 mm beams, alternating direction per 2 layers, in a 2 mm band
        beams = (np.arange(40) // 2) % 2 == (z // 2) % 2
        if (z // 2) % 2:
            lattice[beams, :, z] = 1 - (z >= 20)
        else:
            lattice[:, beams, z] = 1 - (z >= 20)
    solids = [("petg", None, None), ("tpu", None, None)]
    assert M._interlock_ratio(solids, flat, 0.4) < 1.2
    assert M._interlock_ratio(solids, lattice, 0.4) > M.INTERLOCK_RATIO


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
