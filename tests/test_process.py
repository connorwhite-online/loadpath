"""
Tests for the powder-bed polymer and resin process checks (processcheck.py),
the material cards and the process field. Small fixtures, under a minute.
Run: `python3 tests/test_process.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.materials import MATERIALS, material_from_spec  # noqa: E402
from loadpath.processcheck import RULES, check_process, rules_for  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def _box(lo, hi):
    b = trimesh.creation.box(extents=np.subtract(hi, lo))
    b.apply_translation(np.add(lo, hi) / 2)
    return b


def _hollow(size=30.0, wall=3.0, hole=None, open_top=False):
    outer = _box((0, 0, 0), (size, size, size))
    top = size + 1 if open_top else size - wall
    part = outer.difference(_box((wall, wall, wall), (size - wall, size - wall, top)))
    if hole:
        c = trimesh.creation.cylinder(radius=hole / 2, height=4 * wall)
        c.apply_translation((size / 2, size / 2, size - wall / 2))
        part = part.difference(c)
    return part


def test_powder_trapped_and_narrow_exits():
    """A closed hollow keeps its powder; a 2 mm hole into it is a narrow
    exit (under SLS's 4 mm escape hole); an 8 mm hole clears it."""
    r = check_process(_hollow(), "sls", material="pa12")
    assert r["escape"]["enclosedCount"] == 1 and r["trappedVolumeMm3"] > 10_000, r["escape"]
    assert not r["ok"] and any("enclosed" in w for w in r["warnings"])
    r = check_process(_hollow(hole=2.0), "sls")
    assert r["escape"]["enclosedCount"] == 0 and r["escape"]["narrowExitCount"] == 1, r["escape"]
    r = check_process(_hollow(hole=8.0), "sls")
    assert r["trappedVolumeMm3"] == 0.0 and r["ok"], r
    # MJF asks for 5 mm: a 4.5 mm hole passes SLS and fails MJF
    assert check_process(_hollow(hole=4.6), "sls")["escape"]["narrowExitCount"] == 0
    assert check_process(_hollow(hole=4.6), "mjf")["escape"]["narrowExitCount"] == 1


def test_powder_warp_panels():
    """A long thin plate warps in SLS; the same plate 4 mm thick, a short
    one, or a solid block do not."""
    r = check_process(_box((0, 0, 0), (140, 40, 1.5)), "mjf", material="pa12")
    assert r["warp"]["maxSpan"] > 130 and r["warp"]["ok"] is False, r["warp"]
    assert any("warp" in w for w in r["warnings"])
    assert check_process(_box((0, 0, 0), (140, 40, 4)), "mjf")["warp"]["maxSpan"] == 0.0
    assert check_process(_box((0, 0, 0), (60, 40, 1.5)), "mjf")["warp"]["ok"] is True
    assert check_process(_box((0, 0, 0), (40, 40, 40)), "sls")["warp"]["panels"] == []


def test_thin_walls():
    """A 0.5 mm fin is under SLS's 0.8 mm minimum; at 1.2 mm it holds."""
    base = _box((0, 0, 0), (30, 20, 4))
    thin = trimesh.boolean.union([base, _box((10, 0, 4), (10.5, 20, 14))])
    ok = trimesh.boolean.union([base, _box((10, 0, 4), (11.2, 20, 14))])
    r = check_process(thin, "sls")
    assert r["walls"]["ok"] is False and r["walls"]["p1WallMm"] < 0.8, r["walls"]
    assert check_process(ok, "sls")["walls"]["ok"] is True


def test_resin_cups_supports_and_peel():
    """A cup sealed on the plate side and open to the vat is a suction cup;
    turned over it is not (and its floor now needs supports). A big flat
    layer is flagged for peel."""
    cup = _hollow(size=30, wall=2.5, open_top=True)
    r = check_process(cup, "sla", build="+z", material="resin")
    assert r["cups"]["maxArea"] > 500 and not r["ok"], r["cups"]
    flipped = check_process(cup, "sla", build="-z", material="resin")
    assert flipped["cups"]["maxArea"] == 0.0 and flipped["ok"], flipped["cups"]
    assert flipped["supports"]["supportArea"] > 300
    big = check_process(_box((0, 0, 0), (70, 50, 3)), "dlp")
    assert big["peel"]["maxLayerArea"] > RULES["dlp"]["peelArea"] and not big["ok"]


def test_material_cards_and_mismatch():
    for name in ("pa12", "pa11", "pa12-gb", "tpu88a", "resin", "resin-tough", "resin-rigid"):
        m = MATERIALS[name]
        assert m.processes and m.E > 0 and m.density > 0
    assert material_from_spec("TPU 88A").name == "tpu88a"
    assert material_from_spec("glass nylon").name == "pa12-gb"
    # TPU's thin walls tear out in depowdering and it does not warp like nylon
    assert rules_for("mjf", "tpu88a")["minWall"] == 1.0 and rules_for("mjf", "tpu88a")["warpSpan"] is None
    r = check_process(_box((0, 0, 0), (20, 20, 20)), "sls", material=MATERIALS["pla"])
    assert any("not a sls material" in w for w in r["warnings"])


def test_process_field_and_spec():
    """dlp is a process; escapeHole overrides the card; minFeature raises
    the wall minimum; a field part works with bounds."""
    base = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [40, 10, 10]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.5, 11, 11]}}],
            "loads": [{"shape": {"type": "box", "min": [39, -1, -1], "max": [41, 11, 11]}, "force": [0, 0, -10]}],
            "goal": {"volumeFraction": 0.3}}
    s = study_from_dict({**base, "process": {"type": "dlp", "escapeHole": 6}, "material": "resin"})
    assert s.process.kind == "dlp" and s.process.escape_hole == 6.0
    assert rules_for(s.process)["escapeHole"] == 6.0
    assert rules_for(study_from_dict({**base, "process": {"type": "sls", "minFeature": 1.2}}).process)["minWall"] == 1.2

    def field(p):
        p = np.asarray(p, float)
        return np.max(np.abs(p - 15.0), axis=-1) - 10.0     # a 20 mm solid cube
    r = check_process(field, "sls", bounds=((0, 0, 0), (30, 30, 30)))
    assert r["ok"] and abs(r["volumeMm3"] - 8000) < 400, r["volumeMm3"]


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


def main():
    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(TESTS)} FAILED")
        sys.exit(1)
    print(f"\nall {len(TESTS)} passed")


if __name__ == "__main__":
    main()
