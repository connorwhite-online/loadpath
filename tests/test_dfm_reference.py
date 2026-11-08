"""Materialize is the DFM reference for printed parts: loadpath's process
defaults never go under its published minimum walls, and a run's report
carries the reference (dfm_reference)."""
import re

import trimesh

from loadpath import dfm_reference as ref
from loadpath.material_library import LIBRARY, describe
from loadpath.processcheck import RULES, check_process, rules_for
from loadpath.sizing import PROCESS_LIMITS
from loadpath.spec import PROCESSES


def test_defaults_not_under_materialize():
    for kind, lim in ref.PROCESSES.items():
        assert PROCESSES[kind]["minFeature"] >= lim["minWall"], kind
        if kind in RULES:
            assert RULES[kind]["minWall"] >= lim["minWall"], kind
        if kind in PROCESS_LIMITS:
            assert PROCESS_LIMITS[kind]["minWall"] >= lim["minWall"], kind


def test_tpu_keeps_its_margin_with_the_hard_limit_on_the_card():
    card = rules_for("sls", "tpu88a")
    assert card["minWall"] == 1.0 and card["materialize"]["minWall"] == 0.8
    assert "0.8" in card["minWallWhy"]


def test_materials_map_to_listed_cards_and_processes():
    uuid = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    for name, per in ref.MATERIALS.items():
        assert name in LIBRARY, name
        for proc, m in per.items():
            assert proc in LIBRARY[name]["processes"], (name, proc)
            assert uuid.match(m["id"]) and len(m["buildVolume"]) == 3 and m["accuracy"] > 0
    assert describe("pa12")["materialize"]["sls"]["material"]["id"] == "03a417a3-d9c9-5717-ab7f-3966a8ad4555"
    assert ref.limits("volumetric") is None and ref.limits("milling") is None


def test_review_flags_thin_floors_and_oversize_parts():
    r = ref.review("fdm", "petg", extents=(100, 50, 20), wall_floor=1.0)
    assert r["fitsBuildVolume"] and not r["warnings"]
    r = ref.review("fdm", "petg", extents=(100, 50, 20), wall_floor=0.8)
    assert any("1 mm minimum wall" in w for w in r["warnings"])
    r = ref.review("mjf", "pa12", extents=(450, 100, 100))
    assert r["fitsBuildVolume"] is False and r["warnings"]
    r = ref.review("lpbf", "in625")
    assert "material" not in r and "not listed" in r["note"]


def test_check_process_carries_the_reference():
    box = trimesh.creation.box((20, 20, 10))
    rep = check_process(box, "sls", material="pa12")
    assert rep["materialize"]["material"]["name"] == "SLS Nylon PA12"
    assert rep["materialize"]["fitsBuildVolume"] is True
