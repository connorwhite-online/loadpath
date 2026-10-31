"""
Tests for the material library (material_library.py, materials.py): every
value cites a known source, the library feeds MATERIALS and thermal
conductivity, cards are selectable by name and as a base with overrides,
across-layer values set a matching process's factors (and the study or a
calibrated card overrides them), and the CLI lists the cards. Run:
`python3 -m pytest tests/test_materials.py` (a few seconds).
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from loadpath import calibration  # noqa: E402
from loadpath.material_library import LIBRARY, SOURCES, describe  # noqa: E402
from loadpath.materials import MATERIALS, material_from_spec  # noqa: E402
from loadpath.schema import lint  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402
from loadpath.thermal import CONDUCTIVITY  # noqa: E402

BRACKET = json.loads((ROOT / "loadpath/evals/cases/wall-bracket.json").read_text())


def test_every_value_is_sourced():
    for name, c in LIBRARY.items():
        v = c["values"]
        for key in ("E", "nu", "strength", "density"):
            assert key in v, (name, key)
        for key, (num, src, note) in v.items():
            assert src in SOURCES, (name, key, src)
            assert note, (name, key)
            if src not in ("estimate", "handbook", "durability", "tychanicz-2025", "dma-estimate"):
                assert SOURCES[src].url.startswith("https://"), src
        assert c["measured"] in c["processes"], name
        if "strengthZ" in v:
            assert 0 < v["strengthZ"][0] <= 1.2 * v["strength"][0], name
    # the generic names all exist and match their card
    for name in ("pla", "petg", "abs", "asa", "tpu", "pa-cf", "pa12", "pa11", "resin", "resin-tough",
                 "alsi10mg", "ti64", "316l"):
        m = MATERIALS[name]
        assert m.E == LIBRARY[name]["values"]["E"][0] and m.density == LIBRARY[name]["values"]["density"][0]
    assert CONDUCTIVITY["alsi10mg"] == 125 and CONDUCTIVITY["pla"] == 0.19
    d = describe("ti64")
    assert d["values"]["strength"]["value"] == 1000 and "EOS" in d["values"]["strength"]["source"]


def test_names_aliases_and_overrides():
    assert material_from_spec("Nylon").name == "pa12"
    assert material_from_spec("aluminium").strength == 275
    m = material_from_spec({"base": "petg", "strength": 40})
    assert (m.E, m.strength, m.density, m.strength_z) == (1500, 40, 1.27, 18)
    assert m.name == "petg-custom" and "strength" in m.note
    plain = material_from_spec({"E": 2000, "strength": 40, "density": 1.2})
    assert plain.strength_z is None and plain.measured == ""
    with pytest.raises(ValueError):
        material_from_spec({"strength": 40})
    with pytest.raises(ValueError):
        material_from_spec({"base": "unobtainium"})
    doc = copy.deepcopy(BRACKET)
    doc["material"] = {"base": "petg", "strength": 40}
    assert not [x for x in lint(doc) if x["level"] == "error"]
    doc["material"] = {"strength": 40}
    assert any("base" in x["message"] for x in lint(doc) if x["level"] == "error")


def test_across_layer_values_feed_the_process():
    doc = copy.deepcopy(BRACKET)
    doc["material"], doc["process"] = "petg", {"type": "fdm"}
    s = study_from_dict(doc)
    assert s.process.interlayer == pytest.approx(18 / 47)
    doc["material"] = "abs"
    s = study_from_dict(doc)
    assert s.process.interlayer == pytest.approx(25.4 / 33.3) and s.process.ez == pytest.approx(1835 / 2174)
    # the study's own value wins
    doc["process"] = {"type": "fdm", "interlayer": 0.5}
    assert study_from_dict(doc).process.interlayer == 0.5
    # a card measured on MJF doesn't set an FDM study's factors
    doc["material"], doc["process"] = "pa12", {"type": "fdm"}
    assert study_from_dict(doc).process.interlayer == 0.6
    doc["process"] = {"type": "mjf"}
    assert study_from_dict(doc).process.interlayer == 1.0


def test_calibrated_card_overrides_library(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", str(tmp_path))
    card = {"name": "petg", "basedOn": "petg",
            "material": {"name": "petg", "E": 1800, "nu": 0.38, "strength": 41, "density": 1.25},
            "process": {"type": "fdm", "interlayer": 0.55}}
    (tmp_path / "petg.json").write_text(json.dumps(card))
    m = material_from_spec("petg")
    assert (m.E, m.strength, m.processes) == (1800, 41, ("fdm",))
    doc = copy.deepcopy(BRACKET)
    doc["material"], doc["process"] = "petg", {"type": "fdm"}
    assert study_from_dict(doc).process.interlayer == 0.55


def test_cli_lists_cards():
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-m", "loadpath", "materials"], capture_output=True, text=True,
                         env=env, check=True)
    rows = json.loads(out.stdout)["library"]
    assert {r["name"] for r in rows} == set(LIBRARY)
    out = subprocess.run([sys.executable, "-m", "loadpath", "materials", "steel"], capture_output=True,
                         text=True, env=env, check=True)
    assert json.loads(out.stdout)["values"]["E"]["value"] == 193000


def test_thermal_tables_come_from_the_cards():
    from loadpath.thermal import EXPANSION, MODULUS_RETENTION, retention

    assert EXPANSION["alsi10mg"] == 21e-6 and EXPANSION["resin-rigid"] == 47e-6
    for name, pts in MODULUS_RETENTION.items():
        assert pts[0] == (23, 1.0), name
        assert all(b[1] <= a[1] for a, b in zip(pts, pts[1:])), name     # never stiffer when warmer
        hdt = LIBRARY[name]["values"].get("hdt")
        if hdt:
            g, _ = retention(name, [hdt[0]])
            assert g[0] < 0.9, (name, g)     # well softened by its own HDT
