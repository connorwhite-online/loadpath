"""
Tests for the interface library (interface_library.py): lookups are
symmetric and take aliases and Material objects, an unknown pair gets the
flagged conservative default, a material with itself is refused, and every
value cites a known source. Run: `python3 -m pytest tests/test_interfaces.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from loadpath.interface_library import (DEFAULT_FRACTION, INTERFACES, SOURCES,  # noqa: E402
                                        describe_interface, interface_strength)
from loadpath.material_library import LIBRARY  # noqa: E402
from loadpath.materials import MATERIALS, material_from_spec  # noqa: E402


def test_lookup_is_symmetric():
    ab, ba = interface_strength("petg", "tpu"), interface_strength("tpu", "petg")
    assert ab == ba and ab.pair == ("petg", "tpu")
    assert ab.tensile > 0 and ab.shear > 0 and ab.bond == "good" and not ab.conservative
    assert interface_strength("pla", "petg").bond == "none"


def test_aliases_objects_and_custom_names():
    assert interface_strength("flex", "PETG") == interface_strength("tpu", "petg")
    assert interface_strength("VeroWhite", "agilus 30").pair == ("agilus30", "vero")
    assert interface_strength(MATERIALS["pla"], MATERIALS["tpu"]).tensile == INTERFACES["pla+tpu"]["tensile"][0]
    custom = material_from_spec({"base": "petg", "strength": 40})
    assert interface_strength(custom, "tpu").pair == ("petg", "tpu")


def test_unknown_pair_is_conservative_and_flagged():
    c = interface_strength("pla", "abs")
    assert c.conservative and c.source == "default" and c.bond == "unknown"
    weak = min(MATERIALS["pla"].strength_z, MATERIALS["abs"].strength_z)
    assert c.tensile == pytest.approx(DEFAULT_FRACTION * weak) and c.shear < c.tensile
    assert c.tensile < 0.1 * weak
    # a measured pair asked for by another process falls back to the default
    assert interface_strength("petg", "tpu", process="jetting").source == "default"
    assert describe_interface("pla", "abs")["conservative"]


def test_same_material_is_not_an_interface():
    with pytest.raises(ValueError):
        interface_strength("petg", "petg")
    with pytest.raises(ValueError):
        interface_strength("tpu", "flex")
    with pytest.raises(ValueError):
        interface_strength("unobtainium", "pla")


def test_every_value_is_sourced():
    for key, e in INTERFACES.items():
        a, b = key.split("+")
        assert a < b and a in LIBRARY and b in LIBRARY, key
        assert e["bond"] in ("good", "fair", "poor", "none") and e["note"], key
        assert e["process"] in LIBRARY[a]["processes"] and e["process"] in LIBRARY[b]["processes"], key
        for k in ("tensile", "shear", "tensileZ", "shearZ"):
            if k not in e:
                assert k.endswith("Z"), (key, k)
                continue
            num, src, note = e[k]
            assert num > 0 and note, (key, k)
            assert src in SOURCES or src == "estimate", (key, k, src)
            if src not in ("estimate", "default"):
                assert SOURCES[src].url.startswith("https://"), src
        assert e.get("bondSource", "estimate") in SOURCES, key
        # a boundary is never stronger than the weaker material's in-plane strength
        weak = min(LIBRARY[a]["values"]["strength"][0], LIBRARY[b]["values"]["strength"][0])
        assert e["tensile"][0] <= weak, key
    d = describe_interface("tpu", "pla")
    assert d["values"]["shearZ"]["value"] == 0.37 and "Brancewicz" in d["values"]["shearZ"]["source"]
