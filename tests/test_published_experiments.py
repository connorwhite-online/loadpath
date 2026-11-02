"""
The published-experiment validation set (loadpath/validation.py,
loadpath/data/published_experiments.json): every specimen traces to a
source, its model builds, and loadpath's readings stay where
docs/VALIDATION.md says they are. A change to the lattice tables or the
homogenized check that moves these numbers has to move the docs too.

    python -m pytest -q tests/test_published_experiments.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loadpath.validation import (load_experiments, measured, specimen_model,  # noqa: E402
                                 table_prediction)

RESULTS = ROOT / "bench" / "results" / "published-experiments.json"


def _specimens(source=None):
    d = load_experiments()
    return d, [s for s in d["specimens"] if source is None or s["source"] == source]


def test_every_specimen_is_sourced():
    d, specs = _specimens()
    assert len(specs) >= 30
    ids = [s["id"] for s in specs]
    assert len(set(ids)) == len(ids)
    for s in specs:
        src = d["sources"][s["source"]]
        assert src["doi"] and src["citation"] and src["url"].startswith("https://")
        mat = d["materials"][s["material"]]
        assert mat["E"] > 0 and mat["strength"] > 0 and mat["basis"]
        m = measured(s)
        assert m["E"] > 0 and m["strength"] > 0


def test_table_density_matches_the_papers():
    """Sheets are matched to the paper's relative density; Shevchenko's
    0.8 mm walls in 10 mm cells must come out near 0.8 mm (the diamond's
    stated 0.33 needs a thicker wall: their level-set offset is not a
    distance)."""
    _, specs = _specimens("shevchenko2023")
    for s in specs:
        t = table_prediction(s, {"E": 1.0, "strength": 1.0})
        assert abs(t["rho"] - s["lattice"]["rho"]) < 0.01
        assert 0.75 < t["thickness"] < 0.95, (s["id"], t["thickness"])


def test_table_readings_against_measurements():
    """The documented bands (docs/VALIDATION.md, published experiments):
    the periodic tables read Shevchenko's sheet TPMS 0.7-2.1x the measured
    modulus, Guerra Silva's FDM strut lattices 1.4-4.5x and Abou-Ali's SLS
    lattices 0.8-2.8x; first yield reads under the measured plateau for
    every FDM strut lattice but one."""
    d, specs = _specimens()
    ratios = {}
    for s in specs:
        t = table_prediction(s, d["materials"][s["material"]])
        m = measured(s)
        ratios[s["id"]] = (t["E"] / m["E"], t["firstYield"] / m["strength"])
    shev = [r for k, r in ratios.items() if k.startswith("shev")]
    assert all(0.7 < e < 2.1 for e, _ in shev), shev
    gs = [r for k, r in ratios.items() if k.startswith("gs-")]
    assert all(1.3 < e < 4.6 for e, _ in gs), gs
    assert sum(y > 1.0 for _, y in gs) <= 1
    aa = [r for k, r in ratios.items() if k.startswith("aa-")]
    assert all(0.8 < e < 2.8 for e, _ in aa), aa


def test_specimen_model_builds():
    from loadpath.spec import study_from_dict

    d, specs = _specimens()
    for s in specs[:3] + specs[9:11]:
        doc, sd, area, height = specimen_model(s, d["materials"][s["material"]])
        st = study_from_dict(sd)
        assert len(st.supports) == 2 and st.supports[1].fix == (True, True, False)
        assert area == s["size"][0] * s["size"][1] and height == s["size"][2]
        assert doc["nodes"]["lat"]["type"] == s["lattice"]["type"]


def test_homogenized_specimen_close_to_table():
    """A quick homogenized check of the gyroid specimen (few cells) lands
    within 30% of the periodic reading: the specimen model loads the
    lattice the way the test does."""
    from loadpath.homogenized import homogenized_check
    from loadpath.spec import study_from_dict

    d, specs = _specimens("shevchenko2023")
    s = next(x for x in specs if x["id"] == "shev-sls-gyroid")
    mat = d["materials"][s["material"]]
    doc, sd, area, height = specimen_model(s, mat)
    r = homogenized_check(doc, study_from_dict(sd), cells=8000, contact_band=10.0)
    e = area / area * height / (r["cases"][0]["deflection"] - 2.0 / mat["E"])
    t = table_prediction(s, mat)
    assert 0.7 < e / t["E"] < 1.3, (e, t["E"])


def test_saved_results_match_the_tables():
    """bench/results/published-experiments.json was written by the current
    tables (rerun bench/published_experiments.py when they change)."""
    if not RESULTS.exists():
        return
    saved = json.loads(RESULTS.read_text())
    d, specs = _specimens()
    for s in specs:
        if s["id"] not in saved:
            continue
        t = table_prediction(s, d["materials"][s["material"]])
        assert np.isclose(saved[s["id"]]["table"]["E"], t["E"], rtol=0.01), s["id"]


def test_saved_results_documented_bands():
    """The numbers docs/VALIDATION.md quotes from the saved run: the
    homogenized specimen check reads 0.78-1.06x the resolved designed
    geometry; against the tests it reads too stiff on all but one specimen
    and its first-yield strength under the measurement on all but one."""
    if not RESULTS.exists():
        return
    saved = json.loads(RESULTS.read_text())
    hom_res = [v["homogenized"]["E"] / v["resolved"]["E"] for v in saved.values() if "resolved" in v]
    assert len(hom_res) >= 30 and 0.78 < min(hom_res) and max(hom_res) < 1.06
    e = [v["homogenized"]["E"] / v["measured"]["E"] for v in saved.values()]
    y = [v["homogenized"]["firstYield"] / v["measured"]["strength"] for v in saved.values()]
    assert sum(r < 1.0 for r in e) <= 3 and max(e) < 4.5
    assert sum(r > 1.0 for r in y) <= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
