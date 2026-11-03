"""
As-printed lattice stiffness (lattice_print.py) and the lattice buckling
check (lattice_buckling.py, homogenized.py):

- the shipped fit is the fit of bench/results/published-experiments.json,
  and its held-out errors are where docs/VALIDATION.md says;
- the factor reaches the homogenized check through the study's process,
  process.latticeStiffness and a calibrated card, and reads strain-based
  (deflection and wall stress both scale by 1/k on a pure lattice);
- the periodic buckling solve matches Euler on a cubic strut cell.

    python -m pytest -q tests/test_lattice_print.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from loadpath.lattice_print import PRINT_FIT, fit, fit_rows, print_factor  # noqa: E402

RESULTS = ROOT / "bench" / "results" / "published-experiments.json"
FIT = ROOT / "bench" / "results" / "lattice-print-fit.json"


def test_shipped_fit_matches_the_data():
    got = fit(fit_rows(json.loads(RESULTS.read_text())))
    for g in ("fdm", "sls", "resin"):
        assert abs(got["intercept"][g] - PRINT_FIT["intercept"][g]) < 1e-3, g
    assert abs(got["strut"] - PRINT_FIT["strut"]) < 1e-3
    assert abs(got["densitySlope"] - PRINT_FIT["densitySlope"]) < 1e-3


def test_held_out_bands():
    """docs/VALIDATION.md: designed geometry median error 1.68x (worst
    4.1x); with the factor, held out one at a time, 1.23x (worst 2.04x);
    each paper held out reads a median 0.9-1.2x; as-printed first yield is
    under every measured strength."""
    import lattice_print_fit

    res = lattice_print_fit.main(RESULTS, FIT)
    assert res["designed"]["medianError"] > 1.6
    assert res["leaveOneOut"]["medianError"] < 1.3
    assert res["leaveOneOut"]["worstError"] < 2.1
    for src, s in res["leaveSourceOut"].items():
        assert 0.85 < s["median"] < 1.25, src
    assert res["strength"]["designed"]["max"] > 1.1           # the PLA Schwarz P miss
    assert res["strength"]["asPrintedHeldOut"]["max"] < 0.95
    # struts: the stress reading holds with either strut paper held out
    st = res["strainShare"]["strut"]
    assert st["shipped"] == 0.0 and all(v["share"] == 0.0 and v["max"] < 0.95 for v in st["sourceOut"].values())
    # FDM struts: tighter than the strain reading's 0.07-0.41x
    fdm_struts = [p["asPrintedFirstYieldHeldOut"] for p in res["specimens"].values() if p["source"] == "guerrasilva2021"]
    assert min(fdm_struts) > 0.15 and max(fdm_struts) < 0.8


def test_factor_by_process():
    sls = print_factor("sls", "gyroid", 0.2)[0]
    mjf = print_factor("mjf", "gyroid", 0.2)[0]
    fdm_sheet = print_factor("fdm", "gyroid", 0.2)[0]
    fdm_strut = print_factor("fdm", "kelvin", 0.2)[0]
    assert sls == mjf and 0.85 < sls <= 1.0
    assert 0.6 < fdm_sheet < 0.7
    assert 0.35 < fdm_strut < 0.45
    assert print_factor("dlp", "diamond", 0.2)[0] == print_factor("sla", "diamond", 0.2)[0]
    # denser lattices are the more over-read; capped at the designed stiffness
    assert print_factor("fdm", "gyroid", 0.35)[0] < fdm_sheet
    assert print_factor("sls", "gyroid", 0.05)[0] == 1.0
    assert print_factor("lpbf", "gyroid", 0.2) is None and print_factor(None, "gyroid", 0.2) is None


def _specimen(sid="shev-fdm-primitive", process=None):
    from loadpath.spec import study_from_dict
    from loadpath.validation import load_experiments, specimen_model

    d = load_experiments()
    spec = next(s for s in d["specimens"] if s["id"] == sid)
    doc, sd, _, _ = specimen_model(spec, d["materials"][spec["material"]])
    if process is not None:
        sd = {**sd, "process": process}
    return doc, study_from_dict(sd)


def test_homogenized_check_reads_the_print():
    from loadpath.homogenized import homogenized_check

    doc, st = _specimen()
    base = homogenized_check(doc, st, cells=8000, contact_band=10.0, buckling=False)
    assert base["latticePrint"]["factor"] is None and base["latticePrint"]["note"]
    doc, st = _specimen(process={"type": "fdm"})
    r = homogenized_check(doc, st, cells=8000, contact_band=10.0, buckling=False)
    k = r["latticePrint"]["factor"][0]
    assert r["latticePrint"]["group"] == "fdm" and 0.6 < k < 0.75
    c0, c1 = base["cases"][0], r["cases"][0]
    # a lattice between platens: the lattice softens by k (platens do not)
    assert 0.85 / k < c1["deflection"] / c0["deflection"] < 1.0 / k
    # strain-based wall stress: the softer lattice strains more
    assert 0.85 / k < c1["peakStress"] / c0["peakStress"] < 1.05 / k
    doc, st = _specimen(process={"type": "fdm", "latticeStiffness": "designed"})
    d = homogenized_check(doc, st, cells=8000, contact_band=10.0, buckling=False)
    assert abs(d["cases"][0]["deflection"] / c0["deflection"] - 1) < 0.02
    doc, st = _specimen(process={"type": "fdm", "latticeStiffness": 0.5})
    h = homogenized_check(doc, st, cells=8000, contact_band=10.0, buckling=False)
    assert h["latticePrint"]["factor"] == [0.5, 0.5]
    with pytest.raises(ValueError):
        _specimen(process={"type": "fdm", "latticeStiffness": 2.0})


def test_calibrated_lattice_coupon_sets_the_factor(tmp_path, monkeypatch):
    from loadpath import spec as spec_mod

    card = {"process": {"type": "fdm"}, "lattice": {"stiffnessFactor": 0.55}}
    monkeypatch.setattr("loadpath.calibration.load_card", lambda name: card)
    p = spec_mod._process({"type": "fdm"})
    p = spec_mod._calibrated_process(p, "my-petg", {"type": "fdm"})
    assert p.lattice_stiffness == 0.55
    p = spec_mod._calibrated_process(spec_mod._process({"type": "fdm", "latticeStiffness": 0.8}), "my-petg",
                                     {"type": "fdm", "latticeStiffness": 0.8})
    assert p.lattice_stiffness == 0.8


def test_unit_cell_buckling_matches_euler():
    """Simple cubic struts along z: one periodic cell holds the column's
    clamped (fixed-fixed) mode, pi^2/4 (d/L)^2 strain; a 2-cell block lets
    neighbouring nodes turn opposite ways, the pinned column, a quarter of
    it."""
    from loadpath.lattice_buckling import unit_cell_buckling

    one = unit_cell_buckling("cubic", 0.1, n=40, cells=1)
    assert abs(one["strainAtBuckling"] / (np.pi ** 2 / 4 * 0.01) - 1) < 0.05
    two = unit_cell_buckling("cubic", 0.1, n=40, cells=2)
    assert 0.2 < two["sigmaOverE"] / one["sigmaOverE"] < 0.32


def test_buckling_table_and_check():
    from loadpath.lattice_buckling import buckling_stress, has_buckling

    if not has_buckling("octet"):
        pytest.skip("no buckling table yet (bench/lattice_buckling.py)")
    b = buckling_stress("octet", [0.05, 0.1, 0.2])
    assert np.all(np.diff(b) > 0)
    from loadpath.homogenized import homogenized_check

    doc, st = _specimen("aa-octet-11", process={"type": "sls"})
    r = homogenized_check(doc, st, cells=8000, contact_band=8.0)
    c = r["cases"][0]
    assert isinstance(r["latticeBuckling"], dict)
    assert c["bucklingFactor"] > 0 and c["safetyFactor"] == min(c["yieldSafetyFactor"], c["bucklingFactor"])


def test_bench_as_printed_rows():
    """bench/published_experiments.py's asPrinted rows (the specimen
    re-checked with its held-out factor and the buckling check), as
    docs/VALIDATION.md quotes them."""
    rows = json.loads(RESULTS.read_text())
    ap = [r for r in rows.values() if "asPrinted" in r]
    assert len(ap) == len(rows) >= 33
    e = np.array([r["asPrinted"]["E"] / r["measured"]["E"] for r in ap])
    assert np.median(np.exp(np.abs(np.log(e)))) < 1.3 and e.max() < 2.1 and e.min() > 0.55
    s = np.array([r["asPrinted"]["firstYield"] / r["measured"]["strength"] for r in ap])
    assert s.max() < 0.95
    b = np.array([r["asPrinted"]["buckling"] / r["asPrinted"]["firstYield"] for r in ap])
    assert b.min() > 1.0


def test_struts_read_the_stress_they_carry():
    from loadpath.homogenized import homogenized_check

    doc, st = _specimen("aa-kelvin-14")
    base = homogenized_check(doc, st, cells=8000, contact_band=8.0, buckling=False)
    doc, st = _specimen("aa-kelvin-14", process={"type": "sls"})
    r = homogenized_check(doc, st, cells=8000, contact_band=8.0, buckling=False)
    k = r["latticePrint"]["factor"][0]
    assert k < 0.8
    c0, c1 = base["cases"][0], r["cases"][0]
    assert c1["deflection"] / c0["deflection"] > 0.85 / k
    assert 0.9 < c1["peakStress"] / c0["peakStress"] < 1.1
