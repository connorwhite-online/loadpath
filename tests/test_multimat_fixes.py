"""Multi-material fixes: material boundaries where the labels put them, a
thin pad's bond checked (not dropped by the contact band), the fraction
search above the preserved share, status/partial/progress from `loadpath
run`, and the 3MF's coordinates."""
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadpath import multimat as M  # noqa: E402
from loadpath.domain import build_domain  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

PAD = {
    "name": "pad",
    "designSpace": [{"type": "box", "min": [0, 0, 0], "max": [40, 8, 6]}],
    "preserve": [{"id": "wall", "shape": {"type": "box", "min": [0, 0, 0], "max": [2, 8, 6]}},
                 {"id": "pad", "shape": {"type": "box", "min": [20, 6, 0], "max": [40, 8, 6]}}],
    "supports": [{"region": "wall"}],
    "loadCases": [{"name": "grip", "loads": [{"shape": {"type": "box", "min": [20, 7.8, 0], "max": [40, 8, 6]},
                                              "force": [-8, -20, 4]}]}],
    "process": {"type": "fdm", "buildDirection": "+z"},
    "goal": {"safetyFactor": 2.0},
    "resolution": {"maxElements": 2500},
    "maxIterations": 16,
    "multiMaterial": {"materials": ["pla", "tpu"], "assign": [{"region": "pad", "material": "tpu"}],
                      "interfaceSafetyFactor": 2.0},
}


def _pad(**over):
    d = json.loads(json.dumps(PAD))
    d.update(over)
    return d


def test_region_mesh_puts_boundaries_where_the_labels_change():
    # a two-cell slab of material 1 across y: its body is two cells thick, at its cells
    st = study_from_dict(_pad(resolution={"maxElements": 4000}))
    dom = build_domain(st)
    j = dom.mesh.elem_ijk[:, 1]
    mid = int(np.median(j))
    lab = np.where((j == mid) | (j == mid + 1), 1, 0)
    m = M._region_mesh(dom, lab, 1)
    c = dom.mesh.elem_centers()[:, 1]
    lo = 0.5 * (c[j == mid - 1].mean() + c[j == mid].mean())
    hi = 0.5 * (c[j == mid + 1].mean() + c[j == mid + 2].mean())
    assert m.bounds[0][1] == pytest.approx(lo, abs=0.1 * dom.pitch)
    assert m.bounds[1][1] == pytest.approx(hi, abs=0.1 * dom.pitch)
    assert m.bounds[1][1] - m.bounds[0][1] == pytest.approx(2 * dom.pitch, abs=0.15 * dom.pitch)
    # and the complement meets it there, from both sides of the grid
    rest = M._region_mesh(dom, lab, 0)
    xs = rest.vertices[np.abs(rest.vertices[:, 1] - lo) < 0.05 * dom.pitch]
    assert len(xs) > 0


def _pad_bodies():
    core = trimesh.creation.box(bounds=[[0, 0, 0], [40, 6, 6]])
    pad = trimesh.creation.box(bounds=[[20, 6, 0], [40, 8, 6]])
    return [(0, core), (1, pad)]


@pytest.mark.parametrize("band", [1.5, 4.0])
def test_thin_pad_bond_is_checked_inside_the_contact_band(band):
    st = study_from_dict(_pad())
    rep = M.verify_multimaterial(st, st.multi_material, _pad_bodies(), max_cells=12000, min_pitch=0.5,
                                 contact_band=band)
    case = rep["cases"][0]
    assert "pla+tpu" in case["interfaces"], "the bond of a pad thinner than the band went unchecked"
    assert "tpu" in case["materials"]
    assert np.isfinite(rep["interfaceSafetyFactor"])
    assert "contactBand" in case["interfaces"]["pla+tpu"]
    assert any("contact band" in w for w in rep["warnings"])
    assert not rep.get("unchecked")


def test_wide_body_keeps_the_full_band():
    st = study_from_dict(_pad())
    rep = M.verify_multimaterial(st, st.multi_material, _pad_bodies(), max_cells=12000, min_pitch=0.5,
                                 contact_band=0.5)
    assert "contactBand" not in rep["cases"][0]["interfaces"]["pla+tpu"]
    assert not rep.get("warnings")


def test_search_starts_above_preserved_share_and_unchecked_never_passes(monkeypatch):
    # the wall and pad are a large share of this design space
    d = _pad()
    d["preserve"][0]["shape"]["max"] = [14, 8, 6]
    d["preserve"][1]["shape"]["min"] = [20, 3, 0]
    st = study_from_dict(d)
    share = float(np.mean(build_domain(st).preserve))
    assert share > 0.3
    real = M.verify_multimaterial

    def fake(*a, **k):
        rep = real(*a, **k)
        rep["unchecked"] = ["pla+tpu"]
        return rep

    monkeypatch.setattr(M, "verify_multimaterial", fake)
    events = []
    r = M.run_multimaterial(st, on_progress=events.append, max_runs=1)
    vf = r.outcomes[0]["volumeFraction"]
    assert vf > share
    assert any(e.get("stage") == "warning" and "preserved" in e["message"] for e in events)
    assert any("preserved" in w for w in r.report["warnings"])
    assert r.report["passes"] is False
    out = [e for e in events if e.get("stage") == "outcome"]
    assert out and out[0]["mass"] > 0 and "safetyFactor" in out[0] and "interfaceSafetyFactor" in out[0]
    # the same process checks a single-material part gets
    assert isinstance(r.report["print"], dict) and r.report["print"]


def _model(path):
    with zipfile.ZipFile(path) as z:
        return z.read("3D/3dmodel.model").decode()


@pytest.mark.parametrize("slicer", ["all", "bambu"])
def test_3mf_coordinates_bed_or_study(tmp_path, slicer):
    from loadpath.materials import material_from_spec

    bodies = [{"name": f"b {k}", "mesh": m, "material": material_from_spec(n)}
              for (k, m), n in zip(_pad_bodies(), ("pla", "tpu"))]
    bed = M.write_bodies_3mf(tmp_path / "bed.3mf", bodies, direction=(2, 1), slicer=slicer)
    assert bed["coordinates"] == "bed"
    tf = np.array(bed["transform"])
    assert tf[0, 3] == pytest.approx(128 - 20) and tf[1, 3] == pytest.approx(128 - 4)
    keep = M.write_bodies_3mf(tmp_path / "keep.3mf", bodies, direction=(2, 1), slicer=slicer, coordinates="study")
    assert keep["coordinates"] == "study" and np.allclose(keep["transform"], np.eye(4))
    build = re.search(r"<build\b.*?</build>", _model(keep["path"]), re.S).group(0)
    assert re.findall(r'transform="([^"]*)"', build) == ["1 0 0 0 1 0 0 0 1 0 0 0"]
    assert "1 0 0 0 1 0 0 0 1 108" in re.search(r"<build\b.*?</build>", _model(bed["path"]), re.S).group(0)
    with pytest.raises(ValueError):
        M.write_bodies_3mf(tmp_path / "x.3mf", bodies, coordinates="plate")


def test_cli_run_writes_status_partial_and_progress(tmp_path):
    d = _pad(goal={"volumeFraction": 0.6})
    study = tmp_path / "pad.json"
    study.write_text(json.dumps(d))
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=os.path.abspath(ROOT), LOADPATH_NO_HINT="1", OMP_NUM_THREADS="2")
    env.pop("LOADPATH_PARTIAL_DIR", None)
    p = subprocess.run([sys.executable, "-m", "loadpath", "run", str(study), "--out", str(out),
                        "--3mf-coordinates", "study"], capture_output=True, text=True, env=env, timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done" and os.path.exists(status["report"])
    assert (out / "partial" / "outcomes.jsonl").exists()
    rows = [json.loads(x) for x in (out / "partial" / "outcomes.jsonl").read_text().splitlines()]
    assert rows and set(rows[0]["stl"]) == {"pla", "tpu"}
    evs = [json.loads(x) for x in p.stderr.splitlines() if x.startswith("{")]
    oc = [e for e in evs if e.get("stage") == "outcome"]
    assert oc and oc[0]["mass"] > 0 and "materials" in oc[0] and "interfaces" in oc[0]
    summary = json.loads(p.stdout)
    assert summary["3mf"]["coordinates"] == "study"
    rep = json.loads((out / "report.json").read_text())
    assert rep["3mf"]["coordinates"] == "study"
    assert rep["print"] is not None


def test_cli_status_failed(tmp_path):
    # a failure inside the run leaves status.json saying so
    d = _pad(goal={"volumeFraction": 0.6})
    d["multiMaterial"]["requirements"] = [{"case": "grip", "knock": {"energy": 0.05, "maxForce": 300}}]
    study = tmp_path / "pad.json"
    study.write_text(json.dumps(d))
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=os.path.abspath(ROOT), LOADPATH_NO_HINT="1", OMP_NUM_THREADS="2")
    p = subprocess.run([sys.executable, "-m", "loadpath", "run", str(study), "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode != 0
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "failed" and "softness requirement" in status["error"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
