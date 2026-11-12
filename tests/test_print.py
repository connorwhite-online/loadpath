"""
Tests for print-aware design: layered elasticity, the AM filter in any
direction, the slicer-style print check, orientation ranking, 3MF output.
Small fixtures, well under a minute. Run: `python3 tests/test_print.py`.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from loadpath.am import AMFilter, TiltedAMFilter, layered_elasticity, parse_build_direction  # noqa: E402
from loadpath.sdf_kit import mesh_union  # noqa: E402
from loadpath.fem import Layered, elastic_c, hex8_ke  # noqa: E402
from loadpath.printcheck import check_print, rotation_to_z  # noqa: E402
from loadpath.spec import study_from_dict  # noqa: E402


def _box(lo, hi):
    b = trimesh.creation.box(extents=np.subtract(hi, lo))
    b.apply_translation(np.add(lo, hi) / 2)
    return b


def _union(*ms):
    m = ms[0]
    for x in ms[1:]:
        m = mesh_union(m, x)
    return m


def test_layered_elasticity_moduli_and_rotation():
    """Directional moduli come out as asked, along any build axis, and the
    isotropic limit is the plain matrix."""
    assert np.abs(Layered(0.38).c() - elastic_c(1.0, 0.38)).max() < 1e-12
    for normal, ax in (((0, 0, 1), 2), ((1, 0, 0), 0), ((0, -1, 0), 1)):
        s = np.linalg.inv(Layered(0.38, 0.6, 0.7, normal).c())
        assert abs(1 / s[ax, ax] - 0.6) < 1e-9
        assert abs(1 / s[(ax + 1) % 3, (ax + 1) % 3] - 1.0) < 1e-9
    # tilted: the modulus along the normal is still ez
    n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
    c = Layered(0.35, 0.5, 0.8, n).c()
    assert np.linalg.eigvalsh(c).min() > 0
    from loadpath.fem import _to_tensor
    st = _to_tensor(np.linalg.inv(c))
    # compliance tensor from inverse Voigt with engineering shear: S_ijkl
    # picks up factors of 1/2 per shear index pair; recompute directly
    e_n = 1.0 / np.einsum("i,j,k,l,ijkl->", n, n, n, n, _voigt_s_tensor(np.linalg.inv(c)))
    assert abs(e_n - 0.5) < 1e-9, e_n
    assert st.shape == (3, 3, 3, 3)
    # the element matrix keeps rigid-body modes in its null space
    ke = hex8_ke(Layered(0.35, 0.6, 0.7, n))
    from loadpath.fem import CORNERS
    u = np.tile([1.0, 0.0, 0.0], 8)
    assert np.abs(ke @ u).max() < 1e-12
    rot = np.zeros(24)
    for i, (x, y, z) in enumerate(CORNERS):
        rot[3 * i:3 * i + 3] = [-y, x, 0.0]
    assert np.abs(ke @ rot).max() < 1e-12


def _voigt_s_tensor(s):
    """Compliance tensor from a Voigt compliance with engineering shear."""
    idx = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))
    t = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            f = (0.5 if a >= 3 else 1.0) * (0.5 if b >= 3 else 1.0)
            for p, q in {(i, j), (j, i)}:
                for r, w in {(k, l), (l, k)}:
                    t[p, q, r, w] = s[a, b] * f
    return t


_BEAM_STUDY = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [20, 10, 10]},
               "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.1, 11, 11]}}],
               "loadCases": [{"name": "t", "loads": [{"shape": {"type": "box", "min": [19.9, -1, -1],
                                                                "max": [21, 11, 11]}, "force": [0, 0, -5]}]}],
               "material": "petg", "goal": {"volumeFraction": 0.4}}


def test_process_sets_layered_design_and_physical_stiffness():
    base = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [20, 10, 10]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.1, 11, 11]}}],
            "loadCases": [{"name": "t", "loads": [{"shape": {"type": "box", "min": [19.9, -1, -1],
                                                             "max": [21, 11, 11]}, "force": [0, 0, -5]}]}],
            "material": "petg", "goal": {"volumeFraction": 0.4}}
    s = study_from_dict({**base, "process": {"type": "fdm", "buildDirection": "+y", "anisotropic": "bond"}})
    phys, design = s.elastic(), s.elastic(design=True)
    assert isinstance(phys, Layered) and isinstance(design, Layered)
    assert phys.normal == (0.0, 1.0, 0.0) and phys.ez == 0.85 and design.ez == s.process.interlayer == 18 / 47
    assert s.print_direction == (1, 1)
    iso = study_from_dict({**base, "process": {"type": "fdm"}})
    assert iso.elastic(design=True) == iso.material.nu and isinstance(iso.elastic(), Layered)
    ph = study_from_dict({**base, "process": {"type": "fdm", "anisotropic": True}}).elastic(design=True)
    assert isinstance(ph, Layered) and ph.ez == 0.85
    assert study_from_dict(base).elastic(design=True) == 0.38
    assert study_from_dict({**base, "process": "sla"}).elastic() == 0.38
    auto = study_from_dict({**base, "process": {"type": "fdm", "buildDirection": "auto"}})
    assert auto.process.build == "auto" and auto.print_direction is None
    tilt = study_from_dict({**base, "process": {"type": "fdm", "buildDirection": [0, 1, 1]}})
    assert len(tilt.print_direction) == 3 and abs(np.linalg.norm(tilt.print_direction) - 1) < 1e-12
    assert parse_build_direction([0, 0, -2]) == (2, -1)
    assert layered_elasticity(0.3, None) == 0.3


def test_tilted_am_filter_matches_axis_and_gradients():
    rng = np.random.default_rng(3)
    dims = (6, 5, 7)
    x = rng.random(dims)
    for d in ((2, 1), (0, -1), (1, 1)):
        a = AMFilter(dims, d).forward(x)
        t = TiltedAMFilter(dims, d).forward(x)
        assert np.abs(a - t).max() < 1e-12, d
    pin = np.zeros(dims, bool)
    pin[2, 2, 3] = True
    for d in ((0.0, 0.5, np.sqrt(0.75)), (0.3, -0.4, np.sqrt(0.75))):
        f = AMFilter(dims, d, pin)
        assert isinstance(f, TiltedAMFilter)
        w = rng.random(dims)
        f.forward(x)
        g = f.backward(w)
        for _ in range(6):
            i = tuple(rng.integers(0, dims))
            e = 1e-6
            xp, xm = x.copy(), x.copy()
            xp[i] += e
            xm[i] -= e
            fd = ((f.forward(xp) * w).sum() - (f.forward(xm) * w).sum()) / (2 * e)
            assert abs(fd - g[i]) < 1e-6 * max(1.0, abs(fd)), (d, i, fd, g[i])
            f.forward(x)


def test_tilted_filter_removes_unsupported_and_keeps_supported():
    """Tilted 45 degrees about x: a column along the build direction prints,
    a slab perpendicular to it held only at one end does not."""
    n = 16
    d = (0.0, -np.sqrt(0.5), np.sqrt(0.5))
    f = TiltedAMFilter((4, n, n), d)
    col = np.zeros((4, n, n))
    for k in range(n):
        j = n - 1 - k
        col[:, j, k] = 1.0
        if j - 1 >= 0:
            col[:, j - 1, k] = 1.0
    out = f.forward(col)
    assert out[col > 0].min() > 0.9
    # a thin layer parallel to the plate, far from it: nothing below it
    sheet = np.zeros((4, n, n))
    for j in range(n):
        k = j - 4
        if 0 <= k < n:
            sheet[:, j, k] = 1.0
    out2 = f.forward(sheet)
    assert out2[sheet > 0].max() < 0.1, out2[sheet > 0].max()


def test_print_check_bridges_ledges_islands_and_bed():
    bridge = _union(_box((0, 0, 0), (5, 10, 10)), _box((17, 0, 0), (22, 10, 10)), _box((0, 0, 10), (22, 10, 13)))
    r = check_print(bridge, (2, 1), pitch=0.5)
    assert r["supportFree"] and r["bridges"] == 1 and 10 <= r["maxBridgeSpan"] <= 13, r
    assert r["bed"]["feet"] == 2 and r["bed"]["comMargin"] > 0
    long_ = _union(_box((0, 0, 0), (5, 10, 10)), _box((35, 0, 0), (40, 10, 10)), _box((0, 0, 10), (40, 10, 13)))
    r = check_print(long_, (2, 1), pitch=0.5)
    assert not r["supportFree"] and r["supportArea"] > 250 and r["supportVolume"] > 2500
    # a part that needs support says so in warnings, not only in the numbers
    assert any("needs build supports" in w and "bridgeSpan" in w for w in r["warnings"]), r["warnings"]
    # the short bridge fails on a printer that only bridges 8 mm, from the study's process block
    assert not check_print(bridge, (2, 1), pitch=0.5, bridge_span=8.0)["supportFree"]
    proc = study_from_dict({**_BEAM_STUDY, "process": {"type": "fdm", "bridgeSpan": 8}}).process
    assert proc.bridge_span == 8.0
    # the same part on its side prints without support
    r = check_print(long_, (1, 1), pitch=0.5)
    assert r["supportFree"], r
    # a horizontal 8 mm bore prints (its crown is a short ledge each layer)
    hole = _box((0, 0, 0), (30, 20, 20)).difference(trimesh.creation.cylinder(
        radius=4, height=40, transform=trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0], [15, 10, 10])))
    assert check_print(hole, (2, 1), pitch=0.5)["supportFree"]
    # a cone on its tip: one tiny foot, warned
    cone = trimesh.creation.cone(radius=10, height=10)
    r = check_print(cone, (2, -1), pitch=0.5)
    assert r["bed"]["smallestFoot"] < 9 and any("foot" in w for w in r["warnings"])
    # a floating start: an L hanging off a post, printed upside down
    t = _union(_box((0, 0, 0), (6, 6, 30)), _box((0, 0, 24), (30, 6, 30)), _box((24, 0, 10), (30, 6, 30)))
    r = check_print(t, (2, -1), pitch=0.5)
    assert r["islands"] == 0  # the hook's tip grows from the bar on its way down
    r = check_print(t, (2, 1), pitch=0.5)
    assert r["islands"] >= 1, r


def test_rotation_to_z_is_proper():
    for d in ((2, 1), (2, -1), (0, 1), (1, -1), (0.3, 0.4, np.sqrt(0.75))):
        from loadpath.am import build_vector
        r = rotation_to_z(d)
        assert abs(np.linalg.det(r) - 1) < 1e-12
        assert np.allclose(r @ build_vector(d), [0, 0, 1])


def test_rank_orientations_prefers_support_free_rest():
    """An L-bracket lies flat; a T standing on its stem loses to its side."""
    from loadpath.orient import rank_orientations

    tee = _union(_box((0, 0, 0), (6, 10, 20)), _box((-12, 0, 20), (18, 10, 24)))
    rows = rank_orientations(tee)
    best = rows[0]
    assert best["print"]["supportFree"], best["direction"]
    # its stem-down orientation needs support and ranks below
    stem = next(r for r in rows if r["direction"] == "+z")
    assert not stem["print"]["supportFree"]
    assert rows.index(stem) > 0


def test_rank_orientations_reads_interlayer_tension():
    """A cantilever loaded in bending: layers along its length (build across
    it) carry the bending in-plane; layers stacked along its length put the
    root's bending tension straight across the bond."""
    from loadpath.orient import rank_orientations

    study = study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [60, 10, 10]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.1, 11, 11]}}],
        "loadCases": [{"name": "tip", "loads": [{"shape": {"type": "box", "min": [59.9, -1, -1],
                                                           "max": [61, 11, 11]}, "force": [0, 0, -40]}]}],
        "material": "petg", "process": {"type": "fdm", "buildDirection": "+z"}, "goal": {"safetyFactor": 2}})
    mesh = _box((0, 0, 0), (60, 10, 10))
    rows = rank_orientations(mesh, study, candidates=[(2, 1), (0, 1)], stress_cells=8000)
    by = {r["direction"]: r for r in rows}
    assert by["+x"]["interlayerStress"] > 5 * by["+z"]["interlayerStress"], by
    assert rows[0]["direction"] == "+z"


def test_rank_orientations_reads_what_verify_reads():
    """orient's across-layer reading for the direction a part was verified
    in is verify's own (same grid, cut cells, contact band, layered
    stiffness): it used to solve isotropically on a plain voxel grid with a
    narrower band and read a coarse wall bracket's boss corner 2.2x high,
    recommending a tilt for a part that holds printed flat."""
    from loadpath.domain import choose_pitch
    from loadpath.orient import rank_orientations
    from loadpath.verify import verify_part

    study = study_from_dict({
        "designSpace": {"type": "box", "min": [0, 0, 0], "max": [60, 10, 10]},
        "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [0.1, 11, 11]}}],
        "loadCases": [{"name": "tip", "loads": [{"shape": {"type": "box", "min": [59.9, -1, -1],
                                                           "max": [61, 11, 11]}, "force": [0, 6, -40]}]},
                      {"name": "side", "loads": [{"shape": {"type": "box", "min": [59.9, -1, -1],
                                                            "max": [61, 11, 11]}, "force": [0, 30, 0]}]}],
        "material": "petg", "process": {"type": "fdm", "buildDirection": "+y"}, "goal": {"safetyFactor": 2}})
    mesh = _box((0, 0, 0), (60, 10, 10))
    design = choose_pitch(study)
    rep = verify_part(study, mesh, max_cells=6000, min_pitch=0.5 * design, contact_band=design, refine=0)
    want = max(c["interlayerStress"] for c in rep["cases"])
    rows = rank_orientations(mesh, study, candidates=["+y", "-y", "+z"], stress_cells=6000)
    by = {r["direction"]: r for r in rows}
    assert abs(by["+y"]["interlayerStress"] - want) <= 1e-6 * want, (by["+y"]["interlayerStress"], want)
    # -y stacks the same layers: the same reading
    assert abs(by["-y"]["interlayerStress"] - want) <= 1e-6 * want
    # pinned to the grid a run verified on, the same again
    rows = rank_orientations(mesh, study, candidates=["+y"], analysis_pitch=rep["analysisPitch"],
                             contact_band=rep["contactBand"])
    assert abs(rows[0]["interlayerStress"] - want) <= 1e-6 * want


def test_choose_build_direction_halves_and_shares():
    """The orientation search: the free design (no overhang control) is one
    design for every direction, verified once per axis and printed on its
    supports; a direction is grown self-supporting only when the free
    design's cost there without any supports would still beat the best
    supported print; every option is priced at equal strength and the
    cheapest is kept. A flat lug pulled in its plane bends along its length
    (x): stood on its root (+-x) that bending crosses the layers."""
    from loadpath.orient import _FINALISTS, _make_cost, choose_build_direction

    spec = {
        "designSpace": {"type": "box", "min": [0, -15, -4], "max": [60, 15, 4]},
        "preserve": [{"id": "eye", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                             "length": 8, "r": 9}}],
        "keepOut": [{"id": "pin", "shape": {"type": "cylinder", "center": [48, 0, 0], "axis": "z",
                                            "length": 12, "r": 4}}],
        "supports": [{"shape": {"type": "box", "min": [-1, -16, -5], "max": [0.01, 16, 5]}}],
        "loadCases": [{"name": "pull", "loads": [{"region": "pin", "force": [0, -80, 0]}]}],
        "material": "petg", "process": {"type": "fdm", "buildDirection": "auto"},
        "goal": {"volumeFraction": 0.35}, "maxIterations": 60}
    study = study_from_dict(spec)
    events = []
    out = choose_build_direction(study, candidates=["+z", "-z", "+y", "+x"], max_elements=1500,
                                 on_progress=lambda e, _: events.append(e))
    rows = out["candidates"]
    free = {r["direction"]: r for r in rows if not r["overhangs"]}
    assert set(free) == {"+z", "-z", "+y", "+x"} and all(r["designed"] for r in free.values())
    # one design: opposite directions read alike, across the layers weaker
    assert free["+z"]["safetyFactor"] == free["-z"]["safetyFactor"]
    assert free["+x"]["safetyFactor"] < free["+z"]["safetyFactor"]
    # self-supporting only where its bound (the free design's cost there with
    # no supports at all) is under the best supported print
    best_free = min(r["makeCost"] for r in free.values())
    tried = {r["direction"] for r in rows if r["overhangs"]}
    for d, r in free.items():
        bound = _make_cost(r | {"supportVolume": 0.0, "supportArea": 0.0}, study, study.process, 1.0
                           if study.goal.safety_factor else max(x["strength"] for x in free.values()))
        assert (d in tried) == (bound < best_free), (d, bound, best_free)
    # the cheapest designed option is the pick, and the study carries it
    designed = [r for r in rows if r["designed"]]
    assert rows[0]["makeCost"] == min(r["makeCost"] for r in designed)
    assert out["study"].process.build == parse_build_direction(out["chosen"])
    assert out["study"].process.overhangs == rows[0]["overhangs"]
    assert any("supported" in e for e in events)
    # more self-supporting candidates than finalists: successive halving
    # (layered design stiffness makes every direction's design its own)
    ani = choose_build_direction(study_from_dict({**spec, "process": {"type": "fdm", "buildDirection": "auto",
                                                                      "anisotropic": True}}),
                                 candidates=["+z", "-z", "+y", "+x"], max_elements=1500,
                                 on_progress=lambda e, _: events.append(e))
    done = [r for r in ani["candidates"] if r["designed"]]
    dropped = [r for r in ani["candidates"] if not r["designed"]]
    assert 1 <= len(done) <= _FINALISTS and dropped, ani["candidates"]
    assert all(r["safetyFactor"] is None for r in dropped)
    assert any("screen" in e for e in events)
    # without overhang control every direction shares the one design: it is
    # verified once per axis, nothing is optimized per direction
    sls = choose_build_direction(study_from_dict({**spec, "process": {"type": "sls", "buildDirection": "auto"}}),
                                 candidates=["+z", "-z", "+x"], max_elements=1500)
    by = {r["direction"]: r for r in sls["candidates"]}
    assert all(r["designed"] and not r["overhangs"] for r in by.values())
    assert by["+z"]["safetyFactor"] == by["-z"]["safetyFactor"]
    assert by["+x"]["safetyFactor"] < by["+z"]["safetyFactor"]


def test_supports_are_priced_from_the_part():
    """costing.estimate with the part's own supports: more support (and more
    supported face) costs more, and none costs less than the flat allowance."""
    from loadpath import costing

    g = {"volume": 10.0, "surface": 0.0, "extent": [0, 0, 0], "box": 0.0, "height": 10.0}
    for kind, mat in (("lpbf", "alsi10mg"), ("fdm", "petg")):
        flat = costing.estimate(kind, mat, mass=27.0, geometry=g)["unitCost"]
        none = costing.estimate(kind, mat, mass=27.0, geometry=g, support={"volume": 0, "area": 0})["unitCost"]
        some = costing.estimate(kind, mat, mass=27.0, geometry=g, support={"volume": 2000, "area": 800})
        assert none < flat and none < some["unitCost"], (kind, none, flat, some)
        assert some["breakdown"]["supportRemoval"] > 0 and some["breakdown"]["supportMaterial"] > 0


def test_write_3mf_places_part_on_bed():
    from loadpath.threemf import write_3mf
    import tempfile
    import os

    m = _box((0, 0, 0), (10, 20, 30)).difference(_box((8, 18, 28), (12, 22, 32)))
    with tempfile.TemporaryDirectory() as td:
        p = write_3mf(os.path.join(td, "p.3mf"), m, (0, -1), name="p", material="petg")
        s = trimesh.load(p, force="mesh")
        assert np.allclose(s.bounds, [[113, 118, 0], [143, 138, 10]], atol=1e-4), s.bounds
        assert abs(s.volume - m.volume) < 1e-6
        # -x is up: the notch at the part's +x end is on the bed
        cut = s.contains([[128 + 13, 128 + 9, 1.0]])
        assert not cut[0]


def test_verify_reports_print_block_and_layered_bond():
    """A study with an FDM process gets the slicer reading in its report, and
    a bar pulled along its build axis is limited by the layer bond."""
    from loadpath.verify import verify_part

    base = {"designSpace": {"type": "box", "min": [0, 0, 0], "max": [10, 10, 40]},
            "supports": [{"shape": {"type": "box", "min": [-1, -1, -1], "max": [11, 11, 0.1]}}],
            "loadCases": [{"name": "pull", "loads": [{"shape": {"type": "box", "min": [-1, -1, 39.9],
                                                                "max": [11, 11, 41]}, "force": [0, 0, 500]}]}],
            "material": "petg", "goal": {"volumeFraction": 0.5}}
    mesh = _box((0, 0, 0), (10, 10, 40))
    up = verify_part(study_from_dict({**base, "process": {"type": "fdm", "buildDirection": "+z"}}), mesh,
                     max_cells=8000, refine=0)
    side = verify_part(study_from_dict({**base, "process": {"type": "fdm", "buildDirection": "+x"}}), mesh,
                       max_cells=8000, refine=0)
    assert up["print"]["supportFree"] and "bed" in up["print"]
    assert up["cases"][0]["interlayerStress"] > 3.0 and side["cases"][0]["interlayerStress"] < 2.0
    assert up["safetyFactor"] < 0.75 * side["safetyFactor"]


def test_delivery_keeps_every_analysed_member():
    """The opening returns a member's analysed core at the blueprint's
    width even where the core is one cell across (a print-support column,
    which the optimizer makes as thin as it is allowed), and still drops a
    sheet that has no core."""
    from types import SimpleNamespace

    from loadpath.simp import _keep_cores

    dims = (9, 9, 6)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), axis=-1).reshape(-1, 3)

    def full_grid(v):
        g = np.zeros(dims)
        g[tuple(ijk.T)] = v
        return g

    dom = SimpleNamespace(pitch=2.0, mesh=SimpleNamespace(elem_ijk=ijk), full_grid=full_grid)
    col = (np.abs(ijk[:, 0] - 2) <= 1) & (np.abs(ijk[:, 1] - 2) <= 1)       # a column 3 x 3 cells in the blueprint
    core = (ijk[:, 0] == 2) & (ijk[:, 1] == 2)                                  # ... analysed one cell across
    sheet = (ijk[:, 0] == 7)                                                    # a one-cell sheet, never analysed
    blueprint = (col | sheet).astype(float)
    analysed = core.astype(float)
    opened = np.zeros(len(ijk))                                                 # the filter lost the thin core
    out = _keep_cores(dom, opened, analysed, blueprint, radius=4.0)
    # reach: 4 / 2.6 + 1 = 2.54 mm, the six face neighbours of each core cell
    back = col & (np.abs(ijk[:, 0] - 2) + np.abs(ijk[:, 1] - 2) <= 1)
    assert np.array_equal(out > 0.5, back), "the column comes back round its core, and only there"
    assert not (out[sheet] > 0).any(), "a sheet with no analysed core stays out"
    whole = np.where(col, 1.0, 0.0)
    assert _keep_cores(dom, whole, analysed, blueprint, radius=4.0) is whole, "nothing lost: delivered as opened"


def test_constrained_stress_limit_reads_the_layer_bond_as_verification_does():
    """constrained._bond_form is failure.bond_stress (the criterion the
    check uses), and its half-gradient is the one the adjoint is fed."""
    from loadpath.constrained import _bond_form
    from loadpath.failure import bond_stress

    rng = np.random.default_rng(0)
    for n in ((0, 0, 1), (1, 0, 0), (0.3, -0.5, 0.8)):
        m, q = _bond_form(n)
        sig, k = rng.standard_normal((40, 6)), 1.9
        sb = np.maximum(sig @ m, 0.0)[:, None] * m + k * k * (sig @ q)
        assert np.abs(np.sqrt(np.einsum("ij,ij->i", sb, sig)) - bond_stress(sig, n, k)[0]).max() < 1e-12
        h, g = 1e-6, np.zeros_like(sig)
        for j in range(6):
            d = np.zeros(6)
            d[j] = h
            g[:, j] = (bond_stress(sig + d, n, k)[0] ** 2 - bond_stress(sig - d, n, k)[0] ** 2) / (4 * h)
        assert np.abs(g - sb).max() < 1e-7


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
