"""
Kernel-verified tests for the component-fit verifier (MTR-204).

Builds REAL build123d Pico enclosures — one deliberately correct, three
deliberately wrong (cavity too small / bosses misplaced / cutout on the wrong
edge) — and asserts `fit.check_fit` passes the right one and catches each
wrong one with the failing check (and only that check). One test drives the
full HTTP seam (`/run` with `checks={"fit": ...}`, engine build123d).

Run: `python3 cad-runner/tests/test_fit.py` (needs build123d + trimesh +
scipy — the same kernel the sidecar image ships).
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ["CAD_RUNNER_ALLOW_NO_AUTH"] = "true"
if __name__ == "__main__":
    os.environ.pop("CAD_RUNNER_SECRET", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

from fit import check_fit

# Mirror of lib/cad/knowledge/components.ts fitSpecForBoard(DEV_BOARDS.pico):
# board 51 x 21 x 1, holes D2.1 at (±23.5, ±5.7), micro-USB centered on a
# short end, 1.3 above the board top; envelope = board + 1 clearance + 3 above.
PICO_HOLES = [[-23.5, -5.7], [-23.5, 5.7], [23.5, -5.7], [23.5, 5.7]]
PICO_SPEC = {
    "components": [
        {
            "id": "pico",
            "label": "Raspberry Pi Pico",
            "boardMm": [51.0, 21.0, 1.0],
            "clearanceMm": 1.0,
            "aboveMm": 3.0,
            "belowMm": 0.0,
            "holes": {"positions": PICO_HOLES, "diaMm": 2.1},
            "ports": [
                {
                    "id": "micro-usb",
                    "edge": "-x",
                    "offsetMm": 0.0,
                    "heightMm": 1.3,
                    "wMm": 8.4,
                    "hMm": 3.4,
                }
            ],
        }
    ]
}

WALL = 2.5
FLOOR = 3.0
BOSS_H = 4.0
DEPTH = 14.0
# Board bottom seats on the boss tops; the micro-USB centerline follows.
USB_Z = FLOOR + BOSS_H + 1.0 + 1.3


def build_enclosure(
    inner_x=55.0,
    inner_y=25.0,
    bosses=True,
    boss_shift_x=0.0,
    cutout_edge="-x",
):
    """A parametric open-top Pico enclosure the way a competent generation
    would build it: floor + walls, four Ø6 bosses with Ø1.7 pilots at the
    mounting-hole grid, and a 9 x 4.5 micro-USB cutout at the connector."""
    from build123d import (
        Align,
        Box,
        BuildPart,
        Cylinder,
        Locations,
        Mode,
    )

    outer_x, outer_y = inner_x + 2 * WALL, inner_y + 2 * WALL
    center_align = (Align.CENTER, Align.CENTER, Align.MIN)
    with BuildPart() as p:
        Box(outer_x, outer_y, FLOOR + DEPTH, align=center_align)
        with Locations((0, 0, FLOOR)):
            Box(inner_x, inner_y, DEPTH + 1, align=center_align, mode=Mode.SUBTRACT)
        if bosses:
            for hx, hy in PICO_HOLES:
                with Locations((hx + boss_shift_x, hy, FLOOR)):
                    Cylinder(radius=3.0, height=BOSS_H, align=center_align)
                with Locations((hx + boss_shift_x, hy, FLOOR + BOSS_H - 6.0)):
                    Cylinder(
                        radius=0.85, height=6.0, align=center_align,
                        mode=Mode.SUBTRACT,
                    )
        if cutout_edge is not None:
            if cutout_edge == "-x":
                loc, dims = (-(inner_x + WALL) / 2.0, 0.0), (WALL + 2.0, 9.0)
                with Locations((loc[0], loc[1], USB_Z)):
                    Box(dims[0], dims[1], 4.5, mode=Mode.SUBTRACT)
            elif cutout_edge == "+y":
                with Locations((0.0, (inner_y + WALL) / 2.0, USB_Z)):
                    Box(9.0, WALL + 2.0, 4.5, mode=Mode.SUBTRACT)
    return p.part


def to_mesh(part):
    from build123d import export_stl

    with tempfile.TemporaryDirectory() as tmp:
        stl = os.path.join(tmp, "part.stl")
        export_stl(part, stl)
        mesh = trimesh.load(stl, force="mesh")
    mesh.merge_vertices()
    assert mesh.is_watertight, "test enclosure must be watertight"
    return mesh


def by_id(report, rid):
    matches = [r for r in report["results"] if r["id"] == rid]
    assert matches, f"no result {rid!r} in {[r['id'] for r in report['results']]}"
    return matches[0]


# ---- the deliberately-correct enclosure passes everything -------------------
def test_correct_enclosure_passes_all_checks():
    mesh = to_mesh(build_enclosure())
    report = check_fit(mesh, PICO_SPEC)
    assert report.get("error") is None, report
    cavity = by_id(report, "fit:pico:cavity")
    bosses = by_id(report, "fit:pico:bosses")
    port = by_id(report, "fit:pico:port:micro-usb")
    assert cavity["ok"] is True, f"cavity: {cavity['note']}"
    assert bosses["ok"] is True, f"bosses: {bosses['note']}"
    assert bosses["got"] == 4, bosses
    assert port["ok"] is True, f"port: {port['note']}"


# ---- cavity too small: the board simply does not fit ------------------------
def test_undersized_cavity_fails_cavity_check():
    mesh = to_mesh(build_enclosure(inner_x=45.0, inner_y=18.0, bosses=False))
    report = check_fit(mesh, PICO_SPEC)
    cavity = by_id(report, "fit:pico:cavity")
    assert cavity["ok"] is False, f"cavity must FAIL: {cavity['note']}"
    assert "no placement" in cavity["note"].lower() or "enlarge" in cavity["note"].lower()
    # Dependent checks honestly report not-run, never a fake pass.
    assert by_id(report, "fit:pico:bosses")["ok"] is None
    assert by_id(report, "fit:pico:port:micro-usb")["ok"] is None


# ---- bosses 4 mm off pattern: cavity fine, boss check catches it -------------
def test_misplaced_bosses_fail_boss_check():
    mesh = to_mesh(build_enclosure(boss_shift_x=4.0))
    report = check_fit(mesh, PICO_SPEC)
    cavity = by_id(report, "fit:pico:cavity")
    bosses = by_id(report, "fit:pico:bosses")
    assert cavity["ok"] is True, f"cavity should still pass: {cavity['note']}"
    assert bosses["ok"] is False, f"bosses must FAIL: {bosses['note']}"
    assert bosses["got"] < 4, bosses


# ---- cutout on a long side instead of the USB end ----------------------------
def test_cutout_on_wrong_edge_fails_port_check():
    mesh = to_mesh(build_enclosure(cutout_edge="+y"))
    report = check_fit(mesh, PICO_SPEC)
    cavity = by_id(report, "fit:pico:cavity")
    port = by_id(report, "fit:pico:port:micro-usb")
    assert cavity["ok"] is True, f"cavity should still pass: {cavity['note']}"
    assert port["ok"] is False, f"port must FAIL: {port['note']}"
    assert "micro-usb" in port["note"]


# ---- open mesh: fit checks refuse to guess -----------------------------------
def test_open_mesh_reports_error_not_verdicts():
    m = trimesh.creation.icosphere(subdivisions=2, radius=30.0)
    m.update_faces(m.triangles_center[:, 2] < 25.0)
    m.remove_unreferenced_vertices()
    assert not m.is_watertight
    report = check_fit(m, PICO_SPEC)
    assert report.get("error"), "open mesh must report an error, not verdicts"
    assert report["results"] == []


# ---- "+z" face ports: screen windows through the lid (MTR-203) --------------
# A 1602-style display spec: body block, no bosses, one top-face window.
DISPLAY_SPEC = {
    "components": [
        {
            "id": "lcd1602",
            "label": "1602 LCD",
            "boardMm": [80.0, 36.0, 1.6],
            "clearanceMm": 1.0,
            "aboveMm": 7.0,
            "belowMm": 0.0,
            "holes": None,
            "ports": [
                {
                    "id": "screen-window",
                    "face": "+z",
                    "xMm": 0.0,
                    "yMm": 0.0,
                    "wMm": 64.5,
                    "hMm": 15.0,
                }
            ],
        }
    ]
}


def build_display_box(window=True):
    """Closed box with an interior cavity for the 1602 block; the lid gets a
    64.5 x 15 window over the display when `window` is True."""
    from build123d import Align, Box, BuildPart, Locations, Mode

    inner_x, inner_y, inner_z = 84.0, 40.0, 10.6
    lid = 2.5
    align = (Align.CENTER, Align.CENTER, Align.MIN)
    with BuildPart() as p:
        Box(inner_x + 2 * WALL, inner_y + 2 * WALL, FLOOR + inner_z + lid, align=align)
        with Locations((0, 0, FLOOR)):
            Box(inner_x, inner_y, inner_z, align=align, mode=Mode.SUBTRACT)
        if window:
            with Locations((0, 0, FLOOR + inner_z - 0.5)):
                Box(64.5, 15.0, lid + 1.5, align=align, mode=Mode.SUBTRACT)
    return p.part


def test_screen_window_passes_when_lid_is_open():
    report = check_fit(to_mesh(build_display_box(window=True)), DISPLAY_SPEC)
    assert by_id(report, "fit:lcd1602:cavity")["ok"] is True
    assert by_id(report, "fit:lcd1602:port:screen-window")["ok"] is True, report


def test_screen_window_fails_under_solid_lid():
    report = check_fit(to_mesh(build_display_box(window=False)), DISPLAY_SPEC)
    assert by_id(report, "fit:lcd1602:cavity")["ok"] is True
    res = by_id(report, "fit:lcd1602:port:screen-window")
    assert res["ok"] is False, res
    assert "lid" in res["note"] or "window" in res["note"]


# ---- full HTTP seam: /run with checks.fit on the build123d engine ------------
def test_run_endpoint_fit_checks_round_trip():
    import app as app_module
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    code = f"""
from build123d import Align, Box, BuildPart, Cylinder, Locations, Mode

WALL, FLOOR, BOSS_H, DEPTH = {WALL}, {FLOOR}, {BOSS_H}, {DEPTH}
INNER_X, INNER_Y = 55.0, 25.0
HOLES = {PICO_HOLES}
USB_Z = {USB_Z}
align = (Align.CENTER, Align.CENTER, Align.MIN)
with BuildPart() as p:
    Box(INNER_X + 2 * WALL, INNER_Y + 2 * WALL, FLOOR + DEPTH, align=align)
    with Locations((0, 0, FLOOR)):
        Box(INNER_X, INNER_Y, DEPTH + 1, align=align, mode=Mode.SUBTRACT)
    for hx, hy in HOLES:
        with Locations((hx, hy, FLOOR)):
            Cylinder(radius=3.0, height=BOSS_H, align=align)
        with Locations((hx, hy, FLOOR + BOSS_H - 6.0)):
            Cylinder(radius=0.85, height=6.0, align=align, mode=Mode.SUBTRACT)
    with Locations((-(INNER_X + WALL) / 2.0, 0.0, USB_Z)):
        Box(WALL + 2.0, 9.0, 4.5, mode=Mode.SUBTRACT)
result = p.part
"""
    r = client.post(
        "/run",
        json={
            "code": code,
            "formats": ["stl"],
            "engine": "build123d",
            "checks": {"fit": PICO_SPEC},
        },
    )
    assert r.status_code == 200, r.text[:300]
    payload = r.json()
    assert payload["ok"] is True, payload.get("error")
    fit = (payload.get("checks") or {}).get("fit")
    assert fit and fit.get("error") is None, f"fit missing/errored: {fit}"
    ids = {res["id"]: res["ok"] for res in fit["results"]}
    assert ids == {
        "fit:pico:cavity": True,
        "fit:pico:bosses": True,
        "fit:pico:port:micro-usb": True,
    }, fit["results"]


TESTS = [
    test_correct_enclosure_passes_all_checks,
    test_undersized_cavity_fails_cavity_check,
    test_misplaced_bosses_fail_boss_check,
    test_cutout_on_wrong_edge_fails_port_check,
    test_screen_window_passes_when_lid_is_open,
    test_screen_window_fails_under_solid_lid,
    test_open_mesh_reports_error_not_verdicts,
    test_run_endpoint_fit_checks_round_trip,
]


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
