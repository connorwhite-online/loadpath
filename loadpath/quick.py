"""
Quick mode: every command on a coarse grid, for a first look in a minute or
two and for the tutorial tests (tests/test_tutorials.py), which run each
tutorial's commands this way so the docs can't drift from the code.

    loadpath run study.json --quick           # or LOADPATH_QUICK=1 for anything

What it lowers: a study's cells, optimizer iterations and outcome-search runs
(the evals' coarse tier, evals/run.py COARSE), a heat exchanger's grid and
whole-exchanger simulation, a field graph's mesh pitch (and its drain search's)
and the analysis behind its `result` nodes, `tune`'s analyses and
`assembly`'s sampled poses. Numbers move (mass a few %, safety factor more on
thin members): read a quick run for direction, run without it to print.
"""
from __future__ import annotations

import copy
import os

# the evals' coarse tier, so a quick run and `evals.run --tier coarse` agree
STUDY = {"maxElements": 6000, "maxIterations": 30, "searchRuns": 3, "confirmRuns": 0}
HX_STEP = 1.6          # x the heat exchanger's grid step (default wall / 3)
HX_SIM_VOXEL = 1.5     # x the whole-exchanger simulation's voxel
FIELD_CELLS = 600_000  # mesh grid cells for a field graph without its own pitch
FIELD_PITCH = 1.6      # x a field graph's own mesh pitch
RESULT_CELLS = 30_000  # analysis cells behind a field graph's `result` node (default 120k)
TUNE_EVALUATIONS = 6
ASSEMBLY_SAMPLES = 150


def on() -> bool:
    return os.environ.get("LOADPATH_QUICK", "").strip().lower() not in ("", "0", "false", "no")


def enable():
    """Quick mode for this process and everything it starts."""
    os.environ["LOADPATH_QUICK"] = "1"


def study(d: dict) -> dict:
    """A study dict as quick mode runs it (a copy; `d` itself when off).
    Also lowers study.py's search limits for this process."""
    if not on() or not isinstance(d, dict):
        return d
    d = copy.deepcopy(d)
    cells = STUDY["maxElements"]
    want = ((d.get("expect") or {}).get("coarse") or {}).get("maxElements")
    if want:
        cells = max(cells, int(want))
    res = dict(d.get("resolution") or {})
    res.pop("pitch", None)
    res["maxElements"] = min(int(res.get("maxElements", 80000)), cells)
    d["resolution"] = res
    d["maxIterations"] = min(int(d.get("maxIterations", 120)), STUDY["maxIterations"])
    from . import study as st
    st._MAX_SEARCH_RUNS = min(st._MAX_SEARCH_RUNS, STUDY["searchRuns"])
    st._CONFIRM_RUNS = min(st._CONFIRM_RUNS, STUDY["confirmRuns"])
    return d


def heatx(hx, sim_opts):
    """Coarsen a HeatExchanger's grid in place; the simulation options as
    quick mode runs them."""
    if not on():
        return sim_opts
    hx.step = (hx.step or hx.wall / 3) * HX_STEP
    if sim_opts is not None and sim_opts.get("voxel"):
        sim_opts = {**sim_opts, "voxel": float(sim_opts["voxel"]) * HX_SIM_VOXEL}
    return sim_opts


def mesh_pitch(pitch, bounds):
    """A field graph's mesh pitch in quick mode: its own x FIELD_PITCH, or
    the box over FIELD_CELLS cells (the default is 4M)."""
    if not on():
        return pitch
    if pitch:
        return float(pitch) * FIELD_PITCH
    import numpy as np
    ext = np.asarray(bounds[1], float) - np.asarray(bounds[0], float)
    return float((np.prod(ext) / FIELD_CELLS) ** (1 / 3))


def result_cells(n):
    """Analysis cells behind a field graph's `result` node."""
    return min(int(n), RESULT_CELLS) if on() else n


def tune_evaluations(n):
    return min(int(n), TUNE_EVALUATIONS) if on() else n


def assembly_samples(n: int) -> int:
    return min(int(n), ASSEMBLY_SAMPLES) if on() else n
