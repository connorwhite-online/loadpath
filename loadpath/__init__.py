"""loadpath: generative design and engineering checks for 3D printed parts.

Studies are JSON (see loadpath.spec and schema/); run one with
loadpath.study.run_study, or use the CLI (`loadpath run`) or the MCP server
(`loadpath mcp`). Units are mm, N, MPa. See docs/VALIDATION.md for what the
models have been checked against.
"""
import os as _os

# One BLAS thread unless the caller says otherwise. The heavy work is sparse
# (CG, multigrid) and already runs a thread per load case and a process per
# outcome; OpenBLAS's own pool of busy-waiting threads, one per core, fought
# both: an arm-link run took 112 s with it and 88 s without, and two side by
# side 330 s against 97 s on 4 cores. (Only effective before numpy loads;
# worker processes start fresh and always get it.)
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")
