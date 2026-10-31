"""A stand-in for evals.run.run_case in tests/test_evals.py: the case named
'dies-*' kills its worker process outright, as the kernel's OOM killer
would; the others pass."""
import os


def run_case(path, out, tier="full", critic=None):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.startswith("dies"):
        os._exit(9)
    return {"case": name, "tier": tier, "pass": True, "skipped": False, "checks": [], "seconds": 0.0, "pid": os.getpid()}
