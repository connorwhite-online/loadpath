"""Moved to `loadpath.exchanger` (it generates geometry; it is not a check). This
alias keeps `loadpath.checks.exchanger` imports working."""
import sys

from .. import exchanger as _moved

sys.modules[__name__] = _moved
