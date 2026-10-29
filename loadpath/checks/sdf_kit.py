"""Moved to `loadpath.sdf_kit` (it generates geometry; it is not a check). This
alias keeps `loadpath.checks.sdf_kit` imports working."""
import sys

from .. import sdf_kit as _moved

sys.modules[__name__] = _moved
