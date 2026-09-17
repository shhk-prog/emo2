"""
DEPRECATED: This directory is preserved for backwards-compatibility redirection.
The unified, canonical package is located at `src/affective_empathy_eval`.
"""

from pathlib import Path
import sys
import warnings

_ROOT_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _ROOT_SRC not in sys.path:
    sys.path.insert(0, _ROOT_SRC)

warnings.warn(
    "Importing from v1/src/affective_empathy_eval is deprecated. "
    "Please import from the root package `affective_empathy_eval`.",
    DeprecationWarning,
    stacklevel=2,
)

from affective_empathy_eval import *
