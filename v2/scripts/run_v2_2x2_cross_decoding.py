#!/usr/bin/env python3
"""
v2/scripts/run_v2_2x2_cross_decoding.py

[Backward-compatibility wrapper]
The canonical implementation now resides in v2/primary/run_rq1_rq2_cross_decoding.py.
This script forwards execution to the canonical primary script.
"""

from pathlib import Path
import runpy
import sys

_PRIMARY_SCRIPT = Path(__file__).resolve().parents[1] / "primary" / "run_rq1_rq2_cross_decoding.py"

if __name__ == "__main__":
    runpy.run_path(str(_PRIMARY_SCRIPT), run_name="__main__")
