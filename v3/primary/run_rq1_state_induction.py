#!/usr/bin/env python3
"""
v3/primary/run_rq1_state_induction.py

V3-RQ1 Primary Script: State Induction and Subspace Geometry
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

v3_scripts_dir = _ROOT / "v3" / "scripts"
if str(v3_scripts_dir) not in sys.path:
    sys.path.insert(0, str(v3_scripts_dir))

import run_v3_state_induction

if __name__ == "__main__":
    run_v3_state_induction.main()
