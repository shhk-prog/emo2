#!/usr/bin/env python3
"""
v3/primary/run_rq3_path_mediation.py

V3-RQ3 Primary Script: Causal Mediation & Circuit Mechanism
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

v3_scripts_dir = _ROOT / "v3" / "scripts"
if str(v3_scripts_dir) not in sys.path:
    sys.path.insert(0, str(v3_scripts_dir))

import run_v3_path_mediation

if __name__ == "__main__":
    run_v3_path_mediation.main()
