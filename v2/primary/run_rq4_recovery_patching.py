#!/usr/bin/env python3
"""
v2/primary/run_rq4_recovery_patching.py

V2-RQ4 Primary Script: 因果的復元パッチング（Recovery Patching）
中立文への感情活性化注入による完全行動復元（True Recovery Ratio）を検証
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

v2_scripts_dir = _ROOT / "v2" / "scripts"
if str(v2_scripts_dir) not in sys.path:
    sys.path.insert(0, str(v2_scripts_dir))

import run_v2_recovery_patching

if __name__ == "__main__":
    run_v2_recovery_patching.main()
