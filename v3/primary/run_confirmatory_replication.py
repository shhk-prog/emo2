#!/usr/bin/env python3
"""
v3/primary/run_confirmatory_replication.py

V3 Confirmatory Replication Primary Script:
厳密な事前登録プロトコルに基づく追試・反証実験
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

v3_scripts_dir = _ROOT / "v3" / "scripts"
if str(v3_scripts_dir) not in sys.path:
    sys.path.insert(0, str(v3_scripts_dir))

import run_v3_confirmatory_replication

if __name__ == "__main__":
    run_v3_confirmatory_replication.main()
