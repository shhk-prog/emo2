#!/usr/bin/env python3
"""
v1/primary/phase_c/run_e4_interchangeability.py

Entry point to execute E4 (Causal Interchangeability) on candidate layers
using the canonical Phase C pipeline.
"""

import sys
from pathlib import Path

primary_dir = Path(__file__).resolve().parents[1]
if str(primary_dir) not in sys.path:
    sys.path.insert(0, str(primary_dir))

from run_phase_c import main

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "e4"])
    main()
