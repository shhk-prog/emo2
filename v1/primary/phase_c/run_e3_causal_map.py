#!/usr/bin/env python3
"""
v1/primary/phase_c/run_e3_causal_map.py

Entry point to execute E3 (Shared Causal Map) across all layers
using the canonical Phase C pipeline.
"""

import sys
from pathlib import Path

# Add v1/primary to path
primary_dir = Path(__file__).resolve().parents[1]
if str(primary_dir) not in sys.path:
    sys.path.insert(0, str(primary_dir))

from run_phase_c import main

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "e3"])
    main()
