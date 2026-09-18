#!/usr/bin/env python3
"""
v1/primary/phase_c/run_e3_causal_map.py

Entry point to execute E3 (Shared Causal Map) across all layers
using the canonical Phase C pipeline.
"""

import subprocess
import sys
from pathlib import Path


def main():
    primary_dir = Path(__file__).resolve().parents[1]
    phase_c_script = primary_dir / "run_phase_c.py"
    args = [sys.executable, str(phase_c_script)] + sys.argv[1:]
    if "--mode" not in sys.argv:
        args.extend(["--mode", "e3"])
    res = subprocess.run(args)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
