#!/usr/bin/env python3
"""
v2/primary/run_confirmatory_analysis.py

V2 Confirmatory Analysis Primary Script:
仮説検証的統計解析 (LMM, Crossover Interaction, FDR Correction)
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

v2_scripts_dir = _ROOT / "v2" / "scripts"
if str(v2_scripts_dir) not in sys.path:
    sys.path.insert(0, str(v2_scripts_dir))

import run_confirmatory_analysis

if __name__ == "__main__":
    run_confirmatory_analysis.main()
