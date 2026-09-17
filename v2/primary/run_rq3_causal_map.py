#!/usr/bin/env python3
"""
v2/primary/run_rq3_causal_map.py

V2-RQ3 Primary Script: Post-training による因果回路の再配置とピーク解離解析
4モデルファミリー (Qwen 2.5, Llama 3.2, Gemma 2, Mistral) × 4条件 (Base/Inst × Reader/Self)
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

v2_scripts_dir = _ROOT / "v2" / "scripts"
if str(v2_scripts_dir) not in sys.path:
    sys.path.insert(0, str(v2_scripts_dir))

import run_v2_2x2_causal_map

if __name__ == "__main__":
    run_v2_2x2_causal_map.main()
