#!/usr/bin/env python3
"""
v2/primary/run_rq1_rq2_cross_decoding.py

V2-RQ1 & RQ2: Post-training による表現幾何の変化と Reader–Self 共有性の再編
4モデルファミリー (Qwen 2.5, Llama 3.2, Gemma 2, Mistral) × 2水準 (Base, Instruct)
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from affective_empathy_eval.geometry import (
    compute_center_of_mass,
    compute_peak_depth,
    compute_relative_depth,
    compute_rsa_correlation,
    eval_held_out_cross_decoding,
    eval_held_out_procrustes,
    train_and_eval_held_out_probe,
)
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
from affective_empathy_eval.models.registry import get_registry
from affective_empathy_eval.prompts import (
    TaskType,
    build_prompt,
    encode_prompt_canonical,
    find_semantic_anchors,
)
from affective_empathy_eval.statistics import (
    compute_bootstrap_ci,
    paired_family_comparison,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run V2-RQ1 & RQ2 Cross-decoding and Geometry Analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v2_experiments.yaml",
        help="Path to V2 config",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default="configs/models.yaml",
        help="Path to models config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in mock/dry-run mode without loading full weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for quick testing",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Target specific family (e.g. Qwen)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="v2/results/derived",
        help="Output directory",
    )
    return parser.parse_args()


# Forward to canonical script implementation in v2/scripts/run_v2_2x2_cross_decoding.py
if __name__ == "__main__":
    v2_scripts_dir = _ROOT / "v2" / "scripts"
    if str(v2_scripts_dir) not in sys.path:
        sys.path.insert(0, str(v2_scripts_dir))
    import run_v2_2x2_cross_decoding

    run_v2_2x2_cross_decoding.main()
