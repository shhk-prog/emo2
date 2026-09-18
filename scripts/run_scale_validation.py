#!/usr/bin/env python3
"""
scripts/run_scale_validation.py: 外部スケール検証・アブレーション実験 (Mistral 7B) 実行スクリプト

Primary 1-1.5B コホートから完全に分離された Ablation スイートとして、
Mistral 7B (Base vs. Instruct) の表現幾何・因果回路・分布回復を単独検証します。
成果物は results/ablation/scale_validation/ 配下に独立保存されます。

使用例:
  python scripts/run_scale_validation.py --dry-run
  python scripts/run_scale_validation.py --device cuda
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

from affective_empathy_eval.models.registry import (
    load_model_set,
)

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Scale Validation & Ablation Experiment (Mistral 7B) via V2 Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default="configs/models.yaml",
        help="Path to models config",
    )
    parser.add_argument(
        "--model-set",
        type=str,
        default="scale_validation",
        help="Model set name in models.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in mock/dry-run mode without loading full model weights",
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
        help="Limit number of dataset samples for rapid testing",
    )
    return parser.parse_args()


def run_command(cmd):
    logger.info(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main():
    args = parse_args()
    python_bin = sys.executable
    logger.info(f"=== Starting Scale Validation Ablation (Model Set: {args.model_set}) ===")
    logger.info(f"Config: {args.models_config} | Dry-run: {args.dry_run} | Device: {args.device}")

    # レジストリ確認
    registered_models = load_model_set(Path(args.models_config), model_set=args.model_set)
    if not registered_models:
        logger.error(f"No models found for model_set '{args.model_set}' in {args.models_config}")
        sys.exit(1)

    for fam_key, fam_cfg in registered_models.items():
        logger.info(f"Scale validation target: {fam_cfg.family_name} ({fam_cfg.scale})")
        logger.info(f"  Base: {fam_cfg.base_model.model_id}")
        logger.info(f"  Instruct: {fam_cfg.instruct_model.model_id}")

    common_flags = [
        "--models-config", args.models_config,
        "--model-set", args.model_set,
        "--device", args.device,
    ]
    if args.dry_run:
        common_flags.append("--dry-run")
    if args.max_samples:
        common_flags.extend(["--max-samples", str(args.max_samples)])

    # 1. クロスデコード・表現幾何 (RQ1 & RQ2)
    logger.info("--- Step 1: Geometry & Held-out Cross-decoding ---")
    run_command([python_bin, "v2/primary/run_rq1_rq2_cross_decoding.py"] + common_flags)

    # 2. 因果マッピング (RQ3)
    logger.info("--- Step 2: Causal Leverage & Peak Dissociation ---")
    run_command([python_bin, "v2/primary/run_rq3_causal_map.py"] + common_flags)

    # 3. 分布回復パッチング (RQ4)
    logger.info("--- Step 3: Distribution Recovery Patching ---")
    run_command([python_bin, "v2/primary/run_rq4_recovery_patching.py"] + common_flags)

    logger.info("=== Scale Validation Ablation completed successfully! ===")


if __name__ == "__main__":
    main()
