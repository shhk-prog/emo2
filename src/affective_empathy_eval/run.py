#!/usr/bin/env python3
"""
affective_empathy_eval.run: リポジトリ全体の統合 CLI エントリポイント

使用例:
  # Primary 1-1.5B コホート (Qwen, Llama, Gemma 3, OLMo 2) で各 Stage を実行
  python -m affective_empathy_eval.run --stage behavioral --model-set primary_small
  python -m affective_empathy_eval.run --stage v1 --model-set primary_small
  python -m affective_empathy_eval.run --stage v2 --model-set primary_small
  python -m affective_empathy_eval.run --stage v3 --model-set primary_small

  # Supplementary 7B 外部スケール検証 (Mistral 7B)
  python -m affective_empathy_eval.run --stage v2 --model-set scale_validation

  # Dry-run による高速スモークテスト
  python -m affective_empathy_eval.run --stage v2 --model-set primary_small --dry-run --family qwen
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys
from typing import List

from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    load_model_set,
    resolve_models_from_args,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for Affective Empathy Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["behavioral", "v1", "v2", "v3", "scale_validation", "all"],
        help="Experiment stage to execute",
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
        help="Device to use for computation (cpu or cuda)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of dataset samples for rapid testing",
    )
    add_model_selection_args(parser)
    return parser.parse_args()


def run_command(cmd: List[str]):
    logger.info(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")
        sys.exit(res.returncode)


def run_v2(args, python_bin: str):
    logger.info(f"=== Running V2 Stage (model-set: {args.model_set}) ===")
    common_flags = ["--models-config", "configs/models.yaml", "--model-set", args.model_set, "--device", args.device]
    if args.dry_run:
        common_flags.append("--dry-run")
    if args.family:
        common_flags.extend(["--family", args.family])
    if args.base_model:
        common_flags.extend(["--base-model", args.base_model])
    if args.instruct_model:
        common_flags.extend(["--instruct-model", args.instruct_model])
    if args.max_samples:
        common_flags.extend(["--max-samples", str(args.max_samples)])

    # 1. RQ1 & RQ2: Cross-decoding & Geometry
    run_command([python_bin, "v2/primary/run_rq1_rq2_cross_decoding.py"] + common_flags)
    # 2. RQ3: Causal Map & Peak Dissociation
    run_command([python_bin, "v2/primary/run_rq3_causal_map.py"] + common_flags)
    # 3. RQ4: Distribution Recovery Patching
    run_command([python_bin, "v2/primary/run_rq4_recovery_patching.py"] + common_flags)


def run_v3(args, python_bin: str):
    logger.info(f"=== Running V3 Stage (model-set: {args.model_set}) ===")
    common_flags = ["--models-config", "configs/models.yaml", "--model-set", args.model_set, "--device", args.device]
    if args.dry_run:
        common_flags.append("--dry-run")
    if args.family:
        common_flags.extend(["--family", args.family])
    if args.base_model:
        common_flags.extend(["--base-model", args.base_model])
    if args.instruct_model:
        common_flags.extend(["--instruct-model", args.instruct_model])

    # 1. RQ1 State Induction & Go/No-Go Gate
    run_command([python_bin, "v3/primary/run_rq1_state_induction.py"] + common_flags)
    # 2. RQ2 Spatiotemporal Maps
    run_command([python_bin, "v3/primary/run_rq2_spatiotemporal_maps.py"] + common_flags)
    # 3. RQ3 Path Mediation
    run_command([python_bin, "v3/primary/run_rq3_path_mediation.py"] + common_flags)
    # 4. Confirmatory Replication
    run_command([python_bin, "v3/primary/run_confirmatory_replication.py"] + common_flags)


def run_v1(args, python_bin: str):
    logger.info(f"=== Running V1 Stage Summary ===")
    run_command([python_bin, "v1/primary/phase_c/summarize_phase_c.py"])


def run_behavioral(args, python_bin: str):
    logger.info(f"=== Running Behavioral Stage (model-set: {args.model_set}) ===")
    models = resolve_models_from_args(args)
    for fid, cfg in models.items():
        logger.info(f"Running behavioral evaluation for {fid} ({cfg.instruct_model.model_id})...")
        cmd = [
            python_bin,
            "behavioral/primary/run_behavioral_emobank.py",
            "--model", cfg.instruct_model.model_id,
            "--is_instruct",
            "--tag", f"{fid.lower()}_instruct",
        ]
        if args.max_samples:
            cmd.extend(["--limit", str(args.max_samples)])
        # dry-run の場合はスキップまたは警告
        if args.dry_run:
            logger.info(f"[Dry-run] Simulated behavioral execution for {cfg.instruct_model.model_id}")
        else:
            run_command(cmd)


def run_scale_validation(args, python_bin: str):
    logger.info("=== Running Scale Validation Ablation (Mistral 7B) ===")
    cmd = [python_bin, "scripts/run_scale_validation.py", "--device", args.device]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    run_command(cmd)


def main():
    args = parse_args()
    python_bin = sys.executable

    logger.info(f"Affective Empathy Evaluation Unified Runner")
    logger.info(f"Stage: {args.stage} | Cohort: {args.model_set} | Dry-run: {args.dry_run}")

    if args.stage == "scale_validation":
        run_scale_validation(args, python_bin)
    else:
        if args.stage in ("v2", "all"):
            run_v2(args, python_bin)
        if args.stage in ("v3", "all"):
            run_v3(args, python_bin)
        if args.stage in ("v1", "all"):
            run_v1(args, python_bin)
        if args.stage in ("behavioral", "all"):
            run_behavioral(args, python_bin)

    logger.info("All requested stages completed successfully!")


if __name__ == "__main__":
    main()
