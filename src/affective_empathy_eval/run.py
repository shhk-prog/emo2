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
import os
from pathlib import Path
import subprocess
import sys
from typing import List

from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    resolve_models_from_args,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# run_production_all.sh と同一の本番実行順。--stage all もこの順に従う。
PRODUCTION_STAGE_ORDER = ("behavioral", "v1", "v2", "v3")

# 件数は手書きせず、実行時に実ファイルを読んでログする。
PRODUCTION_DATASET_INVENTORY = (
    ("EmoBank 3-way", "v1/data/processed/stimuli_vad_3way_test1k.csv"),
    ("AIPsy 4-split", "v1/data/processed/aipsy_4split_all.csv"),
    ("V1 Phase B rule-based controls", "v1/data/processed/v1_e5_semantic_controls.csv"),
)


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
    parser.add_argument(
        "--all-layers",
        action="store_true",
        default=False,
        help="Test all layers in V1 Phase C causal patching",
    )
    parser.add_argument(
        "--force-after-no-go",
        action="store_true",
        help="Continue V3 RQ2/RQ3/Confirmatory even if RQ1 gate is NO_GO",
    )
    return parser.parse_args()


def log_production_dataset_inventory() -> None:
    """固定件数コメントの代わりに、実CSVの行数・pair数をログする。"""
    from affective_empathy_eval.data import describe_loaded_frame
    import pandas as pd

    logger.info("Dataset inventory (actual loaded counts; wall-clock time is unmeasured):")
    for name, path in PRODUCTION_DATASET_INVENTORY:
        csv_path = Path(path)
        if not csv_path.exists():
            logger.warning(f"Dataset not found for inventory logging: {path}")
            continue
        df = pd.read_csv(csv_path)
        logger.info(describe_loaded_frame(df, name, path))


def run_command(cmd: List[str]):
    logger.info(f"Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    res = subprocess.run(cmd, env=env)
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


def v3_gate_allows_continuation(
    gate_path: str | Path = "v3/results/derived/v3_gate_decision.json",
    force: bool = False,
) -> bool:
    """RQ1 の Go/No-Go を production 経路で実際に適用する。"""
    import json

    path = Path(gate_path)
    if not path.exists():
        logger.error(f"V3 gate file not found: {path}")
        return False
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    decision = str(payload.get("decision", "")).upper()
    logger.info(f"V3 RQ1 gate decision from {path}: {decision}")
    if decision == "GO":
        return True
    if force:
        logger.warning("V3 gate is not GO, but --force-after-no-go was set. Continuing.")
        return True
    return False


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
    cmd_rq1 = [python_bin, "v3/primary/run_rq1_state_induction.py"] + common_flags
    # 2. RQ2 Spatiotemporal Maps
    cmd_rq2 = [python_bin, "v3/primary/run_rq2_spatiotemporal_maps.py"] + common_flags
    # 3. RQ3 Path Mediation
    cmd_rq3 = [python_bin, "v3/primary/run_rq3_path_mediation.py"] + common_flags
    # 4. Confirmatory Replication
    cmd_rq4 = [python_bin, "v3/primary/run_confirmatory_replication.py"] + common_flags

    if args.max_samples:
        cmd_rq2.extend(["--subsample", str(args.max_samples)])
        cmd_rq3.extend(["--subsample", str(args.max_samples)])
        cmd_rq4.extend(["--subsample", str(args.max_samples)])

    run_command(cmd_rq1)
    if not v3_gate_allows_continuation(force=args.force_after_no_go):
        logger.error(
            "V3 RQ1 gate is NO_GO. Stopping before RQ2/RQ3/Confirmatory. "
            "Pass --force-after-no-go only for an explicit override."
        )
        sys.exit(2)
    run_command(cmd_rq2)
    run_command(cmd_rq3)
    run_command(cmd_rq4)


def run_v1(args, python_bin: str):
    logger.info(f"=== Running V1 Stage Pipeline (model-set: {args.model_set}) ===")
    controls_csv = Path("v1/data/processed/v1_e5_semantic_controls.csv")
    if not controls_csv.exists():
        logger.info("v1_e5_semantic_controls.csv not found. Auto-generating via prepare_v1_phase_b_controls.py...")
        run_command([python_bin, "v1/primary/prepare_v1_phase_b_controls.py"])

    models = resolve_models_from_args(args)

    for fid, cfg in models.items():
        # Iterate over both Base and Instruct models
        variants = [
            (cfg.base_model.model_id, f"{fid.lower()}_base", False),
            (cfg.instruct_model.model_id, f"{fid.lower()}_instruct", True),
        ]
        for model_id, prefix, is_instruct in variants:
            logger.info(f"--- Running V1 Pipeline for {prefix} ({model_id}) ---")
            common_flags = [
                "--model-id", model_id,
                "--model-prefix", prefix,
                "--device", args.device,
            ]
            if is_instruct:
                common_flags.append("--is-instruct")
            if args.dry_run:
                common_flags.append("--dry-run")
            if args.max_samples:
                common_flags.extend(["--limit", str(args.max_samples)])

            # 1. Phase A: Probing & Geometry
            run_command([python_bin, "v1/primary/run_phase_a.py"] + common_flags)

            # 2. Phase B: Semantic vs Lexical Controls Audit
            run_command([python_bin, "v1/primary/run_phase_b.py"] + common_flags)

            # 3. Phase C: Causal Interventions (E3/E4)
            cmd_phase_c = [python_bin, "v1/primary/run_phase_c.py"] + common_flags
            if getattr(args, "all_layers", False):
                cmd_phase_c.append("--all-layers")
            run_command(cmd_phase_c)

            # 4. Phase C E6: Task-Specific Causal Specialization & LMM
            run_command([python_bin, "v1/primary/phase_c/run_e6_specialization.py"] + common_flags)

    # 5. Summarize Phase C
    logger.info("--- Summarizing V1 Phase C Results ---")
    run_command([python_bin, "v1/primary/phase_c/summarize_phase_c.py"])


def run_behavioral(args, python_bin: str):
    logger.info(f"=== Running Behavioral Stage (model-set: {args.model_set}) ===")
    models = resolve_models_from_args(args)
    for fid, cfg in models.items():
        variants = [
            (cfg.base_model.model_id, f"{fid.lower()}_base", False),
            (cfg.instruct_model.model_id, f"{fid.lower()}_instruct", True),
        ]
        for model_id, tag, is_instruct in variants:
            logger.info(f"Running behavioral evaluation for {tag} ({model_id})...")

            # 1. EmoBank 3-Way VAD
            cmd_emobank = [
                python_bin,
                "behavioral/primary/run_behavioral_emobank.py",
                "--model", model_id,
                "--tag", tag,
                "--device", args.device,
            ]
            if is_instruct:
                cmd_emobank.append("--is_instruct")
            if args.max_samples:
                cmd_emobank.extend(["--limit", str(args.max_samples)])

            # 2. AIPsy-Affect 4-Split
            cmd_aipsy = [
                python_bin,
                "behavioral/primary/run_behavioral_aipsy.py",
                "--model", model_id,
                "--tag", tag,
                "--device", args.device,
            ]
            if is_instruct:
                cmd_aipsy.append("--is-instruct")
            if args.max_samples:
                cmd_aipsy.extend(["--limit", str(args.max_samples)])

            if args.dry_run:
                logger.info(f"[Dry-run] Simulated EmoBank execution for {model_id} (tag={tag})")
                logger.info(f"[Dry-run] Simulated AIPsy execution for {model_id} (tag={tag})")
            else:
                run_command(cmd_emobank)
                run_command(cmd_aipsy)

    # 3. 自動要約・統計集計・論文用 derived CSV 生成 (Unified Pipeline)
    logger.info("--- Generating Behavioral Summary and Derived Tables ---")
    cmd_sum_emobank = [python_bin, "behavioral/analysis/summarize_behavioral_emobank.py"]
    cmd_sum_aipsy = [python_bin, "behavioral/analysis/summarize_behavioral_aipsy.py"]
    if args.dry_run:
        logger.info("[Dry-run] Simulated Behavioral summarization (EmoBank & AIPsy)")
    else:
        run_command(cmd_sum_emobank)
        run_command(cmd_sum_aipsy)



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

    logger.info("Affective Empathy Evaluation Unified Runner")
    logger.info(f"Stage: {args.stage} | Cohort: {args.model_set} | Dry-run: {args.dry_run}")
    logger.info("Wall-clock estimates are unmeasured until a Qwen-family benchmark is recorded.")
    log_production_dataset_inventory()

    stage_runners = {
        "behavioral": run_behavioral,
        "v1": run_v1,
        "v2": run_v2,
        "v3": run_v3,
    }

    if args.stage == "scale_validation":
        run_scale_validation(args, python_bin)
    elif args.stage == "all":
        logger.info(f"Running all stages in order: {' -> '.join(PRODUCTION_STAGE_ORDER)}")
        for stage in PRODUCTION_STAGE_ORDER:
            stage_runners[stage](args, python_bin)
    else:
        stage_runners[args.stage](args, python_bin)

    logger.info("All requested stages completed successfully!")


if __name__ == "__main__":
    main()
