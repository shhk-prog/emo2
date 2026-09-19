#!/usr/bin/env python3
"""
tests/run_all_v3_dryruns.py

V3 Dry-run 実行・検証テストハーネス
Production Primary スクリプト (v3/primary/run_*.py) を --dry-run 付きで実行し、
結果 JSON が dry_run サブディレクトリ配下に完全・正確に生成されることを検証する。
"""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v3_dryrun_test")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable


def run_script(args: list[str]):
    cmd = [PYTHON_BIN] + args
    logger.info(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Command failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {res.returncode}")
    return res.stdout


def test_v3_rq1_state_induction_dryrun():
    logger.info("Testing V3-RQ1 State Induction dry-run...")
    run_script(["v3/primary/run_rq1_state_induction.py", "--dry-run"])
    raw_path = PROJECT_ROOT / "v3/results/raw/dry_run/v3_rq1_results.json"
    gate_path = PROJECT_ROOT / "v3/results/derived/dry_run/v3_gate_decision.json"
    assert raw_path.exists(), f"Missing {raw_path}"
    assert gate_path.exists(), f"Missing {gate_path}"

    with open(gate_path, "r", encoding="utf-8") as f:
        gate_data = json.load(f)
    assert gate_data.get("decision") in ["GO", "GO (Valence-only)", "GO (Arousal-only)"]
    logger.info("V3-RQ1 State Induction dry-run passed.")


def test_v3_rq2_spatiotemporal_maps_dryrun():
    logger.info("Testing V3-RQ2 Spatiotemporal Maps dry-run...")
    run_script(["v3/primary/run_rq2_spatiotemporal_maps.py", "--dry-run"])
    maps_path = PROJECT_ROOT / "v3/results/raw/dry_run/v3_discovery_spatiotemporal_maps_qwen.json"
    assert maps_path.exists(), f"Missing {maps_path}"

    with open(maps_path, "r", encoding="utf-8") as f:
        maps_data = json.load(f)
    assert "maps" in maps_data
    assert "D_V" in maps_data["maps"]
    assert "C_V" in maps_data["maps"]
    assert maps_data.get("causal_reference_alpha") == 1.0
    logger.info("V3-RQ2 Spatiotemporal Maps dry-run passed.")


def test_v3_rq3_path_mediation_dryrun():
    logger.info("Testing V3-RQ3 Path Mediation dry-run...")
    run_script(["v3/primary/run_rq3_path_mediation.py", "--dry-run"])
    med_path = PROJECT_ROOT / "v3/results/raw/dry_run/v3_path_mediation_qwen.json"
    assert med_path.exists(), f"Missing {med_path}"

    with open(med_path, "r", encoding="utf-8") as f:
        med_data = json.load(f)
    assert "confirmation" in med_data
    assert "valence" in med_data["confirmation"]
    logger.info("V3-RQ3 Path Mediation dry-run passed.")


def test_v3_confirmatory_replication_dryrun():
    logger.info("Testing V3 Confirmatory Replication dry-run...")
    run_script(["v3/primary/run_confirmatory_replication.py", "--dry-run"])
    conf_raw = PROJECT_ROOT / "v3/results/raw/dry_run/v3_confirmatory_llama.json"
    conf_summary = PROJECT_ROOT / "v3/results/derived/dry_run/v3_cross_model_replication_summary.json"
    assert conf_raw.exists(), f"Missing {conf_raw}"
    assert conf_summary.exists(), f"Missing {conf_summary}"

    with open(conf_summary, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert "primary_effect_estimates" in summary_data
    logger.info("V3 Confirmatory Replication dry-run passed.")


def main():
    logger.info("Starting all V3 dry-run tests using production primary entrypoints...")
    test_v3_rq1_state_induction_dryrun()
    test_v3_rq2_spatiotemporal_maps_dryrun()
    test_v3_rq3_path_mediation_dryrun()
    test_v3_confirmatory_replication_dryrun()
    logger.info("All V3 dry-run tests successfully passed!")


if __name__ == "__main__":
    main()
