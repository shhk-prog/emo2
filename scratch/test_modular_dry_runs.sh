#!/bin/bash
set -e

PYTHON=".venv/bin/python"

echo "=== 1. Testing Behavioral Dry-Runs ==="
$PYTHON behavioral/primary/run_behavioral_emobank.py --model "Qwen/Qwen2.5-1.5B-Instruct" --tag "qwen_instruct" --dry-run
$PYTHON behavioral/primary/run_behavioral_aipsy.py --model "Qwen/Qwen2.5-1.5B-Instruct" --tag "qwen_instruct" --dry-run

echo "=== 2. Testing V1 Dry-Runs ==="
$PYTHON v1/primary/run_phase_a.py --model-id "Qwen/Qwen2.5-1.5B-Instruct" --model-prefix "qwen_instruct" --dry-run
$PYTHON v1/primary/run_phase_b.py --model-id "Qwen/Qwen2.5-1.5B-Instruct" --model-prefix "qwen_instruct" --dry-run
$PYTHON v1/primary/run_phase_c.py --model-id "Qwen/Qwen2.5-1.5B-Instruct" --model-prefix "qwen_instruct" --dry-run
$PYTHON v1/primary/phase_c/run_e6_specialization.py --model-id "Qwen/Qwen2.5-1.5B-Instruct" --model-prefix "qwen_instruct" --dry-run

echo "=== 3. Testing V2 Dry-Runs ==="
$PYTHON v2/primary/run_rq1_rq2_cross_decoding.py --dry-run --family qwen
$PYTHON v2/primary/run_rq3_causal_map.py --dry-run --family qwen
$PYTHON v2/primary/run_rq4_recovery_patching.py --dry-run --family qwen

echo "=== 4. Testing V3 Dry-Runs ==="
$PYTHON v3/primary/run_rq1_state_induction.py --dry-run --family qwen
$PYTHON v3/primary/run_rq2_spatiotemporal_maps.py --dry-run --family qwen
$PYTHON v3/primary/run_rq3_path_mediation.py --dry-run --family qwen
$PYTHON v3/primary/run_confirmatory_replication.py --dry-run --family qwen

echo "=== ALL DRY-RUNS COMPLETED SUCCESSFULLY ==="
