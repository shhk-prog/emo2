#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

echo "=== Running Strict Patching Screening ==="
python3 v2/scripts/run_strict_patching_screening.py

echo "=== Running Strict Path Patching ==="
python3 v2/scripts/run_strict_path_patching.py

echo "=== Running Unembedding Swap ==="
python3 v2/scripts/run_unembedding_swap.py

echo "=== Done ==="
