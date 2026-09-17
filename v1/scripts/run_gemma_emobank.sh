#!/bin/bash
# Run Gemma-2-2B Base and Instruct on EmoBank

set -e

source .venv/bin/activate

OUT_DIR="v1/results/recognition_baseline"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "Starting EmoBank Evaluation on Gemma-2-2B Models (N=321)"
echo "=========================================================="

# 1. Gemma-2-2B Base
echo "--- Running Gemma-2-2B Base ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "google/gemma-2-2b" \
    --tag "gemma2_2b_base" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

# 2. Gemma-2-2B Instruct
echo "--- Running Gemma-2-2B-it Instruct ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "google/gemma-2-2b-it" \
    --is_instruct \
    --tag "gemma2_2b_instruct" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

echo "=========================================================="
echo "Gemma-2-2B evaluations completed. Summarizing..."
echo "=========================================================="

python v1/scripts/summarize_cross_family_recognition.py --results-dir "$OUT_DIR"
