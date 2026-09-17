#!/bin/bash
# Run Mistral-7B Base and Instruct on EmoBank

set -e

source .venv/bin/activate

OUT_DIR="v1/results/recognition_baseline"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "Starting EmoBank Evaluation on Mistral Models (N=321)"
echo "=========================================================="

# 1. Mistral-7B Base
echo "--- Running Mistral-7B-v0.1 Base ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --tag "mistral7b_v0.1_base" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

# 2. Mistral-7B Instruct
echo "--- Running Mistral-7B-Instruct-v0.2 ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --is_instruct \
    --tag "mistral7b_v0.2_instruct" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

echo "=========================================================="
echo "Mistral evaluations completed. Summarizing..."
echo "=========================================================="

python v1/scripts/summarize_cross_family_recognition.py --results-dir "$OUT_DIR"
