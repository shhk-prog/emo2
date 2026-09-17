#!/bin/bash
set -e

source .venv/bin/activate

OUT_DIR="v1/results/recognition_baseline"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "Starting EmoBank Evaluation on Qwen and Llama Models (N=321)"
echo "=========================================================="

# 1. Qwen2.5-1.5B Base
echo "--- Running Qwen2.5-1.5B Base ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --tag "qwen2.5_1.5b_base" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

# 2. Qwen2.5-1.5B Instruct
echo "--- Running Qwen2.5-1.5B Instruct ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --is_instruct \
    --tag "qwen2.5_1.5b_instruct" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

# 3. Llama-3.2-1B Base
echo "--- Running Llama-3.2-1B Base ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "meta-llama/Llama-3.2-1B" \
    --tag "llama3.2_1b_base" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

# 4. Llama-3.2-1B Instruct
echo "--- Running Llama-3.2-1B Instruct ---"
python v1/scripts/run_cross_family_recognition.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --is_instruct \
    --tag "llama3.2_1b_instruct" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

echo "=========================================================="
echo "Completed Qwen and Llama evaluations. Summarizing..."
echo "=========================================================="

python v1/scripts/summarize_cross_family_recognition.py --results-dir "$OUT_DIR"
