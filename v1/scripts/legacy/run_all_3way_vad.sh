#!/bin/bash
# Run 3-Way VAD Evaluation on EmoBank for All 8 Models
# Tasks: 1. Writer Estimation, 2. Reader Prediction, 3. Self-Report
# Evaluation: 729 VAD candidates in {1..9}^3

set -e
source .venv/bin/activate

OUT_DIR="v1/results/emobank_3way_vad_test1k"
STIM_PATH="v1/data/processed/stimuli_vad_3way_test1k.csv"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "Starting EmoBank 3-Way VAD Evaluation Pipeline (729 candidates)"
echo "=========================================================="

# 1. Prepare 3-Way EmoBank dataset (if not already done)
if [ ! -f "$STIM_PATH" ]; then
    echo "Preparing 3-way stimuli dataset..."
    python v1/scripts/prepare_3way_emobank.py
fi

# Qwen 1.5B Base & Instruct
echo "=== Running Qwen2.5-1.5B (Base) ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --tag "qwen2.5_1.5b_base" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

echo "=== Running Qwen2.5-1.5B-Instruct ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --is_instruct \
    --tag "qwen2.5_1.5b_instruct" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

# Llama 3.2 1B Base & Instruct
echo "=== Running Llama-3.2-1B (Base) ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "meta-llama/Llama-3.2-1B" \
    --tag "llama3.2_1b_base" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

echo "=== Running Llama-3.2-1B-Instruct ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --is_instruct \
    --tag "llama3.2_1b_instruct" \
    --dtype "float16" \
    --out-dir "$OUT_DIR"

# Mistral 7B v0.1 Base & Instruct (Same Version!)
echo "=== Running Mistral-7B-v0.1 (Base) ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --tag "mistral7b_v0.1_base" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

echo "=== Running Mistral-7B-Instruct-v0.1 ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "mistralai/Mistral-7B-Instruct-v0.1" \
    --is_instruct \
    --tag "mistral7b_v0.1_instruct" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

# Gemma 2 2B Base & Instruct
echo "=== Running Gemma-2-2B (Base) ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "google/gemma-2-2b" \
    --tag "gemma2_2b_base" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

echo "=== Running Gemma-2-2B-it ==="
python v1/scripts/run_3way_vad_evaluation.py \
    --model "google/gemma-2-2b-it" \
    --is_instruct \
    --tag "gemma2_2b_instruct" \
    --dtype "bfloat16" \
    --out-dir "$OUT_DIR"

echo "=========================================================="
echo "All 8 models finished! Summarizing..."
echo "=========================================================="

python v1/scripts/summarize_3way_vad.py --results-dir "$OUT_DIR"
