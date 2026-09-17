#!/bin/bash
# =============================================================================
# Run Complete AIPsy-Affect 4-Split Evaluation across 8 Core Models
# Model Families: Qwen 2.5 (1.5B), Llama 3.2 (1B), Mistral (7B v0.1), Gemma 2 (2B)
# Conditions: Base vs. Instruct
# Stimuli: N=480 (Clinical, Neutral, Moderate, Complex Neutral)
# =============================================================================

set -e
mkdir -p v1/results/aipsy_4split_eval
mkdir -p v1/logs

STIMULI="v1/data/processed/aipsy_4split_all.csv"
OUT_DIR="v1/results/aipsy_4split_eval"

# Check if stimuli dataset exists
if [ ! -f "$STIMULI" ]; then
    echo "Preparing AIPsy-Affect 4-split stimuli..."
    python v1/scripts/prepare_aipsy_4splits.py
fi

echo "================================================================="
echo "Starting AIPsy-Affect 4-Split Comprehensive Evaluation Suite"
echo "Stimuli: $STIMULI"
echo "Output Directory: $OUT_DIR"
echo "================================================================="

# 1. Qwen 2.5 1.5B Base
echo "[1/8] Running Qwen 2.5 1.5B Base..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model Qwen/Qwen2.5-1.5B \
  --tag qwen2.5_1.5b_base \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_qwen_base.log

# 2. Qwen 2.5 1.5B Instruct
echo "[2/8] Running Qwen 2.5 1.5B Instruct..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tag qwen2.5_1.5b_instruct \
  --is-instruct \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_qwen_instruct.log

# 3. Llama 3.2 1B Base
echo "[3/8] Running Llama 3.2 1B Base..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model meta-llama/Llama-3.2-1B \
  --tag llama3.2_1b_base \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_llama_base.log

# 4. Llama 3.2 1B Instruct
echo "[4/8] Running Llama 3.2 1B Instruct..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --tag llama3.2_1b_instruct \
  --is-instruct \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_llama_instruct.log

# 5. Mistral 7B v0.1 Base
echo "[5/8] Running Mistral 7B v0.1 Base..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model mistralai/Mistral-7B-v0.1 \
  --tag mistral_7b_base \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_mistral_base.log

# 6. Mistral 7B Instruct v0.2
echo "[6/8] Running Mistral 7B Instruct v0.2..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --tag mistral_7b_instruct \
  --is-instruct \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_mistral_instruct.log

# 7. Gemma 2 2B Base
echo "[7/8] Running Gemma 2 2B Base..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model google/gemma-2-2b \
  --tag gemma2_2b_base \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_gemma_base.log

# 8. Gemma 2 2B IT
echo "[8/8] Running Gemma 2 2B IT..."
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model google/gemma-2-2b-it \
  --tag gemma2_2b_it \
  --is-instruct \
  --stimuli-path "$STIMULI" \
  --out-dir "$OUT_DIR" 2>&1 | tee v1/logs/aipsy_gemma_it.log

# Generate Summaries
echo "================================================================="
echo "Generating Comprehensive AIPsy-Affect 4-Split Report..."
echo "================================================================="
python v1/scripts/summarize_aipsy_4split.py --results-dir "$OUT_DIR"

echo "All AIPsy 4-split evaluations and summaries completed successfully!"
