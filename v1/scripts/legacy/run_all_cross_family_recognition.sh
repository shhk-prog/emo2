#!/bin/bash
# Batch Runner for Cross-Family EmoBank Recognition and Self-Report Evaluation
# Evaluates Base vs. Instruct models across 4 families: Qwen2.5, Llama-3.2, Gemma-2, Mistral-7B

set -e

# Activate project virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

LIMIT_ARG=""
if [ -n "$1" ]; then
    LIMIT_ARG="--limit $1"
    echo "Running with limit: $1"
fi

OUT_DIR="v1/results/recognition_baseline"
mkdir -p "$OUT_DIR"

# List of models: "TAG|MODEL_PATH|IS_INSTRUCT|DTYPE"
MODELS=(
    # Qwen2.5 (1.5B)
    "qwen2.5_1.5b_base|Qwen/Qwen2.5-1.5B|0|float16"
    "qwen2.5_1.5b_instruct|Qwen/Qwen2.5-1.5B-Instruct|1|float16"
    
    # Llama-3.2 (1B)
    "llama3.2_1b_base|meta-llama/Llama-3.2-1B|0|float16"
    "llama3.2_1b_instruct|meta-llama/Llama-3.2-1B-Instruct|1|float16"
    
    # Gemma-2 (2B)
    "gemma2_2b_base|google/gemma-2-2b|0|bfloat16"
    "gemma2_2b_instruct|google/gemma-2-2b-it|1|bfloat16"
    
    # Mistral-7B (v0.3)
    "mistral_7b_base|mistralai/Mistral-7B-v0.3|0|bfloat16"
    "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3|1|bfloat16"
)

echo "=========================================================="
echo "Starting Cross-Family Base vs Instruct Recognition Suite"
echo "Total models to evaluate: ${#MODELS[@]}"
echo "Output directory: ${OUT_DIR}"
echo "=========================================================="

for item in "${MODELS[@]}"; do
    IFS="|" read -r TAG MODEL_PATH IS_INSTRUCT DTYPE <<< "$item"
    
    echo ""
    echo "----------------------------------------------------------"
    echo " Running Model: [${TAG}]"
    echo " Model Path  : ${MODEL_PATH}"
    echo " Instruct    : ${IS_INSTRUCT}"
    echo " Dtype       : ${DTYPE}"
    echo "----------------------------------------------------------"
    
    INSTRUCT_FLAG=""
    if [ "$IS_INSTRUCT" -eq 1 ]; then
        INSTRUCT_FLAG="--is_instruct"
    fi
    
    python v1/scripts/run_cross_family_recognition.py \
        --model "${MODEL_PATH}" \
        ${INSTRUCT_FLAG} \
        --tag "${TAG}" \
        --dtype "${DTYPE}" \
        --out-dir "${OUT_DIR}" \
        ${LIMIT_ARG}
        
    echo "[COMPLETED] Finished ${TAG}."
done

echo ""
echo "=========================================================="
echo "All Model Evaluations Completed Successfully!"
echo "Generating cross-family summary table..."
echo "=========================================================="

python v1/scripts/summarize_cross_family_recognition.py --results-dir "${OUT_DIR}"

echo "Done!"
