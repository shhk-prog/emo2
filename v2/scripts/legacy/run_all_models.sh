#!/bin/bash

# 全モデルに対して抽出と尤度計算を行うスクリプト

set -e

# Models
MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "google/gemma-2-2b"
)

INSTRUCT_MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "google/gemma-2-2b-it"
)

echo "Starting evaluation across all model families..."

for i in "${!MODELS[@]}"; do
    BASE_MODEL="${MODELS[$i]}"
    INST_MODEL="${INSTRUCT_MODELS[$i]}"
    
    echo "================================================="
    echo " Evaluating Base Model: $BASE_MODEL"
    echo "================================================="
    PYTHONPATH=. python v2/scripts/run_extract_and_likelihood.py --model "$BASE_MODEL" --limit 0
    
    echo "================================================="
    echo " Evaluating Instruct Model: $INST_MODEL"
    echo "================================================="
    PYTHONPATH=. python v2/scripts/run_extract_and_likelihood.py --model "$INST_MODEL" --is_instruct --limit 0
    
    echo "Done with pair $i."
done

echo "All models evaluated successfully."
