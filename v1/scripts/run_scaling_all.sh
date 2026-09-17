#!/bin/bash
# Batch Runner for Scaling & Multi-Family Experiments (No Slurm required)
# Evaluates Base vs. Instruct models across sizes (0.5B, 1.5B, 3B, 7B) and families (Qwen2.5, Llama-3.2)

set -e

# Activate virtual environment
source .venv/bin/activate

# Define Model Pairs: "TAG|BASE_MODEL|INSTRUCT_MODEL|BATCH_SIZE"
PAIRS=(
    "qwen2.5_0.5b|Qwen/Qwen2.5-0.5B|Qwen/Qwen2.5-0.5B-Instruct|384"
    "qwen2.5_1.5b|Qwen/Qwen2.5-1.5B|Qwen/Qwen2.5-1.5B-Instruct|384"
    "qwen2.5_3b|Qwen/Qwen2.5-3B|Qwen/Qwen2.5-3B-Instruct|192"
    "qwen2.5_7b|Qwen/Qwen2.5-7B|Qwen/Qwen2.5-7B-Instruct|96"
    "llama3.2_1b|meta-llama/Llama-3.2-1B|meta-llama/Llama-3.2-1B-Instruct|384"
    "llama3.2_3b|meta-llama/Llama-3.2-3B|meta-llama/Llama-3.2-3B-Instruct|192"
)

echo "=========================================================="
echo "Starting Scaling & Multi-Family Alignment Suppression Suite"
echo "Total pairs to evaluate: ${#PAIRS[@]}"
echo "=========================================================="

for item in "${PAIRS[@]}"; do
    IFS="|" read -r TAG BASE_MODEL INSTRUCT_MODEL BATCH_SIZE <<< "$item"
    
    echo ""
    echo "----------------------------------------------------------"
    echo " Running Pair: [${TAG}]"
    echo " Base    : ${BASE_MODEL}"
    echo " Instruct: ${INSTRUCT_MODEL}"
    echo " Batch   : ${BATCH_SIZE}"
    echo "----------------------------------------------------------"
    
    # Run python script for this model pair
    python scripts/run_scaling_experiments.py \
        --tag "${TAG}" \
        --base-model "${BASE_MODEL}" \
        --instruct-model "${INSTRUCT_MODEL}" \
        --batch-size "${BATCH_SIZE}" \
        --base-out-dir "results/derived/scaling" \
        --base-log-dir "logs/scaling"
        
    echo "[COMPLETED] Finished pair ${TAG}."
done

echo ""
echo "=========================================================="
echo "All Scaling Experiments Completed Successfully!"
echo "Results stored in results/derived/scaling/"
echo "Logs stored in logs/scaling/"
echo "=========================================================="
