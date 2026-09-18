#!/usr/bin/env bash
set -e

# ==============================================================================
# V1 Phase A: All-Models Full-Layer Probing & Representation Geometry
# Evaluates:
#   - E1: Shared Decodability (EmoBank 1,006 items & AIPsy-Affect 2,196 items)
#   - E2: Shared Geometry (Direct Cross-Decoding, RSA, Orthogonal Procrustes)
# ==============================================================================

# Resolve project root robustly
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/.venv" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "/mnt/nas/home/hiromi/src/emo/.venv" ]; then
    PROJECT_ROOT="/mnt/nas/home/hiromi/src/emo"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "$PROJECT_ROOT"

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "$PROJECT_ROOT/.venv" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo "Error: Virtual environment (.venv) not found in $PROJECT_ROOT."
        exit 1
    fi
fi

LIMIT=${1:-0}
BATCH_SIZE=${2:-16}
DEVICE=${3:-"cuda"}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="v1/results/logs/phase_a"
mkdir -p "$LOG_DIR"
GLOBAL_LOG="$LOG_DIR/phase_a_run_${TIMESTAMP}.log"

echo "========================================================================" | tee -a "$GLOBAL_LOG"
echo "Starting V1 Phase A (All Models, Full Layers, Full Dataset)" | tee -a "$GLOBAL_LOG"
echo "Timestamp: $TIMESTAMP | Log: $GLOBAL_LOG" | tee -a "$GLOBAL_LOG"
echo "Limit: $LIMIT (0 = full dataset) | Batch Size: $BATCH_SIZE | Device: $DEVICE" | tee -a "$GLOBAL_LOG"
echo "========================================================================" | tee -a "$GLOBAL_LOG"

MODELS=(
    "Qwen/Qwen2.5-1.5B:qwen2.5_1.5b_base:false"
    "Qwen/Qwen2.5-1.5B-Instruct:qwen2.5_1.5b_instruct:true"
    "meta-llama/Llama-3.2-1B:llama3.2_1b_base:false"
    "meta-llama/Llama-3.2-1B-Instruct:llama3.2_1b_instruct:true"
    "google/gemma-2-2b:gemma2_2b_base:false"
    "google/gemma-2-2b-it:gemma2_2b_instruct:true"
    "mistralai/Mistral-7B-v0.1:mistral7b_v0.1_base:false"
    "mistralai/Mistral-7B-Instruct-v0.1:mistral7b_v0.1_instruct:true"
)

TOTAL_MODELS=${#MODELS[@]}
CURRENT=0

for m in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    IFS=":" read -r model_id model_prefix is_instruct <<< "$m"

    MODEL_LOG="$LOG_DIR/${model_prefix}_${TIMESTAMP}.log"

    echo "" | tee -a "$GLOBAL_LOG"
    echo "[$CURRENT/$TOTAL_MODELS] Running Phase A for: $model_id ($model_prefix)" | tee -a "$GLOBAL_LOG"
    echo "Model Log: $MODEL_LOG" | tee -a "$GLOBAL_LOG"
    echo "------------------------------------------------------------------------" | tee -a "$GLOBAL_LOG"

    CMD="python v1/scripts/run_v1_phase_a_probe.py \
        --model-id \"$model_id\" \
        --model-prefix \"$model_prefix\" \
        --dataset \"both\" \
        --limit $LIMIT \
        --batch-size $BATCH_SIZE \
        --device \"$DEVICE\""

    if [ "$is_instruct" = "true" ]; then
        CMD="$CMD --is-instruct"
    fi

    # Execute and log to both model-specific log and global log while displaying in console
    eval "$CMD" 2>&1 | tee "$MODEL_LOG" | tee -a "$GLOBAL_LOG"
    echo "✓ Finished Phase A for $model_prefix" | tee -a "$GLOBAL_LOG"
done

echo "" | tee -a "$GLOBAL_LOG"
echo "========================================================================" | tee -a "$GLOBAL_LOG"
echo "All Phase A runs completed successfully!" | tee -a "$GLOBAL_LOG"
echo "Results stored in: v1/results/derived/v1_phase_a/" | tee -a "$GLOBAL_LOG"
echo "Logs stored in: $LOG_DIR/" | tee -a "$GLOBAL_LOG"
echo "========================================================================" | tee -a "$GLOBAL_LOG"
