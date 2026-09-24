#!/usr/bin/env bash
# ==============================================================================
# run_production_scale_ablation.sh
#
# Production V2 Scale Ablation Runner
# (scale_3b: Qwen 3B, Llama 3.2 3B)
# (scale_7b: Qwen 7B, Llama 3.1 8B, OLMo 2 7B)
#
# Usage:
#   bash scripts/run_production_scale_ablation.sh scale_3b [device] [extra_args...]
#   bash scripts/run_production_scale_ablation.sh scale_7b [device] [extra_args...]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "[INFO] Using existing virtual environment: ${VIRTUAL_ENV}"
else
    echo "[ERROR] .venv not found. Please activate the project virtual environment." >&2
    exit 1
fi

MODEL_SET="${1:-scale_3b}"
if [ "$#" -gt 0 ]; then
    shift
fi

DEVICE="${1:-cuda:0}"
if [ "$#" -gt 0 ]; then
    shift
fi

EXTRA_ARGS=("$@")

# Ablation 専用ログディレクトリ
LOG_DIR="results/ablation/${MODEL_SET}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/production_v2_${TIMESTAMP}.log"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "Starting Production V2 Scale Ablation Evaluation" | tee -a "${LOG_FILE}"
echo "Timestamp : $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${LOG_FILE}"
echo "Model Set : ${MODEL_SET}" | tee -a "${LOG_FILE}"
echo "Device    : ${DEVICE}" | tee -a "${LOG_FILE}"
echo "Python    : $(which python)" | tee -a "${LOG_FILE}"
echo "Python Ver: $(python --version)" | tee -a "${LOG_FILE}"
echo "Pip Ver   : $(python -m pip --version 2>/dev/null || true)" | tee -a "${LOG_FILE}"
echo "Git SHA   : $(git rev-parse HEAD 2>/dev/null || echo 'unknown')" | tee -a "${LOG_FILE}"
if [ -f "uv.lock" ]; then
    echo "uv.lock   : $(sha256sum uv.lock | awk '{print $1}')" | tee -a "${LOG_FILE}"
fi
echo "Extra Args: ${EXTRA_ARGS[*]:-none}" | tee -a "${LOG_FILE}"
echo "Log File  : ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Pipeline  : RQ1/RQ2 (Cross-decoding) -> RQ3 (Causal Map) -> RQ4 (Recovery Patching)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

CMD=(python -m affective_empathy_eval.run --stage v2 --model-set "${MODEL_SET}" --device "${DEVICE}")
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

START_SEC=$(date +%s)
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "Scale Ablation (${MODEL_SET}) completed successfully!" | tee -a "${LOG_FILE}"
echo "Elapsed Time: ${ELAPSED} seconds" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
