#!/usr/bin/env bash
# ==============================================================================
# run_production_v1.sh
#
# Production V1 Evaluation Runner across 4 Primary Small Families
# (Qwen 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B: 8 models total)
# Evaluates:
#   1. Phase A: Representation Probing & Geometry (EmoBank 1,000 + AIPsy 480 full)
#   2. Phase B: Semantic vs Lexical Matched Controls (400 pairs full, d ≈ 0.5)
#   3. Phase C: Causal Mapping (E3) & Interchangeability (E4) (AIPsy 480 full)
#   4. Phase C E6: Double Dissociation & LMM (AIPsy 480 full)
#   5. Summarize Phase C Results
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

DEVICE="${1:-cuda:0}"
DRY_RUN="${2:-}"

LOG_DIR="results/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/production_v1_${TIMESTAMP}.log"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "Starting Production V1 Pipeline Evaluation" | tee -a "${LOG_FILE}"
echo "Timestamp : $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${LOG_FILE}"
echo "Device    : ${DEVICE}" | tee -a "${LOG_FILE}"
echo "Log File  : ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Cohort    : primary_small (4 families, 8 models)" | tee -a "${LOG_FILE}"
echo "Pipeline  : Phase A -> Phase B -> Phase C (E3/E4) -> E6 -> Summarize" | tee -a "${LOG_FILE}"
echo "Data Scale: Full production datasets without subsampling" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

CMD=(python -m affective_empathy_eval.run --stage v1 --model-set primary_small --device "${DEVICE}")
if [ "${DRY_RUN}" = "--dry-run" ]; then
    CMD+=(--dry-run)
fi

START_SEC=$(date +%s)
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "V1 Pipeline Evaluation completed successfully!" | tee -a "${LOG_FILE}"
echo "Elapsed Time: ${ELAPSED} seconds" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
