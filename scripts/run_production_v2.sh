#!/usr/bin/env bash
# ==============================================================================
# run_production_v2.sh
#
# Production V2 Evaluation Runner across 4 Primary Small Families
# (Qwen 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B: Base vs Instruct comparisons)
# Evaluates:
#   1. RQ1 & RQ2: Cross-decoding & Intrinsic Representation Geometry
#   2. RQ3: Causal Mapping & Peak Dissociation
#   3. RQ4: Distribution Recovery Patching & Optimal Transport
# Dataset sizes are logged from the loaded CSVs. Wall-clock time is unmeasured.
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
shift || true
EXTRA_ARGS=("$@")

LOG_DIR="results/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/production_v2_${TIMESTAMP}.log"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "Starting Production V2 Pipeline Evaluation" | tee -a "${LOG_FILE}"
echo "Timestamp : $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${LOG_FILE}"
echo "Device    : ${DEVICE}" | tee -a "${LOG_FILE}"
echo "Extra Args: ${EXTRA_ARGS[*]:-none}" | tee -a "${LOG_FILE}"
echo "Log File  : ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Cohort    : primary_small (4 families: Qwen, Llama, Gemma, OLMo)" | tee -a "${LOG_FILE}"
echo "Pipeline  : RQ1/RQ2 (Cross-decoding) -> RQ3 (Causal Map) -> RQ4 (Recovery Patching)" | tee -a "${LOG_FILE}"
echo "Data Scale: Full production CSVs (actual counts logged by the Python runner)" | tee -a "${LOG_FILE}"
echo "Runtime   : unmeasured (do not use pre-benchmark hour estimates)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

CMD=(python -m affective_empathy_eval.run --stage v2 --model-set primary_small --device "${DEVICE}")
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

START_SEC=$(date +%s)
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "V2 Pipeline Evaluation completed successfully!" | tee -a "${LOG_FILE}"
echo "Elapsed Time: ${ELAPSED} seconds" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
