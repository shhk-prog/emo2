#!/usr/bin/env bash
# ==============================================================================
# run_production_v3.sh
#
# Production V3 Evaluation Runner across 4 Primary Small Families
# (Target: Qwen 1.5B, Confirmatory: Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B)
# Evaluates:
#   1. RQ1: State Induction & Go/No-Go Gate
#   2. RQ2: Spatiotemporal 4-Maps
#   3. RQ3: Mediated attenuation (Discovery / Confirmation split)
#   4. Step 7: Confirmatory Cross-Architecture Replication
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
DRY_RUN="${2:-}"
FORCE_AFTER_NO_GO="${3:-}"

LOG_DIR="results/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/production_v3_${TIMESTAMP}.log"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "Starting Production V3 Pipeline Evaluation" | tee -a "${LOG_FILE}"
echo "Timestamp : $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${LOG_FILE}"
echo "Device    : ${DEVICE}" | tee -a "${LOG_FILE}"
echo "Log File  : ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Cohort    : primary_small (Target: Qwen | Confirmatory: Llama, Gemma, OLMo)" | tee -a "${LOG_FILE}"
echo "Pipeline  : RQ1 (Gate) -> RQ2 (4-Maps) -> RQ3 (Mediation) -> Confirmatory" | tee -a "${LOG_FILE}"
echo "Data Scale: Full production CSVs (actual counts logged by the Python runner)" | tee -a "${LOG_FILE}"
echo "Runtime   : unmeasured (do not use pre-benchmark hour estimates)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

CMD=(python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device "${DEVICE}")
if [ "${DRY_RUN}" = "--dry-run" ]; then
    CMD+=(--dry-run)
fi
if [ "${DRY_RUN}" = "--force-after-no-go" ] || [ "${FORCE_AFTER_NO_GO}" = "--force-after-no-go" ]; then
    CMD+=(--force-after-no-go)
fi

START_SEC=$(date +%s)
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "V3 Pipeline Evaluation completed successfully!" | tee -a "${LOG_FILE}"
echo "Elapsed Time: ${ELAPSED} seconds" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
