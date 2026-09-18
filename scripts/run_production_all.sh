#!/usr/bin/env bash
# ==============================================================================
# run_production_all.sh
#
# Master Production Evaluation Runner across All Stages (Behavioral, V1, V2, V3)
# for 4 Primary Small Families (Qwen 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B)
#
# Usage:
#   bash scripts/run_production_all.sh [DEVICE] [--dry-run]
# Example:
#   bash scripts/run_production_all.sh cuda:0
#   bash scripts/run_production_all.sh cuda:0 --dry-run
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
MASTER_LOG="${LOG_DIR}/production_all_master_${TIMESTAMP}.log"

echo "==================================================================" | tee -a "${MASTER_LOG}"
echo "MASTER PRODUCTION PIPELINE EXECUTION STARTED" | tee -a "${MASTER_LOG}"
echo "Timestamp : $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${MASTER_LOG}"
echo "Device    : ${DEVICE}" | tee -a "${MASTER_LOG}"
echo "Dry Run   : ${DRY_RUN:-false}" | tee -a "${MASTER_LOG}"
echo "Master Log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "Cohort    : primary_small (4 families, 8 models total)" | tee -a "${MASTER_LOG}"
echo "Stages    : Behavioral -> V1 -> V2 -> V3 (full production CSVs; counts logged at load)" | tee -a "${MASTER_LOG}"
echo "Runtime   : unmeasured. Prefer stage scripts over this master run for recoverability." | tee -a "${MASTER_LOG}"
echo "==================================================================" | tee -a "${MASTER_LOG}"

GLOBAL_START=$(date +%s)

# Stage 1: Behavioral
echo -e "\n>>> [1/4] EXECUTING BEHAVIORAL STAGE..." | tee -a "${MASTER_LOG}"
bash "${SCRIPT_DIR}/run_production_behavioral.sh" "${DEVICE}" "${DRY_RUN}" 2>&1 | tee -a "${MASTER_LOG}"

# Stage 2: V1
echo -e "\n>>> [2/4] EXECUTING V1 STAGE..." | tee -a "${MASTER_LOG}"
bash "${SCRIPT_DIR}/run_production_v1.sh" "${DEVICE}" "${DRY_RUN}" 2>&1 | tee -a "${MASTER_LOG}"

# Stage 3: V2
echo -e "\n>>> [3/4] EXECUTING V2 STAGE..." | tee -a "${MASTER_LOG}"
bash "${SCRIPT_DIR}/run_production_v2.sh" "${DEVICE}" "${DRY_RUN}" 2>&1 | tee -a "${MASTER_LOG}"

# Stage 4: V3
echo -e "\n>>> [4/4] EXECUTING V3 STAGE..." | tee -a "${MASTER_LOG}"
bash "${SCRIPT_DIR}/run_production_v3.sh" "${DEVICE}" "${DRY_RUN}" 2>&1 | tee -a "${MASTER_LOG}"

GLOBAL_END=$(date +%s)
GLOBAL_ELAPSED=$((GLOBAL_END - GLOBAL_START))

echo "==================================================================" | tee -a "${MASTER_LOG}"
echo "ALL PRODUCTION STAGES COMPLETED SUCCESSFULLY!" | tee -a "${MASTER_LOG}"
echo "Total Wall Time: $((GLOBAL_ELAPSED / 3600))h $(((GLOBAL_ELAPSED % 3600) / 60))m $((GLOBAL_ELAPSED % 60))s" | tee -a "${MASTER_LOG}"
echo "Finished at    : $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${MASTER_LOG}"
echo "==================================================================" | tee -a "${MASTER_LOG}"
