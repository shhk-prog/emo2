#!/usr/bin/env bash
# scripts/run_production_reruns.sh
# 監査修正反映後の本番再実行スクリプト (GPU環境推奨)
#
# 対象:
# [必須]
#   1. V1 Phase B (Reader & Self)
#   2. V3 RQ2 (Spatiotemporal 4-Maps & Causal Sites)
#   3. V3 RQ3 (Path Mediation & Frozen Confirmatory Sites)
#   4. V3 Confirmatory Replication (Llama, Gemma, OLMo)
# [推奨]
#   5. V2 RQ4 (Distribution Recovery Patching)

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${1:-cuda:0}"

# GPU利用可能チェック
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU Check] Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[Warning] nvidia-smi not found. Target device: ${DEVICE}"
fi

PYTHON=".venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] .venv is required (AGENTS.md 1.1). Please create and activate project virtual environment." >&2
    exit 1
fi

echo ""
echo "================================================================="
echo "=== Step 1: V1 Phase B (Semantic Control Audit: Reader & Self) ==="
echo "================================================================="
FAMILIES=("qwen" "llama" "gemma" "olmo")
for fam in "${FAMILIES[@]}"; do
    echo "--- V1 Phase B (Base): Family=${fam}, task-type=reader ---"
    ${PYTHON} v1/primary/run_phase_b.py --family "${fam}" --task-type reader --device "${DEVICE}" --force

    echo "--- V1 Phase B (Base): Family=${fam}, task-type=self ---"
    ${PYTHON} v1/primary/run_phase_b.py --family "${fam}" --task-type self --device "${DEVICE}" --force

    echo "--- V1 Phase B (Instruct): Family=${fam}, task-type=reader ---"
    ${PYTHON} v1/primary/run_phase_b.py --family "${fam}" --is-instruct --task-type reader --device "${DEVICE}" --force

    echo "--- V1 Phase B (Instruct): Family=${fam}, task-type=self ---"
    ${PYTHON} v1/primary/run_phase_b.py --family "${fam}" --is-instruct --task-type self --device "${DEVICE}" --force
done

echo ""
echo "================================================================="
echo "=== Step 2: V3 RQ2 (Spatiotemporal 4-Maps & Causal Discovery) ==="
echo "================================================================="
# Discovery Model: Qwen 2.5 1.5B Instruct
${PYTHON} v3/primary/run_rq2_spatiotemporal_maps.py --family qwen --device "${DEVICE}" --force

echo ""
echo "================================================================="
echo "=== Step 3: V3 RQ3 (Path Mediation & Frozen Confirmation Sites)=="
echo "================================================================="
# Discovery Model: Qwen 2.5 1.5B Instruct
${PYTHON} v3/primary/run_rq3_path_mediation.py --family qwen --device "${DEVICE}" --force

echo ""
echo "================================================================="
echo "=== Step 4: V3 Confirmatory Replication across Hold-out Models =="
echo "================================================================="
# Pre-registered Confirmatory Models: Llama 3.2, Gemma 3, OLMo 2
CONF_FAMILIES=("llama" "gemma" "olmo")
for fam in "${CONF_FAMILIES[@]}"; do
    echo "--- V3 Confirmatory: Family=${fam} ---"
    ${PYTHON} v3/primary/run_confirmatory_replication.py --family "${fam}" --device "${DEVICE}" --force
done

echo ""
echo "================================================================="
echo "=== Step 5: V2 RQ4 (Recovery Patching across Families) [推奨]  ==="
echo "================================================================="
for fam in "${FAMILIES[@]}"; do
    echo "--- V2 RQ4: Family=${fam} ---"
    ${PYTHON} v2/primary/run_rq4_recovery_patching.py --family "${fam}" --device "${DEVICE}" --force
done

echo ""
echo "================================================================="
echo "=== ALL PRODUCTION RERUNS COMPLETED SUCCESSFULLY! ==="
echo "================================================================="
