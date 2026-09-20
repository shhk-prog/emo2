#!/usr/bin/env bash
# ==============================================================================
# archive_targets_for_rerun.sh
#
# 指定された再実行対象ステージの結果を old_results/archive_20260921_audit/ へ安全に退避移動します。
# 対象:
#   1. V2 RQ4 (Recovery Patching)
#   2. V3 全体 (RQ1〜Confirmatory)
#   3. V1 Phase A (Probing & Geometry)
#   4. V1 Phase B (Semantic Control Audit)
#   5. V1 Phase C E6 (Specialization & LMM)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ARCHIVE_DIR="old_results/archive_20260921_audit"
mkdir -p "${ARCHIVE_DIR}"
echo "=== Archiving Re-run Targets to ${ARCHIVE_DIR} ==="

# 1. V2 RQ4
mkdir -p "${ARCHIVE_DIR}/v2_rq4"
if [ -d "v2/results/raw" ]; then
    find v2/results/raw/ -maxdepth 1 -name "v2_recovery_*.json" -exec mv -v {} "${ARCHIVE_DIR}/v2_rq4/" \; 2>/dev/null || true
    find v2/results/raw/ -maxdepth 1 -name "manifest_recovery_*.json" -exec mv -v {} "${ARCHIVE_DIR}/v2_rq4/" \; 2>/dev/null || true
fi

# 2. V3 全体
mkdir -p "${ARCHIVE_DIR}/v3/raw" "${ARCHIVE_DIR}/v3/derived"
if [ -d "v3/results/raw" ]; then
    find v3/results/raw/ -mindepth 1 -maxdepth 1 ! -name ".gitkeep" -exec mv -v {} "${ARCHIVE_DIR}/v3/raw/" \; 2>/dev/null || true
    touch v3/results/raw/.gitkeep
fi
if [ -d "v3/results/derived" ]; then
    find v3/results/derived/ -mindepth 1 -maxdepth 1 ! -name ".gitkeep" -exec mv -v {} "${ARCHIVE_DIR}/v3/derived/" \; 2>/dev/null || true
    touch v3/results/derived/.gitkeep
fi

# 3. V1 Phase A
mkdir -p "${ARCHIVE_DIR}/v1_phase_a"
if [ -d "v1/results/derived/v1_phase_a" ]; then
    find v1/results/derived/v1_phase_a/ -mindepth 1 -maxdepth 1 ! -name ".gitkeep" -exec mv -v {} "${ARCHIVE_DIR}/v1_phase_a/" \; 2>/dev/null || true
    touch v1/results/derived/v1_phase_a/.gitkeep
fi

# 4. V1 Phase B
mkdir -p "${ARCHIVE_DIR}/v1_phase_b"
if [ -d "v1/results/derived/v1_phase_b" ]; then
    find v1/results/derived/v1_phase_b/ -mindepth 1 -maxdepth 1 ! -name ".gitkeep" -exec mv -v {} "${ARCHIVE_DIR}/v1_phase_b/" \; 2>/dev/null || true
    touch v1/results/derived/v1_phase_b/.gitkeep
fi

# 5. V1 Phase C E6
mkdir -p "${ARCHIVE_DIR}/v1_phase_c_e6"
if [ -d "v1/results/derived" ]; then
    find v1/results/derived/ -maxdepth 1 -name "*e6*" -exec mv -v {} "${ARCHIVE_DIR}/v1_phase_c_e6/" \; 2>/dev/null || true
    find v1/results/derived/ -maxdepth 1 -name "*specialization*" -exec mv -v {} "${ARCHIVE_DIR}/v1_phase_c_e6/" \; 2>/dev/null || true
fi

echo "=== Archiving Complete! Ready for Clean Production Re-run ==="
