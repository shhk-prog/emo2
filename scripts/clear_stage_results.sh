#!/usr/bin/env bash
# ==============================================================================
# clear_stage_results.sh (Safe Archive Version - AGENTS.md 1.3 / Audit Item 14)
#
# 実験結果を上書き・削除せず、archive/<timestamp>/ 配下へ安全に退避移動します。
# results ディレクトリには .gitkeep を残して安全に再初期化します。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
ARCHIVE_BASE="archive/results_${TIMESTAMP}"

echo "=== Safe Archiving Stage Results to ${ARCHIVE_BASE} ==="
mkdir -p "${ARCHIVE_BASE}"

TARGETS=(
    "behavioral/results"
    "v1/results"
    "v2/results"
    "v3/results"
)

for dir in "${TARGETS[@]}"; do
    if [ ! -d "${dir}" ]; then
        echo "[SKIP] missing: ${dir}"
        continue
    fi
    stage_name="$(dirname "${dir}")"
    dst_stage="${ARCHIVE_BASE}/${stage_name}"
    mkdir -p "${dst_stage}"

    echo "--- Archiving ${dir} -> ${dst_stage} ---"
    for item in "${dir}"/*; do
        [ -e "${item}" ] || continue
        base_item="$(basename "${item}")"
        if [ "${base_item}" != ".gitkeep" ]; then
            mv "${item}" "${dst_stage}/"
            echo "  Moved: ${base_item}"
        fi
    done

    mkdir -p "${dir}/raw" "${dir}/derived"
    touch "${dir}/.gitkeep" "${dir}/raw/.gitkeep" "${dir}/derived/.gitkeep"
done

if [ -d "results/logs" ]; then
    mkdir -p "${ARCHIVE_BASE}/logs"
    find results/logs -type f -name '*.log' -exec mv {} "${ARCHIVE_BASE}/logs/" \; 2>/dev/null || true
fi

echo "=== Archiving Complete. Safe placeholders (.gitkeep) created. ==="
