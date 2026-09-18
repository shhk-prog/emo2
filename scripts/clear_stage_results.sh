#!/usr/bin/env bash
# ==============================================================================
# clear_stage_results.sh
#
# Clear generated results under stage directories before a formal re-run.
# Keeps .gitkeep. Does not touch data/raw, configs, or results/logs.
#
# Usage:
#   bash scripts/clear_stage_results.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGETS=(
    "behavioral/results"
    "v1/results"
    "v2/results"
    "v3/results"
)

echo "Clearing stage results (keeping .gitkeep)..."
for dir in "${TARGETS[@]}"; do
    if [ ! -d "${dir}" ]; then
        echo "[SKIP] missing: ${dir}"
        continue
    fi
    echo "--- ${dir} ---"
    find "${dir}" -mindepth 1 ! -name '.gitkeep' -print
    # トップレベル項目だけ削除すれば配下も消える（find -delete のディレクトリ競合を避ける）
    for item in "${dir}"/* "${dir}"/.[!.]*; do
        [ -e "${item}" ] || continue
        [ "$(basename "${item}")" = ".gitkeep" ] && continue
        rm -rf "${item}"
    done
done
echo "Done. Stage result directories now contain only .gitkeep."
