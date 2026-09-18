#!/usr/bin/env bash
# ==============================================================================
# clear_stage_results.sh
#
# Clear generated results under stage directories before a formal re-run.
# Keeps .gitkeep files and recreates raw/derived placeholders.
# Does not touch data/, configs/, or source.
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
    # Delete generated files; keep any .gitkeep in the tree.
    find "${dir}" -type f ! -name '.gitkeep' -print -delete
    # Drop empty nested directories left after file deletion.
    find "${dir}" -depth -type d -empty -delete
    mkdir -p "${dir}/raw" "${dir}/derived"
    touch "${dir}/.gitkeep" "${dir}/raw/.gitkeep" "${dir}/derived/.gitkeep"
done

if [ -d "results/logs" ]; then
    echo "--- results/logs ---"
    find results/logs -type f -name '*.log' -print -delete || true
fi

echo "Done. Stage result directories now contain only .gitkeep placeholders."
