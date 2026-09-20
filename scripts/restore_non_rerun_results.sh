#!/usr/bin/env bash
# ==============================================================================
# restore_non_rerun_results.sh
#
# 再実行が不要なステージ（V2 RQ3 Causal Map, V3 RQ1 State Induction）の結果ファイルを
# old_results から正規の results ディレクトリへ復元し、
# production スクリプト実行時に不要な再計算が走るのを防ぎます。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== Restoring Non-Rerun Results (V2 RQ3 & V3 RQ1) ==="

# 1. V2 RQ3 (Causal Map)
mkdir -p v2/results/raw v2/results/derived/pair_level
if [ -d "old_results/archive_20260921/v2_rq4" ]; then
    cp -v old_results/archive_20260921/v2_rq4/*causal_map*.json v2/results/raw/ 2>/dev/null || true
    if [ -d "old_results/archive_20260921/v2_rq4/pair_level" ]; then
        cp -v old_results/archive_20260921/v2_rq4/pair_level/* v2/results/derived/pair_level/ 2>/dev/null || true
    fi
fi

# 2. V3 RQ1 (State Induction & Gate Decision)
mkdir -p v3/results/raw v3/results/derived
if [ -d "old_results/v3" ]; then
    cp -v old_results/v3/raw/v3_rq1_results.json v3/results/raw/ 2>/dev/null || true
    cp -v old_results/v3/raw/manifest_rq1_*.json v3/results/raw/ 2>/dev/null || true
    cp -v old_results/v3/derived/v3_gate_decision.json v3/results/derived/ 2>/dev/null || true
fi

echo "=== Restoration Complete ==="
