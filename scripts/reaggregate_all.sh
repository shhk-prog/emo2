#!/usr/bin/env bash
# scripts/reaggregate_all.sh
# 再集計対象（V1 Phase A, V2 RQ1/RQ2）の一括実行スクリプト
# モデル推論（GPU）は不要で、既存のキャッシュ／生データから再集計を実行します。

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== [1/2] Reaggregating V1 Phase A (Geometry & Raw Transfer Scores) ==="
.venv/bin/python scripts/reaggregate_v1_phase_a.py

echo ""
echo "=== [2/2] Reaggregating V2 RQ1/RQ2 (Matched-Plain Primary & Bootstrap CI) ==="
.venv/bin/python scripts/reaggregate_v2_summary.py

echo ""
echo "=== All Reaggregations Completed Successfully! ==="
