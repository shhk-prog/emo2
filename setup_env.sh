#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh — LLM情動反応性評価実験 仮想環境自動構築スクリプト
#
# AGENTS.md 規約準拠:
#   - プロジェクト専用の仮想環境 (.venv) を構築
#   - uv が利用可能な場合は uv を最優先で使用 (超高速セットアップ)
#   - uv がない場合は標準の python3 -m venv を使用
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_TARGET_VERSION="3.12"

echo "================================================================="
echo "  Affective Empathy Evaluation: 仮想環境セットアップ開始"
echo "  作業ディレクトリ: ${SCRIPT_DIR}"
echo "================================================================="

# 0. README.md の存在確認 (hatchling build 用の安全措置)
if [ ! -f "README.md" ]; then
    echo "README.md が見つかりません。一時的な README.md を生成します..."
    echo "# affective_empathy_eval" > README.md
fi
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "エラー: python3 が見つかりません。Python 3.12 をインストールしてください。" >&2
    exit 1
fi

PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "検出された Python: $($PYTHON_CMD --version) (${PYTHON_CMD})"

# 2. 既存の .venv の扱い
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "既存の .venv が検出されました。"
    echo "再作成する場合は手動で 'rm -rf .venv' を実行してから本スクリプトを再実行してください。"
    echo "既存環境を更新・パッケージ同期します..."
fi

# 3. 仮想環境の作成 (uv 優先、フォールバック: venv)
if command -v uv &> /dev/null; then
    echo ""
    echo "==> [1/3] uv を使用して仮想環境を構築します..."
    if [ ! -d "$VENV_DIR" ]; then
        uv venv .venv --python "$PYTHON_CMD"
    fi
    echo "==> [2/3] パッケージをインストールします (uv pip install)..."
    uv pip install -e ".[dev]"
    # requirements.txt 経由の直接同期もサポート
    if [ -f "requirements.txt" ]; then
        uv pip install -r requirements.txt
    fi
else
    echo ""
    echo "==> [1/3] python3 -m venv を使用して仮想環境を構築します..."
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_CMD -m venv .venv
    fi
    echo "==> [2/3] pip をアップグレードし、パッケージをインストールします..."
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
    "${VENV_DIR}/bin/python" -m pip install -e ".[dev]"
    if [ -f "requirements.txt" ]; then
        "${VENV_DIR}/bin/python" -m pip install -r requirements.txt
    fi
fi

# 4. 動作検証
echo ""
echo "==> [3/3] 環境の整合性を検証しています..."
"${VENV_DIR}/bin/python" - << 'PY'
import sys
print(f"Python: {sys.version}")

packages = [
    "torch",
    "transformers",
    "accelerate",
    "scipy",
    "sklearn",
    "ot",  # POT (Python Optimal Transport)
    "pandas",
    "pytest",
    "affective_empathy_eval"
]

all_ok = True
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError as e:
        print(f"  [NG] {pkg}: {e}")
        all_ok = False

if not all_ok:
    sys.exit(1)
print("\nすべての主要パッケージおよび自作モジュールのインポートに成功しました！")
PY

echo ""
echo "================================================================="
echo "  仮想環境のセットアップが正常に完了しました！"
echo ""
echo "  【使用開始方法】"
echo "    source .venv/bin/activate"
echo ""
echo "  【テスト実行例】"
echo "    .venv/bin/pytest -q"
echo "================================================================="
