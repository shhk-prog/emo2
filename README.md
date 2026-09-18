# Affective Empathy Evaluation (LLM情動反応性評価実験)

本リポジトリは、EmoBank等の人間アノテーション済みVAD（Valence–Arousal–Dominance）データを用いて、大規模言語モデル（LLM）の情動反応性（Affective Reactivity）およびその内部表現・因果回路メカニズムを測定・分析する研究実験基盤です。

---

## 1. プロジェクト概要

本研究は、行動実験（Behavioral Stage）から内部メカニズム解明（V1, V2, V3 Stage）まで一貫した理論・測定基盤に基づき構成されています：

- **Behavioral Stage (`behavioral/`)**: 3-Way VAD と AIPsy 4-Split。4軸は Human-affect correspondence, Sensitivity, Dose-response / Specificity, Reader–Self coupling。詳細は [`behavioral/README.md`](behavioral/README.md)。
- **V1 Stage (`v1/`)**: Reader ↔ Self の decodability / 幾何（E1/E2）、意味統制（Phase B）、因果マップと交換可能性（E3/E4）、課題特異化（E6）。詳細は [`v1/README.md`](v1/README.md)。
- **V2 Stage (`v2/`)**: Base ↔ Instruct の幾何再編、ピーク解離、2D OT 分布回復（RQ1〜RQ4）。詳細は [`v2/README.md`](v2/README.md)。
- **V3 Stage (`v3/`)**: 状態誘導ゲート、時空間 4-Map、mediated attenuation、確証的再現。詳細は [`v3/README.md`](v3/README.md)。

### 対象モデル構成（コホート設計）
モデルサイズ差交絡を排除し、事後学習（Post-training）による幾何・因果再編を純粋に検証するため、狭い 1〜1.5B パラメータ帯の 4 大ファミリーを Primary コホートとしています：
- **Primary 1–1.5B Cohort (`primary_small`)**:
  - **Qwen 2.5 (1.5B)**: `Qwen/Qwen2.5-1.5B` $\leftrightarrow$ `Qwen/Qwen2.5-1.5B-Instruct`
  - **Llama 3.2 (1.23B)**: `meta-llama/Llama-3.2-1B` $\leftrightarrow$ `meta-llama/Llama-3.2-1B-Instruct`
  - **Gemma 3 (1B)**: `google/gemma-3-1b-pt` $\leftrightarrow$ `google/gemma-3-1b-it`
  - **OLMo 2 (1B)**: `allenai/OLMo-2-0425-1B` $\leftrightarrow$ `allenai/OLMo-2-0425-1B-Instruct`
- **Supplementary Scale Validation (`scale_validation`)**:
  - **Mistral (7B)**: `mistralai/Mistral-7B-v0.3` $\leftrightarrow$ `mistralai/Mistral-7B-Instruct-v0.3`（大規模モデルでの頑健性・再現性検証）

---

## 2. 環境構築とセットアップ

AGENTS.md の規定に従い、必ずプロジェクト専用の仮想環境（`.venv`）を構築して実行してください。

### 自動セットアップ（推奨: 1コマンド）
```bash
./setup_env.sh
```
- `uv` がインストールされている環境では数秒で自動構築されます。
- `uv` がない場合も標準の `python3 -m venv` を用いて自動セットアップされます。

### 手動セットアップ (`uv`)
```bash
uv venv .venv --python python3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 3. テストの実行

仮想環境を有効化（または `.venv/bin/pytest` を直接指定）してテストを実行します：

```bash
.venv/bin/pytest -q
```

---

## 4. 統合実験実行 CLI (Quick Start)

共通パッケージ `affective_empathy_eval` の統合ランナーから、全ステージを統一コマンドで実行できます：

```bash
# Primary 1-1.5B コホートで各ステージを実行
# --stage all も Behavioral → V1 → V2 → V3 の順。所要時間は未計測。
# 本番は stage 分割スクリプトの方が安全（途中失敗時の切り分けが容易）。
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small
python -m affective_empathy_eval.run --stage v1 --model-set primary_small
python -m affective_empathy_eval.run --stage v2 --model-set primary_small
python -m affective_empathy_eval.run --stage v3 --model-set primary_small
python -m affective_empathy_eval.run --stage all --model-set primary_small


# Supplementary 7B 外部スケール検証 (Mistral 7B)
python -m affective_empathy_eval.run --stage v2 --model-set scale_validation

# Dry-run による高速動作検証
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run
```

---

## 5. ディレクトリ構成

```text
├── README.md                      # 本ドキュメント
├── pyproject.toml                 # プロジェクト共通依存関係およびテスト設定定義（正本）
├── setup_env.sh                   # 仮想環境自動構築スクリプト
├── src/
│   └── affective_empathy_eval/    # 共通基盤ライブラリ (唯一の正本パッケージ)
│       ├── run.py                 # 統合実験実行 CLI
│       ├── models/                # 共通 ModelRegistry & ModelAdapters
│       └── ...
├── tests/                         # 共通テストスイート (All tests should pass)
├── configs/                       # 共通設定ファイル (models.yaml: 唯一のモデル定義正本)
├── behavioral/                    # 行動実験 Primary パイプライン & 分析
├── v1/                            # V1 表現幾何・Prompt-End 因果パッチング実験
├── v2/                            # V2 幾何・因果結合・分布回復実験 (Primary: v2/primary/)
├── v3/                            # V3 時空間ダイナミクス・媒介分析実験 (Primary: v3/primary/)
└── docs/                          # 実験記録・仕様書・決定ログ
```
