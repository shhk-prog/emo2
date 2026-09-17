# Affective Empathy Evaluation (LLM情動反応性評価実験)

本リポジトリは、EmoBank等の人間アノテーション済みVAD（Valence–Arousal–Dominance）データを用いて、大規模言語モデル（LLM）の情動反応性（Affective Reactivity）およびその内部表現・因果回路メカニズムを測定・分析する研究実験基盤です。

---

## 1. プロジェクト概要

本研究は、行動実験（Behavioral Stage）から内部メカニズム解明（V1, V2, V3 Stage）まで一貫した理論・測定基盤に基づき構成されています：

- **Behavioral Stage (`v1/`)**: 3-Way VAD 評価、AIPsy 評価（4大 Primary 評価軸：Human grounding, Affective sensitivity, Dose-response, Reader–Self coupling）
- **V1 Stage (`v1/`)**: 表現空間の幾何構造（E1/E2）、Response-Onset Causal Map（E3）、因果的交換可能性（E4）、Task-Specific Causal Specialization（E6）
- **V2 Stage (`v2/`)**: 4モデルファミリー横断（Base ↔ Instruct）幾何・因果結合マッピング、Matched-plain 統制、EMD 分布回復パッチング（RQ1〜RQ4）
- **V3 Stage (`v3/`)**: 時空間ダイナミクス、実モデル情動状態誘導、共変量統制偏回帰 $\beta(l,t)$、媒介分析（Path Mediation）、確証的再現性（Confirmatory Replication）

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

## 4. ディレクトリ構成

```text
├── README.md                      # 本ドキュメント
├── pyproject.toml                 # プロジェクト共通依存関係定義
├── setup_env.sh                   # 仮想環境自動構築スクリプト
├── pytest.ini                     # テスト共通設定
├── src/
│   └── affective_empathy_eval/    # 共通基盤ライブラリ (prompts, likelihood, metrics 等)
├── tests/                         # 共通テストスイート
├── v1/                            # Behavioral & V1 Phase A/B/C 実験
├── v2/                            # V2 幾何・因果結合・分布回復実験
│   ├── scripts/                   # Primary パイプライン (RQ1〜RQ4)
│   └── scripts/legacy/            # 旧スクリプト隔離アーカイブ
├── v3/                            # V3 時空間ダイナミクス・媒介分析実験
└── docs/                          # 実験記録・仕様書・決定ログ
```
