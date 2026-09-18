# Affective Empathy Evaluation（LLM 情動反応性評価実験）

人間評定済みの感情刺激（EmoBank VAD、AIPsy 4-split）に対して、大規模言語モデルの **操作的な認識出力** と **操作的な自己報告出力** を測り、その内部表現と因果利用を段階的に調べる実験基盤である。

本研究は、モデルが主観的な感情を経験すること、あるいは認知的共感 / 情動的共感を持つことを検証しない。論文構成の正本は現行 README 群であり、旧 `iclr2027/iclr2027_conference2.tex` は旧稿である。構成メモは [`docs/v3_prerun_five_fixes/paper_outline.md`](docs/v3_prerun_five_fixes/paper_outline.md)。

操作定義:

- **Reader**: 平均的読者の VA を推定する認識課題
- **Self**: 刺激提示後の自己報告 / 反応性課題

仮説図は一直線（刺激 → 認識 → 内部状態 → 自己報告）ではなく、共有表現からの分岐である。

```text
Stimulus
  → Shared affect representation
     → Reader readout
     → Self readout
```

---

## 1. 研究階段

```text
Behavioral  →  V1  →  V2  →  V3
```

| Stage | 問い | 比較軸 |
|---|---|---|
| Behavioral | Do Reader and Self covary? | 出力分布。認識と自己報告は独立セッション |
| V1 | What do Reader and Self share internally? | 同一モデル内の Reader ↔ Self |
| V2 | What does post-training reorganize? | 同一ファミリーの Base ↔ Instruct |
| V3 | When/where does information acquire causal leverage? | Instruct 側の層 × 生成段階 |

- **Behavioral** (`behavioral/`): EmoBank 3-Way と AIPsy 4-Split。4軸は correspondence, Sensitivity, Dose-response / Specificity, Reader–Self coupling。[`behavioral/README.md`](behavioral/README.md)
- **V1** (`v1/`): decodability / 幾何（E1/E2）、rule-based 文脈統制（Phase B）、因果マップと交換可能性（E3/E4）、課題特異化（E6）。Base / Instruct 8 条件は各モデル内の再現であり、差の解釈は V2。[`v1/README.md`](v1/README.md)
- **V2** (`v2/`): 幾何再編、ピーク解離、2D OT 分布回復（RQ1〜RQ4）。[`v2/README.md`](v2/README.md)
- **V3** (`v3/`): AIPsy matched-neutral 192 pair での状態誘導ゲート、時空間 4-Map、mediated attenuation、確証的再現。EmoBank 3-way は使わない。RQ1 が完全一致の `GO` のときだけ RQ2 以降へ進む。[`v3/README.md`](v3/README.md)

候補空間（数値を Stage 間で直接比較しない）:

- Behavioral / V1 Primary: $9^3=729$ VAD（Dominance は補助次元）
- V2 / V3 Primary: $9^2=81$ VA
- 729 空間の $E[V],E[A]$ と 81 空間のそれを同一尺度として混ぜない。

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
# 本番は stage 分割を推奨する（V1/V2/V3 が重いため、障害切り分けと resume が容易）。
# まず Behavioral → V1 を完走し、問題なければ V2 → V3。
bash scripts/run_production_behavioral.sh cuda:0
bash scripts/run_production_v1.sh cuda:0
bash scripts/run_production_v2.sh cuda:0
bash scripts/run_production_v3.sh cuda:0

# 統合 CLI。--stage all も Behavioral → V1 → V2 → V3 の順。所要時間は未計測。
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small
python -m affective_empathy_eval.run --stage v1 --model-set primary_small
python -m affective_empathy_eval.run --stage v2 --model-set primary_small
python -m affective_empathy_eval.run --stage v3 --model-set primary_small
python -m affective_empathy_eval.run --stage all --model-set primary_small


# Supplementary 7B 外部スケール検証 (Mistral 7B)
python -m affective_empathy_eval.run --stage v2 --model-set scale_validation

# Dry-run による高速動作検証（モデル重み不要。transformers 未導入でも Primary は起動する）
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run

# V3 だけ、RQ1 が GO でない場合に明示的に継続するとき
# python -m affective_empathy_eval.run --stage v3 --model-set primary_small --force-after-no-go
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
